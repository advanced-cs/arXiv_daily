# 计算机视觉 cs.CV

- **最新发布 170 篇**

- **更新 94 篇**

## 最新发布

#### [new 001] Gesture-Aware Pretraining and Token Fusion for 3D Hand Pose Estimation
- **分类: cs.CV**

- **简介: 该论文属于3D手部姿态估计任务，旨在利用手势语义提升估计精度。通过两阶段框架，结合手势标签进行预训练和参数回归，提升单手识别效果。**

- **链接: [https://arxiv.org/pdf/2603.17396](https://arxiv.org/pdf/2603.17396)**

> **作者:** Rui Hong; Jana Kosecka
>
> **备注:** 6 pages, 6 figures
>
> **摘要:** Estimating 3D hand pose from monocular RGB images is fundamental for applications in AR/VR, human-computer interaction, and sign language understanding. In this work we focus on a scenario where a discrete set of gesture labels is available and show that gesture semantics can serve as a powerful inductive bias for 3D pose estimation. We present a two-stage framework: gesture-aware pretraining that learns an informative embedding space using coarse and fine gesture labels from InterHand2.6M, followed by a per-joint token Transformer guided by gesture embeddings as intermediate representations for final regression of MANO hand parameters. Training is driven by a layered objective over parameters, joints, and structural constraints. Experiments on InterHand2.6M demonstrate that gesture-aware pretraining consistently improves single-hand accuracy over the state-of-the-art EANet baseline, and that the benefit transfers across architectures without any modification.
>
---
#### [new 002] MM-OVSeg:Multimodal Optical-SAR Fusion for Open-Vocabulary Segmentation in Remote Sensing
- **分类: cs.CV**

- **简介: 该论文属于遥感中的开放词汇分割任务，旨在解决天气恶劣条件下分割性能下降的问题。通过融合光学与SAR数据，提升模型的鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.17528](https://arxiv.org/pdf/2603.17528)**

> **作者:** Yimin Wei; Aoran Xiao; Hongruixuan Chen; Junshi Xia; Naoto Yokoya
>
> **摘要:** Open-vocabulary segmentation enables pixel-level recognition from an open set of textual categories, allowing generalization beyond fixed classes. Despite great potential in remote sensing, progress in this area remains largely limited to clear-sky optical data and struggles under cloudy or haze-contaminated conditions. We present MM-OVSeg, a multimodal Optical-SAR fusion framework for resilient open-vocabulary segmentation under adverse weather conditions. MM-OVSeg leverages the complementary strengths of the two modalities--optical imagery provides rich spectral semantics, while synthetic aperture radar (SAR) offers cloud-penetrating structural cues. To address the cross-modal domain gap and the limited dense prediction capability of current vision-language models, we propose two key designs: a cross-modal unification process for multi-sensor representation alignment, and a dual-encoder fusion module that integrates hierarchical features from multiple vision foundation models for text-aligned multimodal segmentation. Extensive experiments demonstrate that MM-OVSeg achieves superior robustness and generalization across diverse cloud conditions. The source dataset and code are available here.
>
---
#### [new 003] DiffVP: Differential Visual Semantic Prompting for LLM-Based CT Report Generation
- **分类: cs.CV**

- **简介: 该论文属于医学影像报告生成任务，旨在解决LLM在CT报告生成中难以区分有用信息与冗余背景的问题。提出DiffVP方法，通过视觉语义提示增强报告准确性。**

- **链接: [https://arxiv.org/pdf/2603.17718](https://arxiv.org/pdf/2603.17718)**

> **作者:** Yuhe Tian; Kun Zhang; Haoran Ma; Rui Yan; Yingtai Li; Rongsheng Wang; Shaohua Kevin Zhou
>
> **摘要:** While large language models (LLMs) have advanced CT report generation, existing methods typically encode 3D volumes holistically, failing to distinguish informative cues from redundant anatomical background. Inspired by radiological cognitive subtraction, we propose Differential Visual Prompting (DiffVP), which conditions report generation on explicit, high-level semantic scan-to-reference differences rather than solely on absolute visual features. DiffVP employs a hierarchical difference extractor to capture complementary global and local semantic discrepancies into a shared latent space, along with a difference-to-prompt generator that transforms these signals into learnable visual prefix tokens for LLM conditioning. These difference prompts serve as structured conditioning signals that implicitly suppress invariant anatomy while amplifying diagnostically relevant visual evidence, thereby facilitating accurate report generation without explicit lesion localization. On two large-scale benchmarks, DiffVP consistently outperforms prior methods, improving the average BLEU-1-4 by +10.98 and +4.36, respectively, and further boosts clinical efficacy on RadGenome-ChestCT (F1 score 0.421). All codes will be released at this https URL.
>
---
#### [new 004] VirPro: Visual-referred Probabilistic Prompt Learning for Weakly-Supervised Monocular 3D Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于弱监督单目3D目标检测任务，旨在解决文本描述与视觉多样性不匹配的问题。提出VirPro方法，通过多模态预训练提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.17470](https://arxiv.org/pdf/2603.17470)**

> **作者:** Chupeng Liu; Jiyong Rao; Shangquan Sun; Runkai Zhao; Weidong Cai
>
> **备注:** Accepted by CVPR 2026 Findings
>
> **摘要:** Monocular 3D object detection typically relies on pseudo-labeling techniques to reduce dependency on real-world annotations. Recent advances demonstrate that deterministic linguistic cues can serve as effective auxiliary weak supervision signals, providing complementary semantic context. However, hand-crafted textual descriptions struggle to capture the inherent visual diversity of individuals across scenes, limiting the model's ability to learn scene-aware representations. To address this challenge, we propose Visual-referred Probabilistic Prompt Learning (VirPro), an adaptive multi-modal pretraining paradigm that can be seamlessly integrated into diverse weakly supervised monocular 3D detection frameworks. Specifically, we generate a diverse set of learnable, instance-conditioned prompts across scenes and store them in an Adaptive Prompt Bank (APB). Subsequently, we introduce Multi-Gaussian Prompt Modeling (MGPM), which incorporates scene-based visual features into the corresponding textual embeddings, allowing the text prompts to express visual uncertainties. Then, from the fused vision-language embeddings, we decode a prompt-targeted Gaussian, from which we derive a unified object-level prompt embedding for each instance. RoI-level contrastive matching is employed to enforce modality alignment, bringing embeddings of co-occurring objects within the same scene closer in the latent space, thus enhancing semantic coherence. Extensive experiments on the KITTI benchmark demonstrate that integrating our pretraining paradigm consistently yields substantial performance gains, achieving up to a 4.8% average precision improvement than the baseline.
>
---
#### [new 005] EmergeNav: Structured Embodied Inference for Zero-Shot Vision-and-Language Navigation in Continuous Environments
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言导航任务，解决零样本连续环境导航问题。提出EmergeNav框架，通过结构化推理实现稳定导航，无需特定训练或地图。**

- **链接: [https://arxiv.org/pdf/2603.16947](https://arxiv.org/pdf/2603.16947)**

> **作者:** Kun Luo; Xiaoguang Ma
>
> **摘要:** Zero-shot vision-and-language navigation in continuous environments (VLN-CE) remains challenging for modern vision-language models (VLMs). Although these models encode useful semantic priors, their open-ended reasoning does not directly translate into stable long-horizon embodied execution. We argue that the key bottleneck is not missing knowledge alone, but missing an execution structure for organizing instruction following, perceptual grounding, temporal progress, and stage verification. We propose EmergeNav, a zero-shot framework that formulates continuous VLN as structured embodied inference. EmergeNav combines a Plan--Solve--Transition hierarchy for stage-structured execution, GIPE for goal-conditioned perceptual extraction, contrastive dual-memory reasoning for progress grounding, and role-separated Dual-FOV sensing for time-aligned local control and boundary verification. On VLN-CE, EmergeNav achieves strong zero-shot performance using only open-source VLM backbones and no task-specific training, explicit maps, graph search, or waypoint predictors, reaching 30.00 SR with Qwen3-VL-8B and 37.00 SR with Qwen3-VL-32B. These results suggest that explicit execution structure is a key ingredient for turning VLM priors into stable embodied navigation behavior.
>
---
#### [new 006] LoGSAM: Parameter-Efficient Cross-Modal Grounding for MRI Segmentation
- **分类: cs.CV**

- **简介: 该论文提出LoGSAM，用于MRI脑肿瘤分割，解决标注数据不足问题。通过语音转文本，结合预训练模型实现高效定位与分割。**

- **链接: [https://arxiv.org/pdf/2603.17576](https://arxiv.org/pdf/2603.17576)**

> **作者:** Mohammad Robaitul Islam Bhuiyan; Sheethal Bhat; Melika Qahqaie; Tri-Thien Nguyen; Paula Andrea Pérez Toro; Tomas Arias Vergara; Andreas Maier
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Precise localization and delineation of brain tumors using Magnetic Resonance Imaging (MRI) are essential for planning therapy and guiding surgical decisions. However, most existing approaches rely on task-specific supervised models and are constrained by the limited availability of annotated data. To address this, we propose LoGSAM, a parameter-efficient, detection-driven framework that transforms radiologist dictation into text prompts for foundation-model-based localization and segmentation. Radiologist speech is first transcribed and translated using a pretrained Whisper ASR model, followed by negation-aware clinical NLP to extract tumor-specific textual prompts. These prompts guide text-conditioned tumor localization via a LoRA-adapted vision-language detection model, Grounding DINO (GDINO). The LoRA adaptation updates using 5% of the model parameters, thereby enabling computationally efficient domain adaptation while preserving pretrained cross-modal knowledge. The predicted bounding boxes are used as prompts for MedSAM to generate pixel-level tumor masks without any additional fine-tuning. Conditioning the frozen MedSAM on LoGSAM-derived priors yields a state-of-the-art dice score of 80.32% on BRISC 2025. In addition, we evaluate the full pipeline using German dictations from a board-certified radiologist on 12 unseen MRI scans, achieving 91.7% case-level accuracy. These results highlight the feasibility of constructing a modular, speech-to-segmentation pipeline by intelligently leveraging pretrained foundation models with minimal parameter updates.
>
---
#### [new 007] PCA-Seg: Revisiting Cost Aggregation for Open-Vocabulary Semantic and Part Segmentation
- **分类: cs.CV**

- **简介: 该论文属于开放词汇语义与部件分割任务，解决类别语义与空间上下文干扰问题，提出PCA-Seg框架，通过并行聚合和特征正交化提升分割性能。**

- **链接: [https://arxiv.org/pdf/2603.17520](https://arxiv.org/pdf/2603.17520)**

> **作者:** Jianjian Yin; Tao Chen; Yi Chen; Gensheng Pei; Xiangbo Shu; Yazhou Yao; Fumin Shen
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Recent advances in vision-language models (VLMs) have garnered substantial attention in open-vocabulary semantic and part segmentation (OSPS). However, existing methods extract image-text alignment cues from cost volumes through a serial structure of spatial and class aggregations, leading to knowledge interference between class-level semantics and spatial context. Therefore, this paper proposes a simple yet effective parallel cost aggregation (PCA-Seg) paradigm to alleviate the above challenge, enabling the model to capture richer vision-language alignment information from cost volumes. Specifically, we design an expert-driven perceptual learning (EPL) module that efficiently integrates semantic and contextual streams. It incorporates a multi-expert parser to extract complementary features from multiple perspectives. In addition, a coefficient mapper is designed to adaptively learn pixel-specific weights for each feature, enabling the integration of complementary knowledge into a unified and robust feature embedding. Furthermore, we propose a feature orthogonalization decoupling (FOD) strategy to mitigate redundancy between the semantic and contextual streams, which allows the EPL module to learn diverse knowledge from orthogonalized features. Extensive experiments on eight benchmarks show that each parallel block in PCA-Seg adds merely 0.35M parameters while achieving state-of-the-art OSPS performance.
>
---
#### [new 008] OpenQlaw: An Agentic AI Assistant for Analysis of 2D Quantum Materials
- **分类: cs.CV**

- **简介: 该论文提出OpenQlaw系统，用于分析2D量子材料。任务是将材料识别与器件制造结合，解决传统方法认知过载问题，通过代理架构实现高效推理与交互。**

- **链接: [https://arxiv.org/pdf/2603.17043](https://arxiv.org/pdf/2603.17043)**

> **作者:** Sankalp Pandey; Xuan-Bac Nguyen; Hoang-Quan Nguyen; Tim Faltermeier; Nicholas Borys; Hugh Churchill; Khoa Luu
>
> **摘要:** The transition from optical identification of 2D quantum materials to practical device fabrication requires dynamic reasoning beyond the detection accuracy. While recent domain-specific Multimodal Large Language Models (MLLMs) successfully ground visual features using physics-informed reasoning, their outputs are optimized for step-by-step cognitive transparency. This yields verbose candidate enumerations followed by dense reasoning that, while accurate, may induce cognitive overload and lack immediate utility for real-world interaction with researchers. To address this challenge, we introduce OpenQlaw, an agentic orchestration system for analyzing 2D materials. The architecture is built upon NanoBot, a lightweight agentic framework inspired by OpenClaw, and QuPAINT, one of the first Physics-Aware Instruction Multi-modal platforms for Quantum Material Discovery. This allows accessibility to the lab floor via a variety of messaging channels. OpenQlaw allows the core Large Language Model (LLM) agent to orchestrate a domain-expert MLLM,with QuPAINT, as a specialized node, successfully decoupling visual identification from reasoning and deterministic image rendering. By parsing spatial data from the expert, the agent can dynamically process user queries, such as performing scale-aware physical computation or generating isolated visual annotations, and answer in a naturalistic manner. Crucially, the system features a persistent memory that enables the agent to save physical scale ratios (e.g., 1 pixel = 0.25 {\mu}m) for area computations and store sample preparation methods for efficacy comparison. The application of an agentic architecture, together with the extension that uses the core agent as an orchestrator for domain-specific experts, transforms isolated inferences into a context-aware assistant capable of accelerating high-throughput device fabrication.
>
---
#### [new 009] GMT: Goal-Conditioned Multimodal Transformer for 6-DOF Object Trajectory Synthesis in 3D Scenes
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GMT模型，解决3D场景中6-DOF物体轨迹生成任务，通过融合几何、语义等信息实现精准轨迹控制。**

- **链接: [https://arxiv.org/pdf/2603.17993](https://arxiv.org/pdf/2603.17993)**

> **作者:** Huajian Zeng; Abhishek Saroha; Daniel Cremers; Xi Wang
>
> **备注:** Accpeted by 3DV 2026. Project Page: this https URL
>
> **摘要:** Synthesizing controllable 6-DOF object manipulation trajectories in 3D environments is essential for enabling robots to interact with complex scenes, yet remains challenging due to the need for accurate spatial reasoning, physical feasibility, and multimodal scene understanding. Existing approaches often rely on 2D or partial 3D representations, limiting their ability to capture full scene geometry and constraining trajectory precision. We present GMT, a multimodal transformer framework that generates realistic and goal-directed object trajectories by jointly leveraging 3D bounding box geometry, point cloud context, semantic object categories, and target end poses. The model represents trajectories as continuous 6-DOF pose sequences and employs a tailored conditioning strategy that fuses geometric, semantic, contextual, and goaloriented information. Extensive experiments on synthetic and real-world benchmarks demonstrate that GMT outperforms state-of-the-art human motion and human-object interaction baselines, such as CHOIS and GIMO, achieving substantial gains in spatial accuracy and orientation control. Our method establishes a new benchmark for learningbased manipulation planning and shows strong generalization to diverse objects and cluttered 3D environments. Project page: https://huajian- this http URL. io/projects/gmt/.
>
---
#### [new 010] Revisiting Cross-Attention Mechanisms: Leveraging Beneficial Noise for Domain-Adaptive Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于无监督域适应任务，旨在解决域间差异导致的性能下降问题。通过引入有益噪声和跨尺度对齐机制，提升模型的域适应能力。**

- **链接: [https://arxiv.org/pdf/2603.17474](https://arxiv.org/pdf/2603.17474)**

> **作者:** Zelin Zang; Yehui Yang; Fei Wang; Liangyu Li; Baigui Sun
>
> **摘要:** Unsupervised Domain Adaptation (UDA) seeks to transfer knowledge from a labeled source domain to an unlabeled target domain but often suffers from severe domain and scale gaps that degrade performance. Existing cross-attention-based transformers can align features across domains, yet they struggle to preserve content semantics under large appearance and scale variations. To explicitly address these challenges, we introduce the concept of beneficial noise, which regularizes cross-attention by injecting controlled perturbations, encouraging the model to ignore style distractions and focus on content. We propose the Domain-Adaptive Cross-Scale Matching (DACSM) framework, which consists of a Domain-Adaptive Transformer (DAT) for disentangling domain-shared content from domain-specific style, and a Cross-Scale Matching (CSM) module that adaptively aligns features across multiple resolutions. DAT incorporates beneficial noise into cross-attention, enabling progressive domain translation with enhanced robustness, yielding content-consistent and style-invariant representations. Meanwhile, CSM ensures semantic consistency under scale changes. Extensive experiments on VisDA-2017, Office-Home, and DomainNet demonstrate that DACSM achieves state-of-the-art performance, with up to +2.3% improvement over CDTrans on VisDA-2017. Notably, DACSM achieves a +5.9% gain on the challenging "truck" class of VisDA, evidencing the strength of beneficial noise in handling scale discrepancies. These results highlight the effectiveness of combining domain translation, beneficial-noise-enhanced attention, and scale-aware alignment for robust cross-domain representation learning.
>
---
#### [new 011] Motion-Adaptive Temporal Attention for Lightweight Video Generation with Stable Diffusion
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决视频生成中的时间一致性与细节保持问题。通过引入动态时间注意力机制，在冻结的Stable Diffusion模型中实现轻量级视频生成。**

- **链接: [https://arxiv.org/pdf/2603.17398](https://arxiv.org/pdf/2603.17398)**

> **作者:** Rui Hong; Shuxue Quan
>
> **备注:** 6 pages, 3 figures, 4 tables. Published at IS&T Electronic Imaging 2026, GENAI Track
>
> **摘要:** We present a motion-adaptive temporal attention mechanism for parameter-efficient video generation built upon frozen Stable Diffusion models. Rather than treating all video content uniformly, our method dynamically adjusts temporal attention receptive fields based on estimated motion content: high-motion sequences attend locally across frames to preserve rapidly changing details, while low-motion sequences attend globally to enforce scene consistency. We inject lightweight temporal attention modules into all UNet transformer blocks via a cascaded strategy -- global attention in down-sampling and middle blocks for semantic stabilization, motion-adaptive attention in up-sampling blocks for fine-grained refinement. Combined with temporally correlated noise initialization and motion-aware gating, the system adds only 25.8M trainable parameters (2.9\% of the base UNet) while achieving competitive results on WebVid validation when trained on 100K videos. We demonstrate that the standard denoising objective alone provides sufficient implicit temporal regularization, outperforming approaches that add explicit temporal consistency losses. Our ablation studies reveal a clear trade-off between noise correlation and motion amplitude, providing a practical inference-time control for diverse generation behaviors.
>
---
#### [new 012] Noise-Aware Misclassification Attack Detection in Collaborative DNN Inference
- **分类: cs.CV**

- **简介: 该论文属于异常检测任务，旨在解决协同DNN推理中的恶意数据注入导致的隐蔽误分类问题。通过引入噪声感知的VAE框架提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.17914](https://arxiv.org/pdf/2603.17914)**

> **作者:** Shima Yousefi; Saptarshi Debroy
>
> **备注:** This work has been accepted for publication in IEEE/ACM CCGrid 2026
>
> **摘要:** Collaborative inference of object classification Deep neural Networks (DNNs) where resource-constrained end-devices offload partially processed data to remote edge servers to complete end-to-end processing, is becoming a key enabler of edge-AI. However, such edge-offloading is vulnerable to malicious data injections leading to stealthy misclassifications that are tricky to detect, especially in the presence of environmental noise. In this paper, we propose a semi-gray-box and noise- aware anomaly detection framework fueled by a variational autoencoder (VAE) to capture deviations caused by adversarial manipulation. The proposed framework incorporates a robust noise-aware feature that captures the characteristic behavior of environmental noise to improve detection accuracy while reducing false alarm rates. Our evaluation with popular object classification DNNs demonstrate the robustness of the proposed detection (up to 90% AUROC across DNN configurations) under realistic noisy conditions while revealing limitations caused by feature similarity and elevated noise levels.
>
---
#### [new 013] ConfusionBench: An Expert-Validated Benchmark for Confusion Recognition and Localization in Educational Videos
- **分类: cs.CV**

- **简介: 该论文属于教育AI中的学生困惑识别与定位任务，旨在解决现有数据集标签噪声大、标注粗略的问题。通过多阶段过滤流程构建高质量基准ConfusionBench，并提供基线评估与可视化工具。**

- **链接: [https://arxiv.org/pdf/2603.17267](https://arxiv.org/pdf/2603.17267)**

> **作者:** Lu Dong; Xiao Wang; Mark Frank; Srirangaraj Setlur; Venu Govindaraju; Ifeoma Nwogu
>
> **摘要:** Recognizing and localizing student confusion from video is an important yet challenging problem in educational AI. Existing confusion datasets suffer from noisy labels, coarse temporal annotations, and limited expert validation, which hinder reliable fine-grained recognition and temporally grounded analysis. To address these limitations, we propose a practical multi-stage filtering pipeline that integrates two stages of model-assisted screening, researcher curation, and expert validation to build a higher-quality benchmark for confusion understanding. Based on this pipeline, we introduce ConfusionBench, a new benchmark for educational videos consisting of a balanced confusion recognition dataset and a video localization dataset. We further provide zero-shot baseline evaluations of a representative open-source model and a proprietary model on clip-level confusion recognition, long-video confusion localization tasks. Experimental results show that the proprietary model performs better overall but tends to over-predict transitional segments, while the open-source model is more conservative and more prone to missed detections. In addition, the proposed student confusion report visualization can support educational experts in making intervention decisions and adapting learning plans accordingly. All datasets and related materials will be made publicly available on our project page.
>
---
#### [new 014] Hidden Clones: Exposing and Fixing Family Bias in Vision-Language Model Ensembles
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型集成任务，解决家族相关错误导致的集成失效问题。通过提出三种方法提升集成效果，显著提高准确率。**

- **链接: [https://arxiv.org/pdf/2603.17111](https://arxiv.org/pdf/2603.17111)**

> **作者:** Zacharie Bugaud
>
> **备注:** 15 pages, 6 figures, 11 tables
>
> **摘要:** Ensembling Vision-Language Models (VLMs) from different providers maximizes benchmark accuracy, yet models from the same architectural family share correlated errors that standard voting ignores. We study this structure across 17 VLMs from 8 families on VQAv2, TextVQA, and GQA. Family-correlated errors reduce effective ensemble dimensionality to 2.5-3.6 independent voters and create a Misleading tier (1.5-6.5% of questions) where correlated majority errors destroy accuracy to 0% despite the best model being correct. We propose three family-aware methods. Hierarchical Family Voting (HFV) aggregates within families before voting across them, recovering +18-26 pp on the Misleading tier. QualRCCV, a training-free method weighting models by calibration, family quality, and inverse family size, is the first to beat calibrated voting on all three benchmarks (p<0.05). Learned Candidate Scoring (LCS) trains a cross-validated classifier to re-rank candidate answers using support breadth, family diversity, and model quality, achieving the largest gains: +0.68% VQAv2, +0.61% TextVQA, +2.45% GQA -- all significant -- and is the only learned method that never degrades any benchmark. On VQAv2 test-standard (EvalAI), LCS reaches 87.83% with 12 models, confirming generalization.
>
---
#### [new 015] HopChain: Multi-Hop Data Synthesis for Generalizable Vision-Language Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言推理任务，旨在解决VLMs在细粒度推理中的不足。提出HopChain框架，生成多跳推理数据以提升模型的泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.17024](https://arxiv.org/pdf/2603.17024)**

> **作者:** Shenzhi Wang; Shixuan Liu; Jing Zhou; Chang Gao; Xiong-Hui Chen; Binghai Wang; An Yang; Shiji Song; Bowen Yu; Gao Huang; Junyang Lin
>
> **备注:** 28 pages, 8 figures, 2 tables
>
> **摘要:** VLMs show strong multimodal capabilities, but they still struggle with fine-grained vision-language reasoning. We find that long CoT reasoning exposes diverse failure modes, including perception, reasoning, knowledge, and hallucination errors, which can compound across intermediate steps. However, most existing vision-language data used for RLVR does not involve complex reasoning chains that rely on visual evidence throughout, leaving these weaknesses largely unexposed. We therefore propose HopChain, a scalable framework for synthesizing multi-hop vision-language reasoning data specifically for RLVR training of VLMs. Each synthesized multi-hop query forms a logically dependent chain of instance-grounded hops, where earlier hops establish the instances, sets, or conditions needed for later hops, while the final answer remains a specific, unambiguous number suitable for verifiable rewards. We add the multi-hop data synthesized by HopChain to the original RLVR data used to train Qwen3.5-35B-A3B and Qwen3.5-397B-A17B, and compare against RLVR on the original RLVR data alone across 24 benchmarks spanning STEM and Puzzle, General VQA, Text Recognition and Document Understanding, and Video Understanding. Although this multi-hop data is not synthesized to target any specific benchmark, adding it improves 20 out of 24 benchmarks on both models, indicating broad and generalizable gains. To demonstrate that full chained queries are important, we replace them with half-multi-hop or single-hop variants, reducing the 24-benchmark average accuracy by 5.3 and 7.0 points, respectively. Multi-hop training also strengthens long-CoT vision-language reasoning, with gains peaking at more than 50 accuracy points in the ultra-long-CoT regime. These experiments establish HopChain as an effective, scalable framework for synthesizing multi-hop data that improves generalizable vision-language reasoning.
>
---
#### [new 016] Joint Optimization of Storage and Loading for High-Performance 3D Point Cloud Data Processing
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对3D点云数据处理中的存储与加载效率问题，提出PcRecord格式和高效处理流水线，提升大规模点云数据的处理速度与资源利用率。**

- **链接: [https://arxiv.org/pdf/2603.16945](https://arxiv.org/pdf/2603.16945)**

> **作者:** Ke Wang; Yanfei Cao; Xiangzhi Tao; Naijie Gu; Jun Yu; Zhengdong Wang; Shouyang Dong; Fan Yu; Cong Wang; Yang Luo
>
> **摘要:** With the rapid development of computer vision and deep learning, significant advancements have been made in 3D vision, partic- ularly in autonomous driving, robotic perception, and augmented reality. 3D point cloud data, as a crucial representation of 3D information, has gained widespread attention. However, the vast scale and complexity of point cloud data present significant chal- lenges for loading and processing and traditional algorithms struggle to handle large-scale this http URL diversity of storage formats for point cloud datasets (e.g., PLY, XYZ, BIN) adds complexity to data handling and results in inefficiencies in data preparation. Al- though binary formats like BIN and NPY have been used to speed up data access, they still do not fully address the time-consuming data loading and processing phase. To overcome these challenges, we propose the .PcRecord format, a unified data storage solution designed to reduce the storage occupation and accelerate the processing of point cloud data. We also introduce a high-performance data processing pipeline equipped with multiple modules. By leveraging a multi-stage parallel pipeline architecture, our system optimizes the use of computational resources, significantly improving processing speed and efficiency. This paper details the im- plementation of this system and demonstrates its effectiveness in addressing the challenges of handling large-scale point cloud this http URL average, our system achieves performance improvements of 6.61x (ModelNet40), 2.69x (S3DIS), 2.23x (ShapeNet), 3.09x (Kitti), 8.07x (SUN RGB-D), and 5.67x (ScanNet) with GPU and 6.9x, 1.88x, 1.29x, 2.28x, 25.4x, and 19.3x with Ascend.
>
---
#### [new 017] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于跨域小样本学习任务，针对CLIP在目标域局部对齐不足的问题，提出CC-CDFSL方法与语义锚机制，提升局部对齐与模型可解释性。**

- **链接: [https://arxiv.org/pdf/2603.17655](https://arxiv.org/pdf/2603.17655)**

> **作者:** Yaze Zhao; Yixiong Zou; Yuhua Li; Ruixuan Li
>
> **备注:** CVPR 2026
>
> **摘要:** Cross-Domain Few-Shot Learning (CDFSL) adapts models trained with large-scale general data (source domain) to downstream target domains with only scarce training data, where the research on vision-language models (e.g., CLIP) is still in the early stages. Typical downstream domains, such as medical diagnosis, require fine-grained visual cues for interpretable recognition, but we find that current fine-tuned CLIP models can hardly focus on these cues, albeit they can roughly focus on important regions in source domains. Although current works have demonstrated CLIP's shortcomings in capturing local subtle patterns, in this paper, we find that the domain gap and scarce training data further exacerbate such shortcomings, much more than that of holistic patterns, which we call the local misalignment problem in CLIP-based CDFSL. To address this problem, due to the lack of supervision in aligning local visual features and text semantics, we turn to self-supervision information. Inspired by the translation task, we propose the CC-CDFSL method with cycle consistency, which translates local visual features into text features and then translates them back into visual features (and vice versa), and constrains the original features close to the translated back features. To reduce the noise imported by richer information in the visual modality, we further propose a Semantic Anchor mechanism, which first augments visual features to provide a larger corpus for the text-to-image mapping, and then shrinks the image features to filter out irrelevant image-to-text mapping. Extensive experiments on various benchmarks, backbones, and fine-tuning methods show we can (1) effectively improve the local vision-language alignment, (2) enhance the interpretability of learned patterns and model decisions by visualizing patches, and (3) achieve state-of-the-art performance.
>
---
#### [new 018] Pixel-level Counterfactual Contrastive Learning for Medical Image Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分割任务，旨在解决标注数据稀缺和AI标签偏差问题。通过结合反事实生成与对比学习，提出DVD-CL和MVD-CL方法，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2603.17110](https://arxiv.org/pdf/2603.17110)**

> **作者:** Marceau Lafargue-Hauret; Raghav Mehta; Fabio De Sousa Ribeiro; Mélanie Roschewitz; Ben Glocker
>
> **备注:** Accepted at ISBI-2026 (oral presentation)
>
> **摘要:** Image segmentation relies on large annotated datasets, which are expensive and slow to produce. Silver-standard (AI-generated) labels are easier to obtain, but they risk introducing bias. Self-supervised learning, needing only images, has become key for pre-training. Recent work combining contrastive learning with counterfactual generation improves representation learning for classification but does not readily extend to pixel-level tasks. We propose a pipeline combining counterfactual generation with dense contrastive learning via Dual-View (DVD-CL) and Multi-View (MVD-CL) methods, along with supervised variants that utilize available silver-standard annotations. A new visualisation algorithm, the Color-coded High Resolution Overlay map (CHRO-map) is also introduced. Experiments show annotation-free DVD-CL outperforms other dense contrastive learning methods, while supervised variants using silver-standard labels outperform training on the silver-standard labeled data directly, achieving $\sim$94% DSC on challenging data. These results highlight that pixel-level contrastive learning, enhanced by counterfactuals and silver-standard annotations, improves robustness to acquisition and pathological variations.
>
---
#### [new 019] Temporal Gains, Spatial Costs: Revisiting Video Fine-Tuning in Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文研究多模态大语言模型中视频微调对视觉能力的影响，探讨视频与静态图像理解间的权衡问题。**

- **链接: [https://arxiv.org/pdf/2603.17541](https://arxiv.org/pdf/2603.17541)**

> **作者:** Linghao Zhang; Jungang Li; Yonghua Hei; Sicheng Tao; Song Dai; Yibo Yan; Zihao Dongfang; Weiting Liu; Chenxi Qin; Hanqian Li; Xin Zou; Jiahao Zhang; Shuhang Xun; Haiyun Jiang; Xuming Hu
>
> **摘要:** Multimodal large language models (MLLMs) are typically trained in multiple stages, with video-based supervised fine-tuning (Video-SFT) serving as a key step for improving visual understanding. Yet its effect on the fine-grained evolution of visual capabilities, particularly the balance between spatial and temporal understanding, remains poorly understood. In this paper, we systematically study how Video-SFT reshapes visual capabilities in MLLMs. Across architectures, parameter scales, and frame sampling settings, we observe a consistent pattern: Video-SFT reliably improves video performance, but often yields limited gains or even degradation on static image benchmarks. We further show that this trade-off is closely tied to temporal budget: increasing the number of sampled frames generally improves video performance, but does not reliably improve static image performance. Motivated by this finding, we study an instruction-aware Hybrid-Frame strategy that adaptively allocates frame counts and partially mitigates the image-video trade-off. Our results indicate that Video-SFT is not a free lunch for MLLMs, and preserving spatial understanding remains a central challenge in joint image-video training.
>
---
#### [new 020] Versatile Editing of Video Content, Actions, and Dynamics without Training
- **分类: cs.CV**

- **简介: 该论文提出DynaEdit，解决无需训练的视频编辑问题，支持动作、动态事件和对象交互的灵活修改。**

- **链接: [https://arxiv.org/pdf/2603.17989](https://arxiv.org/pdf/2603.17989)**

> **作者:** Vladimir Kulikov; Roni Paiss; Andrey Voynov; Inbar Mosseri; Tali Dekel; Tomer Michaeli
>
> **备注:** Project page at this https URL
>
> **摘要:** Controlled video generation has seen drastic improvements in recent years. However, editing actions and dynamic events, or inserting contents that should affect the behaviors of other objects in real-world videos, remains a major challenge. Existing trained models struggle with complex edits, likely due to the difficulty of collecting relevant training data. Similarly, existing training-free methods are inherently restricted to structure- and motion-preserving edits and do not support modification of motion or interactions. Here, we introduce DynaEdit, a training-free editing method that unlocks versatile video editing capabilities with pretrained text-to-video flow models. Our method relies on the recently introduced inversion-free approach, which does not intervene in the model internals, and is thus model-agnostic. We show that naively attempting to adapt this approach to general unconstrained editing results in severe low-frequency misalignment and high-frequency jitter. We explain the sources for these phenomena and introduce novel mechanisms for overcoming them. Through extensive experiments, we show that DynaEdit achieves state-of-the-art results on complex text-based video editing tasks, including modifying actions, inserting objects that interact with the scene, and introducing global effects.
>
---
#### [new 021] Learning Transferable Temporal Primitives for Video Reasoning via Synthetic Videos
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决视频中时间推理不足的问题。通过生成合成视频，学习可迁移的时间基础单元，提升模型在时间动态上的推理能力。**

- **链接: [https://arxiv.org/pdf/2603.17693](https://arxiv.org/pdf/2603.17693)**

> **作者:** Songtao Jiang; Sibo Song; Chenyi Zhou; Yuan Wang; Ruizhe Chen; Tongkun Guan; Ruilin Luo; Yan Zhang; Zhihang Tang; Yuchong Sun; Hang Zhang; Zhibo Yang; Shuai Bai; Junyang Lin; Zuozhu Liu
>
> **摘要:** The transition from image to video understanding requires vision-language models (VLMs) to shift from recognizing static patterns to reasoning over temporal dynamics such as motion trajectories, speed changes, and state transitions. Yet current post-training methods fall short due to two critical limitations: (1) existing datasets often lack temporal-centricity, where answers can be inferred from isolated keyframes rather than requiring holistic temporal integration; and (2) training data generated by proprietary models contains systematic errors in fundamental temporal perception, such as confusing motion directions or misjudging speeds. We introduce SynRL, a post-training framework that teaches models temporal primitives, the fundamental building blocks of temporal understanding including direction, speed, and state tracking. Our key insight is that these abstract primitives, learned from programmatically generated synthetic videos, transfer effectively to real-world scenarios. We decompose temporal understanding into short-term perceptual primitives (speed, direction) and long-term cognitive primitives, constructing 7.7K CoT and 7K RL samples with ground-truth frame-level annotations through code-based video generation. Despite training on simple geometric shapes, SynRL achieves substantial improvements across 15 benchmarks spanning temporal grounding, complex reasoning, and general video understanding. Remarkably, our 7.7K synthetic CoT samples outperform Video-R1 with 165K real-world samples. We attribute this to fundamental temporal skills, such as tracking frame by frame changes and comparing velocity, that transfer effectively from abstract synthetic patterns to complex real-world scenarios. This establishes a new paradigm for video post-training: video temporal learning through carefully designed synthetic data provides a more cost efficient scaling path.
>
---
#### [new 022] SARE: Sample-wise Adaptive Reasoning for Training-free Fine-grained Visual Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于细粒度视觉识别任务，解决LVLM在子类识别中的视觉模糊问题。提出SARE框架，通过自适应推理提升准确率并减少计算开销。**

- **链接: [https://arxiv.org/pdf/2603.17729](https://arxiv.org/pdf/2603.17729)**

> **作者:** Jingxiao Yang; DaLin He; Miao Pan; Ge Su; Wenqi Zhang; Yifeng Hu; Tangwei Li; Yuke Li; Xuhong Zhang
>
> **备注:** preprint, under review
>
> **摘要:** Recent advances in Large Vision-Language Models (LVLMs) have enabled training-free Fine-Grained Visual Recognition (FGVR). However, effectively exploiting LVLMs for FGVR remains challenging due to the inherent visual ambiguity of subordinate-level categories. Existing methods predominantly adopt either retrieval-oriented or reasoning-oriented paradigms to tackle this challenge, but both are constrained by two fundamental limitations:(1) They apply the same inference pipeline to all samples without accounting for uneven recognition difficulty, thereby leading to suboptimal accuracy and efficiency; (2) The lack of mechanisms to consolidate and reuse error-specific experience causes repeated failures on similar challenging cases. To address these limitations, we propose SARE, a Sample-wise Adaptive textbfREasoning framework for training-free FGVR. Specifically, SARE adopts a cascaded design that combines fast candidate retrieval with fine-grained reasoning, invoking the latter only when necessary. In the reasoning process, SARE incorporates a self-reflective experience mechanism that leverages past failures to provide transferable discriminative guidance during inference, without any parameter updates. Extensive experiments across 14 datasets substantiate that SARE achieves state-of-the-art performance while substantially reducing computational overhead.
>
---
#### [new 023] Symphony: A Cognitively-Inspired Multi-Agent System for Long-Video Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Symphony系统，解决长视频理解任务中的复杂推理与信息定位问题，通过多智能体协作和认知模仿提升性能。**

- **链接: [https://arxiv.org/pdf/2603.17307](https://arxiv.org/pdf/2603.17307)**

> **作者:** Haiyang Yan; Hongyun Zhou; Peng Xu; Xiaoxue Feng; Mengyi Liu
>
> **备注:** Accepted by cvpr2026
>
> **摘要:** Despite rapid developments and widespread applications of MLLM agents, they still struggle with long-form video understanding (LVU) tasks, which are characterized by high information density and extended temporal spans. Recent research on LVU agents demonstrates that simple task decomposition and collaboration mechanisms are insufficient for long-chain reasoning tasks. Moreover, directly reducing the time context through embedding-based retrieval may lose key information of complex problems. In this paper, we propose Symphony, a multi-agent system, to alleviate these limitations. By emulating human cognition patterns, Symphony decomposes LVU into fine-grained subtasks and incorporates a deep reasoning collaboration mechanism enhanced by reflection, effectively improving the reasoning capability. Additionally, Symphony provides a VLM-based grounding approach to analyze LVU tasks and assess the relevance of video segments, which significantly enhances the ability to locate complex problems with implicit intentions and large temporal spans. Experimental results show that Symphony achieves state-of-the-art performance on LVBench, LongVideoBench, VideoMME, and MLVU, with a 5.0% improvement over the prior state-of-the-art method on LVBench. Code is available at this https URL.
>
---
#### [new 024] DesertFormer: Transformer-Based Semantic Segmentation for Off-Road Desert Terrain Classification in Autonomous Navigation Systems
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于语义分割任务，解决沙漠非结构化环境下的地形分类问题。提出DesertFormer模型，提升沙漠地形识别精度，并分析错误模式以优化训练方法。**

- **链接: [https://arxiv.org/pdf/2603.17056](https://arxiv.org/pdf/2603.17056)**

> **作者:** Yasaswini Chebolu
>
> **备注:** 10 pages, 6 figures, 3 tables. Preprint also available on Zenodo (DOI: https://doi.org/10.5281/zenodo.19053085)
>
> **摘要:** Reliable terrain perception is a fundamental requirement for autonomous navigation in unstructured, off-road environments. Desert landscapes present unique challenges due to low chromatic contrast between terrain categories, extreme lighting variability, and sparse vegetation that defy the assumptions of standard road-scene segmentation models. We present DesertFormer, a semantic segmentation pipeline for off-road desert terrain analysis based on SegFormer B2 with a hierarchical Mix Transformer (MiT-B2) backbone. The system classifies terrain into ten ecologically meaningful categories -- Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Flowers, Logs, Rocks, Landscape, and Sky -- enabling safety-aware path planning for ground robots and autonomous vehicles. Trained on a purpose-built dataset of 4,176 annotated off-road images at 512x512 resolution, DesertFormer achieves a mean Intersection-over-Union (mIoU) of 64.4% and pixel accuracy of 86.1%, representing a +24.2% absolute improvement over a DeepLabV3 MobileNetV2 baseline (41.0% mIoU). We further contribute a systematic failure analysis identifying the primary confusion patterns -- Ground Clutter to Landscape and Dry Grass to Landscape -- and propose class-weighted training and copy-paste augmentation for rare terrain categories. Code, checkpoints, and an interactive inference dashboard are released at this https URL.
>
---
#### [new 025] ReLaGS: Relational Language Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文提出ReLaGS框架，解决3D场景中开放词汇的语义感知与关系推理问题。通过构建层次化语言蒸馏高斯场景和3D语义图，实现无需特定场景训练的高效推理。**

- **链接: [https://arxiv.org/pdf/2603.17605](https://arxiv.org/pdf/2603.17605)**

> **作者:** Yaxu Xie; Abdalla Arafa; Alireza Javanmardi; Christen Millerdurai; Jia Cheng Hu; Shaoxiang Wang; Alain Pagani; Didier Stricker
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Achieving unified 3D perception and reasoning across tasks such as segmentation, retrieval, and relation understanding remains challenging, as existing methods are either object-centric or rely on costly training for inter-object reasoning. We present a novel framework that constructs a hierarchical language-distilled Gaussian scene and its 3D semantic scene graph without scene-specific training. A Gaussian pruning mechanism refines scene geometry, while a robust multi-view language alignment strategy aggregates noisy 2D features into accurate 3D object embeddings. On top of this hierarchy, we build an open-vocabulary 3D scene graph with Vision Language derived annotations and Graph Neural Network-based relational reasoning. Our approach enables efficient and scalable open-vocabulary 3D reasoning by jointly modeling hierarchical semantics and inter/intra-object relationships, validated across tasks including open-vocabulary segmentation, scene graph generation, and relation-guided retrieval. Project page: this https URL
>
---
#### [new 026] Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文属于图像水印任务，旨在解决AI编辑对数字图像真实性威胁的问题。通过利用图像块间关系的不变性，提出Rel-Zero框架，实现无损且鲁棒的零水印方案。**

- **链接: [https://arxiv.org/pdf/2603.17531](https://arxiv.org/pdf/2603.17531)**

> **作者:** Pengzhen Chen; Yanwei Liu; Xiaoyan Gu; Xiaojun Chen; Wu Liu; Weiping Wang
>
> **备注:** accepted to CVPR 2026
>
> **摘要:** Recent advancements in diffusion-based image editing pose a significant threat to the authenticity of digital visual content. Traditional embedding-based watermarking methods often introduce perceptible perturbations to maintain robustness, inevitably compromising visual fidelity. Meanwhile, existing zero-watermarking approaches, typically relying on global image features, struggle to withstand sophisticated manipulations. In this work, we uncover a key observation: while individual image patches undergo substantial alterations during AI-based editing, the relational distance between patch pairs remains relatively invariant. Leveraging this property, we propose Relational Zero-Watermarking (Rel-Zero), a novel framework that requires no modification to the original image but derives a unique zero-watermark from these editing-invariant patch relations. By grounding the watermark in intrinsic structural consistency rather than absolute appearance, Rel-Zero provides a non-invasive yet resilient mechanism for content authentication. Extensive experiments demonstrate that Rel-Zero achieves substantially improved robustness across diverse editing models and manipulations compared to prior zero-watermarking approaches.
>
---
#### [new 027] AHOY! Animatable Humans under Occlusion from YouTube Videos with Gaussian Splatting and Video Diffusion Priors
- **分类: cs.CV**

- **简介: 该论文提出AHOY方法，用于从存在遮挡的单目视频中重建完整可动画的3D高斯人像。解决真实场景下遮挡导致的重建难题，通过生成监督、分阶段架构等技术提升重建质量。**

- **链接: [https://arxiv.org/pdf/2603.17975](https://arxiv.org/pdf/2603.17975)**

> **作者:** Aymen Mir; Riza Alp Guler; Xiangjun Tang; Peter Wonka; Gerard Pons-Moll
>
> **备注:** Our project page is available at this https URL
>
> **摘要:** We present AHOY, a method for reconstructing complete, animatable 3D Gaussian avatars from in-the-wild monocular video despite heavy occlusion. Existing methods assume unoccluded input-a fully visible subject, often in a canonical pose-excluding the vast majority of real-world footage where people are routinely occluded by furniture, objects, or other people. Reconstructing from such footage poses fundamental challenges: large body regions may never be observed, and multi-view supervision per pose is unavailable. We address these challenges with four contributions: (i) a hallucination-as-supervision pipeline that uses identity-finetuned diffusion models to generate dense supervision for previously unobserved body regions; (ii) a two-stage canonical-to-pose-dependent architecture that bootstraps from sparse observations to full pose-dependent Gaussian maps; (iii) a map-pose/LBS-pose decoupling that absorbs multi-view inconsistencies from the generated data; (iv) a head/body split supervision strategy that preserves facial identity. We evaluate on YouTube videos and on multi-view capture data with significant occlusion and demonstrate state-of-the-art reconstruction quality. We also demonstrate that the resulting avatars are robust enough to be animated with novel poses and composited into 3DGS scenes captured using cell-phone video. Our project page is available at this https URL
>
---
#### [new 028] VideoAtlas: Navigating Long-Form Video in Logarithmic Compute
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出VideoAtlas，解决视频理解中的长上下文与损失性表示问题，通过分层网格结构实现高效、无损视频导航与处理。**

- **链接: [https://arxiv.org/pdf/2603.17948](https://arxiv.org/pdf/2603.17948)**

> **作者:** Mohamed Eltahir; Ali Habibullah; Yazan Alshoibi; Lama Ayash; Tanveer Hussain; Naeemullah Khan
>
> **摘要:** Extending language models to video introduces two challenges: representation, where existing methods rely on lossy approximations, and long-context, where caption- or agent-based pipelines collapse video into text and lose visual fidelity. To overcome this, we introduce \textbf{VideoAtlas}, a task-agnostic environment to represent video as a hierarchical grid that is simultaneously lossless, navigable, scalable, caption- and preprocessing-free. An overview of the video is available at a glance, and any region can be recursively zoomed into, with the same visual representation used uniformly for the video, intermediate investigations, and the agent's memory, eliminating lossy text conversion end-to-end. This hierarchical structure ensures access depth grows only logarithmically with video length. For long-context, Recursive Language Models (RLMs) recently offered a powerful solution for long text, but extending them to visual domain requires a structured environment to recurse into, which \textbf{VideoAtlas} provides. \textbf{VideoAtlas} as a Markov Decision Process unlocks Video-RLM: a parallel Master-Worker architecture where a Master coordinates global exploration while Workers concurrently drill into assigned regions to accumulate lossless visual evidence. We demonstrate three key findings: (1)~logarithmic compute growth with video duration, further amplified by a 30-60\% multimodal cache hit rate arising from the grid's structural reuse. (2)~environment budgeting, where bounding the maximum exploration depth provides a principled compute-accuracy hyperparameter. (3)~emergent adaptive compute allocation that scales with question granularity. When scaling from 1-hour to 10-hour benchmarks, Video-RLM remains the most duration-robust method with minimal accuracy degradation, demonstrating that structured environment navigation is a viable and scalable paradigm for video understanding.
>
---
#### [new 029] S-VGGT: Structure-Aware Subscene Decomposition for Scalable 3D Foundation Models
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决3D基础模型的计算成本过高问题。提出S-VGGT方法，在结构层级减少冗余，提升计算效率。**

- **链接: [https://arxiv.org/pdf/2603.17625](https://arxiv.org/pdf/2603.17625)**

> **作者:** Xinze Li; Pengxu Chen; Yiyuan Wang; Weifeng Su; Wentao Cheng
>
> **备注:** 7 pages, 5 figures. Accepted by ICME 2026
>
> **摘要:** Feed-forward 3D foundation models face a key challenge: the quadratic computational cost introduced by global attention, which severely limits scalability as input length increases. Concurrent acceleration methods, such as token merging, operate at the token level. While they offer local savings, the required nearest-neighbor searches introduce undesirable overhead. Consequently, these techniques fail to tackle the fundamental issue of structural redundancy dominant in dense capture data. In this work, we introduce \textbf{S-VGGT}, a novel approach that addresses redundancy at the structural frame level, drastically shifting the optimization focus. We first leverage the initial features to build a dense scene graph, which characterizes structural scene redundancy and guides the subsequent scene partitioning. Using this graph, we softly assign frames to a small number of subscenes, guaranteeing balanced groups and smooth geometric transitions. The core innovation lies in designing the subscenes to share a common reference frame, establishing a parallel geometric bridge that enables independent and highly efficient processing without explicit geometric alignment. This structural reorganization provides strong intrinsic acceleration by cutting the global attention cost at its source. Crucially, S-VGGT is entirely orthogonal to token-level acceleration methods, allowing the two to be seamlessly combined for compounded speedups without compromising reconstruction fidelity. Code is available at this https URL.
>
---
#### [new 030] Visual Product Search Benchmark
- **分类: cs.CV; cs.IR**

- **简介: 该论文属于视觉产品检索任务，旨在解决工业场景中精确匹配产品的问题。通过构建基准测试，评估多种模型的检索能力，以提升产品识别的准确性。**

- **链接: [https://arxiv.org/pdf/2603.17186](https://arxiv.org/pdf/2603.17186)**

> **作者:** Karthik Sulthanpete Govindappa
>
> **备注:** 21 pages
>
> **摘要:** Reliable product identification from images is a critical requirement in industrial and commercial applications, particularly in maintenance, procurement, and operational workflows where incorrect matches can lead to costly downstream failures. At the core of such systems lies the visual search component, which must retrieve and rank the exact object instance from large and continuously evolving catalogs under diverse imaging conditions. This report presents a structured benchmark of modern visual embedding models for instance-level image retrieval, with a focus on industrial applications. A curated set of open-source foundation embedding models, proprietary multi-modal embedding systems, and domain-specific vision-only models are evaluated under a unified image-to-image retrieval protocol. The benchmark includes curated datasets, which includes industrial datasets derived from production deployments in Manufacturing, Automotive, DIY, and Retail, as well as established public benchmarks. Evaluation is conducted without post-processing, isolating the retrieval capability of each model. The results provide insight into how well contemporary foundation and unified embedding models transfer to fine-grained instance retrieval tasks, and how they compare to models explicitly trained for industrial applications. By emphasizing realistic constraints, heterogeneous image conditions, and exact instance matching requirements, this benchmark aims to inform both practitioners and researchers about the strengths and limitations of current visual embedding approaches in production-level product identification systems. An interactive companion website presenting the benchmark results, evaluation details, and additional visualizations is available at this https URL.
>
---
#### [new 031] Edit-As-Act: Goal-Regressive Planning for Open-Vocabulary 3D Indoor Scene Editing
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Edit-As-Act框架，解决开放词汇3D室内场景编辑问题。通过目标逆向规划，实现语义一致、物理合理的场景修改。**

- **链接: [https://arxiv.org/pdf/2603.17583](https://arxiv.org/pdf/2603.17583)**

> **作者:** Seongrae Noh; SeungWon Seo; Gyeong-Moon Park; HyeongYeop Kang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Editing a 3D indoor scene from natural language is conceptually straightforward but technically challenging. Existing open-vocabulary systems often regenerate large portions of a scene or rely on image-space edits that disrupt spatial structure, resulting in unintended global changes or physically inconsistent layouts. These limitations stem from treating editing primarily as a generative task. We take a different view. A user instruction defines a desired world state, and editing should be the minimal sequence of actions that makes this state true while preserving everything else. This perspective motivates Edit-As-Act, a framework that performs open-vocabulary scene editing as goal-regressive planning in 3D space. Given a source scene and free-form instruction, Edit-As-Act predicts symbolic goal predicates and plans in EditLang, a PDDL-inspired action language that we design with explicit preconditions and effects encoding support, contact, collision, and other geometric relations. A language-driven planner proposes actions, and a validator enforces goal-directedness, monotonicity, and physical feasibility, producing interpretable and physically coherent transformations. By separating reasoning from low-level generation, Edit-As-Act achieves instruction fidelity, semantic consistency, and physical plausibility - three criteria that existing paradigms cannot satisfy together. On E2A-Bench, our benchmark of 63 editing tasks across 9 indoor environments, Edit-As-Act significantly outperforms prior approaches across all edit types and scene categories.
>
---
#### [new 032] FrescoDiffusion: 4K Image-to-Video with Prior-Regularized Tiled Diffusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出FrescoDiffusion，解决4K图像到视频生成中的全局一致性与细节保留问题。通过引入先验正则化，提升大尺寸视频生成的连贯性与质量。**

- **链接: [https://arxiv.org/pdf/2603.17555](https://arxiv.org/pdf/2603.17555)**

> **作者:** Hugo Caselles-Dupré; Mathis Koroglu; Guillaume Jeanneret; Arnaud Dapogny; Matthieu Cord
>
> **备注:** 5 authors. Hugo Caselles-Dupré, Mathis Koroglu, and Guillaume Jeanneret contributed equally. 14 pages, 7 figures
>
> **摘要:** Diffusion-based image-to-video (I2V) models are increasingly effective, yet they struggle to scale to ultra-high-resolution inputs (e.g., 4K). Generating videos at the model's native resolution often loses fine-grained structure, whereas high-resolution tiled denoising preserves local detail but breaks global layout consistency. This failure mode is particularly severe in the fresco animation setting: monumental artworks containing many distinct characters, objects, and semantically different sub-scenes that must remain spatially coherent over time. We introduce FrescoDiffusion, a training-free method for coherent large-format I2V generation from a single complex image. The key idea is to augment tiled denoising with a precomputed latent prior: we first generate a low-resolution video at the underlying model resolution and upsample its latent trajectory to obtain a global reference that captures long-range temporal and spatial structure. For 4K generation, we compute per-tile noise predictions and fuse them with this reference at every diffusion timestep by minimizing a single weighted least-squares objective in model-output space. The objective combines a standard tile-merging criterion with our regularization term, yielding a closed-form fusion update that strengthens global coherence while retaining fine detail. We additionally provide a spatial regularization variable that enables region-level control over where motion is allowed. Experiments on the VBench-I2V dataset and our proposed fresco I2V dataset show improved global consistency and fidelity over tiled baselines, while being computationally efficient. Our regularization enables explicit controllability of the trade-off between creativity and consistency.
>
---
#### [new 033] Interpretable Traffic Responsibility from Dashcam Video via Legal Multi Agent Reasoning
- **分类: cs.CV**

- **简介: 该论文属于交通责任判定任务，旨在将 dashcam 视频转化为法律责任判断。通过构建多模态数据集和双阶段框架，实现视频理解与法律推理的结合，解决视频证据与法律条款映射的问题。**

- **链接: [https://arxiv.org/pdf/2603.17930](https://arxiv.org/pdf/2603.17930)**

> **作者:** Jingchun Yang; Jinchang Zhang
>
> **摘要:** The widespread adoption of dashcams has made video evidence in traffic accidents increasingly abundant, yet transforming "what happened in the video" into "who is responsible under which legal provisions" still relies heavily on human experts. Existing ego-view traffic accident studies mainly focus on perception and semantic understanding, while LLM-based legal methods are mostly built on textual case descriptions and rarely incorporate video evidence, leaving a clear gap between the two. We first propose C-TRAIL, a multimodal legal dataset that, under the Chinese traffic regulation system, explicitly aligns dashcam videos and textual descriptions with a closed set of responsibility modes and their corresponding Chinese traffic statutes. On this basis, we introduce a two-stage framework: (1) a traffic accident understanding module that generates textual video descriptions; and (2) a legal multi-agent framework that outputs responsibility modes, statute sets, and complete judgment reports. Experimental results on C-TRAIL and MM-AU show that our method outperforms general and legal LLMs, as well as existing agent-based approaches, while providing a transparent and interpretable legal reasoning process.
>
---
#### [new 034] Omni-I2C: A Holistic Benchmark for High-Fidelity Image-to-Code Generation
- **分类: cs.CV**

- **简介: 该论文提出Omni-I2C基准，用于评估大模型将复杂图形转化为可执行代码的能力。解决多模态代码生成任务中的结构与语义准确性问题，通过多样化数据和细化评估框架揭示模型缺陷。**

- **链接: [https://arxiv.org/pdf/2603.17508](https://arxiv.org/pdf/2603.17508)**

> **作者:** Jiawei Zhou; Chi Zhang; Xiang Feng; Qiming Zhang; Haibo Qiu; Lihuo He; Dengpan Ye; Xinbo Gao; Jing Zhang
>
> **备注:** 35 pages, 26 figures
>
> **摘要:** We present Omni-I2C, a comprehensive benchmark designed to evaluate the capability of Large Multimodal Models (LMMs) in converting complex, structured digital graphics into executable code. We argue that this task represents a non-trivial challenge for the current generation of LMMs: it demands an unprecedented synergy between high-fidelity visual perception -- to parse intricate spatial hierarchies and symbolic details -- and precise generative expression -- to synthesize syntactically sound and logically consistent code. Unlike traditional descriptive tasks, Omni-I2C requires a holistic understanding where any minor perceptual hallucination or coding error leads to a complete failure in visual reconstruction. Omni-I2C features 1080 meticulously curated samples, defined by its breadth across subjects, image modalities, and programming languages. By incorporating authentic user-sourced cases, the benchmark spans a vast spectrum of digital content -- from scientific visualizations to complex symbolic notations -- each paired with executable reference code. To complement this diversity, our evaluation framework provides necessary depth; by decoupling performance into perceptual fidelity and symbolic precision, it transcends surface-level accuracy to expose the granular structural failures and reasoning bottlenecks of current LMMs. Our evaluation reveals a substantial performance gap among leading LMMs; even state-of-the-art models struggle to preserve structural integrity in complex scenarios, underscoring that multimodal code generation remains a formidable challenge. Data and code are available at this https URL.
>
---
#### [new 035] Parameter-Efficient Modality-Balanced Symmetric Fusion for Multimodal Remote Sensing Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于多模态遥感语义分割任务，旨在解决模型参数量大和模态不平衡问题。提出MoBaNet框架，通过对称结构和自适应融合模块实现高效、平衡的多模态融合。**

- **链接: [https://arxiv.org/pdf/2603.17705](https://arxiv.org/pdf/2603.17705)**

> **作者:** Haocheng Li; Juepeng Zheng; Shuangxi Miao; Ruibo Lu; Guosheng Cai; Haohuan Fu; Jianxi Huang
>
> **备注:** 14 pages, 6 figures
>
> **摘要:** Multimodal remote sensing semantic segmentation enhances scene interpretation by exploiting complementary physical cues from heterogeneous data. Although pretrained Vision Foundation Models (VFMs) provide strong general-purpose representations, adapting them to multimodal tasks often incurs substantial computational overhead and is prone to modality imbalance, where the contribution of auxiliary modalities is suppressed during optimization. To address these challenges, we propose MoBaNet, a parameter-efficient and modality-balanced symmetric fusion framework. Built upon a largely frozen VFM backbone, MoBaNet adopts a symmetric dual-stream architecture to preserve generalizable representations while minimizing the number of trainable parameters. Specifically, we design a Cross-modal Prompt-Injected Adapter (CPIA) to enable deep semantic interaction by generating shared prompts and injecting them into bottleneck adapters under the frozen backbone. To obtain compact and discriminative multimodal representations for decoding, we further introduce a Difference-Guided Gated Fusion Module (DGFM), which adaptively fuses paired stage features by explicitly leveraging cross-modal discrepancy to guide feature selection. Furthermore, we propose a Modality-Conditional Random Masking (MCRM) strategy to mitigate modality imbalance by masking one modality only during training and imposing hard-pixel auxiliary supervision on modality-specific branches. Extensive experiments on the ISPRS Vaihingen and Potsdam benchmarks demonstrate that MoBaNet achieves state-of-the-art performance with significantly fewer trainable parameters than full fine-tuning, validating its effectiveness for robust and balanced multimodal fusion. The source code in this work is available at this https URL.
>
---
#### [new 036] A Multi-Agent System for Building-Age Cohort Mapping to Support Urban Energy Planning
- **分类: cs.CV**

- **简介: 该论文属于城市建筑年龄分类任务，旨在解决数据不一致和缺失问题。通过多智能体系统融合多源数据，并使用深度学习模型提升分类精度，支持城市能源规划。**

- **链接: [https://arxiv.org/pdf/2603.17626](https://arxiv.org/pdf/2603.17626)**

> **作者:** Kundan Thota; Thorsten Schlachter; Veit Hagenmeyer
>
> **摘要:** Determining the age distribution of the urban building stock is crucial for sustainable municipal heat planning and upgrade prioritization. However, existing approaches often rely on datasets gathered via sensors or remote sensing techniques, leaving inconsistencies and gaps in data. We present a multi-agent LLM system comprising three key agents, the Zensus agent, the OSM agent, and the Monument agent, that fuse data from heterogeneous sources. A data orchestrator and harmonizer geocodes and deduplicates building imprints. Using this fused ground truth, we introduce BuildingAgeCNN, a satellite-only classifier based on a ConvNeXt backbone augmented with a Feature Pyramid Network (FPN), CoordConv spatial channels, and Squeeze-and-Excitation (SE) blocks. Under spatial cross validation, BuildingAgeCNN attains an overall accuracy of 90.69% but a modest macro-F1 of 67.25%, reflecting strong class imbalance and persistent confusions between adjacent historical cohorts. To mitigate risk for planning applications, the address-to prediction pipeline includes calibrated confidence estimates and flags low-confidence cases for manual review. This multi-agent LLM system not only assists in gathering structured data but also helps energy demand planners optimize district-heating networks and target low-carbon sustainable energy systems.
>
---
#### [new 037] Steering Video Diffusion Transformers with Massive Activations
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在提升视频质量与时间一致性。通过分析视频扩散Transformer中的大规模激活，提出STAS方法，无额外计算开销地优化关键帧激活值。**

- **链接: [https://arxiv.org/pdf/2603.17825](https://arxiv.org/pdf/2603.17825)**

> **作者:** Xianhang Cheng; Yujian Zheng; Zhenyu Xie; Tingting Liao; Hao Li
>
> **摘要:** Despite rapid progress in video diffusion transformers, how their internal model signals can be leveraged with minimal overhead to enhance video generation quality remains underexplored. In this work, we study the role of Massive Activations (MAs), which are rare, high-magnitude hidden state spikes in video diffusion transformers. We observed that MAs emerge consistently across all visual tokens, with a clear magnitude hierarchy: first-frame tokens exhibit the largest MA magnitudes, latent-frame boundary tokens (the head and tail portions of each temporal chunk in the latent space) show elevated but slightly lower MA magnitudes than the first frame, and interior tokens within each latent frame remain elevated, yet are comparatively moderate in magnitude. This structured pattern suggests that the model implicitly prioritizes token positions aligned with the temporal chunking in the latent space. Based on this observation, we propose Structured Activation Steering (STAS), a training-free self-guidance-like method that steers MA values at first-frame and boundary tokens toward a scaled global maximum reference magnitude. STAS achieves consistent improvements in terms of video quality and temporal coherence across different text-to-video models, while introducing negligible computational overhead.
>
---
#### [new 038] LoST: Level of Semantics Tokenization for 3D Shapes
- **分类: cs.CV; cs.GR; cs.LG**

- **简介: 该论文提出LoST，解决3D形状的语义化分词问题，通过按语义重要性排序tokens，提升生成质量和效率。**

- **链接: [https://arxiv.org/pdf/2603.17995](https://arxiv.org/pdf/2603.17995)**

> **作者:** Niladri Shekhar Dutt; Zifan Shi; Paul Guerrero; Chun-Hao Paul Huang; Duygu Ceylan; Niloy J. Mitra; Xuelin Chen
>
> **备注:** CVPR 2026; Project website-- this https URL
>
> **摘要:** Tokenization is a fundamental technique in the generative modeling of various modalities. In particular, it plays a critical role in autoregressive (AR) models, which have recently emerged as a compelling option for 3D generation. However, optimal tokenization of 3D shapes remains an open question. State-of-the-art (SOTA) methods primarily rely on geometric level-of-detail (LoD) hierarchies, originally designed for rendering and compression. These spatial hierarchies are often token-inefficient and lack semantic coherence for AR modeling. We propose Level-of-Semantics Tokenization (LoST), which orders tokens by semantic salience, such that early prefixes decode into complete, plausible shapes that possess principal semantics, while subsequent tokens refine instance-specific geometric and semantic details. To train LoST, we introduce Relational Inter-Distance Alignment (RIDA), a novel 3D semantic alignment loss that aligns the relational structure of the 3D shape latent space with that of the semantic DINO feature space. Experiments show that LoST achieves SOTA reconstruction, surpassing previous LoD-based 3D shape tokenizers by large margins on both geometric and semantic reconstruction metrics. Moreover, LoST achieves efficient, high-quality AR 3D generation and enables downstream tasks like semantic retrieval, while using only 0.1%-10% of the tokens needed by prior AR models.
>
---
#### [new 039] Harnessing the Power of Foundation Models for Accurate Material Classification
- **分类: cs.CV**

- **简介: 该论文属于材料分类任务，解决标注数据不足导致的模型性能问题。通过生成合成数据和融合视觉语言模型先验知识，提升分类准确性。**

- **链接: [https://arxiv.org/pdf/2603.17390](https://arxiv.org/pdf/2603.17390)**

> **作者:** Qingran Lin; Fengwei Yang; Chaolun Zhu
>
> **摘要:** Material classification has emerged as a critical task in computer vision and graphics, supporting the assignment of accurate material properties to a wide range of digital and real-world applications. While traditionally framed as an image classification task, this domain faces significant challenges due to the scarcity of annotated data, limiting the accuracy and generalizability of trained models. Recent advances in vision-language foundation models (VLMs) offer promising avenues to address these issues, yet existing solutions leveraging these models still exhibit unsatisfying results in material recognition tasks. In this work, we propose a novel framework that effectively harnesses foundation models to overcome data limitations and enhance classification accuracy. Our method integrates two key innovations: (a) a robust image generation and auto-labeling pipeline that creates a diverse and high-quality training dataset with material-centric images, and automatically assigns labels by fusing object semantics and material attributes in text prompts; (b) a prior incorporation strategy to distill information from VLMs, combined with a joint fine-tuning method that optimizes a pre-trained vision foundation model alongside VLM-derived priors, preserving broad generalizability while adapting to material-specific this http URL experiments demonstrate significant improvements on multiple datasets. We show that our synthetic dataset effectively captures the characteristics of real world materials, and the integration of priors from vision-language models significantly enhances the final performance. The source code and dataset will be released.
>
---
#### [new 040] Identity as Presence: Towards Appearance and Voice Personalized Joint Audio-Video Generation
- **分类: cs.CV**

- **简介: 该论文属于跨模态生成任务，旨在解决多身份下音频视频个性化生成的问题。提出统一框架，实现面部和声音的精细控制与一致性生成。**

- **链接: [https://arxiv.org/pdf/2603.17889](https://arxiv.org/pdf/2603.17889)**

> **作者:** Yingjie Chen; Shilun Lin; Cai Xing; Qixin Yan; Wenjing Wang; Dingming Liu; Hao Liu; Chen Li; Jing Lyu
>
> **摘要:** Recent advances have demonstrated compelling capabilities in synthesizing real individuals into generated videos, reflecting the growing demand for identity-aware content creation. Nevertheless, an openly accessible framework enabling fine-grained control over facial appearance and voice timbre across multiple identities remains unavailable. In this work, we present a unified and scalable framework for identity-aware joint audio-video generation, enabling high-fidelity and consistent personalization. Specifically, we introduce a data curation pipeline that automatically extracts identity-bearing information with paired annotations across audio and visual modalities, covering diverse scenarios from single-subject to multi-subject interactions. We further propose a flexible and scalable identity injection mechanism for single- and multi-subject scenarios, in which both facial appearance and vocal timbre act as identity-bearing control signals. Moreover, in light of modality disparity, we design a multi-stage training strategy to accelerate convergence and enforce cross-modal coherence. Experiments demonstrate the superiority of the proposed framework. For more details and qualitative results, please refer to our webpage: \href{this https URL}{Identity-as-Presence}.
>
---
#### [new 041] UniSem: Generalizable Semantic 3D Reconstruction from Sparse Unposed Images
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决稀疏无姿态图像下的语义重建问题。提出UniSem框架，通过EGD和MTC提升深度精度和语义泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.17519](https://arxiv.org/pdf/2603.17519)**

> **作者:** Guibiao Liao; Qian Ren; Kaimin Liao; Hua Wang; Zhi Chen; Luchao Wang; Yaohua Tang
>
> **摘要:** Semantic-aware 3D reconstruction from sparse, unposed images remains challenging for feed-forward 3D Gaussian Splatting (3DGS). Existing methods often predict an over-complete set of Gaussian primitives under sparse-view supervision, leading to unstable geometry and inferior depth quality. Meanwhile, they rely solely on 2D segmenter features for semantic lifting, which provides weak 3D-level and limited generalizable supervision, resulting in incomplete 3D semantics in novel scenes. To address these issues, we propose UniSem, a unified framework that jointly improves depth accuracy and semantic generalization via two key components. First, Error-aware Gaussian Dropout (EGD) performs error-guided capacity control by suppressing redundancy-prone Gaussians using rendering error cues, producing meaningful, geometrically stable Gaussian representations for improved depth estimation. Second, we introduce a Mix-training Curriculum (MTC) that progressively blends 2D segmenter-lifted semantics with the model's own emergent 3D semantic priors, implemented with object-level prototype alignment to enhance semantic coherence and completeness. Extensive experiments on ScanNet and Replica show that UniSem achieves superior performance in depth prediction and open-vocabulary 3D segmentation across varying numbers of input views. Notably, with 16-view inputs, UniSem reduces depth Rel by 15.2% and improves open-vocabulary segmentation mAcc by 3.7% over strong baselines.
>
---
#### [new 042] ECHO: Towards Emotionally Appropriate and Contextually Aware Interactive Head Generation
- **分类: cs.CV**

- **简介: 该论文属于交互式头部生成任务，旨在解决现有方法在上下文适配性和情感合理性上的不足。提出ECHO框架，通过长程上下文理解和空间解耦注意力模块提升生成效果。**

- **链接: [https://arxiv.org/pdf/2603.17427](https://arxiv.org/pdf/2603.17427)**

> **作者:** Xiangyu Kong; Xiaoyu Jin; Yihan Pan; Haoqin Sun; Hengde Zhu; Xiaoming Xu; Xiaoming Wei; Lu Liu; Siyang Song
>
> **备注:** 21 pages, 5 figures
>
> **摘要:** In natural face-to-face interaction, participants seamlessly alternate between speaking and listening, producing facial behaviors (FBs) that are finely informed by long-range context and naturally exhibit contextual appropriateness and emotional rationality. Interactive Head Generation (IHG) aims to synthesize lifelike avatar head video emulating such capabilities. Existing IHG methods typically condition on dual-track signals (i.e., human user's behaviors and pre-defined audio for avatar) within a short temporal window, jointly driving generation of avatar's audio-aligned lip articulation and non-verbal FBs. However, two main challenges persist in these methods: (i) the reliance on short-clip behavioral cues without long-range contextual modeling leads them to produce facial behaviors lacking contextual appropriateness; and (ii) the entangled, role-agnostic fusion of dual-track signals empirically introduces cross-signal interference, potentially compromising lip-region synchronization during speaking. To this end, we propose ECHO, a novel IHG framework comprising two key components: a Long-range Contextual Understanding (LCU) component that facilitates contextual understanding of both behavior-grounded dynamics and linguistic-driven affective semantics to promote contextual appropriateness and emotional rationality of synthesized avatar FBs; and a block-wise Spatial-aware Decoupled Cross-attention Modulation (SDCM) module, that preserves self-audio-driven lip articulation while adaptively integrating user contextual behavioral cues for non-lip facial regions, complemented by our designed two-stage training paradigm, to jointly enhance lip synchronization and visual fidelity. Extensive experiments demonstrate the effectiveness of proposed components and ECHO's superior IHG performance.
>
---
#### [new 043] AdaRadar: Rate Adaptive Spectral Compression for Radar-based Perception
- **分类: cs.CV**

- **简介: 该论文属于雷达数据压缩任务，解决高维雷达数据传输带宽不足问题。通过自适应压缩和离散余弦变换，实现高效数据传输与性能保持。**

- **链接: [https://arxiv.org/pdf/2603.17979](https://arxiv.org/pdf/2603.17979)**

> **作者:** Jinho Park; Se Young Chun; Mingoo Seok
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Radar is a critical perception modality in autonomous driving systems due to its all-weather characteristics and ability to measure range and Doppler velocity. However, the sheer volume of high-dimensional raw radar data saturates the communication link to the computing engine (e.g., an NPU), which is often a low-bandwidth interface with data rate provisioned only for a few low-resolution range-Doppler frames. A generalized codec for utilizing high-dimensional radar data is notably absent, while existing image-domain approaches are unsuitable, as they typically operate at fixed compression ratios and fail to adapt to varying or adversarial conditions. In light of this, we propose radar data compression with adaptive feedback. It dynamically adjusts the compression ratio by performing gradient descent from the proxy gradient of detection confidence with respect to the compression rate. We employ a zeroth-order gradient approximation as it enables gradient computation even with non-differentiable core operations--pruning and quantization. This also avoids transmitting the gradient tensors over the band-limited link, which, if estimated, would be as large as the original radar data. In addition, we have found that radar feature maps are heavily concentrated on a few frequency components. Thus, we apply the discrete cosine transform to the radar data cubes and selectively prune out the coefficients effectively. We preserve the dynamic range of each radar patch through scaled quantization. Combining those techniques, our proposed online adaptive compression scheme achieves over 100x feature size reduction at minimal performance drop (~1%p). We validate our results on the RADIal, CARRADA, and Radatron datasets.
>
---
#### [new 044] ResNet-50 with Class Reweighting and Anatomy-Guided Temporal Decoding for Gastrointestinal Video Analysis
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多标签胃肠视频分析任务，解决类别不平衡和时间事件匹配问题。通过改进ResNet-50和引入解码策略提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.17784](https://arxiv.org/pdf/2603.17784)**

> **作者:** Romil Imtiaz; Dimitris K. Iakovidis
>
> **备注:** ICPR 2026 RARE-VISION Competition
>
> **摘要:** We developed a multi-label gastrointestinal video analysis pipeline based on a ResNet-50 frame classifier followed by anatomy-guided temporal event decoding. The system predicts 17 labels, including 5 anatomy classes and 12 pathology classes, from frames resized to 336x336. A major challenge was severe class imbalance, particularly for rare pathology labels. To address this, we used clipped class-wise positive weighting in the training loss, which improved rare-class learning while maintaining stable optimization. At the temporal stage, we found that direct frame-to-event conversion produced fragmented mismatches with the official ground truth. The final submission therefore combined GT-style framewise event composition, anatomy vote smoothing, and anatomy-based pathology gating with a conservative hysteresis decoder. This design improved the final temporal mAP from 0.3801 to 0.4303 on the challenge test set.
>
---
#### [new 045] FineViT: Progressively Unlocking Fine-Grained Perception with Dense Recaptions
- **分类: cs.CV**

- **简介: 该论文提出FineViT，解决MLLM视觉编码器在细粒度感知上的性能瓶颈。通过高分辨率预训练和局部对齐提升视觉理解能力，显著改善零样本识别与检索效果。**

- **链接: [https://arxiv.org/pdf/2603.17326](https://arxiv.org/pdf/2603.17326)**

> **作者:** Peisen Zhao; Xiaopeng Zhang; Mingxing Xu; Ruoyu Sun; Zewei Du; Dunzheng Wang; Guanghao Zheng; Haohang Xu; Zhibo Zhang; Yuhang Zhang; Yi Ai; Lin Liu; Qi Tian
>
> **摘要:** While Multimodal Large Language Models (MLLMs) have experienced rapid advancements, their visual encoders frequently remain a performance bottleneck. Conventional CLIP-based encoders struggle with dense spatial tasks due to the loss of visual details caused by low-resolution pretraining and the reliance on noisy, coarse web-crawled image-text pairs. To overcome these limitations, we introduce FineViT, a novel vision encoder specifically designed to unlock fine-grained perception. By replacing coarse web data with dense recaptions, we systematically mitigate information loss through a progressive training paradigm.: first, the encoder is trained from scratch at a high native resolution on billions of global recaptioned image-text pairs, establishing a robust, detail rich semantic foundation. Subsequently, we further enhance its local perception through LLM alignment, utilizing our curated FineCap-450M dataset that comprises over $450$ million high quality local captions. Extensive experiments validate the effectiveness of the progressive strategy. FineViT achieves state-of-the-art zero-shot recognition and retrieval performance, especially in long-context retrieval, and consistently outperforms multimodal visual encoders such as SigLIP2 and Qwen-ViT when integrated into MLLMs. We hope FineViT could serve as a powerful new baseline for fine-grained visual perception.
>
---
#### [new 046] Exploring parameter-efficient fine-tuning (PEFT) of billion-parameter vision models with QLoRA and DoRA: insights into generalization for limited-data image classification under a 98:1 test-to-train regime
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决农业领域数据有限下的模型泛化问题。通过PEFT方法优化大模型，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2603.17782](https://arxiv.org/pdf/2603.17782)**

> **作者:** Haiyu Yang; Sumit Sharma; Enhong Liu; Miel Hostens
>
> **摘要:** Automated behavior classification is essential for precision livestock farming but faces challenges of high computational costs and limited labeled data. This study systematically compared three approaches: training from scratch (ResNet-18, ViT-Small), frozen feature extraction, and parameter-efficient fine-tuning (PEFT) of the DINOv3 foundation model (6.7 billion parameters). We evaluated QLoRA and DoRA across multiple configurations varying rank (8, 16, 64) and target modules (q_proj versus all-linear layers). With 2,160 verified training images, we assessed generalization of our model on 211,800 test samples, which is essentially a 98:1 test-to-train ratio. Results demonstrated that PEFT substantially outperformed alternatives, where the best QLoRA configuration (all-linear layers and rank=64) achieved 83.16% test accuracy with only 2.72% parameters (183.0M) in 5.8 hours, compared to 72.87% for ResNet-18 (16.8 hours), 61.91% for ViT-Small (18.7 hours), and 76.56% for frozen DINOv3 (17.5 hours). DoRA achieved comparable accuracy (83.14%) but with longer training time (11.0 hours). Notably, increasing adapter capacity consistently improved generalization while simultaneously not causing overfitting: reducing rank from 16 to 8 decreased test accuracy from 78.38% to 77.17%, while expanding from q_proj-only to all-linear layers with rank=64 improved accuracy from 78.38% to 83.16%. This suggests underfitting, instead of overfitting, is the primary challenge when adapting foundation models to agricultural imagery. Our findings provide guidelines for deploying billion-parameter vision models with PEFT in agricultural livestock applications.
>
---
#### [new 047] ProGVC: Progressive-based Generative Video Compression via Auto-Regressive Context Modeling
- **分类: cs.CV**

- **简介: 该论文属于视频压缩任务，旨在解决传统编码器在低比特率下细节丢失及缺乏可变码率支持的问题。提出ProGVC框架，结合渐进传输与生成模型，提升压缩效率与视觉质量。**

- **链接: [https://arxiv.org/pdf/2603.17546](https://arxiv.org/pdf/2603.17546)**

> **作者:** Daowen Li; Ruixiao Dong; Ying Chen; Kai Li; Ding Ding; Li Li
>
> **摘要:** Perceptual video compression leverages generative priors to reconstruct realistic textures and motions at low bitrates. However, existing perceptual codecs often lack native support for variable bitrate and progressive delivery, and their generative modules are weakly coupled with entropy coding, limiting bitrate reduction. Inspired by the next-scale prediction in the Visual Auto-Regressive (VAR) models, we propose ProGVC, a Progressive-based Generative Video Compression framework that unifies progressive transmission, efficient entropy coding, and detail synthesis within a single codec. ProGVC encodes videos into hierarchical multi-scale residual token maps, enabling flexible rate adaptation by transmitting a coarse-to-fine subset of scales in a progressive manner. A Transformer-based multi-scale autoregressive context model estimates token probabilities, utilized both for efficient entropy coding of the transmitted tokens and for predicting truncated fine-scale tokens at the decoder to restore perceptual details. Extensive experiments demonstrate that as a new coding paradigm, ProGVC delivers promising perceptual compression performance at low bitrates while offering practical scalability at the same time.
>
---
#### [new 048] Script-to-Slide Grounding: Grounding Script Sentences to Slide Objects for Automatic Instructional Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Script-to-Slide Grounding任务，解决视频编辑中手动将脚本与幻灯片对象对齐的问题，利用大语言模型实现自动匹配。**

- **链接: [https://arxiv.org/pdf/2603.16931](https://arxiv.org/pdf/2603.16931)**

> **作者:** Rena Suzuki; Masato Kikuchi; Tadachika Ozono
>
> **备注:** The 21st International Conference on E-Service and Knowledge Management (ESKM 2025-Winter)
>
> **摘要:** While slide-based videos augmented with visual effects are widely utilized in education and research presentations, the video editing process -- particularly applying visual effects to ground spoken content to slide objects -- remains highly labor-intensive. This study aims to develop a system that automatically generates such instructional videos from slides and corresponding scripts. As a foundational step, this paper proposes and formulates Script-to-Slide Grounding (S2SG), defined as the task of grounding script sentences to their corresponding slide objects. Furthermore, as an initial step, we propose ``Text-S2SG,'' a method that utilizes a large language model (LLM) to perform this grounding task for text objects. Our experiments demonstrate that the proposed method achieves high performance (F1-score: 0.924). The contribution of this work is the formalization of a previously implicit slide-based video editing process into a computable task, thereby paving the way for its automation.
>
---
#### [new 049] Leveraging Large Vision Model for Multi-UAV Co-perception in Low-Altitude Wireless Networks
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于多无人机协同感知任务，旨在解决低空无线网络中通信延迟和资源效率问题。通过引入稀疏传输和深度强化学习，提升感知性能并降低通信开销。**

- **链接: [https://arxiv.org/pdf/2603.16927](https://arxiv.org/pdf/2603.16927)**

> **作者:** Yunting Xu; Jiacheng Wang; Ruichen Zhang; Changyuan Zhao; Yinqiu Liu; Dusit Niyato; Liang Yu; Haibo Zhou; Dong In Kim
>
> **摘要:** Multi-uncrewed aerial vehicle (UAV) cooperative perception has emerged as a promising paradigm for diverse low-altitude economy applications, where complementary multi-view observations are leveraged to enhance perception performance via wireless communications. However, the massive visual data generated by multiple UAVs poses significant challenges in terms of communication latency and resource efficiency. To address these challenges, this paper proposes a communication-efficient cooperative perception framework, termed Base-Station-Helped UAV (BHU), which reduces communication overhead while enhancing perception performance. Specifically, we employ a Top-K selection mechanism to identify the most informative pixels from UAV-captured RGB images, enabling sparsified visual transmission with reduced data volume and latency. The sparsified images are transmitted to a ground server via multi-user MIMO (MU-MIMO), where a Swin-large-based MaskDINO encoder extracts bird's-eye-view (BEV) features and performs cooperative feature fusion for ground vehicle perception. Furthermore, we develop a diffusion model-based deep reinforcement learning (DRL) algorithm to jointly select cooperative UAVs, sparsification ratios, and precoding matrices, achieving a balance between communication efficiency and perception utility. Simulation results on the Air-Co-Pred dataset demonstrate that, compared with traditional CNN-based BEV fusion baselines, the proposed BHU framework improves perception performance by over 5% while reducing communication overhead by 85%, providing an effective solution for multi-UAV cooperative perception under resource-constrained wireless environments.
>
---
#### [new 050] M2P: Improving Visual Foundation Models with Mask-to-Point Weakly-Supervised Learning for Dense Point Tracking
- **分类: cs.CV**

- **简介: 该论文针对视频点跟踪任务，提出M2P方法，通过弱监督学习提升视觉基础模型的密集点追踪性能。**

- **链接: [https://arxiv.org/pdf/2603.17813](https://arxiv.org/pdf/2603.17813)**

> **作者:** Qiangqiang Wu; Tianyu Yang; Bo Fang; Jia Wan; Matias Di Martino; Guillermo Sapiro; Antoni B. Chan
>
> **摘要:** Tracking Any Point (TAP) has emerged as a fundamental tool for video understanding. Current approaches adapt Vision Foundation Models (VFMs) like DINOv2 via offline finetuning or test-time optimization. However, these VFMs rely on static image pre-training, which is inherently sub-optimal for capturing dense temporal correspondence in videos. To address this, we propose Mask-to-Point (M2P) learning, which leverages rich video object segmentation (VOS) mask annotations to improve VFMs for dense point tracking. Our M2P introduces three new mask-based constraints for weakly-supervised representation learning. First, we propose a local structure consistency loss, which leverages Procrustes analysis to model the cohesive motion of points lying within a local structure, achieving more reliable point-to-point matching learning. Second, we propose a mask label consistency (MLC) loss, which enforces that sampled foreground points strictly match foreground regions across frames. The proposed MLC loss can be regarded as a regularization, which stabilizes training and prevents convergence to trivial solutions. Finally, mask boundary constrain is applied to explicitly supervise boundary points. We show that our weaklysupervised M2P models significantly outperform baseline VFMs with efficient training by using only 3.6K VOS training videos. Notably, M2P achieves 12.8% and 14.6% performance gains over DINOv2-B/14 and DINOv3-B/16 on the TAP-Vid-DAVIS benchmark, respectively. Moreover, the proposed M2P models are used as pre-trained backbones for both test-time optimized and offline fine-tuned TAP tasks, demonstrating its potential to serve as general pre-trained models for point tracking. Code will be made publicly available upon acceptance.
>
---
#### [new 051] A Creative Agent is Worth a 64-Token Template
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决T2I模型创造力不足的问题。通过引入CAT框架，利用创意分词器生成可复用的创意模板，提升生成图像的创造性与效率。**

- **链接: [https://arxiv.org/pdf/2603.17895](https://arxiv.org/pdf/2603.17895)**

> **作者:** Ruixiao Shi; Fu Feng; Yucheng Xie; Xu Yang; Jing Wang; Xin Geng
>
> **摘要:** Text-to-image (T2I) models have substantially improved image fidelity and prompt adherence, yet their creativity remains constrained by reliance on discrete natural language prompts. When presented with fuzzy prompts such as ``a creative vinyl record-inspired skyscraper'', these models often fail to infer the underlying creative intent, leaving creative ideation and prompt design largely to human users. Recent reasoning- or agent-driven approaches iteratively augment prompts but incur high computational and monetary costs, as their instance-specific generation makes ``creativity'' costly and non-reusable, requiring repeated queries or reasoning for subsequent generations. To address this, we introduce \textbf{CAT}, a framework for \textbf{C}reative \textbf{A}gent \textbf{T}okenization that encapsulates agents' intrinsic understanding of ``creativity'' through a \textit{Creative Tokenizer}. Given the embeddings of fuzzy prompts, the tokenizer generates a reusable token template that can be directly concatenated with them to inject creative semantics into T2I models without repeated reasoning or prompt augmentation. To enable this, the tokenizer is trained via creative semantic disentanglement, leveraging relations among partially overlapping concept pairs to capture the agent's latent creative representations. Extensive experiments on \textbf{\textit{Architecture Design}}, \textbf{\textit{Furniture Design}}, and \textbf{\textit{Nature Mixture}} tasks demonstrate that CAT provides a scalable and effective paradigm for enhancing creativity in T2I generation, achieving a $3.7\times$ speedup and a $4.8\times$ reduction in computational cost, while producing images with superior human preference and text-image alignment compared to state-of-the-art T2I models and creative generation methods.
>
---
#### [new 052] CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image
- **分类: cs.CV**

- **简介: 该论文属于单视角多人3D重建任务，旨在解决多人群体场景中因遮挡、模糊和多样外观导致的重建难题。提出CrowdGaussian框架，直接从单图重建高质量3D高斯表示。**

- **链接: [https://arxiv.org/pdf/2603.17779](https://arxiv.org/pdf/2603.17779)**

> **作者:** Yizheng Song; Yiyu Zhuang; Qipeng Xu; Haixiang Wang; Jiahe Zhu; Jing Tian; Siyu Zhu; Hao Zhu
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Single-view 3D human reconstruction has garnered significant attention in recent years. Despite numerous advancements, prior research has concentrated on reconstructing 3D models from clear, close-up images of individual subjects, often yielding subpar results in the more prevalent multi-person scenarios. Reconstructing 3D human crowd models is a highly intricate task, laden with challenges such as: 1) extensive occlusions, 2) low clarity, and 3) numerous and various appearances. To address this task, we propose CrowdGaussian, a unified framework that directly reconstructs multi-person 3D Gaussian Splatting (3DGS) representations from single-image inputs. To handle occlusions, we devise a self-supervised adaptation pipeline that enables the pretrained large human model to reconstruct complete 3D humans with plausible geometry and appearance from heavily occluded inputs. Furthermore, we introduce Self-Calibrated Learning (SCL). This training strategy enables single-step diffusion models to adaptively refine coarse renderings to optimal quality by blending identity-preserving samples with clean/corrupted image pairs. The outputs can be distilled back to enhance the quality of multi-person 3DGS representations. Extensive experiments demonstrate that CrowdGaussian generates photorealistic, geometrically coherent reconstructions of multi-person scenes.
>
---
#### [new 053] Generalist Multimodal LLMs Gain Biometric Expertise via Human Salience
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于生物特征安全任务，解决虹膜防攻击检测问题。通过结合通用多模态大模型与人类专家知识，在隐私约束下实现高效检测。**

- **链接: [https://arxiv.org/pdf/2603.17173](https://arxiv.org/pdf/2603.17173)**

> **作者:** Jacob Piland; Byron Dowling; Christopher Sweet; Adam Czajka
>
> **摘要:** Iris presentation attack detection (PAD) is critical for secure biometric deployments, yet developing specialized models faces significant practical barriers: collecting data representing future unknown attacks is impossible, and collecting diverse-enough data, yet still limited in terms of its predictive power, is expensive. Additionally, sharing biometric data raises privacy concerns. Due to rapid emergence of new attack vectors demanding adaptable solutions, we thus investigate in this paper whether general-purpose multimodal large language models (MLLMs) can perform iris PAD when augmented with human expert knowledge, operating under strict privacy constraints that prohibit sending biometric data to public cloud MLLM services. Through analysis of vision encoder embeddings applied to our dataset, we demonstrate that pre-trained vision transformers in MLLMs inherently cluster many iris attack types despite never being explicitly trained for this task. However, where clustering shows overlap between attack classes, we find that structured prompts incorporating human salience (verbal descriptions from subjects identifying attack indicators) enable these models to resolve ambiguities. Testing on an IRB-restricted dataset of 224 iris images spanning seven attack types, using only university-approved services (Gemini 2.5 Pro) or locally-hosted models (e.g., Llama 3.2-Vision), we show that Gemini with expert-informed prompts outperforms both a specialized convolutional neural networks (CNN)-based baseline and human examiners, while the locally-deployable Llama achieves near-human performance. Our results establish that MLLMs deployable within institutional privacy constraints offer a viable path for iris PAD.
>
---
#### [new 054] Understanding and Defending VLM Jailbreaks via Jailbreak-Related Representation Shift
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于安全防护任务，旨在解决VLM在视觉输入下易被越狱的问题。通过分析表示空间变化，提出移除越狱相关偏移的防御方法。**

- **链接: [https://arxiv.org/pdf/2603.17372](https://arxiv.org/pdf/2603.17372)**

> **作者:** Zhihua Wei; Qiang Li; Jian Ruan; Zhenxin Qin; Leilei Wen; Dongrui Liu; Wen Shen
>
> **摘要:** Large vision-language models (VLMs) often exhibit weakened safety alignment with the integration of the visual modality. Even when text prompts contain explicit harmful intent, adding an image can substantially increase jailbreak success rates. In this paper, we observe that VLMs can clearly distinguish benign inputs from harmful ones in their representation space. Moreover, even among harmful inputs, jailbreak samples form a distinct internal state that is separable from refusal samples. These observations suggest that jailbreaks do not arise from a failure to recognize harmful intent. Instead, the visual modality shifts representations toward a specific jailbreak state, thereby leading to a failure to trigger refusal. To quantify this transition, we identify a jailbreak direction and define the jailbreak-related shift as the component of the image-induced representation shift along this direction. Our analysis shows that the jailbreak-related shift reliably characterizes jailbreak behavior, providing a unified explanation for diverse jailbreak scenarios. Finally, we propose a defense method that enhances VLM safety by removing the jailbreak-related shift (JRS-Rem) at inference time. Experiments show that JRS-Rem provides strong defense across multiple scenarios while preserving performance on benign tasks.
>
---
#### [new 055] AR-CoPO: Align Autoregressive Video Generation with Contrastive Policy Optimization
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决AR视频生成与RLHF对齐难题。提出AR-CoPO框架，通过分块对齐和半策略训练提升生成质量与对齐效果。**

- **链接: [https://arxiv.org/pdf/2603.17461](https://arxiv.org/pdf/2603.17461)**

> **作者:** Dailan He; Guanlin Feng; Xingtong Ge; Yi Zhang; Bingqi Ma; Guanglu Song; Yu Liu; Hongsheng Li
>
> **摘要:** Streaming autoregressive (AR) video generators combined with few-step distillation achieve low-latency, high-quality synthesis, yet remain difficult to align via reinforcement learning from human feedback (RLHF). Existing SDE-based GRPO methods face challenges in this setting: few-step ODEs and consistency model samplers deviate from standard flow-matching ODEs, and their short, low-stochasticity trajectories are highly sensitive to initialization noise, rendering intermediate SDE exploration ineffective. We propose AR-CoPO (AutoRegressive Contrastive Policy Optimization), a framework that adapts the Neighbor GRPO contrastive perspective to streaming AR generation. AR-CoPO introduces chunk-level alignment via a forking mechanism that constructs neighborhood candidates at a randomly selected chunk, assigns sequence-level rewards, and performs localized GRPO updates. We further propose a semi-on-policy training strategy that complements on-policy exploration with exploitation over a replay buffer of reference rollouts, improving generation quality across domains. Experiments on Self-Forcing demonstrate that AR-CoPO improves both out-of-domain generalization and in-domain human preference alignment over the baseline, providing evidence of genuine alignment rather than reward hacking.
>
---
#### [new 056] MosaicMem: Hybrid Spatial Memory for Controllable Video World Models
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决视频世界模型中的空间记忆问题。提出MosaicMem混合空间记忆方法，提升一致性与动态建模能力。**

- **链接: [https://arxiv.org/pdf/2603.17117](https://arxiv.org/pdf/2603.17117)**

> **作者:** Wei Yu; Runjia Qian; Yumeng Li; Liquan Wang; Songheng Yin; Sri Siddarth Chakaravarthy P; Dennis Anthony; Yang Ye; Yidi Li; Weiwei Wan; Animesh Garg
>
> **备注:** Project Page: this https URL
>
> **摘要:** Video diffusion models are moving beyond short, plausible clips toward world simulators that must remain consistent under camera motion, revisits, and intervention. Yet spatial memory remains a key bottleneck: explicit 3D structures can improve reprojection-based consistency but struggle to depict moving objects, while implicit memory often produces inaccurate camera motion even with correct poses. We propose Mosaic Memory (MosaicMem), a hybrid spatial memory that lifts patches into 3D for reliable localization and targeted retrieval, while exploiting the model's native conditioning to preserve prompt-following generation. MosaicMem composes spatially aligned patches in the queried view via a patch-and-compose interface, preserving what should persist while allowing the model to inpaint what should evolve. With PRoPE camera conditioning and two new memory alignment methods, experiments show improved pose adherence compared to implicit memory and stronger dynamic modeling than explicit baselines. MosaicMem further enables minute-level navigation, memory-based scene editing, and autoregressive rollout.
>
---
#### [new 057] DeepCORO-CLIP: A Multi-View Foundation Model for Comprehensive Coronary Angiography Video-Text Analysis and External Validation
- **分类: cs.CV**

- **简介: 该论文提出DeepCORO-CLIP，用于冠状动脉造影视频与文本的综合分析，解决诊断、预后和疾病进展评估问题，通过多视角学习提升准确性。**

- **链接: [https://arxiv.org/pdf/2603.17675](https://arxiv.org/pdf/2603.17675)**

> **作者:** Sarra Harrabi; Yichen Wu; Geoffrey H. Tison; Minhaj Ansari; Milos Vukadinovic; David Ouyang; Joshua P. Barrios; Jacques Delfrate; Robert Avram
>
> **备注:** 69 pages, 5 figures
>
> **摘要:** Coronary angiography is the reference standard for evaluating coronary artery disease, yet visual interpretation remains variable between readers. Existing artificial intelligence methods typically analyze single frames or projections and focus mainly on stenosis, limiting comprehensive coronary assessment. We present DeepCORO-CLIP, a multi-view foundation model trained with video-text contrastive learning on 203,808 angiography videos from 28,117 patients across 32,473 studies at the Montreal Heart Institute and externally validated on 4,249 studies from the University of California, San Francisco. DeepCORO-CLIP integrates multiple projections with attention-based pooling for study-level assessment across diagnostic, prognostic, and disease progression tasks. For significant stenosis detection, the model achieved an AUROC of 0.888 internally and 0.89 on external validation. Mean absolute error against core laboratory quantitative coronary angiography was 13.6%, lower than clinical reports at 19.0%. The model also performed strongly for chronic total occlusion, intracoronary thrombus, and coronary calcification detection. Transfer learning enabled prediction of one-year major adverse cardiovascular events with AUROC 0.79 and estimation of left ventricular ejection fraction with mean absolute error 7.3%. Embeddings also captured disease progression across serial examinations. With a mean inference time of 4.2 seconds in hospital deployment, DeepCORO-CLIP provides a foundation for automated coronary angiography interpretation at the point of care. Code, sample data, model weights, and deployment infrastructure are publicly released.
>
---
#### [new 058] UniSAFE: A Comprehensive Benchmark for Safety Evaluation of Unified Multimodal Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态模型安全评估任务，旨在解决现有基准碎片化问题。提出UniSAFE基准，涵盖7种模态组合，评估15个先进模型的安全性，揭示多图像生成等场景中的安全漏洞。**

- **链接: [https://arxiv.org/pdf/2603.17476](https://arxiv.org/pdf/2603.17476)**

> **作者:** Segyu Lee; Boryeong Cho; Hojung Jung; Seokhyun An; Juhyeong Kim; Jaehyun Kwak; Yongjin Yang; Sangwon Jang; Youngrok Park; Wonjun Chang; Se-Young Yun
>
> **备注:** Equal contribution by first three authors, 55 pages
>
> **摘要:** Unified Multimodal Models (UMMs) offer powerful cross-modality capabilities but introduce new safety risks not observed in single-task models. Despite their emergence, existing safety benchmarks remain fragmented across tasks and modalities, limiting the comprehensive evaluation of complex system-level vulnerabilities. To address this gap, we introduce UniSAFE, the first comprehensive benchmark for system-level safety evaluation of UMMs across 7 I/O modality combinations, spanning conventional tasks and novel multimodal-context image generation settings. UniSAFE is built with a shared-target design that projects common risk scenarios across task-specific I/O configurations, enabling controlled cross-task comparisons of safety failures. Comprising 6,802 curated instances, we use UniSAFE to evaluate 15 state-of-the-art UMMs, both proprietary and open-source. Our results reveal critical vulnerabilities across current UMMs, including elevated safety violations in multi-image composition and multi-turn settings, with image-output tasks consistently more vulnerable than text-output tasks. These findings highlight the need for stronger system-level safety alignment for UMMs. Our code and data are publicly available at this https URL
>
---
#### [new 059] SpiderCam: Low-Power Snapshot Depth from Differential Defocus
- **分类: cs.CV**

- **简介: 该论文提出SpiderCam，一种低功耗实时深度感知系统，解决传统3D相机功耗高的问题，通过FPGA实现差分失焦算法，生成稀疏深度图。**

- **链接: [https://arxiv.org/pdf/2603.17910](https://arxiv.org/pdf/2603.17910)**

> **作者:** Marcos A. Ferreira; Tianao Li; John Mamish; Josiah Hester; Yaman Sangar; Qi Guo; Emma Alexander
>
> **备注:** Accepted to IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026
>
> **摘要:** We introduce SpiderCam, an FPGA-based snapshot depth-from-defocus camera which produces 480x400 sparse depth maps in real-time at 32.5 FPS over a working range of 52 cm while consuming 624 mW of power in total. SpiderCam comprises a custom camera that simultaneously captures two differently focused images of the same scene, processed with a SystemVerilog implementation of depth from differential defocus (DfDD) on a low-power FPGA. To achieve state-of-the-art power consumption, we present algorithmic improvements to DfDD that overcome challenges caused by low-power sensors, and design a memory-local implementation for streaming depth computation on a device that is too small to store even a single image pair. We report the first sub-Watt total power measurement for passive FPGA-based 3D cameras in the literature.
>
---
#### [new 060] Feeling the Space: Egomotion-Aware Video Representation for Efficient and Accurate 3D Scene Understanding
- **分类: cs.CV**

- **简介: 该论文属于3D场景理解任务，旨在解决MLLMs在空间推理中依赖高成本3D表示或缺乏物理依据的问题。通过引入IMU数据，提出Motion-MLLM框架，提升准确性和效率。**

- **链接: [https://arxiv.org/pdf/2603.17980](https://arxiv.org/pdf/2603.17980)**

> **作者:** Shuyao Shi; Kang G. Shin
>
> **摘要:** Recent Multimodal Large Language Models (MLLMs) have shown high potential for spatial reasoning within 3D scenes. However, they typically rely on computationally expensive 3D representations like point clouds or reconstructed Bird's-Eye View (BEV) maps, or lack physical grounding to resolve ambiguities in scale and size. This paper significantly enhances MLLMs with egomotion modality data, captured by Inertial Measurement Units (IMUs) concurrently with the video. In particular, we propose a novel framework, called Motion-MLLM, introducing two key components: (1) a cascaded motion-visual keyframe filtering module that leverages both IMU data and visual features to efficiently select a sparse yet representative set of keyframes, and (2) an asymmetric cross-modal fusion module where motion tokens serve as intermediaries that channel egomotion cues and cross-frame visual context into the visual representation. By grounding visual content in physical egomotion trajectories, Motion-MLLM can reason about absolute scale and spatial relationships across the scene. Our extensive evaluation shows that Motion-MLLM makes significant improvements in various tasks related to 3D scene understanding and spatial reasoning. Compared to state-of-the-art (SOTA) methods based on video frames and explicit 3D data, Motion-MLLM exhibits similar or even higher accuracy with significantly less overhead (i.e., 1.40$\times$ and 1.63$\times$ higher cost-effectiveness, respectively).
>
---
#### [new 061] Few-Step Diffusion Sampling Through Instance-Aware Discretizations
- **分类: cs.CV**

- **简介: 该论文属于生成模型任务，旨在解决扩散模型采样效率问题。针对全局时间步调度的不足，提出基于输入的实例感知离散化方法，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.17671](https://arxiv.org/pdf/2603.17671)**

> **作者:** Liangyu Yuan; Ruoyu Wang; Tong Zhao; Dingwen Fu; Mingkun Lei; Beier Zhu; Chi Zhang
>
> **备注:** 24 pages, 20 figures. code: this https URL
>
> **摘要:** Diffusion and flow matching models generate high-fidelity data by simulating paths defined by Ordinary or Stochastic Differential Equations (ODEs/SDEs), starting from a tractable prior distribution. The probability flow ODE formulation enables the use of advanced numerical solvers to accelerate sampling. Orthogonal yet vital to solver design is the discretization strategy. While early approaches employed handcrafted heuristics and recent methods adopt optimization-based techniques, most existing strategies enforce a globally shared timestep schedule across all samples. This uniform treatment fails to account for instance-specific complexity in the generative process, potentially limiting performance. Motivated by controlled experiments on synthetic data, which reveals the suboptimality of global schedules under instance-specific dynamics, we propose an instance-aware discretization framework. Our method learns to adapt timestep allocations based on input-dependent priors, extending gradient-based discretization search to the conditional generative setting. Empirical results across diverse settings, including synthetic data, pixel-space diffusion, latent-space images and video flow matching models, demonstrate that our method consistently improves generation quality with marginal tuning cost compared to training and negligible inference overhead.
>
---
#### [new 062] Accurate Shift Invariant Convolutional Neural Networks Using Gaussian-Hermite Moments
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，解决CNN缺乏平移不变性的问题。提出GHS方法，在不修改架构的情况下实现层级平移不变性，提升分类一致性与准确率。**

- **链接: [https://arxiv.org/pdf/2603.17098](https://arxiv.org/pdf/2603.17098)**

> **作者:** Jaspreet Singh; Petra Bosilj; Grzegorz Cielniak
>
> **摘要:** The convolutional neural networks (CNNs) are not inherently shift invariant or equivariant. The downsampling operation, used in CNNs, is one of the key reasons which breaks the shift invariant property of a CNN. Conversely, downsampling operation is important to improve computational efficiency and increase the area of the receptive field for more contextual information. In this work, we propose Gaussian-Hermite Sampling (GHS), a novel downsampling strategy designed to achieve accurate shift invariance. GHS leverages Gaussian-Hermite polynomials to perform shift-consistent sampling, enabling CNN layers to maintain invariance to arbitrary spatial shifts prior to training. When integrated into standard CNN architectures, the proposed method embeds shift invariance directly at the layer level without requiring architectural modifications or additional training procedures. We evaluate the proposed approach on CIFAR-10, CIFAR-100, and MNIST-rot datasets. Experimental results demonstrate that GHS significantly improves shift consistency, achieving 100% classification consistency under spatial shifts, while also improving classification accuracy compared to baseline CNN models.
>
---
#### [new 063] The Unreasonable Effectiveness of Text Embedding Interpolation for Continuous Image Steering
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，解决文本条件生成模型的连续可控编辑问题。通过文本嵌入空间插值实现无需训练的平滑编辑控制。**

- **链接: [https://arxiv.org/pdf/2603.17998](https://arxiv.org/pdf/2603.17998)**

> **作者:** Yigit Ekin; Yossi Gandelsman
>
> **备注:** Project Page: this https URL
>
> **摘要:** We present a training-free framework for continuous and controllable image editing at test time for text-conditioned generative models. In contrast to prior approaches that rely on additional training or manual user intervention, we find that a simple steering in the text-embedding space is sufficient to produce smooth edit control. Given a target concept (e.g., enhancing photorealism or changing facial expression), we use a large language model to automatically construct a small set of debiased contrastive prompt pairs, from which we compute a steering vector in the generator's text-encoder space. We then add this vector directly to the input prompt representation to control generation along the desired semantic axis. To obtain a continuous control, we propose an elastic range search procedure that automatically identifies an effective interval of steering magnitudes, avoiding both under-steering (no-edit) and over-steering (changing other attributes). Adding the scaled versions of the same vector within this interval yields smooth and continuous edits. Since our method modifies only textual representations, it naturally generalizes across text-conditioned modalities, including image and video generation. To quantify the steering continuity, we introduce a new evaluation metric that measures the uniformity of semantic change across edit strengths. We compare the continuous editing behavior across methods and find that, despite its simplicity and lightweight design, our approach is comparable to training-based alternatives, outperforming other training-free methods.
>
---
#### [new 064] EI: Early Intervention for Multimodal Imaging based Disease Recognition
- **分类: cs.CV**

- **简介: 该论文属于多模态医学图像疾病识别任务，解决数据互补性不足和标注稀缺的问题。提出EI框架和MoR方法，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.17514](https://arxiv.org/pdf/2603.17514)**

> **作者:** Qijie Wei; Hailan Lin; Xirong Li
>
> **备注:** Accepted to CVPR 2026 Findings
>
> **摘要:** Current methods for multimodal medical imaging based disease recognition face two major challenges. First, the prevailing "fusion after unimodal image embedding" paradigm cannot fully leverage the complementary and correlated information in the multimodal data. Second, the scarcity of labeled multimodal medical images, coupled with their significant domain shift from natural images, hinders the use of cutting-edge Vision Foundation Models (VFMs) for medical image embedding. To jointly address the challenges, we propose a novel Early Intervention (EI) framework. Treating one modality as target and the rest as reference, EI harnesses high-level semantic tokens from the reference as intervention tokens to steer the target modality's embedding process at an early stage. Furthermore, we introduce Mixture of Low-varied-Ranks Adaptation (MoR), a parameter-efficient fine-tuning method that employs a set of low-rank adapters with varied ranks and a weight-relaxed router for VFM adaptation. Extensive experiments on three public datasets for retinal disease, skin lesion, and keen anomaly classification verify the effectiveness of the proposed method against a number of competitive baselines.
>
---
#### [new 065] Edit Spillover as a Probe: Do Image Editing Models Implicitly Understand World Relations?
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，研究模型在编辑时产生的语义扩散现象。通过构建基准数据集和分析框架，探讨模型是否具备隐含的世界理解能力。**

- **链接: [https://arxiv.org/pdf/2603.17876](https://arxiv.org/pdf/2603.17876)**

> **作者:** Guandong Li; Zhaobin Chu
>
> **摘要:** Instruction-following image editing models are expected to modify only the specified region while keeping the rest of the image unchanged. However, in practice, we observe a pervasive phenomenon -- edit spillover: models alter semantically related but unspecified content outside the edit region. This raises a fundamental question -- does spillover reflect genuine implicit world understanding, or is it merely attention leakage? We propose EditSpilloverProbe, a systematic framework that repurposes edit spillover as a natural probe for world knowledge in image editing models. We introduce a spillover taxonomy (spatial, semantic, mixed, random), an automated detection-and-classification pipeline, and a benchmark dataset constructed from real-world Chinese text editing tasks, EditSpilloverBench. Systematic evaluation of 5 representative editing models reveals three core findings: (1) spillover rates vary dramatically across architectures, from 3.49% to 11.46%, with a 3.3x ratio; (2) absolute semantic spillover quantity reveals models' world understanding capability -- nano_banana produces the most semantic spillover (27.8 per image), while qwen_2511 has the most precise editing control but lower semantic spillover (16.3 per image), revealing a trade-off between editing control and world understanding; (3) spatial decay analysis shows spillover area density decays exponentially with distance, but the proportion of semantically relevant spillover remains constant (40%-58%), providing direct evidence that semantic spillover reflects genuine world understanding rather than spatial diffusion.
>
---
#### [new 066] SA-CycleGAN-2.5D: Self-Attention CycleGAN with Tri-Planar Context for Multi-Site MRI Harmonization
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出SA-CycleGAN-2.5D，用于多中心MRI图像谐波化，解决扫描仪差异导致的分布偏移问题，通过全局注意力和三平面结构提升图像一致性。**

- **链接: [https://arxiv.org/pdf/2603.17219](https://arxiv.org/pdf/2603.17219)**

> **作者:** Ishrith Gowda; Chunwei Liu
>
> **备注:** 12 pages, 5 figures, 5 tables. Submitted to MICCAI 2026
>
> **摘要:** Multi-site neuroimaging analysis is fundamentally confounded by scanner-induced covariate shifts, where the marginal distribution of voxel intensities $P(\mathbf{x})$ varies non-linearly across acquisition protocols while the conditional anatomy $P(\mathbf{y}|\mathbf{x})$ remains constant. This is particularly detrimental to radiomic reproducibility, where acquisition variance often exceeds biological pathology variance. Existing statistical harmonization methods (e.g., ComBat) operate in feature space, precluding spatial downstream tasks, while standard deep learning approaches are theoretically bounded by local effective receptive fields (ERF), failing to model the global intensity correlations characteristic of field-strength bias. We propose SA-CycleGAN-2.5D, a domain adaptation framework motivated by the $H\Delta H$-divergence bound of Ben-David et al., integrating three architectural innovations: (1) A 2.5D tri-planar manifold injection preserving through-plane gradients $\nabla_z$ at $O(HW)$ complexity; (2) A U-ResNet generator with dense voxel-to-voxel self-attention, surpassing the $O(\sqrt{L})$ receptive field limit of CNNs to model global scanner field biases; and (3) A spectrally-normalized discriminator constraining the Lipschitz constant ($K_D \le 1$) for stable adversarial optimization. Evaluated on 654 glioma patients across two institutional domains (BraTS and UPenn-GBM), our method reduces Maximum Mean Discrepancy (MMD) by 99.1% ($1.729 \to 0.015$) and degrades domain classifier accuracy to near-chance (59.7%). Ablation confirms that global attention is statistically essential (Cohen's $d = 1.32$, $p < 0.001$) for the harder heterogeneous-to-homogeneous translation direction. By bridging 2D efficiency and 3D consistency, our framework yields voxel-level harmonized images that preserve tumor pathophysiology, enabling reproducible multi-center radiomic analysis.
>
---
#### [new 067] Learning Coordinate-based Convolutional Kernels for Continuous SE(3) Equivariant and Efficient Point Cloud Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于点云分析任务，解决SE(3)等变性与可扩展性的矛盾。提出ECKConv，通过坐标基核实现高效等变特征提取。**

- **链接: [https://arxiv.org/pdf/2603.17538](https://arxiv.org/pdf/2603.17538)**

> **作者:** Jaein Kim; Hee Bin Yoo; Dong-Sig Han; Byoung-Tak Zhang
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** A symmetry on rigid motion is one of the salient factors in efficient learning of 3D point cloud problems. Group convolution has been a representative method to extract equivariant features, but its realizations have struggled to retain both rigorous symmetry and scalability simultaneously. We advocate utilizing the intertwiner framework to resolve this trade-off, but previous works on it, which did not achieve complete SE(3) symmetry or scalability to large-scale problems, necessitate a more advanced kernel architecture. We present Equivariant Coordinate-based Kernel Convolution, or ECKConv. It acquires SE(3) equivariance from the kernel domain defined in a double coset space, and its explicit kernel design using coordinate-based networks enhances its learning capability and memory efficiency. The experiments on diverse point cloud tasks, e.g., classification, pose registration, part segmentation, and large-scale semantic segmentation, validate the rigid equivariance, memory scalability, and outstanding performance of ECKConv compared to state-of-the-art equivariant methods.
>
---
#### [new 068] Adaptive Anchor Policies for Efficient 4D Gaussian Streaming
- **分类: cs.CV**

- **简介: 该论文属于4D高斯流媒体任务，解决固定锚点选择导致计算浪费的问题，提出EGS方法通过强化学习动态选择锚点，提升效率与质量平衡。**

- **链接: [https://arxiv.org/pdf/2603.17227](https://arxiv.org/pdf/2603.17227)**

> **作者:** Ashim Dahal; Rabab Abdelfattah; Nick Rahimi
>
> **摘要:** Dynamic scene reconstruction with Gaussian Splatting has enabled efficient streaming for real-time rendering and free-viewpoint video. However, most pipelines rely on fixed anchor selection such as Farthest Point Sampling (FPS), typically using 8,192 anchors regardless of scene complexity, which over-allocates computation under strict budgets. We propose Efficient Gaussian Streaming (EGS), a plug-in, budget-aware anchor sampler that replaces FPS with a reinforcement-learned policy while keeping the Gaussian streaming reconstruction backbone unchanged. The policy jointly selects an anchor budget and a subset of informative anchors under discrete constraints, balancing reconstruction quality and runtime using spatial features of the Gaussian representation. We evaluate EGS in two settings: fast rendering, which prioritizes runtime efficiency, and high-quality refinement, which enables additional optimization. Experiments on dynamic multi-view datasets show consistent improvements in the quality--efficiency trade-off over FPS sampling. On unseen data, in fast rendering at 256 anchors ($32\times$ fewer than 8,192), EGS improves PSNR by $+0.52$--$0.61$\,dB while running $1.29$--$1.35\times$ faster than IGS@8192 (N3DV and MeetingRoom). In high-quality refinement, EGS remains competitive with the full-anchor baseline at substantially lower anchor budgets. \emph{Code and pretrained checkpoints will be released upon acceptance.} \keywords{4D Gaussian Splatting \and 4D Gaussian Streaming \and Reinforcement Learning}
>
---
#### [new 069] A practical artificial intelligence framework for legal age estimation using clavicle computed tomography scans
- **分类: cs.CV**

- **简介: 该论文属于法律年龄估计任务，旨在通过胸骨CT扫描准确评估年龄。提出一种多阶段AI框架，结合自动检测、切片选择和不确定性量化，提升估计精度与可解释性。**

- **链接: [https://arxiv.org/pdf/2603.17926](https://arxiv.org/pdf/2603.17926)**

> **作者:** Javier Venema; Stefano De Luca; Pablo Mesejo; Óscar Ibáñez
>
> **备注:** 15 pages, 8 figures, submitted to Engineering Applications of Artificial Intelligence
>
> **摘要:** Legal age estimation plays a critical role in forensic and medico-legal contexts, where decisions must be supported by accurate, robust, and reproducible methods with explicit uncertainty quantification. While prior artificial intelligence (AI)-based approaches have primarily focused on hand radiographs or dental imaging, clavicle computed tomography (CT) scans remain underexplored despite their documented effectiveness for legal age estimation. In this work, we present an interpretable, multi-stage pipeline for legal age estimation from clavicle CT scans. The proposed framework combines (i) a feature-based connected-component method for automatic clavicle detection that requires minimal manual annotation, (ii) an Integrated Gradients-guided slice selection strategy used to construct the input data for a multi-slice convolutional neural network that estimates legal age, and (iii) conformal prediction intervals to support uncertainty-aware decisions in accordance with established international protocols. The pipeline is evaluated on 1,158 full-body post-mortem CT scans from a public forensic dataset (the New Mexico Decedent Image Database). The final model achieves state-of-the-art performance with a mean absolute error (MAE) of 1.55 $\pm$ 0.16 years on a held-out test set, outperforming both human experts (MAE of approximately 1.90 years) and previous methods (MAEs above 1.75 years in our same dataset). Furthermore, conformal prediction enables configurable coverage levels aligned with forensic requirements. Attribution maps indicate that the model focuses on anatomically relevant regions of the medial clavicular epiphysis. The proposed method, which is currently being added as part of the Skeleton-ID software (this https URL), is intended as a decision-support component within multi-factorial forensic workflows.
>
---
#### [new 070] AdaZoom-GUI: Adaptive Zoom-based GUI Grounding with Instruction Refinement
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于GUI接地任务，解决高分辨率界面中元素定位与指令理解难题。提出AdaZoom-GUI框架，结合指令优化和条件缩放策略提升定位精度。**

- **链接: [https://arxiv.org/pdf/2603.17441](https://arxiv.org/pdf/2603.17441)**

> **作者:** Siqi Pei; Liang Tang; Tiaonan Duan; Long Chen; Shuxian Li; Kaer Huang; Yanzhe Jing; Yiqiang Yan; Bo Zhang; Chenghao Jiang; Borui Zhang; Jiwen Lu
>
> **摘要:** GUI grounding is a critical capability for vision-language models (VLMs) that enables automated interaction with graphical user interfaces by locating target elements from natural language instructions. However, grounding on GUI screenshots remains challenging due to high-resolution images, small UI elements, and ambiguous user instructions. In this work, we propose AdaZoom-GUI, an adaptive zoom-based GUI grounding framework that improves both localization accuracy and instruction understanding. Our approach introduces an instruction refinement module that rewrites natural language commands into explicit and detailed descriptions, allowing the grounding model to focus on precise element localization. In addition, we design a conditional zoom-in strategy that selectively performs a second-stage inference on predicted small elements, improving localization accuracy while avoiding unnecessary computation and context loss on simpler cases. To support this framework, we construct a high-quality GUI grounding dataset and train the grounding model using Group Relative Policy Optimization (GRPO), enabling the model to predict both click coordinates and element bounding boxes. Experiments on public benchmarks demonstrate that our method achieves state-of-the-art performance among models with comparable or even larger parameter sizes, highlighting its effectiveness for high-resolution GUI understanding and practical GUI agent deployment.
>
---
#### [new 071] Multi-Modal Multi-Agent Reinforcement Learning for Radiology Report Generation: Radiologist-Like Workflow with Clinically Verifiable Rewards
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学影像报告生成任务，旨在提升报告的临床有效性。通过多智能体强化学习框架MARL-Rad，协同多个区域代理与全局代理，优化临床可验证奖励，提高报告准确性与一致性。**

- **链接: [https://arxiv.org/pdf/2603.16876](https://arxiv.org/pdf/2603.16876)**

> **作者:** Kaito Baba; Satoshi Kodera
>
> **备注:** 17 pages, 3 figures
>
> **摘要:** We propose MARL-Rad, a novel multi-modal multi-agent reinforcement learning framework for radiology report generation that coordinates region-specific agents and a global integrating agent, optimized via clinically verifiable rewards. Unlike prior single-model reinforcement learning or post-hoc agentization of independently trained models, our method jointly trains multiple agents and optimizes the entire agent system through reinforcement learning. Experiments on the MIMIC-CXR and IU X-ray datasets show that MARL-Rad consistently improves clinically efficacy (CE) metrics such as RadGraph, CheXbert, and GREEN scores, achieving state-of-the-art CE performance. Further analyses confirm that MARL-Rad enhances laterality consistency and produces more accurate, detail-informed reports.
>
---
#### [new 072] GigaWorld-Policy: An Efficient Action-Centered World--Action Model
- **分类: cs.CV**

- **简介: 该论文属于机器人策略学习任务，解决世界-动作模型（WAM）的推理效率和运动预测准确性问题。提出GigaWorld-Policy，通过耦合动作预测与视频生成提升性能。**

- **链接: [https://arxiv.org/pdf/2603.17240](https://arxiv.org/pdf/2603.17240)**

> **作者:** Angen Ye; Boyuan Wang; Chaojun Ni; Guan Huang; Guosheng Zhao; Hao Li; Hengtao Li; Jie Li; Jindi Lv; Jingyu Liu; Min Cao; Peng Li; Qiuping Deng; Wenjun Mei; Xiaofeng Wang; Xinze Chen; Xinyu Zhou; Yang Wang; Yifan Chang; Yifan Li; Yukun Zhou; Yun Ye; Zhichao Liu; Zheng Zhu
>
> **摘要:** World-Action Models (WAM) initialized from pre-trained video generation backbones have demonstrated remarkable potential for robot policy learning. However, existing approaches face two critical bottlenecks that hinder performance and deployment. First, jointly reasoning over future visual dynamics and corresponding actions incurs substantial inference overhead. Second, joint modeling often entangles visual and motion representations, making motion prediction accuracy heavily dependent on the quality of future video forecasts. To address these issues, we introduce GigaWorld-Policy, an action-centered WAM that learns 2D pixel-action dynamics while enabling efficient action decoding, with optional video generation. Specifically, we formulate policy training into two coupled components: the model predicts future action sequences conditioned on the current observation, and simultaneously generates future videos conditioned on the predicted actions and the same observation. The policy is supervised by both action prediction and video generation, providing richer learning signals and encouraging physically plausible actions through visual-dynamics constraints. With a causal design that prevents future-video tokens from influencing action tokens, explicit future-video generation is optional at inference time, allowing faster action prediction during deployment. To support this paradigm, we curate a diverse, large-scale robot dataset to pre-train an action-centered video generation model, which is then adapted as the backbone for robot policy learning. Experimental results on real-world robotic platforms show that GigaWorld-Policy runs 9x faster than the leading WAM baseline, Motus, while improving task success rates by 7%. Moreover, compared with pi-0.5, GigaWorld-Policy improves performance by 95% on RoboTwin 2.0.
>
---
#### [new 073] BEV-SLD: Self-Supervised Scene Landmark Detection for Global Localization with LiDAR Bird's-Eye View Images
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出BEV-SLD，用于LiDAR全局定位的自监督场景地标检测方法，解决场景特定地标识别问题，通过BEV图像和一致性损失实现精准定位。**

- **链接: [https://arxiv.org/pdf/2603.17159](https://arxiv.org/pdf/2603.17159)**

> **作者:** David Skuddis; Vincent Ress; Wei Zhang; Vincent Ofosu Nyako; Norbert Haala
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** We present BEV-SLD, a LiDAR global localization method building on the Scene Landmark Detection (SLD) concept. Unlike scene-agnostic pipelines, our self-supervised approach leverages bird's-eye-view (BEV) images to discover scene-specific patterns at a prescribed spatial density and treat them as landmarks. A consistency loss aligns learnable global landmark coordinates with per-frame heatmaps, yielding consistent landmark detections across the scene. Across campus, industrial, and forest environments, BEV-SLD delivers robust localization and achieves strong performance compared to state-of-the-art methods.
>
---
#### [new 074] Directing the Narrative: A Finetuning Method for Controlling Coherence and Style in Story Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于故事生成任务，旨在解决叙事一致性与风格统一性问题。提出两阶段框架，结合GSA和DPO，提升角色身份与视觉风格的一致性。**

- **链接: [https://arxiv.org/pdf/2603.17295](https://arxiv.org/pdf/2603.17295)**

> **作者:** Jianzhang Zhang; Yijing Tian; Jiwang Qu; Chuang Liu
>
> **摘要:** Story visualization requires generating sequential imagery that aligns semantically with evolving narratives while maintaining rigorous consistency in character identity and visual style. However, existing methodologies often struggle with subject inconsistency and identity drift, particularly when depicting complex interactions or extended narrative arcs. To address these challenges, we propose a cohesive two-stage framework designed for robust and consistent story generation. First, we introduce Group-Shared Attention (GSA), a mechanism that fosters intrinsic consistency by enabling lossless cross-sample information flow within attention layers. This allows the model to structurally encode identity correspondence across frames without relying on external encoders. Second, we leverage Direct Preference Optimization (DPO) to align generated outputs with human aesthetic and narrative standards. Unlike conventional methods that rely on conflicting auxiliary losses, our approach simultaneously enhances visual fidelity and identity preservation by learning from holistic preference data. Extensive evaluations on the ViStoryBench benchmark demonstrate that our method establishes a new state-of-the-art, significantly outperforming strong baselines with gains of +10.0 in Character Identity (CIDS) and +18.7 in Style Consistency (CSD), all while preserving high-fidelity generation.
>
---
#### [new 075] SegFly: A 2D-3D-2D Paradigm for Aerial RGB-Thermal Semantic Segmentation at Scale
- **分类: cs.CV**

- **简介: 该论文属于遥感图像语义分割任务，解决RGB-T数据集规模小、标注效率低的问题。提出2D-3D-2D框架，通过几何驱动实现自动标注与对齐，构建大规模SegFly数据集。**

- **链接: [https://arxiv.org/pdf/2603.17920](https://arxiv.org/pdf/2603.17920)**

> **作者:** Markus Gross; Sai Bharadhwaj Matha; Rui Song; Viswanathan Muthuveerappan; Conrad Christoph; Julius Huber; Daniel Cremers
>
> **摘要:** Semantic segmentation for uncrewed aerial vehicles (UAVs) is fundamental for aerial scene understanding, yet existing RGB and RGB-T datasets remain limited in scale, diversity, and annotation efficiency due to the high cost of manual labeling and the difficulties of accurate RGB-T alignment on off-the-shelf UAVs. To address these challenges, we propose a scalable geometry-driven 2D-3D-2D paradigm that leverages multi-view redundancy in high-overlap aerial imagery to automatically propagate labels from a small subset of manually annotated RGB images to both RGB and thermal modalities within a unified framework. By lifting less than 3% of RGB images into a semantic 3D point cloud and reprojecting it into all views, our approach enables dense pseudo ground-truth generation across large image collections, automatically producing 97% of RGB labels and 100% of thermal labels while achieving 91% and 88% annotation accuracy without any 2D manual refinement. We further extend this 2D-3D-2D paradigm to cross-modal image registration, using 3D geometry as an intermediate alignment space to obtain fully automatic, strong pixel-level RGB-T alignment with 87% registration accuracy and no hardware-level synchronization. Applying our framework to existing geo-referenced aerial imagery, we construct SegFly, a large-scale benchmark with over 20,000 high-resolution RGB images and more than 15,000 geometrically aligned RGB-T pairs spanning diverse urban, industrial, and rural environments across multiple altitudes and seasons. On SegFly, we establish the Firefly baseline for RGB and thermal semantic segmentation and show that both conventional architectures and vision foundation models benefit substantially from SegFly supervision, highlighting the potential of geometry-driven 2D-3D-2D pipelines for scalable multi-modal scene understanding. Data and Code available at this https URL.
>
---
#### [new 076] OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery
- **分类: cs.CV**

- **简介: 该论文属于人体网格重建任务，解决视频中在线重建人体姿态与轨迹的问题。提出OnlineHMR框架，实现高效、实时的在线处理。**

- **链接: [https://arxiv.org/pdf/2603.17355](https://arxiv.org/pdf/2603.17355)**

> **作者:** Yiwen Zhao; Ce Zheng; Yufu Wang; Hsueh-Han Daniel Yang; Liting Wen; Laszlo A. Jeni
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Human mesh recovery (HMR) models 3D human body from monocular videos, with recent works extending it to world-coordinate human trajectory and motion reconstruction. However, most existing methods remain offline, relying on future frames or global optimization, which limits their applicability in interactive feedback and perception-action loop scenarios such as AR/VR and telepresence. To address this, we propose OnlineHMR, a fully online framework that jointly satisfies four essential criteria of online processing, including system-level causality, faithfulness, temporal consistency, and efficiency. Built upon a two-branch architecture, OnlineHMR enables streaming inference via a causal key-value cache design and a curated sliding-window learning strategy. Meanwhile, a human-centric incremental SLAM provides online world-grounded alignment under physically plausible trajectory correction. Experimental results show that our method achieves performance comparable to existing chunk-based approaches on the standard EMDB benchmark and highly dynamic custom videos, while uniquely supporting online processing. Page and code are available at this https URL.
>
---
#### [new 077] CineSRD: Leveraging Visual, Acoustic, and Linguistic Cues for Open-World Visual Media Speaker Diarization
- **分类: cs.CV; cs.AI; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于视觉媒体说话人日志任务，解决开放场景下的多模态说话人识别问题。提出CineSRD框架，融合视觉、音频和语言信息，提升复杂视频中的说话人标注效果。**

- **链接: [https://arxiv.org/pdf/2603.16966](https://arxiv.org/pdf/2603.16966)**

> **作者:** Liangbin Huang; Xiaohua Liao; Chaoqun Cui; Shijing Wang; Zhaolong Huang; Yanlong Du; Wenji Mao
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Traditional speaker diarization systems have primarily focused on constrained scenarios such as meetings and interviews, where the number of speakers is limited and acoustic conditions are relatively clean. To explore open-world speaker diarization, we extend this task to the visual media domain, encompassing complex audiovisual programs such as films and TV series. This new setting introduces several challenges, including long-form video understanding, a large number of speakers, cross-modal asynchrony between audio and visual cues, and uncontrolled in-the-wild variability. To address these challenges, we propose Cinematic Speaker Registration & Diarization (CineSRD), a unified multimodal framework that leverages visual, acoustic, and linguistic cues from video, speech, and subtitles for speaker annotation. CineSRD first performs visual anchor clustering to register initial speakers and then integrates an audio language model for speaker turn detection, refining annotations and supplementing unregistered off-screen speakers. Furthermore, we construct and release a dedicated speaker diarization benchmark for visual media that includes Chinese and English programs. Experimental results demonstrate that CineSRD achieves superior performance on the proposed benchmark and competitive results on conventional datasets, validating its robustness and generalizability in open-world visual media settings.
>
---
#### [new 078] VisionNVS: Self-Supervised Inpainting for Novel View Synthesis under the Virtual-Shift Paradigm
- **分类: cs.CV**

- **简介: 该论文属于新型视角合成任务，解决自动驾驶中新视角缺乏真实标注的问题。提出VisionNVS框架，通过自监督修复实现更精确的视图合成。**

- **链接: [https://arxiv.org/pdf/2603.17382](https://arxiv.org/pdf/2603.17382)**

> **作者:** Hongbo Lu; Liang Yao; Chenghao He; Fan Liu; Wenlong Liao; Tao He; Pai Peng
>
> **摘要:** A fundamental bottleneck in Novel View Synthesis (NVS) for autonomous driving is the inherent supervision gap on novel trajectories: models are tasked with synthesizing unseen views during inference, yet lack ground truth images for these shifted poses during training. In this paper, we propose VisionNVS, a camera-only framework that fundamentally reformulates view synthesis from an ill-posed extrapolation problem into a self-supervised inpainting task. By introducing a ``Virtual-Shift'' strategy, we use monocular depth proxies to simulate occlusion patterns and map them onto the original view. This paradigm shift allows the use of raw, recorded images as pixel-perfect supervision, effectively eliminating the domain gap inherent in previous approaches. Furthermore, we address spatial consistency through a Pseudo-3D Seam Synthesis strategy, which integrates visual data from adjacent cameras during training to explicitly model real-world photometric discrepancies and calibration errors. Experiments demonstrate that VisionNVS achieves superior geometric fidelity and visual quality compared to LiDAR-dependent baselines, offering a robust solution for scalable driving simulation.
>
---
#### [new 079] MSRAMIE: Multimodal Structured Reasoning Agent for Multi-instruction Image Editing
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MSRAMIE，解决多指令图像编辑任务中模型表现下降的问题。通过结构化多模态推理，提升复杂指令的遵循能力和编辑效率。**

- **链接: [https://arxiv.org/pdf/2603.16967](https://arxiv.org/pdf/2603.16967)**

> **作者:** Zhaoyuan Qiu; Ken Chen; Xiangwei Wang; Yu Xia; Sachith Seneviratne; Saman Halgamuge
>
> **备注:** 14 pages, 6 figures, 3 tables, appendix and references provided
>
> **摘要:** Existing instruction-based image editing models perform well with simple, single-step instructions but degrade in realistic scenarios that involve multiple, lengthy, and interdependent directives. A main cause is the scarcity of training data with complex multi-instruction annotations. However, it is costly to collect such data and retrain these models. To address this challenge, we propose MSRAMIE, a training-free agent framework built on Multimodal Large Language Model (MLLM). MSRAMIE takes existing editing models as plug-in components and handle multi-instruction tasks via structured multimodal reasoning. It orchestrates iterative interactions between an MLLM-based Instructor and an image editing Actor, introducing a novel reasoning topology that comprises the proposed Tree-of-States and Graph-of-References. During inference, complex instructions are decomposed into multiple editing steps which enable state transitions, cross-step information aggregation, and original input recall, which enables systematic exploration of the image editing space and flexible progressive output refinement. The visualizable inference topology further provides interpretable and controllable decision pathways. Experiments show that as the instruction complexity increases, MSRAMIE can improve instruction following over 15% and increases the probability of finishing all modifications in a single run over 100%, while preserving perceptual quality and maintaining visual consistency.
>
---
#### [new 080] Tokenization vs. Augmentation: A Systematic Study of Writer Variance in IMU-Based Online Handwriting Recognition
- **分类: cs.CV; cs.CL; cs.LG; eess.SP**

- **简介: 该论文研究在线惯性传感器手写识别中的作者差异问题，探讨子词分词与拼接增强两种策略的效果。任务是提升识别准确率，解决不同作者风格差异和字符分布不均问题。**

- **链接: [https://arxiv.org/pdf/2603.16883](https://arxiv.org/pdf/2603.16883)**

> **作者:** Jindong Li; Dario Zanca; Vincent Christlein; Tim Hamann; Jens Barth; Peter Kämpf; Björn Eskofier
>
> **摘要:** Inertial measurement unit-based online handwriting recognition enables the recognition of input signals collected across different writing surfaces but remains challenged by uneven character distributions and inter-writer variability. In this work, we systematically investigate two strategies to address these issues: sub-word tokenization and concatenation-based data augmentation. Our experiments on the OnHW-Words500 dataset reveal a clear dichotomy between handling inter-writer and intra-writer variance. On the writer-independent split, structural abstraction via Bigram tokenization significantly improves performance to unseen writing styles, reducing the word error rate (WER) from 15.40% to 12.99%. In contrast, on the writer-dependent split, tokenization degrades performance due to vocabulary distribution shifts between the training and validation sets. Instead, our proposed concatenation-based data augmentation acts as a powerful regularizer, reducing the character error rate by 34.5% and the WER by 25.4%. Further analysis shows that short, low-level tokens benefit model performance and that concatenation-based data augmentation performance gain surpasses those achieved by proportionally extended training. These findings reveal a clear variance-dependent effect: sub-word tokenization primarily mitigates inter-writer stylistic variability, whereas concatenation-based data augmentation effectively compensates for intra-writer distributional sparsity.
>
---
#### [new 081] Are a Thousand Words Better Than a Single Picture? Beyond Images -- A Framework for Multi-Modal Knowledge Graph Dataset Enrichment
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态知识图谱任务，解决图像数据难以收集和处理的问题。通过自动流程将图像转为文本，提升知识图谱的完整性与准确性。**

- **链接: [https://arxiv.org/pdf/2603.16974](https://arxiv.org/pdf/2603.16974)**

> **作者:** Pengyu Zhang; Klim Zaporojets; Jie Liu; Jia-Hong Huang; Paul Groth
>
> **摘要:** Multi-Modal Knowledge Graphs (MMKGs) benefit from visual information, yet large-scale image collection is hard to curate and often excludes ambiguous but relevant visuals (e.g., logos, symbols, abstract scenes). We present Beyond Images, an automatic data-centric enrichment pipeline with optional human auditing. This pipeline operates in three stages: (1) large-scale retrieval of additional entity-related images, (2) conversion of all visual inputs into textual descriptions to ensure that ambiguous images contribute usable semantics rather than noise, and (3) fusion of multi-source descriptions using a large language model (LLM) to generate concise, entity-aligned summaries. These summaries replace or augment the text modality in standard MMKG models without changing their architectures or loss functions. Across three public MMKG datasets and multiple baseline models, we observe consistent gains (up to 7% Hits@1 overall). Furthermore, on a challenging subset of entities with visually ambiguous logos and symbols, converting images into text yields large improvements (201.35% MRR and 333.33% Hits@1). Additionally, we release a lightweight Text-Image Consistency Check Interface for optional targeted audits, improving description quality and dataset reliability. Our results show that scaling image coverage and converting ambiguous visuals into text is a practical path to stronger MMKG completion. Code, datasets, and supplementary materials are available at this https URL.
>
---
#### [new 082] WeatherReasonSeg: A Benchmark for Weather-Aware Reasoning Segmentation in Visual Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型的推理分割任务，旨在解决恶劣天气下模型性能下降的问题。通过构建合成与真实天气数据集，评估模型在不同天气条件下的推理分割能力。**

- **链接: [https://arxiv.org/pdf/2603.17680](https://arxiv.org/pdf/2603.17680)**

> **作者:** Wanjun Du; Zifeng Yuan; Tingting Chen; Fucai Ke; Beibei Lin; Shunli Zhang
>
> **摘要:** Existing vision-language models (VLMs) have demonstrated impressive performance in reasoning-based segmentation. However, current benchmarks are primarily constructed from high-quality images captured under idealized conditions. This raises a critical question: when visual cues are severely degraded by adverse weather conditions such as rain, snow, or fog, can VLMs sustain reliable reasoning segmentation capabilities? In response to this challenge, we introduce WeatherReasonSeg, a benchmark designed to evaluate VLM performance in reasoning-based segmentation under adverse weather conditions. It consists of two complementary components. First, we construct a controllable reasoning dataset by applying synthetic weather with varying severity levels to existing segmentation datasets, enabling fine-grained robustness analysis. Second, to capture real-world complexity, we curate a real-world adverse-weather reasoning segmentation dataset with semantically consistent queries generated via mask-guided LLM prompting. We further broaden the evaluation scope across five reasoning dimensions, including functionality, application scenarios, structural attributes, interactions, and requirement matching. Extensive experiments across diverse VLMs reveal two key findings: (1) VLM performance degrades monotonically with increasing weather severity, and (2) different weather types induce distinct vulnerability patterns. We hope WeatherReasonSeg will serve as a foundation for advancing robust, weather-aware reasoning.
>
---
#### [new 083] TDMM-LM: Bridging Facial Understanding and Animation via Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于面部动画任务，解决缺乏标注文本-面部数据的问题。通过生成大量面部数据，训练语言模型实现文本到面部动作的生成与理解。**

- **链接: [https://arxiv.org/pdf/2603.16936](https://arxiv.org/pdf/2603.16936)**

> **作者:** Luchuan Song; Pinxin Liu; Haiyang Liu; Zhenchao Jin; Yolo Yunlong Tang; Zichong Xu; Susan Liang; Jing Bi; Jason J Corso; Chenliang Xu
>
> **备注:** 12 pages, 13 figures
>
> **摘要:** Text-guided human body animation has advanced rapidly, yet facial animation lags due to the scarcity of well-annotated, text-paired facial corpora. To close this gap, we leverage foundation generative models to synthesize a large, balanced corpus of facial behavior. We design prompts suite covering emotions and head motions, generate about 80 hours of facial videos with multiple generators, and fit per-frame 3D facial parameters, yielding large-scale (prompt and parameter) pairs for training. Building on this dataset, we probe language models for bidirectional competence over facial motion via two complementary tasks: (1) Motion2Language: given a sequence of 3D facial parameters, the model produces natural-language descriptions capturing content, style, and dynamics; and (2) Language2Motion: given a prompt, the model synthesizes the corresponding sequence of 3D facial parameters via quantized motion tokens for downstream animation. Extensive experiments show that in this setting language models can both interpret and synthesize facial motion with strong generalization. To best of our knowledge, this is the first work to cast facial-parameter modeling as a language problem, establishing a unified path for text-conditioned facial animation and motion understanding.
>
---
#### [new 084] Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于模型压缩任务，旨在解决大视觉语言模型量化中精度下降的问题。通过引入细粒度的量化策略，提升量化效果。**

- **链接: [https://arxiv.org/pdf/2603.17809](https://arxiv.org/pdf/2603.17809)**

> **作者:** Ziwei Xiang; Fanhu Zeng; Hongjian Fang; Rui-Qi Wang; Renxing Chen; Yanan Zhu; Yi Chen; Peipei Yang; Xu-Yao Zhang
>
> **备注:** Accepted by CVPR 2026 Main Conference
>
> **摘要:** Large Vision Language Models (LVLMs) have achieved remarkable success in a range of downstream tasks that require multimodal interaction, but their capabilities come with substantial computational and memory overhead, which hinders practical deployment. Among numerous acceleration techniques, post-training quantization is a popular and effective strategy for reducing memory cost and accelerating inference. However, existing LVLM quantization methods typically measure token sensitivity at the modality level, which fails to capture the complex cross-token interactions and falls short in quantitatively measuring the quantization error at the token level. As tokens interact within the model, the distinction between modalities gradually diminishes, suggesting the need for fine-grained calibration. Inspired by axiomatic attribution in mechanistic interpretability, we introduce a fine-grained quantization strategy on Quantization-aware Integrated Gradients (QIG), which leverages integrated gradients to quantitatively evaluate token sensitivity and push the granularity from modality level to token level, reflecting both inter-modality and intra-modality dynamics. Extensive experiments on multiple LVLMs under both W4A8 and W3A16 settings show that our method improves accuracy across models and benchmarks with negligible latency overhead. For example, under 3-bit weight-only quantization, our method improves the average accuracy of LLaVA-onevision-7B by 1.60%, reducing the gap to its full-precision counterpart to only 1.33%. The code is available at this https URL.
>
---
#### [new 085] Recurrent Reasoning with Vision-Language Models for Estimating Long-Horizon Embodied Task Progress
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于机器人任务进度估计任务，旨在解决长时序多步骤任务中进度估计难题。提出R²VLM模型，通过循环推理框架高效处理视频并保持全局上下文，提升任务推理能力。**

- **链接: [https://arxiv.org/pdf/2603.17312](https://arxiv.org/pdf/2603.17312)**

> **作者:** Yuelin Zhang; Sijie Cheng; Chen Li; Zongzhao Li; Yuxin Huang; Yang Liu; Wenbing Huang
>
> **备注:** CVPR 2026
>
> **摘要:** Accurately estimating task progress is critical for embodied agents to plan and execute long-horizon, multi-step tasks. Despite promising advances, existing Vision-Language Models (VLMs) based methods primarily leverage their video understanding capabilities, while neglecting their complex reasoning potential. Furthermore, processing long video trajectories with VLMs is computationally prohibitive for real-world deployment. To address these challenges, we propose the Recurrent Reasoning Vision-Language Model ($\text{R}^2$VLM). Our model features a recurrent reasoning framework that processes local video snippets iteratively, maintaining a global context through an evolving Chain of Thought (CoT). This CoT explicitly records task decomposition, key steps, and their completion status, enabling the model to reason about complex temporal dependencies. This design avoids the high cost of processing long videos while preserving essential reasoning capabilities. We train $\text{R}^2$VLM on large-scale, automatically generated datasets from ALFRED and Ego4D. Extensive experiments on progress estimation and downstream applications, including progress-enhanced policy learning, reward modeling for reinforcement learning, and proactive assistance, demonstrate that $\text{R}^2$VLM achieves strong performance and generalization, achieving a new state-of-the-art in long-horizon task progress estimation. The models and benchmarks are publicly available at \href{this https URL}{huggingface}.
>
---
#### [new 086] SMAL-pets: SMAL Based Avatars of Pets from Single Image
- **分类: cs.CV**

- **简介: 该论文属于3D动物角色生成任务，解决单图生成高质量可编辑宠物模型的问题。提出SMAL-pets框架，结合3D高斯泼溅与SMAL模型，实现高保真、可编辑的宠物虚拟形象。**

- **链接: [https://arxiv.org/pdf/2603.17131](https://arxiv.org/pdf/2603.17131)**

> **作者:** Piotr Borycki; Joanna Waczyńska; Yizhe Zhu; Yongqiang Gao; Przemysław Spurek
>
> **摘要:** Creating high-fidelity, animatable 3D dog avatars remains a formidable challenge in computer vision. Unlike human digital doubles, animal reconstruction faces a critical shortage of large-scale, annotated datasets for specialized applications. Furthermore, the immense morphological diversity across species, breeds, and crosses, which varies significantly in size, proportions, and features, complicates the generalization of existing models. Current reconstruction methods often struggle to capture realistic fur textures. Additionally, ensuring these avatars are fully editable and capable of performing complex, naturalistic movements typically necessitates labor-intensive manual mesh manipulation and expert rigging. This paper introduces SMAL-pets, a comprehensive framework that generates high-quality, editable animal avatars from a single input image. Our approach bridges the gap between reconstruction and generative modeling by leveraging a hybrid architecture. Our method integrates 3D Gaussian Splatting with the SMAL parametric model to provide a representation that is both visually high-fidelity and anatomically grounded. We introduce a multimodal editing suite that enables users to refine the avatar's appearance and execute complex animations through direct textual prompts. By allowing users to control both the aesthetic and behavioral aspects of the model via natural language, SMAL-pets provides a flexible, robust tool for animation and virtual reality.
>
---
#### [new 087] FACE-net: Factual Calibration and Emotion Augmentation for Retrieval-enhanced Emotional Video Captioning
- **分类: cs.CV**

- **简介: 该论文属于情感视频描述任务，解决事实与情感偏差问题。提出FACE-net框架，通过检索增强、事实校准和情感增强，提升描述准确性与适应性。**

- **链接: [https://arxiv.org/pdf/2603.17455](https://arxiv.org/pdf/2603.17455)**

> **作者:** Weidong Chen; Cheng Ye; Zhendong Mao; Peipei Song; Xinyan Liu; Lei Zhang; Xiaojun Chang; Yongdong Zhang
>
> **备注:** Submitted to TPAMI. 16 pages, 9 figures
>
> **摘要:** Emotional Video Captioning (EVC) is an emerging task, which aims to describe factual content with the intrinsic emotions expressed in videos. Existing works perceive global emotional cues and then combine with video content to generate descriptions. However, insufficient factual and emotional cues mining and coordination during generation make their methods difficult to deal with the factual-emotional bias, which refers to the factual and emotional requirements being different in different samples on generation. To this end, we propose a retrieval-enhanced framework with FActual Calibration and Emotion augmentation (FACE-net), which through a unified architecture collaboratively mines factual-emotional semantics and provides adaptive and accurate guidance for generation, breaking through the compromising tendency of factual-emotional descriptions in all sample learning. Technically, we firstly introduces an external repository and retrieves the most relevant sentences with the video content to augment the semantic information. Subsequently, our factual calibration via uncertainty estimation module splits the retrieved information into subject-predicate-object triplets, and self-refines and cross-refines different components through video content to effectively mine the factual semantics; while our progressive visual emotion augmentation module leverages the calibrated factual semantics as experts, interacts with the video content and emotion dictionary to generate visual queries and candidate emotions, and then aggregates them to adaptively augment emotions to each factual semantics. Moreover, to alleviate the factual-emotional bias, we design a dynamic bias adjustment routing module to predict and adjust the degree of bias of a sample.
>
---
#### [new 088] Concept-to-Pixel: Prompt-Free Universal Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决传统方法依赖人工提示和跨模态泛化不足的问题。提出C2P框架，通过分离解剖知识为几何与语义表示，实现无需提示的通用分割。**

- **链接: [https://arxiv.org/pdf/2603.17746](https://arxiv.org/pdf/2603.17746)**

> **作者:** Haoyun Chen; Fenghe Tang; Wenxin Ma; Shaohua Kevin Zhou
>
> **备注:** 32 pages, code is available at: this https URL
>
> **摘要:** Universal medical image segmentation seeks to use a single foundational model to handle diverse tasks across multiple imaging modalities. However, existing approaches often rely heavily on manual visual prompts or retrieved reference images, which limits their automation and robustness. In addition, naive joint training across modalities often fails to address large domain shifts. To address these limitations, we propose Concept-to-Pixel (C2P), a novel prompt-free universal segmentation framework. C2P explicitly separates anatomical knowledge into two components: Geometric and Semantic representations. It leverages Multimodal Large Language Models (MLLMs) to distill abstract, high-level medical concepts into learnable Semantic Tokens and introduces explicitly supervised Geometric Tokens to enforce universal physical and structural constraints. These disentangled tokens interact deeply with image features to generate input-specific dynamic kernels for precise mask prediction. Furthermore, we introduce a Geometry-Aware Inference Consensus mechanism, which utilizes the model's predicted geometric constraints to assess prediction reliability and suppress outliers. Extensive experiments and analysis on a unified benchmark comprising eight diverse datasets across seven modalities demonstrate the significant superiority of our jointly trained approach, compared to universe- or single-model approaches. Remarkably, our unified model demonstrates strong generalization, achieving impressive results not only on zero-shot tasks involving unseen cases but also in cross-modal transfers across similar tasks. Code is available at: this https URL
>
---
#### [new 089] SHIFT: Motion Alignment in Video Diffusion Models with Adversarial Hybrid Fine-Tuning
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决视频扩散模型在微调后运动对齐不足的问题。提出SHIFT框架，结合像素运动奖励与混合微调，提升运动一致性与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.17426](https://arxiv.org/pdf/2603.17426)**

> **作者:** Xi Ye; Wenjia Yang; Yangyang Xu; Xiaoyang Liu; Duo Su; Mengfei Xia; Jun Zhu
>
> **摘要:** Image-conditioned Video diffusion models achieve impressive visual realism but often suffer from weakened motion fidelity, e.g., reduced motion dynamics or degraded long-term temporal coherence, especially after fine-tuning. We study the problem of motion alignment in video diffusion models post-training. To address this, we introduce pixel-motion rewards based on pixel flux dynamics, capturing both instantaneous and long-term motion consistency. We further propose Smooth Hybrid Fine-tuning (SHIFT), a scalable reward-driven fine-tuning framework for video diffusion models. SHIFT fuses the normal supervised fine-tuning and advantage weighted fine-tuning into a unified framework. Benefiting from novel adversarial advantages, SHIFT improves convergence speed and mitigates reward hacking. Experiments show that our approach efficiently resolves dynamic-degree collapse in modern video diffusion models supervised fine-tuning.
>
---
#### [new 090] DANCE: Dynamic 3D CNN Pruning: Joint Frame, Channel, and Feature Adaptation for Energy Efficiency on the Edge
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于模型压缩任务，旨在提升3D CNN在边缘设备的能效。通过动态剪枝，减少计算和内存开销，同时保持性能。**

- **链接: [https://arxiv.org/pdf/2603.17275](https://arxiv.org/pdf/2603.17275)**

> **作者:** Mohamed Mejri; Ashiqur Rasul; Abhijit Chatterjee
>
> **摘要:** Modern convolutional neural networks (CNNs) are workhorses for video and image processing, but fail to adapt to the computational complexity of input samples in a dynamic manner to minimize energy consumption. In this research, we propose DANCE, a fine-grained, input-aware, dynamic pruning framework for 3D CNNs to maximize power efficiency with negligible to zero impact on performance. In the proposed two-step approach, the first step is called activation variability amplification (AVA), and the 3D CNN model is retrained to increase the variance of the magnitude of neuron activations across the network in this step, facilitating pruning decisions across diverse CNN input scenarios. In the second step, called adaptive activation pruning (AAP), a lightweight activation controller network is trained to dynamically prune frames, channels, and features of 3D convolutional layers of the network (different for each layer), based on statistics of the outputs of the first layer of the network. Our method achieves substantial savings in multiply-accumulate (MAC) operations and memory accesses by introducing sparsity within convolutional layers. Hardware validation on the NVIDIA Jetson Nano GPU and the Qualcomm Snapdragon 8 Gen 1 platform demonstrates respective speedups of 1.37X and 2.22X, achieving up to 1.47X higher energy efficiency compared to the state of the art.
>
---
#### [new 091] Revisiting foundation models for cell instance segmentation
- **分类: cs.CV**

- **简介: 该论文聚焦于显微图像中的细胞实例分割任务，评估并改进了基于SAM的模型，提出自动提示生成策略以提升分割效果。**

- **链接: [https://arxiv.org/pdf/2603.17845](https://arxiv.org/pdf/2603.17845)**

> **作者:** Anwai Archit; Constantin Pape
>
> **备注:** Published in MIDL 2026
>
> **摘要:** Cell segmentation is a fundamental task in microscopy image analysis. Several foundation models for cell segmentation have been introduced, virtually all of them are extensions of Segment Anything Model (SAM), improving it for microscopy data. Recently, SAM2 and SAM3 have been published, further improving and extending the capabilities of general-purpose segmentation foundation models. Here, we comprehensively evaluate foundation models for cell segmentation (CellPoseSAM, CellSAM, $\mu$SAM) and for general-purpose segmentation (SAM, SAM2, SAM3) on a diverse set of (light) microscopy datasets, for tasks including cell, nucleus and organoid segmentation. Furthermore, we introduce a new instance segmentation strategy called automatic prompt generation (APG) that can be used to further improve SAM-based microscopy foundation models. APG consistently improves segmentation results for $\mu$SAM, which is used as the base model, and is competitive with the state-of-the-art model CellPoseSAM. Moreover, our work provides important lessons for adaptation strategies of SAM-style models to microscopy and provides a strategy for creating even more powerful microscopy foundation models. Our code is publicly available at this https URL.
>
---
#### [new 092] Part-Aware Open-Vocabulary 3D Affordance Grounding via Prototypical Semantic and Geometric Alignment
- **分类: cs.CV**

- **简介: 该论文属于3D语义对齐任务，解决开放词汇下3D对象的语义与几何对齐问题。提出两阶段框架，增强语义与几何表示，提升精准度。**

- **链接: [https://arxiv.org/pdf/2603.17647](https://arxiv.org/pdf/2603.17647)**

> **作者:** Dongqiang Gou; Xuming He
>
> **摘要:** Grounding natural language questions to functionally relevant regions in 3D objects -- termed language-driven 3D affordance grounding -- is essential for embodied intelligence and human-AI interaction. Existing methods, while progressing from label-based to language-driven approaches, still face challenges in open-vocabulary generalization, fine-grained geometric alignment, and part-level semantic consistency. To address these issues, we propose a novel two-stage cross-modal framework that enhances both semantic and geometric representations for open-vocabulary 3D affordance grounding. In the first stage, large language models generate part-aware instructions to recover missing semantics, enabling the model to link semantically similar affordances. In the second stage, we introduce two key components: Affordance Prototype Aggregation (APA), which captures cross-object geometric consistency for each affordance, and Intra-Object Relational Modeling (IORM), which refines geometric differentiation within objects to support precise semantic alignment. We validate the effectiveness of our method through extensive experiments on a newly introduced benchmark, as well as two existing benchmarks, demonstrating superior performance in comparison with existing methods.
>
---
#### [new 093] MedSAD-CLIP: Supervised CLIP with Token-Patch Cross-Attention for Medical Anomaly Detection and Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学异常检测与分割任务，解决传统方法在定位和分割质量上的不足。通过改进CLIP模型，引入细粒度文本视觉对齐和对比损失，提升检测精度。**

- **链接: [https://arxiv.org/pdf/2603.17325](https://arxiv.org/pdf/2603.17325)**

> **作者:** Thuy Truong Tran; Minh Kha Do; Phuc Nguyen Duy; Min Hun Lee
>
> **摘要:** Medical anomaly detection (MAD) and segmentation play a critical role in assisting clinical diagnosis by identifying abnormal regions in medical images and localizing pathological regions. Recent CLIP-based studies are promising for anomaly detection in zero-/few-shot settings, and typically rely on global representations and weak supervision, often producing coarse localization and limited segmentation quality. In this work, we study supervised adaptation of CLIP for MAD under a realistic clinical setting where a limited yet meaningful amount of labeled abnormal data is available. Our model MedSAD-CLIP leverages fine-grained text-visual cues via the Token-Patch Cross-Attention(TPCA) to improve lesion localization while preserving the generalization capability of CLIP representations. Lightweight image adapters and learnable prompt tokens efficiently adapt the pretrained CLIP encoder to the medical domain while preserving its rich semantic alignment. Furthermore, a Margin-based image-text Contrastive Loss is designed to enhance global feature discrimination between normal and abnormal representations. Extensive experiments on four diverse benchmarks-Brain, Retina, Lung, and Breast datasets-demonstrate the effectiveness of our approach, achieving superior performance in both pixel-level segmentation and image-level classification over state-of-the-art methods. Our results highlight the potential of supervised CLIP adaptation as a unified and scalable paradigm for medical anomaly understanding. Code will be made available at this https URL
>
---
#### [new 094] A 3D Reconstruction Benchmark for Asset Inspection
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于3D重建任务，旨在解决资产检测中高精度模型构建的问题。针对现有数据集不足，作者构建了包含真实场景和复杂表面条件的新数据集，并评估了现有方法的局限性。**

- **链接: [https://arxiv.org/pdf/2603.17358](https://arxiv.org/pdf/2603.17358)**

> **作者:** James L. Gray; Nikolai Goncharov; Alexandre Cardaillac; Ryan Griffiths; Jack Naylor; Donald G. Dansereau
>
> **备注:** 29 pages, 15 figures, 8 tables
>
> **摘要:** Asset management requires accurate 3D models to inform the maintenance, repair, and assessment of buildings, maritime vessels, and other key structures as they age. These downstream applications rely on high-fidelity models produced from aerial surveys in close proximity to the asset, enabling operators to locate and characterise deterioration or damage and plan repairs. Captured images typically have high overlap between adjacent camera poses, sufficient detail at millimetre scale, and challenging visual appearances such as reflections and transparency. However, existing 3D reconstruction datasets lack examples of these conditions, making it difficult to benchmark methods for this task. We present a new dataset with ground truth depth maps, camera poses, and mesh models of three synthetic scenes with simulated inspection trajectories and varying levels of surface condition on non-Lambertian scene content. We evaluate state-of-the-art reconstruction methods on this dataset. Our results demonstrate that current approaches struggle significantly with the dense capture trajectories and complex surface conditions inherent to this domain, exposing a critical scalability gap and pointing toward new research directions for deployable 3D reconstruction in asset inspection. Project page: this https URL
>
---
#### [new 095] FINER: MLLMs Hallucinate under Fine-grained Negative Queries
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态语言模型任务，旨在解决MLLM在细粒度负查询下的幻觉问题。通过构建基准和提出FINER-Tuning方法，有效减少幻觉并提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.17662](https://arxiv.org/pdf/2603.17662)**

> **作者:** Rui Xiao; Sanghwan Kim; Yongqin Xian; Zeynep Akata; Stephan Alaniz
>
> **备注:** CVPR 2026
>
> **摘要:** Multimodal large language models (MLLMs) struggle with hallucinations, particularly with fine-grained queries, a challenge underrepresented by existing benchmarks that focus on coarse image-related questions. We introduce FIne-grained NEgative queRies (FINER), alongside two benchmarks: FINER-CompreCap and FINER-DOCCI. Using FINER, we analyze hallucinations across four settings: multi-object, multi-attribute, multi-relation, and ``what'' questions. Our benchmarks reveal that MLLMs hallucinate when fine-grained mismatches co-occur with genuinely present elements in the image. To address this, we propose FINER-Tuning, leveraging Direct Preference Optimization (DPO) on FINER-inspired data. Finetuning four frontier MLLMs with FINER-Tuning yields up to 24.2\% gains (InternVL3.5-14B) on hallucinations from our benchmarks, while simultaneously improving performance on eight existing hallucination suites, and enhancing general multimodal capabilities across six benchmarks. Code, benchmark, and models are available at \href{this https URL}{this https URL}.
>
---
#### [new 096] Omni IIE Bench: Benchmarking the Practical Capabilities of Image Editing Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像编辑任务，旨在解决模型在不同语义尺度任务中表现不一致的问题。提出Omni IIE Bench基准，评估并诊断模型的编辑一致性。**

- **链接: [https://arxiv.org/pdf/2603.16944](https://arxiv.org/pdf/2603.16944)**

> **作者:** Yujia Yang; Yuanxiang Wang; Zhenyu Guan; Tiankun Yang; Chenxi Bao; Haopeng Jin; Jinwen Luo; Xinyu Zuo; Lisheng Duan; Haijin Liang; Jin Ma; Xinming Wang; Ruiwen Tao; Hongzhu Yi
>
> **摘要:** While Instruction-based Image Editing (IIE) has achieved significant progress, existing benchmarks pursue task breadth via mixed evaluations. This paradigm obscures a critical failure mode crucial in professional applications: the inconsistent performance of models across tasks of varying semantic scales. To address this gap, we introduce Omni IIE Bench, a high-quality, human-annotated benchmark specifically designed to diagnose the editing consistency of IIE models in practical application scenarios. Omni IIE Bench features an innovative dual-track diagnostic design: (1) Single-turn Consistency, comprising shared-context task pairs of attribute modification and entity replacement; and (2) Multi-turn Coordination, involving continuous dialogue tasks that traverse semantic scales. The benchmark is constructed via an exceptionally rigorous multi-stage human filtering process, incorporating a quality standard enforced by computer vision graduate students and an industry relevance review conducted by professional designers. We perform a comprehensive evaluation of 8 mainstream IIE models using Omni IIE Bench. Our analysis quantifies, for the first time, a prevalent performance gap: nearly all models exhibit a significant performance degradation when transitioning from low-semantic-scale to high-semantic-scale tasks. Omni IIE Bench provides critical diagnostic tools and insights for the development of next-generation, more reliable, and stable IIE models.
>
---
#### [new 097] VISER: Visually-Informed System for Enhanced Robustness in Open-Set Iris Presentation Attack Detection
- **分类: cs.CV**

- **简介: 该论文属于开放集虹膜活体检测任务，旨在提升模型鲁棒性。通过比较不同人类视觉线索，发现去噪眼动热图效果最佳。**

- **链接: [https://arxiv.org/pdf/2603.17859](https://arxiv.org/pdf/2603.17859)**

> **作者:** Byron Dowling; Eleanor Frederick; Jacob Piland; Adam Czajka
>
> **摘要:** Human perceptual priors have shown promise in saliency-guided deep learning training, particularly in the domain of iris presentation attack detection (PAD). Common saliency approaches include hand annotations obtained via mouse clicks and eye gaze heatmaps derived from eye tracking data. However, the most effective form of human saliency for open-set iris PAD remains underexplored. In this paper, we conduct a series of experiments comparing hand annotations, eye tracking heatmaps, segmentation masks, and DINOv2 embeddings to a state-of-the-art deep learning-based baseline on the task of open-set iris PAD. Results for open-set PAD in a leave-one-attack-type out paradigm indicate that denoised eye tracking heatmaps show the best generalization improvement over cross entropy in terms of Area Under the ROC curve (AUROC) and Attack Presentation Classification Error Rate (APCER) at Bona Fide Presentation Classification Error Rate (BPCER) of 1%. Along with this paper, we offer trained models, code, and saliency maps for reproducibility and to facilitate follow-up research efforts.
>
---
#### [new 098] 3D MRI-Based Alzheimer's Disease Classification Using Multi-Modal 3D CNN with Leakage-Aware Subject-Level Evaluation
- **分类: cs.CV**

- **简介: 该论文属于阿尔茨海默病分类任务，旨在利用三维MRI数据提升分类效果。通过多模态3D CNN模型，结合结构和概率图信息，实现更准确的疾病识别。**

- **链接: [https://arxiv.org/pdf/2603.17304](https://arxiv.org/pdf/2603.17304)**

> **作者:** Md Sifat; Sania Akter; Akif Islam; Md. Ekramul Hamid; Abu Saleh Musa Miah; Najmul Hassan; Md Abdur Rahim; Jungpil Shin
>
> **备注:** 5 tables, 6 figures, Submitted to International Conference on Power, Electronics, Communications, Computing, and Intelligent Infrastructure 2026
>
> **摘要:** Deep learning has become an important tool for Alzheimer's disease (AD) classification from structural MRI. Many existing studies analyze individual 2D slices extracted from MRI volumes, while clinical neuroimaging practice typically relies on the full three dimensional structure of the brain. From this perspective, volumetric analysis may better capture spatial relationships among brain regions that are relevant to disease progression. Motivated by this idea, this work proposes a multimodal 3D convolutional neural network for AD classification using raw OASIS 1 MRI volumes. The model combines structural T1 information with gray matter, white matter, and cerebrospinal fluid probability maps obtained through FSL FAST segmentation in order to capture complementary neuroanatomical information. The proposed approach is evaluated on the clinically labelled OASIS 1 cohort using 5 fold subject level cross validation, achieving a mean accuracy of 72.34% plus or minus 4.66% and a ROC AUC of 0.7781 plus or minus 0.0365. GradCAM visualizations further indicate that the model focuses on anatomically meaningful regions, including the medial temporal lobe and ventricular areas that are known to be associated with Alzheimer's related structural changes. To better understand how data representation and evaluation strategies may influence reported performance, additional diagnostic experiments were conducted on a slice based version of the dataset under both slice level and subject level protocols. These observations help provide context for the volumetric results. Overall, the proposed multimodal 3D framework establishes a reproducible subject level benchmark and highlights the potential benefits of volumetric MRI analysis for Alzheimer's disease classification.
>
---
#### [new 099] Stereo World Model: Camera-Guided Stereo Video Generation
- **分类: cs.CV**

- **简介: 该论文提出StereoWorld，属于立体视频生成任务，解决单目方法生成立体视频的不足，通过联合学习外观与双目几何，提升一致性与效率。**

- **链接: [https://arxiv.org/pdf/2603.17375](https://arxiv.org/pdf/2603.17375)**

> **作者:** Yang-Tian Sun; Zehuan Huang; Yifan Niu; Lin Ma; Yan-Pei Cao; Yuewen Ma; Xiaojuan Qi
>
> **备注:** Project Page: this https URL
>
> **摘要:** We present StereoWorld, a camera-conditioned stereo world model that jointly learns appearance and binocular geometry for end-to-end stereo video this http URL monocular RGB or RGBD approaches, StereoWorld operates exclusively within the RGB modality, while simultaneously grounding geometry directly from disparity. To efficiently achieve consistent stereo generation, our approach introduces two key designs: (1) a unified camera-frame RoPE that augments latent tokens with camera-aware rotary positional encoding, enabling relative, view- and time-consistent conditioning while preserving pretrained video priors via a stable attention initialization; and (2) a stereo-aware attention decomposition that factors full 4D attention into 3D intra-view attention plus horizontal row attention, leveraging the epipolar prior to capture disparity-aligned correspondences with substantially lower compute. Across benchmarks, StereoWorld improves stereo consistency, disparity accuracy, and camera-motion fidelity over strong monocular-then-convert pipelines, achieving more than 3x faster generation with an additional 5% gain in viewpoint consistency. Beyond benchmarks, StereoWorld enables end-to-end binocular VR rendering without depth estimation or inpainting, enhances embodied policy learning through metric-scale depth grounding, and is compatible with long-video distillation for extended interactive stereo synthesis.
>
---
#### [new 100] LED: A Benchmark for Evaluating Layout Error Detection in Document Analysis
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出LED基准，用于评估文档布局分析中的结构错误检测。解决传统指标无法捕捉逻辑不一致的问题，定义八种错误类型并构建数据集与评估任务。**

- **链接: [https://arxiv.org/pdf/2603.17265](https://arxiv.org/pdf/2603.17265)**

> **作者:** Inbum Heo; Taewook Hwang; Jeesu Jung; Sangkeun Jung
>
> **备注:** 8pages
>
> **摘要:** Recent advances in Large Language Models (LLMs) and Large Multimodal Models (LMMs) have improved Document Layout Analysis (DLA), yet structural errors such as region merging, splitting, and omission remain persistent. Conventional overlap-based metrics (e.g., IoU, mAP) fail to capture such logical inconsistencies. To overcome this limitation, we propose Layout Error Detection (LED), a benchmark that evaluates structural reasoning in DLA predictions beyond surface-level accuracy. LED defines eight standardized error types (Missing, Hallucination, Size Error, Split, Merge, Overlap, Duplicate, and Misclassification) and provides quantitative rules and injection algorithms for realistic error simulation. Using these definitions, we construct LED-Dataset and design three evaluation tasks: document-level error detection, document-level error-type classification, and element-level error-type classification. Experiments with state-of-the-art multimodal models show that LED enables fine-grained and interpretable assessment of structural understanding, revealing clear weaknesses across modalities and architectures. Overall, LED establishes a unified and explainable benchmark for diagnosing the structural robustness and reasoning capability of document understanding models.
>
---
#### [new 101] A Proposal-Free Query-Guided Network for Grounded Multimodal Named Entity Recognition
- **分类: cs.CV**

- **简介: 该论文属于GMNER任务，解决实体与图像区域对齐不精准的问题。提出QGN模型，通过文本引导和跨模态交互实现统一推理，提升识别准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.17314](https://arxiv.org/pdf/2603.17314)**

> **作者:** Hongbing Li; Jiamin Liu; Shuo Zhang; Bo Xiao
>
> **摘要:** Grounded Multimodal Named Entity Recognition (GMNER) identifies named entities, including their spans and types, in natural language text and grounds them to the corresponding regions in associated images. Most existing approaches split this task into two steps: they first detect objects using a pre-trained general-purpose detector and then match named entities to the detected objects. However, these methods face a major limitation. Because pre-trained general-purpose object detectors operate independently of textual entities, they tend to detect common objects and frequently overlook specific fine-grained regions required by named entities. This misalignment between object detectors and entities introduces imprecision and can impair overall system performance. In this paper, we propose a proposal-free Query-Guided Network (QGN) that unifies multimodal reasoning and decoding through text guidance and cross- modal interaction. QGN enables accurate grounding and robust performance in open-domain scenarios. Extensive experiments demonstrate that QGN achieves top performance among compared GMNER models on widely used benchmarks.
>
---
#### [new 102] Omni-3DEdit: Generalized Versatile 3D Editing in One-Pass
- **分类: cs.CV**

- **简介: 该论文提出Omni-3DEdit，解决3D编辑任务通用性差和效率低的问题。通过学习方法实现多任务统一编辑，提升效率并减少计算时间。**

- **链接: [https://arxiv.org/pdf/2603.17841](https://arxiv.org/pdf/2603.17841)**

> **作者:** Chen Liyi; Wang Pengfei; Zhang Guowen; Ma Zhiyuan; Zhang Lei
>
> **备注:** accepted by CVPR26
>
> **摘要:** Most instruction-driven 3D editing methods rely on 2D models to guide the explicit and iterative optimization of 3D representations. This paradigm, however, suffers from two primary drawbacks. First, it lacks a universal design of different 3D editing tasks because the explicit manipulation of 3D geometry necessitates task-dependent rules, e.g., 3D appearance editing demands inherent source 3D geometry, while 3D removal alters source geometry. Second, the iterative optimization process is highly time-consuming, often requiring thousands of invocations of 2D/3D updating. We present Omni-3DEdit, a unified, learning-based model that generalizes various 3D editing tasks implicitly. One key challenge to achieve our goal is the scarcity of paired source-edited multi-view assets for training. To address this issue, we construct a data pipeline, synthesizing a relatively rich number of high-quality paired multi-view editing samples. Subsequently, we adapt the pre-trained generative model SEVA as our backbone by concatenating source view latents along with conditional tokens in sequence space. A dual-stream LoRA module is proposed to disentangle different view cues, largely enhancing our model's representational learning capability. As a learning-based model, our model is free of the time-consuming online optimization, and it can complete various 3D editing tasks in one forward pass, reducing the inference time from tens of minutes to approximately two minutes. Extensive experiments demonstrate the effectiveness and efficiency of Omni-3DEdit.
>
---
#### [new 103] Universal Skeleton Understanding via Differentiable Rendering and MLLMs
- **分类: cs.CV**

- **简介: 该论文提出SkeletonLLM，解决MLLM无法直接处理骨骼数据的问题。通过不同iable渲染将骨骼序列转为视觉模态，提升多任务理解能力。**

- **链接: [https://arxiv.org/pdf/2603.18003](https://arxiv.org/pdf/2603.18003)**

> **作者:** Ziyi Wang; Peiming Li; Xinshun Wang; Yang Tang; Kai-Kuang Ma; Mengyuan Liu
>
> **备注:** 32 pages, 15 figures
>
> **摘要:** Multimodal large language models (MLLMs) exhibit strong visual-language reasoning, yet remain confined to their native modalities and cannot directly process structured, non-visual data such as human skeletons. Existing methods either compress skeleton dynamics into lossy feature vectors for text alignment, or quantize motion into discrete tokens that generalize poorly across heterogeneous skeleton formats. We present SkeletonLLM, which achieves universal skeleton understanding by translating arbitrary skeleton sequences into the MLLM's native visual modality. At its core is DrAction, a differentiable, format-agnostic renderer that converts skeletal kinematics into compact image sequences. Because the pipeline is end-to-end differentiable, MLLM gradients can directly guide the rendering to produce task-informative visual tokens. To further enhance reasoning capabilities, we introduce a cooperative training strategy: Causal Reasoning Distillation transfers structured, step-by-step reasoning from a teacher model, while Discriminative Finetuning sharpens decision boundaries between confusable actions. SkeletonLLM demonstrates strong generalization on diverse tasks including recognition, captioning, reasoning, and cross-format transfer -- suggesting a viable path for applying MLLMs to non-native modalities. Code will be released upon acceptance.
>
---
#### [new 104] PC-CrossDiff: Point-Cluster Dual-Level Cross-Modal Differential Attention for Unified 3D Referring and Segmentation
- **分类: cs.CV**

- **简介: 该论文属于3D视觉定位任务，解决复杂场景下3D引用表达识别与分割的问题。提出PC-CrossDiff框架，通过双级注意力机制提升定位精度。**

- **链接: [https://arxiv.org/pdf/2603.17753](https://arxiv.org/pdf/2603.17753)**

> **作者:** Wenbin Tan; Jiawen Lin; Fangyong Wang; Yuan Xie; Yong Xie; Yachao Zhang; Yanyun Qu
>
> **摘要:** 3D Visual Grounding (3DVG) aims to localize the referent of natural language referring expressions through two core tasks: Referring Expression Comprehension (3DREC) and Segmentation (3DRES). While existing methods achieve high accuracy in simple, single-object scenes, they suffer from severe performance degradation in complex, multi-object scenes that are common in real-world settings, hindering practical deployment. Existing methods face two key challenges in complex, multi-object scenes: inadequate parsing of implicit localization cues critical for disambiguating visually similar objects, and ineffective suppression of dynamic spatial interference from co-occurring objects, resulting in degraded grounding accuracy. To address these challenges, we propose PC-CrossDiff, a unified dual-task framework with a dual-level cross-modal differential attention architecture for 3DREC and 3DRES. Specifically, the framework introduces: (i) Point-Level Differential Attention (PLDA) modules that apply bidirectional differential attention between text and point clouds, adaptively extracting implicit localization cues via learnable weights to improve discriminative representation; (ii) Cluster-Level Differential Attention (CLDA) modules that establish a hierarchical attention mechanism to adaptively enhance localization-relevant spatial relationships while suppressing ambiguous or irrelevant spatial relations through a localization-aware differential attention block. Our method achieves state-of-the-art performance on the ScanRefer, NR3D, and SR3D benchmarks. Notably, on the Implicit subsets of ScanRefer, it improves the Overall@0.50 score by +10.16% for the 3DREC task, highlighting its strong ability to parse implicit spatial cues.
>
---
#### [new 105] Differential Attention-Augmented BiomedCLIP with Asymmetric Focal Optimization for Imbalanced Multi-Label Video Capsule Endoscopy Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多标签视频胶囊内镜分类任务，解决数据极端不平衡问题。通过改进BiomedCLIP模型和优化策略提升分类性能。**

- **链接: [https://arxiv.org/pdf/2603.17879](https://arxiv.org/pdf/2603.17879)**

> **作者:** Podakanti Satyajith Chary; Nagarajan Ganapathy
>
> **备注:** 9 pages, 1 figure, ICPR 2026 RARE-VISION Competition
>
> **摘要:** This work presents a multi-label classification framework for video capsule endoscopy (VCE) that addresses the extreme class imbalance inherent in the Galar dataset through a combination of architectural and optimization-level strategies. Our approach modifies BiomedCLIP, a biomedical vision-language foundation model, by replacing its standard multi-head self-attention with a differential attention mechanism that computes the difference between two softmax attention maps to suppress attention noise. To counteract the skewed label distribution, where pathological findings constitute less than 0.1% of all annotated frames, a sqrt-frequency weighted sampler, asymmetric focal loss, mixup regularization, and per-class threshold optimization are employed. Temporal coherence is enforced through median-filter smoothing and gap merging prior to event-level JSON generation. On the held-out RARE-VISION test set comprising three NaviCam examinations (161,025 frames), the pipeline achieves an overall temporal mAP@0.5 of 0.2456 and mAP@0.95 of 0.2353, with total inference completed in approximately 8.6 minutes on a single GPU.
>
---
#### [new 106] Empirical Recipes for Efficient and Compact Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型任务，解决紧凑模型效率不足的问题。通过分析瓶颈并提出优化方法，提升推理速度，同时保持精度，构建高效模型ArgusVLM。**

- **链接: [https://arxiv.org/pdf/2603.16987](https://arxiv.org/pdf/2603.16987)**

> **作者:** Jiabo Huang; Zhizhong Li; Sina Sajadmanesh; Weiming Zhuang; Lingjuan Lyu
>
> **摘要:** Deploying vision-language models (VLMs) in resource-constrained settings demands low latency and high throughput, yet existing compact VLMs often fall short of the inference speedups their smaller parameter counts suggest. To explain this discrepancy, we conduct an empirical end-to-end efficiency analysis and systematically profile inference to identify the dominant bottlenecks. Based on these findings, we develop optimization recipes tailored to compact VLMs that substantially reduce latency while preserving accuracy. These techniques cut time to first token (TTFT) by 53% on InternVL3-2B and by 93% on SmolVLM-256M. Our recipes are broadly applicable across both VLM architectures and common serving frameworks, providing practical guidance for building efficient VLM systems. Beyond efficiency, we study how to extend compact VLMs with structured perception outputs and introduce the resulting model family, ArgusVLM. Across diverse benchmarks, ArgusVLM achieves strong performance while maintaining a compact and efficient design.
>
---
#### [new 107] Does YOLO Really Need to See Every Training Image in Every Epoch?
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决YOLO训练效率低的问题。通过提出AFSS策略，动态选择训练图像，提升训练速度并保持精度。**

- **链接: [https://arxiv.org/pdf/2603.17684](https://arxiv.org/pdf/2603.17684)**

> **作者:** Xingxing Xie; Jiahua Dong; Junwei Han; Gong Cheng
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** YOLO detectors are known for their fast inference speed, yet training them remains unexpectedly time-consuming due to their exhaustive pipeline that processes every training image in every epoch, even when many images have already been sufficiently learned. This stands in clear contrast to the efficiency suggested by the ``You Only Look Once'' philosophy. This naturally raises an important question: \textit{Does YOLO really need to see every training image in every epoch?} To explore this, we propose an Anti-Forgetting Sampling Strategy (AFSS) that dynamically determines which images should be used and which can be skipped during each epoch, allowing the detector to learn more effectively and efficiently. Specifically, AFSS measures the learning sufficiency of each training image as the minimum of its detection recall and precision, and dynamically categorizes training images into easy, medium, or hard levels accordingly. Easy training images are sparsely resampled during training in a continuous review manner, with priority given to those that have not been used for a long time to reduce redundancy and prevent forgetting. Moderate training images are partially selected, prioritizing recently unused ones and randomly choosing the rest from unselected images to ensure coverage and prevent forgetting. Hard training images are fully sampled in every epoch to ensure sufficient learning. The learning sufficiency of each training image is periodically updated, enabling detectors to adaptively shift its focus toward the informative training images over time while progressively discarding redundant ones. On widely used natural image detection benchmarks (MS COCO 2017 and PASCAL VOC 2007) and remote sensing detection datasets (DOTA-v1.0 and DIOR-R), AFSS achieves more than $1.43\times$ training speedup for YOLO-series detectors while also improving accuracy.
>
---
#### [new 108] Prompt-Free Universal Region Proposal Network
- **分类: cs.CV**

- **简介: 该论文提出PF-RPN，解决无需外部提示的通用目标区域建议问题。通过SIA、CSP和CG-QS模块，实现高效目标检测，适用于多种场景。**

- **链接: [https://arxiv.org/pdf/2603.17554](https://arxiv.org/pdf/2603.17554)**

> **作者:** Qihong Tang; Changhan Liu; Shaofeng Zhang; Wenbin Li; Qi Fan; Yang Gao
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Identifying potential objects is critical for object recognition and analysis across various computer vision applications. Existing methods typically localize potential objects by relying on exemplar images, predefined categories, or textual descriptions. However, their reliance on image and text prompts often limits flexibility, restricting adaptability in real-world scenarios. In this paper, we introduce a novel Prompt-Free Universal Region Proposal Network (PF-RPN), which identifies potential objects without relying on external prompts. First, the Sparse Image-Aware Adapter (SIA) module performs initial localization of potential objects using a learnable query embedding dynamically updated with visual features. Next, the Cascade Self-Prompt (CSP) module identifies the remaining potential objects by leveraging the self-prompted learnable embedding, autonomously aggregating informative visual features in a cascading manner. Finally, the Centerness-Guided Query Selection (CG-QS) module facilitates the selection of high-quality query embeddings using a centerness scoring network. Our method can be optimized with limited data (e.g., 5% of MS COCO data) and applied directly to various object detection application domains for identifying potential objects without fine-tuning, such as underwater object detection, industrial defect detection, and remote sensing image object detection. Experimental results across 19 datasets validate the effectiveness of our method. Code is available at this https URL.
>
---
#### [new 109] Anchoring and Rescaling Attention for Semantically Coherent Inbetweening
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于生成式中间帧任务，解决GI模型在稀疏序列中的不一致和语义错位问题。通过关键帧锚定注意力和重缩放时间RoPE提升帧一致性与语义准确性。**

- **链接: [https://arxiv.org/pdf/2603.17651](https://arxiv.org/pdf/2603.17651)**

> **作者:** Tae Eun Choi; Sumin Shim; Junhyeok Kim; Seong Jae Hwang
>
> **备注:** Accepted to CVPR 2026; Code is released at this https URL
>
> **摘要:** Generative inbetweening (GI) seeks to synthesize realistic intermediate frames between the first and last keyframes beyond mere interpolation. As sequences become sparser and motions larger, previous GI models struggle with inconsistent frames with unstable pacing and semantic misalignment. Since GI involves fixed endpoints and numerous plausible paths, this task requires additional guidance gained from the keyframes and text to specify the intended path. Thus, we give semantic and temporal guidance from the keyframes and text onto each intermediate frame through Keyframe-anchored Attention Bias. We also better enforce frame consistency with Rescaled Temporal RoPE, which allows self-attention to attend to keyframes more faithfully. TGI-Bench, the first benchmark specifically designed for text-conditioned GI evaluation, enables challenge-targeted evaluation to analyze GI models. Without additional training, our method achieves state-of-the-art frame consistency, semantic fidelity, and pace stability for both short and long sequences across diverse challenges.
>
---
#### [new 110] TransText: Transparency Aware Image-to-Video Typography Animation
- **分类: cs.CV**

- **简介: 该论文提出TransText，解决图像到视频的文字动画生成问题，通过Alpha-as-RGB方法联合建模外观与透明度，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.17944](https://arxiv.org/pdf/2603.17944)**

> **作者:** Fei Zhang; Zijian Zhou; Bohao Tang; Sen He; Hang Li; Zhe Wang; Soubhik Sanyal; Pengfei Liu; Viktar Atliha; Tao Xiang; Frost Xu; Semih Gunel
>
> **备注:** 19 pages, publication review
>
> **摘要:** We introduce the first method, to the best of our knowledge, for adapting image-to-video models to layer-aware text (glyph) animation, a capability critical for practical dynamic visual design. Existing approaches predominantly handle the transparency-encoding (alpha channel) as an extra latent dimension appended to the RGB space, necessitating the reconstruction of the underlying RGB-centric variational autoencoder (VAE). However, given the scarcity of high-quality transparent glyph data, retraining the VAE is computationally expensive and may erode the robust semantic priors learned from massive RGB corpora, potentially leading to latent pattern mixing. To mitigate these limitations, we propose TransText, a framework based on a novel Alpha-as-RGB paradigm to jointly model appearance and transparency without modifying the pre-trained generative manifold. TransText embeds the alpha channel as an RGB-compatible visual signal through latent spatial concatenation, explicitly ensuring strict cross-modal (RGB-and-Alpha) consistency while preventing feature entanglement. Our experiments demonstrate that TransText significantly outperforms baselines, generating coherent, high-fidelity transparent animations with diverse, fine-grained effects.
>
---
#### [new 111] Eye image segmentation using visual and concept prompts with Segment Anything Model 3 (SAM3)
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，旨在比较SAM3与SAM2在眼图像分割中的表现，并探索SAM3的文本提示效果。结果表明SAM2性能更优且更快。**

- **链接: [https://arxiv.org/pdf/2603.17715](https://arxiv.org/pdf/2603.17715)**

> **作者:** Diederick C. Niehorster; Marcus Nyström
>
> **摘要:** Previous work has reported that vision foundation models show promising zero-shot performance in eye image segmentation. Here we examine whether the latest iteration of the Segment Anything Model, SAM3, offers better eye image segmentation performance than SAM2, and explore the performance of its new concept (text) prompting mode. Eye image segmentation performance was evaluated using diverse datasets encompassing both high-resolution high-quality videos from a lab environment and the TEyeD dataset consisting of challenging eye videos acquired in the wild. Results show that in most cases SAM3 with either visual or concept prompts did not perform better than SAM2, for both lab and in-the-wild datasets. Since SAM2 not only performed better but was also faster, we conclude that SAM2 remains the best option for eye image segmentation. We provide our adaptation of SAM3's codebase that allows processing videos of arbitrary duration.
>
---
#### [new 112] MCoT-MVS: Multi-level Vision Selection by Multi-modal Chain-of-Thought Reasoning for Composed Image Retrieval
- **分类: cs.CV**

- **简介: 该论文针对组合图像检索任务，解决文本修改下从参考图中提取准确语义线索的问题。提出MCoT-MVS方法，结合多模态推理与视觉选择模块，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2603.17360](https://arxiv.org/pdf/2603.17360)**

> **作者:** Xuri Ge; Chunhao Wang; Xindi Wang; Zheyun Qin; Zhumin Chen; Xin Xin
>
> **备注:** Accepted by The Web Conference 2026 (WWW2026)
>
> **摘要:** Composed Image Retrieval (CIR) aims to retrieve target images based on a reference image and modified texts. However, existing methods often struggle to extract the correct semantic cues from the reference image that best reflect the user's intent under textual modification prompts, resulting in interference from irrelevant visual noise. In this paper, we propose a novel Multi-level Vision Selection by Multi-modal Chain-of-Thought Reasoning (MCoT-MVS) for CIR, integrating attention-aware multi-level vision features guided by reasoning cues from a multi-modal large language model (MLLM). Specifically, we leverage an MLLM to perform chain-of-thought reasoning on the multimodal composed input, generating the retained, removed, and target-inferred texts. These textual cues subsequently guide two reference visual attention selection modules to selectively extract discriminative patch-level and instance-level semantics from the reference image. Finally, to effectively fuse these multi-granular visual cues with the modified text and the imagined target description, we design a weighted hierarchical combination module to align the composed query with target images in a unified embedding space. Extensive experiments on two CIR benchmarks, namely CIRR and FashionIQ, demonstrate that our approach consistently outperforms existing methods and achieves new state-of-the-art performance. Code and trained models are publicly released.
>
---
#### [new 113] GenLie: A Global-Enhanced Lie Detection Network under Sparsity and Semantic Interference
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频谎言检测任务，旨在解决稀疏且易受干扰的欺骗信号识别问题。提出GenLie网络，通过全局监督增强局部特征建模，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2603.16935](https://arxiv.org/pdf/2603.16935)**

> **作者:** Zongshun Zhang; Yao Liu; Qiao Liu; Xuefeng Peng; Peiyuan Jiang; Jiaye Yang; Daibing Yao; Wei Lin
>
> **备注:** Accepted to IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2026
>
> **摘要:** Video-based lie detection aims to identify deceptive behaviors from visual cues. Despite recent progress, its core challenge lies in learning sparse yet discriminative representations. Deceptive signals are typically subtle and short-lived, easily overwhelmed by redundant information, while individual and contextual variations introduce strong identity-related noise. To address this issue, we propose GenLie, a Global-Enhanced Lie Detection Network that performs local feature modeling under global supervision. Specifically, sparse and subtle deceptive cues are captured at the local level, while global supervision and optimization ensure robust and discriminative representations by suppressing identity-related noise. Experiments on three public datasets, covering both high- and low-stakes scenarios, show that GenLie consistently outperforms state-of-the-art methods. Source code is available at this https URL.
>
---
#### [new 114] KGS-GCN: Enhancing Sparse Skeleton Sensing via Kinematics-Driven Gaussian Splatting and Probabilistic Topology for Action Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于动作识别任务，旨在解决稀疏骨架数据与固定拓扑结构的问题。提出KGS-GCN，结合运动驱动的高斯点云和概率拓扑，提升时空动态建模能力。**

- **链接: [https://arxiv.org/pdf/2603.16943](https://arxiv.org/pdf/2603.16943)**

> **作者:** Yuhan Chen; Yicui Shi; Guofa Li; Liping Zhang; Jie Li; Jiaxin Gao; Wenbo Chu
>
> **摘要:** Skeleton-based action recognition is widely utilized in sensor systems including human-computer interaction and intelligent surveillance. Nevertheless, current sensor devices typically generate sparse skeleton data as discrete coordinates, which inevitably discards fine-grained spatiotemporal details during highly dynamic movements. Moreover, the rigid constraints of predefined physical sensor topologies hinder the modeling of latent long-range dependencies. To overcome these limitations, we propose KGS-GCN, a graph convolutional network that integrates kinematics-driven Gaussian splatting with probabilistic topology. Our framework explicitly addresses the challenges of sensor data sparsity and topological rigidity by transforming discrete joints into continuous generative representations. Firstly, a kinematics-driven Gaussian splatting module is designed to dynamically construct anisotropic covariance matrices using instantaneous joint velocity vectors. This module enhances visual representation by rendering sparse skeleton sequences into multi-view continuous heatmaps rich in spatiotemporal semantics. Secondly, to transcend the limitations of fixed physical connections, a probabilistic topology construction method is proposed. This approach generates an adaptive prior adjacency matrix by quantifying statistical correlations via the Bhattacharyya distance between joint Gaussian distributions. Ultimately, the GCN backbone is adaptively modulated by the rendered visual features via a visual context gating mechanism. Empirical results demonstrate that KGS-GCN significantly enhances the modeling of complex spatiotemporal dynamics. By addressing the inherent limitations of sparse inputs, our framework offers a robust solution for processing low-fidelity sensor data. This approach establishes a practical pathway for improving perceptual reliability in real-world sensing applications.
>
---
#### [new 115] Towards Motion-aware Referring Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于Referring Image Segmentation任务，解决现有方法在运动相关查询上表现不佳的问题。通过数据增强和多模态对比学习提升模型对运动描述的识别能力。**

- **链接: [https://arxiv.org/pdf/2603.17413](https://arxiv.org/pdf/2603.17413)**

> **作者:** Chaeyun Kim; Seunghoon Yi; Yejin Kim; Yohan Jo; Joonseok Lee
>
> **备注:** Accepted at AISTATS 2026. * Equal contribution
>
> **摘要:** Referring Image Segmentation (RIS) requires identifying objects from images based on textual descriptions. We observe that existing methods significantly underperform on motion-related queries compared to appearance-based ones. To address this, we first introduce an efficient data augmentation scheme that extracts motion-centric phrases from original captions, exposing models to more motion expressions without additional annotations. Second, since the same object can be described differently depending on the context, we propose Multimodal Radial Contrastive Learning (MRaCL), performed on fused image-text embeddings rather than unimodal representations. For comprehensive evaluation, we introduce a new test split focusing on motion-centric queries, and introduce a new benchmark called M-Bench, where objects are distinguished primarily by actions. Extensive experiments show our method substantially improves performance on motion-centric queries across multiple RIS models, maintaining competitive results on appearance-based descriptions. Codes are available at this https URL
>
---
#### [new 116] Mutually Causal Semantic Distillation Network for Zero-Shot Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于零样本学习任务，旨在解决视觉与属性特征间语义知识迁移的问题。提出MSDN++模型，通过双向因果注意力机制，提升语义表示的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.17412](https://arxiv.org/pdf/2603.17412)**

> **作者:** Shiming Chen; Shuhuang Chen; Guo-Sen Xie; Xinge You
>
> **备注:** Accepted to IJCV. arXiv admin note: text overlap with arXiv:2203.03137
>
> **摘要:** Zero-shot learning (ZSL) aims to recognize the unseen classes in the open-world guided by the side-information (e.g., attributes). Its key task is how to infer the latent semantic knowledge between visual and attribute features on seen classes, and thus conducting a desirable semantic knowledge transfer from seen classes to unseen ones. Prior works simply utilize unidirectional attention within a weakly-supervised manner to learn the spurious and limited latent semantic representations, which fail to effectively discover the intrinsic semantic knowledge (e.g., attribute semantic) between visual and attribute features. To solve the above challenges, we propose a mutually causal semantic distillation network (termed MSDN++) to distill the intrinsic and sufficient semantic representations for ZSL. MSDN++ consists of an attribute$\rightarrow$visual causal attention sub-net that learns attribute-based visual features, and a visual$\rightarrow$attribute causal attention sub-net that learns visual-based attribute features. The causal attentions encourages the two sub-nets to learn causal vision-attribute associations for representing reliable features with causal visual/attribute learning. With the guidance of semantic distillation loss, the two mutual attention sub-nets learn collaboratively and teach each other throughout the training process. Extensive experiments on three widely-used benchmark datasets (e.g., CUB, SUN, AWA2, and FLO) show that our MSDN++ yields significant improvements over the strong baselines, leading to new state-of-the-art performances.
>
---
#### [new 117] TrackDeform3D: Markerless and Autonomous 3D Keypoint Tracking and Dataset Collection for Deformable Objects
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D关键点跟踪任务，解决变形物体数据集构建与轨迹跟踪难题。提出TrackDeform3D框架，利用RGB-D相机自主采集高质量数据，提升跟踪精度与数据一致性。**

- **链接: [https://arxiv.org/pdf/2603.17068](https://arxiv.org/pdf/2603.17068)**

> **作者:** Yeheng Zong; Yizhou Chen; Alexander Bowler; Chia-Tung Yang; Ram Vasudevan
>
> **摘要:** Structured 3D representations such as keypoints and meshes offer compact, expressive descriptions of deformable objects, jointly capturing geometric and topological information useful for downstream tasks such as dynamics modeling and motion planning. However, robustly extracting such representations remains challenging, as current perception methods struggle to handle complex deformations. Moreover, large-scale 3D data collection remains a bottleneck: existing approaches either require prohibitive data collection efforts, such as labor-intensive annotation or expensive motion capture setups, or rely on simplifying assumptions that break down in unstructured environments. As a result, large-scale 3D datasets and benchmarks for deformable objects remain scarce. To address these challenges, this paper presents an affordable and autonomous framework for collecting 3D datasets of deformable objects using only RGB-D cameras. The proposed method identifies 3D keypoints and robustly tracks their trajectories, incorporating motion consistency constraints to produce temporally smooth and geometrically coherent data. TrackDeform3D is evaluated against several state-of-the-art tracking methods across diverse object categories and demonstrates consistent improvements in both geometric and tracking accuracy. Using this framework, this paper presents a high-quality, large-scale dataset consisting of 6 deformable objects, totaling 110 minutes of trajectory data.
>
---
#### [new 118] Edge-Efficient Two-Stream Multimodal Architecture for Non-Intrusive Bathroom Fall Detection
- **分类: cs.CV**

- **简介: 该论文属于非侵入式跌倒检测任务，旨在解决浴室环境中老年人跌倒的隐私与实时检测问题。提出一种双流架构，结合雷达与振动信号，提升检测精度并降低能耗。**

- **链接: [https://arxiv.org/pdf/2603.17069](https://arxiv.org/pdf/2603.17069)**

> **作者:** Haitian Wang; Yiren Wang; Xinyu Wang; Sheldon Fung; Atif Mansoor
>
> **备注:** This paper has been accepted for poster presenation at IEEE ICME 2026
>
> **摘要:** Falls in wet bathroom environments are a major safety risk for seniors living alone. Recent work has shown that mmWave-only, vibration-only, and existing multimodal schemes, such as vibration-triggered radar activation, early feature concatenation, and decision-level score fusion, can support privacy-preserving, non-intrusive fall detection. However, these designs still treat motion and impact as loosely coupled streams, depending on coarse temporal alignment and amplitude thresholds, and do not explicitly encode the causal link between radar-observed collapse and floor impact or address timing drift, object drop confounders, and latency and energy constraints on low-power edge devices. To this end, we propose a two-stream architecture that encodes radar signals with a Motion--Mamba branch for long-range motion patterns and processes floor vibration with an Impact--Griffin branch that emphasizes impact transients and cross-axis coupling. Cross-conditioned fusion uses low-rank bilinear interaction and a Switch--MoE head to align motion and impact tokens and suppress object-drop confounders. The model keeps inference cost suitable for real-time execution on a Raspberry Pi 4B gateway. We construct a bathroom fall detection benchmark dataset with frame-level annotations, comprising more than 3~h of synchronized mmWave radar and triaxial vibration recordings across eight scenarios under running water, together with subject-independent training, validation, and test splits. On the test split, our model attains 96.1% accuracy, 94.8% precision, 88.0% recall, a 91.1% macro F1 score, and an AUC of 0.968. Compared with the strongest baseline, it improves accuracy by 2.0 percentage points and fall recall by 1.3 percentage points, while reducing latency from 35.9 ms to 15.8 ms and lowering energy per 2.56 s window from 14200 mJ to 10750 mJ on the Raspberry Pi 4B gateway.
>
---
#### [new 119] TAPESTRY: From Geometry to Appearance via Consistent Turntable Videos
- **分类: cs.CV**

- **简介: 该论文提出TAPESTRY，解决3D模型无纹理时生成一致外观的问题，通过几何条件视频扩散生成高质量转台视频，用于后续纹理合成与重建。**

- **链接: [https://arxiv.org/pdf/2603.17735](https://arxiv.org/pdf/2603.17735)**

> **作者:** Yan Zeng; Haoran Jiang; Kaixin Yao; Qixuan Zhang; Longwen Zhang; Lan Xu; Jingyi Yu
>
> **摘要:** Automatically generating photorealistic and self-consistent appearances for untextured 3D models is a critical challenge in digital content creation. The advancement of large-scale video generation models offers a natural approach: directly synthesizing 360-degree turntable videos (TTVs), which can serve not only as high-quality dynamic previews but also as an intermediate representation to drive texture synthesis and neural rendering. However, existing general-purpose video diffusion models struggle to maintain strict geometric consistency and appearance stability across the full range of views, making their outputs ill-suited for high-quality 3D reconstruction. To this end, we introduce TAPESTRY, a framework for generating high-fidelity TTVs conditioned on explicit 3D geometry. We reframe the 3D appearance generation task as a geometry-conditioned video diffusion problem: given a 3D mesh, we first render and encode multi-modal geometric features to constrain the video generation process with pixel-level precision, thereby enabling the creation of high-quality and consistent TTVs. Building upon this, we also design a method for downstream reconstruction tasks from the TTV input, featuring a multi-stage pipeline with 3D-Aware Inpainting. By rotating the model and performing a context-aware secondary generation, this pipeline effectively completes self-occluded regions to achieve full surface coverage. The videos generated by TAPESTRY are not only high-quality dynamic previews but also serve as a reliable, 3D-aware intermediate representation that can be seamlessly back-projected into UV textures or used to supervise neural rendering methods like 3DGS. This enables the automated creation of production-ready, complete 3D assets from untextured meshes. Experimental results demonstrate that our method outperforms existing approaches in both video consistency and final reconstruction quality.
>
---
#### [new 120] Shot-Aware Frame Sampling for Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决长视频采样中难以平衡全局覆盖与关键事件的问题。提出InfoShot方法，通过分段采样提升异常检测和视频问答性能。**

- **链接: [https://arxiv.org/pdf/2603.17374](https://arxiv.org/pdf/2603.17374)**

> **作者:** Mengyu Zhao; Di Fu; Yongyu Xie; Jiaxing Zhang; Zhigang Yuan; Shirin Jalali; Yong Cao
>
> **摘要:** Video frame sampling is essential for efficient long-video understanding with Vision-Language Models (VLMs), since dense inputs are costly and often exceed context limits. Yet when only a small number of frames can be retained, existing samplers often fail to balance broad video coverage with brief but critical events, which can lead to unreliable downstream predictions. To address this issue, we present InfoShot, a task-agnostic, shot-aware frame sampler for long-video understanding. InfoShot first partitions a video into semantically consistent shots, and then selects two complementary keyframes from each shot: one to represent the main content and one to capture unusual within-shot changes. This design is guided by an information-theoretic objective that encourages the sampled set to retain high information about both shot structure and sparse within-shot deviations. In this way, it improves the chance of preserving both overall video context and short decision-critical moments without requiring any retraining. To better evaluate such short-lived events, we further introduce SynFlash, a synthetic benchmark with controllable sub-second anomaly patterns and frame-level ground truth, and we also evaluate InfoShot on existing anomaly datasets and general video understanding tasks. Experiments show that InfoShot improves anomaly hit rate and downstream Video-QA accuracy under frame number constraints, while matching or outperforming strong baselines on standard video understanding benchmarks.
>
---
#### [new 121] Look Where It Matters: High-Resolution Crops Retrieval for Efficient VLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出AwaRes框架，解决VLMs在高分辨率图像处理中的准确率与效率矛盾，通过按需检索关键区域提升性能。**

- **链接: [https://arxiv.org/pdf/2603.16932](https://arxiv.org/pdf/2603.16932)**

> **作者:** Nimrod Shabtay; Moshe Kimhi; Artem Spector; Sivan Haray; Ehud Rivlin; Chaim Baskin; Raja Giryes; Eli Schwartz
>
> **摘要:** Vision-language models (VLMs) typically process images at a native high-resolution, forcing a trade-off between accuracy and computational efficiency: high-resolution inputs capture fine details but incur significant computational costs, while low-resolution inputs advocate for efficiency, they potentially miss critical visual information, like small text. We present AwaRes, a spatial-on-demand framework that resolves this accuracy-efficiency trade-off by operating on a low-resolution global view and using tool-calling to retrieve only high-resolution segments needed for a given query. We construct supervised data automatically: a judge compares low- vs.\ high-resolution answers to label whether cropping is needed, and an oracle grounding model localizes the evidence for the correct answer, which we map to a discrete crop set to form multi-turn tool-use trajectories. We train our framework with cold-start SFT followed by multi-turn GRPO with a composite reward that combines semantic answer correctness with explicit crop-cost penalties. Project page: this https URL
>
---
#### [new 122] AdapTS: Lightweight Teacher-Student Approach for Multi-Class and Continual Visual Anomaly Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉异常检测任务，解决多类别和持续学习场景下的工业检测问题。提出AdapTS框架，通过轻量教师-学生结构实现高效边缘部署。**

- **链接: [https://arxiv.org/pdf/2603.17530](https://arxiv.org/pdf/2603.17530)**

> **作者:** Manuel Barusco; Davide Dalle Pezze; Francesco Borsatti; Gian Antonio Susto
>
> **摘要:** Visual Anomaly Detection (VAD) is crucial for industrial inspection, yet most existing methods are limited to single-category scenarios, failing to address the multi-class and continual learning demands of real-world environments. While Teacher-Student (TS) architectures are efficient, they remain unexplored for the Continual Setting. To bridge this gap, we propose AdapTS, a unified TS framework designed for multi-class and continual settings, optimized for edge deployment. AdapTS eliminates the need for two different architectures by utilizing a single shared frozen backbone and injecting lightweight trainable adapters into the student pathway. Training is enhanced via a segmentation-guided objective and synthetic Perlin noise, while a prototype-based task identification mechanism dynamically selects adapters at inference with 99\% accuracy. Experiments on MVTec AD and VisA demonstrate that AdapTS matches the performance of existing TS methods across multi-class and continual learning scenarios, while drastically reducing memory overhead. Our lightest variant, AdapTS-S, requires only 8 MB of additional memory, 13x less than STFPM (95 MB), 48x less than RD4AD (360 MB), and 149x less than DeSTSeg (1120 MB), making it a highly scalable solution for edge deployment in complex industrial environments.
>
---
#### [new 123] PaAgent: Portrait-Aware Image Restoration Agent via Subjective-Objective Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，旨在解决现有修复代理缺乏历史总结机制导致的搜索效率低问题。提出PaAgent，结合自进化画像库和RAG技术提升修复效果。**

- **链接: [https://arxiv.org/pdf/2603.17055](https://arxiv.org/pdf/2603.17055)**

> **作者:** Yijian Wang; Qingsen Yan; Jiantao Zhou; Duwei Dai; Wei Dong
>
> **摘要:** Image Restoration (IR) agents, leveraging multimodal large language models to perceive degradation and invoke restoration tools, have shown promise in automating IR tasks. However, existing IR agents typically lack an insight summarization mechanism for past interactions, which results in an exhaustive search for the optimal IR tool. To address this limitation, we propose a portrait-aware IR agent, dubbed PaAgent, which incorporates a self-evolving portrait bank for IR tools and Retrieval-Augmented Generation (RAG) to select a suitable IR tool for input. Specifically, to construct and evolve the portrait bank, the PaAgent continuously enriches it by summarizing the characteristics of various IR tools with restored images, selected IR tools, and degraded images. In addition, the RAG is employed to select the optimal IR tool for the input image by retrieving relevant insights from the portrait bank. Furthermore, to enhance PaAgent's ability to perceive degradation in complex scenes, we propose a subjective-objective reinforcement learning strategy that considers both image quality scores and semantic insights in reward generation, which accurately provides the degradation information even under partial and non-uniform degradation. Extensive experiments across 8 IR benchmarks, covering six single-degradation and eight mixed-degradation scenarios, validate PaAgent's superiority in addressing complex IR tasks. Our project page is \href{this https URL}{PaAgent}.
>
---
#### [new 124] Evidence Packing for Cross-Domain Image Deepfake Detection with LVLMs
- **分类: cs.CV**

- **简介: 该论文属于图像深度伪造检测任务，旨在解决LVLM在该任务中需 costly fine-tuning 且泛化能力差的问题。提出SCEP框架，通过证据驱动推理实现无需微调的高效检测。**

- **链接: [https://arxiv.org/pdf/2603.17761](https://arxiv.org/pdf/2603.17761)**

> **作者:** Yuxin Liu; Fei Wang; Kun Li; Yiqi Nie; Junjie Chen; Zhangling Duan; Zhaohong Jia
>
> **摘要:** Image Deepfake Detection (IDD) separates manipulated images from authentic ones by spotting artifacts of synthesis or tampering. Although large vision-language models (LVLMs) offer strong image understanding, adapting them to IDD often demands costly fine-tuning and generalizes poorly to diverse, evolving manipulations. We propose the Semantic Consistent Evidence Pack (SCEP), a training-free LVLM framework that replaces whole-image inference with evidence-driven reasoning. SCEP mines a compact set of suspicious patch tokens that best reveal manipulation cues. It uses the vision encoder's CLS token as a global reference, clusters patch features into coherent groups, and scores patches with a fused metric combining CLS-guided semantic mismatch with frequency-and noise-based anomalies. To cover dispersed traces and avoid redundancy, SCEP samples a few high-confidence patches per cluster and applies grid-based NMS, producing an evidence pack that conditions a frozen LVLM for prediction. Experiments on diverse benchmarks show SCEP outperforms strong baselines without LVLM fine-tuning.
>
---
#### [new 125] ACE-LoRA: Graph-Attentive Context Enhancement for Parameter-Efficient Adaptation of Medical Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于医学视觉-语言模型任务，旨在解决通用模型与专用模型间的性能平衡问题。提出ACE-LoRA框架，通过参数高效微调增强模型的零样本泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.17079](https://arxiv.org/pdf/2603.17079)**

> **作者:** M. Arda Aydın; Melih B. Yilmaz; Aykut Koç; Tolga Çukur
>
> **摘要:** The success of CLIP-like vision-language models (VLMs) on natural images has inspired medical counterparts, yet existing approaches largely fall into two extremes: specialist models trained on single-domain data, which capture domain-specific details but generalize poorly, and generalist medical VLMs trained on multi-domain data, which retain broad semantics but dilute fine-grained diagnostic cues. Bridging this specialization-generalization trade-off remains challenging. To address this problem, we propose ACE-LoRA, a parameter-efficient adaptation framework for generalist medical VLMs that maintains robust zero-shot generalization. ACE-LoRA integrates Low-Rank Adaptation (LoRA) modules into frozen image-text encoders and introduces an Attention-based Context Enhancement Hypergraph Neural Network (ACE-HGNN) module that captures higher-order contextual interactions beyond pairwise similarity to enrich global representations with localized diagnostic cues, addressing a key limitation of prior Parameter-Efficient Fine-Tuning (PEFT) methods that overlook fine-grained details. To further enhance cross-modal alignment, we formulate a label-guided InfoNCE loss to effectively suppress false negatives between semantically related image-text pairs. Despite adding only 0.95M trainable parameters, ACE-LoRA consistently outperforms state-of-the-art medical VLMs and PEFT baselines across zero-shot classification, segmentation, and detection benchmarks spanning multiple domains. Our code is available at this https URL.
>
---
#### [new 126] Video Understanding: From Geometry and Semantics to Unified Models
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决动态视觉场景的感知与推理问题。文章综述了视频理解的几何、语义及统一模型，提出统一建模范式以应对多样化任务需求。**

- **链接: [https://arxiv.org/pdf/2603.17840](https://arxiv.org/pdf/2603.17840)**

> **作者:** Zhaochong An; Zirui Li; Mingqiao Ye; Feng Qiao; Jiaang Li; Zongwei Wu; Vishal Thengane; Chengzu Li; Lei Li; Luc Van Gool; Guolei Sun; Serge Belongie
>
> **备注:** A comprehensive survey of video understanding, spanning low-level geometry, high-level semantics, and unified understanding models
>
> **摘要:** Video understanding aims to enable models to perceive, reason about, and interact with the dynamic visual world. In contrast to image understanding, video understanding inherently requires modeling temporal dynamics and evolving visual context, placing stronger demands on spatiotemporal reasoning and making it a foundational problem in computer vision. In this survey, we present a structured overview of video understanding by organizing the literature into three complementary perspectives: low-level video geometry understanding, high-level semantic understanding, and unified video understanding models. We further highlight a broader shift from isolated, task-specific pipelines toward unified modeling paradigms that can be adapted to diverse downstream objectives, enabling a more systematic view of recent progress. By consolidating these perspectives, this survey provides a coherent map of the evolving video understanding landscape, summarizes key modeling trends and design principles, and outlines open challenges toward building robust, scalable, and unified video foundation models.
>
---
#### [new 127] Continual Multimodal Egocentric Activity Recognition via Modality-Aware Novel Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态第一视角活动识别任务，旨在解决开放世界中持续学习与新活动检测问题。提出MAND框架，通过模态感知方法提升新颖性检测和分类准确率。**

- **链接: [https://arxiv.org/pdf/2603.16970](https://arxiv.org/pdf/2603.16970)**

> **作者:** Wonseon Lim; Hyejeong Im; Dae-Won Kim
>
> **摘要:** Multimodal egocentric activity recognition integrates visual and inertial cues for robust first-person behavior understanding. However, deploying such systems in open-world environments requires detecting novel activities while continuously learning from non-stationary streams. Existing methods rely on the main logits for novelty scoring, without fully exploiting the complementary evidence available from individual modalities. Because these logits are often dominated by RGB, cues from other modalities, particularly IMU, remain underutilized, and this imbalance worsens over time under catastrophic forgetting. To address this, we propose MAND, a modality-aware framework for multimodal egocentric open-world continual learning. At inference, Modality-aware Adaptive Scoring (MoAS) estimates sample-wise modality reliability from energy scores and adaptively integrates modality logits to better exploit complementary modality cues for novelty detection. During training, Modality-wise Representation Stabilization Training (MoRST) preserves modality-specific discriminability across tasks via auxiliary heads and modality-wise logit distillation. Experiments on a public multimodal egocentric benchmark show that MAND improves novel activity detection AUC by up to 10\% and known-class classification accuracy by up to 2.8\% over state-of-the-art baselines.
>
---
#### [new 128] ChopGrad: Pixel-Wise Losses for Latent Video Diffusion via Truncated Backpropagation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出ChopGrad，解决视频扩散模型训练内存过高的问题，通过截断反向传播实现高效像素级损失优化，适用于多种视频生成任务。**

- **链接: [https://arxiv.org/pdf/2603.17812](https://arxiv.org/pdf/2603.17812)**

> **作者:** Dmitriy Rivkin; Parker Ewen; Lili Gao; Julian Ost; Stefanie Walz; Rasika Kangutkar; Mario Bijelic; Felix Heide
>
> **摘要:** Recent video diffusion models achieve high-quality generation through recurrent frame processing where each frame generation depends on previous frames. However, this recurrent mechanism means that training such models in the pixel domain incurs prohibitive memory costs, as activations accumulate across the entire video sequence. This fundamental limitation also makes fine-tuning these models with pixel-wise losses computationally intractable for long or high-resolution videos. This paper introduces ChopGrad, a truncated backpropagation scheme for video decoding, limiting gradient computation to local frame windows while maintaining global consistency. We provide a theoretical analysis of this approximation and show that it enables efficient fine-tuning with frame-wise losses. ChopGrad reduces training memory from scaling linearly with the number of video frames (full backpropagation) to constant memory, and compares favorably to existing state-of-the-art video diffusion models across a suite of conditional video generation tasks with pixel-wise losses, including video super-resolution, video inpainting, video enhancement of neural-rendered scenes, and controlled driving video generation.
>
---
#### [new 129] LaDe: Unified Multi-Layered Graphic Media Generation and Decomposition
- **分类: cs.CV**

- **简介: 该论文提出LaDe框架，解决多层图形媒体生成与分解问题。通过自然语言生成可编辑的多层设计，提升文本与图层对齐效果。**

- **链接: [https://arxiv.org/pdf/2603.17965](https://arxiv.org/pdf/2603.17965)**

> **作者:** Vlad-Constantin Lungu-Stan; Ionut Mironica; Mariana-Iuliana Georgescu
>
> **备注:** 18 pages (main + supp)
>
> **摘要:** Media design layer generation enables the creation of fully editable, layered design documents such as posters, flyers, and logos using only natural language prompts. Existing methods either restrict outputs to a fixed number of layers or require each layer to contain only spatially continuous regions, causing the layer count to scale linearly with design complexity. We propose LaDe (Layered Media Design), a latent diffusion framework that generates a flexible number of semantically meaningful layers. LaDe combines three components: an LLM-based prompt expander that transforms a short user intent into structured per-layer descriptions that guide the generation, a Latent Diffusion Transformer with a 4D RoPE positional encoding mechanism that jointly generates the full media design and its constituent RGBA layers, and an RGBA VAE that decodes each layer with full alpha-channel support. By conditioning on layer samples during training, our unified framework supports three tasks: text-to-image generation, text-to-layers media design generation, and media design decomposition. We compare LaDe to Qwen-Image-Layered on text-to-layers and image-to-layers tasks on the Crello test set. LaDe outperforms Qwen-Image-Layered in text-to-layers generation by improving text-to-layer alignment, as validated by two VLM-as-a-judge evaluators (GPT-4o mini and Qwen3-VL).
>
---
#### [new 130] Robust-ComBat: Mitigating Outlier Effects in Diffusion MRI Data Harmonization
- **分类: cs.CV**

- **简介: 该论文属于医学影像数据标准化任务，旨在解决扩散MRI数据中病理异常值导致的偏差问题。通过改进ComBat方法，提出Robust-ComBat以提升数据和谐化效果。**

- **链接: [https://arxiv.org/pdf/2603.17968](https://arxiv.org/pdf/2603.17968)**

> **作者:** Yoan David; Pierre-Marc Jodoin; Alzheimer's Disease Neuroimaging Initiative; TRACK-TBI Investigators
>
> **备注:** 20 pages, 8 figures
>
> **摘要:** Harmonization methods such as ComBat and its variants are widely used to mitigate diffusion MRI (dMRI) site-specific biases. However, ComBat assumes that subject distributions exhibit a Gaussian profile. In practice, patients with neurological disorders often present diffusion metrics that deviate markedly from those of healthy controls, introducing pathological outliers that distort site-effect estimation. This problem is particularly challenging in clinical practice as most patients undergoing brain imaging have an underlying and yet undiagnosed condition, making it difficult to exclude them from harmonization cohorts, as their scans were precisely prescribed to establish a diagnosis. In this paper, we show that harmonizing data to a normative reference population with ComBat while including pathological cases induces significant distortions. Across 7 neurological conditions, we evaluated 10 outlier rejection methods with 4 ComBat variants over a wide range of scenarios, revealing that many filtering strategies fail in the presence of pathology. In contrast, a simple MLP provides robust outlier compensation enabling reliable harmonization while preserving disease-related signal. Experiments on both control and real multi-site cohorts, comprising up to 80% of subjects with neurological disorders, demonstrate that Robust-ComBat consistently outperforms conventional statistical baselines with lower harmonization error across all ComBat variants.
>
---
#### [new 131] Patient4D: Temporally Consistent Patient Body Mesh Recovery from Monocular Operating Room Video
- **分类: cs.CV**

- **简介: 该论文属于人体姿态与形状重建任务，解决手术室视频中因遮挡和相机移动导致的3D人体网格恢复难题。提出Patient4D方法，利用静态先验和时间一致性机制提升重建稳定性。**

- **链接: [https://arxiv.org/pdf/2603.17178](https://arxiv.org/pdf/2603.17178)**

> **作者:** Mingxiao Tu; Hoijoon Jung; Alireza Moghadam; Andre Kyme; Jinman Kim
>
> **摘要:** Recovering a dense 3D body mesh from monocular video remains challenging under occlusion from draping and continuously moving camera viewpoints. This configuration arises in surgical augmented reality (AR), where an anesthetized patient lies under surgical draping while a surgeon's head-mounted camera continuously changes viewpoint. Existing human mesh recovery (HMR) methods are typically trained on upright, moving subjects captured from relatively stable cameras, leading to performance degradation under such conditions. To address this, we present Patient4D, a stationarity-constrained reconstruction pipeline that explicitly exploits the stationarity prior. The pipeline combines image-level foundation models for perception with lightweight geometric mechanisms that enforce temporal consistency across frames. Two key components enable robust reconstruction: Pose Locking, which anchors pose parameters using stable keyframes, and Rigid Fallback, which recovers meshes under severe occlusion through silhouette-guided rigid alignment. Together, these mechanisms stabilize predictions while remaining compatible with off-the-shelf HMR models. We evaluate Patient4D on 4,680 synthetic surgical sequences and three public HMR video benchmarks. Under surgical drape occlusion, Patient4D achieves a 0.75 mean IoU, reducing failure frames from 30.5% to 1.3% compared to the best baseline. Our findings demonstrate that exploiting stationarity priors can substantially improve monocular reconstruction in clinical AR scenarios.
>
---
#### [new 132] AgriChat: A Multimodal Large Language Model for Agriculture Image Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出AgriChat，一个用于农业图像理解的多模态大语言模型，解决农业数据不足和领域知识验证问题，构建了AgriMM基准并展示其优越性能。**

- **链接: [https://arxiv.org/pdf/2603.16934](https://arxiv.org/pdf/2603.16934)**

> **作者:** Abderrahmene Boudiaf; Irfan Hussain; Sajid Javed
>
> **摘要:** The deployment of Multimodal Large Language Models (MLLMs) in agriculture is currently stalled by a critical trade-off: the existing literature lacks the large-scale agricultural datasets required for robust model development and evaluation, while current state-of-the-art models lack the verified domain expertise necessary to reason across diverse taxonomies. To address these challenges, we propose the Vision-to-Verified-Knowledge (V2VK) pipeline, a novel generative AI-driven annotation framework that integrates visual captioning with web-augmented scientific retrieval to autonomously generate the AgriMM benchmark, effectively eliminating biological hallucinations by grounding training data in verified phytopathological literature. The AgriMM benchmark contains over 3,000 agricultural classes and more than 607k VQAs spanning multiple tasks, including fine-grained plant species identification, plant disease symptom recognition, crop counting, and ripeness assessment. Leveraging this verifiable data, we present AgriChat, a specialized MLLM that presents broad knowledge across thousands of agricultural classes and provides detailed agricultural assessments with extensive explanations. Extensive evaluation across diverse tasks, datasets, and evaluation conditions reveals both the capabilities and limitations of current agricultural MLLMs, while demonstrating AgriChat's superior performance over other open-source models, including internal and external benchmarks. The results validate that preserving visual detail combined with web-verified knowledge constitutes a reliable pathway toward robust and trustworthy agricultural AI. The code and dataset are publicly available at this https URL .
>
---
#### [new 133] Astrolabe: Steering Forward-Process Reinforcement Learning for Distilled Autoregressive Video Models
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决蒸馏自回归视频模型与人类视觉偏好不一致的问题。提出Astrolabe框架，通过在线强化学习提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.17051](https://arxiv.org/pdf/2603.17051)**

> **作者:** Songchun Zhang; Zeyue Xue; Siming Fu; Jie Huang; Xianghao Kong; Y Ma; Haoyang Huang; Nan Duan; Anyi Rao
>
> **备注:** 53 pages, 37 figures
>
> **摘要:** Distilled autoregressive (AR) video models enable efficient streaming generation but frequently misalign with human visual preferences. Existing reinforcement learning (RL) frameworks are not naturally suited to these architectures, typically requiring either expensive re-distillation or solver-coupled reverse-process optimization that introduces considerable memory and computational overhead. We present Astrolabe, an efficient online RL framework tailored for distilled AR models. To overcome existing bottlenecks, we introduce a forward-process RL formulation based on negative-aware fine-tuning. By contrasting positive and negative samples directly at inference endpoints, this approach establishes an implicit policy improvement direction without requiring reverse-process unrolling. To scale this alignment to long videos, we propose a streaming training scheme that generates sequences progressively via a rolling KV-cache, applying RL updates exclusively to local clip windows while conditioning on prior context to ensure long-range coherence. Finally, to mitigate reward hacking, we integrate a multi-reward objective stabilized by uncertainty-aware selective regularization and dynamic reference updates. Extensive experiments demonstrate that our method consistently enhances generation quality across multiple distilled AR video models, serving as a robust and scalable alignment solution.
>
---
#### [new 134] Material Magic Wand: Material-Aware Grouping of 3D Parts in Untextured Meshes
- **分类: cs.CV**

- **简介: 该论文提出Material Magic Wand工具，解决无纹理网格中基于材质的3D部件分组问题，通过材料感知嵌入实现自动分组。**

- **链接: [https://arxiv.org/pdf/2603.17370](https://arxiv.org/pdf/2603.17370)**

> **作者:** Umangi Jain; Vladimir Kim; Matheus Gadelha; Igor Gilitschenski; Zhiqin Chen
>
> **备注:** Project Page: this https URL
>
> **摘要:** We introduce the problem of material-aware part grouping in untextured meshes. Many real-world shapes, such as scales of pinecones or windows of buildings, contain repeated structures that share the same material but exhibit geometric variations. When assigning materials to such meshes, these repeated parts often require piece-by-piece manual identification and selection, which is tedious and time-consuming. To address this, we propose Material Magic Wand, a tool that allows artists to select part groups based on their estimated material properties -- when one part is selected, our algorithm automatically retrieves all other parts likely to share the same material. The key component of our approach is a part encoder that generates a material-aware embedding for each 3D part, accounting for both local geometry and global context. We train our model with a supervised contrastive loss that brings embeddings of material-consistent parts closer while separating those of different materials; therefore, part grouping can be achieved by retrieving embeddings that are close to the embedding of the selected part. To benchmark this task, we introduce a curated dataset of 100 shapes with 241 part-level queries. We verify the effectiveness of our method through extensive experiments and demonstrate its practical value in an interactive material assignment application.
>
---
#### [new 135] Face anonymization preserving facial expressions and photometric realism
- **分类: cs.CV**

- **简介: 该论文属于人脸匿名化任务，旨在保护隐私同时保留表情和光照一致性。提出一种新框架，结合面部关键点和轻量后处理模块，提升匿名化效果。**

- **链接: [https://arxiv.org/pdf/2603.17567](https://arxiv.org/pdf/2603.17567)**

> **作者:** Luigi Celona; Simone Bianco; Raimondo Schettini
>
> **摘要:** The widespread sharing of face images on social media platforms and in large-scale datasets raises pressing privacy concerns, as biometric identifiers can be exploited without consent. Face anonymization seeks to generate realistic facial images that irreversibly conceal the subject's identity while preserving their usefulness for downstream tasks. However, most existing generative approaches focus on identity removal and image realism, often neglecting facial expressions as well as photometric consistency -- specifically attributes such as illumination and skin tone -- that are critical for applications like relighting, color constancy, and medical or affective analysis. In this work, we propose a feature-preserving anonymization framework that extends DeepPrivacy by incorporating dense facial landmarks to better retain expressions, and by introducing lightweight post-processing modules that ensure consistency in lighting direction and skin color. We further establish evaluation metrics specifically designed to quantify expression fidelity, lighting consistency, and color preservation, complementing standard measures of image realism, pose accuracy, and re-identification resistance. Experiments on the CelebA-HQ dataset demonstrate that our method produces anonymized faces with improved realism and significantly higher fidelity in expression, illumination, and skin tone compared to state-of-the-art baselines. These results underscore the importance of feature-aware anonymization as a step toward more useful, fair, and trustworthy privacy-preserving facial data.
>
---
#### [new 136] Behavior-Centric Extraction of Scenarios from Highway Traffic Data and their Domain-Knowledge-Guided Clustering using CVQ-VAE
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于交通场景提取与聚类任务，解决场景可比性和聚类可解释性问题。提出基于领域知识的场景提取与聚类方法，提升自动化驾驶验证效率。**

- **链接: [https://arxiv.org/pdf/2603.16964](https://arxiv.org/pdf/2603.16964)**

> **作者:** Niklas Roßberg; Sinan Hasirlioglu; Mohamed Essayed Bouzouraa; Wolfgang Utschick; Michael Botsch
>
> **备注:** Accepted as a conference paper in IEEE Intelligent Vehicles Symposium (IV) 2026, Detroit, MI, United States
>
> **摘要:** Approval of ADS depends on evaluating its behavior within representative real-world traffic scenarios. A common way to obtain such scenarios is to extract them from real-world data recordings. These can then be grouped and serve as basis on which the ADS is subsequently tested. This poses two central challenges: how scenarios are extracted and how they are grouped. Existing extraction methods rely on heterogeneous definitions, hindering scenario comparability. For the grouping of scenarios, rule-based or ML-based methods can be utilized. However, while modern ML-based approaches can handle the complexity of traffic scenarios, unlike rule-based approaches, they lack interpretability and may not align with domain-knowledge. This work contributes to a standardized scenario extraction based on the Scenario-as-Specification concept, as well as a domain-knowledge-guided scenario clustering process. Experiments on the highD dataset demonstrate that scenarios can be extracted reliably and that domain-knowledge can be effectively integrated into the clustering process. As a result, the proposed methodology supports a more standardized process for deriving scenario categories from highway data recordings and thus enables a more efficient validation process of automated vehicles.
>
---
#### [new 137] EchoGen: Cycle-Consistent Learning for Unified Layout-Image Generation and Understanding
- **分类: cs.CV**

- **简介: 该论文提出EchoGen，解决布局到图像生成与图像定位任务。通过联合训练和渐进策略提升模型性能，实现高效且准确的图像生成与理解。**

- **链接: [https://arxiv.org/pdf/2603.18001](https://arxiv.org/pdf/2603.18001)**

> **作者:** Kai Zou; Hongbo Liu; Dian Zheng; Jianxiong Gao; Zhiwei Zhao; Bin Liu
>
> **备注:** 9 pages, Accepted at the 40th AAAI Conference on Artificial Intelligence (AAAI 2026)
>
> **摘要:** In this work, we present EchoGen, a unified framework for layout-to-image generation and image grounding, capable of generating images with accurate layouts and high fidelity to text descriptions (e.g., spatial relationships), while grounding the image robustly at the same time. We believe that image grounding possesses strong text and layout understanding abilities, which can compensate for the corresponding limitations in layout-to-image generation. At the same time, images generated from layouts exhibit high diversity in content, thereby enhancing the robustness of image grounding. Jointly training both tasks within a unified model can promote performance improvements for each. However, we identify that this joint training paradigm encounters several optimization challenges and results in restricted performance. To address these issues, we propose progressive training strategies. First, the Parallel Multi-Task Pre-training (PMTP) stage equips the model with basic abilities for both tasks, leveraging shared tokens to accelerate training. Next, the Dual Joint Optimization (DJO) stage exploits task duality to sequentially integrate the two tasks, enabling unified optimization. Finally, the Cycle RL stage eliminates reliance on visual supervision by using consistency constraints as rewards, significantly enhancing the model's unified capabilities via the GRPO strategy. Extensive experiments demonstrate state-of-the-art results on both layout-to-image generation and image grounding benchmarks, and reveal clear synergistic gains from optimizing the two tasks together.
>
---
#### [new 138] LLM-Powered Flood Depth Estimation from Social Media Imagery: A Vision-Language Model Framework with Mechanistic Interpretability for Transportation Resilience
- **分类: cs.CV**

- **简介: 该论文属于洪水深度估计任务，旨在解决实时、高精度交通洪水监测问题。通过构建FloodLlama模型和TikTok数据管道，实现街道级洪水深度的准确预测与解释。**

- **链接: [https://arxiv.org/pdf/2603.17108](https://arxiv.org/pdf/2603.17108)**

> **作者:** Nafis Fuad; Xiaodong Qian
>
> **摘要:** Urban flooding poses an escalating threat to transportation network continuity, yet no operational system currently provides real-time, street-level flood depth information at the centimeter resolution required for dynamic routing, electric vehicle (EV) safety, and autonomous vehicle (AV) operations. This study presents FloodLlama, a fine-tuned open-source vision-language model (VLM) for continuous flood depth estimation from single street-level images, supported by a multimodal sensing pipeline using TikTok data. A synthetic dataset of approximately 190000 images was generated, covering seven vehicle types, four weather conditions, and 41 depth levels (0-40 cm at 1 cm resolution). Progressive curriculum training enabled coarse-to-fine learning, while LLaMA 3.2-11B Vision was fine-tuned using QLoRA. Evaluation across 34797 trials reveals a depth-dependent prompt effect: simple prompts perform better for shallow flooding, whereas chain-of-thought (CoT) reasoning improves performance at greater depths. FloodLlama achieves a mean absolute error (MAE) below 0.97 cm and Acc@5cm above 93.7% for deep flooding, exceeding 96.8% for shallow depths. A five-phase mechanistic interpretability framework identifies layer L23 as the critical depth-encoding transition and enables selective fine-tuning that reduces trainable parameters by 76-80% while maintaining accuracy. The Tier 3 configuration achieves 98.62% accuracy on real-world data and shows strong robustness under visual occlusion. A TikTok-based data pipeline, validated on 676 annotated flood frames from Detroit, demonstrates the feasibility of real-time, crowd-sourced flood sensing. The proposed framework provides a scalable, infrastructure-free solution with direct implications for EV safety, AV deployment, and resilient transportation management.
>
---
#### [new 139] Facial beauty prediction fusing transfer learning and broad learning system
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于面部美观预测任务，旨在解决数据不足和模型泛化差的问题。通过融合迁移学习与广义学习系统，提出E-BLS和ER-BLS模型，提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2603.16930](https://arxiv.org/pdf/2603.16930)**

> **作者:** Junying Gan; Xiaoshan Xie; Yikui Zhai; Guohui He; Chaoyun Mai; Heng Luo
>
> **摘要:** Facial beauty prediction (FBP) is an important and challenging problem in the fields of computer vision and machine learning. Not only it is easily prone to overfitting due to the lack of large-scale and effective data, but also difficult to quickly build robust and effective facial beauty evaluation models because of the variability of facial appearance and the complexity of human perception. Transfer Learning can be able to reduce the dependence on large amounts of data as well as avoid overfitting problems. Broad learning system (BLS) can be capable of quickly completing models building and training. For this purpose, Transfer Learning was fused with BLS for FBP in this paper. Firstly, a feature extractor is constructed by way of CNNs models based on transfer learning for facial feature extraction, in which EfficientNets are used in this paper, and the fused features of facial beauty extracted are transferred to BLS for FBP, called E-BLS. Secondly, on the basis of E-BLS, a connection layer is designed to connect the feature extractor and BLS, called ER-BLS. Finally, experimental results show that, compared with the previous BLS and CNNs methods existed, the accuracy of FBP was improved by E-BLS and ER-BLS, demonstrating the effectiveness and superiority of the method presented, which can also be widely used in pattern recognition, object detection and image classification.
>
---
#### [new 140] Loc3R-VLM: Language-based Localization and 3D Reasoning with Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出Loc3R-VLM，解决视觉-语言模型在3D空间理解与定位上的不足。通过结合场景布局重建和视角建模，提升模型的3D推理能力。**

- **链接: [https://arxiv.org/pdf/2603.18002](https://arxiv.org/pdf/2603.18002)**

> **作者:** Kevin Qu; Haozhe Qi; Mihai Dusmanu; Mahdi Rad; Rui Wang; Marc Pollefeys
>
> **备注:** Project Page: this https URL
>
> **摘要:** Multimodal Large Language Models (MLLMs) have made impressive progress in connecting vision and language, but they still struggle with spatial understanding and viewpoint-aware reasoning. Recent efforts aim to augment the input representations with geometric cues rather than explicitly teaching models to reason in 3D space. We introduce Loc3R-VLM, a framework that equips 2D Vision-Language Models with advanced 3D understanding capabilities from monocular video input. Inspired by human spatial cognition, Loc3R-VLM relies on two joint objectives: global layout reconstruction to build a holistic representation of the scene structure, and explicit situation modeling to anchor egocentric perspective. These objectives provide direct spatial supervision that grounds both perception and language in a 3D context. To ensure geometric consistency and metric-scale alignment, we leverage lightweight camera pose priors extracted from a pre-trained 3D foundation model. Loc3R-VLM achieves state-of-the-art performance in language-based localization and outperforms existing 2D- and video-based approaches on situated and general 3D question-answering benchmarks, demonstrating that our spatial supervision framework enables strong 3D understanding. Project page: this https URL
>
---
#### [new 141] Toward Phonology-Guided Sign Language Motion Generation: A Diffusion Baseline and Conditioning Analysis
- **分类: cs.CV**

- **简介: 该论文属于手势语言生成任务，旨在通过文本生成自然的3D手语动作。研究提出扩散模型基线，并分析语音属性条件对生成效果的影响。**

- **链接: [https://arxiv.org/pdf/2603.17388](https://arxiv.org/pdf/2603.17388)**

> **作者:** Rui Hong; Jana Kosecka
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Generating natural, correct, and visually smooth 3D avatar sign language motion conditioned on the text inputs continues to be very challenging. In this work, we train a generative model of 3D body motion and explore the role of phonological attribute conditioning for sign language motion generation, using ASL-LEX 2.0 annotations such as hand shape, hand location and movement. We first establish a strong diffusion baseline using an Human Motion MDM-style diffusion model with SMPL-X representation, which outperforms SignAvatar, a state-of-the-art CVAE method, on gloss discriminability metrics. We then systematically study the role of text conditioning using different text encoders (CLIP vs. T5), conditioning modes (gloss-only vs. gloss+phonological attributes), and attribute notation format (symbolic vs. natural language). Our analysis reveals that translating symbolic ASL-LEX notations to natural language is a necessary condition for effective CLIP-based attribute conditioning, while T5 is largely unaffected by this translation. Furthermore, our best-performing variant (CLIP with mapped attributes) outperforms SignAvatar across all metrics. These findings highlight input representation as a critical factor for text-encoder-based attribute conditioning, and motivate structured conditioning approaches where gloss and phonological attributes are encoded through independent pathways.
>
---
#### [new 142] UAV-CB: A Complex-Background RGB-T Dataset and Local Frequency Bridge Network for UAV Detection
- **分类: cs.CV**

- **简介: 该论文属于无人机检测任务，解决复杂背景和伪装下的检测难题。构建了UAV-CB数据集，并提出LFBNet网络提升检测性能。**

- **链接: [https://arxiv.org/pdf/2603.17492](https://arxiv.org/pdf/2603.17492)**

> **作者:** Shenghui Huang; Menghao Hu; Longkun Zou; Hongyu Chi; Zekai Li; Feng Gao; Fan Yang; Qingyao Wu; Ke Chen
>
> **摘要:** Detecting Unmanned Aerial Vehicles (UAVs) in low-altitude environments is essential for perception and defense systems but remains highly challenging due to complex backgrounds, camouflage, and multimodal interference. In real-world scenarios, UAVs are frequently visually blended with surrounding structures such as buildings, vegetation, and power lines, resulting in low contrast, weak boundaries, and strong confusion with cluttered background textures. Existing UAV detection datasets, though diverse, are not specifically designed to capture these camouflage and complex-background challenges, which limits progress toward robust real-world perception. To fill this gap, we construct UAV-CB, a new RGB-T UAV detection dataset deliberately curated to emphasize complex low-altitude backgrounds and camouflage characteristics. Furthermore, we propose the Local Frequency Bridge Network (LFBNet), which models features in localized frequency space to bridge both the frequency-spatial fusion gap and the cross-modality discrepancy gap in RGB-T fusion. Extensive experiments on UAV-CB and public benchmarks demonstrate that LFBNet achieves state-of-the-art detection performance and strong robustness under camouflaged and cluttered conditions, offering a frequency-aware perspective on multimodal UAV perception in real-world applications.
>
---
#### [new 143] Solution for 10th Competition on Ambivalence/Hesitancy (AH) Video Recognition Challenge using Divergence-Based Multimodal Fusion
- **分类: cs.CV**

- **简介: 该论文属于视频情感识别任务，解决A/H视频识别问题。提出基于分歧的多模态融合方法，通过测量视觉、音频和文本间的冲突提升识别效果。**

- **链接: [https://arxiv.org/pdf/2603.16939](https://arxiv.org/pdf/2603.16939)**

> **作者:** Aislan Gabriel O. Souza; Agostinho Freire; Leandro Honorato Silva; Igor Lucas B. da Silva; João Vinícius R. de Andrade; Gabriel C. de Albuquerque; Lucas Matheus da S. Oliveira; Mário Stela Guerra; Luciana Machado
>
> **摘要:** We address the Ambivalence/Hesitancy (A/H) Video Recognition Challenge at the 10th ABAW Competition (CVPR 2026). We propose a divergence-based multimodal fusion that explicitly measures cross-modal conflict between visual, audio, and textual channels. Visual features are encoded as Action Units (AUs) extracted via Py-Feat, audio via Wav2Vec 2.0, and text via BERT. Each modality is processed by a BiLSTM with attention pooling and projected into a shared embedding space. The fusion module computes pairwise absolute differences between modality embeddings, directly capturing the incongruence that characterizes A/H. On the BAH dataset, our approach achieves a Macro F1 of 0.6808 on the validation test set, outperforming the challenge baseline of 0.2827. Statistical analysis across 1{,}132 videos confirms that temporal variability of AUs is the dominant visual discriminator of A/H.
>
---
#### [new 144] Unified Spatio-Temporal Token Scoring for Efficient Video VLMs
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对视频视觉语言模型（VLM）的效率问题，提出STTS方法，在不依赖文本条件的情况下，统一剪枝视觉令牌，提升计算效率。**

- **链接: [https://arxiv.org/pdf/2603.18004](https://arxiv.org/pdf/2603.18004)**

> **作者:** Jianrui Zhang; Yue Yang; Rohun Tripathi; Winson Han; Ranjay Krishna; Christopher Clark; Yong Jae Lee; Sangho Lee
>
> **摘要:** Token pruning is essential for enhancing the computational efficiency of vision-language models (VLMs), particularly for video-based tasks where temporal redundancy is prevalent. Prior approaches typically prune tokens either (1) within the vision transformer (ViT) exclusively for unimodal perception tasks such as action recognition and object segmentation, without adapting to downstream vision-language tasks; or (2) only within the LLM while leaving the ViT output intact, often requiring complex text-conditioned token selection mechanisms. In this paper, we introduce Spatio-Temporal Token Scoring (STTS), a simple and lightweight module that prunes vision tokens across both the ViT and the LLM without text conditioning or token merging, and is fully compatible with end-to-end training. By learning how to score temporally via an auxiliary loss and spatially via LLM downstream gradients, aided by our efficient packing algorithm, STTS prunes 50% of vision tokens throughout the entire architecture, resulting in a 62% improvement in efficiency during both training and inference with only a 0.7% drop in average performance across 13 short and long video QA tasks. Efficiency gains increase with more sampled frames per video. Applying test-time scaling for long-video QA further yields performance gains of 0.5-1% compared to the baseline. Overall, STTS represents a novel, simple yet effective technique for unified, architecture-wide vision token pruning.
>
---
#### [new 145] From Drop-off to Recovery: A Mechanistic Analysis of Segmentation in MLLMs
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究MLLM在图像分割任务中的表现，分析其视觉理解机制。通过分层探针和注意力干预，揭示了适配器导致的分割性能下降及LLM层的逐步恢复过程。**

- **链接: [https://arxiv.org/pdf/2603.17228](https://arxiv.org/pdf/2603.17228)**

> **作者:** Boyong Wu; Sanghwan Kim; Zeynep Akata
>
> **摘要:** Multimodal Large Language Models (MLLMs) are increasingly applied to pixel-level vision tasks, yet their intrinsic capacity for spatial understanding remains poorly understood. We investigate segmentation capacity through a layerwise linear probing evaluation across the entire MLLM pipeline: vision encoder, adapter, and LLM. We further conduct an intervention based attention knockout analysis to test whether cross-token attention progressively refines visual representations, and an evaluation of bidirectional attention among image tokens on spatial consistency. Our analysis reveals that the adapter introduces a segmentation representation drop-off, but LLM layers progressively recover through attention-mediated refinement, where correctly classified tokens steer misclassified neighbors toward the correct label. At early image token positions, this recovery is bounded by causal attention, which bidirectional attention among image tokens alleviates. These findings provide a mechanistic account of how MLLMs process visual information for segmentation, informing the design of future segmentation-capable models.
>
---
#### [new 146] Illumination-Aware Contactless Fingerprint Spoof Detection via Paired Flash-Non-Flash Imaging
- **分类: cs.CV**

- **简介: 该论文属于生物特征识别中的防欺骗检测任务，旨在解决无接触指纹识别中伪造攻击的检测问题。通过配对闪光与非闪光图像，分析光照引起的差异特征，提升 spoof 检测的鲁棒性与可解释性。**

- **链接: [https://arxiv.org/pdf/2603.17679](https://arxiv.org/pdf/2603.17679)**

> **作者:** Roja Sahoo; Anoop Namboodiri
>
> **备注:** Accepted at IWBF 2026 (14th International Workshop on Biometrics and Forensics)
>
> **摘要:** Contactless fingerprint recognition enables hygienic and convenient biometric authentication but poses new challenges for spoof detection due to the absence of physical contact and traditional liveness cues. Most existing methods rely on single-image acquisition and appearance-based features, which often generalize poorly across devices, capture conditions, and spoof materials. In this work, we study paired flash-non-flash contactless fingerprint acquisition as a lightweight active sensing mechanism for spoof detection. Through a preliminary empirical analysis, we show that flash illumination accentuates material- and structure-dependent properties, including ridge visibility, subsurface scattering, micro-geometry, and surface oils, while non-flash images provide a baseline appearance context. We analyze lighting-induced differences using interpretable metrics such as inter-channel correlation, specular reflection characteristics, texture realism, and differential imaging. These complementary features help discriminate genuine fingerprints from printed, digital, and molded presentation attacks. We further examine the limitations of paired acquisition, including sensitivity to imaging settings, dataset scale, and emerging high-fidelity spoofs. Our findings demonstrate the potential of illumination-aware analysis to improve robustness and interpretability in contactless fingerprint presentation attack detection, motivating future work on paired acquisition and physics-informed feature design. Code is available in the repository.
>
---
#### [new 147] EvoGuard: An Extensible Agentic RL-based Framework for Practical and Evolving AI-Generated Image Detection
- **分类: cs.CV**

- **简介: 该论文属于AI生成图像检测任务，旨在解决传统方法在泛化性和可扩展性上的不足。提出EvoGuard框架，利用多模态大语言模型和非MLLM检测器，通过强化学习实现动态协作与自主决策。**

- **链接: [https://arxiv.org/pdf/2603.17343](https://arxiv.org/pdf/2603.17343)**

> **作者:** Chenyang Zhu; Maorong Wang; Jun Liu; Ching-Chun Chang; Isao Echizen
>
> **摘要:** The rapid proliferation of AI-Generated Images (AIGIs) has introduced severe risks of misinformation, making AIGI detection a critical yet challenging task. While traditional detection paradigms mainly rely on low-level features, recent research increasingly focuses on leveraging the general understanding ability of Multimodal Large Language Models (MLLMs) to achieve better generalization, but still suffer from limited extensibility and expensive training data annotations. To better address complex and dynamic real-world environments, we propose EvoGuard, a novel agentic framework for AIGI detection. It encapsulates various state-of-the-art (SOTA) off-the-shelf MLLM and non-MLLM detectors as callable tools, and coordinates them through a capability-aware dynamic orchestration mechanism. Empowered by the agent's capacities for autonomous planning and reflection, it intelligently selects suitable tools for given samples, reflects intermediate results, and decides the next action, reaching a final conclusion through multi-turn invocation and reasoning. This design effectively exploits the complementary strengths among heterogeneous detectors, transcending the limits of any single model. Furthermore, optimized by a GRPO-based Agentic Reinforcement Learning algorithm using only low-cost binary labels, it eliminates the reliance on fine-grained annotations. Extensive experiments demonstrate that EvoGuard achieves SOTA accuracy while mitigating the bias between positive and negative samples. More importantly, it allows the plug-and-play integration of new detectors to boost overall performance in a train-free manner, offering a highly practical, long-term solution to ever-evolving AIGI threats. Source code will be publicly available upon acceptance.
>
---
#### [new 148] GazeOnce360: Fisheye-Based 360° Multi-Person Gaze Estimation with Global-Local Feature Fusion
- **分类: cs.CV**

- **简介: 该论文属于多人体注视方向估计任务，解决从全景鱼眼相机中准确估计多人3D注视方向的问题。提出GazeOnce360模型，结合全局与局部特征融合，提升估计精度。**

- **链接: [https://arxiv.org/pdf/2603.17161](https://arxiv.org/pdf/2603.17161)**

> **作者:** Zhuojiang Cai; Zhenghui Sun; Feng Lu
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** We present GazeOnce360, a novel end-to-end model for multi-person gaze estimation from a single tabletop-mounted upward-facing fisheye camera. Unlike conventional approaches that rely on forward-facing cameras in constrained viewpoints, we address the underexplored setting of estimating the 3D gaze direction of multiple people distributed across a 360° scene from an upward fisheye perspective. To support research in this setting, we introduce MPSGaze360, a large-scale synthetic dataset rendered using Unreal Engine, featuring diverse multi-person configurations with accurate 3D gaze and eye landmark annotations. Our model tackles the severe distortion and perspective variation inherent in fisheye imagery by incorporating rotational convolutions and eye landmark supervision. To better capture fine-grained eye features crucial for gaze estimation, we propose a dual-resolution architecture that fuses global low-resolution context with high-resolution local eye regions. Experimental results demonstrate the effectiveness of each component in our model. This work highlights the feasibility and potential of fisheye-based 360° gaze estimation in practical multi-person scenarios. Project page: this https URL.
>
---
#### [new 149] PhysQuantAgent: An Inference Pipeline of Mass Estimation for Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于物理属性估计任务，旨在解决VLMs在真实场景中质量推断能力不足的问题。提出PhysQuantAgent框架和VisPhysQuant数据集，通过视觉提示提升质量估计精度。**

- **链接: [https://arxiv.org/pdf/2603.16958](https://arxiv.org/pdf/2603.16958)**

> **作者:** Hisayuki Yokomizo; Taiki Miyanishi; Yan Gang; Shuhei Kurita; Nakamasa Inoue; Yusuke Iwasawa
>
> **备注:** Code and dataset will be available at this https URL
>
> **摘要:** Vision-Language Models (VLMs) are increasingly applied to robotic perception and manipulation, yet their ability to infer physical properties required for manipulation remains limited. In particular, estimating the mass of real-world objects is essential for determining appropriate grasp force and ensuring safe interaction. However, current VLMs lack reliable mass reasoning capabilities, and most existing benchmarks do not explicitly evaluate physical quantity estimation under realistic sensing conditions. In this work, we propose PhysQuantAgent, a framework for real-world object mass estimation using VLMs, together with VisPhysQuant, a new benchmark dataset for evaluation. VisPhysQuant consists of RGB-D videos of real objects captured from multiple viewpoints, annotated with precise mass measurements. To improve estimation accuracy, we introduce three visual prompting methods that enhance the input image with object detection, scale estimation, and cross-sectional image generation to help the model comprehend the size and internal structure of the target object. Experiments show that visual prompting significantly improves mass estimation accuracy on real-world data, suggesting the efficacy of integrating spatial reasoning with VLM knowledge for physical inference.
>
---
#### [new 150] TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于文本生成安全任务，旨在解决概念擦除失效问题。通过TINA攻击，证明现有方法仅掩盖概念，需直接作用于视觉知识。**

- **链接: [https://arxiv.org/pdf/2603.17828](https://arxiv.org/pdf/2603.17828)**

> **作者:** Qianlong Xiang; Miao Zhang; Haoyu Zhang; Kun Wang; Junhui Hou; Liqiang Nie
>
> **备注:** 16 pages, accepted by CVPR 2026
>
> **摘要:** Although text-to-image diffusion models exhibit remarkable generative power, concept erasure techniques are essential for their safe deployment to prevent the creation of harmful content. This has fostered a dynamic interplay between the development of erasure defenses and the adversarial probes designed to bypass them, and this co-evolution has progressively enhanced the efficacy of erasure methods. However, this adversarial co-evolution has converged on a narrow, text-centric paradigm that equates erasure with severing the text-to-image mapping, ignoring that the underlying visual knowledge related to undesired concepts still persist. To substantiate this claim, we investigate from a visual perspective, leveraging DDIM inversion to probe whether a generative pathway for the erased concept can still be found. However, identifying such a visual generative pathway is challenging because standard text-guided DDIM inversion is actively resisted by text-centric defenses within the erased model. To address this, we introduce TINA, a novel Text-free INversion Attack, which enforces this visual-only probe by operating under a null-text condition, thereby avoiding existing text-centric defenses. Moreover, TINA integrates an optimization procedure to overcome the accumulating approximation errors that arise when standard inversion operates without its usual textual guidance. Our experiments demonstrate that TINA regenerates erased concepts from models treated with state-of-the-art unlearning. The success of TINA proves that current methods merely obscure concepts, highlighting an urgent need for paradigms that operate directly on internal visual knowledge.
>
---
#### [new 151] PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery
- **分类: cs.CV**

- **简介: 该论文提出PanoVGGT，解决全景图像的3D重建任务，通过Transformer框架联合预测相机位姿、深度图和点云，提升全景图像的几何建模能力。**

- **链接: [https://arxiv.org/pdf/2603.17571](https://arxiv.org/pdf/2603.17571)**

> **作者:** Yijing Guo; Mengjun Chao; Luo Wang; Tianyang Zhao; Haizhao Dai; Yingliang Zhang; Jingyi Yu; Yujiao Shi
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Panoramic imagery offers a full 360° field of view and is increasingly common in consumer devices. However, it introduces non-pinhole distortions that challenge joint pose estimation and 3D reconstruction. Existing feed-forward models, built for perspective cameras, generalize poorly to this setting. We propose PanoVGGT, a permutation-equivariant Transformer framework that jointly predicts camera poses, depth maps, and 3D point clouds from one or multiple panoramas in a single forward pass. The model incorporates spherical-aware positional embeddings and a panorama-specific three-axis SO(3) rotation augmentation, enabling effective geometric reasoning in the spherical domain. To resolve inherent global-frame ambiguity, we further introduce a stochastic anchoring strategy during training. In addition, we contribute PanoCity, a large-scale outdoor panoramic dataset with dense depth and 6-DoF pose annotations. Extensive experiments on PanoCity and standard benchmarks demonstrate that PanoVGGT achieves competitive accuracy, strong robustness, and improved cross-domain generalization. Code and dataset will be released.
>
---
#### [new 152] Trust the Unreliability: Inward Backward Dynamic Unreliability Driven Coreset Selection for Medical Image Classification
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决大规模数据下模型效率与准确率的平衡问题。通过引入动态不可靠性驱动的样本选择策略，提升模型对决策边界的建模能力。**

- **链接: [https://arxiv.org/pdf/2603.17603](https://arxiv.org/pdf/2603.17603)**

> **作者:** Yan Liang; Ziyuan Yang; Zhuxin Lei; Mengyu Sun; Yingyu Chen; Yi Zhang
>
> **摘要:** Efficiently managing and utilizing large-scale medical imaging datasets with limited resources presents significant challenges. While coreset selection helps reduce computational costs, its effectiveness in medical data remains limited due to inherent complexity, such as large intra-class variation and high inter-class similarity. To address this, we revisit the training process and observe that neural networks consistently produce stable confidence predictions and better remember samples near class centers in training. However, concentrating on these samples may complicate the modeling of decision boundaries. Hence, we argue that the more unreliable samples are, in fact, the more informative in helping build the decision boundary. Based on this, we propose the Dynamic Unreliability-Driven Coreset Selection(DUCS) strategy. Specifically, we introduce an inward-backward unreliability assessment perspective: 1) Inward Self-Awareness: The model introspects its behavior by analyzing the evolution of confidence during training, thereby quantifying uncertainty of each sample. 2) Backward Memory Tracking: The model reflects on its training tracking by tracking the frequency of forgetting samples, thus evaluating its retention ability for each sample. Next, we select unreliable samples that exhibit substantial confidence fluctuations and are repeatedly forgotten during training. This selection process ensures that the chosen samples are near the decision boundary, thereby aiding the model in refining the boundary. Extensive experiments on public medical datasets demonstrate our superior performance compared to state-of-the-art(SOTA) methods, particularly at high compression rates.
>
---
#### [new 153] Joint Degradation-Aware Arbitrary-Scale Super-Resolution for Variable-Rate Extreme Image Compression
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像压缩任务，解决超低比特率下重建质量差和模型适应性不足的问题。提出ASSR-EIC框架，结合任意尺度超分辨率与联合退化感知，实现灵活比特率控制和自适应重建。**

- **链接: [https://arxiv.org/pdf/2603.17408](https://arxiv.org/pdf/2603.17408)**

> **作者:** Xinning Chai; Zhengxue Cheng; Xin Li; Rong Xie; Li Song
>
> **备注:** Accepted by IEEE Transactions on BroadCasting
>
> **摘要:** Recent diffusion-based extreme image compression methods have demonstrated remarkable performance at ultra-low bitrates. However, most approaches require training separate diffusion models for each target bitrate, resulting in substantial computational overhead and hindering practical deployment. Meanwhile, recent studies have shown that joint super-resolution can serve as an effective approach for enhancing low-bitrate reconstruction. However, when moving toward ultra-low bitrate regimes, these methods struggle due to severe information loss, and their reliance on fixed super-resolution scales prevents flexible adaptation across diverse bitrates. To address these limitations, we propose ASSR-EIC, a novel image compression framework that leverages arbitrary-scale super-resolution (ASSR) to support variable-rate extreme image compression (EIC). An arbitrary-scale downsampling module is introduced at the encoder side to provide controllable rate reduction, while a diffusion-based, joint degradation-aware ASSR decoder enables rate-adaptive reconstruction within a single model. We exploit the compression- and rescaling-aware diffusion prior to guide the reconstruction, yielding high fidelity and high realism restoration across diverse compression and rescaling settings. Specifically, we design a global compression-rescaling adaptor that offers holistic guidance for rate adaptation, and a local compression-rescaling modulator that dynamically balances generative and fidelity-oriented behaviors to achieve fine-grained, bitrate-adaptive detail restoration. To further enhance reconstruction quality, we introduce a dual semantic-enhanced design. Extensive experiments demonstrate that ASSR-EIC delivers state-of-the-art performance in extreme image compression while simultaneously supporting flexible bitrate control and adaptive rate-dependent reconstruction.
>
---
#### [new 154] AERR-Nav: Adaptive Exploration-Recovery-Reminiscing Strategy for Zero-Shot Object Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于零样本目标导航任务，旨在解决未知多层环境中机器人导航困难的问题。提出AERR-Nav框架，通过自适应策略提升导航效果。**

- **链接: [https://arxiv.org/pdf/2603.17712](https://arxiv.org/pdf/2603.17712)**

> **作者:** Jingzhi Huang; Junkai Huang; Haoyang Yang; Haoang Li; Yi Wang
>
> **摘要:** Zero-Shot Object Navigation (ZSON) in unknown multi-floor environments presents a significant challenge. Recent methods, mostly based on semantic value greedy waypoint selection, spatial topology-enhanced memory, and Multimodal Large Language Model (MLLM) as a decision-making framework, have led to improvements. However, these architectures struggle to balance exploration and exploitation for ZSON when encountering unseen environments, especially in multi-floor settings, such as robots getting stuck at narrow intersections, endlessly wandering, or failing to find stair entrances. To overcome these challenges, we propose AERR-Nav, a Zero-Shot Object Navigation framework that dynamically adjusts its state based on the robot's environment. Specifically, AERR-Nav has the following two key advantages: (1) An Adaptive Exploration-Recovery-Reminiscing Strategy, enables robots to dynamically transition between three states, facilitating specialized responses to diverse navigation scenarios. (2) An Adaptive Exploration State featuring Fast and Slow-Thinking modes helps robots better balance exploration, exploitation, and higher-level reasoning based on evolving environmental information. Extensive experiments on the HM3D and MP3D benchmarks demonstrate that our AERR-Nav achieves state-of-the-art performance among zero-shot methods. Comprehensive ablation studies further validate the efficacy of our proposed strategy and modules.
>
---
#### [new 155] Structured SIR: Efficient and Expressive Importance-Weighted Inference for High-Dimensional Image Registration
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文针对高维图像配准任务，解决传统方法在不确定性建模上的不足，提出Structured SIR方法，实现高效且表达能力强的不确定性量化。**

- **链接: [https://arxiv.org/pdf/2603.17415](https://arxiv.org/pdf/2603.17415)**

> **作者:** Ivor J. A. Simpson; Neill D. F. Campbell
>
> **摘要:** Image registration is an ill-posed dense vision task, where multiple solutions achieve similar loss values, motivating probabilistic inference. Variational inference has previously been employed to capture these distributions, however restrictive assumptions about the posterior form can lead to poor characterisation, overconfidence and low-quality samples. More flexible posteriors are typically bottlenecked by the complexity of high-dimensional covariance matrices required for dense 3D image registration. In this work, we present a memory and computationally efficient inference method, Structured SIR, that enables expressive, multi-modal, characterisation of uncertainty with high quality samples. We propose the use of a Sampled Importance Resampling (SIR) algorithm with a novel memory-efficient high-dimensional covariance parameterisation as the sum of a low-rank covariance and a sparse, spatially structured Cholesky precision factor. This structure enables capturing complex spatial correlations while remaining computationally tractable. We evaluate the efficacy of this approach in 3D dense image registration of brain MRI data, which is a very high-dimensional problem. We demonstrate that our proposed methods produces uncertainty estimates that are significantly better calibrated than those produced by variational methods, achieving equivalent or better accuracy. Crucially, we show that the model yields highly structured multi-modal posterior distributions, enable effective and efficient uncertainty quantification.
>
---
#### [new 156] Visual SLAM with DEM Anchoring for Lunar Surface Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉SLAM任务，旨在解决月球表面长期导航中的定位漂移问题。通过融合数字高程模型（DEM）约束，提升导航精度与地图一致性。**

- **链接: [https://arxiv.org/pdf/2603.17229](https://arxiv.org/pdf/2603.17229)**

> **作者:** Adam Dai; Guillem Casadesus Vila; Grace Gao
>
> **备注:** Accepted to IEEE Aerospace Conference 2026
>
> **摘要:** Future lunar missions will require autonomous rovers capable of traversing tens of kilometers across challenging terrain while maintaining accurate localization and producing globally consistent maps. However, the absence of global positioning systems, extreme illumination, and low-texture regolith make long-range navigation on the Moon particularly difficult, as visual-inertial odometry pipelines accumulate drift over extended traverses. To address this challenge, we present a stereo visual simultaneous localization and mapping (SLAM) system that integrates learned feature detection and matching with global constraints from digital elevation models (DEMs). Our front-end employs learning-based feature extraction and matching to achieve robustness to illumination extremes and repetitive terrain, while the back-end incorporates DEM-derived height and surface-normal factors into a pose graph, providing absolute surface constraints that mitigate long-term drift. We validate our approach using both simulated lunar traverse data generated in Unreal Engine and real Moon/Mars analog data collected from Mt. Etna. Results demonstrate that DEM anchoring consistently reduces absolute trajectory error compared to baseline SLAM methods, lowering drift in long-range navigation even in repetitive or visually aliased terrain.
>
---
#### [new 157] Do Understanding and Generation Fight? A Diagnostic Study of DPO for Unified Multimodal Models
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究统一多模态模型中理解与生成任务在DPO训练中的冲突问题，通过实验发现生成质量难以提升，分析梯度差异为主要原因。**

- **链接: [https://arxiv.org/pdf/2603.17044](https://arxiv.org/pdf/2603.17044)**

> **作者:** Abinav Rao; Sujan Rachuri
>
> **备注:** 8 pages, CVPR MMF5M Workshop 2026
>
> **摘要:** Unified multimodal models share a language model backbone for both understanding and generating images. Can DPO align both capabilities simultaneously? We present the first systematic study of this question, applying DPO to Janus-Pro at 1B and 7B parameters under seven training strategies and two post-hoc methods. The central finding is negative: generation quality resists DPO alignment across all tested conditions on this architecture. No method improves generation CLIPScore at 7B (|Delta| < 0.2, p > 0.5 at n=200 per seed, 3 seeds); at 1B, all methods degrade generation, and the result holds across preference data types (real-vs-generated and model-vs-model) and the data volumes tested (150-288 pairs). Gradient analysis reveals why: understanding and generation gradients are near-orthogonal (cos ~ 0) with ~11-14x magnitude imbalance driven by VQ token count asymmetry (576 generation tokens vs. ~30-100 text tokens). This imbalance is the dominant interference mechanism in multi-task DPO; magnitude-balancing yields directionally positive understanding deltas (+0.01-0.04 VQA, though individually not significant), but the generation gap persists regardless. We identify discrete VQ tokenization as a likely structural bottleneck -- supported by the generation DPO loss converging to ln(2) -- and provide practical guidance for practitioners working with VQ-based unified models.
>
---
#### [new 158] Facial Movement Dynamics Reveal Workload During Complex Multitasking
- **分类: cs.HC; cs.CV**

- **简介: 该论文属于认知负荷监测任务，旨在解决传统方法成本高、侵入性强的问题。通过分析面部动态运动，提出一种低成本的实时监测方法。**

- **链接: [https://arxiv.org/pdf/2603.17767](https://arxiv.org/pdf/2603.17767)**

> **作者:** Carter Sale; Melissa N. Stolar; Gaurav Patil; Michael J. Gostelow; Julia Wallier; Margaret C. Macpherson; Jan-Louis Kruger; Mark Dras; Simon G. Hosking; Rachel W. Kallen; Michael J. Richardson
>
> **备注:** 26 pages, 7 figures, under review at Royal Society Open Science
>
> **摘要:** Real-time cognitive workload monitoring is crucial in safety-critical environments, yet established measures are intrusive, expensive, or lack temporal resolution. We tested whether facial movement dynamics from a standard webcam could provide a low-cost alternative. Seventy-two participants completed a multitasking simulation (OpenMATB) under varied load while facial keypoints were tracked via OpenPose. Linear kinematics (velocity, acceleration, displacement) and recurrence quantification features were extracted. Increasing load altered dynamics across timescales: movement magnitudes rose, temporal organisation fragmented then reorganised into complex patterns, and eye-head coordination weakened. Random forest classifiers trained on pose kinematics outperformed task performance metrics (85% vs. 55% accuracy) but generalised poorly across participants (43% vs. 33% chance). Participant-specific models reached 50% accuracy with minimal calibration (2 minutes per condition), improving continuously to 73% without plateau. Facial movement dynamics sensitively track workload with brief calibration, enabling adaptive interfaces using commodity cameras, though individual differences limit cross-participant generalisation.
>
---
#### [new 159] A Lensless Polarization Camera
- **分类: eess.IV; cs.CV; physics.optics**

- **简介: 该论文属于 polarization imaging 任务，旨在解决传统相机体积大、成本高的问题。通过设计无透镜系统，结合算法从单张快照恢复四幅线偏振图像。**

- **链接: [https://arxiv.org/pdf/2603.17156](https://arxiv.org/pdf/2603.17156)**

> **作者:** Noa Kraicer; Shay Elmalem; Erez Yosef; Hani Barhum; Raja Giryes
>
> **摘要:** Polarization imaging is a technique that creates a pixel map of the polarization state in a scene. Although invisible to the human eye, polarization can assist various sensing and computer vision tasks. Existing polarization cameras use spatial or temporal multiplexing, which increases the camera volume, weight, cost, or all of the above. Recent lensless imaging approaches, such as DiffuserCam, have demonstrated that compact imaging systems can be realized by replacing the lens with a coding element and performing computational reconstruction. In this work, we propose a compact lensless polarization camera composed of a diffuser and a simple striped polarization mask. By combining this optical design with a reconstruction algorithm that explicitly models the polarization-encoded lensless measurements, four linear polarization images are recovered from a single snapshot. Our results demonstrate the potential of lensless approaches for polarization imaging and reveal the physical factors that govern reconstruction quality, guiding the development of high-quality practical systems.
>
---
#### [new 160] SCE-LITE-HQ: Smooth visual counterfactual explanations with generative foundation models
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于可解释性AI任务，旨在生成高质量的反事实解释。解决现有方法依赖特定数据集生成模型和计算成本高的问题，提出SCE-LITE-HQ框架，利用预训练模型生成多样且真实的反事实样本。**

- **链接: [https://arxiv.org/pdf/2603.17048](https://arxiv.org/pdf/2603.17048)**

> **作者:** Ahmed Zeid; Sidney Bender
>
> **摘要:** Modern neural networks achieve strong performance but remain difficult to interpret in high-dimensional visual domains. Counterfactual explanations (CFEs) provide a principled approach to interpreting black-box predictions by identifying minimal input changes that alter model outputs. However, existing CFE methods often rely on dataset-specific generative models and incur substantial computational cost, limiting their scalability to high-resolution data. We propose SCE-LITE-HQ, a scalable framework for counterfactual generation that leverages pretrained generative foundation models without task-specific retraining. The method operates in the latent space of the generator, incorporates smoothed gradients to improve optimization stability, and applies mask-based diversification to promote realistic and structurally diverse counterfactuals. We evaluate SCE-LITE-HQ on natural and medical datasets using a desiderata-driven evaluation protocol. Results show that SCE-LITE-HQ produces valid, realistic, and diverse counterfactuals competitive with or outperforming existing baselines, while avoiding the overhead of training dedicated generative models.
>
---
#### [new 161] On the Degrees of Freedom of Gridded Control Points in Learning-Based Medical Image Registration
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 论文研究医学图像配准任务，解决传统方法在噪声或同质区域的不稳定性问题。提出GridReg框架，用稀疏控制点替代密集解码，减少参数量并提升效率。**

- **链接: [https://arxiv.org/pdf/2603.16940](https://arxiv.org/pdf/2603.16940)**

> **作者:** Wen Yan; Qianye Yang; Yipei Wang; Shonit Punwani; Mark Emberton; Vasilis Stavrinides; Yipeng Hu; Dean Barratt
>
> **备注:** 27 pages; 8 figures
>
> **摘要:** Many registration problems are ill-posed in homogeneous or noisy regions, and dense voxel-wise decoders can be unnecessarily high-dimensional. A sparse control-point parameterisation provides a compact, smooth deformation representation while reducing memory and improving stability. This work investigates the required control points for learning-based registration network development. We present GridReg, a learning-based registration framework that replaces dense voxel-wise decoding with displacement predictions at a sparse grid of control points. This design substantially cuts the parameter count and memory while retaining registration accuracy. Multiscale 3D encoder feature maps are flattened into a 1D token sequence with positional encoding to retain spatial context. The model then predicts a sparse gridded deformation field using a cross-attention module. We further introduce grid-adaptive training, enabling an adaptive model to operate at multiple grid sizes at inference without retraining. This work quantitatively demonstrates the benefits of using sparse grids. Using three data sets for registering prostate gland, pelvic organs and neurological structures, the results suggested a significant improvement with the usage of grid-controled displacement field. Alternatively, the superior registration performance was obtained using the proposed approach, with a similar or less computational cost, compared with existing algorithms that predict DDFs or displacements sampled on scattered key points.
>
---
#### [new 162] UNICORN: Ultrasound Nakagami Imaging via Score Matching and Adaptation for Assessing Hepatic Steatosis
- **分类: eess.IV; cs.AI; cs.CV; q-bio.QM**

- **简介: 该论文属于医学图像分析任务，旨在解决肝脂肪变性评估中的成像问题。提出UNICORN方法，通过分数匹配实现高精度、高分辨率的 Nakagami 参数估计。**

- **链接: [https://arxiv.org/pdf/2603.16942](https://arxiv.org/pdf/2603.16942)**

> **作者:** Kwanyoung Kim; Jaa-Yeon Lee; Youngjun Ko; GunWoo Lee; Jong Chul Ye
>
> **备注:** 12pages, 7 figures, 6 tables. arXiv admin note: text overlap with arXiv:2403.06275
>
> **摘要:** Ultrasound imaging is an essential first-line tool for assessing hepatic steatosis. While conventional B-mode ultrasound imaging has limitations in providing detailed tissue characterization, ultrasound Nakagami imaging holds promise for visualizing and quantifying tissue scattering in backscattered signals, with potential applications in fat fraction analysis. However, existing methods for Nakagami imaging struggle with optimal window size selection and suffer from estimator instability, leading to degraded image resolution. To address these challenges, we propose a novel method called UNICORN (Ultrasound Nakagami Imaging via Score Matching and Adaptation), which offers an accurate, closed-form estimator for Nakagami parameter estimation based on the score function of the ultrasound envelope signal. Unlike methods that visualize only specific regions of interest (ROI) and estimate parameters within fixed window sizes, our approach provides comprehensive parameter mapping by providing a pixel-by-pixel estimator, resulting in high-resolution imaging. We demonstrated that our proposed estimator effectively assesses hepatic steatosis and provides visual distinction in the backscattered statistics associated with this condition. Through extensive experiments using real envelope data from patient, we validated that UNICORN enables clinical detection of hepatic steatosis and exhibits robustness and generalizability.
>
---
#### [new 163] Topology-Guided Biomechanical Profiling: A White-Box Framework for Opportunistic Screening of Spinal Instability on Routine CT
- **分类: q-bio.QM; cs.CV**

- **简介: 该论文提出TGBP框架，用于自动评估脊柱不稳，解决SINS评分中几何推理复杂的问题，通过拓扑引导的生物力学分析实现准确筛查。**

- **链接: [https://arxiv.org/pdf/2603.16963](https://arxiv.org/pdf/2603.16963)**

> **作者:** Zanting Ye; Xuanbin Wu; Guoqing Zhong; Shengyuan Liu; Jiashuai Liu; Ge Song; Zhisong Wang; Jing Hao; Xiaolong Niu; Yefeng Zheng; Yu Zhang; Lijun Lu
>
> **备注:** 11 pages, 3 tables, 2 figures
>
> **摘要:** Routine oncologic computed tomography (CT) presents an ideal opportunity for screening spinal instability, yet prophylactic stabilization windows are frequently missed due to the complex geometric reasoning required by the Spinal Instability Neoplastic Score (SINS). Automating SINS is fundamentally hindered by metastatic osteolysis, which induces topological ambiguity that confounds standard segmentation and black-box AI. We propose Topology-Guided Biomechanical Profiling (TGBP), an auditable white-box framework decoupling anatomical perception from structural reasoning. TGBP anchors SINS assessment on two deterministic geometric innovations: (i) canal-referenced partitioning to resolve posterolateral boundary ambiguity, and (ii) context-aware morphometric normalization via covariance-based oriented bounding boxes (OBB) to quantify vertebral collapse. Integrated with auxiliary radiomic and large language model (LLM) modules, TGBP provides an end-to-end, interpretable SINS evaluation. Validated on a multi-center, multi-cancer cohort ($N=482$), TGBP achieved 90.2\% accuracy in 3-tier stability triage. In a blinded reader study ($N=30$), TGBP significantly outperformed medical oncologists on complex structural features ($\kappa=0.857$ vs.\ $0.570$) and prevented compounding errors in Total Score estimation ($\kappa=0.625$ vs.\ $0.207$), democratizing expert-level opportunistic screening.
>
---
#### [new 164] Deep Learning-Based Airway Segmentation in Systemic Lupus Erythematosus Patients with Interstitial Lung Disease (SLE-ILD): A Comparative High-Resolution CT Analysis
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在通过深度学习分析SLE患者肺部气道体积差异，解决ILD诊断问题。工作包括构建分割模型并比较不同组别气道体积变化。**

- **链接: [https://arxiv.org/pdf/2603.17547](https://arxiv.org/pdf/2603.17547)**

> **作者:** Sirong Piao; Ying Ming; Ruijie Zhao; Jiaru Wang; Ran Xiao; Rui Zhao; Zicheng Liao; Qiqi Xu; Shaoze Luo; Bing Li; Lin Li; Zhuangfei Ma; Fuling Zheng; Wei Song
>
> **摘要:** To characterize lobar and segmental airway volume differences between systemic lupus erythematosus (SLE) patients with interstitial lung disease (ILD) and those without ILD (non-ILD) using a deep learning-based approach on non-contrast chest high-resolution CT (HRCT). Methods: A retrospective analysis was conducted on 106 SLE patients (27 SLE-ILD, 79 SLE-non-ILD) who underwent HRCT. A customized deep learning framework based on the U-Net architecture was developed to automatically segment airway structures at the lobar and segmental levels via HRCT. Volumetric measurements of lung lobes and segments derived from the segmentations were statistically compared between the two groups using two-sample t-tests (significance threshold: p < 0.05). Results: At lobar level, significant airway volume enlargement in SLE-ILD patients was observed in the right upper lobe (p=0.009) and left upper lobe (p=0.039) compared to SLE-non-ILD. At the segmental level, significant differences were found in segments including R1 (p=0.016), R3 (p<0.001), and L3 (p=0.038), with the most marked changes in the upper lung zones, while lower zones showed non-significant trends. Conclusion: Our study demonstrates that an automated deep learning-based approach can effectively quantify airway volumes on HRCT scans and reveal significant, region-specific airway dilation in patients with SLE-ILD compared to those without ILD. The pattern of involvement, predominantly affecting the upper lobes and specific segments, highlights a distinct topographic phenotype of SLE-ILD and implicates airway structural alterations as a potential biomarker for disease presence. This AI-powered quantitative imaging biomarker holds promise for enhancing the early detection and monitoring of ILD in the SLE population, ultimately contributing to more personalized patient management.
>
---
#### [new 165] VectorWorld: Efficient Streaming World Model via Diffusion Flow on Vector Graphs
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出VectorWorld，用于自动驾驶的闭环仿真。解决历史初始化不匹配、采样延迟和长期不一致问题，通过增量生成矢量图块和物理对齐策略，实现高效实时模拟。**

- **链接: [https://arxiv.org/pdf/2603.17652](https://arxiv.org/pdf/2603.17652)**

> **作者:** Chaokang Jiang; Desen Zhou; Jiuming Liu; Kevin Li Sun
>
> **备注:** Under Review
>
> **摘要:** Closed-loop evaluation of autonomous-driving policies requires interactive simulation beyond log replay. However, existing generative world models often degrade in closed loop due to (i) history-free initialization that mismatches policy inputs, (ii) multi-step sampling latency that violates real-time budgets, and (iii) compounding kinematic infeasibility over long horizons. We propose VectorWorld, a streaming world model that incrementally generates ego-centric $64 \mathrm{m}\times 64\mathrm{m}$ lane--agent vector-graph tiles during rollout. VectorWorld aligns initialization with history-conditioned policies by producing a policy-compatible interaction state via a motion-aware gated VAE. It enables real-time outpainting via solver-free one-step masked completion with an edge-gated relational DiT trained with interval-conditioned MeanFlow and JVP-based large-step supervision. To stabilize long-horizon rollouts, we introduce $\Delta$Sim, a physics-aligned non-ego (NPC) policy with hybrid discrete--continuous actions and differentiable kinematic logit shaping. On Waymo open motion and nuPlan, VectorWorld improves map-structure fidelity and initialization validity, and supports stable, real-time $1\mathrm{km}+$ closed-loop rollouts (\href{this https URL}{code}).
>
---
#### [new 166] Neural Radiance Maps for Extraterrestrial Navigation and Path Planning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自主导航任务，旨在解决外星探测器路径规划问题。通过使用NeRF构建全局地图，结合局部信息进行高效路径规划。**

- **链接: [https://arxiv.org/pdf/2603.17236](https://arxiv.org/pdf/2603.17236)**

> **作者:** Adam Dai; Shubh Gupta; Grace Gao
>
> **备注:** Published in the Proceedings of the ION GNSS+ 2023 Conference
>
> **摘要:** Autonomous vehicles such as the Mars rovers currently lead the vanguard of surface exploration on extraterrestrial planets and moons. In order to accelerate the pace of exploration and science objectives, it is critical to plan safe and efficient paths for these vehicles. However, current rover autonomy is limited by a lack of global maps which can be easily constructed and stored for onboard re-planning. Recently, Neural Radiance Fields (NeRFs) have been introduced as a detailed 3D scene representation which can be trained from sparse 2D images and efficiently stored. We propose to use NeRFs to construct maps for online use in autonomous navigation, and present a planning framework which leverages the NeRF map to integrate local and global information. Our approach interpolates local cost observations across global regions using kernel ridge regression over terrain features extracted from the NeRF map, allowing the rover to re-route itself around untraversable areas discovered during online operation. We validate our approach in high-fidelity simulation and demonstrate lower cost and higher percentage success rate path planning compared to various baselines.
>
---
#### [new 167] The Truth, the Whole Truth, and Nothing but the Truth: Automatic Visualization Evaluation from Reconstruction Quality
- **分类: cs.HC; cs.CV**

- **简介: 该论文属于可视化评估任务，旨在解决自动评估生成可视化质量的问题。通过重建原始数据来衡量可视化质量，实现无需人工标注的自动化评估。**

- **链接: [https://arxiv.org/pdf/2603.16873](https://arxiv.org/pdf/2603.16873)**

> **作者:** Roxana Bujack; Li-Ta Lo; Ethan Stam; Ayan Biswas; David Rogers
>
> **摘要:** Recent advances in AI enable the automatic generation of visualizations directly from textual prompts using agentic workflows. However, visualizations produced via one-shot generative methods often suffer from insufficient quality, typically requiring a human in the loop to refine the outputs. Human evaluation, though effective, is costly and impractical at scale. To alleviate this problem, we propose an automated metric that evaluates visualization quality without relying on extensive human-labeled datasets. Instead, our approach uses the original underlying data as implicit ground truth. Specifically, we introduce a method that measures visualization quality by assessing the reconstruction accuracy of the original data from the visualization itself. This reconstruction-based metric provides an autonomous and scalable proxy for thorough human evaluation, facilitating more efficient and reliable AI-driven visualization workflows.
>
---
#### [new 168] SLAM Adversarial Lab: An Extensible Framework for Visual SLAM Robustness Evaluation under Adverse Conditions
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SAL框架，用于评估视觉SLAM在恶劣条件下的鲁棒性。解决SLAM系统在复杂环境下的稳定性问题，通过生成对抗数据集并测试不同算法表现。**

- **链接: [https://arxiv.org/pdf/2603.17165](https://arxiv.org/pdf/2603.17165)**

> **作者:** Mohamed Hefny; Karthik Dantu; Steven Y. Ko
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** We present SAL (SLAM Adversarial Lab), a modular framework for evaluating visual SLAM systems under adversarial conditions such as fog and rain. SAL represents each adversarial condition as a perturbation that transforms an existing dataset into an adversarial dataset. When transforming a dataset, SAL supports severity levels using easily-interpretable real-world units such as meters for fog visibility. SAL's extensible architecture decouples datasets, perturbations, and SLAM algorithms through common interfaces, so users can add new components without rewriting integration code. Moreover, SAL includes a search procedure that finds the severity level of a perturbation at which a SLAM system fails. To showcase the capabilities of SAL, our evaluation integrates seven SLAM algorithms and evaluates them across three datasets under weather, camera, and video transport perturbations.
>
---
#### [new 169] DancingBox: A Lightweight MoCap System for Character Animation from Physical Proxies
- **分类: cs.GR; cs.CV; cs.HC**

- **简介: 该论文提出DancingBox，一个轻量级动作捕捉系统，通过日常物体的粗略运动生成真实角色动画，解决普通用户难以使用专业设备进行角色动画的问题。**

- **链接: [https://arxiv.org/pdf/2603.17704](https://arxiv.org/pdf/2603.17704)**

> **作者:** Haocheng Yuan; Adrien Bousseau; Hao Pan; Lei Zhong; Changjian Li
>
> **备注:** Accepted to CHI2026
>
> **摘要:** Creating compelling 3D character animations typically requires either expert use of professional software or expensive motion capture systems operated by skilled actors. We present DancingBox, a lightweight, vision-based system that makes motion capture accessible to novices by reimagining the process as digital puppetry. Instead of tracking precise human motions, DancingBox captures the approximate movements of everyday objects manipulated by users with a single webcam. These coarse proxy motions are then refined into realistic character animations by conditioning a generative motion model on bounding-box representations, enriched with human motion priors learned from large-scale datasets. To overcome the lack of paired proxy-animation data, we synthesize training pairs by converting existing motion capture sequences into proxy representations. A user study demonstrates that DancingBox enables intuitive and creative character animation using diverse proxies, from plush toys to bananas, lowering the barrier to entry for novice animators.
>
---
#### [new 170] DSS-GAN: Directional State Space GAN with Mamba backbone for Class-Conditional Image Synthesis
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出DSS-GAN，用于类别条件图像生成，解决传统条件机制局限问题。采用Mamba作为生成器骨干，引入方向潜向路由机制，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.17637](https://arxiv.org/pdf/2603.17637)**

> **作者:** Aleksander Ogonowski; Konrad Klimaszewski; Przemysław Rokita
>
> **摘要:** We present DSS-GAN, the first generative adversarial network to employ Mamba as a hierarchical generator backbone for noise-to-image synthesis. The central contribution is Directional Latent Routing (DLR), a novel conditioning mechanism that decomposes the latent vector into direction-specific subvectors, each jointly projected with a class embedding to produce a feature-wise affine modulation of the corresponding Mamba scan. Unlike conventional class conditioning that injects a global signal, DLR couples class identity and latent structure along distinct spatial axes of the feature map, applied consistently across all generative scales. DSS-GAN achieves improved FID, KID, and precision-recall scores compared to StyleGAN2-ADA across multiple tested datasets. Analysis of the latent space reveals that directional subvectors exhibit measurable specialization: perturbations along individual components produce structured, direction-correlated changes in the synthesized image.
>
---
## 更新

#### [replaced 001] Automated Wicket-Taking Delivery Segmentation and Trajectory-Based Dismissal-Zone Analysis in Cricket Videos Using OCR-Guided YOLOv8
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.18405](https://arxiv.org/pdf/2510.18405)**

> **作者:** Joy Karmoker; Masum Billah; Mst Jannatun Ferdous; Akif Islam; Mohd Ruhul Ameen; Md. Omar Faruqe
>
> **备注:** 3 figures, 5 tables, submitted to the IEEE International Conference on Engineering and Frontier Technologies (ICEFronT) 2026
>
> **摘要:** Cricket generates a rich stream of visual and contextual information, yet much of its tactical analysis still depends on slow and subjective manual review. Motivated by the need for a more efficient and data-driven alternative, this paper presents an automated approach for cricket video analysis that identifies wicket-taking deliveries, detects the pitch and ball, and models ball trajectories for post-match assessment. The proposed system combines optical character recognition (OCR) with image preprocessing techniques, including grayscale conversion, power transformation, and morphological operations, to robustly extract scorecard information and detect wicket events from broadcast videos. For visual understanding, YOLOv8 is employed for both pitch and ball detection. The pitch detection model achieved 99.5% mAP50 with a precision of 0.999, while the transfer learning-based ball detection model attained 99.18% mAP50 with 0.968 precision and 0.978 recall. Based on these detections, the system further models ball trajectories to reveal regions associated with wicket-taking deliveries, offering analytical cues for trajectory-based dismissal-zone interpretation and potential batting vulnerability assessment. Experimental results on multiple cricket match videos demonstrate the effectiveness of the proposed approach and highlight its potential for supporting coaching, tactical evaluation, and data-driven decision-making in cricket.
>
---
#### [replaced 002] Dynamic Black-hole Emission Tomography with Physics-informed Neural Fields
- **分类: gr-qc; astro-ph.IM; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.08029](https://arxiv.org/pdf/2602.08029)**

> **作者:** Berthy T. Feng; Andrew A. Chael; David Bromley; Aviad Levis; William T. Freeman; Katherine L. Bouman
>
> **备注:** CVPR 2026
>
> **摘要:** With the success of static black-hole imaging, the next frontier is the dynamic and 3D imaging of black holes. Recovering the dynamic 3D gas near a black hole would reveal previously-unseen parts of the universe and inform new physics models. However, only sparse radio measurements from a single viewpoint are possible, making the dynamic 3D reconstruction problem significantly ill-posed. Previously, BH-NeRF addressed the ill-posed problem by assuming Keplerian dynamics of the gas, but this assumption breaks down near the black hole, where the strong gravitational pull of the black hole and increased electromagnetic activity complicate fluid dynamics. To overcome the restrictive assumptions of BH-NeRF, we propose PI-DEF, a physics-informed approach that uses differentiable neural rendering to fit a 4D (time + 3D) emissivity field given EHT measurements. Our approach jointly reconstructs the 3D velocity field with the 4D emissivity field and enforces the velocity as a soft constraint on the dynamics of the emissivity. In experiments on simulated data, we find significantly improved reconstruction accuracy over both BH-NeRF and a physics-agnostic approach. We demonstrate how our method may be used to estimate other physics parameters of the black hole, such as its spin.
>
---
#### [replaced 003] No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.19248](https://arxiv.org/pdf/2602.19248)**

> **作者:** Zunkai Dai; Ke Li; Jiajia Liu; Jie Yang; Yuanyuan Qiao
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** The collection and detection of video anomaly data has long been a challenging problem due to its rare occurrence and spatio-temporal scarcity. Existing video anomaly detection (VAD) methods under perform in open-world scenarios. Key contributing factors include limited dataset diversity, and inadequate understanding of context-dependent anomalous semantics. To address these issues, i) we propose LAVIDA, an end-to-end zero-shot video anomaly detection framework. ii) LAVIDA employs an Anomaly Exposure Sampler that transforms segmented objects into pseudo-anomalies to enhance model adaptability to unseen anomaly categories. It further integrates a Multimodal Large Language Model (MLLM) to bolster semantic comprehension capabilities. Additionally, iii) we design a token compression approach based on reverse attention to handle the spatio-temporal scarcity of anomalous patterns and decrease computational cost. The training process is conducted solely on pseudo anomalies without any VAD data. Evaluations across four benchmark VAD datasets demonstrate that LAVIDA achieves SOTA performance in both frame-level and pixel-level anomaly detection under the zero-shot setting. Our code is available in this https URL.
>
---
#### [replaced 004] The MCC approaches the geometric mean of precision and recall as true negatives approach infinity
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2305.00594](https://arxiv.org/pdf/2305.00594)**

> **作者:** Jon Crall
>
> **备注:** 9 pages, 0 figures. Major revision: adds Lean 4 formalization, expanded related work, and revised discussion of the object-detection setting; includes a brief note on LLM-assisted formalization and literature search
>
> **摘要:** The performance of a binary classifier is described by a confusion matrix with four entries: the number of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). The Matthews Correlation Coefficient (MCC), F1, and Fowlkes-Mallows (FM) scores are scalars that summarize a confusion matrix. Both the F1 and FM scores are based on only three of the four entries in a confusion matrix (they ignore TN). Unlike F1 and FM, the MCC depends on all four entries of the confusion matrix, which can make it attractive in some cases. However, in some open world settings, measuring the number of true negatives is not straightforward. Object detection is such a case because the number of candidate negative boxes is effectively unbounded. This motivates the question: what is the limit of the MCC as the number of true negatives tends to infinity? Put plainly, as the true negative count grows, the MCC converges to the FM score, which is the geometric mean of precision and recall. This result was previously noted in the ecology literature in terms of the phi-coefficient and the Ochiai index, but we discuss it in the context of binary classifiers. Furthermore, we provide a full proof of the result, including a Lean formalization. We also briefly comment on the emerging role of LLMs in proof assistance and in locating prior work.
>
---
#### [replaced 005] SkyReels-V4: Multi-modal Video-Audio Generation, Inpainting and Editing model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.21818](https://arxiv.org/pdf/2602.21818)**

> **作者:** Guibin Chen; Dixuan Lin; Jiangping Yang; Youqiang Zhang; Zhengcong Fei; Debang Li; Sheng Chen; Chaofeng Ao; Nuo Pang; Yiming Wang; Yikun Dou; Zheng Chen; Mingyuan Fan; Tuanhui Li; Mingshan Chang; Hao Zhang; Xiaopeng Sun; Jingtao Xu; Yuqiang Xie; Jiahua Wang; Zhiheng Xu; Weiming Xiong; Yuzhe Jin; Baoxuan Gu; Binjie Mao; Yunjie Yu; Jujie He; Yuhao Feng; Shiwen Tu; Chaojie Wang; Rui Yan; Wei Shen; Jingchen Wu; Peng Zhao; Xuanyue Zhong; Zhuangzhuang Liu; Kaifei Wang; Fuxiang Zhang; Weikai Xu; Wenyan Liu; Binglu Zhang; Yu Shen; Tianhui Xiong; Bin Peng; Liang Zeng; Xuchen Song; Haoxiang Guo; Peiyu Wang; Max W. Y. Lam; Chien-Hung Liu; Yahui Zhou
>
> **摘要:** SkyReels V4 is a unified multi modal video foundation model for joint video audio generation, inpainting, and editing. The model adopts a dual stream Multimodal Diffusion Transformer (MMDiT) architecture, where one branch synthesizes video and the other generates temporally aligned audio, while sharing a powerful text encoder based on the Multimodal Large Language Models (MLLM). SkyReels V4 accepts rich multi modal instructions, including text, images, video clips, masks, and audio references. By combining the MLLMs multi modal instruction following capability with in context learning in the video branch MMDiT, the model can inject fine grained visual guidance under complex conditioning, while the audio branch MMDiT simultaneously leverages audio references to guide sound generation. On the video side, we adopt a channel concatenation formulation that unifies a wide range of inpainting style tasks, such as image to video, video extension, and video editing under a single interface, and naturally extends to vision referenced inpainting and editing via multi modal prompts. SkyReels V4 supports up to 1080p resolution, 32 FPS, and 15 second duration, enabling high fidelity, multi shot, cinema level video generation with synchronized audio. To make such high resolution, long-duration generation computationally feasible, we introduce an efficiency strategy: Joint generation of low resolution full sequences and high-resolution keyframes, followed by dedicated super-resolution and frame interpolation models. To our knowledge, SkyReels V4 is the first video foundation model that simultaneously supports multi-modal input, joint video audio generation, and a unified treatment of generation, inpainting, and editing, while maintaining strong efficiency and quality at cinematic resolutions and durations.
>
---
#### [replaced 006] Event-Driven Video Generation
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.13402](https://arxiv.org/pdf/2603.13402)**

> **作者:** Chika Maduabuchi
>
> **摘要:** State-of-the-art text-to-video models often look realistic frame-by-frame yet fail on simple interactions: motion starts before contact, actions are not realized, objects drift after placement, and support relations break. We argue this stems from frame-first denoising, which updates latent state everywhere at every step without an explicit notion of when and where an interaction is active. We introduce Event-Driven Video Generation (EVD), a minimal DiT-compatible framework that makes sampling event-grounded: a lightweight event head predicts token-aligned event activity, event-grounded losses couple activity to state change during training, and event-gated sampling (with hysteresis and early-step scheduling) suppresses spurious updates while concentrating updates during interactions. On EVD-Bench, EVD consistently improves human preference and VBench dynamics, substantially reducing failure modes in state persistence, spatial accuracy, support relations, and contact stability without sacrificing appearance. These results indicate that explicit event grounding is a practical abstraction for reducing interaction hallucinations in video generation.
>
---
#### [replaced 007] CARPE: Context-Aware Image Representation Prioritization via Ensemble for Large Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.13622](https://arxiv.org/pdf/2601.13622)**

> **作者:** Donghee Lee; Rui Cai; Zhe Zhao
>
> **摘要:** Large vision-language models (LVLMs) are typically trained using autoregressive language modeling objectives, which align visual representations with linguistic space. While effective for multimodal reasoning, this alignment can weaken vision-centric capabilities, causing LVLMs to underperform their base vision encoders on tasks such as image classification. To address this limitation, we propose Context-Aware Image Representation Prioritization via Ensemble (CARPE), a lightweight framework that integrates raw vision features with aligned LLM representations through vision-integration layers and a context-aware ensemble mechanism. This design enhances the model's ability to adaptively weight visual and textual modalities and enables the model to capture various aspects of image representations. Extensive experiments demonstrate that CARPE improves performance on both image classification and diverse vision-language benchmarks. Our results suggest that modality balancing plays a critical role in multimodal generalization by improving representation utilization within autoregressive LVLMs.
>
---
#### [replaced 008] Towards One-step Causal Video Generation via Adversarial Self-Distillation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.01419](https://arxiv.org/pdf/2511.01419)**

> **作者:** Yongqi Yang; Huayang Huang; Xu Peng; Xiaobin Hu; Donghao Luo; Jiangning Zhang; Chengjie Wang; Yu Wu
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** Recent hybrid video generation models combine autoregressive temporal dynamics with diffusion-based spatial denoising, but their sequential, iterative nature leads to error accumulation and long inference times. In this work, we propose a distillation-based framework for efficient causal video generation that enables high-quality synthesis with extremely limited denoising steps. Our approach builds upon the Distribution Matching Distillation (DMD) framework and proposes a novel Adversarial Self-Distillation (ASD) strategy, which aligns the outputs of the student model's n-step denoising process with its (n+1)-step version at the distribution level. This design provides smoother supervision by bridging small intra-student gaps and more informative guidance by combining teacher knowledge with locally consistent student behavior, substantially improving training stability and generation quality in extremely few-step scenarios (e.g., 1-2 steps). In addition, we present a First-Frame Enhancement (FFE) strategy, which allocates more denoising steps to the initial frames to mitigate error propagation while applying larger skipping steps to later frames. Extensive experiments on VBench demonstrate that our method surpasses state-of-the-art approaches in both one-step and two-step video generation. Notably, our framework produces a single distilled model that flexibly supports multiple inference-step settings, eliminating the need for repeated re-distillation and enabling efficient, high-quality video synthesis.
>
---
#### [replaced 009] PAND: Prompt-Aware Neighborhood Distillation for Lightweight Fine-Grained Visual Classification
- **分类: cs.CV; cs.AI; cs.LG; cs.MM**

- **链接: [https://arxiv.org/pdf/2602.07768](https://arxiv.org/pdf/2602.07768)**

> **作者:** Qiuming Luo; Yuebing Li; Feng Li; Chang Kong
>
> **备注:** 6pages, 3 figures, conference
>
> **摘要:** Distilling knowledge from large Vision-Language Models (VLMs) into lightweight networks is crucial yet challenging in Fine-Grained Visual Classification (FGVC), due to the reliance on fixed prompts and global alignment. To address this, we propose PAND (Prompt-Aware Neighborhood Distillation), a two-stage framework that decouples semantic calibration from structural transfer. First, we incorporate Prompt-Aware Semantic Calibration to generate adaptive semantic anchors. Second, we introduce a neighborhood-aware structural distillation strategy to constrain the student's local decision structure. PAND consistently outperforms state-of-the-art methods on four FGVC benchmarks. Notably, our ResNet-18 student achieves 76.09% accuracy on CUB-200, surpassing the strong baseline VL2Lite by 3.4%. Code is available at this https URL.
>
---
#### [replaced 010] NutVLM: A Self-Adaptive Defense Framework against Full-Dimension Attacks for Vision Language Models in Autonomous Driving
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2602.13293](https://arxiv.org/pdf/2602.13293)**

> **作者:** Xiaoxu Peng; Dong Zhou; Jianwen Zhang; Guanghui Sun; Anh Tu Ngo; Anupam Chattopadhyay
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Vision Language Models (VLMs) have advanced perception in autonomous driving (AD), but they remain vulnerable to adversarial threats. These risks range from localized physical patches to imperceptible global perturbations. Existing defense methods for VLMs remain limited and often fail to reconcile robustness with clean-sample performance. To bridge these gaps, we propose NutVLM, a comprehensive self-adaptive defense framework designed to secure the entire perception-decision lifecycle. Specifically, we first employ NutNet++ as a sentinel, which is a unified detection-purification mechanism. It identifies benign samples, local patches, and global perturbations through three-way classification. Subsequently, localized threats are purified via efficient grayscale masking, while global perturbations trigger Expert-guided Adversarial Prompt Tuning (EAPT). Instead of the costly parameter updates of full-model fine-tuning, EAPT generates "corrective driving prompts" via gradient-based latent optimization and discrete projection. These prompts refocus the VLM's attention without requiring exhaustive full-model retraining. Evaluated on the Dolphins benchmark, our NutVLM yields a 4.89% improvement in overall metrics (e.g., Accuracy, Language Score, and GPT Score). These results validate NutVLM as a scalable security solution for intelligent transportation. Our code is available at this https URL.
>
---
#### [replaced 011] F2HDR: Two-Stage HDR Video Reconstruction via Flow Adapter and Physical Motion Modeling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.14920](https://arxiv.org/pdf/2603.14920)**

> **作者:** Huanjing Yue; Dawei Li; Shaoxiong Tu; Jingyu Yang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Reconstructing High Dynamic Range (HDR) videos from sequences of alternating-exposure Low Dynamic Range (LDR) frames remains highly challenging, especially under dynamic scenes where cross-exposure inconsistencies and complex motion make inter-frame alignment difficult, leading to ghosting and detail loss. Existing methods often suffer from inaccurate alignment, suboptimal feature aggregation, and degraded reconstruction quality in motion-dominated regions. To address these challenges, we propose F2HDR, a two-stage HDR video reconstruction framework that robustly perceives inter-frame motion and restores fine details in complex dynamic scenarios. The proposed framework integrates a flow adapter that adapts generic optical flow for robust cross-exposure alignment, a physical motion modeling to identify salient motion regions, and a motion-aware refinement network that aggregates complementary information while removing ghosting and noise. Extensive experiments demonstrate that F2HDR achieves state-of-the-art performance on real-world HDR video benchmarks, producing ghost-free and high-fidelity results under large motion and exposure variations.
>
---
#### [replaced 012] EdiVal-Agent: An Object-Centric Framework for Automated, Fine-Grained Evaluation of Multi-Turn Editing
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.13399](https://arxiv.org/pdf/2509.13399)**

> **作者:** Tianyu Chen; Yasi Zhang; Zhi Zhang; Peiyu Yu; Shu Wang; Zhendong Wang; Kevin Lin; Xiaofei Wang; Zhengyuan Yang; Linjie Li; Chung-Ching Lin; Jianwen Xie; Oscar Leong; Lijuan Wang; Ying Nian Wu; Mingyuan Zhou
>
> **备注:** Tianyu Chen and Yasi Zhang contributed equally; Oscar Leong, Lijuan Wang, Ying Nian Wu, and Mingyuan Zhou advised equally
>
> **摘要:** Instruction-based image editing has advanced rapidly, yet reliable and interpretable evaluation remains a bottleneck. Current protocols either (i) depend on paired reference images, resulting in limited coverage and inheriting biases from prior generative models or (ii) rely solely on zero-shot vision language models (VLMs), whose prompt-based assessments of instruction following, content consistency, and visual quality are often imprecise. To address this, we introduce EdiVal, an automated and fine-grained evaluation framework grounded in an object-centric perspective, designed to assess not only standard single-turn but also multi-turn instruction-based editing with precision. Given an input image, EdiVal first decomposes it into semantically meaningful objects, then synthesizes diverse, context-aware editing instructions while dynamically updating object pools across turns. These two stages enable two novel object centric metrics tailored for multi turn evaluation and one global metric of visual quality: 1) EdiVal-IF, which measures instruction following by combining open vocabulary object detectors for symbolic checks with VLMs for semantic verification on detector guided crops; 2) EdiVal-CC, which evaluates content consistency by calculating semantic similarity of unchanged objects and background using the evolving object pools; and 3) EdiVal-VQ, which quantifies changes in overall visual quality with human preference models. Instantiating this pipeline, we build EdiVal Bench, a multi-turn editing benchmark covering 9 instruction types and 16 state-of-the-art editing models, spanning in-context, flow-matching, and diffusion paradigms. We demonstrate that EdiVal can be used to identify existing failure modes, thereby informing the development of the next generation of editing models.
>
---
#### [replaced 013] Test-Time 3D Occupancy Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.08485](https://arxiv.org/pdf/2503.08485)**

> **作者:** Fengyi Zhang; Xiangyu Sun; Huitong Yang; Zheng Zhang; Zi Huang; Yadan Luo
>
> **备注:** CVPR 2026
>
> **摘要:** Self-supervised 3D occupancy prediction offers a promising solution for understanding complex driving scenes without requiring costly 3D annotations. However, training dense occupancy decoders to capture fine-grained geometry and semantics can demand hundreds of GPU hours, and once trained, such models struggle to adapt to varying voxel resolutions or novel object categories without extensive retraining. To overcome these limitations, we propose a practical and flexible test-time occupancy prediction framework termed TT-Occ. Our method incrementally constructs, optimizes, and voxelizes time-aware 3D Gaussians from raw sensor streams by integrating vision foundation models (VFMs) at runtime. The flexible representation of 3D Gaussians enables voxelization at arbitrary user-specified resolutions, while the strong generalization capability of VFMs supports accurate perception and open-vocabulary recognition without requiring any network training or fine-tuning. To validate the generality and effectiveness of our framework, we present two variants: a LiDAR-based version and a vision-centric version, and conduct extensive experiments on the Occ3D-nuScenes and nuCraft benchmarks under varying voxel resolutions. Experimental results show that TT-Occ significantly outperforms existing computationally expensive pretrained self-supervised counterparts. Code is available at this https URL.
>
---
#### [replaced 014] Domain and Task-Focused Example Selection for Data-Efficient Contrastive Medical Image Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.19208](https://arxiv.org/pdf/2505.19208)**

> **作者:** Tyler Ward; Aaron Moseley; Abdullah-Al-Zubaer Imran
>
> **备注:** Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) this https URL
>
> **摘要:** Segmentation is one of the most important tasks in the medical imaging pipeline as it influences a number of image-based decisions. To be effective, fully supervised segmentation approaches require large amounts of manually annotated training data. However, the pixel-level annotation process is expensive, time-consuming, and error-prone, hindering progress and making it challenging to perform effective segmentations. Therefore, models must learn efficiently from limited labeled data. Self-supervised learning (SSL), particularly contrastive learning via pre-training on unlabeled data and fine-tuning on limited annotations, can facilitate such limited labeled image segmentation. To this end, we propose a novel self-supervised contrastive learning framework for medical image segmentation, leveraging inherent relationships of different images, dubbed PolyCL. Without requiring any pixel-level annotations or unreasonable data augmentations, our PolyCL learns and transfers context-aware discriminant features useful for segmentation from an innovative surrogate, in a task-related manner. Additionally, we integrate the Segment Anything Model (SAM) into our framework in two novel ways: as a post-processing refinement module that improves the accuracy of predicted masks using bounding box prompts derived from coarse outputs, and as a propagation mechanism via SAM 2 that generates volumetric segmentations from a single annotated 2D slice. Experimental evaluations on three public computed tomography (CT) datasets demonstrate that PolyCL outperforms fully-supervised and self-supervised baselines in both low-data and cross-domain scenarios. Our code is available at this https URL.
>
---
#### [replaced 015] Unsupervised Decomposition and Recombination with Discriminator-Driven Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.22057](https://arxiv.org/pdf/2601.22057)**

> **作者:** Archer Wang; Emile Anand; Yilun Du; Marin Soljačić
>
> **备注:** 28 pages, 16 figures, 4 tables
>
> **摘要:** Decomposing complex data into factorized representations can reveal reusable components and enable synthesizing new samples via component recombination. We investigate this in the context of diffusion-based models that learn factorized latent spaces without factor-level supervision. In images, factors can capture background, illumination, and object attributes; in robotic videos, they can capture reusable motion components. To improve both latent factor discovery and quality of compositional generation, we introduce an adversarial training signal via a discriminator trained to distinguish between single-source samples and those generated by recombining factors across sources. By optimizing the generator to fool this discriminator, we encourage physical and semantic consistency in the resulting recombinations. Our method outperforms implementations of prior baselines on CelebA-HQ, Virtual KITTI, CLEVR, and Falcor3D, achieving lower FID scores and better disentanglement as measured by MIG and MCC. Furthermore, we demonstrate a novel application to robotic video trajectories: by recombining learned action components, we generate diverse sequences that significantly increase state-space coverage for exploration on the LIBERO benchmark.
>
---
#### [replaced 016] Spatial Transcriptomics as Images for Large-Scale Pretraining
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.13432](https://arxiv.org/pdf/2603.13432)**

> **作者:** Yishun Zhu; Jiaxin Qi; Jian Wang; Yuhua Zheng; Jianqiang Huang
>
> **摘要:** Spatial Transcriptomics (ST) profiles thousands of gene expression values at discrete spots with precise coordinates on tissue sections, preserving spatial context essential for clinical and pathological studies. With rising sequencing throughput and advancing platforms, the expanding data volumes motivate large-scale ST pretraining. However, the fundamental unit for pretraining, i.e., what constitutes a single training sample, remains ill-posed. Existing choices fall into two camps: (1) treating each spot as an independent sample, which discards spatial dependencies and collapses ST into single-cell transcriptomics; and (2) treating an entire slide as a single sample, which produces prohibitively large inputs and drastically fewer training examples, undermining effective pretraining. To address this gap, we propose treating spatial transcriptomics as croppable images. Specifically, we define a multi-channel image representation with fixed spatial size by cropping patches from raw slides, thereby preserving spatial context while substantially increasing the number of training samples. Along the channel dimension, we define gene subset selection rules to control input dimensionality and improve pretraining stability. Extensive experiments show that the proposed image-like dataset construction for ST pretraining consistently improves downstream performance, outperforming conventional pretraining schemes. Ablation studies verify that both spatial patching and channel design are necessary, establishing a unified, practical paradigm for organizing ST data and enabling large-scale pretraining.
>
---
#### [replaced 017] HGP-Mamba: Integrating Histology and Generated Protein Features for Mamba-based Multimodal Survival Risk Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.16421](https://arxiv.org/pdf/2603.16421)**

> **作者:** Jing Dai; Chen Wu; Ming Wu; Qibin Zhang; Zexi Wu; Jingdong Zhang; Hongming Xu
>
> **备注:** Accepted at IEEE ICME 2026. This arXiv version includes additional supplementary experiments and extended discussions beyond the conference version
>
> **摘要:** Recent advances in multimodal learning have significantly improved cancer survival risk prediction. However, the joint prognostic potential of protein markers and histopathology images remains underexplored, largely due to the high cost and limited availability of protein expression profiling. To address this challenge, we propose HGP-Mamba, a Mamba-based multimodal framework that efficiently integrates histological with generated protein features for survival risk prediction. Specifically, we introduce a protein feature extractor (PFE) that leverages pretrained foundation models to derive high-throughput protein embeddings directly from Whole Slide Images (WSIs), enabling data-efficient incorporation of molecular information. Together with histology embeddings that capture morphological patterns, we further introduce the Local Interaction-aware Mamba (LiAM) for fine-grained feature interaction and the Global Interaction-enhanced Mamba (GiEM) to promote holistic modality fusion at the slide level, thus capture complex cross-modal dependencies. Experiments on four public cancer datasets demonstrate that HGP-Mamba achieves state-of-the-art performance while maintaining superior computational efficiency compared with existing methods. Our source code is publicly available at this https URL.
>
---
#### [replaced 018] High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.22179](https://arxiv.org/pdf/2503.22179)**

> **作者:** Dailan He; Xiahong Wang; Shulun Wang; Guanglu Song; Bingqi Ma; Hao Shao; Yu Liu; Hongsheng Li
>
> **备注:** CVPR 2026
>
> **摘要:** Face swapping aims to seamlessly transfer a source facial identity onto a target while preserving target attributes such as pose and expression. Diffusion models, known for their superior generative capabilities, have recently shown promise in advancing face-swapping quality. This paper addresses two key challenges in diffusion-based face swapping: the prioritized preservation of identity over target attributes and the inherent conflict between identity and attribute conditioning. To tackle these issues, we introduce an identity-constrained attribute-tuning framework for face swapping that first ensures identity preservation and then fine-tunes for attribute alignment, achieved through a decoupled condition injection. We further enhance fidelity by incorporating identity and adversarial losses in a post-training refinement stage. Our proposed identity-constrained diffusion-based face-swapping model outperforms existing methods in both qualitative and quantitative evaluations, demonstrating superior identity similarity and attribute consistency, achieving a new state-of-the-art performance in high-fidelity face swapping.
>
---
#### [replaced 019] Multi-modal 3D Pose and Shape Estimation with Computed Tomography
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.19405](https://arxiv.org/pdf/2503.19405)**

> **作者:** Mingxiao Tu; Hoijoon Jung; Alireza Moghadam; Jineel Raythatha; Lachlan Allan; Jeremy Hsu; Andre Kyme; Jinman Kim
>
> **摘要:** In perioperative care, precise in-bed 3D patient pose and shape estimation (PSE) can be vital in optimizing patient positioning in preoperative planning, enabling accurate overlay of medical images for augmented reality-based surgical navigation, and mitigating risks of prolonged immobility during recovery. Conventional PSE methods relying on modalities such as RGB-D, infrared, or pressure maps often struggle with occlusions caused by bedding and complex patient positioning, leading to inaccurate estimation that can affect clinical outcomes. To address these challenges, we present the first multi-modal in-bed patient 3D PSE network that fuses detailed geometric features extracted from routinely acquired computed tomography (CT) scans with depth maps (mPSE-CT). mPSE-CT incorporates a shape estimation module that utilizes probabilistic correspondence alignment, a pose estimation module with a refined neural network, and a final parameters mixing module. This multi-modal network robustly reconstructs occluded body regions and enhances the accuracy of the estimated 3D human mesh model. We validated mPSE-CT using proprietary whole-body rigid phantom and volunteer datasets in clinical scenarios. mPSE-CT outperformed the best-performing prior method by 23% and 49.16% in pose and shape estimation respectively, demonstrating its potential for improving clinical outcomes in challenging perioperative environments.
>
---
#### [replaced 020] KeyframeFace: Language-Driven Facial Animation via Semantic Keyframes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.11321](https://arxiv.org/pdf/2512.11321)**

> **作者:** Jingchao Wu; Zejian Kang; Haibo Liu; Yuanchen Fei; Xiangru Huang
>
> **摘要:** Facial animation is a core component for creating digital characters in Computer Graphics (CG) industry. A typical production workflow relies on sparse, semantically meaningful keyframes to precisely control facial expressions. Enabling such animation directly from natural-language descriptions could significantly improve content creation efficiency and accessibility. However, most existing methods adopt a text-to-continuous-frames paradigm, directly regressing dense facial motion trajectories from language. This formulation entangles high-level semantic intent with low-level motion, lacks explicit semantic control structure, and limits precise editing and interpretability. Inspired by the keyframe paradigm in animation production, we propose KeyframeFace, a framework for semantic facial animation from language via interpretable keyframes. Instead of predicting dense motion trajectories, our method represents animation as a sequence of semantically meaningful keyframes in an interpretable ARKit-based facial control space. A language-driven model leverages large language model (LLM) priors to generate keyframes that align with contextual text descriptions and emotion cues. To support this formulation, we construct a multimodal dataset comprising 2,100 expression scripts paired with monocular videos, per-frame ARKit coefficients, and manually annotated semantic keyframes. Experiments show that incorporating semantic keyframe supervision and language priors significantly improves expression fidelity and semantic alignment compared to methods that do not use facial action semantics.
>
---
#### [replaced 021] Rationale Matters: Learning Transferable Rubrics via Proxy-Guided Critique for VLM Reward Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.16600](https://arxiv.org/pdf/2603.16600)**

> **作者:** Weijie Qiu; Dai Guan; Junxin Wang; Zhihang Li; Yongbo Gai; Mengyu Zhou; Erchao Zhao; Xiaoxi Jiang; Guanjun Jiang
>
> **备注:** 25 pages, 10 figures,
>
> **摘要:** Generative reward models (GRMs) for vision-language models (VLMs) often evaluate outputs via a three-stage pipeline: rubric generation, criterion-based scoring, and a final verdict. However, the intermediate rubric is rarely optimized directly. Prior work typically either treats rubrics as incidental or relies on expensive LLM-as-judge checks that provide no differentiable signal and limited training-time guidance. We propose Proxy-GRM, which introduces proxy-guided rubric verification into Reinforcement Learning (RL) to explicitly enhance rubric quality. Concretely, we train lightweight proxy agents (Proxy-SFT and Proxy-RL) that take a candidate rubric together with the original query and preference pair, and then predict the preference ordering using only the rubric as evidence. The proxy's prediction accuracy serves as a rubric-quality reward, incentivizing the model to produce rubrics that are internally consistent and transferable. With ~50k data samples, Proxy-GRM reaches state-of-the-art results on the VL-Reward Bench, Multimodal Reward Bench, and MM-RLHF-Reward Bench, outperforming the methods trained on four times the data. Ablations show Proxy-SFT is a stronger verifier than Proxy-RL, and implicit reward aggregation performs best. Crucially, the learned rubrics transfer to unseen evaluators, improving reward accuracy at test time without additional training. Our code is available at this https URL.
>
---
#### [replaced 022] Skyfall-GS: Synthesizing Immersive 3D Urban Scenes from Satellite Imagery
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.15869](https://arxiv.org/pdf/2510.15869)**

> **作者:** Jie-Ying Lee; Yi-Ruei Liu; Shr-Ruei Tsai; Wei-Cheng Chang; Chung-Ho Wu; Jiewen Chan; Zhenjun Zhao; Chieh Hubert Lin; Yu-Lun Liu
>
> **备注:** Project page: this https URL
>
> **摘要:** Synthesizing large-scale, explorable, and geometrically accurate 3D urban scenes is a challenging yet valuable task for immersive and embodied applications. The challenge lies in the lack of large-scale and high-quality real-world 3D scans for training generalizable generative models. In this paper, we take an alternative route to create large-scale 3D scenes by leveraging readily available satellite imagery for realistic coarse geometry and open-domain diffusion models for high-quality close-up appearance synthesis. We propose Skyfall-GS, a novel hybrid framework that synthesizes immersive city-block scale 3D urban scenes by combining satellite reconstruction with diffusion refinement, eliminating the need for costly 3D annotations, and also featuring real-time, immersive 3D exploration. We tailor a curriculum-driven iterative refinement strategy to progressively enhance geometric completeness and photorealistic texture. Extensive experiments demonstrate that Skyfall-GS provides improved cross-view consistent geometry and more realistic textures compared to state-of-the-art approaches. Project page: this https URL
>
---
#### [replaced 023] Learning Goal-Oriented Vision-and-Language Navigation with Self-Improving Demonstrations at Scale
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.24910](https://arxiv.org/pdf/2509.24910)**

> **作者:** Songze Li; Zun Wang; Gengze Zhou; Jialu Li; Xiangyu Zeng; Ziyang Gong; Limin Wang; Yu Qiao; Qi Wu; Mohit Bansal; Yi Wang
>
> **摘要:** Goal-oriented vision-language navigation requires robust exploration capabilities for agents to navigate to specified goals in unknown environments without step-by-step instructions. Existing methods tend to exclusively utilize shortest-path trajectories, lacking effective exploration priors for training navigation agents. To address the above challenges, we present SID, a goal-oriented vision-and-language navigation learning approach with Self-Improving Demonstrations. Specifically, SID learns an initial agent on the shortest-path data sampled from environments and then leverages this agent to generate novel exploration trajectories. The novel rollouts provide demonstrations with stronger exploration strategies to train a better agent, which in turn produces higher-quality agent demonstrations for the next round of training. We show that this iterative self-improving pipeline readily scales to new environments, and the resulting demonstrations are highly transferable, elevating the performance ceiling across a variety of vision-and-language navigation tasks. Extensive experiments demonstrate that SID significantly boosts the exploration capabilities and generalization of navigation agents. The resulting agent achieves new state-of-the-art performance on goal-oriented vision-and-language navigation benchmarks, including REVERIE, SOON as well as strong transferability to object-goal navigation and VLN-CE. It notably achieves a 50.9% success rate on the unseen validation splits of SOON, surpassing prior leading approaches by a margin of 13.9%.
>
---
#### [replaced 024] SimScale: Learning to Drive via Real-World Simulation at Scale
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶任务，解决真实场景数据不足问题。通过SimScale框架生成大量模拟数据，并结合真实数据训练，提升模型的鲁棒性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.23369](https://arxiv.org/pdf/2511.23369)**

> **作者:** Haochen Tian; Tianyu Li; Haochen Liu; Jiazhi Yang; Yihang Qiu; Guang Li; Junli Wang; Yinfeng Gao; Zhang Zhang; Liang Wang; Hangjun Ye; Tieniu Tan; Long Chen; Hongyang Li
>
> **备注:** Accepted to CVPR 2026. Project page: this https URL
>
> **摘要:** Achieving fully autonomous driving systems requires learning rational decisions in a wide span of scenarios, including safety-critical and out-of-distribution ones. However, such cases are underrepresented in real-world corpus collected by human experts. To complement for the lack of data diversity, we introduce a novel and scalable simulation framework capable of synthesizing massive unseen states upon existing driving logs. Our pipeline utilizes advanced neural rendering with a reactive environment to generate high-fidelity multi-view observations controlled by the perturbed ego trajectory. Furthermore, we develop a pseudo-expert trajectory generation mechanism for these newly simulated states to provide action supervision. Upon the synthesized data, we find that a simple co-training strategy on both real-world and simulated samples can lead to significant improvements in both robustness and generalization for various planning methods on challenging real-world benchmarks, up to +8.6 EPDMS on navhard and +2.9 on navtest. More importantly, such policy improvement scales smoothly by increasing simulation data only, even without extra real-world data streaming in. We further reveal several crucial findings of such a sim-real learning system, which we term SimScale, including the design of pseudo-experts and the scaling properties for different policy architectures. Simulation data and code have been released at this https URL.
>
---
#### [replaced 025] Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.10744](https://arxiv.org/pdf/2603.10744)**

> **作者:** Wenhao Sun; Ji Li; Zhaoqiang Liu
>
> **备注:** Accepted by CVPR2026. Project Page: this https URL
>
> **摘要:** Diffusion Transformers have established a new state-of-the-art in image synthesis, but the high computational cost of iterative sampling severely hampers their practical deployment. While existing acceleration methods often focus on the temporal domain, they overlook the substantial spatial redundancy inherent in the generative process, where global structures emerge long before fine-grained details are formed. The uniform computational treatment of all spatial regions represents a critical inefficiency. In this paper, we introduce Just-in-Time (JiT), a novel training-free framework that addresses this challenge by acceleration in the spatial domain. JiT formulates a spatially approximated generative ordinary differential equation (ODE) that drives the full latent state evolution based on computations from a dynamically selected, sparse subset of anchor tokens. To ensure seamless transitions as new tokens are incorporated to expand the dimensions of the latent state, we propose a deterministic micro-flow, a simple and effective finite-time ODE that maintains both structural coherence and statistical correctness. Extensive experiments on the state-of-the-art FLUX.1-dev model demonstrate that JiT achieves up to a 7x speedup with nearly lossless performance, significantly outperforming existing acceleration methods and establishing a new and superior trade-off between inference speed and generation fidelity.
>
---
#### [replaced 026] Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决视觉-语言-动作模型在遮挡和跨任务迁移中的性能问题。提出TVVE框架，通过任务感知的视角选择和视觉特征路由提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2508.05186](https://arxiv.org/pdf/2508.05186)**

> **作者:** Yongjie Bai; Zhouxia Wang; Yang Liu; Kaijun Luo; Yifan Wen; Mingtong Dai; Weixing Chen; Ziliang Chen; Lingbo Liu; Guanbin Li; Liang Lin
>
> **备注:** 24 pages, 15 figures, Project page: this https URL, Code: this https URL, Accepted at CVPR 2026
>
> **摘要:** Recent vision-language-action (VLA) models for multi-task robot manipulation often rely on fixed camera setups and shared visual encoders, which limit their performance under occlusions and during cross-task transfer. To address these challenges, we propose Task-aware Virtual View Exploration (TVVE), a framework that learns to select task-relevant virtual camera viewpoints and dynamically re-render observations from a reconstructed scene representation using the selected viewpoints. To enable efficient view selection, we train an exploration policy in a pseudo-environment. In addition, we introduce a Task-aware Mixture-of-Experts (TaskMoE) visual encoder that routes visual features to task-specialized experts, mitigating interference in multi-task learning. To evaluate robustness under distribution shifts, we construct RLBench-OG, an out-of-distribution benchmark with visual perturbations and camera pose variations. Experiments on RLBench and RLBench-OG demonstrate that TVVE achieves higher success rates than strong baselines, while real-robot experiments further confirm its robustness to visual disturbances and unseen instructions. Code and visualizations are available at: this https URL.
>
---
#### [replaced 027] From Geometric Mimicry to Comprehensive Generation: A Context-Informed Multimodal Diffusion Model for Urban Morphology Synthesis
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2409.17049](https://arxiv.org/pdf/2409.17049)**

> **作者:** Fangshuo Zhou; Huaxia Li; Liuchang Xu; Rui Hu; Sensen Wu; Liang Xu; Hailin Feng; Zhenhong Du
>
> **备注:** Accepted
>
> **摘要:** Urban morphology is fundamental to determining urban functionality and vitality. Prevailing simulation methods, however, often oversimplify morphological generation as a geometric problem, lacking a profound understanding of urban semantics and geographical context. To address this limitation, this study proposes ControlCity, a diffusion model that achieves comprehensive urban morphology generation through multimodal information fusion. We first constructed a quadruple dataset comprising ``image-text-metadata-building footprints" from 22 cities worldwide. ControlCity utilizes these multidimensional information as joint control conditions, where an enhanced ControlNet architecture encodes spatial constraints from images, while text and metadata provide semantic guidance and geographical priors respectively, collectively directing the generation process. Experimental results demonstrate that compared to unimodal baselines, this method achieves significant advantages in morphological fidelity, with visual error (FID) reduced by 71.01%, reaching 50.94, and spatial overlap (MIoU) improved by 38.46%, reaching 0.36. Furthermore, the model demonstrates robust knowledge generalization and controllability, enabling cross-city style transfer and zero-shot generation for unknown cities. Ablation studies further reveal the distinct roles of images, text, and metadata in the generation process. This study confirms that multimodal fusion is crucial for achieving the transition from ``geometric mimicry" to ``understanding-based comprehensive generation," providing a novel paradigm for urban morphology research and applications.
>
---
#### [replaced 028] Search2Motion: Training-Free Object-Level Motion Control via Attention-Consensus Search
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.16711](https://arxiv.org/pdf/2603.16711)**

> **作者:** Sainan Liu; Tz-Ying Wu; Hector A Valdez; Subarna Tripathi
>
> **备注:** 14 pages, 9 figures
>
> **摘要:** We present Search2Motion, a training-free framework for object-level motion editing in image-to-video generation. Unlike prior methods requiring trajectories, bounding boxes, masks, or motion fields, Search2Motion adopts target-frame-based control, leveraging first-last-frame motion priors to realize object relocation while preserving scene stability without fine-tuning. Reliable target-frame construction is achieved through semantic-guided object insertion and robust background inpainting. We further show that early-step self-attention maps predict object and camera dynamics, offering interpretable user feedback and motivating ACE-Seed (Attention Consensus for Early-step Seed selection), a lightweight search strategy that improves motion fidelity without look-ahead sampling or external evaluators. Noting that existing benchmarks conflate object and camera motion, we introduce S2M-DAVIS and S2M-OMB for stable-camera, object-only evaluation, alongside FLF2V-obj metrics that isolate object artifacts without requiring ground-truth trajectories. Search2Motion consistently outperforms baselines on FLF2V-obj and VBench.
>
---
#### [replaced 029] Unifying Heterogeneous Degradations: Uncertainty-Aware Diffusion Bridge Model for All-in-One Image Restoration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.21592](https://arxiv.org/pdf/2601.21592)**

> **作者:** Luwei Tu; Jiawei Wu; Xing Luo; Zhi Jin
>
> **摘要:** All-in-One Image Restoration (AiOIR) faces the fundamental challenge in reconciling conflicting optimization objectives across heterogeneous degradations. Existing methods are often constrained by coarse-grained control mechanisms or fixed mapping schedules, yielding suboptimal adaptation. To address this, we propose an Uncertainty-Aware Diffusion Bridge Model (UDBM), which innovatively reformulates AiOIR as a stochastic transport problem steered by pixel-wise uncertainty. By introducing a relaxed diffusion bridge formulation which replaces the strict terminal constraint with a relaxed constraint, we model the uncertainty of degradations while theoretically resolving the drift singularity inherent in standard diffusion bridges. Furthermore, we devise a dual modulation strategy: the noise schedule aligns diverse degradations into a shared high-entropy latent space, while the path schedule adaptively regulates the transport trajectory motivated by the viscous dynamics of entropy regularization. By effectively rectifying the transport geometry and dynamics, UDBM achieves state-of-the-art performance across diverse restoration tasks within a single inference step.
>
---
#### [replaced 030] Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.22496](https://arxiv.org/pdf/2509.22496)**

> **作者:** Ruoyu Chen; Xiaoqing Guo; Kangwei Liu; Siyuan Liang; Shiming Liu; Qunli Zhang; Laiyuan Wang; Hua Zhang; Xiaochun Cao
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Multimodal large language models (MLLMs) have demonstrated remarkable capabilities in aligning visual inputs with natural language outputs. Yet, the extent to which generated tokens depend on visual modalities remains poorly understood, limiting interpretability and reliability. In this work, we present EAGLE, a lightweight black-box framework for explaining autoregressive token generation in MLLMs. EAGLE attributes any selected tokens to compact perceptual regions while quantifying the relative influence of language priors and perceptual evidence. The framework introduces an objective function that unifies sufficiency (insight score) and indispensability (necessity score), optimized via greedy search over sparsified image regions for faithful and efficient attribution. Beyond spatial attribution, EAGLE performs modality-aware analysis that disentangles what tokens rely on, providing fine-grained interpretability of model decisions. Extensive experiments across open-source MLLMs show that EAGLE consistently outperforms existing methods in faithfulness, localization, and hallucination diagnosis, while requiring substantially less GPU memory. These results highlight its effectiveness and practicality for advancing the interpretability of MLLMs.
>
---
#### [replaced 031] YOLO26: An Analysis of NMS-Free End to End Framework for Real-Time Object Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.12882](https://arxiv.org/pdf/2601.12882)**

> **作者:** Sudip Chakrabarty
>
> **摘要:** The ``You Only Look Once'' (YOLO) framework has long served as a standard for real-time object detection, though traditional iterations have utilized Non-Maximum Suppression (NMS) post-processing, which introduces specific latency and hyperparameter variables. This paper presents a comprehensive architectural analysis of YOLO26, a model that shifts toward a native end-to-end learning strategy by eliminating NMS. This study examines the core mechanisms driving this framework: the MuSGD optimizer for backbone stabilization, Small-Target-Aware Label Assignment (STAL), and ProgLoss for dynamic supervision. To contextualize its performance, this article reviews exhaustive benchmark data from the COCO \texttt{val2017} leaderboard. This evaluation provides an objective comparison of YOLO26 across various model scales (Nano to Extra-Large) against both prior CNN lineages and contemporary Transformer-based architectures (e.g., RT-DETR, DEIM, RF-DETR), detailing the observed speed-accuracy trade-offs and parameter requirements without asserting a singular optimal model. Additionally, the analysis covers the framework's unified multi-task capabilities, including the YOLOE-26 open-vocabulary module for promptable detection. Ultimately, this paper serves to document how decoupling representation learning from heuristic post-processing impacts the "Export Gap" and deterministic latency in modern edge-based computer vision deployments.
>
---
#### [replaced 032] Vision to Geometry: 3D Spatial Memory for Sequential Embodied MLLM Reasoning and Exploration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.02458](https://arxiv.org/pdf/2512.02458)**

> **作者:** Zhongyi Cai; Yi Du; Chen Wang; Yu Kong
>
> **备注:** Computer Vision
>
> **摘要:** Embodied agents are expected to assist humans by actively exploring unknown environments and reasoning about spatial contexts. When deployed in real life, agents often face sequential tasks where each new task follows the completion of the previous one and may include infeasible objectives, such as searching for non-existent objects. However, most existing research focuses on isolated goals, overlooking the core challenge of sequential tasks: the ability to reuse spatial knowledge accumulated from previous explorations to guide subsequent reasoning and exploration. In this work, we investigate this underexplored yet practically significant embodied AI challenge. Specifically, we propose 3DSPMR, a 3D SPatial Memory Reasoning framework that utilizes Field-of-View (FoV) coverage as an explicit geometric prior. By integrating FoV-based constraints, 3DSPMR significantly enhances an agent's memory, reasoning, and exploration capabilities across sequential tasks. To facilitate research in this area, we further introduce SEER-Bench, a novel Sequential Embodied Exploration and Reasoning Benchmark that spans two foundational tasks: Embodied Question Answering (EQA) and Embodied Multi-modal Navigation (EMN). SEER-Bench uniquely incorporates both feasible and infeasible tasks to provide a rigorous and comprehensive evaluation of agent performance. Extensive experiments verify that 3DSPMR achieves substantial performance gains on both sequential EQA and EMN tasks.
>
---
#### [replaced 033] Halfway to 3D: Ensembling 2.5D and 3D Models for Robust COVID-19 CT Diagnosis
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.14832](https://arxiv.org/pdf/2603.14832)**

> **作者:** Tuan-Anh Yang; Bao V. Q. Bui; Chanh-Quang Vo-Van; Truong-Son Hy
>
> **摘要:** We propose a deep learning framework for COVID-19 detection and disease classification from chest CT scans that integrates both 2.5D and 3D representations to capture complementary slice-level and volumetric information. The 2.5D branch processes multi-view CT slices (axial, coronal, sagittal) using a DINOv3 vision transformer to extract robust visual features, while the 3D branch employs a ResNet-18 architecture to model volumetric context and is pretrained with Variance Risk Extrapolation (VREx) followed by supervised contrastive learning to improve cross-source robustness. Predictions from both branches are combined through logit-level ensemble inference. Experiments on the PHAROS-AIF-MIH benchmark demonstrate the effectiveness of the proposed approach: for binary COVID-19 detection, the ensemble achieves 94.48% accuracy and a 0.9426 Macro F1-score, outperforming both individual models, while for multi-class disease classification the 2.5D DINOv3 model achieves the best performance with 79.35% accuracy and a 0.7497 Macro F1-score. These results highlight the benefit of combining pretrained slice-based representations with volumetric modeling for robust multi-source medical imaging analysis. Code is available at this https URL
>
---
#### [replaced 034] Draft and Refine with Visual Experts
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11005](https://arxiv.org/pdf/2511.11005)**

> **作者:** Sungheon Jeong; Ryozo Masukawa; Jihong Park; Sanggeon Yun; Wenjun Huang; Hanning Chen; Mahdi Imani; Mohsen Imani
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** While recent Large Vision-Language Models (LVLMs) exhibit strong multimodal reasoning abilities, they often produce ungrounded or hallucinated responses because they rely too heavily on linguistic priors instead of visual evidence. This limitation highlights the absence of a quantitative measure of how much these models actually use visual information during reasoning. We propose Draft and Refine (DnR), an agent framework driven by a question-conditioned utilization metric. The metric quantifies the model's reliance on visual evidence by first constructing a query-conditioned relevance map to localize question-specific cues and then measuring dependence through relevance-guided probabilistic masking. Guided by this metric, the DnR agent refines its initial draft using targeted feedback from external visual experts. Each expert's output (such as boxes or masks) is rendered as visual cues on the image, and the model is re-queried to select the response that yields the largest improvement in utilization. This process strengthens visual grounding without retraining or architectural changes. Experiments across VQA and captioning benchmarks show consistent accuracy gains and reduced hallucination, demonstrating that measuring visual utilization provides a principled path toward more interpretable and evidence-driven multimodal agent systems. Code is available at this https URL.
>
---
#### [replaced 035] AutoMoT: A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出一种统一的视觉-语言-动作模型AutoMoT，解决端到端自动驾驶中推理与决策的协同问题，通过异步Transformer混合架构提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.14851](https://arxiv.org/pdf/2603.14851)**

> **作者:** Wenhui Huang; Songyan Zhang; Qihang Huang; Zhidong Wang; Zhiqi Mao; Collister Chua; Zhan Chen; Long Chen; Chen Lv
>
> **摘要:** Integrating vision-language models (VLMs) into end-to-end (E2E) autonomous driving (AD) systems has shown promise in improving scene understanding. However, existing integration strategies suffer from several limitations: they either struggle to resolve distribution misalignment between reasoning and action spaces, underexploit the general reasoning capabilities of pretrained VLMs, or incur substantial inference latency during action policy generation, which degrades driving performance. To address these challenges, we propose \OURS in this work, an end-to-end AD framework that unifies reasoning and action generation within a single vision-language-action (VLA) model. Our approach leverages a mixture-of-transformer (MoT) architecture with joint attention sharing, which preserves the general reasoning capabilities of pre-trained VLMs while enabling efficient fast-slow inference through asynchronous execution at different task frequencies. Extensive experiments on multiple benchmarks, under both open- and closed-loop settings, demonstrate that \OURS achieves competitive performance compared to state-of-the-art methods. We further investigate the functional boundary of pre-trained VLMs in AD, examining when AD-tailored fine-tuning is necessary. Our results show that pre-trained VLMs can achieve competitive multi-task scene understanding performance through semantic prompting alone, while fine-tuning remains essential for action-level tasks such as decision-making and trajectory planning. We refer to \href{this https URL}{Project Page} for the demonstration videos and qualitative results.
>
---
#### [replaced 036] Multimodal Emotion Recognition via Bi-directional Cross-Attention and Temporal Modeling
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.11971](https://arxiv.org/pdf/2603.11971)**

> **作者:** Junhyeong Byeon; Jeongyeol Kim; Sejoon Lim
>
> **备注:** 7 pages
>
> **摘要:** Expression recognition in in-the-wild video data remains challenging due to substantial variations in facial appearance, background conditions, audio noise, and the inherently dynamic nature of human affect. Relying on a single modality, such as facial expressions or speech, is often insufficient for capturing these complex emotional cues. To address this limitation, we propose a multimodal emotion recognition framework for the Expression (EXPR) task in the 10th Affective Behavior Analysis in-the-wild (ABAW) Challenge. Our framework builds on large-scale pre-trained models for visual and audio representation learning and integrates them in a unified multimodal architecture. To better capture temporal patterns in facial expression sequences, we incorporate temporal visual modeling over video windows. We further introduce a bi-directional cross-attention fusion module that enables visual and audio features to interact in a symmetric manner, facilitating cross-modal contextualization and complementary emotion understanding. In addition, we employ a text-guided contrastive objective to encourage semantically meaningful visual representations through alignment with emotion-related text prompts. Experimental results on the ABAW 10th EXPR benchmark demonstrate the effectiveness of the proposed framework, achieving a Macro F1 score of 0.32 compared to the baseline score of 0.25, and highlight the benefit of combining temporal visual modeling, audio representation learning, and cross-modal fusion for robust emotion recognition in unconstrained real-world environments.
>
---
#### [replaced 037] LaS-Comp: Zero-shot 3D Completion with Latent-Spatial Consistency
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出LaS-Comp，解决3D形状补全任务，通过两阶段设计实现零样本、类别无关的补全，提升完整性与边界一致性。**

- **链接: [https://arxiv.org/pdf/2602.18735](https://arxiv.org/pdf/2602.18735)**

> **作者:** Weilong Yan; Haipeng Li; Hao Xu; Nianjin Ye; Yihao Ai; Shuaicheng Liu; Jingyu Hu
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** This paper introduces LaS-Comp, a zero-shot and category-agnostic approach that leverages the rich geometric priors of 3D foundation models to enable 3D shape completion across diverse types of partial observations. Our contributions are threefold: First, \ourname{} harnesses these powerful generative priors for completion through a complementary two-stage design: (i) an explicit replacement stage that preserves the partial observation geometry to ensure faithful completion; and (ii) an implicit refinement stage ensures seamless boundaries between the observed and synthesized regions. Second, our framework is training-free and compatible with different 3D foundation models. Third, we introduce Omni-Comp, a comprehensive benchmark combining real-world and synthetic data with diverse and challenging partial patterns, enabling a more thorough and realistic evaluation. Both quantitative and qualitative experiments demonstrate that our approach outperforms previous state-of-the-art approaches. Our code and data will be available at \href{this https URL}{LaS-Comp}.
>
---
#### [replaced 038] M3DLayout: A Multi-Source Dataset of 3D Indoor Layouts and Structured Descriptions for 3D Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.23728](https://arxiv.org/pdf/2509.23728)**

> **作者:** Yiheng Zhang; Zhuojiang Cai; Mingdao Wang; Meitong Guo; Tianxiao Li; Li Lin; Yuwang Wang
>
> **备注:** CVPR 2026; Project Page: this https URL
>
> **摘要:** In text-driven 3D scene generation, object layout serves as a crucial intermediate representation that bridges high-level language instructions with detailed geometric output. It not only provides a structural blueprint for ensuring physical plausibility but also supports semantic controllability and interactive editing. However, the learning capabilities of current 3D indoor layout generation models are constrained by the limited scale, diversity, and annotation quality of existing datasets. To address this, we introduce M3DLayout, a large-scale, multi-source dataset for 3D indoor layout generation. M3DLayout comprises 21,367 layouts and over 433k object instances, integrating three distinct sources: real-world scans, professional CAD designs, and procedurally generated scenes. Each layout is paired with detailed structured text describing global scene summaries, relational placements of large furniture, and fine-grained arrangements of smaller items. This diverse and richly annotated resource enables models to learn complex spatial and semantic patterns across a wide variety of indoor environments. To assess the potential of M3DLayout, we establish a benchmark using both a text-conditioned diffusion model and a text-conditioned autoregressive model. Experimental results demonstrate that our dataset provides a solid foundation for training layout generation models. Its multi-source composition enhances diversity, notably through the Inf3DLayout subset which provides rich small-object information, enabling the generation of more complex and detailed scenes. We hope that M3DLayout can serve as a valuable resource for advancing research in text-driven 3D scene synthesis. All dataset and code will be made public upon acceptance.
>
---
#### [replaced 039] OccTENS: 3D Occupancy World Model via Temporal Next-Scale Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.03887](https://arxiv.org/pdf/2509.03887)**

> **作者:** Bu Jin; Songen Gu; Xiaotao Hu; Yupeng Zheng; Xiaoyang Guo; Qian Zhang; Xiaoxiao Long; Wei Yin
>
> **摘要:** In this paper, we propose OccTENS, a generative occupancy world model that enables controllable, high-fidelity long-term occupancy generation while maintaining computational efficiency. Different from visual generation, the occupancy world model must capture the fine-grained 3D geometry and dynamic evolution of the 3D scenes, posing great challenges for the generative models. Recent approaches based on autoregression (AR) have demonstrated the potential to predict vehicle movement and future occupancy scenes simultaneously from historical observations, but they typically suffer from \textbf{inefficiency}, \textbf{temporal degradation} in long-term generation and \textbf{lack of controllability}. To holistically address these issues, we reformulate the occupancy world model as a temporal next-scale prediction (TENS) task, which decomposes the temporal sequence modeling problem into the modeling of spatial scale-by-scale generation and temporal scene-by-scene prediction. With a \textbf{TensFormer}, OccTENS can effectively manage the temporal causality and spatial relationships of occupancy sequences in a flexible and scalable way. To enhance the pose controllability, we further propose a holistic pose aggregation strategy, which features a unified sequence modeling for occupancy and ego-motion. Experiments show that OccTENS outperforms the state-of-the-art method with both higher occupancy quality and faster inference time.
>
---
#### [replaced 040] IRIS-SLAM: Unified Geo-Instance Representations for Robust Semantic Localization and Mapping
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出IRIS-SLAM，解决语义定位与建图中的几何-实例统一表示问题，提升地图一致性与回环检测可靠性。**

- **链接: [https://arxiv.org/pdf/2602.18709](https://arxiv.org/pdf/2602.18709)**

> **作者:** Tingyang Xiao; Liu Liu; Wei Feng; Zhengyu Zou; Xiaolin Zhou; Wei Sui; Hao Li; Dingwen Zhang; Zhizhong Su
>
> **备注:** The reason for this withdrawal is that the current version was submitted without the final review and formal authorization of all co-authors. To ensure the academic consensus and integrity of our research group, we have decided to withdraw this submission from the repository
>
> **摘要:** Geometry foundation models have significantly advanced dense geometric SLAM, yet existing systems often lack deep semantic understanding and robust loop closure capabilities. Meanwhile, contemporary semantic mapping approaches are frequently hindered by decoupled architectures and fragile data association. We propose IRIS-SLAM, a novel RGB semantic SLAM system that leverages unified geometric-instance representations derived from an instance-extended foundation model. By extending a geometry foundation model to concurrently predict dense geometry and cross-view consistent instance embeddings, we enable a semantic-synergized association mechanism and instance-guided loop closure detection. Our approach effectively utilizes viewpoint-agnostic semantic anchors to bridge the gap between geometric reconstruction and open-vocabulary mapping. Experimental results demonstrate that IRIS-SLAM significantly outperforms state-of-the-art methods, particularly in map consistency and wide-baseline loop closure reliability.
>
---
#### [replaced 041] Parameterizing Dataset Distillation via Gaussian Splatting
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.26219](https://arxiv.org/pdf/2509.26219)**

> **作者:** Chenyang Jiang; Zhengcen Li; Hang Zhao; Qiben Shan; Shaocong Wu; Jingyong Su
>
> **备注:** 19 pages; Code is available on this https URL
>
> **摘要:** Dataset distillation aims to compress training data while preserving training-aware knowledge, alleviating the reliance on large-scale datasets in modern model training. Dataset parameterization provides a more efficient storage structure for dataset distillation, reducing redundancy and accommodating richer information. However, existing methods either rely on complex auxiliary modules or fail to balance representational capacity and efficiency. In this paper, we propose GSDD, a simple, novel, and effective dataset parameterization technique for Dataset Distillation based on Gaussian Splatting. We adapt CUDA-based splatting operators for parallel training in batch, enabling high-quality rendering with minimal computational and memory overhead. Gaussian primitives can effectively capture meaningful training features, allowing a sparse yet expressive representation of individual images. Leveraging both high representational capacity and efficiency, GSDD substantially increases the diversity of distilled datasets under a given storage budget, thereby improving distillation performance. Beyond achieving competitive results on multiple standard benchmarks, GSDD also delivers significant performance gains on large-scale datasets such as ImageNet-1K and on video distillation tasks. In addition, we conduct comprehensive benchmarks to evaluate the computational efficiency, memory footprint, and cross-GPU architectural stability of GSDD. Code is available on this https URL
>
---
#### [replaced 042] MagicWorld: Towards Long-Horizon Stability for Interactive Video World Exploration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18886](https://arxiv.org/pdf/2511.18886)**

> **作者:** Guangyuan Li; Bo Li; Jinwei Chen; Xiaobin Hu; Lei Zhao; Peng-Tao Jiang
>
> **摘要:** Recent interactive video world model methods generate scene evolution conditioned on user instructions. Although they achieve impressive results, two key limitations remain. First, they exhibit motion drift in complex environments with multiple interacting subjects, where dynamic subjects fail to follow realistic motion patterns during scene evolution. Second, they suffer from error accumulation in long-horizon interactions, where autoregressive generation gradually drifts from earlier scene states and causes structural and semantic inconsistencies. In this paper, we propose MagicWorld, an interactive video world model built upon an autoregressive framework. To address motion drift, we incorporate a flow-guided motion preservation constraint that mitigates motion degradation in dynamic subjects, encouraging realistic motion patterns and stable interactions during scene evolution. To mitigate error accumulation in long-horizon interactions, we design two complementary strategies, including a history cache retrieval strategy and an enhanced interactive training strategy. The former reinforces historical scene states by retrieving past generations during interaction, while the latter adopts multi-shot aggregated distillation with dual-reward weighting for interactive training, enhancing long-term stability and reducing error accumulation. In addition, we construct RealWM120K, a real-world dataset with diverse city-walk videos and multimodal annotations to support dynamic perception and long-horizon world modeling. Experimental results demonstrate that MagicWorld improves motion realism and alleviates error accumulation during long-horizon interactions.
>
---
#### [replaced 043] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.11298](https://arxiv.org/pdf/2603.11298)**

> **作者:** Dingqiang Ye; Jiacong Xu; Jianglu Ping; Yuxiang Guo; Chao Fan; Vishal M. Patel
>
> **摘要:** High dynamic range (HDR) novel view synthesis (NVS) aims to reconstruct HDR scenes from multi-exposure low dynamic range (LDR) images. Existing HDR pipelines heavily rely on known camera poses, well-initialized dense point clouds, and time-consuming per-scene optimization. Current feed-forward alternatives overlook the HDR problem by assuming exposure-invariant appearance. To bridge this gap, we propose InstantHDR, a feed-forward network that reconstructs 3D HDR scenes from uncalibrated multi-exposure LDR collections in a single forward pass. Specifically, we design a geometry-guided appearance modeling for multi-exposure fusion, and a meta-network for generalizable scene-specific tone mapping. Due to the lack of HDR scene data, we build a pre-training dataset, called HDR-Pretrain, for generalizable feed-forward HDR models, featuring 168 Blender-rendered scenes, diverse lighting types, and multiple camera response functions. Comprehensive experiments show that our InstantHDR delivers comparable synthesis performance to the state-of-the-art optimization-based HDR methods while enjoying $\sim700\times$ and $\sim20\times$ reconstruction speed improvement with our single-forward and post-optimization settings. All code, models, and datasets will be released after the review process.
>
---
#### [replaced 044] Generative Hints
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.02933](https://arxiv.org/pdf/2511.02933)**

> **作者:** Andy Dimnaku; Abdullah Yusuf Kavranoglu; Yaser Abu-Mostafa
>
> **备注:** 15 pages, 15 figures
>
> **摘要:** Data augmentation is widely used in vision to introduce variation and mitigate overfitting, by enabling models to learn invariant properties. However, augmentation only indirectly captures these properties and does not explicitly constrain the learned function to satisfy them beyond the empirical training set. We propose generative hints, a training methodology that directly enforces known functional invariances over the input distribution. Our approach leverages a generative model trained on the training data to approximate the input distribution and to produce unlabeled synthetic images, which we refer to as virtual examples. On these virtual examples, we impose hint objectives that explicitly constrain the model's predictions to satisfy known invariance properties, such as spatial invariance. Although the original training dataset is fully labeled, generative hints train the model in a semi-supervised manner by combining the standard classification objective on real data with an auxiliary hint objectives applied to unlabeled virtual examples. Across multiple datasets, architectures, invariance types, and loss functions, generative hints consistently outperform standard data augmentation, achieving accuracy improvements of up to 2.10% on fine-grained visual classification benchmarks and an average gain of 1.29% on the CheXpert medical imaging dataset.
>
---
#### [replaced 045] Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step Few-shot Image Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.18281](https://arxiv.org/pdf/2511.18281)**

> **作者:** Yara Bahram; Melodie Desbos; Mohammadhadi Shateri; Eric Granger
>
> **备注:** Accepted at IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** Diffusion models (DMs) produce high-quality images, yet their sampling remains costly when adapted to new domains. Distilled DMs are faster but typically remain confined within their teacher's domain. Thus, fast and high-quality generation for novel domains relies on two-stage pipelines: Adapt-then-Distill or Distill-then-Adapt. However, both add design complexity and often degrade quality or diversity. We introduce Uni-DAD, a single-stage pipeline that unifies DM distillation and adaptation. It couples two training signals: (i) a dual-domain distribution-matching distillation (DMD) objective that guides the student toward the distributions of the source teacher and a target teacher, and (ii) a multi-head generative adversarial network (GAN) loss that encourages target realism across multiple feature scales. The source domain distillation preserves diverse source knowledge, while the multi-head GAN stabilizes training and reduces overfitting, especially in few-shot regimes. The inclusion of a target teacher facilitates adaptation to more structurally distant domains. We evaluate Uni-DAD on two comprehensive benchmarks for few-shot image generation (FSIG) and subject-driven personalization (SDP) using diffusion backbones. It delivers better or comparable quality to state-of-the-art (SoTA) adaptation methods even with less than 4 sampling steps, and often surpasses two-stage pipelines in quality and diversity. Code: this https URL.
>
---
#### [replaced 046] SemanticFace: Semantic Facial Action Estimation via Semantic Distillation in Interpretable Space
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.14827](https://arxiv.org/pdf/2603.14827)**

> **作者:** Zejian Kang; Kai Zheng; Yuanchen Fei; Wentao Yang; Hongyuan Zou; Xiangru Huang
>
> **摘要:** Facial action estimation from a single image is often formulated as predicting or fitting parameters in compact expression spaces, which lack explicit semantic interpretability. However, many practical applications, such as avatar control and human-computer interaction, require interpretable facial actions that correspond to meaningful muscle movements. In this work, we propose SemanticFace, a framework for facial action estimation in the interpretable ARKit blendshape space that reformulates coefficient prediction as structured semantic reasoning. SemanticFace adopts a two-stage semantic distillation paradigm: it first derives structured semantic supervision from ground-truth ARKit coefficients and then distills this knowledge into a multimodal large language model to predict interpretable facial action coefficients from images. Extensive experiments demonstrate that language-aligned semantic supervision improves both coefficient accuracy and perceptual consistency, while enabling strong cross-identity generalization and robustness to large domain shifts, including cartoon faces.
>
---
#### [replaced 047] Self-Attention And Beyond the Infinite: Towards Linear Transformers with Infinite Self-Attention
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.00175](https://arxiv.org/pdf/2603.00175)**

> **作者:** Giorgio Roffo; Luke Palmer; Nilli Lavie
>
> **备注:** This work builds in part on conceptual directions previously explored in the MVL/Toyota Motor Europe collaboration. Code available: HF: this https URL Github: this https URL
>
> **摘要:** The quadratic cost of softmax attention limits Transformer scalability in high-resolution vision. We introduce Infinite Self-Attention (InfSA), a spectral reformulation that treats each attention layer as a diffusion step on a content-adaptive token graph, accumulating multi-hop interactions through a discounted Neumann series over attention matrices. This links self-attention to classical graph centrality (Katz, PageRank, eigenvector centrality) for interpretable token weighting. We also show the Neumann kernel equals the fundamental matrix of an absorbing Markov chain, so a token's centrality is its expected number of random-walk visits before absorption. We then propose Linear-InfSA, a linear-time variant that approximates the principal eigenvector of the implicit attention operator without forming the full attention matrix. It keeps an auxiliary state of fixed size proportional to per-head dimension dh (independent of sequence length N), is drop-in compatible with Vision Transformers, and supports stable training at 4096 by 4096 and inference at 9216 by 9216 (about 332k tokens). In a 4-layer ViT (53.5M parameters, 59 GFLOPs at 224 by 224), Linear-InfSA reaches 84.7% top-1 on ImageNet-1K, a +3.2 point architectural gain over an equal-depth softmax ViT trained with the same recipe. On ImageNet-V2, InfViT variants outperform all compared baselines (up to 79.8% vs 76.8%), indicating robustness under distribution shift. On an A100 40GB GPU, Linear-InfViT runs at 231 images/s and 0.87 J/image (13x better throughput and energy than equal-depth ViT) and is the only tested model to complete 9216 by 9216 inference without out-of-memory. The linear approximation closely matches the dominant eigenvector of the quadratic operator (cosine 0.985). Code available at: this https URL or this https URL
>
---
#### [replaced 048] TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.02341](https://arxiv.org/pdf/2512.02341)**

> **作者:** Fengyi Zhang; Tianjun Zhang; Kasra Khosoussi; Zheng Zhang; Zi Huang; Yadan Luo
>
> **备注:** CVPR 2026
>
> **摘要:** 3D vision foundation models have shown strong generalization in reconstructing key 3D attributes from uncalibrated images through a single feed-forward pass. However, when deployed in online settings such as driving scenarios, predictions are made over temporal windows, making it non-trivial to maintain consistency across time. Recent strategies align consecutive predictions by solving global transformation, yet our analysis reveals their fundamental limitations in assumption validity, local alignment scope, and robustness under noisy geometry. In this work, we propose a higher-DOF and long-term alignment framework based on Thin Plate Spline, leveraging globally propagated control points to correct spatially varying inconsistencies. In addition, we adopt a point-agnostic submap registration design that is inherently robust to noisy geometry predictions. The proposed framework is fully plug-and-play, compatible with diverse 3D foundation models and camera configurations (e.g., monocular or surround-view). Extensive experiments demonstrate that our method consistently yields more coherent geometry and lower trajectory errors across multiple datasets, backbone models, and camera setups, highlighting its robustness and generality. Code is available at this https URL.
>
---
#### [replaced 049] S-VAM: Shortcut Video-Action Model by Self-Distilling Geometric and Semantic Foresight
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出S-VAM，解决视频动作模型实时与高精度不足的问题。通过单次前向传播和自蒸馏策略，提升机器人操作效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.16195](https://arxiv.org/pdf/2603.16195)**

> **作者:** Haodong Yan; Zhide Zhong; Jiaguan Zhu; Junjie He; Weilin Yuan; Wenxuan Song; Xin Gong; Yingjie Cai; Guanyi Zhao; Xu Yan; Bingbing Liu; Ying-Cong Chen; Haoang Li
>
> **摘要:** Video action models (VAMs) have emerged as a promising paradigm for robot learning, owing to their powerful visual foresight for complex manipulation tasks. However, current VAMs, typically relying on either slow multi-step video generation or noisy one-step feature extraction, cannot simultaneously guarantee real-time inference and high-fidelity foresight. To address this limitation, we propose S-VAM, a shortcut video-action model that foresees coherent geometric and semantic representations via a single forward pass. Serving as a stable blueprint, these foreseen representations significantly simplify the action prediction. To enable this efficient shortcut, we introduce a novel self-distillation strategy that condenses structured generative priors of multi-step denoising into one-step inference. Specifically, vision foundation model (VFM) representations extracted from the diffusion model's own multi-step generated videos provide teacher targets. Lightweight decouplers, as students, learn to directly map noisy one-step features to these targets. Extensive experiments in simulation and the real world demonstrate that our S-VAM outperforms state-of-the-art methods, enabling efficient and precise manipulation in complex environments. Our project page is this https URL
>
---
#### [replaced 050] Context-Nav: Context-Driven Exploration and Viewpoint-Aware 3D Spatial Reasoning for Instance Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于文本目标实例导航任务，解决在同类物体中精确定位目标的问题。提出Context-Nav方法，通过上下文引导探索和3D空间验证实现精准导航。**

- **链接: [https://arxiv.org/pdf/2603.09506](https://arxiv.org/pdf/2603.09506)**

> **作者:** Won Shik Jang; Ue-Hwan Kim
>
> **备注:** Accepted to CVPR 2026. Code is available at this https URL
>
> **摘要:** Text-goal instance navigation (TGIN) asks an agent to resolve a single, free-form description into actions that reach the correct object instance among same-category distractors. We present \textit{Context-Nav}, which elevates long, contextual captions from a local matching cue to a global exploration prior and verifies candidates through 3D spatial reasoning. First, we compute dense text-image alignments for a value map that ranks frontiers -- guiding exploration toward regions consistent with the entire description rather than early detections. Second, upon observing a candidate, we perform a viewpoint-aware relation check: the agent samples plausible observer poses, aligns local frames, and accepts a target only if the spatial relations can be satisfied from at least one viewpoint. The pipeline requires no task-specific training or fine-tuning; we attain state-of-the-art performance on InstanceNav and CoIN-Bench. Ablations show that (i) encoding full captions into the value map avoids wasted motion and (ii) explicit, viewpoint-aware 3D verification prevents semantically plausible but incorrect stops. This suggests that geometry-grounded spatial reasoning is a scalable alternative to heavy policy training or human-in-the-loop interaction for fine-grained instance disambiguation in cluttered 3D scenes.
>
---
#### [replaced 051] Learning from Oblivion: Predicting Knowledge Overflowed Weights via Retrodiction of Forgetting
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.05059](https://arxiv.org/pdf/2508.05059)**

> **作者:** Jinhyeok Jang; Jaehong Kim; Jung Uk Kim
>
> **备注:** To appear in CVPR 2026
>
> **摘要:** Pre-trained weights have become a cornerstone of modern deep learning, enabling efficient knowledge transfer and improving downstream task performance, especially in data-scarce scenarios. However, a fundamental question remains: how can we obtain better pre-trained weights that encapsulate more knowledge beyond the given dataset? In this work, we introduce KNowledge-Overflowed Weights (KNOW) prediction, a novel strategy that leverages structured forgetting and its inversion to synthesize knowledge-enriched weights. Our key insight is that sequential fine-tuning on progressively downsized datasets induces a structured forgetting process, which can be modeled and reversed to recover knowledge as if trained on a larger dataset. We construct a dataset of weight transitions governed by this controlled forgetting and employ meta-learning to model weight prediction effectively. Specifically, our KNowledge-Overflowed Weights Nowcaster (KNOWN) acts as a hyper-model that learns the general evolution of weights and predicts enhanced weights with improved generalization. Extensive experiments across diverse datasets and architectures demonstrate that KNOW prediction consistently outperforms Naive fine-tuning and simple weight prediction, leading to superior downstream performance. Our work provides a new perspective on reinterpreting forgetting dynamics to push the limits of knowledge transfer. The code and pre-trained model are available at this https URL
>
---
#### [replaced 052] AvatarForcing: One-Step Streaming Talking Avatars via Local-Future Sliding-Window Denoising
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.14331](https://arxiv.org/pdf/2603.14331)**

> **作者:** Liyuan Cui; Wentao Hu; Wenyuan Zhang; Zesong Yang; Fan Shi; Xiaoqiang Liu
>
> **摘要:** Real-time talking avatar generation requires low latency and minute-level temporal stability. Autoregressive (AR) forcing enables streaming inference but suffers from exposure bias, which causes errors to accumulate and become irreversible over long rollouts. In contrast, full-sequence diffusion transformers mitigate drift but remain computationally prohibitive for real-time long-form synthesis. We present AvatarForcing, a one-step streaming diffusion framework that denoises a fixed local-future window with heterogeneous noise levels and emits one clean block per step under constant per-step cost. To stabilize unbounded streams, the method introduces dual-anchor temporal forcing: a style anchor that re-indexes RoPE to maintain a fixed relative position with respect to the active window and applies anchor-audio zero-padding, and a temporal anchor that reuses recently emitted clean blocks to ensure smooth transitions. Real-time one-step inference is enabled by two-stage streaming distillation with offline ODE backfill and distribution matching. Experiments on standard benchmarks and a new 400-video long-form benchmark show strong visual quality and lip synchronization at 34 ms/frame using a 1.3B-parameter student model for realtime streaming. Our page is available at: this https URL
>
---
#### [replaced 053] Workflow-Aware Structured Layer Decomposition for Illustration Production
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2603.14925](https://arxiv.org/pdf/2603.14925)**

> **作者:** Tianyu Zhang; Dongchi Li; Keiichi Sawada; Haoran Xie
>
> **备注:** 17 pages, 15 figures
>
> **摘要:** Recent generative image editing methods adopt layered representations to mitigate the entangled nature of raster images and improve controllability, typically relying on object-based segmentation. However, such strategies may fail to capture the structural and stylized properties of human-created images, such as anime illustrations. To solve this issue, we propose a workflow-aware structured layer decomposition framework tailored to the illustration production of anime artwork. Inspired by the creation pipeline of anime production, our method decomposes the illustration into semantically meaningful production layers, including line art, flat color, shadow, and highlight. To decouple all these layers, we introduce lightweight layer semantic embeddings to provide specific task guidance for each layer. Furthermore, a set of layer-wise losses is incorporated to supervise the training process of individual layers. To overcome the lack of ground-truth layered data, we construct a high-quality illustration dataset that simulated the standard anime production workflow. Experiments demonstrate that the accurate and visually coherent layer decompositions were achieved by using our method. We believe that the resulting layered representation further enables downstream tasks such as recoloring and embedding texture, supporting content creation, and illustration editing. Code is available at: this https URL
>
---
#### [replaced 054] A Comprehensive Benchmark of Histopathology Foundation Models for Kidney Digital Pathology Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.15967](https://arxiv.org/pdf/2603.15967)**

> **作者:** Harishwar Reddy Kasireddy; Patricio S. La Rosa; Akshita Gupta; Anindya S. Paul; Jamie L. Fermin; William L. Clapp; Meryl A. Waldman; Tarek M. El-Ashkar; Sanjay Jain; Luis Rodrigues; Kuang Yu Jen; Avi Z. Rosenberg; Michael T. Eadon; Jeffrey B. Hodgin; Pinaki Sarder
>
> **备注:** 31 Pages, 14 Tables, 12 figures, Co-correspondence to jhodgin@med.this http URL and this http URL@ufl.edu
>
> **摘要:** Histopathology foundation models (HFMs), pretrained on large-scale cancer datasets, have advanced computational pathology. However, their applicability to non-cancerous chronic kidney disease remains underexplored, despite coexistence of renal pathology with malignancies such as renal cell and urothelial carcinoma. We systematically evaluate 11 publicly available HFMs across 11 kidney-specific downstream tasks spanning multiple stains (PAS, H&E, PASM, and IHC), spatial scales (tile and slide-level), task types (classification, regression, and copy detection), and clinical objectives, including detection, diagnosis, and prognosis. Tile-level performance is assessed using repeated stratified group cross-validation, while slide-level tasks are evaluated using repeated nested stratified cross-validation. Statistical significance is examined using Friedman test followed by pairwise Wilcoxon signed-rank testing with Holm-Bonferroni correction and compact letter display visualization. To promote reproducibility, we release an open-source Python package, kidney-hfm-eval, available at this https URL , that reproduces the evaluation pipelines. Results show moderate to strong performance on tasks driven by coarse meso-scale renal morphology, including diagnostic classification and detection of prominent structural alterations. In contrast, performance consistently declines for tasks requiring fine-grained microstructural discrimination, complex biological phenotypes, or slide-level prognostic inference, largely independent of stain type. Overall, current HFMs appear to encode predominantly static meso-scale representations and may have limited capacity to capture subtle renal pathology or prognosis-related signals. Our results highlight the need for kidney-specific, multi-stain, and multimodal foundation models to support clinically reliable decision-making in nephrology.
>
---
#### [replaced 055] Synergistic Bleeding Region and Point Detection in Laparoscopic Surgical Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.22174](https://arxiv.org/pdf/2503.22174)**

> **作者:** Jialun Pei; Zhangjun Zhou; Diandian Guo; Zhixi Li; Jing Qin; Bo Du; Pheng-Ann Heng
>
> **备注:** This work has been accepted by CVPR 2026
>
> **摘要:** Intraoperative bleeding in laparoscopic surgery causes rapid obscuration of the operative field to hinder the surgical process and increases the risk of postoperative complications. Intelligent detection of bleeding areas can quantify the blood loss to assist decision-making, while locating bleeding points helps surgeons quickly identify the source of bleeding and achieve hemostasis in time to improve surgical success rates. To fill the benchmark gap, we first construct a real-world laparoscopic surgical bleeding detection dataset, named SurgBlood, comprising 5,330 frames from 95 surgical video clips with bleeding region and point annotations. Accordingly, we develop a dual-task synergistic online detector called BlooDet, enabling simultaneous detection of bleeding regions and points in laparoscopic surgery. The baseline embraces a dual-branch bidirectional guid- ance design based on Segment Anything Model 2. The mask branch detects bleeding regions through adaptive edge and point prompt embeddings, while the point branch leverages mask memory to induce bleeding point memory modeling and captures point motion direction via inter-frame optical flow. By coupled bidirectional guidance, our framework explores spatial-temporal correlations while exploiting memory modeling to infer current bleeding status. Extensive experiments indicate that our method outperforms 13 counterparts in bleeding detection.
>
---
#### [replaced 056] DSeq-JEPA: Discriminative Sequential Joint-Embedding Predictive Architecture
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.17354](https://arxiv.org/pdf/2511.17354)**

> **作者:** Xiangteng He; Shunsuke Sakai; Shivam Chandhok; Sara Beery; Kun Yuan; Nicolas Padoy; Tatsuhito Hasegawa; Leonid Sigal
>
> **备注:** Project page: this https URL
>
> **摘要:** Recent advances in self-supervised visual representation learning have demonstrated the effectiveness of predictive latent-space objectives for learning transferable features. In particular, Image-based Joint-Embedding Predictive Architecture (I-JEPA) learns representations by predicting latent embeddings of masked target regions from visible context. However, it predicts target regions in parallel and all at once, lacking ability to order predictions meaningfully. Inspired by human visual perception, which attends selectively and progressively from primary to secondary cues, we propose DSeq-JEPA, a Discriminative Sequential Joint-Embedding Predictive Architecture that bridges latent predictive and autoregressive self-supervised learning. Specifically, DSeq-JEPA integrates a discriminatively ordered sequential process with JEPA-style learning objective. This is achieved by (i) identifying primary discriminative regions using an attention-derived saliency map that serves as a proxy for visual importance, and (ii) predicting subsequent regions in discriminative order, inducing a curriculum-like semantic progression from primary to secondary cues in pre-training. Extensive experiments across tasks -- image classification (ImageNet), fine-grained visual categorization (iNaturalist21, CUB, Stanford Cars), detection/segmentation (MS-COCO, ADE20K), and low-level reasoning (CLEVR) -- show that DSeq-JEPA consistently learns more discriminative and generalizable representations compared to I-JEPA variants. Project page: this https URL.
>
---
#### [replaced 057] Aion: Towards Hierarchical 4D Scene Graphs with Temporal Flow Dynamics
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自主导航任务，解决动态环境中时空表示问题。提出Aion框架，将时间流动态嵌入层次化3D场景图，提升导航预测与规划效果。**

- **链接: [https://arxiv.org/pdf/2512.11903](https://arxiv.org/pdf/2512.11903)**

> **作者:** Iacopo Catalano; Eduardo Montijano; Javier Civera; Julio A. Placed; Jorge Pena-Queralta
>
> **备注:** Accepted at ICRA 2026, 8 pages
>
> **摘要:** Autonomous navigation in dynamic environments requires spatial representations that capture both semantic structure and temporal evolution. 3D Scene Graphs (3DSGs) provide hierarchical multi-resolution abstractions that encode geometry and semantics, but existing extensions toward dynamics largely focus on individual objects or agents. In parallel, Maps of Dynamics (MoDs) model typical motion patterns and temporal regularities, yet are usually tied to grid-based discretizations that lack semantic awareness and do not scale well to large environments. In this paper we introduce Aion, a framework that embeds temporal flow dynamics directly within a hierarchical 3DSG, effectively incorporating the temporal dimension. Aion employs a graph-based sparse MoD representation to capture motion flows over arbitrary time intervals and attaches them to navigational nodes in the scene graph, yielding more interpretable and scalable predictions that improve planning and interaction in complex dynamic environments. We provide the code at this https URL
>
---
#### [replaced 058] Synergizing Deep Learning and Biological Heuristics for Extreme Long-Tail White Blood Cell Classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.16249](https://arxiv.org/pdf/2603.16249)**

> **作者:** Duc T. Nguyen; Hoang-Long Nguyen; Huy-Hieu Pham
>
> **备注:** Accepted at IEEE ISBI 2026
>
> **摘要:** Automated white blood cell (WBC) classification is essential for leukemia screening but remains challenged by extreme class imbalance, long-tail distributions, and domain shift, leading deep models to overfit dominant classes and fail on rare subtypes. We propose a hybrid framework for rare-class generalization that integrates a generative Pix2Pix-based restoration module for artifact removal, a Swin Transformer ensemble with MedSigLIP contrastive embeddings for robust representation learning, and a biologically-inspired refinement step using geometric spikiness and Mahalanobis-based morphological constraints to recover out-of-distribution predictions. Evaluated on the WBCBench 2026 challenge, our method achieves a Macro-F1 of 0.77139 on the private leaderboard, demonstrating strong performance under severe imbalance and highlighting the value of incorporating biological priors into deep learning for hematological image analysis.
>
---
#### [replaced 059] Mamba2D: A Natively Multi-Dimensional State-Space Model for Vision Tasks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.16146](https://arxiv.org/pdf/2412.16146)**

> **作者:** Enis Baty; Alejandro Hernández Díaz; Rebecca Davidson; Chris Bridges; Simon Hadfield
>
> **摘要:** State-Space Models (SSMs) have emerged as an efficient alternative to transformers, yet existing visual SSMs retain deeply ingrained biases from their origins in natural language processing. In this paper, we address these limitations by introducing M2D-SSM, a ground-up re-derivation of selective state-space techniques for multidimensional data. Unlike prior works that apply 1D SSMs directly to images through arbitrary rasterised scanning, our M2D-SSM employs a single 2D scan that factors in both spatial dimensions natively. On ImageNet-1K classification, M2D-T achieves 84.0% top-1 accuracy with only 27M parameters, surpassing all prior SSM-based vision models at that size. M2D-S further achieves 85.3%, establishing state-of-the-art results among SSM-based architectures. Across downstream tasks, Mamba2D achieves 52.2 box AP on MS-COCO object detection (3$\times$ schedule) and 51.7 mIoU on ADE20K segmentation, demonstrating strong generalisation and efficiency at scale. Source code is available at this https URL.
>
---
#### [replaced 060] Semi-supervised Shelter Mapping for WASH Accessibility Assessment in Rohingya Refugee Camps
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.07231](https://arxiv.org/pdf/2511.07231)**

> **作者:** Kyeongjin Ahn; YongHun Suh; Sungwon Han; Jeasurk Yang; Hannes Taubenböck; Meeyoung Cha
>
> **备注:** 22 pages, 13 figures, 2 tables
>
> **摘要:** Lack of access to Water, Sanitation, and Hygiene (WASH) services is a major public health concern in refugee camps, where extreme crowding accelerates the spread of communicable diseases. The Rohingya settlements in Cox's Bazar, Bangladesh, exemplify these conditions, with large populations living under severe spatial constraints. We develop a semi-supervised segmentation framework using the Segment Anything Model (SAM) to map shelters from multi-temporal sub-meter remote sensing imagery (2017-2025), improving detection in complex camp environments by 4.9% in F1-score over strong baselines. The detected shelter maps show that shelter expansion stabilized after 2020, whereas continued population growth reduced per capita living space by approximately 14% between 2020 and 2025. WASH accessibility, measured with an enhanced network-based two-step floating catchment area (2SFCA) method, declined from 2022 to 2025, increasing facility loads and exceeding global benchmarks. Gender-disaggregated scenarios that incorporate safety penalty further reveal pronounced inequities, with female accessibility approximately 27% lower than male. Together, these results demonstrate that remote sensing-driven AI diagnostics can generate equity-focused evidence to prioritize WASH investments and mitigate health risks in protracted displacement settings.
>
---
#### [replaced 061] VIEW2SPACE: Studying Multi-View Visual Reasoning from Sparse Observations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.16506](https://arxiv.org/pdf/2603.16506)**

> **作者:** Fucai Ke; Zhixi Cai; Boying Li; Long Chen; Beibei Lin; Weiqing Wang; Pari Delir Haghighi; Gholamreza Haffari; Hamid Rezatofighi
>
> **摘要:** Multi-view visual reasoning is essential for intelligent systems that must understand complex environments from sparse and discrete viewpoints, yet existing research has largely focused on single-image or temporally dense video settings. In real-world scenarios, reasoning across views requires integrating partial observations without explicit guidance, while collecting large-scale multi-view data with accurate geometric and semantic annotations remains challenging. To address this gap, we leverage physically grounded simulation to construct diverse, high-fidelity 3D scenes with precise per-view metadata, enabling scalable data generation that remains transferable to real-world settings. Based on this engine, we introduce VIEW2SPACE, a multi-dimensional benchmark for sparse multi-view reasoning, together with a scalable, disjoint training split supporting millions of grounded question-answer pairs. Using this benchmark, a comprehensive evaluation of state-of-the-art vision-language and spatial models reveals that multi-view reasoning remains largely unsolved, with most models performing only marginally above random guessing. We further investigate whether training can bridge this gap. Our proposed Grounded Chain-of-Thought with Visual Evidence substantially improves performance under moderate difficulty, and generalizes to real-world data, outperforming existing approaches in cross-dataset evaluation. We further conduct difficulty-aware scaling analyses across model size, data scale, reasoning depth, and visibility constraints, indicating that while geometric perception can benefit from scaling under sufficient visibility, deep compositional reasoning across sparse views remains a fundamental challenge.
>
---
#### [replaced 062] Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.15026](https://arxiv.org/pdf/2603.15026)**

> **作者:** Omer Ben Hayun; Roy Betser; Meir Yossef Levi; Levi Kassel; Guy Gilboa
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Following major advances in text and image generation, the video domain has surged, producing highly realistic and controllable sequences. Along with this progress, these models also raise serious concerns about misinformation, making reliable detection of synthetic videos increasingly crucial. Image-based detectors are fundamentally limited because they operate per frame and ignore temporal dynamics, while supervised video detectors generalize poorly to unseen generators, a critical drawback given the rapid emergence of new models. These challenges motivate zero-shot approaches, which avoid synthetic data and instead score content against real-data statistics, enabling training-free, model-agnostic detection. We introduce STALL, a simple, training-free, theoretically justified detector that provides likelihood-based scoring for videos, jointly modeling spatial and temporal evidence within a probabilistic framework. We evaluate STALL on two public benchmarks and introduce ComGenVid, a new benchmark with state-of-the-art generative models. STALL consistently outperforms prior image- and video-based baselines. Code and data are available at this https URL.
>
---
#### [replaced 063] VisBrowse-Bench: Benchmarking Visual-Native Search for Multimodal Browsing Agents
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.16289](https://arxiv.org/pdf/2603.16289)**

> **作者:** Zhengbo Zhang; Jinbo Su; Zhaowen Zhou; Changtao Miao; Yuhan Hong; Qimeng Wu; Yumeng Liu; Feier Wu; Yihe Tian; Yuhao Liang; Zitong Shan; Wanke Xia; Yi-Fan Zhang; Bo Zhang; Zhe Li; Shiming Xiang; Ying Yan
>
> **摘要:** The rapid advancement of Multimodal Large Language Models (MLLMs) has enabled browsing agents to acquire and reason over multimodal information in the real world. But existing benchmarks suffer from two limitations: insufficient evaluation of visual reasoning ability and the neglect of native visual information of web pages in the reasoning chains. To address these challenges, we introduce a new benchmark for visual-native search, VisBrowse-Bench. It contains 169 VQA instances covering multiple domains and evaluates the models' visual reasoning capabilities during the search process through multimodal evidence cross-validation via text-image retrieval and joint reasoning. These data were constructed by human experts using a multi-stage pipeline and underwent rigorous manual verification. We additionally propose an agent workflow that can effectively drive the browsing agent to actively collect and reason over visual information during the search process. We comprehensively evaluated both open-source and closed-source models in this workflow. Experimental results show that even the best-performing model, Claude-4.6-Opus only achieves an accuracy of 47.6%, while the proprietary Deep Research model, o3-deep-research only achieves an accuracy of 41.1%. The code and data can be accessed at: this https URL
>
---
#### [replaced 064] CoT-PL: Chain-of-Thought Pseudo-Labeling for Open-Vocabulary Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.14792](https://arxiv.org/pdf/2510.14792)**

> **作者:** Hojun Choi; Youngsun Lim; Jaeyo Shin; Hyunjung Shim
>
> **备注:** 38 pages, 15 Figures, 12 Tables
>
> **摘要:** Open-vocabulary object detection (OVD) aims to recognize and localize object categories beyond the training set. Recent approaches leverage vision-language models to generate pseudo-labels using image-text alignment, allowing detectors to generalize to unseen classes without explicit supervision. However, these methods depend heavily on single-step image-text matching, neglecting the intermediate reasoning steps crucial for interpreting semantically complex visual contexts, such as crowding or occlusion. In this paper, we introduce CoT-PL, a framework that incorporates visual chain-of-thought reasoning into the pseudo-labeling process for OVD. It decomposes complex scene understanding into three interpretable steps-object localization, category recognition, and background grounding-where these intermediate reasoning states serve as rich supervision sources. Extensive experiments on standard OVD evaluation protocols demonstrate that CoT-PL achieves state-of-the-art performance with superior pseudo-labeling efficiency, outperforming the strong baseline by 9.4 AP50 for novel classes on OV-COCO and improving box and mask APr by 3.2 and 2.2, respectively, on OV-LVIS. Code and models are available at this https URL.
>
---
#### [replaced 065] Mechanistic Interpretability of Diffusion Models: Circuit-Level Analysis and Causal Validation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.17237](https://arxiv.org/pdf/2506.17237)**

> **作者:** Dip Roy
>
> **摘要:** We present a quantitative circuit-level analysis of diffusion models, establishing computational pathways and mechanistic principles underlying image generation processes. Through systematic intervention experiments across 2,000 synthetic and 2,000 CelebA facial images, we discover fundamental algorithmic differences in how diffusion architectures process synthetic versus naturalistic data distributions. Our investigation reveals that real-world face processing requires circuits with measurably higher computational complexity (complexity ratio = 1.084 plus/minus 0.008, p < 0.001), exhibiting distinct attention specialization patterns with entropy divergence ranging from 0.015 to 0.166 across denoising timesteps. We identify eight functionally distinct attention mechanisms showing specialized computational roles: edge detection (entropy = 3.18 plus/minus 0.12), texture analysis (entropy = 4.16 plus/minus 0.08), and semantic understanding (entropy = 2.67 plus/minus 0.15). Intervention analysis demonstrates critical computational bottlenecks where targeted ablations produce 25.6% to 128.3% performance degradation, providing causal evidence for identified circuit functions. These findings establish quantitative foundations for algorithmic understanding and control of generative model behavior through mechanistic intervention strategies.
>
---
#### [replaced 066] SO-Bench: A Structural Output Evaluation of Multimodal LLMs
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文提出SO-Bench基准，用于评估多模态大模型的结构化输出能力，解决其在视觉输入下符合预定义数据模式的问题。**

- **链接: [https://arxiv.org/pdf/2511.21750](https://arxiv.org/pdf/2511.21750)**

> **作者:** Di Feng; Kaixin Ma; Feng Nan; Haofeng Chen; Bohan Zhai; David Griffiths; Mingfei Gao; Zhe Gan; Eshan Verma; Yinfei Yang; Zhifeng Chen; Afshin Dehghan
>
> **备注:** v3 preprint. Added the link to the public benchmark
>
> **摘要:** Multimodal large language models (MLLMs) are increasingly deployed in real-world, agentic settings where outputs must not only be correct, but also conform to predefined data schemas. Despite recent progress in structured generation in textual domain, there is still no benchmark that systematically evaluates schema-grounded information extraction and reasoning over visual inputs. In this work, we conduct a comprehensive study of visual structural output capabilities for MLLMs with our carefully designed SO-Bench benchmark. Covering four visual domains, including UI screens, natural images, documents, and charts, SO-Bench is built from over 6.5K diverse JSON schemas and 1.8K curated image-schema pairs with human-verified quality. Benchmarking experiments on open-sourced and frontier proprietary models reveal persistent gaps in predicting accurate, schema compliant outputs, highlighting the need for better multimodal structured reasoning. Beyond benchmarking, we further conduct training experiments to largely improve the model's structured output capability. We make the benchmark and evaluation publicly available at this https URL
>
---
#### [replaced 067] Efficient Diffusion as Low Light Enhancer
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2410.12346](https://arxiv.org/pdf/2410.12346)**

> **作者:** Guanzhou Lan; Qianli Ma; Yuqi Yang; Zhigang Wang; Dong Wang; Xuelong Li; Bin Zhao
>
> **备注:** CVPR 2025 Camera Ready
>
> **摘要:** The computational burden of the iterative sampling process remains a major challenge in diffusion-based Low-Light Image Enhancement (LLIE). Current acceleration methods, whether training-based or training-free, often lead to significant performance degradation, highlighting the trade-off between performance and efficiency. In this paper, we identify two primary factors contributing to performance degradation: fitting errors and the inference gap. Our key insight is that fitting errors can be mitigated by linearly extrapolating the incorrect score functions, while the inference gap can be reduced by shifting the Gaussian flow to a reflectance-aware residual space. Based on the above insights, we design Reflectance-Aware Trajectory Refinement (RATR) module, a simple yet effective module to refine the teacher trajectory using the reflectance component of images. Following this, we introduce \textbf{Re}flectance-aware \textbf{D}iffusion with \textbf{Di}stilled \textbf{T}rajectory (\textbf{ReDDiT}), an efficient and flexible distillation framework tailored for LLIE. Our framework achieves comparable performance to previous diffusion-based methods with redundant steps in just 2 steps while establishing new state-of-the-art (SOTA) results with 8 or 4 steps. Comprehensive experimental evaluations on 10 benchmark datasets validate the effectiveness of our method, consistently outperforming existing SOTA methods.
>
---
#### [replaced 068] Generative Refocusing: Flexible Defocus Control from a Single Image
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.16923](https://arxiv.org/pdf/2512.16923)**

> **作者:** Chun-Wei Tuan Mu; Cheng-De Fan; Jia-Bin Huang; Yu-Lun Liu
>
> **备注:** Project website: this https URL
>
> **摘要:** Depth-of-field control is essential in photography, but achieving perfect focus often requires multiple attempts or specialized equipment. Single-image refocusing is still difficult. It involves recovering sharp content and creating realistic bokeh. Current methods have significant drawbacks. They require all-in-focus inputs, rely on synthetic data from simulators, and have limited control over the aperture. We introduce Generative Refocusing, a two-step process that uses DeblurNet to recover all-in-focus images from diverse inputs and BokehNet to create controllable bokeh. This method combines synthetic and real bokeh images to achieve precise control while preserving authentic optical characteristics. Our experiments show we achieve top performance in defocus deblurring, bokeh synthesis, and refocusing benchmarks. Additionally, our Generative Refocusing allows custom aperture shapes. Project page: this https URL
>
---
#### [replaced 069] Vector sketch animation generation with differentiable motion trajectories
- **分类: cs.GR; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.25857](https://arxiv.org/pdf/2509.25857)**

> **作者:** Xinding Zhu; Xinye Yang; Shuyang Zheng; Zhexin Zhang; Fei Gao; Jing Huang; Jiazhou Chen
>
> **备注:** 14 pages, 12 figures
>
> **摘要:** Sketching is a direct and inexpensive means of visual expression. Though image-based sketching has been well studied, video-based sketch animation generation is still very challenging due to the temporal coherence requirement. In this paper, we propose a novel end-to-end automatic generation approach for vector sketch animation. To solve the flickering issue, we introduce a Differentiable Motion Trajectory (DMT) representation that describes the frame-wise movement of stroke control points using differentiable polynomial-based trajectories. DMT enables global semantic gradient propagation across multiple frames, significantly improving the semantic consistency and temporal coherence, and producing high-framerate output. DMT employs a Bernstein basis to balance the sensitivity of polynomial parameters, thus achieving more stable optimization. Instead of implicit fields, we introduce sparse track points for explicit spatial modeling, which improves efficiency and supports long-duration video processing. Evaluations on DAVIS and LVOS datasets demonstrate the superiority of our approach over SOTA methods. Cross-domain validation on 3D models and text-to-video data confirms the robustness and compatibility of our approach.
>
---
#### [replaced 070] WPT: World-to-Policy Transfer via Online World Model Distillation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20095](https://arxiv.org/pdf/2511.20095)**

> **作者:** Guangfeng Jiang; Yueru Luo; Jun Liu; Yi Huang; Yiyao Zhu; Zhan Qu; Dave Zhenyu Chen; Bingbing Liu; Xu Yan
>
> **备注:** CVPR2026 Accepted
>
> **摘要:** Recent years have witnessed remarkable progress in world models, which primarily aim to capture the spatio-temporal correlations between an agent's actions and the evolving environment. However, existing approaches often suffer from tight runtime coupling or depend on offline reward signals, resulting in substantial inference overhead or hindering end-to-end optimization. To overcome these limitations, we introduce WPT, a World-to-Policy Transfer training paradigm that enables online distillation under the guidance of an end-to-end world model. Specifically, we develop a trainable reward model that infuses world knowledge into a teacher policy by aligning candidate trajectories with the future dynamics predicted by the world model. Subsequently, we propose policy distillation and world reward distillation to transfer the teacher's reasoning ability into a lightweight student policy, enhancing planning performance while preserving real-time deployability. Extensive experiments on both open-loop and closed-loop benchmarks show that our WPT achieves state-of-the-art performance with a simple policy architecture: it attains a 0.11 collision rate (open-loop) and achieves a 79.23 driving score (closed-loop) surpassing both world-model-based and imitation-learning methods in accuracy and safety. Moreover, the student sustains up to 4.9x faster inference, while retaining most of the gains.
>
---
#### [replaced 071] TechImage-Bench: Rubric-Based Evaluation for Technical Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.12220](https://arxiv.org/pdf/2512.12220)**

> **作者:** Minheng Ni; Zhengyuan Yang; Yaowen Zhang; Linjie Li; Chung-Ching Lin; Kevin Lin; Zhendong Wang; Xiaofei Wang; Shujie Liu; Lei Zhang; Wangmeng Zuo; Lijuan Wang
>
> **摘要:** We study technical image generation, where a model must synthesize information-dense, scientifically precise illustrations from detailed descriptions rather than merely produce visually plausible pictures. To quantify the progress, we introduce TechImage-Bench, a rubric-based benchmark that targets biology schematics, engineering/patent drawings, and general technical illustrations. For 654 figures collected from real textbooks and technical reports, we construct detailed image instructions and a hierarchy of rubrics that decompose correctness into 6,076 criteria and 44,131 binary checks. Rubrics are derived from surrounding text and reference figures using large multimodal models, and are evaluated by an automated LMM-based judge with a principled penalty scheme that aggregates sub-question outcomes into interpretable criterion scores. We benchmark several representative text-to-image models on TechImage-Bench and find that, despite strong open-domain performance, the best base model reaches only 0.801 rubric accuracy and 0.576 criterion score overall, revealing substantial gaps in fine-grained scientific fidelity. Finally, we show that the same rubrics provide actionable supervision: feeding failed checks back into an editing model for iterative refinement boosts a strong generator from 0.660 to 0.865 in rubric accuracy and from 0.382 to 0.697 in criterion score. TechImage-Bench thus offers both a rigorous diagnostic for technical image generation and a scalable signal for improving specification-faithful scientific illustrations.
>
---
#### [replaced 072] Look Before You Fuse: 2D-Guided Cross-Modal Alignment for Robust 3D Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.16861](https://arxiv.org/pdf/2507.16861)**

> **作者:** Xiang Li; Zhangchi Hu; Xiao Xu; Bin Kong
>
> **备注:** accepted to cvpr 2026
>
> **摘要:** Integrating LiDAR and camera inputs into a unified Bird's-Eye-View (BEV) representation is crucial for enhancing 3D perception capabilities of autonomous vehicles. However, existing methods suffer from spatial misalignment between LiDAR and camera features, which causes inaccurate depth supervision in camera branch and erroneous fusion during cross-modal feature aggregation. The root cause of this misalignment lies in projection errors, stemming from calibration inaccuracies and rolling shutter this http URL key insight of this work is that locations of these projection errors are not random but highly predictable, as they are concentrated at object-background boundaries which 2D detectors can reliably identify. Based on this, our main motivation is to utilize 2D object priors to pre-align cross-modal features before fusion. To address local misalignment, we propose Prior Guided Depth Calibration (PGDC), which leverages 2D priors to alleviate misalignment and preserve correct cross-modal feature pairs. To resolve global misalignment, we introduce Discontinuity Aware Geometric Fusion (DAGF) to suppress residual noise from PGDC and explicitly enhance sharp depth transitions at object-background boundaries, yielding a structurally aware representation. To effectively utilize these aligned representations, we incorporate Structural Guidance Depth Modulator (SGDM), using a gated attention mechanism to efficiently fuse aligned depth and image features. Our method achieves SOTA performance on nuScenes validation dataset, with its mAP and NDS reaching 71.5% and 73.6% respectively. Additionally, on the Argoverse 2 validation set, we achieve a competitive mAP of 41.7%.
>
---
#### [replaced 073] Bodhi VLM: Privacy-Alignment Modeling for Hierarchical Visual Representations in Vision Backbones and VLM Encoders via Bottom-Up and Top-Down Feature Search
- **分类: cs.CV; cs.CR**

- **链接: [https://arxiv.org/pdf/2603.13728](https://arxiv.org/pdf/2603.13728)**

> **作者:** Bo Ma; Wei Qi Yan; Jinsong Wu
>
> **摘要:** Learning systems that preserve privacy often inject noise into hierarchical visual representations; a central challenge is to \emph{model} how such perturbations align with a declared privacy budget in a way that is interpretable and applicable across vision backbones and vision--language models (VLMs). We propose \emph{Bodhi VLM}, a \emph{privacy-alignment modeling} framework for \emph{hierarchical neural representations}: it (1) links sensitive concepts to layer-wise grouping via NCP and MDAV-based clustering; (2) locates sensitive feature regions using bottom-up (BUA) and top-down (TDA) strategies over multi-scale representations (e.g., feature pyramids or vision-encoder layers); and (3) uses an Expectation-Maximization Privacy Assessment (EMPA) module to produce an interpretable \emph{budget-alignment signal} by comparing the fitted sensitive-feature distribution to an evaluator-specified reference (e.g., Laplace or Gaussian with scale $c/\epsilon$). The output is reference-relative and is \emph{not} a formal differential-privacy estimator. We formalize BUA/TDA over hierarchical feature structures and validate the framework on object detectors (YOLO, PPDPTS, DETR) and on the \emph{visual encoders} of VLMs (CLIP, LLaVA, BLIP). BUA and TDA yield comparable deviation trends; EMPA provides a stable alignment signal under the reported setups. We compare with generic discrepancy baselines (Chi-square, K-L, MMD) and with task-relevant baselines (MomentReg, NoiseMLE, Wass-1). Results are reported as mean$\pm$std over multiple seeds with confidence intervals in the supplementary materials. This work contributes a learnable, interpretable modeling perspective for privacy-aligned hierarchical representations rather than a post hoc audit only. Source code: \href{this https URL}{Bodhi-VLM GitHub repository}
>
---
#### [replaced 074] Frequency Autoregressive Image Generation with Continuous Tokens
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2503.05305](https://arxiv.org/pdf/2503.05305)**

> **作者:** Hu Yu; Hao Luo; Hangjie Yuan; Yu Rong; Jie Huang; Feng Zhao
>
> **摘要:** Autoregressive (AR) models for image generation typically adopt a two-stage paradigm of vector quantization and raster-scan ``next-token prediction", inspired by its great success in language modeling. However, due to the huge modality gap, image autoregressive models may require a systematic reevaluation from two perspectives: tokenizer format and regression direction. In this paper, we introduce the frequency progressive autoregressive (\textbf{FAR}) paradigm and instantiate FAR with the continuous tokenizer. Specifically, we identify spectral dependency as the desirable regression direction for FAR, wherein higher-frequency components build upon the lower one to progressively construct a complete image. This design seamlessly fits the causality requirement for autoregressive models and preserves the unique spatial locality of image data. Besides, we delve into the integration of FAR and the continuous tokenizer, introducing a series of techniques to address optimization challenges and improve the efficiency of training and inference processes. We demonstrate the efficacy of FAR through comprehensive experiments on the ImageNet dataset and verify its potential on text-to-image generation.
>
---
#### [replaced 075] Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.12789](https://arxiv.org/pdf/2603.12789)**

> **作者:** Sangmin Kim; Minhyuk Hwang; Geonho Cha; Dongyoon Wee; Jaesik Park
>
> **备注:** Project page: this https URL
>
> **摘要:** Recent advances in 3D foundation models have led to growing interest in reconstructing humans and their surrounding environments. However, most existing approaches focus on monocular inputs, and extending them to multi-view settings requires additional overhead modules or preprocessed data. To this end, we present CHROMM, a unified framework that jointly estimates cameras, scene point clouds, and human meshes from multi-person multi-view videos without relying on external modules or preprocessing. We integrate strong geometric and human priors from Pi3X and Multi-HMR into a single trainable neural network architecture, and introduce a scale adjustment module to solve the scale discrepancy between humans and the scene. We also introduce a multi-view fusion strategy to aggregate per-view estimates into a single representation at test-time. Finally, we propose a geometry-based multi-person association method, which is more robust than appearance-based approaches. Experiments on EMDB, RICH, EgoHumans, and EgoExo4D show that CHROMM achieves competitive performance in global human motion and multi-view pose estimation while running over 8x faster than prior optimization-based multi-view approaches. Project page: this https URL.
>
---
#### [replaced 076] Neighbor GRPO: Contrastive ODE Policy Optimization Aligns Flow Models
- **分类: cs.CV; cs.LG; eess.IV**

- **链接: [https://arxiv.org/pdf/2511.16955](https://arxiv.org/pdf/2511.16955)**

> **作者:** Dailan He; Guanlin Feng; Xingtong Ge; Yazhe Niu; Yi Zhang; Bingqi Ma; Guanglu Song; Yu Liu; Hongsheng Li
>
> **备注:** CVPR 2026
>
> **摘要:** Group Relative Policy Optimization (GRPO) has shown promise in aligning image and video generative models with human preferences. However, applying it to modern flow matching models is challenging because of its deterministic sampling paradigm. Current methods address this issue by converting Ordinary Differential Equations (ODEs) to Stochastic Differential Equations (SDEs), which introduce stochasticity. However, this SDE-based GRPO suffers from issues of inefficient credit assignment and incompatibility with high-order solvers for fewer-step sampling. In this paper, we first reinterpret existing SDE-based GRPO methods from a distance optimization perspective, revealing their underlying mechanism as a form of contrastive learning. Based on this insight, we propose Neighbor GRPO, a novel alignment algorithm that completely bypasses the need for SDEs. Neighbor GRPO generates a diverse set of candidate trajectories by perturbing the initial noise conditions of the ODE and optimizes the model using a softmax distance-based surrogate leaping policy. We establish a theoretical connection between this distance-based objective and policy gradient optimization, rigorously integrating our approach into the GRPO framework. Our method fully preserves the advantages of deterministic ODE sampling, including efficiency and compatibility with high-order solvers. We further introduce symmetric anchor sampling for computational efficiency and group-wise quasi-norm reweighting to address reward flattening. Extensive experiments demonstrate that Neighbor GRPO significantly outperforms SDE-based counterparts in terms of training cost, convergence speed, and generation quality.
>
---
#### [replaced 077] Bundle Adjustment in the Eager Mode
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉SLAM任务，解决传统BA库与深度学习框架不兼容的问题，提出一种与PyTorch无缝集成的高效GPU版BA方法。**

- **链接: [https://arxiv.org/pdf/2409.12190](https://arxiv.org/pdf/2409.12190)**

> **作者:** Zitong Zhan; Huan Xu; Zihang Fang; Xinpeng Wei; Yaoyu Hu; Chen Wang
>
> **摘要:** Bundle adjustment (BA) is a critical technique in various robotic applications such as simultaneous localization and mapping (SLAM), augmented reality (AR), and photogrammetry. BA optimizes parameters such as camera poses and 3D landmarks to align them with observations. With the growing importance of deep learning in perception systems, there is an increasing need to integrate BA with deep learning frameworks for enhanced reliability and performance. However, widely-used C++-based BA libraries, such as GTSAM, g$^2$o, and Ceres Solver, lack native integration with modern deep learning libraries like PyTorch. This limitation affects their flexibility, ease of debugging, and overall implementation efficiency. To address this gap, we introduce an eager-mode BA library seamlessly integrated with PyTorch with high efficiency. Our approach includes a sparsity-aware auto-differentiation design and GPU-accelerated sparse operations designed for 2nd-order optimization. Our eager-mode BA on GPU demonstrates substantial runtime efficiency, achieving an average speedup of 18.5$\times$, 22$\times$, and 23$\times$ across all benchmarks compared to GTSAM, g$^2$o, and Ceres, respectively.
>
---
#### [replaced 078] Digital FAST: An AI-Driven Multimodal Framework for Rapid and Early Stroke Screening
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.11896](https://arxiv.org/pdf/2601.11896)**

> **作者:** Ngoc-Khai Hoang; Thi-Nhu-Mai Nguyen; Huy-Hieu Pham
>
> **摘要:** Early identification of stroke symptoms is essential for enabling timely intervention and improving patient outcomes, particularly in prehospital settings. This study presents a fast, non-invasive multimodal deep learning framework for automatic binary stroke screening based on data collected during the F.A.S.T. assessment. The proposed approach integrates complementary information from facial expressions, speech signals, and upper-body movements to enhance diagnostic robustness. Facial dynamics are represented using landmark based features and modeled with a Transformer architecture to capture temporal dependencies. Speech signals are converted into mel spectrograms and processed using an Audio Spectrogram Transformer, while upper-body pose sequences are analyzed with an MLP-Mixer network to model spatiotemporal motion patterns. The extracted modality specific representations are combined through an attention-based fusion mechanism to effectively learn cross modal interactions. Experiments conducted on a self-collected dataset of 222 videos from 37 subjects demonstrate that the proposed multimodal model consistently outperforms unimodal baselines, achieving 95.83% accuracy and a 96.00% F1-score. The model attains a strong balance between sensitivity and specificity and successfully detects all stroke cases in the test set. These results highlight the potential of multimodal learning and transfer learning for early stroke screening, while emphasizing the need for larger, clinically representative datasets to support reliable real-world deployment.
>
---
#### [replaced 079] Rolling Sink: Bridging Limited-Horizon Training and Open-Ended Testing in Autoregressive Video Diffusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.07775](https://arxiv.org/pdf/2602.07775)**

> **作者:** Haodong Li; Shaoteng Liu; Zhe Lin; Manmohan Chandraker
>
> **备注:** v4: Fix some typos. Figures were compressed to 150 dpi to comply with arXiv's submission size limit. Project page: this https URL
>
> **摘要:** Recently, autoregressive (AR) video diffusion models have achieved remarkable performance. However, due to their limited training durations, a train-test gap emerges when testing at longer horizons, leading to rapid visual degradations. Following Self Forcing, which studies the train-test gap within the training duration, this work studies the train-test gap beyond the training duration, i.e., the gap between the limited horizons during training and open-ended horizons during testing. Since open-ended testing can extend beyond any finite training window, and long-video training is computationally expensive, we pursue a training-free solution to bridge this gap. To explore a training-free solution, we conduct a systematic analysis of AR cache maintenance. These insights lead to Rolling Sink. Built on Self Forcing (trained on only 5s clips), Rolling Sink effectively scales the AR video synthesis to ultra-long durations (e.g., 5-30 minutes at 16 FPS) at test time, with consistent subjects, stable colors, coherent structures, and smooth motions. As demonstrated by extensive experiments, Rolling Sink achieves superior long-horizon visual fidelity and temporal consistency compared to SOTA baselines. Project page: this https URL
>
---
#### [replaced 080] HyperMotionX: The Dataset and Benchmark with DiT-Based Pose-Guided Human Image Animation of Complex Motions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.22977](https://arxiv.org/pdf/2505.22977)**

> **作者:** Shuolin Xu; Siming Zheng; Ziyi Wang; HC Yu; Jinwei Chen; Huaqi Zhang; Daquan Zhou; Tong-Yee Lee; Bo Li; Peng-Tao Jiang
>
> **备注:** 17 pages, 7 figures
>
> **摘要:** Recent advances in diffusion models have significantly improved conditional video generation, particularly in the pose-guided human image animation task. Although existing methods are capable of generating high-fidelity and time-consistent animation sequences in regular motions and static scenes. However there are still obvious limitations when facing complex human body motions that contain highly dynamic, non-standard motions, and the lack of a high-quality benchmark for evaluation of complex human motion animations. To address this challenge, we propose a concise yet powerful DiT-based human animation generation baseline and design spatial low-frequency enhanced RoPE, a novel module that selectively enhances low-frequency spatial feature modeling by introducing learnable frequency scaling. Furthermore, we introduce the Open-HyperMotionX Dataset and HyperMotionX Bench, which provide high-quality human pose annotations and curated video clips for evaluating and improving pose-guided human image animation models under complex human motion conditions. Our method significantly improves structural stability and appearance consistency in highly dynamic human motion sequences. Extensive experiments demonstrate the effectiveness of our dataset and proposed approach in advancing the generation quality of complex human motion image animations. The codes, model weights, and dataset have been made publicly available at this https URL
>
---
#### [replaced 081] Towards Clinical Practice in CT-Based Pulmonary Disease Screening: An Efficient and Reliable Framework
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2412.01525](https://arxiv.org/pdf/2412.01525)**

> **作者:** Qian Shao; Bang Du; Yixuan Wu; Zepeng Li; Qiyuan Chen; Qianqian Tang; Jian Wu; Jintai Chen; Hongxia Xu
>
> **摘要:** Deep learning models for pulmonary disease screening from Computed Tomography (CT) scans promise to alleviate the immense workload on radiologists. Still, their high computational cost, stemming from processing entire 3D volumes, remains a major barrier to widespread clinical adoption. Current sub-sampling techniques often compromise diagnostic integrity by introducing artifacts or discarding critical information. To overcome these limitations, we propose an Efficient and Reliable Framework (ERF) that fundamentally improves the practicality of automated CT analysis. Our framework introduces two core innovations: (1) A Cluster-based Sub-Sampling (CSS) method that efficiently selects a compact yet comprehensive subset of CT slices by optimizing for both representativeness and diversity. By integrating an efficient k-nearest neighbor search with an iterative refinement process, CSS bypasses the computational bottlenecks of previous methods while preserving vital diagnostic features. (2) An Ambiguity-aware Uncertainty Quantification (AUQ) mechanism, which enhances reliability by specifically targeting data ambiguity arising from subtle lesions and artifacts. Unlike standard uncertainty measures, AUQ leverages the predictive discrepancy between auxiliary classifiers to construct a specialized ambiguity score. By maximizing this discrepancy during training, the system effectively flags ambiguous samples where the model lacks confidence due to visual noise or intricate pathologies. Validated on two public datasets with 2,654 CT volumes across diagnostic tasks for 3 pulmonary diseases, ERF achieves diagnostic performance comparable to the full-volume analysis (over 90% accuracy and recall) while reducing processing time by more than 60%. This work represents a significant step towards deploying fast, accurate, and trustworthy AI-powered screening tools in time-sensitive clinical settings.
>
---
#### [replaced 082] LMOD+: A Comprehensive Multimodal Dataset and Benchmark for Developing and Evaluating Multimodal Large Language Models in Ophthalmology
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.25620](https://arxiv.org/pdf/2509.25620)**

> **作者:** Zhenyue Qin; Yang Liu; Yu Yin; Jinyu Ding; Haoran Zhang; Anran Li; Dylan Campbell; Xuansheng Wu; Ke Zou; Tiarnan D. L. Keenan; Emily Y. Chew; Zhiyong Lu; Yih Chung Tham; Ninghao Liu; Xiuzhen Zhang; Qingyu Chen
>
> **备注:** ACM Transactions on Computing for Healthcare
>
> **摘要:** Vision-threatening eye diseases pose a major global health burden, with timely diagnosis limited by workforce shortages and restricted access to specialized care. While multimodal large language models (MLLMs) show promise for medical image interpretation, advancing MLLMs for ophthalmology is hindered by the lack of comprehensive benchmark datasets suitable for evaluating generative models. We present a large-scale multimodal ophthalmology benchmark comprising 32,633 instances with multi-granular annotations across 12 common ophthalmic conditions and 5 imaging modalities. The dataset integrates imaging, anatomical structures, demographics, and free-text annotations, supporting anatomical structure recognition, disease screening, disease staging, and demographic prediction for bias evaluation. This work extends our preliminary LMOD benchmark with three major enhancements: (1) nearly 50% dataset expansion with substantial enlargement of color fundus photography; (2) broadened task coverage including binary disease diagnosis, multi-class diagnosis, severity classification with international grading standards, and demographic prediction; and (3) systematic evaluation of 24 state-of-the-art MLLMs. Our evaluations reveal both promise and limitations. Top-performing models achieved ~58% accuracy in disease screening under zero-shot settings, and performance remained suboptimal for challenging tasks like disease staging. We will publicly release the dataset, curation pipeline, and leaderboard to potentially advance ophthalmic AI applications and reduce the global burden of vision-threatening diseases.
>
---
#### [replaced 083] Towards Inclusive Communication: A Unified Framework for Generating Spoken Language from Sign, Lip, and Audio
- **分类: cs.CV; cs.MM; eess.AS; eess.IV**

- **简介: 该论文属于多模态语言生成任务，旨在解决聋哑人群通信障碍问题。提出统一框架融合手语、唇读和音频，提升语音文本生成效果。**

- **链接: [https://arxiv.org/pdf/2508.20476](https://arxiv.org/pdf/2508.20476)**

> **作者:** Jeong Hun Yeo; Hyeongseop Rha; Sungjune Park; Junil Won; Yong Man Ro
>
> **摘要:** Audio is the primary modality for human communication and has driven the success of Automatic Speech Recognition (ASR) technologies. However, such audio-centric systems inherently exclude individuals who are deaf or hard of hearing. Visual alternatives such as sign language and lip reading offer effective substitutes, and recent advances in Sign Language Translation (SLT) and Visual Speech Recognition (VSR) have improved audio-less communication. Yet, these modalities have largely been studied in isolation, and their integration within a unified framework remains underexplored. In this paper, we propose the first unified framework capable of handling diverse combinations of sign language, lip movements, and audio for spoken-language text generation. We focus on three main objectives: (i) designing a unified, modality-agnostic architecture capable of effectively processing heterogeneous inputs; (ii) exploring the underexamined synergy among modalities, particularly the role of lip movements as non-manual cues in sign language comprehension; and (iii) achieving performance on par with or superior to state-of-the-art models specialized for individual tasks. Building on this framework, we achieve performance on par with or better than task-specific state-of-the-art models across SLT, VSR, ASR, and Audio-Visual Speech Recognition. Furthermore, our analysis reveals a key linguistic insight: explicitly modeling lip movements as a distinct modality significantly improves SLT performance by capturing critical non-manual cues.
>
---
#### [replaced 084] ASAP: Attention-Shift-Aware Pruning for Efficient LVLM Inference
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.14549](https://arxiv.org/pdf/2603.14549)**

> **作者:** Surendra Pathak; Bo Han
>
> **备注:** Update in V2: Added citations, refrences, and other minor rewrites
>
> **摘要:** While Large Vision-Language Models (LVLMs) demonstrate exceptional multi-modal capabilities, the quadratic computational cost of processing high-resolution visual tokens remains a critical bottleneck. Though recent token reduction strategies attempt to accelerate inference, such methods inadequately exploit attention values and fail to address token redundancy. More critically, they overlook the ``attention shift'' phenomenon inherent in LVLMs, which skews token attention scores. In this work, we propose ASAP, a novel training-free, KV-Cache-compatible pruning recipe that comprehensively addresses these limitations. First, we mitigate the attention shift by utilizing a dynamic bidirectional soft attention mask, ensuring the selection of genuinely informative tokens rather than naive attention-based selection. Second, we posit that high semantic redundancy within the token set degrades performance. We therefore introduce a weighted soft merging component that merges semantically similar tokens, preserving only the most feature-dense visual patches for subsequent layers. ASAP achieves virtually lossless compression of visual context, retaining 99.02% of the original LLaVA-NeXT-7B performance while aggressively slashing computational FLOPs by ~80%.
>
---
#### [replaced 085] Soft Dice Confidence: A Near-Optimal Confidence Estimator for Selective Prediction in Semantic Segmentation
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2402.10665](https://arxiv.org/pdf/2402.10665)**

> **作者:** Bruno Laboissiere Camargos Borges; Bruno Machado Pacheco; Danilo Silva
>
> **备注:** 48 pages, 11 figures
>
> **摘要:** In semantic segmentation, even state-of-the-art deep learning models fall short of the performance required in certain high-stakes applications such as medical image analysis. In these cases, performance can be improved by allowing a model to abstain from making predictions when confidence is low, an approach known as selective prediction. While well-known in the classification literature, selective prediction has been underexplored in the context of semantic segmentation. This paper tackles the problem by focusing on image-level abstention, which involves producing a single confidence estimate for the entire image, in contrast to previous approaches that focus on pixel-level uncertainty. Assuming the Dice coefficient as the evaluation metric for segmentation, two main contributions are provided in this paper: (i) In the case of known marginal posterior probabilities, we derive the optimal confidence estimator, which is observed to be intractable for typical image sizes. Then, an approximation computable in linear time, named Soft Dice Confidence (SDC), is proposed and proven to be tightly bounded to the optimal estimator. (ii) When only an estimate of the marginal posterior probabilities are known, we propose a plug-in version of the SDC and show it outperforms all previous methods, including those requiring additional tuning data. These findings are supported by experimental results on both synthetic data and real-world data from six medical imaging tasks, including out-of-distribution scenarios, positioning the SDC as a reliable and efficient tool for selective prediction in semantic segmentation.
>
---
#### [replaced 086] Benchmarking Endoscopic Surgical Image Restoration and Beyond
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.19161](https://arxiv.org/pdf/2505.19161)**

> **作者:** Jialun Pei; Diandian Guo; Donghui Yang; Zhixi Li; Yuxin Feng; Long Ma; Bo Du; Pheng-Ann Heng
>
> **备注:** This work has been accepted by CVPR 2026
>
> **摘要:** In endoscopic surgery, a clear and high-quality visual field is critical for surgeons to make accurate intraoperative decisions. However, persistent visual degradation, including smoke generated by energy devices, lens fogging from thermal gradients, and lens contamination due to blood or tissue fluid splashes during surgical procedures, severely impairs visual clarity. These degenerations can seriously hinder surgical workflow and pose risks to patient safety. To systematically investigate and address various forms of surgical scene degradation, we introduce a real- world open-source surgical image restoration dataset covering endoscopic environments, called SurgClean, which involves multi-type image restoration tasks from two medical sites, i.e., desmoking, defogging, and desplashing. SurgClean comprises 3,113 images with diverse degradation types and corresponding paired reference labels. Based on SurgClean, we establish a standardized evaluation benchmark and provide performance for 22 representative generic task-specific image restoration approaches, including 12 generic and 10 task-specific image restoration approaches. Experimental results reveal substantial performance gaps relative to clinical requirements, highlighting a critical opportunity for algorithm advancements in intelligent surgical restoration. Furthermore, we explore the degradation discrepancies between surgical and natural scenes from structural perception and semantic under- standing perspectives, providing fundamental insights for domain-specific image restoration research. Our work aims to empower restoration algorithms and improve the efficiency of clinical procedures.
>
---
#### [replaced 087] World Reconstruction From Inconsistent Views
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.16736](https://arxiv.org/pdf/2603.16736)**

> **作者:** Lukas Höllein; Matthias Nießner
>
> **备注:** project website: this https URL video: this https URL code: this https URL
>
> **摘要:** Video diffusion models generate high-quality and diverse worlds; however, individual frames often lack 3D consistency across the output sequence, which makes the reconstruction of 3D worlds difficult. To this end, we propose a new method that handles these inconsistencies by non-rigidly aligning the video frames into a globally-consistent coordinate frame that produces sharp and detailed pointcloud reconstructions. First, a geometric foundation model lifts each frame into a pixel-wise 3D pointcloud, which contains unaligned surfaces due to these inconsistencies. We then propose a tailored non-rigid iterative frame-to-model ICP to obtain an initial alignment across all frames, followed by a global optimization that further sharpens the pointcloud. Finally, we leverage this pointcloud as initialization for 3D reconstruction and propose a novel inverse deformation rendering loss to create high quality and explorable 3D environments from inconsistent views. We demonstrate that our 3D scenes achieve higher quality than baselines, effectively turning video models into 3D-consistent world generators.
>
---
#### [replaced 088] Next-Frame Decoding for Ultra-Low-Bitrate Image Compression with Video Diffusion Priors
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.15129](https://arxiv.org/pdf/2603.15129)**

> **作者:** Yunuo Chen; Chuqin Zhou; Jiangchuan Li; Xiaoyue Ling; Bing He; Jincheng Dai; Li Song; Guo Lu
>
> **摘要:** We present a novel paradigm for ultra-low-bitrate image compression (ULB-IC) that exploits the "temporal" evolution in generative image compression. Specifically, we define an explicit intermediate state during decoding: a compact anchor frame, which preserves the scene geometry and semantic layout while discarding high-frequency details. We then reinterpret generative decoding as a virtual temporal transition from this anchor to the final reconstructed this http URL model this progression, we leverage a pretrained video diffusion model (VDM) as temporal priors: the anchor frame serves as the initial frame and the original image as the target frame, transforming the decoding process into a next-frame prediction this http URL contrast to image diffusion-based ULB-IC models, our decoding proceeds from a visible, semantically faithful anchor, which improves both fidelity and realism for perceptual image compression. Extensive experiments demonstrate that our method achieves superior objective and subjective performance. On the CLIC2020 test set, our method achieves over 50% bitrate savings across LPIPS, DISTS, FID, and KID compared to DiffC, while also delivering a significant decoding speedup of up to $\times$5. Code will be released later.
>
---
#### [replaced 089] PubTables-v2: A new large-scale dataset for full-page and multi-page table extraction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.10888](https://arxiv.org/pdf/2512.10888)**

> **作者:** Brandon Smock; Valerie Faucon-Morin; Max Sokolov; Libin Liang; Tayyibah Khanam; Amrit Ramesh; Maury Courtland
>
> **备注:** 30 pages, added more experiments
>
> **摘要:** Table extraction (TE) is a key challenge in visual document understanding. Traditional approaches detect tables first, then recognize their structure. Recently, interest has surged in developing methods, such as vision-language models (VLMs), that can extract tables directly in their full page or document context. However, progress has been difficult to demonstrate due to a lack of annotated data. To address this, we create a new large-scale dataset, PubTables-v2. PubTables-v2 supports a number of challenging table extraction tasks. Notably, it is the first large-scale benchmark for multi-page table structure recognition. We evaluate several smaller specialized VLMs to establish baseline performance on these tasks. As we show, multi-page table recognition is a key gap in current models' capabilities. Interestingly, we show that introducing an image classifier that predicts when to merge tables across pages can significantly improve performance. Data, code, and models will be released at this https URL.
>
---
#### [replaced 090] MultiMedEval: A Benchmark and a Toolkit for Evaluating Medical Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2402.09262](https://arxiv.org/pdf/2402.09262)**

> **作者:** Corentin Royer; Bjoern Menze; Anjany Sekuboyina
>
> **备注:** Accepted at MIDL 2024
>
> **摘要:** We introduce MultiMedEval, an open-source toolkit for fair and reproducible evaluation of large, medical vision-language models (VLM). MultiMedEval comprehensively assesses the models' performance on a broad array of six multi-modal tasks, conducted over 23 datasets, and spanning over 11 medical domains. The chosen tasks and performance metrics are based on their widespread adoption in the community and their diversity, ensuring a thorough evaluation of the model's overall generalizability. We open-source a Python toolkit (this http URL) with a simple interface and setup process, enabling the evaluation of any VLM in just a few lines of code. Our goal is to simplify the intricate landscape of VLM evaluation, thus promoting fair and uniform benchmarking of future models.
>
---
#### [replaced 091] A Tutorial on ALOS2 SAR Utilization: Dataset Preparation, Self-Supervised Pretraining, and Semantic Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.15119](https://arxiv.org/pdf/2603.15119)**

> **作者:** Nevrez Imamoglu; Ali Caglayan; Toru Kouyama
>
> **备注:** 10 pages, 8 figures, 1 Table
>
> **摘要:** Masked auto-encoders (MAE) and related approaches have shown promise for satellite imagery, but their application to synthetic aperture radar (SAR) remains limited due to challenges in semantic labeling and high noise levels. Building on our prior work with SAR-W-MixMAE, which adds SAR-specific intensity-weighted loss to standard MixMAE for pretraining, we also introduce SAR-W-SimMIM; a weighted variant of SimMIM applied to ALOS-2 single-channel SAR imagery. This method aims to reduce the impact of speckle and extreme intensity values during self-supervised pretraining. We evaluate its effect on semantic segmentation compared to our previous trial with SAR-W-MixMAE and random initialization, observing notable improvements. In addition, pretraining and fine-tuning models on satellite imagery pose unique challenges, particularly when developing region-specific models. Imbalanced land cover distributions such as dominant water, forest, or desert areas can introduce bias, affecting both pretraining and downstream tasks like land cover segmentation. To address this, we constructed a SAR dataset using ALOS-2 single-channel (HH polarization) imagery focused on the Japan region, marking the initial phase toward a national-scale foundation model. This dataset was used to pretrain a vision transformer-based autoencoder, with the resulting encoder fine-tuned for semantic segmentation using a task-specific decoder. Initial results demonstrate significant performance improvements compared to training from scratch with random initialization. In summary, this work provides a guide to process and prepare ALOS2 observations to create dataset so that it can be taken advantage of self-supervised pretraining of models and finetuning downstream tasks such as semantic segmentation.
>
---
#### [replaced 092] Echo Planning for Autonomous Driving: From Current Observations to Future Trajectories and Back
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出EchoP框架，解决自动驾驶中轨迹预测与场景动态不一致的问题。通过CFC循环机制，提升轨迹预测的可靠性与一致性。**

- **链接: [https://arxiv.org/pdf/2505.18945](https://arxiv.org/pdf/2505.18945)**

> **作者:** Jintao Sun; Hu Zhang; Gangyi Ding; Zhedong Zheng
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Modern end-to-end autonomous driving systems suffer from a critical limitation: their planners lack mechanisms to enforce temporal consistency between predicted trajectories and evolving scene dynamics. This absence of self-supervision allows early prediction errors to compound catastrophically over time. We introduce Echo Planning (EchoP), a new self-correcting framework that establishes an end-to-end Current - Future - Current (CFC) cycle to harmonize trajectory prediction with scene coherence. Our key insight is that plausible future trajectories should be bi-directionally consistent, i.e., not only generated from current observations but also capable of reconstructing them. The CFC mechanism first predicts future trajectories from the Bird's-Eye-View (BEV) scene representation, then inversely maps these trajectories back to estimate the current BEV state. By enforcing consistency between the original and reconstructed BEV representations through a cycle loss, the framework intrinsically penalizes physically implausible or misaligned trajectories. Experiments on nuScenes show that the proposed method yields competitive performance, reducing L2 error (Avg) by -0.04 m and collision rate by -0.12% compared to one-shot planners. Moreover, EchoP seamlessly extends to closed-loop evaluation, i.e., Bench2Drive, attaining a 26.54% success rate. Notably, EchoP requires no additional supervision: the CFC cycle acts as an inductive bias that stabilizes long-horizon planning. Overall, EchoP offers a simple, deployable pathway to improve reliability in safety-critical autonomous driving.
>
---
#### [replaced 093] HyPER-GAN: Hybrid Patch-Based Image-to-Image Translation for Real-Time Photorealism Enhancement
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.10604](https://arxiv.org/pdf/2603.10604)**

> **作者:** Stefanos Pasios; Nikos Nikolaidis
>
> **备注:** This paper is under consideration at Pattern Recognition Letters
>
> **摘要:** Generative models are widely employed to enhance the photorealism of visual synthetic data for training computer vision algorithms. However, they often introduce visual artifacts that degrade the accuracy of these algorithms and require high computational resources, limiting their applicability in real-time training or evaluation scenarios. In this paper, we propose Hybrid Patch Enhanced Realism Generative Adversarial Network (HyPER-GAN), a lightweight image-to-image translation method based on a U-Net-style generator designed for real-time inference. The model is trained using paired synthetic and photorealism-enhanced images, complemented by a hybrid training strategy that incorporates matched patches from real-world images to improve visual realism and semantic consistency. Experimental results demonstrate that HyPER-GAN outperforms state-of-the-art lightweight paired image-to-image translation methods in terms of inference latency, visual realism, and semantic robustness. Moreover, it is illustrated that the proposed hybrid training strategy indeed improves visual quality and semantic consistency compared to training the model solely with paired synthetic and photorealism-enhanced images. Code and pretrained models are publicly available for download at: this https URL
>
---
#### [replaced 094] Den-TP: A Density-Balanced Data Curation and Evaluation Framework for Trajectory Prediction
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2409.17385](https://arxiv.org/pdf/2409.17385)**

> **作者:** Ruining Yang; Yi Xu; Yun Fu; Lili Su
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Trajectory prediction in autonomous driving has traditionally been studied from a model-centric perspective. However, existing datasets exhibit a strong long-tail distribution in scenario density, where common low-density cases dominate and safety-critical high-density cases are severely underrepresented. This imbalance limits model robustness and hides failure modes when standard evaluations average errors across all scenarios. We revisit trajectory prediction from a data-centric perspective and present Den-TP, a framework for density-aware dataset curation and evaluation. Den-TP first partitions data into density-conditioned regions using agent count as a dataset-agnostic proxy for interaction complexity. It then applies a gradient-based submodular selection objective to choose representative samples within each region while explicitly rebalancing across densities. The resulting subset reduces the dataset size by 50\% yet preserves overall performance and significantly improves robustness in high-density scenarios. We further introduce density-conditioned evaluation protocols that reveal long-tail failure modes overlooked by conventional metrics. Experiments on Argoverse 1 and 2 with state-of-the-art models show that robust trajectory prediction depends not only on data scale, but also on balancing scenario density.
>
---
