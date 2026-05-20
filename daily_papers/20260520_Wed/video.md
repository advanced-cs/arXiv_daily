# 计算机视觉 cs.CV

- **最新发布 168 篇**

- **更新 111 篇**

## 最新发布

#### [new 001] RECIPE: Procedural Planning via Grounding in Instructional Video
- **分类: cs.CV**

- **简介: 该论文提出RECIPE框架，解决视觉规划任务中标注数据不足和轨迹单一的问题。通过利用文本嵌入验证生成步骤的时序合理性，提升规划效果。**

- **链接: [https://arxiv.org/pdf/2605.19976](https://arxiv.org/pdf/2605.19976)**

> **作者:** Luigi Seminara; Antonino Furnari; Lorenzo Torresani
>
> **摘要:** Visual planning asks a model to generate the remaining steps of a procedure in natural language given a partial video context and a goal. Progress on this task is bottlenecked by annotation: clean labeled datasets are small, domain-narrow, and encode a single execution trajectory per example, even though many valid orderings exist. Large-scale instructional video corpora offer orders of magnitude more procedural content, but supervised fine-tuning on pseudo-labels from their noisy ASR narrations propagates segmentation and alignment errors and stays single-trajectory. We identify a key asymmetry: extracting clean step labels from noisy video is hard, but verifying whether a generated step sequence is temporally grounded in ASR transcripts is cheap and scales to millions of videos via precomputed text embeddings. We exploit this asymmetry in RECIPE, which uses grounding quality as a reward for GRPO, turning the noisy corpus into a verifier rather than a label source. The framework applies uniformly to two planner input configurations (Socratic, with a textual history extracted by a frozen VLM, and Video, consuming video tokens directly) and to annotated and weakly supervised regimes. We evaluate on 7 procedural benchmarks using a reference-based LLM-as-judge protocol scoring plans across 6 procedural criteria. RECIPE-RL improves over the base checkpoint at all scales (0.5B, 3B, 7B) and every benchmark, with macro-accuracy gains of +7 to +8 points in-domain and up to +16 points zero-shot. It outperforms supervised fine-tuning on both annotated and pseudo-labeled plans (the latter degrades the base) and remains robust without human annotations. Used as the proposal stage of a prior propose-assess-search planner, it improves over the strongest zero-shot baseline at every horizon on Visual Planning for Assistance, and on COIN it preserves the generation diversity that SFT collapses.
>
---
#### [new 002] FPED: A Functional-Network Prior-Guided Mixture-of-Experts Framework for Interpretable Brain Decoding
- **分类: cs.CV**

- **简介: 该论文属于脑解码任务，旨在提升视觉图像重建的可解释性。针对现有方法忽略脑网络结构的问题，提出FPED框架，通过功能网络先验引导专家模型，实现更准确和可解释的脑信号解码。**

- **链接: [https://arxiv.org/pdf/2605.19279](https://arxiv.org/pdf/2605.19279)**

> **作者:** Yudan Ren; Pengcheng Shi; Zihan Ma; Xiaowei He; Xiao Li
>
> **备注:** 15 pages,4 figures
>
> **摘要:** Visual image reconstruction from functional Magnetic Resonance Imaging (fMRI) is a fundamental task in brain decoding, providing a crucial pathway for understanding human perceptual mechanisms and developing advanced brain-computer interfaces (BCIs). However, most current methods simply flatten fMRI signals from localized visual cortices into one-dimensional (1D) vectors, mapping them directly into latent spaces such as that of Contrastive Language-Image Pre-training (CLIP). This paradigm not only disrupts the inherent network topology of the brain-leading to limited neuroscientific interpretability-but also overlooks the synergistic contributions of other distributed functional networks in processing high-level visual semantics. To address these limitations, we propose FPED, a Functional-Network Prior-Guided Mixture of Experts (MoE) framework for interpretable brain decoding. FPED explicitly models different functional brain networks as specialized experts and employs adaptive routing to capture their complementary contributions to visual semantic understanding. Unlike conventional homogeneous decoding paradigms, our framework incorporates neurobiologically grounded priors to enable structured and interpretable network-level representation learning. Experimental results demonstrate that FPED achieves highly competitive semantic reconstruction performance with only 0.68B parameters. The learned routing dynamics reveal biologically meaningful correspondence between functional brain networks and modality-specific semantic processing, providing transparent neuroscientific interpretability. This suggests that brain network-aware expert modeling is a promising direction for bridging neural decoding and biologically inspired artificial intelligence.
>
---
#### [new 003] A novel YOLO26-MoE optimized by an LLM agent for insulator fault detection considering UAV images
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于目标检测任务，旨在解决UAV图像中绝缘子故障检测难题。通过优化YOLO26-MoE模型并引入LLM代理进行优化，提升检测精度与效率。**

- **链接: [https://arxiv.org/pdf/2605.19595](https://arxiv.org/pdf/2605.19595)**

> **作者:** João Pedro Matos-Carvalho; Laio Oriel Seman; Stefano Frizzo Stefenon; Mohammad Khalaf Mohammad Khreasat; Gabriel Villarrubia González
>
> **摘要:** The inspection of electrical power line insulators is essential for ensuring grid reliability and preventing failures caused by damaged or degraded insulation components. In recent years, Unmanned Aerial Vehicles (UAVs) combined with deep learning-based vision systems have emerged as an effective solution for automating this process. However, insulator fault detection remains challenging due to small defect regions, heterogeneous fault patterns, complex backgrounds, and varying imaging conditions. To address these challenges, this paper proposes an optimized YOLO26-MoE, a novel object detection architecture that integrates a sparse Mixture-of-Experts (MoE) module into the high-resolution branch of the YOLO26 detector. The proposed modification enables adaptive feature refinement for subtle and diverse fault patterns while preserving the efficiency of a one-stage detection framework. Hyperparameter optimization, final training, and evaluation were coordinated through a tool-augmented Large Language Model (LLM) agent. The proposed model achieved 0.9900 mAP@0.5 and 0.9515 mAP@0.5:0.95, outperforming the latest YOLO versions. These results demonstrate that the proposed model provides an effective and reliable solution for UAV-based insulator fault detection.
>
---
#### [new 004] Targeted Downstream-Agnostic Attack
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于安全攻击任务，旨在解决预训练编码器在无目标攻击下的脆弱性问题。提出一种新的有目标的下游无关攻击方法，通过生成特定扰动使编码器输出相同特征，揭示其漏洞。**

- **链接: [https://arxiv.org/pdf/2605.19446](https://arxiv.org/pdf/2605.19446)**

> **作者:** Zhuxin Lei; Ziyuan Yang; Yi Zhang
>
> **摘要:** Recently, pre-trained encoders have gained widespread use due to their strong capability in representation extraction. However, they are vulnerable to downstream-agnostic attacks (DAAs). Existing DAA methods operate under a permissive threat model, where an attack is successful if the generated downstream-agnostic adversarial examples (DAEs) change the original prediction, without requiring a specific target. In this paper, we propose a Targeted DAA (TDAA) method under a stricter threat model requiring the attack to be both targeted and downstream-agnostic. Since the downstream task is unknown and encoders do not directly produce predictions, achieving a targeted attack is particularly challenging. To address this, we introduce a novel component termed the 'threat image', pre-selected by the attacker as the target. Specifically, a generator is designed to produce example-specific adversarial perturbations that compel the victim encoder to output identical features for both the DAEs and the threat image. Unlike previous DAA methods that generate a single shared perturbation for all samples, which often fails due to image diversity, our method adopts an example-specific paradigm. This generates tailored perturbations for each image to ensure a high attack success rate and invisibility. By leveraging the threat image as a feature-level anchor, our method builds a task-agnostic bridge to reveal the vulnerabilities of the victim encoder. Extensive experiments on 10 self-supervised methods across 3 benchmark datasets demonstrate the effectiveness of our approach and reveal the pronounced vulnerability of pre-trained encoders. The code will be made publicly available after the review period.
>
---
#### [new 005] Towards Camera-Robust 3D Localization: Equation-Anchored Tool-Use for MLLMs
- **分类: cs.CV**

- **简介: 该论文聚焦3D定位任务，解决相机内参模糊导致的模型泛化问题。通过引入方程锚定工具使用，显式整合相机参数与深度信息，提升模型在不同相机尺度下的性能。**

- **链接: [https://arxiv.org/pdf/2605.19528](https://arxiv.org/pdf/2605.19528)**

> **作者:** Xueying Jiang; Wenhao Li; Quanhao Qian; Deli Zhao; Shijian Lu; Gongjie Zhang; Ran Xu
>
> **摘要:** 3D localization in Multimodal Large Language Models (MLLMs), including 3D object detection and 3D visual grounding, is fundamentally limited by camera intrinsic ambiguity: the same image admits different 3D scenes under different cameras. Existing MLLMs either ignore camera parameters and overfit to a canonical training intrinsic, or retrieve depth and 3D cues from external tools but treat the returned values as reference cues (numerical hints that the model is free to interpret implicitly), both preventing camera information from being deterministically propagated into the prediction. We propose an equation-anchored tool-use framework that re-purposes spatial tools as formula variables. The proposed framework proactively retrieves camera intrinsics and samples multi-point metric depths, writes the pinhole back-projection equation $\hat{X} = (u_c - c_x)\bar{Z}/f_x$ explicitly in Chain-of-Thought (CoT), and substitutes tool outputs into the formula before regressing the final 9-DoF bounding box. On both 3D object detection and 3D visual grounding tasks under rescaled camera intrinsics from $0.5\times$ to $1.5\times$, our method outperforms RGB-only and tool-augmented baselines, with significant gains where the camera deviates most from the training scale. Code and data will be released.
>
---
#### [new 006] EgoTraj: Real-World Egocentric Human Trajectory Dataset for Multimodal Prediction
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出EgoTraj数据集，用于解决真实世界第一视角人类轨迹预测问题。通过多模态数据收集与分析，提升AR导航与辅助系统性能。**

- **链接: [https://arxiv.org/pdf/2605.19004](https://arxiv.org/pdf/2605.19004)**

> **作者:** Ahmad Yehia; Abduallah Mohamed; Tianyi Wang; Jiseop Byeon; Kun Qian; Junfeng Jiao; Christian Claudel
>
> **备注:** 21 pages, 14 figures. Project page: this https URL
>
> **摘要:** Accurately forecasting human trajectories from an egocentric perspective plays a central role in applications such as humanoid robotics, wearable sensing systems, and assistive navigation. However, progress in this direction remains limited due to the scarcity of egocentric trajectory datasets collected in real-world environments. Addressing this need, we introduce EgoTraj, an egocentric multimodal open dataset recorded using Meta Quest Pro (MQPro). EgoTraj contains 75 sequences of human navigation collected from multiple MQPro wearers in real-world urban environments. Each recording provides synchronized RGB video along with ground-truth data, including continuous time-synchronized 6-degree-of-freedom head poses, per-frame 3D eye gaze vectors, scene annotations. To the best of our knowledge, EgoTraj differs from typical egocentric trajectory datasets by capturing long-horizon, self-directed navigation across diverse urban routes with broad participant diversity. To demonstrate the potential of the dataset, we benchmark several state-of-the-art methods for egocentric trajectory prediction and conduct ablation studies to analyze the contributions of gaze, scene, and motion cues. The results highlight the utility of EgoTraj for AR-based perception, navigation, and assistive systems. The EgoTraj dataset, code, and EgoViz Dashboard are publicly available at this https URL.
>
---
#### [new 007] iDiff: Interpretable Difference-aware Framework for Pairwise Image Quality Assessment
- **分类: cs.CV**

- **简介: 该论文属于图像质量评估任务，解决 pairwise IQA 中同时预测偏好和生成解释的问题。提出 iDiff 框架，通过双分支模型实现准确决策与结构化解释。**

- **链接: [https://arxiv.org/pdf/2605.19522](https://arxiv.org/pdf/2605.19522)**

> **作者:** Xinli Yue; JianHui Sun; Tao Shao; Liangchao Yao; Fan Xia; Yuetang Deng
>
> **备注:** Accepted to CVPR 2026 Workshop
>
> **摘要:** Pairwise image quality assessment (IQA) in professional photography requires a model not only to identify the preferred image between two candidates, but also to provide convincing and image-grounded reasoning. In the NTIRE 2026 RAIM challenge, this requirement is further emphasized by jointly evaluating preference prediction and rationale generation. To address this task, we propose iDiff, an Interpretable Difference-aware framework for pairwise image quality assessment. Our method adopts a dual-branch design consisting of an Answer Model and a Thinking Model. The Answer Model performs robust preference prediction by explicitly decomposing each sample into left/right global and local views, followed by content-aware specialization for person and scene images and ensemble-based aggregation across backbones. The Thinking Model focuses on rationale generation and is progressively enhanced with expert-style templates, multi-source quality features, and answer-aware supervision conditioned on the Answer Model prediction. In this way, iDiff jointly models discriminative decision making and structured explanation, improving both robustness and interpretability. Extensive experiments demonstrate the effectiveness of the proposed framework on both accuracy and reasoning-quality metrics. Our method achieved first place in the NTIRE 2026 RAIM challenge, showing the effectiveness of integrating explicit difference modeling with structured multimodal reasoning for pairwise IQA.
>
---
#### [new 008] OP2GS: Object-Aware 3D Gaussian Splatting with Dual-Opacity Primitives
- **分类: cs.CV**

- **简介: 该论文提出OP2GS，解决3DGS缺乏对象级身份的问题，通过双透明度机制区分视觉存在与对象占据，提升开放词汇场景理解效果。**

- **链接: [https://arxiv.org/pdf/2605.20044](https://arxiv.org/pdf/2605.20044)**

> **作者:** Guiyu Liu; Niklas Vaara; Janne Mustaniemi; Juho Kannala; Janne Heikkilä
>
> **备注:** Under review
>
> **摘要:** 3D Gaussian Splatting (3DGS) provides an explicit and efficient scene representation, but its primitives lack inherent object-level identity, hindering downstream tasks such as open-vocabulary scene understanding. Existing methods typically address this by either distilling high-dimensional feature embeddings into Gaussians or by lifting 2D mask labels into 3D via heuristic refinement. However, feature-based approaches incur heavy storage and decoding overhead, while lifting-based pipelines remain vulnerable to label contamination: Gaussians necessary for appearance reconstruction often receive incorrect object labels during 2D-to-3D projection. We propose OP2GS, an object-aware Gaussian representation that augments each primitive with an explicit instance identity and a dedicated instance opacity $\sigma^{*}$ for object-mask rendering. The original opacity $\sigma$ remains responsible for visual reconstruction, while $\sigma^{*}$ models whether a Gaussian should contribute to a particular object mask. This dual-opacity formulation decouples visual existence from instance occupancy: mislabeled Gaussians can remain available for image rendering while becoming transparent in the object-mask branch. To learn this representation, we introduce a random object loss that optimizes the 1D instance occupancy field using the standard transmittance-based visibility of 3DGS. Semantic descriptors are then attached at the object level through multi-view aggregation, eliminating per-Gaussian feature storage. Compared with feature-training approaches, OP2GS achieves competitive open-vocabulary performance while significantly reducing computational overhead. Compared with training-free pipelines, it leverages physically consistent occupancy learning to resolve visibility ambiguities.
>
---
#### [new 009] Real-World On-Vehicle Evaluation of Embedding-Based Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于异常检测任务，旨在解决自动驾驶中真实场景异常检测难题。通过预训练视觉Transformer嵌入，利用最近邻相似性实现无需监督的实时异常检测。**

- **链接: [https://arxiv.org/pdf/2605.19744](https://arxiv.org/pdf/2605.19744)**

> **作者:** Albert Schotschneider; Daniel Bogdoll; Svetlana Pavlitska; Ahmed Abouelazm; Johann Marius Zoellner
>
> **备注:** Accepted at CVPR 2026 Workshop AUTOPILOT-NA
>
> **摘要:** Detecting anomalies in traffic scenes is crucial for ensuring safety in autonomous driving, yet collecting representative anomalous data remains challenging. Existing anomaly detection methods are highly specialized and rely on normality as defined by the abstract semantic Cityscapes classes, making it difficult to adapt to diverse real-world scenarios. We propose an adaptable real-time anomaly detection method that leverages foundation models in the form of pretrained vision transformer embeddings to detect deviations via nearest-neighbor similarity in the latent semantic feature space. Based on patch-wise processing, the algorithm produces dense anomaly masks, allowing for the localization of detected anomalies. The method robustly models normality through a single reference image. This formulation avoids explicit supervision and dataset-specific training, making it suitable for real-world deployment. We evaluate the method on standard benchmarks and on an automated vehicle in real-world scenarios. Despite its simplicity, the method achieves good performance on the Road Anomaly benchmark and demonstrates consistent qualitative behavior in practice, successfully highlighting semantically unusual objects in diverse scenes. These results suggest that simple, reference-based methods can provide useful anomaly signals under realistic operating conditions.
>
---
#### [new 010] Mechanisms of Object Localization in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型中的目标定位机制，解决其定位过程不明确的问题。通过分析两个模型，发现定位依赖于对象对齐的token容器机制。**

- **链接: [https://arxiv.org/pdf/2605.19792](https://arxiv.org/pdf/2605.19792)**

> **作者:** Timothy Schaumlöffel; Martina G. Vilas; Gemma Roig
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Visually-grounded language models (VLMs) are highly effective in linking visual and textual information, yet they often struggle with basic classification and localization tasks. While classification mechanisms have been studied more extensively, the processes that support object localization remain poorly understood. In this work, we investigate two representative families, LLaVA-1.5 and InternVL-3.5, using a suite of mechanistic interpretability tools, including token ablations, attention knockout, and causal mediation analysis. We find that localization is driven by a containerization mechanism in which object-aligned tokens define the spatial extent of the object, while the semantic arrangement of tokens within those boundaries is largely irrelevant to the predicted box. Only a very small set of attention heads mediates the causal effect for both classification and localization, concentrating in early-mid layers for LLaVA and mid-late layers for InternVL. The two tasks share some early processing but ultimately depend on largely distinct specialized heads. Overall, we provide the first layer- and head-level account of localization in VLMs, revealing narrow computational pathways that can guide future model design and grounding objectives.
>
---
#### [new 011] X-Ray cardiac angiographic vessel segmentation based on pixel classification using machine learning and region growing
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决X射线心脏造影中血管的精确分割问题。通过像素分类与区域生长方法，结合多种特征和随机森林分类器，实现了高精度的血管分割。**

- **链接: [https://arxiv.org/pdf/2605.20073](https://arxiv.org/pdf/2605.20073)**

> **作者:** E O Rodrigues; L O Rodrigues; J J Lima; D Casanova; F Favarim; E R Dosciatti; V Pegorini; L S N Oliveira; F F C Morais
>
> **摘要:** This work proposes a pixel-classification approach for vessel segmentation in x-ray angiograms. The proposal uses textural features such as anisotropic diffusion, features based on the Hessian matrix, mathematical morphology and statistics. These features are extracted from the neighborhood of each pixel. The approach also uses the ELEMENT methodology, which consists of creating a pixel-classification controlled by region-growing where the result of the classification affects further classifications of pixels. The Random Forests classifier is used to predict whether the pixel belongs to the vessel structure. The approach achieved the best accuracy in the literature (95.48%) outperforming unsupervised state-of-the-art approaches.
>
---
#### [new 012] Breaking Modality Heterogeneity in Low-Bit Quantization for Large Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型量化任务，解决低比特量化中跨模态分布不一致导致的精度下降问题，提出SplitQ框架有效分离异常通道并校准差异。**

- **链接: [https://arxiv.org/pdf/2605.19929](https://arxiv.org/pdf/2605.19929)**

> **作者:** Yi Zhong; Haotong Qin; Xindong Zhang; Lei Zhang; Guolei Sun
>
> **摘要:** Low-bit post-training quantization (PTQ) is a pivotal technique for deploying Vision-Language Models (VLMs) on resource-constrained devices. However, existing PTQ methods often degrade VLMs' accuracy due to the heterogeneous activation distributions of text and vision modalities during quantization. We find that this cross-modal heterogeneity is distributed unevenly across channels: a small subset of channels contains most modality-specific outliers, and these outliers typically reside in different channels for each modality. Motivated by this, we propose SplitQ, a channel-Splitting-driven post-training Quantization framework. At its core, SplitQ introduces a novel Modality-specific Outlier Channel Decoupling (MOCD) module that effectively isolates salient modality-specific outlier channels with minimal overhead. To further address the remaining cross-modal distribution discrepancies, we design an Adaptive Cross-Modal Calibration (ACC) module that employs dual lightweight learnable branches to dynamically mitigate modality-induced quantization errors. Extensive experiments on popular VLMs demonstrate that SplitQ significantly outperforms existing approaches across 6 popular multi-modal datasets under all evaluated quantization settings, including W4A8, W4A4, W3A3, and W3A2. Notably, SplitQ preserves 93.5% of FP16 performance under the challenging W3A3 setting (69.5 vs. 74.3), pushing the efficiency frontier for deploying advanced VLMs. Our code is available at this https URL
>
---
#### [new 013] HAVEN: Hierarchically Aligned Multimodal Benchmark for Unified Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决多模态模型在复杂叙事总结与推理上的评估不足。提出HAVEN基准，实现跨模态的层级对齐与统一评估。**

- **链接: [https://arxiv.org/pdf/2605.19223](https://arxiv.org/pdf/2605.19223)**

> **作者:** Mengqi Shi; Haopeng Zhang
>
> **摘要:** While Multimodal Large Language Models (MLLMs) exhibit strong performance on standard video tasks, their ability to faithfully summarize and reason over complex narratives remains poorly evaluated. Existing summarization benchmarks fragment supervision across isolated granularities, such as keyframes, key shots, or disjointed text summaries, failing to capture the inherently hierarchical structure of cross-modal alignment. To address this critical gap, we introduce HAVEN, a hierarchically aligned multimodal benchmark for unified video understanding. HAVEN pioneers a fully granular (frame, shot, and video levels) and fully multimodal (video and text) dataset architecture, complete with explicit, continuous alignment between modalities. Built upon this unified annotation paradigm, we propose a comprehensive evaluation suite spanning summarization, temporal reasoning, multimodal grounding, and saliency ranking. Extensive benchmarking of state-of-the-art MLLMs exposes a persistent gap between surface-level textual fluency and grounded multimodal understanding. Ultimately, HAVEN advances the evaluation of multimodal systems beyond traditional QA formats, offering a rigorous, standardized testbed to drive future research in interpretable, hierarchical video understanding. We publicly release the dataset, benchmark suite, and evaluation protocols.
>
---
#### [new 014] Self-Creative Text-to-Object Generation using Semantic-Aware Spatial Weighting
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决模型缺乏创造力的问题。通过引入空间加权和视觉语义混合损失模块，提升生成图像的创意性和语义一致性。**

- **链接: [https://arxiv.org/pdf/2605.19554](https://arxiv.org/pdf/2605.19554)**

> **作者:** Yue Yu; Haibo Chen; Shuo Chen; Jian Yang; Jun Li
>
> **摘要:** Instilling creativity in text-to-image (T2I) generation presents a significant challenge, as it requires synthesized images to exhibit not only visual novelty and surprise, but also artistic value. Current T2I models, however, are largely optimized for literal text-image alignment with their data distribution, and their noise prediction networks constrain the generation to high-probability regions, consequently generating outputs that lack authentic creativity. To address this, we propose a Self-Creative Diffusion (SCDiff) model for meaningful T2I generations featuring two core modules: a learnable spatial weighting (LSW) module and a visual-semantic mixing loss (VSML). The LSW module designs a parametric Kaiser-Bessel window to reinforce central image features, fostering novel and surprising generation. The VSML module introduces a dual loss function: a similarity loss constrains that the new images align with its textual description, while a diversity loss maximizes its distinction from the original image, enhancing both semantic value and visual novelty. Extensive experiments demonstrate that our model substantially improves creativity, semantic alignment, and visual coherence, offering a simple yet powerful framework for generating creative objects.
>
---
#### [new 015] Personalized Face Privacy Protection From a Single Image
- **分类: cs.CV**

- **简介: 该论文属于人脸隐私保护任务，旨在防止面部信息被滥用。通过单张图像生成个性化防护面具，有效干扰人脸识别，提升用户隐私安全。**

- **链接: [https://arxiv.org/pdf/2605.19032](https://arxiv.org/pdf/2605.19032)**

> **作者:** Zachary Yahn; Fatih Ilhan; Tiansheng Huang; Selim Tekin; Sihao Hu; Yichang Xu; Margaret Loper; Ling Liu
>
> **摘要:** Photos of faces uploaded online are vulnerable to malicious actors who can scrape facial images from online sources and intrude on personal privacy via unauthorized use of facial recognition models. This paper presents FaceCloak, a novel personalized face privacy protection system, which can generate defensive identity-specific universal face privacy masks from a single image of a user, causing facial recognition to fail. FaceCloak introduces a three-stage personalized face perturbation learning methodology: (1) It generates a small set of high-variety synthetic face images of a person based on a single image of the person. (2) It learns face cloaking by adding more protection to key facial-identity leakage regions through iterative perturbation generation over the small set of synthetic images, effectively shifting a user's identity embedding towards a distant anchor identity and away from a similar one. (3) It generates a personalized identity-protective mask in the form of pixel-wise cloaking, which is light-weight and can be efficiently applied to any facial image of a user while maintaining good perceptual quality. Extensive experiments on three popular face datasets across ten recognition models show the effectiveness of FaceCloak compared to 29 other existing representative methods. Code is available at this https URL
>
---
#### [new 016] Fast 4D Mesh Generation by Spatio-Temporal Attention Chains
- **分类: cs.CV**

- **简介: 该论文属于4D mesh生成任务，旨在解决现有方法速度慢、计算成本高及难以扩展的问题。提出Spatio-Temporal Attention Chain框架，提升生成效率和时间一致性。**

- **链接: [https://arxiv.org/pdf/2605.19786](https://arxiv.org/pdf/2605.19786)**

> **作者:** Dvir Samuel; Yuval Atzmon; Gal Chechik; Yoni Kasten
>
> **备注:** this https URL
>
> **摘要:** 4D mesh generation has recently emerged as a powerful paradigm for recovering dynamic 3D structure from videos, but existing methods remain slow, computationally expensive, and difficult to scale to longer sequences. We introduce a training-free approach that accelerates 4D mesh generation while improving temporal correspondence quality. Our key observation is that temporal correspondences emerge inside a 4D backbone long before its generated meshes become visually accurate. We exploit this with a general framework we call Spatio-Temporal Attention Chain which propagates information across space and time. Starting from vertices on an anchor mesh, the chain maps vertices to latent tokens. It then follows temporal correspondences in latent space, and recovers frame-specific vertices through latent-to-vertex attention. This design avoids expensive explicit matching while preserving anchor mesh details and thereby improving dynamic mesh geometry and temporal consistency. Compared to state-of-the-art, our method generates a 4D mesh in 9 seconds, achieving a $13\times$ speedup while producing higher-quality results. Moreover, our approach scales to videos up to $16\times$ longer without degrading mesh quality. Beyond generation, the improved correspondences enable competitive zero-shot performance on two downstream tasks: 2D object tracking and 4D tracking. We further show that our framework enables reliable camera estimation, a capability not supported by prior 4D mesh generation methods.
>
---
#### [new 017] Sparse Mixture-of-Experts Routing in Visual Diffusion Transformers:Diagnosis, Boundary Calibration and Evolutionary Roadmap from Routing Collapse to Selective Deadlock
- **分类: cs.CV**

- **简介: 该论文研究视频扩散Transformer中的稀疏专家路由问题，诊断训练失败模式并提出解决方案，属于模型优化任务。**

- **链接: [https://arxiv.org/pdf/2605.19378](https://arxiv.org/pdf/2605.19378)**

> **作者:** Haiying Sha
>
> **摘要:** This paper systematically diagnoses the training failure modes of Token-Choice sparse Mixture-of-Experts (MoE) on video Diffusion Transformers. Starting from a pretrained dense model of about 5 billion parameters, we convert it into an MoE architecture following three laws: routed experts exactly clone the original FFN weights, shared experts are initialized to zero for verification and then to extremely small non-zero noise for actual training, while only the gating networks start from random initialization. Experiments reveal a hierarchy of five failure modes: (1) linear routers suffer global soft saturation with complete expert homogenization; (2) MLP routers introduce selective deadlock, where roughly one-third of layers degenerate into a single-expert mode that cannot be prevented by increasing the auxiliary loss; (3) cross-attention routers exhibit preliminary self-recovery, yet about nine layers remain stubbornly deadlocked; (4) deadlocked layers display a U-shaped distribution, concentrated in shallow visual processing layers and deep semantic integration layers; (5) bfloat16 mixed precision causes tiny weight updates to be truncated to zero by hardware. Based on routing decision time series over 65 million tokens across 5,000 training steps, we propose the Functional Redundancy Hypothesis: deadlock is a rational waiting strategy before the shared expert matures within the gate-shared expert-routed expert triadic system. This hypothesis is supported by the theory of functional redundancy in systems biology. On the engineering side, we summarize the Three Laws of dense-to-MoE conversion and provide a complete solution for the bfloat16 precision trap. We calibrate the current capability boundary of the Token-Choice paradigm and outline a three-step evolutionary roadmap from visual unification to a world model.
>
---
#### [new 018] Rotation-Aligned Key Channel Pruning for Efficient Vision-Language Model Inference
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型优化任务，针对推理时KV缓存压力大的问题，提出RotateK方法通过结构化通道剪枝，在保持精度的同时提升推理效率。**

- **链接: [https://arxiv.org/pdf/2605.19218](https://arxiv.org/pdf/2605.19218)**

> **作者:** Beomseok Kang; Dongwon Jo; Jiwon Song; Donghwee Son; Jae-Joon Kim
>
> **摘要:** Vision-Language Models suffer severe KV cache pressure at inference, as a single image often encodes into thousands of tokens. Most existing methods exploit token sparsity through token pruning, but permanently discarding visual content causes substantial degradation on fine-grained perception tasks. This motivates a complementary axis, feature sparsity: under a fixed KV cache budget, compressing the channel dimension preserves more visual tokens at the same memory cost. Prior Key channel pruning methods, however, face a structural trade-off: token-wise channel pruning is expressive but unstructured and slow, while head-wise approach is hardware-friendly but less robust. We resolve this with RotateK, a rotation-based structured Key channel pruning framework. RotateK applies an online PCA-based rotation that aligns token-dependent channel importance into a shared low-dimensional subspace, enabling accurate pruning under lightweight head-wise masks; a fused Triton attention kernel operates directly on sparse-channel Keys for efficient decoding. Experiments on two representative VLM backbones show that RotateK consistently outperforms prior Key channel pruning in both accuracy and decoding latency, while joint token-channel pruning improves over token-only baselines at matched KV cache budgets.
>
---
#### [new 019] Multi-Scale Generative Modeling with Heat Dissipation Flow Matching
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像生成任务，解决blur-based模型与ODE框架整合及逆热耗散问题。提出HDFM，引入连续模糊过程并采用x-预测提升性能。**

- **链接: [https://arxiv.org/pdf/2605.19371](https://arxiv.org/pdf/2605.19371)**

> **作者:** Jun Ma; Hanquan Zhang; Yanjun Qin; Haoyuan Guan; Ke Zhang
>
> **摘要:** Diffusion models are widely used in image generation, with most relying on noise-based corruption and denoising. A distinct branch instead uses blur as the main corruption, preserving better color budgets and multi-scale detail by providing multi-scale priors. However, blur-based models remain in SDE-based frameworks and are not integrated into ODE-based frameworks, such as Flow Matching (FM). Meanwhile, in the blur-based formulation, the classical inverse heat-dissipation (IHD) process faces an ill-posed challenge. Moreover, under the data-manifold assumption, regressing blurred images from high-dimensional noise (or velocity) space is also difficult. We propose Heat Dissipation Flow Matching (HDFM), which introduces a continuous blurred (heat-dissipation) process into FM to inject multi-scale priors. HDFM aligns an interpolated heat-dissipation path to address ill-posedness and adopts $x$-prediction to mitigate high-dimensional regression difficulty. Toy experiments and ablation studies show that HDFM consistently benefits from both blur and $x$-prediction. The performance of HDFM outperforms most baseline methods on all datasets.
>
---
#### [new 020] Component-Aware Structure-Preserving Style Transfer for Satellite Sim2Real 6D Pose Estimation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于卫星6D位姿估计任务，解决真实与合成数据间外观差异问题。通过组件感知的风格迁移方法，提升合成数据在真实场景中的适用性。**

- **链接: [https://arxiv.org/pdf/2605.19624](https://arxiv.org/pdf/2605.19624)**

> **作者:** Yonglong Zhang
>
> **摘要:** Monocular 6D pose estimation for non-cooperative satellites depends heavily on annotated training data, yet real satellite images with reliable pose labels and component-level masks are difficult to acquire at scale. Synthetic rendering can provide exact geometric annotations, but the appearance gap between rendered and real observations limits direct transfer to the real domain. This paper presents a component-aware structure-preserving style transfer framework for satellite synthetic-to-real data construction. The method builds weakly paired real--synthetic samples from calibrated real acquisition, ArUco-based camera-pose measurement, CAD rendering, and component masks. It then extracts part-wise real-domain style codes from unlabeled real images and injects them into corresponding synthetic satellite regions through mask-aligned modulation. To keep the generated images usable for downstream supervision, adversarial training is combined with local contrastive consistency, self-regularization, and edge-preserving constraints. Experiments are conducted on 5,000 rendered satellite images and 100 real images captured in a calibrated setup. The real images provide target-domain appearance references and final evaluation images, while the downstream GDRNet pose estimator is trained only on synthetic or translated synthetic images. Compared with representative image-translation baselines, the proposed method achieves the lowest image distribution discrepancy, with an FID of 54.32 and a KID of 0.048. When the translated data are used to train GDRNet in this target-domain adaptation setting, the ADD pass rate improves to 0.260 and the AUC improves to 0.611. These results indicate that component-level appearance transfer can improve satellite Sim2Real pose estimation in the considered calibrated setup while retaining simulation-derived geometric annotations.
>
---
#### [new 021] CogOmniControl: Reasoning-Driven Controllable Video Generation via Creative Intent Cognition
- **分类: cs.CV**

- **简介: 该论文提出CogOmniControl，解决视频生成中对用户创意意图理解不足的问题。通过专门训练的CogVLM和统一控制机制，提升生成视频与用户意图的一致性。**

- **链接: [https://arxiv.org/pdf/2605.19995](https://arxiv.org/pdf/2605.19995)**

> **作者:** Hongji Yang; Songlian Li; Yucheng Zhou; Xiaotong Zhao; Alan Zhao; Chengzhong Xu; Jianbing Shen
>
> **摘要:** Recent diffusion models achieve strong photorealism and fluency in video generation, yet remain fragile under abstract, sparse or complex conditions, leading to poor performance in professional production workflows such as storyboard sketches and clay render conditions. Existing video generation models, either inject conditions through adapters or couple a generic vision-language model (VLM) within a diffusion backbone, leaving a capability gap and failing to produce the videos that align with the user's creative intent. We present CogOmniControl, a reasoning-driven framework that factorizes controllable video generation into creative intent cognition and generation. Specifically, we train a specialized CogVLM using authentic anime production data. Compared to generic VLMs, it generates more professional and clear outputs, accurately cognizing user creative intent from sparse and abstract conditions and tuning these cues into dense reasoning output. Besides, CogOmniDiT unifies the controls from various conditions through in-context generation and is aligned to the CogVLM reasoning outputs via reinforcement learning. Furthermore, leveraging CogVLM's robust capability in guiding video generation, we release its potential in planning specific evaluators and enable a Best-of-N selection for the generated videos. This integration transforms the entire framework into a closed-loop "harness-like" architecture. We further introduce CogReasonBench and CogControlBench, built from professional workflows data that carry genuine creative intent rather than simulated ones. Experiments on two benchmarks show that CogOmniControl surpassed the existing open-source models. The project website: this https URL
>
---
#### [new 022] Aero-World: Action-Conditioned Aerial Video Generation from Inertial Controls
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决空中飞行中动作控制与视频生成不匹配的问题。提出Aero-World方法，通过注入惯性信号生成可控的空中视频，并构建AeroBench评估基准。**

- **链接: [https://arxiv.org/pdf/2605.19728](https://arxiv.org/pdf/2605.19728)**

> **作者:** Abdul Mohaimen Al Radi; Kunyang Li; Yuzhang Shang; Mubarak Shah; Yu Tian
>
> **摘要:** Foundation video models produce visually impressive results, but their use in embodied AI remains limited because they are primarily trained on natural language rather than low-level control signals. This limitation is especially pronounced for aerial flight, where motion occurs in unconstrained 6-DoF space and small errors in ego-motion can produce large trajectory drift. Generating aerial videos that follow fine-grained inertial actions can support scalable training and evaluation of aerial agents by providing a controllable proxy for real-world or expensive simulation data. To address this problem, we propose \textbf{Aero-World}, a method for converting a pretrained image-to-video diffusion model into a controllable aerial video generator. Aero-World injects sequences of translational acceleration and angular velocity into a pretrained latent diffusion transformer through an action-token stream. A frozen latent-space Physics Probe, trained independently on real video--IMU pairs, provides differentiable inertial-consistency supervision during LoRA finetuning while avoiding computationally expensive video decoding. We further propose \textbf{AeroBench}, a benchmark for evaluating whether generated drone videos adhere to low-level action signals. AeroBench uses Action Alignment Score (AAS) to measure agreement with commanded inertial actions and Physical Consistency Rate (PCR) to measure temporal motion stability. On AeroBench, Aero-World improves mean AAS from 57.7 to 63.6 over action-only finetuning and gives a stronger quality-control trade-off than AirScape, with lower FVD (596.5 vs. 1058.6), higher SSIM (0.595 vs. 0.505), and higher Flow-IMU correlation (0.44 vs. 0.20). These results suggest that frozen Physics Probe supervision is a practical mechanism for adapting pretrained video generators toward more action-aligned aerial motion.
>
---
#### [new 023] Boosting Text-to-Image Diffusion Models via Core Token Attention-Based Seed Selection
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于文本到图像生成任务，解决随机种子对生成质量影响大的问题。通过分析核心词注意力动态，提出ABSS方法，提升图像质量和文本对齐。**

- **链接: [https://arxiv.org/pdf/2605.19532](https://arxiv.org/pdf/2605.19532)**

> **作者:** Yunzhe Zhang; Hongfu Liu; Pengyu Hong
>
> **备注:** Preprint
>
> **摘要:** Text-to-image diffusion models can synthesize high-quality images, yet the outcome is notoriously sensitive to the random seed: different initial seeds often yield large variations in image quality and prompt-image alignment. We revisit this "seed effect" and show that attention dynamics over prompt core tokens, the content-bearing words, measured during the first few denoising steps, strongly predict final generation quality. Building on this observation, we introduce Attention-Based Seed Selection (ABSS), a training-free, plug-and-play method that ranks seeds for a given prompt by leveraging cross-attention to core tokens during the denoising process. ABSS requires no finetuning and does not alter the initial noise; it scores and ranks all candidate seeds, keeps only the top-k for full generation, and discards the rest, without relying on a fixed accept/reject threshold. Operating purely at inference time, ABSS can serve as a lightweight pre-selection add-on for existing seed-optimization pipelines, enabling additional gains. Across three benchmarks, extensive experiments show that ABSS enables consistent improvements in text-image alignment and visual quality for Stable Diffusion variants, as corroborated by human preference and alignment metrics.
>
---
#### [new 024] Physics-informed simulation framework for realistic sonar image generation and statistical validation
- **分类: cs.CV**

- **简介: 该论文提出ACOUSIM框架，用于生成真实声呐图像并进行统计验证。任务是提升合成数据的可靠性，解决真实数据获取成本高与验证不足的问题。通过物理建模生成图像，并与真实数据对比评估。**

- **链接: [https://arxiv.org/pdf/2605.19712](https://arxiv.org/pdf/2605.19712)**

> **作者:** Kamal Basha S; Athira Nambiar
>
> **摘要:** Synthetic sonar datasets offer a scalable alternative to costly real-world acquisition, yet their utility remains limited by the absence of rigorous quantitative validation. We present ACOUSIM (ACOustic SIMulation and Validation Platform), a physics-informed framework that evaluates the statistical alignment between synthetic and real sonar imagery without relying on generative models. A Gazebo-based environment generates sonar-like images by explicitly controlling seabed texture, illumination-driven shadowing, platform altitude, and noise. Realism is quantified against two public sonar datasets, SeabedObjects-KLSG-II and Sonar Common Target Detection (SCTD), using global intensity and local texture (LBP) distributions assessed via Kullback-Leibler divergence, Jensen-Shannon divergence, and Earth Mover's Distance. Results show strong texture alignment (KL < 0.07) across all classes, with plane-class intensity alignment outperforming ship-class due to shadow geometry complexity. ACOUSIM establishes a reproducible, distribution-level baseline for sim-to-real sonar evaluation and directly supports reliable dataset validation for underwater image analysis.
>
---
#### [new 025] UniRefiner: Teaching Pre-trained ViTs to Self-Dispose Dross via Contrastive Register
- **分类: cs.CV**

- **简介: 该论文属于视觉Transformer的改进任务，旨在解决spurious tokens影响空间表示的问题。通过引入UniRefiner框架，提升模型的语义对齐能力。**

- **链接: [https://arxiv.org/pdf/2605.19622](https://arxiv.org/pdf/2605.19622)**

> **作者:** Congpei Qiu; Zhaoyu Hu; Wei Ke; Zhuotao Tian; Yanhao Wu; Tong Zhang
>
> **备注:** CVPR 2026
>
> **摘要:** Representation learning with Vision Transformers (ViTs) has advanced rapidly, yet the utility of large-scale models in spatially sensitive tasks is hindered by spurious tokens. Prior efforts to mitigate this have been limited, often defining these artifacts narrowly, for example, as simple high-norm outliers. We argue that this scope is insufficient. For dense prediction tasks, we posit that any token failing to encode location-aligned semantics should be treated as a spurious artifact. This broader definition reveals a more complex problem, leading us to systematically categorize and characterize three fundamental types of spurious tokens that corrupt spatial representations. Based on this comprehensive diagnosis, we propose UniRefiner, a universal refinement framework that teaches pre-trained ViTs to self-dispose of these artifacts. UniRefiner uses contrastive registers to explicitly isolate and redistribute spurious tokens via a dual objective: (i) it aligns image tokens with filtered regular tokens to preserve semantics, and (ii) it aligns register tokens with detected spurious tokens to capture the spurious signals. Our method requires only a few epochs of fine-tuning on ~5k images to refine diverse ViTs, including massive models like EVA-CLIP-8B and InternViT-6B. Experiments demonstrate consistent and significant improvements: notably, the refined EVA-CLIP-8B achieves 51.9\% mIoU on ADE20K (+9.4\%), surpassing specialized vision models like DINOv2 (49.1\%), while zero-shot segmentation accuracy improves by up to 22\%. UniRefiner unlocks the latent spatial potential of existing large-scale foundation models, paving the way for their broader application.
>
---
#### [new 026] Selective, Regularized, and Calibrated: Harnessing Vision Foundation Models for Cross-Domain Few-Shot Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于跨域小样本语义分割任务，解决在领域迁移下少量标注样本的分割问题。提出HERA框架，通过选择、正则化和校准三个阶段，有效利用视觉基础模型，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2605.19340](https://arxiv.org/pdf/2605.19340)**

> **作者:** Junyuan Ma; Xunzhi Xiang; Wenbin Li; Qi Fan; Yang Gao
>
> **备注:** 20 pages, 11 figures, 13 tables. Accepted to CVPR 2026
>
> **摘要:** Vision foundation models (VFMs) have achieved strong performance across various vision tasks. However, it still remains challenging to apply VFMs for cross-domain few-shot segmentation (CD-FSS), which segments objects of novel classes under domain shifts using only a few labeled exemplars. The challenge is mainly driven by two factors: (1) limited labeled exemplars per novel class relative to the scale of VFM pre-training, making the model prone to overfitting during retraining, and (2) target-domain shifts underrepresented during pre-training, inducing cross-domain inconsistency and layer-wise sensitivity. To address these issues, we propose Hierarchical Exemplar Representation Adaptation (HERA), a three-stage select-regularize-calibrate VFM-based segmentation framework that learns effectively from limited labels and adapts to novel domains without source-data retraining. We first design Hierarchical Layer Selection (HLS) to adaptively identify the most informative VFM layer using a data-dependent Exemplar Transfer Risk (ETR) computed for each candidate layer. Then, Prior-Guided Regularization (PGR) regularizes interactions on the selected representation, yielding well-structured local signals for the subsequent stage. Furthermore, Pixelwise Adaptive Calibration (PAC) combines the selected representation with the refined interaction maps to calibrate pixel-wise predictions, producing consistent masks. Together, these stages form a hierarchical select-regularize-calibrate pipeline that guides frozen VFM features in new domains while fine-tuning less than 2.7% of parameters at test time. Extensive experiments show that HERA surpasses the state of the art by more than 4.1 mIoU across multiple CD-FSS benchmarks.
>
---
#### [new 027] LIFT and PLACE: A Simple, Stable, and Effective Knowledge Distillation Framework for Lightweight Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于知识蒸馏任务，旨在解决扩散模型中学生模型难以模仿教师模型复杂去噪过程的问题。提出LIFT和PLACE框架，实现稳定高效的模型压缩。**

- **链接: [https://arxiv.org/pdf/2605.19729](https://arxiv.org/pdf/2605.19729)**

> **作者:** Hyunsoo Han; Sangyeop Yeo; Jaejun Yoo
>
> **备注:** 15 pages, 11 figure, 9 tables, To appear in CVPR 2026
>
> **摘要:** We demonstrate that in knowledge distillation for diffusion models, the teacher network's highly complex denoising process - stemming from its substantially larger capacity - poses a significant challenge for the student model to faithfully mimic. To address this problem, we propose a coarse-to-fine distillation framework with LInear FiTtingbased distillation (LIFT) and Piecewise Local Adaptive Coefficient Estimation (PLACE). First, LIFT decomposes the objective into a "coarse" alignment and a "fine" refinement. The student is then trained on coarse alignment before proceeding to hard refinement. Second, PLACE extends LIFT to address spatially non-uniform errors by partitioning outputs into error-based groups, providing locally adaptive guidance. Our experiments show that LIFT and PLACE is effective across diffusion spaces (image/latent), backbones (U-Net/DiT), tasks (unconditional/conditional), datasets, and even extends to flow-based models such as MMDiT (SD3). Furthermore, under extreme compression with a 1.3M-parameter student (only 1.6% of the teacher), conventional KD fails to provide sufficient guidance for stable training, with FID scores often degrading to 50-200+, but our method remains stably convergent and achieves an FID of 15.73.
>
---
#### [new 028] Physics-in-the-Loop: A Hybrid Agentic Architecture for Validated CAD Engineering Design
- **分类: cs.CV**

- **简介: 该论文提出一种混合代理物理架构，用于验证的CAD工程设计。解决LLM缺乏物理理解的问题，通过嵌入工程工具实现物理验证的闭环设计流程。**

- **链接: [https://arxiv.org/pdf/2605.19717](https://arxiv.org/pdf/2605.19717)**

> **作者:** Elias Berger; Muhammad Usama; Jan Mehlstäubl; Bernhard Saske; Kristin Paetzold-Byhain
>
> **备注:** Accepted in IJCAI-ECAI 2026 (Special Track on AI4Tech)
>
> **摘要:** Large Language Models (LLMs) can generate Computer-Aided Design (CAD), yet lack physical comprehension required for reliable engineering design. Instead of attempting to implicitly learn physical laws from data, we propose a Hybrid Agentic-Physical Architecture that embeds validated knowledge-based engineering tools directly into the decision making loop of autonomous AI agents. In this framework, engineering design is formulated as a closed-loop, sequential decision making process guided by explicit physical verification. Based on a load case, dedicated agents iteratively plan, generate, evaluate, and revise engineering designs using knowledge-based tools as a feedback signal. We introduce a benchmark dataset and metrics for assessing functional validity in generative CAD. Our system generates more complex and physically verified designs, with a 4.2 increase in structural complexity and improving compile rate by 3.5% compared to similar agentic methods. The codebase, prompts and dataset will be made publicly available to support reproducibility and future research.
>
---
#### [new 029] CAD-Free Learning of Spacecraft Pose Estimators via NeRF-Based Augmentations
- **分类: cs.CV**

- **简介: 该论文属于航天器姿态估计任务，解决依赖CAD数据的问题。通过NeRF生成增强图像，实现少量真实图像训练高精度姿态估计器。**

- **链接: [https://arxiv.org/pdf/2605.19649](https://arxiv.org/pdf/2605.19649)**

> **作者:** Antoine Legrand; Renaud Detry; Christophe De Vleeschouwer
>
> **备注:** (under review)
>
> **摘要:** Spacecraft pose estimation networks require tens of thousands of CAD-rendered images to be trained. This reliance on synthetic CAD data (i) limits applicability to targets with reliable geometry prior, excluding uncooperative or poorly documented spacecraft, and (ii) causes poor generalization to real on-orbit conditions due to unrealistic illumination and material appearance. This paper introduces a NeRF-based image augmentation method that enables the learning of spacecraft pose estimators from only a few tens to a few hundreds of images. The method learns a Neural Radiance Field of the target and generates a large, diverse dataset through geometrically-consistent viewpoint and appearance augmentation. This augmented dataset enables the training of accurate target-specific pose estimators without requiring a CAD model or large synthetic datasets. Experiments show that our approach supports the training of accurate pose estimators from only 25 to 400 realistic images, even under severe illumination variations. When applied on large CAD-based synthetic datasets, the NeRF-based augmentation also enhances out-of-domain generalization, yielding improved robustness to real on-orbit conditions.
>
---
#### [new 030] GeoMamba: A Geometry-driven MambaVision Framework and Dataset for Fine-grained Optical-SAR Object Retrieval
- **分类: cs.CV**

- **简介: 该论文属于跨模态细粒度目标检索任务，旨在解决光学与SAR图像不匹配下的检索难题。提出GeoMamba框架，结合几何约束提升跨模态表示学习效果。**

- **链接: [https://arxiv.org/pdf/2605.19734](https://arxiv.org/pdf/2605.19734)**

> **作者:** Tiantong Fang; Xiuwei Wang; Jing Xiao; Wujie Zhou; Liang Liao; Mi Wang
>
> **摘要:** Multi-source remote sensing enables complementary observation of ground objects, while cross-modal fine-grained object retrieval remains challenging, especially under unaligned optical and SAR conditions. Unlike conventional retrieval settings that rely on paired or spatially aligned samples, practical optical-SAR retrieval is affected by substantial modality discrepancy, speckle noise, and structural inconsistency, which limit robust cross-modal representation learning. To address this problem, we propose GeoMamba, a geometry-driven framework tailored for optical-SAR fine-grained retrieval. Specifically, GeoMamba introduces a Geometric Feature Injection (GFI) module that enhances cross-modal feature interaction and incorporates structural priors, thereby improving the robustness of SAR representations and promoting geometry-consistent feature learning. In addition, a Geometric Consistency Constraint (GCC) module, together with a Deep Supervision (DS) strategy, imposes hierarchical geometric constraints using classical operators, which helps preserve informative object structures during representation learning. We further construct a new dataset, FGOS-as, containing 11 aerospace and maritime categories for evaluating unaligned cross-modal fine-grained object retrieval in realistic remote sensing scenarios. Extensive experiments on FGOS-as demonstrate that GeoMamba outperforms existing methods, achieving 63.3% mAP and 77.0% Rank-1 accuracy in all-to-all retrieval setting.
>
---
#### [new 031] PrAda: Few-Shot Visual Adaptation for Text-Prompted Segmentation
- **分类: cs.CV**

- **简介: 该论文属于文本提示分割任务，解决目标域适应问题。通过原型适配方法PrAda，提升模型在新域的性能，同时保持零样本能力。**

- **链接: [https://arxiv.org/pdf/2605.19623](https://arxiv.org/pdf/2605.19623)**

> **作者:** Gabriele Rosi; Fabio Cermelli; Carlo Masone; Barbara Caputo
>
> **备注:** CVPR 2026 Findings. Code: this https URL
>
> **摘要:** Segmenting images is critical for visual understanding but demands extensive pixel-level annotations. Foundational models have enabled new paradigms for predicting new classes guided by textual prompts, without annotations from the target domain. Yet, on specialized target domains, far from the original pre-training, their performance degrades. We study the errors of existing methods under such domain-shift, finding that misclassification rather than mask generation is the main culprit. To address this, we introduce the novel problem of Few-Shot Visual Adaptation for text-prompted Segmentation. This kind of adaptation has been largely studied for image classification, but it remains unexplored for segmentation. We tackle this task with Prototype Adaptation (PrAda), a novel, parameter-efficient method that adapts a frozen text-prompted segmentation model. Our approach learns class-specific prototypes by combining fine-grained pixel features and high-level transformer representations, which are then fused with the original text-based predictions through a learned importance factor. This preserves the model's zero-shot potential while enabling strong adaptation to new domains. Experiments across semantic, instance, and panoptic segmentation on five benchmarks demonstrate that PrAda yields significant improvements over state-of-the-art and proposed baselines.
>
---
#### [new 032] RE-VLM: Event-Augmented Vision-Language Model for Scene Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出RE-VLM，一种结合RGB图像与事件流的视觉-语言模型，解决复杂环境下场景理解问题。通过双流架构和数据增强方法，提升模型在恶劣条件下的表现。**

- **链接: [https://arxiv.org/pdf/2605.19329](https://arxiv.org/pdf/2605.19329)**

> **作者:** Hanqing Liu; Mingjie Liu; Luoping Cui; Endian Lin; Donghong Jiang; Chuang Zhu
>
> **备注:** 10 pages, 6 figures, 6 tables
>
> **摘要:** Conventional vision-language models (VLMs) struggle to interpret scenes captured under adverse conditions (e.g., low light, high dynamic range, or fast motion) because standard RGB images degrade in such environments. Event cameras provide a complementary modality: they asynchronously record per-pixel brightness changes with high temporal resolution and wide dynamic range, preserving motion cues where frames fail. We propose RE-VLM, the first dual-stream vision-language model that jointly leverages RGB images and event streams for robust scene understanding across both normal and challenging conditions. RE-VLM employs parallel RGB and event encoders together with a progressive training strategy that aligns heterogeneous visual features with language. To address the scarcity of RGB-Event-Text supervision, we further propose a graph-driven pipeline that converts synchronized RGB-Event streams into verifiable scene graphs, from which we synthesize captions and question-answer (QA) pairs. To develop and evaluate RE-VLM, we construct two datasets: PEOD-Chat, targeting illumination-challenged scenes, and RGBE-Chat, covering diverse scenarios. On captioning and VQA benchmarks, RE-VLM consistently outperforms state-of-the-art RGB-only and event-only models with comparable parameter counts, with particularly large gains under challenging conditions. These results demonstrate the effectiveness of event-augmented VLMs in achieving robust vision-language understanding across a wide range of real-world environments. Code and datasets are available at this https URL.
>
---
#### [new 033] Eyes on VLM: Benchmarking Gaze Following and Social Gaze Prediction in Vision Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的 gaze 理解任务，旨在评估 VLM 在眼神跟随和社会注视预测中的表现，揭示其在精确理解人类注视方面的能力不足。**

- **链接: [https://arxiv.org/pdf/2605.19859](https://arxiv.org/pdf/2605.19859)**

> **作者:** Hengfei Wang; Anshul Gupta; Pierre Vuillecard; Jean-Marc Odobez
>
> **备注:** Under review
>
> **摘要:** Vision-language models (VLMs) have rapidly evolved into general-purpose multimodal reasoners with strong zero-shot generalization. In this context, VLMs could greatly benefit the analysis of human gaze and attention, a central task in human behavior understanding that requires reasoning about the physical scene as well as the activity, interactions, and social context. However, the extent to which VLMs can reliably understand human gaze and related attentional behaviors remains largely unexplored. In this work, we present EyeVLM, a systematic evaluation framework for gaze understanding in VLMs across two complementary dimensions: tasks and models. To assess gaze understanding capabilities, we focus on two core tasks. The first, gaze following, i.e., predicting the 2D location where a person is looking, has a geometric and visual processing focus, requiring a precise understanding of the human face, attention direction, 3D scene structure, and spatial grounding of attended targets. The second, social gaze prediction, requires social and relational reasoning over multi-person interactions (e.g., mutual gaze and shared attention), and may benefit more from the LLM semantic reasoning capabilities within VLMs. Regarding models, EyeVLM evaluates these tasks in two ways: a zero-shot setting with a diverse set of state-of-the-art open- and closed-source VLMs, exploring different prompting strategies; and a fine-tuning approach based on task-specific QA pairs, studying the impact of model scale and data scale. As benchmarks, we rely on existing gaze understanding datasets and perform a systematic comparison with state-of-the-art purely visual models. Overall, our results show that current VLMs lack precise gaze understanding capabilities. While standard training helps reduce the gap with visual models, significant improvements are still needed.
>
---
#### [new 034] StruMPL: Multi-task Dense Regression under Disjoint Partial Supervision and MNAR Labels
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出StruMPL模型，解决在不完全监督和MNAR标签下估计森林生物量的任务，通过多任务回归和物理约束提升预测精度。**

- **链接: [https://arxiv.org/pdf/2605.19931](https://arxiv.org/pdf/2605.19931)**

> **作者:** Reza M. Asiyabi; Juan Alberto Molina-Valero; SEOSAW Partnership; Steven Hancock; Casey M. Ryan
>
> **备注:** 10 pages with 3 figures and 4 tables, References and Appendix 12 pages with 1 figure and 4 tables
>
> **摘要:** Estimating forest aboveground biomass (AGB) from Earth observation combines two structurally incompatible label sources: spaceborne lidar provides canopy structure at millions of locations but no biomass estimate, and ground-based plots provide biomass at thousands of biased locations but no metrics of structure. No single training sample carries labels for all target variables, plot labels are missing not at random (MNAR), and biomass is linked to the structural variables by known but biome-specific allometric laws. We formalise this as multi-task dense regression under heterogeneous disjoint partial supervision with MNAR labels and inter-task physical constraints, and propose StruMPL to address it jointly. A shared encoder feeds per-variable regression, imputation, and propensity heads for spatial MNAR correction, and a learnable physics module that evaluates the inter-task constraint on the model's own predictions at every pixel. The supervised loss uses an Augmented IPW (AIPW) pseudo-outcome with stop-gradients on the propensity and on the imputation baseline; we show analytically and empirically that both are necessary for joint optimisation to recover IPW-weighted stationary points while keeping the loss bounded. On two ecologically distinct biomes, StruMPL outperforms ablation variants and the closest published method on AGB RMSE and bias, with a stratified analysis showing AIPW reduces high-AGB bias by ~54%.
>
---
#### [new 035] CaptchaMind: Training CAPTCHA Solvers via Reinforcement Learning with Explicit Reasoning Supervision
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于CAPTCHA破解任务，旨在解决传统方法在复杂CAPTCHA上的失效问题。作者构建了首个大规模CAPTCHA基准数据集，并提出基于强化学习的求解器CaptchaMind，显著提升破解成功率。**

- **链接: [https://arxiv.org/pdf/2605.19538](https://arxiv.org/pdf/2605.19538)**

> **作者:** Pengcheng Wang; Haoxiang Liu; Yang Dai; Xiangxiang Zeng; Guanhua Chen; Baotian Hu; Longyue Wang; Weihua Luo
>
> **备注:** 17 pages, 12 figures
>
> **摘要:** CAPTCHAs are widely deployed as human verification mechanisms and frequently block intelligent agents from completing end-to-end automation in real-world web environments. Solving modern CAPTCHAs requires robust multi-step visual reasoning and interaction capabilities, yet training-based approaches have remained absent due to the lack of large-scale training data and process-level annotations. We introduce CaptchaBench, the first CAPTCHA benchmark designed to support large-scale training, comprising 16,000 programmatically generated samples across eight task categories with detailed region and process-level annotations. Systematic evaluation on CaptchaBench reveals that existing methods fail consistently on tasks requiring fine-grained visual detail capture and region-level comparison. We therefore present CaptchaMind, an RL-based solver trained with explicit reasoning process supervision, achieving 82.9% average success rate across eight tasks and 71.0% on real-world instances, substantially outperforming all existing methods without closed-source APIs.
>
---
#### [new 036] CRAFT: Critic-Refined Adaptive Key-Frame Targeting for Multimodal Video Question Answering
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CRAFT系统，解决多模态视频问答中的证据定位与引用问题，通过关键帧选择、ASR和批评循环提升答案准确性。**

- **链接: [https://arxiv.org/pdf/2605.19075](https://arxiv.org/pdf/2605.19075)**

> **作者:** Mahesh Bhosale; Abdul Wasi; Vishvesh Trivedi; Pengyu Yan; Akhil Gorugantu; David Doermann
>
> **备注:** Accepted at ACL 2026 Multimodal Augmented Generation via MultimodAl Retrieval Workshop
>
> **摘要:** Grounded multi-video question answering over real-world news events requires systems to surface query-relevant evidence across heterogeneous video archives while attributing every claim to its supporting source. We introduce CRAFT (Critic-Refined Adaptive Key-Frame Targeting), a query-conditioned pipeline that combines dynamic keyframe selection, per-video ASR with multilingual fallback, and a hybrid critic loop to iteratively verify and repair claims before consolidation. The pipeline integrates UNLI temporal entailment, DeBERTa-v3 cross-claim screening, and a Llama-3.2-3B adjudicator, with a final citation-merging stage that emits each fact once with all supporting source identifiers. On MAGMaR 2026, CRAFT achieves the best overall average (0.739), reference recall (0.810), and citation F1 (0.635). We further evaluate on a MAGMaR-style conversion of WikiVideo with 52 non-overlapping event queries, where CRAFT also performs strongly (0.823 Avg), showing that its claim-centric evidence aggregation generalizes beyond MAGMaR. Ablations show that atomic claims, ASR, and the critic loop drive the main gains over the vanilla query-conditioned baseline. Code and implementation details are publicly available at this https URL.
>
---
#### [new 037] Feed-Forward Gaussian Splatting from Sparse Aerial Views
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决稀疏航拍视角下城市场景重建问题。提出AnyCity框架，通过观测支撑的几何和条件生成策略，提升重建一致性与质量。**

- **链接: [https://arxiv.org/pdf/2605.19949](https://arxiv.org/pdf/2605.19949)**

> **作者:** Dongli Wu; Zhuoxiao Li; Tongyan Hua; Yinrui Ren; Xiaobao Wei; Rongjun Qin; Wufan Zhao
>
> **摘要:** Reconstructing large-scale urban scenes from sparse aerial views is a crucial yet challenging task. Due to biased top-down and shallow-oblique camera poses, sparse aerial captures exhibit strong evidence imbalance: roofs and open regions are repeatedly observed, while facades, distant buildings, and occluded structures receive little multi-view support. Existing feed-forward 3D Gaussian Splatting methods directly regress a deterministic representation from sparse inputs, but this often leads to ghosting, melted facades, and stretched textures. Recent pseudo-view and video-based generative reconstruction methods use additional supervision or generative priors. However, they often lack a clear separation between observed geometry and prior-driven content, which can lead to plausible but inconsistent structures. We propose AnyCity, an observation-grounded generative reconstruction framework for sparse aerial urban scenes. AnyCity first predicts an observation-supported geometry latent to anchor reliable structures, and then uses scaffold-conditioned aerial completion tokens to predict a gated residual update for weakly constrained content before Gaussian decoding. During training, dense-to-sparse distillation transfers structural cues from dense-view reconstruction, while an aerial-adapted video diffusion prior provides fine-grained urban appearance cues through gated token conditioning. Observation-preserving objectives keep the refined representation consistent with input-supported geometry. At inference time, AnyCity reconstructs the final 3D Gaussian scene from sparse aerial views in a single feed-forward pass, achieving coherent urban novel-view synthesis with second-level inference. Experiments on synthetic, aerial-domain, UAV-textured, and real-world scenes show consistent improvements over feed-forward baselines.
>
---
#### [new 038] Replacement Learning: Training Neural Networks with Fewer Parameters
- **分类: cs.CV**

- **简介: 该论文提出Replacement Learning（RepL），用于减少神经网络训练中的参数和计算成本。针对深度模型训练效率低的问题，通过替换部分块并引入轻量层来保持性能，适用于分类、检测和分割任务。**

- **链接: [https://arxiv.org/pdf/2605.19533](https://arxiv.org/pdf/2605.19533)**

> **作者:** Yuming Zhang; Peizhe Wang; Tianyang Han; Hengyu Shi; Junhao Su; Dongzhi Guan; Jiabin Liu; Jiaji Wang
>
> **备注:** 16pages
>
> **摘要:** End-to-end training with full-depth backpropagation remains the dominant paradigm for optimizing deep neural networks, but its efficiency deteriorates as models grow deeper. Since every block must be executed and differentiated under a single global objective, full-depth BP introduces substantial parameter redundancy, activation-memory cost, and training latency, especially when neighboring layers exhibit highly correlated learning patterns. Directly skipping or removing layers can reduce cost, but often weakens representation capacity or requires architecture-specific reuse designs. In this paper, we propose Replacement Learning (RepL), a training-time paradigm that reduces full-depth redundancy by replacing selected blocks rather than simply discarding them. For each removed block, RepL inserts a lightweight computing layer that synthesizes a surrogate operator from the parameters of its adjacent preceding and succeeding blocks through a learnable transformation, and applies the synthesized operator to the preceding activation. In this way, RepL preserves local contextual continuity while avoiding unnecessary full-layer computation. We instantiate RepL for CNNs and ViTs with tailored parameter-fusion blocks that handle convolutional channels, feature resolutions, and transformer submodules. Extensive experiments on CIFAR-10, SVHN, STL-10, ImageNet, COCO, and CityScapes show that RepL reduces trainable parameters, GPU memory usage, and training time while matching or surpassing standard end-to-end training across classification, detection, and segmentation. Additional results on WikiText-2, transfer learning, inference throughput, checkpointing, stochastic depth, and INT8 quantization further demonstrate its generality and compatibility.
>
---
#### [new 039] MetaRA: Metamorphic Robustness Assessment for Multimodal Large Language Model-based Visual Question Answering Systems
- **分类: cs.CV**

- **简介: 该论文属于视觉问答任务，旨在解决MLLMs在鲁棒性评估上的不足。提出MetaRA框架，通过元变换关系检测模型漏洞，提升评估的全面性。**

- **链接: [https://arxiv.org/pdf/2605.19307](https://arxiv.org/pdf/2605.19307)**

> **作者:** Quanxing Xu; Yuhao Tian; Ling Zhou; Xian Zhong; Xiaohua Huang; Rubing Huang; Chia-Wen Lin
>
> **摘要:** Visual Question Answering (VQA), as the representative multimodal task, serves as a key benchmark for evaluating the reasoning capabilities of Multimodal Large Language Models (MLLMs). However, existing evaluations largely rely on static datasets and accuracy-based metrics, which fail to capture robustness, consistency, and generalization. Inspired by Metamorphic Testing (MT), we propose Metamorphic Robustness Assessment (MetaRA), a testing framework that employs Metamorphic Relations (MRs) to systematically probe vulnerabilities in MLLM-based VQA systems. MetaRA generates controlled variations of image-question inputs based on specific MRs and evaluates models across diverse conditions. Applying MetaRA to multiple MLLM-based VQA models across different tasks reveals nuanced failure patterns, including sensitivity to linguistic perturbations, over-reliance on superficial visual cues, and deeper weaknesses in multimodal reasoning. Experimental results demonstrate that MetaRA provides richer diagnostic insights than conventional accuracy metrics, exposing failure modes that remain hidden under standard benchmarks. Overall, this work highlights the need for systematic robustness evaluation in VQA and positions metamorphic assessment as a scalable, model-agnostic approach toward trustworthy multimodal AI.
>
---
#### [new 040] Scalable, Energy-Efficient Optical-Neural Architecture for Multiplexed Deepfake Video Detection
- **分类: cs.CV; cs.LG; cs.NE; physics.app-ph; physics.optics**

- **简介: 该论文属于深度伪造视频检测任务，旨在解决传统方法计算成本高、效率低的问题。提出一种混合数字-光学架构，实现高效、准确的检测。**

- **链接: [https://arxiv.org/pdf/2605.19360](https://arxiv.org/pdf/2605.19360)**

> **作者:** Parnian Ghapandar Kashani; Shiqi Chen; Aydogan Ozcan
>
> **备注:** 30 Pages, 8 Figures
>
> **摘要:** The rapid proliferation of AI-generated visual media has created an urgent need for efficient, trustworthy deepfake detection systems. However, existing deep learning-based detection methods rely on computationally intensive and energy-demanding inference algorithms, limiting their scalability. Here, we present a hybrid digital-analog deepfake video detection framework that combines a lightweight digital front-end with a spatially multiplexed optical decoding back-end for massively parallel analog inference through a programmable spatial light modulator. By simultaneously processing 15 or more video streams within a single optical propagation pass, the system enables high-throughput and accurate video-level authenticity prediction at reduced computational cost compared with purely digital methods. We validated this hybrid deepfake video processor using different datasets spanning classical face-swapping, real-world deepfake recordings, and fully AI-generated videos. Using a spatially multiplexed experimental set-up operating in the visible spectrum, we achieved average deepfake detection accuracy, sensitivity and specificity of 97.79%, 99.86% and 95.72%, respectively, on the Celeb-DF video dataset with 15 videos tested in parallel in a single optical pass per inference. The multiplexed optical decoder also demonstrates resilience against various types of video degradation, noise, compression, experimental misalignments and black-box adversarial attacks. Our results show that integrating optical computation into AI inference enables simultaneous gains in throughput, energy efficiency, and adversarial robustness - three properties that are difficult to achieve together in purely digital systems.
>
---
#### [new 041] WoundFormer: Multi-Scale Spatial Feature Fusion for Multi-Class Wound Tissue Segmentation
- **分类: cs.CV**

- **简介: 该论文属于多类伤口组织分割任务，旨在解决慢性伤口中组织多样性带来的分割难题。提出WoundFormer框架，通过改进的空间特征融合提升分割精度。**

- **链接: [https://arxiv.org/pdf/2605.19868](https://arxiv.org/pdf/2605.19868)**

> **作者:** Muhammad Ashad Kabir; Rabin Dulal
>
> **备注:** 10 pages
>
> **摘要:** Chronic wounds such as diabetic foot ulcers and pressure injuries require accurate tissue-level assessment to guide treatment planning and monitor healing progression. While deep learning methods have advanced automated wound analysis, most existing approaches focus on binary segmentation and inadequately model heterogeneous tissue composition due to high intra-class variability and limited annotated data. Multi-class wound tissue segmentation, therefore, remains a challenging and clinically relevant problem. We propose WoundFormer, a transformer-based framework that enhances hierarchical spatial feature fusion for multi-class wound tissue segmentation. Specifically, we replace the standard SegFormer decoder with a spatially-preserving multi-scale aggregation head that maintains feature topology during cross-scale integration and strengthens contextual interactions through convolutional fusion. This design improves boundary localization and discrimination between visually similar tissue categories while preserving transformer efficiency. We evaluate WoundFormer on the WoundTissueSeg dataset (147 images, six tissue classes) and a second benchmark (DFUTissue dataset). The proposed method achieves an overall Dice score of 81.9%, outperforming strong CNN- and transformer-based baselines by up to 4.3 Dice points on the WoundTissueSeg benchmark, with consistent improvements across minority tissue classes. These results indicate that explicit modeling of hierarchical spatial interactions enhances transformer representations for heterogeneous wound tissue segmentation and supports more reliable quantitative wound assessment.
>
---
#### [new 042] SetCon: Towards Open-Ended Referring Segmentation via Set-Level Concept Prediction
- **分类: cs.CV**

- **简介: 该论文属于 referring segmentation 任务，解决多目标、开放集场景下的分割问题。提出 SetCon，通过集合级概念预测实现更准确的多目标分割。**

- **链接: [https://arxiv.org/pdf/2605.20110](https://arxiv.org/pdf/2605.20110)**

> **作者:** Zhixiong Zhang; Yizhuo Li; Shuangrui Ding; Yuhang Zang; Shengyuan Ding; Long Xing; Yibin Wang; Qiaosheng Zhang; Jiaqi Wang
>
> **摘要:** Referring segmentation grounds natural-language queries to pixel-level masks, but extending it to complex scenarios with multiple instances, cross-category groups, or open-ended target sets remains challenging. Previous Large Vision Language Model (LVLM)-based methods represent referred targets with one or more special tokens sequentially, treating multiple targets as separate outputs rather than a coherent set and offering little incentive to capture set-level properties such as completeness and mutual exclusivity. We reformulate open-ended referring segmentation as explicit set-level concept prediction and propose Set-Concept Segmentation (SetCon), which uses LVLM-generated natural-language concepts, instead of segmentation-specific tokens, as semantic conditions for joint mask-set decoding. A hierarchical semantic decomposition first predicts a shared set-level concept defining the target scope and then refines it into fine-grained concept groups aligned with target subsets. To support this, a two-stage annotation pipeline augments existing reasoning segmentation datasets with hierarchical semantic supervision (236k samples, 784k concept phrases). SetCon achieves state-of-the-art results on image benchmarks (+3.3 gIoU on gRefCOCO, +12.1 gIoU on MUSE), with margins that grow as the number of referred targets increases. The concept interface also transfers to video under a detect-and-track setting, yielding new state-of-the-art results on seven referring video benchmarks, including +10.9 J&F on MeViS and +12.4 J&F on Ref-SeCVOS.
>
---
#### [new 043] InterLight: Leveraging Intrinsic Illumination Priors for Low-Light Image Enhancement
- **分类: cs.CV**

- **简介: 该论文属于低光图像增强任务，旨在解决光照不足导致的对比度低、细节丢失和噪声问题。提出InterLight框架，利用内在光照先验提升增强效果。**

- **链接: [https://arxiv.org/pdf/2605.19982](https://arxiv.org/pdf/2605.19982)**

> **作者:** Ziqi Wang; Xu Zhang; Laibin Chang; Shi Chen; Jiaqi Ma; Huan Zhang
>
> **备注:** Accepted by IJCAI 2026. Code: this https URL
>
> **摘要:** Low-Light Image Enhancement (LLIE) has long been a challenging problem in low-level vision, as insufficient illumination often leads to low contrast, detail loss, and noise. Recent studies show that deep learning-based Retinex theory can effectively decouple illumination and reflectance. However, existing methods frequently suffer from over-enhancement or color distortion, and often assume uniform noise or ideal lighting. To address these limitations, we propose InterLight, a novel framework that systematically excavates and operationalizes intrinsic illumination priors for this http URL core insight is that robust enhancement requires not just estimating illumination, but constructing an illumination-aware pipeline. We first inject sensor-level illumination-response priors via physics-guided augmentation, then represent the degradation through adaptive prompts conditioned on the scene's latent illumination state. This explicit representation directly guides a luminance-gated intrinsic memory mechanism to selectively compensate for information loss, prioritizing reconstruction in dark regions while preserving fidelity in bright ones. Finally, the entire process is regularized by a self-supervised consistency objective that distills illumination-invariant features. By deeply exploiting intrinsic illumination priors, our method achieves clearer textures and more visually coherent enhancement results. Extensive experiments across multiple benchmarks demonstrate the effectiveness of our approach. Code is available at: this https URL.
>
---
#### [new 044] Knowing When Not to Predict: Self Supervised Learning and Abstention for Safer DR Screening
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分类任务，旨在提升糖尿病视网膜病变筛查的安全性。研究解决模型在不确定时无法主动拒绝预测的问题，通过自监督学习和置信度校准实现更可靠的预测。**

- **链接: [https://arxiv.org/pdf/2605.19133](https://arxiv.org/pdf/2605.19133)**

> **作者:** Muskaan Chopra; Lorenz Sparrenberg; Jan H. Terheyden; Rafet Sifa
>
> **备注:** Accepted at IJCAI 2026
>
> **摘要:** Self-supervised learning (SSL) is now a standard way to pretrain medical image models, but performance is still mostly judged by downstream accuracy. For safety-critical screening tasks such as diabetic retinopathy grading, this is not enough: a model must also know when its predictions are unreliable and defer uncertain cases for clinical review. In this work, we examine how the length of SSL pretraining influences calibrated confidence and confidence-based abstention. We evaluate multiple SSL checkpoints under a fixed fine-tuning protocol and assess calibrated confidence, coverage, selective accuracy, and selective macro-F1. Across datasets and data regimes, SSL pretraining improves selective prediction compared to training from scratch. Unlike prior SSL studies that primarily evaluate downstream accuracy or AUROC, we analyze how SSL pretraining duration influences confidence behavior under calibrated confidence-based abstention. However, once accuracy saturates, selective performance can still change markedly across checkpoints, and longer pretraining does not consistently improve reliability. These results underscore the importance of abstention-aware evaluation and suggest that pretraining length should be treated as an important reliability-related design choice rather than only a computational detail. Code is available at GitHub.
>
---
#### [new 045] AffectVerse: Emotional World Models for Multimodal Affective Computing
- **分类: cs.CV**

- **简介: 该论文提出AffectVerse，用于多模态情感计算任务，解决情感动态建模问题。通过引入情感世界模块，实现短时情感预测与推理。**

- **链接: [https://arxiv.org/pdf/2605.19950](https://arxiv.org/pdf/2605.19950)**

> **作者:** Bo Zhao; Fanghua Ye; Yixin Ji; Sicheng Zhao; Xiaojiang Peng; Zitong YU
>
> **摘要:** Humans infer emotions by integrating observed multimodal cues with expectations about how affective states may unfold. Existing multimodal large language models (MLLMs), however, often treat emotion recognition as static fusion over complete audiovisual-text inputs, leaving affective dynamics implicit. We propose AffectVerse, a Qwen2.5-Omni-based model equipped with an Emotion World Module (EWM), an action-free representation-level module for short-horizon latent affective prediction. \rev{EWM contains three modules: 1) Cross-Modal Temporal Imagination predicts future video/audio representations from past tokens with multi-step rollout. 2) MAMA(Modality-Aware Multi-step Attention) Belief Aggregation compresses imagined tokens into modality-aware belief tokens. 3) Belief Injection inserts these belief tokens into the LLM for affective reasoning.} AffectVerse uses future prediction as a past-conditioned self-supervised signal: it does not replace modeling observed history or require unseen signals at inference, but forces the current belief state to encode transition cues that are predictive of subsequent affective change. Across nine benchmarks, AffectVerse improves at least 2.57\% over other models, while controlled ablations show additive gains from temporal imagination, cross-modal rollout, and belief aggregation. These results suggest predictive belief-state modeling is a practical alternative for affective computing.
>
---
#### [new 046] Spectral Integrated Gradients for Coarse-to-Fine Feature Attribution
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于特征归因任务，旨在解决传统集成梯度方法因路径选择导致的噪声问题。通过SVD构建积分路径，实现从粗到细的特征激活，提升归因质量。**

- **链接: [https://arxiv.org/pdf/2605.19607](https://arxiv.org/pdf/2605.19607)**

> **作者:** Soyeon Kim; Seongwoo Lim; Kyowoon Lee; Jaesik Choi
>
> **备注:** 21 pages, 13 figures, 9 tables. Accepted to ACM KDD 2026; includes appendix
>
> **摘要:** Integrated Gradients (IG) is a widely adopted feature attribution method that satisfies desirable axiomatic properties. However, the choice of integration path significantly affects the quality of attributions, and the standard straight-line path introduces all input features simultaneously, often accumulating noisy gradients along the way. To address this limitation, we propose Spectral Integrated Gradients, which constructs integration paths based on singular value decomposition (SVD) of the baseline-to-input difference. By progressively activating singular components from largest to smallest, SIG introduces global structure before fine-grained details, naturally following a coarse-to-fine progression. Through extensive evaluation across diverse image classification datasets, we demonstrate that SIG produces cleaner attribution maps with reduced noise and achieves improved quantitative performance compared to existing path-based attribution methods. Our code is available at this https URL.
>
---
#### [new 047] Structuring Open-Ended NAS: Semi-Automated Design Knowledge Structuring with LLMs for Efficient Neural Architecture Search
- **分类: cs.CV**

- **简介: 该论文属于神经网络架构搜索任务，旨在解决传统NAS搜索空间受限和LLM辅助方法效率低的问题。通过半自动结构化设计知识，构建多样化搜索空间，并引入FairNAD算法提升搜索效率与性能。**

- **链接: [https://arxiv.org/pdf/2605.19247](https://arxiv.org/pdf/2605.19247)**

> **作者:** Yuiko Sakuma; Masakazu Yoshimura; Marcel Gröpl; Zitang Sun; Junji Otsuka; Atsushi Irie; Takeshi Ohashi
>
> **备注:** 42 pages
>
> **摘要:** Current neural architecture search (NAS) methods are often limited by their predefined, restrictive search spaces. While recent large language model (LLM)-assisted NAS methods enable open-ended search spaces, they often suffer from inefficient exploration due to biased or low-quality design ideas. To address these issues, we propose to semi-automatically structure model design knowledge to guide the search process. Our approach first defines a high-level structural template of architectural attributes. An LLM then populates this template by analyzing papers, creating a rich and diverse search space that embodies this structured design knowledge. To efficiently explore this vast space, we introduce FairNAD, using a multi-type mutation that enables broad exploration through mutation with fair idea sampling, Pareto-aware mutation, LLM-driven iterative mutation, and a fine-grained feedback loop. We demonstrate the effectiveness of FairNAD in discovering high-performing architectures that yield 0.84, 2.17, and 2.35 points improvement on CIFAR-10, CIFAR-100, and ImageNet16-120, respectively, compared to current state-of-the-art methods.
>
---
#### [new 048] A Systematic Failure Analysis of Vision Foundation Models for Open Set Iris Presentation Attack Detection
- **分类: cs.CV**

- **简介: 该论文属于生物特征识别中的虹膜防攻击检测任务，研究基础模型在开放集条件下的失败原因，分析其在不同攻击工具和光谱转换下的泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.19020](https://arxiv.org/pdf/2605.19020)**

> **作者:** Rahul Anand; Siddharth Singh; Dileep A D; Mahadeva Prasanna; Raghavendra Ramachandra
>
> **摘要:** Vision foundation models have demonstrated strong transferability across diverse visual recognition tasks and are increasingly considered for biometric applications. Their suitability for iris Presentation Attack Detection (PAD), particularly under realistic open-set operating conditions, remains insufficiently examined. This work presents a systematic failure analysis of general-purpose vision foundation models for open-set iris PAD using periocular imagery. Five representative foundation models are evaluated under three open-set protocols that explicitly separate different sources of distribution shift: unseen Presentation Attack Instruments (PAIs), unseen datasets captured with different sensors and cross-spectral transfer from near-infrared (NIR) to visible spectrum (VIS) imagery. Both frozen feature representations and parameter-efficient task adaptation using Low-Rank Adaptation (LoRA) are assessed within a unified experimental framework. The results indicate that foundation models can transfer across datasets with similar sensing characteristics, but fail to generalise reliably to unseen attack instruments and degrade sharply under cross-spectral evaluation. While LoRA improves performance in certain cross-dataset settings, it frequently amplifies failure under attack-level and spectral shifts. Additional validation experiments using segmented iris inputs, full backbone fine-tuning, joint cross-dataset and cross-PAI shifts, and reverse VIS to NIR transfer further confirm that these failures are not simply artefacts of periocular input, weak adaptation, or one-directional spectral evaluation. These findings show that strong closed-set or cross-dataset performance should not be treated as evidence of robust open-set security, and highlight the need for PAD representations that maintain sensitivity to presentation artefacts while remaining stable under realistic deployment variation.
>
---
#### [new 049] Bézier Degradation Modeling for LiDAR-based Human Motion Capture
- **分类: cs.CV**

- **简介: 该论文属于LiDAR人体动作捕捉任务，解决输入不稳定和遮挡导致的运动重建问题。提出BMLiCap框架，利用贝塞尔曲线建模运动，提升精度与时间连贯性。**

- **链接: [https://arxiv.org/pdf/2605.19620](https://arxiv.org/pdf/2605.19620)**

> **作者:** Xiaoqi An; Lin Zhao; Jun Li; Chen Gong; Jian Yang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** LiDAR-based 3D human motion capture has broad applications in fields such as autonomous driving and robotics, where accurate motion reconstruction is crucial. However, existing methods often struggle with unstable inputs and severe occlusions, leading to jittery or even failed pose predictions. To address these challenges, we propose BMLiCap, a coarse-to-fine framework that models motion using temporally compressible Bézier curves. By reducing control points through a trajectory-preserving strategy, we obtain a coherent and learning-friendly motion representation. To reconstruct human actions from LiDAR point-cloud cues, we design a progressive motion-reconstruction module. Specifically, a Time-scale Motion Transformer (TMT) is introduced to predict motion curves at multiple temporal scales, and a Multi-level Motion Aggregator (MMA) is utilized to adaptively fuse the multi-scale curves to recover detailed, temporally coherent poses, effectively bridging observation gaps caused by occlusions and noise. Across four mainstream benchmarks LiDARHuman26M, FreeMotion, NoiseMotion, and SLOPER4D, BMLiCap achieves state-of-the-art accuracy and temporal continuity in complex scenes, demonstrating its ability to compensate for severe occlusions and reduce prediction jitter.
>
---
#### [new 050] MetaEarth-MM: Unified Multimodal Remote Sensing Image Generation with Scene-centered Joint Modeling
- **分类: cs.CV**

- **简介: 该论文属于多模态遥感图像生成任务，解决数据稀缺和跨模态翻译问题。提出MetaEarth-MM模型，通过场景中心建模实现多模态联合生成与任意转换。**

- **链接: [https://arxiv.org/pdf/2605.20090](https://arxiv.org/pdf/2605.20090)**

> **作者:** Zhiping Yu; Chenyang Liu; Jinqi Cao; Qinzhe Yang; Siwei Yu; Zhengxia Zou; Zhenwei Shi
>
> **摘要:** Multi-modal remote sensing images are vital for Earth observation, yet complete paired observations are often scarce in practice. Existing generative methods commonly address this problem through isolated pairwise modality translation, but their versatility and scalability remain limited as the number of modalities and generation tasks increases. Here, we develop a generative foundation model MetaEarth-MM for multi-modal remote sensing imagery, enabling paired joint generation and any-to-any translation across five modalities within a unified model. Recognizing the intrinsic scene consistency underlying multi-modal observations, we introduce a scene-centered joint modeling paradigm in MetaEarth-MM. Unlike previous methods that rely on direct appearance-level cross-modal mapping, our model organizes the generation around the underlying scene content. Specifically, MetaEarth-MM adopts a decoupled architecture that first infers a latent scene representation from available observations, and then generates target modalities conditioned on this intermediate state. To support training, we further construct EarthMM, a large-scale dataset comprising 2.8 million multi-resolution global images with 2.2 million aligned pairs. Extensive experiments demonstrate that MetaEarth-MM not only exhibits strong generative capability and robust generalization across diverse generation tasks, but also supports downstream tasks at both data and representation levels, highlighting its potential as a general foundation model for cross-modal Earth observation. The code and dataset will be available at this https URL.
>
---
#### [new 051] MatPhys: Learning Material-Aware Physics Parameters for Deformable Object Simulation from Videos
- **分类: cs.CV**

- **简介: 该论文属于物理模拟任务，解决从单目视频中重建一致的可变形物体物理参数的问题。通过引入材料感知框架，提升参数一致性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.19386](https://arxiv.org/pdf/2605.19386)**

> **作者:** Yang Yang; Yiyan Wang; Zheming Liu; Naoya Iwamoto
>
> **备注:** Submitted to Siggrah Asia 2026
>
> **摘要:** Reconstructing simulation-ready deformable objects is important for vision, graphics, and robotics. Existing physics-driven methods can recover physical digital twins from videos, but they suffer from two fundamental limitations: they typically assume a homogeneous material across the whole object, and their scene-specific inverse optimization, combined with the inherent ambiguity of monocular observation, yields inconsistent parameters for the same material across different scenes or interactions. We propose MatPhys, a material-aware feed-forward framework that predicts spring-mass parameters from a single-view video, addressing these two issues with two coupled designs. To relax the homogeneous material assumption, we use DINO features to decompose the object into semantically meaningful parts and to query a part-level material prior, assigning each part its own physical behavior. To enforce cross-scene consistency, we introduce a learned material codebook of shared material embeddings as the bridge between appearance and physics, and further use the part-level prior as a reference distribution that constrains the decoder so that the same material yields consistent parameters across scenes and interactions. Together, these designs turn an under-constrained monocular problem into feed-forward inference grounded on shared, reusable material concepts. Experiments show that our method matches per-scene optimization baselines in reconstruction and future prediction, while achieving stronger generalization to unseen interactions and objects with more consistent physical parameters.
>
---
#### [new 052] EgoCoT-Bench: Benchmarking Grounded and Verifiable Operation-Centric Chain of Thought Reasoning for MLLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出EgoCoT-Bench，用于评估多模态大模型在第一人称视频中的操作中心推理能力，解决现有基准缺乏细粒度因果验证的问题。**

- **链接: [https://arxiv.org/pdf/2605.19559](https://arxiv.org/pdf/2605.19559)**

> **作者:** Yang Dai; Dian Jiao; Tianwei Lin; Wenqiao Zhang
>
> **摘要:** The rapid development of Multimodal Large Language Models (MLLMs) has led to growing interest in egocentric video understanding, specifically the ability for MLLMs to recognize fine-grained hand-object interactions, track object state changes over time, and reason about manipulative processes in dynamic environments from a first-person perspective. However, existing egocentric video benchmarks suffer from \textbf{limited grounded rationale evaluation}, offering limited support for fine-grained operation-centric reasoning and rarely examining whether model rationales are grounded in explicit spatio-temporal evidence. To address this gap, we introduce \textbf{EgoCoT-Bench}, a fine-grained egocentric benchmark for grounded and verifiable operation-centric reasoning with explicit step-by-step rationale annotations. Overall, EgoCoT-Bench comprises 3,172 verifiable QA pairs over 351 egocentric videos separated into four task groups for a total of 12 sub-task groups, encompassing perception and retrospection, anticipation, and high-level reasoning. The benchmark is constructed through a spatio-temporal scene graphs (STSG) guided generation framework and is further refined by human annotators to ensure correctness, egocentric relevance and fine-grained quality. Experimental results show continuing difficulties with egocentric fine-grained reasoning and further reveal that many multimodal models produce explanations that are answer-correct, but have evidence that is inconsistent with the answer. We hope EgoCoT-Bench can serve as a useful testbed for grounded and verifiable reasoning in egocentric video understanding. Project page and supplementary materials are available at: this https URL.
>
---
#### [new 053] EventPrune: Cascaded Event-Assisted Token Pruning for Efficient First-Person Dynamic Spatial Reasoning
- **分类: cs.CV**

- **简介: 该论文针对第一人称动态空间推理任务，解决Transformer模型中视觉令牌计算成本高的问题。提出ECP框架，通过事件相机的运动线索进行高效令牌剪枝，提升推理速度与效率。**

- **链接: [https://arxiv.org/pdf/2605.19506](https://arxiv.org/pdf/2605.19506)**

> **作者:** Pengtao Ma; Ziliang Zhou; Ciyu Ruan; Haoyang Wang; Kaiyuan Li; Zihang Gong; Wenhua Ding; Chen Gao; Jingao Xu; Xinlei Chen
>
> **摘要:** First-person dynamic spatial reasoning requires models to track continuous motion and precise geometric structure, but the quadratic attention cost of Transformer-based Video-LLMs makes dense visual tokens computationally expensive. Existing token pruning paradigms predominantly rely on discrete static snapshots, failing to preserve the motion and geometric cues essential for reasoning. We propose Event Cascade Pruning (ECP), to our knowledge the first training-free framework that leverages the high-frequency motion cues from event cameras as a continuous event-guided motion prior to guide token selection. ECP combines three stages: Event-Triggered Causal Sampling to anchor motion-informative keyframes, Event-guided Motion Saliency Filtering to suppress event-inactive visual tokens, and Event-Attention Ranking Fusion to calibrate spatial attention with motion-salient dynamics. With 80% visual token reduction, ECP outperforms the full-token baseline (37.62% vs. 36.31%) while achieving 1.89x inference speedup and 52% GFLOPs reduction. We further introduce ESR-Real, the first real-world RGB-event benchmark for first-person spatial reasoning, where ECP improves accuracy by 2.68 percentage points over full-token baselines.
>
---
#### [new 054] Are Watermarked Images Editable? SafeMark for Watermark-Preserving Text-Guided Image Editing
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，解决水印图像在编辑过程中保持水印完整性的问题。提出SafeMark框架，在文本引导编辑中保留水印，确保编辑后图像的水印可恢复。**

- **链接: [https://arxiv.org/pdf/2605.19511](https://arxiv.org/pdf/2605.19511)**

> **作者:** Xiaodong Wu; Qi Li; Xiangman Li; Zelin Zhang; Lingshuang Liu; Jianbing Ni
>
> **摘要:** This paper investigates a fundamental yet underexplored question: can watermarked images remain editable without compromising watermark integrity? We propose SafeMark, a framework for watermark-preserving text-guided image manipulation that explicitly integrates watermark integrity into the editing process. Specifically, SafeMark adds a thresholded watermark-decoding loss directly to the diffusion editor's training objective, fine-tuning the editor so that semantically valid edits also preserve the embedded watermark at the final output. This design admits a clean information-theoretic justification: maintaining high bit-accuracy on the edited image lower-bounds the mutual information that the editor channel preserves between watermark and edited output, the quantity that fundamentally controls watermark recoverability. SafeMark is compatible with differentiable diffusion-based editors, and requires no architectural modification. Extensive evaluations across multiple datasets, text-guided editing methods, and post-edit distortion settings demonstrate that SafeMark achieves high watermark bit accuracy across diverse editing settings while maintaining high-quality semantic edits, without sacrificing robustness to common post-edit distortions. These results demonstrate that semantic editability and watermark integrity are fundamentally compatible, enabling trustworthy image provenance in generative editing pipelines.
>
---
#### [new 055] Thinking in Scales: Accelerating Gigapixel Pathology Image Analysis via Adaptive Continuous Reasoning
- **分类: cs.CV**

- **简介: 该论文属于病理图像分析任务，解决传统方法计算效率低的问题。提出PathCTM模型，通过自适应连续推理减少图像块数量，提升分析速度。**

- **链接: [https://arxiv.org/pdf/2605.19491](https://arxiv.org/pdf/2605.19491)**

> **作者:** Jiusong Ge; Yingkang Zhan; Wenjie Zhao; Di Zhang; Ke Wang; Jiashuai Liu; Chunze Yang; Chengzu Li; Jian Zhang; Yuxin Dong; Ni Zhang; Qidong Liu; Mireia Crispin-Ortuzar; Huazhu Fu; Chen Li; Zeyu Gao
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Traditional whole slide image (WSI) analysis methods typically rely on the multiple instance learning (MIL) paradigm, which extracts patch-level features at high magnification and aggregates them for slide-level prediction. However, such exhaustive patch-level processing is computationally expensive, severely limiting the efficiency and scalability of WSI analysis. To address this challenge, we propose PathCTM (a Pathology-oriented Continuous Thought Model) that enables token-efficient scale-space continuous reasoning for gigapixel WSIs. PathCTM formulates diagnostic inference as a dynamic sequential information pursuit. It progressively transitions from low-magnification global to high-magnification local inspection, and adaptively terminates inference when sufficient evidence is gathered to effectively bound decision uncertainty. Specifically, it uses conditional computation for dynamic scale switching with attention-guided region pruning, coupled with confidence-aware early stopping. Extensive experiments demonstrate that, compared with standard MIL-based methods, PathCTM reduces the number of required image patches by 95.95% and shortens inference time by approximately 95.62%, while maintaining AUC without degradation. Code is available at this https URL.
>
---
#### [new 056] Efficient Long-Context Modeling in Diffusion Language Models via Block Approximate Sparse Attention
- **分类: cs.CV**

- **简介: 该论文属于自然语言处理任务，解决长序列生成中注意力计算效率低的问题。提出BA-Att框架，通过块级下采样和近似方法提升效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2605.19726](https://arxiv.org/pdf/2605.19726)**

> **作者:** Wenhu Zhang; Yiming Wu; Huanyu Wang; Yaoyang Liu; Huanzhang Dou; Senqiao Yang; Sitong Wu; Hanbin Zhao; Jiaya Jia
>
> **备注:** CVPR 2026 Findings paper
>
> **摘要:** Diffusion Language Models (DLMs) enable globally coherent, bidirectional, and controllable text generation, offering advantages over traditional autoregressive LLMs, while scaling to ultra-long sequences remains costly. Many existing block-sparse attention methods select blocks by fixed sampling patterns over the high-resolution attention space, such as tail regions or anti-diagonal stripes. Such prior-driven sampling can miss salient tokens and introduce instability under distribution shifts. In this paper, we propose the Block Approximate Sparse Attention framework (BA-Att) with block-wise pre-downsampled operation, which identifies informative regions within a compact downsampled space, avoiding reliance on brittle positional priors. To analyze its theoretical behavior, we define an oracle post-downsample attention map and formalize the approximation error between pre- and post-downsample schemes. Based on this insight, we introduce a lightweight norm-sorting module and a covariance-compensated correction that approximates full covariance using diagonal QK variances, reducing computational complexity. Extensive experiments show that our operator achieves up to 6.95x acceleration over FlashAttention in attention computation, and maintains near full-attention performance at 50% sparsity across language models, multimodal language models, and video generation models, demonstrating strong efficiency and generalization.
>
---
#### [new 057] KappaPlace: Learning Hyperspherical Uncertainty for Visual Place Recognition via Prototype-Anchored Supervision
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉定位任务，解决VPR中不确定性估计不足的问题。提出KappaPlace框架，通过原型监督学习，提升定位的可靠性与准确性。**

- **链接: [https://arxiv.org/pdf/2605.19435](https://arxiv.org/pdf/2605.19435)**

> **作者:** Maya Yanko; Yoli Shavit
>
> **摘要:** Visual Place Recognition (VPR) is critical for autonomous navigation, yet state-of-the-art methods lack well-calibrated uncertainty estimation. Standard pipelines cannot reliably signal when a query is ambiguous or a match is likely incorrect, posing risks in safety-critical robotics. We propose KappaPlace, a principled framework for learning uncertainty-aware VPR representations. Our core contribution is a Prototype-Anchored supervision strategy that leverages latent class representatives as targets for a probabilistic objective. By modeling image descriptors as von Mises-Fisher (vMF) variables, we learn a lightweight module to predict the concentration parameter as a direct proxy for aleatoric uncertainty. While existing VPR uncertainty methods are typically restricted to a query-centric view, we derive a novel match-level formulation to quantify the reliability of specific query-reference pairs. Across five diverse benchmarks, KappaPlace reduces Expected Calibration Error (ECE@K) by up to 50% compared to existing methods while maintaining or improving retrieval recall. We provide both a joint-training variant and a post-training extension for frozen backbones. Our results demonstrate that KappaPlace provides a robust, stable, and well-calibrated signal that enables reliable decision-making within the VPR pipeline. Our code is available at: this https URL
>
---
#### [new 058] Concept-Guided Noisy Negative Suppression for Zero-Shot Classification and Grounding of Chest X-Ray Findings
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于零样本分类与定位任务，解决因患者间相似发现导致的噪声负样本问题。通过构建概念本体和改进负样本筛选策略提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.19374](https://arxiv.org/pdf/2605.19374)**

> **作者:** Chenyu Lian; Hong-Yu Zhou; Chun-Ka Wong; Jing Qin
>
> **备注:** Early accepted by MICCAI 2026
>
> **摘要:** Vision-language alignment using chest X-rays and radiology reports has emerged as an advanced paradigm for zero-shot classification and grounding of chest X-ray findings. However, standard contrastive learning typically treats radiographs and reports from different patients simply as negative pairs. This assumption introduces noisy negatives, as different patients frequently exhibit similar findings. Such noisy negatives cause semantic ambiguity and degrade performance in zero-shot understanding tasks. To address this challenge, we propose CoNNS, a concept-guided noisy-negative suppression framework. To support the negative suppression mechanism, unlike previous methods that use raw reports or templatized texts, we construct a hierarchical concept ontology using large language models. The ontology structures 41 key clinical concepts by explicitly modeling presence, attributes (location and characteristics), and texts (evidential segment and presence statement). Leveraging this ontology, we implement a cross-patient pair relabeling strategy comprising three steps: (1) Fine-Grained Breakdown to categorize pairs based on finding presence; (2) Noisy Negative Filtering to resolve semantic conflicts by removing false negatives; and (3) Hard Negative Mining to identify subtle attribute discrepancies using a lightweight language model. Finally, we propose a Concept-Aware NCE loss to align visual features with text while suppressing the identified noisy negatives. Extensive experiments across multi-granularity zero-shot grounding tasks and five zero-shot classification datasets validate that CoNNS outperforms existing state-of-the-art models. The code is available at this https URL.
>
---
#### [new 059] deadtrees.earth-aerial: A Multi-Resolution Aerial Image Dataset for Tree Cover and Mortality Detection
- **分类: cs.CV**

- **简介: 该论文提出DTE-aerial数据集，用于树冠和死亡检测任务，解决缺乏全球统一、高分辨率航空影像数据的问题，包含训练与基准测试集，支持机器学习模型训练与评估。**

- **链接: [https://arxiv.org/pdf/2605.19605](https://arxiv.org/pdf/2605.19605)**

> **作者:** Ayushi Sharma; Clemens Mosig; Lukas Drees; Salim Soltani; Janusch Vajna-Jehle; Aaron Sheppard; Belqis Ahmadi; Jonathan Schmid; Paul Neumeier; Nathan Jacobs; Jan Dirk Wegner; Teja Kattenborn
>
> **备注:** Preprint. Under review. All rights reserved
>
> **摘要:** Forests worldwide are increasingly threatened by climate change and disturbances such as fire, pests, and pathogens, creating an urgent need for scalable monitoring of tree cover and tree mortality. Aerial imagery from drones and aircraft is a key data source for detailed and large-scale mapping of tree crowns and mortality. However, related progress is limited by the lack of globally representative, harmonized datasets for joint segmentation of tree cover and mortality. We introduce two novel, open, machine-learning-ready datasets to enable joint segmentation of tree cover and tree mortality from centimeter-scale aerial imagery for the first time at global scales. With DTE-aerial-train, we provide a training dataset comprising 385K image patches of size 1024x1024 pixels, with resolutions ranging from 2.5 to 20 cm. It includes multi-class expert-annotated and -audited pseudo-labels for tree cover and mortality. With DTE-aerial-bench, we provide a geographically balanced benchmark test set of 25 globally distributed orthoimages totaling 525 patches with high-quality expert annotations for both tree cover and mortality. Both the training and benchmark datasets span tropical, temperate, boreal, and dryland biomes and cover a wide range of forest structures and mortality patterns. Using the benchmark test set for evaluation, we establish strong reference baselines that improve mortality segmentation across all biomes and scales with significant gains in challenging regions, such as boreal forests, where the F1 score increases from 0.40 to 0.58 with around 45% relative improvement. All data, models, and code will be publicly released under permissive open-source licenses. An interactive visualization of the benchmark dataset is available at this http URL.
>
---
#### [new 060] What Makes Synthetic Data Effective in Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务，探讨合成数据在分割中的有效性。研究分析了合成图像的特性，提出SENSE框架提升分割性能。**

- **链接: [https://arxiv.org/pdf/2605.19289](https://arxiv.org/pdf/2605.19289)**

> **作者:** Jinjin Zhang; Xiefan Guo; Yizhou Jin; Nan Zhou; Di Huang
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Driven by rapid advances in large-scale generative models, synthetic data has emerged as a promising solution for visual understanding. While modern diffusion models achieve remarkable photorealistic image synthesis, their potential in complex visual segmentation tasks remains underexplored. In this work, we conduct a systematic analysis of synthetic images from state-of-the-art diffusion models to uncover the factors governing their utility. In particular, synthetic images characterized by dense scene composition and fine instance fidelity demonstrate distinctive benefits, yielding significantly more discriminative spatial representations. Building on these insights, we propose SENSE, a unified framework that leverages flexible and scalable synthetic data to substantially enhance segmentation performance. Notably, SENSE is model-agnostic, compatible with diverse architectures (e.g., DPT and Mask2Former), and scales effectively across models with varying parameter capacities. Extensive experiments on Cityscapes, COCO, and ADE20K validate the effectiveness and generalization capability of our approach. Code is available at this https URL.
>
---
#### [new 061] MMGS: 10$\times$ Compressed 3DGS through Optimal Transport Aggregation based on Multi-view Ranking
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于3D重建任务，旨在解决3DGS中冗余点云导致的计算效率低问题。通过多视角排序和最优传输聚合，实现10倍压缩与加速。**

- **链接: [https://arxiv.org/pdf/2605.19304](https://arxiv.org/pdf/2605.19304)**

> **作者:** Beizhen Zhao; Sicheng Yu; Ziran Yin; Dongxu Shen; Hao Wang
>
> **备注:** 19 pages
>
> **摘要:** While 3D Gaussian Splatting (3DGS) has revolutionized 3D reconstruction, it suffers from significant overhead due to massive redundant primitives. Existing compression methods typically rely on local sampling or fixed pruning thresholds, which often struggle to balance redundancy reduction with high-fidelity rendering. To address this, we propose a novel framework that formulates Gaussian optimization as a global geometric distribution matching problem. Specifically, our approach integrates three components: (1) we introduce a multi-view 3D Gaussian contribution ranking mechanism that filters primitives using geometric consistency instead of local heuristics; (2) we propose a global Optimal Transport (OT)-based aggregation algorithm that merges redundant primitives while preserving the underlying geometry; and (3) we design an OT-based densification operator that maintains the Gaussian's distributional properties for stable optimization. Our approach achieves state-of-the-art rendering quality with only \textbf{10$\%$} primitives and \textbf{10$\times$} accelerated training speeds compared to vanilla 3DGS.
>
---
#### [new 062] Learning Long-Term Temporal Dependencies in Photovoltaic Power Output Prediction Through Multi-Horizon Forecasting
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于时间序列预测任务，旨在解决光伏功率输出的长期依赖问题。通过多时段预测框架提升模型准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.19074](https://arxiv.org/pdf/2605.19074)**

> **作者:** Sumit Laha; Ankit Sharma; Hassan Foroosh
>
> **摘要:** The rapid global expansion of solar photovoltaic (PV) capacity-reaching a record 597 GW in 2024-highlights the urgent need for robust forecasting models to mitigate the grid instability caused by the intermittent nature of solar irradiance. While deep learning-based direct forecasting using ground-based sky images (GSI) has emerged as a dominant approach, existing literature is often constrained by single-architecture evaluations and an exclusive focus on single-horizon (point) prediction. This paper proposes a transition from traditional single-horizon estimation toward a multi-horizon forecasting framework, leading to an architecture-independent improvement in accuracy. We hypothesize and demonstrate experimentally that joint optimization over a sequence of future values allows deep neural networks to better capture latent inter-step temporal dependencies by avoiding precocious convergence of the network in terms of both weight gradients and filter diversity. Leveraging this architecture-independent improvement that integrates sequential sky imagery with historical PV generation data, we evaluate the models' abilities to predict power output across multiple discrete future time steps simultaneously. Our methodology is validated through a comparative analysis across diverse deep learning architectures. The results demonstrate that this multi-horizon approach significantly enhances predictive accuracy and robustness across the entire forecast horizon while maintaining computational parsimony. By achieving superior performance with negligible overhead compared to single-horizon models, this work provides a scalable and efficient solution to improve the resilience of modern power grids.
>
---
#### [new 063] LiFT: Lifted Inter-slice Feature Trajectories for 3D Image Generation from 2D Generators
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文提出LiFT框架，解决3D医学图像生成中计算成本高与跨切片一致性差的问题，通过分阶段生成和轨迹学习实现高效高质量的3D合成。**

- **链接: [https://arxiv.org/pdf/2605.19060](https://arxiv.org/pdf/2605.19060)**

> **作者:** Xinhe Zhang; Yuyang Zhang; Pengfei Jin; Arnau Marin-Llobet; Na Li; Quanzheng Li
>
> **摘要:** High-resolution 3D medical image generation remains challenging because fully volumetric models are computationally expensive, while efficient 2D slice generators often fail to preserve anatomical consistency across the third dimension. We propose LiFT, a framework for Lifted inter-slice Feature Trajectories that factorizes 3D volume synthesis into per-slice image generation and inter-slice trajectory learning. Rather than modeling the volumetric distribution end-to-end, LiFT treats a volume as an ordered trajectory in feature space, capturing how anatomical structures appear, transform, and disappear across depth. A tri-planar drifting loss aligns the trajectory of generated slices with the trajectories of real volumes, enabling distributional learning over inter-slice progressions in unconditional generation; in paired translation, a bidirectional $z$-context mixer trained against the registered target supplies through-plane coherence while preserving per-slice fidelity. We evaluate LiFT on BraTS 2023 (unconditional and missing-modality MR) and SynthRAD2023 (MR-to-CT). Across these settings, LiFT preserves per-slice quality, approaches the reported cWDM missing-MR reconstruction quality at $\sim$$135\times$ lower inference cost (without formal equivalence testing), and improves through-plane coherence on MR-to-CT relative to a no-mapper ablation, demonstrating that lightweight inter-slice trajectory learning is a viable route to high-resolution 3D medical synthesis.
>
---
#### [new 064] D-Convexity: A Unified Differentiable Convex Shape Prior via Quasi-Concavity for Data-driven Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务，旨在解决如何有效引入凸性先验的问题。通过引入基于拟凹性的统一可微凸性约束，提升分割结果的形状规则性。**

- **链接: [https://arxiv.org/pdf/2605.19210](https://arxiv.org/pdf/2605.19210)**

> **作者:** Shengzhe Chen; Hao Yan
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Convexity is a fundamental geometric prior that underlies many natural and man-made structures, yet remains challenging to impose effectively in end-to-end trainable segmentation networks. We revisit convexity from a functional perspective and propose a unified, threshold-free convexity prior based on the quasi-concavity of the network's output mask function u. Instead of constraining a single binary segmentation, we require all super-level sets of u to be convex, transforming global shape constraints into local, differentiable inequalities on u and its derivatives. From this principle, we derive zero, first, and second-order characterizations, yielding respectively a local midpoint convexification algorithm, a gradient-based condition linked to supporting hyperplanes, and a sufficient second-order inequality expressed as a quadratic form on the tangent plane. The first and second-order formulations produce a compact convolutional loss that can be densely applied across the image without thresholding. Our quasi-concavity losses integrate seamlessly with modern segmentation networks via the proposed convex gradient projection module (CGPM). They consistently enforce convexity and improve shape regularity across multiple datasets, outperforming networks tailored for retinal segmentation and surpassing previous shape-aware methods. Remarkably, our analysis unifies a wide spectrum of previous convex shape models, from discrete 1-0-1 line constraints and graph-cuts convexity formulations to curvature or signed distance Laplacian based level-set priors, within a single continuous and differentiable framework.
>
---
#### [new 065] Towards Fine-Grained Robustness: Attention-Guided Test-Time Prompt Tuning for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型的鲁棒性提升任务，旨在解决对抗攻击导致的性能下降问题。提出A-TPT方法，通过注意力引导实现细粒度的测试时提示调优。**

- **链接: [https://arxiv.org/pdf/2605.19956](https://arxiv.org/pdf/2605.19956)**

> **作者:** Jia-Wei Hai; Yijun Wang; Xiu-Shen Wei
>
> **备注:** Accepted by ICML 2026, Project Page: this https, URL Code URL: this https URL
>
> **摘要:** Vision-Language Models (VLMs), such as CLIP, have achieved significant zero-shot performance on downstream tasks with various fine-tuning adaptation methods. However, recent studies have proven that adversarial attacks can significantly degrade the inference ability of VLMs, posing substantial risks to their practical applications. Prevalent test-time adaptation methods typically rely on multi-view augmentation to implement various fine-tuning strategies, which struggle to identify semantic information and are prone to destroying discriminative regions in fine-grained scenarios. To address these limitations, we propose Attention-Guided Test-Time Prompt Tuning (A-TPT), a semantics-preserving method designed for test-time adaptation. We first refine the gradient attention rollout mechanism to identify semantically meaningful regions surviving under adversarial attacks. Furthermore, we leverage them to guide the spatially varying augmentation intensities and multi-view ensemble for prompt tuning and inference. Extensive experiments demonstrate that A-TPT outperforms existing test-time adaptation methods on both adversarial and clean data. Codes are available at this https URL .
>
---
#### [new 066] Cross-View Splatter: Feed-Forward View Synthesis with Georeferenced Images
- **分类: cs.CV**

- **简介: 该论文属于场景重建任务，旨在通过融合卫星与地面图像，提升户外场景的视图合成效果。工作包括构建统一3D坐标系，预测高精度高斯点云。**

- **链接: [https://arxiv.org/pdf/2605.19656](https://arxiv.org/pdf/2605.19656)**

> **作者:** Matias Turkulainen; Akshay Krishnan; Filippo Aleotti; Mohamed Sayed; Guillermo Garcia-Hernando; Juho Kannala; Arno Solin; Gabriel Brostow; Daniyar Turmukhambetov
>
> **备注:** Submitted to CVPR 2026. 8 figures, 3 tables. Project page: this https URL
>
> **摘要:** We present Cross-View Splatter, a feed-forward method that predicts pixel-aligned Gaussian splats for outdoor scenes captured at ground level AND by satellite. Faithful reconstructions require good camera coverage, but ground imagery is time-consuming and hard to capture at scale for large outdoor scenes. Fortunately, satellite imagery can provide a global geometric prior that is easy to access via public APIs. Cross-View Splatter fuses orthorectified satellite views with GPS-tagged ground photos to predict Gaussian splats in a unified 3D coordinate frame. By aligning ground and bird's-eye feature representations, our model improves scene coverage and novel-view synthesis, compared to ground imagery alone. We train on curated georeferenced datasets and paired satellite-terrain data, mined from open mapping services. We evaluate our method on a new benchmark for novel-view synthesis with georeferenced imagery allowing comparison to prior state-of-the-art methods. Our code and data preparation will be available at this https URL.
>
---
#### [new 067] LaCoVL-FER: Landmark-Guided Contrastive Learning Network with Vision-Language Enhancement for Facial Expression Recognition
- **分类: cs.CV**

- **简介: 该论文属于面部表情识别任务，旨在解决复杂场景下的表达识别问题。提出LaCoVL-FER网络，结合面部关键点和视觉-语言模型，提升识别性能。**

- **链接: [https://arxiv.org/pdf/2605.19821](https://arxiv.org/pdf/2605.19821)**

> **作者:** Jiaxin Wang; Muwei Jian; Hui Yu; Junyu Dong; Yifan Xia
>
> **摘要:** Facial Expression Recognition (FER) in the wild is still challenging due to uncontrolled variations in pose, occlusion, and illumination. Most existing attention-based methods primarily rely on visual appearance cues, suffering from attention redundancy and instability, which limits their performance in complex scenarios. To address these issues, we propose a novel landmark-guided contrastive learning network with vision-language enhancement for FER (LaCoVL-FER), which integrates geometric priors from facial landmarks and semantic priors from a vision-language model. Specifically, a Landmark-Guided Adaptive Encoder (LGAE) is designed to introduce geometric priors through a Bi-branch Gated Cross Attention (BGCA) mechanism, which achieves adaptive fusion of landmark-based geometric and visual appearance features to produce expression-relevant features, thereby focusing on key facial regions and suppressing noise interference. In parallel, a Vision-Language Enhancement Strategy (VLES) is presented to leverage the expression-relevant features to refine the generalizable visual features extracted by the frozen pretrained CLIP image encoder, yielding expression-specific visual representations. Based on these representations, an Expression-Conditioned Prompting (ECP) mechanism is utilized to further adapt the textual features of fixed class-level prompts from the frozen pretrained CLIP text encoder, generating more instance-aware textual representations. These visual-textual representations are aligned as semantic priors to enhance the robustness and generalization of FER. Quantitative and qualitative experiments demonstrate that our LaCoVL-FER outperforms state-of-the-art methods on three representative real-world FER datasets, including RAF-DB, FERPlus, and AffectNet. The code is available at this https URL.
>
---
#### [new 068] Lens Privacy Sealing: A New Benchmark and Method for Physical Privacy-Preserving Action Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于隐私保护下的动作识别任务，旨在解决摄像头采集过程中的隐私泄露问题。提出LPS硬件方案和MSPNet算法，实现高效隐私保护与准确动作识别。**

- **链接: [https://arxiv.org/pdf/2605.19578](https://arxiv.org/pdf/2605.19578)**

> **作者:** Mengyuan Liu; Ziyi Wang; Peiming Li; Junsong Yuan
>
> **备注:** 15 pages, 9 figures,
>
> **摘要:** RGB camera-based surveillance systems enable human action recognition for public safety and healthcare, yet raise serious privacy concerns. Existing methods rely on post-capture algorithms, which fail to protect privacy during data acquisition. We propose Lens Privacy Sealing (LPS), a simple hardware solution that physically obscures camera lenses with adjustable laminating film, providing pre-sensor privacy protection at minimal cost. Unlike software methods or expensive engineered optics, LPS achieves strong privacy through stochastic multi-layer scattering that is physically irreversible. We introduce the P$^3$AR dataset for privacy-preserving action recognition, featuring both large-scale replay-captured (P$^3$AR-NTU, 114K videos) and real-world collected (P$^3$AR-PKU) subsets with privacy attribute annotations. To handle video degradation from LPS, we propose MSPNet, a single-stage framework incorporating Inter-Frame Noise Suppressor (IFNS) and Cross-Frame Semantic Aggregator (CFSA), enhanced by contrastive language-image pre-training for robust semantic extraction. Extensive experiments demonstrate that MSPNet with IFNS and CFSA nearly doubles action recognition accuracy compared to baseline methods while suppressing identity recognition to low levels. Comprehensive validation shows LPS achieves a superior privacy-utility trade-off compared to state-of-the-art hardware methods, resists reconstruction attacks including PSF inversion and data-driven recovery, and generalizes robustly across optical configurations and challenging environments. Code is available at this https URL.
>
---
#### [new 069] WBCAtt+: Fine-Grained Pixel-Level Morphological Annotations for White Blood Cell Images
- **分类: cs.CV**

- **简介: 该论文提出WBCAtt+数据集，解决白细胞图像细粒度形态标注不足的问题，包含11个形态属性和5个像素级组件，用于支持白细胞分析与解释性AI研究。**

- **链接: [https://arxiv.org/pdf/2605.19692](https://arxiv.org/pdf/2605.19692)**

> **作者:** Satoshi Tsutsui; Winnie Pang; Shuting He; Bihan Wen
>
> **备注:** Accepted to Medical Image Analysis. arXiv admin note: substantial text overlap with arXiv:2306.13531
>
> **摘要:** The microscopic examination of white blood cells (WBCs) plays a fundamental role in pathology and is essential for diagnosing blood disorders such as leukemia and anemia. To support further research on WBC images, multiple datasets have been proposed. However, they mainly annotate cell categories, and lack detailed morphological characteristics that pathologists use to explain their interpretations of cells. To address this gap, we introduce WBCAtt+, a novel dataset of WBC images densely annotated with 11 morphological attributes and five pixel-level cell components. With 113k image-level labels and 10k segmentation maps, WBCAtt+ is the first to provide comprehensive annotations for WBC images. Leveraging this dataset, we provide baseline models for attribute recognition and semantic segmentation. We also design an attribute recognition model to incorporate compositional structure of cells, further improving the recognition performance. Lastly, we showcase various applications enabled by our dataset, such as explainable AI models, including counterfactual example generation. \revision{The dataset and code are publicly available\footnote{this https URL}}.
>
---
#### [new 070] PixVerve: Advancing Native UHR Image Generation to 100MP with a Large-Scale High-Quality Dataset
- **分类: cs.CV**

- **简介: 该论文属于文本生成图像任务，解决UHR图像生成难题。构建了95K张100MP高质量数据集PixVerve-95K，并提出训练方案与评估基准PixVerve-Bench。**

- **链接: [https://arxiv.org/pdf/2605.20147](https://arxiv.org/pdf/2605.20147)**

> **作者:** Haojun Chen; Haoyang He; Chengming Xu; Qingdong He; Junwei Zhu; Yabiao Wang; Zhucun Xue; Xianfang Zeng; Zhennan Chen; Xiaobin Hu; Hao Zhao; Yong Liu; Jiangning Zhang; Dacheng Tao
>
> **备注:** Project page is available at this https URL
>
> **摘要:** Text-to-Image (T2I) models have recently seen notable progress around 1K and 2K resolution. With the extreme desire for better visual experience and the rapid development of imaging technology, the demand for Ultra-High-Resolution (UHR) image generation has grown significantly. However, UHR image generation poses great challenges due to the scarcity and complexity of high-resolution content. In this paper, we first introduce PixVerve-95K, a high-quality, open-source UHR T2I dataset curated with a carefully designed data pipeline, which contains 95K images across diverse scenarios (each image has a minimum pixel-count of 100M) and seven-dimensional annotations. Based on our large-scale image-text dataset, we take a pioneering step to extend various T2I foundation models to native 100MP generation with three training schemes. Finally, leveraging both conventional metrics and multimodal large language model-based assessments, our proposed PixVerve-Bench benchmark establishes a comprehensive evaluation protocol for UHR images encompassing visual quality and semantic alignment. Extensive experimental results on our benchmark and the constructive exploration of training strategies collaboratively provide valuable insights for future breakthroughs.
>
---
#### [new 071] Stage-adaptive Token Selection for Efficient Omni-modal LLMs
- **分类: cs.CV**

- **简介: 该论文属于多模态大模型优化任务，旨在解决非文本token计算冗余问题。提出SEATS方法，在不同阶段动态选择关键token，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2605.20035](https://arxiv.org/pdf/2605.20035)**

> **作者:** Zijie Xin; Jie Yang; Ruixiang Zhao; Tianyi Wang; Fengyun Rao; Jing Lyu; Xirong Li
>
> **备注:** Code Link: this https URL
>
> **摘要:** Omni-modal large language models (om-LLMs) achieve unified audio-visual understanding by encoding video and audio into temporally aligned token sequences interleaved at the window level. However, processing these dense non-textual tokens throughout the LLM incurs substantial computational overhead. Although training-free token selection can reduce this cost, existing methods either focus on visual-only inputs or prune om-LLM tokens only before the LLM with fixed per-modality ratios, failing to capture how cross-modal token importance evolves across layers. To address this limitation, we first analyze the layer-wise token dependency of om-LLMs. We find that visual and audio dependencies follow a block-wise pattern and gradually weaken with depth, indicating that many late-layer non-textual tokens become redundant after cross-modal fusion. Motivated by this observation, we propose SEATS, a training-free, stage-adaptive token selection method for efficient om-LLM inference. Before the LLM, SEATS removes spatiotemporal redundancy via attention-weighted diversity selection. Inside the LLM, it progressively prunes tokens across blocks and dynamically allocates the retention budget from temporal windows to modalities using query relevance scores. In late layers, it removes all remaining non-textual tokens once cross-modal fusion is complete. Experiments on Qwen2.5-Omni and Qwen3-Omni demonstrate that SEATS effectively improves inference efficiency. Retaining only 10% of visual and audio tokens, it achieves a 9.3x FLOPs reduction and a 4.8x prefill speedup while preserving 96.3% of the original performance.
>
---
#### [new 072] TideGS: Scalable Training of Over One Billion 3D Gaussian Splatting Primitives via Out-of-Core Optimization
- **分类: cs.CV; cs.PF**

- **简介: 该论文提出TideGS，解决大规模3D高斯点云训练的内存限制问题，通过外存优化实现超十亿高斯基元的高效训练。**

- **链接: [https://arxiv.org/pdf/2605.20150](https://arxiv.org/pdf/2605.20150)**

> **作者:** Chonghao Zhong; Linfeng Shi; Hua Chen; Tiecheng Sun; Hao Zhao; Binhang Yuan; Chaojian Li
>
> **备注:** Accepted to ICML 2026 as Spotlight. Website: this https URL
>
> **摘要:** Training 3D Gaussian Splatting (3DGS) at billion-primitive scale is fundamentally memory-bound: each Gaussian primitive carries a large attribute vector, and the aggregate parameter table quickly exceeds GPU capacity, limiting prior systems to tens of millions of Gaussians on commodity single-GPU hardware. We observe that 3DGS training is inherently sparse and trajectory-conditioned: each iteration activates only the Gaussians visible from the current camera batch, so GPU memory can serve as a working-set cache rather than a persistent parameter store. Building on this insight, we introduce TideGS, an out-of-core training framework that manages parameters across an SSD-CPU-GPU hierarchy via three synergistic techniques: block-virtualized geometry for SSD-aligned spatial locality, a hierarchical asynchronous pipeline to overlap I/O with computation, and trajectory-adaptive differential streaming that transfers only incremental working-set deltas between iterations. Experiments show that TideGS enables training with over one billion Gaussians on a single 24 GB GPU while achieving the best reconstruction quality among evaluated single-GPU baselines on large-scale scenes, scaling beyond prior out-of-core baselines (e.g., approximately 100M Gaussians) and standard in-memory training (e.g., approximately 11M Gaussians).
>
---
#### [new 073] Synergistic Foundation Models for Semi-Supervised Fetal Cardiac Ultrasound Analysis: SAM-Med2D Boundary Refinement and DINOv3 Semantic Enhancement
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于胎儿心脏超声图像的分割与分类任务，旨在提升先天性心脏病筛查效果。通过融合SAM-Med2D和DINOv3，优化伪标签质量，提高模型性能。**

- **链接: [https://arxiv.org/pdf/2605.19799](https://arxiv.org/pdf/2605.19799)**

> **作者:** Tonghao Zhuang; Shanglong Hu; Yongsheng Luo; Zhiqi Zhang; Yu Li
>
> **备注:** Accepted to the ISBI 2026 Fetal HearT UltraSound Segmentation and Diagnosis (FETUS) Challenge
>
> **摘要:** We present a semi-supervised framework for joint segmentation and classification of fetal cardiac ultrasound images. Built upon the EchoCare multi-task backbone, our method integrates SAM-Med2D for boundary refinement and leverages DINOv3 to enhance pseudo-label quality. We introduce view-specific hard masking along with a two-stage optimization strategy: an EMA phase to consolidate segmentation capabilities, followed by a Classification Fine-Tuning phase that freezes segmentation parameters and resets the classification head to recover classification performance without compromising segmentation gains. Evaluated on the FETUS 2026 leaderboard, our method achieves a Dice Similarity Coefficient at 79.99%, Normalized Surface Distance at 61.62%, and F1-score at 41.20%, validating the effectiveness of our approach for prenatal congenital heart disease screening. Source code is publicly available at: this https URL.
>
---
#### [new 074] FineBench: Benchmarking and Enhancing Vision-Language Models for Fine-grained Human Activity Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出FineBench，一个用于评估视觉语言模型在细粒度人类活动理解能力的基准，解决现有模型在空间推理和细微动作区分上的不足。**

- **链接: [https://arxiv.org/pdf/2605.19846](https://arxiv.org/pdf/2605.19846)**

> **作者:** Gueter Josmy Faure; Min-Hung Chen; Jia-Fong Yeh; Hung-Ting Su; Winston H. Hsu
>
> **备注:** CVPR'26 (Workshop on Video Large Language Models)
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated remarkable capabilities in general video understanding, yet they often struggle with the fine-grained comprehension crucial for real-world applications requiring nuanced interpretation of human actions and interactions. While some recent human-centric benchmarks evaluate aspects of model behaviour such as fairness/ethics, emotion perception, and broader human-centric metrics, they do not combine long-form videos, very dense QA coverage, and frame-level spatial/temporal grounding at scale. To bridge this gap, we introduce FineBench, a human-centric video question answering (VQA) benchmark specifically designed to assess fine-grained understanding. FineBench comprises 199,420 multiple-choice QA pairs densely annotated across 64 long-form videos (15 minutes each), focusing on detailed person movement, person interaction, and object manipulation, including compositional actions. Our extensive evaluation reveals that while proprietary models like GPT-5 achieve respectable performance, current open-source VLMs significantly underperform, struggling particularly with spatial reasoning in multi-person scenes and distinguishing subtle differences in human movements and interactions. To address these identified weaknesses, we propose FineAgent, a modular framework that enhances VLMs by leveraging a Localizer and a Descriptor. Experiments show that FineAgent consistently improves the performance of various open VLMs on FineBench. FineBench provides a rigorous testbed for future research into fine-grained human-centric video understanding, while FineAgent offers a practical approach to enhance such reasoning in current VLMs.
>
---
#### [new 075] CADENet: Condition-Adaptive Asynchronous Dual-Stream Enhancement Network for Adverse Weather Perception in Autonomous Driving
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文属于自动驾驶中的恶劣天气目标检测任务，解决增强与检测无法实时协同的问题。提出CADENet，实现无延迟检测与条件自适应增强，提升恶劣天气下感知效果。**

- **链接: [https://arxiv.org/pdf/2605.19837](https://arxiv.org/pdf/2605.19837)**

> **作者:** Sherif Khairy; Catherine M. Elias
>
> **摘要:** Adverse weather (rain, fog, sand, and snow) degrades camera-based object detection in autonomous vehicles. Existing enhancement-then-detect approaches stall the safety-critical perception loop, violating hard real-time requirements. Progress on this problem is also constrained by an under-recognized evaluation ceiling: ground truth annotated on degraded images cannot credit a detector that recovers objects the annotators themselves could not see, so a genuinely useful enhancement can register as a near-flat F1 gain. This paper presents CADENet (Condition-Adaptive Asynchronous Dual-stream Enhancement Network), a training-free three-thread system: Thread S (YOLOv11n) delivers detections at full frame rate with zero added latency; Thread Q applies condition-adaptive enhancement (CAPE) and fuses results via entropy-guided NMS (EG-NMS) without blocking Thread S; Thread E provides CLIP zero-shot weather classification, so new weather categories require only a new text prompt, with no labeled data and no retraining. Evaluated on 1327 DAWN images (YOLOv11m, IoU = 0.5, confidence = 0.25), CADENet achieves Recall = 0.0103 (micro), F1 = 0.0230 on snow, and F1 = 0.0038 on rain. We formalize the annotation completeness bias on DAWN-class data, so the reported F1 values are lower bounds on the true gain; recall is the annotation-gap-immune headline metric. Thread S sustains approximately 44 FPS regardless of enhancement load. No model retraining or additional sensor hardware is required.
>
---
#### [new 076] MAM-CLIP: Vision-Language Pretraining on Mammography Atlases for BI-RADS Classification
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在提升BI-RADS评分的准确性。通过引入视觉-语言预训练方法，利用影像与文本信息增强模型性能，尤其在数据稀缺时效果显著。**

- **链接: [https://arxiv.org/pdf/2605.19359](https://arxiv.org/pdf/2605.19359)**

> **作者:** Halil Ibrahim Gulluk; Olivier Gevaert
>
> **摘要:** Deep learning methods have demonstrated promising results in predicting BI-RADS scores from mammography images. However, the interpretation of these images can vary, leading to discrepancies even among radiologists. Given the inherent complexity of mammograms, training classification models solely on image labels often yields limited performance. To address this challenge, we curated 2313 mammogram images and their corresponding captions from two mammography atlases. Our proposed approach employs a multi-modal model that uses a pretrained PubMedBERT as the language component. By training this model on image-text pairs with contrastive learning, we enable the vision encoder to absorb the rich information contained in the captions, thereby improving its understanding of mammography findings. We then fine-tune the vision encoder on two datasets for BI-RADS prediction, achieving superior performance compared with models trained without this pretraining, particularly when labeled samples are scarce. The improvement in the 3-class average F1 score ranges from +1% to +14%: a +1% increase with 40K training samples, and a +14% increase with 1K samples. Furthermore, our experiments reveal that 2K image-text pairs from mammography atlases can be more informative than 2K labeled samples for label prediction, with an average margin of +1.1% when more than 10K training samples are available. Overall, our work provides a vision-language model for mammography and highlights the value of textual information from mammography atlases. In addition, we publicly release preprocessed mammography images of the TEKNOFEST dataset. The training code, pre-trained model weights, data extraction scripts, and the released dataset are publicly available at: this https URL
>
---
#### [new 077] MotionMERGE: A Multi-granular Framework for Human Motion Editing, Reasoning, Generation, and Explanation
- **分类: cs.CV**

- **简介: 该论文提出MotionMERGE，解决人体运动编辑与推理中的细粒度控制问题，通过多粒度框架实现精准运动生成与理解。**

- **链接: [https://arxiv.org/pdf/2605.18956](https://arxiv.org/pdf/2605.18956)**

> **作者:** Bizhu Wu; Jinheng Xie; Wenting Chen; Zhe Kong; Jianfeng Ren; Linlin Shen; Ruibin Bai; Rong Qu
>
> **摘要:** Recent motion-language models unify tasks like comprehension and generation but operate at a coarse granularity, lacking fine-grained understanding and nuanced control over body parts needed for animation or interaction. This stems from fundamental issues in both the model and the data, in which the model can't focus on motion's localized pattern, and the training data lacks fine-grained supervision. To tackle this, we propose MotionMERGE, a unified framework that bridges the granularity gap. First, we pioneer the study of fine-grained languageguided motion control, including detailed understanding and localized editing, by explicitly modeling motion at part and temporal levels within a single LLM, thereby endowing the model with robust priors for precise control. Second, we design ReasoningAware Granularity-Synergy pre-training, a novel strategy that employs joint supervision for cross-granularity alignment, temporal grounding, localized alignment, motion coherency, and motion-grounded chain-of-thought (CoT) reasoning. This equips the model with fine-grained motion-language alignment, crossgranularity synergy, and explicit reasoning ability. Third, we curate MotionFineEdit, a large-scale dataset (837K atomic + 144K complex triplets) with the first fine-grained spatio-temporal corrective instructions and motion-grounded CoT annotations, establishing a new benchmark for fine-grained text-driven motion editing and motion-grounded reasoning. Extensive experiments demonstrate the capability of MotionMERGE for more precise motion generation, understanding, and editing, and compelling zero-shot generalization to other complex motion tasks. This work represents a significant step toward models that interact with motion in finer granularity and human-like reasoning.
>
---
#### [new 078] When Preference Labels Fall Short: Aligning Diffusion Models from Real Data
- **分类: cs.CV**

- **简介: 该论文属于生成模型对齐任务，旨在解决依赖模型生成样本构建偏好对的局限性。工作通过使用真实数据作为监督信号，提升扩散模型的对齐效果。**

- **链接: [https://arxiv.org/pdf/2605.19839](https://arxiv.org/pdf/2605.19839)**

> **作者:** Weiyan Chen; Weijian Deng; Yao Xiao; Weijie Tu; ZiYi Dong; Ibrahim Radwan; Liang Lin; Pengxu Wei
>
> **备注:** ICML 2026 Camera Ready; Project Page: this https URL
>
> **摘要:** Preference alignment aims to guide generative models by learning from comparisons between preferred and non-preferred samples. In practice, most existing approaches rely on preference pairs constructed from model-generated images. Such supervision is inherently relative and can be ambiguous when both samples exhibit artifacts or limited visual quality, making it difficult to infer what constitutes a truly desirable output. In this work, we investigate whether real data can serve as an alternative source of supervision for preference alignment. We adopt a data-centric perspective and study a curation strategy that treats real images as reference points and constructs preference signals by contrasting them with generated or perturbed samples, without requiring manually annotated preference pairs. Through empirical analysis, we show that real-data-based supervision provides effective guidance for aligning diffusion models and achieves performance comparable to existing preference-based methods. Our results suggest that real data offers a practical and complementary source of supervision for preference alignment and highlight directions of label-efficient alignment strategies. Code and models are available at this https URL.
>
---
#### [new 079] Interpretable Computer Vision for Defect Detection in X-ray Tomography of Aerospace SiC/SiC Composites
- **分类: cs.CV; cond-mat.mtrl-sci; cs.LG**

- **简介: 该论文属于缺陷检测任务，旨在解决X射线断层扫描中缺陷分类的透明性问题。通过引入可解释的p-ResNet-50框架，实现高精度且可追溯的缺陷识别。**

- **链接: [https://arxiv.org/pdf/2605.20159](https://arxiv.org/pdf/2605.20159)**

> **作者:** Antonio Peña Corredor; Julien Lesseur; Romain Nunez; Paul Rivalland; Thomas Philippe
>
> **摘要:** Non-destructive testing of aerospace SiC/SiC composites via X-ray computed tomography (XCT) relies on expert visual assessment, with current workflows offering limited traceability for accept/reject decisions. Deep convolutional networks can automate defect detection, yet their black-box nature conflicts with the transparency that industrial inspection practice demands. To close this gap, we introduce p-ResNet-50, a convolutional framework extended with a prototype layer that couples high detection accuracy with case-based explanations. Six learned prototypes are explicitly aligned with expert-defined semantic categories-healthy matrix, matrix--air interfaces, pores, line-like defects, and mixed morphologies-so that every classification is traceable to a physically meaningful reference. Two novel regularisation terms, anchor-based and medoid-based, tether prototypes to expert-selected patches and prevent prototype collapse, addressing a known limitation of prototype networks. Latent-space analysis via UMAP delineates semantically coherent sub-domains and maps zones of uncertainty where misclassifications concentrate, giving inspectors an explicit picture of where the model is-and is not-reliable. The framework is validated on an XCT patch dataset of approximately 12,000 patches extracted from four defect-rich SiC/SiC laboratory specimens. Taking a black-box ResNet-50 as a baseline (ROC-AUC = 0.991), the prototype extension achieves comparable performance (accuracy 0.957 vs. 0.959; ROC-AUC 0.994 vs. 0.993) while trading a slight reduction in sensitivity for higher precision and specificity. Each decision is backed by representative evidence patches, and the model explicitly flags its uncertainty regions. Beyond defect mapping, the framework establishes a reusable methodology for embedding domain-expert knowledge into prototype networks, applicable to other XCT inspection scenarios requiring traceable, auditable decisions.
>
---
#### [new 080] Efficient coding along the visual hierarchy
- **分类: cs.CV**

- **简介: 该论文研究生物视觉系统如何用有限数据高效学习，提出通过高效编码构建视觉层次特征。任务是理解视觉表征形成机制，解决数据效率问题，工作包括设计无监督学习模型并验证其与人类视觉的一致性。**

- **链接: [https://arxiv.org/pdf/2605.19155](https://arxiv.org/pdf/2605.19155)**

> **作者:** Ananya Passi; Brian S. Robinson; Michael F. Bonner
>
> **备注:** 34 pages, 6 figures
>
> **摘要:** Biological visual systems learn from limited experience, unlike deep learning models that rely on millions of training images. What learning principles make this possible? We tested whether efficient coding, the idea that neural representations capture the statistical structure of natural inputs, can build a hierarchy of human-aligned visual features from limited data. We developed an unsupervised learning procedure in which each layer of a deep network compresses its inputs onto the dominant modes of variation in natural images, using only local statistics and no labels, tasks, or backpropagation. This unsupervised procedure yields features that progress from edges and colors to textures and shapes. The features of this deep efficient coding model are readily recognized by human observers and are predictive of image-evoked fMRI responses in human visual cortex. Furthermore, a hybrid learning procedure that combines efficient coding with supervised fine-tuning yields better brain alignment in low-data settings and more rapid category learning. These findings suggest that efficient coding may shape representations across the entire visual hierarchy and help explain the data efficiency of biological vision.
>
---
#### [new 081] Robust Mitigation of Age-Dependent Confounding Effects via Sample-Difficulty Decorrelation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在解决年龄相关偏差问题。通过消除年龄与疾病预后的虚假关联，提升模型在不同年龄分布下的诊断准确性。**

- **链接: [https://arxiv.org/pdf/2605.19230](https://arxiv.org/pdf/2605.19230)**

> **作者:** Nikhil Cherian Kurian; Victor Caquilpan Parra; Abin Shoby; Luke Whitbread; Lyle J. Palmer
>
> **备注:** 10 Pages, 3 Figures
>
> **摘要:** Age dependent performance disparities in medical image classification often arise because age acts as a confounder, linking imaging morphology with disease prevalence. In practice, disparities can manifest as overdiagnosis at ages where disease prevalence is higher and underdiagnosis at ages where prevalence is lower, and can worsen under train test shifts in the age distribution. Conventional mitigation approaches that enforce strict age invariance may suppress diagnostically meaningful information encoded in age. We therefore propose a robust framework that mitigates the effects of age-dependent confounding by targeting spurious age linked trends rather than enforcing invariance. Following a warm-up phase, we characterize sample difficulty and model its age-dependent trends in a label-conditioned manner. We decorrelate age from dominant age difficulty trends using robust, Huber weighted affinity weights, attenuating confounding-driven shortcuts while preserving clinically meaningful, nonlinear age information. We further introduce an Age Coverage Score that scales the decorrelation penalty by minibatch age variance to ensure stable optimization under limited age diversity. Across two radiology datasets, our approach reduces age dependent true and false positive disparities with minimal AUC impact and remains robust to increasing train test age distribution shifts.
>
---
#### [new 082] MSAVBench: Towards Comprehensive and Reliable Evaluation of Multi-Shot Audio-Video Generation
- **分类: cs.CV**

- **简介: 该论文属于多镜头音视频生成任务，旨在解决现有评估基准不足的问题。提出MSAVBench，涵盖多维度评估框架，提升模型评价的全面性与可靠性。**

- **链接: [https://arxiv.org/pdf/2605.20183](https://arxiv.org/pdf/2605.20183)**

> **作者:** Yujie Wei; Yujin Han; Zhekai Chen; Yongming Li; Kaixun Jiang; Zhihang Liu; Quanhao Li; Zhiwu Qing; Xiang Wang; Zhen Xing; Ruihang Chu; Lingyi Hong; Yefei He; Junjie Zhou; Junqiu Yu; Yang Shi; Difan Zou; Kai Zhu; Shiwei Zhang; Yingya Zhang; Yu Liu; Xihui Liu; Hongming Shan
>
> **摘要:** Video generation is rapidly evolving from single-shot synthesis to complex multi-shot audio-video (MSAV) narratives to meet real-world demands. However, evaluating such frontier models remains a fundamental challenge. Existing benchmarks are limited in scope and data diversity, and rely on rigid evaluation pipelines, preventing systematic and reliable assessment of modern MSAV models. To bridge these gaps, we introduce MSAVBench, the first comprehensive benchmark and adaptive hybrid evaluation framework for multi-shot audio-video generation. Our benchmark spans four key dimensions, video, audio, shot, and reference, covering diverse task settings, varying shot counts of up to 15, and challenging non-realistic scenarios. Our evaluation framework improves robustness through an adaptive self-correction mechanism for shot segmentation, instance-wise rubrics for subjective metrics, and tool-grounded evidence extraction for complex judgments. Furthermore, MSAVBench achieves high alignment with human judgments, reaching a Spearman rank correlation of 91.5%. Our systematic evaluation of 19 state-of-the-art closed- and open-source models shows that current systems still struggle with director-level control and fine-grained audio-visual synchronization, while modular or agentic generation pipelines offer a promising path toward narrowing the gap between open- and closed-source models. We will release the benchmark data and evaluation code to facilitate future research.
>
---
#### [new 083] Rethinking Visual Attribution for Chest X-ray Reasoning in Large Vision Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于医学视觉问答任务，旨在解决LVLMs解释能力不足的问题。通过构建因果评估框架，发现现有方法无法准确定位视觉证据，并提出MedFocus提升解释可靠性。**

- **链接: [https://arxiv.org/pdf/2605.20158](https://arxiv.org/pdf/2605.20158)**

> **作者:** Guangzhi Xiong; Qiao Jin; Sanchit Sinha; Zhiyong Lu; Aidong Zhang
>
> **摘要:** Large Vision Language Models (LVLMs) show promise in medical applications, but their inability to faithfully ground responses in visual evidence raises serious concerns about clinical trustworthiness. While visual attribution methods are widely used to explain LVLM predictions, whether these explanations actually reflect the visual evidence underlying the model's decision is largely unverified, since ground-truth annotations for internal model reasoning are typically unavailable. We address this question for chest X-ray (CXR) reasoning by developing a causal evaluation framework that retains only CXR-VQA samples for which the expert-annotated region is verified, via counterfactual editing, to be causally responsible for the model's prediction. Using this framework across 11 attribution methods, six open-source LVLMs, and two output modes (direct answer and step-by-step reasoning), we find that existing attribution methods often fail to identify the evidence used by LVLMs. To address this failure, we propose MedFocus, a concept-based attribution method that localizes clinically meaningful anatomical regions via unbalanced optimal transport and measures their causal effect on model outputs through targeted interventions. MedFocus produces spatial, concept-level, and token-level attributions and substantially outperforms prior methods, taking a step toward more trustworthy attribution for medical LVLMs. Our data and code are available at this https URL.
>
---
#### [new 084] SphericalDreamer: Generating Navigable Immersive 3D Worlds with Panorama Fusion
- **分类: cs.CV**

- **简介: 该论文属于3D环境生成任务，旨在解决生成可导航且全景覆盖的3D世界的问题。工作包括生成多张全景图像并融合为一致的3D环境。**

- **链接: [https://arxiv.org/pdf/2605.19974](https://arxiv.org/pdf/2605.19974)**

> **作者:** Antoine Schnepf; Karim Kassab; Flavian Vasile; Andrew Comport
>
> **备注:** Accepted at ICML 2026. Project page available at this https URL
>
> **摘要:** The generation of immersive and navigable 3D environments is increasingly prevalent with the growing adoption of virtual reality and 3D content. However, recent methods face a fundamental limitation: they cannot produce 3D worlds that simultaneously (i) are navigable over long-range spatial extents and (ii) cover the complete omnidirectional field of view ($360^\circ$ horizontally and $180^\circ$ vertically). To address this challenge, we introduce SphericalDreamer, a method for generating fully immersive and long-range 3D outdoor environments from textual prompts. Our approach is built on the generation of multiple panoramic images, which are subsequently lifted into 3D and fused together while maintaining visual and geometric consistency. SphericalDreamer produces highly detailed, fully immersive 3D environments, while substantially improving scale and navigability compared to prior approaches.
>
---
#### [new 085] Landscape-Awareness for Geometric View Diffusion Model
- **分类: cs.CV**

- **简介: 该论文属于视角估计任务，解决稀疏视图下视角准确估计问题。提出一种基于得分的方法优化损失景观，提升收敛性和效率。**

- **链接: [https://arxiv.org/pdf/2605.19865](https://arxiv.org/pdf/2605.19865)**

> **作者:** Yan-Ting Chen; Hao-Wei Chen; Tsu-Ching Hsiao; Chun-Yi Lee
>
> **备注:** CVPR2026
>
> **摘要:** Accurate camera viewpoint estimation under sparse-view conditions remains challenging, particularly in two-view scenarios. Recent approaches leverage diffusion models such as Zero123 to synthesize novel views conditioned on relative viewpoint, showing promising results when repurposed for viewpoint estimation via optimization with MSE loss. However, existing methods often suffer from nonconvex loss landscape with numerous local minima, making them sensitive to initialization and reliant on naive multistart strategies. We analyze these optimization challenges and visualize failure cases, showing that geometric ambiguities, such as symmetry and self-similarity, can mislead gradient-based updates toward incorrect viewpoints. To address these limitations, we propose a score-based method that reshapes the optimization landscape to guide updates toward the ground-truth viewpoint, followed by a refinement stage using a viewpoint-conditioned diffusion model. Experiments show that our method improves convergence, reduces reliance on brute-force sampling, and achieves competitive accuracy with higher sample-efficiency.
>
---
#### [new 086] Tango3D: Towards Alignment for Global and Local 2D-3D Correspondence
- **分类: cs.CV**

- **简介: 该论文提出Tango3D，解决2D-3D细粒度对齐与全局检索的联合任务，通过统一编码和训练策略实现像素到点的精确对应。**

- **链接: [https://arxiv.org/pdf/2605.19727](https://arxiv.org/pdf/2605.19727)**

> **作者:** Zebin He; Mingxin Yang; Shuhui Yang; Hanxiao Sun; Xintong Han; Chunchao Guo; Wenhan Luo
>
> **摘要:** Existing 3D foundation models typically align point clouds to frozen vision-language spaces like CLIP, which achieve strong cross-modal retrieval by compressing 3D shape into a global vector. However, this global-only alignment cannot establish fine-grained pixel-to-point correspondence. To solve this, we present Tango3D, a foundation model that unifies dense correspondence and global retrieval. We use a geometry-aware 2D visual backbone and a pretrained 3D VAE to encode images into 2D patches and point clouds into 3D tokens. These are mapped into a single shared space to achieve both local pixel-to-point alignment and global semantic alignment. To stabilize the joint learning of dense and global objectives, we introduce a three-stage progressive training strategy. Experiments show our model successfully achieves object-level pixel-to-point alignment while maintaining competitive global retrieval, a joint capability not offered by existing 3D foundation models. By establishing a fine-grained alignment feature space, Tango3D injects rich semantics into purely geometric 3D tokens, paving the way for a wide range of dense 3D downstream tasks.
>
---
#### [new 087] Stitched Value Model for Diffusion Alignment
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出StitchVM，用于将清洁图像的奖励模型迁移至噪声潜在空间，解决扩散模型对齐问题，提升效率与效果。**

- **链接: [https://arxiv.org/pdf/2605.19804](https://arxiv.org/pdf/2605.19804)**

> **作者:** Hyojun Go; Hyungjin Chung; Prune Truong; Goutam Bhat; Li Mi; Zhaochong An; Zixiang Zhao; Dominik Narnhofer; Serge Belongie; Federico Tombari; Konrad Schindler
>
> **备注:** Project page: this https URL
>
> **摘要:** For practical use, diffusion- or flow-based generative models must be aligned with task-specific rewards, such as prompt fidelity or aesthetic preference. That alignment is challenging because the reward is defined for clean output images, but the alignment procedure requires value function estimates at noisy intermediate latents. Existing methods resort to Tweedie-style or Monte Carlo approximations, trading off estimator bias against computational cost: Tweedie estimates are efficient but biased, while Monte Carlo estimates are more accurate but require expensive rollouts. A natural alternative would be a learned value function, but it remains an open question how to effectively train a strong and general value model specifically for noisy latents. Here, we propose StitchVM, a model stitching framework that efficiently transfers reward models pretrained for clean images to the noisy latent regime. StitchVM starts from an existing, truncated pixel-space reward model and attaches a frozen diffusion backbone to it as its head. From the pixel-space model, the resulting hybrid retains a carefully pretrained, robust reward capability; from the diffusion backbone, it inherits its native ability to handle noisy latents. The stitching procedure is exceptionally lightweight, e.g., stitching and finetuning CLIP ViT-L and SD 3.5 Medium takes only 10 GPU-hours. By lifting powerful pixel-space reward models to latent space, StitchVM opens up a new style of diffusion alignment: instead of rough, yet costly per-sample approximation of the value function, the correct function for the actual, noisy latents is constructed once and then amortized over many samples and iterations. We show that this approach yields improvements across a broad range of downstream steering and post-training methods: DPS becomes $3.2\times$ faster while halving peak GPU memory, and DiffusionNFT becomes $2.3\times$ faster.
>
---
#### [new 088] PhyWorld: Physics-Faithful World Model for Video Generation
- **分类: cs.CV; cs.AI; cs.ET; cs.LG; cs.MM**

- **简介: 该论文提出PhyWorld，用于视频生成中的物理一致性问题。属于视频生成任务，解决生成视频与物理规律不符的问题，通过两阶段训练提升视频连贯性和物理合理性。**

- **链接: [https://arxiv.org/pdf/2605.19242](https://arxiv.org/pdf/2605.19242)**

> **作者:** Pu Zhao; Juyi Lin; Timothy Rupprecht; Arash Akbari; Chence Yang; Rahul Chowdhury; Elaheh Motamedi; Arman Akbari; Yumei He; Chen Wang; Geng Yuan; Weiwei Chen; Yanzhi Wang
>
> **摘要:** World simulators can provide safe and scalable environments for training Physical AI systems before real-world deployment. Large video generation models are emerging as a promising basis for such simulators because they can generate diverse and realistic visual futures. However, using them as world simulators requires physically faithful video continuations, namely, generated videos that preserve the physical state implied by the conditioning input, and evolve in ways consistent with basic physical principles. We propose PhyWorld, a video generation world model designed to produce temporally coherent and physically faithful scene continuations through two-stage post-training. In the first stage, we improve video-to-video continuation with flow matching fine-tuning, encouraging stable visual attributes and coherent motion dynamics across frames. In the second stage, we align generated dynamics with physical principles using Direct Preference Optimization (DPO) over physics preference pairs, guiding the model toward outputs with higher physical plausibility. To evaluate PhyWorld, we use both standard video-quality benchmarks and a dedicated physical-faithfulness benchmark with per-law scoring. Experiments show that PhyWorld improves video consistency, achieving an average score of 0.769 on VBench compared with 0.756 or below for state-of-the-art baselines. PhyWorld also improves physical plausibility, reaching an average score of 3.09 on our physical-faithfulness benchmark compared with 2.99 for the strongest baseline. These results suggest that post-training large video generation models with continuation and physics-preference signals can make them more effective world simulators for Physical AI.
>
---
#### [new 089] Depth2Pose: A Pose-Based Benchmark for Monocular Depth Estimation without Ground-Truth Depth
- **分类: cs.CV**

- **简介: 该论文属于单目深度估计任务，旨在解决传统评估方法无法反映深度对下游任务效果的问题。提出Depth2Pose框架，通过相对相机位姿准确度评估深度质量。**

- **链接: [https://arxiv.org/pdf/2605.19797](https://arxiv.org/pdf/2605.19797)**

> **作者:** Viktor Kocur; Sithu Aung; Gabrielle Flood; Yaqing Ding; Lukas Bujnak; Torsten Sattler; Zuzana Kukelova
>
> **摘要:** Monocular depth estimation has improved significantly in recent years, driven by increasingly powerful models and large-scale training data. Predicted depth is increasingly used as an input signal for downstream tasks such as Structure-from-Motion (SfM), visual localization, and SLAM. However, monocular depth estimators (MDEs) are still primarily evaluated in terms of depth accuracy. Standard metrics aggregate errors globally and may not reflect the usefulness of depth for downstream geometric tasks. We therefore propose Depth2Pose, a framework for evaluating MDEs in the context of downstream tasks. By combining depth predictions with feature correspondences in depth-aware geometric solvers, we use relative camera pose estimation accuracy as a task-driven proxy for depth quality. Traditional benchmarks require dense ground truth in the form of per-pixel depth, which is expensive to obtain. In contrast, our formulation requires only camera poses, which can be estimated efficiently, e.g., using Structure-from-Motion pipelines. As a result, our framework can be applied to scenes where ground-truth depth is difficult to obtain, for example due to large scene scale or heavy occlusions (e.g., vegetated environments). Leveraging this, we introduce the D2P dataset, which contains challenging scenes outside the distribution of commonly used training data. We show that methods performing well under standard depth error metrics on existing benchmarks also perform well under our pose-based metric when evaluated on the same datasets, but do not necessarily generalize to our more challenging dataset. Finally, we provide a simple and extensible evaluation framework. The dataset and code are available at this http URL.
>
---
#### [new 090] Benchmarking and Evolving Reason-Reflect-Rectify for Reflective Visual Generation
- **分类: cs.CV**

- **简介: 该论文属于视觉生成任务，旨在解决复杂提示下生成质量不足的问题。提出R^3框架与基准测试，引入R^3-Refiner提升迭代修正能力。**

- **链接: [https://arxiv.org/pdf/2605.19639](https://arxiv.org/pdf/2605.19639)**

> **作者:** Junjie Wang; Xinghua Lou; Jason Li; Ye Tian; Keyu Chen; Yulin Li; Bin Kang; Jacky Mai; Yanwei Li; Zhuotao Tian; Liqiang Nie
>
> **摘要:** Text-to-Image (T2I) models and Unified Multimodal Models (UMMs) have achieved remarkable progress in visual generation. However, their reliance on a single-pass generation paradigm limits their ability to handle complex prompts requiring iterative refinement. To enable multi-round Reflective Visual Generation (RVG), we formalize the Reason-Reflect-Rectify (R^3) loop as a core framework and introduce R^3-Bench, a benchmark of over 600 expert-annotated instances that quantifies iterative reasoning and rectification capabilities. Evaluation on R^3-Bench reveals a critical gap: while state-of-the-art models can identify generation errors, they fail to generate actionable rectification instructions. To bridge this gap, we propose R^3-Refiner, a dual-stage framework leveraging Group Relative Policy Optimization (GRPO) and a Hierarchical Reward Mechanism (HRM) to better align rectification with reflective reasoning. Experiments show that R^3-Refiner achieves significant improvements on R^3-Bench (+12.0% in Reflective Verdict Score, +9.0% in Rectification Score), and can be seamlessly integrated with various MLLMs to enhance the generation quality of different T2I models on GenEval++ and T2I-CompBench. Code is available at this https URL.
>
---
#### [new 091] Structured Layout Priors for Robust Out-of-Distribution Visual Document Understanding
- **分类: cs.CV**

- **简介: 该论文属于文档理解任务，解决VLM在布局不常见时性能下降的问题。通过引入结构化布局先验，提升模型对分布外文档的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.19866](https://arxiv.org/pdf/2605.19866)**

> **作者:** Peter El Hachem; Ahmed Nassar; A. Said Gurbuz; Christoph Auer; Peter W. J. Staar
>
> **备注:** 18 pages, 7 figures. Main text: 9 pages (4 figures); Appendix: 9 pages (3 figures)
>
> **摘要:** Vision-Language Models (VLMs) parse documents end-to-end but frequently break down on layouts unlike those seen in training. We attribute this to a two-hop bottleneck: before the decoder can extract content (Hop 2), it must first classify and localize the enclosing layout entity (Hop 1), and when the first hop fails the second collapses into omissions, malformed structure, or autoregressive repetition. We pre-resolve Hop 1 outside the decoder by running a lightweight RT-DETR detector, serializing its outputs in the parser's native DocTags vocabulary, and injecting them into the prompt alongside the full page image. Unlike analyze-then-parse approaches that crop the page, or prior prompt-level priors written in plain text, our prior shares the decoder's generation space and leaves the global image in view as a fallback when detections are noisy. On a 10k-page structural out-of-distribution benchmark, markdown F1 rises from $0.37$ to $0.92$; on the Chinese subset of OmniDocBench, table TEDS rises from $0.01$ to $0.36$; and on the 26k-page ViDoRe V3 benchmark, infinite-loop decoding failures drop across every industrial domain tested. These gains cost $15\%$ wall-clock latency and a median of $74$ prompt tokens, with no architectural change to the base VLM. An attention-level analysis further reveals a bimodal phase shift in which the decoder attends to injected layout tokens when emitting structure and to image patches when emitting content, consistent with the two-hop bottleneck being alleviated. Model weights will be released to support reproducibility.
>
---
#### [new 092] GoTTA be Diverse: Rethinking Memory Policies for Test-Time Adaptation
- **分类: cs.CV**

- **简介: 该论文属于测试时适应（TTA）任务，旨在解决记忆策略对适应效果的影响问题。通过系统评估不同记忆策略，提出一种关注类内多样性的GOTTA方法，提升适应性能。**

- **链接: [https://arxiv.org/pdf/2605.19890](https://arxiv.org/pdf/2605.19890)**

> **作者:** Shyma Alhuwaider; Yasmeen Alsaedy; Merey Ramazanova; Silvio Giancola; Bernard Ghanem
>
> **摘要:** Test-time adaptation (TTA) enables a pre-trained model to adapt online to an unlabeled test stream under distribution shift. While most TTA research focuses on the adaptation objective, practical streams also depend critically on the memory used to select which test samples drive adaptation. Existing memory mechanisms are usually evaluated as components of specific TTA algorithms, making it difficult to isolate which memory design choices matter and when they matter. In this work, we provide a systematic benchmark that decouples memory from the adaptation algorithm and evaluates memory policies under unified conditions across i.i.d., non-i.i.d., continual, and practical test streams. Our study shows that effective memory management requires more than retaining recent or class-balanced samples. In particular, intra-class diversity is a key factor for avoiding redundant buffers and maintaining representative adaptation signals under temporally correlated and label-skewed streams. Motivated by this finding, we introduce Guided Observational Test-Time Adaptation (GOTTA), a family of diversity-aware memory policies that combine class-balanced allocation with feature-space diversity. GOTTA memories act as drop-in replacements for existing buffers and can be paired with different TTA objectives. Across corruption benchmarks and video-stream settings, diversity-aware memory improves adaptation most clearly under constrained memory budgets and challenging non-i.i.d. streams, while remaining competitive as memory capacity increases. These results highlight memory management as a first-class component of robust test-time adaptation and identify diversity as a central principle for practical TTA.
>
---
#### [new 093] White-Balance First, Adjust Later: Cross-Camera Color Constancy via Vision-Language Evaluation
- **分类: cs.CV**

- **简介: 该论文属于颜色恒常性任务，解决跨相机泛化问题。提出VLM-CC框架，通过视觉语言模型反馈迭代修正光照估计，提升跨相机颜色一致性。**

- **链接: [https://arxiv.org/pdf/2605.19613](https://arxiv.org/pdf/2605.19613)**

> **作者:** Shuwei Li; Lei Tan; Robby T. Tan
>
> **备注:** In CVPR 2026
>
> **摘要:** Color constancy aims to keep object colors consistent under varying illumination. Cross-camera generalization in color constancy remains challenging because learning-based models often overfit to the color response characteristics of the training camera, resulting in degraded performance on images captured by other cameras. We propose VLM-CC, a feedback-guided framework that formulates color constancy as an iterative refinement process. Instead of directly estimating the illuminant from raw input, VLM-CC performs iterative correction driven by vision-language model (VLM)-based evaluation. At each iteration, the image is white-balanced using the current estimate and converted to pseudo-sRGB. A lightweight LoRA-tuned VLM then assesses the corrected image, identifying the dominant residual color cast and providing qualitative feedback. This feedback is mapped to a residual illumination direction (red, green, or blue) and used to update the illuminant estimate until convergence. Our key idea is to reframe color constancy as an iterative perceptual feedback problem, leveraging VLM evaluation instead of direct RGB regression. By replacing direct RGB estimation with VLM-guided perceptual feedback, VLM-CC achieves state-of-the-art robustness in cross-camera color constancy across multiple datasets. Code will be available at this https URL.
>
---
#### [new 094] Quantized Machine Learning Models for Medical Imaging in Low-Resource Healthcare Settings
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学图像分析任务，旨在解决低资源环境下深度学习模型部署困难的问题。通过模型压缩技术提升模型效率，实现轻量级脑肿瘤分类。**

- **链接: [https://arxiv.org/pdf/2605.19207](https://arxiv.org/pdf/2605.19207)**

> **作者:** Sumanth Meenan Kanneti; Aryan Shah
>
> **摘要:** Deep learning models have shown strong performance in medical image analysis, but deploying them in low-resource clinical environments remains difficult due to computational, memory, and power constraints. This paper presents a multi-strategy compression framework for brain tumor classification from MRI, encompassing quantization-aware training, knowledge distillation from a DenseNet-101 teacher to a compact DenseNet-32 student with low-bit post-training quantization, and Float16 post-training quantization on a lightweight MobileNetV2 backbone. Using a multi-class brain tumor MRI dataset containing glioma, meningioma, pituitary tumors, and healthy controls, we provide full experimental validation of the MobileNetV2-based pipeline, training the classifier through a three-stage transfer learning process and applying Float16 quantization via TensorFlow Lite. The DenseNet-based distillation and quantization-aware training strategies are described as complementary compression approaches within the framework, with their complete empirical evaluation reserved for future work. Experimental results on the MobileNetV2 pipeline show that the quantized model achieves 82.37 percent validation accuracy compared to the 82.20 percent full-precision baseline, reducing model size from 35.34 MB to 5.76 MB, a 6.14x compression ratio with no meaningful accuracy loss. Per-class evaluation confirms that quantization preserves diagnostic performance uniformly across all four tumor categories. These findings demonstrate that lightweight quantized models can deliver clinically viable brain tumor screening in resource-constrained healthcare settings.
>
---
#### [new 095] TextAlign: Preference Alignment for Text Rendering with Hierarchical Rewards
- **分类: cs.CV; cs.DB**

- **简介: 该论文属于文本渲染任务，解决大模型文本生成不准确的问题。提出AlignText框架，通过层次化奖励机制提升文本准确性，无需修改模型结构。**

- **链接: [https://arxiv.org/pdf/2605.19320](https://arxiv.org/pdf/2605.19320)**

> **作者:** Mingxuan Cui; Jingpu Yang; Fengxian Ji; Qian Jiang; Zhecheng Shi; Jiaming Wang; Zirui Song; Fajri Koto; Xiuying Chen
>
> **摘要:** Faithful text rendering remains a persistent weakness of large text-to-image generative models, as it requires both semantic instruction following and fine-grained glyph-level structure. Prior methods often improve this ability through architecture-specific modules or encoder modifications, which complicate deployment across foundation models. We study text rendering as a post-training preference-alignment problem and propose TextAlign, a non-invasive framework that keeps the generator architecture unchanged. The key component is a hierarchical vision-language model (VLM)-based reward that decomposes rendering errors into global, word, and glyph levels, then converts binary defect judgments into a scalar preference signal. The resulting signal supports both Group Relative Policy Optimization (GRPO) and Direct Preference Optimization (DPO). Experiments on FLUX.1-dev and Z-Image-Turbo show consistent gains in OCR-based text accuracy without degrading general generation quality. Compared with strong foundation and text-rendering baselines, including SD3.5, Qwen-Image, AnyText, and TextDiffuser, these results indicate that reward design offers a scalable alternative to model redesign for improving text rendering.
>
---
#### [new 096] EpiDiffVO: Geometry-Aware Epipolar Diffusion for Robust Visual Odometry
- **分类: cs.CV**

- **简介: 该论文属于视觉里程计任务，旨在解决图像对中相对位姿估计的问题。通过稀疏匹配、扩散精修和图结构选择，减少冗余并提升估计鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.19556](https://arxiv.org/pdf/2605.19556)**

> **作者:** Prateeth Rao
>
> **备注:** 8 pages, 5 figures, in revision to be submitted to IEEE RA-L
>
> **摘要:** Estimating relative pose from image pairs fundamentally requires only a minimal subset of geometrically consistent correspondences. However, most learning-based approaches rely on dense matching or direct regression, leading to redundancy and reduced geometric interpretability. In this work, we propose a sparse epipolar matching framework that predicts a compact set of correspondences optimized for geometric consistency across varying temporal baselines. To address residual noise and misalignment, we introduce an epipolar diffusion process that models correspondence uncertainty and refines keypoints toward epipolar consistency. The refined correspondences, along with depth cues, are lifted into a graph representation forming a Steiner graph that encodes relational structure between points. A graph neural network learns a compact subset of informative correspondences, which are passed to a differentiable singular value decomposition solver for end-to-end geometric estimation. Relative pose is recovered from the resulting essential matrix and evaluated in a visual odometry setting on the TartanAir and KITTI SLAM datasets. Experimental results demonstrate that combining sparse matching, diffusion-based refinement, and graph-based subset selection reduces correspondence redundancy while maintaining robust pose estimation across challenging baselines.
>
---
#### [new 097] Trust It or Not: Evidential Uncertainty for Feed-Forward 3D Reconstruction with Trust3R
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决现有方法预测不确定性不足的问题。提出Trust3R框架，提供概率性不确定性估计，提升几何准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2605.19539](https://arxiv.org/pdf/2605.19539)**

> **作者:** Zihao Zhu; Wenyuan Zhao; Nuo Chen; Chao Tian; Zhiwen Fan
>
> **备注:** Accepted at ICML 2026. 10 pages main paper, with appendix
>
> **摘要:** Geometric foundation models hold promise for unconstrained dense geometry prediction from uncalibrated images. However, in current feed-forward designs, their predicted confidence scores are heuristic, lack probabilistic interpretation, and often fail to indicate where and how much the predicted geometry can be trusted. To address this gap, we present Trust3R, a lightweight evidential uncertainty framework for feed-forward 3D reconstruction. Trust3R combines gated residual mean refinement with a Normal-Inverse-Wishart evidential head, yielding a closed-form multivariate Student-t distribution for per-point geometric uncertainty. This design provides probabilistically grounded pointmap uncertainty estimates while adding moderate inference overhead. We evaluate on diverse indoor and outdoor benchmarks and compare against MASt3R's built-in confidence map as well as common uncertainty-aware baselines spanning single-pass heteroscedastic regression and sampling-based methods such as MC dropout and deep ensembles. Experimental results show that Trust3R consistently improves risk-coverage and sparsification, and generally improves geometric accuracy. These gains are reflected in stronger uncertainty ranking across benchmarks, with 25% lower AURC and 41% lower AUSE on ScanNet++, providing a practical reliability signal for uncertainty-aware weighting in downstream geometry pipelines. The project page and code are available at this https URL.
>
---
#### [new 098] P2DNav: Panorama-to-Downview Reasoning for Zero-shot Vision-and-Language Navigation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言导航任务，解决零样本环境下导航决策不稳定的问题。提出P2DNav框架，通过分解导航步骤提升性能。**

- **链接: [https://arxiv.org/pdf/2605.19634](https://arxiv.org/pdf/2605.19634)**

> **作者:** Kai Sheng; Liuyi Wang; Haojie Dai; Jinlong Li; Yongrui Qin; Zongtao He; Chengju Liu; Qijun Chen
>
> **摘要:** Vision-and-language navigation (VLN) requires an embodied agent to ground natural-language instructions into executable navigation actions in unseen environments. Existing zero-shot methods typically rely on additional waypoint prediction modules, which often entangle high-level directional reasoning with fine-grained local grounding, leading to error-prone and unstable decisions. In this paper, we propose P2DNav, a hierarchical framework for zero-shot vision-and-language navigation. P2DNav consists of three core components: Panorama-to-Downview (P2D), Sliding-Window Dialogue Memory (SDM), and Reflective Reorientation Mechanism (RRM). P2D explicitly decomposes navigation decision-making into two stages: panoramic direction selection and downview local grounding. It first selects the instruction-relevant direction from a 360° panorama, and then predicts a pixel-level target point from the downview RGB observation in that direction. In addition, SDM organizes navigation history as a multi-turn dialogue context and maintains recent visual observations within a sliding window to support long-horizon navigation. RRM further enables reflective reorientation by assessing the reliability of local grounding based on the downview observation and returning to panoramic direction selection when necessary. Experiments on the R2R-CE benchmark show that P2DNav achieves strong performance among zero-shot methods. In particular, compared with the state-of-the-art (SOTA) zero-shot waypoint-based and waypoint-free methods, P2DNav achieves SR gains of 146.6% and 58.9%, respectively, demonstrating the effectiveness of P2D, SDM, and RRM for zero-shot VLN. Code will be released for public use.
>
---
#### [new 099] Multi-axis Analysis of Image Manipulation Localization
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像篡改检测任务，旨在解决跨视觉领域检测高级篡改的问题。提出AUDITS基准，涵盖多种篡改类型和尺寸，评估现有方法的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.20174](https://arxiv.org/pdf/2605.20174)**

> **作者:** Keanu Nichols; Divya Appapogu; Giscard Biamby; Dina Bashkirova; Anna Rohrbach; Bryan A. Plummer
>
> **备注:** 28 pages, 5 figures, 5 tables
>
> **摘要:** Advanced image editing software enables easy creation of highly convincing image manipulations, which has been made even more accessible in recent years due to advances in generative AI. Manipulated images, while often harmless, could spread misinformation, create false narratives, and influence people's opinions on important issues. Despite this growing threat, there is limited research on detecting advanced manipulations across different visual domains. Thus, we introduce Analysis Under Domain-shifts, qualIty, Type, and Size (AUDITS), a comprehensive benchmark designed for studying axes of analysis in image manipulation detection. AUDITS comprises over 530K images from two distinct sources (user and news photos). We curate our dataset to support analysis across multiple axes using recent diffusion-based inpaintings, spanning a diverse range of manipulation types and sizes. We conduct experiments under different types of domain shift to evaluate robustness of existing image manipulation detection methods. Our goal is to drive further research in this area by offering new insights that would help develop more reliable and generalizable image manipulation detection methods.
>
---
#### [new 100] Dual-Prompt CLIP with Hybrid Visual Encoders for Occluded Person Re-Identification
- **分类: cs.CV**

- **简介: 该论文属于行人重识别任务，解决遮挡情况下跨摄像头匹配问题。提出DPL-ReID模型，结合双提示学习和真实遮挡增强，提升遮挡场景下的识别性能。**

- **链接: [https://arxiv.org/pdf/2605.19527](https://arxiv.org/pdf/2605.19527)**

> **作者:** Zhangjian Ji; Shaotong Qiao; Kai Feng; Wei Wei
>
> **摘要:** Occluded person re-identification focuses on matching partially visible pedestrians across multiple camera views. However, occlusions disrupt body-region cues, thereby complicating cross-view matching. Most person ReID methods built on pretrained vision-language models only focus on enhancing prompt-based feature learning while ignoring the semantic information of occluders. Based on the success of CLIP-ReID, we propose a novel Dual Prompt Learning ReID (DPL-ReID) model for occluded person ReID. It incorporates a Dual Prompt Learning (Dual-PL) strategy, which can utilize textual cues to capture complete pedestrian semantics and keep robustness against occlusion, and a Real-World Occlusion Augmentation (RWOA) method that realistically simulates occlusion scenarios encountered in real word to enrich occluded samples. In addition, we also design a Weighted Gated Feature Fusion (WGFF) method, which in corporates LSNet to capture global information and act as a feature-gating mechanism. This mechanism can effectively guide the CLIP visual encoder toward generating more comprehensive feature representations. Extensive experiments on several benchmark occluded ReID datasets show that our proposed DPL-ReID achieves the state-of-the art performance. The occlusion instance library are available at this https URL.
>
---
#### [new 101] A Framework for Evaluating Zero-Shot Image Generation in Concept-based Explainability
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于概念解释型XAI任务，旨在解决标注数据依赖过高的问题。通过零样本文本生成图像模型构建合成概念数据集，评估其在解释任务中的有效性与可靠性。**

- **链接: [https://arxiv.org/pdf/2605.19855](https://arxiv.org/pdf/2605.19855)**

> **作者:** Giacomo Astolfi; Matteo Bianchi; Riccardo Campi; Antonio De Santis; Marco Brambilla
>
> **备注:** G. Astolfi, M. Bianchi, and R. Campi contributed equally
>
> **摘要:** Concept-based Explainable Artificial Intelligence (XAI) interprets deep learning models using human-understandable visual features (e.g., textures or object parts) by linking internal representations to class predictions, thereby bridging the gap between low-level image data and high-level semantics. A major challenge, however, is the reliance on large sets of labeled images to represent each concept, which limits scalability. In this work, we investigate the use of zero-shot Text-to-Image (T2I) generative models as a source of synthetic concept datasets for concept-based XAI methods. Specifically, we generate concepts using predefined prompts and evaluate their faithfulness to real ones through four complementary analyses: (1) comparing synthetic vs. real concept images via concept representation similarity; (2) evaluating their intra-similarity by comparing pairs of subsets of the same concept with progressively increasing size; (3) evaluating their performance for downstream explanation tasks using relevant class images; (4) evaluating how removing a concept from tested class images affects explanations of generated concepts. While current T2I generative models promise a shortcut to concept-based XAI, our study highlights challenges and raises open questions about the use of synthetic data generated by zero-shot pipelines in model analyses. The resulting dataset is available at this https URL.
>
---
#### [new 102] Preferences Order, Ratings Anchor: From Fused Expert Aesthetic Ground Truth to Self-Distillation
- **分类: cs.CV**

- **简介: 该论文属于图像美学评估任务，解决现有基准仅采用单一标注协议的问题。通过构建PPaint基准，融合偏好与评分数据，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.19776](https://arxiv.org/pdf/2605.19776)**

> **作者:** Yuanpei Zhao; Jie Lin; Chao Zhang; Yilin Wang; Mao Li; Chenhui Li; Jie Hou; Tangjie Lv
>
> **备注:** 27 pages, 7 pages
>
> **摘要:** Pairwise preferences and pointwise ratings are the two dominant annotation protocols in image aesthetic assessment (IAA), yet existing benchmarks adopt only one, leaving their complementarity unmeasured under controlled conditions. We introduce PPaint, a matched dual-protocol benchmark in which 15 domain experts, 5 per category, annotate 150 Chinese paintings under both protocols across five aesthetic dimensions, collecting 45,900 pairwise expert judgments through a locally dense preference design alongside the matched ratings. The matched design reveals complementary strengths: preferences yield more consistent ordinal rankings, while ratings anchor the absolute score scale. Fusing both signals via two independent preference-to-score methods yields a fused expert ground truth on which the two constructions converge to nearly identical scores. The same preference-to-score principle extends to label-free VLM training. PSDistill converts VLM pairwise judgments into calibrated pseudo-scores via an Elo reference pool, and trains the same VLM with confidence-weighted ranking optimization to produce a single-pass aesthetic scorer. Trained on a single painting category, the distilled Qwen3-VL-8B improves mean SRCC from 0.504 to 0.709 across all three categories, outperforming all open-source baselines including the dedicated aesthetic model ArtiMuse and matching closed-source Gemini-3.1-Pro within 0.04 SRCC at single-pass inference cost, with cross-domain transfer further validated on APDDv2. We will release the full PPaint dataset and training code.
>
---
#### [new 103] CutVerse: A Compositional GUI Agents Benchmark for Media Post-Production Editing
- **分类: cs.CV; cs.AI; cs.GR; cs.HC**

- **简介: 该论文提出CutVerse，一个用于评估GUI代理在媒体后期制作任务中表现的基准。旨在解决专业创意流程中GUI代理能力不足的问题，通过构建复杂任务数据集和解析工具进行系统评估。**

- **链接: [https://arxiv.org/pdf/2605.19484](https://arxiv.org/pdf/2605.19484)**

> **作者:** Haobo Hu; Xiangwu Guo; Zhiheng Chen; Difei Gao; Haotian Liu; Libiao Jin; Qi Mao
>
> **摘要:** While GUI agents have made significant progress in web navigation and basic operating system tasks, their capabilities in professional creative workflows remain largely underexplored. To bridge this gap, we introduce Cutverse, a benchmark designed to systematically evaluate autonomous GUI agents in realistic media post-production environments. We curate expert demonstrations across 7 professional applications (e.g., Premiere Pro, Photoshop), covering 186 complex, long-horizon tasks grounded in authentic editing workflows, involving dense multimodal interfaces and tightly coupled interaction sequences. To support scalable evaluation, we develop a lightweight parser that transforms raw screen recordings and low-level interaction logs into structured, compositional GUI action trajectories with precise grounding. Extensive evaluations reveal that existing agents achieve only 36.0\% task success on realistic media editing tasks, underscoring the challenges posed by complex, long-horizon media post-production workflows in our this http URL current models demonstrate promising spatial grounding, multimodal alignment, and coordinated action execution, they remain limited in long-horizon reliability and domain-specific planning.
>
---
#### [new 104] SWEET: Sparse World Modeling with Image Editing for Embodied Task Execution
- **分类: cs.CV**

- **简介: 该论文属于机器人操作任务，旨在通过图像编辑构建稀疏视觉世界模型，解决传统密集视频生成计算成本高、不必要的问题。工作包括提出SWEET框架，实现从语言指令到可执行动作的规划。**

- **链接: [https://arxiv.org/pdf/2605.19319](https://arxiv.org/pdf/2605.19319)**

> **作者:** Yiren Song; Yihan Wang; Xiyao Deng; Zhuoran Yan; Mike Zheng Shou
>
> **摘要:** Visual prediction has emerged as a promising paradigm for embodied control, where future observations are generated and then translated into actions. However, dense video generation is computationally expensive and often unnecessary for many manipulation tasks, whose progress can be summarized by a small number of task-relevant visual states. In this work, we study whether image editing models can serve as sparse visual world models for robot manipulation by predicting task-level future states without dense video rollout. We first conduct a controlled comparison between the video generation model Wan2.2 and the image editing model FLUX-Kontext under the same robotic data setting, and find that image editing produces more reliable task-level keyframes with better visual fidelity and substantially lower inference cost. Motivated by this observation, we propose SWEET, a one-shot sparse visual planning framework that progressively generates a sequence of task-relevant manipulation keyframes through successive image editing, conditioned on language instructions and optional arrow-based spatial guidance. A goal-conditioned diffusion action predictor then converts adjacent imagined keyframes into executable action chunks. To reduce the mismatch between real and edited visual subgoals, we further introduce a mixed-training strategy with filtered edited targets. Experiments on DROID and RoboMimic show that SWEET improves keyframe prediction across seen and unseen scenes and enables a full pipeline from sequential keyframe planning to executable robot actions, suggesting that image editing is a promising and underexplored direction for embodied visual prediction.
>
---
#### [new 105] Vision Harnessing Agent for Open Ad-hoc Segmentation
- **分类: cs.CV**

- **简介: 该论文属于开放性分割任务，解决无先验知识的动态概念分割问题。提出VASA模型，通过视觉引导和工作掩码实现有效分割。**

- **链接: [https://arxiv.org/pdf/2605.19410](https://arxiv.org/pdf/2605.19410)**

> **作者:** Zilin Wang; Stella X. Yu
>
> **备注:** 23 pages, 11 figures
>
> **摘要:** Segmentation has become easy when the concept is known, requiring retrieval of a learned visual grounding from text. It remains hard for open ad-hoc concepts, where the grounding may not exist as one learned mask and must often be constructed from image evidence through parts, relations, exclusions, and collections. We propose a Vision-guided Ad-hoc Segmentation Agent (VASA), the first vision harnessing agent for open ad-hoc segmentation. VASA is training-free and couples a VLM agent, a segmentation foundation model, and a visually grounded workflow. Rather than revising text prompts alone, VASA uses a persistent working mask to reason, construct, and validate a solution. It plans visual operations, invokes segmentation tools, inspects results, edits the mask, and recovers from errors. We construct PARS, a new benchmark that turns part-level labels in PartImageNet into open ad-hoc concepts through long-form definition queries. On PARS, VASA outperforms open-vocabulary, reasoning-based, and agentic baselines, surpassing SAM3 Agent by 14-25%. On RefCOCOm, a standard multi-granularity referring segmentation benchmark, VASA improves over SAM3 Agent by 5-9% and over other agentic baselines by up to 20%. These results validate agentic visual construction for open ad-hoc segmentation. Our work points to a path for AI agents beyond wrapping foundation models as tools: Programming them with task knowledge, VLM behavior, visual routines, working memory, and failure-aware workflows.
>
---
#### [new 106] Probability-Conserving Flow Guidance
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **简介: 该论文属于图像生成任务，解决生成模型中引导方法导致概率不守恒的问题。通过分析连续性方程，提出AdaMaG方法，有效控制生成过程，提升真实性和稳定性。**

- **链接: [https://arxiv.org/pdf/2605.20079](https://arxiv.org/pdf/2605.20079)**

> **作者:** Parsa Esmati; Junha Hyung; Amirhossein Dadashzadeh; Jaegul Choo; Majid Mirmehdi
>
> **摘要:** Diffusion and flow-based generative models dominate visual synthesis, with guidance aligning samples to user input and improving perceptual quality. However, Classifier-Free Guidance (CFG) and extrapolation-based methods are heuristic linear combinations of velocities/scores that ignore the generative manifold geometry, breaking probability conservation and driving samples off the learned manifold under strong guidance. We analyse guidance through the continuity equation and show its effect decomposes into a divergence term and a score-parallel term defined invariantly across parameterisations. We prove the divergence term blows up structurally as sampling approaches the data manifold, motivating a time-dependent schedule alongside score-parallel attenuation. The resulting plug-and-play rule, Adaptive Manifold Guidance (AdaMaG), bounds both terms at no additional inference cost. Finally, we show that most empirical heuristics for reducing saturation or improving generation quality correspond directly to the two terms in our decomposition. Across image generation benchmarks, AdaMaG improves realism, reduces hallucinations, and induces controlled desaturation in high-guidance regimes.
>
---
#### [new 107] CPC-VAR:Continual Personalized and Compositional Generation in Visual Autoregressive Models
- **分类: cs.CV**

- **简介: 该论文属于视觉自回归模型的个性化生成任务，解决持续个性化和多概念合成中的遗忘与特征纠缠问题，提出GCNS和上下文感知组合策略。**

- **链接: [https://arxiv.org/pdf/2605.19750](https://arxiv.org/pdf/2605.19750)**

> **作者:** Junhao Li; Xinhao Zhong; Yi sun; Yuxia Qiao; Bin Chen; Shu-Tao Xia; Yaowei Wang
>
> **摘要:** Visual autoregressive (VAR) models have recently emerged as an efficient paradigm for text-to-image generation. Despite their strong generative capability, existing VAR-based personalization methods remain limited to static settings, failing to accommodate evolving user demands. In particular, sequential concept learning leads to severe catastrophic forgetting, while multi-concept synthesis often suffers from feature entanglement and attribute inconsistency. In this work, we present the first systematic study of continual personalized generation in VAR models. We identify two key challenges: (i) preserving previously learned concepts during sequential customization, and (ii) composing multiple personalized concepts in a controllable manner. To address these issues, we propose a unified framework with two core components. For continual single-concept learning, we introduce Gradient-based Concept Neuron Selection (GCNS), which identifies concept-relevant neurons and constrains only conflicting parameters across tasks, effectively mitigating forgetting without additional model expansion. For multi-concept synthesis, we propose a context-aware composition strategy that performs multi-branch feature modeling and localized cross-attention fusion guided by spatial conditions, enabling precise and disentangled concept composition. Extensive experiments demonstrate that our method significantly improves performance in long-sequence continual personalization while achieving superior results in multi-concept image synthesis compared to existing baselines. These findings highlight the potential of VAR models for scalable and controllable personalized generation.
>
---
#### [new 108] Artifact-Bench: Evaluating MLLMs on Detecting and Assessing the Artifacts of AI-Generated Videos
- **分类: cs.CV**

- **简介: 该论文提出Artifact-Bench，用于评估MLLMs在检测和分析AI生成视频伪影方面的能力。任务是解决MLLM在视频真实性感知与推理上的不足，通过构建基准测试，揭示其性能局限。**

- **链接: [https://arxiv.org/pdf/2605.18984](https://arxiv.org/pdf/2605.18984)**

> **作者:** Yuqi Tang; Yang Shi; Zhuoran Zhang; Qixun Wang; Xuehai Bai; Yue Ding; Ruizhe Chen; Bohan Zeng; Xinlong Chen; Xuanyu Zhu; Bozhou Li; Yuran Wang; Yifan Dai; Chengzhuo Tong; Xinyu Liu; Yiyan Ji; Yujie Wei; Yuhao Dong; Shilin Yan; Fengxiang Wang; Yi-Fan Zhang; Haotian Wang; Yuanxing Zhang; Pengfei Wan
>
> **摘要:** Recent video generative models have greatly improved the realism of AI-generated videos, yet their outputs still exhibit artifacts such as temporal inconsistencies, structural distortions, and semantic incoherence. While Multimodal Large Language Models (MLLMs) show strong visual understanding capabilities, their ability to perceive and reason about such artifacts remains unclear. Existing benchmarks often lack systematic evaluation of artifact-aware perception and fine-grained diagnostic reasoning, especially across diverse AI-generated video domains beyond photorealistic content. To address this gap, we introduce Artifact-Bench, a comprehensive benchmark for evaluating MLLMs on AI-generated video artifact detection and analysis. We first establish a three-level hierarchical taxonomy of realism artifacts, covering photorealistic, animated, and CG-style videos. Based on this taxonomy, Artifact-Bench defines three complementary tasks: real vs. AI-generated video classification, pairwise realism comparison, and fine-grained artifact identification. Experiments on 19 leading MLLMs reveal substantial limitations in artifact perception and reasoning, with many models approaching random or even below-random performance in challenging settings. We further observe significant misalignment between MLLM judgments and human perceptual preferences, highlighting their limited reliability as general evaluators for AI-generated video realism.
>
---
#### [new 109] Harnessing Self-Supervised Features for Art Classification
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文属于艺术分类任务，旨在提升绘画分类性能。通过对比监督与自监督模型，验证自监督特征提取的有效性，并探讨其在VR等实际应用中的潜力。**

- **链接: [https://arxiv.org/pdf/2605.18974](https://arxiv.org/pdf/2605.18974)**

> **作者:** Federico Melis; Davide Bilardello; Emanuele Prato; Evelyn Turri; Lorenzo Baraldi
>
> **备注:** IRCDL 2026
>
> **摘要:** Classifying artworks presents a significant challenge due to the complex interplay of fine-grained details and abstract features that condition the style or genre of an artwork. This paper presents a systematic investigation of the effectiveness of supervised and self-supervised backbones as feature extractors for both artwork classification and retrieval, with a particular focus on paintings. We conduct an extensive experimental evaluation using the DINO family and CLIP models, assessing multiple classification strategies and feature representations. Our results demonstrate that employing a self-supervised backbone leads to consistent improvements in artwork classification performance. Moreover, our work provides insights into the applicability of classification and retrieval modules in real-world applications, such as virtual reality (VR) applications that support museum navigation.
>
---
#### [new 110] Distribution Matching Distillation without Fake Score Network
- **分类: cs.CV**

- **简介: 该论文属于生成模型任务，解决DMD中依赖辅助fake-score网络的问题。提出FSF-DMD，利用生成器自身结构替代fake-score网络，提升效率与效果。**

- **链接: [https://arxiv.org/pdf/2605.19256](https://arxiv.org/pdf/2605.19256)**

> **作者:** Youngjoong Kim; Deokyeong Lee; Jaesik Park
>
> **摘要:** Distribution Matching Distillation (DMD) provides an effective distribution-level correction for few-step generation, while relying on an auxiliary fake-score network to track the evolving generative distribution. Recent work combines DMD-style objectives with flow-map generators to exploit both forward-divergence training and reverse-divergence correction. The fake-score estimator remains an additional component with memory and update overhead. In this work, we study whether this explicit tracker can be avoided when the generator itself has a flow-map structure. We propose Fake-Score-network-Free DMD (FSF-DMD), a DMD formulation for flow-map generators that replaces the auxiliary fake-score estimator with a generator-induced pseudo-velocity surrogate. The key observation is that the endpoint pseudo-velocity of a flow-map generator provides a tractable proxy for fake-velocity estimation, allowing the generator itself to supply the reverse-divergence signal. Building on this observation, we derive a practical objective, extend it with flow-map-consistent backward simulation, and introduce a self-teacher variant for training from scratch. In our ImageNet-1K $256 \times 256$ experiments, FSF-DMD improves flow-map baselines, reaches lower FID than the listed DMD2 comparisons in the flow-map-initialized setting, and remains effective under flow-matching initialization and training from scratch.
>
---
#### [new 111] iGSP:Implicit Gradient Subspace Projection for Efficient Continual Learning of Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于持续学习任务，旨在解决视觉-语言模型在不断新增任务下的参数爆炸与负迁移问题。提出iGSP框架，通过梯度子空间投影实现高效适应。**

- **链接: [https://arxiv.org/pdf/2605.19301](https://arxiv.org/pdf/2605.19301)**

> **作者:** Xuezhi Cui; Dongbo Zhou; Wang Guo; Zeyuan Wang; Ziyu Li; Gaozhi Zhou; Xian Li; Ling Zhao; Wentao Yang; Chao Tao; Haifeng Li
>
> **摘要:** Vision-Language Models require efficient adaptation to continually emerging downstream tasks. While Parameter-Efficient Fine-Tuning mitigates catastrophic forgetting, assigning isolated modules per task leads to parameter explosion. Conversely, recent similarity-driven sharing mechanisms falsely equate superficial visual similarity with underlying alignment consistency. This fundamental mismatch triggers severe negative transfer between visually similar but logically distinct tasks and fails to exploit alignment reuse across visually diverse ones. We argue thatalignment sharing is fundamentally a geometric problem of overlapping optimization trajectories within shared low-rank subspaces. Grounded in this insight, we propose iGSP, a novel framework that achieves efficient adaptation via implicit gradient subspace projection. Leveraging the early convergence of MoE routers to establish the subspace basis, iGSP bifurcates the adaptation process into two phases. First, the Subspace Identification phase introduces candidate experts via basis pre-expansion, applies a novel subspace-constrained regularization to implicitly project new task gradients onto the historical subspace, and precisely prunes redundant dimensions by treating routing probabilities as gradient flow indicators, ultimately to maximize knowledge reuse. Second, the Orthogonal Subspace Fine-Tuning phase fixes this structural basis and removes the regularization to rapidly fit the task-specific residual loss. Extensive experiments on the MTIL benchmark demonstrate that iGSP achieves state-of-the-art accuracy while significantly improving training efficiency, reducing the average trainable parameters by 42.7\% compared to current SOTA methods, and decreasing the final total parameters by 86.9\% relative to counterparts. The source code is available at this https URL.
>
---
#### [new 112] MedFM-Robust: Benchmarking Robustness of Medical Foundation Models
- **分类: cs.CV**

- **简介: 该论文属于医疗基础模型的鲁棒性评估任务，旨在解决模型在真实场景下的可靠性问题，通过基准测试分析Med-VLMs和分割模型的性能。**

- **链接: [https://arxiv.org/pdf/2605.19027](https://arxiv.org/pdf/2605.19027)**

> **作者:** Xiangxiang Cui; Tianjin Huang; Yifang Wang; Lijie Hu; Lu Yin
>
> **备注:** MICCAI2026
>
> **摘要:** Medical foundation models (MedFMs) have emerged as transformative tools in healthcare, demonstrating capabilities across diverse clinical applications. These models can be broadly categorized into two paradigms: Medical Vision-Language Models (Med-VLMs) and segmentation foundation models. Med-VLMs range from medical-specialized models such as LLaVA-Med and MedGemma, to general-purpose models like GPT-4o and Gemini, all capable of medical image understanding tasks including visual question answering (VQA), report generation, and visual grounding. Concurrently, the Segment Anything Model (SAM) has catalyzed a new generation of medical segmentation models, with adaptations like SAM-Med2D and MedSAM. The widespread clinical deployment of these models thus necessitates rigorous evaluation of their reliability under real-world conditions.
>
---
#### [new 113] VL-DPO: Vision-Language-Guided Finetuning for Preference-Aligned Autonomous Driving
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自动驾驶任务，旨在解决模型与人类驾驶偏好对齐的问题。通过引入视觉语言模型生成偏好数据，优化运动预测模型，提升其符合人类驾驶习惯。**

- **链接: [https://arxiv.org/pdf/2605.20082](https://arxiv.org/pdf/2605.20082)**

> **作者:** Zhefan Xu; Ghassen Jerfel; Marina Haliem; Qi Zhao; Jeonhyung Kang; Khaled S. Refaat
>
> **备注:** Published in International Conference on Robotics and Automation (ICRA), 2026 8 pages, 6 figures, 4 tables
>
> **摘要:** The rapid growth of autonomous driving datasets has enabled the scaling of powerful motion forecasting models. While large-scale pretraining provides strong performance, the standard imitation objective may not fully capture the complex nuances of human driving preferences. Meanwhile, recent advances in vision-language models (VLMs) have demonstrated impressive reasoning and commonsense understanding. Building on these capabilities, this paper presents VL-DPO, a vision-language-guided framework that aligns ego-vehicle motion forecasting models with human preferences. Our approach leverages a VLM as a zero-shot reasoner to automatically generate preference pairs from a pretrained model's rollouts, which are then used to finetune the model via Direct Preference Optimization (DPO). We finetune our models on the Waymo Open End-to-End Driving Dataset (WOD-E2E) and evaluate performance against held-out human preference annotations using rater feedback score (RFS) and average displacement error (ADE). Our experiments confirm that the VLM's trajectory selection is a high-quality proxy for human preference. Our final model, VL-DPO, yields an 11.94% increase in RFS and a 10.01% reduction in ADE over the pretrained model.
>
---
#### [new 114] CaMo: Camera Motion Grounded Evaluation and Training for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决现有模型缺乏相机运动理解的问题。提出SNS评估框架和CaMo模型，提升模型的3D空间理解能力。**

- **链接: [https://arxiv.org/pdf/2605.20165](https://arxiv.org/pdf/2605.20165)**

> **作者:** Hsiang-Wei Huang; Junbin Lu; Kuang-Ming Chen; Jianxu Shangguan; Cheng-Yen Yang; Jenq-Neng Hwang
>
> **备注:** Code and model available at this https URL
>
> **摘要:** Vision-Language Models (VLMs) achieve strong performance on spatial question answering benchmarks, yet it remains unclear whether such gains reflect genuine spatial intelligence. We show that existing spatial VLMs lack basic camera motion understanding, a key component of spatial cognition. We propose the Spatial Narrative Score (SNS), an evaluation framework that requires VLMs to generate explicit spatial narratives capturing both scene semantics and camera motion, followed by reasoning with a frozen proxy LLM. Under SNS, state-of-the-art spatial VLMs exhibit significant performance degradation despite high direct question answering accuracy. To address this gap, we introduce CaMo, a camera motion grounded VLM that achieves consistent performance across SNS evaluation and direct spatial question answering accuracy. Our results highlight the importance of explicit spatial narrative externalization for evaluating VLMs with transferable 3D spatial understanding. Our code, data, and model is available at this https URL
>
---
#### [new 115] DocQT: Improving Document Forgery Localization Robustness via Diverse JPEG Quantization Tables
- **分类: cs.CV**

- **简介: 该论文属于文档篡改定位任务，解决模型在真实场景中泛化能力不足的问题。通过引入多样化的JPEG量化表，提升模型对实际压缩环境的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.19688](https://arxiv.org/pdf/2605.19688)**

> **作者:** Kylian Ronfleux-Corail; Guillaume Bernard; Mickaël Coustaty; Nicolas Sidère
>
> **摘要:** Document manipulation localization models achieve strong performance on public benchmarks yet fail to generalize to operational document workflows. We identify a critical and overlooked source of this gap: the mismatch between the narrow distribution of JPEG quantization tables used during training -restricted to standard libjpeg quality factors -and the heterogeneous compression profiles encountered in real-world insurance document pipelines. To isolate this factor, we conduct a controlled factorial study comparing two architectures with contrasting levels of quantization table awareness -FFDN [2] and Mesorch [20] -each trained under either standard quality factor augmentation (Standard-QT ) or operationally calibrated quantization tables sampled from DocQT, a quantization-table bank derived from a MAIF operational image corpus (Real-QT ), and evaluated under three recompression conditions. Training under Real-QT yields substantial localization gains on DocTamper [15] and significantly reduces the pixel-level false positive rate on authentic operational documents, but only for architectures that explicitly ingest the quantization table as input. The released DocQT quantization-table dataset and compression-reproduction material are directly available at this https URL. These results demonstrate that standard quality factor augmentation does not adequately proxy operational compression diversity, and that architectural choices explicitly conditioning on the quantization table provide a meaningful robustness advantage for real-world deployment.
>
---
#### [new 116] A Nash Equilibrium Framework For Training-Free Multimodal Step Verification
- **分类: cs.CV; cs.GT**

- **简介: 该论文属于多模态验证任务，旨在解决大模型推理链中的错误验证问题。提出一种无需训练的验证方法，通过纳什均衡框架处理不同判断的一致性，提升验证准确性。**

- **链接: [https://arxiv.org/pdf/2605.20033](https://arxiv.org/pdf/2605.20033)**

> **作者:** Rohit Sinha; Kunal Tilaganji; Tanuja Ganu; Nagarajan Natarajan; Amit Sharma; Vineeth N. Balasubramanian
>
> **备注:** ICLR 2026 Workshop VerifAI-2
>
> **摘要:** Multimodal large language models often generate reasoning chains containing subtle errors that lead to incorrect answers. Current verification approaches have notable limitations. Learned critics need extensive labeled data and show inconsistent performance across different tasks. Meanwhile, existing training-free methods simply average scores from different sources, missing a key insight: when these scores disagree, that disagreement itself carries important information about whether a reasoning step is truly valid or not. We propose a training-free verification approach that treats step-wise verification as a coordination problem among specialized judges. We formalize these judges' interaction as a Nash equilibrium game where agreement signals valid steps while disagreement reveals instability. Our method computes equilibrium scores through a closed-form solution, enabling both disagreement-aware filtering and stability-conscious ranking of reasoning steps. Evaluated across six benchmarks, our approach achieves consistent improvements of 2.4% to 5.2% over baseline models and shows competitive performance against learned critics, demonstrating that cross-modal agreement (not just average confidence) provides robust verification signals without task-specific adaptation.
>
---
#### [new 117] World-Ego Modeling for Long-Horizon Evolution in Hybrid Embodied Tasks
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于具身智能任务，旨在解决长时序混合任务中世界与自我状态纠缠的问题。提出World-Ego Modeling，分离世界与自我演化，构建WEM模型并建立基准HTEWorld进行评估。**

- **链接: [https://arxiv.org/pdf/2605.19957](https://arxiv.org/pdf/2605.19957)**

> **作者:** Zuyao Lin; Jianhui Zhang; Peidong Jia; Xiaoguang Zhao; Shanghang Zhang; Xingyu Chen
>
> **摘要:** World models are widely explored in embodied intelligence, yet they typically predict distinct evolutions of the world and the ego within a single stream, where the world captures persistent instruction-agnostic scene regularities and the ego captures robot-centric instruction-conditioned dynamics. This world-ego entanglement leads to a degradation in long-horizon embodied scenarios, particularly in hybrid tasks with interleaved navigation and manipulation behaviors. In this paper, we introduce \emph{World-Ego Modeling}, a new conceptual paradigm that decomposes future evolution into world and ego components. We define the world-ego boundary from three perspectives, i.e., motion-, semantic-, and intention-based views, and analyze three disentanglement strategies with post-, pre-, and full disentanglement. Further, we instantiate this paradigm as the World-Ego Model (WEM), a unified embodied world model that couples an implicit separate world-ego planner with a cascade-parallel mixture-of-experts (CP-MoE) diffusion generator. To enable rigorous evaluation, we further construct HTEWorld, the first benchmark for long-horizon world modeling with hybrid navigation-manipulation tasks, providing 125K video clips (over 4.5M frames) with fine-grained action annotations and 300 multi-turn evaluation trajectories (over 2K instructions). Extensive experiments show that WEM achieves state-of-the-art performance on HTEWorld while remaining competitive on existing manipulation-only benchmarks.
>
---
#### [new 118] LMM-Track4D: Eliciting 4D Dynamic Reasoning in LMMs via Trajectory-Grounded Dialogue
- **分类: cs.CV**

- **简介: 该论文提出LMM-Track4D，解决LMM在4D动态推理上的不足，通过轨迹引导的对话任务和新基准Track4D-Bench，提升模型对视频中目标轨迹的持续理解能力。**

- **链接: [https://arxiv.org/pdf/2605.19390](https://arxiv.org/pdf/2605.19390)**

> **作者:** Chaoyue Li; Yongxue Xu; Jie Feng; Jiayu Ding
>
> **摘要:** Recent large multimodal models (LMMs) have become increasingly capable on image and video understanding, yet still struggle to sustain 4D continuous spatiotemporal dynamic reasoning. To study this capability gap, we formulate trajectory-grounded multi-turn spatiotemporal dialogue, a new task in which a model must answer spatiotemporal queries while returning structured 3D target trajectories over an entire short clip or a specified segment of a longer clip, and introduce Track4D-Bench, a benchmark with 526 clip-level dialogue samples spanning 23.5k frames and 7.5k object annotations, for training and evaluation. Building on this task, we propose LMM-Track4D, which combines RTGE (Ray--Time Geometry Encoding), a dedicated streaming state token TRK for long-horizon dynamic propagation, and an Object-Slot Kinematic, Residual-Anchor (OSK-RA) decoder for stable 4-step 3D state estimation under occlusion and viewpoint variation. Experiments on Track4D-Bench show consistent improvements over strong baselines, suggesting that explicit dynamic state modeling is a useful design principle for eliciting 4D dynamic reasoning in LMMs. Our code and dataset will be publicly available at this https URL.
>
---
#### [new 119] FAGER: Factually Grounded Evaluation and Refinement of Text-to-Image Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成的评估任务，解决现有指标无法准确评估隐含事实的问题。提出FAGER框架，通过结构化事实评分和反馈提升生成图像的事实准确性。**

- **链接: [https://arxiv.org/pdf/2605.19111](https://arxiv.org/pdf/2605.19111)**

> **作者:** Youngsun Lim; Cusuh Ham; Pin-Yu Chen; Deepti Ghadiyaram
>
> **备注:** It was accepted for an oral presentation at the 2nd Workshop on the Evaluation of Generative Foundation Models (EVGENFM2026) at CVPR 2026. Total 8 pages (1 page for references). 5 figures
>
> **摘要:** Existing text-to-image (T2I) evaluation metrics mainly assess whether generated images align with information explicitly stated in the prompt, but often fail to capture factual requirements that are implicit, externally grounded, or identity-defining. As a result, they are not well suited for evaluating factual correctness in prompts involving scientific knowledge, historical facts, products, or culture-specific concepts. We propose FActually Grounded Evaluation and Refinement (FAGER), an agentic framework that evaluates whether generated images correctly reflect visually verifiable facts grounded in or implied by the prompt, while also providing actionable feedback for improvement. FAGER first constructs a structured factual rubric by combining LLM-based fact proposal with reference-guided visual fact extraction and verification, then converts the rubric into question-answer pairs for VLM-based evaluation. To validate FAGER as a factuality metric, we introduce a Factual A/B test, which measures whether a metric prefers factual reference images over corresponding generated images. Across five datasets spanning science, history, products, culture, and knowledge-intensive concepts, FAGER consistently outperforms prior metrics on this test. We further show that FAGER can be used to refine T2I outputs in a fully training-free manner, yielding substantial factuality gains across datasets.
>
---
#### [new 120] Spatially Prompted Visual Trajectory Prediction for Egocentric Manipulation
- **分类: cs.CV**

- **简介: 该论文提出SP-VTP任务，解决视觉轨迹预测问题。通过空间提示定义操作目标，构建SPOT模型进行轨迹预测。**

- **链接: [https://arxiv.org/pdf/2605.20085](https://arxiv.org/pdf/2605.20085)**

> **作者:** Yifan Li; Xinyu Zhou; Yunhao Ge; Yu Kong
>
> **摘要:** Robotic manipulation is often specified through language instructions or task identifiers, yet cluttered environments with similar objects are better handled by spatially indicating what to move and where to place it. Addressing the vision-centric challenge of object and goal specification, we present, to the best of our knowledge, the first formalization of Spatially Prompted Visual Trajectory Prediction (SP-VTP). This novel setting utilizes initial spatial prompts (like bounding boxes or points) to define task objectives, tasking the model with forecasting future end-effector trajectories from egocentric streams. To study this problem, we collect and annotate EgoSPT, a dataset of egocentric spatially prompted manipulation trajectories with first-frame object and target grounding annotations and recovered 3D end-effector motion. SP-VTP is challenging because the task specification is static, while the scene configuration evolves over time. To solve this problem, we propose SPOT(Spatially Prompted Object-Target Policy), which combines a task encoder for first-frame visual and coordinate spatial prompts, an observation encoder for current visual and history context, and a trajectory generator for future end-effector motion. Experiments under strict scene-level splits show that SPOT improves cross-scene trajectory prediction over non-prompted or single-source prompted baselines. Together, EgoSPT and SPOT establish a new spatial prompting problem SP-VTP, as a simple and scalable task condition for egocentric manipulation.
>
---
#### [new 121] DynaTok: Temporally Adaptive and Positional Bias-Aware Token Compression for Video-LLMs
- **分类: cs.CV**

- **简介: 该论文属于视频大模型任务，旨在解决视频token压缩效率低的问题。提出DynaTok框架，通过动态分配时空预算，提升压缩效果并保持语义精度。**

- **链接: [https://arxiv.org/pdf/2605.19322](https://arxiv.org/pdf/2605.19322)**

> **作者:** Minyoung Park; Taehun Kong; Sangjun Ahn
>
> **摘要:** Recent advances in Video Large Language Models (Video-LLMs) have greatly expanded multimodal reasoning capabilities. However, the massive number of visual tokens extracted from long video sequences incurs prohibitive computational costs, limiting their deployment in real-world scenarios. Existing training-free token compression methods select tokens based on attention magnitude as a proxy for semantic importance, but often overlook positional bias and rely only on short-term temporal locality, leading to redundant spatio-temporal coverage and inefficient token usage. We present DynaTok, a training-free, temporally adaptive and bias-aware token compression framework that allocates token budgets across both temporal and spatial dimensions. Through a lightweight exponential moving average (EMA) memory, the Temporal Budget Allocation (TBA) module dynamically assigns fewer tokens to redundant frames and more to novel frames, capturing long-term temporal variation. The Spatial Budget Allocation (SBA) module complements this by selecting spatially diverse and semantically important features using activation-based attention maps, while leveraging a spatial memory to reduce redundancy from previously selected regions and mitigate positional bias. DynaTok integrates seamlessly with existing Video-LLMs such as LLaVA-OneVision and LLaVA-Video without retraining, and effectively preserves semantic coverage under aggressive compression. Experiments on four representative VideoQA benchmarks-MVBench, LongVideoBench, MLVU, and VideoMME-show that DynaTok retains over 95% of baseline accuracy even with a 90% token reduction, surpassing recent training-free approaches. These results demonstrate that DynaTok provides a principled foundation for efficient and robust video reasoning, paving the way toward real-time streaming video understanding with future Video-LLMs.
>
---
#### [new 122] Semantic-Enriched Latent Visual Reasoning
- **分类: cs.CV**

- **简介: 该论文提出SLVR框架，解决多模态潜在空间推理中语义不足的问题，通过两阶段学习增强潜在表示的语义丰富性与一致性。**

- **链接: [https://arxiv.org/pdf/2605.19342](https://arxiv.org/pdf/2605.19342)**

> **作者:** Tianrun Xu; Yue Sun; Qixun Wang; Jingyi Lu; Yuan Wang; Tianren Zhang; Longteng Guo; Fengyun Rao; Jing Lyu; Feng Chen; Jing Liu
>
> **摘要:** Multimodal latent-space reasoning aims to replace explicit thinking with images by performing visual reasoning directly in a compact latent space. However, existing approaches largely rely on visual supervision and produce latent representations that lack sufficient semantic richness, limiting their ability to support diverse region-level reasoning tasks. In this work, we introduce Semantic-Enriched Latent Visual Reasoning (SLVR), a two-stage learning framework that enriches latent representations with attribute-level visual semantics and aligns them with diverse reasoning objectives. In the first stage, SLVR learns semantically enriched region-centric latents under fine-grained attribute supervision. In the second stage, we design Multi-query Group Relative Policy Optimization (M-GRPO) to align latent representations across multiple queries grounded in the same region. To support this framework, we construct SLV-Set, comprising approximately 400K region-level attribute annotations and 800K multi-query question answering samples, and introduce SV-QA, a benchmark that evaluates latent reasoning under semantic variation. Experiments demonstrate that SLVR improves the robustness and semantic consistency of latent visual reasoning compared to existing baselines.
>
---
#### [new 123] Structural Energy Guidance for View-Consistent Text-to-3D Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到3D生成任务，解决多视角一致性问题。针对扩散模型的Janus问题，提出SEGS框架，通过结构能量引导提升视图一致性。**

- **链接: [https://arxiv.org/pdf/2605.19876](https://arxiv.org/pdf/2605.19876)**

> **作者:** Qing Zhang; Jinguang Tong; Jing Zhang; Jie Hong; Xuesong Li
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2508.16917
>
> **摘要:** Text-to-3D generation based on diffusion models often suffers from the Janus problem, leading to inconsistent geometry across viewpoints. This work identifies viewpoint bias in 2D diffusion priors as the main cause and proposes Structural Energy-Guided Sampling (SEGS), a training-free and plug-and-play framework to improve multi-view consistency. SEGS constructs a structural energy in the PCA subspace of U-Net features and injects its gradient into the denoising process. It can be easily integrated into SDS/VSD pipelines without retraining. Experiments show that SEGS reduces the Janus Rate by about 10% on average and improves View-CS scores across multiple baselines, including DreamFusion, Magic3D, and LucidDreamer. This method effectively alleviates viewpoint artifacts while preserving appearance fidelity, providing a flexible solution for high-quality text-to-3D content generation.
>
---
#### [new 124] Towards Data-Efficient Video Pre-training with Frozen Image Foundation Models
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在降低视频预训练的数据和计算成本。通过冻结图像基础模型，仅训练时间模块，探索轻量级视频模型构建方法。**

- **链接: [https://arxiv.org/pdf/2605.19137](https://arxiv.org/pdf/2605.19137)**

> **作者:** Svetlana Orlova; Niccolò Cavagnero; Gijs Dubbelman
>
> **备注:** Accepted to CVPR 2026 Workshops CV4Smalls
>
> **摘要:** Video foundation models achieve strong performance across many video understanding tasks, but typically require large-scale pre-training on massive video datasets, resulting in substantial data and compute costs. In contrast, modern image foundation models already provide powerful spatial representations. This raises an important question: can competitive video models be built by reusing these spatial representations and pre-training only for temporal reasoning? We take initial steps toward exploring a lightweight training paradigm that freezes a pre-trained image foundation model and trains only a recurrent temporal module to process streaming video. By reusing an image foundation model as a spatial encoder, this approach could significantly reduce the amount of video data and compute required compared to end-to-end video pre-training. In this work, we explore the feasibility of this approach before investing in computing for video pre-training. Our empirical findings across multiple video understanding tasks suggest that strong temporal performance can emerge without large-scale video pre-training, motivating future work on recurrent video foundation models obtained by pre-training a temporal module on top of a frozen image foundation model. Code: this https URL .
>
---
#### [new 125] Rebalancing Reference Frame Dominance to Improve Motion in Image-to-Video Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像到视频生成任务，旨在解决视频过于静态的问题。通过分析参考帧主导机制，提出DyMoS方法，在不改变模型的情况下提升运动动态。**

- **链接: [https://arxiv.org/pdf/2605.19398](https://arxiv.org/pdf/2605.19398)**

> **作者:** Wooseok Jeon; Seungho Park; Seunghyun Shin; Sangeyl Lee; Hyeonho Jeong; Hae-Gon Jeon
>
> **备注:** Preprint
>
> **摘要:** Image-to-video models often generate videos that remain overly static, compared to text-to-video models. While prior approaches mitigate this issue by weakening or modifying the image-conditioning signal, they often require additional training or sacrifice fidelity to the reference image. In this work, we identify \emph{reference-frame dominance} as a key mechanism behind motion suppression. We observe that non-reference frames in I2V models allocate excessive self-attention to reference-frame key tokens, causing reference information to be over-propagated across time and suppressing inter-frame dynamics. Based on this finding, we propose DyMoS~(Dynamic Motion Slider), a training-free and model-agnostic method that rebalances the attention pathway from generated frames to the reference frame during initial denoising steps. DyMoS leaves both the input image and model weights unchanged and introduces a single scalar parameter for continuous control over motion strength. Experiments across multiple state-of-the-art I2V backbones demonstrate that DyMoS consistently improves motion dynamics while maintaining visual quality and fidelity to the reference image.
>
---
#### [new 126] Return of Frustratingly Easy Unsupervised Video Domain Adaptation
- **分类: cs.CV**

- **简介: 该论文研究无监督视频域适应任务，解决跨域视频动作识别中的域差异问题。提出MetaTrans方法，通过分离处理空间和时间偏差，提升适应性能。**

- **链接: [https://arxiv.org/pdf/2605.19510](https://arxiv.org/pdf/2605.19510)**

> **作者:** Pengfei Wei; Yiqun Sun; Zhiqiang Xu; Yiping Ke; Lawrence B. Hsieh
>
> **备注:** To appear in ICML 2026
>
> **摘要:** Unsupervised video domain adaptation (UVDA) is a practical but under-explored problem. In this paper, we propose a frustratingly easy UVDA method, called MetaTrans. Specifically, MetaTrans adopts a concise learning objective that contains only two fundamental loss terms. Despite the simplicity of the learning objective, MetaTrans embodies an advanced UVDA idea, that is, handling the spatial and temporal divergence of cross-domain videos separately, through a subtle model architecture design. By implementing a temporal-static subtraction module, MetaTrans effectively removes spatial and temporal divergence. Extensive empirical evaluations, particularly on various cross-domain action recognition tasks, show substantial absolute adaptation performance enhancement and significantly superior relative performance gain compared with state-of-the-art UVDA baselines.
>
---
#### [new 127] Neuron Incidence Redistribution for Fairness in Medical Image Classification
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医疗图像分类任务，解决模型在不同人口特征上的性能差异问题。通过提出NIR方法，减少因隐层激活模式导致的诊断偏差。**

- **链接: [https://arxiv.org/pdf/2605.19393](https://arxiv.org/pdf/2605.19393)**

> **作者:** Abin Shoby; Lyle John Palmer; Nikhil Cherian Kurian
>
> **备注:** 4 Pages, 1 Figure
>
> **摘要:** Deep learning models for medical image classification are susceptible to subgroup performance disparities across demographic attributes such as age, gender, and race. We identify a latent representational mechanism underlying these disparities: in transfer-learned models, the dominant penultimate-layer activation channel under positive predictions is co-activated by both disease-positive samples and privileged demographic groups (male, older patients), producing over-diagnosis; conversely, the dominant channel under negative predictions is co-activated by disadvantaged groups (female, younger patients), producing systematic under-diagnosis. To address this, we propose Neuron Incidence Redistribution (NIR), a lightweight regularization method that penalizes the variance of predicted-probability-weighted mean activations across penultimate-layer neurons, requiring no demographic labels at training time. On HAM10000, TPR disparity drops from 10.81% to 0.93% across age groups and from 12.04% to 0.74% across gender, with a marginal AUC improvement of 0.51 points. On Harvard OCT-RNFL, NIR reduces FPR disparity for race (from 15.68% to 10.66%) and age (from 12.69% to 1.80%), demonstrating that distributing latent disease evidence across the full penultimate layer is a principled and effective strategy for improving demographic fairness in medical AI.
>
---
#### [new 128] Passive Construction Site Safety Monitoring via Persona-Scaffolded Adversarial Chain-of-Thought VLM Verification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于施工安全监控任务，旨在解决传统方法成本高、依赖人工的问题。通过三阶段架构实现被动安全检测与合规验证。**

- **链接: [https://arxiv.org/pdf/2605.19869](https://arxiv.org/pdf/2605.19869)**

> **作者:** Ananth Sriram; Neel Mokaria; Rajveer Singh
>
> **备注:** 10 pages, 4 figures. First place, this http URL Spatial Intelligence Hackathon, University of Maryland, February 2026. Code available at this https URL
>
> **摘要:** Construction remains the deadliest industry sector in the United States, with 1,055 fatal worker injuries recorded in 2023, and the majority preventable. Existing monitoring approaches are expensive, require real-time human operators, or address only a narrow subset of violations. This paper presents a passive, end-of-shift construction safety monitoring pipeline processing video from POV body-worn and fixed wall-mounted cameras through a three-stage architecture: (1) fine-tuned YOLO11 for primary PPE and hazard detection, (2) SAM 3 for segmentation refinement and worker deduplication, and (3) Qwen3-VL-8B-Instruct with a method-prompted, persona-scaffolded three-pass adversarial chain-of-thought protocol for compliance verification and hallucination control. The principal contribution is the Stage 3 prompt design: professional persona backstories following the method-actor framing drive an observed 12% precision improvement over single-pass prompting in an informal three-author review of the 12-video Ironsite development corpus, with the largest gains on hallucination-prone violation categories. Structural message isolation enforces observational independence between a generator, discriminator, and reconciliation pass governed by asymmetric rules encoding priors about human observation versus automated detection reliability. The system maps violations to OSHA standards, performs REBA-inspired ergonomic risk scoring from pose keypoints, and produces per-worker safety reports with timestamped evidence. An evaluation harness is released for future reproduction.
>
---
#### [new 129] Inverse Design of Metasurface based Absorbers using Physics Guided Conditional Diffusion Models
- **分类: cs.CV; cs.ET**

- **简介: 该论文属于逆向设计任务，旨在解决电磁波吸收器的高效设计问题。通过物理引导的扩散模型生成满足特定反射特性的可制造结构，提升设计效率与精度。**

- **链接: [https://arxiv.org/pdf/2605.19611](https://arxiv.org/pdf/2605.19611)**

> **作者:** Vineetha Joy; Jamshed Palai; Satwik Sahoo; Anshuman Kumar; Amit Sethi; Hema Singh
>
> **摘要:** Inverse design of metasurfaces for specific electromagnetic responses requires generating geometries that satisfy stringent spectral constraints while maintaining manufacturability. Conventional design methodologies rely on iterative optimization routines using full wave simulations, which become extremely time consuming and computationally intensive for large design spaces. In addition, commonly employed generative approaches often exhibit limited conditional fidelity and the generated designs often contain fine or irregular features that are impractical to fabricate. In this regard, we propose a physics guided condition quality enhanced diffusion framework for the inverse design of metasurface based absorbers. Here, the conditioning information consisting of target reflection characteristics is integrated into the model using feature wise linear modulation (FiLM). Furthermore, to enforce adherence to target spectra, a pre trained surrogate EM simulator is embedded into the framework introducing physics aware regularization through spectrum level loss functions. The efficiency of the proposed model is demonstrated by generating practically realizable metasurfaces for different types of reflection characteristics in the frequency range of 2 to 18 GHz. The proposed framework achieves an average spectral mean squared error of 0.0006 and band alignment accuracy of 0.958 between the target spectra and the spectra produced by the generated designs, demonstrating high conditional accuracy. In addition, the model generates multiple geometries for the same condition, thereby providing diverse design alternatives to the engineer. The proposed model produces the suitable design in approximately 30 seconds, whereas the conventional approach can take several months under comparable computational resources. The efficiency of the model is also established via experimental measurements.
>
---
#### [new 130] Cardiac fat segmentation using computed tomography and an image-to-image conditional generative adversarial neural network
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决心脏脂肪自动分割问题。采用生成对抗网络实现精准、高效分割，提升临床诊断效率。**

- **链接: [https://arxiv.org/pdf/2605.20064](https://arxiv.org/pdf/2605.20064)**

> **作者:** Guilherme Santos da Silva; Dalcimar Casanova; Jefferson Tales Oliva; Erick Oliveira Rodrigues
>
> **摘要:** In recent years, research has highlighted the association between increased adipose tissue surrounding the human heart and elevated susceptibility to cardiovascular diseases such as atrial fibrillation and coronary heart disease. However, the manual segmentation of these fat deposits has not been widely implemented in clinical practice due to the substantial workload it entails for medical professionals and the associated costs. Consequently, the demand for more precise and time-efficient quantitative analysis has driven the emergence of novel computational methods for fat segmentation. This study presents a novel deep learning-based methodology that offers autonomous segmentation and quantification of two distinct types of cardiac fat deposits. The proposed approach leverages the pix2pix network, a generative conditional adversarial network primarily designed for image-to-image translation tasks. By applying this network architecture, we aim to investigate its efficacy in tackling the specific challenge of cardiac fat segmentation, despite not being originally tailored for this purpose. The two types of fat deposits of interest in this study are referred to as epicardial and mediastinal fats, which are spatially separated by the pericardium. The experimental results demonstrated an average accuracy of 99.08% and f1-score 98.73 for the segmentation of the epicardial fat and 97.90% of accuracy and f1-score of 98.40 for the mediastinal fat. These findings represent the high precision and overlap agreement achieved by the proposed methodology. In comparison to existing studies, our approach exhibited superior performance in terms of f1-score and run time, enabling the images to be segmented in real time.
>
---
#### [new 131] Smartphone-based Circular Plot Sampling for Forest Inventory
- **分类: cs.CV**

- **简介: 该论文属于森林调查任务，旨在解决传统测量方法成本高、操作复杂的难题。通过智能手机视频实现树干直径和位置的自动测量，提升效率与可及性。**

- **链接: [https://arxiv.org/pdf/2605.19213](https://arxiv.org/pdf/2605.19213)**

> **作者:** Su Sun; Jui-Cheng Chiu; Nabin Khanal; Songlin Fei; Yingjie Victor Chen
>
> **摘要:** Circular sample plots are a cornerstone of forest inventory, yet accurate measurement of tree diameter at breast height (DBH) and spatial location within such plots remains challenging. Conventional approaches rely either on costly terrestrial LiDAR systems or labor-intensive manual methods involving calipers and compass bearings, limiting their scalability and accessibility in large scale environments. We present a lightweight, smartphone-based pipeline that enables complete plot sampling based tree measurement from a single walkthrough video, requiring no specialized hardware beyond a consumer smartphone mounted on a portable stand. The proposed method integrates pretrained monocular depth estimation and tree instance segmentation with a simultaneous localization and mapping (SLAM) framework to jointly refine camera trajectories and depth across the video sequence. Tree positions and DBH estimates are recovered by fusing SLAM-derived camera poses with segmented depth maps, with absolute real-world scale anchored via a calibrated reference length. The system was evaluated in both managed forest plots and natural forest plot, achieving a mean absolute error of 1.51 cm (MARE 3.98%) and 2.30 cm (MARE 5.69%) respectively, with consistent performance across varying starting directions and positions. Cross-video consistency analysis further demonstrated stable and reproducible tree localization across measurements initiated from different starting positions. The proposed approach achieves accuracy comparable to established field methods while substantially reducing equipment cost and operational complexity, making it accessible to both professional researchers and non-expert forest managers in diverse operational settings.
>
---
#### [new 132] FlowErase-RL: Rethinking Concept Erasure as Reward Optimization in Flow Matching Models
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成中的安全控制任务，旨在解决生成有害内容的问题。提出FlowErase-RL框架，通过奖励优化实现概念擦除，提升生成安全性与质量。**

- **链接: [https://arxiv.org/pdf/2605.19739](https://arxiv.org/pdf/2605.19739)**

> **作者:** Yi Sun; Zhiqi Zhang; Xinhao Zhong; Yimin Zhou; Shuoyang Sun; Bin Chen; Shu-Tao Xia; Ke Xu
>
> **摘要:** Recent advances in flow matching models have significantly improved text-to-image generation quality, but also introduce growing safety risks due to the generation of harmful or undesirable content. Existing concept erasure methods are either inference-time interventions with limited effectiveness or rely on supervised fine-tuning (SFT), which requires precisely aligned data and struggles with scalability and multi-concept settings. In this paper, we propose \emph{FlowErase-RL}, the first GRPO-based framework for concept erasure in flow matching models. We reformulate concept erasure as a reward optimization problem and introduce a \textbf{dynamic dual-path reward mechanism} that jointly optimizes (i) a Concept Erasure (CE) reward to suppress target concepts and (ii) a Non-target Space (NS) reward to preserve generative fidelity. The two reward paths are adaptively balanced during training via a performance-driven switching strategy, enabling stable optimization without explicit supervision. Extensive experiments on nudity, object, and artistic style erasure demonstrate that our method achieves state-of-the-art erasure performance while maintaining strong image quality and semantic alignment. Moreover, it exhibits robust resistance to adversarial attacks and scales effectively to multi-concept scenarios. Our results establish a new paradigm for safe and controllable generation in flow matching models.
>
---
#### [new 133] Exposing Functional Fusion: A New Class of Strategic Backdoor in Dynamic Prompt Architectures
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于视觉模型安全任务，针对动态提示架构中的后门攻击问题，提出VIPER框架，揭示功能融合风险，实现高效隐蔽攻击。**

- **链接: [https://arxiv.org/pdf/2605.19478](https://arxiv.org/pdf/2605.19478)**

> **作者:** Zeyao Liu; Zhendong Zhao; Xiaojun Chen; Xin Zhao; Yuexin Xuan; Xiaoshuang Ji
>
> **摘要:** Existing ViT backdoor attacks based on backbone-overwriting full-tuning are computationally expensive and inflict performance degradation. This has forced adversaries towards the Visual Parameter-Efficient Fine-Tuning (PEFT) paradigm, dominated by adapter-based (e.g., LoRA) and prompt-based (e.g., VPT) approaches. While adapter security has seen initial study, the risks of the burgeoning prompt-based ecosystem remain critically unexplored. We fill this critical gap, exposing how the evolution of VPT towards dynamic and context-aware architectures can facilitate a far more dangerous and emergent threat. This vulnerability arises even though these dynamic modules unlock superior benign performance. We propose VIPER, an attack framework built on a lightweight, dynamic Visual Prompt Generator (VPG) that demonstrates this vulnerability. Critically, this dynamic architecture enables Functional Fusion: an emergent phenomenon where malicious logic and benign task utility are tightly fused into the same sparse, high-magnitude parameter core. This fusion creates a formidable ``hostage" dilemma, as pruning the attack necessarily destroys the benign performance. Comprehensive evaluations show VIPER effectively addresses the attacker's trilemma: VIPER not only achieves state-of-the-art performance on clean data, but also maintains near-100% ASR even under 90% VPG-module pruning (where LoRA attacks collapse), while adding only an imperceptible 0.06ms (1.16%) of inference latency. VIPER's results, driven by Functional Fusion, expose a new, paradigm-level risk in dynamic prompt architectures.
>
---
#### [new 134] Beyond Binary Success: A Diagnostic Meta-Evaluation Framework for Fine-Grained Manipulation
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人操作任务，旨在解决现有基准评估过于简化的问题。提出MetaFine框架，通过分解理解、感知和行为三方面诊断模型缺陷，提升细粒度操作能力评估的准确性。**

- **链接: [https://arxiv.org/pdf/2605.19986](https://arxiv.org/pdf/2605.19986)**

> **作者:** He-Yang Xu; Pengyuan Zhang; Zongyuan Ge; Xiaoshuai Hao; Serge Belongie; Xin Geng; Yuxin Peng; Xiu-Shen Wei
>
> **备注:** Project page: this https URL
>
> **摘要:** Fine-grained manipulation marks a regime where global scene context no longer suffices, and success hinges on the tight coupling of local attribute grounding, high-fidelity spatial perception, and constraint-respecting motor execution. However, current embodied AI benchmarks collapse these capacities into binary success rates, systematically inflating reported capabilities by up to 70% and masking the architectural bottlenecks that impede real-world deployment. We introduce MetaFine, a diagnostic meta-evaluation framework that disentangles manipulation competency along three axes: understanding, perception, and controlled behavior. Built on a compositional task graph, MetaFine absorbs heterogeneous external benchmarks and reconstructs them into diagnostic scenarios of varying complexity under a unified protocol. Evaluating state-of-the-art vision-language-action (VLA) models through this lens exposes severe dimension-specific failures invisible to conventional metrics. Through targeted causal intervention, we identify the visual encoder's ability to preserve local spatial structure as a key bottleneck for fine-grained precision: improving it directly unlocks previously inaccessible manipulation capabilities without modifying downstream policies. MetaFine further supports hybrid real-sim validation, using limited paired real-world rollouts to calibrate scalable simulation-based estimates for more stable physical benchmarking. By shifting evaluation from ranking to diagnosis, MetaFine turns benchmarking into an actionable compass for repairing the layered capacities underlying genuine physical dexterity. The MetaFine framework, benchmarks, and supporting resources will be publicly released at our project page: this https URL.
>
---
#### [new 135] Investigating Cross-Modal Skill Injection: Scenarios, Methods, and Hyperparameters
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态学习任务，旨在解决VLM难以高效获取领域技能的问题。通过分析跨模态技能注入的场景、方法和超参数，提出有效整合领域专家模型的策略。**

- **链接: [https://arxiv.org/pdf/2605.19523](https://arxiv.org/pdf/2605.19523)**

> **作者:** Zhiyu Xu; Lean Wang; Yuanxin Liu; Lei Li; Hao Zhou; Fandong Meng; Jie Zhou; Xu Sun
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated remarkable proficiency in general multi-modal understanding; yet they struggle to efficiently acquire continually evolving domain-specific skills. Conventional approaches to enhancing VLM capabilities, such as Supervised Fine-Tuning (SFT), require extensive dataset curation and substantial computational resources. Model merging has emerged as an efficient alternative that enables the transfer of domain-specific expertise from Large Language Models (LLMs) to VLMs without incurring additional training data requirements or significant computational overhead. Unlike conventional merging of homogeneous LLMs, which mainly aggregates existing capabilities, cross-modal skill injection aims to induce emergent cross-modal capabilities by integrating a domain-expert LLM into a VLM. However, existing research lacks a systematic analysis of the applicability and methodology of cross-modal skill injection. In this study, we investigate cross-modal skill injection across three main aspects: scenarios, methods, and hyperparameters. For scenarios, we find that cross-modal skill injection generally performs well in instruction-following and cross-lingual settings, yet struggles with mathematical reasoning. For methods, we find that classic approaches such as TA and DARE consistently achieve superior performance over alternative merging methods. We also provide a systematic and quantitative analysis of the hyperparameter tuning that these classic methods critically depend on.
>
---
#### [new 136] INAR-VL: Input-Aware Routing for Edge-Cloud Vision-Language Inference
- **分类: cs.LG; cs.CV; cs.DC**

- **简介: 该论文属于边缘计算任务，解决VLM在边缘与云间部署的延迟与精度平衡问题。提出INAR-VL系统，通过轻量信号引导路由，提升效率并保持精度。**

- **链接: [https://arxiv.org/pdf/2605.18853](https://arxiv.org/pdf/2605.18853)**

> **作者:** Ahmed Šabanović; Paul Joe Maliakel; Ivona Brandić
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Edge deployment of Vision-Language Models (VLMs) faces a tradeoff between latency and accuracy: cloud execution provides high-quality predictions but incurs communication delay and energy cost, while edge-only execution is faster but less accurate due to limited model capacity. This trade-off is further complicated by heterogeneity in image quality and reasoning complexity, making static placement suboptimal. We present INAR-VL, a lightweight edge-cloud routing system for multimodal inference in a two-tier deployment. INAR-VL maintains complementary VLMs across edge and cloud and uses lightweight image and text complexity signals to guide routing and model selection, executing simple queries locally while offloading complex ones when beneficial. Evaluation on visual question answering shows that INAR-VL executes 36% of requests on the edge, reduces latency by 24%, lowers energy by 26%, and preserves 97% of cloud-level accuracy.
>
---
#### [new 137] Matérn Noise for Triangulation-Agnostic Flow Matching on Meshes
- **分类: cs.GR; cs.CV; cs.LG**

- **简介: 该论文属于网格信号生成任务，解决如何在不依赖三角剖分的情况下生成信号的问题。通过引入Matérn噪声分布，改进流匹配方法，实现高质量、多样化的网格生成。**

- **链接: [https://arxiv.org/pdf/2605.19305](https://arxiv.org/pdf/2605.19305)**

> **作者:** Tianshu Kuai; Arman Maesumi; Daniel Ritchie; Noam Aigerman
>
> **备注:** In ACM Transactions on Graphics (SIGGRAPH 2026). Project page: this https URL
>
> **摘要:** This paper tackles the task of learning to generate signals over triangle meshes in a triangulation-agnostic manner, meaning the trained model can be applied to different meshes and triangulations effectively. Practically, the paper adapts the flow matching (FM) paradigm to a mesh-based, triangulation-agnostic setting. Theoretically, it proposes a specific noise distribution which is triangulation agnostic, to be used inside the FM model's denoising process. While noise distributions are usually trivial to devise for, e.g., images, devising a triangulation-agnostic distribution proves to be a much more difficult task. We formulate a mathematical definition of triangulation agnosticism of distributions, via their spectrum. We then show that a discretization of a specific Gaussian random field called a Matérn process holds these desired properties, and provides a simple and efficient sampling algorithm. We use it as our noise model, and adapt FM to the triangulation-agnostic setting by using a state-of-the-art approach for learning signals on meshes in the gradient domain -- PoissonNet -- as the denoiser. We conduct experiments on elaborate tasks such as sampling elastic rest states, and generating poses of humanoids. Our method is shown to be capable of producing highly realistic results for meshes of over one million triangles, significantly exceeding the state-of-the-art in quality and diversity.
>
---
#### [new 138] Spectral Gradient Surgery for Domain-Generalizable Dataset Distillation
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于数据集蒸馏任务，旨在提升合成数据在分布外场景下的泛化能力。针对现有方法缺乏域多样性的问题，提出谱梯度手术方法，增强跨域共享信息并促进数据集多样性。**

- **链接: [https://arxiv.org/pdf/2605.18836](https://arxiv.org/pdf/2605.18836)**

> **作者:** Minyoung Oh; Najeong Chae; Jae-Young Sim
>
> **备注:** 17pages
>
> **摘要:** Dataset Distillation (DD) synthesizes a compact synthetic dataset that preserves the training utility of a full dataset. However, its standard formulation assumes that test data follow the same distribution as training data, an assumption that rarely holds in practice. A straightforward extension-applying post-hoc Domain Generalization (DG) techniques to distilled data-is ill-suited because existing DG methods rely on the natural diversity of real datasets, which compact synthetic sets inherently lack, while also incurring substantial augmentation overhead that conflicts with the efficiency objective of dataset distillation. To address this limitation, we introduce Domain Generalizable Dataset Distillation (DGDD), a new problem setting that explicitly targets out-of-distribution (OOD) generalization of distilled datasets. We study this problem through a widely adopted DD baseline of Distribution Matching (DM). We attribute the OOD vulnerability of DM to the entanglement of class-discriminative and domain-specific information within the compressed synthetic set, and propose Spectral Gradient Surgery (SGS) to disentangle the two. The key insight of SGS is that cross-domain agreement among domain-wise gradients in the spectral domain reveals which gradient components are shared across source domains-and are therefore class-discriminative-and which are domain-specific. Based on this observation, SGS augments the standard DM update with two complementary gradients: one that reinforces cross-domain shared components and another that explicitly promotes diversity within the distilled dataset. Extensive experiments on diverse-scale benchmarks demonstrate that SGS substantially improves OOD generalization while remaining plug-and-play compatible with existing DM methods.
>
---
#### [new 139] EgoBabyVLM: Benchmarking Cross-Modal Learning from Naturalistic Egocentric Video Data
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态学习任务，旨在解决当前视觉语言模型在自然视角视频数据上的泛化能力不足问题。通过构建基准测试和挑战，推动模型从婴儿视角数据中进行 grounded 语言学习。**

- **链接: [https://arxiv.org/pdf/2605.19130](https://arxiv.org/pdf/2605.19130)**

> **作者:** Dongyan Lin; Phillip Rust; Angel Villar Corrales; Alvin W. M. Tan; Mahi Luthra; Charles-Éric Saint-James; Rashel Moritz; Sheila Krogh-Jespersen; Vanessa Stark; Surya Parimi; Jiayi Shen; Youssef Benchekroun; Yosuke Higuchi; Martin Gleize; Tom Fizycki; Nicolas Hamilakis; Manel Khentout; Sho Tsuji; Balázs Kégl; Juan Pino; Michael C. Frank; Emmanuel Dupoux
>
> **摘要:** Children acquire language grounding with remarkable robustness from limited visuo-linguistic input in ways that surpass today's best large multimodal models. Recent research suggests current vision-language models (VLMs) trained on curated web data fail to generalize to the sparse, weakly-aligned egocentric streams produced by wearable devices, embodied agents, and infant head-cams -- and no fixed evaluation pipeline exists for measuring progress on this regime. We train VLMs on datasets with varying degrees of semantic alignment between visual and linguistic inputs, including naturalistic infant and adult egocentric videos, and evaluate them with a comprehensive suite spanning multimodal language grounding and unimodal vision and language tasks. At the core of this suite is Machine-DevBench, a corpus-grounded benchmark of lexical and grammatical competence, automatically generated from the model's training vocabulary across logarithmic frequency bins to eliminate the train/eval mismatch and low statistical power of prior developmental benchmarks. Our results show that current VLM paradigms hinge on the tight semantic alignment of curated data and fail to exploit the weakly-aligned signal that dominates naturalistic egocentric input -- the very regime in which humans thrive. To motivate progress, we introduce the EgoBabyVLM Challenge to drive the development of models capable of grounded language learning from the kind of naturalistic data that human infants experience.
>
---
#### [new 140] SafeAlign-VLA: A Negative-Enhanced Safe Alignment Framework for Risk-Aware Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决安全关键场景下的风险感知不足问题。通过引入负样本增强安全对齐框架，提升系统对危险行为的理解与应对能力。**

- **链接: [https://arxiv.org/pdf/2605.19524](https://arxiv.org/pdf/2605.19524)**

> **作者:** Kefei Tian; Yuansheng Lian; Kai Yang; Xiangdong Chen; Shen Li
>
> **摘要:** End-to-end autonomous driving systems excel in common scenarios but struggle with safety-critical long-tail cases. Vision-Language-Action (VLA) models are promising due to their strong reasoning capabilities. However, most VLA-based approaches rely on positive expert demonstrations, rarely exploiting negative samples, leading to insufficient understanding of risky behaviors and safety boundaries. To address this limitation, we propose SafeAlign-VLA, a unified negative-enhanced safe alignment framework that incorporates negative data into supervised learning and reinforcement learning. First, we develop a counterfactual safety pairing paradigm to generate structured safety labels and counterfactual positive trajectories from risky scenarios via counterfactual reasoning. Then, a two-stage training strategy is adopted: negative-enhanced supervised fine-tuning for failure feedback and trajectory correction, followed by anchor-based group relative policy optimization that uses positive and negative trajectories as contrastive anchors to steer sampling and penalize high-risk behaviors via group-relative advantages. Experiments on NAVSIM and DeepAccident validate the proposed framework. SafeAlign-VLA achieves 89.1 PDMS on the NAVSIM v1 testset, improving over the baseline without negative data by 1.3%. On DeepAccident, it reduces the collision rate to 3.36%, while achieving 84.2% language accuracy and 85.8% risk prediction accuracy. These results demonstrate the effectiveness of the proposed negative-enhanced safe alignment framework for safe and robust autonomous driving.
>
---
#### [new 141] Beyond Imitation: Learning Safe End-to-End Autonomous Driving from Hard Negatives
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，解决模仿学习中安全性和轨迹偏差不匹配的问题。通过引入失败样本和新的损失函数，提升驾驶安全性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.19771](https://arxiv.org/pdf/2605.19771)**

> **作者:** Junli Wang; Zhihua Hua; Xueyi Liu; Zebin Xing; Haochen Tian; Kun Ma; Hangjun Ye; Guang Chen; Long Chen; Qichao Zhang
>
> **摘要:** Existing imitation learning methods for end-to-end autonomous driving predominantly learn from successful demonstrations by minimizing geometric deviations from expert trajectories. This paradigm implicitly assumes that spatial proximity implies behavioral safety, leading to a critical objective mismatch: trajectories with nearly identical imitation losses may exhibit drastically different safety outcomes, where one remains recoverable while the other results in collision. To address this limitation, we propose BeyondDrive, a failure-aware imitation learning framework that jointly learns from successful and failed driving behaviors. First, we introduce a flow matching-based negative trajectory generator that synthesizes safety-critical yet expert-proximate trajectories, enabling explicit modeling of safety asymmetry. Second, we develop a diversity-aware sampling strategy that mitigates mode collapse and improves coverage of diverse failure modes during negative trajectory generation. Third, we propose a Repulsive Distance Loss that simultaneously attracts predictions toward expert demonstrations while repelling them from hard negative trajectories, thereby establishing discriminative safety boundaries in trajectory space. Applied to the uni-modal baseline Latent TransFuser, BeyondDrive achieves 89.7 PDMS on the NAVSIMv1 closed-loop benchmark, outperforming prior state-of-the-art methods. Moreover, BeyondDrive generalizes effectively across different autonomous driving architectures, including multi-modal planners, and further demonstrates strong zero-shot transferability on the HUGSIM benchmark.
>
---
#### [new 142] Worst-Group Equalized Odds Regularization for Multi-Attribute Fair Medical Image Classification
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于医疗图像分类任务，旨在解决不同群体间诊断性能差异问题。通过引入一种新的正则化方法，减少Equalized Odds和Equalized Opportunity的不公平现象，同时保持AUC性能。**

- **链接: [https://arxiv.org/pdf/2605.19214](https://arxiv.org/pdf/2605.19214)**

> **作者:** Nikhil Cherian Kurian; Victor Caquilpan Parra; Abin Shoby; Luke Whitbread; Lauren Oakden-Rayner; Robert Vandersluis; Jessica Schrouff; Lyle J. Palmer; Mark Jenkinson
>
> **备注:** 11 Pages, 2 Figures
>
> **摘要:** Diagnostic performance in medical AI varies systematically across demographic groups, yet subgroup AUC can mask clinically important disparities. At a fixed inference-time operating point, some groups may exhibit over-diagnostic behaviour, characterized by elevated true and false positive rates, while others show under-diagnostic patterns with reduced true and false positive rates. These opposing tendencies can cancel in aggregate AUCs while producing meaningful inequities in clinical decision-making. Motivated by the need to assess and mitigate such disparities at the operating point and across multiple demographic attributes simultaneously, we propose a worst-group equalized-odds margin regularizer. The proposed regularizer explicitly targets subgroup-level deviations on both the true positive and false positive sides at inference. At each update, the method identifies subgroups defined by explicit demographic attributes (e.g., age, sex, and race) that exhibit the most extreme margin deviations and applies a unified penalty, enabling fairness optimization across multiple demographic axes without requiring explicit intersectional constraints. Across two medical imaging datasets in realistic multi-label settings, our method consistently reduces disparities in Equalized Odds and Equalized Opportunity with minimal impact on AUC, preserving diagnostic performance while improving fairness.
>
---
#### [new 143] Skinned Motion Retargeting with Spatially Adaptive Interaction Guidance
- **分类: cs.GR; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于动作迁移任务，解决不同体型角色间保持交互语义的问题。通过动态调整锚点并结合图结构生成目标动作，提升迁移效果。**

- **链接: [https://arxiv.org/pdf/2605.19355](https://arxiv.org/pdf/2605.19355)**

> **作者:** Soojin Choi; Seokhyeon Hong; Chaelin Kim; Junghyun Nam; Junhyuk Jeon; Junyong Noh
>
> **备注:** SIGGRAPH 2026 / ACM TOG. Project page available at this https URL
>
> **摘要:** Retargeting motion across characters with varying body shapes while preserving interaction semantics, such as self-contact and near-body proximity, remains a challenging problem. While recent geometry-aware approaches address this by maintaining spatial relationships between predefined corresponding regions, their reliance on static correspondences often struggles when the target character exhibits exaggerated body proportions. In this paper, we present a geometry-aware motion retargeting framework that preserves interaction semantics by performing proximity matching over spatially adaptive anchors. Unlike prior methods with static anchor definitions, the proposed method dynamically repositions anchors to reachable regions on the target character. This is achieved via a Transformer-based anchor refinement strategy that predicts anchor displacements and constrains the translated anchors to remain on the target character geometry through differentiable soft projection. By incorporating pose-dependent spatial structures from the source character, the adapted anchors provide structurally coherent guidance for interaction-aware retargeting. Conditioned on these anchors, a graph-based autoencoder predicts target skeletal motion that preserves the spatial configuration of the source. To encourage task-aligned optimization between anchor adaptation and motion retargeting, we adopt an alternating training scheme in which each module is optimized in turn. Through extensive evaluations, we demonstrate that our method outperforms state-of-the-art approaches in preserving interaction fidelity across diverse character geometries.
>
---
#### [new 144] Closed-Loop Hybrid Digital Twin Platform for Connected and Automated Vehicle Validation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶车辆验证任务，旨在解决仿真与实车脱节的问题。提出一种闭环混合数字孪生平台，实现仿真与实车的实时同步和控制。**

- **链接: [https://arxiv.org/pdf/2605.19490](https://arxiv.org/pdf/2605.19490)**

> **作者:** Kanglong Quan; Zhebing Xia; Linfeng Jiang; Hao Yu; Ziheng Qiao; Dapeng Dong; Dongyao Jia
>
> **摘要:** Comprehensive and efficient validation of connected and automated vehicles (CAVs) is critical prior to real-world deployment. While simulation-based testing offers scalability, existing approaches often lack seamless integration with real vehicles and field data, limiting their fidelity in capturing dynamic, real-world interactions. To bridge this gap, this paper proposes a novel real-time hybrid digital twin platform. Its core innovation lies in the tight coupling of a high-fidelity CARLA-SUMO co-simulation with a physical test site and vehicle via a low-latency Vehicle-to-Everything (V2X) communication link. A custom-developed middleware serves as the critical bridge, synchronizing a real CAV's kinematic state as a shadow vehicle in the simulation and translating virtual control commands into chassis-actuating Controller Area Network (CAN) messages for closed-loop control. Detailed implementation includes using photogrammetry for full-scale asset reconstruction and a cloud-edge collaborative architecture for scalable, multi-user operation. Experimental results demonstrate stable synchronization and effective closed-loop control with low latency, confirming the platform's practicality for multi-scenario CAV verification.
>
---
#### [new 145] SpecX: A Large-Scale Benchmark for Multi-Modal Spectroscopy and Cross-Paradigm Evaluation
- **分类: eess.IV; cs.CV; cs.LG; q-bio.OT**

- **简介: 该论文提出SpecX，一个大规模多模态光谱基准，解决现有基准规模小、模态对齐差的问题。任务为光谱智能，工作包括构建多模态数据集和跨范式评估体系。**

- **链接: [https://arxiv.org/pdf/2605.18791](https://arxiv.org/pdf/2605.18791)**

> **作者:** Chengrui Xiang; Tengfei Ma; Yujie Chen; Tong Wang; Haowen Chen; Xiangxiang Zeng
>
> **备注:** 9 pages,1 figures
>
> **摘要:** Existing spectral benchmarks are limited in scale, modality alignment, and evaluation scope, and typically focus on either specialized models or multimodal language models (MLLMs). We introduce SpecX, a large-scale benchmark for multi-modal spectroscopy with cross-paradigm evaluation. SpecX contains 1.7M molecules with diverse spectral modalities, including NMR (1H, 13C, HSQC), IR, MS,UV,Raman and FL, and is organized into three tiers: a large-scale dataset for pretraining, an aligned multi-spectral subset for benchmarking, and a high-quality experimental subset for evaluation. SpecX supports a range of tasks such as molecular elucidation, spectrum simulation, and spectral understanding, and enables unified evaluation across both specialized spectral models and MLLMs. Experiments show that specialized models excel at signal-level modeling, while MLLMs exhibit strengths in high-level reasoning but lack precise spectral grounding. SpecX establishes a unified benchmark for spectral intelligence and highlights the need for spectrum-native foundation models.
>
---
#### [new 146] RLFTSim: Realistic and Controllable Multi-Agent Traffic Simulation via Reinforcement Learning Fine-Tuning
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文属于交通仿真任务，解决传统方法难以捕捉动态多智能体交互的问题。通过强化学习微调提升仿真真实性和可控性。**

- **链接: [https://arxiv.org/pdf/2605.19033](https://arxiv.org/pdf/2605.19033)**

> **作者:** Ehsan Ahmadi; Hunter Schofield; Behzad Khamidehi; Fazel Arasteh; Jinjun Shan; Lili Mou; Dongfeng Bai; Kasra Rezaee
>
> **备注:** CVPR 2026 Highlight; Project page at this https URL
>
> **摘要:** Supervised open-loop training has been widely adopted for training traffic simulation models; however, it fails to capture the inherently dynamic, multi-agent interactions common in complex driving scenarios. We introduce RLFTSim, a reinforcement-learning-based fine-tuning framework that enhances scenario realism by aligning simulator rollouts with real-world data distributions and provides a method for distilling goal-conditioned controllability in scenario generation. We instantiate RLFTSim on top of a pre-trained simulation model, design a reward that balances fidelity and controllability, and perform comprehensive experiments on the Waymo Open Motion Dataset. Our results show improvements in realism, achieving state-of-the-art performance. Compared with other heuristic search-based fine-tuning methods, RLFTSim requires significantly fewer samples due to a proposed low-variance and dense reward signal, and it directly addresses the realism alignment issue by design. We also demonstrate the effectiveness of our approach for distilling traffic simulation controllability through goal conditioning. The project page is available at this https URL.
>
---
#### [new 147] FGSVQA: Frequency-Guided Short-form Video Quality Assessment
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于视频质量评估任务，解决短格式视频质量预测问题。提出FGSVQA框架，结合视觉特征与频率域先验，提升评估准确性与效率。**

- **链接: [https://arxiv.org/pdf/2605.20016](https://arxiv.org/pdf/2605.20016)**

> **作者:** Xinyi Wang; Angeliki Katsenou; Junxiao Shen; David Bull
>
> **备注:** 4 pages, 1 figure
>
> **摘要:** Short-form video poses new challenges to the quality assessment of user-generated content (UGC) due to its complex generation pipeline, rapid content variation, and mixed distortions. To address this challenge, we propose an end-to-end video quality assessment (VQA) framework that employs a dense visual encoder based on CLIP, and incorporates compression priors derived from the frequency domain to generate artifact- and structure-aware weight maps for feature aggregation. By explicitly decomposing artifact, structure, and original visual feature branches and adaptively fusing them over time through a learned gating module, the proposed method achieves accurate and efficient quality prediction. Experimental results show that our method achieves strong performance on short-form video datasets in terms of average rank and linear correlation (SRCC: 0.736, PLCC: 0.787), while maintaining efficient inference runtime. The code and additional results are available at: this https URL.
>
---
#### [new 148] AnchorFlow: Editable SVG Reconstruction via Sparse Anchor Point Fields
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于图像到SVG的重建任务，旨在解决矢量结构参数化中的保真度与可编辑性平衡问题。提出AnchorFlow框架，通过稀疏锚点场实现高效可编辑的SVG生成。**

- **链接: [https://arxiv.org/pdf/2605.19551](https://arxiv.org/pdf/2605.19551)**

> **作者:** Mengnan Jiang; Christian Franke; Michele Franco Adesso; Antonio Haas; Grace Li Zhang
>
> **摘要:** Image-to-SVG reconstruction aims to produce vector graphics that are faithful to raster inputs and easy to edit. Existing methods face a structural trade-off in how vector structure is parameterized, including how many paths represent an image and how many anchor points define each path. High-fidelity methods often rely on many paths or densely parameterized curves, whereas overly compact SVG generation may deviate from the input geometry. This issue becomes more pronounced when local raster evidence is imperfect, where boundary-following reconstruction can introduce redundant anchors and fragmented structures. We argue that this trade-off should be addressed at the level of anchor placement, since anchors on Bezier curves define local path structure and strongly affect both accuracy and editability. We propose AnchorFlow, an editable SVG reconstruction framework that models path-level anchor placement with sparse anchor point fields. Given path-like foreground components extracted from a raster image, AnchorFlow predicts an image-conditioned sparse anchor field for each component and resolves it into an ordered Bezier path. Rendering-guided feedback then corrects local structural errors before re-resolution. The recovered paths are then assembled and optimized into the final SVG. Experiments on isolated paths and full images show that AnchorFlow achieves a favorable fidelity-editability trade-off, substantially reducing editable complexity while preserving competitive raster fidelity.
>
---
#### [new 149] From Prompts to Pavement Through Time: Temporal Grounding in Agentic Scene-to-Plan Reasoning
- **分类: cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文属于自主车辆场景到计划推理任务，旨在解决时间感知不足导致的推理不一致问题。通过引入具有时间整合的规划器架构，评估其在BDD-X数据集上的表现。**

- **链接: [https://arxiv.org/pdf/2605.19824](https://arxiv.org/pdf/2605.19824)**

> **作者:** Ahmed Y. Gado; Omar Y. Goba; Alaa Hassanein; Catherine M. Elias; Ahmed Hussein
>
> **摘要:** Recent attempts to support high-level scene interpretation and planning in Autonomous Vehicles (AVs) using ensembles of Large Language Models (LLMs) and Large Multimodal Models (LMMs) continue to treat time as a secondary property. This lack of temporal grounding leads to inconsistencies in reasoning about continuous actions, undermining both safety and interpretability. This work explores whether temporal conditioning within inter-agent communication can preserve or enhance coherence without introducing degradation in semantic or logical consistency. To investigate this, we introduce three planner architectures with progressively increasing temporal integration and evaluate them on curated subsets of the BDD-X dataset using semantic, syntactic, and logical metrics. Results show that while temporal conditioning reshapes reasoning style, it yields no statistically significant improvements in standard NLP-based correctness metrics. However, qualitative analysis reveals predictive hazard reasoning, stable corrective behavior, and strategic divergence in the Sentinel. These findings clarify the limits of prompt-based temporal grounding and establish the first empirical benchmark for temporal scene-to-plan reasoning.
>
---
#### [new 150] Delta Attention Residuals
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出Delta Attention Residuals，解决注意力残差中冗余导致的路由失效问题，通过关注层间变化而非累积状态，提升模型性能。属于自然语言处理任务。**

- **链接: [https://arxiv.org/pdf/2605.18855](https://arxiv.org/pdf/2605.18855)**

> **作者:** Cheng Luo; Zefan Cai; Junjie Hu
>
> **摘要:** Attention Residuals replace standard additive residual connections with learned softmax attention over previous layer outputs, enabling selective cross-layer routing. However, standard Attention Residuals still attend over cumulative hidden states in previous layers, which are highly redundant. We show that this redundancy leads to routing collapse in deeper layers: attention weights become low-contrast and closer to uniform (max weight ${\approx}$0.2), limiting the model's ability to select informative states in previous layers. This raises a key but underexplored design question: what layer-wise representations should be routed in Attention Residuals? To answer this question, we propose Delta Attention Residuals, which attend over deltas -- the change introduced by each sublayer ($\mathbf{v}_i = \mathbf{h}_{i+1} - \mathbf{h}_i$) -- instead of cumulative states. Delta representations are structurally diverse and yield higher-contrast attention distributions (max weight ${\approx}$0.6), enabling more selective and effective routing across layers. This principle applies at both per-sublayer and block granularity. Across all tested scales (220M--7.6B), Delta Attention Residuals consistently outperform both standard residuals and Attention Residuals, with 1.7--8.2\% validation perplexity gains. Delta Attention Residuals also enables converting pretrained checkpoints into Delta Attention Residuals via standard fine-tuning. Code is available at this https URL.
>
---
#### [new 151] Navigating the Emotion Tree: Hierarchical Hyperbolic RAG for Multimodal Emotion Recognition
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于多模态情感识别任务，旨在解决情感类别独立处理和缺乏上下文知识的问题。提出HyperEmo-RAG框架，结合层次化超球嵌入与结构化证据注入，提升情感分类性能。**

- **链接: [https://arxiv.org/pdf/2605.18884](https://arxiv.org/pdf/2605.18884)**

> **作者:** Zeheng Wang; Bo Zhao; Yijie Zhu; Zhishu Liu; Hui Ma; Ruixin Zhang; Shouhong Ding; Qianyu Xie; Zitong Yu
>
> **摘要:** Multimodal emotion recognition aims to integrate text, audio, and video sources to understand human affective states. Although multimodal large language models excel at multimodal reasoning, they typically treat emotion categories as independent labels, ignoring the rich hierarchical taxonomy of human psychology. Moreover, lacking external contextual knowledge makes them highly susceptible to over-interpreting noisy cues, further complicating fine-grained emotion classification. To address these issues, we propose \textbf{HyperEmo-RAG}, a retrieval-augmented generation framework that leverages a structured emotional knowledge base. Our framework introduces two key innovations. 1) Hierarchical hyperbolic grounding. Recognizing the inherent hierarchical tree structure of emotion taxonomies, we jointly embed hierarchical emotion labels and multimodal samples into a continuous hyperbolic space (Poincaré ball) and design a hierarchical beam-search deliberation process that progressively retrieves samples from coarse to fine-grained levels. 2) Structured evidence injection. Based on the retrieved evidence, we construct an evidence graph and inject the structured knowledge as explicit cognitive context into the LLM through a Tree-Aware Attention mechanism and an EmotionGraphFormer, preserving the integrity of graph-structured information. Experiments on multiple datasets demonstrate that HyperEmo-RAG significantly outperforms existing methods.
>
---
#### [new 152] Minimalist Visual Inertial Odometry
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于视觉-惯性里程计任务，解决资源消耗大的问题。通过四点光敏传感器和IMU实现高效精准的平面定位。**

- **链接: [https://arxiv.org/pdf/2605.19990](https://arxiv.org/pdf/2605.19990)**

> **作者:** Francesco Pasti; Jeremy Klotz; Nicola Bellotto; Shree K. Nayar
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Visual-Inertial Odometry(VIO), which is critical to mobile robot navigation, uses cameras with a large number of pixels. Capturing and processing camera images requires significant resources. This work presents a minimalist approach to planar odometry, demonstrating that just four visual measurements and an IMU can provide robust motion estimation for differential-drive robots. Our key insight is that four downward-facing photodiodes that sense the world through optical Gabor masks produce signals that encode speed. Based on this, we jointly optimize the mask parameters alongside a Temporal Convolutional Network (TCN) using a physically-grounded simulator. The resulting model decodes speed from just the four measurements produced by the photodiodes. Pairing these estimates with the angular speed from an IMU yields a continuous planar trajectory. We validate our approach with a prototype sensor mounted on a differential drive robot. Across diverse indoor and outdoor terrains, our system closely tracks the reference ground truth without any real-world fine-tuning. Our work shows that minimalist sensing enables efficient and accurate planar odometry.
>
---
#### [new 153] Next-Acceleration-Scale Prediction for Autoregressive MRI Reconstruction
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于MRI重建任务，解决高加速下图像模糊问题。通过离散多尺度自回归预测，提升稀疏数据下的重建质量。**

- **链接: [https://arxiv.org/pdf/2605.19354](https://arxiv.org/pdf/2605.19354)**

> **作者:** Yilmaz Korkmaz; Vishal M. Patel
>
> **摘要:** MRI reconstruction is an inherently ill-posed inverse problem, since incomplete measurements admit many plausible solutions. This ambiguity becomes more severe under high acceleration, where pixel-domain continuous predictors tend to average over feasible reconstructions and suppress high-frequency anatomy. We address this limitation by moving reconstruction to discrete multi-scale latent space and posing it as autoregressive next-acceleration-scale prediction. Leveraging discrete priors proven effective in visual autoregressive modeling, our method restricts the solution to compact sequences of codebook tokens, enabling sharp reconstructions even from extremely sparse measurements. This discrete autoregressive formulation also aligns naturally with modern large language model post-training techniques. Building on this observation, we introduce on-policy privileged information distillation for visual autoregressive modeling, where a teacher is provided training only privileged context that is unavailable at inference, in our case fully sampled acquisitions, and supervises a student trained on its own rollouts, leading to consistent reconstruction gains. Through extensive experiments on the fastMRI benchmark, we show that our approach delivers improved reconstruction performance across diverse sampling patterns under extreme undersampling. Project website is \hyperlink{this https URL}{here}.
>
---
#### [new 154] DarkLLM: Learning Language-Driven Adversarial Attacks with Large Language Models
- **分类: cs.CR; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出DarkLLM，解决基础模型易受对抗攻击的问题。通过自然语言指令生成对抗扰动，统一多种攻击类型，提升攻击灵活性与有效性。**

- **链接: [https://arxiv.org/pdf/2605.18868](https://arxiv.org/pdf/2605.18868)**

> **作者:** Ye Sun; Xin Wang; Jiaming Zhang; Yifeng Gao; Yixu Wang; Yifan Ding; Qixian Zhang; Henghui Ding; Xingjun Ma; Yu-Gang Jiang
>
> **备注:** 23 pages, 13 figures
>
> **摘要:** While vision and multimodal foundation models underpin critical tasks from perception to complex reasoning, they remain highly vulnerable to adversarial attacks. However, traditional adversarial attacks are typically limited to single, predefined objectives, tightly coupling each attack to a specific model or task, which restricts their scalability and flexibility in real-world scenarios. In this work, we present DarkLLM, a novel attack framework that trains an LLM to translate natural-language attack instructions into latent attack vectors, which are then decoded into visual adversarial perturbations. By leveraging natural-language instruction tuning, DarkLLM not only unifies targeted, untargeted, segmentation, and multi-model attacks within a single framework, but also achieves flexible and controllable adversarial generation, enabling each instruction to produce a perturbation that induces desired behaviors across heterogeneous models. Through extensive experiments across 4 tasks, 13 datasets, and 15 models, we demonstrate that DarkLLM with only 1B parameters can follow attacker instructions and generate highly effective attacks against CLIP, SAM, and frontier LLMs, revealing a systemic vulnerability in modern foundation models.
>
---
#### [new 155] Reasoning Portability: Guiding Continual Learning for MLLMs in the RLVR Era
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于持续学习任务，解决MLLM在RLVR中的适应问题。提出Reasoning Portability机制，提升模型在新任务上的表现。**

- **链接: [https://arxiv.org/pdf/2605.18903](https://arxiv.org/pdf/2605.18903)**

> **作者:** Qiuhe Hong; Yuyang Liu; Shuo Yang; Tiantian Peng; Fei Zhu; Yonghong Tian
>
> **摘要:** Vision-Language Models in Continual Learning (VLM-CL) aim to continuously adapt to new multimodal tasks while retaining prior knowledge. The emerging paradigm that couples Multimodal Large Language Models (MLLMs) with Reinforcement Learning with Verifiable Rewards (RLVR) calls for a new pattern to guide continual adaptation. Advances in reasoning capability now make it feasible to impose constraints at the reasoning level. We formalize portability, a sample-level measure of how reusable the previous policy's behavior is on a new task, and empirically show that reasoning-level signals remain reliable on out-of-distribution samples while answer-level signals do not. We instantiate this as Reasoning Portability (RP) and propose Reasoning-based Dynamic Balance Continual Learning (RDB-CL), which modulates the per-sample Kullback-Leibler regularization in RLVR according to RP: a tight anchor preserves reusable reasoning on high-RP samples, while a relaxed anchor on low-RP samples permits exploration of new reasoning pathways. Experiments show that RDB-CL consistently outperforms baselines, improving Last accuracy by +12.0% over the vanilla RLVR baseline.
>
---
#### [new 156] From Division to Decision: Leveraging Temporal Cell-Stage Segmentation for Embryo Transferability Prediction
- **分类: eess.IV; cs.CV; cs.LG; q-bio.QM**

- **简介: 该论文属于胚胎评估任务，旨在解决牛胚胎移植成功率低的问题。通过分析早期发育视频，提出TransFACT模型，结合时间特征与阶段信息，提升移植预测准确性。**

- **链接: [https://arxiv.org/pdf/2605.18923](https://arxiv.org/pdf/2605.18923)**

> **作者:** Yasmine Hachani; Patrick Bouthemy; Elisa Fromont; Véronique Duranthon; Ludivine Laffont; Alline de Paula Reis
>
> **摘要:** Accurate selection of bovine embryos is a challenging task, as current practice relies on a single expert assessment on the seventh day after insemination, resulting in high rates of pregnancy loss. Time-lapse videomicroscopy provides detailed information on early development, but is difficult to exploit because of complex motion patterns and time-consuming analysis. We propose TransFACT, a transformer-based framework for modeling early developmental stages and embryo transferability using 2D time-lapse videos from the first four days of development. TransFACT combines frame-level temporal features with stage-level representations, using developmental stages as auxiliary supervision to predict transferability on day four. Our experiments demonstrate that TransFACT, by leveraging an existing method designed for action recognition, achieves superior performance than its competitor in predicting embryo transferability.
>
---
#### [new 157] GLUT: 3D Gaussian Lookup Table for Continuous Color Transformation
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出GLUT，用于连续颜色变换的3D高斯查找表。解决传统3D LUT存储效率与精度的矛盾，提升可解释性与编辑灵活性。**

- **链接: [https://arxiv.org/pdf/2605.19889](https://arxiv.org/pdf/2605.19889)**

> **作者:** Danna Xue; David Serrano-Lozano; Shaolin Su; Javier Vazquez-Corral
>
> **备注:** Project page: this https URL
>
> **摘要:** 3D Lookup Tables (3D LUTs) are widely used for color mapping, but their grid-based representation requires discretizing the RGB space, leading to a capacity-memory trade-off that becomes prohibitive when storing large numbers of LUTs. Recent approaches adopt implicit neural representations to improve scalability, yet their black-box nature limits interpretability and hinders intuitive, localized editing. In this paper, we propose Gaussian LUT (GLUT), a continuous and explicit color representation that models color transformations using a set of learnable 3D Gaussian primitives. By avoiding fixed-resolution grids, GLUT achieves flexible representational capacity while maintaining a compact memory footprint. Its explicit, spatially localized formulation further enables both accurate modeling and interpretability. Building on this representation, we introduce a compact conditional generator (CGLUT) that predicts GLUT parameters for multiple LUT instances, encoding diverse color styles in a single framework to enable smooth and controllable LUT style blending. Moreover, GLUT supports efficient, user-friendly editing by allowing localized adjustments to specific color regions without global retraining. Experimental results demonstrate that our approach outperforms prior neural LUT representations in both accuracy and efficiency, while offering improved interpretability and interactive control.
>
---
#### [new 158] CounterFlow: A Two-Phase Inference-Time Sampling for Counterfactual Video Foley Generation
- **分类: cs.MM; cs.AI; cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于视频音频生成任务，解决视频与文本不一致时的反事实音频生成问题。提出ConterFlow方法，在推理阶段分两步优化音频，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2605.18916](https://arxiv.org/pdf/2605.18916)**

> **作者:** Gyubin Lee; Junwon Lee; Juhan Nam
>
> **备注:** accepted to CVPR 2026 Workshop on Sight and Sound
>
> **摘要:** We investigate Counterfactual Video Foley Generation, which aims to adopt a sound-source identity that contradicts the visual evidence while remaining temporally synchronized to a silent video. Existing Video&Text-to-Audio (VT2A) models struggle with this, often remaining anchored to the visually implied sound source when video and text contents disagree. We present ConterFlow, an inference-time dual-phase sampling scheme for pretrained flow-matching VT2A models. Phase 1 builds a video-derived temporal structure while suppressing the visually implied source; Phase 2 drops video conditioning to focus entirely on shaping audio timbre toward the target prompt. ConterFlow substantially improves counterfactual Video Foley generation compared to naive negative prompting and state-of-the-art baselines. To evaluate replacement quality, we propose a metric leveraging a text-audio co-embedding space to measure both target-prompt evidence and residual visually implied source leakage. Video demonstrations and code are available at this https URL
>
---
#### [new 159] XFlowMap: Cross-Scale Generalization and Mapping of Massive Origin-Destination Data
- **分类: cs.SI; cs.CV**

- **简介: 该论文属于地理信息可视化任务，解决大规模OD数据映射中的多尺度模式识别与清晰表达问题。提出XFlowMap框架，实现自动化的多尺度流模式检测与可视化。**

- **链接: [https://arxiv.org/pdf/2605.18777](https://arxiv.org/pdf/2605.18777)**

> **作者:** Diansheng Guo; Hai Jin
>
> **摘要:** Mapping large origin-destination (OD) datasets remains challenging because flow maps become cluttered, meaningful patterns occur at multiple spatial scales, and existing flow-mapping approaches frequently rely on predefined aggregation units or manual generalization. This paper presents XFlowMap, a framework for the cross-scale generalization and mapping of massive OD data. Specifically, the framework integrates cross-scale flow pattern (cluster) detection, automated flow map generalization, and a new cartographic representation for analyzing and visualizing complex origin-destination flow structures. The approach detects salient flow patterns at their appropriate origin and destination scales, extracts high-level structures, and generates a new flow map representation that supports holistic interpretation of complex origin-destination flow patterns. A scan-statistic-based procedure is developed to evaluate and generalize cross-scale flow clusters. The detected clusters are then visualized using a novel flow symbol that integrates location, direction, strength, and OD scales in a single representation. The framework supports both area-based and point-based OD data, is robust to sparse and noisy datasets, and enables comparative mapping of stratified flow data. Experiments with synthetic data and U.S. migration data demonstrate that the method effectively extracts meaningful cross-scale flow patterns and produces clear, information-rich flow maps for large mobility datasets, supporting both static presentation and interactive exploration.
>
---
#### [new 160] A Multi-Dimensional Clustering Approach for Identifying Inborn Errors of Immunity
- **分类: cs.LG; cs.CV; q-bio.QM**

- **简介: 该论文属于罕见病诊断任务，旨在通过多维聚类方法识别免疫缺陷疾病。解决数据处理与模式识别难题，构建数据处理管道并优化聚类算法。**

- **链接: [https://arxiv.org/pdf/2605.18880](https://arxiv.org/pdf/2605.18880)**

> **作者:** Nishad Kulkarni; Alexandra K. Martinson; Nicholas L. Rider; Michael Keller; Syed Muhammad Anwar
>
> **备注:** Accepted at EMBC 2026
>
> **摘要:** Rare diseases such as inborn errors of immunity (IEI) require early diagnosis to prevent end organ damage and improve quality of life. Hurdles in accessing and curating large scale electronic health record (EHR) data limit routine data driven analyses to remain on the forefront of IEI and other rare disease trends. Development of machine learning (ML) algorithms in IEI for pattern recognition as well as published methodology examining how to systematically process and integrate complex medical data is limited. Our proposed pipeline, including data curation and ML clustering algorithms, is designed to recognize novel rare disease patterns and extract IEI- associated features from a national data registry. Our methodology for EHR data formatting and processing presents the pipeline that transforms raw immunologic lab data into vectors. This is further combined with hyperparameter tuning for diseases pattern recognition via clustering. This study refines IEI feature awareness, develops data tool kits for rare disease populations analysis, and expands on transforming complex medical records in data structures interpretable by unsupervised ML.
>
---
#### [new 161] From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉语言模型训练任务，旨在解决模型感知与推理能力不足的问题。通过分阶段训练提升视觉感知，优化推理效果，实验表明该方法有效提升性能。**

- **链接: [https://arxiv.org/pdf/2605.20177](https://arxiv.org/pdf/2605.20177)**

> **作者:** Juncheng Wu; Hardy Chen; Haoqin Tu; Xianfeng Tang; Freda Shi; Hui Liu; Hanqing Lu; Cihang Xie; Yuyin Zhou
>
> **备注:** 19 pages, 9 figures; Accepted to ICML 2026; Project Page: this https URL
>
> **摘要:** Recent advances in vision-language models (VLMs) emphasize long chain-of-thought reasoning; yet, we find that their performance on visual tasks is primarily limited by a lack of visual perception as opposed to reasoning itself. In this work, we systematically study the interplay between perception and reasoning in VLM post-training by decomposing their capabilities into three separate training stages: visual perception, visual reasoning, and textual reasoning, incorporating specialized training data. We demonstrate that visual perception (a) requires targeted optimization with specialized data; (b) serves as a fundamental scaffold that should be solidified through staged training before refining visual reasoning; and (c) is more effectively learned via RL than caption-based SFT. Our experiments across multiple VLMs demonstrate that staged training consistently improves both visual perception and reasoning performance over merged training. Notably, models trained with our approach achieve 1.5% higher reasoning accuracy with 20.8% shorter reasoning traces, suggesting that superior perception reduces the need for excessive reasoning. Furthermore, we show that this capability-based staging represents a new curriculum dimension orthogonal to traditional difficulty-based curricula, and combining both yields further additive gains. Our staged-training models achieve superior performance among open-weight VLMs, establishing advanced results on several visual math and perception (e.g., +5.2% on WeMath and +3.7% on RealWorldQA) tasks compared with the base counterpart.
>
---
#### [new 162] HEAT: Heterogeneous End-to-End Autonomous Driving via Trajectory-Guided World Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，解决多领域泛化问题。通过轨迹驱动学习和世界模型，提升模型在异构环境中的性能。**

- **链接: [https://arxiv.org/pdf/2605.19631](https://arxiv.org/pdf/2605.19631)**

> **作者:** Hoonhee Cho; Giwon Lee; Jae-Young Kang; Hyemin Yang; Heejun Park; Kuk-Jin Yoon
>
> **摘要:** End-to-end autonomous driving has emerged as a compelling alternative to traditional modular pipelines by directly mapping raw sensor data to driving actions. While recent approaches achieve strong performance on single-domain datasets, their performance degrades significantly when trained jointly across multiple heterogeneous domains. In practice, however, autonomous systems must operate across diverse environments with heterogeneous distributions, including different cities, sensor configurations, and traffic patterns, without domain-specific retraining. This gap highlights a key challenge in multi-domain learning: domain-specific variations across heterogeneous domains introduce conflicting learning signals, driving models toward compromised solutions that are suboptimal across domains. To address this, we propose a trajectory-driven learning paradigm that organizes training around planning trajectories, enabling the model to capture domain-invariant representations of driving intent. Furthermore, we incorporate a world model that predicts future latent features conditioned on ego actions, improving feature consistency and mitigating domain-induced biases. We evaluate our approach on three benchmarks, nuScenes, NAVSIM, and the Waymo end-to-end dataset, and show substantial improvements over existing methods across all domains. Our results demonstrate that a single unified model can be trained on heterogeneous datasets while maintaining strong performance within each domain, highlighting a step toward scalable real-world deployment. We will make our code publicly available.
>
---
#### [new 163] From Llama to Cria: Scaling Down Neural Networks via Neuron-Level Spectral Structural Importance Evaluation
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于神经网络压缩任务，旨在通过神经元剪枝减小模型规模。工作包括构建图结构分析神经元重要性，迭代剪枝并进行微调以保持性能。**

- **链接: [https://arxiv.org/pdf/2605.18860](https://arxiv.org/pdf/2605.18860)**

> **作者:** Yongyu Wang
>
> **摘要:** This paper proposes a neuron pruning framework based on neuron-level spectral structural importance evaluation. Given a trained neural network, we record the hidden states of each hidden layer during inference and model neurons as graph nodes, with hidden states treated as graph signals. Using ideas from graph signal processing, we infer layer-wise input and output graphs that characterize the structural relationships among neurons before and after each layer transformation. We then evaluate the spectral structural importance of neurons by analyzing the transformation between these graphs based on spectral graph theory. Neurons with high spectral structural importance are regarded as strongly involved in the internal representation transformation and are therefore preserved, while neurons with low importance scores are selected as pruning candidates. The pruning process is conducted iteratively until a predefined effective parameter reduction target is reached. Instead of fine-tuning after every pruning step, the proposed strategy first removes low-importance neurons to obtain a compact architecture and then applies a final recovery fine-tuning stage to restore task performance. By connecting neuron pruning with graph signal processing and spectral structural analysis, the proposed framework offers a principled way to reduce neural network size while maintaining solution quality. Experimental results on CIFAR-10 image classification and SST-2 sentiment classification show that our method can effectively remove low-importance neurons and achieve compact networks with competitive performance after recovery fine-tuning.
>
---
#### [new 164] Decentralized Direct Volume Rendering: A Browser-Native GPU Architecture for MRI Digital Twins in Resource-Constrained Settings
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决资源受限环境下数字孪生技术部署难的问题。通过浏览器原生GPU架构实现MRI数字孪生的实时交互渲染。**

- **链接: [https://arxiv.org/pdf/2605.19737](https://arxiv.org/pdf/2605.19737)**

> **作者:** Oserebameh Augustine Beckley
>
> **备注:** 10 pages, 4 figures. Live interactive browser demo available at: this https URL . Source code repository: this https URL
>
> **摘要:** Digital Twin (DT) technology holds immense potential for surgical planning and personalized medicine. However, generating interactive, patient-specific anatomical twins currently relies on computationally heavy Server-Side Rendering (SSR) or expensive local workstations, creating significant barriers to deployment, especially in resource-constrained settings (RCS). This paper presents a decentralized, client-side WebGPU architecture that democratizes access to high-fidelity anatomical Digital Twins. By bypassing standard server-side rendering pipelines, the framework executes deterministic single-pass raymarching and morphological gradient calculations directly on low-cost integrated edge GPUs. Eliminating the network latency inherent to cloud-rendered solutions, the system achieves a Time to First Pixel (TTFP) of under 920.0ms and maintains stable interactivity at >= 82.0 FPS. Continuous Interaction Fidelity is maintained via uniform buffers, enabling zero-latency manipulation of tissue parameters for dynamic clinical decision-making. By proving that complex 3D medical simulations of patient-specific MRI scan can be executed natively in the browser without deep learning or external computational dependencies, this architecture provides a scalable, affordable foundation for the widespread clinical adoption of healthcare Digital Twins.
>
---
#### [new 165] AQuaUI: Visual Token Reduction for GUI Agents with Adaptive Quadtrees
- **分类: cs.AI; cs.CV; cs.MA**

- **简介: 该论文属于GUI代理任务，旨在解决高分辨率截图中视觉token冗余问题。提出AQuaUI方法，通过自适应四叉树减少token数量，提升效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2605.19260](https://arxiv.org/pdf/2605.19260)**

> **作者:** Yuankai Li; Tinghui Zhu; Ha Min Son; Zhe Zhao; Xin Liu; Muhao Chen
>
> **摘要:** Large Multimodal Models (LMMs) have recently emerged as promising backbones for GUI-agent models, where high-resolution GUI screenshots are introduced to the prompts at each iteration step. However, these screenshots exhibit highly non-uniform spatial information density: large regions may carry little information and are visually homogeneous, while key text and icons may require high visual fidelity. Existing approaches to this problem either require additional training or rely on attention-based token compression, ignoring the structured layout and spatial redundancy of GUI screenshots. To fill the gap, this paper proposes AquaUI, a training-free inference-time token reduction method for GUI agent models that utilizes the non-uniform information density in screenshots. AQuaUI constructs an adaptive quadtree on each screenshot input and keeps one representative merged token per leaf of the quadtree. AQuaUI preserves the spatial positions of retained tokens throughout the pipeline to ensure that all position-encoding stages remain consistent. To further improve temporal consistency across multi-step GUI interactions, we propose a conditional quadtree algorithm that leverages the continuity between consecutive screenshots within a single request. Specifically, it refines the current quadtree using previous quadtrees as references, helping preserve fine-grained regions across static or mildly shifted GUI states. We implement AQuaUI on state-of-the-art GUI agent models and conduct experiments on standard grounding and navigational benchmarks. AQuaUI consistently shows improved accuracy-efficiency trade-offs over prior baselines. Notably, on GUI-Owl-1.5-32B-Instruct, AQuaUI achieves up to 13.22% speedup and 29.52% fewer visual tokens while retaining 99.06% of full-token performance, suggesting that the spatial redundancy of GUI screenshots can be exploited at inference without retraining.
>
---
#### [new 166] CEPO: RLVR Self-Distillation using Contrastive Evidence Policy Optimization
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文提出CEPO方法，解决RLVR中奖励信号不区分关键推理步骤与填充词的问题。通过对比正确与错误答案，增强关键步骤的信用分配，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.19436](https://arxiv.org/pdf/2605.19436)**

> **作者:** Ahmed Heakl; Abdelrahman M. Shaker; Youssef Mohamed; Rania Elbadry; Omar Fetouh; Fahad Shahbaz Khan; Salman Khan
>
> **备注:** 9 pages
>
> **摘要:** When a model produces a correct solution under reinforcement learning with verifiable rewards (RLVR), every token receives the same reward signal regardless of whether it was a decisive reasoning step or a grammatical filler. A natural fix is to condition the model on the correct answer as a teacher, identifying tokens it would have generated differently had it known the answer. Prior work shows this either corrupts training by leaking the answer into the gradient, or produces a weak signal that cannot distinguish decisive steps from filler, since both look equally surprising relative to the model's baseline. We propose Contrastive Evidence Policy Optimization (CEPO), which asks a sharper question at every token: not just "does the correct answer favor this token?" but "does the correct answer favor it while the wrong answer disfavors it?" A token satisfying both is a genuine reasoning step; one satisfying neither is filler. The wrong-answer teacher is constructed from rejected rollouts already in the training batch, incurring no additional sampling cost. We prove CEPO inherits all structural safety guarantees of the prior state of the art while strictly sharpening credit at decisive tokens, with the improvement vanishing exactly at filler positions. Empirically, CEPO achieves 43.43% and 60.56% average accuracy across five multimodal mathematical reasoning benchmarks at 2B and 4B scale, respectively, versus 41.17% and 57.43% for GRPO under identical training budgets. Distribution-matching self-distillation methods (OPSD, SDPO) fall below the untrained baseline, empirically confirming the information leakage our theory predicts. Our code is available at this https URL.
>
---
#### [new 167] Prognostic Value of Lung Ultrasound Biomarkers for Readmission Risk in Congestive Heart Failure: A Pilot Data-Driven Analysis
- **分类: eess.SP; cs.CV; cs.LG; eess.IV**

- **简介: 该论文属于医疗预测任务，旨在通过肺部超声生物标志物预测心衰患者30天内再入院风险。研究利用机器学习分析超声图像，发现关键影像因素并提升预测性能。**

- **链接: [https://arxiv.org/pdf/2605.18878](https://arxiv.org/pdf/2605.18878)**

> **作者:** Jana Armouti; Laura Hutchins; Jacob Duplantis; Thomas Deiss; Thales Nogueira Gomes; Keyur H. Patel; Seema Walvekar; Shane Guillory; Thomas H. Fox; Amita Krishnan; Ricardo Rodriguez; Bennett DeBoisblanc; Deva Ramanan; John Galeotti; Gautam Gare
>
> **摘要:** Hospital readmission within 30 days of discharge is a leading driver of morbidity, mortality, and avoidable healthcare expenditure in congestive heart failure (CHF). Current clinical risk stratification tools rely primarily on non-imaging data and exhibit limited predictive performance. Point-of-care lung ultrasound (LUS) offers a sensitive, noninvasive window into the pulmonary congestion that characterizes CHF decompensation, yet its prognostic utility for readmission prediction remains largely unexplored. We present a pilot feasibility study, the first systematic machine learning study using B-mode LUS acquired during hospitalization to predict 30-day CHF readmission. Quantitative spatiotemporal embeddings are extracted from a pretrained Temporal Shift Module (TSM) ResNet-18 encoder, and interpretable biomarker features are separately evaluated. Through structured ablations over lung view, temporal representation, multi-view fusion, and cross-lung augmentation, we identify the key imaging factors driving readmission risk. Our findings reveal that (1) dependent lower-lung regions (Left-3, Right-3) carry the strongest prognostic signal, consistent with their greater susceptibility to hydrostatic congestion; (2) temporal difference features between sequential examinations substantially outperform single-timepoint representations, highlighting the importance of capturing disease trajectory; and (3) multi-view feature concatenation yields the best overall performance, with our top MLP model achieving an F1 score of 0.80 (95% CI: 0.62-0.96). Biomarker analysis further reveals that pleural-line abnormalities, including breaks and indentations, are as informative as the canonical A-line and B-line markers. These results support POCUS-derived biomarkers as practical, interpretable tools for noninvasive CHF risk stratification.
>
---
#### [new 168] PiG-Avatar: Hierarchical Neural-Field-Guided Gaussian Avatars
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出PiG-Avatar，解决3D虚拟人像重建中服装几何表达受限的问题，通过神经场引导的高斯表示实现更精确、灵活的重建。**

- **链接: [https://arxiv.org/pdf/2605.20185](https://arxiv.org/pdf/2605.20185)**

> **作者:** Julian Kaltheuner; Jan Spindler; Sina Kitz; Patrick Stotko; Reinhard Klein
>
> **摘要:** Existing Gaussian avatar methods typically parameterize geometry on a body-template surface, which entangles the avatar's representation space with the template's deformation space and limits the capture of layered, off-body, and non-rigid clothing geometry. We present PiG-Avatar, which addresses this limitation by using the parametric body model solely for kinematic transport, while representing the avatar as Gaussians anchored in a volumetric canonical space governed by a continuous neural field. This decouples representation from template topology, avoiding the geometric constraints of surface-based parameterizations. Kinematic coherence is maintained through 3D barycentric anchor transport, which guides motion without constraining geometry and allows anchors to deviate freely from the template surface, yielding dense, stable temporal surface correspondences by construction. To make this unconstrained formulation tractable, we introduce dual-level spatially coherent optimization, combining Sobolev-preconditioned neural-field updates with a novel KNN-based preconditioning of canonical anchor geometry. Together, these mechanisms induce an emergent self-organization of anchor density: anchors migrate toward regions of high curvature, appearance variation, and non-coherent motion without explicit heuristics. As a result, complex clothing geometry and layered surfaces emerge as natural, high-fidelity outputs. This single representation further supports hierarchical reconstruction across multiple levels of detail, with coarse-level supervision propagating to finer levels through the shared field and coupled anchor graph. On established benchmarks featuring subjects with complex clothing and challenging non-rigid motion, PiG-Avatar achieves state-of-the-art rendering quality, generalizes robustly to imperfect body model initialization, and renders in real time across all detail levels.
>
---
## 更新

#### [replaced 001] DLEBench: Evaluating Small-scale Object Editing Ability for Instruction-based Image Editing Model
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.23622](https://arxiv.org/pdf/2602.23622)**

> **作者:** Shibo Hong; Boxian Ai; Jun Kuang; Wei Wang; FengJiao Chen; Zhongyuan Peng; Chenhao Huang; Yixin Cao
>
> **摘要:** Significant progress has been made in the field of Instruction-based Image Editing Models (IIEMs). However, while these models demonstrate plausible adherence to instructions and strong reasoning ability on current benchmarks, their ability to edit small objects remains underexplored, despite its importance for precise local editing and refining details in both real and generated images. In this paper, we introduce DeepLookEditBench (DLEBench), the first benchmark dedicated to assessing the abilities of IIEMs in editing small-scale objects. Specifically, we construct a challenging testbed comprising 1889 samples across seven instruction types. In these samples, target objects occupy only 1%-10% of the image area, covering complex scenarios such as partial occlusion and multi-object editing. To ensure robust evaluation on this benchmark, we propose an evaluation protocol with refined score rubrics to minimize subjectivity and ambiguity in two criteria: Instruction Following and Visual Consistency. This protocol also introduces a dual-mode evaluation framework (Tool-driven and Oracle-guided Modes) addressing the misalignment between LMM-as-a-Judge and human judgements on DLEBench. Empirical results on 10 IIEMs reveal significant performance gaps in small-scale object editing, highlighting the need for specialized benchmarks to advance this ability.
>
---
#### [replaced 002] TrajectoryMover: Generative Movement of Object Trajectories in Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.29092](https://arxiv.org/pdf/2603.29092)**

> **作者:** Kiran Chhatre; Hyeonho Jeong; Yulia Gryaditskaya; Christopher E. Peters; Chun-Hao Paul Huang; Paul Guerrero
>
> **备注:** 15 pages, 9 figures. Project page: this https URL
>
> **摘要:** Generative video editing has enabled several intuitive editing operations for short video clips that would previously have been difficult to achieve, especially for non-expert editors. Existing methods focus on prescribing an object's 3D or 2D motion trajectory in a video, or on altering the appearance of an object or a scene, while preserving both the video's plausibility and identity. Yet a method to move an object's 3D motion trajectory in a video, i.e., moving an object while preserving its relative 3D motion, is currently still missing. The main challenge lies in obtaining paired video data for this scenario. Previous methods typically rely on clever data generation approaches to construct plausible paired data from unpaired videos, but this approach fails if one of the videos in a pair can not easily be constructed from the other. Instead, we introduce TrajectoryAtlas, a new data generation pipeline for large-scale synthetic paired video data and a video generator TrajectoryMover fine-tuned with this data. We show that this successfully enables generative movement of object trajectories. Project page: this https URL
>
---
#### [replaced 003] SEAL: Semantic Aware Image Watermarking
- **分类: cs.LG; cs.CR; cs.CV**

- **链接: [https://arxiv.org/pdf/2503.12172](https://arxiv.org/pdf/2503.12172)**

> **作者:** Kasra Arabi; R. Teal Witter; Chinmay Hegde; Niv Cohen
>
> **摘要:** Generative models have rapidly evolved to generate realistic outputs. However, their synthetic outputs increasingly challenge the clear distinction between natural and AI-generated content, necessitating robust watermarking techniques. Watermarks are typically expected to preserve the integrity of the target image, withstand removal attempts, and prevent unauthorized replication onto unrelated images. To address this need, recent methods embed persistent watermarks into images produced by diffusion models using the initial noise. Yet, to do so, they either distort the distribution of generated images or rely on searching through a long dictionary of used keys for detection. In this paper, we propose a novel watermarking method that embeds semantic information about the generated image directly into the watermark, enabling a distortion-free watermark that can be verified without requiring a database of key patterns. Instead, the key pattern can be inferred from the semantic embedding of the image using locality-sensitive hashing. Furthermore, conditioning the watermark detection on the original image content improves robustness against forgery attacks. To demonstrate that, we consider two largely overlooked attack strategies: (i) an attacker extracting the initial noise and generating a novel image with the same pattern; (ii) an attacker inserting an unrelated (potentially harmful) object into a watermarked image, possibly while preserving the watermark. We empirically validate our method's increased robustness to these attacks. Taken together, our results suggest that content-aware watermarks can mitigate risks arising from image-generative models.
>
---
#### [replaced 004] AnyAct: Towards Human Reenactment of Character Motion From Video
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2605.15497](https://arxiv.org/pdf/2605.15497)**

> **作者:** Liuhan Chen; Lei Zhong; Jiewei Wang; Qin Shuai; Li Yuan; Leidong Fan; Qing Li; Kanglin Liu
>
> **备注:** 12 pages
>
> **摘要:** We study the problem of directly deriving an initial human reenactment from a monocular video of a non-human character. Our goal is not to reconstruct the source character itself but to reinterpret its motion as a plausible and editable human performance for downstream animation authoring. This task is challenging because existing video-based motion capture methods are largely restricted to human-centric structural spaces, while motion retargeting methods typically require structured 3D source motions and known source topologies. Our key insight is that sparse local articulated motion cues can preserve essential dynamics across large structural differences, providing a stable bridge from character video to human reenactment. Based on this observation, we propose AnyAct, which formulates character-video-driven human reenactment as conditional human motion generation from transferable sparse local 2D articulated motion. To make this practical, we introduce three key designs: human-motion-only supervision via augmented 3D-to-2D projection, progressive 3D-to-2D training to alleviate conditioning ambiguity, and global-local motion decoupling for reliable local motion control. We further construct a benchmark primarily covering diverse non-human character videos. Experiments on the benchmark show that AnyAct produces high-fidelity initial human reenactments that preserve the essential dynamics of the characters in reference videos, and further ablation studies validate the effectiveness of its core designs.
>
---
#### [replaced 005] NEWTON: Agentic Planning for Physically Grounded Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.18396](https://arxiv.org/pdf/2605.18396)**

> **作者:** Yuxiang Feng; Juncheng Wang; Chao Xu; Yijie Qian; Huihan Wang; Wenlong Hou; Yang Liu; Baigui Sun; Yong Liu; Shujun Wang
>
> **备注:** project page: this https URL
>
> **摘要:** Video generation models produce visually compelling results but systematically violate physical commonsense -- on VideoPhy-2, the best model achieves only 32.6% joint accuracy. We identify a specification bottleneck: text prompts are lossy compression of the physical world, omitting the parameters that fully determine dynamics, and no amount of model scaling can recover what was never specified. From this diagnosis we derive three properties that physics conditioning must satisfy -- sufficiency, dynamism, and verifiability -- and show that no existing approach satisfies all three. We present NEWTON, in which video generation is demoted from the system output to one action inside an agent's toolbox: a learned planner orchestrates physics-aware tools (keyframe generation, scientific computation, prompt refinement) to construct rich conditioning, and a verifier closes the loop for iterative re-planning. The planner is the sole trainable component, optimized on-policy via Flow-GRPO inside the live multi-turn loop. On VideoPhy-2, NEWTON improves joint accuracy from 21.4% to 29.7% on LTX-Video and from 30.7% to 37.4% on Veo-3.1, without modifying either generator. Our project page: this https URL
>
---
#### [replaced 006] MapAnything: Evaluating Monocular Metric Depth Models for 3D Urban Asset Localization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.14839](https://arxiv.org/pdf/2509.14839)**

> **作者:** Miriam Louise Carnot; Jonas Kunze; Erik Quinten Fastermann; Eric Peukert; André Ludwig; Bogdan Franczyk
>
> **摘要:** City administrations increasingly rely on comprehensive databases and urban digital twins of city assets, such as traffic signs and trees, as well as incidents like graffiti or road damage, to maintain an effective overview of urban conditions. Digitization has increased the demand for continuously updated spatial datasets, yet current data acquisition and maintenance processes still involve considerable manual effort, posing significant scalability challenges. This paper introduces MapAnything, a novel geo-localization framework that automates the spatial mapping of urban objects and incidents from a single monocular image. By leveraging advanced Metric Depth Estimation models, MapAnything accurately calculates object geocoordinates, converting 2D image data into valuable 3D spatial information. The methodology integrates the estimated camera-to-object distance with geometric principles and known camera specifications. We present a detailed validation of the framework, comparing its distance-estimation accuracy against high-precision LiDAR point clouds in complex urban environments. Our evaluation provides a granular analysis of spatial performance across various distance intervals and semantic areas, such as roads and vegetation. Finally, we demonstrate the framework's practical efficacy through specific use cases, including mapping traffic signs and road pavement damage, and provide recommendations for its integration into automated urban inventory systems.
>
---
#### [replaced 007] Perceptual misalignment of texture representations in convolutional neural networks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.01341](https://arxiv.org/pdf/2604.01341)**

> **作者:** Ludovica de Paolis; Fabio Anselmi; Alessio Ansuini; Eugenio Piasini
>
> **摘要:** Mathematical modeling of visual textures traces back to Julesz's intuition that texture perception in humans is based on local correlations between image features. An influential approach for texture analysis and generation generalizes this notion to linear correlations between the nonlinear features computed by convolutional neural networks (CNNs), compiled into Gram matrices. Given that CNNs are often used as models for the visual system, it is natural to ask whether such "texture representations" spontaneously align with the textures' perceptual content, and in particular whether those CNNs that are regarded as better models for the visual system also possess more human-like texture representations. Here we quantify the perceptual content captured by feature correlations computed for a diverse pool of CNNs, and we compare it to the models' perceptual alignment with the mammalian visual system as measured by Brain-Score. Surprisingly, we find that there is no connection between conventional measures of CNN quality as a model of the visual system and its alignment with human texture perception. We conclude that texture perception involves mechanisms that are distinct from those that are commonly modeled using approaches based on CNNs trained on object recognition, possibly depending on the integration of contextual information.
>
---
#### [replaced 008] FIKA-Bench: From Fine-grained Recognition to Fine-Grained Knowledge Acquisition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.13193](https://arxiv.org/pdf/2605.13193)**

> **作者:** Geng Li; Yuxin Peng
>
> **备注:** Project page with code: this https URL
>
> **摘要:** Fine-grained recognition in everyday life is often not a closed-book classification problem: when encountering unfamiliar objects, humans actively search, compare visual details, and verify evidence before deciding. Existing benchmarks primarily evaluate visually recognition, leaving this active external knowledge acquisition ability underexplored. We study fine-grained knowledge acquisition, where a system must seek, verify, and use external evidence to answer open-ended fine-grained recognition questions. We introduce FIKA-Bench, a leakage-aware and evidence-grounded collection of 311 public-source and real-life instances. To ensure high quality, every example is filtered against frontier closed-book models to remove memorized cases and audited to eliminate image-answer leakage, retaining only samples supported by verified evidence. Our evaluation of latest Large Multimodal Models (LMMs) and agents reveals that the task remains a formidable challenge: the best system reaches only 25.1% accuracy, with no model exceeding 30%. Crucially, we find that merely equipping models with tools is insufficient to bridge this gap; agent failures are predominantly driven by wrong entity retrieval and poor visual judgement. These results show that reliable knowledge acquisition needs better agent designs that focus on fine-grained recognition.
>
---
#### [replaced 009] PERL: Parameter Efficient Reasoning in CLIP Latent Space
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.18464](https://arxiv.org/pdf/2605.18464)**

> **作者:** Simone Carnemolla; Salvatore Calcagno; Daniela Giordano; Concetto Spampinato; Matteo Pennisi
>
> **备注:** Submitted to NeurIPS 2026
>
> **摘要:** Contrastively trained vision-language models such as CLIP provide strong zero-shot transfer by aligning images and text in a shared embedding space. However, adapting these models to downstream tasks without degrading their open-vocabulary generalization remains challenging. Existing parameter-efficient adaptation methods typically improve task specialization through learned prompts, adapters, or multimodal transformations, where adaptation capacity is primarily expressed through additional trainable parameters. Inspired by recent latent reasoning methods in language models, we investigate a complementary perspective: can adaptation emerge from iterative reasoning on latent representations rather than from increasing parameter count alone? We introduce PERL (Parameter-Efficient Reasoning in CLIP Latent Space), a lightweight adaptation framework that augments a frozen CLIP model with a compact shared reasoning module applied recurrently across refinement steps. At each step, PERL generates a latent reasoning token conditioned on the current representation and injects it into an intermediate encoder layer, progressively refining higher-level semantic representations while preserving CLIP's pretrained multimodal structure. Across 15 benchmarks spanning base-to-novel generalization, cross-dataset transfer, and out-of-distribution ImageNet variants, PERL achieves the best parameter-performance trade-off among the compared methods under a fast-adaptation few-shot setting, combining strong novel-class accuracy and competitive transfer performance with only about 6K trainable parameters, up to 817x fewer than the largest compared approach. Overall, our results suggest that iterative latent reasoning provides a complementary adaptation mechanism to parameter scaling in discriminative vision-language models.
>
---
#### [replaced 010] Seeing Together: Multi-Robot Cooperative Egocentric Spatial Reasoning with Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.18431](https://arxiv.org/pdf/2605.18431)**

> **作者:** Kunyu Peng; Zhikun Zhou; Kailun Yang; Di Wen; Ruiping Liu; Yufan Chen; Junwei Zheng; Hao Shi; Yi Zhou; M. Saquib Sarfraz; Danda Pani Paudel; Luc Van Gool
>
> **摘要:** Multimodal Large Language Models (MLLMs) have made substantial progress in egocentric video understanding, but their ability to reason cooperatively from multiple embodied viewpoints remains largely unexplored. We study this problem through multi-robot cooperative dynamic spatial reasoning, where a model must answer spatial, temporal, visibility, and coordination questions by integrating synchronized egocentric videos from a team of moving robots. To support this setting, we introduce CoopSR, the first benchmark for this task, together with EgoTeam, a multi-robot egocentric QA dataset. EgoTeam contains 114,227 QA pairs spanning 19 question types, four difficulty tiers, and three team sizes in Habitat and iGibson, along with a real-world test set of around 2,326 QAs collected using two quadruped robots. We further propose SP-CoR (Spectral and Physics-Informed Cooperative Reasoner), an MLLM framework for fine-grained cooperative spatial reasoning. SP-CoR combines dynamics-aware multi-robot frame sampling, spectral- and physics-guided view fusion, and physics-aligned prompt distillation, enabling the model to benefit from privileged robot-pose supervision during training while requiring only egocentric videos at test time. Across 22 MLLM baselines, SP-CoR consistently improves cooperative reasoning, outperforming the strongest fine-tuned baseline by +3.87% on Habitat and +7.12% on iGibson. It also shows stronger generalization to unseen team sizes and real-world robot tests. Code can be found at this https URL.
>
---
#### [replaced 011] Adaptive Camera Sensor for Vision Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2503.02170](https://arxiv.org/pdf/2503.02170)**

> **作者:** Eunsu Baek; Sunghwan Han; Taesik Gong; Hyung-Sin Kim
>
> **备注:** The International Conference on Learning Representations (ICLR 2025)
>
> **摘要:** Domain shift remains a persistent challenge in deep-learning-based computer vision, often requiring extensive model modifications or large labeled datasets to address. Inspired by human visual perception, which adjusts input quality through corrective lenses rather than over-training the brain, we propose Lens, a novel camera sensor control method that enhances model performance by capturing high-quality images from the model's perspective rather than relying on traditional human-centric sensor control. Lens is lightweight and adapts sensor parameters to specific models and scenes in real-time. At its core, Lens utilizes VisiT, a training-free, model-specific quality indicator that evaluates individual unlabeled samples at test time using confidence scores without additional adaptation costs. To validate Lens, we introduce ImageNet-ES Diverse, a new benchmark dataset capturing natural perturbations from varying sensor and lighting conditions. Extensive experiments on both ImageNet-ES and our new ImageNet-ES Diverse show that Lens significantly improves model accuracy across various baseline schemes for sensor control and model modification while maintaining low latency in image captures. Lens effectively compensates for large model size differences and integrates synergistically with model improvement techniques. Our code and dataset are available at this http URL.
>
---
#### [replaced 012] A Lightweight Transformer for Pain Recognition from Brain Activity
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.16491](https://arxiv.org/pdf/2604.16491)**

> **作者:** Stefanos Gkikas; Christian Arzate Cruz; Yu Fang; Lu Cao; Muhammad Umar Khan; Thomas Kassiotis; Giorgos Giannakakis; Raul Fernandez Rojas; Randy Gomez
>
> **摘要:** Pain is a multifaceted and widespread phenomenon with substantial clinical and societal burden, making reliable automated assessment a critical objective. This paper presents a lightweight transformer architecture that fuses multiple fNIRS representations through a unified tokenization mechanism, enabling joint modeling of complementary signal views without requiring modality-specific adaptations or increasing architectural complexity. The proposed token-mixing strategy preserves spatial, temporal, and time-frequency characteristics by projecting heterogeneous inputs onto a shared latent representation, using a structured segmentation scheme to control the granularity of local aggregation and global interaction. The model is evaluated on the AI4Pain dataset using stacked raw waveform and power spectral density representations of fNIRS inputs. Experimental results demonstrate competitive pain recognition performance while remaining computationally compact, making the approach suitable for real-time inference on both GPU and CPU hardware.
>
---
#### [replaced 013] 3D Modeling and Automated Measurement of Concrete Cracks via Segment Anything Refinement and Visual Inertial LiDAR Fusion
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于混凝土裂缝检测任务，解决传统方法在复杂场景下的适应性差、精度不足问题，通过融合视觉与LiDAR数据实现精准3D裂缝建模与测量。**

- **链接: [https://arxiv.org/pdf/2501.09203](https://arxiv.org/pdf/2501.09203)**

> **作者:** Pengru Deng; Jiapeng Yao; Chun Li; Su Wang; Xinrun Li; Varun Ojha; Xuhui He
>
> **备注:** Title and author list updated
>
> **摘要:** Visual-Spatial Systems has become increasingly essential in concrete crack inspection. However, existing methods often lacks adaptability to diverse scenarios, exhibits limited robustness in image-based approaches, and struggles with curved or complex geometries. To address these limitations, an innovative framework for two-dimensional (2D) crack detection, three-dimensional (3D) reconstruction, and 3D automatic crack measurement was proposed by integrating computer vision technologies and multi-modal Simultaneous localization and mapping (SLAM) in this study. Firstly, building on a base DeepLabv3+ segmentation model, and incorporating specific refinements utilizing foundation model Segment Anything Model (SAM), we developed a crack segmentation method with strong generalization across unfamiliar scenarios, enabling the generation of precise 2D crack masks. To enhance the accuracy and robustness of 3D reconstruction, Light Detection and Ranging (LiDAR) point clouds were utilized together with image data and segmentation masks. By leveraging both image- and LiDAR-SLAM, we developed a multi-frame and multi-modal fusion framework that produces dense, colorized point clouds, effectively capturing crack semantics at a 3D real-world scale. Furthermore, the crack geometric attributions were measured automatically and directly within 3D dense point cloud space, surpassing the limitations of conventional 2D image-based measurements. This advancement makes the method suitable for structural components with curved and complex 3D geometries. Experimental results across various concrete structures highlight the significant improvements and unique advantages of the proposed method, demonstrating its effectiveness, accuracy, and robustness in real-world applications.
>
---
#### [replaced 014] Efficient Transferable Optimal Transport via Min-Sliced Transport Plans
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19741](https://arxiv.org/pdf/2511.19741)**

> **作者:** Xinran Liu; Elaheh Akbari; Rocio Diaz Martin; Navid NaderiAlizadeh; Soheil Kolouri
>
> **摘要:** Optimal Transport (OT) offers a powerful framework for finding correspondences between distributions and addressing matching and alignment problems in various areas of computer vision, including shape analysis, image generation, and multimodal tasks. The computation cost of OT, however, hinders its scalability. Slice-based transport plans have recently shown promise for reducing the computational cost by leveraging the closed-form solutions of 1D OT problems. These methods optimize a one-dimensional projection (slice) to obtain a conditional transport plan that minimizes the transport cost in the ambient space. While efficient, these methods leave open the question of whether learned optimal slicers can transfer to new distribution pairs under distributional shift. Understanding this transferability is crucial in settings with evolving data or repeated OT computations across closely related distributions. In this paper, we study the min-Sliced Transport Plan (min-STP) framework and investigate the transferability of optimized slicers: can a slicer trained on one distribution pair yield effective transport plans for new, unseen pairs? Theoretically, we show that optimized slicers remain close under slight perturbations of the data distributions, enabling efficient transfer across related tasks. To further improve scalability, we introduce a minibatch formulation of min-STP and provide statistical guarantees on its accuracy. Empirically, we demonstrate that the transferable min-STP achieves strong one-shot matching performance and facilitates amortized training for point cloud alignment and flow-based generative modeling.
>
---
#### [replaced 015] VGGT-Edit: Feed-forward Native 3D Scene Editing with Residual Field Prediction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.15186](https://arxiv.org/pdf/2605.15186)**

> **作者:** Kaixin Zhu; Yiwen Tang; Yifan Yang; Renrui Zhang; Bohan Zeng; Ziyu Guo; Ruichuan An; Zhou Liu; Qizhi Chen; Delin Qu; Jaehong Yoon; Wentao Zhang
>
> **摘要:** High-quality 3D scene reconstruction has recently advanced toward generalizable feed-forward architectures, enabling the generation of complex environments in a single forward pass. However, despite their strong performance in static scene perception, these models remain limited in responding to dynamic human instructions, which restricts their use in interactive applications. Existing editing methods typically rely on a 2D-lifting strategy, where individual views are edited independently and then lifted back into 3D space. This indirect pipeline often leads to blurry textures and inconsistent geometry, as 2D editors lack the spatial awareness required to preserve structure across viewpoints. To address these limitations, we propose VGGT-Edit, a feed-forward framework for text-conditioned native 3D scene editing. VGGT-Edit introduces depth-synchronized text injection to align semantic guidance with the backbone's spatial poses, ensuring stable instruction grounding. This semantic signal is then processed by a residual transformation head, which directly predicts 3D geometric displacements to deform the scene while preserving background stability. To ensure high-fidelity results, we supervise the framework with a multi-term objective function that enforces geometric accuracy and cross-view consistency. We also construct the DeltaScene Dataset, a large-scale dataset generated through an automated pipeline with 3D agreement filtering to ensure ground-truth quality. Experiments show that VGGT-Edit substantially outperforms 2D-lifting baselines, producing sharper object details, stronger multi-view consistency, and near-instant inference speed. The project page is this https URL.
>
---
#### [replaced 016] A Grid-Based Framework for E-Scooter Demand Representation and Temporal Input Design for Deep Learning: Evidence from Austin, Texas
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.13609](https://arxiv.org/pdf/2603.13609)**

> **作者:** Mohammad Sahnoon; Merkebe Getachew Demissie; Roberto Souza
>
> **备注:** 16 pages, 7 tables, 10 figures
>
> **摘要:** Despite progress in deep learning for shared micromobility demand prediction, the systematic design and statistical validation of temporal input structures remain underexplored. Temporal features are often selected heuristically, even though historical demand strongly affects model performance and generalizability. This paper introduces a reproducible data-processing pipeline and a statistically grounded method for designing temporal input structures for image-to-image demand prediction. Using large-scale e-scooter data from Austin, Texas, we build a grid-based spatiotemporal dataset by converting trip records into hourly pickup and dropoff demand images. The pipeline includes trip filtering, mapping Census Tracts to spatial locations, grid construction, demand aggregation, and creation of a global activity mask that limits evaluation to historically active areas. This representation supports consistent spatial learning while preserving demand patterns. We then introduce a combined correlation- and error-based procedure to identify informative historical inputs. Optimal temporal depth is selected through an ablation study using a baseline UNET model with paired non-parametric tests and Holm correction. The resulting temporal structures capture short-term persistence as well as daily and weekly cycles. Compared with adjacent-hour and fixed-period baselines, the proposed design reduces mean squared error by up to 37 percent for next-hour prediction and 35 percent for next-24-hour prediction. These results highlight the value of principled dataset construction and statistically validated temporal input design for spatiotemporal micromobility demand prediction.
>
---
#### [replaced 017] StrLoRA: Towards Streaming Continual Visual Instruction Tuning for MLLMs
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.16353](https://arxiv.org/pdf/2605.16353)**

> **作者:** Chang Che; Ziqi Wang; Hui Ma; Cheems Wang; Zenglin Shi
>
> **摘要:** Continual Visual Instruction Tuning (CVIT) enables Multimodal Large Language Models to incrementally acquire new abilities. However, existing CVIT methods operate under a restrictive task-incremental setting, where each training phase corresponds to a single, predefined task. This does not reflect real-world conditions, where data arrives as a continuous stream of interleaved and dynamically evolving tasks. To bridge this gap, we introduce Streaming CVIT (StrCVIT), a more general and realistic setting where models learn from a stream of data chunks containing a dynamic mixture of tasks. In StrCVIT, a model must simultaneously acquire new abilities, reinforce recurring abilities, and mitigate forgetting. Existing CVIT methods fail here as they cannot reliably distinguish or adapt to the heterogeneous task samples within each chunk. We therefore propose StrLoRA, a regularized two-stage expert routing framework. StrLoRA first performs task-aware expert selection using the textual instruction to activate a sparse subset of relevant experts, reducing cross-task interference. It then applies token-wise expert weighting within this subset, where contribution weights are computed via cross-modal attention between local visual tokens and the global instruction representation. To maintain stability across the non-stationary stream, a routing-stability regularization aligns current routing distributions with a historical exponential moving average reference. Extensive experiments on a newly developed StrCVIT benchmark show that StrLoRA substantially outperforms existing methods, effectively enhancing model's abilities from continuously evolving data streams. The code is available at this https URL.
>
---
#### [replaced 018] Taming Real-World Space-Time Video Super-Resolution with One-Step Diffusion
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2601.20308](https://arxiv.org/pdf/2601.20308)**

> **作者:** Shuoyan Wei; Feng Li; Chen Zhou; Runmin Cong; Yao Zhao; Huihui Bai
>
> **备注:** 12 pages, 9 figures
>
> **摘要:** Diffusion models have demonstrated exceptional success in video super-resolution (VSR), exhibiting powerful capabilities for generating fine-grained details. However, their potential for space-time video super-resolution (STVSR), which necessitates not only recovering realistic high-resolution visual content but also improving the frame rate with coherent temporal dynamics, remains largely underexplored. Moreover, existing STVSR methods predominantly address spatiotemporal upsampling under simple degradation assumptions, thus failing in real-world scenarios with complex unknown degradations. To address these challenges, we propose OSDEnhancer, the first framework that achieves robust STVSR in one-step diffusion. OSDEnhancer begins with a linear initialization to establish essential spatiotemporal structures and adapt the model for one-step reconstruction. It then applies a divide-and-conquer strategy, introducing the temporal coherence (TC) and texture enrichment (TE) LoRAs that progressively specialize in inter-frame dynamics modeling and fine-grained texture recovery, respectively, while collaborating during inference for enhanced overall performance. A bidirectional VAE decoder employs deformable recurrent blocks to leverage the multi-scale structure of the vanilla VAE, enhancing latent-to-pixel reconstruction through joint multi-scale deformable aggregation and inter-frame feature propagation. Experimental results demonstrate that the proposed method attains state-of-the-art performance with superior generalization in real-world scenarios. The code is available at this https URL.
>
---
#### [replaced 019] One-to-All Animation: Alignment-Free Character Animation and Image Pose Transfer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22940](https://arxiv.org/pdf/2511.22940)**

> **作者:** Shijun Shi; Jing Xu; Zhihang Li; Chunli Peng; Xiaoda Yang; Lijing Lu; Kai Hu; Jiangning Zhang
>
> **备注:** Project Page:this https URL
>
> **摘要:** Recent advances in diffusion models have greatly improved pose-driven character animation. However, existing methods are limited to spatially aligned reference-pose pairs with matched skeletal structures. Handling reference-pose misalignment remains unsolved. To address this, we present One-to-All Animation, a unified framework for high-fidelity character animation and image pose transfer for references with arbitrary layouts. First, to handle spatially misaligned reference, we reformulate training as a self-supervised outpainting task that transforms diverse-layout reference into a unified occluded-input format. Second, to process partially visible reference, we design a reference extractor for comprehensive identity feature extraction. Further, we integrate hybrid reference fusion attention to handle varying resolutions and dynamic sequence lengths. Finally, from the perspective of generation quality, we introduce identity-robust pose control that decouples appearance from skeletal structure to mitigate pose overfitting, and a token replace strategy for coherent long-video generation. Extensive experiments show that our method outperforms existing approaches. The code and model are available at this https URL.
>
---
#### [replaced 020] Pretraining Objective Matters in Extreme Low-Data FGVC: A Backbone-Controlled Study
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.15599](https://arxiv.org/pdf/2605.15599)**

> **作者:** Alexander Hackett; Srikanth Thudumu; Ginny Fisher; Jason Fisher
>
> **备注:** Presented at the 13th Workshop on Fine-Grained Visual Categorization (FGVC13) at CVPR 2026
>
> **摘要:** Extreme low-data fine-grained classification is common in expert domains where labeling is expensive, yet practitioners still need principled guidance for selecting pretrained encoders. We study emerald inclusion grading with a custom dataset of labeled images across three classes and ask: under matched backbone capacity, how does pretraining objective affect downstream representation quality? We compare four frozen ViT-B/16 encoders trained with supervised classification, contrastive learning (SigLIP2), masked reconstruction (MAE), and self-distillation (DINOv3), and evaluate them with leave-one-out cross-validation using linear and nonlinear probes. To control statistical noise in the low-N regime, we use permutation testing (N=1000) on macro one-vs-rest AUC. Supervised and contrastive encoders provide the strongest linear separability (logistic AUC: 0.768 and 0.735; SVM AUC: 0.739 and 0.697), while MAE improves under nonlinear probes (XGBoost AUC: 0.713). We find that DINOv3 underperforms across probe families in this domain. These results support a practical recommendation for extreme low-data FGVC: prioritize margin-enforcing pretraining objectives when data scarcity restricts probing to linear decision rules, and consider reconstruction-style encoders when nonlinear classifiers are feasible given dataset constraints.
>
---
#### [replaced 021] Adapted Center and Scale Prediction: More Stable and More Accurate
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2002.09053](https://arxiv.org/pdf/2002.09053)**

> **作者:** Wenhao Wang; Jusheng Zhang
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** Pedestrian detection benefits from deep learning technology and gains rapid development in recent years. Most of detectors follow general object detection frame, i.e. default boxes and two-stage process. Recently, anchor-free and one-stage detectors have been introduced into this area. However, their accuracies are unsatisfactory. Therefore, in order to enjoy the simplicity of anchor-free detectors and the accuracy of two-stage ones simultaneously, we propose some adaptations based on a detector, Center and Scale Prediction(CSP). The main contributions of our paper are: (1) We improve the robustness of CSP and make it easier to train. (2) We propose a novel method to predict width, namely compressing width. (3) We achieve the second best performance on CityPersons benchmark, i.e. 9.3% log-average miss rate(MR) on reasonable set, 8.7% MR on partial set and 5.6% MR on bare set, which shows an anchor-free and one-stage detector can still have high accuracy. (4) We explore some capabilities of Switchable Normalization which are not mentioned in its original paper. The code is publicly available at this https URL.
>
---
#### [replaced 022] No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.25722](https://arxiv.org/pdf/2603.25722)**

> **作者:** Hai X. Pham; David T. Hoffmann; Ricardo Guerrero; Brais Martinez
>
> **备注:** Accepted at CVPR 2026. 2nd rev: update github repo URL
>
> **摘要:** Contrastive vision-language (V&L) models remain a popular choice for various applications. However, several limitations have emerged, most notably the limited ability of V&L models to learn compositional representations. Prior methods often addressed this limitation by generating custom training data to obtain hard negative samples. Hard negatives have been shown to improve performance on compositionality tasks, but are often specific to a single benchmark, do not generalize, and can cause substantial degradation of basic V&L capabilities such as zero-shot or retrieval performance, rendering them impractical. In this work we follow a different approach. We identify two root causes that limit compositionality performance of V&Ls: 1) Long training captions do not require a compositional representation; and 2) The final global pooling in the text and image encoders lead to a complete loss of the necessary information to learn binding in the first place. As a remedy, we propose two simple solutions: 1) We obtain short concept centric caption parts using standard NLP software and align those with the image; and 2) We introduce a parameter-free cross-modal attention-pooling to obtain concept centric visual embeddings from the image encoder. With these two changes and simple auxiliary contrastive losses, we obtain SOTA performance on standard compositionality benchmarks, while maintaining or improving strong zero-shot and retrieval capabilities. This is achieved without increasing inference cost. We release the code for this work at this https URL.
>
---
#### [replaced 023] GRLoc: Geometric Representation Regression for Visual Localization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13864](https://arxiv.org/pdf/2511.13864)**

> **作者:** Changyang Li; Xuejian Ma; Lixiang Liu; Zhan Li; Qingan Yan; Yi Xu
>
> **摘要:** Absolute Pose Regression (APR) has emerged as a compelling paradigm for visual localization. However, APR models typically operate as black boxes, directly regressing a 6-DoF pose from a query image, which can lead to memorizing training views rather than understanding 3D scene geometry. In this work, we propose a geometrically-grounded alternative. Inspired by novel view synthesis, which renders images from intermediate geometric representations, we reformulate APR as its inverse that regresses the underlying 3D representations directly from the image, and we name this paradigm Geometric Representation Regression (GRR). Our model explicitly predicts two disentangled geometric representations in the world coordinate system: (1) a raymap's directions to estimate camera rotation, and (2) a corresponding pointmap to estimate camera translation. The final camera pose is then recovered from these geometric components using a differentiable deterministic solver. This disentangled approach, which separates the learned visual-to-geometry mapping from the final pose calculation, introduces a strong geometric prior into the network. We find that the explicit decoupling of rotation and translation predictions measurably boosts performance. We demonstrate state-of-the-art performance on 7-Scenes and Cambridge Landmarks datasets, validating that modeling the inverse rendering process is a more robust path toward generalizable absolute pose estimation.
>
---
#### [replaced 024] RELO: Reinforcement Learning to Localize for Visual Object Tracking
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.07379](https://arxiv.org/pdf/2605.07379)**

> **作者:** Xin Chen; Chuanyu Sun; Jiao Xu; Houwen Peng; Dong Wang; Huchuan Lu; Kede Ma
>
> **备注:** ICML 2026 paper
>
> **摘要:** Conventional visual object trackers localize targets using handcrafted spatial priors, often in the form of heatmaps. Such priors provide only surrogate supervision and are poorly aligned with tracking optimization and evaluation metrics, such as intersection over union (IoU) and area under the success curve (AUC). Here, we introduce RELO, a REinforcement-learning-to-LOcalize method for visual object tracking that formulates target localization as a Markov decision process. Specifically, RELO replaces handcrafted spatial priors with a localization policy learned over spatial positions via reinforcement learning, with rewards combining frame-level IoU and sequence-level AUC. We additionally introduce layer-aligned temporal token propagation to improve semantic consistency across frames, with negligible computational overhead. Across multiple benchmarks, RELO achieves superior results, attaining 57.5% AUC on LaSOText without template updates. This confirms that reward-driven localization provides an effective alternative to prior-driven localization for visual object tracking.
>
---
#### [replaced 025] Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.08503](https://arxiv.org/pdf/2604.08503)**

> **作者:** Ying Shen; Jerry Xiong; Tianjiao Yu; Ismini Lourentzou
>
> **备注:** 15 pages, 6 figures, CVPR 2026
>
> **摘要:** Recent advances in generative video modeling, driven by large-scale datasets and powerful architectures, have yielded remarkable visual realism. However, emerging evidence suggests that simply scaling data and model size does not endow these systems with an understanding of the underlying physical laws that govern real-world dynamics. Existing approaches often fail to capture or enforce such physical consistency, resulting in unrealistic motion and dynamics. In his work, we investigate whether integrating the inference of latent physical properties directly into the video generation process can equip models with the ability to produce physically plausible videos. To this end, we propose Phantom, a Physics-Infused Video Generation model that jointly models the visual content and latent physical dynamics. Conditioned on observed video frames and inferred physical states, Phantom jointly predicts latent physical dynamics and generates future video frames. Phantom leverages a physics-aware video representation that serves as an abstract yet informaive embedding of the underlying physics, facilitating the joint prediction of physical dynamics alongside video content without requiring an explicit specification of a complex set of physical dynamics and properties. By integrating the inference of physical-aware video representation directly into the video generation process, Phantom produces video sequences that are both visually realistic and physically consistent. Quantitative and qualitative results on both standard video generation and physics-aware benchmarks demonstrate that Phantom not only outperforms existing methods in terms of adherence to physical dynamics but also delivers competitive perceptual fidelity.
>
---
#### [replaced 026] RoomPilot: Controllable Indoor Scene Synthesis via Multimodal Semantic Parsing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.11234](https://arxiv.org/pdf/2512.11234)**

> **作者:** Wentang Chen; Shougao Zhang; Yiman Zhang; Tianhao Zhou; Ruihui Li
>
> **备注:** 30 pages, 8 figures
>
> **摘要:** Generating controllable indoor scenes is fundamental to applications in game development, architectural visualization, and embodied AI. However, existing approaches either support a limited input modalities or rely on implicit generation processes that hinder precise control over scene structure and semantics. To address these limitations, we introduce RoomPilot, a unified framework for controllable indoor scene synthesis from multi-modal inputs, including textual descriptions and CAD floor plans. RoomPilot maps heterogeneous inputs into an Indoor Domain-Specific Language (IDSL), which serves as a structured and interpretable semantic representation for describing indoor scenes. Built upon IDSL, RoomPilot presents a hierarchical synthesis pipeline that progressively organizes scenes at the building, room, and object levels, promoting structural coherence and functional consistency across multi-room layouts. Moreover, RoomPilot constructs a curated asset dataset with rich semantic annotations to support high-quality scene synthesis, improving visual realism and appearance consistency. Extensive experiments demonstrate effective multi-modal understanding, fine-grained controllability in scene generation, and improved physical consistency and visual fidelity, marking a significant step toward controllable 3D indoor scene synthesis. Code and model will be available.
>
---
#### [replaced 027] EnsemHalDet: Robust VLM Hallucination Detection via Ensemble of Internal State Detectors
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态 hallucination 检测任务，旨在解决VLMs生成不准确或无依据内容的问题。通过集成多个内部表示，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.02784](https://arxiv.org/pdf/2604.02784)**

> **作者:** Ryuhei Miyazato; Shunsuke Kitada; Kei Harada
>
> **摘要:** Vision-Language Models (VLMs) excel at multimodal tasks, but they remain vulnerable to hallucinations that are factually incorrect or ungrounded in the input image. Recent work suggests that hallucination detection using internal representations is more efficient and accurate than approaches that rely solely on model outputs. However, existing internal-representation-based methods typically rely on a single representation or detector, limiting their ability to capture diverse hallucination signals. In this paper, we propose EnsemHalDet, an ensemble-based hallucination detection framework that leverages multiple internal representations of VLMs, including attention outputs and hidden states. EnsemHalDet trains independent detectors for each representation and combines them through ensemble learning. Experimental results across multiple VQA datasets and VLMs show that EnsemHalDet consistently outperforms prior methods and single-detector models in terms of AUC. These results demonstrate that ensembling diverse internal signals significantly improves robustness in multimodal hallucination detection.
>
---
#### [replaced 028] Needles in the Landscape: Semi-Supervised Pseudolabeling for Archaeological Site Discovery under Label Scarcity
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.16814](https://arxiv.org/pdf/2510.16814)**

> **作者:** Simon Jaxy; Anton Theys; Patrick Willett; W. Chris Carleton; Ralf Vandam; Pieter Libin
>
> **摘要:** Archaeological predictive modelling estimates where undiscovered sites are likely to occur by combining known locations with environmental and geospatial variables, presenting a positive-unlabeled (PU) learning challenge where confirmed sites are rare and most locations are unlabeled rather than truly negative. To overcome this, we propose asymmetric dual pseudolabeling (DPL), an end-to-end deep learning method that learns from sparse positives directly from multi-band geospatial imagery without hand-crafted feature engineering or assumptions about site absence, and evaluate on two prominent archaeological datasets. On the Sagalassos dataset, evaluated against an independent, held-out field survey, DPL outperforms the LAMAP baseline by 12% in F1 and 29% in Recall, while LAMAP maintains advantages in probability ranking. Standard supervised baselines fail catastrophically when negatives are uncertain; positive-only training collapses to predicting everywhere, es- tablishing empirical bounds. On the Cyprus dataset, a pure PU setting without confirmed negatives, SL inverts probability rankings while DPL recovers discrimination. DPL ensembles produce interpretable probability surfaces supporting survey planning, enabling effective site discovery from minimal labeled data.
>
---
#### [replaced 029] NGL: Natural Garment Language for Training-Free Sewing Pattern Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.20700](https://arxiv.org/pdf/2602.20700)**

> **作者:** Anna Badalyan; Pratheba Selvaraju; Giorgio Becherini; Omid Taheri; Victoria Fernandez Abrevaya; Michael Black
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** Estimating sewing patterns from images is a practical approach for creating high-quality 3D garments, but it remains challenging due to the scarcity of paired real-world image and sewing-pattern data. Existing methods address this limitation by training vision-language models (VLMs) to learn low-level sewing-pattern representations from synthetic garments sampled from parametric garment models. However, they often struggle to generalize to in-the-wild images, fail to capture real-world correlations between garment parts, and are restricted to single-layer outfits. In contrast, we observe that VLMs are effective at describing garments in natural language, but mapping these descriptions into valid sewing patterns remains difficult. To bridge this gap, we propose NGL (Natural Garment Language), a novel domain-specific language that represents garments in terms aligned with VLMs' natural descriptive abilities. Leveraging NGL, we introduce a fully training-free pipeline that queries large VLMs to extract structured garment specifications and deterministically converts them into valid sewing patterns. We evaluate our method on the Dress4D, CloSe and a newly collected dataset of 253 in-the-wild fashion images. Our approach achieves state-of-the-art performance on standard geometry metrics and is preferred in both human and GPT-based perceptual evaluations compared to existing baselines. Furthermore, NGL recovers multi-layer outfits whereas competing methods focus mostly on single-layer garments, highlighting its strong generalization to real-world images even with occluded parts. These results demonstrate that an efficient garment representation is critical for sewing pattern estimation with VLMs. Our code and data will be released for research use.
>
---
#### [replaced 030] Beyond Size and Growth: Rethinking Lung Cancer Screening with AI Based Nodule Detection and Diagnosis
- **分类: cs.CV; q-bio.NC**

- **链接: [https://arxiv.org/pdf/2512.00281](https://arxiv.org/pdf/2512.00281)**

> **作者:** Sylvain Bodard; Pierre Baudot; Benjamin Renoust; Charles Voyton; Gwendoline De Bie; Ezequiel Geremia; Van-Khoa Le; Danny Francis; Pierre-Henri Siot; Yousra Haddou; Vincent Bobin; Jean-Christophe Brisset; Carey C. Thomson; Valerie Bourdes; Benoit Huet
>
> **备注:** 25 pages, 8 figures, with supplementary information containing 11 figures
>
> **摘要:** Early detection of malignant lung nodules remains constrained by size and growth based screening criteria, often delaying diagnosis. We present an integrated AI system that jointly performs nodule detection and malignancy assessment directly at the nodule level from low dose CT scans, within a unified CADe/CADx framework. Unlike conventional pipelines separating detection and diagnosis, our approach targets malignant nodules directly, redefining evaluation at the point where clinical decisions are made. To address limitations in dataset scale and explainability, the system consists of a Large Ensemble Model (LEM) combining ensembles of shallow deep learning and feature based models. It was trained and evaluated on 25,709 scans with 69,449 annotated nodules, with external validation on an independent cohort. It achieved an AUC of 0.98 internally and 0.945 externally, outperforming all growth based metrics, Lung RADS size based triage, European volume and VDT based screening criteria, radiologists, and leading AI models. The model maintains high sensitivity at low false positive rates, excels for small and early stage cancers, and enables malignancy assessment up to one year earlier than radiologists for indeterminate and slow growing nodules. This approach has the potential to streamline lung cancer screening workflows and support earlier, more actionable clinical decision making.
>
---
#### [replaced 031] Scene-Action Prompt Fusion for Coherent Text-to-Video Storytelling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.06310](https://arxiv.org/pdf/2503.06310)**

> **作者:** Taewon Kang; Divya Kothandaraman; Ming C. Lin
>
> **备注:** Accepted to the 2026 IEEE International Conference on Image Processing (ICIP 2026). 13 pages, 4 figures
>
> **摘要:** Generating coherent long-form video sequences from discrete text prompts remains challenging due to difficulties in maintaining temporal coherence, semantic consistency, and scene-action continuity across segments. We propose a novel storytelling framework that integrates scene and action prompts through dynamics-inspired prompt mixing. Our approach combines three key components: (i) a bidirectional time-weighted latent blending strategy that enforces temporal consistency between consecutive video segments, (ii) a dynamics-informed prompt weighting (DIPW) mechanism that adaptively balances scene and action prompts at each diffusion timestep based on CLIP-based alignment, narrative progression, and temporal smoothness, and (iii) a semantic action representation that encodes high-level action semantics to modulate transitions according to action similarity. Latent-space blending preserves spatial coherence within scenes, while time-weighted blending introduces bidirectional temporal constraints to prevent abrupt transitions. Together, these components enable fluid and coherent video narratives that faithfully reflect both scene context and action dynamics. Extensive experiments demonstrate that our method significantly outperforms baselines, producing temporally consistent and visually compelling long-form videos without any additional training, thereby bridging the gap between short clips and extended text-driven video storytelling.
>
---
#### [replaced 032] Mitigating Mask Prior Drift and Positional Attention Collapse in Large Diffusion Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.14530](https://arxiv.org/pdf/2605.14530)**

> **作者:** Sujung Hong; Chanyong Yoon; Seong Jae Hwang
>
> **摘要:** Large diffusion vision-language models (LDVLMs) have recently emerged as a promising alternative to autoregressive models, enabling parallel decoding for efficient inference and leveraging bidirectional attention for global context. Despite these advances, their behavior under long-form generation remains underexplored. In this work, we show that existing LDVLMs suffer from repetitive generation and degraded visual grounding, and identify two underlying causes. First, repetitive generation originates from a mask token prior: since generation tokens are initialized as mask tokens, their hidden representations progressively drift toward a shared prior direction over generation steps. Second, a fundamental misalignment between the positional attention bias and the iterative unmasking process suppresses attention toward informative visual tokens, degrading visual grounding. Based on these insights, we propose a training-free approach, introducing Mask Prior Suppression and Monotonic RoPE Scaling to mitigate mask prior drift and positional attention collapse during decoding. Experiments on general multimodal benchmarks and visual grounding tasks demonstrate improvements over baseline LDVLMs, with robust gains on long-form description benchmarks. Our results show that these failures can be effectively addressed with a lightweight, plug-and-play strategy that requires no additional training and generalizes across diverse LDVLM architectures.
>
---
#### [replaced 033] Spark3R: Asymmetric Token Reduction Makes Fast Feed-Forward 3D Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.06270](https://arxiv.org/pdf/2605.06270)**

> **作者:** Zecheng Tang; Jiaye Fu; Qiankun Gao; Haijie Li; Yanmin Wu; Jiaqi Zhang; Siwei Ma; Jian Zhang
>
> **摘要:** Feed-forward 3D reconstruction models based on Vision Transformers can directly estimate scene geometry and camera poses from a small set of input images, but scaling them to video inputs with hundreds or thousands of frames remains challenging due to the quadratic cost of global attention layers. Recent token-merging methods accelerate these models by compressing the token sequence within the global attention layers, but they apply a uniform reduction to query tokens and key-value tokens, ignoring their functionally distinct roles in 3D reconstruction. In this work, we identify a key property of feed-forward 3D reconstruction models: query tokens encode view-specific geometric requests and are sensitive to compression, while key-value tokens represent shared scene context and tolerate aggressive compression. Guided by this insight, we propose Spark3R, a training-free acceleration framework that decouples the compression of query tokens and key-value tokens by assigning distinct reduction factors, with intra-group token merging applied to query tokens and lightweight token pruning to key-value tokens. Additionally, Spark3R adaptively adjusts the key-value reduction factor across layers, further improving the quality-efficiency trade-off. As a plug-and-play framework requiring no retraining, Spark3R integrates directly into multiple pretrained feed-forward 3D reconstruction models, including VGGT, $\pi^3$, Depth-Anything-3, and VGGT-$\Omega$, and achieves up to $28\times$ speedup on 1,000-frame inputs while maintaining competitive reconstruction quality.
>
---
#### [replaced 034] Class Unlearning via Depth-Aware Removal of Forget-Specific Directions
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.15166](https://arxiv.org/pdf/2604.15166)**

> **作者:** Arman Hatami; Romina Aalishah; Ilya E. Monosov
>
> **备注:** Accepted for oral presentation at the CVPR 2026 Workshop on Machine Unlearning for Vision (MUV). Code: this https URL
>
> **摘要:** Machine unlearning aims to remove targeted knowledge from a trained model without the cost of retraining from scratch. In class unlearning, however, reducing accuracy on forget classes does not necessarily imply true forgetting: forgotten information can remain encoded in internal representations, and apparent forgetting may arise from classifier-head suppression rather than representational removal. We show that existing class-unlearning methods often exhibit weak or negative selectivity, preserve forget-class structure in deep representations, or rely heavily on final-layer bias shifts. We then introduce DAMP (Depth-Aware Modulation by Projection), a one-shot, closed-form weight-surgery method that removes forget-specific directions from a pretrained network without gradient-based optimization. At each stage, DAMP computes class prototypes in the input space of the next learnable operator, extracts forget directions as residuals relative to retain-class prototypes, and applies a projection-based update to reduce downstream sensitivity to those directions. To preserve utility, DAMP uses a parameter-free depth-aware scaling rule derived from probe separability, applying smaller edits in early layers and larger edits in deeper layers. The method naturally extends to multi-class forgetting through low-rank subspace removal. Across MNIST, CIFAR-10, CIFAR-100, and Tiny ImageNet, and across convolutional and transformer architectures, DAMP more closely resembles the retraining gold standard than some of the prior methods, improving selective forgetting while better preserving retain-class performance and reducing residual forget-class structure in deep layers.
>
---
#### [replaced 035] LongLive-2.0: An NVFP4 Parallel Infrastructure for Long Video Generation
- **分类: cs.CV; cs.DC**

- **链接: [https://arxiv.org/pdf/2605.18739](https://arxiv.org/pdf/2605.18739)**

> **作者:** Yukang Chen; Luozhou Wang; Wei Huang; Shuai Yang; Bohan Zhang; Yicheng Xiao; Ruihang Chu; Weian Mao; Qixin Hu; Shaoteng Liu; Yuyang Zhao; Huizi Mao; Ying-Cong Chen; Enze Xie; Xiaojuan Qi; Song Han
>
> **备注:** Code, model, and demos are available at this https URL
>
> **摘要:** We present LongLive-2.0, an NVFP4-based parallel infrastructure throughout the full training and inference workflow of long video generation, addressing speed and memory bottlenecks. For training, we introduce sequence-parallel autoregressive (AR) training, instantiated as Balanced SP, which co-designs the efficient teacher-forcing layout with SP execution by pairing clean-history and noisy-target temporal chunks on each rank, enabling a natural teacher-forcing mask with SP-aware chunked VAE encoding. Combined with NVFP4 precision, it reduces GPU memory cost and accelerates GEMM computation during training, the proportion of which increases as video length grows. Moreover, we show that a high-quality infrastructure and dataset enable a remarkably clean training pipeline. Unlike existing Self-Forcing series methods that rely on ODE initialization and subsequent distribution matching distillation (DMD), LongLive-2.0 directly tunes a diffusion model into a long, multi-shot, interactive auto-regressive (AR) diffusion model. It can be further converted to real-time generation (4 to 2 denoising steps) with standalone LoRA weights. For inference on Blackwell GPUs, we enable W4A4 NVFP4 inference, quantize KV cache into NVFP4 for memory savings, and boost end-to-end throughput with asynchronous streaming VAE decoding. On non-Blackwell GPU architectures, we deploy SP inference to match the speed on Blackwell GPUs, while the quantized KV cache can lower inter-GPU communication of SP. Experiments show up to 2.15x speedup in training, and 1.84x in inference. LongLive-2.0-5B achieves 45.7 FPS inference while attaining strong performance on benchmarks. To our knowledge, LongLive-2.0 is the first NVFP4 training and inference system for long video generation.
>
---
#### [replaced 036] Flow-OPD: On-Policy Distillation for Flow Matching Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.08063](https://arxiv.org/pdf/2605.08063)**

> **作者:** Zhen Fang; Wenxuan Huang; Yu Zeng; Yiming Zhao; Shuang Chen; Kaituo Feng; Yunlong Lin; Lin Chen; Zehui Chen; Shaosheng Cao; Feng Zhao
>
> **备注:** Project Page: this https URL , Code: this https URL
>
> **摘要:** Existing Flow Matching (FM) text-to-image models suffer from two critical bottlenecks under multi-task alignment: the reward sparsity induced by scalar-valued rewards, and the gradient interference arising from jointly optimizing heterogeneous objectives, which together give rise to a 'seesaw effect' of competing metrics and pervasive reward hacking. Inspired by the success of On-Policy Distillation (OPD) in the large language model community, we propose Flow-OPD, the first unified post-training framework that integrates on-policy distillation into Flow Matching models. Flow-OPD adopts a two-stage alignment strategy: it first cultivates domain-specialized teacher models via single-reward GRPO fine-tuning, allowing each expert to reach its performance ceiling in isolation; it then establishes a robust initial policy through a Flow-based Cold-Start scheme and seamlessly consolidates heterogeneous expertise into a single student via a three-step orchestration of on-policy sampling, task-routing labeling, and dense trajectory-level supervision. We further introduce Manifold Anchor Regularization (MAR), which leverages a task-agnostic teacher to provide full-data supervision that anchors generation to a high-quality manifold, effectively mitigating the aesthetic degradation commonly observed in purely RL-driven alignment. Built upon Stable Diffusion 3.5 Medium, Flow-OPD raises the GenEval score from 63 to 92 and the OCR accuracy from 59 to 94, yielding an overall improvement of roughly 10 points over vanilla GRPO, while preserving image fidelity and human-preference alignment and exhibiting an emergent 'teacher-surpassing' effect. These results establish Flow-OPD as a scalable alignment paradigm for building generalist text-to-image models. The codes and weights will be released in: this https URL .
>
---
#### [replaced 037] ORCA: An Agentic Reasoning Framework for Hallucination and Adversarial Robustness in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.MA**

- **链接: [https://arxiv.org/pdf/2509.15435](https://arxiv.org/pdf/2509.15435)**

> **作者:** Chung-En Johnny Yu; Brian Jalaian; Nathaniel D. Bastian
>
> **备注:** Accepted at the ACM International Conference on Cloud and Big Data Computing (ICCBDC 2026)
>
> **摘要:** Large Vision-Language Models (LVLMs) exhibit strong multimodal capabilities but remain vulnerable to hallucinations from intrinsic errors and adversarial attacks from external exploitations, limiting their reliability in real-world applications. We present ORCA, an agentic reasoning framework that improves the factual accuracy and adversarial robustness of pretrained LVLMs through inference-time structured inference reasoning with a suite of small vision models (less than 3B parameters). ORCA operates via an Observe-Reason-Critique-Act loop, querying multiple visual tools with evidential questions, validating cross-model inconsistencies, and refining predictions iteratively without access to model internals or retraining. ORCA also stores intermediate reasoning traces, which supports auditable decision-making. Though designed primarily to mitigate object-level hallucinations, ORCA also exhibits emergent adversarial robustness without requiring adversarial training or defense mechanisms. We evaluate ORCA across three settings: (1) clean images on hallucination benchmarks, (2) adversarially perturbed images without defense, and (3) adversarially perturbed images with defense applied. On the POPE hallucination benchmark, ORCA improves standalone LVLMs performance by +3.64% to +40.67% across different subsets. Under adversarial perturbations on POPE, ORCA achieves an average accuracy gain of +20.11% across LVLMs. When combined with defense techniques on adversarially perturbed AMBER images, ORCA further improves standalone LVLM performance, with gains ranging from +1.20% to +48.00% across metrics. These results demonstrate that ORCA offers a promising path toward building more reliable and robust multimodal systems.
>
---
#### [replaced 038] GemDepth: Geometry-Embedded Features for 3D-Consistent Video Depth
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.10525](https://arxiv.org/pdf/2605.10525)**

> **作者:** Yuecheng Liu; Junda Cheng; Longliang Liu; Wenjing Liao; Hanrui Cheng; Yuzhou Wang; Xin Yang
>
> **摘要:** Video depth estimation extends monocular prediction into the temporal domain to ensure coherence. However, existing methods often suffer from spatial blurring in fine-detail regions and temporal inconsistencies. We argue that current approaches, which primarily rely on temporal smoothing via Transformers, struggle to maintain strict 3D geometric consistency-particularly under rotations or drastic view changes. To address this, we propose GemDepth, a framework built on the insight that an explicit awareness of camera motion and global 3D structure is a prerequisite for 3D consistency. Distinctively, GemDepth introduces a Geometry-Embedding Module (GEM) that predicts inter-frame camera poses to generate implicit geometric embeddings. This injection of motion priors equips the network with intrinsic 3D perception and alignment capabilities. Guided by these geometric cues, our Alternating Spatio-Temporal Transformer (ASTT) captures latent point-level correspondences to simultaneously enhance spatial precision for sharp details and enforce rigorous temporal consistency. Furthermore, GemDepth employs a data-efficient training strategy, effectively bridging the gap between high efficiency and robust geometric consistency. As shown in Fig.2, comprehensive evaluations demonstrate that GemDepth achieves state-of-the-art performance across multiple datasets, particularly in complex dynamic scenarios. The code is publicly available at: this https URL.
>
---
#### [replaced 039] Low-Compute Watermark Removal via Dual-Domain Natural Projection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.07538](https://arxiv.org/pdf/2510.07538)**

> **作者:** Pragati Shuddhodhan Meshram; Varun Chandrasekaran
>
> **摘要:** Effective removal of semantic watermarks requires balancing three competing objectives: \emph{high removal success}, \emph{low perceptual distortion}, and \emph{low computational cost}. However, existing single-image attacks typically optimize only for the first two, achieving strong watermark suppression but relying on expensive, multi-step optimization that limits practical deployment. In this work, we show that this trade-off is fundamental: no current approach achieves all three properties simultaneously. We introduce \textsc{DAWN}, a lightweight, training-free attack that explicitly targets the low-cost regime while maintaining competitive removal performance. \textsc{DAWN} works by projecting a watermarked image onto natural-image priors in complementary frequency and semantic spaces, suppressing watermark signals that deviate from natural statistics, and then applying a decoupled perceptual-alignment step to restore visual consistency with minimal artifact. Across diverse pixel-, frequency-, and latent-space watermarking schemes, \textsc{DAWN} consistently reduces detectability while preserving structural and semantic fidelity, demonstrating that efficient, low-resource watermark removal is feasible with only modest perceptual degradation. Our code is available at this https URL.
>
---
#### [replaced 040] CAB: Accelerating Flow and Diffusion Sampling via Rectification and Corrected Adams-Bashforth
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.16736](https://arxiv.org/pdf/2605.16736)**

> **作者:** Anuska Roy; Pravin Nair
>
> **摘要:** Flow and diffusion models achieve high-fidelity, high-resolution image synthesis, but often require many function evaluations (NFEs) at sampling time. Existing acceleration methods either require additional training through distillation or rely on training-free high-order solvers, and both can degrade sample quality at low NFE budgets. We propose CAB (Corrected Adams-Bashforth), a training-free sampler that accelerates both flow and diffusion models. CAB first transforms the sampling dynamics to a common rectified coordinate system, and then applies a multistep Adams-Bashforth predictor augmented with a simple correction term based on past velocity evaluations and therefore incurs no additional NFEs. The resulting method is simple, has the same algorithmic form across model classes, and has at least third-order local truncation error and second-order global error. Experiments on pretrained flow and diffusion models, including class-conditional and large-scale text-to-image benchmarks, show that CAB improves quality-NFE trade-offs in the low-step regime of 6-20 NFEs. It also remains competitive with strong training-free samplers at higher step counts across most tested models. The official implementation is available at this https URL.
>
---
#### [replaced 041] Unsupervised Unfolded rPCA (U2-rPCA): Deep Interpretable Clutter Filtering for Ultrasound Microvascular Imaging
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.00660](https://arxiv.org/pdf/2510.00660)**

> **作者:** Huaying Li; Chuling Ye; Manfei Liao; Xiaobo Qu; Liansheng Wang; Yinran Chen
>
> **摘要:** High-sensitivity clutter filtering is a fundamental step in ultrasound microvascular imaging. Singular value decomposition (SVD) and robust principal component analysis (rPCA) are the main clutter filtering strategies. However, both strategies are limited in feature modeling and separation of tissue and blood flow for high-quality microvascular imaging. Recently, deep learning-based clutter filtering has shown potential in more thoroughly separating tissue and blood flow signals. However, the existing supervised filters face the lack of interpretability and the training ground truth. While the interpretability issue can be addressed by algorithm deep unfolding, the training ground truth remains unsolved. This paper proposes an unsupervised unfolded rPCA (U2-rPCA) method that preserves mathematical interpretability and is insusceptible to learning labels. Specifically, U2-rPCA is unfolded from an iteratively reweighted least squares (IRLS) rPCA baseline with intrinsic low-rank and sparse regularization. In addition, a sparse-enhancement unit is plugged into the network to strengthen its capability to capture the sparse micro-flow signals. U2-rPCA is like an adaptive filter that is trained with part of the image sequence and then used for the following frames. Experimental validations on a in-silico dataset and public in-vivo datasets demonstrated the outperformance of U2-rPCA when compared with the SVD filter, the rPCA baseline, and another deep learning-based filter. Particularly, the proposed method improved the contrast-to-noise ratio (CNR) of the power Doppler image by 1.91 dB to 8.48 dB compared to other methods. Furthermore, the effectiveness of the building modules of U2-rPCA was validated through ablation studies.
>
---
#### [replaced 042] Contextualized Visual Personalization in Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.03454](https://arxiv.org/pdf/2602.03454)**

> **作者:** Yeongtak Oh; Sangwon Yu; Junsung Park; Han Cheol Moon; Jisoo Mok; Sungroh Yoon
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Despite recent progress in vision-language models (VLMs), existing approaches often fail to generate personalized responses based on the user's specific experiences, as they lack the ability to associate visual inputs with a user's accumulated visual-textual context. We newly formalize this challenge as contextualized visual personalization, which requires the visual recognition and textual retrieval of personalized visual experiences by VLMs when interpreting new images. To address this issue, we propose CoViP, a unified framework that treats personalized image captioning as a core task for contextualized visual personalization and improves this capability through reinforcement-learning-based post-training and caption-augmented generation. We further introduce diagnostic evaluations that explicitly rule out textual shortcut solutions and verify whether VLMs truly leverage visual context. Extensive experiments demonstrate that existing open-source and proprietary VLMs exhibit substantial limitations, while CoViP not only improves personalized image captioning but also yields holistic gains across downstream personalization tasks. These results highlight CoViP as a crucial stage for enabling robust and generalizable contextualized visual personalization.
>
---
#### [replaced 043] Locate-then-Sparsify: Attribution Guided Sparse Strategy for Visual Hallucination Mitigation
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.16284](https://arxiv.org/pdf/2603.16284)**

> **作者:** Tiantian Dang; Chao Bi; Shufan Shen; Jinzhe Liu; Qingming Huang; Shuhui Wang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Despite the significant advancements in Large Vision-Language Models (LVLMs), their tendency to generate hallucinations undermines reliability and restricts broader practical deployment. Among the hallucination mitigation methods, feature steering emerges as a promising approach that reduces erroneous outputs in LVLMs without increasing inference costs. However, current methods apply uniform feature steering across all layers. This heuristic strategy ignores inter-layer differences, potentially disrupting layers unrelated to hallucinations and ultimately leading to performance degradation on general tasks. In this paper, we propose Locate-Then-Sparsify for Feature Steering (LTS-FS), a plug-and-play framework which controls the steering intensity according to the hallucination relevance of each layer. We first construct a dataset comprising token-level and sentence-level hallucination cases. Based on this dataset, we introduce an attribution method based on causal interventions to quantify the hallucination relevance of each layer. With the attribution scores across layers, we propose a layerwise strategy that converts these scores into feature steering intensities for individual layers, enabling more precise adjustments specifically on hallucination-relevant layers. Extensive experiments across multiple LVLMs and benchmarks demonstrate that LTS-FS effectively mitigates hallucination while preserving strong performance. Codes are available at this https URL.
>
---
#### [replaced 044] Motion-2-To-3: Leveraging 2D Motion Data for 3D Motion Generations
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2412.13111](https://arxiv.org/pdf/2412.13111)**

> **作者:** Ruoxi Guo; Huaijin Pi; Zehong Shen; Qing Shuai; Zechen Hu; Zhumei Wang; Yajiao Dong; Ruizhen Hu; Taku Komura; Sida Peng; Xiaowei Zhou
>
> **备注:** Project page: this https URL
>
> **摘要:** Text-driven human motion synthesis has showcased its potential for revolutionizing motion design in the movie and game industry. Existing methods often rely on 3D motion capture data, which requires special setups, resulting in high costs for data acquisition, ultimately limiting the diversity and scope of human motion. In contrast, 2D human videos offer a vast and accessible source of motion data, covering a wider range of styles and activities. In this paper, we explore the use of 2D human motion extracted from videos as an alternative data source to improve text-driven 3D motion generation. Our approach introduces a novel framework that disentangles local joint motion from global movements, enabling efficient learning of local motion priors from 2D data. We first train a single-view 2D local motion generator on a large dataset of text-2D motion pairs. Then we fine-tune the generator with 3D data, transforming it into a multi-view generator that predicts view-consistent local joint motion and root dynamics. Evaluations on the well-acknowledged dataset and novel text prompts demonstrate that our method can efficiently utilize 2D data, supporting a wider range of realistic 3D human motion generation. Our code is publicly available at this https URL.
>
---
#### [replaced 045] What's Holding Back Latent Visual Reasoning?
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉推理任务，探讨为何模型无法有效利用潜在视觉中间步骤。研究发现，现有数据集中的潜在标记信息不足，且生成的标记与真实标记偏差大，导致模型无法依赖它们。**

- **链接: [https://arxiv.org/pdf/2605.18445](https://arxiv.org/pdf/2605.18445)**

> **作者:** André G. Viveiros; Nuno Gonçalves; André F. T. Martins; Matthias Lindemann
>
> **摘要:** Humans can approach complex visual problems by mentally simulating intermediate visual steps, rather than reasoning through language alone. Inspired by this, several works on Vision-Language Models have recently explored chain-of-thought reasoning with continuous latent tokens as intermediate visual imagination steps. In this work, we investigate how recent models leverage such latent tokens. Surprisingly, we find that model accuracy is unaffected when latent tokens are replaced by uninformative dummy tokens. This indicates that latent tokens play a minimal causal role in the model's final prediction. To better understand this phenomenon, we analyze both the training signal provided by oracle latent representations and the quality of the latent tokens generated at inference time. Our experiments reveal two crucial issues holding back latent visual reasoning: First, in most existing datasets, oracle latent tokens provide limited additional information beyond the original image and do not substantially simplify the task, leading models to ignore them during training and effectively bypassing them at inference time. When fine-tuned on a diagnostic dataset, in which latent tokens provide sufficient support for the final prediction, we show that models can causally rely on them. Second, the latent tokens produced at inference time deviate from their corresponding oracle representations, collapsing to a narrow region and preventing benefits even when the model relies on them. Overall, our findings suggest that future progress in latent visual reasoning depends on two key pillars: high-quality datasets with informative intermediate steps and more precise latent token prediction.
>
---
#### [replaced 046] Where Not to Learn: Prior-Aligned Training with Subset-based Attribution Constraints for Reliable Decision-Making
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.07008](https://arxiv.org/pdf/2602.07008)**

> **作者:** Ruoyu Chen; Shangquan Sun; Xiaoqing Guo; Sanyi Zhang; Kangwei Liu; Shiming Liu; Zhangcheng Wang; Qunli Zhang; Hua Zhang; Xiaochun Cao
>
> **摘要:** Reliable models should not only predict correctly, but also justify decisions with acceptable evidence. Yet conventional supervised learning typically provides only class-level labels, allowing models to achieve high accuracy through shortcut correlations rather than the intended evidence. Human priors can help constrain such behavior, but aligning models to these priors remains challenging because learned representations often diverge from human perception. To address this challenge, we propose an attribution-based human prior alignment method. We encode human priors as input regions that the model is expected to rely on (e.g., bounding boxes), and leverage a highly faithful subset-selection-based attribution approach to expose the model's decision evidence during training. When the attribution region deviates substantially from the prior regions, we penalize reliance on off-prior evidence, encouraging the model to shift its attribution toward the intended regions. This is achieved through a training objective that imposes attribution constraints induced by the human prior. We validate our method on both image classification and click decision tasks in MLLM-based GUI agent models. Across conventional classification and autoregressive generation settings, human prior alignment consistently improves task accuracy while also enhancing the model's decision reasonability.
>
---
#### [replaced 047] TextBoost: Boosting Text Encoder for Personalized Text-to-Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2409.08248](https://arxiv.org/pdf/2409.08248)**

> **作者:** NaHyeon Park; Kunhee Kim; Hyunjung Shim
>
> **备注:** Project page: this https URL. Accepted to TMLR
>
> **摘要:** In this paper, we introduce TextBoost, an efficient one-shot personalization approach for text-to-image diffusion models. Traditional personalization methods typically involve fine-tuning extensive portions of the model, leading to substantial storage requirements and slow convergence. In contrast, we propose selectively fine-tuning only the text encoder, significantly improving computational and storage efficiency. To preserve the original semantic integrity, we develop a novel causality-preserving adaptation mechanism. Additionally, lightweight adapters are employed to locally refine text embeddings immediately before their interaction with cross-attention layers, greatly enhancing the expressiveness of text embeddings with minimal computational overhead. Empirical evaluations across diverse concepts demonstrate that TextBoost achieves faster convergence and substantially reduces storage demands by minimizing the number of trainable parameters. Furthermore, TextBoost maintains comparable subject fidelity, superior text fidelity, and greater generation diversity compared to existing methods. We show that our proposed method offers an efficient, scalable, and practically applicable solution for high-quality text-to-image personalization, particularly beneficial in resource-constrained environments.
>
---
#### [replaced 048] Contrastive Learning under Noisy Temporal Self-Supervision for Colonoscopy Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.12320](https://arxiv.org/pdf/2605.12320)**

> **作者:** Luca Parolari; Pietro Gori; Lamberto Ballan; Carlo Biffi; Loic Le Folgoc
>
> **备注:** Accepted to MICCAI 2026
>
> **摘要:** Learning robust representations of polyp tracklets is key to enabling multiple AI-assisted colonoscopy applications, from polyp characterization to automated reporting and retrieval. Supervised contrastive learning is an effective approach for learning such representations, but it typically relies on correct positive and negative definitions. Collecting these labels requires linking tracklets that depict the same underlying polyp entity throughout the video, which is costly and demands specialized clinical expertise. In this work, we leverage the sequential workflow of colonoscopy procedures to derive self-supervised associations from temporal structure. Since temporally derived associations are not guaranteed to be correct, we introduce a noise-aware contrastive loss to account for noisy associations. We demonstrate the effectiveness of the learned representations across multiple downstream tasks, including polyp retrieval and re-identification, size estimation, and histology classification. Our method outperforms prior self-supervised and supervised baselines, and matches or exceeds recent foundation models across all tasks, using a lightweight encoder trained on only 27 videos. Code is available at this https URL.
>
---
#### [replaced 049] Slot-MLLM: Object-Centric Visual Tokenization for Multimodal LLM
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.17726](https://arxiv.org/pdf/2505.17726)**

> **作者:** Donghwan Chi; Hyomin Kim; Yoonjin Oh; Yongjin Kim; Donghoon Lee; Daejin Jo; Jongmin Kim; Junyeob Baek; Sungjin Ahn; Sungwoong Kim
>
> **摘要:** Recently, multimodal large language models (MLLMs) have emerged as a key approach in achieving artificial general intelligence. In particular, vision-language MLLMs have been developed to generate not only text but also visual outputs from multimodal inputs. This advancement requires efficient image tokens that LLMs can process effectively both in input and output. However, existing image tokenization methods for MLLMs typically capture only global abstract concepts or uniformly segmented image patches, restricting MLLMs' capability to effectively understand or generate detailed visual content, particularly at the object level. To address this limitation, we propose an object-centric visual tokenizer based on Slot Attention specifically for MLLMs. In particular, based on the Q-Former encoder, diffusion decoder, and residual vector quantization, our proposed discretized slot tokens can encode local visual details while maintaining high-level semantics, and also align with textual data to be integrated seamlessly within a unified next-token prediction framework of LLMs. The resulting Slot-MLLM demonstrates significant performance improvements over baselines with previous visual tokenizers across various vision-language tasks that entail local detailed comprehension and generation. Notably, this work is the first demonstration of the feasibility of object-centric slot attention performed with MLLMs and in-the-wild natural images.
>
---
#### [replaced 050] PEPL: Precision-Enhanced Pseudo-Labeling for Fine-Grained Image Classification in Semi-Supervised Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2409.03192](https://arxiv.org/pdf/2409.03192)**

> **作者:** Bowen Tian; Songning Lai; Lujundong Li; Zhihao Shuai; Runwei Guan; Tian Wu; Yutao Yue
>
> **备注:** Accepted by ICASSP 2025
>
> **摘要:** Fine-grained image classification has witnessed significant advancements with the advent of deep learning and computer vision technologies. However, the scarcity of detailed annotations remains a major challenge, especially in scenarios where obtaining high-quality labeled data is costly or time-consuming. To address this limitation, we introduce Precision-Enhanced Pseudo-Labeling(PEPL) approach specifically designed for fine-grained image classification within a semi-supervised learning framework. Our method leverages the abundance of unlabeled data by generating high-quality pseudo-labels that are progressively refined through two key phases: initial pseudo-label generation and semantic-mixed pseudo-label generation. These phases utilize Class Activation Maps (CAMs) to accurately estimate the semantic content and generate refined labels that capture the essential details necessary for fine-grained classification. By focusing on semantic-level information, our approach effectively addresses the limitations of standard data augmentation and image-mixing techniques in preserving critical fine-grained features. We achieve state-of-the-art performance on benchmark datasets, demonstrating significant improvements over existing semi-supervised strategies, with notable boosts in accuracy and robustness.
>
---
#### [replaced 051] CXR-LanIC: Language-Grounded Interpretable Classifier for Chest X-Ray Diagnosis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.21464](https://arxiv.org/pdf/2510.21464)**

> **作者:** Yiming Tang; Wenjia Zhong; Rushi Shah; Dianbo Liu
>
> **摘要:** Deep learning models have achieved remarkable accuracy in chest X-ray diagnosis, yet their widespread clinical adoption remains limited by the black-box nature of their predictions. Clinicians require transparent, verifiable explanations to trust automated diagnoses and identify potential failure modes. We introduce CXR-LanIC (Language-Grounded Interpretable Classifier for Chest X-rays), a novel framework that addresses this interpretability challenge through task-aligned pattern discovery. Our approach trains transcoder-based sparse autoencoders on a BiomedCLIP diagnostic classifier to decompose medical image representations into interpretable visual patterns. By training an ensemble of 100 transcoders on multimodal embeddings from the MIMIC-CXR dataset, we discover approximately 5,000 monosemantic patterns spanning cardiac, pulmonary, pleural, structural, device, and artifact categories. Each pattern exhibits consistent activation behavior across images sharing specific radiological features, enabling transparent attribution where predictions decompose into 20-50 interpretable patterns with verifiable activation galleries. CXR-LanIC achieves competitive diagnostic accuracy on five key findings while providing the foundation for natural language explanations through planned large multimodal model annotation. Our key innovation lies in extracting interpretable features from a classifier trained on specific diagnostic objectives rather than general-purpose embeddings, ensuring discovered patterns are directly relevant to clinical decision-making, demonstrating that medical AI systems can be both accurate and interpretable, supporting safer clinical deployment through transparent, clinically grounded explanations.
>
---
#### [replaced 052] Motif-Video 2B: Technical Report
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.16503](https://arxiv.org/pdf/2604.16503)**

> **作者:** Junghwan Lim; Wai Ting Cheung; Minsu Ha; Beomgyu Kim; Taewhan Kim; Haesol Lee; Dongpin Oh; Jeesoo Lee; Taehyun Kim; Minjae Kim; Sungmin Lee; Hyeyeon Cho; Dahye Choi; Jaeheui Her; Jaeyeon Huh; Hanbin Jung; Changjin Kang; Dongseok Kim; Jangwoong Kim; Youngrok Kim; Hyukjin Kweon; Hongjoo Lee; Jeongdoo Lee; Junhyeok Lee; Eunhwan Park; Yeongjae Park; Bokki Ryu; Dongjoo Weon
>
> **摘要:** Training strong video generation models usually requires massive datasets, large parameter counts, and substantial compute. In this work, we ask whether strong text-to-video quality is possible at a much smaller budget: fewer than 10M clips and less than 100,000 H200 GPU hours. Our core claim is that part of the answer lies in how model capacity is organized, not only in how much of it is used. In video generation, prompt alignment, temporal consistency, and fine-detail recovery can interfere with one another when they are handled through the same pathway. Motif-Video 2B addresses this by separating these roles architecturally, rather than relying on scale alone. The model combines two key ideas. First, Shared Cross-Attention strengthens text control when video token sequences become long. Second, a three-part backbone separates early fusion, joint representation learning, and detail refinement. To make this design effective under a limited compute budget, we pair it with an efficient training recipe based on dynamic token routing and early-phase feature alignment to a frozen pretrained video encoder. Our analysis shows that later blocks develop clearer cross-frame attention structure than standard single-stream baselines. On VBench, Motif-Video~2B reaches 83.76\%, surpassing Wan2.1 14B while using 7$\times$ fewer parameters and substantially less training data. These results suggest that careful architectural specialization, combined with an efficiency-oriented training recipe, can narrow or exceed the quality gap typically associated with much larger video models.
>
---
#### [replaced 053] Enabling Real-Time Colonoscopic Polyp Segmentation on Commodity CPUs via Ultra-Lightweight Architecture
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.04381](https://arxiv.org/pdf/2602.04381)**

> **作者:** Weihao Gao; Zhuo Deng; Zheng Gong; Lan Ma
>
> **备注:** 18pages, 4 figures
>
> **摘要:** Real-time polyp segmentation is essential for early colorectal cancer detection, yet clinical deployment remains blocked by GPU dependency. We introduce the UltraSeg family, a set of CPU-native segmentation models operating below 0.3M parameters. UltraSeg-108K (0.108M) establishes the extreme-compression frontier, while UltraSeg-130K (0.130M) integrates cross-layer lightweight fusion for enhanced multi-center generalization. The architecture replaces parameter-heavy components with grouped multi-rate dilated convolutions and attention-gated cross-layer fusion, achieving real-time throughput on a single CPU core (exceeding 50 FPS at 256*256 and 30 FPS at 352*352) without sacrificing clinical-grade accuracy. Evaluated on seven public datasets, UltraSeg-130K attains Dice scores exceeding 0.8 at both resolutions, substantially outperforming all existing sub-0.3M competitors. Notably, it approaches or exceeds UNet-Medium (7.76M parameters) on zero-shot external validations while using only 1.7% of its parameters, establishing the first strong baseline for CPU-native real-time polyp segmentation. When scaled to 4.38M parameters, UltraSeg achieves accuracy competitive with heavyweight state-of-the-art models while maintaining an order-of-magnitude parameter advantage, demonstrating that the proposed design principles yield intrinsic representational gains across the entire efficiency spectrum. By delivering the first clinically deployable, CPU-native real-time solution, this work provides an immediately usable tool for resource-limited settings and a reproducible blueprint for real-time medical AI beyond endoscopy. Source code is publicly available.
>
---
#### [replaced 054] MVI-Bench: A Comprehensive Benchmark for Evaluating Robustness to Misleading Visual Inputs in LVLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14159](https://arxiv.org/pdf/2511.14159)**

> **作者:** Huiyi Chen; Jiawei Peng; Dehai Min; Changchang Sun; Kaijie Chen; Yan Yan; Xu Yang; Lu Cheng
>
> **备注:** 18 pages, 9 figures
>
> **摘要:** Evaluating the robustness of Large Vision-Language Models (LVLMs) is essential for their continued development and responsible deployment in real-world applications. However, existing robustness benchmarks typically focus on hallucination or misleading textual inputs, while largely overlooking the equally critical challenge posed by misleading visual inputs in assessing visual understanding. To fill this important gap, we introduce MVI-Bench, the first comprehensive benchmark specially designed for evaluating how Misleading Visual Inputs undermine the robustness of LVLMs. Grounded in fundamental visual primitives, the design of MVI-Bench centers on three hierarchical levels of misleading visual inputs: Visual Concept, Visual Attribute, and Visual Relationship. Using this taxonomy, we curate six representative categories and compile 1,248 expertly annotated VQA instances. To facilitate fine-grained robustness evaluation, we further introduce MVI-Sensitivity, a novel metric that characterizes LVLM robustness at a granular level. Empirical results across 18 state-of-the-art LVLMs uncover pronounced vulnerabilities to misleading visual inputs, and our in-depth analyses on MVI-Bench provide actionable insights that can guide the development of more reliable and robust LVLMs. The benchmark and codebase can be accessed at this https URL.
>
---
#### [replaced 055] Feature-Space Smoothing: Certified Robustness of Deep Representations
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2601.16200](https://arxiv.org/pdf/2601.16200)**

> **作者:** Song Xia; Meiwen Ding; Chenqi Kong; Wenhan Yang; Xudong Jiang
>
> **备注:** Under review
>
> **摘要:** Modern deep learning models exhibit strong capabilities across diverse applications, yet remain vulnerable to malicious inputs that induce erroneous predictions via feature-space distortion. To address this vulnerability, we propose Feature-space Smoothing (FS), a general defense framework that provides certified robustness at the feature representation level. We show that FS converts a given feature encoder into a smoothed variant that is guaranteed to maintain a certified lower bound on the cosine similarity between clean and adversarial features under l_2-bounded perturbations. We then establish that this Feature Cosine Similarity Bound (FCSB) can be extended to the prediction-wise certification under the cosine similarity measure, and the value of FCSB is determined by the encoder intrinsic Gaussian robustness score. Building on those insights, we introduce the Gaussian Smoothness Booster (GSB), a plug-and-play module to improve the encoder Gaussian robustness score. Specifically, the GSB module is plugged to enhance the feature-space consistency and maintain the feature utility for downstream tasks under Gaussian perturbations. This design enables seamless integration of FS on the protected model, e.g., Multimodal Large Language Models (MLLMs), without additional model retraining or alignment, improving its robustness while preserving the performance for downstream task-oriented decoding. Extensive experiments demonstrate that integrating FS consistently provides non-trivial certified robustness and significantly improves task-oriented performance under strong white-box adversarial attacks across diverse models and applications.
>
---
#### [replaced 056] Does AI See like Art Historians? Interpreting How Vision Language Models Recognize Artistic Style
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.11024](https://arxiv.org/pdf/2603.11024)**

> **作者:** Marvin Limpijankit; Milad Alshomary; Yassin Oulad Daoud; Amith Ananthram; Tim Trombley; Emily L. Spratt; Anna Filonenko; Hannah Pivo; Elias Stengel-Eskin; Mohit Bansal; Noam M. Elcott; Kathleen McKeown
>
> **备注:** 20 pages, 18 figures
>
> **摘要:** VLMs have become increasingly proficient at a range of computer vision tasks, such as visual question answering and object detection. This includes increasingly strong capabilities in the domain of art, from analyzing artwork to generation of art. In an interdisciplinary collaboration between computer scientists and art historians, we characterize the mechanisms underlying VLMs' ability to predict artistic style and assess the extent to which they align with the criteria art historians use to reason about artistic style. We employ a latent-space decomposition approach to identify concepts that drive art style prediction and conduct quantitative evaluations, causal analysis and assessment by art historians. Our findings indicate that 73% of the extracted concepts are judged by art historians to exhibit a coherent and semantically meaningful visual feature and 90% of concepts used to predict style of a given artwork were judged relevant. In cases where an irrelevant concept was used to successfully predict style, art historians identified possible reasons for its success; for example, the model might "understand" a concept in more formal terms, such as dark/light contrasts.
>
---
#### [replaced 057] An Automated Framework for Large-Scale Graph-Based Cerebrovascular Analysis
- **分类: cs.CV; cs.CY**

- **链接: [https://arxiv.org/pdf/2512.03869](https://arxiv.org/pdf/2512.03869)**

> **作者:** Daniele Falcetta; Liane S. Canas; Lorenzo Suppa; Matteo Pentassuglia; Jon Cleary; Marc Modat; Sébastien Ourselin; Maria A. Zuluaga
>
> **备注:** Accepted at IEEE ISBI 2026
>
> **摘要:** We present CaravelMetrics, a computational framework for automated cerebrovascular analysis that models vessel morphology through skeletonization-derived graph representations. The framework integrates atlas-based regional parcellation, centerline extraction, and graph construction to compute fifteen morphometric, topological, fractal, and geometric features. The features can be estimated globally from the complete vascular network or regionally within arterial territories, enabling multiscale characterization of cerebrovascular organization. Applied to 570 3D TOF-MRA scans from the IXI dataset (ages 20-86), CaravelMetrics yields reproducible vessel graphs capturing age- and sex-related variations and education-associated increases in vascular complexity, consistent with findings reported in the literature. The framework provides a scalable and fully automated approach for quantitative cerebrovascular feature extraction, supporting normative modeling and population-level studies of vascular health and aging.
>
---
#### [replaced 058] HOI-PAGE: Zero-Shot Human-Object Interaction Generation with Part Affordance Guidance
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.07209](https://arxiv.org/pdf/2506.07209)**

> **作者:** Lei Li; Angela Dai
>
> **备注:** ICML 2026. Project page: this https URL Video: this https URL
>
> **摘要:** We present HOI-PAGE, a new approach that prioritizes part-level affordance reasoning to generate high-fidelity 4D human-object interactions (HOIs) from text prompts in a zero-shot fashion. In contrast to prior works that focus on global, whole body-object motion synthesis, our approach explicitly reasons about the underlying part-level mechanics of interactions using large language models (LLMs). We capture this reasoning in a structured part affordance graph (PAG) representation, serving as a high-level interaction scaffolding to guide a three-stage synthesis: first, decomposing input 3D objects into semantic parts; then, generating reference HOI videos from text prompts to extract part-based motion constraints; and finally, optimizing for 4D HOI motion sequences that mimic the reference dynamics while satisfying part-level contact constraints. Extensive experiments show that our approach is flexible and capable of generating complex multi-object or multi-person interaction sequences, with significantly improved realism and text alignment for zero-shot 4D HOI generation.
>
---
#### [replaced 059] PlantTraitNet: An Uncertainty-Aware Multimodal Framework for Global-Scale Plant Trait Inference from Citizen Science Data
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.06943](https://arxiv.org/pdf/2511.06943)**

> **作者:** Ayushi Sharma; Johanna Trost; Daniel Lusk; Johannes Dollinger; Julian Schrader; Christian Rossi; Javier Lopatin; Etienne Laliberté; Simon Haberstroh; Jana Eichel; Daniel Mederer; Jose Miguel Cerda-Paredes; Shyam S. Phartyal; Lisa-Maricia Schwarz; Anja Linstädter; Maria Conceição Caldeira; Teja Kattenborn
>
> **备注:** Accepted at the 40th AAAI Conference on Artificial Intelligence (AAAI-26). Link: this https URL
>
> **摘要:** Global plant maps of plant traits, such as leaf nitrogen or plant height, are essential for understanding ecosystem processes, including the carbon and energy cycles of the Earth system. However, existing trait maps remain limited by the high cost and sparse geographic coverage of field-based measurements. Citizen science initiatives offer a largely untapped resource to overcome these limitations, with over 50 million geotagged plant photographs worldwide capturing valuable visual information on plant morphology and physiology. In this study, we introduce PlantTraitNet, a multi-modal, multi-task uncertainty-aware deep learning framework that predictsfour key plant traits (plant height, leaf area, specific leaf area, and nitrogen content) from citizen science photos using weak supervision. By aggregating individual trait predictions across space, we generate global maps of trait distributions. We validate these maps against independent vegetation survey data (sPlotOpen) and benchmark them against leading global trait products. Our results show that PlantTraitNet consistently outperforms existing trait maps across all evaluated traits, demonstrating that citizen science imagery, when integrated with computer vision and geospatial AI, enables not only scalable but also more accurate global trait mapping. This approach offers a powerful new pathway for ecological research and Earth system modeling.
>
---
#### [replaced 060] PureCC: Pure Learning for Text-to-Image Concept Customization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.07561](https://arxiv.org/pdf/2603.07561)**

> **作者:** Zhichao Liao; Xiaole Xian; Qingyu Li; Wenyu Qin; Meng Wang; Weicheng Xie; Siyang Song; Pingfa Feng; Long Zeng; Liang Pan
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Existing concept customization methods have achieved remarkable outcomes in high-fidelity and multi-concept customization. However, they often neglect the influence on the original model's behavior and capabilities when learning new personalized concepts. To address this issue, we propose PureCC. PureCC introduces a novel decoupled learning objective for concept customization, which combines the implicit guidance of the target concept with the original conditional prediction. This separated form enables PureCC to substantially focus on the original model during training. Moreover, based on this objective, PureCC designs a dual-branch training pipeline that includes a frozen extractor providing purified target concept representations as implicit guidance and a trainable flow model producing the original conditional prediction, jointly achieving pure learning for personalized concepts. Furthermore, PureCC introduces a novel adaptive guidance scale $\lambda^\star$ to dynamically adjust the guidance strength of the target concept, balancing customization fidelity and model preservation. Extensive experiments show that PureCC achieves state-of-the-art performance in preserving the original behavior and capabilities while enabling high-fidelity concept customization. The code is available at this https URL.
>
---
#### [replaced 061] HL-OutPaint: Coarse-to-Fine Video Outpainting for High-Resolution Long-Range Videos
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2605.17543](https://arxiv.org/pdf/2605.17543)**

> **作者:** Jeongeun Park; Janghyeok Han; Geonung Kim; Hyun-Seung Lee; Kyuha Choi; Youngseok Han; Sunghyun Cho
>
> **备注:** Supplementary material and video included. Project page: this https URL
>
> **摘要:** Video outpainting generates plausible visual content beyond the original spatial extent of a video, playing a key role in adapting videos to diverse display formats. To support such use cases, it must enable large spatial extrapolation over long sequences. However, most existing methods address only one of these challenges or lack explicit mechanisms for ensuring global spatio-temporal consistency, leading to notable limitations. In this paper, we propose HL-OutPaint, a high-resolution video outpainting framework for long sequences. Our approach follows a coarse-to-fine strategy with a two-stage pipeline. We first construct Global Coarse Guidance (GCG), a low-resolution representation that captures global structure and dominant motion across the video. Unlike naive downsampling, GCG is built via a novel global-local frame swapping mechanism that couples sparse global keyframes with local temporal windows and exchanges information during sampling. This enables GCG to encode both long-term structural consistency and short-term temporal dynamics in a unified representation. Guided by this representation, HL-OutPaint then performs high-resolution outpainting to generate spatially detailed and temporally consistent content. By separating global structure modeling from fine-grained synthesis, our framework achieves stable, coherent generation for large spatial expansion and long video sequences. Extensive experiments show that HL-OutPaint outperforms existing methods in challenging scenarios involving wide spatial extrapolation and long video sequences.
>
---
#### [replaced 062] Multimodal system for skin cancer detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.14822](https://arxiv.org/pdf/2601.14822)**

> **作者:** Volodymyr Sydorskyi; Igor Krashenyi; Oleksii Yakubenko
>
> **备注:** Accepted to System research and information technologies
>
> **摘要:** Melanoma detection is vital for early diagnosis and effective treatment. While deep learning models on dermoscopic images have shown promise, they require specialized equipment, limiting their use in broader clinical settings. This study introduces a multi-modal melanoma detection system using conventional photo images, making it more accessible and versatile. Our system integrates image data with tabular metadata, such as patient demographics and lesion characteristics, to improve detection accuracy. It employs a multi-modal neural network combining image and metadata processing and supports a two-step model for cases with or without metadata. A three-stage pipeline further refines predictions by boosting algorithms and enhancing performance. To address the challenges of a highly imbalanced dataset, specific techniques were implemented to ensure robust training. An ablation study evaluated recent vision architectures, boosting algorithms, and loss functions, achieving a peak Partial ROC AUC of 0.18068 (0.2 maximum) and top-15 retrieval sensitivity of 0.78371. Results demonstrate that integrating photo images with metadata in a structured, multi-stage pipeline yields significant performance improvements. This system advances melanoma detection by providing a scalable, equipment-independent solution suitable for diverse healthcare environments, bridging the gap between specialized and general clinical practices.
>
---
#### [replaced 063] UAVFF3D: A Geometry-Aware Benchmark for Feed-Forward UAV 3D Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.17942](https://arxiv.org/pdf/2605.17942)**

> **作者:** Xiang Yang; Yongli Wang; HaiFeng Li; Yunsheng Zhang
>
> **备注:** 19 pages, 16 figures, 16 tables
>
> **摘要:** Feed-forward 3D reconstruction has advanced rapidly, but current models remain unreliable in UAV photogrammetric acquisition. We argue that this failure is caused not only by appearance-domain shift, but also by UAV-specific camera-geometry variations, especially oblique views and HFOV-height ambiguity. Existing UAV datasets mainly emphasize scene diversity and provide limited coverage of camera configurations, which restricts robustness evaluation and UAV-domain adaptation. To address this gap, we introduce UAVFF3D, a geometry-aware real-synthetic benchmark for feed-forward UAV 3D reconstruction. UAVFF3D contains more than 170k real UAV images and more than 370k synthetic images rendered from high-quality textured 3D models, covering diverse HFOVs, flight altitudes, viewing directions, and acquisition patterns. It also includes a controlled HFOV-height test subset for diagnosing projection-geometry ambiguity. We further propose an evaluation protocol that jointly assesses camera-geometry estimation and dense scene reconstruction under a shared global alignment, avoiding the bias caused by separate camera and geometry alignments. Experiments on representative feed-forward reconstruction models show that UAVFF3D-based domain adaptation consistently improves camera and geometry estimation, reducing Ray Error by up to 84.2%, Pose ATE by up to 76.0%, and Chamfer Distance by up to 41.1%. In oblique scenes, adaptation reduces the oblique-nadir rotation gap by up to 90.7%. Under HFOV-height ambiguity, it improves robustness across HFOV-height configurations and yields more stable performance across HFOV settings. Incorporating camera priors further improves reconstruction under UAV-specific acquisition geometries. The dataset and evaluation code are available at this https URL .
>
---
#### [replaced 064] EchoSR: Efficient Context Harnessing for Lightweight Image Super-Resolution
- **分类: cs.CV; cs.MM; eess.IV**

- **链接: [https://arxiv.org/pdf/2605.17470](https://arxiv.org/pdf/2605.17470)**

> **作者:** Hanli Zhao; Binhao Wang; Shihao Zhao; Tao Wang; Kaihao Zhang; Wanglong Lu
>
> **备注:** Accepted by Information Fusion; 20 pages, 17 figures
>
> **摘要:** Image super-resolution (SR) aims to reconstruct high-quality, high-resolution (HR) images from low-resolution (LR) inputs and plays a critical role in various downstream applications. Despite recent advancements, balancing reconstruction fidelity and computational efficiency remains a fundamental challenge, particularly in resource-constrained scenarios. While existing lightweight methods attempt to expand receptive fields, many of them either incur substantial computational overhead, naively scale up kernel sizes, or lack mechanisms for coherent multi-scale integration, limiting their overall effectiveness and scalability. To address these limitations, we propose EchoSR, an efficient context-harnessing framework for lightweight image super-resolution, which unifies multi-scale receptive field modeling and hierarchical context fusion. EchoSR decouples feature learning into disentangled local, multi-scale, and global modeling stages through an efficient context-harnessing strategy, and further promotes seamless cross-scale integration via a cross-scale overlapping fusion mechanism. Extensive experiments have shown that EchoSR consistently outperforms state-of-the-art lightweight super-resolution methods across multiple benchmarks, while also achieving a faster speed $(\sim 2\times)$. The source code is available at this https URL.
>
---
#### [replaced 065] Jailbreaking on Text-to-Video Models via Scene Splitting Strategy
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.22292](https://arxiv.org/pdf/2509.22292)**

> **作者:** Wonjun Lee; Haon Park; Doehyeon Lee; Bumsub Ham; Suhyun Kim
>
> **备注:** ICLR 2026. Project page at this https URL
>
> **摘要:** Along with the rapid advancement of numerous Text-to-Video (T2V) models, growing concerns have emerged regarding their safety risks. While recent studies have explored vulnerabilities in models like LLMs, VLMs, and Text-to-Image (T2I) models through jailbreak attacks, T2V models remain largely unexplored, leaving a significant safety gap. To address this gap, we introduce SceneSplit, a novel black-box jailbreak method that works by fragmenting a harmful narrative into multiple scenes, each individually benign. This approach manipulates the generative output space, the abstract set of all potential video outputs for a given prompt, using the combination of scenes as a powerful constraint to guide the final outcome. While each scene individually corresponds to a wide and safe space where most outcomes are benign, their sequential combination collectively restricts this space, narrowing it to an unsafe region and significantly increasing the likelihood of generating a harmful video. This core mechanism is further enhanced through iterative scene manipulation, which bypasses the safety filter within this constrained unsafe region. Additionally, a strategy library that reuses successful attack patterns further improves the attack's overall effectiveness and robustness. To validate our method, we evaluate SceneSplit across 11 safety categories from T2VSafetyBench on T2V models. Our results show that it achieves a high average Attack Success Rate (ASR) of 77.2% on Luma Ray2, 84.1% on Hailuo, 78.2% on Veo2, 78.6% on Kling V1.0, and 68.6% on Sora2, significantly outperforming the existing baselines. Through this work, we demonstrate that current T2V safety mechanisms are vulnerable to attacks that exploit narrative structure, providing new insights for understanding and improving the safety of T2V models.
>
---
#### [replaced 066] Landslide Detection and Mapping Using Deep Learning Across Multi-Source Satellite Data and Geographic Regions
- **分类: cs.CV; cs.LG; eess.IV**

- **链接: [https://arxiv.org/pdf/2507.01123](https://arxiv.org/pdf/2507.01123)**

> **作者:** Rahul A. Burange; Harsh K. Shinde; Omkar Mutyalwar
>
> **备注:** 17 pages, 22 figures
>
> **摘要:** Landslides pose severe threats to infrastructure, economies, and human lives, necessitating accurate detection and predictive mapping across diverse geographic regions. With advancements in deep learning and remote sensing, automated landslide detection has become increasingly effective. This study presents a comprehensive approach integrating multi-source satellite imagery and deep learning models to enhance landslide identification and prediction. We leverage Sentinel-2 multispectral data and ALOS PALSAR-derived slope and Digital Elevation Model (DEM) layers to capture critical environmental features influencing landslide occurrences. Various geospatial analysis techniques are employed to assess the impact of terra in characteristics, vegetation cover, and rainfall on detection accuracy. Additionally, we evaluate the performance of multiple stateof-the-art deep learning segmentation models, including U-Net, DeepLabV3+, and Res-Net, to determine their effectiveness in landslide detection. The proposed framework contributes to the development of reliable early warning systems, improved disaster risk management, and sustainable land-use planning. Our findings provide valuable insights into the potential of deep learning and multi-source remote sensing in creating robust, scalable, and transferable landslide prediction models.
>
---
#### [replaced 067] Open-Set Domain Adaptation Under Background Distribution Shift: Challenges and A Provably Efficient Solution
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.01152](https://arxiv.org/pdf/2512.01152)**

> **作者:** Shravan Chaudhari; Yoav Wald; Suchi Saria
>
> **备注:** Project page at this https URL
>
> **摘要:** As we deploy machine learning systems in the real world, a core challenge is to maintain a model that is performant even as the data shifts. Such shifts can take many forms: new classes may emerge that were absent during training, a problem known as open-set recognition, and the distribution of known categories may change. Guarantees on open-set recognition are mostly derived under the assumption that the distribution of known classes, which we call the background distribution, is fixed. In this paper we develop CoLOR, a method that is guaranteed to solve open-set recognition even in the challenging case where the background distribution shifts. We prove that the method works under benign assumptions that the novel class is separable from the non-novel classes, and provide theoretical guarantees that it outperforms a representative baseline in a simplified overparameterized setting. We develop techniques to make CoLOR scalable and robust, and perform comprehensive empirical evaluations on image and text data. The results show that CoLOR significantly outperforms existing open-set recognition methods under background shift. Moreover, we provide new insights into how factors such as the size of the novel class influences performance, an aspect that has not been extensively explored in prior work.
>
---
#### [replaced 068] Cracks in the Foundation: A Civil Infrastructure Dataset to Challenge Vision Foundation Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.18413](https://arxiv.org/pdf/2605.18413)**

> **作者:** Nicola Farronato; Niccolo Avogaro; Thomas Frick; Mattia Rigotti; Rizwan Ullah Khan; Michele Magno; Konrad Schindler; Cristiano Malossi; Florian Scheidegger
>
> **摘要:** Automated structural health monitoring is essential to prevent catastrophic infrastructure failures. Precise, pixel-level defect segmentation is needed to accurately assess structural integrity, but progress in defect segmentation for civil infrastructures has been held back by an extreme scarcity of data, which requires costly expert annotation. The need for data is accentuated by algorithmic hurdles intrinsic to the problem, including center-bias and the need to rely more on shape when inspecting nearly textureless building materials. To remove the bottleneck, we introduce Cracks in the Foundation (CiF), the largest and most detailed civil infrastructure (instance) segmentation dataset to date, comprising $\approx$150,000 high-resolution images meticulously curated over five years in collaboration with civil engineering experts. With the help of this unprecedented data source, we expose a blind spot of current visual AI: despite the advent of promptable Foundation Models (FMs) and Vision Language Models (VLMs), and despite the impressive abilities of today's specialised segmentation models, it turns out that dense image understanding in the built environment is nowhere near solved. Our evaluations indicate that even the most recent zero-shot FMs face significant challenges when deployed on real-world infrastructure and even the performance of specialised models with domain-specific supervision plateaus at $\approx$25% mAP. CiF establishes inspection of civil infrastructure, an elementary and seemingly easy perceptual task, as an open challenge that reveals fundamental weaknesses of present-day models trained predominantly on internet images, literally and figuratively highlighting cracks in the current foundation model paradigm.
>
---
#### [replaced 069] Seeing SDG 6 from space: local-scale monitoring of piped water and sewage system access across Africa using satellite imagery and self-supervised learning
- **分类: cs.CV; cs.CY; cs.LG**

- **链接: [https://arxiv.org/pdf/2411.19093](https://arxiv.org/pdf/2411.19093)**

> **作者:** Othmane Echchabi; Aya Lahlou; Nizar Talty; Josh Malcolm Manto; Tongshu Zheng; Ka Leung Lam
>
> **备注:** Under Review
>
> **摘要:** Access to drinking water and sanitation is essential for health and well-being, yet major disparities remain, especially in data-scarce regions such as Africa. SDG 6 aims for universal access, but current monitoring relies on costly, infrequent, and spatially uneven surveys and censuses with long reporting delays. This study develops a scalable remote-sensing framework to estimate piped water and sewage system access at approximately 2.56 km resolution using Sentinel-2 imagery, Afrobarometer survey responses, 30 m population data, and DINO self-supervised Vision Transformer features. The best model achieves AUROC values of 91.54% for piped water and 93.24% for sewage access. Across 50 African countries, population-weighted estimates strongly align with WHO/UNICEF JMP statistics for piped water ($R^2 = 0.92$) and show meaningful agreement for sewage access ($R^2 = 0.72$). In countries without Afrobarometer coverage, MAEs are 9.5% and 10.7%, with estimates within 15% of JMP values for 121.4 million and 159.7 million people, respectively. A Nigeria case study across 767 Local Government Areas (LGAs) shows that the framework reveals fine-scale environmental inequality. The largest no-access burdens reach 1.155 million people for piped water and 1.452 million for sewage, 7.9 and 8.3 times the median LGA burden, while top-decile no-access thresholds of 0.805 and 0.952 indicate that deprivation is widespread. These findings show that DINO-based satellite models can complement household surveys with low-cost, spatially detailed evidence for SDG 6 monitoring, infrastructure targeting, and environmental equity assessment.
>
---
#### [replaced 070] MIRO: MultI-Reward cOnditioned pretraining improves T2I quality and efficiency
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.25897](https://arxiv.org/pdf/2510.25897)**

> **作者:** Nicolas Dufour; Lucas Degeorge; Arijit Ghosh; Vicky Kalogeiton; David Picard
>
> **备注:** Accepted at ICML 2026. Project page: this https URL
>
> **摘要:** The default paradigm of post-training text-to-image generators includes post-hoc selection of generated images, and subsequent training with one reward model to align the generator to the reward, typically user preference. This discards informative data as well as optimizes only for a single reward, hence harming diversity, semantic fidelity and efficiency. Instead, we propose MIRO, a method that conditions the model on multiple rewards during training, thus letting the model learn user preferences directly. MIRO pre-training both improves the visual quality of the generated images and speeds up the training, achieving state of the art on the GenEval compositional benchmark and user-preference scores (PickAScore, ImageReward, HPSv2).
>
---
#### [replaced 071] Hybrid Training for Vision-Language-Action Models
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文研究视觉-语言-动作模型的混合训练方法，旨在解决长链式思考影响推理效率的问题，通过允许推理时省略思考步骤，提升模型灵活性与实用性。**

- **链接: [https://arxiv.org/pdf/2510.00600](https://arxiv.org/pdf/2510.00600)**

> **作者:** Pietro Mazzaglia; Cansu Sancaktar; Markus Peschl; Daniel Dijkman
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** Using Large Language Models to produce intermediate thoughts, a.k.a. Chain-of-thought (CoT), before providing an answer has been a successful recipe for solving complex language tasks. In robotics, similar embodied CoT strategies, generating thoughts before actions, have also been shown to lead to improved performance when using Vision-Language-Action models (VLAs). As these techniques increase the length of the model's generated outputs to include the thoughts, the inference time is negatively affected. Delaying an agent's actions in real-world executions, as in robotic manipulation settings, strongly affects the usability of a method, as tasks require long sequences of actions. However, is the generation of long chains-of-thought a strong prerequisite for achieving performance improvements? In this work, we explore the idea of Hybrid Training (HyT), a framework that enables VLAs to learn from thoughts and benefit from the associated performance gains, while enabling the possibility to leave out CoT generation during inference. Furthermore, by learning to conditionally predict a diverse set of outputs, HyT supports flexibility at inference time, enabling the model to either predict actions directly, generate thoughts or follow instructions. We evaluate the proposed method in a series of simulated benchmarks and real-world experiments.
>
---
#### [replaced 072] SVG360: Editable Multiview Vector Graphics from a Single SVG
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16766](https://arxiv.org/pdf/2511.16766)**

> **作者:** Mengnan Jiang; Zhaolin Sun; Christian Franke; Michele Franco Adesso; Antonio Haas; Grace Li Zhang
>
> **摘要:** Scalable Vector Graphics are a standard representation for editable visual design, yet they are usually authored as single view two dimensional illustrations. This limits their use in applications that require object level assets to remain coherent when observed, edited, or animated from different viewpoints. We present SVG360, a framework that converts a single input SVG into geometrically and visually consistent multiview SVG assets. The key challenge is that direct per view generation or vectorization produces view dependent regions, fragmented paths, and unstable colors, making the resulting SVGs difficult to edit as a coherent object. SVG360 addresses this problem through a view consistent vectorization pipeline. It first lifts the rasterized input into a view conditioned object representation and renders target views under prescribed cameras. It then propagates part identity across neighboring views using a spatial memory mechanism adapted from video segmentation, establishing consistent region decomposition, path correspondence, and color assignment without task specific retraining. Finally, each view is reconstructed as an editable SVG through structure aware vectorization, where redundant paths are consolidated and local geometry is optimized while preserving boundaries and semantic parts. Experiments on object level SVG assets show that SVG360 improves multiview consistency, reduces path redundancy, and better preserves fine structures compared with direct per view vectorization. By turning a single view SVG into a coherent 360 degree vector asset, SVG360 expands vector graphics from static illustration toward editable multiview content for design, animation, and structured visual editing.
>
---
#### [replaced 073] VT-Bench: A Unified Benchmark for Visual-Tabular Multi-Modal Learning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.08146](https://arxiv.org/pdf/2605.08146)**

> **作者:** Zi-Yi Jia; Zi-Jian Cheng; Xin-Yue Zhang; Kun-Yang Yu; Zhi Zhou; Yu-Feng Li; Lan-Zhe Guo
>
> **摘要:** Multi-model learning has attracted great attention in visual-text tasks. However, visual-tabular data, which plays a pivotal role in high-stakes domains like healthcare and industry, remains underexplored. In this paper, we introduce \textit{VT-Bench}, the first unified benchmark for standardizing vision-tabular discriminative prediction and generative reasoning tasks. VT-Bench aggregates 14 datasets across 9 domains (medical-centric, while covering pets, media, and transportation) with over 756K samples. We evaluate 23 representative models, including unimodal experts, specialized visual-tabular models, general-purpose vision-language models (VLMs), and tool-augmented methods, highlighting substantial challenges of visual-tabular learning. We believe VT-Bench will stimulate the community to build more powerful multi-modal vision-tabular foundation models. Benchmark: this https URL
>
---
#### [replaced 074] Spatial-MLLM: Boosting MLLM Capabilities in Visual-based Spatial Intelligence
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2505.23747](https://arxiv.org/pdf/2505.23747)**

> **作者:** Diankun Wu; Fangfu Liu; Yi-Hsin Hung; Yueqi Duan
>
> **备注:** 22 pages
>
> **摘要:** Recent advancements in Multimodal Large Language Models (MLLMs) have significantly enhanced performance on 2D visual tasks. However, improving their spatial intelligence remains a challenge. Existing 3D MLLMs always rely on additional 3D or 2.5D data to incorporate spatial awareness, restricting their utility in scenarios with only 2D inputs, such as images or videos. In this paper, we present Spatial-MLLM, a novel framework for visual-based spatial reasoning from purely 2D observations. Unlike conventional video MLLMs which rely on CLIP-based visual encoders optimized for semantic understanding, our key insight is to unleash the strong structure prior from the feed-forward visual geometry foundation model. Specifically, we propose a dual-encoder architecture: a pretrained 2D visual encoder to extract semantic features, and a 3D spatial encoder-initialized from the backbone of the visual geometry model-to extract 3D structure features. A connector then integrates both features into unified visual tokens for enhanced spatial understanding. Furthermore, we propose a space-aware frame sampling strategy at inference time, which selects the spatially informative frames of a video sequence, ensuring that even under limited token length, the model focuses on frames critical for spatial reasoning. Beyond architecture improvements, we construct a training dataset from multiple sources and train the model on it using supervised fine-tuning and GRPO. Extensive experiments on various real-world datasets demonstrate that Spatial-MLLM achieves state-of-the-art performance in a wide range of visual-based spatial understanding and reasoning tasks. Project page: this https URL.
>
---
#### [replaced 075] VECTOR-Drive: Tightly Coupled Vision-Language and Trajectory Expert Routing for End-to-End Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出VECTOR-DRIVE，解决端到端自动驾驶中语义与轨迹耦合问题。通过共享注意力和专家路由，实现视觉语言与轨迹预测的紧密融合，提升驾驶性能。**

- **链接: [https://arxiv.org/pdf/2605.08830](https://arxiv.org/pdf/2605.08830)**

> **作者:** Rui Zhao; Jianlin Yu; Zhenhai Gao; Jiaqiao Liu; Fei Gao
>
> **摘要:** End-to-end autonomous driving requires models to understand traffic scenes, infer driving intent, and generate executable motion plans. Recent vision-language-action (VLA) models inherit semantic priors from large-scale vision-language pretraining, yet still face a coupling trade-off: fully shared backbones preserve multimodal interaction but may entangle language reasoning and trajectory prediction, whereas decou pled reasoning-action pipelines reduce task conflict but weaken semantic-motion coupling. We propose VECTOR-DRIVE, a tightly coupled VLA framework built on Qwen2.5-VL-3B. VECTOR-DRIVE keeps all tokens coupled through shared self attention and routes feed-forward computation according to token semantics. Vision and language tokens are processed by a Vision-Language Expert to preserve semantic priors, while target-point, ego-state, and noisy action tokens are routed to a Trajectory Expert for motion-specific computation. On the action-token pathway, a flow-matching planner refines noisy action tokens into future waypoints and speed profiles. This design couples semantic reasoning and motion planning within a single multimodal Transformer while separating task-specific FFN computation. On Bench2Drive, VECTOR-DRIVE achieves 88.91 Driving Score and outperforms representative end-to end and VLA-based baselines. Qualitative results and ablations further validate the benefits of shared attention, semantic-aware expert routing, progressive training, and flow-based action de coding.
>
---
#### [replaced 076] Structured State-Space Regularization for Generation-Friendly Image Tokenization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.11089](https://arxiv.org/pdf/2604.11089)**

> **作者:** Jinsung Lee; Jaemin Oh; Namhun Kim; Dongwon Kim; Byung-Jun Yoon; Suha Kwak
>
> **备注:** Related blog posts in this https URL : Towards 2-Dimensional State-Space Models series
>
> **摘要:** Image tokenizers play a central role in modern generative models, where the structure of the latent space critically determines the downstream generation performance. A key but underexplored property of effective latent representations is spectral organization, the ability to encode information across frequency components. In this work, we introduce structured state-space regularization, a principled approach to inducing spectral structure in latent spaces. We derive a regularization objective by revisiting state-space models (SSMs) as systems mimicking a basis function's behavior. This perspective reveals that hidden states of SSMs are induced to capture the frequency components, resulting in a novel regularizer that enforces the latent space to capture spectral structure of images. Experiments demonstrate that our regularizer improves the generative performance of image tokenizers while incurring only minimal loss in their reconstruction fidelity.
>
---
#### [replaced 077] HyperCap: Hyperspectral Land Cover Captioning Dataset for Vision Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.12217](https://arxiv.org/pdf/2505.12217)**

> **作者:** Aryan Das; Tanishq Rachamalla; Pravendra Singh; Koushik Biswas; Vinay Kumar Verma; Salvador Garcia; Antonio Plaza; Swalpa Kumar Roy
>
> **备注:** Accepted for publication in IEEE Geoscience and Remote Sensing Magazine (GRSM), 2026
>
> **摘要:** We introduce HyperCap, the first large-scale hyperspectral captioning dataset designed to enhance model performance and effectiveness in remote sensing applications. Unlike traditional hyperspectral imaging (HSI) benchmarks, HyperCap integrates spectral data with pixel-wise textual annotations, enabling deeper semantic understanding. This dataset enhances model performance in tasks like classification and feature extraction, providing a valuable resource for advanced remote sensing applications. HyperCap is constructed from four benchmark datasets and annotated through a hybrid approach combining automated and manual methods to ensure accuracy and consistency. Empirical evaluations using state-of-the-art encoders and diverse fusion techniques demonstrate significant improvements in classification performance. These results underscore the potential of vision-language learning in HSI and position HyperCap as a foundational dataset for future research in the field. The code and dataset are available at this https URL.
>
---
#### [replaced 078] MambaPanoptic: A Vision Mamba-based Structured State Space Framework for Panoptic Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.12640](https://arxiv.org/pdf/2605.12640)**

> **作者:** Qing Cheng; Damiano Bertolini; Wei Zhang; Dong Wang; Niclas Zeller; Daniel Cremers
>
> **备注:** Accepted to ISPRS Congress 2026, camera-ready version
>
> **摘要:** Panoptic segmentation requires the simultaneous recognition of countable thing instances and amorphous stuff regions, placing joint demands on long-range context modelling, multi-scale feature representation, and efficient dense prediction. Existing convolutional and transformer-based methods struggle to satisfy all three requirements concurrently: convolutional architectures are limited in their capacity to model long-range dependencies, while transformer-based methods incur quadratic computational cost that is prohibitive at high resolutions. In this paper, we propose MambaPanoptic, a fully Mamba-based panoptic segmentation framework that addresses these limitations through two principal contributions. First, we introduce MambaFPN, a top-down feature pyramid that leverages Mamba blocks to generate globally coherent, multi-scale feature representations with linear computational complexity. Second, we adopt a PanopticFCN-style kernel generator that produces unified thing and stuff kernels for proposal-free panoptic prediction, enhanced by a QuadMamba-based feature refinement module applied at multiple network stages. Experiments on the Cityscapes and COCO panoptic segmentation benchmarks demonstrate that MambaPanoptic consistently outperforms PanopticDeepLab and PanopticFCN under comparable model sizes, and matches or surpasses Mask2Former on Cityscapes in PQ and AP while requiring fewer parameters.
>
---
#### [replaced 079] Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.10292](https://arxiv.org/pdf/2511.10292)**

> **作者:** Zhengtao Zou; Ya Gao; Jiarui Guan; Bin Li; Pekka Marttinen
>
> **备注:** Accepted by ICML 2026; Code available at: this https URL
>
> **摘要:** Large Vision-Language Models (LVLMs) typically process visual inputs as a prefix to the language decoder. As the model autoregressively generates text, this initial visual information inevitably undergoes "dilution" leading the model to over-rely on language priors and hallucinate objects. Existing interventions attempt to correct this by contrasting logits or iteratively refining outputs, but they incur prohibitive latency costs. We propose Residual-Update Directed DEcoding Regulation (RUDDER), a framework that counters visual dilution by creating a persistent visual anchor. We extract a robust evidence direction (CARD) directly from the model's prefill residual updates, and inject it into the decoding process. This injection is modulated by an adaptive gate, the Beta Gate, which acts as a trust mechanism and ensures the visual reminder is applied only when necessary. Experiments on LLaVA-1.5 (7B/13B), Idefics2, InstructBLIP, and Qwen2.5-VL demonstrate that RUDDER consistently mitigates hallucination (with greedy decoding, RUDDER reduces CHAIR_S by an average of 24.4% and CHAIR_i by 23.6% relative) and scales effectively across architectures, all while maintaining >96.0% throughput.
>
---
#### [replaced 080] Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态语言模型任务，旨在解决细粒度视觉理解中的注意力不足问题。通过区域到全局的自蒸馏框架，提升模型对关键视觉证据的聚焦能力。**

- **链接: [https://arxiv.org/pdf/2605.18740](https://arxiv.org/pdf/2605.18740)**

> **作者:** Qianhao Yuan; Jie Lou; Xing Yu; Hongyu Lin; Le Sun; Xianpei Han; Yaojie Lu
>
> **备注:** Project page: this https URL
>
> **摘要:** Multimodal Large Language Models (MLLMs) still struggle with fine-grained visual understanding, where answers often depend on small but decisive evidence in the full image. We observe a regional-to-global perception gap: the same MLLM answers fine-grained questions more accurately when conditioned on evidence-centered crops than on the corresponding full images, suggesting that many failures stem from difficulty to focus on relevant evidence rather than insufficient local recognition ability. Motivated by this observation, we propose Vision-OPD (Vision On-Policy Distillation), a regional-to-global self-distillation framework that transfers the model's own privileged regional perception to its full-image policy. Vision-OPD instantiates two conditional policies from the same MLLM: a crop-conditioned teacher and a full-image-conditioned student. The student generates on-policy rollouts, and Vision-OPD minimizes token-level divergence between the teacher and student next-token distributions along these rollouts. This enables the model to internalize the benefit of visual zooming without external teacher models, ground-truth labels, reward verifiers, or inference-time tool use. Experiments on multiple fine-grained visual understanding benchmarks show that Vision-OPD models achieve competitive or superior performance against much larger open-source, closed-source, and "Thinking-with-Images" agentic models.
>
---
#### [replaced 081] PanoWorld: A Generative Spatial World Model for Consistent Whole-House Panorama Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.17916](https://arxiv.org/pdf/2605.17916)**

> **作者:** Jinrang Jia; Zhenjia Li; Yijiang Hu; Yifeng Shi
>
> **备注:** 17
>
> **摘要:** Generating a consistent whole-house VR tour from a floorplan and style reference requires both photorealistic panoramas and cross-view spatial coherence. Pure 2D generators produce appealing single panoramas but re-imagine geometry and materials when the viewpoint changes, whereas monolithic 3D generation becomes expensive and loses fine texture at multi-room scale. We introduce PanoWorld, a generative spatial world model that treats whole-house synthesis as autoregressive generation of node-based 360-degree panoramas, matching the discrete navigation used by real VR tour products. PanoWorld uses a floorplan-derived 3D shell as a global geometric proxy and a dynamic 3D Gaussian Splatting cache as renderable spatial memory. A feed-forward panoramic LRM designed for metric-scale multi-room 360-degree inputs lifts generated panoramas into local 3DGS updates, while Room-aware Group Attention suppresses cross-room feature interference. A topology-aware progressive caching strategy fuses these local updates without repeatedly reconstructing the full history. By decoupling shell-based geometry guidance from cache-rendered visual memory, PanoWorld preserves high-frequency 2D synthesis quality while improving cross-node layout and material consistency. The project link is this https URL
>
---
#### [replaced 082] 3DMambaComplete: Exploring Structured State Space Model for Point Cloud Completion
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2404.07106](https://arxiv.org/pdf/2404.07106)**

> **作者:** Yixuan Li; Weidong Yang; Ben Fei
>
> **备注:** 24 pages, 14 figures, 10 tables
>
> **摘要:** Point cloud completion aims to generate a complete and high-fidelity point cloud from an initially incomplete and low-quality input. A prevalent strategy involves leveraging Transformer-based models to encode global features and facilitate the reconstruction process. However, the adoption of pooling operations to obtain global feature representations often results in the loss of local details within the point cloud. Moreover, the attention mechanism inherent in Transformers introduces additional computational complexity, rendering it challenging to handle long sequences effectively. To address these issues, we propose 3DMambaComplete, a point cloud completion network built on the novel Mamba framework. It comprises three modules: HyperPoint Generation encodes point cloud features using Mamba's selection mechanism and predicts a set of Hyperpoints. A specific offset is estimated, and the down-sampled points become HyperPoints. The HyperPoint Spread module disperses these HyperPoints across different spatial locations to avoid concentration. Finally, a deformation method transforms the 2D mesh representation of HyperPoints into a fine-grained 3D structure for point cloud reconstruction. Extensive experiments conducted on various established benchmarks demonstrate that 3DMambaComplete surpasses state-of-the-art point cloud completion methods, as confirmed by qualitative and quantitative analyses.
>
---
#### [replaced 083] DISK: Differentiable Sparse Kernel Complex for Efficient Spatially-Variant Convolution
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.04556](https://arxiv.org/pdf/2512.04556)**

> **作者:** Zhizhen Wu; Zhe Cao; Yuchi Huo
>
> **备注:** Accepted as a conference paper at ICLR 2026. OpenReview: this https URL
>
> **摘要:** Image convolution with complex kernels is a fundamental operation in photography, scientific imaging, and animation effects, yet direct dense convolution is computationally prohibitive on resource-limited devices. Existing approximations, such as simulated annealing or low-rank decompositions, either lack efficiency or fail to capture non-convex kernels. We introduce a differentiable kernel decomposition framework that represents a target spatially-variant, dense, complex kernel using a set of sparse kernel samples. Our approach features (i) a decomposition that enables differentiable optimization of sparse kernels, (ii) a dedicated initialization strategy for non-convex shapes to avoid poor local minima, and (iii) a kernel-space interpolation scheme that extends single-kernel filtering to spatially varying filtering without retraining and additional runtime overhead. Experiments on Gaussian and non-convex kernels show that our method achieves higher fidelity than simulated annealing and significantly lower cost than low-rank decompositions. Our approach provides a practical solution for mobile imaging and real-time rendering, while remaining fully differentiable for integration into broader learning pipelines.
>
---
#### [replaced 084] Universal Skeleton Understanding via Differentiable Rendering and MLLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.18003](https://arxiv.org/pdf/2603.18003)**

> **作者:** Ziyi Wang; Peiming Li; Xinshun Wang; Yang Tang; Kai-Kuang Ma; Mengyuan Liu
>
> **备注:** 33 pages, 16 figures
>
> **摘要:** Multimodal large language models (MLLMs) exhibit strong visual-language reasoning, yet cannot process structured, non-visual data such as human skeletons. Existing methods either compress skeleton dynamics into lossy feature vectors for text alignment, or quantize motion into discrete tokens that generalize poorly across heterogeneous skeleton formats. We present SkeletonLLM, which achieves universal skeleton understanding by translating arbitrary skeleton sequences into the MLLM's native visual modality. At its core is DrAction, a differentiable, format-agnostic renderer that converts skeletal kinematics into compact image sequences. Because the pipeline is end-to-end differentiable, MLLM gradients can directly guide the rendering to produce task-informative visual tokens. To further enhance reasoning capabilities, we introduce a cooperative training strategy: Causal Reasoning Distillation transfers structured, step-by-step reasoning from a teacher model, while Discriminative Finetuning sharpens decision boundaries between confusable actions. SkeletonLLM demonstrates strong generalization \revise{in open-vocabulary action recognition, while its learned reasoning capabilities naturally extend to motion captioning and question answering across heterogeneous skeleton formats} -- suggesting a viable path for applying MLLMs to non-native modalities. Code: this https URL.
>
---
#### [replaced 085] Fast-BEV++: Fast by Algorithm, Deployable by Design
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.08237](https://arxiv.org/pdf/2512.08237)**

> **作者:** Yuanpeng Chen; Hui Song; Sheng Yang; Wei Tao; Shanhui Mo; Shuang Zhang; Xiao Hua; Tiankun Zhao
>
> **备注:** most up-to-date version
>
> **摘要:** The advancement of vision-only Bird's-Eye-View (BEV) perception, a core paradigm for cost-effective autonomous driving, is hindered by the long-standing fundamental trade-off between perception accuracy and on-device deployment efficiency. In this work, we introduce Fast-BEV++, a BEV perception framework that resolves this tension through two fundamental design principles: Fast by Algorithm and Deployable by Design. By decomposing the core view transformation module into a hardware-oriented standard Index-Gather-Reshape pipeline, Fast-BEV++ eliminates dependencies on custom kernels while achieving no less than 3 times speedup over the Fast-BEV baseline across mainstream edge platforms. Empirically, Fast-BEV++ establishes a new state-of-the-art result of 0.488 NDS on the nuScenes 3D object detection benchmark, simultaneously delivering real-time inference at more than 134 FPS via our acceleration design. In particular, our integrated, learnable depth module yields consistent performance gains, maintaining the highest accuracy among comparable methods. Overall, this inherently decomposed architecture enables seamless real-time deployment across diverse production-grade automotive platforms, alleviating hardware limitations without compromising perception accuracy or inference efficiency.
>
---
#### [replaced 086] STABLE: Simulation-Ready Tabletop Layout Generation via a Semantics-Physics Dual System
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于场景生成任务，解决从任务指令生成符合物理规则的桌面场景问题。提出STABLE系统，结合语义与物理修正模块，提升场景的合理性与准确性。**

- **链接: [https://arxiv.org/pdf/2605.16137](https://arxiv.org/pdf/2605.16137)**

> **作者:** Zhen Luo; Yixuan Yang; Xudong Xu; Jinkun Hao; Zhaoyang Lyu; Feng Zheng; Jiangmiao Pang; Yanwei Fu
>
> **备注:** ICML 2026
>
> **摘要:** Generating simulation-ready tabletop scenes from task instructions is an intriguing and promising research direction in the field of Embodied AI. However, existing task-to-scene generation methods rely exclusively on large language models (LLMs) to predict scene layouts, inevitably yielding object collisions or floating due to LLMs' inherent limitations in 3D spatial reasoning. In this paper, we present STABLE, a semantics-physics dual-system tailored for simulation-ready tabletop scene generation. STABLE consists of two complementary modules: (i) a Semantic Reasoner, a fine-tuned LLM trained on a structured tabletop scene dataset to generate coarse layouts from input task instructions, and (ii) a Physics Corrector, a physics-aware flow-based denoising model that outputs pose updates to refine layouts, which ensures the physical plausibility of scenes while preserves semantic alignment with task instructions. STABLE adopts a progressive generation paradigm: by alternating between the Semantic Reasoner and Physics Corrector, it incrementally expands the scene from task-critical objects to background objects. Experiments demonstrate that STABLE successfully generates simulation-ready tabletop scenes that strictly conform to task instructions and significantly enhances the physical validity of scenes over prior art.
>
---
#### [replaced 087] Weakly Supervised Cross-Modal Learning for 4D Radar Scene Flow Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.18507](https://arxiv.org/pdf/2605.18507)**

> **作者:** Jingyun Fu; Zhiyu Xiang; Na Zhao
>
> **备注:** Accepted by ICML2026
>
> **摘要:** Due to the difficulty of obtaining ground-truth data for 4D radar scene flow estimation, previous methods typically rely on either self-supervised losses or cross-modal supervision using 3D LiDAR data, 2D images, and odometry. However, self-supervised approaches often yield suboptimal results due to radar's inherently low-fidelity measurements, while existing cross-modal supervised methods introduce complex multi-task architecture and require costly LiDAR sensors to generate pseudo radar scene flow labels from pretrained 3D tracking models. To overcome these limitations, we propose a task-specific iterative framework for weakly supervised radar scene flow learning, using only images and odometry for auxiliary supervision during training. Specially, we establish two novel instance-aware self-supervised losses by exploiting off-the-shelf 2D tracking and segmentation algorithms to obtain tracked instance masks, which are back-projected into 3D space to provide instance-level semantic guidance; for static regions, we integrate vehicle odometry with radar's intrinsic motion cues to construct a rigid static loss. Extensive experiments on the real-world View-of-Delft (VoD) dataset demonstrate that our method not only surpasses state-of-the-art cross-modal supervised approaches that rely on 3D multi-object tracking on dense LiDAR point clouds but also outperforms existing fully supervised scene flow estimation methods. The code is open-sourced at \href{this https URL}{this https URL}.
>
---
#### [replaced 088] NeRF-based Spacecraft Reconstruction from Monocular Imagery Under Illumination Variability and Pose Uncertainty
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.18447](https://arxiv.org/pdf/2605.18447)**

> **作者:** Antoine Legrand; Renaud Detry; Christophe De Vleeschouwer
>
> **备注:** (under review)
>
> **摘要:** Autonomous rendezvous and proximity operations around uncooperative, unknown spacecraft are critical for active debris removal and on-orbit servicing missions. A key component of such operations is the offline reconstruction of a 3D model of the target from a set of 2D images. This task is challenging due to two main factors. First, in-orbit illumination conditions exhibit considerable variability, and change rapidly over time. Second, the inaccuracy of pose information in the images, results in 3D reconstruction uncertainty. To overcome these challenges, we propose to extend Neural Radiance Fields with per-image degrees of freedom: a learnable appearance embedding that captures the illumination conditions specific to each image, and an image-specific pose correction term that refines its noisy pose label to increase 3D consistency across images. These parameters add minimal complexity, as they are learned jointly with the NeRF, yet they substantially improve robustness to illumination variability and pose inaccuracies. We validate our approach on three image sets representative of in-orbit operations, demonstrating its effectiveness for offline reconstruction and highlighting its suitability for online reconstruction, an open problem in the field.
>
---
#### [replaced 089] Diversity-Preserved Distribution Matching Distillation for Fast Visual Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.03139](https://arxiv.org/pdf/2602.03139)**

> **作者:** Tianhe Wu; Ruibin Li; Lei Zhang; Kede Ma
>
> **摘要:** Distribution matching distillation (DMD) facilitates few-step image generation by aligning a distilled student with a reference multi-step teacher. In practice, however, optimizing DMD can reduce sample diversity in few-step synthesis, and existing remedies typically rely on perceptual or adversarial regularization, leading to stability and scalability challenges during training. Here, we describe diversity-preserved DMD (DP-DMD), a role-separated distillation method inspired by the complementary roles of early and late denoising steps. Specifically, the first distillation step is trained with a teacher-derived target-prediction objective (e.g., v-prediction) to preserve sample diversity, while the remaining steps are optimized with the standard DMD loss to refine perceptual quality. DP-DMD, with no perceptual or adversarial regularization, no additional modules, and no teacher-generated reference samples, preserves sample diversity while maintaining competitive visual quality under few-step sampling, providing a simple and stable alternative to other DMD variants.
>
---
#### [replaced 090] Hard-Label Black-Box Attacks on 3D Point Clouds
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.00404](https://arxiv.org/pdf/2412.00404)**

> **作者:** Daizong Liu; Yunbo Tao; Junhao Dong; Keke Tang; Pan Zhou; Wei Hu; Yew-Soon Ong
>
> **摘要:** With the maturity of depth sensors in various 3D safety-critical applications, 3D point cloud models have been shown to be vulnerable to adversarial attacks. Almost all existing 3D attackers simply follow the white-box or black-box setting to iteratively update coordinate perturbations based on back-propagated or estimated gradients. However, these methods are hard to deploy in real-world scenarios (no model details are provided) as they severely rely on parameters or output logits of victim models. To this end, we propose point cloud attacks from a more practical setting, i.e., hard-label black-box attack, in which attackers can only access the prediction label of 3D input. We introduce a novel 3D attack method based on a new spectrum-aware decision boundary algorithm to generate high-quality adversarial samples. In particular, we first construct a class-aware model decision boundary, by developing a learnable spectrum-fusion strategy to adaptively fuse point clouds of different classes in the spectral domain, aiming to craft their intermediate samples without distorting the original geometry. Then, we devise an iterative coordinate-spectrum optimization method with curvature-aware boundary search to move the intermediate sample along the decision boundary for generating adversarial point clouds with trivial perturbations. Experiments demonstrate that our attack competitively outperforms existing white/black-box attackers in terms of attack performance and adversary quality.
>
---
#### [replaced 091] EduVQA: Towards Concept-Aware Assessment of Educational AI-Generated Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.03066](https://arxiv.org/pdf/2603.03066)**

> **作者:** Baoliang Chen; Xinlong Bu; Hanwei Zhu; Lingyu Zhu; Jieyu Zhan
>
> **摘要:** Existing AI-generated video quality assessment (AIGVQA) methods mainly focus on global perceptual realism and coarse text-video alignment, while overlooking a critical requirement in educational scenarios: concept correctness. In early mathematics education, subtle errors in numerical quantities, geometric relations, or spatial configurations may fundamentally alter the conveyed knowledge despite visually plausible generation. To address this problem, we introduce EduAVQABench, the first benchmark for concept-aware educational AIGV assessment, containing 1,130 videos generated by ten state-of-the-art T2V models together with over 310,650 fine-grained human annotations spanning perceptual quality and semantic alignment. Built upon this benchmark, we further propose EduVQA, a concept-aware AIGVQA framework equipped with a Structured 2D Mixture-of-Experts (S2D-MoE) architecture. By jointly modeling fine-grained concept assessment and overall quality prediction through shared experts and adaptive two-dimensional routing, EduVQA effectively captures subtle concept-level inconsistencies overlooked by conventional global scoring methods. Extensive experiments demonstrate that EduVQA consistently outperforms existing AIGVQA approaches across both perceptual and semantic evaluation tasks while exhibiting strong generalization capability on unseen benchmarks. Code and dataset will be publicly available at: this https URL.
>
---
#### [replaced 092] Federated Distillation for Whole Slide Image via Gaussian-Mixture Feature Alignment and Curriculum Integration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.00578](https://arxiv.org/pdf/2605.00578)**

> **作者:** Luru Jing; Cong Cong; Yanyuan Chen; Yongzhi Cao
>
> **备注:** Accepted by ICML 2026, Camera-Ready version updated
>
> **摘要:** Federated learning (FL) offers a promising framework for collaborative digital pathology by enabling model training across institutions. However, real-world deployments face heterogeneity arising from diverse multiple instance learning (MIL) architectures and heterogeneous feature extractors across institutions. We propose FedHD, a novel FL framework that performs local Gaussian-mixture feature alignment tailored for WSI analysis. Instead of exchanging model parameters, each client independently distills semantically rich synthetic feature representations aligned with the distribution of real WSIs. To preserve diagnostic diversity, FedHD adopts a one-to-one distillation strategy, generating a synthetic counterpart for each real slide to avoid over-compression. During federation, a curriculum-based integration strategy progressively incorporates cross-site synthetic features into local training once performance plateaus. Furthermore, an optional interpretation module reconstructs pseudo-patches from synthetic embeddings, enhancing transparency. FedHD is architecture-agnostic, privacy-preserving, and supports personalized yet collaborative training across diverse institutions. Experiments on TCGA-IDH, CAMELYON16, and CAMELYON17 show that FedHD consistently outperforms state-of-the-art federated and distillation baselines.
>
---
#### [replaced 093] Improved visual-information-driven model for crowd simulation and its modular application
- **分类: cs.CY; cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2504.03758](https://arxiv.org/pdf/2504.03758)**

> **作者:** Xuanwen Liang; Jiayu Chen; Eric Wai Ming Lee; Wei Xie
>
> **摘要:** Crowd movement simulation is crucial for pedestrian safety management and facility design. Data-driven models offer the potential to improve realism and predictive accuracy, but most are developed for a single scenario, limiting their flexibility. We propose a data-driven crowd simulation model that incorporates refined visual-information extraction and explicit exit cues, aiming to improve flexibility across multiple scenarios by more effectively capturing core navigational features. The model is tested on four fundamental modules (bottleneck, corridor, corner, and T-junction) and further evaluated in a composite scenario using a modular approach. Results show that our model performs well across these scenarios, aligning with pedestrian movement in real-world experiments, and outperforms the classical knowledge-driven model in these scenarios. The research outcomes can provide inspiration for the development of data-driven crowd simulation models and advance the application of data-driven approaches.
>
---
#### [replaced 094] R$^3$L: Reasoning 3D Layouts from Relative Spatial Relations
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出R$^3$L框架，解决3D布局生成中的相对空间推理问题，提升布局的物理可行性和语义一致性。**

- **链接: [https://arxiv.org/pdf/2605.06758](https://arxiv.org/pdf/2605.06758)**

> **作者:** Zhifeng Gu; Yuqi Wang; Bing Wang
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Relative spatial relations provide a compact representation of spatial structure and are fundamental to relative spatial reasoning in 3D layout generation. Recent works leverage Multimodal Large Language Models (MLLMs) to infer such relations, but the inferred relations are often unreliable and are typically handled with post-hoc heuristics. In this paper, we propose R$^3$L, a general framework that improves the reliability and consistency of relative spatial reasoning for 3D layout generation. Our key motivation is that multi-hop reasoning requires repeated reference-frame transformations, which accumulate errors in inferred relations and lead to semantic and metric drift. To mitigate this, we propose invariant spatial decomposition to break coupled relation chains, and consistent spatial imagination to promote self-consistency through an imagine-and-revise loop. We further introduce supportive spatial optimization to ease pose optimization via global-to-local coordinate re-parameterization. Extensive experiments across diverse scene types and instructions demonstrate that R$^3$L produces more physically feasible and semantically consistent layouts. Notably, our analysis shows that resolving frame-induced inconsistencies is crucial for reliable multi-hop relative spatial reasoning. The code is available at this https URL.
>
---
#### [replaced 095] Xiaomi EV World Model: A Joint World Model Integrating Reconstruction and Generation for Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.18137](https://arxiv.org/pdf/2605.18137)**

> **作者:** Lijun Zhou; Hongcheng Luo; Zhenxin Zhu; Cheng Chi; Mingfei Tu; Kaixin Xiong; Lei Gong; Zhanqian Wu; Zehan Zhang; Fangzhen Li; Hao Li; Yingying Shen; Jiale He; Haohui Zhu; Shan Zhao; Kai Wang; Zhiwei Zhan; Yuechuan Pu; Kaiyuan Tan; Ruiling Yang; Xianqi Wang; Tianyi Yan; Jiawei Zhou; Lei Zhang; Jingyang Zhao; Xi Zhou; Chitian Sun; Chenming Wu; Jiong Deng; Hongwei Xie; Ming Lu; Kun Ma; Long Chen; Guang Chen; Hangjun Ye; Bing Wang; Haiyang Sun
>
> **摘要:** This report presents a unified technical system addressing the two core capabilities of world models for autonomous driving: world representation and world generation. For world representation, we propose WorldRec, a feed-forward reconstruction architecture driven by sparse scene queries. WorldRec initializes structured queries in 3D space, leveraging them to aggregate cross-view, cross-temporal features, thereby naturally enforcing spatial consistency across frames and yielding compact yet high-fidelity 3D Gaussian scene representations. For world generation, we propose WorldGen, a two-stage training framework of bidirectional pretraining followed by causal fine-tuning through three progressive stages (Teacher Forcing, ODE distillation, and DMD), enabling high-quality online causal video generation in as few as 4 denoising steps. Building on both modules, we further introduce the JWM, which deeply integrates WorldRec and WorldGen to achieve synergistic gains in generation stability, cross-frame consistency, and visual fidelity, providing a solid foundation for closed-loop simulation, data synthesis, and end-to-end training in autonomous driving.
>
---
#### [replaced 096] Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.20981](https://arxiv.org/pdf/2502.20981)**

> **作者:** Fuyun Wang; Tong Zhang; Yuanzhi Wang; Yide Qiu; Xin Liu; Xu Guo; Zhen Cui
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** In Open-set Supervised Anomaly Detection (OSAD), the existing methods typically generate pseudo anomalies to compensate for the scarcity of observed anomaly samples, while overlooking critical priors of normal samples, leading to less effective discriminative boundaries. To address this issue, we propose a Distribution Prototype Diffusion Learning (DPDL) method aimed at enclosing normal samples within a compact and discriminative distribution space. Specifically, we construct multiple learnable Gaussian prototypes to create a latent representation space for abundant and diverse normal samples and learn a Schrödinger bridge to facilitate a diffusive transition toward these prototypes for normal samples while steering anomaly samples away. Moreover, to enhance inter-sample separation, we design a dispersion feature learning way in hyperspherical space, which benefits the identification of out-of-distribution anomalies. Experimental results demonstrate the effectiveness and superiority of our proposed DPDL, achieving state-of-the-art performance on 9 public datasets.
>
---
#### [replaced 097] SAMe: A Semantic Anatomy Mapping Engine for Robotic Ultrasound
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出SAMe，解决机器人超声扫描初始化问题。通过语义解剖映射，实现自主定位与控制，提升扫描效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.25646](https://arxiv.org/pdf/2604.25646)**

> **作者:** Jing Zhang; Duojie Chen; Wentao Jiang; Zihan Lou; Jianxin Liu; Xinwu Cui; Qinghong Zhao; Bo Du; Christoph F. Dietrich; Dacheng Tao
>
> **备注:** Supplementary information included. Code will be released at this https URL
>
> **摘要:** Robotic ultrasound has advanced local image-driven control, contact regulation, and view optimization, yet current systems lack the anatomical understanding needed to determine what to scan, where to begin, and how to adapt to individual patient anatomy. These gaps make systems still reliant on expert intervention to initiate scanning. Here we present SAMe, a semantic anatomy mapping engine that provides robotic ultrasound with an explicit anatomical prior layer. SAMe addresses scan initiation as a target-to-anatomy-to-action process: it grounds under-specified clinical complaints into structured target organs, instantiates a patient-specific anatomical representation for the grounded targets from a single external body image, and translates this representation into control-facing 6-DoF probe initialization states without any additional registration using preoperative CT or MRI. The anatomical representation maintained by SAMe is explicit, lightweight (single-organ inference in 0.08s), and compatible with downstream control by design. Across semantic grounding, anatomical instantiation, and real-robot evaluation, SAMe shows strong performance across the full initialization pipeline. In real-robot experiments, centroid-based SAMe initialization outperformed the body-keypoint-based heuristic baseline under a budget-matched single-target setting for both liver (86.7% versus 46.7%) and kidney (80.0% versus 73.3%) initialization. Furthermore, The trial-level organ-hit rate reached 97.3% for liver and 83.3% for kidney when multiple candidate targets were available. These results establish an explicit anatomical prior layer that addresses scan initialization and is designed to support broader downstream autonomous scanning pipelines, providing the anatomical foundation for complaint-driven, anatomically informed robotic ultrasonography.
>
---
#### [replaced 098] Rapid patient-specific neural networks for intraoperative X-ray to volume registration
- **分类: eess.IV; cs.CV; physics.med-ph**

- **链接: [https://arxiv.org/pdf/2503.16309](https://arxiv.org/pdf/2503.16309)**

> **作者:** Vivek Gopalakrishnan; David-Dimitris Chlorogiannis; Andrew Abumoussa; Anna M. Larson; Nazim Haouchine; Darren B. Orbach; Sarah Frisken; Neel Dey; Polina Golland
>
> **摘要:** Advanced navigation techniques in image-guided interventions and surgical robotics require the rapid and precise alignment of 3D preoperative volumes (e.g., CT, MRI) to 2D intraoperative images (e.g., X-ray fluoroscopy). However, existing 2D/3D registration methods fail to generalize across the broad spectrum of fluoroscopy-guided procedures: traditional intensity-based optimizers require careful hyperparameter tuning for each subject, while deep learning approaches demand extensive manually labeled datasets and remain constrained to the specific anatomy on which they were trained. To address these limitations, we present xvr, a self-supervised framework that combines patient-specific neural networks with gradient-based optimization for automatic 2D/3D registration. xvr leverages physics-based simulation to generate training data from a patient's own preoperative scan, eliminating the need for manual annotation. We present a foundation model pretrained on thousands of whole-body scans, achieving patient-specific adaptation for any anatomical region in only 5 minutes of finetuning. In the largest evaluation of 2D/3D registration on real fluoroscopy to date, xvr achieves high accuracy in seconds across diverse anatomical structures, imaging modalities, and hospitals, improving upon the accuracy of existing methods by an order of magnitude. xvr makes pan-anatomical 2D/3D rigid registration accessible to broad clinical and research communities through open-source software at this https URL.
>
---
#### [replaced 099] MobileEgo Anywhere: Open Infrastructure for long horizon egocentric data on commodity hardware
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出MobileEgo Anywhere框架，解决长时序自指数据收集难题。通过手机传感器实现高精度长期姿态跟踪，释放大规模长时序数据集，支持视觉语言动作模型研究。**

- **链接: [https://arxiv.org/pdf/2605.05945](https://arxiv.org/pdf/2605.05945)**

> **作者:** Senthil Palanisamy; Abhishek Anand; Satpal Singh Rathor; Pratyush Patnaik; Shubhanshu Khatana; Ekaksh Janweja
>
> **摘要:** The recent advancement of Vision Language Action (VLA) models has driven a critical demand for large scale egocentric datasets. However, existing datasets are often limited by short episode durations, typically spanning only a few minutes, which fails to capture the long horizon temporal dependencies necessary for complex robotic task execution. To bridge this gap, we present MobileEgo Anywhere, a framework designed to facilitate the collection of robust, hour plus egocentric trajectories using commodity mobile hardware. We leverage the ubiquitous sensor suites of modern smartphones to provide high fidelity, long term camera pose tracking, effectively removing the high hardware barriers associated with traditional robotics data collection. Our contributions are three fold: (1) we release a novel dataset comprising 200 hours of diverse, long form egocentric data with persistent state tracking; (2) we open source our whole video processing infrastructure - STERA - that enables any user to record and process egocentric data, and (3) we provide a comprehensive processing pipeline to convert raw mobile captures into standardized, training ready formats for Vision Language Action model and foundation model research. By democratizing the data collection process, this work enables the massive scale acquisition of long horizon data across varied global environments, accelerating the development of generalizable robotic policies. Dataset and code can be accessed from this https URL
>
---
#### [replaced 100] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11688](https://arxiv.org/pdf/2511.11688)**

> **作者:** Aihua Zhu; Rui Su; Qinglin Zhao; Li Feng; Meng Shen; Shibo He
>
> **备注:** Preprint, accepted to AAAI 2026
>
> **摘要:** Diffusion probabilistic models have set a new standard for generative fidelity but are hindered by a slow iterative sampling process. A powerful training-free strategy to accelerate this process is Schedule Optimization, which aims to find an optimal distribution of timesteps for a fixed and small Number of Function Evaluations (NFE) to maximize sample quality. To this end, a successful schedule optimization method must adhere to four core principles: effectiveness, adaptivity, practical robustness, and computational efficiency. However, existing paradigms struggle to satisfy these principles simultaneously, motivating the need for a more advanced solution. To overcome these limitations, we propose the Hierarchical-Schedule-Optimizer (HSO), a novel and efficient bi-level optimization framework. HSO reframes the search for a globally optimal schedule into a more tractable problem by iteratively alternating between two synergistic levels: an upper-level global search for an optimal initialization strategy and a lower-level local optimization for schedule refinement. This process is guided by two key innovations: the Midpoint Error Proxy (MEP), a solver-agnostic and numerically stable objective for effective local optimization, and the Spacing-Penalized Fitness (SPF) function, which ensures practical robustness by penalizing pathologically close timesteps. Extensive experiments show that HSO sets a new state-of-the-art for training-free sampling in the extremely low-NFE regime. For instance, with an NFE of just 5, HSO achieves a remarkable FID of 11.94 on LAION-Aesthetics with Stable Diffusion v2.1. Crucially, this level of performance is attained not through costly retraining, but with a one-time optimization cost of less than 8 seconds, presenting a highly practical and efficient paradigm for diffusion model acceleration.
>
---
#### [replaced 101] Character-Centered Dialogue Generation from Scene-Level Prompts
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.16819](https://arxiv.org/pdf/2505.16819)**

> **作者:** Taewon Kang; Ming C. Lin
>
> **备注:** Accepted to the 2026 IEEE International Conference on Image Processing (ICIP 2026). 18 pages, 5 figures
>
> **摘要:** Recent advances in scene-based video generation enable coherent visual narratives from structured prompts, yet a key aspect of storytelling -- character-driven dialogue and speech -- remains underexplored. We present a modular pipeline that transforms action-level prompts into visually and auditorily grounded dialogue, enriching scene-based storytelling with natural voice and character expression. Our method takes a pair of prompts per scene, defining the setting and character behavior. While a story generation model such as Text2Story produces the visual scene, we focus on generating expressive, character-consistent utterances grounded in both the prompts and a representative scene image. A pretrained vision-language encoder extracts high-level visual semantics, which are combined with structured prompts to guide a large language model for dialogue synthesis. To maintain contextual and emotional consistency across scenes, we introduce a Recursive Narrative Bank, a speaker-aware, temporally structured memory that accumulates each character's dialogue history. Inspired by Script Theory, this design enables dialogue that reflects evolving goals, social context, and narrative roles. Finally, we render each utterance as expressive, character-conditioned speech, producing fully voiced, multimodal video narratives. Our training-free framework generalizes across diverse story settings, providing a scalable solution for coherent, character-grounded audiovisual storytelling.
>
---
#### [replaced 102] HiDe: Rethinking The Zoom-IN method in High Resolution MLLMs via Hierarchical Decoupling
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.00054](https://arxiv.org/pdf/2510.00054)**

> **作者:** Xianjie Liu; Yiman Hu; Yixiong Zou; Liang Wu; Jian Xu; Bo Zheng
>
> **摘要:** Multimodal Large Language Models (MLLMs) have made significant strides in visual understanding tasks. However, their performance on high-resolution images remains suboptimal. While existing approaches often attribute this limitation to perceptual constraints and argue that MLLMs struggle to recognize small objects, leading them to use "zoom in" strategies for better detail, our analysis reveals a different cause: the main issue is not object size, but rather caused by complex background interference. We systematically analyze this "zoom in" operation through a series of decoupling experiments and propose the Hierarchical Decoupling Framework (HiDe), a training-free framework that uses Token-wise Attention Decoupling (TAD) to decouple the question tokens and identify the key information tokens, then leverages their attention weights to achieve precise alignment with the target visual regions. Subsequently, it employs Layout-Preserving Decoupling (LPD) to decouple these regions from the background and reconstructs a compact representation that preserves essential spatial layouts while eliminating background interference. HiDe sets a new SOTA on V*Bench, HRBench4K, and HRBench8K, boosting Qwen2.5-VL 7B and InternVL3 8B to SOTA (92.1% and 91.6% on V*Bench), even surpassing RL methods. After optimization, HiDe uses 75% less memory than the previous training-free approach. Code is provided in this https URL.
>
---
#### [replaced 103] Differential-Integral Neural Operator for Long-Term Turbulence Forecasting
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.21196](https://arxiv.org/pdf/2509.21196)**

> **作者:** Hao Wu; Yuan Gao; Fan Xu; Fan Zhang; Qingsong Wen; Kun Wang; Xiaomeng Huang; Xian Wu
>
> **摘要:** Accurately forecasting the long-term evolution of turbulence represents a grand challenge in scientific computing and is crucial for applications ranging from climate modeling to aerospace engineering. Existing deep learning methods, particularly neural operators, often fail in long-term autoregressive predictions, suffering from catastrophic error accumulation and a loss of physical fidelity. This failure stems from their inability to simultaneously capture the distinct mathematical structures that govern turbulent dynamics: local, dissipative effects and global, non-local interactions. In this paper, we propose the {\textbf{\underline{D}}}ifferential-{\textbf{\underline{I}}}ntegral {\textbf{\underline{N}}}eural {\textbf{\underline{O}}}perator (\method{}), a novel framework designed from a first-principles approach of operator decomposition. \method{} explicitly models the turbulent evolution through parallel branches that learn distinct physical operators: a local differential operator, realized by a constrained convolutional network that provably converges to a derivative, and a global integral operator, captured by a Transformer architecture that learns a data-driven global kernel. This physics-based decomposition endows \method{} with exceptional stability and robustness. Through extensive experiments on the challenging 2D Kolmogorov flow benchmark, we demonstrate that \method{} significantly outperforms state-of-the-art models in long-term forecasting. It successfully suppresses error accumulation over hundreds of timesteps, maintains high fidelity in both the vorticity fields and energy spectra, and establishes a new benchmark for physically consistent, long-range turbulence forecast.
>
---
#### [replaced 104] GEASS: Gated Evidence-Adaptive Selective Caption Trust for Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.01733](https://arxiv.org/pdf/2605.01733)**

> **作者:** Zeshang Li; Shuoyang Zhang
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Vision-Language Models (VLMs) excel at grounded reasoning but remain prone to object hallucination. Recent work treats self-generated captions as a uniformly positive resource, yet we find that naively embedding one can degrade rather than help--dropping Qwen2.5-VL-3B accuracy on HallusionBench by nearly 10 points. Two structural properties explain this. First, captions anchor not only the model's final answer but also its reasoning trajectory and lexical choices. Second, caption errors are asymmetric: omissions vastly outnumber fabrications, yet each fabrication carries a much larger per-instance impact. A caption's usefulness is therefore a per-query property, not a per-corpus one. We propose GEASS (ated Evidence-Adaptive Selective Caption Trust ), a training-free module that decides on each query how much of the caption the model consumes: it gates the caption by the clean path's confidence, weights it by the entropy reduction it produces, and raises the evidence bar when the two pathways disagree. Experiments on POPE and HallusionBench across four VLMs show that GEASS consistently improves over vanilla inference and contrastive decoding, with only two extra forward passes per query.
>
---
#### [replaced 105] FreeOrbit4D: Training-Free Arbitrary Camera Redirection for Monocular Videos via Foreground-Complete 4D Reconstruction
- **分类: cs.CV; cs.AI; cs.GR**

- **链接: [https://arxiv.org/pdf/2601.18993](https://arxiv.org/pdf/2601.18993)**

> **作者:** Wei Cao; Hao Zhang; Fengrui Tian; Yulun Wu; Yingying Li; Shenlong Wang; Ning Yu; Yaoyao Liu
>
> **备注:** 12 pages, 10 figures. Accepted to SIGGRAPH Conference Papers 2026
>
> **摘要:** Camera redirection aims to replay a dynamic scene from a single monocular video under a user-specified camera trajectory. However, large-angle redirection is inherently ill-posed: a monocular video captures only a narrow spatio-temporal view of a dynamic 3D scene, providing severely limited observations of the underlying 4D world. The key challenge is therefore to recover a complete and coherent representation from this limited input, with consistent geometry and motion. While recent diffusion-based methods achieve impressive visual generation quality, they often break down under large-angle viewpoint changes far from the original trajectory, where missing visual grounding leads to severe geometric ambiguity and temporal inconsistency. We present FreeOrbit4D, an effective training-free framework that tackles this ambiguity by recovering a foreground-complete 4D proxy as structural grounding for video generation. We obtain this proxy by decoupling foreground and background reconstructions: we unproject the monocular video into a static background and partial foreground point clouds in a unified global space, then use an object-centric multi-view diffusion model to synthesize multi-view images and reconstruct complete foreground point clouds in canonical object space. By aligning the canonical foreground point cloud to the global scene space via dense pixel-synchronized 3D-3D correspondences and projecting the foreground-complete 4D proxy onto target camera viewpoints, we provide geometric scaffolds that guide a conditional video diffusion model. Extensive experiments show that FreeOrbit4D produces more faithful and temporally coherent redirected videos under challenging large-angle trajectories, and our proxy further enables applications such as edit propagation and 4D data generation. Project page: this https URL
>
---
#### [replaced 106] Is SAM3 ready for pathology segmentation?
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.18225](https://arxiv.org/pdf/2604.18225)**

> **作者:** Qiuyu Kong; Shakiba Sharifi; Yiming Wang; Marco Cristani; Zanxi Ruan
>
> **备注:** accept to icip2026
>
> **摘要:** Is Segment Anything Model 3 (SAM3) capable in segmenting Any Pathology Images? Digital pathology segmentation spans tissue-level and nuclei-level scales, where traditional methods often suffer from high annotation costs and poor generalization. SAM3 introduces Promptable Concept Segmentation, offering a potential automated interface via text prompts. With this work, we propose a systematic evaluation protocol to explore the capability space of SAM3 in a structured manner. Specifically, we evaluate SAM3 under different supervision settings including zero-shot, few-shot, and supervised with varying prompting strategies. Our extensive evaluation on pathological datasets including NuInsSeg, PanNuke and GlaS, reveals that: (1) text-only prompts poorly activate nuclear concepts; (2) performance is highly sensitive to visual prompt types and budgets; (3) few-shot learning offers gains, but SAM3 lacks robustness against visual prompt noise; and (4) a significant gap persists between prompt-based usage and task-trained adapter-based reference. Our study delineates SAM3's boundaries in pathology image segmentation and provides practical guidance on the necessity of pathology domain adaptation.
>
---
#### [replaced 107] BabyMamba-HAR: Lightweight Selective State Space Models for Efficient Human Activity Recognition on Resource Constrained Devices
- **分类: cs.CV; cs.HC**

- **链接: [https://arxiv.org/pdf/2602.09872](https://arxiv.org/pdf/2602.09872)**

> **作者:** Mridankan Mandal
>
> **摘要:** Human activity recognition (HAR) on resource constrained devices requires high accuracy across diverse sensor setups. Selective state space models (SSMs) offer efficient linear time sequence processing, presenting a compelling alternative to attention mechanisms. However, their TinyML design space remains unexplored. This paper introduces BabyMamba-HAR, comprising two lightweight architectures: (1) CI-BabyMamba-HAR, utilizing a channel independent stem for noise robustness, and (2) Crossover-BiDir-BabyMamba-HAR, utilizing an early fusion stem for channel count independent complexity. Both integrate weight tied bidirectional scanning and gated temporal attention pooling. Across eight benchmarks, Crossover-BiDir-BabyMamba-HAR averages an 86.52% F1-score with 27K parameters and 2.21M MACs, matching TinyHAR (86.16%) while requiring 11x fewer MACs on high channel datasets. On-device deployment on the Raspberry Pi Pico 2 and ESP32 utilized a mixed precision C++ runtime (INT8 projections, float32 states). A fused computation strategy with lifetime aware memory management reduces peak memory footprint from O(B*dmodel*L*dstate) to O(B*dmodel*dstate), adapting to support weight-tied bidirectional and channel-streaming execution. Both architectures achieved full 8/8 dataset coverage with >99.2% PyTorch parity, whereas INT8 quantized TFLite baselines showed degraded coverage and parity (TinyHAR: 7/8 and 4/8 coverage at 60.4% and 88.6% parity, TinierHAR: 8/8 and 6/8 at 54.2% and 90.8%, DeepConvLSTM: 1/8 and 0/8 on Pico 2 and ESP32, respectively). Crossover-BiDir-BabyMamba-HAR averages 154.4 ms latency on ESP32 and 481.9 ms on Pico 2. Ablations confirm bidirectional scanning and gated attention improve F1-scores by up to 8.42% and 8.94%, respectively, establishing practical principles for TinyML SSM deployment.
>
---
#### [replaced 108] How does longer temporal context enhance multimodal narrative video processing in the brain?
- **分类: q-bio.NC; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.07570](https://arxiv.org/pdf/2602.07570)**

> **作者:** Prachi Jindal; Anant Khandelwal; Manish Gupta; Bapi S. Raju; Subba Reddy Oota; Tanmoy Chakraborty
>
> **备注:** 22 pages, 15 figures
>
> **摘要:** Understanding how humans and artificial intelligence systems process complex narrative videos is a fundamental challenge at the intersection of neuroscience and machine learning. This study investigates how the temporal context length of video clips (3--24 s clips) and the narrative-task prompting shape brain-model alignment during naturalistic movie watching. Using fMRI recordings from participants viewing full-length movies, we examine how brain regions sensitive to narrative context dynamically represent information over varying timescales and how these neural patterns align with model-derived features. We find that increasing clip duration substantially improves brain alignment for multimodal large language models (MLLMs), whereas unimodal video models show little to no gain. Further, shorter temporal windows align with perceptual and early language regions, while longer windows preferentially align higher-order integrative regions, mirrored by a layer-to-cortex hierarchy in MLLMs. Finally, experiments with four narrative-task prompts show that they elicit task-specific, region-dependent brain alignment patterns and context-dependent shifts in clip-level tuning in higher-order regions. Our work positions long-form narrative movies as a principled testbed for studying long-timescale temporal integration in long-context MLLMs and its relationship to cortical responses during narrative comprehension.
>
---
#### [replaced 109] ProJo4D: Progressive Joint Optimization for Sparse-View Inverse Physics Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.05317](https://arxiv.org/pdf/2506.05317)**

> **作者:** Daniel Rho; Jun Myeong Choi; Biswadip Dey; Roni Sengupta
>
> **备注:** TMLR 2026
>
> **摘要:** Neural rendering has advanced significantly in 3D reconstruction and novel view synthesis, and integrating physics into these frameworks opens new applications such as physically accurate digital twins for robotics and XR. However, the inverse problem of estimating physical parameters from visual observations remains challenging. Existing physics-aware neural rendering methods typically require dense multi-view videos, making them impractical for scalable, real-world deployment. Under sparse-view settings, the sequential optimization strategies employed by current approaches suffer from severe error accumulation: inaccuracies in initial 3D reconstruction propagate to subsequent stages, degrading physical state and material parameter estimates. On the other hand, simultaneous optimization of all parameters fails due to the highly non-convex and often non-differentiable nature of the problem. We propose ProJo4D, a progressive joint optimization framework that gradually expands the set of jointly optimized parameters. This design enables physics-informed gradients to refine geometry while avoiding the instability of direct joint optimization over all parameters. Evaluations on synthetic and real-world datasets demonstrate that ProJo4D substantially outperforms prior work in 4D future state prediction and physical parameter estimation, achieving up to 10x improvement in geometric accuracy while maintaining computational efficiency. Please visit the project webpage: this https URL
>
---
#### [replaced 110] Segment Anything with Robust Uncertainty-Accuracy Correlation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.10603](https://arxiv.org/pdf/2605.10603)**

> **作者:** Hongyou Zhou; Marc Toussaint; Ling Shao; Zihan Ye
>
> **备注:** ICML 2026
>
> **摘要:** Despite strong zero-shot performance, SAM is unreliable under domain shift due to Mask-level Confidence Confusion (MCC), where a single IoU-based mask score fails to reflect pixel-wise reliability near boundaries. Motivated by the contrast between texture-biased shortcuts in neural networks and shape-centric processing in human vision, we model out-of-domain variation as appearance shifts and non-rigid deformations that jointly stress calibration. We propose Segment Anything with Robust Uncertainty-Accuracy Correlation (RUAC) for robust pixel-wise uncertainty estimation under appearance and deformation shifts. RUAC adds a lightweight uncertainty head, trains it with a collaborative style-deformation attack that jointly perturbs texture and geometry, and applies Uncertainty-Accuracy Alignment to ensure uncertainty consistently highlights erroneous pixels even under adversarial perturbations. Across 23 zero-shot domains, RUAC improves segmentation quality and yields more faithful uncertainty with stronger uncertainty-accuracy correlation. Project page: this https URL.
>
---
#### [replaced 111] Less is More: Efficient Black-box Attribution via Minimal Interpretable Subset Selection
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2504.00470](https://arxiv.org/pdf/2504.00470)**

> **作者:** Ruoyu Chen; Siyuan Liang; Jingzhi Li; Shiming Liu; Li Liu; Hua Zhang; Xiaochun Cao
>
> **摘要:** To develop a trustworthy AI system, which aim to identify the input regions that most influence the models decisions. The primary task of existing attribution methods lies in efficiently and accurately identifying the relationships among input-prediction interactions. Particularly when the input data is discrete, such as images, analyzing the relationship between inputs and outputs poses a significant challenge due to the combinatorial explosion. In this paper, we propose a novel and efficient black-box attribution mechanism, LiMA (Less input is More faithful for Attribution), which reformulates the attribution of important regions as an optimization problem for submodular subset selection. First, to accurately assess interactions, we design a submodular function that quantifies subset importance and effectively captures their impact on decision outcomes. Then, efficiently ranking input sub-regions by their importance for attribution, we improve optimization efficiency through a novel bidirectional greedy search algorithm. LiMA identifies both the most and least important samples while ensuring an optimal attribution boundary that minimizes errors. Extensive experiments on eight foundation models demonstrate that our method provides faithful interpretations with fewer regions and exhibits strong generalization, shows an average improvement of 36.3% in Insertion and 39.6% in Deletion. Our method also outperforms the naive greedy search in attribution efficiency, being 1.6 times faster. Furthermore, when explaining the reasons behind model prediction errors, the average highest confidence achieved by our method is, on average, 86.1% higher than that of state-of-the-art attribution algorithms. The code is available at this https URL.
>
---
