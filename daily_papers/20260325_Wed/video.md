# 计算机视觉 cs.CV

- **最新发布 157 篇**

- **更新 113 篇**

## 最新发布

#### [new 001] SIGMA: A Physics-Based Benchmark for Gas Chimney Understanding in Seismic Images
- **分类: cs.CV**

- **简介: 该论文属于地震图像中气体烟囱识别任务，旨在解决检测困难问题。构建了SIGMA数据集，包含标注图像和增强对，用于提升识别效果。**

- **链接: [https://arxiv.org/pdf/2603.23439](https://arxiv.org/pdf/2603.23439)**

> **作者:** Bao Truong; Quang Nguyen; Baoru Huang; Jinpei Han; Van Nguyen; Ngan Le; Minh-Tan Pham; Doan Huy Hien; Anh Nguyen
>
> **备注:** Accepted at The IEEE/CVF Conference on Computer Vision and Pattern Recognition 2026
>
> **摘要:** Seismic images reconstruct subsurface reflectivity from field recordings, guiding exploration and reservoir monitoring. Gas chimneys are vertical anomalies caused by subsurface fluid migration. Understanding these phenomena is crucial for assessing hydrocarbon potential and avoiding drilling hazards. However, accurate detection is challenging due to strong seismic attenuation and scattering. Traditional physics-based methods are computationally expensive and sensitive to model errors, while deep learning offers efficient alternatives, yet lacks labeled datasets. In this work, we introduce \textbf{SIGMA}, a new physics-based dataset for gas chimney understanding in seismic images, featuring (i) pixel-level gas-chimney mask for detection and (ii) paired degraded and ground-truth image for enhancement. We employed physics-based methods that cover a wide range of geological settings and data acquisition conditions. Comprehensive experiments demonstrate that SIGMA serves as a challenging benchmark for gas chimney interpretation and benefits general seismic understanding.
>
---
#### [new 002] Designing to Forget: Deep Semi-parametric Models for Unlearning
- **分类: cs.CV**

- **简介: 该论文研究机器遗忘任务，解决模型删除特定训练样本的问题。提出深度半参数模型（SPMs），实现高效遗忘且不影响性能。**

- **链接: [https://arxiv.org/pdf/2603.22870](https://arxiv.org/pdf/2603.22870)**

> **作者:** Amber Yijia Zheng; Yu-Shan Tai; Raymond A. Yeh
>
> **备注:** CVPR 2026
>
> **摘要:** Recent advances in machine unlearning have focused on developing algorithms to remove specific training samples from a trained model. In contrast, we observe that not all models are equally easy to unlearn. Hence, we introduce a family of deep semi-parametric models (SPMs) that exhibit non-parametric behavior during unlearning. SPMs use a fusion module that aggregates information from each training sample, enabling explicit test-time deletion of selected samples without altering model parameters. Empirically, we demonstrate that SPMs achieve competitive task performance to parametric models in image classification and generation, while being significantly more efficient for unlearning. Notably, on ImageNet classification, SPMs reduce the prediction gap relative to a retrained (oracle) baseline by $11\%$ and achieve over $10\times$ faster unlearning compared to existing approaches on parametric models. The code is available at this https URL.
>
---
#### [new 003] Gau-Occ: Geometry-Completed Gaussians for Multi-Modal 3D Occupancy Prediction
- **分类: cs.CV**

- **简介: 该论文属于3D语义占用预测任务，旨在解决多模态融合计算成本高的问题。提出Gau-Occ框架，通过语义3D高斯模型实现高效准确的占用预测。**

- **链接: [https://arxiv.org/pdf/2603.22852](https://arxiv.org/pdf/2603.22852)**

> **作者:** Chengxin Lv; Yihui Li; Hongyu Yang; YunHong Wang
>
> **摘要:** 3D semantic occupancy prediction is crucial for autonomous driving. While multi-modal fusion improves accuracy over vision-only methods, it typically relies on computationally expensive dense voxel or BEV tensors. We present Gau-Occ, a multi-modal framework that bypasses dense volumetric processing by modeling the scene as a compact collection of semantic 3D Gaussians. To ensure geometric completeness, we propose a LiDAR Completion Diffuser (LCD) that recovers missing structures from sparse LiDAR to initialize robust Gaussian anchors. Furthermore, we introduce Gaussian Anchor Fusion (GAF), which efficiently integrates multi-view image semantics via geometry-aligned 2D sampling and cross-modal alignment. By refining these compact Gaussian descriptors, Gau-Occ captures both spatial consistency and semantic discriminability. Extensive experiments across challenging benchmarks demonstrate that Gau-Occ achieves state-of-the-art performance with significant computational efficiency.
>
---
#### [new 004] Static Scene Reconstruction from Dynamic Egocentric Videos
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于3D重建任务，解决动态第一人称视频中的静态场景重建问题。通过引入掩码感知机制和分块重建策略，提升重建精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.22450](https://arxiv.org/pdf/2603.22450)**

> **作者:** Qifei Cui; Patrick Chen
>
> **摘要:** Egocentric videos present unique challenges for 3D reconstruction due to rapid camera motion and frequent dynamic interactions. State-of-the-art static reconstruction systems, such as MapAnything, often degrade in these settings, suffering from catastrophic trajectory drift and "ghost" geometry caused by moving hands. We bridge this gap by proposing a robust pipeline that adapts static reconstruction backbones to long-form egocentric video. Our approach introduces a mask-aware reconstruction mechanism that explicitly suppresses dynamic foreground in the attention layers, preventing hand artifacts from contaminating the static map. Furthermore, we employ a chunked reconstruction strategy with pose-graph stitching to ensure global consistency and eliminate long-term drift. Experiments on HD-EPIC and indoor drone datasets demonstrate that our pipeline significantly improves absolute trajectory error and yields visually clean static geometry compared to naive baselines, effectively extending the capability of foundation models to dynamic first-person scenes.
>
---
#### [new 005] Predictive Photometric Uncertainty in Gaussian Splatting for Novel View Synthesis
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决高斯点云在新视角合成中的不确定性估计问题。提出一种轻量级框架，实现像素级不确定性预测，提升场景可靠性。**

- **链接: [https://arxiv.org/pdf/2603.22786](https://arxiv.org/pdf/2603.22786)**

> **作者:** Chamuditha Jayanga Galappaththige; Thomas Gottwald; Peter Stehr; Edgar Heinert; Niko Suenderhauf; Dimity Miller; Matthias Rottmann
>
> **备注:** Project Page: this https URL
>
> **摘要:** Recent advances in 3D Gaussian Splatting have enabled impressive photorealistic novel view synthesis. However, to transition from a pure rendering engine to a reliable spatial map for autonomous agents and safety-critical applications, knowing where the representation is uncertain is as important as the rendering fidelity itself. We bridge this critical gap by introducing a lightweight, plug-and-play framework for pixel-wise, view-dependent predictive uncertainty estimation. Our post-hoc method formulates uncertainty as a Bayesian-regularized linear least-squares optimization over reconstruction residuals. This architecture-agnostic approach extracts a per-primitive uncertainty channel without modifying the underlying scene representation or degrading baseline visual fidelity. Crucially, we demonstrate that providing this actionable reliability signal successfully translates 3D Gaussian splatting into a trustworthy spatial map, further improving state-of-the-art performance across three critical downstream perception tasks: active view selection, pose-agnostic scene change detection, and pose-agnostic anomaly detection.
>
---
#### [new 006] WildWorld: A Large-Scale Dataset for Dynamic World Modeling with Actions and Explicit State toward Generative ARPG
- **分类: cs.CV**

- **简介: 该论文提出WildWorld数据集，用于动态世界建模任务，解决现有数据集动作语义不足、状态不明确的问题。**

- **链接: [https://arxiv.org/pdf/2603.23497](https://arxiv.org/pdf/2603.23497)**

> **作者:** Zhen Li; Zian Meng; Shuwei Shi; Wenshuo Peng; Yuwei Wu; Bo Zheng; Chuanhao Li; Kaipeng Zhang
>
> **摘要:** Dynamical systems theory and reinforcement learning view world evolution as latent-state dynamics driven by actions, with visual observations providing partial information about the state. Recent video world models attempt to learn this action-conditioned dynamics from data. However, existing datasets rarely match the requirement: they typically lack diverse and semantically meaningful action spaces, and actions are directly tied to visual observations rather than mediated by underlying states. As a result, actions are often entangled with pixel-level changes, making it difficult for models to learn structured world dynamics and maintain consistent evolution over long horizons. In this paper, we propose WildWorld, a large-scale action-conditioned world modeling dataset with explicit state annotations, automatically collected from a photorealistic AAA action role-playing game (Monster Hunter: Wilds). WildWorld contains over 108 million frames and features more than 450 actions, including movement, attacks, and skill casting, together with synchronized per-frame annotations of character skeletons, world states, camera poses, and depth maps. We further derive WildBench to evaluate models through Action Following and State Alignment. Extensive experiments reveal persistent challenges in modeling semantically rich actions and maintaining long-horizon state consistency, highlighting the need for state-aware video generation. The project page is this https URL.
>
---
#### [new 007] Multimodal Industrial Anomaly Detection via Geometric Prior
- **分类: cs.CV**

- **简介: 该论文属于工业异常检测任务，旨在解决2D方法难以检测的复杂几何缺陷问题。通过引入几何先验，提出GPAD网络提升检测精度。**

- **链接: [https://arxiv.org/pdf/2603.22757](https://arxiv.org/pdf/2603.22757)**

> **作者:** Min Li; Jinghui He; Gang Li; Jiachen Li; Jin Wan; Delong Han
>
> **备注:** Accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)
>
> **摘要:** The purpose of multimodal industrial anomaly detection is to detect complex geometric shape defects such as subtle surface deformations and irregular contours that are difficult to detect in 2D-based methods. However, current multimodal industrial anomaly detection lacks the effective use of crucial geometric information like surface normal vectors and 3D shape topology, resulting in low detection accuracy. In this paper, we propose a novel Geometric Prior-based Anomaly Detection network (GPAD). Firstly, we propose a point cloud expert model to perform fine-grained geometric feature extraction, employing differential normal vector computation to enhance the geometric details of the extracted features and generate geometric prior. Secondly, we propose a two-stage fusion strategy to efficiently leverage the complementarity of multimodal data as well as the geometric prior inherent in 3D points. We further propose attention fusion and anomaly regions segmentation based on geometric prior, which enhance the model's ability to perceive geometric defects. Extensive experiments show that our multimodal industrial anomaly detection model outperforms the State-of-the-art (SOTA) methods in detection accuracy on both MVTec-3D AD and Eyecandies datasets.
>
---
#### [new 008] WiFi2Cap: Semantic Action Captioning from Wi-Fi CSI via Limb-Level Semantic Alignment
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出WiFi2Cap，解决从Wi-Fi CSI生成细粒度动作描述的问题，通过多阶段框架和镜像一致性损失提升准确性。**

- **链接: [https://arxiv.org/pdf/2603.22690](https://arxiv.org/pdf/2603.22690)**

> **作者:** Tzu-Ti Wei; Chu-Yu Huang; Yu-Chee Tseng; Jen-Jee Chen
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** Privacy-preserving semantic understanding of human activities is important for indoor sensing, yet existing Wi-Fi CSI-based systems mainly focus on pose estimation or predefined action classification rather than fine-grained language generation. Mapping CSI to natural-language descriptions remains challenging because of the semantic gap between wireless signals and language and direction-sensitive ambiguities such as left/right limb confusion. We propose WiFi2Cap, a three-stage framework for generating action captions directly from Wi-Fi CSI. A vision-language teacher learns transferable supervision from synchronized video-text pairs, and a CSI student is aligned to the teacher's visual space and text embeddings. To improve direction-sensitive captioning, we introduce a Mirror-Consistency Loss that reduces mirrored-action and left-right ambiguities during cross-modal alignment. A prefix-tuned language model then generates action descriptions from CSI embeddings. We also introduce the WiFi2Cap Dataset, a synchronized CSI-RGB-sentence benchmark for semantic captioning from Wi-Fi signals. Experimental results show that WiFi2Cap consistently outperforms baseline methods on BLEU-4, METEOR, ROUGE-L, CIDEr, and SPICE, demonstrating effective privacy-friendly semantic sensing.
>
---
#### [new 009] Unleashing Spatial Reasoning in Multimodal Large Language Models via Textual Representation Guided Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态语言模型任务，旨在解决3D空间推理不足的问题。通过引入TRACE方法，生成文本化空间表示以提升空间问答准确性。**

- **链接: [https://arxiv.org/pdf/2603.23404](https://arxiv.org/pdf/2603.23404)**

> **作者:** Jiacheng Hua; Yishu Yin; Yuhang Wu; Tai Wang; Yifei Huang; Miao Liu
>
> **备注:** 26 pages, 6 figures
>
> **摘要:** Existing Multimodal Large Language Models (MLLMs) struggle with 3D spatial reasoning, as they fail to construct structured abstractions of the 3D environment depicted in video inputs. To bridge this gap, drawing inspiration from cognitive theories of allocentric spatial reasoning, we investigate how to enable MLLMs to model and reason over text-based spatial representations of video. Specifically, we introduce Textual Representation of Allocentric Context from Egocentric Video (TRACE), a prompting method that induces MLLMs to generate text-based representations of 3D environments as intermediate reasoning traces for more accurate spatial question answering. TRACE encodes meta-context, camera trajectories, and detailed object entities to support structured spatial reasoning over egocentric videos. Extensive experiments on VSI-Bench and OST-Bench demonstrate that TRACE yields notable and consistent improvements over prior prompting strategies across a diverse range of MLLM backbones, spanning different parameter scales and training schemas. We further present ablation studies to validate our design choices, along with detailed analyses that probe the bottlenecks of 3D spatial reasoning in MLLMs.
>
---
#### [new 010] Object Pose Transformer: Unifying Unseen Object Pose Estimation
- **分类: cs.CV**

- **简介: 该论文属于3D视觉中的物体位姿估计任务，旨在解决未见物体的位姿估计问题。提出统一框架Object Pose Transformer，结合绝对与相对位姿估计，提升精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.23370](https://arxiv.org/pdf/2603.23370)**

> **作者:** Weihang Li; Lorenzo Garattoni; Fabien Despinoy; Nassir Navab; Benjamin Busam
>
> **备注:** Project Page: this https URL
>
> **摘要:** Learning model-free object pose estimation for unseen instances remains a fundamental challenge in 3D vision. Existing methods typically fall into two disjoint paradigms: category-level approaches predict absolute poses in a canonical space but rely on predefined taxonomies, while relative pose methods estimate cross-view transformations but cannot recover single-view absolute pose. In this work, we propose Object Pose Transformer (\ours{}), a unified feed-forward framework that bridges these paradigms through task factorization within a single model. \ours{} jointly predicts depth, point maps, camera parameters, and normalized object coordinates (NOCS) from RGB inputs, enabling both category-level absolute SA(3) pose and unseen-object relative SE(3) pose. Our approach leverages contrastive object-centric latent embeddings for canonicalization without requiring semantic labels at inference time, and uses point maps as a camera-space representation to enable multi-view relative geometric reasoning. Through cross-frame feature interaction and shared object embeddings, our model leverages relative geometric consistency across views to improve absolute pose estimation, reducing ambiguity in single-view predictions. Furthermore, \ours{} is camera-agnostic, learning camera intrinsics on-the-fly and supporting optional depth input for metric-scale recovery, while remaining fully functional in RGB-only settings. Extensive experiments on diverse benchmarks (NOCS, HouseCat6D, Omni6DPose, Toyota-Light) demonstrate state-of-the-art performance in both absolute and relative pose estimation tasks within a single unified architecture.
>
---
#### [new 011] TimeWeaver: Age-Consistent Reference-Based Face Restoration with Identity Preservation
- **分类: cs.CV**

- **简介: 该论文属于人脸修复任务，解决跨年龄参考下保持身份和年龄一致的问题。提出TimeWeaver框架，通过分离身份与年龄条件，实现精准的年龄控制和身份保留。**

- **链接: [https://arxiv.org/pdf/2603.22701](https://arxiv.org/pdf/2603.22701)**

> **作者:** Teer Song; Yue Zhang; Yu Tian; Ziyang Wang; Xianlin Zhang; Guixuan Zhang; Xuan Liu; Xueming Li; Yasen Zhang
>
> **备注:** This is an improved version based on arXiv:2603.18645
>
> **摘要:** Recent progress in face restoration has shifted from visual fidelity to identity fidelity, driving a transition from reference-free to reference-based paradigms that condition restoration on reference images of the same person. However, these methods assume the reference and degraded input are age-aligned. When only cross-age references are available, as in historical restoration or missing-person retrieval, they fail to maintain age fidelity. To address this limitation, we propose TimeWeaver, the first reference-based face restoration framework supporting cross-age references. Given arbitrary reference images and a target-age prompt, TimeWeaver produces restorations with both identity fidelity and age consistency. Specifically, we decouple identity and age conditioning across training and inference. During training, the model learns an age-robust identity representation by fusing a global identity embedding with age-suppressed facial tokens via a transformer-based ID-Fusion module. During inference, two training-free techniques, Age-Aware Gradient Guidance and Token-Targeted Attention Boost, steer sampling toward desired age semantics, enabling precise adherence to the target-age prompt. Extensive experiments show that TimeWeaver surpasses existing methods in visual quality, identity preservation, and age consistency.
>
---
#### [new 012] CAM3R: Camera-Agnostic Model for 3D Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决非标准相机模型下几何退化问题。提出CAM3R模型，无需标定即可处理广角图像，提升重建精度。**

- **链接: [https://arxiv.org/pdf/2603.22631](https://arxiv.org/pdf/2603.22631)**

> **作者:** Namitha Guruprasad; Abhay Yadav; Cheng Peng; Rama Chellappa
>
> **摘要:** Recovering dense 3D geometry from unposed images remains a foundational challenge in computer vision. Current state-of-the-art models are predominantly trained on perspective datasets, which implicitly constrains them to a standard pinhole camera geometry. As a result, these models suffer from significant geometric degradation when applied to wide-angle imagery captured via non-rectilinear optics, such as fisheye or panoramic sensors. To address this, we present CAM3R, a Camera-Agnostic, feed-forward Model for 3D Reconstruction capable of processing images from wide-angle camera models without prior calibration. Our framework consists of a two-view network which is bifurcated into a Ray Module (RM) to estimate per-pixel ray directions and a Cross-view Module (CVM) to infer radial distance with confidence maps, pointmaps, and relative poses. To unify these pairwise predictions into a consistent 3D scene, we introduce a Ray-Aware Global Alignment framework for pose refinement and scale optimization while strictly preserving the predicted local geometry. Extensive experiments on various camera model datasets, including panorama, fisheye and pinhole imagery, demonstrate that CAM3R establishes a new state-of-the-art in pose estimation and reconstruction.
>
---
#### [new 013] Mamba-driven MRI-to-CT Synthesis for MRI-only Radiotherapy Planning
- **分类: cs.CV**

- **简介: 该论文属于MRI到CT图像合成任务，旨在解决MRI-only放疗中图像模态转换问题，采用Mamba架构提升合成效果与效率。**

- **链接: [https://arxiv.org/pdf/2603.23295](https://arxiv.org/pdf/2603.23295)**

> **作者:** Konstantinos Barmpounakis; Theodoros P. Vagenas; Maria Vakalopoulou; George K. Matsopoulos
>
> **摘要:** Radiotherapy workflows for oncological patients increasingly rely on multi-modal medical imaging, commonly involving both Magnetic Resonance Imaging (MRI) and Computed Tomography (CT). MRI-only treatment planning has emerged as an attractive alternative, as it reduces patient exposure to ionizing radiation and avoids errors introduced by inter-modality registration. While nnU-Net-based frameworks are predominantly used for MRI-to-CT synthesis, we explore Mamba-based architectures for this task, aiming to showcase the advantages of state-space modeling for cross-modality translation compared to standard convolutional neural networks. Specifically, we adapt both the U-Mamba and the SegMamba architecture, originally proposed for segmentation, to perform cross-modality image generation. Our 3D Mamba architecture effectively captures complex volumetric features and long-range dependencies, thus allowing accurate CT synthesis while maintaining fast inference times. Experiments were conducted on a subset of SynthRAD2025 dataset, comprising registered single-channel MRI-CT volume pairs across three anatomical regions. Quantitative evaluation is performed via a combination of image similarity metrics computed in Hounsefield Units (HU) and segmentation-based metrics obtained from TotalSegmentator to ensure geometric consistency is preserved. The findings pave the way for the integration of state-space models into radiotherapy workflows.
>
---
#### [new 014] Q-Tacit: Image Quality Assessment via Latent Visual Reasoning
- **分类: cs.CV**

- **简介: 该论文属于图像质量评估任务，旨在解决视觉信息在语言空间中表达不足的问题。提出Q-Tacit方法，在潜在质量空间中进行视觉推理，提升评估效果。**

- **链接: [https://arxiv.org/pdf/2603.22641](https://arxiv.org/pdf/2603.22641)**

> **作者:** Yuxuan Jiang; Yixuan Li; Hanwei Zhu; Siyue Teng; Fan Zhang; David Bull
>
> **摘要:** Vision-Language Model (VLM)-based image quality assessment (IQA) has been significantly advanced by incorporating Chain-of-Thought (CoT) reasoning. Recent work has refined image quality reasoning by applying reinforcement learning (RL) and leveraging active visual tools. However, such strategies are typically language-centric, with visual information being treated as static preconditions. Quality-related visual cues often cannot be abstracted into text in extenso due to the gap between discrete textual tokens and quality perception space, which in turn restricts the reasoning effectiveness for visually intensive IQA tasks. In this paper, we revisit this by asking the question, "Is natural language the ideal space for quality reasoning?" and, as a consequence, we propose Q-Tacit, a new paradigm that elicits VLMs to reason beyond natural language in the latent quality space. Our approach follows a synergistic two-stage process: (i) injecting structural visual quality priors into the latent space, and (ii) calibrating latent reasoning trajectories to improve quality assessment ability. Extensive experiments demonstrate that Q-Tacit can effectively perform quality reasoning with significantly fewer tokens than previous reasoning-based methods, while achieving strong overall performance. This paper validates the proposition that language is not the only compact representation suitable for visual quality, opening possibilities for further exploration of effective latent reasoning paradigms for IQA. Source code will be released to support future research.
>
---
#### [new 015] From Instructions to Assistance: a Dataset Aligning Instruction Manuals with Assembly Videos for Evaluating Multimodal LLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态语言模型评估任务，旨在解决模型在技术任务中的辅助能力问题。通过构建M2AD数据集，评估模型对步骤推理、进度跟踪和手册引用的能力。**

- **链接: [https://arxiv.org/pdf/2603.22321](https://arxiv.org/pdf/2603.22321)**

> **作者:** Federico Toschi; Nicolò Brunello; Andrea Sassella; Vincenzo Scotti; Mark James Carman
>
> **摘要:** The recent advancements introduced by Large Language Models (LLMs) have transformed how Artificial Intelligence (AI) can support complex, real world tasks, pushing research outside the text boundaries towards multi modal contexts and leading to Multimodal Large Language Models (MLMs). Given the current adoption of LLM based assistants in solving technical or domain specific problems, the natural continuation of this trend is to extend the input domains of these assistants exploiting MLMs. Ideally, these MLMs should be used as real time assistants in procedural tasks, hopefully integrating a view of the environment where the user being assisted is, or even better sharing the same point of view via Virtual Reality (VR) or Augmented Reality (AR) supports, to reason over the same scenario the user is experiencing. With this work, we aim at evaluating the quality of currently openly available MLMs to provide this kind of assistance on technical tasks. To this end, we annotated a data set of furniture assembly with step by step labels and manual references: the Manual to Action Dataset (M2AD). We used this dataset to assess (1) to which extent the reasoning abilities of MLMs can be used to reduce the need for detailed labelling, allowing for more efficient, cost effective annotation practices, (2) whether MLMs are able to track the progression of assembly steps (3) and whether MLMs can refer correctly to the instruction manual pages. Our results showed that while some models understand procedural sequences, their performance is limited by architectural and hardware constraints, highlighting the need for multi image and interleaved text image reasoning.
>
---
#### [new 016] SOUPLE: Enhancing Audio-Visual Localization and Segmentation with Learnable Prompt Contexts
- **分类: cs.CV**

- **简介: 该论文提出SOUPLE，用于音频-视觉定位与分割任务，解决CLIP模型在该领域表现不佳的问题，通过引入可学习的上下文标记增强语义关联。**

- **链接: [https://arxiv.org/pdf/2603.22732](https://arxiv.org/pdf/2603.22732)**

> **作者:** Khanh Binh Nguyen; Chae Jung Park
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Large-scale pre-trained image-text models exhibit robust multimodal representations, yet applying the Contrastive Language-Image Pre-training (CLIP) model to audio-visual localization remains challenging. Replacing the classification token ([CLS]) with an audio-embedded token ([V_A]) struggles to capture semantic cues, and the prompt "a photo of a [V_A]" fails to establish meaningful connections between audio embeddings and context tokens. To address these issues, we propose Sound-aware Prompt Learning (SOUPLE), which replaces fixed prompts with learnable context tokens. These tokens incorporate visual features to generate conditional context for a mask decoder, effectively bridging semantic correspondence between audio and visual inputs. Experiments on VGGSound, SoundNet, and AVSBench demonstrate that SOUPLE improves localization and segmentation performance.
>
---
#### [new 017] Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference
- **分类: cs.CV**

- **简介: 该论文属于空间转录组预测任务，旨在解决高成本限制大规模应用的问题。通过多模态异构图模型和对比学习，提升跨切片基因表达预测精度。**

- **链接: [https://arxiv.org/pdf/2603.22821](https://arxiv.org/pdf/2603.22821)**

> **作者:** Zhiceng Shi; Changmiao Wang; Jun Wan; Wenwen Min
>
> **备注:** Accepted by CVPR-2026
>
> **摘要:** While spatial transcriptomics (ST) has advanced our understanding of gene expression in tissue context, its high experimental cost limits its large-scale application. Predicting ST from pathology images is a promising, cost-effective alternative, but existing methods struggle to capture complex cross-slide spatial relationships. To address the challenge, we propose SpaHGC, a multi-modal heterogeneous graph-based model that captures both intra-slice and inter-slice spot-spot relationships from histology images. It integrates local spatial context within the target slide and cross-slide similarities computed from image embeddings extracted by a pathology foundation model. These embeddings enable inter-slice knowledge transfer, and SpaHGC further incorporates Masked Graph Contrastive Learning to enhance feature representation and transfer spatial gene expression knowledge from reference to target slides, enabling it to model complex spatial dependencies and significantly improve prediction accuracy. We conducted comprehensive benchmarking on seven matched histology-ST datasets from different platforms, tissues, and cancer subtypes. The results demonstrate that SpaHGC significantly outperforms the existing nine state-of-the-art methods across all evaluation metrics. Additionally, the predictions are significantly enriched in multiple cancer-related pathways, thereby highlighting its strong biological relevance and application potential.
>
---
#### [new 018] Generalized multi-object classification and tracking with sparse feature resonator networks
- **分类: cs.CV**

- **简介: 该论文属于视觉场景理解任务，解决对象识别与跟踪问题。通过稀疏特征共振网络，同时获取不变性和等变性信息，实现精准定位与多对象跟踪。**

- **链接: [https://arxiv.org/pdf/2603.22539](https://arxiv.org/pdf/2603.22539)**

> **作者:** Lazar Supic; Alec Mullen; E. Paxon Frady
>
> **备注:** 6 pages, 2 figures, NICE 2026
>
> **摘要:** In visual scene understanding tasks, it is essential to capture both invariant and equivariant structure. While neural networks are frequently trained to achieve invariance to transformations such as translation, this often comes at the cost of losing access to equivariant information - e.g., the precise location of an object. Moreover, invariance is not naturally guaranteed through supervised learning alone, and many architectures generalize poorly to input transformations not encountered during training. Here, we take an approach based on analysis-by-synthesis and factoring using resonator networks. A generative model describes the construction of simple scenes containing MNIST digits and their transformations, like color and position. The resonator network inverts the generative model, and provides both invariant and equivariant information about particular objects. Sparse features learned from training data act as a basis set to provide flexibility in representing variable shapes of objects, allowing the resonator network to handle previously unseen digit shapes from the test set. The modular structure provides a shape module which contains information about the object shape with translation factored out, allowing a simple classifier to operate on centered digits. The classification layer is trained solely on centered data, requiring much less training data, and the network as a whole can identify objects with arbitrary translations without data augmentation. The natural attention-like mechanism of the resonator network also allows for analysis of scenes with multiple objects, where the network dynamics selects and centers only one object at a time. Further, the specific position information of a particular object can be extracted from the translation module, and we show that the resonator can be designed to track multiple moving objects with precision of a few pixels.
>
---
#### [new 019] Knot-10:A Tightness-Stratified Benchmark for Real-World Knot Classification with Topological Difficulty Analysis
- **分类: cs.CV**

- **简介: 该论文属于物理绳结分类任务，旨在解决外观线索被抑制下的细粒度识别问题。构建了Knots-10数据集，分析拓扑结构与模型表现的关系，并提出TACA正则化提升拓扑对齐。**

- **链接: [https://arxiv.org/pdf/2603.23286](https://arxiv.org/pdf/2603.23286)**

> **作者:** Shiheng Nie; Yunguang Yue
>
> **备注:** 48 pages, 12 figures, 10 supplementary sections
>
> **摘要:** Physical knot classification is a fine-grained visual classification (FGVC) scenario in which appearance cues are deliberately suppressed: different classes share the same rope material, color, and background, and class identity resides primarily in crossing structure. We introduce the Knots-10 benchmark, comprising 1,440 images with a deployment-oriented split that trains on loosely tied knots and tests on tightly dressed ones. Swin-T and TransFG both average 97.2% accuracy; PMG scores 94.5%, consistent with the hypothesis that jigsaw shuffling disrupts crossing continuity. McNemar tests cannot separate four of the five general-purpose backbones, so small ranking margins should be interpreted with caution. A Mantel permutation test shows that topological distance significantly correlates with confusion patterns in three of the five models (p < 0.01). We propose TACA regularization, which improves embedding-topology alignment from rho=0.46 to rho=0.65 without improving classification accuracy; a random-distance ablation yields comparable alignment, indicating the benefit is likely driven by generic regularization. A pilot cross-domain test with 100 phone photographs reveals a 58-69 percentage-point accuracy drop, exposing rope appearance bias as the dominant failure mode.
>
---
#### [new 020] TETO: Tracking Events with Teacher Observation for Motion Estimation and Frame Interpolation
- **分类: cs.CV**

- **简介: 该论文提出TETO框架，解决事件相机运动估计与帧插值问题。通过知识蒸馏和数据优化，用少量真实数据训练模型，提升运动估计精度，从而改善帧插值效果。**

- **链接: [https://arxiv.org/pdf/2603.23487](https://arxiv.org/pdf/2603.23487)**

> **作者:** Jini Yang; Eunbeen Hong; Soowon Son; Hyunkoo Lee; Sunghwan Hong; Sunok Kim; Seungryong Kim
>
> **摘要:** Event cameras capture per-pixel brightness changes with microsecond resolution, offering continuous motion information lost between RGB frames. However, existing event-based motion estimators depend on large-scale synthetic data that often suffers from a significant sim-to-real gap. We propose TETO (Tracking Events with Teacher Observation), a teacher-student framework that learns event motion estimation from only $\sim$25 minutes of unannotated real-world recordings through knowledge distillation from a pretrained RGB tracker. Our motion-aware data curation and query sampling strategy maximizes learning from limited data by disentangling object motion from dominant ego-motion. The resulting estimator jointly predicts point trajectories and dense optical flow, which we leverage as explicit motion priors to condition a pretrained video diffusion transformer for frame interpolation. We achieve state-of-the-art point tracking on EVIMO2 and optical flow on DSEC using orders of magnitude less training data, and demonstrate that accurate motion estimation translates directly to superior frame interpolation quality on BS-ERGB and HQ-EVFI.
>
---
#### [new 021] WaveSFNet: A Wavelet-Based Codec and Spatial--Frequency Dual-Domain Gating Network for Spatiotemporal Prediction
- **分类: cs.CV**

- **简介: 该论文属于时空预测任务，旨在解决长距离动态建模与高频细节保留的难题。提出WaveSFNet，结合小波编码器和时空门控融合机制，提升预测精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.23284](https://arxiv.org/pdf/2603.23284)**

> **作者:** Xinyong Cai; Runming Xie; Hu Chen; Yuankai Wu
>
> **备注:** Accepted to IJCNN 2026
>
> **摘要:** Spatiotemporal predictive learning aims to forecast future frames from historical observations in an unsupervised manner, and is critical to a wide range of applications. The key challenge is to model long-range dynamics while preserving high-frequency details for sharp multi-step predictions. Existing efficient recurrent-free frameworks typically rely on strided convolutions or pooling for sampling, which tends to discard textures and boundaries, while purely spatial operators often struggle to balance local interactions with global propagation. To address these issues, we propose WaveSFNet, an efficient framework that unifies a wavelet-based codec with a spatial--frequency dual-domain gated spatiotemporal translator. The wavelet-based codec preserves high-frequency subband cues during downsampling and reconstruction. Meanwhile, the translator first injects adjacent-frame differences to explicitly enhance dynamic information, and then performs dual-domain gated fusion between large-kernel spatial local modeling and frequency-domain global modulation, together with gated channel interaction for cross-channel feature exchange. Extensive experiments demonstrate that WaveSFNet achieves competitive prediction accuracy on Moving MNIST, TaxiBJ, and WeatherBench, while maintaining low computational complexity. Our code is available at this https URL.
>
---
#### [new 022] Conformal Cross-Modal Active Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉-语言多模态任务，旨在提升数据效率。解决现有主动学习方法未充分利用多模态信息的问题，提出CCMA框架，结合语义和多样性选择策略，提高模型性能。**

- **链接: [https://arxiv.org/pdf/2603.23159](https://arxiv.org/pdf/2603.23159)**

> **作者:** Huy Hoang Nguyen; Cédric Jung; Shirin Salehi; Tobias Glück; Anke Schmeink; Andreas Kugi
>
> **备注:** 20 pages, 14 figures
>
> **摘要:** Foundation models for vision have transformed visual recognition with powerful pretrained representations and strong zero-shot capabilities, yet their potential for data-efficient learning remains largely untapped. Active Learning (AL) aims to minimize annotation costs by strategically selecting the most informative samples for labeling, but existing methods largely overlook the rich multimodal knowledge embedded in modern vision-language models (VLMs). We introduce Conformal Cross-Modal Acquisition (CCMA), a novel AL framework that bridges vision and language modalities through a teacher-student architecture. CCMA employs a pretrained VLM as a teacher to provide semantically grounded uncertainty estimates, conformally calibrated to guide sample selection for a vision-only student model. By integrating multimodal conformal scoring with diversity-aware selection strategies, CCMA achieves superior data efficiency across multiple benchmarks. Our approach consistently outperforms state-of-the-art AL baselines, demonstrating clear advantages over methods relying solely on uncertainty or diversity metrics.
>
---
#### [new 023] FullCircle: Effortless 3D Reconstruction from Casual 360$^\circ$ Captures
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决从普通360°拍摄中实现可靠重建的问题。提出无需特殊协议的流水线，提升重建鲁棒性与效果。**

- **链接: [https://arxiv.org/pdf/2603.22572](https://arxiv.org/pdf/2603.22572)**

> **作者:** Yalda Foroutan; Ipek Oztas; Daniel Rebain; Aysegul Dundar; Kwang Moo Yi; Lily Goli; Andrea Tagliasacchi
>
> **摘要:** Radiance fields have emerged as powerful tools for 3D scene reconstruction. However, casual capture remains challenging due to the narrow field of view of perspective cameras, which limits viewpoint coverage and feature correspondences necessary for reliable camera calibration and reconstruction. While commercially available 360$^\circ$ cameras offer significantly broader coverage than perspective cameras for the same capture effort, existing 360$^\circ$ reconstruction methods require special capture protocols and pre-processing steps that undermine the promise of radiance fields: effortless workflows to capture and reconstruct 3D scenes. We propose a practical pipeline for reconstructing 3D scenes directly from raw 360$^\circ$ camera captures. We require no special capture protocols or pre-processing, and exhibit robustness to a prevalent source of reconstruction errors: the human operator that is visible in all 360$^\circ$ imagery. To facilitate evaluation, we introduce a multi-tiered dataset of scenes captured as raw dual-fisheye images, establishing a benchmark for robust casual 360$^\circ$ reconstruction. Our method significantly outperforms not only vanilla 3DGS for 360$^\circ$ cameras but also robust perspective baselines when perspective cameras are simulated from the same capture, demonstrating the advantages of 360$^\circ$ capture for casual reconstruction. Additional results are available at: this https URL
>
---
#### [new 024] SLARM: Streaming and Language-Aligned Reconstruction Model for Dynamic Scenes
- **分类: cs.CV**

- **简介: 该论文提出SLARM，用于动态场景重建任务，解决动态场景语义理解与实时流处理问题。通过高阶运动建模和语义对齐，提升重建精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.22893](https://arxiv.org/pdf/2603.22893)**

> **作者:** Zhicheng Qiu; Jiarui Meng; Tong-an Luo; Yican Huang; Xuan Feng; Xuanfu Li; ZHan Xu
>
> **摘要:** We propose SLARM, a feed-forward model that unifies dynamic scene reconstruction, semantic understanding, and real-time streaming inference. SLARM captures complex, non-uniform motion through higher-order motion modeling, trained solely on differentiable renderings without any flow supervision. Besides, SLARM distills semantic features from LSeg to obtain language-aligned representations. This design enables semantic querying via natural language, and the tight coupling between semantics and geometry further enhances the accuracy and robustness of dynamic reconstruction. Moreover, SLARM processes image sequences using window-based causal attention, achieving stable, low-latency streaming inference without accumulating memory cost. Within this unified framework, SLARM achieves state-of-the-art results in dynamic estimation, rendering quality, and scene parsing, improving motion accuracy by 21%, reconstruction PSNR by 1.6 dB, and segmentation mIoU by 20% over existing methods.
>
---
#### [new 025] Large-Scale Avalanche Mapping from SAR Images with Deep Learning-based Change Detection
- **分类: cs.CV**

- **简介: 该论文属于遥感变化检测任务，旨在通过SAR图像实现大规模雪崩自动测绘。通过深度学习方法，提升雪崩区域的识别精度与覆盖率。**

- **链接: [https://arxiv.org/pdf/2603.22658](https://arxiv.org/pdf/2603.22658)**

> **作者:** Mattia Gatti; Alberto Mariani; Ignazio Gallo; Fabiano Monti
>
> **摘要:** Accurate change detection from satellite imagery is essential for monitoring rapid mass-movement hazards such as snow avalanches, which increasingly threaten human life, infrastructure, and ecosystems due to their rising frequency and intensity. This study presents a systematic investigation of large-scale avalanche mapping through bi-temporal change detection using Sentinel-1 synthetic aperture radar (SAR) imagery. Extensive experiments across multiple alpine ecoregions with manually validated avalanche inventories show that treating the task as a unimodal change detection problem, relying solely on pre- and post-event SAR images, achieves the most consistent performance. The proposed end-to-end pipeline achieves an F1-score of 0.8061 in a conservative (F1-optimized) configuration and attains an F2-score of 0.8414 with 80.36% avalanche-polygon hit rate under a less conservative, recall-oriented (F2-optimized) tuning. These results highlight the trade-off between precision and completeness and demonstrate how threshold adjustment can improve the detection of smaller or marginal avalanches. The release of the annotated multi-region dataset establishes a reproducible benchmark for SAR-based avalanche mapping.
>
---
#### [new 026] Cog3DMap: Multi-View Vision-Language Reasoning with 3D Cognitive Maps
- **分类: cs.CV**

- **简介: 该论文属于多模态任务，旨在解决MLLM在多视角图像中缺乏几何定位的问题。提出Cog3DMap框架，构建带有语义和几何信息的3D记忆，提升空间推理能力。**

- **链接: [https://arxiv.org/pdf/2603.23023](https://arxiv.org/pdf/2603.23023)**

> **作者:** Chanyoung Gwak; Yoonwoo Jeong; Byungwoo Jeon; Hyunseok Lee; Jinwoo Shin; Minsu Cho
>
> **备注:** Project Page: this https URL
>
> **摘要:** Precise spatial understanding from multi-view images remains a fundamental challenge for Multimodal Large Language Models (MLLMs), as their visual representations are predominantly semantic and lack explicit geometric grounding. While existing approaches augment visual tokens with geometric cues from visual geometry models, their MLLM is still required to implicitly infer the underlying 3D structure of the scene from these augmented tokens, limiting its spatial reasoning capability. To address this issue, we introduce Cog3DMap, a framework that recurrently constructs an explicit 3D memory from multi-view images, where each token is grounded in 3D space and possesses both semantic and geometric information. By feeding these tokens into the MLLM, our framework enables direct reasoning over a spatially structured 3D map, achieving state-of-the-art performance on various spatial reasoning benchmarks. Code will be made publicly available.
>
---
#### [new 027] Spatially-Aware Evaluation Framework for Aerial LiDAR Point Cloud Semantic Segmentation: Distance-Based Metrics on Challenging Regions
- **分类: cs.CV**

- **简介: 该论文属于点云语义分割任务，针对传统评估指标在航空LiDAR数据中的不足，提出基于距离的度量和聚焦难例的评估方法，以更准确反映模型在复杂区域的性能。**

- **链接: [https://arxiv.org/pdf/2603.22420](https://arxiv.org/pdf/2603.22420)**

> **作者:** Alex Salvatierra; José Antonio Sanz; Christian Gutiérrez; Mikel Galar
>
> **备注:** 11 pages, 1 figure
>
> **摘要:** Semantic segmentation metrics for 3D point clouds, such as mean Intersection over Union (mIoU) and Overall Accuracy (OA), present two key limitations in the context of aerial LiDAR data. First, they treat all misclassifications equally regardless of their spatial context, overlooking cases where the geometric severity of errors directly impacts the quality of derived geospatial products such as Digital Terrain Models. Second, they are often dominated by the large proportion of easily classified points, which can mask meaningful differences between models and under-represent performance in challenging regions. To address these limitations, we propose a novel evaluation framework for comparing semantic segmentation models through two complementary approaches. First, we introduce distance-based metrics that account for the spatial deviation between each misclassified point and the nearest ground-truth point of the predicted class, capturing the geometric severity of errors. Second, we propose a focused evaluation on a common subset of hard points, defined as the points misclassified by at least one of the evaluated models, thereby reducing the bias introduced by easily classified points and better revealing differences in model performance in challenging regions. We validate our framework by comparing three state-of-the-art deep learning models on three aerial LiDAR datasets. Results demonstrate that the proposed metrics provide complementary information to traditional measures, revealing spatial error patterns that are critical for Earth Observation applications but invisible to conventional evaluation approaches. The proposed framework enables more informed model selection for scenarios where spatial consistency is critical.
>
---
#### [new 028] Rethinking Token-Level Policy Optimization for Multimodal Chain-of-Thought
- **分类: cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决CoT推理中视觉与推理结合不充分的问题。通过分析token级动态，提出PEPO方法提升推理效果。**

- **链接: [https://arxiv.org/pdf/2603.22847](https://arxiv.org/pdf/2603.22847)**

> **作者:** Yunheng Li; Hangyi Kuang; Hengrui Zhang; Jiangxia Cao; Zhaojie Liu; Qibin Hou; Ming-Ming Cheng
>
> **摘要:** Multimodal Chain-of-Thought (CoT) reasoning requires large vision-language models to construct reasoning trajectories that interleave perceptual grounding with multi-step inference. However, existing Reinforcement Learning with Verifiable Rewards (RLVR) methods typically optimize reasoning at a coarse granularity, treating CoT uniformly without distinguishing their varying degrees of visual grounding. In this work, we conduct a token-level analysis of multimodal reasoning trajectories and show that successful reasoning is characterized by structured token dynamics reflecting both perceptual grounding and exploratory inference. Building upon this analysis, we propose Perception-Exploration Policy Optimization (PEPO), which derives a perception prior from hidden state similarity and integrates it with token entropy through a smooth gating mechanism to produce token-level advantages. PEPO integrates seamlessly with existing RLVR frameworks such as GRPO and DAPO, requiring neither additional supervision nor auxiliary branches. Extensive experiments across diverse multimodal benchmarks demonstrate consistent and robust improvements over strong RL baselines, spanning geometry reasoning, visual grounding, visual puzzle solving, and few-shot classification, while maintaining stable training dynamics. Code: this https URL
>
---
#### [new 029] AgentRVOS: Reasoning over Object Tracks for Zero-Shot Referring Video Object Segmentation
- **分类: cs.CV**

- **简介: 该论文属于视频目标分割任务，解决零样本引用视频对象分割问题。提出AgentRVOS方法，结合SAM3与MLLM，提升分割精度与时空覆盖。**

- **链接: [https://arxiv.org/pdf/2603.23489](https://arxiv.org/pdf/2603.23489)**

> **作者:** Woojeong Jin; Jaeho Lee; Heeseong Shin; Seungho Jang; Junhwan Heo; Seungryong Kim
>
> **摘要:** Referring Video Object Segmentation (RVOS) aims to segment a target object throughout a video given a natural language query. Training-free methods for this task follow a common pipeline: a MLLM selects keyframes, grounds the referred object within those frames, and a video segmentation model propagates the results. While intuitive, this design asks the MLLM to make temporal decisions before any object-level evidence is available, limiting both reasoning quality and spatio-temporal coverage. To overcome this, we propose AgentRVOS, a training-free agentic pipeline built on the complementary strengths of SAM3 and a MLLM. Given a concept derived from the query, SAM3 provides reliable perception over the full spatio-temporal extent through generated mask tracks. The MLLM then identifies the target through query-grounded reasoning over this object-level evidence, iteratively pruning guided by SAM3's temporal existence information. Extensive experiments show that AgentRVOS achieves state-of-the-art performance among training-free methods across multiple benchmarks, with consistent results across diverse MLLM backbones. Our project page is available at: this https URL.
>
---
#### [new 030] PiCo: Active Manifold Canonicalization for Robust Robotic Visual Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于机器人视觉异常检测任务，旨在解决复杂环境下物体姿态变化和环境干扰导致的检测不稳定问题。提出PiCo框架，通过主动对齐和多阶段去噪提升检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.23122](https://arxiv.org/pdf/2603.23122)**

> **作者:** Teng Yan; Binkai Liu; Shuai Liu; Yue Yu; Bingzhuo Zhong
>
> **备注:** 16 pages. Submitted to the European Conference on Computer Vision (ECCV) 2026
>
> **摘要:** Industrial deployment of robotic visual anomaly detection (VAD) is fundamentally constrained by passive perception under diverse 6-DoF pose configurations and unstable operating conditions such as illumination changes and shadows, where intrinsic semantic anomalies and physical disturbances coexist and interact. To overcome these limitations, a paradigm shift from passive feature learning to Active Canonicalization is proposed. PiCo (Pose-in-Condition Canonicalization) is introduced as a unified framework that actively projects observations onto a condition-invariant canonical manifold. PiCo operates through a cascaded mechanism. The first stage, Active Physical Canonicalization, enables a robotic agent to reorient objects in order to reduce geometric uncertainty at its source. The second stage, Neural Latent Canonicalization, adopts a three-stage denoising hierarchy consisting of photometric processing at the input level, latent refinement at the feature level, and contextual reasoning at the semantic level, progressively eliminating nuisance factors across representational scales. Extensive evaluations on the large-scale M2AD benchmark demonstrate the superiority of this paradigm. PiCo achieves a state-of-the-art 93.7% O-AUROC, representing a 3.7% improvement over prior methods in static settings, and attains 98.5% accuracy in active closed-loop scenarios. These results demonstrate that active manifold canonicalization is critical for robust embodied perception.
>
---
#### [new 031] Drop-In Perceptual Optimization for 3D Gaussian Splatting
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 该论文属于3D高斯点云渲染任务，旨在解决渲染模糊问题。通过研究感知优化策略，提出WD-R损失函数，显著提升纹理细节和视觉质量。**

- **链接: [https://arxiv.org/pdf/2603.23297](https://arxiv.org/pdf/2603.23297)**

> **作者:** Ezgi Ozyilkan; Zhiqi Chen; Oren Rippel; Jona Ballé; Kedar Tatwawadi
>
> **备注:** Project page: this https URL
>
> **摘要:** Despite their output being ultimately consumed by human viewers, 3D Gaussian Splatting (3DGS) methods often rely on ad-hoc combinations of pixel-level losses, resulting in blurry renderings. To address this, we systematically explore perceptual optimization strategies for 3DGS by searching over a diverse set of distortion losses. We conduct the first-of-its-kind large-scale human subjective study on 3DGS, involving 39,320 pairwise ratings across several datasets and 3DGS frameworks. A regularized version of Wasserstein Distortion, which we call WD-R, emerges as the clear winner, excelling at recovering fine textures without incurring a higher splat count. WD-R is preferred by raters more than $2.3\times$ over the original 3DGS loss, and $1.5\times$ over current best method Perceptual-GS. WD-R also consistently achieves state-of-the-art LPIPS, DISTS, and FID scores across various datasets, and generalizes across recent frameworks, such as Mip-Splatting and Scaffold-GS, where replacing the original loss with WD-R consistently enhances perceptual quality within a similar resource budget (number of splats for Mip-Splatting, model size for Scaffold-GS), and leads to reconstructions being preferred by human raters $1.8\times$ and $3.6\times$, respectively. We also find that this carries over to the task of 3DGS scene compression, with $\approx 50\%$ bitrate savings for comparable perceptual metric performance.
>
---
#### [new 032] To Agree or To Be Right? The Grounding-Sycophancy Tradeoff in Medical Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医疗视觉问答任务，研究VLMs在幻觉和盲从间的权衡问题。通过提出新指标评估模型安全性，发现现有模型难以同时具备准确性和抗压性。**

- **链接: [https://arxiv.org/pdf/2603.22623](https://arxiv.org/pdf/2603.22623)**

> **作者:** OFM Riaz Rahman Aranya; Kevin Desai
>
> **摘要:** Vision-language models (VLMs) adapted to the medical domain have shown strong performance on visual question answering benchmarks, yet their robustness against two critical failure modes, hallucination and sycophancy, remains poorly understood, particularly in combination. We evaluate six VLMs (three general-purpose, three medical-specialist) on three medical VQA datasets and uncover a grounding-sycophancy tradeoff: models with the lowest hallucination propensity are the most sycophantic, while the most pressure-resistant model hallucinates more than all medical-specialist models. To characterize this tradeoff, we propose three metrics: L-VASE, a logit-space reformulation of VASE that avoids its double-normalization; CCS, a confidence-calibrated sycophancy score that penalizes high-confidence capitulation; and Clinical Safety Index (CSI), a unified safety index that combines grounding, autonomy, and calibration via a geometric mean. Across 1,151 test cases, no model achieves a CSI above 0.35, indicating that none of the evaluated 7-8B parameter VLMs is simultaneously well-grounded and robust to social pressure. Our findings suggest that joint evaluation of both properties is necessary before these models can be considered for clinical use. Code is available at this https URL
>
---
#### [new 033] ARGENT: Adaptive Hierarchical Image-Text Representations
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出ARGENT模型，解决超球几何在视觉-语言模型中层次结构表达不足的问题，通过自适应损失和角度评估协议提升性能。**

- **链接: [https://arxiv.org/pdf/2603.23311](https://arxiv.org/pdf/2603.23311)**

> **作者:** Chuong Huynh; Hossein Souri; Abhinav Kumar; Vitali Petsiuk; Deen Dayal Mohan; Suren Kumar
>
> **摘要:** Large-scale Vision-Language Models (VLMs) such as CLIP learn powerful semantic representations but operate in Euclidean space, which fails to capture the inherent hierarchical structure of visual and linguistic concepts. Hyperbolic geometry, with its exponential volume growth, offers a principled alternative for embedding such hierarchies with low distortion. However, existing hyperbolic VLMs use entailment losses that are unstable: as parent embeddings contract toward the origin, their entailment cones widen toward a half-space, causing catastrophic cone collapse that destroys the intended hierarchy. Additionally, hierarchical evaluation of these models remains unreliable, being largely retrieval-based and correlation-based metrics and prone to taxonomy dependence and ambiguous negatives. To address these limitations, we propose an adaptive entailment loss paired with a norm regularizer that prevents cone collapse without heuristic aperture clipping. We further introduce an angle-based probabilistic entailment protocol (PEP) for evaluating hierarchical understanding, scored with AUC-ROC and Average Precision. This paper introduces a stronger hyperbolic VLM baseline ARGENT, Adaptive hieRarchical imaGe-tExt represeNTation. ARGENT improves the SOTA hyperbolic VLM by 0.7, 1.1, and 0.8 absolute points on image classification, text-to-image retrieval, and proposed hierarchical metrics, respectively.
>
---
#### [new 034] UniGRPO: Unified Policy Optimization for Reasoning-Driven Visual Generation
- **分类: cs.CV**

- **简介: 该论文属于视觉生成任务，旨在解决多模态生成中的策略优化问题。提出UniGRPO框架，统一优化文本与图像生成策略，提升生成质量与可扩展性。**

- **链接: [https://arxiv.org/pdf/2603.23500](https://arxiv.org/pdf/2603.23500)**

> **作者:** Jie Liu; Zilyu Ye; Linxiao Yuan; Shenhan Zhu; Yu Gao; Jie Wu; Kunchang Li; Xionghui Wang; Xiaonan Nie; Weilin Huang; Wanli Ouyang
>
> **摘要:** Unified models capable of interleaved generation have emerged as a promising paradigm, with the community increasingly converging on autoregressive modeling for text and flow matching for image generation. To advance this direction, we propose a unified reinforcement learning framework tailored for interleaved generation. We validate our approach on its fundamental unit: a single round of reasoning-driven image generation, where the model first expands the user prompt through reasoning, followed by image synthesis. Formulating this multimodal generation process as a Markov Decision Process with sparse terminal rewards, we introduce UniGRPO to jointly optimize text and image generation policies using GRPO. Adopting a minimalist methodology to avoid over-design, we leverage established training recipes for both modalities by seamlessly integrating standard GRPO for reasoning and FlowGRPO for visual synthesis. To ensure scalability to multi-round interleaved generation, we introduce two critical modifications to the original FlowGRPO: (1) eliminating classifier-free guidance to maintain linear, unbranched rollouts, which is essential for scaling to complex scenarios involving multi-turn interactions and multi-condition generation (e.g., editing); and (2) replacing the standard latent KL penalty with an MSE penalty directly on the velocity fields, providing a more robust and direct regularization signal to mitigate reward hacking effectively. Our experiments demonstrate that this unified training recipe significantly enhances image generation quality through reasoning, providing a robust and scalable baseline for the future post-training of fully interleaved models.
>
---
#### [new 035] VLA-IAP: Training-Free Visual Token Pruning via Interaction Alignment for Vision-Language-Action Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言-动作模型优化任务，旨在解决模型推理成本高、剪枝不准确的问题。提出无需训练的VLA-IAP方法，通过交互对齐实现高效视觉标记剪枝。**

- **链接: [https://arxiv.org/pdf/2603.22991](https://arxiv.org/pdf/2603.22991)**

> **作者:** Jintao Cheng; Haozhe Wang; Weibin Li; Gang Wang; Yipu Zhang; Xiaoyu Tang; Jin Wu; Xieyuanli Chen; Yunhui Liu; Wei Zhang
>
> **备注:** 27 pages, 8 figures
>
> **摘要:** Vision-Language-Action (VLA) models have rapidly advanced embodied intelligence, enabling robots to execute complex, instruction-driven tasks. However, as model capacity and visual context length grow, the inference cost of VLA systems becomes a major bottleneck for real-world deployment on resource-constrained platforms. Existing visual token pruning methods mainly rely on semantic saliency or simple temporal cues, overlooking the continuous physical interaction, a fundamental property of VLA tasks. Consequently, current approaches often prune visually sparse yet structurally critical regions that support manipulation, leading to unstable behavior during early task phases. To overcome this, we propose a shift toward an explicit Interaction-First paradigm. Our proposed \textbf{training-free} method, VLA-IAP (Interaction-Aligned Pruning), introduces a geometric prior mechanism to preserve structural anchors and a dynamic scheduling strategy that adapts pruning intensity based on semantic-motion alignment. This enables a conservative-to-aggressive transition, ensuring robustness during early uncertainty and efficiency once interaction is locked. Extensive experiments show that VLA-IAP achieves a \textbf{97.8\% success rate} with a \textbf{$1.25\times$ speedup} on the LIBERO benchmark, and up to \textbf{$1.54\times$ speedup} while maintaining performance \textbf{comparable to the unpruned backbone}. Moreover, the method demonstrates superior and consistent performance across multiple model architectures and three different simulation environments, as well as a real robot platform, validating its strong generalization capability and practical applicability. Our project website is: \href{this https URL}{this http URL}.
>
---
#### [new 036] Generative Event Pretraining with Foundation Model Alignment
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GEP框架，解决事件相机数据训练视觉基础模型的难题。通过语义对齐和生成序列建模，提升模型在多个下游任务中的性能。**

- **链接: [https://arxiv.org/pdf/2603.23032](https://arxiv.org/pdf/2603.23032)**

> **作者:** Jianwen Cao; Jiaxu Xing; Nico Messikommer; Davide Scaramuzza
>
> **摘要:** Event cameras provide robust visual signals under fast motion and challenging illumination conditions thanks to their microsecond latency and high dynamic range. However, their unique sensing characteristics and limited labeled data make it challenging to train event-based visual foundation models (VFMs), which are crucial for learning visual features transferable across tasks. To tackle this problem, we propose GEP (Generative Event Pretraining), a two-stage framework that transfers semantic knowledge learned from internet-scale image datasets to event data while learning event-specific temporal dynamics. First, an event encoder is aligned to a frozen VFM through a joint regression-contrastive objective, grounding event features in image semantics. Second, a transformer backbone is autoregressively pretrained on mixed event-image sequences to capture the temporal structure unique to events. Our approach outperforms state-of-the-art event pretraining methods on a diverse range of downstream tasks, including object recognition, segmentation, and depth estimation. Together, VFM-guided alignment and generative sequence modeling yield a semantically rich, temporally aware event model that generalizes robustly across domains.
>
---
#### [new 037] Know3D: Prompting 3D Generation with Knowledge from Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于3D生成任务，旨在解决单视角生成中未见区域控制困难的问题。通过引入视觉语言模型知识，实现语言可控的3D生成。**

- **链接: [https://arxiv.org/pdf/2603.22782](https://arxiv.org/pdf/2603.22782)**

> **作者:** Wenyue Chen; Wenjue Chen; Peng Li; Qinghe Wang; Xu Jia; Heliang Zheng; Rongfei Jia; Yuan Liu; Ronggang Wang
>
> **备注:** page: this https URL
>
> **摘要:** Recent advances in 3D generation have improved the fidelity and geometric details of synthesized 3D assets. However, due to the inherent ambiguity of single-view observations and the lack of robust global structural priors caused by limited 3D training data, the unseen regions generated by existing models are often stochastic and difficult to control, which may sometimes fail to align with user intentions or produce implausible geometries. In this paper, we propose Know3D, a novel framework that incorporates rich knowledge from multimodal large language models into 3D generative processes via latent hidden-state injection, enabling language-controllable generation of the back-view for 3D assets. We utilize a VLM-diffusion-based model, where the VLM is responsible for semantic understanding and guidance. The diffusion model acts as a bridge that transfers semantic knowledge from the VLM to the 3D generation model. In this way, we successfully bridge the gap between abstract textual instructions and the geometric reconstruction of unobserved regions, transforming the traditionally stochastic back-view hallucination into a semantically controllable process, demonstrating a promising direction for future 3D generation models.
>
---
#### [new 038] InterDyad: Interactive Dyadic Speech-to-Video Generation by Querying Intermediate Visual Guidance
- **分类: cs.CV**

- **简介: 该论文属于语音到视频生成任务，旨在解决双人互动中依赖关系捕捉和行为控制问题。提出InterDyad框架，通过运动引导和多模态对齐实现自然对话视频生成。**

- **链接: [https://arxiv.org/pdf/2603.23132](https://arxiv.org/pdf/2603.23132)**

> **作者:** Dongwei Pan; Longwei Guo; Jiazhi Guan; Luying Huang; Yiding Li; Haojie Liu; Haocheng Feng; Wei He; Kaisiyuan Wang; Hang Zhou
>
> **备注:** Project Page: this https URL
>
> **摘要:** Despite progress in speech-to-video synthesis, existing methods often struggle to capture cross-individual dependencies and provide fine-grained control over reactive behaviors in dyadic settings. To address these challenges, we propose InterDyad, a framework that enables naturalistic interactive dynamics synthesis via querying structural motion guidance. Specifically, we first design an Interactivity Injector that achieves video reenactment based on identity-agnostic motion priors extracted from reference videos. Building upon this, we introduce a MetaQuery-based modality alignment mechanism to bridge the gap between conversational audio and these motion priors. By leveraging a Multimodal Large Language Model (MLLM), our framework is able to distill linguistic intent from audio to dictate the precise timing and appropriateness of reactions. To further improve lip-sync quality under extreme head poses, we propose Role-aware Dyadic Gaussian Guidance (RoDG) for enhanced lip-synchronization and spatial consistency. Finally, we introduce a dedicated evaluation suite with novelly designed metrics to quantify dyadic interaction. Comprehensive experiments demonstrate that InterDyad significantly outperforms state-of-the-art methods in producing natural and contextually grounded two-person interactions. Please refer to our project page for demo videos: this https URL.
>
---
#### [new 039] MVRD-Bench: Multi-View Learning and Benchmarking for Dynamic Remote Photoplethysmography under Occlusion
- **分类: cs.CV**

- **简介: 该论文属于rPPG任务，解决遮挡下生理信号估计问题。构建了多视角数据集MVRD，提出MVRD-rPPG框架，融合多视角信息提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.22826](https://arxiv.org/pdf/2603.22826)**

> **作者:** Zuxian He; Xu Cheng; Zhaodong Sun; Haoyu Chen; Jingang Shi; Xiaobai Li; Guoying Zhao
>
> **摘要:** Remote photoplethysmography (rPPG) is a non-contact technique that estimates physiological signals by analyzing subtle skin color changes in facial videos. Existing rPPG methods often encounter performance degradation under facial motion and occlusion scenarios due to their reliance on static and single-view facial videos. Thus, this work focuses on tackling the motion-induced occlusion problem for rPPG measurement in unconstrained multi-view facial videos. Specifically, we introduce a Multi-View rPPG Dataset (MVRD), a high-quality benchmark dataset featuring synchronized facial videos from three viewpoints under stationary, speaking, and head movement scenarios to better match real-world conditions. We also propose MVRD-rPPG, a unified multi-view rPPG learning framework that fuses complementary visual cues to maintain robust facial skin coverage, especially under motion conditions. Our method integrates an Adaptive Temporal Optical Compensation (ATOC) module for motion artifact suppression, a Rhythm-Visual Dual-Stream Network to disentangle rhythmic and appearance-related features, and a Multi-View Correlation-Aware Attention (MVCA) for adaptive view-wise signal aggregation. Furthermore, we introduce a Correlation Frequency Adversarial (CFA) learning strategy, which jointly enforces temporal accuracy, spectral consistency, and perceptual realism in the predicted signals. Extensive experiments and ablation studies on the MVRD dataset demonstrate the superiority of our approach. In the MVRD movement scenario, MVRD-rPPG achieves an MAE of 0.90 and a Pearson correlation coefficient (R) of 0.99. The source code and dataset will be made available.
>
---
#### [new 040] URA-Net: Uncertainty-Integrated Anomaly Perception and Restoration Attention Network for Unsupervised Anomaly Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于无监督异常检测任务，旨在解决传统方法因过度泛化导致异常重建良好、检测效果差的问题。提出URA-Net，通过引入不确定性感知和恢复注意力机制，提升异常检测与定位性能。**

- **链接: [https://arxiv.org/pdf/2603.22840](https://arxiv.org/pdf/2603.22840)**

> **作者:** Wei Luo; Peng Xing; Yunkang Cao; Haiming Yao; Weiming Shen; Zechao Li
>
> **备注:** Accepted by IEEE TCSVT
>
> **摘要:** Unsupervised anomaly detection plays a pivotal role in industrial defect inspection and medical image analysis, with most methods relying on the reconstruction framework. However, these methods may suffer from over-generalization, enabling them to reconstruct anomalies well, which leads to poor detection performance. To address this issue, instead of focusing solely on normality reconstruction, we propose an innovative Uncertainty-Integrated Anomaly Perception and Restoration Attention Network (URA-Net), which explicitly restores abnormal patterns to their corresponding normality. First, unlike traditional image reconstruction methods, we utilize a pre-trained convolutional neural network to extract multi-level semantic features as the reconstruction target. To assist the URA-Net learning to restore anomalies, we introduce a novel feature-level artificial anomaly synthesis module to generate anomalous samples for training. Subsequently, a novel uncertainty-integrated anomaly perception module based on Bayesian neural networks is introduced to learn the distributions of anomalous and normal features. This facilitates the estimation of anomalous regions and ambiguous boundaries, laying the foundation for subsequent anomaly restoration. Then, we propose a novel restoration attention mechanism that leverages global normal semantic information to restore detected anomalous regions, thereby obtaining defect-free restored features. Finally, we employ residual maps between input features and restored features for anomaly detection and localization. The comprehensive experimental results on two industrial datasets, MVTec AD and BTAD, along with a medical image dataset, OCT-2017, unequivocally demonstrate the effectiveness and superiority of the proposed method.
>
---
#### [new 041] From Feature Learning to Spectral Basis Learning: A Unifying and Flexible Framework for Efficient and Robust Shape Matching
- **分类: cs.CV**

- **简介: 该论文属于3D形状匹配任务，旨在解决传统方法中忽略谱基优化及计算效率低的问题。通过引入可学习的谱基函数，实现高效鲁棒的非刚性形状匹配。**

- **链接: [https://arxiv.org/pdf/2603.23383](https://arxiv.org/pdf/2603.23383)**

> **作者:** Feifan Luo; Hongyang Chen
>
> **摘要:** Shape matching is a fundamental task in computer graphics and vision, with deep functional maps becoming a prominent paradigm. However, existing methods primarily focus on learning informative feature representations by constraining pointwise and functional maps, while neglecting the optimization of the spectral basis-a critical component of the functional map pipeline. This oversight often leads to suboptimal matching results. Furthermore, many current approaches rely on conventional, time-consuming functional map solvers, incurring significant computational overhead. To bridge these gaps, we introduce Advanced Functional Maps, a framework that generalizes standard functional maps by replacing fixed basis functions with learnable ones, supported by rigorous theoretical guarantees. Specifically, the spectral basis is optimized through a set of learned inhibition functions. Building on this, we propose the first unsupervised spectral basis learning method for robust non-rigid 3D shape matching, enabling the joint, end-to-end optimization of feature extraction and basis functions. Our approach incorporates a novel heat diffusion module and an unsupervised loss function, alongside a streamlined architecture that bypasses expensive solvers and auxiliary losses. Extensive experiments demonstrate that our method significantly outperforms state-of-the-art feature-learning approaches, particularly in challenging non-isometric and topological noise scenarios, while maintaining high efficiency. Finally, we reveal that optimizing basis functions is equivalent to spectral convolution, where inhibition functions act as filters. This insight enables enhanced representations inspired by spectral graph networks, opening new avenues for future research. Our code is available at this https URL.
>
---
#### [new 042] PhotoAgent: A Robotic Photographer with Spatial and Aesthetic Understanding
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出PhotoAgent，解决摄影中语义与几何控制的桥梁问题。整合大模型推理与控制范式，提升图像质量与空间理解。**

- **链接: [https://arxiv.org/pdf/2603.22796](https://arxiv.org/pdf/2603.22796)**

> **作者:** Lirong Che; Zhenfeng Gan; Yanbo Chen; Junbo Tan; Xueqian Wang
>
> **备注:** Accepted to the IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Embodied agents for creative tasks like photography must bridge the semantic gap between high-level language commands and geometric control. We introduce PhotoAgent, an agent that achieves this by integrating Large Multimodal Models (LMMs) reasoning with a novel control paradigm. PhotoAgent first translates subjective aesthetic goals into solvable geometric constraints via LMM-driven, chain-of-thought (CoT) reasoning, allowing an analytical solver to compute a high-quality initial viewpoint. This initial pose is then iteratively refined through visual reflection within a photorealistic internal world model built with 3D Gaussian Splatting (3DGS). This ``mental simulation'' replaces costly and slow physical trial-and-error, enabling rapid convergence to aesthetically superior results. Evaluations confirm that PhotoAgent excels in spatial reasoning and achieves superior final image quality.
>
---
#### [new 043] Multi-Modal Image Fusion via Intervention-Stable Feature Learning
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于多模态图像融合任务，旨在解决现有方法依赖数据诱导的虚假关联问题。通过因果干预策略，提取稳定特征以学习鲁棒的跨模态依赖。**

- **链接: [https://arxiv.org/pdf/2603.23272](https://arxiv.org/pdf/2603.23272)**

> **作者:** Xue Wang; Zheng Guan; Wenhua Qian; Chengchao Wang; Runzhuo Ma
>
> **备注:** Accpted by CVPR 2026
>
> **摘要:** Multi-modal image fusion integrates complementary information from different modalities into a unified representation. Current methods predominantly optimize statistical correlations between modalities, often capturing dataset-induced spurious associations that degrade under distribution shifts. In this paper, we propose an intervention-based framework inspired by causal principles to identify robust cross-modal dependencies. Drawing insights from Pearl's causal hierarchy, we design three principled intervention strategies to probe different aspects of modal relationships: i) complementary masking with spatially disjoint perturbations tests whether modalities can genuinely compensate for each other's missing information, ii) random masking of identical regions identifies feature subsets that remain informative under partial observability, and iii) modality dropout evaluates the irreplaceable contribution of each modality. Based on these interventions, we introduce a Causal Feature Integrator (CFI) that learns to identify and prioritize intervention-stable features maintaining importance across different perturbation patterns through adaptive invariance gating, thereby capturing robust modal dependencies rather than spurious correlations. Extensive experiments demonstrate that our method achieves SOTA performance on both public benchmarks and downstream high-level vision tasks.
>
---
#### [new 044] NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOv3 Initialization
- **分类: cs.CV**

- **简介: 该论文属于3D神经元分割任务，旨在解决3D医学图像标注稀缺的问题。通过将2D视觉模型DINOv3迁移到3D分割，提升分割精度与结构保真度。**

- **链接: [https://arxiv.org/pdf/2603.23104](https://arxiv.org/pdf/2603.23104)**

> **作者:** Yik San Cheng; Runkai Zhao; Weidong Cai
>
> **备注:** 17 pages, 12 figures, and 11 tables. Accepted to CVPR 2026
>
> **摘要:** 2D visual foundation models, such as DINOv3, a self-supervised model trained on large-scale natural images, have demonstrated strong zero-shot generalization, capturing both rich global context and fine-grained structural cues. However, an analogous 3D foundation model for downstream volumetric neuroimaging remains lacking, largely due to the challenges of 3D image acquisition and the scarcity of high-quality annotations. To address this gap, we propose to adapt the 2D visual representations learned by DINOv3 to a 3D biomedical segmentation model, enabling more data-efficient and morphologically faithful neuronal reconstruction. Specifically, we design an inflation-based adaptation strategy that inflates 2D filters into 3D operators, preserving semantic priors from DINOv3 while adapting to 3D neuronal volume patches. In addition, we introduce a topology-aware skeleton loss to explicitly enforce structural fidelity of graph-based neuronal arbor reconstruction. Extensive experiments on four neuronal imaging datasets, including two from BigNeuron and two public datasets, NeuroFly and CWMBS, demonstrate consistent improvements in reconstruction accuracy over SoTA methods, with average gains of 2.9% in Entire Structure Average, 2.8% in Different Structure Average, and 3.8% in Percentage of Different Structure. Code: this https URL.
>
---
#### [new 045] CCF: Complementary Collaborative Fusion for Domain Generalized Multi-Modal 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文属于多模态3D目标检测任务，解决域泛化问题。针对模态退化和LiDAR主导的问题，提出三个组件提升跨域性能。**

- **链接: [https://arxiv.org/pdf/2603.23276](https://arxiv.org/pdf/2603.23276)**

> **作者:** Yuchen Wu; Kun Wang; Yining Pan; Na Zhao
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Multi-modal fusion has emerged as a promising paradigm for accurate 3D object detection. However, performance degrades substantially when deployed in target domains different from training. In this work, focusing on dual-branch proposal-level detectors, we identify two factors that limit robust cross-domain generalization: 1) in challenging domains such as rain or nighttime, one modality may undergo severe degradation; 2) the LiDAR branch often dominates the detection process, leading to systematic underutilization of visual cues and vulnerability when point clouds are compromised. To address these challenges, we propose three components. First, Query-Decoupled Loss provides independent supervision for 2D-only, 3D-only, and fused queries, rebalancing gradient flow across modalities. Second, LiDAR-Guided Depth Prior augments 2D queries with instance-aware geometric priors through probabilistic fusion of image-predicted and LiDAR-derived depth distributions, improving their spatial initialization. Third, Complementary Cross-Modal Masking applies complementary spatial masks to the image and point cloud, encouraging queries from both modalities to compete within the fused decoder and thereby promoting adaptive fusion. Extensive experiments demonstrate substantial gains over state-of-the-art baselines while preserving source-domain performance. Code and models are publicly available at this https URL.
>
---
#### [new 046] WorldMesh: Generating Navigable Multi-Room 3D Scenes via Mesh-Conditioned Image Diffusion
- **分类: cs.CV**

- **简介: 该论文属于3D场景生成任务，旨在解决大规模场景中对象与结构一致性问题。通过先构建网格骨架，再基于其进行真实感图像合成，实现高保真、多样化的3D场景生成。**

- **链接: [https://arxiv.org/pdf/2603.22972](https://arxiv.org/pdf/2603.22972)**

> **作者:** Manuel-Andreas Schneider; Angela Dai
>
> **备注:** Project page: this https URL Video: this https URL Code: this https URL
>
> **摘要:** Recent progress in image and video synthesis has inspired their use in advancing 3D scene generation. However, we observe that text-to-image and -video approaches struggle to maintain scene- and object-level consistency beyond a limited environment scale due to the absence of explicit geometry. We thus present a geometry-first approach that decouples this complex problem of large-scale 3D scene synthesis into its structural composition, represented as a mesh scaffold, and realistic appearance synthesis, which leverages powerful image synthesis models conditioned on the mesh scaffold. From an input text description, we first construct a mesh capturing the environment's geometry (walls, floors, etc.), and then use image synthesis, segmentation and object reconstruction to populate the mesh structure with objects in realistic layouts. This mesh scaffold is then rendered to condition image synthesis, providing a structural backbone for consistent appearance generation. This enables scalable, arbitrarily-sized 3D scenes of high object richness and diversity, combining robust 3D consistency with photorealistic detail. We believe this marks a significant step toward generating truly environment-scale, immersive 3D worlds.
>
---
#### [new 047] SIMART: Decomposing Monolithic Meshes into Sim-ready Articulated Assets via MLLM
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文提出SIMART，解决将单体网格分解为可模拟的关节资产的问题。通过引入稀疏3D VQ-VAE，提升生成效率与精度，适用于物理仿真与机器人模拟。**

- **链接: [https://arxiv.org/pdf/2603.23386](https://arxiv.org/pdf/2603.23386)**

> **作者:** Chuanrui Zhang; Minghan Qin; Yuang Wang; Baifeng Xie; Hang Li; Ziwei Wang
>
> **摘要:** High-quality articulated 3D assets are indispensable for embodied AI and physical simulation, yet 3D generation still focuses on static meshes, leaving a gap in "sim-ready" interactive objects. Most recent articulated object creation methods rely on multi-stage pipelines that accumulate errors across decoupled modules. Alternatively, unified MLLMs offer a single-stage path to joint static asset understanding and sim-ready asset generation. However dense voxel-based 3D tokenization yields long 3D token sequences and high memory overhead, limiting scalability to complex articulated objects. To address this, we propose SIMART, a unified MLLM framework that jointly performs part-level decomposition and kinematic prediction. By introducing a Sparse 3D VQ-VAE, SIMART reduces token counts by 70% vs. dense voxel tokens, enabling high-fidelity multi-part assemblies. SIMART achieves state-of-the-art performance on PartNet-Mobility and in-the-wild AIGC datasets, and enables physics-based robotic simulation.
>
---
#### [new 048] I3DM: Implicit 3D-aware Memory Retrieval and Injection for Consistent Video Scene Generation
- **分类: cs.CV**

- **简介: 该论文属于视频场景生成任务，解决长期场景一致性问题。提出I3DM方法，通过隐式3D感知记忆机制提升重访一致性和相机控制精度。**

- **链接: [https://arxiv.org/pdf/2603.23413](https://arxiv.org/pdf/2603.23413)**

> **作者:** Jia Li; Han Yan; Yihang Chen; Siqi Li; Xibin Song; Yifu Wang; Jianfei Cai; Tien-Tsin Wong; Pan Ji
>
> **备注:** Project page: this https URL
>
> **摘要:** Despite remarkable progress in video generation, maintaining long-term scene consistency upon revisiting previously explored areas remains challenging. Existing solutions rely either on explicitly constructing 3D geometry, which suffers from error accumulation and scale ambiguity, or on naive camera Field-of-View (FoV) retrieval, which typically fails under complex occlusions. To overcome these limitations, we propose I3DM, a novel implicit 3D-aware memory mechanism for consistent video scene generation that bypasses explicit 3D reconstruction. At the core of our approach is a 3D-aware memory retrieval strategy, which leverages the intermediate features of a pre-trained Feed-Forward Novel View Synthesis (FF-NVS) model to score view relevance, enabling robust retrieval even in highly occluded scenarios. Furthermore, to fully utilize the retrieved historical frames, we introduce a 3D-aligned memory injection module. This module implicitly warps historical content to the target view and adaptively conditions the generation on reliable warping regions, leading to improved revisit consistency and accurate camera control. Extensive experiments demonstrate that our method outperforms state-of-the-art approaches, achieving superior revisit consistency, generation fidelity, and camera control precision.
>
---
#### [new 049] Concept-based explanations of Segmentation and Detection models in Natural Disaster Management
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自然灾难管理中的图像分割与目标检测任务，旨在提升模型的可解释性。针对模型决策不透明的问题，提出一种结合LRP和PCX的解释框架，实现可靠且实时的解释，适用于无人机等资源受限平台。**

- **链接: [https://arxiv.org/pdf/2603.23020](https://arxiv.org/pdf/2603.23020)**

> **作者:** Samar Heydari; Jawher Said; Galip Ümit Yolcu; Evgenii Kortukov; Elena Golimblevskaia; Evgenios Vlachos; Vasileios Mygdalis; Ioannis Pitas; Sebastian Lapuschkin; Leila Arras
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Deep learning models for flood and wildfire segmentation and object detection enable precise, real-time disaster localization when deployed on embedded drone platforms. However, in natural disaster management, the lack of transparency in their decision-making process hinders human trust required for emergency response. To address this, we present an explainability framework for understanding flood segmentation and car detection predictions on the widely used PIDNet and YOLO architectures. More specifically, we introduce a novel redistribution strategy that extends Layer-wise Relevance Propagation (LRP) explanations for sigmoid-gated element-wise fusion layers. This extension allows LRP relevances to flow through the fusion modules of PIDNet, covering the entire computation graph back to the input image. Furthermore, we apply Prototypical Concept-based Explanations (PCX) to provide both local and global explanations at the concept level, revealing which learned features drive the segmentation and detection of specific disaster semantic classes. Experiments on a publicly available flood dataset show that our framework provides reliable and interpretable explanations while maintaining near real-time inference capabilities, rendering it suitable for deployment on resource-constrained platforms, such as Unmanned Aerial Vehicles (UAVs).
>
---
#### [new 050] GO-Renderer: Generative Object Rendering with 3D-aware Controllable Video Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于3D物体渲染任务，旨在解决传统方法难以准确建模外观及控制视角的问题。提出GO-Renderer框架，结合3D重建与扩散模型，实现高质量、可控的物体渲染。**

- **链接: [https://arxiv.org/pdf/2603.23246](https://arxiv.org/pdf/2603.23246)**

> **作者:** Zekai Gu; Shuoxuan Feng; Yansong Wang; Hanzhuo Huang; Zhongshuo Du; Chengfeng Zhao; Chengwei Ren; Peng Wang; Yuan Liu
>
> **备注:** Project page: this https URL
>
> **摘要:** Reconstructing a renderable 3D model from images is a useful but challenging task. Recent feedforward 3D reconstruction methods have demonstrated remarkable success in efficiently recovering geometry, but still cannot accurately model the complex appearances of these 3D reconstructed models. Recent diffusion-based generative models can synthesize realistic images or videos of an object using reference images without explicitly modeling its appearance, which provides a promising direction for object rendering, but lacks accurate control over the viewpoints. In this paper, we propose GO-Renderer, a unified framework integrating the reconstructed 3D proxies to guide the video generative models to achieve high-quality object rendering on arbitrary viewpoints under arbitrary lighting conditions. Our method not only enjoys the accurate viewpoint control using the reconstructed 3D proxy but also enables high-quality rendering in different lighting environments using diffusion generative models without explicitly modeling complex materials and lighting. Extensive experiments demonstrate that GO-Renderer achieves state-of-the-art performance across the object rendering tasks, including synthesizing images on new viewpoints, rendering the objects in a novel lighting environment, and inserting an object into an existing video.
>
---
#### [new 051] SMSP: A Plug-and-Play Strategy of Multi-Scale Perception for MLLMs to Perceive Visual Illusions
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于视觉感知任务，旨在解决MLLMs对视觉幻觉的敏感问题。通过提出SMSP框架，抑制高频干扰，提升模型对隐藏内容的感知能力。**

- **链接: [https://arxiv.org/pdf/2603.23118](https://arxiv.org/pdf/2603.23118)**

> **作者:** Jinzhe Tu; Ruilei Guo; Zihan Guo; Junxiao Yang; Shiyao Cui; Minlie Huang
>
> **摘要:** Recent works have shown that Multimodal Large Language Models (MLLMs) are highly vulnerable to hidden-pattern visual illusions, where the hidden content is imperceptible to models but obvious to humans. This deficiency highlights a perceptual misalignment between current MLLMs and humans, and also introduces potential safety concerns. To systematically investigate this failure, we introduce IlluChar, a comprehensive and challenging illusion dataset, and uncover a key underlying mechanism for the models' failure: high-frequency attention bias, where the models are easily distracted by high-frequency background textures in illusion images, causing them to overlook hidden patterns. To address the issue, we propose the Strategy of Multi-Scale Perception (SMSP), a plug-and-play framework that aligns with human visual perceptual strategies. By suppressing distracting high-frequency backgrounds, SMSP generates images closer to human perception. Our experiments demonstrate that SMSP significantly improves the performance of all evaluated MLLMs on illusion images, for instance, increasing the accuracy of Qwen3-VL-8B-Instruct from 13.0% to 84.0%. Our work provides novel insights into MLLMs' visual perception, and offers a practical and robust solution to enhance it. Our code is publicly available at this https URL.
>
---
#### [new 052] Think 360°: Evaluating the Width-centric Reasoning Capability of MLLMs Beyond Depth
- **分类: cs.CV**

- **简介: 该论文属于多模态大模型推理能力评估任务，旨在解决模型在广度与深度推理上的不足。提出基准测试，评估模型的宽广推理能力，并分析其失败模式。**

- **链接: [https://arxiv.org/pdf/2603.22689](https://arxiv.org/pdf/2603.22689)**

> **作者:** Mingrui Chen; Hexiong Yang; Haogeng Liu; Huaibo Huang; Ran He
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** In this paper, we present a holistic multimodal benchmark that evaluates the reasoning capabilities of MLLMs with an explicit focus on reasoning width, a complementary dimension to the more commonly studied reasoning depth. Specifically, reasoning depth measures the model's ability to carry out long-chain, sequential reasoning in which each step is tightly and rigorously linked to the next. Reasoning width tends to focus more on the model's capacity for broad trial-and-error search or multi-constrained optimization: it must systematically traverse many possible and parallelized reasoning paths, apply diverse constraints to prune unpromising branches, and identify valid solution routes for efficient iteration or backtracking. To achieve it, we carefully curate 1200+ high-quality multimodal cases spanning heterogeneous domains, and propose a fine-grained tree-of-thought evaluation protocol that jointly quantifies reasoning width and depth. We evaluate 12 major model families (over 30 advanced MLLMs) across difficulty tiers, question types, and required skills. Results show that while current models exhibit strong performance on general or common-sense VQA tasks, they still struggle to combine deep sequential thought chains with wide exploratory search to perform genuine insight-based reasoning. Finally, we analyze characteristic failure modes to provide possible directions for building MLLMs that reason not only deeper but also wider.
>
---
#### [new 053] Template-Based Feature Aggregation Network for Industrial Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于工业异常检测任务，旨在解决现有方法在特征重建中出现的捷径学习问题。提出TFA-Net模型，通过模板特征聚合有效过滤异常特征，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2603.22874](https://arxiv.org/pdf/2603.22874)**

> **作者:** Wei Luo; Haiming Yao; Wenyong Yu
>
> **备注:** Accepted by Engineering Applications of Artificial Intelligence
>
> **摘要:** Industrial anomaly detection plays a crucial role in ensuring product quality control. Therefore, proposing an effective anomaly detection model is of great significance. While existing feature-reconstruction methods have demonstrated excellent performance, they face challenges with shortcut learning, which can lead to undesirable reconstruction of anomalous features. To address this concern, we present a novel feature-reconstruction model called the \textbf{T}emplate-based \textbf{F}eature \textbf{A}ggregation \textbf{Net}work (TFA-Net) for anomaly detection via template-based feature aggregation. Specifically, TFA-Net first extracts multiple hierarchical features from a pre-trained convolutional neural network for a fixed template image and an input image. Instead of directly reconstructing input features, TFA-Net aggregates them onto the template features, effectively filtering out anomalous features that exhibit low similarity to normal template features. Next, TFA-Net utilizes the template features that have already fused normal features in the input features to refine feature details and obtain the reconstructed feature map. Finally, the defective regions can be located by comparing the differences between the input and reconstructed features. Additionally, a random masking strategy for input features is employed to enhance the overall inspection performance of the model. Our template-based feature aggregation schema yields a nontrivial and meaningful feature reconstruction task. The simple, yet efficient, TFA-Net exhibits state-of-the-art detection performance on various real-world industrial datasets. Additionally, it fulfills the real-time demands of industrial scenarios, rendering it highly suitable for practical applications in the industry. Code is available at this https URL.
>
---
#### [new 054] Cluster-Wise Spatio-Temporal Masking for Efficient Video-Language Pretraining
- **分类: cs.CV**

- **简介: 该论文属于视频-语言预训练任务，旨在解决高掩码比导致的信息丢失和时间信息泄露问题。提出ClusterSTM策略，通过聚类与时间密度保留关键视觉token，提升模型效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.22953](https://arxiv.org/pdf/2603.22953)**

> **作者:** Weijun Zhuang; Yuqing Huang; Weikang Meng; Xin Li; Ming Liu; Xiaopeng Hong; Yaowei Wang; Wangmeng Zuo
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Large-scale video-language pretraining enables strong generalization across multimodal tasks but often incurs prohibitive computational costs. Although recent advances in masked visual modeling help mitigate this issue, they still suffer from two fundamental limitations: severe visual information loss under high masking ratios and temporal information leakage caused by inter-frame correlations. To address these challenges, we propose ClusterSTM, a Cluster-Wise Spatio-Temporal Masking strategy for efficient video-language pretraining. ClusterSTM first performs intra-frame clustering to partition visual tokens into multiple semantically independent clusters, then conducts cluster-wise masking by retaining the token with the highest temporal density within each cluster. Our masking strategy ensure that the retained tokens capture holistic video content while exhibit strong temporal correlation. Additionally, we introduce a video-text relevance reconstruction objective that aligns high-level multimodal semantics beyond conventional visual reconstruction. Extensive experiments across multiple benchmarks demonstrate that ClusterSTM achieves superior performance on video-text retrieval, video question answering, and video captioning tasks, establishing a new state-of-the-art among efficient video-language models.
>
---
#### [new 055] A Vision Language Model for Generating Procedural Plant Architecture Representations from Simulated Images
- **分类: cs.CV**

- **简介: 该论文属于植物结构建模任务，旨在从图像生成3D植物架构模型。通过视觉语言模型从合成图像中提取参数，解决人工测量劳动密集的问题。**

- **链接: [https://arxiv.org/pdf/2603.22622](https://arxiv.org/pdf/2603.22622)**

> **作者:** Heesup Yun; Isaac Kazuo Uyehara; Ioannis Droutsas; Earl Ranario; Christine H. Diepenbrock; Brian N. Bailey; J. Mason Earles
>
> **摘要:** Three-dimensional (3D) procedural plant architecture models have emerged as an important tool for simulation-based studies of plant structure and function, extracting plant architectural parameters from field measurements, and for generating realistic plants in computer graphics. However, measuring the architectural parameters and nested structures for these models at the field scales remains prohibitively labor-intensive. We present a novel algorithm that generates a 3D plant architecture from an image, creating a functional structural plant model that reflects organ-level geometric and topological parameters and provides a more comprehensive representation of the plant's architecture. Instead of using 3D sensors or processing multi-view images with computer vision to obtain the 3D structure of plants, we proposed a method that generates token sequences that encode a procedural definition of plant architecture. This work used only synthetic images for training and testing, with exact architectural parameters known, allowing testing of the hypothesis that organ-level architectural parameters could be extracted from image data using a vision-language model (VLM). A synthetic dataset of cowpea plant images was generated using the Helios 3D plant simulator, with the detailed plant architecture encoded in XML files. We developed a plant architecture tokenizer for the XML file defining plant architecture, converting it into a token sequence that a language model can predict. The model achieved a token F1 score of 0.73 during teacher-forced training. Evaluation of the model was performed through autoregressive generation, achieving a BLEU-4 score of 94.00% and a ROUGE-L score of 0.5182. This led to the conclusion that such plant architecture model generation and parameter extraction were possible from synthetic images; thus, future work will extend the approach to real imagery data.
>
---
#### [new 056] Caption Generation for Dongba Paintings via Prompt Learning and Semantic Fusion
- **分类: cs.CV**

- **简介: 该论文属于图像描述生成任务，旨在解决Dongba绘画自动文本描述问题。通过引入提示学习和语义融合机制，提升模型对文化特异性图像的描述能力。**

- **链接: [https://arxiv.org/pdf/2603.22946](https://arxiv.org/pdf/2603.22946)**

> **作者:** Shuangwu Qian; Xiaochan Yuan; Pengfei Liu
>
> **摘要:** Dongba paintings, the treasured pictorial legacy of the Naxi people in southwestern China, feature richly layered visual elements, vivid color palettes, and pronounced ethnic and regional cultural symbolism, yet their automatic textual description remains largely unexplored owing to severe domain shift when mainstream captioning models are applied directly. This paper proposes \textbf{PVGF-DPC} (\textit{Prompt and Visual Semantic-Generation Fusion-based Dongba Painting Captioning}), an encoder-decoder framework that integrates a content prompt module with a novel visual semantic-generation fusion loss to bridge the gap between generic natural-image captioning and the culturally specific imagery found in Dongba art. A MobileNetV2 encoder extracts discriminative visual features, which are injected into the layer normalization of a 10-layer Transformer decoder initialized with pretrained BERT weights; meanwhile, the content prompt module maps the image feature vector to culture-aware labels -- such as \emph{deity}, \emph{ritual pattern}, or \emph{hell ghost} -- and constructs a post-prompt that steers the decoder toward thematically accurate descriptions. The visual semantic-generation fusion loss jointly optimizes the cross-entropy objectives of both the prompt predictor and the caption generator, encouraging the model to extract key cultural and visual cues and to produce captions that are semantically aligned with the input image. We construct a dedicated Dongba painting captioning dataset comprising 9{}408 augmented images with culturally grounded annotations spanning seven thematic categories.
>
---
#### [new 057] When Visuals Aren't the Problem: Evaluating Vision-Language Models on Misleading Data Visualizations
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型评估任务，旨在解决模型检测误导性数据可视化的能力问题。通过构建基准测试，分析模型对不同错误类型的识别效果。**

- **链接: [https://arxiv.org/pdf/2603.22368](https://arxiv.org/pdf/2603.22368)**

> **作者:** Harsh Nishant Lalai; Raj Sanjay Shah; Hanspeter Pfister; Sashank Varma; Grace Guo
>
> **摘要:** Visualizations help communicate data insights, but deceptive data representations can distort their interpretation and propagate misinformation. While recent Vision Language Models (VLMs) perform well on many chart understanding tasks, their ability to detect misleading visualizations, especially when deception arises from subtle reasoning errors in captions, remains poorly understood. Here, we evaluate VLMs on misleading visualization-caption pairs grounded in a fine-grained taxonomy of reasoning errors (e.g., Cherry-picking, Causal inference) and visualization design errors (e.g., Truncated axis, Dual axis, inappropriate encodings). To this end, we develop a benchmark that combines real-world visualization with human-authored, curated misleading captions designed to elicit specific reasoning and visualization error types, enabling controlled analysis across error categories and modalities of misleadingness. Evaluating many commercial and open-source VLMs, we find that models detect visual design errors substantially more reliably than reasoning-based misinformation, and frequently misclassify non-misleading visualizations as deceptive. Overall, our work fills a gap between coarse detection of misleading content and the attribution of the specific reasoning or visualization errors that give rise to it.
>
---
#### [new 058] FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning
- **分类: cs.CV**

- **简介: 该论文属于弱监督伪装目标检测任务，旨在解决标注成本高和检测效果差的问题。提出FCL-COD框架，结合频率感知和对比学习提升检测精度。**

- **链接: [https://arxiv.org/pdf/2603.22969](https://arxiv.org/pdf/2603.22969)**

> **作者:** Jingchen Ni; Quan Zhang; Dan Jiang; Keyu Lv; Ke Zhang; Chun Yuan
>
> **备注:** CVPR 2026 Findings
>
> **摘要:** Existing camouflage object detection (COD) methods typically rely on fully-supervised learning guided by mask annotations. However, obtaining mask annotations is time-consuming and labor-intensive. Compared to fully-supervised methods, existing weakly-supervised COD methods exhibit significantly poorer performance. Even for the Segment Anything Model (SAM), there are still challenges in handling weakly-supervised camouflage object detection (WSCOD), such as: a. non-camouflage target responses, b. local responses, c. extreme responses, and d. lack of refined boundary awareness, which leads to unsatisfactory results in camouflage scenes. To alleviate these issues, we propose a frequency-aware and contrastive learning-based WSCOD framework in this paper, named FCL-COD. To mitigate the problem of non-camouflaged object responses, we propose the Frequency-aware Low-rank Adaptation (FoRA) method, which incorporates frequency-aware camouflage scene knowledge into SAM. To overcome the challenges of local and extreme responses, we introduce a gradient-aware contrastive learning approach that effectively delineates precise foreground-background boundaries. Additionally, to address the lack of refined boundary perception, we present a multi-scale frequency-aware representation learning strategy that facilitates the modeling of more refined boundaries. We validate the effectiveness of our approach through extensive empirical experiments on three widely recognized COD benchmarks. The results confirm that our method surpasses both state-of-the-art weakly supervised and even fully supervised techniques.
>
---
#### [new 059] Foveated Diffusion: Efficient Spatially Adaptive Image and Video Generation
- **分类: cs.CV**

- **简介: 该论文属于图像和视频生成任务，旨在解决高分辨率下生成效率低的问题。通过引入视网膜聚焦机制，优化令牌分配，提升生成效率。**

- **链接: [https://arxiv.org/pdf/2603.23491](https://arxiv.org/pdf/2603.23491)**

> **作者:** Brian Chao; Lior Yariv; Howard Xiao; Gordon Wetzstein
>
> **备注:** Project website at this https URL
>
> **摘要:** Diffusion and flow matching models have unlocked unprecedented capabilities for creative content creation, such as interactive image and streaming video generation. The growing demand for higher resolutions, frame rates, and context lengths, however, makes efficient generation increasingly challenging, as computational complexity grows quadratically with the number of generated tokens. Our work seeks to optimize the efficiency of the generation process in settings where the user's gaze location is known or can be estimated, for example, by using eye tracking. In these settings, we leverage the eccentricity-dependent acuity of human vision: while a user perceives very high-resolution visual information in a small region around their gaze location (the foveal region), the ability to resolve detail quickly degrades in the periphery of the visual field. Our approach starts with a mask modeling the foveated resolution to allocate tokens non-uniformly, assigning higher token density to foveal regions and lower density to peripheral regions. An image or video is generated in a mixed-resolution token setting, yielding results perceptually indistinguishable from full-resolution generation, while drastically reducing the token count and generation time. To this end, we develop a principled mechanism for constructing mixed-resolution tokens directly from high-resolution data, allowing a foveated diffusion model to be post-trained from an existing base model while maintaining content consistency across resolutions. We validate our approach through extensive analysis and a carefully designed user study, demonstrating the efficacy of foveation as a practical and scalable axis for efficient generation.
>
---
#### [new 060] HUydra: Full-Range Lung CT Synthesis via Multiple HU Interval Generative Modelling
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学图像合成任务，旨在解决肺部CT数据稀缺问题。通过分阶段生成HU区间图像并融合，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2603.23041](https://arxiv.org/pdf/2603.23041)**

> **作者:** António Cardoso; Pedro Sousa; Tania Pereira; Hélder P. Oliveira
>
> **备注:** Submitted to iEEE TPAMI (Transactions on Pattern Analysis and Machine Intelligence)
>
> **摘要:** Currently, a central challenge and bottleneck in the deployment and validation of computer-aided diagnosis (CAD) models within the field of medical imaging is data scarcity. For lung cancer, one of the most prevalent types worldwide, limited datasets can delay diagnosis and have an impact on patient outcome. Generative AI offers a promising solution for this issue, but dealing with the complex distribution of full Hounsfield Unit (HU) range lung CT scans is challenging and remains as a highly computationally demanding task. This paper introduces a novel decomposition strategy that synthesizes CT images one HU interval at a time, rather than modelling the entire HU domain at once. This framework focuses on training generative architectures on individual tissue-focused HU windows, then merges their output into a full-range scan via a learned reconstruction network that effectively reverses the HU-windowing process. We further propose multi-head and multi-decoder models to better capture textures while preserving anatomical consistency, with a multi-head VQVAE achieving the best performance for the generative task. Quantitative evaluation shows this approach significantly outperforms conventional 2D full-range baselines, achieving a 6.2% improvement in FID and superior MMD, Precision, and Recall across all HU intervals. The best performance is achieved by a multi-head VQVAE variant, demonstrating that it is possible to enhance visual fidelity and variability while also reducing model complexity and computational cost. This work establishes a new paradigm for structure-aware medical image synthesis, aligning generative modelling with clinical interpretation.
>
---
#### [new 061] DetPO: In-Context Learning with Multi-Modal LLMs for Few-Shot Object Detection
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，解决多模态大模型在少样本情况下的泛化问题。提出DetPO方法，通过优化文本提示提升检测精度。**

- **链接: [https://arxiv.org/pdf/2603.23455](https://arxiv.org/pdf/2603.23455)**

> **作者:** Gautam Rajendrakumar Gare; Neehar Peri; Matvei Popov; Shruti Jain; John Galeotti; Deva Ramanan
>
> **备注:** Project Page: this https URL
>
> **摘要:** Multi-Modal LLMs (MLLMs) demonstrate strong visual grounding capabilities on popular object detection benchmarks like OdinW-13 and RefCOCO. However, state-of-the-art models still struggle to generalize to out-of-distribution classes, tasks and imaging modalities not typically found in their pre-training. While in-context prompting is a common strategy to improve performance across diverse tasks, we find that it often yields lower detection accuracy than prompting with class names alone. This suggests that current MLLMs cannot yet effectively leverage few-shot visual examples and rich textual descriptions for object detection. Since frontier MLLMs are typically only accessible via APIs, and state-of-the-art open-weights models are prohibitively expensive to fine-tune on consumer-grade hardware, we instead explore black-box prompt optimization for few-shot object detection. To this end, we propose Detection Prompt Optimization (DetPO), a gradient-free test-time optimization approach that refines text-only prompts by maximizing detection accuracy on few-shot visual training examples while calibrating prediction confidence. Our proposed approach yields consistent improvements across generalist MLLMs on Roboflow20-VL and LVIS, outperforming prior black-box approaches by up to 9.7%. Our code is available at this https URL
>
---
#### [new 062] Gimbal360: Differentiable Auto-Leveling for Canonicalized $360^\circ$ Panoramic Image Completion
- **分类: cs.CV**

- **简介: 该论文提出Gimbal360，解决360°全景图像补全任务中的几何与拓扑不匹配问题，通过引入规范视图空间和拓扑等变性约束提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.23179](https://arxiv.org/pdf/2603.23179)**

> **作者:** Yuqin Lu; Haofeng Liu; Yang Zhou; Jun Liang; Shengfeng He; Jing Li
>
> **备注:** Project page: this https URL
>
> **摘要:** Diffusion models excel at 2D outpainting, but extending them to $360^\circ$ panoramic completion from unposed perspective images is challenging due to the geometric and topological mismatch between perspective projections and spherical panoramas. We present Gimbal360, a principled framework that explicitly bridges perspective observations and spherical panoramas. We introduce a Canonical Viewing Space that regularizes projective geometry and provides a consistent intermediate representation between the two domains. To anchor in-the-wild inputs to this space, we propose a Differentiable Auto-Leveling module that stabilizes feature orientation without requiring camera parameters at inference. Panoramic generation also introduces a topological challenge. Standard generative architectures assume a bounded Euclidean image plane, while Equirectangular Projection (ERP) panoramas exhibit intrinsic $S^1$ periodicity. Euclidean operations therefore break boundary continuity. We address this mismatch by enforcing topological equivariance in the latent space to preserve seamless periodic structure. To support this formulation, we introduce Horizon360, a curated large-scale dataset of gravity-aligned panoramic environments. Extensive experiments show that explicitly standardizing geometric and topological priors enables Gimbal360 to achieve state-of-the-art performance in structurally consistent $360^\circ$ scene completion.
>
---
#### [new 063] ForestPrune: High-ratio Visual Token Compression for Video Multimodal Large Language Models via Spatial-Temporal Forest Modeling
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频多模态大语言模型任务，旨在解决视频token高比例压缩问题。提出ForestPrune方法，通过时空森林建模实现高效压缩。**

- **链接: [https://arxiv.org/pdf/2603.22911](https://arxiv.org/pdf/2603.22911)**

> **作者:** Shaobo Ju; Baiyang Song; Tao Chen; Jiapeng Zhang; Qiong Wu; Chao Chang; HuaiXi Wang; Yiyi Zhou; Rongrong Ji
>
> **摘要:** Due to the great saving of computation and memory overhead, token compression has become a research hot-spot for MLLMs and achieved remarkable progress in image-language tasks. However, for the video, existing methods still fall short of high-ratio token compression. We attribute this shortcoming to the insufficient modeling of temporal and continual video content, and propose a novel and training-free token pruning method for video MLLMs, termed ForestPrune, which achieves effective and high-ratio pruning via Spatial-temporal Forest Modeling. In practice, ForestPrune construct token forests across video frames based on the semantic, spatial and temporal constraints, making an overall comprehension of videos. Afterwards, ForestPrune evaluates the importance of token trees and nodes based on tree depth and node roles, thereby obtaining a globally optimal pruning decision. To validate ForestPrune, we apply it to two representative video MLLMs, namely LLaVA-Video and LLaVA-OneVision, and conduct extensive experiments on a bunch of video benchmarks. The experimental results not only show the great effectiveness for video MLLMs, e.g., retaining 95.8% average accuracy while reducing 90% tokens for LLaVA-OneVision, but also show its superior performance and efficiency than the compared token compression methods, e.g., +10.1% accuracy on MLVU and -81.4% pruning time than FrameFusion on LLaVA-Video.
>
---
#### [new 064] VISion On Request: Enhanced VLLM efficiency with sparse, dynamically selected, vision-language interactions
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉语言模型效率优化任务，旨在解决现有方法因减少视觉令牌导致的信息瓶颈问题。提出VISOR方法，通过稀疏交互提升效率，保留视觉信息。**

- **链接: [https://arxiv.org/pdf/2603.23495](https://arxiv.org/pdf/2603.23495)**

> **作者:** Adrian Bulat; Alberto Baldrati; Ioannis Maniadis Metaxas; Yassine Ouali; Georgios Tzimiropoulos
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Existing approaches for improving the efficiency of Large Vision-Language Models (LVLMs) are largely based on the concept of visual token reduction. This approach, however, creates an information bottleneck that impairs performance, especially on challenging tasks that require fine-grained understanding and reasoning. In this work, we challenge this paradigm by introducing VISion On Request (VISOR), a method that reduces inference cost without discarding visual information. Instead of compressing the image, VISOR improves efficiency by sparsifying the interaction between image and text tokens. Specifically, the language model attends to the full set of high-resolution visual tokens through a small, strategically placed set of attention layers: general visual context is provided by efficient cross-attention between text-image, while a few well-placed and dynamically selected self-attention layers refine the visual representations themselves, enabling complex, high-resolution reasoning when needed. Based on this principle, we first train a single universal network on a range of computational budgets by varying the number of self-attention layers, and then introduce a lightweight policy mechanism that dynamically allocates visual computation based on per-sample complexity. Extensive experiments show that VISOR drastically reduces computational cost while matching or exceeding state-of-the-art results across a diverse suite of benchmarks, and excels in challenging tasks that require detailed visual understanding.
>
---
#### [new 065] Automatic Segmentation of 3D CT scans with SAM2 using a zero-shot approach
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决3D CT扫描的自动分割问题。通过零样本方法应用SAM2模型，改进其推理流程以适应三维数据，实现无需微调的准确分割。**

- **链接: [https://arxiv.org/pdf/2603.23116](https://arxiv.org/pdf/2603.23116)**

> **作者:** Miquel Lopez Escoriza; Pau Amargant Alvarez
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Foundation models for image segmentation have shown strong generalization in natural images, yet their applicability to 3D medical imaging remains limited. In this work, we study the zero-shot use of Segment Anything Model 2 (SAM2) for automatic segmentation of volumetric CT data, without any fine-tuning or domain-specific training. We analyze how SAM2 should be applied to CT volumes and identify its main limitation: the lack of inherent volumetric awareness. To address this, we propose a set of inference-alone architectural and procedural modifications that adapt SAM2's video-based memory mechanism to 3D data by treating CT slices as ordered sequences. We conduct a systematic ablation study on a subset of 500 CT scans from the TotalSegmentator dataset to evaluate prompt strategies, memory propagation schemes and multi-pass refinement. Based on these findings, we select the best-performing configuration and report final results on a bigger sample of the TotalSegmentator dataset comprising 2,500 CT scans. Our results show that, even with frozen weights, SAM2 can produce coherent 3D segmentations when its inference pipeline is carefully structured, demonstrating the feasibility of a fully zero-shot approach for volumetric medical image segmentation.
>
---
#### [new 066] Dual Contrastive Network for Few-Shot Remote Sensing Image Scene Classification
- **分类: cs.CV**

- **简介: 该论文属于少样本遥感图像场景分类任务，旨在解决遥感图像中类间差异小、类内差异大的问题。提出双对比网络（DCN），通过上下文和细节引导的对比学习分支提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.23161](https://arxiv.org/pdf/2603.23161)**

> **作者:** Zhong Ji; Liyuan Hou; Xuan Wang; Gang Wang; Yanwei Pang
>
> **摘要:** Few-shot remote sensing image scene classification (FS-RSISC) aims at classifying remote sensing images with only a few labeled samples. The main challenges lie in small inter-class variances and large intra-class variances, which are the inherent property of remote sensing images. To address these challenges, we propose a transfer-based Dual Contrastive Network (DCN), which incorporates two auxiliary supervised contrastive learning branches during the training process. Specifically, one is a Context-guided Contrastive Learning (CCL) branch and the other is a Detail-guided Contrastive Learning (DCL) branch, which focus on inter-class discriminability and intra-class invariance, respectively. In the CCL branch, we first devise a Condenser Network to capture context features, and then leverage a supervised contrastive learning on top of the obtained context features to facilitate the model to learn more discriminative features. In the DCL branch, a Smelter Network is designed to highlight the significant local detail information. And then we construct a supervised contrastive learning based on the detail feature maps to fully exploit the spatial information in each map, enabling the model to concentrate on invariant detail features. Extensive experiments on four public benchmark remote sensing datasets demonstrate the competitive performance of our proposed DCN.
>
---
#### [new 067] Zero-Shot Personalization of Objects via Textual Inversion
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决对象个性化生成的效率与通用性问题。提出一种无需训练的框架，通过文本反转嵌入实现快速、零样本的对象个性化。**

- **链接: [https://arxiv.org/pdf/2603.23010](https://arxiv.org/pdf/2603.23010)**

> **作者:** Aniket Roy; Maitreya Suin; Rama Chellappa
>
> **摘要:** Recent advances in text-to-image diffusion models have substantially improved the quality of image customization, enabling the synthesis of highly realistic images. Despite this progress, achieving fast and efficient personalization remains a key challenge, particularly for real-world applications. Existing approaches primarily accelerate customization for human subjects by injecting identity-specific embeddings into diffusion models, but these strategies do not generalize well to arbitrary object categories, limiting their applicability. To address this limitation, we propose a novel framework that employs a learned network to predict object-specific textual inversion embeddings, which are subsequently integrated into the UNet timesteps of a diffusion model for text-conditional customization. This design enables rapid, zero-shot personalization of a wide range of objects in a single forward pass, offering both flexibility and scalability. Extensive experiments across multiple tasks and settings demonstrate the effectiveness of our approach, highlighting its potential to support fast, versatile, and inclusive image customization. To the best of our knowledge, this work represents the first attempt to achieve such general-purpose, training-free personalization within diffusion models, paving the way for future research in personalized image generation.
>
---
#### [new 068] Pose-Free Omnidirectional Gaussian Splatting for 360-Degree Videos with Consistent Depth Priors
- **分类: cs.CV**

- **简介: 该论文属于3D场景重建任务，解决无姿态全景视频的3D高斯溅射问题。提出PFGS360方法，通过深度先验实现准确位姿估计和高质量新视角合成。**

- **链接: [https://arxiv.org/pdf/2603.23324](https://arxiv.org/pdf/2603.23324)**

> **作者:** Chuanqing Zhuang; Xin Lu; Zehui Deng; Zhengda Lu; Yiqun Wang; Junqi Diao; Jun Xiao
>
> **摘要:** Omnidirectional 3D Gaussian Splatting with panoramas is a key technique for 3D scene representation, and existing methods typically rely on slow SfM to provide camera poses and sparse points priors. In this work, we propose a pose-free omnidirectional 3DGS method, named PFGS360, that reconstructs 3D Gaussians from unposed omnidirectional videos. To achieve accurate camera pose estimation, we first construct a spherical consistency-aware pose estimation module, which recovers poses by establishing consistent 2D-3D correspondences between the reconstructed Gaussians and the unposed images using Gaussians' internal depth priors. Besides, to enhance the fidelity of novel view synthesis, we introduce a depth-inlier-aware densification module to extract depth inliers and Gaussian outliers with consistent monocular depth priors, enabling efficient Gaussian densification and achieving photorealistic novel view synthesis. The experiments show significant outperformance over existing pose-free and pose-aware 3DGS methods on both real-world and synthetic 360-degree videos. Code is available at this https URL.
>
---
#### [new 069] UAV-DETR: DETR for Anti-Drone Target Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于反无人机目标检测任务，旨在解决小目标检测中特征表示与计算效率的平衡问题。提出UAV-DETR框架，提升检测精度与实时性。**

- **链接: [https://arxiv.org/pdf/2603.22841](https://arxiv.org/pdf/2603.22841)**

> **作者:** Jun Yang; Dong Wang; Hongxu Yin; Hongpeng Li; Jianxiong Yu
>
> **摘要:** Drone detection is pivotal in numerous security and counter-UAV applications. However, existing deep learning-based methods typically struggle to balance robust feature representation with computational efficiency. This challenge is particularly acute when detecting miniature drones against complex backgrounds under severe environmental interference. To address these issues, we introduce UAV-DETR, a novel framework that integrates a small-target-friendly architecture with real-time detection capabilities. Specifically, UAV-DETR features a WTConv-enhanced backbone and a Sliding Window Self-Attention (SWSA-IFI) encoder, capturing the high-frequency structural details of tiny targets while drastically reducing parameter overhead. Furthermore, we propose an Efficient Cross-Scale Feature Recalibration and Fusion Network (ECFRFN) to suppress background noise and aggregate multi-scale semantics. To further enhance accuracy, UAV-DETR incorporates a hybrid Inner-CIoU and NWD loss strategy, mitigating the extreme sensitivity of standard IoU metrics to minor positional deviations in small objects. Extensive experiments demonstrate that UAV-DETR significantly outperforms the baseline RT-DETR on our custom UAV dataset (+6.61% in mAP50:95, with a 39.8% reduction in parameters) and the public DUT-ANTI-UAV benchmark (+1.4% in Precision, +1.0% in F1-Score). These results establish UAV-DETR as a superior trade-off between efficiency and precision in counter-UAV object detection. The code is available at this https URL.
>
---
#### [new 070] A vision-language model and platform for temporally mapping surgery from video
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出Halsted模型及平台，用于从手术视频中时序映射手术行为，解决手术指南制定和机器人自主手术中的数据标准化与可访问性问题。**

- **链接: [https://arxiv.org/pdf/2603.22583](https://arxiv.org/pdf/2603.22583)**

> **作者:** Dani Kiyasseh
>
> **摘要:** Mapping surgery is fundamental to developing operative guidelines and enabling autonomous robotic surgery. Recent advances in artificial intelligence (AI) have shown promise in mapping the behaviour of surgeons from videos, yet current models remain narrow in scope, capturing limited behavioural components within single procedures, and offer limited translational value, as they remain inaccessible to practising surgeons. Here we introduce Halsted, a vision-language model trained on the Halsted Surgical Atlas (HSA), one of the most comprehensive annotated video libraries grown through an iterative self-labelling framework and encompassing over 650,000 videos across eight surgical specialties. To facilitate benchmarking, we publicly release HSA-27k, a subset of the Halsted Surgical Atlas. Halsted surpasses previous state-of-the-art models in mapping surgical activity while offering greater comprehensiveness and computational efficiency. To bridge the longstanding translational gap of surgical AI, we develop the Halsted web platform (this https URL) to provide surgeons anywhere in the world with the previously-unavailable capability of automatically mapping their own procedures within minutes. By standardizing unstructured surgical video data and making these capabilities directly accessible to surgeons, our work brings surgical AI closer to clinical deployment and helps pave the way toward autonomous robotic surgery.
>
---
#### [new 071] A Feature Shuffling and Restoration Strategy for Universal Unsupervised Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于异常检测任务，旨在解决重建方法中的相同捷径问题。提出FSR框架，通过特征打乱与恢复提升模型对全局上下文的感知能力。**

- **链接: [https://arxiv.org/pdf/2603.22861](https://arxiv.org/pdf/2603.22861)**

> **作者:** Wei Luo; Haiming Yao; Zhenfeng Qiang; Xiaotian Zhang; Weihang Zhang
>
> **备注:** Accepted by Knowledge-Based Systems
>
> **摘要:** Unsupervised anomaly detection is vital in industrial fields, with reconstruction-based methods favored for their simplicity and effectiveness. However, reconstruction methods often encounter an identical shortcut issue, where both normal and anomalous regions can be well reconstructed and fail to identify outliers. The severity of this problem increases with the complexity of the normal data distribution. Consequently, existing methods may exhibit excellent detection performance in a specific scenario, but their performance sharply declines when transferred to another scenario. This paper focuses on establishing a universal model applicable to anomaly detection tasks across different settings, termed as universal anomaly detection. In this work, we introduce a novel, straightforward yet efficient framework for universal anomaly detection: \uline{F}eature \uline{S}huffling and \uline{R}estoration (FSR), which can alleviate the identical shortcut issue across different settings. First and foremost, FSR employs multi-scale features with rich semantic information as reconstruction targets, rather than raw image pixels. Subsequently, these multi-scale features are partitioned into non-overlapping feature blocks, which are randomly shuffled and then restored to their original state using a restoration network. This simple paradigm encourages the model to focus more on global contextual information. Additionally, we introduce a novel concept, the shuffling rate, to regulate the complexity of the FSR task, thereby alleviating the identical shortcut across different settings. Furthermore, we provide theoretical explanations for the effectiveness of FSR framework from two perspectives: network structure and mutual information. Extensive experimental results validate the superiority and efficiency of the FSR framework across different this http URL is available at this https URL.
>
---
#### [new 072] CanViT: Toward Active-Vision Foundation Models
- **分类: cs.CV**

- **简介: 该论文提出CanViT，解决主动视觉基础模型的构建问题，通过场景相对RoPE和Canvas Attention实现高效感知，适用于分割和分类任务。**

- **链接: [https://arxiv.org/pdf/2603.22570](https://arxiv.org/pdf/2603.22570)**

> **作者:** Yohaï-Eliel Berreby; Sabrina Du; Audrey Durand; B. Suresh Krishna
>
> **备注:** Code and weights: this https URL
>
> **摘要:** Active computer vision promises efficient, biologically plausible perception through sequential, localized glimpses, but lacks scalable general-purpose architectures and pretraining pipelines. As a result, Active-Vision Foundation Models (AVFMs) have remained unexplored. We introduce CanViT, the first task- and policy-agnostic AVFM. CanViT uses scene-relative RoPE to bind a retinotopic Vision Transformer backbone and a spatiotopic scene-wide latent workspace, the canvas. Efficient interaction with this high-capacity working memory is supported by Canvas Attention, a novel asymmetric cross-attention mechanism. We decouple thinking (backbone-level) and memory (canvas-level), eliminating canvas-side self-attention and fully-connected layers to achieve low-latency sequential inference and scalability to large scenes. We propose a label-free active vision pretraining scheme, policy-agnostic passive-to-active dense latent distillation: reconstructing scene-wide DINOv3 embeddings from sequences of low-resolution glimpses with randomized locations, zoom levels, and lengths. We pretrain CanViT-B from a random initialization on 13.2 million ImageNet-21k scenes -- an order of magnitude more than previous active models -- and 1 billion random glimpses, in 166 hours on a single H100. On ADE20K segmentation, a frozen CanViT-B achieves 38.5% mIoU in a single low-resolution glimpse, outperforming the best active model's 27.6% with 19.5x fewer inference FLOPs and no fine-tuning, as well as its FLOP- or input-matched DINOv3 teacher. Given additional glimpses, CanViT-B reaches 45.9% ADE20K mIoU. On ImageNet-1k classification, CanViT-B reaches 81.2% top-1 accuracy with frozen teacher probes. CanViT generalizes to longer rollouts, larger scenes, and new policies. Our work closes the wide gap between passive and active vision on semantic segmentation and demonstrates the potential of AVFMs as a new research axis.
>
---
#### [new 073] Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出Ego2Web基准，解决AI代理在真实物理环境与网络任务间协同的问题，通过结合第一视角视频与网络任务进行评估。**

- **链接: [https://arxiv.org/pdf/2603.22529](https://arxiv.org/pdf/2603.22529)**

> **作者:** Shoubin Yu; Lei Shu; Antoine Yang; Yao Fu; Srinivas Sunkara; Maria Wang; Jindong Chen; Mohit Bansal; Boqing Gong
>
> **备注:** CVPR 2026. Project page: this https URL
>
> **摘要:** Multimodal AI agents are increasingly automating complex real-world workflows that involve online web execution. However, current web-agent benchmarks suffer from a critical limitation: they focus entirely on web-based interaction and perception, lacking grounding in the user's real-world physical surroundings. This limitation prevents evaluation in crucial scenarios, such as when an agent must use egocentric visual perception (e.g., via AR glasses) to recognize an object in the user's surroundings and then complete a related task online. To address this gap, we introduce Ego2Web, the first benchmark designed to bridge egocentric video perception and web agent execution. Ego2Web pairs real-world first-person video recordings with web tasks that require visual understanding, web task planning, and interaction in an online environment for successful completion. We utilize an automatic data-generation pipeline combined with human verification and refinement to curate well-constructed, high-quality video-task pairs across diverse web task types, including e-commerce, media retrieval, knowledge lookup, etc. To facilitate accurate and scalable evaluation for our benchmark, we also develop a novel LLM-as-a-Judge automatic evaluation method, Ego2WebJudge, which achieves approximately 84% agreement with human judgment, substantially higher than existing evaluation methods. Experiments with diverse SoTA agents on our Ego2Web show that their performance is weak, with substantial headroom across all task categories. We also conduct a comprehensive ablation study on task design, highlighting the necessity of accurate video understanding in the proposed task and the limitations of current agents. We hope Ego2Web can be a critical new resource for developing truly capable AI assistants that can seamlessly see, understand, and act across the physical and digital worlds.
>
---
#### [new 074] ViKey: Enhancing Temporal Understanding in Videos via Visual Prompting
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决视频大语言模型在减少帧数后出现的时序推理性能下降问题。通过引入视觉提示和关键词-帧映射模块，提升模型的时序理解能力。**

- **链接: [https://arxiv.org/pdf/2603.23186](https://arxiv.org/pdf/2603.23186)**

> **作者:** Yeonkyung Lee; Dayun Ju; Youngmin Kim; Seil Kang; Seong Jae Hwang
>
> **备注:** accepted to CVPR2026
>
> **摘要:** Recent advancements in Video Large Language Models (VideoLLMs) have enabled strong performance across diverse multimodal video tasks. To reduce the high computational cost of processing dense video frames, efficiency-oriented methods such as frame selection have been widely adopted. While effective at minimizing redundancy, these methods often cause notable performance drops on tasks requiring temporal reasoning. Unlike humans, who can infer event progression from sparse visual cues, VideoLLMs frequently misinterpret temporal relations when intermediate frames are omitted. To address this limitation, we explore visual prompting (VP) as a lightweight yet effective way to enhance temporal understanding in VideoLLMs. Our analysis reveals that simply annotating each frame with explicit ordinal information helps the model perceive temporal continuity. This visual cue also supports frame-level referencing and mitigates positional ambiguity within a sparsely sampled sequence. Building on these insights, we introduce ViKey, a training-free framework that combines VP with a lightweight Keyword-Frame Mapping (KFM) module. KFM leverages frame indices as dictionary-like keys to link textual cues to the most relevant frames, providing explicit temporal anchors during inference. Despite its simplicity, our approach substantially improves temporal reasoning and, on some datasets, preserves dense-frame baseline performance with as few as 20% of frames.
>
---
#### [new 075] Traffic Sign Recognition in Autonomous Driving: Dataset, Benchmark, and Field Experiment
- **分类: cs.CV**

- **简介: 该论文属于交通标志识别任务，解决跨区域泛化、罕见类别识别等问题。构建了TS-1M数据集和基准，评估不同模型在多种挑战下的表现。**

- **链接: [https://arxiv.org/pdf/2603.23034](https://arxiv.org/pdf/2603.23034)**

> **作者:** Guoyang Zhao; Weiqing Qi; Kai Zhang; Chenguang Zhang; Zeying Gong; Zhihai Bi; Kai Chen; Benshan Ma; Ming Liu; Jun Ma
>
> **摘要:** Traffic Sign Recognition (TSR) is a core perception capability for autonomous driving, where robustness to cross-region variation, long-tailed categories, and semantic ambiguity is essential for reliable real-world deployment. Despite steady progress in recognition accuracy, existing traffic sign datasets and benchmarks offer limited diagnostic insight into how different modeling paradigms behave under these practical challenges. We present TS-1M, a large-scale and globally diverse traffic sign dataset comprising over one million real-world images across 454 standardized categories, together with a diagnostic benchmark designed to analyze model capability boundaries. Beyond standard train-test evaluation, we provide a suite of challenge-oriented settings, including cross-region recognition, rare-class identification, low-clarity robustness, and semantic text understanding, enabling systematic and fine-grained assessment of modern TSR models. Using TS-1M, we conduct a unified benchmark across three representative learning paradigms: classical supervised models, self-supervised pretrained models, and multimodal vision-language models (VLMs). Our analysis reveals consistent paradigm-dependent behaviors, showing that semantic alignment is a key factor for cross-region generalization and rare-category recognition, while purely visual models remain sensitive to appearance shift and data imbalance. Finally, we validate the practical relevance of TS-1M through real-scene autonomous driving experiments, where traffic sign recognition is integrated with semantic reasoning and spatial localization to support map-level decision constraints. Overall, TS-1M establishes a reference-level diagnostic benchmark for TSR and provides principled insights into robust and semantic-aware traffic sign perception. Project page: this https URL.
>
---
#### [new 076] RealMaster: Lifting Rendered Scenes into Photorealistic Video
- **分类: cs.CV**

- **简介: 该论文提出RealMaster，解决视频生成中几何与真实感不一致的问题。通过结合3D引擎与扩散模型，提升渲染视频的逼真度，同时保持场景结构一致。**

- **链接: [https://arxiv.org/pdf/2603.23462](https://arxiv.org/pdf/2603.23462)**

> **作者:** Dana Cohen-Bar; Ido Sobol; Raphael Bensadoun; Shelly Sheynin; Oran Gafni; Or Patashnik; Daniel Cohen-Or; Amit Zohar
>
> **备注:** Project page: this https URL
>
> **摘要:** State-of-the-art video generation models produce remarkable photorealism, but they lack the precise control required to align generated content with specific scene requirements. Furthermore, without an underlying explicit geometry, these models cannot guarantee 3D consistency. Conversely, 3D engines offer granular control over every scene element and provide native 3D consistency by design, yet their output often remains trapped in the "uncanny valley". Bridging this sim-to-real gap requires both structural precision, where the output must exactly preserve the geometry and dynamics of the input, and global semantic transformation, where materials, lighting, and textures must be holistically transformed to achieve photorealism. We present RealMaster, a method that leverages video diffusion models to lift rendered video into photorealistic video while maintaining full alignment with the output of the 3D engine. To train this model, we generate a paired dataset via an anchor-based propagation strategy, where the first and last frames are enhanced for realism and propagated across the intermediate frames using geometric conditioning cues. We then train an IC-LoRA on these paired videos to distill the high-quality outputs of the pipeline into a model that generalizes beyond the pipeline's constraints, handling objects and characters that appear mid-sequence and enabling inference without requiring anchor frames. Evaluated on complex GTA-V sequences, RealMaster significantly outperforms existing video editing baselines, improving photorealism while preserving the geometry, dynamics, and identity specified by the original 3D control.
>
---
#### [new 077] OsteoFlow: Lyapunov-Guided Flow Distillation for Predicting Bone Remodeling after Mandibular Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于医学影像预测任务，旨在解决颌骨重建后长期骨重塑预测问题。提出OsteoFlow框架，通过Lyapunov引导的轨迹蒸馏方法，提升预测精度与 anatomical fidelity。**

- **链接: [https://arxiv.org/pdf/2603.22421](https://arxiv.org/pdf/2603.22421)**

> **作者:** Hamidreza Aftabi; Faye Yu; Brooke Switzer; Zachary Fishman; Eitan Prisman; Antony Hodgson; Cari Whyne; Sidney Fels; Michael Hardisty
>
> **摘要:** Predicting long-term bone remodeling after mandibular reconstruction would be of great clinical benefit, yet standard generative models struggle to maintain trajectory-level consistency and anatomical fidelity over long horizons. We introduce OsteoFlow, a flow-based framework predicting Year-1 post-operative CT scans from Day-5 scans. Our core contribution is Lyapunov-guided trajectory distillation: Unlike one-step distillation, our method distills a continuous trajectory over transport time from a registration-derived stationary velocity field teacher. Combined with a resection-aware image loss, this enforces geometric correspondence without sacrificing generative capacity. Evaluated on 344 paired regions of interest, OsteoFlow significantly outperforms state of-the-art baselines, reducing mean absolute error in the surgical resection zone by ~20%. This highlights the promise of trajectory distillation for long-term prediction. Code is available on GitHub: OsteoFlow.
>
---
#### [new 078] FixationFormer: Direct Utilization of Expert Gaze Trajectories for Chest X-Ray Classification
- **分类: cs.CV; cs.LG**

- **简介: 论文提出FixationFormer，将专家注视轨迹作为序列输入，用于胸部X光分类任务。解决传统方法难以直接利用注视数据的问题，通过Transformer结构实现更精细的诊断线索整合。**

- **链接: [https://arxiv.org/pdf/2603.22939](https://arxiv.org/pdf/2603.22939)**

> **作者:** Daniel Beckmann; Benjamin Risse
>
> **摘要:** Expert eye movements provide a rich, passive source of domain knowledge in radiology, offering a powerful cue for integrating diagnostic reasoning into computer-aided analysis. However, direct integration into CNN-based systems, which historically have dominated the medical image analysis domain, is challenging: gaze recordings are sequential, temporally dense yet spatially sparse, noisy, and variable across experts. As a consequence, most existing image-based models utilize reduced representations such as heatmaps. In contrast, gaze naturally aligns with transformer architectures, as both are sequential in nature and rely on attention to highlight relevant input regions. In this work, we introduce FixationFormer, a transformer-based architecture that represents expert gaze trajectories as sequences of tokens, thereby preserving their temporal and spatial structure. By modeling gaze sequences jointly with image features, our approach addresses sparsity and variability in gaze data while enabling a more direct and fine-grained integration of expert diagnostic cues through explicit cross-attention between the image and gaze token sequences. We evaluate our method on three publicly available benchmark chest X-ray datasets and demonstrate that it achieves state-of-the-art classification performance, highlighting the value of representing gaze as a sequence in transformer-based medical image analysis.
>
---
#### [new 079] Group Editing : Edit Multiple Images in One Go
- **分类: cs.CV**

- **简介: 该论文属于图像组编辑任务，旨在实现多张相关图像的一致修改。通过建立显式和隐式关系，提出GroupEditing框架解决跨视角一致性问题。**

- **链接: [https://arxiv.org/pdf/2603.22883](https://arxiv.org/pdf/2603.22883)**

> **作者:** Yue Ma; Xinyu Wang; Qianli Ma; Qinghe Wang; Mingzhe Zheng; Xiangpeng Yang; Hao Li; Chongbo Zhao; Jixuan Ying; Harry Yang; Hongyu Liu; Qifeng Chen
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** In this paper, we tackle the problem of performing consistent and unified modifications across a set of related images. This task is particularly challenging because these images may vary significantly in pose, viewpoint, and spatial layout. Achieving coherent edits requires establishing reliable correspondences across the images, so that modifications can be applied accurately to semantically aligned regions. To address this, we propose GroupEditing, a novel framework that builds both explicit and implicit relationships among images within a group. On the explicit side, we extract geometric correspondences using VGGT, which provides spatial alignment based on visual features. On the implicit side, we reformulate the image group as a pseudo-video and leverage the temporal coherence priors learned by pre-trained video models to capture latent relationships. To effectively fuse these two types of correspondences, we inject the explicit geometric cues from VGGT into the video model through a novel fusion mechanism. To support large-scale training, we construct GroupEditData, a new dataset containing high-quality masks and detailed captions for numerous image groups. Furthermore, to ensure identity preservation during editing, we introduce an alignment-enhanced RoPE module, which improves the model's ability to maintain consistent appearance across multiple images. Finally, we present GroupEditBench, a dedicated benchmark designed to evaluate the effectiveness of group-level image editing. Extensive experiments demonstrate that GroupEditing significantly outperforms existing methods in terms of visual quality, cross-view consistency, and semantic alignment.
>
---
#### [new 080] From Pixels to Semantics: A Multi-Stage AI Framework for Structural Damage Detection in Satellite Imagery
- **分类: cs.CV**

- **简介: 该论文属于灾害后建筑损伤检测任务，解决遥感图像语义解释不足的问题。通过AI框架提升图像分辨率并评估损伤等级，为应急响应提供支持。**

- **链接: [https://arxiv.org/pdf/2603.22768](https://arxiv.org/pdf/2603.22768)**

> **作者:** Bijay Shakya; Catherine Hoier; Khandaker Mamun Ahmed
>
> **摘要:** Rapid and accurate structural damage assessment following natural disasters is critical for effective emergency response and recovery. However, remote sensing imagery often suffers from low spatial resolution, contextual ambiguity, and limited semantic interpretability, reducing the reliability of traditional detection pipelines. In this work, we propose a novel hybrid framework that integrates AI-based super-resolution, deep learning object detection, and Vision-Language Models (VLMs) for comprehensive post-disaster building damage assessment. First, we enhance pre- and post-disaster satellite imagery using a Video Restoration Transformer (VRT) to upscale images from 1024x1024 to 4096x4096 resolution, improving structural detail visibility. Next, a YOLOv11-based detector localizes buildings in pre-disaster imagery, and cropped building regions are analyzed using VLMs to semantically assess structural damage across four severity levels. To ensure robust evaluation in the absence of ground-truth captions, we employ CLIPScore for reference-free semantic alignment and introduce a multi-model VLM-as-a-Jury strategy to reduce individual model bias in safety-critical decision making. Experiments on subsets of the xBD dataset, including the Moore Tornado and Hurricane Matthew events, demonstrate that the proposed framework enhances the semantic interpretation of damaged buildings. In addition, our framework provides helpful recommendations to first responders for recovery based on damage analysis.
>
---
#### [new 081] GeoSANE: Learning Geospatial Representations from Models, Not Data
- **分类: cs.CV**

- **简介: 该论文提出GeoSANE，解决多模型知识统一问题，通过生成共享表示提升遥感任务性能。**

- **链接: [https://arxiv.org/pdf/2603.23408](https://arxiv.org/pdf/2603.23408)**

> **作者:** Joelle Hanna; Damian Falk; Stella X. Yu; Damian Borth
>
> **摘要:** Recent advances in remote sensing have led to an increase in the number of available foundation models; each trained on different modalities, datasets, and objectives, yet capturing only part of the vast geospatial knowledge landscape. While these models show strong results within their respective domains, their capabilities remain complementary rather than unified. Therefore, instead of choosing one model over another, we aim to combine their strengths into a single shared representation. We introduce GeoSANE, a geospatial model foundry that learns a unified neural representation from the weights of existing foundation models and task-specific models, able to generate novel neural networks weights on-demand. Given a target architecture, GeoSANE generates weights ready for finetuning for classification, segmentation, and detection tasks across multiple modalities. Models generated by GeoSANE consistently outperform their counterparts trained from scratch, match or surpass state-of-the-art remote sensing foundation models, and outperform models obtained through pruning or knowledge distillation when generating lightweight networks. Evaluations across ten diverse datasets and on GEO-Bench confirm its strong generalization capabilities. By shifting from pre-training to weight generation, GeoSANE introduces a new framework for unifying and transferring geospatial knowledge across models and tasks. Code is available at \href{this https URL}{this http URL}.
>
---
#### [new 082] It Takes Two: A Duet of Periodicity and Directionality for Burst Flicker Removal
- **分类: cs.CV**

- **简介: 该论文属于图像去噪任务，解决短曝光摄影中的闪烁伪影问题。提出Flickerformer架构，结合周期性和方向性特征，有效去除闪烁且避免鬼影。**

- **链接: [https://arxiv.org/pdf/2603.22794](https://arxiv.org/pdf/2603.22794)**

> **作者:** Lishen Qu; Shihao Zhou; Jie Liang; Hui Zeng; Lei Zhang; Jufeng Yang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Flicker artifacts, arising from unstable illumination and row-wise exposure inconsistencies, pose a significant challenge in short-exposure photography, severely degrading image quality. Unlike typical artifacts, e.g., noise and low-light, flicker is a structured degradation with specific spatial-temporal patterns, which are not accounted for in current generic restoration frameworks, leading to suboptimal flicker suppression and ghosting artifacts. In this work, we reveal that flicker artifacts exhibit two intrinsic characteristics, periodicity and directionality, and propose Flickerformer, a transformer-based architecture that effectively removes flicker without introducing ghosting. Specifically, Flickerformer comprises three key components: a phase-based fusion module (PFM), an autocorrelation feed-forward network (AFFN), and a wavelet-based directional attention module (WDAM). Based on the periodicity, PFM performs inter-frame phase correlation to adaptively aggregate burst features, while AFFN exploits intra-frame structural regularities through autocorrelation, jointly enhancing the network's ability to perceive spatially recurring patterns. Moreover, motivated by the directionality of flicker artifacts, WDAM leverages high-frequency variations in the wavelet domain to guide the restoration of low-frequency dark regions, yielding precise localization of flicker artifacts. Extensive experiments demonstrate that Flickerformer outperforms state-of-the-art approaches in both quantitative metrics and visual quality. The source code is available at this https URL.
>
---
#### [new 083] ForeSea: AI Forensic Search with Multi-modal Queries for Video Surveillance
- **分类: cs.CV**

- **简介: 该论文提出ForeSea系统，解决视频监控中多模态查询的精准检索问题。通过构建ForeSeaQA基准，提升视频问答与时间定位的准确性。**

- **链接: [https://arxiv.org/pdf/2603.22872](https://arxiv.org/pdf/2603.22872)**

> **作者:** Hyojin Park; Yi Li; Janghoon Cho; Sungha Choi; Jungsoo Lee; Taotao Jing; Shuai Zhang; Munawar Hayat; Dashan Gao; Ning Bi; Fatih Porikli
>
> **摘要:** Despite decades of work, surveillance still struggles to find specific targets across long, multi-camera video. Prior methods -- tracking pipelines, CLIP based models, and VideoRAG -- require heavy manual filtering, capture only shallow attributes, and fail at temporal reasoning. Real-world searches are inherently multimodal (e.g., "When does this person join the fight?" with the person's image), yet this setting remains underexplored. Also, there are no proper benchmarks to evaluate those setting - asking video with multimodal queries. To address this gap, we introduce ForeSeaQA, a new benchmark specifically designed for video QA with image-and-text queries and timestamped annotations of key events. The dataset consists of long-horizon surveillance footage paired with diverse multimodal questions, enabling systematic evaluation of retrieval, temporal grounding, and multimodal reasoning in realistic forensic conditions. Not limited to this benchmark, we propose ForeSea, an AI forensic search system with a 3-stage, plug-and-play pipeline. (1) A tracking module filters irrelevant footage; (2) a multimodal embedding module indexes the remaining clips; and (3) during inference, the system retrieves top-K candidate clips for a Video Large Language Model (VideoLLM) to answer queries and localize events. On ForeSeaQA, ForeSea improves accuracy by 3.5% and temporal IoU by 11.0 over prior VideoRAG models. To our knowledge, ForeSeaQA is the first benchmark to support complex multimodal queries with precise temporal grounding, and ForeSea is the first VideoRAG system built to excel in this setting.
>
---
#### [new 084] FHAvatar: Fast and High-Fidelity Reconstruction of Face-and-Hair Composable 3D Head Avatar from Few Casual Captures
- **分类: cs.CV**

- **简介: 该论文提出FHAvatar，用于从少量视角图像中快速重建高保真可组合的3D人脸与发型头像，解决传统方法依赖多视角或高成本优化的问题。**

- **链接: [https://arxiv.org/pdf/2603.23345](https://arxiv.org/pdf/2603.23345)**

> **作者:** Yujie Sun; Zhuoqiang Cai; Chaoyue Niu; Jianchuan Chen; Zhiwen Chen; Chengfei Lv; Fan Wu
>
> **摘要:** We present FHAvatar, a novel framework for reconstructing 3D Gaussian avatars with composable face and hair components from an arbitrary number of views. Unlike previous approaches that couple facial and hair representations within a unified modeling process, we explicitly decouple two components in texture space by representing the face with planar Gaussians and the hair with strand-based Gaussians. To overcome the limitations of existing methods that rely on dense multi-view captures or costly per-identity optimization, we propose an aggregated transformer backbone to learn geometry-aware cross-view priors and head-hair structural coherence from multi-view datasets, enabling effective and efficient feature extraction and fusion from few casual captures. Extensive quantitative and qualitative experiments demonstrate that FHAvatar achieves state-of-the-art reconstruction quality from only a few observations of new identities within minutes, while supporting real-time animation, convenient hairstyle transfer, and stylized editing, broadening the accessibility and applicability of digital avatar creation.
>
---
#### [new 085] MultiCam: On-the-fly Multi-Camera Pose Estimation Using Spatiotemporal Overlaps of Known Objects
- **分类: cs.CV**

- **简介: 论文提出MultiCam方法，用于实时多相机位姿估计，解决传统标记跟踪的局限性。通过利用已知物体的时空重叠信息，提升AR应用中的相机定位精度。**

- **链接: [https://arxiv.org/pdf/2603.22839](https://arxiv.org/pdf/2603.22839)**

> **作者:** Shiyu Li; Hannah Schieber; Kristoffer Waldow; Benjamin Busam; Julian Kreimeier; Daniel Roth
>
> **摘要:** Multi-camera dynamic Augmented Reality (AR) applications require a camera pose estimation to leverage individual information from each camera in one common system. This can be achieved by combining contextual information, such as markers or objects, across multiple views. While commonly cameras are calibrated in an initial step or updated through the constant use of markers, another option is to leverage information already present in the scene, like known objects. Another downside of marker-based tracking is that markers have to be tracked inside the field-of-view (FoV) of the cameras. To overcome these limitations, we propose a constant dynamic camera pose estimation leveraging spatiotemporal FoV overlaps of known objects on the fly. To achieve that, we enhance the state-of-the-art object pose estimator to update our spatiotemporal scene graph, enabling a relation even among non-overlapping FoV cameras. To evaluate our approach, we introduce a multi-camera, multi-object pose estimation dataset with temporal FoV overlap, including static and dynamic cameras. Furthermore, in FoV overlapping scenarios, we outperform the state-of-the-art on the widely used YCB-V and T-LESS dataset in camera pose accuracy. Our performance on both previous and our proposed datasets validates the effectiveness of our marker-less approach for AR applications. The code and dataset are available on this https URL.
>
---
#### [new 086] UrbanVGGT: Scalable Sidewalk Width Estimation from Street View Images
- **分类: cs.CV**

- **简介: 该论文属于城市基础设施测量任务，旨在解决 sidewalk 宽度数据稀缺问题。通过分析街景图像，提出 UrbanVGGT 方法，实现精准宽度估计。**

- **链接: [https://arxiv.org/pdf/2603.22531](https://arxiv.org/pdf/2603.22531)**

> **作者:** Kaizhen Tan; Fan Zhang
>
> **摘要:** Sidewalk width is an important indicator of pedestrian accessibility, comfort, and network quality, yet large-scale width data remain scarce in most cities. Existing approaches typically rely on costly field surveys, high-resolution overhead imagery, or simplified geometric assumptions that limit scalability or introduce systematic error. To address this gap, we present UrbanVGGT, a measurement pipeline for estimating metric sidewalk width from a single street-view image. The method combines semantic segmentation, feed-forward 3D reconstruction, adaptive ground-plane fitting, camera-height-based scale calibration, and directional width measurement on the recovered plane. On a ground-truth benchmark from Washington, D.C., UrbanVGGT achieves a mean absolute error of 0.252 m, with 95.5% of estimates within 0.50 m of the reference width. Ablation experiments show that metric scale calibration is the most critical component, and controlled comparisons with alternative geometry backbones support the effectiveness of the overall design. As a feasibility demonstration, we further apply the pipeline to three cities and generate SV-SideWidth, a prototype sidewalk-width dataset covering 527 OpenStreetMap street segments. The results indicate that street-view imagery can support scalable generation of candidate sidewalk-width attributes, while broader cross-city validation and local ground-truth auditing remain necessary before deployment as authoritative planning data.
>
---
#### [new 087] YOLOv10 with Kolmogorov-Arnold networks and vision-language foundation models for interpretable object detection and trustworthy multimodal AI in computer vision perception
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文属于目标检测任务，旨在解决自动驾驶中模型置信度不透明的问题。通过结合Kolmogorov-Arnold网络和YOLOv10，提升检测结果的可解释性和可信度。**

- **链接: [https://arxiv.org/pdf/2603.23037](https://arxiv.org/pdf/2603.23037)**

> **作者:** Marios Impraimakis; Daniel Vazquez; Feiyu Zhou
>
> **备注:** 14 pages, 23 Figures, 6 Tables
>
> **摘要:** The interpretable object detection capabilities of a novel Kolmogorov-Arnold network framework are examined here. The approach refers to a key limitation in computer vision for autonomous vehicles perception, and beyond. These systems offer limited transparency regarding the reliability of their confidence scores in visually degraded or ambiguous scenes. To address this limitation, a Kolmogorov-Arnold network is employed as an interpretable post-hoc surrogate to model the trustworthiness of the You Only Look Once (Yolov10) detections using seven geometric and semantic features. The additive spline-based structure of the Kolmogorov-Arnold network enables direct visualisation of each feature's influence. This produces smooth and transparent functional mappings that reveal when the model's confidence is well supported and when it is unreliable. Experiments on both Common Objects in Context (COCO), and images from the University of Bath campus demonstrate that the framework accurately identifies low-trust predictions under blur, occlusion, or low texture. This provides actionable insights for filtering, review, or downstream risk mitigation. Furthermore, a bootstrapped language-image (BLIP) foundation model generates descriptive captions of each scene. This tool enables a lightweight multimodal interface without affecting the interpretability layer. The resulting system delivers interpretable object detection with trustworthy confidence estimates. It offers a powerful tool for transparent and practical perception component for autonomous and multimodal artificial intelligence applications.
>
---
#### [new 088] ENC-Bench: A Benchmark for Evaluating Multimodal Large Language Models in Electronic Navigational Chart Understanding
- **分类: cs.CV**

- **简介: 该论文属于多模态语言模型在电子海图理解任务中的评估。旨在解决MLLMs在专业海图解读上的可靠性问题，构建了首个ENC-Bench基准并测试了多种模型。**

- **链接: [https://arxiv.org/pdf/2603.22763](https://arxiv.org/pdf/2603.22763)**

> **作者:** Ao Cheng; Xingming Li; Xuanyu Ji; Xixiang He; Qiyao Sun; Chunping Qiu; Runke Huang; Qingyong Hu
>
> **备注:** Accepted to CVPR 2026, Project page: this https URL
>
> **摘要:** Electronic Navigational Charts (ENCs) are the safety-critical backbone of modern maritime navigation, yet it remains unclear whether multimodal large language models (MLLMs) can reliably interpret them. Unlike natural images or conventional charts, ENCs encode regulations, bathymetry, and route constraints via standardized vector symbols, scale-dependent rendering, and precise geometric structure -- requiring specialized maritime expertise for interpretation. We introduce ENC-Bench, the first benchmark dedicated to professional ENC understanding. ENC-Bench contains 20,490 expert-validated samples from 840 authentic National Oceanic and Atmospheric Administration (NOAA) ENCs, organized into a three-level hierarchy: Perception (symbol and feature recognition), Spatial Reasoning (coordinate localization, bearing, distance), and Maritime Decision-Making (route legality, safety assessment, emergency planning under multiple constraints). All samples are generated from raw S-57 data through a calibrated vector-to-image pipeline with automated consistency checks and expert review. We evaluate 10 state-of-the-art MLLMs such as GPT-4o, Gemini 2.5, Qwen3-VL, InternVL-3, and GLM-4.5V, under a unified zero-shot protocol. The best model achieves only 47.88% accuracy, with systematic challenges in symbolic grounding, spatial computation, multi-constraint reasoning, and robustness to lighting and scale variations. By establishing the first rigorous ENC benchmark, we open a new research frontier at the intersection of specialized symbolic reasoning and safety-critical AI, providing essential infrastructure for advancing MLLMs toward professional maritime applications.
>
---
#### [new 089] TDATR: Improving End-to-End Table Recognition via Table Detail-Aware Learning and Cell-Level Visual Alignment
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于表格识别任务，解决传统方法结构与内容分离、数据依赖性强的问题。提出TDATR模型，通过细节感知学习和单元格对齐提升端到端表格识别效果。**

- **链接: [https://arxiv.org/pdf/2603.22819](https://arxiv.org/pdf/2603.22819)**

> **作者:** Chunxia Qin; Chenyu Liu; Pengcheng Xia; Jun Du; Baocai Yin; Bing Yin; Cong Liu
>
> **备注:** Acceptd by CVPR 2026. Project Page: this https URL
>
> **摘要:** Tables are pervasive in diverse documents, making table recognition (TR) a fundamental task in document analysis. Existing modular TR pipelines separately model table structure and content, leading to suboptimal integration and complex workflows. End-to-end approaches rely heavily on large-scale TR data and struggle in data-constrained scenarios. To address these issues, we propose TDATR (Table Detail-Aware Table Recognition) improves end-to-end TR through table detail-aware learning and cell-level visual alignment. TDATR adopts a ``perceive-then-fuse'' strategy. The model first performs table detail-aware learning to jointly perceive table structure and content through multiple structure understanding and content recognition tasks designed under a language modeling paradigm. These tasks can naturally leverage document data from diverse scenarios to enhance model robustness. The model then integrates implicit table details to generate structured HTML outputs, enabling more efficient TR modeling when trained with limited data. Furthermore, we design a structure-guided cell localization module integrated into the end-to-end TR framework, which efficiently locates cell and strengthens vision-language alignment. It enhances the interpretability and accuracy of TR. We achieve state-of-the-art or highly competitive performance on seven benchmarks without dataset-specific fine-tuning.
>
---
#### [new 090] Curriculum-Driven 3D CT Report Generation via Language-Free Visual Grafting and Zone-Constrained Compression
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学影像报告生成任务，解决3D CT图像生成准确报告的问题。通过自监督视觉编码和课程学习框架，提升报告质量。**

- **链接: [https://arxiv.org/pdf/2603.23308](https://arxiv.org/pdf/2603.23308)**

> **作者:** V. K. Cody Bumgardner; Mitchell A. Klusty; Mahmut S. Gokmen; Evan W. Damron
>
> **备注:** 10 pages, 2 figures
>
> **摘要:** Automated radiology report generation from 3D computed tomography (CT) volumes is challenging due to extreme sequence lengths, severe class imbalance, and the tendency of large language models (LLMs) to ignore visual tokens in favor of linguistic priors. We present Ker-VLJEPA-3B, a four-phase curriculum learning framework for free-text report generation from thoracic CT volumes. A phased training curriculum progressively adapts a Llama 3.2 3B decoder to ground its output in visual features from a frozen, self-supervised encoder. Our visual backbone (LeJEPA ViT-Large) is trained via self-supervised joint-embedding prediction on unlabeled CTs, without text supervision. Unlike contrastive models (CLIP, BiomedCLIP), this language-free backbone yields modality-pure representations. Vision-language alignment is deferred to the curriculum's bridge and generation phases. This modality-agnostic design can integrate any self-supervised encoder into an LLM without paired text during foundation training. Methodological innovations include: (1) zone-constrained cross-attention compressing slice embeddings into 32 spatially-grounded visual tokens; (2) PCA whitening of anisotropic LLM embeddings; (3) a positive-findings-only strategy eliminating posterior collapse; (4) warm bridge initialization transferring projection weights; and (5) selective cross-attention freezing with elastic weight consolidation to prevent catastrophic forgetting. Evaluated on the CT-RATE benchmark (2,984 validation volumes, 18 classes), Ker-VLJEPA-3B achieves a macro F1 of 0.429, surpassing the state-of-the-art (U-VLM, macro F1 = 0.414) by 3.6%, and reaching 0.448 (+8.2%) with threshold optimization. Ablation studies confirm 56.6% of generation quality derives from patient-specific visual content. Code and weights are available.
>
---
#### [new 091] PoseDriver: A Unified Approach to Multi-Category Skeleton Detection for Autonomous Driving
- **分类: cs.CV; cs.HC**

- **简介: 该论文提出PoseDriver，解决自动驾驶中多类别骨骼检测问题，通过统一框架处理多种目标，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.23215](https://arxiv.org/pdf/2603.23215)**

> **作者:** Yasamin Borhani; Taylor Mordan; Yihan Wang; Reyhaneh Hosseininejad; Javad Khoramdel; Alexandre Alahi
>
> **摘要:** Object skeletons offer a concise representation of structural information, capturing essential aspects of posture and orientation that are crucial for autonomous driving applications. However, a unified architecture that simultaneously handles multiple instances and categories using only the input image remains elusive. In this paper, we introduce PoseDriver, a unified framework for bottom-up multi-category skeleton detection tailored to common objects in driving scenarios. We model each category as a distinct task to systematically address the challenges of multi-task learning. Specifically, we propose a novel approach for lane detection based on skeleton representations, achieving state-of-the-art performance on the OpenLane dataset. Moreover, we present a new dataset for bicycle skeleton detection and assess the transferability of our framework to novel categories. Experimental results validate the effectiveness of the proposed approach.
>
---
#### [new 092] SpecEyes: Accelerating Agentic Multimodal LLMs via Speculative Perception and Planning
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出SpecEyes，解决agentic MLLMs的序列延迟问题，通过推测规划和并行加速提升效率与吞吐量。**

- **链接: [https://arxiv.org/pdf/2603.23483](https://arxiv.org/pdf/2603.23483)**

> **作者:** Haoyu Huang; Jinfa Huang; Zhongwei Wan; Xiawu Zheng; Rongrong Ji; Jiebo Luo
>
> **备注:** Code: this https URL
>
> **摘要:** Agentic multimodal large language models (MLLMs) (e.g., OpenAI o3 and Gemini Agentic Vision) achieve remarkable reasoning capabilities through iterative visual tool invocation. However, the cascaded perception, reasoning, and tool-calling loops introduce significant sequential overhead. This overhead, termed agentic depth, incurs prohibitive latency and seriously limits system-level concurrency. To this end, we propose SpecEyes, an agentic-level speculative acceleration framework that breaks this sequential bottleneck. Our key insight is that a lightweight, tool-free MLLM can serve as a speculative planner to predict the execution trajectory, enabling early termination of expensive tool chains without sacrificing accuracy. To regulate this speculative planning, we introduce a cognitive gating mechanism based on answer separability, which quantifies the model's confidence for self-verification without requiring oracle labels. Furthermore, we design a heterogeneous parallel funnel that exploits the stateless concurrency of the small model to mask the stateful serial execution of the large model, maximizing system throughput. Extensive experiments on V* Bench, HR-Bench, and POPE demonstrate that SpecEyes achieves 1.1-3.35x speedup over the agentic baseline while preserving or even improving accuracy (up to +6.7%), thereby boosting serving throughput under concurrent workloads.
>
---
#### [new 093] Typography-Based Monocular Distance Estimation Framework for Vehicle Safety Systems
- **分类: cs.CV**

- **简介: 该论文属于视觉距离估计任务，解决单目视觉的尺度模糊问题。通过车牌字体特征实现精准距离估算，提升车辆安全系统性能。**

- **链接: [https://arxiv.org/pdf/2603.22781](https://arxiv.org/pdf/2603.22781)**

> **作者:** Manognya Lokesh Reddy; Zheng Liu
>
> **备注:** 25 pages, 11 figures
>
> **摘要:** Accurate inter-vehicle distance estimation is a cornerstone of advanced driver assistance systems and autonomous driving. While LiDAR and radar provide high precision, their cost prohibits widespread adoption in mass-market vehicles. Monocular vision offers a low-cost alternative but suffers from scale ambiguity and sensitivity to environmental disturbances. This paper introduces a typography-based monocular distance estimation framework, which exploits the standardized typography of license plates as passive fiducial markers for metric distance estimation. The core geometric module uses robust plate detection and character segmentation to measure character height and computes distance via the pinhole camera model. The system incorporates interactive calibration, adaptive detection with strict and permissive modes, and multi-method character segmentation leveraging both adaptive and global thresholding. To enhance robustness, the framework further includes camera pose compensation using lane-based horizon estimation, hybrid deep-learning fusion, temporal Kalman filtering for velocity estimation, and multi-feature fusion that exploits additional typographic cues such as stroke width, character spacing, and plate border thickness. Experimental validation with a calibrated monocular camera in a controlled indoor setup achieved a coefficient of variation of 2.3% in character height across consecutive frames and a mean absolute error of 7.7%. The framework operates without GPU acceleration, demonstrating real-time feasibility. A comprehensive comparison with a plate-width based method shows that character-based ranging reduces the standard deviation of estimates by 35%, translating to smoother, more consistent distance readings in practice, where erratic estimates could trigger unnecessary braking or acceleration.
>
---
#### [new 094] Tiny Inference-Time Scaling with Latent Verifiers
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文属于生成模型优化任务，解决推理阶段效率低的问题。提出VHS verifier，在扩散模型中间层验证，提升效率并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2603.22492](https://arxiv.org/pdf/2603.22492)**

> **作者:** Davide Bucciarelli; Evelyn Turri; Lorenzo Baraldi; Marcella Cornia; Lorenzo Baraldi; Rita Cucchiara
>
> **摘要:** Inference-time scaling has emerged as an effective way to improve generative models at test time by using a verifier to score and select candidate outputs. A common choice is to employ Multimodal Large Language Models (MLLMs) as verifiers, which can improve performance but introduce substantial inference-time cost. Indeed, diffusion pipelines operate in an autoencoder latent space to reduce computation, yet MLLM verifiers still require decoding candidates to pixel space and re-encoding them into the visual embedding space, leading to redundant and costly operations. In this work, we propose Verifier on Hidden States (VHS), a verifier that operates directly on intermediate hidden representations of Diffusion Transformer (DiT) single-step generators. VHS analyzes generator features without decoding to pixel space, thereby reducing the per-candidate verification cost while improving or matching the performance of MLLM-based competitors. We show that, under tiny inference budgets with only a small number of candidates per prompt, VHS enables more efficient inference-time scaling reducing joint generation-and-verification time by 63.3%, compute FLOPs by 51% and VRAM usage by 14.5% with respect to a standard MLLM verifier, achieving a +2.7% improvement on GenEval at the same inference-time budget.
>
---
#### [new 095] 3DCity-LLM: Empowering Multi-modality Large Language Models for 3D City-scale Perception and Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出3DCity-LLM，解决3D城市尺度下多模态语言模型的感知与理解问题，通过统一框架和高质量数据集提升城市场景分析能力。**

- **链接: [https://arxiv.org/pdf/2603.23447](https://arxiv.org/pdf/2603.23447)**

> **作者:** Yiping Chen; Jinpeng Li; Wenyu Ke; Yang Luo; Jie Ouyang; Zhongjie He; Li Liu; Hongchao Fan; Hao Wu
>
> **备注:** 24 pages, 11 figures, 12 tables
>
> **摘要:** While multi-modality large language models excel in object-centric or indoor scenarios, scaling them to 3D city-scale environments remains a formidable challenge. To bridge this gap, we propose 3DCity-LLM, a unified framework designed for 3D city-scale vision-language perception and understanding. 3DCity-LLM employs a coarse-to-fine feature encoding strategy comprising three parallel branches for target object, inter-object relationship, and global scene. To facilitate large-scale training, we introduce 3DCity-LLM-1.2M dataset that comprises approximately 1.2 million high-quality samples across seven representative task categories, ranging from fine-grained object analysis to multi-faceted scene planning. This strictly quality-controlled dataset integrates explicit 3D numerical information and diverse user-oriented simulations, enriching the question-answering diversity and realism of urban scenarios. Furthermore, we apply a multi-dimensional protocol based on text-similarity metrics and LLM-based semantic assessment to ensure faithful and comprehensive evaluations for all methods. Extensive experiments on two benchmarks demonstrate that 3DCity-LLM significantly outperforms existing state-of-the-art methods, offering a promising and meaningful direction for advancing spatial reasoning and urban intelligence. The source code and dataset are available at this https URL.
>
---
#### [new 096] VoDaSuRe: A Large-Scale Dataset Revealing Domain Shift in Volumetric Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于体积超分辨率任务，旨在解决数据域偏移问题。通过构建VoDaSuRe数据集，揭示现有方法在真实低分辨率数据上的性能下降原因。**

- **链接: [https://arxiv.org/pdf/2603.23153](https://arxiv.org/pdf/2603.23153)**

> **作者:** August Leander Høeg; Sophia Wiinberg Bardenfleth; Hans Martin Kjer; Tim Bjørn Dyrby; Vedrana Andersen Dahl; Anders Bjorholm Dahl
>
> **备注:** 18 pages, 15 figures. To be published in the proceedings of the Computer Vision and Pattern Recognition Conference 2026
>
> **摘要:** Recent advances in volumetric super-resolution (SR) have demonstrated strong performance in medical and scientific imaging, with transformer- and CNN-based approaches achieving impressive results even at extreme scaling factors. In this work, we show that much of this performance stems from training on downsampled data rather than real low-resolution scans. This reliance on downsampling is partly driven by the scarcity of paired high- and low-resolution 3D datasets. To address this, we introduce VoDaSuRe, a large-scale volumetric dataset containing paired high- and low-resolution scans. When training models on VoDaSuRe, we reveal a significant discrepancy: SR models trained on downsampled data produce substantially sharper predictions than those trained on real low-resolution scans, which smooth fine structures. Conversely, applying models trained on downsampled data to real scans preserves more structure but is inaccurate. Our findings suggest that current SR methods are overstated - when applied to real data, they do not recover structures lost in low-resolution scans and instead predict a smoothed average. We argue that progress in deep learning-based volumetric SR requires datasets with paired real scans of high complexity, such as VoDaSuRe. Our dataset and code are publicly available through: this https URL
>
---
#### [new 097] UniFunc3D: Unified Active Spatial-Temporal Grounding for 3D Functionality Segmentation
- **分类: cs.CV**

- **简介: 该论文属于3D功能分割任务，解决自然语言指令到精细交互元素的定位问题。提出UniFunc3D框架，通过统一的时空推理实现高效精准分割。**

- **链接: [https://arxiv.org/pdf/2603.23478](https://arxiv.org/pdf/2603.23478)**

> **作者:** Jiaying Lin; Dan Xu
>
> **摘要:** Functionality segmentation in 3D scenes requires an agent to ground implicit natural-language instructions into precise masks of fine-grained interactive elements. Existing methods rely on fragmented pipelines that suffer from visual blindness during initial task parsing. We observe that these methods are limited by single-scale, passive and heuristic frame selection. We present UniFunc3D, a unified and training-free framework that treats the multimodal large language model as an active observer. By consolidating semantic, temporal, and spatial reasoning into a single forward pass, UniFunc3D performs joint reasoning to ground task decomposition in direct visual evidence. Our approach introduces active spatial-temporal grounding with a coarse-to-fine strategy. This allows the model to select correct video frames adaptively and focus on high-detail interactive parts while preserving the global context necessary for disambiguation. On SceneFun3D, UniFunc3D achieves state-of-the-art performance, surpassing both training-free and training-based methods by a large margin with a relative 59.9\% mIoU improvement, without any task-specific training. Code will be released on our project page: this https URL.
>
---
#### [new 098] An Explainable AI-Driven Framework for Automated Brain Tumor Segmentation Using an Attention-Enhanced U-Net
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决脑肿瘤自动分割难题。通过改进的U-Net模型和注意力机制，提升分割精度与可解释性。**

- **链接: [https://arxiv.org/pdf/2603.23344](https://arxiv.org/pdf/2603.23344)**

> **作者:** MD Rashidul Islam; Bakary Gibba
>
> **摘要:** Computer-aided segmentation of brain tumors from MRI data is of crucial significance to clinical decision-making in diagnosis, treatment planning, and follow-up disease monitoring. Gliomas, owing to their high malignancy and heterogeneity, represent a very challenging task for accurate and reliable segmentation into intra-tumoral sub-regions. Manual segmentation is typically time-consuming and not reliable, which justifies the need for robust automated this http URL research resolves this problem by leveraging the BraTS 2020 dataset, where we have labeled MRI scans of glioma patients with four significant classes: background/healthy tissue, necrotic/non-enhancing core, edema, and enhancing tumor. In this work, we present a new segmentation technique based on a U-Net model augmented with executed attention gates to focus on the most significant regions of images. To counter class imbalance, we employ manually designed loss functions like Dice Loss and Categorical Dice Loss, in conjunction with standard categorical cross-entropy. Other evaluation metrics, like sensitivity and specificity, were used to measure discriminability of the model between tumor classes. Besides, we introduce Grad-CAM-based explainable AI to enable visualizing attention regions and improve model interpretability, together with a smooth heatmap generation technique through Gaussian filtering. Our approach achieved superior performance with accuracy of 0.9919, Dice coefficient of 0.9901, mean IoU of 0.9873, sensitivity of 0.9908, and specificity of 0.9974. This study demonstrates that the use of attention mechanisms, personalized loss functions, and explainable AI significantly improves highly complex tumor structure segmentation precision in MRI scans, providing a reliable and explainable method for clinical applications.
>
---
#### [new 099] High Resolution Flood Extent Detection Using Deep Learning with Random Forest Derived Training Labels
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于洪水范围检测任务，旨在解决数据稀缺和云覆盖问题。通过结合光学影像与地形特征，利用深度学习方法进行洪水映射。**

- **链接: [https://arxiv.org/pdf/2603.22518](https://arxiv.org/pdf/2603.22518)**

> **作者:** Azizbek Nuriddinov; Ebrahim Ahmadisharaf; Mohammad Reza Alizadeh
>
> **备注:** Accepted to IGARSS 2026
>
> **摘要:** Validation of flood models, used to support risk mitigation strategies, remains challenging due to limited observations during extreme events. High-frequency, high-resolution optical imagery (~3 m), such as PlanetScope, offers new opportunities for flood mapping, although applications remain limited by cloud cover and the lack of labeled training data during disasters. To address this, we develop a flood mapping framework that integrates PlanetScope optical imagery with topographic features using machine learning (ML) and deep learning (DL) algorithms. A Random Forest model was applied to expert-annotated flood masks to generate training labels for DL models, U-Net. Two U-Net models with ResNet18 backbone were trained using optical imagery only (4 bands) and optical imagery combined with Height Above Nearest Drainage (HAND) and topographic slope (6 bands). Hurricane Ida (September 2021), which caused catastrophic flooding across the eastern United States, including the New York City metropolitan area, was used as an example to evaluate the framework. Results demonstrate that the U-Net model with topographic features achieved very close performance to the optical-only configuration (F1=0.92 and IoU=0.85 by both modeling scenarios), indicating that HAND and slope provide only marginal value to inundation extent detection. The proposed framework offers a scalable and label-efficient approach for mapping inundation extent that enables modeling under data-scarce flood scenarios.
>
---
#### [new 100] Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于训练-free 开放词汇语义分割任务，解决滑动窗口导致的语义不一致问题。提出GLA-CLIP框架，实现跨窗口信息融合与动态归一化，提升分割效果。**

- **链接: [https://arxiv.org/pdf/2603.23030](https://arxiv.org/pdf/2603.23030)**

> **作者:** ByeongCheol Lee; Hyun Seok Seong; Sangeek Hyun; Gilhan Park; WonJun Moon; Jae-Pil Heo
>
> **备注:** 18 pages, 13 figures, 12 tables, Accepted to CVPR 2026
>
> **摘要:** A sliding-window inference strategy is commonly adopted in recent training-free open-vocabulary semantic segmentation methods to overcome limitation of the CLIP in processing high-resolution images. However, this approach introduces a new challenge: each window is processed independently, leading to semantic discrepancy across windows. To address this issue, we propose Global-Local Aligned CLIP~(GLA-CLIP), a framework that facilitates comprehensive information exchange across windows. Rather than limiting attention to tokens within individual windows, GLA-CLIP extends key-value tokens to incorporate contextual cues from all windows. Nevertheless, we observe a window bias: outer-window tokens are less likely to be attended, since query features are produced through interactions within the inner window patches, thereby lacking semantic grounding beyond their local context. To mitigate this, we introduce a proxy anchor, constructed by aggregating tokens highly similar to the given query from all windows, which provides a unified semantic reference for measuring similarity across both inner- and outer-window patches. Furthermore, we propose a dynamic normalization scheme that adjusts attention strength according to object scale by dynamically scaling and thresholding the attention map to cope with small-object scenarios. Moreover, GLA-CLIP can be equipped on existing methods and broad their receptive field. Extensive experiments validate the effectiveness of GLA-CLIP in enhancing training-free open-vocabulary semantic segmentation performance. Code is available at this https URL.
>
---
#### [new 101] PIVM: Diffusion-Based Prior-Integrated Variation Modeling for Anatomically Precise Abdominal CT Synthesis
- **分类: cs.CV**

- **简介: 该论文属于医学图像合成任务，旨在解决腹部CT数据不足的问题。提出PIVM框架，通过扩散模型结合先验知识生成精确的CT图像。**

- **链接: [https://arxiv.org/pdf/2603.22626](https://arxiv.org/pdf/2603.22626)**

> **作者:** Dinglun He; Baoming Zhang; Xu Wang; Yao Hao; Deshan Yang; Ye Duan
>
> **备注:** Accepted at the IEEE International Symposium on Biomedical Imaging (ISBI) 2026 (Oral). Equal contribution by the first three authors
>
> **摘要:** Abdominal CT data are limited by high annotation costs and privacy constraints, which hinder the development of robust segmentation and diagnostic models. We present a Prior-Integrated Variation Modeling (PIVM) framework, a diffusion-based method for anatomically accurate CT image synthesis. Instead of generating full images from noise, PIVM predicts voxel-wise intensity variations relative to organ-specific intensity priors derived from segmentation labels. These priors and labels jointly guide the diffusion process, ensuring spatial alignment and realistic organ boundaries. Unlike latent-space diffusion models, our approach operates directly in image space while preserving the full Hounsfield Unit (HU) range, capturing fine anatomical textures without smoothing. Source code is available at this https URL.
>
---
#### [new 102] GeoTikzBridge: Advancing Multimodal Code Generation for Geometric Perception and Reasoning
- **分类: cs.CV**

- **简介: 该论文属于多模态代码生成任务，旨在解决MLLM在几何感知与推理上的不足。通过构建两个数据集并提出GeoTikzBridge框架，提升几何结构的理解与推理能力。**

- **链接: [https://arxiv.org/pdf/2603.22687](https://arxiv.org/pdf/2603.22687)**

> **作者:** Jiayin Sun; Caixia Sun; Boyu Yang; Hailin Li; Xiao Chen; Yi Zhang; Errui Ding; Liang Li; Chao Deng; Junlan Feng
>
> **备注:** accepted by CVPR 2026
>
> **摘要:** Multimodal Large Language Models (MLLMs) have recently demonstrated remarkable perceptual and reasoning abilities. However, they struggle to perceive fine-grained geometric structures, constraining their ability of geometric understanding and visual reasoning. To address this, we propose GeoTikzBridge, a framework that enhances local geometric perception and visual reasoning through tikz-based code generation. Within this framework, we build two models supported by two complementary datasets. The GeoTikzBridge-Base model is trained on GeoTikz-Base dataset, the largest image-to-tikz dataset to date with 2.5M pairs (16 $\times$ larger than existing open-sourced datasets). This process is achieved via iterative data expansion and a localized geometric transformation strategy. Subsequently, GeoTikzBridge-Instruct is fine-tuned on GeoTikz-Instruct dataset which is the first instruction-augmented tikz dataset supporting visual reasoning. Extensive experimental results demonstrate that our models achieve state-of-the-art performance among open-sourced MLLMs. Furthermore, GeoTikzBridge models can serve as plug-and-play reasoning modules for any MLLM(LLM), enhancing reasoning performance in geometric problem-solving. Datasets and codes are publicly available at: this https URL.
>
---
#### [new 103] MAGICIAN: Efficient Long-Term Planning with Imagined Gaussians for Active Mapping
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于主动地图构建任务，解决传统方法因贪心策略导致的探索效率低和场景重建不完整问题。提出MAGICIAN框架，通过想象高斯表示实现长期规划，提升覆盖效率。**

- **链接: [https://arxiv.org/pdf/2603.22650](https://arxiv.org/pdf/2603.22650)**

> **作者:** Shiyao Li; Antoine Guédon; Shizhe Chen; Vincent Lepetit
>
> **备注:** Accepted at CVPR 2026. Project webpage: this https URL
>
> **摘要:** Active mapping aims to determine how an agent should move to efficiently reconstruct an unknown environment. Most existing approaches rely on greedy next-best-view prediction, resulting in inefficient exploration and incomplete scene reconstruction. To address this limitation, we introduce MAGICIAN, a novel long-term planning framework that maximizes accumulated surface coverage gain through Imagined Gaussians, a scene representation derived from a pre-trained occupancy network with strong structural priors. This representation enables efficient computation of coverage gain for any novel viewpoint via fast volumetric rendering, allowing its integration into a tree-search algorithm for long-horizon planning. We update Imagined Gaussians and refine the planned trajectory in a closed-loop manner. Our method achieves state-of-the-art performance across indoor and outdoor benchmarks with varying action spaces, demonstrating the critical advantage of long-term planning in active mapping.
>
---
#### [new 104] MVPBench: A Multi-Video Perception Evaluation Benchmark for Multi-Modal Video Understanding
- **分类: cs.CV**

- **简介: 该论文提出MVPBench，一个用于评估多模态视频理解模型的基准，解决多视频感知能力不足的问题，包含14个子任务和5K题库。**

- **链接: [https://arxiv.org/pdf/2603.22756](https://arxiv.org/pdf/2603.22756)**

> **作者:** Purui Bai; Tao Wu; Jiayang Sun; Xinyue Liu; Huaibo Huang; Ran He
>
> **备注:** 15 pages, 7 figures, accepted by IJCNN 2026, code and dataset available at this https URL
>
> **摘要:** The rapid progress of Large Language Models (LLMs) has spurred growing interest in Multi-modal LLMs (MLLMs) and motivated the development of benchmarks to evaluate their perceptual and comprehension abilities. Existing benchmarks, however, are limited to static images or single videos, overlooking the complex interactions across multiple videos. To address this gap, we introduce the Multi-Video Perception Evaluation Benchmark (MVPBench), a new benchmark featuring 14 subtasks across diverse visual domains designed to evaluate models on extracting relevant information from video sequences to make informed decisions. MVPBench includes 5K question-answering tests involving 2.7K video clips sourced from existing datasets and manually annotated clips. Extensive evaluations reveal that current models struggle to process multi-video inputs effectively, underscoring substantial limitations in their multi-video comprehension. We anticipate MVPBench will drive advancements in multi-video perception.
>
---
#### [new 105] Reconstruction-Guided Slot Curriculum: Addressing Object Over-Fragmentation in Video Object-Centric Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视频对象中心学习任务，旨在解决对象过度碎片化问题。通过引入重建引导的槽位课程机制，逐步分配槽位并增强语义边界，提升视频中对象表示的一致性与准确性。**

- **链接: [https://arxiv.org/pdf/2603.22758](https://arxiv.org/pdf/2603.22758)**

> **作者:** WonJun Moon; Hyun Seok Seong; Jae-Pil Heo
>
> **备注:** CVPR 2026 paper. Our code is available at this http URL
>
> **摘要:** Video Object-Centric Learning seeks to decompose raw videos into a small set of object slots, but existing slot-attention models often suffer from severe over-fragmentation. This is because the model is implicitly encouraged to occupy all slots to minimize the reconstruction objective, thereby representing a single object with multiple redundant slots. We tackle this limitation with a reconstruction-guided slot curriculum (SlotCurri). Training starts with only a few coarse slots and progressively allocates new slots where reconstruction error remains high, thus expanding capacity only where it is needed and preventing fragmentation from the outset. Yet, during slot expansion, meaningful sub-parts can emerge only if coarse-level semantics are already well separated; however, with a small initial slot budget and an MSE objective, semantic boundaries remain blurry. Therefore, we augment MSE with a structure-aware loss that preserves local contrast and edge information to encourage each slot to sharpen its semantic boundaries. Lastly, we propose a cyclic inference that rolls slots forward and then backward through the frame sequence, producing temporally consistent object representations even in the earliest frames. All combined, SlotCurri addresses object over-fragmentation by allocating representational capacity where reconstruction fails, further enhanced by structural cues and cyclic inference. Notable FG-ARI gains of +6.8 on YouTube-VIS and +8.3 on MOVi-C validate the effectiveness of SlotCurri. Our code is available at this http URL.
>
---
#### [new 106] Harnessing Lightweight Transformer with Contextual Synergic Enhancement for Efficient 3D Medical Image Segmentation
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于3D医学图像分割任务，旨在解决Transformer模型计算量大、依赖大量标注数据的问题。提出Light-UNETR模型和CSE策略，提升模型与数据效率。**

- **链接: [https://arxiv.org/pdf/2603.23390](https://arxiv.org/pdf/2603.23390)**

> **作者:** Xinyu Liu; Zhen Chen; Wuyang Li; Chenxin Li; Yixuan Yuan
>
> **备注:** Accepted to IEEE TPAMI
>
> **摘要:** Transformers have shown remarkable performance in 3D medical image segmentation, but their high computational requirements and need for large amounts of labeled data limit their applicability. To address these challenges, we consider two crucial aspects: model efficiency and data efficiency. Specifically, we propose Light-UNETR, a lightweight transformer designed to achieve model efficiency. Light-UNETR features a Lightweight Dimension Reductive Attention (LIDR) module, which reduces spatial and channel dimensions while capturing both global and local features via multi-branch attention. Additionally, we introduce a Compact Gated Linear Unit (CGLU) to selectively control channel interaction with minimal parameters. Furthermore, we introduce a Contextual Synergic Enhancement (CSE) learning strategy, which aims to boost the data efficiency of Transformers. It first leverages the extrinsic contextual information to support the learning of unlabeled data with Attention-Guided Replacement, then applies Spatial Masking Consistency that utilizes intrinsic contextual information to enhance the spatial context reasoning for unlabeled data. Extensive experiments on various benchmarks demonstrate the superiority of our approach in both performance and efficiency. For example, with only 10% labeled data on the Left Atrial Segmentation dataset, our method surpasses BCP by 1.43% Jaccard while drastically reducing the FLOPs by 90.8% and parameters by 85.8%. Code is released at this https URL.
>
---
#### [new 107] MLLM-HWSI: A Multimodal Large Language Model for Hierarchical Whole Slide Image Understanding
- **分类: cs.CV**

- **简介: 该论文提出MLLM-HWSI，解决WSI多尺度理解问题。通过多尺度对齐与融合，提升病理诊断的准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2603.23067](https://arxiv.org/pdf/2603.23067)**

> **作者:** Basit Alawode; Arif Mahmood; Muaz Khalifa Al-Radi; Shahad Albastaki; Asim Khan; Muhammad Bilal; Moshira Ali Abdalla; Mohammed Bennamoun; Sajid Javed
>
> **摘要:** Whole Slide Images (WSIs) exhibit hierarchical structure, where diagnostic information emerges from cellular morphology, regional tissue organization, and global context. Existing Computational Pathology (CPath) Multimodal Large Language Models (MLLMs) typically compress an entire WSI into a single embedding, which hinders fine-grained grounding and ignores how pathologists synthesize evidence across different scales. We introduce \textbf{MLLM-HWSI}, a Hierarchical WSI-level MLLM that aligns visual features with pathology language at four distinct scales, cell as word, patch as phrase, region as sentence, and WSI as paragraph to support interpretable evidence-grounded reasoning. MLLM-HWSI decomposes each WSI into multi-scale embeddings with scale-specific projectors and jointly enforces (i) a hierarchical contrastive objective and (ii) a cross-scale consistency loss, preserving semantic coherence from cells to the WSI. We compute diagnostically relevant patches and aggregate segmented cell embeddings into a compact cellular token per-patch using a lightweight \textit{Cell-Cell Attention Fusion (CCAF)} transformer. The projected multi-scale tokens are fused with text tokens and fed to an instruction-tuned LLM for open-ended reasoning, VQA, report, and caption generation tasks. Trained in three stages, MLLM-HWSI achieves new SOTA results on 13 WSI-level benchmarks across six CPath tasks. By aligning language with multi-scale visual evidence, MLLM-HWSI provides accurate, interpretable outputs that mirror diagnostic workflows and advance holistic WSI understanding. Code is available at: \href{this https URL}{GitHub}.
>
---
#### [new 108] ViBe: Ultra-High-Resolution Video Synthesis Born from Pure Images
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决高分辨率视频合成中因3D注意力导致的计算成本高和图像视频模态差异问题。通过纯图像微调和Relay LoRA策略提升视频分辨率与细节质量。**

- **链接: [https://arxiv.org/pdf/2603.23326](https://arxiv.org/pdf/2603.23326)**

> **作者:** Yunfeng Wu; Hongying Cheng; Zihao He; Songhua Liu
>
> **摘要:** Transformer-based video diffusion models rely on 3D attention over spatial and temporal tokens, which incurs quadratic time and memory complexity and makes end-to-end training for ultra-high-resolution videos prohibitively expensive. To overcome this bottleneck, we propose a pure image adaptation framework that upgrades a video Diffusion Transformer pre-trained at its native scale to synthesize higher-resolution videos. Unfortunately, naively fine-tuning with high-resolution images alone often introduces noticeable noise due to the image-video modality gap. To address this, we decouple the learning objective to separately handle modality alignment and spatial extrapolation. At the core of our approach is Relay LoRA, a two-stage adaptation strategy. In the first stage, the video diffusion model is adapted to the image domain using low-resolution images to bridge the modality gap. In the second stage, the model is further adapted with high-resolution images to acquire spatial extrapolation capability. During inference, only the high-resolution adaptation is retained to preserve the video generation modality while enabling high-resolution video synthesis. To enhance fine-grained detail synthesis, we further propose a High-Frequency-Awareness-Training-Objective, which explicitly encourages the model to recover high-frequency components from degraded latent representations via a dedicated reconstruction loss. Extensive experiments demonstrate that our method produces ultra-high-resolution videos with rich visual details without requiring any video training data, even outperforming previous state-of-the-art models trained on high-resolution videos by 0.8 on the VBench benchmark. Code will be available at this https URL.
>
---
#### [new 109] Gaze-Regularized Vision-Language-Action Models for Robotic Manipulation
- **分类: cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决细粒度操作中视觉注意力不足的问题。通过引入眼动正则化框架，提升模型的注意力分配能力，提高任务性能和可解释性。**

- **链接: [https://arxiv.org/pdf/2603.23202](https://arxiv.org/pdf/2603.23202)**

> **作者:** Anupam Pani; Yanchao Yang
>
> **摘要:** Despite advances in Vision-Language-Action (VLA) models, robotic manipulation struggles with fine-grained tasks because current models lack mechanisms for active visual attention allocation. Human gaze naturally encodes intent, planning, and execution patterns -- offering a powerful supervisory signal for guiding robot perception. We introduce a gaze-regularized training framework that aligns VLA models' internal attention with human visual patterns without architectural modifications or inference-time overhead. Our method transforms temporally aggregated gaze heatmaps into patch-level distributions and regularizes the transformer's attention through KL divergence, creating an inductive bias toward task-relevant features while preserving deployment efficiency. When integrated into existing VLA architectures, our approach yields 4-12% improvements across manipulation benchmarks. The gaze-regularized models reach equivalent performance with fewer training steps and maintain robustness under lighting variations and sensor noise. Beyond performance metrics, the learned attention patterns produce interpretable visualizations that mirror human strategies, enhancing trust in robotic systems. Moreover, our framework requires no eye-tracking equipment and applies directly to existing datasets. These results demonstrate that human perceptual priors can significantly accelerate robot learning while improving both task performance and system interpretability.
>
---
#### [new 110] How Far Can VLMs Go for Visual Bug Detection? Studying 19,738 Keyframes from 41 Hours of Gameplay Videos
- **分类: cs.CV; cs.SE**

- **简介: 该论文属于视觉错误检测任务，旨在评估VLM在游戏视频中的实际表现。通过分析19,738帧，发现VLM在无微调情况下已能检测部分视觉错误，但提升有限。**

- **链接: [https://arxiv.org/pdf/2603.22706](https://arxiv.org/pdf/2603.22706)**

> **作者:** Wentao Lu; Alexander Senchenko; Alan Sayle; Abram Hindle; Cor-Paul Bezemer
>
> **摘要:** Video-based quality assurance (QA) for long-form gameplay video is labor-intensive and error-prone, yet valuable for assessing game stability and visual correctness over extended play sessions. Vision language models (VLMs) promise general-purpose visual reasoning capabilities and thus appear attractive for detecting visual bugs directly from video frames. Recent benchmarks suggest that VLMs can achieve promising results in detecting visual glitches on curated datasets. Building on these findings, we conduct a real-world study using industrial QA gameplay videos to evaluate how well VLMs perform in practical scenarios. Our study samples keyframes from long gameplay videos and asks a VLM whether each keyframe contains a bug. Starting from a single-prompt baseline, the model achieves a precision of 0.50 and an accuracy of 0.72. We then examine two common enhancement strategies used to improve VLM performance without fine-tuning: (1) a secondary judge model that re-evaluates VLM outputs, and (2) metadata-augmented prompting through the retrieval of prior bug reports. Across \textbf{100 videos} totaling \textbf{41 hours} and \textbf{19,738 keyframes}, these strategies provide only marginal improvements over the simple baseline, while introducing additional computational cost and output variance. Our findings indicate that off-the-shelf VLMs are already capable of detecting a certain range of visual bugs in QA gameplay videos, but further progress likely requires hybrid approaches that better separate textual and visual anomaly detection.
>
---
#### [new 111] One View Is Enough! Monocular Training for In-the-Wild Novel View Generation
- **分类: cs.CV**

- **简介: 该论文属于单目新视角生成任务，解决多视角数据依赖问题。通过单视图训练，利用深度估计构建3D结构，实现高效、无几何依赖的视角合成。**

- **链接: [https://arxiv.org/pdf/2603.23488](https://arxiv.org/pdf/2603.23488)**

> **作者:** Adrien Ramanana Rahary; Nicolas Dufour; Patrick Perez; David Picard
>
> **备注:** 34 pages, 16 figures
>
> **摘要:** Monocular novel-view synthesis has long required multi-view image pairs for supervision, limiting training data scale and diversity. We argue it is not necessary: one view is enough. We present OVIE, trained entirely on unpaired internet images. We leverage a monocular depth estimator as a geometric scaffold at training time: we lift a source image into 3D, apply a sampled camera transformation, and project to obtain a pseudo-target view. To handle disocclusions, we introduce a masked training formulation that restricts geometric, perceptual, and textural losses to valid regions, enabling training on 30 million uncurated images. At inference, OVIE is geometry-free, requiring no depth estimator or 3D representation. Trained exclusively on in-the-wild images, OVIE outperforms prior methods in a zero-shot setting, while being 600x faster than the second-best baseline. Code and models are publicly available at this https URL.
>
---
#### [new 112] Efficient Universal Perception Encoder
- **分类: cs.CV**

- **简介: 该论文提出EUPE，解决边缘设备上高效多任务视觉编码问题。通过知识蒸馏构建小型但强大的通用视觉编码器。**

- **链接: [https://arxiv.org/pdf/2603.22387](https://arxiv.org/pdf/2603.22387)**

> **作者:** Chenchen Zhu; Saksham Suri; Cijo Jose; Maxime Oquab; Marc Szafraniec; Wei Wen; Yunyang Xiong; Patrick Labatut; Piotr Bojanowski; Raghuraman Krishnamoorthi; Vikas Chandra
>
> **摘要:** Running AI models on smart edge devices can unlock versatile user experiences, but presents challenges due to limited compute and the need to handle multiple tasks simultaneously. This requires a vision encoder with small size but powerful and versatile representations. We present our method, Efficient Universal Perception Encoder (EUPE), which offers both inference efficiency and universally good representations for diverse downstream tasks. We achieve this by distilling from multiple domain-expert foundation vision encoders. Unlike previous agglomerative methods that directly scale down from multiple teachers to an efficient encoder, we demonstrate the importance of first scaling up to a large proxy teacher and then scaling down from this single teacher. Experiments show that EUPE achieves on-par or better performance than individual domain experts of the same size on diverse task domains and also outperforms previous agglomerative encoders. We will release the full family of EUPE models and the code to foster future research.
>
---
#### [new 113] FG-Portrait: 3D Flow Guided Editable Portrait Animation
- **分类: cs.CV**

- **简介: 该论文属于人脸动画任务，解决驱动动作到源人脸的运动迁移问题。提出3D流方法，利用几何信息提升运动对应准确性，支持表情和姿态编辑。**

- **链接: [https://arxiv.org/pdf/2603.23381](https://arxiv.org/pdf/2603.23381)**

> **作者:** Yating Xu; Yunqi Miao; Evangelos Ververas; Jiankang Deng; Jifei Song
>
> **备注:** CVPR 2026
>
> **摘要:** Motion transfer from the driving to the source portrait remains a key challenge in the portrait animation. Current diffusion-based approaches condition only on the driving motion, which fails to capture source-to-driving correspondences and consequently yields suboptimal motion transfer. Although flow estimation provides an alternative, predicting dense correspondences from 2D input is ill-posed and often yields inaccurate animation. We address this problem by introducing 3D flows, a learning-free and geometry-driven motion correspondence directly computed from parametric 3D head models. To integrate this 3D prior into diffusion model, we introduce 3D flow encoding to query potential 3D flows for each target pixel to indicate its displacement back to the source location. To obtain 3D flows aligned with 2D motion changes, we further propose depth-guided sampling to accurately locate the corresponding 3D points for each pixel. Beyond high-fidelity portrait animation, our model further supports user-specified editing of facial expression and head pose. Extensive experiments demonstrate the superiority of our method on consistent driving motion transfer as well as faithful source identity preservation.
>
---
#### [new 114] EVA: Efficient Reinforcement Learning for End-to-End Video Agent
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出EVA，一种用于视频理解的高效强化学习框架，解决长视频处理效率低的问题。通过规划前感知策略，实现智能视频分析。**

- **链接: [https://arxiv.org/pdf/2603.22918](https://arxiv.org/pdf/2603.22918)**

> **作者:** Yaolun Zhang; Ruohui Wang; Jiahao Wang; Yepeng Tang; Xuanyu Zheng; Haonan Duan; Hao Lu; Hanming Deng; Lewei Lu
>
> **备注:** CVPR2026
>
> **摘要:** Video understanding with multimodal large language models (MLLMs) remains challenging due to the long token sequences of videos, which contain extensive temporal dependencies and redundant frames. Existing approaches typically treat MLLMs as passive recognizers, processing entire videos or uniformly sampled frames without adaptive reasoning. Recent agent-based methods introduce external tools, yet still depend on manually designed workflows and perception-first strategies, resulting in inefficiency on long videos. We present EVA, an Efficient Reinforcement Learning framework for End-to-End Video Agent, which enables planning-before-perception through iterative summary-plan-action-reflection reasoning. EVA autonomously decides what to watch, when to watch, and how to watch, achieving query-driven and efficient video understanding. To train such agents, we design a simple yet effective three-stage learning pipeline - comprising supervised fine-tuning (SFT), Kahneman-Tversky Optimization (KTO), and Generalized Reward Policy Optimization (GRPO) - that bridges supervised imitation and reinforcement learning. We further construct high-quality datasets for each stage, supporting stable and reproducible training. We evaluate EVA on six video understanding benchmarks, demonstrating its comprehensive capabilities. Compared with existing baselines, EVA achieves a substantial improvement of 6-12% over general MLLM baselines and a further 1-3% gain over prior adaptive agent methods. Our code and model are available at this https URL.
>
---
#### [new 115] Gaze-Regularized VLMs for Ego-Centric Behavior Understanding
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在提升第一视角行为理解。通过引入眼动信息，增强模型对人类意图和未来动作的预测能力。**

- **链接: [https://arxiv.org/pdf/2603.23190](https://arxiv.org/pdf/2603.23190)**

> **作者:** Anupam Pani; Yanchao Yang
>
> **摘要:** Eye gaze, encompassing fixations and saccades, provides critical insights into human intentions and future actions. This study introduces a gaze-regularized framework that enhances Vision Language Models (VLMs) for egocentric behavior understanding. Unlike existing methods that rely solely on visual data and overlook gaze information, our approach directly incorporates gaze information into the VLM architecture during training. By generating gaze-based queries, the model dynamically focuses on gaze-highlighted regions, while a gaze-regularization mechanism ensures the alignment of model attention with human attention patterns. To better understand how gaze can be effectively integrated into VLMs, we conducted extensive experiments exploring various strategies for incorporating gaze data. These innovations enable the prediction of future events with detailed action descriptions. Experimental results demonstrate a nearly 13 % improvement in semantic scores compared to baseline models not leveraging gaze data, highlighting the effectiveness of our approach. This work establishes a foundation for leveraging the human gaze in VLMs, significantly boosting their predictive capabilities in applications requiring accurate and robust future event prediction.
>
---
#### [new 116] AgentFoX: LLM Agent-Guided Fusion with eXplainability for AI-Generated Image Detection
- **分类: cs.CV**

- **简介: 该论文属于AI生成图像检测任务，旨在解决现有检测方法针对性强、判断不一致的问题。提出AgentFoX框架，通过多阶段分析和可解释报告提升检测准确性与可信度。**

- **链接: [https://arxiv.org/pdf/2603.23115](https://arxiv.org/pdf/2603.23115)**

> **作者:** Yangxin Yu; Yue Zhou; Bin Li; Kaiqing Lin; Haodong Li; Jiangqun Ni; Bo Cao
>
> **摘要:** The increasing realism of AI-Generated Images (AIGI) has created an urgent need for forensic tools capable of reliably distinguishing synthetic content from authentic imagery. Existing detectors are typically tailored to specific forgery artifacts--such as frequency-domain patterns or semantic inconsistencies--leading to specialized performance and, at times, conflicting judgments. To address these limitations, we present \textbf{AgentFoX}, a Large Language Model-driven framework that redefines AIGI detection as a dynamic, multi-phase analytical process. Our approach employs a quick-integration fusion mechanism guided by a curated knowledge base comprising calibrated Expert Profiles and contextual Clustering Profiles. During inference, the agent begins with high-level semantic assessment, then transitions to fine-grained, context-aware synthesis of signal-level expert evidence, resolving contradictions through structured reasoning. Instead of returning a coarse binary output, AgentFoX produces a detailed, human-readable forensic report that substantiates its verdict, enhancing interpretability and trustworthiness for real-world deployment. Beyond providing a novel detection solution, this work introduces a scalable agentic paradigm that facilitates intelligent integration of future and evolving forensic tools.
>
---
#### [new 117] Exposure-Normalized Bed and Chair Fall Rates via Continuous AI Monitoring
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医疗安全研究，旨在解决患者跌倒率评估问题。通过AI监控分析椅床跌倒率，发现椅类跌倒风险高于床类，建议优化椅子设计。**

- **链接: [https://arxiv.org/pdf/2603.22785](https://arxiv.org/pdf/2603.22785)**

> **作者:** Paolo Gabriel; Peter Rehani; Zack Drumm; Tyler Troy; Tiffany Wyatt; Narinder Singh
>
> **备注:** 23 pages, 6 figures
>
> **摘要:** This retrospective cohort study used continuous AI monitoring to estimate fall rates by exposure time rather than occupied bed-days. From August 2024 to December 2025, 3,980 eligible monitoring units contributed 292,914 hourly rows, yielding probability-weighted rates of 17.8 falls per 1,000 chair exposure-hours and 4.3 per 1,000 bed exposure-hours. Within the study window, 43 adjudicated falls matched the monitoring pipeline, and 40 linked to eligible exposure hours for the primary Poisson model, producing an adjusted chair-versus-bed rate ratio of 2.35 (95% confidence interval 0.87 to 6.33; p=0.0907). In a separate broader observation cohort (n=32 deduplicated events), 6 of 7 direct chair falls involved footrest-positioning failures. Because this was an observational study in a single health system, these findings remain hypothesis-generating and support testing safer chair setups rather than using chairs less.
>
---
#### [new 118] Dress-ED: Instruction-Guided Editing for Virtual Try-On and Try-Off
- **分类: cs.CV**

- **简介: 该论文属于虚拟试穿与试脱任务，解决现有数据集缺乏指令驱动编辑的问题。提出Dress-ED数据集和统一的多模态扩散框架，支持文本引导的服装编辑。**

- **链接: [https://arxiv.org/pdf/2603.22607](https://arxiv.org/pdf/2603.22607)**

> **作者:** Fulvio Sanguigni; Davide Lobba; Bin Ren; Marcella Cornia; Nicu Sebe; Rita Cucchiara
>
> **摘要:** Recent advances in Virtual Try-On (VTON) and Virtual Try-Off (VTOFF) have greatly improved photo-realistic fashion synthesis and garment reconstruction. However, existing datasets remain static, lacking instruction-driven editing for controllable and interactive fashion generation. In this work, we introduce the Dress Editing Dataset (Dress-ED), the first large-scale benchmark that unifies VTON, VTOFF, and text-guided garment editing within a single framework. Each sample in Dress-ED includes an in-shop garment image, the corresponding person image wearing the garment, their edited counterparts, and a natural-language instruction of the desired modification. Built through a fully automated multimodal pipeline that integrates MLLM-based garment understanding, diffusion-based editing, and LLM-guided verification, Dress-ED comprises over 146k verified quadruplets spanning three garment categories and seven edit types, including both appearance (e.g., color, pattern, material) and structural (e.g., sleeve length, neckline) modifications. Based on this benchmark, we further propose a unified multimodal diffusion framework that jointly reasons over linguistic instructions and visual garment cues, serving as a strong baseline for instruction-driven VTON and VTOFF. Dataset and code will be made publicly available.
>
---
#### [new 119] 3rd Place of MeViS-Audio Track of the 5th PVUW: VIRST-Audio
- **分类: cs.CV**

- **简介: 该论文属于音频引导的视频目标分割任务，解决音频与视觉信息对齐问题。通过将音频转为文本并结合视觉语言模型进行分割，提升分割准确性与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.23126](https://arxiv.org/pdf/2603.23126)**

> **作者:** Jihwan Hong; Jaeyoung Do
>
> **备注:** 4 pages, 2 figures. Technical report for the CVPR 2026 PVUW Workshop (MeViS-Audio Track)
>
> **摘要:** Audio-based Referring Video Object Segmentation (ARVOS) requires grounding audio queries into pixel-level object masks over time, posing challenges in bridging acoustic signals with spatio-temporal visual representations. In this report, we present VIRST-Audio, a practical framework built upon a pretrained RVOS model integrated with a vision-language architecture. Instead of relying on audio-specific training, we convert input audio into text using an ASR module and perform segmentation using text-based supervision, enabling effective transfer from text-based reasoning to audio-driven scenarios. To improve robustness, we further incorporate an existence-aware gating mechanism that estimates whether the referred target object is present in the video and suppresses predictions when it is absent, reducing hallucinated masks and stabilizing segmentation behavior. We evaluate our approach on the MeViS-Audio track of the 5th PVUW Challenge, where VIRST-Audio achieves 3rd place, demonstrating strong generalization and reliable performance in audio-based referring video segmentation.
>
---
#### [new 120] MedObvious: Exposing the Medical Moravec's Paradox in VLMs via Clinical Triage
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于医疗视觉语言模型领域，旨在解决输入验证问题。提出MedObvious基准，测试模型在多图集中的输入一致性判断能力，揭示现有模型在预诊断验证上的不足。**

- **链接: [https://arxiv.org/pdf/2603.23501](https://arxiv.org/pdf/2603.23501)**

> **作者:** Ufaq Khan; Umair Nawaz; L D M S S Teja; Numaan Saeed; Muhammad Bilal; Yutong Xie; Mohammad Yaqub; Muhammad Haris Khan
>
> **备注:** 11 Pages
>
> **摘要:** Vision Language Models (VLMs) are increasingly used for tasks like medical report generation and visual question answering. However, fluent diagnostic text does not guarantee safe visual understanding. In clinical practice, interpretation begins with pre-diagnostic sanity checks: verifying that the input is valid to read (correct modality and anatomy, plausible viewpoint and orientation, and no obvious integrity violations). Existing benchmarks largely assume this step is solved, and therefore miss a critical failure mode: a model can produce plausible narratives even when the input is inconsistent or invalid. We introduce MedObvious, a 1,880-task benchmark that isolates input validation as a set-level consistency capability over small multi-panel image sets: the model must identify whether any panel violates expected coherence. MedObvious spans five progressive tiers, from basic orientation/modality mismatches to clinically motivated anatomy/viewpoint verification and triage-style cues, and includes five evaluation formats to test robustness across interfaces. Evaluating 17 different VLMs, we find that sanity checking remains unreliable: several models hallucinate anomalies on normal (negative-control) inputs, performance degrades when scaling to larger image sets, and measured accuracy varies substantially between multiple-choice and open-ended settings. These results show that pre-diagnostic verification remains unsolved for medical VLMs and should be treated as a distinct, safety-critical capability before deployment.
>
---
#### [new 121] A Synchronized Audio-Visual Multi-View Capture System
- **分类: cs.CV**

- **简介: 该论文属于多模态数据采集任务，解决音频视频同步不足的问题。提出一种同步的音视频多视角采集系统，实现高质量、可重复的对话行为分析。**

- **链接: [https://arxiv.org/pdf/2603.23089](https://arxiv.org/pdf/2603.23089)**

> **作者:** Xiangwei Shi; Era Dorta Perez; Ruud de Jong; Ojas Shirekar; Chirag Raman
>
> **摘要:** Multi-view capture systems have been an important tool in research for recording human motion under controlling conditions. Most existing systems are specified around video streams and provide little or no support for audio acquisition and rigorous audio-video alignment, despite both being essential for studying conversational interaction where timing at the level of turn-taking, overlap, and prosody matters. In this technical report, we describe an audio-visual multi-view capture system that addresses this gap by treating synchronized audio and synchronized video as first-class signals. The system combines a multi-camera pipeline with multi-channel microphone recording under a unified timing architecture and provides a practical workflow for calibration, acquisition, and quality control that supports repeatable recordings at scale. We quantify synchronization performance in deployment and show that the resulting recordings are temporally consistent enough to support fine-grained analysis and data-driven modeling of conversation behavior.
>
---
#### [new 122] Pretext Matters: An Empirical Study of SSL Methods in Medical Imaging
- **分类: cs.CV**

- **简介: 该论文属于医学图像领域，研究SSL方法的选择对特征学习的影响。通过实验对比JEAs与JEPAs在不同模态中的表现，找出最佳匹配策略。**

- **链接: [https://arxiv.org/pdf/2603.22649](https://arxiv.org/pdf/2603.22649)**

> **作者:** Vedrana Ivezić; Mara Pleasure; Ashwath Radhachandran; Saarang Panchavati; Shreeram Athreya; Vivek Sant; Benjamin Emert; Gregory Fishbein; Corey Arnold; William Speier
>
> **摘要:** Though self-supervised learning (SSL) has demonstrated incredible ability to learn robust representations from unlabeled data, the choice of optimal SSL strategy can lead to vastly different performance outcomes in specialized domains. Joint embedding architectures (JEAs) and joint embedding predictive architectures (JEPAs) have shown robustness to noise and strong semantic feature learning compared to pixel reconstruction-based SSL methods, leading to widespread adoption in medical imaging. However, no prior work has systematically investigated which SSL objective is better aligned with the spatial organization of clinically relevant signal. In this work, we empirically investigate how the choice of SSL method impacts the learned representations in medical imaging. We select two representative imaging modalities characterized by unique noise profiles: ultrasound and histopathology. When informative signal is spatially localized, as in histopathology, JEAs are more effective due to their view-invariance objective. In contrast, when diagnostically relevant information is globally structured, such as the macroscopic anatomy present in liver ultrasounds, JEPAs are optimal. These differences are especially evident in the clinical relevance of the learned features, as independently validated by board-certified radiologists and pathologists. Together, our results provide a framework for matching SSL objectives to the structural and noise properties of medical imaging modalities.
>
---
#### [new 123] DA-Flow: Degradation-Aware Optical Flow Estimation with Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于光学流估计任务，旨在解决真实场景下退化视频中的准确对应问题。通过结合扩散模型与卷积网络，提出DA-Flow方法，提升退化条件下的光学流性能。**

- **链接: [https://arxiv.org/pdf/2603.23499](https://arxiv.org/pdf/2603.23499)**

> **作者:** Jaewon Min; Jaeeun Lee; Yeji Choi; Paul Hyunbin Cho; Jin Hyeon Kim; Tae-Young Lee; Jongsik Ahn; Hwayeong Lee; Seonghyun Park; Seungryong Kim
>
> **备注:** Project page: this https URL
>
> **摘要:** Optical flow models trained on high-quality data often degrade severely when confronted with real-world corruptions such as blur, noise, and compression artifacts. To overcome this limitation, we formulate Degradation-Aware Optical Flow, a new task targeting accurate dense correspondence estimation from real-world corrupted videos. Our key insight is that the intermediate representations of image restoration diffusion models are inherently corruption-aware but lack temporal awareness. To address this limitation, we lift the model to attend across adjacent frames via full spatio-temporal attention, and empirically demonstrate that the resulting features exhibit zero-shot correspondence capabilities. Based on this finding, we present DA-Flow, a hybrid architecture that fuses these diffusion features with convolutional features within an iterative refinement framework. DA-Flow substantially outperforms existing optical flow methods under severe degradation across multiple benchmarks.
>
---
#### [new 124] TrajLoom: Dense Future Trajectory Generation from Video
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在预测视频中密集点轨迹的未来状态。通过提出TrajLoom框架，解决轨迹生成的准确性与稳定性问题，提升预测时长与真实感。**

- **链接: [https://arxiv.org/pdf/2603.22606](https://arxiv.org/pdf/2603.22606)**

> **作者:** Zewei Zhang; Jia Jun Cheng Xian; Kaiwen Liu; Ming Liang; Hang Chu; Jun Chen; Renjie Liao
>
> **备注:** Project page, code, model checkpoints, and datasets: this https URL
>
> **摘要:** Predicting future motion is crucial in video understanding and controllable video generation. Dense point trajectories are a compact, expressive motion representation, but modeling their future evolution from observed video remains challenging. We propose a framework that predicts future trajectories and visibility from past trajectories and video context. Our method has three components: (1) Grid-Anchor Offset Encoding, which reduces location-dependent bias by representing each point as an offset from its pixel-center anchor; (2) TrajLoom-VAE, which learns a compact spatiotemporal latent space for dense trajectories with masked reconstruction and a spatiotemporal consistency regularizer; and (3) TrajLoom-Flow, which generates future trajectories in latent space via flow matching, with boundary cues and on-policy K-step fine-tuning for stable sampling. We also introduce TrajLoomBench, a unified benchmark spanning real and synthetic videos with a standardized setup aligned with video-generation benchmarks. Compared with state-of-the-art methods, our approach extends the prediction horizon from 24 to 81 frames while improving motion realism and stability across datasets. The predicted trajectories directly support downstream video generation and editing. Code, model checkpoints, and datasets are available at this https URL.
>
---
#### [new 125] Focus, Don't Prune: Identifying Instruction-Relevant Regions for Information-Rich Image Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态任务，旨在解决视觉语言模型处理复杂图像时计算效率低的问题。提出PinPoint框架，通过定位相关区域提升推理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.22815](https://arxiv.org/pdf/2603.22815)**

> **作者:** Mincheol Kwon; Minseung Lee; Seonga Choi; Miso Choi; Kyeong-Jin Oh; Hyunyoung Lee; Cheonyoung Park; Yongho Song; Seunghyun Park; Jinkyu Kim
>
> **备注:** CVPR 2026
>
> **摘要:** Large Vision-Language Models (LVLMs) have shown strong performance across various multimodal tasks by leveraging the reasoning capabilities of Large Language Models (LLMs). However, processing visually complex and information-rich images, such as infographics or document layouts, requires these models to generate a large number of visual tokens, leading to significant computational overhead. To address this, we propose PinPoint, a novel two-stage framework that first identifies instruction-relevant image regions and then refines them to extract fine-grained visual features for improved reasoning and efficiency. Central to our approach is the Instruction-Region Alignment, which localizes relevant regions using both visual input and textual instructions. We further introduce new annotations that provide richer ground-truth supervision for instruction-relevant regions across challenging VQA benchmarks: InfographicVQA, MultiPageDocVQA, and SinglePageDocVQA. Experimental results show that PinPoint not only achieves superior accuracy compared to existing methods but also reduces computational overhead by minimizing irrelevant visual tokens.
>
---
#### [new 126] When AVSR Meets Video Conferencing: Dataset, Degradation, and the Hidden Mechanism Behind Performance Collapse
- **分类: cs.CV**

- **简介: 该论文属于音频-视觉语音识别任务，针对视频会议中AVSR性能下降的问题，构建了MLD-VC数据集，并分析了传输失真和人类超表达的影响。**

- **链接: [https://arxiv.org/pdf/2603.22915](https://arxiv.org/pdf/2603.22915)**

> **作者:** Yihuan Huang; Jun Xue; Liu Jiajun; Daixian Li; Tong Zhang; Zhuolin Yi; Yanzhen Ren; Kai Li
>
> **摘要:** Audio-Visual Speech Recognition (AVSR) has achieved remarkable progress in offline conditions, yet its robustness in real-world video conferencing (VC) remains largely unexplored. This paper presents the first systematic evaluation of state-of-the-art AVSR models across mainstream VC platforms, revealing severe performance degradation caused by transmission distortions and spontaneous human hyper-expression. To address this gap, we construct \textbf{MLD-VC}, the first multimodal dataset tailored for VC, comprising 31 speakers, 22.79 hours of audio-visual data, and explicit use of the Lombard effect to enhance human hyper-expression. Through comprehensive analysis, we find that speech enhancement algorithms are the primary source of distribution shift, which alters the first and second formants of audio. Interestingly, we find that the distribution shift induced by the Lombard effect closely resembles that introduced by speech enhancement, which explains why models trained on Lombard data exhibit greater robustness in VC. Fine-tuning AVSR models on MLD-VC mitigates this issue, achieving an average 17.5% reduction in CER across several VC platforms. Our findings and dataset provide a foundation for developing more robust and generalizable AVSR systems in real-world video conferencing. MLD-VC is available at this https URL.
>
---
#### [new 127] PolarAPP: Beyond Polarization Demosaicking for Polarimetric Applications
- **分类: cs.CV**

- **简介: 该论文提出PolarAPP，解决极化图像重建与下游任务性能受限的问题。通过联合优化去马赛克和任务，提升图像质量和应用效果。**

- **链接: [https://arxiv.org/pdf/2603.23071](https://arxiv.org/pdf/2603.23071)**

> **作者:** Yidong Luo; Chenggong Li; Yunfeng Song; Ping Wang; Boxin Shi; Junchao Zhang; Xin Yuan
>
> **摘要:** Polarimetric imaging enables advanced vision applications such as normal estimation and de-reflection by capturing unique surface-material interactions. However, existing applications (alternatively called downstream tasks) rely on datasets constructed by naively regrouping raw measurements from division-of-focal-plane sensors, where pixels of the same polarization angle are extracted and aligned into sparse images without proper demosaicking. This reconstruction strategy results in suboptimal, incomplete targets that limit downstream performance. Moreover, current demosaicking methods are task-agnostic, optimizing only for photometric fidelity rather than utility in downstream tasks. Towards this end, we propose PolarAPP, the first framework to jointly optimize demosaicking and its downstream tasks. PolarAPP introduces a feature alignment mechanism that semantically aligns the representations of demosaicking and downstream networks via meta-learning, guiding the reconstruction to be task-aware. It further employs an equivalent imaging constraint for demosaicking training, enabling direct regression to physically meaningful outputs without relying on rearranged data. Finally, a task-refinement stage fine-tunes the task network using the stable demosaicking front-end to further enhance accuracy. Extensive experimental results demonstrate that PolarAPP outperforms existing methods in both demosaicking quality and downstream performance. Code is available upon acceptance.
>
---
#### [new 128] Language Models Can Explain Visual Features via Steering
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉模型可解释性任务，旨在解决如何自动解释视觉模型中的特征。通过因果干预方法，引导语言模型描述视觉特征，提升解释质量。**

- **链接: [https://arxiv.org/pdf/2603.22593](https://arxiv.org/pdf/2603.22593)**

> **作者:** Javier Ferrando; Enrique Lopez-Cuena; Pablo Agustin Martin-Torres; Daniel Hinjos; Anna Arias-Duart; Dario Garcia-Gasulla
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Sparse Autoencoders uncover thousands of features in vision models, yet explaining these features without requiring human intervention remains an open challenge. While previous work has proposed generating correlation-based explanations based on top activating input examples, we present a fundamentally different alternative based on causal interventions. We leverage the structure of Vision-Language Models and steer individual SAE features in the vision encoder after providing an empty image. Then, we prompt the language model to explain what it ``sees'', effectively eliciting the visual concept represented by each feature. Results show that Steering offers an scalable alternative that complements traditional approaches based on input examples, serving as a new axis for automated interpretability in vision models. Moreover, the quality of explanations improves consistently with the scale of the language model, highlighting our method as a promising direction for future research. Finally, we propose Steering-informed Top-k, a hybrid approach that combines the strengths of causal interventions and input-based approaches to achieve state-of-the-art explanation quality without additional computational cost.
>
---
#### [new 129] Dual-Teacher Distillation with Subnetwork Rectification for Black-Box Domain Adaptation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于黑盒域适应任务，解决在无法访问源数据或模型时的迁移学习问题。提出DDSR模型，结合黑盒模型与视觉语言模型的知识，生成可靠伪标签并提升适应性能。**

- **链接: [https://arxiv.org/pdf/2603.22908](https://arxiv.org/pdf/2603.22908)**

> **作者:** Zhe Zhang; Jing Li; Wanli Xue; Xu Cheng; Jianhua Zhang; Qinghua Hu; Shengyong Chen
>
> **备注:** This manuscript is under review at IEEE Transactions on Multimedia
>
> **摘要:** Assuming that neither source data nor the source model is accessible, black box domain adaptation represents a highly practical yet extremely challenging setting, as transferable information is restricted to the predictions of the black box source model, which can only be queried using target samples. Existing approaches attempt to extract transferable knowledge through pseudo label refinement or by leveraging external vision language models (ViLs), but they often suffer from noisy supervision or insufficient utilization of the semantic priors provided by ViLs, which ultimately hinder adaptation performance. To overcome these limitations, we propose a dual teacher distillation with subnetwork rectification (DDSR) model that jointly exploits the specific knowledge embedded in black box source models and the general semantic information of a ViL. DDSR adaptively integrates their complementary predictions to generate reliable pseudo labels for the target domain and introduces a subnetwork driven regularization strategy to mitigate overfitting caused by noisy supervision. Furthermore, the refined target predictions iteratively enhance both the pseudo labels and ViL prompts, enabling more accurate and semantically consistent adaptation. Finally, the target model is further optimized through self training with classwise prototypes. Extensive experiments on multiple benchmark datasets validate the effectiveness of our approach, demonstrating consistent improvements over state of the art methods, including those using source data or models.
>
---
#### [new 130] Sketch2CT: Multimodal Diffusion for Structure-Aware 3D Medical Volume Generation
- **分类: cs.CV**

- **简介: 该论文属于医学图像生成任务，旨在解决多模态下生成结构一致的3D医学体积问题。提出Sketch2CT框架，结合2D草图和文本描述生成准确的3D医学影像。**

- **链接: [https://arxiv.org/pdf/2603.22509](https://arxiv.org/pdf/2603.22509)**

> **作者:** Delin An; Chaoli Wang
>
> **摘要:** Diffusion probabilistic models have demonstrated significant potential in generating high-quality, realistic medical images, providing a promising solution to the persistent challenge of data scarcity in the medical field. Nevertheless, producing 3D medical volumes with anatomically consistent structures under multimodal conditions remains a complex and unresolved problem. We introduce Sketch2CT, a multimodal diffusion framework for structure-aware 3D medical volume generation, jointly guided by a user-provided 2D sketch and a textual description that captures 3D geometric semantics. The framework initially generates 3D segmentation masks of the target organ from random noise, conditioned on both modalities. To effectively align and fuse these inputs, we propose two key modules that refine sketch features with localized textual cues and integrate global sketch-text representations. Built upon a capsule-attention backbone, these modules leverage the complementary strengths of sketches and text to produce anatomically accurate organ shapes. The synthesized segmentation masks subsequently guide a latent diffusion model for 3D CT volume synthesis, enabling realistic reconstruction of organ appearances that are consistent with user-defined sketches and descriptions. Extensive experiments on public CT datasets demonstrate that Sketch2CT achieves superior performance in generating multimodal medical volumes. Its controllable, low-cost generation pipeline enables principled, efficient augmentation of medical datasets. Code is available at this https URL.
>
---
#### [new 131] Color When It Counts: Grayscale-Guided Online Triggering for Always-On Streaming Video Sensing
- **分类: cs.CV; cs.AI; cs.HC; cs.MM**

- **简介: 该论文属于视频理解任务，解决资源受限设备上高成本视频采集问题。提出ColorTrigger方法，通过灰度引导按需激活彩色帧，降低计算与存储开销。**

- **链接: [https://arxiv.org/pdf/2603.22466](https://arxiv.org/pdf/2603.22466)**

> **作者:** Weitong Cai; Hang Zhang; Yukai Huang; Shitong Sun; Jiankang Deng; Songcen Xu; Jifei Song; Zhensong Zhang
>
> **备注:** Accepted at CVPR 2026 (Main track)
>
> **摘要:** Always-on sensing is essential for next-generation edge/wearable AI systems, yet continuous high-fidelity RGB video capture remains prohibitively expensive for resource-constrained mobile and edge platforms. We present a new paradigm for efficient streaming video understanding: grayscale-always, color-on-demand. Through preliminary studies, we discover that color is not always necessary. Sparse RGB frames suffice for comparable performance when temporal structure is preserved via continuous grayscale streams. Building on this insight, we propose ColorTrigger, an online training-free trigger that selectively activates color capture based on windowed grayscale affinity analysis. Designed for real-time edge deployment, ColorTrigger uses lightweight quadratic programming to detect chromatic redundancy causally, coupled with credit-budgeted control and dynamic token routing to jointly reduce sensing and inference costs. On streaming video understanding benchmarks, ColorTrigger achieves 91.6% of full-color baseline performance while using only 8.1% RGB frames, demonstrating substantial color redundancy in natural videos and enabling practical always-on video sensing on resource-constrained devices.
>
---
#### [new 132] Few-Shot Generative Model Adaption via Identity Injection and Preservation
- **分类: cs.CV**

- **简介: 该论文属于少样本生成模型适应任务，旨在解决适应过程中源域身份信息丢失的问题。提出I²P方法，通过身份注入与一致性对齐保留源域特征。**

- **链接: [https://arxiv.org/pdf/2603.22965](https://arxiv.org/pdf/2603.22965)**

> **作者:** Yeqi He; Liang Li; Jiehua Zhang; Yaoqi Sun; Xichun Sheng; Zhidong Zhao; Chenggang Yan
>
> **摘要:** Training generative models with limited data presents severe challenges of mode collapse. A common approach is to adapt a large pretrained generative model upon a target domain with very few samples (fewer than 10), known as few-shot generative model adaptation. However, existing methods often suffer from forgetting source domain identity knowledge during adaptation, which degrades the quality of generated images in the target domain. To address this, we propose Identity Injection and Preservation (I$^2$P), which leverages identity injection and consistency alignment to preserve the source identity knowledge. Specifically, we first introduce an identity injection module that integrates source domain identity knowledge into the target domain's latent space, ensuring the generated images retain key identity knowledge of the source domain. Second, we design an identity substitution module, which includes a style-content decoupler and a reconstruction modulator, to further enhance source domain identity preservation. We enforce identity consistency constraints by aligning features from identity substitution, thereby preserving identity knowledge. Both quantitative and qualitative experiments show that our method achieves substantial improvements over state-of-the-art methods on multiple public datasets and 5 metrics.
>
---
#### [new 133] VQ-Jarvis: Retrieval-Augmented Video Restoration Agent with Sharp Vision and Fast Thought
- **分类: cs.CV**

- **简介: 该论文属于视频修复任务，解决真实场景下多样退化导致的修复难题。提出VQ-Jarvis，通过检索增强和高效策略实现精准退化感知与优化修复路径。**

- **链接: [https://arxiv.org/pdf/2603.22998](https://arxiv.org/pdf/2603.22998)**

> **作者:** Xuanyu Zhang; Weiqi Li; Qunliang Xing; Jingfen Xie; Bin Chen; Junlin Li; Li Zhang; Jian Zhang; Shijie Zhao
>
> **备注:** Video restoration, Agent-based restoration
>
> **摘要:** Video restoration in real-world scenarios is challenged by heterogeneous degradations, where static architectures and fixed inference pipelines often fail to generalize. Recent agent-based approaches offer dynamic decision making, yet existing video restoration agents remain limited by insufficient quality perception and inefficient search strategies. We propose VQ-Jarvis, a retrieval-augmented, all-in-one intelligent video restoration agent with sharper vision and faster thought. VQ-Jarvis is designed to accurately perceive degradations and subtle differences among paired restoration results, while efficiently discovering optimal restoration trajectories. To enable sharp vision, we construct VSR-Compare, the first large-scale video paired enhancement dataset with 20K comparison pairs covering 7 degradation types, 11 enhancement operators, and diverse content domains. Based on this dataset, we train a multiple operator judge model and a degradation perception model to guide agent decisions. To achieve fast thought, we introduce a hierarchical operator scheduling strategy that adapts to video difficulty: for easy cases, optimal restoration trajectories are retrieved in a one-step manner from a retrieval-augmented generation (RAG) library; for harder cases, a step-by-step greedy search is performed to balance efficiency and accuracy. Extensive experiments demonstrate that VQ-Jarvis consistently outperforms existing methods on complex degraded videos.
>
---
#### [new 134] Founder effects shape the evolutionary dynamics of multimodality in open LLM families
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究开放大语言模型家族中多模态能力的演化，分析其传播路径与速度，揭示多模态通过少数创始事件引入并快速扩展的机制。**

- **链接: [https://arxiv.org/pdf/2603.22287](https://arxiv.org/pdf/2603.22287)**

> **作者:** Manuel Cebrian
>
> **备注:** 7 pages, 4 figures, 2 tables
>
> **摘要:** Large language model (LLM) families are improving rapidly, yet it remains unclear how quickly multimodal capabilities emerge and propagate within open families. Using the ModelBiome AI Ecosystem dataset of Hugging Face model metadata and recorded lineage fields (>1.8x10^6 model entries), we quantify multimodality over time and along recorded parent-to-child relations. Cross-modal tasks are widespread in the broader ecosystem well before they become common within major open LLM families: within these families, multimodality remains rare through 2023 and most of 2024, then increases sharply in 2024-2025 and is dominated by image-text vision-language tasks. Across major families, the first vision-language model (VLM) variants typically appear months after the first text-generation releases, with lags ranging from ~1 month (Gemma) to more than a year for several families and ~26 months for GLM. Lineage-conditioned transition rates show weak cross-type transfer: among fine-tuning edges from text-generation parents, only 0.218% yield VLM descendants. Instead, multimodality expands primarily within existing VLM lineages: 94.5% of VLM-child fine-tuning edges originate from VLM parents, versus 4.7% from text-generation parents. At the model level, most VLM releases appear as new roots without recorded parents (~60%), while the remainder are predominantly VLM-derived; founder concentration analyses indicate rapid within-lineage amplification followed by diversification. Together, these results show that multimodality enters open LLM families through rare founder events and then expands rapidly within their descendant lineages, producing punctuated adoption dynamics that likely induce distinct, transfer-limited scaling behavior for multimodal capabilities.
>
---
#### [new 135] GSwap: Realistic Head Swapping with Dynamic Neural Gaussian Field
- **分类: cs.CV**

- **简介: 该论文提出GSwap，解决视频头像替换任务中的3D一致性、表情自然度和背景融合问题。通过动态神经高斯场实现高质量、逼真的头像交换。**

- **链接: [https://arxiv.org/pdf/2603.23168](https://arxiv.org/pdf/2603.23168)**

> **作者:** Jingtao Zhou; Xuan Gao; Dongyu Liu; Junhui Hou; Yudong Guo; Juyong Zhang
>
> **备注:** Accepted to TVCG, Project page: this https URL
>
> **摘要:** We present GSwap, a novel consistent and realistic video head-swapping system empowered by dynamic neural Gaussian portrait priors, which significantly advances the state of the art in face and head replacement. Unlike previous methods that rely primarily on 2D generative models or 3D Morphable Face Models (3DMM), our approach overcomes their inherent limitations, including poor 3D consistency, unnatural facial expressions, and restricted synthesis quality. Moreover, existing techniques struggle with full head-swapping tasks due to insufficient holistic head modeling and ineffective background blending, often resulting in visible artifacts and misalignments. To address these challenges, GSwap introduces an intrinsic 3D Gaussian feature field embedded within a full-body SMPL-X surface, effectively elevating 2D portrait videos into a dynamic neural Gaussian field. This innovation ensures high-fidelity, 3D-consistent portrait rendering while preserving natural head-torso relationships and seamless motion dynamics. To facilitate training, we adapt a pretrained 2D portrait generative model to the source head domain using only a few reference images, enabling efficient domain adaptation. Furthermore, we propose a neural re-rendering strategy that harmoniously integrates the synthesized foreground with the original background, eliminating blending artifacts and enhancing realism. Extensive experiments demonstrate that GSwap surpasses existing methods in multiple aspects, including visual quality, temporal coherence, identity preservation, and 3D consistency.
>
---
#### [new 136] InverFill: One-Step Inversion for Enhanced Few-Step Diffusion Inpainting
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像修复任务，解决少步扩散模型在修复时出现的语义不一致和伪影问题。提出InverFill方法，在初始噪声中注入语义信息，提升修复质量。**

- **链接: [https://arxiv.org/pdf/2603.23463](https://arxiv.org/pdf/2603.23463)**

> **作者:** Duc Vu; Kien Nguyen; Trong-Tung Nguyen; Ngan Nguyen; Phong Nguyen; Khoi Nguyen; Cuong Pham; Anh Tran
>
> **备注:** Accepted to CVPR'26 (Main Conference)
>
> **摘要:** Recent diffusion-based models achieve photorealism in image inpainting but require many sampling steps, limiting practical use. Few-step text-to-image models offer faster generation, but naively applying them to inpainting yields poor harmonization and artifacts between the background and inpainted region. We trace this cause to random Gaussian noise initialization, which under low function evaluations causes semantic misalignment and reduced fidelity. To overcome this, we propose InverFill, a one-step inversion method tailored for inpainting that injects semantic information from the input masked image into the initial noise, enabling high-fidelity few-step inpainting. Instead of training inpainting models, InverFill leverages few-step text-to-image models in a blended sampling pipeline with semantically aligned noise as input, significantly improving vanilla blended sampling and even matching specialized inpainting models at low NFEs. Moreover, InverFill does not require real-image supervision and only adds minimal inference overhead. Extensive experiments show that InverFill consistently boosts baseline few-step models, improving image quality and text coherence without costly retraining or heavy iterative optimization.
>
---
#### [new 137] FDIF: Formula-Driven supervised Learning with Implicit Functions for 3D Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文提出FDIF框架，用于3D医学图像分割，解决数据获取困难问题，通过公式生成数据，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.23199](https://arxiv.org/pdf/2603.23199)**

> **作者:** Yukinori Yamamoto; Kazuya Nishimura; Tsukasa Fukusato; Hirokazu Nosato; Tetsuya Ogata; Hirokatsu Kataoka
>
> **备注:** Submitted to ECCV2026
>
> **摘要:** Deep learning-based 3D medical image segmentation methods relies on large-scale labeled datasets, yet acquiring such data is difficult due to privacy constraints and the high cost of expert annotation. Formula-Driven Supervised Learning (FDSL) offers an appealing alternative by generating training data and labels directly from mathematical formulas. However, existing voxel-based approaches are limited in geometric expressiveness and cannot synthesize realistic textures. We introduce Formula-Driven supervised learning with Implicit Functions (FDIF), a framework that enables scalable pre-training without using any real data and medical expert annotations. FDIF introduces an implicit-function representation based on signed distance functions (SDFs), enabling compact modeling of complex geometries while exploiting the surface representation of SDFs to support controllable synthesis of both geometric and intensity textures. Across three medical image segmentation benchmarks (AMOS, ACDC, and KiTS) and three architectures (SwinUNETR, nnUNet ResEnc-L, and nnUNet Primus-M), FDIF consistently improves over a formula-driven method, and achieves performance comparable to self-supervised approaches pre-trained on large-scale real datasets. We further show that FDIF pre-training also benefits 3D classification tasks, highlighting implicit-function-based formula supervision as a promising paradigm for data-free representation learning. Code is available at this https URL.
>
---
#### [new 138] OccAny: Generalized Unconstrained Urban 3D Occupancy
- **分类: cs.CV**

- **简介: 该论文属于3D城市场景占用预测任务，旨在解决模型在非领域场景下的泛化能力不足问题。提出OccAny模型，支持多种输入并提升预测精度与几何补全能力。**

- **链接: [https://arxiv.org/pdf/2603.23502](https://arxiv.org/pdf/2603.23502)**

> **作者:** Anh-Quan Cao; Tuan-Hung Vu
>
> **备注:** Accepted to CVPR 2026. Project page: this https URL
>
> **摘要:** Relying on in-domain annotations and precise sensor-rig priors, existing 3D occupancy prediction methods are limited in both scalability and out-of-domain generalization. While recent visual geometry foundation models exhibit strong generalization capabilities, they were mainly designed for general purposes and lack one or more key ingredients required for urban occupancy prediction, namely metric prediction, geometry completion in cluttered scenes and adaptation to urban scenarios. We address this gap and present OccAny, the first unconstrained urban 3D occupancy model capable of operating on out-of-domain uncalibrated scenes to predict and complete metric occupancy coupled with segmentation features. OccAny is versatile and can predict occupancy from sequential, monocular, or surround-view images. Our contributions are three-fold: (i) we propose the first generalized 3D occupancy framework with (ii) Segmentation Forcing that improves occupancy quality while enabling mask-level prediction, and (iii) a Novel View Rendering pipeline that infers novel-view geometry to enable test-time view augmentation for geometry completion. Extensive experiments demonstrate that OccAny outperforms all visual geometry baselines on 3D occupancy prediction task, while remaining competitive with in-domain self-supervised methods across three input settings on two established urban occupancy prediction datasets. Our code is available at this https URL .
>
---
#### [new 139] Toward Faithful Segmentation Attribution via Benchmarking and Dual-Evidence Fusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于语义分割的可解释性任务，旨在解决 attribution map 的忠实性和稳定性问题。通过构建基准测试并提出 DEA 方法，提升分割模型解释的可靠性。**

- **链接: [https://arxiv.org/pdf/2603.22624](https://arxiv.org/pdf/2603.22624)**

> **作者:** Abu Noman Md Sakib; OFM Riaz Rahman Aranya; Kevin Desai; Zijie Zhang
>
> **摘要:** Attribution maps for semantic segmentation are almost always judged by visual plausibility. Yet looking convincing does not guarantee that the highlighted pixels actually drive the model's prediction, nor that attribution credit stays within the target region. These questions require a dedicated evaluation protocol. We introduce a reproducible benchmark that tests intervention-based faithfulness, off-target leakage, perturbation robustness, and runtime on Pascal VOC and SBD across three pretrained backbones. To further demonstrate the benchmark, we propose Dual-Evidence Attribution (DEA), a lightweight correction that fuses gradient evidence with region-level intervention signals through agreement-weighted fusion. DEA increases emphasis where both sources agree and retains causal support when gradient responses are unstable. Across all completed runs, DEA consistently improves deletion-based faithfulness over gradient-only baselines and preserves strong robustness, at the cost of additional compute from intervention passes. The benchmark exposes a faithfulness-stability tradeoff among attribution families that is entirely hidden under visual evaluation, providing a foundation for principled method selection in segmentation explainability. Code is available at this https URL.
>
---
#### [new 140] MinerU-Diffusion: Rethinking Document OCR as Inverse Rendering via Diffusion Decoding
- **分类: cs.CV**

- **简介: 该论文属于文档OCR任务，旨在解决长文档识别中的序列延迟和错误传播问题。提出MinerU-Diffusion框架，采用扩散解码替代自回归方法，提升效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.22458](https://arxiv.org/pdf/2603.22458)**

> **作者:** Hejun Dong; Junbo Niu; Bin Wang; Weijun Zeng; Wentao Zhang; Conghui He
>
> **摘要:** Optical character recognition (OCR) has evolved from line-level transcription to structured document parsing, requiring models to recover long-form sequences containing layout, tables, and formulas. Despite recent advances in vision-language models, most existing systems rely on autoregressive decoding, which introduces sequential latency and amplifies error propagation in long documents. In this work, we revisit document OCR from an inverse rendering perspective, arguing that left-to-right causal generation is an artifact of serialization rather than an intrinsic property of the task. Motivated by this insight, we propose MinerU-Diffusion, a unified diffusion-based framework that replaces autoregressive sequential decoding with parallel diffusion denoising under visual conditioning. MinerU-Diffusion employs a block-wise diffusion decoder and an uncertainty-driven curriculum learning strategy to enable stable training and efficient long-sequence inference. Extensive experiments demonstrate that MinerU-Diffusion consistently improves robustness while achieving up to 3.2x faster decoding compared to autoregressive baselines. Evaluations on the proposed Semantic Shuffle benchmark further confirm its reduced dependence on linguistic priors and stronger visual OCR capability.
>
---
#### [new 141] UniQueR: Unified Query-based Feedforward 3D Reconstruction
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出UniQueR，用于从无姿态图像中进行高效精确的3D重建。解决传统方法在可见表面限制的问题，通过稀疏3D查询实现单次前向传播重建场景结构。**

- **链接: [https://arxiv.org/pdf/2603.22851](https://arxiv.org/pdf/2603.22851)**

> **作者:** Chensheng Peng; Quentin Herau; Jiezhi Yang; Yichen Xie; Yihan Hu; Wenzhao Zheng; Matthew Strong; Masayoshi Tomizuka; Wei Zhan
>
> **摘要:** We present UniQueR, a unified query-based feedforward framework for efficient and accurate 3D reconstruction from unposed images. Existing feedforward models such as DUSt3R, VGGT, and AnySplat typically predict per-pixel point maps or pixel-aligned Gaussians, which remain fundamentally 2.5D and limited to visible surfaces. In contrast, UniQueR formulates reconstruction as a sparse 3D query inference problem. Our model learns a compact set of 3D anchor points that act as explicit geometric queries, enabling the network to infer scene structure, including geometry in occluded regions--in a single forward pass. Each query encodes spatial and appearance priors directly in global 3D space (instead of per-frame camera space) and spawns a set of 3D Gaussians for differentiable rendering. By leveraging unified query interactions across multi-view features and a decoupled cross-attention design, UniQueR achieves strong geometric expressiveness while substantially reducing memory and computational cost. Experiments on Mip-NeRF 360 and VR-NeRF demonstrate that UniQueR surpasses state-of-the-art feedforward methods in both rendering quality and geometric accuracy, using an order of magnitude fewer primitives than dense alternatives.
>
---
#### [new 142] ABot-PhysWorld: Interactive World Foundation Model for Robotic Manipulation with Physics Alignment
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决视频生成中物理不一致的问题。提出ABot-PhysWorld模型，结合物理对齐和动作控制，提升生成视频的物理合理性。**

- **链接: [https://arxiv.org/pdf/2603.23376](https://arxiv.org/pdf/2603.23376)**

> **作者:** Yuzhi Chen; Ronghan Chen; Dongjie Huo; Yandan Yang; Dekang Qi; Haoyun Liu; Tong Lin; Shuang Zeng; Junjin Xiao; Xinyuan Chang; Feng Xiong; Xing Wei; Zhiheng Ma; Mu Xu
>
> **摘要:** Video-based world models offer a powerful paradigm for embodied simulation and planning, yet state-of-the-art models often generate physically implausible manipulations - such as object penetration and anti-gravity motion - due to training on generic visual data and likelihood-based objectives that ignore physical laws. We present ABot-PhysWorld, a 14B Diffusion Transformer model that generates visually realistic, physically plausible, and action-controllable videos. Built on a curated dataset of three million manipulation clips with physics-aware annotation, it uses a novel DPO-based post-training framework with decoupled discriminators to suppress unphysical behaviors while preserving visual quality. A parallel context block enables precise spatial action injection for cross-embodiment control. To better evaluate generalization, we introduce EZSbench, the first training-independent embodied zero-shot benchmark combining real and synthetic unseen robot-task-scene combinations. It employs a decoupled protocol to separately assess physical realism and action alignment. ABot-PhysWorld achieves new state-of-the-art performance on PBench and EZSbench, surpassing Veo 3.1 and Sora v2 Pro in physical plausibility and trajectory consistency. We will release EZSbench to promote standardized evaluation in embodied video generation.
>
---
#### [new 143] ST-GDance++: A Scalable Spatial-Temporal Diffusion for Long-Duration Group Choreography
- **分类: cs.LG; cs.AI; cs.CV; cs.SD**

- **简介: 该论文属于群体舞蹈生成任务，解决多舞者协调与长序列生成效率问题。提出ST-GDance++框架，通过解耦时空依赖提升生成效率和稳定性。**

- **链接: [https://arxiv.org/pdf/2603.22316](https://arxiv.org/pdf/2603.22316)**

> **作者:** Jing Xu; Weiqiang Wang; Cunjian Chen; Jun Liu; Qiuhong Ke
>
> **摘要:** Group dance generation from music requires synchronizing multiple dancers while maintaining spatial coordination, making it highly relevant to applications such as film production, gaming, and animation. Recent group dance generation models have achieved promising generation quality, but they remain difficult to deploy in interactive scenarios due to bidirectional attention dependencies. As the number of dancers and the sequence length increase, the attention computation required for aligning music conditions with motion sequences grows quadratically, leading to reduced efficiency and increased risk of motion collisions. Effectively modeling dense spatial-temporal interactions is therefore essential, yet existing methods often struggle to capture such complexity, resulting in limited scalability and unstable multi-dancer coordination. To address these challenges, we propose ST-GDance++, a scalable framework that decouples spatial and temporal dependencies to enable efficient and collision-aware group choreography generation. For spatial modeling, we introduce lightweight distance-aware graph convolutions to capture inter-dancer relationships while reducing computational overhead. For temporal modeling, we design a diffusion noise scheduling strategy together with an efficient temporal-aligned attention mask, enabling stream-based generation for long motion sequences and improving scalability in long-duration scenarios. Experiments on the AIOZ-GDance dataset show that ST-GDance++ achieves competitive generation quality with significantly reduced latency compared to existing methods.
>
---
#### [new 144] Learning Sidewalk Autopilot from Multi-Scale Imitation with Corrective Behavior Expansion
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，解决复杂城市环境中步行道微出行的控制问题。通过多尺度模仿学习和修正行为扩展，提升策略的鲁棒性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.22527](https://arxiv.org/pdf/2603.22527)**

> **作者:** Honglin He; Yukai Ma; Brad Squicciarini; Wayne Wu; Bolei Zhou
>
> **摘要:** Sidewalk micromobility is a promising solution for last-mile transportation, but current learning-based control methods struggle in complex urban environments. Imitation learning (IL) learns policies from human demonstrations, yet its reliance on fixed offline data often leads to compounding errors, limited robustness, and poor generalization. To address these challenges, we propose a framework that advances IL through corrective behavior expansion and multi-scale imitation learning. On the data side, we augment teleoperation datasets with diverse corrective behaviors and sensor augmentations to enable the policy to learn to recover from its own mistakes. On the model side, we introduce a multi-scale IL architecture that captures both short-horizon interactive behaviors and long-horizon goal-directed intentions via horizon-based trajectory clustering and hierarchical supervision. Real-world experiments show that our approach significantly improves robustness and generalization in diverse sidewalk scenarios.
>
---
#### [new 145] Strain-Parameterized Coupled Dynamics and Dual-Camera Visual Servoing for Aerial Continuum Manipulators
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究空中连续机械臂的动态建模与视觉伺服控制，解决高计算成本和平台欠驱动问题，提出融合应变参数化模型与无人机刚体模型的统一动力学框架，并设计鲁棒双目视觉伺服方案。**

- **链接: [https://arxiv.org/pdf/2603.23333](https://arxiv.org/pdf/2603.23333)**

> **作者:** Niloufar Amiri; Farrokh Janabi-Sharifi
>
> **摘要:** Tendon-driven aerial continuum manipulators (TD-ACMs) combine the maneuverability of uncrewed aerial vehicles (UAVs) with the compliance of lightweight continuum robots (CRs). Existing coupled dynamic modeling approaches for TD-ACMs incur high computational costs and do not explicitly account for aerial platform underactuation. To address these limitations, this paper presents a generalized dynamic formulation of a coupled TD-ACM with an underactuated base. The proposed approach integrates a strain-parameterized Cosserat rod model with a rigid-body model of the UAV into a unified Lagrangian ordinary differential equation (ODE) framework on $\mathrm{SE}(3)$, thereby eliminating computationally intensive symbolic derivations. Building upon the developed model, a robust dual-camera image-based visual servoing (IBVS) scheme is introduced. The proposed controller mitigates the field-of-view (FoV) limitations of conventional IBVS, compensates for attitude-induced image motion caused by UAV lateral dynamics, and incorporates a low-level adaptive controller to address modeling uncertainties with formal stability guarantees. Extensive simulations and experimental validation on a compact custom-built prototype demonstrate the effectiveness and robustness of the proposed framework in real-world scenarios.
>
---
#### [new 146] Single-Subject Multi-View MRI Super-Resolution via Implicit Neural Representations
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出SIMS-MRI，解决单病患多视角MRI超分辨率问题，通过隐式表示和视图对齐生成一致的各向同性图像。**

- **链接: [https://arxiv.org/pdf/2603.22627](https://arxiv.org/pdf/2603.22627)**

> **作者:** Heejong Kim; Abhishek Thanki; Roel van Herten; Daniel Margolis; Mert R Sabuncu
>
> **摘要:** Clinical MRI frequently acquires anisotropic volumes with high in-plane resolution and low through-plane resolution to reduce acquisition time. Multiple orientations are therefore acquired to provide complementary anatomical information. Conventional integration of these views relies on registration followed by interpolation, which can degrade fine structural details. Recent deep learning-based super-resolution (SR) approaches have demonstrated strong performance in enhancing single-view images. However, their clinical reliability is often limited by the need for large-scale training datasets, resulting in increased dependence on cohort-level priors. Self-supervised strategies offer an alternative by learning directly from the target scans. Prior work either neglects the existence of multi-view information or assumes that in-plane information can supervise through-plane reconstruction under the assumption of pre-alignment between images. However, this assumption is rarely satisfied in clinical settings. In this work, we introduce Single-Subject Implicit Multi-View Super-Resolution for MRI (SIMS-MRI), a framework that operates solely on anisotropic multi-view scans from a single patient without requiring pre- or post-processing. Our method combines a multi-resolution hash-encoded implicit representation with learned inter-view alignment to generate a spatially consistent isotropic reconstruction. We validate the SIMS-MRI pipeline on both simulated brain and clinical prostate MRI datasets. Code will be made publicly available for reproducibility: this https URL
>
---
#### [new 147] Policy-based Tuning of Autoregressive Image Models with Instance- and Distribution-Level Rewards
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决AR模型样本质量与多样性不足的问题。通过引入分布级奖励机制和多目标优化，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2603.23086](https://arxiv.org/pdf/2603.23086)**

> **作者:** Orhun Buğra Baran; Melih Kandemir; Ramazan Gokberk Cinbis
>
> **摘要:** Autoregressive (AR) models are highly effective for image generation, yet their standard maximum-likelihood estimation training lacks direct optimization for sample quality and diversity. While reinforcement learning (RL) has been used to align diffusion models, these methods typically suffer from output diversity collapse. Similarly, concurrent RL methods for AR models rely strictly on instance-level rewards, often trading off distributional coverage for quality. To address these limitations, we propose a lightweight RL framework that casts token-based AR synthesis as a Markov Decision Process, optimized via Group Relative Policy Optimization (GRPO). Our core contribution is the introduction of a novel distribution-level Leave-One-Out FID (LOO-FID) reward; by leveraging an exponential moving average of feature moments, it explicitly encourages sample diversity and prevents mode collapse during policy updates. We integrate this with composite instance-level rewards (CLIP and HPSv2) for strict semantic and perceptual fidelity, and stabilize the multi-objective learning with an adaptive entropy regularization term. Extensive experiments on LlamaGen and VQGAN architectures demonstrate clear improvements across standard quality and diversity metrics within only a few hundred tuning iterations. The results also show that the model can be updated to produce competitive samples even without Classifier-Free Guidance, and bypass its 2x inference cost.
>
---
#### [new 148] MCLR: Improving Conditional Modeling in Visual Generative Models via Inter-Class Likelihood-Ratio Maximization and Establishing the Equivalence between Classifier-Free Guidance and Alignment Objectives
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于生成模型任务，旨在解决扩散模型依赖推理阶段引导的问题。通过提出MCLR方法，提升类别间分离度，实现无需引导的高质量生成。**

- **链接: [https://arxiv.org/pdf/2603.22364](https://arxiv.org/pdf/2603.22364)**

> **作者:** Xiang Li; Yixuan Jia; Xiao Li; Jeffrey A. Fessler; Rongrong Wang; Qing Qu
>
> **摘要:** Diffusion models have achieved state-of-the-art performance in generative modeling, but their success often relies heavily on classifier-free guidance (CFG), an inference-time heuristic that modifies the sampling trajectory. From a theoretical perspective, diffusion models trained with standard denoising score matching (DSM) are expected to recover the target data distribution, raising the question of why inference-time guidance is necessary in practice. In this work, we ask whether the DSM training objective can be modified in a principled manner such that standard reverse-time sampling, without inference-time guidance, yields effects comparable to CFG. We identify insufficient inter-class separation as a key limitation of standard diffusion models. To address this, we propose MCLR, a principled alignment objective that explicitly maximizes inter-class likelihood-ratios during training. Models fine-tuned with MCLR exhibit CFG-like improvements under standard sampling, achieving comparable qualitative and quantitative gains without requiring inference-time guidance. Beyond empirical benefits, we provide a theoretical result showing that the CFG-guided score is exactly the optimal solution to a weighted MCLR objective. This establishes a formal equivalence between classifier-free guidance and alignment-based objectives, offering a mechanistic interpretation of CFG.
>
---
#### [new 149] PhysSkin: Real-Time and Generalizable Physics-Based Animation via Self-Supervised Neural Skinning
- **分类: cs.GR; cs.CV; cs.LG**

- **简介: 该论文提出PhysSkin，解决实时、通用的物理驱动动画问题。通过神经皮肤场和自监督学习，实现跨不同3D形状的高效动画生成。**

- **链接: [https://arxiv.org/pdf/2603.23194](https://arxiv.org/pdf/2603.23194)**

> **作者:** Yuanhang Lei; Tao Cheng; Xingxuan Li; Boming Zhao; Siyuan Huang; Ruizhen Hu; Peter Yichen Chen; Hujun Bao; Zhaopeng Cui
>
> **备注:** Accepted by CVPR 2026. Project Page: this https URL
>
> **摘要:** Achieving real-time physics-based animation that generalizes across diverse 3D shapes and discretizations remains a fundamental challenge. We introduce PhysSkin, a physics-informed framework that addresses this challenge. In the spirit of Linear Blend Skinning, we learn continuous skinning fields as basis functions lifting motion subspace coordinates to full-space deformation, with subspace defined by handle transformations. To generate mesh-free, discretization-agnostic, and physically consistent skinning fields that generalize well across diverse 3D shapes, PhysSkin employs a new neural skinning fields autoencoder which consists of a transformer-based encoder and a cross-attention decoder. Furthermore, we also develop a novel physics-informed self-supervised learning strategy that incorporates on-the-fly skinning-field normalization and conflict-aware gradient correction, enabling effective balancing of energy minimization, spatial smoothness, and orthogonality constraints. PhysSkin shows outstanding performance on generalizable neural skinning and enables real-time physics-based animation.
>
---
#### [new 150] Ca2+ transient detection and segmentation with the Astronomically motivated algorithm for Background Estimation And Transient Segmentation (Astro-BEATS)
- **分类: q-bio.NC; astro-ph.IM; cs.CV**

- **简介: 该论文属于钙离子瞬态检测任务，解决荧光成像中微小钙信号检测难题。提出Astro-BEATS算法，结合天文学方法提升检测精度与速度。**

- **链接: [https://arxiv.org/pdf/2603.22311](https://arxiv.org/pdf/2603.22311)**

> **作者:** Bolin Fan; Anthony Bilodeau; Frederic Beaupre; Theresa Wiesner; Christian Gagne; Flavie Lavoie-Cardinal; Renee Hlozek
>
> **备注:** 29 pages, 4 figures, 12 supplementary pages, 5 supplementary figures
>
> **摘要:** Fluorescence-based Ca$^{2+}$-imaging is a powerful tool for studying localized neuronal activity, including miniature Synaptic Calcium Transients, providing real-time insights into synaptic activity. These transients induce only subtle changes in the fluorescence signal, often barely above baseline, which poses a significant challenge for automated synaptic transient detection and segmentation. Detecting astronomical transients similarly requires efficient algorithms that will remain robust over a large field of view with varying noise properties. We leverage techniques used in astronomical transient detection for miniature Synaptic Calcium Transient detection in fluorescence microscopy. We present Astro-BEATS, an automatic miniature Synaptic Calcium Transient segmentation algorithm that incorporates image estimation and source-finding techniques used in astronomy and designed for Ca$^{2+}$-imaging videos. Astro-BEATS outperforms current threshold-based approaches for synaptic Ca$^{2+}$ transient detection and segmentation. The produced segmentation masks can be used to train a supervised deep learning algorithm for improved synaptic Ca$^{2+}$ transient detection in Ca$^{2+}$-imaging data. The speed of Astro-BEATS and its applicability to previously unseen datasets without re-optimization makes it particularly useful for generating training datasets for deep learning-based approaches.
>
---
#### [new 151] TreeTeaming: Autonomous Red-Teaming of Vision-Language Models via Hierarchical Strategy Exploration
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于AI安全领域，旨在解决VLMs的安全漏洞问题。提出TreeTeaming框架，通过动态策略探索实现更有效的红队测试，提升攻击成功率并增强策略多样性。**

- **链接: [https://arxiv.org/pdf/2603.22882](https://arxiv.org/pdf/2603.22882)**

> **作者:** Chunxiao Li; Lijun Li; Jing Shao
>
> **备注:** CVPR2026
>
> **摘要:** The rapid advancement of Vision-Language Models (VLMs) has brought their safety vulnerabilities into sharp focus. However, existing red teaming methods are fundamentally constrained by an inherent linear exploration paradigm, confining them to optimizing within a predefined strategy set and preventing the discovery of novel, diverse exploits. To transcend this limitation, we introduce TreeTeaming, an automated red teaming framework that reframes strategy exploration from static testing to a dynamic, evolutionary discovery process. At its core lies a strategic Orchestrator, powered by a Large Language Model (LLM), which autonomously decides whether to evolve promising attack paths or explore diverse strategic branches, thereby dynamically constructing and expanding a strategy tree. A multimodal actuator is then tasked with executing these complex strategies. In the experiments across 12 prominent VLMs, TreeTeaming achieves state-of-the-art attack success rates on 11 models, outperforming existing methods and reaching up to 87.60\% on GPT-4o. The framework also demonstrates superior strategic diversity over the union of previously public jailbreak strategies. Furthermore, the generated attacks exhibit an average toxicity reduction of 23.09\%, showcasing their stealth and subtlety. Our work introduces a new paradigm for automated vulnerability discovery, underscoring the necessity of proactive exploration beyond static heuristics to secure frontier AI models.
>
---
#### [new 152] Contrastive Metric Learning for Point Cloud Segmentation in Highly Granular Detectors
- **分类: hep-ex; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于点云分割任务，旨在解决高粒度探测器中重叠粒子流的分离问题。提出基于对比度量学习的方法，学习具有区分性的嵌入空间，提升分割效果。**

- **链接: [https://arxiv.org/pdf/2603.23356](https://arxiv.org/pdf/2603.23356)**

> **作者:** Max Marriott-Clarke; Lazar Novakovic; Elizabeth Ratzer; Robert J. Bainbridge; Loukas Gouskos; Benedikt Maier
>
> **摘要:** We propose a novel clustering approach for point-cloud segmentation based on supervised contrastive metric learning (CML). Rather than predicting cluster assignments or object-centric variables, the method learns a latent representation in which points belonging to the same object are embedded nearby while unrelated points are separated. Clusters are then reconstructed using a density-based readout in the learned metric space, decoupling representation learning from cluster formation and enabling flexible inference. The approach is evaluated on simulated data from a highly granular calorimeter, where the task is to separate highly overlapping particle showers represented as sets of calorimeter hits. A direct comparison with object condensation (OC) is performed using identical graph neural network backbones and equal latent dimensionality, isolating the effect of the learning objective. The CML method produces a more stable and separable embedding geometry for both electromagnetic and hadronic particle showers, leading to improved local neighbourhood consistency, a more reliable separation of overlapping showers, and better generalization when extrapolating to unseen multiplicities and energies. This translates directly into higher reconstruction efficiency and purity, particularly in high-multiplicity regimes, as well as improved energy resolution. In mixed-particle environments, CML maintains strong performance, suggesting robust learning of the shower topology, while OC exhibits significant degradation. These results demonstrate that similarity-based representation learning combined with density-based aggregation is a promising alternative to object-centric approaches for point cloud segmentation in highly granular detectors.
>
---
#### [new 153] VTAM: Video-Tactile-Action Models for Complex Physical Interaction Beyond VLAs
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出VTAM模型，解决视觉-触觉交互中视觉感知不足的问题，通过融合触觉信息提升物理交互任务性能。**

- **链接: [https://arxiv.org/pdf/2603.23481](https://arxiv.org/pdf/2603.23481)**

> **作者:** Haoran Yuan; Weigang Yi; Zhenyu Zhang; Wendi Chen; Yuchen Mo; Jiashi Yin; Xinzhuo Li; Xiangyu Zeng; Chuan Wen; Cewu Lu; Katherine Driggs-Campbell; Ismini Lourentzou
>
> **备注:** this https URL
>
> **摘要:** Video-Action Models (VAMs) have emerged as a promising framework for embodied intelligence, learning implicit world dynamics from raw video streams to produce temporally consistent action predictions. Although such models demonstrate strong performance on long-horizon tasks through visual reasoning, they remain limited in contact-rich scenarios where critical interaction states are only partially observable from vision alone. In particular, fine-grained force modulation and contact transitions are not reliably encoded in visual tokens, leading to unstable or imprecise behaviors. To bridge this gap, we introduce the Video-Tactile Action Model (VTAM), a multimodal world modeling framework that incorporates tactile perception as a complementary grounding signal. VTAM augments a pretrained video transformer with tactile streams via a lightweight modality transfer finetuning, enabling efficient cross-modal representation learning without tactile-language paired data or independent tactile pretraining. To stabilize multimodal fusion, we introduce a tactile regularization loss that enforces balanced cross-modal attention, preventing visual latent dominance in the action model. VTAM demonstrates superior performance in contact-rich manipulation, maintaining a robust success rate of 90 percent on average. In challenging scenarios such as potato chip pick-and-place requiring high-fidelity force awareness, VTAM outperforms the pi 0.5 baseline by 80 percent. Our findings demonstrate that integrating tactile feedback is essential for correcting visual estimation errors in world action models, providing a scalable approach to physically grounded embodied foundation models.
>
---
#### [new 154] Three Creates All: You Only Sample 3 Steps
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决扩散模型推理速度慢的问题。通过优化时间嵌入，实现少步采样，提升效率。**

- **链接: [https://arxiv.org/pdf/2603.22375](https://arxiv.org/pdf/2603.22375)**

> **作者:** Yuren Cai; Guangyi Wang; Zongqing Li; Li Li; Zhihui Liu; Songzhi Su
>
> **摘要:** Diffusion models deliver high-fidelity generation but remain slow at inference time due to many sequential network evaluations. We find that standard timestep conditioning becomes a key bottleneck for few-step sampling. Motivated by layer-dependent denoising dynamics, we propose Multi-layer Time Embedding Optimization (MTEO), which freeze the pretrained diffusion backbone and distill a small set of step-wise, layer-wise time embeddings from reference trajectories. MTEO is plug-and-play with existing ODE solvers, adds no inference-time overhead, and trains only a tiny fraction of parameters. Extensive experiments across diverse datasets and backbones show state-of-the-art performance in the few-step sampling and substantially narrow the gap between distillation-based and lightweight methods. Code will be available.
>
---
#### [new 155] L-UNet: An LSTM Network for Remote Sensing Image Change Detection
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于遥感图像变化检测任务，旨在解决传统方法缺乏空间特征的问题。提出L-UNet和AL-UNet模型，融合时空特征以提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.22842](https://arxiv.org/pdf/2603.22842)**

> **作者:** Shuting Sun; Lin Mu; Lizhe Wang; Peng Liu
>
> **摘要:** Change detection of high-resolution remote sensing images is an important task in earth observation and was extensively investigated. Recently, deep learning has shown to be very successful in plenty of remote sensing tasks. The current deep learning-based change detection method is mainly based on conventional long short-term memory (Conv-LSTM), which does not have spatial characteristics. Since change detection is a process with both spatiality and temporality, it is necessary to propose an end-to-end spatiotemporal network. To achieve this, Conv-LSTM, an extension of the Conv-LSTM structure, is introduced. Since it shares similar spatial characteristics with the convolutional layer, L-UNet, which substitutes partial convolution layers of UNet-to-Conv-LSTM and Atrous L-UNet (AL-UNet), which further using Atrous structure to multiscale spatial information is proposed. Experiments on two data sets are conducted and the proposed methods show the advantages both in quantity and quality when compared with some other methods.
>
---
#### [new 156] Viewport-based Neural 360° Image Compression
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于360°图像压缩任务，旨在解决传统投影方法导致的过采样和失真问题。通过viewport提取和神经编码，提升压缩效率并保留全局信息。**

- **链接: [https://arxiv.org/pdf/2603.22776](https://arxiv.org/pdf/2603.22776)**

> **作者:** Jingwei Liao; Bo Chen; Klara Nahrstedt; Zhisheng Yan
>
> **摘要:** Given the popularity of 360° images on social media platforms, 360° image compression becomes a critical technology for media storage and transmission. Conventional 360° image compression pipeline projects the spherical image into a single 2D plane, leading to issues of oversampling and distortion. In this paper, we propose a novel viewport-based neural compression pipeline for 360° images. By replacing the image projection in conventional 360° image compression pipelines with viewport extraction and efficiently compressing multiple viewports, the proposed pipeline minimizes the inherent oversampling and distortion issues. However, viewport extraction impedes information sharing between multiple viewports during compression, causing the loss of global information about the spherical image. To tackle this global information loss, we design a neural viewport codec to capture global prior information across multiple viewports and maximally compress the viewport data. The viewport codec is empowered by a transformer-based ViewPort ConText (VPCT) module that can be integrated with canonical learning-based 2D image compression structures. We compare the proposed pipeline with existing 360° image compression models and conventional 360° image compression pipelines building on learning-based 2D image codecs and standard hand-crafted codecs. Results show that our pipeline saves an average of $14.01\%$ bit consumption compared to the best-performing 360° image compression methods without compromising quality. The proposed VPCT-based codec also outperforms existing 2D image codecs in the viewport-based neural compression pipeline. Our code can be found at: this https URL.
>
---
#### [new 157] Abnormalities and Disease Detection in Gastro-Intestinal Tract Images
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于胃肠道图像分析任务，旨在解决异常检测与分割问题。通过传统方法与深度学习结合，提升实时检测的准确性与效率。**

- **链接: [https://arxiv.org/pdf/2603.22378](https://arxiv.org/pdf/2603.22378)**

> **作者:** Zeshan Khan; Muhammad Atif Tahir
>
> **备注:** PhD Thesis
>
> **摘要:** Gastrointestinal (GI) tract image analysis plays a crucial role in medical diagnosis. This research addresses the challenge of accurately classifying and segmenting GI images for real-time applications, where traditional methods often struggle due to the diversity and complexity of abnormalities. The high computational demands of this domain require efficient and adaptable solutions. This PhD thesis presents a multifaceted approach to GI image analysis. Initially, texture-based feature extraction and classification methods were explored, achieving high processing speed (over 4000 FPS) and strong performance (F1-score: 0.76, Accuracy: 0.98) on the Kvasir V2 dataset. The study then transitions to deep learning, where an optimized model combined with data bagging techniques improved performance, reaching an accuracy of 0.92 and an F1-score of 0.60 on the HyperKvasir dataset, and an F1-score of 0.88 on Kvasir V2. To support real-time detection, a streamlined neural network integrating texture and local binary patterns was developed. By addressing inter-class similarity and intra-class variation through a learned threshold, the system achieved 41 FPS with high accuracy (0.99) and an F1-score of 0.91 on HyperKvasir. Additionally, two segmentation tools are proposed to enhance usability, leveraging Depth-Wise Separable Convolution and neural network ensembles for improved detection, particularly in low-FPS scenarios. Overall, this research introduces novel and adaptable methodologies, progressing from traditional texture-based techniques to deep learning and ensemble approaches, providing a comprehensive framework for advancing GI image analysis.
>
---
## 更新

#### [replaced 001] Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.16740](https://arxiv.org/pdf/2512.16740)**

> **作者:** Yunkai Yang; Yudong Zhang; Kunquan Zhang; Jinxiao Zhang; Xinying Chen; Haohuan Fu; Runmin Dong
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** With the rapid progress of controllable generation, training data synthesis has become a promising way to expand labeled datasets and alleviate manual annotation in remote sensing (RS). However, the complexity of semantic mask control and the uncertainty of sampling quality often limit the utility of synthetic data in downstream semantic segmentation tasks. To address these challenges, we propose a task-oriented data synthesis framework (TODSynth), including a Multimodal Diffusion Transformer (MM-DiT) with unified triple attention and a plug-and-play sampling strategy guided by task feedback. Built upon the powerful DiT-based generative foundation model, we systematically evaluate different control schemes, showing that a text-image-mask joint attention scheme combined with full fine-tuning of the image and mask branches significantly enhances the effectiveness of RS semantic segmentation data synthesis, particularly in few-shot and complex-scene scenarios. Furthermore, we propose a control-rectify flow matching (CRFM) method, which dynamically adjusts sampling directions guided by semantic loss during the early high-plasticity stage, mitigating the instability of generated images and bridging the gap between synthetic data and downstream segmentation tasks. Extensive experiments demonstrate that our approach consistently outperforms state-of-the-art controllable generation methods, producing more stable and task-oriented synthetic data for RS semantic segmentation.
>
---
#### [replaced 002] From Editor to Dense Geometry Estimator
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.04338](https://arxiv.org/pdf/2509.04338)**

> **作者:** JiYuan Wang; Chunyu Lin; Lei Sun; Rongying Liu; Lang Nie; Mingxing Li; Kang Liao; Xiangxiang Chu
>
> **备注:** Accepted to CVPR 2026, 18pages, with appendix
>
> **摘要:** Leveraging visual priors from pre-trained text-to-image (T2I) generative models has shown success in dense prediction. However, dense prediction is inherently an image-to-image task, suggesting that image editing models, rather than T2I generative models, may be a more suitable foundation for fine-tuning. Motivated by this, we conduct a systematic analysis of the fine-tuning behaviors of both editors and generators for dense geometry estimation. Our findings show that editing models possess inherent structural priors, which enable them to converge more stably by ``refining" their innate features, and ultimately achieve higher performance than their generative counterparts. Based on these findings, we introduce \textbf{FE2E}, a framework that pioneeringly adapts an advanced editing model based on Diffusion Transformer (DiT) architecture for dense geometry prediction. Specifically, to tailor the editor for this deterministic task, we reformulate the editor's original flow matching loss into the ``consistent velocity" training objective. And we use logarithmic quantization to resolve the precision conflict between the editor's native BFloat16 format and the high precision demand of our tasks. Additionally, we leverage the DiT's global attention for a cost-free joint estimation of depth and normals in a single forward pass, enabling their supervisory signals to mutually enhance each other. Without scaling up the training data, FE2E achieves impressive performance improvements in zero-shot monocular depth and normal estimation across multiple datasets. Notably, it achieves over 35\% performance gains on the ETH3D dataset and outperforms the DepthAnything series, which is trained on 100$\times$ data. The project page can be accessed \href{this https URL}{here}.
>
---
#### [replaced 003] Uni3R: Unified 3D Reconstruction and Semantic Understanding via Generalizable Gaussian Splatting from Unposed Multi-View Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.03643](https://arxiv.org/pdf/2508.03643)**

> **作者:** Xiangyu Sun; Haoyi Jiang; Liu Liu; Seungtae Nam; Gyeongjin Kang; Xinjie Wang; Wei Sui; Zhizhong Su; Wenyu Liu; Xinggang Wang; Eunbyung Park
>
> **备注:** The code is available at this https URL
>
> **摘要:** Reconstructing and semantically interpreting 3D scenes from sparse 2D views remains a fundamental challenge in computer vision. Conventional methods often decouple semantic understanding from reconstruction or necessitate costly per-scene optimization, thereby restricting their scalability and generalizability. In this paper, we introduce Uni3R, a novel feed-forward framework that jointly reconstructs a unified 3D scene representation enriched with open-vocabulary semantics, directly from unposed multi-view images. Our approach leverages a Cross-View Transformer to robustly integrate information across arbitrary multi-view inputs, which then regresses a set of 3D Gaussian primitives endowed with semantic feature fields. This unified representation facilitates high-fidelity novel view synthesis, open-vocabulary 3D semantic segmentation, and depth prediction, all within a single, feed-forward pass. Extensive experiments demonstrate that Uni3R establishes a new state-of-the-art across multiple benchmarks, including 25.07 PSNR on RE10K and 55.84 mIoU on ScanNet. Our work signifies a novel paradigm towards generalizable, unified 3D scene reconstruction and understanding. The code is available at this https URL.
>
---
#### [replaced 004] 1S-DAug: One-Shot Data Augmentation for Robust Few-Shot Generalization
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.00114](https://arxiv.org/pdf/2602.00114)**

> **作者:** Yunwei Bai; Ying Kiat Tan; Yao Shu; Tsuhan Chen
>
> **摘要:** Few-shot learning (FSL) challenges model generalization to novel classes based on just a few shots of labeled examples, a testbed where traditional test-time augmentations fail to be effective. We introduce 1S-DAug, a one-shot generative augmentation operator that synthesizes diverse yet faithful variants from just one example image at test time. 1S-DAug couples traditional geometric perturbations with controlled noise injection and a denoising diffusion process conditioned on the original image. The generated images are then encoded and aggregated, alongside the original image, into a combined representation for more robust FSL predictions. Integrated as a training-free model-agnostic plugin, 1S-DAug consistently improves FSL across standard benchmarks of 4 different datasets without any model parameter update, including achieving up to 20% relative accuracy improvement on the miniImagenet 5-way-1-shot benchmark. Code will be released.
>
---
#### [replaced 005] From Inpainting to Editing: Unlocking Robust Mask-Free Visual Dubbing via Generative Bootstrapping
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.25066](https://arxiv.org/pdf/2512.25066)**

> **作者:** Xu He; Haoxian Zhang; Hejia Chen; Changyuan Zheng; Liyang Chen; Songlin Tang; Jiehui Huang; Xiaoqiang Liu; Pengfei Wan; Zhiyong Wu
>
> **备注:** Project Page: this https URL
>
> **摘要:** Audio-driven visual dubbing aims to synchronize a video's lip movements with new speech but is fundamentally challenged by the lack of ideal training data: paired videos differing only in lip motion. Existing methods circumvent this via mask-based inpainting. However, masking inevitably destroys spatiotemporal context, leading to identity drift and poor robustness (e.g., to occlusions), while also inducing lip-shape leakage that degrades lip sync. To bridge this gap, we propose X-Dub, a novel two-stage generative bootstrapping framework leveraging powerful Diffusion Transformers to unlock mask-free dubbing. Our core insight is to repurpose a mask-based inpainting model exclusively as a dedicated data generator to synthesize scalable, high-fidelity pseudo-paired data, which is subsequently utilized to train and bootstrap a robust, mask-free editing model as the final video dubber. The final dubber is liberated from masking artifacts and leverages the complete video input for high-fidelity inference. We further introduce timestep-adaptive multi-phase learning to disentangle conflicting objectives (structure, lip motion, and texture) across diffusion phases, facilitating stable convergence and advanced editing quality. Additionally, we present X-DubBench, a benchmark for diverse scenarios. Extensive experiments demonstrate that our method achieves state-of-the-art performance with superior lip sync, visual quality, and robustness.
>
---
#### [replaced 006] Catalogue Grounded Multimodal Attribution for Museum Video under Resource and Regulatory Constraints
- **分类: cs.MM; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.11147](https://arxiv.org/pdf/2603.11147)**

> **作者:** Minsak Nanang; Adrian Hilton; Armin Mustafa
>
> **备注:** Demo video url: this https URL
>
> **摘要:** Audiovisual (AV) archives in museums and galleries are growing rapidly, but much of this material remains effectively locked away because it lacks consistent, searchable metadata. Existing method for archiving requires extensive manual effort. We address this by automating the most labour intensive part of the workflow: catalogue style metadata curation for in gallery video, grounded in an existing collection database. Concretely, we propose catalogue-grounded multimodal attribution for museum AV content using an open, locally deployable video language model. We design a multi pass pipeline that (i) summarises artworks in a video, (ii) generates catalogue style descriptions and genre labels, and (iii) attempts to attribute title and artist via conservative similarity matching to the structured catalogue. Early deployments on a painting catalogue suggest that this framework can improve AV archive discoverability while respecting resource constraints, data sovereignty, and emerging regulation, offering a transferable template for application-driven machine learning in other high-stakes domains.
>
---
#### [replaced 007] Selective Noise Suppression and Discriminative Mutual Interaction for Robust Audio-Visual Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.14203](https://arxiv.org/pdf/2603.14203)**

> **作者:** Kai Peng; Yunzhe Shen; Miao Zhang; Leiye Liu; Yidong Han; Wei Ji; Jingjing Li; Yongri Piao; Huchuan Lu
>
> **备注:** Accepted to IEEE Transactions on Multimedia (TMM) 2026. Code: this https URL
>
> **摘要:** The ability to capture and segment sounding objects in dynamic visual scenes is crucial for the development of Audio-Visual Segmentation (AVS) tasks. While significant progress has been made in this area, the interaction between audio and visual modalities still requires further exploration. In this work, we aim to answer the following questions: How can a model effectively suppress audio noise while enhancing relevant audio information? How can we achieve discriminative interaction between the audio and visual modalities? To this end, we propose SDAVS, equipped with the Selective Noise-Resilient Processor (SNRP) module and the Discriminative Audio-Visual Mutual Fusion (DAMF) strategy. The proposed SNRP mitigates audio noise interference by selectively emphasizing relevant auditory cues, while DAMF ensures more consistent audio-visual representations. Experimental results demonstrate that our proposed method achieves state-of-the-art performance on benchmark AVS datasets, especially in multi-source and complex scenes. \textit{The code and model are available at this https URL}.
>
---
#### [replaced 008] Local Precise Refinement: A Dual-Gated Mixture-of-Experts for Enhancing Foundation Model Generalization against Spectral Shifts
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.13352](https://arxiv.org/pdf/2603.13352)**

> **作者:** Xi Chen; Maojun Zhang; Yu Liu; Shen Yan
>
> **摘要:** Domain Generalization Semantic Segmentation (DGSS) in spectral remote sensing is severely challenged by spectral shifts across diverse acquisition conditions, which cause significant performance degradation for models deployed in unseen domains. While fine-tuning foundation models is a promising direction, existing methods employ global, homogeneous adjustments. This "one-size-fits-all" tuning struggles with the spatial heterogeneity of land cover, causing semantic confusion. We argue that the key to robust DGSS lies not in a single global adaptation, but in performing fine-grained, spatially-adaptive refinement of a foundation model's features. To achieve this, we propose SpectralMoE, a novel fine-tuning framework for DGSS. It operationalizes this principle by utilizing a Mixture-of-Experts (MoE) architecture to perform \textbf{local precise refinement} on the foundation model's features, incorporating depth features estimated from selected RGB bands of the spectral remote sensing imagery to guide the fine-tuning process. Specifically, SpectralMoE employs a dual-gated MoE architecture that independently routes visual and depth features to top-k selected experts for specialized refinement, enabling modality-specific adjustments. A subsequent cross-attention mechanism then judiciously fuses the refined structural cues into the visual stream, mitigating semantic ambiguities caused by spectral variations. Extensive experiments show that SpectralMoE sets a new state-of-the-art on multiple DGSS benchmarks across hyperspectral, multispectral, and RGB remote sensing imagery.
>
---
#### [replaced 009] Efficient and High-Fidelity Omni Modality Retrieval
- **分类: cs.IR; cs.CL; cs.CV**

- **简介: 该论文属于多模态检索任务，旨在解决现有模型仅支持两种模态的问题。提出OmniRet模型，支持文本、视觉和音频三种模态，提升计算效率和表示精度。**

- **链接: [https://arxiv.org/pdf/2603.02098](https://arxiv.org/pdf/2603.02098)**

> **作者:** Chuong Huynh; Manh Luong; Abhinav Shrivastava
>
> **备注:** CVPR 2026. Project page: this https URL
>
> **摘要:** Multimodal retrieval is the task of aggregating information from queries across heterogeneous modalities to retrieve desired targets. State-of-the-art multimodal retrieval models can understand complex queries, yet they are typically limited to two modalities: text and vision. This limitation impedes the development of universal retrieval systems capable of comprehending queries that combine more than two modalities. To advance toward this goal, we present OmniRet, the first retrieval model capable of handling complex, composed queries spanning three key modalities: text, vision, and audio. Our OmniRet model addresses two critical challenges for universal retrieval: computational efficiency and representation fidelity. First, feeding massive token sequences from modality-specific encoders to Large Language Models (LLMs) is computationally inefficient. We therefore introduce an attention-based resampling mechanism to generate compact, fixed-size representations from these sequences. Second, compressing rich omni-modal data into a single embedding vector inevitably causes information loss and discards fine-grained details. We propose Attention Sliced Wasserstein Pooling to preserve these fine-grained details, leading to improved omni-modal representations. OmniRet is trained on an aggregation of approximately 6 million query-target pairs spanning 30 datasets. We benchmark our model on 13 retrieval tasks and a MMEBv2 subset. Our model demonstrates significant improvements on composed query, audio and video retrieval tasks, while achieving on-par performance with state-of-the-art models on others. Furthermore, we curate a new Audio-Centric Multimodal Benchmark (ACM). This new benchmark introduces two critical, previously missing tasks-composed audio retrieval and audio-visual retrieval to more comprehensively evaluate a model's omni-modal embedding capacity.
>
---
#### [replaced 010] Temporal Slowness in Central Vision Drives Semantic Object Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.04462](https://arxiv.org/pdf/2602.04462)**

> **作者:** Timothy Schaumlöffel; Arthur Aubret; Gemma Roig; Jochen Triesch
>
> **备注:** ICLR 2026
>
> **摘要:** Humans acquire semantic object representations from egocentric visual streams with minimal supervision, but the underlying mechanisms remain unclear. Importantly, the visual system only processes the center of its field of view with high resolution and it learns similar representations for visual inputs occurring close in time. This emphasizes slowly changing information around gaze locations. This study investigates the role of central vision and slowness learning in the formation of semantic object representations from human-like visual experience. We simulate five months of human-like visual experience using the Ego4D dataset and a state-of-the-art gaze prediction model. We extract image crops around predicted gaze locations to train a time-contrastive Self-Supervised Learning model. Our results show that exploiting temporal slowness when learning from central visual field experience improves the encoding of different facets of object semantics. Specifically, focusing on central vision strengthens the extraction of foreground object features, while considering temporal slowness, especially in conjunction with eye movements, allows the model to encode broader semantic information about objects. These findings provide new insights into the mechanisms by which humans may develop semantic object representations from natural visual experience. Our code will be made public upon acceptance. Code is available at this https URL.
>
---
#### [replaced 011] Background Fades, Foreground Leads: Curriculum-Guided Background Pruning for Efficient Foreground-Centric Collaborative Perception
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶中的协同感知任务，旨在解决车载网络带宽受限下背景信息丢失的问题。通过引入课程学习策略，将背景上下文融入前景特征，提升感知效果。**

- **链接: [https://arxiv.org/pdf/2510.19250](https://arxiv.org/pdf/2510.19250)**

> **作者:** Yuheng Wu; Xiangbo Gao; Quang Tau; Zhengzhong Tu; Dongman Lee
>
> **备注:** ICRA 2026
>
> **摘要:** Collaborative perception enhances the reliability and spatial coverage of autonomous vehicles by sharing complementary information across vehicles, offering a promising solution to long-tail scenarios that challenge single-vehicle perception. However, the bandwidth constraints of vehicular networks make transmitting the entire feature map impractical. Recent methods, therefore, adopt a foreground-centric paradigm, transmitting only predicted foreground-region features while discarding the background, which encodes essential context. We propose FadeLead, a foreground-centric framework that overcomes this limitation by learning to encapsulate background context into compact foreground features during training. At the core of our design is a curricular learning strategy that leverages background cues early on but progressively prunes them away, forcing the model to internalize context into foreground representations without transmitting background itself. Extensive experiments on both simulated and real-world benchmarks show that FadeLead outperforms prior methods under different bandwidth settings, underscoring the effectiveness of context-enriched foreground sharing.
>
---
#### [replaced 012] Elastic Weight Consolidation Done Right for Continual Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.18596](https://arxiv.org/pdf/2603.18596)**

> **作者:** Xuan Liu; Xiaobin Chang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Weight regularization methods in continual learning (CL) alleviate catastrophic forgetting by assessing and penalizing changes to important model weights. Elastic Weight Consolidation (EWC) is a foundational and widely used approach within this framework that estimates weight importance based on gradients. However, it has consistently shown suboptimal performance. In this paper, we conduct a systematic analysis of importance estimation in EWC from a gradient-based perspective. For the first time, we find that EWC's reliance on the Fisher Information Matrix (FIM) results in gradient vanishing and inaccurate importance estimation in certain scenarios. Our analysis also reveals that Memory Aware Synapses (MAS), a variant of EWC, imposes unnecessary constraints on parameters irrelevant to prior tasks, termed the redundant protection. Consequently, both EWC and its variants exhibit fundamental misalignments in estimating weight importance, leading to inferior performance. To tackle these issues, we propose the Logits Reversal (LR) operation, a simple yet effective modification that rectifies EWC's importance estimation. Specifically, reversing the logit values during the calculation of FIM can effectively prevent both gradient vanishing and redundant protection. Extensive experiments across various CL tasks and datasets show that the proposed method significantly outperforms existing EWC and its variants. Therefore, we refer to it as EWC Done Right (EWC-DR). Code is available at <this https URL.
>
---
#### [replaced 013] Unsupervised Hyperspectral Image Super-Resolution via Self-Supervised Modality Decoupling
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2412.04802](https://arxiv.org/pdf/2412.04802)**

> **作者:** Songcheng Du; Yang Zou; Zixu Wang; Xingyuan Li; Ying Li; Changjing Shang; Qiang Shen
>
> **备注:** 27 pages, 15 figures
>
> **摘要:** Fusion-based hyperspectral image super-resolution aims to fuse low-resolution hyperspectral images (LR-HSIs) and high-resolution multispectral images (HR-MSIs) to reconstruct high spatial and high spectral resolution images. Current methods typically apply direct fusion from the two modalities without effective supervision, leading to an incomplete perception of deep modality-complementary information and a limited understanding of inter-modality correlations. To address these issues, we propose a simple yet effective solution for unsupervised HMIF, revealing that modality decoupling is key to improving fusion performance. Specifically, we propose an end-to-end self-supervised Modality-Decoupled Spatial-Spectral Fusion (MossFuse) framework that decouples shared and complementary information across modalities and aggregates a concise representation of both LR-HSIs and HR-MSIs to reduce modality redundancy. Also, we introduce the subspace clustering loss as a clear guide to decouple modality-shared features from modality-complementary ones. Systematic experiments over multiple datasets demonstrate that our simple and effective approach consistently outperforms the existing HMIF methods while requiring considerably fewer parameters with reduced inference time. The source source code is in \href{this https URL}{MossFuse}.
>
---
#### [replaced 014] FiGKD: Fine-Grained Knowledge Distillation via High-Frequency Detail Transfer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.11897](https://arxiv.org/pdf/2505.11897)**

> **作者:** Seonghak Kim
>
> **备注:** 18 pages, 6 figures
>
> **摘要:** Knowledge distillation (KD) is a widely adopted technique for transferring knowledge from a high-capacity teacher model to a smaller student model by aligning their output distributions. However, existing methods often underperform in fine-grained visual recognition tasks, where distinguishing subtle differences between visually similar classes is essential. This performance gap stems from the fact that conventional approaches treat the teacher's output logits as a single, undifferentiated signal-assuming all contained information is equally beneficial to the student. Consequently, student models may become overloaded with redundant signals and fail to capture the teacher's nuanced decision boundaries. To address this issue, we propose Fine-Grained Knowledge Distillation (FiGKD), a novel frequency-aware framework that decomposes a model's logits into low-frequency (content) and high-frequency (detail) components using the discrete wavelet transform (DWT). FiGKD selectively transfers only the high-frequency components, which encode the teacher's semantic decision patterns, while discarding redundant low-frequency content already conveyed through ground-truth supervision. Our approach is simple, architecture-agnostic, and requires no access to intermediate feature maps. Extensive experiments on CIFAR-100, TinyImageNet, and multiple fine-grained recognition benchmarks show that FiGKD consistently outperforms state-of-the-art logit-based and feature-based distillation methods across a variety of teacher-student configurations. These findings confirm that frequency-aware logit decomposition enables more efficient and effective knowledge transfer, particularly in resource-constrained settings.
>
---
#### [replaced 015] LLM-Powered Flood Depth Estimation from Social Media Imagery: A Vision-Language Model Framework with Mechanistic Interpretability for Transportation Resilience
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.17108](https://arxiv.org/pdf/2603.17108)**

> **作者:** Nafis Fuad; Xiaodong Qian
>
> **备注:** There is a update in result, which is needed to be addressed
>
> **摘要:** Urban flooding poses an escalating threat to transportation network continuity, yet no operational system currently provides real-time, street-level flood depth information at the centimeter resolution required for dynamic routing, electric vehicle (EV) safety, and autonomous vehicle (AV) operations. This study presents FloodLlama, a fine-tuned open-source vision-language model (VLM) for continuous flood depth estimation from single street-level images, supported by a multimodal sensing pipeline using TikTok data. A synthetic dataset of approximately 190000 images was generated, covering seven vehicle types, four weather conditions, and 41 depth levels (0-40 cm at 1 cm resolution). Progressive curriculum training enabled coarse-to-fine learning, while LLaMA 3.2-11B Vision was fine-tuned using QLoRA. Evaluation across 34797 trials reveals a depth-dependent prompt effect: simple prompts perform better for shallow flooding, whereas chain-of-thought (CoT) reasoning improves performance at greater depths. FloodLlama achieves a mean absolute error (MAE) below 0.97 cm and Acc@5cm above 93.7% for deep flooding, exceeding 96.8% for shallow depths. A five-phase mechanistic interpretability framework identifies layer L23 as the critical depth-encoding transition and enables selective fine-tuning that reduces trainable parameters by 76-80% while maintaining accuracy. The Tier 3 configuration achieves 98.62% accuracy on real-world data and shows strong robustness under visual occlusion. A TikTok-based data pipeline, validated on 676 annotated flood frames from Detroit, demonstrates the feasibility of real-time, crowd-sourced flood sensing. The proposed framework provides a scalable, infrastructure-free solution with direct implications for EV safety, AV deployment, and resilient transportation management.
>
---
#### [replaced 016] PRISM: Video Dataset Condensation with Progressive Refinement and Insertion for Sparse Motion
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2505.22564](https://arxiv.org/pdf/2505.22564)**

> **作者:** Jaehyun Choi; Jiwan Hur; Gyojin Han; Jaemyung Yu; Junmo Kim
>
> **备注:** CVPR 2026
>
> **摘要:** Video dataset condensation aims to reduce the immense computational cost of video processing. However, it faces a fundamental challenge regarding the inseparable interdependence between spatial appearance and temporal dynamics. Prior work follows a static/dynamic disentanglement paradigm where videos are decomposed into static content and auxiliary motion signals. This multi-stage approach often misrepresents the intrinsic coupling of real-world actions. We introduce Progressive Refinement and Insertion for Sparse Motion (PRISM), a holistic approach that treats the video as a unified and fully coupled spatiotemporal structure from the outset. To maximize representational efficiency, PRISM addresses the inherent temporal redundancy of video by avoiding fixed-frame optimization. It begins with minimal temporal anchors and progressively inserts key-frames only where linear interpolation fails to capture non-linear dynamics. These critical moments are identified through gradient misalignments. Such an adaptive process ensures that representational capacity is allocated precisely where needed, minimizing storage requirements while preserving complex motion. Extensive experiments demonstrate that PRISM achieves competitive performance across standard benchmarks while providing state-of-the-art storage efficiency through its sparse and holistically learned representation.
>
---
#### [replaced 017] Mitigating Object Hallucinations in Large Vision-Language Models via Attention Calibration
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2502.01969](https://arxiv.org/pdf/2502.01969)**

> **作者:** Younan Zhu; Linwei Tao; Minjing Dong; Chang Xu
>
> **摘要:** Large Vision-Language Models (LVLMs) exhibit impressive multimodal reasoning capabilities but remain highly susceptible to object hallucination, where models generate responses that are not factually aligned with the visual content. Recent works attribute this issue to an inherent bias of LVLMs where the vision token attention map has spurious focus on certain positions, and propose to mitigate this issue by reordering visual tokens. However, we find that different LVLMs exhibit different correlations between attention and spatial position, which makes existing static solutions difficult to generalize to other LVLMs. To begin with, we investigate the attention bias introduced by image tokens through a toy experiment, in which a blank image is fed into the model to capture its position-dependent bias. We then remove this bias from the original attention map, which already leads to a substantial reduction in hallucinations. This proof of concept validates the core intuition behind attention calibration. Building upon this insight, we propose Dynamic Attention Calibration (DAC), a lightweight, plug-and-play module that leverages contrastive learning to dynamically enforce positional invariance. Unlike static baselines, DAC adapts to different models and inputs in a robust and learnable manner, offering a generalizable solution to mitigate attention-related hallucinations in LVLMs. Comprehensive experiments across multiple benchmarks demonstrate that DAC significantly reduces object hallucination while improving general multimodal alignment. Our method achieves state-of-the-art performance across diverse LVLM architectures on various metrics. Our code is available at this https URL.
>
---
#### [replaced 018] Generating Findings for Jaw Cysts in Dental Panoramic Radiographs Using a GPT-Based VLM: A Preliminary Study on Building a Two-Stage Self-Correction Loop with Structured Output (SLSO) Framework
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.02001](https://arxiv.org/pdf/2510.02001)**

> **作者:** Nanaka Hosokawa; Ryo Takahashi; Tomoya Kitano; Yukihiro Iida; Chisako Muramatsu; Tatsuro Hayashi; Yuta Seino; Xiangrong Zhou; Takeshi Hara; Akitoshi Katsumata; Hiroshi Fujita
>
> **备注:** Revised manuscript; supplementary materials added. Submitted to Diagnostics
>
> **摘要:** Vision-language models (VLMs) such as GPT (Generative Pre-Trained Transformer) have shown potential for medical image interpretation; however, challenges remain in generating reliable radiological findings in clinical practice, as exemplified by dental pathologies. This study proposes a Self-correction Loop with Structured Output (SLSO) framework as an integrated processing methodology to enhance the accuracy and reliability of AI-generated findings for jaw cysts in dental panoramic radiographs. Dental panoramic radiographs with jaw cysts were used to implement a 10-step integrated processing framework incorporating image analysis, structured data generation, tooth number extraction, consistency checking, and iterative regeneration. The framework functioned as an external validation mechanism for GPT outputs. Performance was compared against the conventional Chain-of-Thought (CoT) method across seven evaluation items: transparency, internal structure, borders, root resorption, tooth movement, relationships with other structures, and tooth number. The SLSO framework improved output accuracy for multiple items compared to the CoT method, with the most notable improvements observed in tooth number identification, tooth movement detection, and root resorption assessment. In successful cases, consistently structured outputs were achieved after up to five regenerations. The framework enforced explicit negative finding descriptions and suppressed hallucinations, although accurate identification of extensive lesions spanning multiple teeth remained limited. This investigation established the feasibility of the proposed integrated processing methodology and provided a foundation for future validation studies with larger, more diverse datasets.
>
---
#### [replaced 019] OmniDiT: Extending Diffusion Transformer to Omni-VTON Framework
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.19643](https://arxiv.org/pdf/2603.19643)**

> **作者:** Weixuan Zeng; Pengcheng Wei; Huaiqing Wang; Boheng Zhang; Jia Sun; Dewen Fan; Lin HE; Long Chen; Qianqian Gan; Fan Yang; Tingting Gao
>
> **摘要:** Despite the rapid advancement of Virtual Try-On (VTON) and Try-Off (VTOFF) technologies, existing VTON methods face challenges with fine-grained detail preservation, generalization to complex scenes, complicated pipeline, and efficient inference. To tackle these problems, we propose OmniDiT, an omni Virtual Try-On framework based on the Diffusion Transformer, which combines try-on and try-off tasks into one unified model. Specifically, we first establish a self-evolving data curation pipeline to continuously produce data, and construct a large VTON dataset Omni-TryOn, which contains over 380k diverse and high-quality garment-model-tryon image pairs and detailed text prompts. Then, we employ the token concatenation and design an adaptive position encoding to effectively incorporate multiple reference conditions. To relieve the bottleneck of long sequence computation, we are the first to introduce Shifted Window Attention into the diffusion model, thus achieving a linear complexity. To remedy the performance degradation caused by local window attention, we utilize multiple timestep prediction and an alignment loss to improve generation fidelity. Experiments reveal that, under various complex scenes, our method achieves the best performance in both the model-free VTON and VTOFF tasks and a performance comparable to current SOTA methods in the model-based VTON task.
>
---
#### [replaced 020] TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.01248](https://arxiv.org/pdf/2512.01248)**

> **作者:** Junyuan Zhang; Bin Wang; Qintong Zhang; Fan Wu; Zichen Wen; Jialin Lu; Junjie Shan; Ziqi Zhao; Shuya Yang; Ziling Wang; Ziyang Miao; Huaping Zhong; Yuhang Zang; Xiaoyi Dong; Ka-Ho Chow; Conghui He
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Table recognition (TR) aims to transform table images into semi-structured representations such as HTML or Markdown. As a core component of document parsing, TR has long relied on supervised learning, with recent efforts dominated by fine-tuning vision-language models (VLMs) using labeled data. While VLMs have brought TR to the next level, pushing performance further demands large-scale labeled data that is costly to obtain. Consequently, although proprietary models have continuously pushed the performance boundary, open-source models, often trained with limited resources and, in practice, the only viable option for many due to privacy regulations, still lag far behind. To bridge this gap, we introduce TRivia, a self-supervised fine-tuning method that enables pretrained VLMs to learn TR directly from unlabeled table images in the wild. Built upon Group Relative Policy Optimization, TRivia automatically identifies unlabeled samples that most effectively facilitate learning and eliminates the need for human annotations through a question-answering-based reward mechanism. An attention-guided module generates diverse questions for each table image, and the ability to interpret the recognition results and answer them correctly provides feedback to optimize the TR model. This closed-loop process allows the TR model to autonomously learn to recognize, structure, and reason over tables without labeled data. Leveraging this pipeline, we present TRivia-3B, an open-sourced, compact, and state-of-the-art TR model that surpasses existing systems (e.g., Gemini 2.5 Pro, MinerU2.5) on three popular benchmarks. Model and code are released at: this https URL
>
---
#### [replaced 021] SODA: Sensitivity-Oriented Dynamic Acceleration for Diffusion Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.07057](https://arxiv.org/pdf/2603.07057)**

> **作者:** Tong Shao; Yusen Fu; Guoying Sun; Jingde Kong; Zhuotao Tian; Jingyong Su
>
> **备注:** 23 pages, CVPR 2026 accepted
>
> **摘要:** Diffusion Transformers have become a dominant paradigm in visual generation, yet their low inference efficiency remains a key bottleneck hindering further advancement. Among common training-free techniques, caching offers high acceleration efficiency but often compromises fidelity, whereas pruning shows the opposite trade-off. Integrating caching with pruning achieves a balance between acceleration and generation quality. However, existing methods typically employ fixed and heuristic schemes to configure caching and pruning strategies. While they roughly follow the overall sensitivity trend of generation models to acceleration, they fail to capture fine-grained and complex variations, inevitably skipping highly sensitive computations and leading to quality degradation. Furthermore, such manually designed strategies exhibit poor generalization. To address these issues, we propose SODA, a Sensitivity-Oriented Dynamic Acceleration method that adaptively performs caching and pruning based on fine-grained sensitivity. SODA builds an offline sensitivity error modeling framework across timesteps, layers, and modules to capture the sensitivity to different acceleration operations. The cache intervals are optimized via dynamic programming with sensitivity error as the cost function, minimizing the impact of caching on model sensitivity. During pruning and cache reuse, SODA adaptively determines the pruning timing and rate to preserve computations of highly sensitive tokens, significantly enhancing generation fidelity. Extensive experiments on DiT-XL/2, PixArt-$\alpha$, and OpenSora demonstrate that SODA achieves state-of-the-art generation fidelity under controllable acceleration ratios. Our code is released publicly at: this https URL.
>
---
#### [replaced 022] GenExam: A Multidisciplinary Text-to-Image Exam
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.14232](https://arxiv.org/pdf/2509.14232)**

> **作者:** Zhaokai Wang; Penghao Yin; Xiangyu Zhao; Changyao Tian; Yu Qiao; Wenhai Wang; Jifeng Dai; Gen Luo
>
> **摘要:** Exams are a fundamental test of expert-level intelligence and require integrated understanding, reasoning, and generation. Existing exam-style benchmarks mainly focus on understanding and reasoning tasks, and current generation benchmarks emphasize the illustration of world knowledge and visual concepts, neglecting the evaluation of rigorous drawing exams. We introduce GenExam, the first benchmark for multidisciplinary text-to-image exams, featuring 1,000 samples across 10 subjects with exam-style prompts organized under a four-level taxonomy. Each problem is equipped with ground-truth images and fine-grained scoring points to enable a precise evaluation of semantic correctness and visual plausibility. Experiments on 17 text-to-image and unified models demonstrate the great challenge of GenExam and the huge gap where open-source models consistently lag behind the leading closed-source ones. By framing image generation as an exam, GenExam offers a rigorous assessment of models' ability to integrate understanding, reasoning, and generation, providing insights for on the path to intelligent generative models. Our benchmark and evaluation code are released at this https URL.
>
---
#### [replaced 023] From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20996](https://arxiv.org/pdf/2511.20996)**

> **作者:** Jingxi Chen; Yixiao Zhang; Xiaoye Qian; Zongxia Li; Cornelia Fermuller; Caren Chen; Yiannis Aloimonos
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Images can be viewed as layered compositions, foreground objects over background, with potential occlusions. This layered representation enables independent editing of elements, offering greater flexibility for content creation. Despite the progress in large generative models, decomposing a single image into layers remains challenging due to limited methods and data. We observe a strong connection between layer decomposition and in/outpainting tasks, and propose adapting a diffusion-based inpainting model for layer decomposition using lightweight finetuning. To further preserve detail in the latent space, we introduce a novel multi-modal context fusion module with linear attention complexity. Our model is trained purely on a synthetic dataset constructed from open-source assets and achieves superior performance in object removal and occlusion recovery, unlocking new possibilities in downstream editing and creative applications.
>
---
#### [replaced 024] MOON2.0: Dynamic Modality-balanced Multimodal Representation Learning for E-commerce Product Understanding
- **分类: cs.CV; cs.AI; cs.IR; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.12449](https://arxiv.org/pdf/2511.12449)**

> **作者:** Zhanheng Nie; Chenghan Fu; Daoze Zhang; Junxian Wu; Wanxian Guan; Pengjie Wang; Jian Xu; Bo Zheng
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** Recent Multimodal Large Language Models (MLLMs) have significantly advanced e-commerce product understanding. However, they still face three challenges: (i) the modality imbalance induced by modality mixed training; (ii) underutilization of the intrinsic alignment relationships among visual and textual information within a product; and (iii) limited handling of noise in e-commerce multimodal data. To address these, we propose MOON2.0, a dynamic modality-balanced MultimOdal representation learning framework for e-commerce prOduct uNderstanding. It comprises: (1) a Modality-driven Mixture-of-Experts (MoE) that adaptively processes input samples by their modality composition, enabling Multimodal Joint Learning to mitigate the modality imbalance; (2) a Dual-level Alignment method to better leverage semantic alignment properties inside individual products; and (3) an MLLM-based Image-text Co-augmentation strategy that integrates textual enrichment with visual expansion, coupled with Dynamic Sample Filtering to improve training data quality. We further release MBE2.0, a co-augmented Multimodal representation Benchmark for E-commerce representation learning and evaluation at this https URL. Experiments show that MOON2.0 delivers state-of-the-art zero-shot performance on MBE2.0 and multiple public datasets. Furthermore, attention-based heatmap visualization provides qualitative evidence of improved multimodal alignment of MOON2.0.
>
---
#### [replaced 025] Metaphor-based Jailbreak Attacks on Text-to-Image Models
- **分类: cs.CR; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.10766](https://arxiv.org/pdf/2512.10766)**

> **作者:** Chenyu Zhang; Lanjun Wang; Yiwen Ma; Wenhui Li; Yi Tu; An-An Liu
>
> **备注:** Code is available in \url{this https URL}
>
> **摘要:** Text-to-image (T2I) models commonly incorporate defense mechanisms to prevent the generation of sensitive images. Unfortunately, recent jailbreak attacks have shown that adversarial prompts can effectively bypass these mechanisms and induce T2I models to produce sensitive content, revealing critical safety vulnerabilities. However, existing attack methods implicitly assume that the attacker knows the type of deployed defenses, which limits their effectiveness against unknown or diverse defense mechanisms. In this work, we reveal an underexplored vulnerability of T2I models to metaphor-based jailbreak attacks (MJA), which aims to attack diverse defense mechanisms without prior knowledge of their type by generating metaphor-based adversarial prompts. Specifically, MJA consists of two modules: an LLM-based multi-agent generation module (LMAG) and an adversarial prompt optimization module (APO). LMAG decomposes the generation of metaphor-based adversarial prompts into three subtasks: metaphor retrieval, context matching, and adversarial prompt generation. Subsequently, LMAG coordinates three LLM-based agents to generate diverse adversarial prompts by exploring various metaphors and contexts. To enhance attack efficiency, APO first trains a surrogate model to predict the attack results of adversarial prompts and then designs an acquisition strategy to adaptively identify optimal adversarial prompts. Extensive experiments on T2I models with various external and internal defense mechanisms demonstrate that MJA achieves stronger attack performance while using fewer queries, compared with six baseline methods. Additionally, we provide an in-depth vulnerability analysis suggesting that metaphor-based adversarial prompts evade safety mechanisms by inducing semantic ambiguity, while sensitive images arise from the model's probabilistic interpretation of concealed semantics.
>
---
#### [replaced 026] HalDec-Bench: Benchmarking Hallucination Detector in Image Captioning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.15253](https://arxiv.org/pdf/2603.15253)**

> **作者:** Kuniaki Saito; Risa Shinoda; Shohei Tanaka; Tosho Hirasawa; Fumio Okura; Yoshitaka Ushiku
>
> **备注:** This work was intended as a replacement of arXiv:2511.20515 and any subsequent updates will appear there
>
> **摘要:** Hallucination detection in captions (HalDec) assesses a vision-language model's ability to correctly align image content with text by identifying errors in captions that misrepresent the image. Beyond evaluation, effective hallucination detection is also essential for curating high-quality image-caption pairs used to train VLMs. However, the generalizability of VLMs as hallucination detectors across different captioning models and hallucination types remains unclear due to the lack of a comprehensive benchmark. In this work, we introduce HalDec-Bench, a benchmark designed to evaluate hallucination detectors in a principled and interpretable manner. HalDec-Bench contains captions generated by diverse VLMs together with human annotations indicating the presence of hallucinations, detailed hallucination-type categories, and segment-level labels. The benchmark provides tasks with a wide range of difficulty levels and reveals performance differences across models that are not visible in existing multimodal reasoning or alignment benchmarks. Our analysis further uncovers two key findings. First, detectors tend to recognize sentences appearing at the beginning of a response as correct, regardless of their actual correctness. Second, our experiments suggest that dataset noise can be substantially reduced by using strong VLMs as filters while employing recent VLMs as caption generators. Our project page is available at this https URL.
>
---
#### [replaced 027] MERLIN: Building Low-SNR Robust Multimodal LLMs for Electromagnetic Signals
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.08174](https://arxiv.org/pdf/2603.08174)**

> **作者:** Junyu Shen; Zhendong She; Chenghanyu Zhang; Yuchuang Sun; Luqing Luo; Dingwei Tan; Zonghao Guo; Bo Guo; Zehua Han; Wupeng Xie; Yaxin Mu; Peng Zhang; Peipei Li; Fengxiang Wang; Yangang Sun; Maosong Sun
>
> **摘要:** The paradigm of Multimodal Large Language Models (MLLMs) offers a promising blueprint for advancing the electromagnetic (EM) domain. However, prevailing approaches often deviate from the native MLLM paradigm, instead using task-specific or pipelined architectures that lead to fundamental limitations in model performance and generalization. Fully realizing the MLLM potential in EM domain requires overcoming three main challenges: (1) Data. The scarcity of high-quality datasets with paired EM signals and descriptive text annotations used for MLLMs pre-training; (2) Benchmark. The absence of comprehensive benchmarks to systematically evaluate and compare the performance of models on EM signal-to-text tasks; (3) Model. A critical fragility in low Signal-to-Noise Ratio (SNR) environments, where critical signal features can be obscured, leading to significant performance degradation. To address these challenges, we introduce a tripartite contribution to establish a foundation for MLLMs in the EM domain. First, to overcome data scarcity, we construct and release EM-100k, a large-scale dataset comprising over 100,000 EM signal-text pairs. Second, to enable rigorous and standardized evaluation, we propose EM-Bench, the most comprehensive benchmark featuring diverse downstream tasks spanning from perception to reasoning. Finally, to tackle the core modeling challenge, we present MERLIN, a novel training framework designed not only to align low-level signal representations with high-level semantic text, but also to explicitly enhance model robustness and performance in challenging low-SNR environments. Comprehensive experiments validate our method, showing that MERLIN is state-of-the-art in the EM-Bench and exhibits remarkable robustness in low-SNR settings.
>
---
#### [replaced 028] MoEGCL: Mixture of Ego-Graphs Contrastive Representation Learning for Multi-View Clustering
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.05876](https://arxiv.org/pdf/2511.05876)**

> **作者:** Jian Zhu; Xin Zou; Jun Sun; Cheng Luo; Lei Liu; Lingfang Zeng; Ning Zhang; Bian Wu; Chang Tang; Lirong Dai
>
> **摘要:** In recent years, the advancement of Graph Neural Networks (GNNs) has significantly propelled progress in Multi-View Clustering (MVC). However, existing methods face the problem of coarse-grained graph fusion. Specifically, current approaches typically generate a separate graph structure for each view and then perform weighted fusion of graph structures at the view level, which is a relatively rough strategy. To address this limitation, we present a novel Mixture of Ego-Graphs Contrastive Representation Learning (MoEGCL). It mainly consists of two modules. In particular, we propose an innovative Mixture of Ego-Graphs Fusion (MoEGF), which constructs ego graphs and utilizes a Mixture-of-Experts network to implement fine-grained fusion of ego graphs at the sample level, rather than the conventional view-level fusion. Additionally, we present the Ego Graph Contrastive Learning (EGCL) module to align the fused representation with the view-specific representation. The EGCL module enhances the representation similarity of samples from the same cluster, not merely from the same sample, further boosting fine-grained graph representation. Extensive experiments demonstrate that MoEGCL achieves state-of-the-art results in deep multi-view clustering tasks. The source code is publicly available at this https URL.
>
---
#### [replaced 029] U4D: Uncertainty-Aware 4D World Modeling from LiDAR Sequences
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于4D环境建模任务，旨在解决LiDAR序列中动态场景的不确定性问题。通过估计不确定性地图并分阶段生成，提升模型的几何真实性和时间一致性。**

- **链接: [https://arxiv.org/pdf/2512.02982](https://arxiv.org/pdf/2512.02982)**

> **作者:** Xiang Xu; Alan Liang; Youquan Liu; Linfeng Li; Lingdong Kong; Ziwei Liu; Qingshan Liu
>
> **备注:** CVPR 2026; 20 pages, 7 figures, 11 tables; Code at this https URL
>
> **摘要:** Modeling dynamic 3D environments from LiDAR sequences is central to building reliable 4D worlds for autonomous driving and embodied AI. Existing generative frameworks, however, often treat all spatial regions uniformly, overlooking the varying uncertainty across real-world scenes. This uniform generation leads to artifacts in complex or ambiguous regions, limiting realism and temporal stability. In this work, we present U4D, an uncertainty-aware framework for 4D LiDAR world modeling. Our approach first estimates spatial uncertainty maps from a pretrained segmentation model to localize semantically challenging regions. It then performs generation in a "hard-to-easy" manner through two sequential stages: (1) uncertainty-region modeling, which reconstructs high-entropy regions with fine geometric fidelity, and (2) uncertainty-conditioned completion, which synthesizes the remaining areas under learned structural priors. To further ensure temporal coherence, U4D incorporates a mixture of spatio-temporal (MoST) block that adaptively fuses spatial and temporal representations during diffusion. Extensive experiments show that U4D produces geometrically faithful and temporally consistent LiDAR sequences, advancing the reliability of 4D world modeling for autonomous perception and simulation.
>
---
#### [replaced 030] Learning to See Through a Baby's Eyes: Early Visual Diets Enable Robust Visual Intelligence in Humans and Machines
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14440](https://arxiv.org/pdf/2511.14440)**

> **作者:** Yusen Cai; Qing Lin; Bhargava Satya Nunna; Mengmi Zhang
>
> **摘要:** Newborns perceive the world with low-acuity, color-degraded, and temporally continuous vision, which gradually sharpens as infants develop. To explore the ecological advantages of such staged "visual diets", we train self-supervised learning (SSL) models on object-centric videos under constraints that simulate infant vision: grayscale-to-color (C), blur-to-sharp (A), and preserved temporal continuity (T)-collectively termed CATDiet. For evaluation, we establish a comprehensive benchmark across ten datasets, covering clean and corrupted image recognition, texture-shape cue conflict tests, silhouette recognition, depth-order classification, and the visual cliff paradigm. All CATDiet variants demonstrate enhanced robustness in object recognition, despite being trained solely on object-centric videos. Remarkably, models also exhibit biologically aligned developmental patterns, including neural plasticity changes mirroring synaptic density in macaque V1 and behaviors resembling infants' visual cliff responses. Building on these insights, CombDiet initializes SSL with CATDiet before standard training while preserving temporal continuity. Trained on object-centric or head-mounted infant videos, CombDiet outperforms standard SSL on both in-domain and out-of-domain object recognition and depth perception. Together, these results suggest that the developmental progression of early infant visual experience offers a powerful reverse-engineering framework for understanding the emergence of robust visual intelligence in machines. All code, data, and models are available at Github.
>
---
#### [replaced 031] PhyUnfold-Net: Advancing Remote Sensing Change Detection with Physics-Guided Deep Unfolding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.19566](https://arxiv.org/pdf/2603.19566)**

> **作者:** Zelin Lei; Yaoxing Ren; Jiaming Chang
>
> **备注:** 18 pages, 8 figures, 9 tables. Appendix included
>
> **摘要:** Bi-temporal change detection is highly sensitive to acquisition discrepancies, including illumination, season, and atmosphere, which often cause false alarms. We observe that genuine changes exhibit higher patch-wise singular-value entropy (SVE) than pseudo changes in the feature-difference space. Motivated by this physical prior, we propose PhyUnfold-Net, a physics-guided deep unfolding framework that formulates change detection as an explicit decomposition problem. The proposed Iterative Change Decomposition Module (ICDM) unrolls a multi-step solver to progressively separate mixed discrepancy features into a change component and a nuisance component. To stabilize this process, we introduce a staged Exploration-and-Constraint loss (S-SEC), which encourages component separation in early steps while constraining nuisance magnitude in later steps to avoid degenerate solutions. We further design a Wavelet Spectral Suppression Module (WSSM) to suppress acquisition-induced spectral mismatch before decomposition. Experiments on four benchmarks show improvements over state-of-the-art methods, with gains under challenging conditions.
>
---
#### [replaced 032] Gradient Descent Provably Solves Nonlinear Tomographic Reconstruction
- **分类: cs.CV; math.OC; physics.med-ph**

- **链接: [https://arxiv.org/pdf/2310.03956](https://arxiv.org/pdf/2310.03956)**

> **作者:** Sara Fridovich-Keil; Fabrizio Valdivia; Gordon Wetzstein; Benjamin Recht; Mahdi Soltanolkotabi
>
> **摘要:** In computed tomography (CT), the forward model consists of a linear Radon transform followed by an exponential nonlinearity based on the attenuation of light according to the Beer-Lambert Law. Conventional reconstruction often involves inverting this nonlinearity and then solving a linear inverse problem. However, this nonlinear measurement preprocessing is poorly conditioned in the vicinity of high-density materials, such as metal. This preprocessing makes CT reconstruction methods numerically sensitive and susceptible to artifacts near high-density regions. In this paper, we study a technique where the signal is directly reconstructed from raw measurements through the nonlinear forward model. Though this optimization is nonconvex, we show that gradient descent provably converges to the global optimum at a geometric rate, perfectly reconstructing the underlying signal with a near minimal number of random measurements. We also prove similar results in the under-determined setting where the number of measurements is significantly smaller than the dimension of the signal. This is achieved by enforcing prior structural information about the signal through constraints on the optimization variables. We illustrate the benefits of direct nonlinear CT reconstruction with cone-beam CT experiments on synthetic and real 3D volumes, in which metal artifacts are reduced compared to standard linear reconstruction methods. Our experiments also demonstrate that logarithmic preprocessing alone is sufficient to produce metal artifacts, even in the absence of other causes such as beam hardening.
>
---
#### [replaced 033] Do Vision-Language Models Measure Up? Benchmarking Visual Measurement Reading with MeasureBench
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.26865](https://arxiv.org/pdf/2510.26865)**

> **作者:** Fenfen Lin; Yesheng Liu; Haiyu Xu; Chen Yue; Zheqi He; Mingxuan Zhao; Miguel Hu Chen; Jiakang Liu; JG Yao; Xi Yang
>
> **备注:** Project page: this https URL
>
> **摘要:** Reading measurement instruments is effortless for humans and requires relatively little domain expertise, yet it remains surprisingly challenging for current vision-language models (VLMs) as we find in preliminary evaluation. In this work, we introduce MeasureBench, a benchmark on visual measurement reading covering both real-world and synthesized images of various types of measurements, along with an extensible pipeline for data synthesis. Our pipeline procedurally generates a specified type of gauge with controllable visual appearance, enabling scalable variation in key details such as pointers, scales, fonts, lighting, and clutter. Evaluation on popular proprietary and open-weight VLMs shows that even the strongest frontier VLMs struggle with measurement reading in general. We have also conducted preliminary experiments with reinforcement finetuning (RFT) over synthetic data, and find a significant improvement on both in-domain synthetic subset and real-world images. Our analysis highlights a fundamental limitation of current VLMs in fine-grained spatial grounding. We hope this resource and our code releases can help future advances on visually grounded numeracy and precise spatial perception of VLMs, bridging the gap between recognizing numbers and measuring the world.
>
---
#### [replaced 034] Where, What, Why: Toward Explainable 3D-GS Watermarking
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.08809](https://arxiv.org/pdf/2603.08809)**

> **作者:** Mingshu Cai; Jiajun Li; Osamu Yoshie; Yuya Ieiri; Yixuan Li
>
> **备注:** CVPR 2026
>
> **摘要:** As 3D Gaussian Splatting becomes the de facto representation for interactive 3D assets, robust yet imperceptible watermarking is critical. We present a representation-native framework that separates where to write from how to preserve quality. A Trio-Experts module operates directly on Gaussian primitives to derive priors for carrier selection, while a Safety and Budget Aware Gate (SBAG) allocates Gaussians to watermark carriers, optimized for bit resilience under perturbation and bitrate budgets, and to visual compensators that are insulated from watermark loss. To maintain fidelity, we introduce a channel-wise group mask that controls gradient propagation for carriers and compensators, thereby limiting Gaussian parameter updates, repairing local artifacts, and preserving high-frequency details without increasing runtime. Our design yields view-consistent watermark persistence and strong robustness against common image distortions such as compression and noise, while achieving a favorable robustness-quality trade-off compared with prior methods. In addition, decoupled finetuning provides per-Gaussian attributions that reveal where the message is carried and why those carriers are selected, enabling auditable explainability. Compared with state-of-the-art methods, our approach achieves a PSNR improvement of +0.83 dB and a bit-accuracy gain of +1.24%.
>
---
#### [replaced 035] Gaze-VLM:Bridging Gaze and VLMs through Attention Regularization for Egocentric Understanding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.21356](https://arxiv.org/pdf/2510.21356)**

> **作者:** Anupam Pani; Yanchao Yang
>
> **摘要:** Eye gaze offers valuable cues about attention, short-term intent, and future actions, making it a powerful signal for modeling egocentric behavior. In this work, we propose a gaze-regularized framework that enhances VLMs for two key egocentric understanding tasks: fine-grained future event prediction and current activity understanding. Unlike prior approaches that rely solely on visual inputs or use gaze as an auxiliary input signal , our method uses gaze only during training. We introduce a gaze-regularized attention mechanism that aligns model focus with human visual gaze. This design is flexible and modular, allowing it to generalize across multiple VLM architectures that utilize attention. Experimental results show that our approach improves semantic prediction scores by up to 11 for future event prediction and around 7 for current activity understanding, compared to the corresponding baseline models trained without gaze regularization. These results highlight the value of gaze-guided training in improving the accuracy and robustness of egocentric VLMs. Overall, this work establishes a foundation for using human gaze to enhance the predictive capabilities of VLMs in real-world scenarios like assistive robots and human-machine collaboration. Code and additional information is available at: this https URL
>
---
#### [replaced 036] LPNSR: Prior-Enhanced Diffusion Image Super-Resolution via LR-Guided Noise Prediction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.21045](https://arxiv.org/pdf/2603.21045)**

> **作者:** Shuwei Huang; Shizhuo Liu; Zijun Wei
>
> **摘要:** Diffusion-based image super-resolution (SR), which aims to reconstruct high-resolution (HR) images from corresponding low-resolution (LR) observations, faces a fundamental trade-off between inference efficiency and reconstruction quality. The state-of-the-art residual-shifting diffusion framework achieves efficient 4-step inference, yet suffers from severe performance degradation in compact sampling trajectories. This is mainly attributed to two core limitations: the inherent suboptimality of unconstrained random Gaussian noise in intermediate steps, which leads to error accumulation and insufficient LR prior guidance, and the initialization bias caused by naive bicubic upsampling. In this paper, we propose LPNSR, a prior-enhanced efficient diffusion framework to address these issues. We first mathematically derive the closed-form analytical solution of the optimal intermediate noise for the residual-shifting diffusion paradigm, and accordingly design an LR-guided multi-input-aware noise predictor to replace random Gaussian noise, embedding LR structural priors into the reverse process while fully preserving the framework's core efficient residual-shifting mechanism. We further mitigate initial bias with a high-quality pre-upsampling network to optimize the diffusion starting point. With a compact 4-step trajectory, LPNSR can be optimized in an end-to-end manner. Extensive experiments demonstrate that LPNSR achieves state-of-the-art perceptual performance on both synthetic and real-world datasets, without relying on any large-scale text-to-image priors. The source code of our method can be found at this https URL.
>
---
#### [replaced 037] Beyond Matching to Tiles: Bridging Unaligned Aerial and Satellite Views for Vision-Only UAV Navigation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.22153](https://arxiv.org/pdf/2603.22153)**

> **作者:** Kejia Liu; Haoyang Zhou; Ruoyu Xu; Peicheng Wang; Mingli Song; Haofei Zhang
>
> **备注:** Accepted as a conference paper by CVPR2026
>
> **摘要:** Recent advances in cross-view geo-localization (CVGL) methods have shown strong potential for supporting unmanned aerial vehicle (UAV) navigation in GNSS-denied environments. However, existing work predominantly focuses on matching UAV views to onboard map tiles, which introduces an inherent trade-off between accuracy and storage overhead, and overlooks the importance of the UAV's heading during navigation. Moreover, the substantial discrepancies and varying overlaps in cross-view scenarios have been insufficiently considered, limiting their generalization to real-world scenarios. In this paper, we present Bearing-UAV, a purely vision-driven cross-view navigation method that jointly predicts UAV absolute location and heading from neighboring features, enabling accurate, lightweight, and robust navigation in the wild. Our method leverages global and local structural features and explicitly encodes relative spatial relationships, making it robust to cross-view variations, misalignment, and feature-sparse conditions. We also present Bearing-UAV-90k, a multi-city benchmark for evaluating cross-view localization and navigation. Extensive experiments show encouraging results that Bearing-UAV yields lower localization error than previous matching/retrieval paradigm across diverse terrains. Our code and dataset will be made publicly available.
>
---
#### [replaced 038] Coarse-to-Fine Hierarchical Alignment for UAV-based Human Detection using Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13869](https://arxiv.org/pdf/2512.13869)**

> **作者:** Wenda Li; Meng Wu; Liangzhao Chen; Sungmin Eum; Heesung Kwon; Qing Qu
>
> **摘要:** Training object detectors demands extensive, task-specific annotations, yet this requirement becomes impractical in UAV-based human detection due to constantly shifting target distributions and the scarcity of labeled images. As a remedy, synthetic simulators are adopted to generate annotated data, with a low annotation cost. However, the domain gap between synthetic and real images hinders the model from being effectively applied to the target domain. Accordingly, we introduce Coarse-to-Fine Hierarchical Alignment (CFHA), a three-stage diffusion-based framework designed to transform synthetic data for UAV-based human detection, narrowing the domain gap while preserving the original synthetic labels. CFHA explicitly decouples global style and local content domain discrepancies and bridges those gaps using three modules: (1) Global Style Transfer -- a diffusion model aligns color, illumination, and texture statistics of synthetic images to the realistic style, using only a small real reference set; (2) Local Refinement -- a super-resolution diffusion model is used to facilitate fine-grained and photorealistic details for the small objects, such as human instances, preserving shape and boundary integrity; (3) Hallucination Removal -- a module that filters out human instances whose visual attributes do not align with real-world data to make the human appearance closer to the target distribution. Extensive experiments on public UAV Sim2Real detection benchmarks demonstrate that our methods significantly improve the detection accuracy compared to the non-transformed baselines. Specifically, our method achieves up to $+14.1$ improvement of mAP50 on Semantic-Drone benchmark. Ablation studies confirm the complementary roles of the global and local stages and highlight the importance of hierarchical alignment. The code is released at \href{this https URL}{this url}.
>
---
#### [replaced 039] POVQA: Preference-Optimized Video Question Answering with Rationales for Data Efficiency
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2510.01009](https://arxiv.org/pdf/2510.01009)**

> **作者:** Ashim Dahal; Ankit Ghimire; Saydul Akbar Murad; Nick Rahimi
>
> **备注:** Accepted in MAR at CVPR Workshop (Proceedings Track)
>
> **摘要:** Video Question Answering (VQA) with Large Vision Language Models (LVLMs) has gained significant traction in research ever since the Flamingo was introduced by Deepmind. Recent advancements in large context/long video question answering have allowed VQA tasks to have context window of 1500+ frames. However, this only leads to 50 seconds of video footage without losing any significant information. We introduce POVQA, a data-efficient pipeline that compresses each second of video into a single temporally pooled image (via motion blur and weighted averaging variants) and then align LVLMs with lightweight supervision. Concretely, we build 1 fps input sources using Blend Blur with Last Frame, Weighted Average, Exponential and Ramp pooling and fine-tune QWEN-2.5-VL 7B with supervised two turn target including reasoning and final answer. We apply Supervised Fine Tuning (SFT) and Direct Preference Optimization (DPO) on our novel dataset ReasonVQA consisting of 12 movies with 239 human annotated question-answer with reasoning prompts. On our ReasonVQA dataset, this method dramatically improves performance over pooled baselines: F1 score improves from 0.212 to 0.543, BLEU-4 from 0.031 to 0.291, and ROUGE-L from 0.196 to 0.528. Rationale quality also significantly increases. Cross-evaluation of SFT + DPO on various pooling functions show that the gains persist regardless of the pooling scheme used at train or test time, indicating strong robustness on summarization of temporal evidence. Similar observations were made on zero-shot in TVQA.
>
---
#### [replaced 040] Learning to Stylize by Learning to Destylize: A Scalable Paradigm for Supervised Style Transfer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.05970](https://arxiv.org/pdf/2509.05970)**

> **作者:** Ye Wang; Zili Yi; Yibo Zhang; Peng Zheng; Xuping Xie; Jiang Lin; Yijun Li; Yilin Wang; Rui Ma
>
> **备注:** Our project page: this https URL
>
> **摘要:** This paper introduces a scalable paradigm for supervised style transfer by inverting the problem: instead of learning to stylize directly, we learn to destylize, reducing stylistic elements from artistic images to recover their natural counterparts and thereby producing authentic, pixel-aligned training pairs at scale. To realize this paradigm, we propose DeStylePipe, a progressive, multi-stage destylization framework that begins with global general destylization, advances to category-wise instruction adaptation, and ultimately deploys specialized model adaptation for complex styles that prompt engineering alone cannot handle. Tightly integrated into this pipeline, DestyleCoT-Filter employs Chain-of-Thought reasoning to assess content preservation and style removal at each stage, routing challenging samples forward while discarding persistently low-quality pairs. Built on this framework, we construct DeStyle-350K, a large-scale dataset aligning diverse artistic styles with their underlying content. We further introduce BCS-Bench, a benchmark featuring balanced content generality and style diversity for systematic evaluation. Extensive experiments demonstrate that models trained on DeStyle-350K achieve superior stylization quality, validating destylization as a reliable and scalable supervision paradigm for style transfer.
>
---
#### [replaced 041] GeoDiffMM: Geometry-Guided Conditional Diffusion for Motion Magnification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.08325](https://arxiv.org/pdf/2512.08325)**

> **作者:** Xuedeng Liu; Jiabao Guo; Zheng Zhang; Fei Wang; Zhi Liu; Dan Guo
>
> **摘要:** Video Motion Magnification (VMM) amplifies subtle macroscopic motions to a perceptible level. Recently, existing mainstream Eulerian approaches address amplification-induced noise via decoupling representation learning such as texture, shape and frequency schemes, but they still struggle to mitigate the interference of photon noise on true micro-motion when motion displacements are very small. We propose GeoDiffMM, a novel diffusion-based Lagrangian VMM framework conditioned on optical flow as a geometric cue, enabling structurally consistent motion magnification. Specifically, we design a Noise-Free Optical Flow Augmentation strategy that synthesizes diverse nonrigid motion fields without photon noise as supervision, helping the model learn more accurate geometry-aware optical flow and generalize better. Next, we develop a Diffusion Motion Magnifier that conditions the denoising process on (i) optical flow as a geometry prior and (ii) a learnable magnification factor controlling magnitude, thereby selectively amplifying motion components consistent with scene semantics and structure. Finally, we perform Flow-based Video Synthesis to map the amplified motion back to the image domain with high fidelity. Extensive experiments on real and synthetic datasets show that GeoDiffMM outperforms state-of-the-art methods and significantly improves motion magnification.
>
---
#### [replaced 042] Refine Now, Query Fast: A Decoupled Refinement Paradigm for Implicit Neural Fields
- **分类: cs.LG; cs.CE; cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2602.15155](https://arxiv.org/pdf/2602.15155)**

> **作者:** Tianyu Xiong; Skylar Wurster; Han-Wei Shen
>
> **备注:** Accepted to ICLR 2026. Code available at this https URL
>
> **摘要:** Implicit Neural Representations (INRs) have emerged as promising surrogates for large 3D scientific simulations due to their ability to continuously model spatial and conditional fields, yet they face a critical fidelity-speed dilemma: deep MLPs suffer from high inference cost, while efficient embedding-based models lack sufficient expressiveness. To resolve this, we propose the Decoupled Representation Refinement (DRR) architectural paradigm. DRR leverages a deep refiner network, alongside non-parametric transformations, in a one-time offline process to encode rich representations into a compact and efficient embedding structure. This approach decouples slow neural networks with high representational capacity from the fast inference path. We introduce DRR-Net, a simple network that validates this paradigm, and a novel data augmentation strategy, Variational Pairs (VP) for improving INRs under complex tasks like high-dimensional surrogate modeling. Experiments on several ensemble simulation datasets demonstrate that our approach achieves state-of-the-art fidelity, while being up to 27$\times$ faster at inference than high-fidelity baselines and remaining competitive with the fastest models. The DRR paradigm offers an effective strategy for building powerful and practical neural field surrogates and INRs in broader applications, with a minimal compromise between speed and quality.
>
---
#### [replaced 043] Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search
- **分类: cs.CV; cs.AI; cs.IR**

- **链接: [https://arxiv.org/pdf/2601.13719](https://arxiv.org/pdf/2601.13719)**

> **作者:** Xinlei Yin; Xiulian Peng; Xiao Li; Zhiwei Xiong; Yan Lu
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Long video understanding presents significant challenges for vision-language models due to extremely long context windows. Existing solutions relying on naive chunking strategies with retrieval-augmented generation, typically suffer from information fragmentation and a loss of global coherence. We present HAVEN, a unified framework for long-video understanding that enables coherent and comprehensive reasoning by integrating audiovisual entity cohesion and hierarchical video indexing with agentic search. First, we preserve semantic consistency by integrating entity-level representations across visual and auditory streams, while organizing content into a structured hierarchy spanning global summary, scene, segment, and entity levels. Then we employ an agentic search mechanism to enable dynamic retrieval and reasoning across these layers, facilitating coherent narrative reconstruction and fine-grained entity tracking. Extensive experiments demonstrate that our method achieves good temporal coherence, entity consistency, and retrieval efficiency, establishing a new state-of-the-art with an overall accuracy of 84.1% on LVBench. Notably, it achieves outstanding performance in the challenging reasoning category, reaching 80.1%. These results highlight the effectiveness of structured, multimodal reasoning for comprehensive and context-consistent understanding of long-form videos.
>
---
#### [replaced 044] Towards Inclusive Communication: A Unified Framework for Generating Spoken Language from Sign, Lip, and Audio
- **分类: cs.CV; cs.MM; eess.AS; eess.IV**

- **简介: 该论文属于多模态语言生成任务，旨在解决聋哑人群沟通障碍问题。通过统一框架处理手语、唇读和音频，提升语音文本生成效果。**

- **链接: [https://arxiv.org/pdf/2508.20476](https://arxiv.org/pdf/2508.20476)**

> **作者:** Jeong Hun Yeo; Hyeongseop Rha; Sungjune Park; Junil Won; Yong Man Ro
>
> **备注:** Updated the professional title of the corresponding author. Added an Acknowledgement section
>
> **摘要:** Audio is the primary modality for human communication and has driven the success of Automatic Speech Recognition (ASR) technologies. However, such audio-centric systems inherently exclude individuals who are deaf or hard of hearing. Visual alternatives such as sign language and lip reading offer effective substitutes, and recent advances in Sign Language Translation (SLT) and Visual Speech Recognition (VSR) have improved audio-less communication. Yet, these modalities have largely been studied in isolation, and their integration within a unified framework remains underexplored. In this paper, we propose the first unified framework capable of handling diverse combinations of sign language, lip movements, and audio for spoken-language text generation. We focus on three main objectives: (i) designing a unified, modality-agnostic architecture capable of effectively processing heterogeneous inputs; (ii) exploring the underexamined synergy among modalities, particularly the role of lip movements as non-manual cues in sign language comprehension; and (iii) achieving performance on par with or superior to state-of-the-art models specialized for individual tasks. Building on this framework, we achieve performance on par with or better than task-specific state-of-the-art models across SLT, VSR, ASR, and Audio-Visual Speech Recognition. Furthermore, our analysis reveals a key linguistic insight: explicitly modeling lip movements as a distinct modality significantly improves SLT performance by capturing critical non-manual cues.
>
---
#### [replaced 045] DINO-Tok: Adapting DINO for Visual Tokenizers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20565](https://arxiv.org/pdf/2511.20565)**

> **作者:** Mingkai Jia; Mingxiao Li; Zhijian Shu; Anlin Zheng; Liaoyuan Fan; Jiaxin Guo; Tianxing Shi; Dongyue Lu; Zeming Li; Xiaoyang Guo; Xiaojuan Qi; Xiao-Xiao Long; Qian Zhang; Ping Tan; Wei Yin
>
> **摘要:** Recent advances in visual generation have emphasized the importance of Latent Generative Models (LGMs), which critically depend on effective visual tokenizers to bridge pixels and semantic representations. However, tokenizers constructed on pre-trained vision foundation models (VFMs) often struggle to balance semantic richness and reconstruction fidelity in high-dimensional latent spaces. In this paper, we introduce DINO-Tok, a visual tokenizer built upon a frozen DINO encoder that supports both continuous autoencoding (DINO-Tok-AE) and discrete vector-quantization (DINO-Tok-VQ). By unifying hierarchical representations from both shallow fine-grained features and deep global semantics into an information-complete latent space, DINO-Tok preserves texture details while maintaining \textit{semantic consistency} for generation. We further investigate VQ in frozen semantic feature spaces of high dimensionality, where information dilution and codebook collapse frequently arise. To address this issue, we propose Dominant-Subspace Quantization (DSQ), which leverages a global PCA analysis to select principal components while suppressing noisy dimensions, thereby stabilizing codebook optimization and improving reconstruction and generation quality. On ImageNet 256x256, DINO-Tok achieves strong reconstruction performance, achieving 0.28 rFID for continuous autoencoding and 1.10 rFID for discrete VQ, as well as strong few-step generation performance 1.82 gFID for diffusion and 2.44 gFID for autoregressive generation. These results demonstrate that pre-trained VFMs such as DINO can be directly adapted into high-fidelity, semantically aligned visual tokenizers for next-generation latent generative models. Code will be publicly available at this https URL.
>
---
#### [replaced 046] DI3CL: Contrastive Learning With Dynamic Instances and Contour Consistency for SAR Land-Cover Classification Foundation Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.07808](https://arxiv.org/pdf/2511.07808)**

> **作者:** Zhongle Ren; Hui Ding; Kai Wang; Biao Hou; Xingyu Luo; Weibin Li; Licheng Jiao
>
> **备注:** 16 pages, 7 figures;Accepted for publication in IEEE Transactions on Image Processing (TIP)
>
> **摘要:** Although significant advances have been achieved in SAR land-cover classification, recent methods remain predominantly focused on supervised learning, which relies heavily on extensive labeled datasets. This dependency not only limits scalability and generalization but also restricts adaptability to diverse application scenarios. In this paper, a general-purpose foundation model for SAR land-cover classification is developed, serving as a robust cornerstone to accelerate the development and deployment of various downstream models. Specifically, a Dynamic Instance and Contour Consistency Contrastive Learning (DI3CL) pre-training framework is presented, which incorporates a Dynamic Instance (DI) module and a Contour Consistency (CC) module. DI module enhances global contextual awareness by enforcing local consistency across different views of the same region. CC module leverages shallow feature maps to guide the model to focus on the geometric contours of SAR land-cover objects, thereby improving structural discrimination. Additionally, to enhance robustness and generalization during pre-training, a large-scale and diverse dataset named SARSense, comprising 460,532 SAR images, is constructed to enable the model to capture comprehensive and representative features. To evaluate the generalization capability of our foundation model, we conducted extensive experiments across a variety of SAR land-cover classification tasks, including SAR land-cover mapping, water body detection, and road extraction. The results consistently demonstrate that the proposed DI3CL outperforms existing methods. Our code and pre-trained weights are publicly available at: this https URL.
>
---
#### [replaced 047] ScaleEdit-12M: Scaling Open-Source Image Editing Data Generation via Multi-Agent Framework
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.20644](https://arxiv.org/pdf/2603.20644)**

> **作者:** Guanzhou Chen; Erfei Cui; Changyao Tian; Danni Yang; Ganlin Yang; Yu Qiao; Hongsheng Li; Gen Luo; Hongjie Zhang
>
> **摘要:** Instruction-based image editing has emerged as a key capability for unified multimodal models (UMMs), yet constructing large-scale, diverse, and high-quality editing datasets without costly proprietary APIs remains challenging. Previous image editing datasets either rely on closed-source models for annotation, which prevents cost-effective scaling, or employ fixed synthetic editing pipelines, which suffer from limited quality and generalizability. To address these challenges, we propose ScaleEditor, a fully open-source hierarchical multi-agent framework for end-to-end construction of large-scale, high-quality image editing datasets. Our pipeline consists of three key components: source image expansion with world-knowledge infusion, adaptive multi-agent editing instruction-image synthesis, and a task-aware data quality verification mechanism. Using ScaleEditor, we curate ScaleEdit-12M, the largest open-source image editing dataset to date, spanning 23 task families across diverse real and synthetic domains. Fine-tuning UniWorld-V1 and Bagel on ScaleEdit yields consistent gains, improving performance by up to 10.4% on ImgEdit and 35.1% on GEdit for general editing benchmarks and by up to 150.0% on RISE and 26.5% on KRIS-Bench for knowledge-infused benchmarks. These results demonstrate that open-source, agentic pipelines can approach commercial-grade data quality while retaining cost-effectiveness and scalability. Both the framework and dataset will be open-sourced.
>
---
#### [replaced 048] PaperBanana: Automating Academic Illustration for AI Scientists
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于学术插图生成任务，旨在解决科研中手工制作高质量图表耗时的问题。提出PaperBanana框架，自动完成图表生成与优化。**

- **链接: [https://arxiv.org/pdf/2601.23265](https://arxiv.org/pdf/2601.23265)**

> **作者:** Dawei Zhu; Rui Meng; Yale Song; Xiyu Wei; Sujian Li; Tomas Pfister; Jinsung Yoon
>
> **备注:** Add Citations
>
> **摘要:** Despite rapid advances in autonomous AI scientists powered by language models, generating publication-ready illustrations remains a labor-intensive bottleneck in the research workflow. To lift this burden, we introduce PaperBanana, an agentic framework for automated generation of publication-ready academic illustrations. Powered by state-of-the-art VLMs and image generation models, PaperBanana orchestrates specialized agents to retrieve references, plan content and style, render images, and iteratively refine via self-critique. To rigorously evaluate our framework, we introduce PaperBananaBench, comprising 292 test cases for methodology diagrams curated from NeurIPS 2025 publications, covering diverse research domains and illustration styles. Comprehensive experiments demonstrate that PaperBanana consistently outperforms leading baselines in faithfulness, conciseness, readability, and aesthetics. We further show that our method effectively extends to the generation of high-quality statistical plots. Collectively, PaperBanana paves the way for the automated generation of publication-ready illustrations.
>
---
#### [replaced 049] Quantifying Noise of Dynamic Vision Sensor
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2404.01948](https://arxiv.org/pdf/2404.01948)**

> **作者:** Evgeny V. Votyakov; Alessandro Artusi
>
> **备注:** 5 pages, 4 figures, submitted to the IEEE Signal Processing Letters
>
> **摘要:** Dynamic visual sensors (DVS) are characterized by a large amount of background activity (BA) noise, which it is mixed with the original (cleaned) sensor signal. The dynamic nature of the signal and the absence in practical application of the ground truth, it clearly makes difficult to distinguish between noise and the cleaned sensor signals using standard image processing techniques. In this letter, a new technique is presented to characterise BA noise derived from the Detrended Fluctuation Analysis (DFA). The proposed technique can be used to address an existing DVS issues, which is how to quantitatively characterised noise and signal without ground truth, and how to derive an optimal denoising filter parameters. The solution of the latter problem is demonstrated for the popular real moving-car dataset.
>
---
#### [replaced 050] CHIMERA: Adaptive Cache Injection and Semantic Anchor Prompting for Zero-shot Image Morphing with Morphing-oriented Metrics
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.07155](https://arxiv.org/pdf/2512.07155)**

> **作者:** Dahyeon Kye; Jeahun Sung; Minkyu Jeon; Jihyong Oh
>
> **备注:** Please visit our project page at this https URL
>
> **摘要:** Recent diffusion-based image morphing methods typically interpolate inverted latents and reuse limited conditioning signals, which often yields unstable intermediates for heterogeneous endpoint pairs. In particular, (i) feature reuse is usually partial or non-adaptive, leading to abrupt structural changes or over-smoothing, and (ii) text conditions are commonly obtained independently per endpoint and then interpolated, which can introduce incompatible semantics. We present CHIMERA, a novel zero-shot diffusion morphing framework that addresses both issues via inversion-guided denoising with complementary feature reuse and text conditioning. ACI caches a broader set of multi-scale diffusion features beyond Key--Value-only reuse during DDIM inversion, and re-injects them with layer- and timestep-aware scheduling to stabilize denoising and enable gradual fusion. Semantic Anchor Prompting (SAP) uses a vision-language model to generate a shared anchor-prompt and anchor-conditioned endpoint prompts, and injects the anchor into cross-attention to improve intermediate semantic coherence. Finally, we propose Global-Local Consistency Score (GLCS), a morphing-oriented metric that jointly captures global domain harmonization and local transition smoothness. Extensive experiments and user study show that CHIMERA produces smoother and more semantically consistent morphs than prior methods, while remaining efficient and applicable across diverse diffusion backbones without retraining. Code and the project page will be released.
>
---
#### [replaced 051] LiveWorld: Simulating Out-of-Sight Dynamics in Generative Video World Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.07145](https://arxiv.org/pdf/2603.07145)**

> **作者:** Zicheng Duan; Jiatong Xia; Zeyu Zhang; Wenbo Zhang; Gengze Zhou; Chenhui Gou; Yefei He; Feng Chen; Xinyu Zhang; Lingqiao Liu
>
> **摘要:** Recent generative video world models aim to simulate visual environment evolution, allowing an observer to interactively explore the scene via camera control. However, they implicitly assume that the world only evolves within the observer's field of view. Once an object leaves the observer's view, its state is "frozen" in memory, and revisiting the same region later often fails to reflect events that should have occurred in the meantime. In this work, we identify and formalize this overlooked limitation as the "out-of-sight dynamics" problem, which impedes video world models from representing a continuously evolving world. To address this issue, we propose LiveWorld, a novel framework that extends video world models to support persistent world evolution. Instead of treating the world as static observational memory, LiveWorld models a persistent global state composed of a static 3D background and dynamic entities that continue evolving even when unobserved. To maintain these unseen dynamics, LiveWorld introduces a monitor-based mechanism that autonomously simulates the temporal progression of active entities and synchronizes their evolved states upon revisiting, ensuring spatially coherent rendering. For evaluation, we further introduce LiveBench, a dedicated benchmark for the task of maintaining out-of-sight dynamics. Extensive experiments show that LiveWorld enables persistent event evolution and long-term scene consistency, bridging the gap between existing 2D observation-based memory and true 4D dynamic world simulation. The baseline and benchmark will be publicly available at this https URL.
>
---
#### [replaced 052] LoD-Loc v3: Generalized Aerial Localization in Dense Cities using Instance Silhouette Alignment
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 论文提出LoD-Loc v3，解决密集城市中航空视觉定位问题。通过实例轮廓对齐和合成数据增强，提升模型泛化能力和定位精度。**

- **链接: [https://arxiv.org/pdf/2603.19609](https://arxiv.org/pdf/2603.19609)**

> **作者:** Shuaibang Peng; Juelin Zhu; Xia Li; Kun Yang; Maojun Zhang; Yu Liu; Shen Yan
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** We present LoD-Loc v3, a novel method for generalized aerial visual localization in dense urban environments. While prior work LoD-Loc v2 achieves localization through semantic building silhouette alignment with low-detail city models, it suffers from two key limitations: poor cross-scene generalization and frequent failure in dense building scenes. Our method addresses these challenges through two key innovations. First, we develop a new synthetic data generation pipeline that produces InsLoD-Loc - the largest instance segmentation dataset for aerial imagery to date, comprising 100k images with precise instance building annotations. This enables trained models to exhibit remarkable zero-shot generalization capability. Second, we reformulate the localization paradigm by shifting from semantic to instance silhouette alignment, which significantly reduces pose estimation ambiguity in dense scenes. Extensive experiments demonstrate that LoD-Loc v3 outperforms existing state-of-the-art (SOTA) baselines, achieving superior performance in both cross-scene and dense urban scenarios with a large margin. The project is available at this https URL.
>
---
#### [replaced 053] From Noisy Labels to Intrinsic Structure: A Geometric-Structural Dual-Guided Framework for Noise-Robust Medical Image Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.02419](https://arxiv.org/pdf/2509.02419)**

> **作者:** Tao Wang; Zhenxuan Zhang; Yuanbo Zhou; Xinlin Zhang; Yuanbin Chen; Tao Tan; Guang Yang; Tong Tong
>
> **摘要:** The effectiveness of convolutional neural networks in medical image segmentation relies on large-scale, high-quality annotations, which are costly and time-consuming to obtain. Even expert-labeled datasets inevitably contain noise arising from subjectivity and coarse delineations, which disrupt feature learning and adversely impact model performance. To address these challenges, this study propose a Geometric-Structural Dual-Guided Network (GSD-Net), which integrates geometric and structural cues to improve robustness against noisy annotations. It incorporates a Geometric Distance-Aware module that dynamically adjusts pixel-level weights using geometric features, thereby strengthening supervision in reliable regions while suppressing noise. A Structure-Guided Label Refinement module further refines labels with structural priors, and a Knowledge Transfer module enriches supervision and improves sensitivity to local details. To comprehensively assess its effectiveness, we evaluated GSD-Net on six publicly available datasets: four containing three types of simulated label noise, and two with multi-expert annotations that reflect real-world subjectivity and labeling inconsistencies. Experimental results demonstrate that GSD-Net achieves state-of-the-art performance under noisy annotations, achieving improvements of 1.58% on Kvasir, 22.76% on Shenzhen, 8.87% on BU-SUC, and 1.77% on BraTS2020 under SR simulated noise. The codes of this study are available at this https URL.
>
---
#### [replaced 054] DifAttack++: Query-Efficient Black-Box Adversarial Attack via Hierarchical Disentangled Feature Space in Cross-Domain
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2406.03017](https://arxiv.org/pdf/2406.03017)**

> **作者:** Jun Liu; Jiantao Zhou; Jiandian Zeng; Jinyu Tian; Isao Echizen
>
> **备注:** 13 pages
>
> **摘要:** This work investigates efficient score-based black-box adversarial attacks that achieve a high Attack Success Rate (ASR) and good generalization ability. We propose a novel attack framework, termed DifAttack++, which operates in a hierarchical disentangled feature space and significantly differs from existing methods that manipulate the entire feature space. Specifically, DifAttack++ firstly disentangles an image's latent representation into an Adversarial Feature (AF) and a Visual Feature (VF) using an autoencoder equipped with a carefully designed Hierarchical Decouple-Fusion (HDF) module. In this formulation, the AF primarily governs the adversarial capability of an image, while the VF largely preserves its visual appearance. To enable the feature disentanglement and image reconstruction, we jointly train two autoencoders for the clean and adversarial image domains, i.e., cross-domain, respectively, using paired clean images and their corresponding Adversarial Examples (AEs) generated by white-box attacks on available surrogate models. During the black-box attack stage, DifAttack++ iteratively optimizes the AF based on query feedback from the victim model, while keeping the VF fixed, until a successful AE is obtained. Extensive experimental results demonstrate that DifAttack++ achieves superior ASR and query efficiency compared to state-of-the-art methods, while producing AEs with comparable visual quality. Our code is available at this https URL.
>
---
#### [replaced 055] Cerebra: A Multidisciplinary AI Board for Multimodal Dementia Characterization and Risk Assessment
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.21597](https://arxiv.org/pdf/2603.21597)**

> **作者:** Sheng Liu; Long Chen; Zeyun Zhao; Qinglin Gou; Qingyue Wei; Arjun Masurkar; Kevin M. Spiegler; Philip Kuball; Stefania C. Bray; Megan Bernath; Deanna R. Willis; Jiang Bian; Lei Xing; Eric Topol; Kyunghyun Cho; Yu Huang; Ruogu Fang; Narges Razavian; James Zou
>
> **摘要:** Modern clinical practice increasingly depends on reasoning over heterogeneous, evolving, and incomplete patient data. Although recent advances in multimodal foundation models have improved performance on various clinical tasks, most existing models remain static, opaque, and poorly aligned with real-world clinical workflows. We present Cerebra, an interactive multi-agent AI team that coordinates specialized agents for EHR, clinical notes, and medical imaging analysis. These outputs are synthesized into a clinician-facing dashboard that combines visual analytics with a conversational interface, enabling clinicians to interrogate predictions and contextualize risk at the point of care. Cerebra supports privacy-preserving deployment by operating on structured representations and remains robust when modalities are incomplete. We evaluated Cerebra using a massive multi-institutional dataset spanning 3 million patients from four independent healthcare systems. Cerebra consistently outperformed both state-of-the-art single-modality models and large multimodal language model baselines. In dementia risk prediction, it achieved AUROCs up to 0.80, compared with 0.74 for the strongest single-modality model and 0.68 for language model baselines. For dementia diagnosis, it achieved an AUROC of 0.86, and for survival prediction, a C-index of 0.81. In a reader study with experienced physicians, Cerebra significantly improved expert performance, increasing accuracy by 17.5 percentage points in prospective dementia risk estimation. These results demonstrate Cerebra's potential for interpretable, robust decision support in clinical care.
>
---
#### [replaced 056] Momentum Memory for Knowledge Distillation in Computational Pathology
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.21395](https://arxiv.org/pdf/2602.21395)**

> **作者:** Yongxin Guo; Hao Lu; Onur C. Koyun; Zhengjie Zhu; Muhammet Fatih Demir; Metin Nafi Gurcan
>
> **备注:** Accepted by CVPR 2026. Code: this https URL
>
> **摘要:** Multimodal learning that integrates genomics and histopathology has shown strong potential in cancer diagnosis, yet its clinical translation is hindered by the limited availability of paired histology-genomics data. Knowledge distillation (KD) offers a practical solution by transferring genomic supervision into histopathology models, enabling accurate inference using histology alone. However, existing KD methods rely on batch-local alignment, which introduces instability due to limited within-batch comparisons and ultimately degrades performance. To address these limitations, we propose Momentum Memory Knowledge Distillation (MoMKD), a cross-modal distillation framework driven by a momentum-updated memory. This memory aggregates genomic and histopathology information across batches, effectively enlarging the supervisory context available to each mini-batch. Furthermore, we decouple the gradients of the genomics and histology branches, preventing genomic signals from dominating histology feature learning during training and eliminating the modality-gap issue at inference time. Extensive experiments on the TCGA-BRCA benchmark (HER2, PR, and ODX classification tasks) and an independent in-house testing dataset demonstrate that MoMKD consistently outperforms state-of-the-art MIL and multimodal KD baselines, delivering strong performance and generalization under histology-only inference. Overall, MoMKD establishes a robust and generalizable knowledge distillation paradigm for computational pathology.
>
---
#### [replaced 057] HalDec-Bench: Benchmarking Hallucination Detector in Image Captioning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20515](https://arxiv.org/pdf/2511.20515)**

> **作者:** Kuniaki Saito; Risa Shinoda; Shohei Tanaka; Tosho Hirasawa; Fumio Okura; Yoshitaka Ushiku
>
> **备注:** Previously this version appeared as arXiv:2603.15253 which was submitted as a new work by accident
>
> **摘要:** Hallucination detection in captions (HalDec) assesses a vision-language model's ability to correctly align image content with text by identifying errors in captions that misrepresent the image. Beyond evaluation, effective hallucination detection is also essential for curating high-quality image-caption pairs used to train VLMs. However, the generalizability of VLMs as hallucination detectors across different captioning models and hallucination types remains unclear due to the lack of a comprehensive benchmark. In this work, we introduce HalDec-Bench, a benchmark designed to evaluate hallucination detectors in a principled and interpretable manner. HalDec-Bench contains captions generated by diverse VLMs together with human annotations indicating the presence of hallucinations, detailed hallucination-type categories, and segment-level labels. The benchmark provides tasks with a wide range of difficulty levels and reveals performance differences across models that are not visible in existing multimodal reasoning or alignment benchmarks. Our analysis further uncovers two key findings. First, detectors tend to recognize sentences appearing at the beginning of a response as correct, regardless of their actual correctness. Second, our experiments suggest that dataset noise can be substantially reduced by using strong VLMs as filters while employing recent VLMs as caption generators. Our project page is available at this https URL.
>
---
#### [replaced 058] Replay-Free Continual Low-Rank Adaptation with Dynamic Memory
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2411.00623](https://arxiv.org/pdf/2411.00623)**

> **作者:** Huancheng Chen; Jingtao Li; Weiming Zhuang; Chen Chen; Lingjuan Lyu
>
> **摘要:** We revisit continual learning~(CL), which enables pre-trained vision transformers (ViTs) to sequentially fine-tune on new downstream tasks over time. However, as the scale of these models increases, catastrophic forgetting remains a more serious challenge. Recent studies highlight a crossover between CL techniques and parameter-efficient fine-tuning (PEFT), which focuses on fine-tuning only a small set of trainable parameters to adapt to downstream tasks, such as low-rank adaptation (LoRA). While LoRA achieves faster convergence and requires fewer trainable parameters, it has seldom been explored in the context of continual learning. To address this gap, we propose a novel PEFT-CL method called Dual Low-Rank Adaptation (DualLoRA), which introduces both an orthogonal LoRA adapter and a residual LoRA adapter parallel to pre-trained weights in each layer. These components are orchestrated by a dynamic memory mechanism to strike a balance between stability and plasticity. Additionally, we propose a scheme to predict task identity with confidence and calibrate the model's outputs accordingly. On ViT-based models, we demonstrate that DualLoRA offers significant advantages in accuracy, inference speed, and computation efficiency in training over existing CL methods across multiple benchmarks.
>
---
#### [replaced 059] Towards a general-purpose foundation model for fMRI analysis
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.11167](https://arxiv.org/pdf/2506.11167)**

> **作者:** Cheng Wang; Yu Jiang; Zhihao Peng; Chenxin Li; Changbae Bang; Lin Zhao; Wanyi Fu; Jinglei Lv; Jorge Sepulcre; Carl Yang; Lifang He; Tianming Liu; Xue-Jun Kong; Quanzheng Li; Daniel S. Barron; Anqi Qiu; Randy Hirschtick; Byung-Hoon Kim; Hongbin Han; Xiang Li; Yixuan Yuan
>
> **摘要:** Functional MRI (fMRI) is crucial for studying brain function and diagnosing neurological disorders. However, existing analysis methods suffer from reproducibility and transferability challenges due to complex preprocessing pipelines and task-specific model designs. In this work, we introduce NeuroSTORM (Neuroimaging Foundation Model with Spatial-Temporal Optimized Representation Modeling) that learns generalizable representations directly from 4D fMRI volumes and enables efficient transfer to diverse downstream applications. Specifically, NeuroSTORM is pre-trained on 28.65 million fMRI frames from over 50,000 subjects, spanning multiple centers and ages 5 to 100. It combines an efficient spatiotemporal modeling design and lightweight task adaptation to enable scalable pre-training and fast transfer to downstream applications. Here we show that NeuroSTORM consistently outperforms existing methods across five downstream tasks, including demographic prediction, phenotype prediction, disease diagnosis, re-identification, and state classification. On two multi-hospital clinical cohorts with 17 diagnoses, NeuroSTORM achieves the best diagnosis performance while remaining predictive of psychological and cognitive phenotypes. These results suggest that NeuroSTORM could become a standardized foundation model for reproducible and transferable fMRI analysis.
>
---
#### [replaced 060] BeltCrack: the First Sequential-image Industrial Conveyor Belt Crack Detection Dataset and Its Baseline with Triple-domain Feature Learning
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.17892](https://arxiv.org/pdf/2506.17892)**

> **作者:** Jianghong Huang; Luping Ji; Xin Ma; Mao Ye
>
> **备注:** Accepted by Pattern Recognition
>
> **摘要:** Conveyor belts are important equipment in modern industry, widely applied in production and manufacturing. Their health is much critical to operational efficiency and safety. Cracks are a major threat to belt health. Currently, considering safety, how to intelligently detect belt cracks is catching an increasing attention. To implement the intelligent detection with machine learning, real crack samples are believed to be necessary. However, existing crack datasets primarily focus on pavement scenarios or synthetic data, no real-world industrial belt crack datasets at all. Cracks are a major threat to belt health. Furthermore, to validate usability and effectiveness, we propose a special baseline method with triple-domain ($i.e.$, time-space-frequency) feature hierarchical fusion learning for the two whole-new datasets. Experimental results demonstrate the availability and effectiveness of our dataset. Besides, they also show that our baseline is obviously superior to other similar detection methods. Our datasets and source codes are available at this https URL.
>
---
#### [replaced 061] Pedestrian Crossing Intention Prediction Using Multimodal Fusion Network
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.20008](https://arxiv.org/pdf/2511.20008)**

> **作者:** Yuanzhe Li; Steffen Müller
>
> **备注:** 29th IAVSD International Symposium on Dynamics of Vehicles on Roads and Tracks (IAVSD 2025)
>
> **摘要:** Pedestrian crossing intention prediction is essential for the deployment of autonomous vehicles (AVs) in urban environments. Ideal prediction provides AVs with critical environmental cues, thereby reducing the risk of pedestrian-related collisions. However, the prediction task is challenging due to the diverse nature of pedestrian behavior and its dependence on multiple contextual factors. This paper proposes a multimodal fusion network that leverages seven modality features from both visual and motion branches, aiming to effectively extract and integrate complementary cues across different modalities. Specifically, motion and visual features are extracted from the raw inputs using multiple Transformer-based extraction modules. Depth-guided attention module leverages depth information to guide attention towards salient regions in another modality through comprehensive spatial feature interactions. To account for the varying importance of different modalities and frames, modality attention and temporal attention are designed to selectively emphasize informative modalities and effectively capture temporal dependencies. Extensive experiments on the JAAD dataset validate the effectiveness of the proposed network, achieving superior performance compared to the baseline methods.
>
---
#### [replaced 062] Do Modern Video-LLMs Need to Listen? A Benchmark Audit and Scalable Remedy
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于视频理解任务，旨在解决现有基准未充分评估音频作用的问题。通过引入语音编码器，验证音频在跨模态任务中的重要性。**

- **链接: [https://arxiv.org/pdf/2509.17901](https://arxiv.org/pdf/2509.17901)**

> **作者:** Geewook Kim; Minjoon Seo
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Speech and audio encoders developed over years of community effort are routinely excluded from video understanding pipelines -- not because they fail, but because benchmarks never required listening. We audit 10 video benchmarks and find items largely solvable from visual cues alone: a single-frame probe answers ~76% of AVQA without audio, suggesting poor measurement of audio-visual reasoning. Building on LLaVA-OneVision, we attach a speech/audio encoder and compare five compressor architectures under 25x token reduction (25 Hz to 1 Hz). Across 10 benchmarks -- with and without filtering -- audio yields clear gains on tasks requiring speech comprehension or cross-modal grounding, while vision-centric suites remain largely unaffected. Our results show that speech encoders play a larger role in video understanding than current benchmarks suggest. We will fully open-source our work at this https URL.
>
---
#### [replaced 063] PiLoT: Neural Pixel-to-3D Registration for UAV-based Ego and Target Geo-localization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.20778](https://arxiv.org/pdf/2603.20778)**

> **作者:** Xiaoya Cheng; Long Wang; Yan Liu; Xinyi Liu; Hanlin Tan; Yu Liu; Maojun Zhang; Shen Yan
>
> **摘要:** We present PiLoT, a unified framework that tackles UAV-based ego and target geo-localization. Conventional approaches rely on decoupled pipelines that fuse GNSS and Visual-Inertial Odometry (VIO) for ego-pose estimation, and active sensors like laser rangefinders for target localization. However, these methods are susceptible to failure in GNSS-denied environments and incur substantial hardware costs and complexity. PiLoT breaks this paradigm by directly registering live video stream against a geo-referenced 3D map. To achieve robust, accurate, and real-time performance, we introduce three key contributions: 1) a Dual-Thread Engine that decouples map rendering from core localization thread, ensuring both low latency while maintaining drift-free accuracy; 2) a large-scale synthetic dataset with precise geometric annotations (camera pose, depth maps). This dataset enables the training of a lightweight network that generalizes in a zero-shot manner from simulation to real data; and 3) a Joint Neural-Guided Stochastic-Gradient Optimizer (JNGO) that achieves robust convergence even under aggressive motion. Evaluations on a comprehensive set of public and newly collected benchmarks show that PiLoT outperforms state-of-the-art methods while running over 25 FPS on NVIDIA Jetson Orin platform. Our code and dataset is available at: this https URL.
>
---
#### [replaced 064] Think Before You Drive: World Model-Inspired Multimodal Grounding for Autonomous Vehicles
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.03454](https://arxiv.org/pdf/2512.03454)**

> **作者:** Haicheng Liao; Huanming Shen; Bonan Wang; Yongkang Li; Yihong Tang; Chengyue Wang; Dingyi Zhuang; Kehua Chen; Hai Yang; Chengzhong Xu; Zhenning Li
>
> **摘要:** Interpreting natural-language commands to localize target objects is critical for autonomous driving (AD). Existing visual grounding (VG) methods for autonomous vehicles (AVs) typically struggle with ambiguous, context-dependent instructions, as they lack reasoning over 3D spatial relations and anticipated scene evolution. Grounded in the principles of world models, we propose ThinkDeeper, a framework that reasons about future spatial states before making grounding decisions. At its core is a Spatial-Aware World Model (SA-WM) that learns to reason ahead by distilling the current scene into a command-aware latent state and rolling out a sequence of future latent states, providing forward-looking cues for disambiguation. Complementing this, a hypergraph-guided decoder then hierarchically fuses these states with the multimodal input, capturing higher-order spatial dependencies for robust localization. In addition, we present DrivePilot, a multi-source VG dataset in AD, featuring semantic annotations generated by a Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT)-prompted LLM pipeline. Extensive evaluations on six benchmarks, ThinkDeeper ranks #1 on the Talk2Car leaderboard and surpasses state-of-the-art baselines on DrivePilot, MoCAD, and RefCOCO/+/g benchmarks. Notably, it shows strong robustness and efficiency in challenging scenes (long-text, multi-agent, ambiguity) and retains superior performance even when trained on 50% of the data.
>
---
#### [replaced 065] ViDiC: Video Difference Captioning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.03405](https://arxiv.org/pdf/2512.03405)**

> **作者:** Jiangtao Wu; Shihao Li; Zhaozhou Bian; Jialu Chen; Runzhe Wen; An Ping; Yiwen He; Jiakai Wang; Yuanxing Zhang; Jiaheng Liu
>
> **摘要:** Understanding visual differences between dynamic scenes requires the comparative perception of compositional, spatial, and temporal changes--a capability that remains underexplored in existing vision-language systems. While prior work on Image Difference Captioning (IDC) has enabled models to describe semantic changes between static images, these approaches fail to capture motion continuity, event evolution, or editing consistency over time. We introduce the ViDiC (Video Difference Captioning) task and its corresponding ViDiC-1K dataset, designed to evaluate the ability of Multimodal Large Language Models (MLLMs) to provide fine-grained descriptions of similarities and differences between video pairs. ViDiC-1K comprises 1,000 curated video pairs annotated with over 4,000 comparative checklist items, covering seven categories: subject, style, background, cinematography, motion, location, and playback techniques. To ensure reliable evaluation, we propose a dual-checklist framework that measures the accuracy of similarity and difference separately, based on the LLM-as-a-Judge protocol. Experiments on nineteen representative multimodal models reveal a significant performance gap in their comparative description and difference perception abilities. We hope ViDiC-1K can be a challenging benchmark that lays a solid foundation for advancing video understanding, edit awareness, and comparative reasoning in multimodal intelligence.
>
---
#### [replaced 066] nuScenes Revisited: Progress and Challenges in Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 本文回顾nuScenes数据集，分析其在自动驾驶中的作用与影响，总结其技术细节及对社区的贡献，旨在提供自动驾驶领域的全面综述。**

- **链接: [https://arxiv.org/pdf/2512.02448](https://arxiv.org/pdf/2512.02448)**

> **作者:** Whye Kit Fong; Venice Erin Liong; Kok Seang Tan; Holger Caesar
>
> **备注:** 18 pages, 17 figures
>
> **摘要:** Autonomous Vehicles (AV) and Advanced Driver Assistance Systems (ADAS) have been revolutionized by Deep Learning. As a data-driven approach, Deep Learning relies on vast amounts of driving data, typically labeled in great detail. As a result, datasets, alongside hardware and algorithms, are foundational building blocks for the development of AVs. In this work we revisit one of the most widely used autonomous driving datasets: the nuScenes dataset. nuScenes exemplifies key trends in AV development, being the first dataset to include radar data, to feature diverse urban driving scenes from two continents, and to be collected using a fully autonomous vehicle operating on public roads, while also promoting multi-modal sensor fusion, standardized benchmarks, and a broad range of tasks including perception, localization & mapping, prediction and planning. We provide an unprecedented look into the creation of nuScenes, as well as its extensions nuImages and Panoptic nuScenes, summarizing many technical details that have hitherto not been revealed in academic publications. Furthermore, we trace how the influence of nuScenes impacted a large number of other datasets that were released later and how it defined numerous standards that are used by the community to this day. Finally, we present an overview of both official and unofficial tasks using the nuScenes dataset and review major methodological developments, thereby offering a comprehensive survey of the autonomous driving literature, with a particular focus on nuScenes.
>
---
#### [replaced 067] SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.18993](https://arxiv.org/pdf/2602.18993)**

> **作者:** Jiwoo Chung; Sangeek Hyun; MinKyu Lee; Byeongju Han; Geonho Cha; Dongyoon Wee; Youngjun Hong; Jae-Pil Heo
>
> **备注:** Accepted to CVPR 2026. Project page:this https URL
>
> **摘要:** Diffusion models are a strong backbone for visual generation, but their inherently sequential denoising process leads to slow inference. Previous methods accelerate sampling by caching and reusing intermediate outputs based on feature distances between adjacent timesteps. However, existing caching strategies typically rely on raw feature differences that entangle content and noise. This design overlooks spectral evolution, where low-frequency structure appears early and high-frequency detail is refined later. We introduce Spectral-Evolution-Aware Cache (SeaCache), a training-free cache schedule that bases reuse decisions on a spectrally aligned representation. Through theoretical and empirical analysis, we derive a Spectral-Evolution-Aware (SEA) filter that preserves content-relevant components while suppressing noise. Employing SEA-filtered input features to estimate redundancy leads to dynamic schedules that adapt to content while respecting the spectral priors underlying the diffusion model. Extensive experiments on diverse visual generative models and the baselines show that SeaCache achieves state-of-the-art latency-quality trade-offs.
>
---
#### [replaced 068] Masking Matters: Unlocking the Spatial Reasoning Capabilities of LLMs for 3D Scene-Language Understanding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.02487](https://arxiv.org/pdf/2512.02487)**

> **作者:** Yerim Jeon; Miso Lee; WonJun Moon; Jae-Pil Heo
>
> **备注:** Accepted to CVPR 2026. GitHub Page: this https URL
>
> **摘要:** Recent advances in 3D scene-language understanding have leveraged Large Language Models (LLMs) for 3D reasoning by transferring their general reasoning ability to 3D multi-modal contexts. However, existing methods typically adopt standard decoders from language modeling, which rely on a causal attention mask. This design introduces two fundamental conflicts in 3D scene understanding: sequential bias among order-agnostic 3D objects and restricted object-instruction attention, hindering task-specific reasoning. To overcome these limitations, we propose 3D Spatial Language Instruction Mask (3D-SLIM), an effective masking strategy that replaces the causal mask with an adaptive attention mask tailored to the spatial structure of 3D scenes. Our 3D-SLIM introduces two key components: a Geometry-adaptive Mask that constrains attention based on spatial density rather than token order, and an Instruction-aware Mask that enables object tokens to directly access instruction context. This design allows the model to process objects based on their spatial relationships while being guided by the user's task. 3D-SLIM is simple, requires no architectural modifications, and adds no extra parameters, yet it yields substantial performance improvements across diverse 3D scene-language tasks. Extensive experiments across multiple benchmarks and LLM baselines validate its effectiveness and underscore the critical role of decoder design in 3D multi-modal reasoning.
>
---
#### [replaced 069] Redefining non-IID Data in Federated Learning for Computer Vision Tasks: Migrating from Labels to Embeddings for Task-Specific Data Distributions
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.14553](https://arxiv.org/pdf/2503.14553)**

> **作者:** Kasra Borazjani; Payam Abdisarabshali; Naji Khosravan; Seyyedali Hosseinalipour
>
> **备注:** Accepted for publication in IEEE Transactions on Artificial Intelligence, 2026
>
> **摘要:** Federated Learning (FL) has emerged as one of the prominent paradigms for distributed machine learning (ML). However, it is well-established that its performance can degrade significantly under non-IID (non-independent and identically distributed) data distributions across clients. To study this effect, the existing works predominantly emulate data heterogeneity by imposing label distribution skew across clients. In this paper, we show that label distribution skew fails to fully capture the data heterogeneity in computer vision tasks beyond classification, exposing an overlooked gap in the literature. Motivated by this, by utilizing pre-trained deep neural networks to extract task-specific data embeddings, we define task-specific data heterogeneity through the lens of each vision task and introduce a new level of data heterogeneity called embedding-based data heterogeneity. Our methodology involves clustering data points based on embeddings and distributing them among clients using the Dirichlet distribution. Through extensive experiments, we evaluate the performance of different FL methods under our revamped notion of data heterogeneity, introducing new benchmark performance measures to the literature. For instance, across seven representative computer vision tasks, our embedding-based heterogeneity formulation leads to up to around 60% increase in the observed loss under FedAvg, indicating that it more accurately exposes the performance degradation caused by data heterogeneity. We further unveil a series of open research directions that can be pursued. (Code: this https URL)
>
---
#### [replaced 070] Follow-Your-Motion: Video Motion Transfer via Efficient Spatial-Temporal Decoupled Finetuning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.05207](https://arxiv.org/pdf/2506.05207)**

> **作者:** Yue Ma; Yulong Liu; Qiyuan Zhu; Ayden Yang; Kunyu Feng; Xinhua Zhang; Zexuan Yan; Zhifeng Li; Sirui Han; Chenyang Qi; Qifeng Chen
>
> **备注:** Accepted by ICLR 2026, project page: this https URL
>
> **摘要:** Recently, breakthroughs in the video diffusion transformer have shown remarkable capabilities in diverse motion generations. As for the motion-transfer task, current methods mainly use two-stage Low-Rank Adaptations (LoRAs) finetuning to obtain better performance. However, existing adaptation-based motion transfer still suffers from motion inconsistency and tuning inefficiency when applied to large video diffusion transformers. Naive two-stage LoRA tuning struggles to maintain motion consistency between generated and input videos due to the inherent spatial-temporal coupling in the 3D attention operator. Additionally, they require time-consuming fine-tuning processes in both stages. To tackle these issues, we propose Follow-Your-Motion, an efficient two-stage video motion transfer framework that finetunes a powerful video diffusion transformer to synthesize complex motion. Specifically, we propose a spatial-temporal decoupled LoRA to decouple the attention architecture for spatial appearance and temporal motion processing. During the second training stage, we design the sparse motion sampling and adaptive RoPE to accelerate the tuning speed. To address the lack of a benchmark for this field, we introduce MotionBench, a comprehensive benchmark comprising diverse motion, including creative camera motion, single object motion, multiple object motion, and complex human motion. We show extensive evaluations on MotionBench to verify the superiority of Follow-Your-Motion.
>
---
#### [replaced 071] FastVMT: Eliminating Redundancy in Video Motion Transfer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.05551](https://arxiv.org/pdf/2602.05551)**

> **作者:** Yue Ma; Zhikai Wang; Tianhao Ren; Mingzhe Zheng; Hongyu Liu; Jiayi Guo; Kunyu Feng; Yuxuan Xue; Zixiang Zhao; Konrad Schindler; Qifeng Chen; Linfeng Zhang
>
> **备注:** Accepted by ICLR2026, Project page: this http URL, Code: this https URL
>
> **摘要:** Video motion transfer aims to synthesize videos by generating visual content according to a text prompt while transferring the motion pattern observed in a reference video. Recent methods predominantly use the Diffusion Transformer (DiT) architecture. To achieve satisfactory runtime, several methods attempt to accelerate the computations in the DiT, but fail to address structural sources of inefficiency. In this work, we identify and remove two types of computational redundancy in earlier work: motion redundancy arises because the generic DiT architecture does not reflect the fact that frame-to-frame motion is small and smooth; gradient redundancy occurs if one ignores that gradients change slowly along the diffusion trajectory. To mitigate motion redundancy, we mask the corresponding attention layers to a local neighborhood such that interaction weights are not computed unnecessarily distant image regions. To exploit gradient redundancy, we design an optimization scheme that reuses gradients from previous diffusion steps and skips unwarranted gradient computations. On average, FastVMT achieves a 3.43x speedup without degrading the visual fidelity or the temporal consistency of the generated videos.
>
---
#### [replaced 072] Captain Safari: A World Engine with Pose-Aligned 3D Memory
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22815](https://arxiv.org/pdf/2511.22815)**

> **作者:** Yu-Cheng Chou; Xingrui Wang; Yitong Li; Jiahao Wang; Hanting Liu; Cihang Xie; Alan Yuille; Junfei Xiao
>
> **摘要:** World engines aim to synthesize long, 3D-consistent videos that support interactive exploration of a scene under user-controlled camera motion. However, existing systems struggle under aggressive 6-DoF trajectories and complex outdoor layouts: they lose long-range geometric coherence, deviate from the target path, or collapse into overly conservative motion. To this end, we introduce Captain Safari, a pose-conditioned world engine that generates videos by retrieving from a persistent world memory. Given a camera path, our method maintains a dynamic local memory and uses a retriever to fetch pose-aligned world tokens, which then condition video generation along the trajectory. This design enables the model to maintain stable 3D structure while accurately executing challenging camera maneuvers. To evaluate this setting, we curate OpenSafari, a new in-the-wild FPV dataset containing high-dynamic drone videos with verified camera trajectories, constructed through a multi-stage geometric and kinematic validation pipeline. Across video quality, 3D consistency, and trajectory following, Captain Safari substantially outperforms state-of-the-art camera-controlled generators. It reduces MEt3R from 0.3703 to 0.3690, improves AUC@30 from 0.181 to 0.200, and yields substantially lower FVD than all camera-controlled baselines. More importantly, in a 50-participant, 5-way human study where annotators select the best result among five anonymized models, 67.6% of preferences favor our method across all axes. Our results demonstrate that pose-conditioned world memory is a powerful mechanism for long-horizon, controllable video generation and provide OpenSafari as a challenging new benchmark for future world-engine research.
>
---
#### [replaced 073] WISER: Wider Search, Deeper Thinking, and Adaptive Fusion for Training-Free Zero-Shot Composed Image Retrieval
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.23029](https://arxiv.org/pdf/2602.23029)**

> **作者:** Tianyue Wang; Leigang Qu; Tianyu Yang; Xiangzhao Hao; Yifan Xu; Haiyun Guo; Jinqiao Wang
>
> **备注:** Accept to CVPR 2026
>
> **摘要:** Zero-Shot Composed Image Retrieval (ZS-CIR) aims to retrieve target images given a multimodal query (comprising a reference image and a modification text), without training on annotated triplets. Existing methods typically convert the multimodal query into a single modality-either as an edited caption for Text-to-Image retrieval (T2I) or as an edited image for Image-to-Image retrieval (I2I). However, each paradigm has inherent limitations: T2I often loses fine-grained visual details, while I2I struggles with complex semantic modifications. To effectively leverage their complementary strengths under diverse query intents, we propose WISER, a training-free framework that unifies T2I and I2I via a "retrieve-verify-refine" pipeline, explicitly modeling intent awareness and uncertainty awareness. Specifically, WISER first performs Wider Search by generating both edited captions and images for parallel retrieval to broaden the candidate pool. Then, it conducts Adaptive Fusion with a verifier to assess retrieval confidence, triggering refinement for uncertain retrievals, and dynamically fusing the dual-path for reliable ones. For uncertain retrievals, WISER generates refinement suggestions through structured self-reflection to guide the next retrieval round toward Deeper Thinking. Extensive experiments demonstrate that WISER significantly outperforms previous methods across multiple benchmarks, achieving relative improvements of 45% on CIRCO (mAP@5) and 57% on CIRR (Recall@1) over existing training-free methods. Notably, it even surpasses many training-dependent methods, highlighting its superiority and generalization under diverse scenarios. Code will be released at this https URL.
>
---
#### [replaced 074] Human Presence Detection via Wi-Fi Range-Filtered Doppler Spectrum on Commodity Laptops
- **分类: eess.SP; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.10845](https://arxiv.org/pdf/2603.10845)**

> **作者:** Jessica Sanson; Rahul C. Shah; Valerio Frascolla
>
> **备注:** 6 pages, Conference
>
> **摘要:** Human Presence Detection (HPD) is key to enable intelligent power management and security features in everyday devices. In this paper we propose the first HPD solution that leverages monostatic Wi-Fi sensing and detects user position using only the built-in Wi-Fi hardware of a device, with no need for external devices, access points, or additional sensors. In contrast, existing HPD solutions for laptops require external dedicated sensors which add cost and complexity, or rely on camera-based approaches that introduce significant privacy concerns. We herewith introduce the Range-Filtered Doppler Spectrum (RF-DS), a novel Wi-Fi sensing technique for presence estimation that enables both range-selective and temporally windowed detection of user presence. By applying targeted range-area filtering in the Channel Impulse Response (CIR) domain before Doppler analysis, our method focuses processing on task-relevant spatial zones, significantly reducing computational complexity. In addition, the use of temporal windows in the spectrum domain provides greater estimator stability compared to conventional 2D Range-Doppler detectors. Furthermore, we propose an adaptive multi-rate processing framework that dynamically adjusts Channel State Information (CSI) sampling rates-operating at low frame rates (10Hz) during idle periods and high rates (100Hz) only when motion is detected. To our knowledge, this is the first low-complexity solution for occupancy detection using monostatic Wi-Fi sensing on a built-in Wi-Fi network interface controller (NIC) of a commercial off-the-shelf laptop that requires no external network infrastructure or specialized sensors. Our solution can scale across different environments and devices without calibration or retraining.
>
---
#### [replaced 075] SARE: Sample-wise Adaptive Reasoning for Training-free Fine-grained Visual Recognition
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.17729](https://arxiv.org/pdf/2603.17729)**

> **作者:** Jingxiao Yang; DaLin He; Miao Pan; Ge Su; Wenqi Zhang; Yifeng Hu; Tangwei Li; Yuke Li; Xuhong Zhang
>
> **备注:** preprint, under review
>
> **摘要:** Recent advances in Large Vision-Language Models (LVLMs) have enabled training-free Fine-Grained Visual Recognition (FGVR). However, effectively exploiting LVLMs for FGVR remains challenging due to the inherent visual ambiguity of subordinate-level categories. Existing methods predominantly adopt either retrieval-oriented or reasoning-oriented paradigms to tackle this challenge, but both are constrained by two fundamental limitations:(1) They apply the same inference pipeline to all samples without accounting for uneven recognition difficulty, thereby leading to suboptimal accuracy and efficiency; (2) The lack of mechanisms to consolidate and reuse error-specific experience causes repeated failures on similar challenging cases. To address these limitations, we propose SARE, a Sample-wise Adaptive textbfREasoning framework for training-free FGVR. Specifically, SARE adopts a cascaded design that combines fast candidate retrieval with fine-grained reasoning, invoking the latter only when necessary. In the reasoning process, SARE incorporates a self-reflective experience mechanism that leverages past failures to provide transferable discriminative guidance during inference, without any parameter updates. Extensive experiments across 14 datasets substantiate that SARE achieves state-of-the-art performance while substantially reducing computational overhead.
>
---
#### [replaced 076] Inverting Neural Networks: New Methods to Generate Neural Network Inputs from Prescribed Outputs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.20461](https://arxiv.org/pdf/2603.20461)**

> **作者:** Rebecca Pattichis; Sebastian Janampa; Constantinos S. Pattichis; Marios S. Pattichis
>
> **备注:** Accepted at 2026 IEEE Southwest Symposium on Image Analysis and Interpretation (SSIAI)
>
> **摘要:** Neural network systems describe complex mappings that can be very difficult to understand. In this paper, we study the inverse problem of determining the input images that get mapped to specific neural network classes. Ultimately, we expect that these images contain recognizable features that are associated with their corresponding class classifications. We introduce two general methods for solving the inverse problem. In our forward pass method, we develop an inverse method based on a root-finding algorithm and the Jacobian with respect to the input image. In our backward pass method, we iteratively invert each layer, at the top. During the inversion process, we add random vectors sampled from the null-space of each linear layer. We demonstrate our new methods on both transformer architectures and sequential networks based on linear layers. Unlike previous methods, we show that our new methods are able to produce random-like input images that yield near perfect classification scores in all cases, revealing vulnerabilities in the underlying networks. Hence, we conclude that the proposed methods provide a more comprehensive coverage of the input image spaces that solve the inverse mapping problem.
>
---
#### [replaced 077] Latent Diffusion Inversion Requires Understanding the Latent Space
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20592](https://arxiv.org/pdf/2511.20592)**

> **作者:** Mingxing Rao; Bowen Qu; Daniel Moyer
>
> **备注:** 14 pages, 4 figures, 7 tables
>
> **摘要:** The recovery of training data from generative models ("model inversion") has been extensively studied for diffusion models in the data domain as a memorization/overfitting phenomenon. Latent diffusion models (LDMs), which operate on the latent codes from encoder/decoder pairs, have been robust to prior inversion methods. In this work we describe two key findings: (1) the diffusion model exhibits non-uniform memorization across latent codes, tending to overfit samples located in high-distortion regions of the decoder pullback metric; (2) even within a single latent code, memorization contributions are unequal across representation dimensions. Our proposed method to ranks latent dimensions by their contribution to the decoder pullback metric, which in turn identifies dimensions that contribute to memorization. For score-based membership inference, a sub-task of model inversion, we find that removing less-memorizing dimensions improves performance on all tested methods and datasets, with average AUROC gains of 1-4% and substantial increases in TPR@1%FPR (1-32%) across diverse datasets including CIFAR-10, CelebA, ImageNet-1K, Pokémon, MS-COCO, and Flickr. Our results highlight the overlooked influence of the auto-encoder geometry on LDM memorization and provide a new perspective for analyzing privacy risks in diffusion-based generative models.
>
---
#### [replaced 078] When Models Judge Themselves: Unsupervised Self-Evolution for Multimodal Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.21289](https://arxiv.org/pdf/2603.21289)**

> **作者:** Zhengxian Wu; Kai Shi; Chuanrui Zhang; Zirui Liao; Jun Yang; Ni Yang; Qiuying Peng; Luyuan Zhang; Hangrui Xu; Tianhuang Su; Zhenyu Yang; Haonan Lu; Haoqian Wang
>
> **备注:** 21 pages, 7 figures
>
> **摘要:** Recent progress in multimodal large language models has led to strong performance on reasoning tasks, but these improvements largely rely on high-quality annotated data or teacher-model distillation, both of which are costly and difficult to scale. To address this, we propose an unsupervised self-evolution training framework for multimodal reasoning that achieves stable performance improvements without using human-annotated answers or external reward models. For each input, we sample multiple reasoning trajectories and jointly model their within group structure. We use the Actor's self-consistency signal as a training prior, and introduce a bounded Judge based modulation to continuously reweight trajectories of different quality. We further model the modulated scores as a group level distribution and convert absolute scores into relative advantages within each group, enabling more robust policy updates. Trained with Group Relative Policy Optimization (GRPO) on unlabeled data, our method consistently improves reasoning performance and generalization on five mathematical reasoning benchmarks, offering a scalable path toward self-evolving multimodal models. The code are available at this https URL.
>
---
#### [replaced 079] GeoDiT: A Diffusion-based Vision-Language Model for Geospatial Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.02505](https://arxiv.org/pdf/2512.02505)**

> **作者:** Jiaqi Liu; Ronghao Fu; Haoran Liu; Lang Sun; Bo Yang
>
> **摘要:** Autoregressive models are structurally misaligned with the inherently parallel nature of geospatial understanding, forcing a rigid sequential narrative onto scenes and fundamentally hindering the generation of structured and coherent outputs. We challenge this paradigm by reframing geospatial generation as a parallel refinement process, enabling a holistic, coarse-to-fine synthesis that resolves all semantic elements simultaneously. To operationalize this, we introduce GeoDiT, the first diffusion-based vision-language model tailored for the geospatial domain. Extensive experiments demonstrate that GeoDiT establishes a new state-of-the-art on benchmarks requiring structured, object-centric outputs. It achieves significant gains in image captioning, visual grounding, and multi-object detection, precisely the tasks where autoregressive models falter. Our work validates that aligning the generative process with the data's intrinsic structure is key to unlocking superior performance in complex geospatial analysis.
>
---
#### [replaced 080] Test-Time Adaptation via Cache Personalization for Facial Expression Recognition in Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.21309](https://arxiv.org/pdf/2603.21309)**

> **作者:** Masoumeh Sharafi; Muhammad Osama Zeeshan; Soufiane Belharbi; Alessandro Lameiras Koerich; Marco Pedersoli; Eric Granger
>
> **摘要:** Facial expression recognition (FER) in videos requires model personalization to capture the considerable variations across subjects. Vision-language models (VLMs) offer strong transfer to downstream tasks through image-text alignment, but their performance can still degrade under inter-subject distribution shifts. Personalizing models using test-time adaptation (TTA) methods can mitigate this challenge. However, most state-of-the-art TTA methods rely on unsupervised parameter optimization, introducing computational overhead that is impractical in many real-world applications. This paper introduces TTA through Cache Personalization (TTA-CaP), a cache-based TTA method that enables cost-effective (gradient-free) personalization of VLMs for video FER. Prior cache-based TTA methods rely solely on dynamic memories that store test samples, which can accumulate errors and drift due to noisy pseudo-labels. TTA-CaP leverages three coordinated caches: a personalized source cache that stores source-domain prototypes, a positive target cache that accumulates reliable subject-specific samples, and a negative target cache that stores low-confidence cases as negative samples to reduce the impact of noisy pseudo-labels. Cache updates and replacement are controlled by a tri-gate mechanism based on temporal stability, confidence, and consistency with the personalized cache. Finally, TTA-CaP refines predictions through fusion of embeddings, yielding refined representations that support temporally stable video-level predictions. Our experiments on three challenging video FER datasets, BioVid, StressID, and BAH, indicate that TTA-CaP can outperform state-of-the-art TTA methods under subject-specific and environmental shifts, while maintaining low computational and memory overhead for real-world deployment.
>
---
#### [replaced 081] Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.01047](https://arxiv.org/pdf/2602.01047)**

> **作者:** Xinrong Chen; Xu Chu; Yingmin Qiu; Hengyuan Zhang; Jing Xiong; Shiyu Tang; Shuai Liu; Shaokang Yang; Cheng Yang; Hayden Kwok-Hay So; Ngai Wong
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Large Vision-Language Models (LVLMs) can reason from image-text inputs and perform well in various multimodal tasks. Despite this success, they are affected by language priors and often produce hallucinations. Hallucinations denote generated content that is grammatically and syntactically coherent, yet bears no match or direct relevance to visual input. To address this problem, we propose Residual Decoding (ResDec). It is a novel training-free method that uses historical information to aid decoding. The method relies on the internal implicit reasoning mechanism and token logits evolution mechanism of LVLMs to correct biases. Extensive experiments demonstrate that ResDec effectively suppresses hallucinations induced by language priors, significantly improves visual grounding, and reduces object hallucinations. In addition to mitigating hallucinations, ResDec also performs exceptionally well on comprehensive LVLM benchmarks, highlighting its broad applicability.
>
---
#### [replaced 082] Uncertainty-guided Compositional Alignment with Part-to-Whole Semantic Representativeness in Hyperbolic Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.22042](https://arxiv.org/pdf/2603.22042)**

> **作者:** Hayeon Kim; Ji Ha Jang; Junghun James Kim; Se Young Chun
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** While Vision-Language Models (VLMs) have achieved remarkable performance, their Euclidean embeddings remain limited in capturing hierarchical relationships such as part-to-whole or parent-child structures, and often face challenges in multi-object compositional scenarios. Hyperbolic VLMs mitigate this issue by better preserving hierarchical structures and modeling part-whole relations (i.e., whole scene and its part images) through entailment. However, existing approaches do not model that each part has a different level of semantic representativeness to the whole. We propose UNcertainty-guided Compositional Hyperbolic Alignment (UNCHA) for enhancing hyperbolic VLMs. UNCHA models part-to-whole semantic representativeness with hyperbolic uncertainty, by assigning lower uncertainty to more representative parts and higher uncertainty to less representative ones for the whole scene. This representativeness is then incorporated into the contrastive objective with uncertainty-guided weights. Finally, the uncertainty is further calibrated with an entailment loss regularized by entropy-based term. With the proposed losses, UNCHA learns hyperbolic embeddings with more accurate part-whole ordering, capturing the underlying compositional structure in an image and improving its understanding of complex multi-object scenes. UNCHA achieves state-of-the-art performance on zero-shot classification, retrieval, and multi-label classification benchmarks. Our code and models are available at: this https URL.
>
---
#### [replaced 083] From Coarse to Continuous: Progressive Refinement Implicit Neural Representation for Motion-Robust Anisotropic MRI Reconstruction
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.16210](https://arxiv.org/pdf/2506.16210)**

> **作者:** Zhenxuan Zhang; Lipei Zhang; Yanqi Cheng; Zi Wang; Fanwen Wang; Haosen Zhang; Yue Yang; Yinzhe Wu; Jiahao Huang; Angelica I Aviles-Rivero; Zhifan Gao; Guang Yang; Peter J. Lally
>
> **摘要:** In motion-robust magnetic resonance imaging (MRI), slice-to-volume reconstruction is critical for recovering anatomically consistent 3D brain volumes from 2D slices, especially under accelerated acquisitions or patient motion. However, this task remains challenging due to hierarchical structural disruptions. It includes local detail loss from k-space undersampling, global structural aliasing caused by motion, and volumetric anisotropy. Therefore, we propose a progressive refinement implicit neural representation (PR-INR) framework. Our PR-INR unifies motion correction, structural refinement, and volumetric synthesis within a geometry-aware coordinate space. Specifically, a motion-aware diffusion module is first employed to generate coarse volumetric reconstructions that suppress motion artifacts and preserve global anatomical structures. Then, we introduce an implicit detail restoration module that performs residual refinement by aligning spatial coordinates with visual features. It corrects local structures and enhances boundary precision. Further, a voxel continuous-aware representation module represents the image as a continuous function over 3D coordinates. It enables accurate inter-slice completion and high-frequency detail recovery. We evaluate PR-INR on five public MRI datasets under various motion conditions (3% and 5% displacement), undersampling rates (4x and 8x) and slice resolutions (scale = 5). Experimental results demonstrate that PR-INR outperforms state-of-the-art methods in both quantitative reconstruction metrics and visual quality. It further shows generalization and robustness across diverse unseen domains.
>
---
#### [replaced 084] Knee or ROC
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2401.07390](https://arxiv.org/pdf/2401.07390)**

> **作者:** Veronica Wendt; Jacob Steiner; Byunggu Yu; Caleb Kelly; Justin Kim
>
> **备注:** 8 pages
>
> **摘要:** Self-attention transformers have demonstrated accuracy for image classification with smaller data sets. However, a limitation is that tests to-date are based upon single class image detection with known representation of image populations. For instances where the input image classes may be greater than one and test sets that lack full information on representation of image populations, accuracy calculations must adapt. The Receiver Operating Characteristic (ROC) accuracy threshold can address the instances of multiclass input images. However, this approach is unsuitable in instances where image population representation is unknown. We then consider calculating accuracy using the knee method to determine threshold values on an ad-hoc basis. Results of ROC curve and knee thresholds for a multi-class data set, created from CIFAR-10 images, are discussed for multiclass image detection.
>
---
#### [replaced 085] CAD-Prompted SAM3: Geometry-Conditioned Instance Segmentation for Industrial Objects
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.20551](https://arxiv.org/pdf/2602.20551)**

> **作者:** Zhenran Tang; Rohan Nagabhirava; Changliu Liu
>
> **摘要:** Verbal-prompted segmentation is inherently limited by the expressiveness of natural language and struggles with uncommon, instance-specific, or difficult-to-describe objects: scenarios frequently encountered in manufacturing and 3D printing environments. While image exemplars provide an alternative, they primarily encode appearance cues such as color and texture, which are often unrelated to a part's geometric identity. In industrial settings, a single component may be produced in different materials, finishes, or colors, making appearance-based prompting unreliable. In contrast, such objects are typically defined by precise CAD models that capture their canonical geometry. We propose a CAD-prompted segmentation framework built on SAM3 that uses canonical multi-view renderings of a CAD model as prompt input. The rendered views provide geometry-based conditioning independent of surface appearance. The model is trained using synthetic data generated from mesh renderings in simulation under diverse viewpoints and scene contexts. Our approach enables single-stage, CAD-prompted mask prediction, extending promptable segmentation to objects that cannot be robustly described by language or appearance alone.
>
---
#### [replaced 086] Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20032](https://arxiv.org/pdf/2511.20032)**

> **作者:** Jianfei Zhao; Feng Zhang; Xin Sun; Chong Feng; Zhixing Tan
>
> **备注:** CVPR 2026
>
> **摘要:** Visual attention serves as the primary mechanism through which MLLMs interpret visual information; however, its limited localization capability often leads to hallucinations. We observe that although MLLMs can accurately extract visual semantics from visual tokens, they fail to fully leverage this advantage during subsequent inference. To address this limitation, we propose Vision-Guided Attention (VGA), a training-free method that first constructs precise visual grounding by exploiting the semantic content of visual tokens, and then uses this grounding to guide the model's focus toward relevant visual regions. In image captioning, VGA further refines this guidance dynamically during generation by suppressing regions that have already been described. In VGA, each token undergoes only a single forward pass, introducing a negligible latency overhead. In addition, VGA is fully compatible with efficient attention implementations such as FlashAttention. Extensive experiments across diverse MLLMs and multiple hallucination benchmarks demonstrate that VGA achieves state-of-the-art dehallucination performance. Further analysis confirms that explicit visual guidance plays a crucial role in enhancing the visual understanding capabilities of MLLMs.
>
---
#### [replaced 087] ImmerseGen: Agent-Guided Immersive World Generation with Alpha-Textured Proxies
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.14315](https://arxiv.org/pdf/2506.14315)**

> **作者:** Jinyan Yuan; Bangbang Yang; Keke Wang; Panwang Pan; Lin Ma; Xuehai Zhang; Xiao Liu; Zhaopeng Cui; Yuewen Ma
>
> **备注:** Accepted by IEEE VR 2026 and TVCG Special Issue. Project webpage: this https URL
>
> **摘要:** Automating immersive VR scene creation remains a primary research challenge. Existing methods typically rely on complex geometry with post-simplification, resulting in inefficient pipelines or limited realism. In this paper, we introduce ImmerseGen, a novel agent-guided framework for compact and photorealistic world generation that decouples realism from exhaustive geometric modeling. ImmerseGen represents scenes as hierarchical compositions of lightweight geometric proxies with synthesized RGBA textures, facilitating real-time rendering on mobile VR headsets. We propose terrain-conditioned texturing for base world generation, combined with context-aware texturing for scenery, to produce diverse and visually coherent worlds. VLM-based agents employ semantic grid-based analysis for precise asset placement and enrich scenes with multimodal enhancements such as visual dynamics and ambient sound. Experiments and real-time VR applications demonstrate that ImmerseGen achieves superior photorealism, spatial coherence, and rendering efficiency compared to existing methods.
>
---
#### [replaced 088] MS-DGCNN++: Multi-Scale Dynamic Graph Convolution with Scale-Dependent Normalization for Robust LiDAR Tree Species Classification
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.12602](https://arxiv.org/pdf/2507.12602)**

> **作者:** Said Ohamouddou; Hanaa El Afia; Mohamed Hamza Boulaich; Abdellatif El Afia; Raddouane Chiheb
>
> **摘要:** Graph-based deep learning on LiDAR point clouds encodes geometry through edge features, yet standard implementations use the same encoding at every scale. In tree species classification, where point density varies by orders of magnitude between trunk and canopy, this is particularly limiting. We prove it is suboptimal: normalized directional features have mean squared error decaying as $\mathcal{O}(1/s^2)$ with inter-point distance~$s$, while raw displacement error is constant, implying each encoding suits a different signal-to-noise ratio (SNR) regime. We propose MS-DGCNN++, a multi-scale dynamic graph convolutional network with \emph{scale-dependent edge encoding}: raw vectors at the local scale (low SNR) and hybrid raw-plus-normalized vectors at the intermediate scale (high SNR). Five ablations validate this design: encoding ablation confirms $+4$--$6\%$ overall accuracy (OA) gain; density dropout shows the flattest degradation under canopy thinning; a noise sweep locates the theoretical crossover near $\text{SNR}_2 \approx 1.22$; max-pooling provenance reveals far neighbors win $85\%$ of competitions under raw encoding, a bias eliminated by normalization; and isotropy analysis shows normalization nearly doubles effective rank. On STPCTLS (seven species, terrestrial laser scanning), MS-DGCNN++ achieves the highest OA ($92.91\%$) among 56 models, surpassing self-supervised methods with $7$--$24\times$ more parameters using only $1.81$M parameters. On HeliALS (nine species, airborne laser scanning, geometry-only), it achieves $73.66\%$ OA with the best balanced accuracy ($50.28\%$), matching FGI-PointTransformer which uses $4\times$ more points. Robustness analysis across five perturbation types reveals complementary variant strengths for deployment in heterogeneous forest environments. Code: this https URL.
>
---
#### [replaced 089] Point What You Mean: Visually Grounded Instruction Policy
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言-动作任务，解决物体指代不明确的问题。通过引入视觉提示（如边界框）增强语言指令，提升物体定位精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.18933](https://arxiv.org/pdf/2512.18933)**

> **作者:** Hang Yu; Juntu Zhao; Yufeng Liu; Kaiyu Li; Cheng Ma; Di Zhang; Yingdong Hu; Guang Chen; Junyuan Xie; Junliang Guo; Junqiao Zhao; Yang Gao
>
> **摘要:** Vision-Language-Action (VLA) models align vision and language with embodied control, but their object referring ability remains limited when relying solely on text prompt, especially in cluttered or out-of-distribution (OOD) scenes. In this study, we introduce the Point-VLA, a plug-and-play policy that augments language instructions with explicit visual cues (e.g., bounding boxes) to resolve referential ambiguity and enable precise object-level grounding. To efficiently scale visually grounded datasets, we further develop an automatic data annotation pipeline requiring minimal human effort. We evaluate Point-VLA on diverse real-world referring tasks and observe consistently stronger performance than text-only instruction VLAs, particularly in cluttered or unseen-object scenarios, with robust generalization. These results demonstrate that Point-VLA effectively resolves object referring ambiguity through pixel-level visual grounding, achieving more generalizable embodied control.
>
---
#### [replaced 090] Spectral Gaps and Spatial Priors: Studying Hyperspectral Downstream Adaptation Using TerraMind
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.06690](https://arxiv.org/pdf/2603.06690)**

> **作者:** Julia Anna Leonardi; Johannes Jakubik; Paolo Fraccaro; Maria Antonia Brovelli
>
> **备注:** Accepted to ICLR 2026 Machine Learning for Remote Sensing (ML4RS) Workshop
>
> **摘要:** Geospatial Foundation Models (GFMs) typically lack native support for Hyperspectral Imaging (HSI) due to the complexity and sheer size of high-dimensional spectral data. This study investigates the adaptability of TerraMind, a multimodal GFM, to address HSI downstream tasks \emph{without} HSI-specific pretraining. Therefore, we implement and compare two channel adaptation strategies: Naive Band Selection and physics-aware Spectral Response Function (SRF) grouping. Overall, our results indicate a general superiority of deep learning models with native support of HSI data. Our experiments also demonstrate the ability of TerraMind to adapt to HSI downstream tasks through band selection with moderate performance decline. Therefore, the findings of this research establish a critical baseline for HSI integration, motivating the need for native spectral tokenization in future multimodal model architectures.
>
---
#### [replaced 091] Phased DMD: Few-step Distribution Matching Distillation via Score Matching within Subintervals
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.27684](https://arxiv.org/pdf/2510.27684)**

> **作者:** Xiangyu Fan; Zesong Qiu; Zhuguanyu Wu; Fanzhou Wang; Zhiqian Lin; Tianxiang Ren; Dahua Lin; Ruihao Gong; Lei Yang
>
> **摘要:** Distribution Matching Distillation (DMD) distills score-based generative models into efficient one-step generators, without requiring a one-to-one correspondence with the sampling trajectories of their teachers. Yet, the limited capacity of one-step distilled models compromises generative diversity and degrades performance in complex generative tasks, e.g., generating intricate object motions in text-to-video task. Directly extending DMD to multi-step distillation increases memory usage and computational depth, leading to instability and reduced efficiency. While prior works propose stochastic gradient truncation as a potential solution, we observe that it substantially reduces the generative diversity in text-to-image generation and slows motion dynamics in video generation, reducing performance to the level of one-step models. To address these limitations, we propose Phased DMD, a multi-step distillation framework that bridges the idea of phase-wise distillation with Mixture-of-Experts (MoE), reducing learning difficulty while enhancing model capacity. Phased DMD incorporates two key ideas: progressive distribution matching and score matching within subintervals. First, our model divides the SNR range into subintervals, progressively refining the model to higher SNR levels, to better capture complex distributions. Next, to ensure accurate training within each subinterval, we derive rigorous mathematical formulations for the objective. We validate Phased DMD by distilling state-of-the-art image and video generation models, including Qwen-Image-20B and Wan2.2-28B. Experiments demonstrate that Phased DMD enhances motion dynamics, improves visual fidelity in video generation, and increases output diversity in image generation. Our code and models are available at this https URL.
>
---
#### [replaced 092] Image Generation from Contextually-Contradictory Prompts
- **分类: cs.GR; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.01929](https://arxiv.org/pdf/2506.01929)**

> **作者:** Saar Huberman; Or Patashnik; Omer Dahary; Ron Mokady; Daniel Cohen-Or
>
> **备注:** Project page: this https URL
>
> **摘要:** Text-to-image diffusion models excel at generating high-quality, diverse images from natural language prompts. However, they often fail to produce semantically accurate results when the prompt contains concept combinations that contradict their learned priors. We define this failure mode as contextual contradiction, where one concept implicitly negates another due to entangled associations learned during training. To address this, we propose a stage-aware prompt decomposition framework that guides the denoising process using a sequence of proxy prompts. Each proxy prompt is constructed to match the semantic content expected to emerge at a specific stage of denoising, while ensuring contextual coherence. To construct these proxy prompts, we leverage a large language model (LLM) to analyze the target prompt, identify contradictions, and generate alternative expressions that preserve the original intent while resolving contextual conflicts. By aligning prompt information with the denoising progression, our method enables fine-grained semantic control and accurate image generation in the presence of contextual contradictions. Experiments across a variety of challenging prompts show substantial improvements in alignment to the textual prompt.
>
---
#### [replaced 093] FlyPrompt: Brain-Inspired Random-Expanded Routing with Temporal-Ensemble Experts for General Continual Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.01976](https://arxiv.org/pdf/2602.01976)**

> **作者:** Hongwei Yan; Guanglong Sun; Kanglei Zhou; Qian Li; Liyuan Wang; Yi Zhong
>
> **备注:** 34 pages. Accepted by ICLR 2026
>
> **摘要:** General continual learning (GCL) challenges intelligent systems to learn from single-pass, non-stationary data streams without clear task boundaries. While recent advances in continual parameter-efficient tuning (PET) of pretrained models show promise, they typically rely on multiple training epochs and explicit task cues, limiting their effectiveness in GCL scenarios. Moreover, existing methods often lack targeted design and fail to address two fundamental challenges in continual PET: how to allocate expert parameters to evolving data distributions, and how to improve their representational capacity under limited supervision. Inspired by the fruit fly's hierarchical memory system characterized by sparse expansion and modular ensembles, we propose FlyPrompt, a brain-inspired framework that decomposes GCL into two subproblems: expert routing and expert competence improvement. FlyPrompt introduces a randomly expanded analytic router for instance-level expert activation and a temporal ensemble of output heads to dynamically adapt decision boundaries over time. Extensive theoretical and empirical evaluations demonstrate FlyPrompt's superior performance, achieving up to 11.23%, 12.43%, and 7.62% gains over state-of-the-art baselines on CIFAR-100, ImageNet-R, and CUB-200, respectively. Our source code is available at this https URL.
>
---
#### [replaced 094] myMNIST: Benchmark of PETNN, KAN, and Classical Deep Learning Models for Burmese Handwritten Digit Recognition
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于缅甸手写数字识别任务，旨在建立基准测试。评估多种模型性能，提供可复现的基线，比较传统与新兴架构效果。**

- **链接: [https://arxiv.org/pdf/2603.18597](https://arxiv.org/pdf/2603.18597)**

> **作者:** Ye Kyaw Thu; Thazin Myint Oo; Thepchai Supnithi
>
> **备注:** 7 pages, 2 figures, 3 tables, Accepted to ICNLP 2026, Xi'an, China
>
> **摘要:** We present the first systematic benchmark on a standardized iteration of the publicly available Burmese Handwritten Digit Dataset (BHDD), which we have designated as myMNIST Benchmarking. While BHDD serves as a foundational resource for Myanmar NLP/AI, it lacks a comprehensive, reproducible performance baseline across modern architectures. We evaluate eleven architectures spanning classical deep learning models (Multi-Layer Perceptron, Convolutional Neural Network, Long Short-Term Memory, Gated Recurrent Unit, Transformer), recent alternatives (FastKAN, EfficientKAN), an energy-based model (JEM), and physics-inspired PETNN variants (Sigmoid, GELU, SiLU). Using Precision, Recall, F1-Score, and Accuracy as evaluation metrics, our results show that the CNN remains a strong baseline, achieving the best overall scores (F1 = 0.9959, Accuracy = 0.9970). The PETNN (GELU) model closely follows (F1 = 0.9955, Accuracy = 0.9966), outperforming LSTM, GRU, Transformer, and KAN variants. JEM, representing energy-based modeling, performs competitively (F1 = 0.9944, Accuracy = 0.9958). KAN-based models (FastKAN, EfficientKAN) trail the top performers but provide a meaningful alternative baseline (Accuracy ~0.992). These findings (i) establish reproducible baselines for BHDD across diverse modeling paradigms, (ii) highlight PETNN's strong performance relative to classical and Transformer-based models, and (iii) quantify the gap between energy-inspired PETNNs and a true energy-based model (JEM). We release this benchmark to facilitate future research on Myanmar digit recognition and to encourage broader evaluation of emerging architectures on regional scripts.
>
---
#### [replaced 095] The Potential of Copernicus Satellites for Disaster Response: Retrieving Building Damage from Sentinel-1 and Sentinel-2
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.05461](https://arxiv.org/pdf/2511.05461)**

> **作者:** Olivier Dietrich; Merlin Alfredsson; Emilia Arens; Nando Metzger; Torben Peters; Linus Scheibenreif; Jan Dirk Wegner; Konrad Schindler
>
> **摘要:** Natural disasters demand rapid damage assessment to guide humanitarian response. Here, we investigate whether medium-resolution Earth observation images from the Copernicus program can support building damage assessment, complementing very-high resolution imagery with often limited availability. We introduce xBD-S12, a dataset of 10,315 pre- and post-disaster image pairs from both Sentinel-1 and Sentinel-2, spatially and temporally aligned with the established xBD benchmark. In a series of experiments, we demonstrate that building damage can be detected and mapped rather well in many disaster scenarios, despite the moderate 10$\,$m ground sampling distance. We also find that, for damage mapping at that resolution, architectural sophistication does not seem to bring much advantage: more complex model architectures tend to struggle with generalization to unseen disasters, and geospatial foundation models bring little practical benefit. Our results suggest that Copernicus images are a viable data source for rapid, wide-area damage assessment and could play an important role alongside VHR imagery. We release the xBD-S12 dataset, code, and trained models to support further research at this https URL .
>
---
#### [replaced 096] SurgMotion: A Video-Native Foundation Model for Universal Understanding of Surgical Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.05638](https://arxiv.org/pdf/2602.05638)**

> **作者:** Jinlin Wu; Felix Holm; Chuxi Chen; An Wang; Yaxin Hu; Xiaofan Ye; Zelin Zang; Miao Xu; Lihua Zhou; Huai Liao; Danny T. M. Chan; Ming Feng; Wai S. Poon; Hongliang Ren; Dong Yi; Nassir Navab; Gaofeng Meng; Jiebo Luo; Hongbin Liu; Zhen Lei
>
> **摘要:** While foundation models have advanced surgical video analysis, current approaches rely predominantly on pixel-level reconstruction objectives that waste model capacity on low-level visual details, such as smoke, specular reflections, and fluid motion, rather than semantic structures essential for surgical understanding. We present SurgMotion, a video-native foundation model that shifts the learning paradigm from pixel-level reconstruction to latent motion prediction. Built on the Video Joint Embedding Predictive Architecture (V-JEPA), SurgMotion introduces three key technical innovations tailored to surgical videos: (1) motion-guided latent masked prediction to prioritize semantically meaningful regions, (2) spatiotemporal affinity self-distillation to enforce relational consistency, and (3) spatiotemporal feature diversity regularization (SFDR) to prevent representation collapse in texture-sparse surgical scenes. To enable large-scale pretraining, we curate SurgMotion-15M, the largest surgical video dataset to date, comprising 3,658 hours of video from 50 sources across 13 anatomical regions. Extensive experiments across 17 benchmarks demonstrate that SurgMotion significantly outperforms state-of-the-art methods on surgical workflow recognition, achieving 14.6 percent improvement in F1 score on EgoSurgery and 10.3 percent on PitVis; on action triplet recognition with 39.54 percent mAP-IVT on CholecT50; as well as on skill assessment, polyp segmentation, and depth estimation. These results establish SurgMotion as a new standard for universal, motion-oriented surgical video understanding.
>
---
#### [replaced 097] Vision-DeepResearch: Incentivizing DeepResearch Capability in Multimodal Large Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.22060](https://arxiv.org/pdf/2601.22060)**

> **作者:** Wenxuan Huang; Yu Zeng; Qiuchen Wang; Zhen Fang; Shaosheng Cao; Zheng Chu; Qingyu Yin; Shuang Chen; Zhenfei Yin; Lin Chen; Zehui Chen; Xu Tang; Yao Hu; Shaohui Lin; Philip Torr; Feng Zhao; Wanli Ouyang
>
> **摘要:** Multimodal large language models (MLLMs) have achieved remarkable success across a broad range of vision tasks. However, constrained by the capacity of their internal world knowledge, prior work has proposed augmenting MLLMs by ``reasoning-then-tool-call'' for visual and textual search engines to obtain substantial gains on tasks requiring extensive factual information. However, these approaches typically define multimodal search in a naive setting, assuming that a single full-level or entity-level image query and few text query suffices to retrieve the key evidence needed to answer the question, which is unrealistic in real-world scenarios with substantial visual noise. Moreover, they are often limited in the reasoning depth and search breadth, making it difficult to solve complex questions that require aggregating evidence from diverse visual and textual sources. Building on this, we propose Vision-DeepResearch, which proposes one new multimodal deep-research paradigm, i.e., performs multi-turn, multi-entity and multi-scale visual and textual search to robustly hit real-world search engines under heavy noise. Our Vision-DeepResearch supports dozens of reasoning steps and hundreds of engine interactions, while internalizing deep-research capabilities into the MLLM via cold-start supervision and RL training, resulting in a strong end-to-end multimodal deep-research MLLM. It substantially outperforming existing multimodal deep-research MLLMs, and workflows built on strong closed-source foundation model such as GPT-5, Gemini-2.5-pro and Claude-4-Sonnet. The code will be released in this https URL.
>
---
#### [replaced 098] Arc Gradient Descent: A Geometrically Motivated Gradient Descent-based Optimiser with Phase-Aware, User-Controlled Step Dynamics (proof-of-concept)
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.NE**

- **简介: 该论文提出ArcGD优化器，解决机器学习中的优化问题。通过实验验证其在非凸函数和实际数据集上的优越性能，展示其泛化能力和抗过拟合特性。**

- **链接: [https://arxiv.org/pdf/2512.06737](https://arxiv.org/pdf/2512.06737)**

> **作者:** Nikhil Verma; Joonas Linnosmaa; Leonardo Espinosa-Leal; Napat Vajragupta
>
> **备注:** 90 pages, 6 appendices, proof-of-concept
>
> **摘要:** The paper presents the formulation, implementation, and evaluation of the ArcGD optimiser. The evaluation is conducted initially on a non-convex benchmark function and subsequently on a real-world ML dataset. The initial comparative study using the Adam optimiser is conducted on a stochastic variant of the highly non-convex and notoriously challenging Rosenbrock function, renowned for its narrow, curved valley, across dimensions ranging from 2D to 1000D and an extreme case of 50,000D. Two configurations were evaluated to eliminate learning-rate bias: (i) both using ArcGD's effective learning rate and (ii) both using Adam's default learning rate. ArcGD consistently outperformed Adam under the first setting and, although slower under the second, achieved superior final solutions in most cases. In the second evaluation, ArcGD is evaluated against state-of-the-art optimizers (Adam, AdamW, Lion, SGD) on the CIFAR-10 image classification dataset across 8 diverse MLP architectures ranging from 1 to 5 hidden layers. ArcGD achieved the highest average test accuracy (50.7%) at 20,000 iterations, outperforming AdamW (46.6%), Adam (46.8%), SGD (49.6%), and Lion (43.4%), winning or tying on 6 of 8 architectures. Notably, while Adam and AdamW showed strong early convergence at 5,000 iterations, but regressed with extended training, whereas ArcGD continued improving, demonstrating generalization and resistance to overfitting without requiring early stopping tuning. Strong performance on geometric stress tests and standard deep-learning benchmarks indicates broad applicability, highlighting the need for further exploration. Moreover, it is also shown that both a limiting variant of ArcGD and a momentum augmented ArcGD, recover sign-based momentum updates, revealing a clear conceptual link between ArcGD's phase structure and the core mechanism of the Lion Optimiser.
>
---
#### [replaced 099] Operational machine learning for remote spectroscopic detection of CH$_{4}$ point sources
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.07719](https://arxiv.org/pdf/2511.07719)**

> **作者:** Vít Růžička; Gonzalo Mateo-García; Itziar Irakulis-Loitxate; Juan Emmanuel Johnson; Manuel Montesino San Martín; Anna Allen; Alma Raunak; Carol Castaneda; Luis Guanter; David R. Thompson
>
> **备注:** 20 pages, 14 figures, 10 tables. In review
>
> **摘要:** Mitigating anthropogenic methane sources is one of the most cost-effective levers to slow down global warming. While satellite-based imaging spectrometers, such as EMIT, PRISMA, and EnMAP, can detect these point sources, current methane retrieval methods based on matched filters produce a high number of false detections requiring manual verification. To address this challenge, we deployed a ML system for detecting methane emissions within the Methane Alert and Response System (MARS) of UNEP's IMEO. This represents the first operational deployment of automated methane point-source detection using spaceborne imaging spectrometers, providing regular global coverage and scalability to future constellations with even higher data volumes. This task required several technical advances. First, we created one of the largest and most diverse and global ML ready datasets to date of annotated methane plumes from three imaging spectrometer missions, and quantitatively compared different deep learning model configurations. Second, we extended prior evaluation methodologies from small, tiled datasets to full granules that are more representative of operational use. This revealed that deep learning models still produce a large number of false detections, a problem we addressed with model ensembling, which reduced false detections by over 74%. During 11 months of operational deployment, our system processed more than 25,000 hyperspectral products faciliting the verification of 2,851 distinct methane leaks, which resulted in 834 stakeholder notifications. We further demonstrate the model's utility in verifying mitigation success through case studies in Libya, Argentina, Oman, and Azerbaijan. Our work represents a critical step towards a global AI-assisted methane leak detection system, which is required to process the dramatically higher data volumes expected from current and future imaging spectrometers.
>
---
#### [replaced 100] Architecture-Aware Minimization (A$^2$M): How to Find Flat Minima in Neural Architecture Search
- **分类: cs.LG; cond-mat.dis-nn; cs.CV**

- **链接: [https://arxiv.org/pdf/2503.10404](https://arxiv.org/pdf/2503.10404)**

> **作者:** Matteo Gambella; Fabrizio Pittorino; Manuel Roveri
>
> **备注:** Published in the journal Machine Learning: Science and Technology - IOPscience
>
> **摘要:** Neural Architecture Search (NAS) has become an essential tool for designing effective and efficient neural networks. In this paper, we investigate the geometric properties of neural architecture spaces commonly used in differentiable NAS methods, specifically NAS-Bench-201 and DARTS. By defining flatness metrics such as neighborhoods and loss barriers along paths in architecture space, we reveal locality and flatness characteristics analogous to the well-known properties of neural network loss landscapes in weight space. In particular, we find that highly accurate architectures cluster together in flat regions, while suboptimal architectures remain isolated, unveiling the detailed geometrical structure of the architecture search landscape. Building on these insights, we propose Architecture-Aware Minimization (A$^2$M), a novel analytically derived algorithmic framework that explicitly biases, for the first time, the gradient of differentiable NAS methods towards flat minima in architecture space. A$^2$M consistently improves generalization over state-of-the-art DARTS-based algorithms on benchmark datasets including CIFAR-10, CIFAR-100, and ImageNet16-120, across both NAS-Bench-201 and DARTS search spaces. Notably, A$^2$M is able to increase the test accuracy, on average across different differentiable NAS methods, by +3.60\% on CIFAR-10, +4.60\% on CIFAR-100, and +3.64\% on ImageNet16-120, demonstrating its superior effectiveness in practice. A$^2$M can be easily integrated into existing differentiable NAS frameworks, offering a versatile tool for future research and applications in automated machine learning. We open-source our code at this https URL.
>
---
#### [replaced 101] Investigating self-supervised representations for audio-visual deepfake detection
- **分类: cs.CV; cs.LG; cs.SD**

- **简介: 该论文属于音频-视频深度伪造检测任务，旨在探索自监督表示的有效性。研究评估了不同模态和领域的自监督特征，发现其具有互补性且能捕捉深度伪造相关信息，但真实场景下仍面临挑战。**

- **链接: [https://arxiv.org/pdf/2511.17181](https://arxiv.org/pdf/2511.17181)**

> **作者:** Dragos-Alexandru Boldisor; Stefan Smeu; Dan Oneata; Elisabeta Oneata
>
> **备注:** Accepted to the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** Self-supervised representations excel at many vision and speech tasks, but their potential for audio-visual deepfake detection remains underexplored. Unlike prior work that uses these features in isolation or buried within complex architectures, we systematically evaluate them across modalities (audio, video, multimodal) and domains (lip movements, generic visual content). We assess three key dimensions: detection effectiveness, interpretability of encoded information, and cross-modal complementarity. We find that most self-supervised features capture deepfake-relevant information, and that this information is complementary. Moreover, models primarily attend to semantically meaningful regions rather than spurious artifacts (such as the leading silence). Among the investigated features, audio-informed representations generalize best and achieve state-of-the-art results. However, generalization to realistic in-the-wild data remains challenging. Our analysis indicates this gap stems from intrinsic dataset difficulty rather than from features latching onto superficial patterns. Project webpage: this https URL.
>
---
#### [replaced 102] Test-time Ego-Exo-centric Adaptation for Action Anticipation via Multi-Label Prototype Growing and Dual-Clue Consistency
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.09798](https://arxiv.org/pdf/2603.09798)**

> **作者:** Zhaofeng Shi; Heqian Qiu; Lanxiao Wang; Qingbo Wu; Fanman Meng; Lili Pan; Hongliang Li
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Efficient adaptation between Egocentric (Ego) and Exocentric (Exo) views is crucial for applications such as human-robot cooperation. However, the success of most existing Ego-Exo adaptation methods relies heavily on target-view data for training, thereby increasing computational and data collection costs. In this paper, we make the first exploration of a Test-time Ego-Exo Adaptation for Action Anticipation (TE$^{2}$A$^{3}$) task, which aims to adjust the source-view-trained model online during test time to anticipate target-view actions. It is challenging for existing Test-Time Adaptation (TTA) methods to address this task due to the multi-action candidates and significant temporal-spatial inter-view gap. Hence, we propose a novel Dual-Clue enhanced Prototype Growing Network (DCPGN), which accumulates multi-label knowledge and integrates cross-modality clues for effective test-time Ego-Exo adaptation and action anticipation. Specifically, we propose a Multi-Label Prototype Growing Module (ML-PGM) to balance multiple positive classes via multi-label assignment and confidence-based reweighting for class-wise memory banks, which are updated by an entropy priority queue strategy. Then, the Dual-Clue Consistency Module (DCCM) introduces a lightweight narrator to generate textual clues indicating action progressions, which complement the visual clues containing various objects. Moreover, we constrain the inferred textual and visual logits to construct dual-clue consistency for temporally and spatially bridging Ego and Exo views. Extensive experiments on the newly proposed EgoMe-anti and the existing EgoExoLearn benchmarks show the effectiveness of our method, which outperforms related state-of-the-art methods by a large margin. Code is available at \href{this https URL}{this https URL}.
>
---
#### [replaced 103] UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.11336](https://arxiv.org/pdf/2512.11336)**

> **作者:** Hewen Pan; Cong Wei; Dashuang Liang; Zepeng Huang; Pengfei Gao; Ziqi Zhou; Lulu Xue; Pengfei Yan; Xiaoming Wei; Minghui Li; Shengshan Hu
>
> **备注:** CVPR 2026 Camera Ready, Github Code: this https URL
>
> **摘要:** With the advancement of multi-modal Large Language Models (LLMs), Video LLMs have been further developed to perform on holistic and specialized video understanding. However, existing works are limited to specialized video understanding tasks, failing to achieve a comprehensive and multi-grained video perception. To bridge this gap, we introduce UFVideo, the first Video LLM with unified multi-grained cooperative understanding capabilities. Specifically, we design unified visual-language guided alignment to flexibly handle video understanding across global, pixel and temporal scales within a single model. UFVideo dynamically encodes the visual and text inputs of different tasks and generates the textual response, temporal localization, or grounded mask. Additionally, to evaluate challenging multi-grained video understanding tasks, we construct the UFVideo-Bench consisting of three distinct collaborative tasks within the scales, which demonstrates UFVideo's flexibility and advantages over GPT-4o. Furthermore, we validate the effectiveness of our model across 9 public benchmarks covering various common video understanding tasks, providing valuable insights for future Video LLMs.
>
---
#### [replaced 104] GazeShift: Unsupervised Gaze Estimation and Dataset for VR
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.07832](https://arxiv.org/pdf/2603.07832)**

> **作者:** Gil Shapira; Ishay Goldin; Evgeny Artyomov; Donghoon Kim; Yosi Keller; Niv Zehngut
>
> **备注:** Accepted to CVPR26
>
> **摘要:** Gaze estimation is instrumental in modern virtual reality (VR) systems. Despite significant progress in remote-camera gaze estimation, VR gaze research remains constrained by data scarcity, particularly the lack of large-scale, accurately labeled datasets captured with the off-axis camera configurations typical of modern headsets. Gaze annotation is difficult since fixation on intended targets cannot be guaranteed. To address these challenges, we introduce VRGaze, the first large-scale off-axis gaze estimation dataset for VR, comprising 2.1 million near-eye infrared images collected from 68 participants. We further propose GazeShift, an attention-guided unsupervised framework for learning gaze representations without labeled data. Unlike prior redirection-based methods that rely on multi-view or 3D geometry, GazeShift is tailored to near-eye imagery, achieving effective gaze-appearance disentanglement in a compact, real-time model. GazeShift embeddings can be optionally adapted to individual users via lightweight few-shot calibration, achieving a 1.84° mean error on VRGaze. On the remote-camera MPIIGaze dataset, the model achieves a 7.15° person-agnostic error, doing so with 10x fewer parameters and 35x fewer FLOPs than baseline methods. Deployed natively on a VR headset GPU, inference takes only 5 ms. Combined with demonstrated robustness to illumination changes, these results highlight GazeShift as a label-efficient, real-time solution for VR gaze tracking. Project code and the VRGaze dataset are released at this https URL
>
---
#### [replaced 105] Cross-Domain Underwater Image Enhancement Guided by No-Reference Image Quality Assessment: A Transfer Learning Approach
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.17937](https://arxiv.org/pdf/2503.17937)**

> **作者:** Zhi Zhang; Minfu Li; Lu Li; Daoyi Chen
>
> **摘要:** Single underwater image enhancement (UIE) is a challenging ill-posed problem, but its development is hindered by two major issues: (1) The labels in underwater reference datasets are pseudo labels, relying on these pseudo ground truths in supervised learning leads to domain discrepancy. (2) Underwater reference datasets are scarce, making training on such small datasets prone to overfitting and distribution shift. To address these challenges, we propose Trans-UIE, a transfer learning-based UIE model that captures the fundamental paradigms of UIE through pretraining and utilizes a dataset composed of both reference and non-reference datasets for fine-tuning. However, fine-tuning the model using only reconstruction loss may introduce confirmation bias. To mitigate this, our method leverages no-reference image quality assessment (NR-IQA) metrics from above-water scenes to guide the transfer learning process across domains while generating enhanced images with the style of the above-water image domain. Additionally, to reduce the risk of overfitting during the pretraining stage, we introduce Pearson correlation loss. Experimental results on both full-reference and no-reference underwater benchmark datasets demonstrate that Trans-UIE significantly outperforms state-of-the-art methods.
>
---
#### [replaced 106] GHOST: Ground-projected Hypotheses from Observed Structure-from-Motion Trajectories
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶中的轨迹分割任务，旨在从单目图像中分割可行车辆轨迹。通过自监督学习，利用行车轨迹生成标签，训练模型预测运动条件下的路径建议。**

- **链接: [https://arxiv.org/pdf/2603.20583](https://arxiv.org/pdf/2603.20583)**

> **作者:** Tomasz Frelek; Rohan Patil; Akshar Tumu; Henrik I. Christensen
>
> **备注:** 8 pages, 27 figures, 1 table
>
> **摘要:** We present a scalable self-supervised approach for segmenting feasible vehicle trajectories from monocular images for autonomous driving in complex urban environments. Leveraging large-scale dashcam videos, we treat recorded ego-vehicle motion as implicit supervision and recover camera trajectories via monocular structure-from-motion, projecting them onto the ground plane to generate spatial masks of traversed regions without manual annotation. These automatically generated labels are used to train a deep segmentation network that predicts motion-conditioned path proposals from a single RGB image at run time, without explicit modeling of road or lane markings. Trained on diverse, unconstrained internet data, the model implicitly captures scene layout, lane topology, and intersection structure, and generalizes across varying camera configurations. We evaluate our approach on NuScenes, demonstrating reliable trajectory prediction, and further show transfer to an electric scooter platform through light fine-tuning. Our results indicate that large-scale ego-motion distillation yields structured and generalizable path proposals beyond the demonstrated trajectory, enabling trajectory hypothesis estimation via image segmentation.
>
---
#### [replaced 107] Schrödinger's Navigator: Imagining an Ensemble of Futures for Zero-Shot Object Navigation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于零样本目标导航任务，旨在解决机器人在未知环境中安全定位目标的问题。提出Schrödinger's Navigator框架，通过推理多个未来场景实现更鲁棒的导航。**

- **链接: [https://arxiv.org/pdf/2512.21201](https://arxiv.org/pdf/2512.21201)**

> **作者:** Yu He; Da Huang; Zhenyang Liu; Zixiao Gu; Qiang Sun; Guangnan Ye; Yanwei Fu; Yu-Gang Jiang
>
> **摘要:** Zero-shot object navigation (ZSON) requires robots to locate target objects in unseen environments without task-specific fine-tuning or pre-built maps, a capability crucial for service and household robotics. Existing methods perform well in simulation but struggle in realistic, cluttered environments where heavy occlusions and latent hazards make large portions of the scene unobserved. These approaches typically act on a single inferred scene, making them prone to overcommitment and unsafe behavior under uncertainty. To address these challenges, we propose Schrödinger's Navigator, a belief-aware framework that explicitly reasons over multiple trajectory-conditioned imagined 3D futures at inference time. A trajectory-conditioned 3D world model generates hypothetical observations along candidate paths, maintaining a superposition of plausible scene realizations. An adaptive, occluder-aware trajectory sampling strategy focuses imagination on uncertain regions, while a Future-Aware Value Map (FAVM) aggregates imagined futures to guide robust, proactive action selection. Evaluations in simulation and on a physical Go2 quadruped robot demonstrate that Schrödinger's Navigator outperforms strong ZSON baselines, achieving more robust self-localization, object localization, and safe navigation under severe occlusions and latent hazards. These results highlight the effectiveness of reasoning over imagined 3D futures as a scalable and generalizable strategy for zero-shot navigation in uncertain real-world environments.
>
---
#### [replaced 108] Classification of Microplastic Particles in Water using Polarized Light Scattering and Machine Learning Methods
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.06901](https://arxiv.org/pdf/2511.06901)**

> **作者:** Leonard Saur; Marc von Pawlowski; Ulrich Gengenbach; Ingo Sieber; Hossein Shirali; Lorenz Wührl; Xiangyu Weng; Rainer Kiko; Christian Pylatiuk
>
> **备注:** 22 pages, 9 figures
>
> **摘要:** The detection and classification of microplastics in water remain a significant challenge due to their diverse properties and the limitations of traditional optical methods. Standard spectroscopic techniques often suffer from the strong infrared absorption of water, while many emerging optical approaches rely on transmission geometries that require sample transparency. This study presents a systematic classification framework utilizing 120 degree backscattering reflection polarimetry and deep learning to identify common polymers (HDPE, LDPE, and PP) directly in water. This backscattering-based approach is specifically designed to analyze opaque, irregularly shaped particles that lack distinguishable surface features under standard illumination. To ensure high-fidelity data, we introduce a feedback review loop to identify and remove outliers, which significantly stabilizes model training and improves generalization. This framework is validated on a dataset of 600 individually imaged microplastic fragments spanning three polymer types. Our results evaluate the distinct contributions of the Angle of Linear Polarization and the Degree of Linear Polarization to the classification process. By implementing a late fusion architecture to combine these signals, we achieve an average test accuracy of 83 percent. Finally, a systematic feature hierarchy analysis reveals that the convolutional neural network relies on internal polarization textures associated with the particle's microstructure, rather than on macro-contours, with classification accuracy declining by over 40 percent when internal structure is removed. This demonstrates that the system extracts polarization-dependent internal structural information that is inaccessible to conventional intensity-only imaging methods.
>
---
#### [replaced 109] Efficient Chest X-ray Representation Learning via Semantic-Partitioned Contrastive Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.07113](https://arxiv.org/pdf/2603.07113)**

> **作者:** Wangyu Feng; Shawn Young; Lijian Xu
>
> **摘要:** Self-supervised learning (SSL) has emerged as a powerful paradigm for Chest X-ray (CXR) analysis under limited annotations. Yet, existing SSL strategies remain suboptimal for medical imaging. Masked image modeling allocates substantial computation to reconstructing high-frequency background details with limited diagnostic value. Contrastive learning, on the other hand, often depends on aggressive augmentations that risk altering clinically meaningful structures. We introduce Semantic-Partitioned Contrastive Learning (S-PCL), an efficient pre-training framework tailored for CXR representation learning. Instead of reconstructing pixels or relying on heavy augmentations, S-PCL randomly partitions patch tokens from a single CXR into two non-overlapping semantic subsets. Each subset provides a complementary but incomplete view. The encoder must maximize agreement between these partitions, implicitly inferring global anatomical layout and local pathological cues from partial evidence. This semantic partitioning forms an internal bottleneck that enforces long-range dependency modeling and structural coherence. S-PCL eliminates the need for hand-crafted augmentations, auxiliary decoders, and momentum encoders. The resulting architecture is streamlined, computationally efficient, and easy to scale. Extensive experiments on large-scale CXR benchmarks, including ChestX-ray14, CheXpert, RSNA Pneumonia and SIIM-ACR Pneumothorax, show that S-PCL achieves competitive performance while attaining the lowest GFLOPs and superior accuracy among existing SSL approaches.
>
---
#### [replaced 110] 2Xplat: Two Experts Are Better Than One Generalist
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.21064](https://arxiv.org/pdf/2603.21064)**

> **作者:** Hwasik Jeong; Seungryong Lee; Gyeongjin Kang; Seungkwon Yang; Xiangyu Sun; Seungtae Nam; Eunbyung Park
>
> **备注:** Project page: this https URL
>
> **摘要:** Pose-free feed-forward 3D Gaussian Splatting (3DGS) has opened a new frontier for rapid 3D modeling, enabling high-quality Gaussian representations to be generated from uncalibrated multi-view images in a single forward pass. The dominant approach in this space adopts unified monolithic architectures, often built on geometry-centric 3D foundation models, to jointly estimate camera poses and synthesize 3DGS representations within a single network. While architecturally streamlined, such "all-in-one" designs may be suboptimal for high-fidelity 3DGS generation, as they entangle geometric reasoning and appearance modeling within a shared representation. In this work, we introduce 2Xplat, a pose-free feed-forward 3DGS framework based on a two-expert design that explicitly separates geometry estimation from Gaussian generation. A dedicated geometry expert first predicts camera poses, which are then explicitly passed to a powerful appearance expert that synthesizes 3D Gaussians. Despite its conceptual simplicity, being largely underexplored in prior works, the proposed approach proves highly effective. In fewer than 5K training iterations, the proposed two-experts pipeline substantially outperforms prior pose-free feed-forward 3DGS approaches and achieves performance on par with state-of-the-art posed methods. These results challenge the prevailing unified paradigm and suggest the potential advantages of modular design principles for complex 3D geometric estimation and appearance synthesis tasks.
>
---
#### [replaced 111] DiffBMP: Differentiable Rendering with Bitmap Primitives
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.22625](https://arxiv.org/pdf/2602.22625)**

> **作者:** Seongmin Hong; Junghun James Kim; Daehyeop Kim; Insoo Chung; Se Young Chun
>
> **备注:** Accepted to CVPR 2026, this https URL
>
> **摘要:** We introduce DiffBMP, a scalable and efficient differentiable rendering engine for a collection of bitmap images. Our work addresses a limitation that traditional differentiable renderers are constrained to vector graphics, given that most images in the world are bitmaps. Our core contribution is a highly parallelized rendering pipeline, featuring a custom CUDA implementation for calculating gradients. This system can, for example, optimize the position, rotation, scale, color, and opacity of thousands of bitmap primitives all in under 1 min using a consumer GPU. We employ and validate several techniques to facilitate the optimization: soft rasterization via Gaussian blur, structure-aware initialization, noisy canvas, and specialized losses/heuristics for videos or spatially constrained images. We demonstrate DiffBMP is not just an isolated tool, but a practical one designed to integrate into creative workflows. It supports exporting compositions to a native, layered file format, and the entire framework is publicly accessible via an easy-to-hack Python package.
>
---
#### [replaced 112] Quasi-Conformal Convolution : A Learnable Convolution for Deep Learning on Simply Connected Open Surfaces
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.01356](https://arxiv.org/pdf/2502.01356)**

> **作者:** Han Zhang; Tsz Lok Ip; Lok Ming Lui
>
> **摘要:** Deep learning on non-Euclidean domains is important for analyzing complex geometric data that lacks common coordinate systems and familiar Euclidean properties. A central challenge in this field is to define convolution on domains, which inherently possess irregular and non-Euclidean structures. In this work, we introduce Quasi-conformal Convolution (QCC), a novel framework for defining convolution on simply-connected open surfaces using quasi-conformal theories. Each QCC operator is linked to a specific quasi-conformal mapping, enabling the adjustment of the convolution operation through manipulation of this mapping. By utilizing trainable estimator modules that produce quasi-conformal mappings, QCC facilitates adaptive and learnable convolution operators that can be dynamically adjusted according to the underlying data structured on the surfaces. QCC unifies a broad range of spatially defined convolutions, facilitating the learning of tailored convolution operators on each underlying surface optimized for specific tasks. Building on this foundation, we develop the Quasi-Conformal Convolutional Neural Network (QCCNN) to address a variety of tasks related to geometric data. We validate the efficacy of QCCNN through the classification of images defined on curvilinear simply-connected open Riemann surfaces, demonstrating superior performance in this context. Additionally, we explore its potential in medical applications, including craniofacial analysis using 3D facial data and lesion segmentation on 3D human faces, achieving enhanced accuracy and reliability.
>
---
#### [replaced 113] UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.03334](https://arxiv.org/pdf/2511.03334)**

> **作者:** Guozhen Zhang; Zixiang Zhou; Teng Hu; Ziqiao Peng; Youliang Zhang; Yi Chen; Yuan Zhou; Qinglin Lu; Limin Wang
>
> **备注:** CVPR 2026
>
> **摘要:** Due to the lack of effective cross-modal modeling, existing open-source audio-video generation methods often exhibit compromised lip synchronization and insufficient semantic consistency. To mitigate these drawbacks, we propose UniAVGen, a unified framework for joint audio and video generation. UniAVGen is anchored in a dual-branch joint synthesis architecture, incorporating two parallel Diffusion Transformers (DiTs) to build a cohesive cross-modal latent space. At its heart lies an Asymmetric Cross-Modal Interaction mechanism, which enables bidirectional, temporally aligned cross-attention, thus ensuring precise spatiotemporal synchronization and semantic consistency. Furthermore, this cross-modal interaction is augmented by a Face-Aware Modulation module, which dynamically prioritizes salient regions in the interaction process. To enhance generative fidelity during inference, we additionally introduce Modality-Aware Classifier-Free Guidance, a novel strategy that explicitly amplifies cross-modal correlation signals. Notably, UniAVGen's robust joint synthesis design enables seamless unification of pivotal audio-video tasks within a single model, such as joint audio-video generation and continuation, video-to-audio dubbing, and audio-driven video synthesis. Comprehensive experiments validate that, with far fewer training samples (1.3M vs. 30.1M), UniAVGen delivers overall advantages in audio-video synchronization, timbre consistency, and emotion consistency.
>
---
