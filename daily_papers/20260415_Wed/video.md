# 计算机视觉 cs.CV

- **最新发布 140 篇**

- **更新 93 篇**

## 最新发布

#### [new 001] GeoAlign: Geometric Feature Realignment for MLLM Spatial Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态大语言模型的 spatial reasoning 任务，旨在解决几何特征与模型需求不匹配的问题。提出 GeoAlign 框架，动态聚合多层几何特征以提升性能。**

- **链接: [https://arxiv.org/pdf/2604.12630](https://arxiv.org/pdf/2604.12630)**

> **作者:** Zhaochen Liu; Limeng Qiao; Guanglu Wan; Tingting Jiang
>
> **摘要:** Multimodal large language models (MLLMs) have exhibited remarkable performance in various visual tasks, yet still struggle with spatial reasoning. Recent efforts mitigate this by injecting geometric features from 3D foundation models, but rely on static single-layer extractions. We identify that such an approach induces a task misalignment bias: the geometric features naturally evolve towards 3D pretraining objectives, which may contradict the heterogeneous spatial demands of MLLMs, rendering any single layer fundamentally insufficient. To resolve this, we propose GeoAlign, a novel framework that dynamically aggregates multi-layer geometric features to realign with the actual demands. GeoAlign constructs a hierarchical geometric feature bank and leverages the MLLM's original visual tokens as content-aware queries to perform layer-wise sparse routing, adaptively fetching the suitable geometric features for each patch. Extensive experiments on VSI-Bench, ScanQA, and SQA3D demonstrate that our compact 4B model effectively achieves state-of-the-art performance, even outperforming larger existing MLLMs.
>
---
#### [new 002] ARGOS: Who, Where, and When in Agentic Multi-Camera Person Search
- **分类: cs.CV; cs.AI; cs.MA**

- **简介: 该论文提出ARGOS，解决多摄像头人员搜索中的交互推理问题，通过构建时空拓扑图进行语义、空间和时间推理，提升目标定位与追踪精度。**

- **链接: [https://arxiv.org/pdf/2604.12762](https://arxiv.org/pdf/2604.12762)**

> **作者:** Myungchul Kim; Kwanyong Park; Junmo Kim; In So Kweon
>
> **备注:** Accepted to CVPR 2026 Workshop on Multimodal Spatial Intelligence (MUSI)
>
> **摘要:** We introduce ARGOS, the first benchmark and framework that reformulates multi-camera person search as an interactive reasoning problem requiring an agent to plan, question, and eliminate candidates under information asymmetry. An ARGOS agent receives a vague witness statement and must decide what to ask, when to invoke spatial or temporal tools, and how to interpret ambiguous responses, all within a limited turn budget. Reasoning is grounded in a Spatio-Temporal Topology Graph (STTG) encoding camera connectivity and empirically validated transition times. The benchmark comprises 2,691 tasks across 14 real-world scenarios in three progressive tracks: semantic perception (Who), spatial reasoning (Where), and temporal reasoning (When). Experiments with four LLM backbones show the benchmark is far from solved (best TWS: 0.383 on Track 2, 0.590 on Track 3), and ablations confirm that removing domain-specific tools drops accuracy by up to 49.6 percentage points.
>
---
#### [new 003] Privacy-Preserving Structureless Visual Localization via Image Obfuscation
- **分类: cs.CV**

- **简介: 该论文属于视觉定位任务，旨在解决隐私泄露问题。通过图像模糊处理，在不改变结构的定位方法中实现隐私保护，保持定位精度。**

- **链接: [https://arxiv.org/pdf/2604.12068](https://arxiv.org/pdf/2604.12068)**

> **作者:** Vojtech Panek; Patrik Beliansky; Zuzana Kukelova; Torsten Sattler
>
> **摘要:** Visual localization is the task of estimating the camera pose of an image relative to a scene representation. In practice, visual localization systems are often cloud-based. Naturally, this raises privacy concerns in terms of revealing private details through the images sent to the server or through the representations stored on the server. Privacy-preserving localization aims to avoid such leakage of private details. However, the resulting localization approaches are significantly more complex, slower, and less accurate than their non-privacy-preserving counterparts. In this paper, we consider structureless localization methods in the context of privacy preservation. Structureless methods represent the scene through a set of reference images with known camera poses and intrinsics. In contrast to existing methods proposing representations that are as privacy-preserving as possible, we study a simple image obfuscation approach based on common image operations, e.g., replacing RGB images with (semantic) segmentations. We show that existing structureless pipelines do not need any special adjustments, as modern feature matchers can match obfuscated images out of the box. The results are easy-to-implement pipelines that can ensure both the privacy of the query images and the scene representations. Detailed experiments on multiple datasets show that the resulting methods achieve state-of-the-art pose accuracy for privacy-preserving approaches.
>
---
#### [new 004] OpenTME: An Open Dataset of AI-powered H&E Tumor Microenvironment Profiles from TCGA
- **分类: cs.CV; cs.AI; cs.LG; q-bio.QM**

- **简介: 该论文提出OpenTME数据集，用于AI驱动的H&E染色肿瘤微环境分析，解决大规模定量TME表征不足的问题。**

- **链接: [https://arxiv.org/pdf/2604.12075](https://arxiv.org/pdf/2604.12075)**

> **作者:** Maaike Galama; Nina Kozar-Gillan; Christina Embacher; Todd Dembo; Cornelius Böhm; Evelyn Ramberger; Julika Ribbat-Idel; Rosemarie Krupar; Verena Aumiller; Miriam Hägele; Kai Standvoss; Gerrit Erdmann; Blanca Pablos; Ari Angelo; Simon Schallenberg; Andrew Norgan; Viktor Matyas; Klaus-Robert Müller; Maximilian Alber; Lukas Ruff; Frederick Klauschen
>
> **摘要:** The tumor microenvironment (TME) plays a central role in cancer progression, treatment response, and patient outcomes, yet large-scale, consistent, and quantitative TME characterization from routine hematoxylin and eosin (H&E)-stained histopathology remains scarce. We introduce OpenTME, an open-access dataset of pre-computed TME profiles derived from 3,634 H&E-stained whole-slide images across five cancer types (bladder, breast, colorectal, liver, and lung cancer) from The Cancer Genome Atlas (TCGA). All outputs were generated using Atlas H&E-TME, an AI-powered application built on the Atlas family of pathology foundation models, which performs tissue quality control, tissue segmentation, cell detection and classification, and spatial neighborhood analysis, yielding over 4,500 quantitative readouts per slide at cell-level resolution. OpenTME is available for non-commercial academic research on Hugging Face. We will continue to expand OpenTME over time and anticipate it will serve as a resource for biomarker discovery, spatial biology research, and the development of computational methods for TME analysis.
>
---
#### [new 005] A Hybrid Architecture for Benign-Malignant Classification of Mammography ROIs
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决乳腺X光片中良性与恶性病灶的准确识别问题。通过结合CNN和状态空间模型，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2604.12437](https://arxiv.org/pdf/2604.12437)**

> **作者:** Mohammed Asad; Mohit Bajpai; Sudhir Singh; Rahul Katarya
>
> **备注:** 4 pages, 2 figures, 2 tables
>
> **摘要:** Accurate characterization of suspicious breast lesions in mammography is important for early diagnosis and treatment planning. While Convolutional Neural Networks (CNNs) are effective at extracting local visual patterns, they are less suited to modeling long-range dependencies. Vision Transformers (ViTs) address this limitation through self-attention, but their quadratic computational cost can be prohibitive. This paper presents a hybrid architecture that combines EfficientNetV2-M for local feature extraction with Vision Mamba, a State Space Model (SSM), for efficient global context modeling. The proposed model performs binary classification of abnormality-centered mammography regions of interest (ROIs) from the CBIS-DDSM dataset into benign and malignant classes. By combining a strong CNN backbone with a linear-complexity sequence model, the approach achieves strong lesion-level classification performance in an ROI-based setting.
>
---
#### [new 006] EigenCoin: sassanid coins classification based on Bhattacharyya distance
- **分类: cs.CV**

- **简介: 该论文属于模式识别任务，解决不平衡数据库下的萨珊银币分类问题。提出EigenCoin方法，结合Bhattacharyya距离，提升分类准确率并缓解过拟合。**

- **链接: [https://arxiv.org/pdf/2604.11932](https://arxiv.org/pdf/2604.11932)**

> **作者:** Rahele Allahverdi; Mohammad Mahdi Dehshibi; Azam Bastanfard; Daryoosh Akbarzadeh
>
> **备注:** 2nd World Conference on Information Technology (WCIT-2011)
>
> **摘要:** Solving pattern recognition problems using imbalanced databases is a hot topic, which entices researchers to bring it into focus. Therefore, we consider this problem in the application of Sassanid coins classification. Our focus is not only on proposing EigenCoin manifold with Bhattacharyya distance for the classification task, but also on testing the influence of the holistic and feature-based approaches. EigenCoin consists of three main steps namely manifold construction, mapping test data, and classification. Conducted experiments show EigenCoin outperformed other observed algorithms and achieved the accuracy from 9.45% up to 21.75%, while it has the capability of handling the over-fitting problem.
>
---
#### [new 007] A Workflow to Efficiently Generate Dense Tissue Ground Truth Masks for Digital Breast Tomosynthesis
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决DBT中密集组织标注数据不足的问题。通过提出一种高效生成分割掩码的框架，减少人工标注工作量。**

- **链接: [https://arxiv.org/pdf/2604.11927](https://arxiv.org/pdf/2604.11927)**

> **作者:** Tamerlan Mustafaev; Oleg Kruglov; Margarita Zuley; Luana de Mero Omena; Guilherme Muniz de Oliveira; Vitor de Sousa Franca; Bruno Barufaldi; Robert Nishikawa; Juhun Lee
>
> **摘要:** Digital breast tomosynthesis (DBT) is now the standard of care for breast cancer screening in the USA. Accurate segmentation of fibroglandular tissue in DBT images is essential for personalized risk estimation, but algorithm development is limited by scarce human-delineated training data. In this study we introduce a time- and labor-saving framework to generate a human-annotated binary segmentation mask for dense tissue in DBT. Our framework enables a user to outline a rough region of interest (ROI) enclosing dense tissue on the central reconstructed slice of a DBT volume and select a segmentation threshold to generate the dense tissue mask. The algorithm then projects the ROI to the remaining slices and iteratively adjusts slice-specific thresholds to maintain consistent dense tissue delineation across the DBT volume. By requiring annotation only on the central slice, the framework substantially reduces annotation time and labor. We used 44 DBT volumes from the DBTex dataset for evaluation. Inter-reader agreement was assessed by computing patient-wise Dice similarity coefficients between segmentation masks produced by two radiologists, yielding a median of 0.84. Accuracy of the proposed method was evaluated by having a radiologist manually segment the 20th and 80th percentile slices from each volume (CC and MLO views; 176 slices total) and calculate Dice scores between the manual and proposed segmentations, yielding a median of 0.83.
>
---
#### [new 008] M3D-Stereo: A Multiple-Medium and Multiple-Degradation Dataset for Stereo Image Restoration
- **分类: cs.CV**

- **简介: 该论文提出M3D-Stereo数据集，用于解决多介质、多退化类型的立体图像修复问题。通过构建包含多种退化场景的高质量数据集，提升图像修复方法的评估效果。**

- **链接: [https://arxiv.org/pdf/2604.12917](https://arxiv.org/pdf/2604.12917)**

> **作者:** Deqing Yang; Yingying Liu; Qicong Wang; Zhi Zeng; Dajiang Lu; Yibin Tian
>
> **摘要:** Image restoration under adverse conditions, such as underwater, haze or fog, and low-light environments, remains a highly challenging problem due to complex physical degradations and severe information loss. Existing datasets are predominantly limited to a single degradation type or heavily rely on synthetic data without stereo consistency, inherently restricting their applicability in real-world scenarios. To address this, we introduce M3D-Stereo, a stereo dataset with 7904 high-resolution image pairs for image restoration research acquired in multiple media with multiple controlled degradation levels. It encompasses four degradation scenarios: underwater scatter, haze/fog, underwater low-light, and haze low-light. Each scenario forms a subset, and is divided into six levels of progressive degradation, allowing fine-grained evaluations of restoration methods with increasing severity of degradation. Collected via a laboratory setup, the dataset provides aligned stereo image pairs along with their pixel-wise consistent clear ground truths. Two restoration tasks, single-level and mixed-level degradation, were performed to verify its validity. M3D-Stereo establishes a better controlled and more realistic benchmark to evaluate image restoration and stereo matching methods in complex degradation environments. It is made public under LGPLv3 license.
>
---
#### [new 009] VidTAG: Temporally Aligned Video to GPS Geolocalization with Denoising Sequence Prediction at a Global Scale
- **分类: cs.CV**

- **简介: 该论文提出VidTAG，解决视频精确定位问题，通过双编码器框架和时空对齐模块，实现视频到GPS的精准映射。**

- **链接: [https://arxiv.org/pdf/2604.12159](https://arxiv.org/pdf/2604.12159)**

> **作者:** Parth Parag Kulkarni; Rohit Gupta; Prakash Chandra Chhipa; Mubarak Shah
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** The task of video geolocalization aims to determine the precise GPS coordinates of a video's origin and map its trajectory; with applications in forensics, social media, and exploration. Existing classification-based approaches operate at a coarse city-level granularity and fail to capture fine-grained details, while image retrieval methods are impractical on a global scale due to the need for extensive image galleries which are infeasible to compile. Comparatively, constructing a gallery of GPS coordinates is straightforward and inexpensive. We propose VidTAG, a dual-encoder framework that performs frame-to-GPS retrieval using both self-supervised and language-aligned features. To address temporal inconsistencies in video predictions, we introduce the TempGeo module, which aligns frame embeddings, and the GeoRefiner module, an encoder-decoder architecture that refines GPS features using the aligned frame embeddings. Evaluations on Mapillary (MSLS) and GAMa datasets demonstrate our model's ability to generate temporally consistent trajectories and outperform baselines, achieving a 20% improvement at the 1 km threshold over GeoCLIP. We also beat current State-of-the-Art by 25% on global coarse grained video geolocalization (CityGuessr68k). Our approach enables fine-grained video geolocalization and lays a strong foundation for future research. More details on the project webpage: this https URL
>
---
#### [new 010] NTIRE 2026 The 3rd Restore Any Image Model (RAIM) Challenge: Professional Image Quality Assessment (Track 1)
- **分类: cs.CV; cs.AI**

- **简介: 该论文介绍NTIRE 2026挑战赛的Track 1，聚焦专业图像质量评估任务。旨在解决传统IQA方法无法区分高质量图像差异及缺乏解释能力的问题，通过MMLL模型提升图像质量比较与解释能力。**

- **链接: [https://arxiv.org/pdf/2604.12512](https://arxiv.org/pdf/2604.12512)**

> **作者:** Guanyi Qin; Jie Liang; Bingbing Zhang; Lishen Qu; Ya-nan Guan; Hui Zeng; Lei Zhang; Radu Timofte; Jianhui Sun; Xinli Yue; Tao Shao; Huan Hou; Wenjie Liao; Shuhao Han; Jieyu Yuan; Chunle Guo; Chongyi Li; Zewen Chen; Yunze Liu; Jian Guo; Juan Wang; Yun Zeng; Bing Li; Weiming Hu; Hesong Li; Dehua Liu; Xinjie Zhang; Qiang Li; Li Yan; Wei Dong; Qingsen Yan; Xingcan Li; Shenglong Zhou; Manjiang Yin; Yinxiang Zhang; Hongbo Wang; Jikai Xu; Zhaohui Fan; Dandan Zhu; Wei Sun; Weixia Zhang; Kun Zhu; Nana Zhang; Kaiwei Zhang; Qianqian Zhang; Zhihan Zhang; William Gordon; Linwei Wu; Jiachen Tu; Guoyi Xu; Yaoxin Jiang; Cici Liu; Yaokun Shi
>
> **备注:** NTIRE Challenge Report. Accepted by CVPRW 2026
>
> **摘要:** In this paper, we present an overview of the NTIRE 2026 challenge on the 3rd Restore Any Image Model in the Wild, specifically focusing on Track 1: Professional Image Quality Assessment. Conventional Image Quality Assessment (IQA) typically relies on scalar scores. By compressing complex visual characteristics into a single number, these methods fundamentally struggle to distinguish subtle differences among uniformly high-quality images. Furthermore, they fail to articulate why one image is superior, lacking the reasoning capabilities required to provide guidance for vision tasks. To bridge this gap, recent advancements in Multimodal Large Language Models (MLLMs) offer a promising paradigm. Inspired by this potential, our challenge establishes a novel benchmark exploring the ability of MLLMs to mimic human expert cognition in evaluating high-quality image pairs. Participants were tasked with overcoming critical bottlenecks in professional scenarios, centering on two primary objectives: (1) Comparative Quality Selection: reliably identifying the visually superior image within a high-quality pair; and (2) Interpretative Reasoning: generating grounded, expert-level explanations that detail the rationale behind the selection. In total, the challenge attracted nearly 200 registrations and over 2,500 submissions. The top-performing methods significantly advanced the state of the art in professional IQA. The challenge dataset is available at this https URL, and the official homepage is accessible at this https URL.
>
---
#### [new 011] SEATrack: Simple, Efficient, and Adaptive Multimodal Tracker
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SEATrack，解决多模态跟踪中性能与效率的矛盾。通过跨模态对齐和高效融合机制，提升跟踪效果同时保持参数效率。**

- **链接: [https://arxiv.org/pdf/2604.12502](https://arxiv.org/pdf/2604.12502)**

> **作者:** Junbin Su; Ziteng Xue; Shihui Zhang; Kun Chen; Weiming Hu; Zhipeng Zhang
>
> **备注:** Accepted as a CVPR 2026 Oral
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) in multimodal tracking reveals a concerning trend where recent performance gains are often achieved at the cost of inflated parameter budgets, which fundamentally erodes PEFT's efficiency promise. In this work, we introduce SEATrack, a Simple, Efficient, and Adaptive two-stream multimodal tracker that tackles this performance-efficiency dilemma from two complementary perspectives. We first prioritize cross-modal alignment of matching responses, an underexplored yet pivotal factor that we argue is essential for breaking the trade-off. Specifically, we observe that modality-specific biases in existing two-stream methods generate conflicting matching attention maps, thereby hindering effective joint representation learning. To mitigate this, we propose AMG-LoRA, which seamlessly integrates Low-Rank Adaptation (LoRA) for domain adaptation with Adaptive Mutual Guidance (AMG) to dynamically refine and align attention maps across modalities. We then depart from conventional local fusion approaches by introducing a Hierarchical Mixture of Experts (HMoE) that enables efficient global relation modeling, effectively balancing expressiveness and computational efficiency in cross-modal fusion. Equipped with these innovations, SEATrack advances notable progress over state-of-the-art methods in balancing performance with efficiency across RGB-T, RGB-D, and RGB-E tracking tasks. \href{this https URL}{\textcolor{cyan}{Code is available}}.
>
---
#### [new 012] Modality-Agnostic Prompt Learning for Multi-Modal Camouflaged Object Detection
- **分类: cs.CV**

- **简介: 该论文属于多模态伪装目标检测任务，旨在解决现有方法依赖特定模态架构、泛化能力差的问题。提出一种无模态感知的提示学习框架，提升检测性能与跨模态适应性。**

- **链接: [https://arxiv.org/pdf/2604.12380](https://arxiv.org/pdf/2604.12380)**

> **作者:** Hao Wang; Jiqing Zhang; Xin Yang; Baocai Yin; Lu Jiang; Zetian Mi; Huibing Wang
>
> **备注:** 10
>
> **摘要:** Camouflaged Object Detection (COD) aims to segment objects that blend seamlessly into complex backgrounds, with growing interest in exploiting additional visual modalities to enhance robustness through complementary information. However, most existing approaches generally rely on modality-specific architectures or customized fusion strategies, which limit scalability and cross-modal generalization. To address this, we propose a novel framework that generates modality-agnostic multi-modal prompts for the Segment Anything Model (SAM), enabling parameter-efficient adaptation to arbitrary auxiliary modalities and significantly improving overall performance on COD tasks. Specifically, we model multi-modal learning through interactions between a data-driven content domain and a knowledge-driven prompt domain, distilling task-relevant cues into unified prompts for SAM decoding. We further introduce a lightweight Mask Refine Module to calibrate coarse predictions by incorporating fine-grained prompt cues, leading to more accurate camouflaged object boundaries. Extensive experiments on RGB-Depth, RGB-Thermal, and RGB-Polarization benchmarks validate the effectiveness and generalization of our modality-agnostic framework.
>
---
#### [new 013] Fall Risk and Gait Analysis in Community-Dwelling Older Adults using World-Spaced 3D Human Mesh Recovery
- **分类: cs.CV**

- **简介: 该论文属于老年跌倒风险分析任务，旨在通过视频提取步态参数以评估跌倒风险。工作包括使用3D人体网格恢复模型分析TUG测试视频，验证步态参数与跌倒风险的关系。**

- **链接: [https://arxiv.org/pdf/2604.11961](https://arxiv.org/pdf/2604.11961)**

> **作者:** Chitra Banarjee; Patrick Kwon; Ania Lipat; Rui Xie; Chen Chen; Ladda Thiamwong
>
> **备注:** Work was accepted at Computer Vision for Biomechanics Workshop (CVBW) at CVPR 2026
>
> **摘要:** Gait assessment is a key clinical indicator of fall risk and overall health in older adults. However, standard clinical practice is largely limited to stopwatch-measured gait speed. We present a pipeline that leverages a 3D Human Mesh Recovery (HMR) model to extract gait parameters from recordings of older adults completing the Timed Up and Go (TUG) test. From videos recorded across different community centers, we extract and analyze spatiotemporal gait parameters, including step time, sit-to-stand duration, and step length. We found that video-derived step time was significantly correlated with IMU-based insole measurements. Using linear mixed effects models, we confirmed that shorter, more variable step lengths and longer sit-to-stand durations were predicted by higher self-rated fall risk and fear of falling. These findings demonstrate that our pipeline can enable accessible and ecologically valid gait analysis in community settings.
>
---
#### [new 014] Scaling In-Context Segmentation with Hierarchical Supervision
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，解决传统方法在高分辨率下计算效率低的问题。提出PatchICL框架，通过分层监督和选择性关注提升性能并减少计算量。**

- **链接: [https://arxiv.org/pdf/2604.12752](https://arxiv.org/pdf/2604.12752)**

> **作者:** T. Camaret Ndir; Marco Reisert; Robin T. Schirrmeister
>
> **摘要:** In-context learning (ICL) enables medical image segmentation models to adapt to new anatomical structures from limited examples, reducing the clinical annotation burden. However, standard ICL methods typically rely on dense, global cross-attention, which scales poorly with image resolution. While recent approaches have introduced localized attention mechanisms, they often lack explicit supervision on the selection process, leading to redundant computation in non-informative regions. We propose PatchICL, a hierarchical framework that combines selective image patching with multi-level supervision. Our approach learns to actively identify and attend only to the most informative anatomical regions. Compared to UniverSeg, a strong global-attention baseline, PatchICL achieves competitive in-domain CT segmentation accuracy while reducing compute by 44\% at $512\times512$ resolution. On 35 out-of-domain datasets spanning diverse imaging modalities, PatchICL outperforms the baseline on 6 of 13 modality categories, with particular strength on modalities dominated by localized pathology such as OCT and dermoscopy. Training and evaluation code are available at this https URL
>
---
#### [new 015] INST-Align: Implicit Neural Alignment for Spatial Transcriptomics via Canonical Expression Fields
- **分类: cs.CV**

- **简介: 该论文提出INST-Align，解决空间转录组多切片对齐与整合问题，通过联合优化变形和表达场实现精准对齐与重建。**

- **链接: [https://arxiv.org/pdf/2604.12084](https://arxiv.org/pdf/2604.12084)**

> **作者:** Bonian Han; Cong Qi; Przemyslaw Musialski; Zhi Wei
>
> **备注:** 10 pages, 2 figures, 3 tables. Submitted to MICCAI 2026
>
> **摘要:** Spatial transcriptomics (ST) measures mRNA expression while preserving spatial organization, but multi-slice analysis faces two coupled difficulties: large non-rigid deformations across slices and inter-slice batch effects when alignment and integration are treated independently. We present INST-Align, an unsupervised pairwise framework that couples a coordinate-based deformation network with a shared Canonical Expression Field, an implicit neural representation mapping spatial coordinates to expression embeddings, for joint alignment and reconstruction. A two-phase training strategy first establishes a stable canonical embedding space and then jointly optimizes deformation and spatial-feature matching, enabling mutually constrained alignment and representation learning. Cross-slice parameter sharing of the canonical field regularizes ambiguous correspondences and absorbs batch variation. Across nine datasets, INST-Align achieves state-of-the-art mean OT Accuracy (0.702), NN Accuracy (0.719), and Chamfer distance, with Chamfer reductions of up to 94.9\% on large-deformation sections relative to the strongest baseline. The framework also yields biologically meaningful spatial embeddings and coherent 3D tissue reconstruction. The code will be released after review phase.
>
---
#### [new 016] Detecting and refurbishing ground truth errors during training of deep learning-based echocardiography segmentation models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，解决深度学习模型训练中地面真实标签错误的问题。通过检测和修正错误标签，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.12832](https://arxiv.org/pdf/2604.12832)**

> **作者:** Iman Islam; Bram Ruijsink; Andrew J. Reader; Andrew P. King
>
> **备注:** 5 pages, 3 figures, 2 tables, International Symposium on Biomedical Imaging 2026
>
> **摘要:** Deep learning-based medical image segmentation typically relies on ground truth (GT) labels obtained through manual annotation, but these can be prone to random errors or systematic biases. This study examines the robustness of deep learning models to such errors in echocardiography (echo) segmentation and evaluates a novel strategy for detecting and refurbishing erroneous labels during model training. Using the CAMUS dataset, we simulate three error types, then compare a loss-based GT label error detection method with one based on Variance of Gradients (VOG). We also propose a pseudo-labelling approach to refurbish suspected erroneous GT labels. We assess the performance of our proposed approach under varying error levels. Results show that VOG proved highly effective in flagging erroneous GT labels during training. However, a standard U-Net maintained strong performance under random label errors and moderate levels of systematic errors (up to 50%). The detection and refurbishment approach improved performance, particularly under high-error conditions.
>
---
#### [new 017] Curvelet-Based Frequency-Aware Feature Enhancement for Deepfake Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于深度伪造检测任务，旨在解决压缩下检测性能下降的问题。提出基于Curvelet的特征增强方法，提升频率域特征质量，提高检测准确率。**

- **链接: [https://arxiv.org/pdf/2604.12028](https://arxiv.org/pdf/2604.12028)**

> **作者:** Salar Adel Sabri; Ramadhan J. Mstafa
>
> **备注:** 10 Pages, 6 Figures, 2 Tables
>
> **摘要:** The proliferation of sophisticated generative models has significantly advanced the realism of synthetic facial content, known as deepfakes, raising serious concerns about digital trust. Although modern deep learning-based detectors perform well, many rely on spatial-domain features that degrade under compression. This limitation has prompted a shift toward integrating frequency-domain representations with deep learning to improve robustness. Prior research has explored frequency transforms such as Discrete Cosine Transform (DCT), Fast Fourier Transform (FFT), and Wavelet Transform, among others. However, to the best of our knowledge, the Curvelet Transform, despite its superior directional and multiscale properties, remains entirely unexplored in the context of deepfake detection. In this work, we introduce a novel Curvelet-based detection approach that enhances feature quality through wedge-level attention and scale-aware spatial masking, both trained to selectively emphasize discriminative frequency components. The refined frequency cues are reconstructed and passed to a modified pretrained Xception network for classification. Evaluated on two compression qualities in the challenging FaceForensics++ dataset, our method achieves 98.48% accuracy and 99.96% AUC on FF++ low compression, while maintaining strong performance under high compression, demonstrating the efficacy and interpretability of Curvelet-informed forgery detection.
>
---
#### [new 018] RSGMamba: Reliability-Aware Self-Gated State Space Model for Multimodal Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于多模态语义分割任务，解决传统方法忽略模态可靠性导致的特征退化问题，提出RSGMamba框架，通过自门控机制提升分割性能。**

- **链接: [https://arxiv.org/pdf/2604.12319](https://arxiv.org/pdf/2604.12319)**

> **作者:** Guoan Xu; Yang Xiao; Guangwei Gao; Dongchen Zhu; Wenjing Jia; Guo-Jun Qi
>
> **备注:** 7tables,9 figures
>
> **摘要:** Multimodal semantic segmentation has emerged as a powerful paradigm for enhancing scene understanding by leveraging complementary information from multiple sensing modalities (e.g., RGB, depth, and thermal). However, existing cross-modal fusion methods often implicitly assume that all modalities are equally reliable, which can lead to feature degradation when auxiliary modalities are noisy, misaligned, or incomplete. In this paper, we revisit cross-modal fusion from the perspective of modality reliability and propose a novel framework termed the Reliability-aware Self-Gated State Space Model (RSGMamba). At the core of our method is the Reliability-aware Self-Gated Mamba Block (RSGMB), which explicitly models modality reliability and dynamically regulates cross-modal interactions through a self-gating mechanism. Unlike conventional fusion strategies that indiscriminately exchange information across modalities, RSGMB enables reliability-aware feature selection and enhancing informative feature aggregation. In addition, a lightweight Local Cross-Gated Modulation (LCGM) is incorporated to refine fine-grained spatial details, complementing the global modeling capability of RSGMB. Extensive experiments demonstrate that RSGMamba achieves state-of-the-art performance on both RGB-D and RGB-T semantic segmentation benchmarks, resulting 58.8% / 54.0% mIoU on NYUDepth V2 and SUN-RGBD (+0.4% / +0.7% over prior best), and 61.1% / 88.9% mIoU on MFNet and PST900 (up to +1.6%), with only 48.6M parameters, thereby validating the effectiveness and superiority of the proposed approach.
>
---
#### [new 019] OmniFood8K: Single-Image Nutrition Estimation via Hierarchical Frequency-Aligned Fusion
- **分类: cs.CV**

- **简介: 该论文属于食品营养估计任务，旨在解决中文菜肴数据不足及依赖深度传感器的问题。提出OmniFood8K数据集和NutritionSynth-115K合成数据集，并设计端到端框架提升预测精度。**

- **链接: [https://arxiv.org/pdf/2604.12356](https://arxiv.org/pdf/2604.12356)**

> **作者:** Dongjian Yu; Weiqing Min; Qian Jiang; Xing Lin; Xin Jin; Shuqiang Jiang
>
> **备注:** Accepted by CVPR 2026 (Highlight Paper)
>
> **摘要:** Accurate estimation of food nutrition plays a vital role in promoting healthy dietary habits and personalized diet management. Most existing food datasets primarily focus on Western cuisines and lack sufficient coverage of Chinese dishes, which restricts accurate nutritional estimation for Chinese meals. Moreover, many state-of-the-art nutrition prediction methods rely on depth sensors, restricting their applicability in daily scenarios. To address these limitations, we introduce OmniFood8K, a comprehensive multimodal dataset comprising 8,036 food samples, each with detailed nutritional annotations and multi-view images. In addition, to enhance models' capability in nutritional prediction, we construct NutritionSynth-115K, a large-scale synthetic dataset that introduces compositional variations while preserving precise nutritional labels. Moreover, we propose an end-to-end framework for nutritional prediction from a single RGB image. First, we predict a depth map from a single RGB image and design the Scale-Shift Residual Adapter (SSRA) to refine it for global scale consistency and local structural preservation. Second, we propose the Frequency-Aligned Fusion Module (FAFM) to hierarchically align and fuse RGB and depth features in the frequency domain. Finally, we design a Mask-based Prediction Head (MPH) to emphasize key ingredient regions via dynamic channel selection for more accurate prediction. Extensive experiments on multiple datasets demonstrate the superiority of our method over existing approaches. Project homepage: this https URL
>
---
#### [new 020] Challenging Vision-Language Models with Physically Deployable Multimodal Semantic Lighting Attacks
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型安全研究，旨在解决物理世界中VLM的脆弱性问题。提出MSLA攻击框架，通过可控光照干扰多模态语义理解，验证VLM在真实环境中的安全风险。**

- **链接: [https://arxiv.org/pdf/2604.12833](https://arxiv.org/pdf/2604.12833)**

> **作者:** Yingying Zhao; Chengyin Hu; Qike Zhang; Xin Li; Xin Wang; Yiwei Wei; Jiujiang Guo; Jiahuan Long; Tingsong Jiang; Wen Yao
>
> **摘要:** Vision-Language Models (VLMs) have shown remarkable performance, yet their security remains insufficiently understood. Existing adversarial studies focus almost exclusively on the digital setting, leaving physical-world threats largely unexplored. As VLMs are increasingly deployed in real environments, this gap becomes critical, since adversarial perturbations must be physically realizable. Despite this practical relevance, physical attacks against VLMs have not been systematically studied. Such attacks may induce recognition failures and further disrupt multimodal reasoning, leading to severe semantic misinterpretation in downstream tasks. Therefore, investigating physical attacks on VLMs is essential for assessing their real-world security risks. To address this gap, we propose Multimodal Semantic Lighting Attacks (MSLA), the first physically deployable adversarial attack framework against VLMs. MSLA uses controllable adversarial lighting to disrupt multimodal semantic understanding in real scenes, attacking semantic alignment rather than only task-specific outputs. Consequently, it degrades zero-shot classification performance of mainstream CLIP variants while inducing severe semantic hallucinations in advanced VLMs such as LLaVA and BLIP across image captioning and visual question answering (VQA). Extensive experiments in both digital and physical domains demonstrate that MSLA is effective, transferable, and practically realizable. Our findings provide the first evidence that VLMs are highly vulnerable to physically deployable semantic attacks, exposing a previously overlooked robustness gap and underscoring the urgent need for physical-world robustness evaluation of VLMs.
>
---
#### [new 021] Rethinking Satellite Image Restoration for Onboard AI: A Lightweight Learning-Based Approach
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于卫星图像恢复任务，旨在解决传统方法计算复杂、速度慢的问题。通过提出轻量级CNN模型ConvBEERS，实现高效高质量的图像恢复，并验证其在星载AI中的可行性。**

- **链接: [https://arxiv.org/pdf/2604.12807](https://arxiv.org/pdf/2604.12807)**

> **作者:** Adrien Dorise; Marjorie Bellizzi; Omar Hlimi
>
> **备注:** AI4SPACE@CVPR conference
>
> **摘要:** Satellite image restoration aims to improve image quality by compensating for degradations (e.g., noise and blur) introduced by the imaging system and acquisition conditions. As a fundamental preprocessing step, restoration directly impacts both ground-based product generation and emerging onboard AI applications. Traditional restoration pipelines based on sequential physical models are computationally intensive and slow, making them unsuitable for onboard environments. In this paper, we introduce ConvBEERS: a Convolutional Board-ready Embedded and Efficient Restoration model for Space to investigate whether a light and non-generative residual convolutional network, trained on simulated satellite data, can match or surpass a traditional ground-processing restoration pipeline across multiple operating conditions. Experiments conducted on simulated datasets and real Pleiades-HR imagery demonstrate that the proposed approach achieves competitive image quality, with a +6.9dB PSNR improvement. Evaluation on a downstream object detection task demonstrates that restoration significantly improves performance, with up to +5.1% mAP@50. In addition, successful deployment on a Xilinx Versal VCK190 FPGA validates its practical feasibility for satellite onboard processing, with a ~41x reduction in latency compared to the traditional pipeline. These results demonstrate the relevance of using lightweight CNNs to achieve competitive restoration quality while addressing real-world constraints in spaceborne systems.
>
---
#### [new 022] AffectAgent: Collaborative Multi-Agent Reasoning for Retrieval-Augmented Multimodal Emotion Recognition
- **分类: cs.CV**

- **简介: 该论文属于多模态情感识别任务，解决单一检索增强生成易受模态歧义影响的问题。提出AffectAgent框架，通过多智能体协作提升情感理解精度。**

- **链接: [https://arxiv.org/pdf/2604.12735](https://arxiv.org/pdf/2604.12735)**

> **作者:** Zeheng Wang; Zitong Yu; Yijie Zhu; Bo Zhao; Haochen Liang; Taorui Wang; Wei Xia; Jiayu Zhang; Zhishu Liu; Hui Ma; Fei Ma; Qi Tian
>
> **摘要:** LLM-based multimodal emotion recognition relies on static parametric memory and often hallucinates when interpreting nuanced affective states. In this paper, given that single-round retrieval-augmented generation is highly susceptible to modal ambiguity and therefore struggles to capture complex affective dependencies across modalities, we introduce AffectAgent, an affect-oriented multi-agent retrieval-augmented generation framework that leverages collaborative decision-making among agents for fine-grained affective understanding. Specifically, AffectAgent comprises three jointly optimized specialized agents, namely a query planner, an evidence filter, and an emotion generator, which collaboratively perform analytical reasoning to retrieve cross-modal samples, assess evidence, and generate predictions. These agents are optimized end-to-end using Multi-Agent Proximal Policy Optimization (MAPPO) with a shared affective reward to ensure consistent emotion understanding. Furthermore, we introduce Modality-Balancing Mixture of Experts (MB-MoE) and Retrieval-Augmented Adaptive Fusion (RAAF), where MB-MoE dynamically regulates the contributions of different modalities to mitigate representation mismatch caused by cross-modal heterogeneity, while RAAF enhances semantic completion under missing-modality conditions by incorporating retrieved audiovisual embeddings. Extensive experiments on MER-UniBench demonstrate that AffectAgent achieves superior performance across complex scenarios. Our code will be released at: this https URL.
>
---
#### [new 023] A Multi-Agent Feedback System for Detecting and Describing News Events in Satellite Imagery
- **分类: cs.CV; cs.MA**

- **简介: 该论文属于多时相事件描述任务，旨在解决卫星影像中多时相事件数据缺失的问题。通过构建多智能体系统，自动检测并生成新闻事件的多时相描述。**

- **链接: [https://arxiv.org/pdf/2604.12772](https://arxiv.org/pdf/2604.12772)**

> **作者:** Madeline Anderson; Mikhail Klassen; Ash Hoover; Kerri Cahoy
>
> **摘要:** Changes in satellite imagery often occur over multiple time steps. Despite the emergence of bi-temporal change captioning datasets, there is a lack of multi-temporal event captioning datasets (at least two images per sequence) in remote sensing. This gap exists because (1) searching for visible events in satellite imagery and (2) labeling multi-temporal sequences require significant time and labor. To address these challenges, we present SkyScraper, an iterative multi-agent workflow that geocodes news articles and synthesizes captions for corresponding satellite image sequences. Our experiments show that SkyScraper successfully finds 5x more events than traditional geocoding methods, demonstrating that agentic feedback is an effective strategy for surfacing new multi-temporal events in satellite imagery. We apply our framework to a large database of global news articles, curating a new multi-temporal captioning dataset with 5,000 sequences. By automatically identifying imagery related to news events, our work also supports journalism and reporting efforts.
>
---
#### [new 024] IAD-Unify: A Region-Grounded Unified Model for Industrial Anomaly Segmentation, Understanding, and Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出IAD-Unify，解决工业异常分割、理解与生成的统一任务，通过区域引导框架提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.12440](https://arxiv.org/pdf/2604.12440)**

> **作者:** Haoyu Zheng; Tianwei Lin; Wei Wang; Zhuonan Wang; Wenqiao Zhang; Jiaqi Zhu; Feifei Shao
>
> **摘要:** Real-world industrial inspection requires not only localizing defects, but also explaining them in natural language and generating controlled defect edits. However, existing approaches fail to jointly support all three capabilities within a unified framework and evaluation protocol. We propose IAD-Unify, a dual-encoder unified framework in which a frozen DINOv2-based region expert supplies precise anomaly evidence to a shared Qwen3.5-4B vision-language backbone via lightweight token injection, jointly enabling anomaly segmentation, region-grounded understanding, and mask-guided generation. To enable unified evaluation, we further construct Anomaly-56K, a comprehensive unified multi-task IAD evaluation platform, spanning 59,916 images across 24 categories and 104 defect variants. Controlled ablations yield four findings: (i) region grounding is the decisive mechanism for understanding, removing it degrades location accuracy by >76 pp; (ii) predicted-region performance closely matches oracle, confirming deployment viability; (iii) region-grounded generation achieves the best full-image fidelity and masked-region perceptual quality; and (iv) pre-initialized joint training improves understanding at negligible generation cost (-0.16 dB). IAD-Unify further achieves strong performance on the MMAD benchmark, including categories unseen during training, demonstrating robust cross-category generalization.
>
---
#### [new 025] Why and When Visual Token Pruning Fails? A Study on Relevant Visual Information Shift in MLLMs Decoding
- **分类: cs.CV**

- **简介: 该论文属于多模态大模型任务，解决视觉token剪枝在复杂推理中失效的问题。通过分析RVIS现象，提出DSTP框架提升剪枝效果。**

- **链接: [https://arxiv.org/pdf/2604.12358](https://arxiv.org/pdf/2604.12358)**

> **作者:** Jiwan Kim; Kibum Kim; Wonjoong Kim; Byung-Kwan Lee; Chanyoung Park
>
> **备注:** Preprint, Project : this https URL
>
> **摘要:** Recently, visual token pruning has been studied to handle the vast number of visual tokens in Multimodal Large Language Models. However, we observe that while existing pruning methods perform reliably on simple visual understanding, they struggle to effectively generalize to complex visual reasoning tasks, a critical gap underexplored in previous studies. Through a systematic analysis, we identify Relevant Visual Information Shift (RVIS) during decoding as the primary failure driver. To address this, we propose Decoding-stage Shift-aware Token Pruning (DSTP), a training-free add-on framework that enables existing pruning methods to align visual tokens with shifting reasoning requirements during the decoding stage. Extensive experiments demonstrate that DSTP significantly mitigates performance degradation of pruning methods in complex reasoning tasks, while consistently yielding performance gains even across visual understanding benchmarks. Furthermore, DSTP demonstrates effectiveness across diverse state-of-the-art architectures, highlighting its generalizability and efficiency with minimal computational overhead.
>
---
#### [new 026] DeferredSeg: A Multi-Expert Deferral Framework for Trustworthy Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决模型置信度不可靠的问题。提出DeferredSeg框架，通过人机协作实现可信分割，提升准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2604.12411](https://arxiv.org/pdf/2604.12411)**

> **作者:** Qiuyu Tian; Haoliang Sun; Yunshan Wang; Yinghuan Shi; Yilong Yin
>
> **备注:** 27 pages,6 figures
>
> **摘要:** Segmentation models based on deep neural networks demonstrate strong generalization for medical image segmentation. However, they often exhibit overconfidence or underconfidence, leading to unreliable confidence scores for segmentation masks, especially in ambiguous regions. This undermines the trustworthiness required for clinical deployment. Motivated by the learning-to-defer (L2D) paradigm, we introduce DeferredSeg, a deferral-aware segmentation framework, i.e., a Human--AI collaboration system that determines whether to defer predictions to human experts in specific regions. DeferredSeg extends the base segmentor with an aggregated deferral predictor and additional routing channels that dynamically route each pixel to either the base segmentor or a human expert. To train this routing efficiently, we introduce a pixel-wise surrogate collaboration loss that supervises deferral decisions. In addition, to preserve spatial coherence within deferral regions, we propose a spatial-coherence loss that enforces smooth deferral masks, thereby enhancing reliability. Beyond single-expert deferral, we further extend the framework to a multi-expert setting by introducing multiple discrepancy experts for collaborative decision-making. To prevent overloading or underutilizing individual experts, we further design a load-balancing penalty that evenly distributes workload across expert branches. We evaluate DeferredSeg on three challenging medical datasets using MedSAM and CENet as the base segmentor for fair comparison. Experimental results show that DeferredSeg consistently outperforms the baseline, demonstrating its effectiveness for trustworthy dense medical segmentation. Moreover, the proposed framework is model-agnostic and can be readily applied to other segmentation architectures.
>
---
#### [new 027] Nucleus-Image: Sparse MoE for Image Generation
- **分类: cs.CV**

- **简介: 该论文提出Nucleus-Image，一种高效文本到图像生成模型，通过稀疏专家混合（MoE）架构在保持高质量的同时降低计算成本。**

- **链接: [https://arxiv.org/pdf/2604.12163](https://arxiv.org/pdf/2604.12163)**

> **作者:** Chandan Akiti; Ajay Modukuri; Murali Nandan Nagarapu; Gunavardhan Akiti; Haozhe Liu
>
> **摘要:** We present Nucleus-Image, a text-to-image generation model that establishes a new Pareto frontier in quality-versus-efficiency by matching or exceeding leading models on GenEval, DPG-Bench, and OneIG-Bench while activating only approximately 2B parameters per forward pass. Nucleus-Image employs a sparse mixture-of-experts (MoE) diffusion transformer architecture with Expert-Choice Routing that scales total model capacity to 17B parameters across 64 routed experts per layer. We adopt a streamlined architecture optimized for inference efficiency by excluding text tokens from the transformer backbone entirely and using joint attention that enables text KV sharing across timesteps. To improve routing stability when using timestep modulation, we introduce a decoupled routing design that separates timestep-aware expert assignment from timestep-conditioned expert computation. We construct a large-scale training corpus of 1.5B high-quality training pairs spanning 700M unique images through multi-stage filtering, deduplication, aesthetic tiering, and caption curation. Training follows a progressive resolution curriculum (256 to 512 to 1024) with multi-aspect-ratio bucketing at every stage, coupled with progressive sparsification of the expert capacity factor. We adopt the Muon optimizer and share our parameter grouping recipe tailored for diffusion models with timestep modulation. Nucleus-Image demonstrates that sparse MoE scaling is a highly effective path to high-quality image generation, reaching the performance of models with significantly larger active parameter budgets at a fraction of the inference cost. These results are achieved without post-training optimization of any kind: no reinforcement learning, no direct preference optimization, and no human preference tuning. We release the training recipe, making Nucleus-Image the first fully open-source MoE diffusion model at this quality.
>
---
#### [new 028] GTPBD-MM: A Global Terraced Parcel and Boundary Dataset with Multi-Modality
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于农业地块提取任务，解决复杂梯田区域的地块分割问题。构建了首个多模态基准GTPBD-MM，并提出ETTerra模型，融合图像、文本和地形数据提升分割精度。**

- **链接: [https://arxiv.org/pdf/2604.12315](https://arxiv.org/pdf/2604.12315)**

> **作者:** Zhiwei Zhang; Xingyuan Zeng; Xinkai Kong; Kunquan Zhang; Haoyuan Liang; Bohan Shi; Juepeng Zheng; Jianxi Huang; Yutong Lu; Haohuan Fu
>
> **备注:** 15 pages, 11 figures. Submitted to ACM Multimedia 2026 Dataset Track
>
> **摘要:** Agricultural parcel extraction plays an important role in remote sensing-based agricultural monitoring, supporting parcel surveying, precision management, and ecological assessment. However, existing public benchmarks mainly focus on regular and relatively flat farmland scenes. In contrast, terraced parcels in mountainous regions exhibit stepped terrain, pronounced elevation variation, irregular boundaries, and strong cross-regional heterogeneity, making parcel extraction a more challenging problem that jointly requires visual recognition, semantic discrimination, and terrain-aware geometric understanding. Although recent studies have advanced visual parcel benchmarks and image-text farmland understanding, a unified benchmark for complex terraced parcel extraction under aligned image-text-DEM settings remains absent. To fill this gap, we present GTPBD-MM, the first multimodal benchmark for global terraced parcel extraction. Built upon GTPBD, GTPBD-MM integrates high-resolution optical imagery, structured text descriptions, and DEM data, and supports systematic evaluation under Image-only, Image+Text, and Image+Text+DEM settings. We further propose Elevation and Text guided Terraced parcel network (ETTerra), a multimodal baseline for terraced parcel delineation. Extensive experiments demonstrate that textual semantics and terrain geometry provide complementary cues beyond visual appearance alone, yielding more accurate, coherent, and structurally consistent delineation results in complex terraced scenes.
>
---
#### [new 029] Risk-Calibrated Learning: Minimizing Fatal Errors in Medical AI
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决深度学习模型中的致命错误问题。通过引入风险校准损失，有效降低关键错误率，提升模型安全性。**

- **链接: [https://arxiv.org/pdf/2604.12693](https://arxiv.org/pdf/2604.12693)**

> **作者:** Abolfazl Mohammadi-Seif; Ricardo Baeza-Yates
>
> **备注:** This work has been accepted for publication in the Proceedings of the 2026 International Joint Conference on Neural Networks (IJCNN 2026). The final published version should be cited
>
> **摘要:** Deep learning models often achieve expert-level accuracy in medical image classification but suffer from a critical flaw: semantic incoherence. These high-confidence mistakes that are semantically incoherent (e.g., classifying a malignant tumor as benign) fundamentally differ from acceptable errors which stem from visual ambiguity. Unlike safe, fine-grained disagreements, these fatal failures erode clinical trust. To address this, we propose Risk-Calibrated Learning, a technique that explicitly distinguishes between visual ambiguity (fine-grained errors) and catastrophic structural errors. By embedding a confusion-aware clinical severity matrix M into the optimization landscape, our method suppresses critical errors (false negatives) without requiring complex architectural changes. We validate our approach in four different imaging modalities: Brain Tumor MRI, ISIC 2018 (Dermoscopy), BreaKHis (Breast Histopathology), and SICAPv2 (Prostate Histopathology). Extensive experiments demonstrate that our Risk-Calibrated Loss consistently reduces the Critical Error Rate (CER) for all four datasets, achieving relative safety improvements ranging from 20.0% (on breast histopathology) to 92.4% (on prostate histopathology) compared to state-of-the-art baselines such as Focal Loss. These results confirm that our method offers a superior safety-accuracy trade-off across both CNN and Transformer architectures.
>
---
#### [new 030] INDOTABVQA: A Benchmark for Cross-Lingual Table Understanding in Bahasa Indonesia Documents
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出INDOTABVQA基准，用于评估跨语言表格视觉问答任务。针对多语言文档中的表格理解问题，构建了包含多种视觉风格的文档图像和多语言问答对的数据集，并验证了模型在不同语言和结构复杂度下的表现。**

- **链接: [https://arxiv.org/pdf/2604.11970](https://arxiv.org/pdf/2604.11970)**

> **作者:** Somraj Gautam; Anathapindika Dravichi; Gaurav Harit
>
> **备注:** Accepted in ACL 2026 (Findings)
>
> **摘要:** We introduce INDOTABVQA, a benchmark for evaluating cross-lingual Table Visual Question Answering (VQA) on real-world document images in Bahasa Indonesia. The dataset comprises 1,593 document images across three visual styles (bordered, borderless, and colorful) with one or more than one tables, and 1,593 question-answer sets in four languages: Bahasa Indonesia, English, Hindi, and Arabic. This enables evaluation of Vision-Language Models (VLMs) in both monolingual (Bahasa documents with Bahasa questions) and cross-lingual settings (Bahasa documents with questions in other languages). We benchmark leading open-source VLMs (Qwen2.5-VL, Gemma-3, LLaMA-3.2) and GPT-4o and reveal substantial performance gaps, particularly on structurally complex tables and in low-resource languages. Fine-tuning a compact 3B and LoRA-finetuned 7B model on our dataset yields 11.6% and 17.8% improvements in accuracy. Providing explicit table region coordinates as additional input further improves performance by 4-7%, demonstrating the value of Spatial priors for table-based reasoning. Our findings underscore the importance of language-diverse, domain-specific datasets and demonstrate that targeted fine-tuning can significantly enhance VLM performance on specialized document understanding tasks. INDOTABVQA provides a valuable resource for advancing research in cross-lingual, structure-aware document understanding, especially in underrepresented regions of the world. Full dataset can be accessed in huggingface at: this https URL}
>
---
#### [new 031] Direct Discrepancy Replay: Distribution-Discrepancy Condensation and Manifold-Consistent Replay for Continual Face Forgery Detection
- **分类: cs.CV**

- **简介: 该论文属于持续人脸伪造检测任务，旨在解决模型遗忘旧伪造模式的问题。提出DDC和MCR方法，在小内存下有效重建旧任务分布，减少遗忘并降低身份泄露风险。**

- **链接: [https://arxiv.org/pdf/2604.12941](https://arxiv.org/pdf/2604.12941)**

> **作者:** Tianshuo Zhang; Haoyuan Zhang; Siran Peng; Weisong Zhao; Xiangyu Zhu; Zhen Lei
>
> **摘要:** Continual face forgery detection (CFFD) requires detectors to learn emerging forgery paradigms without forgetting previously seen manipulations. Existing CFFD methods commonly rely on replaying a small amount of past data to mitigate forgetting. Such replay is typically implemented either by storing a few historical samples or by synthesizing pseudo-forgeries from detector-dependent perturbations. Under strict memory budgets, the former cannot adequately cover diverse forgery cues and may expose facial identities, while the latter remains strongly tied to past decision boundaries. We argue that the core role of replay in CFFD is to reinstate the distributions of previous forgery tasks during subsequent training. To this end, we directly condense the discrepancy between real and fake distributions and leverage real faces from the current stage to perform distribution-level replay. Specifically, we introduce Distribution-Discrepancy Condensation (DDC), which models the real-to-fake discrepancy via a surrogate factorization in characteristic-function space and condenses it into a tiny bank of distribution discrepancy maps. We further propose Manifold-Consistent Replay (MCR), which synthesizes replay samples through variance-preserving composition of these maps with current-stage real faces, yielding samples that reflect previous-task forgery cues while remaining compatible with current real-face statistics. Operating under an extremely small memory budget and without directly storing raw historical face images, our framework consistently outperforms prior CFFD baselines and significantly mitigates catastrophic forgetting. Replay-level privacy analysis further suggests reduced identity leakage risk relative to selection-based replay.
>
---
#### [new 032] ELoG-GS: Dual-Branch Gaussian Splatting with Luminance-Guided Enhancement for Extreme Low-light 3D Reconstruction
- **分类: cs.CV**

- **简介: 该论文针对极端低光下的3D重建任务，提出ELoG-GS方法，通过点云初始化和亮度增强提升重建质量。**

- **链接: [https://arxiv.org/pdf/2604.12592](https://arxiv.org/pdf/2604.12592)**

> **作者:** Yuhao Liu; Dingju Wang; Ziyang Zheng
>
> **备注:** Our method achieved a ranking of 9 out of 148 participants in Track 1 of the NTIRE 3DRR Challenge, as reported on the official competition website: this https URL
>
> **摘要:** This paper presents our approach to the NTIRE 2026 3D Restoration and Reconstruction Challenge (Track 1), which focuses on reconstructing high-quality 3D representations from degraded multi-view inputs. The challenge involves recovering geometrically consistent and photorealistic 3D scenes in extreme low-light environments. To address this task, we propose Extreme Low-light Optimized Gaussian Splatting (ELoG-GS), a robust low-light 3D reconstruction pipeline that integrates learning-based point cloud initialization and luminance-guided color enhancement for stable and photorealistic Gaussian Splatting. Our method incorporates both geometry-aware initialization and photometric adaptation strategies to improve reconstruction fidelity under challenging conditions. Extensive experiments on the NTIRE Track 1 benchmark demonstrate that our approach significantly improves reconstruction quality over the baselines, achieving superior visual fidelity and geometric consistency. The proposed method provides a practical solution for robust 3D reconstruction in real-world degraded scenarios. In the final testing phase, our method achieved a PSNR of 18.6626 and an SSIM of 0.6855 on the official platform leaderboard. Code is available at this https URL.
>
---
#### [new 033] EgoEsportsQA: An Egocentric Video Benchmark for Perception and Reasoning in Esports
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文提出EgoEsportsQA，用于评估视频大模型在电子竞技中的感知与推理能力。针对现有模型在高动态虚拟环境中的表现不足，构建了包含1745个高质量问答对的数据集，揭示模型在战术推理和微观操作上的短板。**

- **链接: [https://arxiv.org/pdf/2604.12320](https://arxiv.org/pdf/2604.12320)**

> **作者:** Jianzhe Ma; Zhonghao Cao; Shangkui Chen; Yichen Xu; Wenxuan Wang; Qin Jin
>
> **备注:** Work in progress
>
> **摘要:** While video large language models (Video-LLMs) excel in understanding slow-paced, real-world egocentric videos, their capabilities in high-velocity, information-dense virtual environments remain under-explored. Existing benchmarks focus on daily activities, yet lack a rigorous testbed for evaluating fast, rule-bound reasoning in virtual scenarios. To fill this gap, we introduce EgoEsportsQA, a pioneering video question-answering (QA) benchmark for grounding perception and reasoning in expert esports knowledge. We curate 1,745 high-quality QA pairs from professional matches across 3 first-person shooter games via a scalable six-stage pipeline. These questions are structured into a two-dimensional decoupled taxonomy: 11 sub-tasks in the cognitive capability dimension (covering perception and reasoning levels) and 6 sub-tasks in the esports knowledge dimension. Comprehensive evaluations of state-of-the-art Video-LLMs reveal that current models still fail to achieve satisfactory performance, with the best model only 71.58%. The results expose notable gaps across both axes: models exhibit stronger capabilities in basic visual perception than in deep tactical reasoning, and they grasp overall macro-progression better than fine-grained micro-operations. Extensive ablation experiments demonstrate the intrinsic weaknesses of current Video-LLM architectures. Further analysis suggests that our dataset not only reveals the connections between real-world and virtual egocentric domains, but also offers guidance for optimizing downstream esports applications, thereby fostering the future advancement of Video-LLMs in various egocentric environments.
>
---
#### [new 034] Generative Anonymization in Event Streams
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于隐私保护任务，解决事件流中身份泄露问题。通过生成式匿名化框架，在保持数据结构完整性的同时，防止身份被重建。**

- **链接: [https://arxiv.org/pdf/2604.12803](https://arxiv.org/pdf/2604.12803)**

> **作者:** Adam T. Müller; Mihai Kocsis; Nicolaj C. Stache
>
> **备注:** Accepted to the 1st Workshop on Low-Level Vision Frontiers (LoViF) at IEEE/CVF CVPR 2026
>
> **摘要:** Neuromorphic vision sensors offer low latency and high dynamic range, but their deployment in public spaces raises severe data protection concerns. Recent Event-to-Video (E2V) models can reconstruct high-fidelity intensity images from sparse event streams, inadvertently exposing human identities. Current obfuscation methods, such as masking or scrambling, corrupt the spatio-temporal structure, severely degrading data utility for downstream perception tasks. In this paper, to the best of our knowledge, we present the first generative anonymization framework for event streams to resolve this utility-privacy trade-off. By bridging the modality gap between asynchronous events and standard spatial generative models, our pipeline projects events into an intermediate intensity representation, leverages pretrained models to synthesize realistic, non-existent identities, and re-encodes the features back into the neuromorphic domain. Experiments demonstrate that our method reliably prevents identity recovery from E2V reconstructions while preserving the structural data integrity required for downstream vision tasks. Finally, to facilitate rigorous evaluation, we introduce a novel, synchronized real-world event and RGB dataset captured via precise robotic trajectories, providing a robust benchmark for future research in privacy-preserving neuromorphic vision.
>
---
#### [new 035] Grasp in Gaussians: Fast Monocular Reconstruction of Dynamic Hand-Object Interactions
- **分类: cs.CV**

- **简介: 该论文提出GraG方法，用于从单目视频中快速重建动态手物交互。解决的是3D手物交互重建任务，通过高效跟踪和轻量表示提升速度与精度。**

- **链接: [https://arxiv.org/pdf/2604.12929](https://arxiv.org/pdf/2604.12929)**

> **作者:** Ayce Idil Aytekin; Xu Chen; Zhengyang Shen; Thabo Beeler; Helge Rhodin; Rishabh Dabral; Christian Theobalt
>
> **备注:** Project page: this https URL
>
> **摘要:** We present Grasp in Gaussians (GraG), a fast and robust method for reconstructing dynamic 3D hand-object interactions from a single monocular video. Unlike recent approaches that optimize heavy neural representations, our method focuses on tracking the hand and the object efficiently, once initialized from pretrained large models. Our key insight is that accurate and temporally stable hand-object motion can be recovered using a compact Sum-of-Gaussians (SoG) representation, revived from classical tracking literature and integrated with generative Gaussian-based initializations. We initialize object pose and geometry using a video-adapted SAM3D pipeline, then convert the resulting dense Gaussian representation into a lightweight SoG via subsampling. This compact representation enables efficient and fast tracking while preserving geometric fidelity. For the hand, we adopt a complementary strategy: starting from off-the-shelf monocular hand pose initialization, we refine hand motion using simple yet effective 2D joint and depth alignment losses, avoiding per-frame refinement of a detailed 3D hand appearance model while maintaining stable articulation. Extensive experiments on public benchmarks demonstrate that GraG reconstructs temporally coherent hand-object interactions on long sequences 6.4x faster than prior work while improving object reconstruction by 13.4% and reducing hand's per-joint position error by over 65%.
>
---
#### [new 036] Unlocking the Potential of Grounding DINO in Videos: Parameter-Efficient Adaptation for Limited-Data Spatial-Temporal Localization
- **分类: cs.CV**

- **简介: 该论文聚焦视频时空定位任务，解决小数据下模型过拟合和缺乏时序感知的问题。通过适配预训练模型，引入轻量级模块提升时空理解能力。**

- **链接: [https://arxiv.org/pdf/2604.12346](https://arxiv.org/pdf/2604.12346)**

> **作者:** Zanyi Wang; Fan Li; Dengyang Jiang; Liuzhuozheng Li; Yunhua Zhong; Guang Dai; Mengmeng Wang
>
> **摘要:** Spatio-temporal video grounding (STVG) aims to localize queried objects within dynamic video segments. Prevailing fully-trained approaches are notoriously data-hungry. However, gathering large-scale STVG data is exceptionally challenging: dense frame-level bounding boxes and complex temporal language alignments are prohibitively expensive to annotate, especially for specialized video domains. Consequently, conventional models suffer from severe overfitting on these inherently limited datasets, while zero-shot foundational models lack the task-specific temporal awareness needed for precise localization. To resolve this small-data challenge, we introduce ST-GD, a data-efficient framework that adapts pre-trained 2D visual-language models (e.g., Grounding DINO) to video tasks. To avoid destroying pre-trained priors on small datasets, ST-GD keeps the base model frozen and strategically injects lightweight adapters (~10M trainable parameters) to instill spatio-temporal awareness, alongside a novel temporal decoder for boundary prediction. This design naturally counters data scarcity. Consequently, ST-GD excels in data-scarce scenarios, achieving highly competitive performance on the limited-scale HC-STVG v1/v2 benchmarks, while maintaining robust generalization on the VidSTG dataset. This validates ST-GD as a powerful paradigm for complex video understanding under strict small-data constraints.
>
---
#### [new 037] StructDiff: A Structure-Preserving and Spatially Controllable Diffusion Model for Single-Image Generation
- **分类: cs.CV**

- **简介: 该论文提出StructDiff，解决单图像生成中的结构保持与空间控制问题，通过自适应感受野和3D位置编码实现。**

- **链接: [https://arxiv.org/pdf/2604.12575](https://arxiv.org/pdf/2604.12575)**

> **作者:** Yinxi He; Kang Liao; Chunyu Lin; Tianyi Wei; Yao Zhao
>
> **备注:** Accepted by IEEE Transactions on Multimedia (Regular Paper)
>
> **摘要:** This paper introduces StructDiff, a generative framework based on a single-scale diffusion model for single-image generation. Single-image generation aims to synthesize diverse samples with similar visual content to the source image by capturing its internal statistics, without relying on external data. However, existing methods often struggle to preserve the structural layout, especially for images with large rigid objects or strict spatial constraints. Moreover, most approaches lack spatial controllability, making it difficult to guide the structure or placement of generated content. To address these challenges, StructDiff introduces an \textit{adaptive receptive field} module to maintain both global and local distributions. Building on this foundation, StructDiff incorporates 3D positional encoding (PE) as a spatial prior, allowing flexible control over positions, scale, and local details of generated objects. To our knowledge, this spatial control capability represents the first exploration of PE-based manipulation in single-image generation. Furthermore, we propose a novel evaluation criterion for single-image generation based on large language models (LLMs). This criterion specifically addresses the limitations of existing objective metrics and the high labor costs associated with user studies. StructDiff also demonstrates broad applicability across downstream tasks, such as text-guided image generation, image editing, outpainting, and paint-to-image synthesis. Extensive experiments demonstrate that StructDiff outperforms existing methods in structural consistency, visual quality, and spatial controllability. The project page is available at this https URL.
>
---
#### [new 038] Does Visual Token Pruning Improve Calibration? An Empirical Study on Confidence in MLLMs
- **分类: cs.CV**

- **简介: 该论文属于多模态模型任务，研究视觉token剪枝对模型校准的影响，即预测置信度与实际正确性的匹配程度。通过实验验证不同剪枝策略对校准指标的效果。**

- **链接: [https://arxiv.org/pdf/2604.12035](https://arxiv.org/pdf/2604.12035)**

> **作者:** Kaizhen Tan
>
> **摘要:** Visual token pruning is a widely used strategy for efficient inference in multimodal large language models (MLLMs), but existing work mainly evaluates it with task accuracy. In this paper, we study how visual token pruning affects model calibration, that is, whether predicted confidence matches actual correctness. Using LLaVA-1.5-7B on POPE and ScienceQA-IMG, we evaluate Expected Calibration Error (ECE), Brier score, and AURC under several pruning strategies, including SCOPE with different saliency weights, saliency-only pruning, FastV, and random pruning, across multiple token budgets. Our results show that pruning does not simply trade reliability for efficiency. On POPE, a pure-coverage setting in SCOPE achieves substantially lower ECE than the full unpruned model while maintaining similar accuracy. An internal alpha-sweep further shows a consistent trend: reducing the saliency weight improves calibration at all tested token budgets, while accuracy changes only slightly. In contrast, saliency-based pruning leads to worse calibration, and real FastV causes severe performance degradation in our setting. On ScienceQA-IMG, pruning also reduces ECE, with accuracy remaining stable or slightly improving. We additionally study the gap power exponent in coverage-based selection and find that its default setting is not always optimal. Overall, our results suggest that visual token pruning should be evaluated not only by accuracy, but also by confidence quality, especially for multimodal systems that need reliable decisions.
>
---
#### [new 039] CoD-Lite: Real-Time Diffusion-Based Generative Image Compression
- **分类: cs.CV**

- **简介: 该论文属于图像压缩任务，旨在解决实时轻量级生成式压缩问题。通过优化预训练和结构设计，提出一种高效扩散编码器，实现高帧率压缩与解压。**

- **链接: [https://arxiv.org/pdf/2604.12525](https://arxiv.org/pdf/2604.12525)**

> **作者:** Zhaoyang Jia; Naifu Xue; Zihan Zheng; Jiahao Li; Bin Li; Xiaoyi Zhang; Zongyu Guo; Yuan Zhang; Houqiang Li; Yan Lu
>
> **摘要:** Recent advanced diffusion methods typically derive strong generative priors by scaling diffusion transformers. However, scaling fails to generalize when adapted for real-time compression scenarios that demand lightweight models. In this paper, we explore the design of real-time and lightweight diffusion codecs by addressing two pivotal questions. First, does diffusion pre-training benefit lightweight diffusion codecs? Through systematic analysis, we find that generation-oriented pre-training is less effective at small model scales whereas compression-oriented pre-training yields consistently better performance. Second, are transformers essential? We find that while global attention is crucial for standard generation, lightweight convolutions suffice for compression-oriented diffusion when paired with distillation. Guided by these findings, we establish a one-step lightweight convolution diffusion codec that achieves real-time $60$~FPS encoding and $42$~FPS decoding at 1080p. Further enhanced by distillation and adversarial learning, the proposed codec reduces bitrate by 85\% at a comparable FID to MS-ILLM, bridging the gap between generative compression and practical real-time deployment. Codes are released at this https URL
>
---
#### [new 040] ArtifactWorld: Scaling 3D Gaussian Splatting Artifact Restoration via Video Generation Models
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决3DGS在稀疏视角下的几何与光度退化问题。通过构建大规模数据集和双模型框架，提升修复效果与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.12251](https://arxiv.org/pdf/2604.12251)**

> **作者:** Xinliang Wang; Yifeng Shi; Zhenyu Wu
>
> **备注:** The second author is the corresponding author
>
> **摘要:** 3D Gaussian Splatting (3DGS) delivers high-fidelity real-time rendering but suffers from geometric and photometric degradations under sparse-view constraints. Current generative restoration approaches are often limited by insufficient temporal coherence, a lack of explicit spatial constraints, and a lack of large-scale training data, resulting in multi-view inconsistencies, erroneous geometric hallucinations, and limited generalization to diverse real-world artifact distributions. In this paper, we present ArtifactWorld, a framework that resolves 3DGS artifact repair through systematic data expansion and a homogeneous dual-model paradigm. To address the data bottleneck, we establish a fine-grained phenomenological taxonomy of 3DGS artifacts and construct a comprehensive training set of 107.5K diverse paired video clips to enhance model robustness. Architecturally, we unify the restoration process within a video diffusion backbone, utilizing an isomorphic predictor to localize structural defects via an artifact heatmap. This heatmap then guides the restoration through an Artifact-Aware Triplet Fusion mechanism, enabling precise, intensity-guided spatio-temporal repair within native self-attention. Extensive experiments demonstrate that ArtifactWorld achieves state-of-the-art performance in sparse novel view synthesis and robust 3D reconstruction. Code and dataset will be made public.
>
---
#### [new 041] Detecting Precise Hand Touch Moments in Egocentric Video
- **分类: cs.CV**

- **简介: 该论文属于动作检测任务，旨在精确识别第一人称视频中手与物体接触的时刻。针对接触判定困难的问题，提出HiCE模块和TouchMoment数据集，提升检测精度。**

- **链接: [https://arxiv.org/pdf/2604.12343](https://arxiv.org/pdf/2604.12343)**

> **作者:** Huy Anh Nguyen; Feras Dayoub; Minh Hoai
>
> **备注:** Accepted to CVPR Findings 2026
>
> **摘要:** We address the challenging task of detecting the precise moment when hands make contact with objects in egocentric videos. This frame-level detection is crucial for augmented reality, human-computer interaction, assistive technologies, and robot learning applications, where contact onset signals action initiation or completion. Temporally precise detection is particularly challenging due to subtle hand motion variations near contact, frequent occlusions, fine-grained manipulation patterns, and the inherent motion dynamics of first-person perspectives. To tackle these challenges, we propose a Hand-informed Context Enhanced module (HiCE; pronounced `high-see') that leverages spatiotemporal features from hand regions and their surrounding context through cross-attention mechanisms, learning to identify potential contact patterns. Our approach is further refined with a grasp-aware loss and soft label that emphasizes hand pose patterns and movement dynamics characteristic of touch events, enabling the model to distinguish between near-contact and actual contact frames. We also introduce TouchMoment, an egocentric dataset containing 4,021 videos and 8,456 annotated contact moments spanning over one million frames. Experiments on TouchMoment show that, under a strict evaluation criterion that counts a prediction as correct only if it falls within a two-frame tolerance of the ground-truth moment, our method achieves substantial gains and outperforms state-of-the-art event-spotting baselines by 16.91% average precision.
>
---
#### [new 042] Relaxing Anchor-Frame Dominance for Mitigating Hallucinations in Video Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于视频大模型任务，旨在解决视频生成中的幻觉问题。通过提出DTR方法，平衡时间证据分配，提升生成稳定性与准确性。**

- **链接: [https://arxiv.org/pdf/2604.12582](https://arxiv.org/pdf/2604.12582)**

> **作者:** Zijian Liu; Sihan Cao; Pengcheng Zheng; Kuien Liu; Caiyan Qin; Xiaolin Qin; Jiwei Wei; Chaoning Zhang
>
> **摘要:** Recent Video Large Language Models (Video-LLMs) have demonstrated strong capability in video understanding, yet they still suffer from hallucinations. Existing mitigation methods typically rely on training, input modification, auxiliary guidance, or additional decoding procedures, while largely overlooking a more fundamental challenge. During generation, Video-LLMs tend to over-rely on a limited portion of temporal evidence, leading to temporally imbalanced evidence aggregation across the video. To address this issue, we investigate a decoder-side phenomenon in which the model exhibits a temporally imbalanced concentration pattern. We term the frame with the highest aggregated frame-level attention mass the anchor frame. We find that this bias is largely independent of the input video and instead appears to reflect a persistent, model-specific structural or positional bias, whose over-dominance is closely associated with hallucination-prone generation. Motivated by this insight, we propose Decoder-side Temporal Rebalancing (DTR), a training-free, layer-selective inference method that rebalances temporal evidence allocation in middle-to-late decoder layers without altering visual encoding or requiring auxiliary models. DTR adaptively calibrates decoder-side visual attention to alleviate temporally imbalanced concentration and encourage under-attended frames to contribute more effectively to response generation. In this way, DTR guides the decoder to ground its outputs in temporally broader and more balanced video evidence. Extensive experiments on hallucination and video understanding benchmarks show that DTR consistently improves hallucination robustness across diverse Video-LLM families, while preserving competitive video understanding performance and high inference efficiency.
>
---
#### [new 043] Physics-Grounded Monocular Vehicle Distance Estimation Using Standardized License Plate Typography
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于车辆距离估计任务，解决单目相机的尺度模糊问题。通过标准化车牌作为标记，结合几何先验和多阶段识别，实现精确距离估算。**

- **链接: [https://arxiv.org/pdf/2604.12239](https://arxiv.org/pdf/2604.12239)**

> **作者:** Manognya Lokesh Reddy; Zheng Liu
>
> **备注:** 17 pages, 9 figures
>
> **摘要:** Accurate inter-vehicle distance estimation is a cornerstone of Advanced Driver Assistance Systems (ADAS) and autonomous driving. While LiDAR and radar provide high precision, their high cost prohibits widespread adoption in mass-market vehicles. Monocular camera-based estimation offers a low-cost alternative but suffers from fundamental scale ambiguity. Recent deep learning methods for monocular depth achieve impressive results yet require expensive supervised training, suffer from domain shift, and produce predictions that are difficult to certify for safety-critical deployment. This paper presents a framework that exploits the standardized typography of United States license plates as passive fiducial markers for metric ranging, resolving scale ambiguity through explicit geometric priors without any training data or active illumination. First, a four-method parallel plate detector achieves robust plate reading across the full automotive lighting range. Second, a three-stage state identification engine fusing OCR text matching, multi-design color scoring, and a lightweight neural network classifier provides robust identification across all ambient conditions. Third, hybrid depth fusion with inverse-variance weighting and online scale alignment, combined with a one-dimensional constant-velocity Kalman filter, delivers smoothed distance, relative velocity, and time-to-collision for collision warning. Baseline validation reproduces a 2.3% coefficient of variation in character height measurements and a 36% reduction in distance-estimate variance compared with plate-width methods from prior work. Extensive outdoor experiments confirm a mean absolute error of 2.3% at 10 m and continuous distance output during brief plate occlusions, outperforming deep learning baselines by a factor of five in relative error.
>
---
#### [new 044] Generative Refinement Networks for Visual Synthesis
- **分类: cs.CV**

- **简介: 该论文提出GRN模型，解决视觉合成中的计算效率与生成质量问题。针对扩散模型和自回归模型的不足，GRN采用分层二进制量化和全局优化机制，提升图像与视频生成性能。**

- **链接: [https://arxiv.org/pdf/2604.13030](https://arxiv.org/pdf/2604.13030)**

> **作者:** Jian Han; Jinlai Liu; Jiahuan Wang; Bingyue Peng; Zehuan Yuan
>
> **备注:** code: this https URL
>
> **摘要:** While diffusion models dominate the field of visual generation, they are computationally inefficient, applying a uniform computational effort regardless of different complexity. In contrast, autoregressive (AR) models are inherently complexity-aware, as evidenced by their variable likelihoods, but are often hindered by lossy discrete tokenization and error accumulation. In this work, we introduce Generative Refinement Networks (GRN), a next-generation visual synthesis paradigm to address these issues. At its core, GRN addresses the discrete tokenization bottleneck through a theoretically near-lossless Hierarchical Binary Quantization (HBQ), achieving a reconstruction quality comparable to continuous counterparts. Built upon HBQ's latent space, GRN fundamentally upgrades AR generation with a global refinement mechanism that progressively perfects and corrects artworks -- like a human artist painting. Besides, GRN integrates an entropy-guided sampling strategy, enabling complexity-aware, adaptive-step generation without compromising visual quality. On the ImageNet benchmark, GRN establishes new records in image reconstruction (0.56 rFID) and class-conditional image generation (1.81 gFID). We also scale GRN to more challenging text-to-image and text-to-video generation, delivering superior performance on an equivalent scale. We release all models and code to foster further research on GRN.
>
---
#### [new 045] Reading Between the Pixels: Linking Text-Image Embedding Alignment to Typographic Attack Success on Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型的文本-图像嵌入对齐与字体攻击成功率的关系，旨在评估不同攻击方式和视觉条件下的模型脆弱性。**

- **链接: [https://arxiv.org/pdf/2604.12371](https://arxiv.org/pdf/2604.12371)**

> **作者:** Ravikumar Balakrishnan; Sanket Mendapara; Ankit Garg
>
> **备注:** Accepted at ICLR 2026 Workshop on Agents in the Wild
>
> **摘要:** We study typographic prompt injection attacks on vision-language models (VLMs), where adversarial text is rendered as images to bypass safety mechanisms, posing a growing threat as VLMs serve as the perceptual backbone of autonomous agents, from browser automation and computer-use systems to camera-equipped embodied agents. In practice, the attack surface is heterogeneous: adversarial text appears at varying font sizes and under diverse visual conditions, while the growing ecosystem of VLMs exhibits substantial variation in vulnerability, complicating defensive approaches. Evaluating 1,000 prompts from SALAD-Bench across four VLMs, namely, GPT-4o, Claude Sonnet 4.5, Mistral-Large-3, and Qwen3-VL-4B-Instruct under varying font sizes (6--28px) and visual transformations (rotation, blur, noise, contrast changes), we find: (1) font size significantly affects attack success rate (ASR), with very small fonts (6px) yielding near-zero ASR while mid-range fonts achieve peak effectiveness; (2) text attacks are more effective than image attacks for GPT-4o (36% vs 8%) and Claude (47% vs 22%), while Qwen3-VL and Mistral show comparable ASR across modalities; (3) text-image embedding distance from two multimodal embedding models (JinaCLIP and Qwen3-VL-Embedding) shows strong negative correlation with ASR across all four models (r = -0.71 to -0.93, p < 0.01); (4) heavy degradations increase embedding distance by 10--12% and reduce ASR by 34--96%, while rotation asymmetrically affects models (Mistral drops 50%, GPT-4o unchanged). These findings highlight that model-specific robustness patterns preclude one-size-fits-all defenses and offer empirical guidance for practitioners selecting VLM backbones for agentic systems operating in adversarial environments.
>
---
#### [new 046] A Sanity Check on Composed Image Retrieval
- **分类: cs.CV**

- **简介: 该论文属于图像检索任务，旨在解决CIR评估不准确的问题。提出FISD基准和多轮评估框架，提升模型评价的准确性与实用性。**

- **链接: [https://arxiv.org/pdf/2604.12904](https://arxiv.org/pdf/2604.12904)**

> **作者:** Yikun Liu; Jiangchao Yao; Weidi Xie; Yanfeng Wang
>
> **摘要:** Composed Image Retrieval (CIR) aims to retrieve a target image based on a query composed of a reference image, and a relative caption that specifies the desired modification. Despite the rapid development of CIR models, their performance is not well characterized by existing benchmarks, which inherently contain indeterminate queries degrading the evaluation (i.e., multiple candidate images, rather than solely the target image, meet the query criteria), and have not considered their effectiveness in the context of the multi-round system. Motivated by this, we consider improving the evaluation procedure from two aspects: 1) we introduce FISD, a Fully-Informed Semantically-Diverse benchmark, which employs generative models to precisely control the variables of reference-target image pairs, enabling a more accurate evaluation of CIR methods across six dimensions, without query ambiguity; 2) we propose an automatic multi-round agentic evaluation framework to probe the potential of the existing models in the interactive scenarios. By observing how models adapt and refine their choices over successive rounds of queries, this framework provides a more realistic appraisal of their efficacy in practical applications. Extensive experiments and comparisons prove the value of our novel evaluation on typical CIR methods.
>
---
#### [new 047] HyperLiDAR: Adaptive Post-Deployment LiDAR Segmentation via Hyperdimensional Computing
- **分类: cs.CV**

- **简介: 该论文属于LiDAR语义分割任务，解决边缘设备上模型适应性差的问题。提出HyperLiDAR框架，利用超维计算实现高效、轻量级的后部署适应。**

- **链接: [https://arxiv.org/pdf/2604.12331](https://arxiv.org/pdf/2604.12331)**

> **作者:** Ivannia Gomez Moreno; Yi Yao; Ye Tian; Xiaofan Yu; Flavio Ponzina; Michael Sullivan; Jingyi Zhang; Mingyu Yang; Hun Seok Kim; Tajana Rosing
>
> **摘要:** LiDAR semantic segmentation plays a pivotal role in 3D scene understanding for edge applications such as autonomous driving. However, significant challenges remain for real-world deployments, particularly for on-device post-deployment adaptation. Real-world environments can shift as the system navigates through different locations, leading to substantial performance degradation without effective and timely model adaptation. Furthermore, edge systems operate under strict computational and energy constraints, making it infeasible to adapt conventional segmentation models (based on large neural networks) directly on-device. To address the above challenges, we introduce HyperLiDAR, the first lightweight, post-deployment LiDAR segmentation framework based on Hyperdimensional Computing (HDC). The design of HyperLiDAR fully leverages the fast learning and high efficiency of HDC, inspired by how the human brain processes information. To further improve the adaptation efficiency, we identify the high data volume per scan as a key bottleneck and introduce a buffer selection strategy that focuses learning on the most informative points. We conduct extensive evaluations on two state-of-the-art LiDAR segmentation benchmarks and two representative devices. Our results show that HyperLiDAR outperforms or achieves comparable adaptation performance to state-of-the-art segmentation methods, while achieving up to a 13.8x speedup in retraining.
>
---
#### [new 048] See, Point, Refine: Multi-Turn Approach to GUI Grounding with Visual Feedback
- **分类: cs.CV**

- **简介: 该论文属于GUI接地任务，解决高密度界面中精准定位问题。通过多轮迭代修正，提升点击精度和任务成功率。**

- **链接: [https://arxiv.org/pdf/2604.13019](https://arxiv.org/pdf/2604.13019)**

> **作者:** Himangi Mittal; Gaurav Mittal; Nelson Daniel Troncoso; Yu Hu
>
> **摘要:** Computer Use Agents (CUAs) fundamentally rely on graphical user interface (GUI) grounding to translate language instructions into executable screen actions, but editing-level grounding in dense coding interfaces, where sub-pixel accuracy is required to interact with dense IDE elements, remains underexplored. Existing approaches typically rely on single-shot coordinate prediction, which lacks a mechanism for error correction and often fails in high-density interfaces. In this technical report, we conduct an empirical study of pixel-precise cursor localization in coding environments. Instead of a single-step execution, our agent engages in an iterative refinement process, utilizing visual feedback from previous attempts to reach the target element. This closed-loop grounding mechanism allows the agent to self-correct displacement errors and adapt to dynamic UI changes. We evaluate our approach across GPT-5.4, Claude, and Qwen on a suite of complex coding benchmarks, demonstrating that multi-turn refinement significantly outperforms state-of-the-art single-shot models in both click precision and overall task success rate. Our results suggest that iterative visual reasoning is a critical component for the next generation of reliable software engineering agents. Code: this https URL.
>
---
#### [new 049] Cross-Modal Knowledge Distillation for PET-Free Amyloid-Beta Detection from MRI
- **分类: cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决无需PET即可通过MRI检测淀粉样蛋白β的问题。通过知识蒸馏方法，从PET数据中学习并迁移至MRI，实现有效预测。**

- **链接: [https://arxiv.org/pdf/2604.12574](https://arxiv.org/pdf/2604.12574)**

> **作者:** Francesco Chiumento; Julia Dietlmeier; Ronan P. Killeen; Kathleen M. Curran; Noel E. O'Connor; Mingming Liu
>
> **备注:** Accepted to CVPR Workshops 2026 (PHAROS-AIF-MIH)
>
> **摘要:** Detecting amyloid-$\beta$ (A$\beta$) positivity is crucial for early diagnosis of Alzheimer's disease but typically requires PET imaging, which is costly, invasive, and not widely accessible, limiting its use for population-level screening. We address this gap by proposing a PET-guided knowledge distillation framework that enables A$\beta$ prediction from MRI alone, without requiring non-imaging clinical covariates or PET at inference. Our approach employs a BiomedCLIP-based teacher model that learns PET-MRI alignment via cross-modal attention and triplet contrastive learning with PET-informed (Centiloid-aware) online negative sampling. An MRI-only student then mimics the teacher via feature-level and logit-level distillation. Evaluated across four MRI contrasts (T1w, T2w, FLAIR, T2*) and two independent datasets, our approach demonstrates effective knowledge transfer (best AUC: 0.74 on OASIS-3, 0.68 on ADNI) while maintaining interpretability and eliminating the need for clinical variables. Saliency analysis confirms that predictions focus on anatomically relevant cortical regions, supporting the clinical viability of PET-free A$\beta$ screening. Code is available at this https URL.
>
---
#### [new 050] LiveMoments: Reselected Key Photo Restoration in Live Photos via Reference-guided Diffusion
- **分类: cs.CV**

- **简介: 该论文提出LiveMoments，解决Live Photos中重新选择帧质量下降的问题。通过参考引导的图像修复框架，提升重选帧的视觉质量。**

- **链接: [https://arxiv.org/pdf/2604.12286](https://arxiv.org/pdf/2604.12286)**

> **作者:** Clara Xue; Zizheng Yan; Zhenning Shi; Yuhang Yu; Jingyu Zhuang; Qi Zhang; Jinwei Chen; Qingnan Fan
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Live Photo captures both a high-quality key photo and a short video clip to preserve the precious dynamics around the captured moment. While users may choose alternative frames as the key photo to capture better expressions or timing, these frames often exhibit noticeable quality degradation, as the photo capture ISP pipeline delivers significantly higher image quality than the video pipeline. This quality gap highlights the need for dedicated restoration techniques to enhance the reselected key photo. To this end, we propose LiveMoments, a reference-guided image restoration framework tailored for the reselected key photo in Live Photos. Our method employs a two-branch neural network: a reference branch that extracts structural and textural information from the original high-quality key photo, and a main branch that restores the reselected frame using the guidance provided by the reference branch. Furthermore, we introduce a unified Motion Alignment module that incorporates motion guidance for spatial alignment at both the latent and image levels. Experiments on real and synthetic Live Photos demonstrate that LiveMoments significantly improves perceptual quality and fidelity over existing solutions, especially in scenes with fast motion or complex structures. Our code is available at this https URL.
>
---
#### [new 051] Ultra-low-light computer vision using trained photon correlations
- **分类: cs.CV; physics.optics**

- **简介: 该论文属于计算机视觉任务，旨在提升超低光条件下的物体识别准确率。通过训练光子相关性与Transformer模型结合，实现更优的识别效果。**

- **链接: [https://arxiv.org/pdf/2604.11993](https://arxiv.org/pdf/2604.11993)**

> **作者:** Mandar M. Sohoni; Jérémie Laydevant; Mathieu Ouellet; Shi-Yuan Ma; Ryotatsu Yanagimoto; Benjamin A. Ash; Tatsuhiro Onodera; Tianyu Wang; Logan G. Wright; Peter L. McMahon
>
> **备注:** 49 pages, 47 figures
>
> **摘要:** Illumination using correlated photon sources has been established as an approach to allowing high-fidelity images to be reconstructed from noisy camera frames by taking advantage of the knowledge that signal photons are spatially correlated whereas detector clicks due to noise are uncorrelated. However, in computer-vision tasks, the goal is often not ultimately to reconstruct an image, but to make inferences about a scene -- such as what object is present. Here we show how correlated-photon illumination can be used to gain an advantage in a hybrid optical-electronic computer-vision pipeline for object recognition. We demonstrate correlation-aware training (CAT): end-to-end optimization of a trainable correlated-photon illumination source and a Transformer backend in a way that the Transformer can learn to benefit from the correlations, using a small number (<= 100) of shots. We show a classification accuracy enhancement of up to 15 percentage points over conventional, uncorrelated-illumination-based computer vision in ultra-low-light and noisy imaging conditions, as well as an improvement over using untrained correlated-photon illumination. Our work illustrates how specializing to a computer-vision task -- object recognition -- and training the pattern of photon correlations in conjunction with a digital backend allows us to push the limits of accuracy in highly photon-budget-constrained scenarios beyond existing methods focused on image reconstruction.
>
---
#### [new 052] Bridging the Micro--Macro Gap: Frequency-Aware Semantic Alignment for Image Manipulation Localization
- **分类: cs.CV**

- **简介: 该论文属于图像篡改定位任务，旨在解决传统与扩散生成篡改识别的微宏差距问题。提出FASA框架，结合频率特征与语义对齐，提升定位精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.12341](https://arxiv.org/pdf/2604.12341)**

> **作者:** Xiaojie Liang; Zhimin Chen; Ziqi Sheng; Wei Lu
>
> **摘要:** As generative image editing advances, image manipulation localization (IML) must handle both traditional manipulations with conspicuous forensic artifacts and diffusion-generated edits that appear locally realistic. Existing methods typically rely on either low-level forensic cues or high-level semantics alone, leading to a fundamental micro--macro gap. To bridge this gap, we propose FASA, a unified framework for localizing both traditional and diffusion-generated manipulations. Specifically, we extract manipulation-sensitive frequency cues through an adaptive dual-band DCT module and learn manipulation-aware semantic priors via patch-level contrastive alignment on frozen CLIP representations. We then inject these priors into a hierarchical frequency pathway through a semantic-frequency side adapter for multi-scale feature interaction, and employ a prototype-guided, frequency-gated mask decoder to integrate semantic consistency with boundary-aware localization for tampered region prediction. Extensive experiments on OpenSDI and multiple traditional manipulation benchmarks demonstrate state-of-the-art localization performance, strong cross-generator and cross-dataset generalization, and robust performance under common image degradations.
>
---
#### [new 053] ViLL-E: Video LLM Embeddings for Retrieval
- **分类: cs.CV**

- **简介: 该论文提出ViLL-E，解决视频检索任务中VideoLLM表现不佳的问题，通过新嵌入生成机制提升性能。**

- **链接: [https://arxiv.org/pdf/2604.12148](https://arxiv.org/pdf/2604.12148)**

> **作者:** Rohit Gupta; Jayakrishnan Unnikrishnan; Fan Fei; Sheng Liu; Son Tran; Mubarak Shah
>
> **备注:** Accepted at ACL 2026 Main conference
>
> **摘要:** Video Large Language Models (VideoLLMs) excel at video understanding tasks where outputs are textual, such as Video Question Answering and Video Captioning. However, they underperform specialized embedding-based models in Retrieval tasks, such as Text-toVideo Retrieval and Moment Retrieval. We introduce ViLL-E (Video-LLM-Embed), a unified VideoLLM architecture endowed with a novel embedding generation mechanism that allows the model to "think longer" for complex videos and stop early for easy ones. We train this model with a three-stage training methodology combining generative and contrastive learning: initial large-scale pre-training with video-caption pairs; followed by continual training on a smaller, detailed-caption dataset; and concluding with task-specific fine-tuning on a novel multi-task dataset covering Video QA, Temporal Localization, Video Retrieval, and Video-Text Matching. Our model significantly improves temporal localization (on avg. 7% over other VideoLLMs) and video retrieval (up to 4% over dual encoder models), achieving performance comparable to state-of-the-art specialized embedding models while remaining competitive on VideoQA tasks. Furthermore, our joint contrastive-generative training unlocks new zero-shot capabilities, significantly outperforming state-of-the-art methods in composed video retrieval (+5% over SotA) and retrieval from long text (+2% over SotA).
>
---
#### [new 054] Fragile Reconstruction: Adversarial Vulnerability of Reconstruction-Based Detectors for Diffusion-Generated Images
- **分类: cs.CV**

- **简介: 该论文属于AI生成图像检测任务，研究重建类检测器的对抗脆弱性。工作包括验证对抗攻击对检测器的破坏性、分析攻击迁移性及评估防御方法有效性，揭示其安全缺陷。**

- **链接: [https://arxiv.org/pdf/2604.12781](https://arxiv.org/pdf/2604.12781)**

> **作者:** Haoyang Jiang; Mingyang Yi; Shaolei Zhang; Junxian Cai; Qingbin Liu; Xi Chen; Ju Fan
>
> **摘要:** Recently, detecting AI-generated images produced by diffusion-based models has attracted increasing attention due to their potential threat to safety. Among existing approaches, reconstruction-based methods have emerged as a prominent paradigm for this task. However, we find that such methods exhibit severe security vulnerabilities to adversarial perturbations; that is, by adding imperceptible adversarial perturbations to input images, the detection accuracy of classifiers collapses to near zero. To verify this threat, we present a systematic evaluation of the adversarial robustness of three representative detectors across four diverse generative backbone models. First, we construct adversarial attacks in white-box scenarios, which degrade the performance of all well-trained detectors. Moreover, we find that these attacks demonstrate transferability; specifically, attacks crafted against one detector can be transferred to others, indicating that adversarial attacks on detectors can also be constructed in a black-box setting. Finally, we assess common countermeasures and find that standard defense methods against adversarial attacks provide limited mitigation. We attribute these failures to the low signal-to-noise ratio (SNR) of attacked samples as perceived by the detectors. Overall, our results reveal fundamental security limitations of reconstruction-based detectors and highlight the need to rethink existing detection strategies.
>
---
#### [new 055] PDF-GS: Progressive Distractor Filtering for Robust 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决3DGS对异常数据敏感的问题。提出PDF-GS框架，通过渐进式过滤提升鲁棒性，实现高质量无干扰重建。**

- **链接: [https://arxiv.org/pdf/2604.12580](https://arxiv.org/pdf/2604.12580)**

> **作者:** Kangmin Seo; MinKyu Lee; Tae-Young Kim; ByeongCheol Lee; JoonSeoung An; Jae-Pil Heo
>
> **备注:** Accepted to CVPR Findings 2026
>
> **摘要:** Recent advances in 3D Gaussian Splatting (3DGS) have enabled impressive real-time photorealistic rendering. However, conventional training pipelines inherently assume full multi-view consistency among input images, which makes them sensitive to distractors that violate this assumption and cause visual artifacts. In this work, we revisit an underexplored aspect of 3DGS: its inherent ability to suppress inconsistent signals. Building on this insight, we propose PDF-GS (Progressive Distractor Filtering for Robust 3D Gaussian Splatting), a framework that amplifies this self-filtering property through a progressive multi-phase optimization. The progressive filtering phases gradually remove distractors by exploiting discrepancy cues, while the following reconstruction phase restores fine-grained, view-consistent details from the purified Gaussian representation. Through this iterative refinement, PDF-GS achieves robust, high-fidelity, and distractor-free reconstructions, consistently outperforming baselines across diverse datasets and challenging real-world conditions. Moreover, our approach is lightweight and easily adaptable to existing 3DGS frameworks, requiring no architectural changes or additional inference overhead, leading to a new state-of-the-art performance. The code is publicly available at this https URL.
>
---
#### [new 056] All in One: A Unified Synthetic Data Pipeline for Multimodal Video Understanding
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出一种统一的合成数据生成管道，用于多模态视频理解任务，解决真实数据标注成本高、多样性不足的问题。通过生成丰富多样的合成数据提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.12335](https://arxiv.org/pdf/2604.12335)**

> **作者:** Tanzila Rahman; Renjie Liao; Leonid Sigal
>
> **备注:** 8 Pages, 4 Tables, 4 Figures
>
> **摘要:** Training multimodal large language models (MLLMs) for video understanding requires large-scale annotated data spanning diverse tasks such as object counting, question answering, and segmentation. However, collecting and annotating multimodal video data in real-world is costly, slow, and inherently limited in diversity and coverage. To address this challenge, we propose a unified synthetic data generation pipeline capable of automatically producing unlimited multimodal video data with rich and diverse supervision. Our framework supports multiple task formats within a single pipeline, enabling scalable and consistent data creation across tasks. To further enhance reasoning ability, we introduce a VQA-based fine-tuning strategy that trains models to answer structured questions about visual content rather than relying solely on captions or simple instructions. This formulation encourages deeper visual grounding and reasoning. We evaluate our approach in three challenging tasks: video object counting, video-based visual question answering, and video object segmentation. Experimental results demonstrate that models trained predominantly on synthetic data generalize effectively to real-world datasets, often outperforming traditionally trained counterparts. Our findings highlight the potential of unified synthetic data pipelines as a scalable alternative to expensive real-world annotation for multimodal video understanding.
>
---
#### [new 057] Dual-Modality Anchor-Guided Filtering for Test-time Prompt Tuning
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的适应任务，解决TPT中视图选择无效的问题。通过引入双模态锚点，提升视图筛选的语义准确性，增强模型适应效果。**

- **链接: [https://arxiv.org/pdf/2604.12403](https://arxiv.org/pdf/2604.12403)**

> **作者:** Jungwon Choi; Eunwoo Kim
>
> **备注:** Accepted by CVPR 2026 findings
>
> **摘要:** Test-Time Prompt Tuning (TPT) adapts vision-language models using augmented views, but its effectiveness is hindered by the challenge of determining which views are beneficial. Standard entropy-based filtering relies on the internal confidence scores of the model, which are often miscalibrated under distribution shift, assigning high confidence to irrelevant crops or background regions while ignoring semantic content. To address this, we propose a dual-modality anchor-guided framework that grounds view selection in semantic evidence. We introduce a text anchor from attribute-rich descriptions, to provide fine-grained class semantics, and an adaptive image anchor that captures evolving test-time statistics. Using these anchors, we filter views based on alignment and confidence, ensuring that only informative views guide adaptation. Moreover, we treat the anchors as auxiliary predictive heads and combine their predictions with the original output in a confidence-weighted ensemble, yielding a stable supervision signal for prompt updates. Extensive experiments on 15 benchmark datasets demonstrate new state-of-the-art performance, highlighting the contribution of anchor-guided supervision as a foundation for robust prompt updates.
>
---
#### [new 058] Fundus Image-based Glaucoma Screening via Retinal Knowledge-Oriented Dynamic Multi-Level Feature Integration
- **分类: cs.CV**

- **简介: 该论文属于糖尿病视网膜病变筛查任务，旨在解决现有模型缺乏解剖知识引导和固定区域特征提取不足的问题。通过引入动态多尺度特征学习与领域先验，提升诊断准确性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.12351](https://arxiv.org/pdf/2604.12351)**

> **作者:** Yuzhuo Zhou; Chi Liu; Sheng Shen; Zongyuan Ge; Fengshi Jing; Shiran Zhang; Yu Jiang; Anli Wang; Wenjian Liu; Feilong Yang; Tianqing Zhu; Xiaotong Han
>
> **备注:** 15 pages. In submission to an Elsevier Journal
>
> **摘要:** Automated diagnosis based on color fundus photography is essential for large-scale glaucoma screening. However, existing deep learning models are typically data-driven and lack explicit integration of retinal anatomical knowledge, which limits their robustness across heterogeneous clinical datasets. Moreover, pathological cues in fundus images may appear beyond predefined anatomical regions, making fixed-region feature extraction insufficient for reliable diagnosis. To address these challenges, we propose a retinal knowledge-oriented glaucoma screening framework that integrates dynamic multi-scale feature learning with domain-specific retinal priors. The framework adopts a tri-branch structure to capture complementary retinal representations, including global retinal context, structural features of the optic disc/cup, and dynamically localized pathological regions. A Dynamic Window Mechanism is devised to adaptively identify diagnostically informative regions, while a Knowledge-Enhanced Convolutional Attention Module incorporates retinal priors extracted from a pre-trained foundation model to guide attention learning. Extensive experiments on the large-scale AIROGS dataset demonstrate that the proposed method outperforms diverse baselines, achieving an AUC of 98.5% and an accuracy of 94.6%. Additional evaluations on multiple datasets from the SMDG-19 benchmark further confirm its strong cross-domain generalization capability, indicating that knowledge-guided attention combined with adaptive lesion localization can significantly improve the robustness of automated glaucoma screening systems.
>
---
#### [new 059] Boosting Visual Instruction Tuning with Self-Supervised Guidance
- **分类: cs.CV**

- **简介: 该论文属于视觉语言任务，解决MLLM在细粒度视觉推理上的不足。通过引入视觉引导的自监督指令，提升模型对视觉信息的利用，无需额外训练或修改架构。**

- **链接: [https://arxiv.org/pdf/2604.12966](https://arxiv.org/pdf/2604.12966)**

> **作者:** Sophia Sirko-Galouchenko; Monika Wysoczanska; Andrei Bursuc; Nicolas Thome; Spyros Gidaris
>
> **摘要:** Multimodal large language models (MLLMs) perform well on many vision-language tasks but often struggle with vision-centric problems that require fine-grained visual reasoning. Recent evidence suggests that this limitation arises not from weak visual representations, but from under-utilization of visual information during instruction tuning, where many tasks can be partially solved using language priors alone. We propose a simple and lightweight approach that augments visual instruction tuning with a small number of visually grounded self-supervised tasks expressed as natural language instructions. By reformulating classical self-supervised pretext tasks, such as rotation prediction, color matching, and cross-view correspondence, as image-instruction-response triplets, we introduce supervision that cannot be solved without relying on visual evidence. Our approach requires no human annotations, no architectural modifications, and no additional training stages. Across multiple models, training regimes, and benchmarks, injecting only a small fraction (3-10%) of such visually grounded instructions consistently improves performance on vision-centric evaluations. Our findings highlight instruction tuning with visually grounded SSL tasks as a powerful lever for improving visual reasoning in MLLMs through simple adjustments to the training data distribution. Code available at: this https URL
>
---
#### [new 060] From Attenuation to Attention: Variational Information Flow Manipulation for Fine-Grained Visual Perception
- **分类: cs.CV**

- **简介: 该论文属于视觉问答任务，解决MLLM在细粒度视觉感知中的信息丢失问题。提出VIF框架，通过概率方法增强视觉信号，提升模型细粒度理解能力。**

- **链接: [https://arxiv.org/pdf/2604.12508](https://arxiv.org/pdf/2604.12508)**

> **作者:** Jilong Zhu; Yang Feng
>
> **摘要:** While Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in general visual understanding, they frequently falter in fine-grained perception tasks that require identifying tiny objects or discerning subtle visual relationships. We attribute this limitation to Visual Attenuation: a phenomenon where sparse fine-grained visual signals are prematurely suppressed or diluted by dominant textual tokens during network propagation, resulting in a "loss of focus" during the deep-level decision-making process. Existing input-centric solutions fail to fundamentally reverse this intrinsic mechanism of information loss. To address this challenge, we propose the Variational Information Flow (VIF) framework. Adopting a probabilistic perspective, VIF leverages a Conditional Variational Autoencoder (CVAE) to model the visual saliency relevant to the question-answer pair as a latent distribution. As a plug-and-play module, VIF can be integrated into existing architectures. Extensive evaluations across diverse benchmarks, covering General VQA, fine-grained perception, and visual grounding, demonstrate that VIF yields competitive improvements over previous methods, validating its effectiveness in enhancing the fine-grained perception of MLLMs.
>
---
#### [new 061] V-Nutri: Dish-Level Nutrition Estimation from Egocentric Cooking Videos
- **分类: cs.CV**

- **简介: 该论文属于饮食营养估计任务，旨在解决从烹饪视频中准确估算餐食营养的问题。通过分析烹饪过程信息，提出V-Nutri框架，提升营养估计效果。**

- **链接: [https://arxiv.org/pdf/2604.11913](https://arxiv.org/pdf/2604.11913)**

> **作者:** Chengkun Yue; Chuanzhi Xu; Jiangpeng He
>
> **备注:** Accepted to the 3rd MetaFood Workshop at CVPR 2026
>
> **摘要:** Nutrition estimation of meals from visual data is an important problem for dietary monitoring and computational health, but existing approaches largely rely on single images of the finally completed dish. This setting is fundamentally limited because many nutritionally relevant ingredients and transformations, such as oils, sauces, and mixed components, become visually ambiguous after cooking, making accurate calorie and macronutrient estimation difficult. In this paper, we investigate whether the cooking process information from egocentric cooking videos can contribute to dish-level nutrition estimation. First, we further manually annotated the HD-EPIC dataset and established the first benchmark for video-based nutrition estimation. Most importantly, we propose V-Nutri, a staged framework that combines Nutrition5K-pretrained visual backbones with a lightweight fusion module that aggregates features from the final dish frame and cooking process keyframes extracted from the egocentric videos. V-Nutri also includes a cooking keyframes selection module, a VideoMamba-based event-detection model that targets ingredient-addition moments. Experiments on the HD-EPIC dataset show that process cues can provide complementary nutritional evidence, improving nutrition estimation under controlled conditions. Our results further indicate that the benefit of process keyframes depends strongly on backbone representation capacity and event detection quality. Our code and annotated dataset is available at this https URL.
>
---
#### [new 062] Cognition-Inspired Dual-Stream Semantic Enhancement for Vision-Based Dynamic Emotion Modeling
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于动态情感建模任务，旨在解决现有方法忽视认知理论的问题。提出DuSE模型，模拟大脑情感处理机制，提升情感识别性能与可解释性。**

- **链接: [https://arxiv.org/pdf/2604.12777](https://arxiv.org/pdf/2604.12777)**

> **作者:** Huanzhen Wang; Ziheng Zhou; Zeng Tao; Aoxing Li; Yingkai Zhao; Yuxuan Lin; Yan Wang; Wenqiang Zhang
>
> **备注:** Accepted by IEEE ICRA 2026
>
> **摘要:** The human brain constructs emotional percepts not by processing facial expressions in isolation, but through a dynamic, hierarchical integration of sensory input with semantic and contextual knowledge. However, existing vision-based dynamic emotion modeling approaches often neglect emotion perception and cognitive theories. To bridge this gap between machine and human emotion perception, we propose cognition-inspired Dual-stream Semantic Enhancement (DuSE). Our model instantiates a dual-stream cognitive architecture. The first stream, a Hierarchical Temporal Prompt Cluster (HTPC), operationalizes the cognitive priming effect. It simulates how linguistic cues pre-sensitize neural pathways, modulating the processing of incoming visual stimuli by aligning textual semantics with fine-grained temporal features of facial dynamics. The second stream, a Latent Semantic Emotion Aggregator (LSEA), computationally models the knowledge integration process, akin to the mechanism described by the Conceptual Act Theory. It aggregates sensory inputs and synthesizes them with learned conceptual knowledge, reflecting the role of the hippocampus and default mode network in constructing a coherent emotional experience. By explicitly modeling these neuro-cognitive mechanisms, DuSE provides a more neurally plausible and robust framework for dynamic facial expression recognition (DFER). Extensive experiments on challenging in-the-wild benchmarks validate our cognition-centric approach, demonstrating that emulating the brain's strategies for emotion processing yields state-of-the-art performance and enhances model interpretability.
>
---
#### [new 063] Evolution-Inspired Sample Competition for Deep Neural Network Optimization
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决深度学习中样本处理不均衡的问题。提出一种受进化启发的样本竞争机制，通过动态调整样本权重提升模型优化效果。**

- **链接: [https://arxiv.org/pdf/2604.12568](https://arxiv.org/pdf/2604.12568)**

> **作者:** Ying Zheng; Yiyi Zhang; Yi Wang; Lap-Pui Chau
>
> **摘要:** Conventional deep network training generally optimizes all samples under a largely uniform learning paradigm, without explicitly modeling the heterogeneous competition among them. Such an oversimplified treatment can lead to several well-known issues, including bias under class imbalance, insufficient learning of hard samples, and the erroneous reinforcement of noisy samples. In this work, we present \textit{Natural Selection} (NS), a novel evolution-inspired optimization method that explicitly incorporates competitive interactions into deep network training. Unlike conventional sample reweighting strategies that rely mainly on predefined heuristics or static criteria, NS estimates the competitive status of each sample in a group-wise context and uses it to adaptively regulate its training contribution. Specifically, NS first assembles multiple samples into a composite image and rescales it to the original input size for model inference. Based on the resulting predictions, a natural selection score is computed for each sample to characterize its relative competitive variation within the constructed group. These scores are then used to dynamically reweight the sample-wise loss, thereby introducing an explicit competition-driven mechanism into the optimization process. In this way, NS provides a simple yet effective means of moving beyond uniform sample treatment and enables more adaptive and balanced model optimization. Extensive experiments on 12 public datasets across four image classification tasks demonstrate the effectiveness of the proposed method. Moreover, NS is compatible with diverse network architectures and does not depend on task-specific assumptions, indicating its strong generality and practical potential. The code will be made publicly available.
>
---
#### [new 064] Task Alignment: A simple and effective proxy for model merging in computer vision
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉领域，解决多任务模型合并问题。针对不同任务需训练解码器的情况，提出任务对齐代理，加速超参数选择并保持性能。**

- **链接: [https://arxiv.org/pdf/2604.12935](https://arxiv.org/pdf/2604.12935)**

> **作者:** Pau de Jorge; César Roberto de Souza; Björn Michele; Mert Bülent Sarıyıldız; Philippe Weinzaepfel; Florent Perronnin; Diane Larlus; Yannis Kalantidis
>
> **摘要:** Efficiently merging several models fine-tuned for different tasks, but stemming from the same pretrained base model, is of great practical interest. Despite extensive prior work, most evaluations of model merging in computer vision are restricted to image classification using CLIP, where different classification datasets define different tasks. In this work, our goal is to make model merging more practical and show its relevance on challenging scenarios beyond this specific setting. In most vision scenarios, different tasks rely on trainable and usually heterogeneous decoders. Differently from previous studies with frozen decoders, where merged models can be evaluated right away, the non-trivial cost of decoder training renders hyperparameter selection based on downstream performance impractical. To address this, we introduce the task alignment proxy, and show how it can be used to speed up hyperparameter selection by orders of magnitude while retaining performance. Equipped with the task alignment proxy, we extend the applicability of model merging to multi-task vision models beyond CLIP-based classification.
>
---
#### [new 065] VideoFlexTok: Flexible-Length Coarse-to-Fine Video Tokenization
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出VideoFlexTok，解决视频编码效率问题。通过可变长度的粗到细token结构，提升生成模型训练效率，支持更长视频生成。**

- **链接: [https://arxiv.org/pdf/2604.12887](https://arxiv.org/pdf/2604.12887)**

> **作者:** Andrei Atanov; Jesse Allardice; Roman Bachmann; Oğuzhan Fatih Kar; R Devon Hjelm; David Griffiths; Peter Fu; Afshin Dehghan; Amir Zamir
>
> **备注:** project page at this https URL
>
> **摘要:** Visual tokenizers map high-dimensional raw pixels into a compressed representation for downstream modeling. Beyond compression, tokenizers dictate what information is preserved and how it is organized. A de facto standard approach to video tokenization is to represent a video as a spatiotemporal 3D grid of tokens, each capturing the corresponding local information in the original signal. This requires the downstream model that consumes the tokens, e.g., a text-to-video model, to learn to predict all low-level details "pixel-by-pixel" irrespective of the video's inherent complexity, leading to high learning complexity. We present VideoFlexTok, which represents videos with a variable-length sequence of tokens structured in a coarse-to-fine manner -- where the first tokens (emergently) capture abstract information, such as semantics and motion, and later tokens add fine-grained details. The generative flow decoder enables realistic video reconstructions from any token count. This representation structure allows adapting the token count according to downstream needs and encoding videos longer than the baselines with the same budget. We evaluate VideoFlexTok on class- and text-to-video generative tasks and show that it leads to more efficient training compared to 3D grid tokens, e.g., achieving comparable generation quality (gFVD and ViCLIP Score) with a 5x smaller model (1.1B vs 5.2B). Finally, we demonstrate how VideoFlexTok can enable long video generation without prohibitive computational cost by training a text-to-video model on 10-second 81-frame videos with only 672 tokens, 8x fewer than a comparable 3D grid tokenizer.
>
---
#### [new 066] Representing 3D Faces with Learnable B-Spline Volumes
- **分类: cs.CV**

- **简介: 该论文提出CUBE，一种结合B-spline体积与学习特征的3D人脸表示方法，用于解决3D人脸重建与配准问题。通过控制特征实现高精度表面重建与局部编辑。**

- **链接: [https://arxiv.org/pdf/2604.12894](https://arxiv.org/pdf/2604.12894)**

> **作者:** Prashanth Chandran; Daoye Wang; Timo Bolkart
>
> **备注:** Accepted to CVPR 2026 (Highlight)
>
> **摘要:** We present CUBE (Control-based Unified B-spline Encoding), a new geometric representation for human faces that combines B-spline volumes with learned features, and demonstrate its use as a decoder for 3D scan registration and monocular 3D face reconstruction. Unlike existing B-spline representations with 3D control points, CUBE is parametrized by a lattice (e.g., 8 x 8 x 8) of high-dimensional control features, increasing the model's expressivity. These features define a continuous, two-stage mapping from a 3D parametric domain to 3D Euclidean space via an intermediate feature space. First, high-dimensional control features are locally blended using the B-spline bases, yielding a high-dimensional feature vector whose first three values define a 3D base mesh. A small MLP then processes this feature vector to predict a residual displacement from the base shape, yielding the final refined 3D coordinates. To reconstruct 3D surfaces in dense semantic correspondence, CUBE is queried at 3D coordinates sampled from a fixed template mesh. Crucially, CUBE retains the local support property of traditional B-spline representations, enabling local surface editing by updating individual control features. We demonstrate the strengths of this representation by training transformer-based encoders to predict CUBE's control features from unstructured point clouds and monocular images, achieving state-of-the-art scan registration results compared to recent baselines.
>
---
#### [new 067] Spatial-Spectral Adaptive Fidelity and Noise Prior Reduction Guided Hyperspectral Image Denoising
- **分类: cs.CV; math.NA**

- **简介: 该论文属于高光谱图像去噪任务，旨在解决数据保真与噪声先验建模间的平衡问题。提出融合噪声先验和自适应保真项的框架，有效去除混合噪声。**

- **链接: [https://arxiv.org/pdf/2604.12600](https://arxiv.org/pdf/2604.12600)**

> **作者:** Xuelin Xie; Xiliang Lu; Zhengshan Wang; Yang Zhang; Long Chen
>
> **摘要:** The core challenge of hyperspectral image denoising is striking the right balance between data fidelity and noise prior modeling. Most existing methods place too much emphasis on the intrinsic priors of the image while overlooking diverse noise assumptions and the dynamic trade-off between fidelity and priors. To address these issues, we propose a denoising framework that integrates noise prior reduction and a spatial-spectral adaptive fidelity term. This framework considers comprehensive noise priors with fewer parameters and introduces an adaptive weight tensor to dynamically balance the fidelity and prior regularization terms. Within this framework, we further develop a fast and robust pixel-wise model combined with the representative coefficient total variation regularizer to accurately remove mixed noise in HSIs. The proposed method not only efficiently handles various types of noise but also accurately captures the spectral low-rank structure and local smoothness of HSIs. An efficient optimization algorithm based on the alternating direction method of multipliers is designed to ensure stable and fast convergence. Extensive experiments on simulated and real-world datasets demonstrate that the proposed model achieves superior denoising performance while maintaining competitive computational efficiency.
>
---
#### [new 068] Representation geometry shapes task performance in vision-language modeling for CT enterography
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视觉-语言模型在CT肠造影中的任务性能，解决如何选择最佳表示方法的问题。通过对比不同聚合策略和编码方式，提出有效提升分类与报告生成的方案。**

- **链接: [https://arxiv.org/pdf/2604.13021](https://arxiv.org/pdf/2604.13021)**

> **作者:** Cristian Minoccheri; Emily Wittrup; Kayvan Najarian; Ryan Stidham
>
> **摘要:** Computed tomography (CT) enterography is a primary imaging modality for assessing inflammatory bowel disease (IBD), yet the representational choices that best support automated analysis of this modality are unknown. We present the first study of vision-language transfer learning on abdominal CT enterography and identify two main findings. First, mean pooling of slice embeddings gives better categorical disease assessment (59.2\% three-class accuracy), whereas attention pooling gives better cross-modal retrieval (0.235 text-to-image MRR). This pattern holds across all LoRA configurations tested and suggests that the two aggregators emphasize different properties of the learned representation. Second, per-slice tissue contrast matters more than broader spatial coverage: multi-window RGB encoding, which maps complementary Hounsfield Unit windows to RGB channels, outperforms all strategies that increase spatial coverage through multiplanar sampling, and in this setting adding coronal and sagittal views reduces classification performance. For report generation, fine-tuning without retrieval context yields within-1 severity accuracy at the prevalence-matched chance level (70.4\% vs.\ 71\% random), suggesting little learned ordering beyond the class distribution. Retrieval-augmented generation (RAG) improves this across all configurations, scoring 7--14 percentage points above the chance baseline and improving ordinal MAE from 0.98 to 0.80--0.89. A three-teacher pseudolabel framework enables all comparisons without expert annotations. Together, these findings provide the first baselines for this underexplored modality and offer practical guidance for building vision-language systems for volumetric medical imaging.
>
---
#### [new 069] Visual Preference Optimization with Rubric Rewards
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉偏好优化任务，解决多模态任务中细粒度视觉推理的偏好数据不足问题。通过实例特定评分标准提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.13029](https://arxiv.org/pdf/2604.13029)**

> **作者:** Ya-Qi Yu; Fangyu Hong; Xiangyang Qu; Hao Wang; Gaojie Wu; Qiaoyu Luo; Nuo Xu; Huixin Wang; Wuheng Xu; Yongxin Liao; Zihao Chen; Haonan Li; Ziming Li; Dezhi Peng; Minghui Liao; Jihao Wu; Haoyu Ren; Dandan Tu
>
> **摘要:** The effectiveness of Direct Preference Optimization (DPO) depends on preference data that reflect the quality differences that matter in multimodal tasks. Existing pipelines often rely on off-policy perturbations or coarse outcome-based signals, which are not well suited to fine-grained visual reasoning. We propose rDPO, a preference optimization framework based on instance-specific rubrics. For each image-instruction pair, we create a checklist-style rubric of essential and additional criteria to score responses from any possible policies. The instruction-rubric pool is built offline and reused during the construction of on-policy data. On public reward modeling benchmarks, rubric-based prompting massively improves a 30B-A3B judge and brings it close to GPT-5.4. On public downstream benchmarks, rubric-based filtering raises the macro average to 82.69, whereas outcome-based filtering drops it to 75.82 from 81.14. When evaluating scalability on a comprehensive benchmark, rDPO achieves 61.01, markedly outperforming the style-constrained baseline (52.36) and surpassing the 59.48 base model. Together, these results show that visual preference optimization benefits from combining on-policy data construction with instance-specific criterion-level feedback.
>
---
#### [new 070] PR-MaGIC: Prompt Refinement Via Mask Decoder Gradient Flow For In-Context Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像分割任务，旨在解决因提示不一致导致的分割质量下降问题。提出PR-MaGIC框架，通过梯度流优化提示，无需额外训练即可提升分割效果。**

- **链接: [https://arxiv.org/pdf/2604.12113](https://arxiv.org/pdf/2604.12113)**

> **作者:** Minjae Lee; Sungwoo Hur; Soojin Hwang; Won Hwa Kim
>
> **摘要:** Visual Foundation Models (VFMs) such as the Segment Anything Model (SAM) have significantly advanced broad use of image segmentation. However, SAM and its variants necessitate substantial manual effort for prompt generation and additional training for specific applications. Recent approaches address these limitations by integrating SAM into in-context (one/few shot) segmentation, enabling auto-prompting through semantic alignment between query and support images. Despite these efforts, they still generate sub-optimal prompts that degrade segmentation quality due to visual inconsistencies between support and query images. To tackle this limitation, we introduce PR-MaGIC (Prompt Refinement via Mask Decoder Gradient Flow for In-Context Segmentation), a training-free test-time framework that refines prompts via gradient flow derived from SAM's mask decoder. PR-MaGIC seamlessly integrates into in-context segmentation frameworks, being theoretically grounded yet practically stabilized through a simple top-1 selection strategy that ensures robust performance across samples. Extensive evaluations demonstrate that PR-MaGIC consistently improves segmentation quality across various benchmarks, effectively mitigating inadequate prompts without requiring additional training or architectural modifications.
>
---
#### [new 071] Lyra 2.0: Explorable Generative 3D Worlds
- **分类: cs.CV**

- **简介: 该论文提出Lyra 2.0，解决3D场景生成中的长期一致性问题，通过改进视频生成以实现可探索的高质量3D世界。**

- **链接: [https://arxiv.org/pdf/2604.13036](https://arxiv.org/pdf/2604.13036)**

> **作者:** Tianchang Shen; Sherwin Bahmani; Kai He; Sangeetha Grama Srinivasan; Tianshi Cao; Jiawei Ren; Ruilong Li; Zian Wang; Nicholas Sharp; Zan Gojcic; Sanja Fidler; Jiahui Huang; Huan Ling; Jun Gao; Xuanchi Ren
>
> **备注:** Project Page: this https URL
>
> **摘要:** Recent advances in video generation enable a new paradigm for 3D scene creation: generating camera-controlled videos that simulate scene walkthroughs, then lifting them to 3D via feed-forward reconstruction techniques. This generative reconstruction approach combines the visual fidelity and creative capacity of video models with 3D outputs ready for real-time rendering and simulation. Scaling to large, complex environments requires 3D-consistent video generation over long camera trajectories with large viewpoint changes and location revisits, a setting where current video models degrade quickly. Existing methods for long-horizon generation are fundamentally limited by two forms of degradation: spatial forgetting and temporal drifting. As exploration proceeds, previously observed regions fall outside the model's temporal context, forcing the model to hallucinate structures when revisited. Meanwhile, autoregressive generation accumulates small synthesis errors over time, gradually distorting scene appearance and geometry. We present Lyra 2.0, a framework for generating persistent, explorable 3D worlds at scale. To address spatial forgetting, we maintain per-frame 3D geometry and use it solely for information routing -- retrieving relevant past frames and establishing dense correspondences with the target viewpoints -- while relying on the generative prior for appearance synthesis. To address temporal drifting, we train with self-augmented histories that expose the model to its own degraded outputs, teaching it to correct drift rather than propagate it. Together, these enable substantially longer and 3D-consistent video trajectories, which we leverage to fine-tune feed-forward reconstruction models that reliably recover high-quality 3D scenes.
>
---
#### [new 072] TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言预训练任务，旨在解决图像块与文本嵌入对齐不足的问题。通过改进预训练方法，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.12012](https://arxiv.org/pdf/2604.12012)**

> **作者:** Bingyi Cao; Koert Chen; Kevis-Kokitsi Maninis; Kaifeng Chen; Arjun Karpur; Ye Xia; Sahil Dua; Tanmaya Dabral; Guangxing Han; Bohyung Han; Joshua Ainslie; Alex Bewley; Mithun Jacob; René Wagner; Washington Ramos; Krzysztof Choromanski; Mojtaba Seyedhosseini; Howard Zhou; André Araujo
>
> **备注:** CVPR2026 camera-ready + appendix
>
> **摘要:** Recent progress in vision-language pretraining has enabled significant improvements to many downstream computer vision applications, such as classification, retrieval, segmentation and depth prediction. However, a fundamental capability that these models still struggle with is aligning dense patch representations with text embeddings of corresponding concepts. In this work, we investigate this critical issue and propose novel techniques to enhance this capability in foundational vision-language models. First, we reveal that a patch-level distillation procedure significantly boosts dense patch-text alignment -- surprisingly, the patch-text alignment of the distilled student model strongly surpasses that of the teacher model. This observation inspires us to consider modifications to pretraining recipes, leading us to propose iBOT++, an upgrade to the commonly-used iBOT masked image objective, where unmasked tokens also contribute directly to the loss. This dramatically enhances patch-text alignment of pretrained models. Additionally, to improve vision-language pretraining efficiency and effectiveness, we modify the exponential moving average setup in the learning recipe, and introduce a caption sampling strategy to benefit from synthetic captions at different granularities. Combining these components, we develop TIPSv2, a new family of image-text encoder models suitable for a wide range of downstream applications. Through comprehensive experiments on 9 tasks and 20 datasets, we demonstrate strong performance, generally on par with or better than recent vision encoder models. Code and models are released via our project page at this https URL .
>
---
#### [new 073] Combating Pattern and Content Bias: Adversarial Feature Learning for Generalized AI-Generated Image Detection
- **分类: cs.CV**

- **简介: 该论文属于图像检测任务，旨在解决生成图像检测中的模式与内容偏差问题。通过提出MAFL框架，提升模型跨模型的泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.12353](https://arxiv.org/pdf/2604.12353)**

> **作者:** Haifeng Zhang; Qinghui He; Xiuli Bi; Bo Liu; Chi-Man Pun; Bin Xiao
>
> **摘要:** In recent years, the rapid development of generative artificial intelligence technology has significantly lowered the barrier to creating high-quality fake images, posing a serious challenge to information authenticity and credibility. Existing generated image detection methods typically enhance generalization through model architecture or network design. However, their generalization performance remains susceptible to data bias, as the training data may drive models to fit specific generative patterns and content rather than the common features shared by images from different generative models (asymmetric bias learning). To address this issue, we propose a Multi-dimensional Adversarial Feature Learning (MAFL) framework. The framework adopts a pretrained multimodal image encoder as the feature extraction backbone, constructs a real-fake feature learning network, and designs an adversarial bias-learning branch equipped with a multi-dimensional adversarial loss, forming an adversarial training mechanism between authenticity-discriminative feature learning and bias feature learning. By suppressing generation-pattern and content biases, MAFL guides the model to focus on the generative features shared across different generative models, thereby effectively capturing the fundamental differences between real and generated images, enhancing cross-model generalization, and substantially reducing the reliance on large-scale training data. Through extensive experimental validation, our method outperforms existing state-of-the-art approaches by 10.89% in accuracy and 8.57% in Average Precision (AP). Notably, even when trained with only 320 images, it can still achieve over 80% detection accuracy on public datasets.
>
---
#### [new 074] Cell Instance Segmentation via Multi-Task Image-to-Image Schrödinger Bridge
- **分类: cs.CV**

- **简介: 该论文属于细胞实例分割任务，旨在解决传统方法依赖后处理和约束不足的问题。提出多任务图像到图像的薛定谔桥框架，通过边界监督实现稳定分割。**

- **链接: [https://arxiv.org/pdf/2604.12318](https://arxiv.org/pdf/2604.12318)**

> **作者:** Hayato Inoue; Shota Harada; Shumpei Takezaki; Ryoma Bise
>
> **摘要:** Existing cell instance segmentation pipelines typically combine deterministic predictions with post-processing, which imposes limited explicit constraints on the global structure of instance masks. In this work, we propose a multi-task image-to-image Schrödinger Bridge framework that formulates instance segmentation as a distribution-based image-to-image generation problem. Boundary-aware supervision is integrated through a reverse distance map, and deterministic inference is employed to produce stable predictions. Experimental results on the PanNuke dataset demonstrate that the proposed method achieves competitive or superior performance without relying on SAM pre-training or additional post-processing. Additional results on the MoNuSeg dataset show robustness under limited training data. These findings indicate that Schrödinger Bridge-based image-to-image generation provides an effective framework for cell instance segmentation.
>
---
#### [new 075] Efficient Semantic Image Communication for Traffic Monitoring at the Edge
- **分类: cs.CV; cs.AI; cs.NI**

- **简介: 该论文属于图像通信任务，解决交通监控中高传输成本问题。提出两种管道MMSD和SAMR，通过语义压缩减少数据量，同时保留关键视觉信息。**

- **链接: [https://arxiv.org/pdf/2604.12622](https://arxiv.org/pdf/2604.12622)**

> **作者:** Damir Assylbek; Nurmukhammed Aitymbetov; Marko Ristin; Dimitrios Zorbas
>
> **摘要:** Many visual monitoring systems operate under strict communication constraints, where transmitting full-resolution images is impractical and often unnecessary. In such settings, visual data is often used for object presence, spatial relationships, and scene context rather than exact pixel fidelity. This paper presents two semantic image communication pipelines for traffic monitoring, MMSD and SAMR, that reduce transmission cost while preserving meaningful visual information. MMSD (Multi-Modal Semantic Decomposition) targets very high compression together with data confidentiality, since sensitive pixel content is not transmitted. It replaces the original image with compact semantic representations, namely segmentation maps, edge maps, and textual descriptions, and reconstructs the scene at the receiver using a diffusion-based generative model. SAMR (Semantic-Aware Masking Reconstruction) targets higher visual quality while maintaining strong compression. It selectively suppresses non-critical image regions according to semantic importance before standard JPEG encoding and restores the missing content at the receiver through generative inpainting. Both designs follow an asymmetric sender-receiver architecture, where lightweight processing is performed at the edge and computationally intensive reconstruction is offloaded to the server. On a Raspberry Pi~5, the edge-side processing time is about 15s for MMSD and 9s for SAMR. Experimental results show average transmitted-data reductions of 99% for MMSD and 99.1% for SAMR. In addition, MMSD achieves lower payload size than the recent SPIC baseline while preserving strong semantic consistency, whereas SAMR provides a better quality-compression trade-off than standard JPEG and SQ-GAN under comparable operating conditions.
>
---
#### [new 076] Hypergraph-State Collaborative Reasoning for Multi-Object Tracking
- **分类: cs.CV**

- **简介: 该论文属于多目标跟踪任务，旨在解决运动估计的不稳定和遮挡问题。通过设计HyperSSM框架，结合超图与状态空间模型，提升轨迹的稳定性和连续性。**

- **链接: [https://arxiv.org/pdf/2604.12665](https://arxiv.org/pdf/2604.12665)**

> **作者:** Zikai Song; Junqing Yu; Yi-Ping Phoebe Chen; Wei Yang; Xinchao Wang
>
> **摘要:** Motion reasoning serves as the cornerstone of multi-object tracking (MOT), as it enables consistent association of targets across frames. However, existing motion estimation approaches face two major limitations: (1) instability caused by noisy or probabilistic predictions, and (2) vulnerability under occlusion, where trajectories often fragment once visual cues disappear. To overcome these issues, we propose a collaborative reasoning framework that enhances motion estimation through joint inference among multiple correlated objects. By allowing objects with similar motion states to mutually constrain and refine each other, our framework stabilizes noisy trajectories and infers plausible motion continuity even when target is occluded. To realize this concept, we design HyperSSM, an architecture that integrates Hypergraph computation and a State Space Model (SSM) for unified spatial-temporal reasoning. The Hypergraph module captures spatial motion correlations through dynamic hyperedges, while the SSM enforces temporal smoothness via structured state transitions. This synergistic design enables simultaneous optimization of spatial consensus and temporal coherence, resulting in robust and stable motion estimation. Extensive experiments on four mainstream and diverse benchmarks(MOT17, MOT20, DanceTrack, and SportsMOT) covering various motion patterns and scene complexities, demonstrate that our approach achieves state-of-the-art performance across a wide range of tracking scenarios.
>
---
#### [new 077] CLASP: Class-Adaptive Layer Fusion and Dual-Stage Pruning for Multimodal Large Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态大语言模型优化任务，旨在解决视觉token冗余导致的计算开销问题。提出CLASP框架，通过类自适应层融合与双阶段剪枝实现高效token缩减。**

- **链接: [https://arxiv.org/pdf/2604.12767](https://arxiv.org/pdf/2604.12767)**

> **作者:** Yunkai Dang; Yizhu Jiang; Yifan Jiang; Qi Fan; Yinghuan Shi; Wenbin Li; Yang Gao
>
> **摘要:** Multimodal Large Language Models (MLLMs) suffer from substantial computational overhead due to the high redundancy in visual token sequences. Existing approaches typically address this issue using single-layer Vision Transformer (ViT) features and static pruning strategies. However, such fixed configurations are often brittle under diverse instructions. To overcome these limitations, we propose CLASP, a plug-and-play token reduction framework based on class-adaptive layer fusion and dual-stage pruning. Specifically, CLASP first constructs category-specific visual representations through multi-layer vision feature fusion. It then performs dual-stage pruning, allocating the token budget between attention-salient pivot tokens for relevance and redundancy-aware completion tokens for coverage. Through class-adaptive pruning, CLASP enables prompt-conditioned feature fusion and budget allocation, allowing aggressive yet robust visual token reduction. Extensive experiments demonstrate that CLASP consistently outperforms existing methods across a wide range of benchmarks, pruning ratios, and MLLM architectures. Code will be available at this https URL.
>
---
#### [new 078] SceneCritic: A Symbolic Evaluator for 3D Indoor Scene Synthesis
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出SceneCritic，用于评估3D室内场景生成，解决传统评估方法不稳定的问题。通过符号化约束和迭代测试，提升场景布局的语义与几何一致性。**

- **链接: [https://arxiv.org/pdf/2604.13035](https://arxiv.org/pdf/2604.13035)**

> **作者:** Kathakoli Sengupta; Kai Ao; Paola Cascante-Bonilla
>
> **备注:** Project Page: this https URL
>
> **摘要:** Large Language Models (LLMs) and Vision-Language Models (VLMs) increasingly generate indoor scenes through intermediate structures such as layouts and scene graphs, yet evaluation still relies on LLM or VLM judges that score rendered views, making judgments sensitive to viewpoint, prompt phrasing, and hallucination. When the evaluator is unstable, it becomes difficult to determine whether a model has produced a spatially plausible scene or whether the output score reflects the choice of viewpoint, rendering, or prompt. We introduce SceneCritic, a symbolic evaluator for floor-plan-level layouts. SceneCritic's constraints are grounded in SceneOnto, a structured spatial ontology we construct by aggregating indoor scene priors from 3D-FRONT, ScanNet, and Visual Genome. SceneOnto traverses this ontology to jointly verify semantic, orientation, and geometric coherence across object relationships, providing object-level and relationship-level assessments that identify specific violations and successful placements. Furthermore, we pair SceneCritic with an iterative refinement test bed that probes how models build and revise spatial structure under different critic modalities: a rule-based critic using collision constraints as feedback, an LLM critic operating on the layout as text, and a VLM critic operating on rendered observations. Through extensive experiments, we show that (a) SceneCritic aligns substantially better with human judgments than VLM-based evaluators, (b) text-only LLMs can outperform VLMs on semantic layout quality, and (c) image-based VLM refinement is the most effective critic modality for semantic and orientation correction.
>
---
#### [new 079] Towards Long-horizon Agentic Multimodal Search
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态搜索任务，旨在解决长序列搜索中信息管理与视觉信号丢失问题。提出LMM-Searcher框架，通过文件化视觉表示和渐进加载策略提升搜索性能。**

- **链接: [https://arxiv.org/pdf/2604.12890](https://arxiv.org/pdf/2604.12890)**

> **作者:** Yifan Du; Zikang Liu; Jinbiao Peng; Jie Wu; Junyi Li; Jinyang Li; Wayne Xin Zhao; Ji-Rong Wen
>
> **摘要:** Multimodal deep search agents have shown great potential in solving complex tasks by iteratively collecting textual and visual evidence. However, managing the heterogeneous information and high token costs associated with multimodal inputs over long horizons remains a critical challenge, as existing methods often suffer from context explosion or the loss of crucial visual signals. To address this, we propose a novel Long-horizon MultiModal deep search framework, named LMM-Searcher, centered on a file-based visual representation mechanism. By offloading visual assets to an external file system and mapping them to lightweight textual identifiers (UIDs), our approach mitigates context overhead while preserving multimodal information for future access. We equip the agent with a tailored fetch-image tool, enabling a progressive, on-demand visual loading strategy for active perception. Furthermore, we introduce a data synthesis pipeline designed to generate queries requiring complex cross-modal multi-hop reasoning. Using this pipeline, we distill 12K high-quality trajectories to fine-tune Qwen3-VL-Thinking-30A3B into a specialized multimodal deep search agent. Extensive experiments across four benchmarks demonstrate that our method successfully scales to 100-turn search horizons, achieving state-of-the-art performance among open-source models on challenging long-horizon benchmarks like MM-BrowseComp and MMSearch-Plus, while also exhibiting strong generalizability across different base models. Our code will be released in this https URL.
>
---
#### [new 080] Image-to-Image Translation Framework Embedded with Rotation Symmetry Priors
- **分类: cs.CV**

- **简介: 该论文属于图像到图像翻译任务，旨在解决无配对数据下的翻译效果问题。通过引入旋转对称性先验和自适应变换卷积，提升模型的对称性保持能力与生成质量。**

- **链接: [https://arxiv.org/pdf/2604.12805](https://arxiv.org/pdf/2604.12805)**

> **作者:** Feiyu Tan; Heran Yang; Qihong Duan; Kai Ye; Qi Xie; Deyu Meng
>
> **备注:** 17 pages, 8 figures, submiting to TPAMI
>
> **摘要:** Image-to-image translation (I2I) is a fundamental task in computer vision, focused on mapping an input image from a source domain to a corresponding image in a target domain while preserving domain-invariant features and adapting domain-specific attributes. Despite the remarkable success of deep learning-based I2I approaches, the lack of paired data and unsupervised learning framework still hinder their effectiveness. In this work, we address the challenge by incorporating transformation symmetry priors into image-to-image translation networks. Specifically, we introduce rotation group equivariant convolutions to achieve rotation equivariant I2I framework, a novel contribution, to the best of our knowledge, along this research direction. This design ensures the preservation of rotation symmetry, one of the most intrinsic and domain-invariant properties of natural and scientific images, throughout the network. Furthermore, we conduct a systematic study on image symmetry priors on real dataset and propose a novel transformation learnable equivariant convolutions (TL-Conv) that adaptively learns transformation groups, enhancing symmetry preservation across diverse datasets. We also provide a theoretical analysis of the equivariance error of TL-Conv, proving that it maintains exact equivariance in continuous domains and provide a bound for the error in discrete cases. Through extensive experiments across a range of I2I tasks, we validate the effectiveness and superior performance of our approach, highlighting the potential of equivariant networks in enhancing generation quality and its broad applicability. Our code is available at this https URL
>
---
#### [new 081] BarbieGait: An Identity-Consistent Synthetic Human Dataset with Versatile Cloth-Changing for Gait Recognition
- **分类: cs.CV**

- **简介: 该论文属于步态识别任务，旨在解决衣物变化带来的识别挑战。提出BarbieGait数据集和GaitCLIF模型，以提升跨衣物场景的识别性能。**

- **链接: [https://arxiv.org/pdf/2604.12221](https://arxiv.org/pdf/2604.12221)**

> **作者:** Qingyuan Cai; Saihui Hou; Xuecai Hu; Yongzhen Huang
>
> **备注:** CVPR 2026, Project Page: this https URL
>
> **摘要:** Gait recognition, as a reliable biometric technology, has seen rapid development in recent years while it faces significant challenges caused by diverse clothing styles in the real world. This paper introduces BarbieGait, a synthetic gait dataset where real-world subjects are uniquely mapped into a virtual engine to simulate extensive clothing changes while preserving their gait identity information. As a pioneering work, BarbieGait provides a controllable gait data generation method, enabling the production of large datasets to validate cross-clothing issues that are difficult to verify with real-world data. However, the diversity of clothing increases intra-class variance and makes one of the biggest challenges to learning cloth-invariant features under varying clothing conditions. Therefore, we propose GaitCLIF (Gait-oriented CLoth-Invariant Feature) as a robust baseline model for cross-clothing gait recognition. Through extensive experiments, we validate that our method significantly improves cross-clothing performance on BarbieGait and the existing popular gait benchmarks. We believe that BarbieGait, with its extensive cross-clothing gait data, will further advance the capabilities of gait recognition in cross-clothing scenarios and promote progress in related research.
>
---
#### [new 082] AbdomenGen: Sequential Volume-Conditioned Diffusion Framework for Abdominal Anatomy Generation
- **分类: cs.CV**

- **简介: 该论文属于医学图像生成任务，旨在解决腹部解剖结构可控生成的问题。提出AbdomenGen框架，通过VCS实现器官体积的可解释调控，提升几何精度与多器官独立控制能力。**

- **链接: [https://arxiv.org/pdf/2604.12969](https://arxiv.org/pdf/2604.12969)**

> **作者:** Yubraj Bhandari; Lavsen Dahal; Paul Segars; Joseph Y. Lo
>
> **摘要:** Computational phantoms are widely used in medical imaging research, yet current systems to generate controlled, clinically meaningful anatomical variations remain limited. We present AbdomenGen, a sequential volume-conditioned diffusion framework for controllable abdominal anatomy generation. We introduce the \textbf{Volume Control Scalar (VCS)}, a standardized residual that decouples organ size from body habitus, enabling interpretable volume modulation. Organ masks are synthesized sequentially, conditioning on the body mask and previously generated structures to preserve global anatomical coherence while supporting independent, multi-organ control. Across 11 abdominal organs, the proposed framework achieves strong geometric fidelity (e.g., liver dice $0.83 \pm 0.05$), stable single-organ calibration over $[-3,+3]$ VCS, and disentangled multi-organ modulation. To showcase clinical utility with a hepatomegaly cohort selected from MERLIN, Wasserstein-based VCS selection reduces distributional distance of training data by 73.6\% . These results demonstrate calibrated, distribution-aware anatomical generation suitable for controllable abdominal phantom construction and simulation studies.
>
---
#### [new 083] HTDC: Hesitation-Triggered Differential Calibration for Mitigating Hallucination in Large Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决模型幻觉问题。通过识别层间犹豫信号，提出HTDC框架，在必要时激活校准以减少幻觉，同时保持推理效率。**

- **链接: [https://arxiv.org/pdf/2604.12115](https://arxiv.org/pdf/2604.12115)**

> **作者:** Xinyun Liu
>
> **备注:** 10 pages, 4 figures, 6 tables
>
> **摘要:** Large vision-language models (LVLMs) achieve strong multimodal performance, but still suffer from hallucinations caused by unstable visual grounding and over-reliance on language priors. Existing training-free decoding methods typically apply calibration at every decoding step, introducing unnecessary computation and potentially disrupting stable predictions. We address this problem by identifying layer-wise hesitation, a simple signal of grounding instability reflected by fluctuations in token preference across intermediate layers. Based on this observation, we propose Hesitation-Triggered Differential Calibration (HTDC), a training-free decoding framework that preserves standard full-branch inference and activates calibration only at hesitation-prone steps. When triggered, HTDC contrasts the full branch with two lightweight probes, a visual-nullification probe and a semantic-nullification probe, to suppress hallucination-prone candidates while avoiding unnecessary intervention on stable steps. Experiments on representative hallucination benchmarks show that HTDC consistently reduces hallucinations while maintaining strong task accuracy, achieving a favorable trade-off between effectiveness and computational overhead.
>
---
#### [new 084] Pi-HOC: Pairwise 3D Human-Object Contact Estimation
- **分类: cs.CV**

- **简介: 该论文提出Pi-HOC，解决多人体与物体的3D接触估计任务。针对现有方法在多人场景和效率上的不足，提出单次处理、实例感知的框架，提升准确性和速度。**

- **链接: [https://arxiv.org/pdf/2604.12923](https://arxiv.org/pdf/2604.12923)**

> **作者:** Sravan Chittupalli; Ayush Jain; Dong Huang
>
> **摘要:** Resolving real-world human-object interactions in images is a many-to-many challenge, in which disentangling fine-grained concurrent physical contact is particularly difficult. Existing semantic contact estimation methods are either limited to single-human settings or require object geometries (e.g., meshes) in addition to the input image. Current state-of-the-art leverages powerful VLM for category-level semantics but struggles with multi-human scenarios and scales poorly in inference. We introduce Pi-HOC, a single-pass, instance-aware framework for dense 3D semantic contact prediction of all human-object pairs. Pi-HOC detects instances, creates dedicated human-object (HO) tokens for each pair, and refines them using an InteractionFormer. A SAM-based decoder then predicts dense contact on SMPL human meshes for each human-object pair. On the MMHOI and DAMON datasets, Pi-HOC significantly improves accuracy and localization over state-of-the-art methods while achieving 20x higher throughput. We further demonstrate that predicted contacts improve SAM-3D image-to-mesh reconstruction via a test-time optimization algorithm and enable referential contact prediction from language queries without additional training.
>
---
#### [new 085] MODIX: A Training-Free Multimodal Information-Driven Positional Index Scaling for Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型任务，解决 positional encoding 效率低的问题。提出 MODIX 框架，通过信息密度动态调整位置索引，提升多模态推理效果。**

- **链接: [https://arxiv.org/pdf/2604.12537](https://arxiv.org/pdf/2604.12537)**

> **作者:** Ruoxiang Huang; Zhen Yuan
>
> **备注:** Accepted by CVPR 2026 (Highlight). 10 pages, 2 figures, 5 tables
>
> **摘要:** Vision-Language Models (VLMs) have achieved remarkable progress in multimodal understanding, yet their positional encoding mechanisms remain suboptimal. Existing approaches uniformly assign positional indices to all tokens, overlooking variations in information density within and across modalities, which leads to inefficient attention allocation where redundant visual regions dominate while informative content is underrepresented. We identify positional granularity as an implicit resource and propose MODIX (Multimodal Information-Driven Positional IndeX Scaling), a training-free framework that dynamically adapts positional strides based on modality-specific contributions. MODIX jointly models intra-modal density via covariance-based entropy and inter-modal interaction via cross-modal alignment to derive unified scores, which rescale positional indices to allocate finer granularity to informative modalities while compressing redundant ones, without requiring any modification to model parameters or architecture. Experiments across diverse architectures and benchmarks demonstrate that MODIX consistently improves multimodal reasoning and adaptively reallocates attention according to task-dependent information distributions, suggesting that positional encoding should be treated as an adaptive resource in Transformers for multimodal sequence modeling.
>
---
#### [new 086] Beyond Perception Errors: Semantic Fixation in Large Vision-Language Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究视觉语言模型的语义固化问题，探讨其在规则映射中的偏差。通过设计基准测试，分析模型对标准与反向规则的处理差异，提出干预方法提升模型灵活性。任务属于视觉语言理解与模型优化。**

- **链接: [https://arxiv.org/pdf/2604.12119](https://arxiv.org/pdf/2604.12119)**

> **作者:** Md Tanvirul Alam
>
> **摘要:** Large vision-language models (VLMs) often rely on familiar semantic priors, but existing evaluations do not cleanly separate perception failures from rule-mapping failures. We study this behavior as semantic fixation: preserving a default interpretation even when the prompt specifies an alternative, equally valid mapping. To isolate this effect, we introduce VLM-Fix, a controlled benchmark over four abstract strategy games that evaluates identical terminal board states under paired standard and inverse rule formulations. Across 14 open and closed VLMs, accuracy consistently favors standard rules, revealing a robust semantic-fixation gap. Prompt interventions support this mechanism: neutral alias prompts substantially narrow the inverse-rule gap, while semantically loaded aliases reopen it. Post-training is strongly rule-aligned: training on one rule improves same-rule transfer but hurts opposite-rule transfer, while joint-rule training improves broader transfer. To test external validity beyond synthetic games, we evaluate analogous defamiliarization interventions on VLMBias and observe the same qualitative pattern. Finally, late-layer activation steering partially recovers degraded performance, indicating that semantic-fixation errors are at least partly editable in late representations. Project page, code, and dataset available at this https URL.
>
---
#### [new 087] T2I-BiasBench: A Multi-Metric Framework for Auditing Demographic and Cultural Bias in Text-to-Image Models
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成模型的偏见评估任务，旨在检测并量化模型中的性别、文化等偏差。工作包括构建多维度评估框架，并对多个模型进行实验分析。**

- **链接: [https://arxiv.org/pdf/2604.12481](https://arxiv.org/pdf/2604.12481)**

> **作者:** Nihal Jaiswal; Siddhartha Arjaria; Gyanendra Chaubey; Ankush Kumar; Aditya Singh; Anchal Chaurasiya
>
> **摘要:** Text-to-image (T2I) generative models achieve impressive visual fidelity but inherit and amplify demographic imbalances and cultural biases embedded in training data. We introduce T2I-BiasBench, a unified evaluation framework of thirteen complementary metrics that jointly captures demographic bias, element omission, and cultural collapse in diffusion models - the first framework to address all three dimensions simultaneously. We evaluate three open-source models - Stable Diffusion v1.5, BK-SDM Base, and Koala Lightning - against Gemini 2.5 Flash (RLHF-aligned) as a reference baseline. The benchmark comprises 1,574 generated images across five structured prompt categories. T2I-BiasBench integrates six established metrics with seven additional measures: four newly proposed (Composite Bias Score, Grounded Missing Rate, Implicit Element Missing Rate, Cultural Accuracy Ratio) and three adapted (Hallucination Score, Vendi Score, CLIP Proxy Score). Three key findings emerge: (1) Stable Diffusion v1.5 and BK-SDM exhibit bias amplification (>1.0) in beauty-related prompts; (2) contextual constraints such as surgical PPE substantially attenuate professional-role gender bias (Doctor CBS = 0.06 for SD v1.5); and (3) all models, including RLHF-aligned Gemini, collapse to a narrow set of cultural representations (CAS: 0.54-1.00), confirming that alignment techniques do not resolve cultural coverage gaps. T2I-BiasBench is publicly released to support standardized, fine-grained bias evaluation of generative models. The project page is available at: this https URL
>
---
#### [new 088] Listening Deepfake Detection: A New Perspective Beyond Speaking-Centric Forgery Analysis
- **分类: cs.CV; cs.MM**

- **简介: 该论文提出“倾听深度伪造检测”任务，解决真实交互中伪造倾听状态的检测问题。构建了ListenForge数据集，提出MANet模型，提升多模态伪造分析能力。**

- **链接: [https://arxiv.org/pdf/2604.12650](https://arxiv.org/pdf/2604.12650)**

> **作者:** Miao Liu; Fangda Wei; Jing Wang; Xinyuan Qian
>
> **备注:** Submitted to ACMMM 2026
>
> **摘要:** Existing deepfake detection research has primarily focused on scenarios where the manipulated subject is actively speaking, i.e., generating fabricated content by altering the speaker's appearance or voice. However, in realistic interaction settings, attackers often alternate between falsifying speaking and listening states to mislead their targets, thereby enhancing the realism and persuasiveness of the scenario. Although the detection of 'listening deepfakes' remains largely unexplored and is hindered by a scarcity of both datasets and methodologies, the relatively limited quality of synthesized listening reactions presents an excellent breakthrough opportunity for current deepfake detection efforts. In this paper, we present the task of Listening Deepfake Detection (LDD). We introduce ListenForge, the first dataset specifically designed for this task, constructed using five Listening Head Generation (LHG) methods. To address the distinctive characteristics of listening forgeries, we propose MANet, a Motion-aware and Audio-guided Network that captures subtle motion inconsistencies in listener videos while leveraging speaker's audio semantics to guide cross-modal fusion. Extensive experiments demonstrate that existing Speaking Deepfake Detection (SDD) models perform poorly in listening scenarios. In contrast, MANet achieves significantly superior performance on ListenForge. Our work highlights the necessity of rethinking deepfake detection beyond the traditional speaking-centric paradigm and opens new directions for multimodal forgery analysis in interactive communication settings. The dataset and code are available at this https URL.
>
---
#### [new 089] Don't Show Pixels, Show Cues: Unlocking Visual Tool Reasoning in Language Models via Perception Programs
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉语言推理任务，解决MLLMs无法有效利用视觉工具输出的问题。通过引入感知程序（P²），将工具输出转化为结构化语言摘要，提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2604.12896](https://arxiv.org/pdf/2604.12896)**

> **作者:** Muhammad Kamran Janjua; Hugo Silva; Di Niu; Bahador Rashidi
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Multimodal language models (MLLMs) are increasingly paired with vision tools (e.g., depth, flow, correspondence) to enhance visual reasoning. However, despite access to these tool-generated visual cues, MLLMs often fail to benefit from them. Existing approaches typically feed raw tool outputs into the model, but these dense, pixel-level representations are misaligned with the language-native reasoning strengths of LLMs, leading to weak perception and reliance on language priors. We argue that, in problems where vision tools can provide the necessary visual cues, the bottleneck is not more tool calls or larger MLLMs, it is how tool outputs are represented. We introduce Perception Programs (P$^2$), a training-free, model-agnostic method that rewrites tool outputs into compact, structured, language-native summaries that MLLMs can directly parse and reason over. Across six perception-centric tasks in BLINK, P$^2$ consistently yields large improvements over base models and raw tool-augmented baselines. With GPT-5 Mini as the base model, P$^2$ raises its accuracy from 41.35\% to 86.47\% on multi-view reasoning, from 52.42\% to 81.45\% on relative depth, and achieves a 22\% average gain across tasks, setting new state-of-the-art results. Even on smaller MLLMs, e.g., InternVL3.5-4B and Qwen3VL-4B, we observe 15-40\% absolute gains from P$^2$, surpassing prior agentic, supervised, and RL-based tool-use methods-without any training or model modifications.
>
---
#### [new 090] Towards Realistic and Consistent Orbital Video Generation via 3D Foundation Priors
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决单图生成连贯轨道视频的问题。通过引入3D基础模型的形状先验，提升生成视频的几何真实性和视角一致性。**

- **链接: [https://arxiv.org/pdf/2604.12309](https://arxiv.org/pdf/2604.12309)**

> **作者:** Rong Wang; Ruyi Zha; Ziang Cheng; Jiayu Yang; Pulak Purkait; Hongdong Li
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** We present a novel method for generating geometrically realistic and consistent orbital videos from a single image of an object. Existing video generation works mostly rely on pixel-wise attention to enforce view consistency across frames. However, such mechanism does not impose sufficient constraints for long-range extrapolation, e.g. rear-view synthesis, in which pixel correspondences to the input image are limited. Consequently, these works often fail to produce results with a plausible and coherent structure. To tackle this issue, we propose to leverage rich shape priors from a 3D foundational generative model as an auxiliary constraint, motivated by its capability of modeling realistic object shape distributions learned from large 3D asset corpora. Specifically, we prompt the video generation with two scales of latent features encoded by the 3D foundation model: (i) a denoised global latent vector as an overall structural guidance, and (ii) a set of latent images projected from volumetric features to provide view-dependent and fine-grained geometry details. In contrast to commonly used 2.5D representations such as depth or normal maps, these compact features can model complete object shapes, and help to improve inference efficiency by avoiding explicit mesh extraction. To achieve effective shape conditioning, we introduce a multi-scale 3D adapter to inject feature tokens to the base video model via cross-attention, which retains its capabilities from general video pretraining and enables a simple and model-agonistic fine-tuning process. Extensive experiments on multiple benchmarks show that our method achieves superior visual quality, shape realism and multi-view consistency compared to state-of-the-art methods, and robustly generalizes to complex camera trajectories and in-the-wild images.
>
---
#### [new 091] Brain-DiT: A Universal Multi-state fMRI Foundation Model with Metadata-Conditioned Pretraining
- **分类: cs.CV; q-bio.NC**

- **简介: 该论文提出Brain-DiT，解决fMRI模型泛化能力不足的问题，通过多状态预训练学习跨脑态的通用表示，提升下游任务性能。**

- **链接: [https://arxiv.org/pdf/2604.12683](https://arxiv.org/pdf/2604.12683)**

> **作者:** Junfeng Xia; Wenhao Ye; Xuanye Pan; Xinke Shen; Mo Wang; Quanying Liu
>
> **摘要:** Current fMRI foundation models primarily rely on a limited range of brain states and mismatched pretraining tasks, restricting their ability to learn generalized representations across diverse brain states. We present \textit{Brain-DiT}, a universal multi-state fMRI foundation model pretrained on 349,898 sessions from 24 datasets spanning resting, task, naturalistic, disease, and sleep states. Unlike prior fMRI foundation models that rely on masked reconstruction in the raw-signal space or a latent space, \textit{Brain-DiT} adopts metadata-conditioned diffusion pretraining with a Diffusion Transformer (DiT), enabling the model to learn multi-scale representations that capture both fine-grained functional structure and global semantics. Across extensive evaluations and ablations on 7 downstream tasks, we find consistent evidence that diffusion-based generative pretraining is a stronger proxy than reconstruction or alignment, with metadata-conditioned pretraining further improving downstream performance by disentangling intrinsic neural dynamics from population-level variability. We also observe that downstream tasks exhibit distinct preferences for representational scale: ADNI classification benefits more from global semantic representations, whereas age/sex prediction comparatively relies more on fine-grained local structure. Code and parameters of Brain-DiT are available at \href{this https URL}{Link}.
>
---
#### [new 092] Boosting Robust AIGI Detection with LoRA-based Pairwise Training
- **分类: cs.CV**

- **简介: 该论文属于AIGI检测任务，旨在解决模型在复杂畸变下检测性能下降的问题。通过LoRA微调和配对训练提升检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.12307](https://arxiv.org/pdf/2604.12307)**

> **作者:** Ruiyang Xia; Qi Zhang; Yaowen Xu; Zhaofan Zou; Hao Sun; Zhongjiang He; Xuelong Li
>
> **备注:** 3th place (3/514) technical report(CVPRW-26) at the NTIRE 2026: Robust AI-Generated Image Detection in the Wild Challenge
>
> **摘要:** The proliferation of highly realistic AI-Generated Image (AIGI) has necessitated the development of practical detection methods. While current AIGI detectors perform admirably on clean datasets, their detection performance frequently decreases when deployed "in the wild", where images are subjected to unpredictable, complex distortions. To resolve the critical vulnerability, we propose a novel LoRA-based Pairwise Training (LPT) strategy designed specifically to achieve robust detection for AIGI under severe distortions. The core of our strategy involves the targeted finetuning of a visual foundation model, the deliberate simulation of data distribution during the training phase, and a unique pairwise training process. Specifically, we introduce distortion and size simulations to better fit the distribution from the validation and test sets. Based on the strong visual representation capability of the visual foundation model, we finetune the model to achieve AIGI detection. The pairwise training is utilized to improve the detection via decoupling the generalization and robustness optimization. Experiments show that our approach secured the 3th placement in the NTIRE Robust AI-Generated Image Detection in the Wild challenge
>
---
#### [new 093] Self-Adversarial One Step Generation via Condition Shifting
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决单步采样中的保真度、速度与训练效率的平衡问题。提出APEX方法，通过条件转移提取对抗信号，提升生成质量与速度。**

- **链接: [https://arxiv.org/pdf/2604.12322](https://arxiv.org/pdf/2604.12322)**

> **作者:** Deyuan Liu; Peng Sun; Yansen Han; Zhenglin Cheng; Chuyan Chen; Tao Lin
>
> **摘要:** The push for efficient text to image synthesis has moved the field toward one step sampling, yet existing methods still face a three way tradeoff among fidelity, inference speed, and training efficiency. Approaches that rely on external discriminators can sharpen one step performance, but they often introduce training instability, high GPU memory overhead, and slow convergence, which complicates scaling and parameter efficient tuning. In contrast, regression based distillation and consistency objectives are easier to optimize, but they typically lose fine details when constrained to a single step. We present APEX, built on a key theoretical insight: adversarial correction signals can be extracted endogenously from a flow model through condition shifting. Using a transformation creates a shifted condition branch whose velocity field serves as an independent estimator of the model's current generation distribution, yielding a gradient that is provably GAN aligned, replacing the sample dependent discriminator terms that cause gradient vanishing. This discriminator free design is architecture preserving, making APEX a plug and play framework compatible with both full parameter and LoRA based tuning. Empirically, our 0.6B model surpasses FLUX-Schnell 12B (20$\times$ more parameters) in one step quality. With LoRA tuning on Qwen-Image 20B, APEX reaches a GenEval score of 0.89 at NFE=1 in 6 hours, surpassing the original 50-step teacher (0.87) and providing a 15.33$\times$ inference speedup. Code is available this https URL.
>
---
#### [new 094] Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CoM-PT方法，用于加速视觉基础模型的预训练，解决训练成本高的问题。通过模型链实现知识迁移，提升效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2604.12391](https://arxiv.org/pdf/2604.12391)**

> **作者:** Jiawei Fan; Shigeng Wang; Chao Li; Xiaolong Liu; Anbang Yao
>
> **备注:** This work is accepted to CVPR 2026. Code is available at this https URL
>
> **摘要:** In this paper, we present Chain-of-Models Pre-Training (CoM-PT), a novel performance-lossless training acceleration method for vision foundation models (VFMs). This approach fundamentally differs from existing acceleration methods in its core motivation: rather than optimizing each model individually, CoM-PT is designed to accelerate the training pipeline at the model family level, scaling efficiently as the model family expands. Specifically, CoM-PT establishes a pre-training sequence for the model family, arranged in ascending order of model size, called model chain. In this chain, only the smallest model undergoes standard individual pre-training, while the other models are efficiently trained through sequential inverse knowledge transfer from their smaller predecessors by jointly reusing the knowledge in the parameter space and the feature space. As a result, CoM-PT enables all models to achieve performance that is mostly superior to standard individual training while significantly reducing training cost, and this is extensively validated across 45 datasets spanning zero-shot and fine-tuning tasks. Notably, its efficient scaling property yields a remarkable phenomenon: training more models even results in higher efficiency. For instance, when pre-training on CC3M: i) given ViT-L as the largest model, progressively prepending smaller models to the model chain reduces computational complexity by up to 72%; ii) within a fixed model size range, as the VFM family scales across 3, 4, and 7 models, the acceleration ratio of CoM-PT exhibits a striking leap: from 4.13X to 5.68X and 7.09X. Since CoM-PT is naturally agnostic to specific pre-training paradigms, we open-source the code to spur further extensions in more computationally intensive scenarios, such as large language model pre-training.
>
---
#### [new 095] PromptEcho: Annotation-Free Reward from Vision-Language Models for Text-to-Image Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成的强化学习任务，旨在解决奖励信号获取困难的问题。提出PromptEcho方法，无需标注和训练，利用预训练视觉语言模型直接计算奖励。**

- **链接: [https://arxiv.org/pdf/2604.12652](https://arxiv.org/pdf/2604.12652)**

> **作者:** Jinlong Liu; Wanggui He; Peng Zhang; Mushui Liu; Hao Jiang; Pipei Huang
>
> **摘要:** Reinforcement learning (RL) can improve the prompt following capability of text-to-image (T2I) models, yet obtaining high-quality reward signals remains challenging: CLIP Score is too coarse-grained, while VLM-based reward models (e.g., RewardDance) require costly human-annotated preference data and additional fine-tuning. We propose PromptEcho, a reward construction method that requires \emph{no} annotation and \emph{no} reward model training. Given a generated image and a guiding query, PromptEcho computes the token-level cross-entropy loss of a frozen VLM with the original prompt as the label, directly extracting the image-text alignment knowledge encoded during VLM pretraining. The reward is deterministic, computationally efficient, and improves automatically as stronger open-source VLMs become available. For evaluation, we develop DenseAlignBench, a benchmark of concept-rich dense captions for rigorously testing prompt following capability. Experimental results on two state-of-the-art T2I models (Z-Image and QwenImage-2512) demonstrate that PromptEcho achieves substantial improvements on DenseAlignBench (+26.8pp / +16.2pp net win rate), along with consistent gains on GenEval, DPG-Bench, and TIIFBench without any task-specific training. Ablation studies confirm that PromptEcho comprehensively outperforms inference-based scoring with the same VLM, and that reward quality scales with VLM size. We will open-source the trained models and the DenseAlignBench.
>
---
#### [new 096] Distorted or Fabricated? A Survey on Hallucination in Video LLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频语言模型任务，旨在解决Vid-LLMs中的幻觉问题。通过分类与分析，提出评估和缓解方法，提升系统可靠性。**

- **链接: [https://arxiv.org/pdf/2604.12944](https://arxiv.org/pdf/2604.12944)**

> **作者:** Yiyang Huang; Yitian Zhang; Yizhou Wang; Mingyuan Zhang; Liang Shi; Huimin Zeng; Yun Fu
>
> **备注:** ACL 2026 findings
>
> **摘要:** Despite significant progress in video-language modeling, hallucinations remain a persistent challenge in Video Large Language Models (Vid-LLMs), referring to outputs that appear plausible yet contradict the content of the input video. This survey presents a comprehensive analysis of hallucinations in Vid-LLMs and introduces a systematic taxonomy that categorizes them into two core types: dynamic distortion and content fabrication, each comprising two subtypes with representative cases. Building on this taxonomy, we review recent advances in the evaluation and mitigation of hallucinations, covering key benchmarks, metrics, and intervention strategies. We further analyze the root causes of dynamic distortion and content fabrication, which often result from limited capacity for temporal representation and insufficient visual grounding. These insights inform several promising directions for future work, including the development of motion-aware visual encoders and the integration of counterfactual learning techniques. This survey consolidates scattered progress to foster a systematic understanding of hallucinations in Vid-LLMs, laying the groundwork for building robust and reliable video-language systems. An up-to-date curated list of related works is maintained at this https URL .
>
---
#### [new 097] DreamStereo: Towards Real-Time Stereo Inpainting for HD Videos
- **分类: cs.CV**

- **简介: 该论文属于立体视频修复任务，旨在解决高精度与实时性问题。通过提出GAPW、PBDP和SASI组件，提升修复效果并加速处理，实现实时HD视频修复。**

- **链接: [https://arxiv.org/pdf/2604.12270](https://arxiv.org/pdf/2604.12270)**

> **作者:** Yuan Huang; Sijie Zhao; Jing Cheng; Hao Xu; Shaohui Jiao
>
> **摘要:** Stereo video inpainting, which aims to fill the occluded regions of warped videos with visually coherent content while maintaining temporal consistency, remains a challenging open problem. The regions to be filled are scattered along object boundaries and occupy only a small fraction of each frame, leading to two key challenges. First, existing approaches perform poorly on such tasks due to the scarcity of high-quality stereo inpainting datasets, which limits their ability to learn effective inpainting priors. Second, these methods apply equal processing to all regions of the frame, even though most pixels require no modification, resulting in substantial redundant computation. To address these issues, we introduce three interconnected components. We first propose Gradient-Aware Parallax Warping (GAPW), which leverages backward warping and the gradient of the coordinate mapping function to obtain continuous edges and smooth occlusion regions. Then, a Parallax-Based Dual Projection (PBDP) strategy is introduced, which incorporates GAPW to produce geometrically consistent stereo inpainting pairs and accurate occlusion masks without requiring stereo videos. Finally, we present Sparsity-Aware Stereo Inpainting (SASI), which reduces over 70% of redundant tokens, achieving a 10.7x speedup during diffusion inference and delivering results comparable to its full-computation counterpart, enabling real-time processing of HD (768 x 1280) videos at 25 FPS on a single A100 GPU.
>
---
#### [new 098] Radar-Camera BEV Multi-Task Learning with Cross-Task Attention Bridge for Joint 3D Detection and Segmentation
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的3D目标检测与分割任务，旨在解决雷达-相机融合中任务间信息孤立的问题。通过引入CTAB模块实现检测与分割特征的双向交互，提升整体性能。**

- **链接: [https://arxiv.org/pdf/2604.12918](https://arxiv.org/pdf/2604.12918)**

> **作者:** Ahmet İnanç; Özgür Erkent
>
> **备注:** 8 pages, 5 figures, 3 Tables, submitted to a venue for consideration
>
> **摘要:** Bird's-eye-view (BEV) representations are the dominant paradigm for 3D perception in autonomous driving, providing a unified spatial canvas where detection and segmentation features are geometrically registered to the same physical coordinate system. However, existing radar-camera fusion methods treat these tasks in isolation, missing the opportunity to share complementary information between them: detection features encode object-level geometry that can sharpen segmentation boundaries, while segmentation features provide dense semantic context that can anchor detection. We propose \textbf{CTAB} (Cross-Task Attention Bridge), a bidirectional module that exchanges features between detection and segmentation branches via multi-scale deformable attention in shared BEV space. CTAB is integrated into a multi-task framework with an Instance Normalization-based segmentation decoder and learnable BEV upsampling to provide a more detailed BEV representation. On nuScenes, CTAB improves segmentation on 7 classes over the joint multi-task baseline at essentially neutral detection. On a 4-class subset (drivable area, pedestrian crossing, walkway, vehicle), our joint multi-task model reaches comparable mIoU on 4 classes while simultaneously providing 3D detection.
>
---
#### [new 099] Domain-Specific Latent Representations Improve the Fidelity of Diffusion-Based Medical Image Super-Resolution
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像超分辨率任务，旨在提升重建质量。通过引入领域特定的自编码器MedVAE，显著提高了PSNR指标，验证了自编码器对后续任务的重要性。**

- **链接: [https://arxiv.org/pdf/2604.12152](https://arxiv.org/pdf/2604.12152)**

> **作者:** Sebastian Cajas; Ashaba Judith; Rahul Gorijavolu; Sahil Kapadia; Hillary Clinton Kasimbazi; Leo Kinyera; Emmanuel Paul Kwesiga; Sri Sri Jaithra Varma Manthena; Luis Filipe Nakayama; Ninsiima Doreen; Leo Anthony Celi
>
> **摘要:** Latent diffusion models for medical image super-resolution universally inherit variational autoencoders designed for natural photographs. We show that this default choice, not the diffusion architecture, is the dominant constraint on reconstruction quality. In a controlled experiment holding all other pipeline components fixed, replacing the generic Stable Diffusion VAE with MedVAE, a domain-specific autoencoder pretrained on more than 1.6 million medical images, yields +2.91 to +3.29 dB PSNR improvement across knee MRI, brain MRI, and chest X-ray (n = 1,820; Cohen's d = 1.37 to 1.86, all p < 10^{-20}, Wilcoxon signed-rank). Wavelet decomposition localises the advantage to the finest spatial frequency bands encoding anatomically relevant fine structure. Ablations across inference schedules, prediction targets, and generative architectures confirm the gap is stable within plus or minus 0.15 dB, while hallucination rates remain comparable between methods (Cohen's h < 0.02 across all datasets), establishing that reconstruction fidelity and generative hallucination are governed by independent pipeline components. These results provide a practical screening criterion: autoencoder reconstruction quality, measurable without diffusion training, predicts downstream SR performance (R^2 = 0.67), suggesting that domain-specific VAE selection should precede diffusion architecture search. Code and trained model weights are publicly available at this https URL.
>
---
#### [new 100] Efficient Adversarial Training via Criticality-Aware Fine-Tuning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉模型鲁棒性增强任务，旨在解决大规模ViT模型在对抗训练中计算成本高的问题。通过提出CAAT方法，仅微调关键参数，提升效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.12780](https://arxiv.org/pdf/2604.12780)**

> **作者:** Wenyun Li; Zheng Zhang; Dongmei Jiang; Yaowei Wang; Xiangyuan Lan
>
> **摘要:** Vision Transformer (ViT) models have achieved remarkable performance across various vision tasks, with scalability being a key advantage when applied to large datasets. This scalability enables ViT models to exhibit strong generalization capabilities. However, as the number of parameters increases, the robustness of ViT models to adversarial examples does not scale proportionally. Adversarial training (AT), one of the most effective methods for enhancing robustness, typically requires fine-tuning the entire model, leading to prohibitively high computational costs, especially for large ViT architectures. In this paper, we aim to robustly fine-tune only a small subset of parameters to achieve robustness comparable to standard AT. To accomplish this, we introduce Criticality-Aware Adversarial Training (CAAT), a novel method that adaptively allocates resources to the most robustness-critical parameters, fine-tuning only selected modules. Specifically, CAAT efficiently identifies parameters that contribute most to adversarial robustness. It then leverages parameter-efficient fine-tuning (PEFT) to robustly adjust weight matrices where the number of critical parameters exceeds a predefined threshold. CAAT exhibits favorable generalization when scaled to larger vision transformer architectures, potentially paving the way for adversarial training at scale, e.g, compared with plain adversarial training, CAAT incurs only a 4.3% decrease in adversarial robustness while tuning approximately 6% of its parameters. Extensive experiments on three widely used adversarial learning datasets demonstrate that CAAT outperforms state-of-the-art lightweight AT methods with fewer trainable parameters.
>
---
#### [new 101] DPC-VQA: Decoupling Quality Perception and Residual Calibration for Video Quality Assessment
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于视频质量评估任务，解决MLLM在新场景下适应成本高的问题。通过解耦感知与校准，提出DPC-VQA框架，降低训练成本和数据需求。**

- **链接: [https://arxiv.org/pdf/2604.12813](https://arxiv.org/pdf/2604.12813)**

> **作者:** Xinyue Li; Shubo Xu; Zhichao Zhang; Zhaolin Cai; Yitong Chen; Guangtao Zhai
>
> **摘要:** Recent multimodal large language models (MLLMs) have shown promising performance on video quality assessment (VQA) tasks. However, adapting them to new scenarios remains expensive due to large-scale retraining and costly mean opinion score (MOS) annotations. In this paper, we argue that a pretrained MLLM already provides a useful perceptual prior for VQA, and that the main challenge is to efficiently calibrate this prior to the target MOS space. Based on this insight, we propose DPC-VQA, a decoupling perception and calibration framework for video quality assessment. Specifically, DPC-VQA uses a frozen MLLM to provide a base quality estimate and perceptual prior, and employs a lightweight calibration branch to predict a residual correction for target-scenario adaptation. This design avoids costly end-to-end retraining while maintaining reliable performance with lower training and data costs. Extensive experiments on both user-generated content (UGC) and AI-generated content (AIGC) benchmarks show that DPC-VQA achieves competitive performance against representative baselines, while using less than 2% of the trainable parameters of conventional MLLM-based VQA methods and remaining effective with only 20\% of MOS labels. The code will be released upon publication.
>
---
#### [new 102] PC-MIL: Decoupling Feature Resolution from Supervision Scale in Whole-Slide Learning
- **分类: cs.CV**

- **简介: 该论文属于病理图像分类任务，解决传统MIL方法因监督尺度与临床推理不匹配导致的模型泛化问题。通过解耦特征分辨率与监督尺度，提出PC-MIL框架，提升模型在不同尺度下的性能。**

- **链接: [https://arxiv.org/pdf/2604.12100](https://arxiv.org/pdf/2604.12100)**

> **作者:** Syed Fahim Ahmed; Gnanesh Rasineni; Florian Koehler; Abu Zahid Bin Aziz; Mei Wang; Attila Gyulassy; Brian Summa; J. Quincy Brown; Valerio Pascucci; Shireen Y. Elhabian
>
> **备注:** 11 pages, 2 figures, 2 tables. Under review at MICCAI 2026
>
> **摘要:** Whole-slide image (WSI) classification in computational pathology is commonly formulated as slide-level Multiple Instance Learning (MIL) with a single global bag representation. However, slide-level MIL is fundamentally underconstrained: optimizing only global labels encourages models to aggregate features without learning anatomically meaningful localization. This creates a mismatch between the scale of supervision and the scale of clinical reasoning. Clinicians assess tumor burden, focal lesions, and architectural patterns within millimeter-scale regions, whereas standard MIL is trained only to predict whether "somewhere in the slide there is cancer." As a result, the model's inductive bias effectively erases anatomical structure. We propose Progressive-Context MIL (PC-MIL), a framework that treats the spatial extent of supervision as a first-class design dimension. Rather than altering magnification, patch size, or introducing pixel-level segmentation, we decouple feature resolution from supervision scale. Using fixed 20x features, we vary MIL bag extent in millimeter units and anchor supervision at a clinically motivated 2mm scale to preserve comparable tumor burden and avoid confounding scale with lesion density. PC-MIL progressively mixes slide- and region-level supervision in controlled proportions, enabling explicit train-context x test-context analysis. On 1,476 prostate WSIs from five public datasets for binary cancer detection, we show that anatomical context is an independent axis of generalization in MIL, orthogonal to feature resolution: modest regional supervision improves cross-context performance, and balanced multi-context training stabilizes accuracy across slide and regional evaluation without sacrificing global performance. These results demonstrate that supervision extent shapes MIL inductive bias and support anatomically grounded WSI generalization.
>
---
#### [new 103] Style-Decoupled Adaptive Routing Network for Underwater Image Enhancement
- **分类: cs.CV**

- **简介: 该论文属于水下图像增强任务，旨在解决现有方法对不同退化程度图像处理不均衡的问题。提出SDAR-Net，通过解耦退化风格并自适应调节增强过程，提升图像质量。**

- **链接: [https://arxiv.org/pdf/2604.12257](https://arxiv.org/pdf/2604.12257)**

> **作者:** Hang Xu; Chen Long; Bing Wang; Hao Chen; Zhen Dong
>
> **摘要:** Underwater Image Enhancement (UIE) is essential for robust visual perception in marine applications. However, existing methods predominantly rely on uniform mapping tailored to average dataset distributions, leading to over-processing mildly degraded images or insufficient recovery for severe ones. To address this challenge, we propose a novel adaptive enhancement framework, SDAR-Net. Unlike existing uniform paradigms, it first decouples specific degradation styles from the input and subsequently modulates the enhancement process adaptively. Specifically, since underwater degradation primarily shifts the appearance while keeping the scene structure, SDAR-Net formulates image features into dynamic degradation style embeddings and static scene structural representations through a carefully designed training framework. Subsequently, we introduce an adaptive routing mechanism. By evaluating style features and adaptively predicting soft weights at different enhancement states, it guides the weighted fusion of the corresponding image representations, accurately satisfying the adaptive restoration demands of each image. Extensive experiments show that SDAR-Net achieves a new state-of-the-art (SOTA) performance with a PSNR of 25.72 dB on real-world benchmark, and demonstrates its utility in downstream vision tasks. Our code is available at this https URL.
>
---
#### [new 104] Conflated Inverse Modeling to Generate Diverse and Temperature-Change Inducing Urban Vegetation Patterns
- **分类: cs.CV**

- **简介: 该论文属于城市气候适应任务，解决如何通过植被配置实现特定温度变化的问题。工作是提出一种融合正向与生成逆向模型的框架，生成多样且物理合理的植被模式。**

- **链接: [https://arxiv.org/pdf/2604.13028](https://arxiv.org/pdf/2604.13028)**

> **作者:** Baris Sarper Tezcan; Hrishikesh Viswanath; Rubab Saher; Daniel Aliaga
>
> **备注:** Accepted to the CVPR 2026 EarthVision Workshop
>
> **摘要:** Urban areas are increasingly vulnerable to thermal extremes driven by rapid urbanization and climate change. Traditionally, thermal extremes have been monitored using Earth-observing satellites and numerical modeling frameworks. For example, land surface temperature derived from Landsat or Sentinel imagery is commonly used to characterize surface heating patterns. These approaches operate as forward models, translating radiative observations or modeled boundary conditions into estimates of surface thermal states. While forward models can predict land surface temperature from vegetation and urban form, the inverse problem of determining spatial vegetation configurations that achieve a desired regional temperature shift remains largely unexplored. This task is inherently underdetermined, as multiple spatial vegetation patterns can yield similar aggregated temperature responses. Conventional regression and deterministic neural networks fail to capture this ambiguity and often produce averaged solutions, particularly under data-scarce conditions. We propose a conflated inverse modeling framework that combines a predictive forward model with a diffusion-based generative inverse model to produce diverse, physically plausible image-based vegetation patterns conditioned on specific temperature goals. Our framework maintains control over thermal outcomes while enabling diverse spatial vegetation configurations, even when such combinations are absent from training data. Altogether, this work introduces a controllable inverse modeling approach for urban climate adaptation that accounts for the inherent diversity of the problem. Code is available at the GitHub repository.
>
---
#### [new 105] A Dataset and Evaluation for Complex 4D Markerless Human Motion Capture
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于4D人体运动捕捉任务，旨在解决真实场景下无标记运动捕捉的难题。提出一个包含复杂多人交互的高质量数据集，并验证现有模型的不足。**

- **链接: [https://arxiv.org/pdf/2604.12765](https://arxiv.org/pdf/2604.12765)**

> **作者:** Yeeun Park; Miqdad Naduthodi; Suryansh Kumar
>
> **备注:** 14 pages, 11 figures, 4 tables. Accepted for publication at CVPR 2026 4D World Models Workshop
>
> **摘要:** Marker-based motion capture (MoCap) systems have long been the gold standard for accurate 4D human modeling, yet their reliance on specialized hardware and markers limits scalability and real-world deployment. Advancing reliable markerless 4D human motion capture requires datasets that reflect the complexity of real-world human interactions. Yet, existing benchmarks often lack realistic multi-person dynamics, severe occlusions, and challenging interaction patterns, leading to a persistent domain gap. In this work, we present a new dataset and evaluation for complex 4D markerless human motion capture. Our proposed MoCap dataset captures both single and multi-person scenarios with intricate motions, frequent inter-person occlusions, rapid position exchanges between similarly dressed subjects, and varying subject distances. It includes synchronized multi-view RGB and depth sequences, accurate camera calibration, ground-truth 3D motion capture from a Vicon system, and corresponding SMPL/SMPL-X parameters. This setup ensures precise alignment between visual observations and motion ground truth. Benchmarking state-of-the-art markerless MoCap models reveals substantial performance degradation under these realistic conditions, highlighting limitations of current approaches. We further demonstrate that targeted fine-tuning improves generalization, validating the dataset's realism and value for model development. Our evaluation exposes critical gaps in existing models and provides a rigorous foundation for advancing robust markerless 4D human motion capture.
>
---
#### [new 106] DiffusionPrint: Learning Generative Fingerprints for Diffusion-Based Inpainting Localization
- **分类: cs.CV**

- **简介: 该论文属于图像伪造定位任务，解决扩散模型 inpainting 带来的定位难题。提出 DiffusionPrint，通过对比学习提取鲁棒的指纹特征，提升定位效果。**

- **链接: [https://arxiv.org/pdf/2604.12443](https://arxiv.org/pdf/2604.12443)**

> **作者:** Paschalis Giakoumoglou; Symeon Papadopoulos
>
> **备注:** CVPRW2026
>
> **摘要:** Modern diffusion-based inpainting models pose significant challenges for image forgery localization (IFL), as their full regeneration pipelines reconstruct the entire image via a latent decoder, disrupting the camera-level noise patterns that existing forensic methods rely on. We propose DiffusionPrint, a patch-level contrastive learning framework that learns a forensic signal robust to the spectral distortions introduced by latent decoding. It exploits the fact that inpainted regions generated by the same model share a consistent generative fingerprint, using this as a self-supervisory signal. DiffusionPrint trains a convolutional backbone via a MoCo-style objective with cross-category hard negative mining and a generator-aware classification head, producing a forensic feature map that serves as a highly discriminative secondary modality in fusion-based IFL frameworks. Integrated into TruFor, MMFusion, and a lightweight fusion baseline, DiffusionPrint consistently improves localization across multiple generative models, with gains of up to +28% on mask types unseen during fine-tuning and confirmed generalization to unseen generative architectures. Code is available at this https URL
>
---
#### [new 107] Cross-Attentive Multiview Fusion of Vision-Language Embeddings
- **分类: cs.CV**

- **简介: 该论文聚焦于3D场景的语义分割任务，解决多视角视觉-语言嵌入融合问题。提出CAMFusion方法，通过交叉注意力机制融合多视角特征，提升3D实例表示效果。**

- **链接: [https://arxiv.org/pdf/2604.12551](https://arxiv.org/pdf/2604.12551)**

> **作者:** Tomas Berriel Martins; Martin R. Oswald; Javier Civera
>
> **摘要:** Vision-language models have been key to the development of open-vocabulary 2D semantic segmentation. Lifting these models from 2D images to 3D scenes, however, remains a challenging problem. Existing approaches typically back-project and average 2D descriptors across views, or heuristically select a single representative one, often resulting in suboptimal 3D representations. In this work, we introduce a novel multiview transformer architecture that cross-attends across vision-language descriptors from multiple viewpoints and fuses them into a unified per-3D-instance embedding. As a second contribution, we leverage multiview consistency as a self-supervision signal for this fusion, which significantly improves performance when added to a standard supervised target-class loss. Our Cross-Attentive Multiview Fusion, which we denote with its acronym CAMFusion, not only consistently outperforms naive averaging or single-view descriptor selection, but also achieves state-of-the-art results on 3D semantic and instance classification benchmarks, including zero-shot evaluations on out-of-domain datasets.
>
---
#### [new 108] The Second Challenge on Cross-Domain Few-Shot Object Detection at NTIRE 2026: Methods and Results
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于跨域小样本目标检测任务，旨在解决在有限标注下从源域到目标域的物体检测问题。文中介绍了NTIRE 2026挑战赛的方法与结果。**

- **链接: [https://arxiv.org/pdf/2604.11998](https://arxiv.org/pdf/2604.11998)**

> **作者:** Xingyu Qiu; Yuqian Fu; Jiawei Geng; Bin Ren; Jiancheng Pan; Zongwei Wu; Hao Tang; Yanwei Fu; Radu Timofte; Nicu Sebe; Mohamed Elhoseiny; Lingyi Hong; Mingxi Cheng; Xingqi He; Runze Li; Xingdong Sheng; Wenqiang Zhang; Jiacong Liu; Shu Luo; Yikai Qin; Yaze Zhao; Yongwei Jiang; Yixiong Zou; Zhe Zhang; Yang Yang; Kaiyu Li; Bowen Fu; Zixuan Jiang; Ke Li; Hui Qiao; Xiangyong Cao; Xuanlong Yu; Youyang Sha; Longfei Liu; Di Yang; Xi Shen; Kyeongryeol Go; Taewoong Jang; Saiprasad Meesiyawar; Ravi Kirasur; Rakshita Kulkarni; Bhoomi Deshpande; Harsh Patil; Uma Mudenagudi; Shuming Hu; Chao Chen; Tao Wang; Wei Zhou; Qi Xu; Zhenzhao Xing; Dandan Zhao; Hanzhe Xia; Dongdong Lu; Zhe Zhang; Jingru Wang; Guangwei Huang; Jiachen Tu; Yaokun Shi; Guoyi Xu; Yaoxin Jiang; Jiajia Liu; Liwei Zhou; Bei Dou; Tao Wu; Zekang Fan; Junjie Liu; Adhémar de Senneville; Flavien Armangeon; Mengbers; Yazhe Lyu; Zhimeng Xin; Zijian Zhuang; Hongchun Zhu; Li Wang
>
> **备注:** accepted by CVPRW 26 @ NTIRE
>
> **摘要:** Cross-domain few-shot object detection (CD-FSOD) remains a challenging problem for existing object detectors and few-shot learning approaches, particularly when generalizing across distinct domains. As part of NTIRE 2026, we hosted the second CD-FSOD Challenge to systematically evaluate and promote progress in detecting objects in unseen target domains under limited annotation conditions. The challenge received strong community interest, with 128 registered participants and a total of 696 submissions. Among them, 31 teams actively participated, and 19 teams submitted valid final results. Participants explored a wide range of strategies, introducing innovative methods that push the performance frontier under both open-source and closed-source tracks. This report presents a detailed overview of the NTIRE 2026 CD-FSOD Challenge, including a summary of the submitted approaches and an analysis of the final results across all participating teams. Challenge Codes: this https URL.
>
---
#### [new 109] Redefining Quality Criteria and Distance-Aware Score Modeling for Image Editing Assessment
- **分类: cs.CV**

- **简介: 该论文属于图像编辑质量评估任务，解决传统方法依赖人工提示和评分连续性不足的问题。提出DS-IEQA框架，结合自动优化指标和距离回归损失，提升评估效果。**

- **链接: [https://arxiv.org/pdf/2604.12175](https://arxiv.org/pdf/2604.12175)**

> **作者:** Xinjie Zhang; Qiang Li; Xiaowen Ma; Axi Niu; Li Yan; Qingsen Yan
>
> **摘要:** Recent advances in image editing have heightened the need for reliable Image Editing Quality Assessment (IEQA). Unlike traditional methods, IEQA requires complex reasoning over multimodal inputs and multi-dimensional assessments. Existing MLLM-based approaches often rely on human heuristic prompting, leading to two key limitations: rigid metric prompting and distance-agnostic score modeling. These issues hinder alignment with implicit human criteria and fail to capture the continuous structure of score spaces. To address this, we propose Define-and-Score Image Editing Quality Assessment (DS-IEQA), a unified framework that jointly learns evaluation criteria and score representations. Specifically, we introduce Feedback-Driven Metric Prompt Optimization (FDMPO) to automatically refine metric definitions via probabilistic feedback. Furthermore, we propose Token-Decoupled Distance Regression Loss (TDRL), which decouples numerical tokens from language modeling to explicitly model score continuity through expected distance minimization. Extensive experiments show our method's superior performance; it ranks 4th in the 2026 NTIRE X-AIGC Quality Assessment Track 2 without any additional training data.
>
---
#### [new 110] Agentic Discovery with Active Hypothesis Exploration for Visual Recognition
- **分类: cs.CV**

- **简介: 该论文提出HypoExplore框架，用于视觉识别的神经网络架构发现。解决如何高效探索设计空间的问题，通过假设驱动的方法进行进化分支，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.12999](https://arxiv.org/pdf/2604.12999)**

> **作者:** Jaywon Koo; Jefferson Hernandez; Ruozhen He; Hanjie Chen; Chen Wei; Vicente Ordonez
>
> **摘要:** We introduce HypoExplore, an agentic framework that formulates neural architecture discovery for visual recognition as a hypothesis-driven scientific inquiry. Given a human-specified high-level research direction, HypoExplore ideates, implements, evaluates, and improves neural architectures through evolutionary branching. New hypotheses are created using a large language model by selecting a parent hypothesis to build upon, guided by a dual strategy that balances exploiting validated principles with resolving uncertain ones. Our proposed framework maintains a Trajectory Tree that records the lineage of all proposed architectures, and a Hypothesis Memory Bank that actively tracks confidence scores acquired through experimental evidence. After each experiment, multiple feedback agents analyze the results from different perspectives and consolidate their findings into hypothesis confidence updates. Our framework is tested on discovering lightweight vision architectures on CIFAR-10, with the best achieving 94.11% accuracy evolved from a root node baseline that starts at 18.91%, and generalizes to CIFAR-100 and Tiny-ImageNet. We further demonstrate applicability to a specialized domain by conducting independent architecture discovery runs on MedMNIST, which yield a state-of-the-art performance. We show that hypothesis confidence scores grow increasingly predictive as evidence accumulates, and that the learned principles transfer across independent evolutionary lineages, suggesting that HypoExplore not only discovers stronger architectures, but can help build a genuine understanding of the design space.
>
---
#### [new 111] OFA-Diffusion Compression: Compressing Diffusion Model in One-Shot Manner
- **分类: cs.CV**

- **简介: 该论文属于模型压缩任务，旨在解决扩散模型在不同设备上部署时重复训练的高开销问题。通过一次训练生成多种规模的子网络，实现高效压缩。**

- **链接: [https://arxiv.org/pdf/2604.12668](https://arxiv.org/pdf/2604.12668)**

> **作者:** Haoyang Jiang; Zekun Wang; Mingyang Yi; Xiuyu Li; Lanqing Hu; Junxian Cai; Qingbin Liu; Xi Chen; Ju Fan
>
> **摘要:** The Diffusion Probabilistic Model (DPM) achieves remarkable performance in image generation, while its increasing parameter size and computational overhead hinder its deployment in practical applications. To improve this, the existing literature focuses on obtaining a smaller model with a fixed architecture through model compression. However, in practice, DPMs usually need to be deployed on various devices with different resource constraints, which leads to multiple compression processes, incurring significant overhead for repeated training. To obviate this, we propose a once-for-all (OFA) compression framework for DPMs that yields different subnetworks with various computations in a one-shot training manner. The existing OFA framework typically involves massive subnetworks with different parameter sizes, while such a huge candidate space slows the optimization. Thus, we propose to restrict the candidate subnetworks with a certain set of parameter sizes, where each size corresponds to a specific subnetwork. Specifically, to construct each subnetwork with a given size, we gradually allocate the maintained channels by their importance. Furthermore, we propose a reweighting strategy to balance the optimization process of different subnetworks. Experimental results show that our approach can produce compressed DPMs for various sizes with significantly lower training overhead while achieving satisfactory performance.
>
---
#### [new 112] MAST: Mask-Guided Attention Mass Allocation for Training-Free Multi-Style Transfer
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像风格迁移任务，解决多风格迁移中的边界伪影和结构不一致问题。提出MAST框架，通过注意力机制实现无训练的多风格融合。**

- **链接: [https://arxiv.org/pdf/2604.12281](https://arxiv.org/pdf/2604.12281)**

> **作者:** Dongkyung Kang; Jaeyeon Hwang; Junseo Park; Minji Kang; Yeryeong Lee; Beomseok Ko; Hanyoung Roh; Jeongmin Shin; Hyeryung Jang
>
> **备注:** 16 pages, 16 figures, 6 tables
>
> **摘要:** Style transfer aims to render a content image with the visual characteristics of a reference style while preserving its underlying semantic layout and structural geometry. While recent diffusion-based models demonstrate strong stylization capabilities by leveraging powerful generative priors and controllable internal representations, they typically assume a single global style. Extending them to multi-style scenarios often leads to boundary artifacts, unstable stylization, and structural inconsistency due to interference between multiple style representations. To overcome these limitations, we propose MAST (Mask-Guided Attention Mass Allocation for Training-Free Multi-Style Transfer), a novel training-free framework that explicitly controls content-style interactions within the diffusion attention mechanism. To achieve artifact-free and structure-preserving stylization, MAST integrates four connected modules. First, Layout-preserving Query Anchoring prevents global layout collapse by firmly anchoring the semantic structure using content queries. Second, Logit-level Attention Mass Allocation deterministically distributes attention probability mass across spatial regions, seamlessly fusing multiple styles without boundary artifacts. Third, Sharpness-aware Temperature Scaling restores the attention sharpness degraded by multi-style expansion. Finally, Discrepancy-aware Detail Injection adaptively compensates for localized high-frequency detail losses by measuring structural discrepancies. Extensive experiments demonstrate that MAST effectively mitigates boundary artifacts and maintains structural consistency, preserving texture fidelity and spatial coherence even as the number of applied styles increases.
>
---
#### [new 113] UniMark: Unified Adaptive Multi-bit Watermarking for Autoregressive Image Generators
- **分类: cs.CV**

- **简介: 该论文提出一种统一的多比特水印框架，解决AR图像生成器的版权保护问题，通过自适应分组、块编码和统一接口实现高效安全的水印嵌入与提取。**

- **链接: [https://arxiv.org/pdf/2604.11843](https://arxiv.org/pdf/2604.11843)**

> **作者:** Yigit Yilmaz; Elena Petrova; Mehmet Kaya; Lucia Rossi; Amir Rahman
>
> **备注:** work in progress
>
> **摘要:** Invisible watermarking for autoregressive (AR) image generation has recently gained attention as a means of protecting image ownership and tracing AI-generated content. However, existing approaches suffer from three key limitations: (1) they embed only zero-bit watermarks for binary verification, lacking the ability to convey multi-bit messages; (2) they rely on static codebook partitioning strategies that are vulnerable to security attacks once the partition is exposed; and (3) they are designed for specific AR architectures, failing to generalize across diverse AR paradigms. We propose \method{}, a training-free, unified watermarking framework for autoregressive image generators that addresses all three limitations. \method{} introduces three core components: \textbf{Adaptive Semantic Grouping (ASG)}, which dynamically partitions codebook entries based on semantic similarity and a secret key, ensuring both image quality preservation and security; \textbf{Block-wise Multi-bit Encoding (BME)}, which divides the token sequence into blocks and encodes different bits across blocks with error-correcting codes for reliable message transmission; and \textbf{a Unified Token-Replacement Interface (UTRI)} that abstracts the watermark embedding process to support both next-token prediction (e.g., LlamaGen) and next-scale prediction (e.g., VAR) paradigms. We provide theoretical analysis on detection error rates and embedding capacity. Extensive experiments on three AR models demonstrate that \method{} achieves state-of-the-art performance in image quality (FID), watermark detection accuracy, and multi-bit message extraction, while maintaining robustness against cropping, JPEG compression, Gaussian noise, blur, color jitter, and random erasing attacks.
>
---
#### [new 114] MedConcept: Unsupervised Concept Discovery for Interpretability in Medical VLMs
- **分类: cs.CV**

- **简介: 该论文属于医学视觉语言模型的可解释性任务，旨在解决模型内部表示不透明的问题。通过无监督方法发现医学概念，并提供可验证的语义解释。**

- **链接: [https://arxiv.org/pdf/2604.11868](https://arxiv.org/pdf/2604.11868)**

> **作者:** Md Rakibul Haque; KM Arefeen Sultan; Tushar Kataria; Shireen Elhabian
>
> **摘要:** While medical Vision-Language models (VLMs) achieve strong performance on tasks such as tumor or organ segmentation and diagnosis prediction, their opaque latent representations limit clinical trust and the ability to explain predictions. Interpretability of these multimodal representations are therefore essential for the trustworthy clinical deployment of pretrained medical VLMs. However, current interpretability methods, such as gradient- or attention-based visualizations, are often limited to specific tasks such as classification. Moreover, they do not provide concept-level explanations derived from shared pretrained representations that can be reused across downstream tasks. We introduce MedConcept, a framework that uncovers latent medical concepts in a fully unsupervised manner and grounds them in clinically verifiable textual semantics. MedConcept identifies sparse neuron-level concept activations from pretrained VLM representations and translates them into pseudo-report-style summaries, enabling physician-level inspection of internal model reasoning. To address the lack of quantitative evaluation in concept-based interpretability, we introduce a quantitative semantic verification protocol that leverages an independent pretrained medical LLM as a frozen external evaluator to assess concept alignment with radiology reports. We define three concept scores, Aligned, Unaligned, and Uncertain, to quantify semantic support, contradiction, or ambiguity relative to radiology reports and use them exclusively for post hoc evaluation. These scores provide a quantitative baseline for assessing interpretability in medical VLMs. All codes, prompt and data to be released on acceptance. Ke
>
---
#### [new 115] PianoFlow: Music-Aware Streaming Piano Motion Generation with Bimanual Coordination
- **分类: cs.CV**

- **简介: 该论文属于音乐生成任务，解决钢琴双人演奏动作生成问题。提出PianoFlow框架，结合MIDI与音频信息，实现精确的跨手协调和实时长序列生成。**

- **链接: [https://arxiv.org/pdf/2604.12856](https://arxiv.org/pdf/2604.12856)**

> **作者:** Xuan Wang; Kai Ruan; Jiayi Han; kaiyue Zhou; Gaoang Wang
>
> **摘要:** Audio-driven bimanual piano motion generation requires precise modeling of complex musical structures and dynamic cross-hand coordination. However, existing methods often rely on acoustic-only representations lacking symbolic priors, employ inflexible interaction mechanisms, and are limited to computationally expensive short-sequence generation. To address these limitations, we propose PianoFlow, a flow-matching framework for precise and coordinated bimanual piano motion synthesis. Our approach strategically leverages MIDI as a privileged modality during training, distilling these structured musical priors to achieve deep semantic understanding while maintaining audio-only inference. Furthermore, we introduce an asymmetric role-gated interaction module to explicitly capture dynamic cross-hand coordination through role-aware attention and temporal gating. To enable real-time streaming generation for arbitrarily long sequences, we design an autoregressive flow continuation scheme that ensures seamless cross-chunk temporal coherence. Extensive experiments on the PianoMotion10M dataset demonstrate that PianoFlow achieves superior quantitative and qualitative performance, while accelerating inference by over 9\times compared to previous methods.
>
---
#### [new 116] ARGen: Affect-Reinforced Generative Augmentation towards Vision-based Dynamic Emotion Perception
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉情感感知任务，解决数据稀缺和长尾分布问题。提出ARGen框架，通过生成增强提升情绪识别性能。**

- **链接: [https://arxiv.org/pdf/2604.12255](https://arxiv.org/pdf/2604.12255)**

> **作者:** Huanzhen Wang; Ziheng Zhou; Jiaqi Song; Li He; Yunshi Lan; Yan Wang; Wenqiang Zhang
>
> **摘要:** Dynamic facial expression recognition in the wild remains challenging due to data scarcity and long-tail distributions, which hinder models from effectively learning the temporal dynamics of scarce emotions. To address these limitations, we propose ARGen, an Affect-Reinforced Generative Augmentation Framework that enables data-adaptive dynamic expression generation for robust emotion perception. ARGen operates in two stages: Affective Semantic Injection (ASI) and Adaptive Reinforcement Diffusion (ARD). The ASI stage establishes affective knowledge alignment through facial Action Units and employs a retrieval-augmented prompt generation strategy to synthesize consistent and fine-grained affective descriptions via large-scale visual-language models, thereby injecting interpretable emotional priors into the generation process. The ARD stage integrates text-conditioned image-to-video diffusion with reinforcement learning, introducing inter-frame conditional guidance and a multi-objective reward function to jointly optimize expression naturalness, facial integrity, and generative efficiency. Extensive experiments on both generation and recognition tasks verify that ARGen substantially enhances synthesis fidelity and improves recognition performance, establishing an interpretable and generalizable generative augmentation paradigm for vision-based affective computing.
>
---
#### [new 117] Euler-inspired Decoupling Neural Operator for Efficient Pansharpening
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像融合任务，旨在解决 pansharpening 中的光谱-空间模糊和计算成本高的问题。提出 EDNO 框架，通过频率域处理实现高效融合。**

- **链接: [https://arxiv.org/pdf/2604.12463](https://arxiv.org/pdf/2604.12463)**

> **作者:** Anqi Zhu; Mengting Ma; Yizhen Jiang; Xiangdong Li; Kai Zheng; Jiaxin Li; Wei Zhang
>
> **摘要:** Pansharpening aims to synthesize high-resolution multispectral (HR-MS) images by fusing the spatial textures of panchromatic (PAN) images with the spectral information of low-resolution multispectral (LR-MS) images. While recent deep learning paradigms, especially diffusion-based operators, have pushed the performance boundaries, they often encounter spectral-spatial blurring and prohibitive computational costs due to their stochastic nature and iterative sampling. In this paper, we propose the Euler-inspired Decoupling Neural Operator (EDNO), a physics-inspired framework that redefines pansharpening as a continuous functional mapping in the frequency domain. Departing from conventional Cartesian feature processing, our EDNO leverages Euler's formula to transform features into a polar coordinate system, enabling a novel explicit-implicit interaction mechanism. Specifically, we develop the Euler Feature Interaction Layer (EFIL), which decouples the fusion task into two specialized modules: 1) Explicit Feature Interaction Module, utilizing a linear weighting scheme to simulate phase rotation for adaptive geometric alignment; and 2) Implicit Feature Interaction Module, employing a feed-forward network to model spectral distributions for superior color consistency. By operating in the frequency domain, EDNO inherently captures global receptive fields while maintaining discretization-invariance. Experimental results on the three datasets demonstrate that EDNO offers a superior efficiency-performance balance compared to heavyweight architectures.
>
---
#### [new 118] Ride the Wave: Precision-Allocated Sparse Attention for Smooth Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成任务，旨在解决视频扩散Transformer计算负担重和视觉闪烁问题。提出PASA框架，通过动态预算分配、分组近似和随机选择机制，实现高效且平滑的视频生成。**

- **链接: [https://arxiv.org/pdf/2604.12219](https://arxiv.org/pdf/2604.12219)**

> **作者:** Wentai Zhang; Ronghui Xi; Shiyao Peng; Jiayu Huang; Haoran Luo; Zichen Tang; Haihong E
>
> **摘要:** Video Diffusion Transformers have revolutionized high-fidelity video generation but suffer from the massive computational burden of self-attention. While sparse attention provides a promising acceleration solution, existing methods frequently provoke severe visual flickering caused by static sparsity patterns and deterministic block routing. To resolve these limitations, we propose Precision-Allocated Sparse Attention (PASA), a training-free framework designed for highly efficient and temporally smooth video generation. First, we implement a curvature-aware dynamic budgeting mechanism. By profiling the generation trajectory acceleration across timesteps, we elastically allocate the exact-computation budget to secure high-precision processing strictly during critical semantic transitions. Second, we replace global homogenizing estimations with hardware-aligned grouped approximations, successfully capturing fine-grained local variations while maintaining peak compute throughput. Finally, we incorporate a stochastic selection bias into the attention routing mechanism. This probabilistic approach softens rigid selection boundaries and eliminates selection oscillation, effectively eradicating the localized computational starvation that drives temporal flickering. Extensive evaluations on leading video diffusion models demonstrate that PASA achieves substantial inference acceleration while consistently producing remarkably fluid and structurally stable video sequences.
>
---
#### [new 119] Scaling Exposes the Trigger: Input-Level Backdoor Detection in Text-to-Image Diffusion Models via Cross-Attention Scaling
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于文本到图像生成模型的后门检测任务，旨在解决隐蔽触发下的输入级后门检测问题。通过分析交叉注意力机制的响应差异，提出SET框架实现有效检测。**

- **链接: [https://arxiv.org/pdf/2604.12446](https://arxiv.org/pdf/2604.12446)**

> **作者:** Zida Li; Jun Li; Yuzhe Sha; Ziqiang Li; Lizhi Xiong; Zhangjie Fu
>
> **备注:** Under Review
>
> **摘要:** Text-to-image (T2I) diffusion models have achieved remarkable success in image synthesis, but their reliance on large-scale data and open ecosystems introduces serious backdoor security risks. Existing defenses, particularly input-level methods, are more practical for deployment but often rely on observable anomalies that become unreliable under stealthy, semantics-preserving trigger designs. As modern backdoor attacks increasingly embed triggers into natural inputs, these methods degrade substantially, raising a critical question: can more stable, implicit, and trigger-agnostic differences between benign and backdoor inputs be exploited for detection? In this work, we address this challenge from an active probing perspective. We introduce controlled scaling perturbations on cross-attention and uncover a novel phenomenon termed Cross-Attention Scaling Response Divergence (CSRD), where benign and backdoor inputs exhibit systematically different response evolution patterns across denoising steps. Building on this insight, we propose SET, an input-level backdoor detection framework that constructs response-offset features under multi-scale perturbations and learns a compact benign response space from a small set of clean samples. Detection is then performed by measuring deviations from this learned space, without requiring prior knowledge of the attack or access to model training. Extensive experiments demonstrate that SET consistently outperforms existing baselines across diverse attack methods, trigger types, and model settings, with particularly strong gains under stealthy implicit-trigger scenarios. Overall, SET improves AUROC by 9.1% and ACC by 6.5% over the best baseline, highlighting its effectiveness and robustness for practical deployment.
>
---
#### [new 120] Decoding by Perturbation: Mitigating MLLM Hallucinations via Dynamic Textual Perturbation
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态语言模型任务，旨在解决模型推理中的幻觉问题。通过动态文本扰动方法，抑制语言先验带来的偏差，提升视觉 grounding 的稳定性。**

- **链接: [https://arxiv.org/pdf/2604.12424](https://arxiv.org/pdf/2604.12424)**

> **作者:** Sihang Jia; Shuliang Liu; Songbo Yang; Yibo Yan; Xin Zou; Xuming Hu
>
> **摘要:** Multimodal Large Language Models frequently suffer from inference hallucinations, partially stemming from language priors dominating visual evidence. Existing training-free mitigation methods either perturb the visual representation and deviate from the natural image distribution, or enforce intrusive manipulations that compromise the model's inherent generative fluency. We introduce a novel perspective that multimodal hallucination manifests as the hypersensitivity of visual grounding to textual phrasing during the decoding phase. Building on this insight, we propose Decoding by Perturbation (DeP), a training-free framework mitigating prior-induced hallucinations via controlled textual interventions. DeP employs a dynamic probe applying multi-level textual perturbations to elicit latent language priors. Leveraging attention variance, it enhances stable evidence regions while suppressing suspicious noise in the feature space. Furthermore, it constructs an interpretable prior drift direction using logits statistics to counteract probability biases from textual co-occurrences. Extensive experiments confirm DeP effectively reduces hallucinations and achieves superior performance across multiple benchmarks.
>
---
#### [new 121] DINO-Explorer: Active Underwater Discovery via Ego-Motion Compensated Semantic Predictive Coding
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DINO-Explorer，用于水下主动感知任务，解决AUV被动记录遗漏重要事件的问题，通过语义预测和自运动补偿实现高效环境监测。**

- **链接: [https://arxiv.org/pdf/2604.12933](https://arxiv.org/pdf/2604.12933)**

> **作者:** Yuhan Jin; Nayari Marie Lessa; Mariela De Lucas Alvarez; Melvin Laux; Lucas Amparo Barbosa; Frank Kirchner; Rebecca Adam
>
> **摘要:** Marine ecosystem degradation necessitates continuous, scientifically selective underwater monitoring. However, most autonomous underwater vehicles (AUVs) operate as passive data loggers, capturing exhaustive video for offline review and frequently missing transient events of high scientific value. Transitioning to active perception requires a causal, online signal that highlights significant phenomena while suppressing maneuver-induced visual changes. We propose DINO-Explorer, a novelty-aware perception framework driven by a continuous semantic surprise signal. Operating within the latent space of a frozen DINOv3 foundation model, it leverages a lightweight, action-conditioned recurrent predictor to anticipate short-horizon semantic evolution. An efference-copy-inspired module utilizes globally pooled optical flow to discount self-induced visual changes without suppressing genuine environmental novelty. We evaluate this signal on the downstream task of asynchronous event triage under variant telemetry constraints. Results demonstrate that DINO-Explorer provides a robust, bandwidth-efficient attention mechanism. At a fixed operating point, the system retains 78.8% of post-discovery human-reviewer consensus events with a 56.8% trigger confirmation rate, effectively surfacing mission-relevant phenomena. Crucially, ego-motion conditioning suppresses 45.5% of false positives relative to an uncompensated surprise signal baseline. In a replay-side Pareto ablation study, DINO-Explorer robustly dominates the validated peak F1 versus telemetry bandwidth frontier, reducing telemetry bandwidth by 48.2% at the selected operating point while maintaining a 62.2% peak F1 score, successfully concentrating data transmission around human-verified novelty events.
>
---
#### [new 122] CBAM-Enhanced DenseNet121 for Multi-Class Chest X-Ray Classification with Grad-CAM Explainability
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于多类胸部X光分类任务，旨在区分正常、细菌性肺炎和病毒性肺炎。通过集成CBAM模块改进DenseNet121模型，提升分类性能并提供可解释性。**

- **链接: [https://arxiv.org/pdf/2604.12305](https://arxiv.org/pdf/2604.12305)**

> **作者:** Utsho Kumar Dey
>
> **备注:** 10 pages, 7 figures, 2 tables. Preprint submitted to IEEE Access
>
> **摘要:** Pneumonia remains a leading cause of childhood mortality worldwide, with a heavy burden in low-resource settings such as Bangladesh where radiologist availability is limited. Most existing deep learning approaches treat pneumonia detection as a binary problem, overlooking the clinically critical distinction between bacterial and viral aetiology. This paper proposes CBAM-DenseNet121, a transfer-learning framework that integrates the Convolutional Block Attention Module (CBAM) into DenseNet121 for three-class chest X-ray classification: Normal, Bacterial Pneumonia, and Viral Pneumonia. We also conduct a systematic binary-task baseline study revealing that EfficientNetB3 (73.88%) underperforms even the custom CNN baseline (78.53%) -- a practically important negative finding for medical imaging model selection. To ensure statistical reliability, all experiments were repeated three times with independent random seeds (42, 7, 123), and results are reported as mean +/- standard deviation. CBAM-DenseNet121 achieves 84.29% +/- 1.14% test accuracy with per-class AUC scores of 0.9565 +/- 0.0010, 0.9610 +/- 0.0014, and 0.9187 +/- 0.0037 for bacterial pneumonia, normal, and viral pneumonia respectively. Grad-CAM visualizations confirm that the model attends to anatomically plausible pulmonary regions for each class, supporting interpretable deployment in resource-constrained clinical environments.
>
---
#### [new 123] Whole-Body Mobile Manipulation using Offline Reinforcement Learning on Sub-optimal Controllers
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究移动操作任务，解决传统控制器依赖人工调优和学习方法数据成本高的问题。通过离线强化学习改进子优控制器，提升机器人操作性能。**

- **链接: [https://arxiv.org/pdf/2604.12509](https://arxiv.org/pdf/2604.12509)**

> **作者:** Snehal Jauhri; Vignesh Prasad; Georgia Chalvatzaki
>
> **备注:** PrePrint. Project website: this http URL
>
> **摘要:** Mobile Manipulation (MoMa) of articulated objects, such as opening doors, drawers, and cupboards, demands simultaneous, whole-body coordination between a robot's base and arms. Classical whole-body controllers (WBCs) can solve such problems via hierarchical optimization, but require extensive hand-tuned optimization and remain brittle. Learning-based methods, on the other hand, show strong generalization capabilities but typically rely on expensive whole-body teleoperation data or heavy reward engineering. We observe that even a sub-optimal WBC is a powerful structural prior: it can be used to collect data in a constrained, task-relevant region of the state-action space, and its behavior can still be improved upon using offline reinforcement learning. Building on this, we propose WHOLE-MoMa, a two-stage pipeline that first generates diverse demonstrations by randomizing a lightweight WBC, and then applies offline RL to identify and stitch together improved behaviors via a reward signal. To support the expressive action-chunked diffusion policies needed for complex coordination tasks, we extend offline implicit Q-learning with Q-chunking for chunk-level critic evaluation and advantage-weighted policy extraction. On three tasks of increasing difficulty using a TIAGo++ mobile manipulator in simulation, WHOLE-MoMa significantly outperforms WBC, behavior cloning, and several offline RL baselines. Policies transfer directly to the real robot without finetuning, achieving 80% success in bimanual drawer manipulation and 68% in simultaneous cupboard opening and object placement, all without any teleoperated or real-world training data.
>
---
#### [new 124] SubFlow: Sub-mode Conditioned Flow Matching for Diverse One-Step Generation
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于生成模型任务，解决一阶段生成中多样性不足的问题。通过子模式条件流匹配，提升生成多样性并保持质量。**

- **链接: [https://arxiv.org/pdf/2604.12273](https://arxiv.org/pdf/2604.12273)**

> **作者:** Yexiong Lin; Jia Shi; Shanshan Ye; Wanyu Wang; Yu Yao; Tongliang Liu
>
> **摘要:** Flow matching has emerged as a powerful generative framework, with recent few-step methods achieving remarkable inference acceleration. However, we identify a critical yet overlooked limitation: these models suffer from severe diversity degradation, concentrating samples on dominant modes while neglecting rare but valid variations of the target distribution. We trace this degradation to averaging distortion: when trained with MSE objectives, class-conditional flows learn a frequency-weighted mean over intra-class sub-modes, causing the model to over-represent high-density modes while systematically neglecting low-density ones. To address this, we propose SubFlow, Sub-mode Conditioned Flow Matching, which eliminates averaging distortion by decomposing each class into fine-grained sub-modes via semantic clustering and conditioning the flow on sub-mode indices. Each conditioned sub-distribution is approximately unimodal, so the learned flow accurately targets individual modes with no averaging distortion, restoring full mode coverage in a single inference step. Crucially, SubFlow is entirely plug-and-play: it integrates seamlessly into existing one-step models such as MeanFlow and Shortcut Models without any architectural modifications. Extensive experiments on ImageNet-256 demonstrate that SubFlow yields substantial gains in generation diversity (Recall) while maintaining competitive image quality (FID), confirming its broad applicability across different one-step generation frameworks. Project page: this https URL.
>
---
#### [new 125] ReefMapGS: Enabling Large-Scale Underwater Reconstruction by Closing the Loop Between Multimodal SLAM and Gaussian Splatting
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于 underwater 3D重建任务，解决传统方法依赖计算密集型姿态估计的问题。提出ReefMapGS框架，结合多模态SLAM与3D高斯点云，实现高效、准确的水下场景重建与姿态估计。**

- **链接: [https://arxiv.org/pdf/2604.11992](https://arxiv.org/pdf/2604.11992)**

> **作者:** Daniel Yang; Jungseok Hong; John J. Leonard; Yogesh Girdhar
>
> **摘要:** 3D Gaussian Splatting is a powerful visual representation, providing high-quality and efficient 3D scene reconstruction, but it is crucially dependent on accurate camera poses typically obtained from computationally intensive processes like structure-from-motion that are unsuitable for field robot applications. However, in these domains, multimodal sensor data from acoustic, inertial, pressure, and visual sensors are available and suitable for pose-graph optimization-based SLAM methods that can estimate the vehicle's trajectory and thus our needed camera poses while providing uncertainty. We propose a 3DGS-based incremental reconstruction framework, ReefMapGS, that builds an initial model from a high certainty region and progressively expands to incorporate the whole scene. We reconstruct the scene incrementally by interleaving local tracking of new image observations with optimization of the underlying 3DGS scene. These refined poses are integrated back into the pose-graph to globally optimize the whole trajectory. We show COLMAP-free 3D reconstruction of two underwater reef sites with complex geometry as well as more accurate global pose estimation of our AUV over survey trajectories spanning up to 700 m.
>
---
#### [new 126] Evolution of Optimization Methods: Algorithms, Scenarios, and Evaluations
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于深度学习优化任务，旨在解决传统优化方法在隐私、效率和通用性上的不足。通过分析算法演进与实验评估，提出改进方向。**

- **链接: [https://arxiv.org/pdf/2604.12968](https://arxiv.org/pdf/2604.12968)**

> **作者:** Tong Zhang; Jiangning Zhang; Zhucun Xue; Juntao Jiang; Yicheng Xu; Chengming Xu; Teng Hu; Xingyu Xie; Xiaobin Hu; Yabiao Wang; Yong Liu; Shuicheng Yan
>
> **摘要:** Balancing convergence speed, generalization capability, and computational efficiency remains a core challenge in deep learning optimization. First-order gradient descent methods, epitomized by stochastic gradient descent (SGD) and Adam, serve as the cornerstone of modern training pipelines. However, large-scale model training, stringent differential privacy requirements, and distributed learning paradigms expose critical limitations in these conventional approaches regarding privacy protection and memory efficiency. To mitigate these bottlenecks, researchers explore second-order optimization techniques to surpass first-order performance ceilings, while zeroth-order methods reemerge to alleviate memory constraints inherent to large-scale training. Despite this proliferation of methodologies, the field lacks a cohesive framework that unifies underlying principles and delineates application scenarios for these disparate approaches. In this work, we retrospectively analyze the evolutionary trajectory of deep learning optimization algorithms and present a comprehensive empirical evaluation of mainstream optimizers across diverse model architectures and training scenarios. We distill key emerging trends and fundamental design trade-offs, pinpointing promising directions for future research. By synthesizing theoretical insights with extensive empirical evidence, we provide actionable guidance for designing next-generation highly efficient, robust, and trustworthy optimization methods. The code is available at this https URL.
>
---
#### [new 127] GlotOCR Bench: OCR Models Still Struggle Beyond a Handful of Unicode Scripts
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于OCR任务，旨在解决现有模型在多语言脚本上的泛化能力不足问题。通过构建涵盖100+ Unicode脚本的基准测试，评估模型表现并揭示其依赖预训练覆盖的问题。**

- **链接: [https://arxiv.org/pdf/2604.12978](https://arxiv.org/pdf/2604.12978)**

> **作者:** Amir Hossein Kargaran; Nafiseh Nikeghbal; Jana Diesner; François Yvon; Hinrich Schütze
>
> **摘要:** Optical character recognition (OCR) has advanced rapidly with the rise of vision-language models, yet evaluation has remained concentrated on a small cluster of high- and mid-resource scripts. We introduce GlotOCR Bench, a comprehensive benchmark evaluating OCR generalization across 100+ Unicode scripts. Our benchmark comprises clean and degraded image variants rendered from real multilingual texts. Images are rendered using fonts from the Google Fonts repository, shaped with HarfBuzz and rasterized with FreeType, supporting both LTR and RTL scripts. Samples of rendered images were manually reviewed to verify correct rendering across all scripts. We evaluate a broad suite of open-weight and proprietary vision-language models and find that most perform well on fewer than ten scripts, and even the strongest frontier models fail to generalize beyond thirty scripts. Performance broadly tracks script-level pretraining coverage, suggesting that current OCR systems rely on language model pretraining as much as on visual recognition. Models confronted with unfamiliar scripts either produce random noise or hallucinate characters from similar scripts they already know. We release the benchmark and pipeline for reproducibility. Pipeline Code: this https URL, Benchmark: this https URL.
>
---
#### [new 128] Spatial Atlas: Compute-Grounded Reasoning for Spatial-Aware Research Agent Benchmarks
- **分类: cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出Compute-Grounded Reasoning（CGR），用于空间感知研究代理，解决空间推理与机器学习任务中的幻觉问题，通过结构化计算提升准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2604.12102](https://arxiv.org/pdf/2604.12102)**

> **作者:** Arun Sharma
>
> **备注:** 11 pages. Submitted to NeurIPS 2026. Code: this https URL
>
> **摘要:** We introduce compute-grounded reasoning (CGR), a design paradigm for spatial-aware research agents in which every answerable sub-problem is resolved by deterministic computation before a language model is asked to generate. Spatial Atlas instantiates CGR as a single Agent-to-Agent (A2A) server that handles two challenging benchmarks: FieldWorkArena, a multimodal spatial question-answering benchmark spanning factory, warehouse, and retail environments, and MLE-Bench, a suite of 75 Kaggle machine learning competitions requiring end-to-end ML engineering. A structured spatial scene graph engine extracts entities and relations from vision descriptions, computes distances and safety violations deterministically, then feeds computed facts to large language models, thereby avoiding hallucinated spatial reasoning. Entropy-guided action selection maximizes information gain per step and routes queries across a three-tier frontier model stack (OpenAI + Anthropic). A self-healing ML pipeline with strategy-aware code generation, a score-driven iterative refinement loop, and a prompt-based leak audit registry round out the system. We evaluate across both benchmarks and show that CGR yields competitive accuracy while maintaining interpretability through structured intermediate representations and deterministic spatial computations.
>
---
#### [new 129] CoSyncDiT: Cognitive Synchronous Diffusion Transformer for Movie Dubbing
- **分类: cs.SD; cs.CV; cs.MM**

- **简介: 该论文属于电影配音任务，旨在解决语音与唇形同步及自然度不足的问题。提出CoSync-DiT框架，通过认知同步机制提升对齐精度和语音质量。**

- **链接: [https://arxiv.org/pdf/2604.12292](https://arxiv.org/pdf/2604.12292)**

> **作者:** Gaoxiang Cong; Liang Li; Jiaxin Ye; Zhedong Zhang; Hongming Shan; Yuankai Qi; Qingming Huang
>
> **摘要:** Movie dubbing aims to synthesize speech that preserves the vocal identity of a reference audio while synchronizing with the lip movements in a target video. Existing methods fail to achieve precise lip-sync and lack naturalness due to explicit alignment at the duration level. While implicit alignment solutions have emerged, they remain susceptible to interference from the reference audio, triggering timbre and pronunciation degradation in in-the-wild scenarios. In this paper, we propose a novel flow matching-based movie dubbing framework driven by the Cognitive Synchronous Diffusion Transformer (CoSync-DiT), inspired by the cognitive process of professional actors. This architecture progressively guides the noise-to-speech generative trajectory by executing acoustic style adapting, fine-grained visual calibrating, and time-aware context aligning. Furthermore, we design the Joint Semantic and Alignment Regularization (JSAR) mechanism to simultaneously constrain frame-level temporal consistency on the contextual outputs and semantic consistency on the flow hidden states, ensuring robust alignment. Extensive experiments on both standard benchmarks and challenging in-the-wild dubbing benchmarks demonstrate that our method achieves the state-of-the-art performance across multiple metrics.
>
---
#### [new 130] Socrates Loss: Unifying Confidence Calibration and Classification by Leveraging the Unknown
- **分类: cs.LG; cs.AI; cs.CV; cs.NE**

- **简介: 该论文属于深度学习中的模型校准任务，旨在解决神经网络分类准确率与置信度不匹配的问题。通过引入未知类构建统一损失函数，提升模型稳定性与校准效果。**

- **链接: [https://arxiv.org/pdf/2604.12245](https://arxiv.org/pdf/2604.12245)**

> **作者:** Sandra Gómez-Gálvez; Tobias Olenyi; Gillian Dobbie; Katerina Taškova
>
> **备注:** Published at TMLR 2026. this https URL Video: this https URL Code: this https URL
>
> **摘要:** Deep neural networks, despite their high accuracy, often exhibit poor confidence calibration, limiting their reliability in high-stakes applications. Current ad-hoc confidence calibration methods attempt to fix this during training but face a fundamental trade-off: two-phase training methods achieve strong classification performance at the cost of training instability and poorer confidence calibration, while single-loss methods are stable but underperform in classification. This paper addresses and mitigates this stability-performance trade-off. We propose Socrates Loss, a novel, unified loss function that explicitly leverages uncertainty by incorporating an auxiliary unknown class, whose predictions directly influence the loss function and a dynamic uncertainty penalty. This unified objective allows the model to be optimized for both classification and confidence calibration simultaneously, without the instability of complex, scheduled losses. We provide theoretical guarantees that our method regularizes the model to prevent miscalibration and overfitting. Across four benchmark datasets and multiple architectures, our comprehensive experiments demonstrate that Socrates Loss consistently improves training stability while achieving more favorable accuracy-calibration trade-off, often converging faster than existing methods.
>
---
#### [new 131] QMC-Net: Data-Aware Quantum Representations for Remote Sensing Image Classification
- **分类: quant-ph; cs.CV**

- **简介: 该论文属于遥感图像分类任务，旨在解决量子电路设计与数据特性不匹配的问题。通过将带级统计信息映射到量子电路参数，提出QMC-Net框架，实现自适应量子特征编码。**

- **链接: [https://arxiv.org/pdf/2604.11817](https://arxiv.org/pdf/2604.11817)**

> **作者:** Md Aminur Hossain; Ayush V. Patel; Biplab Banerjee
>
> **备注:** Accepted in ICPR 2026, 15 pages
>
> **摘要:** Hybrid quantum-classical models offer a promising route for learning from complex data; however, their application to multi-band remote sensing imagery often relies on generic, data-agnostic quantum circuits that fail to account for channel-specific statistical variability. In this work, we propose a data-driven framework that maps band-level statistics such as Shannon Entropy, Variance, Spectral Flatness, and Edge Density to the hyperparameters of customized quantum circuits. Building on this framework, we introduce QMC-Net, a hybrid architecture that processes six data channels using band-specific quantum circuits, enabling adaptive quantum feature encoding and transformation across channels. Experiments on the EuroSAT and SAT-6 datasets demonstrate that QMC-Net achieves accuracies of 93.80 % and 99.34 %, respectively, while a residual-enhanced variant further improves performance to 94.69 % and 99.39 %. These results consistently outperform strong classical baselines and monolithic hybrid quantum models, highlighting the effectiveness of data-aware quantum circuit design under NISQ constraints.
>
---
#### [new 132] Probabilistic Feature Imputation and Uncertainty-Aware Multimodal Federated Aggregation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于联邦学习任务，解决多模态数据缺失问题。提出P-FIN网络，输出不确定性估计，提升医疗分类可靠性。**

- **链接: [https://arxiv.org/pdf/2604.12970](https://arxiv.org/pdf/2604.12970)**

> **作者:** Nafis Fuad Shahid; Maroof Ahmed; Md Akib Haider; Saidur Rahman Sagor; Aashnan Rahman; Md Azam Hossain
>
> **备注:** Accepted for publication at the Medical Imaging with Deep Learning (MIDL) 2026 conference
>
> **摘要:** Multimodal federated learning enables privacy-preserving collaborative model training across healthcare institutions. However, a fundamental challenge arises from modality heterogeneity: many clinical sites possess only a subset of modalities due to resource constraints or workflow variations. Existing approaches address this through feature imputation networks that synthesize missing modality representations, yet these methods produce point estimates without reliability measures, forcing downstream classifiers to treat all imputed features as equally trustworthy. In safety-critical medical applications, this limitation poses significant risks. We propose the Probabilistic Feature Imputation Network (P-FIN), which outputs calibrated uncertainty estimates alongside imputed features. This uncertainty is leveraged at two levels: (1) locally, through sigmoid gating that attenuates unreliable feature dimensions before classification, and (2) globally, through Fed-UQ-Avg, an aggregation strategy that prioritizes updates from clients with reliable imputation. Experiments on federated chest X-ray classification using CheXpert, NIH Open-I, and PadChest demonstrate consistent improvements over deterministic baselines, with +5.36% AUC gain in the most challenging configuration.
>
---
#### [new 133] ReflectCAP: Detailed Image Captioning with Reflective Memory
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于图像描述任务，旨在解决详细图像描述中事实准确性和细节覆盖的平衡问题。提出ReflectCAP方法，通过反思笔记引导模型生成更准确且全面的描述。**

- **链接: [https://arxiv.org/pdf/2604.12357](https://arxiv.org/pdf/2604.12357)**

> **作者:** Kyungmin Min; Minbeom Kim; Kang-il Lee; Seunghyun Yoon; Kyomin Jung
>
> **摘要:** Detailed image captioning demands both factual grounding and fine-grained coverage, yet existing methods have struggled to achieve them simultaneously. We address this tension with Reflective Note-Guided Captioning (ReflectCAP), where a multi-agent pipeline analyzes what the target large vision-language model (LVLM) consistently hallucinates and what it systematically overlooks, distilling these patterns into reusable guidelines called Structured Reflection Notes. At inference time, these notes steer the captioning model along both axes -- what to avoid and what to attend to -- yielding detailed captions that jointly improve factuality and coverage. Applying this method to 8 LVLMs spanning the GPT-4.1 family, Qwen series, and InternVL variants, ReflectCAP reaches the Pareto frontier of the trade-off between factuality and coverage, and delivers substantial gains on CapArena-Auto, where generated captions are judged head-to-head against strong reference models. Moreover, ReflectCAP offers a more favorable trade-off between caption quality and compute cost than model scaling or existing multi-agent pipelines, which incur 21--36\% greater overhead. This makes high-quality detailed captioning viable under real-world cost and latency constraints.
>
---
#### [new 134] Benchmarking Deflection and Hallucination in Large Vision-Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于视觉语言模型评估任务，旨在解决现有基准在检索依赖性和幻觉检测上的不足。通过构建动态数据集和新基准，评估模型在证据冲突时的应对能力。**

- **链接: [https://arxiv.org/pdf/2604.12033](https://arxiv.org/pdf/2604.12033)**

> **作者:** Nicholas Moratelli; Christopher Davis; Leonardo F. R. Ribeiro; Bill Byrne; Gonzalo Iglesias
>
> **备注:** Accepted to ACL 2026
>
> **摘要:** Large Vision-Language Models (LVLMs) increasingly rely on retrieval to answer knowledge-intensive multimodal questions. Existing benchmarks overlook conflicts between visual and textual evidence and the importance of generating deflections (e.g., Sorry, I cannot answer...) when retrieved knowledge is incomplete. These benchmarks also suffer from rapid obsolescence, as growing LVLM training sets allow models to answer many questions without retrieval. We address these gaps with three contributions. First, we propose a dynamic data curation pipeline that preserves benchmark difficulty over time by filtering for genuinely retrieval-dependent samples. Second, we introduce VLM-DeflectionBench, a benchmark of 2,775 samples spanning diverse multimodal retrieval settings, designed to probe model behaviour under conflicting or insufficient evidence. Third, we define a fine-grained evaluation protocol with four scenarios that disentangle parametric memorization from retrieval robustness. Experiments across 20 state-of-the-art LVLMs indicate that models usually fail to deflect in the presence of noisy or misleading evidence. Our results highlight the need to evaluate not only what models know, but how they behave when they do not, and serve as a reusable and extensible benchmark for reliable KB-VQA evaluation. All resources will be publicly available upon publication.
>
---
#### [new 135] Adaptive Data Dropout: Towards Self-Regulated Learning in Deep Neural Networks
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于深度学习任务，旨在解决训练效率与泛化能力的平衡问题。提出Adaptive Data Dropout方法，动态调整训练数据，提升模型效率和稳定性。**

- **链接: [https://arxiv.org/pdf/2604.12945](https://arxiv.org/pdf/2604.12945)**

> **作者:** Amar Gahir; Varshil Patel; Shreyank N Gowda
>
> **摘要:** Deep neural networks are typically trained by uniformly sampling large datasets across epochs, despite evidence that not all samples contribute equally throughout learning. Recent work shows that progressively reducing the amount of training data can improve efficiency and generalization, but existing methods rely on fixed schedules that do not adapt during training. In this work, we propose Adaptive Data Dropout, a simple framework that dynamically adjusts the subset of training data based on performance feedback. Inspired by self-regulated learning, our approach treats data selection as an adaptive process, increasing or decreasing data exposure in response to changes in training accuracy. We introduce a lightweight stochastic update mechanism that modulates the dropout schedule online, allowing the model to balance exploration and consolidation over time. Experiments on standard image classification benchmarks show that our method reduces effective training steps while maintaining competitive accuracy compared to static data dropout strategies. These results highlight adaptive data selection as a promising direction for efficient and robust training. Code will be released.
>
---
#### [new 136] DoseRAD2026 Challenge dataset: AI accelerated photon and proton dose calculation for radiotherapy
- **分类: physics.med-ph; cs.AI; cs.CV**

- **简介: 本文介绍DoseRAD2026数据集，用于加速放射治疗中的光子和质子剂量计算。解决快速准确剂量计算问题，包含CT/MRI配对数据及蒙特卡洛剂量分布。**

- **链接: [https://arxiv.org/pdf/2604.12778](https://arxiv.org/pdf/2604.12778)**

> **作者:** Fan Xiao; Nikolaos Delopoulos; Niklas Wahl; Lennart Volz; Lina Bucher; Matteo Maspero; Miguel Palacios; Muheng Li; Samir Schulz; Viktor Rogowski; Ye Zhang; Zoltan Perko; Christopher Kurz; George Dedes; Guillaume Landry; Adrian Thummerer
>
> **摘要:** Purpose: Accurate dose calculation is essential in radiotherapy for precise tumor irradiation while sparing healthy tissue. With the growing adoption of MRI-guided and real-time adaptive radiotherapy, fast and accurate dose calculation on CT and MRI is increasingly needed. The DoseRAD2026 dataset and challenge provide a public benchmark of paired CT and MRI data with beam-level photon and proton Monte Carlo dose distributions for developing and evaluating advanced dose calculation methods. Acquisition and validation methods: The dataset comprises paired CT and MRI from 115 patients (75 training, 40 testing) treated on an MRI-linac for thoracic or abdominal lesions, derived from the SynthRAD2025 dataset. Pre-processing included deformable image registration, air-cavity correction, and resampling. Ground-truth photon (6 MV) and proton dose distributions were computed using open-source Monte Carlo algorithms, yielding 40,500 photon beams and 81,000 proton beamlets. Data format and usage notes: Data are organized into photon and proton subsets with paired CT-MRI images, beam-level dose distributions, and JSON beam configuration files. Files are provided in compressed MetaImage (.mha) format. The dataset is released under CC BY-NC 4.0, with training data available from April 2026 and the test set withheld until March 2030. Potential applications: The dataset supports benchmarking of fast dose calculation methods, including beam-level dose estimation for photon and proton therapy, MRI-based dose calculation in MRI-guided workflows, and real-time adaptive radiotherapy.
>
---
#### [new 137] Information-Theoretic Optimization for Task-Adapted Compressed Sensing Magnetic Resonance Imaging
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于医学影像重建任务，解决传统CS-MRI在临床任务适应性和不确定性预测上的不足。提出基于信息论的优化方法，实现自适应采样与任务推理的联合优化。**

- **链接: [https://arxiv.org/pdf/2604.12709](https://arxiv.org/pdf/2604.12709)**

> **作者:** Xinyu Peng; Ziyang Zheng; Wenrui Dai; Duoduo Xue; Shaohui Li; Chenglin Li; Junni Zou; Hongkai Xiong
>
> **备注:** 68 pages, 15 figures, accepted by IEEE TPAMI
>
> **摘要:** Task-adapted compressed sensing magnetic resonance imaging (CS-MRI) is emerging to address the specific demands of downstream clinical tasks with significantly fewer k-space measurements than required by Nyquist sampling. However, existing task-adapted CS-MRI methods suffer from the uncertainty problem for medical diagnosis and cannot achieve adaptive sampling in end-to-end optimization with reconstruction or clinical tasks. To address these limitations, we propose the first task-adapted CS-MRI from the information-theoretic perspective to simultaneously achieve probabilistic inference for uncertainty prediction and adapt to arbitrary sampling ratios and versatile clinical applications. Specifically, we formalize the task-adapted CS-MRI optimization problem by maximizing the mutual information between undersampled k-space measurements and clinical tasks to enable probabilistic inference for addressing the uncertainty problem. We leverage amortized optimization and construct tractable variational bounds for mutual information to jointly optimize sampling, reconstruction, and task-inference models, which enables flexible sampling ratio control using a single end-to-end trained model. Furthermore, the proposed framework addresses two kinds of distinct clinical scenarios within a unified approach, i.e., i) joint task and reconstruction, where reconstruction serves as an auxiliary process to enhance task performance; and ii) task implementation with suppressed reconstruction, applicable for privacy protection. Extensive experiments on large-scale MRI datasets demonstrate that the proposed framework achieves highly competitive performance on standard metrics like Dice compared to deterministic counterpart but provides better distribution matching to the ground-truth posterior distribution as measured by the generalized energy distance (GED).
>
---
#### [new 138] Habitat-GS: A High-Fidelity Navigation Simulator with Dynamic Gaussian Splatting
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于 embodied AI 导航任务，旨在提升模拟环境的视觉真实性和动态人类建模能力。工作包括引入 3D Gaussian Splatting 渲染和 gaussian avatars，增强代理对真实场景的适应能力。**

- **链接: [https://arxiv.org/pdf/2604.12626](https://arxiv.org/pdf/2604.12626)**

> **作者:** Ziyuan Xia; Jingyi Xu; Chong Cui; Yuanhong Yu; Jiazhao Zhang; Qingsong Yan; Tao Ni; Junbo Chen; Xiaowei Zhou; Hujun Bao; Ruizhen Hu; Sida Peng
>
> **备注:** Project page: this https URL
>
> **摘要:** Training embodied AI agents depends critically on the visual fidelity of simulation environments and the ability to model dynamic humans. Current simulators rely on mesh-based rasterization with limited visual realism, and their support for dynamic human avatars, where available, is constrained to mesh representations, hindering agent generalization to human-populated real-world scenarios. We present Habitat-GS, a navigation-centric embodied AI simulator extended from Habitat-Sim that integrates 3D Gaussian Splatting scene rendering and drivable gaussian avatars while maintaining full compatibility with the Habitat ecosystem. Our system implements a 3DGS renderer for real-time photorealistic rendering and supports scalable 3DGS asset import from diverse sources. For dynamic human modeling, we introduce a gaussian avatar module that enables each avatar to simultaneously serve as a photorealistic visual entity and an effective navigation obstacle, allowing agents to learn human-aware behaviors in realistic settings. Experiments on point-goal navigation demonstrate that agents trained on 3DGS scenes achieve stronger cross-domain generalization, with mixed-domain training being the most effective strategy. Evaluations on avatar-aware navigation further confirm that gaussian avatars enable effective human-aware navigation. Finally, performance benchmarks validate the system's scalability across varying scene complexity and avatar counts.
>
---
#### [new 139] Scalable Trajectory Generation for Whole-Body Mobile Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于移动操作任务，解决协调全身运动轨迹生成的问题。提出AutoMoMa框架，实现大规模、物理有效的轨迹数据生成，提升数据规模与多样性。**

- **链接: [https://arxiv.org/pdf/2604.12565](https://arxiv.org/pdf/2604.12565)**

> **作者:** Yida Niu; Xinhai Chang; Xin Liu; Ziyuan Jiao; Yixin Zhu
>
> **摘要:** Robots deployed in unstructured environments must coordinate whole-body motion -- simultaneously moving a mobile base and arm -- to interact with the physical world. This coupled mobility and dexterity yields a state space that grows combinatorially with scene and object diversity, demanding datasets far larger than those sufficient for fixed-base manipulation. Yet existing acquisition methods, including teleoperation and planning, are either labor-intensive or computationally prohibitive at scale. The core bottleneck is the lack of a scalable pipeline for generating large-scale, physically valid, coordinated trajectory data across diverse embodiments and environments. Here we introduce AutoMoMa, a GPU-accelerated framework that unifies AKR modeling, which consolidates base, arm, and object kinematics into a single chain, with parallelized trajectory optimization. AutoMoMa achieves 5,000 episodes per GPU-hour (over $80\times$ faster than CPU-based baselines), producing a dataset of over 500k physically valid trajectories spanning 330 scenes, diverse articulated objects, and multiple robot embodiments. Prior datasets were forced to compromise on scale, diversity, or kinematic fidelity; AutoMoMa addresses all three simultaneously. Training downstream IL policies further reveals that even a single articulated-object task requires tens of thousands of demonstrations for SOTA methods to reach $\approx 80\%$ success, confirming that data scarcity -- not algorithmic limitations -- has been the binding constraint. AutoMoMa thus bridges high-performance planning and reliable IL-based control, providing the infrastructure previously missing for coordinated mobile manipulation research. By making large-scale, kinematically valid training data practical, AutoMoMa showcases generalizable whole-body robot policies capable of operating in the diverse, unstructured settings of the real world.
>
---
#### [new 140] CoLA: A Choice Leakage Attack Framework to Expose Privacy Risks in Subset Training
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于隐私安全任务，研究subset training中的隐私泄露问题。提出CoLA框架，揭示subset选择过程中的隐私风险，解决模型训练中数据选择导致的信息泄露问题。**

- **链接: [https://arxiv.org/pdf/2604.12342](https://arxiv.org/pdf/2604.12342)**

> **作者:** Qi Li; Cheng-Long Wang; Yinzhi Cao; Di Wang
>
> **摘要:** Training models on a carefully chosen portion of data rather than the full dataset is now a standard preprocess for modern ML. From vision coreset selection to large-scale filtering in language models, it enables scalability with minimal utility loss. A common intuition is that training on fewer samples should also reduce privacy risks. In this paper, we challenge this assumption. We show that subset training is not privacy free: the very choices of which data are included or excluded can introduce new privacy surface and leak more sensitive information. Such information can be captured by adversaries either through side-channel metadata from the subset selection process or via the outputs of the target model. To systematically study this phenomenon, we propose CoLA (Choice Leakage Attack), a unified framework for analyzing privacy leakage in subset selection. In CoLA, depending on the adversary's knowledge of the side-channel information, we define two practical attack scenarios: Subset-aware Side-channel Attacks and Black-box Attacks. Under both scenarios, we investigate two privacy surfaces unique to subset training: (1) Training-membership MIA (TM-MIA), which concerns only the privacy of training data membership, and (2) Selection-participation MIA (SP-MIA), which concerns the privacy of all samples that participated in the subset selection process. Notably, SP-MIA enlarges the notion of membership from model training to the entire data-model supply chain. Experiments on vision and language models show that existing threat models underestimate subset-training privacy risks: the expanded privacy surface leaks both training and selection membership, extending risks from individual models to the broader ML ecosystem.
>
---
## 更新

#### [replaced 001] NoisePrints: Distortion-Free Watermarks for Authorship in Private Diffusion Models
- **分类: cs.CV; cs.CR; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.13793](https://arxiv.org/pdf/2510.13793)**

> **作者:** Nir Goren; Oren Katzir; Abhinav Nakarmi; Eyal Ronen; Mahmood Sharif; Or Patashnik
>
> **备注:** code available at: this https URL
>
> **摘要:** With the rapid adoption of diffusion models for visual content generation, proving authorship and protecting copyright have become critical. This challenge is particularly important when model owners keep their models private and may be unwilling or unable to handle authorship issues, making third-party verification essential. A natural solution is to embed watermarks for later verification. However, existing methods require access to model weights and rely on computationally heavy procedures, rendering them impractical and non-scalable. To address these challenges, we propose NoisePrints, a lightweight watermarking scheme that utilizes the random seed used to initialize the diffusion process as a proof of authorship without modifying the generation process. Our key observation is that the initial noise derived from a seed is highly correlated with the generated visual content. By incorporating a hash function into the noise sampling process, we further ensure that recovering a valid seed from the content is infeasible. We also show that sampling an alternative seed that passes verification is infeasible, and demonstrate the robustness of our method under various manipulations. Finally, we show how to use cryptographic zero-knowledge proofs to prove ownership without revealing the seed. By keeping the seed secret, we increase the difficulty of watermark removal. In our experiments, we validate NoisePrints on multiple state-of-the-art diffusion models for images and videos, demonstrating efficient verification using only the seed and output, without requiring access to model weights.
>
---
#### [replaced 002] Mitigating Shortcut Learning via Feature Disentanglement in Medical Imaging: A Benchmark Study
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.18502](https://arxiv.org/pdf/2602.18502)**

> **作者:** Sarah Müller; Philipp Berens
>
> **备注:** Minor edits: formatting improvements and typo fixes; no changes to content or results
>
> **摘要:** Although deep learning models in medical imaging often achieve excellent classification performance, they can rely on shortcut learning, exploiting spurious correlations or confounding factors that are not causally related to the target task. This poses risks in clinical settings, where models must generalize across institutions, populations, and acquisition conditions. Feature disentanglement is a promising approach to mitigate shortcut learning by separating task-relevant information from confounder-related features in latent representations. In this study, we systematically evaluated feature disentanglement methods for mitigating shortcuts in medical imaging, including adversarial learning and latent space splitting based on dependence minimization. We assessed classification performance and disentanglement quality using latent space analyses across one artificial and two medical datasets with natural and synthetic confounders. We also examined robustness under varying levels of confounding and compared computational efficiency across methods. We found that shortcut mitigation methods improved classification performance under strong spurious correlations during training. Latent space analyses revealed differences in representation quality not captured by classification metrics, highlighting the strengths and limitations of each method. Model reliance on shortcuts depended on the degree of confounding in the training data. The best-performing models combine data-centric rebalancing with model-centric disentanglement, achieving stronger and more robust shortcut mitigation than rebalancing alone while maintaining similar computational efficiency. The project code is publicly available at this https URL.
>
---
#### [replaced 003] CoFusion: Multispectral and Hyperspectral Image Fusion via Spectral Coordinate Attention
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.10584](https://arxiv.org/pdf/2604.10584)**

> **作者:** Baisong Li
>
> **摘要:** Multispectral and Hyperspectral Image Fusion (MHIF) aims to reconstruct high-resolution images by integrating low-resolution hyperspectral images (LRHSI) and high-resolution multispectral images (HRMSI). However, existing methods face limitations in modeling cross-scale interactions and spatial-spectral collaboration, making it difficult to achieve an optimal trade-off between spatial detail enhancement and spectral fidelity. To address this challenge, we propose CoFusion: a unified spatial-spectral collaborative fusion framework that explicitly models cross-scale and cross-modal dependencies. Specifically, a Multi-Scale Generator (MSG) is designed to construct a three-level pyramidal architecture, enabling the effective integration of global semantics and local details. Within each scale, a dual-branch strategy is employed: the Spatial Coordinate-Aware Mixing module (SpaCAM) is utilized to capture multi-scale spatial contexts, while the Spectral Coordinate-Aware Mixing module (SpeCAM) enhances spectral representations through frequency decomposition and coordinate mixing. Furthermore, we introduce the Spatial-Spectral Cross-Fusion Module (SSCFM) to perform dynamic cross-modal alignment and complementary feature fusion. Extensive experiments on multiple benchmark datasets demonstrate that CoFusion consistently outperforms state-of-the-art methods, achieving superior performance in both spatial reconstruction and spectral consistency.
>
---
#### [replaced 004] Enhancing Text-to-Image Diffusion Transformer via Split-Text Conditioning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.19261](https://arxiv.org/pdf/2505.19261)**

> **作者:** Yu Zhang; Jialei Zhou; Xinchen Li; Qi Zhang; Zhongwei Wan; Tianyu Wang; Duoqian Miao; Changwei Wang; Longbing Cao
>
> **备注:** NeurIPS 2025
>
> **摘要:** Current text-to-image diffusion generation typically employs complete-text conditioning. Due to the intricate syntax, diffusion transformers (DiTs) inherently suffer from a comprehension defect of complete-text captions. One-fly complete-text input either overlooks critical semantic details or causes semantic confusion by simultaneously modeling diverse semantic primitive types. To mitigate this defect of DiTs, we propose a novel split-text conditioning framework named DiT-ST. This framework converts a complete-text caption into a split-text caption, a collection of simplified sentences, to explicitly express various semantic primitives and their interconnections. The split-text caption is then injected into different denoising stages of DiT-ST in a hierarchical and incremental manner. Specifically, DiT-ST leverages Large Language Models to parse captions, extracting diverse primitives and hierarchically sorting out and constructing these primitives into a split-text input. Moreover, we partition the diffusion denoising process according to its differential sensitivities to diverse semantic primitive types and determine the appropriate timesteps to incrementally inject tokens of diverse semantic primitive types into input tokens via cross-attention. In this way, DiT-ST enhances the representation learning of specific semantic primitive types across different stages. Extensive experiments validate the effectiveness of our proposed DiT-ST in mitigating the complete-text comprehension defect.
>
---
#### [replaced 005] DAV-GSWT: Diffusion-Active-View Sampling for Data-Efficient Gaussian Splatting Wang Tiles
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.15355](https://arxiv.org/pdf/2602.15355)**

> **作者:** Rong Fu; Jiekai Wu; Haiyun Wei; Yee Tan Jia; Yang Li; Xiaowen Ma; Wangyu Wu; Simon Fong
>
> **备注:** 16 pages, 7 figures
>
> **摘要:** The emergence of 3D Gaussian Splatting has fundamentally redefined the capabilities of photorealistic neural rendering by enabling high-throughput synthesis of complex environments. While procedural methods like Wang Tiles have recently been integrated to facilitate the generation of expansive landscapes, these systems typically remain constrained by a reliance on densely sampled exemplar reconstructions. We present DAV-GSWT, a data-efficient framework that leverages diffusion priors and active view sampling to synthesize high-fidelity Gaussian Splatting Wang Tiles from minimal input observations. By integrating a hierarchical uncertainty quantification mechanism with generative diffusion models, our approach autonomously identifies the most informative viewpoints while hallucinating missing structural details to ensure seamless tile transitions. Experimental results indicate that our system significantly reduces the required data volume while maintaining the visual integrity and interactive performance necessary for large-scale virtual environments.
>
---
#### [replaced 006] Point Prompting: Counterfactual Tracking with Video Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.11715](https://arxiv.org/pdf/2510.11715)**

> **作者:** Ayush Shrivastava; Sanyam Mehta; Daniel Geng; Andrew Owens
>
> **备注:** ICLR 2026. Project link: this https URL
>
> **摘要:** Trackers and video generators solve closely related problems: the former analyze motion, while the latter synthesize it. We show that this connection enables pretrained video diffusion models to perform zero-shot point tracking by simply prompting them to visually mark points as they move over time. We place a distinctively colored marker at the query point, then regenerate the rest of the video from an intermediate noise level. This propagates the marker across frames, tracing the point's trajectory. To ensure that the marker remains visible in this counterfactual generation, despite such markers being unlikely in natural videos, we use the unedited initial frame as a negative prompt. Through experiments with multiple image-conditioned video diffusion models, we find that these "emergent" tracks outperform those of prior zero-shot methods and persist through occlusions, often obtaining performance that is competitive with specialized self-supervised models.
>
---
#### [replaced 007] NTIRE 2026 Challenge on Bitstream-Corrupted Video Restoration: Methods and Results
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.06945](https://arxiv.org/pdf/2604.06945)**

> **作者:** Wenbin Zou; Tianyi Liu; Kejun Wu; Huiping Zhuang; Zongwei Wu; Zhuyun Zhou; Radu Timofte; Kim-Hui Yap; Lap-Pui Chau; Yi Wang; Shiqi Zhou; Xiaodi Shi; Yuxiang Chen; Yilian Zhong; Shibo Yin; Yushun Fang; Xilei Zhu; Yahui Wang; Chen Lu; Zhitao Wang; Lifa Ha; Hengyu Man; Xiaopeng Fan; Priyansh Singh; Sidharth; Krrish Dev; Soham Kakkar; Vinit Jakhetiya; Ovais Iqbal Shah; Wei Zhou; Linfeng Li; Qi Xu; Zhenyang Liu; Kepeng Xu; Tong Qiao; Jiachen Tu; Guoyi Xu; Yaoxin Jiang; Jiajia Liu; Yaokun Shi
>
> **备注:** 15 pages, 8 figures, 1 table, CVPRW2026 NTIRE Challenge Report
>
> **摘要:** This paper reports on the NTIRE 2026 Challenge on Bitstream-Corrupted Video Restoration (BSCVR). The challenge aims to advance research on recovering visually coherent videos from corrupted bitstreams, whose decoding often produces severe spatial-temporal artifacts and content distortion. Built upon recent progress in bitstream-corrupted video recovery, the challenge provides a common benchmark for evaluating restoration methods under realistic corruption settings. We describe the dataset, evaluation protocol, and participating methods, and summarize the final results and main technical trends. The challenge highlights the difficulty of this emerging task and provides useful insights for future research on robust video restoration under practical bitstream corruption.
>
---
#### [replaced 008] Variational Autoencoding Discrete Diffusion with Enhanced Dimensional Correlations Modeling
- **分类: cs.LG; cs.CV; stat.ML**

- **链接: [https://arxiv.org/pdf/2505.17384](https://arxiv.org/pdf/2505.17384)**

> **作者:** Tianyu Xie; Shuchen Xue; Zijin Feng; Tianyang Hu; Jiacheng Sun; Zhenguo Li; Cheng Zhang
>
> **备注:** ICLR 2026 Poster; 24 pages, 13 figures
>
> **摘要:** Discrete diffusion models have recently shown great promise for modeling complex discrete data, with masked diffusion models (MDMs) offering a compelling trade-off between quality and generation speed. MDMs denoise by progressively unmasking multiple dimensions from an all-masked input, but their performance can degrade when using few denoising steps due to limited modeling of inter-dimensional dependencies. In this paper, we propose Variational Autoencoding Discrete Diffusion (VADD), a novel framework that enhances discrete diffusion with latent variable modeling to implicitly capture correlations among dimensions. By introducing an auxiliary recognition model, VADD enables stable training via variational lower bounds maximization and amortized inference over the training set. Our approach retains the efficiency of traditional MDMs while significantly improving sample quality, especially when the number of denoising steps is small. Empirical results on 2D toy data, pixel-level image generation, and text generation demonstrate that VADD consistently outperforms MDM baselines in sample quality with few denoising steps.
>
---
#### [replaced 009] IMU: Influence-guided Machine Unlearning
- **分类: cs.LG; cs.CR; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.01620](https://arxiv.org/pdf/2508.01620)**

> **作者:** Xindi Fan; Jing Wu; Mingyi Zhou; Pengwei Liang; Mehrtash Harandi; Dinh Phung
>
> **摘要:** Machine Unlearning (MU) aims to selectively erase the influence of specific data points from pretrained models. However, most existing MU methods rely on the retain set to preserve model utility, which is often impractical due to privacy restrictions and storage constraints. While several retain-data-free methods attempt to bypass this using geometric feature shifts or auxiliary statistics, they typically treat forgetting samples uniformly, overlooking their heterogeneous contributions. To address this, we propose \ul{I}nfluence-guided \ul{M}achine \ul{U}nlearning (IMU), a principled method that conducts MU using only the forget set. Departing from uniform Gradient Ascent (GA) or implicit weighting mechanisms, IMU leverages influence functions as an explicit priority signal to allocate unlearning strength. To circumvent the prohibitive cost of full-model Hessian inversion, we introduce a theoretically grounded classifier-level influence approximation. This efficient design allows IMU to dynamically reweight unlearning updates, aggressively targeting samples that most strongly support the forgetting objective while minimizing unnecessary perturbation to retained knowledge. Extensive experiments across vision and language tasks show that IMU achieves highly competitive results. Compared to standard uniform GA, IMU maintains identical unlearning depth while enhancing model utility by an average of 30%, effectively overcoming the inherent utility-forgetting trade-off.
>
---
#### [replaced 010] BRAIN: Bias-Mitigation Continual Learning Approach to Vision-Brain Understanding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.18187](https://arxiv.org/pdf/2508.18187)**

> **作者:** Xuan-Bac Nguyen; Thanh-Dat Truong; Pawan Sinha; Khoa Luu
>
> **摘要:** Memory decay makes it harder for the human brain to recognize visual objects and retain details. Consequently, recorded brain signals become weaker, uncertain, and contain poor visual context over time. This paper presents one of the first vision-learning approaches to address this problem. First, we statistically and experimentally demonstrate the existence of inconsistency in brain signals and its impact on the Vision-Brain Understanding (VBU) model. Our findings show that brain signal representations shift over recording sessions, leading to compounding bias, which poses challenges for model learning and degrades performance. Then, we propose a new Bias-Mitigation Continual Learning (BRAIN) approach to address these limitations. In this approach, the model is trained in a continual learning setup and mitigates the growing bias from each learning step. A new loss function named De-bias Contrastive Learning is also introduced to address the bias problem. In addition, to prevent catastrophic forgetting, where the model loses knowledge from previous sessions, the new Angular-based Forgetting Mitigation approach is introduced to preserve learned knowledge in the model. Finally, the empirical experiments demonstrate that our approach achieves State-of-the-Art (SOTA) performance across various benchmarks, surpassing prior and non-continual learning methods.
>
---
#### [replaced 011] Habitat Classification from Ground-Level Imagery Using Deep Neural Networks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.04017](https://arxiv.org/pdf/2507.04017)**

> **作者:** Hongrui Shi; Lisa Norton; Lucy Ridding; Simon Rolph; Tom August; Claire M Wood; Lan Qie; Petra Bosilj; James M Brown
>
> **备注:** Accepted to Ecological Informatics. Main paper has 18 pages, 7 figures, 4 tables. Appendix has 10 pages, 8 figures, 2 tables
>
> **摘要:** Habitat assessment at local scales -- critical for enhancing biodiversity and guiding conservation priorities -- often relies on expert field surveys that can be costly, motivating the exploration of AI-driven tools to automate and refine this process. While most AI-driven habitat mapping depends on remote sensing, it is often constrained by sensor availability, weather, and coarse resolution. In contrast, ground-level imagery captures essential structural and compositional cues invisible from above and remains underexplored for robust, fine-grained habitat classification. This study addresses this gap by applying state-of-the-art deep neural network architectures to ground-level habitat imagery. Leveraging data from the UK Countryside Survey covering 18 broad habitat types, we evaluate two families of models - convolutional neural networks (CNNs) and vision transformers (ViTs) - under both supervised and supervised contrastive learning paradigms. Our results demonstrate that ViTs consistently outperform state-of-the-art CNN baselines on key classification metrics (Top-3 accuracy = 91%, MCC = 0.66) and offer more interpretable scene understanding tailored to ground-level images. Moreover, supervised contrastive learning significantly reduces misclassification rates among visually similar habitats (e.g., Improved vs. Neutral Grassland), driven by a more discriminative embedding space. Finally, our best model performs on par with experienced ecological experts in habitat classification from images, underscoring the promise of expert-level automated assessment. By integrating advanced AI with ecological expertise, this research establishes a scalable, cost-effective framework for ground-level habitat monitoring to accelerate biodiversity conservation and inform land-use decisions at a national scale.
>
---
#### [replaced 012] WikiSeeker: Rethinking the Role of Vision-Language Models in Knowledge-Based Visual Question Answering
- **分类: cs.CV; cs.CL; cs.IR**

- **简介: 该论文属于知识驱动的视觉问答任务，旨在解决现有方法过度依赖图像、忽视VLM潜力的问题。提出WikiSeeker框架，重新定义VLM角色，提升检索与回答效果。**

- **链接: [https://arxiv.org/pdf/2604.05818](https://arxiv.org/pdf/2604.05818)**

> **作者:** Yingjian Zhu; Xinming Wang; Kun Ding; Ying Wang; Bin Fan; Shiming Xiang
>
> **备注:** Accepted by ACL 2026 Findings
>
> **摘要:** Multi-modal Retrieval-Augmented Generation (RAG) has emerged as a highly effective paradigm for Knowledge-Based Visual Question Answering (KB-VQA). Despite recent advancements, prevailing methods still primarily depend on images as the retrieval key, and often overlook or misplace the role of Vision-Language Models (VLMs), thereby failing to leverage their potential fully. In this paper, we introduce WikiSeeker, a novel multi-modal RAG framework that bridges these gaps by proposing a multi-modal retriever and redefining the role of VLMs. Rather than serving merely as answer generators, we assign VLMs two specialized agents: a Refiner and an Inspector. The Refiner utilizes the capability of VLMs to rewrite the textual query according to the input image, significantly improving the performance of the multimodal retriever. The Inspector facilitates a decoupled generation strategy by selectively routing reliable retrieved context to another LLM for answer generation, while relying on the VLM's internal knowledge when retrieval is unreliable. Extensive experiments on EVQA, InfoSeek, and M2KR demonstrate that WikiSeeker achieves state-of-the-art performance, with substantial improvements in both retrieval accuracy and answer quality. Our code will be released on this https URL.
>
---
#### [replaced 013] Are Pretrained Image Matchers Good Enough for SAR-Optical Satellite Registration?
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.10217](https://arxiv.org/pdf/2604.10217)**

> **作者:** Isaac Corley; Alex Stoken; Gabriele Berton
>
> **备注:** CVPR 2026 Image Matching Workshop
>
> **摘要:** Cross-modal optical-SAR (Synthetic Aperture Radar) registration is a bottleneck for disaster-response via remote sensing, yet modern image matchers are developed and benchmarked almost exclusively on natural-image domains. We evaluate twenty-four pretrained matcher families--in a zero-shot setting with no fine-tuning or domain adaptation on satellite or SAR data--on SpaceNet9 and two additional cross-modal benchmarks under a deterministic protocol with tiled large-image inference, robust geometric filtering, and tie-point-grounded metrics. Our results reveal asymmetric transfer--matchers with explicit cross-modal training do not uniformly outperform those without it. While XoFTR (trained for visible-thermal matching) and RoMa achieve the lowest reported mean error at $3.0$ px on the labeled SpaceNet9 training scenes, RoMa achieves this without any cross-modal training, and MatchAnything-ELoFTR ($3.4$ px)--trained on synthetic cross-modal pairs--matches closely, suggesting (as a working hypothesis) that foundation-model features (DINOv2) may contribute to modality invariance that partially substitutes for explicit cross-modal supervision. 3D-reconstruction matchers (MASt3R, DUSt3R), which are not designed for traditional 2D image matching, are highly protocol-sensitive and remain fragile under default settings. Deployment protocol choices (geometry model, tile size, inlier gating) shift accuracy by up to $33\times$ for a single matcher, sometimes exceeding the effect of swapping matchers entirely within the evaluated sweep--affine geometry alone reduces mean error from $12.34$ to $9.74$ px. These findings inform both practical deployment of existing matchers and future matcher design for cross-modal satellite registration.
>
---
#### [replaced 014] CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出CropVLM，解决细粒度视觉-语言感知问题，通过动态聚焦图像区域提升模型性能，无需标注或修改原有模型。**

- **链接: [https://arxiv.org/pdf/2511.19820](https://arxiv.org/pdf/2511.19820)**

> **作者:** Miguel Carvalho; Helder Dias; Bruno Martins
>
> **备注:** Accepted to the GRAIL-V Workshop at CVPR 2026
>
> **摘要:** Vision-Language Models (VLMs) often struggle with tasks that require fine-grained image understanding, such as scene-text recognition or document analysis, due to perception limitations and visual fragmentation. To address these challenges, we introduce CropVLM as an external low-cost method for boosting performance, enabling VLMs to dynamically ''zoom in'' on relevant image regions, enhancing their ability to capture fine details. CropVLM is trained using reinforcement learning, without using human-labeled bounding boxes as a supervision signal, and without expensive synthetic evaluations. The model is trained once and can be paired with both open-source and proprietary VLMs to improve their performance. Our approach delivers significant improvements on tasks that require high-resolution image understanding, notably for benchmarks that are out-of-domain for the target VLM, without modifying or fine-tuning the VLM, thus avoiding catastrophic forgetting.
>
---
#### [replaced 015] CamReasoner: Reinforcing Camera Movement Understanding via Structured Spatial Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.00181](https://arxiv.org/pdf/2602.00181)**

> **作者:** Hang Wu; Yujun Cai; Zehao Li; Haonan Ge; Bowen Sun; Junsong Yuan; Yiwei Wang
>
> **摘要:** Understanding camera dynamics is a fundamental pillar of video spatial intelligence. However, existing multimodal models predominantly treat this task as a black-box classification, often confusing physically distinct motions by relying on superficial visual patterns rather than geometric cues. We present \textbf{CamReasoner}, a framework that reformulates camera movement understanding as a structured inference process to bridge the gap between perception and cinematic logic. Our approach centers on the Observation-Thinking-Answer (O-T-A) paradigm, which compels the model to articulate spatio-temporal observations and reason about motion patterns within an explicit reasoning block. To instill this capability, we construct a Large-scale Inference Trajectory Suite comprising 18k SFT reasoning chains and 38k RL feedback samples. To the best of our knowledge, \textbf{we are the first to employ RL for logical alignment in camera movement understanding}, ensuring motion inferences are grounded in structured visual reasoning rather than contextual guesswork. Built upon Qwen2.5-VL-7B, CamReasoner-7B improves binary classification accuracy from 73.8\% to 78.4\% and VQA accuracy from 60.9\% to 74.5\% over its backbone, consistently outperforming both proprietary and open-source baselines across multiple benchmarks.
>
---
#### [replaced 016] ASTRA: Let Arbitrary Subjects Transform in Video Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.01186](https://arxiv.org/pdf/2510.01186)**

> **作者:** Fei Shen; Weihao Xu; Rui Yan; Dong Zhang; Xiangbo Shu; Jinhui Tang; Maocheng Zhao
>
> **摘要:** While existing video editing methods excel with single subjects, they struggle in dense, multi-subject scenes, frequently suffering from attention dilution and mask boundary entanglement that cause attribute leakage and temporal instability. To address this, we propose ASTRA, a training-free framework for seamless, arbitrary-subject video editing. Without requiring model fine-tuning, ASTRA precisely manipulates multiple designated subjects while strictly preserving non-target regions. It achieves this via two core components: a prompt-guided multimodal alignment module that generates robust conditions to mitigate attention dilution, and a prior-based mask retargeting module that produces temporally coherent mask sequences to resolve boundary entanglement. Functioning as a versatile plug-and-play module, ASTRA seamlessly integrates with diverse mask-driven video generators. Extensive experiments on our newly constructed benchmark, MSVBench, demonstrate that ASTRA consistently outperforms state-of-the-art methods. Code, models, and data are available at this https URL.
>
---
#### [replaced 017] ArtiCAD: Articulated CAD Assembly Design via Multi-Agent Code Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.10992](https://arxiv.org/pdf/2604.10992)**

> **作者:** Yuan Shui; Yandong Guan; Zhanwei Zhang; Juncheng Hu; Jing Zhang; Dong Xu; Qian Yu
>
> **摘要:** Parametric Computer-Aided Design (CAD) of articulated assemblies is essential for product development, yet generating these multi-part, movable models from high-level descriptions remains unexplored. To address this, we propose ArtiCAD, the first training-free multi-agent system capable of generating editable, articulated CAD assemblies directly from text or images. Our system divides this complex task among four specialized agents: Design, Generation, Assembly, and Review. One of our key insights is to predict assembly relationships during the initial design stage rather than the assembly stage. By utilizing a Connector that explicitly defines attachment points and joint parameters, ArtiCAD determines these relationships before geometry generation, effectively bypassing the limited spatial reasoning capabilities of current LLMs and VLMs. To further ensure high-quality outputs, we introduce validation steps in the generation and assembly stages, accompanied by a cross-stage rollback mechanism that accurately isolates and corrects design- and code-level errors. Additionally, a self-evolving experience store accumulates design knowledge to continuously improve performance on future tasks. Extensive evaluations on three datasets (ArtiCAD-Bench, CADPrompt, and ACD) validate the effectiveness of our approach. We further demonstrate the applicability of ArtiCAD in requirement-driven conceptual design, physical prototyping, and the generation of embodied AI training assets through URDF export.
>
---
#### [replaced 018] Toward Efficient and Robust Behavior Models for Multi-Agent Driving Simulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于多智能体驾驶仿真任务，旨在提升行为模型的效率与鲁棒性。通过优化场景表示和交互建模，实现更高效的训练与推理。**

- **链接: [https://arxiv.org/pdf/2512.05812](https://arxiv.org/pdf/2512.05812)**

> **作者:** Fabian Konstantinidis; Moritz Sackmann; Ulrich Hofmann; Christoph Stiller
>
> **备注:** This is the author's accepted version of a paper to appear in the IEEE International Conference on Robotics & Automation (ICRA 2026)
>
> **摘要:** Scalable multi-agent driving simulation requires behavior models that are both realistic and computationally efficient. We address this by optimizing the behavior model that controls individual traffic participants. To improve efficiency, we adopt an instance-centric scene representation, where each traffic participant and map element is modeled in its own local coordinate frame. This design enables efficient, viewpoint-invariant scene encoding and allows static map tokens to be reused across simulation steps. To model interactions, we employ a query-centric symmetric context encoder with relative positional encodings between local frames. We use Adversarial Inverse Reinforcement Learning to learn the behavior model and propose an adaptive reward transformation that automatically balances robustness and realism during training. Experiments demonstrate that our approach scales efficiently with the number of tokens, significantly reducing training and inference times, while outperforming several agent-centric baselines in terms of positional accuracy and robustness.
>
---
#### [replaced 019] Ambivalence/Hesitancy Recognition in Videos for Personalized Digital Health Interventions
- **分类: cs.CV; cs.HC; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.11730](https://arxiv.org/pdf/2604.11730)**

> **作者:** Manuela González-González; Soufiane Belharbi; Muhammad Osama Zeeshan; Masoumeh Sharafi; Muhammad Haseeb Aslam; Lorenzo Sia; Nicolas Richet; Marco Pedersoli; Alessandro Lameiras Koerich; Simon L Bacon; Eric Granger
>
> **备注:** 13 pages, 3 figures. arXiv admin note: substantial text overlap with arXiv:2505.19328
>
> **摘要:** Using behavioural science, health interventions focus on behaviour change by providing a framework to help patients acquire and maintain healthy habits that improve medical outcomes. In-person interventions are costly and difficult to scale, especially in resource-limited regions. Digital health interventions offer a cost-effective approach, potentially supporting independent living and self-management. Automating such interventions, especially through machine learning, has gained considerable attention recently. Ambivalence and hesitancy (A/H) play a primary role for individuals to delay, avoid, or abandon health interventions. A/H are subtle and conflicting emotions that place a person in a state between positive and negative evaluations of a behaviour, or between acceptance and refusal to engage in it. They manifest as affective inconsistency across modalities or within a modality, such as language, facial, vocal expressions, and body language. While experts can be trained to recognize A/H, integrating them into digital health interventions is costly and less effective. Automatic A/H recognition is therefore critical for the personalization and cost-effectiveness of digital health interventions. Here, we explore the application of deep learning models for A/H recognition in videos, a multi-modal task by nature. In particular, this paper covers three learning setups: supervised learning, unsupervised domain adaptation for personalization, and zero-shot inference via large language models (LLMs). Our experiments are conducted on the unique and recently published BAH video dataset for A/H recognition. Our results show limited performance, suggesting that more adapted multi-modal models are required for accurate A/H recognition. Better methods for modeling spatio-temporal and multimodal fusion are necessary to leverage conflicts within/across modalities.
>
---
#### [replaced 020] LoViF 2026 The First Challenge on Weather Removal in Videos
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [https://arxiv.org/pdf/2604.10655](https://arxiv.org/pdf/2604.10655)**

> **作者:** Chenghao Qian; Xin Li; Yeying Jin; Shangguan Sun; Yilian Zhong; Yuxiang Chen; Shibo Yin; Yushun Fang; Xilei Zhu; Yahui Wang; Chen Lu; Ying Fu; Jianan Tian; Jifan Zhang; Chen Zhou; Junyang Jiang; Yuping Sun; Zhuohang Shi; Xiaojing Liu; Jiao Liu; Yatong Zhou; Shuai Liu; Qiang Deng; Jiajia Mi; Qianhao Luo; Weiling Li
>
> **备注:** CVPR Workshop Challenge Report
>
> **摘要:** This paper presents a review of the LoViF 2026 Challenge on Weather Removal in Videos. The challenge encourages the development of methods for restoring clean videos from inputs degraded by adverse weather conditions such as rain and snow, with an emphasis on achieving visually plausible and temporally consistent results while preserving scene structure and motion dynamics. To support this task, we introduce a new short-form WRV dataset tailored for video weather removal. It consists of 18 videos 1,216 synthesized frames paired with 1,216 real-world ground-truth frames at a resolution of 832 x 480, and is split into training, validation, and test sets with a ratio of 1:1:1. The goal of this challenge is to advance robust and realistic video restoration under real-world weather conditions, with evaluation protocols that jointly consider fidelity and perceptual quality. The challenge attracted 37 participants and received 5 valid final submissions with corresponding fact sheets, contributing to progress in weather removal for videos. The project is publicly available at this https URL.
>
---
#### [replaced 021] Bridging Time and Space: Decoupled Spatio-Temporal Alignment for Video Grounding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.08014](https://arxiv.org/pdf/2604.08014)**

> **作者:** Xuezhen Tu; Jingyu Wu; Fangyu Kang; Qingpeng Nong; Kaijin Zhang; Chaoyue Niu; Fan Wu
>
> **备注:** Some numbers errors
>
> **摘要:** Spatio-Temporal Video Grounding requires jointly localizing target objects across both temporal and spatial dimensions based on natural language queries, posing fundamental challenges for existing Multimodal Large Language Models (MLLMs). We identify two core challenges: \textit{entangled spatio-temporal alignment}, arising from coupling two heterogeneous sub-tasks within the same autoregressive output space, and \textit{dual-domain visual token redundancy}, where target objects exhibit simultaneous temporal and spatial sparsity, rendering the overwhelming majority of visual tokens irrelevant to the grounding query. To address these, we propose \textbf{Bridge-STG}, an end-to-end framework that decouples temporal and spatial localization while maintaining semantic coherence. While decoupling is the natural solution to this entanglement, it risks creating a semantic gap between the temporal MLLM and the spatial decoder. Bridge-STG resolves this through two pivotal designs: the \textbf{Spatio-Temporal Semantic Bridging (STSB)} mechanism with Explicit Temporal Alignment (ETA) distills the MLLM's temporal reasoning context into enriched bridging queries as a robust semantic interface; and the \textbf{Query-Guided Spatial Localization (QGSL)} module leverages these queries to drive a purpose-built spatial decoder with multi-layer interactive queries and positive/negative frame sampling, jointly eliminating dual-domain visual token redundancy. Extensive experiments across multiple benchmarks demonstrate that Bridge-STG achieves state-of-the-art performance among MLLM-based methods. Bridge-STG improves average m\_vIoU from $26.4$ to $34.3$ on VidSTG and demonstrates strong cross-task transfer across various fine-grained video understanding tasks under a unified multi-task training regime.
>
---
#### [replaced 022] Edu-MMBias: A Three-Tier Multimodal Benchmark for Auditing Social Bias in Vision-Language Models under Educational Contexts
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2604.10200](https://arxiv.org/pdf/2604.10200)**

> **作者:** Ruijia Li; Mingzi Zhang; Zengyi Yu; Yuang Wei; Bo Jiang
>
> **摘要:** As Vision-Language Models (VLMs) become integral to educational decision-making, ensuring their fairness is paramount. However, current text-centric evaluations neglect the visual modality, leaving an unregulated channel for latent social biases. To bridge this gap, we present Edu-MMBias, a systematic auditing framework grounded in the tri-component model of attitudes from social psychology. This framework diagnoses bias across three hierarchical dimensions: cognitive, affective, and behavioral. Utilizing a specialized generative pipeline that incorporates a self-correct mechanism and human-in-the-loop verification, we synthesize contamination-resistant student profiles to conduct a holistic stress test on state-of-the-art VLMs. Our extensive audit reveals critical, counter-intuitive patterns: models exhibit a compensatory class bias favoring lower-status narratives while simultaneously harboring deep-seated health and racial stereotypes. Crucially, we find that visual inputs act as a safety backdoor, triggering a resurgence of biases that bypass text-based alignment safeguards and revealing a systematic misalignment between latent cognition and final decision-making. The contributions of this paper are available at: this https URL.
>
---
#### [replaced 023] Causal Fingerprints of AI Generative Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.15406](https://arxiv.org/pdf/2509.15406)**

> **作者:** Hui Xu; Chi Liu; Congcong Zhu; Minghao Wang; Youyang Qu; Longxiang Gao
>
> **备注:** 5 page, accepted for presentation at IEEE ICASSP 2026
>
> **摘要:** AI generative models leave implicit traces in their generated images, which are commonly referred to as model fingerprints and are exploited for source attribution. Prior methods rely on model-specific cues or synthesis artifacts, yielding limited fingerprints that may generalize poorly across different generative models. We argue that a complete model fingerprint should reflect the causality between image provenance and model traces, a direction largely unexplored. To this end, we conceptualize the causal fingerprint of generative models, and propose a causality-decoupling framework that disentangles it from image-specific content and style in a semantic-invariant latent space derived from pre-trained diffusion reconstruction residual. We further enhance fingerprint granularity with diverse feature representations. We validate causality by assessing attribution performance across representative GANs and diffusion models and by achieving source anonymization using counterfactual examples generated from causal fingerprints. Experiments show our approach outperforms existing methods in model attribution, indicating strong potential for forgery detection, model copyright tracing, and identity protection.
>
---
#### [replaced 024] ReXSonoVQA: A Video QA Benchmark for Procedure-Centric Ultrasound Understanding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.10916](https://arxiv.org/pdf/2604.10916)**

> **作者:** Xucheng Wang; Xiaoman Zhang; Sung Eun Kim; Ankit Pal; Pranav Rajpurkar
>
> **摘要:** Ultrasound acquisition requires skilled probe manipulation and real-time adjustments. Vision-language models (VLMs) could enable autonomous ultrasound systems, but existing benchmarks evaluate only static images, not dynamic procedural understanding. We introduce ReXSonoVQA, a video QA benchmark with 514 video clips and 514 questions (249 MCQ, 265 free-response) targeting three competencies: Action-Goal Reasoning, Artifact Resolution & Optimization, and Procedure Context & Planning. Zero-shot evaluation of Gemini 3 Pro, Qwen3.5-397B, LLaVA-Video-72B, and Seed 2.0 Pro shows VLMs can extract some procedural information, but troubleshooting questions remain challenging with minimal gains over text-only baselines, exposing limitations in causal reasoning. ReXSonoVQA enables developing perception systems for ultrasound training, guidance, and robotic automation.
>
---
#### [replaced 025] HSG-12M: A Large-Scale Benchmark of Spatial Multigraphs from the Energy Spectra of Non-Hermitian Crystals
- **分类: cs.LG; cond-mat.mes-hall; cond-mat.other; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.08618](https://arxiv.org/pdf/2506.08618)**

> **作者:** Xianquan Yan; Hakan Akgün; Kenji Kawaguchi; N. Duane Loh; Ching Hua Lee
>
> **备注:** Accepted to ICLR 2026, OpenReview: [this https URL]. 49 pages, 13 figures, 14 tables. Code & pipeline: [this https URL]. Dataset: [this https URL]. Dataset released under CC BY 4.0. Benchmark scripts and data loaders included. The Fourteenth International Conference on Learning Representations (ICLR 2026)
>
> **摘要:** AI is transforming scientific research by revealing new ways to understand complex physical systems, but its impact remains constrained by the lack of large, high-quality domain-specific datasets. A rich, largely untapped resource lies in non-Hermitian quantum physics, where the energy spectra of crystals form intricate geometries on the complex plane -- termed as Hamiltonian spectral graphs. Despite their significance as fingerprints for electronic behavior, their systematic study has been intractable due to the reliance on manual extraction. To unlock this potential, we introduce Poly2Graph: a high-performance, open-source pipeline that automates the mapping of 1-D crystal Hamiltonians to spectral graphs. Using this tool, we present HSG-12M: a dataset containing 11.6 million static and 5.1 million dynamic Hamiltonian spectral graphs across 1401 characteristic-polynomial classes, distilled from 177 TB of spectral potential data. Crucially, HSG-12M is the first large-scale dataset of spatial multigraphs -- graphs embedded in a metric space where multiple geometrically distinct trajectories between two nodes are retained as separate edges. This simultaneously addresses a critical gap, as existing graph benchmarks overwhelmingly assume simple, non-spatial edges, discarding vital geometric information. Benchmarks with popular GNNs expose new challenges in learning spatial multi-edges at scale. Beyond its practical utility, we show that spectral graphs serve as universal topological fingerprints of polynomials, vectors, and matrices, forging a new algebra-to-graph link. HSG-12M lays the groundwork for data-driven scientific discovery in condensed matter physics, new opportunities in geometry-aware graph learning and beyond.
>
---
#### [replaced 026] Are Video Reasoning Models Ready to Go Outside?
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.10652](https://arxiv.org/pdf/2603.10652)**

> **作者:** Yangfan He; Changgyu Boo; Jaehong Yoon
>
> **备注:** Project Page: this https URL
>
> **摘要:** In real-world deployment, vision-language models often encounter disturbances such as weather, occlusion, and camera motion. Under such conditions, their understanding and reasoning degrade substantially, revealing a gap between clean, controlled (i.e., unperturbed) evaluation settings and real-world robustness. To address this limitation, we propose ROVA, a novel training framework that improves robustness by modeling a robustness-aware consistency reward under spatio-temporal corruptions. ROVA introduces a difficulty-aware online training strategy that prioritizes informative samples based on the model's evolving capability. Specifically, it continuously re-estimates sample difficulty via self-reflective evaluation, enabling adaptive training with a robustness-aware consistency reward. We also introduce PVRBench, a new benchmark that injects real-world perturbations into embodied video datasets to assess both accuracy and reasoning quality under realistic disturbances. We evaluate ROVA and baselines on PVRBench, UrbanVideo, and VisBench, where open-source and proprietary models suffer up to 35% and 28% drops in accuracy and reasoning under realistic perturbations. ROVA effectively mitigates performance degradation, boosting relative accuracy by at least 24% and reasoning by over 9% compared with baseline models (QWen2.5/3-VL, InternVL2.5, Embodied-R). These gains transfer to clean standard benchmarks, yielding consistent improvements.
>
---
#### [replaced 027] SinkSAM-Net: Knowledge-Driven Self-Supervised Sinkhole Segmentation Using Topographic Priors and Segment Anything Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2410.01473](https://arxiv.org/pdf/2410.01473)**

> **作者:** Osher Rafaeli; Tal Svoray; Ariel Nahlieli
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** Soil sinkholes significantly influence soil degradation, infrastructure vulnerability, and landscape evolution. However, their irregular shapes, combined with interference from shadows and vegetation, make it challenging to accurately quantify their properties using remotely sensed data. In addition, manual annotation can be laborious and costly. In this study, we introduce a novel self-supervised framework for sinkhole segmentation, termed SinkSAM-Net, which integrates traditional topographic computations of closed depressions with an iterative, geometry-aware, prompt-based Segment Anything Model (SAM). We generate high-quality pseudo-labels through pixel-level refinement of sinkhole boundaries by integrating monocular depth information with random prompts augmentation technique named coordinate-wise bounding box jittering (CWBJ). These pseudo-labels iteratively enhance a lightweight EfficientNetV2-UNet target model, ultimately transferring knowledge to a prompt-free, low-parameter, and fast inference model. Our proposed approach achieves approximately 95\% of the performance obtained through manual supervision by human annotators. The framework's performance was evaluated on a large sinkhole database, covering diverse sinkhole dateset-induced sinkholes using both aerial and high-resolution drone imagery. This paper presents the first self-supervised framework for sinkhole segmentation, demonstrating the robustness of foundational models (such as SAM and Depth Anything V2) when combined with prior topographic and geometry knowledge and an iterative self-learning pipeline. SinkSAM-Net has the potential to be trained effectively on extensive unlabeled RGB sinkholes datasets, achieving comparable performance to a supervised model. The code and interactive demo for SinkSAM-Net are available on the project page \href{this https URL}{at this URL}.
>
---
#### [replaced 028] LPNSR: Optimal Noise-Guided Diffusion Image Super-Resolution Via Learnable Noise Prediction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.21045](https://arxiv.org/pdf/2603.21045)**

> **作者:** Shuwei Huang; Shizhuo Liu; Zijun Wei
>
> **摘要:** Diffusion-based image super-resolution (SR) aims to reconstruct high-resolution (HR) images from low-resolution (LR) observations. However, the inherent randomness injected during the reverse diffusion process causes the performance of diffusion-based SR models to vary significantly across different sampling runs, particularly when the sampling trajectory is compressed into a limited number of steps. A critical yet underexplored question is: what is the optimal noise to inject at each intermediate diffusion step? In this paper, we establish a theoretical framework that derives the closed-form analytical solution for optimal intermediate noise in diffusion models from a maximum likelihood estimation perspective, revealing a consistent conditional dependence structure that generalizes across diffusion paradigms. We instantiate this framework under the residual-shifting diffusion paradigm and accordingly design an LR-guided multi-input-aware noise predictor to replace random Gaussian noise. We further mitigate initialization bias with a high-quality pre-upsampling network. The compact 4-step trajectory uniquely enables end-to-end optimization of the entire reverse chain, which is computationally prohibitive for conventional long-trajectory diffusion models. Extensive experiments demonstrate that LPNSR achieves state-of-the-art perceptual performance on both synthetic and real-world datasets, without relying on any large-scale text-to-image priors. The source code of our method can be found at this https URL.
>
---
#### [replaced 029] LOLGORITHM: Funny Comment Generation Agent For Short Videos
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.09729](https://arxiv.org/pdf/2604.09729)**

> **作者:** Xuan Ouyang; Bouzhou Wang; Senan Wang; Siyuan Xiahou; Jinrong Zhou; Yuekang Li
>
> **摘要:** Short-form video platforms have become central to multimedia information dissemination, where comments play a critical role in driving engagement, propagation, and algorithmic feedback. However, existing approaches -- including video summarization and live-streaming danmaku generation -- fail to produce authentic comments that conform to platform-specific cultural and linguistic norms. In this paper, we present LOLGORITHM, a novel modular multi-agent framework for stylized short-form video comment generation. LOLGORITHM supports six controllable comment styles and comprises three core modules: video content summarization, video classification, and comment generation with semantic retrieval and hot meme augmentation. We further construct a bilingual dataset of 3,267 videos and 16,335 comments spanning five high-engagement categories across YouTube and Douyin. Evaluation combining automatic scoring and large-scale human preference analysis demonstrates that LOLGORITHM consistently outperforms baseline methods, achieving human preference selection rates of 80.46\% on YouTube and 84.29\% on Douyin across 107 respondents. Ablation studies confirm that these gains are attributable to the framework architecture rather than the choice of backbone LLM, underscoring the robustness and generalizability of our approach.
>
---
#### [replaced 030] Deep Learning using Rectified Linear Units (ReLU)
- **分类: cs.NE; cs.CV; cs.LG; stat.ML**

- **链接: [https://arxiv.org/pdf/1803.08375](https://arxiv.org/pdf/1803.08375)**

> **作者:** Abien Fred Agarap
>
> **备注:** 9 pages, 5 figures, 5 tables
>
> **摘要:** The Rectified Linear Unit (ReLU) is a foundational activation function in artficial neural networks. Recent literature frequently misattributes its origin to the 2018 (initial) version of this paper, which exclusively investigated ReLU at the classification layer. This paper formally corrects the citation record by tracing the mathematical lineage of piecewise linear functions from early biological models to their definitive integration into deep learning by Nair & Hinton (2010). Alongside this historical rectification, we present a comprehensive empirical comparison of the ReLU, Hyperbolic Tangent (Tanh), and Logistic (Sigmoid) activation functions across image classification, text classification, and image reconstruction tasks. To ensure statistical robustness, we evaluated these functions using 10 independent randomized trials and assessed significance using the non-parametric Kruskal-Wallis $H$ test. The empirical data validates the theoretical limitations of saturating functions. Sigmoid failed to converge in deep convolutional vision tasks due to the vanishing gradient problem, thus yielding accuracies equivalent to random probability. Conversely, ReLU and Tanh exhibited stable convergence. ReLU achieved the highest mean accuracy and F1-score on image classification and text classification tasks, while Tanh yielded the highest peak signal to noise ratio in image reconstruction. Ultimately, this study confirms a statistically significant performance variance among activations, thus reaffirming the necessity of non-saturating functions in deep architectures, and restores proper historical attribution to prior literature.
>
---
#### [replaced 031] ABot-M0: VLA Foundation Model for Robotic Manipulation with Action Manifold Learning
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决多硬件通用智能体构建难题。通过数据标准化与动作流形学习，提升动作预测效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.11236](https://arxiv.org/pdf/2602.11236)**

> **作者:** Yandan Yang; Shuang Zeng; Tong Lin; Xinyuan Chang; Dekang Qi; Junjin Xiao; Haoyun Liu; Ronghan Chen; Yuzhi Chen; Dongjie Huo; Feng Xiong; Xing Wei; Zhiheng Ma; Mu Xu
>
> **备注:** Project website: this https URL . Code: this https URL . 22 pages, 10 figures, 10 tables
>
> **摘要:** Building general-purpose embodied agents across diverse hardware remains a central challenge in robotics, often framed as the ''one-brain, many-forms'' paradigm. Progress is hindered by fragmented data, inconsistent representations, and misaligned training objectives. We present ABot-M0, a framework that builds a systematic data curation pipeline while jointly optimizing model architecture and training strategies, enabling end-to-end transformation of heterogeneous raw data into unified, efficient representations. From six public datasets, we clean, standardize, and balance samples to construct UniACT-dataset, a large-scale dataset with over 6 million trajectories and 9,500 hours of data, covering diverse robot morphologies and task scenarios. Unified pre-training improves knowledge transfer and generalization across platforms and tasks, supporting general-purpose embodied intelligence. To improve action prediction efficiency and stability, we propose the Action Manifold Hypothesis: effective robot actions lie not in the full high-dimensional space but on a low-dimensional, smooth manifold governed by physical laws and task constraints. Based on this, we introduce Action Manifold Learning (AML), which uses a DiT backbone to predict clean, continuous action sequences directly. This shifts learning from denoising to projection onto feasible manifolds, improving decoding speed and policy stability. ABot-M0 supports modular perception via a dual-stream mechanism that integrates VLM semantics with geometric priors and multi-view inputs from plug-and-play 3D modules such as VGGT and Qwen-Image-Edit, enhancing spatial understanding without modifying the backbone and mitigating standard VLM limitations in 3D reasoning. Experiments show components operate independently with additive benefits. We will release all code and pipelines for reproducibility and future research.
>
---
#### [replaced 032] EDGE-Shield: Efficient Denoising-staGE Shield for Violative Content Filtering via Scalable Reference-Based Matching
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2604.06063](https://arxiv.org/pdf/2604.06063)**

> **作者:** Takara Taniguchi; Ryohei Shimizu; Duc Minh Vo; Kota Izumi; Shiqi Yang; Teppei Suzuki
>
> **摘要:** The advent of Text-to-Image generative models poses significant risks of copyright violation and deepfake generation. Since the rapid proliferation of new copyrighted works and private individuals constantly emerges, reference-based training-free content filters are essential for providing up-to-date protection without the constraints of a fixed knowledge cutoff. However, existing reference-based approaches often lack scalability when handling numerous references and require waiting for finishing image generation. To solve these problems, we propose EDGE-Shield, a scalable content filter during the denoising process that maintains practical latency while effectively blocking violative content. We leverage embedding-based matching for efficient reference comparison. Additionally, we introduce an \textit{$x$}-pred transformation that converts the model's noisy intermediate latent into the pseudo-estimated clean latent at the later stage, enhancing classification accuracy of violative content at earlier denoising stages. We conduct experiments of violative content filtering against two generative models including Z-Image-Turbo and Qwen-Image. EDGE-Shield significantly outperforms traditional reference-based methods in terms of latency; it achieves an approximate $79\%$ reduction in processing time for Z-Image-Turbo and approximate $50\%$ reduction for Qwen-Image, maintaining the filtering accuracy across different model architectures.
>
---
#### [replaced 033] Time-reversed Flow Matching with Worst Transport in High-dimensional Latent Space for Image Anomaly Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.05461](https://arxiv.org/pdf/2508.05461)**

> **作者:** Liangwei Li; Lin Liu; Hanzhe Liang; Juanxiu Liu; Jing Zhang; Ruqian Hao; Xiaohui Du; Yong Liu; Pan Li
>
> **摘要:** Likelihood-based deep generative models have been widely investigated for Image Anomaly Detection (IAD), particularly Normalizing Flows, yet their strict architectural invertibility needs often constrain scalability, particularly in large-scale data regimes. Although time-parameterized Flow Matching (FM) serves as a scalable alternative, it remains computationally challenging in IAD due to the prohibitive costs of Jacobian-trace estimation. This paper proposes time-reversed Flow Matching (rFM), which shifts the objective from exact likelihood computation to evaluating target-domain regularity through density proxy estimation. We uncover two fundamental theoretical bottlenecks in this paradigm: first, the reversed vector field exhibits a non-Lipschitz singularity at the initial temporal boundary, precipitating explosive estimation errors. Second, the concentration of measure in high-dimensional Gaussian manifolds induces structured irregularities, giving rise to a Centripetal Potential Field (CPF) that steers trajectories away from Optimal Transport (OT) paths. We identify these observations as the inherent dualities between FM and rFM. To address these issues, we introduce local Worst Transport Flow matching (WT-Flow), which amplifies the observed CPF of rFM to mitigate the initial singularity while circumventing the need for exact distribution transformations via density proxy. Experiments on five datasets demonstrate that WT-Flow achieves state-of-the-art performance among single-scale flow-based methods, and competitive performance against leading multi-scale approaches. Furthermore, the proposed framework enables superior one-step inference, achieving a per-image flow latency of only 6.7 ms. Our code is available on this https URL.
>
---
#### [replaced 034] CREG: Compass Relational Evidence Graph for Characterizing Directional Structure in VLM Spatial-Reasoning Attribution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.20475](https://arxiv.org/pdf/2603.20475)**

> **作者:** Kaizhen Tan; Yang Feng; Heqing Du
>
> **摘要:** Standard attribution heatmaps show where a vision-language model (VLM) focuses, but they do not reveal whether the recovered evidence is organized by the queried spatial relation or merely reflects image layout. To address this problem, we introduce CREG (Compass Relational Evidence Graph), a training-free diagnostic framework that converts token-level attribution into a reference-centered compass distribution and measures its directional alignment. CREG provides a shared directional readout across attribution methods and makes comparison with geometric controls explicit. Across three spatial-relation benchmarks, box-only geometry achieves Direction Alignment Error more than 30 degrees lower than current model-based attribution methods, leaving a substantial gap between attribution structure and simple target localization. To examine this gap, we apply a diagnostic battery including target intervention, reference-center randomization, and variance partition. Taken together, the results suggest that the directional structure recoverable from current attribution methods is limited and often mixed with image layout. We further find that higher task accuracy does not reliably coincide with better directional attribution: small-scale LoRA training and newer model generations can improve task accuracy while leaving Direction Alignment Error unchanged or worse. These findings characterize what current attribution methods reveal rather than the model's internal spatial representation. CREG provides a controlled protocol for testing whether improvements in spatial reasoning are accompanied by more directionally organized evidence.
>
---
#### [replaced 035] OmniHands: Towards Robust 4D Hand Mesh Recovery via A Versatile Transformer
- **分类: cs.CV; cs.AI; cs.GR**

- **链接: [https://arxiv.org/pdf/2405.20330](https://arxiv.org/pdf/2405.20330)**

> **作者:** Dixuan Lin; Yuxiang Zhang; Mengcheng Li; Wei Jing; Qi Yan; Qianying Wang; Yebin Liu; Hongwen Zhang
>
> **备注:** An extended journal version of 4DHands, featured with versatile module that can adapt to temporal task and multi-view task. Additional detailed comparison experiments and results presentation have been added. More demo videos can be seen at our project page: this https URL
>
> **摘要:** In this paper, we introduce OmniHands, a universal approach to recovering interactive hand meshes and their relative movement from monocular or multi-view inputs. Our approach addresses two major limitations of previous methods: lacking a unified solution for handling various hand image inputs and neglecting the positional relationship of two hands within images. To overcome these challenges, we develop a universal architecture with novel tokenization and contextual feature fusion strategies, capable of adapting to a variety of tasks. Specifically, we propose a Relation-aware Two-Hand Tokenization (RAT) method to embed positional relation information into the hand tokens. In this way, our network can handle both single-hand and two-hand inputs and explicitly leverage relative hand positions, facilitating the reconstruction of intricate hand interactions in real-world scenarios. As such tokenization indicates the relative relationship of two hands, it also supports more effective feature fusion. To this end, we further develop a 4D Interaction Reasoning (FIR) module to fuse hand tokens in 4D with attention and decode them into 3D hand meshes and relative temporal movements. The efficacy of our approach is validated on several benchmark datasets. The results on in-the-wild videos and real-world scenarios demonstrate the superior performances of our approach for interactive hand reconstruction. More video results can be found on the project page: this https URL.
>
---
#### [replaced 036] LaV-CoT: Language-Aware Visual CoT with Multi-Aspect Reward Optimization for Real-World Multilingual VQA
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.10026](https://arxiv.org/pdf/2509.10026)**

> **作者:** Jing Huang; Zhiya Tan; Shutao Gong; Fanwei Zeng; Joey Tianyi Zhou; Changtao Miao; Huazhe Tan; Weibin Yao; Jianshu Li
>
> **备注:** Accepted by WWW 2026 Industry Track - Oral
>
> **摘要:** As large vision language models (VLMs) advance, their capabilities in multilingual visual question answering (mVQA) have significantly improved. Chain-of-thought (CoT) reasoning has been proven to enhance interpretability and complex reasoning. However, most existing approaches rely primarily on textual CoT and provide limited support for multilingual multimodal reasoning, constraining their deployment in real-world applications. To address this gap, we introduce LaV-CoT, the first Language-aware Visual CoT framework with Multi-Aspect Reward Optimization. LaV-CoT incorporates an interpretable multi-stage reasoning pipeline consisting of Text Summary with Bounding Box (BBox), Language Identification, Spatial Object-level Captioning, and Step-by-step Logical Reasoning. Following this reasoning pipeline, we design an automated data curation method that generates multilingual CoT annotations through iterative generation, correction, and refinement, enabling scalable and high-quality training data. To improve reasoning and generalization, LaV-CoT adopts a two-stage training paradigm combining Supervised Fine-Tuning (SFT) with Language-aware Group Relative Policy Optimization (GRPO), guided by verifiable multi-aspect rewards including language consistency, structural accuracy, and semantic alignment. Extensive evaluations on public datasets including MMMB, Multilingual MMBench, and MTVQA show that LaV-CoT achieves up to ~9.5% accuracy improvements over open-source baselines of similar size and even surpasses models with 2$\times$ larger scales by ~2.6%. Moreover, LaV-CoT outperforms advanced proprietary models such as GPT-4o-0513 and Gemini-2.5-flash. We further conducted an online A/B test to validate our method on real-world data, highlighting its effectiveness for industrial deployment. Our code is available at this link: this https URL
>
---
#### [replaced 037] Pictorial and apictorial polygonal jigsaw puzzles from arbitrary number of crossing cuts
- **分类: cs.CV; cs.AI; cs.CG**

- **链接: [https://arxiv.org/pdf/2008.07644](https://arxiv.org/pdf/2008.07644)**

> **作者:** Peleg Harel Ofir Itzhak Shahar; Ohad Ben-Shahar
>
> **摘要:** Jigsaw puzzle solving, the problem of constructing a coherent whole from a set of non-overlapping unordered visual fragments, is fundamental to numerous applications, and yet most of the literature of the last two decades has focused thus far on less realistic puzzles whose pieces are identical squares. Here, we formalize a new type of jigsaw puzzle where the pieces are general convex polygons generated by cutting through a global polygonal shape with an arbitrary number of straight cuts, a generation model inspired by the celebrated Lazy caterer sequence. We analyze the theoretical properties of such puzzles, including the inherent challenges in solving them once pieces are contaminated with geometrical noise. To cope with such difficulties and obtain tractable solutions, we abstract the problem as a multi-body spring-mass dynamical system endowed with hierarchical loop constraints and a layered reconstruction process. We define evaluation metrics and present experimental results on both apictorial and pictorial puzzles to show that they are solvable completely automatically.
>
---
#### [replaced 038] Uncertainty-Aware Image Classification In Biomedical Imaging Using Spectral-normalized Neural Gaussian Processes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.02370](https://arxiv.org/pdf/2602.02370)**

> **作者:** Uma Meleti; Jeffrey J. Nirschl
>
> **备注:** Published at the IEEE International Symposium on Biomedical Imaging (ISBI) 2026
>
> **摘要:** Accurate histopathologic interpretation is key for clinical decision-making; however, current deep learning models for digital pathology are often overconfident and poorly calibrated in out-of-distribution (OOD) settings, which limit trust and clinical adoption. Safety-critical medical imaging workflows benefit from intrinsic uncertainty-aware properties that can accurately reject OOD input. We implement the Spectral-normalized Neural Gaussian Process (SNGP), a set of lightweight modifications that apply spectral normalization and replace the final dense layer with a Gaussian process layer to improve single-model uncertainty estimation and OOD detection. We evaluate SNGP vs. deterministic and MonteCarlo dropout on six datasets across three biomedical classification tasks: white blood cells, amyloid plaques, and colorectal histopathology. SNGP has comparable in-distribution performance while significantly improving uncertainty estimation and OOD detection. Thus, SNGP or related models offer a useful framework for uncertainty-aware classification in digital pathology, supporting safe deployment and building trust with pathologists.
>
---
#### [replaced 039] Subspace-Guided Feature Reconstruction for Unsupervised Anomaly Localization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2309.13904](https://arxiv.org/pdf/2309.13904)**

> **作者:** Katsuya Hotta; Chao Zhang; Yoshihiro Hagihara; Takuya Akashi
>
> **摘要:** Unsupervised anomaly localization aims to identify anomalous regions that deviate from normal sample patterns. Most recent methods perform feature matching or reconstruction for the target sample with pre-trained deep neural networks. However, they still struggle to address challenging anomalies because the deep embeddings stored in the memory bank can be less powerful and informative. Specifically, prior methods often overly rely on the finite resources stored in the memory bank, which leads to low robustness to unseen targets. In this paper, we propose a novel subspace-guided feature reconstruction framework to pursue adaptive feature approximation for anomaly localization. It first learns to construct low-dimensional subspaces from the given nominal samples, and then learns to reconstruct the given deep target embedding by linearly combining the subspace basis vectors using the self-expressive model. Our core is that, despite the limited resources in the memory bank, the out-of-bank features can be alternatively ``mimicked'' to adaptively model the target. Moreover, we propose a sampling method that leverages the sparsity of subspaces and allows the feature reconstruction to depend only on a small resource subset, contributing to less memory overhead. Extensive experiments on three benchmark datasets demonstrate that our approach generally achieves state-of-the-art anomaly localization performance.
>
---
#### [replaced 040] AGMA: Adaptive Gaussian Mixture Anchors for Prior-Guided Multimodal Human Trajectory Forecasting
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.04204](https://arxiv.org/pdf/2602.04204)**

> **作者:** Chao Li; Rui Zhang; Siyuan Huang; Xian Zhong; Hongbo Jiang
>
> **备注:** Withdrawn for substantial revision and will be re-uploaded as a new manuscript
>
> **摘要:** Human trajectory forecasting requires capturing the multimodal nature of pedestrian behavior. However, existing approaches suffer from prior misalignment. Their learned or fixed priors often fail to capture the full distribution of plausible futures, limiting both prediction accuracy and diversity. We theoretically establish that prediction error is lower-bounded by prior quality, making prior modeling a key performance bottleneck. Guided by this insight, we propose AGMA (Adaptive Gaussian Mixture Anchors), which constructs expressive priors through two stages: extracting diverse behavioral patterns from training data and distilling them into a scene-adaptive global prior for inference. Extensive experiments on ETH-UCY, Stanford Drone, and JRDB datasets demonstrate that AGMA achieves state-of-the-art performance, confirming the critical role of high-quality priors in trajectory forecasting.
>
---
#### [replaced 041] SynthPix: A lightspeed PIV image generator
- **分类: cs.DC; cs.CV; cs.LG; eess.IV**

- **链接: [https://arxiv.org/pdf/2512.09664](https://arxiv.org/pdf/2512.09664)**

> **作者:** Antonio Terpin; Alan Bonomi; Francesco Banelli; Raffaello D'Andrea
>
> **备注:** Code: this https URL. Published in SoftwareX
>
> **摘要:** We describe SynthPix, a synthetic image generator for Particle Image Velocimetry (PIV) with a focus on performance and parallelism on accelerators, implemented in JAX. SynthPix produces PIV image pairs from prescribed flow fields while exposing a configuration interface aligned with common PIV imaging and acquisition parameters (e.g., seeding density, particle image size, illumination nonuniformity, noise, blur, and timing). In contrast to offline dataset generation workflows, SynthPix is built to stream images on-the-fly directly into learning and benchmarking pipelines, enabling data-hungry methods and closed-loop procedures -- such as adaptive sampling and acquisition/parameter co-design -- without prohibitive storage and input-output costs. We demonstrate that SynthPix is compatible with a broad range of application scenarios, including controlled laboratory experiments and riverine image velocimetry, and supports rapid sweeps over nuisance factors for systematic robustness evaluation. SynthPix is a tool that supports the flow quantification community and in this paper we describe the main ideas behind the software package.
>
---
#### [replaced 042] SecureWebArena: A Holistic Security Evaluation Benchmark for LVLM-based Web Agents
- **分类: cs.CR; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.10073](https://arxiv.org/pdf/2510.10073)**

> **作者:** Zonghao Ying; Yangguang Shao; Jianle Gan; Gan Xu; Wenxin Zhang; Quanchen Zou; Junzheng Shi; Zhenfei Yin; Mingchuan Zhang; Aishan Liu; Xianglong Liu
>
> **备注:** ACL
>
> **摘要:** Large vision-language model (LVLM)-based web agents are emerging as powerful tools for automating complex online tasks. However, when deployed in real-world environments, they face serious security risks, motivating the design of security evaluation benchmarks. Existing benchmarks provide only partial coverage, typically restricted to narrow scenarios such as user-level prompt manipulation, and thus fail to capture the broad range of agent vulnerabilities. To address this gap, we present \tool{}, the first holistic benchmark for evaluating the security of LVLM-based web agents. \tool{} first introduces a unified evaluation suite comprising six simulated but realistic web environments (\eg, e-commerce platforms, community forums) and includes 2,970 high-quality trajectories spanning diverse tasks and attack settings. The suite defines a structured taxonomy of six attack vectors spanning both user-level and environment-level manipulations. In addition, we introduce a multi-layered evaluation protocol that analyzes agent failures across three critical dimensions: internal reasoning, behavioral trajectory, and task outcome, facilitating a fine-grained risk analysis that goes far beyond simple success metrics. Using this benchmark, we conduct large-scale experiments on 9 representative LVLMs, which fall into three categories: general-purpose, agent-specialized, and GUI-grounded. Our results show that all tested agents are consistently vulnerable to subtle adversarial manipulations and reveal critical trade-offs between model specialization and security. By providing (1) a comprehensive benchmark suite with diverse environments and a multi-layered evaluation pipeline, and (2) empirical insights into the security challenges of modern LVLM-based web agents, \tool{} establishes a foundation for advancing trustworthy web agent deployment.
>
---
#### [replaced 043] Face Density as a Proxy for Data Complexity: Quantifying the Hardness of Instance Count
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.09689](https://arxiv.org/pdf/2604.09689)**

> **作者:** Abolfazl Mohammadi-Seif; Ricardo Baeza-Yates
>
> **备注:** This work has been accepted for publication in the Proceedings of IEEE CAI 2026. The final published version should be cited
>
> **摘要:** Machine learning progress has historically prioritized model-centric innovations, yet achievable performance is frequently capped by the intrinsic complexity of the data itself. In this work, we isolate and quantify the impact of instance density (measured by face count) as a primary driver of data complexity. Rather than simply observing that ``crowded scenes are harder,'' we rigorously control for class imbalance to measure the precise degradation caused by density alone. Controlled experiments on the WIDER FACE and Open Images datasets, restricted to exactly 1 to 18 faces per image with perfectly balanced sampling, reveal that model performance degrades monotonically with increasing face count. This trend holds across classification, regression, and detection paradigms, even when models are fully exposed to the entire density range. Furthermore, we demonstrate that models trained on low-density regimes fail to generalize to higher densities, exhibiting a systematic under-counting bias, with error rates increasing by up to 4.6x, which suggests density acts as a domain shift. These findings establish instance density as an intrinsic, quantifiable dimension of data hardness and motivate specific interventions in curriculum learning and density-stratified evaluation.
>
---
#### [replaced 044] IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.07926](https://arxiv.org/pdf/2603.07926)**

> **作者:** Sunghyun Baek; Jaemyung Yu; Seunghee Koh; Minsu Kim; Hyeonseong Jeon; Junmo Kim
>
> **备注:** ICLR 2026
>
> **摘要:** Test-time adaptation (TTA) has been widely explored to prevent performance degradation when test data differ from the training distribution. However, fully leveraging the rich representations of large pretrained models with minimal parameter updates remains underexplored. In this paper, we propose Intrinsic Mixture of Spectral Experts (IMSE) that leverages the spectral experts inherently embedded in Vision Transformers. We decompose each linear layer via singular value decomposition (SVD) and adapt only the singular values, while keeping the singular vectors fixed. We further identify a key limitation of entropy minimization in TTA: it often induces feature collapse, causing the model to rely on domain-specific features rather than class-discriminative features. To address this, we propose a diversity maximization loss based on expert-input alignment, which encourages diverse utilization of spectral experts during adaptation. In the continual test-time adaptation (CTTA) scenario, beyond preserving pretrained knowledge, it is crucial to retain and reuse knowledge from previously observed domains. We introduce Domain-Aware Spectral Code Retrieval, which estimates input distributions to detect domain shifts, and retrieves adapted singular values for rapid adaptation. Consequently, our method achieves state-of-the-art performance on various distribution-shift benchmarks under the TTA setting. In CTTA and Gradual CTTA, it further improves accuracy by 3.4 percentage points (pp) and 2.4 pp, respectively, while requiring 385 times fewer trainable parameters. Our code is available at this https URL.
>
---
#### [replaced 045] VPTracker: Global Vision-Language Tracking via Visual Prompt
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.22799](https://arxiv.org/pdf/2512.22799)**

> **作者:** Jingchao Wang; Kaiwen Zhou; Zhijian Wu; Kunhua Ji; Dingjiang Huang; Yefeng Zheng
>
> **备注:** 7 pages
>
> **摘要:** Vision-Language Tracking aims to continuously localize objects described by a visual template and a language description. Existing methods, however, are typically limited to local search, making them prone to failures under viewpoint changes, occlusions, and rapid target movements. In this work, we introduce the first global tracking framework based on Multimodal Large Language Models (VPTracker), exploiting their powerful semantic reasoning to locate targets across the entire image space. While global search improves robustness and reduces drift, it also introduces distractions from visually or semantically similar objects. To address this, we propose a location-aware visual prompting mechanism that incorporates spatial priors into the MLLM. Specifically, we construct a region-level prompt based on the target's previous location, enabling the model to prioritize region-level recognition and resort to global inference only when necessary. This design retains the advantages of global tracking while effectively suppressing interference from distracting visual content. Extensive experiments show that our approach significantly enhances tracking stability and target disambiguation under challenging scenarios, opening a new avenue for integrating MLLMs into visual tracking. Code is available at this https URL.
>
---
#### [replaced 046] PixelCAM: Pixel Class Activation Mapping for Histology Image Classification and ROI Localization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.24135](https://arxiv.org/pdf/2503.24135)**

> **作者:** Alexis Guichemerre; Soufiane Belharbi; Mohammadhadi Shateri; Luke McCaffrey; Eric Granger
>
> **备注:** 43 pages, 24 figures, Medical Imaging with Deep Learning (MIDL 2025)
>
> **摘要:** Weakly supervised object localization (WSOL) methods allow training models to classify images and localize ROIs. WSOL only requires low-cost image-class annotations yet provides a visually interpretable classifier. Standard WSOL methods rely on class activation mapping (CAM) methods to produce spatial localization maps according to a single- or two-step strategy. While both strategies have made significant progress, they still face several limitations with histology images. Single-step methods can easily result in under- or over-activation due to the limited visual ROI saliency in histology images and scarce localization cues. They also face the well-known issue of asynchronous convergence between classification and localization tasks. The two-step approach is sub-optimal because it is constrained to a frozen classifier, limiting the capacity for localization. Moreover, these methods also struggle when applied to out-of-distribution (OOD) datasets. In this paper, a multi-task approach for WSOL is introduced for simultaneous training of both tasks to address the asynchronous convergence problem. In particular, localization is performed in the pixel-feature space of an image encoder that is shared with classification. This allows learning discriminant features and accurate delineation of foreground/background regions to support ROI localization and image classification. We propose PixelCAM, a cost-effective foreground/background pixel-wise classifier in the pixel-feature space that allows for spatial object localization. Using partial-cross entropy, PixelCAM is trained using pixel pseudo-labels collected from a pretrained WSOL model. Both image and pixel-wise classifiers are trained simultaneously using standard gradient descent. In addition, our pixel classifier can easily be integrated into CNN- and transformer-based architectures without any modifications.
>
---
#### [replaced 047] Rein3D: Reinforced 3D Indoor Scene Generation with Panoramic Video Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.10578](https://arxiv.org/pdf/2604.10578)**

> **作者:** Dehui Wang; Congsheng Xu; Rong Wei; Yue Shi; Shoufa Chen; Dingxiang Luo; Tianshuo Yang; Xiaokang Yang; Wei Sui; Yusen Qin; Rui Tang; Yao Mu
>
> **摘要:** The growing demand for Embodied AI and VR applications has highlighted the need for synthesizing high-quality 3D indoor scenes from sparse inputs. However, existing approaches struggle to infer massive amounts of missing geometry in large unseen areas while maintaining global consistency, often producing locally plausible but globally inconsistent reconstructions. We present Rein3D, a framework that reconstructs full 360-degree indoor environments by coupling explicit 3D Gaussian Splatting (3DGS) with temporally coherent priors from video diffusion models. Our approach follows a "restore-and-refine" paradigm: we employ a radial exploration strategy to render imperfect panoramic videos along trajectories starting from the origin, effectively uncovering occluded regions from a coarse 3DGS initialization. These sequences are restored by a panoramic video-to-video diffusion model and further enhanced via video super-resolution to synthesize high-fidelity geometry and textures. Finally, these refined videos serve as pseudo-ground truths to update the global 3D Gaussian field. To support this task, we construct PanoV2V-15K, a dataset of over 15K paired clean and degraded panoramic videos for diffusion-based scene restoration. Experiments demonstrate that Rein3D produces photorealistic and globally consistent 3D scenes and significantly improves long-range camera exploration compared with existing baselines.
>
---
#### [replaced 048] One Model for All: Unified Try-On and Try-Off in Any Pose via LLM-Inspired Bidirectional Tweedie Diffusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.04559](https://arxiv.org/pdf/2508.04559)**

> **作者:** Jinxi Liu; Zijian He; Guangrun Wang; Guanbin Li; Liang Lin
>
> **摘要:** Recent diffusion-based approaches have made significant advances in image-based virtual try-on, enabling more realistic and end-to-end garment synthesis. However, most existing methods remain constrained by their reliance on exhibition garments and segmentation masks, as well as their limited ability to handle flexible pose variations. These limitations reduce their practicality in real-world scenarios; for instance, users cannot easily transfer garments worn by one person onto another, and the generated try-on results are typically restricted to the same pose as the reference image. In this paper, we introduce OMFA (One Model For All), a unified diffusion framework for both virtual try-on and try-off that operates without the need for exhibition garments and supports arbitrary poses. OMFA is inspired by the mask-based paradigm of discrete diffusion language models and unifies try-on and try-off within a bidirectional framework. It is built upon a Bidirectional Tweedie Diffusion process for target-selective denoising in latent space. Instead of imposing lower body constraints, OMFA is an entirely mask-free framework that requires only a single portrait and a target garment as inputs, and is designed to support flexible outfit combinations and cross-person garment transfer, making it better aligned with practical usage scenarios. Additionally, by leveraging SMPL-X-based pose conditioning, OMFA supports multi-view and arbitrary-pose try-on from just one image. Extensive experiments demonstrate that OMFA achieves state-of-the-art results on both try-on and try-off tasks, providing a practical and generalizable solution for virtual garment synthesis. Project page: this https URL
>
---
#### [replaced 049] FAST-DIPS: Adjoint-Free Analytic Steps and Hard-Constrained Likelihood Correction for Diffusion-Prior Inverse Problems
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.01591](https://arxiv.org/pdf/2603.01591)**

> **作者:** Minwoo Kim; Seunghyeok Shin; Hongki Lim
>
> **摘要:** Training-free diffusion priors enable inverse-problem solvers without retraining, but for nonlinear forward operators data consistency often relies on repeated derivatives or inner optimization/MCMC loops with conservative step sizes, incurring many iterations and denoiser/score evaluations. We propose a training-free solver that replaces these inner loops with a hard measurement-space feasibility constraint (closed-form projection) and an analytic, model-optimal step size, enabling a small, fixed compute budget per noise level. Anchored at the denoiser prediction, the correction is approximated via an adjoint-free, ADMM-style splitting with projection and a few steepest-descent updates, using one VJP and either one JVP or a forward-difference probe, followed by backtracking and decoupled re-annealing. We prove local model optimality and descent under backtracking for the step-size rule, and derive an explicit KL bound for mode-substitution re-annealing under a local Gaussian conditional surrogate. We also develop a latent variant and a one-parameter pixel$\rightarrow$latent hybrid schedule. Experiments achieve competitive PSNR/SSIM/LPIPS with up to 19.5$\times$ speedup, without hand-coded adjoints or inner MCMC.
>
---
#### [replaced 050] Architecture-Agnostic Modality-Isolated Gated Fusion for Robust Multi-Modal Prostate MRI Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.10702](https://arxiv.org/pdf/2604.10702)**

> **作者:** Yongbo Shu; Wenzhao Xie; Shanhu Yao; Zirui Xin; Luo Lei; Kewen Chen; Aijing Luo
>
> **备注:** 36 pages, 4 figures, 5 tables
>
> **摘要:** Multi-parametric prostate MRI -- combining T2-weighted, apparent diffusion coefficient, and high b-value diffusion-weighted sequences -- is central to non-invasive detection of clinically significant prostate cancer, yet in routine practice individual sequences may be missing or degraded by motion, artifacts, or abbreviated protocols. Existing multi-modal fusion strategies typically assume complete inputs and entangle modality-specific information at early layers, offering limited resilience when one channel is corrupted or absent. We propose Modality-Isolated Gated Fusion (MIGF), an architecture-agnostic module that maintains separate modality-specific encoding streams before a learned gating stage, combined with modality dropout training to enforce compensation behavior under incomplete inputs. We benchmark six bare backbones and assess MIGF-equipped models under seven missing-modality and artifact scenarios on the PI-CAI dataset (1,500 studies, fold-0 split, five random seeds). Among bare backbones, nnUNet provided the strongest balance of performance and stability. MIGF improved ideal-scenario Ranking Score for UNet, nnUNet, and Mamba by 2.8%, 4.6%, and 13.4%, respectively; the best model, MIGFNet-nnUNet (gating + ModDrop, no deep supervision), achieved 0.7304 +/- 0.056. Mechanistic analysis reveals that robustness gains arise from strict modality isolation and dropout-driven compensation rather than adaptive per-sample quality routing: the gate converged to a stable modality prior, and deep supervision was beneficial only for the largest backbone while degrading lighter models. These findings support a simpler design principle for robust multi-modal segmentation: structurally contain corrupted inputs first, then train explicitly for incomplete-input compensation.
>
---
#### [replaced 051] JanusCoder: Towards a Foundational Visual-Programmatic Interface for Code Intelligence
- **分类: cs.AI; cs.CL; cs.CV; cs.SE**

- **简介: 该论文提出JanusCoder，解决代码与视觉输出的多模态生成问题。通过构建大规模数据集，训练统一模型实现文本或视觉输入生成代码，提升代码智能应用效果。**

- **链接: [https://arxiv.org/pdf/2510.23538](https://arxiv.org/pdf/2510.23538)**

> **作者:** Qiushi Sun; Jingyang Gong; Yang Liu; Qiaosheng Chen; Lei Li; Kai Chen; Qipeng Guo; Ben Kao; Fei Yuan
>
> **备注:** ICLR 2026 Camera Ready Version
>
> **摘要:** The scope of neural code intelligence is rapidly expanding beyond text-based source code to encompass the rich visual outputs that programs generate. This visual dimension is critical for advanced applications like flexible content generation and precise, program-driven editing of visualizations. However, progress has been impeded by the scarcity of high-quality multimodal code data, a bottleneck stemming from challenges in synthesis and quality assessment. To address these challenges, we make contributions from both a data and modeling perspective. We first introduce a complete synthesis toolkit that leverages reciprocal synergies between data modalities to efficiently produce a large-scale, high-quality corpus spanning from standard charts to complex interactive web UIs and code-driven animations. Leveraging this toolkit, we construct JanusCode-800K, the largest multimodal code corpus to date. This powers the training of our models, JanusCoder and JanusCoderV, which establish a visual-programmatic interface for generating code from textual instructions, visual inputs, or a combination of both. Our unified model is a departure from existing approaches that build specialized models for isolated tasks. Extensive experiments on both text-centric and vision-centric coding tasks demonstrate the superior performance of the JanusCoder series, with our 7B to 14B scale models approaching or even exceeding the performance of commercial models. Furthermore, extensive analysis provides key insights into harmonizing programmatic logic with its visual expression. Our code and checkpoints are available at this https URL.
>
---
#### [replaced 052] Dress-ED: Instruction-Guided Editing for Virtual Try-On and Try-Off
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.22607](https://arxiv.org/pdf/2603.22607)**

> **作者:** Fulvio Sanguigni; Davide Lobba; Bin Ren; Marcella Cornia; Nicu Sebe; Rita Cucchiara
>
> **备注:** Project page: this https URL
>
> **摘要:** Recent advances in Virtual Try-On (VTON) and Virtual Try-Off (VTOFF) have greatly improved photo-realistic fashion synthesis and garment reconstruction. However, existing datasets remain static, lacking instruction-driven editing for controllable and interactive fashion generation. In this work, we introduce the Dress Editing Dataset (Dress-ED), the first large-scale benchmark that unifies VTON, VTOFF, and text-guided garment editing within a single framework. Each sample in Dress-ED includes an in-shop garment image, the corresponding person image wearing the garment, their edited counterparts, and a natural-language instruction of the desired modification. Built through a fully automated multimodal pipeline that integrates MLLM-based garment understanding, diffusion-based editing, and LLM-guided verification, Dress-ED comprises over 146k verified quadruplets spanning three garment categories and seven edit types, including both appearance (e.g., color, pattern, material) and structural (e.g., sleeve length, neckline) modifications. Based on this benchmark, we further propose a unified multimodal diffusion framework that jointly reasons over linguistic instructions and visual garment cues, serving as a strong baseline for instruction-driven VTON and VTOFF. Dataset and code will be made publicly available. Project page: this https URL
>
---
#### [replaced 053] StableSketcher: Enhancing Diffusion Model for Pixel-based Sketch Generation via Visual Question Answering Feedback
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.20093](https://arxiv.org/pdf/2510.20093)**

> **作者:** Jiho Park; Sieun Choi; Jaeyoon Seo; Jihie Kim
>
> **备注:** Under review at IEEE Access. Author-submitted preprint. Not the IEEE-published version
>
> **摘要:** Although recent advancements in diffusion models have significantly enriched the quality of generated images, challenges remain in synthesizing pixel-based human-drawn sketches, a representative example of abstract expression. To combat these challenges, we propose StableSketcher, a novel framework that empowers diffusion models to generate hand-drawn sketches with high prompt fidelity. Within this framework, we fine-tune the variational autoencoder to optimize latent decoding, enabling it to better capture the characteristics of sketches. In parallel, we integrate a new reward function for reinforcement learning based on visual question answering, which improves text-image alignment and semantic consistency. Extensive experiments demonstrate that StableSketcher generates sketches with improved stylistic fidelity, achieving better alignment with prompts compared to the Stable Diffusion baseline. Additionally, we introduce SketchDUO, to the best of our knowledge, the first dataset comprising instance-level sketches paired with captions and question-answer pairs, thereby addressing the limitations of existing datasets that rely on image-label pairs. Our code and dataset will be made publicly available upon acceptance. Project page: this https URL
>
---
#### [replaced 054] Do vision models perceive illusory motion in static images like humans?
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.09853](https://arxiv.org/pdf/2604.09853)**

> **作者:** Isabella Elaine Rosario; Fan L. Cheng; Zitang Sun; Nikolaus Kriegeskorte
>
> **备注:** Accepted to CVPR 2026 Findings
>
> **摘要:** Understanding human motion processing is essential for building reliable, human-centered computer vision systems. Although deep neural networks (DNNs) achieve strong performance in optical flow estimation, they remain less robust than humans and rely on fundamentally different computational strategies. Visual motion illusions provide a powerful probe into these mechanisms, revealing how human and machine vision align or diverge. While recent DNN-based motion models can reproduce dynamic illusions such as reverse-phi, it remains unclear whether they can perceive illusory motion in static images, exemplified by the Rotating Snakes illusion. We evaluate several representative optical flow models on Rotating Snakes and show that most fail to generate flow fields consistent with human perception. Under simulated conditions mimicking saccadic eye movements, only the human-inspired Dual-Channel model exhibits the expected rotational motion, with the closest correspondence emerging during the saccade simulation. Ablation analyses further reveal that both luminance-based and higher-order color--feature--based motion signals contribute to this behavior and that a recurrent attention mechanism is critical for integrating local cues. Our results highlight a substantial gap between current optical-flow models and human visual motion processing, and offer insights for developing future motion-estimation systems with improved correspondence to human perception and human-centric AI.
>
---
#### [replaced 055] DC-TTA: Divide-and-Conquer Framework for Test-Time Adaptation of Interactive Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.23104](https://arxiv.org/pdf/2506.23104)**

> **作者:** Jihun Kim; Hoyong Kwon; Hyeokjun Kweon; Wooseong Jeong; Kuk-Jin Yoon
>
> **备注:** accepted at ICCV 2025
>
> **摘要:** Interactive segmentation (IS) allows users to iteratively refine object boundaries with minimal cues, such as positive and negative clicks. While the Segment Anything Model (SAM) has garnered attention in the IS community for its promptable segmentation capabilities, it often struggles in specialized domains or when handling complex scenarios (e.g., camouflaged or multi-part objects). To overcome these challenges, we propose DC-TTA, a novel test-time adaptation (TTA) framework that adapts SAM on a per-sample basis by leveraging user interactions as supervision. Instead of forcing a single model to incorporate all user clicks at once, DC-TTA partitions the clicks into more coherent subsets, each processed independently via TTA with a separated model. This Divide-and-Conquer strategy reduces conflicts among diverse cues and enables more localized updates. Finally, we merge the adapted models to form a unified predictor that integrates the specialized knowledge from each subset. Experimental results across various benchmarks demonstrate that DC-TTA significantly outperforms SAM's zero-shot results and conventional TTA methods, effectively handling complex tasks such as camouflaged object segmentation with fewer interactions and improved accuracy.
>
---
#### [replaced 056] OctoTools: An Agentic Framework with Extensible Tools for Complex Reasoning
- **分类: cs.LG; cs.CL; cs.CV; cs.MA**

- **简介: 该论文提出OctoTools，一个无需训练的多智能体框架，用于解决跨领域的复杂推理任务。针对现有方法在工具类型和领域适应性上的不足，OctoTools通过标准化工具卡、规划器和执行器实现高效多步骤问题解决。**

- **链接: [https://arxiv.org/pdf/2502.11271](https://arxiv.org/pdf/2502.11271)**

> **作者:** Pan Lu; Bowen Chen; Sheng Liu; Rahul Thapa; Joseph Boen; James Zou
>
> **备注:** 88 pages, 18 figures. Accepted to ACL 2026
>
> **摘要:** Solving complex reasoning tasks may involve visual understanding, domain knowledge retrieval, numerical calculation, and multi-step reasoning. Existing methods augment large language models (LLMs) with external tools but are restricted to specialized domains, limited tool types, or require additional training data. In this paper, we introduce OctoTools, a training-free, user-friendly, and easily extensible multi-agent framework designed to tackle complex reasoning across diverse domains. OctoTools introduces standardized tool cards to encapsulate tool functionality, a planner for both high-level and low-level planning, and an executor to carry out tool usage. We validate OctoTools' generality across 16 diverse tasks (including MathVista, MMLU-Pro, MedQA, and GAIA-Text), achieving substantial average accuracy gains of 9.3% over GPT-4o. Furthermore, OctoTools also outperforms AutoGen, GPT-Functions, and LangChain by up to 10.6% when given the same set of tools. Through comprehensive analysi, ablations, and robustness tests with compact backbones and noisy tool environments, OctoTools demonstrates advantages in task planning, effective tool usage, and multi-step problem solving. Code, demos, and visualization are publicly available at this https URL.
>
---
#### [replaced 057] Visual Diffusion Models are Geometric Solvers
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.21697](https://arxiv.org/pdf/2510.21697)**

> **作者:** Nir Goren; Shai Yehezkel; Omer Dahary; Andrey Voynov; Or Patashnik; Daniel Cohen-Or
>
> **备注:** Project page: this https URL
>
> **摘要:** In this paper we show that visual diffusion models can serve as effective geometric solvers: they can directly reason about geometric problems by working in pixel space. We first demonstrate this on the Inscribed Square Problem, a long-standing problem in geometry that asks whether every Jordan curve contains four points forming a square. We then extend the approach to two other well-known hard geometric problems: the Steiner Tree Problem and the Simple Polygon Problem. Our method treats each problem instance as an image and trains a standard visual diffusion model that transforms Gaussian noise into an image representing a valid approximate solution that closely matches the exact one. The model learns to transform noisy geometric structures into correct configurations, effectively recasting geometric reasoning as image generation. Unlike prior work that necessitates specialized architectures and domain-specific adaptations when applying diffusion to parametric geometric representations, we employ a standard visual diffusion model that operates on the visual representation of the problem. This simplicity highlights a surprising bridge between generative modeling and geometric problem solving. Beyond the specific problems studied here, our results point toward a broader paradigm: operating in image space provides a general and practical framework for approximating notoriously hard problems, and opens the door to tackling a far wider class of challenging geometric tasks.
>
---
#### [replaced 058] WebChain: A Large-Scale Human-Annotated Dataset of Real-World Web Interaction Traces
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.05295](https://arxiv.org/pdf/2603.05295)**

> **作者:** Sicheng Fan; Rui Wan; Yifei Leng; Gaoning Liang; Li Ling; Yanyi Shang; Dehan Kong
>
> **摘要:** We introduce WebChain, the largest open-source dataset of human-annotated trajectories on real-world websites, designed to accelerate reproducible research in web agents. It contains 31,725 trajectories and 318k steps, featuring a core Triple Alignment of visual, structural, and action data to provide rich, multi-modal supervision. The data is collected via a scalable pipeline that ensures coverage of complex, high-value tasks often missed by synthetic methods. Leveraging this dataset, we propose a Dual Mid-Training recipe that decouples spatial grounding from planning, achieving state-of-the-art performance on our proposed WebChainBench and other public GUI benchmarks. Our work provides the data and insights necessary to build and rigorously evaluate the next generation of scalable web agents.
>
---
#### [replaced 059] STGV: Spatio-Temporal Hash Encoding for Gaussian-based Video Representation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.10910](https://arxiv.org/pdf/2604.10910)**

> **作者:** Jierun Lin; Jiacong Chen; Qingyu Mao; Shuai Liu; Xiandong Meng; Fanyang Meng; Yongsheng Liang
>
> **摘要:** 2D Gaussian Splatting (2DGS) has recently become a promising paradigm for high-quality video representation. However, existing methods employ content-agnostic or spatio-temporal feature overlapping embeddings to predict canonical Gaussian primitive deformations, which entangles static and dynamic components in videos and prevents modeling their distinct properties effectively. These result in inaccurate predictions for spatio-temporal deformations and unsatisfactory representation quality. To address these problems, this paper proposes a Spatio-Temporal hash encoding framework for Gaussian-based Video representation (STGV). By decomposing video features into learnable 2D spatial and 3D temporal hash encodings, STGV effectively facilitates the learning of motion patterns for dynamic components while maintaining background details for static elements. In addition, we construct a more stable and consistent initial canonical Gaussian representation through a key frame canonical initialization strategy, preventing from feature overlapping and a structurally incoherent geometry representation. Experimental results demonstrate that our method attains better video representation quality (+0.98 PSNR) against other Gaussian-based methods and achieves competitive performance in downstream video tasks.
>
---
#### [replaced 060] Retrieving to Recover: Towards Incomplete Audio-Visual Question Answering via Semantic-consistent Purification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.10695](https://arxiv.org/pdf/2604.10695)**

> **作者:** Jiayu Zhang; Shuo Ye; Qilang Ye; Zihan Song; Jiajian Huang; Zitong Yu
>
> **备注:** Accepted by ACL 2026 Main Conference
>
> **摘要:** Recent Audio-Visual Question Answering (AVQA) methods have advanced significantly. However, most AVQA methods lack effective mechanisms for handling missing modalities, suffering from severe performance degradation in real-world scenarios with data interruptions. Furthermore, prevailing methods for handling missing modalities predominantly rely on generative imputation to synthesize missing features. While partially effective, these methods tend to capture inter-modal commonalities but struggle to acquire unique, modality-specific knowledge within the missing data, leading to hallucinations and compromised reasoning accuracy. To tackle these challenges, we propose R$^{2}$ScP, a novel framework that shifts the paradigm of missing modality handling from traditional generative imputation to retrieval-based recovery. Specifically, we leverage cross-modal retrieval via unified semantic embeddings to acquire missing domain-specific knowledge. To maximize semantic restoration, we introduce a context-aware adaptive purification mechanism that eliminates latent semantic noise within the retrieved data. Additionally, we employ a two-stage training strategy to explicitly model the semantic relationships between knowledge from different sources. Extensive experiments demonstrate that R$^{2}$ScP significantly improves AVQA and enhances robustness in modal-incomplete scenarios.
>
---
#### [replaced 061] AniGen: Unified $S^3$ Fields for Animatable 3D Asset Generation
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2604.08746](https://arxiv.org/pdf/2604.08746)**

> **作者:** Yi-Hua Huang; Zi-Xin Zou; Yuting He; Chirui Chang; Cheng-Feng Pu; Ziyi Yang; Yuan-Chen Guo; Yan-Pei Cao; Xiaojuan Qi
>
> **备注:** 16 pages, 12 figures
>
> **摘要:** Animatable 3D assets, defined as geometry equipped with an articulated skeleton and skinning weights, are fundamental to interactive graphics, embodied agents, and animation production. While recent 3D generative models can synthesize visually plausible shapes from images, the results are typically static. Obtaining usable rigs via post-hoc auto-rigging is brittle and often produces skeletons that are topologically inconsistent with the generated geometry. We present AniGen, a unified framework that directly generates animate-ready 3D assets conditioned on a single image. Our key insight is to represent shape, skeleton, and skinning as mutually consistent $S^3$ Fields (Shape, Skeleton, Skin) defined over a shared spatial domain. To enable the robust learning of these fields, we introduce two technical innovations: (i) a confidence-decaying skeleton field that explicitly handles the geometric ambiguity of bone prediction at Voronoi boundaries, and (ii) a dual skin feature field that decouples skinning weights from specific joint counts, allowing a fixed-architecture network to predict rigs of arbitrary complexity. Built upon a two-stage flow-matching pipeline, AniGen first synthesizes a sparse structural scaffold and then generates dense geometry and articulation in a structured latent space. Extensive experiments demonstrate that AniGen substantially outperforms state-of-the-art sequential baselines in rig validity and animation quality, generalizing effectively to in-the-wild images across diverse categories including animals, humanoids, and machinery. Homepage: this https URL
>
---
#### [replaced 062] A document is worth a structured record: Principled inductive bias design for document recognition
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.08458](https://arxiv.org/pdf/2507.08458)**

> **作者:** Benjamin Meyer; Lukas Tuggener; Sascha Hänzi; Daniel Schmid; Erdal Ayfer; Benjamin F. Grewe; Ahmed Abdulkadir; Thilo Stadelmann
>
> **摘要:** Many document types use intrinsic, convention-driven structures that serve to encode precise and structured information, such as the conventions governing engineering drawings. However, many state-of-the-art approaches treat document recognition as a mere computer vision problem, neglecting these underlying document-type-specific structural properties, making them dependent on sub-optimal heuristic post-processing and rendering many less frequent or more complicated document types inaccessible to modern document recognition. We suggest a novel perspective that frames document recognition as a transcription task from a document to a record. This implies a natural grouping of documents based on the intrinsic structure inherent in their transcription, where related document types can be treated (and learned) similarly. We propose a method to design structure-specific relational inductive biases for the underlying machine-learned end-to-end document recognition systems, and a respective base transformer architecture that we successfully adapt to different structures. We demonstrate the effectiveness of the so-found inductive biases in extensive experiments with progressively complex record structures from monophonic sheet music, shape drawings, and simplified engineering drawings. By integrating an inductive bias for unrestricted graph structures, we train the first-ever successful end-to-end model to transcribe mechanical engineering drawings to their inherently interlinked information. Our approach is relevant to inform the design of document recognition systems for document types that are less well understood than standard OCR, OMR, etc., and serves as a guide to unify the design of future document foundation models.
>
---
#### [replaced 063] SIRI-Bench: Challenging VLMs' Spatial Intelligence through Complex Reasoning Tasks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.14512](https://arxiv.org/pdf/2506.14512)**

> **作者:** Zijian Song; Xiaoxin Lin; Qiuming Huang; Sihan Qin; Guangrun Wang; Liang Lin
>
> **备注:** 20 pages, 11 figures
>
> **摘要:** Large Language Models (LLMs) have undergone rapid progress, largely attributed to reinforcement learning on complex reasoning tasks. In contrast, while spatial intelligence is fundamental for Vision-Language Models (VLMs) in real-world interaction, the systematic study of their complex spatial reasoning remains underexplored. To bridge this gap, we introduce SIRI-Bench, a benchmark designed to evaluate VLMs' structural spatial intelligence through spatial-grounded reasoning tasks. SIRI-Bench comprises 9,000 video-question-answer triplets, where each problem is embedded in a realistic 3D scene. The benchmark is carefully designed so that solving each problem requires both spatial comprehension and structural reasoning. To facilitate large-scale data synthesis, we develop an Automatic Scene Creation Engine that employs collaborative LLM agents to translate abstract mathematical problems into faithful 3D scenes. Experimental results reveal that state-of-the-art VLMs struggle significantly on SIRI-Bench, underscoring the challenge of structural spatial reasoning. We hope that our study will bring researchers' attention to spatially grounded reasoning and advance VLMs in visual problem-solving.
>
---
#### [replaced 064] Mema: Memory-Augmented Adapter for Enhanced Vision-Language Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.00655](https://arxiv.org/pdf/2603.00655)**

> **作者:** Ying Liu; Yudong Han; Kean Shi; Liyuan Pan
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved remarkable performance by aligning pretrained visual representations with the linguistic knowledge embedded in Large Language Models (LLMs). However, existing approaches typically rely on final-layer visual features or learnable multi-layer fusion, which often fail to sufficiently exploit hierarchical visual cues without explicit cross-layer interaction design. In this work, we propose a Memory-Augmented Adapter (Mema) within the vision encoder. Specifically, Mema maintains a stateful memory that accumulates hierarchical visual representations across layers, with its evolution conditioned on both query embeddings and step-wise visual features. A portion of this memory is selectively injected into token representations via a feedback mechanism, thereby mitigating the attenuation of fine-grained visual cues from shallow layers. Designed as a lightweight and plug-and-play module, Mema integrates seamlessly into pretrained vision encoders without modifying the vanilla backbone architecture. Only a minimal set of additional parameters requires training, enabling adaptive visual feature refinement while reducing training overhead. Extensive experiments across multiple benchmarks demonstrate that Mema consistently improves performance, validating its effectiveness in complex multimodal reasoning tasks. The code have been released at this https URL.
>
---
#### [replaced 065] Lightweight Low-Light Image Enhancement via Distribution-Normalizing Preprocessing and Depthwise U-Net
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.11071](https://arxiv.org/pdf/2604.11071)**

> **作者:** Shimon Murai; Teppei Kurita; Ryuta Satoh; Yusuke Moriuchi
>
> **备注:** Technical report for the NTIRE 2026 Efficient Low-Light Image Enhancement Challenge (CVPR 2026 Workshops), 4th place solution
>
> **摘要:** We present a lightweight two-stage framework for low-light image enhancement (LLIE) that achieves competitive perceptual quality with significantly fewer parameters than existing methods. Our approach combines frozen algorithm-based preprocessing with a compact U-Net built entirely from depthwise-separable convolutions. The preprocessing normalizes the input distribution by providing complementary brightness-corrected views, enabling the trainable network to focus on residual color correction. Our method achieved 4th place in the CVPR 2026 NTIRE Efficient Low-Light Image Enhancement Challenge. We further provide extended benchmarks and ablations to demonstrate the general effectiveness of our methods.
>
---
#### [replaced 066] Beyond Reconstruction: Reconstruction-to-Vector Diffusion for Hyperspectral Anomaly Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.11390](https://arxiv.org/pdf/2604.11390)**

> **作者:** Jijun Xiang; Tao Wang; Jiayi Wang; Pengxiang Wang; Cheng Chen; Nian Wang
>
> **摘要:** While Hyperspectral Anomaly Detection (HAD) excels at identifying sparse targets in complex scenes, existing models remain trapped in a scalar "reconstruction-as-endpoint" paradigm. This reliance on ambiguous scalar residuals consistently triggers sub-pixel anomaly vanishing during spatial downsampling, alongside severe confirmation bias when unpurified anomalies corrupt training weights. In this paper, we propose Reconstruction-to-Vector Diffusion (R2VD), which fundamentally redefines reconstruction as a manifold purification origin to establish a novel residual-guided generative dynamics paradigm. Our framework introduces a four-stage pipeline: (1) a Physical Prior Extraction (PPE) stage that mitigates early confirmation bias via dual-stream statistical guidance; (2) a Guided Manifold Purification (GMP) stage utilizing an OmniContext Autoencoder (OCA) to extract purified residual maps while preserving fragile sub-pixel topologies; (3) a Residual Score Modeling (RSM) stage where a Diffusion Transformer (DiT), guarded by a Physical Spectral Firewall (PSF), effectively isolates cross-spectral leakage; and (4) a Vector Dynamics Inference (VDI) stage that robustly decouples targets from backgrounds by evaluating high-dimensional vector interference patterns instead of conventional scalar errors. Comprehensive evaluations on eight datasets confirm that R2VD establishes a new state-of-the-art, delivering exceptional target detectability and background suppression. The code is available at this https URL.
>
---
#### [replaced 067] Towards Interpretable Foundation Models for Retinal Fundus Images
- **分类: cs.CV; cs.LG; stat.CO**

- **链接: [https://arxiv.org/pdf/2603.18846](https://arxiv.org/pdf/2603.18846)**

> **作者:** Samuel Ofosu Mensah; Camila Roa; Kerol Djoumessi; Philipp Berens
>
> **备注:** 11 pages, 3 figures, 2 tables, submitted to MICCAI 2026
>
> **摘要:** Foundation models are used to extract transferable representations from large amounts of unlabeled data, typically via self-supervised learning (SSL). However, many of these models rely on architectures that offer limited interpretability, which is a critical issue in high-stakes domains such as medical imaging. We propose Dual-IFM, a foundation model that is interpretable-by-design in two ways: First, it provides local interpretability for individual images through class evidence maps that are faithful to the decision-making process. Second, it provides global interpretability for entire datasets through a 2D projection layer that allows for direct visualization of the model's representation space. We trained our model on over 800,000 color fundus photography from various sources to learn generalizable, interpretable representations for different downstream tasks. Our results show that our model reaches a performance range similar to that of state-of-the-art foundation models with up to $16\times$ the number of parameters, while providing interpretable predictions on out-of-distribution data. Our results suggest that large-scale SSL pretraining paired with inherent interpretability can lead to robust representations for retinal imaging.
>
---
#### [replaced 068] Latent Chain-of-Thought World Modeling for End-to-End Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在提升复杂场景下的驾驶性能与安全。通过引入隐式链式思维（Latent-CoT）模型，结合动作与世界模型进行推理决策，提高推理效率与轨迹质量。**

- **链接: [https://arxiv.org/pdf/2512.10226](https://arxiv.org/pdf/2512.10226)**

> **作者:** Shuhan Tan; Kashyap Chitta; Yuxiao Chen; Ran Tian; Yurong You; Yan Wang; Wenjie Luo; Yulong Cao; Philipp Krahenbuhl; Marco Pavone; Boris Ivanovic
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Recent Vision-Language-Action (VLA) models for autonomous driving explore inference-time reasoning as a way to improve driving performance and safety in challenging scenarios. Most prior work uses natural language to express chain-of-thought (CoT) reasoning before producing driving actions. However, text may not be the most efficient representation for reasoning. In this work, we present Latent-CoT-Drive (LCDrive): a model that expresses CoT in a latent language that captures possible outcomes of the driving actions being considered. Our approach unifies CoT reasoning and decision making by representing both in an action-aligned latent space. Instead of natural language, the model reasons by interleaving (1) action-proposal tokens, which use the same vocabulary as the model's output actions; and (2) world model tokens, which are grounded in a learned latent world model and express future outcomes of these actions. We cold start latent CoT by supervising the model's action proposals and world model tokens based on ground-truth future rollouts of the scene. We then post-train with closed-loop reinforcement learning to strengthen reasoning capabilities. On a large-scale end-to-end driving benchmark, LCDrive achieves faster inference, better trajectory quality, and larger improvements from interactive reinforcement learning compared to both non-reasoning and text-reasoning baselines.
>
---
#### [replaced 069] MorphDistill: Distilling Unified Morphological Knowledge from Pathology Foundation Models for Colorectal Cancer Survival Prediction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.06390](https://arxiv.org/pdf/2604.06390)**

> **作者:** Hikmat Khan; Usama Sajjad; Metin N. Gurcan; Anil Parwani; Wendy L. Frankel; Wei Chen; Muhammad Khalid Khan Niazi
>
> **摘要:** Background: Colorectal cancer (CRC) remains a leading cause of cancer-related mortality worldwide. Accurate survival prediction is essential for treatment stratification, yet existing pathology foundation models often overlook organ-specific features critical for CRC prognostication. Methods: We propose MorphDistill, a two-stage framework that distills complementary knowledge from multiple pathology foundation models into a compact CRC-specific encoder. In Stage I, a student encoder is trained using dimension-agnostic multi-teacher relational distillation with supervised contrastive regularization on large-scale colorectal datasets. This preserves inter-sample relationships from ten foundation models without explicit feature alignment. In Stage II, the encoder extracts patch-level features from whole-slide images, which are aggregated via attention-based multiple instance learning to predict five-year survival. Results: On the Alliance/CALGB 89803 cohort (n=424, stage III CRC), MorphDistill achieves an AUC of 0.68 (SD 0.08), an approximately 8% relative improvement over the strongest baseline (AUC 0.63). It also attains a C-index of 0.661 and a hazard ratio of 2.52 (95% CI: 1.73-3.65), outperforming all baselines. On an external TCGA cohort (n=562), it achieves a C-index of 0.628, demonstrating strong generalization across datasets and robustness across clinical subgroups. Conclusion: MorphDistill enables task-specific representation learning by integrating knowledge from multiple foundation models into a unified encoder. This approach provides an efficient strategy for prognostic modeling in computational pathology, with potential for broader oncology applications. Further validation across additional cohorts and disease stages is warranted.
>
---
#### [replaced 070] Bootstrapping Video Semantic Segmentation Model via Distillation-assisted Test-Time Adaptation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.10950](https://arxiv.org/pdf/2604.10950)**

> **作者:** Jihun Kim; Hoyong Kwon; Hyeokjun Kweon; Kuk-Jin Yoon
>
> **备注:** accepted at CVPR 2026
>
> **摘要:** Fully supervised Video Semantic Segmentation (VSS) relies heavily on densely annotated video data, limiting practical applicability. Alternatively, applying pre-trained Image Semantic Segmentation (ISS) models frame-by-frame avoids annotation costs but ignores crucial temporal coherence. Recent foundation models such as SAM2 enable high-quality mask propagation yet remain impractical for direct VSS due to limited semantic understanding and computational overhead. In this paper, we propose DiTTA (Distillation-assisted Test-Time Adaptation), a novel framework that converts an ISS model into a temporally-aware VSS model through efficient test-time adaptation (TTA), without annotated videos. DiTTA distills SAM2's temporal segmentation knowledge into the ISS model during a brief, single-pass initialization phase, complemented by a lightweight temporal fusion module to aggregate cross-frame context. Crucially, DiTTA achieves robust generalization even when adapting with highly limited partial video snippets (e.g., initial 10%), significantly outperforming zero-shot refinement approaches that repeatedly invoke SAM2 during inference. Extensive experiments on VSPW and Cityscapes demonstrate DiTTA's effectiveness, achieving competitive or superior performance relative to fully-supervised VSS methods, thus providing a practical and annotation-free solution for real-world VSS tasks.
>
---
#### [replaced 071] Navigating the Accuracy-Size Trade-Off with Flexible Model Merging
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.23209](https://arxiv.org/pdf/2505.23209)**

> **作者:** Akash Dhasade; Divyansh Jhunjhunwala; Milos Vujasinovic; Gauri Joshi; Anne-Marie Kermarrec
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Model merging has emerged as an efficient method to combine multiple single-task fine-tuned models. The merged model can enjoy multi-task capabilities without expensive training. While promising, merging into a single model often suffers from an accuracy gap with respect to the fine-tuned models. On the other hand, deploying all individual fine-tuned models incurs high storage costs. We propose FlexMerge, a novel data-free model merging framework that: (a) flexibly generates merged models of varying sizes, spanning the full spectrum from a single merged model to retaining all fine-tuned models; and (b) supports multiple merging algorithms in a unified framework. Using FlexMerge, we systematically characterize the accuracy-size trade-off of different algorithms. Our study reveals two key findings: first, even modestly larger merged models can yield steep accuracy gains (up to 13.5% when just doubling the size); second, algorithm rankings are not consistent as size increases, with some methods overtaking others beyond the one-model regime. These results uncover a new design dimension for model merging: developing and comparing algorithms across the full spectrum of sizes rather than only at the single-model limit. Extensive experiments on vision and NLP benchmarks, with up to 30 tasks, confirm the generality and practicality of FlexMerge.
>
---
#### [replaced 072] SAM3-I: Segment Anything with Instructions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.04585](https://arxiv.org/pdf/2512.04585)**

> **作者:** Jingjing Li; Yue Feng; Yuchen Guo; Jincai Huang; Wei Ji; Qi Bi; Yongri Piao; Miao Zhang; Xiaoqi Zhao; Qiang Chen; Shihao Zou; Huchuan Lu; Li Cheng
>
> **摘要:** Segment Anything Model 3 (SAM3) advances open-vocabulary segmentation through promptable concept segmentation, enabling users to segment all instances associated with a given concept using short noun-phrase (NP) prompts. While effective for concept-level grounding, real-world interactions often involve far richer natural-language instructions that combine attributes, relations, actions, states, or implicit reasoning. Currently, SAM3 relies on external multi-modal agents to convert complex instructions into NPs and conducts iterative mask filtering, leading to coarse representations and limited instance specificity. In this work, we present SAM3-I, an instruction-following extension of the SAM family that unifies concept-level grounding and instruction-level reasoning within a single segmentation framework. Built upon SAM3, SAM3-I introduces an instruction-aware cascaded adaptation mechanism with dedicated alignment losses that progressively aligns expressive instruction semantics with SAM3's vision-language representations, enabling direct interpretation of natural-language instructions while preserving its strong concept recall ability. To enable instruction-following learning, we introduce HMPL-Instruct, a large-scale instruction-centric dataset that systematically covers hierarchical instruction semantics and diverse target granularities. Experiments demonstrate that SAM3-I achieves appealing performance across referring and reasoning-based segmentation, showing that SAM3 can be effectively extended to follow complex natural-language instructions without sacrificing its original concept-driven strengths. Code and dataset are available at this https URL.
>
---
#### [replaced 073] BLaDA: Bridging Language to Functional Dexterous Actions within 3DGS Fields
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出BLaDA框架，解决功能性灵巧操作任务中的语义-姿态耦合问题，通过语言解析、三维定位和姿态生成实现零样本功能操作。**

- **链接: [https://arxiv.org/pdf/2604.08410](https://arxiv.org/pdf/2604.08410)**

> **作者:** Fan Yang; Wenrui Chen; Guorun Yan; Ruize Liao; Wanjun Jia; Dongsheng Luo; Jiacheng Lin; Kailun Yang; Zhiyong Li; Yaonan Wang
>
> **备注:** Code will be publicly available at this https URL
>
> **摘要:** In unstructured environments, functional dexterous grasping calls for the tight integration of semantic understanding, precise 3D functional localization, and physically interpretable execution. Modular hierarchical methods are more controllable and interpretable than end-to-end VLA approaches, but existing ones still rely on predefined affordance labels and lack the tight semantic--pose coupling needed for functional dexterous manipulation. To address this, we propose BLaDA (Bridging Language to Dexterous Actions in 3DGS fields), an interpretable zero-shot framework that grounds open-vocabulary instructions as perceptual and control constraints for functional dexterous manipulation. BLaDA establishes an interpretable reasoning chain by first parsing natural language into a structured sextuple of manipulation constraints via a Knowledge-guided Language Parsing (KLP) module. To achieve pose-consistent spatial reasoning, we introduce the Triangular Functional Point Localization (TriLocation) module, which utilizes 3D Gaussian Splatting as a continuous scene representation and identifies functional regions under triangular geometric constraints. Finally, the 3D Keypoint Grasp Matrix Transformation Execution (KGT3D+) module decodes these semantic-geometric constraints into physically plausible wrist poses and finger-level commands. Extensive experiments on complex benchmarks demonstrate that BLaDA significantly outperforms existing methods in both affordance grounding precision and the success rate of functional manipulation across diverse categories and tasks. Code will be publicly available at this https URL.
>
---
#### [replaced 074] FaCT: Faithful Concept Traces for Explaining Neural Network Decisions
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.25512](https://arxiv.org/pdf/2510.25512)**

> **作者:** Amin Parchami-Araghi; Sukrut Rao; Jonas Fischer; Bernt Schiele
>
> **备注:** 35 pages, 23 figures, 2 tables, Neural Information Processing Systems (NeurIPS) 2025; Code is available at this https URL
>
> **摘要:** Deep networks have shown remarkable performance across a wide range of tasks, yet getting a global concept-level understanding of how they function remains a key challenge. Many post-hoc concept-based approaches have been introduced to understand their workings, yet they are not always faithful to the model. Further, they make restrictive assumptions on the concepts a model learns, such as class-specificity, small spatial extent, or alignment to human expectations. In this work, we put emphasis on the faithfulness of such concept-based explanations and propose a new model with model-inherent mechanistic concept-explanations. Our concepts are shared across classes and, from any layer, their contribution to the logit and their input-visualization can be faithfully traced. We also leverage foundation models to propose a new concept-consistency metric, C$^2$-Score, that can be used to evaluate concept-based methods. We show that, compared to prior work, our concepts are quantitatively more consistent and users find our concepts to be more interpretable, all while retaining competitive ImageNet performance.
>
---
#### [replaced 075] Geometry Aware Cross-Modal Alignment for Light Field-LiDAR Semantic Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.06687](https://arxiv.org/pdf/2510.06687)**

> **作者:** Jie Luo; Yuxuan Jiang; Xin Jin; Mingyu Liu; Yihui Fan
>
> **摘要:** Semantic segmentation serves as a cornerstone of scene understanding in autonomous driving but continues to face significant challenges under complex conditions such as occlusion. Light field and LiDAR modalities provide complementary visual and spatial cues that are beneficial for robust perception; however, their effective integration is hindered by limited viewpoint diversity and inherent modality discrepancies. To address these challenges, the first multimodal semantic segmentation dataset integrating light field data and point cloud data is proposed. Based on this dataset, we proposed a multi-modal light field point-cloud fusion segmentation network(Mlpfseg), incorporating feature completion and depth perception to segment both camera images and LiDAR point clouds simultaneously. The feature completion module addresses the density mismatch between point clouds and image pixels by performing differential reconstruction of point-cloud feature maps, enhancing the fusion of these modalities. The depth perception module improves the segmentation of occluded objects by reinforcing attention scores for better occlusion awareness. Our method outperforms image-only segmentation by 1.71 Mean Intersection over Union(mIoU) and point cloud-only segmentation by 2.38 mIoU, demonstrating its effectiveness.
>
---
#### [replaced 076] RPG-SAM: Reliability-Weighted Prototypes and Geometric Adaptive Threshold Selection for Training-Free One-Shot Polyp Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.07436](https://arxiv.org/pdf/2603.07436)**

> **作者:** Weikun Lin; Yunhao Bai; Yan Wang
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Training-free one-shot segmentation offers a scalable alternative to expert annotations where knowledge is often transferred from support images and foundation models. But existing methods often treat all pixels in support images and query response intensities models in a homogeneous way. They ignore the regional heterogeity in support images and response heterogeity in this http URL resolve this, we propose RPG-SAM, a framework that systematically tackles these heterogeneity gaps. Specifically, to address regional heterogeneity, we introduce Reliability-Weighted Prototype Mining (RWPM) to prioritize high-fidelity support features while utilizing background anchors as contrastive references for noise suppression. To address response heterogeneity, we develop Geometric Adaptive Selection (GAS) to dynamically recalibrate binarization thresholds by evaluating the morphological consensus of candidates. Finally, an iterative refinement loop method is designed to polishes anatomical boundaries. By accounting for multi-layered information heterogeneity, RPG-SAM achieves a 5.56\% mIoU improvement on the Kvasir dataset. Code will be released.
>
---
#### [replaced 077] On Efficient Variants of Segment Anything Model: A Survey
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2410.04960](https://arxiv.org/pdf/2410.04960)**

> **作者:** Xiaorui Sun; Jun Liu; Heng Tao Shen; Xiaofeng Zhu; Ping Hu
>
> **备注:** IJCV
>
> **摘要:** The Segment Anything Model (SAM) is a foundational model for image segmentation tasks, known for its strong generalization across diverse applications. However, its impressive performance comes with significant computational and resource demands, making it challenging to deploy in resource-limited environments such as edge devices. To address this, a variety of SAM variants have been proposed to enhance efficiency while keeping accuracy. This survey provides the first comprehensive review of these efficient SAM variants. We begin by exploring the motivations driving this research. We then present core techniques used in SAM and model acceleration. This is followed by a detailed exploration of SAM acceleration strategies, categorized by approach, and a discussion of several future research directions. Finally, we offer a unified and extensive evaluation of these methods across various hardware, assessing their efficiency and accuracy on representative benchmarks, and providing a clear comparison of their overall performance.
>
---
#### [replaced 078] MVOS_HSI: A Python Library for Preprocessing Agricultural Crop Hyperspectral Data
- **分类: cs.SE; cs.CV**

- **链接: [https://arxiv.org/pdf/2604.07656](https://arxiv.org/pdf/2604.07656)**

> **作者:** Rishik Aggarwal; Krisha Joshi; Pappu Kumar Yadav; Jianwei Qin; Thomas F. Burks; Moon S. Kim
>
> **备注:** 11 pages
>
> **摘要:** Hyperspectral imaging (HSI) allows researchers to study plant traits non-destructively. By capturing hundreds of narrow spectral bands per pixel, it reveals details about plant biochemistry and stress that standard cameras miss. However, processing this data is often challenging. Many labs still rely on loosely organized collections of lab-specific MATLAB or Python scripts, which makes workflows difficult to share and results difficult to reproduce. MVOS_HSI is an open-source Python library that provides an end-to-end workflow for processing leaf-level HSI data. The software handles everything from calibrating raw ENVI files to detecting and clipping individual leaves based on multiple vegetation indices (NDVI, CIRedEdge and GCI). It also includes tools for data augmentation to create training-time variations for machine learning and utilities to visualize spectral profiles. MVOS_HSI can be used as an importable Python library or run directly from the command line. The code and documentation are available on GitHub. By consolidating these common tasks into a single package, MVOS_HSI helps researchers produce consistent and reproducible results in plant phenotyping
>
---
#### [replaced 079] Prompt Evolution for Generative AI: A Classifier-Guided Approach
- **分类: cs.LG; cs.AI; cs.CV; cs.NE**

- **链接: [https://arxiv.org/pdf/2305.16347](https://arxiv.org/pdf/2305.16347)**

> **作者:** Melvin Wong; Yew-Soon Ong; Abhishek Gupta; Kavitesh K. Bali; Caishun Chen
>
> **备注:** This work is published in the Proceedings of the IEEE Conference on Artificial Intelligence (CAI 2023). IEEE copyrights applies
>
> **摘要:** Synthesis of digital artifacts conditioned on user prompts has become an important paradigm facilitating an explosion of use cases with generative AI. However, such models often fail to connect the generated outputs and desired target concepts/preferences implied by the prompts. Current research addressing this limitation has largely focused on enhancing the prompts before output generation or improving the model's performance up front. In contrast, this paper conceptualizes prompt evolution, imparting evolutionary selection pressure and variation during the generative process to produce multiple outputs that satisfy the target concepts/preferences better. We propose a multi-objective instantiation of this broader idea that uses a multi-label image classifier-guided approach. The predicted labels from the classifiers serve as multiple objectives to optimize, with the aim of producing diversified images that meet user preferences. A novelty of our evolutionary algorithm is that the pre-trained generative model gives us implicit mutation operations, leveraging the model's stochastic generative capability to automate the creation of Pareto-optimized images more faithful to user preferences.
>
---
#### [replaced 080] GroupKAN: Efficient Kolmogorov-Arnold Networks via Grouped Spline Modeling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.05477](https://arxiv.org/pdf/2511.05477)**

> **作者:** Guojie Li; Tianyi Liu; Anwar P.P. Abdul Majeed; Muhammad Ateeq; Anh Nguyen; Fan Zhang
>
> **摘要:** Medical image segmentation demands models that achieve high accuracy while maintaining computational efficiency and clinical interpretability. While recent Kolmogorov-Arnold Networks (KANs) offer powerful adaptive non-linearities, their full-channel spline transformations incur a quadratic parameter growth of $\mathcal{O}(C^{2}(G+k))$ with respect to the channel dimension $C$, where $G$ and $k$ denote the number of grid intervals and spline polynomial order, respectively. Moreover, unconstrained spline mappings lack structural constraints, leading to excessive functional freedom, which may cause overfitting under limited medical annotations. To address these challenges, we propose GroupKAN (Grouped Kolmogorov-Arnold Networks), an efficient architecture driven by group-structured spline modeling. Specifically, we introduce: (1) Grouped KAN Transform (GKT), which restricts spline interactions to intra-group channel mappings across $g$ groups, effectively reducing the spline-induced quadratic expansion to \textbf{$\mathcal{O}(C^2(\frac{G+k}{g} + 1))$}, thereby significantly lowering the effective quadratic coefficient; and (2) Grouped KAN Activation (GKA), which applies shared spline functions within each group to enable efficient token-wise non-linearities. By imposing structured constraints on channel interactions, GroupKAN achieves a substantial reduction in parameter redundancy without sacrificing expressive this http URL evaluations on three medical benchmarks (BUSI, GlaS, and CVC) demonstrate that GroupKAN achieves an average IoU of 79.80\%, outperforming the strong U-KAN baseline by +1.11\% while requiring only 47.6\% of the parameters (3.02M vs. 6.35M). Qualitative results further reveal that GroupKAN produces sharply localized activation maps that better align with the ground truth than MLPs and KANs, significantly enhancing clinical interpretability.
>
---
#### [replaced 081] Vision Transformers Need More Than Registers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.22394](https://arxiv.org/pdf/2602.22394)**

> **作者:** Cheng Shi; Yizhou Yu; Sibei Yang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Vision Transformers (ViTs), when pre-trained on large-scale data, provide general-purpose representations for diverse downstream tasks. However, artifacts in ViTs are widely observed across different supervision paradigms and downstream tasks. Through systematic analysis of artifacts in ViTs, we find that their fundamental mechanisms have yet to be sufficiently elucidated. In this paper, through systematic analysis, we conclude that these artifacts originate from a lazy aggregation behavior: ViT uses semantically irrelevant background patches as shortcuts to represent global semantics, driven by global attention and Coarse-grained semantic supervision. Our solution selectively integrates patch features into the CLS token, reducing the influence of background-dominated shortcuts and consistently improving performance across 12 benchmarks under label-, text-, and self-supervision. We hope this work offers a new perspective on ViT behavior.
>
---
#### [replaced 082] Energy-Regularized Spatial Masking: A Novel Approach to Enhancing Robustness and Interpretability in Vision Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.06893](https://arxiv.org/pdf/2604.06893)**

> **作者:** Tom Devynck; Bilal Faye; Djamel Bouchaffra; Nadjib Lazaar; Hanane Azzag; Mustapha Lebbah
>
> **摘要:** Deep convolutional neural networks achieve remarkable performance by exhaustively processing dense spatial feature maps, yet this brute-force strategy introduces significant computational redundancy and encourages reliance on spurious background correlations. As a result, modern vision models remain brittle and difficult to interpret. We propose Energy-Regularized Spatial Masking (ERSM), a novel framework that reformulates feature selection as a differentiable energy minimization problem. By embedding a lightweight Energy-Mask Layer inside standard convolutional backbones, each visual token is assigned a scalar energy composed of two competing forces: an intrinsic Unary importance cost and a Pairwise spatial coherence penalty. Unlike prior pruning methods that enforce rigid sparsity budgets or rely on heuristic importance scores, ERSM allows the network to autonomously discover an optimal information-density equilibrium tailored to each input. We validate ERSM on convolutional architectures and demonstrate that it produces emergent sparsity, improved robustness to structured occlusion, and highly interpretable spatial masks, while preserving classification accuracy. Furthermore, we show that the learned energy ranking significantly outperforms magnitude-based pruning in deletion-based robustness tests, revealing ERSM as an intrinsic denoising mechanism that isolates semantic object regions without pixel-level supervision.
>
---
#### [replaced 083] One View Is Enough! Monocular Training for In-the-Wild Novel View Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.23488](https://arxiv.org/pdf/2603.23488)**

> **作者:** Adrien Ramanana Rahary; Nicolas Dufour; Patrick Perez; David Picard
>
> **备注:** 36 pages, 17 figures
>
> **摘要:** Monocular novel-view synthesis has long required multi-view image pairs for supervision, limiting training data scale and diversity. We argue it is not necessary: one view is enough. We present OVIE, trained entirely on unpaired internet images. We leverage a monocular depth estimator as a geometric scaffold at training time: we lift a source image into 3D, apply a sampled camera transformation, and project to obtain a pseudo-target view. To handle disocclusions, we introduce a masked training formulation that restricts geometric, perceptual, and textural losses to valid regions, enabling training on 30 million uncurated images. At inference, OVIE is geometry-free, requiring no depth estimator or 3D representation. Trained exclusively on in-the-wild images, OVIE outperforms prior methods in a zero-shot setting, while being 600x faster than the second-best baseline. Code and models are publicly available at this https URL.
>
---
#### [replaced 084] Retrievals Can Be Detrimental: Unveiling the Backdoor Vulnerability of Retrieval-Augmented Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.13340](https://arxiv.org/pdf/2501.13340)**

> **作者:** Hao Fang; Xiaohang Sui; Hongyao Yu; Kuofeng Gao; Jiawei Kong; Sijin Yu; Bin Chen; Shu-Tao Xia
>
> **备注:** Accepted by ACL-2026
>
> **摘要:** Diffusion models (DMs) have recently demonstrated remarkable generation capability. However, their training generally requires huge computational resources and large-scale datasets. To solve these, recent studies empower DMs with the advanced Retrieval-Augmented Generation (RAG) technique and propose retrieval-augmented diffusion models (RDMs). By incorporating rich knowledge from an auxiliary database, RAG enhances diffusion models' generation and generalization ability while significantly reducing model parameters. Despite the great success, RAG may introduce novel security issues that warrant further investigation. In this paper, we reveal that the RDM is susceptible to backdoor attacks by proposing a multimodal contrastive attack approach named BadRDM. Our framework fully considers RAG's characteristics and is devised to manipulate the retrieved items for given text triggers, thereby further controlling the generated contents. Specifically, we first insert a tiny portion of images into the retrieval database as target toxicity surrogates. Subsequently, a malicious variant of contrastive learning is adopted to inject backdoors into the retriever, which builds shortcuts from triggers to the toxicity surrogates. Furthermore, we enhance the attacks through novel entropy-based selection and generative augmentation strategies that can derive better toxicity surrogates. Extensive experiments on two mainstream tasks demonstrate the proposed BadRDM achieves outstanding attack effects while preserving the model's benign utility.
>
---
#### [replaced 085] Intelligent bear deterrence system based on computer vision: Reducing human-bear conflicts in remote areas
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.23178](https://arxiv.org/pdf/2503.23178)**

> **作者:** Pengyu Chen; Teng Fei; John A. Kupfer; Yunyan Du; Jiawei Yi; Yi Li
>
> **摘要:** Human-bear conflicts on the Tibetan Plateau threaten both local livelihoods and the conservation of Tibetan brown bears (Ursus arctos pruinosus). To address this challenge, we developed a low-power, network-independent deterrence system that combines computer vision with Internet of Things (IoT) hardware. The system integrates a YOLOv5-MobileNet detection model deployed on a low-power edge artificial intelligence (AI) board with a solar-powered bear spray device. We compiled a data set of 1,243 wildlife images (including 795 bears with 100 infrared captures for nighttime detection, plus other common objects and animals such as mastiffs, yaks, humans, and vehicles), from which 80% were used for training and 20% for validation. Validation showed robust performance (mean average precision = 91.4%, recall = 93.6%). In 100 controlled activation tests involving simulated approaches by bears, humans, and other animals, the spray deployed within 0.2 seconds of detection with 97.2% accuracy, confirming timely and reliable responses. A 30-day field trial in Zadoi County, Qinghai Province, China, recorded 3 successful deterrence events without false activations. By using energy-efficient components and ensuring continuous and stable system operation, this solution provides a practical, sustainable, and scalable approach to mitigating human-bear conflicts, effectively enhancing human safety and bear conservation in remote areas without network or grid coverage.
>
---
#### [replaced 086] MedVeriSeg: Teaching MLLM-Based Medical Segmentation Models to Verify Query Validity Without Extra Training
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.10242](https://arxiv.org/pdf/2604.10242)**

> **作者:** Ziqian Lu; Qinyue Tong; Jun Liu; Yunlong Yu
>
> **备注:** 7 pages, 4 figures; the paper is under consideration at Pattern Recognition Letters
>
> **摘要:** Despite recent advances in MLLM-based medical image segmentation, existing LISA-like methods cannot reliably reject false queries and often produce hallucinated segmentation masks for absent targets. This limitation reduces practical reliability in both medical education and clinical use. In this work, we propose MedVeriSeg, a training-free verification framework that equips LISA-like medical segmentation models with the ability to identify and reject false queries which contain non-existent targets. Our key observation is that the similarity map between the [SEG] token feature and MLLM image features exhibits markedly different distribution patterns for true and false queries. Based on this, we introduce a Similarity Response Quality Scoring Module that characterizes the similarity map from three aspects: strength, compactness, and purity, producing an initial target-existence prediction. We further incorporate qualitative visual evidence by using GPT-4o to jointly assess the similarity heatmap and the results of Similarity Response Quality Scoring Module for final verification. Experiments on a small-scale benchmark constructed from SA-Med2D-20M show that MedVeriSeg effectively rejects false-query segmentation requests while maintaining reliable recognition of true queries.
>
---
#### [replaced 087] Automatic Road Subsurface Distress Recognition from Ground Penetrating Radar Images using Deep Learning-based Cross-verification
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.11081](https://arxiv.org/pdf/2507.11081)**

> **作者:** Chang Peng; Bao Yang; Meiqi Li; Ge Zhang; Hui Sun; Zhenyu Jiang
>
> **摘要:** Ground penetrating radar (GPR) has become a rapid and non-destructive solution for road subsurface distress (RSD) detection. However, recognizing RSD from GPR images is labor-intensive and heavily relies on the expertise of inspectors. Deep learning-based automatic RSD recognition, though ameliorating the burden of data processing, suffers from insufficient capability to recognize defects. In this study, a novel cross-verification strategy was proposed to fully exploit the complementary abilities of region proposal networks in object recognition from different views of GPR images. Following this strategy, three YOLO-based models were used to detect the RSD (voids and loose structures) and manholes. Each model was trained with a specific view of 3D GPR dataset, which contains rigorously validated 2134 samples of diverse types obtained through field scanning. The cross-verification strategy achieves outstanding accuracy with a recall of over 98.6% in the tests using real field-scanning data. Field tests also show that deep learning-based automatic RSD recognition can reduce the human labor of inspection by around 90%.
>
---
#### [replaced 088] Turbo-DDCM: Fast and Flexible Zero-Shot Diffusion-Based Image Compression
- **分类: eess.IV; cs.AI; cs.CV; eess.SP; stat.ML**

- **链接: [https://arxiv.org/pdf/2511.06424](https://arxiv.org/pdf/2511.06424)**

> **作者:** Amit Vaisman; Guy Ohayon; Hila Manor; Michael Elad; Tomer Michaeli
>
> **备注:** ICLR 2026. Code is available at this https URL
>
> **摘要:** While zero-shot diffusion-based compression methods have seen significant progress in recent years, they remain notoriously slow and computationally demanding. This paper presents an efficient zero-shot diffusion-based compression method that runs substantially faster than existing methods, while maintaining performance that is on par with the state-of-the-art techniques. Our method builds upon the recently proposed Denoising Diffusion Codebook Models (DDCMs) compression scheme. Specifically, DDCM compresses an image by sequentially choosing the diffusion noise vectors from reproducible random codebooks, guiding the denoiser's output to reconstruct the target image. We modify this framework with Turbo-DDCM, which efficiently combines a large number of noise vectors at each denoising step, thereby significantly reducing the number of required denoising operations. This modification is also coupled with an improved encoding protocol. Furthermore, we introduce two flexible variants of Turbo-DDCM, a priority-aware variant that prioritizes user-specified regions and a distortion-controlled variant that compresses an image based on a target PSNR rather than a target BPP. Comprehensive experiments position Turbo-DDCM as a compelling, practical, and flexible image compression scheme.
>
---
#### [replaced 089] INFORM-CT: INtegrating LLMs and VLMs FOR Incidental Findings Management in Abdominal CT
- **分类: cs.LG; cs.AI; cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2512.14732](https://arxiv.org/pdf/2512.14732)**

> **作者:** Idan Tankel; Nir Mazor; Rafi Brada; Christina LeBedis; Guy ben-Yosef
>
> **备注:** Accepted for Spotlight presentation at MIDL 2026
>
> **摘要:** Incidental findings in CT scans, though often benign, can have significant clinical implications and should be reported following established guidelines. Traditional manual inspection by radiologists is time-consuming and variable. This paper proposes a novel framework that leverages large language models (LLMs) and foundational vision-language models (VLMs) in a plan-and-execute agentic approach to improve the efficiency and precision of incidental findings detection, classification, and reporting for abdominal CT scans. Given medical guidelines for abdominal organs, the process of managing incidental findings is automated through a planner-executor framework. The planner, based on LLM, generates Python scripts using predefined base functions, while the executor runs these scripts to perform the necessary checks and detections, via VLMs, segmentation models, and image processing subroutines. We demonstrate the effectiveness of our approach through experiments on a CT abdominal benchmark for three organs, in a fully automatic end-to-end manner. Our results show that the proposed framework outperforms existing pure VLM-based approaches in terms of accuracy and efficiency.
>
---
#### [replaced 090] TempR1: Improving Temporal Understanding of MLLMs via Temporal-Aware Multi-Task Reinforcement Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.03963](https://arxiv.org/pdf/2512.03963)**

> **作者:** Tao Wu; Li Yang; Gen Zhan; Yabin Zhang; Yiting Liao; Junlin Li; Deliang Fu; Li Zhang; Limin Wang
>
> **摘要:** Enhancing the temporal understanding of Multimodal Large Language Models (MLLMs) is essential for advancing long-form video analysis, enabling tasks such as temporal localization, action detection, and time-sensitive question answering. While reinforcement learning (RL) has recently been explored for improving temporal reasoning, existing approaches are often confined to limited task types and data, restricting their generalization across diverse temporal understanding scenarios. To address this challenge, we present TempR1, a temporal-aware multi-task reinforcement learning framework that systematically strengthens MLLMs' temporal comprehension. We curate a multi-task corpus that exposes the model to diverse temporal structures and semantics, and build upon the Group Relative Policy Optimization (GRPO) algorithm to achieve stable and effective cross-task optimization. Specifically, we categorize temporal tasks into three correspondence types between predicted intervals and ground-truth instances, and design tailored localization rewards for each, enabling TempR1 to capture fine-grained temporal dependencies and adapt to different temporal patterns. Extensive experiments demonstrate that TempR1 attains state-of-the-art performance across multiple benchmarks. Moreover, its joint optimization over complementary tasks yields a strong synergistic effect, enhancing both generalization and single-task performance, establishing a scalable and principled paradigm for temporal reasoning in MLLMs.
>
---
#### [replaced 091] Label-Efficient Cross-Modality Generalization for Liver Segmentation in Multi-Phase MRI
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.04705](https://arxiv.org/pdf/2510.04705)**

> **作者:** Quang-Khai Bui-Tran; Minh-Toan Dinh; Thanh-Huy Nguyen; Ba-Thinh Lam; Mai-Anh Vu; Ulas Bagci
>
> **备注:** Accepted at MICCAI 2025 Workshop
>
> **摘要:** Accurate liver segmentation in multi-phase MRI is vital for liver fibrosis assessment, yet labeled data is often scarce and unevenly distributed across imaging modalities and vendor systems. We propose a label-efficient segmentation approach that promotes cross-modality generalization under real-world conditions, where GED4 hepatobiliary-phase annotations are limited, non-contrast sequences (T1WI, T2WI, DWI) are unlabeled, and spatial misalignment and missing phases are common. Our method integrates a foundation-scale 3D segmentation backbone adapted via fine-tuning, co-training with cross pseudo supervision to leverage unlabeled volumes, and a standardized preprocessing pipeline. Without requiring spatial registration, the model learns to generalize across MRI phases and vendors, demonstrating robust segmentation performance in both labeled and unlabeled domains. Our results exhibit the effectiveness of our proposed label-efficient baseline for liver segmentation in multi-phase, multi-vendor MRI and highlight the potential of combining foundation model adaptation with co-training for real-world clinical imaging tasks.
>
---
#### [replaced 092] MedGS: Gaussian Splatting for Multi-Modal 3D Medical Imaging
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.16806](https://arxiv.org/pdf/2509.16806)**

> **作者:** Kacper Marzol; Ignacy Kolton; Weronika Smolak-Dyżewska; Joanna Kaleta; Żaneta Świderska-Chadaj; Marcin Mazur; Mirosław Dziekiewicz; Tomasz Markiewicz; Przemysław Spurek
>
> **摘要:** Endoluminal endoscopic procedures are essential for diagnosing colorectal cancer and other severe conditions in the digestive tract, urogenital system, and airways. 3D reconstruction and novel-view synthesis from endoscopic images are promising tools for enhancing diagnosis. Moreover, integrating physiological deformations and interaction with the endoscope enables the development of simulation tools from real video data. However, constrained camera trajectories and view-dependent lighting create artifacts, leading to inaccurate or overfitted reconstructions. We present MedGS, a novel 3D reconstruction framework leveraging the unique property of endoscopic imaging, where a single light source is closely aligned with the camera. Our method separates light effects from tissue properties. MedGS enhances 3D Gaussian Splatting with a physically based relightable model. We boost the traditional light transport formulation with a specialized MLP capturing complex light-related effects while ensuring reduced artifacts and better generalization across novel views. MedGS achieves superior reconstruction quality compared to baseline methods on both public and in-house datasets. Unlike existing approaches, MedGS enables tissue modifications while preserving a physically accurate response to light, making it closer to real-world clinical use. Repository: this https URL
>
---
#### [replaced 093] SparseWorld-TC: Trajectory-Conditioned Sparse Occupancy World Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22039](https://arxiv.org/pdf/2511.22039)**

> **作者:** Jiayuan Du; Yiming Zhao; Zhenglong Guo; Yong Pan; Wenbo Hou; Zhihui Hao; Kun Zhan; Qijun Chen
>
> **备注:** Accepted by CVPR2026 as an oral
>
> **摘要:** This paper introduces a novel architecture for trajectory-conditioned forecasting of future 3D scene occupancy. In contrast to methods that rely on variational autoencoders (VAEs) to generate discrete occupancy tokens, which inherently limit representational capacity, our approach predicts multi-frame future occupancy in an end-to-end manner directly from raw image features. Inspired by the success of attention-based transformer architectures in foundational vision and language models such as GPT and VGGT, we employ a sparse occupancy representation that bypasses the intermediate bird's eye view (BEV) projection and its explicit geometric priors. This design allows the transformer to capture spatiotemporal dependencies more effectively. By avoiding both the finite-capacity constraint of discrete tokenization and the structural limitations of BEV representations, our method achieves state-of-the-art performance on the nuScenes benchmark for 1-3 second occupancy forecasting, outperforming existing approaches by a significant margin. Furthermore, it demonstrates robust scene dynamics understanding, consistently delivering high accuracy under arbitrary future trajectory conditioning.
>
---
