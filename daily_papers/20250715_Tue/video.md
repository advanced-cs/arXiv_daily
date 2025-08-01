# 计算机视觉 cs.CV

- **最新发布 197 篇**

- **更新 142 篇**

## 最新发布

#### [new 001] Stable Score Distillation
- **分类: cs.CV**

- **简介: 该论文属于文本引导的图像和3D编辑任务，旨在解决现有方法在稳定性、空间控制和编辑强度上的不足。作者提出了Stable Score Distillation（SSD），通过引入基于分类器的跨提示对齐、常数项“null-text”分支和提示增强分支，提升了编辑过程的稳定性和准确性，同时实现了更高效的局部修改与整体一致性保持。**

- **链接: [http://arxiv.org/pdf/2507.09168v1](http://arxiv.org/pdf/2507.09168v1)**

> **作者:** Haiming Zhu; Yangyang Xu; Chenshu Xu; Tingrui Shen; Wenxi Liu; Yong Du; Jun Yu; Shengfeng He
>
> **摘要:** Text-guided image and 3D editing have advanced with diffusion-based models, yet methods like Delta Denoising Score often struggle with stability, spatial control, and editing strength. These limitations stem from reliance on complex auxiliary structures, which introduce conflicting optimization signals and restrict precise, localized edits. We introduce Stable Score Distillation (SSD), a streamlined framework that enhances stability and alignment in the editing process by anchoring a single classifier to the source prompt. Specifically, SSD utilizes Classifier-Free Guidance (CFG) equation to achieves cross-prompt alignment, and introduces a constant term null-text branch to stabilize the optimization process. This approach preserves the original content's structure and ensures that editing trajectories are closely aligned with the source prompt, enabling smooth, prompt-specific modifications while maintaining coherence in surrounding regions. Additionally, SSD incorporates a prompt enhancement branch to boost editing strength, particularly for style transformations. Our method achieves state-of-the-art results in 2D and 3D editing tasks, including NeRF and text-driven style edits, with faster convergence and reduced complexity, providing a robust and efficient solution for text-guided editing.
>
---
#### [new 002] Binomial Self-Compensation: Mechanism and Suppression of Motion Error in Phase-Shifting Profilometry
- **分类: cs.CV**

- **简介: 该论文属于3D扫描任务，旨在解决动态测量中物体运动导致的相位误差问题。作者提出了图像序列二项自补偿（I-BSC）方法，在时间同步前提下通过加权和消除运动误差，并降低计算复杂度，实现快速、高分辨率的三维重建。**

- **链接: [http://arxiv.org/pdf/2507.10009v1](http://arxiv.org/pdf/2507.10009v1)**

> **作者:** Geyou Zhang; Kai Liu; Ce Zhu
>
> **摘要:** Phase shifting profilometry (PSP) is widely used in high-precision 3D scanning due to its high accuracy, robustness, and pixel-wise handling. However, a fundamental assumption of PSP that the object should remain static does not hold in dynamic measurement, making PSP susceptible to object motion. To address this challenge, our proposed solution, phase-sequential binomial self-compensation (P-BSC), sums successive motion-affected phase frames weighted by binomial coefficients. This approach exponentially reduces the motion error in a pixel-wise and frame-wise loopable manner. Despite its efficacy, P-BSC suffers from high computational overhead and error accumulation due to its reliance on multi-frame phase calculations and weighted summations. Inspired by P-BSC, we propose an image-sequential binomial self-compensation (I-BSC) to weight sum the homogeneous fringe images instead of successive phase frames, which generalizes the BSC concept from phase sequences to image sequences. I-BSC computes the arctangent function only once, resolving both limitations in P-BSC. Extensive analysis, simulations, and experiments show that 1) the proposed BSC outperforms existing methods in reducing motion error while achieving a quasi-single-shot frame rate, i.e., depth map frame rate equals to the camera's acquisition rate, enabling 3D reconstruction with high pixel-depth-temporal resolution; 2) compared to P-BSC, our I-BSC reduces the computational complexity by one polynomial order, thereby accelerating the computational frame rate by several to dozen times, while also reaching faster motion error convergence.
>
---
#### [new 003] Boosting Multimodal Learning via Disentangled Gradient Learning
- **分类: cs.CV**

- **简介: 该论文属于多模态学习任务，旨在解决多模态模型中各模态性能劣于单模态模型的问题。作者提出解耦梯度学习框架DGL，通过分离编码器与融合模块的优化过程，减少梯度干扰，提升多模态学习效果。**

- **链接: [http://arxiv.org/pdf/2507.10213v1](http://arxiv.org/pdf/2507.10213v1)**

> **作者:** Shicai Wei; Chunbo Luo; Yang Luo
>
> **备注:** Accepted to ICCV2025
>
> **摘要:** Multimodal learning often encounters the under-optimized problem and may have worse performance than unimodal learning. Existing methods attribute this problem to the imbalanced learning between modalities and rebalance them through gradient modulation. However, they fail to explain why the dominant modality in multimodal models also underperforms that in unimodal learning. In this work, we reveal the optimization conflict between the modality encoder and modality fusion module in multimodal models. Specifically, we prove that the cross-modal fusion in multimodal models decreases the gradient passed back to each modality encoder compared with unimodal models. Consequently, the performance of each modality in the multimodal model is inferior to that in the unimodal model. To this end, we propose a disentangled gradient learning (DGL) framework to decouple the optimization of the modality encoder and modality fusion module in the multimodal model. DGL truncates the gradient back-propagated from the multimodal loss to the modality encoder and replaces it with the gradient from unimodal loss. Besides, DGL removes the gradient back-propagated from the unimodal loss to the modality fusion module. This helps eliminate the gradient interference between the modality encoder and modality fusion module while ensuring their respective optimization processes. Finally, extensive experiments on multiple types of modalities, tasks, and frameworks with dense cross-modal interaction demonstrate the effectiveness and versatility of the proposed DGL. Code is available at \href{https://github.com/shicaiwei123/ICCV2025-GDL}{https://github.com/shicaiwei123/ICCV2025-GDL}
>
---
#### [new 004] ProactiveBench: A Comprehensive Benchmark Evaluating Proactive Interactions in Video Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于多模态对话系统任务，旨在解决现有系统缺乏主动交互能力评估的问题。作者构建了ProactiveBench基准测试，并提出PAUC指标，以更准确评估模型在视频播放中主动响应的时序表现。实验表明PAUC与用户偏好更一致，提升了用户体验评估效果。**

- **链接: [http://arxiv.org/pdf/2507.09313v1](http://arxiv.org/pdf/2507.09313v1)**

> **作者:** Yueqian Wang; Xiaojun Meng; Yifan Wang; Huishuai Zhang; Dongyan Zhao
>
> **摘要:** With the growing research focus on multimodal dialogue systems, the capability for proactive interaction is gradually gaining recognition. As an alternative to conventional turn-by-turn dialogue, users increasingly expect multimodal systems to be more initiative, for example, by autonomously determining the timing of multi-turn responses in real time during video playback. To facilitate progress in this emerging area, we introduce ProactiveBench, the first comprehensive benchmark to evaluate a system's ability to engage in proactive interaction. Since model responses are generated at varying timestamps, we further propose PAUC, the first metric that accounts for the temporal dynamics of model responses. This enables a more accurate evaluation of systems operating in proactive settings. Through extensive benchmarking of various baseline systems on ProactiveBench and a user study of human preferences, we show that PAUC is in better agreement with human preferences than traditional evaluation metrics, which typically only consider the textual content of responses. These findings demonstrate that PAUC provides a more faithful assessment of user experience in proactive interaction scenarios. Project homepage: https://github.com/yellow-binary-tree/ProactiveBench
>
---
#### [new 005] A Transfer Learning-Based Method for Water Body Segmentation in Remote Sensing Imagery: A Case Study of the Zhada Tulin Area
- **分类: cs.CV; cs.LG**

- **简介: 论文属于遥感图像水体分割任务，旨在解决域间差异和小样本导致的模型性能下降问题。研究提出一种基于SegFormer的两阶段迁移学习方法，在西藏札达土林区实验显示，该方法将IoU从25.50%提升至64.84%，有效应对了数据稀缺与复杂环境下的信息提取挑战。**

- **链接: [http://arxiv.org/pdf/2507.10084v1](http://arxiv.org/pdf/2507.10084v1)**

> **作者:** Haonan Chen; Xin Tong
>
> **备注:** 13 pages, 6 figures, 2 tables
>
> **摘要:** To address the prevalent challenges of domain shift and small sample sizes in remote sensing image water body segmentation, this study proposes and validates a two-stage transfer learning strategy based on the SegFormer model. The approach begins by training a foundational segmentation model on a diverse source domain, where it achieves an Intersection over Union (IoU) of 68.80% on its validation set, followed by fine-tuning on data from the distinct target domain. Focusing on the Zhada Tulin area in Tibet -- a region characterized by highly complex topography and spectral features -- the experimental results demonstrate that this strategy significantly boosts the IoU for the water body segmentation task from 25.50% (for direct transfer) to 64.84%. This not only effectively resolves the model performance degradation caused by domain discrepancy but also provides an effective technical paradigm for high-precision thematic information extraction in data-scarce and environmentally unique remote sensing scenarios.
>
---
#### [new 006] Cross-modal Associations in Vision and Language Models: Revisiting the bouba-kiki effect
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉-语言模型（VLMs）是否具备类似人类的跨模态认知能力，聚焦于“bouba-kiki效应”这一经典测试案例。作者以CLIP的两种变体（ResNet和ViT）为对象，采用基于提示的概率评估与Grad-CAM视觉注意力解释方法，发现模型未能稳定表现出该效应，揭示其在跨模态理解上的局限性。**

- **链接: [http://arxiv.org/pdf/2507.10013v1](http://arxiv.org/pdf/2507.10013v1)**

> **作者:** Tom Kouwenhoven; Kiana Shahrasbi; Tessa Verhoef
>
> **摘要:** Recent advances in multimodal models have raised questions about whether vision-and-language models (VLMs) integrate cross-modal information in ways that reflect human cognition. One well-studied test case in this domain is the bouba-kiki effect, where humans reliably associate pseudowords like "bouba" with round shapes and "kiki" with jagged ones. Given the mixed evidence found in prior studies for this effect in VLMs, we present a comprehensive re-evaluation focused on two variants of CLIP, ResNet and Vision Transformer (ViT), given their centrality in many state-of-the-art VLMs. We apply two complementary methods closely modelled after human experiments: a prompt-based evaluation that uses probabilities as model preference, and we use Grad-CAM as a novel way to interpret visual attention in shape-word matching tasks. Our findings show that these models do not consistently exhibit the bouba-kiki effect. While ResNet shows a preference for round shapes, overall performance across both models lacks the expected associations. Moreover, direct comparison with prior human data on the same task shows that the models' responses fall markedly short of the robust, modality-integrated behaviour characteristic of human cognition. These results contribute to the ongoing debate about the extent to which VLMs truly understand cross-modal concepts, highlighting limitations in their internal representations and alignment with human intuitions.
>
---
#### [new 007] Online Micro-gesture Recognition Using Data Augmentation and Spatial-Temporal Attention
- **分类: cs.CV**

- **简介: 该论文属于微手势在线识别任务，旨在精确定位并识别视频中多个未剪辑的微手势实例。针对微手势差异大、难以区分的问题，作者提出了手工数据增强和时空注意力方法，提升了分类与定位效果，最终以38.03的F1分数取得当前最优性能。**

- **链接: [http://arxiv.org/pdf/2507.09512v1](http://arxiv.org/pdf/2507.09512v1)**

> **作者:** Pengyu Liu; Kun Li; Fei Wang; Yanyan Wei; Junhui She; Dan Guo
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** In this paper, we introduce the latest solution developed by our team, HFUT-VUT, for the Micro-gesture Online Recognition track of the IJCAI 2025 MiGA Challenge. The Micro-gesture Online Recognition task is a highly challenging problem that aims to locate the temporal positions and recognize the categories of multiple micro-gesture instances in untrimmed videos. Compared to traditional temporal action detection, this task places greater emphasis on distinguishing between micro-gesture categories and precisely identifying the start and end times of each instance. Moreover, micro-gestures are typically spontaneous human actions, with greater differences than those found in other human actions. To address these challenges, we propose hand-crafted data augmentation and spatial-temporal attention to enhance the model's ability to classify and localize micro-gestures more accurately. Our solution achieved an F1 score of 38.03, outperforming the previous state-of-the-art by 37.9%. As a result, our method ranked first in the Micro-gesture Online Recognition track.
>
---
#### [new 008] Self-supervised Learning on Camera Trap Footage Yields a Strong Universal Face Embedder
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于野生动物个体识别任务，旨在解决手动识别动物个体效率低的问题。通过自监督学习方法DINOv2，利用无人工标注的相机陷阱视频数据训练视觉变换模型，成功实现对黑猩猩面部的高效重识别，效果优于有监督基线方法。**

- **链接: [http://arxiv.org/pdf/2507.10552v1](http://arxiv.org/pdf/2507.10552v1)**

> **作者:** Vladimir Iashin; Horace Lee; Dan Schofield; Andrew Zisserman
>
> **备注:** Accepted for publication. Project page, code and weights: https://www.robots.ox.ac.uk/~vgg/research/ChimpUFE/
>
> **摘要:** Camera traps are revolutionising wildlife monitoring by capturing vast amounts of visual data; however, the manual identification of individual animals remains a significant bottleneck. This study introduces a fully self-supervised approach to learning robust chimpanzee face embeddings from unlabeled camera-trap footage. Leveraging the DINOv2 framework, we train Vision Transformers on automatically mined face crops, eliminating the need for identity labels. Our method demonstrates strong open-set re-identification performance, surpassing supervised baselines on challenging benchmarks such as Bossou, despite utilising no labelled data during training. This work underscores the potential of self-supervised learning in biodiversity monitoring and paves the way for scalable, non-invasive population studies.
>
---
#### [new 009] Efficient Multi-Person Motion Prediction by Lightweight Spatial and Temporal Interactions
- **分类: cs.CV**

- **简介: 该论文属于3D多人运动预测任务，旨在解决个体间复杂交互建模及高计算成本问题。论文提出了一种轻量级模型，通过双分支学习个体与群体的局部和全局表示，并引入跨层级交互模块融合时空信息，同时结合空间距离嵌入以提升交互建模效果，在多个数据集上取得了优异性能并降低了计算开销。**

- **链接: [http://arxiv.org/pdf/2507.09446v1](http://arxiv.org/pdf/2507.09446v1)**

> **作者:** Yuanhong Zheng; Ruixuan Yu; Jian Sun
>
> **备注:** ICCV 2025
>
> **摘要:** 3D multi-person motion prediction is a highly complex task, primarily due to the dependencies on both individual past movements and the interactions between agents. Moreover, effectively modeling these interactions often incurs substantial computational costs. In this work, we propose a computationally efficient model for multi-person motion prediction by simplifying spatial and temporal interactions. Our approach begins with the design of lightweight dual branches that learn local and global representations for individual and multiple persons separately. Additionally, we introduce a novel cross-level interaction block to integrate the spatial and temporal representations from both branches. To further enhance interaction modeling, we explicitly incorporate the spatial inter-person distance embedding. With above efficient temporal and spatial design, we achieve state-of-the-art performance for multiple metrics on standard datasets of CMU-Mocap, MuPoTS-3D, and 3DPW, while significantly reducing the computational cost. Code is available at https://github.com/Yuanhong-Zheng/EMPMP.
>
---
#### [new 010] Beyond Graph Model: Reliable VLM Fine-Tuning via Random Graph Adapter
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型（VLM）微调任务，旨在解决传统确定性适配器无法充分捕捉文本描述多样性和类别间关系的问题。作者提出VRGAdapter，通过引入随机图模型和不确定性引导的多分支融合策略，提升下游任务的性能。**

- **链接: [http://arxiv.org/pdf/2507.10355v1](http://arxiv.org/pdf/2507.10355v1)**

> **作者:** Bo Jiang; Xueyang Ze; Beibei Wang; Xixi Wang; Xixi Wan; Bin Luo
>
> **摘要:** Textual adapter-based tuning methods have shown significant potential in transferring knowledge from pre-trained Vision-Language Models (VLMs) to downstream tasks. Existing works generally employ the deterministic textual feature adapter to refine each category textual representation. However, due to inherent factors such as different attributes and contexts, there exists significant diversity in textual descriptions for each category. Such description diversity offers rich discriminative semantic knowledge that can benefit downstream visual learning tasks. Obviously, traditional deterministic adapter model cannot adequately capture this varied semantic information. Also, it is desirable to exploit the inter-class relationships in VLM adapter. To address these issues, we propose to exploit random graph model into VLM adapter and develop a novel Vertex Random Graph Adapter (VRGAdapter). VRGAdapter first models the inherent diverse descriptions of each category and inter-class relationships of different categories simultaneously by leveraging a Vertex Random Knowledge Graph (VRKG) model. Then, it employs probabilistic message propagation on VRKG to learn context-aware distribution representation for each class node. Finally, it adopts a reparameterized sampling function to achieve textual adapter learning. Note that, VRGAdapter provides a more general adapter solution that encompasses traditional graph-based adapter as a special case. In addition, to enable more robust performance for downstream tasks, we also introduce a new Uncertainty-guided Multi-branch Fusion (UMF) scheme that dynamically integrates multiple pre-trained models for ensemble prediction. Extensive experiments on multiple benchmark datasets demonstrate the effectiveness of our approach.
>
---
#### [new 011] Fast3D: Accelerating 3D Multi-modal Large Language Models for Efficient 3D Scene Understanding
- **分类: cs.CV**

- **简介: 论文提出Fast3D，旨在加速3D多模态大语言模型的推理过程。该工作属于3D场景理解任务，针对视觉token冗余导致的计算效率低下问题，设计了基于全局注意力预测和自适应剪枝的技术，在不修改模型参数的前提下实现高效推理。**

- **链接: [http://arxiv.org/pdf/2507.09334v1](http://arxiv.org/pdf/2507.09334v1)**

> **作者:** Wencan Huang; Daizong Liu; Wei Hu
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** While 3D Multi-modal Large Language Models (MLLMs) demonstrate remarkable scene understanding capabilities, their practical deployment faces critical challenges due to computational inefficiency. The key bottleneck stems from processing excessive object-centric visual tokens required for comprehensive 3D scene representation. Although visual token pruning has shown promise in accelerating 2D MLLMs, its applicability to 3D domains remains largely unexplored due to fundamental disparities in token structures. In this paper, we reveal two critical insights: (1) Significant redundancy exists in object-level 3D token representations, analogous to patch-level redundancy in 2D systems; (2) Global attention patterns exhibit strong predictive power for identifying non-essential tokens in 3D contexts. Building on these observations, we propose Fast3D, a plug-and-play visual token pruning framework for 3D MLLMs featuring two technical innovations: (1) Global Attention Prediction (GAP), where a lightweight neural network learns to predict the global attention distributions of the target model, enabling efficient token importance estimation for precise pruning guidance; (2) Sample-Adaptive visual token Pruning (SAP), which introduces dynamic token budgets through attention-based complexity assessment, automatically adjusting layer-wise pruning ratios based on input characteristics. Both of these two techniques operate without modifying the parameters of the target model. Extensive evaluations across five benchmarks validate the effectiveness of Fast3D, particularly under high visual token pruning ratios. Code is available at https://github.com/wencan25/Fast3D
>
---
#### [new 012] Devanagari Handwritten Character Recognition using Convolutional Neural Network
- **分类: cs.CV; cs.AI; cs.CL; 14J60; I.2.7; I.4; I.5; I.7.5**

- **简介: 该论文属于图像识别任务，旨在解决手写Devanagari字符的自动识别问题。作者使用卷积神经网络，基于DHCD数据集进行训练与测试，提升了识别准确率，达到96.36%的测试精度。**

- **链接: [http://arxiv.org/pdf/2507.10398v1](http://arxiv.org/pdf/2507.10398v1)**

> **作者:** Diksha Mehta; Prateek Mehta
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Handwritten character recognition is getting popular among researchers because of its possible applications in facilitating technological search engines, social media, recommender systems, etc. The Devanagari script is one of the oldest language scripts in India that does not have proper digitization tools. With the advancement of computing and technology, the task of this research is to extract handwritten Hindi characters from an image of Devanagari script with an automated approach to save time and obsolete data. In this paper, we present a technique to recognize handwritten Devanagari characters using two deep convolutional neural network layers. This work employs a methodology that is useful to enhance the recognition rate and configures a convolutional neural network for effective Devanagari handwritten text recognition (DHTR). This approach uses the Devanagari handwritten character dataset (DHCD), an open dataset with 36 classes of Devanagari characters. Each of these classes has 1700 images for training and testing purposes. This approach obtains promising results in terms of accuracy by achieving 96.36% accuracy in testing and 99.55% in training time.
>
---
#### [new 013] Prompt2DEM: High-Resolution DEMs for Urban and Open Environments from Global Prompts Using a Monocular Foundation Model
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于遥感与地形建模任务，旨在解决传统方法在高分辨率数字高程模型（DEM）生成中的局限性。作者提出Prompt2DEM框架，结合全球提示信息与单目深度估计，实现从低分辨率SRTM数据和高分辨率RGB图像生成高精度、高分辨率DEM，显著提升了城市和开放环境下的地形重建效果。**

- **链接: [http://arxiv.org/pdf/2507.09681v1](http://arxiv.org/pdf/2507.09681v1)**

> **作者:** Osher Rafaeli; Tal Svoray; Ariel Nahlieli
>
> **备注:** 18 pages
>
> **摘要:** High-resolution elevation estimations are essential to understand catchment and hillslope hydrology, study urban morphology and dynamics, and monitor the growth, decline, and mortality of terrestrial ecosystems. Various deep learning approaches (e.g., super-resolution techniques, monocular depth estimation) have been developed to create high-resolution Digital Elevation Models (DEMs). However, super-resolution techniques are limited by the upscaling factor, and monocular depth estimation lacks global elevation context, making its conversion to a seamless DEM restricted. The recently introduced technique of prompt-based monocular depth estimation has opened new opportunities to extract estimates of absolute elevation in a global context. We present here a framework for the estimation of high-resolution DEMs as a new paradigm for absolute global elevation mapping. It is exemplified using low-resolution Shuttle Radar Topography Mission (SRTM) elevation data as prompts and high-resolution RGB imagery from the National Agriculture Imagery Program (NAIP). The approach fine-tunes a vision transformer encoder with LiDAR-derived DEMs and employs a versatile prompting strategy, enabling tasks such as DEM estimation, void filling, and updating. Our framework achieves a 100x resolution gain (from 30-m to 30-cm), surpassing prior methods by an order of magnitude. Evaluations across three diverse U.S. landscapes show robust generalization, capturing urban structures and fine-scale terrain features with < 5 m MAE relative to LiDAR, improving over SRTM by up to 18%. Hydrological analysis confirms suitability for hazard and environmental studies. We demonstrate scalability by applying the framework to large regions in the U.S. and Israel. All code and pretrained models are publicly available at: https://osherr1996.github.io/prompt2dem_propage/.
>
---
#### [new 014] Frequency Regulation for Exposure Bias Mitigation in Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决扩散模型中的曝光偏差问题。通过观察噪声图像能量变化，提出频域调节方法，利用小波变换分别调整低频和高频子带，有效缓解曝光偏差，提升生成质量。方法无需训练，适用于多种模型架构。**

- **链接: [http://arxiv.org/pdf/2507.10072v1](http://arxiv.org/pdf/2507.10072v1)**

> **作者:** Meng Yu; Kun Zhan
>
> **备注:** ACM Multimedia 2025 accepted!
>
> **摘要:** Diffusion models exhibit impressive generative capabilities but are significantly impacted by exposure bias. In this paper, we make a key observation: the energy of the predicted noisy images decreases during the diffusion process. Building on this, we identify two important findings: 1) The reduction in energy follows distinct patterns in the low-frequency and high-frequency subbands; 2) This energy reduction results in amplitude variations between the network-reconstructed clean data and the real clean data. Based on the first finding, we introduce a frequency-domain regulation mechanism utilizing wavelet transforms, which separately adjusts the low- and high-frequency subbands. Leveraging the second insight, we provide a more accurate analysis of exposure bias in the two subbands. Our method is training-free and plug-and-play, significantly improving the generative quality of various diffusion models and providing a robust solution to exposure bias across different model architectures. The source code is available at https://github.com/kunzhan/wpp.
>
---
#### [new 015] CoralVQA: A Large-Scale Visual Question Answering Dataset for Coral Reef Image Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉问答（VQA）任务，旨在解决珊瑚礁图像理解中缺乏专业数据集的问题。作者构建了包含12,805张珊瑚图像和277,653个问题答案对的CoralVQA数据集，用于评估生态与健康状况，推动基于视觉语言模型的珊瑚保护研究。**

- **链接: [http://arxiv.org/pdf/2507.10449v1](http://arxiv.org/pdf/2507.10449v1)**

> **作者:** Hongyong Han; Wei Wang; Gaowei Zhang; Mingjie Li; Yi Wang
>
> **摘要:** Coral reefs are vital yet vulnerable ecosystems that require continuous monitoring to support conservation. While coral reef images provide essential information in coral monitoring, interpreting such images remains challenging due to the need for domain expertise. Visual Question Answering (VQA), powered by Large Vision-Language Models (LVLMs), has great potential in user-friendly interaction with coral reef images. However, applying VQA to coral imagery demands a dedicated dataset that addresses two key challenges: domain-specific annotations and multidimensional questions. In this work, we introduce CoralVQA, the first large-scale VQA dataset for coral reef analysis. It contains 12,805 real-world coral images from 67 coral genera collected from 3 oceans, along with 277,653 question-answer pairs that comprehensively assess ecological and health-related conditions. To construct this dataset, we develop a semi-automatic data construction pipeline in collaboration with marine biologists to ensure both scalability and professional-grade data quality. CoralVQA presents novel challenges and provides a comprehensive benchmark for studying vision-language reasoning in the context of coral reef images. By evaluating several state-of-the-art LVLMs, we reveal key limitations and opportunities. These insights form a foundation for future LVLM development, with a particular emphasis on supporting coral conservation efforts.
>
---
#### [new 016] Simplifying Traffic Anomaly Detection with Video Foundation Models
- **分类: cs.CV**

- **简介: 该论文属于交通异常检测任务，旨在解决现有方法依赖复杂架构的问题。作者采用简单的视频视觉变换器（Video ViT），通过预训练策略提升检测性能，发现自监督的MVM预训练和领域自适应DAPT可显著提高效果。结果表明，简单模型也能高效实现先进检测性能。**

- **链接: [http://arxiv.org/pdf/2507.09338v1](http://arxiv.org/pdf/2507.09338v1)**

> **作者:** Svetlana Orlova; Tommie Kerssies; Brunó B. Englert; Gijs Dubbelman
>
> **备注:** ICCVW 2025 accepted. Code: https://github.com/tue-mps/simple-tad
>
> **摘要:** Recent methods for ego-centric Traffic Anomaly Detection (TAD) often rely on complex multi-stage or multi-representation fusion architectures, yet it remains unclear whether such complexity is necessary. Recent findings in visual perception suggest that foundation models, enabled by advanced pre-training, allow simple yet flexible architectures to outperform specialized designs. Therefore, in this work, we investigate an architecturally simple encoder-only approach using plain Video Vision Transformers (Video ViTs) and study how pre-training enables strong TAD performance. We find that: (i) strong pre-training enables simple encoder-only models to match or even surpass the performance of specialized state-of-the-art TAD methods, while also being significantly more efficient; (ii) although weakly- and fully-supervised pre-training are advantageous on standard benchmarks, we find them less effective for TAD. Instead, self-supervised Masked Video Modeling (MVM) provides the strongest signal; and (iii) Domain-Adaptive Pre-Training (DAPT) on unlabeled driving videos further improves downstream performance, without requiring anomalous examples. Our findings highlight the importance of pre-training and show that effective, efficient, and scalable TAD models can be built with minimal architectural complexity. We release our code, domain-adapted encoders, and fine-tuned models to support future work: https://github.com/tue-mps/simple-tad.
>
---
#### [new 017] AGCD-Net: Attention Guided Context Debiasing Network for Emotion Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于情感识别任务，旨在解决上下文偏差问题。传统方法易受背景与情绪的虚假关联影响，作者提出AGCD-Net模型，结合Hybrid ConvNeXt与AG-CIM模块，通过因果干预和注意力机制减少上下文偏倚，提升真实场景中情绪识别的准确性。**

- **链接: [http://arxiv.org/pdf/2507.09248v1](http://arxiv.org/pdf/2507.09248v1)**

> **作者:** Varsha Devi; Amine Bohi; Pardeep Kumar
>
> **备注:** 13 Pages, 4 figures, 2 tables ICIAP 2025
>
> **摘要:** Context-aware emotion recognition (CAER) enhances affective computing in real-world scenarios, but traditional methods often suffer from context bias-spurious correlation between background context and emotion labels (e.g. associating ``garden'' with ``happy''). In this paper, we propose \textbf{AGCD-Net}, an Attention Guided Context Debiasing model that introduces \textit{Hybrid ConvNeXt}, a novel convolutional encoder that extends the ConvNeXt backbone by integrating Spatial Transformer Network and Squeeze-and-Excitation layers for enhanced feature recalibration. At the core of AGCD-Net is the Attention Guided - Causal Intervention Module (AG-CIM), which applies causal theory, perturbs context features, isolates spurious correlations, and performs an attention-driven correction guided by face features to mitigate context bias. Experimental results on the CAER-S dataset demonstrate the effectiveness of AGCD-Net, achieving state-of-the-art performance and highlighting the importance of causal debiasing for robust emotion recognition in complex settings.
>
---
#### [new 018] BenchReAD: A systematic benchmark for retinal anomaly detection
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学图像分析任务，旨在解决视网膜异常检测中缺乏全面公开基准的问题。作者构建了BenchReAD基准，并提出NFM-DRA方法，在利用监督学习的同时结合正常特征记忆，提升了对未见异常的检测性能，推动领域发展。**

- **链接: [http://arxiv.org/pdf/2507.10492v1](http://arxiv.org/pdf/2507.10492v1)**

> **作者:** Chenyu Lian; Hong-Yu Zhou; Zhanli Hu; Jing Qin
>
> **备注:** MICCAI 2025
>
> **摘要:** Retinal anomaly detection plays a pivotal role in screening ocular and systemic diseases. Despite its significance, progress in the field has been hindered by the absence of a comprehensive and publicly available benchmark, which is essential for the fair evaluation and advancement of methodologies. Due to this limitation, previous anomaly detection work related to retinal images has been constrained by (1) a limited and overly simplistic set of anomaly types, (2) test sets that are nearly saturated, and (3) a lack of generalization evaluation, resulting in less convincing experimental setups. Furthermore, existing benchmarks in medical anomaly detection predominantly focus on one-class supervised approaches (training only with negative samples), overlooking the vast amounts of labeled abnormal data and unlabeled data that are commonly available in clinical practice. To bridge these gaps, we introduce a benchmark for retinal anomaly detection, which is comprehensive and systematic in terms of data and algorithm. Through categorizing and benchmarking previous methods, we find that a fully supervised approach leveraging disentangled representations of abnormalities (DRA) achieves the best performance but suffers from significant drops in performance when encountering certain unseen anomalies. Inspired by the memory bank mechanisms in one-class supervised learning, we propose NFM-DRA, which integrates DRA with a Normal Feature Memory to mitigate the performance degradation, establishing a new SOTA. The benchmark is publicly available at https://github.com/DopamineLcy/BenchReAD.
>
---
#### [new 019] VISTA: A Visual Analytics Framework to Enhance Foundation Model-Generated Data Labels
- **分类: cs.CV**

- **简介: 该论文属于多模态模型数据标注任务，旨在提升基础模型生成标签的质量。针对自动生成标签质量参差不齐且缺乏有效验证的问题，论文提出VISTA框架，结合多阶段数据验证策略与人工专家干预，识别并修正标签中的隐藏问题，从而提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.09008v1](http://arxiv.org/pdf/2507.09008v1)**

> **作者:** Xiwei Xuan; Xiaoqi Wang; Wenbin He; Jorge Piazentin Ono; Liang Gou; Kwan-Liu Ma; Liu Ren
>
> **备注:** IEEE Transactions on Visualization and Computer Graphics (2025)
>
> **摘要:** The advances in multi-modal foundation models (FMs) (e.g., CLIP and LLaVA) have facilitated the auto-labeling of large-scale datasets, enhancing model performance in challenging downstream tasks such as open-vocabulary object detection and segmentation. However, the quality of FM-generated labels is less studied as existing approaches focus more on data quantity over quality. This is because validating large volumes of data without ground truth presents a considerable challenge in practice. Existing methods typically rely on limited metrics to identify problematic data, lacking a comprehensive perspective, or apply human validation to only a small data fraction, failing to address the full spectrum of potential issues. To overcome these challenges, we introduce VISTA, a visual analytics framework that improves data quality to enhance the performance of multi-modal models. Targeting the complex and demanding domain of open-vocabulary image segmentation, VISTA integrates multi-phased data validation strategies with human expertise, enabling humans to identify, understand, and correct hidden issues within FM-generated labels. Through detailed use cases on two benchmark datasets and expert reviews, we demonstrate VISTA's effectiveness from both quantitative and qualitative perspectives.
>
---
#### [new 020] Harnessing Text-to-Image Diffusion Models for Point Cloud Self-Supervised Learning
- **分类: cs.CV**

- **简介: 该论文属于3D点云自监督学习任务，旨在解决现有3D扩散模型因训练数据有限而性能受限的问题。作者提出PointSD框架，利用大规模文本到图像扩散模型Stable Diffusion（SD）辅助点云表征学习，通过构建点云到图像的扩散模型提取语义特征，并用于训练3D骨干网络，有效提升了点云学习效果。**

- **链接: [http://arxiv.org/pdf/2507.09102v1](http://arxiv.org/pdf/2507.09102v1)**

> **作者:** Yiyang Chen; Shanshan Zhao; Lunhao Duan; Changxing Ding; Dacheng Tao
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Diffusion-based models, widely used in text-to-image generation, have proven effective in 2D representation learning. Recently, this framework has been extended to 3D self-supervised learning by constructing a conditional point generator for enhancing 3D representations. However, its performance remains constrained by the 3D diffusion model, which is trained on the available 3D datasets with limited size. We hypothesize that the robust capabilities of text-to-image diffusion models, particularly Stable Diffusion (SD), which is trained on large-scale datasets, can help overcome these limitations. To investigate this hypothesis, we propose PointSD, a framework that leverages the SD model for 3D self-supervised learning. By replacing the SD model's text encoder with a 3D encoder, we train a point-to-image diffusion model that allows point clouds to guide the denoising of rendered noisy images. With the trained point-to-image diffusion model, we use noise-free images as the input and point clouds as the condition to extract SD features. Next, we train a 3D backbone by aligning its features with these SD features, thereby facilitating direct semantic learning. Comprehensive experiments on downstream point cloud tasks and ablation studies demonstrate that the SD model can enhance point cloud self-supervised learning. Code is publicly available at https://github.com/wdttt/PointSD.
>
---
#### [new 021] Spatial Lifting for Dense Prediction
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 论文提出“空间提升”（SL）方法，用于密集预测任务。该方法通过将2D图像升到高维空间，再用高维网络（如3D U-Net）处理，在减少模型参数和推理成本的同时保持性能。研究涵盖语义分割与深度估计，在19个数据集上验证效果，实现高效、准确的密集预测。**

- **链接: [http://arxiv.org/pdf/2507.10222v1](http://arxiv.org/pdf/2507.10222v1)**

> **作者:** Mingzhi Xu; Yizhe Zhang
>
> **备注:** Preprint. Under review
>
> **摘要:** We present Spatial Lifting (SL), a novel methodology for dense prediction tasks. SL operates by lifting standard inputs, such as 2D images, into a higher-dimensional space and subsequently processing them using networks designed for that higher dimension, such as a 3D U-Net. Counterintuitively, this dimensionality lifting allows us to achieve good performance on benchmark tasks compared to conventional approaches, while reducing inference costs and significantly lowering the number of model parameters. The SL framework produces intrinsically structured outputs along the lifted dimension. This emergent structure facilitates dense supervision during training and enables robust, near-zero-additional-cost prediction quality assessment at test time. We validate our approach across 19 benchmark datasets (13 for semantic segmentation and 6 for depth estimation), demonstrating competitive dense prediction performance while reducing the model parameter count by over 98% (in the U-Net case) and lowering inference costs. Spatial Lifting introduces a new vision modeling paradigm that offers a promising path toward more efficient, accurate, and reliable deep networks for dense prediction tasks in vision.
>
---
#### [new 022] Taming generative video models for zero-shot optical flow extraction
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决无监督光流提取问题。作者提出KL-tracing方法，通过扰动输入帧并分析模型预测分布差异来提取光流，无需微调即可在真实和合成数据集上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2507.09082v1](http://arxiv.org/pdf/2507.09082v1)**

> **作者:** Seungwoo Kim; Khai Loong Aw; Klemen Kotar; Cristobal Eyzaguirre; Wanhee Lee; Yunong Liu; Jared Watrous; Stefan Stojanov; Juan Carlos Niebles; Jiajun Wu; Daniel L. K. Yamins
>
> **备注:** Project webpage: https://neuroailab.github.io/projects/kl_tracing
>
> **摘要:** Extracting optical flow from videos remains a core computer vision problem. Motivated by the success of large general-purpose models, we ask whether frozen self-supervised video models trained only for future frame prediction can be prompted, without fine-tuning, to output flow. Prior work reading out depth or illumination from video generators required fine-tuning, which is impractical for flow where labels are scarce and synthetic datasets suffer from a sim-to-real gap. Inspired by the Counterfactual World Model (CWM) paradigm, which can obtain point-wise correspondences by injecting a small tracer perturbation into a next-frame predictor and tracking its propagation, we extend this idea to generative video models. We explore several popular architectures and find that successful zero-shot flow extraction in this manner is aided by three model properties: (1) distributional prediction of future frames (avoiding blurry or noisy outputs); (2) factorized latents that treat each spatio-temporal patch independently; and (3) random-access decoding that can condition on any subset of future pixels. These properties are uniquely present in the recent Local Random Access Sequence (LRAS) architecture. Building on LRAS, we propose KL-tracing: a novel test-time procedure that injects a localized perturbation into the first frame, rolls out the model one step, and computes the Kullback-Leibler divergence between perturbed and unperturbed predictive distributions. Without any flow-specific fine-tuning, our method outperforms state-of-the-art models on real-world TAP-Vid DAVIS dataset (16.6% relative improvement for endpoint error) and synthetic TAP-Vid Kubric (4.7% relative improvement). Our results indicate that counterfactual prompting of controllable generative video models is a scalable and effective alternative to supervised or photometric-loss approaches for high-quality flow.
>
---
#### [new 023] SDTN and TRN: Adaptive Spectral-Spatial Feature Extraction for Hyperspectral Image Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于遥感图像分类任务，旨在解决高光谱图像分类中数据高维、谱空冗余及标注样本少导致的性能下降问题。作者提出了SDTN和TRN网络，通过自适应张量分解与正则化机制提取多尺度谱空特征，在降低计算复杂度的同时提升了分类精度。**

- **链接: [http://arxiv.org/pdf/2507.09492v1](http://arxiv.org/pdf/2507.09492v1)**

> **作者:** Fuyin Ye; Erwen Yao; Jianyong Chen; Fengmei He; Junxiang Zhang; Lihao Ni
>
> **备注:** 4 pages, 2 figures
>
> **摘要:** Hyperspectral image classification plays a pivotal role in precision agriculture, providing accurate insights into crop health monitoring, disease detection, and soil analysis. However, traditional methods struggle with high-dimensional data, spectral-spatial redundancy, and the scarcity of labeled samples, often leading to suboptimal performance. To address these challenges, we propose the Self-Adaptive Tensor- Regularized Network (SDTN), which combines tensor decomposition with regularization mechanisms to dynamically adjust tensor ranks, ensuring optimal feature representation tailored to the complexity of the data. Building upon SDTN, we propose the Tensor-Regularized Network (TRN), which integrates the features extracted by SDTN into a lightweight network capable of capturing spectral-spatial features at multiple scales. This approach not only maintains high classification accuracy but also significantly reduces computational complexity, making the framework highly suitable for real-time deployment in resource-constrained environments. Experiments on PaviaU datasets demonstrate significant improvements in accuracy and reduced model parameters compared to state-of-the-art methods.
>
---
#### [new 024] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于零样本OOD检测任务，旨在解决基于负标签的方法误判问题。通过过滤子类和专有名词，并引入多匹配感知评分函数，提升分布内外样本区分的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.09795v1](http://arxiv.org/pdf/2507.09795v1)**

> **作者:** Amirhossein Ansari; Ke Wang; Pulei Xiong
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Recent advancements in Vision-Language Models like CLIP have enabled zero-shot OOD detection by leveraging both image and textual label information. Among these, negative label-based methods such as NegLabel and CSP have shown promising results by utilizing a lexicon of words to define negative labels for distinguishing OOD samples. However, these methods suffer from detecting in-distribution samples as OOD due to negative labels that are subcategories of in-distribution labels or proper nouns. They also face limitations in handling images that match multiple in-distribution and negative labels. We propose NegRefine, a novel negative label refinement framework for zero-shot OOD detection. By introducing a filtering mechanism to exclude subcategory labels and proper nouns from the negative label set and incorporating a multi-matching-aware scoring function that dynamically adjusts the contributions of multiple labels matching an image, NegRefine ensures a more robust separation between in-distribution and OOD samples. We evaluate NegRefine on large-scale benchmarks, including ImageNet-1K. Source code is available at https://github.com/ah-ansari/NegRefine.
>
---
#### [new 025] Cross Knowledge Distillation between Artificial and Spiking Neural Networks
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于知识蒸馏任务，旨在提升脉冲神经网络（SNNs）在事件数据上的性能。通过利用传统人工神经网络（ANNs）和RGB数据进行跨模态、跨架构的知识迁移，提出了一种名为CKD的方法，解决了模态与架构差异带来的挑战，并在多个数据集上验证了其优越性。**

- **链接: [http://arxiv.org/pdf/2507.09269v1](http://arxiv.org/pdf/2507.09269v1)**

> **作者:** Shuhan Ye; Yuanbin Qian; Chong Wang; Sunqi Lin; Jiazhen Xu; Jiangbo Qian; Yuqi Li
>
> **备注:** This paper has been accepted by ICME2025
>
> **摘要:** Recently, Spiking Neural Networks (SNNs) have demonstrated rich potential in computer vision domain due to their high biological plausibility, event-driven characteristic and energy-saving efficiency. Still, limited annotated event-based datasets and immature SNN architectures result in their performance inferior to that of Artificial Neural Networks (ANNs). To enhance the performance of SNNs on their optimal data format, DVS data, we explore using RGB data and well-performing ANNs to implement knowledge distillation. In this case, solving cross-modality and cross-architecture challenges is necessary. In this paper, we propose cross knowledge distillation (CKD), which not only leverages semantic similarity and sliding replacement to mitigate the cross-modality challenge, but also uses an indirect phased knowledge distillation to mitigate the cross-architecture challenge. We validated our method on main-stream neuromorphic datasets, including N-Caltech101 and CEP-DVS. The experimental results show that our method outperforms current State-of-the-Art methods. The code will be available at https://github.com/ShawnYE618/CKD
>
---
#### [new 026] EmbRACE-3K: Embodied Reasoning and Action in Complex Environments
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文提出EmRACE-3K数据集，属于具身智能任务，旨在解决视觉-语言模型在动态交互环境中推理与规划能力不足的问题。数据集包含3000余个语言引导任务，涵盖导航、操作与多步骤目标执行，用于评估模型在探索、空间语义推理和多阶段任务中的表现，并通过微调提升模型能力。**

- **链接: [http://arxiv.org/pdf/2507.10548v1](http://arxiv.org/pdf/2507.10548v1)**

> **作者:** Mingxian Lin; Wei Huang; Yitang Li; Chengjie Jiang; Kui Wu; Fangwei Zhong; Shengju Qian; Xin Wang; Xiaojuan Qi
>
> **备注:** Project page: https://mxllc.github.io/EmbRACE-3K/
>
> **摘要:** Recent advanced vision-language models(VLMs) have demonstrated strong performance on passive, offline image and video understanding tasks. However, their effectiveness in embodied settings, which require online interaction and active scene understanding remains limited. In such scenarios, an agent perceives the environment from a first-person perspective, with each action dynamically shaping subsequent observations. Even state-of-the-art models such as GPT-4o, Claude 3.5 Sonnet, and Gemini 2.5 Pro struggle in open-environment interactions, exhibiting clear limitations in spatial reasoning and long-horizon planning. To address this gap, we introduce EmRACE-3K, a dataset of over 3,000 language-guided tasks situated in diverse, photorealistic environments constructed using Unreal Engine and the UnrealCV-Zoo framework. The tasks encompass a wide range of embodied challenges, including navigation, object manipulation, and multi-stage goal execution. Each task unfolds as a multi-step trajectory, pairing first-person visual observations with high-level instructions, grounded actions, and natural language rationales that express the agent's intent at every step. Using EmRACE-3K, we establish a benchmark to evaluate the embodied reasoning capabilities of VLMs across three key dimensions: Exploration, Dynamic Spatial-Semantic Reasoning, and Multi-stage Goal Execution. In zero-shot settings, all models achieve success rates below 20%, underscoring the challenge posed by our benchmark and the current limitations of VLMs in interactive environments. To demonstrate the utility of EmRACE-3K, we further fine-tune Qwen2.5-VL-7B using supervised learning followed by reinforcement learning. This approach yields substantial improvements across all three challenge categories, highlighting the dataset's effectiveness in enabling the development of embodied reasoning capabilities.
>
---
#### [new 027] Disentanglement and Assessment of Shortcuts in Ophthalmological Retinal Imaging Exams
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分析任务，旨在解决糖尿病视网膜病变（DR）诊断中AI模型的公平性与泛化能力问题。研究评估了三种模型在预测DR及敏感属性（如年龄、性别）中的表现，并探索解耦技术对缓解偏差的影响，发现解耦效果因模型而异，强调医疗AI中公平性的重要性。**

- **链接: [http://arxiv.org/pdf/2507.09640v1](http://arxiv.org/pdf/2507.09640v1)**

> **作者:** Leonor Fernandes; Tiago Gonçalves; João Matos; Luis Filipe Nakayama; Jaime S. Cardoso
>
> **备注:** 10 pages. Under review
>
> **摘要:** Diabetic retinopathy (DR) is a leading cause of vision loss in working-age adults. While screening reduces the risk of blindness, traditional imaging is often costly and inaccessible. Artificial intelligence (AI) algorithms present a scalable diagnostic solution, but concerns regarding fairness and generalization persist. This work evaluates the fairness and performance of image-trained models in DR prediction, as well as the impact of disentanglement as a bias mitigation technique, using the diverse mBRSET fundus dataset. Three models, ConvNeXt V2, DINOv2, and Swin V2, were trained on macula images to predict DR and sensitive attributes (SAs) (e.g., age and gender/sex). Fairness was assessed between subgroups of SAs, and disentanglement was applied to reduce bias. All models achieved high DR prediction performance in diagnosing (up to 94% AUROC) and could reasonably predict age and gender/sex (91% and 77% AUROC, respectively). Fairness assessment suggests disparities, such as a 10% AUROC gap between age groups in DINOv2. Disentangling SAs from DR prediction had varying results, depending on the model selected. Disentanglement improved DINOv2 performance (2% AUROC gain), but led to performance drops in ConvNeXt V2 and Swin V2 (7% and 3%, respectively). These findings highlight the complexity of disentangling fine-grained features in fundus imaging and emphasize the importance of fairness in medical imaging AI to ensure equitable and reliable healthcare solutions.
>
---
#### [new 028] SegVec3D: A Method for Vector Embedding of 3D Objects Oriented Towards Robot manipulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D点云实例分割与多模态理解任务，旨在提升机器人操作中对3D物体的识别与语义理解能力。论文提出SegVec3D框架，融合注意力机制、嵌入学习和跨模态对齐，实现无监督实例分割与零样本检索，解决了现有方法在监督依赖和多模态融合上的不足。**

- **链接: [http://arxiv.org/pdf/2507.09459v1](http://arxiv.org/pdf/2507.09459v1)**

> **作者:** Zhihan Kang; Boyu Wang
>
> **备注:** Undergraduate Theis; 12 pages, 6 figures
>
> **摘要:** We propose SegVec3D, a novel framework for 3D point cloud instance segmentation that integrates attention mechanisms, embedding learning, and cross-modal alignment. The approach builds a hierarchical feature extractor to enhance geometric structure modeling and enables unsupervised instance segmentation via contrastive clustering. It further aligns 3D data with natural language queries in a shared semantic space, supporting zero-shot retrieval. Compared to recent methods like Mask3D and ULIP, our method uniquely unifies instance segmentation and multimodal understanding with minimal supervision and practical deployability.
>
---
#### [new 029] PPJudge: Towards Human-Aligned Assessment of Artistic Painting Process
- **分类: cs.CV**

- **简介: 该论文属于艺术绘画过程评估任务，旨在解决现有方法仅关注静态图像、忽视创作过程的问题。作者构建了包含真实与合成绘画过程的数据集PPAD，并提出PPJudge模型，结合时序编码与专家混合架构，更贴合人类评价，提升对艺术创作过程的计算理解与教育应用。**

- **链接: [http://arxiv.org/pdf/2507.09242v1](http://arxiv.org/pdf/2507.09242v1)**

> **作者:** Shiqi Jiang; Xinpeng Li; Xi Mao; Changbo Wang; Chenhui Li
>
> **备注:** ACM International Conference on Multimedia 2025
>
> **摘要:** Artistic image assessment has become a prominent research area in computer vision. In recent years, the field has witnessed a proliferation of datasets and methods designed to evaluate the aesthetic quality of paintings. However, most existing approaches focus solely on static final images, overlooking the dynamic and multi-stage nature of the artistic painting process. To address this gap, we propose a novel framework for human-aligned assessment of painting processes. Specifically, we introduce the Painting Process Assessment Dataset (PPAD), the first large-scale dataset comprising real and synthetic painting process images, annotated by domain experts across eight detailed attributes. Furthermore, we present PPJudge (Painting Process Judge), a Transformer-based model enhanced with temporally-aware positional encoding and a heterogeneous mixture-of-experts architecture, enabling effective assessment of the painting process. Experimental results demonstrate that our method outperforms existing baselines in accuracy, robustness, and alignment with human judgment, offering new insights into computational creativity and art education.
>
---
#### [new 030] Can GPT-4o mini and Gemini 2.0 Flash Predict Fine-Grained Fashion Product Attributes? A Zero-Shot Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于时尚产品属性识别任务，旨在解决大型语言模型在细粒度时尚属性理解上的性能问题。论文通过零样本方法评估GPT-4o-mini和Gemini 2.0 Flash在DeepFashion-MultiModal数据集上的表现，使用图像作为输入，分析18类时尚属性的预测效果，并提供错误分析与实用见解。**

- **链接: [http://arxiv.org/pdf/2507.09950v1](http://arxiv.org/pdf/2507.09950v1)**

> **作者:** Shubham Shukla; Kunal Sonalkar
>
> **备注:** 11 pages, 2 figures
>
> **摘要:** The fashion retail business is centered around the capacity to comprehend products. Product attribution helps in comprehending products depending on the business process. Quality attribution improves the customer experience as they navigate through millions of products offered by a retail website. It leads to well-organized product catalogs. In the end, product attribution directly impacts the 'discovery experience' of the customer. Although large language models (LLMs) have shown remarkable capabilities in understanding multimodal data, their performance on fine-grained fashion attribute recognition remains under-explored. This paper presents a zero-shot evaluation of state-of-the-art LLMs that balance performance with speed and cost efficiency, mainly GPT-4o-mini and Gemini 2.0 Flash. We have used the dataset DeepFashion-MultiModal (https://github.com/yumingj/DeepFashion-MultiModal) to evaluate these models in the attribution tasks of fashion products. Our study evaluates these models across 18 categories of fashion attributes, offering insight into where these models excel. We only use images as the sole input for product information to create a constrained environment. Our analysis shows that Gemini 2.0 Flash demonstrates the strongest overall performance with a macro F1 score of 56.79% across all attributes, while GPT-4o-mini scored a macro F1 score of 43.28%. Through detailed error analysis, our findings provide practical insights for deploying these LLMs in production e-commerce product attribution-related tasks and highlight the need for domain-specific fine-tuning approaches. This work also lays the groundwork for future research in fashion AI and multimodal attribute extraction.
>
---
#### [new 031] Memory-Efficient Personalization of Text-to-Image Diffusion Models via Selective Optimization Strategies
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于文本到图像生成模型的个性化任务，旨在解决在边缘设备上高效微调模型的问题。论文提出一种选择性优化框架，结合低分辨率反向传播与高分辨率零阶优化，动态选择优化策略以减少内存消耗并保持生成质量，实现隐私保护和高效个性化。**

- **链接: [http://arxiv.org/pdf/2507.10029v1](http://arxiv.org/pdf/2507.10029v1)**

> **作者:** Seokeon Choi; Sunghyun Park; Hyoungwoo Park; Jeongho Kim; Sungrack Yun
>
> **摘要:** Memory-efficient personalization is critical for adapting text-to-image diffusion models while preserving user privacy and operating within the limited computational resources of edge devices. To this end, we propose a selective optimization framework that adaptively chooses between backpropagation on low-resolution images (BP-low) and zeroth-order optimization on high-resolution images (ZO-high), guided by the characteristics of the diffusion process. As observed in our experiments, BP-low efficiently adapts the model to target-specific features, but suffers from structural distortions due to resolution mismatch. Conversely, ZO-high refines high-resolution details with minimal memory overhead but faces slow convergence when applied without prior adaptation. By complementing both methods, our framework leverages BP-low for effective personalization while using ZO-high to maintain structural consistency, achieving memory-efficient and high-quality fine-tuning. To maximize the efficacy of both BP-low and ZO-high, we introduce a timestep-aware probabilistic function that dynamically selects the appropriate optimization strategy based on diffusion timesteps. This function mitigates the overfitting from BP-low at high timesteps, where structural information is critical, while ensuring ZO-high is applied more effectively as training progresses. Experimental results demonstrate that our method achieves competitive performance while significantly reducing memory consumption, enabling scalable, high-quality on-device personalization without increasing inference latency.
>
---
#### [new 032] Privacy-Preserving Multi-Stage Fall Detection Framework with Semi-supervised Federated Learning and Robotic Vision Confirmation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于智能医疗任务，旨在解决老年人跌倒检测中的隐私与准确率问题。作者提出了一种结合半监督联邦学习（SF2D）、室内定位导航和机器人视觉确认的多阶段跌倒检测框架。通过可穿戴设备初步识别跌倒，再由机器人导航至事发地点并用视觉系统确认，最终实现高精度（99.99%）且保护隐私的跌倒检测。**

- **链接: [http://arxiv.org/pdf/2507.10474v1](http://arxiv.org/pdf/2507.10474v1)**

> **作者:** Seyed Alireza Rahimi Azghadi; Truong-Thanh-Hung Nguyen; Helene Fournier; Monica Wachowicz; Rene Richard; Francis Palma; Hung Cao
>
> **摘要:** The aging population is growing rapidly, and so is the danger of falls in older adults. A major cause of injury is falling, and detection in time can greatly save medical expenses and recovery time. However, to provide timely intervention and avoid unnecessary alarms, detection systems must be effective and reliable while addressing privacy concerns regarding the user. In this work, we propose a framework for detecting falls using several complementary systems: a semi-supervised federated learning-based fall detection system (SF2D), an indoor localization and navigation system, and a vision-based human fall recognition system. A wearable device and an edge device identify a fall scenario in the first system. On top of that, the second system uses an indoor localization technique first to localize the fall location and then navigate a robot to inspect the scenario. A vision-based detection system running on an edge device with a mounted camera on a robot is used to recognize fallen people. Each of the systems of this proposed framework achieves different accuracy rates. Specifically, the SF2D has a 0.81% failure rate equivalent to 99.19% accuracy, while the vision-based fallen people detection achieves 96.3% accuracy. However, when we combine the accuracy of these two systems with the accuracy of the navigation system (95% success rate), our proposed framework creates a highly reliable performance for fall detection, with an overall accuracy of 99.99%. Not only is the proposed framework safe for older adults, but it is also a privacy-preserving solution for detecting falls.
>
---
#### [new 033] SAGE: Segment-Aware Gloss-Free Encoding for Token-Efficient Sign Language Translation
- **分类: cs.CV**

- **简介: 该论文属于手语翻译任务，旨在解决无标注数据下模型复杂度高、计算需求大的问题。工作包括：提出基于分割的视觉标记化方法，降低序列长度与内存使用；引入对比对齐目标和双级监督，提升跨模态对齐效果，从而在减少输入长度的同时实现更优性能。**

- **链接: [http://arxiv.org/pdf/2507.09266v1](http://arxiv.org/pdf/2507.09266v1)**

> **作者:** JianHe Low; Ozge Mercanoglu Sincan; Richard Bowden
>
> **备注:** Accepted in International Conference on Computer Vision (ICCV) Workshops
>
> **摘要:** Gloss-free Sign Language Translation (SLT) has advanced rapidly, achieving strong performances without relying on gloss annotations. However, these gains have often come with increased model complexity and high computational demands, raising concerns about scalability, especially as large-scale sign language datasets become more common. We propose a segment-aware visual tokenization framework that leverages sign segmentation to convert continuous video into discrete, sign-informed visual tokens. This reduces input sequence length by up to 50% compared to prior methods, resulting in up to 2.67x lower memory usage and better scalability on larger datasets. To bridge the visual and linguistic modalities, we introduce a token-to-token contrastive alignment objective, along with a dual-level supervision that aligns both language embeddings and intermediate hidden states. This improves fine-grained cross-modal alignment without relying on gloss-level supervision. Our approach notably exceeds the performance of state-of-the-art methods on the PHOENIX14T benchmark, while significantly reducing sequence length. Further experiments also demonstrate our improved performance over prior work under comparable sequence-lengths, validating the potential of our tokenization and alignment strategies.
>
---
#### [new 034] HMID-Net: An Exploration of Masked Image Modeling and Knowledge Distillation in Hyperbolic Space
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语义层次建模任务，旨在更高效地捕捉和利用数据中的层次结构。作者提出了HMID-Net，首次将掩码图像建模和知识蒸馏引入双曲空间，设计了适用于双曲空间的蒸馏损失函数。实验表明其方法在图像分类与检索等下游任务中显著优于MERU和CLIP。**

- **链接: [http://arxiv.org/pdf/2507.09487v1](http://arxiv.org/pdf/2507.09487v1)**

> **作者:** Changli Wang; Fang Yin; Jiafeng Liu; Rui Wu
>
> **摘要:** Visual and semantic concepts are often structured in a hierarchical manner. For instance, textual concept `cat' entails all images of cats. A recent study, MERU, successfully adapts multimodal learning techniques from Euclidean space to hyperbolic space, effectively capturing the visual-semantic hierarchy. However, a critical question remains: how can we more efficiently train a model to capture and leverage this hierarchy? In this paper, we propose the \textit{Hyperbolic Masked Image and Distillation Network} (HMID-Net), a novel and efficient method that integrates Masked Image Modeling (MIM) and knowledge distillation techniques within hyperbolic space. To the best of our knowledge, this is the first approach to leverage MIM and knowledge distillation in hyperbolic space to train highly efficient models. In addition, we introduce a distillation loss function specifically designed to facilitate effective knowledge transfer in hyperbolic space. Our experiments demonstrate that MIM and knowledge distillation techniques in hyperbolic space can achieve the same remarkable success as in Euclidean space. Extensive evaluations show that our method excels across a wide range of downstream tasks, significantly outperforming existing models like MERU and CLIP in both image classification and retrieval.
>
---
#### [new 035] From Physics to Foundation Models: A Review of AI-Driven Quantitative Remote Sensing Inversion
- **分类: cs.CV**

- **简介: 该论文属于定量遥感反演任务，旨在通过人工智能方法估计地表连续变量。论文回顾了从物理模型到基础模型的发展，比较了不同方法的假设、应用场景与局限性，重点分析了基础模型在自监督预训练、多模态融合和跨任务适应中的进展，并指出了物理可解释性、领域泛化等挑战。**

- **链接: [http://arxiv.org/pdf/2507.09081v1](http://arxiv.org/pdf/2507.09081v1)**

> **作者:** Zhenyu Yu; Mohd Yamani Idna Idris; Hua Wang; Pei Wang; Junyi Chen; Kun Wang
>
> **摘要:** Quantitative remote sensing inversion aims to estimate continuous surface variables-such as biomass, vegetation indices, and evapotranspiration-from satellite observations, supporting applications in ecosystem monitoring, carbon accounting, and land management. With the evolution of remote sensing systems and artificial intelligence, traditional physics-based paradigms are giving way to data-driven and foundation model (FM)-based approaches. This paper systematically reviews the methodological evolution of inversion techniques, from physical models (e.g., PROSPECT, SCOPE, DART) to machine learning methods (e.g., deep learning, multimodal fusion), and further to foundation models (e.g., SatMAE, GFM, mmEarth). We compare the modeling assumptions, application scenarios, and limitations of each paradigm, with emphasis on recent FM advances in self-supervised pretraining, multi-modal integration, and cross-task adaptation. We also highlight persistent challenges in physical interpretability, domain generalization, limited supervision, and uncertainty quantification. Finally, we envision the development of next-generation foundation models for remote sensing inversion, emphasizing unified modeling capacity, cross-domain generalization, and physical interpretability.
>
---
#### [new 036] Navigating the Challenges of AI-Generated Image Detection in the Wild: What Truly Matters?
- **分类: cs.CV**

- **简介: 该论文属于AI生成图像检测任务，旨在解决当前检测模型在真实场景中效果不佳的问题。作者通过引入新数据集ITW-SM和系统分析影响检测性能的四个因素，提升了模型在现实条件下的检测能力。**

- **链接: [http://arxiv.org/pdf/2507.10236v1](http://arxiv.org/pdf/2507.10236v1)**

> **作者:** Despina Konstantinidou; Dimitrios Karageorgiou; Christos Koutlis; Olga Papadopoulou; Emmanouil Schinas; Symeon Papadopoulos
>
> **备注:** 35 pages, 4 figures
>
> **摘要:** The rapid advancement of generative technologies presents both unprecedented creative opportunities and significant challenges, particularly in maintaining social trust and ensuring the integrity of digital information. Following these concerns, the challenge of AI-Generated Image Detection (AID) becomes increasingly critical. As these technologies become more sophisticated, the quality of AI-generated images has reached a level that can easily deceive even the most discerning observers. Our systematic evaluation highlights a critical weakness in current AI-Generated Image Detection models: while they perform exceptionally well on controlled benchmark datasets, they struggle significantly with real-world variations. To assess this, we introduce ITW-SM, a new dataset of real and AI-generated images collected from major social media platforms. In this paper, we identify four key factors that influence AID performance in real-world scenarios: backbone architecture, training data composition, pre-processing strategies and data augmentation combinations. By systematically analyzing these components, we shed light on their impact on detection efficacy. Our modifications result in an average AUC improvement of 26.87% across various AID models under real-world conditions.
>
---
#### [new 037] Text Embedding Knows How to Quantize Text-Guided Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于文本生成图像任务，旨在解决扩散模型计算复杂度高、难以部署的问题。作者提出QLIP方法，利用文本提示指导模型量化，动态选择每层的精度，降低计算开销并提升生成质量。**

- **链接: [http://arxiv.org/pdf/2507.10340v1](http://arxiv.org/pdf/2507.10340v1)**

> **作者:** Hongjae Lee; Myungjun Son; Dongjea Kang; Seung-Won Jung
>
> **备注:** ICCV 2025
>
> **摘要:** Despite the success of diffusion models in image generation tasks such as text-to-image, the enormous computational complexity of diffusion models limits their use in resource-constrained environments. To address this, network quantization has emerged as a promising solution for designing efficient diffusion models. However, existing diffusion model quantization methods do not consider input conditions, such as text prompts, as an essential source of information for quantization. In this paper, we propose a novel quantization method dubbed Quantization of Language-to-Image diffusion models using text Prompts (QLIP). QLIP leverages text prompts to guide the selection of bit precision for every layer at each time step. In addition, QLIP can be seamlessly integrated into existing quantization methods to enhance quantization efficiency. Our extensive experiments demonstrate the effectiveness of QLIP in reducing computational complexity and improving the quality of the generated images across various datasets.
>
---
#### [new 038] Synthesizing Near-Boundary OOD Samples for Out-of-Distribution Detection
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型中的分布外检测任务，旨在解决近边界分布外样本易被误分类的问题。通过利用基础模型生成边界对齐的合成分布外样本，并结合梯度优化与负标签微调，提升CLIP模型的边界区分能力，显著优化检测性能。**

- **链接: [http://arxiv.org/pdf/2507.10225v1](http://arxiv.org/pdf/2507.10225v1)**

> **作者:** Jinglun Li; Kaixun Jiang; Zhaoyu Chen; Bo Lin; Yao Tang; Weifeng Ge; Wenqiang Zhang
>
> **摘要:** Pre-trained vision-language models have exhibited remarkable abilities in detecting out-of-distribution (OOD) samples. However, some challenging OOD samples, which lie close to in-distribution (InD) data in image feature space, can still lead to misclassification. The emergence of foundation models like diffusion models and multimodal large language models (MLLMs) offers a potential solution to this issue. In this work, we propose SynOOD, a novel approach that harnesses foundation models to generate synthetic, challenging OOD data for fine-tuning CLIP models, thereby enhancing boundary-level discrimination between InD and OOD samples. Our method uses an iterative in-painting process guided by contextual prompts from MLLMs to produce nuanced, boundary-aligned OOD samples. These samples are refined through noise adjustments based on gradients from OOD scores like the energy score, effectively sampling from the InD/OOD boundary. With these carefully synthesized images, we fine-tune the CLIP image encoder and negative label features derived from the text encoder to strengthen connections between near-boundary OOD samples and a set of negative labels. Finally, SynOOD achieves state-of-the-art performance on the large-scale ImageNet benchmark, with minimal increases in parameters and runtime. Our approach significantly surpasses existing methods, improving AUROC by 2.80% and reducing FPR95 by 11.13%. Codes are available in https://github.com/Jarvisgivemeasuit/SynOOD.
>
---
#### [new 039] (Almost) Free Modality Stitching of Foundation Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出Hyma方法，解决多模态模型中单模模型选择和连接模块训练的高昂计算成本问题。通过超网络预测参数，实现多组合单模模型的连接模块联合训练，大幅降低搜索成本，同时保持性能。属于多模态模型优化任务。**

- **链接: [http://arxiv.org/pdf/2507.10015v1](http://arxiv.org/pdf/2507.10015v1)**

> **作者:** Jaisidh Singh; Diganta Misra; Boris Knyazev; Antonio Orvieto
>
> **备注:** Pre-print
>
> **摘要:** Foundation multi-modal models are often designed by stitching of multiple existing pretrained uni-modal models: for example, an image classifier with an autoregressive text model. This stitching process is performed by training a connector module that aims to align the representation-representation or representation-input spaces of these uni-modal models. However, given the complexity of training such connectors on large scale web-based datasets coupled with the ever-increasing number of available pretrained uni-modal models, the task of uni-modal models selection and subsequent connector module training becomes computationally demanding. To address this under-studied critical problem, we propose Hypernetwork Model Alignment (Hyma), a novel all-in-one solution for optimal uni-modal model selection and connector training by leveraging hypernetworks. Specifically, our framework utilizes the parameter prediction capability of a hypernetwork to obtain jointly trained connector modules for $N \times M$ combinations of uni-modal models. In our experiments, Hyma reduces the optimal uni-modal model pair search cost by $10\times$ (averaged across all experiments), while matching the ranking and trained connector performance obtained via grid search across a suite of diverse multi-modal benchmarks.
>
---
#### [new 040] Mind the Gap: Aligning Vision Foundation Models to Image Feature Matching
- **分类: cs.CV**

- **简介: 该论文属于图像特征匹配任务，旨在解决视觉基础模型在跨图像理解中的错位问题。现有方法忽略单图理解与多图匹配间的差异，导致多实例场景效果差。作者提出IMD框架，结合扩散模型和跨图像提示机制，提升匹配性能，并构建新基准IMIM验证效果。**

- **链接: [http://arxiv.org/pdf/2507.10318v1](http://arxiv.org/pdf/2507.10318v1)**

> **作者:** Yuhan Liu; Jingwen Fu; Yang Wu; Kangyi Wu; Pengna Li; Jiayi Wu; Sanping Zhou; Jingmin Xin
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Leveraging the vision foundation models has emerged as a mainstream paradigm that improves the performance of image feature matching. However, previous works have ignored the misalignment when introducing the foundation models into feature matching. The misalignment arises from the discrepancy between the foundation models focusing on single-image understanding and the cross-image understanding requirement of feature matching. Specifically, 1) the embeddings derived from commonly used foundation models exhibit discrepancies with the optimal embeddings required for feature matching; 2) lacking an effective mechanism to leverage the single-image understanding ability into cross-image understanding. A significant consequence of the misalignment is they struggle when addressing multi-instance feature matching problems. To address this, we introduce a simple but effective framework, called IMD (Image feature Matching with a pre-trained Diffusion model) with two parts: 1) Unlike the dominant solutions employing contrastive-learning based foundation models that emphasize global semantics, we integrate the generative-based diffusion models to effectively capture instance-level details. 2) We leverage the prompt mechanism in generative model as a natural tunnel, propose a novel cross-image interaction prompting module to facilitate bidirectional information interaction between image pairs. To more accurately measure the misalignment, we propose a new benchmark called IMIM, which focuses on multi-instance scenarios. Our proposed IMD establishes a new state-of-the-art in commonly evaluated benchmarks, and the superior improvement 12% in IMIM indicates our method efficiently mitigates the misalignment.
>
---
#### [new 041] EyeSeg: An Uncertainty-Aware Eye Segmentation Framework for AR/VR
- **分类: cs.CV**

- **简介: 该论文提出EyeSeg，一种用于AR/VR的不确定性感知眼部分割框架。任务是提升复杂场景下的眼部分割与注视估计效果。针对运动模糊、眼睑遮挡等问题，引入贝叶斯不确定性学习方法建模分割不确定性，并融合多估计结果增强鲁棒性，实现了更优性能表现。**

- **链接: [http://arxiv.org/pdf/2507.09649v1](http://arxiv.org/pdf/2507.09649v1)**

> **作者:** Zhengyuan Peng; Jianqing Xu; Shen Li; Jiazhen Ji; Yuge Huang; Jingyun Zhang; Jinmin Li; Shouhong Ding; Rizen Guo; Xin Tan; Lizhuang Ma
>
> **备注:** Accepted to IJCAI
>
> **摘要:** Human-machine interaction through augmented reality (AR) and virtual reality (VR) is increasingly prevalent, requiring accurate and efficient gaze estimation which hinges on the accuracy of eye segmentation to enable smooth user experiences. We introduce EyeSeg, a novel eye segmentation framework designed to overcome key challenges that existing approaches struggle with: motion blur, eyelid occlusion, and train-test domain gaps. In these situations, existing models struggle to extract robust features, leading to suboptimal performance. Noting that these challenges can be generally quantified by uncertainty, we design EyeSeg as an uncertainty-aware eye segmentation framework for AR/VR wherein we explicitly model the uncertainties by performing Bayesian uncertainty learning of a posterior under the closed set prior. Theoretically, we prove that a statistic of the learned posterior indicates segmentation uncertainty levels and empirically outperforms existing methods in downstream tasks, such as gaze estimation. EyeSeg outputs an uncertainty score and the segmentation result, weighting and fusing multiple gaze estimates for robustness, which proves to be effective especially under motion blur, eyelid occlusion and cross-domain challenges. Moreover, empirical results suggest that EyeSeg achieves segmentation improvements of MIoU, E1, F1, and ACC surpassing previous approaches. The code is publicly available at https://github.com/JethroPeng/EyeSeg.
>
---
#### [new 042] CKAA: Cross-subspace Knowledge Alignment and Aggregation for Robust Continual Learning
- **分类: cs.CV**

- **简介: 该论文属于持续学习任务，旨在解决参数高效微调方法中因子模块特征空间不一致导致的误判问题。作者提出CKAA框架，通过双层次知识对齐和基于任务置信度的适配器混合机制，提升模型在误导任务标识下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.09471v1](http://arxiv.org/pdf/2507.09471v1)**

> **作者:** Lingfeng He; De Cheng; Zhiheng Ma; Huaijie Wang; Dingwen Zhang; Nannan Wang; Xinbo Gao
>
> **摘要:** Continual Learning (CL) empowers AI models to continuously learn from sequential task streams. Recently, parameter-efficient fine-tuning (PEFT)-based CL methods have garnered increasing attention due to their superior performance. They typically allocate a unique sub-module for learning each task, with a task recognizer to select the appropriate sub-modules for testing images. However, due to the feature subspace misalignment from independently trained sub-modules, these methods tend to produce ambiguous decisions under misleading task-ids. To address this, we propose Cross-subspace Knowledge Alignment and Aggregation (CKAA), a novel framework that enhances model robustness against misleading task-ids through two key innovations: (1) Dual-level Knowledge Alignment (DKA): By aligning intra-class feature distributions across different subspaces and learning a robust global classifier through a feature simulation process, DKA enables the model to distinguish features from both correct and incorrect subspaces during training. (2) Task-Confidence-guided Mixture of Adapters (TC-MoA): A robust inference scheme that adaptively aggregates task-specific knowledge from relevant sub-modules based on task-confidence scores, avoiding overconfidence in misleading task-id predictions. Extensive experiments demonstrate that CKAA outperforms existing PEFT-based CL methods.
>
---
#### [new 043] Numerically Computing Galois Groups of Minimal Problems
- **分类: cs.CV; cs.SC; math.AG; 68W30**

- **简介: 论文属于代数与计算机视觉交叉任务，旨在解决参数化多项式方程组求解问题。该文探讨了通过数值方法计算Galois群以衡量问题复杂性，并应用于RanSaC等模型拟合方法中，致力于评估并简化此类问题的求解难度。**

- **链接: [http://arxiv.org/pdf/2507.10407v1](http://arxiv.org/pdf/2507.10407v1)**

> **作者:** Timothy Duff
>
> **备注:** abstract accompanying invited tutorial at ISSAC 2025; 10 pages w/ references
>
> **摘要:** I discuss a seemingly unlikely confluence of topics in algebra, numerical computation, and computer vision. The motivating problem is that of solving multiples instances of a parametric family of systems of algebraic (polynomial or rational function) equations. No doubt already of interest to ISSAC attendees, this problem arises in the context of robust model-fitting paradigms currently utilized by the computer vision community (namely "Random Sampling and Consensus", aka "RanSaC".) This talk will give an overview of work in the last 5+ years that aspires to measure the intrinsic difficulty of solving such parametric systems, and makes strides towards practical solutions.
>
---
#### [new 044] MCGA: Mixture of Codebooks Hyperspectral Reconstruction via Grayscale-Aware Attention
- **分类: cs.CV**

- **简介: 该论文属于图像处理任务，旨在解决从RGB图像重建高光谱图像（HSI）的问题。现有方法直接学习RGB到HSI的映射，忽视了低维到高维转换的挑战。为此，作者提出MCGA方法，分两阶段进行重建：第一阶段通过多尺度VQ-VAE提取混合码本（MoC），第二阶段利用Grayscale-Aware Attention和Quantized Self-Attention机制优化重建效果，并引入测试时自适应策略提升鲁棒性。实验表明该方法达到先进水平。**

- **链接: [http://arxiv.org/pdf/2507.09885v1](http://arxiv.org/pdf/2507.09885v1)**

> **作者:** Zhanjiang Yang; Lijun Sun; Jiawei Dong; Xiaoxin An; Yang Liu; Meng Li
>
> **摘要:** Reconstructing hyperspectral images (HSI) from RGB images is a cost-effective solution for various vision-based applications. However, most existing learning-based hyperspectral reconstruction methods directly learn the RGB-to-HSI mapping using complex attention mechanisms, neglecting the inherent challenge of transitioning from low-dimensional to high-dimensional information. To address this limitation, we propose a two-stage approach, MCGA, which first learns spectral patterns before estimating the mapping. In the first stage, a multi-scale VQ-VAE learns representations from heterogeneous HSI datasets, extracting a Mixture of Codebooks (MoC). In the second stage, the RGB-to-HSI mapping is refined by querying features from the MoC to replace latent HSI representations, incorporating prior knowledge rather than forcing a direct high-dimensional transformation. To further enhance reconstruction quality, we introduce Grayscale-Aware Attention and Quantized Self-Attention, which adaptively adjust feature map intensities to meet hyperspectral reconstruction requirements. This physically motivated attention mechanism ensures lightweight and efficient HSI recovery. Moreover, we propose an entropy-based Test-Time Adaptation strategy to improve robustness in real-world scenarios. Extensive experiments demonstrate that our method, MCGA, achieves state-of-the-art performance. The code and models will be released at https://github.com/Fibonaccirabbit/MCGA
>
---
#### [new 045] Can Contrastive Learning Improve Class-Imbalanced Diffusion Model?
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究在类别不平衡数据下，如何提升条件扩散模型生成尾部类别图像的多样性。通过引入两种对比学习损失：无监督InfoNCE损失和条件-无条件对齐的MSE损失，缓解尾类生成图像的模式崩溃问题，同时保持头部类别的生成质量。属于图像生成任务中的长尾分布学习问题。**

- **链接: [http://arxiv.org/pdf/2507.09052v1](http://arxiv.org/pdf/2507.09052v1)**

> **作者:** Fang Chen; Alex Villa; Gongbo Liang; Xiaoyi Lu; Meng Tang
>
> **备注:** 20 pages, 11 figures
>
> **摘要:** Training data for class-conditional image synthesis often exhibit a long-tailed distribution with limited images for tail classes. Such an imbalance causes mode collapse and reduces the diversity of synthesized images for tail classes. For class-conditional diffusion models trained on imbalanced data, we aim to improve the diversity of tail class images without compromising the fidelity and diversity of head class images. We achieve this by introducing two deceptively simple but highly effective contrastive loss functions. Firstly, we employ an unsupervised InfoNCE loss utilizing negative samples to increase the distance/dissimilarity among synthetic images, particularly for tail classes. To further enhance the diversity of tail classes, our second loss is an MSE loss that contrasts class-conditional generation with unconditional generation at large timesteps. This second loss makes the denoising process insensitive to class conditions for the initial steps, which enriches tail classes through knowledge sharing from head classes. Conditional-unconditional alignment has been shown to enhance the performance of long-tailed GAN. We are the first to adapt such alignment to diffusion models. We successfully leveraged contrastive learning for class-imbalanced diffusion models. Our contrastive learning framework is easy to implement and outperforms standard DDPM and alternative methods for class-imbalanced diffusion models across various datasets, including CIFAR10/100-LT, PlacesLT, TinyImageNetLT, and ImageNetLT.
>
---
#### [new 046] DEARLi: Decoupled Enhancement of Recognition and Localization for Semi-supervised Panoptic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于半监督全景分割任务，旨在解决像素级标注昂贵且耗时的问题。作者提出DEARLi方法，结合两个基础模型，通过解耦增强识别与定位，提升分割效果。在少量标注数据下表现优异，节省显存并超越现有技术。**

- **链接: [http://arxiv.org/pdf/2507.10118v1](http://arxiv.org/pdf/2507.10118v1)**

> **作者:** Ivan Martinović; Josip Šarić; Marin Oršić; Matej Kristan; Siniša Šegvić
>
> **备注:** ICCV 2025 Findings Workshop
>
> **摘要:** Pixel-level annotation is expensive and time-consuming. Semi-supervised segmentation methods address this challenge by learning models on few labeled images alongside a large corpus of unlabeled images. Although foundation models could further account for label scarcity, effective mechanisms for their exploitation remain underexplored. We address this by devising a novel semi-supervised panoptic approach fueled by two dedicated foundation models. We enhance recognition by complementing unsupervised mask-transformer consistency with zero-shot classification of CLIP features. We enhance localization by class-agnostic decoder warm-up with respect to SAM pseudo-labels. The resulting decoupled enhancement of recognition and localization (DEARLi) particularly excels in the most challenging semi-supervised scenarios with large taxonomies and limited labeled data. Moreover, DEARLi outperforms the state of the art in semi-supervised semantic segmentation by a large margin while requiring 8x less GPU memory, in spite of being trained only for the panoptic objective. We observe 29.9 PQ and 38.9 mIoU on ADE20K with only 158 labeled images. The source code is available at https://github.com/helen1c/DEARLi.
>
---
#### [new 047] GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space
- **分类: cs.CV**

- **简介: 该论文属于图像时空定位任务，旨在解决仅通过视觉信息联合预测图像拍摄时间与地理位置的问题。作者提出GT-Loc方法，通过统一嵌入空间实现图像、时间和位置的联合建模，并设计了基于时间周期性的度量学习目标，提升了时间预测与地理定位性能。**

- **链接: [http://arxiv.org/pdf/2507.10473v1](http://arxiv.org/pdf/2507.10473v1)**

> **作者:** David G. Shatwell; Ishan Rajendrakumar Dave; Sirnam Swetha; Mubarak Shah
>
> **备注:** Accepted in ICCV2025
>
> **摘要:** Timestamp prediction aims to determine when an image was captured using only visual information, supporting applications such as metadata correction, retrieval, and digital forensics. In outdoor scenarios, hourly estimates rely on cues like brightness, hue, and shadow positioning, while seasonal changes and weather inform date estimation. However, these visual cues significantly depend on geographic context, closely linking timestamp prediction to geo-localization. To address this interdependence, we introduce GT-Loc, a novel retrieval-based method that jointly predicts the capture time (hour and month) and geo-location (GPS coordinates) of an image. Our approach employs separate encoders for images, time, and location, aligning their embeddings within a shared high-dimensional feature space. Recognizing the cyclical nature of time, instead of conventional contrastive learning with hard positives and negatives, we propose a temporal metric-learning objective providing soft targets by modeling pairwise time differences over a cyclical toroidal surface. We present new benchmarks demonstrating that our joint optimization surpasses previous time prediction methods, even those using the ground-truth geo-location as an input during inference. Additionally, our approach achieves competitive results on standard geo-localization tasks, and the unified embedding space facilitates compositional and text-based image retrieval.
>
---
#### [new 048] Taming Modern Point Tracking for Speckle Tracking Echocardiography via Impartial Motion
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像处理任务，旨在解决超声心动图中组织跟踪不准确的问题。通过改进训练策略和引入轻量网络结构，提升了点跟踪方法在心脏运动估计中的准确性与泛化能力，从而提高临床测量的可靠性。**

- **链接: [http://arxiv.org/pdf/2507.10127v1](http://arxiv.org/pdf/2507.10127v1)**

> **作者:** Md Abulkalam Azad; John Nyberg; Håvard Dalen; Bjørnar Grenne; Lasse Lovstakken; Andreas Østvik
>
> **备注:** Accepted to CVAMD workshop at ICCV 2025
>
> **摘要:** Accurate motion estimation for tracking deformable tissues in echocardiography is essential for precise cardiac function measurements. While traditional methods like block matching or optical flow struggle with intricate cardiac motion, modern point tracking approaches remain largely underexplored in this domain. This work investigates the potential of state-of-the-art (SOTA) point tracking methods for ultrasound, with a focus on echocardiography. Although these novel approaches demonstrate strong performance in general videos, their effectiveness and generalizability in echocardiography remain limited. By analyzing cardiac motion throughout the heart cycle in real B-mode ultrasound videos, we identify that a directional motion bias across different views is affecting the existing training strategies. To mitigate this, we refine the training procedure and incorporate a set of tailored augmentations to reduce the bias and enhance tracking robustness and generalization through impartial cardiac motion. We also propose a lightweight network leveraging multi-scale cost volumes from spatial context alone to challenge the advanced spatiotemporal point tracking models. Experiments demonstrate that fine-tuning with our strategies significantly improves models' performances over their baselines, even for out-of-distribution (OOD) cases. For instance, EchoTracker boosts overall position accuracy by 60.7% and reduces median trajectory error by 61.5% across heart cycle phases. Interestingly, several point tracking models fail to outperform our proposed simple model in terms of tracking accuracy and generalization, reflecting their limitations when applied to echocardiography. Nevertheless, clinical evaluation reveals that these methods improve GLS measurements, aligning more closely with expert-validated, semi-automated tools and thus demonstrating better reproducibility in real-world applications.
>
---
#### [new 049] Uncertainty Quantification for Incomplete Multi-View Data Using Divergence Measures
- **分类: cs.CV**

- **简介: 该论文属于多视图分类与聚类任务，旨在解决不完整多视图数据下的不确定性量化问题。现有方法使用KL散度忽略模态差异，可靠性不足。作者提出KPHD-Net，基于Hölder散度和Dempster-Shafer证据理论，结合Kalman滤波提升融合效果，并通过Dirichlet分布建模证据，增强不确定估计的准确性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.09980v1](http://arxiv.org/pdf/2507.09980v1)**

> **作者:** Zhipeng Xue; Yan Zhang; Ming Li; Chun Li; Yue Liu; Fei Yu
>
> **摘要:** Existing multi-view classification and clustering methods typically improve task accuracy by leveraging and fusing information from different views. However, ensuring the reliability of multi-view integration and final decisions is crucial, particularly when dealing with noisy or corrupted data. Current methods often rely on Kullback-Leibler (KL) divergence to estimate uncertainty of network predictions, ignoring domain gaps between different modalities. To address this issue, KPHD-Net, based on H\"older divergence, is proposed for multi-view classification and clustering tasks. Generally, our KPHD-Net employs a variational Dirichlet distribution to represent class probability distributions, models evidences from different views, and then integrates it with Dempster-Shafer evidence theory (DST) to improve uncertainty estimation effects. Our theoretical analysis demonstrates that Proper H\"older divergence offers a more effective measure of distribution discrepancies, ensuring enhanced performance in multi-view learning. Moreover, Dempster-Shafer evidence theory, recognized for its superior performance in multi-view fusion tasks, is introduced and combined with the Kalman filter to provide future state estimations. This integration further enhances the reliability of the final fusion results. Extensive experiments show that the proposed KPHD-Net outperforms the current state-of-the-art methods in both classification and clustering tasks regarding accuracy, robustness, and reliability, with theoretical guarantees.
>
---
#### [new 050] National level satellite-based crop field inventories in smallholder landscapes
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于遥感与农业制图任务，旨在解决小农户地区农田分布与规模不清的问题。利用高分辨率卫星数据和深度迁移学习，绘制了莫桑比克全国2100万个农田的分布图，准确率达93%，并分析了田块大小与社会经济及环境因素的关系。**

- **链接: [http://arxiv.org/pdf/2507.10499v1](http://arxiv.org/pdf/2507.10499v1)**

> **作者:** Philippe Rufin; Pauline Lucie Hammer; Leon-Friedrich Thomas; Sá Nogueira Lisboa; Natasha Ribeiro; Almeida Sitoe; Patrick Hostert; Patrick Meyfroidt
>
> **摘要:** The design of science-based policies to improve the sustainability of smallholder agriculture is challenged by a limited understanding of fundamental system properties, such as the spatial distribution of active cropland and field size. We integrate very high spatial resolution (1.5 m) Earth observation data and deep transfer learning to derive crop field delineations in complex agricultural systems at the national scale, while maintaining minimum reference data requirements and enhancing transferability. We provide the first national-level dataset of 21 million individual fields for Mozambique (covering ~800,000 km2) for 2023. Our maps separate active cropland from non-agricultural land use with an overall accuracy of 93% and balanced omission and commission errors. Field-level spatial agreement reached median intersection over union (IoU) scores of 0.81, advancing the state-of-the-art in large-area field delineation in complex smallholder systems. The active cropland maps capture fragmented rural regions with low cropland shares not yet identified in global land cover or cropland maps. These regions are mostly located in agricultural frontier regions which host 7-9% of the Mozambican population. Field size in Mozambique is very low overall, with half of the fields being smaller than 0.16 ha, and 83% smaller than 0.5 ha. Mean field size at aggregate spatial resolution (0.05{\deg}) is 0.32 ha, but it varies strongly across gradients of accessibility, population density, and net forest cover change. This variation reflects a diverse set of actors, ranging from semi-subsistence smallholder farms to medium-scale commercial farming, and large-scale farming operations. Our results highlight that field size is a key indicator relating to socio-economic and environmental outcomes of agriculture (e.g., food production, livelihoods, deforestation, biodiversity), as well as their trade-offs.
>
---
#### [new 051] 3DGAA: Realistic and Robust 3D Gaussian-based Adversarial Attack for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶安全任务，旨在解决目标检测系统在真实环境中的对抗攻击脆弱性问题。作者提出3DGAA，利用3D高斯散射参数化方法，联合优化几何与外观属性，生成物理可实现的对抗物体，提升攻击的现实性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.09993v1](http://arxiv.org/pdf/2507.09993v1)**

> **作者:** Yixun Zhang; Lizhi Wang; Junjun Zhao; Wending Zhao; Feng Zhou; Yonghao Dang; Jianqin Yin
>
> **备注:** Submitted to WACV 2026
>
> **摘要:** Camera-based object detection systems play a vital role in autonomous driving, yet they remain vulnerable to adversarial threats in real-world environments. While existing 2D and 3D physical attacks typically optimize texture, they often struggle to balance physical realism and attack robustness. In this work, we propose 3D Gaussian-based Adversarial Attack (3DGAA), a novel adversarial object generation framework that leverages the full 14-dimensional parameterization of 3D Gaussian Splatting (3DGS) to jointly optimize geometry and appearance in physically realizable ways. Unlike prior works that rely on patches or texture, 3DGAA jointly perturbs both geometric attributes (shape, scale, rotation) and appearance attributes (color, opacity) to produce physically realistic and transferable adversarial objects. We further introduce a physical filtering module to preserve geometric fidelity, and a physical augmentation module to simulate complex physical scenarios, thus enhancing attack generalization under real-world conditions. We evaluate 3DGAA on both virtual benchmarks and physical-world setups using miniature vehicle models. Experimental results show that 3DGAA achieves to reduce the detection mAP from 87.21% to 7.38%, significantly outperforming existing 3D physical attacks. Moreover, our method maintains high transferability across different physical conditions, demonstrating a new state-of-the-art in physically realizable adversarial attacks. These results validate 3DGAA as a practical attack framework for evaluating the safety of perception systems in autonomous driving.
>
---
#### [new 052] Pairwise Alignment & Compatibility for Arbitrarily Irregular Image Fragments
- **分类: cs.CV**

- **简介: 该论文属于图像拼接任务，旨在解决不规则碎片的匹配与对齐问题。现有方法受限于碎片形状，而本文提出一种几何与图像结合的混合方法，适用于任意形状的碎片，并构建了新数据集与评估指标，提升了考古拼图中的兼容性计算性能。**

- **链接: [http://arxiv.org/pdf/2507.09767v1](http://arxiv.org/pdf/2507.09767v1)**

> **作者:** Ofir Itzhak Shahar; Gur Elkin; Ohad Ben-Shahar
>
> **摘要:** Pairwise compatibility calculation is at the core of most fragments-reconstruction algorithms, in particular those designed to solve different types of the jigsaw puzzle problem. However, most existing approaches fail, or aren't designed to deal with fragments of realistic geometric properties one encounters in real-life puzzles. And in all other cases, compatibility methods rely strongly on the restricted shapes of the fragments. In this paper, we propose an efficient hybrid (geometric and pictorial) approach for computing the optimal alignment for pairs of fragments, without any assumptions about their shapes, dimensions, or pictorial content. We introduce a new image fragments dataset generated via a novel method for image fragmentation and a formal erosion model that mimics real-world archaeological erosion, along with evaluation metrics for the compatibility task. We then embed our proposed compatibility into an archaeological puzzle-solving framework and demonstrate state-of-the-art neighborhood-level precision and recall on the RePAIR 2D dataset, directly reflecting compatibility performance improvements.
>
---
#### [new 053] Generative Latent Kernel Modeling for Blind Motion Deblurring
- **分类: cs.CV**

- **简介: 该论文属于盲运动去模糊（BMD）任务，旨在解决传统方法对模糊核初始化敏感的问题。作者提出了一种基于生成对抗网络（GAN）的新框架，通过预训练核生成器和初始化器，提供更优的初始模糊核，从而提升BMD的稳定性和性能，并可扩展至非均匀运动去模糊，达到先进效果。**

- **链接: [http://arxiv.org/pdf/2507.09285v1](http://arxiv.org/pdf/2507.09285v1)**

> **作者:** Chenhao Ding; Jiangtao Zhang; Zongsheng Yue; Hui Wang; Qian Zhao; Deyu Meng
>
> **摘要:** Deep prior-based approaches have demonstrated remarkable success in blind motion deblurring (BMD) recently. These methods, however, are often limited by the high non-convexity of the underlying optimization process in BMD, which leads to extreme sensitivity to the initial blur kernel. To address this issue, we propose a novel framework for BMD that leverages a deep generative model to encode the kernel prior and induce a better initialization for the blur kernel. Specifically, we pre-train a kernel generator based on a generative adversarial network (GAN) to aptly characterize the kernel's prior distribution, as well as a kernel initializer to provide a well-informed and high-quality starting point for kernel estimation. By combining these two components, we constrain the BMD solution within a compact latent kernel manifold, thus alleviating the aforementioned sensitivity for kernel initialization. Notably, the kernel generator and initializer are designed to be easily integrated with existing BMD methods in a plug-and-play manner, enhancing their overall performance. Furthermore, we extend our approach to tackle blind non-uniform motion deblurring without the need for additional priors, achieving state-of-the-art performance on challenging benchmark datasets. The source code is available at https://github.com/dch0319/GLKM-Deblur.
>
---
#### [new 054] FTCFormer: Fuzzy Token Clustering Transformer for Image Classification
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决现有视觉Transformer忽视图像区域语义信息的问题。作者提出FTCFormer，通过语义驱动的模糊聚类生成视觉token，优化特征表示，提升了分类性能。**

- **链接: [http://arxiv.org/pdf/2507.10283v1](http://arxiv.org/pdf/2507.10283v1)**

> **作者:** Muyi Bao; Changyu Zeng; Yifan Wang; Zhengni Yang; Zimu Wang; Guangliang Cheng; Jun Qi; Wei Wang
>
> **摘要:** Transformer-based deep neural networks have achieved remarkable success across various computer vision tasks, largely attributed to their long-range self-attention mechanism and scalability. However, most transformer architectures embed images into uniform, grid-based vision tokens, neglecting the underlying semantic meanings of image regions, resulting in suboptimal feature representations. To address this issue, we propose Fuzzy Token Clustering Transformer (FTCFormer), which incorporates a novel clustering-based downsampling module to dynamically generate vision tokens based on the semantic meanings instead of spatial positions. It allocates fewer tokens to less informative regions and more to represent semantically important regions, regardless of their spatial adjacency or shape irregularity. To further enhance feature extraction and representation, we propose a Density Peak Clustering-Fuzzy K-Nearest Neighbor (DPC-FKNN) mechanism for clustering center determination, a Spatial Connectivity Score (SCS) for token assignment, and a channel-wise merging (Cmerge) strategy for token merging. Extensive experiments on 32 datasets across diverse domains validate the effectiveness of FTCFormer on image classification, showing consistent improvements over the TCFormer baseline, achieving gains of improving 1.43% on five fine-grained datasets, 1.09% on six natural image datasets, 0.97% on three medical datasets and 0.55% on four remote sensing datasets. The code is available at: https://github.com/BaoBao0926/FTCFormer/tree/main.
>
---
#### [new 055] SpeakerVid-5M: A Large-Scale High-Quality Dataset for Audio-Visual Dyadic Interactive Human Generation
- **分类: cs.CV; eess.AS**

- **简介: 该论文属于音频-视觉虚拟人类生成任务，旨在解决缺乏大规模高质量数据的问题。论文构建了SpeakerVid-5M数据集，包含520万视频片段，覆盖多种交互类型，并提供预训练和高质量微调子集。此外，还提出了基于自回归模型的视频聊天基线和评估基准，推动相关研究发展。**

- **链接: [http://arxiv.org/pdf/2507.09862v1](http://arxiv.org/pdf/2507.09862v1)**

> **作者:** Youliang Zhang; Zhaoyang Li; Duomin Wang; Jiahe Zhang; Deyu Zhou; Zixin Yin; Xili Dai; Gang Yu; Xiu Li
>
> **摘要:** The rapid development of large-scale models has catalyzed significant breakthroughs in the digital human domain. These advanced methodologies offer high-fidelity solutions for avatar driving and rendering, leading academia to focus on the next major challenge: audio-visual dyadic interactive virtual human. To facilitate research in this emerging area, we present SpeakerVid-5M dataset, the first large-scale, high-quality dataset designed for audio-visual dyadic interactive virtual human generation. Totaling over 8,743 hours, SpeakerVid-5M contains more than 5.2 million video clips of human portraits. It covers diverse scales and interaction types, including monadic talking, listening, and dyadic conversations. Crucially, the dataset is structured along two key dimensions: interaction type and data quality. First, it is categorized into four types (dialogue branch, single branch, listening branch and multi-turn branch) based on the interaction scenario. Second, it is stratified into a large-scale pre-training subset and a curated, high-quality subset for Supervised Fine-Tuning (SFT). This dual structure accommodates a wide array of 2D virtual human tasks. In addition, we provide an autoregressive (AR)-based video chat baseline trained on this data, accompanied by a dedicated set of metrics and test data to serve as a benchmark VidChatBench for future work. Both the dataset and the corresponding data processing code will be publicly released. Project page: https://dorniwang.github.io/SpeakerVid-5M/
>
---
#### [new 056] Latent Diffusion Models with Masked AutoEncoders
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决现有潜在扩散模型（LDMs）中自编码器设计不足的问题。作者提出了一种新的自编码器结构——变分掩码自编码器（VMAEs），并将其集成到LDM框架中，形成LDMAEs模型，以提升图像生成质量和计算效率。**

- **链接: [http://arxiv.org/pdf/2507.09984v1](http://arxiv.org/pdf/2507.09984v1)**

> **作者:** Junho Lee; Jeongwoo Shin; Hyungwook Choi; Joonseok Lee
>
> **摘要:** In spite of remarkable potential of the Latent Diffusion Models (LDMs) in image generation, the desired properties and optimal design of the autoencoders have been underexplored. In this work, we analyze the role of autoencoders in LDMs and identify three key properties: latent smoothness, perceptual compression quality, and reconstruction quality. We demonstrate that existing autoencoders fail to simultaneously satisfy all three properties, and propose Variational Masked AutoEncoders (VMAEs), taking advantage of the hierarchical features maintained by Masked AutoEncoder. We integrate VMAEs into the LDM framework, introducing Latent Diffusion Models with Masked AutoEncoders (LDMAEs). Through comprehensive experiments, we demonstrate significantly enhanced image generation quality and computational efficiency.
>
---
#### [new 057] Vision-Based Anti Unmanned Aerial Technology: Opportunities and Challenges
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉与无人机防御任务，旨在解决复杂环境中高效精准追踪无人机的问题。论文综述了主流视觉及多传感器融合的检测跟踪算法，整理了公开数据集，并分析了未来研究方向。**

- **链接: [http://arxiv.org/pdf/2507.10006v1](http://arxiv.org/pdf/2507.10006v1)**

> **作者:** Guanghai Ding; Yihua Ren; Yuting Liu; Qijun Zhao; Shuiwang Li
>
> **摘要:** With the rapid advancement of UAV technology and its extensive application in various fields such as military reconnaissance, environmental monitoring, and logistics, achieving efficient and accurate Anti-UAV tracking has become essential. The importance of Anti-UAV tracking is increasingly prominent, especially in scenarios such as public safety, border patrol, search and rescue, and agricultural monitoring, where operations in complex environments can provide enhanced security. Current mainstream Anti-UAV tracking technologies are primarily centered around computer vision techniques, particularly those that integrate multi-sensor data fusion with advanced detection and tracking algorithms. This paper first reviews the characteristics and current challenges of Anti-UAV detection and tracking technologies. Next, it investigates and compiles several publicly available datasets, providing accessible links to support researchers in efficiently addressing related challenges. Furthermore, the paper analyzes the major vision-based and vision-fusion-based Anti-UAV detection and tracking algorithms proposed in recent years. Finally, based on the above research, this paper outlines future research directions, aiming to provide valuable insights for advancing the field.
>
---
#### [new 058] When Schrödinger Bridge Meets Real-World Image Dehazing with Unpaired Training
- **分类: cs.CV**

- **简介: 该论文属于图像去雾任务，旨在解决真实场景中无配对训练数据的去雾问题。现有GAN方法受限于生成器的映射能力。作者提出DehazeSB，基于Schrödinger Bridge理论实现更优分布映射，并引入细节保持正则化与提示学习，以提升去雾效果与结构一致性。**

- **链接: [http://arxiv.org/pdf/2507.09524v1](http://arxiv.org/pdf/2507.09524v1)**

> **作者:** Yunwei Lan; Zhigao Cui; Xin Luo; Chang Liu; Nian Wang; Menglin Zhang; Yanzhao Su; Dong Liu
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Recent advancements in unpaired dehazing, particularly those using GANs, show promising performance in processing real-world hazy images. However, these methods tend to face limitations due to the generator's limited transport mapping capability, which hinders the full exploitation of their effectiveness in unpaired training paradigms. To address these challenges, we propose DehazeSB, a novel unpaired dehazing framework based on the Schr\"odinger Bridge. By leveraging optimal transport (OT) theory, DehazeSB directly bridges the distributions between hazy and clear images. This enables optimal transport mappings from hazy to clear images in fewer steps, thereby generating high-quality results. To ensure the consistency of structural information and details in the restored images, we introduce detail-preserving regularization, which enforces pixel-level alignment between hazy inputs and dehazed outputs. Furthermore, we propose a novel prompt learning to leverage pre-trained CLIP models in distinguishing hazy images and clear ones, by learning a haze-aware vision-language alignment. Extensive experiments on multiple real-world datasets demonstrate our method's superiority. Code: https://github.com/ywxjm/DehazeSB.
>
---
#### [new 059] EHPE: A Segmented Architecture for Enhanced Hand Pose Estimation
- **分类: cs.CV**

- **简介: 该论文属于3D手部姿态估计任务，旨在解决现有方法对远端关节（如指尖和手腕）估计不准及误差累积导致姿态重建质量下降的问题。论文提出了EHPE分段架构，通过先提取指尖和手腕关节，再引导估计其余关节，有效减少了预测误差，提升了整体手部姿态估计的精度。**

- **链接: [http://arxiv.org/pdf/2507.09560v1](http://arxiv.org/pdf/2507.09560v1)**

> **作者:** Bolun Zheng; Xinjie Liu; Qianyu Zhang; Canjin Wang; Fangni Chen; Mingen Xu
>
> **摘要:** 3D hand pose estimation has garnered great attention in recent years due to its critical applications in human-computer interaction, virtual reality, and related fields. The accurate estimation of hand joints is essential for high-quality hand pose estimation. However, existing methods neglect the importance of Distal Phalanx Tip (TIP) and Wrist in predicting hand joints overall and often fail to account for the phenomenon of error accumulation for distal joints in gesture estimation, which can cause certain joints to incur larger errors, resulting in misalignments and artifacts in the pose estimation and degrading the overall reconstruction quality. To address this challenge, we propose a novel segmented architecture for enhanced hand pose estimation (EHPE). We perform local extraction of TIP and wrist, thus alleviating the effect of error accumulation on TIP prediction and further reduce the predictive errors for all joints on this basis. EHPE consists of two key stages: In the TIP and Wrist Joints Extraction stage (TW-stage), the positions of the TIP and wrist joints are estimated to provide an initial accurate joint configuration; In the Prior Guided Joints Estimation stage (PG-stage), a dual-branch interaction network is employed to refine the positions of the remaining joints. Extensive experiments on two widely used benchmarks demonstrate that EHPE achieves state-of-the-arts performance. Code is available at https://github.com/SereinNout/EHPE.
>
---
#### [new 060] Token Compression Meets Compact Vision Transformers: A Survey and Comparative Evaluation for Edge AI
- **分类: cs.CV**

- **简介: 论文属于计算机视觉任务，旨在解决视觉Transformer在边缘设备上的推理加速问题。系统调研并比较了多种令牌压缩方法，在标准与紧凑型ViT上进行评估，发现现有方法在紧凑模型上效果有限，为未来研究提供了方向。**

- **链接: [http://arxiv.org/pdf/2507.09702v1](http://arxiv.org/pdf/2507.09702v1)**

> **作者:** Phat Nguyen; Ngai-Man Cheung
>
> **摘要:** Token compression techniques have recently emerged as powerful tools for accelerating Vision Transformer (ViT) inference in computer vision. Due to the quadratic computational complexity with respect to the token sequence length, these methods aim to remove less informative tokens before the attention layers to improve inference throughput. While numerous studies have explored various accuracy-efficiency trade-offs on large-scale ViTs, two critical gaps remain. First, there is a lack of unified survey that systematically categorizes and compares token compression approaches based on their core strategies (e.g., pruning, merging, or hybrid) and deployment settings (e.g., fine-tuning vs. plug-in). Second, most benchmarks are limited to standard ViT models (e.g., ViT-B, ViT-L), leaving open the question of whether such methods remain effective when applied to structurally compressed transformers, which are increasingly deployed on resource-constrained edge devices. To address these gaps, we present the first systematic taxonomy and comparative study of token compression methods, and we evaluate representative techniques on both standard and compact ViT architectures. Our experiments reveal that while token compression methods are effective for general-purpose ViTs, they often underperform when directly applied to compact designs. These findings not only provide practical insights but also pave the way for future research on adapting token optimization techniques to compact transformer-based networks for edge AI and AI agent applications.
>
---
#### [new 061] Crucial-Diff: A Unified Diffusion Model for Crucial Image and Annotation Synthesis in Data-scarce Scenarios
- **分类: cs.CV**

- **简介: 该论文属于图像生成与数据增强任务，旨在解决数据稀缺场景下的模型过拟合与数据不平衡问题。论文提出Crucial-Diff框架，通过结合场景无关特征提取与弱点感知样本生成，统一合成高质量关键样本，提升检测与分割性能。**

- **链接: [http://arxiv.org/pdf/2507.09915v1](http://arxiv.org/pdf/2507.09915v1)**

> **作者:** Siyue Yao; Mingjie Sun; Eng Gee Lim; Ran Yi; Baojiang Zhong; Moncef Gabbouj
>
> **摘要:** The scarcity of data in various scenarios, such as medical, industry and autonomous driving, leads to model overfitting and dataset imbalance, thus hindering effective detection and segmentation performance. Existing studies employ the generative models to synthesize more training samples to mitigate data scarcity. However, these synthetic samples are repetitive or simplistic and fail to provide "crucial information" that targets the downstream model's weaknesses. Additionally, these methods typically require separate training for different objects, leading to computational inefficiencies. To address these issues, we propose Crucial-Diff, a domain-agnostic framework designed to synthesize crucial samples. Our method integrates two key modules. The Scene Agnostic Feature Extractor (SAFE) utilizes a unified feature extractor to capture target information. The Weakness Aware Sample Miner (WASM) generates hard-to-detect samples using feedback from the detection results of downstream model, which is then fused with the output of SAFE module. Together, our Crucial-Diff framework generates diverse, high-quality training data, achieving a pixel-level AP of 83.63% and an F1-MAX of 78.12% on MVTec. On polyp dataset, Crucial-Diff reaches an mIoU of 81.64% and an mDice of 87.69%. Code will be released after acceptance.
>
---
#### [new 062] Is Micro-expression Ethnic Leaning?
- **分类: cs.CV**

- **简介: 该论文研究微表情识别中的民族倾向性问题，挑战了传统认为情感表达具有跨文化一致性的假设。作者构建了一个跨文化的微表情数据库，并提出一种融合民族背景特征的识别框架，通过实验验证民族差异对微表情识别的影响，旨在提升识别的准确性和公平性。**

- **链接: [http://arxiv.org/pdf/2507.10209v1](http://arxiv.org/pdf/2507.10209v1)**

> **作者:** Huai-Qian Khor; Yante Li; Xingxun Jiang; Guoying Zhao
>
> **摘要:** How much does ethnicity play its part in emotional expression? Emotional expression and micro-expression research probe into understanding human psychological responses to emotional stimuli, thereby revealing substantial hidden yet authentic emotions that can be useful in the event of diagnosis and interviews. While increased attention had been provided to micro-expression analysis, the studies were done under Ekman's assumption of emotion universality, where emotional expressions are identical across cultures and social contexts. Our computational study uncovers some of the influences of ethnic background in expression analysis, leading to an argument that the emotional universality hypothesis is an overgeneralization from the perspective of manual psychological analysis. In this research, we propose to investigate the level of influence of ethnicity in a simulated micro-expression scenario. We construct a cross-cultural micro-expression database and algorithmically annotate the ethnic labels to facilitate the investigation. With the ethnically annotated dataset, we perform a prima facie study to compare mono-ethnicity and stereo-ethnicity in a controlled environment, which uncovers a certain influence of ethnic bias via an experimental way. Building on this finding, we propose a framework that integrates ethnic context into the emotional feature learning process, yielding an ethnically aware framework that recognises ethnicity differences in micro-expression recognition. For improved understanding, qualitative analyses have been done to solidify the preliminary investigation into this new realm of research. Code is publicly available at https://github.com/IcedDoggie/ICMEW2025_EthnicMER
>
---
#### [new 063] Detecting Deepfake Talking Heads from Facial Biometric Anomalies
- **分类: cs.CV**

- **简介: 该论文属于图像取证任务，旨在检测深度伪造的说话头像视频。通过分析面部生物特征中的异常模式，提出一种新的机器学习方法，以识别伪造内容，并评估其在多种攻击下的鲁棒性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.08917v1](http://arxiv.org/pdf/2507.08917v1)**

> **作者:** Justin D. Norman; Hany Farid
>
> **备注:** 10 pages, 3 figures, 3 tables
>
> **摘要:** The combination of highly realistic voice cloning, along with visually compelling avatar, face-swap, or lip-sync deepfake video generation, makes it relatively easy to create a video of anyone saying anything. Today, such deepfake impersonations are often used to power frauds, scams, and political disinformation. We propose a novel forensic machine learning technique for the detection of deepfake video impersonations that leverages unnatural patterns in facial biometrics. We evaluate this technique across a large dataset of deepfake techniques and impersonations, as well as assess its reliability to video laundering and its generalization to previously unseen video deepfake generators.
>
---
#### [new 064] Mind the Gap: Preserving and Compensating for the Modality Gap in CLIP-Based Continual Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于持续学习任务，旨在解决CLIP模型在类增量学习中遗忘旧任务和适应新数据的问题。作者分析了模态差距变化，提出MG-CLIP方法，通过保留和补偿模态差距来缓解遗忘并提升性能，无需额外回放数据。**

- **链接: [http://arxiv.org/pdf/2507.09118v1](http://arxiv.org/pdf/2507.09118v1)**

> **作者:** Linlan Huang; Xusheng Cao; Haori Lu; Yifan Meng; Fei Yang; Xialei Liu
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Continual learning aims to enable models to learn sequentially from continuously incoming data while retaining performance on previously learned tasks. With the Contrastive Language-Image Pre-trained model (CLIP) exhibiting strong capabilities across various downstream tasks, there has been growing interest in leveraging CLIP for continual learning in such scenarios. Most existing works overlook the inherent modality gap in CLIP, a key factor in its generalization and adaptability. In this paper, we analyze the variations in the modality gap during the fine-tuning of vision-language pre-trained models. Our observations reveal that the modality gap effectively reflects the extent to which pre-trained knowledge is preserved. Based on these insights, we propose a simple yet effective method, MG-CLIP, that improves CLIP's performance in class-incremental learning. Our approach leverages modality gap preservation to mitigate forgetting and modality gap compensation to enhance the capacity for new data, introducing a novel modality-gap-based perspective for continual learning. Extensive experiments on multiple benchmarks demonstrate that our method outperforms existing approaches without requiring additional replay data. Our code is available at https://github.com/linlany/MindtheGap.
>
---
#### [new 065] BlindSight: Harnessing Sparsity for Efficient VLMs
- **分类: cs.CV; I.2.10**

- **简介: 论文提出BlindSight，通过利用注意力机制中的稀疏性，优化视觉语言模型（VLMs）的推理效率。该方法在不影响准确率的前提下，显著减少计算量，适用于多图像理解任务。**

- **链接: [http://arxiv.org/pdf/2507.09071v1](http://arxiv.org/pdf/2507.09071v1)**

> **作者:** Tharun Adithya Srikrishnan; Deval Shah; Steven K. Reinhardt
>
> **摘要:** Large vision-language models (VLMs) enable the joint processing of text and images. However, the inclusion of vision data significantly expands the prompt length. Along with the quadratic complexity of the attention computation, this results in a longer prefill duration. An approach to mitigate this bottleneck is to leverage the inherent sparsity in the attention computation. In our analysis of attention patterns in VLMs, we observe that a substantial portion of layers exhibit minimal cross-image attention, except through attention-sink tokens per image. These sparse attention patterns fall into distinct categories: sink-only, document mask and a hybrid document-sink mask. Based on this, we propose BlindSight: a training-free approach to optimize VLM inference using a input template-aware attention sparsity mask. We utilize samples from a dataset to derive a prompt-agnostic sparsity categorization for every attention head. We evaluate the proposed technique using VLMs such as Qwen2-VL, Qwen2.5-VL and Gemma-3. BlindSight results in a 32%-41% reduction in FLOPs on average with -2%-+2% accuracy compared to the original model in most evaluated multi-image understanding benchmarks.
>
---
#### [new 066] ExpStar: Towards Automatic Commentary Generation for Multi-discipline Scientific Experiments
- **分类: cs.CV**

- **简介: 该论文提出ExpStar模型与ExpInstruct数据集，旨在解决多学科科学实验中自动生成详细解说的任务。现有大模型在理解实验步骤、科学原理与安全规范方面不足。论文通过构建包含7K样本的数据集，并引入检索增强机制，显著提升生成效果，优于14种主流模型。**

- **链接: [http://arxiv.org/pdf/2507.09693v1](http://arxiv.org/pdf/2507.09693v1)**

> **作者:** Jiali Chen; Yujie Jia; Zihan Wu; Jinyu Yang; Jianpeng Chen; Xusen Hei; Jiayuan Xie; Yi Cai; Qing Li
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Experiment commentary is crucial in describing the experimental procedures, delving into underlying scientific principles, and incorporating content-related safety guidelines. In practice, human teachers rely heavily on subject-specific expertise and invest significant time preparing such commentary. To address this challenge, we introduce the task of automatic commentary generation across multi-discipline scientific experiments. While recent progress in large multimodal models (LMMs) has demonstrated promising capabilities in video understanding and reasoning, their ability to generate fine-grained and insightful experiment commentary remains largely underexplored. In this paper, we make the following contributions: (i) We construct \textit{ExpInstruct}, the first dataset tailored for experiment commentary generation, featuring over 7\textit{K} step-level commentaries across 21 scientific subjects from 3 core disciplines (\ie, science, healthcare and engineering). Each sample includes procedural descriptions along with potential scientific principles (\eg, chemical equations and physical laws) and safety guidelines. (ii) We propose ExpStar, an automatic experiment commentary generation model that leverages a retrieval-augmented mechanism to adaptively access, evaluate, and utilize external knowledge. (iii) Extensive experiments show that our ExpStar substantially outperforms 14 leading LMMs, which highlights the superiority of our dataset and model. We believe that ExpStar holds great potential for advancing AI-assisted scientific experiment instruction.
>
---
#### [new 067] EgoAnimate: Generating Human Animations from Egocentric top-down Views
- **分类: cs.CV**

- **简介: 该论文属于数字虚拟人任务，旨在解决从单目第一视角图像生成可驱动的高质量人体动画问题。利用Stable Diffusion和ControlNet技术，提出EgoAnimate方法，将顶部视角图像转换为正面视图，实现基于最小输入的虚拟人动作生成。**

- **链接: [http://arxiv.org/pdf/2507.09230v1](http://arxiv.org/pdf/2507.09230v1)**

> **作者:** G. Kutay Türkoglu; Julian Tanke; Iheb Belgacem; Lev Markhasin
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** An ideal digital telepresence experience requires accurate replication of a person's body, clothing, and movements. To capture and transfer these movements into virtual reality, the egocentric (first-person) perspective can be adopted, which enables the use of a portable and cost-effective device without front-view cameras. However, this viewpoint introduces challenges such as occlusions and distorted body proportions. There are few works reconstructing human appearance from egocentric views, and none use a generative prior-based approach. Some methods create avatars from a single egocentric image during inference, but still rely on multi-view datasets during training. To our knowledge, this is the first study using a generative backbone to reconstruct animatable avatars from egocentric inputs. Based on Stable Diffusion, our method reduces training burden and improves generalizability. Inspired by methods such as SiTH and MagicMan, which perform 360-degree reconstruction from a frontal image, we introduce a pipeline that generates realistic frontal views from occluded top-down images using ControlNet and a Stable Diffusion backbone. Our goal is to convert a single top-down egocentric image into a realistic frontal representation and feed it into an image-to-motion model. This enables generation of avatar motions from minimal input, paving the way for more accessible and generalizable telepresence systems.
>
---
#### [new 068] DisCo: Towards Distinct and Coherent Visual Encapsulation in Video MLLMs
- **分类: cs.CV**

- **简介: 该论文属于视频多模态大语言模型任务，旨在解决现有视觉封装方法在语义区分性和时间连贯性上的不足。作者提出了DisCo方法，包含视觉概念判别模块和时间焦点校准模块，以提升视频理解性能并提高令牌效率。**

- **链接: [http://arxiv.org/pdf/2507.10302v1](http://arxiv.org/pdf/2507.10302v1)**

> **作者:** Jiahe Zhao; Rongkun Zheng; Yi Wang; Helin Wang; Hengshuang Zhao
>
> **备注:** ICCV 2025
>
> **摘要:** In video Multimodal Large Language Models (video MLLMs), the visual encapsulation process plays a pivotal role in converting video contents into representative tokens for LLM input. While linear projectors are widely employed for encapsulation, they introduce semantic indistinctness and temporal incoherence when applied to videos. Conversely, the structure of resamplers shows promise in tackling these challenges, but an effective solution remains unexplored. Drawing inspiration from resampler structures, we introduce DisCo, a novel visual encapsulation method designed to yield semantically distinct and temporally coherent visual tokens for video MLLMs. DisCo integrates two key components: (1) A Visual Concept Discriminator (VCD) module, assigning unique semantics for visual tokens by associating them in pair with discriminative concepts in the video. (2) A Temporal Focus Calibrator (TFC) module, ensuring consistent temporal focus of visual tokens to video elements across every video frame. Through extensive experiments on multiple video MLLM frameworks, we demonstrate that DisCo remarkably outperforms previous state-of-the-art methods across a variety of video understanding benchmarks, while also achieving higher token efficiency thanks to the reduction of semantic indistinctness. The code: https://github.com/ZJHTerry18/DisCo.
>
---
#### [new 069] ViTCoT: Video-Text Interleaved Chain-of-Thought for Boosting Video Understanding in Large Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频理解任务，旨在解决现有大语言模型在视频推理中忽视视觉信息的问题。作者提出ViTCoT方法，通过图文交错的思维链提升模型对视频内容的理解能力，并构建了相关基准数据集ViTIB进行验证。实验表明该方法优于传统文本思维链。**

- **链接: [http://arxiv.org/pdf/2507.09876v1](http://arxiv.org/pdf/2507.09876v1)**

> **作者:** Yongheng Zhang; Xu Liu; Ruihan Tao; Qiguang Chen; Hao Fei; Wanxiang Che; Libo Qin
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Video understanding plays a vital role in bridging low-level visual signals with high-level cognitive reasoning, and is fundamental to applications such as autonomous driving, embodied AI, and the broader pursuit of AGI. The rapid development of large language models (LLMs), particularly those utilizing Chain-of-Thought (CoT) technology, has significantly advanced video reasoning capabilities. However, current approaches primarily depend on textual information for reasoning, overlooking the visual modality in the actual video reasoning process. In contrast, humans naturally re-examine visual content while reasoning. Motivated by this, we introduce a novel video reasoning paradigm: Video-Text Interleaved CoT (ViTCoT), which facilitates more intuitive and cognitively aligned reasoning. To the end, first, we construct the Video-Text Interleaved Benchmark (ViTIB), which is created using MLLMs for key-video selection and manually verified. Furthermore, we extensively explore the potential of the ViTCoT paradigm in the video understanding field. Extensive experiments demonstrate that ViTCoT significantly enhances performance compared to the traditional text-only CoT paradigm and effectively activates more neuron values in MLLMs.
>
---
#### [new 070] Memory-Augmented SAM2 for Training-Free Surgical Video Segmentation
- **分类: cs.CV**

- **简介: 该论文属于视频目标分割任务，旨在解决SAM2在复杂手术视频中因快速器械运动、遮挡等问题导致的性能下降。工作提出MA-SAM2，通过上下文感知和抗遮挡的记忆模型增强SAM2，实现无需训练的高效多器械跟踪与分割，在EndoVis数据集上取得显著性能提升。**

- **链接: [http://arxiv.org/pdf/2507.09577v1](http://arxiv.org/pdf/2507.09577v1)**

> **作者:** Ming Yin; Fu Wang; Xujiong Ye; Yanda Meng; Zeyu Fu
>
> **摘要:** Surgical video segmentation is a critical task in computer-assisted surgery, essential for enhancing surgical quality and patient outcomes. Recently, the Segment Anything Model 2 (SAM2) framework has demonstrated remarkable advancements in both image and video segmentation. However, the inherent limitations of SAM2's greedy selection memory design are amplified by the unique properties of surgical videos-rapid instrument movement, frequent occlusion, and complex instrument-tissue interaction-resulting in diminished performance in the segmentation of complex, long videos. To address these challenges, we introduce Memory Augmented (MA)-SAM2, a training-free video object segmentation strategy, featuring novel context-aware and occlusion-resilient memory models. MA-SAM2 exhibits strong robustness against occlusions and interactions arising from complex instrument movements while maintaining accuracy in segmenting objects throughout videos. Employing a multi-target, single-loop, one-prompt inference further enhances the efficiency of the tracking process in multi-instrument videos. Without introducing any additional parameters or requiring further training, MA-SAM2 achieved performance improvements of 4.36% and 6.1% over SAM2 on the EndoVis2017 and EndoVis2018 datasets, respectively, demonstrating its potential for practical surgical applications.
>
---
#### [new 071] Inter2Former: Dynamic Hybrid Attention for Efficient High-Precision Interactive
- **分类: cs.CV**

- **简介: 该论文属于交互式分割任务，旨在解决现有方法在精度与效率间的权衡问题。通过提出Inter2Former模型，结合动态提示嵌入、混合注意力机制、专家混合模块和局部上采样策略，实现高效高精度的交互式分割。**

- **链接: [http://arxiv.org/pdf/2507.09612v1](http://arxiv.org/pdf/2507.09612v1)**

> **作者:** You Huang; Lichao Chen; Jiayi Ji; Liujuan Cao; Shengchuan Zhang; Rongrong Ji
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Interactive segmentation (IS) improves annotation efficiency by segmenting target regions from user prompts, with widespread applications in real-world scenarios. Current approaches face a critical trade-off: dense-token methods achieve superior accuracy and detail preservation but suffer from prohibitively slow processing on CPU devices, while the Segment Anything Model (SAM) advances the field with sparse prompt tokens for fast inference but compromises segmentation quality. In this paper, we propose Inter2Former to address this challenge by optimizing computation allocation in dense-token processing, which introduces four key enhancements. First, we propose Dynamic Prompt Embedding (DPE) that adaptively processes only regions of interest while avoiding additional overhead from background tokens. Second, we introduce Dynamic Hybrid Attention (DHA), which leverages previous segmentation masks to route tokens through either full attention (O(N2)) for boundary regions or our proposed efficient BSQ attention (O(N)) for non-boundary regions. Third, we develop Hybrid Mixture of Experts (HMoE), which applies similar adaptive computation strategies in FFN modules with CPU-optimized parallel processing. Finally, we present Dynamic Local Upsampling (DLU), a reverse operation of DPE, which localizes objects with a lightweight MLP and performs fine-grained upsampling only in detected regions. Experimental results on high-precision IS benchmarks demonstrate that Inter2Former achieves SOTA performance with high efficiency on CPU devices.
>
---
#### [new 072] Prompt4Trust: A Reinforcement Learning Prompt Augmentation Framework for Clinically-Aligned Confidence Calibration in Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于医疗领域多模态大语言模型任务，旨在解决模型对提示敏感和置信度不准的问题。论文提出Prompt4Trust框架，通过强化学习生成辅助提示，提升模型在医疗视觉问答中的准确性和置信度一致性，并实现跨模型泛化。**

- **链接: [http://arxiv.org/pdf/2507.09279v1](http://arxiv.org/pdf/2507.09279v1)**

> **作者:** Anita Kriz; Elizabeth Laura Janes; Xing Shen; Tal Arbel
>
> **备注:** Preprint version. The peer-reviewed version of this paper has been accepted to ICCV 2025 Workshop CVAMD
>
> **摘要:** Multimodal large language models (MLLMs) hold considerable promise for applications in healthcare. However, their deployment in safety-critical settings is hindered by two key limitations: (i) sensitivity to prompt design, and (ii) a tendency to generate incorrect responses with high confidence. As clinicians may rely on a model's stated confidence to gauge the reliability of its predictions, it is especially important that when a model expresses high confidence, it is also highly accurate. We introduce Prompt4Trust, the first reinforcement learning (RL) framework for prompt augmentation targeting confidence calibration in MLLMs. A lightweight LLM is trained to produce context-aware auxiliary prompts that guide a downstream task MLLM to generate responses in which the expressed confidence more accurately reflects predictive accuracy. Unlike conventional calibration techniques, Prompt4Trust specifically prioritizes aspects of calibration most critical for safe and trustworthy clinical decision-making. Beyond improvements driven by this clinically motivated calibration objective, our proposed method also improves task accuracy, achieving state-of-the-art medical visual question answering (VQA) performance on the PMC-VQA benchmark, which is composed of multiple-choice questions spanning diverse medical imaging modalities. Moreover, our framework trained with a small downstream task MLLM showed promising zero-shot generalization to larger MLLMs in our experiments, suggesting the potential for scalable calibration without the associated computational costs. This work demonstrates the potential of automated yet human-aligned prompt engineering for improving the the trustworthiness of MLLMs in safety critical settings. Our codebase can be found at https://github.com/xingbpshen/vccrl-llm.
>
---
#### [new 073] Uncertainty-Driven Expert Control: Enhancing the Reliability of Medical Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于医疗视觉语言模型任务，旨在解决模型在医疗应用中的不确定性与可靠性问题。作者提出Expert-CFG框架，通过不确定性估计、专家参考和无分类器引导优化输出，无需额外训练即可提升模型性能，实验证明其在资源有限场景下优于大参数模型。**

- **链接: [http://arxiv.org/pdf/2507.09209v1](http://arxiv.org/pdf/2507.09209v1)**

> **作者:** Xiao Liang; Di Wang; Zhicheng Jiao; Ronghan Li; Pengfei Yang; Quan Wang; Tat-Seng Chua
>
> **摘要:** The rapid advancements in Vision Language Models (VLMs) have prompted the development of multi-modal medical assistant systems. Despite this progress, current models still have inherent probabilistic uncertainties, often producing erroneous or unverified responses-an issue with serious implications in medical applications. Existing methods aim to enhance the performance of Medical Vision Language Model (MedVLM) by adjusting model structure, fine-tuning with high-quality data, or through preference fine-tuning. However, these training-dependent strategies are costly and still lack sufficient alignment with clinical expertise. To address these issues, we propose an expert-in-the-loop framework named Expert-Controlled Classifier-Free Guidance (Expert-CFG) to align MedVLM with clinical expertise without additional training. This framework introduces an uncertainty estimation strategy to identify unreliable outputs. It then retrieves relevant references to assist experts in highlighting key terms and applies classifier-free guidance to refine the token embeddings of MedVLM, ensuring that the adjusted outputs are correct and align with expert highlights. Evaluations across three medical visual question answering benchmarks demonstrate that the proposed Expert-CFG, with 4.2B parameters and limited expert annotations, outperforms state-of-the-art models with 13B parameters. The results demonstrate the feasibility of deploying such a system in resource-limited settings for clinical use.
>
---
#### [new 074] Domain Adaptation and Multi-view Attention for Learnable Landmark Tracking with Sparse Data
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于地表特征跟踪任务，旨在解决稀疏数据下自主航天器地形导航中地标检测与跟踪难题。论文提出轻量级神经网络架构，结合领域自适应与多视角注意力机制，实现高效实时地标检测与描述，提升了现有方法的泛化能力与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.09420v1](http://arxiv.org/pdf/2507.09420v1)**

> **作者:** Timothy Chase Jr; Karthik Dantu
>
> **备注:** Presented at the RSS Space Robotics Workshop 2025. Poster available online at https://tjchase34.github.io/assets/pdfs/rss_poster.pdf
>
> **摘要:** The detection and tracking of celestial surface terrain features are crucial for autonomous spaceflight applications, including Terrain Relative Navigation (TRN), Entry, Descent, and Landing (EDL), hazard analysis, and scientific data collection. Traditional photoclinometry-based pipelines often rely on extensive a priori imaging and offline processing, constrained by the computational limitations of radiation-hardened systems. While historically effective, these approaches typically increase mission costs and duration, operate at low processing rates, and have limited generalization. Recently, learning-based computer vision has gained popularity to enhance spacecraft autonomy and overcome these limitations. While promising, emerging techniques frequently impose computational demands exceeding the capabilities of typical spacecraft hardware for real-time operation and are further challenged by the scarcity of labeled training data for diverse extraterrestrial environments. In this work, we present novel formulations for in-situ landmark tracking via detection and description. We utilize lightweight, computationally efficient neural network architectures designed for real-time execution on current-generation spacecraft flight processors. For landmark detection, we propose improved domain adaptation methods that enable the identification of celestial terrain features with distinct, cheaply acquired training data. Concurrently, for landmark description, we introduce a novel attention alignment formulation that learns robust feature representations that maintain correspondence despite significant landmark viewpoint variations. Together, these contributions form a unified system for landmark tracking that demonstrates superior performance compared to existing state-of-the-art techniques.
>
---
#### [new 075] Advancing Text-to-3D Generation with Linearized Lookahead Variational Score Distillation
- **分类: cs.CV**

- **简介: 该论文属于文本生成3D模型任务，旨在解决现有变分得分蒸馏（VSD）方法在收敛速度和质量上的不足。作者发现并解决了LoRA与3D分布间的不匹配问题，提出线性前瞻VSD（L²-VSD），通过优化顺序调整及线性化模型提升训练稳定性与生成效果。**

- **链接: [http://arxiv.org/pdf/2507.09748v1](http://arxiv.org/pdf/2507.09748v1)**

> **作者:** Yu Lei; Bingde Liu; Qingsong Xie; Haonan Lu; Zhijie Deng
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Text-to-3D generation based on score distillation of pre-trained 2D diffusion models has gained increasing interest, with variational score distillation (VSD) as a remarkable example. VSD proves that vanilla score distillation can be improved by introducing an extra score-based model, which characterizes the distribution of images rendered from 3D models, to correct the distillation gradient. Despite the theoretical foundations, VSD, in practice, is likely to suffer from slow and sometimes ill-posed convergence. In this paper, we perform an in-depth investigation of the interplay between the introduced score model and the 3D model, and find that there exists a mismatching problem between LoRA and 3D distributions in practical implementation. We can simply adjust their optimization order to improve the generation quality. By doing so, the score model looks ahead to the current 3D state and hence yields more reasonable corrections. Nevertheless, naive lookahead VSD may suffer from unstable training in practice due to the potential over-fitting. To address this, we propose to use a linearized variant of the model for score distillation, giving rise to the Linearized Lookahead Variational Score Distillation ($L^2$-VSD). $L^2$-VSD can be realized efficiently with forward-mode autodiff functionalities of existing deep learning libraries. Extensive experiments validate the efficacy of $L^2$-VSD, revealing its clear superiority over prior score distillation-based methods. We also show that our method can be seamlessly incorporated into any other VSD-based text-to-3D framework.
>
---
#### [new 076] WordCraft: Interactive Artistic Typography with Attention Awareness and Noise Blending
- **分类: cs.CV**

- **简介: 该论文属于艺术排版任务，旨在解决现有生成模型在交互性、局部编辑和多字符组合上的局限性。作者提出了WordCraft系统，结合扩散模型与区域注意力机制，实现精准多区域生成与连续优化，并利用大语言模型解析用户指令，提升创意排版的交互性与灵活性。**

- **链接: [http://arxiv.org/pdf/2507.09573v1](http://arxiv.org/pdf/2507.09573v1)**

> **作者:** Zhe Wang; Jingbo Zhang; Tianyi Wei; Wanchao Su; Can Wang
>
> **备注:** 14 pages, 16 figures
>
> **摘要:** Artistic typography aims to stylize input characters with visual effects that are both creative and legible. Traditional approaches rely heavily on manual design, while recent generative models, particularly diffusion-based methods, have enabled automated character stylization. However, existing solutions remain limited in interactivity, lacking support for localized edits, iterative refinement, multi-character composition, and open-ended prompt interpretation. We introduce WordCraft, an interactive artistic typography system that integrates diffusion models to address these limitations. WordCraft features a training-free regional attention mechanism for precise, multi-region generation and a noise blending that supports continuous refinement without compromising visual quality. To support flexible, intent-driven generation, we incorporate a large language model to parse and structure both concrete and abstract user prompts. These components allow our framework to synthesize high-quality, stylized typography across single- and multi-character inputs across multiple languages, supporting diverse user-centered workflows. Our system significantly enhances interactivity in artistic typography synthesis, opening up creative possibilities for artists and designers.
>
---
#### [new 077] A Training-Free, Task-Agnostic Framework for Enhancing MLLM Performance on High-Resolution Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态任务，旨在解决多模态大语言模型（MLLM）在高分辨率图像上的细粒度定位与推理能力不足的问题。通过提出一种无需训练、任务无关的两阶段框架ECP，先提取候选区域再进行预测，有效提升了MLLM在高分辨率图像上的表现。**

- **链接: [http://arxiv.org/pdf/2507.10202v1](http://arxiv.org/pdf/2507.10202v1)**

> **作者:** Jaeseong Lee; Yeeun Choi; Heechan Choi; Hanjung Kim; Seonjoo Kim
>
> **备注:** Accepted at CVPR 2025 Workshop on Emergent Visual Abilities and Limits of Foundation Models
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in vision-language understanding, reasoning, and generation. However, they struggle with tasks requiring fine-grained localization and reasoning in high-resolution images. This constraint stems from the fact that MLLMs are fine-tuned with fixed image resolution to align with the pre-trained image encoder used in MLLM. Consequently, feeding high-resolution images directly into MLLMs leads to poor generalization due to a train-test resolution discrepancy, while downsampling these images-although ensuring consistency-compromises fine-grained visual details and ultimately degrades performance. To address this challenge, we propose Extract Candidate then Predict (ECP), a novel training-free, task-agnostic two-stage framework designed to enhance MLLM performance on high-resolution images. The key intuition behind ECP is that while MLLMs struggle with high-resolution images, their predictions on downsampled images still contain implicit localization cues. By first identifying candidate region using the coarse prediction and then predicting the final output based on candidate region, ECP effectively preserves fine-grained details while mitigating the challenges posed by high-resolution data. We validate our framework on 4K GUI grounding and 4K, 8K MLLM perception, achieving +21.3%, +5.8%, +5.2% absolute improvement compared to baseline respectively, demonstrating its effectiveness. Code is available at https://github.com/yenncye/ECP.
>
---
#### [new 078] Deep Recurrence for Dynamical Segmentation Models
- **分类: cs.CV; cs.LG**

- **简介: 论文提出一种受预测编码启发的反馈机制，将循环结构引入U-Net，用于动态分割模型。任务是图像分割，旨在提升模型在噪声环境下的鲁棒性和小样本学习能力。工作包括设计softmax投影与指数衰减操作，实现反馈驱动的迭代优化，实验证明其优于前馈模型。**

- **链接: [http://arxiv.org/pdf/2507.10143v1](http://arxiv.org/pdf/2507.10143v1)**

> **作者:** David Calhas; Arlindo L. Oliveira
>
> **备注:** 12 pages
>
> **摘要:** While biological vision systems rely heavily on feedback connections to iteratively refine perception, most artificial neural networks remain purely feedforward, processing input in a single static pass. In this work, we propose a predictive coding inspired feedback mechanism that introduces a recurrent loop from output to input, allowing the model to refine its internal state over time. We implement this mechanism within a standard U-Net architecture and introduce two biologically motivated operations, softmax projection and exponential decay, to ensure stability of the feedback loop. Through controlled experiments on a synthetic segmentation task, we show that the feedback model significantly outperforms its feedforward counterpart in noisy conditions and generalizes more effectively with limited supervision. Notably, feedback achieves above random performance with just two training examples, while the feedforward model requires at least four. Our findings demonstrate that feedback enhances robustness and data efficiency, and offer a path toward more adaptive and biologically inspired neural architectures. Code is available at: github.com/DCalhas/feedback_segmentation.
>
---
#### [new 079] SeqCSIST: Sequential Closely-Spaced Infrared Small Target Unmixing
- **分类: cs.CV**

- **简介: 该论文属于红外小目标解混任务，旨在从密集的近距红外小目标群中通过亚像素定位检测所有目标。论文提出了SeqCSIST数据集和DeRefNet模型，采用时序可变形特征对齐模块提升多帧信息融合效果，推动了该领域的研究进展。**

- **链接: [http://arxiv.org/pdf/2507.09556v1](http://arxiv.org/pdf/2507.09556v1)**

> **作者:** Ximeng Zhai; Bohan Xu; Yaohong Chen; Hao Wang; Kehua Guo; Yimian Dai
>
> **备注:** Accepted by TGRS
>
> **摘要:** Due to the limitation of the optical lens focal length and the resolution of the infrared detector, distant Closely-Spaced Infrared Small Target (CSIST) groups typically appear as mixing spots in the infrared image. In this paper, we propose a novel task, Sequential CSIST Unmixing, namely detecting all targets in the form of sub-pixel localization from a highly dense CSIST group. However, achieving such precise detection is an extremely difficult challenge. In addition, the lack of high-quality public datasets has also restricted the research progress. To this end, firstly, we contribute an open-source ecosystem, including SeqCSIST, a sequential benchmark dataset, and a toolkit that provides objective evaluation metrics for this special task, along with the implementation of 23 relevant methods. Furthermore, we propose the Deformable Refinement Network (DeRefNet), a model-driven deep learning framework that introduces a Temporal Deformable Feature Alignment (TDFA) module enabling adaptive inter-frame information aggregation. To the best of our knowledge, this work is the first endeavor to address the CSIST Unmixing task within a multi-frame paradigm. Experiments on the SeqCSIST dataset demonstrate that our method outperforms the state-of-the-art approaches with mean Average Precision (mAP) metric improved by 5.3\%. Our dataset and toolkit are available from https://github.com/GrokCV/SeqCSIST.
>
---
#### [new 080] DRPCA-Net: Make Robust PCA Great Again for Infrared Small Target Detection
- **分类: cs.CV**

- **简介: 该论文属于红外小目标检测任务，旨在解决现有深度学习模型复杂、缺乏可解释性且忽视目标稀疏先验的问题。作者提出DRPCA-Net，结合鲁棒主成分分析与动态超网络，实现参数自适应生成和更精确的背景建模，提升了检测性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.09541v1](http://arxiv.org/pdf/2507.09541v1)**

> **作者:** Zihao Xiong; Fei Zhou; Fengyi Wu; Shuai Yuan; Maixia Fu; Zhenming Peng; Jian Yang; Yimian Dai
>
> **备注:** Accepted by TGRS
>
> **摘要:** Infrared small target detection plays a vital role in remote sensing, industrial monitoring, and various civilian applications. Despite recent progress powered by deep learning, many end-to-end convolutional models tend to pursue performance by stacking increasingly complex architectures, often at the expense of interpretability, parameter efficiency, and generalization. These models typically overlook the intrinsic sparsity prior of infrared small targets--an essential cue that can be explicitly modeled for both performance and efficiency gains. To address this, we revisit the model-based paradigm of Robust Principal Component Analysis (RPCA) and propose Dynamic RPCA Network (DRPCA-Net), a novel deep unfolding network that integrates the sparsity-aware prior into a learnable architecture. Unlike conventional deep unfolding methods that rely on static, globally learned parameters, DRPCA-Net introduces a dynamic unfolding mechanism via a lightweight hypernetwork. This design enables the model to adaptively generate iteration-wise parameters conditioned on the input scene, thereby enhancing its robustness and generalization across diverse backgrounds. Furthermore, we design a Dynamic Residual Group (DRG) module to better capture contextual variations within the background, leading to more accurate low-rank estimation and improved separation of small targets. Extensive experiments on multiple public infrared datasets demonstrate that DRPCA-Net significantly outperforms existing state-of-the-art methods in detection accuracy. Code is available at https://github.com/GrokCV/DRPCA-Net.
>
---
#### [new 081] Contrastive Pretraining with Dual Visual Encoders for Gloss-Free Sign Language Translation
- **分类: cs.CV**

- **简介: 该论文属于手语翻译任务，旨在无需手语标注的情况下将手语视频转化为文本。为解决标注依赖问题，作者提出双视觉编码器框架，通过对比学习联合对齐视觉与文本特征。预训练后，模型在下游任务中融合视觉特征并使用编码器-解码器结构，最终在Phoenix-2014T数据集上取得最佳性能。**

- **链接: [http://arxiv.org/pdf/2507.10306v1](http://arxiv.org/pdf/2507.10306v1)**

> **作者:** Ozge Mercanoglu Sincan; Richard Bowden
>
> **备注:** Accepted at 9th Workshop on Sign Language Translation and Avatar Technologies (SLTAT), will be held in conjunction with IVA'25
>
> **摘要:** Sign Language Translation (SLT) aims to convert sign language videos into spoken or written text. While early systems relied on gloss annotations as an intermediate supervision, such annotations are costly to obtain and often fail to capture the full complexity of continuous signing. In this work, we propose a two-phase, dual visual encoder framework for gloss-free SLT, leveraging contrastive visual-language pretraining. During pretraining, our approach employs two complementary visual backbones whose outputs are jointly aligned with each other and with sentence-level text embeddings via a contrastive objective. During the downstream SLT task, we fuse the visual features and input them into an encoder-decoder model. On the Phoenix-2014T benchmark, our dual encoder architecture consistently outperforms its single stream variants and achieves the highest BLEU-4 score among existing gloss-free SLT approaches.
>
---
#### [new 082] 4D-MISR: A unified model for low-dose super-resolution imaging via feature fusion
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率重建任务，旨在解决低剂量电子显微成像中分辨率不足的问题。作者提出4D-MISR方法，通过融合多视角低分辨率图像并结合CNN增强重建，实现对辐射敏感材料的原子级成像。**

- **链接: [http://arxiv.org/pdf/2507.09953v1](http://arxiv.org/pdf/2507.09953v1)**

> **作者:** Zifei Wang; Zian Mao; Xiaoya He; Xi Huang; Haoran Zhang; Chun Cheng; Shufen Chu; Tingzheng Hou; Xiaoqin Zeng; Yujun Xie
>
> **摘要:** While electron microscopy offers crucial atomic-resolution insights into structure-property relationships, radiation damage severely limits its use on beam-sensitive materials like proteins and 2D materials. To overcome this challenge, we push beyond the electron dose limits of conventional electron microscopy by adapting principles from multi-image super-resolution (MISR) that have been widely used in remote sensing. Our method fuses multiple low-resolution, sub-pixel-shifted views and enhances the reconstruction with a convolutional neural network (CNN) that integrates features from synthetic, multi-angle observations. We developed a dual-path, attention-guided network for 4D-STEM that achieves atomic-scale super-resolution from ultra-low-dose data. This provides robust atomic-scale visualization across amorphous, semi-crystalline, and crystalline beam-sensitive specimens. Systematic evaluations on representative materials demonstrate comparable spatial resolution to conventional ptychography under ultra-low-dose conditions. Our work expands the capabilities of 4D-STEM, offering a new and generalizable method for the structural analysis of radiation-vulnerable materials.
>
---
#### [new 083] LifelongPR: Lifelong knowledge fusion for point cloud place recognition based on replay and prompt learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于点云场景识别任务，旨在解决模型在持续学习新环境时遗忘旧知识的问题。作者提出了LifelongPR框架，结合回放样本选择和提示学习，提升模型的跨域适应性与知识保留能力，从而增强实际应用中的可扩展性与稳定性。**

- **链接: [http://arxiv.org/pdf/2507.10034v1](http://arxiv.org/pdf/2507.10034v1)**

> **作者:** Xianghong Zou; Jianping Li; Zhe Chen; Zhen Cao; Zhen Dong; Qiegen Liu; Bisheng Yang
>
> **摘要:** Point cloud place recognition (PCPR) plays a crucial role in photogrammetry and robotics applications such as autonomous driving, intelligent transportation, and augmented reality. In real-world large-scale deployments of a positioning system, PCPR models must continuously acquire, update, and accumulate knowledge to adapt to diverse and dynamic environments, i.e., the ability known as continual learning (CL). However, existing PCPR models often suffer from catastrophic forgetting, leading to significant performance degradation in previously learned scenes when adapting to new environments or sensor types. This results in poor model scalability, increased maintenance costs, and system deployment difficulties, undermining the practicality of PCPR. To address these issues, we propose LifelongPR, a novel continual learning framework for PCPR, which effectively extracts and fuses knowledge from sequential point cloud data. First, to alleviate the knowledge loss, we propose a replay sample selection method that dynamically allocates sample sizes according to each dataset's information quantity and selects spatially diverse samples for maximal representativeness. Second, to handle domain shifts, we design a prompt learning-based CL framework with a lightweight prompt module and a two-stage training strategy, enabling domain-specific feature adaptation while minimizing forgetting. Comprehensive experiments on large-scale public and self-collected datasets are conducted to validate the effectiveness of the proposed method. Compared with state-of-the-art (SOTA) methods, our method achieves 6.50% improvement in mIR@1, 7.96% improvement in mR@1, and an 8.95% reduction in F. The code and pre-trained models are publicly available at https://github.com/zouxianghong/LifelongPR.
>
---
#### [new 084] Hierarchical Abstraction Enables Human-Like 3D Object Recognition in Deep Learning Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究3D物体识别任务，探讨深度学习模型是否具备类似人类的3D形状表示能力。通过对比人类与两种模型（DGCNN和点Transformer）在不同条件下的表现，发现点Transformer因支持层次化抽象，更接近人类识别能力。**

- **链接: [http://arxiv.org/pdf/2507.09830v1](http://arxiv.org/pdf/2507.09830v1)**

> **作者:** Shuhao Fu; Philip J. Kellman; Hongjing Lu
>
> **摘要:** Both humans and deep learning models can recognize objects from 3D shapes depicted with sparse visual information, such as a set of points randomly sampled from the surfaces of 3D objects (termed a point cloud). Although deep learning models achieve human-like performance in recognizing objects from 3D shapes, it remains unclear whether these models develop 3D shape representations similar to those used by human vision for object recognition. We hypothesize that training with 3D shapes enables models to form representations of local geometric structures in 3D shapes. However, their representations of global 3D object shapes may be limited. We conducted two human experiments systematically manipulating point density and object orientation (Experiment 1), and local geometric structure (Experiment 2). Humans consistently performed well across all experimental conditions. We compared two types of deep learning models, one based on a convolutional neural network (DGCNN) and the other on visual transformers (point transformer), with human performance. We found that the point transformer model provided a better account of human performance than the convolution-based model. The advantage mainly results from the mechanism in the point transformer model that supports hierarchical abstraction of 3D shapes.
>
---
#### [new 085] 4D-Animal: Freely Reconstructing Animatable 3D Animals from Videos
- **分类: cs.CV**

- **简介: 该论文属于3D动物重建任务，旨在从视频中重建可动画的3D动物模型。现有方法依赖稀疏关键点标注，费时且不稳定。本文提出4D-Animal，无需关键点标注，通过密集特征网络和分层对齐策略，实现高效、稳定的3D重建，生成高质量的3D资产，适用于大规模应用。**

- **链接: [http://arxiv.org/pdf/2507.10437v1](http://arxiv.org/pdf/2507.10437v1)**

> **作者:** Shanshan Zhong; Jiawei Peng; Zehan Zheng; Zhongzhan Huang; Wufei Ma; Guofeng Zhang; Qihao Liu; Alan Yuille; Jieneng Chen
>
> **摘要:** Existing methods for reconstructing animatable 3D animals from videos typically rely on sparse semantic keypoints to fit parametric models. However, obtaining such keypoints is labor-intensive, and keypoint detectors trained on limited animal data are often unreliable. To address this, we propose 4D-Animal, a novel framework that reconstructs animatable 3D animals from videos without requiring sparse keypoint annotations. Our approach introduces a dense feature network that maps 2D representations to SMAL parameters, enhancing both the efficiency and stability of the fitting process. Furthermore, we develop a hierarchical alignment strategy that integrates silhouette, part-level, pixel-level, and temporal cues from pre-trained 2D visual models to produce accurate and temporally coherent reconstructions across frames. Extensive experiments demonstrate that 4D-Animal outperforms both model-based and model-free baselines. Moreover, the high-quality 3D assets generated by our method can benefit other 3D tasks, underscoring its potential for large-scale applications. The code is released at https://github.com/zhongshsh/4D-Animal.
>
---
#### [new 086] ViT-ProtoNet for Few-Shot Image Classification: A Multi-Benchmark Evaluation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像分类任务，旨在解决小样本学习中视觉变换器（ViT）未被充分利用的问题。作者提出ViT-ProtoNet方法，将ViT-Small集成到原型网络框架中，并在多个基准上评估其性能，结果显示其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.09299v1](http://arxiv.org/pdf/2507.09299v1)**

> **作者:** Abdulvahap Mutlu; Şengül Doğan; Türker Tuncer
>
> **备注:** All codes are available at https://github.com/abdulvahapmutlu/vit-protonet
>
> **摘要:** The remarkable representational power of Vision Transformers (ViTs) remains underutilized in few-shot image classification. In this work, we introduce ViT-ProtoNet, which integrates a ViT-Small backbone into the Prototypical Network framework. By averaging class conditional token embeddings from a handful of support examples, ViT-ProtoNet constructs robust prototypes that generalize to novel categories under 5-shot settings. We conduct an extensive empirical evaluation on four standard benchmarks: Mini-ImageNet, FC100, CUB-200, and CIFAR-FS, including overlapped support variants to assess robustness. Across all splits, ViT-ProtoNet consistently outperforms CNN-based prototypical counterparts, achieving up to a 3.2\% improvement in 5-shot accuracy and demonstrating superior feature separability in latent space. Furthermore, it outperforms or is competitive with transformer-based competitors using a more lightweight backbone. Comprehensive ablations examine the impact of transformer depth, patch size, and fine-tuning strategy. To foster reproducibility, we release code and pretrained weights. Our results establish ViT-ProtoNet as a powerful, flexible approach for few-shot classification and set a new baseline for transformer-based meta-learners.
>
---
#### [new 087] MCA-LLaVA: Manhattan Causal Attention for Reducing Hallucination in Large Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型任务，旨在减少大模型中的幻觉问题。通过分析发现，位置编码的长期衰减导致图像与文本对齐偏差，进而引发幻觉。作者提出MCA-LLaVA，利用曼哈顿距离建模二维空间衰减，改善图像与指令的对齐效果，从而缓解幻觉问题。**

- **链接: [http://arxiv.org/pdf/2507.09184v1](http://arxiv.org/pdf/2507.09184v1)**

> **作者:** Qiyan Zhao; Xiaofeng Zhang; Yiheng Li; Yun Xing; Xiaosong Yuan; Feilong Tang; Sinan Fan; Xuhang Chen; Xuyao Zhang; Dahan Wang
>
> **备注:** Accepted in ACM MM 2025
>
> **摘要:** Hallucinations pose a significant challenge in Large Vision Language Models (LVLMs), with misalignment between multimodal features identified as a key contributing factor. This paper reveals the negative impact of the long-term decay in Rotary Position Encoding (RoPE), used for positional modeling in LVLMs, on multimodal alignment. Concretely, under long-term decay, instruction tokens exhibit uneven perception of image tokens located at different positions within the two-dimensional space: prioritizing image tokens from the bottom-right region since in the one-dimensional sequence, these tokens are positionally closer to the instruction tokens. This biased perception leads to insufficient image-instruction interaction and suboptimal multimodal alignment. We refer to this phenomenon as image alignment bias. To enhance instruction's perception of image tokens at different spatial locations, we propose MCA-LLaVA, based on Manhattan distance, which extends the long-term decay to a two-dimensional, multi-directional spatial decay. MCA-LLaVA integrates the one-dimensional sequence order and two-dimensional spatial position of image tokens for positional modeling, mitigating hallucinations by alleviating image alignment bias. Experimental results of MCA-LLaVA across various hallucination and general benchmarks demonstrate its effectiveness and generality. The code can be accessed in https://github.com/ErikZ719/MCA-LLaVA.
>
---
#### [new 088] Online Long-term Point Tracking in the Foundation Model Era
- **分类: cs.CV**

- **简介: 该论文属于视频点跟踪任务，旨在解决在线长期点跟踪问题，即在无法访问未来帧的情况下，保持跨视频帧的物理点一致性。论文提出Track-On模型，利用视觉基础模型提取空间特征，并通过Transformer结构维护时序连贯性，实现无未来信息的实时跟踪。**

- **链接: [http://arxiv.org/pdf/2507.09217v1](http://arxiv.org/pdf/2507.09217v1)**

> **作者:** Görkay Aydemir
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2501.18487
>
> **摘要:** Point tracking aims to identify the same physical point across video frames and serves as a geometry-aware representation of motion. This representation supports a wide range of applications, from robotics to augmented reality, by enabling accurate modeling of dynamic environments. Most existing long-term tracking approaches operate in an offline setting, where future frames are available to refine predictions and recover from occlusions. However, real-world scenarios often demand online predictions: the model must operate causally, using only current and past frames. This constraint is critical in streaming video and embodied AI, where decisions must be made immediately based on past observations. Under such constraints, viewpoint invariance becomes essential. Visual foundation models, trained on diverse large-scale datasets, offer the potential for robust geometric representations. While they lack temporal reasoning on their own, they can be integrated into tracking pipelines to enrich spatial features. In this thesis, we address the problem of long-term point tracking in an online setting, where frames are processed sequentially without access to future information or sliding windows. We begin by evaluating the suitability of visual foundation models for this task and find that they can serve as useful initializations and be integrated into tracking pipelines. However, to enable long-term tracking in an online setting, a dedicated design is still required. In particular, maintaining coherence over time in this causal regime requires memory to propagate appearance and context across frames. To address this, we introduce Track-On, a transformer-based model that treats each tracked point as a query and processes video frames one at a time. Track-On sets a new state of the art across seven public benchmarks, demonstrating the feasibility of long-term tracking without future access.
>
---
#### [new 089] Visual Surface Wave Elastography: Revealing Subsurface Physical Properties via Visible Surface Waves
- **分类: cs.CV**

- **简介: 该论文提出了一种名为“视觉表面波弹性成像”的方法，旨在通过视频捕捉材料表面波传播情况，推断其内部结构的厚度和刚度。任务是利用表面波信息反演材料物理属性。工作包括从视频中提取色散关系，并通过物理优化模型求解参数，验证了方法在模拟与真实数据上的有效性，可用于家庭健康监测等领域。**

- **链接: [http://arxiv.org/pdf/2507.09207v1](http://arxiv.org/pdf/2507.09207v1)**

> **作者:** Alexander C. Ogren; Berthy T. Feng; Jihoon Ahn; Katherine L. Bouman; Chiara Daraio
>
> **备注:** ICCV 2025
>
> **摘要:** Wave propagation on the surface of a material contains information about physical properties beneath its surface. We propose a method for inferring the thickness and stiffness of a structure from just a video of waves on its surface. Our method works by extracting a dispersion relation from the video and then solving a physics-based optimization problem to find the best-fitting thickness and stiffness parameters. We validate our method on both simulated and real data, in both cases showing strong agreement with ground-truth measurements. Our technique provides a proof-of-concept for at-home health monitoring of medically-informative tissue properties, and it is further applicable to fields such as human-computer interaction.
>
---
#### [new 090] VST-Pose: A Velocity-Integrated Spatiotem-poral Attention Network for Human WiFi Pose Estimation
- **分类: cs.CV**

- **简介: 该论文属于人体姿态估计任务，旨在通过WiFi信号实现非视觉的连续姿态追踪。为提升对细微动作的敏感性，作者提出了ViSTA-Former网络结构，并融合速度建模分支。在自建2D数据集上取得92.2%的PCK@50准确率，且在MMFi数据集上验证了3D姿态估计的有效性。**

- **链接: [http://arxiv.org/pdf/2507.09672v1](http://arxiv.org/pdf/2507.09672v1)**

> **作者:** Xinyu Zhang; Zhonghao Ye; Jingwei Zhang; Xiang Tian; Zhisheng Liang; Shipeng Yu
>
> **备注:** 8 pages, 7 figures, 8 tables. WiFi CSI, VST-Pose framework + ViSTA-Former dual-stream attention backbone. Code: https://github.com/CarmenQing/VST-Pose
>
> **摘要:** WiFi-based human pose estimation has emerged as a promising non-visual alternative approaches due to its pene-trability and privacy advantages. This paper presents VST-Pose, a novel deep learning framework for accurate and continuous pose estimation using WiFi channel state information. The proposed method introduces ViSTA-Former, a spatiotemporal attention backbone with dual-stream architecture that adopts a dual-stream architecture to separately capture temporal dependencies and structural relationships among body joints. To enhance sensitivity to subtle human motions, a velocity modeling branch is integrated into the framework, which learns short-term keypoint dis-placement patterns and improves fine-grained motion representation. We construct a 2D pose dataset specifically designed for smart home care scenarios and demonstrate that our method achieves 92.2% accuracy on the PCK@50 metric, outperforming existing methods by 8.3% in PCK@50 on the self-collected dataset. Further evaluation on the public MMFi dataset confirms the model's robustness and effectiveness in 3D pose estimation tasks. The proposed system provides a reliable and privacy-aware solution for continuous human motion analysis in indoor environments. Our codes are available in https://github.com/CarmenQing/VST-Pose.
>
---
#### [new 091] $I^{2}$-World: Intra-Inter Tokenization for Efficient Dynamic 4D Scene Forecasting
- **分类: cs.CV**

- **简介: 该论文属于4D场景预测任务，旨在解决复杂3D场景的高效建模与未来状态生成问题。作者提出了“I²-World”框架，通过“场景内-场景间”双阶段令牌化方法，实现空间细节保留与时间动态表达，并采用编码器-解码器结构提升预测准确性与时效性。实验表明其性能优越且计算效率高。**

- **链接: [http://arxiv.org/pdf/2507.09144v1](http://arxiv.org/pdf/2507.09144v1)**

> **作者:** Zhimin Liao; Ping Wei; Ruijie Zhang; Shuaijia Chen; Haoxuan Wang; Ziyang Ren
>
> **摘要:** Forecasting the evolution of 3D scenes and generating unseen scenarios via occupancy-based world models offers substantial potential for addressing corner cases in autonomous driving systems. While tokenization has revolutionized image and video generation, efficiently tokenizing complex 3D scenes remains a critical challenge for 3D world models. To address this, we propose $I^{2}$-World, an efficient framework for 4D occupancy forecasting. Our method decouples scene tokenization into intra-scene and inter-scene tokenizers. The intra-scene tokenizer employs a multi-scale residual quantization strategy to hierarchically compress 3D scenes while preserving spatial details. The inter-scene tokenizer residually aggregates temporal dependencies across timesteps. This dual design preserves the compactness of 3D tokenizers while retaining the dynamic expressiveness of 4D tokenizers. Unlike decoder-only GPT-style autoregressive models, $I^{2}$-World adopts an encoder-decoder architecture. The encoder aggregates spatial context from the current scene and predicts a transformation matrix to enable high-level control over scene generation. The decoder, conditioned on this matrix and historical tokens, ensures temporal consistency during generation. Experiments demonstrate that $I^{2}$-World achieves state-of-the-art performance, outperforming existing methods by 25.1\% in mIoU and 36.9\% in IoU for 4D occupancy forecasting while exhibiting exceptional computational efficiency: it requires merely 2.9 GB of training memory and achieves real-time inference at 37.0 FPS. Our code is available on https://github.com/lzzzzzm/II-World.
>
---
#### [new 092] Show and Polish: Reference-Guided Identity Preservation in Face Video Restoration
- **分类: cs.CV**

- **简介: 该论文属于人脸视频修复任务，旨在解决严重退化下身份特征丢失问题。方法IP-FVR引入参考图像作为视觉提示，结合解耦交叉注意力机制与反馈学习策略，有效保持身份一致性，并提出指数混合策略和多流负提示提升修复质量。**

- **链接: [http://arxiv.org/pdf/2507.10293v1](http://arxiv.org/pdf/2507.10293v1)**

> **作者:** Wenkang Han; Wang Lin; Yiyun Zhou; Qi Liu; Shulei Wang; Chang Yao; Jingyuan Chen
>
> **备注:** Accepted by MM 2025
>
> **摘要:** Face Video Restoration (FVR) aims to recover high-quality face videos from degraded versions. Traditional methods struggle to preserve fine-grained, identity-specific features when degradation is severe, often producing average-looking faces that lack individual characteristics. To address these challenges, we introduce IP-FVR, a novel method that leverages a high-quality reference face image as a visual prompt to provide identity conditioning during the denoising process. IP-FVR incorporates semantically rich identity information from the reference image using decoupled cross-attention mechanisms, ensuring detailed and identity consistent results. For intra-clip identity drift (within 24 frames), we introduce an identity-preserving feedback learning method that combines cosine similarity-based reward signals with suffix-weighted temporal aggregation. This approach effectively minimizes drift within sequences of frames. For inter-clip identity drift, we develop an exponential blending strategy that aligns identities across clips by iteratively blending frames from previous clips during the denoising process. This method ensures consistent identity representation across different clips. Additionally, we enhance the restoration process with a multi-stream negative prompt, guiding the model's attention to relevant facial attributes and minimizing the generation of low-quality or incorrect features. Extensive experiments on both synthetic and real-world datasets demonstrate that IP-FVR outperforms existing methods in both quality and identity preservation, showcasing its substantial potential for practical applications in face video restoration.
>
---
#### [new 093] Straighten Viscous Rectified Flow via Noise Optimization
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决Reflow方法在快速生成高质量图像时存在的分布差异问题。作者提出VRFNO，通过引入历史速度项和噪声优化策略，提升轨迹预测准确性并缩小与真实图像的差距，从而改进单步及少量步骤下的图像生成效果。**

- **链接: [http://arxiv.org/pdf/2507.10218v1](http://arxiv.org/pdf/2507.10218v1)**

> **作者:** Jimin Dai; Jiexi Yan; Jian Yang; Lei Luo
>
> **摘要:** The Reflow operation aims to straighten the inference trajectories of the rectified flow during training by constructing deterministic couplings between noises and images, thereby improving the quality of generated images in single-step or few-step generation. However, we identify critical limitations in Reflow, particularly its inability to rapidly generate high-quality images due to a distribution gap between images in its constructed deterministic couplings and real images. To address these shortcomings, we propose a novel alternative called Straighten Viscous Rectified Flow via Noise Optimization (VRFNO), which is a joint training framework integrating an encoder and a neural velocity field. VRFNO introduces two key innovations: (1) a historical velocity term that enhances trajectory distinction, enabling the model to more accurately predict the velocity of the current trajectory, and (2) the noise optimization through reparameterization to form optimized couplings with real images which are then utilized for training, effectively mitigating errors caused by Reflow's limitations. Comprehensive experiments on synthetic data and real datasets with varying resolutions show that VRFNO significantly mitigates the limitations of Reflow, achieving state-of-the-art performance in both one-step and few-step generation tasks.
>
---
#### [new 094] Text-Visual Semantic Constrained AI-Generated Image Quality Assessment
- **分类: cs.CV; I.4.7**

- **简介: 该论文属于AI生成图像质量评估任务，旨在解决现有方法在语义对齐和细节感知上的不足。作者提出SC-AGIQA框架，结合文本-视觉语义约束，通过TSAM模块提升文本与图像的一致性检查，FFDPM模块增强对图像细微失真的感知，从而更全面地评估AI生成图像的质量。**

- **链接: [http://arxiv.org/pdf/2507.10432v1](http://arxiv.org/pdf/2507.10432v1)**

> **作者:** Qiang Li; Qingsen Yan; Haojian Huang; Peng Wu; Haokui Zhang; Yanning Zhang
>
> **备注:** 9 pages, 5 figures, Accepted at ACMMM 2025
>
> **摘要:** With the rapid advancements in Artificial Intelligence Generated Image (AGI) technology, the accurate assessment of their quality has become an increasingly vital requirement. Prevailing methods typically rely on cross-modal models like CLIP or BLIP to evaluate text-image alignment and visual quality. However, when applied to AGIs, these methods encounter two primary challenges: semantic misalignment and details perception missing. To address these limitations, we propose Text-Visual Semantic Constrained AI-Generated Image Quality Assessment (SC-AGIQA), a unified framework that leverages text-visual semantic constraints to significantly enhance the comprehensive evaluation of both text-image consistency and perceptual distortion in AI-generated images. Our approach integrates key capabilities from multiple models and tackles the aforementioned challenges by introducing two core modules: the Text-assisted Semantic Alignment Module (TSAM), which leverages Multimodal Large Language Models (MLLMs) to bridge the semantic gap by generating an image description and comparing it against the original prompt for a refined consistency check, and the Frequency-domain Fine-Grained Degradation Perception Module (FFDPM), which draws inspiration from Human Visual System (HVS) properties by employing frequency domain analysis combined with perceptual sensitivity weighting to better quantify subtle visual distortions and enhance the capture of fine-grained visual quality details in images. Extensive experiments conducted on multiple benchmark datasets demonstrate that SC-AGIQA outperforms existing state-of-the-art methods. The code is publicly available at https://github.com/mozhu1/SC-AGIQA.
>
---
#### [new 095] A Survey on MLLM-based Visually Rich Document Understanding: Methods, Challenges, and Emerging Trends
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态文档理解任务，旨在解决复杂视觉、文本和布局信息的自动处理问题。论文综述了基于多模态大语言模型（MLLM）的方法，分析了特征融合、训练范式与数据集，并探讨了挑战与未来方向。**

- **链接: [http://arxiv.org/pdf/2507.09861v1](http://arxiv.org/pdf/2507.09861v1)**

> **作者:** Yihao Ding; Siwen Luo; Yue Dai; Yanbei Jiang; Zechuan Li; Geoffrey Martin; Yifan Peng
>
> **备注:** Work in progress
>
> **摘要:** Visually-Rich Document Understanding (VRDU) has emerged as a critical field, driven by the need to automatically process documents containing complex visual, textual, and layout information. Recently, Multimodal Large Language Models (MLLMs) have shown remarkable potential in this domain, leveraging both Optical Character Recognition (OCR)-dependent and OCR-free frameworks to extract and interpret information in document images. This survey reviews recent advancements in MLLM-based VRDU, highlighting three core components: (1) methods for encoding and fusing textual, visual, and layout features; (2) training paradigms, including pretraining strategies, instruction-response tuning, and the trainability of different model modules; and (3) datasets utilized for pretraining, instruction-tuning, and supervised fine-tuning. Finally, we discuss the challenges and opportunities in this evolving field and propose future directions to advance the efficiency, generalizability, and robustness of VRDU systems.
>
---
#### [new 096] Brain Stroke Detection and Classification Using CT Imaging with Transformer Models and Explainable AI
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分类任务，旨在解决脑卒中类型（缺血性、出血性和无卒中）的自动识别问题。作者采用MaxViT等Transformer模型对CT图像进行分类，并结合数据增强技术提升性能。最终模型在准确率和F1分数上表现优异，并引入Grad-CAM++实现模型决策可视化，提升可解释性，推动AI在临床中的应用。**

- **链接: [http://arxiv.org/pdf/2507.09630v1](http://arxiv.org/pdf/2507.09630v1)**

> **作者:** Shomukh Qari; Maha A. Thafar
>
> **备注:** 5 figures
>
> **摘要:** Stroke is one of the leading causes of death globally, making early and accurate diagnosis essential for improving patient outcomes, particularly in emergency settings where timely intervention is critical. CT scans are the key imaging modality because of their speed, accessibility, and cost-effectiveness. This study proposed an artificial intelligence framework for multiclass stroke classification (ischemic, hemorrhagic, and no stroke) using CT scan images from a dataset provided by the Republic of Turkey's Ministry of Health. The proposed method adopted MaxViT, a state-of-the-art Vision Transformer, as the primary deep learning model for image-based stroke classification, with additional transformer variants (vision transformer, transformer-in-transformer, and ConvNext). To enhance model generalization and address class imbalance, we applied data augmentation techniques, including synthetic image generation. The MaxViT model trained with augmentation achieved the best performance, reaching an accuracy and F1-score of 98.00%, outperforming all other evaluated models and the baseline methods. The primary goal of this study was to distinguish between stroke types with high accuracy while addressing crucial issues of transparency and trust in artificial intelligence models. To achieve this, Explainable Artificial Intelligence (XAI) was integrated into the framework, particularly Grad-CAM++. It provides visual explanations of the model's decisions by highlighting relevant stroke regions in the CT scans and establishing an accurate, interpretable, and clinically applicable solution for early stroke detection. This research contributed to the development of a trustworthy AI-assisted diagnostic tool for stroke, facilitating its integration into clinical practice and enhancing access to timely and optimal stroke diagnosis in emergency departments, thereby saving more lives.
>
---
#### [new 097] RAPNet: A Receptive-Field Adaptive Convolutional Neural Network for Pansharpening
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; eess.IV**

- **简介: 该论文属于遥感图像处理任务，旨在解决全色锐化问题。现有卷积神经网络忽略局部内容变化，导致空间细节提取精度不足。作者提出RAPNet，引入自适应感受野卷积（RAPConv）和动态特征融合模块（PAN-DFF），实现空间细节与光谱特性的更好平衡，提升了全色锐化效果。**

- **链接: [http://arxiv.org/pdf/2507.10461v1](http://arxiv.org/pdf/2507.10461v1)**

> **作者:** Tao Tang; Chengxu Yang
>
> **备注:** To appear in the proceedings of the 6th International Conference on Artificial Intelligence and Electromechanical Automation (AIEA 2025). 5 pages, 6 figures
>
> **摘要:** Pansharpening refers to the process of integrating a high resolution panchromatic (PAN) image with a lower resolution multispectral (MS) image to generate a fused product, which is pivotal in remote sensing. Despite the effectiveness of CNNs in addressing this challenge, they are inherently constrained by the uniform application of convolutional kernels across all spatial positions, overlooking local content variations. To overcome this issue, we introduce RAPNet, a new architecture that leverages content-adaptive convolution. At its core, RAPNet employs the Receptive-field Adaptive Pansharpening Convolution (RAPConv), designed to produce spatially adaptive kernels responsive to local feature context, thereby enhancing the precision of spatial detail extraction. Additionally, the network integrates the Pansharpening Dynamic Feature Fusion (PAN-DFF) module, which incorporates an attention mechanism to achieve an optimal balance between spatial detail enhancement and spectral fidelity. Comprehensive evaluations on publicly available datasets confirm that RAPNet delivers superior performance compared to existing approaches, as demonstrated by both quantitative metrics and qualitative assessments. Ablation analyses further substantiate the effectiveness of the proposed adaptive components.
>
---
#### [new 098] Automated Multi-Class Crop Pathology Classification via Convolutional Neural Networks: A Deep Learning Approach for Real-Time Precision Agriculture
- **分类: cs.CV; I.2.6; I.5.4**

- **简介: 该论文属于图像分类任务，旨在解决农作物病害自动识别问题。通过构建基于卷积神经网络的模型，实现对八种常见作物病害的精准检测，并提供防治建议。研究包括数据预处理、模型训练与部署，提升了农业病害诊断效率与可及性。**

- **链接: [http://arxiv.org/pdf/2507.09375v1](http://arxiv.org/pdf/2507.09375v1)**

> **作者:** Sourish Suri; Yifei Shao
>
> **备注:** 29 pages, 10 figures, 1 table. Code available at: https://github.com/Sourish85/CNN-CROP-DIS-DETECTOR
>
> **摘要:** Crop diseases present a significant barrier to agricultural productivity and global food security, especially in large-scale farming where early identification is often delayed or inaccurate. This research introduces a Convolutional Neural Network (CNN)-based image classification system designed to automate the detection and classification of eight common crop diseases using leaf imagery. The methodology involves a complete deep learning pipeline: image acquisition from a large, labeled dataset, preprocessing via resizing, normalization, and augmentation, and model training using TensorFlow with Keras' Sequential API. The CNN architecture comprises three convolutional layers with increasing filter sizes and ReLU activations, followed by max pooling, flattening, and fully connected layers, concluding with a softmax output for multi-class classification. The system achieves high training accuracy (~90%) and demonstrates reliable performance on unseen data, although a validation accuracy of ~60% suggests minor overfitting. Notably, the model integrates a treatment recommendation module, providing actionable guidance by mapping each detected disease to suitable pesticide or fungicide interventions. Furthermore, the solution is deployed on an open-source, mobile-compatible platform, enabling real-time image-based diagnostics for farmers in remote areas. This research contributes a scalable and accessible tool to the field of precision agriculture, reducing reliance on manual inspection and promoting sustainable disease management practices. By merging deep learning with practical agronomic support, this work underscores the potential of CNNs to transform crop health monitoring and enhance food production resilience on a global scale.
>
---
#### [new 099] RefSTAR: Blind Facial Image Restoration with Reference Selection, Transfer, and Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于盲脸图像修复任务，旨在解决复杂退化下的人脸图像恢复与身份保持问题。作者提出了RefSTAR方法，包含参考选择、特征迁移和重建机制，并构建了相关数据集与损失函数，有效提升了修复效果与参考特征融合质量。**

- **链接: [http://arxiv.org/pdf/2507.10470v1](http://arxiv.org/pdf/2507.10470v1)**

> **作者:** Zhicun Yin; Junjie Chen; Ming Liu; Zhixin Wang; Fan Li; Renjing Pei; Xiaoming Li; Rynson W. H. Lau; Wangmeng Zuo
>
> **摘要:** Blind facial image restoration is highly challenging due to unknown complex degradations and the sensitivity of humans to faces. Although existing methods introduce auxiliary information from generative priors or high-quality reference images, they still struggle with identity preservation problems, mainly due to improper feature introduction on detailed textures. In this paper, we focus on effectively incorporating appropriate features from high-quality reference images, presenting a novel blind facial image restoration method that considers reference selection, transfer, and reconstruction (RefSTAR). In terms of selection, we construct a reference selection (RefSel) module. For training the RefSel module, we construct a RefSel-HQ dataset through a mask generation pipeline, which contains annotating masks for 10,000 ground truth-reference pairs. As for the transfer, due to the trivial solution in vanilla cross-attention operations, a feature fusion paradigm is designed to force the features from the reference to be integrated. Finally, we propose a reference image reconstruction mechanism that further ensures the presence of reference image features in the output image. The cycle consistency loss is also redesigned in conjunction with the mask. Extensive experiments on various backbone models demonstrate superior performance, showing better identity preservation ability and reference feature transfer quality. Source code, dataset, and pre-trained models are available at https://github.com/yinzhicun/RefSTAR.
>
---
#### [new 100] PoseLLM: Enhancing Language-Guided Human Pose Estimation with MLP Alignment
- **分类: cs.CV**

- **简介: 该论文属于语言引导的人体姿态估计任务，旨在解决传统方法泛化性差及现有语言引导方法空间-文本交互不足的问题。作者提出PoseLLM，用非线性MLP替代线性连接器，提升视觉与文本特征融合能力，实现更高精度定位并保持零样本泛化性能。**

- **链接: [http://arxiv.org/pdf/2507.09139v1](http://arxiv.org/pdf/2507.09139v1)**

> **作者:** Dewen Zhang; Tahir Hussain; Wangpeng An; Hayaru Shouno
>
> **备注:** Preprint
>
> **摘要:** Human pose estimation traditionally relies on architectures that encode keypoint priors, limiting their generalization to novel poses or unseen keypoints. Recent language-guided approaches like LocLLM reformulate keypoint localization as a vision-language task, enabling zero-shot generalization through textual descriptions. However, LocLLM's linear projector fails to capture complex spatial-textual interactions critical for high-precision localization. To address this, we propose PoseLLM, the first Large Language Model (LLM)-based pose estimation framework that replaces the linear projector with a nonlinear MLP vision-language connector. This lightweight two-layer MLP with GELU activation enables hierarchical cross-modal feature transformation, enhancing the fusion of visual patches and textual keypoint descriptions. Trained exclusively on COCO data, PoseLLM achieves 77.8 AP on the COCO validation set, outperforming LocLLM by +0.4 AP, while maintaining strong zero-shot generalization on Human-Art and MPII. Our work demonstrates that a simple yet powerful nonlinear connector significantly boosts localization accuracy without sacrificing generalization, advancing the state-of-the-art in language-guided pose estimation. Code is available at https://github.com/Ody-trek/PoseLLM.
>
---
#### [new 101] 360-Degree Full-view Image Segmentation by Spherical Convolution compatible with Large-scale Planar Pre-trained Models
- **分类: cs.CV**

- **简介: 该论文属于全景图像分割任务，旨在解决现有二维预训练模型在处理全景图像时因畸变和不连续性导致的性能下降问题。论文提出了一种新的球面采样方法，兼容大规模二维预训练模型，通过球面离散采样减轻畸变，并将该方法应用于全景图像分割，取得了良好效果。**

- **链接: [http://arxiv.org/pdf/2507.09216v1](http://arxiv.org/pdf/2507.09216v1)**

> **作者:** Jingguo Liu; Han Yu; Shigang Li; Jianfeng Li
>
> **备注:** This paper is accecpted by ICMEW 2025
>
> **摘要:** Due to the current lack of large-scale datasets at the million-scale level, tasks involving panoramic images predominantly rely on existing two-dimensional pre-trained image benchmark models as backbone networks. However, these networks are not equipped to recognize the distortions and discontinuities inherent in panoramic images, which adversely affects their performance in such tasks. In this paper, we introduce a novel spherical sampling method for panoramic images that enables the direct utilization of existing pre-trained models developed for two-dimensional images. Our method employs spherical discrete sampling based on the weights of the pre-trained models, effectively mitigating distortions while achieving favorable initial training values. Additionally, we apply the proposed sampling method to panoramic image segmentation, utilizing features obtained from the spherical model as masks for specific channel attentions, which yields commendable results on commonly used indoor datasets, Stanford2D3D.
>
---
#### [new 102] From images to properties: a NeRF-driven framework for granular material parameter inversion
- **分类: cs.CV; physics.geo-ph**

- **简介: 该论文属于计算机视觉与材料科学交叉任务，旨在通过视觉观测推断颗粒材料的摩擦角参数。作者提出一种结合NeRF和MPM仿真的框架，利用多视角图像重建3D几何，并通过贝叶斯优化匹配模拟与真实图像，最终实现摩擦角的高精度估计。**

- **链接: [http://arxiv.org/pdf/2507.09005v1](http://arxiv.org/pdf/2507.09005v1)**

> **作者:** Cheng-Hsi Hsiao; Krishna Kumar
>
> **摘要:** We introduce a novel framework that integrates Neural Radiance Fields (NeRF) with Material Point Method (MPM) simulation to infer granular material properties from visual observations. Our approach begins by generating synthetic experimental data, simulating an plow interacting with sand. The experiment is rendered into realistic images as the photographic observations. These observations include multi-view images of the experiment's initial state and time-sequenced images from two fixed cameras. Using NeRF, we reconstruct the 3D geometry from the initial multi-view images, leveraging its capability to synthesize novel viewpoints and capture intricate surface details. The reconstructed geometry is then used to initialize material point positions for the MPM simulation, where the friction angle remains unknown. We render images of the simulation under the same camera setup and compare them to the observed images. By employing Bayesian optimization, we minimize the image loss to estimate the best-fitting friction angle. Our results demonstrate that friction angle can be estimated with an error within 2 degrees, highlighting the effectiveness of inverse analysis through purely visual observations. This approach offers a promising solution for characterizing granular materials in real-world scenarios where direct measurement is impractical or impossible.
>
---
#### [new 103] IGD: Instructional Graphic Design with Multimodal Layer Generation
- **分类: cs.CV**

- **简介: 该论文属于图形设计自动化任务，旨在解决现有方法缺乏创造力、生成文件不可编辑及文字可读性差的问题。论文提出IGD模型，通过多模态理解和扩散生成技术，实现根据自然语言指令快速生成可编辑的多层次设计文件，提升自动化设计效果与实用性。**

- **链接: [http://arxiv.org/pdf/2507.09910v1](http://arxiv.org/pdf/2507.09910v1)**

> **作者:** Yadong Qu; Shancheng Fang; Yuxin Wang; Xiaorui Wang; Zhineng Chen; Hongtao Xie; Yongdong Zhang
>
> **备注:** ICCV 2025
>
> **摘要:** Graphic design visually conveys information and data by creating and combining text, images and graphics. Two-stage methods that rely primarily on layout generation lack creativity and intelligence, making graphic design still labor-intensive. Existing diffusion-based methods generate non-editable graphic design files at image level with poor legibility in visual text rendering, which prevents them from achieving satisfactory and practical automated graphic design. In this paper, we propose Instructional Graphic Designer (IGD) to swiftly generate multimodal layers with editable flexibility with only natural language instructions. IGD adopts a new paradigm that leverages parametric rendering and image asset generation. First, we develop a design platform and establish a standardized format for multi-scenario design files, thus laying the foundation for scaling up data. Second, IGD utilizes the multimodal understanding and reasoning capabilities of MLLM to accomplish attribute prediction, sequencing and layout of layers. It also employs a diffusion model to generate image content for assets. By enabling end-to-end training, IGD architecturally supports scalability and extensibility in complex graphic design tasks. The superior experimental results demonstrate that IGD offers a new solution for graphic design.
>
---
#### [new 104] CoSMo: A Multimodal Transformer for Page Stream Segmentation in Comic Books
- **分类: cs.CV**

- **简介: 该论文属于自然语言处理与计算机视觉交叉任务，旨在解决漫画书页面流分割（PSS）问题。作者提出了CoSMo模型，基于多模态Transformer架构，并构建了包含20,800页的标注数据集。模型在多个指标上优于基线方法和大型通用模型，验证了视觉特征在宏观结构分割中的主导作用及多模态信息在歧义消除中的优势。**

- **链接: [http://arxiv.org/pdf/2507.10053v1](http://arxiv.org/pdf/2507.10053v1)**

> **作者:** Marc Serra Ortega; Emanuele Vivoli; Artemis Llabrés; Dimosthenis Karatzas
>
> **摘要:** This paper introduces CoSMo, a novel multimodal Transformer for Page Stream Segmentation (PSS) in comic books, a critical task for automated content understanding, as it is a necessary first stage for many downstream tasks like character analysis, story indexing, or metadata enrichment. We formalize PSS for this unique medium and curate a new 20,800-page annotated dataset. CoSMo, developed in vision-only and multimodal variants, consistently outperforms traditional baselines and significantly larger general-purpose vision-language models across F1-Macro, Panoptic Quality, and stream-level metrics. Our findings highlight the dominance of visual features for comic PSS macro-structure, yet demonstrate multimodal benefits in resolving challenging ambiguities. CoSMo establishes a new state-of-the-art, paving the way for scalable comic book analysis.
>
---
#### [new 105] MENTOR: Efficient Multimodal-Conditioned Tuning for Autoregressive Vision Generation Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文提出MENTOR，一种高效的多模态条件调优框架，用于自回归视觉生成模型。旨在解决文本到图像生成中视觉控制不精确、多模态输入平衡困难及训练成本高的问题。通过两阶段训练方法，实现细粒度的多模态对齐与指令调优，提升生成质量与可控性。**

- **链接: [http://arxiv.org/pdf/2507.09574v1](http://arxiv.org/pdf/2507.09574v1)**

> **作者:** Haozhe Zhao; Zefan Cai; Shuzheng Si; Liang Chen; Jiuxiang Gu; Wen Xiao; Junjie Hu
>
> **备注:** 24 pages,12 figures
>
> **摘要:** Recent text-to-image models produce high-quality results but still struggle with precise visual control, balancing multimodal inputs, and requiring extensive training for complex multimodal image generation. To address these limitations, we propose MENTOR, a novel autoregressive (AR) framework for efficient Multimodal-conditioned Tuning for Autoregressive multimodal image generation. MENTOR combines an AR image generator with a two-stage training paradigm, enabling fine-grained, token-level alignment between multimodal inputs and image outputs without relying on auxiliary adapters or cross-attention modules. The two-stage training consists of: (1) a multimodal alignment stage that establishes robust pixel- and semantic-level alignment, followed by (2) a multimodal instruction tuning stage that balances the integration of multimodal inputs and enhances generation controllability. Despite modest model size, suboptimal base components, and limited training resources, MENTOR achieves strong performance on the DreamBench++ benchmark, outperforming competitive baselines in concept preservation and prompt following. Additionally, our method delivers superior image reconstruction fidelity, broad task adaptability, and improved training efficiency compared to diffusion-based methods. Dataset, code, and models are available at: https://github.com/HaozheZhao/MENTOR
>
---
#### [new 106] THYME: Temporal Hierarchical-Cyclic Interactivity Modeling for Video Scene Graphs in Aerial Footage
- **分类: cs.CV**

- **简介: 该论文属于视频场景图生成任务，旨在解决现有方法在细粒度空间细节和长时序依赖建模上的不足。作者提出了THYME方法，结合层次化特征聚合与循环时序优化，并构建了包含多种交互类型的航拍视频数据集AeroEye-v1.0，提升了视频场景理解效果。**

- **链接: [http://arxiv.org/pdf/2507.09200v1](http://arxiv.org/pdf/2507.09200v1)**

> **作者:** Trong-Thuan Nguyen; Pha Nguyen; Jackson Cothren; Alper Yilmaz; Minh-Triet Tran; Khoa Luu
>
> **摘要:** The rapid proliferation of video in applications such as autonomous driving, surveillance, and sports analytics necessitates robust methods for dynamic scene understanding. Despite advances in static scene graph generation and early attempts at video scene graph generation, previous methods often suffer from fragmented representations, failing to capture fine-grained spatial details and long-range temporal dependencies simultaneously. To address these limitations, we introduce the Temporal Hierarchical Cyclic Scene Graph (THYME) approach, which synergistically integrates hierarchical feature aggregation with cyclic temporal refinement to address these limitations. In particular, THYME effectively models multi-scale spatial context and enforces temporal consistency across frames, yielding more accurate and coherent scene graphs. In addition, we present AeroEye-v1.0, a novel aerial video dataset enriched with five types of interactivity that overcome the constraints of existing datasets and provide a comprehensive benchmark for dynamic scene graph generation. Empirically, extensive experiments on ASPIRe and AeroEye-v1.0 demonstrate that the proposed THYME approach outperforms state-of-the-art methods, offering improved scene understanding in ground-view and aerial scenarios.
>
---
#### [new 107] Infinite Video Understanding
- **分类: cs.CV; cs.AI; cs.IR; cs.LG; cs.MM**

- **简介: 该论文属于视频理解任务，旨在解决现有模型在处理超长视频时面临的计算、内存及时间连贯性等问题。作者提出“无限视频理解”这一研究方向，强调需发展持续处理、记忆机制、分层表示等技术，以实现对任意长度视频的理解与推理。**

- **链接: [http://arxiv.org/pdf/2507.09068v1](http://arxiv.org/pdf/2507.09068v1)**

> **作者:** Dell Zhang; Xiangyu Chen; Jixiang Luo; Mengxi Jia; Changzhi Sun; Ruilong Ren; Jingren Liu; Hao Sun; Xuelong Li
>
> **摘要:** The rapid advancements in Large Language Models (LLMs) and their multimodal extensions (MLLMs) have ushered in remarkable progress in video understanding. However, a fundamental challenge persists: effectively processing and comprehending video content that extends beyond minutes or hours. While recent efforts like Video-XL-2 have demonstrated novel architectural solutions for extreme efficiency, and advancements in positional encoding such as HoPE and VideoRoPE++ aim to improve spatio-temporal understanding over extensive contexts, current state-of-the-art models still encounter significant computational and memory constraints when faced with the sheer volume of visual tokens from lengthy sequences. Furthermore, maintaining temporal coherence, tracking complex events, and preserving fine-grained details over extended periods remain formidable hurdles, despite progress in agentic reasoning systems like Deep Video Discovery. This position paper posits that a logical, albeit ambitious, next frontier for multimedia research is Infinite Video Understanding -- the capability for models to continuously process, understand, and reason about video data of arbitrary, potentially never-ending duration. We argue that framing Infinite Video Understanding as a blue-sky research objective provides a vital north star for the multimedia, and the wider AI, research communities, driving innovation in areas such as streaming architectures, persistent memory mechanisms, hierarchical and adaptive representations, event-centric reasoning, and novel evaluation paradigms. Drawing inspiration from recent work on long/ultra-long video understanding and several closely related fields, we outline the core challenges and key research directions towards achieving this transformative capability.
>
---
#### [new 108] Towards Fine-Grained Adaptation of CLIP via a Self-Trained Alignment Score
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的无监督适应任务，旨在解决细粒度分类中伪标签不准确和计算成本高的问题。作者提出了FAIR方法，通过动态对齐图像与文本特征，结合自训练机制，提升了跨模态交互与伪标签质量，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.09615v1](http://arxiv.org/pdf/2507.09615v1)**

> **作者:** Eman Ali; Sathira Silva; Chetan Arora; Muhammad Haris Khan
>
> **摘要:** Vision-language models (VLMs) like CLIP excel in zero-shot learning by aligning image and text representations through contrastive pretraining. Existing approaches to unsupervised adaptation (UA) for fine-grained classification with VLMs either rely on fixed alignment scores that cannot capture evolving, subtle class distinctions or use computationally expensive pseudo-labeling strategies that limit scalability. In contrast, we show that modeling fine-grained cross-modal interactions during adaptation produces more accurate, class-discriminative pseudo-labels and substantially improves performance over state-of-the-art (SOTA) methods. We introduce Fine-grained Alignment and Interaction Refinement (FAIR), an innovative approach that dynamically aligns localized image features with descriptive language embeddings through a set of Class Description Anchors (CDA). This enables the definition of a Learned Alignment Score (LAS), which incorporates CDA as an adaptive classifier, facilitating cross-modal interactions to improve self-training in unsupervised adaptation. Furthermore, we propose a self-training weighting mechanism designed to refine pseudo-labels in the presence of inter-class ambiguities. Our approach, FAIR, delivers a substantial performance boost in fine-grained unsupervised adaptation, achieving a notable overall gain of 2.78% across 13 fine-grained datasets compared to SOTA methods.
>
---
#### [new 109] Glance-MCMT: A General MCMT Framework with Glance Initialization and Progressive Association
- **分类: cs.CV**

- **简介: 该论文属于多摄像头多目标（MCMT）跟踪任务，旨在解决跨视角目标身份一致性分配问题。提出了Glance-MCMT框架，通过轨迹和外观特征匹配实现初始全局ID分配，并采用优先级匹配策略在后续帧中关联新轨迹到已有全局ID，必要时才创建新ID，结合3D位置估计提升空间验证效果。**

- **链接: [http://arxiv.org/pdf/2507.10115v1](http://arxiv.org/pdf/2507.10115v1)**

> **作者:** Hamidreza Hashempoor
>
> **摘要:** We propose a multi-camera multi-target (MCMT) tracking framework that ensures consistent global identity assignment across views using trajectory and appearance cues. The pipeline starts with BoT-SORT-based single-camera tracking, followed by an initial glance phase to initialize global IDs via trajectory-feature matching. In later frames, new tracklets are matched to existing global identities through a prioritized global matching strategy. New global IDs are only introduced when no sufficiently similar trajectory or feature match is found. 3D positions are estimated using depth maps and calibration for spatial validation.
>
---
#### [new 110] MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second
- **分类: cs.CV**

- **简介: 该论文属于动态视图合成任务，旨在解决从单目视频中快速生成4D动态新视角的问题。作者提出了MoVieS模型，采用高斯基元网格表示动态3D场景，并统一建模外观、几何与运动，实现了高效的新视角合成与多任务应用。**

- **链接: [http://arxiv.org/pdf/2507.10065v1](http://arxiv.org/pdf/2507.10065v1)**

> **作者:** Chenguo Lin; Yuchen Lin; Panwang Pan; Yifan Yu; Honglei Yan; Katerina Fragkiadaki; Yadong Mu
>
> **备注:** Project page: https://chenguolin.github.io/projects/MoVieS
>
> **摘要:** We present MoVieS, a novel feed-forward model that synthesizes 4D dynamic novel views from monocular videos in one second. MoVieS represents dynamic 3D scenes using pixel-aligned grids of Gaussian primitives, explicitly supervising their time-varying motion. This allows, for the first time, the unified modeling of appearance, geometry and motion, and enables view synthesis, reconstruction and 3D point tracking within a single learning-based framework. By bridging novel view synthesis with dynamic geometry reconstruction, MoVieS enables large-scale training on diverse datasets with minimal dependence on task-specific supervision. As a result, it also naturally supports a wide range of zero-shot applications, such as scene flow estimation and moving object segmentation. Extensive experiments validate the effectiveness and efficiency of MoVieS across multiple tasks, achieving competitive performance while offering several orders of magnitude speedups.
>
---
#### [new 111] Ambiguity-Aware and High-Order Relation Learning for Multi-Grained Image-Text Matching
- **分类: cs.CV; cs.IR; cs.MM**

- **简介: 该论文属于图像-文本匹配任务，旨在解决多粒度语义关联中的模糊样本和高阶关系学习问题。提出了AAHR框架，通过动态聚类对比学习、全局与局部特征提取、图神经网络等方法，提升模型对软正负样本的判别能力和语义理解能力，从而改善跨模态匹配效果。**

- **链接: [http://arxiv.org/pdf/2507.09256v1](http://arxiv.org/pdf/2507.09256v1)**

> **作者:** Junyu Chen; Yihua Gao; Mingyuan Ge; Mingyong Li
>
> **备注:** Accepted by the Knowledge-Based Systems(KBS), 2025
>
> **摘要:** Image-text matching is crucial for bridging the semantic gap between computer vision and natural language processing. However, existing methods still face challenges in handling high-order associations and semantic ambiguities among similar instances. These ambiguities arise from subtle differences between soft positive samples (semantically similar but incorrectly labeled) and soft negative samples (locally matched but globally inconsistent), creating matching uncertainties. Furthermore, current methods fail to fully utilize the neighborhood relationships among semantically similar instances within training batches, limiting the model's ability to learn high-order shared knowledge. This paper proposes the Ambiguity-Aware and High-order Relation learning framework (AAHR) to address these issues. AAHR constructs a unified representation space through dynamic clustering prototype contrastive learning, effectively mitigating the soft positive sample problem. The framework introduces global and local feature extraction mechanisms and an adaptive aggregation network, significantly enhancing full-grained semantic understanding capabilities. Additionally, AAHR employs intra-modal and inter-modal correlation matrices to investigate neighborhood relationships among sample instances thoroughly. It incorporates GNN to enhance semantic interactions between instances. Furthermore, AAHR integrates momentum contrastive learning to expand the negative sample set. These combined strategies significantly improve the model's ability to discriminate between features. Experimental results demonstrate that AAHR outperforms existing state-of-the-art methods on Flickr30K, MSCOCO, and ECCV Caption datasets, considerably improving the accuracy and efficiency of image-text matching. The code and model checkpoints for this research are available at https://github.com/Image-Text-Matching/AAHR .
>
---
#### [new 112] VRU-Accident: A Vision-Language Benchmark for Video Question Answering and Dense Captioning for Accident Scene Understanding
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言任务，旨在解决自动驾驶中脆弱道路使用者（如行人、骑车人）事故理解的挑战。作者构建了VRU-Accident基准，包含1K事故视频和6K问答对，用于评估多模态大模型在事故场景中的推理与描述能力。**

- **链接: [http://arxiv.org/pdf/2507.09815v1](http://arxiv.org/pdf/2507.09815v1)**

> **作者:** Younggun Kim; Ahmed S. Abdelrahman; Mohamed Abdel-Aty
>
> **备注:** 22 pages, 11 figures, 5 tables
>
> **摘要:** Ensuring the safety of vulnerable road users (VRUs), such as pedestrians and cyclists, is a critical challenge for autonomous driving systems, as crashes involving VRUs often result in severe or fatal consequences. While multimodal large language models (MLLMs) have shown promise in enhancing scene understanding and decision making in autonomous vehicles, there is currently no standardized benchmark to quantitatively evaluate their reasoning abilities in complex, safety-critical scenarios involving VRUs. To address this gap, we present VRU-Accident, a large-scale vision-language benchmark designed to evaluate MLLMs in high-risk traffic scenarios involving VRUs. VRU-Accident comprises 1K real-world dashcam accident videos, annotated with 6K multiple-choice question-answer pairs across six safety-critical categories (with 24K candidate options and 3.4K unique answer choices), as well as 1K dense scene descriptions. Unlike prior works, our benchmark focuses explicitly on VRU-vehicle accidents, providing rich, fine-grained annotations that capture both spatial-temporal dynamics and causal semantics of accidents. To assess the current landscape of MLLMs, we conduct a comprehensive evaluation of 17 state-of-the-art models on the multiple-choice VQA task and on the dense captioning task. Our findings reveal that while MLLMs perform reasonably well on visually grounded attributes, they face significant challenges in reasoning and describing accident causes, types, and preventability.
>
---
#### [new 113] ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉与医疗康复交叉任务，旨在解决假肢步态分析中视觉检测与运动模式识别困难的问题。作者构建了名为ProGait的多用途视频数据集，包含412个视频片段，支持视频目标分割、2D人体姿态估计和步态分析等任务，并提供基准模型，证明其在假肢特定任务中的优越泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.10223v1](http://arxiv.org/pdf/2507.10223v1)**

> **作者:** Xiangyu Yin; Boyuan Yang; Weichen Liu; Qiyao Xue; Abrar Alamri; Goeran Fiedler; Wei Gao
>
> **备注:** Accepted by ICCV'25
>
> **摘要:** Prosthetic legs play a pivotal role in clinical rehabilitation, allowing individuals with lower-limb amputations the ability to regain mobility and improve their quality of life. Gait analysis is fundamental for optimizing prosthesis design and alignment, directly impacting the mobility and life quality of individuals with lower-limb amputations. Vision-based machine learning (ML) methods offer a scalable and non-invasive solution to gait analysis, but face challenges in correctly detecting and analyzing prosthesis, due to their unique appearances and new movement patterns. In this paper, we aim to bridge this gap by introducing a multi-purpose dataset, namely ProGait, to support multiple vision tasks including Video Object Segmentation, 2D Human Pose Estimation, and Gait Analysis (GA). ProGait provides 412 video clips from four above-knee amputees when testing multiple newly-fitted prosthetic legs through walking trials, and depicts the presence, contours, poses, and gait patterns of human subjects with transfemoral prosthetic legs. Alongside the dataset itself, we also present benchmark tasks and fine-tuned baseline models to illustrate the practical application and performance of the ProGait dataset. We compared our baseline models against pre-trained vision models, demonstrating improved generalizability when applying the ProGait dataset for prosthesis-specific tasks. Our code is available at https://github.com/pittisl/ProGait and dataset at https://huggingface.co/datasets/ericyxy98/ProGait.
>
---
#### [new 114] Text-to-Remote-Sensing-Image Retrieval beyond RGB Sources
- **分类: cs.CV; cs.CL; cs.IR; cs.MM**

- **简介: 该论文属于文本到遥感图像检索任务，旨在解决现有方法局限于RGB图像、难以利用多源传感器数据的问题。作者构建了包含超64万对SAR与光学图像及文本标注的数据集CrisisLandMark，并提出CLOSP框架实现跨模态检索，显著提升性能。**

- **链接: [http://arxiv.org/pdf/2507.10403v1](http://arxiv.org/pdf/2507.10403v1)**

> **作者:** Daniele Rege Cambrin; Lorenzo Vaiani; Giuseppe Gallipoli; Luca Cagliero; Paolo Garza
>
> **摘要:** Retrieving relevant imagery from vast satellite archives is crucial for applications like disaster response and long-term climate monitoring. However, most text-to-image retrieval systems are limited to RGB data, failing to exploit the unique physical information captured by other sensors, such as the all-weather structural sensitivity of Synthetic Aperture Radar (SAR) or the spectral signatures in optical multispectral data. To bridge this gap, we introduce CrisisLandMark, a new large-scale corpus of over 647,000 Sentinel-1 SAR and Sentinel-2 multispectral images paired with structured textual annotations for land cover, land use, and crisis events harmonized from authoritative land cover systems (CORINE and Dynamic World) and crisis-specific sources. We then present CLOSP (Contrastive Language Optical SAR Pretraining), a novel framework that uses text as a bridge to align unpaired optical and SAR images into a unified embedding space. Our experiments show that CLOSP achieves a new state-of-the-art, improving retrieval nDGC by 54% over existing models. Additionally, we find that the unified training strategy overcomes the inherent difficulty of interpreting SAR imagery by transferring rich semantic knowledge from the optical domain with indirect interaction. Furthermore, GeoCLOSP, which integrates geographic coordinates into our framework, creates a powerful trade-off between generality and specificity: while the CLOSP excels at general semantic tasks, the GeoCLOSP becomes a specialized expert for retrieving location-dependent crisis events and rare geographic features. This work highlights that the integration of diverse sensor data and geographic context is essential for unlocking the full potential of remote sensing archives.
>
---
#### [new 115] SnapMoGen: Human Motion Generation from Expressive Texts
- **分类: cs.CV**

- **简介: 该论文属于文本到动作生成任务，旨在解决现有方法在复杂文本控制和长序列动作生成上的局限性。作者构建了高质量数据集SnapMoGen，并提出MoMask++模型提升生成效果，同时结合大语言模型处理用户输入。**

- **链接: [http://arxiv.org/pdf/2507.09122v1](http://arxiv.org/pdf/2507.09122v1)**

> **作者:** Chuan Guo; Inwoo Hwang; Jian Wang; Bing Zhou
>
> **备注:** Project Webpage: https://snap-research.github.io/SnapMoGen/
>
> **摘要:** Text-to-motion generation has experienced remarkable progress in recent years. However, current approaches remain limited to synthesizing motion from short or general text prompts, primarily due to dataset constraints. This limitation undermines fine-grained controllability and generalization to unseen prompts. In this paper, we introduce SnapMoGen, a new text-motion dataset featuring high-quality motion capture data paired with accurate, expressive textual annotations. The dataset comprises 20K motion clips totaling 44 hours, accompanied by 122K detailed textual descriptions averaging 48 words per description (vs. 12 words of HumanML3D). Importantly, these motion clips preserve original temporal continuity as they were in long sequences, facilitating research in long-term motion generation and blending. We also improve upon previous generative masked modeling approaches. Our model, MoMask++, transforms motion into multi-scale token sequences that better exploit the token capacity, and learns to generate all tokens using a single generative masked transformer. MoMask++ achieves state-of-the-art performance on both HumanML3D and SnapMoGen benchmarks. Additionally, we demonstrate the ability to process casual user prompts by employing an LLM to reformat inputs to align with the expressivity and narration style of SnapMoGen. Project webpage: https://snap-research.github.io/SnapMoGen/
>
---
#### [new 116] Demystifying Flux Architecture
- **分类: cs.CV**

- **简介: 该论文旨在解析FLUX.1文本到图像生成模型的架构，属于模型逆向工程任务。为支持其在研究中的应用，作者通过源码分析，揭示了这一未公开技术细节的SOTA模型的设计与训练设置。**

- **链接: [http://arxiv.org/pdf/2507.09595v1](http://arxiv.org/pdf/2507.09595v1)**

> **作者:** Or Greenberg
>
> **摘要:** FLUX.1 is a diffusion-based text-to-image generation model developed by Black Forest Labs, designed to achieve faithful text-image alignment while maintaining high image quality and diversity. FLUX is considered state-of-the-art in text-to-image generation, outperforming popular models such as Midjourney, DALL-E 3, Stable Diffusion 3 (SD3), and SDXL. Although publicly available as open source, the authors have not released official technical documentation detailing the model's architecture or training setup. This report summarizes an extensive reverse-engineering effort aimed at demystifying FLUX's architecture directly from its source code, to support its adoption as a backbone for future research and development. This document is an unofficial technical report and is not published or endorsed by the original developers or their affiliated institutions.
>
---
#### [new 117] Dynamic Inter-Class Confusion-Aware Encoder for Audio-Visual Fusion in Human Activity Recognition
- **分类: cs.CV**

- **简介: 该论文属于音频-视频融合的人类活动识别任务，旨在解决易混淆类别区分不足的问题。作者提出了动态类别间混淆感知编码器（DICCAE），通过细粒度对齐和自监督预训练策略，提升模型辨别相似活动的能力，并在VGGSound数据集上取得优异性能。**

- **链接: [http://arxiv.org/pdf/2507.09323v1](http://arxiv.org/pdf/2507.09323v1)**

> **作者:** Kaixuan Cong; Yifan Wang; Rongkun Xue; Yuyang Jiang; Yiming Feng; Jing Yang
>
> **摘要:** Humans do not understand individual events in isolation; rather, they generalize concepts within classes and compare them to others. Existing audio-video pre-training paradigms only focus on the alignment of the overall audio-video modalities, without considering the reinforcement of distinguishing easily confused classes through cognitive induction and contrast during training. This paper proposes the Dynamic Inter-Class Confusion-Aware Encoder (DICCAE), an encoder that aligns audio-video representations at a fine-grained, category-level. DICCAE addresses category confusion by dynamically adjusting the confusion loss based on inter-class confusion degrees, thereby enhancing the model's ability to distinguish between similar activities. To further extend the application of DICCAE, we also introduce a novel training framework that incorporates both audio and video modalities, as well as their fusion. To mitigate the scarcity of audio-video data in the human activity recognition task, we propose a cluster-guided audio-video self-supervised pre-training strategy for DICCAE. DICCAE achieves near state-of-the-art performance on the VGGSound dataset, with a top-1 accuracy of 65.5%. We further evaluate its feature representation quality through extensive ablation studies, validating the necessity of each module.
>
---
#### [new 118] GreenCrossingAI: A Camera Trap/Computer Vision Pipeline for Environmental Science Research Groups
- **分类: cs.CV; cs.LG**

- **简介: 论文提出了GreenCrossingAI，一个适用于资源有限科研团队的相机陷阱数据处理流程。它属于环境科学中的野生动物监测任务，旨在解决相机陷阱数据处理中数据量大、标注难、环境干扰和ML/AI工具集成困难等问题，通过低资源本地解决方案实现数据传输、推理与评估。**

- **链接: [http://arxiv.org/pdf/2507.09410v1](http://arxiv.org/pdf/2507.09410v1)**

> **作者:** Bernie Boscoe; Shawn Johnson; Andrea Osborn; Chandler Campbell; Karen Mager
>
> **备注:** This is the preprint version of the paper in Practice and Experience in Advanced Research Computing, PEARC25
>
> **摘要:** Camera traps have long been used by wildlife researchers to monitor and study animal behavior, population dynamics, habitat use, and species diversity in a non-invasive and efficient manner. While data collection from the field has increased with new tools and capabilities, methods to develop, process, and manage the data, especially the adoption of ML/AI tools, remain challenging. These challenges include the sheer volume of data generated, the need for accurate labeling and annotation, variability in environmental conditions affecting data quality, and the integration of ML/AI tools into existing workflows that often require domain-specific customization and computational resources. This paper provides a guide to a low-resource pipeline to process camera trap data on-premise, incorporating ML/AI capabilities tailored for small research groups with limited resources and computational expertise. By focusing on practical solutions, the pipeline offers accessible approaches for data transmission, inference, and evaluation, enabling researchers to discover meaningful insights from their ever-increasing camera trap datasets.
>
---
#### [new 119] Learning and Transferring Better with Depth Information in Visual Reinforcement Learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉强化学习任务，旨在提升智能体在不同环境中的泛化能力。通过融合RGB与深度信息，采用基于视觉Transformer的网络结构，并引入对比无监督学习策略以提高样本效率，同时设计课程学习方案促进仿真到现实的迁移。**

- **链接: [http://arxiv.org/pdf/2507.09180v1](http://arxiv.org/pdf/2507.09180v1)**

> **作者:** Zichun Xu; Yuntao Li; Zhaomin Wang; Lei Zhuang; Guocai Yang; Jingdong Zhao
>
> **摘要:** Depth information is robust to scene appearance variations and inherently carries 3D spatial details. In this paper, a visual backbone based on the vision transformer is proposed to fuse RGB and depth modalities for enhancing generalization. Different modalities are first processed by separate CNN stems, and the combined convolutional features are delivered to the scalable vision transformer to obtain visual representations. Moreover, a contrastive unsupervised learning scheme is designed with masked and unmasked tokens to accelerate the sample efficiency during the reinforcement learning progress. For sim2real transfer, a flexible curriculum learning schedule is developed to deploy domain randomization over training processes.
>
---
#### [new 120] PRISM: Reducing Spurious Implicit Biases in Vision-Language Models with LLM-Guided Embedding Projection
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉-语言模型（VLM）中的偏差缓解任务，旨在解决模型因训练数据偏差导致的预测偏倚问题。作者提出PRISM方法，利用大语言模型生成包含虚假相关性的描述，并通过对比式去偏损失学习嵌入投影，减少偏差同时保持图文对齐。**

- **链接: [http://arxiv.org/pdf/2507.08979v1](http://arxiv.org/pdf/2507.08979v1)**

> **作者:** Mahdiyar Molahasani; Azadeh Motamedi; Michael Greenspan; Il-Min Kim; Ali Etemad
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** We introduce Projection-based Reduction of Implicit Spurious bias in vision-language Models (PRISM), a new data-free and task-agnostic solution for bias mitigation in VLMs like CLIP. VLMs often inherit and amplify biases in their training data, leading to skewed predictions. PRISM is designed to debias VLMs without relying on predefined bias categories or additional external data. It operates in two stages: first, an LLM is prompted with simple class prompts to generate scene descriptions that contain spurious correlations. Next, PRISM uses our novel contrastive-style debiasing loss to learn a projection that maps the embeddings onto a latent space that minimizes spurious correlations while preserving the alignment between image and text embeddings.Extensive experiments demonstrate that PRISM outperforms current debiasing methods on the commonly used Waterbirds and CelebA datasets We make our code public at: https://github.com/MahdiyarMM/PRISM.
>
---
#### [new 121] RadEyeVideo: Enhancing general-domain Large Vision Language Model for chest X-ray analysis with video representations of eye gaze
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决如何提升通用视觉语言模型在胸部X光分析中的表现。论文提出RadEyeVideo方法，将放射科医生的眼动数据以视频形式融入模型输入，有效捕捉眼动的时空特征。实验表明，该方法显著提升了报告生成与疾病诊断性能，优于专业医学模型。**

- **链接: [http://arxiv.org/pdf/2507.09097v1](http://arxiv.org/pdf/2507.09097v1)**

> **作者:** Yunsoo Kim; Jinge Wu; Honghan Wu
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated promising performance in chest X-ray (CXR) analysis. To enhance human-computer interaction, several studies have incorporated radiologists' eye gaze, typically through heatmaps or textual prompts. However, these methods often overlook the sequential order of eye movements, which could provide valuable insights by highlighting both the areas of interest and the order in which they are examined. In this work, we propose a novel approach called RadEyeVideo that integrates radiologists' eye-fixation data as a video sequence, capturing both the temporal and spatial dynamics of their gaze. We evaluate this method in CXR report generation and disease diagnosis using three general-domain, open-source LVLMs with video input capabilities. When prompted with eye-gaze videos, model performance improves by up to 24.6% in the report generation task and on average 15.2% for both tasks using scaled evaluation metrics. Notably, RadEyeVideo enhanced an open-domain LVLM model, LLaVA-OneVision, to surpass task-specific medical LVLMs such as MAIRA-2 and CheXagent, trained on large Chest X-ray data. This work highlights that domain expert's knowledge (eye-gaze information in this case), when effectively integrated with LVLMs, can significantly enhance general-domain models' capabilities in clinical tasks. RadEyeVideo is a step toward a scalable human-centered approach of utilizing LVLMs in medical image analytics.
>
---
#### [new 122] Cameras as Relative Positional Encoding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多视角计算机视觉任务，旨在解决如何有效利用相机几何信息提升Transformer模型的3D感知能力。论文比较了多种将相机参数融入Transformer的方法，提出了一种新的相对位置编码PRoPE，能够更全面地捕捉相机视锥信息（包括内参和外参），并在多个任务（如新视角合成、立体深度估计等）中验证了其有效性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.10496v1](http://arxiv.org/pdf/2507.10496v1)**

> **作者:** Ruilong Li; Brent Yi; Junchen Liu; Hang Gao; Yi Ma; Angjoo Kanazawa
>
> **备注:** Project Page: https://www.liruilong.cn/prope/
>
> **摘要:** Transformers are increasingly prevalent for multi-view computer vision tasks, where geometric relationships between viewpoints are critical for 3D perception. To leverage these relationships, multi-view transformers must use camera geometry to ground visual tokens in 3D space. In this work, we compare techniques for conditioning transformers on cameras: token-level raymap encodings, attention-level relative pose encodings, and a new relative encoding we propose -- Projective Positional Encoding (PRoPE) -- that captures complete camera frustums, both intrinsics and extrinsics, as a relative positional encoding. Our experiments begin by showing how relative camera conditioning improves performance in feedforward novel view synthesis, with further gains from PRoPE. This holds across settings: scenes with both shared and varying intrinsics, when combining token- and attention-level conditioning, and for generalization to inputs with out-of-distribution sequence lengths and camera intrinsics. We then verify that these benefits persist for different tasks, stereo depth estimation and discriminative spatial cognition, as well as larger model sizes.
>
---
#### [new 123] Measuring the Impact of Rotation Equivariance on Aerial Object Detection
- **分类: cs.CV**

- **简介: 该论文属于遥感图像目标检测任务，旨在解决旋转等变性对航拍目标检测性能的影响问题。作者构建了严格旋转等变的主干和颈部网络，并提出多分支头网络，在保持高精度的同时降低参数量，最终提出MessDet方法，在多个航拍数据集上取得先进性能。**

- **链接: [http://arxiv.org/pdf/2507.09896v1](http://arxiv.org/pdf/2507.09896v1)**

> **作者:** Xiuyu Wu; Xinhao Wang; Xiubin Zhu; Lan Yang; Jiyuan Liu; Xingchen Hu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Due to the arbitrary orientation of objects in aerial images, rotation equivariance is a critical property for aerial object detectors. However, recent studies on rotation-equivariant aerial object detection remain scarce. Most detectors rely on data augmentation to enable models to learn approximately rotation-equivariant features. A few detectors have constructed rotation-equivariant networks, but due to the breaking of strict rotation equivariance by typical downsampling processes, these networks only achieve approximately rotation-equivariant backbones. Whether strict rotation equivariance is necessary for aerial image object detection remains an open question. In this paper, we implement a strictly rotation-equivariant backbone and neck network with a more advanced network structure and compare it with approximately rotation-equivariant networks to quantitatively measure the impact of rotation equivariance on the performance of aerial image detectors. Additionally, leveraging the inherently grouped nature of rotation-equivariant features, we propose a multi-branch head network that reduces the parameter count while improving detection accuracy. Based on the aforementioned improvements, this study proposes the Multi-branch head rotation-equivariant single-stage Detector (MessDet), which achieves state-of-the-art performance on the challenging aerial image datasets DOTA-v1.0, DOTA-v1.5 and DIOR-R with an exceptionally low parameter count.
>
---
#### [new 124] Advancing Reliable Test-Time Adaptation of Vision-Language Models under Visual Variations
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的测试时自适应任务，旨在解决模型在无标注数据情况下应对分布偏移的可靠性问题。论文提出ReTA方法，包括一致性感知的熵重加权和多样性驱动的分布校准策略，以提升缓存质量和决策边界灵活性，从而增强模型在视觉变化下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.09500v1](http://arxiv.org/pdf/2507.09500v1)**

> **作者:** Yiwen Liang; Hui Chen; Yizhe Xiong; Zihan Zhou; Mengyao Lyu; Zijia Lin; Shuaicheng Niu; Sicheng Zhao; Jungong Han; Guiguang Ding
>
> **备注:** Accepted at the 33rd ACM International Conference on Multimedia(ACM MM 2025)
>
> **摘要:** Vision-language models (VLMs) exhibit remarkable zero-shot capabilities but struggle with distribution shifts in downstream tasks when labeled data is unavailable, which has motivated the development of Test-Time Adaptation (TTA) to improve VLMs' performance during inference without annotations. Among various TTA approaches, cache-based methods show promise by preserving historical knowledge from low-entropy samples in a dynamic cache and fostering efficient adaptation. However, these methods face two critical reliability challenges: (1) entropy often becomes unreliable under distribution shifts, causing error accumulation in the cache and degradation in adaptation performance; (2) the final predictions may be unreliable due to inflexible decision boundaries that fail to accommodate large downstream shifts. To address these challenges, we propose a Reliable Test-time Adaptation (ReTA) method that integrates two complementary strategies to enhance reliability from two perspectives. First, to mitigate the unreliability of entropy as a sample selection criterion for cache construction, we introduce Consistency-aware Entropy Reweighting (CER), which incorporates consistency constraints to weight entropy during cache updating. While conventional approaches rely solely on low entropy for cache prioritization and risk introducing noise, our method leverages predictive consistency to maintain a high-quality cache and facilitate more robust adaptation. Second, we present Diversity-driven Distribution Calibration (DDC), which models class-wise text embeddings as multivariate Gaussian distributions, enabling adaptive decision boundaries for more accurate predictions across visually diverse content. Extensive experiments demonstrate that ReTA consistently outperforms state-of-the-art methods, particularly under challenging real-world distribution shifts.
>
---
#### [new 125] Stereo-based 3D Anomaly Object Detection for Autonomous Driving: A New Dataset and Baseline
- **分类: cs.CV**

- **简介: 该论文属于3D目标检测任务，旨在解决自动驾驶中罕见异常物体的检测问题。现有模型在封闭数据集训练下难以泛化，易漏检异常物体。作者提出S3AD算法，解耦2D与3D训练，并引入基于前景置信度的异常评分方法。同时构建了包含97个新类别的KITTI-AR数据集，用于验证模型在零样本场景下的异常检测能力。**

- **链接: [http://arxiv.org/pdf/2507.09214v1](http://arxiv.org/pdf/2507.09214v1)**

> **作者:** Shiyi Mu; Zichong Gu; Hanqi Lyu; Yilin Gao; Shugong Xu
>
> **备注:** under review
>
> **摘要:** 3D detection technology is widely used in the field of autonomous driving, with its application scenarios gradually expanding from enclosed highways to open conventional roads. For rare anomaly categories that appear on the road, 3D detection models trained on closed sets often misdetect or fail to detect anomaly objects. To address this risk, it is necessary to enhance the generalization ability of 3D detection models for targets of arbitrary shapes and to possess the capability to filter out anomalies. The generalization of 3D detection is limited by two factors: the coupled training of 2D and 3D, and the insufficient diversity in the scale distribution of training samples. This paper proposes a Stereo-based 3D Anomaly object Detection (S3AD) algorithm, which decouples the training strategy of 3D and 2D to release the generalization ability for arbitrary 3D foreground detection, and proposes an anomaly scoring algorithm based on foreground confidence prediction, achieving target-level anomaly scoring. In order to further verify and enhance the generalization of anomaly detection, we use a 3D rendering method to synthesize two augmented reality binocular stereo 3D detection datasets which named KITTI-AR. KITTI-AR extends upon KITTI by adding 97 new categories, totaling 6k pairs of stereo images. The KITTI-AR-ExD subset includes 39 common categories as extra training data to address the sparse sample distribution issue. Additionally, 58 rare categories form the KITTI-AR-OoD subset, which are not used in training to simulate zero-shot scenarios in real-world settings, solely for evaluating 3D anomaly detection. Finally, the performance of the algorithm and the dataset is verified in the experiments. (Code and dataset can be obtained at https://github.com/xxxx/xxx).
>
---
#### [new 126] Minimizing the Pretraining Gap: Domain-aligned Text-Based Person Retrieval
- **分类: cs.CV**

- **简介: 该论文属于文本检索任务，旨在解决合成数据与真实数据间的领域差异问题。通过提出Domain-aware Diffusion和Multi-granularity Relation Alignment方法，分别实现图像级和区域级的领域自适应，有效缩小预训练与微调间的差距，提升模型在真实场景中的表现。**

- **链接: [http://arxiv.org/pdf/2507.10195v1](http://arxiv.org/pdf/2507.10195v1)**

> **作者:** Shuyu Yang; Yaxiong Wang; Yongrui Li; Li Zhu; Zhedong Zheng
>
> **摘要:** In this work, we focus on text-based person retrieval, which aims to identify individuals based on textual descriptions. Given the significant privacy issues and the high cost associated with manual annotation, synthetic data has become a popular choice for pretraining models, leading to notable advancements. However, the considerable domain gap between synthetic pretraining datasets and real-world target datasets, characterized by differences in lighting, color, and viewpoint, remains a critical obstacle that hinders the effectiveness of the pretrain-finetune paradigm. To bridge this gap, we introduce a unified text-based person retrieval pipeline considering domain adaptation at both image and region levels. In particular, it contains two primary components, i.e., Domain-aware Diffusion (DaD) for image-level adaptation and Multi-granularity Relation Alignment (MRA) for region-level adaptation. As the name implies, Domain-aware Diffusion is to migrate the distribution of images from the pretraining dataset domain to the target real-world dataset domain, e.g., CUHK-PEDES. Subsequently, MRA performs a meticulous region-level alignment by establishing correspondences between visual regions and their descriptive sentences, thereby addressing disparities at a finer granularity. Extensive experiments show that our dual-level adaptation method has achieved state-of-the-art results on the CUHK-PEDES, ICFG-PEDES, and RSTPReid datasets, outperforming existing methodologies. The dataset, model, and code are available at https://github.com/Shuyu-XJTU/MRA.
>
---
#### [new 127] Improving Remote Sensing Classification using Topological Data Analysis and Convolutional Neural Networks
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于遥感图像分类任务，旨在解决卷积神经网络对纹理特征的局部偏差问题。通过引入拓扑数据分析（TDA）方法提取几何特征，并与ResNet18模型结合，在EuroSAT和RESISC45数据集上提升了分类精度，取得了优于更大模型的效果。**

- **链接: [http://arxiv.org/pdf/2507.10381v1](http://arxiv.org/pdf/2507.10381v1)**

> **作者:** Aaryam Sharma
>
> **备注:** 9 pages, 8 figures
>
> **摘要:** Topological data analysis (TDA) is a relatively new field that is gaining rapid adoption due to its robustness and ability to effectively describe complex datasets by quantifying geometric information. In imaging contexts, TDA typically models data as filtered cubical complexes from which we can extract discriminative features using persistence homology. Meanwhile, convolutional neural networks (CNNs) have been shown to be biased towards texture based local features. To address this limitation, we propose a TDA feature engineering pipeline and a simple method to integrate topological features with deep learning models on remote sensing classification. Our method improves the performance of a ResNet18 model on the EuroSAT dataset by 1.44% achieving 99.33% accuracy, which surpasses all previously reported single-model accuracies, including those with larger architectures, such as ResNet50 (2x larger) and XL Vision Transformers (197x larger). We additionally show that our method's accuracy is 1.82% higher than our ResNet18 baseline on the RESISC45 dataset. To our knowledge, this is the first application of TDA features in satellite scene classification with deep learning. This demonstrates that TDA features can be integrated with deep learning models, even on datasets without explicit topological structures, thereby increasing the applicability of TDA. A clean implementation of our method will be made publicly available upon publication.
>
---
#### [new 128] OpenHuman4D: Open-Vocabulary 4D Human Parsing
- **分类: cs.CV**

- **简介: 该论文属于4D人体解析任务，旨在解决现有方法依赖封闭数据集和推理速度慢的问题。作者提出OpenHuman4D框架，结合视频目标跟踪、掩码验证和4D融合模块，实现高效、开放词汇的4D人体解析，推理速度提升最高达93.3%。**

- **链接: [http://arxiv.org/pdf/2507.09880v1](http://arxiv.org/pdf/2507.09880v1)**

> **作者:** Keito Suzuki; Bang Du; Runfa Blark Li; Kunyao Chen; Lei Wang; Peng Liu; Ning Bi; Truong Nguyen
>
> **摘要:** Understanding dynamic 3D human representation has become increasingly critical in virtual and extended reality applications. However, existing human part segmentation methods are constrained by reliance on closed-set datasets and prolonged inference times, which significantly restrict their applicability. In this paper, we introduce the first 4D human parsing framework that simultaneously addresses these challenges by reducing the inference time and introducing open-vocabulary capabilities. Building upon state-of-the-art open-vocabulary 3D human parsing techniques, our approach extends the support to 4D human-centric video with three key innovations: 1) We adopt mask-based video object tracking to efficiently establish spatial and temporal correspondences, avoiding the necessity of segmenting all frames. 2) A novel Mask Validation module is designed to manage new target identification and mitigate tracking failures. 3) We propose a 4D Mask Fusion module, integrating memory-conditioned attention and logits equalization for robust embedding fusion. Extensive experiments demonstrate the effectiveness and flexibility of the proposed method on 4D human-centric parsing tasks, achieving up to 93.3% acceleration compared to the previous state-of-the-art method, which was limited to parsing fixed classes.
>
---
#### [new 129] Hybrid Autoregressive-Diffusion Model for Real-Time Streaming Sign Language Production
- **分类: cs.CV**

- **简介: 该论文属于手语生成任务，旨在解决实时流式生成中的错误累积与质量下降问题。论文提出了一种融合自回归与扩散模型的方法，并设计了多尺度姿态表示模块和置信度感知因果注意力机制，以提升生成质量与实时性。**

- **链接: [http://arxiv.org/pdf/2507.09105v1](http://arxiv.org/pdf/2507.09105v1)**

> **作者:** Maoxiao Ye; Xinfeng Ye; Mano Manoharan
>
> **摘要:** Earlier Sign Language Production (SLP) models typically relied on autoregressive methods that generate output tokens one by one, which inherently provide temporal alignment. Although techniques like Teacher Forcing can prevent model collapse during training, they still cannot solve the problem of error accumulation during inference, since ground truth is unavailable at that stage. In contrast, more recent approaches based on diffusion models leverage step-by-step denoising to enable high-quality generation. However, the iterative nature of these models and the requirement to denoise entire sequences limit their applicability in real-time tasks like SLP. To address it, we apply a hybrid approach combining autoregressive and diffusion models to SLP for the first time, leveraging the strengths of both models in sequential dependency modeling and output refinement. To capture fine-grained body movements, we design a Multi-Scale Pose Representation module that separately extracts detailed features from distinct articulators and integrates them via a Multi-Scale Fusion module. Furthermore, we introduce a Confidence-Aware Causal Attention mechanism that utilizes joint-level confidence scores to dynamically guide the pose generation process, improving accuracy and robustness. Extensive experiments on the PHOENIX14T and How2Sign datasets demonstrate the effectiveness of our method in both generation quality and real-time streaming efficiency.
>
---
#### [new 130] MI CAM: Mutual Information Weighted Activation Mapping for Causal Visual Explanations of Convolutional Neural Networks
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于计算机视觉任务，旨在解释卷积神经网络的推理机制。为实现此目标，作者提出了一种新的可视化解释方法MI CAM，通过互信息加权激活图生成显著性可视化，并利用反事实分析验证因果解释。**

- **链接: [http://arxiv.org/pdf/2507.09092v1](http://arxiv.org/pdf/2507.09092v1)**

> **作者:** Ram S Iyer; Narayan S Iyer; Rugmini Ammal P
>
> **备注:** 12 pages, 10 figures
>
> **摘要:** With the intervention of machine vision in our crucial day to day necessities including healthcare and automated power plants, attention has been drawn to the internal mechanisms of convolutional neural networks, and the reason why the network provides specific inferences. This paper proposes a novel post-hoc visual explanation method called MI CAM based on activation mapping. Differing from previous class activation mapping based approaches, MI CAM produces saliency visualizations by weighing each feature map through its mutual information with the input image and the final result is generated by a linear combination of weights and activation maps. It also adheres to producing causal interpretations as validated with the help of counterfactual analysis. We aim to exhibit the visual performance and unbiased justifications for the model inferencing procedure achieved by MI CAM. Our approach works at par with all state-of-the-art methods but particularly outperforms some in terms of qualitative and quantitative measures. The implementation of proposed method can be found on https://anonymous.4open.science/r/MI-CAM-4D27
>
---
#### [new 131] Video Inference for Human Mesh Recovery with Vision Transformer
- **分类: cs.CV**

- **简介: 该论文属于人体网格重建任务，旨在从视频中恢复人体三维网格。为解决单帧重建存在的歧义问题，作者提出HMR-ViT，首次同时融合时间信息与运动学关系。通过构建时空特征图像并应用视觉Transformer，提升了重建精度，在3DPW和Human3.6M数据集上表现优异。**

- **链接: [http://arxiv.org/pdf/2507.08981v1](http://arxiv.org/pdf/2507.08981v1)**

> **作者:** Hanbyel Cho; Jaesung Ahn; Yooshin Cho; Junmo Kim
>
> **备注:** Accepted to IEEE FG 2023
>
> **摘要:** Human Mesh Recovery (HMR) from an image is a challenging problem because of the inherent ambiguity of the task. Existing HMR methods utilized either temporal information or kinematic relationships to achieve higher accuracy, but there is no method using both. Hence, we propose "Video Inference for Human Mesh Recovery with Vision Transformer (HMR-ViT)" that can take into account both temporal and kinematic information. In HMR-ViT, a Temporal-kinematic Feature Image is constructed using feature vectors obtained from video frames by an image encoder. When generating the feature image, we use a Channel Rearranging Matrix (CRM) so that similar kinematic features could be located spatially close together. The feature image is then further encoded using Vision Transformer, and the SMPL pose and shape parameters are finally inferred using a regression network. Extensive evaluation on the 3DPW and Human3.6M datasets indicates that our method achieves a competitive performance in HMR.
>
---
#### [new 132] VDInstruct: Zero-Shot Key Information Extraction via Content-Aware Vision Tokenization
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉文档关键信息抽取任务，旨在解决现有模型在密集文档中效率低、依赖冗余图像分块的问题。作者提出VDInstruct，通过内容感知的图像分块策略和三阶段训练方法，在减少图像分块数量的同时提升准确率，实现了零样本下的优越性能。**

- **链接: [http://arxiv.org/pdf/2507.09531v1](http://arxiv.org/pdf/2507.09531v1)**

> **作者:** Son Nguyen; Giang Nguyen; Hung Dao; Thao Do; Daeyoung Kim
>
> **备注:** Under Review
>
> **摘要:** Key Information Extraction (KIE) underpins the understanding of visual documents (e.g., receipts and contracts) by extracting precise semantic content and accurately capturing spatial structure. Yet existing multimodal large language models (MLLMs) often perform poorly on dense documents and rely on vision tokenization approaches that scale with image size, leading to redundant computation and memory inefficiency. To address these challenges, we introduce VDInstruct, an MLLM that separates spatial region detection from semantic feature extraction. Central to our model is a content-aware tokenization strategy: rather than fragmenting the entire image uniformly, it generates tokens in proportion to document complexity, preserving critical structure while eliminating wasted tokens. Leveraging a three-stage training paradigm, our model achieves state-of-the-art (SOTA) results on KIE benchmarks, matching or exceeding the accuracy of leading approaches while reducing the number of image tokens by roughly 3.6x. In zero-shot evaluations, VDInstruct surpasses strong baselines-such as DocOwl 1.5-by +5.5 F1 points, highlighting its robustness to unseen documents. These findings show that content-aware tokenization combined with explicit layout modeling offers a promising direction forward for document understanding. Data, source code, and model weights will be made publicly available.
>
---
#### [new 133] Leveraging Swin Transformer for enhanced diagnosis of Alzheimer's disease using multi-shell diffusion MRI
- **分类: cs.CV; q-bio.NC; q-bio.QM**

- **简介: 该论文属于医学影像分析任务，旨在解决阿尔茨海默病的早期诊断和淀粉样蛋白积累检测问题。研究使用多壳扩散MRI数据，结合Swin Transformer模型进行分类，并利用迁移学习与低秩适应提升性能，在有限数据下实现高准确率分类，并通过可视化识别关键脑区。**

- **链接: [http://arxiv.org/pdf/2507.09996v1](http://arxiv.org/pdf/2507.09996v1)**

> **作者:** Quentin Dessain; Nicolas Delinte; Bernard Hanseeuw; Laurence Dricot; Benoît Macq
>
> **摘要:** Objective: This study aims to support early diagnosis of Alzheimer's disease and detection of amyloid accumulation by leveraging the microstructural information available in multi-shell diffusion MRI (dMRI) data, using a vision transformer-based deep learning framework. Methods: We present a classification pipeline that employs the Swin Transformer, a hierarchical vision transformer model, on multi-shell dMRI data for the classification of Alzheimer's disease and amyloid presence. Key metrics from DTI and NODDI were extracted and projected onto 2D planes to enable transfer learning with ImageNet-pretrained models. To efficiently adapt the transformer to limited labeled neuroimaging data, we integrated Low-Rank Adaptation. We assessed the framework on diagnostic group prediction (cognitively normal, mild cognitive impairment, Alzheimer's disease dementia) and amyloid status classification. Results: The framework achieved competitive classification results within the scope of multi-shell dMRI-based features, with the best balanced accuracy of 95.2% for distinguishing cognitively normal individuals from those with Alzheimer's disease dementia using NODDI metrics. For amyloid detection, it reached 77.2% balanced accuracy in distinguishing amyloid-positive mild cognitive impairment/Alzheimer's disease dementia subjects from amyloid-negative cognitively normal subjects, and 67.9% for identifying amyloid-positive individuals among cognitively normal subjects. Grad-CAM-based explainability analysis identified clinically relevant brain regions, including the parahippocampal gyrus and hippocampus, as key contributors to model predictions. Conclusion: This study demonstrates the promise of diffusion MRI and transformer-based architectures for early detection of Alzheimer's disease and amyloid pathology, supporting biomarker-driven diagnostics in data-limited biomedical settings.
>
---
#### [new 134] FIX-CLIP: Dual-Branch Hierarchical Contrastive Learning via Synthetic Captions for Better Understanding of Long Text
- **分类: cs.CV**

- **简介: 该论文属于多模态任务，旨在解决CLIP模型在处理长文本输入时效果不佳的问题。作者提出了FIX-CLIP，通过双分支训练、区域提示和层次特征对齐模块提升长文本理解能力。实验表明其在长、短文本检索任务上均表现优异，并可有效应用于扩散模型的长文本生成。**

- **链接: [http://arxiv.org/pdf/2507.10095v1](http://arxiv.org/pdf/2507.10095v1)**

> **作者:** Bingchao Wang; Zhiwei Ning; Jianyu Ding; Xuanang Gao; Yin Li; Dongsheng Jiang; Jie Yang; Wei Liu
>
> **摘要:** CLIP has shown promising performance across many short-text tasks in a zero-shot manner. However, limited by the input length of the text encoder, CLIP struggles on under-stream tasks with long-text inputs (>77 tokens). To remedy this issue, we propose FIX-CLIP which includes three novel modules: (1) A dual-branch training pipeline that aligns short and long texts with masked and raw images respectively, which boosts the long-text representation while preserving the short-text ability. (2) Multiple learnable regional prompts with unidirectional masks in Transformer layers for regional information extraction. (3) A hierarchical feature alignment module in the intermediate encoder layers to promote the consistency of multi-scale features. Furthermore, we collect 30M images and utilize existing MLLMs to synthesize long-text captions for training. Extensive experiments show that FIX-CLIP achieves state-of-the-art performance on both long-text and short-text retrieval benchmarks. For downstream applications, we reveal that FIX-CLIP's text encoder delivers promising performance in a plug-and-play manner for diffusion models with long-text input.
>
---
#### [new 135] AlphaVAE: Unified End-to-End RGBA Image Reconstruction and Generation with Alpha-Aware Representation Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像生成与重建任务，旨在解决RGBA透明图像生成与重建中缺乏大规模数据集和有效模型的问题。作者提出了AlphaVAE模型和Alpha数据集，实现了更优的透明图像处理效果。**

- **链接: [http://arxiv.org/pdf/2507.09308v1](http://arxiv.org/pdf/2507.09308v1)**

> **作者:** Zile Wang; Hao Yu; Jiabo Zhan; Chun Yuan
>
> **摘要:** Recent advances in latent diffusion models have achieved remarkable results in high-fidelity RGB image synthesis by leveraging pretrained VAEs to compress and reconstruct pixel data at low computational cost. However, the generation of transparent or layered content (RGBA image) remains largely unexplored, due to the lack of large-scale benchmarks. In this work, we propose ALPHA, the first comprehensive RGBA benchmark that adapts standard RGB metrics to four-channel images via alpha blending over canonical backgrounds. We further introduce ALPHAVAE, a unified end-to-end RGBA VAE that extends a pretrained RGB VAE by incorporating a dedicated alpha channel. The model is trained with a composite objective that combines alpha-blended pixel reconstruction, patch-level fidelity, perceptual consistency, and dual KL divergence constraints to ensure latent fidelity across both RGB and alpha representations. Our RGBA VAE, trained on only 8K images in contrast to 1M used by prior methods, achieves a +4.9 dB improvement in PSNR and a +3.2% increase in SSIM over LayerDiffuse in reconstruction. It also enables superior transparent image generation when fine-tuned within a latent diffusion framework. Our code, data, and models are released on https://github.com/o0o0o00o0/AlphaVAE for reproducibility.
>
---
#### [new 136] Geo-RepNet: Geometry-Aware Representation Learning for Surgical Phase Recognition in Endoscopic Submucosal Dissection
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于手术阶段识别任务，旨在解决微创手术中视觉相似性高和结构线索不足导致的识别难题。论文提出了Geo-RepNet模型，融合RGB图像与深度信息，引入几何感知模块，提升复杂手术场景下的识别性能。实验表明其在自建ESD数据集上达到最优效果。**

- **链接: [http://arxiv.org/pdf/2507.09294v1](http://arxiv.org/pdf/2507.09294v1)**

> **作者:** Rui Tang; Haochen Yin; Guankun Wang; Long Bai; An Wang; Huxin Gao; Jiazheng Wang; Hongliang Ren
>
> **备注:** IEEE ICIA 2025
>
> **摘要:** Surgical phase recognition plays a critical role in developing intelligent assistance systems for minimally invasive procedures such as Endoscopic Submucosal Dissection (ESD). However, the high visual similarity across different phases and the lack of structural cues in RGB images pose significant challenges. Depth information offers valuable geometric cues that can complement appearance features by providing insights into spatial relationships and anatomical structures. In this paper, we pioneer the use of depth information for surgical phase recognition and propose Geo-RepNet, a geometry-aware convolutional framework that integrates RGB image and depth information to enhance recognition performance in complex surgical scenes. Built upon a re-parameterizable RepVGG backbone, Geo-RepNet incorporates the Depth-Guided Geometric Prior Generation (DGPG) module that extracts geometry priors from raw depth maps, and the Geometry-Enhanced Multi-scale Attention (GEMA) to inject spatial guidance through geometry-aware cross-attention and efficient multi-scale aggregation. To evaluate the effectiveness of our approach, we construct a nine-phase ESD dataset with dense frame-level annotations from real-world ESD videos. Extensive experiments on the proposed dataset demonstrate that Geo-RepNet achieves state-of-the-art performance while maintaining robustness and high computational efficiency under complex and low-texture surgical environments.
>
---
#### [new 137] The Power of Certainty: How Confident Models Lead to Better Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决现有深度学习模型因参数过多易过拟合、泛化能力差的问题。作者提出一种基于置信度的自蒸馏方法，在训练中利用前一次迭代数据提升性能，无需额外计算或存储资源。实验表明该方法在多个临床数据集中表现优异。**

- **链接: [http://arxiv.org/pdf/2507.10490v1](http://arxiv.org/pdf/2507.10490v1)**

> **作者:** Tugberk Erol; Tuba Caglikantar; Duygu Sarikaya
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Deep learning models have been proposed for automatic polyp detection and precise segmentation of polyps during colonoscopy procedures. Although these state-of-the-art models achieve high performance, they often require a large number of parameters. Their complexity can make them prone to overfitting, particularly when trained on biased datasets, and can result in poor generalization across diverse datasets. Knowledge distillation and self-distillation are proposed as promising strategies to mitigate the limitations of large, over-parameterized models. These approaches, however, are resource-intensive, often requiring multiple models and significant memory during training. We propose a confidence-based self-distillation approach that outperforms state-of-the-art models by utilizing only previous iteration data storage during training, without requiring extra computation or memory usage during testing. Our approach calculates the loss between the previous and current iterations within a batch using a dynamic confidence coefficient. To evaluate the effectiveness of our approach, we conduct comprehensive experiments on the task of polyp segmentation. Our approach outperforms state-of-the-art models and generalizes well across datasets collected from multiple clinical centers. The code will be released to the public once the paper is accepted.
>
---
#### [new 138] Counterfactual Visual Explanation via Causally-Guided Adversarial Steering
- **分类: cs.CV**

- **简介: 该论文属于视觉解释任务，旨在解决现有反事实解释方法忽视因果关系导致的问题。作者提出CECAS框架，通过因果引导的对抗方法生成高质量反事实图像，避免无关扰动，提升解释的准确性与可信度。**

- **链接: [http://arxiv.org/pdf/2507.09881v1](http://arxiv.org/pdf/2507.09881v1)**

> **作者:** Yiran Qiao; Disheng Liu; Yiren Lu; Yu Yin; Mengnan Du; Jing Ma
>
> **摘要:** Recent work on counterfactual visual explanations has contributed to making artificial intelligence models more explainable by providing visual perturbation to flip the prediction. However, these approaches neglect the causal relationships and the spurious correlations behind the image generation process, which often leads to unintended alterations in the counterfactual images and renders the explanations with limited quality. To address this challenge, we introduce a novel framework CECAS, which first leverages a causally-guided adversarial method to generate counterfactual explanations. It innovatively integrates a causal perspective to avoid unwanted perturbations on spurious factors in the counterfactuals. Extensive experiments demonstrate that our method outperforms existing state-of-the-art approaches across multiple benchmark datasets and ultimately achieves a balanced trade-off among various aspects of validity, sparsity, proximity, and realism.
>
---
#### [new 139] RoHOI: Robustness Benchmark for Human-Object Interaction Detection
- **分类: cs.CV; cs.HC; cs.RO; eess.IV**

- **简介: 该论文属于人类-物体交互（HOI）检测任务，旨在解决模型在真实场景中因环境干扰导致的性能下降问题。作者构建了首个鲁棒性基准RoHOI，包含20种数据干扰类型，并提出SAMPL学习策略提升模型鲁棒性，实验表明其方法优于现有技术。**

- **链接: [http://arxiv.org/pdf/2507.09111v1](http://arxiv.org/pdf/2507.09111v1)**

> **作者:** Di Wen; Kunyu Peng; Kailun Yang; Yufan Chen; Ruiping Liu; Junwei Zheng; Alina Roitberg; Rainer Stiefelhagen
>
> **备注:** Benchmarks, datasets, and code will be made publicly available at https://github.com/Kratos-Wen/RoHOI
>
> **摘要:** Human-Object Interaction (HOI) detection is crucial for robot-human assistance, enabling context-aware support. However, models trained on clean datasets degrade in real-world conditions due to unforeseen corruptions, leading to inaccurate prediction. To address this, we introduce the first robustness benchmark for HOI detection, evaluating model resilience under diverse challenges. Despite advances, current models struggle with environmental variability, occlusion, and noise. Our benchmark, RoHOI, includes 20 corruption types based on HICO-DET and V-COCO datasets and a new robustness-focused metric. We systematically analyze existing models in the related field, revealing significant performance drops under corruptions. To improve robustness, we propose a Semantic-Aware Masking-based Progressive Learning (SAMPL) strategy to guide the model to be optimized based on holistic and partial cues, dynamically adjusting the model's optimization to enhance robust feature learning. Extensive experiments show our approach outperforms state-of-the-art methods, setting a new standard for robust HOI detection. Benchmarks, datasets, and code will be made publicly available at https://github.com/Kratos-Wen/RoHOI.
>
---
#### [new 140] FGSSNet: Feature-Guided Semantic Segmentation of Real World Floorplans
- **分类: cs.CV**

- **简介: 该论文属于语义分割任务，旨在提升真实世界平面图中墙体分割的泛化能力。作者提出FGSSNet，通过引入多头特征提取器，将领域特定特征注入U-Net潜在空间，以指导分割过程。实验表明该方法优于传统U-Net。**

- **链接: [http://arxiv.org/pdf/2507.10343v1](http://arxiv.org/pdf/2507.10343v1)**

> **作者:** Hugo Norrby; Gabriel Färm; Kevin Hernandez-Diaz; Fernando Alonso-Fernandez
>
> **备注:** Accepted at International Workshop on Artificial Intelligence and Pattern Recognition, IWAIPR 2025
>
> **摘要:** We introduce FGSSNet, a novel multi-headed feature-guided semantic segmentation (FGSS) architecture designed to improve the generalization ability of wall segmentation on floorplans. FGSSNet features a U-Net segmentation backbone with a multi-headed dedicated feature extractor used to extract domain-specific feature maps which are injected into the latent space of U-Net to guide the segmentation process. This dedicated feature extractor is trained as an encoder-decoder with selected wall patches, representative of the walls present in the input floorplan, to produce a compressed latent representation of wall patches while jointly trained to predict the wall width. In doing so, we expect that the feature extractor encodes texture and width features of wall patches that are useful to guide the wall segmentation process. Our experiments show increased performance by the use of such injected features in comparison to the vanilla U-Net, highlighting the validity of the proposed approach.
>
---
#### [new 141] Test-Time Canonicalization by Foundation Models for Robust Perception
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉感知任务，旨在解决模型对多样变换的鲁棒性问题。现有方法依赖特定架构或预定义增强训练，泛化能力受限。论文提出FOCAL框架，在测试阶段利用基础模型的先验知识生成典型视图，优化变换以提升鲁棒性，无需重新训练或修改结构。实验表明其在CLIP和SAM上对多种变换具有更好鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.10375v1](http://arxiv.org/pdf/2507.10375v1)**

> **作者:** Utkarsh Singhal; Ryan Feng; Stella X. Yu; Atul Prakash
>
> **备注:** Published at ICML 2025
>
> **摘要:** Real-world visual perception requires invariance to diverse transformations, yet current methods rely heavily on specialized architectures or training on predefined augmentations, limiting generalization. We propose FOCAL, a test-time, data-driven framework that achieves robust perception by leveraging internet-scale visual priors from foundation models. By generating and optimizing candidate transformations toward visually typical, "canonical" views, FOCAL enhances robustness without re-training or architectural changes. Our experiments demonstrate improved robustness of CLIP and SAM across challenging transformations, including 2D/3D rotations, illumination shifts (contrast and color), and day-night variations. We also highlight potential applications in active vision. Our approach challenges the assumption that transform-specific training is necessary, instead offering a scalable path to invariance. Our code is available at: https://github.com/sutkarsh/focal.
>
---
#### [new 142] Transferring Styles for Reduced Texture Bias and Improved Robustness in Semantic Segmentation Networks
- **分类: cs.CV**

- **简介: 该论文属于语义分割任务，旨在减少网络对纹理的依赖，提升模型鲁棒性。作者通过在不同图像区域进行风格迁移，生成多样化训练数据，使模型更关注形状特征。实验表明，该方法有效降低了纹理偏差，并增强了对图像损坏和对抗攻击的鲁棒性，适用于卷积神经网络与Transformer架构。**

- **链接: [http://arxiv.org/pdf/2507.10239v1](http://arxiv.org/pdf/2507.10239v1)**

> **作者:** Ben Hamscher; Edgar Heinert; Annika Mütze; Kira Maag; Matthias Rottmann
>
> **备注:** accepted at ECAI 2025
>
> **摘要:** Recent research has investigated the shape and texture biases of deep neural networks (DNNs) in image classification which influence their generalization capabilities and robustness. It has been shown that, in comparison to regular DNN training, training with stylized images reduces texture biases in image classification and improves robustness with respect to image corruptions. In an effort to advance this line of research, we examine whether style transfer can likewise deliver these two effects in semantic segmentation. To this end, we perform style transfer with style varying across artificial image areas. Those random areas are formed by a chosen number of Voronoi cells. The resulting style-transferred data is then used to train semantic segmentation DNNs with the objective of reducing their dependence on texture cues while enhancing their reliance on shape-based features. In our experiments, it turns out that in semantic segmentation, style transfer augmentation reduces texture bias and strongly increases robustness with respect to common image corruptions as well as adversarial attacks. These observations hold for convolutional neural networks and transformer architectures on the Cityscapes dataset as well as on PASCAL Context, showing the generality of the proposed method.
>
---
#### [new 143] Calibrated and Robust Foundation Models for Vision-Language and Medical Image Tasks Under Distribution Shift
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉-语言与医学图像任务，旨在解决模型在分布偏移下的校准与鲁棒性问题。作者提出了统一框架StaRFM，引入FIP和CMP方法，分别应对协变量偏移和置信度错位问题，并通过理论分析和实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2507.09222v1](http://arxiv.org/pdf/2507.09222v1)**

> **作者:** Behraj Khan; Tahir Syed
>
> **摘要:** Foundation models like CLIP and SAM have transformed computer vision and medical imaging via low-shot transfer learning. However, deployment of these models hindered by two key challenges: \textit{distribution shift} between training and test data, and \textit{confidence misalignment} that leads to overconfident incorrect predictions. These issues manifest differently in vision-language classification and medical segmentation tasks, yet existing solutions remain domain-specific. We propose \textit{StaRFM}, a unified framework addressing both challenges. It introduces a Fisher information penalty (FIP), extended to 3D medical data via patch-wise regularization, to reduce covariate shift in CLIP and SAM embeddings. Additionally, a confidence misalignment penalty (CMP), reformulated for voxel-level predictions, calibrates uncertainty in segmentation tasks. We theoretically derive PAC-Bayes bounds showing FIP controls generalization via the Fisher-Rao norm, while CMP minimizes calibration error through Brier score optimization. StaRFM shows consistent performance like \texttt{+}3.5\% accuracy and 28\% lower ECE on 19 vision datasets (e.g., ImageNet, Office-Home), 84.7\% DSC and 4.8mm HD95 in medical segmentation (e.g., BraTS, ATLAS), and 40\% lower cross-domain performance gap compared to prior benchmarking methods. The framework is plug-and-play, requiring minimal architectural changes for seamless integration with foundation models. Code and models will be released at https://anonymous.4open.science/r/StaRFM-C0CD/README.md
>
---
#### [new 144] DAA*: Deep Angular A Star for Image-based Path Planning
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 该论文属于路径规划任务，旨在解决路径模仿学习中路径平滑性不足的问题。作者提出DAA*方法，结合路径角度自由度（PAF）与A*算法，通过自适应路径平滑提升预测路径与参考路径的相似性，并在多个数据集上验证其有效性。**

- **链接: [http://arxiv.org/pdf/2507.09305v1](http://arxiv.org/pdf/2507.09305v1)**

> **作者:** Zhiwei Xu
>
> **备注:** International Conference on Computer Vision (ICCV), 2025
>
> **摘要:** Path smoothness is often overlooked in path imitation learning from expert demonstrations. In this paper, we introduce a novel learning method, termed deep angular A* (DAA*), by incorporating the proposed path angular freedom (PAF) into A* to improve path similarity through adaptive path smoothness. The PAF aims to explore the effect of move angles on path node expansion by finding the trade-off between their minimum and maximum values, allowing for high adaptiveness for imitation learning. DAA* improves path optimality by closely aligning with the reference path through joint optimization of path shortening and smoothing, which correspond to heuristic distance and PAF, respectively. Throughout comprehensive evaluations on 7 datasets, including 4 maze datasets, 2 video-game datasets, and a real-world drone-view dataset containing 2 scenarios, we demonstrate remarkable improvements of our DAA* over neural A* in path similarity between the predicted and reference paths with a shorter path length when the shortest path is plausible, improving by 9.0% SPR, 6.9% ASIM, and 3.9% PSIM. Furthermore, when jointly learning pathfinding with both path loss and path probability map loss, DAA* significantly outperforms the state-of-the-art TransPath by 6.7% SPR, 6.5% PSIM, and 3.7% ASIM. We also discuss the minor trade-off between path optimality and search efficiency where applicable.
>
---
#### [new 145] BrainLesion Suite: A Flexible and User-Friendly Framework for Modular Brain Lesion Image Analysis
- **分类: cs.CV; cs.AI; cs.LG; cs.SE**

- **简介: 该论文属于医学图像分析任务，旨在解决脑病变图像分析流程构建复杂、认知负担重的问题。作者开发了BrainLesion Suite，一个灵活、模块化的Python工具包，支持多模态图像预处理、病变分割与性能评估，便于临床与科研应用。**

- **链接: [http://arxiv.org/pdf/2507.09036v1](http://arxiv.org/pdf/2507.09036v1)**

> **作者:** Florian Kofler; Marcel Rosier; Mehdi Astaraki; Hendrik Möller; Ilhem Isra Mekki; Josef A. Buchner; Anton Schmick; Arianna Pfiffer; Eva Oswald; Lucas Zimmer; Ezequiel de la Rosa; Sarthak Pati; Julian Canisius; Arianna Piffer; Ujjwal Baid; Mahyar Valizadeh; Akis Linardos; Jan C. Peeken; Surprosanna Shit; Felix Steinbauer; Daniel Rueckert; Rolf Heckemann; Spyridon Bakas; Jan Kirschke; Constantin von See; Ivan Ezhov; Marie Piraud; Benedikt Wiestler; Bjoern Menze
>
> **备注:** 16p, 3f
>
> **摘要:** BrainLesion Suite is a versatile toolkit for building modular brain lesion image analysis pipelines in Python. Following Pythonic principles, BrainLesion Suite is designed to provide a 'brainless' development experience, minimizing cognitive effort and streamlining the creation of complex workflows for clinical and scientific practice. At its core is an adaptable preprocessing module that performs co-registration, atlas registration, and optional skull-stripping and defacing on arbitrary multi-modal input images. BrainLesion Suite leverages algorithms from the BraTS challenge to synthesize missing modalities, inpaint lesions, and generate pathology-specific tumor segmentations. BrainLesion Suite also enables quantifying segmentation model performance, with tools such as panoptica to compute lesion-wise metrics. Although BrainLesion Suite was originally developed for image analysis pipelines of brain lesions such as glioma, metastasis, and multiple sclerosis, it can be adapted for other biomedical image analysis applications. The individual BrainLesion Suite packages and tutorials are accessible on GitHub.
>
---
#### [new 146] QuarterMap: Efficient Post-Training Token Pruning for Visual State Space Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉模型优化任务，旨在解决视觉状态空间模型（如VMamba）因空间冗余导致的效率瓶颈问题。作者提出了QuarterMap方法，在不重新训练的情况下，通过剪枝冗余空间激活并使用最近邻上采样恢复维度，提升了模型吞吐量，同时保持了精度。**

- **链接: [http://arxiv.org/pdf/2507.09514v1](http://arxiv.org/pdf/2507.09514v1)**

> **作者:** Tien-Yu Chi; Hung-Yueh Chiang; Diana Marculescu; Kai-Chiang Wu
>
> **备注:** Accepted by Efficient Systems for Foundation Models Workshop at the International Conference on Machine Learning (ICML) 2025
>
> **摘要:** State space models (SSMs) reduce the quadratic complexity of transformers by leveraging linear recurrence. Recently, VMamba has emerged as a strong SSM-based vision backbone, yet remains bottlenecked by spatial redundancy in its four-directional scan. We propose QuarterMap, a post-training activation pruning method that removes redundant spatial activations before scanning and restores dimensions via nearest-neighbor upsampling. Our method improves throughput without retraining. On ImageNet-1K, QuarterMap achieves up to 11% speedup on VMamba with less than 0.9% accuracy drop, and yields similar gains on ADE20K segmentation. Beyond VMamba, we validate QuarterMap on MedMamba, a domain-specific model that shares the same four-directional scanning structure, where it consistently improves throughput while preserving accuracy across multiple medical imaging tasks. Compared to token merging methods like ToMe, QuarterMap is tailored for SSMs and avoids costly merge-unmerge operations. Our method offers a plug-and-play tool for deployment-time efficiency without compromising transferability.
>
---
#### [new 147] View Invariant Learning for Vision-Language Navigation in Continuous Environments
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于视觉-语言导航任务，旨在解决视角变化影响导航策略的问题。作者提出了VIL方法，通过对比学习和师生框架提升模型对视角变化的鲁棒性。实验表明其在R2R-CE和RxR-CE数据集上显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.08831v1](http://arxiv.org/pdf/2507.08831v1)**

> **作者:** Josh Qixuan Sun; Xiaoying Xing; Huaiyuan Weng; Chul Min Yeum; Mark Crowley
>
> **备注:** Under review
>
> **摘要:** Vision-Language Navigation in Continuous Environments (VLNCE), where an agent follows instructions and moves freely to reach a destination, is a key research problem in embodied AI. However, most navigation policies are sensitive to viewpoint changes, i.e., variations in camera height and viewing angle that alter the agent's observation. In this paper, we introduce a generalized scenario, V2-VLNCE (VLNCE with Varied Viewpoints), and propose VIL (View Invariant Learning), a view-invariant post-training strategy that enhances the robustness of existing navigation policies to changes in camera viewpoint. VIL employs a contrastive learning framework to learn sparse and view-invariant features. Additionally, we introduce a teacher-student framework for the Waypoint Predictor Module, a core component of most VLNCE baselines, where a view-dependent teacher model distills knowledge into a view-invariant student model. We employ an end-to-end training paradigm to jointly optimize these components, thus eliminating the cost for individual module training. Empirical results show that our method outperforms state-of-the-art approaches on V2-VLNCE by 8-15% measured on Success Rate for two standard benchmark datasets R2R-CE and RxR-CE. Furthermore, we evaluate VIL under the standard VLNCE setting and find that, despite being trained for varied viewpoints, it often still improves performance. On the more challenging RxR-CE dataset, our method also achieved state-of-the-art performance across all metrics when compared to other map-free methods. This suggests that adding VIL does not diminish the standard viewpoint performance and can serve as a plug-and-play post-training method.
>
---
#### [new 148] Improving Multimodal Learning via Imbalanced Learning
- **分类: cs.CV**

- **简介: 该论文属于多模态学习任务，旨在解决多模态学习中因模态间学习不平衡导致的性能下降问题。作者提出了一种不对称表示学习（ARL）策略，通过引入基于模态预测方差的辅助正则化项，动态调整各模态的学习权重，使模态依赖比例与其方差比成反比，并联合优化多模态损失和预测偏差，从而提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.10203v1](http://arxiv.org/pdf/2507.10203v1)**

> **作者:** Shicai Wei; Chunbo Luo; Yang Luo
>
> **备注:** Accepted to ICCV2025
>
> **摘要:** Multimodal learning often encounters the under-optimized problem and may perform worse than unimodal learning. Existing approaches attribute this issue to imbalanced learning across modalities and tend to address it through gradient balancing. However, this paper argues that balanced learning is not the optimal setting for multimodal learning. With bias-variance analysis, we prove that imbalanced dependency on each modality obeying the inverse ratio of their variances contributes to optimal performance. To this end, we propose the Asymmetric Representation Learning(ARL) strategy to assist multimodal learning via imbalanced optimization. ARL introduces auxiliary regularizers for each modality encoder to calculate their prediction variance. ARL then calculates coefficients via the unimodal variance to re-weight the optimization of each modality, forcing the modality dependence ratio to be inversely proportional to the modality variance ratio. Moreover, to minimize the generalization error, ARL further introduces the prediction bias of each modality and jointly optimizes them with multimodal loss. Notably, all auxiliary regularizers share parameters with the multimodal model and rely only on the modality representation. Thus the proposed ARL strategy introduces no extra parameters and is independent of the structures and fusion methods of the multimodal model. Finally, extensive experiments on various datasets validate the effectiveness and versatility of ARL. Code is available at \href{https://github.com/shicaiwei123/ICCV2025-ARL}{https://github.com/shicaiwei123/ICCV2025-ARL}
>
---
#### [new 149] Generate Aligned Anomaly: Region-Guided Few-Shot Anomaly Image-Mask Pair Synthesis for Industrial Inspection
- **分类: cs.CV**

- **简介: 该论文属于工业检测中的异常合成任务，旨在解决异常样本稀缺导致的定位与分类效果差的问题。论文提出了GAA框架，通过区域引导生成对齐的异常图像与掩码对，提升了合成质量与下游任务表现。**

- **链接: [http://arxiv.org/pdf/2507.09619v1](http://arxiv.org/pdf/2507.09619v1)**

> **作者:** Yilin Lu; Jianghang Lin; Linhuang Xie; Kai Zhao; Yansong Qu; Shengchuan Zhang; Liujuan Cao; Rongrong Ji
>
> **摘要:** Anomaly inspection plays a vital role in industrial manufacturing, but the scarcity of anomaly samples significantly limits the effectiveness of existing methods in tasks such as localization and classification. While several anomaly synthesis approaches have been introduced for data augmentation, they often struggle with low realism, inaccurate mask alignment, and poor generalization. To overcome these limitations, we propose Generate Aligned Anomaly (GAA), a region-guided, few-shot anomaly image-mask pair generation framework. GAA leverages the strong priors of a pretrained latent diffusion model to generate realistic, diverse, and semantically aligned anomalies using only a small number of samples. The framework first employs Localized Concept Decomposition to jointly model the semantic features and spatial information of anomalies, enabling flexible control over the type and location of anomalies. It then utilizes Adaptive Multi-Round Anomaly Clustering to perform fine-grained semantic clustering of anomaly concepts, thereby enhancing the consistency of anomaly representations. Subsequently, a region-guided mask generation strategy ensures precise alignment between anomalies and their corresponding masks, while a low-quality sample filtering module is introduced to further improve the overall quality of the generated samples. Extensive experiments on the MVTec AD and LOCO datasets demonstrate that GAA achieves superior performance in both anomaly synthesis quality and downstream tasks such as localization and classification.
>
---
#### [new 150] Lightweight Model for Poultry Disease Detection from Fecal Images Using Multi-Color Space Feature Optimization and Machine Learning
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像分类任务，旨在解决家禽粪便图像中的疾病检测问题。为实现低资源环境下的实时检测，研究者提取多颜色空间特征并优化特征集，结合机器学习方法构建轻量模型。最终模型在无GPU环境下实现了高准确率和快速推理。**

- **链接: [http://arxiv.org/pdf/2507.10056v1](http://arxiv.org/pdf/2507.10056v1)**

> **作者:** A. K. M. Shoriful Islam; Md. Rakib Hassan; Macbah Uddin; Md. Shahidur Rahman
>
> **摘要:** Poultry farming is a vital component of the global food supply chain, yet it remains highly vulnerable to infectious diseases such as coccidiosis, salmonellosis, and Newcastle disease. This study proposes a lightweight machine learning-based approach to detect these diseases by analyzing poultry fecal images. We utilize multi-color space feature extraction (RGB, HSV, LAB) and explore a wide range of color, texture, and shape-based descriptors, including color histograms, local binary patterns (LBP), wavelet transforms, and edge detectors. Through a systematic ablation study and dimensionality reduction using PCA and XGBoost feature selection, we identify a compact global feature set that balances accuracy and computational efficiency. An artificial neural network (ANN) classifier trained on these features achieved 95.85% accuracy while requiring no GPU and only 638 seconds of execution time in Google Colab. Compared to deep learning models such as Xception and MobileNetV3, our proposed model offers comparable accuracy with drastically lower resource usage. This work demonstrates a cost-effective, interpretable, and scalable alternative to deep learning for real-time poultry disease detection in low-resource agricultural settings.
>
---
#### [new 151] Prompt Engineering in Segment Anything Model: Methodologies, Applications, and Emerging Challenges
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像分割任务，聚焦Segment Anything Model中的提示工程。论文系统综述了提示工程技术的方法、应用与挑战，分析了其从简单几何输入到多模态方法的演变，并探讨了在医学成像和遥感等领域的适应性。重点在于揭示提示优化的挑战并提出未来研究方向。**

- **链接: [http://arxiv.org/pdf/2507.09562v1](http://arxiv.org/pdf/2507.09562v1)**

> **作者:** Yidong Jiang
>
> **摘要:** The Segment Anything Model (SAM) has revolutionized image segmentation through its innovative prompt-based approach, yet the critical role of prompt engineering in its success remains underexplored. This paper presents the first comprehensive survey focusing specifically on prompt engineering techniques for SAM and its variants. We systematically organize and analyze the rapidly growing body of work in this emerging field, covering fundamental methodologies, practical applications, and key challenges. Our review reveals how prompt engineering has evolved from simple geometric inputs to sophisticated multimodal approaches, enabling SAM's adaptation across diverse domains including medical imaging and remote sensing. We identify unique challenges in prompt optimization and discuss promising research directions. This survey fills an important gap in the literature by providing a structured framework for understanding and advancing prompt engineering in foundation models for segmentation.
>
---
#### [new 152] Revisiting Pool-based Prompt Learning for Few-shot Class-incremental Learning
- **分类: cs.CV**

- **简介: 该论文属于小样本类增量学习任务，旨在解决数据稀缺和增量学习中的性能下降问题。作者分析了现有提示方法在该任务中的局限性，并提出LGSP-Prompt方法，通过空间维度的局部-全局提示学习，有效保留旧知识并提升新类别的学习效果。**

- **链接: [http://arxiv.org/pdf/2507.09183v1](http://arxiv.org/pdf/2507.09183v1)**

> **作者:** Yongwei Jiang; Yixiong Zou; Yuhua Li; Ruixuan Li
>
> **备注:** Accepted to ICCV 2025, 11 pages
>
> **摘要:** Few-Shot Class-Incremental Learning (FSCIL) faces dual challenges of data scarcity and incremental learning in real-world scenarios. While pool-based prompting methods have demonstrated success in traditional incremental learning, their effectiveness in FSCIL settings remains unexplored. This paper presents the first study of current prompt pool methods in FSCIL tasks, revealing an unanticipated performance degradation in incremental sessions. Through comprehensive analysis, we identify that this phenomenon stems from token-dimension saturation: with limited data, excessive prompts compete for task-relevant information, leading to model overfitting. Based on this finding, we propose LGSP-Prompt (Local-Global Spatial Prompting), which innovatively shifts pool-based prompt learning from the token dimension to the spatial dimension. LGSP-Prompt generates spatial prompts by synergistically combining local spatial features and global frequency-domain representations to highlight key patterns in input images. We construct two spatial prompt pools enabling dynamic prompt selection to maintain acquired knowledge while effectively learning novel sessions. Extensive experiments demonstrate that our approach achieves state-of-the-art performance across multiple FSCIL benchmarks, showing significant advantages in both base knowledge preservation and incremental learning. Our implementation is available at https://github.com/Jywsuperman/LGSP.
>
---
#### [new 153] GLIMPSE: Do Large Vision-Language Models Truly Think With Videos or Just Glimpse at Them?
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决当前模型是否能真正“思考”视频内容而非仅分析关键帧的问题。作者构建了GLIMPSE基准，包含需全程观看与综合推理的4,342个问题，评估大型视觉语言模型的表现，发现现有模型仍停留在表面推理层面。**

- **链接: [http://arxiv.org/pdf/2507.09491v1](http://arxiv.org/pdf/2507.09491v1)**

> **作者:** Yiyang Zhou; Linjie Li; Shi Qiu; Zhengyuan Yang; Yuyang Zhao; Siwei Han; Yangfan He; Kangqi Li; Haonian Ji; Zihao Zhao; Haibo Tong; Lijuan Wang; Huaxiu Yao
>
> **备注:** 15 pages, 10 figures
>
> **摘要:** Existing video benchmarks often resemble image-based benchmarks, with question types like "What actions does the person perform throughout the video?" or "What color is the woman's dress in the video?" For these, models can often answer by scanning just a few key frames, without deep temporal reasoning. This limits our ability to assess whether large vision-language models (LVLMs) can truly think with videos rather than perform superficial frame-level analysis. To address this, we introduce GLIMPSE, a benchmark specifically designed to evaluate whether LVLMs can genuinely think with videos. Unlike prior benchmarks, GLIMPSE emphasizes comprehensive video understanding beyond static image cues. It consists of 3,269 videos and over 4,342 highly visual-centric questions across 11 categories, including Trajectory Analysis, Temporal Reasoning, and Forensics Detection. All questions are carefully crafted by human annotators and require watching the entire video and reasoning over full video context-this is what we mean by thinking with video. These questions cannot be answered by scanning selected frames or relying on text alone. In human evaluations, GLIMPSE achieves 94.82% accuracy, but current LVLMs face significant challenges. Even the best-performing model, GPT-o3, reaches only 66.43%, highlighting that LVLMs still struggle to move beyond surface-level reasoning to truly think with videos.
>
---
#### [new 154] Fine-Grained Zero-Shot Object Detection
- **分类: cs.CV**

- **简介: 该论文属于细粒度零样本目标检测（FG-ZSD）任务，旨在解决视觉相似类别的目标检测问题。作者提出了MSHC方法，并构建了首个FG-ZSD基准数据集FGZSD-Birds，包含1432种鸟类图像。实验表明其方法优于现有模型。**

- **链接: [http://arxiv.org/pdf/2507.10358v1](http://arxiv.org/pdf/2507.10358v1)**

> **作者:** Hongxu Ma; Chenbo Zhang; Lu Zhang; Jiaogen Zhou; Jihong Guan; Shuigeng Zhou
>
> **备注:** Accepted by ACM MM'25
>
> **摘要:** Zero-shot object detection (ZSD) aims to leverage semantic descriptions to localize and recognize objects of both seen and unseen classes. Existing ZSD works are mainly coarse-grained object detection, where the classes are visually quite different, thus are relatively easy to distinguish. However, in real life we often have to face fine-grained object detection scenarios, where the classes are too similar to be easily distinguished. For example, detecting different kinds of birds, fishes, and flowers. In this paper, we propose and solve a new problem called Fine-Grained Zero-Shot Object Detection (FG-ZSD for short), which aims to detect objects of different classes with minute differences in details under the ZSD paradigm. We develop an effective method called MSHC for the FG-ZSD task, which is based on an improved two-stage detector and employs a multi-level semantics-aware embedding alignment loss, ensuring tight coupling between the visual and semantic spaces. Considering that existing ZSD datasets are not suitable for the new FG-ZSD task, we build the first FG-ZSD benchmark dataset FGZSD-Birds, which contains 148,820 images falling into 36 orders, 140 families, 579 genera and 1432 species. Extensive experiments on FGZSD-Birds show that our method outperforms existing ZSD models.
>
---
#### [new 155] FaceLLM: A Multimodal Large Language Model for Face Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态语言模型任务，旨在解决现有模型在面部图像理解上的局限性。作者构建了专门的FaceLLM模型和FairFaceGPT数据集，利用ChatGPT生成高质量图文对，提升模型在表情、姿态等面部属性上的推理能力，推动领域专用多模态AI发展。**

- **链接: [http://arxiv.org/pdf/2507.10300v1](http://arxiv.org/pdf/2507.10300v1)**

> **作者:** Hatef Otroshi Shahreza; Sébastien Marcel
>
> **备注:** Accepted in ICCV 2025 workshops
>
> **摘要:** Multimodal large language models (MLLMs) have shown remarkable performance in vision-language tasks. However, existing MLLMs are primarily trained on generic datasets, limiting their ability to reason on domain-specific visual cues such as those in facial images. In particular, tasks that require detailed understanding of facial structure, expression, emotion, and demographic features remain underexplored by MLLMs due to the lack of large-scale annotated face image-text datasets. In this work, we introduce FaceLLM, a multimodal large language model trained specifically for facial image understanding. To construct the training data, we propose a novel weakly supervised pipeline that uses ChatGPT with attribute-aware prompts to generate high-quality question-answer pairs based on images from the FairFace dataset. The resulting corpus, called FairFaceGPT, covers a diverse set of attributes including expression, pose, skin texture, and forensic information. Our experiments demonstrate that FaceLLM improves the performance of MLLMs on various face-centric tasks and achieves state-of-the-art performance. This work highlights the potential of synthetic supervision via language models for building domain-specialized MLLMs, and sets a precedent for trustworthy, human-centric multimodal AI systems. FairFaceGPT dataset and pretrained FaceLLM models are publicly available in the project page.
>
---
#### [new 156] Kaleidoscopic Background Attack: Disrupting Pose Estimation with Multi-Fold Radial Symmetry Textures
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决背景纹理对相机姿态估计模型的影响问题。作者提出“Kaleidoscopic Background Attack”方法，通过生成多折叠径向对称的圆盘纹理，干扰模型在稀疏输入下的位姿估计效果，并引入一致性损失优化攻击效果，有效降低了多种模型的估计精度。**

- **链接: [http://arxiv.org/pdf/2507.10265v1](http://arxiv.org/pdf/2507.10265v1)**

> **作者:** Xinlong Ding; Hongwei Yu; Jiawei Li; Feifan Li; Yu Shang; Bochao Zou; Huimin Ma; Jiansheng Chen
>
> **备注:** Accepted at ICCV 2025. Project page is available at https://wakuwu.github.io/KBA
>
> **摘要:** Camera pose estimation is a fundamental computer vision task that is essential for applications like visual localization and multi-view stereo reconstruction. In the object-centric scenarios with sparse inputs, the accuracy of pose estimation can be significantly influenced by background textures that occupy major portions of the images across different viewpoints. In light of this, we introduce the Kaleidoscopic Background Attack (KBA), which uses identical segments to form discs with multi-fold radial symmetry. These discs maintain high similarity across different viewpoints, enabling effective attacks on pose estimation models even with natural texture segments. Additionally, a projected orientation consistency loss is proposed to optimize the kaleidoscopic segments, leading to significant enhancement in the attack effectiveness. Experimental results show that optimized adversarial kaleidoscopic backgrounds can effectively attack various camera pose estimation models.
>
---
#### [new 157] Supercharging Floorplan Localization with Semantic Rays
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于室内定位任务，旨在解决现有方法忽略建筑语义信息的问题。作者提出一种结合深度与语义射线的概率框架，构建结构-语义概率体，通过粗到精的射线采样策略提升定位精度，并可融合房间标签等元数据，显著优化召回率与效率。**

- **链接: [http://arxiv.org/pdf/2507.09291v1](http://arxiv.org/pdf/2507.09291v1)**

> **作者:** Yuval Grader; Hadar Averbuch-Elor
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Floorplans provide a compact representation of the building's structure, revealing not only layout information but also detailed semantics such as the locations of windows and doors. However, contemporary floorplan localization techniques mostly focus on matching depth-based structural cues, ignoring the rich semantics communicated within floorplans. In this work, we introduce a semantic-aware localization framework that jointly estimates depth and semantic rays, consolidating over both for predicting a structural-semantic probability volume. Our probability volume is constructed in a coarse-to-fine manner: We first sample a small set of rays to obtain an initial low-resolution probability volume. We then refine these probabilities by performing a denser sampling only in high-probability regions and process the refined values for predicting a 2D location and orientation angle. We conduct an evaluation on two standard floorplan localization benchmarks. Our experiments demonstrate that our approach substantially outperforms state-of-the-art methods, achieving significant improvements in recall metrics compared to prior works. Moreover, we show that our framework can easily incorporate additional metadata such as room labels, enabling additional gains in both accuracy and efficiency.
>
---
#### [new 158] From Wardrobe to Canvas: Wardrobe Polyptych LoRA for Part-level Controllable Human Image Generation
- **分类: cs.CV**

- **简介: 该论文属于个性化人物图像生成任务，旨在解决生成高质量、身份保留且可控的人物图像问题。现有方法依赖大量数据或计算资源，不适用于实时应用。论文提出Wardrobe Polyptych LoRA，通过训练低秩适配层，实现部分级别的控制与高保真合成，无需额外参数和复杂训练，提升了生成效果的一致性与真实性。**

- **链接: [http://arxiv.org/pdf/2507.10217v1](http://arxiv.org/pdf/2507.10217v1)**

> **作者:** Jeongho Kim; Sunghyun Park; Hyoungwoo Park; Sungrack Yun; Jaegul Choo; Seokeon Cho
>
> **备注:** 10 pages, 8 figures
>
> **摘要:** Recent diffusion models achieve personalization by learning specific subjects, allowing learned attributes to be integrated into generated images. However, personalized human image generation remains challenging due to the need for precise and consistent attribute preservation (e.g., identity, clothing details). Existing subject-driven image generation methods often require either (1) inference-time fine-tuning with few images for each new subject or (2) large-scale dataset training for generalization. Both approaches are computationally expensive and impractical for real-time applications. To address these limitations, we present Wardrobe Polyptych LoRA, a novel part-level controllable model for personalized human image generation. By training only LoRA layers, our method removes the computational burden at inference while ensuring high-fidelity synthesis of unseen subjects. Our key idea is to condition the generation on the subject's wardrobe and leverage spatial references to reduce information loss, thereby improving fidelity and consistency. Additionally, we introduce a selective subject region loss, which encourages the model to disregard some of reference images during training. Our loss ensures that generated images better align with text prompts while maintaining subject integrity. Notably, our Wardrobe Polyptych LoRA requires no additional parameters at the inference stage and performs generation using a single model trained on a few training samples. We construct a new dataset and benchmark tailored for personalized human image generation. Extensive experiments show that our approach significantly outperforms existing techniques in fidelity and consistency, enabling realistic and identity-preserving full-body synthesis.
>
---
#### [new 159] Quantize-then-Rectify: Efficient VQ-VAE Training
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像压缩与生成任务，旨在解决高效训练高压缩率VQ-VAE的问题。现有方法计算成本高，训练耗时长。论文提出ReVQ框架，通过量化预训练VAE并引入多组量化与后修正策略，在保持良好重建质量的同时大幅降低训练开销。**

- **链接: [http://arxiv.org/pdf/2507.10547v1](http://arxiv.org/pdf/2507.10547v1)**

> **作者:** Borui Zhang; Qihang Rao; Wenzhao Zheng; Jie Zhou; Jiwen Lu
>
> **摘要:** Visual tokenizers are pivotal in multimodal large models, acting as bridges between continuous inputs and discrete tokens. Nevertheless, training high-compression-rate VQ-VAEs remains computationally demanding, often necessitating thousands of GPU hours. This work demonstrates that a pre-trained VAE can be efficiently transformed into a VQ-VAE by controlling quantization noise within the VAE's tolerance threshold. We present \textbf{Quantize-then-Rectify (ReVQ)}, a framework leveraging pre-trained VAEs to enable rapid VQ-VAE training with minimal computational overhead. By integrating \textbf{channel multi-group quantization} to enlarge codebook capacity and a \textbf{post rectifier} to mitigate quantization errors, ReVQ compresses ImageNet images into at most 512 tokens while sustaining competitive reconstruction quality (rFID = 1.06). Significantly, ReVQ reduces training costs by over two orders of magnitude relative to state-of-the-art approaches: ReVQ finishes full training on a single NVIDIA 4090 in approximately 22 hours, whereas comparable methods require 4.5 days on 32 A100 GPUs. Experimental results show that ReVQ achieves superior efficiency-reconstruction trade-offs.
>
---
#### [new 160] SlumpGuard: An AI-Powered Real-Time System for Automated Concrete Slump Prediction via Video Analysis
- **分类: cs.CV**

- **简介: 论文提出SlumpGuard，一个基于AI的视频分析系统，用于实时预测混凝土坍落度，解决传统人工测试效率低、一致性差的问题。系统实现了全批次自动检测，提升了混凝土质量控制的准确性与效率。**

- **链接: [http://arxiv.org/pdf/2507.10171v1](http://arxiv.org/pdf/2507.10171v1)**

> **作者:** Youngmin Kim; Giyeong Oh; Kwangsoo Youm; Youngjae Yu
>
> **摘要:** Concrete workability is essential for construction quality, with the slump test being the most common on-site method for its assessment. However, traditional slump testing is manual, time-consuming, and prone to inconsistency, limiting its applicability for real-time monitoring. To address these challenges, we propose SlumpGuard, an AI-powered, video-based system that automatically analyzes concrete flow from the truck chute to assess workability in real time. Our system enables full-batch inspection without manual intervention, improving both the accuracy and efficiency of quality control. We present the system design, a the construction of a dedicated dataset, and empirical results from real-world deployment, demonstrating the effectiveness of SlumpGuard as a practical solution for modern concrete quality assurance.
>
---
#### [new 161] CADmium: Fine-Tuning Code Language Models for Text-Driven Sequential CAD Design
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 该论文属于文本驱动的CAD设计任务，旨在解决CAD建模效率低的问题。作者构建了一个大规模带标注的CAD数据集，并提出CADmium方法，通过微调代码语言模型，实现从自然语言描述生成CAD序列。同时引入几何和拓扑指标评估生成质量，验证了该方法在加速CAD设计上的有效性。**

- **链接: [http://arxiv.org/pdf/2507.09792v1](http://arxiv.org/pdf/2507.09792v1)**

> **作者:** Prashant Govindarajan; Davide Baldelli; Jay Pathak; Quentin Fournier; Sarath Chandar
>
> **摘要:** Computer-aided design (CAD) is the digital construction of 2D and 3D objects, and is central to a wide range of engineering and manufacturing applications like automobile and aviation. Despite its importance, CAD modeling remains largely a time-intensive, manual task. Recent works have attempted to automate this process with small transformer-based models and handcrafted CAD sequence representations. However, there has been little effort to leverage the potential of large language models (LLMs) for sequential CAD design. In this work, we introduce a new large-scale dataset of more than 170k CAD models annotated with high-quality, human-like descriptions generated with our pipeline based on GPT-4.1. Using this dataset, we fine-tune powerful code-LLMs to generate CAD sequences represented in a JSON-based format from natural language descriptions, demonstrating the viability and effectiveness of this approach for text-conditioned CAD generation. Because simple metrics often fail to reflect the quality of generated objects, we introduce geometric and topological metrics based on sphericity, mean curvature, and Euler characteristic to provide richer structural insights. Our experiments and ablation studies on both synthetic and human-annotated data demonstrate that CADmium is able to automate CAD design, drastically speeding up the design of new objects. The dataset, code, and fine-tuned models are available online.
>
---
#### [new 162] Multimodal HD Mapping for Intersections by Intelligent Roadside Units
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于高精地图构建任务，旨在解决传统车载方法在复杂路口因遮挡和视角受限导致的地图构建难题。论文提出了基于路侧智能单元的相机与激光雷达融合框架，并构建了包含多模态数据的RS-seq数据集，以提升语义分割精度，推动车路协同自动驾驶研究。**

- **链接: [http://arxiv.org/pdf/2507.08903v1](http://arxiv.org/pdf/2507.08903v1)**

> **作者:** Zhongzhang Chen; Miao Fan; Shengtong Xu; Mengmeng Yang; Kun Jiang; Xiangzeng Liu; Haoyi Xiong
>
> **备注:** Accepted by ITSC'25
>
> **摘要:** High-definition (HD) semantic mapping of complex intersections poses significant challenges for traditional vehicle-based approaches due to occlusions and limited perspectives. This paper introduces a novel camera-LiDAR fusion framework that leverages elevated intelligent roadside units (IRUs). Additionally, we present RS-seq, a comprehensive dataset developed through the systematic enhancement and annotation of the V2X-Seq dataset. RS-seq includes precisely labelled camera imagery and LiDAR point clouds collected from roadside installations, along with vectorized maps for seven intersections annotated with detailed features such as lane dividers, pedestrian crossings, and stop lines. This dataset facilitates the systematic investigation of cross-modal complementarity for HD map generation using IRU data. The proposed fusion framework employs a two-stage process that integrates modality-specific feature extraction and cross-modal semantic integration, capitalizing on camera high-resolution texture and precise geometric data from LiDAR. Quantitative evaluations using the RS-seq dataset demonstrate that our multimodal approach consistently surpasses unimodal methods. Specifically, compared to unimodal baselines evaluated on the RS-seq dataset, the multimodal approach improves the mean Intersection-over-Union (mIoU) for semantic segmentation by 4\% over the image-only results and 18\% over the point cloud-only results. This study establishes a baseline methodology for IRU-based HD semantic mapping and provides a valuable dataset for future research in infrastructure-assisted autonomous driving systems.
>
---
#### [new 163] ESG-Net: Event-Aware Semantic Guided Network for Dense Audio-Visual Event Localization
- **分类: cs.MM; cs.CV**

- **简介: 该论文属于密集音视频事件定位任务，旨在准确识别未剪辑视频中的事件类别及时间边界。现有方法缺乏跨模态语义融合与事件间关系建模。论文提出ESG-Net，包含ESI和MoDE模块，实现多阶段语义引导与多事件依赖建模，提升定位性能，减少参数与计算量。**

- **链接: [http://arxiv.org/pdf/2507.09945v1](http://arxiv.org/pdf/2507.09945v1)**

> **作者:** Huilai Li; Yonghao Dang; Ying Xing; Yiming Wang; Jianqin Yin
>
> **摘要:** Dense audio-visual event localization (DAVE) aims to identify event categories and locate the temporal boundaries in untrimmed videos. Most studies only employ event-related semantic constraints on the final outputs, lacking cross-modal semantic bridging in intermediate layers. This causes modality semantic gap for further fusion, making it difficult to distinguish between event-related content and irrelevant background content. Moreover, they rarely consider the correlations between events, which limits the model to infer concurrent events among complex scenarios. In this paper, we incorporate multi-stage semantic guidance and multi-event relationship modeling, which respectively enable hierarchical semantic understanding of audio-visual events and adaptive extraction of event dependencies, thereby better focusing on event-related information. Specifically, our eventaware semantic guided network (ESG-Net) includes a early semantics interaction (ESI) module and a mixture of dependency experts (MoDE) module. ESI applys multi-stage semantic guidance to explicitly constrain the model in learning semantic information through multi-modal early fusion and several classification loss functions, ensuring hierarchical understanding of event-related content. MoDE promotes the extraction of multi-event dependencies through multiple serial mixture of experts with adaptive weight allocation. Extensive experiments demonstrate that our method significantly surpasses the state-of-the-art methods, while greatly reducing parameters and computational load. Our code will be released on https://github.com/uchiha99999/ESG-Net.
>
---
#### [new 164] TRACER: Efficient Object Re-Identification in Networked Cameras through Adaptive Query Processing
- **分类: cs.DB; cs.CV**

- **简介: 论文提出TRACER系统，针对多摄像头网络中的目标重识别任务，解决现有方法在大规模场景下精度低、无法自适应查询的问题。工作包括：设计基于循环网络的历史相关性建模方法、结合探索-利用策略的概率自适应搜索模型，并构建合成数据集。实验表明其性能优于现有系统3.9倍。**

- **链接: [http://arxiv.org/pdf/2507.09448v1](http://arxiv.org/pdf/2507.09448v1)**

> **作者:** Pramod Chunduri; Yao Lu; Joy Arulraj
>
> **摘要:** Efficiently re-identifying and tracking objects across a network of cameras is crucial for applications like traffic surveillance. Spatula is the state-of-the-art video database management system (VDBMS) for processing Re-ID queries. However, it suffers from two limitations. Its spatio-temporal filtering scheme has limited accuracy on large camera networks due to localized camera history. It is not suitable for critical video analytics applications that require high recall due to a lack of support for adaptive query processing. In this paper, we present Tracer, a novel VDBMS for efficiently processing Re-ID queries using an adaptive query processing framework. Tracer selects the optimal camera to process at each time step by training a recurrent network to model long-term historical correlations. To accelerate queries under a high recall constraint, Tracer incorporates a probabilistic adaptive search model that processes camera feeds in incremental search windows and dynamically updates the sampling probabilities using an exploration-exploitation strategy. To address the paucity of benchmarks for the Re-ID task due to privacy concerns, we present a novel synthetic benchmark for generating multi-camera Re-ID datasets based on real-world traffic distribution. Our evaluation shows that Tracer outperforms the state-of-the-art cross-camera analytics system by 3.9x on average across diverse datasets.
>
---
#### [new 165] A Brain Tumor Segmentation Method Based on CLIP and 3D U-Net with Cross-Modal Semantic Guidance and Multi-Level Feature Fusion
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分割任务，旨在提升脑肿瘤磁共振图像的自动分割精度。针对现有方法忽略医学报告语义信息的问题，提出结合CLIP和3D U-Net的多模态融合架构，引入跨模态语义引导与多级特征融合机制，显著提升了分割效果，尤其在增强肿瘤区域表现优异。**

- **链接: [http://arxiv.org/pdf/2507.09966v1](http://arxiv.org/pdf/2507.09966v1)**

> **作者:** Mingda Zhang
>
> **备注:** 13 pages,6 figures
>
> **摘要:** Precise segmentation of brain tumors from magnetic resonance imaging (MRI) is essential for neuro-oncology diagnosis and treatment planning. Despite advances in deep learning methods, automatic segmentation remains challenging due to tumor morphological heterogeneity and complex three-dimensional spatial relationships. Current techniques primarily rely on visual features extracted from MRI sequences while underutilizing semantic knowledge embedded in medical reports. This research presents a multi-level fusion architecture that integrates pixel-level, feature-level, and semantic-level information, facilitating comprehensive processing from low-level data to high-level concepts. The semantic-level fusion pathway combines the semantic understanding capabilities of Contrastive Language-Image Pre-training (CLIP) models with the spatial feature extraction advantages of 3D U-Net through three mechanisms: 3D-2D semantic bridging, cross-modal semantic guidance, and semantic-based attention mechanisms. Experimental validation on the BraTS 2020 dataset demonstrates that the proposed model achieves an overall Dice coefficient of 0.8567, representing a 4.8% improvement compared to traditional 3D U-Net, with a 7.3% Dice coefficient increase in the clinically important enhancing tumor (ET) region.
>
---
#### [new 166] Zero-Shot Neural Architecture Search with Weighted Response Correlation
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于神经网络架构搜索（NAS）任务，旨在解决现有方法计算成本高、效率低的问题。作者提出了一种无需训练的评估代理WRCor，通过样本响应相关性矩阵计算架构得分，提升搜索效率与性能。实验表明其方法在多个搜索空间中优于现有技术。**

- **链接: [http://arxiv.org/pdf/2507.08841v1](http://arxiv.org/pdf/2507.08841v1)**

> **作者:** Kun Jing; Luoyu Chen; Jungang Xu; Jianwei Tai; Yiyu Wang; Shuaimin Li
>
> **摘要:** Neural architecture search (NAS) is a promising approach for automatically designing neural network architectures. However, the architecture estimation of NAS is computationally expensive and time-consuming because of training multiple architectures from scratch. Although existing zero-shot NAS methods use training-free proxies to accelerate the architecture estimation, their effectiveness, stability, and generality are still lacking. We present a novel training-free estimation proxy called weighted response correlation (WRCor). WRCor utilizes correlation coefficient matrices of responses across different input samples to calculate the proxy scores of estimated architectures, which can measure their expressivity and generalizability. Experimental results on proxy evaluation demonstrate that WRCor and its voting proxies are more efficient estimation strategies than existing proxies. We also apply them with different search strategies in architecture search. Experimental results on architecture search show that our zero-shot NAS algorithm outperforms most existing NAS algorithms in different search spaces. Our NAS algorithm can discover an architecture with a 22.1% test error on the ImageNet-1k dataset within 4 GPU hours. All codes are publicly available at https://github.com/kunjing96/ZSNAS-WRCor.git.
>
---
#### [new 167] Confounder-Free Continual Learning via Recursive Feature Normalization
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于持续学习任务，旨在解决模型在连续学习过程中因混杂因素变化导致的预测偏差和灾难性遗忘问题。作者提出了一种递归特征归一化方法（R-MDN），可集成到深度模型中，通过递归最小二乘算法动态调整特征表示，消除混杂因素的影响，提升模型在不同群体和学习阶段的公平性和稳定性。**

- **链接: [http://arxiv.org/pdf/2507.09031v1](http://arxiv.org/pdf/2507.09031v1)**

> **作者:** Yash Shah; Camila Gonzalez; Mohammad H. Abbasi; Qingyu Zhao; Kilian M. Pohl; Ehsan Adeli
>
> **摘要:** Confounders are extraneous variables that affect both the input and the target, resulting in spurious correlations and biased predictions. There are recent advances in dealing with or removing confounders in traditional models, such as metadata normalization (MDN), where the distribution of the learned features is adjusted based on the study confounders. However, in the context of continual learning, where a model learns continuously from new data over time without forgetting, learning feature representations that are invariant to confounders remains a significant challenge. To remove their influence from intermediate feature representations, we introduce the Recursive MDN (R-MDN) layer, which can be integrated into any deep learning architecture, including vision transformers, and at any model stage. R-MDN performs statistical regression via the recursive least squares algorithm to maintain and continually update an internal model state with respect to changing distributions of data and confounding variables. Our experiments demonstrate that R-MDN promotes equitable predictions across population groups, both within static learning and across different stages of continual learning, by reducing catastrophic forgetting caused by confounder effects changing over time.
>
---
#### [new 168] Scene-Aware Conversational ADAS with Generative AI for Real-Time Driver Assistance
- **分类: cs.RO; cs.AI; cs.CV; cs.HC**

- **简介: 论文提出SC-ADAS框架，将生成式AI与ADAS结合，实现基于场景感知的自然语言交互。解决当前ADAS缺乏语境理解和对话能力的问题，支持多轮对话与实时驾驶辅助决策，提升系统灵活性与适应性。**

- **链接: [http://arxiv.org/pdf/2507.10500v1](http://arxiv.org/pdf/2507.10500v1)**

> **作者:** Kyungtae Han; Yitao Chen; Rohit Gupta; Onur Altintas
>
> **摘要:** While autonomous driving technologies continue to advance, current Advanced Driver Assistance Systems (ADAS) remain limited in their ability to interpret scene context or engage with drivers through natural language. These systems typically rely on predefined logic and lack support for dialogue-based interaction, making them inflexible in dynamic environments or when adapting to driver intent. This paper presents Scene-Aware Conversational ADAS (SC-ADAS), a modular framework that integrates Generative AI components including large language models, vision-to-text interpretation, and structured function calling to enable real-time, interpretable, and adaptive driver assistance. SC-ADAS supports multi-turn dialogue grounded in visual and sensor context, allowing natural language recommendations and driver-confirmed ADAS control. Implemented in the CARLA simulator with cloud-based Generative AI, the system executes confirmed user intents as structured ADAS commands without requiring model fine-tuning. We evaluate SC-ADAS across scene-aware, conversational, and revisited multi-turn interactions, highlighting trade-offs such as increased latency from vision-based context retrieval and token growth from accumulated dialogue history. These results demonstrate the feasibility of combining conversational reasoning, scene perception, and modular ADAS control to support the next generation of intelligent driver assistance.
>
---
#### [new 169] RectifiedHR: High-Resolution Diffusion via Energy Profiling and Adaptive Guidance Scheduling
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决高分辨率扩散模型中的能量不稳定和引导伪影问题。通过分析采样过程中的潜在能量分布，提出了一种自适应无分类器引导调度策略，优化能量轨迹稳定性与视觉质量，提升了生成图像的清晰度与一致性。**

- **链接: [http://arxiv.org/pdf/2507.09441v1](http://arxiv.org/pdf/2507.09441v1)**

> **作者:** Ankit Sanjyal
>
> **备注:** 8 Pages, 10 Figures, Pre-Print Version, Code Available at: https://github.com/ANKITSANJYAL/RectifiedHR
>
> **摘要:** High-resolution image synthesis with diffusion models often suffers from energy instabilities and guidance artifacts that degrade visual quality. We analyze the latent energy landscape during sampling and propose adaptive classifier-free guidance (CFG) schedules that maintain stable energy trajectories. Our approach introduces energy-aware scheduling strategies that modulate guidance strength over time, achieving superior stability scores (0.9998) and consistency metrics (0.9873) compared to fixed-guidance approaches. We demonstrate that DPM++ 2M with linear-decreasing CFG scheduling yields optimal performance, providing sharper, more faithful images while reducing artifacts. Our energy profiling framework serves as a powerful diagnostic tool for understanding and improving diffusion model behavior.
>
---
#### [new 170] IM-LUT: Interpolation Mixing Look-Up Tables for Image Super-Resolution
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像超分辨率任务，旨在解决任意尺度图像超分辨率的效率与质量平衡问题。作者提出IM-LUT框架，通过学习混合插值函数实现高效重建，并将网络转化为查找表（LUT），减少计算开销，提升CPU推理速度，同时保持高质量图像恢复。**

- **链接: [http://arxiv.org/pdf/2507.09923v1](http://arxiv.org/pdf/2507.09923v1)**

> **作者:** Sejin Park; Sangmin Lee; Kyong Hwan Jin; Seung-Won Jung
>
> **备注:** ICCV 2025
>
> **摘要:** Super-resolution (SR) has been a pivotal task in image processing, aimed at enhancing image resolution across various applications. Recently, look-up table (LUT)-based approaches have attracted interest due to their efficiency and performance. However, these methods are typically designed for fixed scale factors, making them unsuitable for arbitrary-scale image SR (ASISR). Existing ASISR techniques often employ implicit neural representations, which come with considerable computational cost and memory demands. To address these limitations, we propose Interpolation Mixing LUT (IM-LUT), a novel framework that operates ASISR by learning to blend multiple interpolation functions to maximize their representational capacity. Specifically, we introduce IM-Net, a network trained to predict mixing weights for interpolation functions based on local image patterns and the target scale factor. To enhance efficiency of interpolation-based methods, IM-Net is transformed into IM-LUT, where LUTs are employed to replace computationally expensive operations, enabling lightweight and fast inference on CPUs while preserving reconstruction quality. Experimental results on several benchmark datasets demonstrate that IM-LUT consistently achieves a superior balance between image quality and efficiency compared to existing methods, highlighting its potential as a promising solution for resource-constrained applications.
>
---
#### [new 171] Universal Physics Simulation: A Foundational Diffusion Approach
- **分类: cs.LG; cs.AI; cs.CV; 68T07, 65M06, 78M34; I.2.6; I.4.8; J.2**

- **简介: 该论文属于物理仿真任务，旨在解决传统方法依赖先验方程建模、泛化能力差的问题。作者提出一种基于扩散模型的神经网络框架，通过边界条件数据直接生成稳态物理解，无需显式编码物理方程，并实现了跨物理域的通用仿真。**

- **链接: [http://arxiv.org/pdf/2507.09733v1](http://arxiv.org/pdf/2507.09733v1)**

> **作者:** Bradley Camburn
>
> **备注:** 10 pages, 3 figures. Foundational AI model for universal physics simulation using sketch-guided diffusion transformers. Achieves SSIM > 0.8 on electromagnetic field generation without requiring a priori physics encoding
>
> **摘要:** We present the first foundational AI model for universal physics simulation that learns physical laws directly from boundary-condition data without requiring a priori equation encoding. Traditional physics-informed neural networks (PINNs) and finite-difference methods necessitate explicit mathematical formulation of governing equations, fundamentally limiting their generalizability and discovery potential. Our sketch-guided diffusion transformer approach reimagines computational physics by treating simulation as a conditional generation problem, where spatial boundary conditions guide the synthesis of physically accurate steady-state solutions. By leveraging enhanced diffusion transformer architectures with novel spatial relationship encoding, our model achieves direct boundary-to-equilibrium mapping and is generalizable to diverse physics domains. Unlike sequential time-stepping methods that accumulate errors over iterations, our approach bypasses temporal integration entirely, directly generating steady-state solutions with SSIM > 0.8 while maintaining sub-pixel boundary accuracy. Our data-informed approach enables physics discovery through learned representations analyzable via Layer-wise Relevance Propagation (LRP), revealing emergent physical relationships without predetermined mathematical constraints. This work represents a paradigm shift from AI-accelerated physics to AI-discovered physics, establishing the first truly universal physics simulation framework.
>
---
#### [new 172] Multi-omic Prognosis of Alzheimer's Disease with Asymmetric Cross-Modal Cross-Attention Network
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于阿尔茨海默病（AD）的辅助诊断任务，旨在解决多模态数据融合效果差的问题。作者提出了一种不对称跨模态交叉注意力机制，融合PET、MRI、基因和临床数据，实现对AD、轻度认知障碍和正常认知状态的准确检测，提升了诊断准确性。**

- **链接: [http://arxiv.org/pdf/2507.08855v1](http://arxiv.org/pdf/2507.08855v1)**

> **作者:** Yang Ming; Jiang Shi Zhong; Zhou Su Juan
>
> **摘要:** Alzheimer's Disease (AD) is an irreversible neurodegenerative disease characterized by progressive cognitive decline as its main symptom. In the research field of deep learning-assisted diagnosis of AD, traditional convolutional neural networks and simple feature concatenation methods fail to effectively utilize the complementary information between multimodal data, and the simple feature concatenation approach is prone to cause the loss of key information during the process of modal fusion. In recent years, the development of deep learning technology has brought new possibilities for solving the problem of how to effectively fuse multimodal features. This paper proposes a novel deep learning algorithm framework to assist medical professionals in AD diagnosis. By fusing medical multi-view information such as brain fluorodeoxyglucose positron emission tomography (PET), magnetic resonance imaging (MRI), genetic data, and clinical data, it can accurately detect the presence of AD, Mild Cognitive Impairment (MCI), and Cognitively Normal (CN). The innovation of the algorithm lies in the use of an asymmetric cross-modal cross-attention mechanism, which can effectively capture the key information features of the interactions between different data modal features. This paper compares the asymmetric cross-modal cross-attention mechanism with the traditional algorithm frameworks of unimodal and multimodal deep learning models for AD diagnosis, and evaluates the importance of the asymmetric cross-modal cross-attention mechanism. The algorithm model achieves an accuracy of 94.88% on the test set.
>
---
#### [new 173] I2I-PR: Deep Iterative Refinement for Phase Retrieval using Image-to-Image Diffusion Models
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像重建任务，旨在解决相位恢复问题，即从仅强度测量中恢复信号。作者提出了一种结合迭代优化与图像到图像扩散模型的新方法I2I-PR，通过改进初始化和迭代细化策略，提升了重建质量和效率，优于传统及现有方法。**

- **链接: [http://arxiv.org/pdf/2507.09609v1](http://arxiv.org/pdf/2507.09609v1)**

> **作者:** Mehmet Onurcan Kaya; Figen S. Oktem
>
> **摘要:** Phase retrieval involves recovering a signal from intensity-only measurements, crucial in many fields such as imaging, holography, optical computing, crystallography, and microscopy. Although there are several well-known phase retrieval algorithms, including classical iterative solvers, the reconstruction performance often remains sensitive to initialization and measurement noise. Recently, image-to-image diffusion models have gained traction in various image reconstruction tasks, yielding significant theoretical insights and practical breakthroughs. In this work, we introduce a novel phase retrieval approach based on an image-to-image diffusion framework called Inversion by Direct Iteration. Our method begins with an enhanced initialization stage that leverages a hybrid iterative technique, combining the Hybrid Input-Output and Error Reduction methods and incorporating a novel acceleration mechanism to obtain a robust crude estimate. Then, it iteratively refines this initial crude estimate using the learned image-to-image pipeline. Our method achieves substantial improvements in both training efficiency and reconstruction quality. Furthermore, our approach utilizes aggregation techniques to refine quality metrics and demonstrates superior results compared to both classical and contemporary techniques. This highlights its potential for effective and efficient phase retrieval across various applications.
>
---
#### [new 174] Automatic Contouring of Spinal Vertebrae on X-Ray using a Novel Sandwich U-Net Architecture
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决脊柱椎体X光图像中手动轮廓提取效率低、易出错的问题。作者提出了一种新型“三明治”U-Net结构，结合双激活函数，在胸椎分割中提升了Dice评分，实现了更准确的椎体轮廓自动提取。**

- **链接: [http://arxiv.org/pdf/2507.09158v1](http://arxiv.org/pdf/2507.09158v1)**

> **作者:** Sunil Munthumoduku Krishna Murthy; Kumar Rajamani; Srividya Tirunellai Rajamani; Yupei Li; Qiyang Sun; Bjoern W. Schuller
>
> **摘要:** In spinal vertebral mobility disease, accurately extracting and contouring vertebrae is essential for assessing mobility impairments and monitoring variations during flexion-extension movements. Precise vertebral contouring plays a crucial role in surgical planning; however, this process is traditionally performed manually by radiologists or surgeons, making it labour-intensive, time-consuming, and prone to human error. In particular, mobility disease analysis requires the individual contouring of each vertebra, which is both tedious and susceptible to inconsistencies. Automated methods provide a more efficient alternative, enabling vertebra identification, segmentation, and contouring with greater accuracy and reduced time consumption. In this study, we propose a novel U-Net variation designed to accurately segment thoracic vertebrae from anteroposterior view on X-Ray images. Our proposed approach, incorporating a ``sandwich" U-Net structure with dual activation functions, achieves a 4.1\% improvement in Dice score compared to the baseline U-Net model, enhancing segmentation accuracy while ensuring reliable vertebral contour extraction.
>
---
#### [new 175] Pre-trained Under Noise: A Framework for Robust Bone Fracture Detection in Medical Imaging
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决不同设备质量对骨科影像诊断的影响问题。通过模拟噪声干扰，研究预训练深度学习模型（如ResNet50、VGG16和EfficientNetv2）在骨折分类中的鲁棒性，并提出一种评估AI模型退化的方法框架。**

- **链接: [http://arxiv.org/pdf/2507.09731v1](http://arxiv.org/pdf/2507.09731v1)**

> **作者:** Robby Hoover; Nelly Elsayed; Zag ElSayed; Chengcheng Li
>
> **备注:** 7 pages, under review
>
> **摘要:** Medical Imagings are considered one of the crucial diagnostic tools for different bones-related diseases, especially bones fractures. This paper investigates the robustness of pre-trained deep learning models for classifying bone fractures in X-ray images and seeks to address global healthcare disparity through the lens of technology. Three deep learning models have been tested under varying simulated equipment quality conditions. ResNet50, VGG16 and EfficientNetv2 are the three pre-trained architectures which are compared. These models were used to perform bone fracture classification as images were progressively degraded using noise. This paper specifically empirically studies how the noise can affect the bone fractures detection and how the pre-trained models performance can be changes due to the noise that affect the quality of the X-ray images. This paper aims to help replicate real world challenges experienced by medical imaging technicians across the world. Thus, this paper establishes a methodological framework for assessing AI model degradation using transfer learning and controlled noise augmentation. The findings provide practical insight into how robust and generalizable different pre-trained deep learning powered computer vision models can be when used in different contexts.
>
---
#### [new 176] prNet: Data-Driven Phase Retrieval via Stochastic Refinement
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像重建任务，旨在解决相位恢复问题。传统方法侧重像素精度，忽视感知质量。论文提出prNet框架，结合随机采样与去噪，通过Langevin动力学优化重建结果，在失真与感知间取得平衡，提升重建质量。**

- **链接: [http://arxiv.org/pdf/2507.09608v1](http://arxiv.org/pdf/2507.09608v1)**

> **作者:** Mehmet Onurcan Kaya; Figen S. Oktem
>
> **摘要:** We propose a novel framework for phase retrieval that leverages Langevin dynamics to enable efficient posterior sampling, yielding reconstructions that explicitly balance distortion and perceptual quality. Unlike conventional approaches that prioritize pixel-wise accuracy, our method navigates the perception-distortion tradeoff through a principled combination of stochastic sampling, learned denoising, and model-based updates. The framework comprises three variants of increasing complexity, integrating theoretically grounded Langevin inference, adaptive noise schedule learning, parallel reconstruction sampling, and warm-start initialization from classical solvers. Extensive experiments demonstrate that our method achieves state-of-the-art performance across multiple benchmarks, both in terms of fidelity and perceptual quality.
>
---
#### [new 177] Warm Starts Accelerate Generative Modelling
- **分类: cs.LG; cs.CV; stat.ML**

- **简介: 该论文属于生成建模任务，旨在加速迭代生成模型（如扩散模型）的采样过程。传统方法从无信息噪声开始，需大量函数评估，而论文提出“warm-start”模型，通过预测基于输入上下文的有信息先验分布，显著减少生成所需步数，实现了在图像补全等任务中高效生成。**

- **链接: [http://arxiv.org/pdf/2507.09212v1](http://arxiv.org/pdf/2507.09212v1)**

> **作者:** Jonas Scholz; Richard E. Turner
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Iterative generative models, like diffusion and flow-matching, create high-fidelity samples by progressively refining a noise vector into data. However, this process is notoriously slow, often requiring hundreds of function evaluations. We introduce the warm-start model, a simple, deterministic model that dramatically accelerates conditional generation by providing a better starting point. Instead of starting generation from an uninformed N(0, I) prior, our warm-start model predicts an informed prior N(mu, sigma), whose moments are conditioned on the input context. This "warm start" substantially reduces the distance the generative process must traverse, particularly when the conditioning information is strongly informative. On tasks like image inpainting, our method achieves results competitive with a 1000-step DDPM baseline using only 11 total function evaluations (1 for the warm start, 10 for generation). A simple conditional normalization trick makes our method compatible with any standard generative model and sampler without modification, allowing it to be combined with other efficient sampling techniques for further acceleration. Our implementation is available at https://github.com/jonas-scholz123/warm-start-model.
>
---
#### [new 178] Learning Private Representations through Entropy-based Adversarial Training
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于隐私保护任务，旨在学习具有高预测性能且保留用户隐私的表示。为解决敏感信息泄露问题，作者提出基于焦点熵的对抗训练方法，有效减少信息泄露，在多个基准数据上验证了方法可行性，取得了较好的隐私与效用平衡。**

- **链接: [http://arxiv.org/pdf/2507.10194v1](http://arxiv.org/pdf/2507.10194v1)**

> **作者:** Tassilo Klein; Moin Nabi
>
> **摘要:** How can we learn a representation with high predictive power while preserving user privacy? We present an adversarial representation learning method for sanitizing sensitive content from the learned representation. Specifically, we introduce a variant of entropy - focal entropy, which mitigates the potential information leakage of the existing entropy-based approaches. We showcase feasibility on multiple benchmarks. The results suggest high target utility at moderate privacy leakage.
>
---
#### [new 179] AI-Enhanced Pediatric Pneumonia Detection: A CNN-Based Approach Using Data Augmentation and Generative Adversarial Networks (GANs)
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决儿童肺炎的准确诊断问题。通过构建基于CNN的模型，结合数据增强与GAN生成合成图像以缓解数据不足及类别不平衡问题，并开发了用于实时分类的Web应用。**

- **链接: [http://arxiv.org/pdf/2507.09759v1](http://arxiv.org/pdf/2507.09759v1)**

> **作者:** Abdul Manaf; Nimra Mughal
>
> **摘要:** Pneumonia is a leading cause of mortality in children under five, requiring accurate chest X-ray diagnosis. This study presents a machine learning-based Pediatric Chest Pneumonia Classification System to assist healthcare professionals in diagnosing pneumonia from chest X-ray images. The CNN-based model was trained on 5,863 labeled chest X-ray images from children aged 0-5 years from the Guangzhou Women and Children's Medical Center. To address limited data, we applied augmentation techniques (rotation, zooming, shear, horizontal flipping) and employed GANs to generate synthetic images, addressing class imbalance. The system achieved optimal performance using combined original, augmented, and GAN-generated data, evaluated through accuracy and F1 score metrics. The final model was deployed via a Flask web application, enabling real-time classification with probability estimates. Results demonstrate the potential of deep learning and GANs in improving diagnostic accuracy and efficiency for pediatric pneumonia classification, particularly valuable in resource-limited clinical settings https://github.com/AbdulManaf12/Pediatric-Chest-Pneumonia-Classification
>
---
#### [new 180] Interpretable Artificial Intelligence for Detecting Acute Heart Failure on Acute Chest CT Scans
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决急性心力衰竭（AHF）在胸部CT扫描中的快速准确检测问题。研究团队开发了一种可解释的人工智能模型，基于胸部CT的分割结构进行预测，并通过专家评审验证了模型的高准确性与可解释性，以辅助急诊医生决策。**

- **链接: [http://arxiv.org/pdf/2507.08952v1](http://arxiv.org/pdf/2507.08952v1)**

> **作者:** Silas Nyboe Ørting; Kristina Miger; Anne Sophie Overgaard Olesen; Mikael Ploug Boesen; Michael Brun Andersen; Jens Petersen; Olav W. Nielsen; Marleen de Bruijne
>
> **备注:** 34 pages, 11 figures, Submitted to "Radiology AI"
>
> **摘要:** Introduction: Chest CT scans are increasingly used in dyspneic patients where acute heart failure (AHF) is a key differential diagnosis. Interpretation remains challenging and radiology reports are frequently delayed due to a radiologist shortage, although flagging such information for emergency physicians would have therapeutic implication. Artificial intelligence (AI) can be a complementary tool to enhance the diagnostic precision. We aim to develop an explainable AI model to detect radiological signs of AHF in chest CT with an accuracy comparable to thoracic radiologists. Methods: A single-center, retrospective study during 2016-2021 at Copenhagen University Hospital - Bispebjerg and Frederiksberg, Denmark. A Boosted Trees model was trained to predict AHF based on measurements of segmented cardiac and pulmonary structures from acute thoracic CT scans. Diagnostic labels for training and testing were extracted from radiology reports. Structures were segmented with TotalSegmentator. Shapley Additive explanations values were used to explain the impact of each measurement on the final prediction. Results: Of the 4,672 subjects, 49% were female. The final model incorporated twelve key features of AHF and achieved an area under the ROC of 0.87 on the independent test set. Expert radiologist review of model misclassifications found that 24 out of 64 (38%) false positives and 24 out of 61 (39%) false negatives were actually correct model predictions, with the errors originating from inaccuracies in the initial radiology reports. Conclusion: We developed an explainable AI model with strong discriminatory performance, comparable to thoracic radiologists. The AI model's stepwise, transparent predictions may support decision-making.
>
---
#### [new 181] CLA: Latent Alignment for Online Continual Self-Supervised Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于在线持续自监督学习任务，旨在解决小批量数据流中模型遗忘问题。作者提出CLA方法，通过对齐当前与过去表征来缓解遗忘，提升训练收敛速度与性能，超越现有技术。**

- **链接: [http://arxiv.org/pdf/2507.10434v1](http://arxiv.org/pdf/2507.10434v1)**

> **作者:** Giacomo Cignoni; Andrea Cossu; Alexandra Gomez-Villa; Joost van de Weijer; Antonio Carta
>
> **备注:** Accepted at CoLLAs 2025 conference
>
> **摘要:** Self-supervised learning (SSL) is able to build latent representations that generalize well to unseen data. However, only a few SSL techniques exist for the online CL setting, where data arrives in small minibatches, the model must comply with a fixed computational budget, and task boundaries are absent. We introduce Continual Latent Alignment (CLA), a novel SSL strategy for Online CL that aligns the representations learned by the current model with past representations to mitigate forgetting. We found that our CLA is able to speed up the convergence of the training process in the online scenario, outperforming state-of-the-art approaches under the same computational budget. Surprisingly, we also discovered that using CLA as a pretraining protocol in the early stages of pretraining leads to a better final performance when compared to a full i.i.d. pretraining.
>
---
#### [new 182] Graph-based Multi-Modal Interaction Lightweight Network for Brain Tumor Segmentation (GMLN-BTS) in Edge Iterative MRI Lesion Localization System (EdgeIMLocSys)
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决不同MRI扫描仪成像质量差异导致的模型泛化能力差问题。作者提出了轻量级网络GMLN-BTS，结合多模态交互与图结构，并引入反馈机制，实现高效、准确的脑肿瘤分割。**

- **链接: [http://arxiv.org/pdf/2507.09995v1](http://arxiv.org/pdf/2507.09995v1)**

> **作者:** Guohao Huo; Ruiting Dai; Hao Tang
>
> **摘要:** Brain tumor segmentation plays a critical role in clinical diagnosis and treatment planning, yet the variability in imaging quality across different MRI scanners presents significant challenges to model generalization. To address this, we propose the Edge Iterative MRI Lesion Localization System (EdgeIMLocSys), which integrates Continuous Learning from Human Feedback to adaptively fine-tune segmentation models based on clinician feedback, thereby enhancing robustness to scanner-specific imaging characteristics. Central to this system is the Graph-based Multi-Modal Interaction Lightweight Network for Brain Tumor Segmentation (GMLN-BTS), which employs a Modality-Aware Adaptive Encoder (M2AE) to extract multi-scale semantic features efficiently, and a Graph-based Multi-Modal Collaborative Interaction Module (G2MCIM) to model complementary cross-modal relationships via graph structures. Additionally, we introduce a novel Voxel Refinement UpSampling Module (VRUM) that synergistically combines linear interpolation and multi-scale transposed convolutions to suppress artifacts while preserving high-frequency details, improving segmentation boundary accuracy. Our proposed GMLN-BTS model achieves a Dice score of 85.1% on the BraTS2017 dataset with only 4.58 million parameters, representing a 98% reduction compared to mainstream 3D Transformer models, and significantly outperforms existing lightweight approaches. This work demonstrates a synergistic breakthrough in achieving high-accuracy, resource-efficient brain tumor segmentation suitable for deployment in resource-constrained clinical environments.
>
---
#### [new 183] Advanced U-Net Architectures with CNN Backbones for Automated Lung Cancer Detection and Segmentation in Chest CT Images
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分析任务，旨在解决肺癌的自动检测与分割问题。研究采用基于U-Net架构并结合不同CNN主干网络（ResNet50、VGG16、Xception）的方法对胸部CT图像进行肺部区域分割及癌症分类。实验结果显示所提方法在分割和分类任务上均优于现有模型，有助于肺癌早期诊断与临床决策支持。**

- **链接: [http://arxiv.org/pdf/2507.09898v1](http://arxiv.org/pdf/2507.09898v1)**

> **作者:** Alireza Golkarieha; Kiana Kiashemshakib; Sajjad Rezvani Boroujenic; Nasibeh Asadi Isakand
>
> **备注:** This manuscript has 20 pages and 10 figures. It is submitted to the Journal 'Scientific Reports'
>
> **摘要:** This study investigates the effectiveness of U-Net architectures integrated with various convolutional neural network (CNN) backbones for automated lung cancer detection and segmentation in chest CT images, addressing the critical need for accurate diagnostic tools in clinical settings. A balanced dataset of 832 chest CT images (416 cancerous and 416 non-cancerous) was preprocessed using Contrast Limited Adaptive Histogram Equalization (CLAHE) and resized to 128x128 pixels. U-Net models were developed with three CNN backbones: ResNet50, VGG16, and Xception, to segment lung regions. After segmentation, CNN-based classifiers and hybrid models combining CNN feature extraction with traditional machine learning classifiers (Support Vector Machine, Random Forest, and Gradient Boosting) were evaluated using 5-fold cross-validation. Metrics included accuracy, precision, recall, F1-score, Dice coefficient, and ROC-AUC. U-Net with ResNet50 achieved the best performance for cancerous lungs (Dice: 0.9495, Accuracy: 0.9735), while U-Net with VGG16 performed best for non-cancerous segmentation (Dice: 0.9532, Accuracy: 0.9513). For classification, the CNN model using U-Net with Xception achieved 99.1 percent accuracy, 99.74 percent recall, and 99.42 percent F1-score. The hybrid CNN-SVM-Xception model achieved 96.7 percent accuracy and 97.88 percent F1-score. Compared to prior methods, our framework consistently outperformed existing models. In conclusion, combining U-Net with advanced CNN backbones provides a powerful method for both segmentation and classification of lung cancer in CT scans, supporting early diagnosis and clinical decision-making.
>
---
#### [new 184] Visual Homing in Outdoor Robots Using Mushroom Body Circuits and Learning Walks
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人视觉导航任务，旨在解决户外环境下自主机器人高效视觉归巢问题。受蚂蚁归巢机制启发，作者基于蘑菇体神经回路设计了一种轻量级模型，结合路径积分与学习行走策略，在真实环境中实现精准归巢。**

- **链接: [http://arxiv.org/pdf/2507.09725v1](http://arxiv.org/pdf/2507.09725v1)**

> **作者:** Gabriel G. Gattaux; Julien R. Serres; Franck Ruffier; Antoine Wystrach
>
> **备注:** Published by Springer Nature with the 14th bioinspired and biohybrid systems conference in Sheffield, and presented at the conference in July 2025
>
> **摘要:** Ants achieve robust visual homing with minimal sensory input and only a few learning walks, inspiring biomimetic solutions for autonomous navigation. While Mushroom Body (MB) models have been used in robotic route following, they have not yet been applied to visual homing. We present the first real-world implementation of a lateralized MB architecture for visual homing onboard a compact autonomous car-like robot. We test whether the sign of the angular path integration (PI) signal can categorize panoramic views, acquired during learning walks and encoded in the MB, into "goal on the left" and "goal on the right" memory banks, enabling robust homing in natural outdoor settings. We validate this approach through four incremental experiments: (1) simulation showing attractor-like nest dynamics; (2) real-world homing after decoupled learning walks, producing nest search behavior; (3) homing after random walks using noisy PI emulated with GPS-RTK; and (4) precise stopping-at-the-goal behavior enabled by a fifth MB Output Neuron (MBON) encoding goal-views to control velocity. This mimics the accurate homing behavior of ants and functionally resembles waypoint-based position control in robotics, despite relying solely on visual input. Operating at 8 Hz on a Raspberry Pi 4 with 32x32 pixel views and a memory footprint under 9 kB, our system offers a biologically grounded, resource-efficient solution for autonomous visual homing.
>
---
#### [new 185] MLoRQ: Bridging Low-Rank and Quantization for Transformer Compression
- **分类: cs.LG; cs.CV**

- **简介: 论文提出MLoRQ方法，用于压缩Transformer模型，解决在资源受限设备上部署的问题。结合低秩近似与量化技术，优化各层的位宽和秩分配，满足内存约束并提升性能，适用于多种任务。**

- **链接: [http://arxiv.org/pdf/2507.09616v1](http://arxiv.org/pdf/2507.09616v1)**

> **作者:** Ofir Gordon; Ariel Lapid; Elad Cohen; Yarden Yagil; Arnon Netzer; Hai Victor Habi
>
> **摘要:** Deploying transformer-based neural networks on resource-constrained edge devices presents a significant challenge. This challenge is often addressed through various techniques, such as low-rank approximation and mixed-precision quantization. In this work, we introduce Mixed Low-Rank and Quantization (MLoRQ), a novel method that integrates both techniques. MLoRQ employs a two-stage optimization process to determine optimal bit-width and rank assignments for each layer, adhering to predefined memory constraints. This process includes: (i) an intra-layer optimization that identifies potentially optimal compression solutions out of all low-rank and quantization combinations; (ii) an inter-layer optimization that assigns bit-width precision and rank to each layer while ensuring the memory constraint is met. An optional final step applies a sequential optimization process using a modified adaptive rounding technique to mitigate compression-induced errors in joint low-rank approximation and quantization. The method is compatible and can be seamlessly integrated with most existing quantization algorithms. MLoRQ shows state-of-the-art results with up to 15\% performance improvement, evaluated on Vision Transformers for image classification, object detection, and instance segmentation tasks.
>
---
#### [new 186] ScaffoldAvatar: High-Fidelity Gaussian Avatars with Patch Expressions
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 该论文属于3D头像生成任务，旨在解决高保真、实时动画化3D人脸 avatar 生成问题。作者提出 ScaffoldAvatar，结合基于局部表情的高斯点阵与 patch 表达，实现高质量、动态逼真的头像生成。**

- **链接: [http://arxiv.org/pdf/2507.10542v1](http://arxiv.org/pdf/2507.10542v1)**

> **作者:** Shivangi Aneja; Sebastian Weiss; Irene Baeza; Prashanth Chandran; Gaspard Zoss; Matthias Nießner; Derek Bradley
>
> **备注:** (SIGGRAPH 2025) Paper Video: https://youtu.be/VyWkgsGdbkk Project Page: https://shivangi-aneja.github.io/projects/scaffoldavatar/
>
> **摘要:** Generating high-fidelity real-time animated sequences of photorealistic 3D head avatars is important for many graphics applications, including immersive telepresence and movies. This is a challenging problem particularly when rendering digital avatar close-ups for showing character's facial microfeatures and expressions. To capture the expressive, detailed nature of human heads, including skin furrowing and finer-scale facial movements, we propose to couple locally-defined facial expressions with 3D Gaussian splatting to enable creating ultra-high fidelity, expressive and photorealistic 3D head avatars. In contrast to previous works that operate on a global expression space, we condition our avatar's dynamics on patch-based local expression features and synthesize 3D Gaussians at a patch level. In particular, we leverage a patch-based geometric 3D face model to extract patch expressions and learn how to translate these into local dynamic skin appearance and motion by coupling the patches with anchor points of Scaffold-GS, a recent hierarchical scene representation. These anchors are then used to synthesize 3D Gaussians on-the-fly, conditioned by patch-expressions and viewing direction. We employ color-based densification and progressive training to obtain high-quality results and faster convergence for high resolution 3K training images. By leveraging patch-level expressions, ScaffoldAvatar consistently achieves state-of-the-art performance with visually natural motion, while encompassing diverse facial expressions and styles in real time.
>
---
#### [new 187] CNeuroMod-THINGS, a densely-sampled fMRI dataset for visual neuroscience
- **分类: q-bio.NC; cs.CV**

- **简介: 该论文属于神经科学与人工智能交叉任务，旨在解决视觉神经模型缺乏高质量、大规模数据的问题。作者通过整合THINGS和CNeuroMod项目资源，构建了CNeuroMod-THINGS这一高采样fMRI数据集，包含四名参与者在连续识别任务中的脑成像数据，用于提升人类视觉体验建模能力。**

- **链接: [http://arxiv.org/pdf/2507.09024v1](http://arxiv.org/pdf/2507.09024v1)**

> **作者:** Marie St-Laurent; Basile Pinsard; Oliver Contier; Elizabeth DuPre; Katja Seeliger; Valentina Borghesani; Julie A. Boyle; Lune Bellec; Martin N. Hebart
>
> **备注:** 29 pages manuscript, 5 figures, 12 pages supplementary material
>
> **摘要:** Data-hungry neuro-AI modelling requires ever larger neuroimaging datasets. CNeuroMod-THINGS meets this need by capturing neural representations for a wide set of semantic concepts using well-characterized stimuli in a new densely-sampled, large-scale fMRI dataset. Importantly, CNeuroMod-THINGS exploits synergies between two existing projects: the THINGS initiative (THINGS) and the Courtois Project on Neural Modelling (CNeuroMod). THINGS has developed a common set of thoroughly annotated images broadly sampling natural and man-made objects which is used to acquire a growing collection of large-scale multimodal neural responses. Meanwhile, CNeuroMod is acquiring hundreds of hours of fMRI data from a core set of participants during controlled and naturalistic tasks, including visual tasks like movie watching and videogame playing. For CNeuroMod-THINGS, four CNeuroMod participants each completed 33-36 sessions of a continuous recognition paradigm using approximately 4000 images from the THINGS stimulus set spanning 720 categories. We report behavioural and neuroimaging metrics that showcase the quality of the data. By bridging together large existing resources, CNeuroMod-THINGS expands our capacity to model broad slices of the human visual experience.
>
---
#### [new 188] PanoDiff-SR: Synthesizing Dental Panoramic Radiographs using Diffusion and Super-resolution
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于医学图像生成任务，旨在解决高质量牙科全景X光片数据稀缺问题。作者提出PanoDiff-SR方法，结合扩散模型与超分辨率技术，先生成低分辨率图像，再提升至高分辨率。实验表明合成图像质量接近真实数据，具备应用潜力。**

- **链接: [http://arxiv.org/pdf/2507.09227v1](http://arxiv.org/pdf/2507.09227v1)**

> **作者:** Sanyam Jain; Bruna Neves de Freitas; Andreas Basse-OConnor; Alexandros Iosifidis; Ruben Pauwels
>
> **摘要:** There has been increasing interest in the generation of high-quality, realistic synthetic medical images in recent years. Such synthetic datasets can mitigate the scarcity of public datasets for artificial intelligence research, and can also be used for educational purposes. In this paper, we propose a combination of diffusion-based generation (PanoDiff) and Super-Resolution (SR) for generating synthetic dental panoramic radiographs (PRs). The former generates a low-resolution (LR) seed of a PR (256 X 128) which is then processed by the SR model to yield a high-resolution (HR) PR of size 1024 X 512. For SR, we propose a state-of-the-art transformer that learns local-global relationships, resulting in sharper edges and textures. Experimental results demonstrate a Frechet inception distance score of 40.69 between 7243 real and synthetic images (in HR). Inception scores were 2.55, 2.30, 2.90 and 2.98 for real HR, synthetic HR, real LR and synthetic LR images, respectively. Among a diverse group of six clinical experts, all evaluating a mixture of 100 synthetic and 100 real PRs in a time-limited observation, the average accuracy in distinguishing real from synthetic images was 68.5% (with 50% corresponding to random guessing).
>
---
#### [new 189] Lightweight Deep Learning-Based Channel Estimation for RIS-Aided Extremely Large-Scale MIMO Systems on Resource-Limited Edge Devices
- **分类: cs.IT; cs.CV; cs.LG; cs.NI; math.IT**

- **简介: 该论文属于无线通信中的信道估计任务，旨在解决6G中大规模MIMO与智能反射面系统级联信道估计的高复杂度问题。作者提出了一种轻量级深度学习框架，利用信道空间相关性，采用基于块的训练机制，降低了输入维度，在保证精度的同时显著减少了计算复杂度，适用于资源受限的边缘设备部署。**

- **链接: [http://arxiv.org/pdf/2507.09627v1](http://arxiv.org/pdf/2507.09627v1)**

> **作者:** Muhammad Kamran Saeed; Ashfaq Khokhar; Shakil Ahmed
>
> **摘要:** Next-generation wireless technologies such as 6G aim to meet demanding requirements such as ultra-high data rates, low latency, and enhanced connectivity. Extremely Large-Scale MIMO (XL-MIMO) and Reconfigurable Intelligent Surface (RIS) are key enablers, with XL-MIMO boosting spectral and energy efficiency through numerous antennas, and RIS offering dynamic control over the wireless environment via passive reflective elements. However, realizing their full potential depends on accurate Channel State Information (CSI). Recent advances in deep learning have facilitated efficient cascaded channel estimation. However, the scalability and practical deployment of existing estimation models in XL-MIMO systems remain limited. The growing number of antennas and RIS elements introduces a significant barrier to real-time and efficient channel estimation, drastically increasing data volume, escalating computational complexity, requiring advanced hardware, and resulting in substantial energy consumption. To address these challenges, we propose a lightweight deep learning framework for efficient cascaded channel estimation in XL-MIMO systems, designed to minimize computational complexity and make it suitable for deployment on resource-constrained edge devices. Using spatial correlations in the channel, we introduce a patch-based training mechanism that reduces the dimensionality of input to patch-level representations while preserving essential information, allowing scalable training for large-scale systems. Simulation results under diverse conditions demonstrate that our framework significantly improves estimation accuracy and reduces computational complexity, regardless of the increasing number of antennas and RIS elements in XL-MIMO systems.
>
---
#### [new 190] DepViT-CAD: Deployable Vision Transformer-Based Cancer Diagnosis in Histopathology
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于癌症诊断任务，旨在解决基于组织病理学切片的多类别癌症诊断问题。作者提出了DepViT-CAD系统，其核心为MAViT模型，采用多注意力机制识别肿瘤形态特征，并在大规模数据上训练和验证，实现高精度癌症分类，提升临床决策效率。**

- **链接: [http://arxiv.org/pdf/2507.10250v1](http://arxiv.org/pdf/2507.10250v1)**

> **作者:** Ashkan Shakarami; Lorenzo Nicole; Rocco Cappellesso; Angelo Paolo Dei Tos; Stefano Ghidoni
>
> **备注:** 25 pages, 15 figures
>
> **摘要:** Accurate and timely cancer diagnosis from histopathological slides is vital for effective clinical decision-making. This paper introduces DepViT-CAD, a deployable AI system for multi-class cancer diagnosis in histopathology. At its core is MAViT, a novel Multi-Attention Vision Transformer designed to capture fine-grained morphological patterns across diverse tumor types. MAViT was trained on expert-annotated patches from 1008 whole-slide images, covering 11 diagnostic categories, including 10 major cancers and non-tumor tissue. DepViT-CAD was validated on two independent cohorts: 275 WSIs from The Cancer Genome Atlas and 50 routine clinical cases from pathology labs, achieving diagnostic sensitivities of 94.11% and 92%, respectively. By combining state-of-the-art transformer architecture with large-scale real-world validation, DepViT-CAD offers a robust and scalable approach for AI-assisted cancer diagnostics. To support transparency and reproducibility, software and code will be made publicly available at GitHub.
>
---
#### [new 191] Generative Audio Language Modeling with Continuous-valued Tokens and Masked Next-Token Prediction
- **分类: eess.AS; cs.CV; cs.SD**

- **简介: 该论文属于音频生成任务，旨在解决连续值音频建模问题。作者提出基于Transformer的因果语言模型，采用token-wise扩散和掩码下一token预测，提升生成质量，参数量更少却取得与SOTA扩散模型相当的效果。**

- **链接: [http://arxiv.org/pdf/2507.09834v1](http://arxiv.org/pdf/2507.09834v1)**

> **作者:** Shu-wen Yang; Byeonggeun Kim; Kuan-Po Huang; Qingming Tang; Huy Phan; Bo-Ru Lu; Harsha Sundar; Shalini Ghosh; Hung-yi Lee; Chieh-Chi Kao; Chao Wang
>
> **备注:** Accepted by ICML 2025. Project website: https://audiomntp.github.io/
>
> **摘要:** Autoregressive next-token prediction with the Transformer decoder has become a de facto standard in large language models (LLMs), achieving remarkable success in Natural Language Processing (NLP) at scale. Extending this paradigm to audio poses unique challenges due to its inherently continuous nature. We research audio generation with a causal language model (LM) without discrete tokens. We leverage token-wise diffusion to model the continuous distribution of the next continuous-valued token. Our approach delivers significant improvements over previous discrete solution, AudioGen, achieving 20% and 40% relative gains on AudioCaps in Frechet Audio Distance (FAD) and Kullback-Leibler (KL) divergence, respectively. Additionally, we propose a novel masked next-token prediction task that incorporates masked prediction into the causal LM framework. On AudioCaps, the innovation yields 41% and 33% relative FAD improvements over AudioGen Base (285M) and AudioGen Large (1B) models, respectively, and is on par with the state-of-the-art (SOTA) diffusion models. Furthermore, we achieve these results with significantly fewer parameters -- 193M for our Base and 462M for our Large models.
>
---
#### [new 192] Resolution Revolution: A Physics-Guided Deep Learning Framework for Spatiotemporal Temperature Reconstruction
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于遥感与温度建模任务，旨在解决现有温度数据源在空间和时间分辨率之间的权衡问题。作者提出了一种结合物理规律的深度学习框架，融合高时间分辨率但低空间分辨率的数据与高空间分辨率但低时间分辨率的数据，实现了高时空分辨率的温度重建。**

- **链接: [http://arxiv.org/pdf/2507.09872v1](http://arxiv.org/pdf/2507.09872v1)**

> **作者:** Shengjie Liu; Lu Zhang; Siqin Wang
>
> **备注:** ICCV 2025 Workshop SEA -- International Conference on Computer Vision 2025 Workshop on Sustainability with Earth Observation and AI
>
> **摘要:** Central to Earth observation is the trade-off between spatial and temporal resolution. For temperature, this is especially critical because real-world applications require high spatiotemporal resolution data. Current technology allows for hourly temperature observations at 2 km, but only every 16 days at 100 m, a gap further exacerbated by cloud cover. Earth system models offer continuous hourly temperature data, but at a much coarser spatial resolution (9-31 km). Here, we present a physics-guided deep learning framework for temperature data reconstruction that integrates these two data sources. The proposed framework uses a convolutional neural network that incorporates the annual temperature cycle and includes a linear term to amplify the coarse Earth system model output into fine-scale temperature values observed from satellites. We evaluated this framework using data from two satellites, GOES-16 (2 km, hourly) and Landsat (100 m, every 16 days), and demonstrated effective temperature reconstruction with hold-out and in situ data across four datasets. This physics-guided deep learning framework opens new possibilities for generating high-resolution temperature data across spatial and temporal scales, under all weather conditions and globally.
>
---
#### [new 193] Learning Diffusion Models with Flexible Representation Guidance
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于生成模型任务，旨在提升扩散模型的生成质量和训练效率。通过引入表示引导框架，结合预训练模型的表征能力，提出两种新策略：多模态对联合建模与最优训练课程设计。实验证明其在图像、蛋白质和分子生成中效果优越，训练加速明显。**

- **链接: [http://arxiv.org/pdf/2507.08980v1](http://arxiv.org/pdf/2507.08980v1)**

> **作者:** Chenyu Wang; Cai Zhou; Sharut Gupta; Zongyu Lin; Stefanie Jegelka; Stephen Bates; Tommi Jaakkola
>
> **摘要:** Diffusion models can be improved with additional guidance towards more effective representations of input. Indeed, prior empirical work has already shown that aligning internal representations of the diffusion model with those of pre-trained models improves generation quality. In this paper, we present a systematic framework for incorporating representation guidance into diffusion models. We provide alternative decompositions of denoising models along with their associated training criteria, where the decompositions determine when and how the auxiliary representations are incorporated. Guided by our theoretical insights, we introduce two new strategies for enhancing representation alignment in diffusion models. First, we pair examples with target representations either derived from themselves or arisen from different synthetic modalities, and subsequently learn a joint model over the multimodal pairs. Second, we design an optimal training curriculum that balances representation learning and data generation. Our experiments across image, protein sequence, and molecule generation tasks demonstrate superior performance as well as accelerated training. In particular, on the class-conditional ImageNet $256\times 256$ benchmark, our guidance results in $23.3$ times faster training than the original SiT-XL as well as four times speedup over the state-of-the-art method REPA. The code is available at https://github.com/ChenyuWang-Monica/REED.
>
---
#### [new 194] Self-supervised pretraining of vision transformers for animal behavioral analysis and neural encoding
- **分类: q-bio.NC; cs.CV**

- **简介: 该论文属于计算机视觉与神经科学交叉任务，旨在解决动物行为分析中标注数据不足的问题。作者提出了BEAST框架，通过自监督预训练视觉Transformer模型，结合掩码自编码和时间对比学习，提升行为特征提取、姿态估计和动作分割的效果，适用于多物种及多动物场景。**

- **链接: [http://arxiv.org/pdf/2507.09513v1](http://arxiv.org/pdf/2507.09513v1)**

> **作者:** Yanchen Wang; Han Yu; Ari Blau; Yizi Zhang; The International Brain Laboratory; Liam Paninski; Cole Hurwitz; Matt Whiteway
>
> **摘要:** The brain can only be fully understood through the lens of the behavior it generates -- a guiding principle in modern neuroscience research that nevertheless presents significant technical challenges. Many studies capture behavior with cameras, but video analysis approaches typically rely on specialized models requiring extensive labeled data. We address this limitation with BEAST (BEhavioral Analysis via Self-supervised pretraining of Transformers), a novel and scalable framework that pretrains experiment-specific vision transformers for diverse neuro-behavior analyses. BEAST combines masked autoencoding with temporal contrastive learning to effectively leverage unlabeled video data. Through comprehensive evaluation across multiple species, we demonstrate improved performance in three critical neuro-behavioral tasks: extracting behavioral features that correlate with neural activity, and pose estimation and action segmentation in both the single- and multi-animal settings. Our method establishes a powerful and versatile backbone model that accelerates behavioral analysis in scenarios where labeled data remains scarce.
>
---
#### [new 195] VIP: Visual Information Protection through Adversarial Attacks on Vision-Language Models
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于视觉-语言模型中的隐私保护任务，旨在解决模型在处理图像时可能暴露敏感区域的问题。作者提出了一种对抗攻击方法，通过在指定感兴趣区域（ROIs）内隐藏信息，使模型无法识别这些区域内容，同时保持图像其余部分的语义完整性。实验表明该方法对LLaVA、Instruct-BLIP和BLIP2-T5等模型具有显著效果。**

- **链接: [http://arxiv.org/pdf/2507.08982v1](http://arxiv.org/pdf/2507.08982v1)**

> **作者:** Hanene F. Z. Brachemi Meftah; Wassim Hamidouche; Sid Ahmed Fezza; Olivier Déforges
>
> **摘要:** Recent years have witnessed remarkable progress in developing Vision-Language Models (VLMs) capable of processing both textual and visual inputs. These models have demonstrated impressive performance, leading to their widespread adoption in various applications. However, this widespread raises serious concerns regarding user privacy, particularly when models inadvertently process or expose private visual information. In this work, we frame the preservation of privacy in VLMs as an adversarial attack problem. We propose a novel attack strategy that selectively conceals information within designated Region Of Interests (ROIs) in an image, effectively preventing VLMs from accessing sensitive content while preserving the semantic integrity of the remaining image. Unlike conventional adversarial attacks that often disrupt the entire image, our method maintains high coherence in unmasked areas. Experimental results across three state-of-the-art VLMs namely LLaVA, Instruct-BLIP, and BLIP2-T5 demonstrate up to 98% reduction in detecting targeted ROIs, while maintaining global image semantics intact, as confirmed by high similarity scores between clean and adversarial outputs. We believe that this work contributes to a more privacy conscious use of multimodal models and offers a practical tool for further research, with the source code publicly available at: https://github.com/hbrachemi/Vlm_defense-attack.
>
---
#### [new 196] Probabilistic Human Intent Prediction for Mobile Manipulation: An Evaluation with Human-Inspired Constraints
- **分类: cs.RO; cs.CV; cs.HC**

- **简介: 论文提出GUIDER框架，用于移动操作中的人机协作，解决人类意图预测问题。通过双阶段（导航与操作）概率模型，结合多源信息实时估计用户意图。实验验证其在导航与操作任务中的稳定性与提前预测能力均优于基线方法。**

- **链接: [http://arxiv.org/pdf/2507.10131v1](http://arxiv.org/pdf/2507.10131v1)**

> **作者:** Cesar Alan Contreras; Manolis Chiou; Alireza Rastegarpanah; Michal Szulik; Rustam Stolkin
>
> **备注:** Submitted to Journal of Intelligent & Robotic Systems (Under Review)
>
> **摘要:** Accurate inference of human intent enables human-robot collaboration without constraining human control or causing conflicts between humans and robots. We present GUIDER (Global User Intent Dual-phase Estimation for Robots), a probabilistic framework that enables a robot to estimate the intent of human operators. GUIDER maintains two coupled belief layers, one tracking navigation goals and the other manipulation goals. In the Navigation phase, a Synergy Map blends controller velocity with an occupancy grid to rank interaction areas. Upon arrival at a goal, an autonomous multi-view scan builds a local 3D cloud. The Manipulation phase combines U2Net saliency, FastSAM instance saliency, and three geometric grasp-feasibility tests, with an end-effector kinematics-aware update rule that evolves object probabilities in real-time. GUIDER can recognize areas and objects of intent without predefined goals. We evaluated GUIDER on 25 trials (five participants x five task variants) in Isaac Sim, and compared it with two baselines, one for navigation and one for manipulation. Across the 25 trials, GUIDER achieved a median stability of 93-100% during navigation, compared with 60-100% for the BOIR baseline, with an improvement of 39.5% in a redirection scenario (T5). During manipulation, stability reached 94-100% (versus 69-100% for Trajectron), with a 31.4% difference in a redirection task (T3). In geometry-constrained trials (manipulation), GUIDER recognized the object intent three times earlier than Trajectron (median remaining time to confident prediction 23.6 s vs 7.8 s). These results validate our dual-phase framework and show improvements in intent inference in both phases of mobile manipulation tasks.
>
---
#### [new 197] LayLens: Improving Deepfake Understanding through Simplified Explanations
- **分类: cs.MM; cs.CV**

- **简介: 该论文属于图像取证任务，旨在解决用户难以理解深伪图像检测结果的问题。作者提出了LayLens工具，通过可解释检测、自然语言简化和图像重建三阶段流程，提升非专业用户的理解与识别信心。**

- **链接: [http://arxiv.org/pdf/2507.10066v1](http://arxiv.org/pdf/2507.10066v1)**

> **作者:** Abhijeet Narang; Parul Gupta; Liuyijia Su; Abhinav Dhall
>
> **摘要:** This demonstration paper presents $\mathbf{LayLens}$, a tool aimed to make deepfake understanding easier for users of all educational backgrounds. While prior works often rely on outputs containing technical jargon, LayLens bridges the gap between model reasoning and human understanding through a three-stage pipeline: (1) explainable deepfake detection using a state-of-the-art forgery localization model, (2) natural language simplification of technical explanations using a vision-language model, and (3) visual reconstruction of a plausible original image via guided image editing. The interface presents both technical and layperson-friendly explanations in addition to a side-by-side comparison of the uploaded and reconstructed images. A user study with 15 participants shows that simplified explanations significantly improve clarity and reduce cognitive load, with most users expressing increased confidence in identifying deepfakes. LayLens offers a step toward transparent, trustworthy, and user-centric deepfake forensics.
>
---
## 更新

#### [replaced 001] Unmixing Optical Signals from Undersampled Volumetric Measurements by Filtering the Pixel Latent Variables
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2312.05357v4](http://arxiv.org/pdf/2312.05357v4)**

> **作者:** Catherine Bouchard; Andréanne Deschênes; Vincent Boulanger; Jean-Michel Bellavance; Julia Chabbert; Alexy Pelletier-Rioux; Flavie Lavoie-Cardinal; Christian Gagné
>
> **备注:** 42 pages, 9 figures (main paper) + 22 pages, 15 figures (supplementary material)
>
> **摘要:** The development of signal unmixing algorithms is essential for leveraging multimodal datasets acquired through a wide array of scientific imaging technologies, including hyperspectral or time-resolved acquisitions. In experimental physics, enhancing the spatio-temporal resolution or expanding the number of detection channels often leads to diminished sampling rate and signal-to-noise ratio, significantly affecting the efficacy of signal unmixing algorithms. We propose Latent Unmixing, a new approach which applies bandpass filters to the latent space of a multidimensional convolutional neural network to disentangle overlapping signal components. It enables better isolation and quantification of individual signal contributions, especially in the context of undersampled distributions. Using multidimensional convolution kernels to process all dimensions simultaneously enhances the network's ability to extract information from adjacent pixels, and time or spectral bins. This approach enables more effective separation of components in cases where individual pixels do not provide clear, well-resolved information. We showcase the method's practical use in experimental physics through two test cases that highlight the versatility of our approach: fluorescence lifetime microscopy and mode decomposition in optical fibers. The latent unmixing method extracts valuable information from complex signals that cannot be resolved by standard methods. It opens up new possibilities in optics and photonics for multichannel separation at an increased sampling rate.
>
---
#### [replaced 002] Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14359v4](http://arxiv.org/pdf/2505.14359v4)**

> **作者:** Ruoxin Chen; Junwei Xi; Zhiyuan Yan; Ke-Yue Zhang; Shuang Wu; Jingyi Xie; Xu Chen; Lei Xu; Isabel Guan; Taiping Yao; Shouhong Ding
>
> **备注:** 12 Pages, 9 figures
>
> **摘要:** Existing detectors are often trained on biased datasets, leading to the possibility of overfitting on non-causal image attributes that are spuriously correlated with real/synthetic labels. While these biased features enhance performance on the training data, they result in substantial performance degradation when applied to unbiased datasets. One common solution is to perform dataset alignment through generative reconstruction, matching the semantic content between real and synthetic images. However, we revisit this approach and show that pixel-level alignment alone is insufficient. The reconstructed images still suffer from frequency-level misalignment, which can perpetuate spurious correlations. To illustrate, we observe that reconstruction models tend to restore the high-frequency details lost in real images (possibly due to JPEG compression), inadvertently creating a frequency-level misalignment, where synthetic images appear to have richer high-frequency content than real ones. This misalignment leads to models associating high-frequency features with synthetic labels, further reinforcing biased cues. To resolve this, we propose Dual Data Alignment (DDA), which aligns both the pixel and frequency domains. Moreover, we introduce two new test sets: DDA-COCO, containing DDA-aligned synthetic images for testing detector performance on the most aligned dataset, and EvalGEN, featuring the latest generative models for assessing detectors under new generative architectures such as visual auto-regressive generators. Finally, our extensive evaluations demonstrate that a detector trained exclusively on DDA-aligned MSCOCO could improve across 8 diverse benchmarks by a non-trivial margin, showing a +7.2% on in-the-wild benchmarks, highlighting the improved generalizability of unbiased detectors.
>
---
#### [replaced 003] Brain Latent Progression: Individual-based Spatiotemporal Disease Progression on 3D Brain MRIs via Latent Diffusion
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.08560v2](http://arxiv.org/pdf/2502.08560v2)**

> **作者:** Lemuel Puglisi; Daniel C. Alexander; Daniele Ravì
>
> **备注:** arXiv admin note: text overlap with arXiv:2405.03328
>
> **摘要:** The growing availability of longitudinal Magnetic Resonance Imaging (MRI) datasets has facilitated Artificial Intelligence (AI)-driven modeling of disease progression, making it possible to predict future medical scans for individual patients. However, despite significant advancements in AI, current methods continue to face challenges including achieving patient-specific individualization, ensuring spatiotemporal consistency, efficiently utilizing longitudinal data, and managing the substantial memory demands of 3D scans. To address these challenges, we propose Brain Latent Progression (BrLP), a novel spatiotemporal model designed to predict individual-level disease progression in 3D brain MRIs. The key contributions in BrLP are fourfold: (i) it operates in a small latent space, mitigating the computational challenges posed by high-dimensional imaging data; (ii) it explicitly integrates subject metadata to enhance the individualization of predictions; (iii) it incorporates prior knowledge of disease dynamics through an auxiliary model, facilitating the integration of longitudinal data; and (iv) it introduces the Latent Average Stabilization (LAS) algorithm, which (a) enforces spatiotemporal consistency in the predicted progression at inference time and (b) allows us to derive a measure of the uncertainty for the prediction at the global and voxel level. We train and evaluate BrLP on 11,730 T1-weighted (T1w) brain MRIs from 2,805 subjects and validate its generalizability on an external test set comprising 2,257 MRIs from 962 subjects. Our experiments compare BrLP-generated MRI scans with real follow-up MRIs, demonstrating state-of-the-art accuracy compared to existing methods. The code is publicly available at: https://github.com/LemuelPuglisi/BrLP.
>
---
#### [replaced 004] AdaAugment: A Tuning-Free and Adaptive Approach to Enhance Data Augmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.11467v3](http://arxiv.org/pdf/2405.11467v3)**

> **作者:** Suorong Yang; Peijia Li; Xin Xiong; Furao Shen; Jian Zhao
>
> **备注:** IEEE Transactions on Image Processing
>
> **摘要:** Data augmentation (DA) is widely employed to improve the generalization performance of deep models. However, most existing DA methods employ augmentation operations with fixed or random magnitudes throughout the training process. While this fosters data diversity, it can also inevitably introduce uncontrolled variability in augmented data, which could potentially cause misalignment with the evolving training status of the target models. Both theoretical and empirical findings suggest that this misalignment increases the risks of both underfitting and overfitting. To address these limitations, we propose AdaAugment, an innovative and tuning-free adaptive augmentation method that leverages reinforcement learning to dynamically and adaptively adjust augmentation magnitudes for individual training samples based on real-time feedback from the target network. Specifically, AdaAugment features a dual-model architecture consisting of a policy network and a target network, which are jointly optimized to adapt augmentation magnitudes in accordance with the model's training progress effectively. The policy network optimizes the variability within the augmented data, while the target network utilizes the adaptively augmented samples for training. These two networks are jointly optimized and mutually reinforce each other. Extensive experiments across benchmark datasets and deep architectures demonstrate that AdaAugment consistently outperforms other state-of-the-art DA methods in effectiveness while maintaining remarkable efficiency. Code is available at https://github.com/Jackbrocp/AdaAugment.
>
---
#### [replaced 005] PrefixKV: Adaptive Prefix KV Cache is What Vision Instruction-Following Models Need for Efficient Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.03409v3](http://arxiv.org/pdf/2412.03409v3)**

> **作者:** Ao Wang; Hui Chen; Jiaxin Li; Jianchao Tan; Kefeng Zhang; Xunliang Cai; Zijia Lin; Jungong Han; Guiguang Ding
>
> **备注:** 12 pages, 5 figures;
>
> **摘要:** Recently, large vision-language models (LVLMs) have rapidly gained popularity for their strong generation and reasoning capabilities given diverse multimodal inputs. However, these models incur significant computational and memory overhead during inference, which greatly hinders the efficient deployment in practical scenarios. The extensive key-value (KV) cache, necessitated by the lengthy input and output sequences, notably contributes to the high inference cost. Based on this, recent works have investigated ways to reduce the KV cache size for higher efficiency. Although effective, they generally overlook the distinct importance distributions of KV vectors across layers and maintain the same cache size for each layer during the next token prediction. This results in the significant contextual information loss for certain layers, leading to notable performance decline. To address this, we present PrefixKV. It reframes the challenge of determining KV cache sizes for all layers into the task of searching for the optimal global prefix configuration. With an adaptive layer-wise KV retention recipe based on binary search, the maximum contextual information can thus be preserved in each layer, facilitating the generation. Extensive experiments demonstrate that our method achieves the state-of-the-art performance compared with others. It exhibits superior inference efficiency and generation quality trade-offs, showing promising potential for practical applications. Code is available at https://github.com/THU-MIG/PrefixKV.
>
---
#### [replaced 006] Following the Clues: Experiments on Person Re-ID using Cross-Modal Intelligence
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.01504v2](http://arxiv.org/pdf/2507.01504v2)**

> **作者:** Robert Aufschläger; Youssef Shoeb; Azarm Nowzad; Michael Heigl; Fabian Bally; Martin Schramm
>
> **备注:** accepted for publication at the 2025 IEEE 28th International Conference on Intelligent Transportation Systems (ITSC 2025), taking place during November 18-21, 2025 in Gold Coast, Australia
>
> **摘要:** The collection and release of street-level recordings as Open Data play a vital role in advancing autonomous driving systems and AI research. However, these datasets pose significant privacy risks, particularly for pedestrians, due to the presence of Personally Identifiable Information (PII) that extends beyond biometric traits such as faces. In this paper, we present cRID, a novel cross-modal framework combining Large Vision-Language Models, Graph Attention Networks, and representation learning to detect textual describable clues of PII and enhance person re-identification (Re-ID). Our approach focuses on identifying and leveraging interpretable features, enabling the detection of semantically meaningful PII beyond low-level appearance cues. We conduct a systematic evaluation of PII presence in person image datasets. Our experiments show improved performance in practical cross-dataset Re-ID scenarios, notably from Market-1501 to CUHK03-np (detected), highlighting the framework's practical utility. Code is available at https://github.com/RAufschlaeger/cRID.
>
---
#### [replaced 007] VIVID-10M: A Dataset and Baseline for Versatile and Interactive Video Local Editing
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.15260v2](http://arxiv.org/pdf/2411.15260v2)**

> **作者:** Jiahao Hu; Tianxiong Zhong; Xuebo Wang; Boyuan Jiang; Xingye Tian; Fei Yang; Pengfei Wan; Di Zhang
>
> **备注:** 10 pages, 10 figures
>
> **摘要:** Diffusion-based image editing models have made remarkable progress in recent years. However, achieving high-quality video editing remains a significant challenge. One major hurdle is the absence of open-source, large-scale video editing datasets based on real-world data, as constructing such datasets is both time-consuming and costly. Moreover, video data requires a significantly larger number of tokens for representation, which substantially increases the training costs for video editing models. Lastly, current video editing models offer limited interactivity, often making it difficult for users to express their editing requirements effectively in a single attempt. To address these challenges, this paper introduces a dataset VIVID-10M and a baseline model VIVID. VIVID-10M is the first large-scale hybrid image-video local editing dataset aimed at reducing data construction and model training costs, which comprises 9.7M samples that encompass a wide range of video editing tasks. VIVID is a Versatile and Interactive VIdeo local eDiting model trained on VIVID-10M, which supports entity addition, modification, and deletion. At its core, a keyframe-guided interactive video editing mechanism is proposed, enabling users to iteratively edit keyframes and propagate it to other frames, thereby reducing latency in achieving desired outcomes. Extensive experimental evaluations show that our approach achieves state-of-the-art performance in video local editing, surpassing baseline methods in both automated metrics and user studies. The VIVID-10M dataset are open-sourced at https://kwaivgi.github.io/VIVID/.
>
---
#### [replaced 008] Self-Supervised Monocular 4D Scene Reconstruction for Egocentric Videos
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2411.09145v4](http://arxiv.org/pdf/2411.09145v4)**

> **作者:** Chengbo Yuan; Geng Chen; Li Yi; Yang Gao
>
> **摘要:** Egocentric videos provide valuable insights into human interactions with the physical world, which has sparked growing interest in the computer vision and robotics communities. A critical challenge in fully understanding the geometry and dynamics of egocentric videos is dense scene reconstruction. However, the lack of high-quality labeled datasets in this field has hindered the effectiveness of current supervised learning methods. In this work, we aim to address this issue by exploring an self-supervised dynamic scene reconstruction approach. We introduce EgoMono4D, a novel model that unifies the estimation of multiple variables necessary for Egocentric Monocular 4D reconstruction, including camera intrinsic, camera poses, and video depth, all within a fast feed-forward framework. Starting from pretrained single-frame depth and intrinsic estimation model, we extend it with camera poses estimation and align multi-frame results on large-scale unlabeled egocentric videos. We evaluate EgoMono4D in both in-domain and zero-shot generalization settings, achieving superior performance in dense pointclouds sequence reconstruction compared to all baselines. EgoMono4D represents the first attempt to apply self-supervised learning for pointclouds sequence reconstruction to the label-scarce egocentric field, enabling fast, dense, and generalizable reconstruction. The interactable visualization, code and trained models are released https://egomono4d.github.io/
>
---
#### [replaced 009] VGLD: Visually-Guided Linguistic Disambiguation for Monocular Depth Scale Recovery
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.02704v3](http://arxiv.org/pdf/2505.02704v3)**

> **作者:** Bojin Wu; Jing Chen
>
> **备注:** 19 pages, conference
>
> **摘要:** Monocular depth estimation can be broadly categorized into two directions: relative depth estimation, which predicts normalized or inverse depth without absolute scale, and metric depth estimation, which aims to recover depth with real-world scale. While relative methods are flexible and data-efficient, their lack of metric scale limits their utility in downstream tasks. A promising solution is to infer absolute scale from textual descriptions. However, such language-based recovery is highly sensitive to natural language ambiguity, as the same image may be described differently across perspectives and styles. To address this, we introduce VGLD (Visually-Guided Linguistic Disambiguation), a framework that incorporates high-level visual semantics to resolve ambiguity in textual inputs. By jointly encoding both image and text, VGLD predicts a set of global linear transformation parameters that align relative depth maps with metric scale. This visually grounded disambiguation improves the stability and accuracy of scale estimation. We evaluate VGLD on representative models, including MiDaS and DepthAnything, using standard indoor (NYUv2) and outdoor (KITTI) benchmarks. Results show that VGLD significantly mitigates scale estimation bias caused by inconsistent or ambiguous language, achieving robust and accurate metric predictions. Moreover, when trained on multiple datasets, VGLD functions as a universal and lightweight alignment module, maintaining strong performance even in zero-shot settings. Code will be released upon acceptance.
>
---
#### [replaced 010] A review of advancements in low-light image enhancement using deep learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.05759v2](http://arxiv.org/pdf/2505.05759v2)**

> **作者:** Fangxue Liu; Lei Fan
>
> **摘要:** In low-light environments, the performance of computer vision algorithms often deteriorates significantly, adversely affecting key vision tasks such as segmentation, detection, and classification. With the rapid advancement of deep learning, its application to low-light image processing has attracted widespread attention and seen significant progress in recent years. However, there remains a lack of comprehensive surveys that systematically examine how recent deep-learning-based low-light image enhancement methods function and evaluate their effectiveness in enhancing downstream vision tasks. To address this gap, this review provides detailed elaboration on how various recent approaches (from 2020) operate and their enhancement mechanisms, supplemented with clear illustrations. It also investigates the impact of different enhancement techniques on subsequent vision tasks, critically analyzing their strengths and limitations. Our review found that image enhancement improved the performance of downstream vision tasks to varying degrees. Although supervised methods often produced images with high perceptual quality, they typically produced modest improvements in vision tasks. In contrast, zero-shot learning, despite achieving lower scores in image quality metrics, showed consistently boosted performance across various vision tasks. These suggest a disconnect between image quality metrics and those evaluating vision task performance. Additionally, unsupervised domain adaptation techniques demonstrated significant gains in segmentation tasks, highlighting their potential in practical low-light scenarios where labelled data is scarce. Observed limitations of existing studies are analyzed, and directions for future research are proposed. This review serves as a useful reference for determining low-light image enhancement techniques and optimizing vision task performance in low-light conditions.
>
---
#### [replaced 011] On the Robustness Tradeoff in Fine-Tuning
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.14836v2](http://arxiv.org/pdf/2503.14836v2)**

> **作者:** Kunyang Li; Jean-Charles Noirot Ferrand; Ryan Sheatsley; Blaine Hoak; Yohan Beugin; Eric Pauley; Patrick McDaniel
>
> **备注:** Accepted to International Conference on Computer Vision, ICCV 2025
>
> **摘要:** Fine-tuning has become the standard practice for adapting pre-trained models to downstream tasks. However, the impact on model robustness is not well understood. In this work, we characterize the robustness-accuracy trade-off in fine-tuning. We evaluate the robustness and accuracy of fine-tuned models over 6 benchmark datasets and 7 different fine-tuning strategies. We observe a consistent trade-off between adversarial robustness and accuracy. Peripheral updates such as BitFit are more effective for simple tasks -- over 75% above the average measured by the area under the Pareto frontiers on CIFAR-10 and CIFAR-100. In contrast, fine-tuning information-heavy layers, such as attention layers via Compacter, achieves a better Pareto frontier on more complex tasks -- 57.5% and 34.6% above the average on Caltech-256 and CUB-200, respectively. Lastly, we observe that the robustness of fine-tuning against out-of-distribution data closely tracks accuracy. These insights emphasize the need for robustness-aware fine-tuning to ensure reliable real-world deployments.
>
---
#### [replaced 012] GI-NAS: Boosting Gradient Inversion Attacks through Adaptive Neural Architecture Search
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2405.20725v3](http://arxiv.org/pdf/2405.20725v3)**

> **作者:** Wenbo Yu; Hao Fang; Bin Chen; Xiaohang Sui; Chuan Chen; Hao Wu; Shu-Tao Xia; Ke Xu
>
> **备注:** accepted by IEEE Transactions on Information Forensics and Security (TIFS)
>
> **摘要:** Gradient Inversion Attacks invert the transmitted gradients in Federated Learning (FL) systems to reconstruct the sensitive data of local clients and have raised considerable privacy concerns. A majority of gradient inversion methods rely heavily on explicit prior knowledge (e.g., a well pre-trained generative model), which is often unavailable in realistic scenarios. This is because real-world client data distributions are often highly heterogeneous, domain-specific, and unavailable to attackers, making it impractical for attackers to obtain perfectly matched pre-trained models, which inevitably suffer from fundamental distribution shifts relative to target private data. To alleviate this issue, researchers have proposed to leverage the implicit prior knowledge of an over-parameterized network. However, they only utilize a fixed neural architecture for all the attack settings. This would hinder the adaptive use of implicit architectural priors and consequently limit the generalizability. In this paper, we further exploit such implicit prior knowledge by proposing Gradient Inversion via Neural Architecture Search (GI-NAS), which adaptively searches the network and captures the implicit priors behind neural architectures. Extensive experiments verify that our proposed GI-NAS can achieve superior attack performance compared to state-of-the-art gradient inversion methods, even under more practical settings with high-resolution images, large-sized batches, and advanced defense strategies. To the best of our knowledge, we are the first to successfully introduce NAS to the gradient inversion community. We believe that this work exposes critical vulnerabilities in real-world federated learning by demonstrating high-fidelity reconstruction of sensitive data without requiring domain-specific priors, forcing urgent reassessment of FL privacy safeguards.
>
---
#### [replaced 013] Enabling Advanced Land Cover Analytics: An Integrated Data Extraction Pipeline for Predictive Modeling with the Dynamic World Dataset
- **分类: cs.CV; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2410.09135v2](http://arxiv.org/pdf/2410.09135v2)**

> **作者:** Victor Radermecker; Andrea Zanon; Nancy Thomas; Annita Vapsi; Saba Rahimi; Rama Ramakrishnan; Daniel Borrajo
>
> **摘要:** Understanding land cover holds considerable potential for a myriad of practical applications, particularly as data accessibility transitions from being exclusive to governmental and commercial entities to now including the broader research community. Nevertheless, although the data is accessible to any community member interested in exploration, there exists a formidable learning curve and no standardized process for accessing, pre-processing, and leveraging the data for subsequent tasks. In this study, we democratize this data by presenting a flexible and efficient end to end pipeline for working with the Dynamic World dataset, a cutting-edge near-real-time land use/land cover (LULC) dataset. This includes a pre-processing and representation framework which tackles noise removal, efficient extraction of large amounts of data, and re-representation of LULC data in a format well suited for several downstream tasks. To demonstrate the power of our pipeline, we use it to extract data for an urbanization prediction problem and build a suite of machine learning models with excellent performance. This task is easily generalizable to the prediction of any type of land cover and our pipeline is also compatible with a series of other downstream tasks.
>
---
#### [replaced 014] SpaCE-10: A Comprehensive Benchmark for Multimodal Large Language Models in Compositional Spatial Intelligence
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.07966v2](http://arxiv.org/pdf/2506.07966v2)**

> **作者:** Ziyang Gong; Wenhao Li; Oliver Ma; Songyuan Li; Jiayi Ji; Xue Yang; Gen Luo; Junchi Yan; Rongrong Ji
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved remarkable progress in various multimodal tasks. To pursue higher intelligence in space, MLLMs require integrating multiple atomic spatial capabilities to handle complex and dynamic tasks. However, existing benchmarks struggle to comprehensively evaluate the spatial intelligence of common MLLMs from the atomic level to the compositional level. To fill this gap, we present SpaCE-10, a comprehensive benchmark for compositional spatial evaluations. In SpaCE-10, we define 10 atomic spatial capabilities, which are combined to form 8 compositional capabilities. Based on these definitions, we propose a novel hierarchical annotation pipeline to generate high-quality and diverse question-answer (QA) pairs. With over 150+ hours of human expert effort, we obtain over 5k QA pairs for 811 real indoor scenes in SpaCE-10, which covers various evaluation settings like point cloud input and multi-choice QA. We conduct an extensive evaluation of common MLLMs on SpaCE-10 and find that even the most advanced MLLM still lags behind humans by large margins. Through our careful study, we also draw several significant findings that benefit the MLLM community. For example, we reveal that the shortcoming of counting capability greatly limits the compositional spatial capabilities of existing MLLMs. The evaluation code and benchmark datasets are available at https://github.com/Cuzyoung/SpaCE-10.
>
---
#### [replaced 015] Unraveling the Connections between Flow Matching and Diffusion Probabilistic Models in Training-free Conditional Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.07625v2](http://arxiv.org/pdf/2411.07625v2)**

> **作者:** Kaiyu Song; Hanjiang Lai
>
> **摘要:** Training-free conditional generation based on flow matching aims to leverage pre-trained unconditional flow matching models to perform conditional generation without retraining. Recently, a successful training-free conditional generation approach incorporates conditions via posterior sampling, which relies on the availability of a score function in the unconditional diffusion model. However, flow matching models do not possess an explicit score function, rendering such a strategy inapplicable. Approximate posterior sampling for flow matching has been explored, but it is limited to linear inverse problems. In this paper, we propose Flow Matching-based Posterior Sampling (FMPS) to expand its application scope. We introduce a correction term by steering the velocity field. This correction term can be reformulated to incorporate a surrogate score function, thereby bridging the gap between flow matching models and score-based posterior sampling. Hence, FMPS enables the posterior sampling to be adjusted within the flow matching framework. Further, we propose two practical implementations of the correction mechanism: one aimed at improving generation quality, and the other focused on computational efficiency. Experimental results on diverse conditional generation tasks demonstrate that our method achieves superior generation quality compared to existing state-of-the-art approaches, validating the effectiveness and generality of FMPS.
>
---
#### [replaced 016] MSVD-Indonesian: A Benchmark for Multimodal Video-Text Tasks in Indonesian
- **分类: cs.MM; cs.CL; cs.CV; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2306.11341v2](http://arxiv.org/pdf/2306.11341v2)**

> **作者:** Willy Fitra Hendria
>
> **备注:** 10 pages, 5 figures, 5 tables
>
> **摘要:** Multimodal learning on video and text has seen significant progress, particularly in tasks like text-to-video retrieval, video-to-text retrieval, and video captioning. However, most existing methods and datasets focus exclusively on English. Despite Indonesian being one of the most widely spoken languages, multimodal research in Indonesian remains under-explored, largely due to the lack of benchmark datasets. To address this gap, we introduce the first public Indonesian video-text dataset by translating the English captions in the MSVD dataset into Indonesian. Using this dataset, we evaluate neural network models which were developed for the English video-text dataset on three tasks, i.e., text-to-video retrieval, video-to-text retrieval, and video captioning. Most existing models rely on feature extractors pretrained on English vision-language datasets, raising concerns about their applicability to Indonesian, given the scarcity of large-scale pretraining resources in the language. We apply a cross-lingual transfer learning approach by leveraging English-pretrained extractors and fine-tuning models on our Indonesian dataset. Experimental results demonstrate that this strategy improves performance across all tasks and metrics. We release our dataset publicly to support future research and hope it will inspire further progress in Indonesian multimodal learning.
>
---
#### [replaced 017] MG-Gen: Single Image to Motion Graphics Generation
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.02361v3](http://arxiv.org/pdf/2504.02361v3)**

> **作者:** Takahiro Shirakawa; Tomoyuki Suzuki; Takuto Narumoto; Daichi Haraguchi
>
> **摘要:** We introduce MG-Gen, a framework that generates motion graphics directly from a single raster image. MG-Gen decompose a single raster image into layered structures represented as HTML, generate animation scripts for each layer, and then render them into a video. Experiments confirm MG-Gen generates dynamic motion graphics while preserving text readability and fidelity to the input conditions, whereas state-of-the-art image-to-video generation methods struggle with them. The code is available at https://github.com/CyberAgentAILab/MG-GEN.
>
---
#### [replaced 018] ECORE: Energy-Conscious Optimized Routing for Deep Learning Models at the Edge
- **分类: cs.DC; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.06011v2](http://arxiv.org/pdf/2507.06011v2)**

> **作者:** Daghash K. Alqahtani; Maria A. Rodriguez; Muhammad Aamir Cheema; Hamid Rezatofighi; Adel N. Toosi
>
> **摘要:** Edge computing enables data processing closer to the source, significantly reducing latency an essential requirement for real-time vision-based analytics such as object detection in surveillance and smart city environments. However, these tasks place substantial demands on resource constrained edge devices, making the joint optimization of energy consumption and detection accuracy critical. To address this challenge, we propose ECORE, a framework that integrates multiple dynamic routing strategies including estimation based techniques and a greedy selection algorithm to direct image processing requests to the most suitable edge device-model pair. ECORE dynamically balances energy efficiency and detection performance based on object characteristics. We evaluate our approach through extensive experiments on real-world datasets, comparing the proposed routers against widely used baseline techniques. The evaluation leverages established object detection models (YOLO, SSD, EfficientDet) and diverse edge platforms, including Jetson Orin Nano, Raspberry Pi 4 and 5, and TPU accelerators. Results demonstrate that our proposed context-aware routing strategies can reduce energy consumption and latency by 45% and 49%, respectively, while incurring only a 2% loss in detection accuracy compared to accuracy-centric methods.
>
---
#### [replaced 019] Frenet-Serret Frame-based Decomposition for Part Segmentation of 3D Curvilinear Structures
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2404.14435v3](http://arxiv.org/pdf/2404.14435v3)**

> **作者:** Leslie Gu; Jason Ken Adhinarta; Mikhail Bessmeltsev; Jiancheng Yang; Yongjie Jessica Zhang; Wenjie Yin; Daniel Berger; Jeff Lichtman; Hanspeter Pfister; Donglai Wei
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Accurately segmenting 3D curvilinear structures in medical imaging remains challenging due to their complex geometry and the scarcity of diverse, large-scale datasets for algorithm development and evaluation. In this paper, we use dendritic spine segmentation as a case study and address these challenges by introducing a novel Frenet--Serret Frame-based Decomposition, which decomposes 3D curvilinear structures into a globally \( C^2 \) continuous curve that captures the overall shape, and a cylindrical primitive that encodes local geometric properties. This approach leverages Frenet--Serret Frames and arc length parameterization to preserve essential geometric features while reducing representational complexity, facilitating data-efficient learning, improved segmentation accuracy, and generalization on 3D curvilinear structures. To rigorously evaluate our method, we introduce two datasets: CurviSeg, a synthetic dataset for 3D curvilinear structure segmentation that validates our method's key properties, and DenSpineEM, a benchmark for dendritic spine segmentation, which comprises 4,476 manually annotated spines from 70 dendrites across three public electron microscopy datasets, covering multiple brain regions and species. Our experiments on DenSpineEM demonstrate exceptional cross-region and cross-species generalization: models trained on the mouse somatosensory cortex subset achieve 91.9\% Dice, maintaining strong performance in zero-shot segmentation on both mouse visual cortex (94.1\% Dice) and human frontal lobe (81.8\% Dice) subsets. Moreover, we test the generalizability of our method on the IntrA dataset, where it achieves 77.08\% Dice (5.29\% higher than prior arts) on intracranial aneurysm segmentation. These findings demonstrate the potential of our approach for accurately analyzing complex curvilinear structures across diverse medical imaging fields.
>
---
#### [replaced 020] From Video to EEG: Adapting Joint Embedding Predictive Architecture to Uncover Visual Concepts in Brain Signal Analysis
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.03633v4](http://arxiv.org/pdf/2507.03633v4)**

> **作者:** Amirabbas Hojjati; Lu Li; Ibrahim Hameed; Anis Yazidi; Pedro G. Lind; Rabindra Khadka
>
> **摘要:** EEG signals capture brain activity with high temporal and low spatial resolution, supporting applications such as neurological diagnosis, cognitive monitoring, and brain-computer interfaces. However, effective analysis is hindered by limited labeled data, high dimensionality, and the absence of scalable models that fully capture spatiotemporal dependencies. Existing self-supervised learning (SSL) methods often focus on either spatial or temporal features, leading to suboptimal representations. To this end, we propose EEG-VJEPA, a novel adaptation of the Video Joint Embedding Predictive Architecture (V-JEPA) for EEG classification. By treating EEG as video-like sequences, EEG-VJEPA learns semantically meaningful spatiotemporal representations using joint embeddings and adaptive masking. To our knowledge, this is the first work that exploits V-JEPA for EEG classification and explores the visual concepts learned by the model. Evaluations on the publicly available Temple University Hospital (TUH) Abnormal EEG dataset show that EEG-VJEPA outperforms existing state-of-the-art models in classification accuracy. Beyond classification accuracy, EEG-VJEPA captures physiologically relevant spatial and temporal signal patterns, offering interpretable embeddings that may support human-AI collaboration in diagnostic workflows. These findings position EEG-VJEPA as a promising framework for scalable, trustworthy EEG analysis in real-world clinical settings.
>
---
#### [replaced 021] BreastDCEDL: A Comprehensive Breast Cancer DCE-MRI Dataset and Transformer Implementation for Treatment Response Prediction
- **分类: cs.CV; cs.AI; 68T07, 68U10, 92C55; I.2.0; I.2.10; I.4.5; J.3**

- **链接: [http://arxiv.org/pdf/2506.12190v3](http://arxiv.org/pdf/2506.12190v3)**

> **作者:** Naomi Fridman; Bubby Solway; Tomer Fridman; Itamar Barnea; Anat Goldstein
>
> **摘要:** Breast cancer remains a leading cause of cancer-related mortality worldwide, making early detection and accurate treatment response monitoring critical priorities. We present BreastDCEDL, a curated, deep learning-ready dataset comprising pre-treatment 3D Dynamic Contrast-Enhanced MRI (DCE-MRI) scans from 2,070 breast cancer patients drawn from the I-SPY1, I-SPY2, and Duke cohorts, all sourced from The Cancer Imaging Archive. The raw DICOM imaging data were rigorously converted into standardized 3D NIfTI volumes with preserved signal integrity, accompanied by unified tumor annotations and harmonized clinical metadata including pathologic complete response (pCR), hormone receptor (HR), and HER2 status. Although DCE-MRI provides essential diagnostic information and deep learning offers tremendous potential for analyzing such complex data, progress has been limited by lack of accessible, public, multicenter datasets. BreastDCEDL addresses this gap by enabling development of advanced models, including state-of-the-art transformer architectures that require substantial training data. To demonstrate its capacity for robust modeling, we developed the first transformer-based model for breast DCE-MRI, leveraging Vision Transformer (ViT) architecture trained on RGB-fused images from three contrast phases (pre-contrast, early post-contrast, and late post-contrast). Our ViT model achieved state-of-the-art pCR prediction performance in HR+/HER2- patients (AUC 0.94, accuracy 0.93). BreastDCEDL includes predefined benchmark splits, offering a framework for reproducible research and enabling clinically meaningful modeling in breast cancer imaging.
>
---
#### [replaced 022] InternVideo2.5: Empowering Video MLLMs with Long and Rich Context Modeling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.12386v3](http://arxiv.org/pdf/2501.12386v3)**

> **作者:** Yi Wang; Xinhao Li; Ziang Yan; Yinan He; Jiashuo Yu; Xiangyu Zeng; Chenting Wang; Changlian Ma; Haian Huang; Jianfei Gao; Min Dou; Kai Chen; Wenhai Wang; Yu Qiao; Yali Wang; Limin Wang
>
> **备注:** technical report
>
> **摘要:** This paper aims to improve the performance of video multimodal large language models (MLLM) via long and rich context (LRC) modeling. As a result, we develop a new version of InternVideo2.5 with a focus on enhancing the original MLLMs' ability to perceive fine-grained details and capture long-form temporal structure in videos. Specifically, our approach incorporates dense vision task annotations into MLLMs using direct preference optimization and develops compact spatiotemporal representations through adaptive hierarchical token compression. Experimental results demonstrate this unique design of LRC greatly improves the results of video MLLM in mainstream video understanding benchmarks (short & long), enabling the MLLM to memorize significantly longer video inputs (at least 6x longer than the original), and master specialized vision capabilities like object tracking and segmentation. Our work highlights the importance of multimodal context richness (length and fineness) in empowering MLLM's innate abilites (focus and memory), providing new insights for future research on video MLLM. Code and models are available at https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2.5
>
---
#### [replaced 023] Relation-aware Hierarchical Prompt for Open-vocabulary Scene Graph Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.19021v2](http://arxiv.org/pdf/2412.19021v2)**

> **作者:** Tao Liu; Rongjie Li; Chongyu Wang; Xuming He
>
> **备注:** Accepted by AAAI-25
>
> **摘要:** Open-vocabulary Scene Graph Generation (OV-SGG) overcomes the limitations of the closed-set assumption by aligning visual relationship representations with open-vocabulary textual representations. This enables the identification of novel visual relationships, making it applicable to real-world scenarios with diverse relationships. However, existing OV-SGG methods are constrained by fixed text representations, limiting diversity and accuracy in image-text alignment. To address these challenges, we propose the Relation-Aware Hierarchical Prompting (RAHP) framework, which enhances text representation by integrating subject-object and region-specific relation information. Our approach utilizes entity clustering to address the complexity of relation triplet categories, enabling the effective integration of subject-object information. Additionally, we utilize a large language model (LLM) to generate detailed region-aware prompts, capturing fine-grained visual interactions and improving alignment between visual and textual modalities. RAHP also introduces a dynamic selection mechanism within Vision-Language Models (VLMs), which adaptively selects relevant text prompts based on the visual content, reducing noise from irrelevant prompts. Extensive experiments on the Visual Genome and Open Images v6 datasets demonstrate that our framework consistently achieves state-of-the-art performance, demonstrating its effectiveness in addressing the challenges of open-vocabulary scene graph generation. The code is available at: https://github.com/Leon022/RAHP
>
---
#### [replaced 024] LLM-enhanced Action-aware Multi-modal Prompt Tuning for Image-Text Matching
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.23502v2](http://arxiv.org/pdf/2506.23502v2)**

> **作者:** Mengxiao Tian; Xinxiao Wu; Shuo Yang
>
> **备注:** accepted by ICCV 2025
>
> **摘要:** Driven by large-scale contrastive vision-language pre-trained models such as CLIP, recent advancements in the image-text matching task have achieved remarkable success in representation learning. Due to image-level visual-language alignment, CLIP falls short in understanding fine-grained details such as object attributes and spatial relationships between objects. Recent efforts have attempted to compel CLIP to acquire structured visual representations by introducing prompt learning to achieve object-level alignment. While achieving promising results, they still lack the capability to perceive actions, which are crucial for describing the states or relationships between objects. Therefore, we propose to endow CLIP with fine-grained action-level understanding by introducing an LLM-enhanced action-aware multi-modal prompt-tuning method, incorporating the action-related external knowledge generated by large language models (LLMs). Specifically, we design an action triplet prompt and an action state prompt to exploit compositional semantic knowledge and state-related causal knowledge implicitly stored in LLMs. Subsequently, we propose an adaptive interaction module to aggregate attentive visual features conditioned on action-aware prompted knowledge for establishing discriminative and action-aware visual representations, which further improves the performance. Comprehensive experimental results on two benchmark datasets demonstrate the effectiveness of our method.
>
---
#### [replaced 025] EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.10061v2](http://arxiv.org/pdf/2411.10061v2)**

> **作者:** Rang Meng; Xingyu Zhang; Yuming Li; Chenguang Ma
>
> **备注:** CVPR2025
>
> **摘要:** Recent work on human animation usually involves audio, pose, or movement maps conditions, thereby achieves vivid animation quality. However, these methods often face practical challenges due to extra control conditions, cumbersome condition injection modules, or limitation to head region driving. Hence, we ask if it is possible to achieve striking half-body human animation while simplifying unnecessary conditions. To this end, we propose a half-body human animation method, dubbed EchoMimicV2, that leverages a novel Audio-Pose Dynamic Harmonization strategy, including Pose Sampling and Audio Diffusion, to enhance half-body details, facial and gestural expressiveness, and meanwhile reduce conditions redundancy. To compensate for the scarcity of half-body data, we utilize Head Partial Attention to seamlessly accommodate headshot data into our training framework, which can be omitted during inference, providing a free lunch for animation. Furthermore, we design the Phase-specific Denoising Loss to guide motion, detail, and low-level quality for animation in specific phases, respectively. Besides, we also present a novel benchmark for evaluating the effectiveness of half-body human animation. Extensive experiments and analyses demonstrate that EchoMimicV2 surpasses existing methods in both quantitative and qualitative evaluations.
>
---
#### [replaced 026] Class-Aware PillarMix: Can Mixed Sample Data Augmentation Enhance 3D Object Detection with Radar Point Clouds?
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.02687v2](http://arxiv.org/pdf/2503.02687v2)**

> **作者:** Miao Zhang; Sherif Abdulatif; Benedikt Loesch; Marco Altmann; Bin Yang
>
> **备注:** 8 pages, 6 figures, 4 tables, accepted to 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** Due to the significant effort required for data collection and annotation in 3D perception tasks, mixed sample data augmentation (MSDA) has been widely studied to generate diverse training samples by mixing existing data. Recently, many MSDA techniques have been developed for point clouds, but they mainly target LiDAR data, leaving their application to radar point clouds largely unexplored. In this paper, we examine the feasibility of applying existing MSDA methods to radar point clouds and identify several challenges in adapting these techniques. These obstacles stem from the radar's irregular angular distribution, deviations from a single-sensor polar layout in multi-radar setups, and point sparsity. To address these issues, we propose Class-Aware PillarMix (CAPMix), a novel MSDA approach that applies MixUp at the pillar level in 3D point clouds, guided by class labels. Unlike methods that rely a single mix ratio to the entire sample, CAPMix assigns an independent ratio to each pillar, boosting sample diversity. To account for the density of different classes, we use class-specific distributions: for dense objects (e.g., large vehicles), we skew ratios to favor points from another sample, while for sparse objects (e.g., pedestrians), we sample more points from the original. This class-aware mixing retains critical details and enriches each sample with new information, ultimately generating more diverse training data. Experimental results demonstrate that our method not only significantly boosts performance but also outperforms existing MSDA approaches across two datasets (Bosch Street and K-Radar). We believe that this straightforward yet effective approach will spark further investigation into MSDA techniques for radar data.
>
---
#### [replaced 027] Random Erasing vs. Model Inversion: A Promising Defense or a False Hope?
- **分类: cs.LG; cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.01062v2](http://arxiv.org/pdf/2409.01062v2)**

> **作者:** Viet-Hung Tran; Ngoc-Bao Nguyen; Son T. Mai; Hans Vandierendonck; Ira Assent; Alex Kot; Ngai-Man Cheung
>
> **备注:** Accepted in Transactions on Machine Learning Research (TMLR). First two authors contributed equally
>
> **摘要:** Model Inversion (MI) attacks pose a significant privacy threat by reconstructing private training data from machine learning models. While existing defenses primarily concentrate on model-centric approaches, the impact of data on MI robustness remains largely unexplored. In this work, we explore Random Erasing (RE), a technique traditionally used for improving model generalization under occlusion, and uncover its surprising effectiveness as a defense against MI attacks. Specifically, our novel feature space analysis shows that models trained with RE-images introduce a significant discrepancy between the features of MI-reconstructed images and those of the private data. At the same time, features of private images remain distinct from other classes and well-separated from different classification regions. These effects collectively degrade MI reconstruction quality and attack accuracy while maintaining reasonable natural accuracy. Furthermore, we explore two critical properties of RE including Partial Erasure and Random Location. Partial Erasure prevents the model from observing entire objects during training. We find this has a significant impact on MI, which aims to reconstruct the entire objects. Random Location of erasure plays a crucial role in achieving a strong privacy-utility trade-off. Our findings highlight RE as a simple yet effective defense mechanism that can be easily integrated with existing privacy-preserving techniques. Extensive experiments across 37 setups demonstrate that our method achieves state-of-the-art (SOTA) performance in the privacy-utility trade-off. The results consistently demonstrate the superiority of our defense over existing methods across different MI attacks, network architectures, and attack configurations. For the first time, we achieve a significant degradation in attack accuracy without a decrease in utility for some configurations.
>
---
#### [replaced 028] MedGemma Technical Report
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.05201v3](http://arxiv.org/pdf/2507.05201v3)**

> **作者:** Andrew Sellergren; Sahar Kazemzadeh; Tiam Jaroensri; Atilla Kiraly; Madeleine Traverse; Timo Kohlberger; Shawn Xu; Fayaz Jamil; Cían Hughes; Charles Lau; Justin Chen; Fereshteh Mahvar; Liron Yatziv; Tiffany Chen; Bram Sterling; Stefanie Anna Baby; Susanna Maria Baby; Jeremy Lai; Samuel Schmidgall; Lu Yang; Kejia Chen; Per Bjornsson; Shashir Reddy; Ryan Brush; Kenneth Philbrick; Mercy Asiedu; Ines Mezerreg; Howard Hu; Howard Yang; Richa Tiwari; Sunny Jansen; Preeti Singh; Yun Liu; Shekoofeh Azizi; Aishwarya Kamath; Johan Ferret; Shreya Pathak; Nino Vieillard; Ramona Merhej; Sarah Perrin; Tatiana Matejovicova; Alexandre Ramé; Morgane Riviere; Louis Rouillard; Thomas Mesnard; Geoffrey Cideron; Jean-bastien Grill; Sabela Ramos; Edouard Yvinec; Michelle Casbon; Elena Buchatskaya; Jean-Baptiste Alayrac; Dmitry Lepikhin; Vlad Feinberg; Sebastian Borgeaud; Alek Andreev; Cassidy Hardin; Robert Dadashi; Léonard Hussenot; Armand Joulin; Olivier Bachem; Yossi Matias; Katherine Chou; Avinatan Hassidim; Kavi Goel; Clement Farabet; Joelle Barral; Tris Warkentin; Jonathon Shlens; David Fleet; Victor Cotruta; Omar Sanseviero; Gus Martins; Phoebe Kirk; Anand Rao; Shravya Shetty; David F. Steiner; Can Kirmizibayrak; Rory Pilgrim; Daniel Golden; Lin Yang
>
> **摘要:** Artificial intelligence (AI) has significant potential in healthcare applications, but its training and deployment faces challenges due to healthcare's diverse data, complex tasks, and the need to preserve privacy. Foundation models that perform well on medical tasks and require less task-specific tuning data are critical to accelerate the development of healthcare AI applications. We introduce MedGemma, a collection of medical vision-language foundation models based on Gemma 3 4B and 27B. MedGemma demonstrates advanced medical understanding and reasoning on images and text, significantly exceeding the performance of similar-sized generative models and approaching the performance of task-specific models, while maintaining the general capabilities of the Gemma 3 base models. For out-of-distribution tasks, MedGemma achieves 2.6-10% improvement on medical multimodal question answering, 15.5-18.1% improvement on chest X-ray finding classification, and 10.8% improvement on agentic evaluations compared to the base models. Fine-tuning MedGemma further improves performance in subdomains, reducing errors in electronic health record information retrieval by 50% and reaching comparable performance to existing specialized state-of-the-art methods for pneumothorax classification and histopathology patch classification. We additionally introduce MedSigLIP, a medically-tuned vision encoder derived from SigLIP. MedSigLIP powers the visual understanding capabilities of MedGemma and as an encoder achieves comparable or better performance than specialized medical image encoders. Taken together, the MedGemma collection provides a strong foundation of medical image and text capabilities, with potential to significantly accelerate medical research and development of downstream applications. The MedGemma collection, including tutorials and model weights, can be found at https://goo.gle/medgemma.
>
---
#### [replaced 029] HGSLoc: 3DGS-based Heuristic Camera Pose Refinement
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.10925v3](http://arxiv.org/pdf/2409.10925v3)**

> **作者:** Zhongyan Niu; Zhen Tan; Jinpu Zhang; Xueliang Yang; Dewen Hu
>
> **摘要:** Visual localization refers to the process of determining camera poses and orientation within a known scene representation. This task is often complicated by factors such as changes in illumination and variations in viewing angles. In this paper, we propose HGSLoc, a novel lightweight plug-and-play pose optimization framework, which integrates 3D reconstruction with a heuristic refinement strategy to achieve higher pose estimation accuracy. Specifically, we introduce an explicit geometric map for 3D representation and high-fidelity rendering, allowing the generation of high-quality synthesized views to support accurate visual localization. Our method demonstrates higher localization accuracy compared to NeRF-based neural rendering localization approaches. We introduce a heuristic refinement strategy, its efficient optimization capability can quickly locate the target node, while we set the step level optimization step to enhance the pose accuracy in the scenarios with small errors. With carefully designed heuristic functions, it offers efficient optimization capabilities, enabling rapid error reduction in rough localization estimations. Our method mitigates the dependence on complex neural network models while demonstrating improved robustness against noise and higher localization accuracy in challenging environments, as compared to neural network joint optimization strategies. The optimization framework proposed in this paper introduces novel approaches to visual localization by integrating the advantages of 3D reconstruction and the heuristic refinement strategy, which demonstrates strong performance across multiple benchmark datasets, including 7Scenes and Deep Blending dataset. The implementation of our method has been released at https://github.com/anchang699/HGSLoc.
>
---
#### [replaced 030] Barriers in Integrating Medical Visual Question Answering into Radiology Workflows: A Scoping Review and Clinicians' Insights
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.08036v2](http://arxiv.org/pdf/2507.08036v2)**

> **作者:** Deepali Mishra; Chaklam Silpasuwanchai; Ashutosh Modi; Madhumita Sushil; Sorayouth Chumnanvej
>
> **备注:** 29 pages, 5 figures (1 in supplementary), 3 tables (1 in main text, 2 in supplementary). Scoping review and clinician survey
>
> **摘要:** Medical Visual Question Answering (MedVQA) is a promising tool to assist radiologists by automating medical image interpretation through question answering. Despite advances in models and datasets, MedVQA's integration into clinical workflows remains limited. This study systematically reviews 68 publications (2018-2024) and surveys 50 clinicians from India and Thailand to examine MedVQA's practical utility, challenges, and gaps. Following the Arksey and O'Malley scoping review framework, we used a two-pronged approach: (1) reviewing studies to identify key concepts, advancements, and research gaps in radiology workflows, and (2) surveying clinicians to capture their perspectives on MedVQA's clinical relevance. Our review reveals that nearly 60% of QA pairs are non-diagnostic and lack clinical relevance. Most datasets and models do not support multi-view, multi-resolution imaging, EHR integration, or domain knowledge, features essential for clinical diagnosis. Furthermore, there is a clear mismatch between current evaluation metrics and clinical needs. The clinician survey confirms this disconnect: only 29.8% consider MedVQA systems highly useful. Key concerns include the absence of patient history or domain knowledge (87.2%), preference for manually curated datasets (51.1%), and the need for multi-view image support (78.7%). Additionally, 66% favor models focused on specific anatomical regions, and 89.4% prefer dialogue-based interactive systems. While MedVQA shows strong potential, challenges such as limited multimodal analysis, lack of patient context, and misaligned evaluation approaches must be addressed for effective clinical integration.
>
---
#### [replaced 031] Beyond Appearance: Geometric Cues for Robust Video Instance Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.05948v2](http://arxiv.org/pdf/2507.05948v2)**

> **作者:** Quanzhu Niu; Yikang Zhou; Shihao Chen; Tao Zhang; Shunping Ji
>
> **备注:** Accepted by ICCV 2025 Workshop LSVOS
>
> **摘要:** Video Instance Segmentation (VIS) fundamentally struggles with pervasive challenges including object occlusions, motion blur, and appearance variations during temporal association. To overcome these limitations, this work introduces geometric awareness to enhance VIS robustness by strategically leveraging monocular depth estimation. We systematically investigate three distinct integration paradigms. Expanding Depth Channel (EDC) method concatenates the depth map as input channel to segmentation networks; Sharing ViT (SV) designs a uniform ViT backbone, shared between depth estimation and segmentation branches; Depth Supervision (DS) makes use of depth prediction as an auxiliary training guide for feature learning. Though DS exhibits limited effectiveness, benchmark evaluations demonstrate that EDC and SV significantly enhance the robustness of VIS. When with Swin-L backbone, our EDC method gets 56.2 AP, which sets a new state-of-the-art result on OVIS benchmark. This work conclusively establishes depth cues as critical enablers for robust video understanding.
>
---
#### [replaced 032] Holistic White-light Polyp Classification via Alignment-free Dense Distillation of Auxiliary Optical Chromoendoscopy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19319v3](http://arxiv.org/pdf/2505.19319v3)**

> **作者:** Qiang Hu; Qimei Wang; Jia Chen; Xuantao Ji; Mei Liu; Qiang Li; Zhiwei Wang
>
> **备注:** Early Accepted by MICCAI 2025
>
> **摘要:** White Light Imaging (WLI) and Narrow Band Imaging (NBI) are the two main colonoscopic modalities for polyp classification. While NBI, as optical chromoendoscopy, offers valuable vascular details, WLI remains the most common and often the only available modality in resource-limited settings. However, WLI-based methods typically underperform, limiting their clinical applicability. Existing approaches transfer knowledge from NBI to WLI through global feature alignment but often rely on cropped lesion regions, which are susceptible to detection errors and neglect contextual and subtle diagnostic cues. To address this, this paper proposes a novel holistic classification framework that leverages full-image diagnosis without requiring polyp localization. The key innovation lies in the Alignment-free Dense Distillation (ADD) module, which enables fine-grained cross-domain knowledge distillation regardless of misalignment between WLI and NBI images. Without resorting to explicit image alignment, ADD learns pixel-wise cross-domain affinities to establish correspondences between feature maps, guiding the distillation along the most relevant pixel connections. To further enhance distillation reliability, ADD incorporates Class Activation Mapping (CAM) to filter cross-domain affinities, ensuring the distillation path connects only those semantically consistent regions with equal contributions to polyp diagnosis. Extensive results on public and in-house datasets show that our method achieves state-of-the-art performance, relatively outperforming the other approaches by at least 2.5% and 16.2% in AUC, respectively. Code is available at: https://github.com/Huster-Hq/ADD.
>
---
#### [replaced 033] COVID-19 Pneumonia Diagnosis Using Medical Images: Deep Learning-Based Transfer Learning Approach
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.12642v3](http://arxiv.org/pdf/2503.12642v3)**

> **作者:** Anjali Dharmik
>
> **摘要:** SARS-CoV-2, the causative agent of COVID-19, remains a global health concern due to its high transmissibility and evolving variants. Although vaccination efforts and therapeutic advancements have mitigated disease severity, emerging mutations continue to challenge diagnostics and containment strategies. As of mid-February 2025, global test positivity has risen to 11%, marking the highest level in over six months despite widespread immunization efforts. Newer variants demonstrate enhanced host cell binding, increasing both infectivity and diagnostic complexity. This study evaluates the effectiveness of deep transfer learning in delivering rapid, accurate, and mutation-resilient COVID-19 diagnosis from medical imaging, with a focus on scalability and accessibility. We developed an automated detection system using state-of-the-art CNNs, including VGG16, ResNet50, ConvNetXtTiny, MobileNet, NASNetMobile, and DenseNet121 among others, to detect COVID-19 from chest X-ray and CT images. Among all the models evaluated, DenseNet121 emerged as the best-performing architecture for COVID-19 diagnosis using CT and X-ray images. It achieved an impressive accuracy of 98%, with 96.9% precision, 98.9% recall, 97.9% F1-score and 99.8% AUC score, indicating a high degree of consistency and reliability in both detecting positive and negative cases. The confusion matrix showed minimal false positives and false negatives, underscoring the model's robustness in real-world diagnostic scenarios.
>
---
#### [replaced 034] Temporal Feature Matters: A Framework for Diffusion Model Quantization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.19547v4](http://arxiv.org/pdf/2407.19547v4)**

> **作者:** Yushi Huang; Ruihao Gong; Xianglong Liu; Jing Liu; Yuhang Li; Jiwen Lu; Dacheng Tao
>
> **备注:** Accepted by TPAMI 2025. arXiv admin note: substantial text overlap with arXiv:2311.16503
>
> **摘要:** The Diffusion models, widely used for image generation, face significant challenges related to their broad applicability due to prolonged inference times and high memory demands. Efficient Post-Training Quantization (PTQ) is crucial to address these issues. However, unlike traditional models, diffusion models critically rely on the time-step for the multi-round denoising. Typically, each time-step is encoded into a hypersensitive temporal feature by several modules. Despite this, existing PTQ methods do not optimize these modules individually. Instead, they employ unsuitable reconstruction objectives and complex calibration methods, leading to significant disturbances in the temporal feature and denoising trajectory, as well as reduced compression efficiency. To address these challenges, we introduce a novel quantization framework that includes three strategies: 1) TIB-based Maintenance: Based on our innovative Temporal Information Block (TIB) definition, Temporal Information-aware Reconstruction (TIAR) and Finite Set Calibration (FSC) are developed to efficiently align original temporal features. 2) Cache-based Maintenance: Instead of indirect and complex optimization for the related modules, pre-computing and caching quantized counterparts of temporal features are developed to minimize errors. 3) Disturbance-aware Selection: Employ temporal feature errors to guide a fine-grained selection between the two maintenance strategies for further disturbance reduction. This framework preserves most of the temporal information and ensures high-quality end-to-end generation. Extensive testing on various datasets, diffusion models and hardware confirms our superior performance and acceleration.
>
---
#### [replaced 035] Visual Test-time Scaling for GUI Agent Grounding
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.00684v2](http://arxiv.org/pdf/2505.00684v2)**

> **作者:** Tiange Luo; Lajanugen Logeswaran; Justin Johnson; Honglak Lee
>
> **备注:** ICCV2025, https://github.com/tiangeluo/RegionFocus
>
> **摘要:** We introduce RegionFocus, a visual test-time scaling approach for Vision Language Model Agents. Understanding webpages is challenging due to the visual complexity of GUI images and the large number of interface elements, making accurate action selection difficult. Our approach dynamically zooms in on relevant regions, reducing background clutter and improving grounding accuracy. To support this process, we propose an image-as-map mechanism that visualizes key landmarks at each step, providing a transparent action record and enables the agent to effectively choose among action candidates. Even with a simple region selection strategy, we observe significant performance gains of 28+\% on Screenspot-pro and 24+\% on WebVoyager benchmarks on top of two state-of-the-art open vision language model agents, UI-TARS and Qwen2.5-VL, highlighting the effectiveness of visual test-time scaling in interactive settings. We achieve a new state-of-the-art grounding performance of 61.6\% on the ScreenSpot-Pro benchmark by applying RegionFocus to a Qwen2.5-VL-72B model. Our code will be released publicly at https://github.com/tiangeluo/RegionFocus.
>
---
#### [replaced 036] M2DAO-Talker: Harmonizing Multi-granular Motion Decoupling and Alternating Optimization for Talking-head Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.08307v2](http://arxiv.org/pdf/2507.08307v2)**

> **作者:** Kui Jiang; Shiyu Liu; Junjun Jiang; Xin Yang; Hongxun Yao; Xiaopeng Fan
>
> **摘要:** Audio-driven talking head generation holds significant potential for film production. While existing 3D methods have advanced motion modeling and content synthesis, they often produce rendering artifacts, such as motion blur, temporal jitter, and local penetration, due to limitations in representing stable, fine-grained motion fields. Through systematic analysis, we reformulate talking head generation into a unified framework comprising three steps: video preprocessing, motion representation, and rendering reconstruction. This framework underpins our proposed M2DAO-Talker, which addresses current limitations via multi-granular motion decoupling and alternating optimization. Specifically, we devise a novel 2D portrait preprocessing pipeline to extract frame-wise deformation control conditions (motion region segmentation masks, and camera parameters) to facilitate motion representation. To ameliorate motion modeling, we elaborate a multi-granular motion decoupling strategy, which independently models non-rigid (oral and facial) and rigid (head) motions for improved reconstruction accuracy. Meanwhile, a motion consistency constraint is developed to ensure head-torso kinematic consistency, thereby mitigating penetration artifacts caused by motion aliasing. In addition, an alternating optimization strategy is designed to iteratively refine facial and oral motion parameters, enabling more realistic video generation. Experiments across multiple datasets show that M2DAO-Talker achieves state-of-the-art performance, with the 2.43 dB PSNR improvement in generation quality and 0.64 gain in user-evaluated video realness versus TalkingGaussian while with 150 FPS inference speed. Our project homepage is https://m2dao-talker.github.io/M2DAO-Talk.github.io.
>
---
#### [replaced 037] SEGS-SLAM: Structure-enhanced 3D Gaussian Splatting SLAM with Appearance Embedding
- **分类: cs.CV; 68T40(Primary)68T45, 68U99 (Secondary); I.4.8; I.3.7**

- **链接: [http://arxiv.org/pdf/2501.05242v3](http://arxiv.org/pdf/2501.05242v3)**

> **作者:** Tianci Wen; Zhiang Liu; Yongchun Fang
>
> **备注:** ICCV 2025 accept;code, video, demos, and project are available at Project page https://segs-slam.github.io/
>
> **摘要:** 3D Gaussian splatting (3D-GS) has recently revolutionized novel view synthesis in the simultaneous localization and mapping (SLAM) problem. However, most existing algorithms fail to fully capture the underlying structure, resulting in structural inconsistency. Additionally, they struggle with abrupt appearance variations, leading to inconsistent visual quality. To address these problems, we propose SEGS-SLAM, a structure-enhanced 3D Gaussian Splatting SLAM, which achieves high-quality photorealistic mapping. Our main contributions are two-fold. First, we propose a structure-enhanced photorealistic mapping (SEPM) framework that, for the first time, leverages highly structured point cloud to initialize structured 3D Gaussians, leading to significant improvements in rendering quality. Second, we propose Appearance-from-Motion embedding (AfME), enabling 3D Gaussians to better model image appearance variations across different camera poses. Extensive experiments on monocular, stereo, and RGB-D datasets demonstrate that SEGS-SLAM significantly outperforms state-of-the-art (SOTA) methods in photorealistic mapping quality, e.g., an improvement of $19.86\%$ in PSNR over MonoGS on the TUM RGB-D dataset for monocular cameras. The project page is available at https://segs-slam.github.io/.
>
---
#### [replaced 038] Towards a Universal Image Degradation Model via Content-Degradation Disentanglement
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2505.12860v2](http://arxiv.org/pdf/2505.12860v2)**

> **作者:** Wenbo Yang; Zhongling Wang; Zhou Wang
>
> **摘要:** Image degradation synthesis is highly desirable in a wide variety of applications ranging from image restoration to simulating artistic effects. Existing models are designed to generate one specific or a narrow set of degradations, which often require user-provided degradation parameters. As a result, they lack the generalizability to synthesize degradations beyond their initial design or adapt to other applications. Here we propose the first universal degradation model that can synthesize a broad spectrum of complex and realistic degradations containing both homogeneous (global) and inhomogeneous (spatially varying) components. Our model automatically extracts and disentangles homogeneous and inhomogeneous degradation features, which are later used for degradation synthesis without user intervention. A disentangle-by-compression method is proposed to separate degradation information from images. Two novel modules for extracting and incorporating inhomogeneous degradations are created to model inhomogeneous components in complex degradations. We demonstrate the model's accuracy and adaptability in film-grain simulation and blind image restoration tasks. The demo video, code, and dataset of this project will be released at github.com/yangwenbo99/content-degradation-disentanglement.
>
---
#### [replaced 039] Multispectral Detection Transformer with Infrared-Centric Feature Fusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15137v2](http://arxiv.org/pdf/2505.15137v2)**

> **作者:** Seongmin Hwang; Daeyoung Han; Moongu Jeon
>
> **备注:** Under Review
>
> **摘要:** Multispectral object detection aims to leverage complementary information from visible (RGB) and infrared (IR) modalities to enable robust performance under diverse environmental conditions. Our key insight, derived from wavelet analysis and empirical observations, is that IR images contain structurally rich high-frequency information critical for object detection, making an infrared-centric approach highly effective. To capitalize on this finding, we propose Infrared-Centric Fusion (IC-Fusion), a lightweight and modality-aware sensor fusion method that prioritizes infrared features while effectively integrating complementary RGB semantic context. IC-Fusion adopts a compact RGB backbone and designs a novel fusion module comprising a Multi-Scale Feature Distillation (MSFD) block to enhance RGB features and a three-stage fusion block with a Cross-Modal Channel Shuffle Gate (CCSG), a Cross-Modal Large Kernel Gate (CLKG), and a Channel Shuffle Projection (CSP) to facilitate effective cross-modal interaction. Experiments on the FLIR and LLVIP benchmarks demonstrate the superior effectiveness and efficiency of our IR-centric fusion strategy, further validating its benefits. Our code is available at https://github.com/smin-hwang/IC-Fusion.
>
---
#### [replaced 040] Easi3R: Estimating Disentangled Motion from DUSt3R Without Training
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.24391v2](http://arxiv.org/pdf/2503.24391v2)**

> **作者:** Xingyu Chen; Yue Chen; Yuliang Xiu; Andreas Geiger; Anpei Chen
>
> **备注:** Page: https://easi3r.github.io/ Code: https://github.com/Inception3D/Easi3R
>
> **摘要:** Recent advances in DUSt3R have enabled robust estimation of dense point clouds and camera parameters of static scenes, leveraging Transformer network architectures and direct supervision on large-scale 3D datasets. In contrast, the limited scale and diversity of available 4D datasets present a major bottleneck for training a highly generalizable 4D model. This constraint has driven conventional 4D methods to fine-tune 3D models on scalable dynamic video data with additional geometric priors such as optical flow and depths. In this work, we take an opposite path and introduce Easi3R, a simple yet efficient training-free method for 4D reconstruction. Our approach applies attention adaptation during inference, eliminating the need for from-scratch pre-training or network fine-tuning. We find that the attention layers in DUSt3R inherently encode rich information about camera and object motion. By carefully disentangling these attention maps, we achieve accurate dynamic region segmentation, camera pose estimation, and 4D dense point map reconstruction. Extensive experiments on real-world dynamic videos demonstrate that our lightweight attention adaptation significantly outperforms previous state-of-the-art methods that are trained or finetuned on extensive dynamic datasets. Our code is publicly available for research purpose at https://easi3r.github.io/
>
---
#### [replaced 041] De-Fake: Style based Anomaly Deepfake Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.03334v2](http://arxiv.org/pdf/2507.03334v2)**

> **作者:** Sudev Kumar Padhi; Harshit Kumar; Umesh Kashyap; Sk. Subidh Ali
>
> **摘要:** Detecting deepfakes involving face-swaps presents a significant challenge, particularly in real-world scenarios where anyone can perform face-swapping with freely available tools and apps without any technical knowledge. Existing deepfake detection methods rely on facial landmarks or inconsistencies in pixel-level features and often struggle with face-swap deepfakes, where the source face is seamlessly blended into the target image or video. The prevalence of face-swap is evident in everyday life, where it is used to spread false information, damage reputations, manipulate political opinions, create non-consensual intimate deepfakes (NCID), and exploit children by enabling the creation of child sexual abuse material (CSAM). Even prominent public figures are not immune to its impact, with numerous deepfakes of them circulating widely across social media platforms. Another challenge faced by deepfake detection methods is the creation of datasets that encompass a wide range of variations, as training models require substantial amounts of data. This raises privacy concerns, particularly regarding the processing and storage of personal facial data, which could lead to unauthorized access or misuse. Our key idea is to identify these style discrepancies to detect face-swapped images effectively without accessing the real facial image. We perform comprehensive evaluations using multiple datasets and face-swapping methods, which showcases the effectiveness of SafeVision in detecting face-swap deepfakes across diverse scenarios. SafeVision offers a reliable and scalable solution for detecting face-swaps in a privacy preserving manner, making it particularly effective in challenging real-world applications. To the best of our knowledge, SafeVision is the first deepfake detection using style features while providing inherent privacy protection.
>
---
#### [replaced 042] High-Quality Live Video Streaming via Transcoding Time Prediction and Preset Selection
- **分类: cs.MM; cs.CV**

- **链接: [http://arxiv.org/pdf/2312.05348v2](http://arxiv.org/pdf/2312.05348v2)**

> **作者:** Zahra Nabizadeh Shahre-Babak; Nader Karimi; Krishna Rapaka; Tarek Amara; Shadrokh Samavi; Shahram Shirani
>
> **备注:** After further review, we found major flaws in the paper that need extensive revision
>
> **摘要:** Video streaming often requires transcoding content into different resolutions and bitrates to match the recipient's internet speed and screen capabilities. Video encoders like x264 offer various presets, each with different tradeoffs between transcoding time and rate-distortion performance. Choosing the best preset for video transcoding is difficult, especially for live streaming, as trying all the presets and choosing the best one is not feasible. One solution is to predict each preset's transcoding time and select the preset that ensures the highest quality while adhering to live streaming time constraints. Prediction of video transcoding time is also critical in minimizing streaming delays, deploying resource management algorithms, and load balancing. We propose a learning-based framework for predicting the transcoding time of videos across various presets. Our predictor's features for video transcoding time prediction are derived directly from the ingested stream, primarily from the header or metadata. As a result, only minimal additional delay is incurred for feature extraction, rendering our approach ideal for live-streaming applications. We evaluated our learning-based transcoding time prediction using a dataset of videos. The results demonstrate that our framework can accurately predict the transcoding time for different presets, with a mean absolute percentage error (MAPE) of nearly 5.0%. Leveraging these predictions, we then select the most suitable transcoding preset for live video streaming. Utilizing our transcoding time prediction-based preset selection improved Peak Signal-to-Noise Ratio (PSNR) of up to 5 dB.
>
---
#### [replaced 043] Hear-Your-Click: Interactive Object-Specific Video-to-Audio Generation
- **分类: cs.CV; cs.AI; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.04959v2](http://arxiv.org/pdf/2507.04959v2)**

> **作者:** Yingshan Liang; Keyu Fan; Zhicheng Du; Yiran Wang; Qingyang Shi; Xinyu Zhang; Jiasheng Lu; Peiwu Qin
>
> **摘要:** Video-to-audio (V2A) generation shows great potential in fields such as film production. Despite significant advances, current V2A methods relying on global video information struggle with complex scenes and generating audio tailored to specific objects. To address these limitations, we introduce Hear-Your-Click, an interactive V2A framework enabling users to generate sounds for specific objects by clicking on the frame. To achieve this, we propose Object-aware Contrastive Audio-Visual Fine-tuning (OCAV) with a Mask-guided Visual Encoder (MVE) to obtain object-level visual features aligned with audio. Furthermore, we tailor two data augmentation strategies, Random Video Stitching (RVS) and Mask-guided Loudness Modulation (MLM), to enhance the model's sensitivity to segmented objects. To measure audio-visual correspondence, we designed a new evaluation metric, the CAV score. Extensive experiments demonstrate that our framework offers more precise control and improves generation performance across various metrics. Project Page: https://github.com/SynapGrid/Hear-Your-Click
>
---
#### [replaced 044] FieldNet: Efficient Real-Time Shadow Removal for Enhanced Vision in Field Robotics
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.08142v3](http://arxiv.org/pdf/2403.08142v3)**

> **作者:** Alzayat Saleh; Alex Olsen; Jake Wood; Bronson Philippa; Mostafa Rahimi Azghadi
>
> **备注:** 22 pages, 9 figures, 8 tables. Published at Expert Systems with Applications
>
> **摘要:** Shadows significantly hinder computer vision tasks in outdoor environments, particularly in field robotics, where varying lighting conditions complicate object detection and localisation. We present FieldNet, a novel deep learning framework for real-time shadow removal, optimised for resource-constrained hardware. FieldNet introduces a probabilistic enhancement module and a novel loss function to address challenges of inconsistent shadow boundary supervision and artefact generation, achieving enhanced accuracy and simplicity without requiring shadow masks during inference. Trained on a dataset of 10,000 natural images augmented with synthetic shadows, FieldNet outperforms state-of-the-art methods on benchmark datasets (ISTD, ISTD+, SRD), with up to $9$x speed improvements (66 FPS on Nvidia 2080Ti) and superior shadow removal quality (PSNR: 38.67, SSIM: 0.991). Real-world case studies in precision agriculture robotics demonstrate the practical impact of FieldNet in enhancing weed detection accuracy. These advancements establish FieldNet as a robust, efficient solution for real-time vision tasks in field robotics and beyond.
>
---
#### [replaced 045] Guided Neural Schrödinger bridge for Brain MR image synthesis with Limited Data
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.14171v2](http://arxiv.org/pdf/2501.14171v2)**

> **作者:** Hanyeol Yang; Sunggyu Kim; Mi Kyung Kim; Yongseon Yoo; Yu-Mi Kim; Min-Ho Shin; Insung Chung; Sang Baek Koh; Hyeon Chang Kim; Jong-Min Lee
>
> **备注:** Single column, 28 pages, 7 figures
>
> **摘要:** Multi-modal brain MRI provides essential complementary information for clinical diagnosis. However, acquiring all modalities in practice is often constrained by time and cost. To address this, various methods have been proposed to generate missing modalities from available ones. Traditional approaches can be broadly categorized into two main types: paired and unpaired methods. While paired methods for synthesizing missing modalities achieve high accuracy, obtaining large-scale paired datasets is typically impractical. In contrast, unpaired methods, though scalable, often fail to preserve critical anatomical features, such as lesions. In this paper, we propose Fully Guided Schr\"odinger Bridge (FGSB), a novel framework designed to overcome these limitations by enabling high-fidelity generation with extremely limited paired data. Furthermore, when provided with lesion-specific information such as expert annotations, segmentation tools, or simple intensity thresholds for critical regions, FGSB can generate missing modalities while preserving these significant lesion with reduced data requirements. Our model comprises two stages: 1) Generation Phase: Iteratively refines synthetic images using paired target image and Gaussian noise. Training Phase: Learns optimal transformation pathways from source to target modality by mapping all intermediate states, ensuring consistent and high-fidelity synthesis. Experimental results across multiple datasets demonstrate that FGSB achieved performance comparable to large-data-trained models, while using only two subjects. Incorporating lesion-specific priors further improves the preservation of clinical features.
>
---
#### [replaced 046] CRISP-SAM2: SAM2 with Cross-Modal Interaction and Semantic Prompting for Multi-Organ Segmentation
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.23121v3](http://arxiv.org/pdf/2506.23121v3)**

> **作者:** Xinlei Yu; Changmiao Wang; Hui Jin; Ahmed Elazab; Gangyong Jia; Xiang Wan; Changqing Zou; Ruiquan Ge
>
> **备注:** Accepted By ACMMM25
>
> **摘要:** Multi-organ medical segmentation is a crucial component of medical image processing, essential for doctors to make accurate diagnoses and develop effective treatment plans. Despite significant progress in this field, current multi-organ segmentation models often suffer from inaccurate details, dependence on geometric prompts and loss of spatial information. Addressing these challenges, we introduce a novel model named CRISP-SAM2 with CRoss-modal Interaction and Semantic Prompting based on SAM2. This model represents a promising approach to multi-organ medical segmentation guided by textual descriptions of organs. Our method begins by converting visual and textual inputs into cross-modal contextualized semantics using a progressive cross-attention interaction mechanism. These semantics are then injected into the image encoder to enhance the detailed understanding of visual information. To eliminate reliance on geometric prompts, we use a semantic prompting strategy, replacing the original prompt encoder to sharpen the perception of challenging targets. In addition, a similarity-sorting self-updating strategy for memory and a mask-refining process is applied to further adapt to medical imaging and enhance localized details. Comparative experiments conducted on seven public datasets indicate that CRISP-SAM2 outperforms existing models. Extensive analysis also demonstrates the effectiveness of our method, thereby confirming its superior performance, especially in addressing the limitations mentioned earlier. Our code is available at: https://github.com/YU-deep/CRISP_SAM2.git.
>
---
#### [replaced 047] Imagine for Me: Creative Conceptual Blending of Real Images and Text via Blended Attention
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.24085v2](http://arxiv.org/pdf/2506.24085v2)**

> **作者:** Wonwoong Cho; Yanxia Zhang; Yan-Ying Chen; David I. Inouye
>
> **备注:** Project website is available at https://imagineforme.github.io/
>
> **摘要:** Blending visual and textual concepts into a new visual concept is a unique and powerful trait of human beings that can fuel creativity. However, in practice, cross-modal conceptual blending for humans is prone to cognitive biases, like design fixation, which leads to local minima in the design space. In this paper, we propose a T2I diffusion adapter "IT-Blender" that can automate the blending process to enhance human creativity. Prior works related to cross-modal conceptual blending are limited in encoding a real image without loss of details or in disentangling the image and text inputs. To address these gaps, IT-Blender leverages pretrained diffusion models (SD and FLUX) to blend the latent representations of a clean reference image with those of the noisy generated image. Combined with our novel blended attention, IT-Blender encodes the real reference image without loss of details and blends the visual concept with the object specified by the text in a disentangled way. Our experiment results show that IT-Blender outperforms the baselines by a large margin in blending visual and textual concepts, shedding light on the new application of image generative models to augment human creativity.
>
---
#### [replaced 048] Bidirectional Prototype-Reward co-Evolution for Test-Time Adaptation of Vision-Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.09394v2](http://arxiv.org/pdf/2503.09394v2)**

> **作者:** Xiaozhen Qiao; Peng Huang; Jiakang Yuan; Xianda Guo; Bowen Ye; Chaocan Xue; Ye Zheng; Zhe Sun; Xuelong Li
>
> **摘要:** Test-time adaptation (TTA) is crucial in maintaining performance of Vision Language Models (VLMs) when facing distribution shifts, particularly when the source data or target labels are inaccessible. Existing TTA methods predominantly leverage the output probability distribution of CLIP for feature evaluation, resulting in biases under domain shifts, which cause misclassified features due to text priors or incorrect textual associations. To address these issues, we propose \underline{B}idirectional Prototype-Reward co-Evolution (BPRE), a novel VLMs framework with TTA that integrates feature quality assessment with prototype evolution via a synergistic feedback loop. First, the Multi-dimensional Quality-aware Reward Module (MQRM) is designed to evaluate feature quality and guide prototype refinement precisely. The continuous refinement of prototype quality via Prototype-Reward Interactive Evolution (PRIE) enhances the computation more robust. Through this bidirectional interaction, the precision of rewards and prototype evolution mutually reinforce each other, forming a self-evolving feedback cycle. Extensive experiments conducted on 15 diverse recognition datasets demonstrate that our model consistently achieves superior performance compared to other SOTA methods, and advances VLM generalization capabilities through emphasizing comprehensive feature evaluation.
>
---
#### [replaced 049] DH-FaceVid-1K: A Large-Scale High-Quality Dataset for Face Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.07151v2](http://arxiv.org/pdf/2410.07151v2)**

> **作者:** Donglin Di; He Feng; Wenzhang Sun; Yongjia Ma; Hao Li; Wei Chen; Lei Fan; Tonghua Su; Xun Yang
>
> **摘要:** Human-centric generative models are becoming increasingly popular, giving rise to various innovative tools and applications, such as talking face videos conditioned on text or audio prompts. The core of these capabilities lies in powerful pre-trained foundation models, trained on large-scale, high-quality datasets. However, many advanced methods rely on in-house data subject to various constraints, and other current studies fail to generate high-resolution face videos, which is mainly attributed to the significant lack of large-scale, high-quality face video datasets. In this paper, we introduce a human face video dataset, \textbf{DH-FaceVid-1K}. Our collection spans 1,200 hours in total, encompassing 270,043 video clips from over 20,000 individuals. Each sample includes corresponding speech audio, facial keypoints, and text annotations. Compared to other publicly available datasets, ours distinguishes itself through its multi-ethnic coverage and high-quality, comprehensive individual attributes. We establish multiple face video generation models supporting tasks such as text-to-video and image-to-video generation. In addition, we develop comprehensive benchmarks to validate the scaling law when using different proportions of proposed dataset. Our primary aim is to contribute a face video dataset, particularly addressing the underrepresentation of Asian faces in existing curated datasets and thereby enriching the global spectrum of face-centric data and mitigating demographic biases. \textbf{Project Page:} https://luna-ai-lab.github.io/DH-FaceVid-1K/
>
---
#### [replaced 050] MEGA-Bench: Scaling Multimodal Evaluation to over 500 Real-World Tasks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.10563v3](http://arxiv.org/pdf/2410.10563v3)**

> **作者:** Jiacheng Chen; Tianhao Liang; Sherman Siu; Zhengqing Wang; Kai Wang; Yubo Wang; Yuansheng Ni; Wang Zhu; Ziyan Jiang; Bohan Lyu; Dongfu Jiang; Xuan He; Yuan Liu; Hexiang Hu; Xiang Yue; Wenhu Chen
>
> **备注:** ICLR 2025 camera-ready version. Project page: https://tiger-ai-lab.github.io/MEGA-Bench/
>
> **摘要:** We present MEGA-Bench, an evaluation suite that scales multimodal evaluation to over 500 real-world tasks, to address the highly heterogeneous daily use cases of end users. Our objective is to optimize for a set of high-quality data samples that cover a highly diverse and rich set of multimodal tasks, while enabling cost-effective and accurate model evaluation. In particular, we collected 505 realistic tasks encompassing over 8,000 samples from 16 expert annotators to extensively cover the multimodal task space. Instead of unifying these problems into standard multi-choice questions (like MMMU, MMBench, and MMT-Bench), we embrace a wide range of output formats like numbers, phrases, code, \LaTeX, coordinates, JSON, free-form, etc. To accommodate these formats, we developed over 40 metrics to evaluate these tasks. Unlike existing benchmarks, MEGA-Bench offers a fine-grained capability report across multiple dimensions (e.g., application, input type, output format, skill), allowing users to interact with and visualize model capabilities in depth. We evaluate a wide variety of frontier vision-language models on MEGA-Bench to understand their capabilities across these dimensions.
>
---
#### [replaced 051] Advancing Automatic Photovoltaic Defect Detection using Semi-Supervised Semantic Segmentation of Electroluminescence Images
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2404.13693v4](http://arxiv.org/pdf/2404.13693v4)**

> **作者:** Abhishek Jha; Yogesh Rawat; Shruti Vyas
>
> **备注:** 19 pages, 10 figures
>
> **摘要:** Photovoltaic (PV) systems allow us to tap into all abundant solar energy, however they require regular maintenance for high efficiency and to prevent degradation. Traditional manual health check, using Electroluminescence (EL) imaging, is expensive and logistically challenging which makes automated defect detection essential. Current automation approaches require extensive manual expert labeling, which is time-consuming, expensive, and prone to errors. We propose PV-S3 (Photovoltaic-Semi-supervised Semantic Segmentation), a Semi-Supervised Learning approach for semantic segmentation of defects in EL images that reduces reliance on extensive labeling. PV-S3 is an artificial intelligence (AI) model trained using a few labeled images along with numerous unlabeled images. We introduce a novel Semi Cross-Entropy loss function to deal with class imbalance. We evaluate PV-S3 on multiple datasets and demonstrate its effectiveness and adaptability. With merely 20% labeled samples, we achieve an absolute improvement of 9.7% in mean Intersection-over-Union (mIoU), 13.5% in Precision, 29.15% in Recall, and 20.42% in F1-Score over prior state-of-the-art supervised method (which uses 100% labeled samples) on University of Central Florida-Electroluminescence (UCF-EL) dataset (largest dataset available for semantic segmentation of EL images) showing improvement in performance while reducing the annotation costs by 80%. For more details, visit our GitHub repository: https://github.com/abj247/PV-S3.
>
---
#### [replaced 052] DNF-Intrinsic: Deterministic Noise-Free Diffusion for Indoor Inverse Rendering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.03924v2](http://arxiv.org/pdf/2507.03924v2)**

> **作者:** Rongjia Zheng; Qing Zhang; Chengjiang Long; Wei-Shi Zheng
>
> **备注:** Accepted to ICCV2025
>
> **摘要:** Recent methods have shown that pre-trained diffusion models can be fine-tuned to enable generative inverse rendering by learning image-conditioned noise-to-intrinsic mapping. Despite their remarkable progress, they struggle to robustly produce high-quality results as the noise-to-intrinsic paradigm essentially utilizes noisy images with deteriorated structure and appearance for intrinsic prediction, while it is common knowledge that structure and appearance information in an image are crucial for inverse rendering. To address this issue, we present DNF-Intrinsic, a robust yet efficient inverse rendering approach fine-tuned from a pre-trained diffusion model, where we propose to take the source image rather than Gaussian noise as input to directly predict deterministic intrinsic properties via flow matching. Moreover, we design a generative renderer to constrain that the predicted intrinsic properties are physically faithful to the source image. Experiments on both synthetic and real-world datasets show that our method clearly outperforms existing state-of-the-art methods.
>
---
#### [replaced 053] LIRA: Inferring Segmentation in Large Multi-modal Models with Local Interleaved Region Assistance
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.06272v2](http://arxiv.org/pdf/2507.06272v2)**

> **作者:** Zhang Li; Biao Yang; Qiang Liu; Shuo Zhang; Zhiyin Ma; Shuo Zhang; Liang Yin; Linger Deng; Yabo Sun; Yuliang Liu; Xiang Bai
>
> **备注:** ICCV 2025
>
> **摘要:** While large multi-modal models (LMMs) demonstrate promising capabilities in segmentation and comprehension, they still struggle with two limitations: inaccurate segmentation and hallucinated comprehension. These challenges stem primarily from constraints in weak visual comprehension and a lack of fine-grained perception. To alleviate these limitations, we propose LIRA, a framework that capitalizes on the complementary relationship between visual comprehension and segmentation via two key components: (1) Semantic-Enhanced Feature Extractor (SEFE) improves object attribute inference by fusing semantic and pixel-level features, leading to more accurate segmentation; (2) Interleaved Local Visual Coupling (ILVC) autoregressively generates local descriptions after extracting local features based on segmentation masks, offering fine-grained supervision to mitigate hallucinations. Furthermore, we find that the precision of object segmentation is positively correlated with the latent related semantics of the <seg> token. To quantify this relationship and the model's potential semantic inferring ability, we introduce the Attributes Evaluation (AttrEval) dataset. Our experiments show that LIRA achieves state-of-the-art performance in both segmentation and comprehension tasks. Code will be available at https://github.com/echo840/LIRA.
>
---
#### [replaced 054] High-Fidelity Differential-information Driven Binary Vision Transformer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.02222v2](http://arxiv.org/pdf/2507.02222v2)**

> **作者:** Tian Gao; Zhiyuan Zhang; Kaijie Yin; Xu-Cheng Zhong; Hui Kong
>
> **摘要:** The binarization of vision transformers (ViTs) offers a promising approach to addressing the trade-off between high computational/storage demands and the constraints of edge-device deployment. However, existing binary ViT methods often suffer from severe performance degradation or rely heavily on full-precision modules. To address these issues, we propose DIDB-ViT, a novel binary ViT that is highly informative while maintaining the original ViT architecture and computational efficiency. Specifically, we design an informative attention module incorporating differential information to mitigate information loss caused by binarization and enhance high-frequency retention. To preserve the fidelity of the similarity calculations between binary Q and K tensors, we apply frequency decomposition using the discrete Haar wavelet and integrate similarities across different frequencies. Additionally, we introduce an improved RPReLU activation function to restructure the activation distribution, expanding the model's representational capacity. Experimental results demonstrate that our DIDB-ViT significantly outperforms state-of-the-art network quantization methods in multiple ViT architectures, achieving superior image classification and segmentation performance.
>
---
#### [replaced 055] Information-Bottleneck Driven Binary Neural Network for Change Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.03504v2](http://arxiv.org/pdf/2507.03504v2)**

> **作者:** Kaijie Yin; Zhiyuan Zhang; Shu Kong; Tian Gao; Chengzhong Xu; Hui Kong
>
> **备注:** ICCV 2025 Accepted
>
> **摘要:** In this paper, we propose Binarized Change Detection (BiCD), the first binary neural network (BNN) designed specifically for change detection. Conventional network binarization approaches, which directly quantize both weights and activations in change detection models, severely limit the network's ability to represent input data and distinguish between changed and unchanged regions. This results in significantly lower detection accuracy compared to real-valued networks. To overcome these challenges, BiCD enhances both the representational power and feature separability of BNNs, improving detection performance. Specifically, we introduce an auxiliary objective based on the Information Bottleneck (IB) principle, guiding the encoder to retain essential input information while promoting better feature discrimination. Since directly computing mutual information under the IB principle is intractable, we design a compact, learnable auxiliary module as an approximation target, leading to a simple yet effective optimization strategy that minimizes both reconstruction loss and standard change detection loss. Extensive experiments on street-view and remote sensing datasets demonstrate that BiCD establishes a new benchmark for BNN-based change detection, achieving state-of-the-art performance in this domain.
>
---
#### [replaced 056] Fair Domain Generalization: An Information-Theoretic View
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.05823v2](http://arxiv.org/pdf/2507.05823v2)**

> **作者:** Tangzheng Lian; Guanyu Hu; Dimitrios Kollias; Xinyu Yang; Oya Celiktutan
>
> **摘要:** Domain generalization (DG) and algorithmic fairness are two critical challenges in machine learning. However, most DG methods focus only on minimizing expected risk in the unseen target domain without considering algorithmic fairness. Conversely, fairness methods typically do not account for domain shifts, so the fairness achieved during training may not generalize to unseen test domains. In this work, we bridge these gaps by studying the problem of Fair Domain Generalization (FairDG), which aims to minimize both expected risk and fairness violations in unseen target domains. We derive novel mutual information-based upper bounds for expected risk and fairness violations in multi-class classification tasks with multi-group sensitive attributes. These bounds provide key insights for algorithm design from an information-theoretic perspective. Guided by these insights, we introduce PAFDG (Pareto-Optimal Fairness for Domain Generalization), a practical framework that solves the FairDG problem and models the utility-fairness trade-off through Pareto optimization. Experiments on real-world vision and language datasets show that PAFDG achieves superior utility-fairness trade-offs compared to existing methods.
>
---
#### [replaced 057] Online Dense Point Tracking with Streaming Memory
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06471v2](http://arxiv.org/pdf/2503.06471v2)**

> **作者:** Qiaole Dong; Yanwei Fu
>
> **备注:** ICCV 2025
>
> **摘要:** Dense point tracking is a challenging task requiring the continuous tracking of every point in the initial frame throughout a substantial portion of a video, even in the presence of occlusions. Traditional methods use optical flow models to directly estimate long-range motion, but they often suffer from appearance drifting without considering temporal consistency. Recent point tracking algorithms usually depend on sliding windows for indirect information propagation from the first frame to the current one, which is slow and less effective for long-range tracking. To account for temporal consistency and enable efficient information propagation, we present a lightweight and fast model with \textbf{S}treaming memory for dense \textbf{PO}int \textbf{T}racking and online video processing. The \textbf{SPOT} framework features three core components: a customized memory reading module for feature enhancement, a sensory memory for short-term motion dynamics modeling, and a visibility-guided splatting module for accurate information propagation. This combination enables SPOT to perform dense point tracking with state-of-the-art accuracy on the CVO benchmark, as well as comparable or superior performance to offline models on sparse tracking benchmarks such as TAP-Vid and RoboTAP. Notably, SPOT with 10$\times$ smaller parameter numbers operates at least 2$\times$ faster than previous state-of-the-art models while maintaining the best performance on CVO. We will release the models and codes at: https://dqiaole.github.io/SPOT/.
>
---
#### [replaced 058] WASABI: A Metric for Evaluating Morphometric Plausibility of Synthetic Brain MRIs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.21771v3](http://arxiv.org/pdf/2504.21771v3)**

> **作者:** Bahram Jafrasteh; Wei Peng; Cheng Wan; Yimin Luo; Ehsan Adeli; Qingyu Zhao
>
> **摘要:** Generative models enhance neuroimaging through data augmentation, quality improvement, and rare condition studies. Despite advances in realistic synthetic MRIs, evaluations focus on texture and perception, lacking sensitivity to crucial anatomical fidelity. This study proposes a new metric, called WASABI (Wasserstein-Based Anatomical Brain Index), to assess the anatomical realism of synthetic brain MRIs. WASABI leverages \textit{SynthSeg}, a deep learning-based brain parcellation tool, to derive volumetric measures of brain regions in each MRI and uses the multivariate Wasserstein distance to compare distributions between real and synthetic anatomies. Based on controlled experiments on two real datasets and synthetic MRIs from five generative models, WASABI demonstrates higher sensitivity in quantifying anatomical discrepancies compared to traditional image-level metrics, even when synthetic images achieve near-perfect visual quality. Our findings advocate for shifting the evaluation paradigm beyond visual inspection and conventional metrics, emphasizing anatomical fidelity as a crucial benchmark for clinically meaningful brain MRI synthesis. Our code is available at https://github.com/BahramJafrasteh/wasabi-mri.
>
---
#### [replaced 059] Benchmarking Unified Face Attack Detection via Hierarchical Prompt Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13327v3](http://arxiv.org/pdf/2505.13327v3)**

> **作者:** Ajian Liu; Haocheng Yuan; Xiao Guo; Hui Ma; Wanyi Zhuang; Changtao Miao; Yan Hong; Chuanbiao Song; Jun Lan; Qi Chu; Tao Gong; Yanyan Liang; Weiqiang Wang; Jun Wan; Xiaoming Liu; Zhen Lei
>
> **摘要:** PAD and FFD are proposed to protect face data from physical media-based Presentation Attacks and digital editing-based DeepFakes, respectively. However, isolated training of these two models significantly increases vulnerability towards unknown attacks, burdening deployment environments. The lack of a Unified Face Attack Detection model to simultaneously handle attacks in these two categories is mainly attributed to two factors: (1) A benchmark that is sufficient for models to explore is lacking. Existing UAD datasets only contain limited attack types and samples, leading to the model's confined ability to address abundant advanced threats. In light of these, through an explainable hierarchical way, we propose the most extensive and sophisticated collection of forgery techniques available to date, namely UniAttackDataPlus. Our UniAttackData+ encompasses 2,875 identities and their 54 kinds of corresponding falsified samples, in a total of 697,347 videos. (2) The absence of a trustworthy classification criterion. Current methods endeavor to explore an arbitrary criterion within the same semantic space, which fails to exist when encountering diverse attacks. Thus, we present a novel Visual-Language Model-based Hierarchical Prompt Tuning Framework that adaptively explores multiple classification criteria from different semantic spaces. Specifically, we construct a VP-Tree to explore various classification rules hierarchically. Then, by adaptively pruning the prompts, the model can select the most suitable prompts guiding the encoder to extract discriminative features at different levels in a coarse-to-fine manner. Finally, to help the model understand the classification criteria in visual space, we propose a DPI module to project the visual prompts to the text encoder to help obtain a more accurate semantics.
>
---
#### [replaced 060] Cross-modal Ship Re-Identification via Optical and SAR Imagery: A Novel Dataset and Method
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.22027v2](http://arxiv.org/pdf/2506.22027v2)**

> **作者:** Han Wang; Shengyang Li; Jian Yang; Yuxuan Liu; Yixuan Lv; Zhuang Zhou
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Detecting and tracking ground objects using earth observation imagery remains a significant challenge in the field of remote sensing. Continuous maritime ship tracking is crucial for applications such as maritime search and rescue, law enforcement, and shipping analysis. However, most current ship tracking methods rely on geostationary satellites or video satellites. The former offer low resolution and are susceptible to weather conditions, while the latter have short filming durations and limited coverage areas, making them less suitable for the real-world requirements of ship tracking. To address these limitations, we present the Hybrid Optical and Synthetic Aperture Radar (SAR) Ship Re-Identification Dataset (HOSS ReID dataset), designed to evaluate the effectiveness of ship tracking using low-Earth orbit constellations of optical and SAR sensors. This approach ensures shorter re-imaging cycles and enables all-weather tracking. HOSS ReID dataset includes images of the same ship captured over extended periods under diverse conditions, using different satellites of different modalities at varying times and angles. Furthermore, we propose a baseline method for cross-modal ship re-identification, TransOSS, which is built on the Vision Transformer architecture. It refines the patch embedding structure to better accommodate cross-modal tasks, incorporates additional embeddings to introduce more reference information, and employs contrastive learning to pre-train on large-scale optical-SAR image pairs, ensuring the model's ability to extract modality-invariant features. Our dataset and baseline method are publicly available on https://github.com/Alioth2000/Hoss-ReID.
>
---
#### [replaced 061] Is Intermediate Fusion All You Need for UAV-based Collaborative Perception?
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.21774v2](http://arxiv.org/pdf/2504.21774v2)**

> **作者:** Jiuwu Hao; Liguo Sun; Yuting Wan; Yueyang Wu; Ti Xiang; Haolin Song; Pin Lv
>
> **备注:** Accepted by ITSC 2025
>
> **摘要:** Collaborative perception enhances environmental awareness through inter-agent communication and is regarded as a promising solution to intelligent transportation systems. However, existing collaborative methods for Unmanned Aerial Vehicles (UAVs) overlook the unique characteristics of the UAV perspective, resulting in substantial communication overhead. To address this issue, we propose a novel communication-efficient collaborative perception framework based on late-intermediate fusion, dubbed LIF. The core concept is to exchange informative and compact detection results and shift the fusion stage to the feature representation level. In particular, we leverage vision-guided positional embedding (VPE) and box-based virtual augmented feature (BoBEV) to effectively integrate complementary information from various agents. Additionally, we innovatively introduce an uncertainty-driven communication mechanism that uses uncertainty evaluation to select high-quality and reliable shared areas. Experimental results demonstrate that our LIF achieves superior performance with minimal communication bandwidth, proving its effectiveness and practicality. Code and models are available at https://github.com/uestchjw/LIF.
>
---
#### [replaced 062] Understanding Pan-Sharpening via Generalized Inverse
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2310.02718v2](http://arxiv.org/pdf/2310.02718v2)**

> **作者:** Shiqi Liu; Yutong Bai; Xinyang Han; Alan Yuille
>
> **摘要:** Pan-sharpening algorithm utilizes panchromatic image and multispectral image to obtain a high spatial and high spectral image. However, the optimizations of the algorithms are designed with different standards. We adopt the simple matrix equation to describe the Pan-sharpening problem. The solution existence condition and the acquirement of spectral and spatial resolution are discussed. A down-sampling enhancement method was introduced for better acquiring the spatial and spectral down-sample matrices. By the generalized inverse theory, we derived two forms of general inverse matrix formulations that can correspond to the two prominent classes of Pan-sharpening methods, that is, component substitution and multi-resolution analysis methods. Specifically, the Gram Schmidt Adaptive(GSA) was proved to follow the general inverse matrix formulation of component substitution. A model prior to the general inverse matrix of the spectral function was rendered. The theoretical errors are analyzed. Synthetic experiments and real data experiments are implemented. The proposed methods are better and sharper than other methods qualitatively in both synthetic and real experiments. The down-sample enhancement effect is shown of better results both quantitatively and qualitatively in real experiments. The generalized inverse matrix theory help us better understand the Pan-sharpening.
>
---
#### [replaced 063] SLGaussian: Fast Language Gaussian Splatting in Sparse Views
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.08331v2](http://arxiv.org/pdf/2412.08331v2)**

> **作者:** Kangjie Chen; BingQuan Dai; Minghan Qin; Dongbin Zhang; Peihao Li; Yingshuang Zou; Haoqian Wang
>
> **备注:** Accepted by ACM MM 2025. Project page: https://chenkangjie1123.github.io/SLGaussian.github.io/
>
> **摘要:** 3D semantic field learning is crucial for applications like autonomous navigation, AR/VR, and robotics, where accurate comprehension of 3D scenes from limited viewpoints is essential. Existing methods struggle under sparse view conditions, relying on inefficient per-scene multi-view optimizations, which are impractical for many real-world tasks. To address this, we propose SLGaussian, a feed-forward method for constructing 3D semantic fields from sparse viewpoints, allowing direct inference of 3DGS-based scenes. By ensuring consistent SAM segmentations through video tracking and using low-dimensional indexing for high-dimensional CLIP features, SLGaussian efficiently embeds language information in 3D space, offering a robust solution for accurate 3D scene understanding under sparse view conditions. In experiments on two-view sparse 3D object querying and segmentation in the LERF and 3D-OVS datasets, SLGaussian outperforms existing methods in chosen IoU, Localization Accuracy, and mIoU. Moreover, our model achieves scene inference in under 30 seconds and open-vocabulary querying in just 0.011 seconds per query.
>
---
#### [replaced 064] VideoChat-Flash: Hierarchical Compression for Long-Context Video Modeling
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.00574v4](http://arxiv.org/pdf/2501.00574v4)**

> **作者:** Xinhao Li; Yi Wang; Jiashuo Yu; Xiangyu Zeng; Yuhan Zhu; Haian Huang; Jianfei Gao; Kunchang Li; Yinan He; Chenting Wang; Yu Qiao; Yali Wang; Limin Wang
>
> **摘要:** Long-context video modeling is critical for multimodal large language models (MLLMs), enabling them to process movies, online video streams, and so on. Despite its advances, handling long videos remains challenging due to the difficulty in efficiently understanding the extremely long video context. This paper aims to address this issue from aspects of model architecture, training data, training strategy and evaluation benchmark. First, we propose a novel Hierarchical video token Compression (HiCo) method, which leverages visual redundancy in long videos to compress long video context from Clip-level to Video-level, reducing the computation significantly while preserving essential details, achieving an extreme compression ratio of approximately 1/50 with almost no performance loss. Second, we introduce a multi-stage short-to-long learning scheme, a large-scale dataset of real-world long videos named LongVid, and a challenging ``Multi-Hop Needle-In-A-Video-Haystack'' benchmark. Finally, we build a powerful video MLLM named VideoChat-Flash, which shows a leading performance on both mainstream long and short video benchmarks at the 2B and 7B model scale. It first gets 99.1% accuracy over 10,000 frames in NIAH among open-source models.
>
---
#### [replaced 065] VisOnlyQA: Large Vision Language Models Still Struggle with Visual Perception of Geometric Information
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.00947v3](http://arxiv.org/pdf/2412.00947v3)**

> **作者:** Ryo Kamoi; Yusen Zhang; Sarkar Snigdha Sarathi Das; Ranran Haoran Zhang; Rui Zhang
>
> **备注:** COLM 2025. VisOnlyQA dataset, code, and model responses are provided at https://github.com/psunlpgroup/VisOnlyQA. Please also refer to our project website at https://visonlyqa.github.io/
>
> **摘要:** Large Vision Language Models (LVLMs) have achieved remarkable performance in various vision-language tasks. However, it is still unclear how accurately LVLMs can perceive visual information in images. In particular, the capability of LVLMs to perceive geometric information, such as shape, angle, and size, remains insufficiently analyzed, although the perception of these properties is crucial for tasks that require a detailed visual understanding. In this work, we introduce VisOnlyQA, a dataset for evaluating the geometric perception of LVLMs, and reveal that LVLMs often cannot accurately perceive basic geometric information in images, while human performance is nearly perfect. VisOnlyQA consists of 12 tasks that directly ask about geometric information in geometric shapes, charts, chemical structures, and 3D shapes. Our experiments highlight the following findings: (i) State-of-the-art LVLMs struggle with basic geometric perception. 23 LVLMs we evaluate, including GPT-4o and Gemini 2.5 Pro, work poorly on VisOnlyQA. (ii) Additional training data does not resolve this issue. Fine-tuning on the training set of VisOnlyQA is not always effective, even for in-distribution tasks. (iii) LLM may be the bottleneck. LVLMs using stronger LLMs exhibit better geometric perception on VisOnlyQA, while it does not require complex reasoning, suggesting that the way LVLMs process information from visual encoders is a bottleneck. The datasets, code, and model responses are provided at https://github.com/psunlpgroup/VisOnlyQA.
>
---
#### [replaced 066] READoc: A Unified Benchmark for Realistic Document Structured Extraction
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.05137v3](http://arxiv.org/pdf/2409.05137v3)**

> **作者:** Zichao Li; Aizier Abulaiti; Yaojie Lu; Xuanang Chen; Jia Zheng; Hongyu Lin; Xianpei Han; Le Sun
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Document Structured Extraction (DSE) aims to extract structured content from raw documents. Despite the emergence of numerous DSE systems, their unified evaluation remains inadequate, significantly hindering the field's advancement. This problem is largely attributed to existing benchmark paradigms, which exhibit fragmented and localized characteristics. To address these limitations and offer a thorough evaluation of DSE systems, we introduce a novel benchmark named READoc, which defines DSE as a realistic task of converting unstructured PDFs into semantically rich Markdown. The READoc dataset is derived from 3,576 diverse and real-world documents from arXiv, GitHub, and Zenodo. In addition, we develop a DSE Evaluation S$^3$uite comprising Standardization, Segmentation and Scoring modules, to conduct a unified evaluation of state-of-the-art DSE approaches. By evaluating a range of pipeline tools, expert visual models, and general VLMs, we identify the gap between current work and the unified, realistic DSE objective for the first time. We aspire that READoc will catalyze future research in DSE, fostering more comprehensive and practical solutions.
>
---
#### [replaced 067] FuseUNet: A Multi-Scale Feature Fusion Method for U-like Networks
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.05821v2](http://arxiv.org/pdf/2506.05821v2)**

> **作者:** Quansong He; Xiangde Min; Kaishen Wang; Tao He
>
> **备注:** ICML2025
>
> **摘要:** Medical image segmentation is a critical task in computer vision, with UNet serving as a milestone architecture. The typical component of UNet family is the skip connection, however, their skip connections face two significant limitations: (1) they lack effective interaction between features at different scales, and (2) they rely on simple concatenation or addition operations, which constrain efficient information integration. While recent improvements to UNet have focused on enhancing encoder and decoder capabilities, these limitations remain overlooked. To overcome these challenges, we propose a novel multi-scale feature fusion method that reimagines the UNet decoding process as solving an initial value problem (IVP), treating skip connections as discrete nodes. By leveraging principles from the linear multistep method, we propose an adaptive ordinary differential equation method to enable effective multi-scale feature fusion. Our approach is independent of the encoder and decoder architectures, making it adaptable to various U-Net-like networks. Experiments on ACDC, KiTS2023, MSD brain tumor, and ISIC2017/2018 skin lesion segmentation datasets demonstrate improved feature utilization, reduced network parameters, and maintained high performance. The code is available at https://github.com/nayutayuki/FuseUNet.
>
---
#### [replaced 068] UniQA: Unified Vision-Language Pre-training for Image Quality and Aesthetic Assessment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.01069v2](http://arxiv.org/pdf/2406.01069v2)**

> **作者:** Hantao Zhou; Longxiang Tang; Rui Yang; Guanyi Qin; Yan Zhang; Yutao Li; Xiu Li; Runze Hu; Guangtao Zhai
>
> **摘要:** Image Quality Assessment (IQA) and Image Aesthetic Assessment (IAA) aim to simulate human subjective perception of image visual quality and aesthetic appeal. Despite distinct learning objectives, they have underlying interconnectedness due to consistent human assessment perception. In this paper, we propose Unified vision-language pre-training of Quality and Aesthetics (UniQA}), to extract useful and common representations from two tasks, thereby benefiting them simultaneously. However, the lack of text in the IQA datasets and the textual noise in the IAA datasets pose severe challenges for multimodal pre-training. To address this, we (1) utilize multimodal large language models (MLLMs) to generate high-quality text descriptions; (2) use the generated text for IAA as metadata to purify noisy IAA data. To effectively adapt the pre-trained UniQA to downstream tasks, we further propose a lightweight adapter that utilizes versatile cues to fully exploit the extensive knowledge of the pre-trained model. UniQA demonstrates high competitiveness in various image assessment tasks, including classical IQA and IAA tasks, few-label IQA, and other downstream tasks, showing promise as a foundational assessment model. Codes are available at https://github.com/zht8506/UniQA.
>
---
#### [replaced 069] Average Calibration Error: A Differentiable Loss for Improved Reliability in Image Segmentation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.06759v4](http://arxiv.org/pdf/2403.06759v4)**

> **作者:** Theodore Barfoot; Luis Garcia-Peraza-Herrera; Ben Glocker; Tom Vercauteren
>
> **备注:** Camera ready version as in 10.1007/978-3-031-72114-4_14
>
> **摘要:** Deep neural networks for medical image segmentation often produce overconfident results misaligned with empirical observations. Such miscalibration, challenges their clinical translation. We propose to use marginal L1 average calibration error (mL1-ACE) as a novel auxiliary loss function to improve pixel-wise calibration without compromising segmentation quality. We show that this loss, despite using hard binning, is directly differentiable, bypassing the need for approximate but differentiable surrogate or soft binning approaches. Our work also introduces the concept of dataset reliability histograms which generalises standard reliability diagrams for refined visual assessment of calibration in semantic segmentation aggregated at the dataset level. Using mL1-ACE, we reduce average and maximum calibration error by 45% and 55% respectively, maintaining a Dice score of 87% on the BraTS 2021 dataset. We share our code here: https://github.com/cai4cai/ACE-DLIRIS
>
---
#### [replaced 070] Learning Traffic Anomalies from Generative Models on Real-Time Observations
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.01391v5](http://arxiv.org/pdf/2502.01391v5)**

> **作者:** Fotis I. Giasemis; Alexandros Sopasakis
>
> **摘要:** Accurate detection of traffic anomalies is crucial for effective urban traffic management and congestion mitigation. We use the Spatiotemporal Generative Adversarial Network (STGAN) framework combining Graph Neural Networks and Long Short-Term Memory networks to capture complex spatial and temporal dependencies in traffic data. We apply STGAN to real-time, minute-by-minute observations from 42 traffic cameras across Gothenburg, Sweden, collected over several months in 2020. The images are processed to compute a flow metric representing vehicle density, which serves as input for the model. Training is conducted on data from April to November 2020, and validation is performed on a separate dataset from November 14 to 23, 2020. Our results demonstrate that the model effectively detects traffic anomalies with high precision and low false positive rates. The detected anomalies include camera signal interruptions, visual artifacts, and extreme weather conditions affecting traffic flow.
>
---
#### [replaced 071] CVVNet: A Cross-Vertical-View Network for Gait Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.01837v2](http://arxiv.org/pdf/2505.01837v2)**

> **作者:** Xiangru Li; Wei Song; Yingda Huang; Wei Meng; Le Chang
>
> **摘要:** Gait recognition enables contact-free, long-range person identification that is robust to clothing variations and non-cooperative scenarios. While existing methods perform well in controlled indoor environments, they struggle with cross-vertical view scenarios, where surveillance angles vary significantly in elevation. Our experiments show up to 60\% accuracy degradation in low-to-high vertical view settings due to severe deformations and self-occlusions of key anatomical features. Current CNN and self-attention-based methods fail to effectively handle these challenges, due to their reliance on single-scale convolutions or simplistic attention mechanisms that lack effective multi-frequency feature integration. To tackle this challenge, we propose CVVNet (Cross-Vertical-View Network), a frequency aggregation architecture specifically designed for robust cross-vertical-view gait recognition. CVVNet employs a High-Low Frequency Extraction module (HLFE) that adopts parallel multi-scale convolution/max-pooling path and self-attention path as high- and low-frequency mixers for effective multi-frequency feature extraction from input silhouettes. We also introduce the Dynamic Gated Aggregation (DGA) mechanism to adaptively adjust the fusion ratio of high- and low-frequency features. The integration of our core Multi-Scale Attention Gated Aggregation (MSAGA) module, HLFE and DGA enables CVVNet to effectively handle distortions from view changes, significantly improving the recognition robustness across different vertical views. Experimental results show that our CVVNet achieves state-of-the-art performance, with $8.6\%$ improvement on DroneGait and $2\%$ on Gait3D compared with the best existing methods.
>
---
#### [replaced 072] Hierarchical Attention Fusion of Visual and Textual Representations for Cross-Domain Sequential Recommendation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.15085v3](http://arxiv.org/pdf/2504.15085v3)**

> **作者:** Wangyu Wu; Zhenhong Chen; Siqi Song; Xianglin Qiu; Xiaowei Huang; Fei Ma; Jimin Xiao
>
> **备注:** Accepted at CogSCI 2025. arXiv admin note: text overlap with arXiv:2502.15694
>
> **摘要:** Cross-Domain Sequential Recommendation (CDSR) predicts user behavior by leveraging historical interactions across multiple domains, focusing on modeling cross-domain preferences through intra- and inter-sequence item relationships. Inspired by human cognitive processes, we propose Hierarchical Attention Fusion of Visual and Textual Representations (HAF-VT), a novel approach integrating visual and textual data to enhance cognitive modeling. Using the frozen CLIP model, we generate image and text embeddings, enriching item representations with multimodal data. A hierarchical attention mechanism jointly learns single-domain and cross-domain preferences, mimicking human information integration. Evaluated on four e-commerce datasets, HAF-VT outperforms existing methods in capturing cross-domain user interests, bridging cognitive principles with computational models and highlighting the role of multimodal data in sequential decision-making.
>
---
#### [replaced 073] Democratizing High-Fidelity Co-Speech Gesture Video Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.06812v2](http://arxiv.org/pdf/2507.06812v2)**

> **作者:** Xu Yang; Shaoli Huang; Shenbo Xie; Xuelin Chen; Yifei Liu; Changxing Ding
>
> **备注:** ICCV 2025
>
> **摘要:** Co-speech gesture video generation aims to synthesize realistic, audio-aligned videos of speakers, complete with synchronized facial expressions and body gestures. This task presents challenges due to the significant one-to-many mapping between audio and visual content, further complicated by the scarcity of large-scale public datasets and high computational demands. We propose a lightweight framework that utilizes 2D full-body skeletons as an efficient auxiliary condition to bridge audio signals with visual outputs. Our approach introduces a diffusion model conditioned on fine-grained audio segments and a skeleton extracted from the speaker's reference image, predicting skeletal motions through skeleton-audio feature fusion to ensure strict audio coordination and body shape consistency. The generated skeletons are then fed into an off-the-shelf human video generation model with the speaker's reference image to synthesize high-fidelity videos. To democratize research, we present CSG-405-the first public dataset with 405 hours of high-resolution videos across 71 speech types, annotated with 2D skeletons and diverse speaker demographics. Experiments show that our method exceeds state-of-the-art approaches in visual quality and synchronization while generalizing across speakers and contexts. Code, models, and CSG-405 are publicly released at https://mpi-lab.github.io/Democratizing-CSG/
>
---
#### [replaced 074] Spiking Transformers Need High Frequency Information
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18608v2](http://arxiv.org/pdf/2505.18608v2)**

> **作者:** Yuetong Fang; Deming Zhou; Ziqing Wang; Hongwei Ren; ZeCui Zeng; Lusong Li; Shibo Zhou; Renjing Xu
>
> **摘要:** Spiking Transformers offer an energy-efficient alternative to conventional deep learning by transmitting information solely through binary (0/1) spikes. However, there remains a substantial performance gap compared to artificial neural networks. A common belief is that their binary and sparse activation transmission leads to information loss, thus degrading feature representation and accuracy. In this work, however, we reveal for the first time that spiking neurons preferentially propagate low-frequency information. We hypothesize that the rapid dissipation of high-frequency components is the primary cause of performance degradation. For example, on Cifar-100, adopting Avg-Pooling (low-pass) for token mixing lowers performance to 76.73%; interestingly, replacing it with Max-Pooling (high-pass) pushes the top-1 accuracy to 79.12%, surpassing the well-tuned Spikformer baseline by 0.97%. Accordingly, we introduce Max-Former that restores high-frequency signals through two frequency-enhancing operators: extra Max-Pooling in patch embedding and Depth-Wise Convolution in place of self-attention. Notably, our Max-Former (63.99 M) hits the top-1 accuracy of 82.39% on ImageNet, showing a +7.58% improvement over Spikformer with comparable model size (74.81%, 66.34 M). We hope this simple yet effective solution inspires future research to explore the distinctive nature of spiking neural networks, beyond the established practice in standard deep learning. \href{https://github.com/bic-L/Spiking-Transformers-Need-High-Frequency-Information}{Code} is available.
>
---
#### [replaced 075] Adversarial Augmentation Training Makes Action Recognition Models More Robust to Realistic Video Distribution Shifts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2401.11406v2](http://arxiv.org/pdf/2401.11406v2)**

> **作者:** Kiyoon Kim; Shreyank N Gowda; Panagiotis Eustratiadis; Antreas Antoniou; Robert B Fisher
>
> **备注:** Accepted to ICPRAI 2024
>
> **摘要:** Despite recent advances in video action recognition achieving strong performance on existing benchmarks, these models often lack robustness when faced with natural distribution shifts between training and test data. We propose two novel evaluation methods to assess model resilience to such distribution disparity. One method uses two different datasets collected from different sources and uses one for training and validation, and the other for testing. More precisely, we created dataset splits of HMDB-51 or UCF-101 for training, and Kinetics-400 for testing, using the subset of the classes that are overlapping in both train and test datasets. The other proposed method extracts the feature mean of each class from the target evaluation dataset's training data (i.e. class prototype) and estimates test video prediction as a cosine similarity score between each sample to the class prototypes of each target class. This procedure does not alter model weights using the target dataset and it does not require aligning overlapping classes of two different datasets, thus is a very efficient method to test the model robustness to distribution shifts without prior knowledge of the target distribution. We address the robustness problem by adversarial augmentation training - generating augmented views of videos that are "hard" for the classification model by applying gradient ascent on the augmentation parameters - as well as "curriculum" scheduling the strength of the video augmentations. We experimentally demonstrate the superior performance of the proposed adversarial augmentation approach over baselines across three state-of-the-art action recognition models - TSM, Video Swin Transformer, and Uniformer. The presented work provides critical insight into model robustness to distribution shifts and presents effective techniques to enhance video action recognition performance in a real-world deployment.
>
---
#### [replaced 076] Gamma: Toward Generic Image Assessment with Mixture of Assessment Experts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06678v2](http://arxiv.org/pdf/2503.06678v2)**

> **作者:** Hantao Zhou; Rui Yang; Longxiang Tang; Guanyi Qin; Runze Hu; Xiu Li
>
> **备注:** Accepted to ACMMM 2025
>
> **摘要:** Image assessment aims to evaluate the quality and aesthetics of images and has been applied across various scenarios, such as natural and AIGC scenes. Existing methods mostly address these sub-tasks or scenes individually. While some works attempt to develop unified image assessment models, they have struggled to achieve satisfactory performance or cover a broad spectrum of assessment scenarios. In this paper, we present \textbf{Gamma}, a \textbf{G}eneric im\textbf{A}ge assess\textbf{M}ent model using \textbf{M}ixture of \textbf{A}ssessment Experts, which can effectively assess images from diverse scenes through mixed-dataset training. Achieving unified training in image assessment presents significant challenges due to annotation biases across different datasets. To address this issue, we first propose a Mixture of Assessment Experts (MoAE) module, which employs shared and adaptive experts to dynamically learn common and specific knowledge for different datasets, respectively. In addition, we introduce a Scene-based Differential Prompt (SDP) strategy, which uses scene-specific prompts to provide prior knowledge and guidance during the learning process, further boosting adaptation for various scenes. Our Gamma model is trained and evaluated on 12 datasets spanning 6 image assessment scenarios. Extensive experiments show that our unified Gamma outperforms other state-of-the-art mixed-training methods by significant margins while covering more scenes. Codes are available at https://github.com/zht8506/Gamma.
>
---
#### [replaced 077] BayesSDF: Surface-Based Laplacian Uncertainty Estimation for 3D Geometry with Neural Signed Distance Fields
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.06269v2](http://arxiv.org/pdf/2507.06269v2)**

> **作者:** Rushil Desai
>
> **备注:** ICCV 2025 Workshops (8 Pages, 6 Figures, 2 Tables)
>
> **摘要:** Quantifying uncertainty in neural implicit 3D representations, particularly those utilizing Signed Distance Functions (SDFs), remains a substantial challenge due to computational inefficiencies, scalability issues, and geometric inconsistencies. Existing methods typically neglect direct geometric integration, leading to poorly calibrated uncertainty maps. We introduce BayesSDF, a novel probabilistic framework for uncertainty quantification in neural implicit SDF models, motivated by scientific simulation applications with 3D environments (e.g., forests) such as modeling fluid flow through forests, where precise surface geometry and reliable uncertainty estimates are essential. Unlike radiance-based models such as Neural Radiance Fields (NeRF) or 3D Gaussian splatting, which lack explicit surface formulations, Signed Distance Functions (SDFs) define continuous and differentiable geometry, making them better suited for physical modeling and analysis. BayesSDF leverages a Laplace approximation to quantify local surface instability using Hessian-based metrics, enabling efficient, surfaceaware uncertainty estimation. Our method shows that uncertainty predictions correspond closely with poorly reconstructed geometry, providing actionable confidence measures for downstream use. Extensive evaluations on synthetic and real-world datasets demonstrate that BayesSDF outperforms existing methods in both calibration and geometric consistency, establishing a strong foundation for uncertainty-aware 3D scene reconstruction, simulation, and robotic decision-making.
>
---
#### [replaced 078] Multi Source COVID-19 Detection via Kernel-Density-based Slice Sampling
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.01564v2](http://arxiv.org/pdf/2507.01564v2)**

> **作者:** Chia-Ming Lee; Bo-Cheng Qiu; Ting-Yao Chen; Ming-Han Sun; Fang-Ying Lin; Jung-Tse Tsai; I-An Tsai; Yu-Fan Lin; Chih-Chung Hsu
>
> **摘要:** We present our solution for the Multi-Source COVID-19 Detection Challenge, which classifies chest CT scans from four distinct medical centers. To address multi-source variability, we employ the Spatial-Slice Feature Learning (SSFL) framework with Kernel-Density-based Slice Sampling (KDS). Our preprocessing pipeline combines lung region extraction, quality control, and adaptive slice sampling to select eight representative slices per scan. We compare EfficientNet and Swin Transformer architectures on the validation set. The EfficientNet model achieves an F1-score of 94.68%, compared to the Swin Transformer's 93.34%. The results demonstrate the effectiveness of our KDS-based pipeline on multi-source data and highlight the importance of dataset balance in multi-institutional medical imaging evaluation.
>
---
#### [replaced 079] CCDM: Continuous Conditional Diffusion Models for Image Generation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.03546v3](http://arxiv.org/pdf/2405.03546v3)**

> **作者:** Xin Ding; Yongwei Wang; Kao Zhang; Z. Jane Wang
>
> **摘要:** Continuous Conditional Generative Modeling (CCGM) estimates high-dimensional data distributions, such as images, conditioned on scalar continuous variables (aka regression labels). While Continuous Conditional Generative Adversarial Networks (CcGANs) were designed for this task, their instability during adversarial learning often leads to suboptimal results. Conditional Diffusion Models (CDMs) offer a promising alternative, generating more realistic images, but their diffusion processes, label conditioning, and model fitting procedures are either not optimized for or incompatible with CCGM, making it difficult to integrate CcGANs' vicinal approach. To address these issues, we introduce Continuous Conditional Diffusion Models (CCDMs), the first CDM specifically tailored for CCGM. CCDMs address existing limitations with specially designed conditional diffusion processes, a novel hard vicinal image denoising loss, a customized label embedding method, and efficient conditional sampling procedures. Through comprehensive experiments on four datasets with resolutions ranging from 64x64 to 192x192, we demonstrate that CCDMs outperform state-of-the-art CCGM models, establishing a new benchmark. Ablation studies further validate the model design and implementation, highlighting that some widely used CDM implementations are ineffective for the CCGM task. Our code is publicly available at https://github.com/UBCDingXin/CCDM.
>
---
#### [replaced 080] Leveraging Segment Anything Model for Source-Free Domain Adaptation via Dual Feature Guided Auto-Prompting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08527v3](http://arxiv.org/pdf/2505.08527v3)**

> **作者:** Zheang Huai; Hui Tang; Yi Li; Zhuangzhuang Chen; Xiaomeng Li
>
> **备注:** Accepted in TMI 2025
>
> **摘要:** Source-free domain adaptation (SFDA) for segmentation aims at adapting a model trained in the source domain to perform well in the target domain with only the source model and unlabeled target data. Inspired by the recent success of Segment Anything Model (SAM) which exhibits the generality of segmenting images of various modalities and in different domains given human-annotated prompts like bounding boxes or points, we for the first time explore the potentials of Segment Anything Model for SFDA via automatedly finding an accurate bounding box prompt. We find that the bounding boxes directly generated with existing SFDA approaches are defective due to the domain gap. To tackle this issue, we propose a novel Dual Feature Guided (DFG) auto-prompting approach to search for the box prompt. Specifically, the source model is first trained in a feature aggregation phase, which not only preliminarily adapts the source model to the target domain but also builds a feature distribution well-prepared for box prompt search. In the second phase, based on two feature distribution observations, we gradually expand the box prompt with the guidance of the target model feature and the SAM feature to handle the class-wise clustered target features and the class-wise dispersed target features, respectively. To remove the potentially enlarged false positive regions caused by the over-confident prediction of the target model, the refined pseudo-labels produced by SAM are further postprocessed based on connectivity analysis. Experiments on 3D and 2D datasets indicate that our approach yields superior performance compared to conventional methods. Code is available at https://github.com/xmed-lab/DFG.
>
---
#### [replaced 081] Re-boosting Self-Collaboration Parallel Prompt GAN for Unsupervised Image Restoration
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2408.09241v2](http://arxiv.org/pdf/2408.09241v2)**

> **作者:** Xin Lin; Yuyan Zhou; Jingtong Yue; Chao Ren; Kelvin C. K. Chan; Lu Qi; Ming-Hsuan Yang
>
> **备注:** Accepted in IEEE T-PAMI
>
> **摘要:** Unsupervised restoration approaches based on generative adversarial networks (GANs) offer a promising solution without requiring paired datasets. Yet, these GAN-based approaches struggle to surpass the performance of conventional unsupervised GAN-based frameworks without significantly modifying model structures or increasing the computational complexity. To address these issues, we propose a self-collaboration (SC) strategy for existing restoration models. This strategy utilizes information from the previous stage as feedback to guide subsequent stages, achieving significant performance improvement without increasing the framework's inference complexity. The SC strategy comprises a prompt learning (PL) module and a restorer ($Res$). It iteratively replaces the previous less powerful fixed restorer $\overline{Res}$ in the PL module with a more powerful $Res$. The enhanced PL module generates better pseudo-degraded/clean image pairs, leading to a more powerful $Res$ for the next iteration. Our SC can significantly improve the $Res$'s performance by over 1.5 dB without adding extra parameters or computational complexity during inference. Meanwhile, existing self-ensemble (SE) and our SC strategies enhance the performance of pre-trained restorers from different perspectives. As SE increases computational complexity during inference, we propose a re-boosting module to the SC (Reb-SC) to improve the SC strategy further by incorporating SE into SC without increasing inference time. This approach further enhances the restorer's performance by approximately 0.3 dB. Extensive experimental results on restoration tasks demonstrate that the proposed model performs favorably against existing state-of-the-art unsupervised restoration methods. Source code and trained models are publicly available at: https://github.com/linxin0/RSCP2GAN.
>
---
#### [replaced 082] Concept Steerers: Leveraging K-Sparse Autoencoders for Test-Time Controllable Generations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.19066v2](http://arxiv.org/pdf/2501.19066v2)**

> **作者:** Dahye Kim; Deepti Ghadiyaram
>
> **备注:** 23 pages, 18 figures
>
> **摘要:** Despite the remarkable progress in text-to-image generative models, they are prone to adversarial attacks and inadvertently generate unsafe, unethical content. Existing approaches often rely on fine-tuning models to remove specific concepts, which is computationally expensive, lacks scalability, and/or compromises generation quality. In this work, we propose a novel framework leveraging k-sparse autoencoders (k-SAEs) to enable efficient and interpretable concept manipulation in diffusion models. Specifically, we first identify interpretable monosemantic concepts in the latent space of text embeddings and leverage them to precisely steer the generation away or towards a given concept (e.g., nudity) or to introduce a new concept (e.g., photographic style) -- all during test time. Through extensive experiments, we demonstrate that our approach is very simple, requires no retraining of the base model nor LoRA adapters, does not compromise the generation quality, and is robust to adversarial prompt manipulations. Our method yields an improvement of $\mathbf{20.01\%}$ in unsafe concept removal, is effective in style manipulation, and is $\mathbf{\sim5}$x faster than the current state-of-the-art. Code is available at: https://github.com/kim-dahye/steerers
>
---
#### [replaced 083] A Novel Streamline-based diffusion MRI Tractography Registration Method with Probabilistic Keypoint Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.02481v2](http://arxiv.org/pdf/2503.02481v2)**

> **作者:** Junyi Wang; Mubai Du; Ye Wu; Yijie Li; William M. Wells III; Lauren J. O'Donnell; Fan Zhang
>
> **摘要:** Registration of diffusion MRI tractography is an essential step for analyzing group similarities and variations in the brain's white matter (WM). Streamline-based registration approaches can leverage the 3D geometric information of fiber pathways to enable spatial alignment after registration. Existing methods usually rely on the optimization of the spatial distances to identify the optimal transformation. However, such methods overlook point connectivity patterns within the streamline itself, limiting their ability to identify anatomical correspondences across tractography datasets. In this work, we propose a novel unsupervised approach using deep learning to perform streamline-based dMRI tractography registration. The overall idea is to identify corresponding keypoint pairs across subjects for spatial alignment of tractography datasets. We model tractography as point clouds to leverage the graph connectivity along streamlines. We propose a novel keypoint detection method for streamlines, framed as a probabilistic classification task to identify anatomically consistent correspondences across unstructured streamline sets. In the experiments, we compare several existing methods and show highly effective and efficient tractography registration performance.
>
---
#### [replaced 084] MEGANet-W: A Wavelet-Driven Edge-Guided Attention Framework for Weak Boundary Polyp Detection
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.02668v2](http://arxiv.org/pdf/2507.02668v2)**

> **作者:** Zhe Yee Tan
>
> **备注:** 7 pages, 3 figures
>
> **摘要:** Colorectal polyp segmentation is critical for early detection of colorectal cancer, yet weak and low contrast boundaries significantly limit automated accuracy. Existing deep models either blur fine edge details or rely on handcrafted filters that perform poorly under variable imaging conditions. We propose MEGANet-W, a Wavelet Driven Edge Guided Attention Network that injects directional, parameter free Haar wavelet edge maps into each decoder stage to recalibrate semantic features. Our two main contributions are: (1) a two-level Haar wavelet head for multi orientation edge extraction; and (2) Wavelet Edge Guided Attention (WEGA) modules that fuse wavelet cues with boundary and input branches. On five public polyp datasets, MEGANet-W consistently outperforms existing methods, improving mIoU by up to 2.3% and mDice by 1.2%, while introducing no additional learnable parameters.
>
---
#### [replaced 085] DriveMRP: Enhancing Vision-Language Models with Synthetic Motion Data for Motion Risk Prediction
- **分类: cs.CV; cs.AI; cs.RO; I.4.8; I.2.7; I.2.10**

- **链接: [http://arxiv.org/pdf/2507.02948v3](http://arxiv.org/pdf/2507.02948v3)**

> **作者:** Zhiyi Hou; Enhui Ma; Fang Li; Zhiyi Lai; Kalok Ho; Zhanqian Wu; Lijun Zhou; Long Chen; Chitian Sun; Haiyang Sun; Bing Wang; Guang Chen; Hangjun Ye; Kaicheng Yu
>
> **备注:** 12 pages, 4 figures. Code available at https://github.com/hzy138/DriveMRP
>
> **摘要:** Autonomous driving has seen significant progress, driven by extensive real-world data. However, in long-tail scenarios, accurately predicting the safety of the ego vehicle's future motion remains a major challenge due to uncertainties in dynamic environments and limitations in data coverage. In this work, we aim to explore whether it is possible to enhance the motion risk prediction capabilities of Vision-Language Models (VLM) by synthesizing high-risk motion data. Specifically, we introduce a Bird's-Eye View (BEV) based motion simulation method to model risks from three aspects: the ego-vehicle, other vehicles, and the environment. This allows us to synthesize plug-and-play, high-risk motion data suitable for VLM training, which we call DriveMRP-10K. Furthermore, we design a VLM-agnostic motion risk estimation framework, named DriveMRP-Agent. This framework incorporates a novel information injection strategy for global context, ego-vehicle perspective, and trajectory projection, enabling VLMs to effectively reason about the spatial relationships between motion waypoints and the environment. Extensive experiments demonstrate that by fine-tuning with DriveMRP-10K, our DriveMRP-Agent framework can significantly improve the motion risk prediction performance of multiple VLM baselines, with the accident recognition accuracy soaring from 27.13% to 88.03%. Moreover, when tested via zero-shot evaluation on an in-house real-world high-risk motion dataset, DriveMRP-Agent achieves a significant performance leap, boosting the accuracy from base_model's 29.42% to 68.50%, which showcases the strong generalization capabilities of our method in real-world scenarios.
>
---
#### [replaced 086] MPG-SAM 2: Adapting SAM 2 with Mask Priors and Global Context for Referring Video Object Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.13667v4](http://arxiv.org/pdf/2501.13667v4)**

> **作者:** Fu Rong; Meng Lan; Qian Zhang; Lefei Zhang
>
> **备注:** ICCV 2025
>
> **摘要:** Referring video object segmentation (RVOS) aims to segment objects in a video according to textual descriptions, which requires the integration of multimodal information and temporal dynamics perception. The Segment Anything Model 2 (SAM 2) has shown great effectiveness across various video segmentation tasks. However, its application to offline RVOS is challenged by the translation of the text into effective prompts and a lack of global context awareness. In this paper, we propose a novel RVOS framework, termed MPG-SAM 2, to address these challenges. Specifically, MPG-SAM 2 employs a unified multimodal encoder to jointly encode video and textual features, generating semantically aligned video and text embeddings, along with multimodal class tokens. A mask prior generator utilizes the video embeddings and class tokens to create pseudo masks of target objects and global context. These masks are fed into the prompt encoder as dense prompts along with multimodal class tokens as sparse prompts to generate accurate prompts for SAM 2. To provide the online SAM 2 with a global view, we introduce a hierarchical global-historical aggregator, which allows SAM 2 to aggregate global and historical information of target objects at both pixel and object levels, enhancing the target representation and temporal consistency. Extensive experiments on several RVOS benchmarks demonstrate the superiority of MPG-SAM 2 and the effectiveness of our proposed modules. The code is available at https://github.com/rongfu-dsb/MPG-SAM2.
>
---
#### [replaced 087] PyVision: Agentic Vision with Dynamic Tooling
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.07998v2](http://arxiv.org/pdf/2507.07998v2)**

> **作者:** Shitian Zhao; Haoquan Zhang; Shaoheng Lin; Ming Li; Qilong Wu; Kaipeng Zhang; Chen Wei
>
> **备注:** 26 Pages, 10 Figures, Technical report
>
> **摘要:** LLMs are increasingly deployed as agents, systems capable of planning, reasoning, and dynamically calling external tools. However, in visual reasoning, prior approaches largely remain limited by predefined workflows and static toolsets. In this report, we present PyVision, an interactive, multi-turn framework that enables MLLMs to autonomously generate, execute, and refine Python-based tools tailored to the task at hand, unlocking flexible and interpretable problem-solving. We develop a taxonomy of the tools created by PyVision and analyze their usage across a diverse set of benchmarks. Quantitatively, PyVision achieves consistent performance gains, boosting GPT-4.1 by +7.8% on V* and Claude-4.0-Sonnet by +31.1% on VLMsAreBlind-mini. These results point to a broader shift: dynamic tooling allows models not just to use tools, but to invent them, advancing toward more agentic visual reasoning.
>
---
#### [replaced 088] Comprehensive Evaluation of OCT-based Automated Segmentation of Retinal Layer, Fluid and Hyper-Reflective Foci: Impact on Clinical Assessment of Diabetic Retinopathy Severity
- **分类: eess.IV; cs.CV; cs.LG; q-bio.TO**

- **链接: [http://arxiv.org/pdf/2503.01248v4](http://arxiv.org/pdf/2503.01248v4)**

> **作者:** S. Chen; D. Ma; M. Raviselvan; S. Sundaramoorthy; K. Popuri; M. J. Ju; M. V. Sarunic; D. Ratra; M. F. Beg
>
> **备注:** 18 pages, 11 figures
>
> **摘要:** Diabetic retinopathy (DR) is a leading cause of vision loss, requiring early and accurate assessment to prevent irreversible damage. Spectral Domain Optical Coherence Tomography (SD-OCT) enables high-resolution retinal imaging, but automated segmentation performance varies, especially in cases with complex fluid and hyperreflective foci (HRF) patterns. This study proposes an active-learning-based deep learning pipeline for automated segmentation of retinal layers, fluid, and HRF, using four state-of-the-art models: U-Net, SegFormer, SwinUNETR, and VM-UNet, trained on expert-annotated SD-OCT volumes. Segmentation accuracy was evaluated with five-fold cross-validation, and retinal thickness was quantified using a K-nearest neighbors algorithm and visualized with Early Treatment Diabetic Retinopathy Study (ETDRS) maps. SwinUNETR achieved the highest overall accuracy (DSC = 0.7719; NSD = 0.8149), while VM-UNet excelled in specific layers. Structural differences were observed between non-proliferative and proliferative DR, with layer-specific thickening correlating with visual acuity impairment. The proposed framework enables robust, clinically relevant DR assessment while reducing the need for manual annotation, supporting improved disease monitoring and treatment planning.
>
---
#### [replaced 089] AHCPTQ: Accurate and Hardware-Compatible Post-Training Quantization for Segment Anything Model
- **分类: cs.CV; cs.AR; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.03088v3](http://arxiv.org/pdf/2503.03088v3)**

> **作者:** Wenlun Zhang; Yunshan Zhong; Shimpei Ando; Kentaro Yoshioka
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** The Segment Anything Model (SAM) has demonstrated strong versatility across various visual tasks. However, its large storage requirements and high computational cost pose challenges for practical deployment. Post-training quantization (PTQ) has emerged as an effective strategy for efficient deployment, but we identify two key challenges in SAM that hinder the effectiveness of existing PTQ methods: the heavy-tailed and skewed distribution of post-GELU activations, and significant inter-channel variation in linear projection activations. To address these challenges, we propose AHCPTQ, an accurate and hardware-efficient PTQ method for SAM. AHCPTQ introduces hardware-compatible Hybrid Log-Uniform Quantization (HLUQ) to manage post-GELU activations, employing log2 quantization for dense small values and uniform quantization for sparse large values to enhance quantization resolution. Additionally, AHCPTQ incorporates Channel-Aware Grouping (CAG) to mitigate inter-channel variation by progressively clustering activation channels with similar distributions, enabling them to share quantization parameters and improving hardware efficiency. The combination of HLUQ and CAG not only enhances quantization effectiveness but also ensures compatibility with efficient hardware execution. For instance, under the W4A4 configuration on the SAM-L model, AHCPTQ achieves 36.6% mAP on instance segmentation with the DINO detector, while achieving a 7.89x speedup and 8.64x energy efficiency over its floating-point counterpart in FPGA implementation.
>
---
#### [replaced 090] SCOOTER: A Human Evaluation Framework for Unrestricted Adversarial Examples
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.07776v2](http://arxiv.org/pdf/2507.07776v2)**

> **作者:** Dren Fazlija; Monty-Maximilian Zühlke; Johanna Schrader; Arkadij Orlov; Clara Stein; Iyiola E. Olatunji; Daniel Kudenko
>
> **备注:** 42 pages, 16 figures, 11 tables, Under Review, Code: https://github.com/DrenFazlija/Scooter, Data: https://doi.org/10.5281/zenodo.15771501
>
> **摘要:** Unrestricted adversarial attacks aim to fool computer vision models without being constrained by $\ell_p$-norm bounds to remain imperceptible to humans, for example, by changing an object's color. This allows attackers to circumvent traditional, norm-bounded defense strategies such as adversarial training or certified defense strategies. However, due to their unrestricted nature, there are also no guarantees of norm-based imperceptibility, necessitating human evaluations to verify just how authentic these adversarial examples look. While some related work assesses this vital quality of adversarial attacks, none provide statistically significant insights. This issue necessitates a unified framework that supports and streamlines such an assessment for evaluating and comparing unrestricted attacks. To close this gap, we introduce SCOOTER - an open-source, statistically powered framework for evaluating unrestricted adversarial examples. Our contributions are: $(i)$ best-practice guidelines for crowd-study power, compensation, and Likert equivalence bounds to measure imperceptibility; $(ii)$ the first large-scale human vs. model comparison across 346 human participants showing that three color-space attacks and three diffusion-based attacks fail to produce imperceptible images. Furthermore, we found that GPT-4o can serve as a preliminary test for imperceptibility, but it only consistently detects adversarial examples for four out of six tested attacks; $(iii)$ open-source software tools, including a browser-based task template to collect annotations and analysis scripts in Python and R; $(iv)$ an ImageNet-derived benchmark dataset containing 3K real images, 7K adversarial examples, and over 34K human ratings. Our findings demonstrate that automated vision systems do not align with human perception, reinforcing the need for a ground-truth SCOOTER benchmark.
>
---
#### [replaced 091] MEDTalk: Multimodal Controlled 3D Facial Animation with Dynamic Emotions by Disentangled Embedding
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2507.06071v2](http://arxiv.org/pdf/2507.06071v2)**

> **作者:** Chang Liu; Ye Pan; Chenyang Ding; Susanto Rahardja; Xiaokang Yang
>
> **备注:** 11 pages, 8 figures
>
> **摘要:** Audio-driven emotional 3D facial animation aims to generate synchronized lip movements and vivid facial expressions. However, most existing approaches focus on static and predefined emotion labels, limiting their diversity and naturalness. To address these challenges, we propose MEDTalk, a novel framework for fine-grained and dynamic emotional talking head generation. Our approach first disentangles content and emotion embedding spaces from motion sequences using a carefully designed cross-reconstruction process, enabling independent control over lip movements and facial expressions. Beyond conventional audio-driven lip synchronization, we integrate audio and speech text, predicting frame-wise intensity variations and dynamically adjusting static emotion features to generate realistic emotional expressions. Furthermore, to enhance control and personalization, we incorporate multimodal inputs-including text descriptions and reference expression images-to guide the generation of user-specified facial expressions. With MetaHuman as the priority, our generated results can be conveniently integrated into the industrial production pipeline.
>
---
#### [replaced 092] CLiFT: Compressive Light-Field Tokens for Compute-Efficient and Adaptive Neural Rendering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.08776v2](http://arxiv.org/pdf/2507.08776v2)**

> **作者:** Zhengqing Wang; Yuefan Wu; Jiacheng Chen; Fuyang Zhang; Yasutaka Furukawa
>
> **备注:** Project page: https://clift-nvs.github.io
>
> **摘要:** This paper proposes a neural rendering approach that represents a scene as "compressed light-field tokens (CLiFTs)", retaining rich appearance and geometric information of a scene. CLiFT enables compute-efficient rendering by compressed tokens, while being capable of changing the number of tokens to represent a scene or render a novel view with one trained network. Concretely, given a set of images, multi-view encoder tokenizes the images with the camera poses. Latent-space K-means selects a reduced set of rays as cluster centroids using the tokens. The multi-view ``condenser'' compresses the information of all the tokens into the centroid tokens to construct CLiFTs. At test time, given a target view and a compute budget (i.e., the number of CLiFTs), the system collects the specified number of nearby tokens and synthesizes a novel view using a compute-adaptive renderer. Extensive experiments on RealEstate10K and DL3DV datasets quantitatively and qualitatively validate our approach, achieving significant data reduction with comparable rendering quality and the highest overall rendering score, while providing trade-offs of data size, rendering quality, and rendering speed.
>
---
#### [replaced 093] WaveNet-SF: A Hybrid Network for Retinal Disease Detection Based on Wavelet Transform in the Spatial-Frequency Domain
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.11854v2](http://arxiv.org/pdf/2501.11854v2)**

> **作者:** Jilan Cheng; Guoli Long; Zeyu Zhang; Zhenjia Qi; Hanyu Wang; Libin Lu; Shuihua Wang; Yudong Zhang; Jin Hong
>
> **摘要:** Retinal diseases are a leading cause of vision impairment and blindness, with timely diagnosis being critical for effective treatment. Optical Coherence Tomography (OCT) has become a standard imaging modality for retinal disease diagnosis, but OCT images often suffer from issues such as speckle noise, complex lesion shapes, and varying lesion sizes, making interpretation challenging. In this paper, we propose a novel framework, WaveNet-SF, to enhance retinal disease detection by integrating the spatial-domain and frequency-domain learning. The framework utilizes wavelet transforms to decompose OCT images into low- and high-frequency components, enabling the model to extract both global structural features and fine-grained details. To improve lesion detection, we introduce a Multi-Scale Wavelet Spatial Attention (MSW-SA) module, which enhances the model's focus on regions of interest at multiple scales. Additionally, a High-Frequency Feature Compensation (HFFC) block is incorporated to recover edge information lost during wavelet decomposition, suppress noise, and preserve fine details crucial for lesion detection. Our approach achieves state-of-the-art (SOTA) classification accuracies of 97.82% and 99.58% on the OCT-C8 and OCT2017 datasets, respectively, surpassing existing methods. These results demonstrate the efficacy of WaveNet-SF in addressing the challenges of OCT image analysis and its potential as a powerful tool for retinal disease diagnosis.
>
---
#### [replaced 094] Explaining the Impact of Training on Vision Models via Activation Clustering
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.19700v4](http://arxiv.org/pdf/2411.19700v4)**

> **作者:** Ahcène Boubekki; Samuel G. Fadel; Sebastian Mair
>
> **摘要:** This paper introduces Neuro-Activated Vision Explanations (NAVE), a method for extracting and visualizing the internal representations of vision model encoders. By clustering feature activations, NAVE provides insights into learned semantics without fine-tuning. Using object localization, we show that NAVE's concepts align with image semantics. Through extensive experiments, we analyze the impact of training strategies and architectures on encoder representation capabilities. Additionally, we apply NAVE to study training artifacts in vision transformers and reveal how weak training strategies and spurious correlations degrade model performance. Our findings establish NAVE as a valuable tool for post-hoc model inspection and improving transparency in vision models.
>
---
#### [replaced 095] On the development of an AI performance and behavioural measures for teaching and classroom management
- **分类: cs.CV; H.5; J.4; I.2.7; I.2.10**

- **链接: [http://arxiv.org/pdf/2506.11143v2](http://arxiv.org/pdf/2506.11143v2)**

> **作者:** Andreea I. Niculescu; Jochen Ehnes; Chen Yi; Du Jiawei; Tay Chiat Pin; Joey Tianyi Zhou; Vigneshwaran Subbaraju; Teh Kah Kuan; Tran Huy Dat; John Komar; Gi Soong Chee; Kenneth Kwok
>
> **备注:** 7 pages, 10 figures, A video demonstration of the teacher trainer dashboard can be accessed here: https://vimeo.com/1076482827
>
> **摘要:** This paper presents a two-year research project focused on developing AI-driven measures to analyze classroom dynamics, with particular emphasis on teacher actions captured through multimodal sensor data. We applied real-time data from classroom sensors and AI techniques to extract meaningful insights and support teacher development. Key outcomes include a curated audio-visual dataset, novel behavioral measures, and a proof-of-concept teaching review dashboard. An initial evaluation with eight researchers from the National Institute for Education (NIE) highlighted the system's clarity, usability, and its non-judgmental, automated analysis approach -- which reduces manual workloads and encourages constructive reflection. Although the current version does not assign performance ratings, it provides an objective snapshot of in-class interactions, helping teachers recognize and improve their instructional strategies. Designed and tested in an Asian educational context, this work also contributes a culturally grounded methodology to the growing field of AI-based educational analytics.
>
---
#### [replaced 096] RadIR: A Scalable Framework for Multi-Grained Medical Image Retrieval via Radiology Report Mining
- **分类: cs.CV; cs.IR; eess.IV**

- **链接: [http://arxiv.org/pdf/2503.04653v2](http://arxiv.org/pdf/2503.04653v2)**

> **作者:** Tengfei Zhang; Ziheng Zhao; Chaoyi Wu; Xiao Zhou; Ya Zhang; Yanfeng Wang; Weidi Xie
>
> **摘要:** Developing advanced medical imaging retrieval systems is challenging due to the varying definitions of `similar images' across different medical contexts. This challenge is compounded by the lack of large-scale, high-quality medical imaging retrieval datasets and benchmarks. In this paper, we propose a novel methodology that leverages dense radiology reports to define image-wise similarity ordering at multiple granularities in a scalable and fully automatic manner. Using this approach, we construct two comprehensive medical imaging retrieval datasets: MIMIC-IR for Chest X-rays and CTRATE-IR for CT scans, providing detailed image-image ranking annotations conditioned on diverse anatomical structures. Furthermore, we develop two retrieval systems, RadIR-CXR and model-ChestCT, which demonstrate superior performance in traditional image-image and image-report retrieval tasks. These systems also enable flexible, effective image retrieval conditioned on specific anatomical structures described in text, achieving state-of-the-art results on 77 out of 78 metrics.
>
---
#### [replaced 097] SpatialViz-Bench: Automatically Generated Spatial Visualization Reasoning Tasks for MLLMs
- **分类: cs.CV; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2507.07610v2](http://arxiv.org/pdf/2507.07610v2)**

> **作者:** Siting Wang; Luoyang Sun; Cheng Deng; Kun Shao; Minnan Pei; Zheng Tian; Haifeng Zhang; Jun Wang
>
> **摘要:** Humans can directly imagine and manipulate visual images in their minds, a capability known as spatial visualization. While multi-modal Large Language Models (MLLMs) support imagination-based reasoning, spatial visualization remains insufficiently evaluated, typically embedded within broader mathematical and logical assessments. Existing evaluations often rely on IQ tests or math competitions that may overlap with training data, compromising assessment reliability. To this end, we introduce SpatialViz-Bench, a comprehensive multi-modal benchmark for spatial visualization with 12 tasks across 4 sub-abilities, comprising 1,180 automatically generated problems. Our evaluation of 33 state-of-the-art MLLMs not only reveals wide performance variations and demonstrates the benchmark's strong discriminative power, but also uncovers counter-intuitive findings: models exhibit unexpected behaviors by showing difficulty perception that misaligns with human intuition, displaying dramatic 2D-to-3D performance cliffs, and defaulting to formula derivation despite spatial tasks requiring visualization alone. SpatialVizBench empirically demonstrates that state-of-the-art MLLMs continue to exhibit deficiencies in spatial visualization tasks, thereby addressing a significant lacuna in the field. The benchmark is publicly available.
>
---
#### [replaced 098] Evaluating the Role of Training Data Origin for Country-Scale Cropland Mapping in Data-Scarce Regions: A Case Study of Nigeria
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2312.10872v2](http://arxiv.org/pdf/2312.10872v2)**

> **作者:** Joaquin Gajardo; Michele Volpi; Daniel Onwude; Thijs Defraeye
>
> **备注:** This article is published in ISPRS Open Journal of Photogrammetry and Remote Sensing under a CC BY 4.0 license: https://www.sciencedirect.com/science/article/pii/S2667393225000109. Code repository: https://github.com/Joaquin-Gajardo/nigeria-crop-mask
>
> **摘要:** Cropland maps are essential for remote sensing-based agricultural monitoring, providing timely insights without extensive field surveys. Machine learning enables large-scale mapping but depends on geo-referenced ground-truth data, which is costly to collect, motivating the use of global datasets in data-scarce regions. A key challenge is understanding how the quantity, quality, and proximity of the training data to the target region influences model performance. We evaluate this in Nigeria, using 1,827 manually labelled samples covering the whole country, and subsets of the Geowiki dataset: Nigeria-only, regional (Nigeria and neighbouring countries), and global. We extract pixel-wise multi-source time series arrays from Sentinel-1, Sentinel-2, ERA5 climate, and a digital elevation model using Google Earth Engine, comparing Random Forests with LSTMs, including a lightweight multi-headed LSTM variant. Results show local data significantly boosts performance, with accuracy gains up to 0.246 (RF) and 0.178 (LSTM). Nigeria-only or regional data outperformed global data despite the lower amount of labels, with the exception of the multi-headed LSTM, which benefited from global data when local samples were absent. Sentinel-1, climate, and topographic data are critical data sources, with their removal reducing F1-score by up to 0.593. Addressing class imbalance also improved LSTM accuracy by up to 0.071. Our top-performing model (Nigeria-only LSTM) achieved an F1-score of 0.814 and accuracy of 0.842, matching the best global land cover product while offering stronger recall, critical for food security. We release code, data, maps, and an interactive web app to support future work.
>
---
#### [replaced 099] Pisces: An Auto-regressive Foundation Model for Image Understanding and Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.10395v2](http://arxiv.org/pdf/2506.10395v2)**

> **作者:** Zhiyang Xu; Jiuhai Chen; Zhaojiang Lin; Xichen Pan; Lifu Huang; Tianyi Zhou; Madian Khabsa; Qifan Wang; Di Jin; Michihiro Yasunaga; Lili Yu; Xi Victoria Lin; Shaoliang Nie
>
> **备注:** Unified image understanding and generation model
>
> **摘要:** Recent advances in large language models (LLMs) have enabled multimodal foundation models to tackle both image understanding and generation within a unified framework. Despite these gains, unified models often underperform compared to specialized models in either task. A key challenge in developing unified models lies in the inherent differences between the visual features needed for image understanding versus generation, as well as the distinct training processes required for each modality. In this work, we introduce Pisces, an auto-regressive multimodal foundation model that addresses this challenge through a novel decoupled visual encoding architecture and tailored training techniques optimized for multimodal generation. Combined with meticulous data curation, pretraining, and finetuning, Pisces achieves competitive performance in both image understanding and image generation. We evaluate Pisces on over 20 public benchmarks for image understanding, where it demonstrates strong performance across a wide range of tasks. Additionally, on GenEval, a widely adopted benchmark for image generation, Pisces exhibits robust generative capabilities. Our extensive analysis reveals the synergistic relationship between image understanding and generation, and the benefits of using separate visual encoders, advancing the field of unified multimodal models.
>
---
#### [replaced 100] LLaVA-CoT: Let Vision Language Models Reason Step-by-Step
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.10440v5](http://arxiv.org/pdf/2411.10440v5)**

> **作者:** Guowei Xu; Peng Jin; Ziang Wu; Hao Li; Yibing Song; Lichao Sun; Li Yuan
>
> **备注:** 17 pages, ICCV 2025
>
> **摘要:** Large language models have demonstrated substantial advancements in reasoning capabilities. However, current Vision-Language Models (VLMs) often struggle to perform systematic and structured reasoning, especially when handling complex visual question-answering tasks. In this work, we introduce LLaVA-CoT, a large VLM designed to conduct autonomous multistage reasoning. Unlike chain-of-thought prompting, LLaVA-CoT independently engages in sequential stages of summarization, visual interpretation, logical reasoning, and conclusion generation. This structured approach enables LLaVA-CoT to achieve marked improvements on reasoning-intensive tasks. To accomplish this, we construct the LLaVA-CoT-100k dataset, integrating samples from various visual question answering sources and providing structured reasoning annotations. Besides, we propose a test-time stage-wise retracing search method (SWIRES), which enables effective and efficient test-time scaling. Remarkably, with only 100k training samples and test-time scaling, LLaVA-CoT not only outperforms its base model by 9.4% on a wide range of multimodal reasoning benchmarks, but also surpasses the performance of larger and even closed-source models, such as Gemini-1.5-pro, GPT-4o-mini, and Llama-3.2-90B-Vision-Instruct. The code, dataset, and pre-trained weights are publicly available at https://github.com/PKU-YuanGroup/LLaVA-CoT.
>
---
#### [replaced 101] MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes
- **分类: cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2410.13613v2](http://arxiv.org/pdf/2410.13613v2)**

> **作者:** Xinjie Zhang; Zhening Liu; Yifan Zhang; Xingtong Ge; Dailan He; Tongda Xu; Yan Wang; Zehong Lin; Shuicheng Yan; Jun Zhang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** 4D Gaussian Splatting (4DGS) has recently emerged as a promising technique for capturing complex dynamic 3D scenes with high fidelity. It utilizes a 4D Gaussian representation and a GPU-friendly rasterizer, enabling rapid rendering speeds. Despite its advantages, 4DGS faces significant challenges, notably the requirement of millions of 4D Gaussians, each with extensive associated attributes, leading to substantial memory and storage cost. This paper introduces a memory-efficient framework for 4DGS. We streamline the color attribute by decomposing it into a per-Gaussian direct color component with only 3 parameters and a shared lightweight alternating current color predictor. This approach eliminates the need for spherical harmonics coefficients, which typically involve up to 144 parameters in classic 4DGS, thereby creating a memory-efficient 4D Gaussian representation. Furthermore, we introduce an entropy-constrained Gaussian deformation technique that uses a deformation field to expand the action range of each Gaussian and integrates an opacity-based entropy loss to limit the number of Gaussians, thus forcing our model to use as few Gaussians as possible to fit a dynamic scene well. With simple half-precision storage and zip compression, our framework achieves a storage reduction by approximately 190$\times$ and 125$\times$ on the Technicolor and Neural 3D Video datasets, respectively, compared to the original 4DGS. Meanwhile, it maintains comparable rendering speeds and scene representation quality, setting a new standard in the field. Code is available at https://github.com/Xinjie-Q/MEGA.
>
---
#### [replaced 102] Deflickering Vision-Based Occupancy Networks through Lightweight Spatio-Temporal Correlation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.15438v3](http://arxiv.org/pdf/2502.15438v3)**

> **作者:** Fengcheng Yu; Haoran Xu; Canming Xia; Ziyang Zong; Guang Tan
>
> **摘要:** Vision-based occupancy networks (VONs) provide an end-to-end solution for reconstructing 3D environments in autonomous driving. However, existing methods often suffer from temporal inconsistencies, manifesting as flickering effects that compromise visual experience and adversely affect decision-making. While recent approaches have incorporated historical data to mitigate the issue, they often incur high computational costs and may introduce noisy information that interferes with object detection. We propose OccLinker, a novel plugin framework designed to seamlessly integrate with existing VONs for boosting performance. Our method efficiently consolidates historical static and motion cues, learns sparse latent correlations with current features through a dual cross-attention mechanism, and produces correction occupancy components to refine the base network's predictions. We propose a new temporal consistency metric to quantitatively identify flickering effects. Extensive experiments on two benchmark datasets demonstrate that our method delivers superior performance with negligible computational overhead, while effectively eliminating flickering artifacts.
>
---
#### [replaced 103] GaussianOcc: Fully Self-supervised and Efficient 3D Occupancy Estimation with Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.11447v4](http://arxiv.org/pdf/2408.11447v4)**

> **作者:** Wanshui Gan; Fang Liu; Hongbin Xu; Ningkai Mo; Naoto Yokoya
>
> **备注:** Project page: https://ganwanshui.github.io/GaussianOcc/
>
> **摘要:** We introduce GaussianOcc, a systematic method that investigates the two usages of Gaussian splatting for fully self-supervised and efficient 3D occupancy estimation in surround views. First, traditional methods for self-supervised 3D occupancy estimation still require ground truth 6D poses from sensors during training. To address this limitation, we propose Gaussian Splatting for Projection (GSP) module to provide accurate scale information for fully self-supervised training from adjacent view projection. Additionally, existing methods rely on volume rendering for final 3D voxel representation learning using 2D signals (depth maps, semantic maps), which is both time-consuming and less effective. We propose Gaussian Splatting from Voxel space (GSV) to leverage the fast rendering properties of Gaussian splatting. As a result, the proposed GaussianOcc method enables fully self-supervised (no ground truth pose) 3D occupancy estimation in competitive performance with low computational cost (2.7 times faster in training and 5 times faster in rendering). The relevant code is available in https://github.com/GANWANSHUI/GaussianOcc.git.
>
---
#### [replaced 104] Sparfels: Fast Reconstruction from Sparse Unposed Imagery
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.02178v2](http://arxiv.org/pdf/2505.02178v2)**

> **作者:** Shubhendu Jena; Amine Ouasfi; Mae Younes; Adnane Boukhayma
>
> **备注:** ICCV 2025. Project page : https://shubhendu-jena.github.io/Sparfels-web/
>
> **摘要:** We present a method for Sparse view reconstruction with surface element splatting that runs within 3 minutes on a consumer grade GPU. While few methods address sparse radiance field learning from noisy or unposed sparse cameras, shape recovery remains relatively underexplored in this setting. Several radiance and shape learning test-time optimization methods address the sparse posed setting by learning data priors or using combinations of external monocular geometry priors. Differently, we propose an efficient and simple pipeline harnessing a single recent 3D foundation model. We leverage its various task heads, notably point maps and camera initializations to instantiate a bundle adjusting 2D Gaussian Splatting (2DGS) model, and image correspondences to guide camera optimization midst 2DGS training. Key to our contribution is a novel formulation of splatted color variance along rays, which can be computed efficiently. Reducing this moment in training leads to more accurate shape reconstructions. We demonstrate state-of-the-art performances in the sparse uncalibrated setting in reconstruction and novel view benchmarks based on established multi-view datasets.
>
---
#### [replaced 105] AGAV-Rater: Adapting Large Multimodal Model for AI-Generated Audio-Visual Quality Assessment
- **分类: cs.MM; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2501.18314v2](http://arxiv.org/pdf/2501.18314v2)**

> **作者:** Yuqin Cao; Xiongkuo Min; Yixuan Gao; Wei Sun; Guangtao Zhai
>
> **摘要:** Many video-to-audio (VTA) methods have been proposed for dubbing silent AI-generated videos. An efficient quality assessment method for AI-generated audio-visual content (AGAV) is crucial for ensuring audio-visual quality. Existing audio-visual quality assessment methods struggle with unique distortions in AGAVs, such as unrealistic and inconsistent elements. To address this, we introduce AGAVQA-3k, the first large-scale AGAV quality assessment dataset, comprising $3,382$ AGAVs from $16$ VTA methods. AGAVQA-3k includes two subsets: AGAVQA-MOS, which provides multi-dimensional scores for audio quality, content consistency, and overall quality, and AGAVQA-Pair, designed for optimal AGAV pair selection. We further propose AGAV-Rater, a LMM-based model that can score AGAVs, as well as audio and music generated from text, across multiple dimensions, and selects the best AGAV generated by VTA methods to present to the user. AGAV-Rater achieves state-of-the-art performance on AGAVQA-3k, Text-to-Audio, and Text-to-Music datasets. Subjective tests also confirm that AGAV-Rater enhances VTA performance and user experience. The dataset and code is available at https://github.com/charlotte9524/AGAV-Rater.
>
---
#### [replaced 106] RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.10059v2](http://arxiv.org/pdf/2502.10059v2)**

> **作者:** Teng Li; Guangcong Zheng; Rui Jiang; Shuigen Zhan; Tao Wu; Yehao Lu; Yining Lin; Chuanyun Deng; Yepan Xiong; Min Chen; Lin Cheng; Xi Li
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Recent advancements in camera-trajectory-guided image-to-video generation offer higher precision and better support for complex camera control compared to text-based approaches. However, they also introduce significant usability challenges, as users often struggle to provide precise camera parameters when working with arbitrary real-world images without knowledge of their depth nor scene scale. To address these real-world application issues, we propose RealCam-I2V, a novel diffusion-based video generation framework that integrates monocular metric depth estimation to establish 3D scene reconstruction in a preprocessing step. During training, the reconstructed 3D scene enables scaling camera parameters from relative to metric scales, ensuring compatibility and scale consistency across diverse real-world images. In inference, RealCam-I2V offers an intuitive interface where users can precisely draw camera trajectories by dragging within the 3D scene. To further enhance precise camera control and scene consistency, we propose scene-constrained noise shaping, which shapes high-level noise and also allows the framework to maintain dynamic and coherent video generation in lower noise stages. RealCam-I2V achieves significant improvements in controllability and video quality on the RealEstate10K and out-of-domain images. We further enables applications like camera-controlled looping video generation and generative frame interpolation. Project page: https://zgctroy.github.io/RealCam-I2V.
>
---
#### [replaced 107] HANDI: Hand-Centric Text-and-Image Conditioned Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.04189v5](http://arxiv.org/pdf/2412.04189v5)**

> **作者:** Yayuan Li; Zhi Cao; Jason J. Corso
>
> **备注:** 16 pages, 7 figures and 4 tables
>
> **摘要:** Despite the recent strides in video generation, state-of-the-art methods still struggle with elements of visual detail. One particularly challenging case is the class of videos in which the intricate motion of the hand coupled with a mostly stable and otherwise distracting environment is necessary to convey the execution of some complex action and its effects. To address these challenges, we introduce a new method for video generation that focuses on hand-centric actions. Our diffusion-based method incorporates two distinct innovations. First, we propose an automatic method to generate the motion area -- the region in the video in which the detailed activities occur -- guided by both the visual context and the action text prompt, rather than assuming this region can be provided manually as is now commonplace. Second, we introduce a critical Hand Refinement Loss to guide the diffusion model to focus on smooth and consistent hand poses. We evaluate our method on challenging augmented datasets based on EpicKitchens and Ego4D, demonstrating significant improvements over state-of-the-art methods in terms of action clarity, especially of the hand motion in the target region, across diverse environments and actions. Video results can be found in https://excitedbutter.github.io/project_page
>
---
#### [replaced 108] When Small Guides Large: Cross-Model Co-Learning for Test-Time Adaptation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.23724v2](http://arxiv.org/pdf/2506.23724v2)**

> **作者:** Chang'an Yi; Xiaohui Deng; Guohao Chen; Yan Zhou; Qinghua Lu; Shuaicheng Niu
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** Test-time Adaptation (TTA) adapts a given model to testing domain data with potential domain shifts through online unsupervised learning, yielding impressive performance. However, to date, existing TTA methods primarily focus on single-model adaptation. In this work, we investigate an intriguing question: how does cross-model knowledge influence the TTA process? Our findings reveal that, in TTA's unsupervised online setting, each model can provide complementary, confident knowledge to the others, even when there are substantial differences in model size. For instance, a smaller model like MobileViT (10.6M parameters) can effectively guide a larger model like ViT-Base (86.6M parameters). In light of this, we propose COCA, a Cross-Model Co-Learning framework for TTA, which mainly consists of two main strategies. 1) Co-adaptation adaptively integrates complementary knowledge from other models throughout the TTA process, reducing individual model biases. 2) Self-adaptation enhances each model's unique strengths via unsupervised learning, enabling diverse adaptation to the target domain. Extensive experiments show that COCA, which can also serve as a plug-and-play module, significantly boosts existing SOTAs, on models with various sizes--including ResNets, ViTs, and Mobile-ViTs--via cross-model co-learned TTA. For example, with Mobile-ViT's guidance, COCA raises ViT-Base's average adaptation accuracy on ImageNet-C from 51.7% to 64.5%. The code is publicly available at https://github.com/ycarobot/COCA.
>
---
#### [replaced 109] CorrCLIP: Reconstructing Patch Correlations in CLIP for Open-Vocabulary Semantic Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.10086v2](http://arxiv.org/pdf/2411.10086v2)**

> **作者:** Dengke Zhang; Fagui Liu; Quan Tang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Open-vocabulary semantic segmentation aims to assign semantic labels to each pixel without being constrained by a predefined set of categories. While Contrastive Language-Image Pre-training (CLIP) excels in zero-shot classification, it struggles to align image patches with category embeddings because of its incoherent patch correlations. This study reveals that inter-class correlations are the main reason for impairing CLIP's segmentation performance. Accordingly, we propose CorrCLIP, which reconstructs the scope and value of patch correlations. Specifically, CorrCLIP leverages the Segment Anything Model (SAM) to define the scope of patch interactions, reducing inter-class correlations. To mitigate the problem that SAM-generated masks may contain patches belonging to different classes, CorrCLIP incorporates self-supervised models to compute coherent similarity values, suppressing the weight of inter-class correlations. Additionally, we introduce two additional branches to strengthen patch features' spatial details and semantic representation. Finally, we update segmentation maps with SAM-generated masks to improve spatial consistency. Based on the improvement across patch correlations, feature representations, and segmentation maps, CorrCLIP achieves superior performance across eight benchmarks. Codes are available at: https://github.com/zdk258/CorrCLIP.
>
---
#### [replaced 110] Enhancing Underwater Imaging with 4-D Light Fields: Dataset and Method
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2408.17339v2](http://arxiv.org/pdf/2408.17339v2)**

> **作者:** Yuji Lin; Junhui Hou; Xianqiang Lyu; Qian Zhao; Deyu Meng
>
> **备注:** 20 pages, 22 figures
>
> **摘要:** In this paper, we delve into the realm of 4-D light fields (LFs) to enhance underwater imaging plagued by light absorption, scattering, and other challenges. Contrasting with conventional 2-D RGB imaging, 4-D LF imaging excels in capturing scenes from multiple perspectives, thereby indirectly embedding geometric information. This intrinsic property is anticipated to effectively address the challenges associated with underwater imaging. By leveraging both explicit and implicit depth cues present in 4-D LF images, we propose a progressive, mutually reinforcing framework for underwater 4-D LF image enhancement and depth estimation. Specifically, our framework explicitly utilizes estimated depth information alongside implicit depth-related dynamic convolutional kernels to modulate output features. The entire framework decomposes this complex task, iteratively optimizing the enhanced image and depth information to progressively achieve optimal enhancement results. More importantly, we construct the first 4-D LF-based underwater image dataset for quantitative evaluation and supervised training of learning-based methods, comprising 75 underwater scenes and 3675 high-resolution 2K pairs. To craft vibrant and varied underwater scenes, we build underwater environments with various objects and adopt several types of degradation. Through extensive experimentation, we showcase the potential and superiority of 4-D LF-based underwater imaging vis-a-vis traditional 2-D RGB-based approaches. Moreover, our method effectively corrects color bias and achieves state-of-the-art performance. The dataset and code will be publicly available at https://github.com/linlos1234/LFUIE.
>
---
#### [replaced 111] Auto-Regressively Generating Multi-View Consistent Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18527v2](http://arxiv.org/pdf/2506.18527v2)**

> **作者:** JiaKui Hu; Yuxiao Yang; Jialun Liu; Jinbo Wu; Chen Zhao; Yanye Lu
>
> **备注:** Accepted by ICCV 2025. Code is at https://github.com/MILab-PKU/MVAR
>
> **摘要:** Generating multi-view images from human instructions is crucial for 3D content creation. The primary challenges involve maintaining consistency across multiple views and effectively synthesizing shapes and textures under diverse conditions. In this paper, we propose the Multi-View Auto-Regressive (\textbf{MV-AR}) method, which leverages an auto-regressive model to progressively generate consistent multi-view images from arbitrary prompts. Firstly, the next-token-prediction capability of the AR model significantly enhances its effectiveness in facilitating progressive multi-view synthesis. When generating widely-separated views, MV-AR can utilize all its preceding views to extract effective reference information. Subsequently, we propose a unified model that accommodates various prompts via architecture designing and training strategies. To address multiple conditions, we introduce condition injection modules for text, camera pose, image, and shape. To manage multi-modal conditions simultaneously, a progressive training strategy is employed. This strategy initially adopts the text-to-multi-view (t2mv) model as a baseline to enhance the development of a comprehensive X-to-multi-view (X2mv) model through the randomly dropping and combining conditions. Finally, to alleviate the overfitting problem caused by limited high-quality data, we propose the ``Shuffle View" data augmentation technique, thus significantly expanding the training data by several magnitudes. Experiments demonstrate the performance and versatility of our MV-AR, which consistently generates consistent multi-view images across a range of conditions and performs on par with leading diffusion-based multi-view image generation models. The code and models are released at https://github.com/MILab-PKU/MVAR.
>
---
#### [replaced 112] An Efficient Deep Learning Framework for Brain Stroke Diagnosis Using Computed Tomography (CT) Images
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.03558v2](http://arxiv.org/pdf/2507.03558v2)**

> **作者:** Md. Sabbir Hossen; Eshat Ahmed Shuvo; Shibbir Ahmed Arif; Pabon Shaha; Md. Saiduzzaman; Mostofa Kamal Nasir
>
> **备注:** Preprint version. Submitted for peer review
>
> **摘要:** Brain stroke is a leading cause of mortality and long-term disability worldwide, underscoring the need for precise and rapid prediction techniques. Computed Tomography (CT) scan is considered one of the most effective methods for diagnosing brain strokes. Most stroke classification techniques use a single slice-level prediction mechanism, requiring radiologists to manually select the most critical CT slice from the original CT volume. Although clinical evaluations are often used in traditional diagnostic procedures, machine learning (ML) has opened up new avenues for improving stroke diagnosis. To supplement traditional diagnostic techniques, this study investigates machine learning models for early brain stroke prediction using CT scan images. This research proposes a novel machine learning approach to brain stroke detection, focusing on optimizing classification performance with pre-trained deep learning models and advanced optimization strategies. Pre-trained models, including DenseNet201, InceptionV3, MobileNetV2, ResNet50, and Xception, are used for feature extraction. Feature engineering techniques, including BFO, PCA, and LDA, further enhance model performance. These features are then classified using machine learning algorithms, including SVC, RF, XGB, DT, LR, KNN, and GNB. Our experiments demonstrate that the combination of MobileNetV2, LDA, and SVC achieved the highest classification accuracy of 97.93%, significantly outperforming other model-optimizer-classifier combinations. The results underline the effectiveness of integrating lightweight pre-trained models with robust optimization and classification techniques for brain stroke diagnosis.
>
---
#### [replaced 113] AI-driven visual monitoring of industrial assembly tasks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.15285v2](http://arxiv.org/pdf/2506.15285v2)**

> **作者:** Mattia Nardon; Stefano Messelodi; Antonio Granata; Fabio Poiesi; Alberto Danese; Davide Boscaini
>
> **摘要:** Visual monitoring of industrial assembly tasks is critical for preventing equipment damage due to procedural errors and ensuring worker safety. Although commercial solutions exist, they typically require rigid workspace setups or the application of visual markers to simplify the problem. We introduce ViMAT, a novel AI-driven system for real-time visual monitoring of assembly tasks that operates without these constraints. ViMAT combines a perception module that extracts visual observations from multi-view video streams with a reasoning module that infers the most likely action being performed based on the observed assembly state and prior task knowledge. We validate ViMAT on two assembly tasks, involving the replacement of LEGO components and the reconfiguration of hydraulic press molds, demonstrating its effectiveness through quantitative and qualitative analysis in challenging real-world scenarios characterized by partial and uncertain visual observations. Project page: https://tev-fbk.github.io/ViMAT
>
---
#### [replaced 114] Video Individual Counting for Moving Drones
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.10701v2](http://arxiv.org/pdf/2503.10701v2)**

> **作者:** Yaowu Fan; Jia Wan; Tao Han; Antoni B. Chan; Andy J. Ma
>
> **备注:** This work has been accepted to ICCV 2025
>
> **摘要:** Video Individual Counting (VIC) has received increasing attention for its importance in intelligent video surveillance. Existing works are limited in two aspects, i.e., dataset and method. Previous datasets are captured with fixed or rarely moving cameras with relatively sparse individuals, restricting evaluation for a highly varying view and time in crowded scenes. Existing methods rely on localization followed by association or classification, which struggle under dense and dynamic conditions due to inaccurate localization of small targets. To address these issues, we introduce the MovingDroneCrowd Dataset, featuring videos captured by fast-moving drones in crowded scenes under diverse illuminations, shooting heights and angles. We further propose a Shared Density map-guided Network (SDNet) using a Depth-wise Cross-Frame Attention (DCFA) module to directly estimate shared density maps between consecutive frames, from which the inflow and outflow density maps are derived by subtracting the shared density maps from the global density maps. The inflow density maps across frames are summed up to obtain the number of unique pedestrians in a video. Experiments on our datasets and publicly available ones show the superiority of our method over the state of the arts in highly dynamic and complex crowded scenes. Our dataset and codes have been released publicly.
>
---
#### [replaced 115] Pathfinder for Low-altitude Aircraft with Binary Neural Network
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.08824v4](http://arxiv.org/pdf/2409.08824v4)**

> **作者:** Kaijie Yin; Tian Gao; Hui Kong
>
> **摘要:** A prior global topological map (e.g., the OpenStreetMap, OSM) can boost the performance of autonomous mapping by a ground mobile robot. However, the prior map is usually incomplete due to lacking labeling in partial paths. To solve this problem, this paper proposes an OSM maker using airborne sensors carried by low-altitude aircraft, where the core of the OSM maker is a novel efficient pathfinder approach based on LiDAR and camera data, i.e., a binary dual-stream road segmentation model. Specifically, a multi-scale feature extraction based on the UNet architecture is implemented for images and point clouds. To reduce the effect caused by the sparsity of point cloud, an attention-guided gated block is designed to integrate image and point-cloud features. To optimize the model for edge deployment that significantly reduces storage footprint and computational demands, we propose a binarization streamline to each model component, including a variant of vision transformer (ViT) architecture as the encoder of the image branch, and new focal and perception losses to optimize the model training. The experimental results on two datasets demonstrate that our pathfinder method achieves SOTA accuracy with high efficiency in finding paths from the low-level airborne sensors, and we can create complete OSM prior maps based on the segmented road skeletons. Code and data are available at: \href{https://github.com/IMRL/Pathfinder}{https://github.com/IMRL/Pathfinder}.
>
---
#### [replaced 116] Towards Open-World Generation of Stereo Images and Unsupervised Matching
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.12720v2](http://arxiv.org/pdf/2503.12720v2)**

> **作者:** Feng Qiao; Zhexiao Xiong; Eric Xing; Nathan Jacobs
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Stereo images are fundamental to numerous applications, including extended reality (XR) devices, autonomous driving, and robotics. Unfortunately, acquiring high-quality stereo images remains challenging due to the precise calibration requirements of dual-camera setups and the complexity of obtaining accurate, dense disparity maps. Existing stereo image generation methods typically focus on either visual quality for viewing or geometric accuracy for matching, but not both. We introduce GenStereo, a diffusion-based approach, to bridge this gap. The method includes two primary innovations (1) conditioning the diffusion process on a disparity-aware coordinate embedding and a warped input image, allowing for more precise stereo alignment than previous methods, and (2) an adaptive fusion mechanism that intelligently combines the diffusion-generated image with a warped image, improving both realism and disparity consistency. Through extensive training on 11 diverse stereo datasets, GenStereo demonstrates strong generalization ability. GenStereo achieves state-of-the-art performance in both stereo image generation and unsupervised stereo matching tasks. Project page is available at https://qjizhi.github.io/genstereo.
>
---
#### [replaced 117] BoundMatch: Boundary detection applied to semi-supervised segmentation for urban-driving scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.23519v3](http://arxiv.org/pdf/2503.23519v3)**

> **作者:** Haruya Ishikawa; Yoshimitsu Aoki
>
> **备注:** 20 pages, 18 figures
>
> **摘要:** Semi-supervised semantic segmentation (SS-SS) aims to mitigate the heavy annotation burden of dense pixel labeling by leveraging abundant unlabeled images alongside a small labeled set. While current consistency regularization methods achieve strong results, they often overlook a critical challenge: the precise delineation of object boundaries. In this paper, we propose BoundMatch, a novel multi-task SS-SS framework that explicitly integrates semantic boundary detection into a teacher-student consistency regularization pipeline. Our core mechanism, Boundary Consistency Regularized Multi-Task Learning (BCRM), enforces prediction agreement between teacher and student models on both segmentation masks and detailed semantic boundaries. To further enhance performance and sharpen boundaries, BoundMatch incorporates two lightweight fusion modules: Boundary-Semantic Fusion (BSF) injects learned boundary cues into the segmentation decoder, while Spatial Gradient Fusion (SGF) refines boundary predictions using mask gradients, leading to higher-quality boundary pseudo-labels. This framework is built upon SAMTH, a strong teacher-student baseline featuring a Harmonious Batch Normalization (HBN) update strategy for improved stability. Extensive experiments on diverse urban-driving scene datasets including Cityscapes, BDD100K, and SYNTHIA show that BoundMatch achieves competitive performance against current state-of-the-art methods. Our approach achieves state-of-the-art results on the new benchmark with DINOv2 foundation model. We further validate our approach's generalizability on Pascal VOC and ADE20K datasets. Ablation studies highlight BoundMatch's ability to improve boundary-specific evaluation metrics, its effectiveness in realistic large-scale unlabeled data scenarios, and applicability to lightweight architectures for mobile deployment.
>
---
#### [replaced 118] Adaptive deep learning framework for robust unsupervised underwater image enhancement
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2212.08983v3](http://arxiv.org/pdf/2212.08983v3)**

> **作者:** Alzayat Saleh; Marcus Sheaves; Dean Jerry; Mostafa Rahimi Azghadi
>
> **备注:** 25 pages, 7 figures, 6 tables, accepted for publication in Expert Systems with Applications
>
> **摘要:** One of the main challenges in deep learning-based underwater image enhancement is the limited availability of high-quality training data. Underwater images are difficult to capture and are often of poor quality due to the distortion and loss of colour and contrast in water. This makes it difficult to train supervised deep learning models on large and diverse datasets, which can limit the model's performance. In this paper, we explore an alternative approach to supervised underwater image enhancement. Specifically, we propose a novel unsupervised underwater image enhancement framework that employs a conditional variational autoencoder (cVAE) to train a deep learning model with probabilistic adaptive instance normalization (PAdaIN) and statistically guided multi-colour space stretch that produces realistic underwater images. The resulting framework is composed of a U-Net as a feature extractor and a PAdaIN to encode the uncertainty, which we call UDnet. To improve the visual quality of the images generated by UDnet, we use a statistically guided multi-colour space stretch module that ensures visual consistency with the input image and provides an alternative to training using a ground truth image. The proposed model does not need manual human annotation and can learn with a limited amount of data and achieves state-of-the-art results on underwater images. We evaluated our proposed framework on eight publicly-available datasets. The results show that our proposed framework yields competitive performance compared to other state-of-the-art approaches in quantitative as well as qualitative metrics. Code available at https://github.com/alzayats/UDnet .
>
---
#### [replaced 119] A Practical Approach to Underwater Depth and Surface Normals Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.02072v2](http://arxiv.org/pdf/2410.02072v2)**

> **作者:** Alzayat Saleh; Melanie Olsen; Bouchra Senadji; Mostafa Rahimi Azghadi
>
> **备注:** 18 pages, 6 figures, 8 tables. Submitted to Elsevier
>
> **摘要:** Monocular Depth and Surface Normals Estimation (MDSNE) is crucial for tasks such as 3D reconstruction, autonomous navigation, and underwater exploration. Current methods rely either on discriminative models, which struggle with transparent or reflective surfaces, or generative models, which, while accurate, are computationally expensive. This paper presents a novel deep learning model for MDSNE, specifically tailored for underwater environments, using a hybrid architecture that integrates Convolutional Neural Networks (CNNs) with Transformers, leveraging the strengths of both approaches. Training effective MDSNE models is often hampered by noisy real-world datasets and the limited generalization of synthetic datasets. To address this, we generate pseudo-labeled real data using multiple pre-trained MDSNE models. To ensure the quality of this data, we propose the Depth Normal Evaluation and Selection Algorithm (DNESA), which evaluates and selects the most reliable pseudo-labeled samples using domain-specific metrics. A lightweight student model is then trained on this curated dataset. Our model reduces parameters by 90% and training costs by 80%, allowing real-time 3D perception on resource-constrained devices. Key contributions include: a novel and efficient MDSNE model, the DNESA algorithm, a domain-specific data pipeline, and a focus on real-time performance and scalability. Designed for real-world underwater applications, our model facilitates low-cost deployments in underwater robots and autonomous vehicles, bridging the gap between research and practical implementation.
>
---
#### [replaced 120] RealKeyMorph: Keypoints in Real-world Coordinates for Resolution-agnostic Image Registration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.10344v2](http://arxiv.org/pdf/2506.10344v2)**

> **作者:** Mina C. Moghadam; Alan Q. Wang; Omer Taub; Martin R. Prince; Mert R. Sabuncu
>
> **备注:** 23 pages, 8 figures
>
> **摘要:** Many real-world settings require registration of a pair of medical images that differ in spatial resolution, which may arise from differences in image acquisition parameters like pixel spacing, slice thickness, and field-of-view. However, all previous machine learning-based registration techniques resample images onto a fixed resolution. This is suboptimal because resampling can introduce artifacts due to interpolation. To address this, we present RealKeyMorph (RKM), a resolution-agnostic method for image registration. RKM is an extension of KeyMorph, a registration framework which works by training a network to learn corresponding keypoints for a given pair of images, after which a closed-form keypoint matching step is used to derive the transformation that aligns them. To avoid resampling and enable operating on the raw data, RKM outputs keypoints in real-world coordinates of the scanner. To do this, we leverage the affine matrix produced by the scanner (e.g., MRI machine) that encodes the mapping from voxel coordinates to real world coordinates. By transforming keypoints into real-world space and integrating this into the training process, RKM effectively enables the extracted keypoints to be resolution-agnostic. In our experiments, we demonstrate the advantages of RKM on the registration task for orthogonal 2D stacks of abdominal MRIs, as well as 3D volumes with varying resolutions in brain datasets.
>
---
#### [replaced 121] Screen Them All: High-Throughput Pan-Cancer Genetic and Phenotypic Biomarker Screening from H&E Whole Slide Images
- **分类: q-bio.QM; cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2408.09554v4](http://arxiv.org/pdf/2408.09554v4)**

> **作者:** Yi Kan Wang; Ludmila Tydlitatova; Jeremy D. Kunz; Gerard Oakley; Bonnie Kar Bo Chow; Ran A. Godrich; Matthew C. H. Lee; Hamed Aghdam; Alican Bozkurt; Michal Zelechowski; Chad Vanderbilt; Christopher Kanan; Juan A. Retamero; Peter Hamilton; Razik Yousfi; Thomas J. Fuchs; David S. Klimstra; Siqi Liu
>
> **摘要:** Molecular assays are standard of care for detecting genomic alterations in cancer prognosis and therapy selection but are costly, tissue-destructive and time-consuming. Artificial intelligence (AI) applied to routine hematoxylin and eosin (H&E)-stained whole slide images (WSIs) offers a fast and economical alternative for screening molecular biomarkers. We introduce OmniScreen, a high-throughput AI-based system leveraging Virchow2 embeddings extracted from 60,529 cancer patients with paired 489-gene MSK-IMPACT targeted biomarker panel and WSIs. Unlike conventional approaches that train separate models for each biomarker, OmniScreen employs a unified model to predict a broad range of clinically relevant biomarkers across cancers, including low-prevalence targets impractical to model individually. OmniScreen reliably identifies therapeutic targets and shared phenotypic features across common and rare tumors. We investigate the biomarker prediction probabilities and accuracies of OmniScreen in relation to tumor area, cohort size, histologic subtype alignment, and pathway-level morphological patterns. These findings underscore the potential of OmniScreen for routine clinical screening.
>
---
#### [replaced 122] Structure-Guided Diffusion Models for High-Fidelity Portrait Shadow Removal
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.04692v2](http://arxiv.org/pdf/2507.04692v2)**

> **作者:** Wanchang Yu; Qing Zhang; Rongjia Zheng; Wei-Shi Zheng
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** We present a diffusion-based portrait shadow removal approach that can robustly produce high-fidelity results. Unlike previous methods, we cast shadow removal as diffusion-based inpainting. To this end, we first train a shadow-independent structure extraction network on a real-world portrait dataset with various synthetic lighting conditions, which allows to generate a shadow-independent structure map including facial details while excluding the unwanted shadow boundaries. The structure map is then used as condition to train a structure-guided inpainting diffusion model for removing shadows in a generative manner. Finally, to restore the fine-scale details (e.g., eyelashes, moles and spots) that may not be captured by the structure map, we take the gradients inside the shadow regions as guidance and train a detail restoration diffusion model to refine the shadow removal result. Extensive experiments on the benchmark datasets show that our method clearly outperforms existing methods, and is effective to avoid previously common issues such as facial identity tampering, shadow residual, color distortion, structure blurring, and loss of details. Our code is available at https://github.com/wanchang-yu/Structure-Guided-Diffusion-for-Portrait-Shadow-Removal.
>
---
#### [replaced 123] MGVQ: Could VQ-VAE Beat VAE? A Generalizable Tokenizer with Multi-group Quantization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.07997v2](http://arxiv.org/pdf/2507.07997v2)**

> **作者:** Mingkai Jia; Wei Yin; Xiaotao Hu; Jiaxin Guo; Xiaoyang Guo; Qian Zhang; Xiao-Xiao Long; Ping Tan
>
> **摘要:** Vector Quantized Variational Autoencoders (VQ-VAEs) are fundamental models that compress continuous visual data into discrete tokens. Existing methods have tried to improve the quantization strategy for better reconstruction quality, however, there still exists a large gap between VQ-VAEs and VAEs. To narrow this gap, we propose MGVQ, a novel method to augment the representation capability of discrete codebooks, facilitating easier optimization for codebooks and minimizing information loss, thereby enhancing reconstruction quality. Specifically, we propose to retain the latent dimension to preserve encoded features and incorporate a set of sub-codebooks for quantization. Furthermore, we construct comprehensive zero-shot benchmarks featuring resolutions of 512p and 2k to evaluate the reconstruction performance of existing methods rigorously. MGVQ achieves the state-of-the-art performance on both ImageNet and 8 zero-shot benchmarks across all VQ-VAEs. Notably, compared with SD-VAE, we outperform them on ImageNet significantly, with rFID 0.49 v.s. 0.91, and achieve superior PSNR on all zero-shot benchmarks. These results highlight the superiority of MGVQ in reconstruction and pave the way for preserving fidelity in HD image processing tasks. Code will be publicly available at https://github.com/MKJia/MGVQ.
>
---
#### [replaced 124] BiDepth: A Bidirectional-Depth Neural Network for Spatio-Temporal Prediction
- **分类: cs.LG; cs.AI; cs.CV; stat.AP**

- **链接: [http://arxiv.org/pdf/2501.08411v3](http://arxiv.org/pdf/2501.08411v3)**

> **作者:** Sina Ehsani; Fenglian Pan; Qingpei Hu; Jian Liu
>
> **备注:** 21 pages, 6 figures. Submitted to ACM TKDD
>
> **摘要:** Accurate spatial-temporal (ST) prediction for dynamic systems, such as urban mobility and weather patterns, is crucial but hindered by complex ST correlations and the challenge of concurrently modeling long-term trends with short-term fluctuations. Existing methods often falter in these areas. This paper proposes the BiDepth Multimodal Neural Network (BDMNN), which integrates two key innovations: 1) a bidirectional depth modulation mechanism that dynamically adjusts network depth to comprehensively capture both long-term seasonality and immediate short-term events; and 2) a novel convolutional self-attention cell (CSAC). Critically, unlike many attention mechanisms that can lose spatial acuity, our CSAC is specifically designed to preserve crucial spatial relationships throughout the network, akin to standard convolutional layers, while simultaneously capturing temporal dependencies. Evaluated on real-world urban traffic and precipitation datasets, BDMNN demonstrates significant accuracy improvements, achieving a 12% Mean Squared Error (MSE) reduction in urban traffic prediction and a 15% improvement in precipitation forecasting over leading deep learning benchmarks like ConvLSTM, using comparable computational resources. These advancements offer robust ST forecasting for smart city management, disaster prevention, and resource optimization.
>
---
#### [replaced 125] CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.05332v2](http://arxiv.org/pdf/2503.05332v2)**

> **作者:** Jungho Lee; Donghyeong Kim; Dogyoon Lee; Suhwan Cho; Minhyeok Lee; Wonjoon Lee; Taeoh Kim; Dongyoon Wee; Sangyoun Lee
>
> **备注:** Revised Version of CRiM-GS, Project Page: https://Jho-Yonsei.github.io/CoMoGaussian
>
> **摘要:** 3D Gaussian Splatting (3DGS) has gained significant attention due to its high-quality novel view rendering, motivating research to address real-world challenges. A critical issue is the camera motion blur caused by movement during exposure, which hinders accurate 3D scene reconstruction. In this study, we propose CoMoGaussian, a Continuous Motion-Aware Gaussian Splatting that reconstructs precise 3D scenes from motion-blurred images while maintaining real-time rendering speed. Considering the complex motion patterns inherent in real-world camera movements, we predict continuous camera trajectories using neural ordinary differential equations (ODEs). To ensure accurate modeling, we employ rigid body transformations, preserving the shape and size of the object but rely on the discrete integration of sampled frames. To better approximate the continuous nature of motion blur, we introduce a continuous motion refinement (CMR) transformation that refines rigid transformations by incorporating additional learnable parameters. By revisiting fundamental camera theory and leveraging advanced neural ODE techniques, we achieve precise modeling of continuous camera trajectories, leading to improved reconstruction accuracy. Extensive experiments demonstrate state-of-the-art performance both quantitatively and qualitatively on benchmark datasets, which include a wide range of motion blur scenarios, from moderate to extreme blur.
>
---
#### [replaced 126] Capsule Networks Do Not Need to Model Everything
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2204.01298v2](http://arxiv.org/pdf/2204.01298v2)**

> **作者:** Riccardo Renzulli; Enzo Tartaglione; Marco Grangetto
>
> **备注:** Accepted at Pattern Recognition
>
> **摘要:** Capsule networks are biologically inspired neural networks that group neurons into vectors called capsules, each explicitly representing an object or one of its parts. The routing mechanism connects capsules in consecutive layers, forming a hierarchical structure between parts and objects, also known as a parse tree. Capsule networks often attempt to model all elements in an image, requiring large network sizes to handle complexities such as intricate backgrounds or irrelevant objects. However, this comprehensive modeling leads to increased parameter counts and computational inefficiencies. Our goal is to enable capsule networks to focus only on the object of interest, reducing the number of parse trees. We accomplish this with REM (Routing Entropy Minimization), a technique that minimizes the entropy of the parse tree-like structure. REM drives the model parameters distribution towards low entropy configurations through a pruning mechanism, significantly reducing the generation of intra-class parse trees. This empowers capsules to learn more stable and succinct representations with fewer parameters and negligible performance loss.
>
---
#### [replaced 127] LVAgent: Long Video Understanding by Multi-Round Dynamical Collaboration of MLLM Agents
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10200v3](http://arxiv.org/pdf/2503.10200v3)**

> **作者:** Boyu Chen; Zhengrong Yue; Siran Chen; Zikang Wang; Yang Liu; Peng Li; Yali Wang
>
> **备注:** accepted in ICCV 2025
>
> **摘要:** Existing MLLMs encounter significant challenges in modeling the temporal context within long videos. Currently, mainstream Agent-based methods use external tools to assist a single MLLM in answering long video questions. Despite such tool-based support, a solitary MLLM still offers only a partial understanding of long videos, resulting in limited performance. In order to better address long video tasks, we introduce LVAgent, the first framework enabling multi-round dynamic collaboration of MLLM agents in long video understanding. Our method consists of four key steps: 1) Selection: We pre-select appropriate agents from the model library to form optimal agent teams based on different tasks. 2) Perception: We design an effective retrieval scheme for long videos to improve the coverage of critical temporal segments while maintaining computational efficiency. 3) Action: Agents answer long video questions and exchange reasons. 4) Reflection: We evaluate each agent's performance in each round of discussion and optimize the agent team for dynamic collaboration. The agents iteratively refine their answers by multi-round dynamical collaboration of MLLM agents. LVAgent is the first agent system method that outperforms all closed-source models (like GPT-4o) and open-source models (like InternVL-2.5 and Qwen2-VL) in the long video understanding tasks. Our LVAgent achieves an accuracy of 80\% on four mainstream long video understanding tasks. Notably, LVAgent improves accuracy by 13.3\% on LongVideoBench. Code is available at https://github.com/64327069/LVAgent.
>
---
#### [replaced 128] 3D Reconstruction of the Human Colon from Capsule Endoscope Video
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.15228v2](http://arxiv.org/pdf/2407.15228v2)**

> **作者:** Pål Anders Floor; Ivar Farup; Marius Pedersen
>
> **备注:** 11 pages, 12 figures
>
> **摘要:** As the number of people affected by diseases in the gastrointestinal system is ever-increasing, a higher demand on preventive screening is inevitable. This will significantly increase the workload on gastroenterologists. To help reduce the workload, tools from computer vision may be helpful. In this paper, we investigate the possibility of constructing 3D models of whole sections of the human colon using image sequences from wireless capsule endoscope video, providing enhanced viewing for gastroenterologists. As capsule endoscope images contain distortion and artifacts non-ideal for many 3D reconstruction algorithms, the problem is challenging. However, recent developments of virtual graphics-based models of the human gastrointestinal system, where distortion and artifacts can be enabled or disabled, makes it possible to ``dissect'' the problem. The graphical model also provides a ground truth, enabling computation of geometric distortion introduced by the 3D reconstruction method. In this paper, most distortions and artifacts are left out to determine if it is feasible to reconstruct whole sections of the human gastrointestinal system by existing methods. We demonstrate that 3D reconstruction is possible using simultaneous localization and mapping. Further, to reconstruct the gastrointestinal wall surface from resulting point clouds, varying greatly in density, Poisson surface reconstruction is a good option. The results are promising, encouraging further research on this problem.
>
---
#### [replaced 129] A General Framework for Inference-time Scaling and Steering of Diffusion Models
- **分类: cs.LG; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.06848v4](http://arxiv.org/pdf/2501.06848v4)**

> **作者:** Raghav Singhal; Zachary Horvitz; Ryan Teehan; Mengye Ren; Zhou Yu; Kathleen McKeown; Rajesh Ranganath
>
> **摘要:** Diffusion models produce impressive results in modalities ranging from images and video to protein design and text. However, generating samples with user-specified properties remains a challenge. Recent research proposes fine-tuning models to maximize rewards that capture desired properties, but these methods require expensive training and are prone to mode collapse. In this work, we present Feynman-Kac (FK) steering, an inference-time framework for steering diffusion models with reward functions. FK steering works by sampling a system of multiple interacting diffusion processes, called particles, and resampling particles at intermediate steps based on scores computed using functions called potentials. Potentials are defined using rewards for intermediate states and are selected such that a high value indicates that the particle will yield a high-reward sample. We explore various choices of potentials, intermediate rewards, and samplers. We evaluate FK steering on text-to-image and text diffusion models. For steering text-to-image models with a human preference reward, we find that FK steering a 0.8B parameter model outperforms a 2.6B parameter fine-tuned model on prompt fidelity, with faster sampling and no training. For steering text diffusion models with rewards for text quality and specific text attributes, we find that FK steering generates lower perplexity, more linguistically acceptable outputs and enables gradient-free control of attributes like toxicity. Our results demonstrate that inference-time scaling and steering of diffusion models - even with off-the-shelf rewards - can provide significant sample quality gains and controllability benefits. Code is available at https://github.com/zacharyhorvitz/Fk-Diffusion-Steering .
>
---
#### [replaced 130] MGA-Net: A Novel Mask-Guided Attention Neural Network for Precision Neonatal Brain Imaging
- **分类: eess.IV; cs.CV; stat.CO**

- **链接: [http://arxiv.org/pdf/2406.17709v3](http://arxiv.org/pdf/2406.17709v3)**

> **作者:** Bahram Jafrasteh; Simon Pedro Lubian-Lopez; Emiliano Trimarco; Macarena Roman Ruiz; Carmen Rodriguez Barrios; Yolanda Marin Almagro; Isabel Benavente-Fernandez
>
> **摘要:** In this study, we introduce MGA-Net, a novel mask-guided attention neural network, which extends the U-net model for precision neonatal brain imaging. MGA-Net is designed to extract the brain from other structures and reconstruct high-quality brain images. The network employs a common encoder and two decoders: one for brain mask extraction and the other for brain region reconstruction. A key feature of MGA-Net is its high-level mask-guided attention module, which leverages features from the brain mask decoder to enhance image reconstruction. To enable the same encoder and decoder to process both MRI and ultrasound (US) images, MGA-Net integrates sinusoidal positional encoding. This encoding assigns distinct positional values to MRI and US images, allowing the model to effectively learn from both modalities. Consequently, features learned from a single modality can aid in learning a modality with less available data, such as US. We extensively validated the proposed MGA-Net on diverse and independent datasets from varied clinical settings and neonatal age groups. The metrics used for assessment included the DICE similarity coefficient, recall, and accuracy for image segmentation; structural similarity for image reconstruction; and root mean squared error for total brain volume estimation from 3D ultrasound images. Our results demonstrate that MGA-Net significantly outperforms traditional methods, offering superior performance in brain extraction and segmentation while achieving high precision in image reconstruction and volumetric analysis. Thus, MGA-Net represents a robust and effective preprocessing tool for MRI and 3D ultrasound images, marking a significant advance in neuroimaging that enhances both research and clinical diagnostics in the neonatal period and beyond.Our code is available at https://github.com/BahramJafrasteh/MGA-Net
>
---
#### [replaced 131] Many-for-Many: Unify the Training of Multiple Video and Image Generation and Manipulation Tasks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.01758v2](http://arxiv.org/pdf/2506.01758v2)**

> **作者:** Tao Yang; Ruibin Li; Yangming Shi; Yuqi Zhang; Qide Dong; Haoran Cheng; Weiguo Feng; Shilei Wen; Bingyue Peng; Lei Zhang
>
> **摘要:** Diffusion models have shown impressive performance in many visual generation and manipulation tasks. Many existing methods focus on training a model for a specific task, especially, text-to-video (T2V) generation, while many other works focus on finetuning the pretrained T2V model for image-to-video (I2V), video-to-video (V2V), image and video manipulation tasks, etc. However, training a strong T2V foundation model requires a large amount of high-quality annotations, which is very costly. In addition, many existing models can perform only one or several tasks. In this work, we introduce a unified framework, namely many-for-many, which leverages the available training data from many different visual generation and manipulation tasks to train a single model for those different tasks. Specifically, we design a lightweight adapter to unify the different conditions in different tasks, then employ a joint image-video learning strategy to progressively train the model from scratch. Our joint learning leads to a unified visual generation and manipulation model with improved video generation performance. In addition, we introduce depth maps as a condition to help our model better perceive the 3D space in visual generation. Two versions of our model are trained with different model sizes (8B and 2B), each of which can perform more than 10 different tasks. In particular, our 8B model demonstrates highly competitive performance in video generation tasks compared to open-source and even commercial engines. Our models and source codes are available at https://github.com/leeruibin/MfM.git.
>
---
#### [replaced 132] Alignment and Adversarial Robustness: Are More Human-Like Models More Secure?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.12377v2](http://arxiv.org/pdf/2502.12377v2)**

> **作者:** Blaine Hoak; Kunyang Li; Patrick McDaniel
>
> **备注:** Accepted to International Workshop on Security and Privacy-Preserving AI/ML (SPAIML) 2025
>
> **摘要:** A small but growing body of work has shown that machine learning models which better align with human vision have also exhibited higher robustness to adversarial examples, raising the question: can human-like perception make models more secure? If true generally, such mechanisms would offer new avenues toward robustness. In this work, we conduct a large-scale empirical analysis to systematically investigate the relationship between representational alignment and adversarial robustness. We evaluate 114 models spanning diverse architectures and training paradigms, measuring their neural and behavioral alignment and engineering task performance across 105 benchmarks as well as their adversarial robustness via AutoAttack. Our findings reveal that while average alignment and robustness exhibit a weak overall correlation, specific alignment benchmarks serve as strong predictors of adversarial robustness, particularly those that measure selectivity toward texture or shape. These results suggest that different forms of alignment play distinct roles in model robustness, motivating further investigation into how alignment-driven approaches can be leveraged to build more secure and perceptually-grounded vision models.
>
---
#### [replaced 133] LIGHT: Multi-Modal Text Linking on Historical Maps
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.22589v2](http://arxiv.org/pdf/2506.22589v2)**

> **作者:** Yijun Lin; Rhett Olson; Junhan Wu; Yao-Yi Chiang; Jerod Weinman
>
> **备注:** Accepted at ICDAR2025
>
> **摘要:** Text on historical maps provides valuable information for studies in history, economics, geography, and other related fields. Unlike structured or semi-structured documents, text on maps varies significantly in orientation, reading order, shape, and placement. Many modern methods can detect and transcribe text regions, but they struggle to effectively ``link'' the recognized text fragments, e.g., determining a multi-word place name. Existing layout analysis methods model word relationships to improve text understanding in structured documents, but they primarily rely on linguistic features and neglect geometric information, which is essential for handling map text. To address these challenges, we propose LIGHT, a novel multi-modal approach that integrates linguistic, image, and geometric features for linking text on historical maps. In particular, LIGHT includes a geometry-aware embedding module that encodes the polygonal coordinates of text regions to capture polygon shapes and their relative spatial positions on an image. LIGHT unifies this geometric information with the visual and linguistic token embeddings from LayoutLMv3, a pretrained layout analysis model. LIGHT uses the cross-modal information to predict the reading-order successor of each text instance directly with a bi-directional learning strategy that enhances sequence robustness. Experimental results show that LIGHT outperforms existing methods on the ICDAR 2024/2025 MapText Competition data, demonstrating the effectiveness of multi-modal learning for historical map text linking.
>
---
#### [replaced 134] HA-RDet: Hybrid Anchor Rotation Detector for Oriented Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.14379v2](http://arxiv.org/pdf/2412.14379v2)**

> **作者:** Phuc D. A. Nguyen
>
> **备注:** Bachelor thesis, Accepted to ICCV'25 SEA
>
> **摘要:** Oriented object detection in aerial images poses a significant challenge due to their varying sizes and orientations. Current state-of-the-art detectors typically rely on either two-stage or one-stage approaches, often employing Anchor-based strategies, which can result in computationally expensive operations due to the redundant number of generated anchors during training. In contrast, Anchor-free mechanisms offer faster processing but suffer from a reduction in the number of training samples, potentially impacting detection accuracy. To address these limitations, we propose the Hybrid-Anchor Rotation Detector (HA-RDet), which combines the advantages of both anchor-based and anchor-free schemes for oriented object detection. By utilizing only one preset anchor for each location on the feature maps and refining these anchors with our Orientation-Aware Convolution technique, HA-RDet achieves competitive accuracies, including 75.41 mAP on DOTA-v1, 65.3 mAP on DIOR-R, and 90.2 mAP on HRSC2016, against current anchor-based state-of-the-art methods, while significantly reducing computational resources.
>
---
#### [replaced 135] MIGE: Mutually Enhanced Multimodal Instruction-Based Image Generation and Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.21291v3](http://arxiv.org/pdf/2502.21291v3)**

> **作者:** Xueyun Tian; Wei Li; Bingbing Xu; Yige Yuan; Yuanzhuo Wang; Huawei Shen
>
> **备注:** This paper have been accepted by ACM MM25
>
> **摘要:** Despite significant progress in diffusion-based image generation, subject-driven generation and instruction-based editing remain challenging. Existing methods typically treat them separately, struggling with limited high-quality data and poor generalization. However, both tasks require capturing complex visual variations while maintaining consistency between inputs and outputs. Inspired by this, we propose MIGE, a unified framework that standardizes task representations using multimodal instructions. It first treats subject-driven generation as creation on a blank canvas and instruction-based editing as modification of an existing image, establishing a shared input-output formulation, then introduces a novel multimodal encoder that maps free-form multimodal instructions into a unified vision-language space, integrating visual and semantic features through a feature fusion mechanism. This unification enables joint training of both tasks, providing two key advantages: (1) Cross-Task Enhancement: by leveraging shared visual and semantic representations, joint training improves instruction adherence and visual consistency in both subject-driven generation and instruction-based editing. (2) Generalization: learning in a unified format facilitates cross-task knowledge transfer, enabling MIGE to generalize to novel compositional tasks, including instruction-based subject-driven editing. Experiments show that MIGE excels in both subject-driven generation and instruction-based editing while setting a SOTA in the new task of instruction-based subject-driven editing. Code and model have been publicly available at https://github.com/Eureka-Maggie/MIGE/tree/main.
>
---
#### [replaced 136] FlexEdit: Marrying Free-Shape Masks to VLLM for Flexible Image Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.12429v2](http://arxiv.org/pdf/2408.12429v2)**

> **作者:** Tianshuo Yuan; Yuxiang Lin; Jue Wang; Zhi-Qi Cheng; Xiaolong Wang; Jiao GH; Wei Chen; Xiaojiang Peng
>
> **备注:** 15 pages, 14 figures
>
> **摘要:** Combining Vision Large Language Models (VLLMs) with diffusion models offers a powerful method for executing image editing tasks based on human language instructions. However, language instructions alone often fall short in accurately conveying user requirements, particularly when users want to add, replace elements in specific areas of an image. Luckily, masks can effectively indicate the exact locations or elements to be edited, while they require users to precisely draw the shapes at the desired locations, which is highly user-unfriendly. To address this, we propose FlexEdit, an end-to-end image editing method that leverages both free-shape masks and language instructions for Flexible Editing. Our approach employs a VLLM in comprehending the image content, mask, and user instructions. Additionally, we introduce the Mask Enhance Adapter (MEA) that fuses the embeddings of the VLLM with the image data, ensuring a seamless integration of mask information and model output embeddings. Furthermore, we construct FSMI-Edit, a benchmark specifically tailored for free-shape mask, including 8 types of free-shape mask. Extensive experiments show that our method achieves state-of-the-art (SOTA) performance in LLM-based image editing, and our simple prompting technique stands out in its effectiveness. The code and data can be found at https://github.com/A-new-b/flex_edit.
>
---
#### [replaced 137] Compression-Aware One-Step Diffusion Model for JPEG Artifact Removal
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.09873v3](http://arxiv.org/pdf/2502.09873v3)**

> **作者:** Jinpei Guo; Zheng Chen; Wenbo Li; Yong Guo; Yulun Zhang
>
> **摘要:** Diffusion models have demonstrated remarkable success in image restoration tasks. However, their multi-step denoising process introduces significant computational overhead, limiting their practical deployment. Furthermore, existing methods struggle to effectively remove severe JPEG artifact, especially in highly compressed images. To address these challenges, we propose CODiff, a compression-aware one-step diffusion model for JPEG artifact removal. The core of CODiff is the compression-aware visual embedder (CaVE), which extracts and leverages JPEG compression priors to guide the diffusion model. We propose a dual learning strategy that combines explicit and implicit learning. Specifically, explicit learning enforces a quality prediction objective to differentiate low-quality images with different compression levels. Implicit learning employs a reconstruction objective that enhances the model's generalization. This dual learning allows for a deeper and more comprehensive understanding of JPEG compression. Experimental results demonstrate that CODiff surpasses recent leading methods in both quantitative and visual quality metrics. The code is released at https://github.com/jp-guo/CODiff.
>
---
#### [replaced 138] EECD-Net: Energy-Efficient Crack Detection with Spiking Neural Networks and Gated Attention
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.04526v2](http://arxiv.org/pdf/2506.04526v2)**

> **作者:** Shuo Zhang
>
> **备注:** Withdrawn by the authors due to a critical bug in our energy consumption analysis. The script for calculating synaptic operations (SOPs) for baseline models was flawed, leading to an incorrect overestimation of our method's energy efficiency
>
> **摘要:** Crack detection on road surfaces is a critical measurement technology in the instrumentation domain, essential for ensuring infrastructure safety and transportation reliability. However, due to limited energy and low-resolution imaging, smart terminal devices struggle to maintain real-time monitoring performance. To overcome these challenges, this paper proposes a multi-stage detection approach for road crack detection, EECD-Net, to enhance accuracy and energy efficiency of instrumentation. Specifically, the sophisticated Super-Resolution Convolutional Neural Network (SRCNN) is employed to address the inherent challenges of low-quality images, which effectively enhance image resolution while preserving critical structural details. Meanwhile, a Spike Convolution Unit (SCU) with Continuous Integrate-and-Fire (CIF) neurons is proposed to convert these images into sparse pulse sequences, significantly reducing power consumption. Additionally, a Gated Attention Transformer (GAT) module is designed to strategically fuse multi-scale feature representations through adaptive attention mechanisms, effectively capturing both long-range dependencies and intricate local crack patterns, and significantly enhancing detection robustness across varying crack morphologies. The experiments on the CrackVision12K benchmark demonstrate that EECD-Net achieves a remarkable 98.6\% detection accuracy, surpassing state-of-the-art counterparts such as Hybrid-Segmentor by a significant 1.5\%. Notably, the EECD-Net maintains exceptional energy efficiency, consuming merely 5.6 mJ, which is a substantial 33\% reduction compared to baseline implementations. This work pioneers a transformative approach in instrumentation-based crack detection, offering a scalable, low-power solution for real-time, large-scale infrastructure monitoring in resource-constrained environments.
>
---
#### [replaced 139] Colorectal Cancer Tumor Grade Segmentation in Digital Histopathology Images: From Giga to Mini Challenge
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.04681v2](http://arxiv.org/pdf/2507.04681v2)**

> **作者:** Alper Bahcekapili; Duygu Arslan; Umut Ozdemir; Berkay Ozkirli; Emre Akbas; Ahmet Acar; Gozde B. Akar; Bingdou He; Shuoyu Xu; Umit Mert Caglar; Alptekin Temizel; Guillaume Picaud; Marc Chaumont; Gérard Subsol; Luc Téot; Fahad Alsharekh; Shahad Alghannam; Hexiang Mao; Wenhua Zhang
>
> **备注:** Accepted Grand Challenge Paper ICIP 2025
>
> **摘要:** Colorectal cancer (CRC) is the third most diagnosed cancer and the second leading cause of cancer-related death worldwide. Accurate histopathological grading of CRC is essential for prognosis and treatment planning but remains a subjective process prone to observer variability and limited by global shortages of trained pathologists. To promote automated and standardized solutions, we organized the ICIP Grand Challenge on Colorectal Cancer Tumor Grading and Segmentation using the publicly available METU CCTGS dataset. The dataset comprises 103 whole-slide images with expert pixel-level annotations for five tissue classes. Participants submitted segmentation masks via Codalab, evaluated using metrics such as macro F-score and mIoU. Among 39 participating teams, six outperformed the Swin Transformer baseline (62.92 F-score). This paper presents an overview of the challenge, dataset, and the top-performing methods
>
---
#### [replaced 140] RIPE: Reinforcement Learning on Unlabeled Image Pairs for Robust Keypoint Extraction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.04839v2](http://arxiv.org/pdf/2507.04839v2)**

> **作者:** Johannes Künzel; Anna Hilsmann; Peter Eisert
>
> **备注:** ICCV 2025
>
> **摘要:** We introduce RIPE, an innovative reinforcement learning-based framework for weakly-supervised training of a keypoint extractor that excels in both detection and description tasks. In contrast to conventional training regimes that depend heavily on artificial transformations, pre-generated models, or 3D data, RIPE requires only a binary label indicating whether paired images represent the same scene. This minimal supervision significantly expands the pool of training data, enabling the creation of a highly generalized and robust keypoint extractor. RIPE utilizes the encoder's intermediate layers for the description of the keypoints with a hyper-column approach to integrate information from different scales. Additionally, we propose an auxiliary loss to enhance the discriminative capability of the learned descriptors. Comprehensive evaluations on standard benchmarks demonstrate that RIPE simplifies data preparation while achieving competitive performance compared to state-of-the-art techniques, marking a significant advancement in robust keypoint extraction and description. To support further research, we have made our code publicly available at https://github.com/fraunhoferhhi/RIPE.
>
---
#### [replaced 141] WeGeFT: Weight-Generative Fine-Tuning for Multi-Faceted Efficient Adaptation of Large Models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2312.00700v5](http://arxiv.org/pdf/2312.00700v5)**

> **作者:** Chinmay Savadikar; Xi Song; Tianfu Wu
>
> **备注:** Accepted to ICML25
>
> **摘要:** Fine-tuning large pretrained Transformer models can focus on either introducing a small number of new learnable parameters (parameter efficiency) or editing representations of a small number of tokens using lightweight modules (representation efficiency). While the pioneering method LoRA (Low-Rank Adaptation) inherently balances parameter, compute, and memory efficiency, many subsequent variants trade off compute and memory efficiency and/or performance to further reduce fine-tuning parameters. To address this limitation and unify parameter-efficient and representation-efficient fine-tuning, we propose Weight-Generative Fine-Tuning (WeGeFT, pronounced wee-gift), a novel approach that learns to generate fine-tuning weights directly from the pretrained weights. WeGeFT employs a simple low-rank formulation consisting of two linear layers, either shared across multiple layers of the pretrained model or individually learned for different layers. This design achieves multi-faceted efficiency in parameters, representations, compute, and memory, while maintaining or exceeding the performance of LoRA and its variants. Extensive experiments on commonsense reasoning, arithmetic reasoning, instruction following, code generation, and visual recognition verify the effectiveness of our proposed WeGeFT. Our code is available at https://github.com/savadikarc/wegeft
>
---
#### [replaced 142] Adapting OpenAI's CLIP Model for Few-Shot Image Inspection in Manufacturing Quality Control: An Expository Case Study with Multiple Application Examples
- **分类: cs.CV; stat.AP; stat.OT**

- **链接: [http://arxiv.org/pdf/2501.12596v2](http://arxiv.org/pdf/2501.12596v2)**

> **作者:** Fadel M. Megahed; Ying-Ju Chen; Bianca Maria Colosimo; Marco Luigi Giuseppe Grasso; L. Allison Jones-Farmer; Sven Knoth; Hongyue Sun; Inez Zwetsloot
>
> **备注:** 36 pages, 13 figures
>
> **摘要:** This expository paper introduces a simplified approach to image-based quality inspection in manufacturing using OpenAI's CLIP (Contrastive Language-Image Pretraining) model adapted for few-shot learning. While CLIP has demonstrated impressive capabilities in general computer vision tasks, its direct application to manufacturing inspection presents challenges due to the domain gap between its training data and industrial applications. We evaluate CLIP's effectiveness through five case studies: metallic pan surface inspection, 3D printing extrusion profile analysis, stochastic textured surface evaluation, automotive assembly inspection, and microstructure image classification. Our results show that CLIP can achieve high classification accuracy with relatively small learning sets (50-100 examples per class) for single-component and texture-based applications. However, the performance degrades with complex multi-component scenes. We provide a practical implementation framework that enables quality engineers to quickly assess CLIP's suitability for their specific applications before pursuing more complex solutions. This work establishes CLIP-based few-shot learning as an effective baseline approach that balances implementation simplicity with robust performance, demonstrated in several manufacturing quality control applications.
>
---
