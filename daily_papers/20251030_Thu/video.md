# 计算机视觉 cs.CV

- **最新发布 87 篇**

- **更新 54 篇**

## 最新发布

#### [new 001] $D^2GS$: Dense Depth Regularization for LiDAR-free Urban Scene Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出$D^2GS$，一种无需LiDAR的城市场景重建方法。针对现有方法依赖复杂校准与易受误差影响的LiDAR数据问题，通过密集深度初始化、联合优化高斯几何与深度及道路区域约束，实现更精确的几何重建，在Waymo数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.25173v1](http://arxiv.org/pdf/2510.25173v1)**

> **作者:** Kejing Xia; Jidong Jia; Ke Jin; Yucai Bai; Li Sun; Dacheng Tao; Youjian Zhang
>
> **摘要:** Recently, Gaussian Splatting (GS) has shown great potential for urban scene reconstruction in the field of autonomous driving. However, current urban scene reconstruction methods often depend on multimodal sensors as inputs, \textit{i.e.} LiDAR and images. Though the geometry prior provided by LiDAR point clouds can largely mitigate ill-posedness in reconstruction, acquiring such accurate LiDAR data is still challenging in practice: i) precise spatiotemporal calibration between LiDAR and other sensors is required, as they may not capture data simultaneously; ii) reprojection errors arise from spatial misalignment when LiDAR and cameras are mounted at different locations. To avoid the difficulty of acquiring accurate LiDAR depth, we propose $D^2GS$, a LiDAR-free urban scene reconstruction framework. In this work, we obtain geometry priors that are as effective as LiDAR while being denser and more accurate. $\textbf{First}$, we initialize a dense point cloud by back-projecting multi-view metric depth predictions. This point cloud is then optimized by a Progressive Pruning strategy to improve the global consistency. $\textbf{Second}$, we jointly refine Gaussian geometry and predicted dense metric depth via a Depth Enhancer. Specifically, we leverage diffusion priors from a depth foundation model to enhance the depth maps rendered by Gaussians. In turn, the enhanced depths provide stronger geometric constraints during Gaussian training. $\textbf{Finally}$, we improve the accuracy of ground geometry by constraining the shape and normal attributes of Gaussians within road regions. Extensive experiments on the Waymo dataset demonstrate that our method consistently outperforms state-of-the-art methods, producing more accurate geometry even when compared with those using ground-truth LiDAR data.
>
---
#### [new 002] FT-ARM: Fine-Tuned Agentic Reflection Multimodal Language Model for Pressure Ulcer Severity Classification with Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出FT-ARM模型，用于压力性溃疡严重程度分类（I-IV期）。针对临床判断主观性强、现有AI模型可解释性差的问题，通过微调多模态大语言模型并引入代理式自我反思机制，结合视觉与文本信息进行推理，提升分类准确率至85%，并生成可解释的自然语言报告，支持实时部署。**

- **链接: [http://arxiv.org/pdf/2510.24980v1](http://arxiv.org/pdf/2510.24980v1)**

> **作者:** Reza Saadati Fard; Emmanuel Agu; Palawat Busaranuvong; Deepak Kumar; Shefalika Gautam; Bengisu Tulu; Diane Strong; Lorraine Loretz
>
> **摘要:** Pressure ulcers (PUs) are a serious and prevalent healthcare concern. Accurate classification of PU severity (Stages I-IV) is essential for proper treatment but remains challenging due to subtle visual distinctions and subjective interpretation, leading to variability among clinicians. Prior AI-based approaches using Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) achieved promising accuracy but offered limited interpretability. We present FT-ARM (Fine-Tuned Agentic Reflection Multimodal model), a fine-tuned multimodal large language model (MLLM) with an agentic self-reflection mechanism for pressure ulcer severity classification. Inspired by clinician-style diagnostic reassessment, FT-ARM iteratively refines its predictions by reasoning over visual features and encoded clinical knowledge from text, enhancing both accuracy and consistency. On the publicly available Pressure Injury Image Dataset (PIID), FT-ARM, fine-tuned from LLaMA 3.2 90B, achieved 85% accuracy in classifying PU stages I-IV, surpassing prior CNN-based models by +4%. Unlike earlier CNN/ViT studies that relied solely on offline evaluations, FT-ARM is designed and tested for live inference, reflecting real-time deployment conditions. Furthermore, it produces clinically grounded natural-language explanations, improving interpretability and trust. By integrating fine-tuning and reflective reasoning across multimodal inputs, FT-ARM advances the reliability, transparency, and clinical applicability of automated wound assessment systems, addressing the critical need for consistent and explainable PU staging to support improved patient care.
>
---
#### [new 003] Towards Fine-Grained Human Motion Video Captioning
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦细粒度人体动作视频描述任务，旨在解决现有模型难以捕捉细微运动细节导致描述模糊的问题。提出M-ACM模型，利用人体网格恢复的运动表征实现运动感知解码，提升描述准确性和空间一致性，并构建了HMI数据集与基准评测体系。**

- **链接: [http://arxiv.org/pdf/2510.24767v1](http://arxiv.org/pdf/2510.24767v1)**

> **作者:** Guorui Song; Guocun Wang; Zhe Huang; Jing Lin; Xuefei Zhe; Jian Li; Haoqian Wang
>
> **摘要:** Generating accurate descriptions of human actions in videos remains a challenging task for video captioning models. Existing approaches often struggle to capture fine-grained motion details, resulting in vague or semantically inconsistent captions. In this work, we introduce the Motion-Augmented Caption Model (M-ACM), a novel generative framework that enhances caption quality by incorporating motion-aware decoding. At its core, M-ACM leverages motion representations derived from human mesh recovery to explicitly highlight human body dynamics, thereby reducing hallucinations and improving both semantic fidelity and spatial alignment in the generated captions. To support research in this area, we present the Human Motion Insight (HMI) Dataset, comprising 115K video-description pairs focused on human movement, along with HMI-Bench, a dedicated benchmark for evaluating motion-focused video captioning. Experimental results demonstrate that M-ACM significantly outperforms previous methods in accurately describing complex human motions and subtle temporal variations, setting a new standard for motion-centric video captioning.
>
---
#### [new 004] Instance-Level Composed Image Retrieval
- **分类: cs.CV**

- **简介: 该论文针对实例级组合图像检索任务，解决现有数据集缺乏高质量训练与评估数据的问题。提出i-CIR数据集，聚焦实例级类别定义，并设计BASIC方法，利用预训练模型实现无需训练的相似度融合，显著提升检索性能，在新旧数据集上均达最优。**

- **链接: [http://arxiv.org/pdf/2510.25387v1](http://arxiv.org/pdf/2510.25387v1)**

> **作者:** Bill Psomas; George Retsinas; Nikos Efthymiadis; Panagiotis Filntisis; Yannis Avrithis; Petros Maragos; Ondrej Chum; Giorgos Tolias
>
> **备注:** NeurIPS 2025
>
> **摘要:** The progress of composed image retrieval (CIR), a popular research direction in image retrieval, where a combined visual and textual query is used, is held back by the absence of high-quality training and evaluation data. We introduce a new evaluation dataset, i-CIR, which, unlike existing datasets, focuses on an instance-level class definition. The goal is to retrieve images that contain the same particular object as the visual query, presented under a variety of modifications defined by textual queries. Its design and curation process keep the dataset compact to facilitate future research, while maintaining its challenge-comparable to retrieval among more than 40M random distractors-through a semi-automated selection of hard negatives. To overcome the challenge of obtaining clean, diverse, and suitable training data, we leverage pre-trained vision-and-language models (VLMs) in a training-free approach called BASIC. The method separately estimates query-image-to-image and query-text-to-image similarities, performing late fusion to upweight images that satisfy both queries, while down-weighting those that exhibit high similarity with only one of the two. Each individual similarity is further improved by a set of components that are simple and intuitive. BASIC sets a new state of the art on i-CIR but also on existing CIR datasets that follow a semantic-level class definition. Project page: https://vrg.fel.cvut.cz/icir/.
>
---
#### [new 005] FreeArt3D: Training-Free Articulated Object Generation using 3D Diffusion
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出FreeArt3D，一个无需训练的可动3D物体生成框架。针对现有方法依赖大量标注数据或生成质量低的问题，利用预训练静态3D扩散模型作为形状先验，将姿态作为额外生成维度，仅需少量多姿态图像即可联合优化几何、纹理与运动参数，实现高质量、高保真且泛化性强的可动3D物体生成。**

- **链接: [http://arxiv.org/pdf/2510.25765v1](http://arxiv.org/pdf/2510.25765v1)**

> **作者:** Chuhao Chen; Isabella Liu; Xinyue Wei; Hao Su; Minghua Liu
>
> **摘要:** Articulated 3D objects are central to many applications in robotics, AR/VR, and animation. Recent approaches to modeling such objects either rely on optimization-based reconstruction pipelines that require dense-view supervision or on feed-forward generative models that produce coarse geometric approximations and often overlook surface texture. In contrast, open-world 3D generation of static objects has achieved remarkable success, especially with the advent of native 3D diffusion models such as Trellis. However, extending these methods to articulated objects by training native 3D diffusion models poses significant challenges. In this work, we present FreeArt3D, a training-free framework for articulated 3D object generation. Instead of training a new model on limited articulated data, FreeArt3D repurposes a pre-trained static 3D diffusion model (e.g., Trellis) as a powerful shape prior. It extends Score Distillation Sampling (SDS) into the 3D-to-4D domain by treating articulation as an additional generative dimension. Given a few images captured in different articulation states, FreeArt3D jointly optimizes the object's geometry, texture, and articulation parameters without requiring task-specific training or access to large-scale articulated datasets. Our method generates high-fidelity geometry and textures, accurately predicts underlying kinematic structures, and generalizes well across diverse object categories. Despite following a per-instance optimization paradigm, FreeArt3D completes in minutes and significantly outperforms prior state-of-the-art approaches in both quality and versatility.
>
---
#### [new 006] Proper Body Landmark Subset Enables More Accurate and 5X Faster Recognition of Isolated Signs in LIBRAS
- **分类: cs.CV**

- **简介: 该论文针对巴西手语（LIBRAS）孤立手势识别任务，解决传统骨架方法计算慢、精度低的问题。通过选择合适的身体关键点子集并结合样条插值修复缺失点，实现5倍加速且保持高准确率，提升了识别效率与实用性。**

- **链接: [http://arxiv.org/pdf/2510.24887v1](http://arxiv.org/pdf/2510.24887v1)**

> **作者:** Daniele L. V. dos Santos; Thiago B. Pereira; Carlos Eduardo G. R. Alves; Richard J. M. G. Tello; Francisco de A. Boldt; Thiago M. Paixão
>
> **备注:** Submitted to Int. Conf. on Computer Vision Theory and Applications (VISAPP 2026)
>
> **摘要:** This paper investigates the feasibility of using lightweight body landmark detection for the recognition of isolated signs in Brazilian Sign Language (LIBRAS). Although the skeleton-based approach by Alves et al. (2024) enabled substantial improvements in recognition performance, the use of OpenPose for landmark extraction hindered time performance. In a preliminary investigation, we observed that simply replacing OpenPose with the lightweight MediaPipe, while improving processing speed, significantly reduced accuracy. To overcome this limitation, we explored landmark subset selection strategies aimed at optimizing recognition performance. Experimental results showed that a proper landmark subset achieves comparable or superior performance to state-of-the-art methods while reducing processing time by more than 5X compared to Alves et al. (2024). As an additional contribution, we demonstrated that spline-based imputation effectively mitigates missing landmark issues, leading to substantial accuracy gains. These findings highlight that careful landmark selection, combined with simple imputation techniques, enables efficient and accurate isolated sign recognition, paving the way for scalable Sign Language Recognition systems.
>
---
#### [new 007] Learning Disentangled Speech- and Expression-Driven Blendshapes for 3D Talking Face Animation
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文聚焦于3D说话人脸动画生成任务，旨在解决情感表达与口型同步难以兼顾的问题。利用语音和表情驱动的可分离混合形状模型，结合两个数据集，通过稀疏约束实现双因素解耦，提升表情自然性与口型准确性，支持FLAME模型驱动3D高斯头像动画。**

- **链接: [http://arxiv.org/pdf/2510.25234v1](http://arxiv.org/pdf/2510.25234v1)**

> **作者:** Yuxiang Mao; Zhijie Zhang; Zhiheng Zhang; Jiawei Liu; Chen Zeng; Shihong Xia
>
> **备注:** 18 pages, 6 figures, accepted to ICXR 2025 conference
>
> **摘要:** Expressions are fundamental to conveying human emotions. With the rapid advancement of AI-generated content (AIGC), realistic and expressive 3D facial animation has become increasingly crucial. Despite recent progress in speech-driven lip-sync for talking-face animation, generating emotionally expressive talking faces remains underexplored. A major obstacle is the scarcity of real emotional 3D talking-face datasets due to the high cost of data capture. To address this, we model facial animation driven by both speech and emotion as a linear additive problem. Leveraging a 3D talking-face dataset with neutral expressions (VOCAset) and a dataset of 3D expression sequences (Florence4D), we jointly learn a set of blendshapes driven by speech and emotion. We introduce a sparsity constraint loss to encourage disentanglement between the two types of blendshapes while allowing the model to capture inherent secondary cross-domain deformations present in the training data. The learned blendshapes can be further mapped to the expression and jaw pose parameters of the FLAME model, enabling the animation of 3D Gaussian avatars. Qualitative and quantitative experiments demonstrate that our method naturally generates talking faces with specified expressions while maintaining accurate lip synchronization. Perceptual studies further show that our approach achieves superior emotional expressivity compared to existing methods, without compromising lip-sync quality.
>
---
#### [new 008] A Study on Inference Latency for Vision Transformers on Mobile Devices
- **分类: cs.CV; cs.LG; cs.PF**

- **简介: 该论文研究视觉变压器（ViT）在移动设备上的推理延迟问题，旨在揭示影响延迟的关键因素。通过对比190个真实ViT与102个CNN，构建包含1000个合成ViT的延迟数据集，提出可准确预测新ViT延迟的方法，为移动端模型优化提供支持。**

- **链接: [http://arxiv.org/pdf/2510.25166v1](http://arxiv.org/pdf/2510.25166v1)**

> **作者:** Zhuojin Li; Marco Paolieri; Leana Golubchik
>
> **备注:** To appear in Springer LNICST, volume 663, Proceedings of VALUETOOLS 2024
>
> **摘要:** Given the significant advances in machine learning techniques on mobile devices, particularly in the domain of computer vision, in this work we quantitatively study the performance characteristics of 190 real-world vision transformers (ViTs) on mobile devices. Through a comparison with 102 real-world convolutional neural networks (CNNs), we provide insights into the factors that influence the latency of ViT architectures on mobile devices. Based on these insights, we develop a dataset including measured latencies of 1000 synthetic ViTs with representative building blocks and state-of-the-art architectures from two machine learning frameworks and six mobile platforms. Using this dataset, we show that inference latency of new ViTs can be predicted with sufficient accuracy for real-world applications.
>
---
#### [new 009] Target-Guided Bayesian Flow Networks for Quantitatively Constrained CAD Generation
- **分类: cs.CV**

- **简介: 该论文针对参数化CAD生成中多模态数据与定量约束难处理的问题，提出目标引导的贝叶斯流网络（TGBFN）。通过统一连续可微的参数空间建模，引入引导贝叶斯流控制设计属性，实现高保真、条件感知的CAD序列生成。**

- **链接: [http://arxiv.org/pdf/2510.25163v1](http://arxiv.org/pdf/2510.25163v1)**

> **作者:** Wenhao Zheng; Chenwei Sun; Wenbo Zhang; Jiancheng Lv; Xianggen Liu
>
> **摘要:** Deep generative models, such as diffusion models, have shown promising progress in image generation and audio generation via simplified continuity assumptions. However, the development of generative modeling techniques for generating multi-modal data, such as parametric CAD sequences, still lags behind due to the challenges in addressing long-range constraints and parameter sensitivity. In this work, we propose a novel framework for quantitatively constrained CAD generation, termed Target-Guided Bayesian Flow Network (TGBFN). For the first time, TGBFN handles the multi-modality of CAD sequences (i.e., discrete commands and continuous parameters) in a unified continuous and differentiable parameter space rather than in the discrete data space. In addition, TGBFN penetrates the parameter update kernel and introduces a guided Bayesian flow to control the CAD properties. To evaluate TGBFN, we construct a new dataset for quantitatively constrained CAD generation. Extensive comparisons across single-condition and multi-condition constrained generation tasks demonstrate that TGBFN achieves state-of-the-art performance in generating high-fidelity, condition-aware CAD sequences. The code is available at https://github.com/scu-zwh/TGBFN.
>
---
#### [new 010] DRIP: Dynamic patch Reduction via Interpretable Pooling
- **分类: cs.CV**

- **简介: 该论文提出DRIP方法，针对视觉语言模型预训练效率低的问题，通过动态融合深层视觉编码器中的图像块，实现可解释的池化。在图像分类与零样本任务中显著降低计算量（GFLOPs），同时保持性能，适用于从头训练及生物科学数据持续预训练。**

- **链接: [http://arxiv.org/pdf/2510.25067v1](http://arxiv.org/pdf/2510.25067v1)**

> **作者:** Yusen Peng; Sachin Kumar
>
> **摘要:** Recently, the advances in vision-language models, including contrastive pretraining and instruction tuning, have greatly pushed the frontier of multimodal AI. However, owing to the large-scale and hence expensive pretraining, the efficiency concern has discouraged researchers from attempting to pretrain a vision language model from scratch. In this work, we propose Dynamic patch Reduction via Interpretable Pooling (DRIP), which adapts to the input images and dynamically merges tokens in the deeper layers of a visual encoder. Our results on both ImageNet training from scratch and CLIP contrastive pretraining demonstrate a significant GFLOP reduction while maintaining comparable classification/zero-shot performance. To further validate our proposed method, we conduct continual pretraining on a large biology dataset, extending its impact into scientific domains.
>
---
#### [new 011] Mapping and Classification of Trees Outside Forests using Deep Learning
- **分类: cs.CV; I.4.6**

- **简介: 该论文属于遥感图像语义分割任务，旨在解决树木非森林（TOF）分类中类别模糊、区域适应性差的问题。研究基于高分辨率影像，对比多种深度学习模型，对四类木本植被进行精准分类，验证了视觉变换器在捕捉空间上下文中的优势，并强调了多区域数据对泛化能力的重要性。**

- **链接: [http://arxiv.org/pdf/2510.25239v1](http://arxiv.org/pdf/2510.25239v1)**

> **作者:** Moritz Lucas; Hamid Ebrahimy; Viacheslav Barkov; Ralf Pecenka; Kai-Uwe Kühnberger; Björn Waske
>
> **摘要:** Trees Outside Forests (TOF) play an important role in agricultural landscapes by supporting biodiversity, sequestering carbon, and regulating microclimates. Yet, most studies have treated TOF as a single class or relied on rigid rule-based thresholds, limiting ecological interpretation and adaptability across regions. To address this, we evaluate deep learning for TOF classification using a newly generated dataset and high-resolution aerial imagery from four agricultural landscapes in Germany. Specifically, we compare convolutional neural networks (CNNs), vision transformers, and hybrid CNN-transformer models across six semantic segmentation architectures (ABCNet, LSKNet, FT-UNetFormer, DC-Swin, BANet, and U-Net) to map four categories of woody vegetation: Forest, Patch, Linear, and Tree, derived from previous studies and governmental products. Overall, the models achieved good classification accuracy across the four landscapes, with the FT-UNetFormer performing best (mean Intersection-over-Union 0.74; mean F1 score 0.84), underscoring the importance of spatial context understanding in TOF mapping and classification. Our results show good results for Forest and Linear class and reveal challenges particularly in classifying complex structures with high edge density, notably the Patch and Tree class. Our generalization experiments highlight the need for regionally diverse training data to ensure reliable large-scale mapping. The dataset and code are openly available at https://github.com/Moerizzy/TOFMapper
>
---
#### [new 012] FruitProm: Probabilistic Maturity Estimation and Detection of Fruits and Vegetables
- **分类: cs.CV**

- **简介: 该论文针对果蔬成熟度估计任务，解决传统分类方法忽视生物成熟连续性的问题。提出在RT-DETRv2基础上引入概率头，实现对成熟度的连续概率分布预测，同时输出均值与不确定性，提升评估精度与机器人决策可靠性。**

- **链接: [http://arxiv.org/pdf/2510.24885v1](http://arxiv.org/pdf/2510.24885v1)**

> **作者:** Sidharth Rai; Rahul Harsha Cheppally; Benjamin Vail; Keziban Yalçın Dokumacı; Ajay Sharda
>
> **备注:** Sidharth Rai, Rahul Harsha Cheppally contributed equally to this work
>
> **摘要:** Maturity estimation of fruits and vegetables is a critical task for agricultural automation, directly impacting yield prediction and robotic harvesting. Current deep learning approaches predominantly treat maturity as a discrete classification problem (e.g., unripe, ripe, overripe). This rigid formulation, however, fundamentally conflicts with the continuous nature of the biological ripening process, leading to information loss and ambiguous class boundaries. In this paper, we challenge this paradigm by reframing maturity estimation as a continuous, probabilistic learning task. We propose a novel architectural modification to the state-of-the-art, real-time object detector, RT-DETRv2, by introducing a dedicated probabilistic head. This head enables the model to predict a continuous distribution over the maturity spectrum for each detected object, simultaneously learning the mean maturity state and its associated uncertainty. This uncertainty measure is crucial for downstream decision-making in robotics, providing a confidence score for tasks like selective harvesting. Our model not only provides a far richer and more biologically plausible representation of plant maturity but also maintains exceptional detection performance, achieving a mean Average Precision (mAP) of 85.6\% on a challenging, large-scale fruit dataset. We demonstrate through extensive experiments that our probabilistic approach offers more granular and accurate maturity assessments than its classification-based counterparts, paving the way for more intelligent, uncertainty-aware automated systems in modern agriculture
>
---
#### [new 013] StreamingCoT: A Dataset for Temporal Dynamics and Multimodal Chain-of-Thought Reasoning in Streaming VideoQA
- **分类: cs.CV**

- **简介: 该论文提出StreamingCoT，首个面向流式视频问答的时序动态推理数据集。针对现有数据集静态标注、缺乏推理过程的问题，构建动态分层标注架构与显式链式推理生成范式，支持时序语义片段建模与逻辑一致的多模态推理，推动流式视频理解与复杂时序推理研究。**

- **链接: [http://arxiv.org/pdf/2510.25332v1](http://arxiv.org/pdf/2510.25332v1)**

> **作者:** Yuhang Hu; Zhenyu Yang; Shihan Wang; Shengsheng Qian; Bin Wen; Fan Yang; Tingting Gao; Changsheng Xu
>
> **摘要:** The rapid growth of streaming video applications demands multimodal models with enhanced capabilities for temporal dynamics understanding and complex reasoning. However, current Video Question Answering (VideoQA) datasets suffer from two critical limitations: 1) Static annotation mechanisms fail to capture the evolving nature of answers in temporal video streams, and 2) The absence of explicit reasoning process annotations restricts model interpretability and logical deduction capabilities. To address these challenges, We introduce StreamingCoT, the first dataset explicitly designed for temporally evolving reasoning in streaming VideoQA and multimodal Chain-of-Thought (CoT) tasks. Our framework first establishes a dynamic hierarchical annotation architecture that generates per-second dense descriptions and constructs temporally-dependent semantic segments through similarity fusion, paired with question-answer sets constrained by temporal evolution patterns. We further propose an explicit reasoning chain generation paradigm that extracts spatiotemporal objects via keyframe semantic alignment, derives object state transition-based reasoning paths using large language models, and ensures logical coherence through human-verified validation. This dataset establishes a foundation for advancing research in streaming video understanding, complex temporal reasoning, and multimodal inference. Our StreamingCoT and its construction toolkit can be accessed at https://github.com/Fleeting-hyh/StreamingCoT.
>
---
#### [new 014] VividCam: Learning Unconventional Camera Motions from Virtual Synthetic Videos
- **分类: cs.CV**

- **简介: 该论文提出VividCam，旨在让扩散模型从合成视频中学习非常规相机运动。针对真实数据难获取的问题，利用低多边形3D场景生成简单合成数据，通过解耦策略分离运动与外观，实现复杂、精确的相机运动控制，无需依赖真实拍摄视频。**

- **链接: [http://arxiv.org/pdf/2510.24904v1](http://arxiv.org/pdf/2510.24904v1)**

> **作者:** Qiucheng Wu; Handong Zhao; Zhixin Shu; Jing Shi; Yang Zhang; Shiyu Chang
>
> **备注:** 19 pages, 9 figures
>
> **摘要:** Although recent text-to-video generative models are getting more capable of following external camera controls, imposed by either text descriptions or camera trajectories, they still struggle to generalize to unconventional camera motions, which is crucial in creating truly original and artistic videos. The challenge lies in the difficulty of finding sufficient training videos with the intended uncommon camera motions. To address this challenge, we propose VividCam, a training paradigm that enables diffusion models to learn complex camera motions from synthetic videos, releasing the reliance on collecting realistic training videos. VividCam incorporates multiple disentanglement strategies that isolates camera motion learning from synthetic appearance artifacts, ensuring more robust motion representation and mitigating domain shift. We demonstrate that our design synthesizes a wide range of precisely controlled and complex camera motions using surprisingly simple synthetic data. Notably, this synthetic data often consists of basic geometries within a low-poly 3D scene and can be efficiently rendered by engines like Unity. Our video results can be found in https://wuqiuche.github.io/VividCamDemoPage/ .
>
---
#### [new 015] Auto3DSeg for Brain Tumor Segmentation from 3D MRI in BraTS 2023 Challenge
- **分类: cs.CV**

- **简介: 该论文针对脑肿瘤3D MRI分割任务，使用MONAI的Auto3DSeg框架参与BraTS 2023五项挑战，解决了多类型脑肿瘤精准分割问题。通过自动化深度学习方法，在三项挑战中获第一名，两项获第二名，验证了方法的有效性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.25058v1](http://arxiv.org/pdf/2510.25058v1)**

> **作者:** Andriy Myronenko; Dong Yang; Yufan He; Daguang Xu
>
> **备注:** BraTS23 winner
>
> **摘要:** In this work, we describe our solution to the BraTS 2023 cluster of challenges using Auto3DSeg from MONAI. We participated in all 5 segmentation challenges, and achieved the 1st place results in three of them: Brain Metastasis, Brain Meningioma, BraTS-Africa challenges, and the 2nd place results in the remaining two: Adult and Pediatic Glioma challenges.
>
---
#### [new 016] Vision-Language Integration for Zero-Shot Scene Understanding in Real-World Environments
- **分类: cs.CV**

- **简介: 该论文聚焦零样本场景理解任务，旨在解决真实环境中新物体、动作与上下文难以识别的问题。通过融合视觉编码器与语言模型，构建跨模态对齐的统一框架，实现视觉与语言在共享空间中的联合推理，显著提升零样本下对象识别、活动检测与场景描述性能。**

- **链接: [http://arxiv.org/pdf/2510.25070v1](http://arxiv.org/pdf/2510.25070v1)**

> **作者:** Manjunath Prasad Holenarasipura Rajiv; B. M. Vidyavathi
>
> **备注:** Preprint under review at IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2025
>
> **摘要:** Zero-shot scene understanding in real-world settings presents major challenges due to the complexity and variability of natural scenes, where models must recognize new objects, actions, and contexts without prior labeled examples. This work proposes a vision-language integration framework that unifies pre-trained visual encoders (e.g., CLIP, ViT) and large language models (e.g., GPT-based architectures) to achieve semantic alignment between visual and textual modalities. The goal is to enable robust zero-shot comprehension of scenes by leveraging natural language as a bridge to generalize over unseen categories and contexts. Our approach develops a unified model that embeds visual inputs and textual prompts into a shared space, followed by multimodal fusion and reasoning layers for contextual interpretation. Experiments on Visual Genome, COCO, ADE20K, and custom real-world datasets demonstrate significant gains over state-of-the-art zero-shot models in object recognition, activity detection, and scene captioning. The proposed system achieves up to 18% improvement in top-1 accuracy and notable gains in semantic coherence metrics, highlighting the effectiveness of cross-modal alignment and language grounding in enhancing generalization for real-world scene understanding.
>
---
#### [new 017] The Underappreciated Power of Vision Models for Graph Structural Understanding
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究视觉模型在图结构理解中的潜力，针对传统图神经网络（GNN）难以捕捉全局结构的问题，提出新基准GraphAbstract，评估模型对图的组织模式、对称性等整体特性的感知能力。结果表明，视觉模型在全局结构理解和跨尺度泛化上显著优于GNN，揭示其在图分析中被低估的优势。**

- **链接: [http://arxiv.org/pdf/2510.24788v1](http://arxiv.org/pdf/2510.24788v1)**

> **作者:** Xinjian Zhao; Wei Pang; Zhongkai Xue; Xiangru Jian; Lei Zhang; Yaoyao Xu; Xiaozhuang Song; Shu Wu; Tianshu Yu
>
> **备注:** NeurIPS 2025
>
> **摘要:** Graph Neural Networks operate through bottom-up message-passing, fundamentally differing from human visual perception, which intuitively captures global structures first. We investigate the underappreciated potential of vision models for graph understanding, finding they achieve performance comparable to GNNs on established benchmarks while exhibiting distinctly different learning patterns. These divergent behaviors, combined with limitations of existing benchmarks that conflate domain features with topological understanding, motivate our introduction of GraphAbstract. This benchmark evaluates models' ability to perceive global graph properties as humans do: recognizing organizational archetypes, detecting symmetry, sensing connectivity strength, and identifying critical elements. Our results reveal that vision models significantly outperform GNNs on tasks requiring holistic structural understanding and maintain generalizability across varying graph scales, while GNNs struggle with global pattern abstraction and degrade with increasing graph size. This work demonstrates that vision models possess remarkable yet underutilized capabilities for graph structural understanding, particularly for problems requiring global topological awareness and scale-invariant reasoning. These findings open new avenues to leverage this underappreciated potential for developing more effective graph foundation models for tasks dominated by holistic pattern recognition.
>
---
#### [new 018] Perception, Understanding and Reasoning, A Multimodal Benchmark for Video Fake News Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视频假新闻检测（VFND）任务，提出多模态基准MVFNDB，涵盖10项细粒度任务，评估模型的感知、理解与推理能力。通过9730个标注问题和新框架MVFND-CoT，揭示视频处理策略与模型能力对检测准确率的影响，推动多模态大模型在假新闻检测中的可解释性发展。**

- **链接: [http://arxiv.org/pdf/2510.24816v1](http://arxiv.org/pdf/2510.24816v1)**

> **作者:** Cui Yakun; Fushuo Huo; Weijie Shi; Juntao Dai; Hang Du; Zhenghao Zhu; Sirui Han; Yike Guo
>
> **摘要:** The advent of multi-modal large language models (MLLMs) has greatly advanced research into applications for Video fake news detection (VFND) tasks. Traditional video-based FND benchmarks typically focus on the accuracy of the final decision, often failing to provide fine-grained assessments for the entire detection process, making the detection process a black box. Therefore, we introduce the MVFNDB (Multi-modal Video Fake News Detection Benchmark) based on the empirical analysis, which provides foundation for tasks definition. The benchmark comprises 10 tasks and is meticulously crafted to probe MLLMs' perception, understanding, and reasoning capacities during detection, featuring 9730 human-annotated video-related questions based on a carefully constructed taxonomy ability of VFND. To validate the impact of combining multiple features on the final results, we design a novel framework named MVFND-CoT, which incorporates both creator-added content and original shooting footage reasoning. Building upon the benchmark, we conduct an in-depth analysis of the deeper factors influencing accuracy, including video processing strategies and the alignment between video features and model capabilities. We believe this benchmark will lay a solid foundation for future evaluations and advancements of MLLMs in the domain of video fake news detection.
>
---
#### [new 019] Ming-Flash-Omni: A Sparse, Unified Architecture for Multimodal Perception and Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Ming-Flash-Omni，一种稀疏统一架构，用于多模态感知与生成。针对高效扩展模型能力与统一多模态智能的问题，采用稀疏MoE结构，实现高计算效率与强多模态性能，在语音识别、图像生成与分割任务上均达领先水平。**

- **链接: [http://arxiv.org/pdf/2510.24821v1](http://arxiv.org/pdf/2510.24821v1)**

> **作者:** Inclusion AI; :; Bowen Ma; Cheng Zou; Canxiang Yan; Chunxiang Jin; Chunjie Shen; Dandan Zheng; Fudong Wang; Furong Xu; GuangMing Yao; Jun Zhou; Jingdong Chen; Jianing Li; Jianxin Sun; Jiajia Liu; Jianjiang Zhu; Jianping Jiang; Jun Peng; Kaixiang Ji; Kaimeng Ren; Libin Wang; Lixiang Ru; Longhua Tan; Lan Wang; Mochen Bai; Ning Gao; Qingpei Guo; Qinglong Zhang; Qiang Xu; Rui Liu; Ruijie Xiong; Ruobing Zheng; Sirui Gao; Tianqi Li; Tinghao Liu; Weilong Chai; Xinyu Xiao; Xiaomei Wang; Xiaolong Wang; Xiao Lu; Xiaoyu Li; Xingning Dong; Xuzheng Yu; Yi Yuan; Yuting Gao; Yuting Xiao; Yunxiao Sun; Yipeng Chen; Yifan Mao; Yifei Wu; Yongjie Lyu; Ziping Ma; Zhiqiang Fang; Zhihao Qiu; Ziyuan Huang; Zizheng Yang; Zhengyu He
>
> **备注:** 18 pages, 5 figures
>
> **摘要:** We propose Ming-Flash-Omni, an upgraded version of Ming-Omni, built upon a sparser Mixture-of-Experts (MoE) variant of Ling-Flash-2.0 with 100 billion total parameters, of which only 6.1 billion are active per token. This architecture enables highly efficient scaling (dramatically improving computational efficiency while significantly expanding model capacity) and empowers stronger unified multimodal intelligence across vision, speech, and language, representing a key step toward Artificial General Intelligence (AGI). Compared to its predecessor, the upgraded version exhibits substantial improvements across multimodal understanding and generation. We significantly advance speech recognition capabilities, achieving state-of-the-art performance in contextual ASR and highly competitive results in dialect-aware ASR. In image generation, Ming-Flash-Omni introduces high-fidelity text rendering and demonstrates marked gains in scene consistency and identity preservation during image editing. Furthermore, Ming-Flash-Omni introduces generative segmentation, a capability that not only achieves strong standalone segmentation performance but also enhances spatial control in image generation and improves editing consistency. Notably, Ming-Flash-Omni achieves state-of-the-art results in text-to-image generation and generative segmentation, and sets new records on all 12 contextual ASR benchmarks, all within a single unified architecture.
>
---
#### [new 020] PISA-Bench: The PISA Index as a Multilingual and Multimodal Metric for the Evaluation of Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出PISA-Bench，一个基于国际学生评估项目（PISA）的多语言多模态基准，旨在解决现有视觉语言模型评测数据质量低、语言单一的问题。通过人工提取并翻译六种语言的高质量测试题，评估模型在跨语言、空间几何推理等方面的能力，揭示小模型与非英语语境下的性能瓶颈。**

- **链接: [http://arxiv.org/pdf/2510.24792v1](http://arxiv.org/pdf/2510.24792v1)**

> **作者:** Patrick Haller; Fabio Barth; Jonas Golde; Georg Rehm; Alan Akbik
>
> **备注:** 8 pages, 11 tables and figures
>
> **摘要:** Vision-language models (VLMs) have demonstrated remarkable progress in multimodal reasoning. However, existing benchmarks remain limited in terms of high-quality, human-verified examples. Many current datasets rely on synthetically generated content by large language models (LLMs). Furthermore, most datasets are limited to English, as manual quality assurance of translated samples is time-consuming and costly. To fill this gap, we introduce PISA-Bench, a multilingual benchmark derived from English examples of the expert-created PISA tests, a unified framework for the assessment of student competencies in over eighty countries. Each example consists of human-extracted instructions, questions, answer options, and images, enriched with question type categories, and has been translated from English into five additional languages (Spanish, German, Chinese, French, and Italian), resulting in a fully parallel corpus covering six languages. We evaluate state-of-the-art vision-language models on PISA-Bench and find that especially small models (<20B parameters) fail to achieve high test scores. We further find substantial performance degradation on non-English splits as well as high error-rates when models are tasked with spatial and geometric reasoning. By releasing the dataset and evaluation framework, we provide a resource for advancing research on multilingual multimodal reasoning.
>
---
#### [new 021] A Survey on Efficient Vision-Language-Action Models
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文聚焦于高效视觉-语言-动作模型（Efficient VLAs），旨在解决大模型在机器人领域部署中计算与数据成本高的问题。通过构建统一分类框架，系统梳理了高效模型设计、训练与数据收集三方面技术，总结现状、挑战与未来方向，为该领域提供全面参考。**

- **链接: [http://arxiv.org/pdf/2510.24795v1](http://arxiv.org/pdf/2510.24795v1)**

> **作者:** Zhaoshu Yu; Bo Wang; Pengpeng Zeng; Haonan Zhang; Ji Zhang; Lianli Gao; Jingkuan Song; Nicu Sebe; Heng Tao Shen
>
> **备注:** 26 pages, 8 figures
>
> **摘要:** Vision-Language-Action models (VLAs) represent a significant frontier in embodied intelligence, aiming to bridge digital knowledge with physical-world interaction. While these models have demonstrated remarkable generalist capabilities, their deployment is severely hampered by the substantial computational and data requirements inherent to their underlying large-scale foundation models. Motivated by the urgent need to address these challenges, this survey presents the first comprehensive review of Efficient Vision-Language-Action models (Efficient VLAs) across the entire data-model-training process. Specifically, we introduce a unified taxonomy to systematically organize the disparate efforts in this domain, categorizing current techniques into three core pillars: (1) Efficient Model Design, focusing on efficient architectures and model compression; (2) Efficient Training, which reduces computational burdens during model learning; and (3) Efficient Data Collection, which addresses the bottlenecks in acquiring and utilizing robotic data. Through a critical review of state-of-the-art methods within this framework, this survey not only establishes a foundational reference for the community but also summarizes representative applications, delineates key challenges, and charts a roadmap for future research. We maintain a continuously updated project page to track our latest developments: https://evla-survey.github.io/
>
---
#### [new 022] Informative Sample Selection Model for Skeleton-based Action Recognition with Limited Training Samples
- **分类: cs.CV**

- **简介: 该论文针对3D动作识别中标注数据稀缺的问题，提出基于强化学习与双曲空间的主动学习方法。通过将样本选择建模为马尔可夫决策过程，利用双曲空间提升表征能力，并引入元调优策略加速实际部署，有效提升了有限标注下的识别性能。**

- **链接: [http://arxiv.org/pdf/2510.25345v1](http://arxiv.org/pdf/2510.25345v1)**

> **作者:** Zhigang Tu; Zhengbo Zhang; Jia Gong; Junsong Yuan; Bo Du
>
> **备注:** Accepted by IEEE Transactions on Image Processing (TIP), 2025
>
> **摘要:** Skeleton-based human action recognition aims to classify human skeletal sequences, which are spatiotemporal representations of actions, into predefined categories. To reduce the reliance on costly annotations of skeletal sequences while maintaining competitive recognition accuracy, the task of 3D Action Recognition with Limited Training Samples, also known as semi-supervised 3D Action Recognition, has been proposed. In addition, active learning, which aims to proactively select the most informative unlabeled samples for annotation, has been explored in semi-supervised 3D Action Recognition for training sample selection. Specifically, researchers adopt an encoder-decoder framework to embed skeleton sequences into a latent space, where clustering information, combined with a margin-based selection strategy using a multi-head mechanism, is utilized to identify the most informative sequences in the unlabeled set for annotation. However, the most representative skeleton sequences may not necessarily be the most informative for the action recognizer, as the model may have already acquired similar knowledge from previously seen skeleton samples. To solve it, we reformulate Semi-supervised 3D action recognition via active learning from a novel perspective by casting it as a Markov Decision Process (MDP). Built upon the MDP framework and its training paradigm, we train an informative sample selection model to intelligently guide the selection of skeleton sequences for annotation. To enhance the representational capacity of the factors in the state-action pairs within our method, we project them from Euclidean space to hyperbolic space. Furthermore, we introduce a meta tuning strategy to accelerate the deployment of our method in real-world scenarios. Extensive experiments on three 3D action recognition benchmarks demonstrate the effectiveness of our method.
>
---
#### [new 023] Diffusion-Driven Progressive Target Manipulation for Source-Free Domain Adaptation
- **分类: cs.CV**

- **简介: 该论文针对源域无关域适应（SFDA）任务，解决大领域差异下伪标签不可靠与生成数据失真问题。提出基于扩散模型的渐进式目标域操纵框架（DPTM），通过可信/非可信样本划分，利用扩散模型对非可信样本进行语义变换与分布保持，并迭代优化伪目标域，显著提升适应性能。**

- **链接: [http://arxiv.org/pdf/2510.25279v1](http://arxiv.org/pdf/2510.25279v1)**

> **作者:** Yuyang Huang; Yabo Chen; Junyu Zhou; Wenrui Dai; Xiaopeng Zhang; Junni Zou; Hongkai Xiong; Qi Tian
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Source-free domain adaptation (SFDA) is a challenging task that tackles domain shifts using only a pre-trained source model and unlabeled target data. Existing SFDA methods are restricted by the fundamental limitation of source-target domain discrepancy. Non-generation SFDA methods suffer from unreliable pseudo-labels in challenging scenarios with large domain discrepancies, while generation-based SFDA methods are evidently degraded due to enlarged domain discrepancies in creating pseudo-source data. To address this limitation, we propose a novel generation-based framework named Diffusion-Driven Progressive Target Manipulation (DPTM) that leverages unlabeled target data as references to reliably generate and progressively refine a pseudo-target domain for SFDA. Specifically, we divide the target samples into a trust set and a non-trust set based on the reliability of pseudo-labels to sufficiently and reliably exploit their information. For samples from the non-trust set, we develop a manipulation strategy to semantically transform them into the newly assigned categories, while simultaneously maintaining them in the target distribution via a latent diffusion model. Furthermore, we design a progressive refinement mechanism that progressively reduces the domain discrepancy between the pseudo-target domain and the real target domain via iterative refinement. Experimental results demonstrate that DPTM outperforms existing methods by a large margin and achieves state-of-the-art performance on four prevailing SFDA benchmark datasets with different scales. Remarkably, DPTM can significantly enhance the performance by up to 18.6% in scenarios with large source-target gaps.
>
---
#### [new 024] Towards Real-Time Inference of Thin Liquid Film Thickness Profiles from Interference Patterns Using Vision Transformers
- **分类: cs.CV**

- **简介: 该论文针对薄液膜厚度反演这一病态逆问题，提出基于视觉变压器的实时厚度重建方法。利用合成与实验数据训练模型，直接从干涉图中快速、自动地恢复出动态泪膜厚度，有效解决相位模糊与噪声干扰问题，实现消费级硬件上的实时诊断，推动干眼症等疾病的非侵入式监测。**

- **链接: [http://arxiv.org/pdf/2510.25157v1](http://arxiv.org/pdf/2510.25157v1)**

> **作者:** Gautam A. Viruthagiri; Arnuv Tandon; Gerald G. Fuller; Vinny Chandran Suja
>
> **备注:** 6 pages, 2 figures, will be updated
>
> **摘要:** Thin film interferometry is a powerful technique for non-invasively measuring liquid film thickness with applications in ophthalmology, but its clinical translation is hindered by the challenges in reconstructing thickness profiles from interference patterns - an ill-posed inverse problem complicated by phase periodicity, imaging noise and ambient artifacts. Traditional reconstruction methods are either computationally intensive, sensitive to noise, or require manual expert analysis, which is impractical for real-time diagnostics. To address this challenge, here we present a vision transformer-based approach for real-time inference of thin liquid film thickness profiles directly from isolated interferograms. Trained on a hybrid dataset combining physiologically-relevant synthetic and experimental tear film data, our model leverages long-range spatial correlations to resolve phase ambiguities and reconstruct temporally coherent thickness profiles in a single forward pass from dynamic interferograms acquired in vivo and ex vivo. The network demonstrates state-of-the-art performance on noisy, rapidly-evolving films with motion artifacts, overcoming limitations of conventional phase-unwrapping and iterative fitting methods. Our data-driven approach enables automated, consistent thickness reconstruction at real-time speeds on consumer hardware, opening new possibilities for continuous monitoring of pre-lens ocular tear films and non-invasive diagnosis of conditions such as the dry eye disease.
>
---
#### [new 025] Point-level Uncertainty Evaluation of Mobile Laser Scanning Point Clouds
- **分类: cs.CV; cs.LG; cs.RO; eess.IV**

- **简介: 该论文针对移动激光扫描点云的点级不确定性评估问题，提出基于随机森林和XGBoost的机器学习框架，利用局部几何特征预测误差。通过空间分区训练避免数据泄露，有效捕捉非线性关系，显著提升评估精度，为大规模点云质量控制提供可扩展的数据驱动方法。**

- **链接: [http://arxiv.org/pdf/2510.24773v1](http://arxiv.org/pdf/2510.24773v1)**

> **作者:** Ziyang Xu; Olaf Wysocki; Christoph Holst
>
> **摘要:** Reliable quantification of uncertainty in Mobile Laser Scanning (MLS) point clouds is essential for ensuring the accuracy and credibility of downstream applications such as 3D mapping, modeling, and change analysis. Traditional backward uncertainty modeling heavily rely on high-precision reference data, which are often costly or infeasible to obtain at large scales. To address this issue, this study proposes a machine learning-based framework for point-level uncertainty evaluation that learns the relationship between local geometric features and point-level errors. The framework is implemented using two ensemble learning models, Random Forest (RF) and XGBoost, which are trained and validated on a spatially partitioned real-world dataset to avoid data leakage. Experimental results demonstrate that both models can effectively capture the nonlinear relationships between geometric characteristics and uncertainty, achieving mean ROC-AUC values above 0.87. The analysis further reveals that geometric features describing elevation variation, point density, and local structural complexity play a dominant role in predicting uncertainty. The proposed framework offers a data-driven perspective of uncertainty evaluation, providing a scalable and adaptable foundation for future quality control and error analysis of large-scale point clouds.
>
---
#### [new 026] Cross-Enhanced Multimodal Fusion of Eye-Tracking and Facial Features for Alzheimer's Disease Diagnosis
- **分类: cs.CV; cs.AI; eess.IV; 68T07; I.2; H.5.1**

- **简介: 该论文属于阿尔茨海默病（AD）辅助诊断任务，旨在融合眼动与面部特征以提升诊断准确性。提出跨增强融合框架，通过交叉注意力与方向感知卷积模块，有效建模模态间交互与特定特征，实现95.11%的分类准确率，显著优于传统方法。**

- **链接: [http://arxiv.org/pdf/2510.24777v1](http://arxiv.org/pdf/2510.24777v1)**

> **作者:** Yujie Nie; Jianzhang Ni; Yonglong Ye; Yuan-Ting Zhang; Yun Kwok Wing; Xiangqing Xu; Xin Ma; Lizhou Fan
>
> **备注:** 35 pages, 8 figures, and 7 tables
>
> **摘要:** Accurate diagnosis of Alzheimer's disease (AD) is essential for enabling timely intervention and slowing disease progression. Multimodal diagnostic approaches offer considerable promise by integrating complementary information across behavioral and perceptual domains. Eye-tracking and facial features, in particular, are important indicators of cognitive function, reflecting attentional distribution and neurocognitive state. However, few studies have explored their joint integration for auxiliary AD diagnosis. In this study, we propose a multimodal cross-enhanced fusion framework that synergistically leverages eye-tracking and facial features for AD detection. The framework incorporates two key modules: (a) a Cross-Enhanced Fusion Attention Module (CEFAM), which models inter-modal interactions through cross-attention and global enhancement, and (b) a Direction-Aware Convolution Module (DACM), which captures fine-grained directional facial features via horizontal-vertical receptive fields. Together, these modules enable adaptive and discriminative multimodal representation learning. To support this work, we constructed a synchronized multimodal dataset, including 25 patients with AD and 25 healthy controls (HC), by recording aligned facial video and eye-tracking sequences during a visual memory-search paradigm, providing an ecologically valid resource for evaluating integration strategies. Extensive experiments on this dataset demonstrate that our framework outperforms traditional late fusion and feature concatenation methods, achieving a classification accuracy of 95.11% in distinguishing AD from HC, highlighting superior robustness and diagnostic performance by explicitly modeling inter-modal dependencies and modality-specific contributions.
>
---
#### [new 027] Visual Diversity and Region-aware Prompt Learning for Zero-shot HOI Detection
- **分类: cs.CV**

- **简介: 该论文针对零样本人-物交互检测任务，解决视觉多样性与类别间视觉混淆问题。提出VDRP框架，通过引入组内视觉差异和区域感知提示，增强提示学习对复杂视觉变化的适应性，显著提升模型在未见组合上的识别性能。**

- **链接: [http://arxiv.org/pdf/2510.25094v1](http://arxiv.org/pdf/2510.25094v1)**

> **作者:** Chanhyeong Yang; Taehoon Song; Jihwan Park; Hyunwoo J. Kim
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Zero-shot Human-Object Interaction detection aims to localize humans and objects in an image and recognize their interaction, even when specific verb-object pairs are unseen during training. Recent works have shown promising results using prompt learning with pretrained vision-language models such as CLIP, which align natural language prompts with visual features in a shared embedding space. However, existing approaches still fail to handle the visual complexity of interaction, including (1) intra-class visual diversity, where instances of the same verb appear in diverse poses and contexts, and (2) inter-class visual entanglement, where distinct verbs yield visually similar patterns. To address these challenges, we propose VDRP, a framework for Visual Diversity and Region-aware Prompt learning. First, we introduce a visual diversity-aware prompt learning strategy that injects group-wise visual variance into the context embedding. We further apply Gaussian perturbation to encourage the prompts to capture diverse visual variations of a verb. Second, we retrieve region-specific concepts from the human, object, and union regions. These are used to augment the diversity-aware prompt embeddings, yielding region-aware prompts that enhance verb-level discrimination. Experiments on the HICO-DET benchmark demonstrate that our method achieves state-of-the-art performance under four zero-shot evaluation settings, effectively addressing both intra-class diversity and inter-class visual entanglement. Code is available at https://github.com/mlvlab/VDRP.
>
---
#### [new 028] Efficient License Plate Recognition via Pseudo-Labeled Supervision with Grounding DINO and YOLOv8
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对复杂环境下车牌识别难题，提出基于YOLOv8与Grounding DINO的半监督方法。通过伪标签技术减少人工标注依赖，提升检测与识别精度，在多个数据集上实现高召回率与低字符错误率，有效增强ALPR系统性能。**

- **链接: [http://arxiv.org/pdf/2510.25032v1](http://arxiv.org/pdf/2510.25032v1)**

> **作者:** Zahra Ebrahimi Vargoorani; Amir Mohammad Ghoreyshi; Ching Yee Suen
>
> **备注:** 6 pages, 8 figures. Presented at 2025 IEEE International Workshop on Machine Learning for Signal Processing (MLSP), August 31 - September 3, 2025, Istanbul, Turkey
>
> **摘要:** Developing a highly accurate automatic license plate recognition system (ALPR) is challenging due to environmental factors such as lighting, rain, and dust. Additional difficulties include high vehicle speeds, varying camera angles, and low-quality or low-resolution images. ALPR is vital in traffic control, parking, vehicle tracking, toll collection, and law enforcement applications. This paper proposes a deep learning strategy using YOLOv8 for license plate detection and recognition tasks. This method seeks to enhance the performance of the model using datasets from Ontario, Quebec, California, and New York State. It achieved an impressive recall rate of 94% on the dataset from the Center for Pattern Recognition and Machine Intelligence (CENPARMI) and 91% on the UFPR-ALPR dataset. In addition, our method follows a semi-supervised learning framework, combining a small set of manually labeled data with pseudo-labels generated by Grounding DINO to train our detection model. Grounding DINO, a powerful vision-language model, automatically annotates many images with bounding boxes for license plates, thereby minimizing the reliance on labor-intensive manual labeling. By integrating human-verified and model-generated annotations, we can scale our dataset efficiently while maintaining label quality, which significantly enhances the training process and overall model performance. Furthermore, it reports character error rates for both datasets, providing additional insight into system performance.
>
---
#### [new 029] Prototype-Driven Adaptation for Few-Shot Object Detection
- **分类: cs.CV**

- **简介: 该论文针对少样本目标检测（FSOD）中的基类偏差与校准不稳问题，提出原型驱动对齐（PDA）方法。通过可学习原型空间与动态原型更新机制，在不增加参数的前提下提升新类识别性能，同时保持基类表现与低计算开销。**

- **链接: [http://arxiv.org/pdf/2510.25318v1](http://arxiv.org/pdf/2510.25318v1)**

> **作者:** Yushen Huang; Zhiming Wang
>
> **备注:** 7 pages,1 figure,2 tables,Preprint
>
> **摘要:** Few-shot object detection (FSOD) often suffers from base-class bias and unstable calibration when only a few novel samples are available. We propose Prototype-Driven Alignment (PDA), a lightweight, plug-in metric head for DeFRCN that provides a prototype-based "second opinion" complementary to the linear classifier. PDA maintains support-only prototypes in a learnable identity-initialized projection space and optionally applies prototype-conditioned RoI alignment to reduce geometric mismatch. During fine-tuning, prototypes can be adapted via exponential moving average(EMA) updates on labeled foreground RoIs-without introducing class-specific parameters-and are frozen at inference to ensure strict protocol compliance. PDA employs a best-of-K matching scheme to capture intra-class multi-modality and temperature-scaled fusion to combine metric similarities with detector logits. Experiments on VOC FSOD and GFSOD benchmarks show that PDA consistently improves novel-class performance with minimal impact on base classes and negligible computational overhead.
>
---
#### [new 030] Comparative Study of UNet-based Architectures for Liver Tumor Segmentation in Multi-Phase Contrast-Enhanced Computed Tomography
- **分类: cs.CV; cs.AI; I.4.6**

- **简介: 该论文聚焦于肝脏肿瘤分割任务，旨在提升多期增强CT图像中肝肿瘤的精准分割。研究比较了基于UNet的多种架构，发现结合残差网络与注意力机制（CBAM）的ResNetUNet3+模型表现最优，显著提升了分割精度与边界定位能力，为临床诊断提供了可靠工具。**

- **链接: [http://arxiv.org/pdf/2510.25522v1](http://arxiv.org/pdf/2510.25522v1)**

> **作者:** Doan-Van-Anh Ly; Thi-Thu-Hien Pham; Thanh-Hai Le
>
> **备注:** 27 pages, 8 figures
>
> **摘要:** Segmentation of liver structures in multi-phase contrast-enhanced computed tomography (CECT) plays a crucial role in computer-aided diagnosis and treatment planning for liver diseases, including tumor detection. In this study, we investigate the performance of UNet-based architectures for liver tumor segmentation, starting from the original UNet and extending to UNet3+ with various backbone networks. We evaluate ResNet, Transformer-based, and State-space (Mamba) backbones, all initialized with pretrained weights. Surprisingly, despite the advances in modern architecture, ResNet-based models consistently outperform Transformer- and Mamba-based alternatives across multiple evaluation metrics. To further improve segmentation quality, we introduce attention mechanisms into the backbone and observe that incorporating the Convolutional Block Attention Module (CBAM) yields the best performance. ResNetUNet3+ with CBAM module not only produced the best overlap metrics with a Dice score of 0.755 and IoU of 0.662, but also achieved the most precise boundary delineation, evidenced by the lowest HD95 distance of 77.911. The model's superiority was further cemented by its leading overall accuracy of 0.925 and specificity of 0.926, showcasing its robust capability in accurately identifying both lesion and healthy tissue. To further enhance interpretability, Grad-CAM visualizations were employed to highlight the region's most influential predictions, providing insights into its decision-making process. These findings demonstrate that classical ResNet architecture, when combined with modern attention modules, remain highly competitive for medical image segmentation tasks, offering a promising direction for liver tumor detection in clinical practice.
>
---
#### [new 031] RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
- **分类: cs.CV**

- **简介: 该论文聚焦实时目标检测任务，针对轻量化模型因特征表达弱导致性能瓶颈的问题，提出基于视觉基础模型的蒸馏框架。通过深度语义注入与梯度引导调制，实现高效语义迁移，显著提升检测精度且不增加推理开销，推动了实时检测在设备端的应用。**

- **链接: [http://arxiv.org/pdf/2510.25257v1](http://arxiv.org/pdf/2510.25257v1)**

> **作者:** Zijun Liao; Yian Zhao; Xin Shan; Yu Yan; Chang Liu; Lei Lu; Xiangyang Ji; Jie Chen
>
> **摘要:** Real-time object detection has achieved substantial progress through meticulously designed architectures and optimization strategies. However, the pursuit of high-speed inference via lightweight network designs often leads to degraded feature representation, which hinders further performance improvements and practical on-device deployment. In this paper, we propose a cost-effective and highly adaptable distillation framework that harnesses the rapidly evolving capabilities of Vision Foundation Models (VFMs) to enhance lightweight object detectors. Given the significant architectural and learning objective disparities between VFMs and resource-constrained detectors, achieving stable and task-aligned semantic transfer is challenging. To address this, on one hand, we introduce a Deep Semantic Injector (DSI) module that facilitates the integration of high-level representations from VFMs into the deep layers of the detector. On the other hand, we devise a Gradient-guided Adaptive Modulation (GAM) strategy, which dynamically adjusts the intensity of semantic transfer based on gradient norm ratios. Without increasing deployment and inference overhead, our approach painlessly delivers striking and consistent performance gains across diverse DETR-based models, underscoring its practical utility for real-time detection. Our new model family, RT-DETRv4, achieves state-of-the-art results on COCO, attaining AP scores of 49.7/53.5/55.4/57.0 at corresponding speeds of 273/169/124/78 FPS.
>
---
#### [new 032] Mask-Robust Face Verification for Online Learning via YOLOv5 and Residual Networks
- **分类: cs.CV**

- **简介: 该论文针对在线学习中身份认证难题，提出基于YOLOv5与残差网络的鲁棒人脸验证方法。通过YOLOv5检测人脸，利用残差网络提取深层特征，结合欧氏距离比对实现身份识别，提升在线教育的安全性与稳定性。**

- **链接: [http://arxiv.org/pdf/2510.25184v1](http://arxiv.org/pdf/2510.25184v1)**

> **作者:** Zhifeng Wang; Minghui Wang; Chunyan Zeng; Jialong Yao; Yang Yang; Hongmin Xu
>
> **备注:** 9 pages, 10 figures
>
> **摘要:** In the contemporary landscape, the fusion of information technology and the rapid advancement of artificial intelligence have ushered school education into a transformative phase characterized by digitization and heightened intelligence. Concurrently, the global paradigm shift caused by the Covid-19 pandemic has catalyzed the evolution of e-learning, accentuating its significance. Amidst these developments, one pivotal facet of the online education paradigm that warrants attention is the authentication of identities within the digital learning sphere. Within this context, our study delves into a solution for online learning authentication, utilizing an enhanced convolutional neural network architecture, specifically the residual network model. By harnessing the power of deep learning, this technological approach aims to galvanize the ongoing progress of online education, while concurrently bolstering its security and stability. Such fortification is imperative in enabling online education to seamlessly align with the swift evolution of the educational landscape. This paper's focal proposition involves the deployment of the YOLOv5 network, meticulously trained on our proprietary dataset. This network is tasked with identifying individuals' faces culled from images captured by students' open online cameras. The resultant facial information is then channeled into the residual network to extract intricate features at a deeper level. Subsequently, a comparative analysis of Euclidean distances against students' face databases is performed, effectively ascertaining the identity of each student.
>
---
#### [new 033] PSTF-AttControl: Per-Subject-Tuning-Free Personalized Image Generation with Controllable Face Attributes
- **分类: cs.CV**

- **简介: 该论文提出PSTF-AttControl，解决个性化图像生成中面部属性控制与身份保真度的平衡问题。无需微调，通过身份特征映射与跨注意力机制，实现单图输入下精确控制面部属性，提升生成质量与可控性。**

- **链接: [http://arxiv.org/pdf/2510.25084v1](http://arxiv.org/pdf/2510.25084v1)**

> **作者:** Xiang liu; Zhaoxiang Liu; Huan Hu; Zipeng Wang; Ping Chen; Zezhou Chen; Kai Wang; Shiguo Lian
>
> **备注:** Accepted by Image and Vision Computing (18 pages, 8 figures)
>
> **摘要:** Recent advancements in personalized image generation have significantly improved facial identity preservation, particularly in fields such as entertainment and social media. However, existing methods still struggle to achieve precise control over facial attributes in a per-subject-tuning-free (PSTF) way. Tuning-based techniques like PreciseControl have shown promise by providing fine-grained control over facial features, but they often require extensive technical expertise and additional training data, limiting their accessibility. In contrast, PSTF approaches simplify the process by enabling image generation from a single facial input, but they lack precise control over facial attributes. In this paper, we introduce a novel, PSTF method that enables both precise control over facial attributes and high-fidelity preservation of facial identity. Our approach utilizes a face recognition model to extract facial identity features, which are then mapped into the $W^+$ latent space of StyleGAN2 using the e4e encoder. We further enhance the model with a Triplet-Decoupled Cross-Attention module, which integrates facial identity, attribute features, and text embeddings into the UNet architecture, ensuring clean separation of identity and attribute information. Trained on the FFHQ dataset, our method allows for the generation of personalized images with fine-grained control over facial attributes, while without requiring additional fine-tuning or training data for individual identities. We demonstrate that our approach successfully balances personalization with precise facial attribute control, offering a more efficient and user-friendly solution for high-quality, adaptable facial image synthesis. The code is publicly available at https://github.com/UnicomAI/PSTF-AttControl.
>
---
#### [new 034] Prompt Estimation from Prototypes for Federated Prompt Tuning of Vision Transformers
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对联邦学习中视觉Transformer的提示调优问题，提出PEP-FedPT框架。针对全局调优泛化差与个性化调优过拟合的矛盾，设计类上下文混合提示（CCMP），通过全局类原型与客户端先验自适应融合，实现无参数存储的样本级个性化，提升跨异构客户端的泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.25372v1](http://arxiv.org/pdf/2510.25372v1)**

> **作者:** M Yashwanth; Sharannya Ghosh; Aditay Tripathi; Anirban Chakraborty
>
> **摘要:** Visual Prompt Tuning (VPT) of pre-trained Vision Transformers (ViTs) has proven highly effective as a parameter-efficient fine-tuning technique for adapting large models to downstream tasks with limited data. Its parameter efficiency makes it particularly suitable for Federated Learning (FL), where both communication and computation budgets are often constrained. However, global prompt tuning struggles to generalize across heterogeneous clients, while personalized tuning overfits to local data and lacks generalization. We propose PEP-FedPT (Prompt Estimation from Prototypes for Federated Prompt Tuning), a unified framework designed to achieve both generalization and personalization in federated prompt tuning of ViTs. Within this framework, we introduce the novel Class-Contextualized Mixed Prompt (CCMP) - based on class-specific prompts maintained alongside a globally shared prompt. For each input, CCMP adaptively combines class-specific prompts using weights derived from global class prototypes and client class priors. This approach enables per-sample prompt personalization without storing client-dependent trainable parameters. The prompts are collaboratively optimized via traditional federated averaging technique on the same. Comprehensive evaluations on CIFAR-100, TinyImageNet, DomainNet, and iNaturalist datasets demonstrate that PEP-FedPT consistently surpasses the state-of-the-art baselines under diverse data heterogeneity scenarios, establishing a strong foundation for efficient and generalizable federated prompt tuning of Vision Transformers.
>
---
#### [new 035] SafeEditor: Unified MLLM for Efficient Post-hoc T2I Safety Editing
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文本生成图像（T2I）模型的安全性问题，提出SafeEditor框架。通过构建多轮图像-文本交错数据集MR-SafeEdit，实现无需重新训练的后置安全编辑，有效平衡安全与实用性，缓解过度拒绝问题。**

- **链接: [http://arxiv.org/pdf/2510.24820v1](http://arxiv.org/pdf/2510.24820v1)**

> **作者:** Ruiyang Zhang; Jiahao Luo; Xiaoru Feng; Qiufan Pang; Yaodong Yang; Juntao Dai
>
> **摘要:** With the rapid advancement of text-to-image (T2I) models, ensuring their safety has become increasingly critical. Existing safety approaches can be categorized into training-time and inference-time methods. While inference-time methods are widely adopted due to their cost-effectiveness, they often suffer from limitations such as over-refusal and imbalance between safety and utility. To address these challenges, we propose a multi-round safety editing framework that functions as a model-agnostic, plug-and-play module, enabling efficient safety alignment for any text-to-image model. Central to this framework is MR-SafeEdit, a multi-round image-text interleaved dataset specifically constructed for safety editing in text-to-image generation. We introduce a post-hoc safety editing paradigm that mirrors the human cognitive process of identifying and refining unsafe content. To instantiate this paradigm, we develop SafeEditor, a unified MLLM capable of multi-round safety editing on generated images. Experimental results show that SafeEditor surpasses prior safety approaches by reducing over-refusal while achieving a more favorable safety-utility balance.
>
---
#### [new 036] MSF-Net: Multi-Stage Feature Extraction and Fusion for Robust Photometric Stereo
- **分类: cs.CV**

- **简介: 该论文针对光度立体任务中特征提取冗余、多阶段交互不足的问题，提出MSF-Net框架，通过多阶段特征提取与选择性更新机制，增强细节区域特征表达，并设计特征融合模块促进特征间交互，显著提升表面法向估计精度。**

- **链接: [http://arxiv.org/pdf/2510.25221v1](http://arxiv.org/pdf/2510.25221v1)**

> **作者:** Shiyu Qin; Zhihao Cai; Kaixuan Wang; Lin Qi; Junyu Dong
>
> **摘要:** Photometric stereo is a technique aimed at determining surface normals through the utilization of shading cues derived from images taken under different lighting conditions. However, existing learning-based approaches often fail to accurately capture features at multiple stages and do not adequately promote interaction between these features. Consequently, these models tend to extract redundant features, especially in areas with intricate details such as wrinkles and edges. To tackle these issues, we propose MSF-Net, a novel framework for extracting information at multiple stages, paired with selective update strategy, aiming to extract high-quality feature information, which is critical for accurate normal construction. Additionally, we have developed a feature fusion module to improve the interplay among different features. Experimental results on the DiLiGenT benchmark show that our proposed MSF-Net significantly surpasses previous state-of-the-art methods in the accuracy of surface normal estimation.
>
---
#### [new 037] EA3D: Online Open-World 3D Object Extraction from Streaming Videos
- **分类: cs.CV**

- **简介: 该论文提出EA3D，一种在线开放世界3D物体提取框架，解决传统方法依赖离线数据或预构建几何的问题。通过融合视觉-语言模型与在线高斯特征更新，实现流式视频下的实时几何重建与语义理解，支持多任务下游应用。**

- **链接: [http://arxiv.org/pdf/2510.25146v1](http://arxiv.org/pdf/2510.25146v1)**

> **作者:** Xiaoyu Zhou; Jingqi Wang; Yuang Jia; Yongtao Wang; Deqing Sun; Ming-Hsuan Yang
>
> **备注:** The Thirty-Ninth Annual Conference on Neural Information Processing Systems(NeurIPS 2025)
>
> **摘要:** Current 3D scene understanding methods are limited by offline-collected multi-view data or pre-constructed 3D geometry. In this paper, we present ExtractAnything3D (EA3D), a unified online framework for open-world 3D object extraction that enables simultaneous geometric reconstruction and holistic scene understanding. Given a streaming video, EA3D dynamically interprets each frame using vision-language and 2D vision foundation encoders to extract object-level knowledge. This knowledge is integrated and embedded into a Gaussian feature map via a feed-forward online update strategy. We then iteratively estimate visual odometry from historical frames and incrementally update online Gaussian features with new observations. A recurrent joint optimization module directs the model's attention to regions of interest, simultaneously enhancing both geometric reconstruction and semantic understanding. Extensive experiments across diverse benchmarks and tasks, including photo-realistic rendering, semantic and instance segmentation, 3D bounding box and semantic occupancy estimation, and 3D mesh generation, demonstrate the effectiveness of EA3D. Our method establishes a unified and efficient framework for joint online 3D reconstruction and holistic scene understanding, enabling a broad range of downstream tasks.
>
---
#### [new 038] MCIHN: A Hybrid Network Model Based on Multi-path Cross-modal Interaction for Multimodal Emotion Recognition
- **分类: cs.CV; cs.MM**

- **简介: 该论文针对多模态情感识别任务，解决模态间差异大及单模态情感表征难的问题。提出基于多路径跨模态交互的混合网络模型（MCIHN），结合对抗自编码器提取判别特征，通过跨模态门控机制对齐模态并生成交互特征，再经特征融合模块提升识别性能，在SIMS和MOSI数据集上表现优异。**

- **链接: [http://arxiv.org/pdf/2510.24827v1](http://arxiv.org/pdf/2510.24827v1)**

> **作者:** Haoyang Zhang; Zhou Yang; Ke Sun; Yucai Pang; Guoliang Xu
>
> **备注:** The paper will be published in the MMAsia2025 conference proceedings
>
> **摘要:** Multimodal emotion recognition is crucial for future human-computer interaction. However, accurate emotion recognition still faces significant challenges due to differences between different modalities and the difficulty of characterizing unimodal emotional information. To solve these problems, a hybrid network model based on multipath cross-modal interaction (MCIHN) is proposed. First, adversarial autoencoders (AAE) are constructed separately for each modality. The AAE learns discriminative emotion features and reconstructs the features through a decoder to obtain more discriminative information about the emotion classes. Then, the latent codes from the AAE of different modalities are fed into a predefined Cross-modal Gate Mechanism model (CGMM) to reduce the discrepancy between modalities, establish the emotional relationship between interacting modalities, and generate the interaction features between different modalities. Multimodal fusion using the Feature Fusion module (FFM) for better emotion recognition. Experiments were conducted on publicly available SIMS and MOSI datasets, demonstrating that MCIHN achieves superior performance.
>
---
#### [new 039] Understanding Multi-View Transformers
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究多视图变换器在3D视觉中的机制，针对其黑箱特性导致的可解释性差问题，提出通过残差连接可视化分析模型内部表示。工作聚焦于DUSt3R变体，揭示了隐状态演化、各层作用及与具强归纳偏置方法的区别，表明其对应匹配依赖于重构几何。**

- **链接: [http://arxiv.org/pdf/2510.24907v1](http://arxiv.org/pdf/2510.24907v1)**

> **作者:** Michal Stary; Julien Gaubil; Ayush Tewari; Vincent Sitzmann
>
> **备注:** Presented at the ICCV 2025 E2E3D Workshop
>
> **摘要:** Multi-view transformers such as DUSt3R are revolutionizing 3D vision by solving 3D tasks in a feed-forward manner. However, contrary to previous optimization-based pipelines, the inner mechanisms of multi-view transformers are unclear. Their black-box nature makes further improvements beyond data scaling challenging and complicates usage in safety- and reliability-critical applications. Here, we present an approach for probing and visualizing 3D representations from the residual connections of the multi-view transformers' layers. In this manner, we investigate a variant of the DUSt3R model, shedding light on the development of its latent state across blocks, the role of the individual layers, and suggest how it differs from methods with stronger inductive biases of explicit global pose. Finally, we show that the investigated variant of DUSt3R estimates correspondences that are refined with reconstructed geometry. The code used for the analysis is available at https://github.com/JulienGaubil/und3rstand .
>
---
#### [new 040] A Re-node Self-training Approach for Deep Graph-based Semi-supervised Classification on Multi-view Image Data
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对多视图图像数据的半监督分类任务，解决传统图方法在图像中缺乏清晰结构及多视图融合困难的问题。提出RSGSLM方法，通过融合多视图特征、动态引入伪标签、调整边界样本权重及引入无监督平滑损失，提升分类性能与效率。**

- **链接: [http://arxiv.org/pdf/2510.24791v1](http://arxiv.org/pdf/2510.24791v1)**

> **作者:** Jingjun Bi; Fadi Dornaika
>
> **摘要:** Recently, graph-based semi-supervised learning and pseudo-labeling have gained attention due to their effectiveness in reducing the need for extensive data annotations. Pseudo-labeling uses predictions from unlabeled data to improve model training, while graph-based methods are characterized by processing data represented as graphs. However, the lack of clear graph structures in images combined with the complexity of multi-view data limits the efficiency of traditional and existing techniques. Moreover, the integration of graph structures in multi-view data is still a challenge. In this paper, we propose Re-node Self-taught Graph-based Semi-supervised Learning for Multi-view Data (RSGSLM). Our method addresses these challenges by (i) combining linear feature transformation and multi-view graph fusion within a Graph Convolutional Network (GCN) framework, (ii) dynamically incorporating pseudo-labels into the GCN loss function to improve classification in multi-view data, and (iii) correcting topological imbalances by adjusting the weights of labeled samples near class boundaries. Additionally, (iv) we introduce an unsupervised smoothing loss applicable to all samples. This combination optimizes performance while maintaining computational efficiency. Experimental results on multi-view benchmark image datasets demonstrate that RSGSLM surpasses existing semi-supervised learning approaches in multi-view contexts.
>
---
#### [new 041] LangHOPS: Language Grounded Hierarchical Open-Vocabulary Part Segmentation
- **分类: cs.CV**

- **简介: 该论文提出LangHOPS，首个基于多模态大模型的开放词汇物体部件实例分割框架。针对传统方法依赖视觉分组、难以处理开放词汇的问题，通过语言空间构建层级部件结构，利用MLLM实现跨粒度语义理解与部件查询优化，在多个数据集上达到领先性能。**

- **链接: [http://arxiv.org/pdf/2510.25263v1](http://arxiv.org/pdf/2510.25263v1)**

> **作者:** Yang Miao; Jan-Nico Zaech; Xi Wang; Fabien Despinoy; Danda Pani Paudel; Luc Van Gool
>
> **备注:** 10 pages, 5 figures, 14 tables, Neurips 2025
>
> **摘要:** We propose LangHOPS, the first Multimodal Large Language Model (MLLM) based framework for open-vocabulary object-part instance segmentation. Given an image, LangHOPS can jointly detect and segment hierarchical object and part instances from open-vocabulary candidate categories. Unlike prior approaches that rely on heuristic or learnable visual grouping, our approach grounds object-part hierarchies in language space. It integrates the MLLM into the object-part parsing pipeline to leverage its rich knowledge and reasoning capabilities, and link multi-granularity concepts within the hierarchies. We evaluate LangHOPS across multiple challenging scenarios, including in-domain and cross-dataset object-part instance segmentation, and zero-shot semantic segmentation. LangHOPS achieves state-of-the-art results, surpassing previous methods by 5.5% Average Precision (AP) (in-domain) and 4.8% (cross-dataset) on the PartImageNet dataset and by 2.5% mIOU on unseen object parts in ADE20K (zero-shot). Ablation studies further validate the effectiveness of the language-grounded hierarchy and MLLM driven part query refinement strategy. The code will be released here.
>
---
#### [new 042] GaTector+: A Unified Head-free Framework for Gaze Object and Gaze Following Prediction
- **分类: cs.CV**

- **简介: 该论文提出GaTector+，一个统一的无头先验框架，用于联合解决眼动目标检测与眼动跟随任务。针对现有方法依赖头部位置信息的问题，模型通过共享骨干网络、头检测分支及基于头部的注意力机制，实现端到端联合优化，并引入注意力监督与新评估指标mSoC，提升性能与实用性。**

- **链接: [http://arxiv.org/pdf/2510.25301v1](http://arxiv.org/pdf/2510.25301v1)**

> **作者:** Yang Jin; Guangyu Guo; Binglu Wang
>
> **摘要:** Gaze object detection and gaze following are fundamental tasks for interpreting human gaze behavior or intent. However, most previous methods usually solve these two tasks separately, and their prediction of gaze objects and gaze following typically depend on head-related prior knowledge during both the training phase and real-world deployment. This dependency necessitates an auxiliary network to extract head location, thus precluding joint optimization across the entire system and constraining the practical applicability. To this end, we propose GaTector+, a unified framework for gaze object detection and gaze following, which eliminates the dependence on the head-related priors during inference. Specifically, GaTector+ uses an expanded specific-general-specific feature extractor that leverages a shared backbone, which extracts general features for gaze following and object detection using the shared backbone while using specific blocks before and after the shared backbone to better consider the specificity of each sub-task. To obtain head-related knowledge without prior information, we first embed a head detection branch to predict the head of each person. Then, before regressing the gaze point, a head-based attention mechanism is proposed to fuse the sense feature and gaze feature with the help of head location. Since the suboptimization of the gaze point heatmap leads to the performance bottleneck, we propose an attention supervision mechanism to accelerate the learning of the gaze heatmap. Finally, we propose a novel evaluation metric, mean Similarity over Candidates (mSoC), for gaze object detection, which is more sensitive to variations between bounding boxes. The experimental results on multiple benchmark datasets demonstrate the effectiveness of our model in both gaze object detection and gaze following tasks.
>
---
#### [new 043] Revisiting Reconstruction-based AI-generated Image Detection: A Geometric Perspective
- **分类: cs.CV**

- **简介: 该论文聚焦于AI生成图像检测任务，针对现有重建方法缺乏理论基础、依赖经验阈值的问题，提出从几何视角分析重建误差。通过引入雅可比谱下界揭示真实与生成图像的误差差异，并设计无需训练的ReGap方法，利用结构化编辑动态测量误差变化，提升检测准确性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.25141v1](http://arxiv.org/pdf/2510.25141v1)**

> **作者:** Wan Jiang; Jing Yan; Ruixuan Zhang; Xiaojing Chen; Changtao Miao; Zhe Li; Chenhao Lin; Yunfeng Diao; Richang Hong
>
> **摘要:** The rise of generative Artificial Intelligence (AI) has made detecting AI-generated images a critical challenge for ensuring authenticity. Existing reconstruction-based methods lack theoretical foundations and on empirical heuristics, limiting interpretability and reliability. In this paper, we introduce the Jacobian-Spectral Lower Bound for reconstruction error from a geometric perspective, showing that real images off the reconstruction manifold exhibit a non-trivial error lower bound, while generated images on the manifold have near-zero error. Furthermore, we reveal the limitations of existing methods that rely on static reconstruction error from a single pass. These methods often fail when some real images exhibit lower error than generated ones. This counterintuitive behavior reduces detection accuracy and requires data-specific threshold tuning, limiting their applicability in real-world scenarios. To address these challenges, we propose ReGap, a training-free method that computes dynamic reconstruction error by leveraging structured editing operations to introduce controlled perturbations. This enables measuring error changes before and after editing, improving detection accuracy by enhancing error separation. Experimental results show that our method outperforms existing baselines, exhibits robustness to common post-processing operations and generalizes effectively across diverse conditions.
>
---
#### [new 044] VADB: A Large-Scale Video Aesthetic Database with Professional and Multi-Dimensional Annotations
- **分类: cs.CV**

- **简介: 该论文针对视频美学评估任务，解决缺乏大规模、多维度标注数据及有效模型的问题。提出VADB数据库（10,490视频，37位专家多维标注）和VADB-Net双模态预训练框架，支持精准评分与下游任务，推动视频美学评估发展。**

- **链接: [http://arxiv.org/pdf/2510.25238v1](http://arxiv.org/pdf/2510.25238v1)**

> **作者:** Qianqian Qiao; DanDan Zheng; Yihang Bo; Bao Peng; Heng Huang; Longteng Jiang; Huaye Wang; Jingdong Chen; Jun Zhou; Xin Jin
>
> **摘要:** Video aesthetic assessment, a vital area in multimedia computing, integrates computer vision with human cognition. Its progress is limited by the lack of standardized datasets and robust models, as the temporal dynamics of video and multimodal fusion challenges hinder direct application of image-based methods. This study introduces VADB, the largest video aesthetic database with 10,490 diverse videos annotated by 37 professionals across multiple aesthetic dimensions, including overall and attribute-specific aesthetic scores, rich language comments and objective tags. We propose VADB-Net, a dual-modal pre-training framework with a two-stage training strategy, which outperforms existing video quality assessment models in scoring tasks and supports downstream video aesthetic assessment tasks. The dataset and source code are available at https://github.com/BestiVictory/VADB.
>
---
#### [new 045] More than a Moment: Towards Coherent Sequences of Audio Descriptions
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对视频音频描述任务，解决自动生成描述序列缺乏连贯性的问题。提出无需训练的CoherentAD方法，通过多候选生成与自回归选择，提升描述序列的连贯性与叙事完整性，并引入StoryRecall等评估指标，显著优于独立生成的方法。**

- **链接: [http://arxiv.org/pdf/2510.25440v1](http://arxiv.org/pdf/2510.25440v1)**

> **作者:** Eshika Khandelwal; Junyu Xie; Tengda Han; Max Bain; Arsha Nagrani; Andrew Zisserman; Gül Varol; Makarand Tapaswi
>
> **摘要:** Audio Descriptions (ADs) convey essential on-screen information, allowing visually impaired audiences to follow videos. To be effective, ADs must form a coherent sequence that helps listeners to visualise the unfolding scene, rather than describing isolated moments. However, most automatic methods generate each AD independently, often resulting in repetitive, incoherent descriptions. To address this, we propose a training-free method, CoherentAD, that first generates multiple candidate descriptions for each AD time interval, and then performs auto-regressive selection across the sequence to form a coherent and informative narrative. To evaluate AD sequences holistically, we introduce a sequence-level metric, StoryRecall, which measures how well the predicted ADs convey the ground truth narrative, alongside repetition metrics that capture the redundancy across consecutive AD outputs. Our method produces coherent AD sequences with enhanced narrative understanding, outperforming prior approaches that rely on independent generations.
>
---
#### [new 046] Region-CAM: Towards Accurate Object Regions in Class Activation Maps for Weakly Supervised Learning Tasks
- **分类: cs.CV**

- **简介: 该论文针对弱监督学习中的类激活图（CAM）方法存在的对象区域覆盖不全、边界对齐差问题，提出Region-CAM。通过提取语义信息图并进行传播，增强对象区域完整性与边界精度，在WSSS和目标定位任务中显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.25134v1](http://arxiv.org/pdf/2510.25134v1)**

> **作者:** Qingdong Cai; Charith Abhayaratne
>
> **备注:** Preprint for journal paper
>
> **摘要:** Class Activation Mapping (CAM) methods are widely applied in weakly supervised learning tasks due to their ability to highlight object regions. However, conventional CAM methods highlight only the most discriminative regions of the target. These highlighted regions often fail to cover the entire object and are frequently misaligned with object boundaries, thereby limiting the performance of downstream weakly supervised learning tasks, particularly Weakly Supervised Semantic Segmentation (WSSS), which demands pixel-wise accurate activation maps to get the best results. To alleviate the above problems, we propose a novel activation method, Region-CAM. Distinct from network feature weighting approaches, Region-CAM generates activation maps by extracting semantic information maps (SIMs) and performing semantic information propagation (SIP) by considering both gradients and features in each of the stages of the baseline classification model. Our approach highlights a greater proportion of object regions while ensuring activation maps to have precise boundaries that align closely with object edges. Region-CAM achieves 60.12% and 58.43% mean intersection over union (mIoU) using the baseline model on the PASCAL VOC training and validation datasets, respectively, which are improvements of 13.61% and 13.13% over the original CAM (46.51% and 45.30%). On the MS COCO validation set, Region-CAM achieves 36.38%, a 16.23% improvement over the original CAM (20.15%). We also demonstrate the superiority of Region-CAM in object localization tasks, using the ILSVRC2012 validation set. Region-CAM achieves 51.7% in Top-1 Localization accuracy Loc1. Compared with LayerCAM, an activation method designed for weakly supervised object localization, Region-CAM achieves 4.5% better performance in Loc1.
>
---
#### [new 047] DINO-YOLO: Self-Supervised Pre-training for Data-Efficient Object Detection in Civil Engineering Applications
- **分类: cs.CV**

- **简介: 该论文针对土木工程中因标注数据少导致的物体检测困难问题，提出DINO-YOLO模型。通过融合YOLOv12与DINOv3自监督视觉变压器，在输入预处理和骨干网络中间层引入特征，实现数据高效检测。在小样本场景下显著提升精度，同时保持实时推理速度，适用于施工安全与基础设施巡检。**

- **链接: [http://arxiv.org/pdf/2510.25140v1](http://arxiv.org/pdf/2510.25140v1)**

> **作者:** Malaisree P; Youwai S; Kitkobsin T; Janrungautai S; Amorndechaphon D; Rojanavasu P
>
> **摘要:** Object detection in civil engineering applications is constrained by limited annotated data in specialized domains. We introduce DINO-YOLO, a hybrid architecture combining YOLOv12 with DINOv3 self-supervised vision transformers for data-efficient detection. DINOv3 features are strategically integrated at two locations: input preprocessing (P0) and mid-backbone enhancement (P3). Experimental validation demonstrates substantial improvements: Tunnel Segment Crack detection (648 images) achieves 12.4% improvement, Construction PPE (1K images) gains 13.7%, and KITTI (7K images) shows 88.6% improvement, while maintaining real-time inference (30-47 FPS). Systematic ablation across five YOLO scales and nine DINOv3 variants reveals that Medium-scale architectures achieve optimal performance with DualP0P3 integration (55.77% mAP@0.5), while Small-scale requires Triple Integration (53.63%). The 2-4x inference overhead (21-33ms versus 8-16ms baseline) remains acceptable for field deployment on NVIDIA RTX 5090. DINO-YOLO establishes state-of-the-art performance for civil engineering datasets (<10K images) while preserving computational efficiency, providing practical solutions for construction safety monitoring and infrastructure inspection in data-constrained environments.
>
---
#### [new 048] Multimodal Spatial Reasoning in the Large Model Era: A Survey and Benchmarks
- **分类: cs.CV**

- **简介: 该论文聚焦于大模型时代的多模态空间推理任务，旨在系统梳理相关技术进展与评估基准。针对现有研究缺乏系统综述和公开评测平台的问题，论文分类总结了多模态大模型在空间理解、3D场景推理、具身智能等方面的工作，并构建了开源基准，推动该领域发展。**

- **链接: [http://arxiv.org/pdf/2510.25760v1](http://arxiv.org/pdf/2510.25760v1)**

> **作者:** Xu Zheng; Zihao Dongfang; Lutao Jiang; Boyuan Zheng; Yulong Guo; Zhenquan Zhang; Giuliano Albanese; Runyi Yang; Mengjiao Ma; Zixin Zhang; Chenfei Liao; Dingcheng Zhen; Yuanhuiyi Lyu; Yuqian Fu; Bin Ren; Linfeng Zhang; Danda Pani Paudel; Nicu Sebe; Luc Van Gool; Xuming Hu
>
> **摘要:** Humans possess spatial reasoning abilities that enable them to understand spaces through multimodal observations, such as vision and sound. Large multimodal reasoning models extend these abilities by learning to perceive and reason, showing promising performance across diverse spatial tasks. However, systematic reviews and publicly available benchmarks for these models remain limited. In this survey, we provide a comprehensive review of multimodal spatial reasoning tasks with large models, categorizing recent progress in multimodal large language models (MLLMs) and introducing open benchmarks for evaluation. We begin by outlining general spatial reasoning, focusing on post-training techniques, explainability, and architecture. Beyond classical 2D tasks, we examine spatial relationship reasoning, scene and layout understanding, as well as visual question answering and grounding in 3D space. We also review advances in embodied AI, including vision-language navigation and action models. Additionally, we consider emerging modalities such as audio and egocentric video, which contribute to novel spatial understanding through new sensors. We believe this survey establishes a solid foundation and offers insights into the growing field of multimodal spatial reasoning. Updated information about this survey, codes and implementation of the open benchmarks can be found at https://github.com/zhengxuJosh/Awesome-Spatial-Reasoning.
>
---
#### [new 049] SPADE: Sparsity Adaptive Depth Estimator for Zero-Shot, Real-Time, Monocular Depth Estimation in Underwater Environments
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出SPADE，一种用于水下环境的零样本、实时单目深度估计方法。针对水下复杂场景中感知受限问题，结合稀疏深度先验与相对深度估计，通过两阶段优化生成稠密度量深度图，提升精度与泛化性，并在嵌入式设备上实现15 FPS以上运行，助力水下自主巡检。**

- **链接: [http://arxiv.org/pdf/2510.25463v1](http://arxiv.org/pdf/2510.25463v1)**

> **作者:** Hongjie Zhang; Gideon Billings; Stefan B. Williams
>
> **摘要:** Underwater infrastructure requires frequent inspection and maintenance due to harsh marine conditions. Current reliance on human divers or remotely operated vehicles is limited by perceptual and operational challenges, especially around complex structures or in turbid water. Enhancing the spatial awareness of underwater vehicles is key to reducing piloting risks and enabling greater autonomy. To address these challenges, we present SPADE: SParsity Adaptive Depth Estimator, a monocular depth estimation pipeline that combines pre-trained relative depth estimator with sparse depth priors to produce dense, metric scale depth maps. Our two-stage approach first scales the relative depth map with the sparse depth points, then refines the final metric prediction with our proposed Cascade Conv-Deformable Transformer blocks. Our approach achieves improved accuracy and generalisation over state-of-the-art baselines and runs efficiently at over 15 FPS on embedded hardware, promising to support practical underwater inspection and intervention. This work has been submitted to IEEE Journal of Oceanic Engineering Special Issue of AUV 2026.
>
---
#### [new 050] VFXMaster: Unlocking Dynamic Visual Effect Generation via In-Context Learning
- **分类: cs.CV**

- **简介: 该论文提出VFXMaster，解决生成式AI在动态视觉效果（VFX）创作中难以泛化的问题。通过引入上下文学习机制，将效果生成建模为基于参考视频的条件任务，实现单一模型对多种未见效果的精准模仿与快速适应，显著提升泛化能力与效率。**

- **链接: [http://arxiv.org/pdf/2510.25772v1](http://arxiv.org/pdf/2510.25772v1)**

> **作者:** Baolu Li; Yiming Zhang; Qinghe Wang; Liqian Ma; Xiaoyu Shi; Xintao Wang; Pengfei Wan; Zhenfei Yin; Yunzhi Zhuge; Huchuan Lu; Xu Jia
>
> **备注:** Project Page URL:https://libaolu312.github.io/VFXMaster/
>
> **摘要:** Visual effects (VFX) are crucial to the expressive power of digital media, yet their creation remains a major challenge for generative AI. Prevailing methods often rely on the one-LoRA-per-effect paradigm, which is resource-intensive and fundamentally incapable of generalizing to unseen effects, thus limiting scalability and creation. To address this challenge, we introduce VFXMaster, the first unified, reference-based framework for VFX video generation. It recasts effect generation as an in-context learning task, enabling it to reproduce diverse dynamic effects from a reference video onto target content. In addition, it demonstrates remarkable generalization to unseen effect categories. Specifically, we design an in-context conditioning strategy that prompts the model with a reference example. An in-context attention mask is designed to precisely decouple and inject the essential effect attributes, allowing a single unified model to master the effect imitation without information leakage. In addition, we propose an efficient one-shot effect adaptation mechanism to boost generalization capability on tough unseen effects from a single user-provided video rapidly. Extensive experiments demonstrate that our method effectively imitates various categories of effect information and exhibits outstanding generalization to out-of-domain effects. To foster future research, we will release our code, models, and a comprehensive dataset to the community.
>
---
#### [new 051] DrivingScene: A Multi-Task Online Feed-Forward 3D Gaussian Splatting Method for Dynamic Driving Scenes
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出DrivingScene，面向动态驾驶场景的实时高保真重建任务。针对复杂动态与稀疏视角下的重建难题，提出基于双帧图像的在线前馈框架，通过轻量级残差光流网络与静态场景先验结合，显式建模非刚性运动，实现高质量深度、场景流与3D高斯点云的在线生成。**

- **链接: [http://arxiv.org/pdf/2510.24734v1](http://arxiv.org/pdf/2510.24734v1)**

> **作者:** Qirui Hou; Wenzhang Sun; Chang Zeng; Chunfeng Wang; Hao Li; Jianxun Cui
>
> **备注:** Autonomous Driving, Novel view Synthesis, Multi task Learning
>
> **摘要:** Real-time, high-fidelity reconstruction of dynamic driving scenes is challenged by complex dynamics and sparse views, with prior methods struggling to balance quality and efficiency. We propose DrivingScene, an online, feed-forward framework that reconstructs 4D dynamic scenes from only two consecutive surround-view images. Our key innovation is a lightweight residual flow network that predicts the non-rigid motion of dynamic objects per camera on top of a learned static scene prior, explicitly modeling dynamics via scene flow. We also introduce a coarse-to-fine training paradigm that circumvents the instabilities common to end-to-end approaches. Experiments on nuScenes dataset show our image-only method simultaneously generates high-quality depth, scene flow, and 3D Gaussian point clouds online, significantly outperforming state-of-the-art methods in both dynamic reconstruction and novel view synthesis.
>
---
#### [new 052] The Generation Phases of Flow Matching: a Denoising Perspective
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究生成模型中的流匹配（Flow Matching）技术，从去噪视角揭示其生成过程的动态阶段。通过建立与去噪器的理论联系，提出噪声与漂移扰动机制，揭示不同阶段去噪效果差异，深化对生成质量影响因素的理解。**

- **链接: [http://arxiv.org/pdf/2510.24830v1](http://arxiv.org/pdf/2510.24830v1)**

> **作者:** Anne Gagneux; Ségolène Martin; Rémi Gribonval; Mathurin Massias
>
> **摘要:** Flow matching has achieved remarkable success, yet the factors influencing the quality of its generation process remain poorly understood. In this work, we adopt a denoising perspective and design a framework to empirically probe the generation process. Laying down the formal connections between flow matching models and denoisers, we provide a common ground to compare their performances on generation and denoising. This enables the design of principled and controlled perturbations to influence sample generation: noise and drift. This leads to new insights on the distinct dynamical phases of the generative process, enabling us to precisely characterize at which stage of the generative process denoisers succeed or fail and why this matters.
>
---
#### [new 053] RegionE: Adaptive Region-Aware Generation for Efficient Image Editing
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对指令图像编辑（IIE）中全图统一生成效率低的问题，提出RegionE框架。通过自适应区域划分，对未编辑区域采用单步预测，编辑区域则结合区域指令缓存与速度衰减缓存，实现高效且保真的局部迭代生成，显著提升速度并保持质量。**

- **链接: [http://arxiv.org/pdf/2510.25590v1](http://arxiv.org/pdf/2510.25590v1)**

> **作者:** Pengtao Chen; Xianfang Zeng; Maosen Zhao; Mingzhu Shen; Peng Ye; Bangyin Xiang; Zhibo Wang; Wei Cheng; Gang Yu; Tao Chen
>
> **备注:** 26 pages, 10 figures, 18 tables
>
> **摘要:** Recently, instruction-based image editing (IIE) has received widespread attention. In practice, IIE often modifies only specific regions of an image, while the remaining areas largely remain unchanged. Although these two types of regions differ significantly in generation difficulty and computational redundancy, existing IIE models do not account for this distinction, instead applying a uniform generation process across the entire image. This motivates us to propose RegionE, an adaptive, region-aware generation framework that accelerates IIE tasks without additional training. Specifically, the RegionE framework consists of three main components: 1) Adaptive Region Partition. We observed that the trajectory of unedited regions is straight, allowing for multi-step denoised predictions to be inferred in a single step. Therefore, in the early denoising stages, we partition the image into edited and unedited regions based on the difference between the final estimated result and the reference image. 2) Region-Aware Generation. After distinguishing the regions, we replace multi-step denoising with one-step prediction for unedited areas. For edited regions, the trajectory is curved, requiring local iterative denoising. To improve the efficiency and quality of local iterative generation, we propose the Region-Instruction KV Cache, which reduces computational cost while incorporating global information. 3) Adaptive Velocity Decay Cache. Observing that adjacent timesteps in edited regions exhibit strong velocity similarity, we further propose an adaptive velocity decay cache to accelerate the local denoising process. We applied RegionE to state-of-the-art IIE base models, including Step1X-Edit, FLUX.1 Kontext, and Qwen-Image-Edit. RegionE achieved acceleration factors of 2.57, 2.41, and 2.06. Evaluations by GPT-4o confirmed that semantic and perceptual fidelity were well preserved.
>
---
#### [new 054] IBIS: A Powerful Hybrid Architecture for Human Activity Recognition
- **分类: cs.CV**

- **简介: 该论文针对Wi-Fi感知中活动识别的过拟合问题，提出IBIS混合架构，融合Inception-BiLSTM与SVM，提升模型泛化能力。基于多普勒数据，实现近99%的识别准确率，有效改善分类边界，推动非侵入式人体活动识别发展。**

- **链接: [http://arxiv.org/pdf/2510.24936v1](http://arxiv.org/pdf/2510.24936v1)**

> **作者:** Alison M. Fernandes; Hermes I. Del Monego; Bruno S. Chang; Anelise Munaretto; Hélder M. Fontes; Rui L. Campos
>
> **备注:** 8 pages. 8 figures. Wireless Days Conference, December 2025
>
> **摘要:** The increasing interest in Wi-Fi sensing stems from its potential to capture environmental data in a low-cost, non-intrusive way, making it ideal for applications like healthcare, space occupancy analysis, and gesture-based IoT control. However, a major limitation in this field is the common problem of overfitting, where models perform well on training data but fail to generalize to new data. To overcome this, we introduce a novel hybrid architecture that integrates Inception-BiLSTM with a Support Vector Machine (SVM), which we refer to as IBIS. Our IBIS approach is uniquely engineered to improve model generalization and create more robust classification boundaries. By applying this method to Doppler-derived data, we achieve a movement recognition accuracy of nearly 99%. Comprehensive performance metrics and confusion matrices confirm the significant effectiveness of our proposed solution.
>
---
#### [new 055] Test-Time Adaptive Object Detection with Foundation Model
- **分类: cs.CV**

- **简介: 该论文提出一种基于基础模型的测试时自适应目标检测方法，解决传统方法依赖源数据、受限于固定类别空间的问题。通过多模态提示调优与动态记忆模块，实现无需源数据的参数高效适应，支持跨域跨类别检测，显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.25175v1](http://arxiv.org/pdf/2510.25175v1)**

> **作者:** Yingjie Gao; Yanan Zhang; Zhi Cai; Di Huang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** In recent years, test-time adaptive object detection has attracted increasing attention due to its unique advantages in online domain adaptation, which aligns more closely with real-world application scenarios. However, existing approaches heavily rely on source-derived statistical characteristics while making the strong assumption that the source and target domains share an identical category space. In this paper, we propose the first foundation model-powered test-time adaptive object detection method that eliminates the need for source data entirely and overcomes traditional closed-set limitations. Specifically, we design a Multi-modal Prompt-based Mean-Teacher framework for vision-language detector-driven test-time adaptation, which incorporates text and visual prompt tuning to adapt both language and vision representation spaces on the test data in a parameter-efficient manner. Correspondingly, we propose a Test-time Warm-start strategy tailored for the visual prompts to effectively preserve the representation capability of the vision branch. Furthermore, to guarantee high-quality pseudo-labels in every test batch, we maintain an Instance Dynamic Memory (IDM) module that stores high-quality pseudo-labels from previous test samples, and propose two novel strategies-Memory Enhancement and Memory Hallucination-to leverage IDM's high-quality instances for enhancing original predictions and hallucinating images without available pseudo-labels, respectively. Extensive experiments on cross-corruption and cross-dataset benchmarks demonstrate that our method consistently outperforms previous state-of-the-art methods, and can adapt to arbitrary cross-domain and cross-category target data. Code is available at https://github.com/gaoyingjay/ttaod_foundation.
>
---
#### [new 056] Aligning What You Separate: Denoised Patch Mixing for Source-Free Domain Adaptation in Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对医疗图像分割中的源域无关域适应（SFDA）任务，解决隐私约束下域偏移导致的噪声标签与样本难易不均问题。提出基于困难样本选择与去噪块混合的框架，通过熵-相似性分析筛选可靠样本，利用蒙特卡洛去噪掩码优化伪标签，并分域混合块特征实现渐进式分布对齐，显著提升分割精度与边界清晰度。**

- **链接: [http://arxiv.org/pdf/2510.25227v1](http://arxiv.org/pdf/2510.25227v1)**

> **作者:** Quang-Khai Bui-Tran; Thanh-Huy Nguyen; Hoang-Thien Nguyen; Ba-Thinh Lam; Nguyen Lan Vi Vu; Phat K. Huynh; Ulas Bagci; Min Xu
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Source-Free Domain Adaptation (SFDA) is emerging as a compelling solution for medical image segmentation under privacy constraints, yet current approaches often ignore sample difficulty and struggle with noisy supervision under domain shift. We present a new SFDA framework that leverages Hard Sample Selection and Denoised Patch Mixing to progressively align target distributions. First, unlabeled images are partitioned into reliable and unreliable subsets through entropy-similarity analysis, allowing adaptation to start from easy samples and gradually incorporate harder ones. Next, pseudo-labels are refined via Monte Carlo-based denoising masks, which suppress unreliable pixels and stabilize training. Finally, intra- and inter-domain objectives mix patches between subsets, transferring reliable semantics while mitigating noise. Experiments on benchmark datasets show consistent gains over prior SFDA and UDA methods, delivering more accurate boundary delineation and achieving state-of-the-art Dice and ASSD scores. Our study highlights the importance of progressive adaptation and denoised supervision for robust segmentation under domain shift.
>
---
#### [new 057] Modality-Aware SAM: Sharpness-Aware-Minimization Driven Gradient Modulation for Harmonized Multimodal Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对多模态学习中主导模态压制其他模态的问题，提出M-SAM框架。通过Shapley值识别主导模态，分解损失景观并调制梯度，平衡各模态贡献，提升模型整体性能。适用于多种融合场景，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.24919v1](http://arxiv.org/pdf/2510.24919v1)**

> **作者:** Hossein R. Nowdeh; Jie Ji; Xiaolong Ma; Fatemeh Afghah
>
> **摘要:** In multimodal learning, dominant modalities often overshadow others, limiting generalization. We propose Modality-Aware Sharpness-Aware Minimization (M-SAM), a model-agnostic framework that applies to many modalities and supports early and late fusion scenarios. In every iteration, M-SAM in three steps optimizes learning. \textbf{First, it identifies the dominant modality} based on modalities' contribution in the accuracy using Shapley. \textbf{Second, it decomposes the loss landscape}, or in another language, it modulates the loss to prioritize the robustness of the model in favor of the dominant modality, and \textbf{third, M-SAM updates the weights} by backpropagation of modulated gradients. This ensures robust learning for the dominant modality while enhancing contributions from others, allowing the model to explore and exploit complementary features that strengthen overall performance. Extensive experiments on four diverse datasets show that M-SAM outperforms the latest state-of-the-art optimization and gradient manipulation methods and significantly balances and improves multimodal learning.
>
---
#### [new 058] ESCA: Enabling Seamless Codec Avatar Execution through Algorithm and Hardware Co-Optimization for Virtual Reality
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对虚拟现实中高保真编码化身实时渲染难题，提出ESCA框架，通过算法与硬件协同优化，实现低精度高效推理。工作包括定制后训练量化方法与专用硬件加速器，显著提升性能并满足实时性要求。**

- **链接: [http://arxiv.org/pdf/2510.24787v1](http://arxiv.org/pdf/2510.24787v1)**

> **作者:** Mingzhi Zhu; Ding Shang; Sai Qian Zhang
>
> **摘要:** Photorealistic Codec Avatars (PCA), which generate high-fidelity human face renderings, are increasingly being used in Virtual Reality (VR) environments to enable immersive communication and interaction through deep learning-based generative models. However, these models impose significant computational demands, making real-time inference challenging on resource-constrained VR devices such as head-mounted displays, where latency and power efficiency are critical. To address this challenge, we propose an efficient post-training quantization (PTQ) method tailored for Codec Avatar models, enabling low-precision execution without compromising output quality. In addition, we design a custom hardware accelerator that can be integrated into the system-on-chip of VR devices to further enhance processing efficiency. Building on these components, we introduce ESCA, a full-stack optimization framework that accelerates PCA inference on edge VR platforms. Experimental results demonstrate that ESCA boosts FovVideoVDP quality scores by up to $+0.39$ over the best 4-bit baseline, delivers up to $3.36\times$ latency reduction, and sustains a rendering rate of 100 frames per second in end-to-end tests, satisfying real-time VR requirements. These results demonstrate the feasibility of deploying high-fidelity codec avatars on resource-constrained devices, opening the door to more immersive and portable VR experiences.
>
---
#### [new 059] 3D CT-Based Coronary Calcium Assessment: A Feature-Driven Machine Learning Framework
- **分类: cs.CV; cs.LG; 68U10; I.2.1**

- **简介: 该论文针对冠状动脉钙化（CAC）检测任务，解决非对比CCTA图像中缺乏标注数据的问题。提出基于放射组学与伪标签的机器学习框架，利用预训练模型提取特征，并比较其与传统放射组学特征的性能。结果表明，放射组学方法在零/非零钙化分类上表现更优。**

- **链接: [http://arxiv.org/pdf/2510.25347v1](http://arxiv.org/pdf/2510.25347v1)**

> **作者:** Ayman Abaid; Gianpiero Guidone; Sara Alsubai; Foziyah Alquahtani; Talha Iqbal; Ruth Sharif; Hesham Elzomor; Emiliano Bianchini; Naeif Almagal; Michael G. Madden; Faisal Sharif; Ihsan Ullah
>
> **备注:** 11 pages, 2 Figures, MICCAI AMAI 2025 workshop, to be published in Volume 16206 of the Lecture Notes in Computer Science series
>
> **摘要:** Coronary artery calcium (CAC) scoring plays a crucial role in the early detection and risk stratification of coronary artery disease (CAD). In this study, we focus on non-contrast coronary computed tomography angiography (CCTA) scans, which are commonly used for early calcification detection in clinical settings. To address the challenge of limited annotated data, we propose a radiomics-based pipeline that leverages pseudo-labeling to generate training labels, thereby eliminating the need for expert-defined segmentations. Additionally, we explore the use of pretrained foundation models, specifically CT-FM and RadImageNet, to extract image features, which are then used with traditional classifiers. We compare the performance of these deep learning features with that of radiomics features. Evaluation is conducted on a clinical CCTA dataset comprising 182 patients, where individuals are classified into two groups: zero versus non-zero calcium scores. We further investigate the impact of training on non-contrast datasets versus combined contrast and non-contrast datasets, with testing performed only on non contrast scans. Results show that radiomics-based models significantly outperform CNN-derived embeddings from foundation models (achieving 84% accuracy and p<0.05), despite the unavailability of expert annotations.
>
---
#### [new 060] DualCap: Enhancing Lightweight Image Captioning via Dual Retrieval with Similar Scenes Visual Prompts
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对轻量级图像描述任务中视觉提示不足的问题，提出DualCap模型。通过双检索机制，结合文本与视觉相似场景，生成视觉提示以增强原图特征，提升细节表达能力。相比现有方法，模型参数更少，性能更优。**

- **链接: [http://arxiv.org/pdf/2510.24813v1](http://arxiv.org/pdf/2510.24813v1)**

> **作者:** Binbin Li; Guimiao Yang; Zisen Qi; Haiping Wang; Yu Ding
>
> **摘要:** Recent lightweight retrieval-augmented image caption models often utilize retrieved data solely as text prompts, thereby creating a semantic gap by leaving the original visual features unenhanced, particularly for object details or complex scenes. To address this limitation, we propose $DualCap$, a novel approach that enriches the visual representation by generating a visual prompt from retrieved similar images. Our model employs a dual retrieval mechanism, using standard image-to-text retrieval for text prompts and a novel image-to-image retrieval to source visually analogous scenes. Specifically, salient keywords and phrases are derived from the captions of visually similar scenes to capture key objects and similar details. These textual features are then encoded and integrated with the original image features through a lightweight, trainable feature fusion network. Extensive experiments demonstrate that our method achieves competitive performance while requiring fewer trainable parameters compared to previous visual-prompting captioning approaches.
>
---
#### [new 061] Breast Cancer VLMs: Clinically Practical Vision-Language Train-Inference Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对乳腺癌早期筛查中CAD系统临床部署难的问题，提出一种融合2D mammogram图像与临床文本的视觉语言模型。通过创新的特征提取与融合机制，实现高精度癌症及钙化检测，提升多国人群适用性，推动可落地的多模态辅助诊断系统发展。**

- **链接: [http://arxiv.org/pdf/2510.25051v1](http://arxiv.org/pdf/2510.25051v1)**

> **作者:** Shunjie-Fabian Zheng; Hyeonjun Lee; Thijs Kooi; Ali Diba
>
> **备注:** Accepted to Computer Vision for Automated Medical Diagnosis (CVAMD) Workshop at ICCV 2025
>
> **摘要:** Breast cancer remains the most commonly diagnosed malignancy among women in the developed world. Early detection through mammography screening plays a pivotal role in reducing mortality rates. While computer-aided diagnosis (CAD) systems have shown promise in assisting radiologists, existing approaches face critical limitations in clinical deployment - particularly in handling the nuanced interpretation of multi-modal data and feasibility due to the requirement of prior clinical history. This study introduces a novel framework that synergistically combines visual features from 2D mammograms with structured textual descriptors derived from easily accessible clinical metadata and synthesized radiological reports through innovative tokenization modules. Our proposed methods in this study demonstrate that strategic integration of convolutional neural networks (ConvNets) with language representations achieves superior performance to vision transformer-based models while handling high-resolution images and enabling practical deployment across diverse populations. By evaluating it on multi-national cohort screening mammograms, our multi-modal approach achieves superior performance in cancer detection and calcification identification compared to unimodal baselines, with particular improvements. The proposed method establishes a new paradigm for developing clinically viable VLM-based CAD systems that effectively leverage imaging data and contextual patient information through effective fusion mechanisms.
>
---
#### [new 062] Deep Feature Optimization for Enhanced Fish Freshness Assessment
- **分类: cs.CV; cs.AI; 68T07, 68U10; I.5.4**

- **简介: 该论文针对鱼鲜度视觉评估任务，解决传统方法主观、低效及深度学习模型可解释性差的问题。提出三阶段框架：细调五种主干网络提取特征，结合七类传统分类器，再用LGBM等方法筛选关键特征。在FFE数据集上达85.99%准确率，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.24814v1](http://arxiv.org/pdf/2510.24814v1)**

> **作者:** Phi-Hung Hoang; Nam-Thuan Trinh; Van-Manh Tran; Thi-Thu-Hong Phan
>
> **备注:** 39 pages; 10 tables; 9 figures
>
> **摘要:** Assessing fish freshness is vital for ensuring food safety and minimizing economic losses in the seafood industry. However, traditional sensory evaluation remains subjective, time-consuming, and inconsistent. Although recent advances in deep learning have automated visual freshness prediction, challenges related to accuracy and feature transparency persist. This study introduces a unified three-stage framework that refines and leverages deep visual representations for reliable fish freshness assessment. First, five state-of-the-art vision architectures - ResNet-50, DenseNet-121, EfficientNet-B0, ConvNeXt-Base, and Swin-Tiny - are fine-tuned to establish a strong baseline. Next, multi-level deep features extracted from these backbones are used to train seven classical machine learning classifiers, integrating deep and traditional decision mechanisms. Finally, feature selection methods based on Light Gradient Boosting Machine (LGBM), Random Forest, and Lasso identify a compact and informative subset of features. Experiments on the Freshness of the Fish Eyes (FFE) dataset demonstrate that the best configuration combining Swin-Tiny features, an Extra Trees classifier, and LGBM-based feature selection achieves an accuracy of 85.99%, outperforming recent studies on the same dataset by 8.69-22.78%. These findings confirm the effectiveness and generalizability of the proposed framework for visual quality evaluation tasks.
>
---
#### [new 063] AtlasGS: Atlanta-world Guided Surface Reconstruction with Implicit Structured Gaussians
- **分类: cs.CV**

- **简介: 该论文聚焦于室内与城市场景的3D表面重建任务，针对低纹理区域重建不一致及现有方法在细节保留与效率间的权衡问题。提出AtlasGS方法，结合Atlanta-world模型与隐式结构化高斯点云，通过语义表示与可学习平面正则化，实现全局一致、高效且高保真的表面重建。**

- **链接: [http://arxiv.org/pdf/2510.25129v1](http://arxiv.org/pdf/2510.25129v1)**

> **作者:** Xiyu Zhang; Chong Bao; Yipeng Chen; Hongjia Zhai; Yitong Dong; Hujun Bao; Zhaopeng Cui; Guofeng Zhang
>
> **备注:** 18 pages, 11 figures. NeurIPS 2025; Project page: https://zju3dv.github.io/AtlasGS/
>
> **摘要:** 3D reconstruction of indoor and urban environments is a prominent research topic with various downstream applications. However, existing geometric priors for addressing low-texture regions in indoor and urban settings often lack global consistency. Moreover, Gaussian Splatting and implicit SDF fields often suffer from discontinuities or exhibit computational inefficiencies, resulting in a loss of detail. To address these issues, we propose an Atlanta-world guided implicit-structured Gaussian Splatting that achieves smooth indoor and urban scene reconstruction while preserving high-frequency details and rendering efficiency. By leveraging the Atlanta-world model, we ensure the accurate surface reconstruction for low-texture regions, while the proposed novel implicit-structured GS representations provide smoothness without sacrificing efficiency and high-frequency details. Specifically, we propose a semantic GS representation to predict the probability of all semantic regions and deploy a structure plane regularization with learnable plane indicators for global accurate surface reconstruction. Extensive experiments demonstrate that our method outperforms state-of-the-art approaches in both indoor and urban scenes, delivering superior surface reconstruction quality.
>
---
#### [new 064] Balanced conic rectified flow
- **分类: cs.CV; 68T07, 68T45, 65C20; I.2.10; I.4.9; I.2.6**

- **简介: 该论文提出平衡锥形修正流（Balanced conic rectified flow），针对传统修正流依赖大量生成数据、易偏差的问题，引入真实图像优化训练。通过保留真实图像的ODE路径，减少对生成数据的依赖，提升生成质量与稳定性，在CIFAR-10上以更少数据实现更优FID得分和更直的生成路径。**

- **链接: [http://arxiv.org/pdf/2510.25229v1](http://arxiv.org/pdf/2510.25229v1)**

> **作者:** Kim Shin Seong; Mingi Kwon; Jaeseok Jeong; Youngjung Uh
>
> **备注:** Main paper: 10 pages (total 40 pages including appendix), 5 figures. Accepted at NeurIPS 2025 (Poster). Acknowledgment: Supported by the NRF of Korea (RS-2023-00223062) and IITP grants (RS-2020-II201361, RS-2024-00439762) funded by the Korean government (MSIT)
>
> **摘要:** Rectified flow is a generative model that learns smooth transport mappings between two distributions through an ordinary differential equation (ODE). Unlike diffusion-based generative models, which require costly numerical integration of a generative ODE to sample images with state-of-the-art quality, rectified flow uses an iterative process called reflow to learn smooth and straight ODE paths. This allows for relatively simple and efficient generation of high-quality images. However, rectified flow still faces several challenges. 1) The reflow process requires a large number of generative pairs to preserve the target distribution, leading to significant computational costs. 2) Since the model is typically trained using only generated image pairs, its performance heavily depends on the 1-rectified flow model, causing it to become biased towards the generated data. In this work, we experimentally expose the limitations of the original rectified flow and propose a novel approach that incorporates real images into the training process. By preserving the ODE paths for real images, our method effectively reduces reliance on large amounts of generated data. Instead, we demonstrate that the reflow process can be conducted efficiently using a much smaller set of generated and real images. In CIFAR-10, we achieved significantly better FID scores, not only in one-step generation but also in full-step simulations, while using only of the generative pairs compared to the original method. Furthermore, our approach induces straighter paths and avoids saturation on generated images during reflow, leading to more robust ODE learning while preserving the distribution of real images.
>
---
#### [new 065] Combining SAR Simulators to Train ATR Models with Synthetic Data
- **分类: cs.CV; cs.AI; eess.SP**

- **简介: 该论文聚焦于合成孔径雷达（SAR）图像的自动目标识别（ATR）任务，旨在解决真实标注数据稀缺问题。通过结合两种基于不同物理模型的SAR模拟器（MOCEM与Salsa），生成更丰富的合成数据，提升深度学习模型在真实数据上的泛化能力。实验表明，所提方法在MSTAR数据集上达到近88%的识别准确率。**

- **链接: [http://arxiv.org/pdf/2510.24768v1](http://arxiv.org/pdf/2510.24768v1)**

> **作者:** Benjamin Camus; Julien Houssay; Corentin Le Barbu; Eric Monteux; Cédric Saleun; Christian Cochin
>
> **摘要:** This work aims to train Deep Learning models to perform Automatic Target Recognition (ATR) on Synthetic Aperture Radar (SAR) images. To circumvent the lack of real labelled measurements, we resort to synthetic data produced by SAR simulators. Simulation offers full control over the virtual environment, which enables us to generate large and diversified datasets at will. However, simulations are intrinsically grounded on simplifying assumptions of the real world (i.e. physical models). Thus, synthetic datasets are not as representative as real measurements. Consequently, ATR models trained on synthetic images cannot generalize well on real measurements. Our contributions to this problem are twofold: on one hand, we demonstrate and quantify the impact of the simulation paradigm on the ATR. On the other hand, we propose a new approach to tackle the ATR problem: combine two SAR simulators that are grounded on different (but complementary) paradigms to produce synthetic datasets. To this end, we use two simulators: MOCEM, which is based on a scattering centers model approach, and Salsa, which resorts on a ray tracing strategy. We train ATR models using synthetic dataset generated both by MOCEM and Salsa and our Deep Learning approach called ADASCA. We reach an accuracy of almost 88 % on the MSTAR measurements.
>
---
#### [new 066] FPGA-based Lane Detection System incorporating Temperature and Light Control Units
- **分类: cs.CV; eess.IV**

- **简介: 该论文提出一种基于FPGA的车道检测系统，旨在解决智能车辆在复杂环境下的车道识别问题。通过Sobel算法实现图像边缘检测，实时处理416×416图像，每1.17ms输出车道数量、索引及边界信息，并集成光温自适应控制单元，提升系统环境适应性。**

- **链接: [http://arxiv.org/pdf/2510.24778v1](http://arxiv.org/pdf/2510.24778v1)**

> **作者:** Ibrahim Qamar; Saber Mahmoud; Seif Megahed; Mohamed Khaled; Saleh Hesham; Ahmed Matar; Saif Gebril; Mervat Mahmoud
>
> **备注:** 5 pages, 8 figures, 3 tables
>
> **摘要:** Intelligent vehicles are one of the most important outcomes gained from the world tendency toward automation. Applications of IVs, whether in urban roads or robot tracks, do prioritize lane path detection. This paper proposes an FPGA-based Lane Detector Vehicle LDV architecture that relies on the Sobel algorithm for edge detection. Operating on 416 x 416 images and 150 MHz, the system can generate a valid output every 1.17 ms. The valid output consists of the number of present lanes, the current lane index, as well as its right and left boundaries. Additionally, the automated light and temperature control units in the proposed system enhance its adaptability to the surrounding environmental conditions.
>
---
#### [new 067] Classifier Enhancement Using Extended Context and Domain Experts for Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文针对语义分割中分类器因固定参数无法适应图像级类别分布差异及数据集级类别不平衡的问题，提出扩展上下文感知分类器（ECAC）。通过记忆库融合全局与局部上下文信息，并引入教师-学生框架由领域专家动态优化分类器，显著提升少数类分割性能。**

- **链接: [http://arxiv.org/pdf/2510.25174v1](http://arxiv.org/pdf/2510.25174v1)**

> **作者:** Huadong Tang; Youpeng Zhao; Min Xu; Jun Wang; Qiang Wu
>
> **备注:** Accepted at IEEE TRANSACTIONS ON MULTIMEDIA (TMM)
>
> **摘要:** Prevalent semantic segmentation methods generally adopt a vanilla classifier to categorize each pixel into specific classes. Although such a classifier learns global information from the training data, this information is represented by a set of fixed parameters (weights and biases). However, each image has a different class distribution, which prevents the classifier from addressing the unique characteristics of individual images. At the dataset level, class imbalance leads to segmentation results being biased towards majority classes, limiting the model's effectiveness in identifying and segmenting minority class regions. In this paper, we propose an Extended Context-Aware Classifier (ECAC) that dynamically adjusts the classifier using global (dataset-level) and local (image-level) contextual information. Specifically, we leverage a memory bank to learn dataset-level contextual information of each class, incorporating the class-specific contextual information from the current image to improve the classifier for precise pixel labeling. Additionally, a teacher-student network paradigm is adopted, where the domain expert (teacher network) dynamically adjusts contextual information with ground truth and transfers knowledge to the student network. Comprehensive experiments illustrate that the proposed ECAC can achieve state-of-the-art performance across several datasets, including ADE20K, COCO-Stuff10K, and Pascal-Context.
>
---
#### [new 068] U-CAN: Unsupervised Point Cloud Denoising with Consistency-Aware Noise2Noise Matching
- **分类: cs.CV**

- **简介: 该论文提出U-CAN，一种无监督点云去噪框架。针对现有方法依赖干净数据标注的问题，利用噪声到噪声匹配与几何一致性约束，通过多步去噪路径学习去噪先验，实现无需成对数据的高效去噪，在点云与图像去噪任务中均达到先进性能。**

- **链接: [http://arxiv.org/pdf/2510.25210v1](http://arxiv.org/pdf/2510.25210v1)**

> **作者:** Junsheng Zhou; Xingyu Shi; Haichuan Song; Yi Fang; Yu-Shen Liu; Zhizhong Han
>
> **备注:** Accepted by NeurIPS 2025. Project page: https://gloriasze.github.io/U-CAN/
>
> **摘要:** Point clouds captured by scanning sensors are often perturbed by noise, which have a highly negative impact on downstream tasks (e.g. surface reconstruction and shape understanding). Previous works mostly focus on training neural networks with noisy-clean point cloud pairs for learning denoising priors, which requires extensively manual efforts. In this work, we introduce U-CAN, an Unsupervised framework for point cloud denoising with Consistency-Aware Noise2Noise matching. Specifically, we leverage a neural network to infer a multi-step denoising path for each point of a shape or scene with a noise to noise matching scheme. We achieve this by a novel loss which enables statistical reasoning on multiple noisy point cloud observations. We further introduce a novel constraint on the denoised geometry consistency for learning consistency-aware denoising patterns. We justify that the proposed constraint is a general term which is not limited to 3D domain and can also contribute to the area of 2D image denoising. Our evaluations under the widely used benchmarks in point cloud denoising, upsampling and image denoising show significant improvement over the state-of-the-art unsupervised methods, where U-CAN also produces comparable results with the supervised methods.
>
---
#### [new 069] DeepShield: Fortifying Deepfake Video Detection with Local and Global Forgery Analysis
- **分类: cs.CV**

- **简介: 该论文聚焦于深度伪造视频检测任务，针对现有方法在跨域场景下泛化能力差的问题，提出DeepShield框架。通过局部细节分析与全局特征多样化增强，提升对未知伪造手法的鲁棒性，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.25237v1](http://arxiv.org/pdf/2510.25237v1)**

> **作者:** Yinqi Cai; Jichang Li; Zhaolun Li; Weikai Chen; Rushi Lan; Xi Xie; Xiaonan Luo; Guanbin Li
>
> **备注:** ICCV 2025
>
> **摘要:** Recent advances in deep generative models have made it easier to manipulate face videos, raising significant concerns about their potential misuse for fraud and misinformation. Existing detectors often perform well in in-domain scenarios but fail to generalize across diverse manipulation techniques due to their reliance on forgery-specific artifacts. In this work, we introduce DeepShield, a novel deepfake detection framework that balances local sensitivity and global generalization to improve robustness across unseen forgeries. DeepShield enhances the CLIP-ViT encoder through two key components: Local Patch Guidance (LPG) and Global Forgery Diversification (GFD). LPG applies spatiotemporal artifact modeling and patch-wise supervision to capture fine-grained inconsistencies often overlooked by global models. GFD introduces domain feature augmentation, leveraging domain-bridging and boundary-expanding feature generation to synthesize diverse forgeries, mitigating overfitting and enhancing cross-domain adaptability. Through the integration of novel local and global analysis for deepfake detection, DeepShield outperforms state-of-the-art methods in cross-dataset and cross-manipulation evaluations, achieving superior robustness against unseen deepfake attacks.
>
---
#### [new 070] Neighborhood Feature Pooling for Remote Sensing Image Classification
- **分类: cs.CV; eess.IV; 68T07; I.4.8; I.2.10**

- **简介: 该论文针对遥感图像分类任务，提出邻域特征池化（NFP）方法，用于有效提取纹理特征。通过捕捉邻近像素间关系并聚合局部相似性，NFP在不显著增加参数量的前提下，提升多种网络架构的分类性能。**

- **链接: [http://arxiv.org/pdf/2510.25077v1](http://arxiv.org/pdf/2510.25077v1)**

> **作者:** Fahimeh Orvati Nia; Amirmohammad Mohammadi; Salim Al Kharsa; Pragati Naikare; Zigfried Hampel-Arias; Joshua Peeples
>
> **备注:** 9 pages, 5 figures. Accepted to WACV 2026 (Winter Conference on Applications of Computer Vision)
>
> **摘要:** In this work, we propose neighborhood feature pooling (NFP) as a novel texture feature extraction method for remote sensing image classification. The NFP layer captures relationships between neighboring inputs and efficiently aggregates local similarities across feature dimensions. Implemented using convolutional layers, NFP can be seamlessly integrated into any network. Results comparing the baseline models and the NFP method indicate that NFP consistently improves performance across diverse datasets and architectures while maintaining minimal parameter overhead.
>
---
#### [new 071] Conflict Adaptation in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉语言模型在冲突任务中的适应行为。针对认知控制资源分配问题，通过序列斯特鲁普任务发现多数VLMs具冲突适应现象。利用稀疏自编码器分析InternVL 3.5 4B，识别出与文本和颜色相关的重叠超节点，并定位一个关键冲突调制超节点，其移除显著增加错误率。**

- **链接: [http://arxiv.org/pdf/2510.24804v1](http://arxiv.org/pdf/2510.24804v1)**

> **作者:** Xiaoyang Hu
>
> **备注:** Workshop on Interpreting Cognition in Deep Learning Models at NeurIPS 2025
>
> **摘要:** A signature of human cognitive control is conflict adaptation: improved performance on a high-conflict trial following another high-conflict trial. This phenomenon offers an account for how cognitive control, a scarce resource, is recruited. Using a sequential Stroop task, we find that 12 of 13 vision-language models (VLMs) tested exhibit behavior consistent with conflict adaptation, with the lone exception likely reflecting a ceiling effect. To understand the representational basis of this behavior, we use sparse autoencoders (SAEs) to identify task-relevant supernodes in InternVL 3.5 4B. Partially overlapping supernodes emerge for text and color in both early and late layers, and their relative sizes mirror the automaticity asymmetry between reading and color naming in humans. We further isolate a conflict-modulated supernode in layers 24-25 whose ablation significantly increases Stroop errors while minimally affecting congruent trials.
>
---
#### [new 072] Hawk: Leveraging Spatial Context for Faster Autoregressive Text-to-Image Generation
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对自回归文本到图像生成速度慢的问题，提出Hawk方法。通过利用图像的二维空间结构，增强轻量级草稿模型的预测准确性，提升推测解码效率。实验表明，该方法在保持图像质量与多样性的同时，实现1.71倍加速。**

- **链接: [http://arxiv.org/pdf/2510.25739v1](http://arxiv.org/pdf/2510.25739v1)**

> **作者:** Zhi-Kai Chen; Jun-Peng Jiang; Han-Jia Ye; De-Chuan Zhan
>
> **摘要:** Autoregressive (AR) image generation models are capable of producing high-fidelity images but often suffer from slow inference due to their inherently sequential, token-by-token decoding process. Speculative decoding, which employs a lightweight draft model to approximate the output of a larger AR model, has shown promise in accelerating text generation without compromising quality. However, its application to image generation remains largely underexplored. The challenges stem from a significantly larger sampling space, which complicates the alignment between the draft and target model outputs, coupled with the inadequate use of the two-dimensional spatial structure inherent in images, thereby limiting the modeling of local dependencies. To overcome these challenges, we introduce Hawk, a new approach that harnesses the spatial structure of images to guide the speculative model toward more accurate and efficient predictions. Experimental results on multiple text-to-image benchmarks demonstrate a 1.71x speedup over standard AR models, while preserving both image fidelity and diversity.
>
---
#### [new 073] Seeing Clearly and Deeply: An RGBD Imaging Approach with a Bio-inspired Monocentric Design
- **分类: cs.CV; cs.RO; eess.IV; physics.optics**

- **简介: 该论文针对紧凑型RGBD成像中图像清晰度与深度精度难以兼顾的问题，提出生物启发的全球面单中心镜头与联合重建框架（BMI）。通过物理建模生成合成数据，结合双头多尺度网络，实现单次拍摄下高保真全焦图像与精确深度图的联合恢复，显著优于现有软硬件方案。**

- **链接: [http://arxiv.org/pdf/2510.25314v1](http://arxiv.org/pdf/2510.25314v1)**

> **作者:** Zongxi Yu; Xiaolong Qian; Shaohua Gao; Qi Jiang; Yao Gao; Kailun Yang; Kaiwei Wang
>
> **备注:** The source code will be publicly available at https://github.com/ZongxiYu-ZJU/BMI
>
> **摘要:** Achieving high-fidelity, compact RGBD imaging presents a dual challenge: conventional compact optics struggle with RGB sharpness across the entire depth-of-field, while software-only Monocular Depth Estimation (MDE) is an ill-posed problem reliant on unreliable semantic priors. While deep optics with elements like DOEs can encode depth, they introduce trade-offs in fabrication complexity and chromatic aberrations, compromising simplicity. To address this, we first introduce a novel bio-inspired all-spherical monocentric lens, around which we build the Bionic Monocentric Imaging (BMI) framework, a holistic co-design. This optical design naturally encodes depth into its depth-varying Point Spread Functions (PSFs) without requiring complex diffractive or freeform elements. We establish a rigorous physically-based forward model to generate a synthetic dataset by precisely simulating the optical degradation process. This simulation pipeline is co-designed with a dual-head, multi-scale reconstruction network that employs a shared encoder to jointly recover a high-fidelity All-in-Focus (AiF) image and a precise depth map from a single coded capture. Extensive experiments validate the state-of-the-art performance of the proposed framework. In depth estimation, the method attains an Abs Rel of 0.026 and an RMSE of 0.130, markedly outperforming leading software-only approaches and other deep optics systems. For image restoration, the system achieves an SSIM of 0.960 and a perceptual LPIPS score of 0.082, thereby confirming a superior balance between image fidelity and depth accuracy. This study illustrates that the integration of bio-inspired, fully spherical optics with a joint reconstruction algorithm constitutes an effective strategy for addressing the intrinsic challenges in high-performance compact RGBD imaging. Source code will be publicly available at https://github.com/ZongxiYu-ZJU/BMI.
>
---
#### [new 074] AI-Powered Early Detection of Critical Diseases using Image Processing and Audio Analysis
- **分类: cs.CV**

- **简介: 该论文提出一种多模态AI诊断框架，用于早期检测皮肤癌、血栓和心肺异常。结合图像、热成像与音频分析，采用轻量级模型实现高精度分类，解决传统诊断成本高、难普及的问题，推动低成本、实时可部署的智能医疗预诊。**

- **链接: [http://arxiv.org/pdf/2510.25199v1](http://arxiv.org/pdf/2510.25199v1)**

> **作者:** Manisha More; Kavya Bhand; Kaustubh Mukdam; Kavya Sharma; Manas Kawtikwar; Hridayansh Kaware; Prajwal Kavhar
>
> **摘要:** Early diagnosis of critical diseases can significantly improve patient survival and reduce treatment costs. However, existing diagnostic techniques are often costly, invasive, and inaccessible in low-resource regions. This paper presents a multimodal artificial intelligence (AI) diagnostic framework integrating image analysis, thermal imaging, and audio signal processing for early detection of three major health conditions: skin cancer, vascular blood clots, and cardiopulmonary abnormalities. A fine-tuned MobileNetV2 convolutional neural network was trained on the ISIC 2019 dataset for skin lesion classification, achieving 89.3% accuracy, 91.6% sensitivity, and 88.2% specificity. A support vector machine (SVM) with handcrafted features was employed for thermal clot detection, achieving 86.4% accuracy (AUC = 0.89) on synthetic and clinical data. For cardiopulmonary analysis, lung and heart sound datasets from PhysioNet and Pascal were processed using Mel-Frequency Cepstral Coefficients (MFCC) and classified via Random Forest, reaching 87.2% accuracy and 85.7% sensitivity. Comparative evaluation against state-of-the-art models demonstrates that the proposed system achieves competitive results while remaining lightweight and deployable on low-cost devices. The framework provides a promising step toward scalable, real-time, and accessible AI-based pre-diagnostic healthcare solutions.
>
---
#### [new 075] Pixels to Signals: A Real-Time Framework for Traffic Demand Estimation
- **分类: cs.CV**

- **简介: 该论文属于交通需求估计任务，旨在解决城市交通拥堵问题。针对实时车辆检测，提出基于视频流的框架：通过帧间平均构建背景模型，利用DBSCAN算法提取前景并检测车辆，实现高效、低改造成本的实时车辆检测。**

- **链接: [http://arxiv.org/pdf/2510.24902v1](http://arxiv.org/pdf/2510.24902v1)**

> **作者:** H Mhatre; M Vyas; A Mittal
>
> **摘要:** Traffic congestion is becoming a challenge in the rapidly growing urban cities, resulting in increasing delays and inefficiencies within urban transportation systems. To address this issue a comprehensive methodology is designed to optimize traffic flow and minimize delays. The framework is structured with three primary components: (a) vehicle detection, (b) traffic prediction, and (c) traffic signal optimization. This paper presents the first component, vehicle detection. The methodology involves analyzing multiple sequential frames from a camera feed to compute the background, i.e. the underlying roadway, by averaging pixel values over time. The computed background is then utilized to extract the foreground, where the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm is applied to detect vehicles. With its computational efficiency and minimal infrastructure modification requirements, the proposed methodology offers a practical and scalable solution for real-world deployment.
>
---
#### [new 076] MMEdge: Accelerating On-device Multimodal Inference via Pipelined Sensing and Encoding
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对边缘设备上多模态实时推理的延迟与资源受限问题，提出MMEdge框架。通过流水线式感知与编码，实现数据到达即计算，并引入时序聚合、自适应配置优化与跨模态推测跳过机制，显著降低延迟并保持高精度。**

- **链接: [http://arxiv.org/pdf/2510.25327v1](http://arxiv.org/pdf/2510.25327v1)**

> **作者:** Runxi Huang; Mingxuan Yu; Mingyu Tsoi; Xiaomin Ouyang
>
> **备注:** Accepted by SenSys 2026
>
> **摘要:** Real-time multimodal inference on resource-constrained edge devices is essential for applications such as autonomous driving, human-computer interaction, and mobile health. However, prior work often overlooks the tight coupling between sensing dynamics and model execution, as well as the complex inter-modality dependencies. In this paper, we propose MMEdge, an new on-device multi-modal inference framework based on pipelined sensing and encoding. Instead of waiting for complete sensor inputs, MMEdge decomposes the entire inference process into a sequence of fine-grained sensing and encoding units, allowing computation to proceed incrementally as data arrive. MMEdge also introduces a lightweight but effective temporal aggregation module that captures rich temporal dynamics across different pipelined units to maintain accuracy performance. Such pipelined design also opens up opportunities for fine-grained cross-modal optimization and early decision-making during inference. To further enhance system performance under resource variability and input data complexity, MMEdge incorporates an adaptive multimodal configuration optimizer that dynamically selects optimal sensing and model configurations for each modality under latency constraints, and a cross-modal speculative skipping mechanism that bypasses future units of slower modalities when early predictions reach sufficient confidence. We evaluate MMEdge using two public multimodal datasets and deploy it on a real-world unmanned aerial vehicle (UAV)-based multimodal testbed. The results show that MMEdge significantly reduces end-to-end latency while maintaining high task accuracy across various system and data dynamics.
>
---
#### [new 077] Transformers in Medicine: Improving Vision-Language Alignment for Medical Image Captioning
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文聚焦医学图像描述生成任务，旨在提升MRI图像与文本的语义对齐。提出基于Transformer的多模态框架，结合视觉编码器、文本嵌入模型与自定义解码器，采用混合损失与对比推理优化对齐效果。在多类别医学影像数据集上验证，表明专注领域数据可显著提升描述准确性与临床相关性。**

- **链接: [http://arxiv.org/pdf/2510.25164v1](http://arxiv.org/pdf/2510.25164v1)**

> **作者:** Yogesh Thakku Suresh; Vishwajeet Shivaji Hogale; Luca-Alexandru Zamfira; Anandavardhana Hegde
>
> **备注:** This work is to appear in the Proceedings of MICAD 2025, the 6th International Conference on Medical Imaging and Computer-Aided Diagnosis
>
> **摘要:** We present a transformer-based multimodal framework for generating clinically relevant captions for MRI scans. Our system combines a DEiT-Small vision transformer as an image encoder, MediCareBERT for caption embedding, and a custom LSTM-based decoder. The architecture is designed to semantically align image and textual embeddings, using hybrid cosine-MSE loss and contrastive inference via vector similarity. We benchmark our method on the MultiCaRe dataset, comparing performance on filtered brain-only MRIs versus general MRI images against state-of-the-art medical image captioning methods including BLIP, R2GenGPT, and recent transformer-based approaches. Results show that focusing on domain-specific data improves caption accuracy and semantic alignment. Our work proposes a scalable, interpretable solution for automated medical image reporting.
>
---
#### [new 078] Modelling the Interplay of Eye-Tracking Temporal Dynamics and Personality for Emotion Detection in Face-to-Face Settings
- **分类: cs.HC; cs.AI; cs.CV**

- **简介: 该论文属于情感识别任务，旨在解决动态对话场景中情绪感知的主观性问题。通过融合眼动时序数据、大五人格特质与情境刺激，构建个性化多模态模型，区分“感知”与“真实”情绪，显著提升预测准确率，推动更自然、个性化的智能交互系统发展。**

- **链接: [http://arxiv.org/pdf/2510.24720v1](http://arxiv.org/pdf/2510.24720v1)**

> **作者:** Meisam J. Seikavandi; Jostein Fimland; Fabricio Batista Narcizo; Maria Barrett; Ted Vucurevich; Jesper Bünsow Boldt; Andrew Burke Dittberner; Paolo Burelli
>
> **摘要:** Accurate recognition of human emotions is critical for adaptive human-computer interaction, yet remains challenging in dynamic, conversation-like settings. This work presents a personality-aware multimodal framework that integrates eye-tracking sequences, Big Five personality traits, and contextual stimulus cues to predict both perceived and felt emotions. Seventy-three participants viewed speech-containing clips from the CREMA-D dataset while providing eye-tracking signals, personality assessments, and emotion ratings. Our neural models captured temporal gaze dynamics and fused them with trait and stimulus information, yielding consistent gains over SVM and literature baselines. Results show that (i) stimulus cues strongly enhance perceived-emotion predictions (macro F1 up to 0.77), while (ii) personality traits provide the largest improvements for felt emotion recognition (macro F1 up to 0.58). These findings highlight the benefit of combining physiological, trait-level, and contextual information to address the inherent subjectivity of emotion. By distinguishing between perceived and felt responses, our approach advances multimodal affective computing and points toward more personalized and ecologically valid emotion-aware systems.
>
---
#### [new 079] FaCT: Faithful Concept Traces for Explaining Neural Network Decisions
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文针对深度神经网络解释性问题，提出FaCT方法，通过模型内生机制实现跨类共享概念的忠实追踪，可准确可视化概念贡献。引入C²-Score评估概念一致性，提升解释可读性与模型性能。**

- **链接: [http://arxiv.org/pdf/2510.25512v1](http://arxiv.org/pdf/2510.25512v1)**

> **作者:** Amin Parchami-Araghi; Sukrut Rao; Jonas Fischer; Bernt Schiele
>
> **备注:** Accepted to NeurIPS 2025; Code is available at https://github.com/m-parchami/FaCT
>
> **摘要:** Deep networks have shown remarkable performance across a wide range of tasks, yet getting a global concept-level understanding of how they function remains a key challenge. Many post-hoc concept-based approaches have been introduced to understand their workings, yet they are not always faithful to the model. Further, they make restrictive assumptions on the concepts a model learns, such as class-specificity, small spatial extent, or alignment to human expectations. In this work, we put emphasis on the faithfulness of such concept-based explanations and propose a new model with model-inherent mechanistic concept-explanations. Our concepts are shared across classes and, from any layer, their contribution to the logit and their input-visualization can be faithfully traced. We also leverage foundation models to propose a new concept-consistency metric, C$^2$-Score, that can be used to evaluate concept-based methods. We show that, compared to prior work, our concepts are quantitatively more consistent and users find our concepts to be more interpretable, all while retaining competitive ImageNet performance.
>
---
#### [new 080] SCOUT: A Lightweight Framework for Scenario Coverage Assessment in Autonomous Driving
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出SCOUT框架，用于高效评估自动驾驶场景覆盖率。针对现有方法依赖昂贵人工标注或高成本大模型的问题，提出基于代理模型的轻量级方案，通过知识蒸馏从大模型标签学习，直接利用预计算感知特征进行快速预测，显著降低计算开销，实现大规模场景覆盖分析。**

- **链接: [http://arxiv.org/pdf/2510.24949v1](http://arxiv.org/pdf/2510.24949v1)**

> **作者:** Anil Yildiz; Sarah M. Thornton; Carl Hildebrandt; Sreeja Roy-Singh; Mykel J. Kochenderfer
>
> **摘要:** Assessing scenario coverage is crucial for evaluating the robustness of autonomous agents, yet existing methods rely on expensive human annotations or computationally intensive Large Vision-Language Models (LVLMs). These approaches are impractical for large-scale deployment due to cost and efficiency constraints. To address these shortcomings, we propose SCOUT (Scenario Coverage Oversight and Understanding Tool), a lightweight surrogate model designed to predict scenario coverage labels directly from an agent's latent sensor representations. SCOUT is trained through a distillation process, learning to approximate LVLM-generated coverage labels while eliminating the need for continuous LVLM inference or human annotation. By leveraging precomputed perception features, SCOUT avoids redundant computations and enables fast, scalable scenario coverage estimation. We evaluate our method across a large dataset of real-life autonomous navigation scenarios, demonstrating that it maintains high accuracy while significantly reducing computational cost. Our results show that SCOUT provides an effective and practical alternative for large-scale coverage analysis. While its performance depends on the quality of LVLM-generated training labels, SCOUT represents a major step toward efficient scenario coverage oversight in autonomous systems.
>
---
#### [new 081] DMVFC: Deep Learning Based Functionally Consistent Tractography Fiber Clustering Using Multimodal Diffusion MRI and Functional MRI
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出DMVFC框架，用于基于多模态扩散MRI和功能MRI的白质纤维簇聚类。旨在解决传统方法忽略功能与微结构信息的问题，通过深度学习融合几何、微结构及功能信号，实现功能一致的白质分区，提升脑连接分析的准确性。**

- **链接: [http://arxiv.org/pdf/2510.24770v1](http://arxiv.org/pdf/2510.24770v1)**

> **作者:** Bocheng Guo; Jin Wang; Yijie Li; Junyi Wang; Mingyu Gao; Puming Feng; Yuqian Chen; Jarrett Rushmore; Nikos Makris; Yogesh Rathi; Lauren J O'Donnell; Fan Zhang
>
> **备注:** 11 pages
>
> **摘要:** Tractography fiber clustering using diffusion MRI (dMRI) is a crucial method for white matter (WM) parcellation to enable analysis of brains structural connectivity in health and disease. Current fiber clustering strategies primarily use the fiber geometric characteristics (i.e., the spatial trajectories) to group similar fibers into clusters, while neglecting the functional and microstructural information of the fiber tracts. There is increasing evidence that neural activity in the WM can be measured using functional MRI (fMRI), providing potentially valuable multimodal information for fiber clustering to enhance its functional coherence. Furthermore, microstructural features such as fractional anisotropy (FA) can be computed from dMRI as additional information to ensure the anatomical coherence of the clusters. In this paper, we develop a novel deep learning fiber clustering framework, namely Deep Multi-view Fiber Clustering (DMVFC), which uses joint multi-modal dMRI and fMRI data to enable functionally consistent WM parcellation. DMVFC can effectively integrate the geometric and microstructural characteristics of the WM fibers with the fMRI BOLD signals along the fiber tracts. DMVFC includes two major components: (1) a multi-view pretraining module to compute embedding features from each source of information separately, including fiber geometry, microstructure measures, and functional signals, and (2) a collaborative fine-tuning module to simultaneously refine the differences of embeddings. In the experiments, we compare DMVFC with two state-of-the-art fiber clustering methods and demonstrate superior performance in achieving functionally meaningful and consistent WM parcellation results.
>
---
#### [new 082] Seeing Through the MiRAGE: Evaluating Multimodal Retrieval Augmented Generation
- **分类: cs.CL; cs.CV; cs.IR**

- **简介: 该论文提出MiRAGE框架，用于评估多模态检索增强生成（RAG）系统。针对现有文本中心评估方法无法有效验证多模态信息的问题，提出InfoF1与CiteF1指标，实现对事实性与引用完整性的量化评估，并提供自动版本与开源实现，推动多模态RAG的可靠评估。**

- **链接: [http://arxiv.org/pdf/2510.24870v1](http://arxiv.org/pdf/2510.24870v1)**

> **作者:** Alexander Martin; William Walden; Reno Kriz; Dengjia Zhang; Kate Sanders; Eugene Yang; Chihsheng Jin; Benjamin Van Durme
>
> **备注:** https://github.com/alexmartin1722/mirage
>
> **摘要:** We introduce MiRAGE, an evaluation framework for retrieval-augmented generation (RAG) from multimodal sources. As audiovisual media becomes a prevalent source of information online, it is essential for RAG systems to integrate information from these sources into generation. However, existing evaluations for RAG are text-centric, limiting their applicability to multimodal, reasoning intensive settings because they don't verify information against sources. MiRAGE is a claim-centric approach to multimodal RAG evaluation, consisting of InfoF1, evaluating factuality and information coverage, and CiteF1, measuring citation support and completeness. We show that MiRAGE, when applied by humans, strongly aligns with extrinsic quality judgments. We additionally introduce automatic variants of MiRAGE and three prominent TextRAG metrics -- ACLE, ARGUE, and RAGAS -- demonstrating the limitations of text-centric work and laying the groundwork for automatic evaluation. We release open-source implementations and outline how to assess multimodal RAG.
>
---
#### [new 083] SynHLMA:Synthesizing Hand Language Manipulation for Articulated Object with Discrete Human Object Interaction Representation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出SynHLMA框架，解决基于语言指令生成关节物体手部操作序列的问题。通过离散的HAOI表示与语言嵌入对齐，在共享空间中建模手物交互，结合关节感知损失，实现生成、预测与插值三类任务，支持机器人灵巧抓握应用。**

- **链接: [http://arxiv.org/pdf/2510.25268v1](http://arxiv.org/pdf/2510.25268v1)**

> **作者:** Wang zhi; Yuyan Liu; Liu Liu; Li Zhang; Ruixuan Lu; Dan Guo
>
> **摘要:** Generating hand grasps with language instructions is a widely studied topic that benefits from embodied AI and VR/AR applications. While transferring into hand articulatied object interaction (HAOI), the hand grasps synthesis requires not only object functionality but also long-term manipulation sequence along the object deformation. This paper proposes a novel HAOI sequence generation framework SynHLMA, to synthesize hand language manipulation for articulated objects. Given a complete point cloud of an articulated object, we utilize a discrete HAOI representation to model each hand object interaction frame. Along with the natural language embeddings, the representations are trained by an HAOI manipulation language model to align the grasping process with its language description in a shared representation space. A joint-aware loss is employed to ensure hand grasps follow the dynamic variations of articulated object joints. In this way, our SynHLMA achieves three typical hand manipulation tasks for articulated objects of HAOI generation, HAOI prediction and HAOI interpolation. We evaluate SynHLMA on our built HAOI-lang dataset and experimental results demonstrate the superior hand grasp sequence generation performance comparing with state-of-the-art. We also show a robotics grasp application that enables dexterous grasps execution from imitation learning using the manipulation sequence provided by our SynHLMA. Our codes and datasets will be made publicly available.
>
---
#### [new 084] Feedback Alignment Meets Low-Rank Manifolds: A Structured Recipe for Local Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对深度神经网络训练中反向传播的高内存与计算开销问题，提出一种基于低秩流形的局部学习框架。通过在奇异值分解（SVD）结构上进行参数更新，结合结构化反馈矩阵与多目标损失函数，实现高效、可扩展的局部训练，在保持BP性能的同时显著降低参数量。**

- **链接: [http://arxiv.org/pdf/2510.25594v1](http://arxiv.org/pdf/2510.25594v1)**

> **作者:** Arani Roy; Marco P. Apolinario; Shristi Das Biswas; Kaushik Roy
>
> **摘要:** Training deep neural networks (DNNs) with backpropagation (BP) achieves state-of-the-art accuracy but requires global error propagation and full parameterization, leading to substantial memory and computational overhead. Direct Feedback Alignment (DFA) enables local, parallelizable updates with lower memory requirements but is limited by unstructured feedback and poor scalability in deeper architectures, specially convolutional neural networks. To address these limitations, we propose a structured local learning framework that operates directly on low-rank manifolds defined by the Singular Value Decomposition (SVD) of weight matrices. Each layer is trained in its decomposed form, with updates applied to the SVD components using a composite loss that integrates cross-entropy, subspace alignment, and orthogonality regularization. Feedback matrices are constructed to match the SVD structure, ensuring consistent alignment between forward and feedback pathways. Our method reduces the number of trainable parameters relative to the original DFA model, without relying on pruning or post hoc compression. Experiments on CIFAR-10, CIFAR-100, and ImageNet show that our method achieves accuracy comparable to that of BP. Ablation studies confirm the importance of each loss term in the low-rank setting. These results establish local learning on low-rank manifolds as a principled and scalable alternative to full-rank gradient-based training.
>
---
#### [new 085] Resi-VidTok: An Efficient and Decomposed Progressive Tokenization Framework for Ultra-Low-Rate and Lightweight Video Transmission
- **分类: cs.IT; cs.CV; cs.MM; eess.IV; math.IT**

- **简介: 该论文针对无线网络下超低码率视频传输难题，提出Resi-VidTok框架。通过分层令牌化与渐进编码，实现轻量级、抗损性强的视频传输，在极低带宽（CBR=0.0004）下仍保持高感知与语义保真，支持实时重建，适用于低功耗、高可靠场景。**

- **链接: [http://arxiv.org/pdf/2510.25002v1](http://arxiv.org/pdf/2510.25002v1)**

> **作者:** Zhenyu Liu; Yi Ma; Rahim Tafazolli; Zhi Ding
>
> **摘要:** Real-time transmission of video over wireless networks remains highly challenging, even with advanced deep models, particularly under severe channel conditions such as limited bandwidth and weak connectivity. In this paper, we propose Resi-VidTok, a Resilient Tokenization-Enabled framework designed for ultra-low-rate and lightweight video transmission that delivers strong robustness while preserving perceptual and semantic fidelity on commodity digital hardware. By reorganizing spatio--temporal content into a discrete, importance-ordered token stream composed of key tokens and refinement tokens, Resi-VidTok enables progressive encoding, prefix-decodable reconstruction, and graceful quality degradation under constrained channels. A key contribution is a resilient 1D tokenization pipeline for video that integrates differential temporal token coding, explicitly supporting reliable recovery from incomplete token sets using a single shared framewise decoder--without auxiliary temporal extractors or heavy generative models. Furthermore, stride-controlled frame sparsification combined with a lightweight decoder-side interpolator reduces transmission load while maintaining motion continuity. Finally, a channel-adaptive source--channel coding and modulation scheme dynamically allocates rate and protection according to token importance and channel condition, yielding stable quality across adverse SNRs. Evaluation results indicate robust visual and semantic consistency at channel bandwidth ratios (CBR) as low as 0.0004 and real-time reconstruction at over 30 fps, demonstrating the practicality of Resi-VidTok for energy-efficient, latency-sensitive, and reliability-critical wireless applications.
>
---
#### [new 086] CT-Less Attenuation Correction Using Multiview Ensemble Conditional Diffusion Model on High-Resolution Uncorrected PET Images
- **分类: q-bio.QM; cs.AI; cs.CV**

- **简介: 该论文提出一种无需真实CT的衰减校正方法，利用多视角联合条件扩散模型从非校正PET图像生成伪CT图像。旨在解决传统PET/CT中辐射暴露与配准误差问题，通过集成投票提升伪CT质量，显著降低重建误差，实现高精度定量PET成像。**

- **链接: [http://arxiv.org/pdf/2510.24805v1](http://arxiv.org/pdf/2510.24805v1)**

> **作者:** Alexandre St-Georges; Gabriel Richard; Maxime Toussaint; Christian Thibaudeau; Etienne Auger; Étienne Croteau; Stephen Cunnane; Roger Lecomte; Jean-Baptiste Michaud
>
> **备注:** This is a preprint and not the final version of this paper
>
> **摘要:** Accurate quantification in positron emission tomography (PET) is essential for accurate diagnostic results and effective treatment tracking. A major issue encountered in PET imaging is attenuation. Attenuation refers to the diminution of photon detected as they traverse biological tissues before reaching detectors. When such corrections are absent or inadequate, this signal degradation can introduce inaccurate quantification, making it difficult to differentiate benign from malignant conditions, and can potentially lead to misdiagnosis. Typically, this correction is done with co-computed Computed Tomography (CT) imaging to obtain structural data for calculating photon attenuation across the body. However, this methodology subjects patients to extra ionizing radiation exposure, suffers from potential spatial misregistration between PET/CT imaging sequences, and demands costly equipment infrastructure. Emerging advances in neural network architectures present an alternative approach via synthetic CT image synthesis. Our investigation reveals that Conditional Denoising Diffusion Probabilistic Models (DDPMs) can generate high quality CT images from non attenuation corrected PET images in order to correct attenuation. By utilizing all three orthogonal views from non-attenuation-corrected PET images, the DDPM approach combined with ensemble voting generates higher quality pseudo-CT images with reduced artifacts and improved slice-to-slice consistency. Results from a study of 159 head scans acquired with the Siemens Biograph Vision PET/CT scanner demonstrate both qualitative and quantitative improvements in pseudo-CT generation. The method achieved a mean absolute error of 32 $\pm$ 10.4 HU on the CT images and an average error of (1.48 $\pm$ 0.68)\% across all regions of interest when comparing PET images reconstructed using the attenuation map of the generated pseudo-CT versus the true CT.
>
---
#### [new 087] CFL-SparseMed: Communication-Efficient Federated Learning for Medical Imaging with Top-k Sparse Updates
- **分类: eess.IV; cs.CV; cs.DC; cs.LG**

- **简介: 该论文针对医疗影像分类中的隐私与通信效率问题，提出CFL-SparseMed方法。通过Top-k梯度稀疏化，减少联邦学习通信开销，有效应对非独立同分布数据挑战，提升模型精度与协作效率，保障患者隐私。**

- **链接: [http://arxiv.org/pdf/2510.24776v1](http://arxiv.org/pdf/2510.24776v1)**

> **作者:** Gousia Habib; Aniket Bhardwaj; Ritvik Sharma; Shoeib Amin Banday; Ishfaq Ahmad Malik
>
> **摘要:** Secure and reliable medical image classification is crucial for effective patient treatment, but centralized models face challenges due to data and privacy concerns. Federated Learning (FL) enables privacy-preserving collaborations but struggles with heterogeneous, non-IID data and high communication costs, especially in large networks. We propose \textbf{CFL-SparseMed}, an FL approach that uses Top-k Sparsification to reduce communication overhead by transmitting only the top k gradients. This unified solution effectively addresses data heterogeneity while maintaining model accuracy. It enhances FL efficiency, preserves privacy, and improves diagnostic accuracy and patient care in non-IID medical imaging settings. The reproducibility source code is available on \href{https://github.com/Aniket2241/APK_contruct}{Github}.
>
---
## 更新

#### [replaced 001] FastJAM: a Fast Joint Alignment Model for Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.22842v2](http://arxiv.org/pdf/2510.22842v2)**

> **作者:** Omri Hirsch; Ron Shapira Weber; Shira Ifergane; Oren Freifeld
>
> **备注:** Accepted to NeurIPS 2025. Pages 1-10 are the Main Paper. Pages 23-31 are Supplemental Material. FastJAM website - https://bgu-cs-vil.github.io/FastJAM/
>
> **摘要:** Joint Alignment (JA) of images aims to align a collection of images into a unified coordinate frame, such that semantically-similar features appear at corresponding spatial locations. Most existing approaches often require long training times, large-capacity models, and extensive hyperparameter tuning. We introduce FastJAM, a rapid, graph-based method that drastically reduces the computational complexity of joint alignment tasks. FastJAM leverages pairwise matches computed by an off-the-shelf image matcher, together with a rapid nonparametric clustering, to construct a graph representing intra- and inter-image keypoint relations. A graph neural network propagates and aggregates these correspondences, efficiently predicting per-image homography parameters via image-level pooling. Utilizing an inverse-compositional loss, that eliminates the need for a regularization term over the predicted transformations (and thus also obviates the hyperparameter tuning associated with such terms), FastJAM performs image JA quickly and effectively. Experimental results on several benchmarks demonstrate that FastJAM achieves results better than existing modern JA methods in terms of alignment quality, while reducing computation time from hours or minutes to mere seconds. Our code is available at our project webpage, https://bgu-cs-vil.github.io/FastJAM/
>
---
#### [replaced 002] Diverse Teaching and Label Propagation for Generic Semi-Supervised Medical Image Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.08549v3](http://arxiv.org/pdf/2508.08549v3)**

> **作者:** Wei Li; Pengcheng Zhou; Linye Ma; Wenyi Zhao; Huihua Yang; Yuchen Guo
>
> **备注:** Under Review
>
> **摘要:** Both limited annotation and domain shift are significant challenges frequently encountered in medical image segmentation, leading to derivative scenarios like semi-supervised medical (SSMIS), semi-supervised medical domain generalization (Semi-MDG) and unsupervised medical domain adaptation (UMDA). Conventional methods are generally tailored to specific tasks in isolation, the error accumulation hinders the effective utilization of unlabeled data and limits further improvements, resulting in suboptimal performance when these issues occur. In this paper, we aim to develop a generic framework that masters all three tasks. We found that the key to solving the problem lies in how to generate reliable pseudo labels for the unlabeled data in the presence of domain shift with labeled data and increasing the diversity of the model. To tackle this issue, we employ a Diverse Teaching and Label Propagation Network (DTLP-Net) to boosting the Generic Semi-Supervised Medical Image Segmentation. Our DTLP-Net involves a single student model and two diverse teacher models, which can generate reliable pseudo-labels for the student model. The first teacher model decouple the training process with labeled and unlabeled data, The second teacher is momentum-updated periodically, thus generating reliable yet divers pseudo-labels. To fully utilize the information within the data, we adopt inter-sample and intra-sample data augmentation to learn the global and local knowledge. In addition, to further capture the voxel-level correlations, we propose label propagation to enhance the model robust. We evaluate our proposed framework on five benchmark datasets for SSMIS, UMDA, and Semi-MDG tasks. The results showcase notable improvements compared to state-of-the-art methods across all five settings, indicating the potential of our framework to tackle more challenging SSL scenarios.
>
---
#### [replaced 003] When are radiology reports useful for training medical image classifiers?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.24385v2](http://arxiv.org/pdf/2510.24385v2)**

> **作者:** Herman Bergström; Zhongqi Yue; Fredrik D. Johansson
>
> **摘要:** Medical images used to train machine learning models are often accompanied by radiology reports containing rich expert annotations. However, relying on these reports as inputs for clinical prediction requires the timely manual work of a trained radiologist. This raises a natural question: when can radiology reports be leveraged during training to improve image-only classification? Prior works are limited to evaluating pre-trained image representations by fine-tuning them to predict diagnostic labels, often extracted from reports, ignoring tasks with labels that are weakly associated with the text. To address this gap, we conduct a systematic study of how radiology reports can be used during both pre-training and fine-tuning, across diagnostic and prognostic tasks (e.g., 12-month readmission), and under varying training set sizes. Our findings reveal that: (1) Leveraging reports during pre-training is beneficial for downstream classification tasks where the label is well-represented in the text; however, pre-training through explicit image-text alignment can be detrimental in settings where it's not; (2) Fine-tuning with reports can lead to significant improvements and even have a larger impact than the pre-training method in certain settings. These results provide actionable insights into when and how to leverage privileged text data to train medical image classifiers while highlighting gaps in current research.
>
---
#### [replaced 004] VLCE: A Knowledge-Enhanced Framework for Image Description in Disaster Assessment
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.21609v3](http://arxiv.org/pdf/2509.21609v3)**

> **作者:** Md. Mahfuzur Rahman; Kishor Datta Gupta; Marufa Kamal; Fahad Rahman; Sunzida Siddique; Ahmed Rafi Hasan; Mohd Ariful Haque; Roy George
>
> **备注:** 29 pages, 40 figures, 3 algorithms
>
> **摘要:** Immediate damage assessment is essential after natural catastrophes; yet, conventional hand evaluation techniques are sluggish and perilous. Although satellite and unmanned aerial vehicle (UAV) photos offer extensive perspectives of impacted regions, current computer vision methodologies generally yield just classification labels or segmentation masks, so constraining their capacity to deliver a thorough situational comprehension. We introduce the Vision Language Caption Enhancer (VLCE), a multimodal system designed to produce comprehensive, contextually-informed explanations of disaster imagery. VLCE employs a dual-architecture approach: a CNN-LSTM model with a ResNet50 backbone pretrained on EuroSat satellite imagery for the xBD dataset, and a Vision Transformer (ViT) model pretrained on UAV pictures for the RescueNet dataset. Both systems utilize external semantic knowledge from ConceptNet and WordNet to expand vocabulary coverage and improve description accuracy. We assess VLCE in comparison to leading vision-language models (LLaVA and QwenVL) utilizing CLIPScore for semantic alignment and InfoMetIC for caption informativeness. Experimental findings indicate that VLCE markedly surpasses baseline models, attaining a maximum of 95.33% on InfoMetIC while preserving competitive semantic alignment. Our dual-architecture system demonstrates significant potential for improving disaster damage assessment by automating the production of actionable, information-dense descriptions from satellite and drone photos.
>
---
#### [replaced 005] InfoChartQA: A Benchmark for Multimodal Question Answering on Infographic Charts
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.19028v4](http://arxiv.org/pdf/2505.19028v4)**

> **作者:** Tianchi Xie; Minzhi Lin; Mengchen Liu; Yilin Ye; Changjian Chen; Shixia Liu
>
> **摘要:** Understanding infographic charts with design-driven visual elements (e.g., pictograms, icons) requires both visual recognition and reasoning, posing challenges for multimodal large language models (MLLMs). However, existing visual-question answering benchmarks fall short in evaluating these capabilities of MLLMs due to the lack of paired plain charts and visual-element-based questions. To bridge this gap, we introduce InfoChartQA, a benchmark for evaluating MLLMs on infographic chart understanding. It includes 5,642 pairs of infographic and plain charts, each sharing the same underlying data but differing in visual presentations. We further design visual-element-based questions to capture their unique visual designs and communicative intent. Evaluation of 20 MLLMs reveals a substantial performance decline on infographic charts, particularly for visual-element-based questions related to metaphors. The paired infographic and plain charts enable fine-grained error analysis and ablation studies, which highlight new opportunities for advancing MLLMs in infographic chart understanding. We release InfoChartQA at https://github.com/CoolDawnAnt/InfoChartQA.
>
---
#### [replaced 006] Graph-Theoretic Consistency for Robust and Topology-Aware Semi-Supervised Histopathology Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.22689v2](http://arxiv.org/pdf/2509.22689v2)**

> **作者:** Ha-Hieu Pham; Minh Le; Han Huynh; Nguyen Quoc Khanh Le; Huy-Hieu Pham
>
> **备注:** Accepted to the AAAI 2026 Student Abstract and Poster Program
>
> **摘要:** Semi-supervised semantic segmentation (SSSS) is vital in computational pathology, where dense annotations are costly and limited. Existing methods often rely on pixel-level consistency, which propagates noisy pseudo-labels and produces fragmented or topologically invalid masks. We propose Topology Graph Consistency (TGC), a framework that integrates graph-theoretic constraints by aligning Laplacian spectra, component counts, and adjacency statistics between prediction graphs and references. This enforces global topology and improves segmentation accuracy. Experiments on GlaS and CRAG demonstrate that TGC achieves state-of-the-art performance under 5-10% supervision and significantly narrows the gap to full supervision.
>
---
#### [replaced 007] Probabilistic Kernel Function for Fast Angle Testing
- **分类: cs.LG; cs.AI; cs.CV; cs.DB; cs.DS**

- **链接: [http://arxiv.org/pdf/2505.20274v2](http://arxiv.org/pdf/2505.20274v2)**

> **作者:** Kejing Lu; Chuan Xiao; Yoshiharu Ishikawa
>
> **摘要:** In this paper, we study the angle testing problem in the context of similarity search in high-dimensional Euclidean spaces and propose two projection-based probabilistic kernel functions, one designed for angle comparison and the other for angle thresholding. Unlike existing approaches that rely on random projection vectors drawn from Gaussian distributions, our approach leverages reference angles and employs a deterministic structure for the projection vectors. Notably, our kernel functions do not require asymptotic assumptions, such as the number of projection vectors tending to infinity, and can be both theoretically and experimentally shown to outperform Gaussian-distribution-based kernel functions. We apply the proposed kernel function to Approximate Nearest Neighbor Search (ANNS) and demonstrate that our approach achieves a 2.5X ~ 3X higher query-per-second (QPS) throughput compared to the widely-used graph-based search algorithm HNSW.
>
---
#### [replaced 008] Explicitly Modeling Subcortical Vision with a Neuro-Inspired Front-End Improves CNN Robustness
- **分类: cs.CV; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2506.03089v2](http://arxiv.org/pdf/2506.03089v2)**

> **作者:** Lucas Piper; Arlindo L. Oliveira; Tiago Marques
>
> **摘要:** Convolutional neural networks (CNNs) trained on object recognition achieve high task performance but continue to exhibit vulnerability under a range of visual perturbations and out-of-domain images, when compared with biological vision. Prior work has demonstrated that coupling a standard CNN with a front-end (VOneBlock) that mimics the primate primary visual cortex (V1) can improve overall model robustness. Expanding on this, we introduce Early Vision Networks (EVNets), a new class of hybrid CNNs that combine the VOneBlock with a novel SubcorticalBlock, whose architecture draws from computational models in neuroscience and is parameterized to maximize alignment with subcortical responses reported across multiple experimental studies. Without being optimized to do so, the assembly of the SubcorticalBlock with the VOneBlock improved V1 alignment across most standard V1 benchmarks, and better modeled extra-classical receptive field phenomena. In addition, EVNets exhibit stronger emergent shape bias and outperform the base CNN architecture by 9.3% on an aggregate benchmark of robustness evaluations, including adversarial perturbations, common corruptions, and domain shifts. Finally, we show that EVNets can be further improved when paired with a state-of-the-art data augmentation technique, surpassing the performance of the isolated data augmentation approach by 6.2% on our robustness benchmark. This result reveals complementary benefits between changes in architecture to better mimic biology and training-based machine learning approaches.
>
---
#### [replaced 009] HyperET: Efficient Training in Hyperbolic Space for Multi-modal Large Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.20322v2](http://arxiv.org/pdf/2510.20322v2)**

> **作者:** Zelin Peng; Zhengqin Xu; Qingyang Liu; Xiaokang Yang; Wei Shen
>
> **备注:** Accepted by NeurIPS2025 (Oral)
>
> **摘要:** Multi-modal large language models (MLLMs) have emerged as a transformative approach for aligning visual and textual understanding. They typically require extremely high computational resources (e.g., thousands of GPUs) for training to achieve cross-modal alignment at multi-granularity levels. We argue that a key source of this inefficiency lies in the vision encoders they widely equip with, e.g., CLIP and SAM, which lack the alignment with language at multi-granularity levels. To address this issue, in this paper, we leverage hyperbolic space, which inherently models hierarchical levels and thus provides a principled framework for bridging the granularity gap between visual and textual modalities at an arbitrary granularity level. Concretely, we propose an efficient training paradigm for MLLMs, dubbed as HyperET, which can optimize visual representations to align with their textual counterparts at an arbitrary granularity level through dynamic hyperbolic radius adjustment in hyperbolic space. HyperET employs learnable matrices with M\"{o}bius multiplication operations, implemented via three effective configurations: diagonal scaling matrices, block-diagonal matrices, and banded matrices, providing a flexible yet efficient parametrization strategy. Comprehensive experiments across multiple MLLM benchmarks demonstrate that HyperET consistently improves both existing pre-training and fine-tuning MLLMs clearly with less than 1\% additional parameters.
>
---
#### [replaced 010] MILo: Mesh-In-the-Loop Gaussian Splatting for Detailed and Efficient Surface Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.24096v2](http://arxiv.org/pdf/2506.24096v2)**

> **作者:** Antoine Guédon; Diego Gomez; Nissim Maruani; Bingchen Gong; George Drettakis; Maks Ovsjanikov
>
> **备注:** 10 pages. A presentation video of our approach is available at https://youtu.be/_SGNhhNz0fE
>
> **摘要:** While recent advances in Gaussian Splatting have enabled fast reconstruction of high-quality 3D scenes from images, extracting accurate surface meshes remains a challenge. Current approaches extract the surface through costly post-processing steps, resulting in the loss of fine geometric details or requiring significant time and leading to very dense meshes with millions of vertices. More fundamentally, the a posteriori conversion from a volumetric to a surface representation limits the ability of the final mesh to preserve all geometric structures captured during training. We present MILo, a novel Gaussian Splatting framework that bridges the gap between volumetric and surface representations by differentiably extracting a mesh from the 3D Gaussians. We design a fully differentiable procedure that constructs the mesh-including both vertex locations and connectivity-at every iteration directly from the parameters of the Gaussians, which are the only quantities optimized during training. Our method introduces three key technical contributions: a bidirectional consistency framework ensuring both representations-Gaussians and the extracted mesh-capture the same underlying geometry during training; an adaptive mesh extraction process performed at each training iteration, which uses Gaussians as differentiable pivots for Delaunay triangulation; a novel method for computing signed distance values from the 3D Gaussians that enables precise surface extraction while avoiding geometric erosion. Our approach can reconstruct complete scenes, including backgrounds, with state-of-the-art quality while requiring an order of magnitude fewer mesh vertices than previous methods. Due to their light weight and empty interior, our meshes are well suited for downstream applications such as physics simulations or animation.
>
---
#### [replaced 011] FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.17685v2](http://arxiv.org/pdf/2505.17685v2)**

> **作者:** Shuang Zeng; Xinyuan Chang; Mengwei Xie; Xinran Liu; Yifan Bai; Zheng Pan; Mu Xu; Xing Wei
>
> **备注:** Accepted to NeurIPS 2025 as Spotlight Presentation. Code: https://github.com/MIV-XJTU/FSDrive
>
> **摘要:** Vision-Language-Action (VLA) models are increasingly used for end-to-end driving due to their world knowledge and reasoning ability. Most prior work, however, inserts textual chains-of-thought (CoT) as intermediate steps tailored to the current scene. Such symbolic compressions can blur spatio-temporal relations and discard fine visual cues, creating a cross-modal gap between perception and planning. We propose FSDrive, a visual spatio-temporal CoT framework that enables VLAs to think in images. The model first acts as a world model to generate a unified future frame that overlays coarse but physically-plausible priors-future lane dividers and 3D boxes-on the predicted future image. This unified frame serves as the visual CoT, capturing both spatial structure and temporal evolution. The same VLA then functions as an inverse-dynamics model, planning trajectories from current observations and the visual CoT. To equip VLAs with image generation while preserving understanding, we introduce a unified pre-training paradigm that expands the vocabulary to include visual tokens and jointly optimizes VQA (for semantics) and future-frame prediction (for dynamics). A progressive easy-to-hard scheme first predicts lane/box priors to enforce physical constraints, then completes full future frames for fine details. On nuScenes and NAVSIM, FSDrive improves trajectory accuracy and reduces collisions under both ST-P3 and UniAD metrics, and attains competitive FID for future-frame generation despite using lightweight autoregression. It also advances scene understanding on DriveLM. Together, these results indicate that visual CoT narrows the cross-modal gap and yields safer, more anticipatory planning. Code is available at https://github.com/MIV-XJTU/FSDrive.
>
---
#### [replaced 012] DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.07464v3](http://arxiv.org/pdf/2506.07464v3)**

> **作者:** Jinyoung Park; Jeehye Na; Jinyoung Kim; Hyunwoo J. Kim
>
> **备注:** NeurIPS 2025
>
> **摘要:** Recent works have demonstrated the effectiveness of reinforcement learning (RL)-based post-training for enhancing the reasoning capabilities of large language models (LLMs). In particular, Group Relative Policy Optimization (GRPO) has shown impressive success using a PPO-style reinforcement algorithm with group-normalized rewards. However, the effectiveness of GRPO in Video Large Language Models (VideoLLMs) has still been less studyed. In this paper, we explore GRPO and identify two problems that deteriorate the effective learning: (1) reliance on safeguards, and (2) vanishing advantage. To mitigate these challenges, we propose DeepVideo-R1, a video large language model trained with Reg-GRPO (Regressive GRPO) and difficulty-aware data augmentation. Reg-GRPO reformulates the GRPO loss function into a regression task that directly predicts the advantage in GRPO, eliminating the need for safeguards such as the clipping and min functions. It directly aligns the model with advantages, providing guidance to prefer better ones. The difficulty-aware data augmentation strategy augments input prompts/videos to locate the difficulty of samples at solvable difficulty levels, enabling diverse reward signals. Our experimental results show that our approach significantly improves video reasoning performance across multiple benchmarks.
>
---
#### [replaced 013] MagicPortrait: Temporally Consistent Face Reenactment with 3D Geometric Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.21497v3](http://arxiv.org/pdf/2504.21497v3)**

> **作者:** Mengting Wei; Yante Li; Tuomas Varanka; Yan Jiang; Guoying Zhao
>
> **摘要:** In this study, we propose a method for video face reenactment that integrates a 3D face parametric model into a latent diffusion framework, aiming to improve shape consistency and motion control in existing video-based face generation approaches. Our approach employs the FLAME (Faces Learned with an Articulated Model and Expressions) model as the 3D face parametric representation, providing a unified framework for modeling face expressions and head pose. This not only enables precise extraction of motion features from driving videos, but also contributes to the faithful preservation of face shape and geometry. Specifically, we enhance the latent diffusion model with rich 3D expression and detailed pose information by incorporating depth maps, normal maps, and rendering maps derived from FLAME sequences. These maps serve as motion guidance and are encoded into the denoising UNet through a specifically designed Geometric Guidance Encoder (GGE). A multi-layer feature fusion module with integrated self-attention mechanisms is used to combine facial appearance and motion latent features within the spatial domain. By utilizing the 3D face parametric model as motion guidance, our method enables parametric alignment of face identity between the reference image and the motion captured from the driving video. Experimental results on benchmark datasets show that our method excels at generating high-quality face animations with precise expression and head pose variation modeling. In addition, it demonstrates strong generalization performance on out-of-domain images. Code is publicly available at https://github.com/weimengting/MagicPortrait.
>
---
#### [replaced 014] Video-LMM Post-Training: A Deep Dive into Video Reasoning with Large Multimodal Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.05034v5](http://arxiv.org/pdf/2510.05034v5)**

> **作者:** Yolo Yunlong Tang; Jing Bi; Pinxin Liu; Zhenyu Pan; Zhangyun Tan; Qianxiang Shen; Jiani Liu; Hang Hua; Junjia Guo; Yunzhong Xiao; Chao Huang; Zhiyuan Wang; Susan Liang; Xinyi Liu; Yizhi Song; Junhua Huang; Jia-Xing Zhong; Bozheng Li; Daiqing Qi; Ziyun Zeng; Ali Vosoughi; Luchuan Song; Zeliang Zhang; Daiki Shimada; Han Liu; Jiebo Luo; Chenliang Xu
>
> **备注:** Version v1.1
>
> **摘要:** Video understanding represents the most challenging frontier in computer vision, requiring models to reason about complex spatiotemporal relationships, long-term dependencies, and multimodal evidence. The recent emergence of Video-Large Multimodal Models (Video-LMMs), which integrate visual encoders with powerful decoder-based language models, has demonstrated remarkable capabilities in video understanding tasks. However, the critical phase that transforms these models from basic perception systems into sophisticated reasoning engines, post-training, remains fragmented across the literature. This survey provides the first comprehensive examination of post-training methodologies for Video-LMMs, encompassing three fundamental pillars: supervised fine-tuning (SFT) with chain-of-thought, reinforcement learning (RL) from verifiable objectives, and test-time scaling (TTS) through enhanced inference computation. We present a structured taxonomy that clarifies the roles, interconnections, and video-specific adaptations of these techniques, addressing unique challenges such as temporal localization, spatiotemporal grounding, long video efficiency, and multimodal evidence integration. Through systematic analysis of representative methods, we synthesize key design principles, insights, and evaluation protocols while identifying critical open challenges in reward design, scalability, and cost-performance optimization. We further curate essential benchmarks, datasets, and metrics to facilitate rigorous assessment of post-training effectiveness. This survey aims to provide researchers and practitioners with a unified framework for advancing Video-LMM capabilities. Additional resources and updates are maintained at: https://github.com/yunlong10/Awesome-Video-LMM-Post-Training
>
---
#### [replaced 015] Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.08619v3](http://arxiv.org/pdf/2412.08619v3)**

> **作者:** Vahid Balazadeh; Mohammadmehdi Ataei; Hyunmin Cheong; Amir Hosein Khasahmadi; Rahul G. Krishnan
>
> **摘要:** Physical reasoning remains a significant challenge for Vision-Language Models (VLMs). This limitation arises from an inability to translate learned knowledge into predictions about physical behavior. Although continual fine-tuning can mitigate this issue, it is expensive for large models and impractical to perform repeatedly for every task. This necessitates the creation of modular and scalable ways to teach VLMs about physical reasoning. To that end, we introduce Physics Context Builders (PCBs), a modular framework where specialized smaller VLMs are fine-tuned to generate detailed physical scene descriptions. These can be used as physical contexts to enhance the reasoning capabilities of larger VLMs. PCBs enable the separation of visual perception from reasoning, allowing us to analyze their relative contributions to physical understanding. We perform experiments on CLEVRER and on Falling Tower, a stability detection dataset with both simulated and real-world scenes, to demonstrate that PCBs provide substantial performance improvements, increasing average accuracy by up to 13.8% on complex physical reasoning tasks. Notably, PCBs also show strong Sim2Real transfer, successfully generalizing from simulated training data to real-world scenes.
>
---
#### [replaced 016] AI in Lung Health: Benchmarking Detection and Diagnostic Models Across Multiple CT Scan Datasets
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2405.04605v4](http://arxiv.org/pdf/2405.04605v4)**

> **作者:** Fakrul Islam Tushar; Avivah Wang; Lavsen Dahal; Ehsan Samei; Michael R. Harowicz; Jayashree Kalpathy-Cramer; Kyle J. Lafata; Tina D. Tailor; Cynthia Rudin; Joseph Y. Lo
>
> **备注:** 2 tables, 5 figures
>
> **摘要:** Background: Development of artificial intelligence (AI) models for lung cancer screening requires large, well-annotated low-dose computed tomography (CT) datasets and rigorous performance benchmarks. Purpose: To create a reproducible benchmarking resource leveraging the Duke Lung Cancer Screening (DLCS) and multiple public datasets to develop and evaluate models for nodule detection and classification. Materials & Methods: This retrospective study uses the DLCS dataset (1,613 patients; 2,487 nodules) and external datasets including LUNA16, LUNA25, and NLST-3D. For detection, MONAI RetinaNet models were trained on DLCS (DLCS-De) and LUNA16 (LUNA16-De) and evaluated using the Competition Performance Metric (CPM). For nodule-level classification, we compare five strategies: pretrained models (Models Genesis, Med3D), a self-supervised foundation model (FMCB), and ResNet50 with random initialization versus Strategic Warm-Start (ResNet50-SWS) pretrained with detection-derived candidate patches stratified by confidence. Results: For detection on the DLCS test set, DLCS-De achieved sensitivity 0.82 at 2 false positives/scan (CPM 0.63) versus LUNA16-De (0.62, CPM 0.45). For external validation on NLST-3D, DLCS-De (sensitivity 0.72, CPM 0.58) also outperformed LUNA16-De (sensitivity 0.64, CPM 0.49). For classification across multiple datasets, ResNet50-SWS attained AUCs of 0.71 (DLCS; 95% CI, 0.61-0.81), 0.90 (LUNA16; 0.87-0.93), 0.81 (NLST-3D; 0.79-0.82), and 0.80 (LUNA25; 0.78-0.82), matching or exceeding pretrained/self-supervised baselines. Performance differences reflected dataset label standards. Conclusion: This work establishes a standardized benchmarking resource for lung cancer AI research, supporting model development, validation, and translation. All code, models, and data are publicly released to promote reproducibility.
>
---
#### [replaced 017] Hyperparameters in Continual Learning: A Reality Check
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2403.09066v5](http://arxiv.org/pdf/2403.09066v5)**

> **作者:** Sungmin Cha; Kyunghyun Cho
>
> **备注:** TMLR 2025 camera ready version
>
> **摘要:** Continual learning (CL) aims to train a model on a sequence of tasks (i.e., a CL scenario) while balancing the trade-off between plasticity (learning new tasks) and stability (retaining prior knowledge). The dominantly adopted conventional evaluation protocol for CL algorithms selects the best hyperparameters (e.g., learning rate, mini-batch size, regularization strengths, etc.) within a given scenario and then evaluates the algorithms using these hyperparameters in the same scenario. However, this protocol has significant shortcomings: it overestimates the CL capacity of algorithms and relies on unrealistic hyperparameter tuning, which is not feasible for real-world applications. From the fundamental principles of evaluation in machine learning, we argue that the evaluation of CL algorithms should focus on assessing the generalizability of their CL capacity to unseen scenarios. Based on this, we propose the Generalizable Two-phase Evaluation Protocol (GTEP) consisting of hyperparameter tuning and evaluation phases. Both phases share the same scenario configuration (e.g., number of tasks) but are generated from different datasets. Hyperparameters of CL algorithms are tuned in the first phase and applied in the second phase to evaluate the algorithms. We apply this protocol to class-incremental learning, both with and without pretrained models. Across more than 8,000 experiments, our results show that most state-of-the-art algorithms fail to replicate their reported performance, highlighting that their CL capacity has been significantly overestimated in the conventional evaluation protocol. Our implementation can be found in https://github.com/csm9493/GTEP.
>
---
#### [replaced 018] Pixel-Perfect Depth with Semantics-Prompted Diffusion Transformers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.07316v2](http://arxiv.org/pdf/2510.07316v2)**

> **作者:** Gangwei Xu; Haotong Lin; Hongcheng Luo; Xianqi Wang; Jingfeng Yao; Lianghui Zhu; Yuechuan Pu; Cheng Chi; Haiyang Sun; Bing Wang; Guang Chen; Hangjun Ye; Sida Peng; Xin Yang
>
> **备注:** NeurIPS 2025. Project page: https://pixel-perfect-depth.github.io/
>
> **摘要:** This paper presents Pixel-Perfect Depth, a monocular depth estimation model based on pixel-space diffusion generation that produces high-quality, flying-pixel-free point clouds from estimated depth maps. Current generative depth estimation models fine-tune Stable Diffusion and achieve impressive performance. However, they require a VAE to compress depth maps into latent space, which inevitably introduces \textit{flying pixels} at edges and details. Our model addresses this challenge by directly performing diffusion generation in the pixel space, avoiding VAE-induced artifacts. To overcome the high complexity associated with pixel-space generation, we introduce two novel designs: 1) Semantics-Prompted Diffusion Transformers (SP-DiT), which incorporate semantic representations from vision foundation models into DiT to prompt the diffusion process, thereby preserving global semantic consistency while enhancing fine-grained visual details; and 2) Cascade DiT Design that progressively increases the number of tokens to further enhance efficiency and accuracy. Our model achieves the best performance among all published generative models across five benchmarks, and significantly outperforms all other models in edge-aware point cloud evaluation.
>
---
#### [replaced 019] Unified Multimodal Chain-of-Thought Reward Model through Reinforcement Fine-Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.03318v3](http://arxiv.org/pdf/2505.03318v3)**

> **作者:** Yibin Wang; Zhimin Li; Yuhang Zang; Chunyu Wang; Qinglin Lu; Cheng Jin; Jiaqi Wang
>
> **备注:** [NeurIPS2025] Project Page: https://codegoat24.github.io/UnifiedReward/think
>
> **摘要:** Recent advances in multimodal Reward Models (RMs) have shown significant promise in delivering reward signals to align vision models with human preferences. However, current RMs are generally restricted to providing direct responses or engaging in shallow reasoning processes with limited depth, often leading to inaccurate reward signals. We posit that incorporating explicit long chains of thought (CoT) into the reward reasoning process can significantly strengthen their reliability and robustness. Furthermore, we believe that once RMs internalize CoT reasoning, their direct response accuracy can also be improved through implicit reasoning capabilities. To this end, this paper proposes UnifiedReward-Think, the first unified multimodal CoT-based reward model, capable of multi-dimensional, step-by-step long-chain reasoning for both visual understanding and generation reward tasks. Specifically, we adopt an exploration-driven reinforcement fine-tuning approach to elicit and incentivize the model's latent complex reasoning ability: (1) We first use a small amount of image generation preference data to distill the reasoning process of GPT-4o, which is then used for the model's cold start to learn the format and structure of CoT reasoning. (2) Subsequently, by leveraging the model's prior knowledge and generalization capabilities, we prepare large-scale unified multimodal preference data to elicit the model's reasoning process across various vision tasks. During this phase, correct reasoning outputs are retained for rejection sampling to refine the model (3) while incorrect predicted samples are finally used for Group Relative Policy Optimization (GRPO) based reinforcement fine-tuning, enabling the model to explore diverse reasoning paths and optimize for correct and robust solutions. Extensive experiments across various vision reward tasks demonstrate the superiority of our model.
>
---
#### [replaced 020] Simulating Automotive Radar with Lidar and Camera Inputs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.08068v2](http://arxiv.org/pdf/2503.08068v2)**

> **作者:** Peili Song; Dezhen Song; Yifan Yang; Enfan Lan; Jingtai Liu
>
> **备注:** Accepted by IROS 2025
>
> **摘要:** Low-cost millimeter automotive radar has received more and more attention due to its ability to handle adverse weather and lighting conditions in autonomous driving. However, the lack of quality datasets hinders research and development. We report a new method that is able to simulate 4D millimeter wave radar signals including pitch, yaw, range, and Doppler velocity along with radar signal strength (RSS) using camera image, light detection and ranging (lidar) point cloud, and ego-velocity. The method is based on two new neural networks: 1) DIS-Net, which estimates the spatial distribution and number of radar signals, and 2) RSS-Net, which predicts the RSS of the signal based on appearance and geometric information. We have implemented and tested our method using open datasets from 3 different models of commercial automotive radar. The experimental results show that our method can successfully generate high-fidelity radar signals. Moreover, we have trained a popular object detection neural network with data augmented by our synthesized radar. The network outperforms the counterpart trained only on raw radar data, a promising result to facilitate future radar-based research and development.
>
---
#### [replaced 021] RoboOmni: Proactive Robot Manipulation in Omni-modal Context
- **分类: cs.RO; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.23763v2](http://arxiv.org/pdf/2510.23763v2)**

> **作者:** Siyin Wang; Jinlan Fu; Feihong Liu; Xinzhe He; Huangxuan Wu; Junhao Shi; Kexin Huang; Zhaoye Fei; Jingjing Gong; Zuxuan Wu; Yugang Jiang; See-Kiong Ng; Tat-Seng Chua; Xipeng Qiu
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have driven rapid progress in Vision-Language-Action (VLA) models for robotic manipulation. Although effective in many scenarios, current approaches largely rely on explicit instructions, whereas in real-world interactions, humans rarely issue instructions directly. Effective collaboration requires robots to infer user intentions proactively. In this work, we introduce cross-modal contextual instructions, a new setting where intent is derived from spoken dialogue, environmental sounds, and visual cues rather than explicit commands. To address this new setting, we present RoboOmni, a Perceiver-Thinker-Talker-Executor framework based on end-to-end omni-modal LLMs that unifies intention recognition, interaction confirmation, and action execution. RoboOmni fuses auditory and visual signals spatiotemporally for robust intention recognition, while supporting direct speech interaction. To address the absence of training data for proactive intention recognition in robotic manipulation, we build OmniAction, comprising 140k episodes, 5k+ speakers, 2.4k event sounds, 640 backgrounds, and six contextual instruction types. Experiments in simulation and real-world settings show that RoboOmni surpasses text- and ASR-based baselines in success rate, inference speed, intention recognition, and proactive assistance.
>
---
#### [replaced 022] Classification of Driver Behaviour Using External Observation Techniques for Autonomous Vehicles
- **分类: cs.CV; cs.AI; cs.ET; cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2509.09349v2](http://arxiv.org/pdf/2509.09349v2)**

> **作者:** Ian Nell; Shane Gilroy
>
> **摘要:** Road traffic accidents remain a significant global concern, with human error, particularly distracted and impaired driving, among the leading causes. This study introduces a novel driver behaviour classification system that uses external observation techniques to detect indicators of distraction and impairment. The proposed framework employs advanced computer vision methodologies, including real-time object tracking, lateral displacement analysis, and lane position monitoring. The system identifies unsafe driving behaviours such as excessive lateral movement and erratic trajectory patterns by implementing the YOLO object detection model and custom lane estimation algorithms. Unlike systems reliant on inter-vehicular communication, this vision-based approach enables behavioural analysis of non-connected vehicles. Experimental evaluations on diverse video datasets demonstrate the framework's reliability and adaptability across varying road and environmental conditions.
>
---
#### [replaced 023] U-DECN: End-to-End Underwater Object Detection ConvNet with Improved DeNoising Training
- **分类: cs.CV; I.4**

- **链接: [http://arxiv.org/pdf/2408.05780v2](http://arxiv.org/pdf/2408.05780v2)**

> **作者:** Zhuoyan Liu; Bo Wang; Bing Wang; Ye Li
>
> **备注:** 10 pages, 6 figures, 7 tables, accepted by IEEE TGRS
>
> **摘要:** Underwater object detection has higher requirements of running speed and deployment efficiency for the detector due to its specific environmental challenges. NMS of two- or one-stage object detectors and transformer architecture of query-based end-to-end object detectors are not conducive to deployment on underwater embedded devices with limited processing power. As for the detrimental effect of underwater color cast noise, recent underwater object detectors make network architecture or training complex, which also hinders their application and deployment on unmanned underwater vehicles. In this paper, we propose the Underwater DECO with improved deNoising training (U-DECN), the query-based end-to-end object detector (with ConvNet encoder-decoder architecture) for underwater color cast noise that addresses the above problems. We integrate advanced technologies from DETR variants into DECO and design optimization methods specifically for the ConvNet architecture, including Deformable Convolution in SIM and Separate Contrastive DeNoising Forward methods. To address the underwater color cast noise issue, we propose an Underwater Color DeNoising Query method to improve the generalization of the model for the biased object feature information by different color cast noise. Our U-DECN, with ResNet-50 backbone, achieves the best 64.0 AP on DUO and the best 58.1 AP on RUOD, and 21 FPS (5 times faster than Deformable DETR and DINO 4 FPS) on NVIDIA AGX Orin by TensorRT FP16, outperforming the other state-of-the-art query-based end-to-end object detectors. The code is available at https://github.com/LEFTeyex/U-DECN.
>
---
#### [replaced 024] Single Image Estimation of Cell Migration Direction by Deep Circular Regression
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.19162v2](http://arxiv.org/pdf/2406.19162v2)**

> **作者:** Lennart Bruns; Lucas Lamparter; Milos Galic; Xiaoyi Jiang
>
> **摘要:** In this paper, we address the problem of estimating the migration direction of cells based on a single image. A solution to this problem lays the foundation for a variety of applications that were previously not possible. To our knowledge, there is only one related work that employs a classification CNN with four classes (quadrants). However, this approach does not allow for detailed directional resolution. We tackle the single image estimation problem using deep circular regression, with a particular focus on cycle-sensitive methods. On two common datasets, we achieve a mean estimation error of $\sim\!17^\circ$, representing a significant improvement over previous work, which reported estimation error of $30^\circ$ and $34^\circ$, respectively.
>
---
#### [replaced 025] Cyst-X: A Federated AI System Outperforms Clinical Guidelines to Detect Pancreatic Cancer Precursors and Reduce Unnecessary Surgery
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.22017v2](http://arxiv.org/pdf/2507.22017v2)**

> **作者:** Hongyi Pan; Gorkem Durak; Elif Keles; Deniz Seyithanoglu; Zheyuan Zhang; Alpay Medetalibeyoglu; Halil Ertugrul Aktas; Andrea Mia Bejar; Ziliang Hong; Yavuz Taktak; Gulbiz Dagoglu Kartal; Mehmet Sukru Erturk; Timurhan Cebeci; Maria Jaramillo Gonzalez; Yury Velichko; Lili Zhao; Emil Agarunov; Federica Proietto Salanitri; Concetto Spampinato; Pallavi Tiwari; Ziyue Xu; Sachin Jambawalikar; Ivo G. Schoots; Marco J. Bruno; Chenchang Huang; Candice W. Bolan; Tamas Gonda; Frank H. Miller; Rajesh N. Keswani; Michael B. Wallace; Ulas Bagci
>
> **摘要:** Pancreatic cancer is projected to be the second-deadliest cancer by 2030, making early detection critical. Intraductal papillary mucinous neoplasms (IPMNs), key cancer precursors, present a clinical dilemma, as current guidelines struggle to stratify malignancy risk, leading to unnecessary surgeries or missed diagnoses. Here, we developed Cyst-X, an AI framework for IPMN risk prediction trained on a unique, multi-center dataset of 1,461 MRI scans from 764 patients. Cyst-X achieves significantly higher accuracy (AUC = 0.82) than both the established Kyoto guidelines (AUC = 0.75) and expert radiologists, particularly in correct identification of high-risk lesions. Clinically, this translates to a 20% increase in cancer detection sensitivity (87.8% vs. 64.1%) for high-risk lesions. We demonstrate that this performance is maintained in a federated learning setting, allowing for collaborative model training without compromising patient privacy. To accelerate research in early pancreatic cancer detection, we publicly release the Cyst-X dataset and models, providing the first large-scale, multi-center MRI resource for pancreatic cyst analysis.
>
---
#### [replaced 026] HAIF-GS: Hierarchical and Induced Flow-Guided Gaussian Splatting for Dynamic Scene
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.09518v2](http://arxiv.org/pdf/2506.09518v2)**

> **作者:** Jianing Chen; Zehao Li; Yujun Cai; Hao Jiang; Chengxuan Qian; Juyuan Kang; Shuqin Gao; Honglong Zhao; Tianlu Mao; Yucheng Zhang
>
> **备注:** Accepted to NeurIPS 2025. Project page: https://echopickle.github.io/HAIF-GS.github.io/
>
> **摘要:** Reconstructing dynamic 3D scenes from monocular videos remains a fundamental challenge in 3D vision. While 3D Gaussian Splatting (3DGS) achieves real-time rendering in static settings, extending it to dynamic scenes is challenging due to the difficulty of learning structured and temporally consistent motion representations. This challenge often manifests as three limitations in existing methods: redundant Gaussian updates, insufficient motion supervision, and weak modeling of complex non-rigid deformations. These issues collectively hinder coherent and efficient dynamic reconstruction. To address these limitations, we propose HAIF-GS, a unified framework that enables structured and consistent dynamic modeling through sparse anchor-driven deformation. It first identifies motion-relevant regions via an Anchor Filter to suppress redundant updates in static areas. A self-supervised Induced Flow-Guided Deformation module induces anchor motion using multi-frame feature aggregation, eliminating the need for explicit flow labels. To further handle fine-grained deformations, a Hierarchical Anchor Propagation mechanism increases anchor resolution based on motion complexity and propagates multi-level transformations. Extensive experiments on synthetic and real-world benchmarks validate that HAIF-GS significantly outperforms prior dynamic 3DGS methods in rendering quality, temporal coherence, and reconstruction efficiency.
>
---
#### [replaced 027] Depth-Aware Super-Resolution via Distance-Adaptive Variational Formulation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.05746v2](http://arxiv.org/pdf/2509.05746v2)**

> **作者:** Tianhao Guo; Bingjie Lu; Feng Wang; Zhengyang Lu
>
> **摘要:** Single image super-resolution traditionally assumes spatially-invariant degradation models, yet real-world imaging systems exhibit complex distance-dependent effects including atmospheric scattering, depth-of-field variations, and perspective distortions. This fundamental limitation necessitates spatially-adaptive reconstruction strategies that explicitly incorporate geometric scene understanding for optimal performance. We propose a rigorous variational framework that characterizes super-resolution as a spatially-varying inverse problem, formulating the degradation operator as a pseudodifferential operator with distance-dependent spectral characteristics that enable theoretical analysis of reconstruction limits across depth ranges. Our neural architecture implements discrete gradient flow dynamics through cascaded residual blocks with depth-conditional convolution kernels, ensuring convergence to stationary points of the theoretical energy functional while incorporating learned distance-adaptive regularization terms that dynamically adjust smoothness constraints based on local geometric structure. Spectral constraints derived from atmospheric scattering theory prevent bandwidth violations and noise amplification in far-field regions, while adaptive kernel generation networks learn continuous mappings from depth to reconstruction filters. Comprehensive evaluation across five benchmark datasets demonstrates state-of-the-art performance, achieving 36.89/0.9516 and 30.54/0.8721 PSNR/SSIM at 2 and 4 scales on KITTI outdoor scenes, outperforming existing methods by 0.44dB and 0.36dB respectively. This work establishes the first theoretically-grounded distance-adaptive super-resolution framework and demonstrates significant improvements on depth-variant scenarios while maintaining competitive performance across traditional benchmarks.
>
---
#### [replaced 028] ScribbleVS: Scribble-Supervised Medical Image Segmentation via Dynamic Competitive Pseudo Label Selection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.10237v2](http://arxiv.org/pdf/2411.10237v2)**

> **作者:** Tao Wang; Xinlin Zhang; Zhenxuan Zhang; Yuanbo Zhou; Yuanbin Chen; Longxuan Zhao; Chaohui Xu; Shun Chen; Guang Yang; Tong Tong
>
> **摘要:** In clinical medicine, precise image segmentation can provide substantial support to clinicians. However, obtaining high-quality segmentation typically demands extensive pixel-level annotations, which are labor-intensive and expensive. Scribble annotations offer a more cost-effective alternative by improving labeling efficiency. Nonetheless, using such sparse supervision for training reliable medical image segmentation models remains a significant challenge. Some studies employ pseudo-labeling to enhance supervision, but these methods are susceptible to noise interference. To address these challenges, we introduce ScribbleVS, a framework designed to learn from scribble annotations. We introduce a Regional Pseudo Labels Diffusion Module to expand the scope of supervision and reduce the impact of noise present in pseudo labels. Additionally, we introduce a Dynamic Competitive Selection module for enhanced refinement in selecting pseudo labels. Experiments conducted on the ACDC, MSCMRseg, WORD, and BraTS2020 datasets demonstrate promising results, achieving segmentation precision comparable to fully supervised models. The codes of this study are available at https://github.com/ortonwang/ScribbleVS.
>
---
#### [replaced 029] WaMaIR: Image Restoration via Multiscale Wavelet Convolutions and Mamba-based Channel Modeling with Texture Enhancement
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.16765v2](http://arxiv.org/pdf/2510.16765v2)**

> **作者:** Shengyu Zhu; Congyi Fan; Fuxuan Zhang
>
> **备注:** Chinese Conference on Pattern Recognition and Computer Vision (PRCV), Oral
>
> **摘要:** Image restoration is a fundamental and challenging task in computer vision, where CNN-based frameworks demonstrate significant computational efficiency. However, previous CNN-based methods often face challenges in adequately restoring fine texture details, which are limited by the small receptive field of CNN structures and the lack of channel feature modeling. In this paper, we propose WaMaIR, which is a novel framework with a large receptive field for image perception and improves the reconstruction of texture details in restored images. Specifically, we introduce the Global Multiscale Wavelet Transform Convolutions (GMWTConvs) for expandding the receptive field to extract image features, preserving and enriching texture features in model inputs. Meanwhile, we propose the Mamba-Based Channel-Aware Module (MCAM), explicitly designed to capture long-range dependencies within feature channels, which enhancing the model sensitivity to color, edges, and texture information. Additionally, we propose Multiscale Texture Enhancement Loss (MTELoss) for image restoration to guide the model in preserving detailed texture structures effectively. Extensive experiments confirm that WaMaIR outperforms state-of-the-art methods, achieving better image restoration and efficient computational performance of the model.
>
---
#### [replaced 030] Think or Not? Selective Reasoning via Reinforcement Learning for Vision-Language Models
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16854v3](http://arxiv.org/pdf/2505.16854v3)**

> **作者:** Jiaqi Wang; Kevin Qinghong Lin; James Cheng; Mike Zheng Shou
>
> **备注:** camera ready revision
>
> **摘要:** Reinforcement Learning (RL) has proven to be an effective post-training strategy for enhancing reasoning in vision-language models (VLMs). Group Relative Policy Optimization (GRPO) is a recent prominent method that encourages models to generate complete reasoning traces before answering, leading to increased token usage and computational cost. Inspired by the human-like thinking process-where people skip reasoning for easy questions but think carefully when needed-we explore how to enable VLMs to first decide when reasoning is necessary. To realize this, we propose TON, a two-stage training strategy: (i) a supervised fine-tuning (SFT) stage with a simple yet effective 'thought dropout' operation, where reasoning traces are randomly replaced with empty thoughts. This introduces a think-or-not format that serves as a cold start for selective reasoning; (ii) a GRPO stage that enables the model to freely explore when to think or not, while maximizing task-aware outcome rewards. Experimental results show that TON can reduce the completion length by up to 90% compared to vanilla GRPO, without sacrificing performance or even improving it. Further evaluations across LLM (GSM8K), VLM (CLEVR, Super-CLEVR, GeoQA), and Agentic (AITZ) tasks-covering a range of reasoning difficulties under both 3B and 7B models-consistently reveal that the model progressively learns to bypass unnecessary reasoning steps as training advances. These findings shed light on the path toward human-like reasoning patterns in RL approaches. Our code is available at https://github.com/kokolerk/TON.
>
---
#### [replaced 031] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21710v2](http://arxiv.org/pdf/2506.21710v2)**

> **作者:** Liangyu Zhong; Fabio Rosenthal; Joachim Sicking; Fabian Hüger; Thorsten Bagdonat; Hanno Gottschalk; Leo Schwinn
>
> **备注:** Accepted by NeurIPS 2025 - main track. Project page: https://focus-mllm-vqa.github.io/
>
> **摘要:** While Multimodal Large Language Models (MLLMs) offer strong perception and reasoning capabilities for image-text input, Visual Question Answering (VQA) focusing on small image details still remains a challenge. Although visual cropping techniques seem promising, recent approaches have several limitations: the need for task-specific fine-tuning, low efficiency due to uninformed exhaustive search, or incompatibility with efficient attention implementations. We address these shortcomings by proposing a training-free visual cropping method, dubbed FOCUS, that leverages MLLM-internal representations to guide the search for the most relevant image region. This is accomplished in four steps: first, we identify the target object(s) in the VQA prompt; second, we compute an object relevance map using the key-value (KV) cache; third, we propose and rank relevant image regions based on the map; and finally, we perform the fine-grained VQA task using the top-ranked region. As a result of this informed search strategy, FOCUS achieves strong performance across four fine-grained VQA datasets and three types of MLLMs. It outperforms three popular visual cropping methods in both accuracy and efficiency, and matches the best-performing baseline, ZoomEye, while requiring 3 - 6.5 x less compute.
>
---
#### [replaced 032] DPMambaIR: All-in-One Image Restoration via Degradation-Aware Prompt State Space Model
- **分类: cs.CV; I.4.4**

- **链接: [http://arxiv.org/pdf/2504.17732v2](http://arxiv.org/pdf/2504.17732v2)**

> **作者:** Zhanwen Liu; Sai Zhou; Yuchao Dai; Yang Wang; Yisheng An; Xiangmo Zhao
>
> **摘要:** All-in-One image restoration aims to address multiple image degradation problems using a single model, offering a more practical and versatile solution compared to designing dedicated models for each degradation type. Existing approaches typically rely on Degradation-specific models or coarse-grained degradation prompts to guide image restoration. However, they lack fine-grained modeling of degradation information and face limitations in balancing multi-task conflicts. To overcome these limitations, we propose DPMambaIR, a novel All-in-One image restoration framework that introduces a fine-grained degradation extractor and a Degradation-Aware Prompt State Space Model (DP-SSM). The DP-SSM leverages the fine-grained degradation features captured by the extractor as dynamic prompts, which are then incorporated into the state space modeling process. This enhances the model's adaptability to diverse degradation types, while a complementary High-Frequency Enhancement Block (HEB) recovers local high-frequency details. Extensive experiments on a mixed dataset containing seven degradation types show that DPMambaIR achieves the best performance, with 27.69dB and 0.893 in PSNR and SSIM, respectively. These results highlight the potential and superiority of DPMambaIR as a unified solution for All-in-One image restoration.
>
---
#### [replaced 033] Functional correspondence by matrix completion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/1412.8070v2](http://arxiv.org/pdf/1412.8070v2)**

> **作者:** Artiom Kovnatsky; Michael M. Bronstein; Xavier Bresson; Pierre Vandergheynst
>
> **备注:** "Functional Correspondence by Matrix Completion" (CVPR 2015): This paper, presented at one of the world's top AI conferences, is almost entirely fabricated, and its results are not reproducible
>
> **摘要:** In this paper, we consider the problem of finding dense intrinsic correspondence between manifolds using the recently introduced functional framework. We pose the functional correspondence problem as matrix completion with manifold geometric structure and inducing functional localization with the $L_1$ norm. We discuss efficient numerical procedures for the solution of our problem. Our method compares favorably to the accuracy of state-of-the-art correspondence algorithms on non-rigid shape matching benchmarks, and is especially advantageous in settings when only scarce data is available.
>
---
#### [replaced 034] Re-ttention: Ultra Sparse Visual Generation via Attention Statistical Reshape
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.22918v4](http://arxiv.org/pdf/2505.22918v4)**

> **作者:** Ruichen Chen; Keith G. Mills; Liyao Jiang; Chao Gao; Di Niu
>
> **备注:** author comment: This version was previously removed by arXiv administrators as the submitter did not have the rights to agree to the license at the time of submission. The authors have now obtained the necessary permissions, and the paper is resubmitted accordingly
>
> **摘要:** Diffusion Transformers (DiT) have become the de-facto model for generating high-quality visual content like videos and images. A huge bottleneck is the attention mechanism where complexity scales quadratically with resolution and video length. One logical way to lessen this burden is sparse attention, where only a subset of tokens or patches are included in the calculation. However, existing techniques fail to preserve visual quality at extremely high sparsity levels and might even incur non-negligible compute overheads. To address this concern, we propose Re-ttention, which implements very high sparse attention for visual generation models by leveraging the temporal redundancy of Diffusion Models to overcome the probabilistic normalization shift within the attention mechanism. Specifically, Re-ttention reshapes attention scores based on the prior softmax distribution history in order to preserve the visual quality of the full quadratic attention at very high sparsity levels. Experimental results on T2V/T2I models such as CogVideoX and the PixArt DiTs demonstrate that Re-ttention requires as few as 3.1% of the tokens during inference, outperforming contemporary methods like FastDiTAttn, Sparse VideoGen and MInference.
>
---
#### [replaced 035] Open3D-VQA: A Benchmark for Comprehensive Spatial Reasoning with Multimodal Large Language Model in Open Space
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11094v3](http://arxiv.org/pdf/2503.11094v3)**

> **作者:** Weichen Zhang; Zile Zhou; Xin Zeng; Xuchen Liu; Jianjie Fang; Chen Gao; Yong Li; Jinqiang Cui; Xinlei Chen; Xiao-Ping Zhang
>
> **摘要:** Spatial reasoning is a fundamental capability of multimodal large language models (MLLMs), yet their performance in open aerial environments remains underexplored. In this work, we present Open3D-VQA, a novel benchmark for evaluating MLLMs' ability to reason about complex spatial relationships from an aerial perspective. The benchmark comprises 73k QA pairs spanning 7 general spatial reasoning tasks, including multiple-choice, true/false, and short-answer formats, and supports both visual and point cloud modalities. The questions are automatically generated from spatial relations extracted from both real-world and simulated aerial scenes. Evaluation on 13 popular MLLMs reveals that: 1) Models are generally better at answering questions about relative spatial relations than absolute distances, 2) 3D LLMs fail to demonstrate significant advantages over 2D LLMs, and 3) Fine-tuning solely on the simulated dataset can significantly improve the model's spatial reasoning performance in real-world scenarios. We release our benchmark, data generation pipeline, and evaluation toolkit to support further research: https://github.com/EmbodiedCity/Open3D-VQA.code.
>
---
#### [replaced 036] PSScreen V2: Partially Supervised Multiple Retinal Disease Screening
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.22589v2](http://arxiv.org/pdf/2510.22589v2)**

> **作者:** Boyi Zheng; Yalin Zheng; Hrvoje Bogunović; Qing Liu
>
> **摘要:** In this work, we propose PSScreen V2, a partially supervised self-training framework for multiple retinal disease screening. Unlike previous methods that rely on fully labelled or single-domain datasets, PSScreen V2 is designed to learn from multiple partially labelled datasets with different distributions, addressing both label absence and domain shift challenges. To this end, PSScreen V2 adopts a three-branch architecture with one teacher and two student networks. The teacher branch generates pseudo labels from weakly augmented images to address missing labels, while the two student branches introduce novel feature augmentation strategies: Low-Frequency Dropout (LF-Dropout), which enhances domain robustness by randomly discarding domain-related low-frequency components, and Low-Frequency Uncertainty (LF-Uncert), which estimates uncertain domain variability via adversarially learned Gaussian perturbations of low-frequency statistics. Extensive experiments on multiple in-domain and out-of-domain fundus datasets demonstrate that PSScreen V2 achieves state-of-the-art performance and superior domain generalization ability. Furthermore, compatibility tests with diverse backbones, including the vision foundation model DINOv2, as well as evaluations on chest X-ray datasets, highlight the universality and adaptability of the proposed framework. The codes are available at https://github.com/boyiZheng99/PSScreen_V2.
>
---
#### [replaced 037] Caption-Driven Explainability: Probing CNNs for Bias via CLIP
- **分类: cs.CV; eess.IV; I.2.6; I.2.8; I.2.10; I.4.8**

- **链接: [http://arxiv.org/pdf/2510.22035v3](http://arxiv.org/pdf/2510.22035v3)**

> **作者:** Patrick Koller; Amil V. Dravid; Guido M. Schuster; Aggelos K. Katsaggelos
>
> **备注:** Accepted and presented at the IEEE ICIP 2025 Satellite Workshop "Generative AI for World Simulations and Communications & Celebrating 40 Years of Excellence in Education: Honoring Professor Aggelos Katsaggelos", Anchorage, Alaska, USA, September 14, 2025. Camera-ready preprint; the official IEEE Xplore publication will follow. Code: https://github.com/patch0816/caption-driven-xai
>
> **摘要:** Robustness has become one of the most critical problems in machine learning (ML). The science of interpreting ML models to understand their behavior and improve their robustness is referred to as explainable artificial intelligence (XAI). One of the state-of-the-art XAI methods for computer vision problems is to generate saliency maps. A saliency map highlights the pixel space of an image that excites the ML model the most. However, this property could be misleading if spurious and salient features are present in overlapping pixel spaces. In this paper, we propose a caption-based XAI method, which integrates a standalone model to be explained into the contrastive language-image pre-training (CLIP) model using a novel network surgery approach. The resulting caption-based XAI model identifies the dominant concept that contributes the most to the models prediction. This explanation minimizes the risk of the standalone model falling for a covariate shift and contributes significantly towards developing robust ML models. Our code is available at https://github.com/patch0816/caption-driven-xai
>
---
#### [replaced 038] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.06677v2](http://arxiv.org/pdf/2506.06677v2)**

> **作者:** Songhao Han; Boxiang Qiu; Yue Liao; Siyuan Huang; Chen Gao; Shuicheng Yan; Si Liu
>
> **备注:** 25 pages, 18 figures, Accepted by NeurIPS 2025
>
> **摘要:** Recent advances in vision-language models (VLMs) have enabled instruction-conditioned robotic systems with improved generalization. However, most existing work focuses on reactive System 1 policies, underutilizing VLMs' strengths in semantic reasoning and long-horizon planning. These System 2 capabilities-characterized by deliberative, goal-directed thinking-remain under explored due to the limited temporal scale and structural complexity of current benchmarks. To address this gap, we introduce RoboCerebra, a benchmark for evaluating high-level reasoning in long-horizon robotic manipulation. RoboCerebra includes: (1) a large-scale simulation dataset with extended task horizons and diverse subtask sequences in household environments; (2) a hierarchical framework combining a high-level VLM planner with a low-level vision-language-action (VLA) controller; and (3) an evaluation protocol targeting planning, reflection, and memory through structured System 1-System 2 interaction. The dataset is constructed via a top-down pipeline, where GPT generates task instructions and decomposes them into subtask sequences. Human operators execute the subtasks in simulation, yielding high-quality trajectories with dynamic object variations. Compared to prior benchmarks, RoboCerebra features significantly longer action sequences and denser annotations. We further benchmark state-of-the-art VLMs as System 2 modules and analyze their performance across key cognitive dimensions, advancing the development of more capable and generalizable robotic planners.
>
---
#### [replaced 039] LightBagel: A Light-weighted, Double Fusion Framework for Unified Multimodal Understanding and Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.22946v2](http://arxiv.org/pdf/2510.22946v2)**

> **作者:** Zeyu Wang; Zilong Chen; Chenhui Gou; Feng Li; Chaorui Deng; Deyao Zhu; Kunchang Li; Weihao Yu; Haoqin Tu; Haoqi Fan; Cihang Xie
>
> **备注:** Withdrawn because the submission was premature and not agreed by all parties in collaboration
>
> **摘要:** Unified multimodal models have recently shown remarkable gains in both capability and versatility, yet most leading systems are still trained from scratch and require substantial computational resources. In this paper, we show that competitive performance can be obtained far more efficiently by strategically fusing publicly available models specialized for either generation or understanding. Our key design is to retain the original blocks while additionally interleaving multimodal self-attention blocks throughout the networks. This double fusion mechanism (1) effectively enables rich multi-modal fusion while largely preserving the original strengths of the base models, and (2) catalyzes synergistic fusion of high-level semantic representations from the understanding encoder with low-level spatial signals from the generation encoder. By training with only ~ 35B tokens, this approach achieves strong results across multiple benchmarks: 0.91 on GenEval for compositional text-to-image generation, 82.16 on DPG-Bench for complex text-to-image generation, 6.06 on GEditBench, and 3.77 on ImgEdit-Bench for image editing. By fully releasing the entire suite of code, model weights, and datasets, we hope to support future research on unified multimodal modeling.
>
---
#### [replaced 040] Vision-Centric 4D Occupancy Forecasting and Planning via Implicit Residual World Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.16729v2](http://arxiv.org/pdf/2510.16729v2)**

> **作者:** Jianbiao Mei; Yu Yang; Xuemeng Yang; Licheng Wen; Jiajun Lv; Botian Shi; Yong Liu
>
> **摘要:** End-to-end autonomous driving systems increasingly rely on vision-centric world models to understand and predict their environment. However, a common ineffectiveness in these models is the full reconstruction of future scenes, which expends significant capacity on redundantly modeling static backgrounds. To address this, we propose IR-WM, an Implicit Residual World Model that focuses on modeling the current state and evolution of the world. IR-WM first establishes a robust bird's-eye-view representation of the current state from the visual observation. It then leverages the BEV features from the previous timestep as a strong temporal prior and predicts only the "residual", i.e., the changes conditioned on the ego-vehicle's actions and scene context. To alleviate error accumulation over time, we further apply an alignment module to calibrate semantic and dynamic misalignments. Moreover, we investigate different forecasting-planning coupling schemes and demonstrate that the implicit future state generated by world models substantially improves planning accuracy. On the nuScenes benchmark, IR-WM achieves top performance in both 4D occupancy forecasting and trajectory planning.
>
---
#### [replaced 041] NoisyGRPO: Incentivizing Multimodal CoT Reasoning via Noise Injection and Bayesian Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.21122v2](http://arxiv.org/pdf/2510.21122v2)**

> **作者:** Longtian Qiu; Shan Ning; Jiaxuan Sun; Xuming He
>
> **备注:** Accepted by Neurips2025, Project page at at https://artanic30.github.io/project_pages/NoisyGRPO/
>
> **摘要:** Reinforcement learning (RL) has shown promise in enhancing the general Chain-of-Thought (CoT) reasoning capabilities of multimodal large language models (MLLMs). However, when applied to improve general CoT reasoning, existing RL frameworks often struggle to generalize beyond the training distribution. To address this, we propose NoisyGRPO, a systematic multimodal RL framework that introduces controllable noise into visual inputs for enhanced exploration and explicitly models the advantage estimation process via a Bayesian framework. Specifically, NoisyGRPO improves RL training by: (1) Noise-Injected Exploration Policy: Perturbing visual inputs with Gaussian noise to encourage exploration across a wider range of visual scenarios; and (2) Bayesian Advantage Estimation: Formulating advantage estimation as a principled Bayesian inference problem, where the injected noise level serves as a prior and the observed trajectory reward as the likelihood. This Bayesian modeling fuses both sources of information to compute a robust posterior estimate of trajectory advantage, effectively guiding MLLMs to prefer visually grounded trajectories over noisy ones. Experiments on standard CoT quality, general capability, and hallucination benchmarks demonstrate that NoisyGRPO substantially improves generalization and robustness, especially in RL settings with small-scale MLLMs such as Qwen2.5-VL 3B. The project page is available at https://artanic30.github.io/project_pages/NoisyGRPO/.
>
---
#### [replaced 042] HF-VTON: High-Fidelity Virtual Try-On via Consistent Geometric and Semantic Alignment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19638v3](http://arxiv.org/pdf/2505.19638v3)**

> **作者:** Ming Meng; Qi Dong; Jiajie Li; Zhe Zhu; Xingyu Wang; Zhaoxin Fan; Wei Zhao; Wenjun Wu
>
> **备注:** After the publication of the paper, we discovered some significant errors/omissions that need to be corrected and improved
>
> **摘要:** Virtual try-on technology has become increasingly important in the fashion and retail industries, enabling the generation of high-fidelity garment images that adapt seamlessly to target human models. While existing methods have achieved notable progress, they still face significant challenges in maintaining consistency across different poses. Specifically, geometric distortions lead to a lack of spatial consistency, mismatches in garment structure and texture across poses result in semantic inconsistency, and the loss or distortion of fine-grained details diminishes visual fidelity. To address these challenges, we propose HF-VTON, a novel framework that ensures high-fidelity virtual try-on performance across diverse poses. HF-VTON consists of three key modules: (1) the Appearance-Preserving Warp Alignment Module (APWAM), which aligns garments to human poses, addressing geometric deformations and ensuring spatial consistency; (2) the Semantic Representation and Comprehension Module (SRCM), which captures fine-grained garment attributes and multi-pose data to enhance semantic representation, maintaining structural, textural, and pattern consistency; and (3) the Multimodal Prior-Guided Appearance Generation Module (MPAGM), which integrates multimodal features and prior knowledge from pre-trained models to optimize appearance generation, ensuring both semantic and geometric consistency. Additionally, to overcome data limitations in existing benchmarks, we introduce the SAMP-VTONS dataset, featuring multi-pose pairs and rich textual annotations for a more comprehensive evaluation. Experimental results demonstrate that HF-VTON outperforms state-of-the-art methods on both VITON-HD and SAMP-VTONS, excelling in visual fidelity, semantic consistency, and detail preservation.
>
---
#### [replaced 043] GENRE-CMR: Generalizable Deep Learning for Diverse Multi-Domain Cardiac MRI Reconstruction
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.20600v2](http://arxiv.org/pdf/2508.20600v2)**

> **作者:** Kian Anvari Hamedani; Narges Razizadeh; Shahabedin Nabavi; Mohsen Ebrahimi Moghaddam
>
> **摘要:** Accelerated Cardiovascular Magnetic Resonance (CMR) image reconstruction remains a critical challenge due to the trade-off between scan time and image quality, particularly when generalizing across diverse acquisition settings. We propose GENRE-CMR, a generative adversarial network (GAN)-based architecture employing a residual deep unrolled reconstruction framework to enhance reconstruction fidelity and generalization. The architecture unrolls iterative optimization into a cascade of convolutional subnetworks, enriched with residual connections to enable progressive feature propagation from shallow to deeper stages. To further improve performance, we integrate two loss functions: (1) an Edge-Aware Region (EAR) loss, which guides the network to focus on structurally informative regions and helps prevent common reconstruction blurriness; and (2) a Statistical Distribution Alignment (SDA) loss, which regularizes the feature space across diverse data distributions via a symmetric KL divergence formulation. Extensive experiments confirm that GENRE-CMR surpasses state-of-the-art methods on training and unseen data, achieving 0.9552 SSIM and 38.90 dB PSNR on unseen distributions across various acceleration factors and sampling trajectories. Ablation studies confirm the contribution of each proposed component to reconstruction quality and generalization. Our framework presents a unified and robust solution for high-quality CMR reconstruction, paving the way for clinically adaptable deployment across heterogeneous acquisition protocols.
>
---
#### [replaced 044] Evaluation of Safety Cognition Capability in Vision-Language Models for Autonomous Driving
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.06497v3](http://arxiv.org/pdf/2503.06497v3)**

> **作者:** Enming Zhang; Peizhe Gong; Xingyuan Dai; Min Huang; Yisheng Lv; Qinghai Miao
>
> **摘要:** Ensuring the safety of vision-language models (VLMs) in autonomous driving systems is of paramount importance, yet existing research has largely focused on conventional benchmarks rather than safety-critical evaluation. In this work, we present SCD-Bench (Safety Cognition Driving Benchmark) a novel framework specifically designed to assess the safety cognition capabilities of VLMs within interactive driving scenarios. To address the scalability challenge of data annotation, we introduce ADA (Autonomous Driving Annotation), a semi-automated labeling system, further refined through expert review by professionals with domain-specific knowledge in autonomous driving. To facilitate scalable and consistent evaluation, we also propose an automated assessment pipeline leveraging large language models, which demonstrates over 98% agreement with human expert judgments. In addressing the broader challenge of aligning VLMs with safety cognition in driving environments, we construct SCD-Training, the first large-scale dataset tailored for this task, comprising 324.35K high-quality samples. Through extensive experiments, we show that models trained on SCD-Training exhibit marked improvements not only on SCD-Bench, but also on general and domain-specific benchmarks, offering a new perspective on enhancing safety-aware interactions in vision-language systems for autonomous driving.
>
---
#### [replaced 045] XY-Cut++: Advanced Layout Ordering via Hierarchical Mask Mechanism on a Novel Benchmark
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2504.10258v2](http://arxiv.org/pdf/2504.10258v2)**

> **作者:** Shuai Liu; Youmeng Li; Jizeng Wei
>
> **摘要:** Document Reading Order Recovery is a fundamental task in document image understanding, playing a pivotal role in enhancing Retrieval-Augmented Generation (RAG) and serving as a critical preprocessing step for large language models (LLMs). Existing methods often struggle with complex layouts(e.g., multi-column newspapers), high-overhead interactions between cross-modal elements (visual regions and textual semantics), and a lack of robust evaluation benchmarks. We introduce XY-Cut++, an advanced layout ordering method that integrates pre-mask processing, multi-granularity segmentation, and cross-modal matching to address these challenges. Our method significantly enhances layout ordering accuracy compared to traditional XY-Cut techniques. Specifically, XY-Cut++ achieves state-of-the-art performance (98.8 BLEU overall) while maintaining simplicity and efficiency. It outperforms existing baselines by up to 24\% and demonstrates consistent accuracy across simple and complex layouts on the newly introduced DocBench-100 dataset. This advancement establishes a reliable foundation for document structure recovery, setting a new standard for layout ordering tasks and facilitating more effective RAG and LLM preprocessing.
>
---
#### [replaced 046] Why Foundation Models in Pathology Are Failing
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.23807v2](http://arxiv.org/pdf/2510.23807v2)**

> **作者:** Hamid R. Tizhoosh
>
> **摘要:** In non-medical domains, foundation models (FMs) have revolutionized computer vision and language processing through large-scale self-supervised and multimodal learning. Consequently, their rapid adoption in computational pathology was expected to deliver comparable breakthroughs in cancer diagnosis, prognostication, and multimodal retrieval. However, recent systematic evaluations reveal fundamental weaknesses: low diagnostic accuracy, poor robustness, geometric instability, heavy computational demands, and concerning safety vulnerabilities. This short paper examines these shortcomings and argues that they stem from deeper conceptual mismatches between the assumptions underlying generic foundation modeling in mainstream AI and the intrinsic complexity of human tissue. Seven interrelated causes are identified: biological complexity, ineffective self-supervision, overgeneralization, excessive architectural complexity, lack of domain-specific innovation, insufficient data, and a fundamental design flaw related to tissue patch size. These findings suggest that current pathology foundation models remain conceptually misaligned with the nature of tissue morphology and call for a fundamental rethinking of the paradigm itself.
>
---
#### [replaced 047] MambaCAFU: Hybrid Multi-Scale and Multi-Attention Model with Mamba-Based Fusion for Medical Image Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.03786v2](http://arxiv.org/pdf/2510.03786v2)**

> **作者:** T-Mai Bui; Fares Bougourzi; Fadi Dornaika; Vinh Truong Hoang
>
> **摘要:** In recent years, deep learning has shown near-expert performance in segmenting complex medical tissues and tumors. However, existing models are often task-specific, with performance varying across modalities and anatomical regions. Balancing model complexity and performance remains challenging, particularly in clinical settings where both accuracy and efficiency are critical. To address these issues, we propose a hybrid segmentation architecture featuring a three-branch encoder that integrates CNNs, Transformers, and a Mamba-based Attention Fusion (MAF) mechanism to capture local, global, and long-range dependencies. A multi-scale attention-based CNN decoder reconstructs fine-grained segmentation maps while preserving contextual consistency. Additionally, a co-attention gate enhances feature selection by emphasizing relevant spatial and semantic information across scales during both encoding and decoding, improving feature interaction and cross-scale communication. Extensive experiments on multiple benchmark datasets show that our approach outperforms state-of-the-art methods in accuracy and generalization, while maintaining comparable computational complexity. By effectively balancing efficiency and effectiveness, our architecture offers a practical and scalable solution for diverse medical imaging tasks. Source code and trained models will be publicly released upon acceptance to support reproducibility and further research.
>
---
#### [replaced 048] Activation Matching for Explanation Generation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.23051v2](http://arxiv.org/pdf/2509.23051v2)**

> **作者:** Pirzada Suhail; Aditya Anand; Amit Sethi
>
> **摘要:** In this paper we introduce an activation-matching--based approach to generate minimal, faithful explanations for the decision-making of a pretrained classifier on any given image. Given an input image \(x\) and a frozen model \(f\), we train a lightweight autoencoder to output a binary mask \(m\) such that the explanation \(e = m \odot x\) preserves both the model's prediction and the intermediate activations of \(x\). Our objective combines: (i) multi-layer activation matching with KL divergence to align distributions and cross-entropy to retain the top-1 label for both the image and the explanation; (ii) mask priors -- L1 area for minimality, a binarization penalty for crisp 0/1 masks, and total variation for compactness; and (iii) abductive constraints for faithfulness and necessity. Together, these objectives yield small, human-interpretable masks that retain classifier behavior while discarding irrelevant input regions, providing practical and faithful minimalist explanations for the decision making of the underlying model.
>
---
#### [replaced 049] Multimodal Recurrent Ensembles for Predicting Brain Responses to Naturalistic Movies (Algonauts 2025)
- **分类: q-bio.NC; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.17897v4](http://arxiv.org/pdf/2507.17897v4)**

> **作者:** Semih Eren; Deniz Kucukahmetler; Nico Scherf
>
> **备注:** 8 pages, 2 figures, 1 table. Invited report, CCN 2025 Algonauts Project session (3rd-place team). Code: https://github.com/erensemih/Algonauts2025_ModalityRNN v3: Added equal contribution footnote to author list. Corrected reference list
>
> **摘要:** Accurately predicting distributed cortical responses to naturalistic stimuli requires models that integrate visual, auditory and semantic information over time. We present a hierarchical multimodal recurrent ensemble that maps pretrained video, audio, and language embeddings to fMRI time series recorded while four subjects watched almost 80 hours of movies provided by the Algonauts 2025 challenge. Modality-specific bidirectional RNNs encode temporal dynamics; their hidden states are fused and passed to a second recurrent layer, and lightweight subject-specific heads output responses for 1000 cortical parcels. Training relies on a composite MSE-correlation loss and a curriculum that gradually shifts emphasis from early sensory to late association regions. Averaging 100 model variants further boosts robustness. The resulting system ranked third on the competition leaderboard, achieving an overall Pearson r = 0.2094 and the highest single-parcel peak score (mean r = 0.63) among all participants, with particularly strong gains for the most challenging subject (Subject 5). The approach establishes a simple, extensible baseline for future multimodal brain-encoding benchmarks.
>
---
#### [replaced 050] Quantizing Space and Time: Fusing Time Series and Images for Earth Observation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.23118v3](http://arxiv.org/pdf/2510.23118v3)**

> **作者:** Gianfranco Basile; Johannes Jakubik; Benedikt Blumenstiel; Thomas Brunschwiler; Juan Bernabe Moreno
>
> **摘要:** We propose a task-agnostic framework for multimodal fusion of time series and single timestamp images, enabling cross-modal generation and robust downstream performance. Our approach explores deterministic and learned strategies for time series quantization and then leverages a masked correlation learning objective, aligning discrete image and time series tokens in a unified representation space. Instantiated in the Earth observation domain, the pretrained model generates consistent global temperature profiles from satellite imagery and is validated through counterfactual experiments. Across downstream tasks, our task-agnostic pretraining outperforms task-specific fusion by 6% in R^2 and 2% in RMSE on average, and exceeds baseline methods by 50% in R^2 and 12% in RMSE. Finally, we analyze gradient sensitivity across modalities, providing insights into model robustness. Code, data, and weights will be released under a permissive license.
>
---
#### [replaced 051] SignMouth: Leveraging Mouthing Cues for Sign Language Translation by Multimodal Contrastive Fusion
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.10266v2](http://arxiv.org/pdf/2509.10266v2)**

> **作者:** Wenfang Wu; Tingting Yuan; Yupeng Li; Daling Wang; Xiaoming Fu
>
> **摘要:** Sign language translation (SLT) aims to translate natural language from sign language videos, serving as a vital bridge for inclusive communication. While recent advances leverage powerful visual backbones and large language models, most approaches mainly focus on manual signals (hand gestures) and tend to overlook non-manual cues like mouthing. In fact, mouthing conveys essential linguistic information in sign languages and plays a crucial role in disambiguating visually similar signs. In this paper, we propose SignClip, a novel framework to improve the accuracy of sign language translation. It fuses manual and non-manual cues, specifically spatial gesture and lip movement features. Besides, SignClip introduces a hierarchical contrastive learning framework with multi-level alignment objectives, ensuring semantic consistency across sign-lip and visual-text modalities. Extensive experiments on two benchmark datasets, PHOENIX14T and How2Sign, demonstrate the superiority of our approach. For example, on PHOENIX14T, in the Gloss-free setting, SignClip surpasses the previous state-of-the-art model SpaMo, improving BLEU-4 from 24.32 to 24.71, and ROUGE from 46.57 to 48.38.
>
---
#### [replaced 052] DGTRSD & DGTRS-CLIP: A Dual-Granularity Remote Sensing Image-Text Dataset and Vision Language Foundation Model for Alignment
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.19311v2](http://arxiv.org/pdf/2503.19311v2)**

> **作者:** Weizhi Chen; Yupeng Deng; Jin Wei; Jingbo Chen; Jiansheng Chen; Yuman Feng; Zhihao Xi; Diyou Liu; Kai Li; Yu Meng
>
> **摘要:** Vision Language Foundation Models based on CLIP architecture for remote sensing primarily rely on short text captions, which often result in incomplete semantic representations. Although longer captions convey richer information, existing models struggle to process them effectively because of limited text-encoding capacity, and there remains a shortage of resources that align remote sensing images with both short text and long text captions. To address this gap, we introduce DGTRSD, a dual-granularity remote sensing image-text dataset, where each image is paired with both a short text caption and a long text description, providing a solid foundation for dual-granularity semantic modeling. Based on this, we further propose DGTRS-CLIP, a dual-granularity curriculum learning framework that combines short text and long text supervision to achieve dual-granularity semantic alignment. Extensive experiments on four typical zero-shot tasks: long text cross-modal retrieval, short text cross-modal retrieval, image classification, and semantic localization demonstrate that DGTRS-CLIP consistently outperforms existing methods across all tasks. The code has been open-sourced and is available at https://github.com/MitsuiChen14/DGTRS.
>
---
#### [replaced 053] L2RSI: Cross-view LiDAR-based Place Recognition for Large-scale Urban Scenes via Remote Sensing Imagery
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11245v4](http://arxiv.org/pdf/2503.11245v4)**

> **作者:** Ziwei Shi; Xiaoran Zhang; Wenjing Xu; Yan Xia; Yu Zang; Siqi Shen; Cheng Wang
>
> **备注:** 17 pages, 7 figures, NeurIPS 2025
>
> **摘要:** We tackle the challenge of LiDAR-based place recognition, which traditionally depends on costly and time-consuming prior 3D maps. To overcome this, we first construct LiRSI-XA dataset, which encompasses approximately $110,000$ remote sensing submaps and $13,000$ LiDAR point cloud submaps captured in urban scenes, and propose a novel method, L2RSI, for cross-view LiDAR place recognition using high-resolution Remote Sensing Imagery. This approach enables large-scale localization capabilities at a reduced cost by leveraging readily available overhead images as map proxies. L2RSI addresses the dual challenges of cross-view and cross-modal place recognition by learning feature alignment between point cloud submaps and remote sensing submaps in the semantic domain. Additionally, we introduce a novel probability propagation method based on particle estimation to refine position predictions, effectively leveraging temporal and spatial information. This approach enables large-scale retrieval and cross-scene generalization without fine-tuning. Extensive experiments on LiRSI-XA demonstrate that, within a $100km^2$ retrieval range, L2RSI accurately localizes $83.27\%$ of point cloud submaps within a $30m$ radius for top-$1$ retrieved location. Our project page is publicly available at https://shizw695.github.io/L2RSI/.
>
---
#### [replaced 054] InstDrive: Instance-Aware 3D Gaussian Splatting for Driving Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.12015v2](http://arxiv.org/pdf/2508.12015v2)**

> **作者:** Hongyuan Liu; Haochen Yu; Bochao Zou; Jianfei Jiang; Qiankun Liu; Jiansheng Chen; Huimin Ma
>
> **摘要:** Reconstructing dynamic driving scenes from dashcam videos has attracted increasing attention due to its significance in autonomous driving and scene understanding. While recent advances have made impressive progress, most methods still unify all background elements into a single representation, hindering both instance-level understanding and flexible scene editing. Some approaches attempt to lift 2D segmentation into 3D space, but often rely on pre-processed instance IDs or complex pipelines to map continuous features to discrete identities. Moreover, these methods are typically designed for indoor scenes with rich viewpoints, making them less applicable to outdoor driving scenarios. In this paper, we present InstDrive, an instance-aware 3D Gaussian Splatting framework tailored for the interactive reconstruction of dynamic driving scene. We use masks generated by SAM as pseudo ground-truth to guide 2D feature learning via contrastive loss and pseudo-supervised objectives. At the 3D level, we introduce regularization to implicitly encode instance identities and enforce consistency through a voxel-based loss. A lightweight static codebook further bridges continuous features and discrete identities without requiring data pre-processing or complex optimization. Quantitative and qualitative experiments demonstrate the effectiveness of InstDrive, and to the best of our knowledge, it is the first framework to achieve 3D instance segmentation in dynamic, open-world driving scenes.More visualizations are available at our project page.
>
---
