# 计算机视觉 cs.CV

- **最新发布 98 篇**

- **更新 49 篇**

## 最新发布

#### [new 001] An Exploratory Study on Abstract Images and Visual Representations Learned from Them
- **分类: cs.CV**

- **简介: 论文研究抽象图像与传统图像在视觉任务中的表现差异，提出HAID数据集，分析抽象图像的语义表达能力。任务包括分类、分割和检测，旨在探索抽象图像是否能有效传达视觉信息。**

- **链接: [http://arxiv.org/pdf/2509.14149v1](http://arxiv.org/pdf/2509.14149v1)**

> **作者:** Haotian Li; Jianbo Jiao
>
> **备注:** Accepted to BMVC 2025
>
> **摘要:** Imagine living in a world composed solely of primitive shapes, could you still recognise familiar objects? Recent studies have shown that abstract images-constructed by primitive shapes-can indeed convey visual semantic information to deep learning models. However, representations obtained from such images often fall short compared to those derived from traditional raster images. In this paper, we study the reasons behind this performance gap and investigate how much high-level semantic content can be captured at different abstraction levels. To this end, we introduce the Hierarchical Abstraction Image Dataset (HAID), a novel data collection that comprises abstract images generated from normal raster images at multiple levels of abstraction. We then train and evaluate conventional vision systems on HAID across various tasks including classification, segmentation, and object detection, providing a comprehensive study between rasterised and abstract image representations. We also discuss if the abstract image can be considered as a potentially effective format for conveying visual semantic information and contributing to vision tasks.
>
---
#### [new 002] VocSegMRI: Multimodal Learning for Precise Vocal Tract Segmentation in Real-time MRI
- **分类: cs.CV**

- **简介: 该论文提出VocSegMRI框架，解决实时MRI中声门结构分割精度问题。通过融合视频、音频和音位信息，利用跨注意力机制和对比学习提升分割性能，实验证明其在 Dice 和 HD_95 指标上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.13767v1](http://arxiv.org/pdf/2509.13767v1)**

> **作者:** Daiqi Liu; Tomás Arias-Vergara; Johannes Enk; Fangxu Xing; Maureen Stone; Jerry L. Prince; Jana Hutter; Andreas Maier; Jonghye Woo; Paula Andrea Pérez-Toro
>
> **备注:** Preprint submitted to ICASSP
>
> **摘要:** Accurately segmenting articulatory structures in real-time magnetic resonance imaging (rtMRI) remains challenging, as most existing methods rely almost entirely on visual cues. Yet synchronized acoustic and phonological signals provide complementary context that can enrich visual information and improve precision. In this paper, we introduce VocSegMRI, a multimodal framework that integrates video, audio, and phonological inputs through cross-attention fusion for dynamic feature alignment. To further enhance cross-modal representation, we incorporate a contrastive learning objective that improves segmentation performance even when the audio modality is unavailable at inference. Evaluated on a sub-set of USC-75 rtMRI dataset, our approach achieves state-of-the-art performance, with a Dice score of 0.95 and a 95th percentile Hausdorff Distance (HD_95) of 4.20 mm, outperforming both unimodal and multimodal baselines. Ablation studies confirm the contributions of cross-attention and contrastive learning to segmentation precision and robustness. These results highlight the value of integrative multimodal modeling for accurate vocal tract analysis.
>
---
#### [new 003] Parking Space Ground Truth Test Automation by Artificial Intelligence Using Convolutional Neural Networks
- **分类: cs.CV; 68U99; J.2**

- **简介: 该论文研究利用卷积神经网络实现停车空间真实数据测试自动化，以优化基于云计算的实时路边停车服务。通过机器学习方法减少人工工作，提升测试效率，最高节省99.58%的人力时间。**

- **链接: [http://arxiv.org/pdf/2509.13366v1](http://arxiv.org/pdf/2509.13366v1)**

> **作者:** Tony Rohe; Martin Margreiter; Markus Moertl
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** This research is part of a study of a real-time, cloud-based on-street parking service using crowd-sourced in-vehicle fleet data. The service provides real-time information about available parking spots by classifying crowd-sourced detections observed via ultrasonic sensors. The goal of this research is to optimize the current parking service quality by analyzing the automation of the existing test process for ground truth tests. Therefore, methods from the field of machine learning, especially image pattern recognition, are applied to enrich the database and substitute human engineering work in major areas of the analysis process. After an introduction into the related areas of machine learning, this paper explains the methods and implementations made to achieve a high level of automation, applying convolutional neural networks. Finally, predefined metrics present the performance level achieved, showing a time reduction of human resources up to 99.58 %. The overall improvements are discussed, summarized, and followed by an outlook for future development and potential application of the analysis automation tool.
>
---
#### [new 004] Improving Generalized Visual Grounding with Instance-aware Joint Learning
- **分类: cs.CV**

- **简介: 论文提出InstanceVG框架，解决广义视觉定位任务中GREC与GRES的独立训练问题，通过实例感知联合学习实现框与掩码的一致预测，提升多粒度目标识别性能。**

- **链接: [http://arxiv.org/pdf/2509.13747v1](http://arxiv.org/pdf/2509.13747v1)**

> **作者:** Ming Dai; Wenxuan Cheng; Jiang-Jiang Liu; Lingfeng Yang; Zhenhua Feng; Wankou Yang; Jingdong Wang
>
> **备注:** Accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) in September 2025
>
> **摘要:** Generalized visual grounding tasks, including Generalized Referring Expression Comprehension (GREC) and Segmentation (GRES), extend the classical visual grounding paradigm by accommodating multi-target and non-target scenarios. Specifically, GREC focuses on accurately identifying all referential objects at the coarse bounding box level, while GRES aims for achieve fine-grained pixel-level perception. However, existing approaches typically treat these tasks independently, overlooking the benefits of jointly training GREC and GRES to ensure consistent multi-granularity predictions and streamline the overall process. Moreover, current methods often treat GRES as a semantic segmentation task, neglecting the crucial role of instance-aware capabilities and the necessity of ensuring consistent predictions between instance-level boxes and masks. To address these limitations, we propose InstanceVG, a multi-task generalized visual grounding framework equipped with instance-aware capabilities, which leverages instance queries to unify the joint and consistency predictions of instance-level boxes and masks. To the best of our knowledge, InstanceVG is the first framework to simultaneously tackle both GREC and GRES while incorporating instance-aware capabilities into generalized visual grounding. To instantiate the framework, we assign each instance query a prior reference point, which also serves as an additional basis for target matching. This design facilitates consistent predictions of points, boxes, and masks for the same instance. Extensive experiments obtained on ten datasets across four tasks demonstrate that InstanceVG achieves state-of-the-art performance, significantly surpassing the existing methods in various evaluation metrics. The code and model will be publicly available at https://github.com/Dmmm1997/InstanceVG.
>
---
#### [new 005] Wan-Animate: Unified Character Animation and Replacement with Holistic Replication
- **分类: cs.CV**

- **简介: 该论文提出Wan-Animate框架，用于统一角色动画生成与替换任务。通过复制参考视频中的表情和动作，生成高保真角色视频，并能无缝替换原视频角色。采用骨骼信号与面部特征提取技术，结合光照模块提升环境融合效果，实现高质量可控动画生成。**

- **链接: [http://arxiv.org/pdf/2509.14055v1](http://arxiv.org/pdf/2509.14055v1)**

> **作者:** Gang Cheng; Xin Gao; Li Hu; Siqi Hu; Mingyang Huang; Chaonan Ji; Ju Li; Dechao Meng; Jinwei Qi; Penchong Qiao; Zhen Shen; Yafei Song; Ke Sun; Linrui Tian; Feng Wang; Guangyuan Wang; Qi Wang; Zhongjian Wang; Jiayu Xiao; Sheng Xu; Bang Zhang; Peng Zhang; Xindi Zhang; Zhe Zhang; Jingren Zhou; Lian Zhuo
>
> **备注:** Project Page: https://humanaigc.github.io/wan-animate/
>
> **摘要:** We introduce Wan-Animate, a unified framework for character animation and replacement. Given a character image and a reference video, Wan-Animate can animate the character by precisely replicating the expressions and movements of the character in the video to generate high-fidelity character videos. Alternatively, it can integrate the animated character into the reference video to replace the original character, replicating the scene's lighting and color tone to achieve seamless environmental integration. Wan-Animate is built upon the Wan model. To adapt it for character animation tasks, we employ a modified input paradigm to differentiate between reference conditions and regions for generation. This design unifies multiple tasks into a common symbolic representation. We use spatially-aligned skeleton signals to replicate body motion and implicit facial features extracted from source images to reenact expressions, enabling the generation of character videos with high controllability and expressiveness. Furthermore, to enhance environmental integration during character replacement, we develop an auxiliary Relighting LoRA. This module preserves the character's appearance consistency while applying the appropriate environmental lighting and color tone. Experimental results demonstrate that Wan-Animate achieves state-of-the-art performance. We are committed to open-sourcing the model weights and its source code.
>
---
#### [new 006] Diving into Mitigating Hallucinations from a Vision Perspective for Large Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对大视觉语言模型（LVLMs）中的幻觉问题，提出VisionWeaver网络，通过动态聚合多专家视觉特征以减少幻觉。属于视觉语言模型优化任务，解决幻觉检测与抑制问题，构建了细粒度评估基准VHBench-10。**

- **链接: [http://arxiv.org/pdf/2509.13836v1](http://arxiv.org/pdf/2509.13836v1)**

> **作者:** Weihang Wang; Xinhao Li; Ziyue Wang; Yan Pang; Jielei Zhang; Peiyi Li; Qiang Zhang; Longwen Gao
>
> **备注:** Accepted by EMNLP2025 Finding
>
> **摘要:** Object hallucination in Large Vision-Language Models (LVLMs) significantly impedes their real-world applicability. As the primary component for accurately interpreting visual information, the choice of visual encoder is pivotal. We hypothesize that the diverse training paradigms employed by different visual encoders instill them with distinct inductive biases, which leads to their diverse hallucination performances. Existing benchmarks typically focus on coarse-grained hallucination detection and fail to capture the diverse hallucinations elaborated in our hypothesis. To systematically analyze these effects, we introduce VHBench-10, a comprehensive benchmark with approximately 10,000 samples for evaluating LVLMs across ten fine-grained hallucination categories. Our evaluations confirm encoders exhibit unique hallucination characteristics. Building on these insights and the suboptimality of simple feature fusion, we propose VisionWeaver, a novel Context-Aware Routing Network. It employs global visual features to generate routing signals, dynamically aggregating visual features from multiple specialized experts. Comprehensive experiments confirm the effectiveness of VisionWeaver in significantly reducing hallucinations and improving overall model performance.
>
---
#### [new 007] Deep Lookup Network
- **分类: cs.CV; cs.AI**

- **简介: 论文提出一种基于查找表的高效神经网络操作，用于替代卷积网络中的乘法运算，以降低计算复杂度和能耗。该方法适用于图像分类、超分辨率和点云分类任务，提升推理效率并保持性能。属于神经网络优化任务，解决移动端部署中的计算瓶颈问题。**

- **链接: [http://arxiv.org/pdf/2509.13662v1](http://arxiv.org/pdf/2509.13662v1)**

> **作者:** Yulan Guo; Longguang Wang; Wendong Mao; Xiaoyu Dong; Yingqian Wang; Li Liu; Wei An
>
> **摘要:** Convolutional neural networks are constructed with massive operations with different types and are highly computationally intensive. Among these operations, multiplication operation is higher in computational complexity and usually requires {more} energy consumption with longer inference time than other operations, which hinders the deployment of convolutional neural networks on mobile devices. In many resource-limited edge devices, complicated operations can be calculated via lookup tables to reduce computational cost. Motivated by this, in this paper, we introduce a generic and efficient lookup operation which can be used as a basic operation for the construction of neural networks. Instead of calculating the multiplication of weights and activation values, simple yet efficient lookup operations are adopted to compute their responses. To enable end-to-end optimization of the lookup operation, we construct the lookup tables in a differentiable manner and propose several training strategies to promote their convergence. By replacing computationally expensive multiplication operations with our lookup operations, we develop lookup networks for the image classification, image super-resolution, and point cloud classification tasks. It is demonstrated that our lookup networks can benefit from the lookup operations to achieve higher efficiency in terms of energy consumption and inference speed while maintaining competitive performance to vanilla convolutional networks. Extensive experiments show that our lookup networks produce state-of-the-art performance on different tasks (both classification and regression tasks) and different data types (both images and point clouds).
>
---
#### [new 008] Intelligent Healthcare Imaging Platform An VLM-Based Framework for Automated Medical Image Analysis and Clinical Report Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出基于视觉语言模型（VLM）的智能医疗影像分析框架，用于自动检测肿瘤、生成临床报告。整合多种影像模态，结合自然语言处理与可视化技术，提升诊断效率与准确性，具有零样本学习能力，旨在优化放射科工作流程。**

- **链接: [http://arxiv.org/pdf/2509.13590v1](http://arxiv.org/pdf/2509.13590v1)**

> **作者:** Samer Al-Hamadani
>
> **备注:** 32 pages, 14 figures, 6 tables
>
> **摘要:** The rapid advancement of artificial intelligence (AI) in healthcare imaging has revolutionized diagnostic medicine and clinical decision-making processes. This work presents an intelligent multimodal framework for medical image analysis that leverages Vision-Language Models (VLMs) in healthcare diagnostics. The framework integrates Google Gemini 2.5 Flash for automated tumor detection and clinical report generation across multiple imaging modalities including CT, MRI, X-ray, and Ultrasound. The system combines visual feature extraction with natural language processing to enable contextual image interpretation, incorporating coordinate verification mechanisms and probabilistic Gaussian modeling for anomaly distribution. Multi-layered visualization techniques generate detailed medical illustrations, overlay comparisons, and statistical representations to enhance clinical confidence, with location measurement achieving 80 pixels average deviation. Result processing utilizes precise prompt engineering and textual analysis to extract structured clinical information while maintaining interpretability. Experimental evaluations demonstrated high performance in anomaly detection across multiple modalities. The system features a user-friendly Gradio interface for clinical workflow integration and demonstrates zero-shot learning capabilities to reduce dependence on large datasets. This framework represents a significant advancement in automated diagnostic support and radiological workflow efficiency, though clinical validation and multi-center evaluation are necessary prior to widespread adoption.
>
---
#### [new 009] SWA-PF: Semantic-Weighted Adaptive Particle Filter for Memory-Efficient 4-DoF UAV Localization in GNSS-Denied Environments
- **分类: cs.CV**

- **简介: 论文提出SWA-PF方法，用于解决GNSS拒止环境下UAV的高效定位问题。通过语义加权和优化粒子滤波，结合多高度数据集MAFS，实现低计算量、高精度的4-DoF定位。**

- **链接: [http://arxiv.org/pdf/2509.13795v1](http://arxiv.org/pdf/2509.13795v1)**

> **作者:** Jiayu Yuan; Ming Dai; Enhui Zheng; Chao Su; Nanxing Chen; Qiming Hu; Shibo Zhu; Yibin Cao
>
> **摘要:** Vision-based Unmanned Aerial Vehicle (UAV) localization systems have been extensively investigated for Global Navigation Satellite System (GNSS)-denied environments. However, existing retrieval-based approaches face limitations in dataset availability and persistent challenges including suboptimal real-time performance, environmental sensitivity, and limited generalization capability, particularly in dynamic or temporally varying environments. To overcome these limitations, we present a large-scale Multi-Altitude Flight Segments dataset (MAFS) for variable altitude scenarios and propose a novel Semantic-Weighted Adaptive Particle Filter (SWA-PF) method. This approach integrates robust semantic features from both UAV-captured images and satellite imagery through two key innovations: a semantic weighting mechanism and an optimized particle filtering architecture. Evaluated using our dataset, the proposed method achieves 10x computational efficiency gain over feature extraction methods, maintains global positioning errors below 10 meters, and enables rapid 4 degree of freedom (4-DoF) pose estimation within seconds using accessible low-resolution satellite maps. Code and dataset will be available at https://github.com/YuanJiayuuu/SWA-PF.
>
---
#### [new 010] EdiVal-Agent: An Object-Centric Framework for Automated, Scalable, Fine-Grained Evaluation of Multi-Turn Editing
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出EdiVal-Agent，用于多轮图像编辑的自动化、细粒度评估。它通过对象分解与多种工具结合，解决现有评估方法覆盖不足和不准确的问题，构建了EdiVal-Bench基准测试集。**

- **链接: [http://arxiv.org/pdf/2509.13399v1](http://arxiv.org/pdf/2509.13399v1)**

> **作者:** Tianyu Chen; Yasi Zhang; Zhi Zhang; Peiyu Yu; Shu Wang; Zhendong Wang; Kevin Lin; Xiaofei Wang; Zhengyuan Yang; Linjie Li; Chung-Ching Lin; Jianwen Xie; Oscar Leong; Lijuan Wang; Ying Nian Wu; Mingyuan Zhou
>
> **备注:** Tianyu Chen and Yasi Zhang contributed equally; Oscar Leong, Lijuan Wang, Ying Nian Wu, and Mingyuan Zhou advised equally
>
> **摘要:** Instruction-based image editing has advanced rapidly, yet reliable and interpretable evaluation remains a bottleneck. Current protocols either (i) depend on paired reference images -- resulting in limited coverage and inheriting biases from prior generative models -- or (ii) rely solely on zero-shot vision-language models (VLMs), whose prompt-based assessments of instruction following, content consistency, and visual quality are often imprecise. To address this, we introduce EdiVal-Agent, an automated, scalable, and fine-grained evaluation framework for multi-turn instruction-based editing from an object-centric perspective, supported by a suite of expert tools. Given an image, EdiVal-Agent first decomposes it into semantically meaningful objects, then synthesizes diverse, context-aware editing instructions. For evaluation, it integrates VLMs with open-vocabulary object detectors to assess instruction following, uses semantic-level feature extractors to evaluate content consistency, and leverages human preference models to judge visual quality. We show that combining VLMs with object detectors yields stronger agreement with human judgments in instruction-following evaluation compared to using VLMs alone and CLIP-based metrics. Furthermore, the pipeline's modular design allows future tools to be seamlessly integrated, enhancing evaluation accuracy over time. Instantiating this pipeline, we build EdiVal-Bench, a multi-turn editing benchmark covering 9 instruction types and 11 state-of-the-art editing models spanning autoregressive (AR) (including Nano Banana, GPT-Image-1), flow-matching, and diffusion paradigms. We demonstrate that EdiVal-Agent can be used to identify existing failure modes, thereby informing the development of the next generation of editing models. Project page: https://tianyucodings.github.io/EdiVAL-page/.
>
---
#### [new 011] MapAnything: Universal Feed-Forward Metric 3D Reconstruction
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 论文提出MapAnything，一种基于Transformer的前馈模型，统一处理多视角图像及几何输入，直接回归度量三维场景和相机参数，解决多种3D视觉任务，如结构从运动、深度估计等，实现高效联合训练。**

- **链接: [http://arxiv.org/pdf/2509.13414v1](http://arxiv.org/pdf/2509.13414v1)**

> **作者:** Nikhil Keetha; Norman Müller; Johannes Schönberger; Lorenzo Porzi; Yuchen Zhang; Tobias Fischer; Arno Knapitsch; Duncan Zauss; Ethan Weber; Nelson Antunes; Jonathon Luiten; Manuel Lopez-Antequera; Samuel Rota Bulò; Christian Richardt; Deva Ramanan; Sebastian Scherer; Peter Kontschieder
>
> **备注:** Project Page: https://map-anything.github.io/
>
> **摘要:** We introduce MapAnything, a unified transformer-based feed-forward model that ingests one or more images along with optional geometric inputs such as camera intrinsics, poses, depth, or partial reconstructions, and then directly regresses the metric 3D scene geometry and cameras. MapAnything leverages a factored representation of multi-view scene geometry, i.e., a collection of depth maps, local ray maps, camera poses, and a metric scale factor that effectively upgrades local reconstructions into a globally consistent metric frame. Standardizing the supervision and training across diverse datasets, along with flexible input augmentation, enables MapAnything to address a broad range of 3D vision tasks in a single feed-forward pass, including uncalibrated structure-from-motion, calibrated multi-view stereo, monocular depth estimation, camera localization, depth completion, and more. We provide extensive experimental analyses and model ablations demonstrating that MapAnything outperforms or matches specialist feed-forward models while offering more efficient joint training behavior, thus paving the way toward a universal 3D reconstruction backbone.
>
---
#### [new 012] Morphology-optimized Multi-Scale Fusion: Combining Local Artifacts and Mesoscopic Semantics for Deepfake Detection and Localization
- **分类: cs.CV**

- **简介: 该论文属于深度伪造检测与定位任务，旨在解决精确识别篡改区域的问题。提出一种多尺度融合方法，结合局部细节与全局语义，通过形态学操作提升定位准确性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.13776v1](http://arxiv.org/pdf/2509.13776v1)**

> **作者:** Chao Shuai; Gaojian Wang; Kun Pan; Tong Wu; Fanli Jin; Haohan Tan; Mengxiang Li; Zhenguang Liu; Feng Lin; Kui Ren
>
> **备注:** The 3rd Place, IJCAI 2025 Workshop on Deepfake Detection, Localization, and Interpretability
>
> **摘要:** While the pursuit of higher accuracy in deepfake detection remains a central goal, there is an increasing demand for precise localization of manipulated regions. Despite the remarkable progress made in classification-based detection, accurately localizing forged areas remains a significant challenge. A common strategy is to incorporate forged region annotations during model training alongside manipulated images. However, such approaches often neglect the complementary nature of local detail and global semantic context, resulting in suboptimal localization performance. Moreover, an often-overlooked aspect is the fusion strategy between local and global predictions. Naively combining the outputs from both branches can amplify noise and errors, thereby undermining the effectiveness of the localization. To address these issues, we propose a novel approach that independently predicts manipulated regions using both local and global perspectives. We employ morphological operations to fuse the outputs, effectively suppressing noise while enhancing spatial coherence. Extensive experiments reveal the effectiveness of each module in improving the accuracy and robustness of forgery localization.
>
---
#### [new 013] Semantic-Enhanced Cross-Modal Place Recognition for Robust Robot Localization
- **分类: cs.CV**

- **简介: 该论文属于机器人定位任务，旨在解决无GPS环境下视觉定位对光照、天气变化的敏感问题。提出SCM-PR框架，结合RGB图像语义与LiDAR几何信息，提升复杂场景下的定位鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.13474v1](http://arxiv.org/pdf/2509.13474v1)**

> **作者:** Yujia Lin; Nicholas Evans
>
> **摘要:** Ensuring accurate localization of robots in environments without GPS capability is a challenging task. Visual Place Recognition (VPR) techniques can potentially achieve this goal, but existing RGB-based methods are sensitive to changes in illumination, weather, and other seasonal changes. Existing cross-modal localization methods leverage the geometric properties of RGB images and 3D LiDAR maps to reduce the sensitivity issues highlighted above. Currently, state-of-the-art methods struggle in complex scenes, fine-grained or high-resolution matching, and situations where changes can occur in viewpoint. In this work, we introduce a framework we call Semantic-Enhanced Cross-Modal Place Recognition (SCM-PR) that combines high-level semantics utilizing RGB images for robust localization in LiDAR maps. Our proposed method introduces: a VMamba backbone for feature extraction of RGB images; a Semantic-Aware Feature Fusion (SAFF) module for using both place descriptors and segmentation masks; LiDAR descriptors that incorporate both semantics and geometry; and a cross-modal semantic attention mechanism in NetVLAD to improve matching. Incorporating the semantic information also was instrumental in designing a Multi-View Semantic-Geometric Matching and a Semantic Consistency Loss, both in a contrastive learning framework. Our experimental work on the KITTI and KITTI-360 datasets show that SCM-PR achieves state-of-the-art performance compared to other cross-modal place recognition methods.
>
---
#### [new 014] Real-Time Detection and Tracking of Foreign Object Intrusions in Power Systems via Feature-Based Edge Intelligence
- **分类: cs.CV; cs.SY; eess.SY**

- **简介: 论文提出一种三阶段框架，用于电力系统中异物入侵的实时检测与跟踪。任务为异物检测与追踪，解决复杂场景下的准确性和鲁棒性问题。采用YOLOv7、ConvNeXt和特征辅助跟踪器，并优化部署于边缘设备。**

- **链接: [http://arxiv.org/pdf/2509.13396v1](http://arxiv.org/pdf/2509.13396v1)**

> **作者:** Xinan Wang; Di Shi; Fengyu Wang
>
> **备注:** 12 page Journal paper, accepted by IEEE Open Access Journal of Power and Energy
>
> **摘要:** This paper presents a novel three-stage framework for real-time foreign object intrusion (FOI) detection and tracking in power transmission systems. The framework integrates: (1) a YOLOv7 segmentation model for fast and robust object localization, (2) a ConvNeXt-based feature extractor trained with triplet loss to generate discriminative embeddings, and (3) a feature-assisted IoU tracker that ensures resilient multi-object tracking under occlusion and motion. To enable scalable field deployment, the pipeline is optimized for deployment on low-cost edge hardware using mixed-precision inference. The system supports incremental updates by adding embeddings from previously unseen objects into a reference database without requiring model retraining. Extensive experiments on real-world surveillance and drone video datasets demonstrate the framework's high accuracy and robustness across diverse FOI scenarios. In addition, hardware benchmarks on NVIDIA Jetson devices confirm the framework's practicality and scalability for real-world edge applications.
>
---
#### [new 015] Gaussian Alignment for Relative Camera Pose Estimation via Single-View Reconstruction
- **分类: cs.CV; I.4.8; I.4.5**

- **简介: 该论文提出GARPS框架，解决单目图像间度量相对相机位姿估计问题。通过单视角重建生成3D高斯混合模型，并优化对齐目标实现鲁棒位姿估计，优于现有方法。属于多视角几何与单视角感知结合的任务。**

- **链接: [http://arxiv.org/pdf/2509.13652v1](http://arxiv.org/pdf/2509.13652v1)**

> **作者:** Yumin Li; Dylan Campbell
>
> **备注:** 12 pages, 4 figures, accepted by AJCAI 2025
>
> **摘要:** Estimating metric relative camera pose from a pair of images is of great importance for 3D reconstruction and localisation. However, conventional two-view pose estimation methods are not metric, with camera translation known only up to a scale, and struggle with wide baselines and textureless or reflective surfaces. This paper introduces GARPS, a training-free framework that casts this problem as the direct alignment of two independently reconstructed 3D scenes. GARPS leverages a metric monocular depth estimator and a Gaussian scene reconstructor to obtain a metric 3D Gaussian Mixture Model (GMM) for each image. It then refines an initial pose from a feed-forward two-view pose estimator by optimising a differentiable GMM alignment objective. This objective jointly considers geometric structure, view-independent colour, anisotropic covariance, and semantic feature consistency, and is robust to occlusions and texture-poor regions without requiring explicit 2D correspondences. Extensive experiments on the Real\-Estate10K dataset demonstrate that GARPS outperforms both classical and state-of-the-art learning-based methods, including MASt3R. These results highlight the potential of bridging single-view perception with multi-view geometry to achieve robust and metric relative pose estimation.
>
---
#### [new 016] LivePyxel: Accelerating image annotations with a Python-integrated webcam live streaming
- **分类: cs.CV**

- **简介: 该论文提出LivePyxel，一种集成Python的实时图像标注工具，解决传统标注软件需预上传数据的问题，支持实时标注与高效编辑，加速AI模型在实验流程中的开发。**

- **链接: [http://arxiv.org/pdf/2509.13504v1](http://arxiv.org/pdf/2509.13504v1)**

> **作者:** Uriel Garcilazo-Cruz; Joseph O. Okeme; Rodrigo A. Vargas--Hernández
>
> **备注:** 8 pages, 10 figures, SM, 5 pages, 4 figures
>
> **摘要:** The lack of flexible annotation tools has hindered the deployment of AI models in some scientific areas. Most existing image annotation software requires users to upload a precollected dataset, which limits support for on-demand pipelines and introduces unnecessary steps to acquire images. This constraint is particularly problematic in laboratory environments, where real-time data acquisition from instruments such as microscopes is increasingly common. In this work, we introduce \texttt{LivePixel}, a Python-based graphical user interface that integrates with imaging systems, such as webcams, microscopes, and others, to enable real-time image annotation. LivePyxel is designed to be easy to use through a simple interface that allows users to precisely delimit areas for annotation using tools commonly found in commercial graphics editing software. Of particular interest is the availability of B\'ezier splines and binary masks, and the software's capacity to work with non-destructive layers that enable high-performance editing. LivePyxel also integrates a wide compatibility across video devices, and it's optimized for object detection operations via the use of OpenCV in combination with high-performance libraries designed to handle matrix and linear algebra operations via Numpy effectively. LivePyxel facilitates seamless data collection and labeling, accelerating the development of AI models in experimental workflows. LivePyxel freely available at https://github.com/UGarCil/LivePyxel
>
---
#### [new 017] PROFUSEme: PROstate Cancer Biochemical Recurrence Prediction via FUSEd Multi-modal Embeddings
- **分类: cs.CV**

- **简介: 该论文提出PROFUSEme模型，用于预测前列腺癌患者术后生化复发。通过融合临床、影像和病理多模态数据，结合Cox回归模型，实现更准确的早期预测，提升患者治疗决策效果。**

- **链接: [http://arxiv.org/pdf/2509.14051v1](http://arxiv.org/pdf/2509.14051v1)**

> **作者:** Suhang You; Carla Pitarch-Abaigar; Sanket Kachole; Sumedh Sonawane; Juhyung Ha; Anish Sudarshan Gada; David Crandall; Rakesh Shiradkar; Spyridon Bakas
>
> **备注:** 11 pages, 1 figure, method paper for CHIMERA 2025 Challenge
>
> **摘要:** Almost 30% of prostate cancer (PCa) patients undergoing radical prostatectomy (RP) experience biochemical recurrence (BCR), characterized by increased prostate specific antigen (PSA) and associated with increased mortality. Accurate early prediction of BCR, at the time of RP, would contribute to prompt adaptive clinical decision-making and improved patient outcomes. In this work, we propose prostate cancer BCR prediction via fused multi-modal embeddings (PROFUSEme), which learns cross-modal interactions of clinical, radiology, and pathology data, following an intermediate fusion configuration in combination with Cox Proportional Hazard regressors. Quantitative evaluation of our proposed approach reveals superior performance, when compared with late fusion configurations, yielding a mean C-index of 0.861 ($\sigma=0.112$) on the internal 5-fold nested cross-validation framework, and a C-index of 0.7103 on the hold out data of CHIMERA 2025 challenge validation leaderboard.
>
---
#### [new 018] Invisible Yet Detected: PelFANet with Attention-Guided Anatomical Fusion for Pelvic Fracture Diagnosis
- **分类: cs.CV**

- **简介: 论文提出PelFANet，一种双流注意力网络，融合原始骨盆X光与分割后的骨骼图像，提升骨折分类效果。任务为骨盆骨折诊断，解决标准X光中细微或不可见骨折的识别问题，通过解剖学引导的特征融合实现高准确率检测。**

- **链接: [http://arxiv.org/pdf/2509.13873v1](http://arxiv.org/pdf/2509.13873v1)**

> **作者:** Siam Tahsin Bhuiyan; Rashedur Rahman; Sefatul Wasi; Naomi Yagi; Syoji Kobashi; Ashraful Islam; Saadia Binte Alam
>
> **备注:** Accepted at MICCAI EMERGE 2025
>
> **摘要:** Pelvic fractures pose significant diagnostic challenges, particularly in cases where fracture signs are subtle or invisible on standard radiographs. To address this, we introduce PelFANet, a dual-stream attention network that fuses raw pelvic X-rays with segmented bone images to improve fracture classification. The network em-ploys Fused Attention Blocks (FABlocks) to iteratively exchange and refine fea-tures from both inputs, capturing global context and localized anatomical detail. Trained in a two-stage pipeline with a segmentation-guided approach, PelFANet demonstrates superior performance over conventional methods. On the AMERI dataset, it achieves 88.68% accuracy and 0.9334 AUC on visible fractures, while generalizing effectively to invisible fracture cases with 82.29% accuracy and 0.8688 AUC, despite not being trained on them. These results highlight the clini-cal potential of anatomy-aware dual-input architectures for robust fracture detec-tion, especially in scenarios with subtle radiographic presentations.
>
---
#### [new 019] Multimodal Hate Detection Using Dual-Stream Graph Neural Networks
- **分类: cs.CV**

- **简介: 该论文属于多模态仇恨内容检测任务，旨在提升视频分类效果。提出双流图神经网络模型，通过构建实例图和互补权重图，突出仇恨内容，系统建模多模态结构关系，实现更准确的分类与解释性。**

- **链接: [http://arxiv.org/pdf/2509.13515v1](http://arxiv.org/pdf/2509.13515v1)**

> **作者:** Jiangbei Yue; Shuonan Yang; Tailin Chen; Jianbo Jiao; Zeyu Fu
>
> **摘要:** Hateful videos present serious risks to online safety and real-world well-being, necessitating effective detection methods. Although multimodal classification approaches integrating information from several modalities outperform unimodal ones, they typically neglect that even minimal hateful content defines a video's category. Specifically, they generally treat all content uniformly, instead of emphasizing the hateful components. Additionally, existing multimodal methods cannot systematically capture structured information in videos, limiting the effectiveness of multimodal fusion. To address these limitations, we propose a novel multimodal dual-stream graph neural network model. It constructs an instance graph by separating the given video into several instances to extract instance-level features. Then, a complementary weight graph assigns importance weights to these features, highlighting hateful instances. Importance weights and instance features are combined to generate video labels. Our model employs a graph-based framework to systematically model structured relationships within and across modalities. Extensive experiments on public datasets show that our model is state-of-the-art in hateful video classification and has strong explainability. Code is available: https://github.com/Multimodal-Intelligence-Lab-MIL/MultiHateGNN.
>
---
#### [new 020] Iterative Prompt Refinement for Safer Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成的安全优化任务，旨在解决生成内容可能包含有害或不适当图像的问题。提出了一种基于视觉语言模型的迭代提示优化算法，结合图文反馈提升安全性和用户意图一致性，并构建了多模态标注数据集辅助训练。**

- **链接: [http://arxiv.org/pdf/2509.13760v1](http://arxiv.org/pdf/2509.13760v1)**

> **作者:** Jinwoo Jeon; JunHyeok Oh; Hayeong Lee; Byung-Jun Lee
>
> **摘要:** Text-to-Image (T2I) models have made remarkable progress in generating images from text prompts, but their output quality and safety still depend heavily on how prompts are phrased. Existing safety methods typically refine prompts using large language models (LLMs), but they overlook the images produced, which can result in unsafe outputs or unnecessary changes to already safe prompts. To address this, we propose an iterative prompt refinement algorithm that uses Vision Language Models (VLMs) to analyze both the input prompts and the generated images. By leveraging visual feedback, our method refines prompts more effectively, improving safety while maintaining user intent and reliability comparable to existing LLM-based approaches. Additionally, we introduce a new dataset labeled with both textual and visual safety signals using off-the-shelf multi-modal LLM, enabling supervised fine-tuning. Experimental results demonstrate that our approach produces safer outputs without compromising alignment with user intent, offering a practical solution for generating safer T2I content. Our code is available at https://github.com/ku-dmlab/IPR. \textbf{\textcolor{red}WARNING: This paper contains examples of harmful or inappropriate images generated by models.
>
---
#### [new 021] GenExam: A Multidisciplinary Text-to-Image Exam
- **分类: cs.CV**

- **简介: 该论文提出GenExam，首个多学科文本到图像考试基准，包含1000个样本，涵盖10个学科。旨在评估模型在知识整合、推理与图像生成能力，解决现有基准忽视严格绘图评估的问题，实验表明当前模型表现不佳，凸显挑战性。**

- **链接: [http://arxiv.org/pdf/2509.14232v1](http://arxiv.org/pdf/2509.14232v1)**

> **作者:** Zhaokai Wang; Penghao Yin; Xiangyu Zhao; Changyao Tian; Yu Qiao; Wenhai Wang; Jifeng Dai; Gen Luo
>
> **摘要:** Exams are a fundamental test of expert-level intelligence and require integrated understanding, reasoning, and generation. Existing exam-style benchmarks mainly focus on understanding and reasoning tasks, and current generation benchmarks emphasize the illustration of world knowledge and visual concepts, neglecting the evaluation of rigorous drawing exams. We introduce GenExam, the first benchmark for multidisciplinary text-to-image exams, featuring 1,000 samples across 10 subjects with exam-style prompts organized under a four-level taxonomy. Each problem is equipped with ground-truth images and fine-grained scoring points to enable a precise evaluation of semantic correctness and visual plausibility. Experiments show that even state-of-the-art models such as GPT-Image-1 and Gemini-2.5-Flash-Image achieve less than 15% strict scores, and most models yield almost 0%, suggesting the great challenge of our benchmark. By framing image generation as an exam, GenExam offers a rigorous assessment of models' ability to integrate knowledge, reasoning, and generation, providing insights on the path to general AGI.
>
---
#### [new 022] Noise-Level Diffusion Guidance: Well Begun is Half Done
- **分类: cs.CV**

- **简介: 该论文提出噪声水平引导（NLG）方法，优化扩散模型初始噪声，提升图像生成质量与条件遵循能力。属于图像生成任务，解决噪声影响输出稳定性问题，无需额外数据或网络，实现高效通用优化。**

- **链接: [http://arxiv.org/pdf/2509.13936v1](http://arxiv.org/pdf/2509.13936v1)**

> **作者:** Harvey Mannering; Zhiwu Huang; Adam Prugel-Bennett
>
> **摘要:** Diffusion models have achieved state-of-the-art image generation. However, the random Gaussian noise used to start the diffusion process influences the final output, causing variations in image quality and prompt adherence. Existing noise-level optimization approaches generally rely on extra dataset construction, additional networks, or backpropagation-based optimization, limiting their practicality. In this paper, we propose Noise Level Guidance (NLG), a simple, efficient, and general noise-level optimization approach that refines initial noise by increasing the likelihood of its alignment with general guidance - requiring no additional training data, auxiliary networks, or backpropagation. The proposed NLG approach provides a unified framework generalizable to both conditional and unconditional diffusion models, accommodating various forms of diffusion-level guidance. Extensive experiments on five standard benchmarks demonstrate that our approach enhances output generation quality and input condition adherence. By seamlessly integrating with existing guidance methods while maintaining computational efficiency, our method establishes NLG as a practical and scalable enhancement to diffusion models. Code can be found at https://github.com/harveymannering/NoiseLevelGuidance.
>
---
#### [new 023] SAMIR, an efficient registration framework via robust feature learning from SAM
- **分类: cs.CV**

- **简介: 该论文提出SAMIR，用于医学图像配准任务，解决特征提取不准确问题。利用SAM模型提取结构感知特征，并设计轻量3D头和层次一致性损失，提升配准精度，在多个数据集上取得显著性能提升。**

- **链接: [http://arxiv.org/pdf/2509.13629v1](http://arxiv.org/pdf/2509.13629v1)**

> **作者:** Yue He; Min Liu; Qinghao Liu; Jiazheng Wang; Yaonan Wang; Hang Zhang; Xiang Chen
>
> **摘要:** Image registration is a fundamental task in medical image analysis. Deformations are often closely related to the morphological characteristics of tissues, making accurate feature extraction crucial. Recent weakly supervised methods improve registration by incorporating anatomical priors such as segmentation masks or landmarks, either as inputs or in the loss function. However, such weak labels are often not readily available, limiting their practical use. Motivated by the strong representation learning ability of visual foundation models, this paper introduces SAMIR, an efficient medical image registration framework that utilizes the Segment Anything Model (SAM) to enhance feature extraction. SAM is pretrained on large-scale natural image datasets and can learn robust, general-purpose visual representations. Rather than using raw input images, we design a task-specific adaptation pipeline using SAM's image encoder to extract structure-aware feature embeddings, enabling more accurate modeling of anatomical consistency and deformation patterns. We further design a lightweight 3D head to refine features within the embedding space, adapting to local deformations in medical images. Additionally, we introduce a Hierarchical Feature Consistency Loss to guide coarse-to-fine feature matching and improve anatomical alignment. Extensive experiments demonstrate that SAMIR significantly outperforms state-of-the-art methods on benchmark datasets for both intra-subject cardiac image registration and inter-subject abdomen CT image registration, achieving performance improvements of 2.68% on ACDC and 6.44% on the abdomen dataset. The source code will be publicly available on GitHub following the acceptance of this paper.
>
---
#### [new 024] Consistent View Alignment Improves Foundation Models for 3D Medical Image Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于3D医学图像分割任务，旨在解决基础模型表示学习中结构信息不足的问题。提出一致视图对齐方法，通过对齐不同视角的表示以提升下游任务性能，并在MICCAI 2025 SSL3D挑战赛中取得优异成绩。**

- **链接: [http://arxiv.org/pdf/2509.13846v1](http://arxiv.org/pdf/2509.13846v1)**

> **作者:** Puru Vaish; Felix Meister; Tobias Heimann; Christoph Brune; Jelmer M. Wolterink
>
> **备注:** MICCAI 2025: 1st Place in Transformer track and 2nd Place in Convolution track of SSL3D-OpenMind challenge
>
> **摘要:** Many recent approaches in representation learning implicitly assume that uncorrelated views of a data point are sufficient to learn meaningful representations for various downstream tasks. In this work, we challenge this assumption and demonstrate that meaningful structure in the latent space does not emerge naturally. Instead, it must be explicitly induced. We propose a method that aligns representations from different views of the data to align complementary information without inducing false positives. Our experiments show that our proposed self-supervised learning method, Consistent View Alignment, improves performance for downstream tasks, highlighting the critical role of structured view alignment in learning effective representations. Our method achieved first and second place in the MICCAI 2025 SSL3D challenge when using a Primus vision transformer and ResEnc convolutional neural network, respectively. The code and pretrained model weights are released at https://github.com/Tenbatsu24/LatentCampus.
>
---
#### [new 025] An Empirical Analysis of VLM-based OOD Detection: Mechanisms, Advantages, and Sensitivity
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究基于视觉-语言模型（VLM）的分布外检测任务，分析其机制、优势及敏感性。通过实证分析，揭示VLM在零样本检测中的有效性、优于单模态方法的原因，以及对提示语高度敏感的弱点，为构建更稳健的AI系统提供指导。**

- **链接: [http://arxiv.org/pdf/2509.13375v1](http://arxiv.org/pdf/2509.13375v1)**

> **作者:** Yuxiao Lee; Xiaofeng Cao; Wei Ye; Jiangchao Yao; Jingkuan Song; Heng Tao Shen
>
> **摘要:** Vision-Language Models (VLMs), such as CLIP, have demonstrated remarkable zero-shot out-of-distribution (OOD) detection capabilities, vital for reliable AI systems. Despite this promising capability, a comprehensive understanding of (1) why they work so effectively, (2) what advantages do they have over single-modal methods, and (3) how is their behavioral robustness -- remains notably incomplete within the research community. This paper presents a systematic empirical analysis of VLM-based OOD detection using in-distribution (ID) and OOD prompts. (1) Mechanisms: We systematically characterize and formalize key operational properties within the VLM embedding space that facilitate zero-shot OOD detection. (2) Advantages: We empirically quantify the superiority of these models over established single-modal approaches, attributing this distinct advantage to the VLM's capacity to leverage rich semantic novelty. (3) Sensitivity: We uncovers a significant and previously under-explored asymmetry in their robustness profile: while exhibiting resilience to common image noise, these VLM-based methods are highly sensitive to prompt phrasing. Our findings contribute a more structured understanding of the strengths and critical vulnerabilities inherent in VLM-based OOD detection, offering crucial, empirically-grounded guidance for developing more robust and reliable future designs.
>
---
#### [new 026] BEVUDA++: Geometric-aware Unsupervised Domain Adaptation for Multi-View 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文属于多视角3D目标检测任务，解决跨域场景下的BEV感知性能下降问题。提出几何感知的BEVUDA++框架，包含可靠深度教师和几何一致学生模型，通过统一几何嵌入空间和不确定性引导方法，提升跨域适应效果。**

- **链接: [http://arxiv.org/pdf/2509.14151v1](http://arxiv.org/pdf/2509.14151v1)**

> **作者:** Rongyu Zhang; Jiaming Liu; Xiaoqi Li; Xiaowei Chi; Dan Wang; Li Du; Yuan Du; Shanghang Zhang
>
> **备注:** Accepted by IEEE TCSVT
>
> **摘要:** Vision-centric Bird's Eye View (BEV) perception holds considerable promise for autonomous driving. Recent studies have prioritized efficiency or accuracy enhancements, yet the issue of domain shift has been overlooked, leading to substantial performance degradation upon transfer. We identify major domain gaps in real-world cross-domain scenarios and initiate the first effort to address the Domain Adaptation (DA) challenge in multi-view 3D object detection for BEV perception. Given the complexity of BEV perception approaches with their multiple components, domain shift accumulation across multi-geometric spaces (e.g., 2D, 3D Voxel, BEV) poses a significant challenge for BEV domain adaptation. In this paper, we introduce an innovative geometric-aware teacher-student framework, BEVUDA++, to diminish this issue, comprising a Reliable Depth Teacher (RDT) and a Geometric Consistent Student (GCS) model. Specifically, RDT effectively blends target LiDAR with dependable depth predictions to generate depth-aware information based on uncertainty estimation, enhancing the extraction of Voxel and BEV features that are essential for understanding the target domain. To collaboratively reduce the domain shift, GCS maps features from multiple spaces into a unified geometric embedding space, thereby narrowing the gap in data distribution between the two domains. Additionally, we introduce a novel Uncertainty-guided Exponential Moving Average (UEMA) to further reduce error accumulation due to domain shifts informed by previously obtained uncertainty guidance. To demonstrate the superiority of our proposed method, we execute comprehensive experiments in four cross-domain scenarios, securing state-of-the-art performance in BEV 3D object detection tasks, e.g., 12.9\% NDS and 9.5\% mAP enhancement on Day-Night adaptation.
>
---
#### [new 027] Improving 3D Gaussian Splatting Compression by Scene-Adaptive Lattice Vector Quantization
- **分类: cs.CV**

- **简介: 论文提出场景自适应晶格向量量化（SALVQ），用于提升3D高斯泼溅（3DGS）压缩效率。针对现有方法依赖简单标量量化的问题，采用更高效的向量量化方法，在保持低复杂度的同时提高压缩性能和适应性。**

- **链接: [http://arxiv.org/pdf/2509.13482v1](http://arxiv.org/pdf/2509.13482v1)**

> **作者:** Hao Xu; Xiaolin Wu; Xi Zhang
>
> **备注:** Code available at https://github.com/hxu160/SALVQ
>
> **摘要:** 3D Gaussian Splatting (3DGS) is rapidly gaining popularity for its photorealistic rendering quality and real-time performance, but it generates massive amounts of data. Hence compressing 3DGS data is necessary for the cost effectiveness of 3DGS models. Recently, several anchor-based neural compression methods have been proposed, achieving good 3DGS compression performance. However, they all rely on uniform scalar quantization (USQ) due to its simplicity. A tantalizing question is whether more sophisticated quantizers can improve the current 3DGS compression methods with very little extra overhead and minimal change to the system. The answer is yes by replacing USQ with lattice vector quantization (LVQ). To better capture scene-specific characteristics, we optimize the lattice basis for each scene, improving LVQ's adaptability and R-D efficiency. This scene-adaptive LVQ (SALVQ) strikes a balance between the R-D efficiency of vector quantization and the low complexity of USQ. SALVQ can be seamlessly integrated into existing 3DGS compression architectures, enhancing their R-D performance with minimal modifications and computational overhead. Moreover, by scaling the lattice basis vectors, SALVQ can dynamically adjust lattice density, enabling a single model to accommodate multiple bit rate targets. This flexibility eliminates the need to train separate models for different compression levels, significantly reducing training time and memory consumption.
>
---
#### [new 028] Controllable-Continuous Color Editing in Diffusion Model via Color Mapping
- **分类: cs.CV**

- **简介: 该论文属于文本驱动图像编辑任务，旨在解决颜色编辑中精度不足与连续控制困难的问题。提出颜色映射模块，建立文本嵌入与RGB值的对应关系，实现精确、连续的颜色控制。**

- **链接: [http://arxiv.org/pdf/2509.13756v1](http://arxiv.org/pdf/2509.13756v1)**

> **作者:** Yuqi Yang; Dongliang Chang; Yuanchen Fang; Yi-Zhe SonG; Zhanyu Ma; Jun Guo
>
> **摘要:** In recent years, text-driven image editing has made significant progress. However, due to the inherent ambiguity and discreteness of natural language, color editing still faces challenges such as insufficient precision and difficulty in achieving continuous control. Although linearly interpolating the embedding vectors of different textual descriptions can guide the model to generate a sequence of images with varying colors, this approach lacks precise control over the range of color changes in the output images. Moreover, the relationship between the interpolation coefficient and the resulting image color is unknown and uncontrollable. To address these issues, we introduce a color mapping module that explicitly models the correspondence between the text embedding space and image RGB values. This module predicts the corresponding embedding vector based on a given RGB value, enabling precise color control of the generated images while maintaining semantic consistency. Users can specify a target RGB range to generate images with continuous color variations within the desired range, thereby achieving finer-grained, continuous, and controllable color editing. Experimental results demonstrate that our method performs well in terms of color continuity and controllability.
>
---
#### [new 029] Deceptive Beauty: Evaluating the Impact of Beauty Filters on Deepfake and Morphing Attack Detection
- **分类: cs.CV**

- **简介: 该论文研究美颜滤镜对深度伪造和换脸攻击检测的影响。属于图像检测任务，旨在解决美颜滤镜削弱检测系统性能的问题。通过实验分析，揭示了滤镜导致检测性能下降，强调需提升模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.14120v1](http://arxiv.org/pdf/2509.14120v1)**

> **作者:** Sara Concas; Simone Maurizio La Cava; Andrea Panzino; Ester Masala; Giulia Orrù; Gian Luca Marcialis
>
> **备注:** Accepted at the 2025 IEEE INTERNATIONAL CONFERENCE ON Metrology for eXtended Reality, Artificial Intelligence and Neural Engineering
>
> **摘要:** Digital beautification through social media filters has become increasingly popular, raising concerns about the reliability of facial images and videos and the effectiveness of automated face analysis. This issue is particularly critical for digital manipulation detectors, systems aiming at distinguishing between genuine and manipulated data, especially in cases involving deepfakes and morphing attacks designed to deceive humans and automated facial recognition. This study examines whether beauty filters impact the performance of deepfake and morphing attack detectors. We perform a comprehensive analysis, evaluating multiple state-of-the-art detectors on benchmark datasets before and after applying various smoothing filters. Our findings reveal performance degradation, highlighting vulnerabilities introduced by facial enhancements and underscoring the need for robust detection models resilient to such alterations.
>
---
#### [new 030] Re-purposing SAM into Efficient Visual Projectors for MLLM-Based Referring Image Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 论文提出一种高效视觉投影方法，用于多模态大语言模型（MLLM）的指称图像分割任务。旨在解决传统视觉投影中视觉token冗余导致计算负担重的问题，通过语义超像素压缩token序列，提升效率并保持性能。**

- **链接: [http://arxiv.org/pdf/2509.13676v1](http://arxiv.org/pdf/2509.13676v1)**

> **作者:** Xiaobo Yang; Xiaojin Gong
>
> **摘要:** Recently, Referring Image Segmentation (RIS) frameworks that pair the Multimodal Large Language Model (MLLM) with the Segment Anything Model (SAM) have achieved impressive results. However, adapting MLLM to segmentation is computationally intensive, primarily due to visual token redundancy. We observe that traditional patch-wise visual projectors struggle to strike a balance between reducing the number of visual tokens and preserving semantic clarity, often retaining overly long token sequences to avoid performance drops. Inspired by text tokenizers, we propose a novel semantic visual projector that leverages semantic superpixels generated by SAM to identify "visual words" in an image. By compressing and projecting semantic superpixels as visual tokens, our approach adaptively shortens the token sequence according to scene complexity while minimizing semantic loss in compression. To mitigate loss of information, we propose a semantic superpixel positional embedding to strengthen MLLM's awareness of superpixel geometry and position, alongside a semantic superpixel aggregator to preserve both fine-grained details inside superpixels and global context outside. Experiments show that our method cuts visual tokens by 93% without compromising performance, notably speeding up MLLM training and inference, and outperforming existing compressive visual projectors on RIS.
>
---
#### [new 031] Task-Aware Image Signal Processor for Advanced Visual Perception
- **分类: cs.CV**

- **简介: 论文提出任务感知图像信号处理器（TA-ISP），用于提升视觉感知任务性能。针对传统ISP计算开销大、表示能力有限的问题，设计轻量级多尺度调制操作，有效降低参数量与推理时间，适用于资源受限设备。**

- **链接: [http://arxiv.org/pdf/2509.13762v1](http://arxiv.org/pdf/2509.13762v1)**

> **作者:** Kai Chen; Jin Xiao; Leheng Zhang; Kexuan Shi; Shuhang Gu
>
> **摘要:** In recent years, there has been a growing trend in computer vision towards exploiting RAW sensor data, which preserves richer information compared to conventional low-bit RGB images. Early studies mainly focused on enhancing visual quality, while more recent efforts aim to leverage the abundant information in RAW data to improve the performance of visual perception tasks such as object detection and segmentation. However, existing approaches still face two key limitations: large-scale ISP networks impose heavy computational overhead, while methods based on tuning traditional ISP pipelines are restricted by limited representational capacity.To address these issues, we propose Task-Aware Image Signal Processing (TA-ISP), a compact RAW-to-RGB framework that produces task-oriented representations for pretrained vision models. Instead of heavy dense convolutional pipelines, TA-ISP predicts a small set of lightweight, multi-scale modulation operators that act at global, regional, and pixel scales to reshape image statistics across different spatial extents. This factorized control significantly expands the range of spatially varying transforms that can be represented while keeping memory usage, computation, and latency tightly constrained. Evaluated on several RAW-domain detection and segmentation benchmarks under both daytime and nighttime conditions, TA-ISP consistently improves downstream accuracy while markedly reducing parameter count and inference time, making it well suited for deployment on resource-constrained devices.
>
---
#### [new 032] Landcover classification and change detection using remote sensing and machine learning: a case study of Western Fiji
- **分类: cs.CV; cs.AI; stat.AP**

- **简介: 该论文属于土地覆盖分类与变化检测任务，旨在利用遥感与机器学习分析2013-2024年斐济纳迪地区土地利用变化。研究使用Landsat-8数据和深度学习方法生成土地覆盖图，并可视化城市扩张变化。**

- **链接: [http://arxiv.org/pdf/2509.13388v1](http://arxiv.org/pdf/2509.13388v1)**

> **作者:** Yadvendra Gurjar; Ruoni Wan; Ehsan Farahbakhsh; Rohitash Chandra
>
> **摘要:** As a developing country, Fiji is facing rapid urbanisation, which is visible in the massive development projects that include housing, roads, and civil works. In this study, we present machine learning and remote sensing frameworks to compare land use and land cover change from 2013 to 2024 in Nadi, Fiji. The ultimate goal of this study is to provide technical support in land cover/land use modelling and change detection. We used Landsat-8 satellite image for the study region and created our training dataset with labels for supervised machine learning. We used Google Earth Engine and unsupervised machine learning via k-means clustering to generate the land cover map. We used convolutional neural networks to classify the selected regions' land cover types. We present a visualisation of change detection, highlighting urban area changes over time to monitor changes in the map.
>
---
#### [new 033] Hybrid Quantum-Classical Model for Image Classification
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出并评估了混合量子-经典模型在图像分类任务中的性能，对比传统CNN模型。通过在MNIST、CIFAR100和STL10数据集上的实验，验证了混合模型在准确率、训练效率和鲁棒性方面的优势。**

- **链接: [http://arxiv.org/pdf/2509.13353v1](http://arxiv.org/pdf/2509.13353v1)**

> **作者:** Muhammad Adnan Shahzad
>
> **摘要:** This study presents a systematic comparison between hybrid quantum-classical neural networks and purely classical models across three benchmark datasets (MNIST, CIFAR100, and STL10) to evaluate their performance, efficiency, and robustness. The hybrid models integrate parameterized quantum circuits with classical deep learning architectures, while the classical counterparts use conventional convolutional neural networks (CNNs). Experiments were conducted over 50 training epochs for each dataset, with evaluations on validation accuracy, test accuracy, training time, computational resource usage, and adversarial robustness (tested with $\epsilon=0.1$ perturbations).Key findings demonstrate that hybrid models consistently outperform classical models in final accuracy, achieving {99.38\% (MNIST), 41.69\% (CIFAR100), and 74.05\% (STL10) validation accuracy, compared to classical benchmarks of 98.21\%, 32.25\%, and 63.76\%, respectively. Notably, the hybrid advantage scales with dataset complexity, showing the most significant gains on CIFAR100 (+9.44\%) and STL10 (+10.29\%). Hybrid models also train 5--12$\times$ faster (e.g., 21.23s vs. 108.44s per epoch on MNIST) and use 6--32\% fewer parameters} while maintaining superior generalization to unseen test data.Adversarial robustness tests reveal that hybrid models are significantly more resilient on simpler datasets (e.g., 45.27\% robust accuracy on MNIST vs. 10.80\% for classical) but show comparable fragility on complex datasets like CIFAR100 ($\sim$1\% robustness for both). Resource efficiency analyses indicate that hybrid models consume less memory (4--5GB vs. 5--6GB for classical) and lower CPU utilization (9.5\% vs. 23.2\% on average).These results suggest that hybrid quantum-classical architectures offer compelling advantages in accuracy, training efficiency, and parameter scalability, particularly for complex vision tasks.
>
---
#### [new 034] MemGS: Memory-Efficient Gaussian Splatting for Real-Time SLAM
- **分类: cs.CV**

- **简介: 该论文提出MemGS方法，针对嵌入式平台SLAM任务中内存与重建质量的矛盾，通过体素空间合并冗余高斯点并采用PG采样提升渲染质量，有效降低GPU内存占用，提高系统性能。**

- **链接: [http://arxiv.org/pdf/2509.13536v1](http://arxiv.org/pdf/2509.13536v1)**

> **作者:** Yinlong Bai; Hongxin Zhang; Sheng Zhong; Junkai Niu; Hai Li; Yijia He; Yi Zhou
>
> **摘要:** Recent advancements in 3D Gaussian Splatting (3DGS) have made a significant impact on rendering and reconstruction techniques. Current research predominantly focuses on improving rendering performance and reconstruction quality using high-performance desktop GPUs, largely overlooking applications for embedded platforms like micro air vehicles (MAVs). These devices, with their limited computational resources and memory, often face a trade-off between system performance and reconstruction quality. In this paper, we improve existing methods in terms of GPU memory usage while enhancing rendering quality. Specifically, to address redundant 3D Gaussian primitives in SLAM, we propose merging them in voxel space based on geometric similarity. This reduces GPU memory usage without impacting system runtime performance. Furthermore, rendering quality is improved by initializing 3D Gaussian primitives via Patch-Grid (PG) point sampling, enabling more accurate modeling of the entire scene. Quantitative and qualitative evaluations on publicly available datasets demonstrate the effectiveness of our improvements.
>
---
#### [new 035] Proximity-Based Evidence Retrieval for Uncertainty-Aware Neural Networks
- **分类: cs.CV; cs.AI; cs.LG; cs.NE; 68T07, 68T09**

- **简介: 论文提出一种基于邻近证据的不确定性感知决策机制，通过检索嵌入空间中的近邻样本并融合其预测分布，实现实例自适应的阈值判断，提升决策透明度与可靠性。属于不确定性感知神经网络任务，解决传统熵阈值法不可靠、不可解释的问题。**

- **链接: [http://arxiv.org/pdf/2509.13338v1](http://arxiv.org/pdf/2509.13338v1)**

> **作者:** Hassan Gharoun; Mohammad Sadegh Khorshidi; Kasra Ranjbarigderi; Fang Chen; Amir H. Gandomi
>
> **备注:** 15 pages, 4 figures, 3 tables
>
> **摘要:** This work proposes an evidence-retrieval mechanism for uncertainty-aware decision-making that replaces a single global cutoff with an evidence-conditioned, instance-adaptive criterion. For each test instance, proximal exemplars are retrieved in an embedding space; their predictive distributions are fused via Dempster-Shafer theory. The resulting fused belief acts as a per-instance thresholding mechanism. Because the supporting evidences are explicit, decisions are transparent and auditable. Experiments on CIFAR-10/100 with BiT and ViT backbones show higher or comparable uncertainty-aware performance with materially fewer confidently incorrect outcomes and a sustainable review load compared with applying threshold on prediction entropy. Notably, only a few evidences are sufficient to realize these gains; increasing the evidence set yields only modest changes. These results indicate that evidence-conditioned tagging provides a more reliable and interpretable alternative to fixed prediction entropy thresholds for operational uncertainty-aware decision-making.
>
---
#### [new 036] EvHand-FPV: Efficient Event-Based 3D Hand Tracking from First-Person View
- **分类: cs.CV**

- **简介: 该论文提出EvHand-FPV，解决第一视角下基于事件相机的高效3D手部跟踪问题。通过构建数据集、引入ROI机制和多任务学习，实现参数减少89%、计算量降低89%，提升跟踪精度与效率，适用于XR设备。**

- **链接: [http://arxiv.org/pdf/2509.13883v1](http://arxiv.org/pdf/2509.13883v1)**

> **作者:** Zhen Xu; Guorui Lu; Chang Gao; Qinyu Chen
>
> **备注:** 8 pages
>
> **摘要:** Hand tracking holds great promise for intuitive interaction paradigms, but frame-based methods often struggle to meet the requirements of accuracy, low latency, and energy efficiency, especially in resource-constrained settings such as Extended Reality (XR) devices. Event cameras provide $\mu$s-level temporal resolution at mW-level power by asynchronously sensing brightness changes. In this work, we present EvHand-FPV, a lightweight framework for egocentric First-Person-View 3D hand tracking from a single event camera. We construct an event-based FPV dataset that couples synthetic training data with 3D labels and real event data with 2D labels for evaluation to address the scarcity of egocentric benchmarks. EvHand-FPV also introduces a wrist-based region of interest (ROI) that localizes the hand region via geometric cues, combined with an end-to-end mapping strategy that embeds ROI offsets into the network to reduce computation without explicit reconstruction, and a multi-task learning strategy with an auxiliary geometric feature head that improves representations without test-time overhead. On our real FPV test set, EvHand-FPV improves 2D-AUCp from 0.77 to 0.85 while reducing parameters from 11.2M to 1.2M by 89% and FLOPs per inference from 1.648G to 0.185G by 89%. It also maintains a competitive 3D-AUCp of 0.84 on synthetic data. These results demonstrate accurate and efficient egocentric event-based hand tracking suitable for on-device XR applications. The dataset and code are available at https://github.com/zen5x5/EvHand-FPV.
>
---
#### [new 037] Annotating Satellite Images of Forests with Keywords from a Specialized Corpus in the Context of Change Detection
- **分类: cs.CV; cs.CL; cs.IR; cs.MM; I.2; I.4; I.7; H.3**

- **简介: 该论文提出一种基于深度学习的卫星图像变化检测方法，用于监测亚马逊雨林的森林砍伐。通过对比不同时间的图像，自动标注变化区域，并利用相关科学文献提取关键词进行注释，以支持环境研究。**

- **链接: [http://arxiv.org/pdf/2509.13586v1](http://arxiv.org/pdf/2509.13586v1)**

> **作者:** Nathalie Neptune; Josiane Mothe
>
> **摘要:** The Amazon rain forest is a vital ecosystem that plays a crucial role in regulating the Earth's climate and providing habitat for countless species. Deforestation in the Amazon is a major concern as it has a significant impact on global carbon emissions and biodiversity. In this paper, we present a method for detecting deforestation in the Amazon using image pairs from Earth observation satellites. Our method leverages deep learning techniques to compare the images of the same area at different dates and identify changes in the forest cover. We also propose a visual semantic model that automatically annotates the detected changes with relevant keywords. The candidate annotation for images are extracted from scientific documents related to the Amazon region. We evaluate our approach on a dataset of Amazon image pairs and demonstrate its effectiveness in detecting deforestation and generating relevant annotations. Our method provides a useful tool for monitoring and studying the impact of deforestation in the Amazon. While we focus on environment applications of our work by using images of deforestation in the Amazon rain forest to demonstrate the effectiveness of our proposed approach, it is generic enough to be applied to other domains.
>
---
#### [new 038] Cross-modal Full-mode Fine-grained Alignment for Text-to-Image Person Retrieval
- **分类: cs.CV**

- **简介: 该论文属于文本到图像人物检索任务，旨在解决文本与图像模态对齐问题。提出FMFA框架，通过显式细粒度对齐和自适应相似性分布匹配模块，提升全局匹配精度，无需额外监督。**

- **链接: [http://arxiv.org/pdf/2509.13754v1](http://arxiv.org/pdf/2509.13754v1)**

> **作者:** Hao Yin; Xin Man; Feiyu Chen; Jie Shao; Heng Tao Shen
>
> **摘要:** Text-to-Image Person Retrieval (TIPR) is a cross-modal matching task that aims to retrieve the most relevant person images based on a given text query. The key challenge in TIPR lies in achieving effective alignment between textual and visual modalities within a common latent space. To address this challenge, prior approaches incorporate attention mechanisms for implicit cross-modal local alignment. However, they lack the ability to verify whether all local features are correctly aligned. Moreover, existing methods primarily focus on hard negative samples during model updates, with the goal of refining distinctions between positive and negative pairs, often neglecting incorrectly matched positive pairs. To alleviate these issues, we propose FMFA, a cross-modal Full-Mode Fine-grained Alignment framework, which enhances global matching through explicit fine-grained alignment and existing implicit relational reasoning -- hence the term ``full-mode" -- without requiring additional supervision. Specifically, we design an Adaptive Similarity Distribution Matching (A-SDM) module to rectify unmatched positive sample pairs. A-SDM adaptively pulls the unmatched positive pairs closer in the joint embedding space, thereby achieving more precise global alignment. Additionally, we introduce an Explicit Fine-grained Alignment (EFA) module, which makes up for the lack of verification capability of implicit relational reasoning. EFA strengthens explicit cross-modal fine-grained interactions by sparsifying the similarity matrix and employs a hard coding method for local alignment. Our proposed method is evaluated on three public datasets, achieving state-of-the-art performance among all global matching methods. Our code is available at https://github.com/yinhao1102/FMFA.
>
---
#### [new 039] DEFT-VTON: Efficient Virtual Try-On with Consistent Generalised H-Transform
- **分类: cs.CV**

- **简介: 该论文属于虚拟试穿（VTO）任务，旨在以有限训练和推理资源实现高效VTO。提出DEFT-VTON方法，通过冻结预训练模型参数并训练小规模h-变换网络，结合自适应一致性损失，显著降低计算成本并提升性能。**

- **链接: [http://arxiv.org/pdf/2509.13506v1](http://arxiv.org/pdf/2509.13506v1)**

> **作者:** Xingzi Xu; Qi Li; Shuwen Qiu; Julien Han; Karim Bouyarmane
>
> **备注:** Published in 2025 CVPR Workshop
>
> **摘要:** Diffusion models enable high-quality virtual try-on (VTO) with their established image synthesis abilities. Despite the extensive end-to-end training of large pre-trained models involved in current VTO methods, real-world applications often prioritize limited training and inference, serving, and deployment budgets for VTO. To solve this obstacle, we apply Doob's h-transform efficient fine-tuning (DEFT) for adapting large pre-trained unconditional models for downstream image-conditioned VTO abilities. DEFT freezes the pre-trained model's parameters and trains a small h-transform network to learn a conditional h-transform. The h-transform network allows training only 1.42 percent of the frozen parameters, compared to a baseline of 5.52 percent in traditional parameter-efficient fine-tuning (PEFT). To further improve DEFT's performance and decrease existing models' inference time, we additionally propose an adaptive consistency loss. Consistency training distills slow but high-performing diffusion models into a fast one while retaining performance by enforcing consistencies along the inference path. Inspired by constrained optimization, instead of distillation, we combine the consistency loss and the denoising score matching loss in a data-adaptive manner for fine-tuning existing VTO models at a low cost. Empirical results show the proposed DEFT-VTON method achieves state-of-the-art performance on VTO tasks, with as few as 15 denoising steps, while maintaining competitive results.
>
---
#### [new 040] AD-DINOv3: Enhancing DINOv3 for Zero-Shot Anomaly Detection with Anomaly-Aware Calibration
- **分类: cs.CV**

- **简介: 该论文提出AD-DINOv3框架，用于零样本异常检测（ZSAD）。针对DINOv3在ZSAD中的领域偏差和语义偏置问题，引入跨模态对比学习与异常感知校准模块，提升模型对细微异常的识别能力，实验表明其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.14084v1](http://arxiv.org/pdf/2509.14084v1)**

> **作者:** Jingyi Yuan; Jianxiong Ye; Wenkang Chen; Chenqiang Gao
>
> **摘要:** Zero-Shot Anomaly Detection (ZSAD) seeks to identify anomalies from arbitrary novel categories, offering a scalable and annotation-efficient solution. Traditionally, most ZSAD works have been based on the CLIP model, which performs anomaly detection by calculating the similarity between visual and text embeddings. Recently, vision foundation models such as DINOv3 have demonstrated strong transferable representation capabilities. In this work, we are the first to adapt DINOv3 for ZSAD. However, this adaptation presents two key challenges: (i) the domain bias between large-scale pretraining data and anomaly detection tasks leads to feature misalignment; and (ii) the inherent bias toward global semantics in pretrained representations often leads to subtle anomalies being misinterpreted as part of the normal foreground objects, rather than being distinguished as abnormal regions. To overcome these challenges, we introduce AD-DINOv3, a novel vision-language multimodal framework designed for ZSAD. Specifically, we formulate anomaly detection as a multimodal contrastive learning problem, where DINOv3 is employed as the visual backbone to extract patch tokens and a CLS token, and the CLIP text encoder provides embeddings for both normal and abnormal prompts. To bridge the domain gap, lightweight adapters are introduced in both modalities, enabling their representations to be recalibrated for the anomaly detection task. Beyond this baseline alignment, we further design an Anomaly-Aware Calibration Module (AACM), which explicitly guides the CLS token to attend to anomalous regions rather than generic foreground semantics, thereby enhancing discriminability. Extensive experiments on eight industrial and medical benchmarks demonstrate that AD-DINOv3 consistently matches or surpasses state-of-the-art methods, verifying its superiority as a general zero-shot anomaly detection framework.
>
---
#### [new 041] AdaThinkDrive: Adaptive Thinking via Reinforcement Learning for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文提出AdaThinkDrive，一种结合强化学习的自适应推理框架，用于自动驾驶。旨在解决传统CoT在简单场景中效率低的问题，通过双模式推理机制提升决策质量与效率。**

- **链接: [http://arxiv.org/pdf/2509.13769v1](http://arxiv.org/pdf/2509.13769v1)**

> **作者:** Yuechen Luo; Fang Li; Shaoqing Xu; Zhiyi Lai; Lei Yang; Qimao Chen; Ziang Luo; Zixun Xie; Shengyin Jiang; Jiaxin Liu; Long Chen; Bing Wang; Zhi-xin Yang
>
> **摘要:** While reasoning technology like Chain of Thought (CoT) has been widely adopted in Vision Language Action (VLA) models, it demonstrates promising capabilities in end to end autonomous driving. However, recent efforts to integrate CoT reasoning often fall short in simple scenarios, introducing unnecessary computational overhead without improving decision quality. To address this, we propose AdaThinkDrive, a novel VLA framework with a dual mode reasoning mechanism inspired by fast and slow thinking. First, our framework is pretrained on large scale autonomous driving (AD) scenarios using both question answering (QA) and trajectory datasets to acquire world knowledge and driving commonsense. During supervised fine tuning (SFT), we introduce a two mode dataset, fast answering (w/o CoT) and slow thinking (with CoT), enabling the model to distinguish between scenarios that require reasoning. Furthermore, an Adaptive Think Reward strategy is proposed in conjunction with the Group Relative Policy Optimization (GRPO), which rewards the model for selectively applying CoT by comparing trajectory quality across different reasoning modes. Extensive experiments on the Navsim benchmark show that AdaThinkDrive achieves a PDMS of 90.3, surpassing the best vision only baseline by 1.7 points. Moreover, ablations show that AdaThinkDrive surpasses both the never Think and always Think baselines, improving PDMS by 2.0 and 1.4, respectively. It also reduces inference time by 14% compared to the always Think baseline, demonstrating its ability to balance accuracy and efficiency through adaptive reasoning.
>
---
#### [new 042] MARS2 2025 Challenge on Multimodal Reasoning: Datasets, Methods, Results, Discussion, and Outlook
- **分类: cs.CV**

- **简介: 该论文介绍MARS2 2025多模态推理挑战赛，旨在推动多模态学习与大语言模型的发展。论文发布了Lens和AdsQA数据集，评估了40+基线模型，并设立了三个竞赛赛道，促进多模态推理在现实与广告场景中的应用。**

- **链接: [http://arxiv.org/pdf/2509.14142v1](http://arxiv.org/pdf/2509.14142v1)**

> **作者:** Peng Xu; Shengwu Xiong; Jiajun Zhang; Yaxiong Chen; Bowen Zhou; Chen Change Loy; David A. Clifton; Kyoung Mu Lee; Luc Van Gool; Ruiming He; Ruilin Yao; Xinwei Long; Jirui Huang; Kai Tian; Sa Yang; Yihua Shao; Jin Feng; Yue Zhong; Jiakai Zhou; Cheng Tang; Tianyu Zou; Yifang Zhang; Junming Liang; Guoyou Li; Zhaoxiang Wang; Qiang Zhou; Yichen Zhao; Shili Xiong; Hyeongjin Nam; Jaerin Lee; Jaeyoung Chung; JoonKyu Park; Junghun Oh; Kanggeon Lee; Wooseok Lee; Juneyoung Ro; Turghun Osman; Can Hu; Chaoyang Liao; Cheng Chen; Chengcheng Han; Chenhao Qiu; Chong Peng; Cong Xu; Dailin Li; Feiyu Wang; Feng Gao; Guibo Zhu; Guopeng Tang; Haibo Lu; Han Fang; Han Qi; Hanxiao Wu; Haobo Cheng; Hongbo Sun; Hongyao Chen; Huayong Hu; Hui Li; Jiaheng Ma; Jiang Yu; Jianing Wang; Jie Yang; Jing He; Jinglin Zhou; Jingxuan Li; Josef Kittler; Lihao Zheng; Linnan Zhao; Mengxi Jia; Muyang Yan; Nguyen Thanh Thien; Pu Luo; Qi Li; Shien Song; Shijie Dong; Shuai Shao; Shutao Li; Taofeng Xue; Tianyang Xu; Tianyi Gao; Tingting Li; Wei Zhang; Weiyang Su; Xiaodong Dong; Xiao-Jun Wu; Xiaopeng Zhou; Xin Chen; Xin Wei; Xinyi You; Xudong Kang; Xujie Zhou; Xusheng Liu; Yanan Wang; Yanbin Huang; Yang Liu; Yang Yang; Yanglin Deng; Yashu Kang; Ye Yuan; Yi Wen; Yicen Tian; Yilin Tao; Yin Tang; Yipeng Lin; Yiqing Wang; Yiting Xi; Yongkang Yu; Yumei Li; Yuxin Qin; Yuying Chen; Yuzhe Cen; Zhaofan Zou; Zhaohong Liu; Zhehao Shen; Zhenglin Du; Zhengyang Li; Zhenni Huang; Zhenwei Shao; Zhilong Song; Zhiyong Feng; Zhiyu Wang; Zhou Yu; Ziang Li; Zihan Zhai; Zijian Zhang; Ziyang Peng; Ziyun Xiao; Zongshu Li
>
> **备注:** ICCV 2025 MARS2 Workshop and Challenge "Multimodal Reasoning and Slow Thinking in the Large Model Era: Towards System 2 and Beyond''
>
> **摘要:** This paper reviews the MARS2 2025 Challenge on Multimodal Reasoning. We aim to bring together different approaches in multimodal machine learning and LLMs via a large benchmark. We hope it better allows researchers to follow the state-of-the-art in this very dynamic area. Meanwhile, a growing number of testbeds have boosted the evolution of general-purpose large language models. Thus, this year's MARS2 focuses on real-world and specialized scenarios to broaden the multimodal reasoning applications of MLLMs. Our organizing team released two tailored datasets Lens and AdsQA as test sets, which support general reasoning in 12 daily scenarios and domain-specific reasoning in advertisement videos, respectively. We evaluated 40+ baselines that include both generalist MLLMs and task-specific models, and opened up three competition tracks, i.e., Visual Grounding in Real-world Scenarios (VG-RS), Visual Question Answering with Spatial Awareness (VQA-SA), and Visual Reasoning in Creative Advertisement Videos (VR-Ads). Finally, 76 teams from the renowned academic and industrial institutions have registered and 40+ valid submissions (out of 1200+) have been included in our ranking lists. Our datasets, code sets (40+ baselines and 15+ participants' methods), and rankings are publicly available on the MARS2 workshop website and our GitHub organization page https://github.com/mars2workshop/, where our updates and announcements of upcoming events will be continuously provided.
>
---
#### [new 043] Adversarial Appearance Learning in Augmented Cityscapes for Pedestrian Recognition in Autonomous Driving
- **分类: cs.CV**

- **简介: 论文提出一种对抗性外观学习方法，用于增强Cityscapes数据集中的行人识别。通过生成虚拟行人和光照条件，缩小合成与真实数据间的域差距，提升自动驾驶中行人识别的语义和实例分割性能。属于计算机视觉与自动驾驶任务。**

- **链接: [http://arxiv.org/pdf/2509.13507v1](http://arxiv.org/pdf/2509.13507v1)**

> **作者:** Artem Savkin; Thomas Lapotre; Kevin Strauss; Uzair Akbar; Federico Tombari
>
> **摘要:** In the autonomous driving area synthetic data is crucial for cover specific traffic scenarios which autonomous vehicle must handle. This data commonly introduces domain gap between synthetic and real domains. In this paper we deploy data augmentation to generate custom traffic scenarios with VRUs in order to improve pedestrian recognition. We provide a pipeline for augmentation of the Cityscapes dataset with virtual pedestrians. In order to improve augmentation realism of the pipeline we reveal a novel generative network architecture for adversarial learning of the data-set lighting conditions. We also evaluate our approach on the tasks of semantic and instance segmentation.
>
---
#### [new 044] Generative Image Coding with Diffusion Prior
- **分类: cs.CV**

- **简介: 该论文提出一种基于扩散先验的生成图像编码框架，旨在提升低比特率下的视觉保真度与压缩效率。通过预优化编码器与轻量适配器结合，实现对AI生成内容的高效压缩，并提升通用性与重建质量。**

- **链接: [http://arxiv.org/pdf/2509.13768v1](http://arxiv.org/pdf/2509.13768v1)**

> **作者:** Jianhui Chang
>
> **摘要:** As generative technologies advance, visual content has evolved into a complex mix of natural and AI-generated images, driving the need for more efficient coding techniques that prioritize perceptual quality. Traditional codecs and learned methods struggle to maintain subjective quality at high compression ratios, while existing generative approaches face challenges in visual fidelity and generalization. To this end, we propose a novel generative coding framework leveraging diffusion priors to enhance compression performance at low bitrates. Our approach employs a pre-optimized encoder to generate generalized compressed-domain representations, integrated with the pretrained model's internal features via a lightweight adapter and an attentive fusion module. This framework effectively leverages existing pretrained diffusion models and enables efficient adaptation to different pretrained models for new requirements with minimal retraining costs. We also introduce a distribution renormalization method to further enhance reconstruction fidelity. Extensive experiments show that our method (1) outperforms existing methods in visual fidelity across low bitrates, (2) improves compression performance by up to 79% over H.266/VVC, and (3) offers an efficient solution for AI-generated content while being adaptable to broader content types.
>
---
#### [new 045] SpecDiff: Accelerating Diffusion Model Inference with Self-Speculation
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出SpecDiff，一种基于自推测信息的多级特征缓存策略，用于加速扩散模型推理。通过结合未来与历史信息，解决速度与精度的权衡问题，提升推理效率。属于扩散模型加速任务。**

- **链接: [http://arxiv.org/pdf/2509.13848v1](http://arxiv.org/pdf/2509.13848v1)**

> **作者:** Jiayi Pan; Jiaming Xu; Yongkang Zhou; Guohao Dai
>
> **摘要:** Feature caching has recently emerged as a promising method for diffusion model acceleration. It effectively alleviates the inefficiency problem caused by high computational requirements by caching similar features in the inference process of the diffusion model. In this paper, we analyze existing feature caching methods from the perspective of information utilization, and point out that relying solely on historical information will lead to constrained accuracy and speed performance. And we propose a novel paradigm that introduces future information via self-speculation based on the information similarity at the same time step across different iteration times. Based on this paradigm, we present \textit{SpecDiff}, a training-free multi-level feature caching strategy including a cached feature selection algorithm and a multi-level feature classification algorithm. (1) Feature selection algorithm based on self-speculative information. \textit{SpecDiff} determines a dynamic importance score for each token based on self-speculative information and historical information, and performs cached feature selection through the importance score. (2) Multi-level feature classification algorithm based on feature importance scores. \textit{SpecDiff} classifies tokens by leveraging the differences in feature importance scores and introduces a multi-level feature calculation strategy. Extensive experiments show that \textit{SpecDiff} achieves average 2.80 \times, 2.74 \times , and 3.17\times speedup with negligible quality loss in Stable Diffusion 3, 3.5, and FLUX compared to RFlow on NVIDIA A800-80GB GPU. By merging speculative and historical information, \textit{SpecDiff} overcomes the speedup-accuracy trade-off bottleneck, pushing the Pareto frontier of speedup and accuracy in the efficient diffusion model inference.
>
---
#### [new 046] StyleProtect: Safeguarding Artistic Identity in Fine-tuned Diffusion Models
- **分类: cs.CV**

- **简介: 论文提出StyleProtect方法，用于保护艺术作品在微调扩散模型中的风格不被恶意模仿。通过更新特定的交叉注意力层，实现高效轻量的风格防御，防止艺术风格被高保真复制。**

- **链接: [http://arxiv.org/pdf/2509.13711v1](http://arxiv.org/pdf/2509.13711v1)**

> **作者:** Qiuyu Tang; Joshua Krinsky; Aparna Bharati
>
> **摘要:** The rapid advancement of generative models, particularly diffusion-based approaches, has inadvertently facilitated their potential for misuse. Such models enable malicious exploiters to replicate artistic styles that capture an artist's creative labor, personal vision, and years of dedication in an inexpensive manner. This has led to a rise in the need and exploration of methods for protecting artworks against style mimicry. Although generic diffusion models can easily mimic an artistic style, finetuning amplifies this capability, enabling the model to internalize and reproduce the style with higher fidelity and control. We hypothesize that certain cross-attention layers exhibit heightened sensitivity to artistic styles. Sensitivity is measured through activation strengths of attention layers in response to style and content representations, and assessing their correlations with features extracted from external models. Based on our findings, we introduce an efficient and lightweight protection strategy, StyleProtect, that achieves effective style defense against fine-tuned diffusion models by updating only selected cross-attention layers. Our experiments utilize a carefully curated artwork dataset based on WikiArt, comprising representative works from 30 artists known for their distinctive and influential styles and cartoon animations from the Anita dataset. The proposed method demonstrates promising performance in safeguarding unique styles of artworks and anime from malicious diffusion customization, while maintaining competitive imperceptibility.
>
---
#### [new 047] VSE-MOT: Multi-Object Tracking in Low-Quality Video Scenes Guided by Visual Semantic Enhancement
- **分类: cs.CV**

- **简介: 该论文属于多目标跟踪任务，旨在解决低质量视频中跟踪性能下降的问题。提出VSE-MOT框架，通过视觉语义增强与适配模块提升跟踪效果，实验表明其在低质量场景下优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.14060v1](http://arxiv.org/pdf/2509.14060v1)**

> **作者:** Jun Du; Weiwei Xing; Ming Li; Fei Richard Yu
>
> **摘要:** Current multi-object tracking (MOT) algorithms typically overlook issues inherent in low-quality videos, leading to significant degradation in tracking performance when confronted with real-world image deterioration. Therefore, advancing the application of MOT algorithms in real-world low-quality video scenarios represents a critical and meaningful endeavor. To address the challenges posed by low-quality scenarios, inspired by vision-language models, this paper proposes a Visual Semantic Enhancement-guided Multi-Object Tracking framework (VSE-MOT). Specifically, we first design a tri-branch architecture that leverages a vision-language model to extract global visual semantic information from images and fuse it with query vectors. Subsequently, to further enhance the utilization of visual semantic information, we introduce the Multi-Object Tracking Adapter (MOT-Adapter) and the Visual Semantic Fusion Module (VSFM). The MOT-Adapter adapts the extracted global visual semantic information to suit multi-object tracking tasks, while the VSFM improves the efficacy of feature fusion. Through extensive experiments, we validate the effectiveness and superiority of the proposed method in real-world low-quality video scenarios. Its tracking performance metrics outperform those of existing methods by approximately 8% to 20%, while maintaining robust performance in conventional scenarios.
>
---
#### [new 048] Taylor-Series Expanded Kolmogorov-Arnold Network for Medical Imaging Classification
- **分类: cs.CV**

- **简介: 该论文提出基于样条函数的Kolmogorov-Arnold网络（SBTAYLOR-KAN等）用于医疗图像分类任务，解决小数据集和资源受限环境下的准确性和可解释性问题。模型参数少、性能高，适用于临床AI场景。**

- **链接: [http://arxiv.org/pdf/2509.13687v1](http://arxiv.org/pdf/2509.13687v1)**

> **作者:** Kaniz Fatema; Emad A. Mohammed; Sukhjit Singh Sehra
>
> **摘要:** Effective and interpretable classification of medical images is a challenge in computer-aided diagnosis, especially in resource-limited clinical settings. This study introduces spline-based Kolmogorov-Arnold Networks (KANs) for accurate medical image classification with limited, diverse datasets. The models include SBTAYLOR-KAN, integrating B-splines with Taylor series; SBRBF-KAN, combining B-splines with Radial Basis Functions; and SBWAVELET-KAN, embedding B-splines in Morlet wavelet transforms. These approaches leverage spline-based function approximation to capture both local and global nonlinearities. The models were evaluated on brain MRI, chest X-rays, tuberculosis X-rays, and skin lesion images without preprocessing, demonstrating the ability to learn directly from raw data. Extensive experiments, including cross-dataset validation and data reduction analysis, showed strong generalization and stability. SBTAYLOR-KAN achieved up to 98.93% accuracy, with a balanced F1-score, maintaining over 86% accuracy using only 30% of the training data across three datasets. Despite class imbalance in the skin cancer dataset, experiments on both imbalanced and balanced versions showed SBTAYLOR-KAN outperforming other models, achieving 68.22% accuracy. Unlike traditional CNNs, which require millions of parameters (e.g., ResNet50 with 24.18M), SBTAYLOR-KAN achieves comparable performance with just 2,872 trainable parameters, making it more suitable for constrained medical environments. Gradient-weighted Class Activation Mapping (Grad-CAM) was used for interpretability, highlighting relevant regions in medical images. This framework provides a lightweight, interpretable, and generalizable solution for medical image classification, addressing the challenges of limited datasets and data-scarce scenarios in clinical AI applications.
>
---
#### [new 049] Where Do Tokens Go? Understanding Pruning Behaviors in STEP at High Resolutions
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对高分辨率语义分割中Vision Transformer计算成本高的问题，提出STEP框架，结合动态超块合并与早期剪枝技术，在保持精度的前提下显著提升效率。**

- **链接: [http://arxiv.org/pdf/2509.14165v1](http://arxiv.org/pdf/2509.14165v1)**

> **作者:** Michal Szczepanski; Martyna Poreba; Karim Haroun
>
> **摘要:** Vision Transformers (ViTs) achieve state-of-the-art performance in semantic segmentation but are hindered by high computational and memory costs. To address this, we propose STEP (SuperToken and Early-Pruning), a hybrid token-reduction framework that combines dynamic patch merging and token pruning to enhance efficiency without significantly compromising accuracy. At the core of STEP is dCTS, a lightweight CNN-based policy network that enables flexible merging into superpatches. Encoder blocks integrate also early-exits to remove high-confident supertokens, lowering computational load. We evaluate our method on high-resolution semantic segmentation benchmarks, including images up to 1024 x 1024, and show that when dCTS is applied alone, the token count can be reduced by a factor of 2.5 compared to the standard 16 x 16 pixel patching scheme. This yields a 2.6x reduction in computational cost and a 3.4x increase in throughput when using ViT-Large as the backbone. Applying the full STEP framework further improves efficiency, reaching up to a 4x reduction in computational complexity and a 1.7x gain in inference speed, with a maximum accuracy drop of no more than 2.0%. With the proposed STEP configurations, up to 40% of tokens can be confidently predicted and halted before reaching the final encoder layer.
>
---
#### [new 050] Curvature as a tool for evaluating dimensionality reduction and estimating intrinsic dimension
- **分类: cs.CV; cs.DM; cs.LG; 51K05 (primary) 57-08, 53Z50, 55U10 (secondary); G.2.2**

- **简介: 论文提出利用截面曲率评估降维效果并估计数据集的内在维度。通过构建曲率几何特征，量化评价数据表示质量，并用于分析网络结构和降维方法的有效性。属于数据表示评估与内在维度估计任务。**

- **链接: [http://arxiv.org/pdf/2509.13385v1](http://arxiv.org/pdf/2509.13385v1)**

> **作者:** Charlotte Beylier; Parvaneh Joharinad; Jürgen Jost; Nahid Torbati
>
> **备注:** 31 pages, 14 figures
>
> **摘要:** Utilizing recently developed abstract notions of sectional curvature, we introduce a method for constructing a curvature-based geometric profile of discrete metric spaces. The curvature concept that we use here captures the metric relations between triples of points and other points. More significantly, based on this curvature profile, we introduce a quantitative measure to evaluate the effectiveness of data representations, such as those produced by dimensionality reduction techniques. Furthermore, Our experiments demonstrate that this curvature-based analysis can be employed to estimate the intrinsic dimensionality of datasets. We use this to explore the large-scale geometry of empirical networks and to evaluate the effectiveness of dimensionality reduction techniques.
>
---
#### [new 051] Performance Optimization of YOLO-FEDER FusionNet for Robust Drone Detection in Visually Complex Environments
- **分类: cs.CV**

- **简介: 该论文针对复杂视觉环境下无人机检测难题，提出改进的YOLO-FEDER FusionNet框架。通过优化训练数据、特征融合与主干网络设计，显著提升检测性能，降低漏检率并提高准确率。**

- **链接: [http://arxiv.org/pdf/2509.14012v1](http://arxiv.org/pdf/2509.14012v1)**

> **作者:** Tamara R. Lenhard; Andreas Weinmann; Tobias Koch
>
> **摘要:** Drone detection in visually complex environments remains challenging due to background clutter, small object scale, and camouflage effects. While generic object detectors like YOLO exhibit strong performance in low-texture scenes, their effectiveness degrades in cluttered environments with low object-background separability. To address these limitations, this work presents an enhanced iteration of YOLO-FEDER FusionNet -- a detection framework that integrates generic object detection with camouflage object detection techniques. Building upon the original architecture, the proposed iteration introduces systematic advancements in training data composition, feature fusion strategies, and backbone design. Specifically, the training process leverages large-scale, photo-realistic synthetic data, complemented by a small set of real-world samples, to enhance robustness under visually complex conditions. The contribution of intermediate multi-scale FEDER features is systematically evaluated, and detection performance is comprehensively benchmarked across multiple YOLO-based backbone configurations. Empirical results indicate that integrating intermediate FEDER features, in combination with backbone upgrades, contributes to notable performance improvements. In the most promising configuration -- YOLO-FEDER FusionNet with a YOLOv8l backbone and FEDER features derived from the DWD module -- these enhancements lead to a FNR reduction of up to 39.1 percentage points and a mAP increase of up to 62.8 percentage points at an IoU threshold of 0.5, compared to the initial baseline.
>
---
#### [new 052] BWCache: Accelerating Video Diffusion Transformers through Block-Wise Caching
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出BWCache方法，用于加速视频扩散Transformer（DiT）生成。针对DiT推理延迟问题，通过动态缓存和重用中间特征，减少冗余计算，在保持视觉质量前提下实现最高2.24倍加速。属于视频生成加速任务。**

- **链接: [http://arxiv.org/pdf/2509.13789v1](http://arxiv.org/pdf/2509.13789v1)**

> **作者:** Hanshuai Cui; Zhiqing Tang; Zhifei Xu; Zhi Yao; Wenyi Zeng; Weijia Jia
>
> **摘要:** Recent advancements in Diffusion Transformers (DiTs) have established them as the state-of-the-art method for video generation. However, their inherently sequential denoising process results in inevitable latency, limiting real-world applicability. Existing acceleration methods either compromise visual quality due to architectural modifications or fail to reuse intermediate features at proper granularity. Our analysis reveals that DiT blocks are the primary contributors to inference latency. Across diffusion timesteps, the feature variations of DiT blocks exhibit a U-shaped pattern with high similarity during intermediate timesteps, which suggests substantial computational redundancy. In this paper, we propose Block-Wise Caching (BWCache), a training-free method to accelerate DiT-based video generation. BWCache dynamically caches and reuses features from DiT blocks across diffusion timesteps. Furthermore, we introduce a similarity indicator that triggers feature reuse only when the differences between block features at adjacent timesteps fall below a threshold, thereby minimizing redundant computations while maintaining visual fidelity. Extensive experiments on several video diffusion models demonstrate that BWCache achieves up to 2.24$\times$ speedup with comparable visual quality.
>
---
#### [new 053] Masked Feature Modeling Enhances Adaptive Segmentation
- **分类: cs.CV**

- **简介: 论文提出Masked Feature Modeling（MFM）方法，用于提升无监督域适应语义分割性能。通过在特征空间进行掩码重建，与主任务紧密耦合，无需修改推理流程，有效增强模型适应能力。**

- **链接: [http://arxiv.org/pdf/2509.13801v1](http://arxiv.org/pdf/2509.13801v1)**

> **作者:** Wenlve Zhou; Zhiheng Zhou; Tiantao Xian; Yikui Zhai; Weibin Wu; Biyun Ma
>
> **摘要:** Unsupervised domain adaptation (UDA) for semantic segmentation aims to transfer models from a labeled source domain to an unlabeled target domain. While auxiliary self-supervised tasks-particularly contrastive learning-have improved feature discriminability, masked modeling approaches remain underexplored in this setting, largely due to architectural incompatibility and misaligned optimization objectives. We propose Masked Feature Modeling (MFM), a novel auxiliary task that performs feature masking and reconstruction directly in the feature space. Unlike existing masked modeling methods that reconstruct low-level inputs or perceptual features (e.g., HOG or visual tokens), MFM aligns its learning target with the main segmentation task, ensuring compatibility with standard architectures like DeepLab and DAFormer without modifying the inference pipeline. To facilitate effective reconstruction, we introduce a lightweight auxiliary module, Rebuilder, which is trained jointly but discarded during inference, adding zero computational overhead at test time. Crucially, MFM leverages the segmentation decoder to classify the reconstructed features, tightly coupling the auxiliary objective with the pixel-wise prediction task to avoid interference with the primary task. Extensive experiments across various architectures and UDA benchmarks demonstrate that MFM consistently enhances segmentation performance, offering a simple, efficient, and generalizable strategy for unsupervised domain-adaptive semantic segmentation.
>
---
#### [new 054] Distractor-Aware Memory-Based Visual Object Tracking
- **分类: cs.CV**

- **简介: 该论文属于视觉目标跟踪任务，旨在解决跟踪过程中因干扰物导致的漂移问题。提出DAM4SAM模块并构建DiDi数据集，提升SAM2模型的跟踪性能，实验证明其在多个基准上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.13864v1](http://arxiv.org/pdf/2509.13864v1)**

> **作者:** Jovana Videnovic; Matej Kristan; Alan Lukezic
>
> **备注:** Code available on Github: https://github.com/jovanavidenovic/DAM4SAM
>
> **摘要:** Recent emergence of memory-based video segmentation methods such as SAM2 has led to models with excellent performance in segmentation tasks, achieving leading results on numerous benchmarks. However, these modes are not fully adjusted for visual object tracking, where distractors (i.e., objects visually similar to the target) pose a key challenge. In this paper we propose a distractor-aware drop-in memory module and introspection-based management method for SAM2, leading to DAM4SAM. Our design effectively reduces the tracking drift toward distractors and improves redetection capability after object occlusion. To facilitate the analysis of tracking in the presence of distractors, we construct DiDi, a Distractor-Distilled dataset. DAM4SAM outperforms SAM2.1 on thirteen benchmarks and sets new state-of-the-art results on ten. Furthermore, integrating the proposed distractor-aware memory into a real-time tracker EfficientTAM leads to 11% improvement and matches tracking quality of the non-real-time SAM2.1-L on multiple tracking and segmentation benchmarks, while integration with edge-based tracker EdgeTAM delivers 4% performance boost, demonstrating a very good generalization across architectures.
>
---
#### [new 055] MINGLE: VLMs for Semantically Complex Region Detection in Urban Scenes
- **分类: cs.CV; cs.CY**

- **简介: 论文提出MINGLE模型，用于检测城市场景中语义复杂的社交群体区域。任务是识别图像中基于人际互动的群体区域，解决传统目标检测无法处理的社交关系问题。工作包括设计三阶段流程、构建新数据集，促进相关研究。**

- **链接: [http://arxiv.org/pdf/2509.13484v1](http://arxiv.org/pdf/2509.13484v1)**

> **作者:** Liu Liu; Alexandra Kudaeva; Marco Cipriano; Fatimeh Al Ghannam; Freya Tan; Gerard de Melo; Andres Sevtsuk
>
> **备注:** 13 pages, 4 figures, under review at AAAI 2026
>
> **摘要:** Understanding group-level social interactions in public spaces is crucial for urban planning, informing the design of socially vibrant and inclusive environments. Detecting such interactions from images involves interpreting subtle visual cues such as relations, proximity, and co-movement - semantically complex signals that go beyond traditional object detection. To address this challenge, we introduce a social group region detection task, which requires inferring and spatially grounding visual regions defined by abstract interpersonal relations. We propose MINGLE (Modeling INterpersonal Group-Level Engagement), a modular three-stage pipeline that integrates: (1) off-the-shelf human detection and depth estimation, (2) VLM-based reasoning to classify pairwise social affiliation, and (3) a lightweight spatial aggregation algorithm to localize socially connected groups. To support this task and encourage future research, we present a new dataset of 100K urban street-view images annotated with bounding boxes and labels for both individuals and socially interacting groups. The annotations combine human-created labels and outputs from the MINGLE pipeline, ensuring semantic richness and broad coverage of real-world scenarios.
>
---
#### [new 056] Towards Robust Defense against Customization via Protective Perturbation Resistant to Diffusion-based Purification
- **分类: cs.CV**

- **简介: 该论文提出一种名为AntiPure的保护性扰动方法，用于对抗扩散模型的净化过程。其任务是增强图像定制后的鲁棒性，防止恶意篡改。通过频率和时间步指导机制，使扰动在净化后仍有效，提升防御能力。**

- **链接: [http://arxiv.org/pdf/2509.13922v1](http://arxiv.org/pdf/2509.13922v1)**

> **作者:** Wenkui Yang; Jie Cao; Junxian Duan; Ran He
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Diffusion models like Stable Diffusion have become prominent in visual synthesis tasks due to their powerful customization capabilities, which also introduce significant security risks, including deepfakes and copyright infringement. In response, a class of methods known as protective perturbation emerged, which mitigates image misuse by injecting imperceptible adversarial noise. However, purification can remove protective perturbations, thereby exposing images again to the risk of malicious forgery. In this work, we formalize the anti-purification task, highlighting challenges that hinder existing approaches, and propose a simple diagnostic protective perturbation named AntiPure. AntiPure exposes vulnerabilities of purification within the "purification-customization" workflow, owing to two guidance mechanisms: 1) Patch-wise Frequency Guidance, which reduces the model's influence over high-frequency components in the purified image, and 2) Erroneous Timestep Guidance, which disrupts the model's denoising strategy across different timesteps. With additional guidance, AntiPure embeds imperceptible perturbations that persist under representative purification settings, achieving effective post-customization distortion. Experiments show that, as a stress test for purification, AntiPure achieves minimal perceptual discrepancy and maximal distortion, outperforming other protective perturbation methods within the purification-customization workflow.
>
---
#### [new 057] BiasMap: Leveraging Cross-Attentions to Discover and Mitigate Hidden Social Biases in Text-to-Image Generation
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出BiasMap框架，用于发现并缓解文本到图像生成中的隐藏社会偏见。通过分析交叉注意力图，量化概念间纠缠，并利用能量引导扩散采样进行偏见缓解，解决现有方法无法解耦概念级关联的问题。属于生成模型公平性研究任务。**

- **链接: [http://arxiv.org/pdf/2509.13496v1](http://arxiv.org/pdf/2509.13496v1)**

> **作者:** Rajatsubhra Chakraborty; Xujun Che; Depeng Xu; Cori Faklaris; Xi Niu; Shuhan Yuan
>
> **摘要:** Bias discovery is critical for black-box generative models, especiall text-to-image (TTI) models. Existing works predominantly focus on output-level demographic distributions, which do not necessarily guarantee concept representations to be disentangled post-mitigation. We propose BiasMap, a model-agnostic framework for uncovering latent concept-level representational biases in stable diffusion models. BiasMap leverages cross-attention attribution maps to reveal structural entanglements between demographics (e.g., gender, race) and semantics (e.g., professions), going deeper into representational bias during the image generation. Using attribution maps of these concepts, we quantify the spatial demographics-semantics concept entanglement via Intersection over Union (IoU), offering a lens into bias that remains hidden in existing fairness discovery approaches. In addition, we further utilize BiasMap for bias mitigation through energy-guided diffusion sampling that directly modifies latent noise space and minimizes the expected SoftIoU during the denoising process. Our findings show that existing fairness interventions may reduce the output distributional gap but often fail to disentangle concept-level coupling, whereas our mitigation method can mitigate concept entanglement in image generation while complementing distributional bias mitigation.
>
---
#### [new 058] CETUS: Causal Event-Driven Temporal Modeling With Unified Variable-Rate Scheduling
- **分类: cs.CV**

- **简介: 该论文提出CETUS模型，用于处理事件相机数据。针对现有方法引入窗口延迟和计算复杂度高的问题，设计轻量空间编码器与Mamba状态空间模型，实现无中间表示的高效时序建模，并自适应调整处理速度以平衡延迟与效率。**

- **链接: [http://arxiv.org/pdf/2509.13784v1](http://arxiv.org/pdf/2509.13784v1)**

> **作者:** Hanfang Liang; Bing Wang; Shizhen Zhang; Wen Jiang; Yizhuo Yang; Weixiang Guo; Shenghai Yuan
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Event cameras capture asynchronous pixel-level brightness changes with microsecond temporal resolution, offering unique advantages for high-speed vision tasks. Existing methods often convert event streams into intermediate representations such as frames, voxel grids, or point clouds, which inevitably require predefined time windows and thus introduce window latency. Meanwhile, pointwise detection methods face computational challenges that prevent real-time efficiency due to their high computational cost. To overcome these limitations, we propose the Variable-Rate Spatial Event Mamba, a novel architecture that directly processes raw event streams without intermediate representations. Our method introduces a lightweight causal spatial neighborhood encoder to efficiently capture local geometric relations, followed by Mamba-based state space models for scalable temporal modeling with linear complexity. During inference, a controller adaptively adjusts the processing speed according to the event rate, achieving an optimal balance between window latency and inference latency.
>
---
#### [new 059] A Generalization of CLAP from 3D Localization to Image Processing, A Connection With RANSAC & Hough Transforms
- **分类: cs.CV; cs.RO**

- **简介: 论文将CLAP算法从2D定位推广到3D定位与图像拼接，用于处理噪声和不确定性。该工作提出了一种基于聚类的鲁棒方法，并探讨了其与RANSAC和霍夫变换的关系，适用于多个领域。**

- **链接: [http://arxiv.org/pdf/2509.13605v1](http://arxiv.org/pdf/2509.13605v1)**

> **作者:** Ruochen Hou; Gabriel I. Fernandez; Alex Xu; Dennis W. Hong
>
> **摘要:** In previous work, we introduced a 2D localization algorithm called CLAP, Clustering to Localize Across $n$ Possibilities, which was used during our championship win in RoboCup 2024, an international autonomous humanoid soccer competition. CLAP is particularly recognized for its robustness against outliers, where clustering is employed to suppress noise and mitigate against erroneous feature matches. This clustering-based strategy provides an alternative to traditional outlier rejection schemes such as RANSAC, in which candidates are validated by reprojection error across all data points. In this paper, CLAP is extended to a more general framework beyond 2D localization, specifically to 3D localization and image stitching. We also show how CLAP, RANSAC, and Hough transforms are related. The generalization of CLAP is widely applicable to many different fields and can be a useful tool to deal with noise and uncertainty.
>
---
#### [new 060] Dynamic Aware: Adaptive Multi-Mode Out-of-Distribution Detection for Trajectory Prediction in Autonomous Vehicles
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶轨迹预测中的分布外检测任务，旨在解决模型在真实场景中因数据分布偏移导致的预测失效问题。提出一种自适应多模式检测框架，通过建模预测误差的动态模式，提升检测效率与准确性。**

- **链接: [http://arxiv.org/pdf/2509.13577v1](http://arxiv.org/pdf/2509.13577v1)**

> **作者:** Tongfei Guo; Lili Su
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Trajectory prediction is central to the safe and seamless operation of autonomous vehicles (AVs). In deployment, however, prediction models inevitably face distribution shifts between training data and real-world conditions, where rare or underrepresented traffic scenarios induce out-of-distribution (OOD) cases. While most prior OOD detection research in AVs has concentrated on computer vision tasks such as object detection and segmentation, trajectory-level OOD detection remains largely underexplored. A recent study formulated this problem as a quickest change detection (QCD) task, providing formal guarantees on the trade-off between detection delay and false alarms [1]. Building on this foundation, we propose a new framework that introduces adaptive mechanisms to achieve robust detection in complex driving environments. Empirical analysis across multiple real-world datasets reveals that prediction errors -- even on in-distribution samples -- exhibit mode-dependent distributions that evolve over time with dataset-specific dynamics. By explicitly modeling these error modes, our method achieves substantial improvements in both detection delay and false alarm rates. Comprehensive experiments on established trajectory prediction benchmarks show that our framework significantly outperforms prior UQ- and vision-based OOD approaches in both accuracy and computational efficiency, offering a practical path toward reliable, driving-aware autonomy.
>
---
#### [new 061] FunKAN: Functional Kolmogorov-Arnold Network for Medical Image Enhancement and Segmentation
- **分类: cs.CV; I.4.3; I.4.6**

- **简介: 该论文提出FunKAN，用于医学图像增强与分割，解决传统深度学习模型可解释性差及KAN破坏空间结构的问题。通过傅里叶分解赫米特基函数学习内函数，提升性能，优于其他KAN模型。**

- **链接: [http://arxiv.org/pdf/2509.13508v1](http://arxiv.org/pdf/2509.13508v1)**

> **作者:** Maksim Penkin; Andrey Krylov
>
> **备注:** 9 pages, 5 figures, submitted to the Fortieth AAAI Conference on Artificial Intelligence (AAAI-26)
>
> **摘要:** Medical image enhancement and segmentation are critical yet challenging tasks in modern clinical practice, constrained by artifacts and complex anatomical variations. Traditional deep learning approaches often rely on complex architectures with limited interpretability. While Kolmogorov-Arnold networks offer interpretable solutions, their reliance on flattened feature representations fundamentally disrupts the intrinsic spatial structure of imaging data. To address this issue we propose a Functional Kolmogorov-Arnold Network (FunKAN) -- a novel interpretable neural framework, designed specifically for image processing, that formally generalizes the Kolmogorov-Arnold representation theorem onto functional spaces and learns inner functions using Fourier decomposition over the basis Hermite functions. We explore FunKAN on several medical image processing tasks, including Gibbs ringing suppression in magnetic resonance images, benchmarking on IXI dataset. We also propose U-FunKAN as state-of-the-art binary medical segmentation model with benchmarks on three medical datasets: BUSI (ultrasound images), GlaS (histological structures) and CVC-ClinicDB (colonoscopy videos), detecting breast cancer, glands and polyps, respectively. Experiments on those diverse datasets demonstrate that our approach outperforms other KAN-based backbones in both medical image enhancement (PSNR, TV) and segmentation (IoU, F1). Our work bridges the gap between theoretical function approximation and medical image analysis, offering a robust, interpretable solution for clinical applications.
>
---
#### [new 062] Bridging the Synthetic-Real Gap: Supervised Domain Adaptation for Robust Spacecraft 6-DoF Pose Estimation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于航天器6自由度位姿估计任务，旨在解决合成数据与真实图像间的领域差异问题。提出一种监督领域自适应框架，结合有限的真实标注数据优化不变特征表示，显著提升模型在真实环境中的性能。**

- **链接: [http://arxiv.org/pdf/2509.13792v1](http://arxiv.org/pdf/2509.13792v1)**

> **作者:** Inder Pal Singh; Nidhal Eddine Chenni; Abd El Rahman Shabayek; Arunkumar Rathinam; Djamila Aouada
>
> **摘要:** Spacecraft Pose Estimation (SPE) is a fundamental capability for autonomous space operations such as rendezvous, docking, and in-orbit servicing. Hybrid pipelines that combine object detection, keypoint regression, and Perspective-n-Point (PnP) solvers have recently achieved strong results on synthetic datasets, yet their performance deteriorates sharply on real or lab-generated imagery due to the persistent synthetic-to-real domain gap. Existing unsupervised domain adaptation approaches aim to mitigate this issue but often underperform when a modest number of labeled target samples are available. In this work, we propose the first Supervised Domain Adaptation (SDA) framework tailored for SPE keypoint regression. Building on the Learning Invariant Representation and Risk (LIRR) paradigm, our method jointly optimizes domain-invariant representations and task-specific risk using both labeled synthetic and limited labeled real data, thereby reducing generalization error under domain shift. Extensive experiments on the SPEED+ benchmark demonstrate that our approach consistently outperforms source-only, fine-tuning, and oracle baselines. Notably, with only 5% labeled target data, our method matches or surpasses oracle performance trained on larger fractions of labeled data. The framework is lightweight, backbone-agnostic, and computationally efficient, offering a practical pathway toward robust and deployable spacecraft pose estimation in real-world space environments.
>
---
#### [new 063] SAIL-VL2 Technical Report
- **分类: cs.CV**

- **简介: 该论文提出SAIL-VL2，一种用于多模态理解和推理的视觉-语言基础模型。其通过数据优化、渐进训练和高效架构设计，提升模型性能，在多个基准测试中取得领先结果，解决多模态任务中的感知与复杂推理问题。**

- **链接: [http://arxiv.org/pdf/2509.14033v1](http://arxiv.org/pdf/2509.14033v1)**

> **作者:** Weijie Yin; Yongjie Ye; Fangxun Shu; Yue Liao; Zijian Kang; Hongyuan Dong; Haiyang Yu; Dingkang Yang; Jiacong Wang; Han Wang; Wenzhuo Liu; Xiao Liang; Shuicheng Yan; Chao Feng
>
> **备注:** Technical Report
>
> **摘要:** We introduce SAIL-VL2, an open-suite vision-language foundation model (LVM) for comprehensive multimodal understanding and reasoning. As the successor to SAIL-VL, SAIL-VL2 achieves state-of-the-art performance at the 2B and 8B parameter scales across diverse image and video benchmarks, demonstrating strong capabilities from fine-grained perception to complex reasoning. Three core innovations drive its effectiveness. First, a large-scale data curation pipeline with scoring and filtering strategies enhances both quality and distribution across captioning, OCR, QA, and video data, improving training efficiency. Second, a progressive training framework begins with a powerful pre-trained vision encoder (SAIL-ViT), advances through multimodal pre-training, and culminates in a thinking-fusion SFT-RL hybrid paradigm that systematically strengthens model capabilities. Third, architectural advances extend beyond dense LLMs to efficient sparse Mixture-of-Experts (MoE) designs. With these contributions, SAIL-VL2 demonstrates competitive performance across 106 datasets and achieves state-of-the-art results on challenging reasoning benchmarks such as MMMU and MathVista. Furthermore, on the OpenCompass leaderboard, SAIL-VL2-2B ranks first among officially released open-source models under the 4B parameter scale, while serving as an efficient and extensible foundation for the open-source multimodal community.
>
---
#### [new 064] UM-Depth : Uncertainty Masked Self-Supervised Monocular Depth Estimation with Visual Odometry
- **分类: cs.CV**

- **简介: 论文提出UM-Depth，一种结合运动与不确定性感知的自监督单目深度估计方法。旨在解决低纹理和动态区域深度不准确问题，通过引入不确定性估计和师生训练策略提升精度，无需额外标注，在KITTI数据集上取得SOTA结果。**

- **链接: [http://arxiv.org/pdf/2509.13713v1](http://arxiv.org/pdf/2509.13713v1)**

> **作者:** Tae-Wook Um; Ki-Hyeon Kim; Hyun-Duck Choi; Hyo-Sung Ahn
>
> **摘要:** Monocular depth estimation has been increasingly adopted in robotics and autonomous driving for its ability to infer scene geometry from a single camera. In self-supervised monocular depth estimation frameworks, the network jointly generates and exploits depth and pose estimates during training, thereby eliminating the need for depth labels. However, these methods remain challenged by uncertainty in the input data, such as low-texture or dynamic regions, which can cause reduced depth accuracy. To address this, we introduce UM-Depth, a framework that combines motion- and uncertainty-aware refinement to enhance depth accuracy at dynamic object boundaries and in textureless regions. Specifically, we develop a teacherstudent training strategy that embeds uncertainty estimation into both the training pipeline and network architecture, thereby strengthening supervision where photometric signals are weak. Unlike prior motion-aware approaches that incur inference-time overhead and rely on additional labels or auxiliary networks for real-time generation, our method uses optical flow exclusively within the teacher network during training, which eliminating extra labeling demands and any runtime cost. Extensive experiments on the KITTI and Cityscapes datasets demonstrate the effectiveness of our uncertainty-aware refinement. Overall, UM-Depth achieves state-of-the-art results in both self-supervised depth and pose estimation on the KITTI datasets.
>
---
#### [new 065] Can Current AI Models Count What We Mean, Not What They See? A Benchmark and Systematic Evaluation
- **分类: cs.CV**

- **简介: 该论文提出PairTally基准数据集，用于评估细粒度视觉计数任务。研究聚焦于模型能否准确理解用户意图进行计数，而非仅基于视觉特征。通过对比多种先进模型，发现当前模型在细粒度和视觉模糊场景下表现不佳，为改进系统提供基础。**

- **链接: [http://arxiv.org/pdf/2509.13939v1](http://arxiv.org/pdf/2509.13939v1)**

> **作者:** Gia Khanh Nguyen; Yifeng Huang; Minh Hoai
>
> **摘要:** Visual counting is a fundamental yet challenging task, especially when users need to count objects of a specific type in complex scenes. While recent models, including class-agnostic counting models and large vision-language models (VLMs), show promise in counting tasks, their ability to perform fine-grained, intent-driven counting remains unclear. In this paper, we introduce PairTally, a benchmark dataset specifically designed to evaluate fine-grained visual counting. Each of the 681 high-resolution images in PairTally contains two object categories, requiring models to distinguish and count based on subtle differences in shape, size, color, or semantics. The dataset includes both inter-category (distinct categories) and intra-category (closely related subcategories) settings, making it suitable for rigorous evaluation of selective counting capabilities. We benchmark a variety of state-of-the-art models, including exemplar-based methods, language-prompted models, and large VLMs. Our results show that despite recent advances, current models struggle to reliably count what users intend, especially in fine-grained and visually ambiguous cases. PairTally provides a new foundation for diagnosing and improving fine-grained visual counting systems.
>
---
#### [new 066] Mitigating Query Selection Bias in Referring Video Object Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频对象分割任务，旨在解决查询选择偏差问题。提出三元查询形成器（TQF），将查询分解为外观、帧内交互和帧间运动三个部分，并引入运动感知聚合模块，提升分割性能。**

- **链接: [http://arxiv.org/pdf/2509.13722v1](http://arxiv.org/pdf/2509.13722v1)**

> **作者:** Dingwei Zhang; Dong Zhang; Jinhui Tang
>
> **摘要:** Recently, query-based methods have achieved remarkable performance in Referring Video Object Segmentation (RVOS) by using textual static object queries to drive cross-modal alignment. However, these static queries are easily misled by distractors with similar appearance or motion, resulting in \emph{query selection bias}. To address this issue, we propose Triple Query Former (TQF), which factorizes the referring query into three specialized components: an appearance query for static attributes, an intra-frame interaction query for spatial relations, and an inter-frame motion query for temporal association. Instead of relying solely on textual embeddings, our queries are dynamically constructed by integrating both linguistic cues and visual guidance. Furthermore, we introduce two motion-aware aggregation modules that enhance object token representations: Intra-frame Interaction Aggregation incorporates position-aware interactions among objects within a single frame, while Inter-frame Motion Aggregation leverages trajectory-guided alignment across frames to ensure temporal coherence. Extensive experiments on multiple RVOS benchmarks demonstrate the advantages of TQF and the effectiveness of our structured query design and motion-aware aggregation modules.
>
---
#### [new 067] Semi-MoE: Mixture-of-Experts meets Semi-Supervised Histopathology Segmentation
- **分类: cs.CV**

- **简介: 该论文提出Semi-MoE，一种基于Mixture-of-Experts的半监督病理图像分割方法。针对标注数据少、伪标签噪声大的问题，设计多专家网络与自适应损失函数，提升分割性能。属于医学图像分割任务。**

- **链接: [http://arxiv.org/pdf/2509.13834v1](http://arxiv.org/pdf/2509.13834v1)**

> **作者:** Nguyen Lan Vi Vu; Thanh-Huy Nguyen; Thien Nguyen; Daisuke Kihara; Tianyang Wang; Xingjian Li; Min Xu
>
> **备注:** Accepted to BMVC 2025
>
> **摘要:** Semi-supervised learning has been employed to alleviate the need for extensive labeled data for histopathology image segmentation, but existing methods struggle with noisy pseudo-labels due to ambiguous gland boundaries and morphological misclassification. This paper introduces Semi-MOE, to the best of our knowledge, the first multi-task Mixture-of-Experts framework for semi-supervised histopathology image segmentation. Our approach leverages three specialized expert networks: A main segmentation expert, a signed distance field regression expert, and a boundary prediction expert, each dedicated to capturing distinct morphological features. Subsequently, the Multi-Gating Pseudo-labeling module dynamically aggregates expert features, enabling a robust fuse-and-refine pseudo-labeling mechanism. Furthermore, to eliminate manual tuning while dynamically balancing multiple learning objectives, we propose an Adaptive Multi-Objective Loss. Extensive experiments on GlaS and CRAG benchmarks show that our method outperforms state-of-the-art approaches in low-label settings, highlighting the potential of MoE-based architectures in advancing semi-supervised segmentation. Our code is available at https://github.com/vnlvi2k3/Semi-MoE.
>
---
#### [new 068] Research on Expressway Congestion Warning Technology Based on YOLOv11-DIoU and GRU-Attention
- **分类: cs.CV**

- **简介: 论文研究基于YOLOv11-DIoU和GRU-Attention的高速公路拥堵预警技术，旨在提升车辆感知精度与拥堵预测能力。通过优化目标检测与轨迹跟踪算法，并构建GRU-Attention模型实现高精度预警，解决遮挡与长序列依赖问题。**

- **链接: [http://arxiv.org/pdf/2509.13361v1](http://arxiv.org/pdf/2509.13361v1)**

> **作者:** Tong Yulin; Liang Xuechen
>
> **摘要:** Expressway traffic congestion severely reduces travel efficiency and hinders regional connectivity. Existing "detection-prediction" systems have critical flaws: low vehicle perception accuracy under occlusion and loss of long-sequence dependencies in congestion forecasting. This study proposes an integrated technical framework to resolve these issues.For traffic flow perception, two baseline algorithms were optimized. Traditional YOLOv11 was upgraded to YOLOv11-DIoU by replacing GIoU Loss with DIoU Loss, and DeepSort was improved by fusing Mahalanobis (motion) and cosine (appearance) distances. Experiments on Chang-Shen Expressway videos showed YOLOv11-DIoU achieved 95.7\% mAP (6.5 percentage points higher than baseline) with 5.3\% occlusion miss rate. DeepSort reached 93.8\% MOTA (11.3 percentage points higher than SORT) with only 4 ID switches. Using the Greenberg model (for 10-15 vehicles/km high-density scenarios), speed and density showed a strong negative correlation (r=-0.97), conforming to traffic flow theory. For congestion warning, a GRU-Attention model was built to capture congestion precursors. Trained 300 epochs with flow, density, and speed, it achieved 99.7\% test accuracy (7-9 percentage points higher than traditional GRU). In 10-minute advance warnings for 30-minute congestion, time error was $\leq$ 1 minute. Validation with an independent video showed 95\% warning accuracy, over 90\% spatial overlap of congestion points, and stable performance in high-flow ($>$5 vehicles/second) scenarios.This framework provides quantitative support for expressway congestion control, with promising intelligent transportation applications.
>
---
#### [new 069] White Aggregation and Restoration for Few-shot 3D Point Cloud Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于少样本3D点云语义分割任务，旨在利用少量标注样本对未标注点云进行分割。为解决原型生成中初始随机性影响性能的问题，提出基于注意力机制的白化聚合与恢复模块（WARM），有效提升原型表示能力，取得SOTA效果。**

- **链接: [http://arxiv.org/pdf/2509.13907v1](http://arxiv.org/pdf/2509.13907v1)**

> **作者:** Jiyun Im; SuBeen Lee; Miso Lee; Jae-Pil Heo
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Few-Shot 3D Point Cloud Segmentation (FS-PCS) aims to predict per-point labels for an unlabeled point cloud, given only a few labeled examples. To extract discriminative representations from the limited support set, existing methods have constructed prototypes using conventional algorithms such as farthest point sampling. However, we point out that its initial randomness significantly affects FS-PCS performance and that the prototype generation process remains underexplored despite its prevalence. This motivates us to investigate an advanced prototype generation method based on attention mechanism. Despite its potential, we found that vanilla module suffers from the distributional gap between learnable prototypical tokens and support features. To overcome this, we propose White Aggregation and Restoration Module (WARM), which resolves the misalignment by sandwiching cross-attention between whitening and coloring transformations. Specifically, whitening aligns the support features to prototypical tokens before attention process, and subsequently coloring restores the original distribution to the attended tokens. This simple yet effective design enables robust attention, thereby generating representative prototypes by capturing the semantic relationships among support features. Our method achieves state-of-the-art performance with a significant margin on multiple FS-PCS benchmarks, demonstrating its effectiveness through extensive experiments.
>
---
#### [new 070] LamiGauss: Pitching Radiative Gaussian for Sparse-View X-ray Laminography Reconstruction
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出LamiGauss算法，用于解决稀疏视角X射线层析成像重建问题。通过结合高斯点绘图与专用检测器变换模型，有效提升稀疏投影下的重建质量，仅用3%全视角数据即优于传统迭代方法。属于医学影像重建任务。**

- **链接: [http://arxiv.org/pdf/2509.13863v1](http://arxiv.org/pdf/2509.13863v1)**

> **作者:** Chu Chen; Ander Biguri; Jean-Michel Morel; Raymond H. Chan; Carola-Bibiane Schönlieb; Jizhou Li
>
> **摘要:** X-ray Computed Laminography (CL) is essential for non-destructive inspection of plate-like structures in applications such as microchips and composite battery materials, where traditional computed tomography (CT) struggles due to geometric constraints. However, reconstructing high-quality volumes from laminographic projections remains challenging, particularly under highly sparse-view acquisition conditions. In this paper, we propose a reconstruction algorithm, namely LamiGauss, that combines Gaussian Splatting radiative rasterization with a dedicated detector-to-world transformation model incorporating the laminographic tilt angle. LamiGauss leverages an initialization strategy that explicitly filters out common laminographic artifacts from the preliminary reconstruction, preventing redundant Gaussians from being allocated to false structures and thereby concentrating model capacity on representing the genuine object. Our approach effectively optimizes directly from sparse projections, enabling accurate and efficient reconstruction with limited data. Extensive experiments on both synthetic and real datasets demonstrate the effectiveness and superiority of the proposed method over existing techniques. LamiGauss uses only 3$\%$ of full views to achieve superior performance over the iterative method optimized on a full dataset.
>
---
#### [new 071] Towards Rationale-Answer Alignment of LVLMs via Self-Rationale Calibration
- **分类: cs.CV**

- **简介: 该论文属于视觉问答任务，旨在解决LVLMs中推理与答案不一致的问题。提出SRC框架，通过自推理校准提升模型的推理一致性与准确性。**

- **链接: [http://arxiv.org/pdf/2509.13919v1](http://arxiv.org/pdf/2509.13919v1)**

> **作者:** Yuanchen Wu; Ke Yan; Shouhong Ding; Ziyin Zhou; Xiaoqiang Li
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Large Vision-Language Models (LVLMs) have manifested strong visual question answering capability. However, they still struggle with aligning the rationale and the generated answer, leading to inconsistent reasoning and incorrect responses. To this end, this paper introduces the Self-Rationale Calibration (SRC) framework to iteratively calibrate the alignment between rationales and answers. SRC begins by employing a lightweight "rationale fine-tuning" approach, which modifies the model's response format to require a rationale before deriving an answer without explicit prompts. Next, SRC searches for a diverse set of candidate responses from the fine-tuned LVLMs for each sample, followed by a proposed pairwise scoring strategy using a tailored scoring model, R-Scorer, to evaluate both rationale quality and factual consistency of candidates. Based on a confidence-weighted preference curation process, SRC decouples the alignment calibration into a preference fine-tuning manner, leading to significant improvements of LVLMs in perception, reasoning, and generalization across multiple benchmarks. Our results emphasize the rationale-oriented alignment in exploring the potential of LVLMs.
>
---
#### [new 072] CSMoE: An Efficient Remote Sensing Foundation Model with Soft Mixture-of-Experts
- **分类: cs.CV**

- **简介: 论文提出CSMoE模型，通过引入Soft MoE机制提升遥感基础模型的效率。解决现有模型计算复杂度高、表示能力有限的问题，实现高效且性能优异的遥感表征学习。**

- **链接: [http://arxiv.org/pdf/2509.14104v1](http://arxiv.org/pdf/2509.14104v1)**

> **作者:** Leonard Hackel; Tom Burgert; Begüm Demir
>
> **摘要:** Self-supervised learning through masked autoencoders has attracted great attention for remote sensing (RS) foundation model (FM) development, enabling improved representation learning across diverse sensors and downstream tasks. However, existing RS FMs often either suffer from substantial computational complexity during both training and inference or exhibit limited representational capacity. These issues restrict their practical applicability in RS. To address this limitation, we propose an adaptation for enhancing the efficiency of RS FMs by integrating the Soft mixture-of-experts (MoE) mechanism into the FM. The integration of Soft MoEs into the FM allows modality-specific expert specialization alongside shared cross-sensor representation learning. To demonstrate the effectiveness of our adaptation, we apply it on the Cross-Sensor Masked Autoencoder (CSMAE) model, resulting in the Cross-Sensor Mixture-of-Experts (CSMoE) model. In addition, we introduce a thematic-climatic descriptor-driven sampling strategy for the construction of a representative and diverse training set to train our CSMoE model. Extensive experiments on scene classification, semantic segmentation, and content-based image retrieval demonstrate that our adaptation yields a reduction in computational requirements while maintaining or improving representational performance. Compared to state-of-the-art RS FMs, CSMoE achieves a superior trade-off between representational capacity, accuracy, and computational efficiency. On average, CSMoE achieves more than twice the computational efficiency of existing RS FMs, while maintaining competitive performance across all experiments. These results show the effectiveness of the proposed adaptation for creating computationally efficient RS FMs. The code for the model, the training set creation, and the model weights will be available at https://git.tu-berlin.de/rsim/csmoe.
>
---
#### [new 073] Cinéaste: A Fine-grained Contextual Movie Question Answering Benchmark
- **分类: cs.CV; I.2.10; I.2.7**

- **简介: 该论文提出Cinéaste基准，用于评估模型对长篇电影的细粒度叙事理解能力。针对现有数据集在长片段推理上的不足，构建包含3119道多选题的数据集，并采用两阶段过滤机制确保问题质量。实验表明当前MLLMs在此任务上表现不佳，突显了长时序推理的挑战。**

- **链接: [http://arxiv.org/pdf/2509.14227v1](http://arxiv.org/pdf/2509.14227v1)**

> **作者:** Nisarg A. Shah; Amir Ziai; Chaitanya Ekanadham; Vishal M. Patel
>
> **备注:** 11 pages, 5 figures, 5 tables
>
> **摘要:** While recent advancements in vision-language models have improved video understanding, diagnosing their capacity for deep, narrative comprehension remains a challenge. Existing benchmarks often test short-clip recognition or use template-based questions, leaving a critical gap in evaluating fine-grained reasoning over long-form narrative content. To address these gaps, we introduce $\mathsf{Cin\acute{e}aste}$, a comprehensive benchmark for long-form movie understanding. Our dataset comprises 3,119 multiple-choice question-answer pairs derived from 1,805 scenes across 200 diverse movies, spanning five novel fine-grained contextual reasoning categories. We use GPT-4o to generate diverse, context-rich questions by integrating visual descriptions, captions, scene titles, and summaries, which require deep narrative understanding. To ensure high-quality evaluation, our pipeline incorporates a two-stage filtering process: Context-Independence filtering ensures questions require video context, while Contextual Veracity filtering validates factual consistency against the movie content, mitigating hallucinations. Experiments show that existing MLLMs struggle on $\mathsf{Cin\acute{e}aste}$; our analysis reveals that long-range temporal reasoning is a primary bottleneck, with the top open-source model achieving only 63.15\% accuracy. This underscores significant challenges in fine-grained contextual understanding and the need for advancements in long-form movie comprehension.
>
---
#### [new 074] FishBEV: Distortion-Resilient Bird's Eye View Segmentation with Surround-View Fisheye Cameras
- **分类: cs.CV**

- **简介: 该论文提出FishBEV框架，解决 fisheye 相机在 BEV 分割中的畸变、多视角对应模糊和时序不稳定问题，通过三个创新模块提升性能。属于自动驾驶中的 BEV 分割任务。**

- **链接: [http://arxiv.org/pdf/2509.13681v1](http://arxiv.org/pdf/2509.13681v1)**

> **作者:** Hang Li; Dianmo Sheng; Qiankun Dong; Zichun Wang; Zhiwei Xu; Tao Li
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** As a cornerstone technique for autonomous driving, Bird's Eye View (BEV) segmentation has recently achieved remarkable progress with pinhole cameras. However, it is non-trivial to extend the existing methods to fisheye cameras with severe geometric distortion, ambiguous multi-view correspondences and unstable temporal dynamics, all of which significantly degrade BEV performance. To address these challenges, we propose FishBEV, a novel BEV segmentation framework specifically tailored for fisheye cameras. This framework introduces three complementary innovations, including a Distortion-Resilient Multi-scale Extraction (DRME) backbone that learns robust features under distortion while preserving scale consistency, an Uncertainty-aware Spatial Cross-Attention (U-SCA) mechanism that leverages uncertainty estimation for reliable cross-view alignment, a Distance-aware Temporal Self-Attention (D-TSA) module that adaptively balances near field details and far field context to ensure temporal coherence. Extensive experiments on the Synwoodscapes dataset demonstrate that FishBEV consistently outperforms SOTA baselines, regarding the performance evaluation of FishBEV on the surround-view fisheye BEV segmentation tasks.
>
---
#### [new 075] Dense Video Understanding with Gated Residual Tokenization
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视频理解任务，解决高帧率视频处理中冗余计算与信息丢失问题。提出DVU框架与GRT方法，通过运动补偿与语义融合减少token数量，提升效率，并构建DIVE基准测试密集时序推理能力。**

- **链接: [http://arxiv.org/pdf/2509.14199v1](http://arxiv.org/pdf/2509.14199v1)**

> **作者:** Haichao Zhang; Wenhao Chai; Shwai He; Ang Li; Yun Fu
>
> **摘要:** High temporal resolution is essential for capturing fine-grained details in video understanding. However, current video large language models (VLLMs) and benchmarks mostly rely on low-frame-rate sampling, such as uniform sampling or keyframe selection, discarding dense temporal information. This compromise avoids the high cost of tokenizing every frame, which otherwise leads to redundant computation and linear token growth as video length increases. While this trade-off works for slowly changing content, it fails for tasks like lecture comprehension, where information appears in nearly every frame and requires precise temporal alignment. To address this gap, we introduce Dense Video Understanding (DVU), which enables high-FPS video comprehension by reducing both tokenization time and token overhead. Existing benchmarks are also limited, as their QA pairs focus on coarse content changes. We therefore propose DIVE (Dense Information Video Evaluation), the first benchmark designed for dense temporal reasoning. To make DVU practical, we present Gated Residual Tokenization (GRT), a two-stage framework: (1) Motion-Compensated Inter-Gated Tokenization uses pixel-level motion estimation to skip static regions during tokenization, achieving sub-linear growth in token count and compute. (2) Semantic-Scene Intra-Tokenization Merging fuses tokens across static regions within a scene, further reducing redundancy while preserving dynamic semantics. Experiments on DIVE show that GRT outperforms larger VLLM baselines and scales positively with FPS. These results highlight the importance of dense temporal information and demonstrate that GRT enables efficient, scalable high-FPS video understanding.
>
---
#### [new 076] NDLPNet: A Location-Aware Nighttime Deraining Network and a Real-World Benchmark Dataset
- **分类: cs.CV**

- **简介: 该论文提出NDLPNet，解决夜间降雨图像去雨任务。针对现有方法在低光环境下效果差的问题，设计位置感知模块以捕捉雨痕空间信息，并构建真实夜间降雨数据集NSR。实验表明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.13766v1](http://arxiv.org/pdf/2509.13766v1)**

> **作者:** Huichun Liu; Xiaosong Li; Yang Liu; Xiaoqi Cheng; Haishu Tan
>
> **摘要:** Visual degradation caused by rain streak artifacts in low-light conditions significantly hampers the performance of nighttime surveillance and autonomous navigation. Existing image deraining techniques are primarily designed for daytime conditions and perform poorly under nighttime illumination due to the spatial heterogeneity of rain distribution and the impact of light-dependent stripe visibility. In this paper, we propose a novel Nighttime Deraining Location-enhanced Perceptual Network(NDLPNet) that effectively captures the spatial positional information and density distribution of rain streaks in low-light environments. Specifically, we introduce a Position Perception Module (PPM) to capture and leverage spatial contextual information from input data, enhancing the model's capability to identify and recalibrate the importance of different feature channels. The proposed nighttime deraining network can effectively remove the rain streaks as well as preserve the crucial background information. Furthermore, We construct a night scene rainy (NSR) dataset comprising 900 image pairs, all based on real-world nighttime scenes, providing a new benchmark for nighttime deraining task research. Extensive qualitative and quantitative experimental evaluations on both existing datasets and the NSR dataset consistently demonstrate our method outperform the state-of-the-art (SOTA) methods in nighttime deraining tasks. The source code and dataset is available at https://github.com/Feecuin/NDLPNet.
>
---
#### [new 077] Teacher-Guided Pseudo Supervision and Cross-Modal Alignment for Audio-Visual Video Parsing
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于弱监督音频-视觉视频解析（AVVP）任务，旨在检测可听、可见及视听事件，无需时间标注。提出EMA伪监督框架和类感知跨模态对齐损失，提升段级稳定性和跨模态一致性，实现SOTA性能。**

- **链接: [http://arxiv.org/pdf/2509.14097v1](http://arxiv.org/pdf/2509.14097v1)**

> **作者:** Yaru Chen; Ruohao Guo; Liting Gao; Yang Xiang; Qingyu Luo; Zhenbo Li; Wenwu Wang
>
> **摘要:** Weakly-supervised audio-visual video parsing (AVVP) seeks to detect audible, visible, and audio-visual events without temporal annotations. Previous work has emphasized refining global predictions through contrastive or collaborative learning, but neglected stable segment-level supervision and class-aware cross-modal alignment. To address this, we propose two strategies: (1) an exponential moving average (EMA)-guided pseudo supervision framework that generates reliable segment-level masks via adaptive thresholds or top-k selection, offering stable temporal guidance beyond video-level labels; and (2) a class-aware cross-modal agreement (CMA) loss that aligns audio and visual embeddings at reliable segment-class pairs, ensuring consistency across modalities while preserving temporal structure. Evaluations on LLP and UnAV-100 datasets shows that our method achieves state-of-the-art (SOTA) performance across multiple metrics.
>
---
#### [new 078] Federated Learning for Deforestation Detection: A Distributed Approach with Satellite Imagery
- **分类: cs.CV; cs.DC; 14J60; F.2.2; I.2.7**

- **简介: 该论文提出基于联邦学习的分布式方法，用于卫星图像中的毁林检测。旨在解决数据隐私问题，通过多客户端协作训练模型，使用YOLOS-small和Faster R-CNN等模型实现高效分割任务。**

- **链接: [http://arxiv.org/pdf/2509.13631v1](http://arxiv.org/pdf/2509.13631v1)**

> **作者:** Yuvraj Dutta; Aaditya Sikder; Basabdatta Palit
>
> **备注:** 6 pages, 7 figures, accepted at IEEE INDISCON 2025
>
> **摘要:** Accurate identification of deforestation from satellite images is essential in order to understand the geographical situation of an area. This paper introduces a new distributed approach to identify as well as locate deforestation across different clients using Federated Learning (FL). Federated Learning enables distributed network clients to collaboratively train a model while maintaining data privacy and security of the active users. In our framework, a client corresponds to an edge satellite center responsible for local data processing. Moreover, FL provides an advantage over centralized training method which requires combining data, thereby compromising with data security of the clients. Our framework leverages the FLOWER framework with RAY framework to execute the distributed learning workload. Furthermore, efficient client spawning is ensured by RAY as it can select definite amount of users to create an emulation environment. Our FL framework uses YOLOS-small (a Vision Transformer variant), Faster R-CNN with a ResNet50 backbone, and Faster R-CNN with a MobileNetV3 backbone models trained and tested on publicly available datasets. Our approach provides us a different view for image segmentation-based tasks on satellite imagery.
>
---
#### [new 079] Generative AI for Misalignment-Resistant Virtual Staining to Accelerate Histopathology Workflows
- **分类: cs.CV**

- **简介: 该论文提出一种生成式AI框架，用于解决虚拟染色中因组织错位导致的像素级监督难题。通过级联配准机制提升对齐精度，在多个数据集上显著优于现有方法，加速病理诊断流程。**

- **链接: [http://arxiv.org/pdf/2509.14119v1](http://arxiv.org/pdf/2509.14119v1)**

> **作者:** Jiabo MA; Wenqiang Li; Jinbang Li; Ziyi Liu; Linshan Wu; Fengtao Zhou; Li Liang; Ronald Cheong Kin Chan; Terence T. W. Wong; Hao Chen
>
> **备注:** the arxiv version of the under review journal paper
>
> **摘要:** Accurate histopathological diagnosis often requires multiple differently stained tissue sections, a process that is time-consuming, labor-intensive, and environmentally taxing due to the use of multiple chemical stains. Recently, virtual staining has emerged as a promising alternative that is faster, tissue-conserving, and environmentally friendly. However, existing virtual staining methods face significant challenges in clinical applications, primarily due to their reliance on well-aligned paired data. Obtaining such data is inherently difficult because chemical staining processes can distort tissue structures, and a single tissue section cannot undergo multiple staining procedures without damage or loss of information. As a result, most available virtual staining datasets are either unpaired or roughly paired, making it difficult for existing methods to achieve accurate pixel-level supervision. To address this challenge, we propose a robust virtual staining framework featuring cascaded registration mechanisms to resolve spatial mismatches between generated outputs and their corresponding ground truth. Experimental results demonstrate that our method significantly outperforms state-of-the-art models across five datasets, achieving an average improvement of 3.2% on internal datasets and 10.1% on external datasets. Moreover, in datasets with substantial misalignment, our approach achieves a remarkable 23.8% improvement in peak signal-to-noise ratio compared to baseline models. The exceptional robustness of the proposed method across diverse datasets simplifies the data acquisition process for virtual staining and offers new insights for advancing its development.
>
---
#### [new 080] EDITS: Enhancing Dataset Distillation with Implicit Textual Semantics
- **分类: cs.CV**

- **简介: 该论文属于数据集蒸馏任务，旨在从大规模数据集中合成紧凑数据集。传统方法忽视图像的高层语义信息，本文提出EDITS框架，利用视觉语言模型生成的文本语义增强蒸馏效果，通过融合图文特征生成高质量合成数据集。**

- **链接: [http://arxiv.org/pdf/2509.13858v1](http://arxiv.org/pdf/2509.13858v1)**

> **作者:** Qianxin Xia; Jiawei Du; Guoming Lu; Zhiyong Shu; Jielei Wang
>
> **摘要:** Dataset distillation aims to synthesize a compact dataset from the original large-scale one, enabling highly efficient learning while preserving competitive model performance. However, traditional techniques primarily capture low-level visual features, neglecting the high-level semantic and structural information inherent in images. In this paper, we propose EDITS, a novel framework that exploits the implicit textual semantics within the image data to achieve enhanced distillation. First, external texts generated by a Vision Language Model (VLM) are fused with image features through a Global Semantic Query module, forming the prior clustered buffer. Local Semantic Awareness then selects representative samples from the buffer to construct image and text prototypes, with the latter produced by guiding a Large Language Model (LLM) with meticulously crafted prompt. Ultimately, Dual Prototype Guidance strategy generates the final synthetic dataset through a diffusion model. Extensive experiments confirm the effectiveness of our method.Source code is available in: https://github.com/einsteinxia/EDITS.
>
---
#### [new 081] ColonCrafter: A Depth Estimation Model for Colonoscopy Videos Using Diffusion Priors
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出ColonCrafter，一种基于扩散模型的深度估计方法，用于结肠镜视频。旨在解决现有模型在时间一致性上的不足，通过合成数据学习几何先验，并应用风格迁移技术提升真实视频的性能，实现高质量的3D重建与临床应用。**

- **链接: [http://arxiv.org/pdf/2509.13525v1](http://arxiv.org/pdf/2509.13525v1)**

> **作者:** Romain Hardy; Tyler Berzin; Pranav Rajpurkar
>
> **备注:** 12 pages, 8 figures
>
> **摘要:** Three-dimensional (3D) scene understanding in colonoscopy presents significant challenges that necessitate automated methods for accurate depth estimation. However, existing depth estimation models for endoscopy struggle with temporal consistency across video sequences, limiting their applicability for 3D reconstruction. We present ColonCrafter, a diffusion-based depth estimation model that generates temporally consistent depth maps from monocular colonoscopy videos. Our approach learns robust geometric priors from synthetic colonoscopy sequences to generate temporally consistent depth maps. We also introduce a style transfer technique that preserves geometric structure while adapting real clinical videos to match our synthetic training domain. ColonCrafter achieves state-of-the-art zero-shot performance on the C3VD dataset, outperforming both general-purpose and endoscopy-specific approaches. Although full trajectory 3D reconstruction remains a challenge, we demonstrate clinically relevant applications of ColonCrafter, including 3D point cloud generation and surface coverage assessment.
>
---
#### [new 082] Data-Efficient Spectral Classification of Hyperspectral Data Using MiniROCKET and HDC-MiniROCKET
- **分类: cs.CV**

- **简介: 论文研究利用MiniROCKET和HDC-MiniROCKET进行高光谱数据的谱分类，旨在解决训练数据有限时模型性能下降的问题。提出的方法在少量数据下表现优于现有模型1D-Justo-LiuNet。**

- **链接: [http://arxiv.org/pdf/2509.13809v1](http://arxiv.org/pdf/2509.13809v1)**

> **作者:** Nick Theisen; Kenny Schlegel; Dietrich Paulus; Peer Neubert
>
> **备注:** Accepted for publication at IEEE CASE 2025
>
> **摘要:** The classification of pixel spectra of hyperspectral images, i.e. spectral classification, is used in many fields ranging from agricultural, over medical to remote sensing applications and is currently also expanding to areas such as autonomous driving. Even though for full hyperspectral images the best-performing methods exploit spatial-spectral information, performing classification solely on spectral information has its own advantages, e.g. smaller model size and thus less data required for training. Moreover, spectral information is complementary to spatial information and improvements on either part can be used to improve spatial-spectral approaches in the future. Recently, 1D-Justo-LiuNet was proposed as a particularly efficient model with very few parameters, which currently defines the state of the art in spectral classification. However, we show that with limited training data the model performance deteriorates. Therefore, we investigate MiniROCKET and HDC-MiniROCKET for spectral classification to mitigate that problem. The model extracts well-engineered features without trainable parameters in the feature extraction part and is therefore less vulnerable to limited training data. We show that even though MiniROCKET has more parameters it outperforms 1D-Justo-LiuNet in limited data scenarios and is mostly on par with it in the general case
>
---
#### [new 083] MOCHA: Multi-modal Objects-aware Cross-arcHitecture Alignment
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出MOCHA方法，通过知识蒸馏将多模态语义从大模型（如LLaVa）转移到轻量检测模型（如YOLO）。该方法在对象级别对齐，无需文本输入即可提升检测性能，适用于实际部署。属于视觉-语言对齐任务，解决轻量模型语义不足问题。**

- **链接: [http://arxiv.org/pdf/2509.14001v1](http://arxiv.org/pdf/2509.14001v1)**

> **作者:** Elena Camuffo; Francesco Barbato; Mete Ozay; Simone Milani; Umberto Michieli
>
> **摘要:** We introduce MOCHA (Multi-modal Objects-aware Cross-arcHitecture Alignment), a knowledge distillation approach that transfers region-level multimodal semantics from a large vision-language teacher (e.g., LLaVa) into a lightweight vision-only object detector student (e.g., YOLO). A translation module maps student features into a joint space, where the training of the student and translator is guided by a dual-objective loss that enforces both local alignment and global relational consistency. Unlike prior approaches focused on dense or global alignment, MOCHA operates at the object level, enabling efficient transfer of semantics without modifying the teacher or requiring textual input at inference. We validate our method across four personalized detection benchmarks under few-shot regimes. Results show consistent gains over baselines, with a +10.1 average score improvement. Despite its compact architecture, MOCHA reaches performance on par with larger multimodal models, proving its suitability for real-world deployment.
>
---
#### [new 084] Semantic 3D Reconstructions with SLAM for Central Airway Obstruction
- **分类: cs.RO; cs.CV**

- **简介: 论文提出一种结合SLAM与语义分割的实时3D重建方法，用于中央气道阻塞的手术导航。任务是实现高精度、带语义标注的气道重建，解决传统手术风险高的问题。方法融合DROID-SLAM与分割模型，提升手术自动化水平。**

- **链接: [http://arxiv.org/pdf/2509.13541v1](http://arxiv.org/pdf/2509.13541v1)**

> **作者:** Ayberk Acar; Fangjie Li; Hao Li; Lidia Al-Zogbi; Kanyifeechukwu Jane Oguine; Susheela Sharma Stern; Jesse F. d'Almeida; Robert J. Webster III; Ipek Oguz; Jie Ying Wu
>
> **备注:** 5 pages, 2 figures, 1 table
>
> **摘要:** Central airway obstruction (CAO) is a life-threatening condition with increasing incidence, caused by tumors in and outside of the airway. Traditional treatment methods such as bronchoscopy and electrocautery can be used to remove the tumor completely; however, these methods carry a high risk of complications. Recent advances allow robotic interventions with lesser risk. The combination of robot interventions with scene understanding and mapping also opens up the possibilities for automation. We present a novel pipeline that enables real-time, semantically informed 3D reconstructions of the central airway using monocular endoscopic video. Our approach combines DROID-SLAM with a segmentation model trained to identify obstructive tissues. The SLAM module reconstructs the 3D geometry of the airway in real time, while the segmentation masks guide the annotation of obstruction regions within the reconstructed point cloud. To validate our pipeline, we evaluate the reconstruction quality using ex vivo models. Qualitative and quantitative results show high similarity between ground truth CT scans and the 3D reconstructions (0.62 mm Chamfer distance). By integrating segmentation directly into the SLAM workflow, our system produces annotated 3D maps that highlight clinically relevant regions in real time. High-speed capabilities of the pipeline allows quicker reconstructions compared to previous work, reflecting the surgical scene more accurately. To the best of our knowledge, this is the first work to integrate semantic segmentation with real-time monocular SLAM for endoscopic CAO scenarios. Our framework is modular and can generalize to other anatomies or procedures with minimal changes, offering a promising step toward autonomous robotic interventions.
>
---
#### [new 085] A Domain Knowledge Informed Approach for Anomaly Detection of Electric Vehicle Interior Sounds
- **分类: cs.SD; cs.AI; cs.CV; cs.LG; eess.AS; I.2.1; I.2.6; I.2.10; I.5.1; I.5.2; J.2; J.7**

- **简介: 论文提出一种基于领域知识的模型选择方法，用于检测电动汽车舱内异常声音。任务为无监督异常检测，解决缺乏故障标签导致的模型选择难题。通过构造代理异常数据验证模型，显著优于传统策略。**

- **链接: [http://arxiv.org/pdf/2509.13390v1](http://arxiv.org/pdf/2509.13390v1)**

> **作者:** Deepti Kunte; Bram Cornelis; Claudio Colangeli; Karl Janssens; Brecht Van Baelen; Konstantinos Gryllias
>
> **备注:** Submitted to: Mechanical Systems and Signal Processing
>
> **摘要:** The detection of anomalies in automotive cabin sounds is critical for ensuring vehicle quality and maintaining passenger comfort. In many real-world settings, this task is more appropriately framed as an unsupervised learning problem rather than the supervised case due to the scarcity or complete absence of labeled faulty data. In such an unsupervised setting, the model is trained exclusively on healthy samples and detects anomalies as deviations from normal behavior. However, in the absence of labeled faulty samples for validation and the limited reliability of commonly used metrics, such as validation reconstruction error, effective model selection remains a significant challenge. To overcome these limitations, a domain-knowledge-informed approach for model selection is proposed, in which proxy-anomalies engineered through structured perturbations of healthy spectrograms are used in the validation set to support model selection. The proposed methodology is evaluated on a high-fidelity electric vehicle dataset comprising healthy and faulty cabin sounds across five representative fault types viz., Imbalance, Modulation, Whine, Wind, and Pulse Width Modulation. This dataset, generated using advanced sound synthesis techniques, and validated via expert jury assessments, has been made publicly available to facilitate further research. Experimental evaluations on the five fault cases demonstrate the selection of optimal models using proxy-anomalies, significantly outperform conventional model selection strategies.
>
---
#### [new 086] 3D Reconstruction of Coronary Vessel Trees from Biplanar X-Ray Images Using a Geometric Approach
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出一种基于几何方法的三维冠状血管重建框架，解决从双平面X光图像中重建血管树的问题。通过图像分割、运动相位匹配和几何重建，提高三维重建精度与效率。**

- **链接: [http://arxiv.org/pdf/2509.13358v1](http://arxiv.org/pdf/2509.13358v1)**

> **作者:** Ethan Koland; Lin Xi; Nadeev Wijesuriya; YingLiang Ma
>
> **摘要:** X-ray angiography is widely used in cardiac interventions to visualize coronary vessels, assess integrity, detect stenoses and guide treatment. We propose a framework for reconstructing 3D vessel trees from biplanar X-ray images which are extracted from two X-ray videos captured at different C-arm angles. The proposed framework consists of three main components: image segmentation, motion phase matching, and 3D reconstruction. An automatic video segmentation method for X-ray angiography to enable semantic segmentation for image segmentation and motion phase matching. The goal of the motion phase matching is to identify a pair of X-ray images that correspond to a similar respiratory and cardiac motion phase to reduce errors in 3D reconstruction. This is achieved by tracking a stationary object such as a catheter or lead within the X-ray video. The semantic segmentation approach assigns different labels to different object classes enabling accurate differentiation between blood vessels, balloons, and catheters. Once a suitable image pair is selected, key anatomical landmarks (vessel branching points and endpoints) are matched between the two views using a heuristic method that minimizes reconstruction errors. This is followed by a novel geometric reconstruction algorithm to generate the 3D vessel tree. The algorithm computes the 3D vessel centrelines by determining the intersection of two 3D surfaces. Compared to traditional methods based on epipolar constraints, the proposed approach simplifies there construction workflow and improves overall accuracy. We trained and validated our segmentation method on 62 X-ray angiography video sequences. On the test set, our method achieved a segmentation accuracy of 0.703. The 3D reconstruction framework was validated by measuring the reconstruction error of key anatomical landmarks, achieving a reprojection errors of 0.62mm +/- 0.38mm.
>
---
#### [new 087] MetricNet: Recovering Metric Scale in Generative Navigation Policies
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人导航任务，解决生成式导航策略中轨迹无尺度和短视问题。提出MetricNet预测航路点间距离，实现真实坐标定位，并集成至MetricNav中提升避障与导航性能。**

- **链接: [http://arxiv.org/pdf/2509.13965v1](http://arxiv.org/pdf/2509.13965v1)**

> **作者:** Abhijeet Nayak; Débora N. P. Oliveira; Samiran Gode; Cordelia Schmid; Wolfram Burgard
>
> **摘要:** Generative navigation policies have made rapid progress in improving end-to-end learned navigation. Despite their promising results, this paradigm has two structural problems. First, the sampled trajectories exist in an abstract, unscaled space without metric grounding. Second, the control strategy discards the full path, instead moving directly towards a single waypoint. This leads to short-sighted and unsafe actions, moving the robot towards obstacles that a complete and correctly scaled path would circumvent. To address these issues, we propose MetricNet, an effective add-on for generative navigation that predicts the metric distance between waypoints, grounding policy outputs in real-world coordinates. We evaluate our method in simulation with a new benchmarking framework and show that executing MetricNet-scaled waypoints significantly improves both navigation and exploration performance. Beyond simulation, we further validate our approach in real-world experiments. Finally, we propose MetricNav, which integrates MetricNet into a navigation policy to guide the robot away from obstacles while still moving towards the goal.
>
---
#### [new 088] Rest2Visual: Predicting Visually Evoked fMRI from Resting-State Scans
- **分类: q-bio.NC; cs.CV**

- **简介: 该论文提出Rest2Visual模型，从静息态fMRI预测视觉刺激引发的fMRI激活图。属于脑成像建模任务，解决任务态fMRI获取成本高的问题，构建了大规模数据集并验证模型在个体化功能表征生成上的有效性。**

- **链接: [http://arxiv.org/pdf/2509.13612v1](http://arxiv.org/pdf/2509.13612v1)**

> **作者:** Chuyang Zhou; Ziao Ji; Daochang Liu; Dongang Wang; Chenyu Wang; Chang Xu
>
> **摘要:** Understanding how spontaneous brain activity relates to stimulus-driven neural responses is a fundamental challenge in cognitive neuroscience. While task-based functional magnetic resonance imaging (fMRI) captures localized stimulus-evoked brain activation, its acquisition is costly, time-consuming, and difficult to scale across populations. In contrast, resting-state fMRI (rs-fMRI) is task-free and abundant, but lacks direct interpretability. We introduce Rest2Visual, a conditional generative model that predicts visually evoked fMRI (ve-fMRI) from resting-state input and 2D visual stimuli. It follows a volumetric encoder--decoder design, where multiscale 3D features from rs-fMRI are modulated by image embeddings via adaptive normalization, enabling spatially accurate, stimulus-specific activation synthesis. To enable model training, we construct a large-scale triplet dataset from the Natural Scenes Dataset (NSD), aligning each rs-fMRI volume with stimulus images and their corresponding ve-fMRI activation maps. Quantitative evaluation shows that the predicted activations closely match ground truth across standard similarity and representational metrics, and support successful image reconstruction in downstream decoding. Notably, the predicted maps preserve subject-specific structure, demonstrating the model's capacity to generate individualized functional surrogates. Our results provide compelling evidence that individualized spontaneous neural activity can be transformed into stimulus-aligned representations, opening new avenues for scalable, task-free functional brain modeling.
>
---
#### [new 089] Generative AI Pipeline for Interactive Prompt-driven 2D-to-3D Vascular Reconstruction for Fontan Geometries from Contrast-Enhanced X-Ray Fluoroscopy Imaging
- **分类: eess.IV; cs.AI; cs.CV; cs.ET; q-bio.QM; 92C50, 68T07, 76D05, 65D18, 92C55; I.4.6; I.4.8; J.3; I.2.10; I.4.9**

- **简介: 该论文提出一种基于生成式AI的2D到3D血管重建方法，用于Fontan手术的几何建模。通过多步骤AI处理 fluoroscopic影像，生成适合CFD分析的3D模型，解决传统2D成像信息不足的问题，实现快速虚拟血流可视化与手术规划。**

- **链接: [http://arxiv.org/pdf/2509.13372v1](http://arxiv.org/pdf/2509.13372v1)**

> **作者:** Prahlad G Menon
>
> **摘要:** Fontan palliation for univentricular congenital heart disease progresses to hemodynamic failure with complex flow patterns poorly characterized by conventional 2D imaging. Current assessment relies on fluoroscopic angiography, providing limited 3D geometric information essential for computational fluid dynamics (CFD) analysis and surgical planning. A multi-step AI pipeline was developed utilizing Google's Gemini 2.5 Flash (2.5B parameters) for systematic, iterative processing of fluoroscopic angiograms through transformer-based neural architecture. The pipeline encompasses medical image preprocessing, vascular segmentation, contrast enhancement, artifact removal, and virtual hemodynamic flow visualization within 2D projections. Final views were processed through Tencent's Hunyuan3D-2mini (384M parameters) for stereolithography file generation. The pipeline successfully generated geometrically optimized 2D projections from single-view angiograms after 16 processing steps using a custom web interface. Initial iterations contained hallucinated vascular features requiring iterative refinement to achieve anatomically faithful representations. Final projections demonstrated accurate preservation of complex Fontan geometry with enhanced contrast suitable for 3D conversion. AI-generated virtual flow visualization identified stagnation zones in central connections and flow patterns in branch arteries. Complete processing required under 15 minutes with second-level API response times. This approach demonstrates clinical feasibility of generating CFD-suitable geometries from routine angiographic data, enabling 3D generation and rapid virtual flow visualization for cursory insights prior to full CFD simulation. While requiring refinement cycles for accuracy, this establishes foundation for democratizing advanced geometric and hemodynamic analysis using readily available imaging data.
>
---
#### [new 090] PREDICT-GBM: Platform for Robust Evaluation and Development of Individualized Computational Tumor Models in Glioblastoma
- **分类: eess.IV; cs.CV; cs.LG; q-bio.QM**

- **简介: 该论文提出PREDICT-GBM平台，用于评估和开发个性化胶质母细胞瘤计算模型，解决传统放疗忽略个体差异的问题。通过集成数据集和系统评估，推动模型临床转化，提升治疗效果。**

- **链接: [http://arxiv.org/pdf/2509.13360v1](http://arxiv.org/pdf/2509.13360v1)**

> **作者:** L. Zimmer; J. Weidner; M. Balcerak; F. Kofler; I. Ezhov; B. Menze; B. Wiestler
>
> **摘要:** Glioblastoma is the most prevalent primary brain malignancy, distinguished by its highly invasive behavior and exceptionally high rates of recurrence. Conventional radiation therapy, which employs uniform treatment margins, fails to account for patient-specific anatomical and biological factors that critically influence tumor cell migration. To address this limitation, numerous computational models of glioblastoma growth have been developed, enabling generation of tumor cell distribution maps extending beyond radiographically visible regions and thus informing more precise treatment strategies. However, despite encouraging preliminary findings, the clinical adoption of these growth models remains limited. To bridge this translational gap and accelerate both model development and clinical validation, we introduce PREDICT-GBM, a comprehensive integrated pipeline and dataset for modeling and evaluation. This platform enables systematic benchmarking of state-of-the-art tumor growth models using an expert-curated clinical dataset comprising 255 subjects with complete tumor segmentations and tissue characterization maps. Our analysis demonstrates that personalized radiation treatment plans derived from tumor growth predictions achieved superior recurrence coverage compared to conventional uniform margin approaches for two of the evaluated models. This work establishes a robust platform for advancing and systematically evaluating cutting-edge tumor growth modeling approaches, with the ultimate goal of facilitating clinical translation and improving patient outcomes.
>
---
#### [new 091] LLM-I: LLMs are Naturally Interleaved Multimodal Creators
- **分类: cs.LG; cs.CV**

- **简介: 论文提出LLM-I框架，解决统一模型在多模态生成中的工具单一问题。通过强化学习训练代理智能调用多种视觉工具，实现更优的图像-文本生成效果，提升多项基准测试表现。属于多模态生成任务。**

- **链接: [http://arxiv.org/pdf/2509.13642v1](http://arxiv.org/pdf/2509.13642v1)**

> **作者:** Zirun Guo; Feng Zhang; Kai Jia; Tao Jin
>
> **摘要:** We propose LLM-Interleaved (LLM-I), a flexible and dynamic framework that reframes interleaved image-text generation as a tool-use problem. LLM-I is designed to overcome the "one-tool" bottleneck of current unified models, which are limited to synthetic imagery and struggle with tasks requiring factual grounding or programmatic precision. Our framework empowers a central LLM or MLLM agent to intelligently orchestrate a diverse toolkit of specialized visual tools, including online image search, diffusion-based generation, code execution, and image editing. The agent is trained to select and apply these tools proficiently via a Reinforcement Learning (RL) framework that features a hybrid reward system combining rule-based logic with judgments from LLM and MLLM evaluators. Trained on a diverse new dataset using four different model backbones, LLM-I demonstrates state-of-the-art performance, outperforming existing methods by a large margin across four benchmarks. We also introduce a novel test-time scaling strategy that provides further performance gains. Project Page: https://github.com/ByteDance-BandAI/LLM-I.
>
---
#### [new 092] InterKey: Cross-modal Intersection Keypoints for Global Localization on OpenStreetMap
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出InterKey框架，用于解决自动驾驶车辆在GNSS不可用环境下的全局定位问题。通过融合点云与OpenStreetMap数据，利用道路交叉口作为特征点，实现跨模态匹配，提升定位精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.13857v1](http://arxiv.org/pdf/2509.13857v1)**

> **作者:** Nguyen Hoang Khoi Tran; Julie Stephany Berrio; Mao Shan; Stewart Worrall
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Reliable global localization is critical for autonomous vehicles, especially in environments where GNSS is degraded or unavailable, such as urban canyons and tunnels. Although high-definition (HD) maps provide accurate priors, the cost of data collection, map construction, and maintenance limits scalability. OpenStreetMap (OSM) offers a free and globally available alternative, but its coarse abstraction poses challenges for matching with sensor data. We propose InterKey, a cross-modal framework that leverages road intersections as distinctive landmarks for global localization. Our method constructs compact binary descriptors by jointly encoding road and building imprints from point clouds and OSM. To bridge modality gaps, we introduce discrepancy mitigation, orientation determination, and area-equalized sampling strategies, enabling robust cross-modal matching. Experiments on the KITTI dataset demonstrate that InterKey achieves state-of-the-art accuracy, outperforming recent baselines by a large margin. The framework generalizes to sensors that can produce dense structural point clouds, offering a scalable and cost-effective solution for robust vehicle localization.
>
---
#### [new 093] MAP: End-to-End Autonomous Driving with Map-Assisted Planning
- **分类: cs.RO; cs.AI; cs.CV; I.2.9; I.2.10**

- **简介: 该论文提出MAP框架，用于端到端自动驾驶轨迹规划。通过融合地图信息与车辆状态，提升规划性能。实验表明其在多个指标上优于基线模型，效果显著。**

- **链接: [http://arxiv.org/pdf/2509.13926v1](http://arxiv.org/pdf/2509.13926v1)**

> **作者:** Huilin Yin; Yiming Kan; Daniel Watzenig
>
> **备注:** 8 pages, 2 figures, accepted by ICCVW Author list updated to match the camera-ready version, in compliance with conference policy
>
> **摘要:** In recent years, end-to-end autonomous driving has attracted increasing attention for its ability to jointly model perception, prediction, and planning within a unified framework. However, most existing approaches underutilize the online mapping module, leaving its potential to enhance trajectory planning largely untapped. This paper proposes MAP (Map-Assisted Planning), a novel map-assisted end-to-end trajectory planning framework. MAP explicitly integrates segmentation-based map features and the current ego status through a Plan-enhancing Online Mapping module, an Ego-status-guided Planning module, and a Weight Adapter based on current ego status. Experiments conducted on the DAIR-V2X-seq-SPD dataset demonstrate that the proposed method achieves a 16.6% reduction in L2 displacement error, a 56.2% reduction in off-road rate, and a 44.5% improvement in overall score compared to the UniV2X baseline, even without post-processing. Furthermore, it achieves top ranking in Track 2 of the End-to-End Autonomous Driving through V2X Cooperation Challenge of MEIS Workshop @CVPR2025, outperforming the second-best model by 39.5% in terms of overall score. These results highlight the effectiveness of explicitly leveraging semantic map features in planning and suggest new directions for improving structure design in end-to-end autonomous driving systems. Our code is available at https://gitee.com/kymkym/map.git
>
---
#### [new 094] Cross-Distribution Diffusion Priors-Driven Iterative Reconstruction for Sparse-View CT
- **分类: eess.IV; cs.CV; 65R32**

- **简介: 该论文提出CDPIR框架，解决稀疏视角CT重建中的域外问题。通过结合跨分布扩散先验与迭代重建方法，提升模型泛化能力，在多数据集下实现更稳定、高质量的图像重建。**

- **链接: [http://arxiv.org/pdf/2509.13576v1](http://arxiv.org/pdf/2509.13576v1)**

> **作者:** Haodong Li; Shuo Han; Haiyang Mao; Yu Shi; Changsheng Fang; Jianjia Zhang; Weiwen Wu; Hengyong Yu
>
> **备注:** 11 pages, 8 figures, under reviewing of IEEE TMI
>
> **摘要:** Sparse-View CT (SVCT) reconstruction enhances temporal resolution and reduces radiation dose, yet its clinical use is hindered by artifacts due to view reduction and domain shifts from scanner, protocol, or anatomical variations, leading to performance degradation in out-of-distribution (OOD) scenarios. In this work, we propose a Cross-Distribution Diffusion Priors-Driven Iterative Reconstruction (CDPIR) framework to tackle the OOD problem in SVCT. CDPIR integrates cross-distribution diffusion priors, derived from a Scalable Interpolant Transformer (SiT), with model-based iterative reconstruction methods. Specifically, we train a SiT backbone, an extension of the Diffusion Transformer (DiT) architecture, to establish a unified stochastic interpolant framework, leveraging Classifier-Free Guidance (CFG) across multiple datasets. By randomly dropping the conditioning with a null embedding during training, the model learns both domain-specific and domain-invariant priors, enhancing generalizability. During sampling, the globally sensitive transformer-based diffusion model exploits the cross-distribution prior within the unified stochastic interpolant framework, enabling flexible and stable control over multi-distribution-to-noise interpolation paths and decoupled sampling strategies, thereby improving adaptation to OOD reconstruction. By alternating between data fidelity and sampling updates, our model achieves state-of-the-art performance with superior detail preservation in SVCT reconstructions. Extensive experiments demonstrate that CDPIR significantly outperforms existing approaches, particularly under OOD conditions, highlighting its robustness and potential clinical value in challenging imaging scenarios.
>
---
#### [new 095] MCGS-SLAM: A Multi-Camera SLAM Framework Using Gaussian Splatting for High-Fidelity Mapping
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MCGS-SLAM，一种基于多摄像头和高斯点云的SLAM框架，用于提升机器人和自动驾驶的高保真地图构建。解决单目SLAM在鲁棒性和几何覆盖上的不足，通过多视角RGB融合与优化实现更准确的轨迹和重建。**

- **链接: [http://arxiv.org/pdf/2509.14191v1](http://arxiv.org/pdf/2509.14191v1)**

> **作者:** Zhihao Cao; Hanyu Wu; Li Wa Tang; Zizhou Luo; Zihan Zhu; Wei Zhang; Marc Pollefeys; Martin R. Oswald
>
> **摘要:** Recent progress in dense SLAM has primarily targeted monocular setups, often at the expense of robustness and geometric coverage. We present MCGS-SLAM, the first purely RGB-based multi-camera SLAM system built on 3D Gaussian Splatting (3DGS). Unlike prior methods relying on sparse maps or inertial data, MCGS-SLAM fuses dense RGB inputs from multiple viewpoints into a unified, continuously optimized Gaussian map. A multi-camera bundle adjustment (MCBA) jointly refines poses and depths via dense photometric and geometric residuals, while a scale consistency module enforces metric alignment across views using low-rank priors. The system supports RGB input and maintains real-time performance at large scale. Experiments on synthetic and real-world datasets show that MCGS-SLAM consistently yields accurate trajectories and photorealistic reconstructions, usually outperforming monocular baselines. Notably, the wide field of view from multi-camera input enables reconstruction of side-view regions that monocular setups miss, critical for safe autonomous operation. These results highlight the promise of multi-camera Gaussian Splatting SLAM for high-fidelity mapping in robotics and autonomous driving.
>
---
#### [new 096] The Art of Saying "Maybe": A Conformal Lens for Uncertainty Benchmarking in VLMs
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于视觉-语言模型（VLMs）的不确定性评估任务，旨在解决其不确定性量化不足的问题。研究对16个VLM在6个多模态数据集上进行全面评估，发现大模型不确定性量化更优，数学与推理任务表现较差，为可靠评估奠定基础。**

- **链接: [http://arxiv.org/pdf/2509.13379v1](http://arxiv.org/pdf/2509.13379v1)**

> **作者:** Asif Azad; Mohammad Sadat Hossain; MD Sadik Hossain Shanto; M Saifur Rahman; Md Rizwan Pervez
>
> **摘要:** Vision-Language Models (VLMs) have achieved remarkable progress in complex visual understanding across scientific and reasoning tasks. While performance benchmarking has advanced our understanding of these capabilities, the critical dimension of uncertainty quantification has received insufficient attention. Therefore, unlike prior conformal prediction studies that focused on limited settings, we conduct a comprehensive uncertainty benchmarking study, evaluating 16 state-of-the-art VLMs (open and closed-source) across 6 multimodal datasets with 3 distinct scoring functions. Our findings demonstrate that larger models consistently exhibit better uncertainty quantification; models that know more also know better what they don't know. More certain models achieve higher accuracy, while mathematical and reasoning tasks elicit poorer uncertainty performance across all models compared to other domains. This work establishes a foundation for reliable uncertainty evaluation in multimodal systems.
>
---
#### [new 097] Autonomous Reporting of Normal Chest X-rays by Artificial Intelligence in the United Kingdom. Can We Take the Human Out of the Loop?
- **分类: q-bio.PE; cs.CV**

- **简介: 论文探讨AI自主报告正常胸片的可行性，旨在解决英国放射科医生短缺导致的报告延迟问题。研究分析了技术挑战、法律合规及对医疗实践的影响，强调需谨慎推进AI在该领域的应用。**

- **链接: [http://arxiv.org/pdf/2509.13428v1](http://arxiv.org/pdf/2509.13428v1)**

> **作者:** Katrina Nash; James Vaz; Ahmed Maiter; Christopher Johns; Nicholas Woznitza; Aditya Kale; Abdala Espinosa Morgado; Rhidian Bramley; Mark Hall; David Lowe; Alex Novak; Sarim Ather
>
> **摘要:** Chest X-rays (CXRs) are the most commonly performed imaging investigation. In the UK, many centres experience reporting delays due to radiologist workforce shortages. Artificial intelligence (AI) tools capable of distinguishing normal from abnormal CXRs have emerged as a potential solution. If normal CXRs could be safely identified and reported without human input, a substantial portion of radiology workload could be reduced. This article examines the feasibility and implications of autonomous AI reporting of normal CXRs. Key issues include defining normal, ensuring generalisability across populations, and managing the sensitivity-specificity trade-off. It also addresses legal and regulatory challenges, such as compliance with IR(ME)R and GDPR, and the lack accountability frameworks for errors. Further considerations include the impact on radiologists practice, the need for robust post-market surveillance, and incorporation of patient perspectives. While the benefits are clear, adoption must be cautious.
>
---
#### [new 098] Object Pose Estimation through Dexterous Touch
- **分类: cs.RO; cs.CV**

- **简介: 论文提出一种基于触觉的物体位姿估计方法，通过双机械手协同操作与强化学习探索物体表面，利用触觉数据迭代优化物体形状与位姿。属于机器人触觉感知任务，解决视觉受限场景下物体位姿估计问题。**

- **链接: [http://arxiv.org/pdf/2509.13591v1](http://arxiv.org/pdf/2509.13591v1)**

> **作者:** Amir-Hossein Shahidzadeh; Jiyue Zhu; Kezhou Chen; Sha Yi; Cornelia Fermüller; Yiannis Aloimonos; Xiaolong Wang
>
> **摘要:** Robust object pose estimation is essential for manipulation and interaction tasks in robotics, particularly in scenarios where visual data is limited or sensitive to lighting, occlusions, and appearances. Tactile sensors often offer limited and local contact information, making it challenging to reconstruct the pose from partial data. Our approach uses sensorimotor exploration to actively control a robot hand to interact with the object. We train with Reinforcement Learning (RL) to explore and collect tactile data. The collected 3D point clouds are used to iteratively refine the object's shape and pose. In our setup, one hand holds the object steady while the other performs active exploration. We show that our method can actively explore an object's surface to identify critical pose features without prior knowledge of the object's geometry. Supplementary material and more demonstrations will be provided at https://amirshahid.github.io/BimanualTactilePose .
>
---
## 更新

#### [replaced 001] Singular Value Few-shot Adaptation of Vision-Language Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.03740v2](http://arxiv.org/pdf/2509.03740v2)**

> **作者:** Taha Koleilat; Hassan Rivaz; Yiming Xiao
>
> **备注:** 10 pages, 2 figures, 8 tables
>
> **摘要:** Vision-language models (VLMs) like CLIP have shown impressive zero-shot and few-shot learning capabilities across diverse applications. However, adapting these models to new fine-grained domains remains difficult due to reliance on prompt engineering and the high cost of full model fine-tuning. Existing adaptation approaches rely on augmented components, such as prompt tokens and adapter modules, which could limit adaptation quality, destabilize the model, and compromise the rich knowledge learned during pretraining. In this work, we present CLIP-SVD, a novel multi-modal and parameter-efficient adaptation technique that leverages Singular Value Decomposition (SVD) to modify the internal parameter space of CLIP without injecting additional modules. Specifically, we fine-tune only the singular values of the CLIP parameter matrices to rescale the basis vectors for domain adaptation while retaining the pretrained model. This design enables enhanced adaptation performance using only 0.04% of the model's total parameters and better preservation of its generalization ability. CLIP-SVD achieves state-of-the-art classification results on 11 natural and 10 biomedical datasets, outperforming previous methods in both accuracy and generalization under few-shot settings. Additionally, we leverage a natural language-based approach to analyze the effectiveness and dynamics of the CLIP adaptation to allow interpretability of CLIP-SVD. The code is publicly available at https://github.com/HealthX-Lab/CLIP-SVD.
>
---
#### [replaced 002] SwiftVideo: A Unified Framework for Few-Step Video Generation through Trajectory-Distribution Alignment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.06082v2](http://arxiv.org/pdf/2508.06082v2)**

> **作者:** Yanxiao Sun; Jiafu Wu; Yun Cao; Chengming Xu; Yabiao Wang; Weijian Cao; Donghao Luo; Chengjie Wang; Yanwei Fu
>
> **摘要:** Diffusion-based or flow-based models have achieved significant progress in video synthesis but require multiple iterative sampling steps, which incurs substantial computational overhead. While many distillation methods that are solely based on trajectory-preserving or distribution-matching have been developed to accelerate video generation models, these approaches often suffer from performance breakdown or increased artifacts under few-step settings. To address these limitations, we propose \textbf{\emph{SwiftVideo}}, a unified and stable distillation framework that combines the advantages of trajectory-preserving and distribution-matching strategies. Our approach introduces continuous-time consistency distillation to ensure precise preservation of ODE trajectories. Subsequently, we propose a dual-perspective alignment that includes distribution alignment between synthetic and real data along with trajectory alignment across different inference steps. Our method maintains high-quality video generation while substantially reducing the number of inference steps. Quantitative evaluations on the OpenVid-1M benchmark demonstrate that our method significantly outperforms existing approaches in few-step video generation.
>
---
#### [replaced 003] Improvement of Human-Object Interaction Action Recognition Using Scene Information and Multi-Task Learning Approach
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.09067v3](http://arxiv.org/pdf/2509.09067v3)**

> **作者:** Hesham M. Shehata; Mohammad Abdolrahmani
>
> **摘要:** Recent graph convolutional neural networks (GCNs) have shown high performance in the field of human action recognition by using human skeleton poses. However, it fails to detect human-object interaction cases successfully due to the lack of effective representation of the scene information and appropriate learning architectures. In this context, we propose a methodology to utilize human action recognition performance by considering fixed object information in the environment and following a multi-task learning approach. In order to evaluate the proposed method, we collected real data from public environments and prepared our data set, which includes interaction classes of hands-on fixed objects (e.g., ATM ticketing machines, check-in/out machines, etc.) and non-interaction classes of walking and standing. The multi-task learning approach, along with interaction area information, succeeds in recognizing the studied interaction and non-interaction actions with an accuracy of 99.25%, outperforming the accuracy of the base model using only human skeleton poses by 2.75%.
>
---
#### [replaced 004] Stereo Anything: Unifying Zero-shot Stereo Matching with Large-Scale Mixed Data
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.14053v3](http://arxiv.org/pdf/2411.14053v3)**

> **作者:** Xianda Guo; Chenming Zhang; Youmin Zhang; Ruilin Wang; Dujun Nie; Wenzhao Zheng; Matteo Poggi; Hao Zhao; Mang Ye; Qin Zou; Long Chen
>
> **备注:** Code will be available at \url{https://github.com/XiandaGuo/OpenStereo}
>
> **摘要:** Stereo matching serves as a cornerstone in 3D vision, aiming to establish pixel-wise correspondences between stereo image pairs for depth recovery. Despite remarkable progress driven by deep neural architectures, current models often exhibit severe performance degradation when deployed in unseen domains, primarily due to the limited diversity of training data. In this work, we introduce StereoAnything, a data-centric framework that substantially enhances the zero-shot generalization capability of existing stereo models. Rather than devising yet another specialized architecture, we scale stereo training to an unprecedented level by systematically unifying heterogeneous stereo sources: (1) curated labeled datasets covering diverse environments, and (2) large-scale synthetic stereo pairs generated from unlabeled monocular images. Our mixed-data strategy delivers consistent and robust learning signals across domains, effectively mitigating dataset bias. Extensive zero-shot evaluations on four public benchmarks demonstrate that Stereo Anything achieves state-of-the-art generalization. This work paves the way towards truly universal stereo matching, offering a scalable data paradigm applicable to any stereo image pair. We extensively evaluate the zero-shot capabilities of our model on four public datasets, showcasing its impressive ability to generalize to any stereo image pair. Code is available at https://github.com/XiandaGuo/OpenStereo.
>
---
#### [replaced 005] DBLP: Noise Bridge Consistency Distillation For Efficient And Reliable Adversarial Purification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.00552v2](http://arxiv.org/pdf/2508.00552v2)**

> **作者:** Chihan Huang; Belal Alsinglawi; Islam Al-qudah
>
> **摘要:** Recent advances in deep neural networks (DNNs) have led to remarkable success across a wide range of tasks. However, their susceptibility to adversarial perturbations remains a critical vulnerability. Existing diffusion-based adversarial purification methods often require intensive iterative denoising, severely limiting their practical deployment. In this paper, we propose Diffusion Bridge Distillation for Purification (DBLP), a novel and efficient diffusion-based framework for adversarial purification. Central to our approach is a new objective, noise bridge distillation, which constructs a principled alignment between the adversarial noise distribution and the clean data distribution within a latent consistency model (LCM). To further enhance semantic fidelity, we introduce adaptive semantic enhancement, which fuses multi-scale pyramid edge maps as conditioning input to guide the purification process. Extensive experiments across multiple datasets demonstrate that DBLP achieves state-of-the-art (SOTA) robust accuracy, superior image quality, and around 0.2s inference time, marking a significant step toward real-time adversarial purification.
>
---
#### [replaced 006] Humor in Pixels: Benchmarking Large Multimodal Models Understanding of Online Comics
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.12248v2](http://arxiv.org/pdf/2509.12248v2)**

> **作者:** Yuriel Ryan; Rui Yang Tan; Kenny Tsu Wei Choo; Roy Ka-Wei Lee
>
> **备注:** 27 pages, 8 figures, EMNLP 2025 Findings
>
> **摘要:** Understanding humor is a core aspect of social intelligence, yet it remains a significant challenge for Large Multimodal Models (LMMs). We introduce PixelHumor, a benchmark dataset of 2,800 annotated multi-panel comics designed to evaluate LMMs' ability to interpret multimodal humor and recognize narrative sequences. Experiments with state-of-the-art LMMs reveal substantial gaps: for instance, top models achieve only 61% accuracy in panel sequencing, far below human performance. This underscores critical limitations in current models' integration of visual and textual cues for coherent narrative and humor understanding. By providing a rigorous framework for evaluating multimodal contextual and narrative reasoning, PixelHumor aims to drive the development of LMMs that better engage in natural, socially aware interactions.
>
---
#### [replaced 007] DiffGAN: A Test Generation Approach for Differential Testing of Deep Neural Networks for Image Analysis
- **分类: cs.CV; cs.LG; cs.SE**

- **链接: [http://arxiv.org/pdf/2410.19794v4](http://arxiv.org/pdf/2410.19794v4)**

> **作者:** Zohreh Aghababaeyan; Manel Abdellatif; Lionel Briand; Ramesh S
>
> **备注:** Accepted into IEEE Transactions on Software Engineering
>
> **摘要:** Deep Neural Networks (DNNs) are increasingly deployed across applications. However, ensuring their reliability remains a challenge, and in many situations, alternative models with similar functionality and accuracy are available. Traditional accuracy-based evaluations often fail to capture behavioral differences between models, especially with limited test datasets, making it difficult to select or combine models effectively. Differential testing addresses this by generating test inputs that expose discrepancies in DNN model behavior. However, existing approaches face significant limitations: many rely on model internals or are constrained by available seed inputs. To address these challenges, we propose DiffGAN, a black-box test image generation approach for differential testing of DNN models. DiffGAN leverages a Generative Adversarial Network (GAN) and the Non-dominated Sorting Genetic Algorithm II to generate diverse and valid triggering inputs that reveal behavioral discrepancies between models. DiffGAN employs two custom fitness functions, focusing on diversity and divergence, to guide the exploration of the GAN input space and identify discrepancies between models' outputs. By strategically searching this space, DiffGAN generates inputs with specific features that trigger differences in model behavior. DiffGAN is black-box, making it applicable in more situations. We evaluate DiffGAN on eight DNN model pairs trained on widely used image datasets. Our results show DiffGAN significantly outperforms a SOTA baseline, generating four times more triggering inputs, with greater diversity and validity, within the same budget. Additionally, the generated inputs improve the accuracy of a machine learning-based model selection mechanism, which selects the best-performing model based on input characteristics and can serve as a smart output voting mechanism when using alternative models.
>
---
#### [replaced 008] Improving Generalizability of Kolmogorov-Arnold Networks via Error-Correcting Output Codes
- **分类: cs.LG; cs.CV; eess.IV; eess.SP**

- **链接: [http://arxiv.org/pdf/2505.05798v2](http://arxiv.org/pdf/2505.05798v2)**

> **作者:** Youngjoon Lee; Jinu Gong; Joonhyuk Kang
>
> **备注:** Accepted to IEEE BioCAS 2025
>
> **摘要:** Kolmogorov-Arnold Networks (KAN) offer universal function approximation using univariate spline compositions without nonlinear activations. In this work, we integrate Error-Correcting Output Codes (ECOC) into the KAN framework to transform multi-class classification into multiple binary tasks, improving robustness via Hamming distance decoding. Our proposed KAN with ECOC framework outperforms vanilla KAN on a challenging blood cell classification dataset, achieving higher accuracy across diverse hyperparameter settings. Ablation studies further confirm that ECOC consistently enhances performance across FastKAN and FasterKAN variants. These results demonstrate that ECOC integration significantly boosts KAN generalizability in critical healthcare AI applications. To the best of our knowledge, this is the first work of ECOC with KAN for enhancing multi-class medical image classification performance.
>
---
#### [replaced 009] SCALP: Superpixels with Contour Adherence using Linear Path
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/1903.07149v2](http://arxiv.org/pdf/1903.07149v2)**

> **作者:** Rémi Giraud; Vinh-Thong Ta; Nicolas Papadakis
>
> **备注:** International Conference on Pattern Recognition (ICPR) 2016
>
> **摘要:** Superpixel decomposition methods are generally used as a pre-processing step to speed up image processing tasks. They group the pixels of an image into homogeneous regions while trying to respect existing contours. For all state-of-the-art superpixel decomposition methods, a trade-off is made between 1) computational time, 2) adherence to image contours and 3) regularity and compactness of the decomposition. In this paper, we propose a fast method to compute Superpixels with Contour Adherence using Linear Path (SCALP) in an iterative clustering framework. The distance computed when trying to associate a pixel to a superpixel during the clustering is enhanced by considering the linear path to the superpixel barycenter. The proposed framework produces regular and compact superpixels that adhere to the image contours. We provide a detailed evaluation of SCALP on the standard Berkeley Segmentation Dataset. The obtained results outperform state-of-the-art methods in terms of standard superpixel and contour detection metrics.
>
---
#### [replaced 010] Superpixel-based Color Transfer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/1903.06010v2](http://arxiv.org/pdf/1903.06010v2)**

> **作者:** Rémi Giraud; Vinh-Thong Ta; Nicolas Papadakis
>
> **摘要:** In this work, we propose a fast superpixel-based color transfer method (SCT) between two images. Superpixels enable to decrease the image dimension and to extract a reduced set of color candidates. We propose to use a fast approximate nearest neighbor matching algorithm in which we enforce the match diversity by limiting the selection of the same superpixels. A fusion framework is designed to transfer the matched colors, and we demonstrate the improvement obtained over exact matching results. Finally, we show that SCT is visually competitive compared to state-of-the-art methods.
>
---
#### [replaced 011] Kling-Avatar: Grounding Multimodal Instructions for Cascaded Long-Duration Avatar Animation Synthesis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.09595v2](http://arxiv.org/pdf/2509.09595v2)**

> **作者:** Yikang Ding; Jiwen Liu; Wenyuan Zhang; Zekun Wang; Wentao Hu; Liyuan Cui; Mingming Lao; Yingchao Shao; Hui Liu; Xiaohan Li; Ming Chen; Xiaoqiang Liu; Yu-Shen Liu; Pengfei Wan
>
> **备注:** Technical Report. Project Page: https://klingavatar.github.io/
>
> **摘要:** Recent advances in audio-driven avatar video generation have significantly enhanced audio-visual realism. However, existing methods treat instruction conditioning merely as low-level tracking driven by acoustic or visual cues, without modeling the communicative purpose conveyed by the instructions. This limitation compromises their narrative coherence and character expressiveness. To bridge this gap, we introduce Kling-Avatar, a novel cascaded framework that unifies multimodal instruction understanding with photorealistic portrait generation. Our approach adopts a two-stage pipeline. In the first stage, we design a multimodal large language model (MLLM) director that produces a blueprint video conditioned on diverse instruction signals, thereby governing high-level semantics such as character motion and emotions. In the second stage, guided by blueprint keyframes, we generate multiple sub-clips in parallel using a first-last frame strategy. This global-to-local framework preserves fine-grained details while faithfully encoding the high-level intent behind multimodal instructions. Our parallel architecture also enables fast and stable generation of long-duration videos, making it suitable for real-world applications such as digital human livestreaming and vlogging. To comprehensively evaluate our method, we construct a benchmark of 375 curated samples covering diverse instructions and challenging scenarios. Extensive experiments demonstrate that Kling-Avatar is capable of generating vivid, fluent, long-duration videos at up to 1080p and 48 fps, achieving superior performance in lip synchronization accuracy, emotion and dynamic expressiveness, instruction controllability, identity preservation, and cross-domain generalization. These results establish Kling-Avatar as a new benchmark for semantically grounded, high-fidelity audio-driven avatar synthesis.
>
---
#### [replaced 012] TrajBooster: Boosting Humanoid Whole-Body Manipulation via Trajectory-Centric Learning
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.11839v2](http://arxiv.org/pdf/2509.11839v2)**

> **作者:** Jiacheng Liu; Pengxiang Ding; Qihang Zhou; Yuxuan Wu; Da Huang; Zimian Peng; Wei Xiao; Weinan Zhang; Lixin Yang; Cewu Lu; Donglin Wang
>
> **摘要:** Recent Vision-Language-Action models show potential to generalize across embodiments but struggle to quickly align with a new robot's action space when high-quality demonstrations are scarce, especially for bipedal humanoids. We present TrajBooster, a cross-embodiment framework that leverages abundant wheeled-humanoid data to boost bipedal VLA. Our key idea is to use end-effector trajectories as a morphology-agnostic interface. TrajBooster (i) extracts 6D dual-arm end-effector trajectories from real-world wheeled humanoids, (ii) retargets them in simulation to Unitree G1 with a whole-body controller trained via a heuristic-enhanced harmonized online DAgger to lift low-dimensional trajectory references into feasible high-dimensional whole-body actions, and (iii) forms heterogeneous triplets that couple source vision/language with target humanoid-compatible actions to post-pre-train a VLA, followed by only 10 minutes of teleoperation data collection on the target humanoid domain. Deployed on Unitree G1, our policy achieves beyond-tabletop household tasks, enabling squatting, cross-height manipulation, and coordinated whole-body motion with markedly improved robustness and generalization. Results show that TrajBooster allows existing wheeled-humanoid data to efficiently strengthen bipedal humanoid VLA performance, reducing reliance on costly same-embodiment data while enhancing action space understanding and zero-shot skill transfer capabilities. For more details, For more details, please refer to our \href{https://jiachengliu3.github.io/TrajBooster/}.
>
---
#### [replaced 013] Brain age identification from diffusion MRI synergistically predicts neurodegenerative disease
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.22454v3](http://arxiv.org/pdf/2410.22454v3)**

> **作者:** Chenyu Gao; Michael E. Kim; Karthik Ramadass; Praitayini Kanakaraj; Aravind R. Krishnan; Adam M. Saunders; Nancy R. Newlin; Ho Hin Lee; Qi Yang; Warren D. Taylor; Brian D. Boyd; Lori L. Beason-Held; Susan M. Resnick; Lisa L. Barnes; David A. Bennett; Marilyn S. Albert; Katherine D. Van Schaik; Derek B. Archer; Timothy J. Hohman; Angela L. Jefferson; Ivana Išgum; Daniel Moyer; Yuankai Huo; Kurt G. Schilling; Lianrui Zuo; Shunxing Bao; Nazirah Mohd Khairi; Zhiyuan Li; Christos Davatzikos; Bennett A. Landman
>
> **备注:** Accepted to Imaging Neuroscience
>
> **摘要:** Estimated brain age from magnetic resonance image (MRI) and its deviation from chronological age can provide early insights into potential neurodegenerative diseases, supporting early detection and implementation of prevention strategies. Diffusion MRI (dMRI) presents an opportunity to build an earlier biomarker for neurodegenerative disease prediction because it captures subtle microstructural changes that precede more perceptible macrostructural changes. However, the coexistence of macro- and micro-structural information in dMRI raises the question of whether current dMRI-based brain age estimation models are leveraging the intended microstructural information or if they inadvertently rely on the macrostructural information. To develop a microstructure-specific brain age, we propose a method for brain age identification from dMRI that mitigates the model's use of macrostructural information by non-rigidly registering all images to a standard template. Imaging data from 13,398 participants across 12 datasets were used for the training and evaluation. We compare our brain age models, trained with and without macrostructural information mitigated, with an architecturally similar T1-weighted (T1w) MRI-based brain age model and two recent, popular, openly available T1w MRI-based brain age models that primarily use macrostructural information. We observe difference between our dMRI-based brain age and T1w MRI-based brain age across stages of neurodegeneration, with dMRI-based brain age being older than T1w MRI-based brain age in participants transitioning from cognitively normal (CN) to mild cognitive impairment (MCI), but younger in participants already diagnosed with Alzheimer's disease (AD). Furthermore, dMRI-based brain age may offer advantages over T1w MRI-based brain age in predicting the transition from CN to MCI up to five years before diagnosis.
>
---
#### [replaced 014] Locally Explaining Prediction Behavior via Gradual Interventions and Measuring Property Gradients
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.05424v2](http://arxiv.org/pdf/2503.05424v2)**

> **作者:** Niklas Penzel; Joachim Denzler
>
> **备注:** Accepted at WACV-2026, 45 pages, 39 figures, 15 tables
>
> **摘要:** Deep learning models achieve high predictive performance but lack intrinsic interpretability, hindering our understanding of the learned prediction behavior. Existing local explainability methods focus on associations, neglecting the causal drivers of model predictions. Other approaches adopt a causal perspective but primarily provide global, model-level explanations. However, for specific inputs, it's unclear whether globally identified factors apply locally. To address this limitation, we introduce a novel framework for local interventional explanations by leveraging recent advances in image-to-image editing models. Our approach performs gradual interventions on semantic properties to quantify the corresponding impact on a model's predictions using a novel score, the expected property gradient magnitude. We demonstrate the effectiveness of our approach through an extensive empirical evaluation on a wide range of architectures and tasks. First, we validate it in a synthetic scenario and demonstrate its ability to locally identify biases. Afterward, we apply our approach to investigate medical skin lesion classifiers, analyze network training dynamics, and study a pre-trained CLIP model with real-life interventional data. Our results highlight the potential of interventional explanations on the property level to reveal new insights into the behavior of deep models.
>
---
#### [replaced 015] A Culturally-diverse Multilingual Multimodal Video Benchmark & Model
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.07032v2](http://arxiv.org/pdf/2506.07032v2)**

> **作者:** Bhuiyan Sanjid Shafique; Ashmal Vayani; Muhammad Maaz; Hanoona Abdul Rasheed; Dinura Dissanayake; Mohammed Irfan Kurpath; Yahya Hmaiti; Go Inoue; Jean Lahoud; Md. Safirur Rashid; Shadid Intisar Quasem; Maheen Fatima; Franco Vidal; Mykola Maslych; Ketan Pravin More; Sanoojan Baliah; Hasindri Watawana; Yuhao Li; Fabian Farestam; Leon Schaller; Roman Tymtsiv; Simon Weber; Hisham Cholakkal; Ivan Laptev; Shin'ichi Satoh; Michael Felsberg; Mubarak Shah; Salman Khan; Fahad Shahbaz Khan
>
> **摘要:** Large multimodal models (LMMs) have recently gained attention due to their effectiveness to understand and generate descriptions of visual content. Most existing LMMs are in English language. While few recent works explore multilingual image LMMs, to the best of our knowledge, moving beyond the English language for cultural and linguistic inclusivity is yet to be investigated in the context of video LMMs. In pursuit of more inclusive video LMMs, we introduce a multilingual Video LMM benchmark, named ViMUL-Bench, to evaluate Video LMMs across 14 languages, including both low- and high-resource languages: English, Chinese, Spanish, French, German, Hindi, Arabic, Russian, Bengali, Urdu, Sinhala, Tamil, Swedish, and Japanese. Our ViMUL-Bench is designed to rigorously test video LMMs across 15 categories including eight culturally diverse categories, ranging from lifestyles and festivals to foods and rituals and from local landmarks to prominent cultural personalities. ViMUL-Bench comprises both open-ended (short and long-form) and multiple-choice questions spanning various video durations (short, medium, and long) with 8k samples that are manually verified by native language speakers. In addition, we also introduce a machine translated multilingual video training set comprising 1.2 million samples and develop a simple multilingual video LMM, named ViMUL, that is shown to provide a better tradeoff between high-and low-resource languages for video understanding. We hope our ViMUL-Bench and multilingual video LMM along with a large-scale multilingual video training set will help ease future research in developing cultural and linguistic inclusive multilingual video LMMs. Our proposed benchmark, video LMM and training data will be publicly released at https://mbzuai-oryx.github.io/ViMUL/.
>
---
#### [replaced 016] Identity-Preserving Text-to-Video Generation Guided by Simple yet Effective Spatial-Temporal Decoupled Representations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.04705v2](http://arxiv.org/pdf/2507.04705v2)**

> **作者:** Yuji Wang; Moran Li; Xiaobin Hu; Ran Yi; Jiangning Zhang; Han Feng; Weijian Cao; Yabiao Wang; Chengjie Wang; Lizhuang Ma
>
> **摘要:** Identity-preserving text-to-video (IPT2V) generation, which aims to create high-fidelity videos with consistent human identity, has become crucial for downstream applications. However, current end-to-end frameworks suffer a critical spatial-temporal trade-off: optimizing for spatially coherent layouts of key elements (e.g., character identity preservation) often compromises instruction-compliant temporal smoothness, while prioritizing dynamic realism risks disrupting the spatial coherence of visual structures. To tackle this issue, we propose a simple yet effective spatial-temporal decoupled framework that decomposes representations into spatial features for layouts and temporal features for motion dynamics. Specifically, our paper proposes a semantic prompt optimization mechanism and stage-wise decoupled generation paradigm. The former module decouples the prompt into spatial and temporal components. Aligned with the subsequent stage-wise decoupled approach, the spatial prompts guide the text-to-image (T2I) stage to generate coherent spatial features, while the temporal prompts direct the sequential image-to-video (I2V) stage to ensure motion consistency. Experimental results validate that our approach achieves excellent spatiotemporal consistency, demonstrating outstanding performance in identity preservation, text relevance, and video quality. By leveraging this simple yet robust mechanism, our algorithm secures the runner-up position in 2025 ACM MultiMedia Challenge.
>
---
#### [replaced 017] UniPLV: Towards Label-Efficient Open-World 3D Scene Understanding by Regional Visual Language Supervision
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.18131v2](http://arxiv.org/pdf/2412.18131v2)**

> **作者:** Yuru Wang; Pei Liu; Songtao Wang; Zehan Zhang; Xinyan Lu; Changwei Cai; Hao Li; Fu Liu; Peng Jia; Xianpeng Lang
>
> **摘要:** Open-world 3D scene understanding is a critical challenge that involves recognizing and distinguishing diverse objects and categories from 3D data, such as point clouds, without relying on manual annotations. Traditional methods struggle with this open-world task, especially due to the limitations of constructing extensive point cloud-text pairs and handling multimodal data effectively. In response to these challenges, we present UniPLV, a robust framework that unifies point clouds, images, and text within a single learning paradigm for comprehensive 3D scene understanding. UniPLV leverages images as a bridge to co-embed 3D points with pre-aligned images and text in a shared feature space, eliminating the need for labor-intensive point cloud-text pair crafting. Our framework achieves precise multimodal alignment through two innovative strategies: (i) Logit and feature distillation modules between images and point clouds to enhance feature coherence; (ii) A vision-point matching module that implicitly corrects 3D semantic predictions affected by projection inaccuracies from points to pixels. To further boost performance, we implement four task-specific losses alongside a two-stage training strategy. Extensive experiments demonstrate that UniPLV significantly surpasses state-of-the-art methods, with average improvements of 15.6% and 14.8% in semantic segmentation for Base-Annotated and Annotation-Free tasks, respectively. These results underscore UniPLV's efficacy in pushing the boundaries of open-world 3D scene understanding. We will release the code to support future research and development.
>
---
#### [replaced 018] GROOD: GRadient-Aware Out-of-Distribution Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2312.14427v3](http://arxiv.org/pdf/2312.14427v3)**

> **作者:** Mostafa ElAraby; Sabyasachi Sahoo; Yann Pequignot; Paul Novello; Liam Paull
>
> **备注:** Accepted to Transactions on Machine Learning Research (TMLR) 2025. 12 pages, 5 figures, 7 tables
>
> **摘要:** Out-of-distribution (OOD) detection is crucial for ensuring the reliability of deep learning models in real-world applications. Existing methods typically focus on feature representations or output-space analysis, often assuming a distribution over these spaces or leveraging gradient norms with respect to model parameters. However, these approaches struggle to distinguish near-OOD samples and often require extensive hyper-parameter tuning, limiting their practicality.In this work, we propose GRadient-aware Out-Of-Distribution detection (GROOD), a method that derives an OOD prototype from synthetic samples and computes class prototypes directly from In-distribution (ID) training data. By analyzing the gradients of a nearest-class-prototype loss function concerning an artificial OOD prototype, our approach achieves a clear separation between in-distribution and OOD samples. Experimental evaluations demonstrate that gradients computed from the OOD prototype enhance the distinction between ID and OOD data, surpassing established baselines in robustness, particularly on ImageNet-1k. These findings highlight the potential of gradient-based methods and prototype-driven approaches in advancing OOD detection within deep neural networks.
>
---
#### [replaced 019] Direct Video-Based Spatiotemporal Deep Learning for Cattle Lameness Detection
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2504.16404v3](http://arxiv.org/pdf/2504.16404v3)**

> **作者:** Md Fahimuzzman Sohan; Raid Alzubi; Hadeel Alzoubi; Eid Albalawi; A. H. Abdul Hafez
>
> **摘要:** Cattle lameness is a prevalent health problem in livestock farming, often resulting from hoof injuries or infections, and severely impacts animal welfare and productivity. Early and accurate detection is critical for minimizing economic losses and ensuring proper treatment. This study proposes a spatiotemporal deep learning framework for automated cattle lameness detection using publicly available video data. We curate and publicly release a balanced set of 50 online video clips featuring 42 individual cattle, recorded from multiple viewpoints in both indoor and outdoor environments. The videos were categorized into lame and non-lame classes based on visual gait characteristics and metadata descriptions. After applying data augmentation techniques to enhance generalization, two deep learning architectures were trained and evaluated: 3D Convolutional Neural Networks (3D CNN) and Convolutional Long-Short-Term Memory (ConvLSTM2D). The 3D CNN achieved a video-level classification accuracy of 90%, with a precision, recall, and F1 score of 90.9% each, outperforming the ConvLSTM2D model, which achieved 85% accuracy. Unlike conventional approaches that rely on multistage pipelines involving object detection and pose estimation, this study demonstrates the effectiveness of a direct end-to-end video classification approach. Compared with the best end-to-end prior method (C3D-ConvLSTM, 90.3%), our model achieves comparable accuracy while eliminating pose estimation pre-processing.The results indicate that deep learning models can successfully extract and learn spatio-temporal features from various video sources, enabling scalable and efficient cattle lameness detection in real-world farm settings.
>
---
#### [replaced 020] Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2503.14171v2](http://arxiv.org/pdf/2503.14171v2)**

> **作者:** Simon Niedermayr; Christoph Neuhauser Rüdiger Westermann
>
> **摘要:** We introduce an image upscaling technique tailored for 3D Gaussian Splatting (3DGS) on lightweight GPUs. Compared to 3DGS, it achieves significantly higher rendering speeds and reduces artifacts commonly observed in 3DGS reconstructions. Our technique upscales low-resolution 3DGS renderings with a marginal increase in cost by directly leveraging the analytical image gradients of Gaussians for gradient-based bicubic spline interpolation. The technique is agnostic to the specific 3DGS implementation, achieving novel view synthesis at rates 3x-4x higher than the baseline implementation. Through extensive experiments on multiple datasets, we showcase the performance improvements and high reconstruction fidelity attainable with gradient-aware upscaling of 3DGS images. We further demonstrate the integration of gradient-aware upscaling into the gradient-based optimization of a 3DGS model and analyze its effects on reconstruction quality and performance.
>
---
#### [replaced 021] xGen-MM (BLIP-3): A Family of Open Large Multimodal Models
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2408.08872v4](http://arxiv.org/pdf/2408.08872v4)**

> **作者:** Le Xue; Manli Shu; Anas Awadalla; Jun Wang; An Yan; Senthil Purushwalkam; Honglu Zhou; Viraj Prabhu; Yutong Dai; Michael S Ryoo; Shrikant Kendre; Jieyu Zhang; Shaoyen Tseng; Gustavo A Lujan-Moreno; Matthew L Olson; Musashi Hinck; David Cobbley; Vasudev Lal; Can Qin; Shu Zhang; Chia-Chih Chen; Ning Yu; Juntao Tan; Tulika Manoj Awalgaonkar; Shelby Heinecke; Huan Wang; Yejin Choi; Ludwig Schmidt; Zeyuan Chen; Silvio Savarese; Juan Carlos Niebles; Caiming Xiong; Ran Xu
>
> **摘要:** This paper introduces BLIP-3, an open framework for developing Large Multimodal Models (LMMs). The framework comprises meticulously curated datasets, a training recipe, model architectures, and a resulting suite of LMMs. We release 4B and 14B models, including both the pre-trained base model and the instruction fine-tuned ones. Our models undergo rigorous evaluation across a range of tasks, including both single and multi-image benchmarks. Our models demonstrate competitive performance among open-source LMMs with similar model sizes. Our resulting LMMs demonstrate competitive performance among open-source LMMs with similar model sizes, with the ability to comprehend interleaved image-text inputs. Our training code, models, and all datasets used in this work, including the three largescale datasets we create and the preprocessed ones, will be open-sourced to better support the research community.
>
---
#### [replaced 022] Embodied Image Captioning: Self-supervised Learning Agents for Spatially Coherent Image Descriptions
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.08531v2](http://arxiv.org/pdf/2504.08531v2)**

> **作者:** Tommaso Galliena; Tommaso Apicella; Stefano Rosa; Pietro Morerio; Alessio Del Bue; Lorenzo Natale
>
> **备注:** 11 pages, 8 figures, 6 tables, code and test set annotations available at https://hsp-iit.github.io/embodied-captioning/
>
> **摘要:** We present a self-supervised method to improve an agent's abilities in describing arbitrary objects while actively exploring a generic environment. This is a challenging problem, as current models struggle to obtain coherent image captions due to different camera viewpoints and clutter. We propose a three-phase framework to fine-tune existing captioning models that enhances caption accuracy and consistency across views via a consensus mechanism. First, an agent explores the environment, collecting noisy image-caption pairs. Then, a consistent pseudo-caption for each object instance is distilled via consensus using a large language model. Finally, these pseudo-captions are used to fine-tune an off-the-shelf captioning model, with the addition of contrastive learning. We analyse the performance of the combination of captioning models, exploration policies, pseudo-labeling methods, and fine-tuning strategies, on our manually labeled test set. Results show that a policy can be trained to mine samples with higher disagreement compared to classical baselines. Our pseudo-captioning method, in combination with all policies, has a higher semantic similarity compared to other existing methods, and fine-tuning improves caption accuracy and consistency by a significant margin. Code and test set annotations available at https://hsp-iit.github.io/embodied-captioning/
>
---
#### [replaced 023] Synthesis and Perceptual Scaling of High Resolution Naturalistic Images Using Stable Diffusion
- **分类: q-bio.NC; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.13034v2](http://arxiv.org/pdf/2410.13034v2)**

> **作者:** Leonardo Pettini; Carsten Bogler; Christian Doeller; John-Dylan Haynes
>
> **备注:** 80 pages, 26 Figures, 6 tables
>
> **摘要:** Naturalistic scenes are of key interest for visual perception, but controlling their perceptual and semantic properties is challenging. Previous work on naturalistic scenes has frequently focused on collections of discrete images with considerable physical differences between stimuli. However, it is often desirable to assess representations of naturalistic images that vary along a continuum. Traditionally, perceptually continuous variations of naturalistic stimuli have been obtained by morphing a source image into a target image. This produces transitions driven mainly by low-level physical features and can result in semantically ambiguous outcomes. More recently, generative adversarial networks (GANs) have been used to generate continuous perceptual variations within a stimulus category. Here we extend and generalize this approach using a different machine learning approach, a text-to-image diffusion model (Stable Diffusion XL), to generate a freely customizable stimulus set of photorealistic images that are characterized by gradual transitions, with each image representing a unique exemplar within a prompted category. We demonstrate the approach by generating a set of 108 object scenes from 6 categories. For each object scene, we generate 10 variants that are ordered along a perceptual continuum. This ordering was first estimated using a machine learning model of perceptual similarity (LPIPS) and then subsequently validated with a large online sample of human participants. In a subsequent experiment we show that this ordering is also predictive of confusability of stimuli in a working memory experiment. Our image set is suited for studies investigating the graded encoding of naturalistic stimuli in visual perception, attention, and memory.
>
---
#### [replaced 024] Texture-Aware Superpixel Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/1901.11111v4](http://arxiv.org/pdf/1901.11111v4)**

> **作者:** Remi Giraud; Vinh-Thong Ta; Nicolas Papadakis; Yannick Berthoumieu
>
> **摘要:** Most superpixel algorithms compute a trade-off between spatial and color features at the pixel level. Hence, they may need fine parameter tuning to balance the two measures, and highly fail to group pixels with similar local texture properties. In this paper, we address these issues with a new Texture-Aware SuperPixel (TASP) method. To accurately segment textured and smooth areas, TASP automatically adjusts its spatial constraint according to the local feature variance. Then, to ensure texture homogeneity within superpixels, a new pixel to superpixel patch-based distance is proposed. TASP outperforms the segmentation accuracy of the state-of-the-art methods on texture and also natural color image datasets.
>
---
#### [replaced 025] MEGANet-W: A Wavelet-Driven Edge-Guided Attention Framework for Weak Boundary Polyp Detection
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.02668v4](http://arxiv.org/pdf/2507.02668v4)**

> **作者:** Zhe Yee Tan; Ashwaq Qasem
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Colorectal polyp segmentation is critical for early detection of colorectal cancer, yet weak and low contrast boundaries significantly limit automated accuracy. Existing deep models either blur fine edge details or rely on handcrafted filters that perform poorly under variable imaging conditions. We propose MEGANet-W, a Wavelet Driven Edge Guided Attention Network that injects directional, parameter free Haar wavelet edge maps into each decoder stage to recalibrate semantic features. The key novelties of MEGANet-W include a two-level Haar wavelet head for multi-orientation edge extraction; and Wavelet Edge Guided Attention (W-EGA) modules that fuse wavelet cues with boundary and input branches. On five public polyp datasets, MEGANet-W consistently outperforms existing methods, improving mIoU by up to 2.3% and mDice by 1.2%, while introducing no additional learnable parameters. This approach improves reliability in difficult cases and offers a robust solution for medical image segmentation tasks requiring precise boundary detection.
>
---
#### [replaced 026] Puzzled by Puzzles: When Vision-Language Models Can't Take a Hint
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.23759v2](http://arxiv.org/pdf/2505.23759v2)**

> **作者:** Heekyung Lee; Jiaxin Ge; Tsung-Han Wu; Minwoo Kang; Trevor Darrell; David M. Chan
>
> **备注:** EMNLP 2025 Main Conference
>
> **摘要:** Rebus puzzles, visual riddles that encode language through imagery, spatial arrangement, and symbolic substitution, pose a unique challenge to current vision-language models (VLMs). Unlike traditional image captioning or question answering tasks, rebus solving requires multi-modal abstraction, symbolic reasoning, and a grasp of cultural, phonetic and linguistic puns. In this paper, we investigate the capacity of contemporary VLMs to interpret and solve rebus puzzles by constructing a hand-generated and annotated benchmark of diverse English-language rebus puzzles, ranging from simple pictographic substitutions to spatially-dependent cues ("head" over "heels"). We analyze how different VLMs perform, and our findings reveal that while VLMs exhibit some surprising capabilities in decoding simple visual clues, they struggle significantly with tasks requiring abstract reasoning, lateral thinking, and understanding visual metaphors.
>
---
#### [replaced 027] Video-Foley: Two-Stage Video-To-Sound Generation via Temporal Event Condition For Foley Sound
- **分类: cs.SD; cs.CV; cs.LG; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2408.11915v3](http://arxiv.org/pdf/2408.11915v3)**

> **作者:** Junwon Lee; Jaekwon Im; Dabin Kim; Juhan Nam
>
> **备注:** Accepted at IEEE/ACM Transactions on Audio, Speech and Language Processing (TASLP)
>
> **摘要:** Foley sound synthesis is crucial for multimedia production, enhancing user experience by synchronizing audio and video both temporally and semantically. Recent studies on automating this labor-intensive process through video-to-sound generation face significant challenges. Systems lacking explicit temporal features suffer from poor alignment and controllability, while timestamp-based models require costly and subjective human annotation. We propose Video-Foley, a video-to-sound system using Root Mean Square (RMS) as an intuitive condition with semantic timbre prompts (audio or text). RMS, a frame-level intensity envelope closely related to audio semantics, acts as a temporal event feature to guide audio generation from video. The annotation-free self-supervised learning framework consists of two stages, Video2RMS and RMS2Sound, incorporating novel ideas including RMS discretization and RMS-ControlNet with a pretrained text-to-audio model. Our extensive evaluation shows that Video-Foley achieves state-of-the-art performance in audio-visual alignment and controllability for sound timing, intensity, timbre, and nuance. Source code, model weights and demos are available on our companion website. (https://jnwnlee.github.io/video-foley-demo)
>
---
#### [replaced 028] Leveraging Perceptual Scores for Dataset Pruning in Computer Vision Tasks
- **分类: cs.CV; cs.IT; math.IT**

- **链接: [http://arxiv.org/pdf/2408.07243v2](http://arxiv.org/pdf/2408.07243v2)**

> **作者:** Raghavendra Singh
>
> **备注:** NON ARCHIVAL PRESENTATION 1st workshop on Dataset Distillation CVPR 2024
>
> **摘要:** In this paper we propose a score of an image to use for coreset selection in image classification and semantic segmentation tasks. The score is the entropy of an image as approximated by the bits-per-pixel of its compressed version. Thus the score is intrinsic to an image and does not require supervision or training. It is very simple to compute and readily available as all images are stored in a compressed format. The motivation behind our choice of score is that most other scores proposed in literature are expensive to compute. More importantly, we want a score that captures the perceptual complexity of an image. Entropy is one such measure, images with clutter tend to have a higher entropy. However sampling only low entropy iconic images, for example, leads to biased learning and an overall decrease in test performance with current deep learning models. To mitigate the bias we use a graph based method that increases the spatial diversity of the selected samples. We show that this simple score yields good results, particularly for semantic segmentation tasks.
>
---
#### [replaced 029] A Novel Compression Framework for YOLOv8: Achieving Real-Time Aerial Object Detection on Edge Devices via Structured Pruning and Channel-Wise Distillation
- **分类: cs.CV; 68T07; I.4.8**

- **链接: [http://arxiv.org/pdf/2509.12918v2](http://arxiv.org/pdf/2509.12918v2)**

> **作者:** Melika Sabaghian; Mohammad Ali Keyvanrad; Seyyedeh Mahila Moghadami
>
> **备注:** 28 pages, 11 figures
>
> **摘要:** Efficient deployment of deep learning models for aerial object detection on resource-constrained devices requires significant compression without com-promising performance. In this study, we propose a novel three-stage compression pipeline for the YOLOv8 object detection model, integrating sparsity-aware training, structured channel pruning, and Channel-Wise Knowledge Distillation (CWD). First, sparsity-aware training introduces dynamic sparsity during model optimization, effectively balancing parameter reduction and detection accuracy. Second, we apply structured channel pruning by leveraging batch normalization scaling factors to eliminate redundant channels, significantly reducing model size and computational complexity. Finally, to mitigate the accuracy drop caused by pruning, we employ CWD to transfer knowledge from the original model, using an adjustable temperature and loss weighting scheme tailored for small and medium object detection. Extensive experiments on the VisDrone dataset demonstrate the effectiveness of our approach across multiple YOLOv8 variants. For YOLOv8m, our method reduces model parameters from 25.85M to 6.85M (a 73.51% reduction), FLOPs from 49.6G to 13.3G, and MACs from 101G to 34.5G, while reducing AP50 by only 2.7%. The resulting compressed model achieves 47.9 AP50 and boosts inference speed from 26 FPS (YOLOv8m baseline) to 45 FPS, enabling real-time deployment on edge devices. We further apply TensorRT as a lightweight optimization step. While this introduces a minor drop in AP50 (from 47.9 to 47.6), it significantly improves inference speed from 45 to 68 FPS, demonstrating the practicality of our approach for high-throughput, re-source-constrained scenarios.
>
---
#### [replaced 030] Well-Conditioned Polynomial Representations for Mathematical Handwriting Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.10815v2](http://arxiv.org/pdf/2509.10815v2)**

> **作者:** Robert M. Corless; Deepak Singh Kalhan; Stephen M. Watt
>
> **摘要:** Previous work has made use of a parameterized plane curve polynomial representation for mathematical handwriting, with the polynomials represented in a Legendre or Legendre-Sobolev graded basis. This provides a compact geometric representation for the digital ink. Preliminary results have also been shown for Chebyshev and Chebyshev-Sobolev bases. This article explores the trade-offs between basis choice and polynomial degree to achieve accurate modeling with a low computational cost. To do this, we consider the condition number for polynomial evaluation in these bases and bound how the various inner products give norms for the variations between symbols.
>
---
#### [replaced 031] StyleSculptor: Zero-Shot Style-Controllable 3D Asset Generation with Texture-Geometry Dual Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.13301v2](http://arxiv.org/pdf/2509.13301v2)**

> **作者:** Zefan Qu; Zhenwei Wang; Haoyuan Wang; Ke Xu; Gerhard Hancke; Rynson W. H. Lau
>
> **备注:** SIGGRAPH Asia 2025, Project page:https://stylesculptor.github.io
>
> **摘要:** Creating 3D assets that follow the texture and geometry style of existing ones is often desirable or even inevitable in practical applications like video gaming and virtual reality. While impressive progress has been made in generating 3D objects from text or images, creating style-controllable 3D assets remains a complex and challenging problem. In this work, we propose StyleSculptor, a novel training-free approach for generating style-guided 3D assets from a content image and one or more style images. Unlike previous works, StyleSculptor achieves style-guided 3D generation in a zero-shot manner, enabling fine-grained 3D style control that captures the texture, geometry, or both styles of user-provided style images. At the core of StyleSculptor is a novel Style Disentangled Attention (SD-Attn) module, which establishes a dynamic interaction between the input content image and style image for style-guided 3D asset generation via a cross-3D attention mechanism, enabling stable feature fusion and effective style-guided generation. To alleviate semantic content leakage, we also introduce a style-disentangled feature selection strategy within the SD-Attn module, which leverages the variance of 3D feature patches to disentangle style- and content-significant channels, allowing selective feature injection within the attention framework. With SD-Attn, the network can dynamically compute texture-, geometry-, or both-guided features to steer the 3D generation process. Built upon this, we further propose the Style Guided Control (SGC) mechanism, which enables exclusive geometry- or texture-only stylization, as well as adjustable style intensity control. Extensive experiments demonstrate that StyleSculptor outperforms existing baseline methods in producing high-fidelity 3D assets.
>
---
#### [replaced 032] Data-Efficient Fine-Tuning of Vision-Language Models for Diagnosis of Alzheimer's Disease
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.07613v2](http://arxiv.org/pdf/2509.07613v2)**

> **作者:** Fangqi Cheng; Surajit Ray; Xiaochen Yang
>
> **摘要:** Medical vision-language models (Med-VLMs) have shown impressive results in tasks such as report generation and visual question answering, but they still face several limitations. Most notably, they underutilize patient metadata and lack integration of clinical diagnostic knowledge. Moreover, most existing models are typically trained from scratch or fine-tuned on large-scale 2D image-text pairs, requiring extensive computational resources, and their effectiveness on 3D medical imaging is often limited due to the absence of structural information. To address these gaps, we propose a data-efficient fine-tuning pipeline to adapt 3D CT-based Med-VLMs for 3D MRI and demonstrate its application in Alzheimer's disease (AD) diagnosis. Our system introduces two key innovations. First, we convert structured metadata into synthetic reports, enriching textual input for improved image-text alignment. Second, we add an auxiliary token trained to predict the mini-mental state examination (MMSE) score, a widely used clinical measure of cognitive function that correlates with AD severity. This provides additional supervision for fine-tuning. Applying lightweight prompt tuning to both image and text modalities, our approach achieves state-of-the-art performance on two AD datasets using 1,500 training images, outperforming existing methods fine-tuned on 10,000 images. Code will be released upon publication.
>
---
#### [replaced 033] A Deep Learning Pipeline for Solid Waste Detection in Remote Sensing Images
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.06607v4](http://arxiv.org/pdf/2502.06607v4)**

> **作者:** Federico Gibellini; Piero Fraternali; Giacomo Boracchi; Luca Morandini; Thomas Martinoli; Andrea Diecidue; Simona Malegori
>
> **摘要:** Improper solid waste management represents both a serious threat to ecosystem health and a significant source of revenues for criminal organizations perpetrating environmental crimes. This issue can be mitigated thanks to the increasing availability of Very-High-Resolution Remote Sensing (VHR RS) images. Modern image-analysis tools support automated photo-interpretation and large territory scanning in search of illegal waste disposal sites. This paper illustrates a semi-automatic waste detection pipeline, developed in collaboration with a regional environmental protection agency, for detecting candidate illegal dumping sites in VHR RS images. To optimize the effectiveness of the waste detector at the core of the pipeline, extensive experiments evaluate such design choices as the network architecture, the ground resolution and geographic span of the input images, as well as the pretraining procedures. The best model attains remarkable performance, achieving 92.02 % F1-Score and 94.56 % Accuracy. A generalization study assesses the performance variation when the detector processes images from various territories substantially different from the one used during training, incurring only a moderate performance loss, namely an average 5.1 % decrease in the F1-Score. Finally, an exercise in which expert photo-interpreters compare the effort required to scan large territories with and without support from the waste detector assesses the practical benefit of introducing a computer-aided image analysis tool in a professional environmental protection agency. Results show that a reduction of up to 30 % of the time spent for waste site detection can be attained.
>
---
#### [replaced 034] Effort-Optimized, Accuracy-Driven Labelling and Validation of Test Inputs for DL Systems: A Mixed-Integer Linear Programming Approach
- **分类: cs.CV; cs.SE**

- **链接: [http://arxiv.org/pdf/2507.04990v2](http://arxiv.org/pdf/2507.04990v2)**

> **作者:** Mohammad Hossein Amini; Mehrdad Sabetzadeh; Shiva Nejati
>
> **摘要:** Software systems increasingly include AI components based on deep learning (DL). Reliable testing of such systems requires near-perfect test-input validity and label accuracy, with minimal human effort. Yet, the DL community has largely overlooked the need to build highly accurate datasets with minimal effort, since DL training is generally tolerant of labelling errors. This challenge, instead, reflects concerns more familiar to software engineering, where a central goal is to construct high-accuracy test inputs, with accuracy as close to 100% as possible, while keeping associated costs in check. In this article we introduce OPAL, a human-assisted labelling method that can be configured to target a desired accuracy level while minimizing the manual effort required for labelling. The main contribution of OPAL is a mixed-integer linear programming (MILP) formulation that minimizes labelling effort subject to a specified accuracy target. To evaluate OPAL we instantiate it for two tasks in the context of testing vision systems: automatic labelling of test inputs and automated validation of test inputs. Our evaluation, based on more than 2500 experiments performed on seven datasets, comparing OPAL with eight baseline methods, shows that OPAL, relying on its MILP formulation, achieves an average accuracy of 98.8%, while cutting manual labelling by more than half. OPAL significantly outperforms automated labelling baselines in labelling accuracy across all seven datasets, when all methods are provided with the same manual-labelling budget. For automated test-input validation, on average, OPAL reduces manual effort by 28.8% while achieving 4.5% higher accuracy than the SOTA test-input validation baselines. Finally, we show that augmenting OPAL with an active-learning loop leads to an additional 4.5% reduction in required manual labelling, without compromising accuracy.
>
---
#### [replaced 035] PlaneRecTR++: Unified Query Learning for Joint 3D Planar Reconstruction and Pose Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2307.13756v4](http://arxiv.org/pdf/2307.13756v4)**

> **作者:** Jingjia Shi; Shuaifeng Zhi; Kai Xu
>
> **备注:** To be published in IEEE T-PAMI 2025. This is the journal extension of our ICCV 2023 paper "PlaneRecTR", which expands from single view reconstruction to simultaneous multi-view reconstruction and camera pose estimation. Note that the ICCV2023 PlaneRecTR paper could be found in the previous arxiv version [v2](arXiv:2307.13756v2)
>
> **摘要:** The challenging task of 3D planar reconstruction from images involves several sub-tasks including frame-wise plane detection, segmentation, parameter regression and possibly depth prediction, along with cross-frame plane correspondence and relative camera pose estimation. Previous works adopt a divide and conquer strategy, addressing above sub-tasks with distinct network modules in a two-stage paradigm. Specifically, given an initial camera pose and per-frame plane predictions from the first stage, further exclusively designed modules relying on external plane correspondence labeling are applied to merge multi-view plane entities and produce refined camera pose. Notably, existing work fails to integrate these closely related sub-tasks into a unified framework, and instead addresses them separately and sequentially, which we identify as a primary source of performance limitations. Motivated by this finding and the success of query-based learning in enriching reasoning among semantic entities, in this paper, we propose PlaneRecTR++, a Transformer-based architecture, which for the first time unifies all tasks of multi-view planar reconstruction and pose estimation within a compact single-stage framework, eliminating the need for the initial pose estimation and supervision of plane correspondence. Extensive quantitative and qualitative experiments demonstrate that our proposed unified learning achieves mutual benefits across sub-tasks, achieving a new state-of-the-art performance on the public ScanNetv1, ScanNetv2, NYUv2-Plane, and MatterPort3D datasets. Codes are available at https://github.com/SJingjia/PlaneRecTR-PP.
>
---
#### [replaced 036] Scattering approach to diffusion quantifies axonal damage in brain injury
- **分类: physics.med-ph; cs.CV; physics.bio-ph**

- **链接: [http://arxiv.org/pdf/2501.18167v2](http://arxiv.org/pdf/2501.18167v2)**

> **作者:** Ali Abdollahzadeh; Ricardo Coronado-Leija; Hong-Hsi Lee; Alejandra Sierra; Els Fieremans; Dmitry S. Novikov
>
> **摘要:** Early diagnosis and noninvasive monitoring of neurological disorders require sensitivity to elusive cellular-level alterations that occur much earlier than volumetric changes observable with the millimeter-resolution of medical imaging modalities. Morphological changes in axons, such as axonal varicosities or beadings, are observed in neurological disorders, as well as in development and aging. Here, we reveal the sensitivity of time-dependent diffusion MRI (dMRI) to the structurally disordered axonal morphology at the micrometer scale. Scattering theory uncovers the two parameters that determine the diffusive dynamics of water along axons: the average reciprocal cross-section and the variance of long-range cross-sectional fluctuations. This theoretical development allows us to predict dMRI metrics sensitive to axonal alterations over tens of thousands of axons in seconds rather than months of simulations in a rat model of traumatic brain injury, and is corroborated with ex vivo dMRI. Our approach bridges the gap between micrometers and millimeters in resolution, offering quantitative and objective biomarkers applicable to a broad spectrum of neurological disorders.
>
---
#### [replaced 037] DPDEdit: Detail-Preserved Diffusion Models for Multimodal Fashion Image Editing
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.01086v3](http://arxiv.org/pdf/2409.01086v3)**

> **作者:** Xiaolong Wang; Zhi-Qi Cheng; Jue Wang; Xiaojiang Peng
>
> **备注:** 13 pages,12 figures
>
> **摘要:** Fashion image editing is a crucial tool for designers to convey their creative ideas by visualizing design concepts interactively. Current fashion image editing techniques, though advanced with multimodal prompts and powerful diffusion models, often struggle to accurately identify editing regions and preserve the desired garment texture detail. To address these challenges, we introduce a new multimodal fashion image editing architecture based on latent diffusion models, called Detail-Preserved Diffusion Models (DPDEdit). DPDEdit guides the fashion image generation of diffusion models by integrating text prompts, region masks, human pose images, and garment texture images. To precisely locate the editing region, we first introduce Grounded-SAM to predict the editing region based on the user's textual description, and then combine it with other conditions to perform local editing. To transfer the detail of the given garment texture into the target fashion image, we propose a texture injection and refinement mechanism. Specifically, this mechanism employs a decoupled cross-attention layer to integrate textual descriptions and texture images, and incorporates an auxiliary U-Net to preserve the high-frequency details of generated garment texture. Additionally, we extend the VITON-HD dataset using a multimodal large language model to generate paired samples with texture images and textual descriptions. Extensive experiments show that our DPDEdit outperforms state-of-the-art methods in terms of image fidelity and coherence with the given multimodal inputs.
>
---
#### [replaced 038] GWM: Towards Scalable Gaussian World Models for Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.17600v2](http://arxiv.org/pdf/2508.17600v2)**

> **作者:** Guanxing Lu; Baoxiong Jia; Puhao Li; Yixin Chen; Ziwei Wang; Yansong Tang; Siyuan Huang
>
> **备注:** Published at ICCV 2025. Project page: https://gaussian-world-model.github.io/
>
> **摘要:** Training robot policies within a learned world model is trending due to the inefficiency of real-world interactions. The established image-based world models and policies have shown prior success, but lack robust geometric information that requires consistent spatial and physical understanding of the three-dimensional world, even pre-trained on internet-scale video sources. To this end, we propose a novel branch of world model named Gaussian World Model (GWM) for robotic manipulation, which reconstructs the future state by inferring the propagation of Gaussian primitives under the effect of robot actions. At its core is a latent Diffusion Transformer (DiT) combined with a 3D variational autoencoder, enabling fine-grained scene-level future state reconstruction with Gaussian Splatting. GWM can not only enhance the visual representation for imitation learning agent by self-supervised future prediction training, but can serve as a neural simulator that supports model-based reinforcement learning. Both simulated and real-world experiments depict that GWM can precisely predict future scenes conditioned on diverse robot actions, and can be further utilized to train policies that outperform the state-of-the-art by impressive margins, showcasing the initial data scaling potential of 3D world model.
>
---
#### [replaced 039] Structured Preference Optimization for Vision-Language Long-Horizon Task Planning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20742v4](http://arxiv.org/pdf/2502.20742v4)**

> **作者:** Xiwen Liang; Min Lin; Weiqi Ruan; Rongtao Xu; Yuecheng Liu; Jiaqi Chen; Bingqian Lin; Yuzheng Zhuang; Xiaodan Liang
>
> **备注:** 18 pages
>
> **摘要:** Existing methods for vision-language task planning excel in short-horizon tasks but often fall short in complex, long-horizon planning within dynamic environments. These challenges primarily arise from the difficulty of effectively training models to produce high-quality reasoning processes for long-horizon tasks. To address this, we propose Structured Preference Optimization (SPO), which aims to enhance reasoning and action selection in long-horizon task planning through structured preference evaluation and optimized training strategies. Specifically, SPO introduces: 1) Preference-Based Scoring and Optimization, which systematically evaluates reasoning chains based on task relevance, visual grounding, and historical consistency; and 2) Curriculum-Guided Training, where the model progressively adapts from simple to complex tasks, improving its generalization ability in long-horizon scenarios and enhancing reasoning robustness. To advance research in vision-language long-horizon task planning, we introduce ExtendaBench, a comprehensive benchmark covering 1,509 tasks across VirtualHome and Habitat 2.0, categorized into ultra-short, short, medium, and long tasks. Experimental results demonstrate that SPO significantly improves reasoning quality and final decision accuracy, outperforming prior methods on long-horizon tasks and underscoring the effectiveness of preference-driven optimization in vision-language task planning. Specifically, SPO achieves a +5.98% GCR and +4.68% SR improvement in VirtualHome and a +3.30% GCR and +2.11% SR improvement in Habitat over the best-performing baselines.
>
---
#### [replaced 040] Deep Learning for Crack Detection: A Review of Learning Paradigms, Generalizability, and Datasets
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.10256v2](http://arxiv.org/pdf/2508.10256v2)**

> **作者:** Xinan Zhang; Haolin Wang; Yung-An Hsieh; Zhongyu Yang; Anthony Yezzi; Yi-Chang Tsai
>
> **备注:** under review
>
> **摘要:** Crack detection plays a crucial role in civil infrastructures, including inspection of pavements, buildings, etc., and deep learning has significantly advanced this field in recent years. While numerous technical and review papers exist in this domain, emerging trends are reshaping the landscape. These shifts include transitions in learning paradigms (from fully supervised learning to semi-supervised, weakly-supervised, unsupervised, few-shot, domain adaptation and fine-tuning foundation models), improvements in generalizability (from single-dataset performance to cross-dataset evaluation), and diversification in dataset acquisition (from RGB images to specialized sensor-based data). In this review, we systematically analyze these trends and highlight representative works. Additionally, we introduce a new annotated dataset collected with 3D laser scans, 3DCrack, to support future research and conduct extensive benchmarking experiments to establish baselines for commonly used deep learning methodologies, including recent foundation models. Our findings provide insights into the evolving methodologies and future directions in deep learning-based crack detection. Project page: https://github.com/nantonzhang/Awesome-Crack-Detection
>
---
#### [replaced 041] CROP: Contextual Region-Oriented Visual Token Pruning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21233v2](http://arxiv.org/pdf/2505.21233v2)**

> **作者:** Jiawei Guo; Feifei Zhai; Pu Jian; Qianrun Wei; Yu Zhou
>
> **备注:** EMNLP2025 Main
>
> **摘要:** Current VLM-based VQA methods often process entire images, leading to excessive visual tokens that include redundant information irrelevant to the posed question. This abundance of unnecessary image details creates numerous visual tokens, drastically increasing memory and computational requirements in VLMs. To address this, we propose Contextual Region-Oriented Visual Token Pruning (CROP), a novel framework to compress visual tokens through a two-step process: Localization and Pruning. Specifically, CROP first employs an efficient model to identify the contextual region relevant to the input query. Subsequently, two distinct strategies are introduced for pruning: (1) Pre-LLM Compression (PLC), which adaptively compresses different image regions with varying ratios, and (2) Inner-LLM Pruning (ILP), a training-free method that prunes tokens within early LLM layers guided by the identified contextual region. Extensive experiments on a wide range of VQA tasks demonstrate that CROP significantly outperforms existing visual token pruning methods and achieves state-of-the-art performance.
>
---
#### [replaced 042] Noise2Ghost: Self-supervised deep convolutional reconstruction for ghost imaging
- **分类: cs.CV; cs.LG; physics.data-an**

- **链接: [http://arxiv.org/pdf/2504.10288v2](http://arxiv.org/pdf/2504.10288v2)**

> **作者:** Mathieu Manni; Dmitry Karpov; K. Joost Batenburg; Sharon Shwartz; Nicola Viganò
>
> **摘要:** We present a new self-supervised deep-learning-based Ghost Imaging (GI) reconstruction method, which provides unparalleled reconstruction performance for noisy acquisitions among unsupervised methods. We present the supporting mathematical framework and results from theoretical and real data use cases. Self-supervision removes the need for clean reference data while offering strong noise reduction. This provides the necessary tools for addressing signal-to-noise ratio concerns for GI acquisitions in emerging and cutting-edge low-light GI scenarios. Notable examples include micro- and nano-scale x-ray emission imaging, e.g., x-ray fluorescence imaging of dose-sensitive samples. Their applications include in-vivo and in-operando case studies for biological samples and batteries.
>
---
#### [replaced 043] Video-Language Critic: Transferable Reward Functions for Language-Conditioned Robotics
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.19988v3](http://arxiv.org/pdf/2405.19988v3)**

> **作者:** Minttu Alakuijala; Reginald McLean; Isaac Woungang; Nariman Farsad; Samuel Kaski; Pekka Marttinen; Kai Yuan
>
> **备注:** 14 pages in the main text, 22 pages including references and supplementary materials. 3 figures and 3 tables in the main text, 6 figures and 3 tables in supplementary materials
>
> **摘要:** Natural language is often the easiest and most convenient modality for humans to specify tasks for robots. However, learning to ground language to behavior typically requires impractical amounts of diverse, language-annotated demonstrations collected on each target robot. In this work, we aim to separate the problem of what to accomplish from how to accomplish it, as the former can benefit from substantial amounts of external observation-only data, and only the latter depends on a specific robot embodiment. To this end, we propose Video-Language Critic, a reward model that can be trained on readily available cross-embodiment data using contrastive learning and a temporal ranking objective, and use it to score behavior traces from a separate actor. When trained on Open X-Embodiment data, our reward model enables 2x more sample-efficient policy training on Meta-World tasks than a sparse reward only, despite a significant domain gap. Using in-domain data but in a challenging task generalization setting on Meta-World, we further demonstrate more sample-efficient training than is possible with prior language-conditioned reward models that are either trained with binary classification, use static images, or do not leverage the temporal information present in video data.
>
---
#### [replaced 044] Skyshield: Event-Driven Submillimetre Thin Obstacle Detection for Drone Flight Safety
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.09397v2](http://arxiv.org/pdf/2508.09397v2)**

> **作者:** Zhengli Zhang; Xinyu Luo; Yucheng Sun; Wenhua Ding; Dongyue Huang; Xinlei Chen
>
> **摘要:** Drones operating in complex environments face a significant threat from thin obstacles, such as steel wires and kite strings at the submillimeter level, which are notoriously difficult for conventional sensors like RGB cameras, LiDAR, and depth cameras to detect. This paper introduces SkyShield, an event-driven, end-to-end framework designed for the perception of submillimeter scale obstacles. Drawing upon the unique features that thin obstacles present in the event stream, our method employs a lightweight U-Net architecture and an innovative Dice-Contour Regularization Loss to ensure precise detection. Experimental results demonstrate that our event-based approach achieves mean F1 Score of 0.7088 with a low latency of 21.2 ms, making it ideal for deployment on edge and mobile platforms.
>
---
#### [replaced 045] Robust Shape Regularity Criteria for Superpixel Evaluation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/1903.07146v2](http://arxiv.org/pdf/1903.07146v2)**

> **作者:** Rémi Giraud; Vinh-Thong Ta; Nicolas Papadakis
>
> **备注:** International Conference on Image Processing 2017
>
> **摘要:** Regular decompositions are necessary for most superpixel-based object recognition or tracking applications. So far in the literature, the regularity or compactness of a superpixel shape is mainly measured by its circularity. In this work, we first demonstrate that such measure is not adapted for superpixel evaluation, since it does not directly express regularity but circular appearance. Then, we propose a new metric that considers several shape regularity aspects: convexity, balanced repartition, and contour smoothness. Finally, we demonstrate that our measure is robust to scale and noise and enables to more relevantly compare superpixel methods.
>
---
#### [replaced 046] Visible Yet Unreadable: A Systematic Blind Spot of Vision Language Models Across Writing Systems
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.06996v2](http://arxiv.org/pdf/2509.06996v2)**

> **作者:** Jie Zhang; Ting Xu; Gelei Deng; Runyi Hu; Han Qiu; Tianwei Zhang; Qing Guo; Ivor Tsang
>
> **摘要:** Writing is a universal cultural technology that reuses vision for symbolic communication. Humans display striking resilience: we readily recognize words even when characters are fragmented, fused, or partially occluded. This paper investigates whether advanced vision language models (VLMs) share this resilience. We construct two psychophysics inspired benchmarks across distinct writing systems, Chinese logographs and English alphabetic words, by splicing, recombining, and overlaying glyphs to yield ''visible but unreadable'' stimuli for models while remaining legible to humans. Despite strong performance on clean text, contemporary VLMs show a severe drop under these perturbations, frequently producing unrelated or incoherent outputs. The pattern suggests a structural limitation: models heavily leverage generic visual invariances but under rely on compositional priors needed for robust literacy. We release stimuli generation code, prompts, and evaluation protocols to facilitate transparent replication and follow up work. Our findings motivate architectures and training strategies that encode symbol segmentation, composition, and binding across scripts, and they delineate concrete challenges for deploying multimodal systems in education, accessibility, cultural heritage, and security.
>
---
#### [replaced 047] Reconstruction and Reenactment Separated Method for Realistic Gaussian Head
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.05582v2](http://arxiv.org/pdf/2509.05582v2)**

> **作者:** Zhiling Ye; Cong Zhou; Xiubao Zhang; Haifeng Shen; Weihong Deng; Quan Lu
>
> **摘要:** In this paper, we explore a reconstruction and reenactment separated framework for 3D Gaussians head, which requires only a single portrait image as input to generate controllable avatar. Specifically, we developed a large-scale one-shot gaussian head generator built upon WebSSL and employed a two-stage training approach that significantly enhances the capabilities of generalization and high-frequency texture reconstruction. During inference, an ultra-lightweight gaussian avatar driven by control signals enables high frame-rate rendering, achieving 90 FPS at a resolution of 512x512. We further demonstrate that the proposed framework follows the scaling law, whereby increasing the parameter scale of the reconstruction module leads to improved performance. Moreover, thanks to the separation design, driving efficiency remains unaffected. Finally, extensive quantitative and qualitative experiments validate that our approach outperforms current state-of-the-art methods.
>
---
#### [replaced 048] Enhancing Generalization in Vision-Language-Action Models by Preserving Pretrained Representations
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.11417v2](http://arxiv.org/pdf/2509.11417v2)**

> **作者:** Shresth Grover; Akshay Gopalkrishnan; Bo Ai; Henrik I. Christensen; Hao Su; Xuanlin Li
>
> **备注:** Project Page: https://gen-vla.github.io/
>
> **摘要:** Vision-language-action (VLA) models finetuned from vision-language models (VLMs) hold the promise of leveraging rich pretrained representations to build generalist robots across diverse tasks and environments. However, direct fine-tuning on robot data often disrupts these representations and limits generalization. We present a framework that better preserves pretrained features while adapting them for robot manipulation. Our approach introduces three components: (i) a dual-encoder design with one frozen vision encoder to retain pretrained features and another trainable for task adaptation, (ii) a string-based action tokenizer that casts continuous actions into character sequences aligned with the model's pretraining domain, and (iii) a co-training strategy that combines robot demonstrations with vision-language datasets emphasizing spatial reasoning and affordances. Evaluations in simulation and on real robots show that our method improves robustness to visual perturbations, generalization to novel instructions and environments, and overall task success compared to baselines.
>
---
#### [replaced 049] Uni-cot: Towards Unified Chain-of-Thought Reasoning Across Text and Vision
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.05606v2](http://arxiv.org/pdf/2508.05606v2)**

> **作者:** Luozheng Qin; Jia Gong; Yuqing Sun; Tianjiao Li; Mengping Yang; Xiaomeng Yang; Chao Qu; Zhiyu Tan; Hao Li
>
> **备注:** Project Page: https://sais-fuxi.github.io/projects/uni-cot/
>
> **摘要:** Chain-of-Thought (CoT) reasoning has been widely adopted to enhance Large Language Models (LLMs) by decomposing complex tasks into simpler, sequential subtasks. However, extending CoT to vision-language reasoning tasks remains challenging, as it often requires interpreting transitions of visual states to support reasoning. Existing methods often struggle with this due to limited capacity of modeling visual state transitions or incoherent visual trajectories caused by fragmented architectures. To overcome these limitations, we propose Uni-CoT, a Unified Chain-of-Thought framework that enables coherent and grounded multimodal reasoning within a single unified model. The key idea is to leverage a model capable of both image understanding and generation to reason over visual content and model evolving visual states. However, empowering a unified model to achieve that is non-trivial, given the high computational cost and the burden of training. To address this, Uni-CoT introduces a novel two-level reasoning paradigm: A Macro-Level CoT for high-level task planning and A Micro-Level CoT for subtask execution. This design significantly reduces the computational overhead. Furthermore, we introduce a structured training paradigm that combines interleaved image-text supervision for macro-level CoT with multi-task objectives for micro-level CoT. Together, these innovations allow Uni-CoT to perform scalable and coherent multi-modal reasoning. Furthermore, thanks to our design, all experiments can be efficiently completed using only 8 A100 GPUs with 80GB VRAM each. Experimental results on reasoning-driven image generation benchmark (WISE) and editing benchmarks (RISE and KRIS) indicates that Uni-CoT demonstrates SOTA performance and strong generalization, establishing Uni-CoT as a promising solution for multi-modal reasoning. Project Page and Code: https://sais-fuxi.github.io/projects/uni-cot/
>
---
