# 计算机视觉 cs.CV

- **最新发布 135 篇**

- **更新 66 篇**

## 最新发布

#### [new 001] Explaining the Unseen: Multimodal Vision-Language Reasoning for Situational Awareness in Underground Mining Disasters
- **分类: cs.CV**

- **简介: 该论文研究地下矿难场景下的视觉-语言推理任务，旨在提升复杂环境中的态势感知。针对视觉受限问题，提出MDSE框架，通过跨模态对齐、双路径编码和高效语言模型，生成准确场景描述，并构建首个矿难图像-文本数据集UMD进行验证。**

- **链接: [https://arxiv.org/pdf/2512.09092v1](https://arxiv.org/pdf/2512.09092v1)**

> **作者:** Mizanur Rahman Jewel; Mohamed Elmahallawy; Sanjay Madria; Samuel Frimpong
>
> **摘要:** Underground mining disasters produce pervasive darkness, dust, and collapses that obscure vision and make situational awareness difficult for humans and conventional systems. To address this, we propose MDSE, Multimodal Disaster Situation Explainer, a novel vision-language framework that automatically generates detailed textual explanations of post-disaster underground scenes. MDSE has three-fold innovations: (i) Context-Aware Cross-Attention for robust alignment of visual and textual features even under severe degradation; (ii) Segmentation-aware dual pathway visual encoding that fuses global and region-specific embeddings; and (iii) Resource-Efficient Transformer-Based Language Model for expressive caption generation with minimal compute cost. To support this task, we present the Underground Mine Disaster (UMD) dataset--the first image-caption corpus of real underground disaster scenes--enabling rigorous training and evaluation. Extensive experiments on UMD and related benchmarks show that MDSE substantially outperforms state-of-the-art captioning models, producing more accurate and contextually relevant descriptions that capture crucial details in obscured environments, improving situational awareness for underground emergency response. The code is at https://github.com/mizanJewel/Multimodal-Disaster-Situation-Explainer.
>
---
#### [new 002] Wasserstein-Aligned Hyperbolic Multi-View Clustering
- **分类: cs.CV**

- **简介: 该论文研究多视图聚类，旨在解决现有方法忽略全局语义一致性的问题。提出Wasserstein对齐的双曲框架，通过双曲编码器和基于Wasserstein距离的全局对齐损失，提升跨视图语义一致性，实现SOTA性能。**

- **链接: [https://arxiv.org/pdf/2512.09402v1](https://arxiv.org/pdf/2512.09402v1)**

> **作者:** Rui Wang; Yuting Jiang; Xiaoqing Luo; Xiao-Jun Wu; Nicu Sebe; Ziheng Chen
>
> **备注:** 14 pages
>
> **摘要:** Multi-view clustering (MVC) aims to uncover the latent structure of multi-view data by learning view-common and view-specific information. Although recent studies have explored hyperbolic representations for better tackling the representation gap between different views, they focus primarily on instance-level alignment and neglect global semantic consistency, rendering them vulnerable to view-specific information (\textit{e.g.}, noise and cross-view discrepancies). To this end, this paper proposes a novel Wasserstein-Aligned Hyperbolic (WAH) framework for multi-view clustering. Specifically, our method exploits a view-specific hyperbolic encoder for each view to embed features into the Lorentz manifold for hierarchical semantic modeling. Whereafter, a global semantic loss based on the hyperbolic sliced-Wasserstein distance is introduced to align manifold distributions across views. This is followed by soft cluster assignments to encourage cross-view semantic consistency. Extensive experiments on multiple benchmarking datasets show that our method can achieve SOTA clustering performance.
>
---
#### [new 003] Traffic Scene Small Target Detection Method Based on YOLOv8n-SPTS Model for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决自动驾驶中小目标检测难的问题。通过改进YOLOv8n模型，引入SPD-Conv、SPPFCSPC模块和TSFP结构，提升小目标的特征提取与融合能力，在VisDrone数据集上取得优异性能。**

- **链接: [https://arxiv.org/pdf/2512.09296v1](https://arxiv.org/pdf/2512.09296v1)**

> **作者:** Songhan Wu
>
> **备注:** 6 pages, 7 figures, 1 table. Accepted to The 2025 IEEE 3rd International Conference on Electrical, Automation and Computer Engineering (ICEACE), 2025. Code available at https://github.com/SonghanWu/yolov8n-SPTS
>
> **摘要:** This paper focuses on the key issue in autonomous driving: small target recognition in dynamic perception. Existing algorithms suffer from poor detection performance due to missing small target information, scale imbalance, and occlusion. We propose an improved YOLOv8n-SPTS model, which enhances the detection accuracy of small traffic targets through three key innovations: First, optimizing the feature extraction module. In the Backbone Bottleneck structure of YOLOv8n, 4 traditional convolution modules are replaced with Space-to-Depth Convolution (SPD-Conv) modules. This module retains fine-grained information through space-to-depth conversion, reduces information loss, and enhances the ability to capture features of low-resolution small targets. Second, enhancing feature fusion capability. The Spatial Pyramid Pooling - Fast Cross Stage Partial Connection (SPPFCSPC) module is introduced to replace the original SPPF module, integrating the multi-scale feature extraction from Spatial Pyramid Pooling (SPP) and the feature fusion mechanism of Cross Stage Partial Connection (CSP), thereby improving the model's contextual understanding of complex scenes and multi-scale feature expression ability. Third, designing a dedicated detection structure for small targets. A Triple-Stage Feature Pyramid (TSFP) structure is proposed, which adds a 160*160 small target detection head to the original detection heads to fully utilize high-resolution features in shallow layers; meanwhile, redundant large target detection heads are removed to balance computational efficiency. Comparative experiments on the VisDrone2019-DET dataset show that YOLOv8n-SPTS model ranks first in precision (61.9%), recall (48.3%), mAP@0.5 (52.6%), and mAP@0.5:0.95 (32.6%). Visualization results verify that the miss rate of small targets such as pedestrians and bicycles in occluded and dense scenes is significantly reduced.
>
---
#### [new 004] Perception-Inspired Color Space Design for Photo White Balance Editing
- **分类: cs.CV**

- **简介: 该论文研究图像白平衡编辑任务，旨在解决sRGB空间因通道耦合和非线性变换导致的色彩校正局限。提出感知启发的可学习HSI颜色空间（LHSI）及配套Mamba网络，实现更优的后ISP白平衡修正。**

- **链接: [https://arxiv.org/pdf/2512.09383v1](https://arxiv.org/pdf/2512.09383v1)**

> **作者:** Yang Cheng; Ziteng Cui; Lin Gu; Shenghan Su; Zenghui Zhang
>
> **备注:** Accepted to WACV 2026
>
> **摘要:** White balance (WB) is a key step in the image signal processor (ISP) pipeline that mitigates color casts caused by varying illumination and restores the scene's true colors. Currently, sRGB-based WB editing for post-ISP WB correction is widely used to address color constancy failures in the ISP pipeline when the original camera RAW is unavailable. However, additive color models (e.g., sRGB) are inherently limited by fixed nonlinear transformations and entangled color channels, which often impede their generalization to complex lighting conditions. To address these challenges, we propose a novel framework for WB correction that leverages a perception-inspired Learnable HSI (LHSI) color space. Built upon a cylindrical color model that naturally separates luminance from chromatic components, our framework further introduces dedicated parameters to enhance this disentanglement and learnable mapping to adaptively refine the flexibility. Moreover, a new Mamba-based network is introduced, which is tailored to the characteristics of the proposed LHSI color space. Experimental results on benchmark datasets demonstrate the superiority of our method, highlighting the potential of perception-inspired color space design in computational photography. The source code is available at https://github.com/YangCheng58/WB_Color_Space.
>
---
#### [new 005] Composing Concepts from Images and Videos via Concept-prompt Binding
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文研究视觉概念组合任务，旨在从图像和视频中准确提取并灵活组合复杂概念。提出Bind & Compose方法，通过概念-提示词绑定、分层绑定结构、多样吸收机制和时序解耦策略，提升概念一致性、提示忠实性和运动质量。**

- **链接: [https://arxiv.org/pdf/2512.09824v1](https://arxiv.org/pdf/2512.09824v1)**

> **作者:** Xianghao Kong; Zeyu Zhang; Yuwei Guo; Zhuoran Zhao; Songchun Zhang; Anyi Rao
>
> **备注:** Project page: https://refkxh.github.io/BiCo_Webpage/
>
> **摘要:** Visual concept composition, which aims to integrate different elements from images and videos into a single, coherent visual output, still falls short in accurately extracting complex concepts from visual inputs and flexibly combining concepts from both images and videos. We introduce Bind & Compose, a one-shot method that enables flexible visual concept composition by binding visual concepts with corresponding prompt tokens and composing the target prompt with bound tokens from various sources. It adopts a hierarchical binder structure for cross-attention conditioning in Diffusion Transformers to encode visual concepts into corresponding prompt tokens for accurate decomposition of complex visual concepts. To improve concept-token binding accuracy, we design a Diversify-and-Absorb Mechanism that uses an extra absorbent token to eliminate the impact of concept-irrelevant details when training with diversified prompts. To enhance the compatibility between image and video concepts, we present a Temporal Disentanglement Strategy that decouples the training process of video concepts into two stages with a dual-branch binder structure for temporal modeling. Evaluations demonstrate that our method achieves superior concept consistency, prompt fidelity, and motion quality over existing approaches, opening up new possibilities for visual creativity.
>
---
#### [new 006] Defect-aware Hybrid Prompt Optimization via Progressive Tuning for Zero-Shot Multi-type Anomaly Detection and Segmentation
- **分类: cs.CV**

- **简介: 该论文研究零样本多类型异常检测与分割，旨在解决现有视觉语言模型忽略细粒度缺陷语义的问题。提出DAPO方法，通过渐进式学习混合提示，对齐图像与文本语义，在分布偏移下提升检测与定位性能。**

- **链接: [https://arxiv.org/pdf/2512.09446v1](https://arxiv.org/pdf/2512.09446v1)**

> **作者:** Nadeem Nazer; Hongkuan Zhou; Lavdim Halilaj; Ylli Sadikaj; Steffen Staab
>
> **摘要:** Recent vision language models (VLMs) like CLIP have demonstrated impressive anomaly detection performance under significant distribution shift by utilizing high-level semantic information through text prompts. However, these models often neglect fine-grained details, such as which kind of anomalies, like "hole", "cut", "scratch" that could provide more specific insight into the nature of anomalies. We argue that recognizing fine-grained anomaly types 1) enriches the representation of "abnormal" with structured semantics, narrowing the gap between coarse anomaly signals and fine-grained defect categories; 2) enables manufacturers to understand the root causes of the anomaly and implement more targeted and appropriate corrective measures quickly. While incorporating such detailed semantic information is crucial, designing handcrafted prompts for each defect type is both time-consuming and susceptible to human bias. For this reason, we introduce DAPO, a novel approach for Defect-aware Prompt Optimization based on progressive tuning for the zero-shot multi-type and binary anomaly detection and segmentation under distribution shifts. Our approach aligns anomaly-relevant image features with their corresponding text semantics by learning hybrid defect-aware prompts with both fixed textual anchors and learnable token embeddings. We conducted experiments on public benchmarks (MPDD, VisA, MVTec-AD, MAD, and Real-IAD) and an internal dataset. The results suggest that compared to the baseline models, DAPO achieves a 3.7% average improvement in AUROC and average precision metrics at the image level under distribution shift, and a 6.5% average improvement in localizing novel anomaly types under zero-shot settings.
>
---
#### [new 007] AgentComp: From Agentic Reasoning to Compositional Mastery in Text-to-Image Models
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决模型在组合性（如对象关系、属性绑定）上的不足。作者提出AgentComp框架，利用具备工具调用能力的大语言模型构建组合数据集，并通过代理偏好优化提升模型对细微差异的分辨与生成能力。**

- **链接: [https://arxiv.org/pdf/2512.09081v1](https://arxiv.org/pdf/2512.09081v1)**

> **作者:** Arman Zarei; Jiacheng Pan; Matthew Gwilliam; Soheil Feizi; Zhenheng Yang
>
> **摘要:** Text-to-image generative models have achieved remarkable visual quality but still struggle with compositionality$-$accurately capturing object relationships, attribute bindings, and fine-grained details in prompts. A key limitation is that models are not explicitly trained to differentiate between compositionally similar prompts and images, resulting in outputs that are close to the intended description yet deviate in fine-grained details. To address this, we propose AgentComp, a framework that explicitly trains models to better differentiate such compositional variations and enhance their reasoning ability. AgentComp leverages the reasoning and tool-use capabilities of large language models equipped with image generation, editing, and VQA tools to autonomously construct compositional datasets. Using these datasets, we apply an agentic preference optimization method to fine-tune text-to-image models, enabling them to better distinguish between compositionally similar samples and resulting in overall stronger compositional generation ability. AgentComp achieves state-of-the-art results on compositionality benchmarks such as T2I-CompBench, without compromising image quality$-$a common drawback in prior approaches$-$and even generalizes to other capabilities not explicitly trained for, such as text rendering.
>
---
#### [new 008] MODA: The First Challenging Benchmark for Multispectral Object Detection in Aerial Images
- **分类: cs.CV**

- **简介: 该论文针对空中多光谱图像中的小目标检测难题，构建了首个大规模数据集MODA，并提出OSSDet框架，融合光谱与空间信息，通过对象感知机制提升检测性能。**

- **链接: [https://arxiv.org/pdf/2512.09489v1](https://arxiv.org/pdf/2512.09489v1)**

> **作者:** Shuaihao Han; Tingfa Xu; Peifu Liu; Jianan Li
>
> **备注:** 8 pages, 9 figures
>
> **摘要:** Aerial object detection faces significant challenges in real-world scenarios, such as small objects and extensive background interference, which limit the performance of RGB-based detectors with insufficient discriminative information. Multispectral images (MSIs) capture additional spectral cues across multiple bands, offering a promising alternative. However, the lack of training data has been the primary bottleneck to exploiting the potential of MSIs. To address this gap, we introduce the first large-scale dataset for Multispectral Object Detection in Aerial images (MODA), which comprises 14,041 MSIs and 330,191 annotations across diverse, challenging scenarios, providing a comprehensive data foundation for this field. Furthermore, to overcome challenges inherent to aerial object detection using MSIs, we propose OSSDet, a framework that integrates spectral and spatial information with object-aware cues. OSSDet employs a cascaded spectral-spatial modulation structure to optimize target perception, aggregates spectrally related features by exploiting spectral similarities to reinforce intra-object correlations, and suppresses irrelevant background via object-aware masking. Moreover, cross-spectral attention further refines object-related representations under explicit object-aware guidance. Extensive experiments demonstrate that OSSDet outperforms existing methods with comparable parameters and efficiency.
>
---
#### [new 009] Temporal-Spatial Tubelet Embedding for Cloud-Robust MSI Reconstruction using MSI-SAR Fusion: A Multi-Head Self-Attention Video Vision Transformer Approach
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对云遮挡下多光谱影像（MSI）作物监测难题，提出一种基于视频ViT的时空管状嵌入方法，通过局部时序建模与SAR-MSI融合，提升重建精度，有效改善云污染区域的光谱信息恢复。**

- **链接: [https://arxiv.org/pdf/2512.09471v1](https://arxiv.org/pdf/2512.09471v1)**

> **作者:** Yiqun Wang; Lujun Li; Meiru Yue; Radu State
>
> **摘要:** Cloud cover in multispectral imagery (MSI) significantly hinders early-season crop mapping by corrupting spectral information. Existing Vision Transformer(ViT)-based time-series reconstruction methods, like SMTS-ViT, often employ coarse temporal embeddings that aggregate entire sequences, causing substantial information loss and reducing reconstruction accuracy. To address these limitations, a Video Vision Transformer (ViViT)-based framework with temporal-spatial fusion embedding for MSI reconstruction in cloud-covered regions is proposed in this study. Non-overlapping tubelets are extracted via 3D convolution with constrained temporal span $(t=2)$, ensuring local temporal coherence while reducing cross-day information degradation. Both MSI-only and SAR-MSI fusion scenarios are considered during the experiments. Comprehensive experiments on 2020 Traill County data demonstrate notable performance improvements: MTS-ViViT achieves a 2.23\% reduction in MSE compared to the MTS-ViT baseline, while SMTS-ViViT achieves a 10.33\% improvement with SAR integration over the SMTS-ViT baseline. The proposed framework effectively enhances spectral reconstruction quality for robust agricultural monitoring.
>
---
#### [new 010] OxEnsemble: Fair Ensembles for Low-Data Classification
- **分类: cs.CV; cs.CY; cs.LG**

- **简介: 该论文属于公平分类任务，旨在解决低数据量且群体样本不均衡下的公平性问题。提出OxEnsemble方法，通过集成多个满足公平约束的模型，在少数据下实现高效、稳定的公平分类，兼顾数据与计算效率，并取得更优的公平-准确权衡。**

- **链接: [https://arxiv.org/pdf/2512.09665v1](https://arxiv.org/pdf/2512.09665v1)**

> **作者:** Jonathan Rystrøm; Zihao Fu; Chris Russell
>
> **摘要:** We address the problem of fair classification in settings where data is scarce and unbalanced across demographic groups. Such low-data regimes are common in domains like medical imaging, where false negatives can have fatal consequences. We propose a novel approach \emph{OxEnsemble} for efficiently training ensembles and enforcing fairness in these low-data regimes. Unlike other approaches, we aggregate predictions across ensemble members, each trained to satisfy fairness constraints. By construction, \emph{OxEnsemble} is both data-efficient, carefully reusing held-out data to enforce fairness reliably, and compute-efficient, requiring little more compute than used to fine-tune or evaluate an existing model. We validate this approach with new theoretical guarantees. Experimentally, our approach yields more consistent outcomes and stronger fairness-accuracy trade-offs than existing methods across multiple challenging medical imaging classification datasets.
>
---
#### [new 011] Efficient Feature Compression for Machines with Global Statistics Preservation
- **分类: cs.CV**

- **简介: 该论文属AI模型压缩任务，旨在解决特征数据传输中的高效压缩问题。提出基于Z-score归一化的方法，在保留全局统计信息的同时减少比特开销，提升解码效果，并兼容现有FCM标准，显著降低码率且不损任务精度。**

- **链接: [https://arxiv.org/pdf/2512.09235v1](https://arxiv.org/pdf/2512.09235v1)**

> **作者:** Md Eimran Hossain Eimon; Hyomin Choi; Fabien Racapé; Mateen Ulhaq; Velibor Adzic; Hari Kalva; Borko Furht
>
> **摘要:** The split-inference paradigm divides an artificial intelligence (AI) model into two parts. This necessitates the transfer of intermediate feature data between the two halves. Here, effective compression of the feature data becomes vital. In this paper, we employ Z-score normalization to efficiently recover the compressed feature data at the decoder side. To examine the efficacy of our method, the proposed method is integrated into the latest Feature Coding for Machines (FCM) codec standard under development by the Moving Picture Experts Group (MPEG). Our method supersedes the existing scaling method used by the current standard under development. It both reduces the overhead bits and improves the end-task accuracy. To further reduce the overhead in certain circumstances, we also propose a simplified method. Experiments show that using our proposed method shows 17.09% reduction in bitrate on average across different tasks and up to 65.69% for object tracking without sacrificing the task accuracy.
>
---
#### [new 012] NordFKB: a fine-grained benchmark dataset for geospatial AI in Norway
- **分类: cs.CV**

- **简介: 该论文提出NordFKB，一个用于挪威地理空间AI的细粒度基准数据集。属于遥感图像理解任务，旨在解决高精度地物识别与分割问题。作者基于权威地理数据库构建数据集，提供高分辨率影像与36类精细标注，并发布配套评测工具，推动地图绘制与空间规划中的AI研究。**

- **链接: [https://arxiv.org/pdf/2512.09913v1](https://arxiv.org/pdf/2512.09913v1)**

> **作者:** Sander Riisøen Jyhne; Aditya Gupta; Ben Worsley; Marianne Andersen; Ivar Oveland; Alexander Salveson Nossum
>
> **备注:** 8 pages, 2 figures, 2 tables
>
> **摘要:** We present NordFKB, a fine-grained benchmark dataset for geospatial AI in Norway, derived from the authoritative, highly accurate, national Felles KartdataBase (FKB). The dataset contains high-resolution orthophotos paired with detailed annotations for 36 semantic classes, including both per-class binary segmentation masks in GeoTIFF format and COCO-style bounding box annotations. Data is collected from seven geographically diverse areas, ensuring variation in climate, topography, and urbanization. Only tiles containing at least one annotated object are included, and training/validation splits are created through random sampling across areas to ensure representative class and context distributions. Human expert review and quality control ensures high annotation accuracy. Alongside the dataset, we release a benchmarking repository with standardized evaluation protocols and tools for semantic segmentation and object detection, enabling reproducible and comparable research. NordFKB provides a robust foundation for advancing AI methods in mapping, land administration, and spatial planning, and paves the way for future expansions in coverage, temporal scope, and data modalities.
>
---
#### [new 013] OmniPSD: Layered PSD Generation with Diffusion Transformer
- **分类: cs.CV**

- **简介: 该论文提出OmniPSD，解决生成和分解带透明通道的分层PSD文件难题。基于扩散Transformer，实现文本到PSD生成与图像到PSD分解，通过空间注意力和迭代上下文编辑，保持层次结构与透明度，支持高保真、可编辑的分层图像合成。**

- **链接: [https://arxiv.org/pdf/2512.09247v1](https://arxiv.org/pdf/2512.09247v1)**

> **作者:** Cheng Liu; Yiren Song; Haofan Wang; Mike Zheng Shou
>
> **摘要:** Recent advances in diffusion models have greatly improved image generation and editing, yet generating or reconstructing layered PSD files with transparent alpha channels remains highly challenging. We propose OmniPSD, a unified diffusion framework built upon the Flux ecosystem that enables both text-to-PSD generation and image-to-PSD decomposition through in-context learning. For text-to-PSD generation, OmniPSD arranges multiple target layers spatially into a single canvas and learns their compositional relationships through spatial attention, producing semantically coherent and hierarchically structured layers. For image-to-PSD decomposition, it performs iterative in-context editing, progressively extracting and erasing textual and foreground components to reconstruct editable PSD layers from a single flattened image. An RGBA-VAE is employed as an auxiliary representation module to preserve transparency without affecting structure learning. Extensive experiments on our new RGBA-layered dataset demonstrate that OmniPSD achieves high-fidelity generation, structural consistency, and transparency awareness, offering a new paradigm for layered design generation and decomposition with diffusion transformers.
>
---
#### [new 014] Splatent: Splatting Diffusion Latents for Novel View Synthesis
- **分类: cs.CV**

- **简介: 该论文聚焦于基于VAE隐空间的辐射场新视角合成任务，旨在解决多视图不一致导致的细节模糊问题。提出Splatent框架，通过在2D输入视图上引入多视图注意力恢复细节，提升3D重建质量。**

- **链接: [https://arxiv.org/pdf/2512.09923v1](https://arxiv.org/pdf/2512.09923v1)**

> **作者:** Or Hirschorn; Omer Sela; Inbar Huberman-Spiegelglas; Netalee Efrat; Eli Alshan; Ianir Ideses; Frederic Devernay; Yochai Zvik; Lior Fritz
>
> **摘要:** Radiance field representations have recently been explored in the latent space of VAEs that are commonly used by diffusion models. This direction offers efficient rendering and seamless integration with diffusion-based pipelines. However, these methods face a fundamental limitation: The VAE latent space lacks multi-view consistency, leading to blurred textures and missing details during 3D reconstruction. Existing approaches attempt to address this by fine-tuning the VAE, at the cost of reconstruction quality, or by relying on pre-trained diffusion models to recover fine-grained details, at the risk of some hallucinations. We present Splatent, a diffusion-based enhancement framework designed to operate on top of 3D Gaussian Splatting (3DGS) in the latent space of VAEs. Our key insight departs from the conventional 3D-centric view: rather than reconstructing fine-grained details in 3D space, we recover them in 2D from input views through multi-view attention mechanisms. This approach preserves the reconstruction quality of pretrained VAEs while achieving faithful detail recovery. Evaluated across multiple benchmarks, Splatent establishes a new state-of-the-art for VAE latent radiance field reconstruction. We further demonstrate that integrating our method with existing feed-forward frameworks, consistently improves detail preservation, opening new possibilities for high-quality sparse-view 3D reconstruction.
>
---
#### [new 015] Gradient-Guided Learning Network for Infrared Small Target Detection
- **分类: cs.CV**

- **简介: 该论文研究红外小目标检测，旨在解决目标边缘定位不准和易被背景淹没的问题。提出梯度引导学习网络（GGL-Net），引入梯度幅值图像，设计双分支网络与双向融合模块，提升检测精度，在多个数据集上达到先进水平。**

- **链接: [https://arxiv.org/pdf/2512.09497v1](https://arxiv.org/pdf/2512.09497v1)**

> **作者:** Jinmiao Zhao; Chuang Yu; Zelin Shi; Yunpeng Liu; Yingdi Zhang
>
> **备注:** Accepted by GRSL 2023
>
> **摘要:** Recently, infrared small target detection has attracted extensive attention. However, due to the small size and the lack of intrinsic features of infrared small targets, the existing methods generally have the problem of inaccurate edge positioning and the target is easily submerged by the background. Therefore, we propose an innovative gradient-guided learning network (GGL-Net). Specifically, we are the first to explore the introduction of gradient magnitude images into the deep learning-based infrared small target detection method, which is conducive to emphasizing the edge details and alleviating the problem of inaccurate edge positioning of small targets. On this basis, we propose a novel dual-branch feature extraction network that utilizes the proposed gradient supplementary module (GSM) to encode raw gradient information into deeper network layers and embeds attention mechanisms reasonably to enhance feature extraction ability. In addition, we construct a two-way guidance fusion module (TGFM), which fully considers the characteristics of feature maps at different levels. It can facilitate the effective fusion of multi-scale feature maps and extract richer semantic information and detailed information through reasonable two-way guidance. Extensive experiments prove that GGL-Net has achieves state-of-the-art results on the public real NUAA-SIRST dataset and the public synthetic NUDT-SIRST dataset. Our code has been integrated into https://github.com/YuChuang1205/MSDA-Net
>
---
#### [new 016] An Automated Tip-and-Cue Framework for Optimized Satellite Tasking and Visual Intelligence
- **分类: cs.CV; eess.SY**

- **简介: 该论文提出一种自动化“提示-引导”框架，用于优化卫星成像任务规划与视觉智能分析。针对地球观测中任务响应滞后问题，利用多源数据生成目标提示，结合传感器约束与效用函数实现多星协同调度，并通过AI模型处理影像，生成结构化报告，验证于船舶跟踪场景。**

- **链接: [https://arxiv.org/pdf/2512.09670v1](https://arxiv.org/pdf/2512.09670v1)**

> **作者:** Gil Weissman; Amir Ivry; Israel Cohen
>
> **备注:** Under review at IEEE Transactions on Geoscience and Remote Sensing (TGRS). 13 pages, 8 figures
>
> **摘要:** The proliferation of satellite constellations, coupled with reduced tasking latency and diverse sensor capabilities, has expanded the opportunities for automated Earth observation. This paper introduces a fully automated Tip-and-Cue framework designed for satellite imaging tasking and scheduling. In this context, tips are generated from external data sources or analyses of prior satellite imagery, identifying spatiotemporal targets and prioritizing them for downstream planning. Corresponding cues are the imaging tasks formulated in response, which incorporate sensor constraints, timing requirements, and utility functions. The system autonomously generates candidate tasks, optimizes their scheduling across multiple satellites using continuous utility functions that reflect the expected value of each observation, and processes the resulting imagery using artificial-intelligence-based models, including object detectors and vision-language models. Structured visual reports are generated to support both interpretability and the identification of new insights for downstream tasking. The efficacy of the framework is demonstrated through a maritime vessel tracking scenario, utilizing Automatic Identification System (AIS) data for trajectory prediction, targeted observations, and the generation of actionable outputs. Maritime vessel tracking is a widely researched application, often used to benchmark novel approaches to satellite tasking, forecasting, and analysis. The system is extensible to broader applications such as smart-city monitoring and disaster response, where timely tasking and automated analysis are critical.
>
---
#### [new 017] VisualActBench: Can VLMs See and Act like a Human?
- **分类: cs.CV**

- **简介: 该论文提出视觉动作推理任务，构建VisualActBench基准，评估VLMs在无文本提示下基于视觉主动决策的能力。通过1,074个视频和人类标注动作，衡量模型对复杂情境的理解与人类对齐的行动优先级判断，揭示当前模型在前瞻性决策上的不足。**

- **链接: [https://arxiv.org/pdf/2512.09907v1](https://arxiv.org/pdf/2512.09907v1)**

> **作者:** Daoan Zhang; Pai Liu; Xiaofei Zhou; Yuan Ge; Guangchen Lan; Jing Bi; Christopher Brinton; Ehsan Hoque; Jiebo Luo
>
> **摘要:** Vision-Language Models (VLMs) have achieved impressive progress in perceiving and describing visual environments. However, their ability to proactively reason and act based solely on visual inputs, without explicit textual prompts, remains underexplored. We introduce a new task, Visual Action Reasoning, and propose VisualActBench, a large-scale benchmark comprising 1,074 videos and 3,733 human-annotated actions across four real-world scenarios. Each action is labeled with an Action Prioritization Level (APL) and a proactive-reactive type to assess models' human-aligned reasoning and value sensitivity. We evaluate 29 VLMs on VisualActBench and find that while frontier models like GPT4o demonstrate relatively strong performance, a significant gap remains compared to human-level reasoning, particularly in generating proactive, high-priority actions. Our results highlight limitations in current VLMs' ability to interpret complex context, anticipate outcomes, and align with human decision-making frameworks. VisualActBench establishes a comprehensive foundation for assessing and improving the real-world readiness of proactive, vision-centric AI agents.
>
---
#### [new 018] Transformer-Driven Multimodal Fusion for Explainable Suspiciousness Estimation in Visual Surveillance
- **分类: cs.CV; cs.CR**

- **简介: 该论文研究视觉监控中的可疑行为识别，旨在实现可解释的实时风险评估。作者构建了大规模数据集USE50k，并提出轻量级多模态融合框架DeepUSEvision，结合改进YOLO、双DCNN与Transformer，提升检测精度与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.09311v1](https://arxiv.org/pdf/2512.09311v1)**

> **作者:** Kuldeep Singh Yadav; Lalan Kumar
>
> **备注:** 12 pages, 10 figures, IEEE Transaction on Image Processing
>
> **摘要:** Suspiciousness estimation is critical for proactive threat detection and ensuring public safety in complex environments. This work introduces a large-scale annotated dataset, USE50k, along with a computationally efficient vision-based framework for real-time suspiciousness analysis. The USE50k dataset contains 65,500 images captured from diverse and uncontrolled environments, such as airports, railway stations, restaurants, parks, and other public areas, covering a broad spectrum of cues including weapons, fire, crowd density, abnormal facial expressions, and unusual body postures. Building on this dataset, we present DeepUSEvision, a lightweight and modular system integrating three key components, i.e., a Suspicious Object Detector based on an enhanced YOLOv12 architecture, dual Deep Convolutional Neural Networks (DCNN-I and DCNN-II) for facial expression and body-language recognition using image and landmark features, and a transformer-based Discriminator Network that adaptively fuses multimodal outputs to yield an interpretable suspiciousness score. Extensive experiments confirm the superior accuracy, robustness, and interpretability of the proposed framework compared to state-of-the-art approaches. Collectively, the USE50k dataset and the DeepUSEvision framework establish a strong and scalable foundation for intelligent surveillance and real-time risk assessment in safety-critical applications.
>
---
#### [new 019] SuperF: Neural Implicit Fields for Multi-Image Super-Resolution
- **分类: cs.CV**

- **简介: 该论文研究多图像超分辨率（MISR），旨在通过多幅低分辨率图像恢复高分辨率图像。提出SuperF方法，利用神经隐式场与测试时优化，联合优化图像对齐与重建，无需高分辨率训练数据，提升重建真实性。**

- **链接: [https://arxiv.org/pdf/2512.09115v1](https://arxiv.org/pdf/2512.09115v1)**

> **作者:** Sander Riisøen Jyhne; Christian Igel; Morten Goodwin; Per-Arne Andersen; Serge Belongie; Nico Lang
>
> **备注:** 23 pages, 13 figures, 8 tables
>
> **摘要:** High-resolution imagery is often hindered by limitations in sensor technology, atmospheric conditions, and costs. Such challenges occur in satellite remote sensing, but also with handheld cameras, such as our smartphones. Hence, super-resolution aims to enhance the image resolution algorithmically. Since single-image super-resolution requires solving an inverse problem, such methods must exploit strong priors, e.g. learned from high-resolution training data, or be constrained by auxiliary data, e.g. by a high-resolution guide from another modality. While qualitatively pleasing, such approaches often lead to "hallucinated" structures that do not match reality. In contrast, multi-image super-resolution (MISR) aims to improve the (optical) resolution by constraining the super-resolution process with multiple views taken with sub-pixel shifts. Here, we propose SuperF, a test-time optimization approach for MISR that leverages coordinate-based neural networks, also called neural fields. Their ability to represent continuous signals with an implicit neural representation (INR) makes them an ideal fit for the MISR task. The key characteristic of our approach is to share an INR for multiple shifted low-resolution frames and to jointly optimize the frame alignment with the INR. Our approach advances related INR baselines, adopted from burst fusion for layer separation, by directly parameterizing the sub-pixel alignment as optimizable affine transformation parameters and by optimizing via a super-sampled coordinate grid that corresponds to the output resolution. Our experiments yield compelling results on simulated bursts of satellite imagery and ground-level images from handheld cameras, with upsampling factors of up to 8. A key advantage of SuperF is that this approach does not rely on any high-resolution training data.
>
---
#### [new 020] An Approach for Detection of Entities in Dynamic Media Contents
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在通过深度学习技术检测视频中特定人物。针对复杂场景下人物识别难题，提出一种基于监督学习的实体检测方法，利用目标简单特征高效定位个体，可应用于公共安全领域的人员搜寻。**

- **链接: [https://arxiv.org/pdf/2512.09011v1](https://arxiv.org/pdf/2512.09011v1)**

> **作者:** Nzakiese Mbongo; Ngombo Armando
>
> **备注:** 12 pages, 8 figures
>
> **摘要:** The notion of learning underlies almost every evolution of Intelligent Agents. In this paper, we present an approach for searching and detecting a given entity in a video sequence. Specifically, we study how the deep learning technique by artificial neuralnetworks allows us to detect a character in a video sequence. The technique of detecting a character in a video is a complex field of study, considering the multitude of objects present in the data under analysis. From the results obtained, we highlight the following, compared to state of the art: In our approach, within the field of Computer Vision, the structuring of supervised learning algorithms allowed us to achieve several successes from simple characteristics of the target character. Our results demonstrate that is new approach allows us to locate, in an efficient way, wanted individuals from a private or public image base. For the case of Angola, the classifier we propose opens the possibility of reinforcing the national security system based on the database of target individuals (disappeared, criminals, etc.) and the video sequences of the Integrated Public Security Centre (CISP).
>
---
#### [new 021] Log NeRF: Comparing Spaces for Learning Radiance Fields
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究NeRF在不同颜色空间下的表现，提出在log RGB空间中学习辐射场可提升渲染质量与鲁棒性。通过线性化真实视频数据，比较多种颜色空间，验证了log空间尤其在低光下更优，且具通用性。**

- **链接: [https://arxiv.org/pdf/2512.09375v1](https://arxiv.org/pdf/2512.09375v1)**

> **作者:** Sihe Chen; Luv Verma; Bruce A. Maxwell
>
> **备注:** The 36th British Machine Vision Conference
>
> **摘要:** Neural Radiance Fields (NeRF) have achieved remarkable results in novel view synthesis, typically using sRGB images for supervision. However, little attention has been paid to the color space in which the network is learning the radiance field representation. Inspired by the BiIlluminant Dichromatic Reflection (BIDR) model, which suggests that a logarithmic transformation simplifies the separation of illumination and reflectance, we hypothesize that log RGB space enables NeRF to learn a more compact and effective representation of scene appearance. To test this, we captured approximately 30 videos using a GoPro camera, ensuring linear data recovery through inverse encoding. We trained NeRF models under various color space interpretations linear, sRGB, GPLog, and log RGB by converting each network output to a common color space before rendering and loss computation, enforcing representation learning in different color spaces. Quantitative and qualitative evaluations demonstrate that using a log RGB color space consistently improves rendering quality, exhibits greater robustness across scenes, and performs particularly well in low light conditions while using the same bit-depth input images. Further analysis across different network sizes and NeRF variants confirms the generalization and stability of the log space advantage.
>
---
#### [new 022] Benchmarking SAM2-based Trackers on FMOX
- **分类: cs.CV**

- **简介: 该论文属于目标跟踪任务，旨在评估基于SAM2的追踪器在快速移动物体上的性能。通过在FMOX数据集上 benchmark 多个先进追踪器，揭示其在挑战性场景中的表现与局限。**

- **链接: [https://arxiv.org/pdf/2512.09633v1](https://arxiv.org/pdf/2512.09633v1)**

> **作者:** Senem Aktas; Charles Markham; John McDonald; Rozenn Dahyot
>
> **摘要:** Several object tracking pipelines extending Segment Anything Model 2 (SAM2) have been proposed in the past year, where the approach is to follow and segment the object from a single exemplar template provided by the user on a initialization frame. We propose to benchmark these high performing trackers (SAM2, EfficientTAM, DAM4SAM and SAMURAI) on datasets containing fast moving objects (FMO) specifically designed to be challenging for tracking approaches. The goal is to understand better current limitations in state-of-the-art trackers by providing more detailed insights on the behavior of these trackers. We show that overall the trackers DAM4SAM and SAMURAI perform well on more challenging sequences.
>
---
#### [new 023] Privacy-Preserving Computer Vision for Industry: Three Case Studies in Human-Centric Manufacturing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属隐私保护的计算机视觉任务，旨在解决工业中AI监控与工人隐私的冲突。作者基于前期框架，在三个真实制造场景中验证其有效性，通过选择性遮蔽敏感信息，在保障任务性能的同时降低隐私风险，推动可信赖的人本AI在工业落地。**

- **链接: [https://arxiv.org/pdf/2512.09463v1](https://arxiv.org/pdf/2512.09463v1)**

> **作者:** Sander De Coninck; Emilio Gamba; Bart Van Doninck; Abdellatif Bey-Temsamani; Sam Leroux; Pieter Simoens
>
> **备注:** Accepted to the AAAI26 HCM workshop
>
> **摘要:** The adoption of AI-powered computer vision in industry is often constrained by the need to balance operational utility with worker privacy. Building on our previously proposed privacy-preserving framework, this paper presents its first comprehensive validation on real-world data collected directly by industrial partners in active production environments. We evaluate the framework across three representative use cases: woodworking production monitoring, human-aware AGV navigation, and multi-camera ergonomic risk assessment. The approach employs learned visual transformations that obscure sensitive or task-irrelevant information while retaining features essential for task performance. Through both quantitative evaluation of the privacy-utility trade-off and qualitative feedback from industrial partners, we assess the framework's effectiveness, deployment feasibility, and trust implications. Results demonstrate that task-specific obfuscation enables effective monitoring with reduced privacy risks, establishing the framework's readiness for real-world adoption and providing cross-domain recommendations for responsible, human-centric AI deployment in industry.
>
---
#### [new 024] GimbalDiffusion: Gravity-Aware Camera Control for Video Generation
- **分类: cs.CV**

- **简介: 该论文属于文本生成视频任务，旨在解决相机运动控制不精确的问题。提出GimbalDiffusion框架，利用重力参考和绝对坐标实现细粒度、物理对齐的相机控制，引入零俯仰条件增强引导，并构建新基准评估大范围俯仰变化下的性能。**

- **链接: [https://arxiv.org/pdf/2512.09112v1](https://arxiv.org/pdf/2512.09112v1)**

> **作者:** Frédéric Fortier-Chouinard; Yannick Hold-Geoffroy; Valentin Deschaintre; Matheus Gadelha; Jean-François Lalonde
>
> **备注:** Project page: https://lvsn.github.io/GimbalDiffusion/
>
> **摘要:** Recent progress in text-to-video generation has achieved remarkable realism, yet fine-grained control over camera motion and orientation remains elusive. Existing approaches typically encode camera trajectories through relative or ambiguous representations, limiting explicit geometric control. We introduce GimbalDiffusion, a framework that enables camera control grounded in physical-world coordinates, using gravity as a global reference. Instead of describing motion relative to previous frames, our method defines camera trajectories in an absolute coordinate system, allowing precise and interpretable control over camera parameters without requiring an initial reference frame. We leverage panoramic 360-degree videos to construct a wide variety of camera trajectories, well beyond the predominantly straight, forward-facing trajectories seen in conventional video data. To further enhance camera guidance, we introduce null-pitch conditioning, an annotation strategy that reduces the model's reliance on text content when conflicting with camera specifications (e.g., generating grass while the camera points towards the sky). Finally, we establish a benchmark for camera-aware video generation by rebalancing SpatialVID-HQ for comprehensive evaluation under wide camera pitch variation. Together, these contributions advance the controllability and robustness of text-to-video models, enabling precise, gravity-aligned camera manipulation within generative frameworks.
>
---
#### [new 025] HSCP: A Two-Stage Spectral Clustering Framework for Resource-Constrained UAV Identification
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对资源受限的无人机射频指纹识别任务，解决深度模型压缩中精度、效率与硬件适配难兼顾的问题。提出HSCP框架，结合基于CKA的谱聚类进行层与通道剪枝，并设计抗噪微调，实现高压缩率、高精度和强鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.08983v1](https://arxiv.org/pdf/2512.08983v1)**

> **作者:** Maoyu Wang; Yao Lu; Bo Zhou; Zhuangzhi Chen; Yun Lin; Qi Xuan; Guan Gui
>
> **摘要:** With the rapid development of Unmanned Aerial Vehicles (UAVs) and the increasing complexity of low-altitude security threats, traditional UAV identification methods struggle to extract reliable signal features and meet real-time requirements in complex environments. Recently, deep learning based Radio Frequency Fingerprint Identification (RFFI) approaches have greatly improved recognition accuracy. However, their large model sizes and high computational demands hinder deployment on resource-constrained edge devices. While model pruning offers a general solution for complexity reduction, existing weight, channel, and layer pruning techniques struggle to concurrently optimize compression rate, hardware acceleration, and recognition accuracy. To this end, in this paper, we introduce HSCP, a Hierarchical Spectral Clustering Pruning framework that combines layer pruning with channel pruning to achieve extreme compression, high performance, and efficient inference. In the first stage, HSCP employs spectral clustering guided by Centered Kernel Alignment (CKA) to identify and remove redundant layers. Subsequently, the same strategy is applied to the channel dimension to eliminate a finer redundancy. To ensure robustness, we further employ a noise-robust fine-tuning strategy. Experiments on the UAV-M100 benchmark demonstrate that HSCP outperforms existing channel and layer pruning methods. Specifically, HSCP achieves $86.39\%$ parameter reduction and $84.44\%$ FLOPs reduction on ResNet18 while improving accuracy by $1.49\%$ compared to the unpruned baseline, and maintains superior robustness even in low signal-to-noise ratio environments.
>
---
#### [new 026] Investigate the Low-level Visual Perception in Vision-Language based Image Quality Assessment
- **分类: cs.CV**

- **简介: 该论文研究基于视觉-语言模型的图像质量评估，探讨其对低级视觉失真的感知缺陷。通过构建失真分类任务，发现模型因过拟合模板而在对齐过程中丢失关键特征，提出强化视觉编码器可显著提升失真识别准确率。**

- **链接: [https://arxiv.org/pdf/2512.09573v1](https://arxiv.org/pdf/2512.09573v1)**

> **作者:** Yuan Li; Zitang Sun; Yen-Ju Chen; Shin'ya Nishida
>
> **摘要:** Recent advances in Image Quality Assessment (IQA) have leveraged Multi-modal Large Language Models (MLLMs) to generate descriptive explanations. However, despite their strong visual perception modules, these models often fail to reliably detect basic low-level distortions such as blur, noise, and compression, and may produce inconsistent evaluations across repeated inferences. This raises an essential question: do MLLM-based IQA systems truly perceive the visual features that matter? To examine this issue, we introduce a low-level distortion perception task that requires models to classify specific distortion types. Our component-wise analysis shows that although MLLMs are structurally capable of representing such distortions, they tend to overfit training templates, leading to biases in quality scoring. As a result, critical low-level features are weakened or lost during the vision-language alignment transfer stage. Furthermore, by computing the semantic distance between visual features and corresponding semantic tokens before and after component-wise fine-tuning, we show that improving the alignment of the vision encoder dramatically enhances distortion recognition accuracy, increasing it from 14.92% to 84.43%. Overall, these findings indicate that incorporating dedicated constraints on the vision encoder can strengthen text-explainable visual representations and enable MLLM-based pipelines to produce more coherent and interpretable reasoning in vision-centric tasks.
>
---
#### [new 027] Modality-Specific Enhancement and Complementary Fusion for Semi-Supervised Multi-Modal Brain Tumor Segmentation
- **分类: cs.CV**

- **简介: 该论文研究半监督多模态脑肿瘤分割，旨在解决模态间语义差异与信息互补利用不足的问题。提出模态特异性增强模块和可学习的互补融合模块，提升小样本下的分割性能。**

- **链接: [https://arxiv.org/pdf/2512.09801v1](https://arxiv.org/pdf/2512.09801v1)**

> **作者:** Tien-Dat Chung; Ba-Thinh Lam; Thanh-Huy Nguyen; Thien Nguyen; Nguyen Lan Vi Vu; Hoang-Loc Cao; Phat Kim Huynh; Min Xu
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Semi-supervised learning (SSL) has become a promising direction for medical image segmentation, enabling models to learn from limited labeled data alongside abundant unlabeled samples. However, existing SSL approaches for multi-modal medical imaging often struggle to exploit the complementary information between modalities due to semantic discrepancies and misalignment across MRI sequences. To address this, we propose a novel semi-supervised multi-modal framework that explicitly enhances modality-specific representations and facilitates adaptive cross-modal information fusion. Specifically, we introduce a Modality-specific Enhancing Module (MEM) to strengthen semantic cues unique to each modality via channel-wise attention, and a learnable Complementary Information Fusion (CIF) module to adaptively exchange complementary knowledge between modalities. The overall framework is optimized using a hybrid objective combining supervised segmentation loss and cross-modal consistency regularization on unlabeled data. Extensive experiments on the BraTS 2019 (HGG subset) demonstrate that our method consistently outperforms strong semi-supervised and multi-modal baselines under 1\%, 5\%, and 10\% labeled data settings, achieving significant improvements in both Dice and Sensitivity scores. Ablation studies further confirm the complementary effects of our proposed MEM and CIF in bridging cross-modality discrepancies and improving segmentation robustness under scarce supervision.
>
---
#### [new 028] TextGuider: Training-Free Guidance for Text Rendering via Attention Alignment
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决扩散模型中文字漏生成的问题。提出TextGuider方法，通过注意力对齐实现训练-free的文本引导，在不需微调的情况下提升文本完整性和识别准确率。**

- **链接: [https://arxiv.org/pdf/2512.09350v1](https://arxiv.org/pdf/2512.09350v1)**

> **作者:** Kanghyun Baek; Sangyub Lee; Jin Young Choi; Jaewoo Song; Daemin Park; Jooyoung Choi; Chaehun Shin; Bohyung Han; Sungroh Yoon
>
> **摘要:** Despite recent advances, diffusion-based text-to-image models still struggle with accurate text rendering. Several studies have proposed fine-tuning or training-free refinement methods for accurate text rendering. However, the critical issue of text omission, where the desired text is partially or entirely missing, remains largely overlooked. In this work, we propose TextGuider, a novel training-free method that encourages accurate and complete text appearance by aligning textual content tokens and text regions in the image. Specifically, we analyze attention patterns in MM-DiT models, particularly for text-related tokens intended to be rendered in the image. Leveraging this observation, we apply latent guidance during the early stage of denoising steps based on two loss functions that we introduce. Our method achieves state-of-the-art performance in test-time text rendering, with significant gains in recall and strong results in OCR accuracy and CLIP score.
>
---
#### [new 029] LiM-YOLO: Less is More with Pyramid Level Shift and Normalized Auxiliary Branch for Ship Detection in Optical Remote Sensing Imagery
- **分类: cs.CV; eess.IV**

- **简介: 该论文针对光学遥感影像中的船舶检测任务，解决因目标尺度差异大导致的小船漏检问题。提出LiM-YOLO模型，通过金字塔层级迁移和归一化辅助分支，提升小目标检测精度与训练稳定性。**

- **链接: [https://arxiv.org/pdf/2512.09700v1](https://arxiv.org/pdf/2512.09700v1)**

> **作者:** Seon-Hoon Kim; Hyeji Sim; Youeyun Jung; Ok-Chul Jung; Yerin Kim
>
> **备注:** 16 pages, 8 figures, 9 tables
>
> **摘要:** Applying general-purpose object detectors to ship detection in satellite imagery presents significant challenges due to the extreme scale disparity and morphological anisotropy of maritime targets. Standard architectures utilizing stride-32 (P5) layers often fail to resolve narrow vessels, resulting in spatial feature dilution. In this work, we propose LiM-YOLO, a specialized detector designed to resolve these domain-specific conflicts. Based on a statistical analysis of ship scales, we introduce a Pyramid Level Shift Strategy that reconfigures the detection head to P2-P4. This shift ensures compliance with Nyquist sampling criteria for small objects while eliminating the computational redundancy of deep layers. To further enhance training stability on high-resolution inputs, we incorporate a Group Normalized Convolutional Block for Linear Projection (GN-CBLinear), which mitigates gradient volatility in micro-batch settings. Validated on SODA-A, DOTA-v1.5, FAIR1M-v2.0, and ShipRSImageNet-V1, LiM-YOLO demonstrates superior detection accuracy and efficiency compared to state-of-the-art models. The code is available at https://github.com/egshkim/LiM-YOLO.
>
---
#### [new 030] Learning to Remove Lens Flare in Event Camera
- **分类: cs.CV**

- **简介: 该论文研究事件相机中的镜头耀斑去除，提出首个系统性框架E-Deflare。通过建立物理模型生成仿真与真实配对数据集，并设计E-DeflareNet网络实现先进去耀斑效果，提升下游任务性能。**

- **链接: [https://arxiv.org/pdf/2512.09016v1](https://arxiv.org/pdf/2512.09016v1)**

> **作者:** Haiqian Han; Lingdong Kong; Jianing Li; Ao Liang; Chengtao Zhu; Jiacheng Lyu; Lai Xing Ng; Xiangyang Ji; Wei Tsang Ooi; Benoit R. Cottereau
>
> **备注:** Preprint; 29 pages, 14 figures, 4 tables; Project Page at https://e-flare.github.io/
>
> **摘要:** Event cameras have the potential to revolutionize vision systems with their high temporal resolution and dynamic range, yet they remain susceptible to lens flare, a fundamental optical artifact that causes severe degradation. In event streams, this optical artifact forms a complex, spatio-temporal distortion that has been largely overlooked. We present E-Deflare, the first systematic framework for removing lens flare from event camera data. We first establish the theoretical foundation by deriving a physics-grounded forward model of the non-linear suppression mechanism. This insight enables the creation of the E-Deflare Benchmark, a comprehensive resource featuring a large-scale simulated training set, E-Flare-2.7K, and the first-ever paired real-world test set, E-Flare-R, captured by our novel optical system. Empowered by this benchmark, we design E-DeflareNet, which achieves state-of-the-art restoration performance. Extensive experiments validate our approach and demonstrate clear benefits for downstream tasks. Code and datasets are publicly available.
>
---
#### [new 031] Kaapana: A Comprehensive Open-Source Platform for Integrating AI in Medical Imaging Research Environments
- **分类: cs.CV**

- **简介: 该论文提出Kaapana，一个开源医疗影像研究平台，旨在解决多中心数据共享难、工具不统一的问题。通过将算法带到数据端，支持数据本地化处理，实现跨机构协作，提升AI研究的可重复性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.09644v1](https://arxiv.org/pdf/2512.09644v1)**

> **作者:** Ünal Akünal; Markus Bujotzek; Stefan Denner; Benjamin Hamm; Klaus Kades; Philipp Schader; Jonas Scherer; Marco Nolden; Peter Neher; Ralf Floca; Klaus Maier-Hein
>
> **摘要:** Developing generalizable AI for medical imaging requires both access to large, multi-center datasets and standardized, reproducible tooling within research environments. However, leveraging real-world imaging data in clinical research environments is still hampered by strict regulatory constraints, fragmented software infrastructure, and the challenges inherent in conducting large-cohort multicentre studies. This leads to projects that rely on ad-hoc toolchains that are hard to reproduce, difficult to scale beyond single institutions and poorly suited for collaboration between clinicians and data scientists. We present Kaapana, a comprehensive open-source platform for medical imaging research that is designed to bridge this gap. Rather than building single-use, site-specific tooling, Kaapana provides a modular, extensible framework that unifies data ingestion, cohort curation, processing workflows and result inspection under a common user interface. By bringing the algorithm to the data, it enables institutions to keep control over their sensitive data while still participating in distributed experimentation and model development. By integrating flexible workflow orchestration with user-facing applications for researchers, Kaapana reduces technical overhead, improves reproducibility and enables conducting large-scale, collaborative, multi-centre imaging studies. We describe the core concepts of the platform and illustrate how they can support diverse use cases, from local prototyping to nation-wide research networks. The open-source codebase is available at https://github.com/kaapana/kaapana
>
---
#### [new 032] A Dual-Domain Convolutional Network for Hyperspectral Single-Image Super-Resolution
- **分类: cs.CV**

- **简介: 该论文研究高光谱单图像超分辨率任务，旨在提升图像空间分辨率。提出轻量级双域网络DDSRNet，结合空间域残差学习与离散小波变换的频域增强，通过共享权重网络优化高低频信息，实现高效高性能超分辨率重建。**

- **链接: [https://arxiv.org/pdf/2512.09546v1](https://arxiv.org/pdf/2512.09546v1)**

> **作者:** Murat Karayaka; Usman Muhammad; Jorma Laaksonen; Md Ziaul Hoque; Tapio Seppänen
>
> **摘要:** This study presents a lightweight dual-domain super-resolution network (DDSRNet) that combines Spatial-Net with the discrete wavelet transform (DWT). Specifically, our proposed model comprises three main components: (1) a shallow feature extraction module, termed Spatial-Net, which performs residual learning and bilinear interpolation; (2) a low-frequency enhancement branch based on the DWT that refines coarse image structures; and (3) a shared high-frequency refinement branch that simultaneously enhances the LH (horizontal), HL (vertical), and HH (diagonal) wavelet subbands using a single CNN with shared weights. As a result, the DWT enables subband decomposition, while the inverse DWT reconstructs the final high-resolution output. By doing so, the integration of spatial- and frequency-domain learning enables DDSRNet to achieve highly competitive performance with low computational cost on three hyperspectral image datasets, demonstrating its effectiveness for hyperspectral image super-resolution.
>
---
#### [new 033] RAG-HAR: Retrieval Augmented Generation-based Human Activity Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文属人类活动识别（HAR）任务，旨在解决传统方法依赖大量标注数据和训练的问题。作者提出RAG-HAR，一种无需训练的检索增强框架，结合统计特征与大语言模型，通过语义检索相似样本实现准确活动识别，并支持未见活动的标注，提升泛化性与实用性。**

- **链接: [https://arxiv.org/pdf/2512.08984v1](https://arxiv.org/pdf/2512.08984v1)**

> **作者:** Nirhoshan Sivaroopan; Hansi Karunarathna; Chamara Madarasingha; Anura Jayasumana; Kanchana Thilakarathna
>
> **摘要:** Human Activity Recognition (HAR) underpins applications in healthcare, rehabilitation, fitness tracking, and smart environments, yet existing deep learning approaches demand dataset-specific training, large labeled corpora, and significant computational resources.We introduce RAG-HAR, a training-free retrieval-augmented framework that leverages large language models (LLMs) for HAR. RAG-HAR computes lightweight statistical descriptors, retrieves semantically similar samples from a vector database, and uses this contextual evidence to make LLM-based activity identification. We further enhance RAG-HAR by first applying prompt optimization and introducing an LLM-based activity descriptor that generates context-enriched vector databases for delivering accurate and highly relevant contextual information. Along with these mechanisms, RAG-HAR achieves state-of-the-art performance across six diverse HAR benchmarks. Most importantly, RAG-HAR attains these improvements without requiring model training or fine-tuning, emphasizing its robustness and practical applicability. RAG-HAR moves beyond known behaviors, enabling the recognition and meaningful labelling of multiple unseen human activities.
>
---
#### [new 034] Relightable and Dynamic Gaussian Avatar Reconstruction from Monocular Video
- **分类: cs.CV; cs.MM**

- **简介: 该论文研究单目视频中可重光照、可动画的人体 avatar 重建。针对现有方法在姿态变化下几何细节（如衣物褶皱）还原不足的问题，提出基于3D高斯点阵的 RnD-Avatar 框架，引入动态蒙皮权重与新正则化，提升形变精度与细节表现，实现高质量新视角、新姿态与任意光照下的渲染。**

- **链接: [https://arxiv.org/pdf/2512.09335v1](https://arxiv.org/pdf/2512.09335v1)**

> **作者:** Seonghwa Choi; Moonkyeong Choi; Mingyu Jang; Jaekyung Kim; Jianfei Cai; Wen-Huang Cheng; Sanghoon Lee
>
> **备注:** 8 pages, 9 figures, published in ACM MM 2025
>
> **摘要:** Modeling relightable and animatable human avatars from monocular video is a long-standing and challenging task. Recently, Neural Radiance Field (NeRF) and 3D Gaussian Splatting (3DGS) methods have been employed to reconstruct the avatars. However, they often produce unsatisfactory photo-realistic results because of insufficient geometrical details related to body motion, such as clothing wrinkles. In this paper, we propose a 3DGS-based human avatar modeling framework, termed as Relightable and Dynamic Gaussian Avatar (RnD-Avatar), that presents accurate pose-variant deformation for high-fidelity geometrical details. To achieve this, we introduce dynamic skinning weights that define the human avatar's articulation based on pose while also learning additional deformations induced by body motion. We also introduce a novel regularization to capture fine geometric details under sparse visual cues. Furthermore, we present a new multi-view dataset with varied lighting conditions to evaluate relight. Our framework enables realistic rendering of novel poses and views while supporting photo-realistic lighting effects under arbitrary lighting conditions. Our method achieves state-of-the-art performance in novel view synthesis, novel pose rendering, and relighting.
>
---
#### [new 035] Video-QTR: Query-Driven Temporal Reasoning Framework for Lightweight Video Understanding
- **分类: cs.CV**

- **简介: 该论文聚焦视频理解任务，旨在解决现有模型处理长视频时计算开销大的问题。提出Video-QTR框架，通过查询驱动的时序推理，动态选择关键帧，减少冗余计算，在降低73%帧消耗的同时实现最优性能。**

- **链接: [https://arxiv.org/pdf/2512.09354v1](https://arxiv.org/pdf/2512.09354v1)**

> **作者:** Xinkui Zhao; Zuxin Wang; Yifan Zhang; Guanjie Cheng; Yueshen Xu; Shuiguang Deng; Chang Liu; Naibo Wang; Jianwei Yin
>
> **摘要:** The rapid development of multimodal large-language models (MLLMs) has significantly expanded the scope of visual language reasoning, enabling unified systems to interpret and describe complex visual content. However, applying these models to long-video understanding remains computationally intensive. Dense frame encoding generates excessive visual tokens, leading to high memory consumption, redundant computation, and limited scalability in real-world applications. This inefficiency highlights a key limitation of the traditional process-then-reason paradigm, which analyzes visual streams exhaustively before semantic reasoning. To address this challenge, we introduce Video-QTR (Query-Driven Temporal Reasoning), a lightweight framework that redefines video comprehension as a query-guided reasoning process. Instead of encoding every frame, Video-QTR dynamically allocates perceptual resources based on the semantic intent of the query, creating an adaptive feedback loop between reasoning and perception. Extensive experiments across five benchmarks: MSVD-QA, Activity Net-QA, Movie Chat, and Video MME demonstrate that Video-QTR achieves state-of-the-art performance while reducing input frame consumption by up to 73%. These results confirm that query-driven temporal reasoning provides an efficient and scalable solution for video understanding.
>
---
#### [new 036] A Physics-Constrained, Design-Driven Methodology for Defect Dataset Generation in Optical Lithography
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对光刻缺陷数据稀缺问题，提出一种物理约束、设计驱动的缺陷数据集生成方法。通过数学形态学合成缺陷布局，经DMD光刻制备样本并获取显微图像，生成含像素级标注的大规模数据集，用于提升AI在半导体检测中的性能。**

- **链接: [https://arxiv.org/pdf/2512.09001v1](https://arxiv.org/pdf/2512.09001v1)**

> **作者:** Yuehua Hu; Jiyeong Kong; Dong-yeol Shin; Jaekyun Kim; Kyung-Tae Kang
>
> **摘要:** The efficacy of Artificial Intelligence (AI) in micro/nano manufacturing is fundamentally constrained by the scarcity of high-quality and physically grounded training data for defect inspection. Lithography defect data from semiconductor industry are rarely accessible for research use, resulting in a shortage of publicly available datasets. To address this bottleneck in lithography, this study proposes a novel methodology for generating large-scale, physically valid defect datasets with pixel-level annotations. The framework begins with the ab initio synthesis of defect layouts using controllable, physics-constrained mathematical morphology operations (erosion and dilation) applied to the original design-level layout. These synthesized layouts, together with their defect-free counterparts, are fabricated into physical samples via high-fidelity digital micromirror device (DMD)-based lithography. Optical micrographs of the synthesized defect samples and their defect-free references are then compared to create consistent defect delineation annotations. Using this methodology, we constructed a comprehensive dataset of 3,530 Optical micrographs containing 13,365 annotated defect instances including four classes: bridge, burr, pinch, and contamination. Each defect instance is annotated with a pixel-accurate segmentation mask, preserving full contour and geometry. The segmentation-based Mask R-CNN achieves AP@0.5 of 0.980, 0.965, and 0.971, compared with 0.740, 0.719, and 0.717 for Faster R-CNN on bridge, burr, and pinch classes, representing a mean AP@0.5 improvement of approximately 34%. For the contamination class, Mask R-CNN achieves an AP@0.5 roughly 42% higher than Faster R-CNN. These consistent gains demonstrate that our proposed methodology to generate defect datasets with pixel-level annotations is feasible for robust AI-based Measurement/Inspection (MI) in semiconductor fabrication.
>
---
#### [new 037] MedForget: Hierarchy-Aware Multimodal Unlearning Testbed for Medical AI
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文聚焦医疗AI中的模型遗忘任务，旨在解决敏感数据合规删除问题。作者提出MedForget测试平台，构建层级化多模态数据 benchmark，评估不同粒度的遗忘效果，并设计重构攻击验证遗忘彻底性，推动符合HIPAA的医疗AI系统发展。**

- **链接: [https://arxiv.org/pdf/2512.09867v1](https://arxiv.org/pdf/2512.09867v1)**

> **作者:** Fengli Wu; Vaidehi Patil; Jaehong Yoon; Yue Zhang; Mohit Bansal
>
> **备注:** Dataset and Code: https://github.com/fengli-wu/MedForget
>
> **摘要:** Pretrained Multimodal Large Language Models (MLLMs) are increasingly deployed in medical AI systems for clinical reasoning, diagnosis support, and report generation. However, their training on sensitive patient data raises critical privacy and compliance challenges under regulations such as HIPAA and GDPR, which enforce the "right to be forgotten". Unlearning, the process of tuning models to selectively remove the influence of specific training data points, offers a potential solution, yet its effectiveness in complex medical settings remains underexplored. To systematically study this, we introduce MedForget, a Hierarchy-Aware Multimodal Unlearning Testbed with explicit retain and forget splits and evaluation sets containing rephrased variants. MedForget models hospital data as a nested hierarchy (Institution -> Patient -> Study -> Section), enabling fine-grained assessment across eight organizational levels. The benchmark contains 3840 multimodal (image, question, answer) instances, each hierarchy level having a dedicated unlearning target, reflecting distinct unlearning challenges. Experiments with four SOTA unlearning methods on three tasks (generation, classification, cloze) show that existing methods struggle to achieve complete, hierarchy-aware forgetting without reducing diagnostic performance. To test whether unlearning truly deletes hierarchical pathways, we introduce a reconstruction attack that progressively adds hierarchical level context to prompts. Models unlearned at a coarse granularity show strong resistance, while fine-grained unlearning leaves models vulnerable to such reconstruction. MedForget provides a practical, HIPAA-aligned testbed for building compliant medical AI systems.
>
---
#### [new 038] SIP: Site in Pieces- A Dataset of Disaggregated Construction-Phase 3D Scans for Semantic Segmentation and Scene Understanding
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于3D场景理解任务，旨在解决现有数据集无法反映真实施工环境LiDAR扫描碎片化特性的问题。作者构建了SIP数据集，包含点级标注的室内外施工场景，保留了实际采集中的稀疏性与不完整性，支持语义分割等任务的鲁棒评估。**

- **链接: [https://arxiv.org/pdf/2512.09062v1](https://arxiv.org/pdf/2512.09062v1)**

> **作者:** Seongyong Kim; Yong Kwon Cho
>
> **摘要:** Accurate 3D scene interpretation in active construction sites is essential for progress monitoring, safety assessment, and digital twin development. LiDAR is widely used in construction because it offers advantages over camera-based systems, performing reliably in cluttered and dynamically changing conditions. Yet most public datasets for 3D perception are derived from densely fused scans with uniform sampling and complete visibility, conditions that do not reflect real construction sites. Field data are often collected as isolated single-station LiDAR views, constrained by safety requirements, limited access, and ongoing operations. These factors lead to radial density decay, fragmented geometry, and view-dependent visibility-characteristics that remain underrepresented in existing datasets. This paper presents SIP, Site in Pieces, a dataset created to reflect the practical constraints of LiDAR acquisition during construction. SIP provides indoor and outdoor scenes captured with a terrestrial LiDAR scanner and annotated at the point level using a taxonomy tailored to construction environments: A. Built Environment, B. Construction Operations, and C. Site Surroundings. The dataset includes both structural components and slender temporary objects such as scaffolding, MEP piping, and scissor lifts, where sparsity caused by occlusion and fragmented geometry make segmentation particularly challenging. The scanning protocol, annotation workflow, and quality control procedures establish a consistent foundation for the dataset. SIP is openly available with a supporting Git repository, offering adaptable class configurations that streamline adoption within modern 3D deep learning frameworks. By providing field data that retain real-world sensing characteristics, SIP enables robust benchmarking and contributes to advancing construction-oriented 3D vision tasks.
>
---
#### [new 039] Mitigating Bias with Words: Inducing Demographic Ambiguity in Face Recognition Templates by Text Encoding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属人脸识别任务，旨在缓解因面部特征与人口统计属性纠缠导致的识别偏见。提出UTIE方法，利用视觉-语言模型将跨群体文本信息融入人脸嵌入，增强身份特征、弱化群体属性，提升跨群体识别公平性。**

- **链接: [https://arxiv.org/pdf/2512.08981v1](https://arxiv.org/pdf/2512.08981v1)**

> **作者:** Tahar Chettaoui; Naser Damer; Fadi Boutros
>
> **备注:** Accepted at BMVC workshop (SRBS) 2025
>
> **摘要:** Face recognition (FR) systems are often prone to demographic biases, partially due to the entanglement of demographic-specific information with identity-relevant features in facial embeddings. This bias is extremely critical in large multicultural cities, especially where biometrics play a major role in smart city infrastructure. The entanglement can cause demographic attributes to overshadow identity cues in the embedding space, resulting in disparities in verification performance across different demographic groups. To address this issue, we propose a novel strategy, Unified Text-Image Embedding (UTIE), which aims to induce demographic ambiguity in face embeddings by enriching them with information related to other demographic groups. This encourages face embeddings to emphasize identity-relevant features and thus promotes fairer verification performance across groups. UTIE leverages the zero-shot capabilities and cross-modal semantic alignment of Vision-Language Models (VLMs). Given that VLMs are naturally trained to align visual and textual representations, we enrich the facial embeddings of each demographic group with text-derived demographic features extracted from other demographic groups. This encourages a more neutral representation in terms of demographic attributes. We evaluate UTIE using three VLMs, CLIP, OpenCLIP, and SigLIP, on two widely used benchmarks, RFW and BFW, designed to assess bias in FR. Experimental results show that UTIE consistently reduces bias metrics while maintaining, or even improving in several cases, the face verification accuracy.
>
---
#### [new 040] WonderZoom: Multi-Scale 3D World Generation
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文提出WonderZoom，解决单图像生成多尺度3D场景的问题。现有方法局限于单一尺度，缺乏跨尺度一致性。作者设计了尺度自适应高斯surfels和渐进细节合成器，实现从宏观到微观的连贯3D内容生成与实时渲染。**

- **链接: [https://arxiv.org/pdf/2512.09164v1](https://arxiv.org/pdf/2512.09164v1)**

> **作者:** Jin Cao; Hong-Xing Yu; Jiajun Wu
>
> **备注:** Project website: https://wonderzoom.github.io/ The first two authors contributed equally
>
> **摘要:** We present WonderZoom, a novel approach to generating 3D scenes with contents across multiple spatial scales from a single image. Existing 3D world generation models remain limited to single-scale synthesis and cannot produce coherent scene contents at varying granularities. The fundamental challenge is the lack of a scale-aware 3D representation capable of generating and rendering content with largely different spatial sizes. WonderZoom addresses this through two key innovations: (1) scale-adaptive Gaussian surfels for generating and real-time rendering of multi-scale 3D scenes, and (2) a progressive detail synthesizer that iteratively generates finer-scale 3D contents. Our approach enables users to "zoom into" a 3D region and auto-regressively synthesize previously non-existent fine details from landscapes to microscopic features. Experiments demonstrate that WonderZoom significantly outperforms state-of-the-art video and 3D models in both quality and alignment, enabling multi-scale 3D world creation from a single image. We show video results and an interactive viewer of generated multi-scale 3D worlds in https://wonderzoom.github.io/
>
---
#### [new 041] UniUGP: Unifying Understanding, Generation, and Planing For End-to-end Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文研究端到端自动驾驶，旨在解决长尾场景下感知、推理与规划能力不足的问题。提出UniUGP框架，融合视觉语言模型与视频生成模型，通过四阶段训练实现可解释推理、轨迹规划与未来视频生成，提升复杂场景的泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.09864v1](https://arxiv.org/pdf/2512.09864v1)**

> **作者:** Hao Lu; Ziyang Liu; Guangfeng Jiang; Yuanfei Luo; Sheng Chen; Yangang Zhang; Ying-Cong Chen
>
> **备注:** Project Page: https://seed-uniugp.github.io/
>
> **摘要:** Autonomous driving (AD) systems struggle in long-tail scenarios due to limited world knowledge and weak visual dynamic modeling. Existing vision-language-action (VLA)-based methods cannot leverage unlabeled videos for visual causal learning, while world model-based methods lack reasoning capabilities from large language models. In this paper, we construct multiple specialized datasets providing reasoning and planning annotations for complex scenarios. Then, a unified Understanding-Generation-Planning framework, named UniUGP, is proposed to synergize scene reasoning, future video generation, and trajectory planning through a hybrid expert architecture. By integrating pre-trained VLMs and video generation models, UniUGP leverages visual dynamics and semantic reasoning to enhance planning performance. Taking multi-frame observations and language instructions as input, it produces interpretable chain-of-thought reasoning, physically consistent trajectories, and coherent future videos. We introduce a four-stage training strategy that progressively builds these capabilities across multiple existing AD datasets, along with the proposed specialized datasets. Experiments demonstrate state-of-the-art performance in perception, reasoning, and decision-making, with superior generalization to challenging long-tail situations.
>
---
#### [new 042] Benchmarking Document Parsers on Mathematical Formula Extraction from PDFs
- **分类: cs.CV**

- **简介: 该论文聚焦PDF中数学公式提取的评测任务，解决现有基准缺乏语义评估的问题。提出合成PDF基准框架，引入LLM作为评判模型，并构建两阶段匹配 pipeline，通过人类验证证明其与人工判断高度相关，系统评估了20余种解析器性能。**

- **链接: [https://arxiv.org/pdf/2512.09874v1](https://arxiv.org/pdf/2512.09874v1)**

> **作者:** Pius Horn; Janis Keuper
>
> **摘要:** Correctly parsing mathematical formulas from PDFs is critical for training large language models and building scientific knowledge bases from academic literature, yet existing benchmarks either exclude formulas entirely or lack semantically-aware evaluation metrics. We introduce a novel benchmarking framework centered on synthetically generated PDFs with precise LaTeX ground truth, enabling systematic control over layout, formulas, and content characteristics. A key methodological contribution is pioneering LLM-as-a-judge for semantic formula assessment, combined with a robust two-stage matching pipeline that handles parser output inconsistencies. Through human validation on 250 formula pairs (750 ratings from 30 evaluators), we demonstrate that LLM-based evaluation achieves substantially higher correlation with human judgment (Pearson r=0.78) compared to CDM (r=0.34) and text similarity (r~0). Evaluating 20+ contemporary PDF parsers (including specialized OCR models, vision-language models, and rule-based approaches) across 100 synthetic documents with 2,000+ formulas reveals significant performance disparities. Our findings provide crucial insights for practitioners selecting parsers for downstream applications and establish a robust, scalable methodology that enables reproducible evaluation of PDF formula extraction quality. Code and benchmark data: https://github.com/phorn1/pdf-parse-bench
>
---
#### [new 043] FROMAT: Multiview Material Appearance Transfer via Few-Shot Self-Attention Adaptation
- **分类: cs.CV**

- **简介: 该论文属于多视角生成任务，旨在解决现有扩散模型在材质外观编辑上的局限。作者提出FROMAT方法，通过少量样本自注意力适配，融合输入对象身份与参考图外观，在无需显式几何表示下实现多视角一致的材质迁移。**

- **链接: [https://arxiv.org/pdf/2512.09617v1](https://arxiv.org/pdf/2512.09617v1)**

> **作者:** Hubert Kompanowski; Varun Jampani; Aaryaman Vasishta; Binh-Son Hua
>
> **摘要:** Multiview diffusion models have rapidly emerged as a powerful tool for content creation with spatial consistency across viewpoints, offering rich visual realism without requiring explicit geometry and appearance representation. However, compared to meshes or radiance fields, existing multiview diffusion models offer limited appearance manipulation, particularly in terms of material, texture, or style. In this paper, we present a lightweight adaptation technique for appearance transfer in multiview diffusion models. Our method learns to combine object identity from an input image with appearance cues rendered in a separate reference image, producing multi-view-consistent output that reflects the desired materials, textures, or styles. This allows explicit specification of appearance parameters at generation time while preserving the underlying object geometry and view coherence. We leverage three diffusion denoising processes responsible for generating the original object, the reference, and the target images, and perform reverse sampling to aggregate a small subset of layer-wise self-attention features from the object and the reference to influence the target generation. Our method requires only a few training examples to introduce appearance awareness to pretrained multiview models. The experiments show that our method provides a simple yet effective way toward multiview generation with diverse appearance, advocating the adoption of implicit generative 3D representations in practice.
>
---
#### [new 044] What Happens When: Learning Temporal Orders of Events in Videos
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视频多模态模型对事件时序的理解能力，发现现有模型依赖先验知识而非真实时序。为此提出新基准VECTOR和方法MECOT，通过细粒度事件描述与思维链提示提升模型时序理解能力。**

- **链接: [https://arxiv.org/pdf/2512.08979v1](https://arxiv.org/pdf/2512.08979v1)**

> **作者:** Daechul Ahn; Yura Choi; Hyeonbeom Choi; Seongwon Cho; San Kim; Jonghyun Choi
>
> **备注:** WACV 2026
>
> **摘要:** Video Large Multimodal Models (VLMMs) have shown impressive performance in video understanding, yet their ability to accurately capture the temporal order of multiple events remains underexplored. We interestingly observe that, even when video frames are scrambled, models perform very well on the existing benchmarks by comprehensive experiments. This implies that VLMMs may not necessarily rely on accurate sequential processing of visual events, but instead depend on prior knowledge of typical scenarios to answer the question. To benchmark temporal understanding capabilities in VLMMs, we propose VECTOR, designed to explicitly assess a model's ability to identify the temporal order of events. On this benchmark, we observe that various VLMMs often fail to understand the orders of events. To address this, we propose MECOT (Multi-Event instruction fine-tuning with Chain-of-Thought), which (1) trains models on detailed, event-by-event video descriptions and (2) using chain-of-thought prompts at inference to enhance temporal awareness. MECOT outperforms prior arts on VECTOR as well as improving performance on existing video benchmarks, implying effectiveness of temporal understanding. We release our code, model and datasets.
>
---
#### [new 045] Deterministic World Models for Verification of Closed-loop Vision-based Systems
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于闭环视觉系统验证任务，旨在解决生成模型因随机隐变量导致的过近似误差问题。提出确定性世界模型（DWM），直接由系统状态生成图像，并结合双目标损失与星集可达分析，提升验证精度与紧致性。**

- **链接: [https://arxiv.org/pdf/2512.08991v1](https://arxiv.org/pdf/2512.08991v1)**

> **作者:** Yuang Geng; Zhuoyang Zhou; Zhongzheng Zhang; Siyuan Pan; Hoang-Dung Tran; Ivan Ruchkin
>
> **备注:** 22 pages, 10 figures. Submitted to FM 2026
>
> **摘要:** Verifying closed-loop vision-based control systems remains a fundamental challenge due to the high dimensionality of images and the difficulty of modeling visual environments. While generative models are increasingly used as camera surrogates in verification, their reliance on stochastic latent variables introduces unnecessary overapproximation error. To address this bottleneck, we propose a Deterministic World Model (DWM) that maps system states directly to generative images, effectively eliminating uninterpretable latent variables to ensure precise input bounds. The DWM is trained with a dual-objective loss function that combines pixel-level reconstruction accuracy with a control difference loss to maintain behavioral consistency with the real system. We integrate DWM into a verification pipeline utilizing Star-based reachability analysis (StarV) and employ conformal prediction to derive rigorous statistical bounds on the trajectory deviation between the world model and the actual vision-based system. Experiments on standard benchmarks show that our approach yields significantly tighter reachable sets and better verification performance than a latent-variable baseline.
>
---
#### [new 046] GAINS: Gaussian-based Inverse Rendering from Sparse Multi-View Captures
- **分类: cs.CV**

- **简介: 该论文属于逆渲染任务，旨在从稀疏多视角图像中恢复高精度几何与材质。针对稀疏输入导致的几何-材质-光照歧义问题，提出GAINS框架，通过单目深度/法线、分割、内在图像分解和扩散先验分两阶段优化几何与材质估计，显著提升稀疏条件下的重建质量。**

- **链接: [https://arxiv.org/pdf/2512.09925v1](https://arxiv.org/pdf/2512.09925v1)**

> **作者:** Patrick Noras; Jun Myeong Choi; Didier Stricker; Pieter Peers; Roni Sengupta
>
> **备注:** 23 pages, 18 figures
>
> **摘要:** Recent advances in Gaussian Splatting-based inverse rendering extend Gaussian primitives with shading parameters and physically grounded light transport, enabling high-quality material recovery from dense multi-view captures. However, these methods degrade sharply under sparse-view settings, where limited observations lead to severe ambiguity between geometry, reflectance, and lighting. We introduce GAINS (Gaussian-based Inverse rendering from Sparse multi-view captures), a two-stage inverse rendering framework that leverages learning-based priors to stabilize geometry and material estimation. GAINS first refines geometry using monocular depth/normal and diffusion priors, then employs segmentation, intrinsic image decomposition (IID), and diffusion priors to regularize material recovery. Extensive experiments on synthetic and real-world datasets show that GAINS significantly improves material parameter accuracy, relighting quality, and novel-view synthesis compared to state-of-the-art Gaussian-based inverse rendering methods, especially under sparse-view settings. Project page: https://patrickbail.github.io/gains/
>
---
#### [new 047] Beyond Sequences: A Benchmark for Atomic Hand-Object Interaction Using a Static RNN Encoder
- **分类: cs.CV**

- **简介: 该论文聚焦手-物交互中的原子状态分类（如接近、抓取、持握），提出一种结构化特征提取方法，并发现将双向RNN的序列长度设为1时，作为静态特征编码器可显著提升性能，达到97.60%准确率，为低层交互识别提供了新基准。**

- **链接: [https://arxiv.org/pdf/2512.09626v1](https://arxiv.org/pdf/2512.09626v1)**

> **作者:** Yousef Azizi Movahed; Fatemeh Ziaeetabar
>
> **备注:** Code available at: https://github.com/YousefAMovahed/beyond-sequences-hoi-benchmark
>
> **摘要:** Reliably predicting human intent in hand-object interactions is an open challenge for computer vision. Our research concentrates on a fundamental sub-problem: the fine-grained classification of atomic interaction states, namely 'approaching', 'grabbing', and 'holding'. To this end, we introduce a structured data engineering process that converts raw videos from the MANIAC dataset into 27,476 statistical-kinematic feature vectors. Each vector encapsulates relational and dynamic properties from a short temporal window of motion. Our initial hypothesis posited that sequential modeling would be critical, leading us to compare static classifiers (MLPs) against temporal models (RNNs). Counter-intuitively, the key discovery occurred when we set the sequence length of a Bidirectional RNN to one (seq_length=1). This modification converted the network's function, compelling it to act as a high-capacity static feature encoder. This architectural change directly led to a significant accuracy improvement, culminating in a final score of 97.60%. Of particular note, our optimized model successfully overcame the most challenging transitional class, 'grabbing', by achieving a balanced F1-score of 0.90. These findings provide a new benchmark for low-level hand-object interaction recognition using structured, interpretable features and lightweight architectures.
>
---
#### [new 048] UnReflectAnything: RGB-Only Highlight Removal by Rendering Synthetic Specular Supervision
- **分类: cs.CV**

- **简介: 该论文研究单图高光去除任务，旨在消除RGB图像中由非朗伯表面和复杂光照引起的高光干扰。作者提出UnReflectAnything框架，通过虚拟高光合成生成配对监督数据，并结合视觉Transformer与特征级修复模块，实现无需真实标注的高光去除，在自然与手术场景均表现优异。**

- **链接: [https://arxiv.org/pdf/2512.09583v1](https://arxiv.org/pdf/2512.09583v1)**

> **作者:** Alberto Rota; Mert Kiray; Mert Asim Karaoglu; Patrick Ruhkamp; Elena De Momi; Nassir Navabm; Benjamin Busam
>
> **摘要:** Specular highlights distort appearance, obscure texture, and hinder geometric reasoning in both natural and surgical imagery. We present UnReflectAnything, an RGB-only framework that removes highlights from a single image by predicting a highlight map together with a reflection-free diffuse reconstruction. The model uses a frozen vision transformer encoder to extract multi-scale features, a lightweight head to localize specular regions, and a token-level inpainting module that restores corrupted feature patches before producing the final diffuse image. To overcome the lack of paired supervision, we introduce a Virtual Highlight Synthesis pipeline that renders physically plausible specularities using monocular geometry, Fresnel-aware shading, and randomized lighting which enables training on arbitrary RGB images with correct geometric structure. UnReflectAnything generalizes across natural and surgical domains where non-Lambertian surfaces and non-uniform lighting create severe highlights and it achieves competitive performance with state-of-the-art results on several benchmarks. Project Page: https://alberto-rota.github.io/UnReflectAnything/
>
---
#### [new 049] VABench: A Comprehensive Benchmark for Audio-Video Generation
- **分类: cs.CV; cs.SD**

- **简介: 该论文聚焦音频-视频同步生成任务，旨在解决现有基准缺乏对音视频协同生成评估的问题。作者提出VABench，包含三类生成任务、七个内容类别及15个评估维度，全面评测音视频生成质量与同步性。**

- **链接: [https://arxiv.org/pdf/2512.09299v1](https://arxiv.org/pdf/2512.09299v1)**

> **作者:** Daili Hua; Xizhi Wang; Bohan Zeng; Xinyi Huang; Hao Liang; Junbo Niu; Xinlong Chen; Quanqing Xu; Wentao Zhang
>
> **备注:** 24 pages, 25 figures
>
> **摘要:** Recent advances in video generation have been remarkable, enabling models to produce visually compelling videos with synchronized audio. While existing video generation benchmarks provide comprehensive metrics for visual quality, they lack convincing evaluations for audio-video generation, especially for models aiming to generate synchronized audio-video outputs. To address this gap, we introduce VABench, a comprehensive and multi-dimensional benchmark framework designed to systematically evaluate the capabilities of synchronous audio-video generation. VABench encompasses three primary task types: text-to-audio-video (T2AV), image-to-audio-video (I2AV), and stereo audio-video generation. It further establishes two major evaluation modules covering 15 dimensions. These dimensions specifically assess pairwise similarities (text-video, text-audio, video-audio), audio-video synchronization, lip-speech consistency, and carefully curated audio and video question-answering (QA) pairs, among others. Furthermore, VABench covers seven major content categories: animals, human sounds, music, environmental sounds, synchronous physical sounds, complex scenes, and virtual worlds. We provide a systematic analysis and visualization of the evaluation results, aiming to establish a new standard for assessing video generation models with synchronous audio capabilities and to promote the comprehensive advancement of the field.
>
---
#### [new 050] GLACIA: Instance-Aware Positional Reasoning for Glacial Lake Segmentation via Multimodal Large Language Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究 glacial lake 分割任务，旨在解决现有方法缺乏高层语义与可解释推理的问题。作者提出 GLACIA 框架，结合多模态大模型生成分割结果与实例感知的空间推理描述，并构建新数据集 GLake-Pos，提升分割性能与决策支持能力。**

- **链接: [https://arxiv.org/pdf/2512.09251v1](https://arxiv.org/pdf/2512.09251v1)**

> **作者:** Lalit Maurya; Saurabh Kaushik; Beth Tellman
>
> **摘要:** Glacial lake monitoring bears great significance in mitigating the anticipated risk of Glacial Lake Outburst Floods. However, existing segmentation methods based on convolutional neural networks (CNNs) and Vision Transformers (ViTs), remain constrained to pixel-level predictions, lacking high-level global scene semantics and human-interpretable reasoning. To address this, we introduce GLACIA (\textbf{G}lacial \textbf{LA}ke segmentation with \textbf{C}ontextual \textbf{I}nstance \textbf{A}wareness), the first framework that integrates large language models with segmentation capabilities to produce both accurate segmentation masks and corresponding spatial reasoning outputs. We construct the Glacial Lake Position Reasoning (GLake-Pos) dataset pipeline, which provides diverse, spatially grounded question-answer pairs designed to overcome the lack of instance-aware positional reasoning data in remote sensing. Comparative evaluation demonstrate that GLACIA (mIoU: 87.30) surpasses state-of-the-art method based on CNNs (mIoU: 78.55 - 79.01), ViTs (mIoU: 69.27 - 81.75), Geo-foundation models (mIoU: 76.37 - 87.10), and reasoning based segmentation methods (mIoU: 60.12 - 75.66). Our approach enables intuitive disaster preparedness and informed policy-making in the context of rapidly changing glacial environments by facilitating natural language interaction, thereby supporting more efficient and interpretable decision-making. The code is released on https://github.com/lalitmaurya47/GLACIA
>
---
#### [new 051] From Graphs to Gates: DNS-HyXNet, A Lightweight and Deployable Sequential Model for Real-Time DNS Tunnel Detection
- **分类: cs.CV**

- **简介: 该论文针对DNS隧道检测任务，解决图方法延迟高、难部署的问题。提出轻量级DNS-HyXNet模型，基于xLSTM直接处理DNS序列数据，无需建图，实现高效实时检测，准确率超99.96%，单样本延迟仅0.041ms。**

- **链接: [https://arxiv.org/pdf/2512.09565v1](https://arxiv.org/pdf/2512.09565v1)**

> **作者:** Faraz Ali; Muhammad Afaq; Mahmood Niazi; Muzammil Behzad
>
> **摘要:** Domain Name System (DNS) tunneling remains a covert channel for data exfiltration and command-and-control communication. Although graph-based methods such as GraphTunnel achieve strong accuracy, they introduce significant latency and computational overhead due to recursive parsing and graph construction, limiting their suitability for real-time deployment. This work presents DNS-HyXNet, a lightweight extended Long Short-Term Memory (xLSTM) hybrid framework designed for efficient sequence-based DNS tunnel detection. DNS-HyXNet integrates tokenized domain embeddings with normalized numerical DNS features and processes them through a two-layer xLSTM network that directly learns temporal dependencies from packet sequences, eliminating the need for graph reconstruction and enabling single-stage multi-class classification. The model was trained and evaluated on two public benchmark datasets with carefully tuned hyperparameters to ensure low memory consumption and fast inference. Across all experimental splits of the DNS-Tunnel-Datasets, DNS-HyXNet achieved up to 99.99% accuracy, with macro-averaged precision, recall, and F1-scores exceeding 99.96%, and demonstrated a per-sample detection latency of just 0.041 ms, confirming its scalability and real-time readiness. These results show that sequential modeling with xLSTM can effectively replace computationally expensive recursive graph generation, offering a deployable and energy-efficient alternative for real-time DNS tunnel detection on commodity hardware.
>
---
#### [new 052] UniPart: Part-Level 3D Generation with Unified 3D Geom-Seg Latents
- **分类: cs.CV**

- **简介: 该论文研究图像引导的部件级3D生成任务，旨在解决现有方法在部分分割控制和几何质量上的不足。作者提出UniPart框架，通过统一的几何-分割潜表示和两阶段扩散模型，实现细粒度可控且高保真的部件级3D生成。**

- **链接: [https://arxiv.org/pdf/2512.09435v1](https://arxiv.org/pdf/2512.09435v1)**

> **作者:** Xufan He; Yushuang Wu; Xiaoyang Guo; Chongjie Ye; Jiaqing Zhou; Tianlei Hu; Xiaoguang Han; Dong Du
>
> **摘要:** Part-level 3D generation is essential for applications requiring decomposable and structured 3D synthesis. However, existing methods either rely on implicit part segmentation with limited granularity control or depend on strong external segmenters trained on large annotated datasets. In this work, we observe that part awareness emerges naturally during whole-object geometry learning and propose Geom-Seg VecSet, a unified geometry-segmentation latent representation that jointly encodes object geometry and part-level structure. Building on this representation, we introduce UniPart, a two-stage latent diffusion framework for image-guided part-level 3D generation. The first stage performs joint geometry generation and latent part segmentation, while the second stage conditions part-level diffusion on both whole-object and part-specific latents. A dual-space generation scheme further enhances geometric fidelity by predicting part latents in both global and canonical spaces. Extensive experiments demonstrate that UniPart achieves superior segmentation controllability and part-level geometric quality compared with existing approaches.
>
---
#### [new 053] FUSER: Feed-Forward MUltiview 3D Registration Transformer and SE(3)$^N$ Diffusion Refinement
- **分类: cs.CV**

- **简介: 该论文研究多视角点云配准，旨在避免传统 pairwise 匹配的高计算成本与病态问题。提出 FUSER，首个前馈 Transformer 模型，在统一隐空间直接预测全局位姿；并结合 SE(3)^N 扩散优化框架 FUSER-DF 进一步精化结果，实现高效高精度配准。**

- **链接: [https://arxiv.org/pdf/2512.09373v1](https://arxiv.org/pdf/2512.09373v1)**

> **作者:** Haobo Jiang; Jin Xie; Jian Yang; Liang Yu; Jianmin Zheng
>
> **备注:** 13 pages, 6 figures
>
> **摘要:** Registration of multiview point clouds conventionally relies on extensive pairwise matching to build a pose graph for global synchronization, which is computationally expensive and inherently ill-posed without holistic geometric constraints. This paper proposes FUSER, the first feed-forward multiview registration transformer that jointly processes all scans in a unified, compact latent space to directly predict global poses without any pairwise estimation. To maintain tractability, FUSER encodes each scan into low-resolution superpoint features via a sparse 3D CNN that preserves absolute translation cues, and performs efficient intra- and inter-scan reasoning through a Geometric Alternating Attention module. Particularly, we transfer 2D attention priors from off-the-shelf foundation models to enhance 3D feature interaction and geometric consistency. Building upon FUSER, we further introduce FUSER-DF, an SE(3)$^N$ diffusion refinement framework to correct FUSER's estimates via denoising in the joint SE(3)$^N$ space. FUSER acts as a surrogate multiview registration model to construct the denoiser, and a prior-conditioned SE(3)$^N$ variational lower bound is derived for denoising supervision. Extensive experiments on 3DMatch, ScanNet and ArkitScenes demonstrate that our approach achieves the superior registration accuracy and outstanding computational efficiency.
>
---
#### [new 054] FunPhase: A Periodic Functional Autoencoder for Motion Generation via Phase Manifolds
- **分类: cs.CV**

- **简介: 该论文属于运动生成任务，旨在解耦人体运动中空间与时间的耦合问题。作者提出FunPhase，一种基于相位流形的功能性周期自编码器，通过函数空间建模实现任意时序采样、超分辨率和跨骨架泛化，统一了运动预测与生成。**

- **链接: [https://arxiv.org/pdf/2512.09423v1](https://arxiv.org/pdf/2512.09423v1)**

> **作者:** Marco Pegoraro; Evan Atherton; Bruno Roy; Aliasghar Khani; Arianna Rampini
>
> **摘要:** Learning natural body motion remains challenging due to the strong coupling between spatial geometry and temporal dynamics. Embedding motion in phase manifolds, latent spaces that capture local periodicity, has proven effective for motion prediction; however, existing approaches lack scalability and remain confined to specific settings. We introduce FunPhase, a functional periodic autoencoder that learns a phase manifold for motion and replaces discrete temporal decoding with a function-space formulation, enabling smooth trajectories that can be sampled at arbitrary temporal resolutions. FunPhase supports downstream tasks such as super-resolution and partial-body motion completion, generalizes across skeletons and datasets, and unifies motion prediction and generation within a single interpretable manifold. Our model achieves substantially lower reconstruction error than prior periodic autoencoder baselines while enabling a broader range of applications and performing on par with state-of-the-art motion generation methods.
>
---
#### [new 055] Hands-on Evaluation of Visual Transformers for Object Recognition and Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉任务，旨在比较Vision Transformers与CNN在物体识别、检测及医学图像分类中的性能。通过在ImageNet、COCO和ChestX-ray14数据集上的实验，评估不同ViT模型的准确性与计算效率，探讨其在全局上下文理解中的优势。**

- **链接: [https://arxiv.org/pdf/2512.09579v1](https://arxiv.org/pdf/2512.09579v1)**

> **作者:** Dimitrios N. Vlachogiannis; Dimitrios A. Koutsomitropoulos
>
> **摘要:** Convolutional Neural Networks (CNNs) for computer vision sometimes struggle with understanding images in a global context, as they mainly focus on local patterns. On the other hand, Vision Transformers (ViTs), inspired by models originally created for language processing, use self-attention mechanisms, which allow them to understand relationships across the entire image. In this paper, we compare different types of ViTs (pure, hierarchical, and hybrid) against traditional CNN models across various tasks, including object recognition, detection, and medical image classification. We conduct thorough tests on standard datasets like ImageNet for image classification and COCO for object detection. Additionally, we apply these models to medical imaging using the ChestX-ray14 dataset. We find that hybrid and hierarchical transformers, especially Swin and CvT, offer a strong balance between accuracy and computational resources. Furthermore, by experimenting with data augmentation techniques on medical images, we discover significant performance improvements, particularly with the Swin Transformer model. Overall, our results indicate that Vision Transformers are competitive and, in many cases, outperform traditional CNNs, especially in scenarios requiring the understanding of global visual contexts like medical imaging.
>
---
#### [new 056] Food Image Generation on Multi-Noun Categories
- **分类: cs.CV**

- **简介: 该论文属食品图像生成任务，旨在解决多名词食物类别（如“蛋面”）因语义误解导致的生成错误。作者提出FoCULR方法，融入食物领域知识并优化布局生成，提升多名词食物图像的真实性和准确性。**

- **链接: [https://arxiv.org/pdf/2512.09095v1](https://arxiv.org/pdf/2512.09095v1)**

> **作者:** Xinyue Pan; Yuhao Chen; Jiangpeng He; Fengqing Zhu
>
> **备注:** Accepted by WACV 2026
>
> **摘要:** Generating realistic food images for categories with multiple nouns is surprisingly challenging. For instance, the prompt "egg noodle" may result in images that incorrectly contain both eggs and noodles as separate entities. Multi-noun food categories are common in real-world datasets and account for a large portion of entries in benchmarks such as UEC-256. These compound names often cause generative models to misinterpret the semantics, producing unintended ingredients or objects. This is due to insufficient multi-noun category related knowledge in the text encoder and misinterpretation of multi-noun relationships, leading to incorrect spatial layouts. To overcome these challenges, we propose FoCULR (Food Category Understanding and Layout Refinement) which incorporates food domain knowledge and introduces core concepts early in the generation process. Experimental results demonstrate that the integration of these techniques improves image generation performance in the food domain.
>
---
#### [new 057] Rethinking Chain-of-Thought Reasoning for Videos
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究视频推理任务，旨在解决现有模型依赖长推理链和大量视觉token导致效率低的问题。作者提出一种高效后训练与推理框架，压缩视觉token并生成简短推理链，在减少计算的同时保持性能，无需人工标注或监督微调。**

- **链接: [https://arxiv.org/pdf/2512.09616v1](https://arxiv.org/pdf/2512.09616v1)**

> **作者:** Yiwu Zhong; Zi-Yuan Hu; Yin Li; Liwei Wang
>
> **备注:** Technical report
>
> **摘要:** Chain-of-thought (CoT) reasoning has been highly successful in solving complex tasks in natural language processing, and recent multimodal large language models (MLLMs) have extended this paradigm to video reasoning. However, these models typically build on lengthy reasoning chains and large numbers of input visual tokens. Motivated by empirical observations from our benchmark study, we hypothesize that concise reasoning combined with a reduced set of visual tokens can be sufficient for effective video reasoning. To evaluate this hypothesis, we design and validate an efficient post-training and inference framework that enhances a video MLLM's reasoning capability. Our framework enables models to operate on compressed visual tokens and generate brief reasoning traces prior to answering. The resulting models achieve substantially improved inference efficiency, deliver competitive performance across diverse benchmarks, and avoid reliance on manual CoT annotations or supervised fine-tuning. Collectively, our results suggest that long, human-like CoT reasoning may not be necessary for general video reasoning, and that concise reasoning can be both effective and efficient. Our code will be released at https://github.com/LaVi-Lab/Rethink_CoT_Video.
>
---
#### [new 058] Content-Adaptive Image Retouching Guided by Attribute-Based Text Representation
- **分类: cs.CV**

- **简介: 该论文研究图像润饰任务，旨在解决传统方法忽略内容差异导致的非自适应问题。提出内容自适应曲线映射与基于属性文本的风格引导，实现兼顾局部内容和用户偏好的高质量图像润饰。**

- **链接: [https://arxiv.org/pdf/2512.09580v1](https://arxiv.org/pdf/2512.09580v1)**

> **作者:** Hancheng Zhu; Xinyu Liu; Rui Yao; Kunyang Sun; Leida Li; Abdulmotaleb El Saddik
>
> **摘要:** Image retouching has received significant attention due to its ability to achieve high-quality visual content. Existing approaches mainly rely on uniform pixel-wise color mapping across entire images, neglecting the inherent color variations induced by image content. This limitation hinders existing approaches from achieving adaptive retouching that accommodates both diverse color distributions and user-defined style preferences. To address these challenges, we propose a novel Content-Adaptive image retouching method guided by Attribute-based Text Representation (CA-ATP). Specifically, we propose a content-adaptive curve mapping module, which leverages a series of basis curves to establish multiple color mapping relationships and learns the corresponding weight maps, enabling content-aware color adjustments. The proposed module can capture color diversity within the image content, allowing similar color values to receive distinct transformations based on their spatial context. In addition, we propose an attribute text prediction module that generates text representations from multiple image attributes, which explicitly represent user-defined style preferences. These attribute-based text representations are subsequently integrated with visual features via a multimodal model, providing user-friendly guidance for image retouching. Extensive experiments on several public datasets demonstrate that our method achieves state-of-the-art performance.
>
---
#### [new 059] View-on-Graph: Zero-shot 3D Visual Grounding via Vision-Language Reasoning on Scene Graphs
- **分类: cs.CV**

- **简介: 该论文研究零样本3D视觉定位任务，旨在解决现有方法因融合三维空间信息导致视觉语言模型处理混乱的问题。提出View-on-Graph方法，将场景建模为多模态图结构，使模型可逐步推理并增强可解释性。**

- **链接: [https://arxiv.org/pdf/2512.09215v1](https://arxiv.org/pdf/2512.09215v1)**

> **作者:** Yuanyuan Liu; Haiyang Mei; Dongyang Zhan; Jiayue Zhao; Dongsheng Zhou; Bo Dong; Xin Yang
>
> **摘要:** 3D visual grounding (3DVG) identifies objects in 3D scenes from language descriptions. Existing zero-shot approaches leverage 2D vision-language models (VLMs) by converting 3D spatial information (SI) into forms amenable to VLM processing, typically as composite inputs such as specified view renderings or video sequences with overlaid object markers. However, this VLM + SI paradigm yields entangled visual representations that compel the VLM to process entire cluttered cues, making it hard to exploit spatial semantic relationships effectively. In this work, we propose a new VLM x SI paradigm that externalizes the 3D SI into a form enabling the VLM to incrementally retrieve only what it needs during reasoning. We instantiate this paradigm with a novel View-on-Graph (VoG) method, which organizes the scene into a multi-modal, multi-layer scene graph and allows the VLM to operate as an active agent that selectively accesses necessary cues as it traverses the scene. This design offers two intrinsic advantages: (i) by structuring 3D context into a spatially and semantically coherent scene graph rather than confounding the VLM with densely entangled visual inputs, it lowers the VLM's reasoning difficulty; and (ii) by actively exploring and reasoning over the scene graph, it naturally produces transparent, step-by-step traces for interpretable 3DVG. Extensive experiments show that VoG achieves state-of-the-art zero-shot performance, establishing structured scene exploration as a promising strategy for advancing zero-shot 3DVG.
>
---
#### [new 060] Color encoding in Latent Space of Stable Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究Stable Diffusion模型中颜色的隐含编码机制。旨在揭示颜色等感知属性如何在潜空间中表示。通过合成数据、PCA和相似性分析，发现颜色由c₃和c₄通道以圆形对立轴编码，而亮度和形状由c₁和c₂编码，表明潜空间具有可解释的高效编码结构。**

- **链接: [https://arxiv.org/pdf/2512.09477v1](https://arxiv.org/pdf/2512.09477v1)**

> **作者:** Guillem Arias; Ariadna Solà; Martí Armengod; Maria Vanrell
>
> **备注:** 6 pages, 8 figures, Color Imaging Conference 33
>
> **摘要:** Recent advances in diffusion-based generative models have achieved remarkable visual fidelity, yet a detailed understanding of how specific perceptual attributes - such as color and shape - are internally represented remains limited. This work explores how color is encoded in a generative model through a systematic analysis of the latent representations in Stable Diffusion. Through controlled synthetic datasets, principal component analysis (PCA) and similarity metrics, we reveal that color information is encoded along circular, opponent axes predominantly captured in latent channels c_3 and c_4, whereas intensity and shape are primarily represented in channels c_1 and c_2. Our findings indicate that the latent space of Stable Diffusion exhibits an interpretable structure aligned with a efficient coding representation. These insights provide a foundation for future work in model understanding, editing applications, and the design of more disentangled generative frameworks.
>
---
#### [new 061] Stylized Meta-Album: Group-bias injection with style transfer to study robustness against distribution shifts
- **分类: cs.CV**

- **简介: 该论文属于图像分类与鲁棒性研究任务，旨在解决现有基准在群体多样性与分布偏移评估上的局限。作者构建了Stylized Meta-Album（SMA）元数据集，通过风格迁移引入群体偏差，支持灵活配置群组、类别和域，用于更全面地评估模型在公平性、OOD泛化和域适应中的表现。**

- **链接: [https://arxiv.org/pdf/2512.09773v1](https://arxiv.org/pdf/2512.09773v1)**

> **作者:** Romain Mussard; Aurélien Gauffre; Ihsan Ullah; Thanh Gia Hieu Khuong; Massih-Reza Amini; Isabelle Guyon; Lisheng Sun-Hosoya
>
> **摘要:** We introduce Stylized Meta-Album (SMA), a new image classification meta-dataset comprising 24 datasets (12 content datasets, and 12 stylized datasets), designed to advance studies on out-of-distribution (OOD) generalization and related topics. Created using style transfer techniques from 12 subject classification datasets, SMA provides a diverse and extensive set of 4800 groups, combining various subjects (objects, plants, animals, human actions, textures) with multiple styles. SMA enables flexible control over groups and classes, allowing us to configure datasets to reflect diverse benchmark scenarios. While ideally, data collection would capture extensive group diversity, practical constraints often make this infeasible. SMA addresses this by enabling large and configurable group structures through flexible control over styles, subject classes, and domains-allowing datasets to reflect a wide range of real-world benchmark scenarios. This design not only expands group and class diversity, but also opens new methodological directions for evaluating model performance across diverse group and domain configurations-including scenarios with many minority groups, varying group imbalance, and complex domain shifts-and for studying fairness, robustness, and adaptation under a broader range of realistic conditions. To demonstrate SMA's effectiveness, we implemented two benchmarks: (1) a novel OOD generalization and group fairness benchmark leveraging SMA's domain, class, and group diversity to evaluate existing benchmarks. Our findings reveal that while simple balancing and algorithms utilizing group information remain competitive as claimed in previous benchmarks, increasing group diversity significantly impacts fairness, altering the superiority and relative rankings of algorithms. We also propose to use \textit{Top-M worst group accuracy} as a new hyperparameter tuning metric, demonstrating broader fairness during optimization and delivering better final worst-group accuracy for larger group diversity. (2) An unsupervised domain adaptation (UDA) benchmark utilizing SMA's group diversity to evaluate UDA algorithms across more scenarios, offering a more comprehensive benchmark with lower error bars (reduced by 73\% and 28\% in closed-set setting and UniDA setting, respectively) compared to existing efforts. These use cases highlight SMA's potential to significantly impact the outcomes of conventional benchmarks.
>
---
#### [new 062] A Survey of Body and Face Motion: Datasets, Performance Evaluation Metrics and Generative Techniques
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于综述任务，旨在解决生成自然、连贯的面部与身体动作难题。作者系统回顾了相关数据集、评估指标和生成技术，涵盖多模态输入下的动作生成方法，并提出未来研究方向，提升双人交互场景中虚拟角色的表现力与真实感。**

- **链接: [https://arxiv.org/pdf/2512.09005v1](https://arxiv.org/pdf/2512.09005v1)**

> **作者:** Lownish Rai Sookha; Nikhil Pakhale; Mudasir Ganaie; Abhinav Dhall
>
> **摘要:** Body and face motion play an integral role in communication. They convey crucial information on the participants. Advances in generative modeling and multi-modal learning have enabled motion generation from signals such as speech, conversational context and visual cues. However, generating expressive and coherent face and body dynamics remains challenging due to the complex interplay of verbal / non-verbal cues and individual personality traits. This survey reviews body and face motion generation, covering core concepts, representations techniques, generative approaches, datasets and evaluation metrics. We highlight future directions to enhance the realism, coherence and expressiveness of avatars in dyadic settings. To the best of our knowledge, this work is the first comprehensive review to cover both body and face motion. Detailed resources are listed on https://lownish23csz0010.github.io/mogen/.
>
---
#### [new 063] Detection and Localization of Subdural Hematoma Using Deep Learning on Computed Tomography
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究基于CT影像的硬膜下血肿（SDH）检测与定位任务，旨在提升自动诊断的准确性与可解释性。作者提出多模态深度学习框架，融合临床数据、3D卷积模型与Transformer分割模型，实现高性能检测与解剖一致的定位，助力临床快速决策。**

- **链接: [https://arxiv.org/pdf/2512.09393v1](https://arxiv.org/pdf/2512.09393v1)**

> **作者:** Vasiliki Stoumpou; Rohan Kumar; Bernard Burman; Diego Ojeda; Tapan Mehta; Dimitris Bertsimas
>
> **摘要:** Background. Subdural hematoma (SDH) is a common neurosurgical emergency, with increasing incidence in aging populations. Rapid and accurate identification is essential to guide timely intervention, yet existing automated tools focus primarily on detection and provide limited interpretability or spatial localization. There remains a need for transparent, high-performing systems that integrate multimodal clinical and imaging information to support real-time decision-making. Methods. We developed a multimodal deep-learning framework that integrates structured clinical variables, a 3D convolutional neural network trained on CT volumes, and a transformer-enhanced 2D segmentation model for SDH detection and localization. Using 25,315 head CT studies from Hartford HealthCare (2015--2024), of which 3,774 (14.9\%) contained clinician-confirmed SDH, tabular models were trained on demographics, comorbidities, medications, and laboratory results. Imaging models were trained to detect SDH and generate voxel-level probability maps. A greedy ensemble strategy combined complementary predictors. Findings. Clinical variables alone provided modest discriminatory power (AUC 0.75). Convolutional models trained on CT volumes and segmentation-derived maps achieved substantially higher accuracy (AUCs 0.922 and 0.926). The multimodal ensemble integrating all components achieved the best overall performance (AUC 0.9407; 95\% CI, 0.930--0.951) and produced anatomically meaningful localization maps consistent with known SDH patterns. Interpretation. This multimodal, interpretable framework provides rapid and accurate SDH detection and localization, achieving high detection performance and offering transparent, anatomically grounded outputs. Integration into radiology workflows could streamline triage, reduce time to intervention, and improve consistency in SDH management.
>
---
#### [new 064] InfoMotion: A Graph-Based Approach to Video Dataset Distillation for Echocardiography
- **分类: cs.CV**

- **简介: 该论文属医学视频数据压缩任务，旨在解决超声心动图数据规模大、计算耗本高的问题。提出InfoMotion方法，通过运动特征提取、图建模与Infomap算法选择25个具代表性的合成视频，保留原数据关键特征，验证其有效性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.09422v1](https://arxiv.org/pdf/2512.09422v1)**

> **作者:** Zhe Li; Hadrien Reynaud; Alberto Gomez; Bernhard Kainz
>
> **备注:** Accepted at MICAD 2025
>
> **摘要:** Echocardiography playing a critical role in the diagnosis and monitoring of cardiovascular diseases as a non-invasive real-time assessment of cardiac structure and function. However, the growing scale of echocardiographic video data presents significant challenges in terms of storage, computation, and model training efficiency. Dataset distillation offers a promising solution by synthesizing a compact, informative subset of data that retains the key clinical features of the original dataset. In this work, we propose a novel approach for distilling a compact synthetic echocardiographic video dataset. Our method leverages motion feature extraction to capture temporal dynamics, followed by class-wise graph construction and representative sample selection using the Infomap algorithm. This enables us to select a diverse and informative subset of synthetic videos that preserves the essential characteristics of the original dataset. We evaluate our approach on the EchoNet-Dynamic datasets and achieve a test accuracy of \(69.38\%\) using only \(25\) synthetic videos. These results demonstrate the effectiveness and scalability of our method for medical video dataset distillation.
>
---
#### [new 065] Enabling Next-Generation Consumer Experience with Feature Coding for Machines
- **分类: cs.CV**

- **简介: 该论文聚焦于机器特征编码（FCM）标准，旨在解决低功耗设备运行大型深度学习模型时的效率问题。通过压缩传输神经网络中间特征，实现高效远程推理，在保持精度的同时降低75.90%比特率，提升智能设备的AI应用体验。**

- **链接: [https://arxiv.org/pdf/2512.09232v1](https://arxiv.org/pdf/2512.09232v1)**

> **作者:** Md Eimran Hossain Eimon; Juan Merlos; Ashan Perera; Hari Kalva; Velibor Adzic; Borko Furht
>
> **摘要:** As consumer devices become increasingly intelligent and interconnected, efficient data transfer solutions for machine tasks have become essential. This paper presents an overview of the latest Feature Coding for Machines (FCM) standard, part of MPEG-AI and developed by the Moving Picture Experts Group (MPEG). FCM supports AI-driven applications by enabling the efficient extraction, compression, and transmission of intermediate neural network features. By offloading computationally intensive operations to base servers with high computing resources, FCM allows low-powered devices to leverage large deep learning models. Experimental results indicate that the FCM standard maintains the same level of accuracy while reducing bitrate requirements by 75.90% compared to remote inference.
>
---
#### [new 066] Consist-Retinex: One-Step Noise-Emphasized Consistency Training Accelerates High-Quality Retinex Enhancement
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究低光图像增强任务，旨在解决扩散模型需多步采样、难以实用的问题。提出Consist-Retinex框架，首次将一致性模型用于Retinex增强，通过双目标损失和噪声强调采样，实现高质量一键式生成，显著提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2512.08982v1](https://arxiv.org/pdf/2512.08982v1)**

> **作者:** Jian Xu; Wei Chen; Shigui Li; Delu Zeng; John Paisley; Qibin Zhao
>
> **摘要:** Diffusion models have achieved remarkable success in low-light image enhancement through Retinex-based decomposition, yet their requirement for hundreds of iterative sampling steps severely limits practical deployment. While recent consistency models offer promising one-step generation for \textit{unconditional synthesis}, their application to \textit{conditional enhancement} remains unexplored. We present \textbf{Consist-Retinex}, the first framework adapting consistency modeling to Retinex-based low-light enhancement. Our key insight is that conditional enhancement requires fundamentally different training dynamics than unconditional generation standard consistency training focuses on low-noise regions near the data manifold, while conditional mapping critically depends on large-noise regimes that bridge degraded inputs to enhanced outputs. We introduce two core innovations: (1) a \textbf{dual-objective consistency loss} combining temporal consistency with ground-truth alignment under randomized time sampling, providing full-spectrum supervision for stable convergence; and (2) an \textbf{adaptive noise-emphasized sampling strategy} that prioritizes training on large-noise regions essential for one-step conditional generation. On VE-LOL-L, Consist-Retinex achieves \textbf{state-of-the-art performance with single-step sampling} (\textbf{PSNR: 25.51 vs. 23.41, FID: 44.73 vs. 49.59} compared to Diff-Retinex++), while requiring only \textbf{1/8 of the training budget} relative to the 1000-step Diff-Retinex baseline.
>
---
#### [new 067] Diffusion Model Regularized Implicit Neural Representation for CT Metal Artifact Reduction
- **分类: cs.CV**

- **简介: 该论文研究CT金属伪影去除任务，旨在解决现有方法因依赖配对数据或弱先验导致的性能不稳定问题。提出结合扩散模型与隐式神经表示的新框架，融入物理约束并增强先验表达，提升去伪影效果与临床适用性。**

- **链接: [https://arxiv.org/pdf/2512.08999v1](https://arxiv.org/pdf/2512.08999v1)**

> **作者:** Jie Wen; Chenhe Du; Xiao Wang; Yuyao Zhang
>
> **摘要:** Computed tomography (CT) images are often severely corrupted by artifacts in the presence of metals. Existing supervised metal artifact reduction (MAR) approaches suffer from performance instability on known data due to their reliance on limited paired metal-clean data, which limits their clinical applicability. Moreover, existing unsupervised methods face two main challenges: 1) the CT physical geometry is not effectively incorporated into the MAR process to ensure data fidelity; 2) traditional heuristics regularization terms cannot fully capture the abundant prior knowledge available. To overcome these shortcomings, we propose diffusion model regularized implicit neural representation framework for MAR. The implicit neural representation integrates physical constraints and imposes data fidelity, while the pre-trained diffusion model provides prior knowledge to regularize the solution. Experimental results on both simulated and clinical data demonstrate the effectiveness and generalization ability of our method, highlighting its potential to be applied to clinical settings.
>
---
#### [new 068] Generative Point Cloud Registration
- **分类: cs.CV**

- **简介: 该论文研究3D点云配准任务，旨在提升跨视角匹配精度。提出生成式配准新范式，通过Match-ControlNet生成几何对齐、纹理一致的图像对，融合颜色与几何特征，增强匹配鲁棒性，并可集成到现有方法中，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.09407v1](https://arxiv.org/pdf/2512.09407v1)**

> **作者:** Haobo Jiang; Jin Xie; Jian Yang; Liang Yu; Jianmin Zheng
>
> **备注:** 14 pages, 9 figures
>
> **摘要:** In this paper, we propose a novel 3D registration paradigm, Generative Point Cloud Registration, which bridges advanced 2D generative models with 3D matching tasks to enhance registration performance. Our key idea is to generate cross-view consistent image pairs that are well-aligned with the source and target point clouds, enabling geometry-color feature fusion to facilitate robust matching. To ensure high-quality matching, the generated image pair should feature both 2D-3D geometric consistency and cross-view texture consistency. To achieve this, we introduce Match-ControlNet, a matching-specific, controllable 2D generative model. Specifically, it leverages the depth-conditioned generation capability of ControlNet to produce images that are geometrically aligned with depth maps derived from point clouds, ensuring 2D-3D geometric consistency. Additionally, by incorporating a coupled conditional denoising scheme and coupled prompt guidance, Match-ControlNet further promotes cross-view feature interaction, guiding texture consistency generation. Our generative 3D registration paradigm is general and could be seamlessly integrated into various registration methods to enhance their performance. Extensive experiments on 3DMatch and ScanNet datasets verify the effectiveness of our approach.
>
---
#### [new 069] GTAvatar: Bridging Gaussian Splatting and Texture Mapping for Relightable and Editable Gaussian Avatars
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出GTAvatar，旨在解决高斯点阵头像缺乏编辑性的问题。通过将高斯基元嵌入UV空间，结合纹理映射实现可编辑、可重光照的高质量头像重建，支持直观外观与几何编辑。**

- **链接: [https://arxiv.org/pdf/2512.09162v1](https://arxiv.org/pdf/2512.09162v1)**

> **作者:** Kelian Baert; Mae Younes; Francois Bourel; Marc Christie; Adnane Boukhayma
>
> **摘要:** Recent advancements in Gaussian Splatting have enabled increasingly accurate reconstruction of photorealistic head avatars, opening the door to numerous applications in visual effects, videoconferencing, and virtual reality. This, however, comes with the lack of intuitive editability offered by traditional triangle mesh-based methods. In contrast, we propose a method that combines the accuracy and fidelity of 2D Gaussian Splatting with the intuitiveness of UV texture mapping. By embedding each canonical Gaussian primitive's local frame into a patch in the UV space of a template mesh in a computationally efficient manner, we reconstruct continuous editable material head textures from a single monocular video on a conventional UV domain. Furthermore, we leverage an efficient physically based reflectance model to enable relighting and editing of these intrinsic material maps. Through extensive comparisons with state-of-the-art methods, we demonstrate the accuracy of our reconstructions, the quality of our relighting results, and the ability to provide intuitive controls for modifying an avatar's appearance and geometry via texture mapping without additional optimization.
>
---
#### [new 070] Building Reasonable Inference for Vision-Language Models in Blind Image Quality Assessment
- **分类: cs.CV**

- **简介: 该论文针对视觉语言模型在盲图像质量评估中推理不一致和不稳定的问题，提出一种两阶段微调方法，分离视觉感知与质量推断，提升推理合理性与稳定性，在多个数据集上显著改善性能。**

- **链接: [https://arxiv.org/pdf/2512.09555v1](https://arxiv.org/pdf/2512.09555v1)**

> **作者:** Yuan Li; Zitang Sun; Yen-ju Chen; Shin'ya Nishida
>
> **备注:** Accepted to the ICONIP (International Conference on Neural Information Processing), 2025
>
> **摘要:** Recent progress in BIQA has been driven by VLMs, whose semantic reasoning abilities suggest that they might extract visual features, generate descriptive text, and infer quality in a human-like manner. However, these models often produce textual descriptions that contradict their final quality predictions, and the predicted scores can change unstably during inference - behaviors not aligned with human reasoning. To understand these issues, we analyze the factors that cause contradictory assessments and instability. We first estimate the relationship between the final quality predictions and the generated visual features, finding that the predictions are not fully grounded in the features and that the logical connection between them is weak. Moreover, decoding intermediate VLM layers shows that the model frequently relies on a limited set of candidate tokens, which contributes to prediction instability. To encourage more human-like reasoning, we introduce a two-stage tuning method that explicitly separates visual perception from quality inference. In the first stage, the model learns visual features; in the second, it infers quality solely from these features. Experiments on SPAQ and KONIQ demonstrate that our approach reduces prediction instability from 22.00% to 12.39% and achieves average gains of 0.3124/0.3507 in SRCC/PLCC across LIVE, CSIQ, SPAQ, and KONIQ compared to the baseline. Further analyses show that our method improves both stability and the reliability of the inference process.
>
---
#### [new 071] Learning Patient-Specific Disease Dynamics with Latent Flow Matching for Longitudinal Imaging Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦疾病进展建模，旨在生成纵向医学图像并提升可解释性。针对现有方法在连续性、语义结构和患者特异性对齐上的不足，提出Δ-LFM框架，结合流匹配与患者特异性潜在空间对齐，实现更一致、语义清晰的疾病动态建模。**

- **链接: [https://arxiv.org/pdf/2512.09185v1](https://arxiv.org/pdf/2512.09185v1)**

> **作者:** Hao Chen; Rui Yin; Yifan Chen; Qi Chen; Chao Li
>
> **备注:** Under review
>
> **摘要:** Understanding disease progression is a central clinical challenge with direct implications for early diagnosis and personalized treatment. While recent generative approaches have attempted to model progression, key mismatches remain: disease dynamics are inherently continuous and monotonic, yet latent representations are often scattered, lacking semantic structure, and diffusion-based models disrupt continuity with random denoising process. In this work, we propose to treat the disease dynamic as a velocity field and leverage Flow Matching (FM) to align the temporal evolution of patient data. Unlike prior methods, it captures the intrinsic dynamic of disease, making the progression more interpretable. However, a key challenge remains: in latent space, Auto-Encoders (AEs) do not guarantee alignment across patients or correlation with clinical-severity indicators (e.g., age and disease conditions). To address this, we propose to learn patient-specific latent alignment, which enforces patient trajectories to lie along a specific axis, with magnitude increasing monotonically with disease severity. This leads to a consistent and semantically meaningful latent space. Together, we present $Δ$-LFM, a framework for modeling patient-specific latent progression with flow matching. Across three longitudinal MRI benchmarks, $Δ$-LFM demonstrates strong empirical performance and, more importantly, offers a new framework for interpreting and visualizing disease dynamics.
>
---
#### [new 072] Training Multi-Image Vision Agents via End2End Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究多图像视觉任务，旨在解决现有开源视觉语言模型（VLM）局限于单图输入、难以应对复杂多图问答的问题。作者提出IMAgent，通过端到端强化学习训练，结合多智能体生成数据与专用视觉工具，实现稳定高效的多图推理。**

- **链接: [https://arxiv.org/pdf/2512.08980v1](https://arxiv.org/pdf/2512.08980v1)**

> **作者:** Chengqi Dong; Chuhuai Yue; Hang He; Rongge Mao; Fenghe Tang; S Kevin Zhou; Zekun Xu; Xiaohan Wang; Jiajun Chai; Wei Lin; Guojun Yin
>
> **摘要:** Recent VLM-based agents aim to replicate OpenAI O3's ``thinking with images" via tool use, but most open-source methods limit input to a single image, falling short on real-world multi-image QA tasks. To address this, we propose IMAgent, an open-source vision agent trained via end-to-end reinforcement learning dedicated for complex multi-image tasks. By leveraging a multi-agent system, we generate challenging and visually-rich multi-image QA pairs to fully activate the tool-use potential of the base VLM. Through manual verification, we obtain MIFG-QA, comprising 10k samples for training and evaluation. With deeper reasoning steps, VLMs may increasingly ignore visual inputs. We therefore develop two specialized tools for visual reflection and confirmation, allowing the model to proactively reallocate its attention to image content during inference. Benefiting from our well-designed action-trajectory two-level mask strategy, IMAgent achieves stable tool use behavior via pure RL training without requiring costly supervised fine-tuning data. Extensive experiments demonstrate that IMAgent maintains strong performance on existing single-image benchmarks while achieving substantial improvements on our proposed multi-image dataset, with our analysis providing actionable insights for the research community. Codes and data will be released soon.
>
---
#### [new 073] Masked Registration and Autoencoding of CT Images for Predictive Tibia Reconstruction
- **分类: cs.CV**

- **简介: 该论文针对复杂胫骨骨折的术前规划难题，提出结合掩码鲁棒的神经配准与自编码模型，从骨折CT中预测患者特异性的健康骨骼结构。实现了CT图像的标准化配准与重建，用于生成完整的胫骨形态。**

- **链接: [https://arxiv.org/pdf/2512.09525v1](https://arxiv.org/pdf/2512.09525v1)**

> **作者:** Hongyou Zhou; Cederic Aßmann; Alaa Bejaoui; Heiko Tzschätzsch; Mark Heyland; Julian Zierke; Niklas Tuttle; Sebastian Hölzl; Timo Auer; David A. Back; Marc Toussaint
>
> **备注:** DGM4MICCAI
>
> **摘要:** Surgical planning for complex tibial fractures can be challenging for surgeons, as the 3D structure of the later desirable bone alignment may be diffi- cult to imagine. To assist in such planning, we address the challenge of predicting a patient-specific reconstruction target from a CT of the fractured tibia. Our ap- proach combines neural registration and autoencoder models. Specifically, we first train a modified spatial transformer network (STN) to register a raw CT to a standardized coordinate system of a jointly trained tibia prototype. Subsequently, various autoencoder (AE) architectures are trained to model healthy tibial varia- tions. Both the STN and AE models are further designed to be robust to masked input, allowing us to apply them to fractured CTs and decode to a prediction of the patient-specific healthy bone in standard coordinates. Our contributions include: i) a 3D-adapted STN for global spatial registration, ii) a comparative analysis of AEs for bone CT modeling, and iii) the extension of both to handle masked inputs for predictive generation of healthy bone structures. Project page: https://github.com/HongyouZhou/repair
>
---
#### [new 074] Integrated Pipeline for Coronary Angiography With Automated Lesion Profiling, Virtual Stenting, and 100-Vessel FFR Validation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出AngioAI-QFR，旨在解决冠状动脉造影中解剖与功能评估分离的问题。通过深度学习实现自动病变检测、血流功能分析（QFR）、虚拟支架植入及效果预测，构建端到端一体化流程，提升临床决策效率与精度。**

- **链接: [https://arxiv.org/pdf/2512.09134v1](https://arxiv.org/pdf/2512.09134v1)**

> **作者:** Georgy Kopanitsa; Oleg Metsker; Alexey Yakovlev
>
> **备注:** 22 pages, 10 figures, 7 tables
>
> **摘要:** Coronary angiography is the main tool for assessing coronary artery disease, but visual grading of stenosis is variable and only moderately related to ischaemia. Wire based fractional flow reserve (FFR) improves lesion selection but is not used systematically. Angiography derived indices such as quantitative flow ratio (QFR) offer wire free physiology, yet many tools are workflow intensive and separate from automated anatomy analysis and virtual PCI planning. We developed AngioAI-QFR, an end to end angiography only pipeline combining deep learning stenosis detection, lumen segmentation, centreline and diameter extraction, per millimetre Relative Flow Capacity profiling, and virtual stenting with automatic recomputation of angiography derived QFR. The system was evaluated in 100 consecutive vessels with invasive FFR as reference. Primary endpoints were agreement with FFR (correlation, mean absolute error) and diagnostic performance for FFR <= 0.80. On held out frames, stenosis detection achieved precision 0.97 and lumen segmentation Dice 0.78. Across 100 vessels, AngioAI-QFR correlated strongly with FFR (r = 0.89, MAE 0.045). The AUC for detecting FFR <= 0.80 was 0.93, with sensitivity 0.88 and specificity 0.86. The pipeline completed fully automatically in 93 percent of vessels, with median time to result 41 s. RFC profiling distinguished focal from diffuse capacity loss, and virtual stenting predicted larger QFR gain in focal than in diffuse disease. AngioAI-QFR provides a practical, near real time pipeline that unifies computer vision, functional profiling, and virtual PCI with automated angiography derived physiology.
>
---
#### [new 075] IF-Bench: Benchmarking and Enhancing MLLMs for Infrared Images with Generative Visual Prompting
- **分类: cs.CV**

- **简介: 该论文聚焦多模态大模型对红外图像的理解任务，旨在解决现有模型在此领域评估缺失的问题。作者构建了首个红外图像基准IF-Bench，并提出无需训练的生成式视觉提示方法GenViP，通过图像编辑提升模型表现，显著改善红外图像理解效果。**

- **链接: [https://arxiv.org/pdf/2512.09663v1](https://arxiv.org/pdf/2512.09663v1)**

> **作者:** Tao Zhang; Yuyang Hong; Yang Xia; Kun Ding; Zeyu Zhang; Ying Wang; Shiming Xiang; Chunhong Pan
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have led to impressive progress across various benchmarks. However, their capability in understanding infrared images remains unexplored. To address this gap, we introduce IF-Bench, the first high-quality benchmark designed for evaluating multimodal understanding of infrared images. IF-Bench consists of 499 images sourced from 23 infrared datasets and 680 carefully curated visual question-answer pairs, covering 10 essential dimensions of image understanding. Based on this benchmark, we systematically evaluate over 40 open-source and closed-source MLLMs, employing cyclic evaluation, bilingual assessment, and hybrid judgment strategies to enhance the reliability of the results. Our analysis reveals how model scale, architecture, and inference paradigms affect infrared image comprehension, providing valuable insights for this area. Furthermore, we propose a training-free generative visual prompting (GenViP) method, which leverages advanced image editing models to translate infrared images into semantically and spatially aligned RGB counterparts, thereby mitigating domain distribution shifts. Extensive experiments demonstrate that our method consistently yields significant performance improvements across a wide range of MLLMs. The benchmark and code are available at https://github.com/casiatao/IF-Bench.
>
---
#### [new 076] CHEM: Estimating and Understanding Hallucinations in Deep Learning for Image Processing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像处理任务，旨在解决深度学习模型在图像重建中产生幻觉伪影的问题。作者提出CHEM方法，结合小波与剪切波表示和保形分位数回归，量化并理解幻觉，提升模型可信度，并从理论上分析U型网络易产生幻觉的原因。**

- **链接: [https://arxiv.org/pdf/2512.09806v1](https://arxiv.org/pdf/2512.09806v1)**

> **作者:** Jianfei Li; Ines Rosellon-Inclan; Gitta Kutyniok; Jean-Luc Starck
>
> **摘要:** U-Net and other U-shaped architectures have achieved significant success in image deconvolution tasks. However, challenges have emerged, as these methods might generate unrealistic artifacts or hallucinations, which can interfere with analysis in safety-critical scenarios. This paper introduces a novel approach for quantifying and comprehending hallucination artifacts to ensure trustworthy computer vision models. Our method, termed the Conformal Hallucination Estimation Metric (CHEM), is applicable to any image reconstruction model, enabling efficient identification and quantification of hallucination artifacts. It offers two key advantages: it leverages wavelet and shearlet representations to efficiently extract hallucinations of image features and uses conformalized quantile regression to assess hallucination levels in a distribution-free manner. Furthermore, from an approximation theoretical perspective, we explore the reasons why U-shaped networks are prone to hallucinations. We test the proposed approach on the CANDELS astronomical image dataset with models such as U-Net, SwinUNet, and Learnlets, and provide new perspectives on hallucination from different aspects in deep learning-based image processing.
>
---
#### [new 077] Towards Lossless Ultimate Vision Token Compression for VLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型效率优化任务，旨在解决高分辨率视觉token冗余导致的计算低效与延迟问题。作者提出LUVC框架，通过视觉编码器的迭代合并和LLM中的频谱剪枝，实现无损、训练-free的视觉token压缩，显著加速推理且几乎不损失精度。**

- **链接: [https://arxiv.org/pdf/2512.09010v1](https://arxiv.org/pdf/2512.09010v1)**

> **作者:** Dehua Zheng; Mouxiao Huang; Borui Jiang; Hailin Hu; Xinghao Chen
>
> **摘要:** Visual language models encounter challenges in computational efficiency and latency, primarily due to the substantial redundancy in the token representations of high-resolution images and videos. Current attention/similarity-based compression algorithms suffer from either position bias or class imbalance, leading to significant accuracy degradation. They also fail to generalize to shallow LLM layers, which exhibit weaker cross-modal interactions. To address this, we extend token compression to the visual encoder through an effective iterative merging scheme that is orthogonal in spatial axes to accelerate the computation across the entire VLM. Furthermoer, we integrate a spectrum pruning unit into LLM through an attention/similarity-free low-pass filter, which gradually prunes redundant visual tokens and is fully compatible to modern FlashAttention. On this basis, we propose Lossless Ultimate Vision tokens Compression (LUVC) framework. LUVC systematically compresses visual tokens until complete elimination at the final layer of LLM, so that the high-dimensional visual features are gradually fused into the multimodal queries. The experiments show that LUVC achieves a 2 speedup inference in language model with negligible accuracy degradation, and the training-free characteristic enables immediate deployment across multiple VLMs.
>
---
#### [new 078] StateSpace-SSL: Linear-Time Self-supervised Learning for Plant Disease Detectio
- **分类: cs.CV**

- **简介: 该论文针对植物病害检测中自监督学习对长距离病斑连续性建模不足的问题，提出StateSpace-SSL框架。其采用视觉Mamba编码器实现线性时间建模，通过方向扫描捕捉叶面病变连续性，并结合原型引导的师生机制提升特征稳定性，在多个数据集上优于CNN和Transformer基线方法。**

- **链接: [https://arxiv.org/pdf/2512.09492v1](https://arxiv.org/pdf/2512.09492v1)**

> **作者:** Abdullah Al Mamun; Miaohua Zhang; David Ahmedt-Aristizabal; Zeeshan Hayder; Mohammad Awrangjeb
>
> **备注:** Accepted to AAAI workshop (AgriAI 2026)
>
> **摘要:** Self-supervised learning (SSL) is attractive for plant disease detection as it can exploit large collections of unlabeled leaf images, yet most existing SSL methods are built on CNNs or vision transformers that are poorly matched to agricultural imagery. CNN-based SSL struggles to capture disease patterns that evolve continuously along leaf structures, while transformer-based SSL introduces quadratic attention cost from high-resolution patches. To address these limitations, we propose StateSpace-SSL, a linear-time SSL framework that employs a Vision Mamba state-space encoder to model long-range lesion continuity through directional scanning across the leaf surface. A prototype-driven teacher-student objective aligns representations across multiple views, encouraging stable and lesion-aware features from labelled data. Experiments on three publicly available plant disease datasets show that StateSpace-SSL consistently outperforms the CNN- and transformer-based SSL baselines in various evaluation metrics. Qualitative analyses further confirm that it learns compact, lesion-focused feature maps, highlighting the advantage of linear state-space modelling for self-supervised plant disease representation learning.
>
---
#### [new 079] LongT2IBench: A Benchmark for Evaluating Long Text-to-Image Generation with Graph-structured Annotations
- **分类: cs.CV**

- **简介: 该论文针对长文本到图像生成的评估难题，提出LongT2IBench基准，包含14K图文对及图结构标注，并设计生成-精炼-验证协议实现细粒度对齐标注。同时提出LongT2IExpert评估模型，结合多模态大模型与分层思维链，实现可解释的量化评估。**

- **链接: [https://arxiv.org/pdf/2512.09271v1](https://arxiv.org/pdf/2512.09271v1)**

> **作者:** Zhichao Yang; Tianjiao Gu; Jianjie Wang; Feiyu Lin; Xiangfei Sheng; Pengfei Chen; Leida Li
>
> **备注:** The paper has been accepted by AAAI 2026
>
> **摘要:** The increasing popularity of long Text-to-Image (T2I) generation has created an urgent need for automatic and interpretable models that can evaluate the image-text alignment in long prompt scenarios. However, the existing T2I alignment benchmarks predominantly focus on short prompt scenarios and only provide MOS or Likert scale annotations. This inherent limitation hinders the development of long T2I evaluators, particularly in terms of the interpretability of alignment. In this study, we contribute LongT2IBench, which comprises 14K long text-image pairs accompanied by graph-structured human annotations. Given the detail-intensive nature of long prompts, we first design a Generate-Refine-Qualify annotation protocol to convert them into textual graph structures that encompass entities, attributes, and relations. Through this transformation, fine-grained alignment annotations are achieved based on these granular elements. Finally, the graph-structed annotations are converted into alignment scores and interpretations to facilitate the design of T2I evaluation models. Based on LongT2IBench, we further propose LongT2IExpert, a LongT2I evaluator that enables multi-modal large language models (MLLMs) to provide both quantitative scores and structured interpretations through an instruction-tuning process with Hierarchical Alignment Chain-of-Thought (CoT). Extensive experiments and comparisons demonstrate the superiority of the proposed LongT2IExpert in alignment evaluation and interpretation. Data and code have been released in https://welldky.github.io/LongT2IBench-Homepage/.
>
---
#### [new 080] 3DID: Direct 3D Inverse Design for Aerodynamics with Physics-Aware Optimization
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究3D空气动力学逆向设计任务，旨在解决现有方法因2D投影或形状微调导致的细节损失与设计受限问题。提出3DID框架，通过物理-几何联合嵌入和分阶段物理感知优化，直接在连续隐空间生成高保真3D几何形状。**

- **链接: [https://arxiv.org/pdf/2512.08987v1](https://arxiv.org/pdf/2512.08987v1)**

> **作者:** Yuze Hao; Linchao Zhu; Yi Yang
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Inverse design aims to design the input variables of a physical system to optimize a specified objective function, typically formulated as a search or optimization problem. However, in 3D domains, the design space grows exponentially, rendering exhaustive grid-based searches infeasible. Recent advances in deep learning have accelerated inverse design by providing powerful generative priors and differentiable surrogate models. Nevertheless, current methods tend to approximate the 3D design space using 2D projections or fine-tune existing 3D shapes. These approaches sacrifice volumetric detail and constrain design exploration, preventing true 3D design from scratch. In this paper, we propose a 3D Inverse Design (3DID) framework that directly navigates the 3D design space by coupling a continuous latent representation with a physics-aware optimization strategy. We first learn a unified physics-geometry embedding that compactly captures shape and physical field data in a continuous latent space. Then, we introduce a two-stage strategy to perform physics-aware optimization. In the first stage, a gradient-guided diffusion sampler explores the global latent manifold. In the second stage, an objective-driven, topology-preserving refinement further sculpts each candidate toward the target objective. This enables 3DID to generate high-fidelity 3D geometries, outperforming existing methods in both solution quality and design versatility.
>
---
#### [new 081] Benchmarking Real-World Medical Image Classification with Noisy Labels: Challenges, Practice, and Outlook
- **分类: cs.CV**

- **简介: 该论文聚焦医学图像分类中的标签噪声问题，旨在评估现有去噪方法在真实场景下的鲁棒性。作者构建了包含10种方法、7个数据集和多种噪声模式的基准LNMBench，并提出改进策略，提升模型在高噪声与类别不平衡下的表现。**

- **链接: [https://arxiv.org/pdf/2512.09315v1](https://arxiv.org/pdf/2512.09315v1)**

> **作者:** Yuan Ma; Junlin Hou; Chao Zhang; Yukun Zhou; Zongyuan Ge; Haoran Xie; Lie Ju
>
> **摘要:** Learning from noisy labels remains a major challenge in medical image analysis, where annotation demands expert knowledge and substantial inter-observer variability often leads to inconsistent or erroneous labels. Despite extensive research on learning with noisy labels (LNL), the robustness of existing methods in medical imaging has not been systematically assessed. To address this gap, we introduce LNMBench, a comprehensive benchmark for Label Noise in Medical imaging. LNMBench encompasses \textbf{10} representative methods evaluated across 7 datasets, 6 imaging modalities, and 3 noise patterns, establishing a unified and reproducible framework for robustness evaluation under realistic conditions. Comprehensive experiments reveal that the performance of existing LNL methods degrades substantially under high and real-world noise, highlighting the persistent challenges of class imbalance and domain variability in medical data. Motivated by these findings, we further propose a simple yet effective improvement to enhance model robustness under such conditions. The LNMBench codebase is publicly released to facilitate standardized evaluation, promote reproducible research, and provide practical insights for developing noise-resilient algorithms in both research and real-world medical applications.The codebase is publicly available on https://github.com/myyy777/LNMBench.
>
---
#### [new 082] VHOI: Controllable Video Generation of Human-Object Interactions from Sparse Trajectories via Motion Densification
- **分类: cs.CV**

- **简介: 该论文研究可控的人-物交互视频生成，旨在解决稀疏控制信号缺乏细节、稠密信号获取成本高的问题。提出VHOI框架，通过运动稠密化将稀疏轨迹转为HOI掩码序列，再微调视频扩散模型，实现高质量、可控的交互视频生成。**

- **链接: [https://arxiv.org/pdf/2512.09646v1](https://arxiv.org/pdf/2512.09646v1)**

> **作者:** Wanyue Zhang; Lin Geng Foo; Thabo Beeler; Rishabh Dabral; Christian Theobalt
>
> **摘要:** Synthesizing realistic human-object interactions (HOI) in video is challenging due to the complex, instance-specific interaction dynamics of both humans and objects. Incorporating controllability in video generation further adds to the complexity. Existing controllable video generation approaches face a trade-off: sparse controls like keypoint trajectories are easy to specify but lack instance-awareness, while dense signals such as optical flow, depths or 3D meshes are informative but costly to obtain. We propose VHOI, a two-stage framework that first densifies sparse trajectories into HOI mask sequences, and then fine-tunes a video diffusion model conditioned on these dense masks. We introduce a novel HOI-aware motion representation that uses color encodings to distinguish not only human and object motion, but also body-part-specific dynamics. This design incorporates a human prior into the conditioning signal and strengthens the model's ability to understand and generate realistic HOI dynamics. Experiments demonstrate state-of-the-art results in controllable HOI video generation. VHOI is not limited to interaction-only scenarios and can also generate full human navigation leading up to object interactions in an end-to-end manner. Project page: https://vcai.mpi-inf.mpg.de/projects/vhoi/.
>
---
#### [new 083] CS3D: An Efficient Facial Expression Recognition via Event Vision
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究事件视觉下的高效面部表情识别任务，旨在解决现有方法计算量大、能耗高的问题。提出CS3D框架，通过分解3D卷积、引入软尖峰神经元和时空注意力机制，在降低能耗的同时提升识别精度。**

- **链接: [https://arxiv.org/pdf/2512.09592v1](https://arxiv.org/pdf/2512.09592v1)**

> **作者:** Zhe Wang; Qijin Song; Yucen Peng; Weibang Bai
>
> **摘要:** Responsive and accurate facial expression recognition is crucial to human-robot interaction for daily service robots. Nowadays, event cameras are becoming more widely adopted as they surpass RGB cameras in capturing facial expression changes due to their high temporal resolution, low latency, computational efficiency, and robustness in low-light conditions. Despite these advantages, event-based approaches still encounter practical challenges, particularly in adopting mainstream deep learning models. Traditional deep learning methods for facial expression analysis are energy-intensive, making them difficult to deploy on edge computing devices and thereby increasing costs, especially for high-frequency, dynamic, event vision-based approaches. To address this challenging issue, we proposed the CS3D framework by decomposing the Convolutional 3D method to reduce the computational complexity and energy consumption. Additionally, by utilizing soft spiking neurons and a spatial-temporal attention mechanism, the ability to retain information is enhanced, thus improving the accuracy of facial expression detection. Experimental results indicate that our proposed CS3D method attains higher accuracy on multiple datasets compared to architectures such as the RNN, Transformer, and C3D, while the energy consumption of the CS3D method is just 21.97\% of the original C3D required on the same device.
>
---
#### [new 084] ConceptPose: Training-Free Zero-Shot Object Pose Estimation using Concept Vectors
- **分类: cs.CV**

- **简介: 该论文研究零样本物体位姿估计，旨在无需训练即可准确估计物体6DoF姿态。提出ConceptPose，利用视觉语言模型生成3D概念向量图，通过3D-3D对应实现高精度位姿估计，显著超越现有方法。**

- **链接: [https://arxiv.org/pdf/2512.09056v1](https://arxiv.org/pdf/2512.09056v1)**

> **作者:** Liming Kuang; Yordanka Velikova; Mahdi Saleh; Jan-Nico Zaech; Danda Pani Paudel; Benjamin Busam
>
> **摘要:** Object pose estimation is a fundamental task in computer vision and robotics, yet most methods require extensive, dataset-specific training. Concurrently, large-scale vision language models show remarkable zero-shot capabilities. In this work, we bridge these two worlds by introducing ConceptPose, a framework for object pose estimation that is both training-free and model-free. ConceptPose leverages a vision-language-model (VLM) to create open-vocabulary 3D concept maps, where each point is tagged with a concept vector derived from saliency maps. By establishing robust 3D-3D correspondences across concept maps, our approach allows precise estimation of 6DoF relative pose. Without any object or dataset-specific training, our approach achieves state-of-the-art results on common zero shot relative pose estimation benchmarks, significantly outperforming existing methods by over 62% in ADD(-S) score, including those that utilize extensive dataset-specific training.
>
---
#### [new 085] Dynamic Facial Expressions Analysis Based Parkinson's Disease Auxiliary Diagnosis
- **分类: cs.CV**

- **简介: 该论文提出一种基于动态面部表情分析的帕金森病辅助诊断方法，旨在解决传统诊断不便的问题。通过构建多模态网络提取面部表情强度特征，并结合LSTM进行分类，实现93.1%的准确率，提升诊断效率与体验。**

- **链接: [https://arxiv.org/pdf/2512.09276v1](https://arxiv.org/pdf/2512.09276v1)**

> **作者:** Xiaochen Huang; Xiaochen Bi; Cuihua Lv; Xin Wang; Haoyan Zhang; Wenjing Jiang; Xin Ma; Yibin Li
>
> **摘要:** Parkinson's disease (PD), a prevalent neurodegenerative disorder, significantly affects patients' daily functioning and social interactions. To facilitate a more efficient and accessible diagnostic approach for PD, we propose a dynamic facial expression analysis-based PD auxiliary diagnosis method. This method targets hypomimia, a characteristic clinical symptom of PD, by analyzing two manifestations: reduced facial expressivity and facial rigidity, thereby facilitating the diagnosis process. We develop a multimodal facial expression analysis network to extract expression intensity features during patients' performance of various facial expressions. This network leverages the CLIP architecture to integrate visual and textual features while preserving the temporal dynamics of facial expressions. Subsequently, the expression intensity features are processed and input into an LSTM-based classification network for PD diagnosis. Our method achieves an accuracy of 93.1%, outperforming other in-vitro PD diagnostic approaches. This technique offers a more convenient detection method for potential PD patients, improving their diagnostic experience.
>
---
#### [new 086] Adaptive Thresholding for Visual Place Recognition using Negative Gaussian Mixture Statistics
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉位置识别任务，旨在解决自动选择匹配阈值的问题。针对不同场景下手工设阈难的问题，提出利用“负”高斯混合统计来自动确定适应性强的阈值，适用于多种图像数据库和描述子。**

- **链接: [https://arxiv.org/pdf/2512.09071v1](https://arxiv.org/pdf/2512.09071v1)**

> **作者:** Nick Trinh; Damian Lyons
>
> **备注:** Accepted and presented at IEEE RoboticCC 2025. 4 pages short paper
>
> **摘要:** Visual place recognition (VPR) is an important component technology for camera-based mapping and navigation applications. This is a challenging problem because images of the same place may appear quite different for reasons including seasonal changes, weather illumination, structural changes to the environment, as well as transient pedestrian or vehicle traffic. Papers focusing on generating image descriptors for VPR report their results using metrics such as recall@K and ROC curves. However, for a robot implementation, determining which matches are sufficiently good is often reduced to a manually set threshold. And it is difficult to manually select a threshold that will work for a variety of visual scenarios. This paper addresses the problem of automatically selecting a threshold for VPR by looking at the 'negative' Gaussian mixture statistics for a place - image statistics indicating not this place. We show that this approach can be used to select thresholds that work well for a variety of image databases and image descriptors.
>
---
#### [new 087] Prompt-Based Continual Compositional Zero-Shot Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究持续组合零样本学习（CCZSL），解决视觉-语言模型在新增属性、对象组合时的遗忘问题。提出PromptCCZSL框架，通过提示学习、多教师蒸馏和多种损失函数，实现知识保留与组合泛化。**

- **链接: [https://arxiv.org/pdf/2512.09172v1](https://arxiv.org/pdf/2512.09172v1)**

> **作者:** Sauda Maryam; Sara Nadeem; Faisal Qureshi; Mohsen Ali
>
> **摘要:** We tackle continual adaptation of vision-language models to new attributes, objects, and their compositions in Compositional Zero-Shot Learning (CZSL), while preventing forgetting of prior knowledge. Unlike classical continual learning where classes are disjoint, CCZSL is more complex as attributes and objects may reoccur across sessions while compositions remain unique. Built on a frozen VLM backbone, we propose the first Prompt-based Continual Compositional Zero-Shot Learning (PromptCCZSL) framework that retains prior knowledge through recency-weighted multi-teacher distillation. It employs session-aware compositional prompts to fuse multimodal features for new compositions, while attribute and object prompts are learned through session-agnostic fusion to maintain global semantic consistency, which is further stabilized by a Cosine Anchor Loss (CAL) to preserve prior knowledge. To enhance adaptation in the current session, an Orthogonal Projection Loss (OPL) ensures that new attribute and object embeddings remain distinct from previous ones, preventing overlap, while an Intra-Session Diversity Loss (IDL) promotes variation among current-session embeddings for richer, more discriminative representations. We also introduce a comprehensive protocol that jointly measures catastrophic forgetting and compositional generalization. Extensive experiments on UT-Zappos and C-GQA benchmarks demonstrate that PromptCCZSL achieves substantial improvements over prior VLM-based and non-VLM baselines, setting a new benchmark for CCZSL in closed-world settings.
>
---
#### [new 088] Label-free Motion-Conditioned Diffusion Model for Cardiac Ultrasound Synthesis
- **分类: cs.CV**

- **简介: 该论文研究无标签心脏超声视频合成，旨在解决标注数据稀缺问题。提出运动条件扩散模型（MCDM），利用自监督运动特征生成逼真超声视频，无需人工标注，提升合成可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.09418v1](https://arxiv.org/pdf/2512.09418v1)**

> **作者:** Zhe Li; Hadrien Reynaud; Johanna P Müller; Bernhard Kainz
>
> **备注:** Accepted at MICAD 2025
>
> **摘要:** Ultrasound echocardiography is essential for the non-invasive, real-time assessment of cardiac function, but the scarcity of labelled data, driven by privacy restrictions and the complexity of expert annotation, remains a major obstacle for deep learning methods. We propose the Motion Conditioned Diffusion Model (MCDM), a label-free latent diffusion framework that synthesises realistic echocardiography videos conditioned on self-supervised motion features. To extract these features, we design the Motion and Appearance Feature Extractor (MAFE), which disentangles motion and appearance representations from videos. Feature learning is further enhanced by two auxiliary objectives: a re-identification loss guided by pseudo appearance features and an optical flow loss guided by pseudo flow fields. Evaluated on the EchoNet-Dynamic dataset, MCDM achieves competitive video generation performance, producing temporally coherent and clinically realistic sequences without reliance on manual labels. These results demonstrate the potential of self-supervised conditioning for scalable echocardiography synthesis. Our code is available at https://github.com/ZheLi2020/LabelfreeMCDM.
>
---
#### [new 089] Cytoplasmic Strings Analysis in Human Embryo Time-Lapse Videos using Deep Learning Framework
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对人类胚胎时序图像中细胞质串（CS）人工检测困难的问题，提出首个基于深度学习的分析框架。通过构建标注数据集，设计两阶段模型并引入NUCE损失，实现CS的自动分类与定位，提升检测准确率。**

- **链接: [https://arxiv.org/pdf/2512.09461v1](https://arxiv.org/pdf/2512.09461v1)**

> **作者:** Anabia Sohail; Mohamad Alansari; Ahmed Abughali; Asmaa Chehab; Abdelfatah Ahmed; Divya Velayudhan; Sajid Javed; Hasan Al Marzouqi; Ameena Saad Al-Sumaiti; Junaid Kashir; Naoufel Werghi
>
> **摘要:** Infertility is a major global health issue, and while in-vitro fertilization has improved treatment outcomes, embryo selection remains a critical bottleneck. Time-lapse imaging enables continuous, non-invasive monitoring of embryo development, yet most automated assessment methods rely solely on conventional morphokinetic features and overlook emerging biomarkers. Cytoplasmic Strings, thin filamentous structures connecting the inner cell mass and trophectoderm in expanded blastocysts, have been associated with faster blastocyst formation, higher blastocyst grades, and improved viability. However, CS assessment currently depends on manual visual inspection, which is labor-intensive, subjective, and severely affected by detection and subtle visual appearance. In this work, we present, to the best of our knowledge, the first computational framework for CS analysis in human IVF embryos. We first design a human-in-the-loop annotation pipeline to curate a biologically validated CS dataset from TLI videos, comprising 13,568 frames with highly sparse CS-positive instances. Building on this dataset, we propose a two-stage deep learning framework that (i) classifies CS presence at the frame level and (ii) localizes CS regions in positive cases. To address severe imbalance and feature uncertainty, we introduce the Novel Uncertainty-aware Contractive Embedding (NUCE) loss, which couples confidence-aware reweighting with an embedding contraction term to form compact, well-separated class clusters. NUCE consistently improves F1-score across five transformer backbones, while RF-DETR-based localization achieves state-of-the-art (SOTA) detection performance for thin, low-contrast CS structures. The source code will be made publicly available at: https://github.com/HamadYA/CS_Detection.
>
---
#### [new 090] Diffusion Posterior Sampler for Hyperspectral Unmixing with Spectral Variability Modeling
- **分类: cs.CV**

- **简介: 该论文研究高光谱解混任务，旨在解决端元光谱变异性和先验建模难题。提出DPS4Un方法，基于超像素构建端元束并训练扩散模型作为后验采样器，结合局部数据保真与迭代优化，实现更准确的丰度与端元估计。**

- **链接: [https://arxiv.org/pdf/2512.09871v1](https://arxiv.org/pdf/2512.09871v1)**

> **作者:** Yimin Zhu; Lincoln Linlin Xu
>
> **摘要:** Linear spectral mixture models (LMM) provide a concise form to disentangle the constituent materials (endmembers) and their corresponding proportions (abundance) in a single pixel. The critical challenges are how to model the spectral prior distribution and spectral variability. Prior knowledge and spectral variability can be rigorously modeled under the Bayesian framework, where posterior estimation of Abundance is derived by combining observed data with endmember prior distribution. Considering the key challenges and the advantages of the Bayesian framework, a novel method using a diffusion posterior sampler for semiblind unmixing, denoted as DPS4Un, is proposed to deal with these challenges with the following features: (1) we view the pretrained conditional spectrum diffusion model as a posterior sampler, which can combine the learned endmember prior with observation to get the refined abundance distribution. (2) Instead of using the existing spectral library as prior, which may raise bias, we establish the image-based endmember bundles within superpixels, which are used to train the endmember prior learner with diffusion model. Superpixels make sure the sub-scene is more homogeneous. (3) Instead of using the image-level data consistency constraint, the superpixel-based data fidelity term is proposed. (4) The endmember is initialized as Gaussian noise for each superpixel region, DPS4Un iteratively updates the abundance and endmember, contributing to spectral variability modeling. The experimental results on three real-world benchmark datasets demonstrate that DPS4Un outperforms the state-of-the-art hyperspectral unmixing methods.
>
---
#### [new 091] ReViSE: Towards Reason-Informed Video Editing in Unified Models with Self-Reflective Learning
- **分类: cs.CV**

- **简介: 该论文研究视频编辑任务中的推理与生成协同问题，提出Reason-Informed Video Editing（RVE）任务及RVE-Bench评测基准，并构建ReViSE框架，通过自反思学习统一推理与编辑，提升编辑的物理合理性和因果逻辑性。**

- **链接: [https://arxiv.org/pdf/2512.09924v1](https://arxiv.org/pdf/2512.09924v1)**

> **作者:** Xinyu Liu; Hangjie Yuan; Yujie Wei; Jiazheng Xing; Yujin Han; Jiahao Pan; Yanbiao Ma; Chi-Min Chan; Kang Zhao; Shiwei Zhang; Wenhan Luo; Yike Guo
>
> **摘要:** Video unified models exhibit strong capabilities in understanding and generation, yet they struggle with reason-informed visual editing even when equipped with powerful internal vision-language models (VLMs). We attribute this gap to two factors: 1) existing datasets are inadequate for training and evaluating reasoning-aware video editing, and 2) an inherent disconnect between the models' reasoning and editing capabilities, which prevents the rich understanding from effectively instructing the editing process. Bridging this gap requires an integrated framework that connects reasoning with visual transformation. To address this gap, we introduce the Reason-Informed Video Editing (RVE) task, which requires reasoning about physical plausibility and causal dynamics during editing. To support systematic evaluation, we construct RVE-Bench, a comprehensive benchmark with two complementary subsets: Reasoning-Informed Video Editing and In-Context Video Generation. These subsets cover diverse reasoning dimensions and real-world editing scenarios. Building upon this foundation, we propose the ReViSE, a Self-Reflective Reasoning (SRF) framework that unifies generation and evaluation within a single architecture. The model's internal VLM provides intrinsic feedback by assessing whether the edited video logically satisfies the given instruction. The differential feedback that refines the generator's reasoning behavior during training. Extensive experiments on RVE-Bench demonstrate that ReViSE significantly enhances editing accuracy and visual fidelity, achieving a 32% improvement of the Overall score in the reasoning-informed video editing subset over state-of-the-art methods.
>
---
#### [new 092] FastPose-ViT: A Vision Transformer for Real-Time Spacecraft Pose Estimation
- **分类: cs.CV**

- **简介: 该论文研究航天器6DoF位姿估计任务，旨在实现实时、轻量化的单图位姿回归。针对传统PnP方法计算量大问题，提出FastPose-ViT模型，基于Vision Transformer直接回归位姿，并通过几何映射与表观旋转修正提升精度，支持边缘设备高效部署。**

- **链接: [https://arxiv.org/pdf/2512.09792v1](https://arxiv.org/pdf/2512.09792v1)**

> **作者:** Pierre Ancey; Andrew Price; Saqib Javed; Mathieu Salzmann
>
> **备注:** Accepted to WACV 2026. Preprint version
>
> **摘要:** Estimating the 6-degrees-of-freedom (6DoF) pose of a spacecraft from a single image is critical for autonomous operations like in-orbit servicing and space debris removal. Existing state-of-the-art methods often rely on iterative Perspective-n-Point (PnP)-based algorithms, which are computationally intensive and ill-suited for real-time deployment on resource-constrained edge devices. To overcome these limitations, we propose FastPose-ViT, a Vision Transformer (ViT)-based architecture that directly regresses the 6DoF pose. Our approach processes cropped images from object bounding boxes and introduces a novel mathematical formalism to map these localized predictions back to the full-image scale. This formalism is derived from the principles of projective geometry and the concept of "apparent rotation", where the model predicts an apparent rotation matrix that is then corrected to find the true orientation. We demonstrate that our method outperforms other non-PnP strategies and achieves performance competitive with state-of-the-art PnP-based techniques on the SPEED dataset. Furthermore, we validate our model's suitability for real-world space missions by quantizing it and deploying it on power-constrained edge hardware. On the NVIDIA Jetson Orin Nano, our end-to-end pipeline achieves a latency of ~75 ms per frame under sequential execution, and a non-blocking throughput of up to 33 FPS when stages are scheduled concurrently.
>
---
#### [new 093] LoGoColor: Local-Global 3D Colorization for 360° Scenes
- **分类: cs.CV**

- **简介: 该论文研究360°场景的3D着色任务，旨在解决现有方法因2D模型颜色平均化导致的色彩单一问题。提出LoGoColor方法，通过局部-全局策略和多视角扩散模型，提升着色多样性与多视角一致性。**

- **链接: [https://arxiv.org/pdf/2512.09278v1](https://arxiv.org/pdf/2512.09278v1)**

> **作者:** Yeonjin Chang; Juhwan Cho; Seunghyeon Seo; Wonsik Shin; Nojun Kwak
>
> **摘要:** Single-channel 3D reconstruction is widely used in fields such as robotics and medical imaging. While this line of work excels at reconstructing 3D geometry, the outputs are not colored 3D models, thus 3D colorization is required for visualization. Recent 3D colorization studies address this problem by distilling 2D image colorization models. However, these approaches suffer from an inherent inconsistency of 2D image models. This results in colors being averaged during training, leading to monotonous and oversimplified results, particularly in complex 360° scenes. In contrast, we aim to preserve color diversity by generating a new set of consistently colorized training views, thereby bypassing the averaging process. Nevertheless, eliminating the averaging process introduces a new challenge: ensuring strict multi-view consistency across these colorized views. To achieve this, we propose LoGoColor, a pipeline designed to preserve color diversity by eliminating this guidance-averaging process with a `Local-Global' approach: we partition the scene into subscenes and explicitly tackle both inter-subscene and intra-subscene consistency using a fine-tuned multi-view diffusion model. We demonstrate that our method achieves quantitatively and qualitatively more consistent and plausible 3D colorization on complex 360° scenes than existing methods, and validate its superior color diversity using a novel Color Diversity Index.
>
---
#### [new 094] MelanomaNet: Explainable Deep Learning for Skin Lesion Classification
- **分类: cs.CV**

- **简介: 该论文提出MelanomaNet，面向皮肤病变分类任务，旨在解决深度学习模型缺乏可解释性的问题。通过结合注意力可视化、临床特征提取、概念激活与不确定性量化，实现高性能且可解释的多类皮肤病变分类，提升临床可信度与应用潜力。**

- **链接: [https://arxiv.org/pdf/2512.09289v1](https://arxiv.org/pdf/2512.09289v1)**

> **作者:** Sukhrobbek Ilyosbekov
>
> **备注:** 7 pages, 3 figures
>
> **摘要:** Automated skin lesion classification using deep learning has shown remarkable accuracy, yet clinical adoption remains limited due to the "black box" nature of these models. We present MelanomaNet, an explainable deep learning system for multi-class skin lesion classification that addresses this gap through four complementary interpretability mechanisms. Our approach combines an EfficientNet V2 backbone with GradCAM++ attention visualization, automated ABCDE clinical criterion extraction, Fast Concept Activation Vectors (FastCAV) for concept-based explanations, and Monte Carlo Dropout uncertainty quantification. We evaluate our system on the ISIC 2019 dataset containing 25,331 dermoscopic images across 9 diagnostic categories. Our model achieves 85.61% accuracy with a weighted F1 score of 0.8564, while providing clinically meaningful explanations that align model attention with established dermatological assessment criteria. The uncertainty quantification module decomposes prediction confidence into epistemic and aleatoric components, enabling automatic flagging of unreliable predictions for clinical review. Our results demonstrate that high classification performance can be achieved alongside comprehensive interpretability, potentially facilitating greater trust and adoption in clinical dermatology workflows. The source code is available at https://github.com/suxrobgm/explainable-melanoma
>
---
#### [new 095] Seeing Soil from Space: Towards Robust and Scalable Remote Soil Nutrient Analysis
- **分类: cs.CV; physics.geo-ph**

- **简介: 该论文属遥感与农业交叉任务，旨在解决土壤养分大范围精准预测难题。作者提出融合物理模型与深度学习的混合建模方法，利用遥感数据和环境协变量，实现对耕地土壤有机碳、氮等关键指标的稳健、可扩展估算，并通过严格空间验证验证有效性。**

- **链接: [https://arxiv.org/pdf/2512.09576v1](https://arxiv.org/pdf/2512.09576v1)**

> **作者:** David Seu; Nicolas Longepe; Gabriel Cioltea; Erik Maidik; Calin Andrei
>
> **备注:** 23 pages, 13 figures, 13 tables
>
> **摘要:** Environmental variables are increasingly affecting agricultural decision-making, yet accessible and scalable tools for soil assessment remain limited. This study presents a robust and scalable modeling system for estimating soil properties in croplands, including soil organic carbon (SOC), total nitrogen (N), available phosphorus (P), exchangeable potassium (K), and pH, using remote sensing data and environmental covariates. The system employs a hybrid modeling approach, combining the indirect methods of modeling soil through proxies and drivers with direct spectral modeling. We extend current approaches by using interpretable physics-informed covariates derived from radiative transfer models (RTMs) and complex, nonlinear embeddings from a foundation model. We validate the system on a harmonized dataset that covers Europes cropland soils across diverse pedoclimatic zones. Evaluation is conducted under a robust validation framework that enforces strict spatial blocking, stratified splits, and statistically distinct train-test sets, which deliberately make the evaluation harder and produce more realistic error estimates for unseen regions. The models achieved their highest accuracy for SOC and N. This performance held across unseen locations, under both spatial cross-validation and an independent test set. SOC obtained a MAE of 5.12 g/kg and a CCC of 0.77, and N obtained a MAE of 0.44 g/kg and a CCC of 0.77. We also assess uncertainty through conformal calibration, achieving 90 percent coverage at the target confidence level. This study contributes to the digital advancement of agriculture through the application of scalable, data-driven soil analysis frameworks that can be extended to related domains requiring quantitative soil evaluation, such as carbon markets.
>
---
#### [new 096] An Efficient Test-Time Scaling Approach for Image Generation
- **分类: cs.CV**

- **简介: 该论文研究图像生成中的测试时计算扩展问题，旨在提升推理效率。针对现有方法在去噪步骤中计算分配不均的问题，提出Verifier-Threshold方法，自动重分配计算资源，在保持性能的同时减少2-4倍计算时间。**

- **链接: [https://arxiv.org/pdf/2512.08985v1](https://arxiv.org/pdf/2512.08985v1)**

> **作者:** Vignesh Sundaresha; Akash Haridas; Vikram Appia; Lav Varshney
>
> **备注:** 11 pages
>
> **摘要:** Image generation has emerged as a mainstream application of large generative AI models. Just as test-time compute and reasoning have helped language models improve their capabilities, similar benefits have also been observed with image generation models. In particular, searching over noise samples for diffusion and flow models has shown to scale well with test-time compute. While recent works have explored allocating non-uniform inference-compute budgets across different denoising steps, they rely on greedy algorithms and allocate the compute budget ineffectively. In this work, we study this problem and propose solutions to fix it. We propose the Verifier-Threshold method which automatically reallocates test-time compute and delivers substantial efficiency improvements. For the same performance on the GenEval benchmark, we achieve a 2-4x reduction in computational time over the state-of-the-art method.
>
---
#### [new 097] Enhancing Knowledge Transfer in Hyperspectral Image Classification via Cross-scene Knowledge Integration
- **分类: cs.CV**

- **简介: 该论文研究跨场景高光谱图像分类中的知识迁移问题，旨在解决传感器差异和语义不一致导致的跨域迁移困难。提出CKI框架，通过光谱对齐、知识共享偏好和互补信息融合，实现非重叠类别下的有效知识转移。**

- **链接: [https://arxiv.org/pdf/2512.08989v1](https://arxiv.org/pdf/2512.08989v1)**

> **作者:** Lu Huo; Wenjian Huang; Jianguo Zhang; Min Xu; Haimin Zhang
>
> **摘要:** Knowledge transfer has strong potential to improve hyperspectral image (HSI) classification, yet two inherent challenges fundamentally restrict effective cross-domain transfer: spectral variations caused by different sensors and semantic inconsistencies across heterogeneous scenes. Existing methods are limited by transfer settings that assume homogeneous domains or heterogeneous scenarios with only co-occurring categories. When label spaces do not overlap, they further rely on complete source-domain coverage and therefore overlook critical target-private information. To overcome these limitations and enable knowledge transfer in fully heterogeneous settings, we propose Cross-scene Knowledge Integration (CKI), a framework that explicitly incorporates target-private knowledge during transfer. CKI includes: (1) Alignment of Spectral Characteristics (ASC) to reduce spectral discrepancies through domain-agnostic projection; (2) Cross-scene Knowledge Sharing Preference (CKSP), which resolves semantic mismatch via a Source Similarity Mechanism (SSM); and (3) Complementary Information Integration (CII) to maximize the use of target-specific complementary cues. Extensive experiments verify that CKI achieves state-of-the-art performance with strong stability across diverse cross-scene HSI scenarios.
>
---
#### [new 098] Demo: Generative AI helps Radiotherapy Planning with User Preference
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究放射治疗计划中的3D剂量预测任务，旨在解决现有模型因依赖参考计划而受限于特定机构偏好的问题。作者提出一种基于用户偏好风格的生成式AI模型，可个性化权衡靶区与危及器官的剂量分布，提升计划灵活性与质量，并实现与临床系统的兼容集成。**

- **链接: [https://arxiv.org/pdf/2512.08996v1](https://arxiv.org/pdf/2512.08996v1)**

> **作者:** Riqiang Gao; Simon Arberet; Martin Kraus; Han Liu; Wilko FAR Verbakel; Dorin Comaniciu; Florin-Cristian Ghesu; Ali Kamen
>
> **备注:** Best paper in GenAI4Health at NeurIPS 2025
>
> **摘要:** Radiotherapy planning is a highly complex process that often varies significantly across institutions and individual planners. Most existing deep learning approaches for 3D dose prediction rely on reference plans as ground truth during training, which can inadvertently bias models toward specific planning styles or institutional preferences. In this study, we introduce a novel generative model that predicts 3D dose distributions based solely on user-defined preference flavors. These customizable preferences enable planners to prioritize specific trade-offs between organs-at-risk (OARs) and planning target volumes (PTVs), offering greater flexibility and personalization. Designed for seamless integration with clinical treatment planning systems, our approach assists users in generating high-quality plans efficiently. Comparative evaluations demonstrate that our method can surpasses the Varian RapidPlan model in both adaptability and plan quality in some scenarios.
>
---
#### [new 099] UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking
- **分类: cs.CV; cs.SD**

- **简介: 该论文研究音频驱动的对话虚拟人生成，旨在解决现有方法难以自然建模听者表情的问题。提出UniLS框架，首次实现仅基于双通道音频的端到端说-听联合生成，通过两阶段训练学习听者内部运动先验并结合音频调节，显著提升听态自然度与多样性。**

- **链接: [https://arxiv.org/pdf/2512.09327v1](https://arxiv.org/pdf/2512.09327v1)**

> **作者:** Xuangeng Chu; Ruicong Liu; Yifei Huang; Yun Liu; Yichen Peng; Bo Zheng
>
> **摘要:** Generating lifelike conversational avatars requires modeling not just isolated speakers, but the dynamic, reciprocal interaction of speaking and listening. However, modeling the listener is exceptionally challenging: direct audio-driven training fails, producing stiff, static listening motions. This failure stems from a fundamental imbalance: the speaker's motion is strongly driven by speech audio, while the listener's motion primarily follows an internal motion prior and is only loosely guided by external speech. This challenge has led most methods to focus on speak-only generation. The only prior attempt at joint generation relies on extra speaker's motion to produce the listener. This design is not end-to-end, thereby hindering the real-time applicability. To address this limitation, we present UniLS, the first end-to-end framework for generating unified speak-listen expressions, driven by only dual-track audio. Our method introduces a novel two-stage training paradigm. Stage 1 first learns the internal motion prior by training an audio-free autoregressive generator, capturing the spontaneous dynamics of natural facial motion. Stage 2 then introduces the dual-track audio, fine-tuning the generator to modulate the learned motion prior based on external speech cues. Extensive evaluations show UniLS achieves state-of-the-art speaking accuracy. More importantly, it delivers up to 44.1\% improvement in listening metrics, generating significantly more diverse and natural listening expressions. This effectively mitigates the stiffness problem and provides a practical, high-fidelity audio-driven solution for interactive digital humans.
>
---
#### [new 100] Explainable Fundus Image Curation and Lesion Detection in Diabetic Retinopathy
- **分类: cs.CV; cs.AI**

- **简介: 该论文属医学图像分析任务，旨在解决糖尿病视网膜病变中因图像质量与标注差异导致的AI训练难题。提出一种可解释的质量控制框架，结合图像处理与对比学习筛选高质量眼底图像，并利用深度学习辅助标注，通过标注一致性评估确保数据可靠性。**

- **链接: [https://arxiv.org/pdf/2512.08986v1](https://arxiv.org/pdf/2512.08986v1)**

> **作者:** Anca Mihai; Adrian Groza
>
> **摘要:** Diabetic Retinopathy (DR) affects individuals with long-term diabetes. Without early diagnosis, DR can lead to vision loss. Fundus photography captures the structure of the retina along with abnormalities indicative of the stage of the disease. Artificial Intelligence (AI) can support clinicians in identifying these lesions, reducing manual workload, but models require high-quality annotated datasets. Due to the complexity of retinal structures, errors in image acquisition and lesion interpretation of manual annotators can occur. We proposed a quality-control framework, ensuring only high-standard data is used for evaluation and AI training. First, an explainable feature-based classifier is used to filter inadequate images. The features are extracted both using image processing and contrastive learning. Then, the images are enhanced and put subject to annotation, using deep-learning-based assistance. Lastly, the agreement between annotators calculated using derived formulas determines the usability of the annotations.
>
---
#### [new 101] ROI-Packing: Efficient Region-Based Compression for Machine Vision
- **分类: cs.CV**

- **简介: 该论文提出ROI-Packing，针对机器视觉任务（如检测与分割）设计高效图像压缩方法。通过优先保留关键区域并高效打包，显著降低比特率而不影响模型精度，优于现有VVC编码标准。**

- **链接: [https://arxiv.org/pdf/2512.09258v1](https://arxiv.org/pdf/2512.09258v1)**

> **作者:** Md Eimran Hossain Eimon; Alena Krause; Ashan Perera; Juan Merlos; Hari Kalva; Velibor Adzic; Borko Furht
>
> **摘要:** This paper introduces ROI-Packing, an efficient image compression method tailored specifically for machine vision. By prioritizing regions of interest (ROI) critical to end-task accuracy and packing them efficiently while discarding less relevant data, ROI-Packing achieves significant compression efficiency without requiring retraining or fine-tuning of end-task models. Comprehensive evaluations across five datasets and two popular tasks-object detection and instance segmentation-demonstrate up to a 44.10% reduction in bitrate without compromising end-task accuracy, along with an 8.88 % improvement in accuracy at the same bitrate compared to the state-of-the-art Versatile Video Coding (VVC) codec standardized by the Moving Picture Experts Group (MPEG).
>
---
#### [new 102] ASSIST-3D: Adapted Scene Synthesis for Class-Agnostic 3D Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文研究类无关3D实例分割，旨在提升模型对未见物体的泛化能力。针对现有合成数据缺乏几何多样性和布局合理性的难题，提出ASSIST-3D框架，通过异构物体选择、大模型引导的布局生成和多视角点云合成，构建高质量训练数据，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.09364v1](https://arxiv.org/pdf/2512.09364v1)**

> **作者:** Shengchao Zhou; Jiehong Lin; Jiahui Liu; Shizhen Zhao; Chirui Chang; Xiaojuan Qi
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Class-agnostic 3D instance segmentation tackles the challenging task of segmenting all object instances, including previously unseen ones, without semantic class reliance. Current methods struggle with generalization due to the scarce annotated 3D scene data or noisy 2D segmentations. While synthetic data generation offers a promising solution, existing 3D scene synthesis methods fail to simultaneously satisfy geometry diversity, context complexity, and layout reasonability, each essential for this task. To address these needs, we propose an Adapted 3D Scene Synthesis pipeline for class-agnostic 3D Instance SegmenTation, termed as ASSIST-3D, to synthesize proper data for model generalization enhancement. Specifically, ASSIST-3D features three key innovations, including 1) Heterogeneous Object Selection from extensive 3D CAD asset collections, incorporating randomness in object sampling to maximize geometric and contextual diversity; 2) Scene Layout Generation through LLM-guided spatial reasoning combined with depth-first search for reasonable object placements; and 3) Realistic Point Cloud Construction via multi-view RGB-D image rendering and fusion from the synthetic scenes, closely mimicking real-world sensor data acquisition. Experiments on ScanNetV2, ScanNet++, and S3DIS benchmarks demonstrate that models trained with ASSIST-3D-generated data significantly outperform existing methods. Further comparisons underscore the superiority of our purpose-built pipeline over existing 3D scene synthesis approaches.
>
---
#### [new 103] From SAM to DINOv2: Towards Distilling Foundation Models to Lightweight Baselines for Generalized Polyp Segmentation
- **分类: cs.CV**

- **简介: 该论文研究结肠镜息肉分割任务，旨在解决轻量模型分割性能不足与大模型难以直接应用于医疗领域的问题。提出Polyp-DiFoM蒸馏框架，将基础模型的知识迁移至U-Net等轻量模型，提升分割精度与泛化能力，实现实验性能超越现有方法且计算开销显著降低。**

- **链接: [https://arxiv.org/pdf/2512.09307v1](https://arxiv.org/pdf/2512.09307v1)**

> **作者:** Shivanshu Agnihotri; Snehashis Majhi; Deepak Ranjan Nayak; Debesh Jha
>
> **摘要:** Accurate polyp segmentation during colonoscopy is critical for the early detection of colorectal cancer and still remains challenging due to significant size, shape, and color variations, and the camouflaged nature of polyps. While lightweight baseline models such as U-Net, U-Net++, and PraNet offer advantages in terms of easy deployment and low computational cost, they struggle to deal with the above issues, leading to limited segmentation performance. In contrast, large-scale vision foundation models such as SAM, DINOv2, OneFormer, and Mask2Former have exhibited impressive generalization performance across natural image domains. However, their direct transfer to medical imaging tasks (e.g., colonoscopic polyp segmentation) is not straightforward, primarily due to the scarcity of large-scale datasets and lack of domain-specific knowledge. To bridge this gap, we propose a novel distillation framework, Polyp-DiFoM, that transfers the rich representations of foundation models into lightweight segmentation baselines, allowing efficient and accurate deployment in clinical settings. In particular, we infuse semantic priors from the foundation models into canonical architectures such as U-Net and U-Net++ and further perform frequency domain encoding for enhanced distillation, corroborating their generalization capability. Extensive experiments are performed across five benchmark datasets, such as Kvasir-SEG, CVC-ClinicDB, ETIS, ColonDB, and CVC-300. Notably, Polyp-DiFoM consistently outperforms respective baseline models significantly, as well as the state-of-the-art model, with nearly 9 times reduced computation overhead. The code is available at https://github.com/lostinrepo/PolypDiFoM.
>
---
#### [new 104] Unconsciously Forget: Mitigating Memorization; Without Knowing What is being Memorized
- **分类: cs.CV**

- **简介: 该论文属于模型去记忆化任务，旨在解决生成模型过度记忆训练数据导致的版权问题。作者提出UniForget方法，通过模型剪枝抑制版权内容生成，无需针对特定概念，保留模型生成能力，且可与现有去记忆方法结合。**

- **链接: [https://arxiv.org/pdf/2512.09687v1](https://arxiv.org/pdf/2512.09687v1)**

> **作者:** Er Jin; Yang Zhang; Yongli Mou; Yanfei Dong; Stefan Decker; Kenji Kawaguchi; Johannes Stegmaier
>
> **摘要:** Recent advances in generative models have demonstrated an exceptional ability to produce highly realistic images. However, previous studies show that generated images often resemble the training data, and this problem becomes more severe as the model size increases. Memorizing training data can lead to legal challenges, including copyright infringement, violations of portrait rights, and trademark violations. Existing approaches to mitigating memorization mainly focus on manipulating the denoising sampling process to steer image embeddings away from the memorized embedding space or employ unlearning methods that require training on datasets containing specific sets of memorized concepts. However, existing methods often incur substantial computational overhead during sampling, or focus narrowly on removing one or more groups of target concepts, imposing a significant limitation on their scalability. To understand and mitigate these problems, our work, UniForget, offers a new perspective on understanding the root cause of memorization. Our work demonstrates that specific parts of the model are responsible for copyrighted content generation. By applying model pruning, we can effectively suppress the probability of generating copyrighted content without targeting specific concepts while preserving the general generative capabilities of the model. Additionally, we show that our approach is both orthogonal and complementary to existing unlearning methods, thereby highlighting its potential to improve current unlearning and de-memorization techniques.
>
---
#### [new 105] DirectSwap: Mask-Free Cross-Identity Training and Benchmarking for Expression-Consistent Video Head Swapping
- **分类: cs.CV**

- **简介: 该论文研究视频换脸任务，旨在实现跨身份、表情一致的头部替换。针对缺乏配对数据和掩码导致的信息丢失问题，提出生成配对数据集HeadSwapBench，并设计无需掩码的DirectSwap框架与MEAR损失，提升身份保真度与动作连贯性。**

- **链接: [https://arxiv.org/pdf/2512.09417v1](https://arxiv.org/pdf/2512.09417v1)**

> **作者:** Yanan Wang; Shengcai Liao; Panwen Hu; Xin Li; Fan Yang; Xiaodan Liang
>
> **摘要:** Video head swapping aims to replace the entire head of a video subject, including facial identity, head shape, and hairstyle, with that of a reference image, while preserving the target body, background, and motion dynamics. Due to the lack of ground-truth paired swapping data, prior methods typically train on cross-frame pairs of the same person within a video and rely on mask-based inpainting to mitigate identity leakage. Beyond potential boundary artifacts, this paradigm struggles to recover essential cues occluded by the mask, such as facial pose, expressions, and motion dynamics. To address these issues, we prompt a video editing model to synthesize new heads for existing videos as fake swapping inputs, while maintaining frame-synchronized facial poses and expressions. This yields HeadSwapBench, the first cross-identity paired dataset for video head swapping, which supports both training (\TrainNum{} videos) and benchmarking (\TestNum{} videos) with genuine outputs. Leveraging this paired supervision, we propose DirectSwap, a mask-free, direct video head-swapping framework that extends an image U-Net into a video diffusion model with a motion module and conditioning inputs. Furthermore, we introduce the Motion- and Expression-Aware Reconstruction (MEAR) loss, which reweights the diffusion loss per pixel using frame-difference magnitudes and facial-landmark proximity, thereby enhancing cross-frame coherence in motion and expressions. Extensive experiments demonstrate that DirectSwap achieves state-of-the-art visual quality, identity fidelity, and motion and expression consistency across diverse in-the-wild video scenes. We will release the source code and the HeadSwapBench dataset to facilitate future research.
>
---
#### [new 106] A Clinically Interpretable Deep CNN Framework for Early Chronic Kidney Disease Prediction Using Grad-CAM-Based Explainable AI
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种可解释的深度CNN框架，用于基于CT图像的早期慢性肾病（CKD）预测。通过SMOTE平衡类别，结合Grad-CAM提升模型可解释性，实现100%准确率，助力临床早期诊断。**

- **链接: [https://arxiv.org/pdf/2512.09244v1](https://arxiv.org/pdf/2512.09244v1)**

> **作者:** Anas Bin Ayub; Nilima Sultana Niha; Md. Zahurul Haque
>
> **摘要:** Chronic Kidney Disease (CKD) constitutes a major global medical burden, marked by the gradual deterioration of renal function, which results in the impaired clearance of metabolic waste and disturbances in systemic fluid homeostasis. Owing to its substantial contribution to worldwide morbidity and mortality, the development of reliable and efficient diagnostic approaches is critically important to facilitate early detection and prompt clinical management. This study presents a deep convolutional neural network (CNN) for early CKD detection from CT kidney images, complemented by class balancing using Synthetic Minority Over-sampling Technique (SMOTE) and interpretability via Gradient-weighted Class Activation Mapping (Grad-CAM). The model was trained and evaluated on the CT KIDNEY DATASET, which contains 12,446 CT images, including 3,709 cyst, 5,077 normal, 1,377 stone, and 2,283 tumor cases. The proposed deep CNN achieved a remarkable classification performance, attaining 100% accuracy in the early detection of chronic kidney disease (CKD). This significant advancement demonstrates strong potential for addressing critical clinical diagnostic challenges and enhancing early medical intervention strategies.
>
---
#### [new 107] DynaIP: Dynamic Image Prompt Adapter for Scalable Zero-shot Personalized Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文研究个性化文生图任务，旨在解决现有方法在概念保持、细节还原和多主体扩展上的难题。提出DynaIP插件，通过动态解耦策略与分层特征融合模块，提升细粒度保真度、平衡概念与提示跟随能力，并支持多主体个性化生成。**

- **链接: [https://arxiv.org/pdf/2512.09814v1](https://arxiv.org/pdf/2512.09814v1)**

> **作者:** Zhizhong Wang; Tianyi Chu; Zeyi Huang; Nanyang Wang; Kehan Li
>
> **摘要:** Personalized Text-to-Image (PT2I) generation aims to produce customized images based on reference images. A prominent interest pertains to the integration of an image prompt adapter to facilitate zero-shot PT2I without test-time fine-tuning. However, current methods grapple with three fundamental challenges: 1. the elusive equilibrium between Concept Preservation (CP) and Prompt Following (PF), 2. the difficulty in retaining fine-grained concept details in reference images, and 3. the restricted scalability to extend to multi-subject personalization. To tackle these challenges, we present Dynamic Image Prompt Adapter (DynaIP), a cutting-edge plugin to enhance the fine-grained concept fidelity, CP-PF balance, and subject scalability of SOTA T2I multimodal diffusion transformers (MM-DiT) for PT2I generation. Our key finding is that MM-DiT inherently exhibit decoupling learning behavior when injecting reference image features into its dual branches via cross attentions. Based on this, we design an innovative Dynamic Decoupling Strategy that removes the interference of concept-agnostic information during inference, significantly enhancing the CP-PF balance and further bolstering the scalability of multi-subject compositions. Moreover, we identify the visual encoder as a key factor affecting fine-grained CP and reveal that the hierarchical features of commonly used CLIP can capture visual information at diverse granularity levels. Therefore, we introduce a novel Hierarchical Mixture-of-Experts Feature Fusion Module to fully leverage the hierarchical features of CLIP, remarkably elevating the fine-grained concept fidelity while also providing flexible control of visual granularity. Extensive experiments across single- and multi-subject PT2I tasks verify that our DynaIP outperforms existing approaches, marking a notable advancement in the field of PT2l generation.
>
---
#### [new 108] FoundIR-v2: Optimizing Pre-Training Data Mixtures for Image Restoration Foundation Model
- **分类: cs.CV**

- **简介: 该论文属图像恢复任务，旨在优化多任务预训练数据混合比例。提出FoundIR-v2模型，采用数据均衡调度与MoE调度器，动态调整数据配比并分配任务自适应扩散先验，提升多任务综合性能。**

- **链接: [https://arxiv.org/pdf/2512.09282v1](https://arxiv.org/pdf/2512.09282v1)**

> **作者:** Xiang Chen; Jinshan Pan; Jiangxin Dong; Jian Yang; Jinhui Tang
>
> **备注:** Project page: https://lowlevelcv.com/
>
> **摘要:** Recent studies have witnessed significant advances in image restoration foundation models driven by improvements in the scale and quality of pre-training data. In this work, we find that the data mixture proportions from different restoration tasks are also a critical factor directly determining the overall performance of all-in-one image restoration models. To this end, we propose a high-capacity diffusion-based image restoration foundation model, FoundIR-v2, which adopts a data equilibrium scheduling paradigm to dynamically optimize the proportions of mixed training datasets from different tasks. By leveraging the data mixing law, our method ensures a balanced dataset composition, enabling the model to achieve consistent generalization and comprehensive performance across diverse tasks. Furthermore, we introduce an effective Mixture-of-Experts (MoE)-driven scheduler into generative pre-training to flexibly allocate task-adaptive diffusion priors for each restoration task, accounting for the distinct degradation forms and levels exhibited by different tasks. Extensive experiments demonstrate that our method can address over 50 sub-tasks across a broader scope of real-world scenarios and achieves favorable performance against state-of-the-art approaches.
>
---
#### [new 109] Representation Calibration and Uncertainty Guidance for Class-Incremental Learning based on Vision Language Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于类增量学习任务，旨在解决视觉语言模型在持续学习中新旧类混淆的问题。作者提出任务特定适配器、跨任务表示校准策略及基于预测不确定性的推理方法，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2512.09441v1](https://arxiv.org/pdf/2512.09441v1)**

> **作者:** Jiantao Tan; Peixian Ma; Tong Yu; Wentao Zhang; Ruixuan Wang
>
> **摘要:** Class-incremental learning requires a learning system to continually learn knowledge of new classes and meanwhile try to preserve previously learned knowledge of old classes. As current state-of-the-art methods based on Vision-Language Models (VLMs) still suffer from the issue of differentiating classes across learning tasks. Here a novel VLM-based continual learning framework for image classification is proposed. In this framework, task-specific adapters are added to the pre-trained and frozen image encoder to learn new knowledge, and a novel cross-task representation calibration strategy based on a mixture of light-weight projectors is used to help better separate all learned classes in a unified feature space, alleviating class confusion across tasks. In addition, a novel inference strategy guided by prediction uncertainty is developed to more accurately select the most appropriate image feature for class prediction. Extensive experiments on multiple datasets under various settings demonstrate the superior performance of our method compared to existing ones.
>
---
#### [new 110] KD-OCT: Efficient Knowledge Distillation for Clinical-Grade Retinal OCT Classification
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在解决深度模型在临床OCT图像诊断中计算开销大的问题。作者提出KD-OCT框架，通过知识蒸馏将大模型知识迁移到轻量模型，实现高效、准确的AMD和CNV分类，支持边缘部署。**

- **链接: [https://arxiv.org/pdf/2512.09069v1](https://arxiv.org/pdf/2512.09069v1)**

> **作者:** Erfan Nourbakhsh; Nasrin Sanjari; Ali Nourbakhsh
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** Age-related macular degeneration (AMD) and choroidal neovascularization (CNV)-related conditions are leading causes of vision loss worldwide, with optical coherence tomography (OCT) serving as a cornerstone for early detection and management. However, deploying state-of-the-art deep learning models like ConvNeXtV2-Large in clinical settings is hindered by their computational demands. Therefore, it is desirable to develop efficient models that maintain high diagnostic performance while enabling real-time deployment. In this study, a novel knowledge distillation framework, termed KD-OCT, is proposed to compress a high-performance ConvNeXtV2-Large teacher model, enhanced with advanced augmentations, stochastic weight averaging, and focal loss, into a lightweight EfficientNet-B2 student for classifying normal, drusen, and CNV cases. KD-OCT employs real-time distillation with a combined loss balancing soft teacher knowledge transfer and hard ground-truth supervision. The effectiveness of the proposed method is evaluated on the Noor Eye Hospital (NEH) dataset using patient-level cross-validation. Experimental results demonstrate that KD-OCT outperforms comparable multi-scale or feature-fusion OCT classifiers in efficiency- accuracy balance, achieving near-teacher performance with substantial reductions in model size and inference time. Despite the compression, the student model exceeds most existing frameworks, facilitating edge deployment for AMD screening. Code is available at https://github.com/erfan-nourbakhsh/KD- OCT.
>
---
#### [new 111] From Detection to Anticipation: Online Understanding of Struggles across Various Tasks and Activities
- **分类: cs.CV**

- **简介: 该论文研究在线挣扎检测与预测任务，旨在实时识别并提前预判用户在不同任务中的操作困难。作者将挣扎定位转为在线检测，并提出预期模型，验证了其在多任务间的泛化能力，兼顾效率与性能，适用于实时辅助系统。**

- **链接: [https://arxiv.org/pdf/2512.09847v1](https://arxiv.org/pdf/2512.09847v1)**

> **作者:** Shijia Feng; Michael Wray; Walterio Mayol-Cuevas
>
> **备注:** Accepted by WACV 2026
>
> **摘要:** Understanding human skill performance is essential for intelligent assistive systems, with struggle recognition offering a natural cue for identifying user difficulties. While prior work focuses on offline struggle classification and localization, real-time applications require models capable of detecting and anticipating struggle online. We reformulate struggle localization as an online detection task and further extend it to anticipation, predicting struggle moments before they occur. We adapt two off-the-shelf models as baselines for online struggle detection and anticipation. Online struggle detection achieves 70-80% per-frame mAP, while struggle anticipation up to 2 seconds ahead yields comparable performance with slight drops. We further examine generalization across tasks and activities and analyse the impact of skill evolution. Despite larger domain gaps in activity-level generalization, models still outperform random baselines by 4-20%. Our feature-based models run at up to 143 FPS, and the whole pipeline, including feature extraction, operates at around 20 FPS, sufficient for real-time assistive applications.
>
---
#### [new 112] MoRel: Long-Range Flicker-Free 4D Motion Modeling via Anchor Relay-based Bidirectional Blending with Hierarchical Densification
- **分类: cs.CV**

- **简介: 该论文属于动态场景建模任务，旨在解决4D高斯溅射中长程运动建模的内存爆炸、闪烁和遮挡问题。提出MoRel框架，通过锚点中继双向融合与分层稠密化，实现高效、连贯的长时序4D重建。**

- **链接: [https://arxiv.org/pdf/2512.09270v1](https://arxiv.org/pdf/2512.09270v1)**

> **作者:** Sangwoon Kwak; Weeyoung Kwon; Jun Young Jeong; Geonho Kim; Won-Sik Cheong; Jihyong Oh
>
> **备注:** Please visit our project page at https://cmlab-korea.github.io/MoRel/
>
> **摘要:** Recent advances in 4D Gaussian Splatting (4DGS) have extended the high-speed rendering capability of 3D Gaussian Splatting (3DGS) into the temporal domain, enabling real-time rendering of dynamic scenes. However, one of the major remaining challenges lies in modeling long-range motion-contained dynamic videos, where a naive extension of existing methods leads to severe memory explosion, temporal flickering, and failure to handle appearing or disappearing occlusions over time. To address these challenges, we propose a novel 4DGS framework characterized by an Anchor Relay-based Bidirectional Blending (ARBB) mechanism, named MoRel, which enables temporally consistent and memory-efficient modeling of long-range dynamic scenes. Our method progressively constructs locally canonical anchor spaces at key-frame time index and models inter-frame deformations at the anchor level, enhancing temporal coherence. By learning bidirectional deformations between KfA and adaptively blending them through learnable opacity control, our approach mitigates temporal discontinuities and flickering artifacts. We further introduce a Feature-variance-guided Hierarchical Densification (FHD) scheme that effectively densifies KfA's while keeping rendering quality, based on an assigned level of feature-variance. To effectively evaluate our model's capability to handle real-world long-range 4D motion, we newly compose long-range 4D motion-contained dataset, called SelfCap$_{\text{LR}}$. It has larger average dynamic motion magnitude, captured at spatially wider spaces, compared to previous dynamic video datasets. Overall, our MoRel achieves temporally coherent and flicker-free long-range 4D reconstruction while maintaining bounded memory usage, demonstrating both scalability and efficiency in dynamic Gaussian-based representations.
>
---
#### [new 113] StereoWorld: Geometry-Aware Monocular-to-Stereo Video Generation
- **分类: cs.CV**

- **简介: 该论文研究单目视频转立体视频任务，旨在解决高质量立体视频生成成本高、伪影多的问题。提出StereoWorld框架，利用预训练模型结合几何感知正则化与时空分块策略，实现高保真、几何一致的立体视频生成。**

- **链接: [https://arxiv.org/pdf/2512.09363v1](https://arxiv.org/pdf/2512.09363v1)**

> **作者:** Ke Xing; Longfei Li; Yuyang Yin; Hanwen Liang; Guixun Luo; Chen Fang; Jue Wang; Konstantinos N. Plataniotis; Xiaojie Jin; Yao Zhao; Yunchao Wei
>
> **摘要:** The growing adoption of XR devices has fueled strong demand for high-quality stereo video, yet its production remains costly and artifact-prone. To address this challenge, we present StereoWorld, an end-to-end framework that repurposes a pretrained video generator for high-fidelity monocular-to-stereo video generation. Our framework jointly conditions the model on the monocular video input while explicitly supervising the generation with a geometry-aware regularization to ensure 3D structural fidelity. A spatio-temporal tiling scheme is further integrated to enable efficient, high-resolution synthesis. To enable large-scale training and evaluation, we curate a high-definition stereo video dataset containing over 11M frames aligned to natural human interpupillary distance (IPD). Extensive experiments demonstrate that StereoWorld substantially outperforms prior methods, generating stereo videos with superior visual fidelity and geometric consistency. The project webpage is available at https://ke-xing.github.io/StereoWorld/.
>
---
#### [new 114] Agreement Disagreement Guided Knowledge Transfer for Cross-Scene Hyperspectral Imaging
- **分类: eess.IV; cs.CV**

- **简介: 该论文研究跨场景高光谱成像中的知识迁移，旨在解决梯度冲突与主导问题，并提升目标场景多样性特征的捕捉。提出ADGKT框架，结合梯度对齐、logit归一化与分歧约束，实现更鲁棒的跨场景知识转移。**

- **链接: [https://arxiv.org/pdf/2512.08990v1](https://arxiv.org/pdf/2512.08990v1)**

> **作者:** Lu Huo; Haimin Zhang; Min Xu
>
> **摘要:** Knowledge transfer plays a crucial role in cross-scene hyperspectral imaging (HSI). However, existing studies often overlook the challenges of gradient conflicts and dominant gradients that arise during the optimization of shared parameters. Moreover, many current approaches fail to simultaneously capture both agreement and disagreement information, relying only on a limited shared subset of target features and consequently missing the rich, diverse patterns present in the target scene. To address these issues, we propose an Agreement Disagreement Guided Knowledge Transfer (ADGKT) framework that integrates both mechanisms to enhance cross-scene transfer. The agreement component includes GradVac, which aligns gradient directions to mitigate conflicts between source and target domains, and LogitNorm, which regulates logit magnitudes to prevent domination by a single gradient source. The disagreement component consists of a Disagreement Restriction (DiR) and an ensemble strategy, which capture diverse predictive target features and mitigate the loss of critical target information. Extensive experiments demonstrate the effectiveness and superiority of the proposed method in achieving robust and balanced knowledge transfer across heterogeneous HSI scenes.
>
---
#### [new 115] Development and Testing for Perception Based Autonomous Landing of a Long-Range QuadPlane
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究无GPS环境下长航程QuadPlane的视觉自主降落。针对边缘算力受限、着陆环境非结构化等问题，提出轻量化的感知系统与视觉惯性里程计方案，实现复杂环境下的稳定自主降落。**

- **链接: [https://arxiv.org/pdf/2512.09343v1](https://arxiv.org/pdf/2512.09343v1)**

> **作者:** Ashik E Rasul; Humaira Tasnim; Ji Yu Kim; Young Hyun Lim; Scott Schmitz; Bruce W. Jo; Hyung-Jin Yoon
>
> **摘要:** QuadPlanes combine the range efficiency of fixed-wing aircraft with the maneuverability of multi-rotor platforms for long-range autonomous missions. In GPS-denied or cluttered urban environments, perception-based landing is vital for reliable operation. Unlike structured landing zones, real-world sites are unstructured and highly variable, requiring strong generalization capabilities from the perception system. Deep neural networks (DNNs) provide a scalable solution for learning landing site features across diverse visual and environmental conditions. While perception-driven landing has been shown in simulation, real-world deployment introduces significant challenges. Payload and volume constraints limit high-performance edge AI devices like the NVIDIA Jetson Orin Nano, which are crucial for real-time detection and control. Accurate pose estimation during descent is necessary, especially in the absence of GPS, and relies on dependable visual-inertial odometry. Achieving this with limited edge AI resources requires careful optimization of the entire deployment framework. The flight characteristics of large QuadPlanes further complicate the problem. These aircraft exhibit high inertia, reduced thrust vectoring, and slow response times further complicate stable landing maneuvers. This work presents a lightweight QuadPlane system for efficient vision-based autonomous landing and visual-inertial odometry, specifically developed for long-range QuadPlane operations such as aerial monitoring. It describes the hardware platform, sensor configuration, and embedded computing architecture designed to meet demanding real-time, physical constraints. This establishes a foundation for deploying autonomous landing in dynamic, unstructured, GPS-denied environments.
>
---
#### [new 116] DermETAS-SNA LLM: A Dermatology Focused Evolutionary Transformer Architecture Search with StackNet Augmented LLM Assistant
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文聚焦皮肤疾病分类与临床解释，提出DermETAS-SNA LLM助理。通过进化搜索优化视觉Transformer，构建StackNet增强分类鲁棒性，并结合RAG与大模型生成医学解释，在23类皮肤病上F1-score达56.30%，显著优于基线。**

- **链接: [https://arxiv.org/pdf/2512.08998v1](https://arxiv.org/pdf/2512.08998v1)**

> **作者:** Nitya Phani Santosh Oruganty; Keerthi Vemula Murali; Chun-Kit Ngan; Paulo Bandeira Pinho
>
> **摘要:** Our work introduces the DermETAS-SNA LLM Assistant that integrates Dermatology-focused Evolutionary Transformer Architecture Search with StackNet Augmented LLM. The assistant dynamically learns skin-disease classifiers and provides medically informed descriptions to facilitate clinician-patient interpretation. Contributions include: (1) Developed an ETAS framework on the SKINCON dataset to optimize a Vision Transformer (ViT) tailored for dermatological feature representation and then fine-tuned binary classifiers for each of the 23 skin disease categories in the DermNet dataset to enhance classification performance; (2) Designed a StackNet architecture that integrates multiple fine-tuned binary ViT classifiers to enhance predictive robustness and mitigate class imbalance issues; (3) Implemented a RAG pipeline, termed Diagnostic Explanation and Retrieval Model for Dermatology, which harnesses the capabilities of the Google Gemini 2.5 Pro LLM architecture to generate personalized, contextually informed diagnostic descriptions and explanations for patients, leveraging a repository of verified dermatological materials; (4) Performed extensive experimental evaluations on 23 skin disease categories to demonstrate performance increase, achieving an overall F1-score of 56.30% that surpasses SkinGPT-4 (48.51%) by a considerable margin, representing a performance increase of 16.06%; (5) Conducted a domain-expert evaluation, with eight licensed medical doctors, of the clinical responses generated by our AI assistant for seven dermatological conditions. Our results show a 92% agreement rate with the assessments provided by our AI assistant (6) Created a proof-of-concept prototype that fully integrates our DermETAS-SNA LLM into our AI assistant to demonstrate its practical feasibility for real-world clinical and educational applications.
>
---
#### [new 117] Enhanced Chest Disease Classification Using an Improved CheXNet Framework with EfficientNetV2-M and Optimization-Driven Learning
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在解决胸部X光片诊断中因放射科医生短缺导致的延迟问题。作者改进CheXNet框架，采用EfficientNetV2-M及优化训练策略，提升多类别胸病分类准确率与效率，实验证明其在COVID-19和结核病检测上性能接近完美。**

- **链接: [https://arxiv.org/pdf/2512.08992v1](https://arxiv.org/pdf/2512.08992v1)**

> **作者:** Ali M. Bahram; Saman Muhammad Omer; Hardi M. Mohammed; Sirwan Abdolwahed Aula
>
> **备注:** 23 pages, 6 figures, 7 tables
>
> **摘要:** The interpretation of Chest X-ray is an important diagnostic issue in clinical practice and especially in the resource-limited setting where the shortage of radiologists plays a role in delayed diagnosis and poor patient outcomes. Although the original CheXNet architecture has shown potential in automated analysis of chest radiographs, DenseNet-121 backbone is computationally inefficient and poorly single-label classifier. To eliminate such shortcomings, we suggest a better classification framework of chest disease that relies on EfficientNetV2-M and incorporates superior training approaches such as Automatic Mixed Precision training, AdamW, Cosine Annealing learning rate scheduling, and Exponential Moving Average regularization. We prepared a dataset of 18,080 chest X-ray images of three source materials of high authority and representing five key clinically significant disease categories which included Cardiomegaly, COVID-19, Normal, Pneumonia, and Tuberculosis. To achieve statistical reliability and reproducibility, nine independent experimental runs were run. The suggested architecture showed significant gains with mean test accuracy of 96.45 percent compared to 95.30 percent at baseline (p less than 0.001) and macro-averaged F1-score increased to 91.08 percent (p less than 0.001). Critical infectious diseases showed near-perfect classification performance with COVID-19 detection having 99.95 percent accuracy and Tuberculosis detection having 99.97 percent accuracy. Although 6.8 times more parameters are included, the training time was reduced by 11.4 percent and performance stability was increased by 22.7 percent. This framework presents itself as a decision-support tool that can be used to respond to a pandemic, screen tuberculosis, and assess thoracic disease regularly in various healthcare facilities.
>
---
#### [new 118] LiePrune: Lie Group and Quantum Geometric Dual Representation for One-Shot Structured Pruning of Quantum Neural Networks
- **分类: quant-ph; cs.CV**

- **简介: 该论文针对量子神经网络参数过多、可扩展性差的问题，提出LiePrune框架，首次结合李群与量子几何双表示，实现有理论保证的一次性结构化剪枝，在显著压缩模型的同时保持甚至提升性能。**

- **链接: [https://arxiv.org/pdf/2512.09469v1](https://arxiv.org/pdf/2512.09469v1)**

> **作者:** Haijian Shao; Bowen Yang; Wei Liu; Xing Deng; Yingtao Jiang
>
> **备注:** 7 pages, 2 figures
>
> **摘要:** Quantum neural networks (QNNs) and parameterized quantum circuits (PQCs) are key building blocks for near-term quantum machine learning. However, their scalability is constrained by excessive parameters, barren plateaus, and hardware limitations. We propose LiePrune, the first mathematically grounded one-shot structured pruning framework for QNNs that leverages Lie group structure and quantum geometric information. Each gate is jointly represented in a Lie group--Lie algebra dual space and a quantum geometric feature space, enabling principled redundancy detection and aggressive compression. Experiments on quantum classification (MNIST, FashionMNIST), quantum generative modeling (Bars-and-Stripes), and quantum chemistry (LiH VQE) show that LiePrune achieves over $10\times$ compression with negligible or even improved task performance, while providing provable guarantees on redundancy detection, functional approximation, and computational complexity.
>
---
#### [new 119] Visual Heading Prediction for Autonomous Aerial Vehicles
- **分类: cs.RO; cs.AI; cs.CV; cs.MA; eess.SY**

- **简介: 该论文研究无人机（UAV）与地面车（UGV）的视觉协同导航，解决无GPS环境下实时姿态对准问题。提出基于YOLOv5和轻量神经网络的框架，实现UGV检测与航向角预测，提升多智能体在复杂环境中的自主协同能力。**

- **链接: [https://arxiv.org/pdf/2512.09898v1](https://arxiv.org/pdf/2512.09898v1)**

> **作者:** Reza Ahmari; Ahmad Mohammadi; Vahid Hemmati; Mohammed Mynuddin; Parham Kebria; Mahmoud Nabil Mahmoud; Xiaohong Yuan; Abdollah Homaifar
>
> **摘要:** The integration of Unmanned Aerial Vehicles (UAVs) and Unmanned Ground Vehicles (UGVs) is increasingly central to the development of intelligent autonomous systems for applications such as search and rescue, environmental monitoring, and logistics. However, precise coordination between these platforms in real-time scenarios presents major challenges, particularly when external localization infrastructure such as GPS or GNSS is unavailable or degraded [1]. This paper proposes a vision-based, data-driven framework for real-time UAV-UGV integration, with a focus on robust UGV detection and heading angle prediction for navigation and coordination. The system employs a fine-tuned YOLOv5 model to detect UGVs and extract bounding box features, which are then used by a lightweight artificial neural network (ANN) to estimate the UAV's required heading angle. A VICON motion capture system was used to generate ground-truth data during training, resulting in a dataset of over 13,000 annotated images collected in a controlled lab environment. The trained ANN achieves a mean absolute error of 0.1506° and a root mean squared error of 0.1957°, offering accurate heading angle predictions using only monocular camera inputs. Experimental evaluations achieve 95% accuracy in UGV detection. This work contributes a vision-based, infrastructure- independent solution that demonstrates strong potential for deployment in GPS/GNSS-denied environments, supporting reliable multi-agent coordination under realistic dynamic conditions. A demonstration video showcasing the system's real-time performance, including UGV detection, heading angle prediction, and UAV alignment under dynamic conditions, is available at: https://github.com/Kooroshraf/UAV-UGV-Integration
>
---
#### [new 120] Causal Attribution of Model Performance Gaps in Medical Imaging Under Distribution Shifts
- **分类: eess.IV; cs.CV; cs.LG; stat.ME**

- **简介: 该论文研究医学图像分割中分布偏移导致的性能下降问题，属于医疗AI任务。通过因果归因框架，量化成像协议与标注差异对性能的影响，揭示不同场景下的主导因素，指导针对性改进。**

- **链接: [https://arxiv.org/pdf/2512.09094v1](https://arxiv.org/pdf/2512.09094v1)**

> **作者:** Pedro M. Gordaliza; Nataliia Molchanova; Jaume Banus; Thomas Sanchez; Meritxell Bach Cuadra
>
> **备注:** Medical Imaging meets EurIPS Workshop: MedEurIPS 2025
>
> **摘要:** Deep learning models for medical image segmentation suffer significant performance drops due to distribution shifts, but the causal mechanisms behind these drops remain poorly understood. We extend causal attribution frameworks to high-dimensional segmentation tasks, quantifying how acquisition protocols and annotation variability independently contribute to performance degradation. We model the data-generating process through a causal graph and employ Shapley values to fairly attribute performance changes to individual mechanisms. Our framework addresses unique challenges in medical imaging: high-dimensional outputs, limited samples, and complex mechanism interactions. Validation on multiple sclerosis (MS) lesion segmentation across 4 centers and 7 annotators reveals context-dependent failure modes: annotation protocol shifts dominate when crossing annotators (7.4% $\pm$ 8.9% DSC attribution), while acquisition shifts dominate when crossing imaging centers (6.5% $\pm$ 9.1%). This mechanism-specific quantification enables practitioners to prioritize targeted interventions based on deployment context.
>
---
#### [new 121] UrbanNav: Learning Language-Guided Urban Navigation from Web-Scale Human Trajectories
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究语言引导的城市导航任务，旨在解决复杂城市环境中基于自然语言指令的自主导航问题。作者提出UrbanNav框架，利用大规模网络视频数据构建真实世界导航数据集，实现对模糊语言、动态场景等挑战的有效应对，并提升模型在未见城市环境中的泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.09607v1](https://arxiv.org/pdf/2512.09607v1)**

> **作者:** Yanghong Mei; Yirong Yang; Longteng Guo; Qunbo Wang; Ming-Ming Yu; Xingjian He; Wenjun Wu; Jing Liu
>
> **备注:** 9 pages, 5 figures, accepted to AAAI 2026
>
> **摘要:** Navigating complex urban environments using natural language instructions poses significant challenges for embodied agents, including noisy language instructions, ambiguous spatial references, diverse landmarks, and dynamic street scenes. Current visual navigation methods are typically limited to simulated or off-street environments, and often rely on precise goal formats, such as specific coordinates or images. This limits their effectiveness for autonomous agents like last-mile delivery robots navigating unfamiliar cities. To address these limitations, we introduce UrbanNav, a scalable framework that trains embodied agents to follow free-form language instructions in diverse urban settings. Leveraging web-scale city walking videos, we develop an scalable annotation pipeline that aligns human navigation trajectories with language instructions grounded in real-world landmarks. UrbanNav encompasses over 1,500 hours of navigation data and 3 million instruction-trajectory-landmark triplets, capturing a wide range of urban scenarios. Our model learns robust navigation policies to tackle complex urban scenarios, demonstrating superior spatial reasoning, robustness to noisy instructions, and generalization to unseen urban settings. Experimental results show that UrbanNav significantly outperforms existing methods, highlighting the potential of large-scale web video data to enable language-guided, real-world urban navigation for embodied agents.
>
---
#### [new 122] H2R-Grounder: A Paired-Data-Free Paradigm for Translating Human Interaction Videos into Physically Grounded Robot Videos
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文研究从人类操作视频生成机器人操作视频的无配对数据方法。针对缺乏配对数据的问题，提出H2R-Grounder框架，通过可迁移表征和视觉提示，实现人体到机器人动作的转换，并利用视频扩散模型保证时序连贯性，生成物理合理的机器人操作视频。**

- **链接: [https://arxiv.org/pdf/2512.09406v1](https://arxiv.org/pdf/2512.09406v1)**

> **作者:** Hai Ci; Xiaokang Liu; Pei Yang; Yiren Song; Mike Zheng Shou
>
> **备注:** 13 pages, 6 figures
>
> **摘要:** Robots that learn manipulation skills from everyday human videos could acquire broad capabilities without tedious robot data collection. We propose a video-to-video translation framework that converts ordinary human-object interaction videos into motion-consistent robot manipulation videos with realistic, physically grounded interactions. Our approach does not require any paired human-robot videos for training only a set of unpaired robot videos, making the system easy to scale. We introduce a transferable representation that bridges the embodiment gap: by inpainting the robot arm in training videos to obtain a clean background and overlaying a simple visual cue (a marker and arrow indicating the gripper's position and orientation), we can condition a generative model to insert the robot arm back into the scene. At test time, we apply the same process to human videos (inpainting the person and overlaying human pose cues) and generate high-quality robot videos that mimic the human's actions. We fine-tune a SOTA video diffusion model (Wan 2.2) in an in-context learning manner to ensure temporal coherence and leveraging of its rich prior knowledge. Empirical results demonstrate that our approach achieves significantly more realistic and grounded robot motions compared to baselines, pointing to a promising direction for scaling up robot learning from unlabeled human videos. Project page: https://showlab.github.io/H2R-Grounder/
>
---
#### [new 123] SynthPix: A lightspeed PIV images generator
- **分类: cs.DC; cs.CV; cs.LG; eess.IV**

- **简介: 该论文提出SynthPix，一种基于JAX的PIV合成图像生成工具，旨在加速粒子图像测速数据生成。针对传统方法速度慢、难以满足强化学习训练需求的问题，实现了高性能并行化，显著提升图像对生成吞吐量，支持快速开发实时流体估计算法。**

- **链接: [https://arxiv.org/pdf/2512.09664v1](https://arxiv.org/pdf/2512.09664v1)**

> **作者:** Antonio Terpin; Alan Bonomi; Francesco Banelli; Raffaello D'Andrea
>
> **备注:** Code: https://github.com/antonioterpin/synthpix
>
> **摘要:** We describe SynthPix, a synthetic image generator for Particle Image Velocimetry (PIV) with a focus on performance and parallelism on accelerators, implemented in JAX. SynthPix supports the same configuration parameters as existing tools but achieves a throughput several orders of magnitude higher in image-pair generation per second. SynthPix was developed to enable the training of data-hungry reinforcement learning methods for flow estimation and for reducing the iteration times during the development of fast flow estimation methods used in recent active fluids control studies with real-time PIV feedback. We believe SynthPix to be useful for the fluid dynamics community, and in this paper we describe the main ideas behind this software package.
>
---
#### [new 124] Visual Categorization Across Minds and Models: Cognitive Analysis of Human Labeling and Neuro-Symbolic Integration
- **分类: cs.AI; cs.CV; cs.LG**

- **简介: 该论文研究人类与AI在识别模糊图像时的异同，旨在揭示认知机制差异。通过对比人类行为与神经网络注意力，结合认知模型分析，提出融合符号推理与连接主义的神经-符号架构，提升AI的可解释性与认知对齐。**

- **链接: [https://arxiv.org/pdf/2512.09340v1](https://arxiv.org/pdf/2512.09340v1)**

> **作者:** Chethana Prasad Kabgere
>
> **备注:** 12 pages, 3 figures. Research manuscript based on the final project for CS6795 (Introduction to Cognitive Science), Georgia Tech
>
> **摘要:** Understanding how humans and AI systems interpret ambiguous visual stimuli offers critical insight into the nature of perception, reasoning, and decision-making. This paper examines image labeling performance across human participants and deep neural networks, focusing on low-resolution, perceptually degraded stimuli. Drawing from computational cognitive science, cognitive architectures, and connectionist-symbolic hybrid models, we contrast human strategies such as analogical reasoning, shape-based recognition, and confidence modulation with AI's feature-based processing. Grounded in Marr's tri-level hypothesis, Simon's bounded rationality, and Thagard's frameworks of representation and emotion, we analyze participant responses in relation to Grad-CAM visualizations of model attention. Human behavior is further interpreted through cognitive principles modeled in ACT-R and Soar, revealing layered and heuristic decision strategies under uncertainty. Our findings highlight key parallels and divergences between biological and artificial systems in representation, inference, and confidence calibration. The analysis motivates future neuro-symbolic architectures that unify structured symbolic reasoning with connectionist representations. Such architectures, informed by principles of embodiment, explainability, and cognitive alignment, offer a path toward AI systems that are not only performant but also interpretable and cognitively grounded.
>
---
#### [new 125] YOPO-Nav: Visual Navigation using 3DGS Graphs from One-Pass Videos
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究视觉导航任务，旨在不依赖精确建图的情况下实现机器人路径重现。提出YOPO-Nav方法，利用单次视频构建基于3D高斯点阵的紧凑空间图，结合视觉定位与局部三维匹配进行导航。在自建YOPO-Campus数据集上验证了有效性。**

- **链接: [https://arxiv.org/pdf/2512.09903v1](https://arxiv.org/pdf/2512.09903v1)**

> **作者:** Ryan Meegan; Adam D'Souza; Bryan Bo Cao; Shubham Jain; Kristin Dana
>
> **摘要:** Visual navigation has emerged as a practical alternative to traditional robotic navigation pipelines that rely on detailed mapping and path planning. However, constructing and maintaining 3D maps is often computationally expensive and memory-intensive. We address the problem of visual navigation when exploration videos of a large environment are available. The videos serve as a visual reference, allowing a robot to retrace the explored trajectories without relying on metric maps. Our proposed method, YOPO-Nav (You Only Pass Once), encodes an environment into a compact spatial representation composed of interconnected local 3D Gaussian Splatting (3DGS) models. During navigation, the framework aligns the robot's current visual observation with this representation and predicts actions that guide it back toward the demonstrated trajectory. YOPO-Nav employs a hierarchical design: a visual place recognition (VPR) module provides coarse localization, while the local 3DGS models refine the goal and intermediate poses to generate control actions. To evaluate our approach, we introduce the YOPO-Campus dataset, comprising 4 hours of egocentric video and robot controller inputs from over 6 km of human-teleoperated robot trajectories. We benchmark recent visual navigation methods on trajectories from YOPO-Campus using a Clearpath Jackal robot. Experimental results show YOPO-Nav provides excellent performance in image-goal navigation for real-world scenes on a physical robot. The dataset and code will be made publicly available for visual navigation and scene representation research.
>
---
#### [new 126] Rates and architectures for learning geometrically non-trivial operators
- **分类: cs.LG; cs.CV; eess.IV; math.DG**

- **简介: 该论文研究科学机器学习中几何结构复杂的算子学习问题，旨在解决非平凡几何算子（如双纤维化变换）的高效学习。作者提出能显式编码几何结构的网络架构，证明其可超代数收敛地学习此类算子，且仅需少量训练样本。**

- **链接: [https://arxiv.org/pdf/2512.09376v1](https://arxiv.org/pdf/2512.09376v1)**

> **作者:** T. Mitchell Roddenberry; Leo Tzou; Ivan Dokmanić; Maarten V. de Hoop; Richard G. Baraniuk
>
> **备注:** 26 pages, 5 figures
>
> **摘要:** Deep learning methods have proven capable of recovering operators between high-dimensional spaces, such as solution maps of PDEs and similar objects in mathematical physics, from very few training samples. This phenomenon of data-efficiency has been proven for certain classes of elliptic operators with simple geometry, i.e., operators that do not change the domain of the function or propagate singularities. However, scientific machine learning is commonly used for problems that do involve the propagation of singularities in a priori unknown ways, such as waves, advection, and fluid dynamics. In light of this, we expand the learning theory to include double fibration transforms--geometric integral operators that include generalized Radon and geodesic ray transforms. We prove that this class of operators does not suffer from the curse of dimensionality: the error decays superalgebraically, that is, faster than any fixed power of the reciprocal of the number of training samples. Furthermore, we investigate architectures that explicitly encode the geometry of these transforms, demonstrating that an architecture reminiscent of cross-attention based on levelset methods yields a parameterization that is universal, stable, and learns double fibration transforms from very few training examples. Our results contribute to a rapidly-growing line of theoretical work on learning operators for scientific machine learning.
>
---
#### [new 127] Simultaneous Tactile-Visual Perception for Learning Multimodal Robot Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究多模态机器人操作，解决现有见皮传感器无法同步触觉-视觉感知及触觉追踪不可靠的问题。提出TacThru传感器与TacThru-UMI学习框架，实现同步感知与高效操作，显著提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2512.09851v1](https://arxiv.org/pdf/2512.09851v1)**

> **作者:** Yuyang Li; Yinghan Chen; Zihang Zhao; Puhao Li; Tengyu Liu; Siyuan Huang; Yixin Zhu
>
> **摘要:** Robotic manipulation requires both rich multimodal perception and effective learning frameworks to handle complex real-world tasks. See-through-skin (STS) sensors, which combine tactile and visual perception, offer promising sensing capabilities, while modern imitation learning provides powerful tools for policy acquisition. However, existing STS designs lack simultaneous multimodal perception and suffer from unreliable tactile tracking. Furthermore, integrating these rich multimodal signals into learning-based manipulation pipelines remains an open challenge. We introduce TacThru, an STS sensor enabling simultaneous visual perception and robust tactile signal extraction, and TacThru-UMI, an imitation learning framework that leverages these multimodal signals for manipulation. Our sensor features a fully transparent elastomer, persistent illumination, novel keyline markers, and efficient tracking, while our learning system integrates these signals through a Transformer-based Diffusion Policy. Experiments on five challenging real-world tasks show that TacThru-UMI achieves an average success rate of 85.5%, significantly outperforming the baselines of alternating tactile-visual (66.3%) and vision-only (55.4%). The system excels in critical scenarios, including contact detection with thin and soft objects and precision manipulation requiring multimodal coordination. This work demonstrates that combining simultaneous multimodal perception with modern learning frameworks enables more precise, adaptable robotic manipulation.
>
---
#### [new 128] Sequential Testing for Descriptor-Agnostic LiDAR Loop Closure in Repetitive Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究LiDAR回环检测任务，旨在解决重复结构环境中误匹配问题。提出一种无需依赖描述子类型、基于序贯概率比检验的多帧验证方法，通过时序累积相似性动态决策，提升精度并抑制假阳性。**

- **链接: [https://arxiv.org/pdf/2512.09447v1](https://arxiv.org/pdf/2512.09447v1)**

> **作者:** Jaehyun Kim; Seungwon Choi; Tae-Wan Kim
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** We propose a descriptor-agnostic, multi-frame loop closure verification method that formulates LiDAR loop closure as a truncated Sequential Probability Ratio Test (SPRT). Instead of deciding from a single descriptor comparison or using fixed thresholds with late-stage Iterative Closest Point (ICP) vetting, the verifier accumulates a short temporal stream of descriptor similarities between a query and each candidate. It then issues an accept/reject decision adaptively once sufficient multi-frame evidence has been observed, according to user-specified Type-I/II error design targets. This precision-first policy is designed to suppress false positives in structurally repetitive indoor environments. We evaluate the verifier on a five-sequence library dataset, using a fixed retrieval front-end with several representative LiDAR global descriptors. Performance is assessed via segment-level K-hit precision-recall and absolute trajectory error (ATE) and relative pose error (RPE) after pose graph optimization. Across descriptors, the sequential verifier consistently improves precision and reduces the impact of aliased loops compared with single-frame and heuristic multi-frame baselines. Our implementation and dataset will be released at: https://github.com/wanderingcar/snu_library_dataset.
>
---
#### [new 129] A Distributed Framework for Privacy-Enhanced Vision Transformers on the Edge
- **分类: cs.DC; cs.CR; cs.CV**

- **简介: 该论文提出一种面向边缘的分布式视觉Transformer框架，解决视觉模型在云端推理时的隐私泄露问题。通过在可信边缘设备上分割数据并分发至多云服务器，避免完整图像外泄，最终在边缘聚合结果，兼顾隐私与性能。**

- **链接: [https://arxiv.org/pdf/2512.09309v1](https://arxiv.org/pdf/2512.09309v1)**

> **作者:** Zihao Ding; Mufeng Zhu; Zhongze Tang; Sheng Wei; Yao Liu
>
> **备注:** 16 pages, 7 figures. Published in the Proceedings of the Tenth ACM/IEEE Symposium on Edge Computing (SEC '25), Dec 3-6, 2025, Washington, D.C., USA
>
> **摘要:** Nowadays, visual intelligence tools have become ubiquitous, offering all kinds of convenience and possibilities. However, these tools have high computational requirements that exceed the capabilities of resource-constrained mobile and wearable devices. While offloading visual data to the cloud is a common solution, it introduces significant privacy vulnerabilities during transmission and server-side computation. To address this, we propose a novel distributed, hierarchical offloading framework for Vision Transformers (ViTs) that addresses these privacy challenges by design. Our approach uses a local trusted edge device, such as a mobile phone or an Nvidia Jetson, as the edge orchestrator. This orchestrator partitions the user's visual data into smaller portions and distributes them across multiple independent cloud servers. By design, no single external server possesses the complete image, preventing comprehensive data reconstruction. The final data merging and aggregation computation occurs exclusively on the user's trusted edge device. We apply our framework to the Segment Anything Model (SAM) as a practical case study, which demonstrates that our method substantially enhances content privacy over traditional cloud-based approaches. Evaluations show our framework maintains near-baseline segmentation performance while substantially reducing the risk of content reconstruction and user data exposure. Our framework provides a scalable, privacy-preserving solution for vision tasks in the edge-cloud continuum.
>
---
#### [new 130] ImageTalk: Designing a Multimodal AAC Text Generation System Driven by Image Recognition and Natural Language Generation
- **分类: cs.HC; cs.AI; cs.CV**

- **简介: 该论文属于人机交互与辅助技术任务，旨在解决运动神经元病患者因言语和运动障碍导致的沟通困难。提出并开发了基于图像识别与自然语言生成的多模态AAC系统ImageTalk，显著提升沟通效率与用户满意度。**

- **链接: [https://arxiv.org/pdf/2512.09610v1](https://arxiv.org/pdf/2512.09610v1)**

> **作者:** Boyin Yang; Puming Jiang; Per Ola Kristensson
>
> **备注:** 24 pages, 10 figures
>
> **摘要:** People living with Motor Neuron Disease (plwMND) frequently encounter speech and motor impairments that necessitate a reliance on augmentative and alternative communication (AAC) systems. This paper tackles the main challenge that traditional symbol-based AAC systems offer a limited vocabulary, while text entry solutions tend to exhibit low communication rates. To help plwMND articulate their needs about the system efficiently and effectively, we iteratively design and develop a novel multimodal text generation system called ImageTalk through a tailored proxy-user-based and an end-user-based design phase. The system demonstrates pronounced keystroke savings of 95.6%, coupled with consistent performance and high user satisfaction. We distill three design guidelines for AI-assisted text generation systems design and outline four user requirement levels tailored for AAC purposes, guiding future research in this field.
>
---
#### [new 131] PathCo-LatticE: Pathology-Constrained Lattice-Of Experts Framework for Fully-supervised Few-Shot Cardiac MRI Segmentation
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文针对心脏MRI分割中少样本学习的域迁移和验证偏差问题，提出全监督框架PathCo-LatticE。通过虚拟患者引擎生成带病理引导的合成数据，结合自强化验证与专家 lattice 模型，实现跨中心、跨厂商的零样本泛化。**

- **链接: [https://arxiv.org/pdf/2512.09779v1](https://arxiv.org/pdf/2512.09779v1)**

> **作者:** Mohamed Elbayumi; Mohammed S. M. Elbaz
>
> **摘要:** Few-shot learning (FSL) mitigates data scarcity in cardiac MRI segmentation but typically relies on semi-supervised techniques sensitive to domain shifts and validation bias, restricting zero-shot generalizability. We propose PathCo-LatticE, a fully supervised FSL framework that replaces unlabeled data with pathology-guided synthetic supervision. First, our Virtual Patient Engine models continuous latent disease trajectories from sparse clinical anchors, using generative modeling to synthesize physiologically plausible, fully labeled 3D cohorts. Second, Self-Reinforcing Interleaved Validation (SIV) provides a leakage-free protocol that evaluates models online with progressively challenging synthetic samples, eliminating the need for real validation data. Finally, a dynamic Lattice-of-Experts (LoE) organizes specialized networks within a pathology-aware topology and activates the most relevant experts per input, enabling robust zero-shot generalization to unseen data without target-domain fine-tuning. We evaluated PathCo-LatticE in a strict out-of-distribution (OOD) setting, deriving all anchors and severity statistics from a single-source domain (ACDC) and performing zero-shot testing on the multi-center, multi-vendor M&Ms dataset. PathCo-LatticE outperforms four state-of-the-art FSL methods by 4.2-11% Dice starting from only 7 labeled anchors, and approaches fully supervised performance (within 1% Dice) with only 19 labeled anchors. The method shows superior harmonization across four vendors and generalization to unseen pathologies. [Code will be made publicly available].
>
---
#### [new 132] ViTA-Seg: Vision Transformer for Amodal Segmentation in Robotics
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究机器人抓取中的非遮挡区域分割任务，旨在解决物体遮挡导致的抓取规划难题。提出ViTA-Seg框架与合成数据集ViTA-SimData，实现高效准确的实时全貌分割。**

- **链接: [https://arxiv.org/pdf/2512.09510v1](https://arxiv.org/pdf/2512.09510v1)**

> **作者:** Donato Caramia; Florian T. Pokorny; Giuseppe Triggiani; Denis Ruffino; David Naso; Paolo Roberto Massenio
>
> **摘要:** Occlusions in robotic bin picking compromise accurate and reliable grasp planning. We present ViTA-Seg, a class-agnostic Vision Transformer framework for real-time amodal segmentation that leverages global attention to recover complete object masks, including hidden regions. We proposte two architectures: a) Single-Head for amodal mask prediction; b) Dual-Head for amodal and occluded mask prediction. We also introduce ViTA-SimData, a photo-realistic synthetic dataset tailored to industrial bin-picking scenario. Extensive experiments on two amodal benchmarks, COOCA and KINS, demonstrate that ViTA-Seg Dual Head achieves strong amodal and occlusion segmentation accuracy with computational efficiency, enabling robust, real-time robotic manipulation.
>
---
#### [new 133] Residual Primitive Fitting of 3D Shapes with SuperFrusta
- **分类: cs.GR; cs.CV**

- **简介: 该论文属3D形状重建任务，旨在平衡重构精度与简洁性。提出SuperFrustum新基元和残差基元拟合算法ResFit，实现高保真、少基元的可编辑形状分解，显著提升IoU并减少基元数量。**

- **链接: [https://arxiv.org/pdf/2512.09201v1](https://arxiv.org/pdf/2512.09201v1)**

> **作者:** Aditya Ganeshan; Matheus Gadelha; Thibault Groueix; Zhiqin Chen; Siddhartha Chaudhuri; Vladimir Kim; Wang Yifan; Daniel Ritchie
>
> **备注:** https://bardofcodes.github.io/superfit/
>
> **摘要:** We introduce a framework for converting 3D shapes into compact and editable assemblies of analytic primitives, directly addressing the persistent trade-off between reconstruction fidelity and parsimony. Our approach combines two key contributions: a novel primitive, termed SuperFrustum, and an iterative fiting algorithm, Residual Primitive Fitting (ResFit). SuperFrustum is an analytical primitive that is simultaneously (1) expressive, being able to model various common solids such as cylinders, spheres, cones & their tapered and bent forms, (2) editable, being compactly parameterized with 8 parameters, and (3) optimizable, with a sign distance field differentiable w.r.t. its parameters almost everywhere. ResFit is an unsupervised procedure that interleaves global shape analysis with local optimization, iteratively fitting primitives to the unexplained residual of a shape to discover a parsimonious yet accurate decompositions for each input shape. On diverse 3D benchmarks, our method achieves state-of-the-art results, improving IoU by over 9 points while using nearly half as many primitives as prior work. The resulting assemblies bridge the gap between dense 3D data and human-controllable design, producing high-fidelity and editable shape programs.
>
---
#### [new 134] LISN: Language-Instructed Social Navigation with VLM-based Controller Modulating
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文研究语言指令下的社交导航任务，旨在让机器人遵循人类指令并遵守社会规范。作者构建了首个包含指令跟随的仿真基准LISN-Bench，并提出基于视觉语言模型的分层控制方法，显著提升导航成功率与适应性。**

- **链接: [https://arxiv.org/pdf/2512.09920v1](https://arxiv.org/pdf/2512.09920v1)**

> **作者:** Junting Chen; Yunchuan Li; Panfeng Jiang; Jiacheng Du; Zixuan Chen; Chenrui Tie; Jiajun Deng; Lin Shao
>
> **备注:** 8 pages
>
> **摘要:** Towards human-robot coexistence, socially aware navigation is significant for mobile robots. Yet existing studies on this area focus mainly on path efficiency and pedestrian collision avoidance, which are essential but represent only a fraction of social navigation. Beyond these basics, robots must also comply with user instructions, aligning their actions to task goals and social norms expressed by humans. In this work, we present LISN-Bench, the first simulation-based benchmark for language-instructed social navigation. Built on Rosnav-Arena 3.0, it is the first standardized social navigation benchmark to incorporate instruction following and scene understanding across diverse contexts. To address this task, we further propose Social-Nav-Modulator, a fast-slow hierarchical system where a VLM agent modulates costmaps and controller parameters. Decoupling low-level action generation from the slower VLM loop reduces reliance on high-frequency VLM inference while improving dynamic avoidance and perception adaptability. Our method achieves an average success rate of 91.3%, which is greater than 63% than the most competitive baseline, with most of the improvements observed in challenging tasks such as following a person in a crowd and navigating while strictly avoiding instruction-forbidden regions. The project website is at: https://social-nav.github.io/LISN-project/
>
---
#### [new 135] ChronusOmni: Improving Time Awareness of Omni Large Language Models
- **分类: cs.CL; cs.CV; cs.MM**

- **简介: 该论文聚焦多模态大模型的时间感知任务，解决现有方法在音频利用和跨模态隐式时序定位上的不足。提出ChronusOmni模型，通过文本化时间戳和强化学习提升显式与隐式音视频时序理解，并构建新数据集ChronusAV验证其优越性。**

- **链接: [https://arxiv.org/pdf/2512.09841v1](https://arxiv.org/pdf/2512.09841v1)**

> **作者:** Yijing Chen; Yihan Wu; Kaisi Guan; Yuchen Ren; Yuyue Wang; Ruihua Song; Liyun Ru
>
> **备注:** Code available at https://github.com/YJCX330/Chronus/
>
> **摘要:** Time awareness is a fundamental ability of omni large language models, especially for understanding long videos and answering complex questions. Previous approaches mainly target vision-language scenarios and focus on the explicit temporal grounding questions, such as identifying when a visual event occurs or determining what event happens at aspecific time. However, they often make insufficient use of the audio modality, and overlook implicit temporal grounding across modalities--for example, identifying what is visually present when a character speaks, or determining what is said when a visual event occurs--despite such cross-modal temporal relations being prevalent in real-world scenarios. In this paper, we propose ChronusOmni, an omni large language model designed to enhance temporal awareness for both explicit and implicit audiovisual temporal grounding. First, we interleave text-based timestamp tokens with visual and audio representations at each time unit, enabling unified temporal modeling across modalities. Second, to enforce correct temporal ordering and strengthen fine-grained temporal reasoning, we incorporate reinforcement learning with specially designed reward functions. Moreover, we construct ChronusAV, a temporally-accurate, modality-complete, and cross-modal-aligned dataset to support the training and evaluation on audiovisual temporal grounding task. Experimental results demonstrate that ChronusOmni achieves state-of-the-art performance on ChronusAV with more than 30% improvement and top results on most metrics upon other temporal grounding benchmarks. This highlights the strong temporal awareness of our model across modalities, while preserving general video and audio understanding capabilities.
>
---
## 更新

#### [replaced 001] TAViS: Text-bridged Audio-Visual Segmentation with Foundation Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.11436v2](https://arxiv.org/pdf/2506.11436v2)**

> **作者:** Ziyang Luo; Nian Liu; Xuguang Yang; Salman Khan; Rao Muhammad Anwer; Hisham Cholakkal; Fahad Shahbaz Khan; Junwei Han
>
> **备注:** ICCV2025,code:https://github.com/Sssssuperior/TAViS
>
> **摘要:** Audio-Visual Segmentation (AVS) faces a fundamental challenge of effectively aligning audio and visual modalities. While recent approaches leverage foundation models to address data scarcity, they often rely on single-modality knowledge or combine foundation models in an off-the-shelf manner, failing to address the cross-modal alignment challenge. In this paper, we present TAViS, a novel framework that \textbf{couples} the knowledge of multimodal foundation models (ImageBind) for cross-modal alignment and a segmentation foundation model (SAM2) for precise segmentation. However, effectively combining these models poses two key challenges: the difficulty in transferring the knowledge between SAM2 and ImageBind due to their different feature spaces, and the insufficiency of using only segmentation loss for supervision. To address these challenges, we introduce a text-bridged design with two key components: (1) a text-bridged hybrid prompting mechanism where pseudo text provides class prototype information while retaining modality-specific details from both audio and visual inputs, and (2) an alignment supervision strategy that leverages text as a bridge to align shared semantic concepts within audio-visual modalities. Our approach achieves superior performance on single-source, multi-source, semantic datasets, and excels in zero-shot settings.
>
---
#### [replaced 002] ConsDreamer: Advancing Multi-View Consistency for Zero-Shot Text-to-3D Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2504.02316v3](https://arxiv.org/pdf/2504.02316v3)**

> **作者:** Yuan Zhou; Shilong Jin; Litao Hua; Wanjun Lv; Haoran Duan; Jungong Han
>
> **备注:** 13 pages, 14 figures, 3 tables
>
> **摘要:** Recent advances in zero-shot text-to-3D generation have revolutionized 3D content creation by enabling direct synthesis from textual descriptions. While state-of-the-art methods leverage 3D Gaussian Splatting with score distillation to enhance multi-view rendering through pre-trained text-to-image (T2I) models, they suffer from inherent prior view biases in T2I priors. These biases lead to inconsistent 3D generation, particularly manifesting as the multi-face Janus problem, where objects exhibit conflicting features across views. To address this fundamental challenge, we propose ConsDreamer, a novel method that mitigates view bias by refining both the conditional and unconditional terms in the score distillation process: (1) a View Disentanglement Module (VDM) that eliminates viewpoint biases in conditional prompts by decoupling irrelevant view components and injecting precise view control; and (2) a similarity-based partial order loss that enforces geometric consistency in the unconditional term by aligning cosine similarities with azimuth relationships. Extensive experiments demonstrate that ConsDreamer can be seamlessly integrated into various 3D representations and score distillation paradigms, effectively mitigating the multi-face Janus problem.
>
---
#### [replaced 003] More than Segmentation: Benchmarking SAM 3 for Segmentation, 3D Perception, and Reconstruction in Robotic Surgery
- **分类: cs.CV; cs.RO**

- **简介: 该论文评估SAM 3在机器人手术中的分割、3D感知与重建能力，属医学图像分析任务。旨在验证其在零样本分割、视频跟踪及深度重建中的表现，发现其在动态手术场景中仍有局限，需进一步优化。**

- **链接: [https://arxiv.org/pdf/2512.07596v2](https://arxiv.org/pdf/2512.07596v2)**

> **作者:** Wenzhen Dong; Jieming Yu; Yiming Huang; Hongqiu Wang; Lei Zhu; Albert C. S. Chung; Hongliang Ren; Long Bai
>
> **备注:** Technical Report
>
> **摘要:** The recent SAM 3 and SAM 3D have introduced significant advancements over the predecessor, SAM 2, particularly with the integration of language-based segmentation and enhanced 3D perception capabilities. SAM 3 supports zero-shot segmentation across a wide range of prompts, including point, bounding box, and language-based prompts, allowing for more flexible and intuitive interactions with the model. In this empirical evaluation, we assess the performance of SAM 3 in robot-assisted surgery, benchmarking its zero-shot segmentation with point and bounding box prompts and exploring its effectiveness in dynamic video tracking, alongside its newly introduced language prompt segmentation. While language prompts show potential, their performance in the surgical domain is currently suboptimal, highlighting the need for further domain-specific training. Additionally, we investigate SAM 3D's depth reconstruction abilities, demonstrating its capacity to process surgical scene data and reconstruct 3D anatomical structures from 2D images. Through comprehensive testing on the MICCAI EndoVis 2017 and EndoVis 2018 benchmarks, SAM 3 shows clear improvements over SAM and SAM 2 in both image and video segmentation under spatial prompts, while the zero-shot evaluations of SAM 3D on SCARED, StereoMIS, and EndoNeRF indicate strong monocular depth estimation and realistic 3D instrument reconstruction, yet also reveal remaining limitations in complex, highly dynamic surgical scenes.
>
---
#### [replaced 004] ICM-SR: Image-Conditioned Manifold Regularization for Image Super-Resoultion
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.22048v2](https://arxiv.org/pdf/2511.22048v2)**

> **作者:** Junoh Kang; Donghun Ryou; Bohyung Han
>
> **摘要:** Real world image super-resolution (Real-ISR) often leverages the powerful generative priors of text-to-image diffusion models by regularizing the output to lie on their learned manifold. However, existing methods often overlook the importance of the regularizing manifold, typically defaulting to a text-conditioned manifold. This approach suffers from two key limitations. Conceptually, it is misaligned with the Real-ISR task, which is to generate high quality (HQ) images directly tied to the low quality (LQ) images. Practically, the teacher model often reconstructs images with color distortions and blurred edges, indicating a flawed generative prior for this task. To correct these flaws and ensure conceptual alignment, a more suitable manifold must incorporate information from the images. While the most straightforward approach is to condition directly on the raw input images, their high information densities make the regularization process numerically unstable. To resolve this, we propose image-conditioned manifold regularization (ICM), a method that regularizes the output towards a manifold conditioned on the sparse yet essential structural information: a combination of colormap and Canny edges. ICM provides a task-aligned and stable regularization signal, thereby avoiding the instability of dense-conditioning and enhancing the final super-resolution quality. Our experiments confirm that the proposed regularization significantly enhances super-resolution performance, particularly in perceptual quality, demonstrating its effectiveness for real-world applications. We will release the source code of our work for reproducibility.
>
---
#### [replaced 005] TranSplat: Instant Cross-Scene Object Relighting in Gaussian Splatting via Spherical Harmonic Transfer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.22676v4](https://arxiv.org/pdf/2503.22676v4)**

> **作者:** Boyang Yu; Yanlin Jin; Yun He; Akshat Dave; Guha Balakrishnan
>
> **摘要:** We present TranSplat, a method for fast and accurate object relighting for the 3D Gaussian Splatting (GS) framework when transferring a 3D object from a source GS scene to a target GS scene. TranSplat is based on a theoretical radiance transfer identity for cross-scene relighting of objects with radially symmetric BRDFs that involves only taking simple products of spherical harmonic appearance coefficients of the object, source, and target environment maps without any explicit computation of scene quantities (e.g., the BRDFs themselves). TranSplat is the first method to demonstrate how this theoretical identity may be used to perform relighting within the GS framework, and furthermore, by automatically inferring unknown source and target environment maps directly from the source and target scene GS representations. We evaluated TranSplat on several synthetic and real-world scenes and objects, demonstrating comparable 3D object relighting performance to recent conventional inverse rendering-based GS methods with a fraction of their runtime. While TranSplat is theoretically best-suited for radially symmetric BRDFs, results demonstrate that TranSplat still offers perceptually realistic renderings on real scenes and opens a valuable, lightweight path forward to relighting with the GS framework.
>
---
#### [replaced 006] Multi-Scale Direction-Aware Network for Infrared Small Target Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2406.02037v4](https://arxiv.org/pdf/2406.02037v4)**

> **作者:** Jinmiao Zhao; Zelin Shi; Chuang Yu; Yunpeng Liu; Xinyi Ying; Yimian Dai
>
> **备注:** Accepted by TGRS 2025
>
> **摘要:** Infrared small target detection faces the problem that it is difficult to effectively separate the background and the target. Existing deep learning-based methods focus on edge and shape features, but ignore the richer structural differences and detailed information embedded in high-frequency components from different directions, thereby failing to fully exploit the value of high-frequency directional features in target perception. To address this limitation, we propose a multi-scale direction-aware network (MSDA-Net), which is the first attempt to integrate the high-frequency directional features of infrared small targets as domain prior knowledge into neural networks. Specifically, to fully mine the high-frequency directional features, on the one hand, a high-frequency direction injection (HFDI) module without trainable parameters is constructed to inject the high-frequency directional information of the original image into the network. On the other hand, a multi-scale direction-aware (MSDA) module is constructed, which promotes the full extraction of local relations at different scales and the full perception of key features in different directions. In addition, considering the characteristics of infrared small targets, we construct a feature aggregation (FA) structure to address target disappearance in high-level feature maps, and a feature calibration fusion (FCF) module to alleviate feature bias during cross-layer feature fusion. Extensive experimental results show that our MSDA-Net achieves state-of-the-art (SOTA) results on multiple public datasets. The code can be available at https://github.com/YuChuang1205/MSDA-Net
>
---
#### [replaced 007] Foveation Improves Payload Capacity in Steganography
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2510.13151v2](https://arxiv.org/pdf/2510.13151v2)**

> **作者:** Lifeng Qiu Lin; Henry Kam; Qi Sun; Kaan Akşit
>
> **备注:** SIGGRAPH Asia 2025 Posters Proceedings
>
> **摘要:** Steganography finds its use in visual medium such as providing metadata and watermarking. With support of efficient latent representations and foveated rendering, we trained models that improve existing capacity limits from 100 to 500 bits, while achieving better accuracy of up to 1 failure bit out of 2000, at 200K test bits. Finally, we achieve a comparable visual quality of 31.47 dB PSNR and 0.13 LPIPS, showing the effectiveness of novel perceptual design in creating multi-modal latent representations in steganography.
>
---
#### [replaced 008] Aligning Text to Image in Diffusion Models is Easier Than You Think
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.08250v5](https://arxiv.org/pdf/2503.08250v5)**

> **作者:** Jaa-Yeon Lee; Byunghee Cha; Jeongsol Kim; Jong Chul Ye
>
> **备注:** NeurIPS 2025
>
> **摘要:** While recent advancements in generative modeling have significantly improved text-image alignment, some residual misalignment between text and image representations still remains. Some approaches address this issue by fine-tuning models in terms of preference optimization, etc., which require tailored datasets. Orthogonal to these methods, we revisit the challenge from the perspective of representation alignment-an approach that has gained popularity with the success of REPresentation Alignment (REPA). We first argue that conventional text-to-image (T2I) diffusion models, typically trained on paired image and text data (i.e., positive pairs) by minimizing score matching or flow matching losses, is suboptimal from the standpoint of representation alignment. Instead, a better alignment can be achieved through contrastive learning that leverages existing dataset as both positive and negative pairs. To enable efficient alignment with pretrained models, we propose SoftREPA- a lightweight contrastive fine-tuning strategy that leverages soft text tokens for representation alignment. This approach improves alignment with minimal computational overhead by adding fewer than 1M trainable parameters to the pretrained model. Our theoretical analysis demonstrates that our method explicitly increases the mutual information between text and image representations, leading to enhanced semantic consistency. Experimental results across text-to-image generation and text-guided image editing tasks validate the effectiveness of our approach in improving the semantic consistency of T2I generative models.
>
---
#### [replaced 009] BridgeDrive: Diffusion Bridge Policy for Closed-Loop Trajectory Planning in Autonomous Driving
- **分类: cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.23589v2](https://arxiv.org/pdf/2509.23589v2)**

> **作者:** Shu Liu; Wenlin Chen; Weihao Li; Zheng Wang; Lijin Yang; Jianing Huang; Yipin Zhang; Zhongzhan Huang; Ze Cheng; Hao Yang
>
> **备注:** 19 pages, 7 figures, 9 tables
>
> **摘要:** Diffusion-based planners have shown great promise for autonomous driving due to their ability to capture multi-modal driving behaviors. However, guiding these models effectively in reactive, closed-loop environments remains a significant challenge. Simple conditioning often fails to provide sufficient guidance in complex and dynamic driving scenarios. Recent work attempts to use typical expert driving behaviors (i.e., anchors) to guide diffusion models but relies on a truncated schedule, which introduces theoretical inconsistencies and can compromise performance. To address this, we introduce BridgeDrive, a novel anchor-guided diffusion bridge policy for closed-loop trajectory planning. Our approach provides a principled diffusion framework that effectively translates anchors into fine-grained trajectory plans, appropriately responding to varying traffic conditions. Our planner is compatible with efficient ODE solvers, a critical factor for real-time autonomous driving deployment. We achieve state-of-the-art performance on the Bench2Drive benchmark, improving the success rate by 7.72% over prior arts.
>
---
#### [replaced 010] Spatial Polarization Multiplexing: Single-Shot Invisible Shape and Reflectance Recovery
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.13177v2](https://arxiv.org/pdf/2504.13177v2)**

> **作者:** Tomoki Ichikawa; Ryo Kawahara; Ko Nishino
>
> **摘要:** We propose spatial polarization multiplexing (SPM) for joint sensing of shape and reflectance of a static or dynamic deformable object, which is also invisible to the naked eye. Past structured-light methods are limited to shape acquisition and cannot recover reflectance as they alter scene appearance. Our key idea is to spatially multiplex a polarization pattern to encode the incident ray and also densely sample the reflected light. We derive a quantized polarized light pattern that can be robustly and uniquely decoded from the reflected Angle of Linear Polarization (AoLP) values. It also enables single-shot disentanglement of polarimetric diffuse and specular reflections for accurate BRDF estimation. We achieve this spatial polarization multiplexing (SPM) with a constrained de Bruijn sequence. We validate this novel invisible single-shot shape and reflectance method with real static and dynamic objects. The results demonstrate the effectiveness of SPM for accurate shape and BRDF measurement which opens new avenues of application for 3D sensing thanks to its invisibility and ability to jointly recover the radiometric properties.
>
---
#### [replaced 011] Beyond the Failures: Rethinking Foundation Models in Pathology
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.23807v4](https://arxiv.org/pdf/2510.23807v4)**

> **作者:** Hamid R. Tizhoosh
>
> **摘要:** Despite their successes in vision and language, foundation models have stumbled in pathology, revealing low accuracy, instability, and heavy computational demands. These shortcomings stem not from tuning problems but from deeper conceptual mismatches: dense embeddings cannot represent the combinatorial richness of tissue, and current architectures inherit flaws in self-supervision, patch design, and noise-fragile pretraining. Biological complexity and limited domain innovation further widen the gap. The evidence is clear-pathology requires models explicitly designed for biological images rather than adaptations of large-scale natural-image methods whose assumptions do not hold for tissue.
>
---
#### [replaced 012] GloTok: Global Perspective Tokenizer for Image Reconstruction and Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14184v3](https://arxiv.org/pdf/2511.14184v3)**

> **作者:** Xuan Zhao; Zhongyu Zhang; Yuge Huang; Yuxi Mi; Guodong Mu; Shouhong Ding; Jun Wang; Rizen Guo; Shuigeng Zhou
>
> **备注:** Accepted at AAAI'26
>
> **摘要:** Existing state-of-the-art image tokenization methods leverage diverse semantic features from pre-trained vision models for additional supervision, to expand the distribution of latent representations and thereby improve the quality of image reconstruction and generation. These methods employ a locally supervised approach for semantic supervision, which limits the uniformity of semantic distribution. However, VA-VAE proves that a more uniform feature distribution yields better generation performance. In this work, we introduce a Global Perspective Tokenizer (GloTok), which utilizes global relational information to model a more uniform semantic distribution of tokenized features. Specifically, a codebook-wise histogram relation learning method is proposed to transfer the semantics, which are modeled by pre-trained models on the entire dataset, to the semantic codebook. Then, we design a residual learning module that recovers the fine-grained details to minimize the reconstruction error caused by quantization. Through the above design, GloTok delivers more uniformly distributed semantic latent representations, which facilitates the training of autoregressive (AR) models for generating high-quality images without requiring direct access to pre-trained models during the training process. Experiments on the standard ImageNet-1k benchmark clearly show that our proposed method achieves state-of-the-art reconstruction performance and generation quality.
>
---
#### [replaced 013] Exploring possible vector systems for faster training of neural networks with preconfigured latent spaces
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.07509v2](https://arxiv.org/pdf/2512.07509v2)**

> **作者:** Nikita Gabdullin
>
> **备注:** 9 pages, 5 figures, 1 table, 4 equations
>
> **摘要:** The overall neural network (NN) performance is closely related to the properties of its embedding distribution in latent space (LS). It has recently been shown that predefined vector systems, specifically An root system vectors, can be used as targets for latent space configurations (LSC) to ensure the desired LS structure. One of the main LSC advantage is the possibility of training classifier NNs without classification layers, which facilitates training NNs on datasets with extremely large numbers of classes. This paper provides a more general overview of possible vector systems for NN training along with their properties and methods for vector system construction. These systems are used to configure LS of encoders and visual transformers to significantly speed up ImageNet-1K and 50k-600k classes LSC training. It is also shown that using the minimum number of LS dimensions for a specific number of classes results in faster convergence. The latter has potential advantages for reducing the size of vector databases used to store NN embeddings.
>
---
#### [replaced 014] MACS: Multi-source Audio-to-image Generation with Contextual Significance and Semantic Alignment
- **分类: cs.SD; cs.CV; cs.GR; eess.AS**

- **简介: 该论文研究多声源音频到图像生成任务，旨在解决现有方法忽略自然场景中多声源特性的问题。作者提出MACS方法，首次显式分离多声源音频，并通过语义对齐与上下文显著性优化生成图像，构建了首个多声源音频-图像生成基准。**

- **链接: [https://arxiv.org/pdf/2503.10287v3](https://arxiv.org/pdf/2503.10287v3)**

> **作者:** Hao Zhou; Xiaobao Guo; Yuzhe Zhu; Adams Wai-Kin Kong
>
> **备注:** Accepted at AAAI 2026. Code available at https://github.com/alxzzhou/MACS
>
> **摘要:** Propelled by the breakthrough in deep generative models, audio-to-image generation has emerged as a pivotal cross-modal task that converts complex auditory signals into rich visual representations. However, previous works only focus on single-source audio inputs for image generation, ignoring the multi-source characteristic in natural auditory scenes, thus limiting the performance in generating comprehensive visual content. To bridge this gap, we propose a method called MACS to conduct multi-source audio-to-image generation. To our best knowledge, this is the first work that explicitly separates multi-source audio to capture the rich audio components before image generation. MACS is a two-stage method. In the first stage, multi-source audio inputs are separated by a weakly supervised method, where the audio and text labels are semantically aligned by casting into a common space using the large pre-trained CLAP model. We introduce a ranking loss to consider the contextual significance of the separated audio signals. In the second stage, effective image generation is achieved by mapping the separated audio signals to the generation condition using only a trainable adapter and a MLP layer. We preprocess the LLP dataset as the first full multi-source audio-to-image generation benchmark. The experiments are conducted on multi-source, mixed-source, and single-source audio-to-image generation tasks. The proposed MACS outperforms the current state-of-the-art methods in 17 out of the 21 evaluation indexes on all tasks and delivers superior visual quality.
>
---
#### [replaced 015] Adversarially Pretrained Transformers May Be Universally Robust In-Context Learners
- **分类: cs.LG; cs.CV; stat.ML**

- **链接: [https://arxiv.org/pdf/2505.14042v2](https://arxiv.org/pdf/2505.14042v2)**

> **作者:** Soichiro Kumano; Hiroshi Kera; Toshihiko Yamasaki
>
> **摘要:** Adversarial training is one of the most effective adversarial defenses, but it incurs a high computational cost. In this study, we present the first theoretical analysis suggesting that adversarially pretrained transformers can serve as universally robust foundation models -- models that can robustly adapt to diverse downstream tasks with only lightweight tuning. Specifically, we demonstrate that single-layer linear transformers, after adversarial pretraining across a variety of classification tasks, can robustly generalize to unseen classification tasks through in-context learning from clean demonstrations (i.e., without requiring additional adversarial training or examples). This universal robustness stems from the model's ability to adaptively focus on robust features within given tasks. We also show the two open challenges for attaining robustness: accuracy--robustness trade-off and sample-hungry training. This study initiates the discussion on the utility of universally robust foundation models. While their training is expensive, the investment would prove worthwhile as downstream tasks can enjoy free adversarial robustness. The code is available at https://github.com/s-kumano/universally-robust-in-context-learner.
>
---
#### [replaced 016] Matrix-game 2.0: An open-source real-time and streaming interactive world model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.13009v3](https://arxiv.org/pdf/2508.13009v3)**

> **作者:** Xianglong He; Chunli Peng; Zexiang Liu; Boyang Wang; Yifan Zhang; Qi Cui; Fei Kang; Biao Jiang; Mengyin An; Yangyang Ren; Baixin Xu; Hao-Xiang Guo; Kaixiong Gong; Size Wu; Wei Li; Xuchen Song; Yang Liu; Yangguang Li; Yahui Zhou
>
> **备注:** Project Page: https://matrix-game-v2.github.io
>
> **摘要:** Recent advances in interactive video generations have demonstrated diffusion model's potential as world models by capturing complex physical dynamics and interactive behaviors. However, existing interactive world models depend on bidirectional attention and lengthy inference steps, severely limiting real-time performance. Consequently, they are hard to simulate real-world dynamics, where outcomes must update instantaneously based on historical context and current actions. To address this, we present Matrix-Game 2.0, an interactive world model generates long videos on-the-fly via few-step auto-regressive diffusion. Our framework consists of three key components: (1) A scalable data production pipeline for Unreal Engine and GTA5 environments to effectively produce massive amounts (about 1200 hours) of video data with diverse interaction annotations; (2) An action injection module that enables frame-level mouse and keyboard inputs as interactive conditions; (3) A few-step distillation based on the casual architecture for real-time and streaming video generation. Matrix Game 2.0 can generate high-quality minute-level videos across diverse scenes at an ultra-fast speed of 25 FPS. We open-source our model weights and codebase to advance research in interactive world modeling.
>
---
#### [replaced 017] Stronger is not better: Better Augmentations in Contrastive Learning for Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.05992v2](https://arxiv.org/pdf/2512.05992v2)**

> **作者:** Azeez Idris; Abdurahman Ali Mohammed; Samuel Fanijo
>
> **备注:** NeurIPS Black in AI workshop - 2022
>
> **摘要:** Self-supervised contrastive learning is among the recent representation learning methods that have shown performance gains in several downstream tasks including semantic segmentation. This paper evaluates strong data augmentation, one of the most important components for self-supervised contrastive learning's improved performance. Strong data augmentation involves applying the composition of multiple augmentation techniques on images. Surprisingly, we find that the existing data augmentations do not always improve performance for semantic segmentation for medical images. We experiment with other augmentations that provide improved performance.
>
---
#### [replaced 018] Adversarial-Robustness-Guided Graph Pruning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.12331v2](https://arxiv.org/pdf/2411.12331v2)**

> **作者:** Yongyu Wang
>
> **摘要:** Graph learning plays a central role in many data mining and machine learning tasks, such as manifold learning, data representation and analysis, dimensionality reduction, clustering, and visualization. In this work, we propose a highly scalable, adversarial-robustness-guided graph pruning framework for learning graph topologies from data. By performing a spectral adversarial robustness evaluation, our method aims to learn sparse, undirected graphs that help the underlying algorithms resist noise and adversarial perturbations. In particular, we explicitly identify and prune edges that are most vulnerable to adversarial attacks. We use spectral clustering, one of the most representative graph-based machine learning algorithms, to evaluate the proposed framework. Compared with prior state-of-the-art graph learning approaches, the proposed method is more scalable and significantly improves both the computational efficiency and the solution quality of spectral clustering.
>
---
#### [replaced 019] Entropy-Informed Weighting Channel Normalizing Flow for Deep Generative Models
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2407.04958v2](https://arxiv.org/pdf/2407.04958v2)**

> **作者:** Wei Chen; Shian Du; Shigui Li; Delu Zeng; John Paisley
>
> **摘要:** Normalizing Flows (NFs) are widely used in deep generative models for their exact likelihood estimation and efficient sampling. However, they require substantial memory since the latent space matches the input dimension. Multi-scale architectures address this by progressively reducing latent dimensions while preserving reversibility. Existing multi-scale architectures use simple, static channel-wise splitting, limiting expressiveness. To improve this, we introduce a regularized, feature-dependent $\mathtt{Shuffle}$ operation and integrate it into vanilla multi-scale architecture. This operation adaptively generates channel-wise weights and shuffles latent variables before splitting them. We observe that such operation guides the variables to evolve in the direction of entropy increase, hence we refer to NFs with the $\mathtt{Shuffle}$ operation as \emph{Entropy-Informed Weighting Channel Normalizing Flow} (EIW-Flow). Extensive experiments on CIFAR-10, CelebA, ImageNet, and LSUN demonstrate that EIW-Flow achieves state-of-the-art density estimation and competitive sample quality for deep generative modeling, with minimal computational overhead.
>
---
#### [replaced 020] RELOCATE: A Simple Training-Free Baseline for Visual Query Localization Using Region-Based Representations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.01826v2](https://arxiv.org/pdf/2412.01826v2)**

> **作者:** Savya Khosla; Sethuraman T; Alexander Schwing; Derek Hoiem
>
> **摘要:** We present RELOCATE, a simple training-free baseline designed to perform the challenging task of visual query localization in long videos. To eliminate the need for task-specific training and efficiently handle long videos, RELOCATE leverages a region-based representation derived from pretrained vision models. At a high level, it follows the classic object localization approach: (1) identify all objects in each video frame, (2) compare the objects with the given query and select the most similar ones, and (3) perform bidirectional tracking to get a spatio-temporal response. However, we propose some key enhancements to handle small objects, cluttered scenes, partial visibility, and varying appearances. Notably, we refine the selected objects for accurate localization and generate additional visual queries to capture visual variations. We evaluate RELOCATE on the challenging Ego4D Visual Query 2D Localization dataset, establishing a new baseline that outperforms prior task-specific methods by 49% (relative improvement) in spatio-temporal average precision.
>
---
#### [replaced 021] Semantic Data Augmentation Enhanced Invariant Risk Minimization for Medical Image Domain Generalization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.05593v2](https://arxiv.org/pdf/2502.05593v2)**

> **作者:** Yaoyao Zhu; Xiuding Cai; Yingkai Wang; Yu Yao; Xu Luo; Zhongliang Fu
>
> **摘要:** Deep learning has achieved remarkable success in medical image classification. However, its clinical application is often hindered by data heterogeneity caused by variations in scanner vendors, imaging protocols, and operators. Approaches such as invariant risk minimization (IRM) aim to address this challenge of out-of-distribution generalization. For instance, VIRM improves upon IRM by tackling the issue of insufficient feature support overlap, demonstrating promising potential. Nonetheless, these methods face limitations in medical imaging due to the scarcity of annotated data and the inefficiency of augmentation strategies. To address these issues, we propose a novel domain-oriented direction selector to replace the random augmentation strategy used in VIRM. Our method leverages inter-domain covariance as a guider for augmentation direction, guiding data augmentation towards the target domain. This approach effectively reduces domain discrepancies and enhances generalization performance. Experiments on a multi-center diabetic retinopathy dataset demonstrate that our method outperforms state-of-the-art approaches, particularly under limited data conditions and significant domain heterogeneity.
>
---
#### [replaced 022] C-DIRA: Computationally Efficient Dynamic ROI Routing and Domain-Invariant Adversarial Learning for Lightweight Driver Behavior Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.08647v2](https://arxiv.org/pdf/2512.08647v2)**

> **作者:** Keito Inoshita
>
> **摘要:** Driver distraction behavior recognition using in-vehicle cameras demands real-time inference on edge devices. However, lightweight models often fail to capture fine-grained behavioral cues, resulting in reduced performance on unseen drivers or under varying conditions. ROI-based methods also increase computational cost, making it difficult to balance efficiency and accuracy. This work addresses the need for a lightweight architecture that overcomes these constraints. We propose Computationally efficient Dynamic region of Interest Routing and domain-invariant Adversarial learning for lightweight driver behavior recognition (C-DIRA). The framework combines saliency-driven Top-K ROI pooling and fused classification for local feature extraction and integration. Dynamic ROI routing enables selective computation by applying ROI inference only to high difficulty data samples. Moreover, pseudo-domain labeling and adversarial learning are used to learn domain-invariant features robust to driver and background variation. Experiments on the State Farm Distracted Driver Detection Dataset show that C-DIRA maintains high accuracy with significantly fewer FLOPs and lower latency than prior lightweight models. It also demonstrates robustness under visual degradation such as blur and low-light, and stable performance across unseen domains. These results confirm C-DIRA's effectiveness in achieving compactness, efficiency, and generalization.
>
---
#### [replaced 023] Efficiently Reconstructing Dynamic Scenes One D4RT at a Time
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.08924v2](https://arxiv.org/pdf/2512.08924v2)**

> **作者:** Chuhan Zhang; Guillaume Le Moing; Skanda Koppula; Ignacio Rocco; Liliane Momeni; Junyu Xie; Shuyang Sun; Rahul Sukthankar; Joëlle K. Barral; Raia Hadsell; Zoubin Ghahramani; Andrew Zisserman; Junlin Zhang; Mehdi S. M. Sajjadi
>
> **备注:** Project Page: https://d4rt-paper.github.io/
>
> **摘要:** Understanding and reconstructing the complex geometry and motion of dynamic scenes from video remains a formidable challenge in computer vision. This paper introduces D4RT, a simple yet powerful feedforward model designed to efficiently solve this task. D4RT utilizes a unified transformer architecture to jointly infer depth, spatio-temporal correspondence, and full camera parameters from a single video. Its core innovation is a novel querying mechanism that sidesteps the heavy computation of dense, per-frame decoding and the complexity of managing multiple, task-specific decoders. Our decoding interface allows the model to independently and flexibly probe the 3D position of any point in space and time. The result is a lightweight and highly scalable method that enables remarkably efficient training and inference. We demonstrate that our approach sets a new state of the art, outperforming previous methods across a wide spectrum of 4D reconstruction tasks. We refer to the project webpage for animated results: https://d4rt-paper.github.io/.
>
---
#### [replaced 024] INRetouch: Context Aware Implicit Neural Representation for Photography Retouching
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2412.03848v4](https://arxiv.org/pdf/2412.03848v4)**

> **作者:** Omar Elezabi; Marcos V. Conde; Zongwei Wu; Radu Timofte
>
> **备注:** Accepted by WACV 2026
>
> **摘要:** Professional photo editing remains challenging, requiring extensive knowledge of imaging pipelines and significant expertise. While recent deep learning approaches, particularly style transfer methods, have attempted to automate this process, they often struggle with output fidelity, editing control, and complex retouching capabilities. We propose a novel retouch transfer approach that learns from professional edits through before-after image pairs, enabling precise replication of complex editing operations. We develop a context-aware Implicit Neural Representation that learns to apply edits adaptively based on image content and context, and is capable of learning from a single example. Our method extracts implicit transformations from reference edits and adaptively applies them to new images. To facilitate this research direction, we introduce a comprehensive Photo Retouching Dataset comprising 100,000 high-quality images edited using over 170 professional Adobe Lightroom presets. Through extensive evaluation, we demonstrate that our approach not only surpasses existing methods in photo retouching but also enhances performance in related image reconstruction tasks like Gamut Mapping and Raw Reconstruction. By bridging the gap between professional editing capabilities and automated solutions, our work presents a significant step toward making sophisticated photo editing more accessible while maintaining high-fidelity results. The source code and the dataset are publicly available at https://omaralezaby.github.io/inretouch .
>
---
#### [replaced 025] Generalised Medical Phrase Grounding
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出通用医学短语定位（GMPG）任务，解决传统方法无法处理多区域、非可定位短语的问题。作者构建MedGrounder模型，支持零个、一个或多个区域定位，通过两阶段训练，在少标注下实现优越性能，并可与现有报告生成模型结合生成定位报告。**

- **链接: [https://arxiv.org/pdf/2512.01085v2](https://arxiv.org/pdf/2512.01085v2)**

> **作者:** Wenjun Zhang; Shekhar S. Chandra; Aaron Nicolson
>
> **备注:** 10 pages
>
> **摘要:** Medical phrase grounding (MPG) maps textual descriptions of radiological findings to corresponding image regions. These grounded reports are easier to interpret, especially for non-experts. Existing MPG systems mostly follow the referring expression comprehension (REC) paradigm and return exactly one bounding box per phrase. Real reports often violate this assumption. They contain multi-region findings, non-diagnostic text, and non-groundable phrases, such as negations or descriptions of normal anatomy. Motivated by this, we reformulate the task as generalised medical phrase grounding (GMPG), where each sentence is mapped to zero, one, or multiple scored regions. To realise this formulation, we introduce the first GMPG model: MedGrounder. We adopted a two-stage training regime: pre-training on report sentence--anatomy box alignment datasets and fine-tuning on report sentence--human annotated box datasets. Experiments on PadChest-GR and MS-CXR show that MedGrounder achieves strong zero-shot transfer and outperforms REC-style and grounded report generation baselines on multi-region and non-groundable phrases, while using far fewer human box annotations. Finally, we show that MedGrounder can be composed with existing report generators to produce grounded reports without retraining the generator.
>
---
#### [replaced 026] Two Causal Principles for Improving Visual Dialog
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对视觉对话（VisDial）任务，指出先前模型忽略的两个因果问题：对话历史引入的偏差和未观测混淆因子导致的虚假相关。作者提出两条因果原则及干预算法，可普遍提升现有模型性能，且与模型无关。**

- **链接: [https://arxiv.org/pdf/1911.10496v3](https://arxiv.org/pdf/1911.10496v3)**

> **作者:** Jiaxin Qi; Yulei Niu; Jianqiang Huang; Hanwang Zhang
>
> **备注:** Accepted by CVPR 2020
>
> **摘要:** This paper unravels the design tricks adopted by us, the champion team MReaL-BDAI, for Visual Dialog Challenge 2019: two causal principles for improving Visual Dialog (VisDial). By "improving", we mean that they can promote almost every existing VisDial model to the state-of-the-art performance on the leader-board. Such a major improvement is only due to our careful inspection on the causality behind the model and data, finding that the community has overlooked two causalities in VisDial. Intuitively, Principle 1 suggests: we should remove the direct input of the dialog history to the answer model, otherwise a harmful shortcut bias will be introduced; Principle 2 says: there is an unobserved confounder for history, question, and answer, leading to spurious correlations from training data. In particular, to remove the confounder suggested in Principle 2, we propose several causal intervention algorithms, which make the training fundamentally different from the traditional likelihood estimation. Note that the two principles are model-agnostic, so they are applicable in any VisDial model. The code is available at https://github.com/simpleshinobu/visdial-principles.
>
---
#### [replaced 027] World in a Frame: Understanding Culture Mixing as a New Challenge for Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22787v2](https://arxiv.org/pdf/2511.22787v2)**

> **作者:** Eunsu Kim; Junyeong Park; Na Min An; Junseong Kim; Hitesh Laxmichand Patel; Jiho Jin; Julia Kruk; Amit Agarwal; Srikant Panda; Fenal Ashokbhai Ilasariya; Hyunjung Shim; Alice Oh
>
> **摘要:** In a globalized world, cultural elements from diverse origins frequently appear together within a single visual scene. We refer to these as culture mixing scenarios, yet how Large Vision-Language Models (LVLMs) perceive them remains underexplored. We investigate culture mixing as a critical challenge for LVLMs and examine how current models behave when cultural items from multiple regions appear together. To systematically analyze these behaviors, we construct CultureMix, a food Visual Question Answering (VQA) benchmark with 23k diffusion-generated, human-verified culture mixing images across four subtasks: (1) food-only, (2) food+food, (3) food+background, and (4) food+food+background. Evaluating 10 LVLMs, we find consistent failures to preserve individual cultural identities in mixed settings. Models show strong background reliance, with accuracy dropping 14% when cultural backgrounds are added to food-only baselines, and they produce inconsistent predictions for identical foods across different contexts. To address these limitations, we explore three robustness strategies. We find supervised fine-tuning using a diverse culture mixing dataset substantially improve model consistency and reduce background sensitivity. We call for increased attention to culture mixing scenarios as a critical step toward developing LVLMs capable of operating reliably in culturally diverse real-world environments.
>
---
#### [replaced 028] Financial Fraud Identification and Interpretability Study for Listed Companies Based on Convolutional Neural Network
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.06648v2](https://arxiv.org/pdf/2512.06648v2)**

> **作者:** Xiao Li
>
> **备注:** in Chinese language
>
> **摘要:** Since the emergence of joint-stock companies, financial fraud by listed firms has repeatedly undermined capital markets. Fraud is difficult to detect because of covert tactics and the high labor and time costs of audits. Traditional statistical models are interpretable but struggle with nonlinear feature interactions, while machine learning models are powerful but often opaque. In addition, most existing methods judge fraud only for the current year based on current year data, limiting timeliness. This paper proposes a financial fraud detection framework for Chinese A-share listed companies based on convolutional neural networks (CNNs). We design a feature engineering scheme that transforms firm-year panel data into image like representations, enabling the CNN to capture cross-sectional and temporal patterns and to predict fraud in advance. Experiments show that the CNN outperforms logistic regression and LightGBM in accuracy, robustness, and early-warning performance, and that proper tuning of the classification threshold is crucial in high-risk settings. To address interpretability, we analyze the model along the dimensions of entity, feature, and time using local explanation techniques. We find that solvency, ratio structure, governance structure, and internal control are general predictors of fraud, while environmental indicators matter mainly in high-pollution industries. Non-fraud firms share stable feature patterns, whereas fraud firms exhibit heterogeneous patterns concentrated in short time windows. A case study of Guanong Shares in 2022 shows that cash flow analysis, social responsibility, governance structure, and per-share indicators are the main drivers of the model's fraud prediction, consistent with the company's documented misconduct.
>
---
#### [replaced 029] Classifying Phonotrauma Severity from Vocal Fold Images with Soft Ordinal Regression
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.09702v2](https://arxiv.org/pdf/2511.09702v2)**

> **作者:** Katie Matton; Purvaja Balaji; Hamzeh Ghasemzadeh; Jameson C. Cooper; Daryush D. Mehta; Jarrad H. Van Stan; Robert E. Hillman; Rosalind Picard; John Guttag; S. Mazdak Abulnaga
>
> **备注:** 16 pages, 9 figures, 5 tables; ML4H 2025; Proceedings of Machine Learning Research 297, 2025
>
> **摘要:** Phonotrauma refers to vocal fold tissue damage resulting from exposure to forces during voicing. It occurs on a continuum from mild to severe, and treatment options can vary based on severity. Assessment of severity involves a clinician's expert judgment, which is costly and can vary widely in reliability. In this work, we present the first method for automatically classifying phonotrauma severity from vocal fold images. To account for the ordinal nature of the labels, we adopt a widely used ordinal regression framework. To account for label uncertainty, we propose a novel modification to ordinal regression loss functions that enables them to operate on soft labels reflecting annotator rating distributions. Our proposed soft ordinal regression method achieves predictive performance approaching that of clinical experts, while producing well-calibrated uncertainty estimates. By providing an automated tool for phonotrauma severity assessment, our work can enable large-scale studies of phonotrauma, ultimately leading to improved clinical understanding and patient care.
>
---
#### [replaced 030] Unsupervised Structural Scene Decomposition via Foreground-Aware Slot Attention with Pseudo-Mask Guidance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.02685v2](https://arxiv.org/pdf/2512.02685v2)**

> **作者:** Huankun Sheng; Ming Li; Yixiang Wei; Yeying Fan; Yu-Hui Wen; Tieliang Gong; Yong-Jin Liu
>
> **摘要:** Recent advances in object-centric representation learning have shown that slot attention-based methods can effectively decompose visual scenes into object slot representations without supervision. However, existing approaches typically process foreground and background regions indiscriminately, often resulting in background interference and suboptimal instance discovery performance on real-world data. To address this limitation, we propose Foreground-Aware Slot Attention (FASA), a two-stage framework that explicitly separates foreground from background to enable precise object discovery. In the first stage, FASA performs a coarse scene decomposition to distinguish foreground from background regions through a dual-slot competition mechanism. These slots are initialized via a clustering-based strategy, yielding well-structured representations of salient regions. In the second stage, we introduce a masked slot attention mechanism where the first slot captures the background while the remaining slots compete to represent individual foreground objects. To further address over-segmentation of foreground objects, we incorporate pseudo-mask guidance derived from a patch affinity graph constructed with self-supervised image features to guide the learning of foreground slots. Extensive experiments on both synthetic and real-world datasets demonstrate that FASA consistently outperforms state-of-the-art methods, validating the effectiveness of explicit foreground modeling and pseudo-mask guidance for robust scene decomposition and object-coherent representation. Code will be made publicly available.
>
---
#### [replaced 031] RingMoE: Mixture-of-Modality-Experts Multi-Modal Foundation Models for Universal Remote Sensing Image Interpretation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.03166v2](https://arxiv.org/pdf/2504.03166v2)**

> **作者:** Hanbo Bi; Yingchao Feng; Boyuan Tong; Mengyu Wang; Haichen Yu; Yongqiang Mao; Hao Chang; Wenhui Diao; Peijin Wang; Yue Yu; Hanyang Peng; Yehong Zhang; Kun Fu; Xian Sun
>
> **摘要:** The rapid advancement of foundation models has revolutionized visual representation learning in a self-supervised manner. However, their application in remote sensing (RS) remains constrained by a fundamental gap: existing models predominantly handle single or limited modalities, overlooking the inherently multi-modal nature of RS observations. Optical, synthetic aperture radar (SAR), and multi-spectral data offer complementary insights that significantly reduce the inherent ambiguity and uncertainty in single-source analysis. To bridge this gap, we introduce RingMoE, a unified multi-modal RS foundation model with 14.7 billion parameters, pre-trained on 400 million multi-modal RS images from nine satellites. RingMoE incorporates three key innovations: (1) A hierarchical Mixture-of-Experts (MoE) architecture comprising modal-specialized, collaborative, and shared experts, effectively modeling intra-modal knowledge while capturing cross-modal dependencies to mitigate conflicts between modal representations; (2) Physics-informed self-supervised learning, explicitly embedding sensor-specific radiometric characteristics into the pre-training objectives; (3) Dynamic expert pruning, enabling adaptive model compression from 14.7B to 1B parameters while maintaining performance, facilitating efficient deployment in Earth observation applications. Evaluated across 23 benchmarks spanning six key RS tasks (i.e., classification, detection, segmentation, tracking, change detection, and depth estimation), RingMoE outperforms existing foundation models and sets new SOTAs, demonstrating remarkable adaptability from single-modal to multi-modal scenarios. Beyond theoretical progress, it has been deployed and trialed in multiple sectors, including emergency response, land management, marine sciences, and urban planning.
>
---
#### [replaced 032] THCRL: Trusted Hierarchical Contrastive Representation Learning for Multi-View Clustering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.00368v2](https://arxiv.org/pdf/2512.00368v2)**

> **作者:** Jian Zhu
>
> **摘要:** Multi-View Clustering (MVC) has garnered increasing attention in recent years. It is capable of partitioning data samples into distinct groups by learning a consensus representation. However, a significant challenge remains: the problem of untrustworthy fusion. This problem primarily arises from two key factors: 1) Existing methods often ignore the presence of inherent noise within individual views; 2) In traditional MVC methods using Contrastive Learning (CL), similarity computations typically rely on different views of the same instance, while neglecting the structural information from nearest neighbors within the same cluster. Consequently, this leads to the wrong direction for multi-view fusion. To address this problem, we present a novel Trusted Hierarchical Contrastive Representation Learning (THCRL). It consists of two key modules. Specifically, we propose the Deep Symmetry Hierarchical Fusion (DSHF) module, which leverages the UNet architecture integrated with multiple denoising mechanisms to achieve trustworthy fusion of multi-view data. Furthermore, we present the Average K-Nearest Neighbors Contrastive Learning (AKCL) module to align the fused representation with the view-specific representation. Unlike conventional strategies, AKCL enhances representation similarity among samples belonging to the same cluster, rather than merely focusing on the same sample across views, thereby reinforcing the confidence of the fused representation. Extensive experiments demonstrate that THCRL achieves the state-of-the-art performance in deep MVC tasks.
>
---
#### [replaced 033] LENVIZ: A High-Resolution Low-Exposure Night Vision Benchmark Dataset
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2503.19804v2](https://arxiv.org/pdf/2503.19804v2)**

> **作者:** Manjushree Aithal; Rosaura G. VidalMata; Manikandtan Kartha; Gong Chen; Eashan Adhikarla; Lucas N. Kirsten; Zhicheng Fu; Nikhil A. Madhusudhana; Joe Nasti
>
> **摘要:** Low-light image enhancement is crucial for a myriad of applications, from night vision and surveillance, to autonomous driving. However, due to the inherent limitations that come in hand with capturing images in low-illumination environments, the task of enhancing such scenes still presents a formidable challenge. To advance research in this field, we introduce our Low Exposure Night Vision (LENVIZ) Dataset, a comprehensive multi-exposure benchmark dataset for low-light image enhancement comprising of over 230K frames showcasing 24K real-world indoor and outdoor, with-and without human, scenes. Captured using 3 different camera sensors, LENVIZ offers a wide range of lighting conditions, noise levels, and scene complexities, making it the largest publicly available up-to 4K resolution benchmark in the field. LENVIZ includes high quality human-generated ground truth, for which each multi-exposure low-light scene has been meticulously curated and edited by expert photographers to ensure optimal image quality. Furthermore, we also conduct a comprehensive analysis of current state-of-the-art low-light image enhancement techniques on our dataset and highlight potential areas of improvement.
>
---
#### [replaced 034] CoD: A Diffusion Foundation Model for Image Compression
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18706v2](https://arxiv.org/pdf/2511.18706v2)**

> **作者:** Zhaoyang Jia; Zihan Zheng; Naifu Xue; Jiahao Li; Bin Li; Zongyu Guo; Xiaoyi Zhang; Houqiang Li; Yan Lu
>
> **摘要:** Existing diffusion codecs typically build on text-to-image diffusion foundation models like Stable Diffusion. However, text conditioning is suboptimal from a compression perspective, hindering the potential of downstream diffusion codecs, particularly at ultra-low bitrates. To address it, we introduce \textbf{CoD}, the first \textbf{Co}mpression-oriented \textbf{D}iffusion foundation model, trained from scratch to enable end-to-end optimization of both compression and generation. CoD is not a fixed codec but a general foundation model designed for various diffusion-based codecs. It offers several advantages: \textbf{High compression efficiency}, replacing Stable Diffusion with CoD in downstream codecs like DiffC achieves SOTA results, especially at ultra-low bitrates (e.g., 0.0039 bpp); \textbf{Low-cost and reproducible training}, 300$\times$ faster training than Stable Diffusion ($\sim$ 20 vs. $\sim$ 6,250 A100 GPU days) on entirely open image-only datasets; \textbf{Providing new insights}, e.g., We find pixel-space diffusion can achieve VTM-level PSNR with high perceptual quality and can outperform GAN-based codecs using fewer parameters. We hope CoD lays the foundation for future diffusion codec research. Codes will be released.
>
---
#### [replaced 035] Adaptive Gradient Calibration for Single-Positive Multi-Label Learning in Remote Sensing Image Scene Classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.08269v2](https://arxiv.org/pdf/2510.08269v2)**

> **作者:** Chenying Liu; Gianmarco Perantoni; Lorenzo Bruzzone; Xiao Xiang Zhu
>
> **备注:** 14 pages, 7 figures; revised version
>
> **摘要:** Multi-label classification (MLC) offers a more comprehensive semantic understanding of Remote Sensing (RS) imagery compared to traditional single-label classification (SLC). However, obtaining complete annotations for MLC is particularly challenging due to the complexity and high cost of the labeling process. As a practical alternative, single-positive multi-label learning (SPML) has emerged, where each image is annotated with only one relevant label, and the model is expected to recover the full set of labels. While scalable, SPML introduces significant supervision ambiguity, demanding specialized solutions for model training. Although various SPML methods have been proposed in the computer vision domain, research in the RS context remains limited. To bridge this gap, we propose Adaptive Gradient Calibration (AdaGC), a novel and generalizable SPML framework tailored to RS imagery. AdaGC adopts a gradient calibration (GC) mechanism with a dual exponential moving average (EMA) module for robust pseudo-label generation. We introduce a theoretically grounded, training-dynamics-based indicator to adaptively trigger GC, which ensures GC's effectiveness by preventing it from being affected by model underfitting or overfitting to label noise. Extensive experiments on two benchmark RS datasets under two distinct label noise types demonstrate that AdaGC achieves state-of-the-art (SOTA) performance while maintaining strong robustness across diverse settings. The codes and data will be released at https://github.com/rslab-unitrento/AdaGC.
>
---
#### [replaced 036] WGAST: Weakly-Supervised Generative Network for Daily 10 m Land Surface Temperature Estimation via Spatio-Temporal Fusion
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.06485v2](https://arxiv.org/pdf/2508.06485v2)**

> **作者:** Sofiane Bouaziz; Adel Hafiane; Raphael Canals; Rachid Nedjai
>
> **摘要:** Urbanization, climate change, and agricultural stress are increasing the demand for precise and timely environmental monitoring. Land Surface Temperature (LST) is a key variable in this context and is retrieved from remote sensing satellites. However, these systems face a trade-off between spatial and temporal resolution. While spatio-temporal fusion methods offer promising solutions, few have addressed the estimation of daily LST at 10 m resolution. In this study, we present WGAST, a weakly-supervised generative network for daily 10 m LST estimation via spatio-temporal fusion of Terra MODIS, Landsat 8, and Sentinel-2. WGAST is the first end-to-end deep learning framework designed for this task. It adopts a conditional generative adversarial architecture, with a generator composed of four stages: feature extraction, fusion, LST reconstruction, and noise suppression. The first stage employs a set of encoders to extract multi-level latent representations from the inputs, which are then fused in the second stage using cosine similarity, normalization, and temporal attention mechanisms. The third stage decodes the fused features into high-resolution LST, followed by a Gaussian filter to suppress high-frequency noise. Training follows a weakly supervised strategy based on physical averaging principles and reinforced by a PatchGAN discriminator. Experiments demonstrate that WGAST outperforms existing methods in both quantitative and qualitative evaluations. Compared to the best-performing baseline, on average, WGAST reduces RMSE by 17.05% and improves SSIM by 4.22%. Furthermore, WGAST effectively captures fine-scale thermal patterns, as validated against near-surface air temperature measurements from 33 near-ground sensors. The code is available at https://github.com/Sofianebouaziz1/WGAST.git.
>
---
#### [replaced 037] MoReGen: Multi-Agent Motion-Reasoning Engine for Code-based Text-to-Video Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.04221v2](https://arxiv.org/pdf/2512.04221v2)**

> **作者:** Xiangyu Bai; He Liang; Bishoy Galoaa; Utsav Nandi; Shayda Moezzi; Yuhang He; Sarah Ostadabbas
>
> **摘要:** While text-to-video (T2V) generation has achieved remarkable progress in photorealism, generating intent-aligned videos that faithfully obey physics principles remains a core challenge. In this work, we systematically study Newtonian motion-controlled text-to-video generation and evaluation, emphasizing physical precision and motion coherence. We introduce MoReGen, a motion-aware, physics-grounded T2V framework that integrates multi-agent LLMs, physics simulators, and renderers to generate reproducible, physically accurate videos from text prompts in the code domain. To quantitatively assess physical validity, we propose object-trajectory correspondence as a direct evaluation metric and present MoReSet, a benchmark of 1,275 human-annotated videos spanning nine classes of Newtonian phenomena with scene descriptions, spatiotemporal relations, and ground-truth trajectories. Using MoReSet, we conduct experiments on existing T2V models, evaluating their physical validity through both our MoRe metrics and existing physics-based evaluators. Our results reveal that state-of-the-art models struggle to maintain physical validity, while MoReGen establishes a principled direction toward physically coherent video synthesis.
>
---
#### [replaced 038] Bring Your Dreams to Life: Continual Text-to-Video Customization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.05802v2](https://arxiv.org/pdf/2512.05802v2)**

> **作者:** Jiahua Dong; Xudong Wang; Wenqi Liang; Zongyan Han; Meng Cao; Duzhen Zhang; Hanbin Zhao; Zhi Han; Salman Khan; Fahad Shahbaz Khan
>
> **备注:** Accepted to AAAI2026
>
> **摘要:** Customized text-to-video generation (CTVG) has recently witnessed great progress in generating tailored videos from user-specific text. However, most CTVG methods assume that personalized concepts remain static and do not expand incrementally over time. Additionally, they struggle with forgetting and concept neglect when continuously learning new concepts, including subjects and motions. To resolve the above challenges, we develop a novel Continual Customized Video Diffusion (CCVD) model, which can continuously learn new concepts to generate videos across various text-to-video generation tasks by tackling forgetting and concept neglect. To address catastrophic forgetting, we introduce a concept-specific attribute retention module and a task-aware concept aggregation strategy. They can capture the unique characteristics and identities of old concepts during training, while combining all subject and motion adapters of old concepts based on their relevance during testing. Besides, to tackle concept neglect, we develop a controllable conditional synthesis to enhance regional features and align video contexts with user conditions, by incorporating layer-specific region attention-guided noise estimation. Extensive experimental comparisons demonstrate that our CCVD outperforms existing CTVG baselines on both the DreamVideo and Wan 2.1 backbones. The code is available at https://github.com/JiahuaDong/CCVD.
>
---
#### [replaced 039] OpenConstruction: A Systematic Synthesis of Open Visual Datasets for Data-Centric Artificial Intelligence in Construction Monitoring
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.11482v2](https://arxiv.org/pdf/2508.11482v2)**

> **作者:** Ruoxin Xiong; Yanyu Wang; Jiannan Cai; Kaijian Liu; Yuansheng Zhu; Pingbo Tang; Nora El-Gohary
>
> **摘要:** The construction industry increasingly relies on visual data to support Artificial Intelligence (AI) and Machine Learning (ML) applications for site monitoring. High-quality, domain-specific datasets, comprising images, videos, and point clouds, capture site geometry and spatiotemporal dynamics, including the location and interaction of objects, workers, and materials. However, despite growing interest in leveraging visual datasets, existing resources vary widely in sizes, data modalities, annotation quality, and representativeness of real-world construction conditions. A systematic review to categorize their data characteristics and application contexts is still lacking, limiting the community's ability to fully understand the dataset landscape, identify critical gaps, and guide future directions toward more effective, reliable, and scalable AI applications in construction. To address this gap, this study conducts an extensive search of academic databases and open-data platforms, yielding 51 publicly available visual datasets that span the 2005-2024 period. These datasets are categorized using a structured data schema covering (i) data fundamentals (e.g., size and license), (ii) data modalities (e.g., RGB and point cloud), (iii) annotation frameworks (e.g., bounding boxes), and (iv) downstream application domains (e.g., progress tracking). This study synthesizes these findings into an open-source catalog, OpenConstruction, supporting data-driven method development. Furthermore, the study discusses several critical limitations in the existing construction dataset landscape and presents a roadmap for future data infrastructure anchored in the Findability, Accessibility, Interoperability, and Reusability (FAIR) principles. By reviewing the current landscape and outlining strategic priorities, this study supports the advancement of data-centric solutions in the construction sector.
>
---
#### [replaced 040] CoPRS: Learning Positional Prior from Chain-of-Thought for Reasoning Segmentation
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2510.11173v2](https://arxiv.org/pdf/2510.11173v2)**

> **作者:** Zhenyu Lu; Liupeng Li; Jinpeng Wang; Yan Feng; Bin Chen; Ke Chen; Yaowei Wang
>
> **备注:** 20 pages, 8 figures, 7 tables
>
> **摘要:** Existing works on reasoning segmentation either connect hidden features from a language model directly to a mask decoder or represent positions in text, which limits interpretability and semantic detail. To solve this, we present CoPRS, a Multi-modal Chain-of-Thought (MCoT)-based positional perception model that bridges language reasoning to segmentation through a differentiable and interpretable positional prior instantiated as a heatmap. By making the reasoning process clear via MCoT and expressing it as a dense, differentiable heatmap, this interface enhances interpretability and diagnostic analysis and yields more concentrated evidence on the target. A learnable concentration token aggregates features of the image and reasoning text to generate this positional prior, which is decoded to precise masks through a lightweight decoder, providing a direct connection between reasoning and segmentation. Across the RefCOCO series and ReasonSeg, CoPRS matches or surpasses the best reported metrics on each standard split under comparable protocols, with performance at or above the prior state of the art across both validation and test partitions. Extensive experiments demonstrate a strong positive correlation among the CoT trajectory, the generated heatmap, and the decoded mask, supporting an interpretable alignment between the reasoning output and downstream mask generation. Collectively, these findings support the utility of this paradigm in bridging reasoning and segmentation and show advantages in concentration driven by reasoning and in more precise mask prediction. Code, checkpoints and logs are released at https://github.com/ZhenyuLU-Heliodore/CoPRS.git.
>
---
#### [replaced 041] OpenSubject: Leveraging Video-Derived Identity and Diversity Priors for Subject-driven Image Generation and Manipulation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.08294v2](https://arxiv.org/pdf/2512.08294v2)**

> **作者:** Yexin Liu; Manyuan Zhang; Yueze Wang; Hongyu Li; Dian Zheng; Weiming Zhang; Changsheng Lu; Xunliang Cai; Yan Feng; Peng Pei; Harry Yang
>
> **摘要:** Despite the promising progress in subject-driven image generation, current models often deviate from the reference identities and struggle in complex scenes with multiple subjects. To address this challenge, we introduce OpenSubject, a video-derived large-scale corpus with 2.5M samples and 4.35M images for subject-driven generation and manipulation. The dataset is built with a four-stage pipeline that exploits cross-frame identity priors. (i) Video Curation. We apply resolution and aesthetic filtering to obtain high-quality clips. (ii) Cross-Frame Subject Mining and Pairing. We utilize vision-language model (VLM)-based category consensus, local grounding, and diversity-aware pairing to select image pairs. (iii) Identity-Preserving Reference Image Synthesis. We introduce segmentation map-guided outpainting to synthesize the input images for subject-driven generation and box-guided inpainting to generate input images for subject-driven manipulation, together with geometry-aware augmentations and irregular boundary erosion. (iv) Verification and Captioning. We utilize a VLM to validate synthesized samples, re-synthesize failed samples based on stage (iii), and then construct short and long captions. In addition, we introduce a benchmark covering subject-driven generation and manipulation, and then evaluate identity fidelity, prompt adherence, manipulation consistency, and background consistency with a VLM judge. Extensive experiments show that training with OpenSubject improves generation and manipulation performance, particularly in complex scenes.
>
---
#### [replaced 042] PlayerOne: Egocentric World Simulator
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.09995v3](https://arxiv.org/pdf/2506.09995v3)**

> **作者:** Yuanpeng Tu; Hao Luo; Xi Chen; Xiang Bai; Fan Wang; Hengshuang Zhao
>
> **备注:** Project page: https://playerone-hku.github.io/
>
> **摘要:** We introduce PlayerOne, the first egocentric realistic world simulator, facilitating immersive and unrestricted exploration within vividly dynamic environments. Given an egocentric scene image from the user, PlayerOne can accurately construct the corresponding world and generate egocentric videos that are strictly aligned with the real scene human motion of the user captured by an exocentric camera. PlayerOne is trained in a coarse-to-fine pipeline that first performs pretraining on large-scale egocentric text-video pairs for coarse-level egocentric understanding, followed by finetuning on synchronous motion-video data extracted from egocentric-exocentric video datasets with our automatic construction pipeline. Besides, considering the varying importance of different components, we design a part-disentangled motion injection scheme, enabling precise control of part-level movements. In addition, we devise a joint reconstruction framework that progressively models both the 4D scene and video frames, ensuring scene consistency in the long-form video generation. Experimental results demonstrate its great generalization ability in precise control of varying human movements and worldconsistent modeling of diverse scenarios. It marks the first endeavor into egocentric real-world simulation and can pave the way for the community to delve into fresh frontiers of world modeling and its diverse applications.
>
---
#### [replaced 043] Enhancing Floor Plan Recognition: A Hybrid Mix-Transformer and U-Net Approach for Precise Wall Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.02413v2](https://arxiv.org/pdf/2512.02413v2)**

> **作者:** Dmitriy Parashchuk; Alexey Kapshitskiy; Yuriy Karyakin
>
> **备注:** 10 pages, 4 figures, 3 tables
>
> **摘要:** Automatic 3D reconstruction of indoor spaces from 2D floor plans necessitates high-precision semantic segmentation of structural elements, particularly walls. However, existing methods often struggle with detecting thin structures and maintaining geometric precision. This study introduces MitUNet, a hybrid neural network combining a Mix-Transformer encoder and a U-Net decoder enhanced with spatial and channel attention blocks. Our approach, optimized with the Tversky loss function, achieves a balance between precision and recall, ensuring accurate boundary recovery. Experiments on the CubiCasa5k dataset and a proprietary regional dataset demonstrate MitUNet's superiority in generating structurally correct masks with high boundary accuracy, outperforming standard models. This tool provides a robust foundation for automated 3D reconstruction pipelines. To ensure reproducibility and facilitate future research, the source code and the proprietary regional dataset are publicly available at https://github.com/aliasstudio/mitunet and https://doi.org/10.5281/zenodo.17871079 respectively.
>
---
#### [replaced 044] InfMasking: Unleashing Synergistic Information by Contrastive Multimodal Interactions
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.25270v3](https://arxiv.org/pdf/2509.25270v3)**

> **作者:** Liangjian Wen; Qun Dai; Jianzhuang Liu; Jiangtao Zheng; Yong Dai; Dongkai Wang; Zhao Kang; Jun Wang; Zenglin Xu; Jiang Duan
>
> **备注:** Conference on Neural Information Processing Systems (NeurIPS) 2025 (Spotlight)
>
> **摘要:** In multimodal representation learning, synergistic interactions between modalities not only provide complementary information but also create unique outcomes through specific interaction patterns that no single modality could achieve alone. Existing methods may struggle to effectively capture the full spectrum of synergistic information, leading to suboptimal performance in tasks where such interactions are critical. This is particularly problematic because synergistic information constitutes the fundamental value proposition of multimodal representation. To address this challenge, we introduce InfMasking, a contrastive synergistic information extraction method designed to enhance synergistic information through an Infinite Masking strategy. InfMasking stochastically occludes most features from each modality during fusion, preserving only partial information to create representations with varied synergistic patterns. Unmasked fused representations are then aligned with masked ones through mutual information maximization to encode comprehensive synergistic information. This infinite masking strategy enables capturing richer interactions by exposing the model to diverse partial modality combinations during training. As computing mutual information estimates with infinite masking is computationally prohibitive, we derive an InfMasking loss to approximate this calculation. Through controlled experiments, we demonstrate that InfMasking effectively enhances synergistic information between modalities. In evaluations on large-scale real-world datasets, InfMasking achieves state-of-the-art performance across seven benchmarks. Code is released at https://github.com/brightest66/InfMasking.
>
---
#### [replaced 045] VFM-ISRefiner: Towards Better Adapting Vision Foundation Models for Interactive Segmentation of Remote Sensing Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.00718v2](https://arxiv.org/pdf/2512.00718v2)**

> **作者:** Deliang Wang; Peng Liu; Yan Ma; Rongkai Zhuang; Lajiao Chen; Bing Li; Yi Zeng
>
> **摘要:** Interactive image segmentation(IIS) plays a critical role in generating precise annotations for remote sensing imagery, where objects often exhibit scale variations, irregular boundaries and complex backgrounds. However, existing IIS methods, primarily designed for natural images, struggle to generalize to remote sensing domains due to limited annotated data and computational overhead. To address these challenges, we proposed RS-ISRefiner, a novel click-based IIS framework tailored for remote sensing images. The framework employs an adapter-based tuning strategy that preserves the general representations of Vision Foundation Models while enabling efficient learning of remote sensing-specific spatial and boundary characteristics. A hybrid attention mechanism integrating convolutional local modeling with Transformer-based global reasoning enhances robustness against scale diversity and scene complexity. Furthermore, an improved probability map modulation scheme effectively incorporates historical user interactions, yielding more stable iterative refinement and higher boundary accuracy. Comprehensive experiments on six remote sensing datasets, including iSAID, ISPRS Potsdam, SandBar, NWPU, LoveDA Urban and WHUBuilding, demonstrate that RS-ISRefiner consistently outperforms state-of-the-art IIS methods in terms of segmentation accuracy, efficiency and interaction cost. These results confirm the effectiveness and generalizability of our framework, making it highly suitable for high-quality instance segmentation in practical remote sensing scenarios. The codes are available at https://github.com/wondelyan/VFM-ISRefiner .
>
---
#### [replaced 046] GeoDM: Geometry-aware Distribution Matching for Dataset Distillation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.08317v2](https://arxiv.org/pdf/2512.08317v2)**

> **作者:** Xuhui Li; Zhengquan Luo; Zihui Cui; Zhiqiang Xu
>
> **摘要:** Dataset distillation aims to synthesize a compact subset of the original data, enabling models trained on it to achieve performance comparable to those trained on the original large dataset. Existing distribution-matching methods are confined to Euclidean spaces, making them only capture linear structures and overlook the intrinsic geometry of real data, e.g., curvature. However, high-dimensional data often lie on low-dimensional manifolds, suggesting that dataset distillation should have the distilled data manifold aligned with the original data manifold. In this work, we propose a geometry-aware distribution-matching framework, called \textbf{GeoDM}, which operates in the Cartesian product of Euclidean, hyperbolic, and spherical manifolds, with flat, hierarchical, and cyclical structures all captured by a unified representation. To adapt to the underlying data geometry, we introduce learnable curvature and weight parameters for three kinds of geometries. At the same time, we design an optimal transport loss to enhance the distribution fidelity. Our theoretical analysis shows that the geometry-aware distribution matching in a product space yields a smaller generalization error bound than the Euclidean counterparts. Extensive experiments conducted on standard benchmarks demonstrate that our algorithm outperforms state-of-the-art data distillation methods and remains effective across various distribution-matching strategies for the single geometries.
>
---
#### [replaced 047] Tokenizing Motion: A Generative Approach for Scene Dynamics Compression
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2410.09768v4](https://arxiv.org/pdf/2410.09768v4)**

> **作者:** Shanzhi Yin; Zihan Zhang; Bolin Chen; Shiqi Wang; Yan Ye
>
> **备注:** 5page, 5 figures
>
> **摘要:** This paper proposes a novel generative video compression framework that leverages motion pattern priors, derived from subtle dynamics in common scenes (e.g., swaying flowers or a boat drifting on water), rather than relying on video content priors (e.g., talking faces or human bodies). These compact motion priors enable a new approach to ultra-low bitrate communication while achieving high-quality reconstruction across diverse scene contents. At the encoder side, motion priors can be streamlined into compact representations via a dense-to-sparse transformation. At the decoder side, these priors facilitate the reconstruction of scene dynamics using an advanced flow-driven diffusion model. Experimental results illustrate that the proposed method can achieve superior rate-distortion-performance and outperform the state-of-the-art conventional-video codec Enhanced Compression Model (ECM) on-scene dynamics sequences. The project page can be found at-https://github.com/xyzysz/GNVDC.
>
---
#### [replaced 048] Weight Space Representation Learning on Diverse NeRF Architectures
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.09623v3](https://arxiv.org/pdf/2502.09623v3)**

> **作者:** Francesco Ballerini; Pierluigi Zama Ramirez; Luigi Di Stefano; Samuele Salti
>
> **备注:** v3: added quantitative Objaverse retrieval and language tasks. Under review
>
> **摘要:** Neural Radiance Fields (NeRFs) have emerged as a groundbreaking paradigm for representing 3D objects and scenes by encoding shape and appearance information into the weights of a neural network. Recent studies have demonstrated that these weights can be used as input for frameworks designed to address deep learning tasks; however, such frameworks require NeRFs to adhere to a specific, predefined architecture. In this paper, we introduce the first framework capable of processing NeRFs with diverse architectures and performing inference on architectures unseen at training time. We achieve this by training a Graph Meta-Network within an unsupervised representation learning framework, and show that a contrastive objective is conducive to obtaining an architecture-agnostic latent space. In experiments conducted across 13 NeRF architectures belonging to three families (MLPs, tri-planes, and, for the first time, hash tables), our approach demonstrates robust performance in classification, retrieval, and language tasks involving multiple architectures, even unseen at training time, while also matching or exceeding the results of existing frameworks limited to single architectures.
>
---
#### [replaced 049] Towards Robust Infrared Small Target Detection: A Feature-Enhanced and Sensitivity-Tunable Framework
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2407.20090v3](https://arxiv.org/pdf/2407.20090v3)**

> **作者:** Jinmiao Zhao; Zelin Shi; Chuang Yu; Yunpeng Liu; Yimian Dai
>
> **备注:** Accepted by Knowledge-Based Systems 2025
>
> **摘要:** Recently, single-frame infrared small target (SIRST) detection technology has attracted widespread attention. Different from most existing deep learning-based methods that focus on improving network architectures, we propose a feature-enhanced and sensitivity-tunable (FEST) framework, which is compatible with existing SIRST detection networks and further enhances their detection performance. The FEST framework improves the model's robustness from two aspects: feature enhancement and target confidence regulation. For feature enhancement, we employ a multi-scale fusion strategy to improve the model's perception to multi-scale features of multi-size targets, and design an edge enhancement difficulty mining (EEDM) loss to guide the network to continuously focus on challenging target regions and edge features during training. For target confidence regulation, an adjustable sensitivity (AS) strategy is proposed for network post-processing. This strategy enhances the model's adaptability in complex scenarios and significantly improves the detection rate of infrared small targets while maintaining segmentation accuracy. Extensive experimental results show that our FEST framework can effectively enhance the performance of existing SIRST detection networks. The code is available at https://github.com/YuChuang1205/FEST-Framework
>
---
#### [replaced 050] Seedream 4.0: Toward Next-generation Multimodal Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.20427v3](https://arxiv.org/pdf/2509.20427v3)**

> **作者:** Team Seedream; :; Yunpeng Chen; Yu Gao; Lixue Gong; Meng Guo; Qiushan Guo; Zhiyao Guo; Xiaoxia Hou; Weilin Huang; Yixuan Huang; Xiaowen Jian; Huafeng Kuang; Zhichao Lai; Fanshi Li; Liang Li; Xiaochen Lian; Chao Liao; Liyang Liu; Wei Liu; Yanzuo Lu; Zhengxiong Luo; Tongtong Ou; Guang Shi; Yichun Shi; Shiqi Sun; Yu Tian; Zhi Tian; Peng Wang; Rui Wang; Xun Wang; Ye Wang; Guofeng Wu; Jie Wu; Wenxu Wu; Yonghui Wu; Xin Xia; Xuefeng Xiao; Shuang Xu; Xin Yan; Ceyuan Yang; Jianchao Yang; Zhonghua Zhai; Chenlin Zhang; Heng Zhang; Qi Zhang; Xinyu Zhang; Yuwei Zhang; Shijia Zhao; Wenliang Zhao; Wenjia Zhu
>
> **备注:** Seedream 4.0/4.5 Technical Report
>
> **摘要:** We introduce Seedream 4.0, an efficient and high-performance multimodal image generation system that unifies text-to-image (T2I) synthesis, image editing, and multi-image composition within a single framework. We develop a highly efficient diffusion transformer with a powerful VAE which also can reduce the number of image tokens considerably. This allows for efficient training of our model, and enables it to fast generate native high-resolution images (e.g., 1K-4K). Seedream 4.0 is pretrained on billions of text-image pairs spanning diverse taxonomies and knowledge-centric concepts. Comprehensive data collection across hundreds of vertical scenarios, coupled with optimized strategies, ensures stable and large-scale training, with strong generalization. By incorporating a carefully fine-tuned VLM model, we perform multi-modal post-training for training both T2I and image editing tasks jointly. For inference acceleration, we integrate adversarial distillation, distribution matching, and quantization, as well as speculative decoding. It achieves an inference time of up to 1.8 seconds for generating a 2K image (without a LLM/VLM as PE model). Comprehensive evaluations reveal that Seedream 4.0 can achieve state-of-the-art results on both T2I and multimodal image editing. In particular, it demonstrates exceptional multimodal capabilities in complex tasks, including precise image editing and in-context reasoning, and also allows for multi-image reference, and can generate multiple output images. This extends traditional T2I systems into an more interactive and multidimensional creative tool, pushing the boundary of generative AI for both creativity and professional applications. We further scale our model and data as Seedream 4.5. Seedream 4.0 and Seedream 4.5 are accessible on Volcano Engine https://www.volcengine.com/experience/ark?launch=seedream.
>
---
#### [replaced 051] Self-Paced and Self-Corrective Masked Prediction for Movie Trailer Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.04426v2](https://arxiv.org/pdf/2512.04426v2)**

> **作者:** Sidan Zhu; Hongteng Xu; Dixin Luo
>
> **摘要:** As a challenging video editing task, movie trailer generation involves selecting and reorganizing movie shots to create engaging trailers. Currently, most existing automatic trailer generation methods employ a "selection-then-ranking" paradigm (i.e., first selecting key shots and then ranking them), which suffers from inevitable error propagation and limits the quality of the generated trailers. Beyond this paradigm, we propose a new self-paced and self-corrective masked prediction method called SSMP, which achieves state-of-the-art results in automatic trailer generation via bi-directional contextual modeling and progressive self-correction. In particular, SSMP trains a Transformer encoder that takes the movie shot sequences as prompts and generates corresponding trailer shot sequences accordingly. The model is trained via masked prediction, reconstructing each trailer shot sequence from its randomly masked counterpart. The mask ratio is self-paced, allowing the task difficulty to adapt to the model and thereby improving model performance. When generating a movie trailer, the model fills the shot positions with high confidence at each step and re-masks the remaining positions for the next prediction, forming a progressive self-correction mechanism that is analogous to how human editors work. Both quantitative results and user studies demonstrate the superiority of SSMP in comparison to existing automatic movie trailer generation methods. Demo is available at: https://github.com/Dixin-Lab/SSMP.
>
---
#### [replaced 052] Evaluating Small Vision-Language Models on Distance-Dependent Traffic Perception
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.08352v2](https://arxiv.org/pdf/2510.08352v2)**

> **作者:** Nikos Theodoridis; Tim Brophy; Reenu Mohandas; Ganesh Sistu; Fiachra Collins; Anthony Scanlan; Ciaran Eising
>
> **备注:** Published in IEEE Open Journal of Vehicular Technology. Final version available at: https://ieeexplore.ieee.org/document/11230063
>
> **摘要:** Vision-Language Models (VLMs) are becoming increasingly powerful, demonstrating strong performance on a variety of tasks that require both visual and textual understanding. Their strong generalisation abilities make them a promising component for automated driving systems, which must handle unexpected corner cases. However, to be trusted in such safety-critical applications, a model must first possess a reliable perception system. Moreover, since critical objects and agents in traffic scenes are often at a distance, we require systems that are not "shortsighted", i.e., systems with strong perception capabilities at both close (up to 20 meters) and long (30+ meters) range. With this in mind, we introduce Distance-Annotated Traffic Perception Question Answering (DTPQA), the first Visual Question Answering (VQA) benchmark focused solely on perception-based questions in traffic scenes, enriched with distance annotations. By excluding questions that require reasoning, we ensure that model performance reflects perception capabilities alone. Since automated driving hardware has limited processing power and cannot support large VLMs, our study centers on smaller VLMs. More specifically, we evaluate several state-of-the-art (SOTA) small VLMs on DTPQA and show that, despite the simplicity of the questions, these models significantly underperform compared to humans (~60% average accuracy for the best-performing small VLM versus ~85% human performance). However, it is important to note that the human sample size was relatively small, which imposes statistical limitations. We also identify specific perception tasks, such as distinguishing left from right, that remain particularly challenging for these models.
>
---
#### [replaced 053] TeleEgo: Benchmarking Egocentric AI Assistants in the Wild
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.23981v4](https://arxiv.org/pdf/2510.23981v4)**

> **作者:** Jiaqi Yan; Ruilong Ren; Jingren Liu; Shuning Xu; Ling Wang; Yiheng Wang; Xinlin Zhong; Yun Wang; Long Zhang; Xiangyu Chen; Changzhi Sun; Jixiang Luo; Dell Zhang; Hao Sun; Chi Zhang; Xuelong Li
>
> **摘要:** Egocentric AI assistants in real-world settings must process multi-modal inputs (video, audio, text), respond in real time, and retain evolving long-term memory. However, existing benchmarks typically evaluate these abilities in isolation, lack realistic streaming scenarios, or support only short-term tasks. We introduce \textbf{TeleEgo}, a long-duration, streaming, omni-modal benchmark for evaluating egocentric AI assistants in realistic daily contexts. The dataset features over 14 hours per participant of synchronized egocentric video, audio, and text across four domains: work \& study, lifestyle \& routines, social activities, and outings \& culture. All data is aligned on a unified global timeline and includes high-quality visual narrations and speech transcripts, curated through human refinement.TeleEgo defines 12 diagnostic subtasks across three core capabilities: Memory (recalling past events), Understanding (interpreting the current moment), and Cross-Memory Reasoning (linking distant events). It contains 3,291 human-verified QA items spanning multiple question formats (single-choice, binary, multi-choice, and open-ended), evaluated strictly in a streaming setting. We propose Real-Time Accuracy (RTA) to jointly capture correctness and responsiveness under tight decision windows, and Memory Persistence Time (MPT) as a forward-looking metric for long-term retention in continuous streams. In this work, we report RTA results for current models and release TeleEgo, together with an MPT evaluation framework, as a realistic and extensible benchmark for future egocentric assistants with stronger streaming memory, enabling systematic study of both real-time behavior and long-horizon memory.
>
---
#### [replaced 054] Learning to Infer Parameterized Representations of Plants from 3D Scans
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.22337v2](https://arxiv.org/pdf/2505.22337v2)**

> **作者:** Samara Ghrer; Christophe Godin; Stefanie Wuhrer
>
> **摘要:** Plants frequently contain numerous organs, organized in 3D branching systems defining the plant's architecture. Reconstructing the architecture of plants from unstructured observations is challenging because of self-occlusion and spatial proximity between organs, which are often thin structures. To achieve the challenging task, we propose an approach that allows to infer a parameterized representation of the plant's architecture from a given 3D scan of a plant. In addition to the plant's branching structure, this representation contains parametric information for each plant organ, and can therefore be used directly in a variety of tasks. In this data-driven approach, we train a recursive neural network with virtual plants generated using a procedural model. After training, the network allows to infer a parametric tree-like representation based on an input 3D point cloud. Our method is applicable to any plant that can be represented as binary axial tree. We quantitatively evaluate our approach on Chenopodium Album plants on reconstruction, segmentation and skeletonization, which are important problems in plant phenotyping. In addition to carrying out several tasks at once, our method achieves results on-par with strong baselines for each task. We apply our method, trained exclusively on synthetic data, to 3D scans and show that it generalizes well.
>
---
#### [replaced 055] Do You See Me : A Multidimensional Benchmark for Evaluating Visual Perception in Multimodal LLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.02022v2](https://arxiv.org/pdf/2506.02022v2)**

> **作者:** Aditya Kanade; Tanuja Ganu
>
> **摘要:** Multimodal Large Language Models (MLLMs) show reasoning promise, yet their visual perception is a critical bottleneck. Strikingly, MLLMs can produce correct answers even while misinterpreting crucial visual elements, masking these underlying failures. Our preliminary study on a joint perception-reasoning dataset revealed that for one leading MLLM, 29% of its correct answers to reasoning questions still exhibited visual perception errors. To systematically address this, we introduce "Do You See Me", a scalable benchmark with 1,758 images and 2,612 questions. It spans seven human-psychology inspired subtasks in 2D and 3D, featuring controllable complexity to rigorously evaluate MLLM visual skills. Our findings on 3 leading closed-source and 5 major open-source models reveal a stark deficit: humans achieve 96.49% accuracy, while top MLLMs average below 50%. This performance gap widens rapidly with increased task complexity (e.g., from 12% to 45% in the visual form constancy subtask). Further analysis into the root causes suggests that failures stem from challenges like misallocated visual attention and the instability of internal representations for fine-grained details, especially at or below encoder patch resolution. This underscores an urgent need for MLLMs with truly robust visual perception. The benchmark dataset, source code and evaluation scripts are available at https://github.com/microsoft/Do-You-See-Me.
>
---
#### [replaced 056] DELTAv2: Accelerating Dense 3D Tracking
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.01170v2](https://arxiv.org/pdf/2508.01170v2)**

> **作者:** Tuan Duc Ngo; Ashkan Mirzaei; Guocheng Qian; Hanwen Liang; Chuang Gan; Evangelos Kalogerakis; Peter Wonka; Chaoyang Wang
>
> **备注:** Project page: https://snap-research.github.io/DELTAv2/
>
> **摘要:** We propose a novel algorithm for accelerating dense long-term 3D point tracking in videos. Through analysis of existing state-of-the-art methods, we identify two major computational bottlenecks. First, transformer-based iterative tracking becomes expensive when handling a large number of trajectories. To address this, we introduce a coarse-to-fine strategy that begins tracking with a small subset of points and progressively expands the set of tracked trajectories. The newly added trajectories are initialized using a learnable interpolation module, which is trained end-to-end alongside the tracking network. Second, we propose an optimization that significantly reduces the cost of correlation feature computation, another key bottleneck in prior methods. Together, these improvements lead to a 5-100x speedup over existing approaches while maintaining state-of-the-art tracking accuracy.
>
---
#### [replaced 057] AURORA:Augmented Understanding via Structured Reasoning and Reinforcement Learning for Reference Audio-Visual Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.02149v2](https://arxiv.org/pdf/2508.02149v2)**

> **作者:** Ziyang Luo; Nian Liu; Fahad Shahbaz Khan; Junwei Han
>
> **备注:** AAAI2026,code:https://github.com/Sssssuperior/AURORA
>
> **摘要:** Reference Audio-Visual Segmentation (Ref-AVS) tasks challenge models to precisely locate sounding objects by integrating visual, auditory, and textual cues. Existing methods often lack genuine semantic understanding, tending to memorize fixed reasoning patterns. Furthermore, jointly training for reasoning and segmentation can compromise pixel-level precision. To address these issues, we introduce AURORA, a novel framework designed to enhance genuine reasoning and language comprehension in reference audio-visual segmentation. We employ a structured Chain-of-Thought (CoT) prompting mechanism to guide the model through a step-by-step reasoning process and introduce a novel segmentation feature distillation loss to effectively integrate these reasoning abilities without sacrificing segmentation performance. To further cultivate the model's genuine reasoning capabilities, we devise a further two-stage training strategy: first, a ``corrective reflective-style training" stage utilizes self-correction to enhance the quality of reasoning paths, followed by reinforcement learning via Group Reward Policy Optimization (GRPO) to bolster robustness in challenging scenarios. Experiments demonstrate that AURORA achieves state-of-the-art performance on Ref-AVS benchmarks and generalizes effectively to unreferenced segmentation.
>
---
#### [replaced 058] Learning What Matters: Steering Diffusion via Spectrally Anisotropic Forward Noise
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.09660v4](https://arxiv.org/pdf/2510.09660v4)**

> **作者:** Luca Scimeca; Thomas Jiralerspong; Berton Earnshaw; Jason Hartford; Yoshua Bengio
>
> **摘要:** Diffusion Probabilistic Models (DPMs) have achieved strong generative performance, yet their inductive biases remain largely implicit. In this work, we aim to build inductive biases into the training and sampling of diffusion models to better accommodate the target distribution of the data to model. We introduce an anisotropic noise operator that shapes these biases by replacing the isotropic forward covariance with a structured, frequency-diagonal covariance. This operator unifies band-pass masks and power-law weightings, allowing us to emphasize or suppress designated frequency bands, while keeping the forward process Gaussian. We refer to this as Spectrally Anisotropic Gaussian Diffusion (SAGD). In this work, we derive the score relation for anisotropic forward covariances and show that, under full support, the learned score converges to the true data score as $t\!\to\!0$, while anisotropy reshapes the probability-flow path from noise to data. Empirically, we show the induced anisotropy outperforms standard diffusion across several vision datasets, and enables selective omission: learning while ignoring known corruptions confined to specific bands. Together, these results demonstrate that carefully designed anisotropic forward noise provides a simple, yet principled, handle to tailor inductive bias in DPMs.
>
---
#### [replaced 059] Decoupling Template Bias in CLIP: Harnessing Empty Prompts for Enhanced Few-Shot Learning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.08606v2](https://arxiv.org/pdf/2512.08606v2)**

> **作者:** Zhenyu Zhang; Guangyao Chen; Yixiong Zou; Zhimeng Huang; Yuhua Li
>
> **备注:** 14 pages, 8 figures, Association for the Advancement of Artificial Intelligence (AAAI2026, poster)
>
> **摘要:** The Contrastive Language-Image Pre-Training (CLIP) model excels in few-shot learning by aligning visual and textual representations. Our study shows that template-sample similarity (TSS), defined as the resemblance between a text template and an image sample, introduces bias. This bias leads the model to rely on template proximity rather than true sample-to-category alignment, reducing both accuracy and robustness in classification. We present a framework that uses empty prompts, textual inputs that convey the idea of "emptiness" without category information. These prompts capture unbiased template features and offset TSS bias. The framework employs two stages. During pre-training, empty prompts reveal and reduce template-induced bias within the CLIP encoder. During few-shot fine-tuning, a bias calibration loss enforces correct alignment between images and their categories, ensuring the model focuses on relevant visual cues. Experiments across multiple benchmarks demonstrate that our template correction method significantly reduces performance fluctuations caused by TSS, yielding higher classification accuracy and stronger robustness. The repository of this project is available at https://github.com/zhenyuZ-HUST/Decoupling-Template-Bias-in-CLIP.
>
---
#### [replaced 060] Human Motion Unlearning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.18674v3](https://arxiv.org/pdf/2503.18674v3)**

> **作者:** Edoardo De Matteis; Matteo Migliarini; Alessio Sampieri; Indro Spinelli; Fabio Galasso
>
> **摘要:** We introduce Human Motion Unlearning and motivate it through the concrete task of preventing violent 3D motion synthesis, an important safety requirement given that popular text-to-motion datasets (HumanML3D and Motion-X) contain from 7\% to 15\% violent sequences spanning both atomic gestures (e.g., a single punch) and highly compositional actions (e.g., loading and swinging a leg to kick). By focusing on violence unlearning, we demonstrate how removing a challenging, multifaceted concept can serve as a proxy for the broader capability of motion "forgetting." To enable systematic evaluation of Human Motion Unlearning, we establish the first motion unlearning benchmark by automatically filtering HumanML3D and Motion-X datasets to create distinct forget sets (violent motions) and retain sets (safe motions). We introduce evaluation metrics tailored to sequential unlearning, measuring both suppression efficacy and the preservation of realism and smooth transitions. We adapt two state-of-the-art, training-free image unlearning methods (UCE and RECE) to leading text-to-motion architectures (MoMask and BAMM), and propose Latent Code Replacement (LCR), a novel, training-free approach that identifies violent codes in a discrete codebook representation and substitutes them with safe alternatives. Our experiments show that unlearning violent motions is indeed feasible and that acting on latent codes strikes the best trade-off between violence suppression and preserving overall motion quality. This work establishes a foundation for advancing safe motion synthesis across diverse applications. Website: https://www.pinlab.org/hmu.
>
---
#### [replaced 061] AugLift: Uncertainty Aware Depth Descriptors for Robust 2D to 3D Pose Lifting
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.07112v3](https://arxiv.org/pdf/2508.07112v3)**

> **作者:** Nikolai Warner; Wenjin Zhang; Hamid Badiozamani; Irfan Essa; Apaar Sadhwani
>
> **备注:** Preprint. Under review
>
> **摘要:** Lifting based 3D human pose estimators infer 3D joints from 2D keypoints, but often struggle to generalize to real world settings with noisy 2D detections. We revisit the input to lifting and propose AugLift, a simple augmentation of standard lifting that enriches each 2D keypoint (x, y) with an Uncertainty Aware Depth Descriptor (UADD). We run a single off the shelf monocular depth estimator to obtain a depth map, and for every keypoint with detector confidence c we extract depth statistics from its confidence scaled neighborhood, forming a compact, interpretable UADD (c, d, d_min, d_max) that captures both local geometry and reliability. AugLift is modular, requires no new sensors or architectural changes, and integrates by expanding the input layer of existing lifting models. Across four datasets and four lifting architectures, AugLift boosts cross dataset (out of distribution) performance on unseen data by an average of 10.1 percent, while also improving in distribution performance by 4.0 percent as measured by MPJPE. A post hoc analysis clarifies when and why it helps: gains are largest on novel poses and significantly occluded joints, where depth statistics resolve front back ambiguities while confidence calibrates the spatial neighborhoods from which they are drawn. We also study interaction with recent image feature lifting methods and find the signals are complementary: adding UADD to image conditioned lifting yields both ID and OOD gains. A learned depth feature extension (AugLiftV2) improves performance further while trading off interpretability. Together, these results indicate that lightweight, confidence aware depth cues are a powerful plug in for robust 2D to 3D pose lifting.
>
---
#### [replaced 062] DISTA-Net: Dynamic Closely-Spaced Infrared Small Target Unmixing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.19148v2](https://arxiv.org/pdf/2505.19148v2)**

> **作者:** Shengdong Han; Shangdong Yang; Xin Zhang; Yuxuan Li; Xiang Li; Jian Yang; Ming-Ming Cheng; Yimian Dai
>
> **备注:** Accepted by ICCV 2025. This updated version fixed the bug in SSIM, while the conclusion remains the same
>
> **摘要:** Resolving closely-spaced small targets in dense clusters presents a significant challenge in infrared imaging, as the overlapping signals hinder precise determination of their quantity, sub-pixel positions, and radiation intensities. While deep learning has advanced the field of infrared small target detection, its application to closely-spaced infrared small targets has not yet been explored. This gap exists primarily due to the complexity of separating superimposed characteristics and the lack of an open-source infrastructure. In this work, we propose the Dynamic Iterative Shrinkage Thresholding Network (DISTA-Net), which reconceptualizes traditional sparse reconstruction within a dynamic framework. DISTA-Net adaptively generates convolution weights and thresholding parameters to tailor the reconstruction process in real time. To the best of our knowledge, DISTA-Net is the first deep learning model designed specifically for the unmixing of closely-spaced infrared small targets, achieving superior sub-pixel detection accuracy. Moreover, we have established the first open-source ecosystem to foster further research in this field. This ecosystem comprises three key components: (1) CSIST-100K, a publicly available benchmark dataset; (2) CSO-mAP, a custom evaluation metric for sub-pixel detection; and (3) GrokCSO, an open-source toolkit featuring DISTA-Net and other models. Our code and dataset are available at https://github.com/GrokCV/GrokCSO.
>
---
#### [replaced 063] Toward Efficient and Robust Behavior Models for Multi-Agent Driving Simulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究多智能体驾驶仿真中的行为建模，旨在提升模型效率与鲁棒性。提出实例中心的场景表示和查询中心的对称上下文编码器，结合对抗逆强化学习与自适应奖励机制，实现高效、高精度的驾驶行为模拟。**

- **链接: [https://arxiv.org/pdf/2512.05812v2](https://arxiv.org/pdf/2512.05812v2)**

> **作者:** Fabian Konstantinidis; Moritz Sackmann; Ulrich Hofmann; Christoph Stiller
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Scalable multi-agent driving simulation requires behavior models that are both realistic and computationally efficient. We address this by optimizing the behavior model that controls individual traffic participants. To improve efficiency, we adopt an instance-centric scene representation, where each traffic participant and map element is modeled in its own local coordinate frame. This design enables efficient, viewpoint-invariant scene encoding and allows static map tokens to be reused across simulation steps. To model interactions, we employ a query-centric symmetric context encoder with relative positional encodings between local frames. We use Adversarial Inverse Reinforcement Learning to learn the behavior model and propose an adaptive reward transformation that automatically balances robustness and realism during training. Experiments demonstrate that our approach scales efficiently with the number of tokens, significantly reducing training and inference times, while outperforming several agent-centric baselines in terms of positional accuracy and robustness.
>
---
#### [replaced 064] WeatherDiffusion: Controllable Weather Editing in Intrinsic Space
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.06982v4](https://arxiv.org/pdf/2508.06982v4)**

> **作者:** Yixin Zhu; Zuoliang Zhu; Jian Yang; Miloš Hašan; Jin Xie; Beibei Wang
>
> **摘要:** We present WeatherDiffusion, a diffusion-based framework for controllable weather editing in intrinsic space. Our framework includes two components based on diffusion priors: an inverse renderer that estimates material properties, scene geometry, and lighting as intrinsic maps from an input image, and a forward renderer that utilizes these geometry and material maps along with a text prompt that describes specific weather conditions to generate a final image. The intrinsic maps enhance controllability compared to traditional pixel-space editing approaches. We propose an intrinsic map-aware attention mechanism that improves spatial correspondence and decomposition quality in large outdoor scenes. For forward rendering, we leverage CLIP-space interpolation of weather prompts to achieve fine-grained weather control. We also introduce a synthetic and a real-world dataset, containing 38k and 18k images under various weather conditions, each with intrinsic map annotations. WeatherDiffusion outperforms state-of-the-art pixel-space editing approaches, weather restoration methods, and rendering-based methods, showing promise for downstream tasks such as autonomous driving, enhancing the robustness of detection and segmentation in challenging weather scenarios.
>
---
#### [replaced 065] Sequence models for continuous cell cycle stage prediction from brightfield images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.02182v2](https://arxiv.org/pdf/2502.02182v2)**

> **作者:** Louis-Alexandre Leger; Maxine Leonardi; Andrea Salati; Felix Naef; Martin Weigert
>
> **摘要:** Understanding cell cycle dynamics is crucial for studying biological processes such as growth, development and disease progression. While fluorescent protein reporters like the Fucci system allow live monitoring of cell cycle phases, they require genetic engineering and occupy additional fluorescence channels, limiting broader applicability in complex experiments. In this study, we conduct a comprehensive evaluation of deep learning methods for predicting continuous Fucci signals using non-fluorescence brightfield imaging, a widely available label-free modality. To that end, we generated a large dataset of 1.3 M images of dividing RPE1 cells with full cell cycle trajectories to quantitatively compare the predictive performance of distinct model categories including single time-frame models, causal state space models and bidirectional transformer models. We show that both causal and transformer-based models significantly outperform single- and fixed frame approaches, enabling the prediction of visually imperceptible transitions like G1/S within 1h resolution. Our findings underscore the importance of sequence models for accurate predictions of cell cycle dynamics and highlight their potential for label-free imaging.
>
---
#### [replaced 066] Make LVLMs Focus: Context-Aware Attention Modulation for Better Multimodal In-Context Learning
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究多模态上下文学习中大视觉语言模型注意力机制的局限性，提出无需训练的上下文感知调制注意力方法（CAMA），通过动态调整注意力增强对关键视觉和语义信息的关注，提升模型在多种任务下的稳定性和性能。**

- **链接: [https://arxiv.org/pdf/2505.17097v4](https://arxiv.org/pdf/2505.17097v4)**

> **作者:** Yanshu Li; Jianjiang Yang; Ziteng Yang; Bozheng Li; Ligong Han; Hongyang He; Zhengtao Yao; Yingjie Victor Chen; Songlin Fei; Dongfang Liu; Ruixiang Tang
>
> **备注:** 14 pages, 8 figures, 5 tables
>
> **摘要:** Multimodal in-context learning (ICL) is becoming a key capability that allows large vision-language models (LVLMs) to adapt to novel tasks without parameter updates, which expands their usefulness in many real-world applications. However, ICL performance remains unstable even when the in-context demonstrations (ICDs) are well matched, showing that LVLMs still struggle to make full use of the provided context. While existing work mainly focuses on prompt engineering or post-hoc logit calibration, we study the attention mechanisms inside LVLMs to address their inherent limitations. We identify two important weaknesses in their self-attention that hinder effective ICL. To address these weaknesses, we propose Context-Aware Modulated Attention (CAMA), a training-free and plug-and-play method that dynamically adjusts attention logits based on the input in-context sequence. CAMA uses a two-stage modulation process that strengthens attention to semantically important tokens, especially visual ones. Across four LVLMs and seven benchmarks, CAMA consistently outperforms vanilla models and baselines, showing clear effectiveness and generalization. It can also activate the intended benefits of prompt engineering methods and remains robust across different sequence configurations. Therefore, CAMA opens up new directions for improving multimodal reasoning through a deeper understanding of attention dynamics.
>
---
