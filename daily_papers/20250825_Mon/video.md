# 计算机视觉 cs.CV

- **最新发布 80 篇**

- **更新 50 篇**

## 最新发布

#### [new 001] Forecast then Calibrate: Feature Caching as ODE for Efficient Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文针对扩散Transformer（DiT）推理效率低的问题，提出FoCa方法，将特征缓存建模为常微分方程求解问题，通过预测后校准提升长步跳转下的生成质量，在图像、视频生成等任务中实现高效且高质量的加速。**

- **链接: [http://arxiv.org/pdf/2508.16211v1](http://arxiv.org/pdf/2508.16211v1)**

> **作者:** Shikang Zheng; Liang Feng; Xinyu Wang; Qinming Zhou; Peiliang Cai; Chang Zou; Jiacheng Liu; Yuqi Lin; Junjie Chen; Yue Ma; Linfeng Zhang
>
> **摘要:** Diffusion Transformers (DiTs) have demonstrated exceptional performance in high-fidelity image and video generation. To reduce their substantial computational costs, feature caching techniques have been proposed to accelerate inference by reusing hidden representations from previous timesteps. However, current methods often struggle to maintain generation quality at high acceleration ratios, where prediction errors increase sharply due to the inherent instability of long-step forecasting. In this work, we adopt an ordinary differential equation (ODE) perspective on the hidden-feature sequence, modeling layer representations along the trajectory as a feature-ODE. We attribute the degradation of existing caching strategies to their inability to robustly integrate historical features under large skipping intervals. To address this, we propose FoCa (Forecast-then-Calibrate), which treats feature caching as a feature-ODE solving problem. Extensive experiments on image synthesis, video generation, and super-resolution tasks demonstrate the effectiveness of FoCa, especially under aggressive acceleration. Without additional training, FoCa achieves near-lossless speedups of 5.50 times on FLUX, 6.45 times on HunyuanVideo, 3.17 times on Inf-DiT, and maintains high quality with a 4.53 times speedup on DiT.
>
---
#### [new 002] DRespNeT: A UAV Dataset and YOLOv8-DRN Model for Aerial Instance Segmentation of Building Access Points for Post-Earthquake Search-and-Rescue Missions
- **分类: cs.CV**

- **简介: 论文提出DRespNeT数据集和YOLOv8-DRN模型，用于无人机航拍图像中建筑物入口的实例分割，解决地震后搜救中快速识别可进入通道的问题。该任务提升实时态势感知与救援效率。**

- **链接: [http://arxiv.org/pdf/2508.16016v1](http://arxiv.org/pdf/2508.16016v1)**

> **作者:** Aykut Sirma; Angelos Plastropoulos; Argyrios Zolotas; Gilbert Tang
>
> **备注:** Technical Paper of Scientific data paper: UAV imagery dataset from 2023 Turkiye earthquakes, annotated for instance segmentation to support SAR robotics. Dataset will be released upon acceptance
>
> **摘要:** Recent advancements in computer vision and deep learning have enhanced disaster-response capabilities, particularly in the rapid assessment of earthquake-affected urban environments. Timely identification of accessible entry points and structural obstacles is essential for effective search-and-rescue (SAR) operations. To address this need, we introduce DRespNeT, a high-resolution dataset specifically developed for aerial instance segmentation of post-earthquake structural environments. Unlike existing datasets, which rely heavily on satellite imagery or coarse semantic labeling, DRespNeT provides detailed polygon-level instance segmentation annotations derived from high-definition (1080p) aerial footage captured in disaster zones, including the 2023 Turkiye earthquake and other impacted regions. The dataset comprises 28 operationally critical classes, including structurally compromised buildings, access points such as doors, windows, and gaps, multiple debris levels, rescue personnel, vehicles, and civilian visibility. A distinctive feature of DRespNeT is its fine-grained annotation detail, enabling differentiation between accessible and obstructed areas, thereby improving operational planning and response efficiency. Performance evaluations using YOLO-based instance segmentation models, specifically YOLOv8-seg, demonstrate significant gains in real-time situational awareness and decision-making. Our optimized YOLOv8-DRN model achieves 92.7% mAP50 with an inference speed of 27 FPS on an RTX-4090 GPU for multi-target detection, meeting real-time operational requirements. The dataset and models support SAR teams and robotic systems, providing a foundation for enhancing human-robot collaboration, streamlining emergency response, and improving survivor outcomes.
>
---
#### [new 003] Learning Long-Range Action Representation by Two-Stream Mamba Pyramid Network for Figure Skating Assessment
- **分类: cs.CV; cs.MM**

- **简介: 论文提出两流Mamba金字塔网络，用于花样滑冰技术分（TES）和节目内容分（PCS）评估。解决视频音频特征混用、动作元素未分段评分及长视频处理效率低的问题，通过分离视觉与视听流并融合多尺度特征，实现精准评分。**

- **链接: [http://arxiv.org/pdf/2508.16291v1](http://arxiv.org/pdf/2508.16291v1)**

> **作者:** Fengshun Wang; Qiurui Wang; Peilin Zhao
>
> **摘要:** Technical Element Score (TES) and Program Component Score (PCS) evaluations in figure skating demand precise assessment of athletic actions and artistic interpretation, respectively. Existing methods face three major challenges. Firstly, video and audio cues are regarded as common features for both TES and PCS predictions in previous works without considering the prior evaluation criterion of figure skating. Secondly, action elements in competitions are separated in time, TES should be derived from each element's score, but existing methods try to give an overall TES prediction without evaluating each action element. Thirdly, lengthy competition videos make it difficult and inefficient to handle long-range contexts. To address these challenges, we propose a two-stream Mamba pyramid network that aligns with actual judging criteria to predict TES and PCS by separating visual-feature based TES evaluation stream from audio-visual-feature based PCS evaluation stream. In the PCS evaluation stream, we introduce a multi-level fusion mechanism to guarantee that video-based features remain unaffected when assessing TES, and enhance PCS estimation by fusing visual and auditory cues across each contextual level of the pyramid. In the TES evaluation stream, the multi-scale Mamba pyramid and TES head we proposed effectively address the challenges of localizing and evaluating action elements with various temporal scales and give score predictions. With Mamba's superior ability to capture long-range dependencies and its linear computational complexity, our method is ideal for handling lengthy figure skating videos. Comprehensive experimentation demonstrates that our framework attains state-of-the-art performance on the FineFS benchmark. Our source code is available at https://github.com/ycwfs/Figure-Skating-Action-Quality-Assessment.
>
---
#### [new 004] Expandable Residual Approximation for Knowledge Distillation
- **分类: cs.CV**

- **简介: 论文提出Expandable Residual Approximation（ERA）方法，用于知识蒸馏任务，解决教师模型与学生模型间能力差距导致的知识迁移不足问题。通过多分支残差网络分解知识并引入教师权重整合策略，提升学生模型性能，在图像分类和目标检测任务上显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.16050v1](http://arxiv.org/pdf/2508.16050v1)**

> **作者:** Zhaoyi Yan; Binghui Chen; Yunfan Liu; Qixiang Ye
>
> **备注:** TNNLS 2025
>
> **摘要:** Knowledge distillation (KD) aims to transfer knowledge from a large-scale teacher model to a lightweight one, significantly reducing computational and storage requirements. However, the inherent learning capacity gap between the teacher and student often hinders the sufficient transfer of knowledge, motivating numerous studies to address this challenge. Inspired by the progressive approximation principle in the Stone-Weierstrass theorem, we propose Expandable Residual Approximation (ERA), a novel KD method that decomposes the approximation of residual knowledge into multiple steps, reducing the difficulty of mimicking the teacher's representation through a divide-and-conquer approach. Specifically, ERA employs a Multi-Branched Residual Network (MBRNet) to implement this residual knowledge decomposition. Additionally, a Teacher Weight Integration (TWI) strategy is introduced to mitigate the capacity disparity by reusing the teacher's head weights. Extensive experiments show that ERA improves the Top-1 accuracy on the ImageNet classification benchmark by 1.41% and the AP on the MS COCO object detection benchmark by 1.40, as well as achieving leading performance across computer vision tasks. Codes and models are available at https://github.com/Zhaoyi-Yan/ERA.
>
---
#### [new 005] Boosting Pathology Foundation Models via Few-shot Prompt-tuning for Rare Cancer Subtyping
- **分类: cs.CV**

- **简介: 该论文针对罕见癌症亚型分类任务，解决病理AI模型在稀有癌症上性能不足的问题。提出PathPT框架，通过视觉-语言模型和提示调优实现细粒度定位与跨模态推理，显著提升准确性和可解释性。**

- **链接: [http://arxiv.org/pdf/2508.15904v1](http://arxiv.org/pdf/2508.15904v1)**

> **作者:** Dexuan He; Xiao Zhou; Wenbin Guan; Liyuan Zhang; Xiaoman Zhang; Sinuo Xu; Ge Wang; Lifeng Wang; Xiaojun Yuan; Xin Sun; Yanfeng Wang; Kun Sun; Ya Zhang; Weidi Xie
>
> **摘要:** Rare cancers comprise 20-25% of all malignancies but face major diagnostic challenges due to limited expert availability-especially in pediatric oncology, where they represent over 70% of cases. While pathology vision-language (VL) foundation models show promising zero-shot capabilities for common cancer subtyping, their clinical performance for rare cancers remains limited. Existing multi-instance learning (MIL) methods rely only on visual features, overlooking cross-modal knowledge and compromising interpretability critical for rare cancer diagnosis. To address this limitation, we propose PathPT, a novel framework that fully exploits the potential of vision-language pathology foundation models through spatially-aware visual aggregation and task-specific prompt tuning. Unlike conventional MIL, PathPT converts WSI-level supervision into fine-grained tile-level guidance by leveraging the zero-shot capabilities of VL models, thereby preserving localization on cancerous regions and enabling cross-modal reasoning through prompts aligned with histopathological semantics. We benchmark PathPT on eight rare cancer datasets(four adult and four pediatric) spanning 56 subtypes and 2,910 WSIs, as well as three common cancer datasets, evaluating four state-of-the-art VL models and four MIL frameworks under three few-shot settings. Results show that PathPT consistently delivers superior performance, achieving substantial gains in subtyping accuracy and cancerous region grounding ability. This work advances AI-assisted diagnosis for rare cancers, offering a scalable solution for improving subtyping accuracy in settings with limited access to specialized expertise.
>
---
#### [new 006] Ensemble learning of foundation models for precision oncology
- **分类: cs.CV**

- **简介: 论文提出ELF框架，通过集成五个病理基础模型生成统一的切片级表示，解决现有模型性能不一、泛化能力弱的问题。适用于癌症分类、生物标志物检测及治疗反应预测等精准 oncology 任务，提升准确性和数据效率。**

- **链接: [http://arxiv.org/pdf/2508.16085v1](http://arxiv.org/pdf/2508.16085v1)**

> **作者:** Xiangde Luo; Xiyue Wang; Feyisope Eweje; Xiaoming Zhang; Sen Yang; Ryan Quinton; Jinxi Xiang; Yuchen Li; Yuanfeng Ji; Zhe Li; Yijiang Chen; Colin Bergstrom; Ted Kim; Francesca Maria Olguin; Kelley Yuan; Matthew Abikenari; Andrew Heider; Sierra Willens; Sanjeeth Rajaram; Robert West; Joel Neal; Maximilian Diehn; Ruijiang Li
>
> **备注:** A conceptual evaluation work; more studies are in progress; examples are here (https://github.com/lilab-stanford/ELF)
>
> **摘要:** Histopathology is essential for disease diagnosis and treatment decision-making. Recent advances in artificial intelligence (AI) have enabled the development of pathology foundation models that learn rich visual representations from large-scale whole-slide images (WSIs). However, existing models are often trained on disparate datasets using varying strategies, leading to inconsistent performance and limited generalizability. Here, we introduce ELF (Ensemble Learning of Foundation models), a novel framework that integrates five state-of-the-art pathology foundation models to generate unified slide-level representations. Trained on 53,699 WSIs spanning 20 anatomical sites, ELF leverages ensemble learning to capture complementary information from diverse models while maintaining high data efficiency. Unlike traditional tile-level models, ELF's slide-level architecture is particularly advantageous in clinical contexts where data are limited, such as therapeutic response prediction. We evaluated ELF across a wide range of clinical applications, including disease classification, biomarker detection, and response prediction to major anticancer therapies, cytotoxic chemotherapy, targeted therapy, and immunotherapy, across multiple cancer types. ELF consistently outperformed all constituent foundation models and existing slide-level models, demonstrating superior accuracy and robustness. Our results highlight the power of ensemble learning for pathology foundation models and suggest ELF as a scalable and generalizable solution for advancing AI-assisted precision oncology.
>
---
#### [new 007] RAGSR: Regional Attention Guided Diffusion for Image Super-Resolution
- **分类: cs.CV**

- **简介: 该论文提出RAGSR方法，用于图像超分辨率任务，解决多物体场景下细节模糊问题。通过区域注意力机制引导文本与图像信息融合，提升局部细节真实性和整体一致性。**

- **链接: [http://arxiv.org/pdf/2508.16158v1](http://arxiv.org/pdf/2508.16158v1)**

> **作者:** Haodong He; Yancheng Bai; Rui Lan; Xu Duan; Lei Sun; Xiangxiang Chu; Gui-Song Xia
>
> **摘要:** The rich textual information of large vision-language models (VLMs) combined with the powerful generative prior of pre-trained text-to-image (T2I) diffusion models has achieved impressive performance in single-image super-resolution (SISR). However, existing methods still face significant challenges in generating clear and accurate regional details, particularly in scenarios involving multiple objects. This challenge primarily stems from a lack of fine-grained regional descriptions and the models' insufficient ability to capture complex prompts. To address these limitations, we propose a Regional Attention Guided Super-Resolution (RAGSR) method that explicitly extracts localized fine-grained information and effectively encodes it through a novel regional attention mechanism, enabling both enhanced detail and overall visually coherent SR results. Specifically, RAGSR localizes object regions in an image and assigns fine-grained caption to each region, which are formatted as region-text pairs as textual priors for T2I models. A regional guided attention is then leveraged to ensure that each region-text pair is properly considered in the attention process while preventing unwanted interactions between unrelated region-text pairs. By leveraging this attention mechanism, our approach offers finer control over the integration of text and image information, thereby effectively overcoming limitations faced by traditional SISR techniques. Experimental results on benchmark datasets demonstrate that our approach exhibits superior performance in generating perceptually authentic visual details while maintaining contextual consistency compared to existing approaches.
>
---
#### [new 008] Seeing Clearly, Forgetting Deeply: Revisiting Fine-Tuned Video Generators for Driving Simulation
- **分类: cs.CV**

- **简介: 论文研究视频生成模型在自动驾驶仿真中的应用，发现微调会提升视觉质量但损害动态元素空间准确性。作者提出通过持续学习策略平衡二者，解决视觉与动态理解目标冲突问题。**

- **链接: [http://arxiv.org/pdf/2508.16512v1](http://arxiv.org/pdf/2508.16512v1)**

> **作者:** Chun-Peng Chang; Chen-Yu Wang; Julian Schmidt; Holger Caesar; Alain Pagani
>
> **摘要:** Recent advancements in video generation have substantially improved visual quality and temporal coherence, making these models increasingly appealing for applications such as autonomous driving, particularly in the context of driving simulation and so-called "world models". In this work, we investigate the effects of existing fine-tuning video generation approaches on structured driving datasets and uncover a potential trade-off: although visual fidelity improves, spatial accuracy in modeling dynamic elements may degrade. We attribute this degradation to a shift in the alignment between visual quality and dynamic understanding objectives. In datasets with diverse scene structures within temporal space, where objects or perspective shift in varied ways, these objectives tend to highly correlated. However, the very regular and repetitive nature of driving scenes allows visual quality to improve by modeling dominant scene motion patterns, without necessarily preserving fine-grained dynamic behavior. As a result, fine-tuning encourages the model to prioritize surface-level realism over dynamic accuracy. To further examine this phenomenon, we show that simple continual learning strategies, such as replay from diverse domains, can offer a balanced alternative by preserving spatial accuracy while maintaining strong visual quality.
>
---
#### [new 009] Vision encoders should be image size agnostic and task driven
- **分类: cs.CV**

- **简介: 论文提出视觉编码器应具备图像尺寸无关性和任务驱动性，以提升效率。针对现有模型计算复杂度固定的问题，作者提出动态调整策略，并通过图像分类任务验证了方法的可行性与潜力。**

- **链接: [http://arxiv.org/pdf/2508.16317v1](http://arxiv.org/pdf/2508.16317v1)**

> **作者:** Nedyalko Prisadnikov; Danda Pani Paudel; Yuqian Fu; Luc Van Gool
>
> **摘要:** This position paper argues that the next generation of vision encoders should be image size agnostic and task driven. The source of our inspiration is biological. Not a structural aspect of biological vision, but a behavioral trait -- efficiency. We focus on a couple of ways in which vision in nature is efficient, but modern vision encoders not. We -- humans and animals -- deal with vast quantities of visual data, and need to be smart where we focus our limited energy -- it depends on the task. It is our belief that vision encoders should be dynamic and the computational complexity should depend on the task at hand rather than the size of the image. We, also, provide concrete first steps towards our vision -- a proof-of-concept solution for image classification. Despite classification being not very representative for what we are trying to achieve, it shows that our approach is feasible and promising.
>
---
#### [new 010] Automated Multi-label Classification of Eleven Retinal Diseases: A Benchmark of Modern Architectures and a Meta-Ensemble on a Large Synthetic Dataset
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对 retinal disease 多标签分类任务，解决临床数据稀缺问题。作者在 SynFundus-1M 合成数据上训练六种模型并构建元集成模型，实现高精度分类，并验证其在真实数据上的良好泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.15986v1](http://arxiv.org/pdf/2508.15986v1)**

> **作者:** Jerry Cao-Xue; Tien Comlekoglu; Keyi Xue; Guanliang Wang; Jiang Li; Gordon Laurie
>
> **备注:** 25 pages, 6 figures, 8 tables
>
> **摘要:** The development of multi-label deep learning models for retinal disease classification is often hindered by the scarcity of large, expertly annotated clinical datasets due to patient privacy concerns and high costs. The recent release of SynFundus-1M, a high-fidelity synthetic dataset with over one million fundus images, presents a novel opportunity to overcome these barriers. To establish a foundational performance benchmark for this new resource, we developed an end-to-end deep learning pipeline, training six modern architectures (ConvNeXtV2, SwinV2, ViT, ResNet, EfficientNetV2, and the RETFound foundation model) to classify eleven retinal diseases using a 5-fold multi-label stratified cross-validation strategy. We further developed a meta-ensemble model by stacking the out-of-fold predictions with an XGBoost classifier. Our final ensemble model achieved the highest performance on the internal validation set, with a macro-average Area Under the Receiver Operating Characteristic Curve (AUC) of 0.9973. Critically, the models demonstrated strong generalization to three diverse, real-world clinical datasets, achieving an AUC of 0.7972 on a combined DR dataset, an AUC of 0.9126 on the AIROGS glaucoma dataset and a macro-AUC of 0.8800 on the multi-label RFMiD dataset. This work provides a robust baseline for future research on large-scale synthetic datasets and establishes that models trained exclusively on synthetic data can accurately classify multiple pathologies and generalize effectively to real clinical images, offering a viable pathway to accelerate the development of comprehensive AI systems in ophthalmology.
>
---
#### [new 011] UniEM-3M: A Universal Electron Micrograph Dataset for Microstructural Segmentation and Generation
- **分类: cs.CV**

- **简介: 论文提出UniEM-3M，一个大规模多模态电子显微图像数据集，用于微结构分割与生成任务。解决标注数据稀缺问题，包含5091张高分辨率图像、300万实例标签及文本描述，并提供生成模型和基准测试，推动材料自动化分析发展。**

- **链接: [http://arxiv.org/pdf/2508.16239v1](http://arxiv.org/pdf/2508.16239v1)**

> **作者:** Nan wang; Zhiyi Xia; Yiming Li; Shi Tang; Zuxin Fan; Xi Fang; Haoyi Tao; Xiaochen Cai; Guolin Ke; Linfeng Zhang; Yanhui Hong
>
> **备注:** 15 pages, 13 figures, Submitted to AAAI2026
>
> **摘要:** Quantitative microstructural characterization is fundamental to materials science, where electron micrograph (EM) provides indispensable high-resolution insights. However, progress in deep learning-based EM characterization has been hampered by the scarcity of large-scale, diverse, and expert-annotated datasets, due to acquisition costs, privacy concerns, and annotation complexity. To address this issue, we introduce UniEM-3M, the first large-scale and multimodal EM dataset for instance-level understanding. It comprises 5,091 high-resolution EMs, about 3 million instance segmentation labels, and image-level attribute-disentangled textual descriptions, a subset of which will be made publicly available. Furthermore, we are also releasing a text-to-image diffusion model trained on the entire collection to serve as both a powerful data augmentation tool and a proxy for the complete data distribution. To establish a rigorous benchmark, we evaluate various representative instance segmentation methods on the complete UniEM-3M and present UniEM-Net as a strong baseline model. Quantitative experiments demonstrate that this flow-based model outperforms other advanced methods on this challenging benchmark. Our multifaceted release of a partial dataset, a generative model, and a comprehensive benchmark -- available at huggingface -- will significantly accelerate progress in automated materials analysis.
>
---
#### [new 012] A Multimodal-Multitask Framework with Cross-modal Relation and Hierarchical Interactive Attention for Semantic Comprehension
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MM-ORIENT框架，用于多任务 multimodal semantic comprehension。针对模态噪声和融合中信息丢失问题，通过跨模态关系图重构特征并引入层次交互注意力机制，提升表示质量与任务性能。**

- **链接: [http://arxiv.org/pdf/2508.16300v1](http://arxiv.org/pdf/2508.16300v1)**

> **作者:** Mohammad Zia Ur Rehman; Devraj Raghuvanshi; Umang Jain; Shubhi Bansal; Nagendra Kumar
>
> **备注:** Published in Information Fusion
>
> **摘要:** A major challenge in multimodal learning is the presence of noise within individual modalities. This noise inherently affects the resulting multimodal representations, especially when these representations are obtained through explicit interactions between different modalities. Moreover, the multimodal fusion techniques while aiming to achieve a strong joint representation, can neglect valuable discriminative information within the individual modalities. To this end, we propose a Multimodal-Multitask framework with crOss-modal Relation and hIErarchical iNteractive aTtention (MM-ORIENT) that is effective for multiple tasks. The proposed approach acquires multimodal representations cross-modally without explicit interaction between different modalities, reducing the noise effect at the latent stage. To achieve this, we propose cross-modal relation graphs that reconstruct monomodal features to acquire multimodal representations. The features are reconstructed based on the node neighborhood, where the neighborhood is decided by the features of a different modality. We also propose Hierarchical Interactive Monomadal Attention (HIMA) to focus on pertinent information within a modality. While cross-modal relation graphs help comprehend high-order relationships between two modalities, HIMA helps in multitasking by learning discriminative features of individual modalities before late-fusing them. Finally, extensive experimental evaluation on three datasets demonstrates that the proposed approach effectively comprehends multimodal content for multiple tasks.
>
---
#### [new 013] Two-flow Feedback Multi-scale Progressive Generative Adversarial Network
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MSPG-SEN模型，用于图像生成任务，解决GAN训练不稳定、成本高及生成质量不足的问题。通过两流反馈机制、动态残差网络和注意力模块提升性能与效率。**

- **链接: [http://arxiv.org/pdf/2508.16089v1](http://arxiv.org/pdf/2508.16089v1)**

> **作者:** Sun Weikai; Song Shijie; Chi Wenjie
>
> **摘要:** Although diffusion model has made good progress in the field of image generation, GAN\cite{huang2023adaptive} still has a large development space due to its unique advantages, such as WGAN\cite{liu2021comparing}, SSGAN\cite{guibas2021adaptive} \cite{zhang2022vsa} \cite{zhou2024adapt} and so on. In this paper, we propose a novel two-flow feedback multi-scale progressive generative adversarial network (MSPG-SEN) for GAN models. This paper has four contributions: 1) : We propose a two-flow feedback multi-scale progressive Generative Adversarial network (MSPG-SEN), which not only improves image quality and human visual perception on the basis of retaining the advantages of the existing GAN model, but also simplifies the training process and reduces the training cost of GAN networks. Our experimental results show that, MSPG-SEN has achieved state-of-the-art generation results on the following five datasets,INKK The dataset is 89.7\%,AWUN The dataset is 78.3\%,IONJ The dataset is 85.5\%,POKL The dataset is 88.7\%,OPIN The dataset is 96.4\%. 2) : We propose an adaptive perception-behavioral feedback loop (APFL), which effectively improves the robustness and training stability of the model and reduces the training cost. 3) : We propose a globally connected two-flow dynamic residual network(). After ablation experiments, it can effectively improve the training efficiency and greatly improve the generalization ability, with stronger flexibility. 4) : We propose a new dynamic embedded attention mechanism (DEMA). After experiments, the attention can be extended to a variety of image processing tasks, which can effectively capture global-local information, improve feature separation capability and feature expression capabilities, and requires minimal computing resources only 88.7\% with INJK With strong cross-task capability.
>
---
#### [new 014] HOSt3R: Keypoint-free Hand-Object 3D Reconstruction from RGB images
- **分类: cs.CV; cs.AI; cs.HC; cs.LG; cs.RO**

- **简介: 论文提出HOSt3R，解决RGB图像下无关键点的手-物3D重建问题。通过无需关键点检测的单目视频运动估计与多视角重建结合，实现无需预扫描模板或相机参数的通用手-物3D形状恢复，性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.16465v1](http://arxiv.org/pdf/2508.16465v1)**

> **作者:** Anilkumar Swamy; Vincent Leroy; Philippe Weinzaepfel; Jean-Sébastien Franco; Grégory Rogez
>
> **备注:** 12 pages, 8 figures
>
> **摘要:** Hand-object 3D reconstruction has become increasingly important for applications in human-robot interaction and immersive AR/VR experiences. A common approach for object-agnostic hand-object reconstruction from RGB sequences involves a two-stage pipeline: hand-object 3D tracking followed by multi-view 3D reconstruction. However, existing methods rely on keypoint detection techniques, such as Structure from Motion (SfM) and hand-keypoint optimization, which struggle with diverse object geometries, weak textures, and mutual hand-object occlusions, limiting scalability and generalization. As a key enabler to generic and seamless, non-intrusive applicability, we propose in this work a robust, keypoint detector-free approach to estimating hand-object 3D transformations from monocular motion video/images. We further integrate this with a multi-view reconstruction pipeline to accurately recover hand-object 3D shape. Our method, named HOSt3R, is unconstrained, does not rely on pre-scanned object templates or camera intrinsics, and reaches state-of-the-art performance for the tasks of object-agnostic hand-object 3D transformation and shape estimation on the SHOWMe benchmark. We also experiment on sequences from the HO3D dataset, demonstrating generalization to unseen object categories.
>
---
#### [new 015] Advances and Trends in the 3D Reconstruction of the Shape and Motion of Animals
- **分类: cs.CV**

- **简介: 该论文属于动物三维形状与运动重建任务，旨在非侵入式地从RGB图像或视频中恢复动物的3D结构和动态行为。论文系统梳理了基于深度学习的最新方法，分类讨论其输入模态、表示方式、重建技术和训练机制，并分析优劣与挑战。**

- **链接: [http://arxiv.org/pdf/2508.16062v1](http://arxiv.org/pdf/2508.16062v1)**

> **作者:** Ziqi Li; Abderraouf Amrani; Shri Rai; Hamid Laga
>
> **摘要:** Reconstructing the 3D geometry, pose, and motion of animals is a long-standing problem, which has a wide range of applications, from biology, livestock management, and animal conservation and welfare to content creation in digital entertainment and Virtual/Augmented Reality (VR/AR). Traditionally, 3D models of real animals are obtained using 3D scanners. These, however, are intrusive, often prohibitively expensive, and difficult to deploy in the natural environment of the animals. In recent years, we have seen a significant surge in deep learning-based techniques that enable the 3D reconstruction, in a non-intrusive manner, of the shape and motion of dynamic objects just from their RGB image and/or video observations. Several papers have explored their application and extension to various types of animals. This paper surveys the latest developments in this emerging and growing field of research. It categorizes and discusses the state-of-the-art methods based on their input modalities, the way the 3D geometry and motion of animals are represented, the type of reconstruction techniques they use, and the training mechanisms they adopt. It also analyzes the performance of some key methods, discusses their strengths and limitations, and identifies current challenges and directions for future research.
>
---
#### [new 016] An Investigation of Visual Foundation Models Robustness
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文研究视觉基础模型（VFMs）的鲁棒性，旨在提升其在动态环境下的稳定性。解决现实挑战如分布偏移、噪声和对抗攻击问题，通过分析防御机制与训练方法，提出评估指标与组件优化方向。**

- **链接: [http://arxiv.org/pdf/2508.16225v1](http://arxiv.org/pdf/2508.16225v1)**

> **作者:** Sandeep Gupta; Roberto Passerone
>
> **摘要:** Visual Foundation Models (VFMs) are becoming ubiquitous in computer vision, powering systems for diverse tasks such as object detection, image classification, segmentation, pose estimation, and motion tracking. VFMs are capitalizing on seminal innovations in deep learning models, such as LeNet-5, AlexNet, ResNet, VGGNet, InceptionNet, DenseNet, YOLO, and ViT, to deliver superior performance across a range of critical computer vision applications. These include security-sensitive domains like biometric verification, autonomous vehicle perception, and medical image analysis, where robustness is essential to fostering trust between technology and the end-users. This article investigates network robustness requirements crucial in computer vision systems to adapt effectively to dynamic environments influenced by factors such as lighting, weather conditions, and sensor characteristics. We examine the prevalent empirical defenses and robust training employed to enhance vision network robustness against real-world challenges such as distributional shifts, noisy and spatially distorted inputs, and adversarial attacks. Subsequently, we provide a comprehensive analysis of the challenges associated with these defense mechanisms, including network properties and components to guide ablation studies and benchmarking metrics to evaluate network robustness.
>
---
#### [new 017] FTIO: Frequent Temporally Integrated Objects
- **分类: cs.CV**

- **简介: 该论文针对无监督视频目标分割（UVOS）任务，解决初始分割不确定性和时序不一致性问题。提出FTIO框架，通过提取高频出现对象优化选择，并设计三阶段方法修复时序错位，提升多目标分割性能。**

- **链接: [http://arxiv.org/pdf/2508.16183v1](http://arxiv.org/pdf/2508.16183v1)**

> **作者:** Mohammad Mohammadzadeh Kalati; Farhad Maleki; Ian McQuillan
>
> **备注:** An updated version (full version) of the accepted paper in ECAI 2025, 8 pages (supplementary materials are added), 5 figures, 4 tables
>
> **摘要:** Predicting and tracking objects in real-world scenarios is a critical challenge in Video Object Segmentation (VOS) tasks. Unsupervised VOS (UVOS) has the additional challenge of finding an initial segmentation of salient objects, which affects the entire process and keeps a permanent uncertainty about the object proposals. Moreover, deformation and fast motion can lead to temporal inconsistencies. To address these problems, we propose Frequent Temporally Integrated Objects (FTIO), a post-processing framework with two key components. First, we introduce a combined criterion to improve object selection, mitigating failures common in UVOS--particularly when objects are small or structurally complex--by extracting frequently appearing salient objects. Second, we present a three-stage method to correct temporal inconsistencies by integrating missing object mask regions. Experimental results demonstrate that FTIO achieves state-of-the-art performance in multi-object UVOS. Code is available at: https://github.com/MohammadMohammadzadehKalati/FTIO
>
---
#### [new 018] OmniCache: A Trajectory-Oriented Global Perspective on Training-Free Cache Reuse for Diffusion Transformer Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出OmniCache，一种无需训练的缓存重用方法，用于加速扩散Transformer模型的采样过程。针对高计算成本问题，通过全局轨迹分析优化缓存分布，动态滤除噪声，提升效率并保持生成质量。**

- **链接: [http://arxiv.org/pdf/2508.16212v1](http://arxiv.org/pdf/2508.16212v1)**

> **作者:** Huanpeng Chu; Wei Wu; Guanyu Fen; Yutao Zhang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Diffusion models have emerged as a powerful paradigm for generative tasks such as image synthesis and video generation, with Transformer architectures further enhancing performance. However, the high computational cost of diffusion Transformers-stemming from a large number of sampling steps and complex per-step computations-presents significant challenges for real-time deployment. In this paper, we introduce OmniCache, a training-free acceleration method that exploits the global redundancy inherent in the denoising process. Unlike existing methods that determine caching strategies based on inter-step similarities and tend to prioritize reusing later sampling steps, our approach originates from the sampling perspective of DIT models. We systematically analyze the model's sampling trajectories and strategically distribute cache reuse across the entire sampling process. This global perspective enables more effective utilization of cached computations throughout the diffusion trajectory, rather than concentrating reuse within limited segments of the sampling procedure.In addition, during cache reuse, we dynamically estimate the corresponding noise and filter it out to reduce its impact on the sampling direction.Extensive experiments demonstrate that our approach accelerates the sampling process while maintaining competitive generative quality, offering a promising and practical solution for efficient deployment of diffusion-based generative models.
>
---
#### [new 019] Wavelet-Enhanced PaDiM for Industrial Anomaly Detection
- **分类: cs.CV**

- **简介: 论文提出Wavelet-Enhanced PaDiM（WE-PaDiM），用于工业图像异常检测与定位。针对PaDiM随机选择特征导致信息丢失的问题，该方法在特征提取前引入离散小波变换，按频率选择子带并融合多层特征，提升检测与定位性能，且更具可解释性。**

- **链接: [http://arxiv.org/pdf/2508.16034v1](http://arxiv.org/pdf/2508.16034v1)**

> **作者:** Cory Gardner; Byungseok Min; Tae-Hyuk Ahn
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** Anomaly detection and localization in industrial images are essential for automated quality inspection. PaDiM, a prominent method, models the distribution of normal image features extracted by pre-trained Convolutional Neural Networks (CNNs) but reduces dimensionality through random channel selection, potentially discarding structured information. We propose Wavelet-Enhanced PaDiM (WE-PaDiM), which integrates Discrete Wavelet Transform (DWT) analysis with multi-layer CNN features in a structured manner. WE-PaDiM applies 2D DWT to feature maps from multiple backbone layers, selects specific frequency subbands (e.g., LL, LH, HL), spatially aligns them, and concatenates them channel-wise before modeling with PaDiM's multivariate Gaussian framework. This DWT-before-concatenation strategy provides a principled method for feature selection based on frequency content relevant to anomalies, leveraging multi-scale wavelet information as an alternative to random selection. We evaluate WE-PaDiM on the challenging MVTec AD dataset with multiple backbones (ResNet-18 and EfficientNet B0-B6). The method achieves strong performance in anomaly detection and localization, yielding average results of 99.32% Image-AUC and 92.10% Pixel-AUC across 15 categories with per-class optimized configurations. Our analysis shows that wavelet choices affect performance trade-offs: simpler wavelets (e.g., Haar) with detail subbands (HL or LH/HL/HH) often enhance localization, while approximation bands (LL) improve image-level detection. WE-PaDiM thus offers a competitive and interpretable alternative to random feature selection in PaDiM, achieving robust results suitable for industrial inspection with comparable efficiency.
>
---
#### [new 020] Panoptic Segmentation of Environmental UAV Images : Litter Beach
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像分割任务，旨在解决无人机拍摄海滩图像中因沙地复杂性导致的垃圾识别难题。作者采用实例和全景分割方法，提升模型在少量样本下的准确性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.15985v1](http://arxiv.org/pdf/2508.15985v1)**

> **作者:** Ousmane Youme; Jean Marie Dembélé; Eugene C. Ezin; Christophe Cambier
>
> **备注:** This paper has been accepted for CNRIA 2023
>
> **摘要:** Convolutional neural networks (CNN) have been used efficiently in several fields, including environmental challenges. In fact, CNN can help with the monitoring of marine litter, which has become a worldwide problem. UAVs have higher resolution and are more adaptable in local areas than satellite images, making it easier to find and count trash. Since the sand is heterogeneous, a basic CNN model encounters plenty of inferences caused by reflections of sand color, human footsteps, shadows, algae present, dunes, holes, and tire tracks. For these types of images, other CNN models, such as CNN-based segmentation methods, may be more appropriate. In this paper, we use an instance-based segmentation method and a panoptic segmentation method that show good accuracy with just a few samples. The model is more robust and less
>
---
#### [new 021] Towards Open World Detection: A Survey
- **分类: cs.CV; cs.AI; 68T45; A.1; I.2; I.4**

- **简介: 论文探讨开放世界检测（OWD），旨在统一各类视觉检测模型。解决传统检测任务局限性问题，综述从基础感知到前沿技术的发展，涵盖方法、数据集及未来融合方向。**

- **链接: [http://arxiv.org/pdf/2508.16527v1](http://arxiv.org/pdf/2508.16527v1)**

> **作者:** Andrei-Stefan Bulzan; Cosmin Cernazanu-Glavan
>
> **备注:** 30 pages
>
> **摘要:** For decades, Computer Vision has aimed at enabling machines to perceive the external world. Initial limitations led to the development of highly specialized niches. As success in each task accrued and research progressed, increasingly complex perception tasks emerged. This survey charts the convergence of these tasks and, in doing so, introduces Open World Detection (OWD), an umbrella term we propose to unify class-agnostic and generally applicable detection models in the vision domain. We start from the history of foundational vision subdomains and cover key concepts, methodologies and datasets making up today's state-of-the-art landscape. This traverses topics starting from early saliency detection, foreground/background separation, out of distribution detection and leading up to open world object detection, zero-shot detection and Vision Large Language Models (VLLMs). We explore the overlap between these subdomains, their increasing convergence, and their potential to unify into a singular domain in the future, perception.
>
---
#### [new 022] Contributions to Label-Efficient Learning in Computer Vision and Remote Sensing
- **分类: cs.CV**

- **简介: 论文聚焦标签高效学习任务，解决计算机视觉与遥感中标注数据稀缺问题。提出四类方法：基于异常感知表示的弱监督学习、多任务联合训练、多模态对比学习及层次少样本学习，提升模型在有限标注下的性能。**

- **链接: [http://arxiv.org/pdf/2508.15973v1](http://arxiv.org/pdf/2508.15973v1)**

> **作者:** Minh-Tan Pham
>
> **备注:** Habilitation \`a Diriger des Recherches (HDR) manuscript
>
> **摘要:** This manuscript presents a series of my selected contributions to the topic of label-efficient learning in computer vision and remote sensing. The central focus of this research is to develop and adapt methods that can learn effectively from limited or partially annotated data, and can leverage abundant unlabeled data in real-world applications. The contributions span both methodological developments and domain-specific adaptations, in particular addressing challenges unique to Earth observation data such as multi-modality, spatial resolution variability, and scene heterogeneity. The manuscript is organized around four main axes including (1) weakly supervised learning for object discovery and detection based on anomaly-aware representations learned from large amounts of background images; (2) multi-task learning that jointly trains on multiple datasets with disjoint annotations to improve performance on object detection and semantic segmentation; (3) self-supervised and supervised contrastive learning with multimodal data to enhance scene classification in remote sensing; and (4) few-shot learning for hierarchical scene classification using both explicit and implicit modeling of class hierarchies. These contributions are supported by extensive experimental results across natural and remote sensing datasets, reflecting the outcomes of several collaborative research projects. The manuscript concludes by outlining ongoing and future research directions focused on scaling and enhancing label-efficient learning for real-world applications.
>
---
#### [new 023] Through the Looking Glass: A Dual Perspective on Weakly-Supervised Few-Shot Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 论文提出TLG模型，解决弱监督少样本语义分割中的语义同质化问题。通过双视角网络设计，引入异构视觉聚合与迁移模块，提升模型泛化能力，在参数更少情况下显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.16159v1](http://arxiv.org/pdf/2508.16159v1)**

> **作者:** Jiaqi Ma; Guo-Sen Xie; Fang Zhao; Zechao Li
>
> **摘要:** Meta-learning aims to uniformly sample homogeneous support-query pairs, characterized by the same categories and similar attributes, and extract useful inductive biases through identical network architectures. However, this identical network design results in over-semantic homogenization. To address this, we propose a novel homologous but heterogeneous network. By treating support-query pairs as dual perspectives, we introduce heterogeneous visual aggregation (HA) modules to enhance complementarity while preserving semantic commonality. To further reduce semantic noise and amplify the uniqueness of heterogeneous semantics, we design a heterogeneous transfer (HT) module. Finally, we propose heterogeneous CLIP (HC) textual information to enhance the generalization capability of multimodal models. In the weakly-supervised few-shot semantic segmentation (WFSS) task, with only 1/24 of the parameters of existing state-of-the-art models, TLG achieves a 13.2\% improvement on Pascal-5\textsuperscript{i} and a 9.7\% improvement on COCO-20\textsuperscript{i}. To the best of our knowledge, TLG is also the first weakly supervised (image-level) model that outperforms fully supervised (pixel-level) models under the same backbone architectures. The code is available at https://github.com/jarch-ma/TLG.
>
---
#### [new 024] CoVeRaP: Cooperative Vehicular Perception through mmWave FMCW Radars
- **分类: cs.CV; cs.AI; cs.LG; cs.NI**

- **简介: 论文提出CoVeRaP框架，解决多车协同雷达感知中点云稀疏噪声问题。通过21k帧时序对齐数据集和融合空间、多普勒、强度信息的网络，提升3D目标检测精度，实现更鲁棒的自动驾驶感知。**

- **链接: [http://arxiv.org/pdf/2508.16030v1](http://arxiv.org/pdf/2508.16030v1)**

> **作者:** Jinyue Song; Hansol Ku; Jayneel Vora; Nelson Lee; Ahmad Kamari; Prasant Mohapatra; Parth Pathak
>
> **备注:** Accepted at ICCCN 2025 (IEEE International Conference on Computer Communications and Networks), Tokyo, Japan, August 2025
>
> **摘要:** Automotive FMCW radars remain reliable in rain and glare, yet their sparse, noisy point clouds constrain 3-D object detection. We therefore release CoVeRaP, a 21 k-frame cooperative dataset that time-aligns radar, camera, and GPS streams from multiple vehicles across diverse manoeuvres. Built on this data, we propose a unified cooperative-perception framework with middle- and late-fusion options. Its baseline network employs a multi-branch PointNet-style encoder enhanced with self-attention to fuse spatial, Doppler, and intensity cues into a common latent space, which a decoder converts into 3-D bounding boxes and per-point depth confidence. Experiments show that middle fusion with intensity encoding boosts mean Average Precision by up to 9x at IoU 0.9 and consistently outperforms single-vehicle baselines. CoVeRaP thus establishes the first reproducible benchmark for multi-vehicle FMCW-radar perception and demonstrates that affordable radar sharing markedly improves detection robustness. Dataset and code are publicly available to encourage further research.
>
---
#### [new 025] Domain Adaptation via Feature Refinement
- **分类: cs.CV; cs.LG**

- **简介: 论文提出DAFR2框架，解决无监督域适应中的分布偏移问题。通过批量归一化统计适配、特征蒸馏和假设迁移，提升特征鲁棒性与跨域泛化能力，无需目标标签或复杂结构。**

- **链接: [http://arxiv.org/pdf/2508.16124v1](http://arxiv.org/pdf/2508.16124v1)**

> **作者:** Savvas Karatsiolis; Andreas Kamilaris
>
> **摘要:** We propose Domain Adaptation via Feature Refinement (DAFR2), a simple yet effective framework for unsupervised domain adaptation under distribution shift. The proposed method synergistically combines three key components: adaptation of Batch Normalization statistics using unlabeled target data, feature distillation from a source-trained model and hypothesis transfer. By aligning feature distributions at the statistical and representational levels, DAFR2 produces robust and domain-invariant feature spaces that generalize across similar domains without requiring target labels, complex architectures or sophisticated training objectives. Extensive experiments on benchmark datasets, including CIFAR10-C, CIFAR100-C, MNIST-C and PatchCamelyon-C, demonstrate that the proposed algorithm outperforms prior methods in robustness to corruption. Theoretical and empirical analyses further reveal that our method achieves improved feature alignment, increased mutual information between the domains and reduced sensitivity to input perturbations.
>
---
#### [new 026] SAMFusion: Sensor-Adaptive Multimodal Fusion for 3D Object Detection in Adverse Weather
- **分类: cs.CV**

- **简介: 论文提出SAMFusion，用于恶劣天气下的3D目标检测任务。针对传感器在雾、雪等条件下失效问题，融合RGB、LiDAR、NIR和雷达数据，通过注意力机制与BEV特征融合提升检测可靠性，在远距离模糊场景中对脆弱行人检测AP提升17.2。**

- **链接: [http://arxiv.org/pdf/2508.16408v1](http://arxiv.org/pdf/2508.16408v1)**

> **作者:** Edoardo Palladin; Roland Dietze; Praveen Narayanan; Mario Bijelic; Felix Heide
>
> **摘要:** Multimodal sensor fusion is an essential capability for autonomous robots, enabling object detection and decision-making in the presence of failing or uncertain inputs. While recent fusion methods excel in normal environmental conditions, these approaches fail in adverse weather, e.g., heavy fog, snow, or obstructions due to soiling. We introduce a novel multi-sensor fusion approach tailored to adverse weather conditions. In addition to fusing RGB and LiDAR sensors, which are employed in recent autonomous driving literature, our sensor fusion stack is also capable of learning from NIR gated camera and radar modalities to tackle low light and inclement weather. We fuse multimodal sensor data through attentive, depth-based blending schemes, with learned refinement on the Bird's Eye View (BEV) plane to combine image and range features effectively. Our detections are predicted by a transformer decoder that weighs modalities based on distance and visibility. We demonstrate that our method improves the reliability of multimodal sensor fusion in autonomous vehicles under challenging weather conditions, bridging the gap between ideal conditions and real-world edge cases. Our approach improves average precision by 17.2 AP compared to the next best method for vulnerable pedestrians in long distances and challenging foggy scenes. Our project page is available at https://light.princeton.edu/samfusion/
>
---
#### [new 027] MV-RAG: Retrieval Augmented Multiview Diffusion
- **分类: cs.CV; cs.AI**

- **简介: 论文提出MV-RAG，用于文本到3D生成任务，解决现有方法在罕见或域外概念上一致性差的问题。通过检索2D图像并条件化多视角扩散模型，结合混合训练策略提升3D一致性和细节准确性。**

- **链接: [http://arxiv.org/pdf/2508.16577v1](http://arxiv.org/pdf/2508.16577v1)**

> **作者:** Yosef Dayani; Omer Benishu; Sagie Benaim
>
> **备注:** Project page: https://yosefdayani.github.io/MV-RAG
>
> **摘要:** Text-to-3D generation approaches have advanced significantly by leveraging pretrained 2D diffusion priors, producing high-quality and 3D-consistent outputs. However, they often fail to produce out-of-domain (OOD) or rare concepts, yielding inconsistent or inaccurate results. To this end, we propose MV-RAG, a novel text-to-3D pipeline that first retrieves relevant 2D images from a large in-the-wild 2D database and then conditions a multiview diffusion model on these images to synthesize consistent and accurate multiview outputs. Training such a retrieval-conditioned model is achieved via a novel hybrid strategy bridging structured multiview data and diverse 2D image collections. This involves training on multiview data using augmented conditioning views that simulate retrieval variance for view-specific reconstruction, alongside training on sets of retrieved real-world 2D images using a distinctive held-out view prediction objective: the model predicts the held-out view from the other views to infer 3D consistency from 2D data. To facilitate a rigorous OOD evaluation, we introduce a new collection of challenging OOD prompts. Experiments against state-of-the-art text-to-3D, image-to-3D, and personalization baselines show that our approach significantly improves 3D consistency, photorealism, and text adherence for OOD/rare concepts, while maintaining competitive performance on standard benchmarks.
>
---
#### [new 028] Representation Learning with Adaptive Superpixel Coding
- **分类: cs.CV; cs.AI**

- **简介: 论文提出自监督模型ASC，用于图像表示学习。针对传统Vision Transformer依赖固定网格划分的问题，该模型采用自适应超像素层动态调整图像分区，提升下游任务性能。**

- **链接: [http://arxiv.org/pdf/2508.15959v1](http://arxiv.org/pdf/2508.15959v1)**

> **作者:** Mahmoud Khalil; Ahmad Khalil; Alioune Ngom
>
> **摘要:** Deep learning vision models are typically tailored for specific modalities and often rely on domain-specific assumptions, such as the grid structures used by nearly all existing vision models. In this work, we propose a self-supervised model based on Transformers, which we call Adaptive Superpixel Coding (ASC). The key insight of our model is to overcome the limitations of traditional Vision Transformers, which depend on fixed-size and non-adaptive patch partitioning. Instead, ASC employs adaptive superpixel layers that dynamically adjust to the underlying image content. We analyze key properties of the approach that make it effective, and find that our method outperforms widely-used alternatives on standard image downstream task benchmarks.
>
---
#### [new 029] SpecVLM: Enhancing Speculative Decoding of Video LLMs via Verifier-Guided Token Pruning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文提出SpecVLM，针对视频大模型推理慢的问题，通过两阶段token剪枝实现无损加速。不依赖训练，利用验证器引导剪枝，最多提速2.68倍，提升视频理解模型的解码效率。**

- **链接: [http://arxiv.org/pdf/2508.16201v1](http://arxiv.org/pdf/2508.16201v1)**

> **作者:** Yicheng Ji; Jun Zhang; Heming Xia; Jinpeng Chen; Lidan Shou; Gang Chen; Huan Li
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** Video large language models (Vid-LLMs) have shown strong capabilities in understanding video content. However, their reliance on dense video token representations introduces substantial memory and computational overhead in both prefilling and decoding. To mitigate the information loss of recent video token reduction methods and accelerate the decoding stage of Vid-LLMs losslessly, we introduce SpecVLM, a training-free speculative decoding (SD) framework tailored for Vid-LLMs that incorporates staged video token pruning. Building on our novel finding that the draft model's speculation exhibits low sensitivity to video token pruning, SpecVLM prunes up to 90% of video tokens, enabling efficient speculation without sacrificing accuracy. To achieve this, it performs a two-stage pruning process: Stage I selects highly informative tokens guided by attention signals from the verifier (target model), while Stage II prunes remaining redundant ones in a spatially uniform manner. Extensive experiments on four video understanding benchmarks demonstrate the effectiveness and robustness of SpecVLM, which achieves up to 2.68$\times$ decoding speedup for LLaVA-OneVision-72B and 2.11$\times$ speedup for Qwen2.5-VL-32B.
>
---
#### [new 030] Exploiting Information Redundancy in Attention Maps for Extreme Quantization of Vision Transformers
- **分类: cs.CV; cs.AI; cs.IT; math.IT**

- **简介: 论文针对视觉Transformer模型计算复杂度高、内存占用大的问题，提出基于注意力图熵值的冗余分析方法，通过冻结低熵注意力头并量化其值，实现极端量化下的高效推理，在ImageNet-1k上验证了有效性。**

- **链接: [http://arxiv.org/pdf/2508.16311v1](http://arxiv.org/pdf/2508.16311v1)**

> **作者:** Lucas Maisonnave; Karim Haroun; Tom Pegeot
>
> **摘要:** Transformer models rely on Multi-Head Self-Attention (MHSA) mechanisms, where each attention head contributes to the final representation. However, their computational complexity and high memory demands due to MHSA hinders their deployment at the edge. In this work, we analyze and exploit information redundancy in attention maps to accelerate model inference. By quantifying the information captured by each attention head using Shannon entropy, our analysis reveals that attention heads with lower entropy, i.e., exhibiting more deterministic behavior, tend to contribute less information, motivating targeted compression strategies. Relying on these insights, we propose Entropy Attention Maps (EAM), a model that freezes the weights of low-entropy attention maps and quantizes these values to low precision to avoid redundant re-computation. Empirical validation on ImageNet-1k shows that EAM achieves similar or higher accuracy at $\leq$20\% sparsity in attention maps and competitive performance beyond this level for the DeiT and Swin Transformer models.
>
---
#### [new 031] IRSAMap:Towards Large-Scale, High-Resolution Land Cover Map Vectorization
- **分类: cs.CV**

- **简介: 论文提出IRSAMap数据集，解决高分辨率遥感土地覆盖矢量化难题。针对现有数据集类别少、规模小、缺乏空间结构的问题，构建全球覆盖、多任务适配的矢量标注数据集，支持精准对象边界与拓扑一致性建模，推动从像素级到对象级地理信息自动提取。**

- **链接: [http://arxiv.org/pdf/2508.16272v1](http://arxiv.org/pdf/2508.16272v1)**

> **作者:** Yu Meng; Ligao Deng; Zhihao Xi; Jiansheng Chen; Jingbo Chen; Anzhi Yue; Diyou Liu; Kai Li; Chenhao Wang; Kaiyu Li; Yupeng Deng; Xian Sun
>
> **摘要:** With the enhancement of remote sensing image resolution and the rapid advancement of deep learning, land cover mapping is transitioning from pixel-level segmentation to object-based vector modeling. This shift demands more from deep learning models, requiring precise object boundaries and topological consistency. However, existing datasets face three main challenges: limited class annotations, small data scale, and lack of spatial structural information. To overcome these issues, we introduce IRSAMap, the first global remote sensing dataset for large-scale, high-resolution, multi-feature land cover vector mapping. IRSAMap offers four key advantages: 1) a comprehensive vector annotation system with over 1.8 million instances of 10 typical objects (e.g., buildings, roads, rivers), ensuring semantic and spatial accuracy; 2) an intelligent annotation workflow combining manual and AI-based methods to improve efficiency and consistency; 3) global coverage across 79 regions in six continents, totaling over 1,000 km; and 4) multi-task adaptability for tasks like pixel-level classification, building outline extraction, road centerline extraction, and panoramic segmentation. IRSAMap provides a standardized benchmark for the shift from pixel-based to object-based approaches, advancing geographic feature automation and collaborative modeling. It is valuable for global geographic information updates and digital twin construction. The dataset is publicly available at https://github.com/ucas-dlg/IRSAMap
>
---
#### [new 032] Interpreting the linear structure of vision-language model embedding spaces
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 论文研究视觉语言模型（VLM）嵌入空间的线性结构，旨在揭示图像与文本如何在共享空间中组织及编码意义。通过训练稀疏自编码器（SAEs），发现概念方向虽具模态特异性，却主要编码跨模态语义，并提出“桥接分数”量化概念间协同机制，揭示了VLM中稀疏线性结构与跨模态整合的关系。**

- **链接: [http://arxiv.org/pdf/2504.11695v4](http://arxiv.org/pdf/2504.11695v4)**

> **作者:** Isabel Papadimitriou; Huangyuan Su; Thomas Fel; Sham Kakade; Stephanie Gil
>
> **备注:** COLM 2025
>
> **摘要:** Vision-language models encode images and text in a joint space, minimizing the distance between corresponding image and text pairs. How are language and images organized in this joint space, and how do the models encode meaning and modality? To investigate this, we train and release sparse autoencoders (SAEs) on the embedding spaces of four vision-language models (CLIP, SigLIP, SigLIP2, and AIMv2). SAEs approximate model embeddings as sparse linear combinations of learned directions, or "concepts". We find that, compared to other methods of linear feature learning, SAEs are better at reconstructing the real embeddings, while also able to retain the most sparsity. Retraining SAEs with different seeds or different data diet leads to two findings: the rare, specific concepts captured by the SAEs are liable to change drastically, but we also show that commonly-activating concepts are remarkably stable across runs. Interestingly, while most concepts activate primarily for one modality, we find they are not merely encoding modality per se. Many are almost orthogonal to the subspace that defines modality, and the concept directions do not function as good modality classifiers, suggesting that they encode cross-modal semantics. To quantify this bridging behavior, we introduce the Bridge Score, a metric that identifies concept pairs which are both co-activated across aligned image-text inputs and geometrically aligned in the shared space. This reveals that even single-modality concepts can collaborate to support cross-modal integration. We release interactive demos of the SAEs for all models, allowing researchers to explore the organization of the concept spaces. Overall, our findings uncover a sparse linear structure within VLM embedding spaces that is shaped by modality, yet stitched together through latent bridges, offering new insight into how multimodal meaning is constructed.
>
---
#### [new 033] PromptFlare: Prompt-Generalized Defense via Cross-Attention Decoy in Diffusion-Based Inpainting
- **分类: cs.CV**

- **简介: 论文提出PromptFlare，一种针对扩散模型图像修复的防御方法，通过在交叉注意力机制中注入对抗噪声作为诱饵，干扰文本提示与图像的对齐，从而防止恶意修改。解决了文本提示驱动的图像篡改问题，显著提升防御效果并降低计算开销。**

- **链接: [http://arxiv.org/pdf/2508.16217v1](http://arxiv.org/pdf/2508.16217v1)**

> **作者:** Hohyun Na; Seunghoo Hong; Simon S. Woo
>
> **备注:** Accepted to ACM MM 2025
>
> **摘要:** The success of diffusion models has enabled effortless, high-quality image modifications that precisely align with users' intentions, thereby raising concerns about their potential misuse by malicious actors. Previous studies have attempted to mitigate such misuse through adversarial attacks. However, these approaches heavily rely on image-level inconsistencies, which pose fundamental limitations in addressing the influence of textual prompts. In this paper, we propose PromptFlare, a novel adversarial protection method designed to protect images from malicious modifications facilitated by diffusion-based inpainting models. Our approach leverages the cross-attention mechanism to exploit the intrinsic properties of prompt embeddings. Specifically, we identify and target shared token of prompts that is invariant and semantically uninformative, injecting adversarial noise to suppress the sampling process. The injected noise acts as a cross-attention decoy, diverting the model's focus away from meaningful prompt-image alignments and thereby neutralizing the effect of prompt. Extensive experiments on the EditBench dataset demonstrate that our method achieves state-of-the-art performance across various metrics while significantly reducing computational overhead and GPU memory usage. These findings highlight PromptFlare as a robust and efficient protection against unauthorized image manipulations. The code is available at https://github.com/NAHOHYUN-SKKU/PromptFlare.
>
---
#### [new 034] Beyond Human-prompting: Adaptive Prompt Tuning with Semantic Alignment for Anomaly Detection
- **分类: cs.CV; cs.AI**

- **简介: 论文提出APT框架，用于少样本异常检测任务。针对传统方法依赖人工提示和缺乏异常样本的问题，该方法通过噪声扰动生成自适应提示，并利用语义对齐防止过拟合，实现无需先验知识的高性能异常检测。**

- **链接: [http://arxiv.org/pdf/2508.16157v1](http://arxiv.org/pdf/2508.16157v1)**

> **作者:** Pi-Wei Chen; Jerry Chun-Wei Lin; Wei-Han Chen; Jia Ji; Zih-Ching Chen; Feng-Hao Yeh; Chao-Chun Chen
>
> **摘要:** Pre-trained Vision-Language Models (VLMs) have recently shown promise in detecting anomalies. However, previous approaches are fundamentally limited by their reliance on human-designed prompts and the lack of accessible anomaly samples, leading to significant gaps in context-specific anomaly understanding. In this paper, we propose \textbf{A}daptive \textbf{P}rompt \textbf{T}uning with semantic alignment for anomaly detection (APT), a groundbreaking prior knowledge-free, few-shot framework and overcomes the limitations of traditional prompt-based approaches. APT uses self-generated anomaly samples with noise perturbations to train learnable prompts that capture context-dependent anomalies in different scenarios. To prevent overfitting to synthetic noise, we propose a Self-Optimizing Meta-prompt Guiding Scheme (SMGS) that iteratively aligns the prompts with general anomaly semantics while incorporating diverse synthetic anomaly. Our system not only advances pixel-wise anomaly detection, but also achieves state-of-the-art performance on multiple benchmark datasets without requiring prior knowledge for prompt crafting, establishing a robust and versatile solution for real-world anomaly detection.
>
---
#### [new 035] Glo-VLMs: Leveraging Vision-Language Models for Fine-Grained Diseased Glomerulus Classification
- **分类: cs.CV**

- **简介: 论文提出Glo-VLMs框架，利用视觉语言模型实现少量标注数据下的细粒度肾小球分类，解决病理图像中细微形态差异导致的诊断难题。**

- **链接: [http://arxiv.org/pdf/2508.15960v1](http://arxiv.org/pdf/2508.15960v1)**

> **作者:** Zhenhao Guo; Rachit Saluja; Tianyuan Yao; Quan Liu; Yuankai Huo; Benjamin Liechty; David J. Pisapia; Kenji Ikemura; Mert R. Sabuncu; Yihe Yang; Ruining Deng
>
> **摘要:** Vision-language models (VLMs) have shown considerable potential in digital pathology, yet their effectiveness remains limited for fine-grained, disease-specific classification tasks such as distinguishing between glomerular subtypes. The subtle morphological variations among these subtypes, combined with the difficulty of aligning visual patterns with precise clinical terminology, make automated diagnosis in renal pathology particularly challenging. In this work, we explore how large pretrained VLMs can be effectively adapted to perform fine-grained glomerular classification, even in scenarios where only a small number of labeled examples are available. In this work, we introduce Glo-VLMs, a systematic framework designed to explore the adaptation of VLMs to fine-grained glomerular classification in data-constrained settings. Our approach leverages curated pathology images alongside clinical text prompts to facilitate joint image-text representation learning for nuanced renal pathology subtypes. By assessing various VLMs architectures and adaptation strategies under a few-shot learning paradigm, we explore how both the choice of method and the amount of labeled data impact model performance in clinically relevant scenarios. To ensure a fair comparison, we evaluate all models using standardized multi-class metrics, aiming to clarify the practical requirements and potential of large pretrained models for specialized clinical research applications. As a result, fine-tuning the VLMs achieved 0.7416 accuracy, 0.9045 macro-AUC, and 0.5277 F1-score with only 8 shots per class, demonstrating that even with highly limited supervision, foundation models can be effectively adapted for fine-grained medical image classification.
>
---
#### [new 036] Semantic-Aware Ship Detection with Vision-Language Integration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.15930v1](http://arxiv.org/pdf/2508.15930v1)**

> **作者:** Jiahao Li; Jiancheng Pan; Yuze Sun; Xiaomeng Huang
>
> **备注:** 5 pages
>
> **摘要:** Ship detection in remote sensing imagery is a critical task with wide-ranging applications, such as maritime activity monitoring, shipping logistics, and environmental studies. However, existing methods often struggle to capture fine-grained semantic information, limiting their effectiveness in complex scenarios. To address these challenges, we propose a novel detection framework that combines Vision-Language Models (VLMs) with a multi-scale adaptive sliding window strategy. To facilitate Semantic-Aware Ship Detection (SASD), we introduce ShipSem-VL, a specialized Vision-Language dataset designed to capture fine-grained ship attributes. We evaluate our framework through three well-defined tasks, providing a comprehensive analysis of its performance and demonstrating its effectiveness in advancing SASD from multiple perspectives.
>
---
#### [new 037] EdgeDoc: Hybrid CNN-Transformer Model for Accurate Forgery Detection and Localization in ID Documents
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.16284v1](http://arxiv.org/pdf/2508.16284v1)**

> **作者:** Anjith George; Sebastien Marcel
>
> **备注:** Idiap Research Report
>
> **摘要:** The widespread availability of tools for manipulating images and documents has made it increasingly easy to forge digital documents, posing a serious threat to Know Your Customer (KYC) processes and remote onboarding systems. Detecting such forgeries is essential to preserving the integrity and security of these services. In this work, we present EdgeDoc, a novel approach for the detection and localization of document forgeries. Our architecture combines a lightweight convolutional transformer with auxiliary noiseprint features extracted from the images, enhancing its ability to detect subtle manipulations. EdgeDoc achieved third place in the ICCV 2025 DeepID Challenge, demonstrating its competitiveness. Experimental results on the FantasyID dataset show that our method outperforms baseline approaches, highlighting its effectiveness in realworld scenarios. Project page : https://www.idiap. ch/paper/edgedoc/
>
---
#### [new 038] Automatic Retrieval of Specific Cows from Unlabeled Videos
- **分类: cs.CV; eess.IV**

- **简介: 论文提出一个无需深度学习的系统，用于从无标签视频中自动识别和检索特定奶牛。任务是奶牛个体识别与定位，解决传统方法依赖人工标注的问题。工作包括构建Cattlog、设计eidetic识别器和CowFinder模块，实现连续视频中的精准检索。**

- **链接: [http://arxiv.org/pdf/2508.15945v1](http://arxiv.org/pdf/2508.15945v1)**

> **作者:** Jiawen Lyu; Manu Ramesh; Madison Simonds; Jacquelyn P. Boerman; Amy R. Reibman
>
> **备注:** Extended abstract. Presented at the 3rd US Conference on Precision Livestock Farming (USPLF), 2025, Lincoln NE
>
> **摘要:** Few automated video systems are described in the open literature that enable hands-free cataloging and identification (ID) of cows in a dairy herd. In this work, we describe our system, composed of an AutoCattloger, which builds a Cattlog of dairy cows in a herd with a single input video clip per cow, an eidetic cow recognizer which uses no deep learning to ID cows, and a CowFinder, which IDs cows in a continuous stream of video. We demonstrate its value in finding individuals in unlabeled, unsegmented videos of cows walking unconstrained through the holding area of a milking parlor.
>
---
#### [new 039] Arbitrary-Scale 3D Gaussian Super-Resolution
- **分类: cs.CV**

- **简介: 该论文提出任意尺度的3D高斯超分辨率方法，解决传统方法无法灵活支持多尺度渲染且易产生伪影的问题。通过引入尺度感知渲染、生成先验优化和渐进式超分，实现单模型支持任意整数与非整数缩放，提升画质与效率。**

- **链接: [http://arxiv.org/pdf/2508.16467v1](http://arxiv.org/pdf/2508.16467v1)**

> **作者:** Huimin Zeng; Yue Bai; Yun Fu
>
> **摘要:** Existing 3D Gaussian Splatting (3DGS) super-resolution methods typically perform high-resolution (HR) rendering of fixed scale factors, making them impractical for resource-limited scenarios. Directly rendering arbitrary-scale HR views with vanilla 3DGS introduces aliasing artifacts due to the lack of scale-aware rendering ability, while adding a post-processing upsampler for 3DGS complicates the framework and reduces rendering efficiency. To tackle these issues, we build an integrated framework that incorporates scale-aware rendering, generative prior-guided optimization, and progressive super-resolving to enable 3D Gaussian super-resolution of arbitrary scale factors with a single 3D model. Notably, our approach supports both integer and non-integer scale rendering to provide more flexibility. Extensive experiments demonstrate the effectiveness of our model in rendering high-quality arbitrary-scale HR views (6.59 dB PSNR gain over 3DGS) with a single model. It preserves structural consistency with LR views and across different scales, while maintaining real-time rendering speed (85 FPS at 1080p).
>
---
#### [new 040] HAMSt3R: Human-Aware Multi-view Stereo 3D Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出HAMSt3R，用于从稀疏多视角图像中联合重建人与场景的3D结构。针对现有方法在人类场景下表现不佳的问题，引入改进的图像编码器和多任务网络头，实现高效、端到端的人类感知3D重建。**

- **链接: [http://arxiv.org/pdf/2508.16433v1](http://arxiv.org/pdf/2508.16433v1)**

> **作者:** Sara Rojas; Matthieu Armando; Bernard Ghamen; Philippe Weinzaepfel; Vincent Leroy; Gregory Rogez
>
> **摘要:** Recovering the 3D geometry of a scene from a sparse set of uncalibrated images is a long-standing problem in computer vision. While recent learning-based approaches such as DUSt3R and MASt3R have demonstrated impressive results by directly predicting dense scene geometry, they are primarily trained on outdoor scenes with static environments and struggle to handle human-centric scenarios. In this work, we introduce HAMSt3R, an extension of MASt3R for joint human and scene 3D reconstruction from sparse, uncalibrated multi-view images. First, we exploit DUNE, a strong image encoder obtained by distilling, among others, the encoders from MASt3R and from a state-of-the-art Human Mesh Recovery (HMR) model, multi-HMR, for a better understanding of scene geometry and human bodies. Our method then incorporates additional network heads to segment people, estimate dense correspondences via DensePose, and predict depth in human-centric environments, enabling a more comprehensive 3D reconstruction. By leveraging the outputs of our different heads, HAMSt3R produces a dense point map enriched with human semantic information in 3D. Unlike existing methods that rely on complex optimization pipelines, our approach is fully feed-forward and efficient, making it suitable for real-world applications. We evaluate our model on EgoHumans and EgoExo4D, two challenging benchmarks con taining diverse human-centric scenarios. Additionally, we validate its generalization to traditional multi-view stereo and multi-view pose regression tasks. Our results demonstrate that our method can reconstruct humans effectively while preserving strong performance in general 3D reconstruction tasks, bridging the gap between human and scene understanding in 3D vision.
>
---
#### [new 041] Enhanced Hybrid Technique for Efficient Digitization of Handwritten Marksheets
- **分类: cs.CV**

- **简介: 论文提出一种混合方法用于手写成绩单的高效数字化，解决手写风格多样和表格结构复杂带来的识别难题。结合OpenCV检测表格、PaddleOCR与改进YOLOv8模型识别文字，显著提升准确率，减少人工干预。**

- **链接: [http://arxiv.org/pdf/2508.16295v1](http://arxiv.org/pdf/2508.16295v1)**

> **作者:** Junaid Ahmed Sifat; Abir Chowdhury; Hasnat Md. Imtiaz; Md. Irtiza Hossain; Md. Imran Bin Azad
>
> **摘要:** The digitization of handwritten marksheets presents huge challenges due to the different styles of handwriting and complex table structures in such documents like marksheets. This work introduces a hybrid method that integrates OpenCV for table detection and PaddleOCR for recognizing sequential handwritten text. The image processing capabilities of OpenCV efficiently detects rows and columns which enable computationally lightweight and accurate table detection. Additionally, YOLOv8 and Modified YOLOv8 are implemented for handwritten text recognition within the detected table structures alongside PaddleOCR which further enhance the system's versatility. The proposed model achieves high accuracy on our custom dataset which is designed to represent different and diverse handwriting styles and complex table layouts. Experimental results demonstrate that YOLOv8 Modified achieves an accuracy of 92.72 percent, outperforming PaddleOCR 91.37 percent and the YOLOv8 model 88.91 percent. This efficiency reduces the necessity for manual work which makes this a practical and fast solution for digitizing academic as well as administrative documents. This research serves the field of document automation, particularly handwritten document understanding, by providing operational and reliable methods to scale, enhance, and integrate the technologies involved.
>
---
#### [new 042] A Lightweight Group Multiscale Bidirectional Interactive Network for Real-Time Steel Surface Defect Detection
- **分类: cs.CV; cs.AI**

- **简介: 论文提出GMBINet，用于实时钢材表面缺陷检测任务。针对现有方法计算复杂度高、跨尺度特征交互弱的问题，设计了组多尺度双向交互模块，实现高效多尺度特征提取与融合，在保持低参数量的同时达到实时推理速度。**

- **链接: [http://arxiv.org/pdf/2508.16397v1](http://arxiv.org/pdf/2508.16397v1)**

> **作者:** Yong Zhang; Cunjian Chen; Qiang Gao; Yi Wang; Bin Fang
>
> **摘要:** Real-time surface defect detection is critical for maintaining product quality and production efficiency in the steel manufacturing industry. Despite promising accuracy, existing deep learning methods often suffer from high computational complexity and slow inference speeds, which limit their deployment in resource-constrained industrial environments. Recent lightweight approaches adopt multibranch architectures based on depthwise separable convolution (DSConv) to capture multiscale contextual information. However, these methods often suffer from increased computational overhead and lack effective cross-scale feature interaction, limiting their ability to fully leverage multiscale representations. To address these challenges, we propose GMBINet, a lightweight framework that enhances multiscale feature extraction and interaction through novel Group Multiscale Bidirectional Interactive (GMBI) modules. The GMBI adopts a group-wise strategy for multiscale feature extraction, ensuring scale-agnostic computational complexity. It further integrates a Bidirectional Progressive Feature Interactor (BPFI) and a parameter-free Element-Wise Multiplication-Summation (EWMS) operation to enhance cross-scale interaction without introducing additional computational overhead. Experiments on SD-Saliency-900 and NRSD-MN datasets demonstrate that GMBINet delivers competitive accuracy with real-time speeds of 1048 FPS on GPU and 16.53 FPS on CPU at 512 resolution, using only 0.19 M parameters. Additional evaluations on the NEU-CLS defect classification dataset further confirm the strong generalization ability of our method, demonstrating its potential for broader industrial vision applications beyond surface defect detection. The dataset and code are publicly available at: https://github.com/zhangyongcode/GMBINet.
>
---
#### [new 043] NeuralMeshing: Complete Object Mesh Extraction from Casual Captures
- **分类: cs.CV; cs.RO**

- **简介: 论文提出NeuralMeshing系统，用于从多视角视频自动重建日常物体的完整网格模型。解决无专业扫描设备时获取高质量3D模型的问题，通过已知点定位与结构光技术融合多视频数据生成完整几何模型。**

- **链接: [http://arxiv.org/pdf/2508.16026v1](http://arxiv.org/pdf/2508.16026v1)**

> **作者:** Floris Erich; Naoya Chiba; Abdullah Mustafa; Ryo Hanai; Noriaki Ando; Yusuke Yoshiyasu; Yukiyasu Domae
>
> **摘要:** How can we extract complete geometric models of objects that we encounter in our daily life, without having access to commercial 3D scanners? In this paper we present an automated system for generating geometric models of objects from two or more videos. Our system requires the specification of one known point in at least one frame of each video, which can be automatically determined using a fiducial marker such as a checkerboard or Augmented Reality (AR) marker. The remaining frames are automatically positioned in world space by using Structure-from-Motion techniques. By using multiple videos and merging results, a complete object mesh can be generated, without having to rely on hole filling. Code for our system is available from https://github.com/FlorisE/NeuralMeshing.
>
---
#### [new 044] FlexMUSE: Multimodal Unification and Semantics Enhancement Framework with Flexible interaction for Creative Writing
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出FlexMUSE框架，解决多模态创意写作（MMCW）中模态语义不一致问题。通过T2I模块、语义对齐门控机制和交叉注意力融合，实现灵活交互与语义增强，提升文本与图像的协同创造力。**

- **链接: [http://arxiv.org/pdf/2508.16230v1](http://arxiv.org/pdf/2508.16230v1)**

> **作者:** Jiahao Chen; Zhiyong Ma; Wenbiao Du; Qingyuan Chuai
>
> **摘要:** Multi-modal creative writing (MMCW) aims to produce illustrated articles. Unlike common multi-modal generative (MMG) tasks such as storytelling or caption generation, MMCW is an entirely new and more abstract challenge where textual and visual contexts are not strictly related to each other. Existing methods for related tasks can be forcibly migrated to this track, but they require specific modality inputs or costly training, and often suffer from semantic inconsistencies between modalities. Therefore, the main challenge lies in economically performing MMCW with flexible interactive patterns, where the semantics between the modalities of the output are more aligned. In this work, we propose FlexMUSE with a T2I module to enable optional visual input. FlexMUSE promotes creativity and emphasizes the unification between modalities by proposing the modality semantic alignment gating (msaGate) to restrict the textual input. Besides, an attention-based cross-modality fusion is proposed to augment the input features for semantic enhancement. The modality semantic creative direct preference optimization (mscDPO) within FlexMUSE is designed by extending the rejected samples to facilitate the writing creativity. Moreover, to advance the MMCW, we expose a dataset called ArtMUSE which contains with around 3k calibrated text-image pairs. FlexMUSE achieves promising results, demonstrating its consistency, creativity and coherence.
>
---
#### [new 045] A Unified Voxel Diffusion Module for Point Cloud 3D Object Detection
- **分类: cs.CV**

- **简介: 论文提出Voxel Diffusion Module（VDM），用于点云3D目标检测任务，解决voxel表示中空间扩散能力弱的问题。通过稀疏3D卷积和残差连接增强特征表达，提升检测精度，在多个数据集上达到SOTA性能。**

- **链接: [http://arxiv.org/pdf/2508.16069v1](http://arxiv.org/pdf/2508.16069v1)**

> **作者:** Qifeng Liu; Dawei Zhao; Yabo Dong; Linzhi Shang; Liang Xiao; Juan Wang; Kunkong Zhao; Dongming Lu; Qi Zhu
>
> **备注:** submit to AAAI2026
>
> **摘要:** Recent advances in point cloud object detection have increasingly adopted Transformer-based and State Space Models (SSMs), demonstrating strong performance. However, voxelbased representations in these models require strict consistency in input and output dimensions due to their serialized processing, which limits the spatial diffusion capability typically offered by convolutional operations. This limitation significantly affects detection accuracy. Inspired by CNN-based object detection architectures, we propose a novel Voxel Diffusion Module (VDM) to enhance voxel-level representation and diffusion in point cloud data. VDM is composed of sparse 3D convolutions, submanifold sparse convolutions, and residual connections. To ensure computational efficiency, the output feature maps are downsampled to one-fourth of the original input resolution. VDM serves two primary functions: (1) diffusing foreground voxel features through sparse 3D convolutions to enrich spatial context, and (2) aggregating fine-grained spatial information to strengthen voxelwise feature representation. The enhanced voxel features produced by VDM can be seamlessly integrated into mainstream Transformer- or SSM-based detection models for accurate object classification and localization, highlighting the generalizability of our method. We evaluate VDM on several benchmark datasets by embedding it into both Transformerbased and SSM-based models. Experimental results show that our approach consistently improves detection accuracy over baseline models. Specifically, VDM-SSMs achieve 74.7 mAPH (L2) on Waymo, 72.9 NDS on nuScenes, 42.3 mAP on Argoverse 2, and 67.6 mAP on ONCE, setting new stateof-the-art performance across all datasets. Our code will be made publicly available.
>
---
#### [new 046] High-Precision Mixed Feature Fusion Network Using Hypergraph Computation for Cervical Abnormal Cell Detection
- **分类: cs.CV**

- **简介: 该论文属于宫颈异常细胞检测任务，旨在解决现有算法无法有效建模细胞间空间相关性及融合多特征的问题。作者提出基于超图计算的混合特征融合网络，通过多层级融合子网和跨层特征融合策略，显著提升检测性能。**

- **链接: [http://arxiv.org/pdf/2508.16140v1](http://arxiv.org/pdf/2508.16140v1)**

> **作者:** Jincheng Li; Danyang Dong; Menglin Zheng; Jingbo Zhang; Yueqin Hang; Lichi Zhang; Lili Zhao
>
> **摘要:** Automatic detection of abnormal cervical cells from Thinprep Cytologic Test (TCT) images is a critical component in the development of intelligent computer-aided diagnostic systems. However, existing algorithms typically fail to effectively model the correlations of visual features, while these spatial correlation features actually contain critical diagnostic information. Furthermore, no detection algorithm has the ability to integrate inter-correlation features of cells with intra-discriminative features of cells, lacking a fusion strategy for the end-to-end detection model. In this work, we propose a hypergraph-based cell detection network that effectively fuses different types of features, combining spatial correlation features and deep discriminative features. Specifically, we use a Multi-level Fusion Sub-network (MLF-SNet) to enhance feature extractioncapabilities. Then we introduce a Cross-level Feature Fusion Strategy with Hypergraph Computation module (CLFFS-HC), to integrate mixed features. Finally, we conducted experiments on three publicly available datasets, and the results demonstrate that our method significantly improves the performance of cervical abnormal cell detection.
>
---
#### [new 047] Diverse Signer Avatars with Manual and Non-Manual Feature Modelling for Sign Language Production
- **分类: cs.CV**

- **简介: 该论文属于手语生成任务，旨在提升手语数字人像的多样性与真实感。针对现有方法难以同时保留语言内容、视觉质量和非手动特征（如表情）的问题，提出基于潜在扩散模型和符号特征聚合模块的新方法，实现高质量、多样化的手语动画生成。**

- **链接: [http://arxiv.org/pdf/2508.15988v1](http://arxiv.org/pdf/2508.15988v1)**

> **作者:** Mohamed Ilyes Lakhal; Richard Bowden
>
> **摘要:** The diversity of sign representation is essential for Sign Language Production (SLP) as it captures variations in appearance, facial expressions, and hand movements. However, existing SLP models are often unable to capture diversity while preserving visual quality and modelling non-manual attributes such as emotions. To address this problem, we propose a novel approach that leverages Latent Diffusion Model (LDM) to synthesise photorealistic digital avatars from a generated reference image. We propose a novel sign feature aggregation module that explicitly models the non-manual features (\textit{e.g.}, the face) and the manual features (\textit{e.g.}, the hands). We show that our proposed module ensures the preservation of linguistic content while seamlessly using reference images with different ethnic backgrounds to ensure diversity. Experiments on the YouTube-SL-25 sign language dataset show that our pipeline achieves superior visual quality compared to state-of-the-art methods, with significant improvements on perceptual metrics.
>
---
#### [new 048] Investigating Different Geo Priors for Image Classification
- **分类: cs.CV**

- **简介: 论文研究如何利用地理先验信息提升图像分类性能，针对物种识别任务，评估不同空间隐式神经表示模型作为地理先验的有效性，并优化未见物种的预测策略。**

- **链接: [http://arxiv.org/pdf/2508.15946v1](http://arxiv.org/pdf/2508.15946v1)**

> **作者:** Angela Zhu; Christian Lange; Max Hamilton
>
> **备注:** Accepted and presented poster at FGVC12 (CVPR 2025 Workshop), Nashville, June 11, 2025
>
> **摘要:** Species distribution models encode spatial patterns of species occurrence making them effective priors for vision-based species classification when location information is available. In this study, we evaluate various SINR (Spatial Implicit Neural Representations) models as a geographical prior for visual classification of species from iNaturalist observations. We explore the impact of different model configurations and adjust how we handle predictions for species not included in Geo Prior training. Our analysis reveals factors that contribute to the effectiveness of these models as Geo Priors, factors that may differ from making accurate range maps.
>
---
#### [new 049] VT-LVLM-AR: A Video-Temporal Large Vision-Language Model Adapter for Fine-Grained Action Recognition in Long-Term Videos
- **分类: cs.CV**

- **简介: 该论文针对长视频中细粒度动作识别任务，解决传统模型难以捕捉长期时序依赖和语义理解不足的问题。提出VT-LVLM-AR框架，通过视觉事件映射模块压缩视频为语义连贯序列，并用提示调优适配LVLM进行分类，实现高精度与可解释性。**

- **链接: [http://arxiv.org/pdf/2508.15903v1](http://arxiv.org/pdf/2508.15903v1)**

> **作者:** Kaining Li; Shuwei He; Zihan Xu
>
> **摘要:** Human action recognition in long-term videos, characterized by complex backgrounds and subtle action differences, poses significant challenges for traditional deep learning models due to computational overhead, difficulty in capturing long-range temporal dependencies, and limited semantic understanding. While Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) have shown remarkable capabilities in multi-modal understanding and reasoning, their direct application to continuous video streams for fine-grained action recognition remains an open problem. This paper introduces VT-LVLM-AR (Video-Temporal Large Vision-Language Model Adapter for Action Recognition), a novel framework designed to bridge this gap. VT-LVLM-AR comprises a Video-to-Event Mapper (VTEM) that efficiently transforms raw video into compact, semantically rich, and temporally coherent "visual event sequences" through lightweight spatio-temporal feature extraction, adaptive temporal pooling, and conceptual quantization with an event coherence bias. These visual event sequences are then fed into an LVLM-based Action Reasoning module, specifically a frozen LLaVA-1.5 model, adapted using parameter-efficient Prompt Tuning (P-Tuning v2) for action classification. Comprehensive evaluations on the NTU RGB+D and NTU RGB+D 120 datasets demonstrate that VT-LVLM-AR consistently achieves state-of-the-art performance, surpassing existing methods (e.g., 94.1% accuracy on NTU RGB+D X-Sub). Ablation studies confirm the critical contributions of VTEM's components and the efficacy of Prompt Tuning, while human evaluations underscore the interpretability of our visual event representations. This work highlights the immense potential of leveraging LVLMs for robust and interpretable video action understanding through effective video-to-language translation and efficient model adaptation.
>
---
#### [new 050] MedOmni-45°: A Safety-Performance Benchmark for Reasoning-Oriented LLMs in Medicine
- **分类: cs.CV**

- **简介: 论文提出MedOmni-45°基准，用于评估医疗大模型在推理过程中的安全与性能权衡。针对现有评测忽略推理忠实性和顺从性风险的问题，构建包含1804题、27K输入的多提示数据集，量化Accuracy、CoT-Faithfulness和Anti-Sycophancy三指标，揭示模型普遍存在的安全-性能 trade-off。**

- **链接: [http://arxiv.org/pdf/2508.16213v1](http://arxiv.org/pdf/2508.16213v1)**

> **作者:** Kaiyuan Ji; Yijin Guo; Zicheng Zhang; Xiangyang Zhu; Yuan Tian; Ning Liu; Guangtao Zhai
>
> **备注:** 9 pages
>
> **摘要:** With the increasing use of large language models (LLMs) in medical decision-support, it is essential to evaluate not only their final answers but also the reliability of their reasoning. Two key risks are Chain-of-Thought (CoT) faithfulness -- whether reasoning aligns with responses and medical facts -- and sycophancy, where models follow misleading cues over correctness. Existing benchmarks often collapse such vulnerabilities into single accuracy scores. To address this, we introduce MedOmni-45 Degrees, a benchmark and workflow designed to quantify safety-performance trade-offs under manipulative hint conditions. It contains 1,804 reasoning-focused medical questions across six specialties and three task types, including 500 from MedMCQA. Each question is paired with seven manipulative hint types and a no-hint baseline, producing about 27K inputs. We evaluate seven LLMs spanning open- vs. closed-source, general-purpose vs. medical, and base vs. reasoning-enhanced models, totaling over 189K inferences. Three metrics -- Accuracy, CoT-Faithfulness, and Anti-Sycophancy -- are combined into a composite score visualized with a 45 Degrees plot. Results show a consistent safety-performance trade-off, with no model surpassing the diagonal. The open-source QwQ-32B performs closest (43.81 Degrees), balancing safety and accuracy but not leading in both. MedOmni-45 Degrees thus provides a focused benchmark for exposing reasoning vulnerabilities in medical LLMs and guiding safer model development.
>
---
#### [new 051] Text-Driven 3D Hand Motion Generation from Sign Language Data
- **分类: cs.CV; 65-XX; I.4.9; I.5.1**

- **简介: 论文提出HandMDM模型，解决从自然语言描述生成3D手部动作的问题。通过构建大规模文本-动作配对数据，训练扩散模型实现跨领域手部运动生成，包括不同手语和非手语动作。**

- **链接: [http://arxiv.org/pdf/2508.15902v1](http://arxiv.org/pdf/2508.15902v1)**

> **作者:** Léore Bensabath; Mathis Petrovich; Gül Varol
>
> **备注:** Project page: https://imagine.enpc.fr/~leore.bensabath/HandMDM/; 24 pages, 14 figures
>
> **摘要:** Our goal is to train a generative model of 3D hand motions, conditioned on natural language descriptions specifying motion characteristics such as handshapes, locations, finger/hand/arm movements. To this end, we automatically build pairs of 3D hand motions and their associated textual labels with unprecedented scale. Specifically, we leverage a large-scale sign language video dataset, along with noisy pseudo-annotated sign categories, which we translate into hand motion descriptions via an LLM that utilizes a dictionary of sign attributes, as well as our complementary motion-script cues. This data enables training a text-conditioned hand motion diffusion model HandMDM, that is robust across domains such as unseen sign categories from the same sign language, but also signs from another sign language and non-sign hand movements. We contribute extensive experimental investigation of these scenarios and will make our trained models and data publicly available to support future research in this relatively new field.
>
---
#### [new 052] \textsc{T-Mask}: Temporal Masking for Probing Foundation Models across Camera Views in Driver Monitoring
- **分类: cs.CV**

- **简介: 论文研究驾驶监控中跨摄像头视角的鲁棒性问题，提出T-Mask方法通过时间掩码增强视频动态区域特征，提升基础模型在未见视角下的识别准确率，尤其改善小样本活动检测。**

- **链接: [http://arxiv.org/pdf/2508.16207v1](http://arxiv.org/pdf/2508.16207v1)**

> **作者:** Thinesh Thiyakesan Ponbagavathi; Kunyu Peng; Alina Roitberg
>
> **备注:** This paper has been accepted by 26th IEEE International Conference on Intelligent Transportation Systems ITSC 2025
>
> **摘要:** Changes of camera perspective are a common obstacle in driver monitoring. While deep learning and pretrained foundation models show strong potential for improved generalization via lightweight adaptation of the final layers ('probing'), their robustness to unseen viewpoints remains underexplored. We study this challenge by adapting image foundation models to driver monitoring using a single training view, and evaluating them directly on unseen perspectives without further adaptation. We benchmark simple linear probes, advanced probing strategies, and compare two foundation models (DINOv2 and CLIP) against parameter-efficient fine-tuning (PEFT) and full fine-tuning. Building on these insights, we introduce \textsc{T-Mask} -- a new image-to-video probing method that leverages temporal token masking and emphasizes more dynamic video regions. Benchmarked on the public Drive\&Act dataset, \textsc{T-Mask} improves cross-view top-1 accuracy by $+1.23\%$ over strong probing baselines and $+8.0\%$ over PEFT methods, without adding any parameters. It proves particularly effective for underrepresented secondary activities, boosting recognition by $+5.42\%$ under the trained view and $+1.36\%$ under cross-view settings. This work provides encouraging evidence that adapting foundation models with lightweight probing methods like \textsc{T-Mask} has strong potential in fine-grained driver observation, especially in cross-view and low-data settings. These results highlight the importance of temporal token selection when leveraging foundation models to build robust driver monitoring systems. Code and models will be made available at https://github.com/th-nesh/T-MASK to support ongoing research.
>
---
#### [new 053] Robust Small Methane Plume Segmentation in Satellite Imagery
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于遥感图像中的小甲烷泄漏检测任务，旨在提升对小型甲烷排放源的识别精度。作者提出基于U-Net与ResNet34编码器的深度学习模型，结合双光谱增强技术，实现400平方米小 plume 检测，F1-score达78.39%，优于传统方法。**

- **链接: [http://arxiv.org/pdf/2508.16282v1](http://arxiv.org/pdf/2508.16282v1)**

> **作者:** Khai Duc Minh Tran; Hoa Van Nguyen; Aimuni Binti Muhammad Rawi; Hareeshrao Athinarayanarao; Ba-Ngu Vo
>
> **备注:** 6 pages, 3 figures. This paper is submitted to the International Conference on Control, Automation and Information Sciences (ICCAIS) 2025, Jeju, Korea
>
> **摘要:** This paper tackles the challenging problem of detecting methane plumes, a potent greenhouse gas, using Sentinel-2 imagery. This contributes to the mitigation of rapid climate change. We propose a novel deep learning solution based on U-Net with a ResNet34 encoder, integrating dual spectral enhancement techniques (Varon ratio and Sanchez regression) to optimise input features for heightened sensitivity. A key achievement is the ability to detect small plumes down to 400 m2 (i.e., for a single pixel at 20 m resolution), surpassing traditional methods limited to larger plumes. Experiments show our approach achieves a 78.39% F1-score on the validation set, demonstrating superior performance in sensitivity and precision over existing remote sensing techniques for automated methane monitoring, especially for small plumes.
>
---
#### [new 054] Structuring GUI Elements through Vision Language Models: Towards Action Space Generation
- **分类: cs.CV; cs.LG**

- **简介: 论文聚焦GUI元素结构化任务，解决MLLM在生成UI坐标时因语义空洞导致的精度不足问题。提出IAML训练范式，通过IoU增强的数据采样策略提升模型对坐标预测的准确性。**

- **链接: [http://arxiv.org/pdf/2508.16271v1](http://arxiv.org/pdf/2508.16271v1)**

> **作者:** Yi Xu; Yesheng Zhang; jiajia Liu; Jingdong Chen
>
> **备注:** 10pageV0
>
> **摘要:** Multimodal large language models (MLLMs) have emerged as pivotal tools in enhancing human-computer interaction. In this paper we focus on the application of MLLMs in the field of graphical user interface (GUI) elements structuring, where they assist in processing user instructions based on screen contents. Despite the promise of MLLMs, their performance in precisely generating UI element coordinates, a critical aspect of GUI understanding, is hindered by the nature of next-token prediction training. This challenge arises from the semantic void surrounding numerical UI coordinates in language representation spaces, necessitating a substantial and diverse dataset to bolster visual module capabilities. To address these limitations, we introduce an IoU-Augmented Maximum Likelihood (IAML) training paradigm. Specifically, our approach involves a novel pipeline for IoU-based coordinate sampling to augment the training data, which considers the proximity to ground truth coordinates. This data augmentation strategy is then employed to fine-tune MLLMs under the IAML paradigm, which is designed to mitigate the exposure bias problem inherent in traditional maximum likelihood estimation. Through extensive experiments, we demonstrate the superior performance of our IAML training approach over traditional training paradigms.
>
---
#### [new 055] 4D Virtual Imaging Platform for Dynamic Joint Assessment via Uni-Plane X-ray and 2D-3D Registration
- **分类: cs.CV**

- **简介: 该论文提出一种4D虚拟成像平台，用于动态关节评估。解决传统CT无法捕捉负重下关节运动的问题，通过结合双机械臂锥形束CT与2D-3D深度学习配准，实现低剂量、高精度的动态关节成像，支持术后生物力学分析与个性化诊疗。**

- **链接: [http://arxiv.org/pdf/2508.16138v1](http://arxiv.org/pdf/2508.16138v1)**

> **作者:** Hao Tang; Rongxi Yi; Lei Li; Kaiyi Cao; Jiapeng Zhao; Yihan Xiao; Minghai Shi; Peng Yuan; Yan Xi; Hui Tang; Wei Li; Zhan Wu; Yixin Zhou
>
> **摘要:** Conventional computed tomography (CT) lacks the ability to capture dynamic, weight-bearing joint motion. Functional evaluation, particularly after surgical intervention, requires four-dimensional (4D) imaging, but current methods are limited by excessive radiation exposure or incomplete spatial information from 2D techniques. We propose an integrated 4D joint analysis platform that combines: (1) a dual robotic arm cone-beam CT (CBCT) system with a programmable, gantry-free trajectory optimized for upright scanning; (2) a hybrid imaging pipeline that fuses static 3D CBCT with dynamic 2D X-rays using deep learning-based preprocessing, 3D-2D projection, and iterative optimization; and (3) a clinically validated framework for quantitative kinematic assessment. In simulation studies, the method achieved sub-voxel accuracy (0.235 mm) with a 99.18 percent success rate, outperforming conventional and state-of-the-art registration approaches. Clinical evaluation further demonstrated accurate quantification of tibial plateau motion and medial-lateral variance in post-total knee arthroplasty (TKA) patients. This 4D CBCT platform enables fast, accurate, and low-dose dynamic joint imaging, offering new opportunities for biomechanical research, precision diagnostics, and personalized orthopedic care.
>
---
#### [new 056] Attention Mechanism in Randomized Time Warping
- **分类: cs.CV**

- **简介: 论文将随机时间扭曲（RTW）解释为一种自注意力机制，揭示其在动作识别任务中可作为Transformer的替代方案。通过分析权重模式相似性与性能对比，证明RTW在保持全局关注的同时提升模型效果，在Something-Something V2数据集上比Transformer高5%准确率。**

- **链接: [http://arxiv.org/pdf/2508.16366v1](http://arxiv.org/pdf/2508.16366v1)**

> **作者:** Yutaro Hiraoka; Kazuya Okamura; Kota Suto; Kazuhiro Fukui
>
> **备注:** Accepted to IEEE ICIP 2025 Workshops
>
> **摘要:** This paper reveals that we can interpret the fundamental function of Randomized Time Warping (RTW) as a type of self-attention mechanism, a core technology of Transformers in motion recognition. The self-attention is a mechanism that enables models to identify and weigh the importance of different parts of an input sequential pattern. On the other hand, RTW is a general extension of Dynamic Time Warping (DTW), a technique commonly used for matching and comparing sequential patterns. In essence, RTW searches for optimal contribution weights for each element of the input sequential patterns to produce discriminative features. Although the two approaches look different, these contribution weights can be interpreted as self-attention weights. In fact, the two weight patterns look similar, producing a high average correlation of 0.80 across the ten smallest canonical angles. However, they work in different ways: RTW attention operates on an entire input sequential pattern, while self-attention focuses on only a local view which is a subset of the input sequential pattern because of the computational costs of the self-attention matrix. This targeting difference leads to an advantage of RTW against Transformer, as demonstrated by the 5\% performance improvement on the Something-Something V2 dataset.
>
---
#### [new 057] NeuroKoop: Neural Koopman Fusion of Structural-Functional Connectomes for Identifying Prenatal Drug Exposure in Adolescents
- **分类: q-bio.NC; cs.CV; eess.IV**

- **简介: 该论文属于脑科学与机器学习交叉任务，旨在识别青少年孕期药物暴露（PDE）状态。针对传统方法难以融合结构与功能脑网络的问题，提出NeuroKoop框架，利用神经Koopman算子融合结构-功能连接组，提升分类性能并揭示关键脑连接。**

- **链接: [http://arxiv.org/pdf/2508.16414v1](http://arxiv.org/pdf/2508.16414v1)**

> **作者:** Badhan Mazumder; Aline Kotoski; Vince D. Calhoun; Dong Hye Ye
>
> **备注:** Preprint version of the paper accepted to IEEE-EMBS International Conference on Biomedical and Health Informatics (BHI'25), 2025. This is the author's original manuscript (preprint). The final published version will appear in IEEE Xplore
>
> **摘要:** Understanding how prenatal exposure to psychoactive substances such as cannabis shapes adolescent brain organization remains a critical challenge, complicated by the complexity of multimodal neuroimaging data and the limitations of conventional analytic methods. Existing approaches often fail to fully capture the complementary features embedded within structural and functional connectomes, constraining both biological insight and predictive performance. To address this, we introduced NeuroKoop, a novel graph neural network-based framework that integrates structural and functional brain networks utilizing neural Koopman operator-driven latent space fusion. By leveraging Koopman theory, NeuroKoop unifies node embeddings derived from source-based morphometry (SBM) and functional network connectivity (FNC) based brain graphs, resulting in enhanced representation learning and more robust classification of prenatal drug exposure (PDE) status. Applied to a large adolescent cohort from the ABCD dataset, NeuroKoop outperformed relevant baselines and revealed salient structural-functional connections, advancing our understanding of the neurodevelopmental impact of PDE.
>
---
#### [new 058] RotaTouille: Rotation Equivariant Deep Learning for Contours
- **分类: cs.LG; cs.CV**

- **简介: 论文提出RotaTouille框架，用于处理轮廓数据的深度学习任务。针对旋转和平移不变性问题，利用复数卷积实现旋转与循环移位等变性，并设计等变非线性、下采样和全局池化层，获得对下游任务的不变表示。**

- **链接: [http://arxiv.org/pdf/2508.16359v1](http://arxiv.org/pdf/2508.16359v1)**

> **作者:** Odin Hoff Gardaa; Nello Blaser
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Contours or closed planar curves are common in many domains. For example, they appear as object boundaries in computer vision, isolines in meteorology, and the orbits of rotating machinery. In many cases when learning from contour data, planar rotations of the input will result in correspondingly rotated outputs. It is therefore desirable that deep learning models be rotationally equivariant. In addition, contours are typically represented as an ordered sequence of edge points, where the choice of starting point is arbitrary. It is therefore also desirable for deep learning methods to be equivariant under cyclic shifts. We present RotaTouille, a deep learning framework for learning from contour data that achieves both rotation and cyclic shift equivariance through complex-valued circular convolution. We further introduce and characterize equivariant non-linearities, coarsening layers, and global pooling layers to obtain invariant representations for downstream tasks. Finally, we demonstrate the effectiveness of RotaTouille through experiments in shape classification, reconstruction, and contour regression.
>
---
#### [new 059] Towards Diagnostic Quality Flat-Panel Detector CT Imaging Using Diffusion Models
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像增强任务，旨在提升介入室中flat-panel detector CT（FDCT）图像质量，使其接近多探测器CT（MDCT）水平。作者使用扩散模型（DDPM）去除FDCT伪影，改善解剖结构可见性，同时保持出血检测能力，从而减少患者转运需求。**

- **链接: [http://arxiv.org/pdf/2508.16252v1](http://arxiv.org/pdf/2508.16252v1)**

> **作者:** Hélène Corbaz; Anh Nguyen; Victor Schulze-Zachau; Paul Friedrich; Alicia Durrer; Florentin Bieder; Philippe C. Cattin; Marios N Psychogios
>
> **摘要:** Patients undergoing a mechanical thrombectomy procedure usually have a multi-detector CT (MDCT) scan before and after the intervention. The image quality of the flat panel detector CT (FDCT) present in the intervention room is generally much lower than that of a MDCT due to significant artifacts. However, using only FDCT images could improve patient management as the patient would not need to be moved to the MDCT room. Several studies have evaluated the potential use of FDCT imaging alone and the time that could be saved by acquiring the images before and/or after the intervention only with the FDCT. This study proposes using a denoising diffusion probabilistic model (DDPM) to improve the image quality of FDCT scans, making them comparable to MDCT scans. Clinicans evaluated FDCT, MDCT, and our model's predictions for diagnostic purposes using a questionnaire. The DDPM eliminated most artifacts and improved anatomical visibility without reducing bleeding detection, provided that the input FDCT image quality is not too low. Our code can be found on github.
>
---
#### [new 060] GUI Based Fuzzy Logic and Spatial Statistics for Unsupervised Microscopy Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出一种无需标注的显微图像细胞分割方法，结合模糊逻辑与空间统计技术，解决低对比度、光照不均等挑战。相比深度学习模型，该方法轻量、可解释且高效，在多个数据集上显著优于现有SOTA方法。**

- **链接: [http://arxiv.org/pdf/2508.15979v1](http://arxiv.org/pdf/2508.15979v1)**

> **作者:** Surajit Das; Pavel Zun
>
> **摘要:** Brightfield microscopy imaging of unstained live cells remains a persistent challenge due to low contrast, temporal changes in specimen phenotypes, irregular illumination, and the absence of training labels. While deep learning (DL) methods (e.g., Cellpose 3.0) achieve state-of-the-art (SOTA) performance, they require extensive labeled data and heavy computational resources, and they often fail under uneven illumination. We present the first unsupervised segmentation framework combining spatial standard deviation from local mean (SSDLM), fuzzy logic, adjusted variograms, Moran's I, and cumulative squared shift of nodal intensity (CSSNI) to address these limitations. Unlike deep learning models, our approach requires no annotations or retraining and operates through a user-friendly GUI tailored for non-programming users. The robustness and generality were validated on three datasets, including cross-domain data. We benchmark our method against 2023--2024 SOTA models, including Cellpose 3.0 and StarDist, using a dataset of unstained myoblast images. Our method achieves a significant improvement in segmentation performance, with an IoU increase of up to 48\% and statistically validated superiority ($p < 0.01$, Wilcoxon signed-rank test). Expert evaluation from two biologists further supports the segmentation quality (Cohen's $\kappa > 0.75$). The proposed algorithm is lightweight, interpretable, and computationally efficient, offering a practical and effective alternative for cell segmentation in label-free microscopy. The code, the dataset, and the results are available for reproducibility*.
>
---
#### [new 061] Spatial Policy: Guiding Visuomotor Robotic Manipulation with Spatial-Aware Modeling and Reasoning
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出Spatial Policy（SP），解决机器人操纵中视觉计划与动作执行间缺乏空间感知的问题。通过显式空间建模、动作预测与推理反馈机制，提升复杂环境下的控制精度与成功率。**

- **链接: [http://arxiv.org/pdf/2508.15874v1](http://arxiv.org/pdf/2508.15874v1)**

> **作者:** Yijun Liu; Yuwei Liu; Yuan Meng; Jieheng Zhang; Yuwei Zhou; Ye Li; Jiacheng Jiang; Kangye Ji; Shijia Ge; Zhi Wang; Wenwu Zhu
>
> **摘要:** Vision-centric hierarchical embodied models have demonstrated strong potential for long-horizon robotic control. However, existing methods lack spatial awareness capabilities, limiting their effectiveness in bridging visual plans to actionable control in complex environments. To address this problem, we propose Spatial Policy (SP), a unified spatial-aware visuomotor robotic manipulation framework via explicit spatial modeling and reasoning. Specifically, we first design a spatial-conditioned embodied video generation module to model spatially guided predictions through a spatial plan table. Then, we propose a spatial-based action prediction module to infer executable actions with coordination. Finally, we propose a spatial reasoning feedback policy to refine the spatial plan table via dual-stage replanning. Extensive experiments show that SP significantly outperforms state-of-the-art baselines, achieving a 33.0% average improvement over the best baseline. With an 86.7% average success rate across 11 diverse tasks, SP substantially enhances the practicality of embodied models for robotic control applications. Code and checkpoints are maintained at https://plantpotatoonmoon.github.io/SpatialPolicy/.
>
---
#### [new 062] TinyML Towards Industry 4.0: Resource-Efficient Process Monitoring of a Milling Machine
- **分类: cs.LG; cs.CV; cs.ET; cs.SY; eess.SP; eess.SY; I.2.1; I.5.4; C.5.3; C.3**

- **简介: 论文提出基于TinyML的铣床工艺监控方案，解决工业4.0中资源受限设备的实时质量监测问题。构建MillingVibes数据集，设计8-bit量化CNN模型，在ARM Cortex M4F上实现100%准确率、15.4ms推理时间，验证了其可行性。**

- **链接: [http://arxiv.org/pdf/2508.16553v1](http://arxiv.org/pdf/2508.16553v1)**

> **作者:** Tim Langer; Matthias Widra; Volkhard Beyer
>
> **备注:** 10 pages, 5 figures, 1 table
>
> **摘要:** In the context of industry 4.0, long-serving industrial machines can be retrofitted with process monitoring capabilities for future use in a smart factory. One possible approach is the deployment of wireless monitoring systems, which can benefit substantially from the TinyML paradigm. This work presents a complete TinyML flow from dataset generation, to machine learning model development, up to implementation and evaluation of a full preprocessing and classification pipeline on a microcontroller. After a short review on TinyML in industrial process monitoring, the creation of the novel MillingVibes dataset is described. The feasibility of a TinyML system for structure-integrated process quality monitoring could be shown by the development of an 8-bit-quantized convolutional neural network (CNN) model with 12.59kiB parameter storage. A test accuracy of 100.0% could be reached at 15.4ms inference time and 1.462mJ per quantized CNN inference on an ARM Cortex M4F microcontroller, serving as a reference for future TinyML process monitoring solutions.
>
---
#### [new 063] Closer to Reality: Practical Semi-Supervised Federated Learning for Foundation Model Adaptation
- **分类: cs.LG; cs.CV**

- **简介: 论文提出PSSFL框架与FedMox方法，解决边缘设备在隐私约束下难以适应基础模型的问题。针对计算资源有限和标签数据稀缺，FedMox通过稀疏专家架构和软混合策略实现高效半监督联邦学习，提升目标检测性能。**

- **链接: [http://arxiv.org/pdf/2508.16568v1](http://arxiv.org/pdf/2508.16568v1)**

> **作者:** Guangyu Sun; Jingtao Li; Weiming Zhuang; Chen Chen; Chen Chen; Lingjuan Lyu
>
> **摘要:** Foundation models (FMs) exhibit remarkable generalization but require adaptation to downstream tasks, particularly in privacy-sensitive applications. Due to data privacy regulations, cloud-based FMs cannot directly access private edge data, limiting their adaptation. Federated learning (FL) provides a privacy-aware alternative, but existing FL approaches overlook the constraints imposed by edge devices -- namely, limited computational resources and the scarcity of labeled data. To address these challenges, we introduce Practical Semi-Supervised Federated Learning (PSSFL), where edge devices hold only unlabeled, low-resolution data, while the server has limited labeled, high-resolution data. In this setting, we propose the Federated Mixture of Experts (FedMox), a novel framework that enhances FM adaptation in FL. FedMox tackles computational and resolution mismatch challenges via a sparse Mixture-of-Experts architecture, employing a spatial router to align features across resolutions and a Soft-Mixture strategy to stabilize semi-supervised learning. We take object detection as a case study, and experiments on real-world autonomous driving datasets demonstrate that FedMox effectively adapts FMs under PSSFL, significantly improving performance with constrained memory costs on edge devices. Our work paves the way for scalable and privacy-preserving FM adaptation in federated scenarios.
>
---
#### [new 064] Deep learning-enabled virtual multiplexed immunostaining of label-free tissue for vascular invasion assessment
- **分类: physics.med-ph; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分析任务，旨在解决传统免疫组化染色成本高、耗时长及组织损耗问题。作者提出基于深度学习的虚拟多重免疫荧光染色方法，仅用无标记组织的自发荧光图像即可生成ERG、PanCK和H&E虚拟染色图像，实现对甲状腺癌血管侵犯的精准定位与诊断。**

- **链接: [http://arxiv.org/pdf/2508.16209v1](http://arxiv.org/pdf/2508.16209v1)**

> **作者:** Yijie Zhang; Cagatay Isil; Xilin Yang; Yuzhu Li; Anna Elia; Karin Atlan; William Dean Wallace; Nir Pillar; Aydogan Ozcan
>
> **备注:** 29 Pages, 7 Figures
>
> **摘要:** Immunohistochemistry (IHC) has transformed clinical pathology by enabling the visualization of specific proteins within tissue sections. However, traditional IHC requires one tissue section per stain, exhibits section-to-section variability, and incurs high costs and laborious staining procedures. While multiplexed IHC (mIHC) techniques enable simultaneous staining with multiple antibodies on a single slide, they are more tedious to perform and are currently unavailable in routine pathology laboratories. Here, we present a deep learning-based virtual multiplexed immunostaining framework to simultaneously generate ERG and PanCK, in addition to H&E virtual staining, enabling accurate localization and interpretation of vascular invasion in thyroid cancers. This virtual mIHC technique is based on the autofluorescence microscopy images of label-free tissue sections, and its output images closely match the histochemical staining counterparts (ERG, PanCK and H&E) of the same tissue sections. Blind evaluation by board-certified pathologists demonstrated that virtual mIHC staining achieved high concordance with the histochemical staining results, accurately highlighting epithelial cells and endothelial cells. Virtual mIHC conducted on the same tissue section also allowed the identification and localization of small vessel invasion. This multiplexed virtual IHC approach can significantly improve diagnostic accuracy and efficiency in the histopathological evaluation of vascular invasion, potentially eliminating the need for traditional staining protocols and mitigating issues related to tissue loss and heterogeneity.
>
---
#### [new 065] UnPose: Uncertainty-Guided Diffusion Priors for Zero-Shot Pose Estimation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出UnPose框架，用于零样本6D物体姿态估计与重建任务。针对缺乏CAD模型时的挑战，利用预训练扩散模型的3D先验和不确定性估计，通过多视角融合与姿态图优化，实现高精度姿态估计和高质量3D重建。**

- **链接: [http://arxiv.org/pdf/2508.15972v1](http://arxiv.org/pdf/2508.15972v1)**

> **作者:** Zhaodong Jiang; Ashish Sinha; Tongtong Cao; Yuan Ren; Bingbing Liu; Binbin Xu
>
> **备注:** Published at the Conference on Robot Learning (CoRL) 2025. For more details please visit https://frankzhaodong.github.io/UnPose
>
> **摘要:** Estimating the 6D pose of novel objects is a fundamental yet challenging problem in robotics, often relying on access to object CAD models. However, acquiring such models can be costly and impractical. Recent approaches aim to bypass this requirement by leveraging strong priors from foundation models to reconstruct objects from single or multi-view images, but typically require additional training or produce hallucinated geometry. To this end, we propose UnPose, a novel framework for zero-shot, model-free 6D object pose estimation and reconstruction that exploits 3D priors and uncertainty estimates from a pre-trained diffusion model. Specifically, starting from a single-view RGB-D frame, UnPose uses a multi-view diffusion model to estimate an initial 3D model using 3D Gaussian Splatting (3DGS) representation, along with pixel-wise epistemic uncertainty estimates. As additional observations become available, we incrementally refine the 3DGS model by fusing new views guided by the diffusion model's uncertainty, thereby continuously improving the pose estimation accuracy and 3D reconstruction quality. To ensure global consistency, the diffusion prior-generated views and subsequent observations are further integrated in a pose graph and jointly optimized into a coherent 3DGS field. Extensive experiments demonstrate that UnPose significantly outperforms existing approaches in both 6D pose estimation accuracy and 3D reconstruction quality. We further showcase its practical applicability in real-world robotic manipulation tasks.
>
---
#### [new 066] Cross-Attention Multimodal Fusion for Breast Cancer Diagnosis: Integrating Mammography and Clinical Data with Explainability
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于乳腺癌诊断任务，旨在融合乳腺X光图像与临床数据提升分类性能。通过对比特征拼接、共注意力和交叉注意力方法，提出基于交叉注意力的多模态模型，显著提高准确率与可解释性。**

- **链接: [http://arxiv.org/pdf/2508.16000v1](http://arxiv.org/pdf/2508.16000v1)**

> **作者:** Muhaisin Tiyumba Nantogmah; Abdul-Barik Alhassan; Salamudeen Alhassan
>
> **备注:** 11 pages, 9 figures
>
> **摘要:** A precise assessment of the risk of breast lesions can greatly lower it and assist physicians in choosing the best course of action. To categorise breast lesions, the majority of current computer-aided systems only use characteristics from mammograms. Although this method is practical, it does not completely utilise clinical reports' valuable information to attain the best results. When compared to utilising mammography alone, will clinical features greatly enhance the categorisation of breast lesions? How may clinical features and mammograms be combined most effectively? In what ways may explainable AI approaches improve the interpretability and reliability of models used to diagnose breast cancer? To answer these basic problems, a comprehensive investigation is desperately needed. In order to integrate mammography and categorical clinical characteristics, this study examines a number of multimodal deep networks grounded on feature concatenation, co-attention, and cross-attention. The model achieved an AUC-ROC of 0.98, accuracy of 0.96, F1-score of 0.94, precision of 0.92, and recall of 0.95 when tested on publicly accessible datasets (TCGA and CBIS-DDSM).
>
---
#### [new 067] Harmonious Color Pairings: Insights from Human Preference and Natural Hue Statistics
- **分类: cs.HC; cs.CV; physics.soc-ph**

- **简介: 该论文研究颜色搭配的审美偏好，旨在解决色彩和谐缺乏量化共识的问题。作者通过控制色调的实验，构建偏好矩阵并提出组合指数，发现人类偏好与自然色分布一致，揭示了色彩和谐的统计规律及其生态基础。**

- **链接: [http://arxiv.org/pdf/2508.15777v1](http://arxiv.org/pdf/2508.15777v1)**

> **作者:** Ortensia Forni; Alexandre Darmon; Michael Benzaquen
>
> **备注:** 7 pages, 7 figures
>
> **摘要:** While color harmony has long been studied in art and design, a clear consensus remains elusive, as most models are grounded in qualitative insights or limited datasets. In this work, we present a quantitative, data-driven study of color pairing preferences using controlled hue-based palettes in the HSL color space. Participants evaluated combinations of thirteen distinct hues, enabling us to construct a preference matrix and define a combinability index for each color. Our results reveal that preferences are highly hue dependent, challenging the assumption of universal harmony rules proposed in the literature. Yet, when averaged over hues, statistically meaningful patterns of aesthetic preference emerge, with certain hue separations perceived as more harmonious. Strikingly, these patterns align with hue distributions found in natural landscapes, pointing to a statistical correspondence between human color preferences and the structure of color in nature. Together, these findings offer a quantitative framework for studying color harmony and its potential perceptual and ecological underpinnings.
>
---
#### [new 068] Seeing is Believing: Emotion-Aware Audio-Visual Language Modeling for Expressive Speech Generation
- **分类: cs.CL; cs.CV; cs.MM; cs.SD; eess.AS**

- **简介: 论文提出Audio-Visual Language Model（AVLM），通过融合面部视觉信息提升表达性语音生成效果，解决仅依赖语音导致的情感表达不足问题。工作包括探索视觉编码器与融合策略，并在情感识别和对话任务上实现显著性能提升。**

- **链接: [http://arxiv.org/pdf/2508.16188v1](http://arxiv.org/pdf/2508.16188v1)**

> **作者:** Weiting Tan; Jiachen Lian; Hirofumi Inaguma; Paden Tomasello; Philipp Koehn; Xutai Ma
>
> **备注:** EMNLP 2025 (Findings)
>
> **摘要:** We present an Audio-Visual Language Model (AVLM) for expressive speech generation by integrating full-face visual cues into a pre-trained expressive speech model. We explore multiple visual encoders and multimodal fusion strategies during pre-training to identify the most effective integration approach. Subsequent fine-tuning on emotion recognition and expressive dialogue tasks yields substantial gains over speech-only baselines (e.g., +5 F1 in emotion recognition). AVLM highlights the value of expressive visual information in guiding speech generation and offers a foundation for end-to-end multimodal conversational systems.
>
---
#### [new 069] A Disease-Centric Vision-Language Foundation Model for Precision Oncology in Kidney Cancer
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出RenalCLIP，一个针对肾癌的视觉语言基础模型，用于影像诊断、预后预测和报告生成。解决肾肿瘤误诊过度治疗问题，通过多中心数据训练提升准确性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.16569v1](http://arxiv.org/pdf/2508.16569v1)**

> **作者:** Yuhui Tao; Zhongwei Zhao; Zilong Wang; Xufang Luo; Feng Chen; Kang Wang; Chuanfu Wu; Xue Zhang; Shaoting Zhang; Jiaxi Yao; Xingwei Jin; Xinyang Jiang; Yifan Yang; Dongsheng Li; Lili Qiu; Zhiqiang Shao; Jianming Guo; Nengwang Yu; Shuo Wang; Ying Xiong
>
> **摘要:** The non-invasive assessment of increasingly incidentally discovered renal masses is a critical challenge in urologic oncology, where diagnostic uncertainty frequently leads to the overtreatment of benign or indolent tumors. In this study, we developed and validated RenalCLIP using a dataset of 27,866 CT scans from 8,809 patients across nine Chinese medical centers and the public TCIA cohort, a visual-language foundation model for characterization, diagnosis and prognosis of renal mass. The model was developed via a two-stage pre-training strategy that first enhances the image and text encoders with domain-specific knowledge before aligning them through a contrastive learning objective, to create robust representations for superior generalization and diagnostic precision. RenalCLIP achieved better performance and superior generalizability across 10 core tasks spanning the full clinical workflow of kidney cancer, including anatomical assessment, diagnostic classification, and survival prediction, compared with other state-of-the-art general-purpose CT foundation models. Especially, for complicated task like recurrence-free survival prediction in the TCIA cohort, RenalCLIP achieved a C-index of 0.726, representing a substantial improvement of approximately 20% over the leading baselines. Furthermore, RenalCLIP's pre-training imparted remarkable data efficiency; in the diagnostic classification task, it only needs 20% training data to achieve the peak performance of all baseline models even after they were fully fine-tuned on 100% of the data. Additionally, it achieved superior performance in report generation, image-text retrieval and zero-shot diagnosis tasks. Our findings establish that RenalCLIP provides a robust tool with the potential to enhance diagnostic accuracy, refine prognostic stratification, and personalize the management of patients with kidney cancer.
>
---
#### [new 070] Clinically-Informed Preprocessing Improves Stroke Segmentation in Low-Resource Settings
- **分类: eess.IV; cs.CV**

- **简介: 论文针对低资源环境下缺血性卒中病灶分割难题，提出基于CT图像的深度学习方法。通过引入临床启发的预处理步骤，显著提升分割精度，Dice分数提高38%，并进一步利用CTA血管分割信息再提升21%。**

- **链接: [http://arxiv.org/pdf/2508.16004v1](http://arxiv.org/pdf/2508.16004v1)**

> **作者:** Juampablo E. Heras Rivera; Hitender Oswal; Tianyi Ren; Yutong Pan; William Henry; Caitlin M. Neher; Mehmet Kurt
>
> **备注:** Accepted at MICCAI MIRASOL Workshop
>
> **摘要:** Stroke is among the top three causes of death worldwide, and accurate identification of ischemic stroke lesion boundaries from imaging is critical for diagnosis and treatment. The main imaging modalities used include magnetic resonance imaging (MRI), particularly diffusion weighted imaging (DWI), and computed tomography (CT)-based techniques such as non-contrast CT (NCCT), contrast-enhanced CT angiography (CTA), and CT perfusion (CTP). DWI is the gold standard for the identification of lesions but has limited applicability in low-resource settings due to prohibitive costs. CT-based imaging is currently the most practical imaging method in low-resource settings due to low costs and simplified logistics, but lacks the high specificity of MRI-based methods in monitoring ischemic insults. Supervised deep learning methods are the leading solution for automated ischemic stroke lesion segmentation and provide an opportunity to improve diagnostic quality in low-resource settings by incorporating insights from DWI when segmenting from CT. Here, we develop a series of models which use CT images taken upon arrival as inputs to predict follow-up lesion volumes annotated from DWI taken 2-9 days later. Furthermore, we implement clinically motivated preprocessing steps and show that the proposed pipeline results in a 38% improvement in Dice score over 10 folds compared to a nnU-Net model trained with the baseline preprocessing. Finally, we demonstrate that through additional preprocessing of CTA maps to extract vessel segmentations, we further improve our best model by 21% over 5 folds.
>
---
#### [new 071] Time-Aware One Step Diffusion Network for Real-World Image Super-Resolution
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 论文提出TADSR，用于真实图像超分辨率任务，解决固定时间步导致无法充分利用预训练扩散模型生成先验的问题。通过时间感知编码器和损失函数，实现单步高效超分并支持保真与真实感可控权衡。**

- **链接: [http://arxiv.org/pdf/2508.16557v1](http://arxiv.org/pdf/2508.16557v1)**

> **作者:** Tainyi Zhang; Zheng-Peng Duan; Peng-Tao Jiang; Bo Li; Ming-Ming Cheng; Chun-Le Guo; Chongyi Li
>
> **摘要:** Diffusion-based real-world image super-resolution (Real-ISR) methods have demonstrated impressive performance. To achieve efficient Real-ISR, many works employ Variational Score Distillation (VSD) to distill pre-trained stable-diffusion (SD) model for one-step SR with a fixed timestep. However, due to the different noise injection timesteps, the SD will perform different generative priors. Therefore, a fixed timestep is difficult for these methods to fully leverage the generative priors in SD, leading to suboptimal performance. To address this, we propose a Time-Aware one-step Diffusion Network for Real-ISR (TADSR). We first introduce a Time-Aware VAE Encoder, which projects the same image into different latent features based on timesteps. Through joint dynamic variation of timesteps and latent features, the student model can better align with the input pattern distribution of the pre-trained SD, thereby enabling more effective utilization of SD's generative capabilities. To better activate the generative prior of SD at different timesteps, we propose a Time-Aware VSD loss that bridges the timesteps of the student model and those of the teacher model, thereby producing more consistent generative prior guidance conditioned on timesteps. Additionally, though utilizing the generative prior in SD at different timesteps, our method can naturally achieve controllable trade-offs between fidelity and realism by changing the timestep condition. Experimental results demonstrate that our method achieves both state-of-the-art performance and controllable SR results with only a single step.
>
---
#### [new 072] Lightweight and Fast Real-time Image Enhancement via Decomposition of the Spatial-aware Lookup Tables
- **分类: eess.IV; cs.CV**

- **简介: 论文提出一种轻量快速的实时图像增强方法，通过分解空间感知3D查找表（LUT）来解决传统方法参数多、运行慢的问题。利用奇异值分解降低冗余，提升缓存效率，实现高效且保持空间信息的图像增强。**

- **链接: [http://arxiv.org/pdf/2508.16121v1](http://arxiv.org/pdf/2508.16121v1)**

> **作者:** Wontae Kim; Keuntek Lee; Nam Ik Cho
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** The image enhancement methods based on 3D lookup tables (3D LUTs) efficiently reduce both model size and runtime by interpolating pre-calculated values at the vertices. However, the 3D LUT methods have a limitation due to their lack of spatial information, as they convert color values on a point-by-point basis. Although spatial-aware 3D LUT methods address this limitation, they introduce additional modules that require a substantial number of parameters, leading to increased runtime as image resolution increases. To address this issue, we propose a method for generating image-adaptive LUTs by focusing on the redundant parts of the tables. Our efficient framework decomposes a 3D LUT into a linear sum of low-dimensional LUTs and employs singular value decomposition (SVD). Furthermore, we enhance the modules for spatial feature fusion to be more cache-efficient. Extensive experimental results demonstrate that our model effectively decreases both the number of parameters and runtime while maintaining spatial awareness and performance.
>
---
#### [new 073] Robust Residual Finite Scalar Quantization for Neural Compression
- **分类: eess.IV; cs.CV; eess.AS**

- **简介: 论文提出RFSQ框架，解决残差量化中信号衰减问题，提升神经压缩性能。通过可学习缩放因子和可逆层归一化策略，实现稳定高效的多阶段量化，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.15860v1](http://arxiv.org/pdf/2508.15860v1)**

> **作者:** Xiaoxu Zhu
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** Finite Scalar Quantization (FSQ) has emerged as a promising alternative to Vector Quantization (VQ) in neural compression, offering simplified training and improved stability. However, naive application of FSQ in residual quantization frameworks suffers from the \textbf{residual magnitude decay problem}, where subsequent FSQ layers receive progressively weaker signals, severely limiting their effectiveness. We propose \textbf{Robust Residual Finite Scalar Quantization (RFSQ)}, a general framework that addresses this fundamental limitation through two novel conditioning strategies: learnable scaling factors and invertible layer normalization. Our approach maintains the simplicity of FSQ while enabling effective multi-stage residual quantization. Comprehensive experiments on ImageNet demonstrate that RFSQ variants significantly outperform strong baselines including VQ-EMA, FSQ, and LFQ, achieving up to 45\% improvement in perceptual loss and 28.7\% reduction in L1 reconstruction error. The proposed LayerNorm strategy shows the most consistent improvements across different configurations, establishing RFSQ as a superior quantization method for neural compression.
>
---
#### [new 074] Modular Embedding Recomposition for Incremental Learning
- **分类: cs.AI; cs.CV**

- **简介: 论文提出MoDER方法，用于提升预训练视觉语言模型在增量学习中的零样本分类能力。通过模块化文本专家存储与重组，增强对未见类别的识别性能，解决传统方法仅保留而未提升零样本能力的问题。**

- **链接: [http://arxiv.org/pdf/2508.16463v1](http://arxiv.org/pdf/2508.16463v1)**

> **作者:** Aniello Panariello; Emanuele Frascaroli; Pietro Buzzega; Lorenzo Bonicelli; Angelo Porrello; Simone Calderara
>
> **备注:** Accepted to the 36th British Machine Vision Conference (BMVC 2025), Sheffield, UK
>
> **摘要:** The advent of pre-trained Vision-Language Models (VLMs) has significantly transformed Continual Learning (CL), mainly due to their zero-shot classification abilities. Such proficiency makes VLMs well-suited for real-world applications, enabling robust performance on novel unseen classes without requiring adaptation. However, fine-tuning remains essential when downstream tasks deviate significantly from the pre-training domain. Prior CL approaches primarily focus on preserving the zero-shot capabilities of VLMs during incremental fine-tuning on a downstream task. We take a step further by devising an approach that transforms preservation into enhancement of the zero-shot capabilities of VLMs. Our approach, named MoDular Embedding Recomposition (MoDER), introduces a modular framework that trains multiple textual experts, each specialized in a single seen class, and stores them in a foundational hub. At inference time, for each unseen class, we query the hub and compose the retrieved experts to synthesize a refined prototype that improves classification. We show the effectiveness of our method across two popular zero-shot incremental protocols, Class-IL and MTIL, comprising a total of 14 datasets. The codebase is available at https://github.com/aimagelab/mammoth.
>
---
#### [new 075] Prompting with Sign Parameters for Low-resource Sign Language Instruction Generation
- **分类: cs.HC; cs.CV**

- **简介: 该论文聚焦于低资源手语指令生成任务，旨在帮助非手语使用者学习手语。提出首个孟加拉手语指令数据集BdSLIG，并设计带手语参数的提示方法SPI，提升视觉语言模型在零样本场景下的性能。**

- **链接: [http://arxiv.org/pdf/2508.16076v1](http://arxiv.org/pdf/2508.16076v1)**

> **作者:** Md Tariquzzaman; Md Farhan Ishmam; Saiyma Sittul Muna; Md Kamrul Hasan; Hasan Mahmud
>
> **备注:** CV4A11y@ICCV 2025
>
> **摘要:** Sign Language (SL) enables two-way communication for the deaf and hard-of-hearing community, yet many sign languages remain under-resourced in the AI space. Sign Language Instruction Generation (SLIG) produces step-by-step textual instructions that enable non-SL users to imitate and learn SL gestures, promoting two-way interaction. We introduce BdSLIG, the first Bengali SLIG dataset, used to evaluate Vision Language Models (VLMs) (i) on under-resourced SLIG tasks, and (ii) on long-tail visual concepts, as Bengali SL is unlikely to appear in the VLM pre-training data. To enhance zero-shot performance, we introduce Sign Parameter-Infused (SPI) prompting, which integrates standard SL parameters, like hand shape, motion, and orientation, directly into the textual prompts. Subsuming standard sign parameters into the prompt makes the instructions more structured and reproducible than free-form natural text from vanilla prompting. We envision that our work would promote inclusivity and advancement in SL learning systems for the under-resourced communities.
>
---
#### [new 076] Disentangled Multi-modal Learning of Histology and Transcriptomics for Cancer Characterization
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于癌症特征分析任务，旨在解决病理图像与转录组数据融合中的异质性、多尺度整合不足及配对数据依赖问题。提出解耦多模态学习框架，通过子空间分解、跨倍数一致性对齐、知识蒸馏和信息令牌聚合等策略提升模型性能与效率。**

- **链接: [http://arxiv.org/pdf/2508.16479v1](http://arxiv.org/pdf/2508.16479v1)**

> **作者:** Yupei Zhang; Xiaofei Wang; Anran Liu; Lequan Yu; Chao Li
>
> **摘要:** Histopathology remains the gold standard for cancer diagnosis and prognosis. With the advent of transcriptome profiling, multi-modal learning combining transcriptomics with histology offers more comprehensive information. However, existing multi-modal approaches are challenged by intrinsic multi-modal heterogeneity, insufficient multi-scale integration, and reliance on paired data, restricting clinical applicability. To address these challenges, we propose a disentangled multi-modal framework with four contributions: 1) To mitigate multi-modal heterogeneity, we decompose WSIs and transcriptomes into tumor and microenvironment subspaces using a disentangled multi-modal fusion module, and introduce a confidence-guided gradient coordination strategy to balance subspace optimization. 2) To enhance multi-scale integration, we propose an inter-magnification gene-expression consistency strategy that aligns transcriptomic signals across WSI magnifications. 3) To reduce dependency on paired data, we propose a subspace knowledge distillation strategy enabling transcriptome-agnostic inference through a WSI-only student model. 4) To improve inference efficiency, we propose an informative token aggregation module that suppresses WSI redundancy while preserving subspace semantics. Extensive experiments on cancer diagnosis, prognosis, and survival prediction demonstrate our superiority over state-of-the-art methods across multiple settings. Code is available at https://github.com/helenypzhang/Disentangled-Multimodal-Learning.
>
---
#### [new 077] Decoding MGMT Methylation: A Step Towards Precision Medicine in Glioblastoma
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在通过MRI预测胶质母细胞瘤患者的MGMT基因甲基化状态。针对现有方法准确性不足的问题，提出CAMP框架，结合自适应稀疏惩罚的卷积自动编码器和CNN模型，显著提升预测性能。**

- **链接: [http://arxiv.org/pdf/2508.16424v1](http://arxiv.org/pdf/2508.16424v1)**

> **作者:** Hafeez Ur Rehman; Sumaiya Fazal; Moutaz Alazab; Ali Baydoun
>
> **摘要:** Glioblastomas, constituting over 50% of malignant brain tumors, are highly aggressive brain tumors that pose substantial treatment challenges due to their rapid progression and resistance to standard therapies. The methylation status of the O-6-Methylguanine-DNA Methyltransferase (MGMT) gene is a critical biomarker for predicting patient response to treatment, particularly with the alkylating agent temozolomide. However, accurately predicting MGMT methylation status using non-invasive imaging techniques remains challenging due to the complex and heterogeneous nature of glioblastomas, that includes, uneven contrast, variability within lesions, and irregular enhancement patterns. This study introduces the Convolutional Autoencoders for MGMT Methylation Status Prediction (CAMP) framework, which is based on adaptive sparse penalties to enhance predictive accuracy. The CAMP framework operates in two phases: first, generating synthetic MRI slices through a tailored autoencoder that effectively captures and preserves intricate tissue and tumor structures across different MRI modalities; second, predicting MGMT methylation status using a convolutional neural network enhanced by adaptive sparse penalties. The adaptive sparse penalty dynamically adjusts to variations in the data, such as contrast differences and tumor locations in MR images. Our method excels in MRI image synthesis, preserving brain tissue, fat, and individual tumor structures across all MRI modalities. Validated on benchmark datasets, CAMP achieved an accuracy of 0.97, specificity of 0.98, and sensitivity of 0.97, significantly outperforming existing methods. These results demonstrate the potential of the CAMP framework to improve the interpretation of MRI data and contribute to more personalized treatment strategies for glioblastoma patients.
>
---
#### [new 078] Wavelet-Space Super-Resolution for Real-Time Rendering
- **分类: cs.GR; cs.CV**

- **简介: 论文提出基于小波域特征分解的超分辨率方法，用于实时渲染中的图像增强。通过引入平稳小波变换（SWT）分离高低频信息，提升纹理细节保留与结构一致性，在保持实时性的同时显著改善感知质量。**

- **链接: [http://arxiv.org/pdf/2508.16024v1](http://arxiv.org/pdf/2508.16024v1)**

> **作者:** Prateek Poudel; Prashant Aryal; Kirtan Kunwar; Navin Nepal; Dinesh Bania Kshatri
>
> **摘要:** We investigate the use of wavelet-space feature decomposition in neural super-resolution for rendering pipelines. Building on the DFASR framework, we introduce a wavelet-domain representation that separates low- and high-frequency details before reconstruction, enabling the network to better preserve fine textures while maintaining structural consistency. Unlike RGB-space regression, our approach leverages the stationary wavelet transform (SWT) to avoid spatial down-sampling, ensuring alignment across subbands and preserving shift invariance. The model predicts wavelet coefficients conditioned on spatial G-buffers and temporally warped history frames, which are then recombined through inverse wavelet synthesis. We conduct a comprehensive ablation study across wavelet families, transform types, and architectural variants, showing that incorporating SWT improves PSNR by up to 1.5 dB and reduces LPIPS by 17% on average, at a computational overhead of roughly +24 ms compared to out DFASR baseline. While absolute runtimes on our RTX 3050 mobile GPU are higher ( 141ms) than the original DFASR report on RTX 4090( 11ms), the relative overhead remains modest, suggesting that on higher-end GPUs our method would also remain real-time capable. Taken together, our results suggest that wavelet-domain representations are a principled and effective way to enhance perceptual quality in neural upscaling for graphics applications.
>
---
#### [new 079] Self-Validated Learning for Particle Separation: A Correctness-Based Self-Training Framework Without Human Labels
- **分类: eess.IV; cs.CV**

- **简介: 论文提出自验证学习框架，用于无监督粒子实例分割任务，解决标注数据稀缺问题。通过迭代自验证机制提升分割精度，实现高准确率且无需人工标签。**

- **链接: [http://arxiv.org/pdf/2508.16224v1](http://arxiv.org/pdf/2508.16224v1)**

> **作者:** Philipp D. Lösel; Aleese Barron; Yulai Zhang; Matthias Fabian; Benjamin Young; Nicolas Francois; Andrew M. Kingston
>
> **摘要:** Non-destructive 3D imaging of large multi-particulate samples is essential for quantifying particle-level properties, such as size, shape, and spatial distribution, across applications in mining, materials science, and geology. However, accurate instance segmentation of particles in tomographic data remains challenging due to high morphological variability and frequent particle contact, which limit the effectiveness of classical methods like watershed algorithms. While supervised deep learning approaches offer improved performance, they rely on extensive annotated datasets that are labor-intensive, error-prone, and difficult to scale. In this work, we propose self-validated learning, a novel self-training framework for particle instance segmentation that eliminates the need for manual annotations. Our method leverages implicit boundary detection and iteratively refines the training set by identifying particles that can be consistently matched across reshuffled scans of the same sample. This self-validation mechanism mitigates the impact of noisy pseudo-labels, enabling robust learning from unlabeled data. After just three iterations, our approach accurately segments over 97% of the total particle volume and identifies more than 54,000 individual particles in tomographic scans of quartz fragments. Importantly, the framework also enables fully autonomous model evaluation without the need for ground truth annotations, as confirmed through comparisons with state-of-the-art instance segmentation techniques. The method is integrated into the Biomedisa image analysis platform (https://github.com/biomedisa/biomedisa/).
>
---
#### [new 080] GelSLAM: A Real-time, High-Fidelity, and Robust 3D Tactile SLAM System
- **分类: cs.RO; cs.CV**

- **简介: 论文提出GelSLAM，一种纯触觉驱动的实时3D SLAM系统，用于高精度物体位姿估计与形状重建。解决视觉方法在接触场景中易受遮挡的问题，通过触觉表面法向与曲率实现稳定跟踪和闭环，支持长时程、高保真感知。**

- **链接: [http://arxiv.org/pdf/2508.15990v1](http://arxiv.org/pdf/2508.15990v1)**

> **作者:** Hung-Jui Huang; Mohammad Amin Mirzaee; Michael Kaess; Wenzhen Yuan
>
> **备注:** 18 pages
>
> **摘要:** Accurately perceiving an object's pose and shape is essential for precise grasping and manipulation. Compared to common vision-based methods, tactile sensing offers advantages in precision and immunity to occlusion when tracking and reconstructing objects in contact. This makes it particularly valuable for in-hand and other high-precision manipulation tasks. In this work, we present GelSLAM, a real-time 3D SLAM system that relies solely on tactile sensing to estimate object pose over long periods and reconstruct object shapes with high fidelity. Unlike traditional point cloud-based approaches, GelSLAM uses tactile-derived surface normals and curvatures for robust tracking and loop closure. It can track object motion in real time with low error and minimal drift, and reconstruct shapes with submillimeter accuracy, even for low-texture objects such as wooden tools. GelSLAM extends tactile sensing beyond local contact to enable global, long-horizon spatial perception, and we believe it will serve as a foundation for many precise manipulation tasks involving interaction with objects in hand. The video demo is available on our website: https://joehjhuang.github.io/gelslam.
>
---
## 更新

#### [replaced 001] MapKD: Unlocking Prior Knowledge with Cross-Modal Distillation for Efficient Online HD Map Construction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.15653v2](http://arxiv.org/pdf/2508.15653v2)**

> **作者:** Ziyang Yan; Ruikai Li; Zhiyong Cui; Bohan Li; Han Jiang; Yilong Ren; Aoyong Li; Zhenning Li; Sijia Wen; Haiyang Yu
>
> **摘要:** Online HD map construction is a fundamental task in autonomous driving systems, aiming to acquire semantic information of map elements around the ego vehicle based on real-time sensor inputs. Recently, several approaches have achieved promising results by incorporating offline priors such as SD maps and HD maps or by fusing multi-modal data. However, these methods depend on stale offline maps and multi-modal sensor suites, resulting in avoidable computational overhead at inference. To address these limitations, we employ a knowledge distillation strategy to transfer knowledge from multimodal models with prior knowledge to an efficient, low-cost, and vision-centric student model. Specifically, we propose MapKD, a novel multi-level cross-modal knowledge distillation framework with an innovative Teacher-Coach-Student (TCS) paradigm. This framework consists of: (1) a camera-LiDAR fusion model with SD/HD map priors serving as the teacher; (2) a vision-centric coach model with prior knowledge and simulated LiDAR to bridge the cross-modal knowledge transfer gap; and (3) a lightweight vision-based student model. Additionally, we introduce two targeted knowledge distillation strategies: Token-Guided 2D Patch Distillation (TGPD) for bird's eye view feature alignment and Masked Semantic Response Distillation (MSRD) for semantic learning guidance. Extensive experiments on the challenging nuScenes dataset demonstrate that MapKD improves the student model by +6.68 mIoU and +10.94 mAP while simultaneously accelerating inference speed. The code is available at:https://github.com/2004yan/MapKD2026.
>
---
#### [replaced 002] Unlocking Robust Semantic Segmentation Performance via Label-only Elastic Deformations against Implicit Label Noise
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.10383v2](http://arxiv.org/pdf/2508.10383v2)**

> **作者:** Yechan Kim; Dongho Yoon; Younkwan Lee; Unse Fatima; Hong Kook Kim; Songjae Lee; Sanga Park; Jeong Ho Park; Seonjong Kang; Moongu Jeon
>
> **摘要:** While previous studies on image segmentation focus on handling severe (or explicit) label noise, real-world datasets also exhibit subtle (or implicit) label imperfections. These arise from inherent challenges, such as ambiguous object boundaries and annotator variability. Although not explicitly present, such mild and latent noise can still impair model performance. Typical data augmentation methods, which apply identical transformations to the image and its label, risk amplifying these subtle imperfections and limiting the model's generalization capacity. In this paper, we introduce NSegment+, a novel augmentation framework that decouples image and label transformations to address such realistic noise for semantic segmentation. By introducing controlled elastic deformations only to segmentation labels while preserving the original images, our method encourages models to focus on learning robust representations of object structures despite minor label inconsistencies. Extensive experiments demonstrate that NSegment+ consistently improves performance, achieving mIoU gains of up to +2.29, +2.38, +1.75, and +3.39 in average on Vaihingen, LoveDA, Cityscapes, and PASCAL VOC, respectively-even without bells and whistles, highlighting the importance of addressing implicit label noise. These gains can be further amplified when combined with other training tricks, including CutMix and Label Smoothing.
>
---
#### [replaced 003] The unrealized potential of agroforestry for an emissions-intensive agricultural commodity
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.20882v2](http://arxiv.org/pdf/2410.20882v2)**

> **作者:** Alexander Becker; Jan D. Wegner; Evans Dawoe; Konrad Schindler; William J. Thompson; Christian Bunn; Rachael D. Garrett; Fabio Castro-Llanos; Simon P. Hart; Wilma J. Blaser-Hart
>
> **摘要:** Reconciling agricultural production with climate-change mitigation is a formidable sustainability problem. Retaining trees in agricultural systems is one proposed solution, but the magnitude of the current and future-potential benefit that trees contribute to climate-change mitigation remains uncertain. Here, we help to resolve these issues across a West African region that produces ~60% of the world's cocoa, a crop contributing one of the highest carbon footprints of all foods. Using machine learning, we mapped shade-tree cover and carbon stocks across the region and found that existing average cover is low (~13%) and poorly aligned with climate threats. Yet, increasing shade-tree cover to a minimum of 30% could sequester an additional 307 million tonnes of CO2e, enough to offset ~167% of contemporary cocoa-related emissions in Ghana and C\^ote d'Ivoire--without reducing production. Our approach is transferable to other shade-grown crops and aligns with emerging carbon market and sustainability reporting frameworks.
>
---
#### [replaced 004] A Curious Case of Remarkable Resilience to Gradient Attacks via Fully Convolutional and Differentiable Front End with a Skip Connection
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2402.17018v2](http://arxiv.org/pdf/2402.17018v2)**

> **作者:** Leonid Boytsov; Ameya Joshi; Filipe Condessa
>
> **备注:** Accepted at TMLR (2025/08)
>
> **摘要:** We experimented with front-end enhanced neural models where a differentiable and fully convolutional model with a skip connection is added before a frozen backbone classifier. By training such composite models using a small learning rate for about one epoch, we obtained models that retained the accuracy of the backbone classifier while being unusually resistant to gradient attacks-including APGD and FAB-T attacks from the AutoAttack package-which we attribute to gradient masking. Although gradient masking is not new, the degree we observe is striking for fully differentiable models without obvious gradient-shattering-e.g., JPEG compression-or gradient-diminishing components. The training recipe to produce such models is also remarkably stable and reproducible: We applied it to three datasets (CIFAR10, CIFAR100, and ImageNet) and several modern architectures (including vision Transformers) without a single failure case. While black-box attacks such as the SQUARE attack and zero-order PGD can partially overcome gradient masking, these attacks are easily defeated by simple randomized ensembles. We estimate that these ensembles achieve near-SOTA AutoAttack accuracy on CIFAR10, CIFAR100, and ImageNet (while retaining almost all clean accuracy of the original classifiers) despite having near-zero accuracy under adaptive attacks. Adversarially training the backbone further amplifies this front-end "robustness". On CIFAR10, the respective randomized ensemble achieved 90.8$\pm 2.5\%$ (99\% CI) accuracy under the full AutoAttack while having only 18.2$\pm 3.6\%$ accuracy under the adaptive attack ($\varepsilon=8/255$, $L^\infty$ norm). We conclude the paper with a discussion of whether randomized ensembling can serve as a practical defense. Code and instructions to reproduce key results are available. https://github.com/searchivarius/curious_case_of_gradient_masking
>
---
#### [replaced 005] Alignment of Diffusion Models: Fundamentals, Challenges, and Future
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.07253v3](http://arxiv.org/pdf/2409.07253v3)**

> **作者:** Buhua Liu; Shitong Shao; Bao Li; Lichen Bai; Zhiqiang Xu; Haoyi Xiong; James Kwok; Sumi Helal; Zeke Xie
>
> **备注:** 35 pages, 5 figures, 3 tables, Paper List: github.com/xie-lab-ml/awesome-alignment-of-diffusion-models
>
> **摘要:** Diffusion models have emerged as the leading paradigm in generative modeling, excelling in various applications. Despite their success, these models often misalign with human intentions and generate results with undesired properties or even harmful content. Inspired by the success and popularity of alignment in tuning large language models, recent studies have investigated aligning diffusion models with human expectations and preferences. This work mainly reviews alignment of text-to-image diffusion models, covering advancements in fundamentals of alignment, alignment techniques of diffusion models, preference benchmarks, and evaluation for diffusion models. Moreover, we discuss key perspectives on current challenges and promising future directions on solving the remaining challenges in alignment of diffusion models. To the best of our knowledge, our work is the first comprehensive review paper for researchers and engineers to comprehend, practice, and research alignment of diffusion models.
>
---
#### [replaced 006] RedDino: A foundation model for red blood cell analysis
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.08180v2](http://arxiv.org/pdf/2508.08180v2)**

> **作者:** Luca Zedda; Andrea Loddo; Cecilia Di Ruberto; Carsten Marr
>
> **摘要:** Red blood cells (RBCs) are essential to human health, and their precise morphological analysis is important for diagnosing hematological disorders. Despite the promise of foundation models in medical diagnostics, comprehensive AI solutions for RBC analysis remain scarce. We present RedDino, a self-supervised foundation model designed for RBC image analysis. RedDino uses an RBC-specific adaptation of the DINOv2 self-supervised learning framework and is trained on a curated dataset of 1.25 million RBC images from diverse acquisition modalities and sources. Extensive evaluations show that RedDino outperforms existing state-of-the-art models on RBC shape classification. Through assessments including linear probing and nearest neighbor classification, we confirm its strong feature representations and generalization ability. Our main contributions are: (1) a foundation model tailored for RBC analysis, (2) ablation studies exploring DINOv2 configurations for RBC modeling, and (3) a detailed evaluation of generalization performance. RedDino addresses key challenges in computational hematology by capturing nuanced morphological features, advancing the development of reliable diagnostic tools. The source code and pretrained models for RedDino are available at https://github.com/Snarci/RedDino, and the pretrained models can be downloaded from our Hugging Face collection at https://huggingface.co/collections/Snarcy/reddino-689a13e29241d2e5690202fc
>
---
#### [replaced 007] VIBE: Video-to-Text Information Bottleneck Evaluation for TL;DR
- **分类: cs.CV; cs.HC; cs.IT; math.IT**

- **链接: [http://arxiv.org/pdf/2505.17423v2](http://arxiv.org/pdf/2505.17423v2)**

> **作者:** Shenghui Chen; Po-han Li; Sandeep Chinchali; Ufuk Topcu
>
> **摘要:** Many decision-making tasks, where both accuracy and efficiency matter, still require human supervision. For example, tasks like traffic officers reviewing hour-long dashcam footage or researchers screening conference videos can benefit from concise summaries that reduce cognitive load and save time. Yet current vision-language models (VLMs) often produce verbose, redundant outputs that hinder task performance. Existing video caption evaluation depends on costly human annotations and overlooks the summaries' utility in downstream tasks. We address these gaps with Video-to-text Information Bottleneck Evaluation (VIBE), an annotation-free method that scores VLM outputs using two metrics: grounding (how well the summary aligns with visual content) and utility (how informative it is for the task). VIBE selects from randomly sampled VLM outputs by ranking them according to the two scores to support effective human decision-making. Human studies on LearningPaper24, SUTD-TrafficQA, and LongVideoBench show that summaries selected by VIBE consistently improve performance-boosting task accuracy by up to 61.23% and reducing response time by 75.77% compared to naive VLM summaries or raw video.
>
---
#### [replaced 008] ScrewSplat: An End-to-End Method for Articulated Object Recognition
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.02146v2](http://arxiv.org/pdf/2508.02146v2)**

> **作者:** Seungyeon Kim; Junsu Ha; Young Hun Kim; Yonghyeon Lee; Frank C. Park
>
> **备注:** 26 pages, 12 figures, Conference on Robot Learning (CoRL) 2025
>
> **摘要:** Articulated object recognition -- the task of identifying both the geometry and kinematic joints of objects with movable parts -- is essential for enabling robots to interact with everyday objects such as doors and laptops. However, existing approaches often rely on strong assumptions, such as a known number of articulated parts; require additional inputs, such as depth images; or involve complex intermediate steps that can introduce potential errors -- limiting their practicality in real-world settings. In this paper, we introduce ScrewSplat, a simple end-to-end method that operates solely on RGB observations. Our approach begins by randomly initializing screw axes, which are then iteratively optimized to recover the object's underlying kinematic structure. By integrating with Gaussian Splatting, we simultaneously reconstruct the 3D geometry and segment the object into rigid, movable parts. We demonstrate that our method achieves state-of-the-art recognition accuracy across a diverse set of articulated objects, and further enables zero-shot, text-guided manipulation using the recovered kinematic model. See the project website at: https://screwsplat.github.io.
>
---
#### [replaced 009] MeshCoder: LLM-Powered Structured Mesh Code Generation from Point Clouds
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.14879v2](http://arxiv.org/pdf/2508.14879v2)**

> **作者:** Bingquan Dai; Li Ray Luo; Qihong Tang; Jie Wang; Xinyu Lian; Hao Xu; Minghan Qin; Xudong Xu; Bo Dai; Haoqian Wang; Zhaoyang Lyu; Jiangmiao Pang
>
> **摘要:** Reconstructing 3D objects into editable programs is pivotal for applications like reverse engineering and shape editing. However, existing methods often rely on limited domain-specific languages (DSLs) and small-scale datasets, restricting their ability to model complex geometries and structures. To address these challenges, we introduce MeshCoder, a novel framework that reconstructs complex 3D objects from point clouds into editable Blender Python scripts. We develop a comprehensive set of expressive Blender Python APIs capable of synthesizing intricate geometries. Leveraging these APIs, we construct a large-scale paired object-code dataset, where the code for each object is decomposed into distinct semantic parts. Subsequently, we train a multimodal large language model (LLM) that translates 3D point cloud into executable Blender Python scripts. Our approach not only achieves superior performance in shape-to-code reconstruction tasks but also facilitates intuitive geometric and topological editing through convenient code modifications. Furthermore, our code-based representation enhances the reasoning capabilities of LLMs in 3D shape understanding tasks. Together, these contributions establish MeshCoder as a powerful and flexible solution for programmatic 3D shape reconstruction and understanding. The project homepage is available at \href{https://daibingquan.github.io/MeshCoder}{this link}.
>
---
#### [replaced 010] HPSv3: Towards Wide-Spectrum Human Preference Score
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.03789v2](http://arxiv.org/pdf/2508.03789v2)**

> **作者:** Yuhang Ma; Yunhao Shui; Xiaoshi Wu; Keqiang Sun; Hongsheng Li
>
> **备注:** ICCV2025
>
> **摘要:** Evaluating text-to-image generation models requires alignment with human perception, yet existing human-centric metrics are constrained by limited data coverage, suboptimal feature extraction, and inefficient loss functions. To address these challenges, we introduce Human Preference Score v3 (HPSv3). (1) We release HPDv3, the first wide-spectrum human preference dataset integrating 1.08M text-image pairs and 1.17M annotated pairwise comparisons from state-of-the-art generative models and low to high-quality real-world images. (2) We introduce a VLM-based preference model trained using an uncertainty-aware ranking loss for fine-grained ranking. Besides, we propose Chain-of-Human-Preference (CoHP), an iterative image refinement method that enhances quality without extra data, using HPSv3 to select the best image at each step. Extensive experiments demonstrate that HPSv3 serves as a robust metric for wide-spectrum image evaluation, and CoHP offers an efficient and human-aligned approach to improve image generation quality. The code and dataset are available at the HPSv3 Homepage.
>
---
#### [replaced 011] Blink-to-code: real-time Morse code communication via eye blink detection and classification
- **分类: cs.CV; 68T45, 92C55; H.5.2; I.2.10; J.3**

- **链接: [http://arxiv.org/pdf/2508.09344v2](http://arxiv.org/pdf/2508.09344v2)**

> **作者:** Anushka Bhatt
>
> **备注:** 4 pages, 4 figures. Preprint on blink-based Morse code communication via webcam for assistive technology. Relevant to computer vision and human-computer interaction
>
> **摘要:** This study proposes a real-time system that translates voluntary eye blinks into Morse code, enabling communication for individuals with severe motor impairments. Using a standard webcam and computer vision, the system detects and classifies blinks as short (dot) or long (dash), then decodes them into alphanumeric characters. Experiments with five participants show 62% decoding accuracy and 18-20 seconds response times, demonstrating a viable, low-cost assistive communication method.
>
---
#### [replaced 012] MambaIC: State Space Models for High-Performance Learned Image Compression
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2503.12461v3](http://arxiv.org/pdf/2503.12461v3)**

> **作者:** Fanhu Zeng; Hao Tang; Yihua Shao; Siyu Chen; Ling Shao; Yan Wang
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** A high-performance image compression algorithm is crucial for real-time information transmission across numerous fields. Despite rapid progress in image compression, computational inefficiency and poor redundancy modeling still pose significant bottlenecks, limiting practical applications. Inspired by the effectiveness of state space models (SSMs) in capturing long-range dependencies, we leverage SSMs to address computational inefficiency in existing methods and improve image compression from multiple perspectives. In this paper, we integrate the advantages of SSMs for better efficiency-performance trade-off and propose an enhanced image compression approach through refined context modeling, which we term MambaIC. Specifically, we explore context modeling to adaptively refine the representation of hidden states. Additionally, we introduce window-based local attention into channel-spatial entropy modeling to reduce potential spatial redundancy during compression, thereby increasing efficiency. Comprehensive qualitative and quantitative results validate the effectiveness and efficiency of our approach, particularly for high-resolution image compression. Code is released at https://github.com/AuroraZengfh/MambaIC.
>
---
#### [replaced 013] Direct Image Classification from Fourier Ptychographic Microscopy Measurements without Reconstruction
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.05054v2](http://arxiv.org/pdf/2505.05054v2)**

> **作者:** Navya Sonal Agarwal; Jan Philipp Schneider; Kanchana Vaishnavi Gandikota; Syed Muhammad Kazim; John Meshreki; Ivo Ihrke; Michael Moeller
>
> **备注:** Presented in ISCS25
>
> **摘要:** The computational imaging technique of Fourier Ptychographic Microscopy (FPM) enables high-resolution imaging with a wide field of view and can serve as an extremely valuable tool, e.g. in the classification of cells in medical applications. However, reconstructing a high-resolution image from tens or even hundreds of measurements is computationally expensive, particularly for a wide field of view. Therefore, in this paper, we investigate the idea of classifying the image content in the FPM measurements directly without performing a reconstruction step first. We show that Convolutional Neural Networks (CNN) can extract meaningful information from measurement sequences, significantly outperforming the classification on a single band-limited image (up to 12 %) while being significantly more efficient than a reconstruction of a high-resolution image. Furthermore, we demonstrate that a learned multiplexing of several raw measurements allows maintaining the classification accuracy while reducing the amount of data (and consequently also the acquisition time) significantly.
>
---
#### [replaced 014] AutoSketch: VLM-assisted Style-Aware Vector Sketch Completion
- **分类: cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2502.06860v3](http://arxiv.org/pdf/2502.06860v3)**

> **作者:** Hsiao-Yuan Chin; I-Chao Shen; Yi-Ting Chiu; Ariel Shamir; Bing-Yu Chen
>
> **备注:** 11 pages, Hsiao-Yuan Chin and I-Chao Shen contributed equally to the paper
>
> **摘要:** The ability to automatically complete a partial sketch that depicts a complex scene, e.g., "a woman chatting with a man in the park", is very useful. However, existing sketch generation methods create sketches from scratch; they do not complete a partial sketch in the style of the original. To address this challenge, we introduce AutoSketch, a styleaware vector sketch completion method that accommodates diverse sketch styles. Our key observation is that the style descriptions of a sketch in natural language preserve the style during automatic sketch completion. Thus, we use a pretrained vision-language model (VLM) to describe the styles of the partial sketches in natural language and replicate these styles using newly generated strokes. We initially optimize the strokes to match an input prompt augmented by style descriptions extracted from the VLM. Such descriptions allow the method to establish a diffusion prior in close alignment with that of the partial sketch. Next, we utilize the VLM to generate an executable style adjustment code that adjusts the strokes to conform to the desired style. We compare our method with existing methods across various sketch styles and prompts, performed extensive ablation studies and qualitative and quantitative evaluations, and demonstrate that AutoSketch can support various sketch scenarios.
>
---
#### [replaced 015] Backpropagation-Free Test-Time Adaptation via Probabilistic Gaussian Alignment
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.15568v2](http://arxiv.org/pdf/2508.15568v2)**

> **作者:** Youjia Zhang; Youngeun Kim; Young-Geun Choi; Hongyeob Kim; Huiling Liu; Sungeun Hong
>
> **摘要:** Test-time adaptation (TTA) enhances the zero-shot robustness under distribution shifts by leveraging unlabeled test data during inference. Despite notable advances, several challenges still limit its broader applicability. First, most methods rely on backpropagation or iterative optimization, which limits scalability and hinders real-time deployment. Second, they lack explicit modeling of class-conditional feature distributions. This modeling is crucial for producing reliable decision boundaries and calibrated predictions, but it remains underexplored due to the lack of both source data and supervision at test time. In this paper, we propose ADAPT, an Advanced Distribution-Aware and backPropagation-free Test-time adaptation method. We reframe TTA as a Gaussian probabilistic inference task by modeling class-conditional likelihoods using gradually updated class means and a shared covariance matrix. This enables closed-form, training-free inference. To correct potential likelihood bias, we introduce lightweight regularization guided by CLIP priors and a historical knowledge bank. ADAPT requires no source data, no gradient updates, and no full access to target data, supporting both online and transductive settings. Extensive experiments across diverse benchmarks demonstrate that our method achieves state-of-the-art performance under a wide range of distribution shifts with superior scalability and robustness.
>
---
#### [replaced 016] Localized Gaussian Splatting Editing with Contextual Awareness
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.00083v2](http://arxiv.org/pdf/2408.00083v2)**

> **作者:** Hanyuan Xiao; Yingshu Chen; Huajian Huang; Haolin Xiong; Jing Yang; Pratusha Prasad; Yajie Zhao
>
> **备注:** WACV 2025
>
> **摘要:** Recent text-guided generation of individual 3D object has achieved great success using diffusion priors. However, these methods are not suitable for object insertion and replacement tasks as they do not consider the background, leading to illumination mismatches within the environment. To bridge the gap, we introduce an illumination-aware 3D scene editing pipeline for 3D Gaussian Splatting (3DGS) representation. Our key observation is that inpainting by the state-of-the-art conditional 2D diffusion model is consistent with background in lighting. To leverage the prior knowledge from the well-trained diffusion models for 3D object generation, our approach employs a coarse-to-fine objection optimization pipeline with inpainted views. In the first coarse step, we achieve image-to-3D lifting given an ideal inpainted view. The process employs 3D-aware diffusion prior from a view-conditioned diffusion model, which preserves illumination present in the conditioning image. To acquire an ideal inpainted image, we introduce an Anchor View Proposal (AVP) algorithm to find a single view that best represents the scene illumination in target region. In the second Texture Enhancement step, we introduce a novel Depth-guided Inpainting Score Distillation Sampling (DI-SDS), which enhances geometry and texture details with the inpainting diffusion prior, beyond the scope of the 3D-aware diffusion prior knowledge in the first coarse step. DI-SDS not only provides fine-grained texture enhancement, but also urges optimization to respect scene lighting. Our approach efficiently achieves local editing with global illumination consistency without explicitly modeling light transport. We demonstrate robustness of our method by evaluating editing in real scenes containing explicit highlight and shadows, and compare against the state-of-the-art text-to-3D editing methods.
>
---
#### [replaced 017] CAMA: Enhancing Multimodal In-Context Learning with Context-Aware Modulated Attention
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17097v2](http://arxiv.org/pdf/2505.17097v2)**

> **作者:** Yanshu Li; Jianjiang Yang; Ziteng Yang; Bozheng Li; Hongyang He; Zhengtao Yao; Ligong Han; Yingjie Victor Chen; Songlin Fei; Dongfang Liu; Ruixiang Tang
>
> **备注:** 14 pages, 8 figures, 5 tables
>
> **摘要:** Multimodal in-context learning (ICL) is emerging as a key capability that enables large vision-language models (LVLMs) to adapt to novel tasks without parameter updates, expanding their utility across various real-world applications. However, ICL remains unstable, even with well-matched in-context demonstrations (ICDs), suggesting that LVLMs struggle to fully utilize the provided context. While existing efforts focus on prompt engineering or post-hoc logit calibration, we instead investigate the underlying attention dynamics to overcome LVLMs' inherent limitations. We identify two critical deficits in their self-attention that impair effective ICL. To bridge the gap, we propose \textbf{Context-Aware Modulated Attention} (CAMA), a plug-and-play and training-free method that dynamically modulates LVLM's attention logits based on the input in-context sequence. CAMA employs a two-stage attention modulation to address both identified deficits, enhancing the focus on semantically significant tokens, particularly visual ones. Across four LVLMs and seven benchmarks, CAMA consistently outperforms vanilla models and baselines, demonstrating great effectiveness and generalization. It can also activate the desired effects of prompt engineering methods and remains robust under diverse sequence configurations. Thus, CAMA paves the way for deeper explorations of attention dynamics to advance multimodal reasoning.
>
---
#### [replaced 018] A Survey on 3D Gaussian Splatting Applications: Segmentation, Editing, and Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.09977v2](http://arxiv.org/pdf/2508.09977v2)**

> **作者:** Shuting He; Peilin Ji; Yitong Yang; Changshuo Wang; Jiayi Ji; Yinglin Wang; Henghui Ding
>
> **备注:** GitHub Repo: https://github.com/heshuting555/Awesome-3DGS-Applications
>
> **摘要:** 3D Gaussian Splatting (3DGS) has recently emerged as a powerful alternative to Neural Radiance Fields (NeRF) for 3D scene representation, offering high-fidelity photorealistic rendering with real-time performance. Beyond novel view synthesis, the explicit and compact nature of 3DGS enables a wide range of downstream applications that require geometric and semantic understanding. This survey provides a comprehensive overview of recent progress in 3DGS applications. It first introduces 2D foundation models that support semantic understanding and control in 3DGS applications, followed by a review of NeRF-based methods that inform their 3DGS counterparts. We then categorize 3DGS applications into segmentation, editing, generation, and other functional tasks. For each, we summarize representative methods, supervision strategies, and learning paradigms, highlighting shared design principles and emerging trends. Commonly used datasets and evaluation protocols are also summarized, along with comparative analyses of recent methods across public benchmarks. To support ongoing research and development, a continually updated repository of papers, code, and resources is maintained at https://github.com/heshuting555/Awesome-3DGS-Applications.
>
---
#### [replaced 019] A Survey of Deep Learning for Geometry Problem Solving
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.11936v5](http://arxiv.org/pdf/2507.11936v5)**

> **作者:** Jianzhe Ma; Wenxuan Wang; Qin Jin
>
> **备注:** Work in progress
>
> **摘要:** Geometry problem solving, a crucial aspect of mathematical reasoning, is vital across various domains, including education, the assessment of AI's mathematical abilities, and multimodal capability evaluation. The recent surge in deep learning technologies, particularly the emergence of multimodal large language models, has significantly accelerated research in this area. This paper provides a survey of the applications of deep learning in geometry problem solving, including (i) a comprehensive summary of the relevant tasks in geometry problem solving; (ii) a thorough review of related deep learning methods; (iii) a detailed analysis of evaluation metrics and methods; and (iv) a critical discussion of the current challenges and future directions that can be explored. Our objective is to offer a comprehensive and practical reference of deep learning for geometry problem solving, thereby fostering further advancements in this field. We create a continuously updated list of papers on GitHub: https://github.com/majianz/dl4gps.
>
---
#### [replaced 020] Explicit Correspondence Matching for Generalizable Neural Radiance Fields
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2304.12294v2](http://arxiv.org/pdf/2304.12294v2)**

> **作者:** Yuedong Chen; Haofei Xu; Qianyi Wu; Chuanxia Zheng; Tat-Jen Cham; Jianfei Cai
>
> **备注:** TPAMI 2025, Project page: https://donydchen.github.io/matchnerf, Code: https://github.com/donydchen/matchnerf
>
> **摘要:** We present a new generalizable NeRF method that is able to directly generalize to new unseen scenarios and perform novel view synthesis with as few as two source views. The key to our approach lies in the explicitly modeled correspondence matching information, so as to provide the geometry prior to the prediction of NeRF color and density for volume rendering. The explicit correspondence matching is quantified with the cosine similarity between image features sampled at the 2D projections of a 3D point on different views, which is able to provide reliable cues about the surface geometry. Unlike previous methods where image features are extracted independently for each view, we consider modeling the cross-view interactions via Transformer cross-attention, which greatly improves the feature matching quality. Our method achieves state-of-the-art results on different evaluation settings, with the experiments showing a strong correlation between our learned cosine feature similarity and volume density, demonstrating the effectiveness and superiority of our proposed method. The code and model are on our project page: https://donydchen.github.io/matchnerf
>
---
#### [replaced 021] Improving U-Net Confidence on TEM Image Data with L2-Regularization, Transfer Learning, and Deep Fine-Tuning
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.16779v2](http://arxiv.org/pdf/2507.16779v2)**

> **作者:** Aiden Ochoa; Xinyuan Xu; Xing Wang
>
> **备注:** Accepted into the ICCV 2025 CV4MS Workshop
>
> **摘要:** With ever-increasing data volumes, it is essential to develop automated approaches for identifying nanoscale defects in transmission electron microscopy (TEM) images. However, compared to features in conventional photographs, nanoscale defects in TEM images exhibit far greater variation due to the complex contrast mechanisms and intricate defect structures. These challenges often result in much less labeled data and higher rates of annotation errors, posing significant obstacles to improving machine learning model performance for TEM image analysis. To address these limitations, we examined transfer learning by leveraging large, pre-trained models used for natural images. We demonstrated that by using the pre-trained encoder and L2-regularization, semantically complex features are ignored in favor of simpler, more reliable cues, substantially improving the model performance. However, this improvement cannot be captured by conventional evaluation metrics such as F1-score, which can be skewed by human annotation errors treated as ground truth. Instead, we introduced novel evaluation metrics that are independent of the annotation accuracy. Using grain boundary detection in UO2 TEM images as a case study, we found that our approach led to a 57% increase in defect detection rate, which is a robust and holistic measure of model performance on the TEM dataset used in this work. Finally, we showed that model self-confidence is only achieved through transfer learning and fine-tuning of very deep layers.
>
---
#### [replaced 022] Evaluation of 3D Counterfactual Brain MRI Generation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.02880v2](http://arxiv.org/pdf/2508.02880v2)**

> **作者:** Pengwei Sun; Wei Peng; Lun Yu Li; Yixin Wang; Kilian M. Pohl
>
> **摘要:** Counterfactual generation offers a principled framework for simulating hypothetical changes in medical imaging, with potential applications in understanding disease mechanisms and generating physiologically plausible data. However, generating realistic structural 3D brain MRIs that respect anatomical and causal constraints remains challenging due to data scarcity, structural complexity, and the lack of standardized evaluation protocols. In this work, we convert six generative models into 3D counterfactual approaches by incorporating an anatomy-guided framework based on a causal graph, in which regional brain volumes serve as direct conditioning inputs. Each model is evaluated with respect to composition, reversibility, realism, effectiveness and minimality on T1-weighted brain MRIs (T1w MRIs) from the Alzheimer's Disease Neuroimaging Initiative (ADNI). In addition, we test the generalizability of each model with respect to T1w MRIs of the National Consortium on Alcohol and Neurodevelopment in Adolescence (NCANDA). Our results indicate that anatomically grounded conditioning successfully modifies the targeted anatomical regions; however, it exhibits limitations in preserving non-targeted structures. Beyond laying the groundwork for more interpretable and clinically relevant generative modeling of brain MRIs, this benchmark highlights the need for novel architectures that more accurately capture anatomical interdependencies.
>
---
#### [replaced 023] UAV-ON: A Benchmark for Open-World Object Goal Navigation with Aerial Agents
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.00288v4](http://arxiv.org/pdf/2508.00288v4)**

> **作者:** Jianqiang Xiao; Yuexuan Sun; Yixin Shao; Boxi Gan; Rongqiang Liu; Yanjing Wu; Weili Guan; Xiang Deng
>
> **备注:** Accepted to ACM MM Dataset Track 2025
>
> **摘要:** Aerial navigation is a fundamental yet underexplored capability in embodied intelligence, enabling agents to operate in large-scale, unstructured environments where traditional navigation paradigms fall short. However, most existing research follows the Vision-and-Language Navigation (VLN) paradigm, which heavily depends on sequential linguistic instructions, limiting its scalability and autonomy. To address this gap, we introduce UAV-ON, a benchmark for large-scale Object Goal Navigation (ObjectNav) by aerial agents in open-world environments, where agents operate based on high-level semantic goals without relying on detailed instructional guidance as in VLN. UAV-ON comprises 14 high-fidelity Unreal Engine environments with diverse semantic regions and complex spatial layouts, covering urban, natural, and mixed-use settings. It defines 1270 annotated target objects, each characterized by an instance-level instruction that encodes category, physical footprint, and visual descriptors, allowing grounded reasoning. These instructions serve as semantic goals, introducing realistic ambiguity and complex reasoning challenges for aerial agents. To evaluate the benchmark, we implement several baseline methods, including Aerial ObjectNav Agent (AOA), a modular policy that integrates instruction semantics with egocentric observations for long-horizon, goal-directed exploration. Empirical results show that all baselines struggle in this setting, highlighting the compounded challenges of aerial navigation and semantic goal grounding. UAV-ON aims to advance research on scalable UAV autonomy driven by semantic goal descriptions in complex real-world environments.
>
---
#### [replaced 024] EHGCN: Hierarchical Euclidean-Hyperbolic Fusion via Motion-Aware GCN for Hybrid Event Stream Perception
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.16616v3](http://arxiv.org/pdf/2504.16616v3)**

> **作者:** Haosheng Chen; Lian Luo; Mengjingcheng Mo; Zhanjie Wu; Guobao Xiao; Ji Gan; Jiaxu Leng; Xinbo Gao
>
> **摘要:** Event cameras, with microsecond temporal resolution and high dynamic range (HDR) characteristics, emit high-speed event stream for perception tasks. Despite the recent advancement in GNN-based perception methods, they are prone to use straightforward pairwise connectivity mechanisms in the pure Euclidean space where they struggle to capture long-range dependencies and fail to effectively characterize the inherent hierarchical structures of non-uniformly distributed event stream. To this end, in this paper we propose a novel approach named EHGCN, which is a pioneer to perceive event stream in both Euclidean and hyperbolic spaces for event vision. In EHGCN, we introduce an adaptive sampling strategy to dynamically regulate sampling rates, retaining discriminative events while attenuating chaotic noise. Then we present a Markov Vector Field (MVF)-driven motion-aware hyperedge generation method based on motion state transition probabilities, thereby eliminating cross-target spurious associations and providing critically topological priors while capturing long-range dependencies between events. Finally, we propose a Euclidean-Hyperbolic GCN to fuse the information locally aggregated and globally hierarchically modeled in Euclidean and hyperbolic spaces, respectively, to achieve hybrid event perception. Experimental results on event perception tasks such as object detection and recognition validate the effectiveness of our approach.
>
---
#### [replaced 025] High-Frequency First: A Two-Stage Approach for Improving Image INR
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.15582v2](http://arxiv.org/pdf/2508.15582v2)**

> **作者:** Sumit Kumar Dam; Mrityunjoy Gain; Eui-Nam Huh; Choong Seon Hong
>
> **备注:** Paper on INR; 4 figures, 8 pages
>
> **摘要:** Implicit Neural Representations (INRs) have emerged as a powerful alternative to traditional pixel-based formats by modeling images as continuous functions over spatial coordinates. A key challenge, however, lies in the spectral bias of neural networks, which tend to favor low-frequency components while struggling to capture high-frequency (HF) details such as sharp edges and fine textures. While prior approaches have addressed this limitation through architectural modifications or specialized activation functions, we propose an orthogonal direction by directly guiding the training process. Specifically, we introduce a two-stage training strategy where a neighbor-aware soft mask adaptively assigns higher weights to pixels with strong local variations, encouraging early focus on fine details. The model then transitions to full-image training. Experimental results show that our approach consistently improves reconstruction quality and complements existing INR methods. As a pioneering attempt to assign frequency-aware importance to pixels in image INR, our work offers a new avenue for mitigating the spectral bias problem.
>
---
#### [replaced 026] LIB-KD: Teaching Inductive Bias for Efficient Vision Transformer Distillation and Compression
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2310.00369v4](http://arxiv.org/pdf/2310.00369v4)**

> **作者:** Gousia Habib; Tausifa Jan Saleem; Ishfaq Ahmad Malik; Brejesh Lall
>
> **备注:** 12 pages, 9 figures
>
> **摘要:** With the rapid development of computer vision, Vision Transformers (ViTs) offer the tantalising prospect of unified information processing across visual and textual domains due to the lack of inherent inductive biases in ViTs. ViTs require enormous datasets for training. We introduce an innovative ensemble-based distillation approach that distils inductive bias from complementary lightweight teacher models to make their applications practical. Prior systems relied solely on convolution-based teaching. However, this method incorporates an ensemble of light teachers with different architectural tendencies, such as convolution and involution, to jointly instruct the student transformer. Because of these unique inductive biases, instructors can accumulate a wide range of knowledge, even from readily identifiable stored datasets, which leads to enhanced student performance. Our proposed framework LIB-KD also involves precomputing and keeping logits in advance, essentially the unnormalized predictions of the model. This optimisation can accelerate the distillation process by eliminating the need for repeated forward passes during knowledge distillation, significantly reducing the computational burden and enhancing efficiency.
>
---
#### [replaced 027] STORM: Token-Efficient Long Video Understanding for Multimodal LLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.04130v3](http://arxiv.org/pdf/2503.04130v3)**

> **作者:** Jindong Jiang; Xiuyu Li; Zhijian Liu; Muyang Li; Guo Chen; Zhiqi Li; De-An Huang; Guilin Liu; Zhiding Yu; Kurt Keutzer; Sungjin Ahn; Jan Kautz; Hongxu Yin; Yao Lu; Song Han; Wonmin Byeon
>
> **摘要:** Recent advances in video-based multimodal large language models (Video-LLMs) have significantly improved video understanding by processing videos as sequences of image frames. However, many existing methods treat frames independently in the vision backbone, lacking explicit temporal modeling, which limits their ability to capture dynamic patterns and efficiently handle long videos. To address these limitations, we introduce STORM (Spatiotemporal TOken Reduction for Multimodal LLMs), a novel architecture incorporating a dedicated temporal encoder between the image encoder and the LLM. Our temporal encoder leverages the Mamba State Space Model to integrate temporal information into image tokens, generating enriched representations that preserve inter-frame dynamics across the entire video sequence. This enriched encoding not only enhances video reasoning capabilities but also enables effective token reduction strategies, including test-time sampling and training-based temporal and spatial pooling, substantially reducing computational demands on the LLM without sacrificing key temporal information. By integrating these techniques, our approach simultaneously reduces training and inference latency while improving performance, enabling efficient and robust video understanding over extended temporal contexts. Extensive evaluations show that STORM achieves state-of-the-art results across various long video understanding benchmarks (more than 5% improvement on MLVU and LongVideoBench) while reducing the computation costs by up to $8\times$ and the decoding latency by 2.4-2.9$\times$ for the fixed numbers of input frames. Project page is available at https://research.nvidia.com/labs/lpr/storm
>
---
#### [replaced 028] Learning Image Priors through Patch-based Diffusion Models for Solving Inverse Problems
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.02462v3](http://arxiv.org/pdf/2406.02462v3)**

> **作者:** Jason Hu; Bowen Song; Xiaojian Xu; Liyue Shen; Jeffrey A. Fessler
>
> **摘要:** Diffusion models can learn strong image priors from underlying data distribution and use them to solve inverse problems, but the training process is computationally expensive and requires lots of data. Such bottlenecks prevent most existing works from being feasible for high-dimensional and high-resolution data such as 3D images. This paper proposes a method to learn an efficient data prior for the entire image by training diffusion models only on patches of images. Specifically, we propose a patch-based position-aware diffusion inverse solver, called PaDIS, where we obtain the score function of the whole image through scores of patches and their positional encoding and utilize this as the prior for solving inverse problems. First of all, we show that this diffusion model achieves an improved memory efficiency and data efficiency while still maintaining the capability to generate entire images via positional encoding. Additionally, the proposed PaDIS model is highly flexible and can be plugged in with different diffusion inverse solvers (DIS). We demonstrate that the proposed PaDIS approach enables solving various inverse problems in both natural and medical image domains, including CT reconstruction, deblurring, and superresolution, given only patch-based priors. Notably, PaDIS outperforms previous DIS methods trained on entire image priors in the case of limited training data, demonstrating the data efficiency of our proposed approach by learning patch-based prior.
>
---
#### [replaced 029] Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11652v2](http://arxiv.org/pdf/2503.11652v2)**

> **作者:** Hiroyasu Akada; Jian Wang; Vladislav Golyanik; Christian Theobalt
>
> **备注:** Project page: https://4dqv.mpi-inf.mpg.de/EgoRear/
>
> **摘要:** Egocentric 3D human pose estimation has been actively studied using cameras installed in front of a head-mounted device (HMD). While frontal placement is the optimal and the only option for some tasks, such as hand tracking, it remains unclear if the same holds for full-body tracking due to self-occlusion and limited field-of-view coverage. Notably, even the state-of-the-art methods often fail to estimate accurate 3D poses in many scenarios, such as when HMD users tilt their heads upward -- a common motion in human activities. A key limitation of existing HMD designs is their neglect of the back of the body, despite its potential to provide crucial 3D reconstruction cues. Hence, this paper investigates the usefulness of rear cameras for full-body tracking. We also show that simply adding rear views to the frontal inputs is not optimal for existing methods due to their dependence on individual 2D joint detectors without effective multi-view integration. To address this issue, we propose a new transformer-based method that refines 2D joint heatmap estimation with multi-view information and heatmap uncertainty, thereby improving 3D pose tracking. Also, we introduce two new large-scale datasets, Ego4View-Syn and Ego4View-RW, for a rear-view evaluation. Our experiments show that the new camera configurations with back views provide superior support for 3D pose tracking compared to only frontal placements. The proposed method achieves significant improvement over the current state of the art (>10% on MPJPE). The source code, trained models, and datasets are available on our project page at https://4dqv.mpi-inf.mpg.de/EgoRear/.
>
---
#### [replaced 030] Review of Demographic Fairness in Face Recognition
- **分类: cs.CV; cs.CR**

- **链接: [http://arxiv.org/pdf/2502.02309v3](http://arxiv.org/pdf/2502.02309v3)**

> **作者:** Ketan Kotwal; Sebastien Marcel
>
> **摘要:** Demographic fairness in face recognition (FR) has emerged as a critical area of research, given its impact on fairness, equity, and reliability across diverse applications. As FR technologies are increasingly deployed globally, disparities in performance across demographic groups -- such as race, ethnicity, and gender -- have garnered significant attention. These biases not only compromise the credibility of FR systems but also raise ethical concerns, especially when these technologies are employed in sensitive domains. This review consolidates extensive research efforts providing a comprehensive overview of the multifaceted aspects of demographic fairness in FR. We systematically examine the primary causes, datasets, assessment metrics, and mitigation approaches associated with demographic disparities in FR. By categorizing key contributions in these areas, this work provides a structured approach to understanding and addressing the complexity of this issue. Finally, we highlight current advancements and identify emerging challenges that need further investigation. This article aims to provide researchers with a unified perspective on the state-of-the-art while emphasizing the critical need for equitable and trustworthy FR systems.
>
---
#### [replaced 031] Geometric-Aware Low-Light Image and Video Enhancement via Depth Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2312.15855v2](http://arxiv.org/pdf/2312.15855v2)**

> **作者:** Yingqi Lin; Xiaogang Xu; Jiafei Wu; Yan Han; Zhe Liu
>
> **备注:** This work has been accepted for publication in the IEEE Transactions on Image Processing
>
> **摘要:** Low-Light Enhancement (LLE) is aimed at improving the quality of photos/videos captured under low-light conditions. It is worth noting that most existing LLE methods do not take advantage of geometric modeling. We believe that incorporating geometric information can enhance LLE performance, as it provides insights into the physical structure of the scene that influences illumination conditions. To address this, we propose a Geometry-Guided Low-Light Enhancement Refine Framework (GG-LLERF) designed to assist low-light enhancement models in learning improved features for LLE by integrating geometric priors into the feature representation space. In this paper, we employ depth priors as the geometric representation. Our approach focuses on the integration of depth priors into various LLE frameworks using a unified methodology. This methodology comprises two key novel modules. First, a depth-aware feature extraction module is designed to inject depth priors into the image representation. Then, Hierarchical Depth-Guided Feature Fusion Module (HDGFFM) is formulated with a cross-domain attention mechanism, which combines depth-aware features with the original image features within the LLE model. We conducted extensive experiments on public low-light image and video enhancement benchmarks. The results illustrate that our designed framework significantly enhances existing LLE methods.
>
---
#### [replaced 032] LBM: Latent Bridge Matching for Fast Image-to-Image Translation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07535v2](http://arxiv.org/pdf/2503.07535v2)**

> **作者:** Clément Chadebec; Onur Tasar; Sanjeev Sreetharan; Benjamin Aubin
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** In this paper, we introduce Latent Bridge Matching (LBM), a new, versatile and scalable method that relies on Bridge Matching in a latent space to achieve fast image-to-image translation. We show that the method can reach state-of-the-art results for various image-to-image tasks using only a single inference step. In addition to its efficiency, we also demonstrate the versatility of the method across different image translation tasks such as object removal, normal and depth estimation, and object relighting. We also derive a conditional framework of LBM and demonstrate its effectiveness by tackling the tasks of controllable image relighting and shadow generation. We provide an implementation at https://github.com/gojasper/LBM.
>
---
#### [replaced 033] Text-to-3D Generation using Jensen-Shannon Score Distillation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.10660v3](http://arxiv.org/pdf/2503.10660v3)**

> **作者:** Khoi Do; Binh-Son Hua
>
> **摘要:** Score distillation sampling is an effective technique to generate 3D models from text prompts, utilizing pre-trained large-scale text-to-image diffusion models as guidance. However, the produced 3D assets tend to be over-saturating, over-smoothing, with limited diversity. These issues are results from a reverse Kullback-Leibler (KL) divergence objective, which makes the optimization unstable and results in mode-seeking behavior. In this paper, we derive a bounded score distillation objective based on Jensen-Shannon divergence (JSD), which stabilizes the optimization process and produces high-quality 3D generation. JSD can match well generated and target distribution, therefore mitigating mode seeking. We provide a practical implementation of JSD by utilizing the theory of generative adversarial networks to define an approximate objective function for the generator, assuming the discriminator is well trained. By assuming the discriminator following a log-odds classifier, we propose a minority sampling algorithm to estimate the gradients of our proposed objective, providing a practical implementation for JSD. We conduct both theoretical and empirical studies to validate our method. Experimental results on T3Bench demonstrate that our method can produce high-quality and diversified 3D assets.
>
---
#### [replaced 034] Adaptive Multi-Order Graph Regularized NMF with Dual Sparsity for Hyperspectral Unmixing
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2503.19258v2](http://arxiv.org/pdf/2503.19258v2)**

> **作者:** Hui Chen; Liangyu Liu; Xianchao Xiu; Wanquan Liu
>
> **备注:** IEEE JSTARS
>
> **摘要:** Hyperspectral unmixing (HU) is a critical yet challenging task in remote sensing. However, existing nonnegative matrix factorization (NMF) methods with graph learning mostly focus on first-order or second-order nearest neighbor relationships and usually require manual parameter tuning, which fails to characterize intrinsic data structures. To address the above issues, we propose a novel adaptive multi-order graph regularized NMF method (MOGNMF) with three key features. First, multi-order graph regularization is introduced into the NMF framework to exploit global and local information comprehensively. Second, these parameters associated with the multi-order graph are learned adaptively through a data-driven approach. Third, dual sparsity is embedded to obtain better robustness, i.e., $\ell_{1/2}$-norm on the abundance matrix and $\ell_{2,1}$-norm on the noise matrix. To solve the proposed model, we develop an alternating minimization algorithm whose subproblems have explicit solutions, thus ensuring effectiveness. Experiments on simulated and real hyperspectral data indicate that the proposed method delivers better unmixing results.
>
---
#### [replaced 035] Not Only Consistency: Enhance Test-Time Adaptation with Spatio-temporal Inconsistency for Remote Physiological Measurement
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.07908v2](http://arxiv.org/pdf/2507.07908v2)**

> **作者:** Xiao Yang; Jiyao Wang; Yuxuan Fan; Can Liu; Houcheng Su; Weichen Guo; Zitong Yu; Dengbo He; Kaishun Wu
>
> **摘要:** Remote physiological measurement (RPM) has emerged as a promising non-invasive method for monitoring physiological signals using the non-contact device. Although various domain adaptation and generalization methods were proposed to promote the adaptability of deep-based RPM models in unseen deployment environments, considerations in aspects such as privacy concerns and real-time adaptation restrict their application in real-world deployment. Thus, we aim to propose a novel fully Test-Time Adaptation (TTA) strategy tailored for RPM tasks in this work. Specifically, based on prior knowledge in physiology and our observations, we noticed not only there is spatio-temporal consistency in the frequency domain of BVP signals, but also that inconsistency in the time domain was significant. Given this, by leveraging both consistency and inconsistency priors, we introduce an innovative expert knowledge-based self-supervised \textbf{C}onsistency-\textbf{i}n\textbf{C}onsistency-\textbf{i}ntegration (\textbf{CiCi}) framework to enhances model adaptation during inference. Besides, our approach further incorporates a gradient dynamic control mechanism to mitigate potential conflicts between priors, ensuring stable adaptation across instances. Through extensive experiments on five diverse datasets under the TTA protocol, our method consistently outperforms existing techniques, presenting state-of-the-art performance in real-time self-supervised adaptation without accessing source data. The code will be released later.
>
---
#### [replaced 036] Continuous Knowledge-Preserving Decomposition with Adaptive Layer Selection for Few-Shot Class-Incremental Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.05017v3](http://arxiv.org/pdf/2501.05017v3)**

> **作者:** Xiaojie Li; Jianlong Wu; Yue Yu; Liqiang Nie; Min Zhang
>
> **备注:** Code: https://github.com/xiaojieli0903/CKPD-FSCIL
>
> **摘要:** Few-Shot Class-Incremental Learning (FSCIL) faces a critical challenge: balancing the retention of prior knowledge with the acquisition of new classes. Existing methods either freeze the backbone to prevent catastrophic forgetting, sacrificing plasticity, or add new modules, incurring high costs. These approaches treat pretrained models as black boxes, overlooking two key opportunities to exploit their internal capacity: reusing redundant representational space within layers and selectively adapting layers based on their sensitivity to forgetting. We propose CKPD-FSCIL, a unified framework that unlocks the underutilized capacity of pretrained weights, achieving a superior stability-plasticity balance with zero inference overhead. Our design integrates two continuously adapting mechanisms: At the weight level, a Continuous Knowledge-Preserving Decomposition mechanism uses feature covariance to split each weight matrix into a frozen subspace that safeguards prior knowledge and a learnable, redundant subspace for new tasks. At the layer level, a Continuous Adaptive Layer Selection mechanism leverages an Adapter Sensitivity Ratio to automatically select layers with the highest redundant capacity and lowest forgetting risk for adaptation. By targeting only safe, high-potential subspaces and layers, CKPD-FSCIL enables efficient adaptation. After each session, the learned adapters are merged back into the original weights, ensuring zero additional parameters or FLOPs during inference. Extensive experiments on multiple FSCIL benchmarks demonstrate that our method consistently outperforms state-of-the-art approaches in both adaptability and knowledge retention. The code is available at https://github.com/xiaojieli0903/CKPD-FSCIL.
>
---
#### [replaced 037] Efficient Density Control for 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.10133v4](http://arxiv.org/pdf/2411.10133v4)**

> **作者:** Xiaobin Deng; Changyu Diao; Min Li; Ruohan Yu; Duanqing Xu
>
> **备注:** Resubmission version in arxiv at arXiv:2508.12313
>
> **摘要:** 3D Gaussian Splatting (3DGS) has demonstrated outstanding performance in novel view synthesis, achieving a balance between rendering quality and real-time performance. 3DGS employs Adaptive Density Control (ADC) to increase the number of Gaussians. However, the clone and split operations within ADC are not sufficiently efficient, impacting optimization speed and detail recovery. Additionally, overfitted Gaussians that affect rendering quality may exist, and the original ADC is unable to remove them. To address these issues, we propose two key innovations: (1) Long-Axis Split, which precisely controls the position, shape, and opacity of child Gaussians to minimize the difference before and after splitting. (2) Recovery-Aware Pruning, which leverages differences in recovery speed after resetting opacity to prune overfitted Gaussians, thereby improving generalization performance. Experimental results show that our method significantly enhances rendering quality. Due to resubmission reasons, this version has been abandoned. The improved version is available at https://xiaobin2001.github.io/improved-gs-web .
>
---
#### [replaced 038] LBONet: Supervised Spectral Descriptors for Shape Analysis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.08272v3](http://arxiv.org/pdf/2411.08272v3)**

> **作者:** Oguzhan Yigit; Richard C. Wilson
>
> **备注:** Accepted to TPAMI 2025. 15 pages, 14 figures
>
> **摘要:** The Laplace-Beltrami operator has established itself in the field of non-rigid shape analysis due to its many useful properties such as being invariant under isometric transformation, having a countable eigensystem forming an orthornormal basis, and fully characterizing geodesic distances of the manifold. However, this invariancy only applies under isometric deformations, which leads to a performance breakdown in many real-world applications. In recent years emphasis has been placed upon extracting optimal features using deep learning methods,however spectral signatures play a crucial role and still add value. In this paper we take a step back, revisiting the LBO and proposing a supervised way to learn several operators on a manifold. Depending on the task, by applying these functions, we can train the LBO eigenbasis to be more task-specific. The optimization of the LBO leads to enormous improvements to established descriptors such as the heat kernel signature in various tasks such as retrieval, classification, segmentation, and correspondence, proving the adaption of the LBO eigenbasis to both global and highly local learning settings.
>
---
#### [replaced 039] Zero-Shot Skeleton-based Action Recognition with Dual Visual-Text Alignment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.14336v2](http://arxiv.org/pdf/2409.14336v2)**

> **作者:** Jidong Kuang; Hongsong Wang; Chaolei Han; Yang Zhang; Jie Gui
>
> **摘要:** Zero-shot action recognition, which addresses the issue of scalability and generalization in action recognition and allows the models to adapt to new and unseen actions dynamically, is an important research topic in computer vision communities. The key to zero-shot action recognition lies in aligning visual features with semantic vectors representing action categories. Most existing methods either directly project visual features onto the semantic space of text category or learn a shared embedding space between the two modalities. However, a direct projection cannot accurately align the two modalities, and learning robust and discriminative embedding space between visual and text representations is often difficult. To address these issues, we introduce Dual Visual-Text Alignment (DVTA) for skeleton-based zero-shot action recognition. The DVTA consists of two alignment modules--Direct Alignment (DA) and Augmented Alignment (AA)--along with a designed Semantic Description Enhancement (SDE). The DA module maps the skeleton features to the semantic space through a specially designed visual projector, followed by the SDE, which is based on cross-attention to enhance the connection between skeleton and text, thereby reducing the gap between modalities. The AA module further strengthens the learning of the embedding space by utilizing deep metric learning to learn the similarity between skeleton and text. Our approach achieves state-of-the-art performances on several popular zero-shot skeleton-based action recognition benchmarks. The code is available at: https://github.com/jidongkuang/DVTA.
>
---
#### [replaced 040] Evaluating the Predictive Value of Preoperative MRI for Erectile Dysfunction Following Radical Prostatectomy
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.03461v2](http://arxiv.org/pdf/2508.03461v2)**

> **作者:** Gideon N. L. Rouwendaal; Daniël Boeke; Inge L. Cox; Henk G. van der Poel; Margriet C. van Dijk-de Haan; Regina G. H. Beets-Tan; Thierry N. Boellaard; Wilson Silva
>
> **备注:** 13 pages, 5 figures, 2 tables. Accepted at PRedictive Intelligence in MEdicine workshop @ MICCAI 2025 (PRIME-MICCAI). This is the submitted manuscript with added link to github repo, funding acknowledgements and authors' names and affiliations. No further post submission improvements or corrections were integrated. Final version not published yet
>
> **摘要:** Accurate preoperative prediction of erectile dysfunction (ED) is important for counseling patients undergoing radical prostatectomy. While clinical features are established predictors, the added value of preoperative MRI remains underexplored. We investigate whether MRI provides additional predictive value for ED at 12 months post-surgery, evaluating four modeling strategies: (1) a clinical-only baseline, representing current state-of-the-art; (2) classical models using handcrafted anatomical features derived from MRI; (3) deep learning models trained directly on MRI slices; and (4) multimodal fusion of imaging and clinical inputs. Imaging-based models (maximum AUC 0.569) slightly outperformed handcrafted anatomical approaches (AUC 0.554) but fell short of the clinical baseline (AUC 0.663). Fusion models offered marginal gains (AUC 0.586) but did not exceed clinical-only performance. SHAP analysis confirmed that clinical features contributed most to predictive performance. Saliency maps from the best-performing imaging model suggested a predominant focus on anatomically plausible regions, such as the prostate and neurovascular bundles. While MRI-based models did not improve predictive performance over clinical features, our findings suggest that they try to capture patterns in relevant anatomical structures and may complement clinical predictors in future multimodal approaches.
>
---
#### [replaced 041] Highly Accurate and Diverse Traffic Data: The DeepScenario Open 3D Dataset
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.17371v3](http://arxiv.org/pdf/2504.17371v3)**

> **作者:** Oussema Dhaouadi; Johannes Meier; Luca Wahl; Jacques Kaiser; Luca Scalerandi; Nick Wandelburg; Zhuolun Zhou; Nijanthan Berinpanathan; Holger Banzhaf; Daniel Cremers
>
> **摘要:** Accurate 3D trajectory data is crucial for advancing autonomous driving. Yet, traditional datasets are usually captured by fixed sensors mounted on a car and are susceptible to occlusion. Additionally, such an approach can precisely reconstruct the dynamic environment in the close vicinity of the measurement vehicle only, while neglecting objects that are further away. In this paper, we introduce the DeepScenario Open 3D Dataset (DSC3D), a high-quality, occlusion-free dataset of 6 degrees of freedom bounding box trajectories acquired through a novel monocular camera drone tracking pipeline. Our dataset includes more than 175,000 trajectories of 14 types of traffic participants and significantly exceeds existing datasets in terms of diversity and scale, containing many unprecedented scenarios such as complex vehicle-pedestrian interaction on highly populated urban streets and comprehensive parking maneuvers from entry to exit. DSC3D dataset was captured in five various locations in Europe and the United States and include: a parking lot, a crowded inner-city, a steep urban intersection, a federal highway, and a suburban intersection. Our 3D trajectory dataset aims to enhance autonomous driving systems by providing detailed environmental 3D representations, which could lead to improved obstacle interactions and safety. We demonstrate its utility across multiple applications including motion prediction, motion planning, scenario mining, and generative reactive traffic agents. Our interactive online visualization platform and the complete dataset are publicly available at https://app.deepscenario.com, facilitating research in motion prediction, behavior modeling, and safety validation.
>
---
#### [replaced 042] OccScene: Semantic Occupancy-based Cross-task Mutual Learning for 3D Scene Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.11183v2](http://arxiv.org/pdf/2412.11183v2)**

> **作者:** Bohan Li; Xin Jin; Jianan Wang; Yukai Shi; Yasheng Sun; Xiaofeng Wang; Zhuang Ma; Baao Xie; Chao Ma; Xiaokang Yang; Wenjun Zeng
>
> **备注:** IEEE Transactions on Pattern Analysis and Machine Intelligence
>
> **摘要:** Recent diffusion models have demonstrated remarkable performance in both 3D scene generation and perception tasks. Nevertheless, existing methods typically separate these two processes, acting as a data augmenter to generate synthetic data for downstream perception tasks. In this work, we propose OccScene, a novel mutual learning paradigm that integrates fine-grained 3D perception and high-quality generation in a unified framework, achieving a cross-task win-win effect. OccScene generates new and consistent 3D realistic scenes only depending on text prompts, guided with semantic occupancy in a joint-training diffusion framework. To align the occupancy with the diffusion latent, a Mamba-based Dual Alignment module is introduced to incorporate fine-grained semantics and geometry as perception priors. Within OccScene, the perception module can be effectively improved with customized and diverse generated scenes, while the perception priors in return enhance the generation performance for mutual benefits. Extensive experiments show that OccScene achieves realistic 3D scene generation in broad indoor and outdoor scenarios, while concurrently boosting the perception models to achieve substantial performance improvements in the 3D perception task of semantic occupancy prediction.
>
---
#### [replaced 043] ViT-FIQA: Assessing Face Image Quality using Vision Transformers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.13957v3](http://arxiv.org/pdf/2508.13957v3)**

> **作者:** Andrea Atzori; Fadi Boutros; Naser Damer
>
> **备注:** Accepted at the IEEE/CVF International Conference on Computer Vision Workshops 2025 (ICCVW 2025)
>
> **摘要:** Face Image Quality Assessment (FIQA) aims to predict the utility of a face image for face recognition (FR) systems. State-of-the-art FIQA methods mainly rely on convolutional neural networks (CNNs), leaving the potential of Vision Transformer (ViT) architectures underexplored. This work proposes ViT-FIQA, a novel approach that extends standard ViT backbones, originally optimized for FR, through a learnable quality token designed to predict a scalar utility score for any given face image. The learnable quality token is concatenated with the standard image patch tokens, and the whole sequence is processed via global self-attention by the ViT encoders to aggregate contextual information across all patches. At the output of the backbone, ViT-FIQA branches into two heads: (1) the patch tokens are passed through a fully connected layer to learn discriminative face representations via a margin-penalty softmax loss, and (2) the quality token is fed into a regression head to learn to predict the face sample's utility. Extensive experiments on challenging benchmarks and several FR models, including both CNN- and ViT-based architectures, demonstrate that ViT-FIQA consistently achieves top-tier performance. These results underscore the effectiveness of transformer-based architectures in modeling face image utility and highlight the potential of ViTs as a scalable foundation for future FIQA research https://cutt.ly/irHlzXUC.
>
---
#### [replaced 044] Cascaded Multi-Scale Attention for Enhanced Multi-Scale Feature Extraction and Interaction with Low-Resolution Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.02197v3](http://arxiv.org/pdf/2412.02197v3)**

> **作者:** Xiangyong Lu; Masanori Suganuma; Takayuki Okatani
>
> **备注:** 10 pages, 6 figures, 5 tables
>
> **摘要:** In real-world applications of image recognition tasks, such as human pose estimation, cameras often capture objects, like human bodies, at low resolutions. This scenario poses a challenge in extracting and leveraging multi-scale features, which is often essential for precise inference. To address this challenge, we propose a new attention mechanism, named cascaded multi-scale attention (CMSA), tailored for use in CNN-ViT hybrid architectures, to handle low-resolution inputs effectively. The design of CMSA enables the extraction and seamless integration of features across various scales without necessitating the downsampling of the input image or feature maps. This is achieved through a novel combination of grouped multi-head self-attention mechanisms with window-based local attention and cascaded fusion of multi-scale features over different scales. This architecture allows for the effective handling of features across different scales, enhancing the model's ability to perform tasks such as human pose estimation, head pose estimation, and more with low-resolution images. Our experimental results show that the proposed method outperforms existing state-of-the-art methods in these areas with fewer parameters, showcasing its potential for broad application in real-world scenarios where capturing high-resolution images is not feasible. Code is available at https://github.com/xyongLu/CMSA.
>
---
#### [replaced 045] FLAIR: Frequency and Locality-Aware Implicit Neural Representations
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.13544v2](http://arxiv.org/pdf/2508.13544v2)**

> **作者:** Sukhun Ko; Dahyeon Kye; Kyle Min; Chanho Eom; Jihyong Oh
>
> **备注:** Please visit our project page at https://cmlab-korea.github.io/FLAIR/
>
> **摘要:** Implicit Neural Representations (INRs) leverage neural networks to map coordinates to corresponding signals, enabling continuous and compact representations. This paradigm has driven significant advances in various vision tasks. However, existing INRs lack frequency selectivity, spatial localization, and sparse representations, leading to an over-reliance on redundant signal components. Consequently, they exhibit spectral bias, tending to learn low-frequency components early while struggling to capture fine high-frequency details. To address these issues, we propose FLAIR (Frequency- and Locality-Aware Implicit Neural Representations), which incorporates two key innovations. The first is RC-GAUSS, a novel activation designed for explicit frequency selection and spatial localization under the constraints of the time-frequency uncertainty principle (TFUP). The second is Wavelet-Energy-Guided Encoding (WEGE), which leverages the discrete wavelet transform (DWT) to compute energy scores and explicitly guide frequency information to the network. Our method consistently outperforms existing INRs in 2D image representation and restoration, as well as 3D reconstruction.
>
---
#### [replaced 046] Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.01225v2](http://arxiv.org/pdf/2508.01225v2)**

> **作者:** Xinyu Chen; Haotian Zhai; Can Zhang; Xiupeng Shi; Ruirui Li
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** In zero-shot setting, test-time adaptation adjusts pre-trained models using unlabeled data from the test phase to enhance performance on unknown test distributions. Existing cache-enhanced TTA methods rely on a low-entropy criterion to select samples for prototype construction, assuming intra-class compactness. However, low-entropy samples may be unreliable under distribution shifts, and the resulting prototypes may not ensure compact intra-class distributions. This study identifies a positive correlation between cache-enhanced performance and intra-class compactness. Based on this observation, we propose a Multi-Cache enhanced Prototype-based Test-Time Adaptation (MCP) featuring three caches: an entropy cache for initializing prototype representations with low-entropy samples, an align cache for integrating visual and textual information to achieve compact intra-class distributions, and a negative cache for prediction calibration using high-entropy samples. We further developed MCP++, a framework incorporating cross-modal prototype alignment and residual learning, introducing prototype residual fine-tuning. Comparative and ablation experiments across 15 downstream tasks demonstrate that the proposed method and framework achieve state-of-the-art generalization performance. Project Page available at: https://zhaihaotian.github.io/MCP-ICCV25/
>
---
#### [replaced 047] HandCraft: Dynamic Sign Generation for Synthetic Data Augmentation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.14345v2](http://arxiv.org/pdf/2508.14345v2)**

> **作者:** Gaston Gustavo Rios; Pedro Dal Bianco; Franco Ronchetti; Facundo Quiroga; Oscar Stanchi; Santiago Ponte Ahón; Waldo Hasperué
>
> **备注:** 26 pages, 4 figures, 9 tables, code available at https://github.com/okason97/HandCraft
>
> **摘要:** Sign Language Recognition (SLR) models face significant performance limitations due to insufficient training data availability. In this article, we address the challenge of limited data in SLR by introducing a novel and lightweight sign generation model based on CMLPe. This model, coupled with a synthetic data pretraining approach, consistently improves recognition accuracy, establishing new state-of-the-art results for the LSFB and DiSPLaY datasets using our Mamba-SL and Transformer-SL classifiers. Our findings reveal that synthetic data pretraining outperforms traditional augmentation methods in some cases and yields complementary benefits when implemented alongside them. Our approach democratizes sign generation and synthetic data pretraining for SLR by providing computationally efficient methods that achieve significant performance improvements across diverse datasets.
>
---
#### [replaced 048] Towards Scalable Training for Handwritten Mathematical Expression Recognition
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.09220v2](http://arxiv.org/pdf/2508.09220v2)**

> **作者:** Haoyang Li; Jiaqing Li; Jialun Cao; Zongyuan Yang; Yongping Xiong
>
> **备注:** We have found that there is a risk of data leakage in our experimental process, which may cause misunderstandings to the community
>
> **摘要:** Large foundation models have achieved significant performance gains through scalable training on massive datasets. However, the field of \textbf{H}andwritten \textbf{M}athematical \textbf{E}xpression \textbf{R}ecognition (HMER) has been impeded by the scarcity of data, primarily due to the arduous and costly process of manual annotation. To bridge this gap, we propose a novel method integrating limited handwritten formulas with large-scale LaTeX-rendered formulas by developing a scalable data engine to generate complex and consistent LaTeX sequences. With this engine, we built the largest formula dataset to date, termed \texttt{Tex80M}, comprising over 80 million high-quality training instances. Then we propose \texttt{TexTeller}, the first HMER model trained at scale, by mix-training \texttt{Tex80M} with a relatively small HME dataset. The expansive training dataset and our refined pipeline have equipped \texttt{TexTeller} with state-of-the-art (SOTA) performance across nearly all benchmarks. To advance the field, we will openly release our complete model, entire dataset, and full codebase, enabling further research building upon our contributions.
>
---
#### [replaced 049] Multi-Level Knowledge Distillation and Dynamic Self-Supervised Learning for Continual Learning
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.12692v2](http://arxiv.org/pdf/2508.12692v2)**

> **作者:** Taeheon Kim; San Kim; Minhyuk Seo; Dongjae Jeon; Wonje Jeung; Jonghyun Choi
>
> **摘要:** Class-incremental with repetition (CIR), where previously trained classes repeatedly introduced in future tasks, is a more realistic scenario than the traditional class incremental setup, which assumes that each task contains unseen classes. CIR assumes that we can easily access abundant unlabeled data from external sources, such as the Internet. Therefore, we propose two components that efficiently use the unlabeled data to ensure the high stability and the plasticity of models trained in CIR setup. First, we introduce multi-level knowledge distillation (MLKD) that distills knowledge from multiple previous models across multiple perspectives, including features and logits, so the model can maintain much various previous knowledge. Moreover, we implement dynamic self-supervised loss (SSL) to utilize the unlabeled data that accelerates the learning of new classes, while dynamic weighting of SSL keeps the focus of training to the primary task. Both of our proposed components significantly improve the performance in CIR setup, achieving 2nd place in the CVPR 5th CLVISION Challenge.
>
---
#### [replaced 050] A Novel Dataset for Video-Based Neurodivergent Classification Leveraging Extra-Stimulatory Behavior
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.04598v2](http://arxiv.org/pdf/2409.04598v2)**

> **作者:** Manuel Serna-Aguilera; Xuan Bac Nguyen; Han-Seok Seo; Khoa Luu
>
> **摘要:** Facial expressions and actions differ among different individuals at varying degrees of intensity given responses to external stimuli, particularly among those that are neurodivergent. Such behaviors affect people in terms of overall health, communication, and sensory processing. Deep learning can be responsibly leveraged to improve productivity in addressing this task, and help medical professionals to accurately understand such behaviors. In this work, we introduce the Video ASD dataset-a dataset that contains video frame convolutional and attention map feature data-to foster further progress in the task of ASD classification. Unlike many recent studies in ASD classification with MRI data, which require expensive specialized equipment, our method utilizes a powerful but relatively affordable GPU, a standard computer setup, and a video camera for inference. Results show that our model effectively generalizes and understands key differences in the distinct movements of the children. Additionally, we test foundation models on this data to showcase how movement noise affects performance and the need for more data and more complex labels.
>
---
