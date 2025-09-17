# 计算机视觉 cs.CV

- **最新发布 132 篇**

- **更新 61 篇**

## 最新发布

#### [new 001] StyleSculptor: Zero-Shot Style-Controllable 3D Asset Generation with Texture-Geometry Dual Guidance
- **分类: cs.CV**

- **简介: 该论文提出StyleSculptor，一种无需训练的零样本3D资产生成方法，解决风格可控问题。通过纹理-几何双引导机制，实现高保真3D生成，提升风格控制精度与灵活性。**

- **链接: [http://arxiv.org/pdf/2509.13301v1](http://arxiv.org/pdf/2509.13301v1)**

> **作者:** Zefan Qu; Zhenwei Wang; Haoyuan Wang; Ke Xu; Gerhard Hancke; Rynson W. H. Lau
>
> **备注:** SIGGRAPH Asia 2025 Conference Paper
>
> **摘要:** Creating 3D assets that follow the texture and geometry style of existing ones is often desirable or even inevitable in practical applications like video gaming and virtual reality. While impressive progress has been made in generating 3D objects from text or images, creating style-controllable 3D assets remains a complex and challenging problem. In this work, we propose StyleSculptor, a novel training-free approach for generating style-guided 3D assets from a content image and one or more style images. Unlike previous works, StyleSculptor achieves style-guided 3D generation in a zero-shot manner, enabling fine-grained 3D style control that captures the texture, geometry, or both styles of user-provided style images. At the core of StyleSculptor is a novel Style Disentangled Attention (SD-Attn) module, which establishes a dynamic interaction between the input content image and style image for style-guided 3D asset generation via a cross-3D attention mechanism, enabling stable feature fusion and effective style-guided generation. To alleviate semantic content leakage, we also introduce a style-disentangled feature selection strategy within the SD-Attn module, which leverages the variance of 3D feature patches to disentangle style- and content-significant channels, allowing selective feature injection within the attention framework. With SD-Attn, the network can dynamically compute texture-, geometry-, or both-guided features to steer the 3D generation process. Built upon this, we further propose the Style Guided Control (SGC) mechanism, which enables exclusive geometry- or texture-only stylization, as well as adjustable style intensity control. Extensive experiments demonstrate that StyleSculptor outperforms existing baseline methods in producing high-fidelity 3D assets.
>
---
#### [new 002] Axis-Aligned 3D Stalk Diameter Estimation from RGB-D Imagery
- **分类: cs.CV**

- **简介: 该论文提出一种基于RGB-D图像的几何感知计算机视觉方法，用于估计作物茎秆直径。任务是高通量表型分析，解决传统测量方法费时、易错的问题。方法结合深度学习分割、点云重建与PCA对齐切片，实现鲁棒直径估计。**

- **链接: [http://arxiv.org/pdf/2509.12511v1](http://arxiv.org/pdf/2509.12511v1)**

> **作者:** Benjamin Vail; Rahul Harsha Cheppally; Ajay Sharda; Sidharth Rai
>
> **备注:** 13 pages, 8 figures, 4 tables
>
> **摘要:** Accurate, high-throughput phenotyping is a critical component of modern crop breeding programs, especially for improving traits such as mechanical stability, biomass production, and disease resistance. Stalk diameter is a key structural trait, but traditional measurement methods are labor-intensive, error-prone, and unsuitable for scalable phenotyping. In this paper, we present a geometry-aware computer vision pipeline for estimating stalk diameter from RGB-D imagery. Our method integrates deep learning-based instance segmentation, 3D point cloud reconstruction, and axis-aligned slicing via Principal Component Analysis (PCA) to perform robust diameter estimation. By mitigating the effects of curvature, occlusion, and image noise, this approach offers a scalable and reliable solution to support high-throughput phenotyping in breeding and agronomic research.
>
---
#### [new 003] A-TDOM: Active TDOM via On-the-Fly 3DGS
- **分类: cs.CV**

- **简介: 该论文提出A-TDOM方法，解决传统TDOM生成延迟与质量下降问题。通过On-the-Fly 3DGS优化，实现实时生成高质量TDOM，提升城市管理和规划等领域的应用效率。**

- **链接: [http://arxiv.org/pdf/2509.12759v1](http://arxiv.org/pdf/2509.12759v1)**

> **作者:** Yiwei Xu; Xiang Wang; Yifei Yu; Wentian Gan; Luca Morelli; Giulio Perda; Xiongwu Xiao; Zongqian Zhan; Xin Wang; Fabio Remondino
>
> **摘要:** True Digital Orthophoto Map (TDOM) serves as a crucial geospatial product in various fields such as urban management, city planning, land surveying, etc. However, traditional TDOM generation methods generally rely on a complex offline photogrammetric pipeline, resulting in delays that hinder real-time applications. Moreover, the quality of TDOM may degrade due to various challenges, such as inaccurate camera poses or Digital Surface Model (DSM) and scene occlusions. To address these challenges, this work introduces A-TDOM, a near real-time TDOM generation method based on On-the-Fly 3DGS optimization. As each image is acquired, its pose and sparse point cloud are computed via On-the-Fly SfM. Then new Gaussians are integrated and optimized into previously unseen or coarsely reconstructed regions. By integrating with orthogonal splatting, A-TDOM can render just after each update of a new 3DGS field. Initial experiments on multiple benchmarks show that the proposed A-TDOM is capable of actively rendering TDOM in near real-time, with 3DGS optimization for each new image in seconds while maintaining acceptable rendering quality and TDOM geometric accuracy.
>
---
#### [new 004] ICDAR 2025 Competition on FEw-Shot Text line segmentation of ancient handwritten documents (FEST)
- **分类: cs.CV**

- **简介: 该论文介绍FEST竞赛，任务是用少量标注数据分割古籍手写文本行。解决传统方法依赖大量标注数据的问题，推动少样本学习方法在历史文献分析中的应用。**

- **链接: [http://arxiv.org/pdf/2509.12965v1](http://arxiv.org/pdf/2509.12965v1)**

> **作者:** Silvia Zottin; Axel De Nardin; Giuseppe Branca; Claudio Piciarelli; Gian Luca Foresti
>
> **备注:** Accepted to ICDAR 2025
>
> **摘要:** Text line segmentation is a critical step in handwritten document image analysis. Segmenting text lines in historical handwritten documents, however, presents unique challenges due to irregular handwriting, faded ink, and complex layouts with overlapping lines and non-linear text flow. Furthermore, the scarcity of large annotated datasets renders fully supervised learning approaches impractical for such materials. To address these challenges, we introduce the Few-Shot Text Line Segmentation of Ancient Handwritten Documents (FEST) Competition. Participants are tasked with developing systems capable of segmenting text lines in U-DIADS-TL dataset, using only three annotated images per manuscript for training. The competition dataset features a diverse collection of ancient manuscripts exhibiting a wide range of layouts, degradation levels, and non-standard formatting, closely reflecting real-world conditions. By emphasizing few-shot learning, FEST competition aims to promote the development of robust and adaptable methods that can be employed by humanities scholars with minimal manual annotation effort, thus fostering broader adoption of automated document analysis tools in historical research.
>
---
#### [new 005] DisorientLiDAR: Physical Attacks on LiDAR-based Localization
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出DisorientLiDAR，针对基于LiDAR的定位系统设计物理攻击。通过移除关键点，破坏定位精度，并在真实环境中验证攻击效果，揭示其安全风险。属于自动驾驶安全领域，解决LiDAR定位系统的物理攻击问题。**

- **链接: [http://arxiv.org/pdf/2509.12595v1](http://arxiv.org/pdf/2509.12595v1)**

> **作者:** Yizhen Lao; Yu Zhang; Ziting Wang; Chengbo Wang; Yifei Xue; Wanpeng Shao
>
> **摘要:** Deep learning models have been shown to be susceptible to adversarial attacks with visually imperceptible perturbations. Even this poses a serious security challenge for the localization of self-driving cars, there has been very little exploration of attack on it, as most of adversarial attacks have been applied to 3D perception. In this work, we propose a novel adversarial attack framework called DisorientLiDAR targeting LiDAR-based localization. By reverse-engineering localization models (e.g., feature extraction networks), adversaries can identify critical keypoints and strategically remove them, thereby disrupting LiDAR-based localization. Our proposal is first evaluated on three state-of-the-art point-cloud registration models (HRegNet, D3Feat, and GeoTransformer) using the KITTI dataset. Experimental results demonstrate that removing regions containing Top-K keypoints significantly degrades their registration accuracy. We further validate the attack's impact on the Autoware autonomous driving platform, where hiding merely a few critical regions induces noticeable localization drift. Finally, we extended our attacks to the physical world by hiding critical regions with near-infrared absorptive materials, thereby successfully replicate the attack effects observed in KITTI data. This step has been closer toward the realistic physical-world attack that demonstrate the veracity and generality of our proposal.
>
---
#### [new 006] More performant and scalable: Rethinking contrastive vision-language pre-training of radiology in the LLM era
- **分类: cs.CV**

- **简介: 该论文研究医学影像与文本的对比预训练任务，旨在提升医疗AI系统的性能与可扩展性。利用LLM自动提取诊断标签生成低成本数据集，改进视觉编码器训练，实现更优的跨模态对齐效果。**

- **链接: [http://arxiv.org/pdf/2509.13175v1](http://arxiv.org/pdf/2509.13175v1)**

> **作者:** Yingtai Li; Haoran Lai; Xiaoqian Zhou; Shuai Ming; Wenxin Ma; Wei Wei; Shaohua Kevin Zhou
>
> **备注:** MICCAI 2025
>
> **摘要:** The emergence of Large Language Models (LLMs) presents unprecedented opportunities to revolutionize medical contrastive vision-language pre-training. In this paper, we show how LLMs can facilitate large-scale supervised pre-training, thereby advancing vision-language alignment. We begin by demonstrate that modern LLMs can automatically extract diagnostic labels from radiology reports with remarkable precision (>96\% AUC in our experiments) without complex prompt engineering, enabling the creation of large-scale "silver-standard" datasets at a minimal cost (~\$3 for 50k CT image-report pairs). Further, we find that vision encoder trained on this "silver-standard" dataset achieves performance comparable to those trained on labels extracted by specialized BERT-based models, thereby democratizing the access to large-scale supervised pre-training. Building on this foundation, we proceed to reveal that supervised pre-training fundamentally improves contrastive vision-language alignment. Our approach achieves state-of-the-art performance using only a 3D ResNet-18 with vanilla CLIP training, including 83.8\% AUC for zero-shot diagnosis on CT-RATE, 77.3\% AUC on RAD-ChestCT, and substantial improvements in cross-modal retrieval (MAP@50=53.7\% for image-image, Recall@100=52.2\% for report-image). These results demonstrate the potential of utilizing LLMs to facilitate {\bf more performant and scalable} medical AI systems. Our code is avaiable at https://github.com/SadVoxel/More-performant-and-scalable.
>
---
#### [new 007] Evaluating Robustness of Vision-Language Models Under Noisy Conditions
- **分类: cs.CV**

- **简介: 该论文评估视觉-语言模型在噪声条件下的鲁棒性，通过引入控制扰动和多种评价指标，揭示模型性能与噪声类型、数据集特性及模型规模的关系，为鲁棒多模态学习提供基准。**

- **链接: [http://arxiv.org/pdf/2509.12492v1](http://arxiv.org/pdf/2509.12492v1)**

> **作者:** Purushoth; Alireza
>
> **摘要:** Vision-Language Models (VLMs) have attained exceptional success across multimodal tasks such as image captioning and visual question answering. However, their robustness under noisy conditions remains unfamiliar. In this study, we present a comprehensive evaluation framework to evaluate the performance of several state-of-the-art VLMs under controlled perturbations, including lighting variation, motion blur, and compression artifacts. We used both lexical-based metrics (BLEU, METEOR, ROUGE, CIDEr) and neural-based similarity measures using sentence embeddings to quantify semantic alignment. Our experiments span diverse datasets, revealing key insights: (1) descriptiveness of ground-truth captions significantly influences model performance; (2) larger models like LLaVA excel in semantic understanding but do not universally outperform smaller models; and (3) certain noise types, such as JPEG compression and motion blur, dramatically degrade performance across models. Our findings highlight the nuanced trade-offs between model size, dataset characteristics, and noise resilience, offering a standardized benchmark for future robust multimodal learning.
>
---
#### [new 008] DyGLNet: Hybrid Global-Local Feature Fusion with Dynamic Upsampling for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文提出DyGLNet用于医学图像分割，解决多尺度病变、边界模糊及计算复杂等问题。通过融合全局与局部特征，并采用动态上采样机制，提升分割精度与效率。实验表明其在多个数据集上表现优异。**

- **链接: [http://arxiv.org/pdf/2509.12763v1](http://arxiv.org/pdf/2509.12763v1)**

> **作者:** Yican Zhao; Ce Wang; You Hao; Lei Li; Tianli Liao
>
> **备注:** 18pages, under review
>
> **摘要:** Medical image segmentation grapples with challenges including multi-scale lesion variability, ill-defined tissue boundaries, and computationally intensive processing demands. This paper proposes the DyGLNet, which achieves efficient and accurate segmentation by fusing global and local features with a dynamic upsampling mechanism. The model innovatively designs a hybrid feature extraction module (SHDCBlock), combining single-head self-attention and multi-scale dilated convolutions to model local details and global context collaboratively. We further introduce a dynamic adaptive upsampling module (DyFusionUp) to realize high-fidelity reconstruction of feature maps based on learnable offsets. Then, a lightweight design is adopted to reduce computational overhead. Experiments on seven public datasets demonstrate that DyGLNet outperforms existing methods, particularly excelling in boundary accuracy and small-object segmentation. Meanwhile, it exhibits lower computation complexity, enabling an efficient and reliable solution for clinical medical image analysis. The code will be made available soon.
>
---
#### [new 009] Cott-ADNet: Lightweight Real-Time Cotton Boll and Flower Detection Under Field Conditions
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出Cott-ADNet，用于实时检测棉花铃和花。针对田间复杂环境下的低效采摘问题，改进YOLOv11n结构，提升检测精度与效率，实现高准确率与低计算量，为自动化采摘提供可靠方案。**

- **链接: [http://arxiv.org/pdf/2509.12442v1](http://arxiv.org/pdf/2509.12442v1)**

> **作者:** Rui-Feng Wang; Mingrui Xu; Matthew C Bauer; Iago Beffart Schardong; Xiaowen Ma; Kangning Cui
>
> **备注:** 14 pages, 5 figures, 1 table
>
> **摘要:** Cotton is one of the most important natural fiber crops worldwide, yet harvesting remains limited by labor-intensive manual picking, low efficiency, and yield losses from missing the optimal harvest window. Accurate recognition of cotton bolls and their maturity is therefore essential for automation, yield estimation, and breeding research. We propose Cott-ADNet, a lightweight real-time detector tailored to cotton boll and flower recognition under complex field conditions. Building on YOLOv11n, Cott-ADNet enhances spatial representation and robustness through improved convolutional designs, while introducing two new modules: a NeLU-enhanced Global Attention Mechanism to better capture weak and low-contrast features, and a Dilated Receptive Field SPPF to expand receptive fields for more effective multi-scale context modeling at low computational cost. We curate a labeled dataset of 4,966 images, and release an external validation set of 1,216 field images to support future research. Experiments show that Cott-ADNet achieves 91.5% Precision, 89.8% Recall, 93.3% mAP50, 71.3% mAP, and 90.6% F1-Score with only 7.5 GFLOPs, maintaining stable performance under multi-scale and rotational variations. These results demonstrate Cott-ADNet as an accurate and efficient solution for in-field deployment, and thus provide a reliable basis for automated cotton harvesting and high-throughput phenotypic analysis. Code and dataset is available at https://github.com/SweefongWong/Cott-ADNet.
>
---
#### [new 010] Intelligent Vacuum Thermoforming Process
- **分类: cs.CV; cs.LG; I.2.10; I.4.9**

- **简介: 该论文提出基于视觉的质量控制系统，用于优化真空热成型工艺参数，解决材料和模具差异导致的质量不一致问题。通过图像数据与k近邻算法调整加热和真空参数，提升产品合格率与生产效率。**

- **链接: [http://arxiv.org/pdf/2509.13250v1](http://arxiv.org/pdf/2509.13250v1)**

> **作者:** Andi Kuswoyo; Christos Margadji; Sebastian W. Pattinson
>
> **备注:** Contains 6 figures in total, 15 pages. Under revision for Journal of Intelligent Manufacturing
>
> **摘要:** Ensuring consistent quality in vacuum thermoforming presents challenges due to variations in material properties and tooling configurations. This research introduces a vision-based quality control system to predict and optimise process parameters, thereby enhancing part quality with minimal data requirements. A comprehensive dataset was developed using visual data from vacuum-formed samples subjected to various process parameters, supplemented by image augmentation techniques to improve model training. A k-Nearest Neighbour algorithm was subsequently employed to identify adjustments needed in process parameters by mapping low-quality parts to their high-quality counterparts. The model exhibited strong performance in adjusting heating power, heating time, and vacuum time to reduce defects and improve production efficiency.
>
---
#### [new 011] Two-Stage Decoupling Framework for Variable-Length Glaucoma Prognosis
- **分类: cs.CV**

- **简介: 该论文提出一种两阶段解耦框架（TSDF），用于解决可变长度青光眼预后的任务。针对现有方法输入长度固定、数据量小的问题，TSDF通过自监督学习聚合多数据集并引入注意力机制处理序列数据，提升模型性能与灵活性。**

- **链接: [http://arxiv.org/pdf/2509.12453v1](http://arxiv.org/pdf/2509.12453v1)**

> **作者:** Yiran Song; Yikai Zhang; Silvia Orengo-Nania; Nian Wang; Fenglong Ma; Rui Zhang; Yifan Peng; Mingquan Lin
>
> **备注:** 11 pages.2 figures, 4 tables
>
> **摘要:** Glaucoma is one of the leading causes of irreversible blindness worldwide. Glaucoma prognosis is essential for identifying at-risk patients and enabling timely intervention to prevent blindness. Many existing approaches rely on historical sequential data but are constrained by fixed-length inputs, limiting their flexibility. Additionally, traditional glaucoma prognosis methods often employ end-to-end models, which struggle with the limited size of glaucoma datasets. To address these challenges, we propose a Two-Stage Decoupling Framework (TSDF) for variable-length glaucoma prognosis. In the first stage, we employ a feature representation module that leverages self-supervised learning to aggregate multiple glaucoma datasets for training, disregarding differences in their supervisory information. This approach enables datasets of varying sizes to learn better feature representations. In the second stage, we introduce a temporal aggregation module that incorporates an attention-based mechanism to process sequential inputs of varying lengths, ensuring flexible and efficient utilization of all available data. This design significantly enhances model performance while maintaining a compact parameter size. Extensive experiments on two benchmark glaucoma datasets:the Ocular Hypertension Treatment Study (OHTS) and the Glaucoma Real-world Appraisal Progression Ensemble (GRAPE),which differ significantly in scale and clinical settings,demonstrate the effectiveness and robustness of our approach.
>
---
#### [new 012] RadGame: An AI-Powered Platform for Radiology Education
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出RadGame，一个AI驱动的放射学教育平台，旨在通过游戏化方式提升定位病变和生成报告的能力。利用公开数据集与AI反馈，解决传统培训中反馈不足的问题，显著提高学习效果。**

- **链接: [http://arxiv.org/pdf/2509.13270v1](http://arxiv.org/pdf/2509.13270v1)**

> **作者:** Mohammed Baharoon; Siavash Raissi; John S. Jun; Thibault Heintz; Mahmoud Alabbad; Ali Alburkani; Sung Eun Kim; Kent Kleinschmidt; Abdulrahman O. Alhumaydhi; Mohannad Mohammed G. Alghamdi; Jeremy Francis Palacio; Mohammed Bukhaytan; Noah Michael Prudlo; Rithvik Akula; Brady Chrisler; Benjamin Galligos; Mohammed O. Almutairi; Mazeen Mohammed Alanazi; Nasser M. Alrashdi; Joel Jihwan Hwang; Sri Sai Dinesh Jaliparthi; Luke David Nelson; Nathaniel Nguyen; Sathvik Suryadevara; Steven Kim; Mohammed F. Mohammed; Yevgeniy R. Semenov; Kun-Hsing Yu; Abdulrhman Aljouie; Hassan AlOmaish; Adam Rodman; Pranav Rajpurkar
>
> **摘要:** We introduce RadGame, an AI-powered gamified platform for radiology education that targets two core skills: localizing findings and generating reports. Traditional radiology training is based on passive exposure to cases or active practice with real-time input from supervising radiologists, limiting opportunities for immediate and scalable feedback. RadGame addresses this gap by combining gamification with large-scale public datasets and automated, AI-driven feedback that provides clear, structured guidance to human learners. In RadGame Localize, players draw bounding boxes around abnormalities, which are automatically compared to radiologist-drawn annotations from public datasets, and visual explanations are generated by vision-language models for user missed findings. In RadGame Report, players compose findings given a chest X-ray, patient age and indication, and receive structured AI feedback based on radiology report generation metrics, highlighting errors and omissions compared to a radiologist's written ground truth report from public datasets, producing a final performance and style score. In a prospective evaluation, participants using RadGame achieved a 68% improvement in localization accuracy compared to 17% with traditional passive methods and a 31% improvement in report-writing accuracy compared to 4% with traditional methods after seeing the same cases. RadGame highlights the potential of AI-driven gamification to deliver scalable, feedback-rich radiology training and reimagines the application of medical AI resources in education.
>
---
#### [new 013] Exploring Metric Fusion for Evaluation of NeRFs
- **分类: cs.CV**

- **简介: 该论文属于NeRF评估任务，旨在解决单一指标难以全面评价NeRF生成效果的问题。通过融合DISTS和VMAF两种指标，并测试不同融合策略，提升与主观评分的相关性，验证其在不同数据集上的有效性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.12836v1](http://arxiv.org/pdf/2509.12836v1)**

> **作者:** Shreyas Shivakumara; Gabriel Eilertsen; Karljohan Lundin Palmerius
>
> **备注:** Accepted for 17th International Conference on Quality of Multimedia Experience (QoMEX 25)
>
> **摘要:** Neural Radiance Fields (NeRFs) have demonstrated significant potential in synthesizing novel viewpoints. Evaluating the NeRF-generated outputs, however, remains a challenge due to the unique artifacts they exhibit, and no individual metric performs well across all datasets. We hypothesize that combining two successful metrics, Deep Image Structure and Texture Similarity (DISTS) and Video Multi-Method Assessment Fusion (VMAF), based on different perceptual methods, can overcome the limitations of individual metrics and achieve improved correlation with subjective quality scores. We experiment with two normalization strategies for the individual metrics and two fusion strategies to evaluate their impact on the resulting correlation with the subjective scores. The proposed pipeline is tested on two distinct datasets, Synthetic and Outdoor, and its performance is evaluated across three different configurations. We present a detailed analysis comparing the correlation coefficients of fusion methods and individual scores with subjective scores to demonstrate the robustness and generalizability of the fusion metrics.
>
---
#### [new 014] CIARD: Cyclic Iterative Adversarial Robustness Distillation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CIARD方法，解决对抗鲁棒性蒸馏中学生模型性能下降问题。通过多教师框架与持续对抗重训练，提升模型在干净样本和对抗样本上的综合表现。属于模型压缩与对抗防御任务。**

- **链接: [http://arxiv.org/pdf/2509.12633v1](http://arxiv.org/pdf/2509.12633v1)**

> **作者:** Liming Lu; Shuchao Pang; Xu Zheng; Xiang Gu; Anan Du; Yunhuai Liu; Yongbin Zhou
>
> **摘要:** Adversarial robustness distillation (ARD) aims to transfer both performance and robustness from teacher model to lightweight student model, enabling resilient performance on resource-constrained scenarios. Though existing ARD approaches enhance student model's robustness, the inevitable by-product leads to the degraded performance on clean examples. We summarize the causes of this problem inherent in existing methods with dual-teacher framework as: 1. The divergent optimization objectives of dual-teacher models, i.e., the clean and robust teachers, impede effective knowledge transfer to the student model, and 2. The iteratively generated adversarial examples during training lead to performance deterioration of the robust teacher model. To address these challenges, we propose a novel Cyclic Iterative ARD (CIARD) method with two key innovations: a. A multi-teacher framework with contrastive push-loss alignment to resolve conflicts in dual-teacher optimization objectives, and b. Continuous adversarial retraining to maintain dynamic teacher robustness against performance degradation from the varying adversarial examples. Extensive experiments on CIFAR-10, CIFAR-100, and Tiny-ImageNet demonstrate that CIARD achieves remarkable performance with an average 3.53 improvement in adversarial defense rates across various attack scenarios and a 5.87 increase in clean sample accuracy, establishing a new benchmark for balancing model robustness and generalization. Our code is available at https://github.com/eminentgu/CIARD
>
---
#### [new 015] AsyMoE: Leveraging Modal Asymmetry for Enhanced Expert Specialization in Large Vision-Language Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出AsyMoE架构，解决LVLM中视觉与语言模态不对称导致的专家专业化不足问题。通过设计三种专家组，提升跨模态交互与上下文保持能力，实验显示其在准确率和参数效率上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.12715v1](http://arxiv.org/pdf/2509.12715v1)**

> **作者:** Heng Zhang; Haichuan Hu; Yaomin Shen; Weihao Yu; Yilei Yuan; Haochen You; Guo Cheng; Zijian Zhang; Lubin Gan; Huihui Wei; Hao Zhang; Jin Huang
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated impressive performance on multimodal tasks through scaled architectures and extensive training. However, existing Mixture of Experts (MoE) approaches face challenges due to the asymmetry between visual and linguistic processing. Visual information is spatially complete, while language requires maintaining sequential context. As a result, MoE models struggle to balance modality-specific features and cross-modal interactions. Through systematic analysis, we observe that language experts in deeper layers progressively lose contextual grounding and rely more on parametric knowledge rather than utilizing the provided visual and linguistic information. To address this, we propose AsyMoE, a novel architecture that models this asymmetry using three specialized expert groups. We design intra-modality experts for modality-specific processing, hyperbolic inter-modality experts for hierarchical cross-modal interactions, and evidence-priority language experts to suppress parametric biases and maintain contextual grounding. Extensive experiments demonstrate that AsyMoE achieves 26.58% and 15.45% accuracy improvements over vanilla MoE and modality-specific MoE respectively, with 25.45% fewer activated parameters than dense models.
>
---
#### [new 016] Vi-SAFE: A Spatial-Temporal Framework for Efficient Violence Detection in Public Surveillance
- **分类: cs.CV; I.2.10; I.4.8**

- **简介: 该论文提出Vi-SAFE框架，用于公共监控中的暴力检测任务。针对小目标、复杂环境和实时分析的挑战，融合改进YOLOv8与TSN网络，提升检测精度与效率。实验表明其在RWF-2000数据集上准确率达0.88，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.13210v1](http://arxiv.org/pdf/2509.13210v1)**

> **作者:** Ligang Chang; Shengkai Xu; Liangchang Shen; Binhan Xu; Junqiao Wang; Tianyu Shi; Yanhui Du
>
> **摘要:** Violence detection in public surveillance is critical for public safety. This study addresses challenges such as small-scale targets, complex environments, and real-time temporal analysis. We propose Vi-SAFE, a spatial-temporal framework that integrates an enhanced YOLOv8 with a Temporal Segment Network (TSN) for video surveillance. The YOLOv8 model is optimized with GhostNetV3 as a lightweight backbone, an exponential moving average (EMA) attention mechanism, and pruning to reduce computational cost while maintaining accuracy. YOLOv8 and TSN are trained separately on pedestrian and violence datasets, where YOLOv8 extracts human regions and TSN performs binary classification of violent behavior. Experiments on the RWF-2000 dataset show that Vi-SAFE achieves an accuracy of 0.88, surpassing TSN alone (0.77) and outperforming existing methods in both accuracy and efficiency, demonstrating its effectiveness for public safety surveillance. Code is available at https://anonymous.4open.science/r/Vi-SAFE-3B42/README.md.
>
---
#### [new 017] Adaptive Sampling Scheduler
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种自适应采样调度器，用于一致性蒸馏框架，解决传统方法灵活性差的问题。通过动态选择时间步、优化采样路径及稳定生成技术，提升扩散模型的生成性能与适用性。**

- **链接: [http://arxiv.org/pdf/2509.12569v1](http://arxiv.org/pdf/2509.12569v1)**

> **作者:** Qi Wang; Shuliang Zhu; Jinjia Zhou
>
> **备注:** 10 pages, 10 figures,2 Tables, 18 Equations
>
> **摘要:** Consistent distillation methods have evolved into effective techniques that significantly accelerate the sampling process of diffusion models. Although existing methods have achieved remarkable results, the selection of target timesteps during distillation mainly relies on deterministic or stochastic strategies, which often require sampling schedulers to be designed specifically for different distillation processes. Moreover, this pattern severely limits flexibility, thereby restricting the full sampling potential of diffusion models in practical applications. To overcome these limitations, this paper proposes an adaptive sampling scheduler that is applicable to various consistency distillation frameworks. The scheduler introduces three innovative strategies: (i) dynamic target timestep selection, which adapts to different consistency distillation frameworks by selecting timesteps based on their computed importance; (ii) Optimized alternating sampling along the solution trajectory by guiding forward denoising and backward noise addition based on the proposed time step importance, enabling more effective exploration of the solution space to enhance generation performance; and (iii) Utilization of smoothing clipping and color balancing techniques to achieve stable and high-quality generation results at high guidance scales, thereby expanding the applicability of consistency distillation models in complex generation scenarios. We validated the effectiveness and flexibility of the adaptive sampling scheduler across various consistency distillation methods through comprehensive experimental evaluations. Experimental results consistently demonstrated significant improvements in generative performance, highlighting the strong adaptability achieved by our method.
>
---
#### [new 018] Enhancing Video Large Language Models with Structured Multi-Video Collaborative Reasoning (early version)
- **分类: cs.CV**

- **简介: 该论文属于视频语言模型任务，旨在解决单视频时空信息不全导致的推理错误问题。提出多视频协作框架，通过结构化图表示和融合模块提升模型推理能力，实验验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2509.13161v1](http://arxiv.org/pdf/2509.13161v1)**

> **作者:** Zhihao He; Tianyao He; Tieyuan Chen; Yun Xu; Huabin Liu; Chaofan Gan; Gui Zou; Weiyao Lin
>
> **摘要:** Despite the prosperity of the video language model, the current pursuit of comprehensive video reasoning is thwarted by the inherent spatio-temporal incompleteness within individual videos, resulting in hallucinations and inaccuracies. A promising solution is to augment the reasoning performance with multiple related videos. However, video tokens are numerous and contain redundant information, so directly feeding the relevant video data into a large language model to enhance responses could be counterproductive. To address this challenge, we propose a multi-video collaborative framework for video language models. For efficient and flexible video representation, we establish a Video Structuring Module to represent the video's knowledge as a spatio-temporal graph. Based on the structured video representation, we design the Graph Fusion Module to fuse the structured knowledge and valuable information from related videos into the augmented graph node tokens. Finally, we construct an elaborate multi-video structured prompt to integrate the graph, visual, and textual tokens as the input to the large language model. Extensive experiments substantiate the effectiveness of our framework, showcasing its potential as a promising avenue for advancing video language models.
>
---
#### [new 019] RU-Net for Automatic Characterization of TRISO Fuel Cross Sections
- **分类: cs.CV; cs.AI**

- **简介: 论文提出RU-Net用于自动分割TRISO燃料截面显微图像，解决人工分析效率低、主观性强的问题。通过对比多种CNN模型，验证RU-Net在IoU指标上表现最优，提升分析客观性与效率。**

- **链接: [http://arxiv.org/pdf/2509.12244v1](http://arxiv.org/pdf/2509.12244v1)**

> **作者:** Lu Cai; Fei Xu; Min Xian; Yalei Tang; Shoukun Sun; John Stempien
>
> **摘要:** During irradiation, phenomena such as kernel swelling and buffer densification may impact the performance of tristructural isotropic (TRISO) particle fuel. Post-irradiation microscopy is often used to identify these irradiation-induced morphologic changes. However, each fuel compact generally contains thousands of TRISO particles. Manually performing the work to get statistical information on these phenomena is cumbersome and subjective. To reduce the subjectivity inherent in that process and to accelerate data analysis, we used convolutional neural networks (CNNs) to automatically segment cross-sectional images of microscopic TRISO layers. CNNs are a class of machine-learning algorithms specifically designed for processing structured grid data. They have gained popularity in recent years due to their remarkable performance in various computer vision tasks, including image classification, object detection, and image segmentation. In this research, we generated a large irradiated TRISO layer dataset with more than 2,000 microscopic images of cross-sectional TRISO particles and the corresponding annotated images. Based on these annotated images, we used different CNNs to automatically segment different TRISO layers. These CNNs include RU-Net (developed in this study), as well as three existing architectures: U-Net, Residual Network (ResNet), and Attention U-Net. The preliminary results show that the model based on RU-Net performs best in terms of Intersection over Union (IoU). Using CNN models, we can expedite the analysis of TRISO particle cross sections, significantly reducing the manual labor involved and improving the objectivity of the segmentation results.
>
---
#### [new 020] MEJO: MLLM-Engaged Surgical Triplet Recognition via Inter- and Intra-Task Joint Optimization
- **分类: cs.CV**

- **简介: 论文提出MEJO框架，解决手术三元组识别中的跨任务和单任务优化冲突问题。通过S²D分解表示与CGL策略，提升模型在长尾数据下的性能，属于多任务学习在手术场景理解中的应用。**

- **链接: [http://arxiv.org/pdf/2509.12893v1](http://arxiv.org/pdf/2509.12893v1)**

> **作者:** Yiyi Zhang; Yuchen Yuan; Ying Zheng; Jialun Pei; Jinpeng Li; Zheng Li; Pheng-Ann Heng
>
> **摘要:** Surgical triplet recognition, which involves identifying instrument, verb, target, and their combinations, is a complex surgical scene understanding challenge plagued by long-tailed data distribution. The mainstream multi-task learning paradigm benefiting from cross-task collaborative promotion has shown promising performance in identifying triples, but two key challenges remain: 1) inter-task optimization conflicts caused by entangling task-generic and task-specific representations; 2) intra-task optimization conflicts due to class-imbalanced training data. To overcome these difficulties, we propose the MLLM-Engaged Joint Optimization (MEJO) framework that empowers both inter- and intra-task optimization for surgical triplet recognition. For inter-task optimization, we introduce the Shared-Specific-Disentangled (S$^2$D) learning scheme that decomposes representations into task-shared and task-specific components. To enhance task-shared representations, we construct a Multimodal Large Language Model (MLLM) powered probabilistic prompt pool to dynamically augment visual features with expert-level semantic cues. Additionally, comprehensive task-specific cues are modeled via distinct task prompts covering the temporal-spatial dimensions, effectively mitigating inter-task ambiguities. To tackle intra-task optimization conflicts, we develop a Coordinated Gradient Learning (CGL) strategy, which dissects and rebalances the positive-negative gradients originating from head and tail classes for more coordinated learning behaviors. Extensive experiments on the CholecT45 and CholecT50 datasets demonstrate the superiority of our proposed framework, validating its effectiveness in handling optimization conflicts.
>
---
#### [new 021] Time-step Mixup for Efficient Spiking Knowledge Transfer from Appearance to Event Domain
- **分类: cs.CV**

- **简介: 该论文提出时间步混合知识迁移（TMKT）方法，解决事件相机与脉冲神经网络间数据分布差异问题，通过跨模态混合RGB与DVS输入，提升脉冲图像分类性能。属于跨模态知识迁移任务。**

- **链接: [http://arxiv.org/pdf/2509.12959v1](http://arxiv.org/pdf/2509.12959v1)**

> **作者:** Yuqi Xie; Shuhan Ye; Chong Wang; Jiazhen Xu; Le Shen; Yuanbin Qian; Jiangbo Qian
>
> **摘要:** The integration of event cameras and spiking neural networks holds great promise for energy-efficient visual processing. However, the limited availability of event data and the sparse nature of DVS outputs pose challenges for effective training. Although some prior work has attempted to transfer semantic knowledge from RGB datasets to DVS, they often overlook the significant distribution gap between the two modalities. In this paper, we propose Time-step Mixup knowledge transfer (TMKT), a novel fine-grained mixing strategy that exploits the asynchronous nature of SNNs by interpolating RGB and DVS inputs at various time-steps. To enable label mixing in cross-modal scenarios, we further introduce modality-aware auxiliary learning objectives. These objectives support the time-step mixup process and enhance the model's ability to discriminate effectively across different modalities. Our approach enables smoother knowledge transfer, alleviates modality shift during training, and achieves superior performance in spiking image classification tasks. Extensive experiments demonstrate the effectiveness of our method across multiple datasets. The code will be released after the double-blind review process.
>
---
#### [new 022] Modeling the Multivariate Relationship with Contextualized Representations for Effective Human-Object Interaction Detection
- **分类: cs.CV**

- **简介: 该论文属于人-物交互检测任务，旨在解决复杂场景下交互关系建模不充分的问题。提出一种结合语义提示与视觉特征的上下文表示学习网络，通过引入工具等辅助实体，提升对依赖工具的交互（如“填充”）的识别能力。**

- **链接: [http://arxiv.org/pdf/2509.12784v1](http://arxiv.org/pdf/2509.12784v1)**

> **作者:** Zhehao Li; Yucheng Qian; Chong Wang; Yinghao Lu; Zhihao Yang; Jiafei Wu
>
> **摘要:** Human-Object Interaction (HOI) detection aims to simultaneously localize human-object pairs and recognize their interactions. While recent two-stage approaches have made significant progress, they still face challenges due to incomplete context modeling. In this work, we introduce a Contextualized Representation Learning Network that integrates both affordance-guided reasoning and contextual prompts with visual cues to better capture complex interactions. We enhance the conventional HOI detection framework by expanding it beyond simple human-object pairs to include multivariate relationships involving auxiliary entities like tools. Specifically, we explicitly model the functional role (affordance) of these auxiliary objects through triplet structures <human, tool, object>. This enables our model to identify tool-dependent interactions such as 'filling'. Furthermore, the learnable prompt is enriched with instance categories and subsequently integrated with contextual visual features using an attention mechanism. This process aligns language with image content at both global and regional levels. These contextualized representations equip the model with enriched relational cues for more reliable reasoning over complex, context-dependent interactions. Our proposed method demonstrates superior performance on both the HICO-Det and V-COCO datasets in most scenarios. Codes will be released upon acceptance.
>
---
#### [new 023] Lego-Edit: A General Image Editing Framework with Model-Level Bricks and MLLM Builder
- **分类: cs.CV**

- **简介: 该论文提出Lego-Edit，一种基于多模态大语言模型（MLLM）的通用图像编辑框架。旨在解决用户指令多样性导致的模型泛化能力不足问题，通过构建模型级工具集和三阶段强化学习方法，提升开放域指令处理能力，实现无需额外微调即可使用新工具的图像编辑。**

- **链接: [http://arxiv.org/pdf/2509.12883v1](http://arxiv.org/pdf/2509.12883v1)**

> **作者:** Qifei Jia; Yu Liu; Yajie Chai; Xintong Yao; Qiming Lu; Yasen Zhang; Runyu Shi; Ying Huang; Guoquan Zhang
>
> **摘要:** Instruction-based image editing has garnered significant attention due to its direct interaction with users. However, real-world user instructions are immensely diverse, and existing methods often fail to generalize effectively to instructions outside their training domain, limiting their practical application. To address this, we propose Lego-Edit, which leverages the generalization capability of Multi-modal Large Language Model (MLLM) to organize a suite of model-level editing tools to tackle this challenge. Lego-Edit incorporates two key designs: (1) a model-level toolkit comprising diverse models efficiently trained on limited data and several image manipulation functions, enabling fine-grained composition of editing actions by the MLLM; and (2) a three-stage progressive reinforcement learning approach that uses feedback on unannotated, open-domain instructions to train the MLLM, equipping it with generalized reasoning capabilities for handling real-world instructions. Experiments demonstrate that Lego-Edit achieves state-of-the-art performance on GEdit-Bench and ImgBench. It exhibits robust reasoning capabilities for open-domain instructions and can utilize newly introduced editing tools without additional fine-tuning. Code is available: https://github.com/xiaomi-research/lego-edit.
>
---
#### [new 024] Few to Big: Prototype Expansion Network via Diffusion Learner for Point Cloud Few-shot Semantic Segmentation
- **分类: cs.CV**

- **简介: 论文提出PENet框架，解决点云少样本语义分割中类内多样性和集间不一致问题。通过结合扩散模型生成特征，扩展原型表示能力，并引入双流学习与原型校准机制，提升分割性能。**

- **链接: [http://arxiv.org/pdf/2509.12878v1](http://arxiv.org/pdf/2509.12878v1)**

> **作者:** Qianguang Zhao; Dongli Wang; Yan Zhou; Jianxun Li; Richard Irampa
>
> **摘要:** Few-shot 3D point cloud semantic segmentation aims to segment novel categories using a minimal number of annotated support samples. While existing prototype-based methods have shown promise, they are constrained by two critical challenges: (1) Intra-class Diversity, where a prototype's limited representational capacity fails to cover a class's full variations, and (2) Inter-set Inconsistency, where prototypes derived from the support set are misaligned with the query feature space. Motivated by the powerful generative capability of diffusion model, we re-purpose its pre-trained conditional encoder to provide a novel source of generalizable features for expanding the prototype's representational range. Under this setup, we introduce the Prototype Expansion Network (PENet), a framework that constructs big-capacity prototypes from two complementary feature sources. PENet employs a dual-stream learner architecture: it retains a conventional fully supervised Intrinsic Learner (IL) to distill representative features, while introducing a novel Diffusion Learner (DL) to provide rich generalizable features. The resulting dual prototypes are then processed by a Prototype Assimilation Module (PAM), which adopts a novel push-pull cross-guidance attention block to iteratively align the prototypes with the query space. Furthermore, a Prototype Calibration Mechanism (PCM) regularizes the final big capacity prototype to prevent semantic drift. Extensive experiments on the S3DIS and ScanNet datasets demonstrate that PENet significantly outperforms state-of-the-art methods across various few-shot settings.
>
---
#### [new 025] GraphDerm: Fusing Imaging, Physical Scale, and Metadata in a Population-Graph Classifier for Dermoscopic Lesions
- **分类: cs.CV; cs.AI**

- **简介: 论文提出GraphDerm，融合皮肤镜图像、物理尺度和元数据构建图神经网络，用于多类别皮肤病分类。解决图像仅模型忽略元数据和尺度信息的问题，通过图结构提升分类性能。**

- **链接: [http://arxiv.org/pdf/2509.12277v1](http://arxiv.org/pdf/2509.12277v1)**

> **作者:** Mehdi Yousefzadeh; Parsa Esfahanian; Sara Rashidifar; Hossein Salahshoor Gavalan; Negar Sadat Rafiee Tabatabaee; Saeid Gorgin; Dara Rahmati; Maryam Daneshpazhooh
>
> **摘要:** Introduction. Dermoscopy aids melanoma triage, yet image-only AI often ignores patient metadata (age, sex, site) and the physical scale needed for geometric analysis. We present GraphDerm, a population-graph framework that fuses imaging, millimeter-scale calibration, and metadata for multiclass dermoscopic classification, to the best of our knowledge the first ISIC-scale application of GNNs to dermoscopy. Methods. We curate ISIC 2018/2019, synthesize ruler-embedded images with exact masks, and train U-Nets (SE-ResNet-18) for lesion and ruler segmentation. Pixels-per-millimeter are regressed from the ruler-mask two-point correlation via a lightweight 1D-CNN. From lesion masks we compute real-scale descriptors (area, perimeter, radius of gyration). Node features use EfficientNet-B3; edges encode metadata/geometry similarity (fully weighted or thresholded). A spectral GNN performs semi-supervised node classification; an image-only ANN is the baseline. Results. Ruler and lesion segmentation reach Dice 0.904 and 0.908; scale regression attains MAE 1.5 px (RMSE 6.6). The graph attains AUC 0.9812, with a thresholded variant using about 25% of edges preserving AUC 0.9788 (vs. 0.9440 for the image-only baseline); per-class AUCs typically fall in the 0.97-0.99 range. Conclusion. Unifying calibrated scale, lesion geometry, and metadata in a population graph yields substantial gains over image-only pipelines on ISIC-2019. Sparser graphs retain near-optimal accuracy, suggesting efficient deployment. Scale-aware, graph-based AI is a promising direction for dermoscopic decision support; future work will refine learned edge semantics and evaluate on broader curated benchmarks.
>
---
#### [new 026] MMMS: Multi-Modal Multi-Surface Interactive Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出MMMS方法，解决多表面图像的交互式分割问题。通过融合多模态输入和用户点击信息，提升分割精度。设计网络架构满足黑盒RGB主干和高效交互条件，并引入新评估指标验证效果。**

- **链接: [http://arxiv.org/pdf/2509.12963v1](http://arxiv.org/pdf/2509.12963v1)**

> **作者:** Robin Schön; Julian Lorenz; Katja Ludwig; Daniel Kienzle; Rainer Lienhart
>
> **备注:** 19 pages, 11 figures, 10 pages
>
> **摘要:** In this paper, we present a method to interactively create segmentation masks on the basis of user clicks. We pay particular attention to the segmentation of multiple surfaces that are simultaneously present in the same image. Since these surfaces may be heavily entangled and adjacent, we also present a novel extended evaluation metric that accounts for the challenges of this scenario. Additionally, the presented method is able to use multi-modal inputs to facilitate the segmentation task. At the center of this method is a network architecture which takes as input an RGB image, a number of non-RGB modalities, an erroneous mask, and encoded clicks. Based on this input, the network predicts an improved segmentation mask. We design our architecture such that it adheres to two conditions: (1) The RGB backbone is only available as a black-box. (2) To reduce the response time, we want our model to integrate the interaction-specific information after the image feature extraction and the multi-modal fusion. We refer to the overall task as Multi-Modal Multi-Surface interactive segmentation (MMMS). We are able to show the effectiveness of our multi-modal fusion strategy. Using additional modalities, our system reduces the NoC@90 by up to 1.28 clicks per surface on average on DeLiVER and up to 1.19 on MFNet. On top of this, we are able to show that our RGB-only baseline achieves competitive, and in some cases even superior performance when tested in a classical, single-mask interactive segmentation scenario.
>
---
#### [new 027] DS@GT AnimalCLEF: Triplet Learning over ViT Manifolds with Nearest Neighbor Classification for Animal Re-identification
- **分类: cs.CV**

- **简介: 论文针对动物重识别任务，研究不同主干模型对重识别性能的影响。通过对比通用模型与领域特定模型，提出使用三元组学习和K近邻分类方法，强调领域特定预训练在小数据场景下的重要性。**

- **链接: [http://arxiv.org/pdf/2509.12353v1](http://arxiv.org/pdf/2509.12353v1)**

> **作者:** Anthony Miyaguchi; Chandrasekaran Maruthaiyannan; Charles R. Clark
>
> **备注:** CLEF 2025 working notes
>
> **摘要:** This paper details the DS@GT team's entry for the AnimalCLEF 2025 re-identification challenge. Our key finding is that the effectiveness of post-hoc metric learning is highly contingent on the initial quality and domain-specificity of the backbone embeddings. We compare a general-purpose model (DINOv2) with a domain-specific model (MegaDescriptor) as a backbone. A K-Nearest Neighbor classifier with robust thresholding then identifies known individuals or flags new ones. While a triplet-learning projection head improved the performance of the specialized MegaDescriptor model by 0.13 points, it yielded minimal gains (0.03) for the general-purpose DINOv2 on averaged BAKS and BAUS. We demonstrate that the general-purpose manifold is more difficult to reshape for fine-grained tasks, as evidenced by stagnant validation loss and qualitative visualizations. This work highlights the critical limitations of refining general-purpose features for specialized, limited-data re-ID tasks and underscores the importance of domain-specific pre-training. The implementation for this work is publicly available at github.com/dsgt-arc/animalclef-2025.
>
---
#### [new 028] Towards Foundational Models for Single-Chip Radar
- **分类: cs.CV**

- **简介: 该论文面向单芯片毫米波雷达，旨在提升其角度分辨率。通过构建最大数据集并训练基础模型GRT，实现高质量3D占用和语义分割，解决传统方法依赖小数据集、任务特定模型的问题，推动通用雷达模型发展。**

- **链接: [http://arxiv.org/pdf/2509.12482v1](http://arxiv.org/pdf/2509.12482v1)**

> **作者:** Tianshu Huang; Akarsh Prabhakara; Chuhan Chen; Jay Karhade; Deva Ramanan; Matthew O'Toole; Anthony Rowe
>
> **备注:** To appear in ICCV 2025
>
> **摘要:** mmWave radars are compact, inexpensive, and durable sensors that are robust to occlusions and work regardless of environmental conditions, such as weather and darkness. However, this comes at the cost of poor angular resolution, especially for inexpensive single-chip radars, which are typically used in automotive and indoor sensing applications. Although many have proposed learning-based methods to mitigate this weakness, no standardized foundational models or large datasets for the mmWave radar have emerged, and practitioners have largely trained task-specific models from scratch using relatively small datasets. In this paper, we collect (to our knowledge) the largest available raw radar dataset with 1M samples (29 hours) and train a foundational model for 4D single-chip radar, which can predict 3D occupancy and semantic segmentation with quality that is typically only possible with much higher resolution sensors. We demonstrate that our Generalizable Radar Transformer (GRT) generalizes across diverse settings, can be fine-tuned for different tasks, and shows logarithmic data scaling of 20\% per $10\times$ data. We also run extensive ablations on common design decisions, and find that using raw radar data significantly outperforms widely-used lossy representations, equivalent to a $10\times$ increase in training data. Finally, we roughly estimate that $\approx$100M samples (3000 hours) of data are required to fully exploit the potential of GRT.
>
---
#### [new 029] SPGen: Spherical Projection as Consistent and Flexible Representation for Single Image 3D Shape Generation
- **分类: cs.CV**

- **简介: 该论文提出SPGen模型，用于单视角3D形状生成。通过将几何信息投影到球面上并展开为多层2D表示，解决多视角不一致、内部结构复杂等问题，提升生成质量与效率。**

- **链接: [http://arxiv.org/pdf/2509.12721v1](http://arxiv.org/pdf/2509.12721v1)**

> **作者:** Jingdong Zhang; Weikai Chen; Yuan Liu; Jionghao Wang; Zhengming Yu; Zhuowen Shen; Bo Yang; Wenping Wang; Xin Li
>
> **摘要:** Existing single-view 3D generative models typically adopt multiview diffusion priors to reconstruct object surfaces, yet they remain prone to inter-view inconsistencies and are unable to faithfully represent complex internal structure or nontrivial topologies. In particular, we encode geometry information by projecting it onto a bounding sphere and unwrapping it into a compact and structural multi-layer 2D Spherical Projection (SP) representation. Operating solely in the image domain, SPGen offers three key advantages simultaneously: (1) Consistency. The injective SP mapping encodes surface geometry with a single viewpoint which naturally eliminates view inconsistency and ambiguity; (2) Flexibility. Multi-layer SP maps represent nested internal structures and support direct lifting to watertight or open 3D surfaces; (3) Efficiency. The image-domain formulation allows the direct inheritance of powerful 2D diffusion priors and enables efficient finetuning with limited computational resources. Extensive experiments demonstrate that SPGen significantly outperforms existing baselines in geometric quality and computational efficiency.
>
---
#### [new 030] Uncertainty-Aware Hourly Air Temperature Mapping at 2 km Resolution via Physics-Guided Deep Learning
- **分类: cs.CV; cs.LG**

- **简介: 论文提出一种基于物理引导的深度学习方法，用于生成美国本土2公里分辨率的每小时气温地图。该方法结合卫星和地面站数据，通过神经网络重建云遮挡的表面温度，并预测空气温度，提升高时空分辨率气温监测的准确性。**

- **链接: [http://arxiv.org/pdf/2509.12329v1](http://arxiv.org/pdf/2509.12329v1)**

> **作者:** Shengjie Kris Liu; Siqin Wang; Lu Zhang
>
> **摘要:** Near-surface air temperature is a key physical property of the Earth's surface. Although weather stations offer continuous monitoring and satellites provide broad spatial coverage, no single data source offers seamless data in a spatiotemporal fashion. Here, we propose a data-driven, physics-guided deep learning approach to generate hourly air temperature data at 2 km resolution over the contiguous United States. The approach, called Amplifier Air-Transformer, first reconstructs GOES-16 surface temperature data obscured by clouds. It does so through a neural network encoded with the annual temperature cycle, incorporating a linear term to amplify ERA5 temperature values at finer scales and convolutional layers to capture spatiotemporal variations. Then, another neural network transforms the reconstructed surface temperature into air temperature by leveraging its latent relationship with key Earth surface properties. The approach is further enhanced with predictive uncertainty estimation through deep ensemble learning to improve reliability. The proposed approach is built and tested on 77.7 billion surface temperature pixels and 155 million air temperature records from weather stations across the contiguous United States (2018-2024), achieving hourly air temperature mapping accuracy of 1.93 C in station-based validation. The proposed approach streamlines surface temperature reconstruction and air temperature prediction, and it can be extended to other satellite sources for seamless air temperature monitoring at high spatiotemporal resolution. The generated data of this study can be downloaded at https://doi.org/10.5281/zenodo.15252812, and the project webpage can be found at https://skrisliu.com/HourlyAirTemp2kmUSA/.
>
---
#### [new 031] A Modern Look at Simplicity Bias in Image Classification Tasks
- **分类: cs.CV; cs.AI**

- **简介: 论文研究CLIP模型中的简单性偏差（SB）与其图像分类任务性能的关系。旨在探讨SB对不同任务的影响，提出更精细的SB度量方法，并验证其有效性，揭示SB与模型表现的相关性。属于图像分类与模型泛化能力研究任务。**

- **链接: [http://arxiv.org/pdf/2509.12265v1](http://arxiv.org/pdf/2509.12265v1)**

> **作者:** Xiaoguang Chang; Teng Wang; Changyin Sun
>
> **摘要:** The simplicity Bias (SB) of neural networks, i.e.\ their tendency to represent simple functions, is a key factor in their generalization capabilities. Recent studies show that an excessive SB may harm performance on complex tasks, and the need for this bias varies across tasks. Many of these studies focus on simple models or synthetic tasks. It remains challenging to measure the SB in large models and little is known about the relevance of the SB to various image classification tasks. In this paper, we investigate the relationship between the SB in CLIP models and their performance across image classification tasks. First, we theoretically analyze the potential limitation of existing measures of complexity that have been used to characterize small models. To address this, we propose a frequency-aware measure capturing finer-grained SB differences. We validate this measure on CLIP models subjected to two recent SB-modulation methods, demonstrating that it is more informative and consistent than previous measures. Second, we examine the relation between the SB of those models and their performance across a range of image classification tasks, including zero-shot and fine-tuning settings. These experiments reveal a range of behaviors. For example, a stronger SB correlates with a better performance on OOD generalization than on adversarial robustness. These results highlight the benefits of aligning a model's inductive biases with the characteristics of the target task.
>
---
#### [new 032] Instance-Guided Class Activation Mapping for Weakly Supervised Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于弱监督语义分割任务，旨在仅用图像级标注训练分割模型。提出IG-CAM方法，通过实例引导和边界增强等创新，提升定位精度与边界清晰度，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.12496v1](http://arxiv.org/pdf/2509.12496v1)**

> **作者:** Ali Torabi; Sanjog Gaihre; MD Mahbubur Rahman; Yaqoob Majeed
>
> **摘要:** Weakly Supervised Semantic Segmentation (WSSS) addresses the challenge of training segmentation models using only image-level annotations, eliminating the need for expensive pixel-level labeling. While existing methods struggle with precise object boundary localization and often focus only on the most discriminative regions, we propose IG-CAM (Instance-Guided Class Activation Mapping), a novel approach that leverages instance-level cues and influence functions to generate high-quality, boundary-aware localization maps. Our method introduces three key innovations: (1) Instance-Guided Refinement that uses ground truth segmentation masks to guide CAM generation, ensuring complete object coverage rather than just discriminative parts; (2) Influence Function Integration that captures the relationship between training samples and model predictions, leading to more robust feature representations; and (3) Multi-Scale Boundary Enhancement that employs progressive refinement strategies to achieve sharp, precise object boundaries. IG-CAM achieves state-of-the-art performance on the PASCAL VOC 2012 dataset with an mIoU of 82.3% before post-processing, which further improves to 86.6% after applying Conditional Random Field (CRF) refinement, significantly outperforming previous WSSS methods. Our approach demonstrates superior localization accuracy, with complete object coverage and precise boundary delineation, while maintaining computational efficiency. Extensive ablation studies validate the contribution of each component, and qualitative comparisons across 600 diverse images showcase the method's robustness and generalization capability. The results establish IG-CAM as a new benchmark for weakly supervised semantic segmentation, offering a practical solution for scenarios where pixel-level annotations are unavailable or prohibitively expensive.
>
---
#### [new 033] Using KL-Divergence to Focus Frequency Information in Low-Light Image Enhancement
- **分类: cs.CV**

- **简介: 该论文属于低光图像增强任务，旨在解决传统方法在频率信息建模中的全局信息丢失问题。提出LLFDisc网络，结合交叉注意力与门控机制，并引入KL散度损失以优化频率域信息对齐，提升增强效果。**

- **链接: [http://arxiv.org/pdf/2509.13083v1](http://arxiv.org/pdf/2509.13083v1)**

> **作者:** Yan Xingyang; Huang Xiaohong; Zhang Zhao; You Tian; Xu Ziheng
>
> **摘要:** In the Fourier domain, luminance information is primarily encoded in the amplitude spectrum, while spatial structures are captured in the phase components. The traditional Fourier Frequency information fitting employs pixel-wise loss functions, which tend to focus excessively on local information and may lead to global information loss. In this paper, we present LLFDisc, a U-shaped deep enhancement network that integrates cross-attention and gating mechanisms tailored for frequency-aware enhancement. We propose a novel distribution-aware loss that directly fits the Fourier-domain information and minimizes their divergence using a closed-form KL-Divergence objective. This enables the model to align Fourier-domain information more robustly than with conventional MSE-based losses. Furthermore, we enhance the perceptual loss based on VGG by embedding KL-Divergence on extracted deep features, enabling better structural fidelity. Extensive experiments across multiple benchmarks demonstrate that LLFDisc achieves state-of-the-art performance in both qualitative and quantitative evaluations. Our code will be released at: https://github.com/YanXY000/LLFDisc
>
---
#### [new 034] Explicit Multimodal Graph Modeling for Human-Object Interaction Detection
- **分类: cs.CV**

- **简介: 该论文属于人-物交互检测任务，旨在解决Transformer方法未能显式建模交互关系的问题。提出MGNM模型，利用图网络显式建模多模态特征，提升检测性能，在HICO-DET和V-COCO数据集上取得SOTA结果。**

- **链接: [http://arxiv.org/pdf/2509.12554v1](http://arxiv.org/pdf/2509.12554v1)**

> **作者:** Wenxuan Ji; Haichao Shi; Xiao-Yu zhang
>
> **摘要:** Transformer-based methods have recently become the prevailing approach for Human-Object Interaction (HOI) detection. However, the Transformer architecture does not explicitly model the relational structures inherent in HOI detection, which impedes the recognition of interactions. In contrast, Graph Neural Networks (GNNs) are inherently better suited for this task, as they explicitly model the relationships between human-object pairs. Therefore, in this paper, we propose \textbf{M}ultimodal \textbf{G}raph \textbf{N}etwork \textbf{M}odeling (MGNM) that leverages GNN-based relational structures to enhance HOI detection. Specifically, we design a multimodal graph network framework that explicitly models the HOI task in a four-stage graph structure. Furthermore, we introduce a multi-level feature interaction mechanism within our graph network. This mechanism leverages multi-level vision and language features to enhance information propagation across human-object pairs. Consequently, our proposed MGNM achieves state-of-the-art performance on two widely used benchmarks: HICO-DET and V-COCO. Moreover, when integrated with a more advanced object detector, our method demonstrates a significant performance gain and maintains an effective balance between rare and non-rare classes.
>
---
#### [new 035] EvoEmpirBench: Dynamic Spatial Reasoning with Agent-ExpVer
- **分类: cs.CV**

- **简介: 该论文提出EvoEmpirBench，解决动态空间推理与长期记忆问题。设计两个动态基准测试，评估模型在部分可观测环境中的适应性规划能力，并引入基于主观体验的记忆机制，揭示主流模型的局限性。**

- **链接: [http://arxiv.org/pdf/2509.12718v1](http://arxiv.org/pdf/2509.12718v1)**

> **作者:** Pukun Zhao; Longxiang Wang; Miaowei Wang; Chen Chen; Fanqing Zhou; Haojian Huang
>
> **备注:** Ongoing Work, 29 pages, 3 figures, 7 tables
>
> **摘要:** Most existing spatial reasoning benchmarks focus on static or globally observable environments, failing to capture the challenges of long-horizon reasoning and memory utilization under partial observability and dynamic changes. We introduce two dynamic spatial benchmarks, locally observable maze navigation and match-2 elimination that systematically evaluate models' abilities in spatial understanding and adaptive planning when local perception, environment feedback, and global objectives are tightly coupled. Each action triggers structural changes in the environment, requiring continuous update of cognition and strategy. We further propose a subjective experience-based memory mechanism for cross-task experience transfer and validation. Experiments show that our benchmarks reveal key limitations of mainstream models in dynamic spatial reasoning and long-term memory, providing a comprehensive platform for future methodological advances. Our code and data are available at https://anonymous.4open.science/r/EvoEmpirBench-143C/.
>
---
#### [new 036] Dual-Stage Reweighted MoE for Long-Tailed Egocentric Mistake Detection
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于错误检测任务，旨在从第一人称视频中识别用户操作失误。提出DR-MoE框架，通过双阶段重加权专家模块提升对罕见和模糊错误的识别能力。**

- **链接: [http://arxiv.org/pdf/2509.12990v1](http://arxiv.org/pdf/2509.12990v1)**

> **作者:** Boyu Han; Qianqian Xu; Shilong Bao; Zhiyong Yang; Sicong Li; Qingming Huang
>
> **摘要:** In this report, we address the problem of determining whether a user performs an action incorrectly from egocentric video data. To handle the challenges posed by subtle and infrequent mistakes, we propose a Dual-Stage Reweighted Mixture-of-Experts (DR-MoE) framework. In the first stage, features are extracted using a frozen ViViT model and a LoRA-tuned ViViT model, which are combined through a feature-level expert module. In the second stage, three classifiers are trained with different objectives: reweighted cross-entropy to mitigate class imbalance, AUC loss to improve ranking under skewed distributions, and label-aware loss with sharpness-aware minimization to enhance calibration and generalization. Their predictions are fused using a classification-level expert module. The proposed method achieves strong performance, particularly in identifying rare and ambiguous mistake instances. The code is available at https://github.com/boyuh/DR-MoE.
>
---
#### [new 037] Modular, On-Site Solutions with Lightweight Anomaly Detection for Sustainable Nutrient Management in Agriculture
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种轻量级异常检测模块，用于农业中可持续养分管理。通过多光谱成像和两种状态估计方法（ViT与RF），实现高效实时监测，解决传统方法计算复杂、无法实时优化的问题。**

- **链接: [http://arxiv.org/pdf/2509.12247v1](http://arxiv.org/pdf/2509.12247v1)**

> **作者:** Abigail R. Cohen; Yuming Sun; Zhihao Qin; Harsh S. Muriki; Zihao Xiao; Yeonju Lee; Matthew Housley; Andrew F. Sharkey; Rhuanito S. Ferrarezi; Jing Li; Lu Gan; Yongsheng Chen
>
> **摘要:** Efficient nutrient management is critical for crop growth and sustainable resource consumption (e.g., nitrogen, energy). Current approaches require lengthy analyses, preventing real-time optimization; similarly, imaging facilitates rapid phenotyping but can be computationally intensive, preventing deployment under resource constraints. This study proposes a flexible, tiered pipeline for anomaly detection and status estimation (fresh weight, dry mass, and tissue nutrients), including a comprehensive energy analysis of approaches that span the efficiency-accuracy spectrum. Using a nutrient depletion experiment with three treatments (T1-100%, T2-50%, and T3-25% fertilizer strength) and multispectral imaging (MSI), we developed a hierarchical pipeline using an autoencoder (AE) for early warning. Further, we compared two status estimation modules of different complexity for more detailed analysis: vegetation index (VI) features with machine learning (Random Forest, RF) and raw whole-image deep learning (Vision Transformer, ViT). Results demonstrated high-efficiency anomaly detection (73% net detection of T3 samples 9 days after transplanting) at substantially lower energy than embodied energy in wasted nitrogen. The state estimation modules show trade-offs, with ViT outperforming RF on phosphorus and calcium estimation (R2 0.61 vs. 0.58, 0.48 vs. 0.35) at higher energy cost. With our modular pipeline, this work opens opportunities for edge diagnostics and practical opportunities for agricultural sustainability.
>
---
#### [new 038] Advancing Real-World Parking Slot Detection with Large-Scale Dataset and Semi-Supervised Baseline
- **分类: cs.CV**

- **简介: 论文提出一种半监督方法SS-PSD用于提升停车槽检测性能，并构建了大规模数据集CRPS-D。任务为停车槽检测，解决现有数据集规模小、噪声少及标注成本高的问题。**

- **链接: [http://arxiv.org/pdf/2509.13133v1](http://arxiv.org/pdf/2509.13133v1)**

> **作者:** Zhihao Zhang; Chunyu Lin; Lang Nie; Jiyuan Wang; Yao Zhao
>
> **备注:** IEEE Transactions on Intelligent Transportation Systems (T-ITS)
>
> **摘要:** As automatic parking systems evolve, the accurate detection of parking slots has become increasingly critical. This study focuses on parking slot detection using surround-view cameras, which offer a comprehensive bird's-eye view of the parking environment. However, the current datasets are limited in scale, and the scenes they contain are seldom disrupted by real-world noise (e.g., light, occlusion, etc.). Moreover, manual data annotation is prone to errors and omissions due to the complexity of real-world conditions, significantly increasing the cost of annotating large-scale datasets. To address these issues, we first construct a large-scale parking slot detection dataset (named CRPS-D), which includes various lighting distributions, diverse weather conditions, and challenging parking slot variants. Compared with existing datasets, the proposed dataset boasts the largest data scale and consists of a higher density of parking slots, particularly featuring more slanted parking slots. Additionally, we develop a semi-supervised baseline for parking slot detection, termed SS-PSD, to further improve performance by exploiting unlabeled data. To our knowledge, this is the first semi-supervised approach in parking slot detection, which is built on the teacher-student model with confidence-guided mask consistency and adaptive feature perturbation. Experimental results demonstrate the superiority of SS-PSD over the existing state-of-the-art (SoTA) solutions on both the proposed dataset and the existing dataset. Particularly, the more unlabeled data there is, the more significant the gains brought by our semi-supervised scheme. The relevant source codes and the dataset have been made publicly available at https://github.com/zzh362/CRPS-D.
>
---
#### [new 039] A Novel Compression Framework for YOLOv8: Achiev-ing Real-Time Aerial Object Detection on Edge Devices via Structured Pruning and Channel-Wise Distillation
- **分类: cs.CV; 68T07; I.4.8**

- **简介: 该论文提出一种YOLOv8压缩框架，用于在边缘设备上实现实时航空目标检测。通过结构化剪枝和通道级知识蒸馏，显著减少模型参数和计算量，同时保持较高检测精度，提升推理速度，适用于资源受限场景。**

- **链接: [http://arxiv.org/pdf/2509.12918v1](http://arxiv.org/pdf/2509.12918v1)**

> **作者:** Melika Sabaghian; Mohammad Ali Keyvanrad; Seyyedeh Mahila Moghadami
>
> **备注:** 28 pages, 11 figures
>
> **摘要:** Efficient deployment of deep learning models for aerial object detection on resource-constrained devices requires significant compression without com-promising performance. In this study, we propose a novel three-stage compression pipeline for the YOLOv8 object detection model, integrating sparsity-aware training, structured channel pruning, and Channel-Wise Knowledge Distillation (CWD). First, sparsity-aware training introduces dynamic sparsity during model optimization, effectively balancing parameter reduction and detection accuracy. Second, we apply structured channel pruning by leveraging batch normalization scaling factors to eliminate redundant channels, significantly reducing model size and computational complexity. Finally, to mitigate the accuracy drop caused by pruning, we employ CWD to transfer knowledge from the original model, using an adjustable temperature and loss weighting scheme tailored for small and medium object detection. Extensive experiments on the VisDrone dataset demonstrate the effectiveness of our approach across multiple YOLOv8 variants. For YOLOv8m, our method reduces model parameters from 25.85M to 6.85M (a 73.51% reduction), FLOPs from 49.6G to 13.3G, and MACs from 101G to 34.5G, while reducing AP50 by only 2.7%. The resulting compressed model achieves 47.9 AP50 and boosts inference speed from 26 FPS (YOLOv8m baseline) to 45 FPS, enabling real-time deployment on edge devices. We further apply TensorRT as a lightweight optimization step. While this introduces a minor drop in AP50 (from 47.9 to 47.6), it significantly improves inference speed from 45 to 68 FPS, demonstrating the practicality of our approach for high-throughput, re-source-constrained scenarios.
>
---
#### [new 040] Curriculum Multi-Task Self-Supervision Improves Lightweight Architectures for Onboard Satellite Hyperspectral Image Segmentation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出CMTSSL框架，用于轻量级卫星高光谱图像分割。解决高维数据与传输限制问题，通过课程多任务自监督学习提升模型效率与性能。**

- **链接: [http://arxiv.org/pdf/2509.13229v1](http://arxiv.org/pdf/2509.13229v1)**

> **作者:** Hugo Carlesso; Josiane Mothe; Radu Tudor Ionescu
>
> **摘要:** Hyperspectral imaging (HSI) captures detailed spectral signatures across hundreds of contiguous bands per pixel, being indispensable for remote sensing applications such as land-cover classification, change detection, and environmental monitoring. Due to the high dimensionality of HSI data and the slow rate of data transfer in satellite-based systems, compact and efficient models are required to support onboard processing and minimize the transmission of redundant or low-value data, e.g. cloud-covered areas. To this end, we introduce a novel curriculum multi-task self-supervised learning (CMTSSL) framework designed for lightweight architectures for HSI analysis. CMTSSL integrates masked image modeling with decoupled spatial and spectral jigsaw puzzle solving, guided by a curriculum learning strategy that progressively increases data complexity during self-supervision. This enables the encoder to jointly capture fine-grained spectral continuity, spatial structure, and global semantic features. Unlike prior dual-task SSL methods, CMTSSL simultaneously addresses spatial and spectral reasoning within a unified and computationally efficient design, being particularly suitable for training lightweight models for onboard satellite deployment. We validate our approach on four public benchmark datasets, demonstrating consistent gains in downstream segmentation tasks, using architectures that are over 16,000x lighter than some state-of-the-art models. These results highlight the potential of CMTSSL in generalizable representation learning with lightweight architectures for real-world HSI applications. Our code is publicly available at https://github.com/hugocarlesso/CMTSSL.
>
---
#### [new 041] Artist-Created Mesh Generation from Raw Observation
- **分类: cs.CV**

- **简介: 论文提出一种端到端框架，从噪声或不完整的点云生成艺术家风格网格。任务为3D网格生成，解决现实传感器输入不完美问题，核心工作是将点云修复转化为2D补全任务，提升生成质量与效率。**

- **链接: [http://arxiv.org/pdf/2509.12501v1](http://arxiv.org/pdf/2509.12501v1)**

> **作者:** Yao He; Youngjoong Kwon; Wenxiao Cai; Ehsan Adeli
>
> **摘要:** We present an end-to-end framework for generating artist-style meshes from noisy or incomplete point clouds, such as those captured by real-world sensors like LiDAR or mobile RGB-D cameras. Artist-created meshes are crucial for commercial graphics pipelines due to their compatibility with animation and texturing tools and their efficiency in rendering. However, existing approaches often assume clean, complete inputs or rely on complex multi-stage pipelines, limiting their applicability in real-world scenarios. To address this, we propose an end-to-end method that refines the input point cloud and directly produces high-quality, artist-style meshes. At the core of our approach is a novel reformulation of 3D point cloud refinement as a 2D inpainting task, enabling the use of powerful generative models. Preliminary results on the ShapeNet dataset demonstrate the promise of our framework in producing clean, complete meshes.
>
---
#### [new 042] Maps for Autonomous Driving: Full-process Survey and Frontiers
- **分类: cs.CV**

- **简介: 该论文综述自动驾驶地图的全流程发展，分为HD、Lite和Implicit三阶段，分析各阶段技术挑战与解决方案，并探讨前沿研究在端到端自动驾驶框架中的应用。属于自动驾驶地图技术调研任务。**

- **链接: [http://arxiv.org/pdf/2509.12632v1](http://arxiv.org/pdf/2509.12632v1)**

> **作者:** Pengxin Chen; Zhipeng Luo; Xiaoqi Jiang; Zhangcai Yin; Jonathan Li
>
> **摘要:** Maps have always been an essential component of autonomous driving. With the advancement of autonomous driving technology, both the representation and production process of maps have evolved substantially. The article categorizes the evolution of maps into three stages: High-Definition (HD) maps, Lightweight (Lite) maps, and Implicit maps. For each stage, we provide a comprehensive review of the map production workflow, with highlighting technical challenges involved and summarizing relevant solutions proposed by the academic community. Furthermore, we discuss cutting-edge research advances in map representations and explore how these innovations can be integrated into end-to-end autonomous driving frameworks.
>
---
#### [new 043] T-SiamTPN: Temporal Siamese Transformer Pyramid Networks for Robust and Efficient UAV Tracking
- **分类: cs.CV**

- **简介: 该论文提出T-SiamTPN，用于无人机航拍目标跟踪任务，解决长期跟踪中因时间依赖缺失导致的鲁棒性不足问题。通过引入时序建模与特征融合，提升跟踪精度与效率，在Jetson Nano上实现实时运行。**

- **链接: [http://arxiv.org/pdf/2509.12913v1](http://arxiv.org/pdf/2509.12913v1)**

> **作者:** Hojat Ardi; Amir Jahanshahi; Ali Diba
>
> **摘要:** Aerial object tracking remains a challenging task due to scale variations, dynamic backgrounds, clutter, and frequent occlusions. While most existing trackers emphasize spatial cues, they often overlook temporal dependencies, resulting in limited robustness in long-term tracking and under occlusion. Furthermore, correlation-based Siamese trackers are inherently constrained by the linear nature of correlation operations, making them ineffective against complex, non-linear appearance changes. To address these limitations, we introduce T-SiamTPN, a temporal-aware Siamese tracking framework that extends the SiamTPN architecture with explicit temporal modeling. Our approach incorporates temporal feature fusion and attention-based interactions, strengthening temporal consistency and enabling richer feature representations. These enhancements yield significant improvements over the baseline and achieve performance competitive with state-of-the-art trackers. Crucially, despite the added temporal modules, T-SiamTPN preserves computational efficiency. Deployed on the resource-constrained Jetson Nano, the tracker runs in real time at 7.1 FPS, demonstrating its suitability for real-world embedded applications without notable runtime overhead. Experimental results highlight substantial gains: compared to the baseline, T-SiamTPN improves success rate by 13.7% and precision by 14.7%. These findings underscore the importance of temporal modeling in Siamese tracking frameworks and establish T-SiamTPN as a strong and efficient solution for aerial object tracking. Code is available at: https://github.com/to/be/released
>
---
#### [new 044] WHU-STree: A Multi-modal Benchmark Dataset for Street Tree Inventory
- **分类: cs.CV**

- **简介: 该论文提出WHU-STree，一个多模态城市行道树数据集，用于解决传统人工调查效率低的问题。数据集包含点云和图像，支持多种任务，推动自动化行道树盘点技术发展。**

- **链接: [http://arxiv.org/pdf/2509.13172v1](http://arxiv.org/pdf/2509.13172v1)**

> **作者:** Ruifei Ding; Zhe Chen; Wen Fan; Chen Long; Huijuan Xiao; Yelu Zeng; Zhen Dong; Bisheng Yang
>
> **摘要:** Street trees are vital to urban livability, providing ecological and social benefits. Establishing a detailed, accurate, and dynamically updated street tree inventory has become essential for optimizing these multifunctional assets within space-constrained urban environments. Given that traditional field surveys are time-consuming and labor-intensive, automated surveys utilizing Mobile Mapping Systems (MMS) offer a more efficient solution. However, existing MMS-acquired tree datasets are limited by small-scale scene, limited annotation, or single modality, restricting their utility for comprehensive analysis. To address these limitations, we introduce WHU-STree, a cross-city, richly annotated, and multi-modal urban street tree dataset. Collected across two distinct cities, WHU-STree integrates synchronized point clouds and high-resolution images, encompassing 21,007 annotated tree instances across 50 species and 2 morphological parameters. Leveraging the unique characteristics, WHU-STree concurrently supports over 10 tasks related to street tree inventory. We benchmark representative baselines for two key tasks--tree species classification and individual tree segmentation. Extensive experiments and in-depth analysis demonstrate the significant potential of multi-modal data fusion and underscore cross-domain applicability as a critical prerequisite for practical algorithm deployment. In particular, we identify key challenges and outline potential future works for fully exploiting WHU-STree, encompassing multi-modal fusion, multi-task collaboration, cross-domain generalization, spatial pattern learning, and Multi-modal Large Language Model for street tree asset management. The WHU-STree dataset is accessible at: https://github.com/WHU-USI3DV/WHU-STree.
>
---
#### [new 045] Agent4FaceForgery: Multi-Agent LLM Framework for Realistic Face Forgery Detection
- **分类: cs.CV**

- **简介: 该论文提出Agent4FaceForgery框架，用于解决人脸伪造检测中真实场景与离线基准不匹配的问题。通过多智能体模拟人类伪造过程及社交媒体中的图文交互，生成高质量数据，提升检测模型性能。属于图像伪造检测任务。**

- **链接: [http://arxiv.org/pdf/2509.12546v1](http://arxiv.org/pdf/2509.12546v1)**

> **作者:** Yingxin Lai; Zitong Yu; Jun Wang; Linlin Shen; Yong Xu; Xiaochun Cao
>
> **摘要:** Face forgery detection faces a critical challenge: a persistent gap between offline benchmarks and real-world efficacy,which we attribute to the ecological invalidity of training data.This work introduces Agent4FaceForgery to address two fundamental problems: (1) how to capture the diverse intents and iterative processes of human forgery creation, and (2) how to model the complex, often adversarial, text-image interactions that accompany forgeries in social media. To solve this,we propose a multi-agent framework where LLM-poweredagents, equipped with profile and memory modules, simulate the forgery creation process. Crucially, these agents interact in a simulated social environment to generate samples labeled for nuanced text-image consistency, moving beyond simple binary classification. An Adaptive Rejection Sampling (ARS) mechanism ensures data quality and diversity. Extensive experiments validate that the data generated by our simulationdriven approach brings significant performance gains to detectors of multiple architectures, fully demonstrating the effectiveness and value of our framework.
>
---
#### [new 046] Dream3DAvatar: Text-Controlled 3D Avatar Reconstruction from a Single Image
- **分类: cs.CV**

- **简介: 该论文提出Dream3DAvatar，解决单图生成可控3D全身虚拟人任务。通过两阶段框架，结合文本控制与多视角信息，提升几何与纹理重建质量，尤其优化遮挡区域与面部细节。**

- **链接: [http://arxiv.org/pdf/2509.13013v1](http://arxiv.org/pdf/2509.13013v1)**

> **作者:** Gaofeng Liu; Hengsen Li; Ruoyu Gao; Xuetong Li; Zhiyuan Ma; Tao Fang
>
> **摘要:** With the rapid advancement of 3D representation techniques and generative models, substantial progress has been made in reconstructing full-body 3D avatars from a single image. However, this task remains fundamentally ill-posedness due to the limited information available from monocular input, making it difficult to control the geometry and texture of occluded regions during generation. To address these challenges, we redesign the reconstruction pipeline and propose Dream3DAvatar, an efficient and text-controllable two-stage framework for 3D avatar generation. In the first stage, we develop a lightweight, adapter-enhanced multi-view generation model. Specifically, we introduce the Pose-Adapter to inject SMPL-X renderings and skeletal information into SDXL, enforcing geometric and pose consistency across views. To preserve facial identity, we incorporate ID-Adapter-G, which injects high-resolution facial features into the generation process. Additionally, we leverage BLIP2 to generate high-quality textual descriptions of the multi-view images, enhancing text-driven controllability in occluded regions. In the second stage, we design a feedforward Transformer model equipped with a multi-view feature fusion module to reconstruct high-fidelity 3D Gaussian Splat representations (3DGS) from the generated images. Furthermore, we introduce ID-Adapter-R, which utilizes a gating mechanism to effectively fuse facial features into the reconstruction process, improving high-frequency detail recovery. Extensive experiments demonstrate that our method can generate realistic, animation-ready 3D avatars without any post-processing and consistently outperforms existing baselines across multiple evaluation metrics.
>
---
#### [new 047] Hunyuan3D Studio: End-to-End AI Pipeline for Game-Ready 3D Asset Generation
- **分类: cs.CV**

- **简介: 该论文提出Hunyuan3D Studio，一个端到端AI平台，用于自动生成高质量游戏用3D资产。其解决传统流程耗时且专业的问题，整合多个神经模块，实现从概念图或文本快速生成符合游戏引擎要求的3D模型。**

- **链接: [http://arxiv.org/pdf/2509.12815v1](http://arxiv.org/pdf/2509.12815v1)**

> **作者:** Biwen Lei; Yang Li; Xinhai Liu; Shuhui Yang; Lixin Xu; Jingwei Huang; Ruining Tang; Haohan Weng; Jian Liu; Jing Xu; Zhen Zhou; Yiling Zhu; Jiankai Xing; Jiachen Xu; Changfeng Ma; Xinhao Yan; Yunhan Yang; Chunshi Wang; Duoteng Xu; Xueqi Ma; Yuguang Chen; Jing Li; Mingxin Yang; Sheng Zhang; Yifei Feng; Xin Huang; Di Luo; Zebin He; Puhua Jiang; Changrong Hu; Zihan Qin; Shiwei Miao; Haolin Liu; Yunfei Zhao; Zeqiang Lai; Qingxiang Lin; Zibo Zhao; Kunhong Li; Xianghui Yang; Huiwen Shi; Xin Yang; Yuxuan Wang; Zebin Yao; Yihang Lian; Sicong Liu; Xintong Han; Wangchen Qin; Caisheng Ouyang; Jianyin Liu; Tianwen Yuan; Shuai Jiang; Hong Duan; Yanqi Niu; Wencong Lin; Yifu Sun; Shirui Huang; Lin Niu; Gu Gong; Guojian Xiao; Bojian Zheng; Xiang Yuan; Qi Chen; Jie Xiao; Dongyang Zheng; Xiaofeng Yang; Kai Liu; Jianchen Zhu; Lifu Wang; Qinglin Lu; Jie Liu; Liang Dong; Fan Jiang; Ruibin Chen; Lei Wang; Chao Zhang; Jiaxin Lin; Hao Zhang; Zheng Ye; Peng He; Runzhou Wu; Yinhe Wu; Jiayao Du; Jupeng Chen; Xinyue Mao; Dongyuan Guo; Yixuan Tang; Yulin Tsai; Yonghao Tan; Jiaao Yu; Junlin Yu; Keren Zhang; Yifan Li; Peng Chen; Tian Liu; Di Wang; Yuhong Liu; Linus; Jie Jiang; Zhuo Chen; Chunchao Guo
>
> **备注:** Technical Report
>
> **摘要:** The creation of high-quality 3D assets, a cornerstone of modern game development, has long been characterized by labor-intensive and specialized workflows. This paper presents Hunyuan3D Studio, an end-to-end AI-powered content creation platform designed to revolutionize the game production pipeline by automating and streamlining the generation of game-ready 3D assets. At its core, Hunyuan3D Studio integrates a suite of advanced neural modules (such as Part-level 3D Generation, Polygon Generation, Semantic UV, etc.) into a cohesive and user-friendly system. This unified framework allows for the rapid transformation of a single concept image or textual description into a fully-realized, production-quality 3D model complete with optimized geometry and high-fidelity PBR textures. We demonstrate that assets generated by Hunyuan3D Studio are not only visually compelling but also adhere to the stringent technical requirements of contemporary game engines, significantly reducing iteration time and lowering the barrier to entry for 3D content creation. By providing a seamless bridge from creative intent to technical asset, Hunyuan3D Studio represents a significant leap forward for AI-assisted workflows in game development and interactive media.
>
---
#### [new 048] Effective Gaussian Management for High-fidelity Object Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出一种高效的高保真物体重建方法，解决传统高斯泼溅方法中属性分配不当导致的梯度冲突问题。通过动态激活高斯函数的属性和轻量化表示，提升重建质量与效率，适用于多种模型框架。**

- **链接: [http://arxiv.org/pdf/2509.12742v1](http://arxiv.org/pdf/2509.12742v1)**

> **作者:** Jiateng Liu; Hao Gao; Jiu-Cheng Xie; Chi-Man Pun; Jian Xiong; Haolun Li; Feng Xu
>
> **摘要:** This paper proposes an effective Gaussian management approach for high-fidelity object reconstruction. Departing from recent Gaussian Splatting (GS) methods that employ indiscriminate attribute assignment, our approach introduces a novel densification strategy that dynamically activates spherical harmonics (SHs) or normals under the supervision of a surface reconstruction module, which effectively mitigates the gradient conflicts caused by dual supervision and achieves superior reconstruction results. To further improve representation efficiency, we develop a lightweight Gaussian representation that adaptively adjusts the SH orders of each Gaussian based on gradient magnitudes and performs task-decoupled pruning to remove Gaussian with minimal impact on a reconstruction task without sacrificing others, which balances the representational capacity with parameter quantity. Notably, our management approach is model-agnostic and can be seamlessly integrated into other frameworks, enhancing performance while reducing model size. Extensive experiments demonstrate that our approach consistently outperforms state-of-the-art approaches in both reconstruction quality and efficiency, achieving superior performance with significantly fewer parameters.
>
---
#### [new 049] BATR-FST: Bi-Level Adaptive Token Refinement for Few-Shot Transformers
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出BATR-FST方法，解决视觉Transformer在少样本学习中的性能不足问题。通过双阶段策略，结合掩码图像建模与自适应token细化模块，提升token表示与分类能力，实验证明其在少样本任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2509.12768v1](http://arxiv.org/pdf/2509.12768v1)**

> **作者:** Mohammed Al-Habib; Zuping Zhang; Abdulrahman Noman
>
> **备注:** This paper has been accepted for publication at the IEEE International Joint Conference on Neural Networks (IJCNN), Rome, Italy 2025
>
> **摘要:** Vision Transformers (ViTs) have shown significant promise in computer vision applications. However, their performance in few-shot learning is limited by challenges in refining token-level interactions, struggling with limited training data, and developing a strong inductive bias. Existing methods often depend on inflexible token matching or basic similarity measures, which limit the effective incorporation of global context and localized feature refinement. To address these challenges, we propose Bi-Level Adaptive Token Refinement for Few-Shot Transformers (BATR-FST), a two-stage approach that progressively improves token representations and maintains a robust inductive bias for few-shot classification. During the pre-training phase, Masked Image Modeling (MIM) provides Vision Transformers (ViTs) with transferable patch-level representations by recreating masked image regions, providing a robust basis for subsequent adaptation. In the meta-fine-tuning phase, BATR-FST incorporates a Bi-Level Adaptive Token Refinement module that utilizes Token Clustering to capture localized interactions, Uncertainty-Aware Token Weighting to prioritize dependable features, and a Bi-Level Attention mechanism to balance intra-cluster and inter-cluster relationships, thereby facilitating thorough token refinement. Furthermore, Graph Token Propagation ensures semantic consistency between support and query instances, while a Class Separation Penalty preserves different class borders, enhancing discriminative capability. Extensive experiments on three benchmark few-shot datasets demonstrate that BATR-FST achieves superior results in both 1-shot and 5-shot scenarios and improves the few-shot classification via transformers.
>
---
#### [new 050] SmokeBench: A Real-World Dataset for Surveillance Image Desmoking in Early-Stage Fire Scenes
- **分类: cs.CV**

- **简介: 该论文提出SmokeBench数据集，用于解决早期火灾场景中监控图像去烟问题。通过提供真实配对的有烟和无烟图像，推动去烟算法的发展，提升应急响应能力。**

- **链接: [http://arxiv.org/pdf/2509.12701v1](http://arxiv.org/pdf/2509.12701v1)**

> **作者:** Wenzhuo Jin; Qianfeng Yang; Xianhao Wu; Hongming Chen; Pengpeng Li; Xiang Chen
>
> **备注:** Accepted by ACMMM 2025 Datasets Track
>
> **摘要:** Early-stage fire scenes (0-15 minutes after ignition) represent a crucial temporal window for emergency interventions. During this stage, the smoke produced by combustion significantly reduces the visibility of surveillance systems, severely impairing situational awareness and hindering effective emergency response and rescue operations. Consequently, there is an urgent need to remove smoke from images to obtain clear scene information. However, the development of smoke removal algorithms remains limited due to the lack of large-scale, real-world datasets comprising paired smoke-free and smoke-degraded images. To address these limitations, we present a real-world surveillance image desmoking benchmark dataset named SmokeBench, which contains image pairs captured under diverse scenes setup and smoke concentration. The curated dataset provides precisely aligned degraded and clean images, enabling supervised learning and rigorous evaluation. We conduct comprehensive experiments by benchmarking a variety of desmoking methods on our dataset. Our dataset provides a valuable foundation for advancing robust and practical image desmoking in real-world fire scenes. This dataset has been released to the public and can be downloaded from https://github.com/ncfjd/SmokeBench.
>
---
#### [new 051] DialNav: Multi-turn Dialog Navigation with a Remote Guide
- **分类: cs.CV**

- **简介: 论文提出DialNav任务，研究导航代理与远程引导者通过多轮对话协作完成导航。任务要求引导者推断导航者位置，强调沟通重要性。论文构建RAIN数据集，设计评估基准，并分析不同模型效果，推动具身对话研究。**

- **链接: [http://arxiv.org/pdf/2509.12894v1](http://arxiv.org/pdf/2509.12894v1)**

> **作者:** Leekyeung Han; Hyunji Min; Gyeom Hwangbo; Jonghyun Choi; Paul Hongsuck Seo
>
> **备注:** 18 pages, 8 figures, ICCV 2025
>
> **摘要:** We introduce DialNav, a novel collaborative embodied dialog task, where a navigation agent (Navigator) and a remote guide (Guide) engage in multi-turn dialog to reach a goal location. Unlike prior work, DialNav aims for holistic evaluation and requires the Guide to infer the Navigator's location, making communication essential for task success. To support this task, we collect and release the Remote Assistance in Navigation (RAIN) dataset, human-human dialog paired with navigation trajectories in photorealistic environments. We design a comprehensive benchmark to evaluate both navigation and dialog, and conduct extensive experiments analyzing the impact of different Navigator and Guide models. We highlight key challenges and publicly release the dataset, code, and evaluation framework to foster future research in embodied dialog.
>
---
#### [new 052] Leveraging Large Language Models to Effectively Generate Visual Data for Canine Musculoskeletal Diagnoses
- **分类: cs.CV**

- **简介: 论文提出利用大语言模型生成犬类肌肉骨骼诊断的合成视觉数据，解决医疗数据稀缺问题。通过映射和提示技术生成1000份合成数据，实现88% F1分数，验证了LLM在生成医学图像数据中的潜力。**

- **链接: [http://arxiv.org/pdf/2509.12866v1](http://arxiv.org/pdf/2509.12866v1)**

> **作者:** Martin Thißen; Thi Ngoc Diep Tran; Barbara Esteve Ratsch; Ben Joel Schönbein; Ute Trapp; Beate Egner; Romana Piat; Elke Hergenröther
>
> **摘要:** It is well-established that more data generally improves AI model performance. However, data collection can be challenging for certain tasks due to the rarity of occurrences or high costs. These challenges are evident in our use case, where we apply AI models to a novel approach for visually documenting the musculoskeletal condition of dogs. Here, abnormalities are marked as colored strokes on a body map of a dog. Since these strokes correspond to distinct muscles or joints, they can be mapped to the textual domain in which large language models (LLMs) operate. LLMs have demonstrated impressive capabilities across a wide range of tasks, including medical applications, offering promising potential for generating synthetic training data. In this work, we investigate whether LLMs can effectively generate synthetic visual training data for canine musculoskeletal diagnoses. For this, we developed a mapping that segments visual documentations into over 200 labeled regions representing muscles or joints. Using techniques like guided decoding, chain-of-thought reasoning, and few-shot prompting, we generated 1,000 synthetic visual documentations for patellar luxation (kneecap dislocation) diagnosis, the diagnosis for which we have the most real-world data. Our analysis shows that the generated documentations are sensitive to location and severity of the diagnosis while remaining independent of the dog's sex. We further generated 1,000 visual documentations for various other diagnoses to create a binary classification dataset. A model trained solely on this synthetic data achieved an F1 score of 88% on 70 real-world documentations. These results demonstrate the potential of LLM-generated synthetic data, which is particularly valuable for addressing data scarcity in rare diseases. While our methodology is tailored to the medical domain, the insights and techniques can be adapted to other fields.
>
---
#### [new 053] Neural Collapse-Inspired Multi-Label Federated Learning under Label-Distribution Skew
- **分类: cs.CV**

- **简介: 该论文属于联邦学习中的多标签分类任务，解决标签分布偏斜下的模型性能下降问题。提出基于神经坍缩理论的方法，通过特征解耦与正则化损失提升跨客户端的表示一致性与聚类质量。**

- **链接: [http://arxiv.org/pdf/2509.12544v1](http://arxiv.org/pdf/2509.12544v1)**

> **作者:** Can Peng; Yuyuan Liu; Yingyu Yang; Pramit Saha; Qianye Yang; J. Alison Noble
>
> **摘要:** Federated Learning (FL) enables collaborative model training across distributed clients while preserving data privacy. However, the performance of deep learning often deteriorates in FL due to decentralized and heterogeneous data. This challenge is further amplified in multi-label scenarios, where data exhibit complex characteristics such as label co-occurrence, inter-label dependency, and discrepancies between local and global label relationships. While most existing FL research primarily focuses on single-label classification, many real-world applications, particularly in domains such as medical imaging, often involve multi-label settings. In this paper, we address this important yet underexplored scenario in FL, where clients hold multi-label data with skewed label distributions. Neural Collapse (NC) describes a geometric structure in the latent feature space where features of each class collapse to their class mean with vanishing intra-class variance, and the class means form a maximally separated configuration. Motivated by this theory, we propose a method to align feature distributions across clients and to learn high-quality, well-clustered representations. To make the NC-structure applicable to multi-label settings, where image-level features may contain multiple semantic concepts, we introduce a feature disentanglement module that extracts semantically specific features. The clustering of these disentangled class-wise features is guided by a predefined shared NC structure, which mitigates potential conflicts between client models due to diverse local data distributions. In addition, we design regularisation losses to encourage compact clustering in the latent feature space. Experiments conducted on four benchmark datasets across eight diverse settings demonstrate that our approach outperforms existing methods, validating its effectiveness in this challenging FL scenario.
>
---
#### [new 054] MATTER: Multiscale Attention for Registration Error Regression
- **分类: cs.CV**

- **简介: 该论文属于点云配准质量验证任务，旨在量化配准误差。提出基于回归的方法，结合多尺度特征提取与注意力机制，实现更精细的误差估计，提升映射质量，优于现有分类方法。**

- **链接: [http://arxiv.org/pdf/2509.12924v1](http://arxiv.org/pdf/2509.12924v1)**

> **作者:** Shipeng Liu; Ziliang Xiong; Khac-Hoang Ngo; Per-Erik Forssén
>
> **摘要:** Point cloud registration (PCR) is crucial for many downstream tasks, such as simultaneous localization and mapping (SLAM) and object tracking. This makes detecting and quantifying registration misalignment, i.e.,~{\it PCR quality validation}, an important task. All existing methods treat validation as a classification task, aiming to assign the PCR quality to a few classes. In this work, we instead use regression for PCR validation, allowing for a more fine-grained quantification of the registration quality. We also extend previously used misalignment-related features by using multiscale extraction and attention-based aggregation. This leads to accurate and robust registration error estimation on diverse datasets, especially for point clouds with heterogeneous spatial densities. Furthermore, when used to guide a mapping downstream task, our method significantly improves the mapping quality for a given amount of re-registered frames, compared to the state-of-the-art classification-based method.
>
---
#### [new 055] Weakly and Self-Supervised Class-Agnostic Motion Prediction for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的运动预测任务，旨在解决无需大量标注的类无关运动预测问题。提出弱监督和自监督方法，利用前景/背景掩码减少标注依赖，并设计一致性感知损失提升性能。**

- **链接: [http://arxiv.org/pdf/2509.13116v1](http://arxiv.org/pdf/2509.13116v1)**

> **作者:** Ruibo Li; Hanyu Shi; Zhe Wang; Guosheng Lin
>
> **备注:** An extension of our CVPR 2023 paper, "Weakly Supervised Class-Agnostic Motion Prediction for Autonomous Driving," accepted for publication in TPAMI
>
> **摘要:** Understanding motion in dynamic environments is critical for autonomous driving, thereby motivating research on class-agnostic motion prediction. In this work, we investigate weakly and self-supervised class-agnostic motion prediction from LiDAR point clouds. Outdoor scenes typically consist of mobile foregrounds and static backgrounds, allowing motion understanding to be associated with scene parsing. Based on this observation, we propose a novel weakly supervised paradigm that replaces motion annotations with fully or partially annotated (1%, 0.1%) foreground/background masks for supervision. To this end, we develop a weakly supervised approach utilizing foreground/background cues to guide the self-supervised learning of motion prediction models. Since foreground motion generally occurs in non-ground regions, non-ground/ground masks can serve as an alternative to foreground/background masks, further reducing annotation effort. Leveraging non-ground/ground cues, we propose two additional approaches: a weakly supervised method requiring fewer (0.01%) foreground/background annotations, and a self-supervised method without annotations. Furthermore, we design a Robust Consistency-aware Chamfer Distance loss that incorporates multi-frame information and robust penalty functions to suppress outliers in self-supervised learning. Experiments show that our weakly and self-supervised models outperform existing self-supervised counterparts, and our weakly supervised models even rival some supervised ones. This demonstrates that our approaches effectively balance annotation effort and performance.
>
---
#### [new 056] Cross-Layer Vision Smoothing: Enhancing Visual Understanding via Sustained Focus on Key Objects in Large Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出跨层视觉平滑（CLVS）方法，旨在解决大视觉语言模型（LVLMs）对关键对象注意力短暂的问题。通过引入视觉记忆机制，使模型在多层中保持对关键对象的持续关注，从而提升视觉理解能力，尤其在关系和属性理解任务上表现优异。**

- **链接: [http://arxiv.org/pdf/2509.12897v1](http://arxiv.org/pdf/2509.12897v1)**

> **作者:** Jianfei Zhao; Feng Zhang; Xin Sun; Lingxing Kong; Zhixing Tan; Chong Feng
>
> **摘要:** Large Vision-Language Models (LVLMs) can accurately locate key objects in images, yet their attention to these objects tends to be very brief. Motivated by the hypothesis that sustained focus on key objects can improve LVLMs' visual capabilities, we propose Cross-Layer Vision Smoothing (CLVS). The core idea of CLVS is to incorporate a vision memory that smooths the attention distribution across layers. Specifically, we initialize this vision memory with position-unbiased visual attention in the first layer. In subsequent layers, the model's visual attention jointly considers the vision memory from previous layers, while the memory is updated iteratively, thereby maintaining smooth attention on key objects. Given that visual understanding primarily occurs in the early and middle layers of the model, we use uncertainty as an indicator of completed visual understanding and terminate the smoothing process accordingly. Experiments on four benchmarks across three LVLMs confirm the effectiveness and generalizability of our method. CLVS achieves state-of-the-art performance on a variety of visual understanding tasks, with particularly significant improvements in relation and attribute understanding.
>
---
#### [new 057] DYNAMO: Dependency-Aware Deep Learning Framework for Articulated Assembly Motion Prediction
- **分类: cs.CV**

- **简介: 该论文提出DYNAMO框架，解决机械装配体运动预测问题。通过 MechBench 数据集，模型直接从CAD点云预测各部件的运动轨迹，无需关节标注，实现几何耦合运动的准确预测。**

- **链接: [http://arxiv.org/pdf/2509.12430v1](http://arxiv.org/pdf/2509.12430v1)**

> **作者:** Mayank Patel; Rahul Jain; Asim Unmesh; Karthik Ramani
>
> **摘要:** Understanding the motion of articulated mechanical assemblies from static geometry remains a core challenge in 3D perception and design automation. Prior work on everyday articulated objects such as doors and laptops typically assumes simplified kinematic structures or relies on joint annotations. However, in mechanical assemblies like gears, motion arises from geometric coupling, through meshing teeth or aligned axes, making it difficult for existing methods to reason about relational motion from geometry alone. To address this gap, we introduce MechBench, a benchmark dataset of 693 diverse synthetic gear assemblies with part-wise ground-truth motion trajectories. MechBench provides a structured setting to study coupled motion, where part dynamics are induced by contact and transmission rather than predefined joints. Building on this, we propose DYNAMO, a dependency-aware neural model that predicts per-part SE(3) motion trajectories directly from segmented CAD point clouds. Experiments show that DYNAMO outperforms strong baselines, achieving accurate and temporally consistent predictions across varied gear configurations. Together, MechBench and DYNAMO establish a novel systematic framework for data-driven learning of coupled mechanical motion in CAD assemblies.
>
---
#### [new 058] Hierarchical Deep Fusion Framework for Multi-dimensional Facial Forgery Detection - The 2024 Global Deepfake Image Detection Challenge
- **分类: cs.CV; cs.AI**

- **简介: 论文提出一种分层深度融合框架（HDFF），用于多维人脸伪造检测任务，解决深度伪造图像识别难题。通过集成四种预训练模型并进行多阶段微调，实现高精度检测，在比赛中取得优异成绩。**

- **链接: [http://arxiv.org/pdf/2509.13107v1](http://arxiv.org/pdf/2509.13107v1)**

> **作者:** Kohou Wang; Huan Hu; Xiang Liu; Zezhou Chen; Ping Chen; Zhaoxiang Liu; Shiguo Lian
>
> **备注:** The 2024 Global Deepfake Image Detection Challenge Top20 Reward, 5 pages
>
> **摘要:** The proliferation of sophisticated deepfake technology poses significant challenges to digital security and authenticity. Detecting these forgeries, especially across a wide spectrum of manipulation techniques, requires robust and generalized models. This paper introduces the Hierarchical Deep Fusion Framework (HDFF), an ensemble-based deep learning architecture designed for high-performance facial forgery detection. Our framework integrates four diverse pre-trained sub-models, Swin-MLP, CoAtNet, EfficientNetV2, and DaViT, which are meticulously fine-tuned through a multi-stage process on the MultiFFDI dataset. By concatenating the feature representations from these specialized models and training a final classifier layer, HDFF effectively leverages their collective strengths. This approach achieved a final score of 0.96852 on the competition's private leaderboard, securing the 20th position out of 184 teams, demonstrating the efficacy of hierarchical fusion for complex image classification tasks.
>
---
#### [new 059] Domain Adaptive SAR Wake Detection: Leveraging Similarity Filtering and Memory Guidance
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于跨模态领域自适应任务，解决SAR与光学图像间船尾流检测性能差异问题。提出SimMemDA框架，通过风格迁移、相似性过滤和记忆指导提升伪标签质量，增强模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.12279v1](http://arxiv.org/pdf/2509.12279v1)**

> **作者:** He Gao; Baoxiang Huang; Milena Radenkovic; Borui Li; Ge Chen
>
> **摘要:** Synthetic Aperture Radar (SAR), with its all- weather and wide-area observation capabilities, serves as a crucial tool for wake detection. However, due to its complex imaging mechanism, wake features in SAR images often appear abstract and noisy, posing challenges for accurate annotation. In contrast, optical images provide more distinct visual cues, but models trained on optical data suffer from performance degradation when applied to SAR images due to domain shift. To address this cross-modal domain adaptation challenge, we propose a Similarity-Guided and Memory-Guided Domain Adap- tation (termed SimMemDA) framework for unsupervised domain adaptive ship wake detection via instance-level feature similarity filtering and feature memory guidance. Specifically, to alleviate the visual discrepancy between optical and SAR images, we first utilize WakeGAN to perform style transfer on optical images, generating pseudo-images close to the SAR style. Then, instance-level feature similarity filtering mechanism is designed to identify and prioritize source samples with target-like dis- tributions, minimizing negative transfer. Meanwhile, a Feature- Confidence Memory Bank combined with a K-nearest neighbor confidence-weighted fusion strategy is introduced to dynamically calibrate pseudo-labels in the target domain, improving the reliability and stability of pseudo-labels. Finally, the framework further enhances generalization through region-mixed training, strategically combining source annotations with calibrated tar- get pseudo-labels. Experimental results demonstrate that the proposed SimMemDA method can improve the accuracy and robustness of cross-modal ship wake detection tasks, validating the effectiveness and feasibility of the proposed method.
>
---
#### [new 060] Humor in Pixels: Benchmarking Large Multimodal Models Understanding of Online Comics
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出PixelHumor基准数据集，用于评估大语言模型对网络漫画中多模态幽默和叙事的理解能力。任务是测试模型在视觉与文本整合、叙事推理方面的表现，揭示当前模型在社会智能交互中的不足。**

- **链接: [http://arxiv.org/pdf/2509.12248v1](http://arxiv.org/pdf/2509.12248v1)**

> **作者:** Yuriel Ryan; Rui Yang Tan; Kenny Tsu Wei Choo; Roy Ka-Wei Lee
>
> **备注:** 27 pages, 8 figures, EMNLP 2025
>
> **摘要:** Understanding humor is a core aspect of social intelligence, yet it remains a significant challenge for Large Multimodal Models (LMMs). We introduce PixelHumor, a benchmark dataset of 2,800 annotated multi-panel comics designed to evaluate LMMs' ability to interpret multimodal humor and recognize narrative sequences. Experiments with state-of-the-art LMMs reveal substantial gaps: for instance, top models achieve only 61% accuracy in panel sequencing, far below human performance. This underscores critical limitations in current models' integration of visual and textual cues for coherent narrative and humor understanding. By providing a rigorous framework for evaluating multimodal contextual and narrative reasoning, PixelHumor aims to drive the development of LMMs that better engage in natural, socially aware interactions.
>
---
#### [new 061] Defense-to-Attack: Bypassing Weak Defenses Enables Stronger Jailbreaks in Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Defense2Attack方法，通过利用弱防御机制提升视觉语言模型的越狱攻击效果。属于安全攻防任务，旨在解决现有越狱攻击效率低的问题，设计了三个优化组件以增强攻击性能。**

- **链接: [http://arxiv.org/pdf/2509.12724v1](http://arxiv.org/pdf/2509.12724v1)**

> **作者:** Yunhan Zhao; Xiang Zheng; Xingjun Ma
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Despite their superb capabilities, Vision-Language Models (VLMs) have been shown to be vulnerable to jailbreak attacks. While recent jailbreaks have achieved notable progress, their effectiveness and efficiency can still be improved. In this work, we reveal an interesting phenomenon: incorporating weak defense into the attack pipeline can significantly enhance both the effectiveness and the efficiency of jailbreaks on VLMs. Building on this insight, we propose Defense2Attack, a novel jailbreak method that bypasses the safety guardrails of VLMs by leveraging defensive patterns to guide jailbreak prompt design. Specifically, Defense2Attack consists of three key components: (1) a visual optimizer that embeds universal adversarial perturbations with affirmative and encouraging semantics; (2) a textual optimizer that refines the input using a defense-styled prompt; and (3) a red-team suffix generator that enhances the jailbreak through reinforcement fine-tuning. We empirically evaluate our method on four VLMs and four safety benchmarks. The results demonstrate that Defense2Attack achieves superior jailbreak performance in a single attempt, outperforming state-of-the-art attack methods that often require multiple tries. Our work offers a new perspective on jailbreaking VLMs.
>
---
#### [new 062] Recurrent Cross-View Object Geo-Localization
- **分类: cs.CV**

- **简介: 该论文提出ReCOT模型，解决跨视角物体地理定位任务，通过递归机制和模块优化提升定位精度与效率，减少参数量，取得SOTA性能。**

- **链接: [http://arxiv.org/pdf/2509.12757v1](http://arxiv.org/pdf/2509.12757v1)**

> **作者:** Xiaohan Zhang; Si-Yuan Cao; Xiaokai Bai; Yiming Li; Zhangkai Shen; Zhe Wu; Xiaoxi Hu; Hui-liang Shen
>
> **摘要:** Cross-view object geo-localization (CVOGL) aims to determine the location of a specific object in high-resolution satellite imagery given a query image with a point prompt. Existing approaches treat CVOGL as a one-shot detection task, directly regressing object locations from cross-view information aggregation, but they are vulnerable to feature noise and lack mechanisms for error correction. In this paper, we propose ReCOT, a Recurrent Cross-view Object geo-localization Transformer, which reformulates CVOGL as a recurrent localization task. ReCOT introduces a set of learnable tokens that encode task-specific intent from the query image and prompt embeddings, and iteratively attend to the reference features to refine the predicted location. To enhance this recurrent process, we incorporate two complementary modules: (1) a SAM-based knowledge distillation strategy that transfers segmentation priors from the Segment Anything Model (SAM) to provide clearer semantic guidance without additional inference cost, and (2) a Reference Feature Enhancement Module (RFEM) that introduces a hierarchical attention to emphasize object-relevant regions in the reference features. Extensive experiments on standard CVOGL benchmarks demonstrate that ReCOT achieves state-of-the-art (SOTA) performance while reducing parameters by 60% compared to previous SOTA approaches.
>
---
#### [new 063] From Orthomosaics to Raw UAV Imagery: Enhancing Palm Detection and Crown-Center Localization
- **分类: cs.CV**

- **简介: 论文研究利用原始无人机影像提升棕榈树检测与树冠中心定位精度。任务为个体树木映射，解决正射影像拼接缺陷与预处理限制问题，通过对比正射影像与原始影像表现，并引入树冠中心标注提升定位准确性。**

- **链接: [http://arxiv.org/pdf/2509.12400v1](http://arxiv.org/pdf/2509.12400v1)**

> **作者:** Rongkun Zhu; Kangning Cui; Wei Tang; Rui-Feng Wang; Sarra Alqahtani; David Lutz; Fan Yang; Paul Fine; Jordan Karubian; Robert Plemmons; Jean-Michel Morel; Victor Pauca; Miles Silman
>
> **备注:** 7 pages, 2 figures, 2 tables
>
> **摘要:** Accurate mapping of individual trees is essential for ecological monitoring and forest management. Orthomosaic imagery from unmanned aerial vehicles (UAVs) is widely used, but stitching artifacts and heavy preprocessing limit its suitability for field deployment. This study explores the use of raw UAV imagery for palm detection and crown-center localization in tropical forests. Two research questions are addressed: (1) how detection performance varies across orthomosaic and raw imagery, including within-domain and cross-domain transfer, and (2) to what extent crown-center annotations improve localization accuracy beyond bounding-box centroids. Using state-of-the-art detectors and keypoint models, we show that raw imagery yields superior performance in deployment-relevant scenarios, while orthomosaics retain value for robust cross-domain generalization. Incorporating crown-center annotations in training further improves localization and provides precise tree positions for downstream ecological analyses. These findings offer practical guidance for UAV-based biodiversity and conservation monitoring.
>
---
#### [new 064] StereoCarla: A High-Fidelity Driving Dataset for Generalizable Stereo
- **分类: cs.CV**

- **简介: 该论文提出StereoCarla，一个高保真合成立体视觉数据集，用于提升自动驾驶场景中立体匹配算法的泛化能力。通过多样化配置和环境条件，解决现有数据集泛化性不足的问题，并在多个基准测试中表现更优。**

- **链接: [http://arxiv.org/pdf/2509.12683v1](http://arxiv.org/pdf/2509.12683v1)**

> **作者:** Xianda Guo; Chenming Zhang; Ruilin Wang; Youmin Zhang; Wenzhao Zheng; Matteo Poggi; Hao Zhao; Qin Zou; Long Chen
>
> **摘要:** Stereo matching plays a crucial role in enabling depth perception for autonomous driving and robotics. While recent years have witnessed remarkable progress in stereo matching algorithms, largely driven by learning-based methods and synthetic datasets, the generalization performance of these models remains constrained by the limited diversity of existing training data. To address these challenges, we present StereoCarla, a high-fidelity synthetic stereo dataset specifically designed for autonomous driving scenarios. Built on the CARLA simulator, StereoCarla incorporates a wide range of camera configurations, including diverse baselines, viewpoints, and sensor placements as well as varied environmental conditions such as lighting changes, weather effects, and road geometries. We conduct comprehensive cross-domain experiments across four standard evaluation datasets (KITTI2012, KITTI2015, Middlebury, ETH3D) and demonstrate that models trained on StereoCarla outperform those trained on 11 existing stereo datasets in terms of generalization accuracy across multiple benchmarks. Furthermore, when integrated into multi-dataset training, StereoCarla contributes substantial improvements to generalization accuracy, highlighting its compatibility and scalability. This dataset provides a valuable benchmark for developing and evaluating stereo algorithms under realistic, diverse, and controllable settings, facilitating more robust depth perception systems for autonomous vehicles. Code can be available at https://github.com/XiandaGuo/OpenStereo, and data can be available at https://xiandaguo.net/StereoCarla.
>
---
#### [new 065] Deep learning for 3D point cloud processing - from approaches, tasks to its implications on urban and environmental applications
- **分类: cs.CV**

- **简介: 该论文综述深度学习在点云处理中的应用，涵盖场景补全、配准等任务，分析其在城市与环境应用中的挑战与不足，旨在推动算法向实际应用转化。**

- **链接: [http://arxiv.org/pdf/2509.12452v1](http://arxiv.org/pdf/2509.12452v1)**

> **作者:** Zhenxin Zhang; Zhihua Xu; Yuwei Cao; Ningli Xu; Shuye Wang; Shen'ao Cui; Zhen Li; Rongjun Qin
>
> **备注:** 57 Pages, 4 Figures
>
> **摘要:** Point cloud processing as a fundamental task in the field of geomatics and computer vision, has been supporting tasks and applications at different scales from air to ground, including mapping, environmental monitoring, urban/tree structure modeling, automated driving, robotics, disaster responses etc. Due to the rapid development of deep learning, point cloud processing algorithms have nowadays been almost explicitly dominated by learning-based approaches, most of which are yet transitioned into real-world practices. Existing surveys primarily focus on the ever-updating network architecture to accommodate unordered point clouds, largely ignoring their practical values in typical point cloud processing applications, in which extra-large volume of data, diverse scene contents, varying point density, data modality need to be considered. In this paper, we provide a meta review on deep learning approaches and datasets that cover a selection of critical tasks of point cloud processing in use such as scene completion, registration, semantic segmentation, and modeling. By reviewing a broad range of urban and environmental applications these tasks can support, we identify gaps to be closed as these methods transformed into applications and draw concluding remarks in both the algorithmic and practical aspects of the surveyed methods.
>
---
#### [new 066] Artificial Intelligence in Breast Cancer Care: Transforming Preoperative Planning and Patient Education with 3D Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出一种基于U-Mamba的机器学习方法，用于提升3D解剖重建的算法泛化能力。任务是解决传统模型在乳腺癌术前规划和患者教育中的泛化不足问题，通过人机协作优化分割结果，提高可视化效果与临床应用价值。**

- **链接: [http://arxiv.org/pdf/2509.12242v1](http://arxiv.org/pdf/2509.12242v1)**

> **作者:** Mustafa Khanbhai; Giulia Di Nardo; Jun Ma; Vivienne Freitas; Caterina Masino; Ali Dolatabadi; Zhaoxun "Lorenz" Liu; Wey Leong; Wagner H. Souza; Amin Madani
>
> **摘要:** Effective preoperative planning requires accurate algorithms for segmenting anatomical structures across diverse datasets, but traditional models struggle with generalization. This study presents a novel machine learning methodology to improve algorithm generalization for 3D anatomical reconstruction beyond breast cancer applications. We processed 120 retrospective breast MRIs (January 2018-June 2023) through three phases: anonymization and manual segmentation of T1-weighted and dynamic contrast-enhanced sequences; co-registration and segmentation of whole breast, fibroglandular tissue, and tumors; and 3D visualization using ITK-SNAP. A human-in-the-loop approach refined segmentations using U-Mamba, designed to generalize across imaging scenarios. Dice similarity coefficient assessed overlap between automated segmentation and ground truth. Clinical relevance was evaluated through clinician and patient interviews. U-Mamba showed strong performance with DSC values of 0.97 ($\pm$0.013) for whole organs, 0.96 ($\pm$0.024) for fibroglandular tissue, and 0.82 ($\pm$0.12) for tumors on T1-weighted images. The model generated accurate 3D reconstructions enabling visualization of complex anatomical features. Clinician interviews indicated improved planning, intraoperative navigation, and decision support. Integration of 3D visualization enhanced patient education, communication, and understanding. This human-in-the-loop machine learning approach successfully generalizes algorithms for 3D reconstruction and anatomical segmentation across patient datasets, offering enhanced visualization for clinicians, improved preoperative planning, and more effective patient education, facilitating shared decision-making and empowering informed patient choices across medical applications.
>
---
#### [new 067] PANORAMA: The Rise of Omnidirectional Vision in the Embodied AI Era
- **分类: cs.CV**

- **简介: 该论文综述了全向视觉在具身AI时代的发展，提出PANORAMA系统架构，涵盖生成、感知、理解等模块，旨在推动全向视觉在工业与学术领域的应用，解决其基础研究滞后问题。**

- **链接: [http://arxiv.org/pdf/2509.12989v1](http://arxiv.org/pdf/2509.12989v1)**

> **作者:** Xu Zheng; Chenfei Liao; Ziqiao Weng; Kaiyu Lei; Zihao Dongfang; Haocong He; Yuanhuiyi Lyu; Lutao Jiang; Lu Qi; Li Chen; Danda Pani Paudel; Kailun Yang; Linfeng Zhang; Luc Van Gool; Xuming Hu
>
> **备注:** This paper presents a draft overview of the emerging field of omnidirectional vision in the context of embodied AI
>
> **摘要:** Omnidirectional vision, using 360-degree vision to understand the environment, has become increasingly critical across domains like robotics, industrial inspection, and environmental monitoring. Compared to traditional pinhole vision, omnidirectional vision provides holistic environmental awareness, significantly enhancing the completeness of scene perception and the reliability of decision-making. However, foundational research in this area has historically lagged behind traditional pinhole vision. This talk presents an emerging trend in the embodied AI era: the rapid development of omnidirectional vision, driven by growing industrial demand and academic interest. We highlight recent breakthroughs in omnidirectional generation, omnidirectional perception, omnidirectional understanding, and related datasets. Drawing on insights from both academia and industry, we propose an ideal panoramic system architecture in the embodied AI era, PANORAMA, which consists of four key subsystems. Moreover, we offer in-depth opinions related to emerging trends and cross-community impacts at the intersection of panoramic vision and embodied AI, along with the future roadmap and open challenges. This overview synthesizes state-of-the-art advancements and outlines challenges and opportunities for future research in building robust, general-purpose omnidirectional AI systems in the embodied AI era.
>
---
#### [new 068] 3D Aware Region Prompted Vision Language Model
- **分类: cs.CV**

- **简介: 该论文提出SR-3D模型，通过共享视觉token空间连接2D图像与3D数据，实现灵活区域提示，提升跨帧空间推理能力。属于视觉语言任务，解决2D与3D统一表征问题，无需多帧标注即可准确理解场景。**

- **链接: [http://arxiv.org/pdf/2509.13317v1](http://arxiv.org/pdf/2509.13317v1)**

> **作者:** An-Chieh Cheng; Yang Fu; Yukang Chen; Zhijian Liu; Xiaolong Li; Subhashree Radhakrishnan; Song Han; Yao Lu; Jan Kautz; Pavlo Molchanov; Hongxu Yin; Xiaolong Wang; Sifei Liu
>
> **备注:** Project Website: https://www.anjiecheng.me/sr3d
>
> **摘要:** We present Spatial Region 3D (SR-3D) aware vision-language model that connects single-view 2D images and multi-view 3D data through a shared visual token space. SR-3D supports flexible region prompting, allowing users to annotate regions with bounding boxes, segmentation masks on any frame, or directly in 3D, without the need for exhaustive multi-frame labeling. We achieve this by enriching 2D visual features with 3D positional embeddings, which allows the 3D model to draw upon strong 2D priors for more accurate spatial reasoning across frames, even when objects of interest do not co-occur within the same view. Extensive experiments on both general 2D vision language and specialized 3D spatial benchmarks demonstrate that SR-3D achieves state-of-the-art performance, underscoring its effectiveness for unifying 2D and 3D representation space on scene understanding. Moreover, we observe applicability to in-the-wild videos without sensory 3D inputs or ground-truth 3D annotations, where SR-3D accurately infers spatial relationships and metric measurements.
>
---
#### [new 069] A Comparative Study of YOLOv8 to YOLOv11 Performance in Underwater Vision Tasks
- **分类: cs.CV; cs.AI**

- **简介: 论文比较YOLOv8到YOLOv11在水下视觉任务中的性能。针对水下图像质量差、计算资源有限的问题，使用两个数据集评估不同模型的精度、速度等指标，发现YOLOv10在速度与精度间取得最佳平衡，并提供开源基准促进海洋视觉研究。**

- **链接: [http://arxiv.org/pdf/2509.12682v1](http://arxiv.org/pdf/2509.12682v1)**

> **作者:** Gordon Hung; Ivan Felipe Rodriguez
>
> **备注:** 9 pages, 8 figures, 10 tables
>
> **摘要:** Autonomous underwater vehicles (AUVs) increasingly rely on on-board computer-vision systems for tasks such as habitat mapping, ecological monitoring, and infrastructure inspection. However, underwater imagery is hindered by light attenuation, turbidity, and severe class imbalance, while the computational resources available on AUVs are limited. One-stage detectors from the YOLO family are attractive because they fuse localization and classification in a single, low-latency network; however, their terrestrial benchmarks (COCO, PASCAL-VOC, Open Images) leave open the question of how successive YOLO releases perform in the marine domain. We curate two openly available datasets that span contrasting operating conditions: a Coral Disease set (4,480 images, 18 classes) and a Fish Species set (7,500 images, 20 classes). For each dataset, we create four training regimes (25 %, 50 %, 75 %, 100 % of the images) while keeping balanced validation and test partitions fixed. We train YOLOv8-s, YOLOv9-s, YOLOv10-s, and YOLOv11-s with identical hyperparameters (100 epochs, 640 px input, batch = 16, T4 GPU) and evaluate precision, recall, mAP50, mAP50-95, per-image inference time, and frames-per-second (FPS). Post-hoc Grad-CAM visualizations probe feature utilization and localization faithfulness. Across both datasets, accuracy saturates after YOLOv9, suggesting architectural innovations primarily target efficiency rather than accuracy. Inference speed, however, improves markedly. Our results (i) provide the first controlled comparison of recent YOLO variants on underwater imagery, (ii) show that lightweight YOLOv10 offers the best speed-accuracy trade-off for embedded AUV deployment, and (iii) deliver an open, reproducible benchmark and codebase to accelerate future marine-vision research.
>
---
#### [new 070] What Makes a Good Generated Image? Investigating Human and Multimodal LLM Image Preference Alignment
- **分类: cs.CV**

- **简介: 该论文研究人类与多模态大语言模型在图像质量评估上的差异。通过构建数据集和分析属性相关性，发现LLMs在某些属性（如解剖准确性）上判断能力较弱，揭示了人与模型感知图像的差异。属于图像质量评估任务。**

- **链接: [http://arxiv.org/pdf/2509.12750v1](http://arxiv.org/pdf/2509.12750v1)**

> **作者:** Rishab Parthasarathy; Jasmine Collins; Cory Stephenson
>
> **备注:** 7 pages, 9 figures, 3 tables; appendix 16 pages, 9 figures, 6 tables
>
> **摘要:** Automated evaluation of generative text-to-image models remains a challenging problem. Recent works have proposed using multimodal LLMs to judge the quality of images, but these works offer little insight into how multimodal LLMs make use of concepts relevant to humans, such as image style or composition, to generate their overall assessment. In this work, we study what attributes of an image--specifically aesthetics, lack of artifacts, anatomical accuracy, compositional correctness, object adherence, and style--are important for both LLMs and humans to make judgments on image quality. We first curate a dataset of human preferences using synthetically generated image pairs. We use inter-task correlation between each pair of image quality attributes to understand which attributes are related in making human judgments. Repeating the same analysis with LLMs, we find that the relationships between image quality attributes are much weaker. Finally, we study individual image quality attributes by generating synthetic datasets with a high degree of control for each axis. Humans are able to easily judge the quality of an image with respect to all of the specific image quality attributes (e.g. high vs. low aesthetic image), however we find that some attributes, such as anatomical accuracy, are much more difficult for multimodal LLMs to learn to judge. Taken together, these findings reveal interesting differences between how humans and multimodal LLMs perceive images.
>
---
#### [new 071] MSGFusion: Multimodal Scene Graph-Guided Infrared and Visible Image Fusion
- **分类: cs.CV**

- **简介: 该论文提出MSGFusion，用于红外与可见光图像融合任务。旨在解决现有方法依赖低级视觉线索、缺乏高级语义信息的问题。通过结合文本和视觉生成的结构化场景图，同步优化语义与细节，提升融合效果与下游任务性能。**

- **链接: [http://arxiv.org/pdf/2509.12901v1](http://arxiv.org/pdf/2509.12901v1)**

> **作者:** Guihui Li; Bowei Dong; Kaizhi Dong; Jiayi Li; Haiyong Zheng
>
> **摘要:** Infrared and visible image fusion has garnered considerable attention owing to the strong complementarity of these two modalities in complex, harsh environments. While deep learning-based fusion methods have made remarkable advances in feature extraction, alignment, fusion, and reconstruction, they still depend largely on low-level visual cues, such as texture and contrast, and struggle to capture the high-level semantic information embedded in images. Recent attempts to incorporate text as a source of semantic guidance have relied on unstructured descriptions that neither explicitly model entities, attributes, and relationships nor provide spatial localization, thereby limiting fine-grained fusion performance. To overcome these challenges, we introduce MSGFusion, a multimodal scene graph-guided fusion framework for infrared and visible imagery. By deeply coupling structured scene graphs derived from text and vision, MSGFusion explicitly represents entities, attributes, and spatial relations, and then synchronously refines high-level semantics and low-level details through successive modules for scene graph representation, hierarchical aggregation, and graph-driven fusion. Extensive experiments on multiple public benchmarks show that MSGFusion significantly outperforms state-of-the-art approaches, particularly in detail preservation and structural clarity, and delivers superior semantic consistency and generalizability in downstream tasks such as low-light object detection, semantic segmentation, and medical image fusion.
>
---
#### [new 072] ResidualViT for Efficient Temporally Dense Video Encoding
- **分类: cs.CV; cs.AI; cs.IR; eess.IV**

- **简介: 该论文提出ResidualViT，用于高效密集视频编码。针对高时序分辨率下计算帧级特征成本高的问题，引入残差连接和令牌缩减模块，并采用轻量蒸馏策略，显著降低计算成本并提升推理速度，同时保持模型精度。**

- **链接: [http://arxiv.org/pdf/2509.13255v1](http://arxiv.org/pdf/2509.13255v1)**

> **作者:** Mattia Soldan; Fabian Caba Heilbron; Bernard Ghanem; Josef Sivic; Bryan Russell
>
> **摘要:** Several video understanding tasks, such as natural language temporal video grounding, temporal activity localization, and audio description generation, require "temporally dense" reasoning over frames sampled at high temporal resolution. However, computing frame-level features for these tasks is computationally expensive given the temporal resolution requirements. In this paper, we make three contributions to reduce the cost of computing features for temporally dense tasks. First, we introduce a vision transformer (ViT) architecture, dubbed ResidualViT, that leverages the large temporal redundancy in videos to efficiently compute temporally dense frame-level features. Our architecture incorporates (i) learnable residual connections that ensure temporal consistency across consecutive frames and (ii) a token reduction module that enhances processing speed by selectively discarding temporally redundant information while reusing weights of a pretrained foundation model. Second, we propose a lightweight distillation strategy to approximate the frame-level features of the original foundation model. Finally, we evaluate our approach across four tasks and five datasets, in both zero-shot and fully supervised settings, demonstrating significant reductions in computational cost (up to 60%) and improvements in inference speed (up to 2.5x faster), all while closely approximating the accuracy of the original foundation model.
>
---
#### [new 073] MFAF: An EVA02-Based Multi-scale Frequency Attention Fusion Method for Cross-View Geo-Localization
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出MFAF方法，用于跨视角地理定位任务，解决视角变化导致的特征提取困难问题。通过多尺度频率注意力融合，提升特征一致性与鲁棒性，实验表明其在无人机定位与导航中表现优异。**

- **链接: [http://arxiv.org/pdf/2509.12673v1](http://arxiv.org/pdf/2509.12673v1)**

> **作者:** YiTong Liu; TianZhu Liu; YanFeng GU
>
> **备注:** 17 pages, 13 figures
>
> **摘要:** Cross-view geo-localization aims to determine the geographical location of a query image by matching it against a gallery of images. This task is challenging due to the significant appearance variations of objects observed from variable views, along with the difficulty in extracting discriminative features. Existing approaches often rely on extracting features through feature map segmentation while neglecting spatial and semantic information. To address these issues, we propose the EVA02-based Multi-scale Frequency Attention Fusion (MFAF) method. The MFAF method consists of Multi-Frequency Branch-wise Block (MFB) and the Frequency-aware Spatial Attention (FSA) module. The MFB block effectively captures both low-frequency structural features and high-frequency edge details across multiple scales, improving the consistency and robustness of feature representations across various viewpoints. Meanwhile, the FSA module adaptively focuses on the key regions of frequency features, significantly mitigating the interference caused by background noise and viewpoint variability. Extensive experiments on widely recognized benchmarks, including University-1652, SUES-200, and Dense-UAV, demonstrate that the MFAF method achieves competitive performance in both drone localization and drone navigation tasks.
>
---
#### [new 074] AREPAS: Anomaly Detection in Fine-Grained Anatomy with Reconstruction-Based Semantic Patch-Scoring
- **分类: cs.CV**

- **简介: 该论文提出AREPAS方法，用于医学图像中的异常检测与分割。针对细粒度组织变异带来的挑战，采用生成重建与补丁相似性评分，提升病变定位精度。在CT和MRI数据上验证，显著提高DICE分数。属于无监督异常检测任务。**

- **链接: [http://arxiv.org/pdf/2509.12905v1](http://arxiv.org/pdf/2509.12905v1)**

> **作者:** Branko Mitic; Philipp Seeböck; Helmut Prosch; Georg Langs
>
> **摘要:** Early detection of newly emerging diseases, lesion severity assessment, differentiation of medical conditions and automated screening are examples for the wide applicability and importance of anomaly detection (AD) and unsupervised segmentation in medicine. Normal fine-grained tissue variability such as present in pulmonary anatomy is a major challenge for existing generative AD methods. Here, we propose a novel generative AD approach addressing this issue. It consists of an image-to-image translation for anomaly-free reconstruction and a subsequent patch similarity scoring between observed and generated image-pairs for precise anomaly localization. We validate the new method on chest computed tomography (CT) scans for the detection and segmentation of infectious disease lesions. To assess generalizability, we evaluate the method on an ischemic stroke lesion segmentation task in T1-weighted brain MRI. Results show improved pixel-level anomaly segmentation in both chest CTs and brain MRIs, with relative DICE score improvements of +1.9% and +4.4%, respectively, compared to other state-of-the-art reconstruction-based methods.
>
---
#### [new 075] HERO: Rethinking Visual Token Early Dropping in High-Resolution Large Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文针对高分辨率视觉语言模型（HR-LVLMs）中视觉token过多导致的计算与内存负担问题，提出HERO框架，通过内容自适应的token分配与功能感知选择，实现高效推理，提升效率-精度平衡。**

- **链接: [http://arxiv.org/pdf/2509.13067v1](http://arxiv.org/pdf/2509.13067v1)**

> **作者:** Xu Li; Yuxuan Liang; Xiaolei Chen; Yi Zheng; Haotian Chen; Bin Li; Xiangyang Xue
>
> **摘要:** By cropping high-resolution images into local tiles and encoding them independently, High-Resolution Large Vision-Language Models (HR-LVLMs) have demonstrated remarkable fine-grained visual understanding capabilities. However, this divide-and-conquer paradigm significantly increases the number of visual tokens, resulting in substantial computational and memory overhead. To better understand and address this challenge, we empirically investigate visual token utilization in HR-LVLMs and uncover three key findings: (1) the local tiles have varying importance, jointly determined by visual saliency and task relevance; (2) the CLS token in CLIP-based vision encoders exhibits a two-stage attention pattern across layers, with each stage attending to different types of visual tokens; (3) the visual tokens emphasized at different stages encode information at varying levels of granularity, playing complementary roles within LVLMs. Building on these insights, we propose HERO, a High-resolution visual token early dropping framework that integrates content-adaptive token budget allocation with function-aware token selection. By accurately estimating tile-level importance and selectively retaining visual tokens with complementary roles, HERO achieves superior efficiency-accuracy trade-offs across diverse benchmarks and model scales, all in a training-free manner. This study provides both empirical insights and practical solutions toward efficient inference in HR-LVLMs.
>
---
#### [new 076] Brought a Gun to a Knife Fight: Modern VFM Baselines Outgun Specialized Detectors on In-the-Wild AI Image Detection
- **分类: cs.CV**

- **简介: 论文研究AI生成图像检测任务，解决专用检测器在真实场景中表现差的问题。提出使用现代视觉基础模型（VFM）上的简单线性分类器作为基线，显著提升野外检测准确率，并分析其性能优势来源。**

- **链接: [http://arxiv.org/pdf/2509.12995v1](http://arxiv.org/pdf/2509.12995v1)**

> **作者:** Yue Zhou; Xinan He; Kaiqing Lin; Bing Fan; Feng Ding; Jinhua Zeng; Bin Li
>
> **摘要:** While specialized detectors for AI-generated images excel on curated benchmarks, they fail catastrophically in real-world scenarios, as evidenced by their critically high false-negative rates on `in-the-wild' benchmarks. Instead of crafting another specialized `knife' for this problem, we bring a `gun' to the fight: a simple linear classifier on a modern Vision Foundation Model (VFM). Trained on identical data, this baseline decisively `outguns' bespoke detectors, boosting in-the-wild accuracy by a striking margin of over 20\%. Our analysis pinpoints the source of the VFM's `firepower': First, by probing text-image similarities, we find that recent VLMs (e.g., Perception Encoder, Meta CLIP2) have learned to align synthetic images with forgery-related concepts (e.g., `AI-generated'), unlike previous versions. Second, we speculate that this is due to data exposure, as both this alignment and overall accuracy plummet on a novel dataset scraped after the VFM's pre-training cut-off date, ensuring it was unseen during pre-training. Our findings yield two critical conclusions: 1) For the real-world `gunfight' of AI-generated image detection, the raw `firepower' of an updated VFM is far more effective than the `craftsmanship' of a static detector. 2) True generalization evaluation requires test data to be independent of the model's entire training history, including pre-training.
>
---
#### [new 077] A Synthetic Data Pipeline for Supporting Manufacturing SMEs in Visual Assembly Control
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出一种基于合成数据的视觉装配控制方法，用于帮助制造中小型企业实现高效质量控制。通过CAD数据生成模拟场景并结合目标检测算法，减少数据采集与标注成本，实现实时高精度装配检测。**

- **链接: [http://arxiv.org/pdf/2509.13089v1](http://arxiv.org/pdf/2509.13089v1)**

> **作者:** Jonas Werheid; Shengjie He; Aymen Gannouni; Anas Abdelrazeq; Robert H. Schmitt
>
> **摘要:** Quality control of assembly processes is essential in manufacturing to ensure not only the quality of individual components but also their proper integration into the final product. To assist in this matter, automated assembly control using computer vision methods has been widely implemented. However, the costs associated with image acquisition, annotation, and training of computer vision algorithms pose challenges for integration, especially for small- and medium-sized enterprises (SMEs), which often lack the resources for extensive training, data collection, and manual image annotation. Synthetic data offers the potential to reduce manual data collection and labeling. Nevertheless, its practical application in the context of assembly quality remains limited. In this work, we present a novel approach for easily integrable and data-efficient visual assembly control. Our approach leverages simulated scene generation based on computer-aided design (CAD) data and object detection algorithms. The results demonstrate a time-saving pipeline for generating image data in manufacturing environments, achieving a mean Average Precision (mAP@0.5:0.95) up to 99,5% for correctly identifying instances of synthetic planetary gear system components within our simulated training data, and up to 93% when transferred to real-world camera-captured testing data. This research highlights the effectiveness of synthetic data generation within an adaptable pipeline and underscores its potential to support SMEs in implementing resource-efficient visual assembly control solutions.
>
---
#### [new 078] SHREC 2025: Protein surface shape retrieval including electrostatic potential
- **分类: cs.CV; q-bio.BM; I.3.8; I.5.4; J.3**

- **简介: 该论文属于蛋白质表面形状检索任务，旨在解决如何有效利用电势信息提升检索性能的问题。论文评估了15种方法在包含电势数据的11,555个蛋白表面数据集上的表现，发现结合电势的方法效果最佳。**

- **链接: [http://arxiv.org/pdf/2509.12976v1](http://arxiv.org/pdf/2509.12976v1)**

> **作者:** Taher Yacoub; Camille Depenveiller; Atsushi Tatsuma; Tin Barisin; Eugen Rusakov; Udo Gobel; Yuxu Peng; Shiqiang Deng; Yuki Kagaya; Joon Hong Park; Daisuke Kihara; Marco Guerra; Giorgio Palmieri; Andrea Ranieri; Ulderico Fugacci; Silvia Biasotti; Ruiwen He; Halim Benhabiles; Adnane Cabani; Karim Hammoudi; Haotian Li; Hao Huang; Chunyan Li; Alireza Tehrani; Fanwang Meng; Farnaz Heidar-Zadeh; Tuan-Anh Yang; Matthieu Montes
>
> **备注:** Published in Computers & Graphics, Elsevier. 59 pages, 12 figures
>
> **摘要:** This SHREC 2025 track dedicated to protein surface shape retrieval involved 9 participating teams. We evaluated the performance in retrieval of 15 proposed methods on a large dataset of 11,555 protein surfaces with calculated electrostatic potential (a key molecular surface descriptor). The performance in retrieval of the proposed methods was evaluated through different metrics (Accuracy, Balanced accuracy, F1 score, Precision and Recall). The best retrieval performance was achieved by the proposed methods that used the electrostatic potential complementary to molecular surface shape. This observation was also valid for classes with limited data which highlights the importance of taking into account additional molecular surface descriptors.
>
---
#### [new 079] EfficientNet-Based Multi-Class Detection of Real, Deepfake, and Plastic Surgery Faces
- **分类: cs.CV**

- **简介: 该论文属于多类人脸检测任务，旨在区分真实、深度伪造和整容面孔。研究基于EfficientNet模型，解决深度伪造技术带来的社会风险问题，提升人脸识别系统的安全性与可靠性。**

- **链接: [http://arxiv.org/pdf/2509.12258v1](http://arxiv.org/pdf/2509.12258v1)**

> **作者:** Li Kun; Milena Radenkovic
>
> **摘要:** Currently, deep learning has been utilised to tackle several difficulties in our everyday lives. It not only exhibits progress in computer vision but also constitutes the foundation for several revolutionary technologies. Nonetheless, similar to all phenomena, the use of deep learning in diverse domains has produced a multifaceted interaction of advantages and disadvantages for human society. Deepfake technology has advanced, significantly impacting social life. However, developments in this technology can affect privacy, the reputations of prominent personalities, and national security via software development. It can produce indistinguishable counterfeit photographs and films, potentially impairing the functionality of facial recognition systems, so presenting a significant risk. The improper application of deepfake technology produces several detrimental effects on society. Face-swapping programs mislead users by altering persons' appearances or expressions to fulfil particular aims or to appropriate personal information. Deepfake technology permeates daily life through such techniques. Certain individuals endeavour to sabotage election campaigns or subvert prominent political figures by creating deceptive pictures to influence public perception, causing significant harm to a nation's political and economic structure.
>
---
#### [new 080] VQT-Light:Lightweight HDR Illumination Map Prediction with Richer Texture.pdf
- **分类: cs.CV**

- **简介: 该论文提出VQT-Light框架，用于轻量级HDR光照图预测。针对现有方法在纹理细节和运行速度上的不足，结合VQVAE与ViT架构，提升光照估计的精度与效率，实现更丰富的纹理和更快的推理速度。**

- **链接: [http://arxiv.org/pdf/2509.12556v1](http://arxiv.org/pdf/2509.12556v1)**

> **作者:** Kunliang Xie
>
> **备注:** 11 pages, 8 figures
>
> **摘要:** Accurate lighting estimation is a significant yet challenging task in computer vision and graphics. However, existing methods either struggle to restore detailed textures of illumination map, or face challenges in run-ning speed and texture fidelity. To tackle this problem, we propose a novel framework (VQT-Light) based on VQVAE and ViT architecture. VQT-Light includes two modules: feature extraction and lighting estima-tion. First, we take advantages of VQVAE to extract discrete features of illumination map rather than con-tinuous features to avoid "posterior collapse". Second, we capture global context and dependencies of in-put image through ViT rather than CNNs to improve the prediction of illumination outside the field of view. Combining the above two modules, we formulate the lighting estimation as a multiclass classification task, which plays a key role in our pipeline. As a result, our model predicts light map with richer texture and better fidelity while keeping lightweight and fast. VQT-Light achieves an inference speed of 40FPS and im-proves multiple evaluation metrics. Qualitative and quantitative experiments demonstrate that the proposed method realizes superior results compared to existing state-of-the-art methods.
>
---
#### [new 081] Runge-Kutta Approximation and Decoupled Attention for Rectified Flow Inversion and Semantic Editing
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对Rectified Flow模型的图像重构与语义编辑任务，提出基于Runge-Kutta方法的高效逆过程和解耦注意力机制DDTA，以提升生成精度与语义控制能力。**

- **链接: [http://arxiv.org/pdf/2509.12888v1](http://arxiv.org/pdf/2509.12888v1)**

> **作者:** Weiming Chen; Zhihan Zhu; Yijia Wang; Zhihai He
>
> **摘要:** Rectified flow (RF) models have recently demonstrated superior generative performance compared to DDIM-based diffusion models. However, in real-world applications, they suffer from two major challenges: (1) low inversion accuracy that hinders the consistency with the source image, and (2) entangled multimodal attention in diffusion transformers, which hinders precise attention control. To address the first challenge, we propose an efficient high-order inversion method for rectified flow models based on the Runge-Kutta solver of differential equations. To tackle the second challenge, we introduce Decoupled Diffusion Transformer Attention (DDTA), a novel mechanism that disentangles text and image attention inside the multimodal diffusion transformers, enabling more precise semantic control. Extensive experiments on image reconstruction and text-guided editing tasks demonstrate that our method achieves state-of-the-art performance in terms of fidelity and editability. Code is available at https://github.com/wmchen/RKSovler_DDTA.
>
---
#### [new 082] MSDNet: Efficient 4D Radar Super-Resolution via Multi-Stage Distillation
- **分类: cs.CV**

- **简介: 该论文提出MSDNet，用于4D雷达点云超分辨率任务，解决现有方法计算成本高、推理延迟大等问题。通过多阶段知识蒸馏，高效融合LiDAR先验，实现高质量重建与低延迟推理。**

- **链接: [http://arxiv.org/pdf/2509.13149v1](http://arxiv.org/pdf/2509.13149v1)**

> **作者:** Minqing Huang; Shouyi Lu; Boyuan Zheng; Ziyao Li; Xiao Tang; Guirong Zhuo
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** 4D radar super-resolution, which aims to reconstruct sparse and noisy point clouds into dense and geometrically consistent representations, is a foundational problem in autonomous perception. However, existing methods often suffer from high training cost or rely on complex diffusion-based sampling, resulting in high inference latency and poor generalization, making it difficult to balance accuracy and efficiency. To address these limitations, we propose MSDNet, a multi-stage distillation framework that efficiently transfers dense LiDAR priors to 4D radar features to achieve both high reconstruction quality and computational efficiency. The first stage performs reconstruction-guided feature distillation, aligning and densifying the student's features through feature reconstruction. In the second stage, we propose diffusion-guided feature distillation, which treats the stage-one distilled features as a noisy version of the teacher's representations and refines them via a lightweight diffusion network. Furthermore, we introduce a noise adapter that adaptively aligns the noise level of the feature with a predefined diffusion timestep, enabling a more precise denoising. Extensive experiments on the VoD and in-house datasets demonstrate that MSDNet achieves both high-fidelity reconstruction and low-latency inference in the task of 4D radar point cloud super-resolution, and consistently improves performance on downstream tasks. The code will be publicly available upon publication.
>
---
#### [new 083] Image Realness Assessment and Localization with Multimodal Features
- **分类: cs.CV; eess.IV**

- **简介: 该论文提出一种基于多模态特征的框架，用于评估AI生成图像的真实感并定位视觉不一致区域。通过视觉语言模型生成文本描述，替代人工标注，提升真实感预测与局部不一致识别效果。属于图像真实性评估任务，解决AI图像质量检测与改进问题。**

- **链接: [http://arxiv.org/pdf/2509.13289v1](http://arxiv.org/pdf/2509.13289v1)**

> **作者:** Lovish Kaushik; Agnij Biswas; Somdyuti Paul
>
> **摘要:** A reliable method of quantifying the perceptual realness of AI-generated images and identifying visually inconsistent regions is crucial for practical use of AI-generated images and for improving photorealism of generative AI via realness feedback during training. This paper introduces a framework that accomplishes both overall objective realness assessment and local inconsistency identification of AI-generated images using textual descriptions of visual inconsistencies generated by vision-language models trained on large datasets that serve as reliable substitutes for human annotations. Our results demonstrate that the proposed multimodal approach improves objective realness prediction performance and produces dense realness maps that effectively distinguish between realistic and unrealistic spatial regions.
>
---
#### [new 084] Improving Accuracy and Efficiency of Implicit Neural Representations: Making SIREN a WINNER
- **分类: cs.CV; cs.LG**

- **简介: 论文提出WINNER方法，改进SIREN网络的初始化方式，通过添加与目标信号频谱相关的高斯噪声，解决其频率支持不匹配导致的“频谱瓶颈”问题，提升音频、图像和3D形状拟合效果。属于神经表示学习任务。**

- **链接: [http://arxiv.org/pdf/2509.12980v1](http://arxiv.org/pdf/2509.12980v1)**

> **作者:** Hemanth Chandravamsi; Dhanush V. Shenoy; Steven H. Frankel
>
> **摘要:** We identify and address a fundamental limitation of sinusoidal representation networks (SIRENs), a class of implicit neural representations. SIRENs Sitzmann et al. (2020), when not initialized appropriately, can struggle at fitting signals that fall outside their frequency support. In extreme cases, when the network's frequency support misaligns with the target spectrum, a 'spectral bottleneck' phenomenon is observed, where the model yields to a near-zero output and fails to recover even the frequency components that are within its representational capacity. To overcome this, we propose WINNER - Weight Initialization with Noise for Neural Representations. WINNER perturbs uniformly initialized weights of base SIREN with Gaussian noise - whose noise scales are adaptively determined by the spectral centroid of the target signal. Similar to random Fourier embeddings, this mitigates 'spectral bias' but without introducing additional trainable parameters. Our method achieves state-of-the-art audio fitting and significant gains in image and 3D shape fitting tasks over base SIREN. Beyond signal fitting, WINNER suggests new avenues in adaptive, target-aware initialization strategies for optimizing deep neural network training. For code and data visit cfdlabtechnion.github.io/siren_square/.
>
---
#### [new 085] Enhancing Dual Network Based Semi-Supervised Medical Image Segmentation with Uncertainty-Guided Pseudo-Labeling
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决半监督学习中伪标签噪声和特征空间监督不足的问题。提出双网络框架，结合交叉一致性增强模块与不确定性引导的动态加权策略，提升分割性能。**

- **链接: [http://arxiv.org/pdf/2509.13084v1](http://arxiv.org/pdf/2509.13084v1)**

> **作者:** Yunyao Lu; Yihang Wu; Ahmad Chaddad; Tareef Daqqaq; Reem Kateb
>
> **备注:** Accpeted in Knowledge-Based Systems
>
> **摘要:** Despite the remarkable performance of supervised medical image segmentation models, relying on a large amount of labeled data is impractical in real-world situations. Semi-supervised learning approaches aim to alleviate this challenge using unlabeled data through pseudo-label generation. Yet, existing semi-supervised segmentation methods still suffer from noisy pseudo-labels and insufficient supervision within the feature space. To solve these challenges, this paper proposes a novel semi-supervised 3D medical image segmentation framework based on a dual-network architecture. Specifically, we investigate a Cross Consistency Enhancement module using both cross pseudo and entropy-filtered supervision to reduce the noisy pseudo-labels, while we design a dynamic weighting strategy to adjust the contributions of pseudo-labels using an uncertainty-aware mechanism (i.e., Kullback-Leibler divergence). In addition, we use a self-supervised contrastive learning mechanism to align uncertain voxel features with reliable class prototypes by effectively differentiating between trustworthy and uncertain predictions, thus reducing prediction uncertainty. Extensive experiments are conducted on three 3D segmentation datasets, Left Atrial, NIH Pancreas and BraTS-2019. The proposed approach consistently exhibits superior performance across various settings (e.g., 89.95\% Dice score on left Atrial with 10\% labeled data) compared to the state-of-the-art methods. Furthermore, the usefulness of the proposed modules is further validated via ablation experiments.
>
---
#### [new 086] CECT-Mamba: a Hierarchical Contrast-enhanced-aware Model for Pancreatic Tumor Subtyping from Multi-phase CECT
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CECT-Mamba模型，用于胰腺肿瘤亚型分类任务，解决多期增强CT图像中肿瘤异质性带来的分类难题。通过引入Mamba结构和时空对比感知模块，有效融合多期影像信息，实现高准确率的PDAC与PNETs区分。**

- **链接: [http://arxiv.org/pdf/2509.12777v1](http://arxiv.org/pdf/2509.12777v1)**

> **作者:** Zhifang Gong; Shuo Gao; Ben Zhao; Yingjing Xu; Yijun Yang; Shenghong Ju; Guangquan Zhou
>
> **摘要:** Contrast-enhanced computed tomography (CECT) is the primary imaging technique that provides valuable spatial-temporal information about lesions, enabling the accurate diagnosis and subclassification of pancreatic tumors. However, the high heterogeneity and variability of pancreatic tumors still pose substantial challenges for precise subtyping diagnosis. Previous methods fail to effectively explore the contextual information across multiple CECT phases commonly used in radiologists' diagnostic workflows, thereby limiting their performance. In this paper, we introduce, for the first time, an automatic way to combine the multi-phase CECT data to discriminate between pancreatic tumor subtypes, among which the key is using Mamba with promising learnability and simplicity to encourage both temporal and spatial modeling from multi-phase CECT. Specifically, we propose a dual hierarchical contrast-enhanced-aware Mamba module incorporating two novel spatial and temporal sampling sequences to explore intra and inter-phase contrast variations of lesions. A similarity-guided refinement module is also imposed into the temporal scanning modeling to emphasize the learning on local tumor regions with more obvious temporal variations. Moreover, we design the space complementary integrator and multi-granularity fusion module to encode and aggregate the semantics across different scales, achieving more efficient learning for subtyping pancreatic tumors. The experimental results on an in-house dataset of 270 clinical cases achieve an accuracy of 97.4% and an AUC of 98.6% in distinguishing between pancreatic ductal adenocarcinoma (PDAC) and pancreatic neuroendocrine tumors (PNETs), demonstrating its potential as a more accurate and efficient tool.
>
---
#### [new 087] Modelling and analysis of the 8 filters from the "master key filters hypothesis" for depthwise-separable deep networks in relation to idealized receptive fields based on scale-space theory
- **分类: cs.CV**

- **简介: 论文分析8种“主键滤波器”，建模深度可分离网络的接收域。基于尺度空间理论，通过聚类和差分算子建模，验证其与理想化滤波器的相似性，证明可替换为离散尺度空间滤波器，提升模型解释性与效率。**

- **链接: [http://arxiv.org/pdf/2509.12746v1](http://arxiv.org/pdf/2509.12746v1)**

> **作者:** Tony Lindeberg; Zahra Babaiee; Peyman M. Kiasari
>
> **备注:** 24 pages, 11 figures, 17 tables
>
> **摘要:** This paper presents the results of analysing and modelling a set of 8 ``master key filters'', which have been extracted by applying a clustering approach to the receptive fields learned in depthwise-separable deep networks based on the ConvNeXt architecture. For this purpose, we first compute spatial spread measures in terms of weighted mean values and weighted variances of the absolute values of the learned filters, which support the working hypotheses that: (i) the learned filters can be modelled by separable filtering operations over the spatial domain, and that (ii) the spatial offsets of the those learned filters that are non-centered are rather close to half a grid unit. Then, we model the clustered ``master key filters'' in terms of difference operators applied to a spatial smoothing operation in terms of the discrete analogue of the Gaussian kernel, and demonstrate that the resulting idealized models of the receptive fields show good qualitative similarity to the learned filters. This modelling is performed in two different ways: (i) using possibly different values of the scale parameters in the coordinate directions for each filter, and (ii) using the same value of the scale parameter in both coordinate directions. Then, we perform the actual model fitting by either (i) requiring spatial spread measures in terms of spatial variances of the absolute values of the receptive fields to be equal, or (ii) minimizing the discrete $l_1$- or $l_2$-norms between the idealized receptive field models and the learned filters. Complementary experimental results then demonstrate the idealized models of receptive fields have good predictive properties for replacing the learned filters by idealized filters in depthwise-separable deep networks, thus showing that the learned filters in depthwise-separable deep networks can be well approximated by discrete scale-space filters.
>
---
#### [new 088] RIS-FUSION: Rethinking Text-Driven Infrared and Visible Image Fusion from the Perspective of Referring Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于文本驱动的红外与可见光图像融合任务，旨在解决文本指导不足的问题。提出RIS-FUSION框架，结合语义分割，提升融合效果，并构建MM-RIS数据集，实现性能提升。**

- **链接: [http://arxiv.org/pdf/2509.12710v1](http://arxiv.org/pdf/2509.12710v1)**

> **作者:** Siju Ma; Changsiyu Gong; Xiaofeng Fan; Yong Ma; Chengjie Jiang
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Text-driven infrared and visible image fusion has gained attention for enabling natural language to guide the fusion process. However, existing methods lack a goal-aligned task to supervise and evaluate how effectively the input text contributes to the fusion outcome. We observe that referring image segmentation (RIS) and text-driven fusion share a common objective: highlighting the object referred to by the text. Motivated by this, we propose RIS-FUSION, a cascaded framework that unifies fusion and RIS through joint optimization. At its core is the LangGatedFusion module, which injects textual features into the fusion backbone to enhance semantic alignment. To support multimodal referring image segmentation task, we introduce MM-RIS, a large-scale benchmark with 12.5k training and 3.5k testing triplets, each consisting of an infrared-visible image pair, a segmentation mask, and a referring expression. Extensive experiments show that RIS-FUSION achieves state-of-the-art performance, outperforming existing methods by over 11% in mIoU. Code and dataset will be released at https://github.com/SijuMa2003/RIS-FUSION.
>
---
#### [new 089] Superpixel Anything: A general object-based framework for accurate yet regular superpixel segmentation
- **分类: cs.CV**

- **简介: 该论文提出SPAM框架，解决传统超像素分割准确性和规则性难以兼顾的问题。通过结合深度学习与预训练模型，实现高精度且规则的超像素分割，适用于多种分割任务。**

- **链接: [http://arxiv.org/pdf/2509.12791v1](http://arxiv.org/pdf/2509.12791v1)**

> **作者:** Julien Walther; Rémi Giraud; Michaël Clément
>
> **摘要:** Superpixels are widely used in computer vision to simplify image representation and reduce computational complexity. While traditional methods rely on low-level features, deep learning-based approaches leverage high-level features but also tend to sacrifice regularity of superpixels to capture complex objects, leading to accurate but less interpretable segmentations. In this work, we introduce SPAM (SuperPixel Anything Model), a versatile framework for segmenting images into accurate yet regular superpixels. We train a model to extract image features for superpixel generation, and at inference, we leverage a large-scale pretrained model for semantic-agnostic segmentation to ensure that superpixels align with object masks. SPAM can handle any prior high-level segmentation, resolving uncertainty regions, and is able to interactively focus on specific objects. Comprehensive experiments demonstrate that SPAM qualitatively and quantitatively outperforms state-of-the-art methods on segmentation tasks, making it a valuable and robust tool for various applications. Code and pre-trained models are available here: https://github.com/waldo-j/spam.
>
---
#### [new 090] Road Obstacle Video Segmentation
- **分类: cs.CV**

- **简介: 该论文属于道路障碍物视频分割任务，旨在解决传统方法忽略时间相关性导致的帧间预测不一致问题。论文构建了四个评估基准，测试11种方法，并提出两种基于视觉基础模型的强基线方法，实现了长序列视频分割的新SOTA。**

- **链接: [http://arxiv.org/pdf/2509.13181v1](http://arxiv.org/pdf/2509.13181v1)**

> **作者:** Shyam Nandan Rai; Shyamgopal Karthik; Mariana-Iuliana Georgescu; Barbara Caputo; Carlo Masone; Zeynep Akata
>
> **备注:** GCPR 2025
>
> **摘要:** With the growing deployment of autonomous driving agents, the detection and segmentation of road obstacles have become critical to ensure safe autonomous navigation. However, existing road-obstacle segmentation methods are applied on individual frames, overlooking the temporal nature of the problem, leading to inconsistent prediction maps between consecutive frames. In this work, we demonstrate that the road-obstacle segmentation task is inherently temporal, since the segmentation maps for consecutive frames are strongly correlated. To address this, we curate and adapt four evaluation benchmarks for road-obstacle video segmentation and evaluate 11 state-of-the-art image- and video-based segmentation methods on these benchmarks. Moreover, we introduce two strong baseline methods based on vision foundation models. Our approach establishes a new state-of-the-art in road-obstacle video segmentation for long-range video sequences, providing valuable insights and direction for future research.
>
---
#### [new 091] Beyond Artificial Misalignment: Detecting and Grounding Semantic-Coordinated Multimodal Manipulations
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态内容检测任务，旨在解决现有数据集因人工对齐破坏导致的检测偏差问题。论文构建了语义一致的SAMM数据集，并提出RamDG框架，通过检索增强实现更准确的多模态操纵检测与定位。**

- **链接: [http://arxiv.org/pdf/2509.12653v1](http://arxiv.org/pdf/2509.12653v1)**

> **作者:** Jinjie Shen; Yaxiong Wang; Lechao Cheng; Nan Pu; Zhun Zhong
>
> **摘要:** The detection and grounding of manipulated content in multimodal data has emerged as a critical challenge in media forensics. While existing benchmarks demonstrate technical progress, they suffer from misalignment artifacts that poorly reflect real-world manipulation patterns: practical attacks typically maintain semantic consistency across modalities, whereas current datasets artificially disrupt cross-modal alignment, creating easily detectable anomalies. To bridge this gap, we pioneer the detection of semantically-coordinated manipulations where visual edits are systematically paired with semantically consistent textual descriptions. Our approach begins with constructing the first Semantic-Aligned Multimodal Manipulation (SAMM) dataset, generated through a two-stage pipeline: 1) applying state-of-the-art image manipulations, followed by 2) generation of contextually-plausible textual narratives that reinforce the visual deception. Building on this foundation, we propose a Retrieval-Augmented Manipulation Detection and Grounding (RamDG) framework. RamDG commences by harnessing external knowledge repositories to retrieve contextual evidence, which serves as the auxiliary texts and encoded together with the inputs through our image forgery grounding and deep manipulation detection modules to trace all manipulations. Extensive experiments demonstrate our framework significantly outperforms existing methods, achieving 2.06\% higher detection accuracy on SAMM compared to state-of-the-art approaches. The dataset and code are publicly available at https://github.com/shen8424/SAMM-RamDG-CAP.
>
---
#### [new 092] PATIMT-Bench: A Multi-Scenario Benchmark for Position-Aware Text Image Machine Translation in Large Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出PATIMT-Bench，解决传统TIMT忽视文本位置信息的问题。其任务是实现位置感知的图文翻译，包含区域翻译与全图定位翻译。构建了10种场景的基准数据集，并引入OCR优化流程，提升模型在多场景下的翻译性能。**

- **链接: [http://arxiv.org/pdf/2509.12278v1](http://arxiv.org/pdf/2509.12278v1)**

> **作者:** Wanru Zhuang; Wenbo Li; Zhibin Lan; Xu Han; Peng Li; Jinsong Su
>
> **摘要:** Text Image Machine Translation (TIMT) aims to translate texts embedded within an image into another language. Current TIMT studies primarily focus on providing translations for all the text within an image, while neglecting to provide bounding boxes and covering limited scenarios. In this work, we extend traditional TIMT into position-aware TIMT (PATIMT), aiming to support fine-grained and layoutpreserving translation, which holds great practical value but remains largely unexplored. This task comprises two key sub-tasks: regionspecific translation and full-image translation with grounding. To support existing models on PATIMT and conduct fair evaluation, we construct the PATIMT benchmark (PATIMTBench), which consists of 10 diverse real-world scenarios. Specifically, we introduce an Adaptive Image OCR Refinement Pipeline, which adaptively selects appropriate OCR tools based on scenario and refines the results of text-rich images. To ensure evaluation reliability, we further construct a test set, which contains 1,200 high-quality instances manually annotated and reviewed by human experts. After fine-tuning on our data, compact Large Vision-Language Models (LVLMs) achieve state-of-the-art performance on both sub-tasks. Experimental results also highlight the scalability and generalizability of our training data
>
---
#### [new 093] End4: End-to-end Denoising Diffusion for Diffusion-Based Inpainting Detection
- **分类: cs.CV**

- **简介: 该论文提出End4方法，用于检测基于扩散模型的图像修复生成结果。针对现有方法难以识别此类图像的问题，设计端到端去噪扩散模型与多尺度特征融合模块，提升检测性能，并构建了包含五类掩码区域的基准数据集。**

- **链接: [http://arxiv.org/pdf/2509.13214v1](http://arxiv.org/pdf/2509.13214v1)**

> **作者:** Fei Wang; Xuecheng Wu; Zheng Zhang; Danlei Huang; Yuheng Huang; BoWang
>
> **摘要:** The powerful generative capabilities of diffusion models have significantly advanced the field of image synthesis, enhancing both full image generation and inpainting-based image editing. Despite their remarkable advancements, diffusion models also raise concerns about potential misuse for malicious purposes. However, existing approaches struggle to identify images generated by diffusion-based inpainting models, even when similar inpainted images are included in their training data. To address this challenge, we propose a novel detection method based on End-to-end denoising diffusion (End4). Specifically, End4 designs a denoising reconstruction model to improve the alignment degree between the latent spaces of the reconstruction and detection processes, thus reconstructing features that are more conducive to detection. Meanwhile, it leverages a Scale-aware Pyramid-like Fusion Module (SPFM) that refines local image features under the guidance of attention pyramid layers at different scales, enhancing feature discriminability. Additionally, to evaluate detection performance on inpainted images, we establish a comprehensive benchmark comprising images generated from five distinct masked regions. Extensive experiments demonstrate that our End4 effectively generalizes to unseen masking patterns and remains robust under various perturbations. Our code and dataset will be released soon.
>
---
#### [new 094] Beyond Averages: Open-Vocabulary 3D Scene Understanding with Gaussian Splatting and Bag of Embeddings
- **分类: cs.CV**

- **简介: 论文提出一种基于高斯泼溅和嵌入包的开放词汇3D场景理解方法，解决传统方法因模糊性导致的语义理解不足问题。通过多视角CLIP特征聚合生成物体级嵌入，实现精准开放词汇检索与任务适配，提升3D场景理解能力。**

- **链接: [http://arxiv.org/pdf/2509.12938v1](http://arxiv.org/pdf/2509.12938v1)**

> **作者:** Abdalla Arafa; Didier Stricker
>
> **摘要:** Novel view synthesis has seen significant advancements with 3D Gaussian Splatting (3DGS), enabling real-time photorealistic rendering. However, the inherent fuzziness of Gaussian Splatting presents challenges for 3D scene understanding, restricting its broader applications in AR/VR and robotics. While recent works attempt to learn semantics via 2D foundation model distillation, they inherit fundamental limitations: alpha blending averages semantics across objects, making 3D-level understanding impossible. We propose a paradigm-shifting alternative that bypasses differentiable rendering for semantics entirely. Our key insight is to leverage predecomposed object-level Gaussians and represent each object through multiview CLIP feature aggregation, creating comprehensive "bags of embeddings" that holistically describe objects. This allows: (1) accurate open-vocabulary object retrieval by comparing text queries to object-level (not Gaussian-level) embeddings, and (2) seamless task adaptation: propagating object IDs to pixels for 2D segmentation or to Gaussians for 3D extraction. Experiments demonstrate that our method effectively overcomes the challenges of 3D open-vocabulary object extraction while remaining comparable to state-of-the-art performance in 2D open-vocabulary segmentation, ensuring minimal compromise.
>
---
#### [new 095] Double Helix Diffusion for Cross-Domain Anomaly Image Generation
- **分类: cs.CV**

- **简介: 论文提出DH-Diff框架，用于跨域异常图像生成，解决合成图像结构不一致与特征纠缠问题。通过双螺旋架构与语义对齐模块，提升生成图像真实性与多样性，优化下游异常检测性能。属于图像生成与缺陷检测任务。**

- **链接: [http://arxiv.org/pdf/2509.12787v1](http://arxiv.org/pdf/2509.12787v1)**

> **作者:** Linchun Wu; Qin Zou; Xianbiao Qi; Bo Du; Zhongyuan Wang; Qingquan Li
>
> **摘要:** Visual anomaly inspection is critical in manufacturing, yet hampered by the scarcity of real anomaly samples for training robust detectors. Synthetic data generation presents a viable strategy for data augmentation; however, current methods remain constrained by two principal limitations: 1) the generation of anomalies that are structurally inconsistent with the normal background, and 2) the presence of undesirable feature entanglement between synthesized images and their corresponding annotation masks, which undermines the perceptual realism of the output. This paper introduces Double Helix Diffusion (DH-Diff), a novel cross-domain generative framework designed to simultaneously synthesize high-fidelity anomaly images and their pixel-level annotation masks, explicitly addressing these challenges. DH-Diff employs a unique architecture inspired by a double helix, cycling through distinct modules for feature separation, connection, and merging. Specifically, a domain-decoupled attention mechanism mitigates feature entanglement by enhancing image and annotation features independently, and meanwhile a semantic score map alignment module ensures structural authenticity by coherently integrating anomaly foregrounds. DH-Diff offers flexible control via text prompts and optional graphical guidance. Extensive experiments demonstrate that DH-Diff significantly outperforms state-of-the-art methods in diversity and authenticity, leading to significant improvements in downstream anomaly detection performance.
>
---
#### [new 096] OnlineHOI: Towards Online Human-Object Interaction Generation and Perception
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出OnlineHOI框架，解决在线人类-物体交互生成与感知任务。传统方法依赖离线数据，而在线场景需实时处理当前及历史信息。本文基于Mamba架构引入记忆机制，实现在线HOI生成与感知的前沿成果。**

- **链接: [http://arxiv.org/pdf/2509.12250v1](http://arxiv.org/pdf/2509.12250v1)**

> **作者:** Yihong Ji; Yunze Liu; Yiyao Zhuo; Weijiang Yu; Fei Ma; Joshua Huang; Fei Yu
>
> **备注:** Accepted at ACM MM 2025
>
> **摘要:** The perception and generation of Human-Object Interaction (HOI) are crucial for fields such as robotics, AR/VR, and human behavior understanding. However, current approaches model this task in an offline setting, where information at each time step can be drawn from the entire interaction sequence. In contrast, in real-world scenarios, the information available at each time step comes only from the current moment and historical data, i.e., an online setting. We find that offline methods perform poorly in an online context. Based on this observation, we propose two new tasks: Online HOI Generation and Perception. To address this task, we introduce the OnlineHOI framework, a network architecture based on the Mamba framework that employs a memory mechanism. By leveraging Mamba's powerful modeling capabilities for streaming data and the Memory mechanism's efficient integration of historical information, we achieve state-of-the-art results on the Core4D and OAKINK2 online generation tasks, as well as the online HOI4D perception task.
>
---
#### [new 097] Exploring Spectral Characteristics for Single Image Reflection Removal
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，旨在解决单幅图像反射去除问题。提出谱码本和谱感知Transformer，利用光谱特性区分反射与背景，提升去除效果。**

- **链接: [http://arxiv.org/pdf/2509.12627v1](http://arxiv.org/pdf/2509.12627v1)**

> **作者:** Pengbo Guo; Chengxu Liu; Guoshuai Zhao; Xingsong Hou; Jialie Shen; Xueming Qian
>
> **摘要:** Eliminating reflections caused by incident light interacting with reflective medium remains an ill-posed problem in the image restoration area. The primary challenge arises from the overlapping of reflection and transmission components in the captured images, which complicates the task of accurately distinguishing and recovering the clean background. Existing approaches typically address reflection removal solely in the image domain, ignoring the spectral property variations of reflected light, which hinders their ability to effectively discern reflections. In this paper, we start with a new perspective on spectral learning, and propose the Spectral Codebook to reconstruct the optical spectrum of the reflection image. The reflections can be effectively distinguished by perceiving the wavelength differences between different light sources in the spectrum. To leverage the reconstructed spectrum, we design two spectral prior refinement modules to re-distribute pixels in the spatial dimension and adaptively enhance the spectral differences along the wavelength dimension. Furthermore, we present the Spectrum-Aware Transformer to jointly recover the transmitted content in spectral and pixel domains. Experimental results on three different reflection benchmarks demonstrate the superiority and generalization ability of our method compared to state-of-the-art models.
>
---
#### [new 098] 4DRadar-GS: Self-Supervised Dynamic Driving Scene Reconstruction with 4D Radar
- **分类: cs.CV**

- **简介: 该论文提出4DRadar-GS框架，解决动态驾驶场景中3D重建问题。利用4D雷达信息初始化高斯点，并设计Velocity-guided PointTrack模型提升动态目标跟踪与时间一致性，实现高质量动态场景重建。**

- **链接: [http://arxiv.org/pdf/2509.12931v1](http://arxiv.org/pdf/2509.12931v1)**

> **作者:** Xiao Tang; Guirong Zhuo; Cong Wang; Boyuan Zheng; Minqing Huang; Lianqing Zheng; Long Chen; Shouyi Lu
>
> **摘要:** 3D reconstruction and novel view synthesis are critical for validating autonomous driving systems and training advanced perception models. Recent self-supervised methods have gained significant attention due to their cost-effectiveness and enhanced generalization in scenarios where annotated bounding boxes are unavailable. However, existing approaches, which often rely on frequency-domain decoupling or optical flow, struggle to accurately reconstruct dynamic objects due to imprecise motion estimation and weak temporal consistency, resulting in incomplete or distorted representations of dynamic scene elements. To address these challenges, we propose 4DRadar-GS, a 4D Radar-augmented self-supervised 3D reconstruction framework tailored for dynamic driving scenes. Specifically, we first present a 4D Radar-assisted Gaussian initialization scheme that leverages 4D Radar's velocity and spatial information to segment dynamic objects and recover monocular depth scale, generating accurate Gaussian point representations. In addition, we propose a Velocity-guided PointTrack (VGPT) model, which is jointly trained with the reconstruction pipeline under scene flow supervision, to track fine-grained dynamic trajectories and construct temporally consistent representations. Evaluated on the OmniHD-Scenes dataset, 4DRadar-GS achieves state-of-the-art performance in dynamic driving scene 3D reconstruction.
>
---
#### [new 099] SAGA: Selective Adaptive Gating for Efficient and Expressive Linear Attention
- **分类: cs.CV**

- **简介: 该论文提出SAGA方法，用于改进线性注意力机制，解决传统方法因均匀压缩KV信息导致的特征冗余和性能下降问题。通过引入自适应门控机制，提升语义多样性与模型表现，实现高效且表达力强的视觉任务处理。**

- **链接: [http://arxiv.org/pdf/2509.12817v1](http://arxiv.org/pdf/2509.12817v1)**

> **作者:** Yuan Cao; Dong Wang
>
> **摘要:** While Transformer architecture excel at modeling long-range dependencies contributing to its widespread adoption in vision tasks the quadratic complexity of softmax-based attention mechanisms imposes a major bottleneck, particularly when processing high-resolution images. Linear attention presents a promising alternative by reformulating the attention computation from $(QK)V$ to $Q(KV)$, thereby reducing the complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$ while preserving the global receptive field. However, most existing methods compress historical key-value (KV) information uniformly, which can lead to feature redundancy and the loss of directional alignment with the query (Q). This uniform compression results in low-rank $KV$ feature maps, contributing to a performance gap compared to softmax attention. To mitigate this limitation, we propose \textbf{S}elective \textbf{A}daptive \textbf{GA}ting for Efficient and Expressive Linear Attention (SAGA) , which introduces input-adaptive learnable gates to selectively modulate information aggregation into the $KV$ feature map. These gates enhance semantic diversity and alleviate the low-rank constraint inherent in conventional linear attention. Additionally, we propose an efficient Hadamard-product decomposition method for gate computation, which introduces no additional memory overhead. Experiments demonstrate that SAGA achieves a 1.76$\times$ improvement in throughput and a 2.69$\times$ reduction in peak GPU memory compared to PVT-T at a resolution of $1280 \times 1280$. Moreover, it improves top-1 accuracy by up to 4.4\% on the ImageNet dataset, demonstrating both computational efficiency and model effectiveness.
>
---
#### [new 100] Cumulative Consensus Score: Label-Free and Model-Agnostic Evaluation of Object Detectors in Deployment
- **分类: cs.CV**

- **简介: 论文提出Cumulative Consensus Score（CCS），一种无需标签的模型评估方法，用于部署环境中目标检测器的持续监控。该方法通过数据增强和预测框重叠计算，实现无标注的可靠性评估，适用于多种检测模型，解决实际部署中缺乏真实标签的问题。**

- **链接: [http://arxiv.org/pdf/2509.12871v1](http://arxiv.org/pdf/2509.12871v1)**

> **作者:** Avinaash Manoharan; Xiangyu Yin; Domenik Helm; Chih-Hong Cheng
>
> **摘要:** Evaluating object detection models in deployment is challenging because ground-truth annotations are rarely available. We introduce the Cumulative Consensus Score (CCS), a label-free metric that enables continuous monitoring and comparison of detectors in real-world settings. CCS applies test-time data augmentation to each image, collects predicted bounding boxes across augmented views, and computes overlaps using Intersection over Union. Maximum overlaps are normalized and averaged across augmentation pairs, yielding a measure of spatial consistency that serves as a proxy for reliability without annotations. In controlled experiments on Open Images and KITTI, CCS achieved over 90% congruence with F1-score, Probabilistic Detection Quality, and Optimal Correction Cost. The method is model-agnostic, working across single-stage and two-stage detectors, and operates at the case level to highlight under-performing scenarios. Altogether, CCS provides a robust foundation for DevOps-style monitoring of object detectors.
>
---
#### [new 101] Perception Before Reasoning: Two-Stage Reinforcement Learning for Visual Reasoning in Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型的推理任务，旨在提升模型的感知与推理能力。针对VLM任务复杂、需先准确感知再推理的问题，提出两阶段强化学习框架，分阶段增强视觉感知与推理能力，最终获得性能优越的PeBR-R1模型。**

- **链接: [http://arxiv.org/pdf/2509.13031v1](http://arxiv.org/pdf/2509.13031v1)**

> **作者:** Yan Chen; Long Li; Teng Xi; Long Zeng; Jingdong Wang
>
> **摘要:** Reinforcement learning (RL) has proven highly effective in eliciting the reasoning capabilities of large language models (LLMs). Inspired by this success, recent studies have explored applying similar techniques to vision-language models (VLMs), aiming to enhance their reasoning performance. However, directly transplanting RL methods from LLMs to VLMs is suboptimal, as the tasks faced by VLMs are inherently more complex. Specifically, VLMs must first accurately perceive and understand visual inputs before reasoning can be effectively performed. To address this challenge, we propose a two-stage reinforcement learning framework designed to jointly enhance both the perceptual and reasoning capabilities of VLMs. To mitigate the vanishing advantage issue commonly observed in RL training, we first perform dataset-level sampling to selectively strengthen specific capabilities using distinct data sources. During training, the first stage focuses on improving the model's visual perception through coarse- and fine-grained visual understanding, while the second stage targets the enhancement of reasoning abilities. After the proposed two-stage reinforcement learning process, we obtain PeBR-R1, a vision-language model with significantly enhanced perceptual and reasoning capabilities. Experimental results on seven benchmark datasets demonstrate the effectiveness of our approach and validate the superior performance of PeBR-R1 across diverse visual reasoning tasks.
>
---
#### [new 102] GhostNetV3-Small: A Tailored Architecture and Comparative Study of Distillation Strategies for Tiny Images
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出GhostNetV3-Small，优化小规模图像分类模型。针对边缘设备计算限制，研究模型压缩与知识蒸馏策略，发现架构调整比蒸馏更有效。**

- **链接: [http://arxiv.org/pdf/2509.12380v1](http://arxiv.org/pdf/2509.12380v1)**

> **作者:** Florian Zager; Hamza A. A. Gardi
>
> **摘要:** Deep neural networks have achieved remarkable success across a range of tasks, however their computational demands often make them unsuitable for deployment on resource-constrained edge devices. This paper explores strategies for compressing and adapting models to enable efficient inference in such environments. We focus on GhostNetV3, a state-of-the-art architecture for mobile applications, and propose GhostNetV3-Small, a modified variant designed to perform better on low-resolution inputs such as those in the CIFAR-10 dataset. In addition to architectural adaptation, we provide a comparative evaluation of knowledge distillation techniques, including traditional knowledge distillation, teacher assistants, and teacher ensembles. Experimental results show that GhostNetV3-Small significantly outperforms the original GhostNetV3 on CIFAR-10, achieving an accuracy of 93.94%. Contrary to expectations, all examined distillation strategies led to reduced accuracy compared to baseline training. These findings indicate that architectural adaptation can be more impactful than distillation in small-scale image classification tasks, highlighting the need for further research on effective model design and advanced distillation techniques for low-resolution domains.
>
---
#### [new 103] Data Scaling Laws for Radiology Foundation Models
- **分类: cs.CV; cs.AI**

- **简介: 论文研究医学影像基础模型在数据规模与预训练方法下的性能差异。通过持续预训练两种视觉编码器（MI2和RAD-DINO），分析其在胸部X光任务中的表现，探索结构化监督与数据量对模型效果的影响，旨在提升医疗影像分析性能。**

- **链接: [http://arxiv.org/pdf/2509.12818v1](http://arxiv.org/pdf/2509.12818v1)**

> **作者:** Maximilian Ilse; Harshita Sharma; Anton Schwaighofer; Sam Bond-Taylor; Fernando Pérez-García; Olesya Melnichenko; Anne-Marie G. Sykes; Kelly K. Horst; Ashish Khandelwal; Maxwell Reynolds; Maria T. Wetscherek; Noel C. F. Codella; Javier Alvarez-Valle; Korfiatis Panagiotis; Valentina Salvatelli
>
> **摘要:** Foundation vision encoders such as CLIP and DINOv2, trained on web-scale data, exhibit strong transfer performance across tasks and datasets. However, medical imaging foundation models remain constrained by smaller datasets, limiting our understanding of how data scale and pretraining paradigms affect performance in this setting. In this work, we systematically study continual pretraining of two vision encoders, MedImageInsight (MI2) and RAD-DINO representing the two major encoder paradigms CLIP and DINOv2, on up to 3.5M chest x-rays from a single institution, holding compute and evaluation protocols constant. We evaluate on classification (radiology findings, lines and tubes), segmentation (lines and tubes), and radiology report generation. While prior work has primarily focused on tasks related to radiology findings, we include lines and tubes tasks to counterbalance this bias and evaluate a model's ability to extract features that preserve continuity along elongated structures. Our experiments show that MI2 scales more effectively for finding-related tasks, while RAD-DINO is stronger on tube-related tasks. Surprisingly, continually pretraining MI2 with both reports and structured labels using UniCL improves performance, underscoring the value of structured supervision at scale. We further show that for some tasks, as few as 30k in-domain samples are sufficient to surpass open-weights foundation models. These results highlight the utility of center-specific continual pretraining, enabling medical institutions to derive significant performance gains by utilizing in-domain data.
>
---
#### [new 104] Drone Detection Using a Low-Power Neuromorphic Virtual Tripwire
- **分类: cs.CV**

- **简介: 论文提出一种基于脉冲神经网络和类脑相机的低功耗无人机检测系统，用于构建虚拟警戒线。该系统能耗远低于传统GPU方案，适用于无电源或复杂环境部署，主要通过无人机外形特征实现检测。**

- **链接: [http://arxiv.org/pdf/2509.12997v1](http://arxiv.org/pdf/2509.12997v1)**

> **作者:** Anton Eldeborg Lundin; Rasmus Winzell; Hanna Hamrell; David Gustafsson; Hannes Ovrén
>
> **摘要:** Small drones are an increasing threat to both military personnel and civilian infrastructure, making early and automated detection crucial. In this work we develop a system that uses spiking neural networks and neuromorphic cameras (event cameras) to detect drones. The detection model is deployed on a neuromorphic chip making this a fully neuromorphic system. Multiple detection units can be deployed to create a virtual tripwire which detects when and where drones enter a restricted zone. We show that our neuromorphic solution is several orders of magnitude more energy efficient than a reference solution deployed on an edge GPU, allowing the system to run for over a year on battery power. We investigate how synthetically generated data can be used for training, and show that our model most likely relies on the shape of the drone rather than the temporal characteristics of its propellers. The small size and low power consumption allows easy deployment in contested areas or locations that lack power infrastructure.
>
---
#### [new 105] Learning by Imagining: Debiased Feature Augmentation for Compositional Zero-Shot Learning
- **分类: cs.CV**

- **简介: 该论文属于零样本学习任务，旨在解决未见过的属性-物体组合识别问题。提出DeFA方法，通过解耦与重建框架及去偏策略，增强特征表示，提升模型泛化能力。实验表明其在闭世界和开世界设置中均表现优异。**

- **链接: [http://arxiv.org/pdf/2509.12711v1](http://arxiv.org/pdf/2509.12711v1)**

> **作者:** Haozhe Zhang; Chenchen Jing; Mingyu Liu; Qingsheng Wang; Hao Chen
>
> **摘要:** Compositional Zero-Shot Learning (CZSL) aims to recognize unseen attribute-object compositions by learning prior knowledge of seen primitives, \textit{i.e.}, attributes and objects. Learning generalizable compositional representations in CZSL remains challenging due to the entangled nature of attributes and objects as well as the prevalence of long-tailed distributions in real-world data. Inspired by neuroscientific findings that imagination and perception share similar neural processes, we propose a novel approach called Debiased Feature Augmentation (DeFA) to address these challenges. The proposed DeFA integrates a disentangle-and-reconstruct framework for feature augmentation with a debiasing strategy. DeFA explicitly leverages the prior knowledge of seen attributes and objects by synthesizing high-fidelity composition features to support compositional generalization. Extensive experiments on three widely used datasets demonstrate that DeFA achieves state-of-the-art performance in both \textit{closed-world} and \textit{open-world} settings.
>
---
#### [new 106] TexTAR : Textual Attribute Recognition in Multi-domain and Multi-lingual Document Images
- **分类: cs.CV**

- **简介: 该论文提出TexTAR，用于多领域、多语言文档图像中的文本属性识别任务。旨在解决现有方法在计算效率和噪声环境下的适应性问题，通过引入上下文感知的Transformer架构和MMTAD数据集，提升属性识别准确率。**

- **链接: [http://arxiv.org/pdf/2509.13151v1](http://arxiv.org/pdf/2509.13151v1)**

> **作者:** Rohan Kumar; Jyothi Swaroopa Jinka; Ravi Kiran Sarvadevabhatla
>
> **备注:** Accepted at ICDAR 2025 (Oral)
>
> **摘要:** Recognizing textual attributes such as bold, italic, underline and strikeout is essential for understanding text semantics, structure, and visual presentation. These attributes highlight key information, making them crucial for document analysis. Existing methods struggle with computational efficiency or adaptability in noisy, multilingual settings. To address this, we introduce TexTAR, a multi-task, context-aware Transformer for Textual Attribute Recognition (TAR). Our novel data selection pipeline enhances context awareness, and our architecture employs a 2D RoPE (Rotary Positional Embedding)-style mechanism to incorporate input context for more accurate attribute predictions. We also introduce MMTAD, a diverse, multilingual, multi-domain dataset annotated with text attributes across real-world documents such as legal records, notices, and textbooks. Extensive evaluations show TexTAR outperforms existing methods, demonstrating that contextual awareness contributes to state-of-the-art TAR performance.
>
---
#### [new 107] Image Tokenizer Needs Post-Training
- **分类: cs.CV**

- **简介: 该论文针对图像生成模型中冻结的图像编码器存在的重建与生成分布差异问题，提出主训练与后训练结合的新方案，提升编码器鲁棒性与生成质量，引入pFID指标评估性能，并验证后训练策略的有效性。属于图像生成任务。**

- **链接: [http://arxiv.org/pdf/2509.12474v1](http://arxiv.org/pdf/2509.12474v1)**

> **作者:** Kai Qiu; Xiang Li; Hao Chen; Jason Kuen; Xiaohao Xu; Jiuxiang Gu; Yinyi Luo; Bhiksha Raj; Zhe Lin; Marios Savvides
>
> **备注:** 21 pages, 16 figures, 10 tables. arXiv admin note: substantial text overlap with arXiv:2503.08354
>
> **摘要:** Recent image generative models typically capture the image distribution in a pre-constructed latent space, relying on a frozen image tokenizer. However, there exists a significant discrepancy between the reconstruction and generation distribution, where current tokenizers only prioritize the reconstruction task that happens before generative training without considering the generation errors during sampling. In this paper, we comprehensively analyze the reason for this discrepancy in a discrete latent space, and, from which, we propose a novel tokenizer training scheme including both main-training and post-training, focusing on improving latent space construction and decoding respectively. During the main training, a latent perturbation strategy is proposed to simulate sampling noises, \ie, the unexpected tokens generated in generative inference. Specifically, we propose a plug-and-play tokenizer training scheme, which significantly enhances the robustness of tokenizer, thus boosting the generation quality and convergence speed, and a novel tokenizer evaluation metric, \ie, pFID, which successfully correlates the tokenizer performance to generation quality. During post-training, we further optimize the tokenizer decoder regarding a well-trained generative model to mitigate the distribution difference between generated and reconstructed tokens. With a $\sim$400M generator, a discrete tokenizer trained with our proposed main training achieves a notable 1.60 gFID and further obtains 1.36 gFID with the additional post-training. Further experiments are conducted to broadly validate the effectiveness of our post-training strategy on off-the-shelf discrete and continuous tokenizers, coupled with autoregressive and diffusion-based generators.
>
---
#### [new 108] TFANet: Three-Stage Image-Text Feature Alignment Network for Robust Referring Image Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出TFANet，用于解决指称图像分割（RIS）任务中多模态对齐不足和语义丢失问题。通过三阶段框架（KPS、KFS、KIS），增强图像与文本特征对齐，提升复杂场景下的分割精度。**

- **链接: [http://arxiv.org/pdf/2509.13070v1](http://arxiv.org/pdf/2509.13070v1)**

> **作者:** Qianqi Lu; Yuxiang Xie; Jing Zhang; Shiwei Zou; Yan Chen; Xidao Luan
>
> **摘要:** Referring Image Segmentation (RIS) is a task that segments image regions based on language expressions, requiring fine-grained alignment between two modalities. However, existing methods often struggle with multimodal misalignment and language semantic loss, especially in complex scenes containing multiple visually similar objects, where uniquely described targets are frequently mislocalized or incompletely segmented. To tackle these challenges, this paper proposes TFANet, a Three-stage Image-Text Feature Alignment Network that systematically enhances multimodal alignment through a hierarchical framework comprising three stages: Knowledge Plus Stage (KPS), Knowledge Fusion Stage (KFS), and Knowledge Intensification Stage (KIS). In the first stage, we design the Multiscale Linear Cross-Attention Module (MLAM), which facilitates bidirectional semantic exchange between visual features and textual representations across multiple scales. This establishes rich and efficient alignment between image regions and different granularities of linguistic descriptions. Subsequently, the KFS further strengthens feature alignment through the Cross-modal Feature Scanning Module (CFSM), which applies multimodal selective scanning to capture long-range dependencies and construct a unified multimodal representation. This is essential for modeling long-range cross-modal dependencies and enhancing alignment accuracy in complex scenes. Finally, in the KIS, we propose the Word-level Linguistic Feature-guided Semantic Deepening Module (WFDM) to compensate for semantic degradation introduced in earlier stages.
>
---
#### [new 109] DeepEyeNet: Generating Medical Report for Retinal Images
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出DeepEyeNet模型，用于自动生成视网膜图像的医学报告。任务是解决眼科医生短缺导致的诊断效率低问题，通过多模态深度学习、关键词表示优化等方法提升报告生成的准确性和可解释性。**

- **链接: [http://arxiv.org/pdf/2509.12534v1](http://arxiv.org/pdf/2509.12534v1)**

> **作者:** Jia-Hong Huang
>
> **备注:** The paper is accepted by the Conference on Information and Knowledge Management (CIKM), 2025
>
> **摘要:** The increasing prevalence of retinal diseases poses a significant challenge to the healthcare system, as the demand for ophthalmologists surpasses the available workforce. This imbalance creates a bottleneck in diagnosis and treatment, potentially delaying critical care. Traditional methods of generating medical reports from retinal images rely on manual interpretation, which is time-consuming and prone to errors, further straining ophthalmologists' limited resources. This thesis investigates the potential of Artificial Intelligence (AI) to automate medical report generation for retinal images. AI can quickly analyze large volumes of image data, identifying subtle patterns essential for accurate diagnosis. By automating this process, AI systems can greatly enhance the efficiency of retinal disease diagnosis, reducing doctors' workloads and enabling them to focus on more complex cases. The proposed AI-based methods address key challenges in automated report generation: (1) A multi-modal deep learning approach captures interactions between textual keywords and retinal images, resulting in more comprehensive medical reports; (2) Improved methods for medical keyword representation enhance the system's ability to capture nuances in medical terminology; (3) Strategies to overcome RNN-based models' limitations, particularly in capturing long-range dependencies within medical descriptions; (4) Techniques to enhance the interpretability of the AI-based report generation system, fostering trust and acceptance in clinical practice. These methods are rigorously evaluated using various metrics and achieve state-of-the-art performance. This thesis demonstrates AI's potential to revolutionize retinal disease diagnosis by automating medical report generation, ultimately improving clinical efficiency, diagnostic accuracy, and patient care.
>
---
#### [new 110] V-Math: An Agentic Approach to the Vietnamese National High School Graduation Mathematics Exams
- **分类: cs.AI; cs.CV; cs.CY**

- **简介: 该论文提出V-Math框架，旨在帮助越南高中生备考数学毕业考试。通过集成问题生成、解答解释和个性化辅导三个AI代理，提升学生自主学习效率，并辅助教师生成合规试题，减轻工作负担，提高教学资源多样性。**

- **链接: [http://arxiv.org/pdf/2509.12251v1](http://arxiv.org/pdf/2509.12251v1)**

> **作者:** Duong Q. Nguyen; Quy P. Nguyen; Nguyen Van Nhon; Quang-Thinh Bui; H. Nguyen-Xuan
>
> **摘要:** This paper develops an autonomous agentic framework called V-Math that aims to assist Vietnamese high school students in preparing for the National High School Graduation Mathematics Exams (NHSGMEs). The salient framework integrates three specialized AI agents: a specification-matrix-conditioned question generator, a solver/explainer for detailed step-by-step reasoning, and a personalized tutor that adapts to student performance. Beyond enabling self-paced student practice, V-Math supports teachers by generating innovative, compliant exam questions and building diverse, high-quality question banks. This reduces manual workload and enriches instructional resources. We describe the system architecture, focusing on practice modes for learners and teacher-oriented features for question generation. Preliminary evaluations demonstrate that V-Math produces matrix-aligned exams with high solution accuracy, delivers coherent explanations, and enhances the variety of practice materials. These results highlight its potential to support scalable, equitable mathematics preparation aligned with national standards while also empowering teachers through AI-assisted exam creation.
>
---
#### [new 111] Neural Diffeomorphic-Neural Operator for Residual Stress-Induced Deformation Prediction
- **分类: cs.LG; cs.CV; eess.IV**

- **简介: 论文提出NDNO框架，解决复杂几何结构中残余应力引起的变形预测问题。通过微分同胚映射将不同几何映射到统一域，提升神经算子效率与适应性，实现高效高精度变形预测。**

- **链接: [http://arxiv.org/pdf/2509.12237v1](http://arxiv.org/pdf/2509.12237v1)**

> **作者:** Changqing Liu; Kaining Dai; Zhiwei Zhao; Tianyi Wu; Yingguang Li
>
> **摘要:** Accurate prediction of machining deformation in structural components is essential for ensuring dimensional precision and reliability. Such deformation often originates from residual stress fields, whose distribution and influence vary significantly with geometric complexity. Conventional numerical methods for modeling the coupling between residual stresses and deformation are computationally expensive, particularly when diverse geometries are considered. Neural operators have recently emerged as a powerful paradigm for efficiently solving partial differential equations, offering notable advantages in accelerating residual stress-deformation analysis. However, their direct application across changing geometric domains faces theoretical and practical limitations. To address this challenge, a novel framework based on diffeomorphic embedding neural operators named neural diffeomorphic-neural operator (NDNO) is introduced. Complex three-dimensional geometries are explicitly mapped to a common reference domain through a diffeomorphic neural network constrained by smoothness and invertibility. The neural operator is then trained on this reference domain, enabling efficient learning of deformation fields induced by residual stresses. Once trained, both the diffeomorphic neural network and the neural operator demonstrate efficient prediction capabilities, allowing rapid adaptation to varying geometries. The proposed method thus provides an effective and computationally efficient solution for deformation prediction in structural components subject to varying geometries. The proposed method is validated to predict both main-direction and multi-direction deformation fields, achieving high accuracy and efficiency across parts with diverse geometries including component types, dimensions and features.
>
---
#### [new 112] Universal Gröbner Bases of (Universal) Multiview Ideals
- **分类: math.AC; cs.CV; math.AG**

- **简介: 论文研究多视角理想及其通用Gröbner基，解决未知相机下的几何建模问题。通过黄与拉森准则，证明一组多项式构成通用Gröbner基，并利用对称性简化和归纳方法推广至无限理想族。**

- **链接: [http://arxiv.org/pdf/2509.12376v1](http://arxiv.org/pdf/2509.12376v1)**

> **作者:** Timothy Duff; Jack Kendrick; Rekha R. Thomas
>
> **摘要:** Multiview ideals arise from the geometry of image formation in pinhole cameras, and universal multiview ideals are their analogs for unknown cameras. We prove that a natural collection of polynomials form a universal Gr\"obner basis for both types of ideals using a criterion introduced by Huang and Larson, and include a proof of their criterion in our setting. Symmetry reduction and induction enable the method to be deployed on an infinite family of ideals. We also give an explicit description of the matroids on which the methodology depends, in the context of multiview ideals.
>
---
#### [new 113] Neural 3D Object Reconstruction with Small-Scale Unmanned Aerial Vehicles
- **分类: cs.RO; cs.AR; cs.CV; cs.ET; cs.SY; eess.SY**

- **简介: 该论文提出一种基于轻量无人机的自主3D重建系统，解决其载重与自主性限制问题。通过实时反馈调整飞行路径，并结合NeRF技术提升重建精度，实现高保真静态物体建模。**

- **链接: [http://arxiv.org/pdf/2509.12458v1](http://arxiv.org/pdf/2509.12458v1)**

> **作者:** Àlmos Veres-Vitàlyos; Genis Castillo Gomez-Raya; Filip Lemic; Daniel Johannes Bugelnig; Bernhard Rinner; Sergi Abadal; Xavier Costa-Pérez
>
> **备注:** 13 pages, 16 figures, 3 tables, 45 references
>
> **摘要:** Small Unmanned Aerial Vehicles (UAVs) exhibit immense potential for navigating indoor and hard-to-reach areas, yet their significant constraints in payload and autonomy have largely prevented their use for complex tasks like high-quality 3-Dimensional (3D) reconstruction. To overcome this challenge, we introduce a novel system architecture that enables fully autonomous, high-fidelity 3D scanning of static objects using UAVs weighing under 100 grams. Our core innovation lies in a dual-reconstruction pipeline that creates a real-time feedback loop between data capture and flight control. A near-real-time (near-RT) process uses Structure from Motion (SfM) to generate an instantaneous pointcloud of the object. The system analyzes the model quality on the fly and dynamically adapts the UAV's trajectory to intelligently capture new images of poorly covered areas. This ensures comprehensive data acquisition. For the final, detailed output, a non-real-time (non-RT) pipeline employs a Neural Radiance Fields (NeRF)-based Neural 3D Reconstruction (N3DR) approach, fusing SfM-derived camera poses with precise Ultra Wide-Band (UWB) location data to achieve superior accuracy. We implemented and validated this architecture using Crazyflie 2.1 UAVs. Our experiments, conducted in both single- and multi-UAV configurations, conclusively show that dynamic trajectory adaptation consistently improves reconstruction quality over static flight paths. This work demonstrates a scalable and autonomous solution that unlocks the potential of miniaturized UAVs for fine-grained 3D reconstruction in constrained environments, a capability previously limited to much larger platforms.
>
---
#### [new 114] Simulating Clinical AI Assistance using Multimodal LLMs: A Case Study in Diabetic Retinopathy
- **分类: cs.AI; cs.CV; cs.HC**

- **简介: 论文研究多模态大语言模型在糖尿病视网膜病变检测中的应用，探索不同输出形式对临床AI辅助效果的影响。通过对比GPT-4o与MedGemma模型，评估其性能及协作潜力，旨在提升筛查效率与可信度。**

- **链接: [http://arxiv.org/pdf/2509.13234v1](http://arxiv.org/pdf/2509.13234v1)**

> **作者:** Nadim Barakat; William Lotter
>
> **摘要:** Diabetic retinopathy (DR) is a leading cause of blindness worldwide, and AI systems can expand access to fundus photography screening. Current FDA-cleared systems primarily provide binary referral outputs, where this minimal output may limit clinical trust and utility. Yet, determining the most effective output format to enhance clinician-AI performance is an empirical challenge that is difficult to assess at scale. We evaluated multimodal large language models (MLLMs) for DR detection and their ability to simulate clinical AI assistance across different output types. Two models were tested on IDRiD and Messidor-2: GPT-4o, a general-purpose MLLM, and MedGemma, an open-source medical model. Experiments included: (1) baseline evaluation, (2) simulated AI assistance with synthetic predictions, and (3) actual AI-to-AI collaboration where GPT-4o incorporated MedGemma outputs. MedGemma outperformed GPT-4o at baseline, achieving higher sensitivity and AUROC, while GPT-4o showed near-perfect specificity but low sensitivity. Both models adjusted predictions based on simulated AI inputs, but GPT-4o's performance collapsed with incorrect ones, whereas MedGemma remained more stable. In actual collaboration, GPT-4o achieved strong results when guided by MedGemma's descriptive outputs, even without direct image access (AUROC up to 0.96). These findings suggest MLLMs may improve DR screening pipelines and serve as scalable simulators for studying clinical AI assistance across varying output configurations. Open, lightweight models such as MedGemma may be especially valuable in low-resource settings, while descriptive outputs could enhance explainability and clinician trust in clinical workflows.
>
---
#### [new 115] The Better You Learn, The Smarter You Prune: Towards Efficient Vision-language-action Models via Differentiable Token Pruning
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出LightVLA，一种通过可微分视觉token剪枝提升视觉-语言-动作模型效率的方法。针对VLA模型在资源受限平台部署时计算量大的问题，通过动态评估token重要性实现高效剪枝，提升性能并减少计算开销。**

- **链接: [http://arxiv.org/pdf/2509.12594v1](http://arxiv.org/pdf/2509.12594v1)**

> **作者:** Titong Jiang; Xuefeng Jiang; Yuan Ma; Xin Wen; Bailin Li; Kun Zhan; Peng Jia; Yahui Liu; Sheng Sun; Xianpeng Lang
>
> **备注:** Under review. Project site: https://liauto-research.github.io/LightVLA
>
> **摘要:** We present LightVLA, a simple yet effective differentiable token pruning framework for vision-language-action (VLA) models. While VLA models have shown impressive capability in executing real-world robotic tasks, their deployment on resource-constrained platforms is often bottlenecked by the heavy attention-based computation over large sets of visual tokens. LightVLA addresses this challenge through adaptive, performance-driven pruning of visual tokens: It generates dynamic queries to evaluate visual token importance, and adopts Gumbel softmax to enable differentiable token selection. Through fine-tuning, LightVLA learns to preserve the most informative visual tokens while pruning tokens which do not contribute to task execution, thereby improving efficiency and performance simultaneously. Notably, LightVLA requires no heuristic magic numbers and introduces no additional trainable parameters, making it compatible with modern inference frameworks. Experimental results demonstrate that LightVLA outperforms different VLA models and existing token pruning methods across diverse tasks on the LIBERO benchmark, achieving higher success rates with substantially reduced computational overhead. Specifically, LightVLA reduces FLOPs and latency by 59.1% and 38.2% respectively, with a 2.9% improvement in task success rate. Meanwhile, we also investigate the learnable query-based token pruning method LightVLA* with additional trainable parameters, which also achieves satisfactory performance. Our work reveals that as VLA pursues optimal performance, LightVLA spontaneously learns to prune tokens from a performance-driven perspective. To the best of our knowledge, LightVLA is the first work to apply adaptive visual token pruning to VLA tasks with the collateral goals of efficiency and performance, marking a significant step toward more efficient, powerful and practical real-time robotic systems.
>
---
#### [new 116] Generalizable Holographic Reconstruction via Amplitude-Only Diffusion Priors
- **分类: physics.optics; cs.CV; cs.LG**

- **简介: 该论文提出一种基于幅度仅扩散先验的可泛化全息重建方法，解决无透镜全息成像中相位恢复这一病态逆问题。通过训练仅依赖幅度数据的扩散模型，实现无需真实相位数据的复场重建，适用于多种物体和系统配置。**

- **链接: [http://arxiv.org/pdf/2509.12728v1](http://arxiv.org/pdf/2509.12728v1)**

> **作者:** Jeongsol Kim; Chanseok Lee; Jong Chul Ye; Mooseok Jang
>
> **备注:** Keywords: Diffusion model, phase retrieval, inline-holography, inverse problem
>
> **摘要:** Phase retrieval in inline holography is a fundamental yet ill-posed inverse problem due to the nonlinear coupling between amplitude and phase in coherent imaging. We present a novel off-the-shelf solution that leverages a diffusion model trained solely on object amplitude to recover both amplitude and phase from diffraction intensities. Using a predictor-corrector sampling framework with separate likelihood gradients for amplitude and phase, our method enables complex field reconstruction without requiring ground-truth phase data for training. We validate the proposed approach through extensive simulations and experiments, demonstrating robust generalization across diverse object shapes, imaging system configurations, and modalities, including lensless setups. Notably, a diffusion prior trained on simple amplitude data (e.g., polystyrene beads) successfully reconstructs complex biological tissue structures, highlighting the method's adaptability. This framework provides a cost-effective, generalizable solution for nonlinear inverse problems in computational imaging, and establishes a foundation for broader coherent imaging applications beyond holography.
>
---
#### [new 117] MEGAN: Mixture of Experts for Robust Uncertainty Estimation in Endoscopy Videos
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出MEGAN，用于内镜视频中溃疡性结肠炎严重程度估计的不确定性量化。针对医疗AI中单一专家标注忽略评分者差异的问题，MEGAN融合多个EDL模型预测与不确定性，提升预测准确性与校准效果。**

- **链接: [http://arxiv.org/pdf/2509.12772v1](http://arxiv.org/pdf/2509.12772v1)**

> **作者:** Damola Agbelese; Krishna Chaitanya; Pushpak Pati; Chaitanya Parmar; Pooya Mobadersany; Shreyas Fadnavis; Lindsey Surace; Shadi Yarandi; Louis R. Ghanem; Molly Lucas; Tommaso Mansi; Oana Gabriela Cula; Pablo F. Damasceno; Kristopher Standish
>
> **备注:** 11 pages, 2 figures, 1 table, accepted at UNSURE, MICCAI
>
> **摘要:** Reliable uncertainty quantification (UQ) is essential in medical AI. Evidential Deep Learning (EDL) offers a computationally efficient way to quantify model uncertainty alongside predictions, unlike traditional methods such as Monte Carlo (MC) Dropout and Deep Ensembles (DE). However, all these methods often rely on a single expert's annotations as ground truth for model training, overlooking the inter-rater variability in healthcare. To address this issue, we propose MEGAN, a Multi-Expert Gating Network that aggregates uncertainty estimates and predictions from multiple AI experts via EDL models trained with diverse ground truths and modeling strategies. MEGAN's gating network optimally combines predictions and uncertainties from each EDL model, enhancing overall prediction confidence and calibration. We extensively benchmark MEGAN on endoscopy videos for Ulcerative colitis (UC) disease severity estimation, assessed by visual labeling of Mayo Endoscopic Subscore (MES), where inter-rater variability is prevalent. In large-scale prospective UC clinical trial, MEGAN achieved a 3.5% improvement in F1-score and a 30.5% reduction in Expected Calibration Error (ECE) compared to existing methods. Furthermore, MEGAN facilitated uncertainty-guided sample stratification, reducing the annotation burden and potentially increasing efficiency and consistency in UC trials.
>
---
#### [new 118] Human + AI for Accelerating Ad Localization Evaluation
- **分类: cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出一种结合AI与人工的框架，用于加速广告本地化评估。任务是解决多语言广告在视觉一致性、空间对齐和风格统一上的挑战。工作包括集成文本检测、修复、机器翻译与文本重置技术，实现高效且视觉连贯的广告本地化。**

- **链接: [http://arxiv.org/pdf/2509.12543v1](http://arxiv.org/pdf/2509.12543v1)**

> **作者:** Harshit Rajgarhia; Shivali Dalmia; Mengyang Zhao; Mukherji Abhishek; Kiran Ganesh
>
> **摘要:** Adapting advertisements for multilingual audiences requires more than simple text translation; it demands preservation of visual consistency, spatial alignment, and stylistic integrity across diverse languages and formats. We introduce a structured framework that combines automated components with human oversight to address the complexities of advertisement localization. To the best of our knowledge, this is the first work to integrate scene text detection, inpainting, machine translation (MT), and text reimposition specifically for accelerating ad localization evaluation workflows. Qualitative results across six locales demonstrate that our approach produces semantically accurate and visually coherent localized advertisements, suitable for deployment in real-world workflows.
>
---
#### [new 119] DinoAtten3D: Slice-Level Attention Aggregation of DinoV2 for 3D Brain MRI Anomaly Classification
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出DinoAtten3D框架，用于3D脑部MRI的异常分类任务。针对数据稀缺和类别不平衡问题，结合DINOv2预训练模型与注意力机制，实现切片级特征聚合，并采用复合损失函数提升分类性能。**

- **链接: [http://arxiv.org/pdf/2509.12512v1](http://arxiv.org/pdf/2509.12512v1)**

> **作者:** Fazle Rafsani; Jay Shah; Catherine D. Chong; Todd J. Schwedt; Teresa Wu
>
> **备注:** ACCEPTED at the ICCV 2025 Workshop on Anomaly Detection with Foundation Models
>
> **摘要:** Anomaly detection and classification in medical imaging are critical for early diagnosis but remain challenging due to limited annotated data, class imbalance, and the high cost of expert labeling. Emerging vision foundation models such as DINOv2, pretrained on extensive, unlabeled datasets, offer generalized representations that can potentially alleviate these limitations. In this study, we propose an attention-based global aggregation framework tailored specifically for 3D medical image anomaly classification. Leveraging the self-supervised DINOv2 model as a pretrained feature extractor, our method processes individual 2D axial slices of brain MRIs, assigning adaptive slice-level importance weights through a soft attention mechanism. To further address data scarcity, we employ a composite loss function combining supervised contrastive learning with class-variance regularization, enhancing inter-class separability and intra-class consistency. We validate our framework on the ADNI dataset and an institutional multi-class headache cohort, demonstrating strong anomaly classification performance despite limited data availability and significant class imbalance. Our results highlight the efficacy of utilizing pretrained 2D foundation models combined with attention-based slice aggregation for robust volumetric anomaly detection in medical imaging. Our implementation is publicly available at https://github.com/Rafsani/DinoAtten3D.git.
>
---
#### [new 120] Flexible Multimodal Neuroimaging Fusion for Alzheimer's Disease Progression Prediction
- **分类: cs.LG; cs.AI; cs.CV; eess.IV**

- **简介: 该论文提出PerM-MoE方法，用于阿尔茨海默病进展预测任务，解决多模态数据缺失下的模型灵活性问题。通过独立路由机制提升模型性能，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.12234v1](http://arxiv.org/pdf/2509.12234v1)**

> **作者:** Benjamin Burns; Yuan Xue; Douglas W. Scharre; Xia Ning
>
> **备注:** Accepted at Applications of Medical AI 2025
>
> **摘要:** Alzheimer's disease (AD) is a progressive neurodegenerative disease with high inter-patient variance in rate of cognitive decline. AD progression prediction aims to forecast patient cognitive decline and benefits from incorporating multiple neuroimaging modalities. However, existing multimodal models fail to make accurate predictions when many modalities are missing during inference, as is often the case in clinical settings. To increase multimodal model flexibility under high modality missingness, we introduce PerM-MoE, a novel sparse mixture-of-experts method that uses independent routers for each modality in place of the conventional, single router. Using T1-weighted MRI, FLAIR, amyloid beta PET, and tau PET neuroimaging data from the Alzheimer's Disease Neuroimaging Initiative (ADNI), we evaluate PerM-MoE, state-of-the-art Flex-MoE, and unimodal neuroimaging models on predicting two-year change in Clinical Dementia Rating-Sum of Boxes (CDR-SB) scores under varying levels of modality missingness. PerM-MoE outperforms the state of the art in most variations of modality missingness and demonstrates more effective utility of experts than Flex-MoE.
>
---
#### [new 121] iCD: A Implicit Clustering Distillation Mathod for Structural Information Mining
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出iCD方法，用于知识蒸馏任务，解决模型解释性差的问题。通过挖掘logit中的结构信息，无需标签和特征对齐，提升学生模型性能，尤其在细粒度分类中效果显著。**

- **链接: [http://arxiv.org/pdf/2509.12553v1](http://arxiv.org/pdf/2509.12553v1)**

> **作者:** Xiang Xue; Yatu Ji; Qing-dao-er-ji Ren; Bao Shi; Min Lu; Nier Wu; Xufei Zhuang; Haiteng Xu; Gan-qi-qi-ge Cha
>
> **摘要:** Logit Knowledge Distillation has gained substantial research interest in recent years due to its simplicity and lack of requirement for intermediate feature alignment; however, it suffers from limited interpretability in its decision-making process. To address this, we propose implicit Clustering Distillation (iCD): a simple and effective method that mines and transfers interpretable structural knowledge from logits, without requiring ground-truth labels or feature-space alignment. iCD leverages Gram matrices over decoupled local logit representations to enable student models to learn latent semantic structural patterns. Extensive experiments on benchmark datasets demonstrate the effectiveness of iCD across diverse teacher-student architectures, with particularly strong performance in fine-grained classification tasks -- achieving a peak improvement of +5.08% over the baseline. The code is available at: https://github.com/maomaochongaa/iCD.
>
---
#### [new 122] ChartGaze: Enhancing Chart Understanding in LVLMs with Eye-Tracking Guided Attention Refinement
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于图表问答任务，旨在解决LVLMs关注无关区域导致准确率低的问题。作者构建了ChartGaze眼动数据集，并提出基于注视的注意力优化方法，提升模型回答准确性和注意力对齐效果。**

- **链接: [http://arxiv.org/pdf/2509.13282v1](http://arxiv.org/pdf/2509.13282v1)**

> **作者:** Ali Salamatian; Amirhossein Abaskohi; Wan-Cyuan Fan; Mir Rayat Imtiaz Hossain; Leonid Sigal; Giuseppe Carenini
>
> **备注:** EMNLP 2025
>
> **摘要:** Charts are a crucial visual medium for communicating and representing information. While Large Vision-Language Models (LVLMs) have made progress on chart question answering (CQA), the task remains challenging, particularly when models attend to irrelevant regions of the chart. In this work, we present ChartGaze, a new eye-tracking dataset that captures human gaze patterns during chart reasoning tasks. Through a systematic comparison of human and model attention, we find that LVLMs often diverge from human gaze, leading to reduced interpretability and accuracy. To address this, we propose a gaze-guided attention refinement that aligns image-text attention with human fixations. Our approach improves both answer accuracy and attention alignment, yielding gains of up to 2.56 percentage points across multiple models. These results demonstrate the promise of incorporating human gaze to enhance both the reasoning quality and interpretability of chart-focused LVLMs.
>
---
#### [new 123] ActiveVLN: Towards Active Exploration via Multi-Turn RL in Vision-and-Language Navigation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉与语言导航（VLN）任务，旨在解决传统方法依赖模仿学习、数据成本高及探索能力不足的问题。提出ActiveVLN框架，通过多轮强化学习实现主动探索，提升导航性能并减少对专家轨迹的依赖。**

- **链接: [http://arxiv.org/pdf/2509.12618v1](http://arxiv.org/pdf/2509.12618v1)**

> **作者:** Zekai Zhang; Weiye Zhu; Hewei Pan; Xiangchen Wang; Rongtao Xu; Xing Sun; Feng Zheng
>
> **摘要:** The Vision-and-Language Navigation (VLN) task requires an agent to follow natural language instructions and navigate through complex environments. Existing MLLM-based VLN methods primarily rely on imitation learning (IL) and often use DAgger for post-training to mitigate covariate shift. While effective, these approaches incur substantial data collection and training costs. Reinforcement learning (RL) offers a promising alternative. However, prior VLN RL methods lack dynamic interaction with the environment and depend on expert trajectories for reward shaping, rather than engaging in open-ended active exploration. This restricts the agent's ability to discover diverse and plausible navigation routes. To address these limitations, we propose ActiveVLN, a VLN framework that explicitly enables active exploration through multi-turn RL. In the first stage, a small fraction of expert trajectories is used for IL to bootstrap the agent. In the second stage, the agent iteratively predicts and executes actions, automatically collects diverse trajectories, and optimizes multiple rollouts via the GRPO objective. To further improve RL efficiency, we introduce a dynamic early-stopping strategy to prune long-tail or likely failed trajectories, along with additional engineering optimizations. Experiments show that ActiveVLN achieves the largest performance gains over IL baselines compared to both DAgger-based and prior RL-based post-training methods, while reaching competitive performance with state-of-the-art approaches despite using a smaller model. Code and data will be released soon.
>
---
#### [new 124] InJecteD: Analyzing Trajectories and Drift Dynamics in Denoising Diffusion Probabilistic Models for 2D Point Cloud Generation
- **分类: cs.LG; cs.CV**

- **简介: 论文提出InJecteD框架，用于分析DDPM在2D点云生成中的轨迹与漂移动态。通过量化轨迹特性，揭示去噪过程的阶段特征，提升模型可解释性，支持模型调试与优化。**

- **链接: [http://arxiv.org/pdf/2509.12239v1](http://arxiv.org/pdf/2509.12239v1)**

> **作者:** Sanyam Jain; Khuram Naveed; Illia Oleksiienko; Alexandros Iosifidis; Ruben Pauwels
>
> **摘要:** This work introduces InJecteD, a framework for interpreting Denoising Diffusion Probabilistic Models (DDPMs) by analyzing sample trajectories during the denoising process of 2D point cloud generation. We apply this framework to three datasets from the Datasaurus Dozen bullseye, dino, and circle using a simplified DDPM architecture with customizable input and time embeddings. Our approach quantifies trajectory properties, including displacement, velocity, clustering, and drift field dynamics, using statistical metrics such as Wasserstein distance and cosine similarity. By enhancing model transparency, InJecteD supports human AI collaboration by enabling practitioners to debug and refine generative models. Experiments reveal distinct denoising phases: initial noise exploration, rapid shape formation, and final refinement, with dataset-specific behaviors example, bullseyes concentric convergence vs. dinos complex contour formation. We evaluate four model configurations, varying embeddings and noise schedules, demonstrating that Fourier based embeddings improve trajectory stability and reconstruction quality
>
---
#### [new 125] Tool-R1: Sample-Efficient Reinforcement Learning for Agentic Tool Use
- **分类: cs.LG; cs.CV**

- **简介: 论文提出Tool-R1框架，通过强化学习使大语言模型高效使用工具完成复杂任务。解决LLM在现实任务中知识更新、精确操作和工具使用不足的问题，采用可执行代码生成与动态样本队列优化训练效率。**

- **链接: [http://arxiv.org/pdf/2509.12867v1](http://arxiv.org/pdf/2509.12867v1)**

> **作者:** Yabo Zhang; Yihan Zeng; Qingyun Li; Zhen Hu; Kavin Han; Wangmeng Zuo
>
> **摘要:** Large language models (LLMs) have demonstrated strong capabilities in language understanding and reasoning, yet they remain limited when tackling real-world tasks that require up-to-date knowledge, precise operations, or specialized tool use. To address this, we propose Tool-R1, a reinforcement learning framework that enables LLMs to perform general, compositional, and multi-step tool use by generating executable Python code. Tool-R1 supports integration of user-defined tools and standard libraries, with variable sharing across steps to construct coherent workflows. An outcome-based reward function, combining LLM-based answer judgment and code execution success, guides policy optimization. To improve training efficiency, we maintain a dynamic sample queue to cache and reuse high-quality trajectories, reducing the overhead of costly online sampling. Experiments on the GAIA benchmark show that Tool-R1 substantially improves both accuracy and robustness, achieving about 10\% gain over strong baselines, with larger improvements on complex multi-step tasks. These results highlight the potential of Tool-R1 for enabling reliable and efficient tool-augmented reasoning in real-world applications. Our code will be available at https://github.com/YBYBZhang/Tool-R1.
>
---
#### [new 126] HLSMAC: A New StarCraft Multi-Agent Challenge for High-Level Strategic Decision-Making
- **分类: cs.AI; cs.CV; cs.GT; cs.LG; cs.MA**

- **简介: 该论文提出HLSMAC，一个基于《三十六计》的StarCraft II多智能体强化学习基准，用于评估高层战略决策能力。其解决现有基准仅关注微观管理的问题，设计12个场景并引入新指标，推动多智能体战略智能发展。**

- **链接: [http://arxiv.org/pdf/2509.12927v1](http://arxiv.org/pdf/2509.12927v1)**

> **作者:** Xingxing Hong; Yungong Wang; Dexin Jin; Ye Yuan; Ximing Huang; Zijian Wu; Wenxin Li
>
> **备注:** 30 pages, 13 figures with appendix
>
> **摘要:** Benchmarks are crucial for assessing multi-agent reinforcement learning (MARL) algorithms. While StarCraft II-related environments have driven significant advances in MARL, existing benchmarks like SMAC focus primarily on micromanagement, limiting comprehensive evaluation of high-level strategic intelligence. To address this, we introduce HLSMAC, a new cooperative MARL benchmark with 12 carefully designed StarCraft II scenarios based on classical stratagems from the Thirty-Six Stratagems. Each scenario corresponds to a specific stratagem and is designed to challenge agents with diverse strategic elements, including tactical maneuvering, timing coordination, and deception, thereby opening up avenues for evaluating high-level strategic decision-making capabilities. We also propose novel metrics across multiple dimensions beyond conventional win rate, such as ability utilization and advancement efficiency, to assess agents' overall performance within the HLSMAC environment. We integrate state-of-the-art MARL algorithms and LLM-based agents with our benchmark and conduct comprehensive experiments. The results demonstrate that HLSMAC serves as a robust testbed for advancing multi-agent strategic decision-making.
>
---
#### [new 127] Developing an aeroponic smart experimental greenhouse for controlling irrigation and plant disease detection using deep learning and IoT
- **分类: cs.AI; cs.CV; cs.LG; 68T07, 68T45, 68U10; I.4.8; I.2.6; I.5.4; C.3**

- **简介: 该论文开发了一个结合IoT与深度学习的智能气雾培温室系统，用于控制灌溉和检测植物病害。通过实时监测环境参数并利用AI算法识别植物病害，提升作物管理效率与准确性。**

- **链接: [http://arxiv.org/pdf/2509.12274v1](http://arxiv.org/pdf/2509.12274v1)**

> **作者:** Mohammadreza Narimani; Ali Hajiahmad; Ali Moghimi; Reza Alimardani; Shahin Rafiee; Amir Hossein Mirzabe
>
> **备注:** Author-accepted version. Presented at ASABE Annual International Meeting (AIM) 2021 (virtual), Paper 2101252. Please cite the published meeting paper: doi:10.13031/aim.202101252. Minor wording and formatting updates in this preprint
>
> **摘要:** Controlling environmental conditions and monitoring plant status in greenhouses is critical to promptly making appropriate management decisions aimed at promoting crop production. The primary objective of this research study was to develop and test a smart aeroponic greenhouse on an experimental scale where the status of Geranium plant and environmental conditions are continuously monitored through the integration of the internet of things (IoT) and artificial intelligence (AI). An IoT-based platform was developed to control the environmental conditions of plants more efficiently and provide insights to users to make informed management decisions. In addition, we developed an AI-based disease detection framework using VGG-19, InceptionResNetV2, and InceptionV3 algorithms to analyze the images captured periodically after an intentional inoculation. The performance of the AI framework was compared with an expert's evaluation of disease status. Preliminary results showed that the IoT system implemented in the greenhouse environment is able to publish data such as temperature, humidity, water flow, and volume of charge tanks online continuously to users and adjust the controlled parameters to provide an optimal growth environment for the plants. Furthermore, the results of the AI framework demonstrate that the VGG-19 algorithm was able to identify drought stress and rust leaves from healthy leaves with the highest accuracy, 92% among the other algorithms.
>
---
#### [new 128] Enhancing Radiographic Disease Detection with MetaCheX, a Context-Aware Multimodal Model
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文提出MetaCheX，一种结合胸片图像与患者元数据的多模态模型，用于提升胸部疾病检测的准确性和公平性。通过整合元数据，模型在CheXpert Plus数据集上显著提高了诊断性能和泛化能力，解决传统模型忽视元数据导致的偏差问题。**

- **链接: [http://arxiv.org/pdf/2509.12287v1](http://arxiv.org/pdf/2509.12287v1)**

> **作者:** Nathan He; Cody Chen
>
> **备注:** All authors contributed equally, 5 pages, 2 figures, 1 table
>
> **摘要:** Existing deep learning models for chest radiology often neglect patient metadata, limiting diagnostic accuracy and fairness. To bridge this gap, we introduce MetaCheX, a novel multimodal framework that integrates chest X-ray images with structured patient metadata to replicate clinical decision-making. Our approach combines a convolutional neural network (CNN) backbone with metadata processed by a multilayer perceptron through a shared classifier. Evaluated on the CheXpert Plus dataset, MetaCheX consistently outperformed radiograph-only baseline models across multiple CNN architectures. By integrating metadata, the overall diagnostic accuracy was significantly improved, measured by an increase in AUROC. The results of this study demonstrate that metadata reduces algorithmic bias and enhances model generalizability across diverse patient populations. MetaCheX advances clinical artificial intelligence toward robust, context-aware radiographic disease detection.
>
---
#### [new 129] Unleashing the Power of Discrete-Time State Representation: Ultrafast Target-based IMU-Camera Spatial-Temporal Calibration
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出一种基于离散时间状态表示的高效IMU-相机时空标定方法，解决传统连续时间方法计算成本高的问题，提升标定效率，适用于大量视觉惯性设备的快速标定。**

- **链接: [http://arxiv.org/pdf/2509.12846v1](http://arxiv.org/pdf/2509.12846v1)**

> **作者:** Junlin Song; Antoine Richard; Miguel Olivares-Mendez
>
> **摘要:** Visual-inertial fusion is crucial for a large amount of intelligent and autonomous applications, such as robot navigation and augmented reality. To bootstrap and achieve optimal state estimation, the spatial-temporal displacements between IMU and cameras must be calibrated in advance. Most existing calibration methods adopt continuous-time state representation, more specifically the B-spline. Despite these methods achieve precise spatial-temporal calibration, they suffer from high computational cost caused by continuous-time state representation. To this end, we propose a novel and extremely efficient calibration method that unleashes the power of discrete-time state representation. Moreover, the weakness of discrete-time state representation in temporal calibration is tackled in this paper. With the increasing production of drones, cellphones and other visual-inertial platforms, if one million devices need calibration around the world, saving one minute for the calibration of each device means saving 2083 work days in total. To benefit both the research and industry communities, our code will be open-source.
>
---
#### [new 130] QDFlow: A Python package for physics simulations of quantum dot devices
- **分类: cond-mat.mes-hall; cs.CV; cs.LG; quant-ph**

- **简介: 该论文提出QDFlow，一个用于模拟量子点器件的物理开源工具，生成带真实标签的合成数据，解决实验数据获取困难的问题，支持机器学习模型的开发与验证。**

- **链接: [http://arxiv.org/pdf/2509.13298v1](http://arxiv.org/pdf/2509.13298v1)**

> **作者:** Donovan L. Buterakos; Sandesh S. Kalantre; Joshua Ziegler; Jacob M Taylor; Justyna P. Zwolak
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** Recent advances in machine learning (ML) have accelerated progress in calibrating and operating quantum dot (QD) devices. However, most ML approaches rely on access to large, high-quality labeled datasets for training, benchmarking, and validation, with labels capturing key features in the data. Obtaining such datasets experimentally is challenging due to limited data availability and the labor-intensive nature of labeling. QDFlow is an open-source physics simulator for multi-QD arrays that generates realistic synthetic data with ground-truth labels. QDFlow combines a self-consistent Thomas-Fermi solver, a dynamic capacitance model, and flexible noise modules to produce charge stability diagrams and ray-based data closely resembling experiments. With extensive tunable parameters and customizable noise models, QDFlow supports the creation of large, diverse datasets for ML development, benchmarking, and quantum device research.
>
---
#### [new 131] Sy-FAR: Symmetry-based Fair Adversarial Robustness
- **分类: cs.LG; cs.AI; cs.CR; cs.CV**

- **简介: 该论文提出Sy-FAR方法，旨在提升机器学习系统的公平对抗鲁棒性。针对现实任务中难以实现完全公平的问题，通过引入对称性约束，使不同类别间的攻击成功率趋于平衡，从而改善系统在面对对抗样本时的公平性和安全性。**

- **链接: [http://arxiv.org/pdf/2509.12939v1](http://arxiv.org/pdf/2509.12939v1)**

> **作者:** Haneen Najjar; Eyal Ronen; Mahmood Sharif
>
> **备注:** 20 pages, 11 figures
>
> **摘要:** Security-critical machine-learning (ML) systems, such as face-recognition systems, are susceptible to adversarial examples, including real-world physically realizable attacks. Various means to boost ML's adversarial robustness have been proposed; however, they typically induce unfair robustness: It is often easier to attack from certain classes or groups than from others. Several techniques have been developed to improve adversarial robustness while seeking perfect fairness between classes. Yet, prior work has focused on settings where security and fairness are less critical. Our insight is that achieving perfect parity in realistic fairness-critical tasks, such as face recognition, is often infeasible -- some classes may be highly similar, leading to more misclassifications between them. Instead, we suggest that seeking symmetry -- i.e., attacks from class $i$ to $j$ would be as successful as from $j$ to $i$ -- is more tractable. Intuitively, symmetry is a desirable because class resemblance is a symmetric relation in most domains. Additionally, as we prove theoretically, symmetry between individuals induces symmetry between any set of sub-groups, in contrast to other fairness notions where group-fairness is often elusive. We develop Sy-FAR, a technique to encourage symmetry while also optimizing adversarial robustness and extensively evaluate it using five datasets, with three model architectures, including against targeted and untargeted realistic attacks. The results show Sy-FAR significantly improves fair adversarial robustness compared to state-of-the-art methods. Moreover, we find that Sy-FAR is faster and more consistent across runs. Notably, Sy-FAR also ameliorates another type of unfairness we discover in this work -- target classes that adversarial examples are likely to be classified into become significantly less vulnerable after inducing symmetry.
>
---
#### [new 132] Gesture Evaluation in Virtual Reality
- **分类: cs.HC; cs.AI; cs.CV; cs.LG; 68T50, 68T07, 68U35; H.5.1; H.5.2; I.2.10; I.3.7**

- **简介: 该论文比较VR与2D环境下AI生成手势的感知差异，评估三种模型在VR中的表现。任务是分析手势在沉浸式环境下的评价变化，解决传统2D评价局限的问题，发现VR提升手势真实感并改变用户感知。**

- **链接: [http://arxiv.org/pdf/2509.12816v1](http://arxiv.org/pdf/2509.12816v1)**

> **作者:** Axel Wiebe Werner; Jonas Beskow; Anna Deichler
>
> **备注:** Published in Proceedings of the 26th International Conference on Multimodal Interaction (ICMI '24), ACM. Copyright 2024 ACM. Licensed under CC BY
>
> **摘要:** Gestures are central to human communication, enriching interactions through non-verbal expression. Virtual avatars increasingly use AI-generated gestures to enhance life-likeness, yet evaluations have largely been confined to 2D. Virtual Reality (VR) provides an immersive alternative that may affect how gestures are perceived. This paper presents a comparative evaluation of computer-generated gestures in VR and 2D, examining three models from the 2023 GENEA Challenge. Results show that gestures viewed in VR were rated slightly higher on average, with the strongest effect observed for motion-capture "true movement." While model rankings remained consistent across settings, VR influenced participants' overall perception and offered unique benefits over traditional 2D evaluation.
>
---
## 更新

#### [replaced 001] Enriched text-guided variational multimodal knowledge distillation network (VMD) for automated diagnosis of plaque vulnerability in 3D carotid artery MRI
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.11924v2](http://arxiv.org/pdf/2509.11924v2)**

> **作者:** Bo Cao; Fan Yu; Mengmeng Feng; SenHao Zhang; Xin Meng; Yue Zhang; Zhen Qian; Jie Lu
>
> **摘要:** Multimodal learning has attracted much attention in recent years due to its ability to effectively utilize data features from a variety of different modalities. Diagnosing the vulnerability of atherosclerotic plaques directly from carotid 3D MRI images is relatively challenging for both radiologists and conventional 3D vision networks. In clinical practice, radiologists assess patient conditions using a multimodal approach that incorporates various imaging modalities and domain-specific expertise, paving the way for the creation of multimodal diagnostic networks. In this paper, we have developed an effective strategy to leverage radiologists' domain knowledge to automate the diagnosis of carotid plaque vulnerability through Variation inference and Multimodal knowledge Distillation (VMD). This method excels in harnessing cross-modality prior knowledge from limited image annotations and radiology reports within training data, thereby enhancing the diagnostic network's accuracy for unannotated 3D MRI images. We conducted in-depth experiments on the dataset collected in-house and verified the effectiveness of the VMD strategy we proposed.
>
---
#### [replaced 002] Disentangling Content from Style to Overcome Shortcut Learning: A Hybrid Generative-Discriminative Learning Framework
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.11598v2](http://arxiv.org/pdf/2509.11598v2)**

> **作者:** Siming Fu; Sijun Dong; Xiaoliang Meng
>
> **摘要:** Despite the remarkable success of Self-Supervised Learning (SSL), its generalization is fundamentally hindered by Shortcut Learning, where models exploit superficial features like texture instead of intrinsic structure. We experimentally verify this flaw within the generative paradigm (e.g., MAE) and argue it is a systemic issue also affecting discriminative methods, identifying it as the root cause of their failure on unseen domains. While existing methods often tackle this at a surface level by aligning or separating domain-specific features, they fail to alter the underlying learning mechanism that fosters shortcut dependency.To address this at its core, we propose HyGDL (Hybrid Generative-Discriminative Learning Framework), a hybrid framework that achieves explicit content-style disentanglement. Our approach is guided by the Invariance Pre-training Principle: forcing a model to learn an invariant essence by systematically varying a bias (e.g., style) at the input while keeping the supervision signal constant. HyGDL operates on a single encoder and analytically defines style as the component of a representation that is orthogonal to its style-invariant content, derived via vector projection. This is operationalized through a synergistic design: (1) a self-distillation objective learns a stable, style-invariant content direction; (2) an analytical projection then decomposes the representation into orthogonal content and style vectors; and (3) a style-conditioned reconstruction objective uses these vectors to restore the image, providing end-to-end supervision. Unlike prior methods that rely on implicit heuristics, this principled disentanglement allows HyGDL to learn truly robust representations, demonstrating superior performance on benchmarks designed to diagnose shortcut learning.
>
---
#### [replaced 003] Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary Segmentation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.19331v3](http://arxiv.org/pdf/2411.19331v3)**

> **作者:** Luca Barsellotti; Lorenzo Bianchi; Nicola Messina; Fabio Carrara; Marcella Cornia; Lorenzo Baraldi; Fabrizio Falchi; Rita Cucchiara
>
> **备注:** ICCV 2025
>
> **摘要:** Open-Vocabulary Segmentation (OVS) aims at segmenting images from free-form textual concepts without predefined training classes. While existing vision-language models such as CLIP can generate segmentation masks by leveraging coarse spatial information from Vision Transformers, they face challenges in spatial localization due to their global alignment of image and text features. Conversely, self-supervised visual models like DINO excel in fine-grained visual encoding but lack integration with language. To bridge this gap, we present Talk2DINO, a novel hybrid approach that combines the spatial accuracy of DINOv2 with the language understanding of CLIP. Our approach aligns the textual embeddings of CLIP to the patch-level features of DINOv2 through a learned mapping function without the need to fine-tune the underlying backbones. At training time, we exploit the attention maps of DINOv2 to selectively align local visual patches with textual embeddings. We show that the powerful semantic and localization abilities of Talk2DINO can enhance the segmentation process, resulting in more natural and less noisy segmentations, and that our approach can also effectively distinguish foreground objects from the background. Experimental results demonstrate that Talk2DINO achieves state-of-the-art performance across several unsupervised OVS benchmarks. Source code and models are publicly available at: https://lorebianchi98.github.io/Talk2DINO/.
>
---
#### [replaced 004] Neuro Symbolic Knowledge Reasoning for Procedural Video Question Answering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.14957v4](http://arxiv.org/pdf/2503.14957v4)**

> **作者:** Thanh-Son Nguyen; Hong Yang; Tzeh Yuan Neoh; Hao Zhang; Ee Yeo Keat; Basura Fernando
>
> **摘要:** We introduce PKR-QA (Procedural Knowledge Reasoning Question Answering), a new benchmark for question answering over procedural tasks that require structured reasoning. PKR-QA is constructed semi-automatically using a procedural knowledge graph (PKG), which encodes task-specific knowledge across diverse domains. The PKG is built by curating and linking information from the COIN instructional video dataset and the ontology, enriched with commonsense knowledge from ConceptNet and structured outputs from Large Language Models (LLMs), followed by manual verification. To generate question-answer pairs, we design graph traversal templates where each template is applied systematically over PKG. To enable interpretable reasoning, we propose a neurosymbolic approach called Knowledge Module Learning (KML), which learns procedural relations via neural modules and composes them for structured reasoning with LLMs. Experiments demonstrate that this paradigm improves reasoning performance on PKR-QA and enables step-by-step reasoning traces that facilitate interpretability. Code and dataset will be released soon https://github.com/LUNAProject22/KML.
>
---
#### [replaced 005] TinyDef-DETR: A DETR-based Framework for Defect Detection in Transmission Lines from UAV Imagery
- **分类: cs.CV; cs.AI; cs.CE**

- **链接: [http://arxiv.org/pdf/2509.06035v3](http://arxiv.org/pdf/2509.06035v3)**

> **作者:** Jiaming Cui; Shuai Zhou; Feng Shen
>
> **摘要:** Automated defect detection from UAV imagery of transmission lines is a challenging task due to the small size, ambiguity, and complex backgrounds of defects. This paper proposes TinyDef-DETR, a DETR-based framework designed to achieve accurate and efficient detection of transmission line defects from UAV-acquired images. The model integrates four major components: an edge-enhanced ResNet backbone to strengthen boundary-sensitive representations, a stride-free space-to-depth module to enable detail-preserving downsampling, a cross-stage dual-domain multi-scale attention mechanism to jointly model global context and local cues, and a Focaler-Wise-SIoU regression loss to improve the localization of small and difficult targets. Together, these designs effectively mitigate the limitations of conventional detectors. Extensive experiments on both public and real-world datasets demonstrate that TinyDef-DETR achieves superior detection performance and strong generalization capability, while maintaining modest computational overhead. The accuracy and efficiency of TinyDef-DETR make it a suitable method for UAV-based transmission line defect detection, particularly in scenarios involving small and ambiguous targets.
>
---
#### [replaced 006] Compressed Video Quality Enhancement: Classifying and Benchmarking over Standards
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.10407v2](http://arxiv.org/pdf/2509.10407v2)**

> **作者:** Xiem HoangVan; Dang BuiDinh; Sang NguyenQuang; Wen-Hsiao Peng
>
> **摘要:** Compressed video quality enhancement (CVQE) is crucial for improving user experience with lossy video codecs like H.264/AVC, H.265/HEVC, and H.266/VVC. While deep learning based CVQE has driven significant progress, existing surveys still suffer from limitations: lack of systematic classification linking methods to specific standards and artifacts, insufficient comparative analysis of architectural paradigms across coding types, and underdeveloped benchmarking practices. To address these gaps, this paper presents three key contributions. First, it introduces a novel taxonomy classifying CVQE methods across architectural paradigms, coding standards, and compressed-domain feature utilization. Second, it proposes a unified benchmarking framework integrating modern compression protocols and standard test sequences for fair multi-criteria evaluation. Third, it provides a systematic analysis of the critical trade-offs between reconstruction performance and computational complexity observed in state-of-the-art methods and highlighting promising directions for future research. This comprehensive review aims to establish a foundation for consistent assessment and informed model selection in CVQE research and deployment.
>
---
#### [replaced 007] Implicit Neural Representations of Intramyocardial Motion and Strain
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.09004v3](http://arxiv.org/pdf/2509.09004v3)**

> **作者:** Andrew Bell; Yan Kit Choi; Steffen E Petersen; Andrew King; Muhummad Sohaib Nazir; Alistair A Young
>
> **备注:** STACOM 2025 @ MICCAI
>
> **摘要:** Automatic quantification of intramyocardial motion and strain from tagging MRI remains an important but challenging task. We propose a method using implicit neural representations (INRs), conditioned on learned latent codes, to predict continuous left ventricular (LV) displacement -- without requiring inference-time optimisation. Evaluated on 452 UK Biobank test cases, our method achieved the best tracking accuracy (2.14 mm RMSE) and the lowest combined error in global circumferential (2.86%) and radial (6.42%) strain compared to three deep learning baselines. In addition, our method is $\sim$380$\times$ faster than the most accurate baseline. These results highlight the suitability of INR-based models for accurate and scalable analysis of myocardial strain in large CMR datasets.
>
---
#### [replaced 008] ByDeWay: Boost Your multimodal LLM with DEpth prompting in a Training-Free Way
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.08679v2](http://arxiv.org/pdf/2507.08679v2)**

> **作者:** Rajarshi Roy; Devleena Das; Ankesh Banerjee; Arjya Bhattacharjee; Kousik Dasgupta; Subarna Tripathi
>
> **摘要:** We introduce ByDeWay, a training-free framework designed to enhance the performance of Multimodal Large Language Models (MLLMs). ByDeWay uses a novel prompting strategy called Layered-Depth-Based Prompting (LDP), which improves spatial reasoning and grounding without modifying any model parameters. It segments the scene into closest, mid-range, and farthest layers using monocular depth estimation, then generates region-specific captions with a grounded vision-language model. These structured, depth-aware captions are appended to the image-question prompt, enriching it with spatial context. This guides MLLMs to produce more grounded and less hallucinated responses. Our method is lightweight, modular, and compatible with black-box MLLMs. Experiments on hallucination-sensitive (POPE) and reasoning-intensive (GQA) benchmarks show consistent improvements across multiple MLLMs, validating the effectiveness of depth-aware prompting in a zero-training setting.
>
---
#### [replaced 009] VARCO-VISION-2.0 Technical Report
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.10105v2](http://arxiv.org/pdf/2509.10105v2)**

> **作者:** Young-rok Cha; Jeongho Ju; SunYoung Park; Jong-Hyeon Lee; Younghyun Yu; Youngjune Kim
>
> **备注:** 19 pages, 1 figure, 14 tables. Technical report for VARCO-VISION-2.0, a Korean-English bilingual VLM in 14B and 1.7B variants. Key features: multi-image understanding, OCR with text localization, improved Korean capabilities
>
> **摘要:** We introduce VARCO-VISION-2.0, an open-weight bilingual vision-language model (VLM) for Korean and English with improved capabilities compared to the previous model VARCO-VISION-14B. The model supports multi-image understanding for complex inputs such as documents, charts, and tables, and delivers layoutaware OCR by predicting both textual content and its spatial location. Trained with a four-stage curriculum with memory-efficient techniques, the model achieves enhanced multimodal alignment, while preserving core language abilities and improving safety via preference optimization. Extensive benchmark evaluations demonstrate strong spatial grounding and competitive results for both languages, with the 14B model achieving 8th place on the OpenCompass VLM leaderboard among models of comparable scale. Alongside the 14B-scale model, we release a 1.7B version optimized for on-device deployment. We believe these models advance the development of bilingual VLMs and their practical applications. Two variants of VARCO-VISION-2.0 are available at Hugging Face: a full-scale 14B model and a lightweight 1.7B model.
>
---
#### [replaced 010] BranchGRPO: Stable and Efficient GRPO with Structured Branching in Diffusion Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.06040v4](http://arxiv.org/pdf/2509.06040v4)**

> **作者:** Yuming Li; Yikai Wang; Yuying Zhu; Zhongyu Zhao; Ming Lu; Qi She; Shanghang Zhang
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Recent progress in aligning image and video generative models with Group Relative Policy Optimization (GRPO) has improved human preference alignment, but existing variants remain inefficient due to sequential rollouts and large numbers of sampling steps, unreliable credit assignment: sparse terminal rewards are uniformly propagated across timesteps, failing to capture the varying criticality of decisions during denoising. In this paper, we present BranchGRPO, a method that restructures the rollout process into a branching tree, where shared prefixes amortize computation and pruning removes low-value paths and redundant depths. BranchGRPO introduces three contributions: (1) a branching scheme that amortizes rollout cost through shared prefixes while preserving exploration diversity; (2) a reward fusion and depth-wise advantage estimator that transforms sparse terminal rewards into dense step-level signals; and (3) pruning strategies that cut gradient computation but leave forward rollouts and exploration unaffected. On HPDv2.1 image alignment, BranchGRPO improves alignment scores by up to \textbf{16\%} over DanceGRPO, while reducing per-iteration training time by nearly \textbf{55\%}. A hybrid variant, BranchGRPO-Mix, further accelerates training to 4.7x faster than DanceGRPO without degrading alignment. On WanX video generation, it further achieves higher Video-Align scores with sharper and temporally consistent frames compared to DanceGRPO. Codes are available at \href{https://fredreic1849.github.io/BranchGRPO-Webpage/}{BranchGRPO}.
>
---
#### [replaced 011] Can Generalist Vision Language Models (VLMs) Rival Specialist Medical VLMs? Benchmarking and Strategic Insights
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.17337v2](http://arxiv.org/pdf/2506.17337v2)**

> **作者:** Yuan Zhong; Ruinan Jin; Qi Dou; Xiaoxiao Li
>
> **备注:** version 2
>
> **摘要:** Vision Language Models (VLMs) have shown promise in automating image diagnosis and interpretation in clinical settings. However, developing specialist medical VLMs requires substantial computational resources and carefully curated datasets, and it remains unclear under which conditions generalist and specialist medical VLMs each perform best. This study highlights the complementary strengths of specialist medical and generalist VLMs. Specialists remain valuable in modality-aligned use cases, but we find that efficiently fine-tuned generalist VLMs can achieve comparable or even superior performance in most tasks, particularly when transferring to unseen or rare OOD medical modalities. These results suggest that generalist VLMs, rather than being constrained by their lack of specialist medical pretraining, may offer a scalable and cost-effective pathway for advancing clinical AI development.
>
---
#### [replaced 012] FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.23318v3](http://arxiv.org/pdf/2507.23318v3)**

> **作者:** Jiajun Cao; Qizhe Zhang; Peidong Jia; Xuhui Zhao; Bo Lan; Xiaoan Zhang; Zhuo Li; Xiaobao Wei; Sixiang Chen; Liyun Li; Xianming Liu; Ming Lu; Yang Wang; Shanghang Zhang
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated significant potential in complex scene understanding and action reasoning, leading to their increasing adoption in end-to-end autonomous driving systems. However, the long visual tokens of VLA models greatly increase computational costs. Current visual token pruning methods in Vision-Language Models (VLM) rely on either visual token similarity or visual-text attention, but both have shown poor performance in autonomous driving scenarios. Given that human drivers concentrate on relevant foreground areas while driving, we assert that retaining visual tokens containing this foreground information is essential for effective decision-making. Inspired by this, we propose FastDriveVLA, a novel reconstruction-based vision token pruning framework designed specifically for autonomous driving. FastDriveVLA includes a plug-and-play visual token pruner called ReconPruner, which prioritizes foreground information through MAE-style pixel reconstruction. A novel adversarial foreground-background reconstruction strategy is designed to train ReconPruner for the visual encoder of VLA models. Once trained, ReconPruner can be seamlessly applied to different VLA models with the same visual encoder without retraining. To train ReconPruner, we also introduce a large-scale dataset called nuScenes-FG, consisting of 241K image-mask pairs with annotated foreground regions. Our approach achieves state-of-the-art results on the nuScenes open-loop planning benchmark across different pruning ratios.
>
---
#### [replaced 013] Sample-Aware Test-Time Adaptation for Medical Image-to-Image Translation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.00766v2](http://arxiv.org/pdf/2508.00766v2)**

> **作者:** Irene Iele; Francesco Di Feola; Valerio Guarrasi; Paolo Soda
>
> **摘要:** Image-to-image translation has emerged as a powerful technique in medical imaging, enabling tasks such as image denoising and cross-modality conversion. However, it suffers from limitations in handling out-of-distribution samples without causing performance degradation. To address this limitation, we propose a novel Test-Time Adaptation (TTA) framework that dynamically adjusts the translation process based on the characteristics of each test sample. Our method introduces a Reconstruction Module to quantify the domain shift and a Dynamic Adaptation Block that selectively modifies the internal features of a pretrained translation model to mitigate the shift without compromising the performance on in-distribution samples that do not require adaptation. We evaluate our approach on two medical image-to-image translation tasks: low-dose CT denoising and T1 to T2 MRI translation, showing consistent improvements over both the baseline translation model without TTA and prior TTA methods. Our analysis highlights the limitations of the state-of-the-art that uniformly apply the adaptation to both out-of-distribution and in-distribution samples, demonstrating that dynamic, sample-specific adjustment offers a promising path to improve model resilience in real-world scenarios. The code is available at: https://github.com/Sample-Aware-TTA/Code.
>
---
#### [replaced 014] Pitfalls of defacing whole-head MRI: re-identification risk with diffusion models and compromised research potential
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.18834v2](http://arxiv.org/pdf/2501.18834v2)**

> **作者:** Chenyu Gao; Kaiwen Xu; Michael E. Kim; Lianrui Zuo; Zhiyuan Li; Derek B. Archer; Timothy J. Hohman; Ann Zenobia Moore; Luigi Ferrucci; Lori L. Beason-Held; Susan M. Resnick; Christos Davatzikos; Jerry L. Prince; Bennett A. Landman
>
> **备注:** Accepted to Computers in Biology and Medicine
>
> **摘要:** Defacing is often applied to head magnetic resonance image (MRI) datasets prior to public release to address privacy concerns. The alteration of facial and nearby voxels has provoked discussions about the true capability of these techniques to ensure privacy as well as their impact on downstream tasks. With advancements in deep generative models, the extent to which defacing can protect privacy is uncertain. Additionally, while the altered voxels are known to contain valuable anatomical information, their potential to support research beyond the anatomical regions directly affected by defacing remains uncertain. To evaluate these considerations, we develop a refacing pipeline that recovers faces in defaced head MRIs using cascaded diffusion probabilistic models (DPMs). The DPMs are trained on images from 180 subjects and tested on images from 484 unseen subjects, 469 of whom are from a different dataset. To assess whether the altered voxels in defacing contain universally useful information, we also predict computed tomography (CT)-derived skeletal muscle radiodensity from facial voxels in both defaced and original MRIs. The results show that DPMs can generate high-fidelity faces that resemble the original faces from defaced images, with surface distances to the original faces significantly smaller than those of a population average face (p < 0.05). This performance also generalizes well to previously unseen datasets. For skeletal muscle radiodensity predictions, using defaced images results in significantly weaker Spearman's rank correlation coefficients compared to using original images (p < 10-4). For shin muscle, the correlation is statistically significant (p < 0.05) when using original images but not statistically significant (p > 0.05) when any defacing method is applied, suggesting that defacing might not only fail to protect privacy but also eliminate valuable information.
>
---
#### [replaced 015] WorldExplorer: Towards Generating Fully Navigable 3D Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.01799v2](http://arxiv.org/pdf/2506.01799v2)**

> **作者:** Manuel-Andreas Schneider; Lukas Höllein; Matthias Nießner
>
> **备注:** Accepted to SIGGRAPH Asia 2025. Project page: see https://mschneider456.github.io/world-explorer, video: see https://youtu.be/N6NJsNyiv6I, code: https://github.com/mschneider456/WorldExplorer
>
> **摘要:** Generating 3D worlds from text is a highly anticipated goal in computer vision. Existing works are limited by the degree of exploration they allow inside of a scene, i.e., produce streched-out and noisy artifacts when moving beyond central or panoramic perspectives. To this end, we propose WorldExplorer, a novel method based on autoregressive video trajectory generation, which builds fully navigable 3D scenes with consistent visual quality across a wide range of viewpoints. We initialize our scenes by creating multi-view consistent images corresponding to a 360 degree panorama. Then, we expand it by leveraging video diffusion models in an iterative scene generation pipeline. Concretely, we generate multiple videos along short, pre-defined trajectories, that explore the scene in depth, including motion around objects. Our novel scene memory conditions each video on the most relevant prior views, while a collision-detection mechanism prevents degenerate results, like moving into objects. Finally, we fuse all generated views into a unified 3D representation via 3D Gaussian Splatting optimization. Compared to prior approaches, WorldExplorer produces high-quality scenes that remain stable under large camera motion, enabling for the first time realistic and unrestricted exploration. We believe this marks a significant step toward generating immersive and truly explorable virtual 3D environments.
>
---
#### [replaced 016] T2V-Turbo-v2: Enhancing Video Generation Model Post-Training through Data, Reward, and Conditional Guidance Design
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.05677v3](http://arxiv.org/pdf/2410.05677v3)**

> **作者:** Jiachen Li; Qian Long; Jian Zheng; Xiaofeng Gao; Robinson Piramuthu; Wenhu Chen; William Yang Wang
>
> **备注:** Accepted by ICLR 2025. Project Page: https://t2v-turbo-v2.github.io/
>
> **摘要:** In this paper, we focus on enhancing a diffusion-based text-to-video (T2V) model during the post-training phase by distilling a highly capable consistency model from a pretrained T2V model. Our proposed method, T2V-Turbo-v2, introduces a significant advancement by integrating various supervision signals, including high-quality training data, reward model feedback, and conditional guidance, into the consistency distillation process. Through comprehensive ablation studies, we highlight the crucial importance of tailoring datasets to specific learning objectives and the effectiveness of learning from diverse reward models for enhancing both the visual quality and text-video alignment. Additionally, we highlight the vast design space of conditional guidance strategies, which centers on designing an effective energy function to augment the teacher ODE solver. We demonstrate the potential of this approach by extracting motion guidance from the training datasets and incorporating it into the ODE solver, showcasing its effectiveness in improving the motion quality of the generated videos with the improved motion-related metrics from VBench and T2V-CompBench. Empirically, our T2V-Turbo-v2 establishes a new state-of-the-art result on VBench, with a Total score of 85.13, surpassing proprietary systems such as Gen-3 and Kling.
>
---
#### [replaced 017] SIFThinker: Spatially-Aware Image Focus for Visual Reasoning
- **分类: cs.CV; cs.AI; I.2.10**

- **链接: [http://arxiv.org/pdf/2508.06259v4](http://arxiv.org/pdf/2508.06259v4)**

> **作者:** Zhangquan Chen; Ruihui Zhao; Chuwei Luo; Mingze Sun; Xinlei Yu; Yangyang Kang; Ruqi Huang
>
> **备注:** 15 pages, 13 figures
>
> **摘要:** Current multimodal large language models (MLLMs) still face significant challenges in complex visual tasks (e.g., spatial understanding, fine-grained perception). Prior methods have tried to incorporate visual reasoning, however, they fail to leverage attention correction with spatial cues to iteratively refine their focus on prompt-relevant regions. In this paper, we introduce SIFThinker, a spatially-aware "think-with-images" framework that mimics human visual perception. Specifically, SIFThinker enables attention correcting and image region focusing by interleaving depth-enhanced bounding boxes and natural language. Our contributions are twofold: First, we introduce a reverse-expansion-forward-inference strategy that facilitates the generation of interleaved image-text chains of thought for process-level supervision, which in turn leads to the construction of the SIF-50K dataset. Besides, we propose GRPO-SIF, a reinforced training paradigm that integrates depth-informed visual grounding into a unified reasoning pipeline, teaching the model to dynamically correct and focus on prompt-relevant regions. Extensive experiments demonstrate that SIFThinker outperforms state-of-the-art methods in spatial understanding and fine-grained visual perception, while maintaining strong general capabilities, highlighting the effectiveness of our method. Code: https://github.com/zhangquanchen/SIFThinker.
>
---
#### [replaced 018] Gradient-Free Adversarial Purification with Diffusion Models
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2501.13336v2](http://arxiv.org/pdf/2501.13336v2)**

> **作者:** Xuelong Dai; Dong Wang; Xiuzhen Cheng; Bin Xiao
>
> **摘要:** Adversarial training and adversarial purification are two widely used defense strategies for enhancing model robustness against adversarial attacks. However, adversarial training requires costly retraining, while adversarial purification often suffers from low efficiency. More critically, existing defenses are primarily designed under the perturbation-based adversarial threat model, which is ineffective against recently introduced unrestricted adversarial attacks. In this paper, we propose an effective and efficient defense framework that counters both perturbation-based and unrestricted adversarial attacks. Our approach is motivated by the observation that adversarial examples typically lie near the decision boundary and are highly sensitive to pixel-level perturbations. To address this, we introduce adversarial anti-aliasing, a preprocessing technique that mitigates adversarial noise by reducing the magnitude of pixel-level perturbations. In addition, we propose adversarial super-resolution, which leverages prior knowledge from clean datasets to benignly restore high-quality images from adversarially degraded ones. Unlike image synthesis methods that generate entirely new images, adversarial super-resolution focuses on image restoration, making it more suitable for purification. Importantly, both techniques require no additional training and are computationally efficient since they do not rely on gradient computations. To further improve robustness across diverse datasets, we introduce a contrastive learning-based adversarial deblurring fine-tuning method. By incorporating adversarial priors during fine-tuning on the target dataset, this method enhances purification effectiveness without the need to retrain diffusion models.
>
---
#### [replaced 019] Test-Time Canonicalization by Foundation Models for Robust Perception
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.10375v2](http://arxiv.org/pdf/2507.10375v2)**

> **作者:** Utkarsh Singhal; Ryan Feng; Stella X. Yu; Atul Prakash
>
> **备注:** Published at ICML 2025
>
> **摘要:** Perception in the real world requires robustness to diverse viewing conditions. Existing approaches often rely on specialized architectures or training with predefined data augmentations, limiting adaptability. Taking inspiration from mental rotation in human vision, we propose FOCAL, a test-time robustness framework that transforms the input into the most typical view. At inference time, FOCAL explores a set of transformed images and chooses the one with the highest likelihood under foundation model priors. This test-time optimization boosts robustness while requiring no retraining or architectural changes. Applied to models like CLIP and SAM, it significantly boosts robustness across a wide range of transformations, including 2D and 3D rotations, contrast and lighting shifts, and day-night changes. We also explore potential applications in active vision. By reframing invariance as a test-time optimization problem, FOCAL offers a general and scalable approach to robustness. Our code is available at: https://github.com/sutkarsh/focal.
>
---
#### [replaced 020] Taming Anomalies with Down-Up Sampling Networks: Group Center Preserving Reconstruction for 3D Anomaly Detection
- **分类: cs.CV; 68T10; I.4; I.5; J.6**

- **链接: [http://arxiv.org/pdf/2507.03903v2](http://arxiv.org/pdf/2507.03903v2)**

> **作者:** Hanzhe Liang; Jie Zhang; Tao Dai; Linlin Shen; Jinbao Wang; Can Gao
>
> **备注:** ACM MM25 Accepted, 9 pages, 2 figures, 8 tables
>
> **摘要:** Reconstruction-based methods have demonstrated very promising results for 3D anomaly detection. However, these methods face great challenges in handling high-precision point clouds due to the large scale and complex structure. In this study, a Down-Up Sampling Network (DUS-Net) is proposed to reconstruct high-precision point clouds for 3D anomaly detection by preserving the group center geometric structure. The DUS-Net first introduces a Noise Generation module to generate noisy patches, which facilitates the diversity of training data and strengthens the feature representation for reconstruction. Then, a Down-sampling Network (Down-Net) is developed to learn an anomaly-free center point cloud from patches with noise injection. Subsequently, an Up-sampling Network (Up-Net) is designed to reconstruct high-precision point clouds by fusing multi-scale up-sampling features. Our method leverages group centers for construction, enabling the preservation of geometric structure and providing a more precise point cloud. Extensive experiments demonstrate the effectiveness of our proposed method, achieving state-of-the-art (SOTA) performance with an Object-level AUROC of 79.9% and 79.5%, and a Point-level AUROC of 71.2% and 84.7% on the Real3D-AD and Anomaly-ShapeNet datasets, respectively.
>
---
#### [replaced 021] SlumpGuard: An AI-Powered Real-Time System for Automated Concrete Slump Prediction via Video Analysis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.10171v2](http://arxiv.org/pdf/2507.10171v2)**

> **作者:** Youngmin Kim; Giyeong Oh; Kwangsoo Youm; Youngjae Yu
>
> **摘要:** Concrete workability is essential for construction quality, with the slump test being the most common on-site method for its assessment. However, traditional slump testing is manual, time-consuming, and prone to inconsistency, limiting its applicability for real-time monitoring. To address these challenges, we propose SlumpGuard, an AI-powered, video-based system that automatically analyzes concrete flow from the truck chute to assess workability in real time. Our system enables full-batch inspection without manual intervention, improving both the accuracy and efficiency of quality control. We present the system design, the construction of a dedicated dataset, and empirical results from real-world deployment, demonstrating the effectiveness of SlumpGuard as a practical solution for modern concrete quality assurance.
>
---
#### [replaced 022] ROVR-Open-Dataset: A Large-Scale Depth Dataset for Autonomous Driving
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.13977v2](http://arxiv.org/pdf/2508.13977v2)**

> **作者:** Xianda Guo; Ruijun Zhang; Yiqun Duan; Ruilin Wang; Matteo Poggi; Keyuan Zhou; Wenzhao Zheng; Wenke Huang; Gangwei Xu; Mike Horton; Yuan Si; Qin Zou; Hao Zhao; Long Chen
>
> **摘要:** Depth estimation is a fundamental task for 3D scene understanding in autonomous driving, robotics, and augmented reality. Existing depth datasets, such as KITTI, nuScenes, and DDAD, have advanced the field but suffer from limitations in diversity and scalability. As benchmark performance on these datasets approaches saturation, there is an increasing need for a new generation of large-scale, diverse, and cost-efficient datasets to support the era of foundation models and multi-modal learning. We present ROVR, a large-scale, diverse, and cost-efficient depth dataset designed to capture the complexity of real-world driving. ROVR comprises 200K high-resolution frames across highway, rural, and urban scenarios, spanning day/night and adverse weather conditions. A lightweight acquisition pipeline ensures scalable collection, while sparse but statistically sufficient ground truth supports robust training. Benchmarking with state-of-the-art monocular depth models reveals severe cross-dataset generalization failures: models achieving near-ceiling accuracy on KITTI degrade drastically on ROVR, and even when trained on ROVR, current methods fall short of saturation. These results highlight the unique challenges posed by ROVR-scene diversity, dynamic environments, and sparse ground truth, establishing it as a demanding new platform for advancing depth estimation and building models with stronger real-world robustness. Extensive ablation studies provide a more intuitive understanding of our dataset across different scenarios, lighting conditions, and generalized ability.
>
---
#### [replaced 023] SFGNet: Semantic and Frequency Guided Network for Camouflaged Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.11539v2](http://arxiv.org/pdf/2509.11539v2)**

> **作者:** Dezhen Wang; Haixiang Zhao; Xiang Shen; Sheng Miao
>
> **备注:** Submitted to ICASSP 2026 by Dezhen Wang et al. Copyright 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, including reprinting/republishing, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work. DOI will be added upon IEEE Xplore publication
>
> **摘要:** Camouflaged object detection (COD) aims to segment objects that blend into their surroundings. However, most existing studies overlook the semantic differences among textual prompts of different targets as well as fine-grained frequency features. In this work, we propose a novel Semantic and Frequency Guided Network (SFGNet), which incorporates semantic prompts and frequency-domain features to capture camouflaged objects and improve boundary perception. We further design Multi-Band Fourier Module(MBFM) to enhance the ability of the network in handling complex backgrounds and blurred boundaries. In addition, we design an Interactive Structure Enhancement Block (ISEB) to ensure structural integrity and boundary details in the predictions. Extensive experiments conducted on three COD benchmark datasets demonstrate that our method significantly outperforms state-of-the-art approaches. The core code of the model is available at the following link: https://github.com/winter794444/SFGNetICASSP2026.
>
---
#### [replaced 024] Learning Environment-Aware Affordance for 3D Articulated Object Manipulation under Occlusions
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2309.07510v5](http://arxiv.org/pdf/2309.07510v5)**

> **作者:** Ruihai Wu; Kai Cheng; Yan Shen; Chuanruo Ning; Guanqi Zhan; Hao Dong
>
> **备注:** In 37th Conference on Neural Information Processing Systems (NeurIPS 2023). Website at https://chengkaiacademycity.github.io/EnvAwareAfford/
>
> **摘要:** Perceiving and manipulating 3D articulated objects in diverse environments is essential for home-assistant robots. Recent studies have shown that point-level affordance provides actionable priors for downstream manipulation tasks. However, existing works primarily focus on single-object scenarios with homogeneous agents, overlooking the realistic constraints imposed by the environment and the agent's morphology, e.g., occlusions and physical limitations. In this paper, we propose an environment-aware affordance framework that incorporates both object-level actionable priors and environment constraints. Unlike object-centric affordance approaches, learning environment-aware affordance faces the challenge of combinatorial explosion due to the complexity of various occlusions, characterized by their quantities, geometries, positions and poses. To address this and enhance data efficiency, we introduce a novel contrastive affordance learning framework capable of training on scenes containing a single occluder and generalizing to scenes with complex occluder combinations. Experiments demonstrate the effectiveness of our proposed approach in learning affordance considering environment constraints. Project page at https://chengkaiacademycity.github.io/EnvAwareAfford/
>
---
#### [replaced 025] Evaluating the Robustness of Open-Source Vision-Language Models to Domain Shift in Object Captioning
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.19579v2](http://arxiv.org/pdf/2506.19579v2)**

> **作者:** Federico Tavella; Amber Drinkwater; Angelo Cangelosi
>
> **摘要:** Vision-Language Models (VLMs) have emerged as powerful tools for generating textual descriptions from visual data. While these models excel on web-scale datasets, their robustness to the domain shifts inherent in many real-world applications remains under-explored. This paper presents a systematic evaluation of VLM performance on a single-view object captioning task when faced with a controlled, physical domain shift. We compare captioning accuracy across two distinct object sets: a collection of multi-material, real-world tools and a set of single-material, 3D-printed items. The 3D-printed set introduces a significant domain shift in texture and material properties, challenging the models' generalization capabilities. Our quantitative results demonstrate that all tested VLMs show a marked performance degradation when describing the 3D-printed objects compared to the real-world tools. This underscores a critical limitation in the ability of current models to generalize beyond surface-level features and highlights the need for more robust architectures for real-world signal processing applications.
>
---
#### [replaced 026] Plane Detection and Ranking via Model Information Optimization
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.09625v2](http://arxiv.org/pdf/2508.09625v2)**

> **作者:** Daoxin Zhong; Jun Li; Meng Yee Michael Chuah
>
> **备注:** Accepted as contributed paper in the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Plane detection from depth images is a crucial subtask with broad robotic applications, often accomplished by iterative methods such as Random Sample Consensus (RANSAC). While RANSAC is a robust strategy with strong probabilistic guarantees, the ambiguity of its inlier threshold criterion makes it susceptible to false positive plane detections. This issue is particularly prevalent in complex real-world scenes, where the true number of planes is unknown and multiple planes coexist. In this paper, we aim to address this limitation by proposing a generalised framework for plane detection based on model information optimization. Building on previous works, we treat the observed depth readings as discrete random variables, with their probability distributions constrained by the ground truth planes. Various models containing different candidate plane constraints are then generated through repeated random sub-sampling to explain our observations. By incorporating the physics and noise model of the depth sensor, we can calculate the information for each model, and the model with the least information is accepted as the most likely ground truth. This information optimization process serves as an objective mechanism for determining the true number of planes and preventing false positive detections. Additionally, the quality of each detected plane can be ranked by summing the information reduction of inlier points for each plane. We validate these properties through experiments with synthetic data and find that our algorithm estimates plane parameters more accurately compared to the default Open3D RANSAC plane segmentation. Furthermore, we accelerate our algorithm by partitioning the depth map using neural network segmentation, which enhances its ability to generate more realistic plane parameters in real-world data.
>
---
#### [replaced 027] MSMA: Multi-Scale Feature Fusion For Multi-Attribute 3D Face Reconstruction From Unconstrained Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.11763v2](http://arxiv.org/pdf/2509.11763v2)**

> **作者:** Danling Cao
>
> **摘要:** Reconstructing 3D face from a single unconstrained image remains a challenging problem due to diverse conditions in unconstrained environments. Recently, learning-based methods have achieved notable results by effectively capturing complex facial structures and details across varying conditions. Consequently, many existing approaches employ projection-based losses between generated and input images to constrain model training. However, learning-based methods for 3D face reconstruction typically require substantial amounts of 3D facial data, which is difficult and costly to obtain. Consequently, to reduce reliance on labeled 3D face datasets, many existing approaches employ projection-based losses between generated and input images to constrain model training. Nonetheless, despite these advancements, existing approaches frequently struggle to capture detailed and multi-scale features under diverse facial attributes and conditions, leading to incomplete or less accurate reconstructions. In this paper, we propose a Multi-Scale Feature Fusion with Multi-Attribute (MSMA) framework for 3D face reconstruction from unconstrained images. Our method integrates multi-scale feature fusion with a focus on multi-attribute learning and leverages a large-kernel attention module to enhance the precision of feature extraction across scales, enabling accurate 3D facial parameter estimation from a single 2D image. Comprehensive experiments on the MICC Florence, Facewarehouse and custom-collect datasets demonstrate that our approach achieves results on par with current state-of-the-art methods, and in some instances, surpasses SOTA performance across challenging conditions.
>
---
#### [replaced 028] CryoSplat: Gaussian Splatting for Cryo-EM Homogeneous Reconstruction
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.04929v2](http://arxiv.org/pdf/2508.04929v2)**

> **作者:** Suyi Chen; Haibin Ling
>
> **摘要:** As a critical modality for structural biology, cryogenic electron microscopy (cryo-EM) facilitates the determination of macromolecular structures at near-atomic resolution. The core computational task in single-particle cryo-EM is to reconstruct the 3D electrostatic potential of a molecule from a large collection of noisy 2D projections acquired at unknown orientations. Gaussian mixture models (GMMs) provide a continuous, compact, and physically interpretable representation for molecular density and have recently gained interest in cryo-EM reconstruction. However, existing methods rely on external consensus maps or atomic models for initialization, limiting their use in self-contained pipelines. Addressing this issue, we introduce cryoGS, a GMM-based method that integrates Gaussian splatting with the physics of cryo-EM image formation. In particular, we develop an orthogonal projection-aware Gaussian splatting, with adaptations such as a normalization term and FFT-aligned coordinate system tailored for cryo-EM imaging. All these innovations enable stable and efficient homogeneous reconstruction directly from raw cryo-EM particle images using random initialization. Experimental results on real datasets validate the effectiveness and robustness of cryoGS over representative baselines. The code will be released upon publication.
>
---
#### [replaced 029] ALL-PET: A Low-resource and Low-shot PET Foundation Model in Projection Domain
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.09130v2](http://arxiv.org/pdf/2509.09130v2)**

> **作者:** Bin Huang; Kang Chen; Bingxuan Li; Huafeng Liu; Qiegen Liu
>
> **摘要:** Building large-scale foundation model for PET imaging is hindered by limited access to labeled data and insufficient computational resources. To overcome data scarcity and efficiency limitations, we propose ALL-PET, a low-resource, low-shot PET foundation model operating directly in projection domain. ALL-PET leverages a latent diffusion model (LDM) with three key innovations. First, we design a Radon mask augmentation strategy (RMAS) that generates over 200,000 structurally diverse training samples by projecting randomized image-domain masks into sinogram space, significantly improving generalization with minimal data. This is extended by a dynamic multi-mask (DMM) mechanism that varies mask quantity and distribution, enhancing data diversity without added model complexity. Second, we implement positive/negative mask constraints to embed strict geometric consistency, reducing parameter burden while preserving generation quality. Third, we introduce transparent medical attention (TMA), a parameter-free, geometry-driven mechanism that enhances lesion-related regions in raw projection data. Lesion-focused attention maps are derived from coarse segmentation, covering both hypermetabolic and hypometabolic areas, and projected into sinogram space for physically consistent guidance. The system supports clinician-defined ROI adjustments, ensuring flexible, interpretable, and task-adaptive emphasis aligned with PET acquisition physics. Experimental results show that ALL-PET achieves high-quality sinogram generation using only 500 samples, with performance comparable to models trained on larger datasets. ALL-PET generalizes across tasks including low-dose reconstruction, attenuation correction, delayed-frame prediction, and tracer separation, operating efficiently with memory use under 24GB.
>
---
#### [replaced 030] Dynamic Relation Inference via Verb Embeddings
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.13021v2](http://arxiv.org/pdf/2503.13021v2)**

> **作者:** Omri Suissa; Muhiim Ali; Ariana Azarbal; Hui Shen; Shekhar Pradhan
>
> **摘要:** CLIP has demonstrated exceptional image-text matching capabilities due to its training on contrastive learning tasks. Past research has suggested that whereas CLIP effectively matches text to images when the matching can be achieved just by matching the text with the objects in the image, CLIP struggles when the matching depends on representing the relationship among the objects in the images (i.e., inferring relations). Previous attempts to address this limitation by training CLIP on relation detection datasets with only linguistic supervision have met with limited success. In this paper, we offer insights and practical methods to advance the field of relation inference from images. This paper approaches the task of creating a model that effectively detects relations among the objects in images by producing text and image embeddings that capture relationships through linguistic supervision. To this end, we propose Dynamic Relation Inference via Verb Embeddings (DRIVE), which augments the COCO dataset, fine-tunes CLIP with hard negatives subject-relation-object triples and corresponding images, and introduces a novel loss function to improve relation detection. Evaluated on multiple CLIP-based models, our method significantly improves zero-shot relation inference accuracy in both frozen and fine-tuned settings, significantly outperforming CLIP and state-of-the-art models while generalizing well on unseen data.
>
---
#### [replaced 031] IMPROVE: Iterative Model Pipeline Refinement and Optimization Leveraging LLM Experts
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.18530v3](http://arxiv.org/pdf/2502.18530v3)**

> **作者:** Eric Xue; Ke Chen; Zeyi Huang; Yuyang Ji; Haohan Wang
>
> **摘要:** Large language model (LLM) agents have emerged as a promising solution to automate the workflow of machine learning, but most existing methods share a common limitation: they attempt to optimize entire pipelines in a single step before evaluation, making it difficult to attribute improvements to specific changes. This lack of granularity leads to unstable optimization and slower convergence, limiting their effectiveness. To address this, we introduce Iterative Refinement, a novel strategy for LLM-driven ML pipeline design inspired by how human ML experts iteratively refine models, focusing on one component at a time rather than making sweeping changes all at once. By systematically updating individual components based on real training feedback, Iterative Refinement improves overall model performance. We also provide some theoretical edvience of the superior properties of this Iterative Refinement. Further, we implement this strategy in IMPROVE, an end-to-end LLM agent framework for automating and optimizing object classification pipelines. Through extensive evaluations across datasets of varying sizes and domains, we demonstrate that Iterative Refinement enables IMPROVE to consistently achieve better performance over existing zero-shot LLM-based approaches.
>
---
#### [replaced 032] HoloDx: Knowledge- and Data-Driven Multimodal Diagnosis of Alzheimer's Disease
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.19075v2](http://arxiv.org/pdf/2504.19075v2)**

> **作者:** Qiuhui Chen; Jintao Wang; Gang Wang; Yi Hong
>
> **备注:** Accepted by IEEE Transactions on Medical Imaging (TMI)
>
> **摘要:** Accurate diagnosis of Alzheimer's disease (AD) requires effectively integrating multimodal data and clinical expertise. However, existing methods often struggle to fully utilize multimodal information and lack structured mechanisms to incorporate dynamic domain knowledge. To address these limitations, we propose HoloDx, a knowledge- and data-driven framework that enhances AD diagnosis by aligning domain knowledge with multimodal clinical data. HoloDx incorporates a knowledge injection module with a knowledge-aware gated cross-attention, allowing the model to dynamically integrate domain-specific insights from both large language models (LLMs) and clinical expertise. Also, a memory injection module with a designed prototypical memory attention enables the model to retain and retrieve subject-specific information, ensuring consistency in decision-making. By jointly leveraging these mechanisms, HoloDx enhances interpretability, improves robustness, and effectively aligns prior knowledge with current subject data. Evaluations on five AD datasets demonstrate that HoloDx outperforms state-of-the-art methods, achieving superior diagnostic accuracy and strong generalization across diverse cohorts. The source code will be released upon publication acceptance.
>
---
#### [replaced 033] RingMo-Aerial: An Aerial Remote Sensing Foundation Model With Affine Transformation Contrastive Learning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.13366v4](http://arxiv.org/pdf/2409.13366v4)**

> **作者:** Wenhui Diao; Haichen Yu; Kaiyue Kang; Tong Ling; Di Liu; Yingchao Feng; Hanbo Bi; Libo Ren; Xuexue Li; Yongqiang Mao; Xian Sun
>
> **摘要:** Aerial Remote Sensing (ARS) vision tasks present significant challenges due to the unique viewing angle characteristics. Existing research has primarily focused on algorithms for specific tasks, which have limited applicability in a broad range of ARS vision applications. This paper proposes RingMo-Aerial, aiming to fill the gap in foundation model research in the field of ARS vision. A Frequency-Enhanced Multi-Head Self-Attention (FE-MSA) mechanism is introduced to strengthen the model's capacity for small-object representation. Complementarily, an affine transformation-based contrastive learning method improves its adaptability to the tilted viewing angles inherent in ARS tasks. Furthermore, the ARS-Adapter, an efficient parameter fine-tuning method, is proposed to improve the model's adaptability and performance in various ARS vision tasks. Experimental results demonstrate that RingMo-Aerial achieves SOTA performance on multiple downstream tasks. This indicates the practicality and efficacy of RingMo-Aerial in enhancing the performance of ARS vision tasks.
>
---
#### [replaced 034] Optimal Transport Based Unsupervised Restoration Learning Exploiting Degradation Sparsity
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2305.00273v2](http://arxiv.org/pdf/2305.00273v2)**

> **作者:** Fei Wen; Wei Wang; Zeyu Yan; Wenbin Jiang
>
> **备注:** 15 pages, 9 figures
>
> **摘要:** Optimal transport (OT) has recently been shown as a promising criterion for unsupervised restoration when no explicit prior model is available. Despite its theoretical appeal, OT still significantly falls short of supervised methods on challenging tasks such as super-resolution, deraining, and dehazing. In this paper, we propose a \emph{sparsity-aware optimal transport} (SOT) framework to bridge this gap by leveraging a key observation: the degradations in these tasks exhibit distinct sparsity in the frequency domain. Incorporating this sparsity prior into OT can significantly reduce the ambiguity of the inverse mapping for restoration and substantially boost performance. We provide analysis to show exploiting degradation sparsity benefits unsupervised restoration learning. Extensive experiments on real-world super-resolution, deraining, and dehazing demonstrate that SOT offers notable performance gains over standard OT, while achieving superior perceptual quality compared to existing supervised and unsupervised methods. In particular, SOT consistently outperforms existing unsupervised methods across all three tasks and narrows the performance gap to supervised counterparts.
>
---
#### [replaced 035] Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.13773v2](http://arxiv.org/pdf/2507.13773v2)**

> **作者:** Pu Jian; Donglei Yu; Wen Yang; Shuo Ren; Jiajun Zhang
>
> **备注:** ACL2025 Main (SAC Highlight Award)
>
> **摘要:** In visual question answering (VQA) context, users often pose ambiguous questions to visual language models (VLMs) due to varying expression habits. Existing research addresses such ambiguities primarily by rephrasing questions. These approaches neglect the inherently interactive nature of user interactions with VLMs, where ambiguities can be clarified through user feedback. However, research on interactive clarification faces two major challenges: (1) Benchmarks are absent to assess VLMs' capacity for resolving ambiguities through interaction; (2) VLMs are trained to prefer answering rather than asking, preventing them from seeking clarification. To overcome these challenges, we introduce \textbf{ClearVQA} benchmark, which targets three common categories of ambiguity in VQA context, and encompasses various VQA scenarios.
>
---
#### [replaced 036] Think Before You Diffuse: LLMs-Guided Physics-Aware Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21653v2](http://arxiv.org/pdf/2505.21653v2)**

> **作者:** Ke Zhang; Cihan Xiao; Jiacong Xu; Yiqun Mei; Vishal M. Patel
>
> **备注:** 19 pages, 8 figures
>
> **摘要:** Recent video diffusion models have demonstrated their great capability in generating visually-pleasing results, while synthesizing the correct physical effects in generated videos remains challenging. The complexity of real-world motions, interactions, and dynamics introduce great difficulties when learning physics from data. In this work, we propose DiffPhy, a generic framework that enables physically-correct and photo-realistic video generation by fine-tuning a pre-trained video diffusion model. Our method leverages large language models (LLMs) to explicitly reason a comprehensive physical context from the text prompt and use it to guide the generation. To incorporate physical context into the diffusion model, we leverage a Multimodal large language model (MLLM) as a supervisory signal and introduce a set of novel training objectives that jointly enforce physical correctness and semantic consistency with the input text. We also establish a high-quality physical video dataset containing diverse phyiscal actions and events to facilitate effective finetuning. Extensive experiments on public benchmarks demonstrate that DiffPhy is able to produce state-of-the-art results across diverse physics-related scenarios. Our project page is available at https://bwgzk-keke.github.io/DiffPhy/
>
---
#### [replaced 037] Zero-shot Hierarchical Plant Segmentation via Foundation Segmentation Models and Text-to-image Attention
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.09116v2](http://arxiv.org/pdf/2509.09116v2)**

> **作者:** Junhao Xing; Ryohei Miyakawa; Yang Yang; Xinpeng Liu; Risa Shinoda; Hiroaki Santo; Yosuke Toda; Fumio Okura
>
> **备注:** WACV 2026 Accepted
>
> **摘要:** Foundation segmentation models achieve reasonable leaf instance extraction from top-view crop images without training (i.e., zero-shot). However, segmenting entire plant individuals with each consisting of multiple overlapping leaves remains challenging. This problem is referred to as a hierarchical segmentation task, typically requiring annotated training datasets, which are often species-specific and require notable human labor. To address this, we introduce ZeroPlantSeg, a zero-shot segmentation for rosette-shaped plant individuals from top-view images. We integrate a foundation segmentation model, extracting leaf instances, and a vision-language model, reasoning about plants' structures to extract plant individuals without additional training. Evaluations on datasets with multiple plant species, growth stages, and shooting environments demonstrate that our method surpasses existing zero-shot methods and achieves better cross-domain performance than supervised methods. Implementations are available at https://github.com/JunhaoXing/ZeroPlantSeg.
>
---
#### [replaced 038] Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13061v4](http://arxiv.org/pdf/2502.13061v4)**

> **作者:** Jingbiao Mei; Jinghong Chen; Guangyu Yang; Weizhe Lin; Bill Byrne
>
> **备注:** EMNLP 2025 Main (Oral)
>
> **摘要:** Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While Large Multimodal Models (LMMs) have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both supervised fine-tuning (SFT) and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Analysis reveals that our approach achieves improved robustness under adversarial attacks compared to SFT models. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems. Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability. Code available at https://github.com/JingbiaoMei/RGCL
>
---
#### [replaced 039] ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.22159v2](http://arxiv.org/pdf/2505.22159v2)**

> **作者:** Jiawen Yu; Hairuo Liu; Qiaojun Yu; Jieji Ren; Ce Hao; Haitong Ding; Guangyu Huang; Guofan Huang; Yan Song; Panpan Cai; Cewu Lu; Wenqiang Zhang
>
> **摘要:** Vision-Language-Action (VLA) models have advanced general-purpose robotic manipulation by leveraging pretrained visual and linguistic representations. However, they struggle with contact-rich tasks that require fine-grained control involving force, especially under visual occlusion or dynamic uncertainty. To address these limitations, we propose ForceVLA, a novel end-to-end manipulation framework that treats external force sensing as a first-class modality within VLA systems. ForceVLA introduces FVLMoE, a force-aware Mixture-of-Experts fusion module that dynamically integrates pretrained visual-language embeddings with real-time 6-axis force feedback during action decoding. This enables context-aware routing across modality-specific experts, enhancing the robot's ability to adapt to subtle contact dynamics. We also introduce \textbf{ForceVLA-Data}, a new dataset comprising synchronized vision, proprioception, and force-torque signals across five contact-rich manipulation tasks. ForceVLA improves average task success by 23.2% over strong pi_0-based baselines, achieving up to 80% success in tasks such as plug insertion. Our approach highlights the importance of multimodal integration for dexterous manipulation and sets a new benchmark for physically intelligent robotic control. Code and data will be released at https://sites.google.com/view/forcevla2025.
>
---
#### [replaced 040] Diagnosis for Less-Prevalent Thyroid Carcinoma Subtype Using a Dual-Branch Attention Deep Network with Ultrasound Images
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.02211v2](http://arxiv.org/pdf/2505.02211v2)**

> **作者:** Peiqi Li; Yincheng Gao; Renxing Li; Haojie Yang; Yunyun Liu; Boji Liu; Jiahui Ni; Ying Zhang; Yulu Wu; Xiaowei Fang; Lehang Guo; Liping Sun; Jiangang Chen
>
> **备注:** 15 pages, 7 figures, 4 tables
>
> **摘要:** Heterogeneous morphological features and data imbalance pose significant challenges in rare thyroid carcinoma classification using ultrasound imaging. To address this issue, we propose a novel multitask learning framework, Channel-Spatial Attention Synergy Network (CSASN), which integrates a dual-branch feature extractor - combining EfficientNet for local spatial encoding and ViT for global semantic modeling, with a cascaded channel-spatial attention refinement module. A residual multiscale classifier and dynamically weighted loss function further enhance classification stability and accuracy. Trained on a multicenter dataset comprising more than 2000 patients from four clinical institutions, our framework leverages a residual multiscale classifier and dynamically weighted loss function to enhance classification stability and accuracy. Extensive ablation studies demonstrate that each module contributes significantly to model performance, particularly in recognizing rare subtypes such as FTC and MTC carcinomas. Experimental results show that CSASN outperforms existing single-stream CNN or Transformer-based models, achieving a superior balance between precision and recall under class-imbalanced conditions. This framework provides a promising strategy for AI-assisted thyroid cancer diagnosis.
>
---
#### [replaced 041] Revisiting Transferable Adversarial Images: Systemization, Evaluation, and New Insights
- **分类: cs.CR; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2310.11850v2](http://arxiv.org/pdf/2310.11850v2)**

> **作者:** Zhengyu Zhao; Hanwei Zhang; Renjue Li; Ronan Sicre; Laurent Amsaleg; Michael Backes; Qi Li; Qian Wang; Chao Shen
>
> **备注:** TPAMI 2025. Code is available at https://github.com/ZhengyuZhao/TransferAttackEval
>
> **摘要:** Transferable adversarial images raise critical security concerns for computer vision systems in real-world, black-box attack scenarios. Although many transfer attacks have been proposed, existing research lacks a systematic and comprehensive evaluation. In this paper, we systemize transfer attacks into five categories around the general machine learning pipeline and provide the first comprehensive evaluation, with 23 representative attacks against 11 representative defenses, including the recent, transfer-oriented defense and the real-world Google Cloud Vision. In particular, we identify two main problems of existing evaluations: (1) for attack transferability, lack of intra-category analyses with fair hyperparameter settings, and (2) for attack stealthiness, lack of diverse measures. Our evaluation results validate that these problems have indeed caused misleading conclusions and missing points, and addressing them leads to new, \textit{consensus-challenging} insights, such as (1) an early attack, DI, even outperforms all similar follow-up ones, (2) the state-of-the-art (white-box) defense, DiffPure, is even vulnerable to (black-box) transfer attacks, and (3) even under the same $L_p$ constraint, different attacks yield dramatically different stealthiness results regarding diverse imperceptibility metrics, finer-grained measures, and a user study. We hope that our analyses will serve as guidance on properly evaluating transferable adversarial images and advance the design of attacks and defenses. Code is available at https://github.com/ZhengyuZhao/TransferAttackEval.
>
---
#### [replaced 042] MEIL-NeRF: Memory-Efficient Incremental Learning of Neural Radiance Fields
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2212.08328v3](http://arxiv.org/pdf/2212.08328v3)**

> **作者:** Jaeyoung Chung; Kanggeon Lee; Sungyong Baik; Kyoung Mu Lee
>
> **备注:** 10 pages for main paper, additional 7 pages for supple. For the project page, see https://robot0321.github.io/meil-nerf/index.html
>
> **摘要:** Hinged on the representation power of neural networks, neural radiance fields (NeRF) have recently emerged as one of the promising and widely applicable methods for 3D object and scene representation. However, NeRF faces challenges in practical applications, such as large-scale scenes and edge devices with a limited amount of memory, where data needs to be processed sequentially. Under such incremental learning scenarios, neural networks are known to suffer catastrophic forgetting: easily forgetting previously seen data after training with new data. We observe that previous incremental learning algorithms are limited by either low performance or memory scalability issues. As such, we develop a Memory-Efficient Incremental Learning algorithm for NeRF (MEIL-NeRF). MEIL-NeRF takes inspiration from NeRF itself in that a neural network can serve as a memory that provides the pixel RGB values, given rays as queries. Upon the motivation, our framework learns which rays to query NeRF to extract previous pixel values. The extracted pixel values are then used to train NeRF in a self-distillation manner to prevent catastrophic forgetting. As a result, MEIL-NeRF demonstrates constant memory consumption and competitive performance.
>
---
#### [replaced 043] PromptEnhancer: A Simple Approach to Enhance Text-to-Image Models via Chain-of-Thought Prompt Rewriting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.04545v4](http://arxiv.org/pdf/2509.04545v4)**

> **作者:** Linqing Wang; Ximing Xing; Yiji Cheng; Zhiyuan Zhao; Jiale Tao; Qixun Wang; Ruihuang Li; Comi Chen; Xin Li; Mingrui Wu; Xinchi Deng; Chunyu Wang; Qinglin Lu
>
> **备注:** Technical Report. Project Page: https://hunyuan-promptenhancer.github.io/
>
> **摘要:** Recent advancements in text-to-image (T2I) diffusion models have demonstrated remarkable capabilities in generating high-fidelity images. However, these models often struggle to faithfully render complex user prompts, particularly in aspects like attribute binding, negation, and compositional relationships. This leads to a significant mismatch between user intent and the generated output. To address this challenge, we introduce PromptEnhancer, a novel and universal prompt rewriting framework that enhances any pretrained T2I model without requiring modifications to its weights. Unlike prior methods that rely on model-specific fine-tuning or implicit reward signals like image-reward scores, our framework decouples the rewriter from the generator. We achieve this by training a Chain-of-Thought (CoT) rewriter through reinforcement learning, guided by a dedicated reward model we term the AlignEvaluator. The AlignEvaluator is trained to provide explicit and fine-grained feedback based on a systematic taxonomy of 24 key points, which are derived from a comprehensive analysis of common T2I failure modes. By optimizing the CoT rewriter to maximize the reward from our AlignEvaluator, our framework learns to generate prompts that are more precisely interpreted by T2I models. Extensive experiments on the HunyuanImage 2.1 model demonstrate that PromptEnhancer significantly improves image-text alignment across a wide range of semantic and compositional challenges. Furthermore, we introduce a new, high-quality human preference benchmark to facilitate future research in this direction.
>
---
#### [replaced 044] Hierarchical MLANet: Multi-level Attention for 3D Face Reconstruction From Single Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.10024v2](http://arxiv.org/pdf/2509.10024v2)**

> **作者:** Danling Cao
>
> **备注:** This work was completed during the author's MPhil studies at the University of Manchester
>
> **摘要:** Recovering 3D face models from 2D in-the-wild images has gained considerable attention in the computer vision community due to its wide range of potential applications. However, the lack of ground-truth labeled datasets and the complexity of real-world environments remain significant challenges. In this chapter, we propose a convolutional neural network-based approach, the Hierarchical Multi-Level Attention Network (MLANet), for reconstructing 3D face models from single in-the-wild images. Our model predicts detailed facial geometry, texture, pose, and illumination parameters from a single image. Specifically, we employ a pre-trained hierarchical backbone network and introduce multi-level attention mechanisms at different stages of 2D face image feature extraction. A semi-supervised training strategy is employed, incorporating 3D Morphable Model (3DMM) parameters from publicly available datasets along with a differentiable renderer, enabling an end-to-end training process. Extensive experiments, including both comparative and ablation studies, were conducted on two benchmark datasets, AFLW2000-3D and MICC Florence, focusing on 3D face reconstruction and 3D face alignment tasks. The effectiveness of the proposed method was evaluated both quantitatively and qualitatively.
>
---
#### [replaced 045] Detection of Synthetic Face Images: Accuracy, Robustness, Generalization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.17547v2](http://arxiv.org/pdf/2406.17547v2)**

> **作者:** Nela Petrzelkova; Jan Cech
>
> **备注:** The paper was presented at the DAGM German Conference on Pattern Recognition (GCPR), 2025
>
> **摘要:** An experimental study on detecting synthetic face images is presented. We collected a dataset, called FF5, of five fake face image generators, including recent diffusion models. We find that a simple model trained on a specific image generator can achieve near-perfect accuracy in separating synthetic and real images. The model handles common image distortions (reduced resolution, compression) by using data augmentation. Moreover, partial manipulations, where synthetic images are blended into real ones by inpainting, are identified and the area of the manipulation is localized by a simple model of YOLO architecture. However, the model turned out to be vulnerable to adversarial attacks and does not generalize to unseen generators. Failure to generalize to detect images produced by a newer generator also occurs for recent state-of-the-art methods, which we tested on Realistic Vision, a fine-tuned version of StabilityAI's Stable Diffusion image generator.
>
---
#### [replaced 046] Cross-Image Contrastive Decoding: Precise, Lossless Suppression of Language Priors in Large Vision-Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10634v5](http://arxiv.org/pdf/2505.10634v5)**

> **作者:** Jianfei Zhao; Feng Zhang; Xin Sun; Lingxing Kong; Zhixing Tan; Chong Feng
>
> **备注:** Under Review
>
> **摘要:** Over-reliance on language priors is a major cause of hallucinations in Large Vision-Language Models (LVLMs), often leading to outputs that are linguistically plausible but visually inconsistent. Recent studies have explored contrastive decoding as a training-free solution. However, these methods typically construct contrastive visual inputs by perturbing the original image, resulting in distorted contrastive distributions, incomplete contrastive signals, and excessive suppression of language priors. Motivated by the observation that language priors tend to remain consistent across different images, we propose Cross-Image Contrastive Decoding (CICD), a simple yet effective training-free method that uses unrelated images as contrastive visual inputs. To address the issue of over-suppressing language priors, which can negatively affect the quality of generated responses, we further introduce a dynamic selection mechanism based on the cross-image differences in model behavior. By selectively suppressing language priors, our method reduces hallucinations without compromising the model's performance. Extensive experiments across multiple benchmarks and LVLMs confirm the effectiveness and generalizability of CICD, particularly in image captioning, where language priors are especially dominant.
>
---
#### [replaced 047] HierRelTriple: Guiding Indoor Layout Generation with Hierarchical Relationship Triplet Losses
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.20289v2](http://arxiv.org/pdf/2503.20289v2)**

> **作者:** Kaifan Sun; Bingchen Yang; Peter Wonka; Jun Xiao; Haiyong Jiang
>
> **摘要:** We present a hierarchical triplet-based indoor relationship learning method, coined HierRelTriple, with a focus on spatial relationship learning. Existing approaches often depend on manually defined spatial rules or simplified pairwise representations, which fail to capture complex, multi-object relationships found in real scenarios and lead to overcrowded or physically implausible arrangements. We introduce HierRelTriple, a hierarchical relational triplets modeling framework that first partitions functional regions and then automatically extracts three levels of spatial relationships: object-to-region (O2R), object-to-object (O2O), and corner-to-corner (C2C). By representing these relationships as geometric triplets and employing approaches based on Delaunay Triangulation to establish spatial priors, we derive IoU loss between denoised and ground truth triplets and integrate them seamlessly into the diffusion denoising process. The introduction of the joint formulation of inter-object distances, angular orientations, and spatial relationships enhances the physical realism of the generated scenes. Extensive experiments on unconditional layout synthesis, floorplan-conditioned layout generation, and scene rearrangement demonstrate that HierRelTriple improves spatial-relation metrics by over 15% and substantially reduces collisions and boundary violations compared to state-of-the-art methods.
>
---
#### [replaced 048] TransDiffuser: Diverse Trajectory Generation with Decorrelated Multi-modal Representation for End-to-end Autonomous Driving
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.09315v2](http://arxiv.org/pdf/2505.09315v2)**

> **作者:** Xuefeng Jiang; Yuan Ma; Pengxiang Li; Leimeng Xu; Xin Wen; Kun Zhan; Zhongpu Xia; Peng Jia; Xianpeng Lang; Sheng Sun
>
> **备注:** Under review
>
> **摘要:** In recent years, diffusion models have demonstrated remarkable potential across diverse domains, from vision generation to language modeling. Transferring its generative capabilities to modern end-to-end autonomous driving systems has also emerged as a promising direction. However, existing diffusion-based trajectory generative models often exhibit mode collapse where different random noises converge to similar trajectories after the denoising process.Therefore, state-of-the-art models often rely on anchored trajectories from pre-defined trajectory vocabulary or scene priors in the training set to mitigate collapse and enrich the diversity of generated trajectories, but such inductive bias are not available in real-world deployment, which can be challenged when generalizing to unseen scenarios. In this work, we investigate the possibility of effectively tackling the mode collapse challenge without the assumption of pre-defined trajectory vocabulary or pre-computed scene priors. Specifically, we propose TransDiffuser, an encoder-decoder based generative trajectory planning model, where the encoded scene information and motion states serve as the multi-modal conditional input of the denoising decoder. Different from existing approaches, we exploit a simple yet effective multi-modal representation decorrelation optimization mechanism during the denoising process to enrich the latent representation space which better guides the downstream generation. Without any predefined trajectory anchors or pre-computed scene priors, TransDiffuser achieves the PDMS of 94.85 on the closed-loop planning-oriented benchmark NAVSIM, surpassing previous state-of-the-art methods. Qualitative evaluation further showcases TransDiffuser generates more diverse and plausible trajectories which explore more drivable area.
>
---
#### [replaced 049] Semantic-ICP: Iterative Closest Point for Non-rigid Multi-Organ Point Cloud Registration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.00972v2](http://arxiv.org/pdf/2503.00972v2)**

> **作者:** Wanwen Chen; Carson Studders; Jamie J. Y. Kwon; Emily H. T. Pang; Eitan Prisman; Septimiu E. Salcudean
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Point cloud registration is important in computer-aided interventions (CAI). While learning-based point cloud registration methods have been developed, their clinical application is hampered by issues of generalizability and explainability. Therefore, classical point cloud registration methods, such as Iterative Closest Point (ICP), are still widely applied in CAI. ICP methods fail to consider that: (1) the points have well-defined semantic meaning, in that each point can be related to a specific anatomical label; (2) the deformation required for registration needs to follow biomechanical energy constraints. In this paper, we present a novel semantic ICP (SemICP) method that handles multiple point labels and uses linear elastic energy regularization. We use semantic labels to improve the robustness of the closest point matching and propose a novel point cloud deformation representation to apply explicit biomechanical energy regularization. Our experiments on a trans-oral robotic surgery ultrasound-computed tomography registration dataset and two public Learn2reg challenge datasets show that our method improves the Hausdorff distance and mean surface distance compared with other point-matching-based registration methods.
>
---
#### [replaced 050] SA-3DGS: A Self-Adaptive Compression Method for 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.03017v2](http://arxiv.org/pdf/2508.03017v2)**

> **作者:** Liheng Zhang; Weihao Yu; Zubo Lu; Haozhi Gu; Jin Huang
>
> **备注:** This paper is being withdrawn as the work is incomplete and requires substantial additional development before it can be presented
>
> **摘要:** Recent advancements in 3D Gaussian Splatting have enhanced efficient and high-quality novel view synthesis. However, representing scenes requires a large number of Gaussian points, leading to high storage demands and limiting practical deployment. The latest methods facilitate the compression of Gaussian models but struggle to identify truly insignificant Gaussian points in the scene, leading to a decline in subsequent Gaussian pruning, compression quality, and rendering performance. To address this issue, we propose SA-3DGS, a method that significantly reduces storage costs while maintaining rendering quality. SA-3DGS learns an importance score to automatically identify the least significant Gaussians in scene reconstruction, thereby enabling effective pruning and redundancy reduction. Next, the importance-aware clustering module compresses Gaussians attributes more accurately into the codebook, improving the codebook's expressive capability while reducing model size. Finally, the codebook repair module leverages contextual scene information to repair the codebook, thereby recovering the original Gaussian point attributes and mitigating the degradation in rendering quality caused by information loss. Experimental results on several benchmark datasets show that our method achieves up to 66x compression while maintaining or even improving rendering quality. The proposed Gaussian pruning approach is not only adaptable to but also improves other pruning-based methods (e.g., LightGaussian), showcasing excellent performance and strong generalization ability.
>
---
#### [replaced 051] Visual Contextual Attack: Jailbreaking MLLMs with Image-Driven Context Injection
- **分类: cs.CV; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2507.02844v2](http://arxiv.org/pdf/2507.02844v2)**

> **作者:** Ziqi Miao; Yi Ding; Lijun Li; Jing Shao
>
> **备注:** Accepted to EMNLP 2025 (Main). 17 pages, 7 figures
>
> **摘要:** With the emergence of strong vision language capabilities, multimodal large language models (MLLMs) have demonstrated tremendous potential for real-world applications. However, the security vulnerabilities exhibited by the visual modality pose significant challenges to deploying such models in open-world environments. Recent studies have successfully induced harmful responses from target MLLMs by encoding harmful textual semantics directly into visual inputs. However, in these approaches, the visual modality primarily serves as a trigger for unsafe behavior, often exhibiting semantic ambiguity and lacking grounding in realistic scenarios. In this work, we define a novel setting: vision-centric jailbreak, where visual information serves as a necessary component in constructing a complete and realistic jailbreak context. Building on this setting, we propose the VisCo (Visual Contextual) Attack. VisCo fabricates contextual dialogue using four distinct vision-focused strategies, dynamically generating auxiliary images when necessary to construct a vision-centric jailbreak scenario. To maximize attack effectiveness, it incorporates automatic toxicity obfuscation and semantic refinement to produce a final attack prompt that reliably triggers harmful responses from the target black-box MLLMs. Specifically, VisCo achieves a toxicity score of 4.78 and an Attack Success Rate (ASR) of 85% on MM-SafetyBench against GPT-4o, significantly outperforming the baseline, which achieves a toxicity score of 2.48 and an ASR of 22.2%. Code: https://github.com/Dtc7w3PQ/Visco-Attack.
>
---
#### [replaced 052] 3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.07825v4](http://arxiv.org/pdf/2412.07825v4)**

> **作者:** Wufei Ma; Haoyu Chen; Guofeng Zhang; Yu-Cheng Chou; Jieneng Chen; Celso M de Melo; Alan Yuille
>
> **备注:** ICCV 2025. Project page: https://3dsrbench.github.io
>
> **摘要:** 3D spatial reasoning is the ability to analyze and interpret the positions, orientations, and spatial relationships of objects within the 3D space. This allows models to develop a comprehensive understanding of the 3D scene, enabling their applicability to a broader range of areas, such as autonomous navigation, robotics, and AR/VR. While large multi-modal models (LMMs) have achieved remarkable progress in a wide range of image and video understanding tasks, their capabilities to perform 3D spatial reasoning on diverse natural images are less studied. In this work we present the first comprehensive 3D spatial reasoning benchmark, 3DSRBench, with 2,772 manually annotated visual question-answer pairs across 12 question types. We conduct robust and thorough evaluation of 3D spatial reasoning abilities by balancing data distribution and adopting a novel FlipEval strategy. To further study the robustness of 3D spatial reasoning w.r.t. camera 3D viewpoints, our 3DSRBench includes two subsets with 3D spatial reasoning questions on paired images with common and uncommon viewpoints. We benchmark a wide range of open-sourced and proprietary LMMs, uncovering their limitations in various aspects of 3D awareness, such as height, orientation, location, and multi-object reasoning, as well as their degraded performance on images from uncommon 6D viewpoints. Our 3DSRBench provide valuable findings and insights about future development of LMMs with strong spatial reasoning abilities. Our project page is available at https://3dsrbench.github.io/.
>
---
#### [replaced 053] DUAL-VAD: Dual Benchmarks and Anomaly-Focused Sampling for Video Anomaly Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.11605v2](http://arxiv.org/pdf/2509.11605v2)**

> **作者:** Seoik Jung; Taekyung Song; Joshua Jordan Daniel; JinYoung Lee; SungJun Lee
>
> **备注:** 6 pages in IEEE double-column format, 1 figure, 5 tables. The paper introduces a unified framework for Video Anomaly Detection (VAD) featuring dual benchmarks and an anomaly-focused sampling strategy
>
> **摘要:** Video Anomaly Detection (VAD) is critical for surveillance and public safety. However, existing benchmarks are limited to either frame-level or video-level tasks, restricting a holistic view of model generalization. This work first introduces a softmax-based frame allocation strategy that prioritizes anomaly-dense segments while maintaining full-video coverage, enabling balanced sampling across temporal scales. Building on this process, we construct two complementary benchmarks. The image-based benchmark evaluates frame-level reasoning with representative frames, while the video-based benchmark extends to temporally localized segments and incorporates an abnormality scoring task. Experiments on UCF-Crime demonstrate improvements at both the frame and video levels, and ablation studies confirm clear advantages of anomaly-focused sampling over uniform and random baselines.
>
---
#### [replaced 054] CoRe-GS: Coarse-to-Refined Gaussian Splatting with Semantic Object Focus
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.04859v2](http://arxiv.org/pdf/2509.04859v2)**

> **作者:** Hannah Schieber; Dominik Frischmann; Victor Schaack; Simon Boche; Angela Schoellig; Stefan Leutenegger; Daniel Roth
>
> **摘要:** Mobile reconstruction has the potential to support time-critical tasks such as tele-guidance and disaster response, where operators must quickly gain an accurate understanding of the environment. Full high-fidelity scene reconstruction is computationally expensive and often unnecessary when only specific points of interest (POIs) matter for timely decision making. We address this challenge with CoRe-GS, a semantic POI-focused extension of Gaussian Splatting (GS). Instead of optimizing every scene element uniformly, CoRe-GS first produces a fast segmentation-ready GS representation and then selectively refines splats belonging to semantically relevant POIs detected during data acquisition. This targeted refinement reduces training time to 25\% compared to full semantic GS while improving novel view synthesis quality in the areas that matter most. We validate CoRe-GS on both real-world (SCRREAM) and synthetic (NeRDS 360) datasets, demonstrating that prioritizing POIs enables faster and higher-quality mobile reconstruction tailored to operational needs.
>
---
#### [replaced 055] Leveraging Geometric Priors for Unaligned Scene Change Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.11292v2](http://arxiv.org/pdf/2509.11292v2)**

> **作者:** Ziling Liu; Ziwei Chen; Mingqi Gao; Jinyu Yang; Feng Zheng
>
> **摘要:** Unaligned Scene Change Detection aims to detect scene changes between image pairs captured at different times without assuming viewpoint alignment. To handle viewpoint variations, current methods rely solely on 2D visual cues to establish cross-image correspondence to assist change detection. However, large viewpoint changes can alter visual observations, causing appearance-based matching to drift or fail. Additionally, supervision limited to 2D change masks from small-scale SCD datasets restricts the learning of generalizable multi-view knowledge, making it difficult to reliably identify visual overlaps and handle occlusions. This lack of explicit geometric reasoning represents a critical yet overlooked limitation. In this work, we introduce geometric priors for the first time to address the core challenges of unaligned SCD, for reliable identification of visual overlaps, robust correspondence establishment, and explicit occlusion detection. Building on these priors, we propose a training-free framework that integrates them with the powerful representations of a visual foundation model to enable reliable change detection under viewpoint misalignment. Through extensive evaluation on the PSCD, ChangeSim, and PASLCD datasets, we demonstrate that our approach achieves superior and robust performance. Our code will be released at https://github.com/ZilingLiu/GeoSCD.
>
---
#### [replaced 056] Adversarial Prompt Distillation for Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.15244v3](http://arxiv.org/pdf/2411.15244v3)**

> **作者:** Lin Luo; Xin Wang; Bojia Zi; Shihao Zhao; Xingjun Ma; Yu-Gang Jiang
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Large pre-trained Vision-Language Models (VLMs) such as Contrastive Language-Image Pre-training (CLIP) have been shown to be susceptible to adversarial attacks, raising concerns about their deployment in safety-critical applications like autonomous driving and medical diagnosis. One promising approach for robustifying pre-trained VLMs is Adversarial Prompt Tuning (APT), which applies adversarial training during the process of prompt tuning. However, existing APT methods are mostly single-modal methods that design prompt(s) for only the visual or textual modality, limiting their effectiveness in either robustness or clean accuracy. In this work, we propose Adversarial Prompt Distillation (APD), a bimodal knowledge distillation framework that enhances APT by integrating it with multi-modal knowledge transfer. APD optimizes prompts for both visual and textual modalities while distilling knowledge from a clean pre-trained teacher CLIP model. Extensive experiments on multiple benchmark datasets demonstrate the superiority of our APD method over the current state-of-the-art APT methods in terms of both adversarial robustness and clean accuracy. The effectiveness of APD also validates the possibility of using a non-robust teacher to improve the generalization and robustness of fine-tuned VLMs.
>
---
#### [replaced 057] Palmprint De-Identification Using Diffusion Model for High-Quality and Diverse Synthesis
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2504.08272v2](http://arxiv.org/pdf/2504.08272v2)**

> **作者:** Licheng Yan; Bob Zhang; Andrew Beng Jin Teoh; Lu Leng; Shuyi Li; Yuqi Wang; Ziyuan Yang
>
> **摘要:** Palmprint recognition techniques have advanced significantly in recent years, enabling reliable recognition even when palmprints are captured in uncontrolled or challenging environments. However, this strength also introduces new risks, as publicly available palmprint images can be misused by adversaries for malicious activities. Despite this growing concern, research on methods to obscure or anonymize palmprints remains largely unexplored. Thus, it is essential to develop a palmprint de-identification technique capable of removing identity-revealing features while retaining the image's utility and preserving non-sensitive information. In this paper, we propose a training-free framework that utilizes pre-trained diffusion models to generate diverse, high-quality palmprint images that conceal identity features for de-identification purposes. To ensure greater stability and controllability in the synthesis process, we incorporate a semantic-guided embedding fusion alongside a prior interpolation mechanism. We further propose the de-identification ratio, a novel metric for intuitive de-identification assessment. Extensive experiments across multiple palmprint datasets and recognition methods demonstrate that our method effectively conceals identity-related traits with significant diversity across de-identified samples. The de-identified samples preserve high visual fidelity and maintain excellent usability, achieving a balance between de-identification and retaining non-identity information.
>
---
#### [replaced 058] IAG: Input-aware Backdoor Attack on VLMs for Visual Grounding
- **分类: cs.CV; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2508.09456v2](http://arxiv.org/pdf/2508.09456v2)**

> **作者:** Junxian Li; Beining Xu; Di Zhang
>
> **备注:** 13 pages, 13 Figures
>
> **摘要:** Vision-language models (VLMs) have shown significant advancements in tasks such as visual grounding, where they localize specific objects in images based on natural language queries and images. However, security issues in visual grounding tasks for VLMs remain underexplored, especially in the context of backdoor attacks. In this paper, we introduce a novel input-aware backdoor attack method, IAG, designed to manipulate the grounding behavior of VLMs. This attack forces the model to ground a specific target object in the input image, regardless of the user's query. We propose an adaptive trigger generator that embeds the semantic information of the attack target's description into the original image using a text-conditional U-Net, thereby overcoming the open-vocabulary attack challenge. To ensure the attack's stealthiness, we utilize a reconstruction loss to minimize visual discrepancies between poisoned and clean images. Additionally, we introduce a unified method for generating attack data. IAG is evaluated theoretically and empirically, demonstrating its feasibility and effectiveness. Notably, our ASR@0.5 on InternVL-2.5-8B reaches over 65\% on various testing sets. IAG also shows promising potential on manipulating Ferret-7B and LlaVA-1.5-7B with very little accuracy decrease on clean samples. Extensive specific experiments, such as ablation study and potential defense, also indicate the robustness and transferability of our attack.
>
---
#### [replaced 059] MedEBench: Diagnosing Reliability in Text-Guided Medical Image Editing
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.01921v5](http://arxiv.org/pdf/2506.01921v5)**

> **作者:** Minghao Liu; Zhitao He; Zhiyuan Fan; Qingyun Wang; Yi R.; Fung
>
> **摘要:** Text-guided image editing has seen significant progress in natural image domains, but its application in medical imaging remains limited and lacks standardized evaluation frameworks. Such editing could revolutionize clinical practices by enabling personalized surgical planning, enhancing medical education, and improving patient communication. To bridge this gap, we introduce MedEBench1, a robust benchmark designed to diagnose reliability in text-guided medical image editing. MedEBench consists of 1,182 clinically curated image-prompt pairs covering 70 distinct editing tasks and 13 anatomical regions. It contributes in three key areas: (1) a clinically grounded evaluation framework that measures Editing Accuracy, Context Preservation, and Visual Quality, complemented by detailed descriptions of intended edits and corresponding Region-of-Interest (ROI) masks; (2) a comprehensive comparison of seven state-of-theart models, revealing consistent patterns of failure; and (3) a diagnostic error analysis technique that leverages attention alignment, using Intersection-over-Union (IoU) between model attention maps and ROI masks to identify mislocalization issues, where models erroneously focus on incorrect anatomical regions. MedEBench sets the stage for developing more reliable and clinically effective text-guided medical image editing tools.
>
---
#### [replaced 060] RailSafeNet: Visual Scene Understanding for Tram Safety
- **分类: cs.CV; 68T45 (Primary), 68T07; I.4.8**

- **链接: [http://arxiv.org/pdf/2509.12125v2](http://arxiv.org/pdf/2509.12125v2)**

> **作者:** Ondřej Valach; Ivan Gruber
>
> **备注:** 11 pages, 5 figures, EPIA2025
>
> **摘要:** Tram-human interaction safety is an important challenge, given that trams frequently operate in densely populated areas, where collisions can range from minor injuries to fatal outcomes. This paper addresses the issue from the perspective of designing a solution leveraging digital image processing, deep learning, and artificial intelligence to improve the safety of pedestrians, drivers, cyclists, pets, and tram passengers. We present RailSafeNet, a real-time framework that fuses semantic segmentation, object detection and a rule-based Distance Assessor to highlight track intrusions. Using only monocular video, the system identifies rails, localises nearby objects and classifies their risk by comparing projected distances with the standard 1435mm rail gauge. Experiments on the diverse RailSem19 dataset show that a class-filtered SegFormer B3 model achieves 65% intersection-over-union (IoU), while a fine-tuned YOLOv8 attains 75.6% mean average precision (mAP) calculated at an intersection over union (IoU) threshold of 0.50. RailSafeNet therefore delivers accurate, annotation-light scene understanding that can warn drivers before dangerous situations escalate. Code available at https://github.com/oValach/RailSafeNet.
>
---
#### [replaced 061] AMF-MedIT: An Efficient Align-Modulation-Fusion Framework for Medical Image-Tabular Data
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.19439v2](http://arxiv.org/pdf/2506.19439v2)**

> **作者:** Congjing Yu; Jing Ye; Yang Liu; Xiaodong Zhang; Zhiyong Zhang
>
> **摘要:** Multimodal medical analysis combining image and tabular data has gained increasing attention. However, effective fusion remains challenging due to cross-modal discrepancies in feature dimensions and modality contributions, as well as the noise from high-dimensional tabular inputs. To address these problems, we present AMF-MedIT, an efficient Align-Modulation-Fusion framework for medical image and tabular data integration, particularly under data-scarce conditions. Built upon a self-supervised learning strategy, we introduce the Adaptive Modulation and Fusion (AMF) module, a novel, streamlined fusion paradigm that harmonizes dimension discrepancies and dynamically balances modality contributions. It integrates prior knowledge to guide the allocation of modality contributions in the fusion and employs feature masks together with magnitude and leakage losses to adjust the dimensionality and magnitude of unimodal features. Additionally, we develop FT-Mamba, a powerful tabular encoder leveraging a selective mechanism to handle noisy medical tabular data efficiently. Extensive experiments, including simulations of clinical noise, demonstrate that AMF-MedIT achieves superior accuracy, robustness, and data efficiency across multimodal classification tasks. Interpretability analyses further reveal how FT-Mamba shapes multimodal pretraining and enhances the image encoder's attention, highlighting the practical value of our framework for reliable and efficient clinical artificial intelligence applications.
>
---
