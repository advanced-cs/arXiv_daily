# 计算机视觉 cs.CV

- **最新发布 139 篇**

- **更新 80 篇**

## 最新发布

#### [new 001] SWAN - Enabling Fast and Mobile Histopathology Image Annotation through Swipeable Interfaces
- **分类: cs.CV; cs.SE**

- **简介: 论文提出SWAN，一种基于滑动手势的移动友好型病理图像标注工具，解决传统文件夹排序标注效率低、易疲劳的问题。支持跨平台实时标注，实验证明其标注速度更快、质量与传统方法相当，且用户满意度高。**

- **链接: []()**

> **作者:** Sweta Banerjee; Timo Gosch; Sara Hester; Viktoria Weiss; Thomas Conrad; Taryn A. Donovan; Nils Porsche; Jonas Ammeling; Christoph Stroblberger; Robert Klopfleisch; Christopher Kaltenecker; Christof A. Bertram; Katharina Breininger; Marc Aubreville
>
> **摘要:** The annotation of large scale histopathology image datasets remains a major bottleneck in developing robust deep learning models for clinically relevant tasks, such as mitotic figure classification. Folder-based annotation workflows are usually slow, fatiguing, and difficult to scale. To address these challenges, we introduce SWipeable ANnotations (SWAN), an open-source, MIT-licensed web application that enables intuitive image patch classification using a swiping gesture. SWAN supports both desktop and mobile platforms, offers real-time metadata capture, and allows flexible mapping of swipe gestures to class labels. In a pilot study with four pathologists annotating 600 mitotic figure image patches, we compared SWAN against a traditional folder-sorting workflow. SWAN enabled rapid annotations with pairwise percent agreement ranging from 86.52% to 93.68% (Cohen's Kappa = 0.61-0.80), while for the folder-based method, the pairwise percent agreement ranged from 86.98% to 91.32% (Cohen's Kappa = 0.63-0.75) for the task of classifying atypical versus normal mitotic figures, demonstrating high consistency between annotators and comparable performance. Participants rated the tool as highly usable and appreciated the ability to annotate on mobile devices. These results suggest that SWAN can accelerate image annotation while maintaining annotation quality, offering a scalable and user-friendly alternative to conventional workflows.
>
---
#### [new 002] FlowFeat: Pixel-Dense Embedding of Motion Profiles
- **分类: cs.CV**

- **简介: FlowFeat提出一种高分辨率运动轮廓嵌入方法，通过自监督蒸馏光学流信息，生成稠密、时序一致的特征表示，提升视频分割、单目深度估计和语义分割等稠密预测任务的性能。**

- **链接: []()**

> **作者:** Nikita Araslanov; Anna Sonnweber; Daniel Cremers
>
> **备注:** Project website: https://tum-vision.github.io/flowfeat
>
> **摘要:** Dense and versatile image representations underpin the success of virtually all computer vision applications. However, state-of-the-art networks, such as transformers, produce low-resolution feature grids, which are suboptimal for dense prediction tasks. To address this limitation, we present FlowFeat, a high-resolution and multi-task feature representation. The key ingredient behind FlowFeat is a novel distillation technique that embeds a distribution of plausible apparent motions, or motion profiles. By leveraging optical flow networks and diverse video data, we develop an effective self-supervised training framework that statistically approximates the apparent motion. With its remarkable level of spatial detail, FlowFeat encodes a compelling degree of geometric and semantic cues while exhibiting high temporal consistency. Empirically, FlowFeat significantly enhances the representational power of five state-of-the-art encoders and alternative upsampling strategies across three dense tasks: video object segmentation, monocular depth estimation and semantic segmentation. Training FlowFeat is computationally inexpensive and robust to inaccurate flow estimation, remaining highly effective even when using unsupervised flow networks. Our work takes a step forward towards reliable and versatile dense image representations.
>
---
#### [new 003] Non-Aligned Reference Image Quality Assessment for Novel View Synthesis
- **分类: cs.CV**

- **简介: 该论文提出一种非对齐参考图像质量评估（NAR-IQA）方法，解决新视角合成中无像素对齐参考图的评估难题。基于对比学习与LoRA-DINOv2，利用合成畸变数据训练，显著提升泛化性，且与人类主观评价高度相关。**

- **链接: []()**

> **作者:** Abhijay Ghildyal; Rajesh Sureddi; Nabajeet Barman; Saman Zadtootaghaj; Alan Bovik
>
> **摘要:** Evaluating the perceptual quality of Novel View Synthesis (NVS) images remains a key challenge, particularly in the absence of pixel-aligned ground truth references. Full-Reference Image Quality Assessment (FR-IQA) methods fail under misalignment, while No-Reference (NR-IQA) methods struggle with generalization. In this work, we introduce a Non-Aligned Reference (NAR-IQA) framework tailored for NVS, where it is assumed that the reference view shares partial scene content but lacks pixel-level alignment. We constructed a large-scale image dataset containing synthetic distortions targeting Temporal Regions of Interest (TROI) to train our NAR-IQA model. Our model is built on a contrastive learning framework that incorporates LoRA-enhanced DINOv2 embeddings and is guided by supervision from existing IQA methods. We train exclusively on synthetically generated distortions, deliberately avoiding overfitting to specific real NVS samples and thereby enhancing the model's generalization capability. Our model outperforms state-of-the-art FR-IQA, NR-IQA, and NAR-IQA methods, achieving robust performance on both aligned and non-aligned references. We also conducted a novel user study to gather data on human preferences when viewing non-aligned references in NVS. We find strong correlation between our proposed quality prediction model and the collected subjective ratings. For dataset and code, please visit our project page: https://stootaghaj.github.io/nova-project/
>
---
#### [new 004] HD$^2$-SSC: High-Dimension High-Density Semantic Scene Completion for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文提出HD²-SSC框架，解决相机输入与三维稠密语义完成间的维度与密度差距问题，通过高维语义解耦和稠密占据优化，提升自动驾驶场景的三维语义完成精度。**

- **链接: []()**

> **作者:** Zhiwen Yang; Yuxin Peng
>
> **备注:** 10 pages, 6 figures, accepted by AAAI 2026
>
> **摘要:** Camera-based 3D semantic scene completion (SSC) plays a crucial role in autonomous driving, enabling voxelized 3D scene understanding for effective scene perception and decision-making. Existing SSC methods have shown efficacy in improving 3D scene representations, but suffer from the inherent input-output dimension gap and annotation-reality density gap, where the 2D planner view from input images with sparse annotated labels leads to inferior prediction of real-world dense occupancy with a 3D stereoscopic view. In light of this, we propose the corresponding High-Dimension High-Density Semantic Scene Completion (HD$^2$-SSC) framework with expanded pixel semantics and refined voxel occupancies. To bridge the dimension gap, a High-dimension Semantic Decoupling module is designed to expand 2D image features along a pseudo third dimension, decoupling coarse pixel semantics from occlusions, and then identify focal regions with fine semantics to enrich image features. To mitigate the density gap, a High-density Occupancy Refinement module is devised with a "detect-and-refine" architecture to leverage contextual geometric and semantic structures for enhanced semantic density with the completion of missing voxels and correction of erroneous ones. Extensive experiments and analyses on the SemanticKITTI and SSCBench-KITTI-360 datasets validate the effectiveness of our HD$^2$-SSC framework.
>
---
#### [new 005] CleverBirds: A Multiple-Choice Benchmark for Fine-grained Human Knowledge Tracing
- **分类: cs.CV; cs.LG**

- **简介: 论文提出CleverBirds基准，用于追踪人类在细粒度鸟类识别中的知识演化。基于1700万答题数据，解决专家级视觉学习过程建模难题，支持知识追踪模型开发与评估。**

- **链接: []()**

> **作者:** Leonie Bossemeyer; Samuel Heinrich; Grant Van Horn; Oisin Mac Aodha
>
> **备注:** To appear at NeurIPS 2025 - Datasets and Benchmarks Track
>
> **摘要:** Mastering fine-grained visual recognition, essential in many expert domains, can require that specialists undergo years of dedicated training. Modeling the progression of such expertize in humans remains challenging, and accurately inferring a human learner's knowledge state is a key step toward understanding visual learning. We introduce CleverBirds, a large-scale knowledge tracing benchmark for fine-grained bird species recognition. Collected by the citizen-science platform eBird, it offers insight into how individuals acquire expertize in complex fine-grained classification. More than 40,000 participants have engaged in the quiz, answering over 17 million multiple-choice questions spanning over 10,000 bird species, with long-range learning patterns across an average of 400 questions per participant. We release this dataset to support the development and evaluation of new methods for visual knowledge tracing. We show that tracking learners' knowledge is challenging, especially across participant subgroups and question types, with different forms of contextual information offering varying degrees of predictive benefit. CleverBirds is among the largest benchmark of its kind, offering a substantially higher number of learnable concepts. With it, we hope to enable new avenues for studying the development of visual expertize over time and across individuals.
>
---
#### [new 006] Knowledge-Guided Textual Reasoning for Explainable Video Anomaly Detection via LLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出TbVAD，面向弱监督视频异常检测任务，通过大语言模型将视频转为文本描述并构建语义知识结构，实现完全基于文本的可解释异常检测与原因分析，提升现实监控场景的可解释性与可靠性。**

- **链接: []()**

> **作者:** Hari Lee
>
> **摘要:** We introduce Text-based Explainable Video Anomaly Detection (TbVAD), a language-driven framework for weakly supervised video anomaly detection that performs anomaly detection and explanation entirely within the textual domain. Unlike conventional WSVAD models that rely on explicit visual features, TbVAD represents video semantics through language, enabling interpretable and knowledge-grounded reasoning. The framework operates in three stages: (1) transforming video content into fine-grained captions using a vision-language model, (2) constructing structured knowledge by organizing the captions into four semantic slots (action, object, context, environment), and (3) generating slot-wise explanations that reveal which semantic factors contribute most to the anomaly decision. We evaluate TbVAD on two public benchmarks, UCF-Crime and XD-Violence, demonstrating that textual knowledge reasoning provides interpretable and reliable anomaly detection for real-world surveillance scenarios.
>
---
#### [new 007] VLMDiff: Leveraging Vision-Language Models for Multi-Class Anomaly Detection with Diffusion
- **分类: cs.CV**

- **简介: 论文提出VLMDiff，用于无监督多类视觉异常检测。通过融合视觉-语言模型生成正常图像描述，作为扩散模型的条件输入，无需人工标注或类专属训练，提升异常定位精度，显著优于现有扩散方法。**

- **链接: []()**

> **作者:** Samet Hicsonmez; Abd El Rahman Shabayek; Djamila Aouada
>
> **备注:** WACV 2026
>
> **摘要:** Detecting visual anomalies in diverse, multi-class real-world images is a significant challenge. We introduce \ours, a novel unsupervised multi-class visual anomaly detection framework. It integrates a Latent Diffusion Model (LDM) with a Vision-Language Model (VLM) for enhanced anomaly localization and detection. Specifically, a pre-trained VLM with a simple prompt extracts detailed image descriptions, serving as additional conditioning for LDM training. Current diffusion-based methods rely on synthetic noise generation, limiting their generalization and requiring per-class model training, which hinders scalability. \ours, however, leverages VLMs to obtain normal captions without manual annotations or additional training. These descriptions condition the diffusion model, learning a robust normal image feature representation for multi-class anomaly detection. Our method achieves competitive performance, improving the pixel-level Per-Region-Overlap (PRO) metric by up to 25 points on the Real-IAD dataset and 8 points on the COCO-AD dataset, outperforming state-of-the-art diffusion-based approaches. Code is available at https://github.com/giddyyupp/VLMDiff.
>
---
#### [new 008] UltraGS: Gaussian Splatting for Ultrasound Novel View Synthesis
- **分类: cs.CV; cs.AI**

- **简介: UltraGS提出一种面向超声成像的高斯泼溅框架，解决超声视场受限导致的新视角合成难题，通过深度感知高斯建模与物理感知渲染，实现高精度、实时的超声新视图生成。**

- **链接: []()**

> **作者:** Yuezhe Yang; Wenjie Cai; Dexin Yang; Yufang Dong; Xingbo Dong; Zhe Jin
>
> **备注:** Under Review
>
> **摘要:** Ultrasound imaging is a cornerstone of non-invasive clinical diagnostics, yet its limited field of view complicates novel view synthesis. We propose \textbf{UltraGS}, a Gaussian Splatting framework optimized for ultrasound imaging. First, we introduce a depth-aware Gaussian splatting strategy, where each Gaussian is assigned a learnable field of view, enabling accurate depth prediction and precise structural representation. Second, we design SH-DARS, a lightweight rendering function combining low-order spherical harmonics with ultrasound-specific wave physics, including depth attenuation, reflection, and scattering, to model tissue intensity accurately. Third, we contribute the Clinical Ultrasound Examination Dataset, a benchmark capturing diverse anatomical scans under real-world clinical protocols. Extensive experiments on three datasets demonstrate UltraGS's superiority, achieving state-of-the-art results in PSNR (up to 29.55), SSIM (up to 0.89), and MSE (as low as 0.002) while enabling real-time synthesis at 64.69 fps. The code and dataset are open-sourced at: https://github.com/Bean-Young/UltraGS.
>
---
#### [new 009] Empowering DINO Representations for Underwater Instance Segmentation via Aligner and Prompter
- **分类: cs.CV**

- **简介: 该论文针对水下实例分割任务，提出DiveSeg框架，利用DINO作为特征学习器，通过AquaStyle Aligner校正水下色彩风格，结合ObjectPrior Prompter引入实例先验，显著提升分割性能。**

- **链接: []()**

> **作者:** Zhiyang Chen; Chen Zhang; Hao Fang; Runmin Cong
>
> **备注:** AAAI 2026
>
> **摘要:** Underwater instance segmentation (UIS), integrating pixel-level understanding and instance-level discrimination, is a pivotal technology in marine resource exploration and ecological protection. In recent years, large-scale pretrained visual foundation models, exemplified by DINO, have advanced rapidly and demonstrated remarkable performance on complex downstream tasks. In this paper, we demonstrate that DINO can serve as an effective feature learner for UIS, and we introduce DiveSeg, a novel framework built upon two insightful components: (1) The AquaStyle Aligner, designed to embed underwater color style features into the DINO fine-tuning process, facilitating better adaptation to the underwater domain. (2) The ObjectPrior Prompter, which incorporates binary segmentation-based prompts to deliver object-level priors, provides essential guidance for instance segmentation task that requires both object- and instance-level reasoning. We conduct thorough experiments on the popular UIIS and USIS10K datasets, and the results show that DiveSeg achieves the state-of-the-art performance. Code: https://github.com/ettof/Diveseg.
>
---
#### [new 010] Federated CLIP for Resource-Efficient Heterogeneous Medical Image Classification
- **分类: cs.CV**

- **简介: 该论文提出FedMedCLIP，用于联邦学习下的医疗图像分类，解决数据异构与资源开销问题。通过冻结CLIP编码器、引入掩码特征适配模块与私有MLP分类器，并结合KL蒸馏与模型压缩，实现高效低通信开销的分布式训练。**

- **链接: []()**

> **作者:** Yihang Wu; Ahmad Chaddad
>
> **备注:** Accepted in AAAI 2026 Main track. Code is available at https://github.com/AIPMLab/FedMedCLIP
>
> **摘要:** Despite the remarkable performance of deep models in medical imaging, they still require source data for training, which limits their potential in light of privacy concerns. Federated learning (FL), as a decentralized learning framework that trains a shared model with multiple hospitals (a.k.a., FL clients), provides a feasible solution. However, data heterogeneity and resource costs hinder the deployment of FL models, especially when using vision language models (VLM). To address these challenges, we propose a novel contrastive language-image pre-training (CLIP) based FL approach for medical image classification (FedMedCLIP). Specifically, we introduce a masked feature adaptation module (FAM) as a communication module to reduce the communication load while freezing the CLIP encoders to reduce the computational overhead. Furthermore, we propose a masked multi-layer perceptron (MLP) as a private local classifier to adapt to the client tasks. Moreover, we design an adaptive Kullback-Leibler (KL) divergence-based distillation regularization method to enable mutual learning between FAM and MLP. Finally, we incorporate model compression to transmit the FAM parameters while using ensemble predictions for classification. Extensive experiments on four publicly available medical datasets demonstrate that our model provides feasible performance (e.g., 8\% higher compared to second best baseline on ISIC2019) with reasonable resource cost (e.g., 120$\times$ faster than FedAVG).
>
---
#### [new 011] RePose-NeRF: Robust Radiance Fields for Mesh Reconstruction under Noisy Camera Poses
- **分类: cs.CV**

- **简介: 该论文提出RePose-NeRF，用于在相机位姿噪声下从多视角图像直接重建高质量可编辑网格。通过联合优化位姿与隐式辐射场，实现鲁棒三维重建，打通神经隐式表示与机器人应用的壁垒。**

- **链接: []()**

> **作者:** Sriram Srinivasan; Gautam Ramachandra
>
> **备注:** Several figures are included to illustrate the reconstruction and rendering quality of the proposed method, which is why the submission exceeds the 50MB file size limit. > Several figures are included to illustrate the reconstruction and rendering quality of the proposed method, which is why the submission exceeds the 50,000 KB file size limit (Now this has been resolved)
>
> **摘要:** Accurate 3D reconstruction from multi-view images is essential for downstream robotic tasks such as navigation, manipulation, and environment understanding. However, obtaining precise camera poses in real-world settings remains challenging, even when calibration parameters are known. This limits the practicality of existing NeRF-based methods that rely heavily on accurate extrinsic estimates. Furthermore, their implicit volumetric representations differ significantly from the widely adopted polygonal meshes, making rendering and manipulation inefficient in standard 3D software. In this work, we propose a robust framework that reconstructs high-quality, editable 3D meshes directly from multi-view images with noisy extrinsic parameters. Our approach jointly refines camera poses while learning an implicit scene representation that captures fine geometric detail and photorealistic appearance. The resulting meshes are compatible with common 3D graphics and robotics tools, enabling efficient downstream use. Experiments on standard benchmarks demonstrate that our method achieves accurate and robust 3D reconstruction under pose uncertainty, bridging the gap between neural implicit representations and practical robotic applications.
>
---
#### [new 012] Toward the Frontiers of Reliable Diffusion Sampling via Adversarial Sinkhorn Attention Guidance
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Adversarial Sinkhorn Attention Guidance（ASAG），用于提升扩散模型采样可靠性。通过最优传输视角重构注意力机制，以对抗性方式降低键-查询相似性，无需重训即可提升生成质量与可控性。**

- **链接: []()**

> **作者:** Kwanyoung Kim
>
> **备注:** Accepted to AAAI 26
>
> **摘要:** Diffusion models have demonstrated strong generative performance when using guidance methods such as classifier-free guidance (CFG), which enhance output quality by modifying the sampling trajectory. These methods typically improve a target output by intentionally degrading another, often the unconditional output, using heuristic perturbation functions such as identity mixing or blurred conditions. However, these approaches lack a principled foundation and rely on manually designed distortions. In this work, we propose Adversarial Sinkhorn Attention Guidance (ASAG), a novel method that reinterprets attention scores in diffusion models through the lens of optimal transport and intentionally disrupt the transport cost via Sinkhorn algorithm. Instead of naively corrupting the attention mechanism, ASAG injects an adversarial cost within self-attention layers to reduce pixel-wise similarity between queries and keys. This deliberate degradation weakens misleading attention alignments and leads to improved conditional and unconditional sample quality. ASAG shows consistent improvements in text-to-image diffusion, and enhances controllability and fidelity in downstream applications such as IP-Adapter and ControlNet. The method is lightweight, plug-and-play, and improves reliability without requiring any model retraining.
>
---
#### [new 013] LatentPrintFormer: A Hybrid CNN-Transformer with Spatial Attention for Latent Fingerprint identification
- **分类: cs.CV**

- **简介: 论文提出LatentPrintFormer，用于低质量潜指纹识别。结合CNN与Transformer提取局部与全局特征，引入空间注意力抑制噪声，融合后生成512维嵌入，通过余弦相似度实现闭集识别，显著提升Rank-10识别率。**

- **链接: []()**

> **作者:** Arnab Maity; Manasa; Pavan Kumar C; Raghavendra Ramachandra
>
> **备注:** Accepted in CVIP 2025
>
> **摘要:** Latent fingerprint identification remains a challenging task due to low image quality, background noise, and partial impressions. In this work, we propose a novel identification approach called LatentPrintFormer. The proposed model integrates a CNN backbone (EfficientNet-B0) and a Transformer backbone (Swin Tiny) to extract both local and global features from latent fingerprints. A spatial attention module is employed to emphasize high-quality ridge regions while suppressing background noise. The extracted features are fused and projected into a unified 512-dimensional embedding, and matching is performed using cosine similarity in a closed-set identification setting. Extensive experiments on two publicly available datasets demonstrate that LatentPrintFormer consistently outperforms three state-of-the-art latent fingerprint recognition techniques, achieving higher identification rates across Rank-10.
>
---
#### [new 014] Anatomy-VLM: A Fine-grained Vision-Language Model for Medical Interpretation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出Anatomy-VLM，面向医学影像理解任务，解决传统VLM忽略细粒度解剖细节的问题。通过定位关键解剖结构、融合临床知识、多尺度对齐，实现精准疾病预测与零样本解剖解释。**

- **链接: []()**

> **作者:** Difei Gu; Yunhe Gao; Mu Zhou; Dimitris Metaxas
>
> **备注:** Accepted to Winter Conference on Applications of Computer Vision (WACV) 2026
>
> **摘要:** Accurate disease interpretation from radiology remains challenging due to imaging heterogeneity. Achieving expert-level diagnostic decisions requires integration of subtle image features with clinical knowledge. Yet major vision-language models (VLMs) treat images as holistic entities and overlook fine-grained image details that are vital for disease diagnosis. Clinicians analyze images by utilizing their prior medical knowledge and identify anatomical structures as important region of interests (ROIs). Inspired from this human-centric workflow, we introduce Anatomy-VLM, a fine-grained, vision-language model that incorporates multi-scale information. First, we design a model encoder to localize key anatomical features from entire medical images. Second, these regions are enriched with structured knowledge for contextually-aware interpretation. Finally, the model encoder aligns multi-scale medical information to generate clinically-interpretable disease prediction. Anatomy-VLM achieves outstanding performance on both in- and out-of-distribution datasets. We also validate the performance of Anatomy-VLM on downstream image segmentation tasks, suggesting that its fine-grained alignment captures anatomical and pathology-related knowledge. Furthermore, the Anatomy-VLM's encoder facilitates zero-shot anatomy-wise interpretation, providing its strong expert-level clinical interpretation capabilities.
>
---
#### [new 015] UCDSC: Open Set UnCertainty aware Deep Simplex Classifier for Medical Image Datasets
- **分类: cs.CV**

- **简介: 该论文提出UCDSC方法，面向医学图像的开放集识别任务，解决小样本与罕见病导致的未知类别误判问题，通过基于单形聚类的损失函数有效拒绝未知样本，在多个医学数据集上超越现有方法。**

- **链接: []()**

> **作者:** Arnav Aditya; Nitin Kumar; Saurabh Shigwan
>
> **备注:** 10 pages, Accepted at IEEE/CVF WACV 2026, Source code is available at this URL https://github.com/Arnavadi19/UCDSC
>
> **摘要:** Driven by advancements in deep learning, computer-aided diagnoses have made remarkable progress. However, outside controlled laboratory settings, algorithms may encounter several challenges. In the medical domain, these difficulties often stem from limited data availability due to ethical and legal restrictions, as well as the high cost and time required for expert annotations-especially in the face of emerging or rare diseases. In this context, open-set recognition plays a vital role by identifying whether a sample belongs to one of the known classes seen during training or should be rejected as an unknown. Recent studies have shown that features learned in the later stages of deep neural networks are observed to cluster around their class means, which themselves are arranged as individual vertices of a regular simplex [32]. The proposed method introduces a loss function designed to reject samples of unknown classes effectively by penalizing open space regions using auxiliary datasets. This approach achieves significant performance gain across four MedMNIST datasets-BloodMNIST, OCTMNIST, DermaMNIST, TissueMNIST and a publicly available skin dataset [29] outperforming state-of-the-art techniques.
>
---
#### [new 016] Invisible Triggers, Visible Threats! Road-Style Adversarial Creation Attack for Visual 3D Detection in Autonomous Driving
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对自动驾驶视觉3D检测的对抗攻击问题，提出AdvRoad方法，生成伪装成路面纹理的隐蔽对抗图案，诱导检测器产生幻觉目标，实现无需人工察觉的物理攻击，提升攻击隐蔽性与泛化性。**

- **链接: []()**

> **作者:** Jian Wang; Lijun He; Yixing Yong; Haixia Bi; Fan Li
>
> **备注:** Accepted by the AAAI 2026 (Main Track)
>
> **摘要:** Modern autonomous driving (AD) systems leverage 3D object detection to perceive foreground objects in 3D environments for subsequent prediction and planning. Visual 3D detection based on RGB cameras provides a cost-effective solution compared to the LiDAR paradigm. While achieving promising detection accuracy, current deep neural network-based models remain highly susceptible to adversarial examples. The underlying safety concerns motivate us to investigate realistic adversarial attacks in AD scenarios. Previous work has demonstrated the feasibility of placing adversarial posters on the road surface to induce hallucinations in the detector. However, the unnatural appearance of the posters makes them easily noticeable by humans, and their fixed content can be readily targeted and defended. To address these limitations, we propose the AdvRoad to generate diverse road-style adversarial posters. The adversaries have naturalistic appearances resembling the road surface while compromising the detector to perceive non-existent objects at the attack locations. We employ a two-stage approach, termed Road-Style Adversary Generation and Scenario-Associated Adaptation, to maximize the attack effectiveness on the input scene while ensuring the natural appearance of the poster, allowing the attack to be carried out stealthily without drawing human attention. Extensive experiments show that AdvRoad generalizes well to different detectors, scenes, and spoofing locations. Moreover, physical attacks further demonstrate the practical threats in real-world environments.
>
---
#### [new 017] Beyond the Pixels: VLM-based Evaluation of Identity Preservation in Reference-Guided Synthesis
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种基于视觉语言模型（VLM）的层次化评估框架，解决生成模型中身份保持的细粒度评价难题，通过结构化推理与特征级分解，提升评估一致性与可解释性，并构建新基准测试多样化主体。**

- **链接: []()**

> **作者:** Aditi Singhania; Krutik Malani; Riddhi Dhawan; Arushi Jain; Garv Tandon; Nippun Sharma; Souymodip Chakraborty; Vineet Batra; Ankit Phogat
>
> **摘要:** Evaluating identity preservation in generative models remains a critical yet unresolved challenge. Existing metrics rely on global embeddings or coarse VLM prompting, failing to capture fine-grained identity changes and providing limited diagnostic insight. We introduce Beyond the Pixels, a hierarchical evaluation framework that decomposes identity assessment into feature-level transformations. Our approach guides VLMs through structured reasoning by (1) hierarchically decomposing subjects into (type, style) -> attribute -> feature decision tree, and (2) prompting for concrete transformations rather than abstract similarity scores. This decomposition grounds VLM analysis in verifiable visual evidence, reducing hallucinations and improving consistency. We validate our framework across four state-of-the-art generative models, demonstrating strong alignment with human judgments in measuring identity consistency. Additionally, we introduce a new benchmark specifically designed to stress-test generative models. It comprises 1,078 image-prompt pairs spanning diverse subject types, including underrepresented categories such as anthropomorphic and animated characters, and captures an average of six to seven transformation axes per prompt.
>
---
#### [new 018] NeuSpring: Neural Spring Fields for Reconstruction and Simulation of Deformable Objects from Videos
- **分类: cs.CV**

- **简介: 论文提出NeuSpring，用于从视频中重建与模拟可变形物体。针对现有方法物理泛化差的问题，创新性地引入分段拓扑优化与神经弹簧场，建模材料异质性与空间关联性，显著提升重建与预测精度。**

- **链接: []()**

> **作者:** Qingshan Xu; Jiao Liu; Shangshu Yu; Yuxuan Wang; Yuan Zhou; Junbao Zhou; Jiequan Cui; Yew-Soon Ong; Hanwang Zhang
>
> **备注:** Accepted by AAAI2026
>
> **摘要:** In this paper, we aim to create physical digital twins of deformable objects under interaction. Existing methods focus more on the physical learning of current state modeling, but generalize worse to future prediction. This is because existing methods ignore the intrinsic physical properties of deformable objects, resulting in the limited physical learning in the current state modeling. To address this, we present NeuSpring, a neural spring field for the reconstruction and simulation of deformable objects from videos. Built upon spring-mass models for realistic physical simulation, our method consists of two major innovations: 1) a piecewise topology solution that efficiently models multi-region spring connection topologies using zero-order optimization, which considers the material heterogeneity of real-world objects. 2) a neural spring field that represents spring physical properties across different frames using a canonical coordinate-based neural network, which effectively leverages the spatial associativity of springs for physical learning. Experiments on real-world datasets demonstrate that our NeuSping achieves superior reconstruction and simulation performance for current state modeling and future prediction, with Chamfer distance improved by 20% and 25%, respectively.
>
---
#### [new 019] MonoCLUE : Object-Aware Clustering Enhances Monocular 3D Object Detection
- **分类: cs.CV**

- **简介: MonoCLUE面向单目3D目标检测任务，解决遮挡与视野受限下的几何模糊与识别不准问题。通过局部视觉特征聚类与跨场景记忆构建，增强物体部分感知与特征一致性，显著提升检测鲁棒性与精度。**

- **链接: []()**

> **作者:** Sunghun Yang; Minhyeok Lee; Jungho Lee; Sangyoun Lee
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Monocular 3D object detection offers a cost-effective solution for autonomous driving but suffers from ill-posed depth and limited field of view. These constraints cause a lack of geometric cues and reduced accuracy in occluded or truncated scenes. While recent approaches incorporate additional depth information to address geometric ambiguity, they overlook the visual cues crucial for robust recognition. We propose MonoCLUE, which enhances monocular 3D detection by leveraging both local clustering and generalized scene memory of visual features. First, we perform K-means clustering on visual features to capture distinct object-level appearance parts (e.g., bonnet, car roof), improving detection of partially visible objects. The clustered features are propagated across regions to capture objects with similar appearances. Second, we construct a generalized scene memory by aggregating clustered features across images, providing consistent representations that generalize across scenes. This improves object-level feature consistency, enabling stable detection across varying environments. Lastly, we integrate both local cluster features and generalized scene memory into object queries, guiding attention toward informative regions. Exploiting a unified local clustering and generalized scene memory strategy, MonoCLUE enables robust monocular 3D detection under occlusion and limited visibility, achieving state-of-the-art performance on the KITTI benchmark.
>
---
#### [new 020] Laytrol: Preserving Pretrained Knowledge in Layout Control for Multimodal Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文面向布局控制的多模态扩散生成任务，解决预训练知识丢失问题。提出Laytrol网络，通过参数继承、零初始化与对象级旋转编码，保留基模型知识，提升生成质量与一致性。**

- **链接: []()**

> **作者:** Sida Huang; Siqi Huang; Ping Luo; Hongyuan Zhang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** With the development of diffusion models, enhancing spatial controllability in text-to-image generation has become a vital challenge. As a representative task for addressing this challenge, layout-to-image generation aims to generate images that are spatially consistent with the given layout condition. Existing layout-to-image methods typically introduce the layout condition by integrating adapter modules into the base generative model. However, the generated images often exhibit low visual quality and stylistic inconsistency with the base model, indicating a loss of pretrained knowledge. To alleviate this issue, we construct the Layout Synthesis (LaySyn) dataset, which leverages images synthesized by the base model itself to mitigate the distribution shift from the pretraining data. Moreover, we propose the Layout Control (Laytrol) Network, in which parameters are inherited from MM-DiT to preserve the pretrained knowledge of the base model. To effectively activate the copied parameters and avoid disturbance from unstable control conditions, we adopt a dedicated initialization scheme for Laytrol. In this scheme, the layout encoder is initialized as a pure text encoder to ensure that its output tokens remain within the data domain of MM-DiT. Meanwhile, the outputs of the layout control network are initialized to zero. In addition, we apply Object-level Rotary Position Embedding to the layout tokens to provide coarse positional information. Qualitative and quantitative experiments demonstrate the effectiveness of our method.
>
---
#### [new 021] ReIDMamba: Learning Discriminative Features with Visual State Space Model for Person Re-Identification
- **分类: cs.CV**

- **简介: 论文提出ReIDMamba，首个纯Mamba架构的人重识别模型，解决Transformer计算开销大问题，通过多粒度特征提取与排序感知正则化，高效学习判别性特征，在五大基准上达到SOTA，参数量仅为TransReID的三分之一。**

- **链接: []()**

> **作者:** Hongyang Gu; Qisong Yang; Lei Pu; Siming Han; Yao Ding
>
> **备注:** 11 pages, 8 figures. Accepted to IEEE Transactions on Multimedia (TMM). Accepted Manuscript version uploaded
>
> **摘要:** Extracting robust discriminative features is a critical challenge in person re-identification (ReID). While Transformer-based methods have successfully addressed some limitations of convolutional neural networks (CNNs), such as their local processing nature and information loss resulting from convolution and downsampling operations, they still face the scalability issue due to the quadratic increase in memory and computational requirements with the length of the input sequence. To overcome this, we propose a pure Mamba-based person ReID framework named ReIDMamba. Specifically, we have designed a Mamba-based strong baseline that effectively leverages fine-grained, discriminative global features by introducing multiple class tokens. To further enhance robust features learning within Mamba, we have carefully designed two novel techniques. First, the multi-granularity feature extractor (MGFE) module, designed with a multi-branch architecture and class token fusion, effectively forms multi-granularity features, enhancing both discrimination ability and fine-grained coverage. Second, the ranking-aware triplet regularization (RATR) is introduced to reduce redundancy in features from multiple branches, enhancing the diversity of multi-granularity features by incorporating both intra-class and inter-class diversity constraints, thus ensuring the robustness of person features. To our knowledge, this is the pioneering work that integrates a purely Mamba-driven approach into ReID research. Our proposed ReIDMamba model boasts only one-third the parameters of TransReID, along with lower GPU memory usage and faster inference throughput. Experimental results demonstrate ReIDMamba's superior and promising performance, achieving state-of-the-art performance on five person ReID benchmarks. Code is available at https://github.com/GuHY777/ReIDMamba.
>
---
#### [new 022] High-Quality Proposal Encoding and Cascade Denoising for Imaginary Supervised Object Detection
- **分类: cs.CV**

- **简介: 该论文针对想象监督目标检测（ISOD）中合成数据质量低、DETR收敛慢与过拟合问题，提出Cascade HQP-DETR，通过高质量数据生成、提议引导查询编码和级联去噪，显著提升模型在真实图像上的检测性能。**

- **链接: []()**

> **作者:** Zhiyuan Chen; Yuelin Guo; Zitong Huang; Haoyu He; Renhao Lu; Weizhe Zhang
>
> **备注:** This work has been submitted to Pattern Recognition for possible publication
>
> **摘要:** Object detection models demand large-scale annotated datasets, which are costly and labor-intensive to create. This motivated Imaginary Supervised Object Detection (ISOD), where models train on synthetic images and test on real images. However, existing methods face three limitations: (1) synthetic datasets suffer from simplistic prompts, poor image quality, and weak supervision; (2) DETR-based detectors, due to their random query initialization, struggle with slow convergence and overfitting to synthetic patterns, hindering real-world generalization; (3) uniform denoising pressure promotes model overfitting to pseudo-label noise. We propose Cascade HQP-DETR to address these limitations. First, we introduce a high-quality data pipeline using LLaMA-3, Flux, and Grounding DINO to generate the FluxVOC and FluxCOCO datasets, advancing ISOD from weak to full supervision. Second, our High-Quality Proposal guided query encoding initializes object queries with image-specific priors from SAM-generated proposals and RoI-pooled features, accelerating convergence while steering the model to learn transferable features instead of overfitting to synthetic patterns. Third, our cascade denoising algorithm dynamically adjusts training weights through progressively increasing IoU thresholds across decoder layers, guiding the model to learn robust boundaries from reliable visual cues rather than overfitting to noisy labels. Trained for just 12 epochs solely on FluxVOC, Cascade HQP-DETR achieves a SOTA 61.04\% mAP@0.5 on PASCAL VOC 2007, outperforming strong baselines, with its competitive real-data performance confirming the architecture's universal applicability.
>
---
#### [new 023] An Image-Based Path Planning Algorithm Using a UAV Equipped with Stereo Vision
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出一种基于双目视觉的无人机图像路径规划算法，解决二维图像无法识别地形凹凸（如陨石坑、山丘）导致路径不安全的问题。通过立体匹配生成深度图，结合视觉特征检测与ArUco标记定位，实现自动路径规划，并与A*、PRM算法对比验证有效性。**

- **链接: []()**

> **作者:** Selim Ahmet Iz; Mustafa Unel
>
> **摘要:** This paper presents a novel image-based path planning algorithm that was developed using computer vision techniques, as well as its comparative analysis with well-known deterministic and probabilistic algorithms, namely A* and Probabilistic Road Map algorithm (PRM). The terrain depth has a significant impact on the calculated path safety. The craters and hills on the surface cannot be distinguished in a two-dimensional image. The proposed method uses a disparity map of the terrain that is generated by using a UAV. Several computer vision techniques, including edge, line and corner detection methods, as well as the stereo depth reconstruction technique, are applied to the captured images and the found disparity map is used to define candidate way-points of the trajectory. The initial and desired points are detected automatically using ArUco marker pose estimation and circle detection techniques. After presenting the mathematical model and vision techniques, the developed algorithm is compared with well-known algorithms on different virtual scenes created in the V-REP simulation program and a physical setup created in a laboratory environment. Results are promising and demonstrate effectiveness of the proposed algorithm.
>
---
#### [new 024] Distributed Zero-Shot Learning for Visual Recognition
- **分类: cs.CV**

- **简介: 该论文提出分布式零样本学习（DistZSL）框架，解决分布式节点数据异构下未见类别识别问题。通过跨节点属性正则化与全局属性-视觉一致性约束，稳定特征空间并统一映射关系，提升零样本识别性能。**

- **链接: []()**

> **作者:** Zhi Chen; Yadan Luo; Zi Huang; Jingjing Li; Sen Wang; Xin Yu
>
> **备注:** Accepted to IEEE Transactions on Multimedia in Oct 2025
>
> **摘要:** In this paper, we propose a Distributed Zero-Shot Learning (DistZSL) framework that can fully exploit decentralized data to learn an effective model for unseen classes. Considering the data heterogeneity issues across distributed nodes, we introduce two key components to ensure the effective learning of DistZSL: a cross-node attribute regularizer and a global attribute-to-visual consensus. Our proposed cross-node attribute regularizer enforces the distances between attribute features to be similar across different nodes. In this manner, the overall attribute feature space would be stable during learning, and thus facilitate the establishment of visual-to-attribute(V2A) relationships. Then, we introduce the global attribute-tovisual consensus to mitigate biased V2A mappings learned from individual nodes. Specifically, we enforce the bilateral mapping between the attribute and visual feature distributions to be consistent across different nodes. Thus, the learned consistent V2A mapping can significantly enhance zero-shot learning across different nodes. Extensive experiments demonstrate that DistZSL achieves superior performance to the state-of-the-art in learning from distributed data.
>
---
#### [new 025] Radar-APLANC: Unsupervised Radar-based Heartbeat Sensing via Augmented Pseudo-Label and Noise Contrast
- **分类: cs.CV; cs.AI; cs.HC; eess.SP**

- **简介: 论文提出Radar-APLANC，首个无监督雷达心率检测框架，通过伪标签增强与噪声对比学习，利用雷达回波中的心跳与噪声区域构建正负样本，无需标注生理信号，实现媲美监督方法的性能。**

- **链接: []()**

> **作者:** Ying Wang; Zhaodong Sun; Xu Cheng; Zuxian He; Xiaobai Li
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Frequency Modulated Continuous Wave (FMCW) radars can measure subtle chest wall oscillations to enable non-contact heartbeat sensing. However, traditional radar-based heartbeat sensing methods face performance degradation due to noise. Learning-based radar methods achieve better noise robustness but require costly labeled signals for supervised training. To overcome these limitations, we propose the first unsupervised framework for radar-based heartbeat sensing via Augmented Pseudo-Label and Noise Contrast (Radar-APLANC). We propose to use both the heartbeat range and noise range within the radar range matrix to construct the positive and negative samples, respectively, for improved noise robustness. Our Noise-Contrastive Triplet (NCT) loss only utilizes positive samples, negative samples, and pseudo-label signals generated by the traditional radar method, thereby avoiding dependence on expensive ground-truth physiological signals. We further design a pseudo-label augmentation approach featuring adaptive noise-aware label selection to improve pseudo-label signal quality. Extensive experiments on the Equipleth dataset and our collected radar dataset demonstrate that our unsupervised method achieves performance comparable to state-of-the-art supervised methods. Our code, dataset, and supplementary materials can be accessed from https://github.com/RadarHRSensing/Radar-APLANC.
>
---
#### [new 026] Boomda: Balanced Multi-objective Optimization for Multimodal Domain Adaptation
- **分类: cs.CV; cs.LG**

- **简介: 论文提出Boomda，解决多模态域自适应中各模态域偏移不一致问题，通过信息瓶颈与相关性对齐构建多目标优化框架，导出闭式解实现模态平衡对齐，显著提升适应性能。**

- **链接: []()**

> **作者:** Jun Sun; Xinxin Zhang; Simin Hong; Jian Zhu; Xiang Gao
>
> **摘要:** Multimodal learning, while contributing to numerous success stories across various fields, faces the challenge of prohibitively expensive manual annotation. To address the scarcity of annotated data, a popular solution is unsupervised domain adaptation, which has been extensively studied in unimodal settings yet remains less explored in multimodal settings. In this paper, we investigate heterogeneous multimodal domain adaptation, where the primary challenge is the varying domain shifts of different modalities from the source to the target domain. We first introduce the information bottleneck method to learn representations for each modality independently, and then match the source and target domains in the representation space with correlation alignment. To balance the domain alignment of all modalities, we formulate the problem as a multi-objective task, aiming for a Pareto optimal solution. By exploiting the properties specific to our model, the problem can be simplified to a quadratic programming problem. Further approximation yields a closed-form solution, leading to an efficient modality-balanced multimodal domain adaptation algorithm. The proposed method features \textbf{B}alanced multi-\textbf{o}bjective \textbf{o}ptimization for \textbf{m}ultimodal \textbf{d}omain \textbf{a}daptation, termed \textbf{Boomda}. Extensive empirical results showcase the effectiveness of the proposed approach and demonstrate that Boomda outperforms the competing schemes. The code is is available at: https://github.com/sunjunaimer/Boomda.git.
>
---
#### [new 027] Evaluating Gemini LLM in Food Image-Based Recipe and Nutrition Description with EfficientNet-B4 Visual Backbone
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究基于图像的食谱与营养信息生成任务，提出融合EfficientNet-B4与Gemini LLM的多模态系统，解决视觉识别误差对生成结果的语义传播问题，并构建中文食物数据集进行评估。**

- **链接: []()**

> **作者:** Rizal Khoirul Anam
>
> **摘要:** The proliferation of digital food applications necessitates robust methods for automated nutritional analysis and culinary guidance. This paper presents a comprehensive comparative evaluation of a decoupled, multimodal pipeline for food recognition. We evaluate a system integrating a specialized visual backbone (EfficientNet-B4) with a powerful generative large language model (Google's Gemini LLM). The core objective is to evaluate the trade-offs between visual classification accuracy, model efficiency, and the quality of generative output (nutritional data and recipes). We benchmark this pipeline against alternative vision backbones (VGG-16, ResNet-50, YOLOv8) and a lightweight LLM (Gemma). We introduce a formalization for "Semantic Error Propagation" (SEP) to analyze how classification inaccuracies from the visual module cascade into the generative output. Our analysis is grounded in a new Custom Chinese Food Dataset (CCFD) developed to address cultural bias in public datasets. Experimental results demonstrate that while EfficientNet-B4 (89.0\% Top-1 Acc.) provides the best balance of accuracy and efficiency, and Gemini (9.2/10 Factual Accuracy) provides superior generative quality, the system's overall utility is fundamentally bottlenecked by the visual front-end's perceptive accuracy. We conduct a detailed per-class analysis, identifying high semantic similarity as the most critical failure mode.
>
---
#### [new 028] OmniAID: Decoupling Semantic and Artifacts for Universal AI-Generated Image Detection in the Wild
- **分类: cs.CV**

- **简介: OmniAID提出一种解耦语义缺陷与通用伪影的MoE框架，用于通用AI生成图像检测，解决现有方法泛化差、基准过时问题，并构建新数据集Mirage验证效果。**

- **链接: []()**

> **作者:** Yuncheng Guo; Junyan Ye; Chenjue Zhang; Hengrui Kang; Haohuan Fu; Conghui He; Weijia Li
>
> **备注:** 11 pages, 7 figures, 5 tables
>
> **摘要:** A truly universal AI-Generated Image (AIGI) detector must simultaneously generalize across diverse generative models and varied semantic content. Current state-of-the-art methods learn a single, entangled forgery representation--conflating content-dependent flaws with content-agnostic artifacts--and are further constrained by outdated benchmarks. To overcome these limitations, we propose OmniAID, a novel framework centered on a decoupled Mixture-of-Experts (MoE) architecture. The core of our method is a hybrid expert system engineered to decouple: (1) semantic flaws across distinct content domains, and (2) these content-dependent flaws from content-agnostic universal artifacts. This system employs a set of Routable Specialized Semantic Experts, each for a distinct domain (e.g., human, animal), complemented by a Fixed Universal Artifact Expert. This architecture is trained using a bespoke two-stage strategy: we first train the experts independently with domain-specific hard-sampling to ensure specialization, and subsequently train a lightweight gating network for effective input routing. By explicitly decoupling "what is generated" (content-specific flaws) from "how it is generated" (universal artifacts), OmniAID achieves robust generalization. To address outdated benchmarks and validate real-world applicability, we introduce Mirage, a new large-scale, contemporary dataset. Extensive experiments, using both traditional benchmarks and our Mirage dataset, demonstrate our model surpasses existing monolithic detectors, establishing a new, robust standard for AIGI authentication against modern, in-the-wild threats.
>
---
#### [new 029] Modulo Video Recovery via Selective Spatiotemporal Vision Transformer
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文针对模量视频重建任务，解决传统传感器动态范围不足导致信号折叠问题，提出首个基于选择性时空视觉变换器（SSViT）的深度学习框架，高效恢复高动态范围视频，实现当前最优性能。**

- **链接: []()**

> **作者:** Tianyu Geng; Feng Ji; Wee Peng Tay
>
> **摘要:** Conventional image sensors have limited dynamic range, causing saturation in high-dynamic-range (HDR) scenes. Modulo cameras address this by folding incident irradiance into a bounded range, yet require specialized unwrapping algorithms to reconstruct the underlying signal. Unlike HDR recovery, which extends dynamic range from conventional sampling, modulo recovery restores actual values from folded samples. Despite being introduced over a decade ago, progress in modulo image recovery has been slow, especially in the use of modern deep learning techniques. In this work, we demonstrate that standard HDR methods are unsuitable for modulo recovery. Transformers, however, can capture global dependencies and spatial-temporal relationships crucial for resolving folded video frames. Still, adapting existing Transformer architectures for modulo recovery demands novel techniques. To this end, we present Selective Spatiotemporal Vision Transformer (SSViT), the first deep learning framework for modulo video reconstruction. SSViT employs a token selection strategy to improve efficiency and concentrate on the most critical regions. Experiments confirm that SSViT produces high-quality reconstructions from 8-bit folded videos and achieves state-of-the-art performance in modulo video recovery.
>
---
#### [new 030] Auto-US: An Ultrasound Video Diagnosis Agent Using Video Classification Framework and LLMs
- **分类: cs.CV; cs.AI**

- **简介: 论文提出Auto-US，面向超声视频诊断任务，解决数据稀缺与诊断精度低问题。构建CUV数据集，设计CTU-Net实现86.73%分类准确率，并融合LLM生成临床诊断建议，经医生验证有效。**

- **链接: []()**

> **作者:** Yuezhe Yang; Yiyue Guo; Wenjie Cai; Qingqing Ruan; Siying Wang; Xingbo Dong; Zhe Jin; Yong Dai
>
> **备注:** Under Review
>
> **摘要:** AI-assisted ultrasound video diagnosis presents new opportunities to enhance the efficiency and accuracy of medical imaging analysis. However, existing research remains limited in terms of dataset diversity, diagnostic performance, and clinical applicability. In this study, we propose \textbf{Auto-US}, an intelligent diagnosis agent that integrates ultrasound video data with clinical diagnostic text. To support this, we constructed \textbf{CUV Dataset} of 495 ultrasound videos spanning five categories and three organs, aggregated from multiple open-access sources. We developed \textbf{CTU-Net}, which achieves state-of-the-art performance in ultrasound video classification, reaching an accuracy of 86.73\% Furthermore, by incorporating large language models, Auto-US is capable of generating clinically meaningful diagnostic suggestions. The final diagnostic scores for each case exceeded 3 out of 5 and were validated by professional clinicians. These results demonstrate the effectiveness and clinical potential of Auto-US in real-world ultrasound applications. Code and data are available at: https://github.com/Bean-Young/Auto-US.
>
---
#### [new 031] Learning Sparse Label Couplings for Multilabel Chest X-Ray Diagnosis
- **分类: cs.CV**

- **简介: 该论文针对胸部X光多标签分类任务，解决标签不平衡与共现关系建模问题。基于SE-ResNeXt101构建高效Pipeline，提出轻量级Label-Graph Refinement模块，通过稀疏可学习标签耦合矩阵优化 logits，显著提升宏AUC且无需额外标注。**

- **链接: []()**

> **作者:** Utkarsh Prakash Srivastava; Kaushik Gupta; Kaushik Nath
>
> **备注:** 7 pages, 8 figures
>
> **摘要:** We study multilabel classification of chest X-rays and present a simple, strong pipeline built on SE-ResNeXt101 $(32 \times 4d)$. The backbone is finetuned for 14 thoracic findings with a sigmoid head, trained using Multilabel Iterative Stratification (MIS) for robust cross-validation splits that preserve label co-occurrence. To address extreme class imbalance and asymmetric error costs, we optimize with Asymmetric Loss, employ mixed-precision (AMP), cosine learning-rate decay with warm-up, gradient clipping, and an exponential moving average (EMA) of weights. We propose a lightweight Label-Graph Refinement module placed after the classifier: given per-label probabilities, it learns a sparse, trainable inter-label coupling matrix that refines logits via a single message-passing step while adding only an L1-regularized parameter head. At inference, we apply horizontal flip test-time augmentation (TTA) and average predictions across MIS folds (a compact deep ensemble). Evaluation uses macro AUC averaging classwise ROC-AUC and skipping single-class labels in a fold to reflect balanced performance across conditions. On our dataset, a strong SE-ResNeXt101 baseline attains competitive macro AUC (e.g., 92.64% in our runs). Adding the Label-Graph Refinement consistently improves validation macro AUC across folds with negligible compute. The resulting method is reproducible, hardware-friendly, and requires no extra annotations, offering a practical route to stronger multilabel CXR classifiers.
>
---
#### [new 032] VectorSynth: Fine-Grained Satellite Image Synthesis with Structured Semantics
- **分类: cs.CV**

- **简介: VectorSynth提出一种基于扩散模型的卫星图像合成方法，利用多边形地理注记与语义属性进行像素级条件生成，解决传统方法缺乏空间语义精准控制的问题，实现地理结构与语义驱动的图像编辑。**

- **链接: []()**

> **作者:** Daniel Cher; Brian Wei; Srikumar Sastry; Nathan Jacobs
>
> **摘要:** We introduce VectorSynth, a diffusion-based framework for pixel-accurate satellite image synthesis conditioned on polygonal geographic annotations with semantic attributes. Unlike prior text- or layout-conditioned models, VectorSynth learns dense cross-modal correspondences that align imagery and semantic vector geometry, enabling fine-grained, spatially grounded edits. A vision language alignment module produces pixel-level embeddings from polygon semantics; these embeddings guide a conditional image generation framework to respect both spatial extents and semantic cues. VectorSynth supports interactive workflows that mix language prompts with geometry-aware conditioning, allowing rapid what-if simulations, spatial edits, and map-informed content generation. For training and evaluation, we assemble a collection of satellite scenes paired with pixel-registered polygon annotations spanning diverse urban scenes with both built and natural features. We observe strong improvements over prior methods in semantic fidelity and structural realism, and show that our trained vision language model demonstrates fine-grained spatial grounding. The code and data are available at https://github.com/mvrl/VectorSynth.
>
---
#### [new 033] 2D Representation for Unguided Single-View 3D Super-Resolution in Real-Time
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出2Dto3D-SR，将单视角3D数据编码为2D表示（PNCC），实现无高分辨率RGB引导的实时3D超分辨率，突破传统方法依赖复杂3D结构或RGB指导的限制，兼顾精度与效率。**

- **链接: []()**

> **作者:** Ignasi Mas; Ivan Huerta; Ramon Morros; Javier Ruiz-Hidalgo
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** We introduce 2Dto3D-SR, a versatile framework for real-time single-view 3D super-resolution that eliminates the need for high-resolution RGB guidance. Our framework encodes 3D data from a single viewpoint into a structured 2D representation, enabling the direct application of existing 2D image super-resolution architectures. We utilize the Projected Normalized Coordinate Code (PNCC) to represent 3D geometry from a visible surface as a regular image, thereby circumventing the complexities of 3D point-based or RGB-guided methods. This design supports lightweight and fast models adaptable to various deployment environments. We evaluate 2Dto3D-SR with two implementations: one using Swin Transformers for high accuracy, and another using Vision Mamba for high efficiency. Experiments show the Swin Transformer model achieves state-of-the-art accuracy on standard benchmarks, while the Vision Mamba model delivers competitive results at real-time speeds. This establishes our geometry-guided pipeline as a surprisingly simple yet viable and practical solution for real-world scenarios, especially where high-resolution RGB data is inaccessible.
>
---
#### [new 034] Human Motion Synthesis in 3D Scenes via Unified Scene Semantic Occupancy
- **分类: cs.CV**

- **简介: 该论文提出SSOMotion框架，用于3D场景中的人体动作合成，解决现有方法忽视语义理解的问题。通过统一语义占据表示（SSO）与CLIP编码，结合双向三平面分解与帧级场景查询，实现细粒度语义驱动的运动生成。**

- **链接: []()**

> **作者:** Gong Jingyu; Tong Kunkun; Chen Zhuoran; Yuan Chuanhan; Chen Mingang; Zhang Zhizhong; Tan Xin; Xie Yuan
>
> **摘要:** Human motion synthesis in 3D scenes relies heavily on scene comprehension, while current methods focus mainly on scene structure but ignore the semantic understanding. In this paper, we propose a human motion synthesis framework that take an unified Scene Semantic Occupancy (SSO) for scene representation, termed SSOMotion. We design a bi-directional tri-plane decomposition to derive a compact version of the SSO, and scene semantics are mapped to an unified feature space via CLIP encoding and shared linear dimensionality reduction. Such strategy can derive the fine-grained scene semantic structures while significantly reduce redundant computations. We further take these scene hints and movement direction derived from instructions for motion control via frame-wise scene query. Extensive experiments and ablation studies conducted on cluttered scenes using ShapeNet furniture, as well as scanned scenes from PROX and Replica datasets, demonstrate its cutting-edge performance while validating its effectiveness and generalization ability. Code will be publicly available at https://github.com/jingyugong/SSOMotion.
>
---
#### [new 035] Semantic-Consistent Bidirectional Contrastive Hashing for Noisy Multi-Label Cross-Modal Retrieval
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向噪声多标签跨模态检索任务，提出SCBCH框架，通过跨模态语义一致性估计样本可靠性，并利用双向软对比哈希挖掘标签语义重叠，提升噪声环境下的检索鲁棒性。**

- **链接: []()**

> **作者:** Likang Peng; Chao Su; Wenyuan Wu; Yuan Sun; Dezhong Peng; Xi Peng; Xu Wang
>
> **摘要:** Cross-modal hashing (CMH) facilitates efficient retrieval across different modalities (e.g., image and text) by encoding data into compact binary representations. While recent methods have achieved remarkable performance, they often rely heavily on fully annotated datasets, which are costly and labor-intensive to obtain. In real-world scenarios, particularly in multi-label datasets, label noise is prevalent and severely degrades retrieval performance. Moreover, existing CMH approaches typically overlook the partial semantic overlaps inherent in multi-label data, limiting their robustness and generalization. To tackle these challenges, we propose a novel framework named Semantic-Consistent Bidirectional Contrastive Hashing (SCBCH). The framework comprises two complementary modules: (1) Cross-modal Semantic-Consistent Classification (CSCC), which leverages cross-modal semantic consistency to estimate sample reliability and reduce the impact of noisy labels; (2) Bidirectional Soft Contrastive Hashing (BSCH), which dynamically generates soft contrastive sample pairs based on multi-label semantic overlap, enabling adaptive contrastive learning between semantically similar and dissimilar samples across modalities. Extensive experiments on four widely-used cross-modal retrieval benchmarks validate the effectiveness and robustness of our method, consistently outperforming state-of-the-art approaches under noisy multi-label conditions.
>
---
#### [new 036] ImagebindDC: Compressing Multi-modal Data with Imagebind-based Condensation
- **分类: cs.CV; cs.AI**

- **简介: 论文提出ImageBindDC，面向多模态数据压缩任务，解决传统方法难以保留跨模态依赖的问题。基于ImageBind统一特征空间，引入特征函数损失，实现三重分布对齐，显著提升压缩效率与性能。**

- **链接: []()**

> **作者:** Yue Min; Shaobo Wang; Jiaze Li; Tianle Niu; Junxin Fan; Yongliang Miao; Lijin Yang; Linfeng Zhang
>
> **备注:** AAAI 2026, 18 pages, 6 figures, 6 tables
>
> **摘要:** Data condensation techniques aim to synthesize a compact dataset from a larger one to enable efficient model training, yet while successful in unimodal settings, they often fail in multimodal scenarios where preserving intricate inter-modal dependencies is crucial. To address this, we introduce ImageBindDC, a novel data condensation framework operating within the unified feature space of ImageBind. Our approach moves beyond conventional distribution-matching by employing a powerful Characteristic Function (CF) loss, which operates in the Fourier domain to facilitate a more precise statistical alignment via exact infinite moment matching. We design our objective to enforce three critical levels of distributional consistency: (i) uni-modal alignment, which matches the statistical properties of synthetic and real data within each modality; (ii) cross-modal alignment, which preserves pairwise semantics by matching the distributions of hybrid real-synthetic data pairs; and (iii) joint-modal alignment, which captures the complete multivariate data structure by aligning the joint distribution of real data pairs with their synthetic counterparts. Extensive experiments highlight the effectiveness of ImageBindDC: on the NYU-v2 dataset, a model trained on just 5 condensed datapoints per class achieves lossless performance comparable to one trained on the full dataset, achieving a new state-of-the-art with an 8.2\% absolute improvement over the previous best method and more than 4$\times$ less condensation time.
>
---
#### [new 037] Class Incremental Medical Image Segmentation via Prototype-Guided Calibration and Dual-Aligned Distillation
- **分类: cs.CV**

- **简介: 该论文面向类增量医学图像分割任务，解决旧类知识遗忘问题，提出原型引导校准与双对齐蒸馏方法，通过局部-全局原型对齐和区域自适应蒸馏，有效保留旧类知识并提升新旧类分割性能。**

- **链接: []()**

> **作者:** Shengqian Zhu; Chengrong Yu; Qiang Wang; Ying Song; Guangjun Li; Jiafei Wu; Xiaogang Xu; Zhang Yi; Junjie Hu
>
> **摘要:** Class incremental medical image segmentation (CIMIS) aims to preserve knowledge of previously learned classes while learning new ones without relying on old-class labels. However, existing methods 1) either adopt one-size-fits-all strategies that treat all spatial regions and feature channels equally, which may hinder the preservation of accurate old knowledge, 2) or focus solely on aligning local prototypes with global ones for old classes while overlooking their local representations in new data, leading to knowledge degradation. To mitigate the above issues, we propose Prototype-Guided Calibration Distillation (PGCD) and Dual-Aligned Prototype Distillation (DAPD) for CIMIS in this paper. Specifically, PGCD exploits prototype-to-feature similarity to calibrate class-specific distillation intensity in different spatial regions, effectively reinforcing reliable old knowledge and suppressing misleading information from old classes. Complementarily, DAPD aligns the local prototypes of old classes extracted from the current model with both global prototypes and local prototypes, further enhancing segmentation performance on old categories. Comprehensive evaluations on two widely used multi-organ segmentation benchmarks demonstrate that our method outperforms state-of-the-art methods, highlighting its robustness and generalization capabilities.
>
---
#### [new 038] Libra-MIL: Multimodal Prototypes Stereoscopic Infused with Task-specific Language Priors for Few-shot Whole Slide Image Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向少样本全切片图像分类任务，解决LLM生成实例描述存在医学偏差与跨模态协同不足问题，提出Libra-MIL，通过任务特异性文本原型与视觉原型的双向交互，结合立体最优传输实现高效多模态融合。**

- **链接: []()**

> **作者:** Zhenfeng Zhuang; Fangyu Zhou; Liansheng Wang
>
> **摘要:** While Large Language Models (LLMs) are emerging as a promising direction in computational pathology, the substantial computational cost of giga-pixel Whole Slide Images (WSIs) necessitates the use of Multi-Instance Learning (MIL) to enable effective modeling. A key challenge is that pathological tasks typically provide only bag-level labels, while instance-level descriptions generated by LLMs often suffer from bias due to a lack of fine-grained medical knowledge. To address this, we propose that constructing task-specific pathological entity prototypes is crucial for learning generalizable features and enhancing model interpretability. Furthermore, existing vision-language MIL methods often employ unidirectional guidance, limiting cross-modal synergy. In this paper, we introduce a novel approach, Multimodal Prototype-based Multi-Instance Learning, that promotes bidirectional interaction through a balanced information compression scheme. Specifically, we leverage a frozen LLM to generate task-specific pathological entity descriptions, which are learned as text prototypes. Concurrently, the vision branch learns instance-level prototypes to mitigate the model's reliance on redundant data. For the fusion stage, we employ the Stereoscopic Optimal Transport (SOT) algorithm, which is based on a similarity metric, thereby facilitating broader semantic alignment in a higher-dimensional space. We conduct few-shot classification and explainability experiments on three distinct cancer datasets, and the results demonstrate the superior generalization capabilities of our proposed method.
>
---
#### [new 039] Cross-pyramid consistency regularization for semi-supervised medical image segmentation
- **分类: cs.CV**

- **简介: 该论文针对半监督医学图像分割任务，提出DBPNet与交叉金字塔一致性正则化（CPCR），利用双解码器的多尺度预测实现知识蒸馏，有效利用未标注数据提升分割性能。**

- **链接: []()**

> **作者:** Matus Bojko; Maros Kollar; Marek Jakab; Wanda Benesova
>
> **摘要:** Semi-supervised learning (SSL) enables training of powerful models with the assumption of limited, carefully labelled data and a large amount of unlabeled data to support the learning. In this paper, we propose a hybrid consistency learning approach to effectively exploit unlabeled data for semi-supervised medical image segmentation by leveraging Cross-Pyramid Consistency Regularization (CPCR) between two decoders. First, we design a hybrid Dual Branch Pyramid Network (DBPNet), consisting of an encoder and two decoders that differ slightly, each producing a pyramid of perturbed auxiliary predictions across multiple resolution scales. Second, we present a learning strategy for this network named CPCR that combines existing consistency learning and uncertainty minimization approaches on the main output predictions of decoders with our novel regularization term. More specifically, in this term, we extend the soft-labeling setting to pyramid predictions across decoders to support knowledge distillation in deep hierarchical features. Experimental results show that DBPNet with CPCR outperforms five state-of-the-art self-supervised learning methods and has comparable performance with recent ones on a public benchmark dataset.
>
---
#### [new 040] CLIP is All You Need for Human-like Semantic Representations in Stable Diffusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文探究Stable Diffusion是否具备人类可理解的语义表征，发现其语义信息主要源自CLIP文本编码器，而非扩散过程，扩散仅起视觉解码作用，揭示了CLIP在语义表征中的核心地位。**

- **链接: []()**

> **作者:** Cameron Braunstein; Mariya Toneva; Eddy Ilg
>
> **备注:** 28 pages, 8 figures, 2 tables
>
> **摘要:** Latent diffusion models such as Stable Diffusion achieve state-of-the-art results on text-to-image generation tasks. However, the extent to which these models have a semantic understanding of the images they generate is not well understood. In this work, we investigate whether the internal representations used by these models during text-to-image generation contain semantic information that is meaningful to humans. To do so, we perform probing on Stable Diffusion with simple regression layers that predict semantic attributes for objects and evaluate these predictions against human annotations. Surprisingly, we find that this success can actually be attributed to the text encoding occurring in CLIP rather than the reverse diffusion process. We demonstrate that groups of specific semantic attributes have markedly different decoding accuracy than the average, and are thus represented to different degrees. Finally, we show that attributes become more difficult to disambiguate from one another during the inverse diffusion process, further demonstrating the strongest semantic representation of object attributes in CLIP. We conclude that the separately trained CLIP vision-language model is what determines the human-like semantic representation, and that the diffusion process instead takes the role of a visual decoder.
>
---
#### [new 041] Vision Transformer Based User Equipment Positioning
- **分类: cs.CV; cs.NI**

- **简介: 该论文提出基于Vision Transformer的UE定位方法，解决传统DL模型对非序列CSI数据关注不均问题，通过ADP特征实现高精度定位，在多个数据集上误差降低约38%。**

- **链接: []()**

> **作者:** Parshwa Shah; Dhaval K. Patel; Brijesh Soni; Miguel López-Benítez; Siddhartan Govindasamy
>
> **备注:** The results are accepted in parts at IEEE CCNC2026
>
> **摘要:** Recently, Deep Learning (DL) techniques have been used for User Equipment (UE) positioning. However, the key shortcomings of such models is that: i) they weigh the same attention to the entire input; ii) they are not well suited for the non-sequential data e.g., when only instantaneous Channel State Information (CSI) is available. In this context, we propose an attention-based Vision Transformer (ViT) architecture that focuses on the Angle Delay Profile (ADP) from CSI matrix. Our approach, validated on the `DeepMIMO' and `ViWi' ray-tracing datasets, achieves an Root Mean Squared Error (RMSE) of 0.55m indoors, 13.59m outdoors in DeepMIMO, and 3.45m in ViWi's outdoor blockage scenario. The proposed scheme outperforms state-of-the-art schemes by $\sim$ 38\%. It also performs substantially better than other approaches that we have considered in terms of the distribution of error distance.
>
---
#### [new 042] ChexFract: From General to Specialized - Enhancing Fracture Description Generation
- **分类: cs.CV**

- **简介: 该论文面向胸片骨折描述生成任务，解决通用模型对罕见骨折描述不足的问题，通过构建基于MAIRA-2和CheXagent的骨折专用视觉-语言模型，显著提升骨折细节生成准确率，并开源最佳模型推动罕见病理报告研究。**

- **链接: []()**

> **作者:** Nikolay Nechaev; Evgeniia Przhezdzetskaia; Dmitry Umerenkov; Dmitry V. Dylov
>
> **备注:** 13 pages, 3 figures
>
> **摘要:** Generating accurate and clinically meaningful radiology reports from chest X-ray images remains a significant challenge in medical AI. While recent vision-language models achieve strong results in general radiology report generation, they often fail to adequately describe rare but clinically important pathologies like fractures. This work addresses this gap by developing specialized models for fracture pathology detection and description. We train fracture-specific vision-language models with encoders from MAIRA-2 and CheXagent, demonstrating significant improvements over general-purpose models in generating accurate fracture descriptions. Analysis of model outputs by fracture type, location, and age reveals distinct strengths and limitations of current vision-language model architectures. We publicly release our best-performing fracture-reporting model, facilitating future research in accurate reporting of rare pathologies.
>
---
#### [new 043] WarpGAN: Warping-Guided 3D GAN Inversion with Style-Based Novel View Inpainting
- **分类: cs.CV**

- **简介: WarpGAN针对3D GAN逆向建模中遮挡区域质量差的问题，提出融合形变与风格化补全的策略，通过深度引导形变和对称性约束的单视图补全网络，提升遮挡区域的真实感与多视角一致性。**

- **链接: []()**

> **作者:** Kaitao Huang; Yan Yan; Jing-Hao Xue; Hanzi Wang
>
> **摘要:** 3D GAN inversion projects a single image into the latent space of a pre-trained 3D GAN to achieve single-shot novel view synthesis, which requires visible regions with high fidelity and occluded regions with realism and multi-view consistency. However, existing methods focus on the reconstruction of visible regions, while the generation of occluded regions relies only on the generative prior of 3D GAN. As a result, the generated occluded regions often exhibit poor quality due to the information loss caused by the low bit-rate latent code. To address this, we introduce the warping-and-inpainting strategy to incorporate image inpainting into 3D GAN inversion and propose a novel 3D GAN inversion method, WarpGAN. Specifically, we first employ a 3D GAN inversion encoder to project the single-view image into a latent code that serves as the input to 3D GAN. Then, we perform warping to a novel view using the depth map generated by 3D GAN. Finally, we develop a novel SVINet, which leverages the symmetry prior and multi-view image correspondence w.r.t. the same latent code to perform inpainting of occluded regions in the warped image. Quantitative and qualitative experiments demonstrate that our method consistently outperforms several state-of-the-art methods.
>
---
#### [new 044] Multi-Modal Assistance for Unsupervised Domain Adaptation on Point Cloud 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文针对LiDAR 3D目标检测的无监督域自适应任务，提出MMAssist方法，利用图像与文本特征作为桥梁对齐跨域3D特征，并融合多模态信息提升伪标签质量，显著提升跨域检测性能。**

- **链接: []()**

> **作者:** Shenao Zhao; Pengpeng Liang; Zhoufan Yang
>
> **备注:** Accepted to AAAI-26
>
> **摘要:** Unsupervised domain adaptation for LiDAR-based 3D object detection (3D UDA) based on the teacher-student architecture with pseudo labels has achieved notable improvements in recent years. Although it is quite popular to collect point clouds and images simultaneously, little attention has been paid to the usefulness of image data in 3D UDA when training the models. In this paper, we propose an approach named MMAssist that improves the performance of 3D UDA with multi-modal assistance. A method is designed to align 3D features between the source domain and the target domain by using image and text features as bridges. More specifically, we project the ground truth labels or pseudo labels to the images to get a set of 2D bounding boxes. For each 2D box, we extract its image feature from a pre-trained vision backbone. A large vision-language model (LVLM) is adopted to extract the box's text description, and a pre-trained text encoder is used to obtain its text feature. During the training of the model in the source domain and the student model in the target domain, we align the 3D features of the predicted boxes with their corresponding image and text features, and the 3D features and the aligned features are fused with learned weights for the final prediction. The features between the student branch and the teacher branch in the target domain are aligned as well. To enhance the pseudo labels, we use an off-the-shelf 2D object detector to generate 2D bounding boxes from images and estimate their corresponding 3D boxes with the aid of point cloud, and these 3D boxes are combined with the pseudo labels generated by the teacher model. Experimental results show that our approach achieves promising performance compared with state-of-the-art methods in three domain adaptation tasks on three popular 3D object detection datasets. The code is available at https://github.com/liangp/MMAssist.
>
---
#### [new 045] LiveNeRF: Efficient Face Replacement Through Neural Radiance Fields Integration
- **分类: cs.CV**

- **简介: LiveNeRF提出一种基于神经辐射场的实时人脸替换方法，实现33 FPS的高质量人脸同步，解决传统方法延迟高、质量差的问题，支持直播与视频会议等场景，并强调负责任使用。**

- **链接: []()**

> **作者:** Tung Vu; Hai Nguyen; Cong Tran
>
> **摘要:** Face replacement technology enables significant advancements in entertainment, education, and communication applications, including dubbing, virtual avatars, and cross-cultural content adaptation. Our LiveNeRF framework addresses critical limitations of existing methods by achieving real-time performance (33 FPS) with superior visual quality, enabling practical deployment in live streaming, video conferencing, and interactive media. The technology particularly benefits content creators, educators, and individuals with speech impairments through accessible avatar communication. While acknowledging potential misuse in unauthorized deepfake creation, we advocate for responsible deployment with user consent verification and integration with detection systems to ensure positive societal impact while minimizing risks.
>
---
#### [new 046] Taming Identity Consistency and Prompt Diversity in Diffusion Models via Latent Concatenation and Masked Conditional Flow Matching
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向主体驱动图像生成任务，解决身份一致性与提示多样性间的权衡问题。提出隐式拼接与掩码流匹配方法，结合两阶段数据筛选与CHARIS评估框架，在不修改架构下实现高效身份保持与多样化生成。**

- **链接: []()**

> **作者:** Aditi Singhania; Arushi Jain; Krutik Malani; Riddhi Dhawan; Souymodip Chakraborty; Vineet Batra; Ankit Phogat
>
> **摘要:** Subject-driven image generation aims to synthesize novel depictions of a specific subject across diverse contexts while preserving its core identity features. Achieving both strong identity consistency and high prompt diversity presents a fundamental trade-off. We propose a LoRA fine-tuned diffusion model employing a latent concatenation strategy, which jointly processes reference and target images, combined with a masked Conditional Flow Matching (CFM) objective. This approach enables robust identity preservation without architectural modifications. To facilitate large-scale training, we introduce a two-stage Distilled Data Curation Framework: the first stage leverages data restoration and VLM-based filtering to create a compact, high-quality seed dataset from diverse sources; the second stage utilizes these curated examples for parameter-efficient fine-tuning, thus scaling the generation capability across various subjects and contexts. Finally, for filtering and quality assessment, we present CHARIS, a fine-grained evaluation framework that performs attribute-level comparisons along five key axes: identity consistency, prompt adherence, region-wise color fidelity, visual quality, and transformation diversity.
>
---
#### [new 047] MAUGIF: Mechanism-Aware Unsupervised General Image Fusion via Dual Cross-Image Autoencoders
- **分类: cs.CV**

- **简介: 该论文提出MAUGIF，用于通用图像融合任务，解决现有方法忽略不同融合机制的问题。通过双交叉自编码器区分加性与乘性融合机制，实现无监督、机制感知的多源图像融合。**

- **链接: []()**

> **作者:** Kunjing Yang; Zhiwei Wang; Minru Bai
>
> **摘要:** Image fusion aims to integrate structural and complementary information from multi-source images. However, existing fusion methods are often either highly task-specific, or general frameworks that apply uniform strategies across diverse tasks, ignoring their distinct fusion mechanisms. To address this issue, we propose a mechanism-aware unsupervised general image fusion (MAUGIF) method based on dual cross-image autoencoders. Initially, we introduce a classification of additive and multiplicative fusion according to the inherent mechanisms of different fusion tasks. Then, dual encoders map source images into a shared latent space, capturing common content while isolating modality-specific details. During the decoding phase, dual decoders act as feature injectors, selectively reintegrating the unique characteristics of each modality into the shared content for reconstruction. The modality-specific features are injected into the source image in the fusion process, generating the fused image that integrates information from both modalities. The architecture of decoders varies according to their fusion mechanisms, enhancing both performance and interpretability. Extensive experiments are conducted on diverse fusion tasks to validate the effectiveness and generalization ability of our method. The code is available at https://anonymous.4open.science/r/MAUGIF.
>
---
#### [new 048] Extreme Model Compression with Structured Sparsity at Low Precision
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出SLOPE框架，联合优化结构化稀疏与低精度量化，解决二者协同时精度骤降问题，通过角对齐正则化实现模型压缩20倍且保持99%准确率，显著优于现有方法。**

- **链接: []()**

> **作者:** Dan Liu; Nikita Dvornik; Xue Liu
>
> **备注:** 36th British Machine Vision Conference 2025
>
> **摘要:** Deep neural networks (DNNs) are used in many applications, but their large size and high computational cost make them hard to run on devices with limited resources. Two widely used techniques to address this challenge are weight quantization, which lowers the precision of all weights, and structured sparsity, which removes unimportant weights while retaining the important ones at full precision. Although both are effective individually, they are typically studied in isolation due to their compounded negative impact on model accuracy when combined. In this work, we introduce SLOPE Structured Sparsity at Low Precision), a unified framework, to effectively combine structured sparsity and low-bit quantization in a principled way. We show that naively combining sparsity and quantization severely harms performance due to the compounded impact of both techniques. To address this, we propose a training-time regularization strategy that minimizes the discrepancy between full-precision weights and their sparse, quantized counterparts by promoting angular alignment rather than direct matching. On ResNet-18, SLOPE achieves $\sim20\times$ model size reduction while retaining $\sim$99% of the original accuracy. It consistently outperforms state-of-the-art quantization and structured sparsity methods across classification, detection, and segmentation tasks on models such as ResNet-18, ViT-Small, and Mask R-CNN.
>
---
#### [new 049] Towards Open-Set Myoelectric Gesture Recognition via Dual-Perspective Inconsistency Learning
- **分类: cs.CV; cs.AI; cs.HC**

- **简介: 该论文面向开放集肌电手势识别任务，解决训练数据稀缺导致的过拟合问题，提出SASG-DA扩散增强方法，通过语义引导与稀疏感知采样生成高保真、高多样性样本，显著提升模型泛化能力。**

- **链接: []()**

> **作者:** Chen Liu; Can Han; Weishi Xu; Yaqi Wang; Dahong Qian
>
> **备注:** Under review
>
> **摘要:** Surface electromyography (sEMG)-based gesture recognition plays a critical role in human-machine interaction (HMI), particularly for rehabilitation and prosthetic control. However, sEMG-based systems often suffer from the scarcity of informative training data, leading to overfitting and poor generalization in deep learning models. Data augmentation offers a promising approach to increasing the size and diversity of training data, where faithfulness and diversity are two critical factors to effectiveness. However, promoting untargeted diversity can result in redundant samples with limited utility. To address these challenges, we propose a novel diffusion-based data augmentation approach, Sparse-Aware Semantic-Guided Diffusion Augmentation (SASG-DA). To enhance generation faithfulness, we introduce the Semantic Representation Guidance (SRG) mechanism by leveraging fine-grained, task-aware semantic representations as generation conditions. To enable flexible and diverse sample generation, we propose a Gaussian Modeling Semantic Modeling (GMSS) strategy, which models the semantic representation distribution and allows stochastic sampling to produce both faithful and diverse samples. To enhance targeted diversity, we further introduce a Sparse-Aware Semantic Sampling strategy to explicitly explore underrepresented regions, improving distribution coverage and sample utility. Extensive experiments on benchmark sEMG datasets, Ninapro DB2, DB4, and DB7, demonstrate that SASG-DA significantly outperforms existing augmentation methods. Overall, our proposed data augmentation approach effectively mitigates overfitting and improves recognition performance and generalization by offering both faithful and diverse samples.
>
---
#### [new 050] The Impact of Longitudinal Mammogram Alignment on Breast Cancer Risk Assessment
- **分类: cs.CV**

- **简介: 该论文研究纵向乳腺X光片对齐对乳腺癌风险预测的影响，解决因图像错位导致模型性能下降的问题。比较了图像级、特征级和隐式对齐方法，发现图像配准最有效，显著提升预测准确性与变形场质量。**

- **链接: []()**

> **作者:** Solveig Thrun; Stine Hansen; Zijun Sun; Nele Blum; Suaiba A. Salahuddin; Xin Wang; Kristoffer Wickstrøm; Elisabeth Wetzer; Robert Jenssen; Maik Stille; Michael Kampffmeyer
>
> **摘要:** Regular mammography screening is crucial for early breast cancer detection. By leveraging deep learning-based risk models, screening intervals can be personalized, especially for high-risk individuals. While recent methods increasingly incorporate longitudinal information from prior mammograms, accurate spatial alignment across time points remains a key challenge. Misalignment can obscure meaningful tissue changes and degrade model performance. In this study, we provide insights into various alignment strategies, image-based registration, feature-level (representation space) alignment with and without regularization, and implicit alignment methods, for their effectiveness in longitudinal deep learning-based risk modeling. Using two large-scale mammography datasets, we assess each method across key metrics, including predictive accuracy, precision, recall, and deformation field quality. Our results show that image-based registration consistently outperforms the more recently favored feature-based and implicit approaches across all metrics, enabling more accurate, temporally consistent predictions and generating smooth, anatomically plausible deformation fields. Although regularizing the deformation field improves deformation quality, it reduces the risk prediction performance of feature-level alignment. Applying image-based deformation fields within the feature space yields the best risk prediction performance. These findings underscore the importance of image-based deformation fields for spatial alignment in longitudinal risk modeling, offering improved prediction accuracy and robustness. This approach has strong potential to enhance personalized screening and enable earlier interventions for high-risk individuals. The code is available at https://github.com/sot176/Mammogram_Alignment_Study_Risk_Prediction.git, allowing full reproducibility of the results.
>
---
#### [new 051] Fast Multi-Organ Fine Segmentation in CT Images with Hierarchical Sparse Sampling and Residual Transformer
- **分类: cs.CV**

- **简介: 该论文针对CT图像多器官精细分割的效率问题，提出基于分层稀疏采样与残差Transformer的快速方法，在显著降低计算开销的同时提升精度，实现约2.24秒/例的CPU实时分割。**

- **链接: []()**

> **作者:** Xueqi Guo; Halid Ziya Yerebakan; Yoshihisa Shinagawa; Kritika Iyer; Gerardo Hermosillo Valadez
>
> **备注:** EMBC 2025 oral
>
> **摘要:** Multi-organ segmentation of 3D medical images is fundamental with meaningful applications in various clinical automation pipelines. Although deep learning has achieved superior performance, the time and memory consumption of segmenting the entire 3D volume voxel by voxel using neural networks can be huge. Classifiers have been developed as an alternative in cases with certain points of interest, but the trade-off between speed and accuracy remains an issue. Thus, we propose a novel fast multi-organ segmentation framework with the usage of hierarchical sparse sampling and a Residual Transformer. Compared with whole-volume analysis, the hierarchical sparse sampling strategy could successfully reduce computation time while preserving a meaningful hierarchical context utilizing multiple resolution levels. The architecture of the Residual Transformer segmentation network could extract and combine information from different levels of information in the sparse descriptor while maintaining a low computational cost. In an internal data set containing 10,253 CT images and the public dataset TotalSegmentator, the proposed method successfully improved qualitative and quantitative segmentation performance compared to the current fast organ classifier, with fast speed at the level of ~2.24 seconds on CPU hardware. The potential of achieving real-time fine organ segmentation is suggested.
>
---
#### [new 052] Top2Ground: A Height-Aware Dual Conditioning Diffusion Model for Robust Aerial-to-Ground View Generation
- **分类: cs.CV**

- **简介: Top2Ground提出一种高度感知的双条件扩散模型，直接从航拍图像生成逼真地面图像，解决视角差异大、遮挡和视野受限问题，无需中间3D表示，融合空间特征与语义嵌入，显著提升生成质量与泛化能力。**

- **链接: []()**

> **作者:** Jae Joong Lee; Bedrich Benes
>
> **摘要:** Generating ground-level images from aerial views is a challenging task due to extreme viewpoint disparity, occlusions, and a limited field of view. We introduce Top2Ground, a novel diffusion-based method that directly generates photorealistic ground-view images from aerial input images without relying on intermediate representations such as depth maps or 3D voxels. Specifically, we condition the denoising process on a joint representation of VAE-encoded spatial features (derived from aerial RGB images and an estimated height map) and CLIP-based semantic embeddings. This design ensures the generation is both geometrically constrained by the scene's 3D structure and semantically consistent with its content. We evaluate Top2Ground on three diverse datasets: CVUSA, CVACT, and the Auto Arborist. Our approach shows 7.3% average improvement in SSIM across three benchmark datasets, showing Top2Ground can robustly handle both wide and narrow fields of view, highlighting its strong generalization capabilities.
>
---
#### [new 053] Hierarchical Direction Perception via Atomic Dot-Product Operators for Rotation-Invariant Point Clouds Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对点云旋转不变学习问题，提出DiPVNet，通过原子点积算子实现局部自适应方向感知与全局球面傅里叶变换，兼具旋转不变性与多尺度方向建模能力，显著提升分类与分割性能。**

- **链接: []()**

> **作者:** Chenyu Hu; Xiaotong Li; Hao Zhu; Biao Hou
>
> **备注:** Accepted to AAAI 2026. Code is available at: https://github.com/wxszreal0/DiPVNet
>
> **摘要:** Point cloud processing has become a cornerstone technology in many 3D vision tasks. However, arbitrary rotations introduce variations in point cloud orientations, posing a long-standing challenge for effective representation learning. The core of this issue is the disruption of the point cloud's intrinsic directional characteristics caused by rotational perturbations. Recent methods attempt to implicitly model rotational equivariance and invariance, preserving directional information and propagating it into deep semantic spaces. Yet, they often fall short of fully exploiting the multiscale directional nature of point clouds to enhance feature representations. To address this, we propose the Direction-Perceptive Vector Network (DiPVNet). At its core is an atomic dot-product operator that simultaneously encodes directional selectivity and rotation invariance--endowing the network with both rotational symmetry modeling and adaptive directional perception. At the local level, we introduce a Learnable Local Dot-Product (L2DP) Operator, which enables interactions between a center point and its neighbors to adaptively capture the non-uniform local structures of point clouds. At the global level, we leverage generalized harmonic analysis to prove that the dot-product between point clouds and spherical sampling vectors is equivalent to a direction-aware spherical Fourier transform (DASFT). This leads to the construction of a global directional response spectrum for modeling holistic directional structures. We rigorously prove the rotation invariance of both operators. Extensive experiments on challenging scenarios involving noise and large-angle rotations demonstrate that DiPVNet achieves state-of-the-art performance on point cloud classification and segmentation tasks. Our code is available at https://github.com/wxszreal0/DiPVNet.
>
---
#### [new 054] Generalized-Scale Object Counting with Gradual Query Aggregation
- **分类: cs.CV**

- **简介: 该论文提出GECO2，用于少样本目标计数与检测任务，解决多尺度与密集小目标检测难题。通过渐进式查询聚合，实现跨尺度高分辨率特征表示，提升精度与速度，降低内存开销。**

- **链接: []()**

> **作者:** Jer Pelhan; Alan Lukezic; Matej Kristan
>
> **备注:** Accepted to AAAI2026, code: https://github.com/jerpelhan/GECO2/
>
> **摘要:** Few-shot detection-based counters estimate the number of instances in the image specified only by a few test-time exemplars. A common approach to localize objects across multiple sizes is to merge backbone features of different resolutions. Furthermore, to enable small object detection in densely populated regions, the input image is commonly upsampled and tiling is applied to cope with the increased computational and memory requirements. Because of these ad-hoc solutions, existing counters struggle with images containing diverse-sized objects and densely populated regions of small objects. We propose GECO2, an end-to-end few-shot counting and detection method that explicitly addresses the object scale issues. A new dense query representation gradually aggregates exemplar-specific feature information across scales that leads to high-resolution dense queries that enable detection of large as well as small objects. GECO2 surpasses state-of-the-art few-shot counters in counting as well as detection accuracy by 10% while running 3x times faster at smaller GPU memory footprint.
>
---
#### [new 055] Text-based Aerial-Ground Person Retrieval
- **分类: cs.CV; cs.AI**

- **简介: 论文提出文本引导的空地行人检索（TAG-PR）任务，解决高空与地面视角差异导致的跨视角检索难题。构建TAG-PEDES数据集，设计TAG-CLIP框架，通过分层专家模块与视角解耦策略提升跨模态对齐效果。**

- **链接: []()**

> **作者:** Xinyu Zhou; Yu Wu; Jiayao Ma; Wenhao Wang; Min Cao; Mang Ye
>
> **摘要:** This work introduces Text-based Aerial-Ground Person Retrieval (TAG-PR), which aims to retrieve person images from heterogeneous aerial and ground views with textual descriptions. Unlike traditional Text-based Person Retrieval (T-PR), which focuses solely on ground-view images, TAG-PR introduces greater practical significance and presents unique challenges due to the large viewpoint discrepancy across images. To support this task, we contribute: (1) TAG-PEDES dataset, constructed from public benchmarks with automatically generated textual descriptions, enhanced by a diversified text generation paradigm to ensure robustness under view heterogeneity; and (2) TAG-CLIP, a novel retrieval framework that addresses view heterogeneity through a hierarchically-routed mixture of experts module to learn view-specific and view-agnostic features and a viewpoint decoupling strategy to decouple view-specific features for better cross-modal alignment. We evaluate the effectiveness of TAG-CLIP on both the proposed TAG-PEDES dataset and existing T-PR benchmarks. The dataset and code are available at https://github.com/Flame-Chasers/TAG-PR.
>
---
#### [new 056] CSF-Net: Context-Semantic Fusion Network for Large Mask Inpainting
- **分类: cs.CV**

- **简介: 该论文提出CSF-Net，用于大掩码图像修复任务，解决缺失区域语义不一致与结构失真问题。通过预训练的Amodal Completion模型生成语义先验，利用Transformer融合上下文与语义信息，提升修复质量，可无缝集成至现有模型。**

- **链接: []()**

> **作者:** Chae-Yeon Heo; Yeong-Jun Cho
>
> **备注:** 8 pages, 5 figures, Accepted to WACV 2026 (to appear)
>
> **摘要:** In this paper, we propose a semantic-guided framework to address the challenging problem of large-mask image inpainting, where essential visual content is missing and contextual cues are limited. To compensate for the limited context, we leverage a pretrained Amodal Completion (AC) model to generate structure-aware candidates that serve as semantic priors for the missing regions. We introduce Context-Semantic Fusion Network (CSF-Net), a transformer-based fusion framework that fuses these candidates with contextual features to produce a semantic guidance image for image inpainting. This guidance improves inpainting quality by promoting structural accuracy and semantic consistency. CSF-Net can be seamlessly integrated into existing inpainting models without architectural changes and consistently enhances performance across diverse masking conditions. Extensive experiments on the Places365 and COCOA datasets demonstrate that CSF-Net effectively reduces object hallucination while enhancing visual realism and semantic alignment. The code for CSF-Net is available at https://github.com/chaeyeonheo/CSF-Net.
>
---
#### [new 057] DiffRegCD: Integrated Registration and Change Detection with Diffusion Features
- **分类: cs.CV; cs.AI**

- **简介: DiffRegCD提出一种联合图像配准与变化检测的统一框架，利用扩散模型特征将配准转化为高斯平滑分类任务，解决大位移下传统方法失效问题，无需伪标签即可实现亚像素精度的双任务优化。**

- **链接: []()**

> **作者:** Seyedehnanita Madani; Rama Chellappa; Vishal M. Patel
>
> **备注:** 10 pages, 8 figures. Accepted to WACV 2026
>
> **摘要:** Change detection (CD) is fundamental to computer vision and remote sensing, supporting applications in environmental monitoring, disaster response, and urban development. Most CD models assume co-registered inputs, yet real-world imagery often exhibits parallax, viewpoint shifts, and long temporal gaps that cause severe misalignment. Traditional two stage methods that first register and then detect, as well as recent joint frameworks (e.g., BiFA, ChangeRD), still struggle under large displacements, relying on regression only flow, global homographies, or synthetic perturbations. We present DiffRegCD, an integrated framework that unifies dense registration and change detection in a single model. DiffRegCD reformulates correspondence estimation as a Gaussian smoothed classification task, achieving sub-pixel accuracy and stable training. It leverages frozen multi-scale features from a pretrained denoising diffusion model, ensuring robustness to illumination and viewpoint variation. Supervision is provided through controlled affine perturbations applied to standard CD datasets, yielding paired ground truth for both flow and change detection without pseudo labels. Extensive experiments on aerial (LEVIR-CD, DSIFN-CD, WHU-CD, SYSU-CD) and ground level (VL-CMU-CD) datasets show that DiffRegCD consistently surpasses recent baselines and remains reliable under wide temporal and geometric variation, establishing diffusion features and classification based correspondence as a strong foundation for unified change detection.
>
---
#### [new 058] Is It Truly Necessary to Process and Fit Minutes-Long Reference Videos for Personalized Talking Face Generation?
- **分类: cs.CV**

- **简介: 该论文针对个性化说话人脸生成任务，发现长参考视频非必需，提出ISExplore策略，自动选取5秒高信息量视频片段，提升处理与训练速度5倍以上，同时保持高保真输出。**

- **链接: []()**

> **作者:** Rui-Qing Sun; Ang Li; Zhijing Wu; Tian Lan; Qianyu Lu; Xingshan Yao; Chen Xu; Xian-Ling Mao
>
> **摘要:** Talking Face Generation (TFG) aims to produce realistic and dynamic talking portraits, with broad applications in fields such as digital education, film and television production, e-commerce live streaming, and other related areas. Currently, TFG methods based on Neural Radiated Field (NeRF) or 3D Gaussian sputtering (3DGS) are received widespread attention. They learn and store personalized features from reference videos of each target individual to generate realistic speaking videos. To ensure models can capture sufficient 3D information and successfully learns the lip-audio mapping, previous studies usually require meticulous processing and fitting several minutes of reference video, which always takes hours. The computational burden of processing and fitting long reference videos severely limits the practical application value of these methods.However, is it really necessary to fit such minutes of reference video? Our exploratory case studies show that using some informative reference video segments of just a few seconds can achieve performance comparable to or even better than the full reference video. This indicates that video informative quality is much more important than its length. Inspired by this observation, we propose the ISExplore (short for Informative Segment Explore), a simple-yet-effective segment selection strategy that automatically identifies the informative 5-second reference video segment based on three key data quality dimensions: audio feature diversity, lip movement amplitude, and number of camera views. Extensive experiments demonstrate that our approach increases data processing and training speed by more than 5x for NeRF and 3DGS methods, while maintaining high-fidelity output. Project resources are available at xx.
>
---
#### [new 059] A Circular Argument : Does RoPE need to be Equivariant for Vision?
- **分类: cs.CV; cs.AI**

- **简介: 该论文质疑RoPE在视觉任务中依赖等变性的必要性，提出非等变的Spherical RoPE，实证表明其性能不逊甚至优于传统等变方法，挑战了相对位置编码在视觉中至关重要的主流观点。**

- **链接: []()**

> **作者:** Chase van de Geijn; Timo Lüddecke; Polina Turishcheva; Alexander S. Ecker
>
> **摘要:** Rotary Positional Encodings (RoPE) have emerged as a highly effective technique for one-dimensional sequences in Natural Language Processing spurring recent progress towards generalizing RoPE to higher-dimensional data such as images and videos. The success of RoPE has been thought to be due to its positional equivariance, i.e. its status as a relative positional encoding. In this paper, we mathematically show RoPE to be one of the most general solutions for equivariant positional embedding in one-dimensional data. Moreover, we show Mixed RoPE to be the analogously general solution for M-dimensional data, if we require commutative generators -- a property necessary for RoPE's equivariance. However, we question whether strict equivariance plays a large role in RoPE's performance. We propose Spherical RoPE, a method analogous to Mixed RoPE, but assumes non-commutative generators. Empirically, we find Spherical RoPE to have the equivalent or better learning behavior compared to its equivariant analogues. This suggests that relative positional embeddings are not as important as is commonly believed, at least within computer vision. We expect this discovery to facilitate future work in positional encodings for vision that can be faster and generalize better by removing the preconception that they must be relative.
>
---
#### [new 060] KPLM-STA: Physically-Accurate Shadow Synthesis for Human Relighting via Keypoint-Based Light Modeling
- **分类: cs.CV**

- **简介: 该论文针对人体重光照中阴影几何不准确与真实感不足的问题，提出KPLM-STA框架，通过关键点线性模型与阴影三角算法，实现物理精确的动态阴影合成，显著提升复杂姿态下的阴影 realism 与几何精度。**

- **链接: []()**

> **作者:** Xinhui Yin; Qifei Li; Yilin Guo; Hongxia Xie; Xiaoli Zhang
>
> **摘要:** Image composition aims to seamlessly integrate a foreground object into a background, where generating realistic and geometrically accurate shadows remains a persistent challenge. While recent diffusion-based methods have outperformed GAN-based approaches, existing techniques, such as the diffusion-based relighting framework IC-Light, still fall short in producing shadows with both high appearance realism and geometric precision, especially in composite images. To address these limitations, we propose a novel shadow generation framework based on a Keypoints Linear Model (KPLM) and a Shadow Triangle Algorithm (STA). KPLM models articulated human bodies using nine keypoints and one bounding block, enabling physically plausible shadow projection and dynamic shading across joints, thereby enhancing visual realism. STA further improves geometric accuracy by computing shadow angles, lengths, and spatial positions through explicit geometric formulations. Extensive experiments demonstrate that our method achieves state-of-the-art performance on shadow realism benchmarks, particularly under complex human poses, and generalizes effectively to multi-directional relighting scenarios such as those supported by IC-Light.
>
---
#### [new 061] UniVA: Universal Video Agent towards Open-Source Next-Generation Video Generalist
- **分类: cs.CV**

- **简介: 论文提出UniVA，一个开源多智能体视频通用框架，解决专用模型难以协同完成复杂视频任务的问题，通过计划-执行双智能体架构整合理解、分割、编辑与生成，支持交互式、可追溯的多步视频工作流。**

- **链接: []()**

> **作者:** Zhengyang Liang; Daoan Zhang; Huichi Zhou; Rui Huang; Bobo Li; Yuechen Zhang; Shengqiong Wu; Xiaohan Wang; Jiebo Luo; Lizi Liao; Hao Fei
>
> **备注:** Technical Report. 24 figures, 37 pages. Website: https://univa.online/
>
> **摘要:** While specialized AI models excel at isolated video tasks like generation or understanding, real-world applications demand complex, iterative workflows that combine these capabilities. To bridge this gap, we introduce UniVA, an open-source, omni-capable multi-agent framework for next-generation video generalists that unifies video understanding, segmentation, editing, and generation into cohesive workflows. UniVA employs a Plan-and-Act dual-agent architecture that drives a highly automated and proactive workflow: a planner agent interprets user intentions and decomposes them into structured video-processing steps, while executor agents execute these through modular, MCP-based tool servers (for analysis, generation, editing, tracking, etc.). Through a hierarchical multi-level memory (global knowledge, task context, and user-specific preferences), UniVA sustains long-horizon reasoning, contextual continuity, and inter-agent communication, enabling interactive and self-reflective video creation with full traceability. This design enables iterative and any-conditioned video workflows (e.g., text/image/video-conditioned generation $\rightarrow$ multi-round editing $\rightarrow$ object segmentation $\rightarrow$ compositional synthesis) that were previously cumbersome to achieve with single-purpose models or monolithic video-language models. We also introduce UniVA-Bench, a benchmark suite of multi-step video tasks spanning understanding, editing, segmentation, and generation, to rigorously evaluate such agentic video systems. Both UniVA and UniVA-Bench are fully open-sourced, aiming to catalyze research on interactive, agentic, and general-purpose video intelligence for the next generation of multimodal AI systems. (https://univa.online/)
>
---
#### [new 062] Large Sign Language Models: Toward 3D American Sign Language Translation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Large Sign Language Models（LSLM），首次将大语言模型用于3D美式手语翻译，解决传统2D视频识别缺乏深度信息的问题，通过3D手势特征直接映射文本，并支持指令引导翻译，推动多模态语言理解与无障碍通信。**

- **链接: []()**

> **作者:** Sen Zhang; Xiaoxiao He; Di Liu; Zhaoyang Xia; Mingyu Zhao; Chaowei Tan; Vivian Li; Bo Liu; Dimitris N. Metaxas; Mubbasir Kapadia
>
> **摘要:** We present Large Sign Language Models (LSLM), a novel framework for translating 3D American Sign Language (ASL) by leveraging Large Language Models (LLMs) as the backbone, which can benefit hearing-impaired individuals' virtual communication. Unlike existing sign language recognition methods that rely on 2D video, our approach directly utilizes 3D sign language data to capture rich spatial, gestural, and depth information in 3D scenes. This enables more accurate and resilient translation, enhancing digital communication accessibility for the hearing-impaired community. Beyond the task of ASL translation, our work explores the integration of complex, embodied multimodal languages into the processing capabilities of LLMs, moving beyond purely text-based inputs to broaden their understanding of human communication. We investigate both direct translation from 3D gesture features to text and an instruction-guided setting where translations can be modulated by external prompts, offering greater flexibility. This work provides a foundational step toward inclusive, multimodal intelligent systems capable of understanding diverse forms of language.
>
---
#### [new 063] Remodeling Semantic Relationships in Vision-Language Fine-Tuning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言多模态对齐任务，旨在解决现有方法忽略图像中语义关系导致性能不足的问题。通过多级视觉特征提取、语义分组与可继承交叉注意力，精准去除冗余关系，提升视觉与语言融合效果。**

- **链接: []()**

> **作者:** Xiangyang Wu; Liu Liu; Baosheng Yu; Jiayan Qiu; Zhenwei Shi
>
> **摘要:** Vision-language fine-tuning has emerged as an efficient paradigm for constructing multimodal foundation models. While textual context often highlights semantic relationships within an image, existing fine-tuning methods typically overlook this information when aligning vision and language, thus leading to suboptimal performance. Toward solving this problem, we propose a method that can improve multimodal alignment and fusion based on both semantics and relationships.Specifically, we first extract multilevel semantic features from different vision encoder to capture more visual cues of the relationships. Then, we learn to project the vision features to group related semantics, among which are more likely to have relationships. Finally, we fuse the visual features with the textual by using inheritable cross-attention, where we globally remove the redundant visual relationships by discarding visual-language feature pairs with low correlation. We evaluate our proposed method on eight foundation models and two downstream tasks, visual question answering and image captioning, and show that it outperforms all existing methods.
>
---
#### [new 064] Cancer-Net PCa-MultiSeg: Multimodal Enhancement of Prostate Cancer Lesion Segmentation Using Synthetic Correlated Diffusion Imaging
- **分类: cs.CV**

- **简介: 该论文针对前列腺癌病灶分割性能低的问题，提出利用合成相关扩散成像（CDIˢ）增强多模态MRI数据，无需额外扫描时间，在6种主流模型中94%配置下提升分割精度，实现临床即用型优化。**

- **链接: []()**

> **作者:** Jarett Dewbury; Chi-en Amy Tai; Alexander Wong
>
> **备注:** Accepted at ML4H 2025 Findings
>
> **摘要:** Current deep learning approaches for prostate cancer lesion segmentation achieve limited performance, with Dice scores of 0.32 or lower in large patient cohorts. To address this limitation, we investigate synthetic correlated diffusion imaging (CDI$^s$) as an enhancement to standard diffusion-based protocols. We conduct a comprehensive evaluation across six state-of-the-art segmentation architectures using 200 patients with co-registered CDI$^s$, diffusion-weighted imaging (DWI) and apparent diffusion coefficient (ADC) sequences. We demonstrate that CDI$^s$ integration reliably enhances or preserves segmentation performance in 94% of evaluated configurations, with individual architectures achieving up to 72.5% statistically significant relative improvement over baseline modalities. CDI$^s$ + DWI emerges as the safest enhancement pathway, achieving significant improvements in half of evaluated architectures with zero instances of degradation. Since CDI$^s$ derives from existing DWI acquisitions without requiring additional scan time or architectural modifications, it enables immediate deployment in clinical workflows. Our results establish validated integration pathways for CDI$^s$ as a practical drop-in enhancement for PCa lesion segmentation tasks across diverse deep learning architectures.
>
---
#### [new 065] CloudMamba: Grouped Selective State Spaces for Point Cloud Analysis
- **分类: cs.CV**

- **简介: CloudMamba针对点云分析中的序列化不佳、几何感知不足和S6过拟合问题，提出序列展开与合并、chainedMamba和分组S6（GS6），在保持线性复杂度下提升建模能力，实现SOTA性能。**

- **链接: []()**

> **作者:** Kanglin Qu; Pan Gao; Qun Dai; Zhanzhi Ye; Rui Ye; Yuanhao Sun
>
> **备注:** Accepted by AAAI '26
>
> **摘要:** Due to the long-range modeling ability and linear complexity property, Mamba has attracted considerable attention in point cloud analysis. Despite some interesting progress, related work still suffers from imperfect point cloud serialization, insufficient high-level geometric perception, and overfitting of the selective state space model (S6) at the core of Mamba. To this end, we resort to an SSM-based point cloud network termed CloudMamba to address the above challenges. Specifically, we propose sequence expanding and sequence merging, where the former serializes points along each axis separately and the latter serves to fuse the corresponding higher-order features causally inferred from different sequences, enabling unordered point sets to adapt more stably to the causal nature of Mamba without parameters. Meanwhile, we design chainedMamba that chains the forward and backward processes in the parallel bidirectional Mamba, capturing high-level geometric information during scanning. In addition, we propose a grouped selective state space model (GS6) via parameter sharing on S6, alleviating the overfitting problem caused by the computational mode in S6. Experiments on various point cloud tasks validate CloudMamba's ability to achieve state-of-the-art results with significantly less complexity.
>
---
#### [new 066] Generalizable Blood Cell Detection via Unified Dataset and Faster R-CNN
- **分类: cs.CV; cs.LG**

- **简介: 该论文面向外周血细胞检测任务，解决数据稀缺与异构问题，通过整合四个公开数据集构建统一数据集，并采用Faster R-CNN（ResNet-50-FPN）对比训练策略，验证迁移学习显著提升检测性能与收敛速度。**

- **链接: []()**

> **作者:** Siddharth Sahay
>
> **备注:** 7 pages, 7 tables, 3 figures, 2 algorithms, Submitted for review at Next-Gen Quantum and Advanced Computing: Algorithms, Security, and Beyond (NQComp-2026)
>
> **摘要:** This paper presents a comprehensive methodology and comparative performance analysis for the automated classification and object detection of peripheral blood cells (PBCs) in microscopic images. Addressing the critical challenge of data scarcity and heterogeneity, robust data pipeline was first developed to standardize and merge four public datasets (PBC, BCCD, Chula, Sickle Cell) into a unified resource. Then employed a state-of-the-art Faster R-CNN object detection framework, leveraging a ResNet-50-FPN backbone. Comparative training rigorously evaluated a randomly initialized baseline model (Regimen 1) against a Transfer Learning Regimen (Regimen 2), initialized with weights pre-trained on the Microsoft COCO dataset. The results demonstrate that the Transfer Learning approach achieved significantly faster convergence and superior stability, culminating in a final validation loss of 0.08666, a substantial improvement over the baseline. This validated methodology establishes a robust foundation for building high-accuracy, deployable systems for automated hematological diagnosis.
>
---
#### [new 067] Multi-modal Deepfake Detection and Localization with FPN-Transformer
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出FPN-Transformer框架，用于多模态深度伪造检测与定位，解决单模态方法难以捕捉跨模态关联与精确定位伪造片段的问题，结合WavLM和CLIP提取特征，实现帧级定位，性能优于现有方法。**

- **链接: []()**

> **作者:** Chende Zheng; Ruiqi Suo; Zhoulin Ji; Jingyi Deng; Fangbin Yi; Chenhao Lin; Chao Shen
>
> **摘要:** The rapid advancement of generative adversarial networks (GANs) and diffusion models has enabled the creation of highly realistic deepfake content, posing significant threats to digital trust across audio-visual domains. While unimodal detection methods have shown progress in identifying synthetic media, their inability to leverage cross-modal correlations and precisely localize forged segments limits their practicality against sophisticated, fine-grained manipulations. To address this, we introduce a multi-modal deepfake detection and localization framework based on a Feature Pyramid-Transformer (FPN-Transformer), addressing critical gaps in cross-modal generalization and temporal boundary regression. The proposed approach utilizes pre-trained self-supervised models (WavLM for audio, CLIP for video) to extract hierarchical temporal features. A multi-scale feature pyramid is constructed through R-TLM blocks with localized attention mechanisms, enabling joint analysis of cross-context temporal dependencies. The dual-branch prediction head simultaneously predicts forgery probabilities and refines temporal offsets of manipulated segments, achieving frame-level localization precision. We evaluate our approach on the test set of the IJCAI'25 DDL-AV benchmark, showing a good performance with a final score of 0.7535 for cross-modal deepfake detection and localization in challenging environments. Experimental results confirm the effectiveness of our approach and provide a novel way for generalized deepfake detection. Our code is available at https://github.com/Zig-HS/MM-DDL
>
---
#### [new 068] Perceptual Quality Assessment of 3D Gaussian Splatting: A Subjective Dataset and Prediction Metric
- **分类: cs.CV**

- **简介: 该论文面向3D高斯溅射（3DGS）的感知质量评估任务，解决其失真下质量无系统评估的问题，构建首个主观数据集3DGS-QA，并提出无需参考的无监督预测模型，直接从高斯原始数据中提取感知特征，实现更准确的质量评估。**

- **链接: []()**

> **作者:** Zhaolin Wan; Yining Diao; Jingqi Xu; Hao Wang; Zhiyang Li; Xiaopeng Fan; Wangmeng Zuo; Debin Zhao
>
> **摘要:** With the rapid advancement of 3D visualization, 3D Gaussian Splatting (3DGS) has emerged as a leading technique for real-time, high-fidelity rendering. While prior research has emphasized algorithmic performance and visual fidelity, the perceptual quality of 3DGS-rendered content, especially under varying reconstruction conditions, remains largely underexplored. In practice, factors such as viewpoint sparsity, limited training iterations, point downsampling, noise, and color distortions can significantly degrade visual quality, yet their perceptual impact has not been systematically studied. To bridge this gap, we present 3DGS-QA, the first subjective quality assessment dataset for 3DGS. It comprises 225 degraded reconstructions across 15 object types, enabling a controlled investigation of common distortion factors. Based on this dataset, we introduce a no-reference quality prediction model that directly operates on native 3D Gaussian primitives, without requiring rendered images or ground-truth references. Our model extracts spatial and photometric cues from the Gaussian representation to estimate perceived quality in a structure-aware manner. We further benchmark existing quality assessment methods, spanning both traditional and learning-based approaches. Experimental results show that our method consistently achieves superior performance, highlighting its robustness and effectiveness for 3DGS content evaluation. The dataset and code are made publicly available at https://github.com/diaoyn/3DGSQA to facilitate future research in 3DGS quality assessment.
>
---
#### [new 069] DI3CL: Contrastive Learning With Dynamic Instances and Contour Consistency for SAR Land-Cover Classification Foundation Model
- **分类: cs.CV**

- **简介: 该论文提出DI3CL框架，面向SAR地物分类，解决监督学习依赖标签数据的问题。通过动态实例与轮廓一致性对比学习，结合大规模SARSense数据集，构建通用基础模型，显著提升多任务泛化能力。**

- **链接: []()**

> **作者:** Zhongle Ren; Hui Ding; Kai Wang; Biao Hou; Xingyu Luo; Weibin Li; Licheng Jiao
>
> **备注:** 18 pages, 10 figures
>
> **摘要:** Although significant advances have been achieved in SAR land-cover classification, recent methods remain predominantly focused on supervised learning, which relies heavily on extensive labeled datasets. This dependency not only limits scalability and generalization but also restricts adaptability to diverse application scenarios. In this paper, a general-purpose foundation model for SAR land-cover classification is developed, serving as a robust cornerstone to accelerate the development and deployment of various downstream models. Specifically, a Dynamic Instance and Contour Consistency Contrastive Learning (DI3CL) pre-training framework is presented, which incorporates a Dynamic Instance (DI) module and a Contour Consistency (CC) module. DI module enhances global contextual awareness by enforcing local consistency across different views of the same region. CC module leverages shallow feature maps to guide the model to focus on the geometric contours of SAR land-cover objects, thereby improving structural discrimination. Additionally, to enhance robustness and generalization during pre-training, a large-scale and diverse dataset named SARSense, comprising 460,532 SAR images, is constructed to enable the model to capture comprehensive and representative features. To evaluate the generalization capability of our foundation model, we conducted extensive experiments across a variety of SAR land-cover classification tasks, including SAR land-cover mapping, water body detection, and road extraction. The results consistently demonstrate that the proposed DI3CL outperforms existing methods. Our code and pre-trained weights are publicly available at: https://github.com/SARpre-train/DI3CL.
>
---
#### [new 070] Theoretical Analysis of Power-law Transformation on Images for Text Polarity Detection
- **分类: cs.CV**

- **简介: 该论文属于图像预处理任务，旨在理论分析幂律变换在文本极性检测中的作用，解决如何通过直方图统计判断文本与背景对比关系的问题，首次为经验现象提供了理论支撑。**

- **链接: []()**

> **作者:** Narendra Singh Yadav; Pavan Kumar Perepu
>
> **摘要:** Several computer vision applications like vehicle license plate recognition, captcha recognition, printed or handwriting character recognition from images etc., text polarity detection and binarization are the important preprocessing tasks. To analyze any image, it has to be converted to a simple binary image. This binarization process requires the knowledge of polarity of text in the images. Text polarity is defined as the contrast of text with respect to background. That means, text is darker than the background (dark text on bright background) or vice-versa. The binarization process uses this polarity information to convert the original colour or gray scale image into a binary image. In the literature, there is an intuitive approach based on power-law transformation on the original images. In this approach, the authors have illustrated an interesting phenomenon from the histogram statistics of the transformed images. Considering text and background as two classes, they have observed that maximum between-class variance between two classes is increasing (decreasing) for dark (bright) text on bright (dark) background. The corresponding empirical results have been presented. In this paper, we present a theoretical analysis of the above phenomenon.
>
---
#### [new 071] OTSNet: A Neurocognitive-Inspired Observation-Thinking-Spelling Pipeline for Scene Text Recognition
- **分类: cs.CV; cs.AI**

- **简介: OTSNet面向场景文本识别任务，解决视觉-语言解耦导致的跨模态错位与误差传播问题，提出仿人认知的“观察-思考-拼写”三阶段架构，通过双注意力编码、位置语义融合与多模态校验，显著提升不规则文本识别精度。**

- **链接: []()**

> **作者:** Lixu Sun; Nurmemet Yolwas; Wushour Silamu
>
> **摘要:** Scene Text Recognition (STR) remains challenging due to real-world complexities, where decoupled visual-linguistic optimization in existing frameworks amplifies error propagation through cross-modal misalignment. Visual encoders exhibit attention bias toward background distractors, while decoders suffer from spatial misalignment when parsing geometrically deformed text-collectively degrading recognition accuracy for irregular patterns. Inspired by the hierarchical cognitive processes in human visual perception, we propose OTSNet, a novel three-stage network embodying a neurocognitive-inspired Observation-Thinking-Spelling pipeline for unified STR modeling. The architecture comprises three core components: (1) a Dual Attention Macaron Encoder (DAME) that refines visual features through differential attention maps to suppress irrelevant regions and enhance discriminative focus; (2) a Position-Aware Module (PAM) and Semantic Quantizer (SQ) that jointly integrate spatial context with glyph-level semantic abstraction via adaptive sampling; and (3) a Multi-Modal Collaborative Verifier (MMCV) that enforces self-correction through cross-modal fusion of visual, semantic, and character-level features. Extensive experiments demonstrate that OTSNet achieves state-of-the-art performance, attaining 83.5% average accuracy on the challenging Union14M-L benchmark and 79.1% on the heavily occluded OST dataset-establishing new records across 9 out of 14 evaluation scenarios.
>
---
#### [new 072] Predicting Coronary Artery Calcium Severity based on Non-Contrast Cardiac CT images using Deep Learning
- **分类: cs.CV**

- **简介: 该论文利用深度学习CNN模型，基于非增强心脏CT图像自动分类冠状动脉钙化严重程度为六类，解决人工评分耗时问题，实现高准确率（96.5%）与强一致性（Kappa=0.962），验证了自动化分层评估的可行性。**

- **链接: []()**

> **作者:** Lachlan Nguyen; Aidan Cousins; Arcot Sowmya; Hugh Dixson; Sonit Singh
>
> **备注:** 6 pages
>
> **摘要:** Cardiovascular disease causes high rates of mortality worldwide. Coronary artery calcium (CAC) scoring is a powerful tool to stratify the risk of atherosclerotic cardiovascular disease. Current scoring practices require time-intensive semiautomatic analysis of cardiac computed tomography by radiologists and trained radiographers. The purpose of this study is to develop a deep learning convolutional neural networks (CNN) model to classify the calcium score in cardiac, non-contrast computed tomography images into one of six clinical categories. A total of 68 patient scans were retrospectively obtained together with their respective reported semiautomatic calcium score using an ECG-gated GE Discovery 570 Cardiac SPECT/CT camera. The dataset was divided into training, validation and test sets. Using the semiautomatic CAC score as the reference label, the model demonstrated high performance on a six-class CAC scoring categorisation task. Of the scans analysed, the model misclassified 32 cases, tending towards overestimating the CAC in 26 out of 32 misclassifications. Overall, the model showed high agreement (Cohen's kappa of 0.962), an overall accuracy of 96.5% and high generalisability. The results suggest that the model outputs were accurate and consistent with current semiautomatic practice, with good generalisability to test data. The model demonstrates the viability of a CNN model to stratify the calcium score into an expanded set of six clinical categories.
>
---
#### [new 073] Contrastive Integrated Gradients: A Feature Attribution-Based Method for Explaining Whole Slide Image Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对全切片图像（WSI）分类的可解释性问题，提出对比积分梯度（CIG）方法，通过对比类间对数几率差异，精准定位肿瘤判别区域，并设计MIL-AIC/SIC指标评估 attributions 质量，在三类癌症数据上验证了其优越性。**

- **链接: []()**

> **作者:** Anh Mai Vu; Tuan L. Vo; Ngoc Lam Quang Bui; Nam Nguyen Le Binh; Akash Awasthi; Huy Quoc Vo; Thanh-Huy Nguyen; Zhu Han; Chandra Mohan; Hien Van Nguyen
>
> **备注:** 10 pages
>
> **摘要:** Interpretability is essential in Whole Slide Image (WSI) analysis for computational pathology, where understanding model predictions helps build trust in AI-assisted diagnostics. While Integrated Gradients (IG) and related attribution methods have shown promise, applying them directly to WSIs introduces challenges due to their high-resolution nature. These methods capture model decision patterns but may overlook class-discriminative signals that are crucial for distinguishing between tumor subtypes. In this work, we introduce Contrastive Integrated Gradients (CIG), a novel attribution method that enhances interpretability by computing contrastive gradients in logit space. First, CIG highlights class-discriminative regions by comparing feature importance relative to a reference class, offering sharper differentiation between tumor and non-tumor areas. Second, CIG satisfies the axioms of integrated attribution, ensuring consistency and theoretical soundness. Third, we propose two attribution quality metrics, MIL-AIC and MIL-SIC, which measure how predictive information and model confidence evolve with access to salient regions, particularly under weak supervision. We validate CIG across three datasets spanning distinct cancer types: CAMELYON16 (breast cancer metastasis in lymph nodes), TCGA-RCC (renal cell carcinoma), and TCGA-Lung (lung cancer). Experimental results demonstrate that CIG yields more informative attributions both quantitatively, using MIL-AIC and MIL-SIC, and qualitatively, through visualizations that align closely with ground truth tumor regions, underscoring its potential for interpretable and trustworthy WSI-based diagnostics
>
---
#### [new 074] RAPTR: Radar-based 3D Pose Estimation using Transformer
- **分类: cs.CV; cs.AI; eess.SP**

- **简介: RAPTR提出一种基于雷达的弱监督3D人体姿态估计方法，仅需3D边界框和2D关键点标签，通过双阶段Transformer解码器缓解深度歧义，显著提升估计精度。**

- **链接: []()**

> **作者:** Sorachi Kato; Ryoma Yataka; Pu Perry Wang; Pedro Miraldo; Takuya Fujihashi; Petros Boufounos
>
> **备注:** 26 pages, Accepted to NeurIPS 2025
>
> **摘要:** Radar-based indoor 3D human pose estimation typically relied on fine-grained 3D keypoint labels, which are costly to obtain especially in complex indoor settings involving clutter, occlusions, or multiple people. In this paper, we propose \textbf{RAPTR} (RAdar Pose esTimation using tRansformer) under weak supervision, using only 3D BBox and 2D keypoint labels which are considerably easier and more scalable to collect. Our RAPTR is characterized by a two-stage pose decoder architecture with a pseudo-3D deformable attention to enhance (pose/joint) queries with multi-view radar features: a pose decoder estimates initial 3D poses with a 3D template loss designed to utilize the 3D BBox labels and mitigate depth ambiguities; and a joint decoder refines the initial poses with 2D keypoint labels and a 3D gravity loss. Evaluated on two indoor radar datasets, RAPTR outperforms existing methods, reducing joint position error by $34.3\%$ on HIBER and $76.9\%$ on MMVR. Our implementation is available at https://github.com/merlresearch/radar-pose-transformer.
>
---
#### [new 075] WEDepth: Efficient Adaptation of World Knowledge for Monocular Depth Estimation
- **分类: cs.CV**

- **简介: WEDepth针对单目深度估计任务，提出一种无需修改预训练模型结构的高效适配方法，通过多级注入世界先验知识，提升深度预测精度，并实现优异的零样本迁移能力。**

- **链接: []()**

> **作者:** Gongshu Wang; Zhirui Wang; Kan Yang
>
> **摘要:** Monocular depth estimation (MDE) has widely applicable but remains highly challenging due to the inherently ill-posed nature of reconstructing 3D scenes from single 2D images. Modern Vision Foundation Models (VFMs), pre-trained on large-scale diverse datasets, exhibit remarkable world understanding capabilities that benefit for various vision tasks. Recent studies have demonstrated significant improvements in MDE through fine-tuning these VFMs. Inspired by these developments, we propose WEDepth, a novel approach that adapts VFMs for MDE without modi-fying their structures and pretrained weights, while effec-tively eliciting and leveraging their inherent priors. Our method employs the VFM as a multi-level feature en-hancer, systematically injecting prior knowledge at differ-ent representation levels. Experiments on NYU-Depth v2 and KITTI datasets show that WEDepth establishes new state-of-the-art (SOTA) performance, achieving competi-tive results compared to both diffusion-based approaches (which require multiple forward passes) and methods pre-trained on relative depth. Furthermore, we demonstrate our method exhibits strong zero-shot transfer capability across diverse scenarios.
>
---
#### [new 076] LandSegmenter: Towards a Flexible Foundation Model for Land Use and Land Cover Mapping
- **分类: cs.CV**

- **简介: 论文提出LandSegmenter，一种面向土地利用/覆盖（LULC）映射的灵活基础模型，解决传统模型模态与分类体系受限、标注数据稀缺问题。通过弱标签数据集LAS、跨模态适配器与文本编码器、置信度融合策略，实现高效零样本迁移。**

- **链接: []()**

> **作者:** Chenying Liu; Wei Huang; Xiao Xiang Zhu
>
> **摘要:** Land Use and Land Cover (LULC) mapping is a fundamental task in Earth Observation (EO). However, current LULC models are typically developed for a specific modality and a fixed class taxonomy, limiting their generability and broader applicability. Recent advances in foundation models (FMs) offer promising opportunities for building universal models. Yet, task-agnostic FMs often require fine-tuning for downstream applications, whereas task-specific FMs rely on massive amounts of labeled data for training, which is costly and impractical in the remote sensing (RS) domain. To address these challenges, we propose LandSegmenter, an LULC FM framework that resolves three-stage challenges at the input, model, and output levels. From the input side, to alleviate the heavy demand on labeled data for FM training, we introduce LAnd Segment (LAS), a large-scale, multi-modal, multi-source dataset built primarily with globally sampled weak labels from existing LULC products. LAS provides a scalable, cost-effective alternative to manual annotation, enabling large-scale FM training across diverse LULC domains. For model architecture, LandSegmenter integrates an RS-specific adapter for cross-modal feature extraction and a text encoder for semantic awareness enhancement. At the output stage, we introduce a class-wise confidence-guided fusion strategy to mitigate semantic omissions and further improve LandSegmenter's zero-shot performance. We evaluate LandSegmenter on six precisely annotated LULC datasets spanning diverse modalities and class taxonomies. Extensive transfer learning and zero-shot experiments demonstrate that LandSegmenter achieves competitive or superior performance, particularly in zero-shot settings when transferred to unseen datasets. These results highlight the efficacy of our proposed framework and the utility of weak supervision for building task-specific FMs.
>
---
#### [new 077] Hardware-Aware YOLO Compression for Low-Power Edge AI on STM32U5 for Weeds Detection in Digital Agriculture
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对农业中杂草检测的低功耗需求，将YOLOv8n模型压缩优化后部署于STM32U5微控制器，通过剪枝、量化和分辨率缩放实现高效边缘推理，仅耗能51.8mJ/次，支持实时精准除草。**

- **链接: []()**

> **作者:** Charalampos S. Kouzinopoulos; Yuri Manna
>
> **摘要:** Weeds significantly reduce crop yields worldwide and pose major challenges to sustainable agriculture. Traditional weed management methods, primarily relying on chemical herbicides, risk environmental contamination and lead to the emergence of herbicide-resistant species. Precision weeding, leveraging computer vision and machine learning methods, offers a promising eco-friendly alternative but is often limited by reliance on high-power computational platforms. This work presents an optimized, low-power edge AI system for weeds detection based on the YOLOv8n object detector deployed on the STM32U575ZI microcontroller. Several compression techniques are applied to the detection model, including structured pruning, integer quantization and input image resolution scaling in order to meet strict hardware constraints. The model is trained and evaluated on the CropAndWeed dataset with 74 plant species, achieving a balanced trade-off between detection accuracy and efficiency. Our system supports real-time, in-situ weeds detection with a minimal energy consumption of 51.8mJ per inference, enabling scalable deployment in power-constrained agricultural environments.
>
---
#### [new 078] StableMorph: High-Quality Face Morph Generation with Stable Diffusion
- **分类: cs.CV; cs.AI**

- **简介: StableMorph提出一种基于扩散模型的高保真人脸融合生成方法，解决现有方法模糊、失真、易检测的问题，生成逼真且能欺骗人脸识别系统的融合图像，提升生物识别安全评估与检测系统研发水平。**

- **链接: []()**

> **作者:** Wassim Kabbani; Kiran Raja; Raghavendra Ramachandra; Christoph Busch
>
> **摘要:** Face morphing attacks threaten the integrity of biometric identity systems by enabling multiple individuals to share a single identity. To develop and evaluate effective morphing attack detection (MAD) systems, we need access to high-quality, realistic morphed images that reflect the challenges posed in real-world scenarios. However, existing morph generation methods often produce images that are blurry, riddled with artifacts, or poorly constructed making them easy to detect and not representative of the most dangerous attacks. In this work, we introduce StableMorph, a novel approach that generates highly realistic, artifact-free morphed face images using modern diffusion-based image synthesis. Unlike prior methods, StableMorph produces full-head images with sharp details, avoids common visual flaws, and offers unmatched control over visual attributes. Through extensive evaluation, we show that StableMorph images not only rival or exceed the quality of genuine face images but also maintain a strong ability to fool face recognition systems posing a greater challenge to existing MAD solutions and setting a new standard for morph quality in research and operational testing. StableMorph improves the evaluation of biometric security by creating more realistic and effective attacks and supports the development of more robust detection systems.
>
---
#### [new 079] EAGLE: Episodic Appearance- and Geometry-aware Memory for Unified 2D-3D Visual Query Localization in Egocentric Vision
- **分类: cs.CV**

- **简介: EAGLE面向人称视觉中的2D-3D视觉查询定位任务，解决相机运动与外观变化带来的挑战。通过类鸟脑记忆机制，融合外观与几何记忆模块，实现精准轮廓分割与3D回投影，提升定位精度，达SOTA性能。**

- **链接: []()**

> **作者:** Yifei Cao; Yu Liu; Guolong Wang; Zhu Liu; Kai Wang; Xianjie Zhang; Jizhe Yu; Xun Tu
>
> **备注:** 13 Pages, AAAI2026
>
> **摘要:** Egocentric visual query localization is vital for embodied AI and VR/AR, yet remains challenging due to camera motion, viewpoint changes, and appearance variations. We present EAGLE, a novel framework that leverages episodic appearance- and geometry-aware memory to achieve unified 2D-3D visual query localization in egocentric vision. Inspired by avian memory consolidation, EAGLE synergistically integrates segmentation guided by an appearance-aware meta-learning memory (AMM), with tracking driven by a geometry-aware localization memory (GLM). This memory consolidation mechanism, through structured appearance and geometry memory banks, stores high-confidence retrieval samples, effectively supporting both long- and short-term modeling of target appearance variations. This enables precise contour delineation with robust spatial discrimination, leading to significantly improved retrieval accuracy. Furthermore, by integrating the VQL-2D output with a visual geometry grounded Transformer (VGGT), we achieve a efficient unification of 2D and 3D tasks, enabling rapid and accurate back-projection into 3D space. Our method achieves state-ofthe-art performance on the Ego4D-VQ benchmark.
>
---
#### [new 080] Revisiting MLLM Based Image Quality Assessment: Errors and Remedy
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型（MLLM）在图像质量评估（IQA）中的离散输出与连续评分不匹配问题，提出Q-Scorer框架，引入轻量回归模块与专用评分标记，有效提升评估精度与泛化能力。**

- **链接: []()**

> **作者:** Zhenchen Tang; Songlin Yang; Bo Peng; Zichuan Wang; Jing Dong
>
> **备注:** 13 pages
>
> **摘要:** The rapid progress of multi-modal large language models (MLLMs) has boosted the task of image quality assessment (IQA). However, a key challenge arises from the inherent mismatch between the discrete token outputs of MLLMs and the continuous nature of quality scores required by IQA tasks. This discrepancy significantly hinders the performance of MLLM-based IQA methods. Previous approaches that convert discrete token predictions into continuous scores often suffer from conversion errors. Moreover, the semantic confusion introduced by level tokens (e.g., ``good'') further constrains the performance of MLLMs on IQA tasks and degrades their original capabilities for related tasks. To tackle these problems, we provide a theoretical analysis of the errors inherent in previous approaches and, motivated by this analysis, propose a simple yet effective framework, Q-Scorer. This framework incorporates a lightweight regression module and IQA-specific score tokens into the MLLM pipeline. Extensive experiments demonstrate that Q-Scorer achieves state-of-the-art performance across multiple IQA benchmarks, generalizes well to mixed datasets, and further improves when combined with other methods.
>
---
#### [new 081] Compression then Matching: An Efficient Pre-training Paradigm for Multimodal Embedding
- **分类: cs.CV; cs.IR**

- **简介: 该论文提出CoMa框架，将视觉语言模型的预训练解耦为压缩理解与对比学习两阶段，解决传统方法效率低、数据依赖强的问题，在小数据下实现高效高质量多模态嵌入，刷新MMEB基准。**

- **链接: []()**

> **作者:** Da Li; Yuxiao Luo; Keping Bi; Jiafeng Guo; Wei Yuan; Biao Yang; Yan Wang; Fan Yang; Tingting Gao; Guorui Zhou
>
> **备注:** Multimodal Embedding
>
> **摘要:** Vision-language models advance multimodal representation learning by acquiring transferable semantic embeddings, thereby substantially enhancing performance across a range of vision-language tasks, including cross-modal retrieval, clustering, and classification. An effective embedding is expected to comprehensively preserve the semantic content of the input while simultaneously emphasizing features that are discriminative for downstream tasks. Recent approaches demonstrate that VLMs can be adapted into competitive embedding models via large-scale contrastive learning, enabling the simultaneous optimization of two complementary objectives. We argue that the two aforementioned objectives can be decoupled: a comprehensive understanding of the input facilitates the embedding model in achieving superior performance in downstream tasks via contrastive learning. In this paper, we propose CoMa, a compressed pre-training phase, which serves as a warm-up stage for contrastive learning. Experiments demonstrate that with only a small amount of pre-training data, we can transform a VLM into a competitive embedding model. CoMa achieves new state-of-the-art results among VLMs of comparable size on the MMEB, realizing optimization in both efficiency and effectiveness.
>
---
#### [new 082] Retrospective motion correction in MRI using disentangled embeddings
- **分类: cs.CV**

- **简介: 该论文提出一种基于解耦嵌入的层次化VQ-VAE模型，用于MRI retrospective运动校正，无需特定运动类型训练，即可泛化至未见运动模式，提升校正通用性。**

- **链接: []()**

> **作者:** Qi Wang; Veronika Ecker; Marcel Früh; Sergios Gatidis; Thomas Küstner
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Physiological motion can affect the diagnostic quality of magnetic resonance imaging (MRI). While various retrospective motion correction methods exist, many struggle to generalize across different motion types and body regions. In particular, machine learning (ML)-based corrections are often tailored to specific applications and datasets. We hypothesize that motion artifacts, though diverse, share underlying patterns that can be disentangled and exploited. To address this, we propose a hierarchical vector-quantized (VQ) variational auto-encoder that learns a disentangled embedding of motion-to-clean image features. A codebook is deployed to capture finite collection of motion patterns at multiple resolutions, enabling coarse-to-fine correction. An auto-regressive model is trained to learn the prior distribution of motion-free images and is used at inference to guide the correction process. Unlike conventional approaches, our method does not require artifact-specific training and can generalize to unseen motion patterns. We demonstrate the approach on simulated whole-body motion artifacts and observe robust correction across varying motion severity. Our results suggest that the model effectively disentangled physical motion of the simulated motion-effective scans, therefore, improving the generalizability of the ML-based MRI motion correction. Our work of disentangling the motion features shed a light on its potential application across anatomical regions and motion types.
>
---
#### [new 083] TrackStudio: An Integrated Toolkit for Markerless Tracking
- **分类: cs.CV; q-bio.QM**

- **简介: 论文提出TrackStudio，一个无需编程的集成式无标记运动追踪工具包，解决非专家使用复杂工具的门槛问题，通过模块化GUI整合开源工具，实现跨环境稳定追踪，并验证其在手部、面部等多部位的通用性。**

- **链接: []()**

> **作者:** Hristo Dimitrov; Giulia Dominijanni; Viktorija Pavalkyte; Tamar R. Makin
>
> **备注:** 26 pages, 5 main text figures, 5 supplementary figures
>
> **摘要:** Markerless motion tracking has advanced rapidly in the past 10 years and currently offers powerful opportunities for behavioural, clinical, and biomechanical research. While several specialised toolkits provide high performance for specific tasks, using existing tools still requires substantial technical expertise. There remains a gap in accessible, integrated solutions that deliver sufficient tracking for non-experts across diverse settings. TrackStudio was developed to address this gap by combining established open-source tools into a single, modular, GUI-based pipeline that works out of the box. It provides automatic 2D and 3D tracking, calibration, preprocessing, feature extraction, and visualisation without requiring any programming skills. We supply a user guide with practical advice for video acquisition, synchronisation, and setup, alongside documentation of common pitfalls and how to avoid them. To validate the toolkit, we tested its performance across three environments using either low-cost webcams or high-resolution cameras, including challenging conditions for body position, lightning, and space and obstructions. Across 76 participants, average inter-frame correlations exceeded 0.98 and average triangulation errors remained low (<13.6mm for hand tracking), demonstrating stable and consistent tracking. We further show that the same pipeline can be extended beyond hand tracking to other body and face regions. TrackStudio provides a practical, accessible route into markerless tracking for researchers or laypeople who need reliable performance without specialist expertise.
>
---
#### [new 084] Divide-and-Conquer Decoupled Network for Cross-Domain Few-Shot Segmentation
- **分类: cs.CV**

- **简介: 该论文针对跨域小样本分割任务，提出DCDNet网络，通过对抗-对比特征解耦与动态融合模块，分离类别与域特征，提升模型在新域与新类上的泛化与适应能力，实现SOTA性能。**

- **链接: []()**

> **作者:** Runmin Cong; Anpeng Wang; Bin Wan; Cong Zhang; Xiaofei Zhou; Wei Zhang
>
> **摘要:** Cross-domain few-shot segmentation (CD-FSS) aims to tackle the dual challenge of recognizing novel classes and adapting to unseen domains with limited annotations. However, encoder features often entangle domain-relevant and category-relevant information, limiting both generalization and rapid adaptation to new domains. To address this issue, we propose a Divide-and-Conquer Decoupled Network (DCDNet). In the training stage, to tackle feature entanglement that impedes cross-domain generalization and rapid adaptation, we propose the Adversarial-Contrastive Feature Decomposition (ACFD) module. It decouples backbone features into category-relevant private and domain-relevant shared representations via contrastive learning and adversarial learning. Then, to mitigate the potential degradation caused by the disentanglement, the Matrix-Guided Dynamic Fusion (MGDF) module adaptively integrates base, shared, and private features under spatial guidance, maintaining structural coherence. In addition, in the fine-tuning stage, to enhanced model generalization, the Cross-Adaptive Modulation (CAM) module is placed before the MGDF, where shared features guide private features via modulation ensuring effective integration of domain-relevant information. Extensive experiments on four challenging datasets show that DCDNet outperforms existing CD-FSS methods, setting a new state-of-the-art for cross-domain generalization and few-shot adaptation.
>
---
#### [new 085] SkelSplat: Robust Multi-view 3D Human Pose Estimation with Differentiable Gaussian Rendering
- **分类: cs.CV**

- **简介: SkelSplat提出一种基于可微高斯渲染的多视角3D人体姿态估计方法，无需3D标注，通过骨架高斯建模与一热编码实现跨视角无缝融合，显著提升泛化性与抗遮挡能力。**

- **链接: []()**

> **作者:** Laura Bragagnolo; Leonardo Barcellona; Stefano Ghidoni
>
> **备注:** WACV 2026
>
> **摘要:** Accurate 3D human pose estimation is fundamental for applications such as augmented reality and human-robot interaction. State-of-the-art multi-view methods learn to fuse predictions across views by training on large annotated datasets, leading to poor generalization when the test scenario differs. To overcome these limitations, we propose SkelSplat, a novel framework for multi-view 3D human pose estimation based on differentiable Gaussian rendering. Human pose is modeled as a skeleton of 3D Gaussians, one per joint, optimized via differentiable rendering to enable seamless fusion of arbitrary camera views without 3D ground-truth supervision. Since Gaussian Splatting was originally designed for dense scene reconstruction, we propose a novel one-hot encoding scheme that enables independent optimization of human joints. SkelSplat outperforms approaches that do not rely on 3D ground truth in Human3.6M and CMU, while reducing the cross-dataset error up to 47.8% compared to learning-based methods. Experiments on Human3.6M-Occ and Occlusion-Person demonstrate robustness to occlusions, without scenario-specific fine-tuning. Our project page is available here: https://skelsplat.github.io.
>
---
#### [new 086] Twist and Compute: The Cost of Pose in 3D Generative Diffusion
- **分类: cs.CV**

- **简介: 该论文研究图像到3D生成模型的视角偏差问题，发现Hunyuan3D 2.0对旋转输入泛化差，提出轻量CNN检测并校正输入姿态，恢复性能，质疑仅靠规模提升是否足够，呼吁引入对称性感知设计。**

- **链接: []()**

> **作者:** Kyle Fogarty; Jack Foster; Boqiao Zhang; Jing Yang; Cengiz Öztireli
>
> **备注:** Accepted to EurIPS 2025 Workshop on Principles of Generative Modeling (PriGM)
>
> **摘要:** Despite their impressive results, large-scale image-to-3D generative models remain opaque in their inductive biases. We identify a significant limitation in image-conditioned 3D generative models: a strong canonical view bias. Through controlled experiments using simple 2D rotations, we show that the state-of-the-art Hunyuan3D 2.0 model can struggle to generalize across viewpoints, with performance degrading under rotated inputs. We show that this failure can be mitigated by a lightweight CNN that detects and corrects input orientation, restoring model performance without modifying the generative backbone. Our findings raise an important open question: Is scale enough, or should we pursue modular, symmetry-aware designs?
>
---
#### [new 087] PEOD: A Pixel-Aligned Event-RGB Benchmark for Object Detection under Challenging Conditions
- **分类: cs.CV**

- **简介: 论文提出PEOD，首个高分辨率（1280×720）像素对齐的事件-RGB数据集，用于挑战场景下的目标检测。涵盖低光、过曝等复杂条件，构建14种方法的基准，揭示融合模型在极端光照下性能受限。**

- **链接: []()**

> **作者:** Luoping Cui; Hanqing Liu; Mingjie Liu; Endian Lin; Donghong Jiang; Yuhao Wang; Chuang Zhu
>
> **摘要:** Robust object detection for challenging scenarios increasingly relies on event cameras, yet existing Event-RGB datasets remain constrained by sparse coverage of extreme conditions and low spatial resolution (<= 640 x 480), which prevents comprehensive evaluation of detectors under challenging scenarios. To address these limitations, we propose PEOD, the first large-scale, pixel-aligned and high-resolution (1280 x 720) Event-RGB dataset for object detection under challenge conditions. PEOD contains 130+ spatiotemporal-aligned sequences and 340k manual bounding boxes, with 57% of data captured under low-light, overexposure, and high-speed motion. Furthermore, we benchmark 14 methods across three input configurations (Event-based, RGB-based, and Event-RGB fusion) on PEOD. On the full test set and normal subset, fusion-based models achieve the excellent performance. However, in illumination challenge subset, the top event-based model outperforms all fusion models, while fusion models still outperform their RGB-based counterparts, indicating limits of existing fusion methods when the frame modality is severely degraded. PEOD establishes a realistic, high-quality benchmark for multimodal perception and facilitates future research.
>
---
#### [new 088] Cross Modal Fine-grained Alignment via Granularity-aware and Region-uncertain Modeling
- **分类: cs.CV; cs.MM**

- **简介: 该论文聚焦细粒度图文对齐任务，解决现有方法因注意力噪声和忽略不确定性导致的匹配不精准问题，提出融合显著性感知与区域不确定性建模的统一框架，提升对齐鲁棒性与可解释性。**

- **链接: []()**

> **作者:** Jiale Liu; Haoming Zhou; Yishu Zhu; Bingzhi Chen; Yuncheng Jiang
>
> **备注:** 10 pages, 6 figures, accepted by AAAI 2026
>
> **摘要:** Fine-grained image-text alignment is a pivotal challenge in multimodal learning, underpinning key applications such as visual question answering, image captioning, and vision-language navigation. Unlike global alignment, fine-grained alignment requires precise correspondence between localized visual regions and textual tokens, often hindered by noisy attention mechanisms and oversimplified modeling of cross-modal relationships. In this work, we identify two fundamental limitations of existing approaches: the lack of robust intra-modal mechanisms to assess the significance of visual and textual tokens, leading to poor generalization in complex scenes; and the absence of fine-grained uncertainty modeling, which fails to capture the one-to-many and many-to-one nature of region-word correspondences. To address these issues, we propose a unified approach that incorporates significance-aware and granularity-aware modeling and region-level uncertainty modeling. Our method leverages modality-specific biases to identify salient features without relying on brittle cross-modal attention, and represents region features as a mixture of Gaussian distributions to capture fine-grained uncertainty. Extensive experiments on Flickr30K and MS-COCO demonstrate that our approach achieves state-of-the-art performance across various backbone architectures, significantly enhancing the robustness and interpretability of fine-grained image-text alignment.
>
---
#### [new 089] Multi-Granularity Mutual Refinement Network for Zero-Shot Learning
- **分类: cs.CV**

- **简介: 该论文面向零样本学习任务，解决现有方法忽略局部视觉特征间交互的问题，提出多粒度互 refine 网络（Mg-MRN），通过解耦多粒度特征提取与跨粒度融合，增强视觉-语义对齐，提升未见类识别性能。**

- **链接: []()**

> **作者:** Ning Wang; Long Yu; Cong Hua; Guangming Zhu; Lin Mei; Syed Afaq Ali Shah; Mohammed Bennamoun; Liang Zhang
>
> **摘要:** Zero-shot learning (ZSL) aims to recognize unseen classes with zero samples by transferring semantic knowledge from seen classes. Current approaches typically correlate global visual features with semantic information (i.e., attributes) or align local visual region features with corresponding attributes to enhance visual-semantic interactions. Although effective, these methods often overlook the intrinsic interactions between local region features, which can further improve the acquisition of transferable and explicit visual features. In this paper, we propose a network named Multi-Granularity Mutual Refinement Network (Mg-MRN), which refine discriminative and transferable visual features by learning decoupled multi-granularity features and cross-granularity feature interactions. Specifically, we design a multi-granularity feature extraction module to learn region-level discriminative features through decoupled region feature mining. Then, a cross-granularity feature fusion module strengthens the inherent interactions between region features of varying granularities. This module enhances the discriminability of representations at each granularity level by integrating region representations from adjacent hierarchies, further improving ZSL recognition performance. Extensive experiments on three popular ZSL benchmark datasets demonstrate the superiority and competitiveness of our proposed Mg-MRN method. Our code is available at https://github.com/NingWang2049/Mg-MRN.
>
---
#### [new 090] Burst Image Quality Assessment: A New Benchmark and Unified Framework for Multiple Downstream Tasks
- **分类: cs.CV**

- **简介: 该论文提出突发图像质量评估（BuIQA）任务，解决冗余帧导致的存储与任务效率问题。构建首个BuIQA基准数据集，并设计统一框架，通过任务驱动提示学习实现高效帧质量评估，显著提升去噪与超分辨率等下游任务性能。**

- **链接: []()**

> **作者:** Xiaoye Liang; Lai Jiang; Minglang Qiao; Yichen Guo; Yue Zhang; Xin Deng; Shengxi Li; Yufan Liu; Mai Xu
>
> **摘要:** In recent years, the development of burst imaging technology has improved the capture and processing capabilities of visual data, enabling a wide range of applications. However, the redundancy in burst images leads to the increased storage and transmission demands, as well as reduced efficiency of downstream tasks. To address this, we propose a new task of Burst Image Quality Assessment (BuIQA), to evaluate the task-driven quality of each frame within a burst sequence, providing reasonable cues for burst image selection. Specifically, we establish the first benchmark dataset for BuIQA, consisting of $7,346$ burst sequences with $45,827$ images and $191,572$ annotated quality scores for multiple downstream scenarios. Inspired by the data analysis, a unified BuIQA framework is proposed to achieve an efficient adaption for BuIQA under diverse downstream scenarios. Specifically, a task-driven prompt generation network is developed with heterogeneous knowledge distillation, to learn the priors of the downstream task. Then, the task-aware quality assessment network is introduced to assess the burst image quality based on the task prompt. Extensive experiments across 10 downstream scenarios demonstrate the impressive BuIQA performance of the proposed approach, outperforming the state-of-the-art. Furthermore, it can achieve $0.33$ dB PSNR improvement in the downstream tasks of denoising and super-resolution, by applying our approach to select the high-quality burst frames.
>
---
#### [new 091] SENCA-st: Integrating Spatial Transcriptomics and Histopathology with Cross Attention Shared Encoder for Region Identification in Cancer Pathology
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出SENCA-st模型，融合空间转录组与病理图像，通过交叉注意力机制保留双模态特征，解决现有方法因侧重单一模态而丢失功能或结构信息的问题，精准识别肿瘤异质性与微环境区域。**

- **链接: []()**

> **作者:** Shanaka Liyanaarachchi; Chathurya Wijethunga; Shihab Aaquil Ahamed; Akthas Absar; Ranga Rodrigo
>
> **备注:** Accepted at WACV 2026
>
> **摘要:** Spatial transcriptomics is an emerging field that enables the identification of functional regions based on the spatial distribution of gene expression. Integrating this functional information present in transcriptomic data with structural data from histopathology images is an active research area with applications in identifying tumor substructures associated with cancer drug resistance. Current histopathology-spatial-transcriptomic region segmentation methods suffer due to either making spatial transcriptomics prominent by using histopathology features just to assist processing spatial transcriptomics data or using vanilla contrastive learning that make histopathology images prominent due to only promoting common features losing functional information. In both extremes, the model gets either lost in the noise of spatial transcriptomics or overly smoothed, losing essential information. Thus, we propose our novel architecture SENCA-st (Shared Encoder with Neighborhood Cross Attention) that preserves the features of both modalities. More importantly, it emphasizes regions that are structurally similar in histopathology but functionally different on spatial transcriptomics using cross-attention. We demonstrate the superior performance of our model that surpasses state-of-the-art methods in detecting tumor heterogeneity and tumor micro-environment regions, a clinically crucial aspect.
>
---
#### [new 092] I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks
- **分类: cs.CV**

- **简介: 论文提出I2E算法，将静态图像实时转换为高保真事件流，解决SNN训练中事件数据稀缺问题。通过模拟微眼动实现300倍加速，支持在线数据增强，并在ImageNet和CIFAR10-DVS上实现SOTA精度，验证合成数据可替代真实传感器数据。**

- **链接: []()**

> **作者:** Ruichen Ma; Liwei Meng; Guanchao Qiao; Ning Ning; Yang Liu; Shaogang Hu
>
> **备注:** AAAI-26 Oral
>
> **摘要:** Spiking neural networks (SNNs) promise highly energy-efficient computing, but their adoption is hindered by a critical scarcity of event-stream data. This work introduces I2E, an algorithmic framework that resolves this bottleneck by converting static images into high-fidelity event streams. By simulating microsaccadic eye movements with a highly parallelized convolution, I2E achieves a conversion speed over 300x faster than prior methods, uniquely enabling on-the-fly data augmentation for SNN training. The framework's effectiveness is demonstrated on large-scale benchmarks. An SNN trained on the generated I2E-ImageNet dataset achieves a state-of-the-art accuracy of 60.50%. Critically, this work establishes a powerful sim-to-real paradigm where pre-training on synthetic I2E data and fine-tuning on the real-world CIFAR10-DVS dataset yields an unprecedented accuracy of 92.5%. This result validates that synthetic event data can serve as a high-fidelity proxy for real sensor data, bridging a long-standing gap in neuromorphic engineering. By providing a scalable solution to the data problem, I2E offers a foundational toolkit for developing high-performance neuromorphic systems. The open-source algorithm and all generated datasets are provided to accelerate research in the field.
>
---
#### [new 093] VideoChain: A Transformer-Based Framework for Multi-hop Video Question Generation
- **分类: cs.CV**

- **简介: 论文提出VideoChain，首个面向多跳视频问答生成的Transformer框架，解决现有视频QG仅限单段零跳问题的局限，通过融合视频与文本特征，在TVQA+上构建MVQ-60数据集，实现跨时序片段的推理型问题生成。**

- **链接: []()**

> **作者:** Arpan Phukan; Anupam Pandey; Deepjyoti Bodo; Asif Ekbal
>
> **摘要:** Multi-hop Question Generation (QG) effectively evaluates reasoning but remains confined to text; Video Question Generation (VideoQG) is limited to zero-hop questions over single segments. To address this, we introduce VideoChain, a novel Multi-hop Video Question Generation (MVQG) framework designed to generate questions that require reasoning across multiple, temporally separated video segments. VideoChain features a modular architecture built on a modified BART backbone enhanced with video embeddings, capturing textual and visual dependencies. Using the TVQA+ dataset, we automatically construct the large-scale MVQ-60 dataset by merging zero-hop QA pairs, ensuring scalability and diversity. Evaluations show VideoChain's strong performance across standard generation metrics: ROUGE-L (0.6454), ROUGE-1 (0.6854), BLEU-1 (0.6711), BERTScore-F1 (0.7967), and semantic similarity (0.8110). These results highlight the model's ability to generate coherent, contextually grounded, and reasoning-intensive questions.
>
---
#### [new 094] Generating Sketches in a Hierarchical Auto-Regressive Process for Flexible Sketch Drawing Manipulation at Stroke-Level
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种分层自回归生成框架，实现笔画级可交互的草图生成。解决传统方法需预设全部笔画条件、无法中途调整的问题，通过逐步预测、定位与绘制笔画，支持生成过程中灵活编辑。**

- **链接: []()**

> **作者:** Sicong Zang; Shuhui Gao; Zhijun Fang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Generating sketches with specific patterns as expected, i.e., manipulating sketches in a controllable way, is a popular task. Recent studies control sketch features at stroke-level by editing values of stroke embeddings as conditions. However, in order to provide generator a global view about what a sketch is going to be drawn, all these edited conditions should be collected and fed into generator simultaneously before generation starts, i.e., no further manipulation is allowed during sketch generating process. In order to realize sketch drawing manipulation more flexibly, we propose a hierarchical auto-regressive sketch generating process. Instead of generating an entire sketch at once, each stroke in a sketch is generated in a three-staged hierarchy: 1) predicting a stroke embedding to represent which stroke is going to be drawn, and 2) anchoring the predicted stroke on the canvas, and 3) translating the embedding to a sequence of drawing actions to form the full sketch. Moreover, the stroke prediction, anchoring and translation are proceeded auto-regressively, i.e., both the recently generated strokes and their positions are considered to predict the current one, guiding model to produce an appropriate stroke at a suitable position to benefit the full sketch generation. It is flexible to manipulate stroke-level sketch drawing at any time during generation by adjusting the exposed editable stroke embeddings.
>
---
#### [new 095] Introducing Nylon Face Mask Attacks: A Dataset for Evaluating Generalised Face Presentation Attack Detection
- **分类: cs.CV; cs.ET**

- **简介: 该论文提出Nylon Face Mask攻击数据集，用于评估人脸识别系统对新型弹性光刻面具攻击的泛化检测能力。工作包括构建大规模真实场景数据集，并测试现有PAD方法的鲁棒性，揭示其在新型攻击下的性能局限。**

- **链接: []()**

> **作者:** Manasa; Sushrut Patwardhan; Narayan Vetrekar; Pavan Kumar; R. S. Gad; Raghavendra Ramachandra
>
> **备注:** Accepted in Proc. of International Conference on Artificial Intelligence, Computer, Data Sciences and Applications (ACDSA 2026)
>
> **摘要:** Face recognition systems are increasingly deployed across a wide range of applications, including smartphone authentication, access control, and border security. However, these systems remain vulnerable to presentation attacks (PAs), which can significantly compromise their reliability. In this work, we introduce a new dataset focused on a novel and realistic presentation attack instrument called Nylon Face Masks (NFMs), designed to simulate advanced 3D spoofing scenarios. NFMs are particularly concerning due to their elastic structure and photorealistic appearance, which enable them to closely mimic the victim's facial geometry when worn by an attacker. To reflect real-world smartphone-based usage conditions, we collected the dataset using an iPhone 11 Pro, capturing 3,760 bona fide samples from 100 subjects and 51,281 NFM attack samples across four distinct presentation scenarios involving both humans and mannequins. We benchmark the dataset using five state-of-the-art PAD methods to evaluate their robustness under unseen attack conditions. The results demonstrate significant performance variability across methods, highlighting the challenges posed by NFMs and underscoring the importance of developing PAD techniques that generalise effectively to emerging spoofing threats.
>
---
#### [new 096] Exploring the Underwater World Segmentation without Extra Training
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对水下目标分割缺乏标注数据与模型适配问题，提出无训练迁移框架Earth2Ocean，构建首个大规模水下开放词汇分割数据集AquaOV255与基准UOVSBench，利用几何引导与语义对齐实现陆地VLM向水下域的高效迁移。**

- **链接: []()**

> **作者:** Bingyu Li; Tao Huo; Da Zhang; Zhiyuan Zhao; Junyu Gao; Xuelong Li
>
> **摘要:** Accurate segmentation of marine organisms is vital for biodiversity monitoring and ecological assessment, yet existing datasets and models remain largely limited to terrestrial scenes. To bridge this gap, we introduce \textbf{AquaOV255}, the first large-scale and fine-grained underwater segmentation dataset containing 255 categories and over 20K images, covering diverse categories for open-vocabulary (OV) evaluation. Furthermore, we establish the first underwater OV segmentation benchmark, \textbf{UOVSBench}, by integrating AquaOV255 with five additional underwater datasets to enable comprehensive evaluation. Alongside, we present \textbf{Earth2Ocean}, a training-free OV segmentation framework that transfers terrestrial vision--language models (VLMs) to underwater domains without any additional underwater training. Earth2Ocean consists of two core components: a Geometric-guided Visual Mask Generator (\textbf{GMG}) that refines visual features via self-similarity geometric priors for local structure perception, and a Category-visual Semantic Alignment (\textbf{CSA}) module that enhances text embeddings through multimodal large language model reasoning and scene-aware template construction. Extensive experiments on the UOVSBench benchmark demonstrate that Earth2Ocean achieves significant performance improvement on average while maintaining efficient inference.
>
---
#### [new 097] PC-Diffusion: Aligning Diffusion Models with Human Preferences via Preference Classifier
- **分类: cs.CV**

- **简介: PC-Diffusion提出一种轻量级偏好分类器，实现扩散模型与人类偏好的对齐，无需微调整个模型或依赖参考模型，降低计算成本并提升稳定性，同时保持与DPO相当的偏好一致性。**

- **链接: []()**

> **作者:** Shaomeng Wang; He Wang; Xiaolu Wei; Longquan Dai; Jinhui Tang
>
> **备注:** 10 pages, 3 figures, 2 tables
>
> **摘要:** Diffusion models have achieved remarkable success in conditional image generation, yet their outputs often remain misaligned with human preferences. To address this, recent work has applied Direct Preference Optimization (DPO) to diffusion models, yielding significant improvements.~However, DPO-like methods exhibit two key limitations: 1) High computational cost,due to the entire model fine-tuning; 2) Sensitivity to reference model quality}, due to its tendency to introduce instability and bias. To overcome these limitations, we propose a novel framework for human preference alignment in diffusion models (PC-Diffusion), using a lightweight, trainable Preference Classifier that directly models the relative preference between samples. By restricting preference learning to this classifier, PC-Diffusion decouples preference alignment from the generative model, eliminating the need for entire model fine-tuning and reference model reliance.~We further provide theoretical guarantees for PC-Diffusion:1) PC-Diffusion ensures that the preference-guided distributions are consistently propagated across timesteps. 2)The training objective of the preference classifier is equivalent to DPO, but does not require a reference model.3) The proposed preference-guided correction can progressively steer generation toward preference-aligned regions.~Empirical results show that PC-Diffusion achieves comparable preference consistency to DPO while significantly reducing training costs and enabling efficient and stable preference-guided generation.
>
---
#### [new 098] 3D4D: An Interactive, Editable, 4D World Model via 3D Video Generation
- **分类: cs.CV**

- **简介: 论文提出3D4D框架，将静态图像与文本转化为可交互、可编辑的4D场景，通过WebGL与Supersplat渲染实现高效实时多模态探索，解决静态内容到动态4D世界建模的难题。**

- **链接: []()**

> **作者:** Yunhong He; Zhengqing Yuan; Zhengzhong Tu; Yanfang Ye; Lichao Sun
>
> **备注:** Accepted by AAAI 2026 Demo Track
>
> **摘要:** We introduce 3D4D, an interactive 4D visualization framework that integrates WebGL with Supersplat rendering. It transforms static images and text into coherent 4D scenes through four core modules and employs a foveated rendering strategy for efficient, real-time multi-modal interaction. This framework enables adaptive, user-driven exploration of complex 4D environments. The project page and code are available at https://yunhonghe1021.github.io/NOVA/.
>
---
#### [new 099] UI2Code$^\text{N}$: A Visual Language Model for Test-Time Scalable Interactive UI-to-Code Generation
- **分类: cs.CV**

- **简介: 论文提出UI2Code$^\text{N}$，一种视觉语言模型，面向交互式UI到代码生成任务，解决多模态编码能力弱与单轮交互局限问题，通过三阶段训练与测试时扩展，实现UI生成、编辑与优化一体化，性能达闭源模型水平。**

- **链接: []()**

> **作者:** Zhen Yang; Wenyi Hong; Mingde Xu; Xinyue Fan; Weihan Wang; Jiele Cheng; Xiaotao Gu; Jie Tang
>
> **备注:** 24 pages
>
> **摘要:** User interface (UI) programming is a core yet highly complex part of modern software development. Recent advances in visual language models (VLMs) highlight the potential of automatic UI coding, but current approaches face two key limitations: multimodal coding capabilities remain underdeveloped, and single-turn paradigms make little use of iterative visual feedback. We address these challenges with an interactive UI-to-code paradigm that better reflects real-world workflows and raises the upper bound of achievable performance. Under this paradigm, we present UI2Code$^\text{N}$, a visual language model trained through staged pretraining, fine-tuning, and reinforcement learning to achieve foundational improvements in multimodal coding. The model unifies three key capabilities: UI-to-code generation, UI editing, and UI polishing. We further explore test-time scaling for interactive generation, enabling systematic use of multi-turn feedback. Experiments on UI-to-code and UI polishing benchmarks show that UI2Code$^\text{N}$ establishes a new state of the art among open-source models and achieves performance comparable to leading closed-source models such as Claude-4-Sonnet and GPT-5. Our code and models are available at https://github.com/zai-org/UI2Code_N.
>
---
#### [new 100] ProSona: Prompt-Guided Personalization for Multi-Expert Medical Image Segmentation
- **分类: cs.CV; cs.AI**

- **简介: ProSona针对医学图像分割中专家标注差异问题，提出基于自然语言提示的个性化分割框架，通过潜在风格空间与提示引导投影，实现可解释、可控的多专家风格建模，显著提升分割准确性与一致性。**

- **链接: []()**

> **作者:** Aya Elgebaly; Nikolaos Delopoulos; Juliane Hörner-Rieber; Carolin Rippke; Sebastian Klüter; Luca Boldrini; Lorenzo Placidi; Riccardo Dal Bello; Nicolaus Andratschke; Michael Baumgartl; Claus Belka; Christopher Kurz; Guillaume Landry; Shadi Albarqouni
>
> **备注:** 5 pages, 5 figures. Submitted to IEEE International Symposium on Biomedical Imaging (ISBI) 2026
>
> **摘要:** Automated medical image segmentation suffers from high inter-observer variability, particularly in tasks such as lung nodule delineation, where experts often disagree. Existing approaches either collapse this variability into a consensus mask or rely on separate model branches for each annotator. We introduce ProSona, a two-stage framework that learns a continuous latent space of annotation styles, enabling controllable personalization via natural language prompts. A probabilistic U-Net backbone captures diverse expert hypotheses, while a prompt-guided projection mechanism navigates this latent space to generate personalized segmentations. A multi-level contrastive objective aligns textual and visual representations, promoting disentangled and interpretable expert styles. Across the LIDC-IDRI lung nodule and multi-institutional prostate MRI datasets, ProSona reduces the Generalized Energy Distance by 17% and improves mean Dice by more than one point compared with DPersona. These results demonstrate that natural-language prompts can provide flexible, accurate, and interpretable control over personalized medical image segmentation. Our implementation is available online 1 .
>
---
#### [new 101] Mitigating Negative Flips via Margin Preserving Training
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于持续学习任务，旨在缓解模型更新时对旧类别的误分类（负翻转）。提出margin-preserving训练方法，通过logit边缘校准与双源焦点蒸馏，兼顾旧类稳定性与新类准确性。**

- **链接: []()**

> **作者:** Simone Ricci; Niccolò Biondi; Federico Pernici; Alberto Del Bimbo
>
> **备注:** Accepted at AAAI2026
>
> **摘要:** Minimizing inconsistencies across successive versions of an AI system is as crucial as reducing the overall error. In image classification, such inconsistencies manifest as negative flips, where an updated model misclassifies test samples that were previously classified correctly. This issue becomes increasingly pronounced as the number of training classes grows over time, since adding new categories reduces the margin of each class and may introduce conflicting patterns that undermine their learning process, thereby degrading performance on the original subset. To mitigate negative flips, we propose a novel approach that preserves the margins of the original model while learning an improved one. Our method encourages a larger relative margin between the previously learned and newly introduced classes by introducing an explicit margin-calibration term on the logits. However, overly constraining the logit margin for the new classes can significantly degrade their accuracy compared to a new independently trained model. To address this, we integrate a double-source focal distillation loss with the previous model and a new independently trained model, learning an appropriate decision margin from both old and new data, even under a logit margin calibration. Extensive experiments on image classification benchmarks demonstrate that our approach consistently reduces the negative flip rate with high overall accuracy.
>
---
#### [new 102] LayerEdit: Disentangled Multi-Object Editing via Conflict-Aware Multi-Layer Learning
- **分类: cs.CV**

- **简介: 论文提出LayerEdit，解决文本驱动多对象图像编辑中的对象间干扰问题，通过冲突感知分层分解、独立编辑与透明融合，实现无泄漏、高协同的多对象精准编辑，无需训练。**

- **链接: []()**

> **作者:** Fengyi Fu; Mengqi Huang; Lei Zhang; Zhendong Mao
>
> **备注:** The 40th Annual AAAI Conference on Artificial Intelligence
>
> **摘要:** Text-driven multi-object image editing which aims to precisely modify multiple objects within an image based on text descriptions, has recently attracted considerable interest. Existing works primarily follow the localize-editing paradigm, focusing on independent object localization and editing while neglecting critical inter-object interactions. However, this work points out that the neglected attention entanglements in inter-object conflict regions, inherently hinder disentangled multi-object editing, leading to either inter-object editing leakage or intra-object editing constraints. We thereby propose a novel multi-layer disentangled editing framework LayerEdit, a training-free method which, for the first time, through precise object-layered decomposition and coherent fusion, enables conflict-free object-layered editing. Specifically, LayerEdit introduces a novel "decompose-editingfusion" framework, consisting of: (1) Conflict-aware Layer Decomposition module, which utilizes an attention-aware IoU scheme and time-dependent region removing, to enhance conflict awareness and suppression for layer decomposition. (2) Object-layered Editing module, to establish coordinated intra-layer text guidance and cross-layer geometric mapping, achieving disentangled semantic and structural modifications. (3) Transparency-guided Layer Fusion module, to facilitate structure-coherent inter-object layer fusion through precise transparency guidance learning. Extensive experiments verify the superiority of LayerEdit over existing methods, showing unprecedented intra-object controllability and inter-object coherence in complex multi-object scenarios. Codes are available at: https://github.com/fufy1024/LayerEdit.
>
---
#### [new 103] Laplacian Score Sharpening for Mitigating Hallucination in Diffusion Models
- **分类: cs.CV; cs.AI; cs.LG; stat.ML**

- **简介: 该论文针对扩散模型中的幻觉问题，提出基于拉普拉斯得分锐化的后处理方法，通过估计得分的锐度抑制模式插值导致的不合理生成，在多维数据上有效降低幻觉率并关联得分不确定性。**

- **链接: []()**

> **作者:** Barath Chandran. C; Srinivas Anumasa; Dianbo Liu
>
> **摘要:** Diffusion models, though successful, are known to suffer from hallucinations that create incoherent or unrealistic samples. Recent works have attributed this to the phenomenon of mode interpolation and score smoothening, but they lack a method to prevent their generation during sampling. In this paper, we propose a post-hoc adjustment to the score function during inference that leverages the Laplacian (or sharpness) of the score to reduce mode interpolation hallucination in unconditional diffusion models across 1D, 2D, and high-dimensional image data. We derive an efficient Laplacian approximation for higher dimensions using a finite-difference variant of the Hutchinson trace estimator. We show that this correction significantly reduces the rate of hallucinated samples across toy 1D/2D distributions and a high- dimensional image dataset. Furthermore, our analysis explores the relationship between the Laplacian and uncertainty in the score.
>
---
#### [new 104] Morphing Through Time: Diffusion-Based Bridging of Temporal Gaps for Robust Alignment in Change Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对遥感时序图像因时间跨度大导致的配准困难问题，提出一种基于扩散模型的语义形态变换框架，无需修改现有检测网络，即可实现高精度时空对齐，显著提升变化检测鲁棒性。**

- **链接: []()**

> **作者:** Seyedehanita Madani; Vishal M. Patel
>
> **备注:** 9 pages, 5 figures. To appear in WACV 2026
>
> **摘要:** Remote sensing change detection is often challenged by spatial misalignment between bi-temporal images, especially when acquisitions are separated by long seasonal or multi-year gaps. While modern convolutional and transformer-based models perform well on aligned data, their reliance on precise co-registration limits their robustness in real-world conditions. Existing joint registration-detection frameworks typically require retraining and transfer poorly across domains. We introduce a modular pipeline that improves spatial and temporal robustness without altering existing change detection networks. The framework integrates diffusion-based semantic morphing, dense registration, and residual flow refinement. A diffusion module synthesizes intermediate morphing frames that bridge large appearance gaps, enabling RoMa to estimate stepwise correspondences between consecutive frames. The composed flow is then refined through a lightweight U-Net to produce a high-fidelity warp that co-registers the original image pair. Extensive experiments on LEVIR-CD, WHU-CD, and DSIFN-CD show consistent gains in both registration accuracy and downstream change detection across multiple backbones, demonstrating the generality and effectiveness of the proposed approach.
>
---
#### [new 105] Pixel-level Quality Assessment for Oriented Object Detection
- **分类: cs.CV**

- **简介: 该论文面向定向目标检测，解决盒级IoU质量评估的偏差问题，提出像素级质量评估（PQA）框架，通过计算像素空间一致性替代盒级IoU预测，更准确反映定位精度，显著提升多种检测器性能。**

- **链接: []()**

> **作者:** Yunhui Zhu; Buliao Huang
>
> **摘要:** Modern oriented object detectors typically predict a set of bounding boxes and select the top-ranked ones based on estimated localization quality. Achieving high detection performance requires that the estimated quality closely aligns with the actual localization accuracy. To this end, existing approaches predict the Intersection over Union (IoU) between the predicted and ground-truth (GT) boxes as a proxy for localization quality. However, box-level IoU prediction suffers from a structural coupling issue: since the predicted box is derived from the detector's internal estimation of the GT box, the predicted IoU--based on their similarity--can be overestimated for poorly localized boxes. To overcome this limitation, we propose a novel Pixel-level Quality Assessment (PQA) framework, which replaces box-level IoU prediction with the integration of pixel-level spatial consistency. PQA measures the alignment between each pixel's relative position to the predicted box and its corresponding position to the GT box. By operating at the pixel level, PQA avoids directly comparing the predicted box with the estimated GT box, thereby eliminating the inherent similarity bias in box-level IoU prediction. Furthermore, we introduce a new integration metric that aggregates pixel-level spatial consistency into a unified quality score, yielding a more accurate approximation of the actual localization quality. Extensive experiments on HRSC2016 and DOTA demonstrate that PQA can be seamlessly integrated into various oriented object detectors, consistently improving performance (e.g., +5.96% AP$_{50:95}$ on Rotated RetinaNet and +2.32% on STD).
>
---
#### [new 106] Visual Bridge: Universal Visual Perception Representations Generating
- **分类: cs.CV**

- **简介: 该论文提出Visual Bridge，一种基于流匹配的通用视觉感知框架，解决多任务模型孤立训练问题，通过统一的velocity场桥接图像块与多任务表征，实现跨任务零样本与微调性能超越专有与通用模型。**

- **链接: []()**

> **作者:** Yilin Gao; Shuguang Dou; Junzhou Li; Zhiheng Yu; Yin Li; Dongsheng Jiang; Shugong Xu
>
> **备注:** Accepted by AAAI2026
>
> **摘要:** Recent advances in diffusion models have achieved remarkable success in isolated computer vision tasks such as text-to-image generation, depth estimation, and optical flow. However, these models are often restricted by a ``single-task-single-model'' paradigm, severely limiting their generalizability and scalability in multi-task scenarios. Motivated by the cross-domain generalization ability of large language models, we propose a universal visual perception framework based on flow matching that can generate diverse visual representations across multiple tasks. Our approach formulates the process as a universal flow-matching problem from image patch tokens to task-specific representations rather than an independent generation or regression problem. By leveraging a strong self-supervised foundation model as the anchor and introducing a multi-scale, circular task embedding mechanism, our method learns a universal velocity field to bridge the gap between heterogeneous tasks, supporting efficient and flexible representation transfer. Extensive experiments on classification, detection, segmentation, depth estimation, and image-text retrieval demonstrate that our model achieves competitive performance in both zero-shot and fine-tuned settings, outperforming prior generalist and several specialist models. Ablation studies further validate the robustness, scalability, and generalization of our framework. Our work marks a significant step towards general-purpose visual perception, providing a solid foundation for future research in universal vision modeling.
>
---
#### [new 107] DANCE: Density-agnostic and Class-aware Network for Point Cloud Completion
- **分类: cs.CV**

- **简介: DANCE提出一种密度无关、类别感知的点云补全方法，通过多视角射线采样与Transformer解码器精准恢复缺失区域，结合几何特征分类头实现无图像监督的语义一致补全，显著提升对稀疏与噪声输入的鲁棒性。**

- **链接: []()**

> **作者:** Da-Yeong Kim; Yeong-Jun Cho
>
> **摘要:** Point cloud completion aims to recover missing geometric structures from incomplete 3D scans, which often suffer from occlusions or limited sensor viewpoints. Existing methods typically assume fixed input/output densities or rely on image-based representations, making them less suitable for real-world scenarios with variable sparsity and limited supervision. In this paper, we introduce Density-agnostic and Class-aware Network (DANCE), a novel framework that completes only the missing regions while preserving the observed geometry. DANCE generates candidate points via ray-based sampling from multiple viewpoints. A transformer decoder then refines their positions and predicts opacity scores, which determine the validity of each point for inclusion in the final surface. To incorporate semantic guidance, a lightweight classification head is trained directly on geometric features, enabling category-consistent completion without external image supervision. Extensive experiments on the PCN and MVP benchmarks show that DANCE outperforms state-of-the-art methods in accuracy and structural consistency, while remaining robust to varying input densities and noise levels.
>
---
#### [new 108] Accurate and Efficient Surface Reconstruction from Point Clouds via Geometry-Aware Local Adaptation
- **分类: cs.CV**

- **简介: 该论文面向点云表面重建任务，解决传统局部区域固定大小与均匀分布导致的适应性不足问题，提出基于曲率自适应调整局部区域大小与间距的方法，提升重建精度与效率。**

- **链接: []()**

> **作者:** Eito Ogawa; Taiga Hayami; Hiroshi Watanabe
>
> **备注:** 4 pages
>
> **摘要:** Point cloud surface reconstruction has improved in accuracy with advances in deep learning, enabling applications such as infrastructure inspection. Recent approaches that reconstruct from small local regions rather than entire point clouds have attracted attention for their strong generalization capability. However, prior work typically places local regions uniformly and keeps their size fixed, limiting adaptability to variations in geometric complexity. In this study, we propose a method that improves reconstruction accuracy and efficiency by adaptively modulating the spacing and size of local regions based on the curvature of the input point cloud.
>
---
#### [new 109] Sparse3DPR: Training-Free 3D Hierarchical Scene Parsing and Task-Adaptive Subgraph Reasoning from Sparse RGB Views
- **分类: cs.CV; cs.AI**

- **简介: Sparse3DPR提出一种无需训练的3D场景理解框架，利用LLM从稀疏RGB视图构建平面增强场景图，通过任务自适应子图提取提升推理效率与准确率，在Space3D-Bench上显著超越基线，性能媲美训练方法。**

- **链接: []()**

> **作者:** Haida Feng; Hao Wei; Zewen Xu; Haolin Wang; Chade Li; Yihong Wu
>
> **摘要:** Recently, large language models (LLMs) have been explored widely for 3D scene understanding. Among them, training-free approaches are gaining attention for their flexibility and generalization over training-based methods. However, they typically struggle with accuracy and efficiency in practical deployment. To address the problems, we propose Sparse3DPR, a novel training-free framework for open-ended scene understanding, which leverages the reasoning capabilities of pre-trained LLMs and requires only sparse-view RGB inputs. Specifically, we introduce a hierarchical plane-enhanced scene graph that supports open vocabulary and adopts dominant planar structures as spatial anchors, which enables clearer reasoning chains and more reliable high-level inferences. Furthermore, we design a task-adaptive subgraph extraction method to filter query-irrelevant information dynamically, reducing contextual noise and improving 3D scene reasoning efficiency and accuracy. Experimental results demonstrate the superiority of Sparse3DPR, which achieves a 28.7% EM@1 improvement and a 78.2% speedup compared with ConceptGraphs on the Space3D-Bench. Moreover, Sparse3DPR obtains comparable performance to training-based methods on ScanQA, with additional real-world experiments confirming its robustness and generalization capability.
>
---
#### [new 110] Re-coding for Uncertainties: Edge-awareness Semantic Concordance for Resilient Event-RGB Segmentation
- **分类: cs.CV**

- **简介: 该论文针对极端条件下RGB图像语义分割性能下降问题，提出边缘感知语义一致性框架，利用事件模态的边缘线索对齐多模态特征，实现鲁棒融合，显著提升分割精度与抗遮挡能力。**

- **链接: []()**

> **作者:** Nan Bao; Yifan Zhao; Lin Zhu; Jia Li
>
> **备注:** Accepted to NeurIPS 2025; code and datasets available at https://github.com/iCVTEAM/ESC
>
> **摘要:** Semantic segmentation has achieved great success in ideal conditions. However, when facing extreme conditions (e.g., insufficient light, fierce camera motion), most existing methods suffer from significant information loss of RGB, severely damaging segmentation results. Several researches exploit the high-speed and high-dynamic event modality as a complement, but event and RGB are naturally heterogeneous, which leads to feature-level mismatch and inferior optimization of existing multi-modality methods. Different from these researches, we delve into the edge secret of both modalities for resilient fusion and propose a novel Edge-awareness Semantic Concordance framework to unify the multi-modality heterogeneous features with latent edge cues. In this framework, we first propose Edge-awareness Latent Re-coding, which obtains uncertainty indicators while realigning event-RGB features into unified semantic space guided by re-coded distribution, and transfers event-RGB distributions into re-coded features by utilizing a pre-established edge dictionary as clues. We then propose Re-coded Consolidation and Uncertainty Optimization, which utilize re-coded edge features and uncertainty indicators to solve the heterogeneous event-RGB fusion issues under extreme conditions. We establish two synthetic and one real-world event-RGB semantic segmentation datasets for extreme scenario comparisons. Experimental results show that our method outperforms the state-of-the-art by a 2.55% mIoU on our proposed DERS-XS, and possesses superior resilience under spatial occlusion. Our code and datasets are publicly available at https://github.com/iCVTEAM/ESC.
>
---
#### [new 111] Filtered-ViT: A Robust Defense Against Multiple Adversarial Patch Attacks
- **分类: cs.CV; cs.AI**

- **简介: 论文提出Filtered-ViT，一种新型视觉Transformer，通过SMART-VMF滤波机制应对多 adversarial patch 攻击，同时抵御自然图像伪影，在保持高清洁准确率的同时显著提升鲁棒性，首次实现对抗与自然干扰的统一防御。**

- **链接: []()**

> **作者:** Aja Khanal; Ahmed Faid; Apurva Narayan
>
> **摘要:** Deep learning vision systems are increasingly deployed in safety-critical domains such as healthcare, yet they remain vulnerable to small adversarial patches that can trigger misclassifications. Most existing defenses assume a single patch and fail when multiple localized disruptions occur, the type of scenario adversaries and real-world artifacts often exploit. We propose Filtered-ViT, a new vision transformer architecture that integrates SMART Vector Median Filtering (SMART-VMF), a spatially adaptive, multi-scale, robustness-aware mechanism that enables selective suppression of corrupted regions while preserving semantic detail. On ImageNet with LaVAN multi-patch attacks, Filtered-ViT achieves 79.8% clean accuracy and 46.3% robust accuracy under four simultaneous 1\% patches, outperforming existing defenses. Beyond synthetic benchmarks, a real-world case study on radiographic medical imagery shows that Filtered-ViT mitigates natural artifacts such as occlusions and scanner noise without degrading diagnostic content. This establishes Filtered-ViT as the first transformer to demonstrate unified robustness against both adversarial and naturally occurring patch-like disruptions, charting a path toward reliable vision systems in truly high-stakes environments.
>
---
#### [new 112] Beyond Randomness: Understand the Order of the Noise in Diffusion
- **分类: cs.CV**

- **简介: 该论文针对文本生成扩散模型，揭示噪声中蕴含语义模式，提出无需训练的“语义擦除-注入”两步法，通过操控初始噪声实现精准语义调控，提升生成一致性。**

- **链接: []()**

> **作者:** Song Yan; Min Li; Bi Xinliang; Jian Yang; Yusen Zhang; Guanye Xiong; Yunwei Lan; Tao Zhang; Wei Zhai; Zheng-Jun Zha
>
> **摘要:** In text-driven content generation (T2C) diffusion model, semantic of generated content is mostly attributed to the process of text embedding and attention mechanism interaction. The initial noise of the generation process is typically characterized as a random element that contributes to the diversity of the generated content. Contrary to this view, this paper reveals that beneath the random surface of noise lies strong analyzable patterns. Specifically, this paper first conducts a comprehensive analysis of the impact of random noise on the model's generation. We found that noise not only contains rich semantic information, but also allows for the erasure of unwanted semantics from it in an extremely simple way based on information theory, and using the equivalence between the generation process of diffusion model and semantic injection to inject semantics into the cleaned noise. Then, we mathematically decipher these observations and propose a simple but efficient training-free and universal two-step "Semantic Erasure-Injection" process to modulate the initial noise in T2C diffusion model. Experimental results demonstrate that our method is consistently effective across various T2C models based on both DiT and UNet architectures and presents a novel perspective for optimizing the generation of diffusion model, providing a universal tool for consistent generation.
>
---
#### [new 113] Foam Segmentation in Wastewater Treatment Plants: A Federated Learning Approach with Segment Anything Model 2
- **分类: cs.CV; cs.DC; cs.LG**

- **简介: 该论文提出一种基于联邦学习与SAM2的泡沫分割框架，解决污水处理厂泡沫检测中数据稀缺、隐私难共享问题，通过分布式训练实现高效、隐私保护的实时泡沫识别。**

- **链接: []()**

> **作者:** Mehmet Batuhan Duman; Alejandro Carnero; Cristian Martín; Daniel Garrido; Manuel Díaz
>
> **备注:** 36 pages, 14 figures, 3 tables, 4 algorithms. This work is part of the Zerovision project. Code available at: https://github.com/ertis-research/zerovision
>
> **摘要:** Foam formation in Wastewater Treatment Plants (WTPs) is a major challenge that can reduce treatment efficiency and increase costs. The ability to automatically examine changes in real-time with respect to the percentage of foam can be of great benefit to the plant. However, large amounts of labeled data are required to train standard Machine Learning (ML) models. The development of these systems is slow due to the scarcity and heterogeneity of labeled data. Additionally, the development is often hindered by the fact that different WTPs do not share their data due to privacy concerns. This paper proposes a new framework to address these challenges by combining Federated Learning (FL) with the state-of-the-art base model for image segmentation, Segment Anything Model 2 (SAM2). The FL paradigm enables collaborative model training across multiple WTPs without centralizing sensitive operational data, thereby ensuring privacy. The framework accelerates training convergence and improves segmentation performance even with limited local datasets by leveraging SAM2's strong pre-trained weights for initialization. The methodology involves fine-tuning SAM2 on distributed clients (edge nodes) using the Flower framework, where a central Fog server orchestrates the process by aggregating model weights without accessing private data. The model was trained and validated using various data collections, including real-world images captured at a WTPs in Granada, Spain, a synthetically generated foam dataset, and images from publicly available datasets to improve generalization. This research offers a practical, scalable, and privacy-aware solution for automatic foam tracking in WTPs. The findings highlight the significant potential of integrating large-scale foundational models into FL systems to solve real-world industrial challenges characterized by distributed and sensitive data.
>
---
#### [new 114] NERVE: Neighbourhood & Entropy-guided Random-walk for training free open-Vocabulary sEgmentation
- **分类: cs.CV; cs.AI**

- **简介: NERVE提出一种无训练的开放词汇语义分割方法，利用扩散模型自注意力的邻域结构与熵引导随机游走，替代固定核与加权融合，实现高效、任意形状物体分割，无需CRF等后处理，达SOTA零样本性能。**

- **链接: []()**

> **作者:** Kunal Mahatha; Jose Dolz; Christian Desrosiers
>
> **摘要:** Despite recent advances in Open-Vocabulary Semantic Segmentation (OVSS), existing training-free methods face several limitations: use of computationally expensive affinity refinement strategies, ineffective fusion of transformer attention maps due to equal weighting or reliance on fixed-size Gaussian kernels to reinforce local spatial smoothness, enforcing isotropic neighborhoods. We propose a strong baseline for training-free OVSS termed as NERVE (Neighbourhood \& Entropy-guided Random-walk for open-Vocabulary sEgmentation), which uniquely integrates global and fine-grained local information, exploiting the neighbourhood structure from the self-attention layer of a stable diffusion model. We also introduce a stochastic random walk for refining the affinity rather than relying on fixed-size Gaussian kernels for local context. This spatial diffusion process encourages propagation across connected and semantically related areas, enabling it to effectively delineate objects with arbitrary shapes. Whereas most existing approaches treat self-attention maps from different transformer heads or layers equally, our method uses entropy-based uncertainty to select the most relevant maps. Notably, our method does not require any conventional post-processing techniques like Conditional Random Fields (CRF) or Pixel-Adaptive Mask Refinement (PAMR). Experiments are performed on 7 popular semantic segmentation benchmarks, yielding an overall state-of-the-art zero-shot segmentation performance, providing an effective approach to open-vocabulary semantic segmentation.
>
---
#### [new 115] Sharp Eyes and Memory for VideoLLMs: Information-Aware Visual Token Pruning for Efficient and Reliable VideoLLM Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 论文提出SharpV，面向视频大语言模型（VideoLLMs），解决视觉令牌与KV缓存冗余导致的高计算开销问题，通过信息感知的自适应剪枝策略，实现高效推理且兼容硬件加速，首次实现无需暴露注意力分数的两阶段剪枝。**

- **链接: []()**

> **作者:** Jialong Qin; Xin Zou; Di Lu; Yibo Yan; Xuming Hu
>
> **摘要:** Current Video Large Language Models (VideoLLMs) suffer from quadratic computational complexity and key-value cache scaling, due to their reliance on processing excessive redundant visual tokens. To address this problem, we propose SharpV, a minimalist and efficient method for adaptive pruning of visual tokens and KV cache. Different from most uniform compression approaches, SharpV dynamically adjusts pruning ratios based on spatial-temporal information. Remarkably, this adaptive mechanism occasionally achieves performance gains over dense models, offering a novel paradigm for adaptive pruning. During the KV cache pruning stage, based on observations of visual information degradation, SharpV prunes degraded visual features via a self-calibration manner, guided by similarity to original visual features. In this way, SharpV achieves hierarchical cache pruning from the perspective of information bottleneck, offering a new insight into VideoLLMs' information flow. Experiments on multiple public benchmarks demonstrate the superiority of SharpV. Moreover, to the best of our knowledge, SharpV is notably the first two-stage pruning framework that operates without requiring access to exposed attention scores, ensuring full compatibility with hardware acceleration techniques like Flash Attention.
>
---
#### [new 116] Two Datasets Are Better Than One: Method of Double Moments for 3-D Reconstruction in Cryo-EM
- **分类: cs.CV; math.NA; stat.ME**

- **简介: 该论文提出“双矩法”（MoDM），用于冷冻电镜三维重构，通过融合两种不同取向分布的二阶矩数据，仅用二阶统计量即可唯一恢复分子结构，提升重建质量，解决噪声环境下重构精度低的问题。**

- **链接: []()**

> **作者:** Joe Kileel; Oscar Mickelin; Amit Singer; Sheng Xu
>
> **摘要:** Cryo-electron microscopy (cryo-EM) is a powerful imaging technique for reconstructing three-dimensional molecular structures from noisy tomographic projection images of randomly oriented particles. We introduce a new data fusion framework, termed the method of double moments (MoDM), which reconstructs molecular structures from two instances of the second-order moment of projection images obtained under distinct orientation distributions--one uniform, the other non-uniform and unknown. We prove that these moments generically uniquely determine the underlying structure, up to a global rotation and reflection, and we develop a convex-relaxation-based algorithm that achieves accurate recovery using only second-order statistics. Our results demonstrate the advantage of collecting and modeling multiple datasets under different experimental conditions, illustrating that leveraging dataset diversity can substantially enhance reconstruction quality in computational imaging tasks.
>
---
#### [new 117] SynWeather: Weather Observation Data Synthesis across Multiple Regions and Variables via a General Diffusion Transformer
- **分类: cs.CV**

- **简介: 论文提出SynWeather数据集与SynWeatherDiff模型，首次实现多区域、多变量气象数据的统一概率合成，解决传统方法单变量、单区域局限及结果过平滑问题，基于扩散变换器提升合成质量。**

- **链接: []()**

> **作者:** Kaiyi Xu; Junchao Gong; Zhiwang Zhou; Zhangrui Li; Yuandong Pu; Yihao Liu; Ben Fei; Fenghua Ling; Wenlong Zhang; Lei Bei
>
> **摘要:** With the advancement of meteorological instruments, abundant data has become available. Current approaches are typically focus on single-variable, single-region tasks and primarily rely on deterministic modeling. This limits unified synthesis across variables and regions, overlooks cross-variable complementarity and often leads to over-smoothed results. To address above challenges, we introduce SynWeather, the first dataset designed for Unified Multi-region and Multi-variable Weather Observation Data Synthesis. SynWeather covers four representative regions: the Continental United States, Europe, East Asia, and Tropical Cyclone regions, as well as provides high-resolution observations of key weather variables, including Composite Radar Reflectivity, Hourly Precipitation, Visible Light, and Microwave Brightness Temperature. In addition, we introduce SynWeatherDiff, a general and probabilistic weather synthesis model built upon the Diffusion Transformer framework to address the over-smoothed problem. Experiments on the SynWeather dataset demonstrate the effectiveness of our network compared with both task-specific and general models.
>
---
#### [new 118] SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control
- **分类: cs.RO; cs.AI; cs.CV; cs.GR; eess.SY**

- **简介: 论文提出SONIC，通过大规模扩展模型、数据与算力，将运动追踪作为通用任务，构建能自然控制人形机器人的基础模型，支持多模态输入与实时任务执行，无需人工奖励设计。**

- **链接: []()**

> **作者:** Zhengyi Luo; Ye Yuan; Tingwu Wang; Chenran Li; Sirui Chen; Fernando Castañeda; Zi-Ang Cao; Jiefeng Li; David Minor; Qingwei Ben; Xingye Da; Runyu Ding; Cyrus Hogg; Lina Song; Edy Lim; Eugene Jeong; Tairan He; Haoru Xue; Wenli Xiao; Zi Wang; Simon Yuen; Jan Kautz; Yan Chang; Umar Iqbal; Linxi "Jim" Fan; Yuke Zhu
>
> **备注:** Project page: https://nvlabs.github.io/SONIC/
>
> **摘要:** Despite the rise of billion-parameter foundation models trained across thousands of GPUs, similar scaling gains have not been shown for humanoid control. Current neural controllers for humanoids remain modest in size, target a limited behavior set, and are trained on a handful of GPUs over several days. We show that scaling up model capacity, data, and compute yields a generalist humanoid controller capable of creating natural and robust whole-body movements. Specifically, we posit motion tracking as a natural and scalable task for humanoid control, leverageing dense supervision from diverse motion-capture data to acquire human motion priors without manual reward engineering. We build a foundation model for motion tracking by scaling along three axes: network size (from 1.2M to 42M parameters), dataset volume (over 100M frames, 700 hours of high-quality motion data), and compute (9k GPU hours). Beyond demonstrating the benefits of scale, we show the practical utility of our model through two mechanisms: (1) a real-time universal kinematic planner that bridges motion tracking to downstream task execution, enabling natural and interactive control, and (2) a unified token space that supports various motion input interfaces, such as VR teleoperation devices, human videos, and vision-language-action (VLA) models, all using the same policy. Scaling motion tracking exhibits favorable properties: performance improves steadily with increased compute and data diversity, and learned representations generalize to unseen motions, establishing motion tracking at scale as a practical foundation for humanoid control.
>
---
#### [new 119] A Hybrid Multimodal Deep Learning Framework for Intelligent Fashion Recommendation
- **分类: cs.IR; cs.CV**

- **简介: 该论文提出一种混合多模态深度学习框架，用于时尚推荐，解决搭配预测与互补品检索问题。基于CLIP提取图文特征，引入专用token通过Transformer建模项间关系，在Polyvore数据集上实现高精度。**

- **链接: []()**

> **作者:** Kamand Kalashi; Babak Teimourpour
>
> **备注:** 8 pages, 1 figure
>
> **摘要:** The rapid expansion of online fashion platforms has created an increasing demand for intelligent recommender systems capable of understanding both visual and textual cues. This paper proposes a hybrid multimodal deep learning framework for fashion recommendation that jointly addresses two key tasks: outfit compatibility prediction and complementary item retrieval. The model leverages the visual and textual encoders of the CLIP architecture to obtain joint latent representations of fashion items, which are then integrated into a unified feature vector and processed by a transformer encoder. For compatibility prediction, an "outfit token" is introduced to model the holistic relationships among items, achieving an AUC of 0.95 on the Polyvore dataset. For complementary item retrieval, a "target item token" representing the desired item description is used to retrieve compatible items, reaching an accuracy of 69.24% under the Fill-in-the-Blank (FITB) metric. The proposed approach demonstrates strong performance across both tasks, highlighting the effectiveness of multimodal learning for fashion recommendation.
>
---
#### [new 120] The Online Patch Redundancy Eliminator (OPRE): A novel approach to online agnostic continual learning using dataset compression
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出OPRE，一种在线数据压缩方法，用于解决持续学习中的灾难性遗忘问题，实现无先验假设的通用持续学习，在CIFAR数据集上超越现有方法，强调数据压缩对真正agnostic持续学习的必要性。**

- **链接: []()**

> **作者:** Raphaël Bayle; Martial Mermillod; Robert M. French
>
> **摘要:** In order to achieve Continual Learning (CL), the problem of catastrophic forgetting, one that has plagued neural networks since their inception, must be overcome. The evaluation of continual learning methods relies on splitting a known homogeneous dataset and learning the associated tasks one after the other. We argue that most CL methods introduce a priori information about the data to come and cannot be considered agnostic. We exemplify this point with the case of methods relying on pretrained feature extractors, which are still used in CL. After showing that pretrained feature extractors imply a loss of generality with respect to the data that can be learned by the model, we then discuss other kinds of a priori information introduced in other CL methods. We then present the Online Patch Redundancy Eliminator (OPRE), an online dataset compression algorithm, which, along with the training of a classifier at test time, yields performance on CIFAR-10 and CIFAR-100 superior to a number of other state-of-the-art online continual learning methods. Additionally, OPRE requires only minimal and interpretable hypothesis on the data to come. We suggest that online dataset compression could well be necessary to achieve fully agnostic CL.
>
---
#### [new 121] Towards Personalized Quantum Federated Learning for Anomaly Detection
- **分类: cs.LG; cs.CV; quant-ph**

- **简介: 该论文提出个性化量子联邦学习（PQFL）框架，用于异构量子客户端的异常检测任务，解决传统QFL因硬件与数据异构导致模型失效问题，通过量子参数化电路与个性化策略提升检测精度。**

- **链接: []()**

> **作者:** Ratun Rahman; Sina Shaham; Dinh C. Nguyen
>
> **备注:** Accepted at IEEE Transactions on Network Science and Engineering
>
> **摘要:** Anomaly detection has a significant impact on applications such as video surveillance, medical diagnostics, and industrial monitoring, where anomalies frequently depend on context and anomaly-labeled data are limited. Quantum federated learning (QFL) overcomes these concerns by distributing model training among several quantum clients, consequently eliminating the requirement for centralized quantum storage and processing. However, in real-life quantum networks, clients frequently differ in terms of hardware capabilities, circuit designs, noise levels, and how classical data is encoded or preprocessed into quantum states. These differences create inherent heterogeneity across clients - not just in their data distributions, but also in their quantum processing behaviors. As a result, training a single global model becomes ineffective, especially when clients handle imbalanced or non-identically distributed (non-IID) data. To address this, we propose a new framework called personalized quantum federated learning (PQFL) for anomaly detection. PQFL enhances local model training at quantum clients using parameterized quantum circuits and classical optimizers, while introducing a quantum-centric personalization strategy that adapts each client's model to its own hardware characteristics and data representation. Extensive experiments show that PQFL significantly improves anomaly detection accuracy under diverse and realistic conditions. Compared to state-of-the-art methods, PQFL reduces false errors by up to 23%, and achieves gains of 24.2% in AUROC and 20.5% in AUPR, highlighting its effectiveness and scalability in practical quantum federated settings.
>
---
#### [new 122] ViPRA: Video Prediction for Robot Actions
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: ViPRA提出一种预训练-微调框架，利用无动作标注的视频学习机器人连续控制。通过视频语言模型预测视觉观测与物理一致的潜在动作，再用少量演示映射为机器人动作，实现高效泛化与高频控制。**

- **链接: []()**

> **作者:** Sandeep Routray; Hengkai Pan; Unnat Jain; Shikhar Bahl; Deepak Pathak
>
> **备注:** Website: https://vipra-project.github.io
>
> **摘要:** Can we turn a video prediction model into a robot policy? Videos, including those of humans or teleoperated robots, capture rich physical interactions. However, most of them lack labeled actions, which limits their use in robot learning. We present Video Prediction for Robot Actions (ViPRA), a simple pretraining-finetuning framework that learns continuous robot control from these actionless videos. Instead of directly predicting actions, we train a video-language model to predict both future visual observations and motion-centric latent actions, which serve as intermediate representations of scene dynamics. We train these latent actions using perceptual losses and optical flow consistency to ensure they reflect physically grounded behavior. For downstream control, we introduce a chunked flow matching decoder that maps latent actions to robot-specific continuous action sequences, using only 100 to 200 teleoperated demonstrations. This approach avoids expensive action annotation, supports generalization across embodiments, and enables smooth, high-frequency continuous control upto 22 Hz via chunked action decoding. Unlike prior latent action works that treat pretraining as autoregressive policy learning, explicitly models both what changes and how. Our method outperforms strong baselines, with a 16% gain on the SIMPLER benchmark and a 13% improvement across real world manipulation tasks. We will release models and code at https://vipra-project.github.io
>
---
#### [new 123] NeuCLIP: Efficient Large-Scale CLIP Training with Neural Normalizer Optimization
- **分类: cs.LG; cs.CV**

- **简介: NeuCLIP针对CLIP训练中对比损失归一化项估算不准的问题，提出通过凸分析和变分分析将归一化项建模为神经网络预测任务，联合优化CLIP模型与辅助网络，显著提升大尺度训练效率与精度。**

- **链接: []()**

> **作者:** Xiyuan Wei; Chih-Jen Lin; Tianbao Yang
>
> **备注:** 20 pages, 4 figures
>
> **摘要:** Accurately estimating the normalization term (also known as the partition function) in the contrastive loss is a central challenge for training Contrastive Language-Image Pre-training (CLIP) models. Conventional methods rely on large batches for approximation, demanding substantial computational resources. To mitigate this issue, prior works introduced per-sample normalizer estimators, which are updated at each epoch in a blockwise coordinate manner to keep track of updated encoders. However, this scheme incurs optimization error that scales with the ratio of dataset size to batch size, limiting effectiveness for large datasets or small batches. To overcome this limitation, we propose NeuCLIP, a novel and elegant optimization framework based on two key ideas: (i) $\textbf{reformulating}$ the contrastive loss for each sample $\textbf{via convex analysis}$ into a minimization problem with an auxiliary variable representing its log-normalizer; and (ii) $\textbf{transforming}$ the resulting minimization over $n$ auxiliary variables (where $n$ is the dataset size) via $\textbf{variational analysis}$ into the minimization over a compact neural network that predicts the log-normalizers. We design an alternating optimization algorithm that jointly trains the CLIP model and the auxiliary network. By employing a tailored architecture and acceleration techniques for the auxiliary network, NeuCLIP achieves more accurate normalizer estimation, leading to improved performance compared with previous methods. Extensive experiments on large-scale CLIP training, spanning datasets from millions to billions of samples, demonstrate that NeuCLIP outperforms previous methods.
>
---
#### [new 124] Aligning by Misaligning: Boundary-aware Curriculum Learning for Multimodal Alignment
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对多模态对齐任务，解决负样本中模糊边界案例被忽略的问题。提出BACL框架，通过边界感知负采样与局部对比注意力，自适应提升训练难度，无需额外标签即显著提升对齐性能。**

- **链接: []()**

> **作者:** Hua Ye; Hang Ding; Siyuan Chen; Yiyang Jiang; Changyuan Zhang; Xuan Zhang
>
> **备注:** 24 pages, 6 figures, 5 tables. Submitted to NeurIPS 2025
>
> **摘要:** Most multimodal models treat every negative pair alike, ignoring the ambiguous negatives that differ from the positive by only a small detail. We propose Boundary-Aware Curriculum with Local Attention (BACL), a lightweight add-on that turns these borderline cases into a curriculum signal. A Boundary-aware Negative Sampler gradually raises difficulty, while a Contrastive Local Attention loss highlights where the mismatch occurs. The two modules are fully differentiable and work with any off-the-shelf dual encoder. Theory predicts a fast O(1/n) error rate; practice shows up to +32% R@1 over CLIP and new SOTA on four large-scale benchmarks, all without extra labels.
>
---
#### [new 125] Operational machine learning for remote spectroscopic detection of CH$_{4}$ point sources
- **分类: cs.AI; cs.CV**

- **简介: 该论文面向卫星光谱数据的甲烷泄漏检测任务，解决传统方法误报率高的问题，构建了全球最大标注数据集，提出基于集成学习的深度学习模型，部署于联合国MARS系统，显著降低误报并加速泄漏验证。**

- **链接: []()**

> **作者:** Vít Růžička; Gonzalo Mateo-García; Itziar Irakulis-Loitxate; Juan Emmanuel Johnson; Manuel Montesino San Martín; Anna Allen; Luis Guanter; David R. Thompson
>
> **备注:** 14 pages, 12 figures, 5 tables. In review
>
> **摘要:** Mitigating anthropogenic methane sources is one the most cost-effective levers to slow down global warming. While satellite-based imaging spectrometers, such as EMIT, PRISMA, and EnMAP, can detect these point sources, current methane retrieval methods based on matched filters still produce a high number of false detections requiring laborious manual verification. This paper describes the operational deployment of a machine learning system for detecting methane emissions within the Methane Alert and Response System (MARS) of the United Nations Environment Programme's International Methane Emissions Observatory. We created the largest and most diverse global dataset of annotated methane plumes from three imaging spectrometer missions and quantitatively compared different deep learning model configurations. Focusing on the requirements for operational deployment, we extended prior evaluation methodologies from small tiled datasets to full granule evaluation. This revealed that deep learning models still produce a large number of false detections, a problem we address with model ensembling, which reduced false detections by over 74%. Deployed in the MARS pipeline, our system processes scenes and proposes plumes to analysts, accelerating the detection and analysis process. During seven months of operational deployment, it facilitated the verification of 1,351 distinct methane leaks, resulting in 479 stakeholder notifications. We further demonstrate the model's utility in verifying mitigation success through case studies in Libya, Argentina, Oman, and Azerbaijan. Our work represents a critical step towards a global AI-assisted methane leak detection system, which is required to process the dramatically higher data volumes expected from new and current imaging spectrometers.
>
---
#### [new 126] RoboTAG: End-to-end Robot Configuration Estimation via Topological Alignment Graph
- **分类: cs.RO; cs.CV**

- **简介: 论文提出RoboTAG，用于从单目图像估计机器人位姿，解决标注数据稀缺与3D先验缺失问题。通过双分支图结构融合2D/3D表示，利用拓扑一致性实现无监督训练，提升跨机器人类型泛化能力。**

- **链接: []()**

> **作者:** Yifan Liu; Fangneng Zhan; Wanhua Li; Haowen Sun; Katerina Fragkiadaki; Hanspeter Pfister
>
> **摘要:** Estimating robot pose from a monocular RGB image is a challenge in robotics and computer vision. Existing methods typically build networks on top of 2D visual backbones and depend heavily on labeled data for training, which is often scarce in real-world scenarios, causing a sim-to-real gap. Moreover, these approaches reduce the 3D-based problem to 2D domain, neglecting the 3D priors. To address these, we propose Robot Topological Alignment Graph (RoboTAG), which incorporates a 3D branch to inject 3D priors while enabling co-evolution of the 2D and 3D representations, alleviating the reliance on labels. Specifically, the RoboTAG consists of a 3D branch and a 2D branch, where nodes represent the states of the camera and robot system, and edges capture the dependencies between these variables or denote alignments between them. Closed loops are then defined in the graph, on which a consistency supervision across branches can be applied. This design allows us to utilize in-the-wild images as training data without annotations. Experimental results demonstrate that our method is effective across robot types, highlighting its potential to alleviate the data bottleneck in robotics.
>
---
#### [new 127] EvoPS: Evolutionary Patch Selection for Whole Slide Image Analysis in Computational Pathology
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: EvoPS提出一种进化算法，用于在病理全切片图像中优化补丁选择，平衡计算效率与诊断精度，通过多目标优化减少90%以上补丁数量，同时保持或提升分类性能。**

- **链接: []()**

> **作者:** Saya Hashemian; Azam Asilian Bidgoli
>
> **摘要:** In computational pathology, the gigapixel scale of Whole-Slide Images (WSIs) necessitates their division into thousands of smaller patches. Analyzing these high-dimensional patch embeddings is computationally expensive and risks diluting key diagnostic signals with many uninformative patches. Existing patch selection methods often rely on random sampling or simple clustering heuristics and typically fail to explicitly manage the crucial trade-off between the number of selected patches and the accuracy of the resulting slide representation. To address this gap, we propose EvoPS (Evolutionary Patch Selection), a novel framework that formulates patch selection as a multi-objective optimization problem and leverages an evolutionary search to simultaneously minimize the number of selected patch embeddings and maximize the performance of a downstream similarity search task, generating a Pareto front of optimal trade-off solutions. We validated our framework across four major cancer cohorts from The Cancer Genome Atlas (TCGA) using five pretrained deep learning models to generate patch embeddings, including both supervised CNNs and large self-supervised foundation models. The results demonstrate that EvoPS can reduce the required number of training patch embeddings by over 90% while consistently maintaining or even improving the final classification F1-score compared to a baseline that uses all available patches' embeddings selected through a standard extraction pipeline. The EvoPS framework provides a robust and principled method for creating efficient, accurate, and interpretable WSI representations, empowering users to select an optimal balance between computational cost and diagnostic performance.
>
---
#### [new 128] Re$^{\text{2}}$MaP: Macro Placement by Recursively Prototyping and Packing Tree-based Relocating
- **分类: cs.AR; cs.CV; eess.SY**

- **简介: 论文提出Re²MaP方法，解决芯片宏模块布局优化问题，通过递归原型构建与基于椭圆分布的角分析布局及打包树重定位，显著提升布线长度、时序和功耗性能，优于现有方法。**

- **链接: []()**

> **作者:** Yunqi Shi; Xi Lin; Zhiang Wang; Siyuan Xu; Shixiong Kai; Yao Lai; Chengrui Gao; Ke Xue; Mingxuan Yuan; Chao Qian; Zhi-Hua Zhou
>
> **备注:** IEEE Transactions on Comupter-Aided Design under review
>
> **摘要:** This work introduces the Re$^{\text{2}}$MaP method, which generates expert-quality macro placements through recursively prototyping and packing tree-based relocating. We first perform multi-level macro grouping and PPA-aware cell clustering to produce a unified connection matrix that captures both wirelength and dataflow among macros and clusters. Next, we use DREAMPlace to build a mixed-size placement prototype and obtain reference positions for each macro and cluster. Based on this prototype, we introduce ABPlace, an angle-based analytical method that optimizes macro positions on an ellipse to distribute macros uniformly near chip periphery, while optimizing wirelength and dataflow. A packing tree-based relocating procedure is then designed to jointly adjust the locations of macro groups and the macros within each group, by optimizing an expertise-inspired cost function that captures various design constraints through evolutionary search. Re$^{\text{2}}$MaP repeats the above process: Only a subset of macro groups are positioned in each iteration, and the remaining macros are deferred to the next iteration to improve the prototype's accuracy. Using a well-established backend flow with sufficient timing optimizations, Re$^{\text{2}}$MaP achieves up to 22.22% (average 10.26%) improvement in worst negative slack (WNS) and up to 97.91% (average 33.97%) improvement in total negative slack (TNS) compared to the state-of-the-art academic placer Hier-RTLMP. It also ranks higher on WNS, TNS, power, design rule check (DRC) violations, and runtime than the conference version ReMaP, across seven tested cases. Our code is available at https://github.com/lamda-bbo/Re2MaP.
>
---
#### [new 129] From Noise to Latent: Generating Gaussian Latents for INR-Based Image Compression
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出一种基于高斯噪声生成隐式神经表示（INR）潜变量的图像压缩方法，无需传输潜码，通过高斯参数预测模块实现确定性潜变量生成，兼顾压缩效率与重建质量，首次探索高斯潜变量生成在学习型图像压缩中的应用。**

- **链接: []()**

> **作者:** Chaoyi Lin; Yaojun Wu; Yue Li; Junru Li; Kai Zhang; Li Zhang
>
> **摘要:** Recent implicit neural representation (INR)-based image compression methods have shown competitive performance by overfitting image-specific latent codes. However, they remain inferior to end-to-end (E2E) compression approaches due to the absence of expressive latent representations. On the other hand, E2E methods rely on transmitting latent codes and requiring complex entropy models, leading to increased decoding complexity. Inspired by the normalization strategy in E2E codecs where latents are transformed into Gaussian noise to demonstrate the removal of spatial redundancy, we explore the inverse direction: generating latents directly from Gaussian noise. In this paper, we propose a novel image compression paradigm that reconstructs image-specific latents from a multi-scale Gaussian noise tensor, deterministically generated using a shared random seed. A Gaussian Parameter Prediction (GPP) module estimates the distribution parameters, enabling one-shot latent generation via reparameterization trick. The predicted latent is then passed through a synthesis network to reconstruct the image. Our method eliminates the need to transmit latent codes while preserving latent-based benefits, achieving competitive rate-distortion performance on Kodak and CLIC dataset. To the best of our knowledge, this is the first work to explore Gaussian latent generation for learned image compression.
>
---
#### [new 130] Multivariate Variational Autoencoder
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出多变量变分自编码器（MVAE），解决传统VAE后验高斯分布对角限制问题，通过全局耦合矩阵与样本级缩放实现全协方差后验，在保持解析KL散度的同时提升重建、校准与结构学习性能。**

- **链接: []()**

> **作者:** Mehmet Can Yavuz
>
> **摘要:** We present the Multivariate Variational Autoencoder (MVAE), a VAE variant that preserves Gaussian tractability while lifting the diagonal posterior restriction. MVAE factorizes each posterior covariance, where a \emph{global} coupling matrix $\mathbf{C}$ induces dataset-wide latent correlations and \emph{per-sample} diagonal scales modulate local uncertainty. This yields a full-covariance family with analytic KL and an efficient reparameterization via $\mathbf{L}=\mathbf{C}\mathrm{diag}(\boldsymbolσ)$. Across Larochelle-style MNIST variants, Fashion-MNIST, CIFAR-10, and CIFAR-100, MVAE consistently matches or improves reconstruction (MSE~$\downarrow$) and delivers robust gains in calibration (NLL/Brier/ECE~$\downarrow$) and unsupervised structure (NMI/ARI~$\uparrow$) relative to diagonal-covariance VAEs with matched capacity, especially at mid-range latent sizes. Latent-plane visualizations further indicate smoother, more coherent factor traversals and sharper local detail. We release a fully reproducible implementation with training/evaluation scripts and sweep utilities to facilitate fair comparison and reuse.
>
---
#### [new 131] DynaQuant: Dynamic Mixed-Precision Quantization for Learned Image Compression
- **分类: eess.IV; cs.CV**

- **简介: 论文提出DynaQuant，用于学习图像压缩中的动态混合精度量化，解决静态量化无法适应特征分布差异的问题。通过内容感知量化与动态位宽选择，实现端到端自适应，显著降低计算开销而不牺牲重建质量。**

- **链接: []()**

> **作者:** Youneng Bao; Yulong Cheng; Yiping Liu; Yichen Yang; Peng Qin; Mu Li; Yongsheng Liang
>
> **备注:** 13 pages,accepted by AAAI 2026
>
> **摘要:** Prevailing quantization techniques in Learned Image Compression (LIC) typically employ a static, uniform bit-width across all layers, failing to adapt to the highly diverse data distributions and sensitivity characteristics inherent in LIC models. This leads to a suboptimal trade-off between performance and efficiency. In this paper, we introduce DynaQuant, a novel framework for dynamic mixed-precision quantization that operates on two complementary levels. First, we propose content-aware quantization, where learnable scaling and offset parameters dynamically adapt to the statistical variations of latent features. This fine-grained adaptation is trained end-to-end using a novel Distance-aware Gradient Modulator (DGM), which provides a more informative learning signal than the standard Straight-Through Estimator. Second, we introduce a data-driven, dynamic bit-width selector that learns to assign an optimal bit precision to each layer, dynamically reconfiguring the network's precision profile based on the input data. Our fully dynamic approach offers substantial flexibility in balancing rate-distortion (R-D) performance and computational cost. Experiments demonstrate that DynaQuant achieves rd performance comparable to full-precision models while significantly reducing computational and storage requirements, thereby enabling the practical deployment of advanced LIC on diverse hardware platforms.
>
---
#### [new 132] Simulating the Visual World with Artificial Intelligence: A Roadmap
- **分类: cs.AI; cs.CV**

- **简介: 该论文提出视频基础模型是“隐式世界模型+视频渲染器”的组合，旨在构建具备物理合理性、交互与规划能力的虚拟世界模拟系统，系统梳理四代演进，明确未来设计原则与挑战。**

- **链接: []()**

> **作者:** Jingtong Yue; Ziqi Huang; Zhaoxi Chen; Xintao Wang; Pengfei Wan; Ziwei Liu
>
> **备注:** Project page: https://world-model-roadmap.github.io/ Github Repo: https://github.com/ziqihuangg/Awesome-From-Video-Generation-to-World-Model
>
> **摘要:** The landscape of video generation is shifting, from a focus on generating visually appealing clips to building virtual environments that support interaction and maintain physical plausibility. These developments point toward the emergence of video foundation models that function not only as visual generators but also as implicit world models, models that simulate the physical dynamics, agent-environment interactions, and task planning that govern real or imagined worlds. This survey provides a systematic overview of this evolution, conceptualizing modern video foundation models as the combination of two core components: an implicit world model and a video renderer. The world model encodes structured knowledge about the world, including physical laws, interaction dynamics, and agent behavior. It serves as a latent simulation engine that enables coherent visual reasoning, long-term temporal consistency, and goal-driven planning. The video renderer transforms this latent simulation into realistic visual observations, effectively producing videos as a "window" into the simulated world. We trace the progression of video generation through four generations, in which the core capabilities advance step by step, ultimately culminating in a world model, built upon a video generation model, that embodies intrinsic physical plausibility, real-time multimodal interaction, and planning capabilities spanning multiple spatiotemporal scales. For each generation, we define its core characteristics, highlight representative works, and examine their application domains such as robotics, autonomous driving, and interactive gaming. Finally, we discuss open challenges and design principles for next-generation world models, including the role of agent intelligence in shaping and evaluating these systems. An up-to-date list of related works is maintained at this link.
>
---
#### [new 133] Class-feature Watermark: A Resilient Black-box Watermark Against Model Extraction Attacks
- **分类: cs.CR; cs.CV; cs.LG**

- **简介: 该论文针对模型提取攻击中的水印易被移除问题，提出Class-Feature Watermark（CFW），通过构造域外合成类消除脆弱决策边界，显著提升水印在提取模型中的抗移除性与稳定性，同时保留模型性能。**

- **链接: []()**

> **作者:** Yaxin Xiao; Qingqing Ye; Zi Liang; Haoyang Li; RongHua Li; Huadi Zheng; Haibo Hu
>
> **备注:** Accepted by AAAI'26
>
> **摘要:** Machine learning models constitute valuable intellectual property, yet remain vulnerable to model extraction attacks (MEA), where adversaries replicate their functionality through black-box queries. Model watermarking counters MEAs by embedding forensic markers for ownership verification. Current black-box watermarks prioritize MEA survival through representation entanglement, yet inadequately explore resilience against sequential MEAs and removal attacks. Our study reveals that this risk is underestimated because existing removal methods are weakened by entanglement. To address this gap, we propose Watermark Removal attacK (WRK), which circumvents entanglement constraints by exploiting decision boundaries shaped by prevailing sample-level watermark artifacts. WRK effectively reduces watermark success rates by at least 88.79% across existing watermarking benchmarks. For robust protection, we propose Class-Feature Watermarks (CFW), which improve resilience by leveraging class-level artifacts. CFW constructs a synthetic class using out-of-domain samples, eliminating vulnerable decision boundaries between original domain samples and their artifact-modified counterparts (watermark samples). CFW concurrently optimizes both MEA transferability and post-MEA stability. Experiments across multiple domains show that CFW consistently outperforms prior methods in resilience, maintaining a watermark success rate of at least 70.15% in extracted models even under the combined MEA and WRK distortion, while preserving the utility of protected models.
>
---
#### [new 134] On the Role of Calibration in Benchmarking Algorithmic Fairness for Skin Cancer Detection
- **分类: cs.LG; cs.CV; eess.IV**

- **简介: 该论文研究皮肤癌AI诊断中的公平性问题，提出将校准（calibration）作为与AUROC并行的公平性评估指标，发现现有模型在跨群体中存在风险高估与校准偏差，呼吁加强模型审计与数据收集以实现公平医疗AI。**

- **链接: []()**

> **作者:** Brandon Dominique; Prudence Lam; Nicholas Kurtansky; Jochen Weber; Kivanc Kose; Veronica Rotemberg; Jennifer Dy
>
> **备注:** 19 pages, 4 figures. Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) https://melba-journal.org/2025:027
>
> **摘要:** Artificial Intelligence (AI) models have demonstrated expert-level performance in melanoma detection, yet their clinical adoption is hindered by performance disparities across demographic subgroups such as gender, race, and age. Previous efforts to benchmark the performance of AI models have primarily focused on assessing model performance using group fairness metrics that rely on the Area Under the Receiver Operating Characteristic curve (AUROC), which does not provide insights into a model's ability to provide accurate estimates. In line with clinical assessments, this paper addresses this gap by incorporating calibration as a complementary benchmarking metric to AUROC-based fairness metrics. Calibration evaluates the alignment between predicted probabilities and observed event rates, offering deeper insights into subgroup biases. We assess the performance of the leading skin cancer detection algorithm of the ISIC 2020 Challenge on the ISIC 2020 Challenge dataset and the PROVE-AI dataset, and compare it with the second and third place models, focusing on subgroups defined by sex, race (Fitzpatrick Skin Tone), and age. Our findings reveal that while existing models enhance discriminative accuracy, they often over-diagnose risk and exhibit calibration issues when applied to new datasets. This study underscores the necessity for comprehensive model auditing strategies and extensive metadata collection to achieve equitable AI-driven healthcare solutions. All code is publicly available at https://github.com/bdominique/testing_strong_calibration.
>
---
#### [new 135] LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **简介: 该论文提出LeJEPA，一种理论严谨的自监督学习框架，解决JEPA缺乏理论指导与工程复杂性问题，通过Sketched Isotropic Gaussian正则化实现无启发式、可扩展、稳定训练，显著简化架构并提升性能。**

- **链接: []()**

> **作者:** Randall Balestriero; Yann LeCun
>
> **摘要:** Learning manipulable representations of the world and its dynamics is central to AI. Joint-Embedding Predictive Architectures (JEPAs) offer a promising blueprint, but lack of practical guidance and theory has led to ad-hoc R&D. We present a comprehensive theory of JEPAs and instantiate it in {\bf LeJEPA}, a lean, scalable, and theoretically grounded training objective. First, we identify the isotropic Gaussian as the optimal distribution that JEPAs' embeddings should follow to minimize downstream prediction risk. Second, we introduce a novel objective--{\bf Sketched Isotropic Gaussian Regularization} (SIGReg)--to constrain embeddings to reach that ideal distribution. Combining the JEPA predictive loss with SIGReg yields LeJEPA with numerous theoretical and practical benefits: (i) single trade-off hyperparameter, (ii) linear time and memory complexity, (iii) stability across hyper-parameters, architectures (ResNets, ViTs, ConvNets) and domains, (iv) heuristics-free, e.g., no stop-gradient, no teacher-student, no hyper-parameter schedulers, and (v) distributed training-friendly implementation requiring only $\approx$50 lines of code. Our empirical validation covers 10+ datasets, 60+ architectures, all with varying scales and domains. As an example, using imagenet-1k for pretraining and linear evaluation with frozen backbone, LeJEPA reaches 79\% with a ViT-H/14. We hope that the simplicity and theory-friendly ecosystem offered by LeJEPA will reestablish self-supervised pre-training as a core pillar of AI research (\href{git@github.com:rbalestr-lab/lejepa.git}{GitHub repo}).
>
---
#### [new 136] CNN-Based Automated Parameter Extraction Framework for Modeling Memristive Devices
- **分类: cs.ET; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出一种基于CNN的自动化框架，用于从RRAM器件I-V特性中快速提取斯坦福模型参数，解决传统手动调参耗时、泛化差的问题，通过深度学习与启发式优化实现高效、精准的参数提取。**

- **链接: []()**

> **作者:** Akif Hamid; Orchi Hassan
>
> **摘要:** Resistive random access memory (RRAM) is a promising candidate for next-generation nonvolatile memory (NVM) and in-memory computing applications. Compact models are essential for analyzing the circuit and system-level performance of experimental RRAM devices. However, most existing RRAM compact models rely on multiple fitting parameters to reproduce the device I-V characteristics, and in most cases, as the parameters are not directly related to measurable quantities, their extraction requires extensive manual tuning, making the process time-consuming and limiting adaptability across different devices. This work presents an automated framework for extracting the fitting parameters of the widely used Stanford RRAM model directly from the device I-V characteristics. The framework employs a convolutional neural network (CNN) trained on a synthetic dataset to generate initial parameter estimates, which are then refined through three heuristic optimization blocks that minimize errors via adaptive binary search in the parameter space. We evaluated the framework using four key NVM metrics: set voltage, reset voltage, hysteresis loop area, and low resistance state (LRS) slope. Benchmarking against RRAM device characteristics derived from previously reported Stanford model fits, other analytical models, and experimental data shows that the framework achieves low error across diverse device characteristics, offering a fast, reliable, and robust solution for RRAM modeling.
>
---
#### [new 137] IBMA: An Imputation-Based Mixup Augmentation Using Self-Supervised Learning for Time Series Data
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出IBMA，一种基于自监督插补与Mixup的数据增强方法，用于时间序列预测任务，解决传统增强策略稀缺问题。实验表明其在多个模型和数据集上显著提升预测性能。**

- **链接: []()**

> **作者:** Dang Nha Nguyen; Hai Dang Nguyen; Khoa Tho Anh Nguyen
>
> **备注:** 9 pages, 1 figure, 1 table, accepted at the AAAI2025 conference
>
> **摘要:** Data augmentation in time series forecasting plays a crucial role in enhancing model performance by introducing variability while maintaining the underlying temporal patterns. However, time series data offers fewer augmentation strategies compared to fields such as image or text, with advanced techniques like Mixup rarely being used. In this work, we propose a novel approach, Imputation-Based Mixup Augmentation (IBMA), which combines Imputation-Augmented data with Mixup augmentation to bolster model generalization and improve forecasting performance. We evaluate the effectiveness of this method across several forecasting models, including DLinear (MLP), TimesNet (CNN), and iTrainformer (Transformer), these models represent some of the most recent advances in time series forecasting. Our experiments, conducted on four datasets (ETTh1, ETTh2, ETTm1, ETTm2) and compared against eight other augmentation techniques, demonstrate that IBMA consistently enhances performance, achieving 22 improvements out of 24 instances, with 10 of those being the best performances, particularly with iTrainformer imputation.
>
---
#### [new 138] From Exploration to Exploitation: A Two-Stage Entropy RLVR Approach for Noise-Tolerant MLLM Training
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对多模态大语言模型（MLLM）在噪声标签下强化学习训练的难题，提出一种两阶段熵优化方法：先最大化熵以探索多样性、抗噪声，再最小化熵以巩固准确预测，提升GRPO奖励排序可靠性。**

- **链接: []()**

> **作者:** Donglai Xu; Hongzheng Yang; Yuzhi Zhao; Pingping Zhang; Jinpeng Chen; Wenao Ma; Zhijian Hou; Mengyang Wu; Xiaolei Li; Senkang Hu; Ziyi Guan; Jason Chun Lok Li; Lai Man Po
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) for Multimodal Large Language Models (MLLMs) is highly dependent on high-quality labeled data, which is often scarce and prone to substantial annotation noise in real-world scenarios. Existing unsupervised RLVR methods, including pure entropy minimization, can overfit to incorrect labels and limit the crucial reward ranking signal for Group-Relative Policy Optimization (GRPO). To address these challenges and enhance noise tolerance, we propose a novel two-stage, token-level entropy optimization method for RLVR. This approach dynamically guides the model from exploration to exploitation during training. In the initial exploration phase, token-level entropy maximization promotes diverse and stochastic output generation, serving as a strong regularizer that prevents premature convergence to noisy labels and ensures sufficient intra-group variation, which enables more reliable reward gradient estimation in GRPO. As training progresses, the method transitions into the exploitation phase, where token-level entropy minimization encourages the model to produce confident and deterministic outputs, thereby consolidating acquired knowledge and refining prediction accuracy. Empirically, across three MLLM backbones - Qwen2-VL-2B, Qwen2-VL-7B, and Qwen2.5-VL-3B - spanning diverse noise settings and multiple tasks, our phased strategy consistently outperforms prior approaches by unifying and enhancing external, internal, and entropy-based methods, delivering robust and superior performance across the board.
>
---
#### [new 139] Deep Learning Analysis of Prenatal Ultrasound for Identification of Ventriculomegaly
- **分类: eess.IV; cs.CV**

- **简介: 该论文利用深度学习构建USF-MAE模型，基于产前超声图像实现室管膜扩张的自动检测，解决早期诊断难题。通过预训练视觉变换器微调，在多组验证中表现优异，且可视化确认模型聚焦病变区域，具备临床可解释性。**

- **链接: []()**

> **作者:** Youssef Megahed; Inok Lee; Robin Ducharme; Aylin Erman; Olivier X. Miguel; Kevin Dick; Adrian D. C. Chan; Steven Hawken; Mark Walker; Felipe Moretti
>
> **备注:** 13 pages, 7 figures, 3 tables
>
> **摘要:** The proposed study aimed to develop a deep learning model capable of detecting ventriculomegaly on prenatal ultrasound images. Ventriculomegaly is a prenatal condition characterized by dilated cerebral ventricles of the fetal brain and is important to diagnose early, as it can be associated with an increased risk for fetal aneuploidies and/or underlying genetic syndromes. An Ultrasound Self-Supervised Foundation Model with Masked Autoencoding (USF-MAE), recently developed by our group, was fine-tuned for a binary classification task to distinguish fetal brain ultrasound images as either normal or showing ventriculomegaly. The USF-MAE incorporates a Vision Transformer encoder pretrained on more than 370,000 ultrasound images from the OpenUS-46 corpus. For this study, the pretrained encoder was adapted and fine-tuned on a curated dataset of fetal brain ultrasound images to optimize its performance for ventriculomegaly detection. Model evaluation was conducted using 5-fold cross-validation and an independent test cohort, and performance was quantified using accuracy, precision, recall, specificity, F1-score, and area under the receiver operating characteristic curve (AUC). The proposed USF-MAE model reached an F1-score of 91.76% on the 5-fold cross-validation and 91.78% on the independent test set, with much higher scores than those obtained by the baseline models by 19.37% and 16.15% compared to VGG-19, 2.31% and 2.56% compared to ResNet-50, and 5.03% and 11.93% compared to ViT-B/16, respectively. The model also showed a high mean test precision of 94.47% and an accuracy of 97.24%. The Eigen-CAM (Eigen Class Activation Map) heatmaps showed that the model was focusing on the ventricle area for the diagnosis of ventriculomegaly, which has explainability and clinical plausibility.
>
---
## 更新

#### [replaced 001] Multi Class Parkinson Disease Detection Based on Finger Tapping Using Attention Enhanced CNN BiLSTM
- **分类: cs.CV**

- **链接: []()**

> **作者:** Abu Saleh Musa Miah; Najmul Hassan; Md Maruf Al Hossain; Yuichi Okuyama; Jungpil Shin
>
> **摘要:** Accurate evaluation of Parkinsons disease (PD) severity is essential for effective clinical management and intervention development. Despite the proposal of several gesture based PD recognition systems, including those using the finger tapping task to assess Parkinsonian symptoms, their performance remains unsatisfactory. In this study, we present a multi class PD detection system based on finger-tapping, using an attention-enhanced CNN BiLSTM framework combined with handcrafted feature extraction and deep learning techniques. In the procedure, we used an existing dataset of finger tapping videos to extract temporal, frequency, and amplitude-based features from wrist and hand movements using their formulas. These handcrafted features were then processed through our attention enhanced CNN BiLSTM model, a hybrid deep learning framework that integrates CNN, BiLSTM, and attention mechanisms to classify PD severity into multiple levels. The features first pass through a Conv1D MaxPooling block to capture local spatial dependencies, followed by processing through a BiLSTM layer to model the temporal dynamics of the motion. An attention mechanism is applied to emphasize the most informative temporal features, which are then refined by a second BiLSTM layer. The CNN derived features and attention enhanced BiLSTM outputs are concatenated, followed by dense and dropout layers, before being passed through a softmax classifier to predict the PD severity level. Our model demonstrated strong performance in distinguishing between the five severity classes, showcasing the effectiveness of combining spatial temporal representations with attention mechanisms for automated PD severity detection. This approach offers a promising non invasive tool to assist clinicians in monitoring PD progression and making informed treatment decisions.
>
---
#### [replaced 002] A Large-scale Benchmark on Geological Fault Delineation Models: Domain Shift, Training Dynamics, Generalizability, Evaluation and Inferential Behavior
- **分类: cs.CV**

- **链接: []()**

> **作者:** Jorge Quesada; Chen Zhou; Prithwijit Chowdhury; Mohammad Alotaibi; Ahmad Mustafa; Yusufjon Kumakov; Mohit Prabhushankar; Ghassan AlRegib
>
> **摘要:** Machine learning has taken a critical role in seismic interpretation workflows, especially in fault delineation tasks. However, despite the recent proliferation of pretrained models and synthetic datasets, the field still lacks a systematic understanding of the generalizability limits of these models across seismic data representing diverse geologic, acquisition and processing settings. Distributional shifts between data sources, limitations in fine-tuning strategies and labeled data accessibility, and inconsistent evaluation protocols all remain major roadblocks to deploying reliable models in real-world exploration. In this paper, we present the first large-scale benchmarking study explicitly designed to provide guidelines for domain shift strategies in seismic interpretation. Our benchmark spans over 200 combinations of model architectures, datasets and training strategies, across three datasets (synthetic and real) including FaultSeg3D, CRACKS, and Thebe. We systematically assess pretraining, fine-tuning, and joint training under varying domain shifts. Our analysis shows that common fine-tuning practices can lead to catastrophic forgetting, especially when source and target datasets are disjoint, and that larger models such as Segformer are more robust than smaller architectures. We also find that domain adaptation methods outperform fine-tuning when shifts are large, yet underperform when domains are similar. Finally, we complement segmentation metrics with a novel analysis based on fault characteristic descriptors, revealing how models absorb structural biases from training datasets. Overall, we establish a robust experimental baseline that provides insights into tradeoffs in current fault delineation workflows and highlights directions for building more generalizable and interpretable models.
>
---
#### [replaced 003] TransParking: A Dual-Decoder Transformer Framework with Soft Localization for End-to-End Automatic Parking
- **分类: cs.CV**

- **链接: []()**

> **作者:** Hangyu Du; Chee-Meng Chew
>
> **摘要:** In recent years, fully differentiable end-to-end autonomous driving systems have become a research hotspot in the field of intelligent transportation. Among various research directions, automatic parking is particularly critical as it aims to enable precise vehicle parking in complex environments. In this paper, we present a purely vision-based transformer model for end-to-end automatic parking, trained using expert trajectories. Given camera-captured data as input, the proposed model directly outputs future trajectory coordinates. Experimental results demonstrate that the various errors of our model have decreased by approximately 50% in comparison with the current state-of-the-art end-to-end trajectory prediction algorithm of the same type. Our approach thus provides an effective solution for fully differentiable automatic parking.
>
---
#### [replaced 004] PLUTO-4: Frontier Pathology Foundation Models
- **分类: cs.CV**

- **链接: []()**

> **作者:** Harshith Padigela; Shima Nofallah; Atchuth Naveen Chilaparasetti; Ryun Han; Andrew Walker; Judy Shen; Chintan Shah; Blake Martin; Aashish Sood; Elliot Miller; Ben Glass; Andy Beck; Harsha Pokkalla; Syed Ashar Javed
>
> **摘要:** Foundation models trained on large-scale pathology image corpora have demonstrated strong transfer capabilities across diverse histopathology tasks. Building on this progress, we introduce PLUTO-4, our next generation of pathology foundation models that extend the Pathology-Universal Transformer (PLUTO) to frontier scale. We share two complementary Vision Transformer architectures in the PLUTO-4 family: a compact and efficient PLUTO-4S model optimized for multi-scale deployment using a FlexiViT setup with 2D-RoPE embeddings, and a frontier-scale PLUTO-4G model trained with a single patch size to maximize representation capacity and stability. Both models are pretrained using a self-supervised objective derived from DINOv2 on a large multi-institutional corpus containing 551,164 WSIs from 137,144 patients across over 50 institutions, spanning over 60 disease types and over 100 stains. Comprehensive evaluation across public and internal benchmarks demonstrates that PLUTO-4 achieves state-of-the-art performance on tasks requiring varying spatial and biological context, including tile classification, segmentation, and slide-level diagnosis. The compact PLUTO-4S provides high-throughput and robust performance for practical deployment, while PLUTO-4G establishes new performance frontiers across multiple pathology benchmarks, including an 11% improvement in dermatopathology diagnosis. These diverse improvements underscore PLUTO-4's potential to transform real-world applications as a backbone for translational research and diagnostic use cases.
>
---
#### [replaced 005] Benchmarking Domain Generalization Algorithms in Computational Pathology
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: []()**

> **作者:** Neda Zamanitajeddin; Mostafa Jahanifar; Kesi Xu; Fouzia Siraj; Nasir Rajpoot
>
> **摘要:** Deep learning models have shown immense promise in computational pathology (CPath) tasks, but their performance often suffers when applied to unseen data due to domain shifts. Addressing this requires domain generalization (DG) algorithms. However, a systematic evaluation of DG algorithms in the CPath context is lacking. This study aims to benchmark the effectiveness of 30 DG algorithms on 3 CPath tasks of varying difficulty through 7,560 cross-validation runs. We evaluate these algorithms using a unified and robust platform, incorporating modality-specific techniques and recent advances like pretrained foundation models. Our extensive cross-validation experiments provide insights into the relative performance of various DG strategies. We observe that self-supervised learning and stain augmentation consistently outperform other methods, highlighting the potential of pretrained models and data augmentation. Furthermore, we introduce a new pan-cancer tumor detection dataset (HISTOPANTUM) as a benchmark for future research. This study offers valuable guidance to researchers in selecting appropriate DG approaches for CPath tasks.
>
---
#### [replaced 006] Robust Nearest Neighbour Retrieval Using Targeted Manifold Manipulation
- **分类: cs.CV**

- **链接: []()**

> **作者:** B. Ghosh; H. Harikumar; S. Rana
>
> **摘要:** Nearest-neighbour retrieval is central to classification and explainable-AI pipelines, but current practice relies on hand-tuning feature layers and distance metrics. We propose Targeted Manifold Manipulation-Nearest Neighbour (TMM-NN), which reconceptualises retrieval by assessing how readily each sample can be nudged into a designated region of the feature manifold; neighbourhoods are defined by a sample's responsiveness to a targeted perturbation rather than absolute geometric distance. TMM-NN implements this through a lightweight, query-specific trigger patch. The patch is added to the query image, and the network is weakly ``backdoored'' so that any input with the patch is steered toward a dummy class. Images similar to the query need only a slight shift and are classified as the dummy class with high probability, while dissimilar ones are less affected. By ranking candidates by this confidence, TMM-NN retrieves the most semantically related neighbours. Robustness analysis and benchmark experiments confirm this trigger-based ranking outperforms traditional metrics under noise and across diverse tasks.
>
---
#### [replaced 007] Physics-Informed Deformable Gaussian Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field
- **分类: cs.CV**

- **链接: []()**

> **作者:** Haoqin Hong; Ding Fan; Fubin Dou; Zhi-Li Zhou; Haoran Sun; Congcong Zhu; Jingrun Chen
>
> **备注:** Accepted by AAAI-26
>
> **摘要:** Recently, 3D Gaussian Splatting (3DGS), an explicit scene representation technique, has shown significant promise for dynamic novel-view synthesis from monocular video input. However, purely data-driven 3DGS often struggles to capture the diverse physics-driven motion patterns in dynamic scenes. To fill this gap, we propose Physics-Informed Deformable Gaussian Splatting (PIDG), which treats each Gaussian particle as a Lagrangian material point with time-varying constitutive parameters and is supervised by 2D optical flow via motion projection. Specifically, we adopt static-dynamic decoupled 4D decomposed hash encoding to reconstruct geometry and motion efficiently. Subsequently, we impose the Cauchy momentum residual as a physics constraint, enabling independent prediction of each particle's velocity and constitutive stress via a time-evolving material field. Finally, we further supervise data fitting by matching Lagrangian particle flow to camera-compensated optical flow, which accelerates convergence and improves generalization. Experiments on a custom physics-driven dataset as well as on standard synthetic and real-world datasets demonstrate significant gains in physical consistency and monocular dynamic reconstruction quality.
>
---
#### [replaced 008] AgentSense: Virtual Sensor Data Generation Using LLM Agents in Simulated Home Environments
- **分类: cs.CV; cs.HC**

- **链接: []()**

> **作者:** Zikang Leng; Megha Thukral; Yaqi Liu; Hrudhai Rajasekhar; Shruthi K. Hiremath; Jiaman He; Thomas Plötz
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** A major challenge in developing robust and generalizable Human Activity Recognition (HAR) systems for smart homes is the lack of large and diverse labeled datasets. Variations in home layouts, sensor configurations, and individual behaviors further exacerbate this issue. To address this, we leverage the idea of embodied AI agents -- virtual agents that perceive and act within simulated environments guided by internal world models. We introduce AgentSense, a virtual data generation pipeline in which agents live out daily routines in simulated smart homes, with behavior guided by Large Language Models (LLMs). The LLM generates diverse synthetic personas and realistic routines grounded in the environment, which are then decomposed into fine-grained actions. These actions are executed in an extended version of the VirtualHome simulator, which we augment with virtual ambient sensors that record the agents' activities. Our approach produces rich, privacy-preserving sensor data that reflects real-world diversity. We evaluate AgentSense on five real HAR datasets. Models pretrained on the generated data consistently outperform baselines, especially in low-resource settings. Furthermore, combining the generated virtual sensor data with a small amount of real data achieves performance comparable to training on full real-world datasets. These results highlight the potential of using LLM-guided embodied agents for scalable and cost-effective sensor data generation in HAR. Our code is publicly available at https://github.com/ZikangLeng/AgentSense.
>
---
#### [replaced 009] X-Scene: Large-Scale Driving Scene Generation with High Fidelity and Flexible Controllability
- **分类: cs.CV**

- **链接: []()**

> **作者:** Yu Yang; Alan Liang; Jianbiao Mei; Yukai Ma; Yong Liu; Gim Hee Lee
>
> **备注:** Accepted by NeurIPS 2025, Project page at https://x-scene.github.io/
>
> **摘要:** Diffusion models are advancing autonomous driving by enabling realistic data synthesis, predictive end-to-end planning, and closed-loop simulation, with a primary focus on temporally consistent generation. However, large-scale 3D scene generation requiring spatial coherence remains underexplored. In this paper, we present X-Scene, a novel framework for large-scale driving scene generation that achieves geometric intricacy, appearance fidelity, and flexible controllability. Specifically, X-Scene supports multi-granular control, including low-level layout conditioning driven by user input or text for detailed scene composition, and high-level semantic guidance informed by user intent and LLM-enriched prompts for efficient customization. To enhance geometric and visual fidelity, we introduce a unified pipeline that sequentially generates 3D semantic occupancy and corresponding multi-view images and videos, ensuring alignment and temporal consistency across modalities. We further extend local regions into large-scale scenes via consistency-aware outpainting, which extrapolates occupancy and images from previously generated areas to maintain spatial and visual coherence. The resulting scenes are lifted into high-quality 3DGS representations, supporting diverse applications such as simulation and scene exploration. Extensive experiments demonstrate that X-Scene substantially advances controllability and fidelity in large-scale scene generation, empowering data generation and simulation for autonomous driving.
>
---
#### [replaced 010] DetailFlow: 1D Coarse-to-Fine Autoregressive Image Generation via Next-Detail Prediction
- **分类: cs.CV**

- **链接: []()**

> **作者:** Yiheng Liu; Liao Qu; Huichao Zhang; Xu Wang; Yi Jiang; Yiming Gao; Hu Ye; Xian Li; Shuai Wang; Daniel K. Du; Fangmin Chen; Zehuan Yuan; Xinglong Wu
>
> **摘要:** This paper presents DetailFlow, a coarse-to-fine 1D autoregressive (AR) image generation method that models images through a novel next-detail prediction strategy. By learning a resolution-aware token sequence supervised with progressively degraded images, DetailFlow enables the generation process to start from the global structure and incrementally refine details. This coarse-to-fine 1D token sequence aligns well with the autoregressive inference mechanism, providing a more natural and efficient way for the AR model to generate complex visual content. Our compact 1D AR model achieves high-quality image synthesis with significantly fewer tokens than previous approaches, i.e. VAR/VQGAN. We further propose a parallel inference mechanism with self-correction that accelerates generation speed by approximately 8x while reducing accumulation sampling error inherent in teacher-forcing supervision. On the ImageNet 256x256 benchmark, our method achieves 2.96 gFID with 128 tokens, outperforming VAR (3.3 FID) and FlexVAR (3.05 FID), which both require 680 tokens in their AR models. Moreover, due to the significantly reduced token count and parallel inference mechanism, our method runs nearly 2x faster inference speed compared to VAR and FlexVAR. Extensive experimental results demonstrate DetailFlow's superior generation quality and efficiency compared to existing state-of-the-art methods.
>
---
#### [replaced 011] MMCL: Correcting Content Query Distributions for Improved Anti-Overlapping X-Ray Object Detection
- **分类: cs.CV**

- **链接: []()**

> **作者:** Mingyuan Li; Tong Jia; Hui Lu; Hao Wang; Bowen Ma; Shiyi Guo; Shuyang Lin; Dongyue Chen; Haoran Wang; Baosheng Yu
>
> **备注:** 16 pages,8 figures
>
> **摘要:** Unlike natural images with occlusion-based overlap, X-ray images exhibit depth-induced superimposition and semi-transparent appearances, where objects at different depths overlap and their features blend together. These characteristics demand specialized mechanisms to disentangle mixed representations between target objects (e.g., prohibited items) and irrelevant backgrounds. While recent studies have explored adapting detection transformers (DETR) for anti-overlapping object detection, the importance of well-distributed content queries that represent object hypotheses remains underexplored. In this paper, we introduce a multi-class min-margin contrastive learning (MMCL) framework to correct the distribution of content queries, achieving balanced intra-class diversity and inter-class separability. The framework first groups content queries by object category and then applies two proposed complementary loss components: a multi-class exclusion loss to enhance inter-class separability, and a min-margin clustering loss to encourage intra-class diversity. We evaluate the proposed method on three widely used X-ray prohibited-item detection datasets, PIXray, OPIXray, and PIDray, using two backbone networks and four DETR variants. Experimental results demonstrate that MMCL effectively enhances anti-overlapping object detection and achieves state-of-the-art performance on both datasets. Code will be made publicly available on GitHub.
>
---
#### [replaced 012] Imbalance in Balance: Online Concept Balancing in Generation Models
- **分类: cs.CV; cs.AI**

- **链接: []()**

> **作者:** Yukai Shi; Jiarong Ou; Rui Chen; Haotian Yang; Jiahao Wang; Xin Tao; Pengfei Wan; Di Zhang; Kun Gai
>
> **备注:** Accepted by ICCV2025. Codes have been released at https://github.com/KwaiVGI/IMBA-Loss
>
> **摘要:** In visual generation tasks, the responses and combinations of complex concepts often lack stability and are error-prone, which remains an under-explored area. In this paper, we attempt to explore the causal factors for poor concept responses through elaborately designed experiments. We also design a concept-wise equalization loss function (IMBA loss) to address this issue. Our proposed method is online, eliminating the need for offline dataset processing, and requires minimal code changes. In our newly proposed complex concept benchmark Inert-CompBench and two other public test sets, our method significantly enhances the concept response capability of baseline models and yields highly competitive results with only a few codes released at https://github.com/KwaiVGI/IMBA-Loss.
>
---
#### [replaced 013] Disentangled Representation Learning via Modular Compositional Bias
- **分类: cs.LG; cs.CV**

- **链接: []()**

> **作者:** Whie Jung; Dong Hoon Lee; Seunghoon Hong
>
> **摘要:** Recent disentangled representation learning (DRL) methods heavily rely on factor specific strategies-either learning objectives for attributes or model architectures for objects-to embed inductive biases. Such divergent approaches result in significant overhead when novel factors of variation do not align with prior assumptions, such as statistical independence or spatial exclusivity, or when multiple factors coexist, as practitioners must redesign architectures or objectives. To address this, we propose a compositional bias, a modular inductive bias decoupled from both objectives and architectures. Our key insight is that different factors obey distinct recombination rules in the data distribution: global attributes are mutually exclusive, e.g., a face has one nose, while objects share a common support (any subset of objects can co-exist). We therefore randomly remix latents according to factor-specific rules, i.e., a mixing strategy, and force the encoder to discover whichever factor structure the mixing strategy reflects through two complementary objectives: (i) a prior loss that ensures every remix decodes into a realistic image, and (ii) the compositional consistency loss introduced by Wiedemer et al. (arXiv:2310.05327), which aligns each composite image with its corresponding composite latent. Under this general framework, simply adjusting the mixing strategy enables disentanglement of attributes, objects, and even both, without modifying the objectives or architectures. Extensive experiments demonstrate that our method shows competitive performance in both attribute and object disentanglement, and uniquely achieves joint disentanglement of global style and objects. Code is available at https://github.com/whieya/Compositional-DRL.
>
---
#### [replaced 014] FALCON: False-Negative Aware Learning of Contrastive Negatives in Vision-Language Alignment
- **分类: cs.CV; cs.AI**

- **链接: []()**

> **作者:** Myunsoo Kim; Seong-Woong Shim; Byung-Jun Lee
>
> **摘要:** False negatives pose a critical challenge in vision-language pretraining (VLP) due to the many-to-many correspondence between images and texts in large-scale datasets. These false negatives introduce conflicting supervision signals that degrade the learned embedding space and diminish the effectiveness of hard negative sampling. In this paper, we propose FALCON (False-negative Aware Learning of COntrastive Negatives), a learning-based mini-batch construction strategy that adaptively balances the trade-off between hard and false negatives during VLP. Rather than relying on fixed heuristics, FALCON employs a negative mining scheduler that dynamically selects negative samples of appropriate hardness for each anchor instance during mini-batch construction, guided by a proxy for cross-modal alignment improvement. Experimental results demonstrate that FALCON significantly improves performance across three vision-language learning frameworks (ALBEF, BLIP-2, SigLIP-2) and a broad range of downstream tasks and evaluation settings, underscoring its effectiveness and robustness in mitigating the impact of false negatives.
>
---
#### [replaced 015] ElastoGen: 4D Generative Elastodynamics
- **分类: cs.LG; cs.CV; cs.GR**

- **链接: []()**

> **作者:** Yutao Feng; Yintong Shang; Xiang Feng; Lei Lan; Shandian Zhe; Tianjia Shao; Hongzhi Wu; Kun Zhou; Chenfanfu Jiang; Yin Yang
>
> **摘要:** We present ElastoGen, a knowledge-driven AI model that generates physically accurate 4D elastodynamics. Unlike deep models that learn from video- or image-based observations, ElastoGen leverages the principles of physics and learns from established mathematical and optimization procedures. The core idea of ElastoGen is converting the differential equation, corresponding to the nonlinear force equilibrium, into a series of iterative local convolution-like operations, which naturally fit deep architectures. We carefully build our network module following this overarching design philosophy. ElastoGen is much more lightweight in terms of both training requirements and network scale than deep generative models. Because of its alignment with actual physical procedures, ElastoGen efficiently generates accurate dynamics for a wide range of hyperelastic materials and can be easily integrated with upstream and downstream deep modules to enable end-to-end 4D generation.
>
---
#### [replaced 016] Token Is All You Need: Cognitive Planning through Belief-Intent Co-Evolution
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: []()**

> **作者:** Shiyao Sang
>
> **备注:** 7 pages, 3 figures. A paradigm shift from reconstructing the world to understanding it: planning through belief-intent co-evolution
>
> **摘要:** We challenge the long-standing assumption that exhaustive scene modeling is required for high-performance end-to-end autonomous driving (E2EAD). Inspired by cognitive science, we propose that effective planning arises not from reconstructing the world, but from the co-evolution of belief and intent within a minimal set of semantically rich tokens. Experiments on the nuPlan benchmark (720 scenarios, 11k+ samples) reveal three principles: (1) sparse intent tokens alone achieve 0.487 m ADE, demonstrating strong performance without future prediction; (2) conditioning trajectory decoding on predicted future tokens reduces ADE to 0.382 m, a 21.6% improvement, showing that performance emerges from cognitive planning; and (3) explicit reconstruction loss degrades performance, confirming that task-driven belief-intent co-evolution suffices under reliable perception inputs. Crucially, we observe the emergence of cognitive consistency: through prolonged training, the model spontaneously develops stable token dynamics that balance current perception (belief) and future goals (intent). This process, accompanied by "temporal fuzziness," enables robustness under uncertainty and continuous self-optimization. Our work establishes a new paradigm: intelligence lies not in pixel fidelity, but in the tokenized duality of belief and intent. By reframing planning as understanding rather than reaction, TIWM bridges the gap between world models and VLA systems, paving the way for foresightful agents that plan through imagination. Note: Numerical comparisons with methods reporting results on nuScenes are indicative only, as nuPlan presents a more challenging planning-focused evaluation.
>
---
#### [replaced 017] Multimodal Adversarial Defense for Vision-Language Models by Leveraging One-To-Many Relationships
- **分类: cs.CV; cs.AI; cs.IR**

- **链接: []()**

> **作者:** Futa Waseda; Antonio Tejero-de-Pablos; Isao Echizen
>
> **备注:** WACV 2026 Accepted
>
> **摘要:** Pre-trained vision-language (VL) models are highly vulnerable to adversarial attacks. However, existing defense methods primarily focus on image classification, overlooking two key aspects of VL tasks: multimodal attacks, where both image and text can be perturbed, and the one-to-many relationship of images and texts, where a single image can correspond to multiple textual descriptions and vice versa (1:N and N:1). This work is the first to explore defense strategies against multimodal attacks in VL tasks, whereas prior VL defense methods focus on vision robustness. We propose multimodal adversarial training (MAT), which incorporates adversarial perturbations in both image and text modalities during training, significantly outperforming existing unimodal defenses. Furthermore, we discover that MAT is limited by deterministic one-to-one (1:1) image-text pairs in VL training data. To address this, we conduct a comprehensive study on leveraging one-to-many relationships to enhance robustness, investigating diverse augmentation techniques. Our analysis shows that, for a more effective defense, augmented image-text pairs should be well-aligned, diverse, yet avoid distribution shift -- conditions overlooked by prior research. This work pioneers defense strategies against multimodal attacks, providing insights for building robust VLMs from both optimization and data perspectives.
>
---
#### [replaced 018] Selective Diabetic Retinopathy Screening with Accuracy-Weighted Deep Ensembles and Entropy-Guided Abstention
- **分类: q-bio.QM; cs.AI; cs.CV**

- **链接: []()**

> **作者:** Jophy Lin
>
> **摘要:** Diabetic retinopathy (DR), a microvascular complication of diabetes and a leading cause of preventable blindness, is projected to affect more than 130 million individuals worldwide by 2030. Early identification is essential to reduce irreversible vision loss, yet current diagnostic workflows rely on methods such as fundus photography and expert review, which remain costly and resource-intensive. This, combined with DR's asymptomatic nature, results in its underdiagnosis rate of approximately 25 percent. Although convolutional neural networks (CNNs) have demonstrated strong performance in medical imaging tasks, limited interpretability and the absence of uncertainty quantification restrict clinical reliability. Therefore, in this study, a deep ensemble learning framework integrated with uncertainty estimation is introduced to improve robustness, transparency, and scalability in DR detection. The ensemble incorporates seven CNN architectures-ResNet-50, DenseNet-121, MobileNetV3 (Small and Large), and EfficientNet (B0, B2, B3)- whose outputs are fused through an accuracy-weighted majority voting strategy. A probability-weighted entropy metric quantifies prediction uncertainty, enabling low-confidence samples to be excluded or flagged for additional review. Training and validation on 35,000 EyePACS retinal fundus images produced an unfiltered accuracy of 93.70 percent (F1 = 0.9376). Uncertainty-filtering later was conducted to remove unconfident samples, resulting in maximum-accuracy of 99.44 percent (F1 = 0.9932). The framework shows that uncertainty-aware, accuracy-weighted ensembling improves reliability without hindering performance. With confidence-calibrated outputs and a tunable accuracy-coverage trade-off, it offers a generalizable paradigm for deploying trustworthy AI diagnostics in high-risk care.
>
---
#### [replaced 019] FaSDiff: Balancing Perception and Semantics in Face Compression via Stable Diffusion Priors
- **分类: cs.CV; cs.AI; eess.IV**

- **链接: []()**

> **作者:** Yimin Zhou; Yichong Xia; Bin Chen; Mingyao Hong; Jiawei Li; Zhi Wang; Yaowei Wang
>
> **摘要:** With the increasing deployment of facial image data across a wide range of applications, efficient compression tailored to facial semantics has become critical for both storage and transmission. While recent learning-based face image compression methods have achieved promising results, they often suffer from degraded reconstruction quality at low bit rates. Directly applying diffusion-based generative priors to this task leads to suboptimal performance in downstream machine vision tasks, primarily due to poor preservation of high-frequency details. In this work, we propose FaSDiff (\textbf{Fa}cial Image Compression with a \textbf{S}table \textbf{Diff}usion Prior), a novel diffusion-driven compression framework designed to enhance both visual fidelity and semantic consistency. FaSDiff incorporates a high-frequency-sensitive compressor to capture fine-grained details and generate robust visual prompts for guiding the diffusion model. To address low-frequency degradation, we further introduce a hybrid low-frequency enhancement module that disentangles and preserves semantic structures, enabling stable modulation of the diffusion prior during reconstruction. By jointly optimizing perceptual quality and semantic preservation, FaSDiff effectively balances human visual fidelity and machine vision accuracy. Extensive experiments demonstrate that FaSDiff outperforms state-of-the-art methods in both perceptual metrics and downstream task performance.
>
---
#### [replaced 020] DODA: Adapting Object Detectors to Dynamic Agricultural Environments in Real-Time with Diffusion
- **分类: cs.CV**

- **链接: []()**

> **作者:** Shuai Xiang; Pieter M. Blok; James Burridge; Haozhou Wang; Wei Guo
>
> **备注:** WACV2026
>
> **摘要:** Object detection has wide applications in agriculture, but domain shifts of diverse environments limit the broader use of the trained models. Existing domain adaptation methods usually require retraining the model for new domains, which is impractical for agricultural applications due to constantly changing environments. In this paper, we propose DODA ($D$iffusion for $O$bject-detection $D$omain Adaptation in $A$griculture), a diffusion-based framework that can adapt the detector to a new domain in just 2 minutes. DODA incorporates external domain embeddings and an improved layout-to-image approach, allowing it to generate high-quality detection data for new domains without additional training. We demonstrate DODA's effectiveness on the Global Wheat Head Detection dataset, where fine-tuning detectors on DODA-generated data yields significant improvements across multiple domains. DODA provides a simple yet powerful solution for agricultural domain adaptation, reducing the barriers for growers to use detection in personalised environments. The code is available at https://github.com/UTokyo-FieldPhenomics-Lab/DODA.
>
---
#### [replaced 021] Association and Consolidation: Evolutionary Memory-Enhanced Incremental Multi-View Clustering
- **分类: cs.CV**

- **链接: []()**

> **作者:** Zisen Kong; Bo Zhong; Pengyuan Li; Dongxia Chang; Yiming Wang; Yongyong Chen
>
> **备注:** Submitted to CVPR2026
>
> **摘要:** Incremental multi-view clustering aims to achieve stable clustering results while addressing the stability-plasticity dilemma (SPD) in view-incremental scenarios. The core challenge is that the model must have enough plasticity to quickly adapt to new data, while maintaining sufficient stability to consolidate long-term knowledge. To address this challenge, we propose a novel Evolutionary Memory-Enhanced Incremental Multi-View Clustering (EMIMC), inspired by the memory regulation mechanisms of the human brain. Specifically, we design a rapid association module to establish connections between new and historical views, thereby ensuring the plasticity required for learning new knowledge. Second, a cognitive forgetting module with a decay mechanism is introduced. By dynamically adjusting the contribution of the historical view to optimize knowledge integration. Finally, we propose a knowledge consolidation module to progressively refine short-term knowledge into stable long-term memory using temporal tensors, thereby ensuring model stability. By integrating these modules, EMIMC achieves strong knowledge retention capabilities in scenarios with growing views. Extensive experiments demonstrate that EMIMC exhibits remarkable advantages over existing state-of-the-art methods.
>
---
#### [replaced 022] PMGS: Reconstruction of Projectile Motion Across Large Spatiotemporal Spans via 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: []()**

> **作者:** Yijun Xu; Jingrui Zhang; Yuhan Chen; Dingwen Wang; Lei Yu; Chu He
>
> **摘要:** Modeling complex rigid motion across large spatiotemporal spans remains an unresolved challenge in dynamic reconstruction. Existing paradigms are mainly confined to short-term, small-scale deformation and offer limited consideration for physical consistency. This study proposes PMGS, focusing on reconstructing Projectile Motion via 3D Gaussian Splatting. The workflow comprises two stages: 1) Target Modeling: achieving object-centralized reconstruction through dynamic scene decomposition and an improved point density control; 2) Motion Recovery: restoring full motion sequences by learning per-frame SE(3) poses. We introduce an acceleration consistency constraint to bridge Newtonian mechanics and pose estimation, and design a dynamic simulated annealing strategy that adaptively schedules learning rates based on motion states. Futhermore, we devise a Kalman fusion scheme to optimize error accumulation from multi-source observations to mitigate disturbances. Experiments show PMGS's superior performance in reconstructing high-speed nonlinear rigid motion compared to mainstream dynamic methods.
>
---
#### [replaced 023] Clinical Uncertainty Impacts Machine Learning Evaluations
- **分类: cs.AI; cs.CV; cs.LG**

- **链接: []()**

> **作者:** Simone Lionetti; Fabian Gröger; Philippe Gottfrois; Alvaro Gonzalez-Jimenez; Ludovic Amruthalingam; Alexander A. Navarini; Marc Pouly
>
> **备注:** ML4H 2025 findings camera-ready
>
> **摘要:** Clinical dataset labels are rarely certain as annotators disagree and confidence is not uniform across cases. Typical aggregation procedures, such as majority voting, obscure this variability. In simple experiments on medical imaging benchmarks, accounting for the confidence in binary labels significantly impacts model rankings. We therefore argue that machine-learning evaluations should explicitly account for annotation uncertainty using probabilistic metrics that directly operate on distributions. These metrics can be applied independently of the annotations' generating process, whether modeled by simple counting, subjective confidence ratings, or probabilistic response models. They are also computationally lightweight, as closed-form expressions have linear-time implementations once examples are sorted by model score. We thus urge the community to release raw annotations for datasets and to adopt uncertainty-aware evaluation so that performance estimates may better reflect clinical data.
>
---
#### [replaced 024] VideoChat-R1: Enhancing Spatio-Temporal Perception via Reinforcement Fine-Tuning
- **分类: cs.CV**

- **链接: []()**

> **作者:** Xinhao Li; Ziang Yan; Desen Meng; Lu Dong; Xiangyu Zeng; Yinan He; Yali Wang; Yu Qiao; Yi Wang; Limin Wang
>
> **摘要:** Reinforcement Learning (RL) benefits Large Language Models (LLMs) for complex reasoning. Inspired by this, we explore integrating spatio-temporal specific rewards into Multimodal Large Language Models (MLLMs) to address the unique challenges of video understanding, such as long-range temporal associations. This paper investigates how rule-based rewards, particularly temporal ones, can improve video reasoning and their generalizability. Our study proposes Reinforcement Fine-Tuning (RFT) as a data-efficient method to enhance video reasoning on specific tasks without sacrificing original capabilities. Through joint RFT on multiple spatio-temporal perception tasks, we developed VideoChat-R1, a powerful Video MLLM. VideoChat-R1 achieves state-of-the-art spatio-temporal perception, demonstrating significant improvements in tasks like temporal grounding (+31.8) and object tracking (+31.2), while also improving general QA benchmarks. The enhanced perception and preserved chat abilities contribute to a more reliable video dialogue system, leading to our ``Temporal Clue-driven Reasoning" inference schema. This work provides a foundation for developing robust, real-world video comprehension agents.
>
---
#### [replaced 025] RedDiffuser: Red Teaming Vision-Language Models for Toxic Continuation via Reinforced Stable Diffusion
- **分类: cs.CV**

- **链接: []()**

> **作者:** Ruofan Wang; Xiang Zheng; Xiaosen Wang; Cong Wang; Xingjun Ma; Yu-Gang Jiang
>
> **摘要:** Vision-Language Models (VLMs) are vulnerable to jailbreak attacks, where adversaries bypass safety mechanisms to elicit harmful outputs. In this work, we examine an insidious variant of this threat: toxic continuation. Unlike standard jailbreaks that rely solely on malicious instructions, toxic continuation arises when the model is given a malicious input alongside a partial toxic output, resulting in harmful completions. This vulnerability poses a unique challenge in multimodal settings, where even subtle image variations can disproportionately affect the model's response. To this end, we propose RedDiffuser (RedDiff), the first red teaming framework that uses reinforcement learning to fine-tune diffusion models into generating natural-looking adversarial images that induce toxic continuations. RedDiffuser integrates a greedy search procedure for selecting candidate image prompts with reinforcement fine-tuning that jointly promotes toxic output and semantic coherence. Experiments demonstrate that RedDiffuser significantly increases the toxicity rate in LLaVA outputs by 10.69% and 8.91% on the original and hold-out sets, respectively. It also exhibits strong transferability, increasing toxicity rates on Gemini by 5.1% and on LLaMA-Vision by 26.83%. These findings uncover a cross-modal toxicity amplification vulnerability in current VLM alignment, highlighting the need for robust multimodal red teaming. We will release the RedDiffuser codebase to support future research.
>
---
#### [replaced 026] HarmoQ: Harmonized Post-Training Quantization for High-Fidelity Image
- **分类: eess.IV; cs.CV**

- **链接: []()**

> **作者:** Hongjun Wang; Jiyuan Chen; Xuan Song; Yinqiang Zheng
>
> **摘要:** Post-training quantization offers an efficient pathway to deploy super-resolution models, yet existing methods treat weight and activation quantization independently, missing their critical interplay. Through controlled experiments on SwinIR, we uncover a striking asymmetry: weight quantization primarily degrades structural similarity, while activation quantization disproportionately affects pixel-level accuracy. This stems from their distinct roles--weights encode learned restoration priors for textures and edges, whereas activations carry input-specific intensity information. Building on this insight, we propose HarmoQ, a unified framework that harmonizes quantization across components through three synergistic steps: structural residual calibration proactively adjusts weights to compensate for activation-induced detail loss, harmonized scale optimization analytically balances quantization difficulty via closed-form solutions, and adaptive boundary refinement iteratively maintains this balance during optimization. Experiments show HarmoQ achieves substantial gains under aggressive compression, outperforming prior art by 0.46 dB on Set5 at 2-bit while delivering 3.2x speedup and 4x memory reduction on A100 GPUs. This work provides the first systematic analysis of weight-activation coupling in super-resolution quantization and establishes a principled solution for efficient high-quality image restoration.
>
---
#### [replaced 027] DGL-RSIS: Decoupling Global Spatial Context and Local Class Semantics for Training-Free Remote Sensing Image Segmentation
- **分类: cs.CV**

- **链接: []()**

> **作者:** Boyi Li; Ce Zhang; Richard M. Timmerman; Wenxuan Bao
>
> **摘要:** The emergence of vision language models (VLMs) bridges the gap between vision and language, enabling multimodal understanding beyond traditional visual-only deep learning models. However, transferring VLMs from the natural image domain to remote sensing (RS) segmentation remains challenging due to the large domain gap and the diversity of RS inputs across tasks, particularly in open-vocabulary semantic segmentation (OVSS) and referring expression segmentation (RES). Here, we propose a training-free unified framework, termed DGL-RSIS, which decouples visual and textual representations and performs visual-language alignment at both local semantic and global contextual levels. Specifically, a Global-Local Decoupling (GLD) module decomposes textual inputs into local semantic tokens and global contextual tokens, while image inputs are partitioned into class-agnostic mask proposals. Then, a Local Visual-Textual Alignment (LVTA) module adaptively extracts context-aware visual features from the mask proposals and enriches textual features through knowledge-guided prompt engineering, achieving OVSS from a local perspective. Furthermore, a Global Visual-Textual Alignment (GVTA) module employs a global-enhanced Grad-CAM mechanism to capture contextual cues for referring expressions, followed by a mask selection module that integrates pixel-level activations into mask-level segmentation outputs, thereby achieving RES from a global perspective. Experiments on the iSAID (OVSS) and RRSIS-D (RES) benchmarks demonstrate that DGL-RSIS outperforms existing training-free approaches. Ablation studies further validate the effectiveness of each module. To the best of our knowledge, this is the first unified training-free framework for RS image segmentation, which effectively transfers the semantic capability of VLMs trained on natural images to the RS domain without additional training.
>
---
#### [replaced 028] Reasoning-Enhanced Domain-Adaptive Pretraining of Multimodal Large Language Models for Short Video Content Governance
- **分类: cs.CV**

- **链接: []()**

> **作者:** Zixuan Wang; Yu Sun; Hongwei Wang; Baoyu Jing; Xiang Shen; Xin Dong; Zhuolin Hao; Hongyu Xiong; Yang Song
>
> **备注:** Camera Ready for EMNLP 2025
>
> **摘要:** Short video platforms are evolving rapidly, making the identification of inappropriate content increasingly critical. Existing approaches typically train separate and small classification models for each type of issue, which requires extensive human-labeled data and lacks cross-issue generalization. We propose a reasoning-enhanced multimodal large language model (MLLM) pretraining paradigm for unified inappropriate content detection. To address the distribution gap between short video content and the original pretraining data of MLLMs, as well as the complex issue definitions, we introduce three targeted pretraining tasks: (1) \textit{Caption}, to enhance the MLLM's perception of video details; (2) \textit{Visual Question Answering (VQA)}, to deepen the MLLM's understanding of issue definitions and annotation guidelines; (3) \textit{Chain-of-Thought (CoT)}, to enhance the MLLM's reasoning capability. Experimental results show that our pretraining approach significantly improves the MLLM's performance in both zero-shot and supervised fine-tuning (SFT) settings. In addition, our pretrained model demonstrates strong generalization capabilities to emergent, previously unseen issues.
>
---
#### [replaced 029] I Detect What I Don't Know: Incremental Anomaly Learning with Stochastic Weight Averaging-Gaussian for Oracle-Free Medical Imaging
- **分类: cs.CV; cs.AI**

- **链接: []()**

> **作者:** Nand Kumar Yadav; Rodrigue Rizk; William CW Chen; KC Santosh
>
> **摘要:** Unknown anomaly detection in medical imaging remains a fundamental challenge due to the scarcity of labeled anomalies and the high cost of expert supervision. We introduce an unsupervised, oracle-free framework that incrementally expands a trusted set of normal samples without any anomaly labels. Starting from a small, verified seed of normal images, our method alternates between lightweight adapter updates and uncertainty-gated sample admission. A frozen pretrained vision backbone is augmented with tiny convolutional adapters, ensuring rapid domain adaptation with negligible computational overhead. Extracted embeddings are stored in a compact coreset enabling efficient k-nearest neighbor anomaly (k-NN) scoring. Safety during incremental expansion is enforced by dual probabilistic gates, a sample is admitted into the normal memory only if its distance to the existing coreset lies within a calibrated z-score threshold, and its SWAG-based epistemic uncertainty remains below a seed-calibrated bound. This mechanism prevents drift and false inclusions without relying on generative reconstruction or replay buffers. Empirically, our system steadily refines the notion of normality as unlabeled data arrive, producing substantial gains over baselines. On COVID-CXR, ROC-AUC improves from 0.9489 to 0.9982 (F1: 0.8048 to 0.9746); on Pneumonia CXR, ROC-AUC rises from 0.6834 to 0.8968; and on Brain MRI ND-5, ROC-AUC increases from 0.6041 to 0.7269 and PR-AUC from 0.7539 to 0.8211. These results highlight the effectiveness and efficiency of the proposed framework for real-world, label-scarce medical imaging applications.
>
---
#### [replaced 030] FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving
- **分类: cs.CV**

- **链接: []()**

> **作者:** Shuang Zeng; Xinyuan Chang; Mengwei Xie; Xinran Liu; Yifan Bai; Zheng Pan; Mu Xu; Xing Wei; Ning Guo
>
> **备注:** Accepted to NeurIPS 2025 as Spotlight Presentation. Code: https://github.com/MIV-XJTU/FSDrive
>
> **摘要:** Vision-Language-Action (VLA) models offer significant potential for end-to-end driving, yet their reasoning is often constrained by textual Chains-of-Thought (CoT). This symbolic compression of visual information creates a modality gap between perception and planning by blurring spatio-temporal relations and discarding fine-grained cues. We introduce FSDrive, a framework that empowers VLAs to "think visually" using a novel visual spatio-temporal CoT. FSDrive first operates as a world model, generating a unified future frame that combines a predicted background with explicit, physically-plausible priors like future lane dividers and 3D object boxes. This imagined scene serves as the visual spatio-temporal CoT, capturing both spatial structure and temporal evolution in a single representation. The same VLA then functions as an inverse-dynamics model to plan trajectories conditioned on current observations and this visual CoT. We enable this with a unified pre-training paradigm that expands the model's vocabulary with visual tokens and jointly optimizes for semantic understanding (VQA) and future-frame prediction. A progressive curriculum first generates structural priors to enforce physical laws before rendering the full scene. Evaluations on nuScenes and NAVSIM show FSDrive improves trajectory accuracy and reduces collisions, while also achieving competitive FID for video generation with a lightweight autoregressive model and advancing scene understanding on DriveLM. These results confirm that our visual spatio-temporal CoT bridges the perception-planning gap, enabling safer, more anticipatory autonomous driving. Code is available at https://github.com/MIV-XJTU/FSDrive.
>
---
#### [replaced 031] Towards Understanding the Mechanisms of Classifier-Free Guidance
- **分类: cs.CV**

- **链接: []()**

> **作者:** Xiang Li; Rongrong Wang; Qing Qu
>
> **摘要:** Classifier-free guidance (CFG) is a core technique powering state-of-the-art image generation systems, yet its underlying mechanisms remain poorly understood. In this work, we begin by analyzing CFG in a simplified linear diffusion model, where we show its behavior closely resembles that observed in the nonlinear case. Our analysis reveals that linear CFG improves generation quality via three distinct components: (i) a mean-shift term that approximately steers samples in the direction of class means, (ii) a positive Contrastive Principal Components (CPC) term that amplifies class-specific features, and (iii) a negative CPC term that suppresses generic features prevalent in unconditional data. We then verify these insights in real-world, nonlinear diffusion models: over a broad range of noise levels, linear CFG resembles the behavior of its nonlinear counterpart. Although the two eventually diverge at low noise levels, we discuss how the insights from the linear analysis still shed light on the CFG's mechanism in the nonlinear regime.
>
---
#### [replaced 032] DynaSolidGeo: A Dynamic Benchmark for Genuine Spatial Mathematical Reasoning of VLMs in Solid Geometry
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: []()**

> **作者:** Changti Wu; Shijie Lian; Zihao Liu; Lei Zhang; Laurence Tianruo Yang; Kai Chen
>
> **备注:** The code and dataset are available at \href{https://zgca-ai4edu.github.io/DynaSolidGeo/}{DynaSolidGeo}
>
> **摘要:** Solid geometry problem solving demands spatial mathematical reasoning that integrates spatial intelligence and symbolic reasoning. However, most existing multimodal mathematical reasoning benchmarks focus primarily on 2D plane geometry, rely on static datasets prone to data contamination and memorization, and evaluate models solely by final answers, overlooking the reasoning process. To address these limitations, we introduce DynaSolidGeo, the first dynamic benchmark for evaluating genuine spatial reasoning in Vision-Language Models (VLMs). Constructed through a semi-automatic annotation pipeline, DynaSolidGeo contains 503 expert-curated seed questions that can, in principle, dynamically generate an unbounded number of diverse multimodal text-visual instances. Beyond answer accuracy, we incorporate process evaluation based on expert-annotated reasoning chains to measure logical validity and causal coherence. Experiments across representative open-source and closed-source VLMs reveal large performance gaps, severe degradation in dynamic settings, and poor performance on tasks requiring high-level spatial intelligence, such as mental rotation and visualization. The code and dataset are available at \href{https://zgca-ai4edu.github.io/DynaSolidGeo/}{DynaSolidGeo}.
>
---
#### [replaced 033] LidarPainter: One-Step Away From Any Lidar View To Novel Guidance
- **分类: cs.CV**

- **链接: []()**

> **作者:** Yuzhou Ji; Ke Ma; Hong Cai; Anchun Zhang; Lizhuang Ma; Xin Tan
>
> **摘要:** Dynamic driving scene reconstruction is of great importance in fields like digital twin system and autonomous driving simulation. However, unacceptable degradation occurs when the view deviates from the input trajectory, leading to corrupted background and vehicle models. To improve reconstruction quality on novel trajectory, existing methods are subject to various limitations including inconsistency, deformation, and time consumption. This paper proposes LidarPainter, a one-step diffusion model that recovers consistent driving views from sparse LiDAR condition and artifact-corrupted renderings in real-time, enabling high-fidelity lane shifts in driving scene reconstruction. Extensive experiments show that LidarPainter outperforms state-of-the-art methods in speed, quality and resource efficiency, specifically 7 x faster than StreetCrafter with only one fifth of GPU memory required. LidarPainter also supports stylized generation using text prompts such as "foggy" and "night", allowing for a diverse expansion of the existing asset library.
>
---
#### [replaced 034] UniMapGen: A Generative Framework for Large-Scale Map Construction from Multi-modal Data
- **分类: cs.CV**

- **链接: []()**

> **作者:** Yujian Yuan; Changjie Wu; Xinyuan Chang; Sijin Wang; Hang Zhang; Shiyi Liang; Shuang Zeng; Mu Xu; Ning Guo
>
> **备注:** AAAI2026 Oral
>
> **摘要:** Large-scale map construction plays a vital role in applications like autonomous driving and navigation systems. Traditional large-scale map construction approaches mainly rely on costly and inefficient special data collection vehicles and labor-intensive annotation processes. While existing satellite-based methods have demonstrated promising potential in enhancing the efficiency and coverage of map construction, they exhibit two major limitations: (1) inherent drawbacks of satellite data (e.g., occlusions, outdatedness) and (2) inefficient vectorization from perception-based methods, resulting in discontinuous and rough roads that require extensive post-processing. This paper presents a novel generative framework, UniMapGen, for large-scale map construction, offering three key innovations: (1) representing lane lines as \textbf{discrete sequence} and establishing an iterative strategy to generate more complete and smooth map vectors than traditional perception-based methods. (2) proposing a flexible architecture that supports \textbf{multi-modal} inputs, enabling dynamic selection among BEV, PV, and text prompt, to overcome the drawbacks of satellite data. (3) developing a \textbf{state update} strategy for global continuity and consistency of the constructed large-scale map. UniMapGen achieves state-of-the-art performance on the OpenSatMap dataset. Furthermore, UniMapGen can infer occluded roads and predict roads missing from dataset annotations. Our code will be released.
>
---
#### [replaced 035] CSPCL: Category Semantic Prior Contrastive Learning for Deformable DETR-Based Prohibited Item Detectors
- **分类: cs.CV**

- **链接: []()**

> **作者:** Mingyuan Li; Tong Jia; Hao Wang; Bowen Ma; Hui Lu; Shiyi Guo; Da Cai; Dongyue Chen
>
> **备注:** 22 pages, 5 figures
>
> **摘要:** Prohibited item detection based on X-ray images is one of the most effective security inspection methods. However, the foreground-background feature coupling caused by the overlapping phenomenon specific to X-ray images makes general detectors designed for natural images perform poorly. To address this issue, we propose a Category Semantic Prior Contrastive Learning (CSPCL) mechanism, which aligns the class prototypes perceived by the classifier with the content queries to correct and supplement the missing semantic information responsible for classification, thereby enhancing the model sensitivity to foreground features. To achieve this alignment, we design a specific contrastive loss, CSP loss, which comprises the Intra-Class Truncated Attraction (ITA) loss and the Inter-Class Adaptive Repulsion (IAR) loss, and outperforms classic contrastive losses. Specifically, the ITA loss leverages class prototypes to attract intra-class content queries and preserves essential intra-class diversity via a gradient truncation function. The IAR loss employs class prototypes to adaptively repel inter-class content queries, with the repulsion strength scaled by prototype-prototype similarity, thereby improving inter-class discriminability, especially among similar categories. CSPCL is general and can be easily integrated into Deformable DETR-based models. Extensive experiments on the PIXray, OPIXray, PIDray, and CLCXray datasets demonstrate that CSPCL significantly enhances the performance of various state-of-the-art models without increasing inference complexity. The code is publicly available at https://github.com/Limingyuan001/CSPCL.
>
---
#### [replaced 036] Oh That Looks Familiar: A Novel Similarity Measure for Spreadsheet Template Discovery
- **分类: cs.LG; cs.CV**

- **链接: []()**

> **作者:** Anand Krishnakumar; Vengadesh Ravikumaran
>
> **备注:** 5 pages, 2 figures, Accepted to EurIPS'25: AI for Tabular Data Workshop
>
> **摘要:** Traditional methods for identifying structurally similar spreadsheets fail to capture the spatial layouts and type patterns defining templates. To quantify spreadsheet similarity, we introduce a hybrid distance metric that combines semantic embeddings, data type information, and spatial positioning. In order to calculate spreadsheet similarity, our method converts spreadsheets into cell-level embeddings and then uses aggregation techniques like Chamfer and Hausdorff distances. Experiments across template families demonstrate superior unsupervised clustering performance compared to the graph-based Mondrian baseline, achieving perfect template reconstruction (Adjusted Rand Index of 1.00 versus 0.90) on the FUSTE dataset. Our approach facilitates large-scale automated template discovery, which in turn enables downstream applications such as retrieval-augmented generation over tabular collections, model training, and bulk data cleaning.
>
---
#### [replaced 037] RealRep: Generalized SDR-to-HDR Conversion via Attribute-Disentangled Representation Learning
- **分类: cs.CV**

- **链接: []()**

> **作者:** Li Xu; Siqi Wang; Kepeng Xu; Gang He; Lin Zhang; Weiran Wang; Yu-Wing Tai
>
> **备注:** Published on AAAI'26(Oral): The Annual AAAI Conference on Artificial Intelligence
>
> **摘要:** High-Dynamic-Range Wide-Color-Gamut (HDR-WCG) technology is becoming increasingly widespread, driving a growing need for converting Standard Dynamic Range (SDR) content to HDR. Existing methods primarily rely on fixed tone mapping operators, which struggle to handle the diverse appearances and degradations commonly present in real-world SDR content. To address this limitation, we propose a generalized SDR-to-HDR framework that enhances robustness by learning attribute-disentangled representations. Central to our approach is Realistic Attribute-Disentangled Representation Learning (RealRep), which explicitly disentangles luminance and chrominance components to capture intrinsic content variations across different SDR distributions. Furthermore, we design a Luma-/Chroma-aware negative exemplar generation strategy that constructs degradation-sensitive contrastive pairs, effectively modeling tone discrepancies across SDR styles. Building on these attribute-level priors, we introduce the Degradation-Domain Aware Controlled Mapping Network (DDACMNet), a lightweight, two-stage framework that performs adaptive hierarchical mapping guided by a control-aware normalization mechanism. DDACMNet dynamically modulates the mapping process via degradation-conditioned features, enabling robust adaptation across diverse degradation domains. Extensive experiments demonstrate that RealRep consistently outperforms state-of-the-art methods in both generalization and perceptually faithful HDR color gamut reconstruction.
>
---
#### [replaced 038] Towards Visual Grounding: A Survey
- **分类: cs.CV**

- **链接: []()**

> **作者:** Linhui Xiao; Xiaoshan Yang; Xiangyuan Lan; Yaowei Wang; Changsheng Xu
>
> **备注:** Accepted by TPAMI 2025. We keep tracing related works at https://github.com/linhuixiao/Awesome-Visual-Grounding
>
> **摘要:** Visual Grounding, also known as Referring Expression Comprehension and Phrase Grounding, aims to ground the specific region(s) within the image(s) based on the given expression text. This task simulates the common referential relationships between visual and linguistic modalities, enabling machines to develop human-like multimodal comprehension capabilities. Consequently, it has extensive applications in various domains. However, since 2021, visual grounding has witnessed significant advancements, with emerging new concepts such as grounded pre-training, grounding multimodal LLMs, generalized visual grounding, and giga-pixel grounding, which have brought numerous new challenges. In this survey, we first examine the developmental history of visual grounding and provide an overview of essential background knowledge. We systematically track and summarize the advancements, and then meticulously define and organize the various settings to standardize future research and ensure a fair comparison. Additionally, we delve into numerous related datasets and applications, and highlight several advanced topics. Finally, we outline the challenges confronting visual grounding and propose valuable directions for future research, which may serve as inspiration for subsequent researchers. By extracting common technical details, this survey encompasses the representative work in each subtopic over the past decade. To the best of our knowledge, this paper represents the most comprehensive overview currently available in the field of visual grounding. This survey is designed to be suitable for both beginners and experienced researchers, serving as an invaluable resource for understanding key concepts and tracking the latest research developments. We keep tracing related work at https://github.com/linhuixiao/Awesome-Visual-Grounding.
>
---
#### [replaced 039] Filling of incomplete sinograms from sparse PET detector configurations using a residual U-Net
- **分类: eess.IV; cs.CV; physics.med-ph**

- **链接: []()**

> **作者:** Klara Leffler; Luigi Tommaso Luppino; Samuel Kuttner; Karin Söderkvist; Jan Axelsson
>
> **备注:** 18 pages, 11 figures
>
> **摘要:** Long axial field-of-view PET scanners offer increased field-of-view and sensitivity compared to traditional PET scanners. However, a significant cost is associated with the densely packed photodetectors required for the extended-coverage systems, limiting clinical utilisation. To mitigate the cost limitations, alternative sparse system configurations have been proposed, allowing an extended field-of-view PET design with detector costs similar to a standard PET system, albeit at the expense of image quality. In this work, we propose a deep sinogram restoration network to fill in the missing sinogram data. Our method utilises a modified Residual U-Net, trained on clinical PET scans from a GE Signa PET/MR, simulating the removal of 50% of the detectors in a chessboard pattern (retaining only 25% of all lines of response). The model successfully recovers missing counts, with a mean absolute error below two events per pixel, outperforming 2D interpolation in both sinogram and reconstructed image domain. Notably, the predicted sinograms exhibit a smoothing effect, leading to reconstructed images lacking sharpness in finer details. Despite these limitations, the model demonstrates a substantial capacity for compensating for the undersampling caused by the sparse detector configuration. This proof-of-concept study suggests that sparse detector configurations, combined with deep learning techniques, offer a viable alternative to conventional PET scanner designs. This approach supports the development of cost-effective, total body PET scanners, allowing a significant step forward in medical imaging technology.
>
---
#### [replaced 040] TiS-TSL: Image-Label Supervised Surgical Video Stereo Matching via Time-Switchable Teacher-Student Learning
- **分类: cs.CV; cs.AI**

- **链接: []()**

> **作者:** Rui Wang; Ying Zhou; Hao Wang; Wenwei Zhang; Qiang Li; Zhiwei Wang
>
> **备注:** 8 pages, 4 figures, accepted by BIBM2025
>
> **摘要:** Stereo matching in minimally invasive surgery (MIS) is essential for next-generation navigation and augmented reality. Yet, dense disparity supervision is nearly impossible due to anatomical constraints, typically limiting annotations to only a few image-level labels acquired before the endoscope enters deep body cavities. Teacher-Student Learning (TSL) offers a promising solution by leveraging a teacher trained on sparse labels to generate pseudo labels and associated confidence maps from abundant unlabeled surgical videos. However, existing TSL methods are confined to image-level supervision, providing only spatial confidence and lacking temporal consistency estimation. This absence of spatio-temporal reliability results in unstable disparity predictions and severe flickering artifacts across video frames. To overcome these challenges, we propose TiS-TSL, a novel time-switchable teacher-student learning framework for video stereo matching under minimal supervision. At its core is a unified model that operates in three distinct modes: Image-Prediction (IP), Forward Video-Prediction (FVP), and Backward Video-Prediction (BVP), enabling flexible temporal modeling within a single architecture. Enabled by this unified model, TiS-TSL adopts a two-stage learning strategy. The Image-to-Video (I2V) stage transfers sparse image-level knowledge to initialize temporal modeling. The subsequent Video-to-Video (V2V) stage refines temporal disparity predictions by comparing forward and backward predictions to calculate bidirectional spatio-temporal consistency. This consistency identifies unreliable regions across frames, filters noisy video-level pseudo labels, and enforces temporal coherence. Experimental results on two public datasets demonstrate that TiS-TSL exceeds other image-based state-of-the-arts by improving TEPE and EPE by at least 2.11% and 4.54%, respectively.
>
---
#### [replaced 041] Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning
- **分类: cs.CV**

- **链接: []()**

> **作者:** Lingfeng He; De Cheng; Di Xu; Huaijie Wang; Nannan Wang
>
> **备注:** AAAI-2026 Poster
>
> **摘要:** Continual learning (CL) aims to equip models with the ability to learn from a stream of tasks without forgetting previous knowledge. With the progress of vision-language models like Contrastive Language-Image Pre-training (CLIP), their promise for CL has attracted increasing attention due to their strong generalizability. However, the potential of rich textual semantic priors in CLIP in addressing the stability-plasticity dilemma remains underexplored. During backbone training, most approaches transfer past knowledge without considering semantic relevance, leading to interference from unrelated tasks that disrupt the balance between stability and plasticity. Besides, while text-based classifiers provide strong generalization, they suffer from limited plasticity due to the inherent modality gap in CLIP. Visual classifiers help bridge this gap, but their prototypes lack rich and precise semantics. To address these challenges, we propose Semantic-Enriched Continual Adaptation (SECA), a unified framework that harnesses the anti-forgetting and structured nature of textual priors to guide semantic-aware knowledge transfer in the backbone and reinforce the semantic structure of the visual classifier. Specifically, a Semantic-Guided Adaptive Knowledge Transfer (SG-AKT) module is proposed to assess new images' relevance to diverse historical visual knowledge via textual cues, and aggregate relevant knowledge in an instance-adaptive manner as distillation signals. Moreover, a Semantic-Enhanced Visual Prototype Refinement (SE-VPR) module is introduced to refine visual prototypes using inter-class semantic relations captured in class-wise textual embeddings. Extensive experiments on multiple benchmarks validate the effectiveness of our approach.
>
---
#### [replaced 042] Breaking the Stealth-Potency Trade-off in Clean-Image Backdoors with Generative Trigger Optimization
- **分类: cs.CV; cs.CR; cs.LG**

- **链接: []()**

> **作者:** Binyan Xu; Fan Yang; Di Tang; Xilin Dai; Kehuan Zhang
>
> **备注:** 19 pages, 22 figures, 15 tables. To appear in AAAI '26 (Oral). This paper extends the AAAI-2026 version by including the Appendix
>
> **摘要:** Clean-image backdoor attacks, which use only label manipulation in training datasets to compromise deep neural networks, pose a significant threat to security-critical applications. A critical flaw in existing methods is that the poison rate required for a successful attack induces a proportional, and thus noticeable, drop in Clean Accuracy (CA), undermining their stealthiness. This paper presents a new paradigm for clean-image attacks that minimizes this accuracy degradation by optimizing the trigger itself. We introduce Generative Clean-Image Backdoors (GCB), a framework that uses a conditional InfoGAN to identify naturally occurring image features that can serve as potent and stealthy triggers. By ensuring these triggers are easily separable from benign task-related features, GCB enables a victim model to learn the backdoor from an extremely small set of poisoned examples, resulting in a CA drop of less than 1%. Our experiments demonstrate GCB's remarkable versatility, successfully adapting to six datasets, five architectures, and four tasks, including the first demonstration of clean-image backdoors in regression and segmentation. GCB also exhibits resilience against most of the existing backdoor defenses.
>
---
#### [replaced 043] Unveiling Visual Perception in Language Models: An Attention Head Analysis Approach
- **分类: cs.CV**

- **链接: []()**

> **作者:** Jing Bi; Junjia Guo; Yunlong Tang; Lianggong Bruce Wen; Zhang Liu; Chenliang Xu
>
> **摘要:** Recent advancements in Multimodal Large Language Models (MLLMs) have demonstrated remarkable progress in visual understanding. This impressive leap raises a compelling question: how can language models, initially trained solely on linguistic data, effectively interpret and process visual content? This paper aims to address this question with systematic investigation across 4 model families and 4 model scales, uncovering a unique class of attention heads that focus specifically on visual content. Our analysis reveals a strong correlation between the behavior of these attention heads, the distribution of attention weights, and their concentration on visual tokens within the input. These findings enhance our understanding of how LLMs adapt to multimodal tasks, demonstrating their potential to bridge the gap between textual and visual understanding. This work paves the way for the development of AI systems capable of engaging with diverse modalities.
>
---
#### [replaced 044] Enhancing Diffusion Model Guidance through Calibration and Regularization
- **分类: cs.CV; cs.AI; cs.IT; cs.LG; eess.IV**

- **链接: []()**

> **作者:** Seyed Alireza Javid; Amirhossein Bagheri; Nuria González-Prelcic
>
> **备注:** Accepted from NeurIPS 2025 Workshop on Structured Probabilistic Inference & Generative Modeling. Code available at https://github.com/ajavid34/guided-info-diffusion
>
> **摘要:** Classifier-guided diffusion models have emerged as a powerful approach for conditional image generation, but they suffer from overconfident predictions during early denoising steps, causing the guidance gradient to vanish. This paper introduces two complementary contributions to address this issue. First, we propose a differentiable calibration objective based on the Smooth Expected Calibration Error (Smooth ECE), which improves classifier calibration with minimal fine-tuning and yields measurable improvements in Frechet Inception Distance (FID). Second, we develop enhanced sampling guidance methods that operate on off-the-shelf classifiers without requiring retraining. These include tilted sampling with batch-level reweighting, adaptive entropy-regularized sampling to preserve diversity, and a novel f-divergence-based sampling strategy that strengthens class-consistent guidance while maintaining mode coverage. Experiments on ImageNet 128x128 demonstrate that our divergence-regularized guidance achieves an FID of 2.13 using a ResNet-101 classifier, improving upon existing classifier-guided diffusion methods while requiring no diffusion model retraining. The results show that principled calibration and divergence-aware sampling provide practical and effective improvements for classifier-guided diffusion.
>
---
#### [replaced 045] X-LeBench: A Benchmark for Extremely Long Egocentric Video Understanding
- **分类: cs.CV**

- **链接: []()**

> **作者:** Wenqi Zhou; Kai Cao; Hao Zheng; Yunze Liu; Xinyi Zheng; Miao Liu; Per Ola Kristensson; Walterio Mayol-Cuevas; Fan Zhang; Weizhe Lin; Junxiao Shen
>
> **摘要:** Long-form egocentric video understanding provides rich contextual information and unique insights into long-term human behaviors, holding significant potential for applications in embodied intelligence, long-term activity analysis, and personalized assistive technologies. However, existing benchmark datasets primarily focus on single, short (\eg, minutes to tens of minutes) to moderately long videos, leaving a substantial gap in evaluating extensive, ultra-long egocentric video recordings. To address this, we introduce X-LeBench, a novel benchmark dataset meticulously designed to fill this gap by focusing on tasks requiring a comprehensive understanding of extremely long egocentric video recordings. Our X-LeBench develops a life-logging simulation pipeline that produces realistic, coherent daily plans aligned with real-world video data. This approach enables the flexible integration of synthetic daily plans with real-world footage from Ego4D-a massive-scale egocentric video dataset covers a wide range of daily life scenarios-resulting in 432 simulated video life logs spanning from 23 minutes to 16.4 hours. The evaluations of several baseline systems and multimodal large language models (MLLMs) reveal their poor performance across the board, highlighting the inherent challenges of long-form egocentric video understanding, such as temporal localization and reasoning, context aggregation, and memory retention, and underscoring the need for more advanced models.
>
---
#### [replaced 046] OSWorld-MCP: Benchmarking MCP Tool Invocation In Computer-Use Agents
- **分类: cs.CV**

- **链接: []()**

> **作者:** Hongrui Jia; Jitong Liao; Xi Zhang; Haiyang Xu; Tianbao Xie; Chaoya Jiang; Ming Yan; Si Liu; Wei Ye; Fei Huang
>
> **摘要:** With advances in decision-making and reasoning capabilities, multimodal agents show strong potential in computer application scenarios. Past evaluations have mainly assessed GUI interaction skills, while tool invocation abilities, such as those enabled by the Model Context Protocol (MCP), have been largely overlooked. Comparing agents with integrated tool invocation to those evaluated only on GUI interaction is inherently unfair. We present OSWorld-MCP, the first comprehensive and fair benchmark for assessing computer-use agents' tool invocation, GUI operation, and decision-making abilities in a real-world environment. We design a novel automated code-generation pipeline to create tools and combine them with a curated selection from existing tools. Rigorous manual validation yields 158 high-quality tools (covering 7 common applications), each verified for correct functionality, practical applicability, and versatility. Extensive evaluations of state-of-the-art multimodal agents on OSWorld-MCP show that MCP tools generally improve task success rates (e.g., from 8.3% to 20.4% for OpenAI o3 at 15 steps, from 40.1% to 43.3% for Claude 4 Sonnet at 50 steps), underscoring the importance of assessing tool invocation capabilities. However, even the strongest models have relatively low tool invocation rates, Only 36.3%, indicating room for improvement and highlighting the benchmark's challenge. By explicitly measuring MCP tool usage skills, OSWorld-MCP deepens understanding of multimodal agents and sets a new standard for evaluating performance in complex, tool-assisted environments. Our code, environment, and data are publicly available at https://osworld-mcp.github.io.
>
---
#### [replaced 047] RSVG-ZeroOV: Exploring a Training-Free Framework for Zero-Shot Open-Vocabulary Visual Grounding in Remote Sensing Images
- **分类: cs.CV; cs.AI**

- **链接: []()**

> **作者:** Ke Li; Di Wang; Ting Wang; Fuyu Dong; Yiming Zhang; Luyao Zhang; Xiangyu Wang; Shaofeng Li; Quan Wang
>
> **备注:** This work is accepted by AAAI 2026
>
> **摘要:** Remote sensing visual grounding (RSVG) aims to localize objects in remote sensing images based on free-form natural language expressions. Existing approaches are typically constrained to closed-set vocabularies, limiting their applicability in open-world scenarios. While recent attempts to leverage generic foundation models for open-vocabulary RSVG, they overly rely on expensive high-quality datasets and time-consuming fine-tuning. To address these limitations, we propose \textbf{RSVG-ZeroOV}, a training-free framework that aims to explore the potential of frozen generic foundation models for zero-shot open-vocabulary RSVG. Specifically, RSVG-ZeroOV comprises three key stages: (i) Overview: We utilize a vision-language model (VLM) to obtain cross-attention\footnote[1]{In this paper, although decoder-only VLMs use self-attention over all tokens, we refer to the image-text interaction part as cross-attention to distinguish it from pure visual self-attention.}maps that capture semantic correlations between text queries and visual regions. (ii) Focus: By leveraging the fine-grained modeling priors of a diffusion model (DM), we fill in gaps in structural and shape information of objects, which are often overlooked by VLM. (iii) Evolve: A simple yet effective attention evolution module is introduced to suppress irrelevant activations, yielding purified segmentation masks over the referred objects. Without cumbersome task-specific training, RSVG-ZeroOV offers an efficient and scalable solution. Extensive experiments demonstrate that the proposed framework consistently outperforms existing weakly-supervised and zero-shot methods.
>
---
#### [replaced 048] Visual Explanation via Similar Feature Activation for Metric Learning
- **分类: cs.CV**

- **链接: []()**

> **作者:** Yi Liao; Ugochukwu Ejike Akpudo; Jue Zhang; Yongsheng Gao; Jun Zhou; Wenyi Zeng; Weichuan Zhang
>
> **摘要:** Visual explanation maps enhance the trustworthiness of decisions made by deep learning models and offer valuable guidance for developing new algorithms in image recognition tasks. Class activation maps (CAM) and their variants (e.g., Grad-CAM and Relevance-CAM) have been extensively employed to explore the interpretability of softmax-based convolutional neural networks, which require a fully connected layer as the classifier for decision-making. However, these methods cannot be directly applied to metric learning models, as such models lack a fully connected layer functioning as a classifier. To address this limitation, we propose a novel visual explanation method termed Similar Feature Activation Map (SFAM). This method introduces the channel-wise contribution importance score (CIS) to measure feature importance, derived from the similarity measurement between two image embeddings. The explanation map is constructed by linearly combining the proposed importance weights with the feature map from a CNN model. Quantitative and qualitative experiments show that SFAM provides highly promising interpretable visual explanations for CNN models using Euclidean distance or cosine similarity as the similarity metric.
>
---
#### [replaced 049] T-GVC: Trajectory-Guided Generative Video Coding at Ultra-Low Bitrates
- **分类: cs.CV; cs.MM**

- **链接: []()**

> **作者:** Zhitao Wang; Hengyu Man; Wenrui Li; Xingtao Wang; Xiaopeng Fan; Debin Zhao
>
> **摘要:** Recent advances in video generation techniques have given rise to an emerging paradigm of generative video coding for Ultra-Low Bitrate (ULB) scenarios by leveraging powerful generative priors. However, most existing methods are limited by domain specificity (e.g., facial or human videos) or excessive dependence on high-level text guidance, which tend to inadequately capture fine-grained motion details, leading to unrealistic or incoherent reconstructions. To address these challenges, we propose Trajectory-Guided Generative Video Coding (dubbed T-GVC), a novel framework that bridges low-level motion tracking with high-level semantic understanding. T-GVC features a semantic-aware sparse motion sampling pipeline that extracts pixel-wise motion as sparse trajectory points based on their semantic importance, significantly reducing the bitrate while preserving critical temporal semantic information. In addition, by integrating trajectory-aligned loss constraints into diffusion processes, we introduce a training-free guidance mechanism in latent space to ensure physically plausible motion patterns without sacrificing the inherent capabilities of generative models. Experimental results demonstrate that T-GVC outperforms both traditional and neural video codecs under ULB conditions. Furthermore, additional experiments confirm that our framework achieves more precise motion control than existing text-guided methods, paving the way for a novel direction of generative video coding guided by geometric motion modeling.
>
---
#### [replaced 050] WildFireCan-MMD: A Multimodal Dataset for Classification of User-Generated Content During Wildfires in Canada
- **分类: cs.CV; cs.AI**

- **链接: []()**

> **作者:** Braeden Sherritt; Isar Nejadgholi; Efstratios Aivaliotis; Khaled Mslmani; Marzieh Amini
>
> **摘要:** Rapid information access is vital during wildfires, yet traditional data sources are slow and costly. Social media offers real-time updates, but extracting relevant insights remains a challenge. In this work, we focus on multimodal wildfire social media data, which, although existing in current datasets, is currently underrepresented in Canadian contexts. We present WildFireCan-MMD, a new multimodal dataset of X posts from recent Canadian wildfires, annotated across twelve key themes. We evaluate zero-shot vision-language models on this dataset and compare their results with those of custom-trained and baseline classifiers. We show that while baseline methods and zero-shot prompting offer quick deployment, custom-trained models outperform them when labelled data is available. Our best-performing custom model reaches 84.48% f-score, outperforming VLMs and baseline classifiers. We also demonstrate how this model can be used to uncover trends during wildfires, through the collection and analysis of a large unlabeled dataset. Our dataset facilitates future research in wildfire response, and our findings highlight the importance of tailored datasets and task-specific training. Importantly, such datasets should be localized, as disaster response requirements vary across regions and contexts.
>
---
#### [replaced 051] Generalizable Holographic Reconstruction via Amplitude-Only Diffusion Priors
- **分类: physics.optics; cs.CV; cs.LG**

- **链接: []()**

> **作者:** Jeongsol Kim; Chanseok Lee; Jongin You; Jong Chul Ye; Mooseok Jang
>
> **备注:** Keywords: Diffusion model, phase retrieval, inline-holography, inverse problem
>
> **摘要:** Phase retrieval in inline holography is a fundamental yet ill-posed inverse problem due to the nonlinear coupling between amplitude and phase in coherent imaging. We present a novel off-the-shelf solution that leverages a diffusion model trained solely on object amplitude to recover both amplitude and phase from diffraction intensities. Using a predictor-corrector sampling framework with separate likelihood gradients for amplitude and phase, our method enables complex field reconstruction without requiring ground-truth phase data for training. We validate the proposed approach through extensive simulations and experiments, demonstrating robust generalization across diverse object shapes, imaging system configurations, and modalities, including lensless setups. Notably, a diffusion prior trained on simple amplitude data (e.g., polystyrene beads) successfully reconstructs complex biological tissue structures, highlighting the method's adaptability. This framework provides a cost-effective, generalizable solution for nonlinear inverse problems in computational imaging, and establishes a foundation for broader coherent imaging applications beyond holography.
>
---
#### [replaced 052] An Artificial Intelligence-based Assistant for the Visually Impaired
- **分类: cs.CV; cs.CY; cs.HC**

- **链接: []()**

> **作者:** Luis Marquez-Carpintero; Francisco Gomez-Donoso; Zuria Bauer; Bessie Dominguez-Dager; Alvaro Belmonte-Baeza; Mónica Pina-Navarro; Francisco Morillas-Espejo; Felix Escalona; Miguel Cazorla
>
> **摘要:** This paper describes an artificial intelligence-based assistant application, AIDEN, developed during 2023 and 2024, aimed at improving the quality of life for visually impaired individuals. Visually impaired individuals face challenges in identifying objects, reading text, and navigating unfamiliar environments, which can limit their independence and reduce their quality of life. Although solutions such as Braille, audio books, and screen readers exist, they may not be effective in all situations. This application leverages state-of-the-art machine learning algorithms to identify and describe objects, read text, and answer questions about the environment. Specifically, it uses You Only Look Once architectures and a Large Language and Vision Assistant. The system incorporates several methods to facilitate the user's interaction with the system and access to textual and visual information in an appropriate manner. AIDEN aims to enhance user autonomy and access to information, contributing to an improved perception of daily usability, as supported by user feedback.
>
---
#### [replaced 053] CountingDINO: A Training-free Pipeline for Class-Agnostic Counting using Unsupervised Backbones
- **分类: cs.CV**

- **链接: []()**

> **作者:** Giacomo Pacini; Lorenzo Bianchi; Luca Ciampi; Nicola Messina; Giuseppe Amato; Fabrizio Falchi
>
> **备注:** [Accepted at WACV 2026] 18 pages, 11 figures, 3 tables. Project website: https://lorebianchi98.github.io/CountingDINO/
>
> **摘要:** Class-agnostic counting (CAC) aims to estimate the number of objects in images without being restricted to predefined categories. However, while current exemplar-based CAC methods offer flexibility at inference time, they still rely heavily on labeled data for training, which limits scalability and generalization to many downstream use cases. In this paper, we introduce CountingDINO, the first training-free exemplar-based CAC framework that exploits a fully unsupervised feature extractor. Specifically, our approach employs self-supervised vision-only backbones to extract object-aware features, and it eliminates the need for annotated data throughout the entire proposed pipeline. At inference time, we extract latent object prototypes via ROI-Align from DINO features and use them as convolutional kernels to generate similarity maps. These are then transformed into density maps through a simple yet effective normalization scheme. We evaluate our approach on the FSC-147 benchmark, where we consistently outperform a baseline based on an SOTA unsupervised object detector under the same label- and training-free setting. Additionally, we achieve competitive results -- and in some cases surpass -- training-free methods that rely on supervised backbones, non-training-free unsupervised methods, as well as several fully supervised SOTA approaches. This demonstrates that label- and training-free CAC can be both scalable and effective. Code: https://lorebianchi98.github.io/CountingDINO/.
>
---
#### [replaced 054] Otter: Mitigating Background Distractions of Wide-Angle Few-Shot Action Recognition with Enhanced RWKV
- **分类: cs.CV**

- **链接: []()**

> **作者:** Wenbo Huang; Jinghui Zhang; Zhenghao Chen; Guang Li; Lei Zhang; Yang Cao; Fang Dong; Takahiro Ogawa; Miki Haseyama
>
> **备注:** Accepted by AAAI 2026 Oral
>
> **摘要:** Wide-angle videos in few-shot action recognition (FSAR) effectively express actions within specific scenarios. However, without a global understanding of both subjects and background, recognizing actions in such samples remains challenging because of the background distractions. Receptance Weighted Key Value (RWKV), which learns interaction between various dimensions, shows promise for global modeling. While directly applying RWKV to wide-angle FSAR may fail to highlight subjects due to excessive background information. Additionally, temporal relation degraded by frames with similar backgrounds is difficult to reconstruct, further impacting performance. Therefore, we design the CompOund SegmenTation and Temporal REconstructing RWKV (Otter). Specifically, the Compound Segmentation Module~(CSM) is devised to segment and emphasize key patches in each frame, effectively highlighting subjects against background information. The Temporal Reconstruction Module (TRM) is incorporated into the temporal-enhanced prototype construction to enable bidirectional scanning, allowing better reconstruct temporal relation. Furthermore, a regular prototype is combined with the temporal-enhanced prototype to simultaneously enhance subject emphasis and temporal modeling, improving wide-angle FSAR performance. Extensive experiments on benchmarks such as SSv2, Kinetics, UCF101, and HMDB51 demonstrate that Otter achieves state-of-the-art performance. Extra evaluation on the VideoBadminton dataset further validates the superiority of Otter in wide-angle FSAR.
>
---
#### [replaced 055] A Two-Stage System for Layout-Controlled Image Generation using Large Language Models and Diffusion Models
- **分类: cs.CV**

- **链接: []()**

> **作者:** Jan-Hendrik Koch; Jonas Krumme; Konrad Gadzicki
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Text-to-image diffusion models exhibit remarkable generative capabilities, but lack precise control over object counts and spatial arrangements. This work introduces a two-stage system to address these compositional limitations. The first stage employs a Large Language Model (LLM) to generate a structured layout from a list of objects. The second stage uses a layout-conditioned diffusion model to synthesize a photorealistic image adhering to this layout. We find that task decomposition is critical for LLM-based spatial planning; by simplifying the initial generation to core objects and completing the layout with rule-based insertion, we improve object recall from 57.2% to 99.9% for complex scenes. For image synthesis, we compare two leading conditioning methods: ControlNet and GLIGEN. After domain-specific finetuning on table-setting datasets, we identify a key trade-off: ControlNet preserves text-based stylistic control but suffers from object hallucination, while GLIGEN provides superior layout fidelity at the cost of reduced prompt-based controllability. Our end-to-end system successfully generates images with specified object counts and plausible spatial arrangements, demonstrating the viability of a decoupled approach for compositionally controlled synthesis.
>
---
#### [replaced 056] CoCoLIT: ControlNet-Conditioned Latent Image Translation for MRI to Amyloid PET Synthesis
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: []()**

> **作者:** Alec Sargood; Lemuel Puglisi; James H. Cole; Neil P. Oxtoby; Daniele Ravì; Daniel C. Alexander
>
> **备注:** Article accepted at AAAI-2026
>
> **摘要:** Synthesizing amyloid PET scans from the more widely available and accessible structural MRI modality offers a promising, cost-effective approach for large-scale Alzheimer's Disease (AD) screening. This is motivated by evidence that, while MRI does not directly detect amyloid pathology, it may nonetheless encode information correlated with amyloid deposition that can be uncovered through advanced modeling. However, the high dimensionality and structural complexity of 3D neuroimaging data pose significant challenges for existing MRI-to-PET translation methods. Modeling the cross-modality relationship in a lower-dimensional latent space can simplify the learning task and enable more effective translation. As such, we present CoCoLIT (ControlNet-Conditioned Latent Image Translation), a diffusion-based latent generative framework that incorporates three main innovations: (1) a novel Weighted Image Space Loss (WISL) that improves latent representation learning and synthesis quality; (2) a theoretical and empirical analysis of Latent Average Stabilization (LAS), an existing technique used in similar generative models to enhance inference consistency; and (3) the introduction of ControlNet-based conditioning for MRI-to-PET translation. We evaluate CoCoLIT's performance on publicly available datasets and find that our model significantly outperforms state-of-the-art methods on both image-based and amyloid-related metrics. Notably, in amyloid-positivity classification, CoCoLIT outperforms the second-best method with improvements of +10.5% on the internal dataset and +23.7% on the external dataset. The code and models of our approach are available at https://github.com/brAIn-science/CoCoLIT.
>
---
#### [replaced 057] Systematic Literature Review on Vehicular Collaborative Perception - A Computer Vision Perspective
- **分类: cs.CV**

- **链接: []()**

> **作者:** Lei Wan; Jianxin Zhao; Andreas Wiedholz; Manuel Bied; Mateus Martinez de Lucena; Abhishek Dinkar Jagtap; Andreas Festag; Antônio Augusto Fröhlich; Hannan Ejaz Keen; Alexey Vinel
>
> **备注:** 38 pages, 8 figures, accepted for publication in IEEE Transactions on Intelligent Transportation Systems (T-ITS)
>
> **摘要:** The effectiveness of autonomous vehicles relies on reliable perception capabilities. Despite significant advancements in artificial intelligence and sensor fusion technologies, current single-vehicle perception systems continue to encounter limitations, notably visual occlusions and limited long-range detection capabilities. Collaborative Perception (CP), enabled by Vehicle-to-Vehicle (V2V) and Vehicle-to-Infrastructure (V2I) communication, has emerged as a promising solution to mitigate these issues and enhance the reliability of autonomous systems. Beyond advancements in communication, the computer vision community is increasingly focusing on improving vehicular perception through collaborative approaches. However, a systematic literature review that thoroughly examines existing work and reduces subjective bias is still lacking. Such a systematic approach helps identify research gaps, recognize common trends across studies, and inform future research directions. In response, this study follows the PRISMA 2020 guidelines and includes 106 peer-reviewed articles. These publications are analyzed based on modalities, collaboration schemes, and key perception tasks. Through a comparative analysis, this review illustrates how different methods address practical issues such as pose errors, temporal latency, communication constraints, domain shifts, heterogeneity, and adversarial attacks. Furthermore, it critically examines evaluation methodologies, highlighting a misalignment between current metrics and CP's fundamental objectives. By delving into all relevant topics in-depth, this review offers valuable insights into challenges, opportunities, and risks, serving as a reference for advancing research in vehicular collaborative perception.
>
---
#### [replaced 058] EndoIR: Degradation-Agnostic All-in-One Endoscopic Image Restoration via Noise-Aware Routing Diffusion
- **分类: eess.IV; cs.AI; cs.CV; cs.RO**

- **链接: []()**

> **作者:** Tong Chen; Xinyu Ma; Long Bai; Wenyang Wang; Yue Sun; Luping Zhou
>
> **摘要:** Endoscopic images often suffer from diverse and co-occurring degradations such as low lighting, smoke, and bleeding, which obscure critical clinical details. Existing restoration methods are typically task-specific and often require prior knowledge of the degradation type, limiting their robustness in real-world clinical use. We propose EndoIR, an all-in-one, degradation-agnostic diffusion-based framework that restores multiple degradation types using a single model. EndoIR introduces a Dual-Domain Prompter that extracts joint spatial-frequency features, coupled with an adaptive embedding that encodes both shared and task-specific cues as conditioning for denoising. To mitigate feature confusion in conventional concatenation-based conditioning, we design a Dual-Stream Diffusion architecture that processes clean and degraded inputs separately, with a Rectified Fusion Block integrating them in a structured, degradation-aware manner. Furthermore, Noise-Aware Routing Block improves efficiency by dynamically selecting only noise-relevant features during denoising. Experiments on SegSTRONG-C and CEC datasets demonstrate that EndoIR achieves state-of-the-art performance across multiple degradation scenarios while using fewer parameters than strong baselines, and downstream segmentation experiments confirm its clinical utility.
>
---
#### [replaced 059] OpenVLThinker: Complex Vision-Language Reasoning via Iterative SFT-RL Cycles
- **分类: cs.CV; cs.CL**

- **链接: []()**

> **作者:** Yihe Deng; Hritik Bansal; Fan Yin; Nanyun Peng; Wei Wang; Kai-Wei Chang
>
> **备注:** 23 pages, 11 figures, 8 tables
>
> **摘要:** We introduce OpenVLThinker, one of the first open-source large vision-language models (LVLMs) to exhibit sophisticated chain-of-thought reasoning, achieving notable performance gains on challenging visual reasoning tasks. While text-based reasoning models (e.g., Deepseek R1) show promising results in text-only tasks, distilling their reasoning into LVLMs via supervised fine-tuning (SFT) often results in performance degradation due to imprecise visual grounding. Conversely, purely reinforcement learning (RL)-based methods face a large search space, hindering the emergence of reflective behaviors in smaller models (e.g., 7B LVLMs). Surprisingly, alternating between SFT and RL ultimately results in significant performance improvements after a few iterations. Our analysis reveals that the base model rarely exhibits reasoning behaviors initially, but SFT effectively surfaces these latent actions and narrows the RL search space, accelerating the development of reasoning capabilities. Each subsequent RL stage further refines the model's reasoning skills, producing higher-quality SFT data for continued self-improvement. OpenVLThinker-7B consistently advances performance across six benchmarks demanding mathematical and general reasoning, notably improving MathVista by 3.8%, EMMA by 2.4%, and HallusionBench by 1.6%. Beyond demonstrating the synergy between SFT and RL for complex reasoning tasks, our findings provide early evidence towards achieving R1-style reasoning in multimodal contexts. The code, model and data are held at https://github.com/yihedeng9/OpenVLThinker.
>
---
#### [replaced 060] SPHERE: Semantic-PHysical Engaged REpresentation for 3D Semantic Scene Completion
- **分类: cs.CV**

- **链接: []()**

> **作者:** Zhiwen Yang; Yuxin Peng
>
> **备注:** 10 pages, 6 figures, accepted by ACM MM 2025
>
> **摘要:** Camera-based 3D Semantic Scene Completion (SSC) is a critical task in autonomous driving systems, assessing voxel-level geometry and semantics for holistic scene perception. While existing voxel-based and plane-based SSC methods have achieved considerable progress, they struggle to capture physical regularities for realistic geometric details. On the other hand, neural reconstruction methods like NeRF and 3DGS demonstrate superior physical awareness, but suffer from high computational cost and slow convergence when handling large-scale, complex autonomous driving scenes, leading to inferior semantic accuracy. To address these issues, we propose the Semantic-PHysical Engaged REpresentation (SPHERE) for camera-based SSC, which integrates voxel and Gaussian representations for joint exploitation of semantic and physical information. First, the Semantic-guided Gaussian Initialization (SGI) module leverages dual-branch 3D scene representations to locate focal voxels as anchors to guide efficient Gaussian initialization. Then, the Physical-aware Harmonics Enhancement (PHE) module incorporates semantic spherical harmonics to model physical-aware contextual details and promote semantic-geometry consistency through focal distribution alignment, generating SSC results with realistic details. Extensive experiments and analyses on the popular SemanticKITTI and SSCBench-KITTI-360 benchmarks validate the effectiveness of SPHERE. The code is available at https://github.com/PKU-ICST-MIPL/SPHERE_ACMMM2025.
>
---
#### [replaced 061] A Multimodal Recaptioning Framework to Account for Perceptual Diversity Across Languages in Vision-Language Modeling
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: []()**

> **作者:** Kyle Buettner; Jacob T. Emmerson; Adriana Kovashka
>
> **备注:** Accepted at IJCNLP-AACL 2025 (Main)
>
> **摘要:** When captioning an image, people describe objects in diverse ways, such as by using different terms and/or including details that are perceptually noteworthy to them. Descriptions can be especially unique across languages and cultures. Modern vision-language models (VLMs) gain understanding of images with text in different languages often through training on machine translations of English captions. However, this process relies on input content written from the perception of English speakers, leading to a perceptual bias. In this work, we outline a framework to address this bias. We specifically use a small amount of native speaker data, nearest-neighbor example guidance, and multimodal LLM reasoning to augment captions to better reflect descriptions in a target language. When adding the resulting rewrites to multilingual CLIP finetuning, we improve on German and Japanese text-image retrieval case studies (up to +3.5 mean recall, +4.4 on native vs. translation errors). We also propose a mechanism to build understanding of object description variation across languages, and offer insights into cross-dataset and cross-language generalization.
>
---
#### [replaced 062] AVAR-Net: A Lightweight Audio-Visual Anomaly Recognition Framework with a Benchmark Dataset
- **分类: cs.CV**

- **链接: []()**

> **作者:** Amjid Ali; Zulfiqar Ahmad Khan; Altaf Hussain; Muhammad Munsif; Adnan Hussain; Sung Wook Baik
>
> **备注:** I would like to request the withdrawal of my paper . The reason for this request is that I am currently working on additional experiments and analyses, which will lead to updates in the results section. Once these updates are complete, I will resubmit the revised version. Thank you for your understanding
>
> **摘要:** Anomaly recognition plays a vital role in surveillance, transportation, healthcare, and public safety. However, most existing approaches rely solely on visual data, making them unreliable under challenging conditions such as occlusion, low illumination, and adverse weather. Moreover, the absence of large-scale synchronized audio-visual datasets has hindered progress in multimodal anomaly recognition. To address these limitations, this study presents AVAR-Net, a lightweight and efficient audio-visual anomaly recognition framework designed for real-world environments. AVAR-Net consists of four main modules: an audio feature extractor, a video feature extractor, fusion strategy, and a sequential pattern learning network that models cross-modal relationships for anomaly recognition. Specifically, the Wav2Vec2 model extracts robust temporal features from raw audio, while MobileViT captures both local and global visual representations from video frames. An early fusion mechanism combines these modalities, and a Multi-Stage Temporal Convolutional Network (MTCN) model that learns long-range temporal dependencies within the fused representation, enabling robust spatiotemporal reasoning. A novel Visual-Audio Anomaly Recognition (VAAR) dataset, is also introduced, serving as a medium-scale benchmark containing 3,000 real-world videos with synchronized audio across ten diverse anomaly classes. Experimental evaluations demonstrate that AVAR-Net achieves 89.29% accuracy on VAAR and 88.56% Average Precision on the XD-Violence dataset, improving Average Precision by 2.8% over existing state-of-the-art methods. These results highlight the effectiveness, efficiency, and generalization capability of the proposed framework, as well as the utility of VAAR as a benchmark for advancing multimodal anomaly recognition research.
>
---
#### [replaced 063] SpatioTemporal Difference Network for Video Depth Super-Resolution
- **分类: cs.CV**

- **链接: []()**

> **作者:** Zhengxue Wang; Yuan Wu; Xiang Li; Zhiqiang Yan; Jian Yang
>
> **备注:** accepted by AAAI 2026
>
> **摘要:** Depth super-resolution has achieved impressive performance, and the incorporation of multi-frame information further enhances reconstruction quality. Nevertheless, statistical analyses reveal that video depth super-resolution remains affected by pronounced long-tailed distributions, with the long-tailed effects primarily manifesting in spatial non-smooth regions and temporal variation zones. To address these challenges, we propose a novel SpatioTemporal Difference Network (STDNet) comprising two core branches: a spatial difference branch and a temporal difference branch. In the spatial difference branch, we introduce a spatial difference mechanism to mitigate the long-tailed issues in spatial non-smooth regions. This mechanism dynamically aligns RGB features with learned spatial difference representations, enabling intra-frame RGB-D aggregation for depth calibration. In the temporal difference branch, we further design a temporal difference strategy that preferentially propagates temporal variation information from adjacent RGB and depth frames to the current depth frame, leveraging temporal difference representations to achieve precise motion compensation in temporal long-tailed areas. Extensive experimental results across multiple datasets demonstrate the effectiveness of our STDNet, outperforming existing approaches.
>
---
#### [replaced 064] GUI-AIMA: Aligning Intrinsic Multimodal Attention with a Context Anchor for GUI Grounding
- **分类: cs.CV; cs.AI; cs.CL; cs.HC; cs.LG**

- **链接: []()**

> **作者:** Shijie Zhou; Viet Dac Lai; Hao Tan; Jihyung Kil; Wanrong Zhu; Changyou Chen; Ruiyi Zhang
>
> **摘要:** Graphical user interface (GUI) grounding is a key function of computer-use agents, which maps natural-language instructions to actionable screen regions. Existing approaches based on Multimodal Large Language Models (MLLMs) typically formulate it as a text-based coordinate generation task, yet directly generating precise coordinates from visual inputs remains challenging and computationally intensive. An intuitive way to implement GUI grounding is to first select visual patches relevant to the instructions and then determine the precise click location within those patches. Based on the observations that general MLLMs have some native grounding capability, nested within their attentions, we propose GUI-AIMA, an attention-based and coordinate-free supervised fine-tuning framework for efficient GUI grounding. GUI-AIMA aligns the intrinsic multimodal attention of MLLMs with patch-wise grounding signals. These signals are calculated adaptively for diverse user instructions by multi-head aggregation on simplified query-visual attention matrices. Besides, its coordinate-free manner can easily integrate a plug-and-play zoom-in stage. GUI-AIMA-3B was trained with only 85k screenshots, demonstrating exceptional data efficiency and verifying that light training can trigger the native grounding capability of MLLMs. It achieves state-of-the-art performance among 3B models, attaining an average accuracy of 59.6% on ScreenSpot-Pro, 63.8% on OSWorld-G and 91.5% on ScreenSpot-v2. Project page: https://github.com/sjz5202/GUI-AIMA
>
---
#### [replaced 065] Relative Energy Learning for LiDAR Out-of-Distribution Detection
- **分类: cs.CV**

- **链接: []()**

> **作者:** Zizhao Li; Zhengkang Xiang; Jiayang Ao; Joseph West; Kourosh Khoshelham
>
> **备注:** The code and checkpoints will be released after paper acceptance
>
> **摘要:** Out-of-distribution (OOD) detection is a critical requirement for reliable autonomous driving, where safety depends on recognizing road obstacles and unexpected objects beyond the training distribution. Despite extensive research on OOD detection in 2D images, direct transfer to 3D LiDAR point clouds has been proven ineffective. Current LiDAR OOD methods struggle to distinguish rare anomalies from common classes, leading to high false-positive rates and overconfident errors in safety-critical settings. We propose Relative Energy Learning (REL), a simple yet effective framework for OOD detection in LiDAR point clouds. REL leverages the energy gap between positive (in-distribution) and negative logits as a relative scoring function, mitigating calibration issues in raw energy values and improving robustness across various scenes. To address the absence of OOD samples during training, we propose a lightweight data synthesis strategy called Point Raise, which perturbs existing point clouds to generate auxiliary anomalies without altering the inlier semantics. Evaluated on SemanticKITTI and the Spotting the Unexpected (STU) benchmark, REL consistently outperforms existing methods by a large margin. Our results highlight that modeling relative energy, combined with simple synthetic outliers, provides a principled and scalable solution for reliable OOD detection in open-world autonomous driving.
>
---
#### [replaced 066] Active Learning for Animal Re-Identification with Ambiguity-Aware Sampling
- **分类: cs.CV; cs.AI**

- **链接: []()**

> **作者:** Depanshu Sani; Mehar Khurana; Saket Anand
>
> **备注:** In Proceedings of AAAI Conference on Artificial Intelligence 2026
>
> **摘要:** Animal Re-ID has recently gained substantial attention in the AI research community due to its high impact on biodiversity monitoring and unique research challenges arising from environmental factors. The subtle distinguishing patterns, handling new species and the inherent open-set nature make the problem even harder. To address these complexities, foundation models trained on labeled, large-scale and multi-species animal Re-ID datasets have recently been introduced to enable zero-shot Re-ID. However, our benchmarking reveals significant gaps in their zero-shot Re-ID performance for both known and unknown species. While this highlights the need for collecting labeled data in new domains, exhaustive annotation for Re-ID is laborious and requires domain expertise. Our analyses show that existing unsupervised (USL) and AL Re-ID methods underperform for animal Re-ID. To address these limitations, we introduce a novel AL Re-ID framework that leverages complementary clustering methods to uncover and target structurally ambiguous regions in the embedding space for mining pairs of samples that are both informative and broadly representative. Oracle feedback on these pairs, in the form of must-link and cannot-link constraints, facilitates a simple annotation interface, which naturally integrates with existing USL methods through our proposed constrained clustering refinement algorithm. Through extensive experiments, we demonstrate that, by utilizing only 0.033% of all annotations, our approach consistently outperforms existing foundational, USL and AL baselines. Specifically, we report an average improvement of 10.49%, 11.19% and 3.99% (mAP) on 13 wildlife datasets over foundational, USL and AL methods, respectively, while attaining state-of-the-art performance on each dataset. Furthermore, we also show an improvement of 11.09%, 8.2% and 2.06% for unknown individuals in an open-world setting.
>
---
#### [replaced 067] Bee: A High-Quality Corpus and Full-Stack Suite to Unlock Advanced Fully Open MLLMs
- **分类: cs.CV; cs.AI**

- **链接: []()**

> **作者:** Yi Zhang; Bolin Ni; Xin-Sheng Chen; Heng-Rui Zhang; Yongming Rao; Houwen Peng; Qinglin Lu; Han Hu; Meng-Hao Guo; Shi-Min Hu
>
> **备注:** homepage: https://open-bee.github.io/
>
> **摘要:** Fully open multimodal large language models (MLLMs) currently lag behind proprietary counterparts, primarily due to a significant gap in data quality for supervised fine-tuning (SFT). Existing open-source datasets are often plagued by widespread noise and a critical deficit in complex reasoning data, such as Chain-of-Thought (CoT), which hinders the development of advanced model capabilities. Addressing these challenges, our work makes three primary contributions. First, we introduce Honey-Data-15M, a new SFT dataset comprising approximately 15 million QA pairs, processed through multiple cleaning techniques and enhanced with a novel dual-level (short and long) CoT enrichment strategy. Second, we introduce HoneyPipe, the data curation pipeline, and its underlying framework DataStudio, providing the community with a transparent and adaptable methodology for data curation that moves beyond static dataset releases. Finally, to validate our dataset and pipeline, we train Bee-8B, an 8B model on Honey-Data-15M. Experiments show that Bee-8B establishes a new state-of-the-art (SOTA) for fully open MLLMs, achieving performance that is competitive with, and in some cases surpasses, recent semi-open models such as InternVL3.5-8B. Our work delivers to the community a suite of foundational resources, including: the Honey-Data-15M corpus; the full-stack suite comprising HoneyPipe and DataStudio; training recipes; an evaluation harness; and the model weights. This effort demonstrates that a principled focus on data quality is a key pathway to developing fully open MLLMs that are highly competitive with their semi-open counterparts.
>
---
#### [replaced 068] PicoSAM2: Low-Latency Segmentation In-Sensor for Edge Vision Applications
- **分类: cs.CV**

- **链接: []()**

> **作者:** Pietro Bonazzi; Nicola Farronato; Stefan Zihlmann; Haotong Qin; Michele Magno
>
> **摘要:** Real-time, on-device segmentation is critical for latency-sensitive and privacy-aware applications like smart glasses and IoT devices. We introduce PicoSAM2, a lightweight (1.3M parameters, 336M MACs) promptable segmentation model optimized for edge and in-sensor execution, including the Sony IMX500. It builds on a depthwise separable U-Net, with knowledge distillation and fixed-point prompt encoding to learn from the Segment Anything Model 2 (SAM2). On COCO and LVIS, it achieves 51.9% and 44.9% mIoU, respectively. The quantized model (1.22MB) runs at 14.3 ms on the IMX500-achieving 86 MACs/cycle, making it the only model meeting both memory and compute constraints for in-sensor deployment. Distillation boosts LVIS performance by +3.5% mIoU and +5.1% mAP. These results demonstrate that efficient, promptable segmentation is feasible directly on-camera, enabling privacy-preserving vision without cloud or host processing.
>
---
#### [replaced 069] Bridged Semantic Alignment for Zero-shot 3D Medical Image Diagnosis
- **分类: cs.CV**

- **链接: []()**

> **作者:** Haoran Lai; Zihang Jiang; Qingsong Yao; Rongsheng Wang; Zhiyang He; Xiaodong Tao; Weifu Lv; Wei Wei; S. Kevin Zhou
>
> **摘要:** 3D medical images such as computed tomography are widely used in clinical practice, offering a great potential for automatic diagnosis. Supervised learning-based approaches have achieved significant progress but rely heavily on extensive manual annotations, limited by the availability of training data and the diversity of abnormality types. Vision-language alignment (VLA) offers a promising alternative by enabling zero-shot learning without additional annotations. However, we empirically discover that the visual and textural embeddings after alignment endeavors from existing VLA methods form two well-separated clusters, presenting a wide gap to be bridged. To bridge this gap, we propose a Bridged Semantic Alignment (BrgSA) framework. First, we utilize a large language model to perform semantic summarization of reports, extracting high-level semantic information. Second, we design a Cross-Modal Knowledge Interaction module that leverages a cross-modal knowledge bank as a semantic bridge, facilitating interaction between the two modalities, narrowing the gap, and improving their alignment. To comprehensively evaluate our method, we construct a benchmark dataset that includes 15 underrepresented abnormalities as well as utilize two existing benchmark datasets. Experimental results demonstrate that BrgSA achieves state-of-the-art performances on both public benchmark datasets and our custom-labeled dataset, with significant improvements in zero-shot diagnosis of underrepresented abnormalities.
>
---
#### [replaced 070] From Semantics, Scene to Instance-awareness: Distilling Foundation Model for Grounded Open-vocabulary Situation Recognition
- **分类: cs.CV**

- **链接: []()**

> **作者:** Chen Cai; Tianyi Liu; Jianjun Gao; Wenyang Liu; Kejun Wu; Ruoyu Wang; Yi Wang; Soo Chin Liew
>
> **摘要:** Recent Multimodal Large Language Models (MLLMs) exhibit strong zero-shot abilities but struggle with complex Grounded Situation Recognition (GSR) and are resource-intensive for edge device deployment. Meanwhile, conventional GSR models often lack generalization ability, falling short in recognizing unseen and rare situations. In this paper, we exploit transferring knowledge from a teacher MLLM to a small GSR model to enhance its generalization and zero-shot abilities, thereby introducing the task of Open-vocabulary Grounded Situation Recognition (Ov-GSR). To achieve this, we propose Multimodal Interactive Prompt Distillation (MIPD), a novel framework that distills enriched multimodal knowledge from the foundation model, enabling the student Ov-GSR model to recognize unseen situations and be better aware of rare situations. Specifically, the MIPD framework first leverages the LLM-based Judgmental Rationales Generator (JRG) to construct positive and negative glimpse and gaze rationales enriched with contextual semantic information. The proposed scene-aware and instance-perception prompts are then introduced to align rationales with visual information from the MLLM teacher via the Negative-Guided Multimodal Prompting Alignment (NMPA) module, effectively capturing holistic and perceptual multimodal knowledge. Finally, the aligned multimodal knowledge is distilled into the student Ov-GSR model, providing a stronger foundation for generalization that enhances situation understanding, bridges the gap between seen and unseen scenarios, and mitigates prediction bias in rare cases. We evaluate MIPD on the refined Ov-SWiG dataset, achieving superior performance on seen, rare, and unseen situations, and further demonstrate improved unseen detection on the HICO-DET dataset.
>
---
#### [replaced 071] Causal Tracing of Object Representations in Large Vision Language Models: Mechanistic Interpretability and Hallucination Mitigation
- **分类: cs.CV**

- **链接: []()**

> **作者:** Qiming Li; Zekai Ye; Xiaocheng Feng; Weihong Zhong; Weitao Ma; Xiachong Feng
>
> **备注:** AAAI2026 Oral
>
> **摘要:** Despite the remarkable advancements of Large Vision-Language Models (LVLMs), the mechanistic interpretability remains underexplored. Existing analyses are insufficiently comprehensive and lack examination covering visual and textual tokens, model components, and the full range of layers. This limitation restricts actionable insights to improve the faithfulness of model output and the development of downstream tasks, such as hallucination mitigation. To address this limitation, we introduce Fine-grained Cross-modal Causal Tracing (FCCT) framework, which systematically quantifies the causal effects on visual object perception. FCCT conducts fine-grained analysis covering the full range of visual and textual tokens, three core model components including multi-head self-attention (MHSA), feed-forward networks (FFNs), and hidden states, across all decoder layers. Our analysis is the first to demonstrate that MHSAs of the last token in middle layers play a critical role in aggregating cross-modal information, while FFNs exhibit a three-stage hierarchical progression for the storage and transfer of visual object representations. Building on these insights, we propose Intermediate Representation Injection (IRI), a training-free inference-time technique that reinforces visual object information flow by precisely intervening on cross-modal representations at specific components and layers, thereby enhancing perception and mitigating hallucination. Consistent improvements across five widely used benchmarks and LVLMs demonstrate IRI achieves state-of-the-art performance, while preserving inference speed and other foundational performance.
>
---
#### [replaced 072] Jamais Vu: Exposing the Generalization Gap in Supervised Semantic Correspondence
- **分类: cs.CV**

- **链接: []()**

> **作者:** Octave Mariotti; Zhipeng Du; Yash Bhalgat; Oisin Mac Aodha; Hakan Bilen
>
> **摘要:** Semantic correspondence (SC) aims to establish semantically meaningful matches across different instances of an object category. We illustrate how recent supervised SC methods remain limited in their ability to generalize beyond sparsely annotated training keypoints, effectively acting as keypoint detectors. To address this, we propose a novel approach for learning dense correspondences by lifting 2D keypoints into a canonical 3D space using monocular depth estimation. Our method constructs a continuous canonical manifold that captures object geometry without requiring explicit 3D supervision or camera annotations. Additionally, we introduce SPair-U, an extension of SPair-71k with novel keypoint annotations, to better assess generalization. Experiments not only demonstrate that our model significantly outperforms supervised baselines on unseen keypoints, highlighting its effectiveness in learning robust correspondences, but that unsupervised baselines outperform supervised counterparts when generalized across different datasets.
>
---
#### [replaced 073] Hestia: Voxel-Face-Aware Hierarchical Next-Best-View Acquisition for Efficient 3D Reconstruction
- **分类: cs.RO; cs.CV**

- **链接: []()**

> **作者:** Cheng-You Lu; Zhuoli Zhuang; Nguyen Thanh Trung Le; Da Xiao; Yu-Cheng Chang; Thomas Do; Srinath Sridhar; Chin-teng Lin
>
> **摘要:** Advances in 3D reconstruction and novel view synthesis have enabled efficient and photorealistic rendering. However, images for reconstruction are still either largely manual or constrained by simple preplanned trajectories. To address this issue, recent works propose generalizable next-best-view planners that do not require online learning. Nevertheless, robustness and performance remain limited across various shapes. Hence, this study introduces Voxel-Face-Aware Hierarchical Next-Best-View Acquisition for Efficient 3D Reconstruction (Hestia), which addresses the shortcomings of the reinforcement learning-based generalizable approaches for five-degree-of-freedom viewpoint prediction. Hestia systematically improves the planners through four components: a more diverse dataset to promote robustness, a hierarchical structure to manage the high-dimensional continuous action search space, a close-greedy strategy to mitigate spurious correlations, and a face-aware design to avoid overlooking geometry. Experimental results show that Hestia achieves non-marginal improvements, with at least a 4% gain in coverage ratio, while reducing Chamfer Distance by 50% and maintaining real-time inference. In addition, Hestia outperforms prior methods by at least 12% in coverage ratio with a 5-image budget and remains robust to object placement variations. Finally, we demonstrate that Hestia, as a next-best-view planner, is feasible for the real-world application. Our project page is https://johnnylu305.github.io/hestia web.
>
---
#### [replaced 074] Pruning at Initialization -- A Sketching Perspective
- **分类: cs.LG; cs.CV**

- **链接: []()**

> **作者:** Noga Bar; Raja Giryes
>
> **摘要:** The lottery ticket hypothesis (LTH) has increased attention to pruning neural networks at initialization. We study this problem in the linear setting. We show that finding a sparse mask at initialization is equivalent to the sketching problem introduced for efficient matrix multiplication. This gives us tools to analyze the LTH problem and gain insights into it. Specifically, using the mask found at initialization, we bound the approximation error of the pruned linear model at the end of training. We theoretically justify previous empirical evidence that the search for sparse networks may be data independent. By using the sketching perspective, we suggest a generic improvement to existing algorithms for pruning at initialization, which we show to be beneficial in the data-independent case.
>
---
#### [replaced 075] A Unified and Fast-Sampling Diffusion Bridge Framework via Stochastic Optimal Control
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: []()**

> **作者:** Mokai Pan; Kaizhen Zhu; Yuexin Ma; Yanwei Fu; Jingyi Yu; Jingya Wang; Ye Shi
>
> **摘要:** Recent advances in diffusion bridge models leverage Doob's $h$-transform to establish fixed endpoints between distributions, demonstrating promising results in image translation and restoration tasks. However, these approaches often produce blurred or excessively smoothed image details and lack a comprehensive theoretical foundation to explain these shortcomings. To address these limitations, we propose UniDB, a unified and fast-sampling framework for diffusion bridges based on Stochastic Optimal Control (SOC). We reformulate the problem through an SOC-based optimization, proving that existing diffusion bridges employing Doob's $h$-transform constitute a special case, emerging when the terminal penalty coefficient in the SOC cost function tends to infinity. By incorporating a tunable terminal penalty coefficient, UniDB achieves an optimal balance between control costs and terminal penalties, substantially improving detail preservation and output quality. To avoid computationally expensive costs of iterative Euler sampling methods in UniDB, we design a training-free accelerated algorithm by deriving exact closed-form solutions for UniDB's reverse-time SDE. It is further complemented by replacing conventional noise prediction with a more stable data prediction model, along with an SDE-Corrector mechanism that maintains perceptual quality for low-step regimes, effectively reducing error accumulation. Extensive experiments across diverse image restoration tasks validate the superiority and adaptability of the proposed framework, bridging the gap between theoretical generality and practical efficiency. Our code is available online https://github.com/2769433owo/UniDB-plusplus.
>
---
#### [replaced 076] UniCUE: Unified Recognition and Generation Framework for Chinese Cued Speech Video-to-Speech Generation
- **分类: cs.CV; cs.SD; eess.AS**

- **链接: []()**

> **作者:** Jinting Wang; Shan Yang; Chenxing Li; Dong Yu; Li Liu
>
> **备注:** 13 pages, 12 figures
>
> **摘要:** Cued Speech (CS) enhances lipreading via hand coding, offering visual phonemic cues that support precise speech perception for the hearing-impaired. The task of CS Video-to-Speech generation (CSV2S) aims to convert CS videos into intelligible speech signals. Most existing research focuses on CS Recognition (CSR), which transcribes video content into text. Consequently, a common solution for CSV2S is to integrate CSR with a text-to-speech (TTS) system. However, this pipeline relies on text as an intermediate medium, which may lead to error propagation and temporal misalignment between speech and CS video dynamics. In contrast, directly generating audio speech from CS video (direct CSV2S) often suffers from the inherent multimodal complexity and the limited availability of CS data. To address these challenges, we propose UniCUE, the first unified framework for CSV2S that directly generates speech from CS videos without relying on intermediate text. The core innovation of UniCUE lies in integrating an understanding task (CSR) that provides fine-grained CS visual-semantic cues to guide speech generation. Specifically, UniCUE incorporates a pose-aware visual processor, a semantic alignment pool that enables precise visual-semantic mapping, and a VisioPhonetic adapter to bridge the understanding and generation tasks within a unified architecture. To support this framework, we construct UniCUE-HI, a large-scale Mandarin CS dataset containing 11282 videos from 14 cuers, including both hearing-impaired and normal-hearing individuals. Extensive experiments on this dataset demonstrate that UniCUE achieves state-of-the-art performance across multiple evaluation metrics.
>
---
#### [replaced 077] One Homography is All You Need: IMM-based Joint Homography and Multiple Object State Estimation
- **分类: cs.CV**

- **链接: []()**

> **作者:** Paul Johannes Claasen; Johan Pieter de Villiers
>
> **备注:** Preprint submitted to Expert Systems with Applications
>
> **摘要:** A novel online MOT algorithm, IMM Joint Homography State Estimation (IMM-JHSE), is proposed. IMM-JHSE uses an initial homography estimate as the only additional 3D information, whereas other 3D MOT methods use regular 3D measurements. By jointly modelling the homography matrix and its dynamics as part of track state vectors, IMM-JHSE removes the explicit influence of camera motion compensation techniques on predicted track position states, which was prevalent in previous approaches. Expanding upon this, static and dynamic camera motion models are combined using an IMM filter. A simple bounding box motion model is used to predict bounding box positions to incorporate image plane information. In addition to applying an IMM to camera motion, a non-standard IMM approach is applied where bounding-box-based BIoU scores are mixed with ground-plane-based Mahalanobis distances in an IMM-like fashion to perform association only, making IMM-JHSE robust to motion away from the ground plane. Finally, IMM-JHSE makes use of dynamic process and measurement noise estimation techniques. IMM-JHSE improves upon related techniques, including UCMCTrack, OC-SORT, C-BIoU and ByteTrack on the DanceTrack and KITTI-car datasets, increasing HOTA by 2.64 and 2.11, respectively, while offering competitive performance on the MOT17, MOT20 and KITTI-pedestrian datasets. Using publicly available detections, IMM-JHSE outperforms almost all other 2D MOT methods and is outperformed only by 3D MOT methods -- some of which are offline -- on the KITTI-car dataset. Compared to tracking-by-attention methods, IMM-JHSE shows remarkably similar performance on the DanceTrack dataset and outperforms them on the MOT17 dataset. The code is publicly available: https://github.com/Paulkie99/imm-jhse.
>
---
#### [replaced 078] Continuous Subspace Optimization for Continual Learning
- **分类: cs.CV; cs.LG**

- **链接: []()**

> **作者:** Quan Cheng; Yuanyu Wan; Lingyu Wu; Chenping Hou; Lijun Zhang
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Continual learning aims to learn multiple tasks sequentially while preserving prior knowledge, but faces the challenge of catastrophic forgetting when adapting to new tasks. Recently, approaches leveraging pre-trained models have gained increasing popularity in mitigating this issue, due to the strong generalization ability of foundation models. To adjust pre-trained models for new tasks, existing methods usually employ low-rank adaptation, which restricts parameter updates to a fixed low-rank subspace. However, constraining the optimization space inherently compromises the model's learning capacity, resulting in inferior performance. To address this limitation, we propose Continuous Subspace Optimization for Continual Learning (CoSO) to fine-tune the model in a series of subspaces rather than a single one. These sequential subspaces are dynamically determined through the singular value decomposition of the gradients. CoSO updates the model by projecting gradients onto these subspaces, ensuring memory-efficient optimization. To mitigate forgetting, the optimization subspace of each task is constrained to be orthogonal to the historical task subspace. During task learning, CoSO maintains a task-specific component that captures the critical update directions for the current task. Upon completing a task, this component is used to update the historical task subspace, laying the groundwork for subsequent learning. Extensive experiments on multiple datasets demonstrate that CoSO significantly outperforms state-of-the-art methods, especially in challenging scenarios with long task sequences.
>
---
#### [replaced 079] Towards Methane Detection Onboard Satellites
- **分类: cs.CV; cs.AI**

- **链接: []()**

> **作者:** Maggie Chen; Hala Lambdouar; Luca Marini; Laura Martínez-Ferrer; Chris Bridges; Giacomo Acciarini
>
> **摘要:** Methane is a potent greenhouse gas and a major driver of climate change, making its timely detection critical for effective mitigation. Machine learning (ML) deployed onboard satellites can enable rapid detection while reducing downlink costs, supporting faster response systems. Conventional methane detection methods often rely on image processing techniques, such as orthorectification to correct geometric distortions and matched filters to enhance plume signals. We introduce a novel approach that bypasses these preprocessing steps by using \textit{unorthorectified} data (UnorthoDOS). We find that ML models trained on this dataset achieve performance comparable to those trained on orthorectified data. Moreover, we also train models on an orthorectified dataset, showing that they can outperform the matched filter baseline (mag1c). We release model checkpoints and two ML-ready datasets comprising orthorectified and unorthorectified hyperspectral images from the Earth Surface Mineral Dust Source Investigation (EMIT) sensor at https://huggingface.co/datasets/SpaceML/UnorthoDOS , along with code at https://github.com/spaceml-org/plume-hunter.
>
---
#### [replaced 080] Adaptive Morph-Patch Transformer for Aortic Vessel Segmentation
- **分类: cs.CV**

- **链接: []()**

> **作者:** Zhenxi Zhang; Fuchen Zheng; Adnan Iltaf; Yifei Han; Zhenyu Cheng; Yue Du; Bin Li; Tianyong Liu; Shoujun Zhou
>
> **备注:** This is the preprint version of a paper accepted by AAAI 2026. The final version will appear in the AAAI Proceedings
>
> **摘要:** Accurate segmentation of aortic vascular structures is critical for diagnosing and treating cardiovascular diseases.Traditional Transformer-based models have shown promise in this domain by capturing long-range dependencies between vascular features. However, their reliance on fixed-size rectangular patches often influences the integrity of complex vascular structures, leading to suboptimal segmentation accuracy. To address this challenge, we propose the adaptive Morph Patch Transformer (MPT), a novel architecture specifically designed for aortic vascular segmentation. Specifically, MPT introduces an adaptive patch partitioning strategy that dynamically generates morphology-aware patches aligned with complex vascular structures. This strategy can preserve semantic integrity of complex vascular structures within individual patches. Moreover, a Semantic Clustering Attention (SCA) method is proposed to dynamically aggregate features from various patches with similar semantic characteristics. This method enhances the model's capability to segment vessels of varying sizes, preserving the integrity of vascular structures. Extensive experiments on three open-source dataset(AVT, AortaSeg24 and TBAD) demonstrate that MPT achieves state-of-the-art performance, with improvements in segmenting intricate vascular structures.
>
---
