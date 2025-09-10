# 计算机视觉 cs.CV

- **最新发布 89 篇**

- **更新 85 篇**

## 最新发布

#### [new 001] Self-Supervised Cross-Encoder for Neurodegenerative Disease Diagnosis
- **分类: cs.CV**

- **简介: 该论文提出一种自监督跨编码器框架，用于神经退行性疾病诊断。旨在解决依赖标注数据和模型可解释性差的问题，通过分离静态与动态表征提升分类精度与可解释性，并在多个数据集上验证其泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.07623v1](http://arxiv.org/pdf/2509.07623v1)**

> **作者:** Fangqi Cheng; Yingying Zhao; Xiaochen Yang
>
> **摘要:** Deep learning has shown significant potential in diagnosing neurodegenerative diseases from MRI data. However, most existing methods rely heavily on large volumes of labeled data and often yield representations that lack interpretability. To address both challenges, we propose a novel self-supervised cross-encoder framework that leverages the temporal continuity in longitudinal MRI scans for supervision. This framework disentangles learned representations into two components: a static representation, constrained by contrastive learning, which captures stable anatomical features; and a dynamic representation, guided by input-gradient regularization, which reflects temporal changes and can be effectively fine-tuned for downstream classification tasks. Experimental results on the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset demonstrate that our method achieves superior classification accuracy and improved interpretability. Furthermore, the learned representations exhibit strong zero-shot generalization on the Open Access Series of Imaging Studies (OASIS) dataset and cross-task generalization on the Parkinson Progression Marker Initiative (PPMI) dataset. The code for the proposed method will be made publicly available.
>
---
#### [new 002] CellPainTR: Generalizable Representation Learning for Cross-Dataset Cell Painting Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CellPainTR，一种基于Transformer的模型，用于跨数据集细胞图像分析。旨在解决批次效应和模型泛化问题，无需微调即可在新数据集上保持高性能，推动跨研究生物学分析。**

- **链接: [http://arxiv.org/pdf/2509.06986v1](http://arxiv.org/pdf/2509.06986v1)**

> **作者:** Cedric Caruzzo; Jong Chul Ye
>
> **备注:** 14 pages, 4 figures. Code available at: https://github.com/CellPainTR/CellPainTR
>
> **摘要:** Large-scale biological discovery requires integrating massive, heterogeneous datasets like those from the JUMP Cell Painting consortium, but technical batch effects and a lack of generalizable models remain critical roadblocks. To address this, we introduce CellPainTR, a Transformer-based architecture designed to learn foundational representations of cellular morphology that are robust to batch effects. Unlike traditional methods that require retraining on new data, CellPainTR's design, featuring source-specific context tokens, allows for effective out-of-distribution (OOD) generalization to entirely unseen datasets without fine-tuning. We validate CellPainTR on the large-scale JUMP dataset, where it outperforms established methods like ComBat and Harmony in both batch integration and biological signal preservation. Critically, we demonstrate its robustness through a challenging OOD task on the unseen Bray et al. dataset, where it maintains high performance despite significant domain and feature shifts. Our work represents a significant step towards creating truly foundational models for image-based profiling, enabling more reliable and scalable cross-study biological analysis.
>
---
#### [new 003] Frustratingly Easy Feature Reconstruction for Out-of-Distribution Detection
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出一种无需训练数据的OOD检测方法ClaFR，通过分类器权重的正交分解提取类已知子空间，并利用特征重构误差判断数据是否分布外，有效解决隐私保护场景下的OOD检测问题。**

- **链接: [http://arxiv.org/pdf/2509.06988v1](http://arxiv.org/pdf/2509.06988v1)**

> **作者:** Yingsheng Wang; Shuo Lu; Jian Liang; Aihua Zheng; Ran He
>
> **备注:** Accepted to PRCV2025
>
> **摘要:** Out-of-distribution (OOD) detection helps models identify data outside the training categories, crucial for security applications. While feature-based post-hoc methods address this by evaluating data differences in the feature space without changing network parameters, they often require access to training data, which may not be suitable for some data privacy scenarios. This may not be suitable in scenarios where data privacy protection is a concern. In this paper, we propose a simple yet effective post-hoc method, termed Classifier-based Feature Reconstruction (ClaFR), from the perspective of subspace projection. It first performs an orthogonal decomposition of the classifier's weights to extract the class-known subspace, then maps the original data features into this subspace to obtain new data representations. Subsequently, the OOD score is determined by calculating the feature reconstruction error of the data within the subspace. Compared to existing OOD detection algorithms, our method does not require access to training data while achieving leading performance on multiple OOD benchmarks. Our code is released at https://github.com/Aie0923/ClaFR.
>
---
#### [new 004] EDFFDNet: Towards Accurate and Efficient Unsupervised Multi-Grid Image Registration
- **分类: cs.CV**

- **简介: 该论文提出EDFFDNet，用于无监督多网格图像配准任务，解决深度差异场景下的配准精度与效率问题。通过引入指数衰减自由形变网络和自适应稀疏运动聚合器，提升准确率并减少参数、内存与运行时间。**

- **链接: [http://arxiv.org/pdf/2509.07662v1](http://arxiv.org/pdf/2509.07662v1)**

> **作者:** Haokai Zhu; Bo Qu; Si-Yuan Cao; Runmin Zhang; Shujie Chen; Bailin Yang; Hui-Liang Shen
>
> **摘要:** Previous deep image registration methods that employ single homography, multi-grid homography, or thin-plate spline often struggle with real scenes containing depth disparities due to their inherent limitations. To address this, we propose an Exponential-Decay Free-Form Deformation Network (EDFFDNet), which employs free-form deformation with an exponential-decay basis function. This design achieves higher efficiency and performs well in scenes with depth disparities, benefiting from its inherent locality. We also introduce an Adaptive Sparse Motion Aggregator (ASMA), which replaces the MLP motion aggregator used in previous methods. By transforming dense interactions into sparse ones, ASMA reduces parameters and improves accuracy. Additionally, we propose a progressive correlation refinement strategy that leverages global-local correlation patterns for coarse-to-fine motion estimation, further enhancing efficiency and accuracy. Experiments demonstrate that EDFFDNet reduces parameters, memory, and total runtime by 70.5%, 32.6%, and 33.7%, respectively, while achieving a 0.5 dB PSNR gain over the state-of-the-art method. With an additional local refinement stage,EDFFDNet-2 further improves PSNR by 1.06 dB while maintaining lower computational costs. Our method also demonstrates strong generalization ability across datasets, outperforming previous deep learning methods.
>
---
#### [new 005] DIET-CP: Lightweight and Data Efficient Self Supervised Continued Pretraining
- **分类: cs.CV; cs.LG; I.2; I.4**

- **简介: 论文提出DIET-CP方法，解决小数据领域下基础模型持续预训练问题。该方法无需标签和额外超参数，在多种模态和主干模型上稳定有效，仅用少量数据显著提升模型性能。属于自监督持续预训练任务。**

- **链接: [http://arxiv.org/pdf/2509.06990v1](http://arxiv.org/pdf/2509.06990v1)**

> **作者:** Bryan Rodas; Natalie Montesino; Jakob Ambsdorf; David Klindt; Randall Balestriero
>
> **摘要:** Continued pretraining offers a promising solution for adapting foundation models to a new target domain. However, in specialized domains, available datasets are often very small, limiting the applicability of SSL methods developed for large-scale pretraining and making hyperparameter search infeasible. In addition, pretrained models are usually released as backbone-weights only, lacking important information to continue pretraining. We propose to bridge this gap with DIET-CP, a simple continued pretraining strategy, where any strong foundation model can be steered towards the new data distribution of interest. DIET-CP relies on a very simple objective, requires no labels, and introduces no more hyperparameters than supervised finetuning. It is stable across data modalities and backbone choices, while providing a significant performance boost for state-of-the-art models such as DINOv3 using only 1000 images.
>
---
#### [new 006] Faster, Self-Supervised Super-Resolution for Anisotropic Multi-View MRI Using a Sparse Coordinate Loss
- **分类: cs.CV**

- **简介: 该论文提出一种自监督超分辨率方法，解决多视角MRI图像重建问题。通过融合两个不同方向的低分辨率图像，利用稀疏坐标损失实现高效、高质量的超分辨率重建，提升医学影像分析效率。**

- **链接: [http://arxiv.org/pdf/2509.07798v1](http://arxiv.org/pdf/2509.07798v1)**

> **作者:** Maja Schlereth; Moritz Schillinger; Katharina Breininger
>
> **备注:** 11 pages, 2 figures
>
> **摘要:** Acquiring images in high resolution is often a challenging task. Especially in the medical sector, image quality has to be balanced with acquisition time and patient comfort. To strike a compromise between scan time and quality for Magnetic Resonance (MR) imaging, two anisotropic scans with different low-resolution (LR) orientations can be acquired. Typically, LR scans are analyzed individually by radiologists, which is time consuming and can lead to inaccurate interpretation. To tackle this, we propose a novel approach for fusing two orthogonal anisotropic LR MR images to reconstruct anatomical details in a unified representation. Our multi-view neural network is trained in a self-supervised manner, without requiring corresponding high-resolution (HR) data. To optimize the model, we introduce a sparse coordinate-based loss, enabling the integration of LR images with arbitrary scaling. We evaluate our method on MR images from two independent cohorts. Our results demonstrate comparable or even improved super-resolution (SR) performance compared to state-of-the-art (SOTA) self-supervised SR methods for different upsampling scales. By combining a patient-agnostic offline and a patient-specific online phase, we achieve a substantial speed-up of up to ten times for patient-specific reconstruction while achieving similar or better SR quality. Code is available at https://github.com/MajaSchle/tripleSR.
>
---
#### [new 007] HU-based Foreground Masking for 3D Medical Masked Image Modeling
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出基于Hounsfield Unit（HU）的前景掩码策略，用于改进3D医学图像分割中的Masked Image Modeling任务。通过关注组织区域、排除非诊断区域，提升分割质量与Dice分数，实验表明效果优于随机掩码方法。**

- **链接: [http://arxiv.org/pdf/2509.07534v1](http://arxiv.org/pdf/2509.07534v1)**

> **作者:** Jin Lee; Vu Dang; Gwang-Hyun Yu; Anh Le; Zahid Rahman; Jin-Ho Jang; Heonzoo Lee; Kun-Yung Kim; Jin-Sul Kim; Jin-Young Kim
>
> **备注:** Accepted by MICCAI AMAI Workshop 2025
>
> **摘要:** While Masked Image Modeling (MIM) has revolutionized fields of computer vision, its adoption in 3D medical image computing has been limited by the use of random masking, which overlooks the density of anatomical objects. To address this limitation, we enhance the pretext task with a simple yet effective masking strategy. Leveraging Hounsfield Unit (HU) measurements, we implement an HU-based Foreground Masking, which focuses on the intensity distribution of visceral organs and excludes non-tissue regions, such as air and fluid, that lack diagnostically meaningful features. Extensive experiments on five public 3D medical imaging datasets demonstrate that our masking consistently improves performance, both in quality of segmentation and Dice score (BTCV:~84.64\%, Flare22:~92.43\%, MM-WHS:~90.67\%, Amos22:~88.64\%, BraTS:~78.55\%). These results underscore the importance of domain-centric MIM and suggest a promising direction for representation learning in medical image segmentation. Implementation is available at github.com/AISeedHub/SubFore/.
>
---
#### [new 008] ScoreHOI: Physically Plausible Reconstruction of Human-Object Interaction via Score-Guided Diffusion
- **分类: cs.CV**

- **简介: 该论文提出ScoreHOI模型，用于人体-物体交互的物理合理重建。针对传统方法缺乏先验知识的问题，引入扩散模型与评分引导采样，结合接触驱动优化，提升重建精度与合理性。属于三维姿态估计任务。**

- **链接: [http://arxiv.org/pdf/2509.07920v1](http://arxiv.org/pdf/2509.07920v1)**

> **作者:** Ao Li; Jinpeng Liu; Yixuan Zhu; Yansong Tang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Joint reconstruction of human-object interaction marks a significant milestone in comprehending the intricate interrelations between humans and their surrounding environment. Nevertheless, previous optimization methods often struggle to achieve physically plausible reconstruction results due to the lack of prior knowledge about human-object interactions. In this paper, we introduce ScoreHOI, an effective diffusion-based optimizer that introduces diffusion priors for the precise recovery of human-object interactions. By harnessing the controllability within score-guided sampling, the diffusion model can reconstruct a conditional distribution of human and object pose given the image observation and object feature. During inference, the ScoreHOI effectively improves the reconstruction results by guiding the denoising process with specific physical constraints. Furthermore, we propose a contact-driven iterative refinement approach to enhance the contact plausibility and improve the reconstruction accuracy. Extensive evaluations on standard benchmarks demonstrate ScoreHOI's superior performance over state-of-the-art methods, highlighting its ability to achieve a precise and robust improvement in joint human-object interaction reconstruction.
>
---
#### [new 009] One View, Many Worlds: Single-Image to 3D Object Meets Generative Domain Randomization for One-Shot 6D Pose Estimation
- **分类: cs.CV**

- **简介: 该论文属于单目6D位姿估计任务，解决从单张图像中估计未知物体位姿的问题。提出OnePoseViaGen方法，通过多视角特征匹配和生成域随机化策略，实现高精度的一次性位姿估计，并在多个基准上取得最优性能。**

- **链接: [http://arxiv.org/pdf/2509.07978v1](http://arxiv.org/pdf/2509.07978v1)**

> **作者:** Zheng Geng; Nan Wang; Shaocong Xu; Chongjie Ye; Bohan Li; Zhaoxi Chen; Sida Peng; Hao Zhao
>
> **备注:** CoRL 2025 Oral, Project page: https://gzwsama.github.io/OnePoseviaGen.github.io/
>
> **摘要:** Estimating the 6D pose of arbitrary unseen objects from a single reference image is critical for robotics operating in the long-tail of real-world instances. However, this setting is notoriously challenging: 3D models are rarely available, single-view reconstructions lack metric scale, and domain gaps between generated models and real-world images undermine robustness. We propose OnePoseViaGen, a pipeline that tackles these challenges through two key components. First, a coarse-to-fine alignment module jointly refines scale and pose by combining multi-view feature matching with render-and-compare refinement. Second, a text-guided generative domain randomization strategy diversifies textures, enabling effective fine-tuning of pose estimators with synthetic data. Together, these steps allow high-fidelity single-view 3D generation to support reliable one-shot 6D pose estimation. On challenging benchmarks (YCBInEOAT, Toyota-Light, LM-O), OnePoseViaGen achieves state-of-the-art performance far surpassing prior approaches. We further demonstrate robust dexterous grasping with a real robot hand, validating the practicality of our method in real-world manipulation. Project page: https://gzwsama.github.io/OnePoseviaGen.github.io/
>
---
#### [new 010] FusWay: Multimodal hybrid fusion approach. Application to Railway Defect Detection
- **分类: cs.CV; cs.AI**

- **简介: 论文提出FusWay多模态融合方法，用于铁路缺陷检测。结合YOLOv8n与ViT，融合图像与音频信号，提升检测精度与准确率，解决单模态方法误检问题。**

- **链接: [http://arxiv.org/pdf/2509.06987v1](http://arxiv.org/pdf/2509.06987v1)**

> **作者:** Alexey Zhukov; Jenny Benois-Pineau; Amira Youssef; Akka Zemmari; Mohamed Mosbah; Virginie Taillandier
>
> **摘要:** Multimodal fusion is a multimedia technique that has become popular in the wide range of tasks where image information is accompanied by a signal/audio. The latter may not convey highly semantic information, such as speech or music, but some measures such as audio signal recorded by mics in the goal to detect rail structure elements or defects. While classical detection approaches such as You Only Look Once (YOLO) family detectors can be efficiently deployed for defect detection on the image modality, the single modality approaches remain limited. They yield an overdetection in case of the appearance similar to normal structural elements. The paper proposes a new multimodal fusion architecture built on the basis of domain rules with YOLO and Vision transformer backbones. It integrates YOLOv8n for rapid object detection with a Vision Transformer (ViT) to combine feature maps extracted from multiple layers (7, 16, and 19) and synthesised audio representations for two defect classes: rail Rupture and Surface defect. Fusion is performed between audio and image. Experimental evaluation on a real-world railway dataset demonstrates that our multimodal fusion improves precision and overall accuracy by 0.2 points compared to the vision-only approach. Student's unpaired t-test also confirms statistical significance of differences in the mean accuracy.
>
---
#### [new 011] Human-in-the-Loop: Quantitative Evaluation of 3D Models Generation by Large Language Models
- **分类: cs.CV; cs.AI; cs.ET**

- **简介: 论文提出一种人机协同框架，用于量化评估大语言模型生成的3D模型质量。通过设计多维度指标，如体积精度、表面对齐等，解决当前缺乏有效评估方法的问题，并以L型支架为例验证方法有效性，提升生成模型的准确性与收敛速度。**

- **链接: [http://arxiv.org/pdf/2509.07010v1](http://arxiv.org/pdf/2509.07010v1)**

> **作者:** Ahmed R. Sadik; Mariusz Bujny
>
> **摘要:** Large Language Models are increasingly capable of interpreting multimodal inputs to generate complex 3D shapes, yet robust methods to evaluate geometric and structural fidelity remain underdeveloped. This paper introduces a human in the loop framework for the quantitative evaluation of LLM generated 3D models, supporting applications such as democratization of CAD design, reverse engineering of legacy designs, and rapid prototyping. We propose a comprehensive suite of similarity and complexity metrics, including volumetric accuracy, surface alignment, dimensional fidelity, and topological intricacy, to benchmark generated models against ground truth CAD references. Using an L bracket component as a case study, we systematically compare LLM performance across four input modalities: 2D orthographic views, isometric sketches, geometric structure trees, and code based correction prompts. Our findings demonstrate improved generation fidelity with increased semantic richness, with code level prompts achieving perfect reconstruction across all metrics. A key contribution of this work is demonstrating that our proposed quantitative evaluation approach enables significantly faster convergence toward the ground truth, especially compared to traditional qualitative methods based solely on visual inspection and human intuition. This work not only advances the understanding of AI assisted shape synthesis but also provides a scalable methodology to validate and refine generative models for diverse CAD applications.
>
---
#### [new 012] Moment- and Power-Spectrum-Based Gaussianity Regularization for Text-to-Image Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出一种基于矩和功率谱的高斯正则化方法，用于文本到图像模型。旨在通过强制样本符合标准高斯分布，提升生成质量与奖励对齐，解决生成图像美观性与文本一致性问题，优化潜在空间中的下游任务。**

- **链接: [http://arxiv.org/pdf/2509.07027v1](http://arxiv.org/pdf/2509.07027v1)**

> **作者:** Jisung Hwang; Jaihoon Kim; Minhyuk Sung
>
> **备注:** Submitted to NeurIPS 2025
>
> **摘要:** We propose a novel regularization loss that enforces standard Gaussianity, encouraging samples to align with a standard Gaussian distribution. This facilitates a range of downstream tasks involving optimization in the latent space of text-to-image models. We treat elements of a high-dimensional sample as one-dimensional standard Gaussian variables and define a composite loss that combines moment-based regularization in the spatial domain with power spectrum-based regularization in the spectral domain. Since the expected values of moments and power spectrum distributions are analytically known, the loss promotes conformity to these properties. To ensure permutation invariance, the losses are applied to randomly permuted inputs. Notably, existing Gaussianity-based regularizations fall within our unified framework: some correspond to moment losses of specific orders, while the previous covariance-matching loss is equivalent to our spectral loss but incurs higher time complexity due to its spatial-domain computation. We showcase the application of our regularization in generative modeling for test-time reward alignment with a text-to-image model, specifically to enhance aesthetics and text alignment. Our regularization outperforms previous Gaussianity regularization, effectively prevents reward hacking and accelerates convergence.
>
---
#### [new 013] G3CN: Gaussian Topology Refinement Gated Graph Convolutional Network for Skeleton-Based Action Recognition
- **分类: cs.CV**

- **简介: 论文提出G$^{3}$CN网络，用于骨骼动作识别任务，旨在解决GCNs在区分模糊动作时的不足。通过高斯滤波优化图结构，并引入GRU增强信息传播，有效提升模型对模糊样本的识别能力。**

- **链接: [http://arxiv.org/pdf/2509.07335v1](http://arxiv.org/pdf/2509.07335v1)**

> **作者:** Haiqing Ren; Zhongkai Luo; Heng Fan; Xiaohui Yuan; Guanchen Wang; Libo Zhang
>
> **备注:** 8 pages, 5 figures, IROS
>
> **摘要:** Graph Convolutional Networks (GCNs) have proven to be highly effective for skeleton-based action recognition, primarily due to their ability to leverage graph topology for feature aggregation, a key factor in extracting meaningful representations. However, despite their success, GCNs often struggle to effectively distinguish between ambiguous actions, revealing limitations in the representation of learned topological and spatial features. To address this challenge, we propose a novel approach, Gaussian Topology Refinement Gated Graph Convolution (G$^{3}$CN), to address the challenge of distinguishing ambiguous actions in skeleton-based action recognition. G$^{3}$CN incorporates a Gaussian filter to refine the skeleton topology graph, improving the representation of ambiguous actions. Additionally, Gated Recurrent Units (GRUs) are integrated into the GCN framework to enhance information propagation between skeleton points. Our method shows strong generalization across various GCN backbones. Extensive experiments on NTU RGB+D, NTU RGB+D 120, and NW-UCLA benchmarks demonstrate that G$^{3}$CN effectively improves action recognition, particularly for ambiguous samples.
>
---
#### [new 014] Parse Graph-Based Visual-Language Interaction for Human Pose Estimation
- **分类: cs.CV**

- **简介: 论文提出PGVL方法，融合视觉与语言信息提升人体姿态估计。通过解析图结构和引导模块，解决遮挡区域特征弱、对齐失败问题，实现多模态有效融合。属于视觉-语言交互任务。**

- **链接: [http://arxiv.org/pdf/2509.07385v1](http://arxiv.org/pdf/2509.07385v1)**

> **作者:** Shibang Liu; Xuemei Xie; Guangming Shi
>
> **摘要:** Parse graphs boost human pose estimation (HPE) by integrating context and hierarchies, yet prior work mostly focuses on single modality modeling, ignoring the potential of multimodal fusion. Notably, language offers rich HPE priors like spatial relations for occluded scenes, but existing visual-language fusion via global feature integration weakens occluded region responses and causes alignment and location failures. To address this issue, we propose Parse Graph-based Visual-Language interaction (PGVL) with a core novel Guided Module (GM). In PGVL, low-level nodes focus on local features, maximizing the maintenance of responses in occluded areas and high-level nodes integrate global features to infer occluded or invisible parts. GM enables high semantic nodes to guide the feature update of low semantic nodes that have undergone cross attention. It ensuring effective fusion of diverse information. PGVL includes top-down decomposition and bottom-up composition. In the first stage, modality specific parse graphs are constructed. Next stage. recursive bidirectional cross-attention is used, purified by GM. We also design network based on PGVL. The PGVL and our network is validated on major pose estimation datasets. We will release the code soon.
>
---
#### [new 015] Multimodal Contrastive Pretraining of CBCT and IOS for Enhanced Tooth Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出ToothMCL框架，通过多模态对比学习实现CBCT与IOS的牙齿分割。任务为提升牙齿分割精度，解决现有方法验证不足、性能有限的问题。构建了最大配对数据集CBCT-IOS3.8K，并在多个数据集上取得SOTA结果。**

- **链接: [http://arxiv.org/pdf/2509.07923v1](http://arxiv.org/pdf/2509.07923v1)**

> **作者:** Moo Hyun Son; Juyoung Bae; Zelin Qiu; Jiale Peng; Kai Xin Li; Yifan Lin; Hao Chen
>
> **摘要:** Digital dentistry represents a transformative shift in modern dental practice. The foundational step in this transformation is the accurate digital representation of the patient's dentition, which is obtained from segmented Cone-Beam Computed Tomography (CBCT) and Intraoral Scans (IOS). Despite the growing interest in digital dental technologies, existing segmentation methodologies frequently lack rigorous validation and demonstrate limited performance and clinical applicability. To the best of our knowledge, this is the first work to introduce a multimodal pretraining framework for tooth segmentation. We present ToothMCL, a Tooth Multimodal Contrastive Learning for pretraining that integrates volumetric (CBCT) and surface-based (IOS) modalities. By capturing modality-invariant representations through multimodal contrastive learning, our approach effectively models fine-grained anatomical features, enabling precise multi-class segmentation and accurate identification of F\'ed\'eration Dentaire Internationale (FDI) tooth numbering. Along with the framework, we curated CBCT-IOS3.8K, the largest paired CBCT and IOS dataset to date, comprising 3,867 patients. We then evaluated ToothMCL on a comprehensive collection of independent datasets, representing the largest and most diverse evaluation to date. Our method achieves state-of-the-art performance in both internal and external testing, with an increase of 12\% for CBCT segmentation and 8\% for IOS segmentation in the Dice Similarity Coefficient (DSC). Furthermore, ToothMCL consistently surpasses existing approaches in tooth groups and demonstrates robust generalizability across varying imaging conditions and clinical scenarios.
>
---
#### [new 016] Object-level Correlation for Few-Shot Segmentation
- **分类: cs.CV**

- **简介: 该论文属于少样本语义分割任务，旨在利用少量标注支持样本分割新类别对象。为解决背景噪声干扰问题，提出OCNet模型，通过建立支持目标与查询通用对象间的对象级关联，有效抑制噪声，提升分割性能。**

- **链接: [http://arxiv.org/pdf/2509.07917v1](http://arxiv.org/pdf/2509.07917v1)**

> **作者:** Chunlin Wen; Yu Zhang; Jie Fan; Hongyuan Zhu; Xiu-Shen Wei; Yijun Wang; Zhiqiang Kou; Shuzhou Sun
>
> **备注:** This paper was accepted by ICCV 2025
>
> **摘要:** Few-shot semantic segmentation (FSS) aims to segment objects of novel categories in the query images given only a few annotated support samples. Existing methods primarily build the image-level correlation between the support target object and the entire query image. However, this correlation contains the hard pixel noise, \textit{i.e.}, irrelevant background objects, that is intractable to trace and suppress, leading to the overfitting of the background. To address the limitation of this correlation, we imitate the biological vision process to identify novel objects in the object-level information. Target identification in the general objects is more valid than in the entire image, especially in the low-data regime. Inspired by this, we design an Object-level Correlation Network (OCNet) by establishing the object-level correlation between the support target object and query general objects, which is mainly composed of the General Object Mining Module (GOMM) and Correlation Construction Module (CCM). Specifically, GOMM constructs the query general object feature by learning saliency and high-level similarity cues, where the general objects include the irrelevant background objects and the target foreground object. Then, CCM establishes the object-level correlation by allocating the target prototypes to match the general object feature. The generated object-level correlation can mine the query target feature and suppress the hard pixel noise for the final prediction. Extensive experiments on PASCAL-${5}^{i}$ and COCO-${20}^{i}$ show that our model achieves the state-of-the-art performance.
>
---
#### [new 017] SplatFill: 3D Scene Inpainting via Depth-Guided Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文提出SplatFill方法，解决3D高斯泼溅（3DGS）场景中缺失区域的修复问题。通过深度引导和一致性修正，提升修复质量与效率，在视觉保真度和训练速度上优于现有方法。属于3D场景修复任务。**

- **链接: [http://arxiv.org/pdf/2509.07809v1](http://arxiv.org/pdf/2509.07809v1)**

> **作者:** Mahtab Dahaghin; Milind G. Padalkar; Matteo Toso; Alessio Del Bue
>
> **摘要:** 3D Gaussian Splatting (3DGS) has enabled the creation of highly realistic 3D scene representations from sets of multi-view images. However, inpainting missing regions, whether due to occlusion or scene editing, remains a challenging task, often leading to blurry details, artifacts, and inconsistent geometry. In this work, we introduce SplatFill, a novel depth-guided approach for 3DGS scene inpainting that achieves state-of-the-art perceptual quality and improved efficiency. Our method combines two key ideas: (1) joint depth-based and object-based supervision to ensure inpainted Gaussians are accurately placed in 3D space and aligned with surrounding geometry, and (2) we propose a consistency-aware refinement scheme that selectively identifies and corrects inconsistent regions without disrupting the rest of the scene. Evaluations on the SPIn-NeRF dataset demonstrate that SplatFill not only surpasses existing NeRF-based and 3DGS-based inpainting methods in visual fidelity but also reduces training time by 24.5%. Qualitative results show our method delivers sharper details, fewer artifacts, and greater coherence across challenging viewpoints.
>
---
#### [new 018] Visual Representation Alignment for Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 论文提出VIRAL方法，通过对齐多模态大语言模型与视觉基础模型的内部视觉表征，解决其在视觉任务中的细节丢失问题。属于视觉-语言对齐任务，旨在提升模型的视觉推理能力。**

- **链接: [http://arxiv.org/pdf/2509.07979v1](http://arxiv.org/pdf/2509.07979v1)**

> **作者:** Heeji Yoon; Jaewoo Jung; Junwan Kim; Hyungyu Choi; Heeseong Shin; Sangbeom Lim; Honggyu An; Chaehyun Kim; Jisang Han; Donghyun Kim; Chanho Eom; Sunghwan Hong; Seungryong Kim
>
> **备注:** Project Page: https://cvlab-kaist.github.io/VIRAL/
>
> **摘要:** Multimodal large language models (MLLMs) trained with visual instruction tuning have achieved strong performance across diverse tasks, yet they remain limited in vision-centric tasks such as object counting or spatial reasoning. We attribute this gap to the prevailing text-only supervision paradigm, which provides only indirect guidance for the visual pathway and often leads MLLMs to discard fine-grained visual details during training. In this paper, we present VIsual Representation ALignment (VIRAL), a simple yet effective regularization strategy that aligns the internal visual representations of MLLMs with those of pre-trained vision foundation models (VFMs). By explicitly enforcing this alignment, VIRAL enables the model not only to retain critical visual details from the input vision encoder but also to complement additional visual knowledge from VFMs, thereby enhancing its ability to reason over complex visual inputs. Our experiments demonstrate consistent improvements across all tasks on widely adopted multimodal benchmarks. Furthermore, we conduct comprehensive ablation studies to validate the key design choices underlying our framework. We believe this simple finding opens up an important direction for the effective integration of visual information in training MLLMs.
>
---
#### [new 019] DEPF: A UAV Multispectral Object Detector with Dual-Domain Enhancement and Priority-Guided Mamba Fusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出DEPF，用于UAV多光谱目标检测。针对低光照、冗余信息干扰和计算复杂度高的问题，设计了双域增强模块和优先级引导的Mamba融合模块，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2509.07327v1](http://arxiv.org/pdf/2509.07327v1)**

> **作者:** Shucong Li; Zhenyu Liu; Zijie Hong; Zhiheng Zhou; Xianghai Cao
>
> **摘要:** Multispectral remote sensing object detection is one of the important application of unmanned aerial vehicle (UAV). However, it faces three challenges. Firstly, the low-light remote sensing images reduce the complementarity during multi-modality fusion. Secondly, the local small target modeling is interfered with redundant information in the fusion stage easily. Thirdly, due to the quadratic computational complexity, it is hard to apply the transformer-based methods on the UAV platform. To address these limitations, motivated by Mamba with linear complexity, a UAV multispectral object detector with dual-domain enhancement and priority-guided mamba fusion (DEPF) is proposed. Firstly, to enhance low-light remote sensing images, Dual-Domain Enhancement Module (DDE) is designed, which contains Cross-Scale Wavelet Mamba (CSWM) and Fourier Details Recovery block (FDR). CSWM applies cross-scale mamba scanning for the low-frequency components to enhance the global brightness of images, while FDR constructs spectrum recovery network to enhance the frequency spectra features for recovering the texture-details. Secondly, to enhance local target modeling and reduce the impact of redundant information during fusion, Priority-Guided Mamba Fusion Module (PGMF) is designed. PGMF introduces the concept of priority scanning, which starts from local targets features according to the priority scores obtained from modality difference. Experiments on DroneVehicle dataset and VEDAI dataset reports that, DEPF performs well on object detection, comparing with state-of-the-art methods. Our code is available in the supplementary material.
>
---
#### [new 020] SAM$^{*}$: Task-Adaptive SAM with Physics-Guided Rewards
- **分类: cs.CV; cond-mat.mtrl-sci; cs.LG**

- **简介: 论文提出SAM$^{*}$，通过物理引导奖励函数优化SAM模型，提升其在显微图像分割中的适应性与实时性能，解决基础模型参数不透明、需手动调优的问题。属于图像分割任务。**

- **链接: [http://arxiv.org/pdf/2509.07047v1](http://arxiv.org/pdf/2509.07047v1)**

> **作者:** Kamyar Barakati; Utkarsh Pratiush; Sheryl L. Sanchez; Aditya Raghavan; Delia J. Milliron; Mahshid Ahmadi; Philip D. Rack; Sergei V. Kalinin
>
> **备注:** 19 pages, 8 figures
>
> **摘要:** Image segmentation is a critical task in microscopy, essential for accurately analyzing and interpreting complex visual data. This task can be performed using custom models trained on domain-specific datasets, transfer learning from pre-trained models, or foundational models that offer broad applicability. However, foundational models often present a considerable number of non-transparent tuning parameters that require extensive manual optimization, limiting their usability for real-time streaming data analysis. Here, we introduce a reward function-based optimization to fine-tune foundational models and illustrate this approach for SAM (Segment Anything Model) framework by Meta. The reward functions can be constructed to represent the physics of the imaged system, including particle size distributions, geometries, and other criteria. By integrating a reward-driven optimization framework, we enhance SAM's adaptability and performance, leading to an optimized variant, SAM$^{*}$, that better aligns with the requirements of diverse segmentation tasks and particularly allows for real-time streaming data segmentation. We demonstrate the effectiveness of this approach in microscopy imaging, where precise segmentation is crucial for analyzing cellular structures, material interfaces, and nanoscale features.
>
---
#### [new 021] Mini-o3: Scaling Up Reasoning Patterns and Interaction Turns for Visual Search
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出Mini-o3系统，解决视觉搜索中复杂任务的推理不足问题。通过扩展工具交互次数、构建视觉探针数据集及改进强化学习策略，实现多步、多轮推理，提升模型在困难任务中的表现。**

- **链接: [http://arxiv.org/pdf/2509.07969v1](http://arxiv.org/pdf/2509.07969v1)**

> **作者:** Xin Lai; Junyi Li; Wei Li; Tao Liu; Tianjian Li; Hengshuang Zhao
>
> **备注:** Code, datasets, models are available at https://github.com/Mini-o3/Mini-o3. Project Page: https://mini-o3.github.io/
>
> **摘要:** Recent advances in large multimodal models have leveraged image-based tools with reinforcement learning to tackle visual problems. However, existing open-source approaches often exhibit monotonous reasoning patterns and allow only a limited number of interaction turns, making them inadequate for difficult tasks that require trial-and-error exploration. In this work, we address this limitation by scaling up tool-based interactions and introduce Mini-o3, a system that executes deep, multi-turn reasoning -- spanning tens of steps -- and achieves state-of-the-art performance on challenging visual search tasks. Our recipe for reproducing OpenAI o3-style behaviors comprises three key components. First, we construct the Visual Probe Dataset, a collection of thousands of challenging visual search problems designed for exploratory reasoning. Second, we develop an iterative data collection pipeline to obtain cold-start trajectories that exhibit diverse reasoning patterns, including depth-first search, trial-and-error, and goal maintenance. Third, we propose an over-turn masking strategy that prevents penalization of over-turn responses (those that hit the maximum number of turns) during reinforcement learning, thereby balancing training-time efficiency with test-time scalability. Despite training with an upper bound of only six interaction turns, our model generates trajectories that naturally scale to tens of turns at inference time, with accuracy improving as the number of turns increases. Extensive experiments demonstrate that Mini-o3 produces rich reasoning patterns and deep thinking paths, effectively solving challenging visual search problems.
>
---
#### [new 022] Dynamic Scene 3D Reconstruction of an Uncooperative Resident Space Object
- **分类: cs.CV**

- **简介: 论文研究非合作空间目标的动态三维重建，用于轨道服务与碎片清除任务。针对旋转目标重建难题，评估现有算法性能，并利用Isaac Sim构建仿真环境生成图像序列，初步验证Neuralangelo在静态场景下的高精度重建效果。**

- **链接: [http://arxiv.org/pdf/2509.07932v1](http://arxiv.org/pdf/2509.07932v1)**

> **作者:** Bala Prenith Reddy Gopu; Timothy Jacob Huber; George M. Nehma; Patrick Quinn; Madhur Tiwari; Matt Ueckermann; David Hinckley; Christopher McKenna
>
> **摘要:** Characterization of uncooperative Resident Space Objects (RSO) play a crucial role in On-Orbit Servicing (OOS) and Active Debris Removal (ADR) missions to assess the geometry and motion properties. To address the challenges of reconstructing tumbling uncooperative targets, this study evaluates the performance of existing state-of-the-art 3D reconstruction algorithms for dynamic scenes, focusing on their ability to generate geometrically accurate models with high-fidelity. To support our evaluation, we developed a simulation environment using Isaac Sim to generate physics-accurate 2D image sequences of tumbling satellite under realistic orbital lighting conditions. Our preliminary results on static scenes using Neuralangelo demonstrate promising reconstruction quality. The generated 3D meshes closely match the original CAD models with minimal errors and artifacts when compared using Cloud Compare (CC). The reconstructed models were able to capture critical fine details for mission planning. This provides a baseline for our ongoing evaluation of dynamic scene reconstruction.
>
---
#### [new 023] Nearest Neighbor Projection Removal Adversarial Training
- **分类: cs.CV; cs.LG; 68T45 (Primary), 68T10 (Secondary); I.5.4**

- **简介: 该论文属于图像分类任务中的对抗训练研究，旨在提升深度神经网络对对抗样本的鲁棒性。提出一种新框架，通过去除类间特征重叠，增强特征可分性，降低网络Lipschitz常数，提升模型鲁棒性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.07673v1](http://arxiv.org/pdf/2509.07673v1)**

> **作者:** Himanshu Singh; A. V. Subramanyam; Shivank Rajput; Mohan Kankanhalli
>
> **摘要:** Deep neural networks have exhibited impressive performance in image classification tasks but remain vulnerable to adversarial examples. Standard adversarial training enhances robustness but typically fails to explicitly address inter-class feature overlap, a significant contributor to adversarial susceptibility. In this work, we introduce a novel adversarial training framework that actively mitigates inter-class proximity by projecting out inter-class dependencies from adversarial and clean samples in the feature space. Specifically, our approach first identifies the nearest inter-class neighbors for each adversarial sample and subsequently removes projections onto these neighbors to enforce stronger feature separability. Theoretically, we demonstrate that our proposed logits correction reduces the Lipschitz constant of neural networks, thereby lowering the Rademacher complexity, which directly contributes to improved generalization and robustness. Extensive experiments across standard benchmarks including CIFAR-10, CIFAR-100, and SVHN show that our method demonstrates strong performance that is competitive with leading adversarial training techniques, highlighting significant achievements in both robust and clean accuracy. Our findings reveal the importance of addressing inter-class feature proximity explicitly to bolster adversarial robustness in DNNs.
>
---
#### [new 024] Beyond Motion Cues and Structural Sparsity: Revisiting Small Moving Target Detection
- **分类: cs.CV**

- **简介: 论文提出TenRPCANet框架，用于小移动目标检测任务，解决复杂背景下低信噪比和模糊视觉线索的问题。通过张量低秩稀疏分解，结合自注意力机制和特征优化模块，提升检测性能与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.07654v1](http://arxiv.org/pdf/2509.07654v1)**

> **作者:** Guoyi Zhang; Siyang Chen; Guangsheng Xu; Zhihua Shen; Han Wang; Xiaohu Zhang
>
> **摘要:** Small moving target detection is crucial for many defense applications but remains highly challenging due to low signal-to-noise ratios, ambiguous visual cues, and cluttered backgrounds. In this work, we propose a novel deep learning framework that differs fundamentally from existing approaches, which often rely on target-specific features or motion cues and tend to lack robustness in complex environments. Our key insight is that small target detection and background discrimination are inherently coupled, even cluttered video backgrounds often exhibit strong low-rank structures that can serve as stable priors for detection. We reformulate the task as a tensor-based low-rank and sparse decomposition problem and conduct a theoretical analysis of the background, target, and noise components to guide model design. Building on these insights, we introduce TenRPCANet, a deep neural network that requires minimal assumptions about target characteristics. Specifically, we propose a tokenization strategy that implicitly enforces multi-order tensor low-rank priors through a self-attention mechanism. This mechanism captures both local and non-local self-similarity to model the low-rank background without relying on explicit iterative optimization. In addition, inspired by the sparse component update in tensor RPCA, we design a feature refinement module to enhance target saliency. The proposed method achieves state-of-the-art performance on two highly distinct and challenging tasks: multi-frame infrared small target detection and space object detection. These results demonstrate both the effectiveness and the generalizability of our approach.
>
---
#### [new 025] Geospatial Foundational Embedder: Top-1 Winning Solution on EarthVision Embed2Scale Challenge (CVPR 2025)
- **分类: cs.CV**

- **简介: 该论文提出一种地学基础嵌入模型，用于将高光谱地理数据立方体转换为嵌入向量，以支持多种下游任务。属于遥感图像嵌入任务，解决了地理数据高效表示与应用的问题。**

- **链接: [http://arxiv.org/pdf/2509.06993v1](http://arxiv.org/pdf/2509.06993v1)**

> **作者:** Zirui Xu; Raphael Tang; Mike Bianco; Qi Zhang; Rishi Madhok; Nikolaos Karianakis; Fuxun Yu
>
> **备注:** CVPR 2025 EarthVision Embed2Scale challenge Top-1 Winning Solution
>
> **摘要:** EarthVision Embed2Scale challenge (CVPR 2025) aims to develop foundational geospatial models to embed SSL4EO-S12 hyperspectral geospatial data cubes into embedding vectors that faciliatetes various downstream tasks, e.g., classification, regression, etc. In this technical report, we introduce our proposed method for the Top-1 winning solution on the Embed2Scale Challenge.
>
---
#### [new 026] XSRD-Net: EXplainable Stroke Relapse Detection
- **分类: cs.CV; cs.AI; I.2.1**

- **简介: 该论文提出XSRD-Net模型，用于早期检测中风复发风险。通过融合影像与表格数据，实现二分类和生存时间预测任务，提升可解释性，揭示心脏疾病与颈动脉与复发的关系，以降低复发率。**

- **链接: [http://arxiv.org/pdf/2509.07772v1](http://arxiv.org/pdf/2509.07772v1)**

> **作者:** Christian Gapp; Elias Tappeiner; Martin Welk; Karl Fritscher; Stephanie Mangesius; Constantin Eisenschink; Philipp Deisl; Michael Knoflach; Astrid E. Grams; Elke R. Gizewski; Rainer Schubert
>
> **备注:** Contribution to MICAD 2025 conference, Nov. 19-21, 2025 | London, UK
>
> **摘要:** Stroke is the second most frequent cause of death world wide with an annual mortality of around 5.5 million. Recurrence rates of stroke are between 5 and 25% in the first year. As mortality rates for relapses are extraordinarily high (40%) it is of utmost importance to reduce the recurrence rates. We address this issue by detecting patients at risk of stroke recurrence at an early stage in order to enable appropriate therapy planning. To this end we collected 3D intracranial CTA image data and recorded concomitant heart diseases, the age and the gender of stroke patients between 2010 and 2024. We trained single- and multimodal deep learning based neural networks for binary relapse detection (Task 1) and for relapse free survival (RFS) time prediction together with a subsequent classification (Task 2). The separation of relapse from non-relapse patients (Task 1) could be solved with tabular data (AUC on test dataset: 0.84). However, for the main task, the regression (Task 2), our multimodal XSRD-net processed the modalities vision:tabular with 0.68:0.32 according to modality contribution measures. The c-index with respect to relapses for the multimodal model reached 0.68, and the AUC is 0.71 for the test dataset. Final, deeper interpretability analysis results could highlight a link between both heart diseases (tabular) and carotid arteries (vision) for the detection of relapses and the prediction of the RFS time. This is a central outcome that we strive to strengthen with ongoing data collection and model retraining.
>
---
#### [new 027] Fine-Tuning Vision-Language Models for Visual Navigation Assistance
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉导航任务，旨在帮助视障人士通过图像和语言指导完成室内导航。研究通过微调BLIP-2模型，结合LoRA技术与自定义评估指标，提升生成导航指令的准确性和方向性。**

- **链接: [http://arxiv.org/pdf/2509.07488v1](http://arxiv.org/pdf/2509.07488v1)**

> **作者:** Xiao Li; Bharat Gandhi; Ming Zhan; Mohit Nehra; Zhicheng Zhang; Yuchen Sun; Meijia Song; Naisheng Zhang; Xi Wang
>
> **摘要:** We address vision-language-driven indoor navigation to assist visually impaired individuals in reaching a target location using images and natural language guidance. Traditional navigation systems are ineffective indoors due to the lack of precise location data. Our approach integrates vision and language models to generate step-by-step navigational instructions, enhancing accessibility and independence. We fine-tune the BLIP-2 model with Low Rank Adaptation (LoRA) on a manually annotated indoor navigation dataset. We propose an evaluation metric that refines the BERT F1 score by emphasizing directional and sequential variables, providing a more comprehensive measure of navigational performance. After applying LoRA, the model significantly improved in generating directional instructions, overcoming limitations in the original BLIP-2 model.
>
---
#### [new 028] The Protocol Genome A Self Supervised Learning Framework from DICOM Headers
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **简介: 该论文提出Protocol Genome，一种基于DICOM头信息的自监督学习框架，用于提升医学影像模型的跨设备和跨模态泛化能力。通过结合影像与协议信息，改进模型校准与鲁棒性，应用于CT、MRI和X光等任务。**

- **链接: [http://arxiv.org/pdf/2509.06995v1](http://arxiv.org/pdf/2509.06995v1)**

> **作者:** Jimmy Joseph
>
> **摘要:** In this paper, we introduce the Protocol Genome, a self-supervised learning system that learns correlations from DICOM headers and achieves AUROC 0.901 (vs 0.847 baseline) and ECE 0.036 (vs 0.058) on fully held-out external validation. Our method also improves calibration and robustness across modalities (CT, MRI, CXR) and vendors. Clinical imaging is funneled through PACS/DICOM, where procedure choices (scanner make/model, sequence, kernel, kVp, TR/TE, and slice thickness) have consequences for contrast, noise, and artifact. These latent confounders impede the generalization of image-only networks across sites. We consider structured DICOM headers as a label and learn protocol-aware but clinically robust image representations. Protocol Genome obtains tokenized embeddings of de-identified header fields and models them along with image features using: (1) protocol-image contrastive learning, (2) masked protocol prediction, and (3) protocol-protocol translation. With 1.26M studies (7 health systems, 31 scanners, 3 vendors; CT, MR, CR/DR), we experiment on: (A) chest CT triage for PE, (B) brain MRI glioma grading, and (C) chest radiograph cardiomegaly detection. Relative to strong SSL baselines (SimCLR, MAE) as well as ImageNet transfer, Protocol Genome (+0.046: PE, +0.058: glioma, +0.041: cardiomegaly) is associated with higher external AUROC; 25-37% calibration improvements are obtained (p < 0.01, DeLong tests). While the gains may be task-dependent, they are preserved with 10-20% of labeled data. From a clinical point of view, the technique reduces false positives at protocol borders and is applicable in a PACS (DICOM C-FIND/C-MOVE, DICOMweb QIDO/WADO). We publish a model card and deployment guide, complete with both de-identification and bias audits.
>
---
#### [new 029] Point Linguist Model: Segment Any Object via Bridged Large 3D-Language Model
- **分类: cs.CV**

- **简介: 该论文提出Point Linguist Model（PLM），解决3D点云与LLM语义表示不匹配问题。通过引入OcDR和GRD模块，实现更准确的3D对象分割，提升跨任务性能。属于3D语义分割任务。**

- **链接: [http://arxiv.org/pdf/2509.07825v1](http://arxiv.org/pdf/2509.07825v1)**

> **作者:** Zhuoxu Huang; Mingqi Gao; Jungong Han
>
> **备注:** Preprint
>
> **摘要:** 3D object segmentation with Large Language Models (LLMs) has become a prevailing paradigm due to its broad semantics, task flexibility, and strong generalization. However, this paradigm is hindered by representation misalignment: LLMs process high-level semantic tokens, whereas 3D point clouds convey only dense geometric structures. In prior methods, misalignment limits both input and output. At the input stage, dense point patches require heavy pre-alignment, weakening object-level semantics and confusing similar distractors. At the output stage, predictions depend only on dense features without explicit geometric cues, leading to a loss of fine-grained accuracy. To address these limitations, we present the Point Linguist Model (PLM), a general framework that bridges the representation gap between LLMs and dense 3D point clouds without requiring large-scale pre-alignment between 3D-text or 3D-images. Specifically, we introduce Object-centric Discriminative Representation (OcDR), which learns object-centric tokens that capture target semantics and scene relations under a hard negative-aware training objective. This mitigates the misalignment between LLM tokens and 3D points, enhances resilience to distractors, and facilitates semantic-level reasoning within LLMs. For accurate segmentation, we introduce the Geometric Reactivation Decoder (GRD), which predicts masks by combining OcDR tokens carrying LLM-inferred geometry with corresponding dense features, preserving comprehensive dense features throughout the pipeline. Extensive experiments show that PLM achieves significant improvements of +7.3 mIoU on ScanNetv2 and +6.0 mIoU on Multi3DRefer for 3D referring segmentation, with consistent gains across 7 benchmarks spanning 4 different tasks, demonstrating the effectiveness of comprehensive object-centric reasoning for robust 3D understanding.
>
---
#### [new 030] Dimensionally Reduced Open-World Clustering: DROWCULA
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出DROWCULA方法，解决开放世界聚类问题，无需标注数据即可发现新类别。利用Vision Transformer生成嵌入，并结合流形学习提升聚类性能，在多个数据集上取得SOTA结果。**

- **链接: [http://arxiv.org/pdf/2509.07184v1](http://arxiv.org/pdf/2509.07184v1)**

> **作者:** Erencem Ozbey; Dimitrios I. Diochnos
>
> **备注:** 16 pages, 12 Figures, 12 Tables
>
> **摘要:** Working with annotated data is the cornerstone of supervised learning. Nevertheless, providing labels to instances is a task that requires significant human effort. Several critical real-world applications make things more complicated because no matter how many labels may have been identified in a task of interest, it could be the case that examples corresponding to novel classes may appear in the future. Not unsurprisingly, prior work in this, so-called, `open-world' context has focused a lot on semi-supervised approaches. Focusing on image classification, somehow paradoxically, we propose a fully unsupervised approach to the problem of determining the novel categories in a particular dataset. Our approach relies on estimating the number of clusters using Vision Transformers, which utilize attention mechanisms to generate vector embeddings. Furthermore, we incorporate manifold learning techniques to refine these embeddings by exploiting the intrinsic geometry of the data, thereby enhancing the overall image clustering performance. Overall, we establish new State-of-the-Art results on single-modal clustering and Novel Class Discovery on CIFAR-10, CIFAR-100, ImageNet-100, and Tiny ImageNet. We do so, both when the number of clusters is known or unknown ahead of time. The code is available at: https://github.com/DROWCULA/DROWCULA.
>
---
#### [new 031] SEEC: Segmentation-Assisted Multi-Entropy Models for Learned Lossless Image Compression
- **分类: cs.CV**

- **简介: 该论文提出SEEC模型，用于无损图像压缩任务。针对传统方法使用单一熵模型无法捕捉不同语义区域统计特性的不足，SEEC结合语义分割，为不同区域分配专用熵模型，提升压缩性能。实验表明其在压缩率和延迟间取得良好平衡。**

- **链接: [http://arxiv.org/pdf/2509.07704v1](http://arxiv.org/pdf/2509.07704v1)**

> **作者:** Chunhang Zheng; Zichang Ren; Dou Li
>
> **备注:** under review
>
> **摘要:** Recently, learned image compression has attracted considerable attention due to its superior performance over traditional methods. However, most existing approaches employ a single entropy model to estimate the probability distribution of pixel values across the entire image, which limits their ability to capture the diverse statistical characteristics of different semantic regions. To overcome this limitation, we propose Segmentation-Assisted Multi-Entropy Models for Lossless Image Compression (SEEC). Our framework utilizes semantic segmentation to guide the selection and adaptation of multiple entropy models, enabling more accurate probability distribution estimation for distinct semantic regions. Specifically, SEEC first extracts image features and then applies semantic segmentation to identify different regions, each assigned a specialized entropy model to better capture its unique statistical properties. Finally, a multi-channel discrete logistic mixture likelihood is employed to model the pixel value distributions effectively. Experimental results on benchmark datasets demonstrate that SEEC achieves state-of-the-art compression ratios while introducing only minimal encoding and decoding latency. With superior performance, the proposed model also supports Regions of Interest (ROIs) coding condition on the provided segmentation mask. Our code is available at https://github.com/chunbaobao/SEEC.
>
---
#### [new 032] HairGS: Hair Strand Reconstruction based on 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文提出HairGS方法，基于3D高斯点扩散技术，解决从多视角图像中重建发丝级几何结构的问题。通过分阶段流程实现发丝连接与拓扑优化，并引入新评估指标。属于三维重建任务。**

- **链接: [http://arxiv.org/pdf/2509.07774v1](http://arxiv.org/pdf/2509.07774v1)**

> **作者:** Yimin Pan; Matthias Nießner; Tobias Kirschstein
>
> **备注:** This is the arXiv preprint of the paper "Hair Strand Reconstruction based on 3D Gaussian Splatting" published at BMVC 2025. Project website: https://yimin-pan.github.io/hair-gs/
>
> **摘要:** Human hair reconstruction is a challenging problem in computer vision, with growing importance for applications in virtual reality and digital human modeling. Recent advances in 3D Gaussians Splatting (3DGS) provide efficient and explicit scene representations that naturally align with the structure of hair strands. In this work, we extend the 3DGS framework to enable strand-level hair geometry reconstruction from multi-view images. Our multi-stage pipeline first reconstructs detailed hair geometry using a differentiable Gaussian rasterizer, then merges individual Gaussian segments into coherent strands through a novel merging scheme, and finally refines and grows the strands under photometric supervision. While existing methods typically evaluate reconstruction quality at the geometric level, they often neglect the connectivity and topology of hair strands. To address this, we propose a new evaluation metric that serves as a proxy for assessing topological accuracy in strand reconstruction. Extensive experiments on both synthetic and real-world datasets demonstrate that our method robustly handles a wide range of hairstyles and achieves efficient reconstruction, typically completing within one hour. The project page can be found at: https://yimin-pan.github.io/hair-gs/
>
---
#### [new 033] Bias-Aware Machine Unlearning: Towards Fairer Vision Models via Controllable Forgetting
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种偏差感知的机器遗忘方法，通过可控遗忘减少视觉模型中的偏差。属于公平性优化任务，解决模型因训练数据偏差导致的不公平预测问题，采用多种策略在多个数据集上验证，显著提升公平性指标。**

- **链接: [http://arxiv.org/pdf/2509.07456v1](http://arxiv.org/pdf/2509.07456v1)**

> **作者:** Sai Siddhartha Chary Aylapuram; Veeraraju Elluru; Shivang Agarwal
>
> **备注:** Accepted for publication at ICCV 2025 UnMe workshop
>
> **摘要:** Deep neural networks often rely on spurious correlations in training data, leading to biased or unfair predictions in safety-critical domains such as medicine and autonomous driving. While conventional bias mitigation typically requires retraining from scratch or redesigning data pipelines, recent advances in machine unlearning provide a promising alternative for post-hoc model correction. In this work, we investigate \textit{Bias-Aware Machine Unlearning}, a paradigm that selectively removes biased samples or feature representations to mitigate diverse forms of bias in vision models. Building on privacy-preserving unlearning techniques, we evaluate various strategies including Gradient Ascent, LoRA, and Teacher-Student distillation. Through empirical analysis on three benchmark datasets, CUB-200-2011 (pose bias), CIFAR-10 (synthetic patch bias), and CelebA (gender bias in smile detection), we demonstrate that post-hoc unlearning can substantially reduce subgroup disparities, with improvements in demographic parity of up to \textbf{94.86\%} on CUB-200, \textbf{30.28\%} on CIFAR-10, and \textbf{97.37\%} on CelebA. These gains are achieved with minimal accuracy loss and with methods scoring an average of 0.62 across the 3 settings on the joint evaluation of utility, fairness, quality, and privacy. Our findings establish machine unlearning as a practical framework for enhancing fairness in deployed vision systems without necessitating full retraining.
>
---
#### [new 034] Realism to Deception: Investigating Deepfake Detectors Against Face Enhancement
- **分类: cs.CV**

- **简介: 论文研究面部增强技术对深度伪造检测器的影响，属于反取证任务。探讨这些技术是否能降低检测准确性，并通过实验验证其效果，强调需更稳健的检测方法。**

- **链接: [http://arxiv.org/pdf/2509.07178v1](http://arxiv.org/pdf/2509.07178v1)**

> **作者:** Muhammad Saad Saeed; Ijaz Ul Haq; Khalid Malik
>
> **摘要:** Face enhancement techniques are widely used to enhance facial appearance. However, they can inadvertently distort biometric features, leading to significant decrease in the accuracy of deepfake detectors. This study hypothesizes that these techniques, while improving perceptual quality, can degrade the performance of deepfake detectors. To investigate this, we systematically evaluate whether commonly used face enhancement methods can serve an anti-forensic role by reducing detection accuracy. We use both traditional image processing methods and advanced GAN-based enhancements to evaluate the robustness of deepfake detectors. We provide a comprehensive analysis of the effectiveness of these enhancement techniques, focusing on their impact on Na\"ive, Spatial, and Frequency-based detection methods. Furthermore, we conduct adversarial training experiments to assess whether exposure to face enhancement transformations improves model robustness. Experiments conducted on the FaceForensics++, DeepFakeDetection, and CelebDF-v2 datasets indicate that even basic enhancement filters can significantly reduce detection accuracy achieving ASR up to 64.63\%. In contrast, GAN-based techniques further exploit these vulnerabilities, achieving ASR up to 75.12\%. Our results demonstrate that face enhancement methods can effectively function as anti-forensic tools, emphasizing the need for more resilient and adaptive forensic methods.
>
---
#### [new 035] Accelerating Local AI on Consumer GPUs: A Hardware-Aware Dynamic Strategy for YOLOv10s
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文针对消费级GPU上YOLOv10s模型的本地AI性能瓶颈问题，提出一种硬件感知的动态推理策略。通过两阶段自适应推理方法，在保证检测精度的前提下显著提升推理速度，为消费级设备部署实时AI提供实用方案。**

- **链接: [http://arxiv.org/pdf/2509.07928v1](http://arxiv.org/pdf/2509.07928v1)**

> **作者:** Mahmudul Islam Masum; Miad Islam; Arif I. Sarwat
>
> **备注:** 6 pages, 7 figures
>
> **摘要:** As local AI grows in popularity, there is a critical gap between the benchmark performance of object detectors and their practical viability on consumer-grade hardware. While models like YOLOv10s promise real-time speeds, these metrics are typically achieved on high-power, desktop-class GPUs. This paper reveals that on resource-constrained systems, such as laptops with RTX 4060 GPUs, performance is not compute-bound but is instead dominated by system-level bottlenecks, as illustrated by a simple bottleneck test. To overcome this hardware-level constraint, we introduce a Two-Pass Adaptive Inference algorithm, a model-independent approach that requires no architectural changes. This study mainly focuses on adaptive inference strategies and undertakes a comparative analysis of architectural early-exit and resolution-adaptive routing, highlighting their respective trade-offs within a unified evaluation framework. The system uses a fast, low-resolution pass and only escalates to a high-resolution model pass when detection confidence is low. On a 5000-image COCO dataset, our method achieves a 1.85x speedup over a PyTorch Early-Exit baseline, with a modest mAP loss of 5.51%. This work provides a practical and reproducible blueprint for deploying high-performance, real-time AI on consumer-grade devices by shifting the focus from pure model optimization to hardware-aware inference strategies that maximize throughput.
>
---
#### [new 036] XBusNet: Text-Guided Breast Ultrasound Segmentation via Multimodal Vision-Language Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出XBusNet，用于乳腺超声图像分割。任务是解决小或低对比度病灶分割难题，通过结合图像与临床文本提示，设计双分支模型提升边界精度和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.07213v1](http://arxiv.org/pdf/2509.07213v1)**

> **作者:** Raja Mallina; Bryar Shareef
>
> **备注:** 15 pages, 3 figures, 4 tables
>
> **摘要:** Background: Precise breast ultrasound (BUS) segmentation supports reliable measurement, quantitative analysis, and downstream classification, yet remains difficult for small or low-contrast lesions with fuzzy margins and speckle noise. Text prompts can add clinical context, but directly applying weakly localized text-image cues (e.g., CAM/CLIP-derived signals) tends to produce coarse, blob-like responses that smear boundaries unless additional mechanisms recover fine edges. Methods: We propose XBusNet, a novel dual-prompt, dual-branch multimodal model that combines image features with clinically grounded text. A global pathway based on a CLIP Vision Transformer encodes whole-image semantics conditioned on lesion size and location, while a local U-Net pathway emphasizes precise boundaries and is modulated by prompts that describe shape, margin, and Breast Imaging Reporting and Data System (BI-RADS) terms. Prompts are assembled automatically from structured metadata, requiring no manual clicks. We evaluate on the Breast Lesions USG (BLU) dataset using five-fold cross-validation. Primary metrics are Dice and Intersection over Union (IoU); we also conduct size-stratified analyses and ablations to assess the roles of the global and local paths and the text-driven modulation. Results: XBusNet achieves state-of-the-art performance on BLU, with mean Dice of 0.8765 and IoU of 0.8149, outperforming six strong baselines. Small lesions show the largest gains, with fewer missed regions and fewer spurious activations. Ablation studies show complementary contributions of global context, local boundary modeling, and prompt-based modulation. Conclusions: A dual-prompt, dual-branch multimodal design that merges global semantics with local precision yields accurate BUS segmentation masks and improves robustness for small, low-contrast lesions.
>
---
#### [new 037] MVAT: Multi-View Aware Teacher for Weakly Supervised 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文提出MVAT框架，用于弱监督3D目标检测。通过多视角时序数据解决2D标注导致的投影模糊与遮挡问题，利用教师-学生蒸馏机制生成高质量伪标签，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2509.07507v1](http://arxiv.org/pdf/2509.07507v1)**

> **作者:** Saad Lahlali; Alexandre Fournier Montgieux; Nicolas Granger; Hervé Le Borgne; Quoc Cuong Pham
>
> **备注:** Accepted at WACV 2026
>
> **摘要:** Annotating 3D data remains a costly bottleneck for 3D object detection, motivating the development of weakly supervised annotation methods that rely on more accessible 2D box annotations. However, relying solely on 2D boxes introduces projection ambiguities since a single 2D box can correspond to multiple valid 3D poses. Furthermore, partial object visibility under a single viewpoint setting makes accurate 3D box estimation difficult. We propose MVAT, a novel framework that leverages temporal multi-view present in sequential data to address these challenges. Our approach aggregates object-centric point clouds across time to build 3D object representations as dense and complete as possible. A Teacher-Student distillation paradigm is employed: The Teacher network learns from single viewpoints but targets are derived from temporally aggregated static objects. Then the Teacher generates high quality pseudo-labels that the Student learns to predict from a single viewpoint for both static and moving objects. The whole framework incorporates a multi-view 2D projection loss to enforce consistency between predicted 3D boxes and all available 2D annotations. Experiments on the nuScenes and Waymo Open datasets demonstrate that MVAT achieves state-of-the-art performance for weakly supervised 3D object detection, significantly narrowing the gap with fully supervised methods without requiring any 3D box annotations. % \footnote{Code available upon acceptance} Our code is available in our public repository (\href{https://github.com/CEA-LIST/MVAT}{code}).
>
---
#### [new 038] DiGS: Accurate and Complete Surface Reconstruction from 3D Gaussians via Direct SDF Learning
- **分类: cs.CV; cs.CG**

- **简介: 该论文提出DiGS框架，解决3D高斯点云表面重建中的精度与完整性问题。通过将SDF学习嵌入3DGS流程，提升几何对齐与跨视角一致性，实现高质量重建。属于三维重建任务。**

- **链接: [http://arxiv.org/pdf/2509.07493v1](http://arxiv.org/pdf/2509.07493v1)**

> **作者:** Wenzhi Guo; Bing Wang
>
> **摘要:** 3D Gaussian Splatting (3DGS) has recently emerged as a powerful paradigm for photorealistic view synthesis, representing scenes with spatially distributed Gaussian primitives. While highly effective for rendering, achieving accurate and complete surface reconstruction remains challenging due to the unstructured nature of the representation and the absence of explicit geometric supervision. In this work, we propose DiGS, a unified framework that embeds Signed Distance Field (SDF) learning directly into the 3DGS pipeline, thereby enforcing strong and interpretable surface priors. By associating each Gaussian with a learnable SDF value, DiGS explicitly aligns primitives with underlying geometry and improves cross-view consistency. To further ensure dense and coherent coverage, we design a geometry-guided grid growth strategy that adaptively distributes Gaussians along geometry-consistent regions under a multi-scale hierarchy. Extensive experiments on standard benchmarks, including DTU, Mip-NeRF 360, and Tanks& Temples, demonstrate that DiGS consistently improves reconstruction accuracy and completeness while retaining high rendering fidelity.
>
---
#### [new 039] Deep Learning-Based Burned Area Mapping Using Bi-Temporal Siamese Networks and AlphaEarth Foundation Datasets
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出基于深度学习的烧伤区域映射方法，利用AlphaEarth数据集和Siamese U-Net模型，解决自动化、高精度烧伤区域检测问题，提升全球火灾监测能力。**

- **链接: [http://arxiv.org/pdf/2509.07852v1](http://arxiv.org/pdf/2509.07852v1)**

> **作者:** Seyd Teymoor Seydi
>
> **摘要:** Accurate and timely mapping of burned areas is crucial for environmental monitoring, disaster management, and assessment of climate change. This study presents a novel approach to automated burned area mapping using the AlphaEArth dataset combined with the Siamese U-Net deep learning architecture. The AlphaEArth Dataset, comprising high-resolution optical and thermal infrared imagery with comprehensive ground-truth annotations, provides an unprecedented resource for training robust burned area detection models. We trained our model with the Monitoring Trends in Burn Severity (MTBS) dataset in the contiguous US and evaluated it with 17 regions cross in Europe. Our experimental results demonstrate that the proposed ensemble approach achieves superior performance with an overall accuracy of 95%, IoU of 0.6, and F1-score of 74% on the test dataset. The model successfully identifies burned areas across diverse ecosystems with complex background, showing particular strength in detecting partially burned vegetation and fire boundaries and its transferability and high generalization in burned area mapping. This research contributes to the advancement of automated fire damage assessment and provides a scalable solution for global burn area monitoring using the AlphaEarth dataset.
>
---
#### [new 040] Visual-TableQA: Open-Domain Benchmark for Reasoning over Table Images
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出Visual-TableQA，一个大规模开放域数据集，用于评估视觉语言模型对表格图像的推理能力。针对现有基准在规模、多样性及推理深度上的不足，其通过多模型协作生成2.5k LaTeX表格和6k问答对，促进模型泛化能力提升。**

- **链接: [http://arxiv.org/pdf/2509.07966v1](http://arxiv.org/pdf/2509.07966v1)**

> **作者:** Boammani Aser Lompo; Marc Haraoui
>
> **备注:** Work in Progress
>
> **摘要:** Visual reasoning over structured data such as tables is a critical capability for modern vision-language models (VLMs), yet current benchmarks remain limited in scale, diversity, or reasoning depth, especially when it comes to rendered table images. Addressing this gap, we introduce Visual-TableQA, a large-scale, open-domain multimodal dataset specifically designed to evaluate and enhance visual reasoning over complex tabular data. Our generation pipeline is modular, scalable, and fully autonomous, involving multiple reasoning LLMs collaborating across distinct roles: generation, validation, and inspiration. Visual-TableQA comprises 2.5k richly structured LaTeX-rendered tables and 6k reasoning-intensive QA pairs, all produced at a cost of under USD 100. To promote diversity and creativity, our pipeline performs multi-model collaborative data generation via cross-model prompting ('inspiration') and LLM-jury filtering. Stronger models seed layouts and topics that weaker models elaborate, collectively distilling diverse reasoning patterns and visual structures into the dataset. Empirical results show that models fine-tuned on Visual-TableQA generalize robustly to external benchmarks, outperforming several proprietary models despite the dataset's synthetic nature. The full pipeline and resources are publicly available at https://github.com/AI-4-Everyone/Visual-TableQA.
>
---
#### [new 041] VLMs-in-the-Wild: Bridging the Gap Between Academic Benchmarks and Enterprise Reality
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出ViLD框架，解决学术评估与企业实际应用间的差距问题。定义10项企业关键任务，引入BlockWeaver算法，构建7500样本基准数据集，评估开源VLM在真实场景中的表现，为企业部署提供依据。**

- **链接: [http://arxiv.org/pdf/2509.06994v1](http://arxiv.org/pdf/2509.06994v1)**

> **作者:** Srihari Bandraupalli; Anupam Purwar
>
> **摘要:** Open-source Vision-Language Models show immense promise for enterprise applications, yet a critical disconnect exists between academic evaluation and enterprise deployment requirements. Current benchmarks rely heavily on multiple-choice questions and synthetic data, failing to capture the complexity of real-world business applications like social media content analysis. This paper introduces VLM-in-the-Wild (ViLD), a comprehensive framework to bridge this gap by evaluating VLMs on operational enterprise requirements. We define ten business-critical tasks: logo detection, OCR, object detection, human presence and demographic analysis, human activity and appearance analysis, scene detection, camera perspective and media quality assessment, dominant colors, comprehensive description, and NSFW detection. To this framework, we bring an innovative BlockWeaver Algorithm that solves the challenging problem of comparing unordered, variably-grouped OCR outputs from VLMs without relying on embeddings or LLMs, achieving remarkable speed and reliability. To demonstrate efficacy of ViLD, we constructed a new benchmark dataset of 7,500 diverse samples, carefully stratified from a corpus of one million real-world images and videos. ViLD provides actionable insights by combining semantic matching (both embedding-based and LLM-as-a-judge approaches), traditional metrics, and novel methods to measure the completeness and faithfulness of descriptive outputs. By benchmarking leading open-source VLMs (Qwen, MIMO, and InternVL) against a powerful proprietary baseline as per ViLD framework, we provide one of the first industry-grounded, task-driven assessment of VLMs capabilities, offering actionable insights for their deployment in enterprise environments.
>
---
#### [new 042] Automated Evaluation of Gender Bias Across 13 Large Multimodal Models
- **分类: cs.CV; cs.AI; cs.CY; I.2.7; F.2.2**

- **简介: 该论文评估13种大 multimodal 模型中的性别偏见，通过生成职业图像并分析性别表现，揭示模型放大职业性别刻板印象的问题，并提出自动化评估工具，推动AI公平性研究。**

- **链接: [http://arxiv.org/pdf/2509.07050v1](http://arxiv.org/pdf/2509.07050v1)**

> **作者:** Juan Manuel Contreras
>
> **摘要:** Large multimodal models (LMMs) have revolutionized text-to-image generation, but they risk perpetuating the harmful social biases in their training data. Prior work has identified gender bias in these models, but methodological limitations prevented large-scale, comparable, cross-model analysis. To address this gap, we introduce the Aymara Image Fairness Evaluation, a benchmark for assessing social bias in AI-generated images. We test 13 commercially available LMMs using 75 procedurally-generated, gender-neutral prompts to generate people in stereotypically-male, stereotypically-female, and non-stereotypical professions. We then use a validated LLM-as-a-judge system to score the 965 resulting images for gender representation. Our results reveal (p < .001 for all): 1) LMMs systematically not only reproduce but actually amplify occupational gender stereotypes relative to real-world labor data, generating men in 93.0% of images for male-stereotyped professions but only 22.5% for female-stereotyped professions; 2) Models exhibit a strong default-male bias, generating men in 68.3% of the time for non-stereotyped professions; and 3) The extent of bias varies dramatically across models, with overall male representation ranging from 46.7% to 73.3%. Notably, the top-performing model de-amplified gender stereotypes and approached gender parity, achieving the highest fairness scores. This variation suggests high bias is not an inevitable outcome but a consequence of design choices. Our work provides the most comprehensive cross-model benchmark of gender bias to date and underscores the necessity of standardized, automated evaluation tools for promoting accountability and fairness in AI development.
>
---
#### [new 043] RayGaussX: Accelerating Gaussian-Based Ray Marching for Real-Time and High-Quality Novel View Synthesis
- **分类: cs.CV**

- **简介: 该论文提出RayGaussX方法，用于加速基于高斯的光线追踪，解决实时高质量新视角合成问题。通过优化训练与渲染速度，提升视觉质量，适用于真实场景。**

- **链接: [http://arxiv.org/pdf/2509.07782v1](http://arxiv.org/pdf/2509.07782v1)**

> **作者:** Hugo Blanc; Jean-Emmanuel Deschaud; Alexis Paljic
>
> **备注:** Project page with videos and code: https://raygaussx.github.io/
>
> **摘要:** RayGauss has achieved state-of-the-art rendering quality for novel-view synthesis on synthetic and indoor scenes by representing radiance and density fields with irregularly distributed elliptical basis functions, rendered via volume ray casting using a Bounding Volume Hierarchy (BVH). However, its computational cost prevents real-time rendering on real-world scenes. Our approach, RayGaussX, builds on RayGauss by introducing key contributions that accelerate both training and inference. Specifically, we incorporate volumetric rendering acceleration strategies such as empty-space skipping and adaptive sampling, enhance ray coherence, and introduce scale regularization to reduce false-positive intersections. Additionally, we propose a new densification criterion that improves density distribution in distant regions, leading to enhanced graphical quality on larger scenes. As a result, RayGaussX achieves 5x to 12x faster training and 50x to 80x higher rendering speeds (FPS) on real-world datasets while improving visual quality by up to +0.56 dB in PSNR. Project page with videos and code: https://raygaussx.github.io/.
>
---
#### [new 044] CAViAR: Critic-Augmented Video Agentic Reasoning
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出CAViAR模型，用于复杂视频推理任务。针对现有模型在长视频和复杂任务中表现下降的问题，设计了一个结合子代理和批评者的语言模型代理，通过动态调用模块并学习成功推理序列，提升视频理解与推理能力。**

- **链接: [http://arxiv.org/pdf/2509.07680v1](http://arxiv.org/pdf/2509.07680v1)**

> **作者:** Sachit Menon; Ahmet Iscen; Arsha Nagrani; Tobias Weyand; Carl Vondrick; Cordelia Schmid
>
> **摘要:** Video understanding has seen significant progress in recent years, with models' performance on perception from short clips continuing to rise. Yet, multiple recent benchmarks, such as LVBench, Neptune, and ActivityNet-RTL, show performance wanes for tasks requiring complex reasoning on videos as queries grow more complex and videos grow longer. In this work, we ask: can existing perception capabilities be leveraged to successfully perform more complex video reasoning? In particular, we develop a large language model agent given access to video modules as subagents or tools. Rather than following a fixed procedure to solve queries as in previous work such as Visual Programming, ViperGPT, and MoReVQA, the agent uses the results of each call to a module to determine subsequent steps. Inspired by work in the textual reasoning domain, we introduce a critic to distinguish between instances of successful and unsuccessful sequences from the agent. We show that the combination of our agent and critic achieve strong performance on the previously-mentioned datasets.
>
---
#### [new 045] EHWGesture -- A dataset for multimodal understanding of clinical gestures
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出EHWGesture数据集，用于多模态临床手势理解任务。旨在解决动态手势识别中时空复杂性及缺乏多模态、多视角数据的问题。工作包括采集1100余条高质量多模态视频，并组织速度分类以评估动作质量。**

- **链接: [http://arxiv.org/pdf/2509.07525v1](http://arxiv.org/pdf/2509.07525v1)**

> **作者:** Gianluca Amprimo; Alberto Ancilotto; Alessandro Savino; Fabio Quazzolo; Claudia Ferraris; Gabriella Olmo; Elisabetta Farella; Stefano Di Carlo
>
> **备注:** Accepted at ICCV 2025 Workshop on AI-driven Skilled Activity Understanding, Assessment & Feedback Generation
>
> **摘要:** Hand gesture understanding is essential for several applications in human-computer interaction, including automatic clinical assessment of hand dexterity. While deep learning has advanced static gesture recognition, dynamic gesture understanding remains challenging due to complex spatiotemporal variations. Moreover, existing datasets often lack multimodal and multi-view diversity, precise ground-truth tracking, and an action quality component embedded within gestures. This paper introduces EHWGesture, a multimodal video dataset for gesture understanding featuring five clinically relevant gestures. It includes over 1,100 recordings (6 hours), captured from 25 healthy subjects using two high-resolution RGB-Depth cameras and an event camera. A motion capture system provides precise ground-truth hand landmark tracking, and all devices are spatially calibrated and synchronized to ensure cross-modal alignment. Moreover, to embed an action quality task within gesture understanding, collected recordings are organized in classes of execution speed that mirror clinical evaluations of hand dexterity. Baseline experiments highlight the dataset's potential for gesture classification, gesture trigger detection, and action quality assessment. Thus, EHWGesture can serve as a comprehensive benchmark for advancing multimodal clinical gesture understanding.
>
---
#### [new 046] FedAPT: Federated Adversarial Prompt Tuning for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文提出FedAPT方法，用于提升联邦视觉-语言模型的对抗鲁棒性。针对非独立同分布下客户端与全局模型间类别信息差距问题，设计类感知提示生成器和跨层共享策略，显著增强模型在对抗攻击下的性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.06992v1](http://arxiv.org/pdf/2509.06992v1)**

> **作者:** Kun Zhai; Siheng Chen; Xingjun Ma; Yu-Gang Jiang
>
> **备注:** ACM MM25
>
> **摘要:** Federated Prompt Tuning (FPT) is an efficient method for cross-client collaborative fine-tuning of large Vision-Language Models (VLMs). However, models tuned using FPT are vulnerable to adversarial attacks, leading to misclassification in downstream tasks. In this work, we introduce Federated Adversarial Prompt Tuning (\textbf{FedAPT}), a novel method designed to enhance the adversarial robustness of FPT. We identify a key issue in FedAPT under non-independent and identically distributed (non-IID) settings: a \textit{class information gap} between clients and the global model. Clients rely solely on limited local label information to generate adversarial samples for training, while the global model must defend against adversarial attacks from global labels. To address this issue, we propose a \textbf{class-aware prompt generator} that generates visual prompts from text prompts. This generator is guided by a \emph{Global Label Embedding} (serving as a ``beacon") which encodes cross-client label information to create more globally-aligned visual prompts. Additionally, we propose a \textbf{cross-layer generator sharing} strategy to enhance prompt coupling across different layers of the model, further boosting adversarial robustness. Extensive experiments on multiple image classification datasets demonstrate the superiority of FedAPT in improving adversarial robustness, outperforming existing methods by a large margin. FedAPT also exhibits exceptional generalization in cross-domain and cross-dataset scenarios, indicating its effectiveness in real-world applications.
>
---
#### [new 047] Semantic Watermarking Reinvented: Enhancing Robustness and Generation Quality with Fourier Integrity
- **分类: cs.CV**

- **简介: 该论文提出一种语义水印技术SFW，解决现有方法在频率完整性丢失和抗攻击能力不足的问题。通过强制Hermitian对称性和中心感知嵌入策略，提升水印鲁棒性与图像保真度，实验表明其检测性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.07647v1](http://arxiv.org/pdf/2509.07647v1)**

> **作者:** Sung Ju Lee; Nam Ik Cho
>
> **备注:** Accepted to the IEEE/CVF International Conference on Computer Vision (ICCV) 2025. Project page: https://thomas11809.github.io/SFWMark/ Code: https://github.com/thomas11809/SFWMark
>
> **摘要:** Semantic watermarking techniques for latent diffusion models (LDMs) are robust against regeneration attacks, but often suffer from detection performance degradation due to the loss of frequency integrity. To tackle this problem, we propose a novel embedding method called Hermitian Symmetric Fourier Watermarking (SFW), which maintains frequency integrity by enforcing Hermitian symmetry. Additionally, we introduce a center-aware embedding strategy that reduces the vulnerability of semantic watermarking due to cropping attacks by ensuring robust information retention. To validate our approach, we apply these techniques to existing semantic watermarking schemes, enhancing their frequency-domain structures for better robustness and retrieval accuracy. Extensive experiments demonstrate that our methods achieve state-of-the-art verification and identification performance, surpassing previous approaches across various attack scenarios. Ablation studies confirm the impact of SFW on detection capabilities, the effectiveness of the center-aware embedding against cropping, and how message capacity influences identification accuracy. Notably, our method achieves the highest detection accuracy while maintaining superior image fidelity, as evidenced by FID and CLIP scores. Conclusively, our proposed SFW is shown to be an effective framework for balancing robustness and image fidelity, addressing the inherent trade-offs in semantic watermarking. Code available at https://github.com/thomas11809/SFWMark
>
---
#### [new 048] Detection and Recovery of Adversarial Slow-Pose Drift in Offloaded Visual-Inertial Odometry
- **分类: cs.CV; cs.MM**

- **简介: 该论文研究边缘服务器中视觉惯性里程计（VIO）的对抗性慢姿态漂移问题，提出一种无监督检测与恢复机制，通过学习运动时间规律，在运行时检测偏差并恢复姿态一致性，有效降低轨迹和姿态误差。**

- **链接: [http://arxiv.org/pdf/2509.07130v1](http://arxiv.org/pdf/2509.07130v1)**

> **作者:** Soruya Saha; Md Nurul Absurd; Saptarshi Debroy
>
> **备注:** 12 Pages, 8 Figures
>
> **摘要:** Visual-Inertial Odometry (VIO) supports immersive Virtual Reality (VR) by fusing camera and Inertial Measurement Unit (IMU) data for real-time pose. However, current trend of offloading VIO to edge servers can lead server-side threat surface where subtle pose spoofing can accumulate into substantial drift, while evading heuristic checks. In this paper, we study this threat and present an unsupervised, label-free detection and recovery mechanism. The proposed model is trained on attack-free sessions to learn temporal regularities of motion to detect runtime deviations and initiate recovery to restore pose consistency. We evaluate the approach in a realistic offloaded-VIO environment using ILLIXR testbed across multiple spoofing intensities. Experimental results in terms of well-known performance metrics show substantial reductions in trajectory and pose error compared to a no-defense baseline.
>
---
#### [new 049] XOCT: Enhancing OCT to OCTA Translation via Cross-Dimensional Supervised Multi-Scale Feature Learning
- **分类: cs.CV; J.3**

- **简介: 该论文提出XOCT框架，解决OCT到OCTA转换中血管细节重建不足的问题。通过跨维度监督与多尺度特征融合，提升层间血管区分与细节还原，提高诊断准确性与可靠性。**

- **链接: [http://arxiv.org/pdf/2509.07455v1](http://arxiv.org/pdf/2509.07455v1)**

> **作者:** Pooya Khosravi; Kun Han; Anthony T. Wu; Arghavan Rezvani; Zexin Feng; Xiaohui Xie
>
> **备注:** 11 pages, 3 figures, Accepted to MICCAI 2025
>
> **摘要:** Optical Coherence Tomography Angiography (OCTA) and its derived en-face projections provide high-resolution visualization of the retinal and choroidal vasculature, which is critical for the rapid and accurate diagnosis of retinal diseases. However, acquiring high-quality OCTA images is challenging due to motion sensitivity and the high costs associated with software modifications for conventional OCT devices. Moreover, current deep learning methods for OCT-to-OCTA translation often overlook the vascular differences across retinal layers and struggle to reconstruct the intricate, dense vascular details necessary for reliable diagnosis. To overcome these limitations, we propose XOCT, a novel deep learning framework that integrates Cross-Dimensional Supervision (CDS) with a Multi-Scale Feature Fusion (MSFF) network for layer-aware vascular reconstruction. Our CDS module leverages 2D layer-wise en-face projections, generated via segmentation-weighted z-axis averaging, as supervisory signals to compel the network to learn distinct representations for each retinal layer through fine-grained, targeted guidance. Meanwhile, the MSFF module enhances vessel delineation through multi-scale feature extraction combined with a channel reweighting strategy, effectively capturing vascular details at multiple spatial scales. Our experiments on the OCTA-500 dataset demonstrate XOCT's improvements, especially for the en-face projections which are significant for clinical evaluation of retinal pathologies, underscoring its potential to enhance OCTA accessibility, reliability, and diagnostic value for ophthalmic disease detection and monitoring. The code is available at https://github.com/uci-cbcl/XOCT.
>
---
#### [new 050] DreamLifting: A Plug-in Module Lifting MV Diffusion Models for 3D Asset Generation
- **分类: cs.CV**

- **简介: 该论文提出LGAA框架，解决端到端生成PBR材质3D资产的问题。通过模块化设计，结合多视角扩散模型与2D高斯溅射技术，实现高效、高质量的3D资产生成。**

- **链接: [http://arxiv.org/pdf/2509.07435v1](http://arxiv.org/pdf/2509.07435v1)**

> **作者:** Ze-Xin Yin; Jiaxiong Qiu; Liu Liu; Xinjie Wang; Wei Sui; Zhizhong Su; Jian Yang; Jin Xie
>
> **备注:** 14 pages, 7 figures, project page: https://zx-yin.github.io/dreamlifting/
>
> **摘要:** The labor- and experience-intensive creation of 3D assets with physically based rendering (PBR) materials demands an autonomous 3D asset creation pipeline. However, most existing 3D generation methods focus on geometry modeling, either baking textures into simple vertex colors or leaving texture synthesis to post-processing with image diffusion models. To achieve end-to-end PBR-ready 3D asset generation, we present Lightweight Gaussian Asset Adapter (LGAA), a novel framework that unifies the modeling of geometry and PBR materials by exploiting multi-view (MV) diffusion priors from a novel perspective. The LGAA features a modular design with three components. Specifically, the LGAA Wrapper reuses and adapts network layers from MV diffusion models, which encapsulate knowledge acquired from billions of images, enabling better convergence in a data-efficient manner. To incorporate multiple diffusion priors for geometry and PBR synthesis, the LGAA Switcher aligns multiple LGAA Wrapper layers encapsulating different knowledge. Then, a tamed variational autoencoder (VAE), termed LGAA Decoder, is designed to predict 2D Gaussian Splatting (2DGS) with PBR channels. Finally, we introduce a dedicated post-processing procedure to effectively extract high-quality, relightable mesh assets from the resulting 2DGS. Extensive quantitative and qualitative experiments demonstrate the superior performance of LGAA with both text-and image-conditioned MV diffusion models. Additionally, the modular design enables flexible incorporation of multiple diffusion priors, and the knowledge-preserving scheme leads to efficient convergence trained on merely 69k multi-view instances. Our code, pre-trained weights, and the dataset used will be publicly available via our project page: https://zx-yin.github.io/dreamlifting/.
>
---
#### [new 051] ANYPORTAL: Zero-Shot Consistent Video Background Replacement
- **分类: cs.CV**

- **简介: 论文提出ANYPORTAL框架，实现零样本视频背景替换。该任务旨在解决视频生成中背景替换时的前景一致性与时间连贯性问题。通过结合视频与图像扩散模型，并引入细化投影算法，实现高质量、无训练的视频编辑。**

- **链接: [http://arxiv.org/pdf/2509.07472v1](http://arxiv.org/pdf/2509.07472v1)**

> **作者:** Wenshuo Gao; Xicheng Lan; Shuai Yang
>
> **备注:** 8 pages, ICCV 2025, Website: https://gaowenshuo.github.io/AnyPortal/
>
> **摘要:** Despite the rapid advancements in video generation technology, creating high-quality videos that precisely align with user intentions remains a significant challenge. Existing methods often fail to achieve fine-grained control over video details, limiting their practical applicability. We introduce ANYPORTAL, a novel zero-shot framework for video background replacement that leverages pre-trained diffusion models. Our framework collaboratively integrates the temporal prior of video diffusion models with the relighting capabilities of image diffusion models in a zero-shot setting. To address the critical challenge of foreground consistency, we propose a Refinement Projection Algorithm, which enables pixel-level detail manipulation to ensure precise foreground preservation. ANYPORTAL is training-free and overcomes the challenges of achieving foreground consistency and temporally coherent relighting. Experimental results demonstrate that ANYPORTAL achieves high-quality results on consumer-grade GPUs, offering a practical and efficient solution for video content creation and editing.
>
---
#### [new 052] Active Membership Inference Test (aMINT): Enhancing Model Auditability with Multi-Task Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出aMINT方法，通过多任务学习提升模型可审计性，解决检测数据是否参与模型训练的问题。其核心工作是同时训练主模型与辅助检测模型，利用激活图增强训练数据识别能力，在多个基准上达到80%以上准确率。**

- **链接: [http://arxiv.org/pdf/2509.07879v1](http://arxiv.org/pdf/2509.07879v1)**

> **作者:** Daniel DeAlcala; Aythami Morales; Julian Fierrez; Gonzalo Mancera; Ruben Tolosana; Javier Ortega-Garcia
>
> **备注:** In Proc. IEEE/CVF Intenational Conference on Computer Vision, ICCV, 2025
>
> **摘要:** Active Membership Inference Test (aMINT) is a method designed to detect whether given data were used during the training of machine learning models. In Active MINT, we propose a novel multitask learning process that involves training simultaneously two models: the original or Audited Model, and a secondary model, referred to as the MINT Model, responsible for identifying the data used for training the Audited Model. This novel multi-task learning approach has been designed to incorporate the auditability of the model as an optimization objective during the training process of neural networks. The proposed approach incorporates intermediate activation maps as inputs to the MINT layers, which are trained to enhance the detection of training data. We present results using a wide range of neural networks, from lighter architectures such as MobileNet to more complex ones such as Vision Transformers, evaluated in 5 public benchmarks. Our proposed Active MINT achieves over 80% accuracy in detecting if given data was used for training, significantly outperforming previous approaches in the literature. Our aMINT and related methodological developments contribute to increasing transparency in AI models, facilitating stronger safeguards in AI deployments to achieve proper security, privacy, and copyright protection.
>
---
#### [new 053] Faster VGGT with Block-Sparse Global Attention
- **分类: cs.CV**

- **简介: 论文提出一种改进的VGGT模型，通过块稀疏全局注意力机制提升多视角重建效率。属于多视角重建任务，解决传统全局注意力计算复杂度高、运行缓慢的问题，采用优化的块稀疏核实现4倍加速，无需重训练。**

- **链接: [http://arxiv.org/pdf/2509.07120v1](http://arxiv.org/pdf/2509.07120v1)**

> **作者:** Chung-Shien Brian Wang; Christian Schmidt; Jens Piekenbrinck; Bastian Leibe
>
> **备注:** Project page at https://vision.rwth-aachen.de/sparse-vggt
>
> **摘要:** Efficient and accurate feed-forward multi-view reconstruction has long been an important task in computer vision. Recent transformer-based models like VGGT and $\pi^3$ have achieved impressive results with simple architectures, yet they face an inherent runtime bottleneck, due to the quadratic complexity of the global attention layers, that limits the scalability to large image sets. In this paper, we empirically analyze the global attention matrix of these models and observe that probability mass concentrates on a small subset of patch-patch interactions that correspond to cross-view geometric matches. Motivated by the structured attention and inspired by recent advancement in large language models, we propose a replacement for the dense global attention operation based on highly optimized block-sparse kernels, yielding up to $4\times$ faster inference with comparable task performance. Our retrofit requires no retraining of the backbone, extends to both VGGT and $\pi^3$, and supports large image collections. Evaluations on a comprehensive suite of multi-view benchmarks demonstrate the effectiveness of our approach.
>
---
#### [new 054] MedicalPatchNet: A Patch-Based Self-Explainable AI Architecture for Chest X-ray Classification
- **分类: cs.CV; cs.LG**

- **简介: 论文提出MedicalPatchNet，用于胸部X光分类任务，旨在提升AI模型的可解释性。通过将图像分割为独立 patches 进行分类与聚合，实现无需后处理的可视化解释，提高病理定位准确性，增强临床信任。**

- **链接: [http://arxiv.org/pdf/2509.07477v1](http://arxiv.org/pdf/2509.07477v1)**

> **作者:** Patrick Wienholt; Christiane Kuhl; Jakob Nikolas Kather; Sven Nebelung; Daniel Truhn
>
> **摘要:** Deep neural networks excel in radiological image classification but frequently suffer from poor interpretability, limiting clinical acceptance. We present MedicalPatchNet, an inherently self-explainable architecture for chest X-ray classification that transparently attributes decisions to distinct image regions. MedicalPatchNet splits images into non-overlapping patches, independently classifies each patch, and aggregates predictions, enabling intuitive visualization of each patch's diagnostic contribution without post-hoc techniques. Trained on the CheXpert dataset (223,414 images), MedicalPatchNet matches the classification performance (AUROC 0.907 vs. 0.908) of EfficientNet-B0, while substantially improving interpretability: MedicalPatchNet demonstrates substantially improved interpretability with higher pathology localization accuracy (mean hit-rate 0.485 vs. 0.376 with Grad-CAM) on the CheXlocalize dataset. By providing explicit, reliable explanations accessible even to non-AI experts, MedicalPatchNet mitigates risks associated with shortcut learning, thus improving clinical trust. Our model is publicly available with reproducible training and inference scripts and contributes to safer, explainable AI-assisted diagnostics across medical imaging domains. We make the code publicly available: https://github.com/TruhnLab/MedicalPatchNet
>
---
#### [new 055] Visible Yet Unreadable: A Systematic Blind Spot of Vision Language Models Across Writing Systems
- **分类: cs.CV; cs.AI**

- **简介: 论文研究视觉语言模型（VLMs）在不同文字系统下的鲁棒性，构建“可见但不可读”基准测试，发现VLMs在文本扰动下表现显著下降。任务是评估模型对文字结构的感知能力，旨在推动更稳健的多模态模型设计。**

- **链接: [http://arxiv.org/pdf/2509.06996v1](http://arxiv.org/pdf/2509.06996v1)**

> **作者:** Jie Zhang; Ting Xu; Gelei Deng; Runyi Hu; Han Qiu; Tianwei Zhang; Qing Guo; Ivor Tsang
>
> **摘要:** Writing is a universal cultural technology that reuses vision for symbolic communication. Humans display striking resilience: we readily recognize words even when characters are fragmented, fused, or partially occluded. This paper investigates whether advanced vision language models (VLMs) share this resilience. We construct two psychophysics inspired benchmarks across distinct writing systems, Chinese logographs and English alphabetic words, by splicing, recombining, and overlaying glyphs to yield ''visible but unreadable'' stimuli for models while remaining legible to humans. Despite strong performance on clean text, contemporary VLMs show a severe drop under these perturbations, frequently producing unrelated or incoherent outputs. The pattern suggests a structural limitation: models heavily leverage generic visual invariances but under rely on compositional priors needed for robust literacy. We release stimuli generation code, prompts, and evaluation protocols to facilitate transparent replication and follow up work. Our findings motivate architectures and training strategies that encode symbol segmentation, composition, and binding across scripts, and they delineate concrete challenges for deploying multimodal systems in education, accessibility, cultural heritage, and security.
>
---
#### [new 056] Feature Space Analysis by Guided Diffusion Model
- **分类: cs.CV; eess.IV**

- **简介: 该论文提出一种基于引导扩散模型的解码器，用于分析DNN的特征空间。通过生成与指定特征高度匹配的图像，揭示DNN对图像属性的编码方式，解决了传统方法无法保证特征匹配的问题。**

- **链接: [http://arxiv.org/pdf/2509.07936v1](http://arxiv.org/pdf/2509.07936v1)**

> **作者:** Kimiaki Shirahama; Miki Yanobu; Kaduki Yamashita; Miho Ohsaki
>
> **备注:** 19 pages, 13 figures, codes: https://github.com/KimiakiShirahama/FeatureSpaceAnalysisByGuidedDiffusionModel
>
> **摘要:** One of the key issues in Deep Neural Networks (DNNs) is the black-box nature of their internal feature extraction process. Targeting vision-related domains, this paper focuses on analysing the feature space of a DNN by proposing a decoder that can generate images whose features are guaranteed to closely match a user-specified feature. Owing to this guarantee that is missed in past studies, our decoder allows us to evidence which of various attributes in an image are encoded into a feature by the DNN, by generating images whose features are in proximity to that feature. Our decoder is implemented as a guided diffusion model that guides the reverse image generation of a pre-trained diffusion model to minimise the Euclidean distance between the feature of a clean image estimated at each step and the user-specified feature. One practical advantage of our decoder is that it can analyse feature spaces of different DNNs with no additional training and run on a single COTS GPU. The experimental results targeting CLIP's image encoder, ResNet-50 and vision transformer demonstrate that images generated by our decoder have features remarkably similar to the user-specified ones and reveal valuable insights into these DNNs' feature spaces.
>
---
#### [new 057] K-Syn: K-space Data Synthesis in Ultra Low-data Regimes
- **分类: cs.CV**

- **简介: 该论文提出K-Syn方法，在频域进行特征级学习，解决动态心脏MRI中k空间数据稀缺问题。通过时序融合策略生成高质量k空间数据，提升低数据条件下的重建鲁棒性。属于医学图像重建任务。**

- **链接: [http://arxiv.org/pdf/2509.06997v1](http://arxiv.org/pdf/2509.06997v1)**

> **作者:** Guan Yu; Zhang Jianhua; Liang Dong; Liu Qiegen
>
> **摘要:** Owing to the inherently dynamic and complex characteristics of cardiac magnetic resonance (CMR) imaging, high-quality and diverse k-space data are rarely available in practice, which in turn hampers robust reconstruction of dynamic cardiac MRI. To address this challenge, we perform feature-level learning directly in the frequency domain and employ a temporal-fusion strategy as the generative guidance to synthesize k-space data. Specifically, leveraging the global representation capacity of the Fourier transform, the frequency domain can be considered a natural global feature space. Therefore, unlike traditional methods that use pixel-level convolution for feature learning and modeling in the image domain, this letter focuses on feature-level modeling in the frequency domain, enabling stable and rich generation even with ultra low-data regimes. Moreover, leveraging the advantages of feature-level modeling in the frequency domain, we integrate k-space data across time frames with multiple fusion strategies to steer and further optimize the generative trajectory. Experimental results demonstrate that the proposed method possesses strong generative ability in low-data regimes, indicating practical potential to alleviate data scarcity in dynamic MRI reconstruction.
>
---
#### [new 058] Data-Efficient Fine-Tuning of Vision-Language Models for Diagnosis of Alzheimer's Disease
- **分类: cs.CV**

- **简介: 该论文属于医学图像诊断任务，旨在解决Med-VLM在AD诊断中数据效率低、利用患者元数据不足的问题。提出数据高效的微调方法，结合合成报告和MMSE分数预测，提升3D MRI的AD诊断效果。**

- **链接: [http://arxiv.org/pdf/2509.07613v1](http://arxiv.org/pdf/2509.07613v1)**

> **作者:** Fangqi Cheng; Surajit Ray; Xiaochen Yang
>
> **摘要:** Medical vision-language models (Med-VLMs) have shown impressive results in tasks such as report generation and visual question answering, but they still face several limitations. Most notably, they underutilize patient metadata and lack integration of clinical diagnostic knowledge. Moreover, most existing models are typically trained from scratch or fine-tuned on large-scale 2D image-text pairs, requiring extensive computational resources, and their effectiveness on 3D medical imaging is often limited due to the absence of structural information. To address these gaps, we propose a data-efficient fine-tuning pipeline to adapt 3D CT-based Med-VLMs for 3D MRI and demonstrate its application in Alzheimer's disease (AD) diagnosis. Our system introduces two key innovations. First, we convert structured metadata into synthetic reports, enriching textual input for improved image-text alignment. Second, we add an auxiliary token trained to predict the mini-mental state examination (MMSE) score, a widely used clinical measure of cognitive function that correlates with AD severity. This provides additional supervision for fine-tuning. Applying lightweight prompt tuning to both image and text modalities, our approach achieves state-of-the-art performance on two AD datasets using 1,500 training images, outperforming existing methods fine-tuned on 10,000 images. Code will be released upon publication.
>
---
#### [new 059] Enhancing Classification of Streaming Data with Image Distillation
- **分类: cs.CV**

- **简介: 该论文属于流数据分类任务，旨在解决资源受限环境下高效准确分类的问题。研究提出基于图像蒸馏的分类方法（DBC），通过提取关键特征提升分类精度，在实验中取得73.1%的准确率，优于传统方法。**

- **链接: [http://arxiv.org/pdf/2509.07049v1](http://arxiv.org/pdf/2509.07049v1)**

> **作者:** Rwad Khatib; Yehudit Aperstein
>
> **备注:** 11 pages
>
> **摘要:** This study tackles the challenge of efficiently classifying streaming data in envi-ronments with limited memory and computational resources. It delves into the application of data distillation as an innovative approach to improve the precision of streaming image data classification. By focusing on distilling essential features from data streams, our method aims to minimize computational demands while preserving crucial information for accurate classification. Our investigation com-pares this approach against traditional algorithms like Hoeffding Trees and Adap-tive Random Forest, adapted through embeddings for image data. The Distillation Based Classification (DBC) demonstrated superior performance, achieving a 73.1% accuracy rate, surpassing both traditional methods and Reservoir Sam-pling Based Classification (RBC) technique. This marks a significant advance-ment in streaming data classification, showcasing the effectiveness of our method in processing complex data streams and setting a new standard for accuracy and efficiency.
>
---
#### [new 060] GLEAM: Learning to Match and Explain in Cross-View Geo-Localization
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究跨视角地理定位任务，解决传统方法缺乏可解释性的问题。提出GLEAM-C模型实现多视角对齐，并引入GLEAM-X任务结合大语言模型进行可解释匹配，构建双语基准推动透明化地理定位。**

- **链接: [http://arxiv.org/pdf/2509.07450v1](http://arxiv.org/pdf/2509.07450v1)**

> **作者:** Xudong Lu; Zhi Zheng; Yi Wan; Yongxiang Yao; Annan Wang; Renrui Zhang; Panwang Xia; Qiong Wu; Qingyun Li; Weifeng Lin; Xiangyu Zhao; Xue Yang; Hongsheng Li
>
> **备注:** 18 pages
>
> **摘要:** Cross-View Geo-Localization (CVGL) focuses on identifying correspondences between images captured from distinct perspectives of the same geographical location. However, existing CVGL approaches are typically restricted to a single view or modality, and their direct visual matching strategy lacks interpretability: they merely predict whether two images correspond, without explaining the rationale behind the match. In this paper, we present GLEAM-C, a foundational CVGL model that unifies multiple views and modalities-including UAV imagery, street maps, panoramic views, and ground photographs-by aligning them exclusively with satellite imagery. Our framework enhances training efficiency through optimized implementation while achieving accuracy comparable to prior modality-specific CVGL models through a two-phase training strategy. Moreover, to address the lack of interpretability in traditional CVGL methods, we leverage the reasoning capabilities of multimodal large language models (MLLMs) to propose a new task, GLEAM-X, which combines cross-view correspondence prediction with explainable reasoning. To support this task, we construct a bilingual benchmark using GPT-4o and Doubao-1.5-Thinking-Vision-Pro to generate training and testing data. The test set is further refined through detailed human revision, enabling systematic evaluation of explainable cross-view reasoning and advancing transparency and scalability in geo-localization. Together, GLEAM-C and GLEAM-X form a comprehensive CVGL pipeline that integrates multi-modal, multi-view alignment with interpretable correspondence analysis, unifying accurate cross-view matching with explainable reasoning and advancing Geo-Localization by enabling models to better Explain And Match. Code and datasets used in this work will be made publicly accessible at https://github.com/Lucky-Lance/GLEAM.
>
---
#### [new 061] Breast Cancer Detection in Thermographic Images via Diffusion-Based Augmentation and Nonlinear Feature Fusion
- **分类: cs.CV; cs.AI**

- **简介: 论文提出一种基于扩散模型增强和非线性特征融合的乳腺癌热成像分类框架，解决医学影像数据稀缺问题，通过生成高质量数据和融合深度与手工特征，实现高准确率诊断。**

- **链接: [http://arxiv.org/pdf/2509.07277v1](http://arxiv.org/pdf/2509.07277v1)**

> **作者:** Sepehr Salem; M. Moein Esfahani; Jingyu Liu; Vince Calhoun
>
> **备注:** Accepted to IEEE-EMBS International Conference on Biomedical and Health Informatics (BHI 2025)
>
> **摘要:** Data scarcity hinders deep learning for medical imaging. We propose a framework for breast cancer classification in thermograms that addresses this using a Diffusion Probabilistic Model (DPM) for data augmentation. Our DPM-based augmentation is shown to be superior to both traditional methods and a ProGAN baseline. The framework fuses deep features from a pre-trained ResNet-50 with handcrafted nonlinear features (e.g., Fractal Dimension) derived from U-Net segmented tumors. An XGBoost classifier trained on these fused features achieves 98.0\% accuracy and 98.1\% sensitivity. Ablation studies and statistical tests confirm that both the DPM augmentation and the nonlinear feature fusion are critical, statistically significant components of this success. This work validates the synergy between advanced generative models and interpretable features for creating highly accurate medical diagnostic tools.
>
---
#### [new 062] D-LEAF: Localizing and Correcting Hallucinations in Multimodal LLMs via Layer-to-head Attention Diagnostics
- **分类: cs.CV**

- **简介: 该论文属于多模态大语言模型（MLLMs）任务，旨在解决生成文本与视觉输入冲突的幻觉问题。提出LIAE和IAF诊断方法，定位错误层与头，并设计D-LEAF动态修正机制，有效抑制幻觉，提升性能。**

- **链接: [http://arxiv.org/pdf/2509.07864v1](http://arxiv.org/pdf/2509.07864v1)**

> **作者:** Tiancheng Yang; Lin Zhang; Jiaye Lin; Guimin Hu; Di Wang; Lijie Hu
>
> **摘要:** Multimodal Large Language Models (MLLMs) achieve strong performance on tasks like image captioning and visual question answering, but remain prone to hallucinations, where generated text conflicts with the visual input. Prior work links this partly to insufficient visual attention, but existing attention-based detectors and mitigation typically apply uniform adjustments across layers and heads, obscuring where errors originate. In this paper, we first show these methods fail to accurately localize problematic layers. Then, we introduce two diagnostics: Layer Image Attention Entropy (LIAE) which flags anomalous layers, and Image Attention Focus (IAF) which scores attention heads within those layers. Analysis shows that LIAE pinpoints faulty layers and IAF reliably ranks heads that warrant correction. Guided by these signals, we propose Dynamic Layer-wise Entropy and Attention Fusion (D-LEAF), a task-agnostic, attention-guided method that dynamically localizes and corrects errors during inference with negligible overhead. Results show our D-LEAF delivers a 53% relative improvement on standard captioning benchmarks, and on VQA both accuracy and F1-score improve by approximately 4%, substantially suppressing hallucinations while preserving efficiency.
>
---
#### [new 063] Universal Few-Shot Spatial Control for Diffusion Models
- **分类: cs.CV**

- **简介: 该论文提出UFC，解决预训练扩散模型在新型空间控制条件下的适应性差和训练成本高的问题。通过少量样本实现任务特定控制，适用于多种扩散模型结构。**

- **链接: [http://arxiv.org/pdf/2509.07530v1](http://arxiv.org/pdf/2509.07530v1)**

> **作者:** Kiet T. Nguyen; Chanhuyk Lee; Donggyun Kim; Dong Hoon Lee; Seunghoon Hong
>
> **摘要:** Spatial conditioning in pretrained text-to-image diffusion models has significantly improved fine-grained control over the structure of generated images. However, existing control adapters exhibit limited adaptability and incur high training costs when encountering novel spatial control conditions that differ substantially from the training tasks. To address this limitation, we propose Universal Few-Shot Control (UFC), a versatile few-shot control adapter capable of generalizing to novel spatial conditions. Given a few image-condition pairs of an unseen task and a query condition, UFC leverages the analogy between query and support conditions to construct task-specific control features, instantiated by a matching mechanism and an update on a small set of task-specific parameters. Experiments on six novel spatial control tasks show that UFC, fine-tuned with only 30 annotated examples of novel tasks, achieves fine-grained control consistent with the spatial conditions. Notably, when fine-tuned with 0.1% of the full training data, UFC achieves competitive performance with the fully supervised baselines in various control tasks. We also show that UFC is applicable agnostically to various diffusion backbones and demonstrate its effectiveness on both UNet and DiT architectures. Code is available at https://github.com/kietngt00/UFC.
>
---
#### [new 064] Reconstruction Alignment Improves Unified Multimodal Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出Reconstruction Alignment（RecA）方法，用于提升统一多模态模型（UMMs）的生成与编辑能力。通过利用视觉编码器嵌入作为密集“文本提示”，在无需字幕的情况下优化模型重建输入图像，从而对齐理解和生成过程，显著提升生成质量与编辑效果。**

- **链接: [http://arxiv.org/pdf/2509.07295v1](http://arxiv.org/pdf/2509.07295v1)**

> **作者:** Ji Xie; Trevor Darrell; Luke Zettlemoyer; XuDong Wang
>
> **备注:** 28 pages, 24 figures and 10 tables
>
> **摘要:** Unified multimodal models (UMMs) unify visual understanding and generation within a single architecture. However, conventional training relies on image-text pairs (or sequences) whose captions are typically sparse and miss fine-grained visual details--even when they use hundreds of words to describe a simple image. We introduce Reconstruction Alignment (RecA), a resource-efficient post-training method that leverages visual understanding encoder embeddings as dense "text prompts," providing rich supervision without captions. Concretely, RecA conditions a UMM on its own visual understanding embeddings and optimizes it to reconstruct the input image with a self-supervised reconstruction loss, thereby realigning understanding and generation. Despite its simplicity, RecA is broadly applicable: across autoregressive, masked-autoregressive, and diffusion-based UMMs, it consistently improves generation and editing fidelity. With only 27 GPU-hours, post-training with RecA substantially improves image generation performance on GenEval (0.73$\rightarrow$0.90) and DPGBench (80.93$\rightarrow$88.15), while also boosting editing benchmarks (ImgEdit 3.38$\rightarrow$3.75, GEdit 6.94$\rightarrow$7.25). Notably, RecA surpasses much larger open-source models and applies broadly across diverse UMM architectures, establishing it as an efficient and general post-training alignment strategy for UMMs
>
---
#### [new 065] Bias in Gender Bias Benchmarks: How Spurious Features Distort Evaluation
- **分类: cs.CV**

- **简介: 该论文研究视觉-语言基础模型中的性别偏见评估问题，发现非性别特征（如物体、背景）的虚假关联会扭曲评估结果。通过扰动这些特征，揭示其对偏见评分的重大影响，并建议结合特征敏感性指标进行更可靠的评估。属于偏见检测与评估任务。**

- **链接: [http://arxiv.org/pdf/2509.07596v1](http://arxiv.org/pdf/2509.07596v1)**

> **作者:** Yusuke Hirota; Ryo Hachiuma; Boyi Li; Ximing Lu; Michael Ross Boone; Boris Ivanovic; Yejin Choi; Marco Pavone; Yu-Chiang Frank Wang; Noa Garcia; Yuta Nakashima; Chao-Han Huck Yang
>
> **备注:** ICCV 2025
>
> **摘要:** Gender bias in vision-language foundation models (VLMs) raises concerns about their safe deployment and is typically evaluated using benchmarks with gender annotations on real-world images. However, as these benchmarks often contain spurious correlations between gender and non-gender features, such as objects and backgrounds, we identify a critical oversight in gender bias evaluation: Do spurious features distort gender bias evaluation? To address this question, we systematically perturb non-gender features across four widely used benchmarks (COCO-gender, FACET, MIAP, and PHASE) and various VLMs to quantify their impact on bias evaluation. Our findings reveal that even minimal perturbations, such as masking just 10% of objects or weakly blurring backgrounds, can dramatically alter bias scores, shifting metrics by up to 175% in generative VLMs and 43% in CLIP variants. This suggests that current bias evaluations often reflect model responses to spurious features rather than gender bias, undermining their reliability. Since creating spurious feature-free benchmarks is fundamentally challenging, we recommend reporting bias metrics alongside feature-sensitivity measurements to enable a more reliable bias assessment.
>
---
#### [new 066] Attention Maps in 3D Shape Classification for Dental Stage Estimation with Class Node Graph Attention Networks
- **分类: cs.CV; cs.AI; 68T07 68T07 68T07 (Primary) 68R10 (Secondary)**

- **简介: 论文提出CGAT网络，用于3D牙齿形状分类任务，解决模型可解释性问题。通过图注意力机制生成可理解的注意力图，提升模型信任度与专家验证效率。**

- **链接: [http://arxiv.org/pdf/2509.07581v1](http://arxiv.org/pdf/2509.07581v1)**

> **作者:** Barkin Buyukcakir; Rocharles Cavalcante Fontenele; Reinhilde Jacobs; Jannick De Tobel; Patrick Thevissen; Dirk Vandermeulen; Peter Claes
>
> **备注:** 25 pages, 8 figures, 2nd International Conference on Explainable AI for Neural or Symbolic Methods
>
> **摘要:** Deep learning offers a promising avenue for automating many recognition tasks in fields such as medicine and forensics. However, the black-box nature of these models hinders their adoption in high-stakes applications where trust and accountability are required. For 3D shape recognition tasks in particular, this paper introduces the Class Node Graph Attention Network (CGAT) architecture to address this need. Applied to 3D meshes of third molars derived from CBCT images, for Demirjian stage allocation, CGAT utilizes graph attention convolutions and an inherent attention mechanism, visualized via attention rollout, to explain its decision-making process. We evaluated the local mean curvature and distance to centroid node features, both individually and in combination, as well as model depth, finding that models incorporating directed edges to a global CLS node produced more intuitive attention maps, while also yielding desirable classification performance. We analyzed the attention-based explanations of the models, and their predictive performances to propose optimal settings for the CGAT. The combination of local mean curvature and distance to centroid as node features yielded a slight performance increase with 0.76 weighted F1 score, and more comprehensive attention visualizations. The CGAT architecture's ability to generate human-understandable attention maps can enhance trust and facilitate expert validation of model decisions. While demonstrated on dental data, CGAT is broadly applicable to graph-based classification and regression tasks, promoting wider adoption of transparent and competitive deep learning models in high-stakes environments.
>
---
#### [new 067] Generating Transferrable Adversarial Examples via Local Mixing and Logits Optimization for Remote Sensing Object Recognition
- **分类: cs.CV; cs.AI**

- **简介: 论文提出一种生成可迁移对抗样本的新方法，用于遥感目标识别。通过局部混合和logits优化提升对抗样本的可迁移性与攻击成功率，解决现有方法语义破坏和梯度消失问题。**

- **链接: [http://arxiv.org/pdf/2509.07495v1](http://arxiv.org/pdf/2509.07495v1)**

> **作者:** Chun Liu; Hailong Wang; Bingqian Zhu; Panpan Ding; Zheng Zheng; Tao Xu; Zhigang Han; Jiayao Wang
>
> **摘要:** Deep Neural Networks (DNNs) are vulnerable to adversarial attacks, posing significant security threats to their deployment in remote sensing applications. Research on adversarial attacks not only reveals model vulnerabilities but also provides critical insights for enhancing robustness. Although current mixing-based strategies have been proposed to increase the transferability of adversarial examples, they either perform global blending or directly exchange a region in the images, which may destroy global semantic features and mislead the optimization of adversarial examples. Furthermore, their reliance on cross-entropy loss for perturbation optimization leads to gradient diminishing during iterative updates, compromising adversarial example quality. To address these limitations, we focus on non-targeted attacks and propose a novel framework via local mixing and logits optimization. First, we present a local mixing strategy to generate diverse yet semantically consistent inputs. Different from MixUp, which globally blends two images, and MixCut, which stitches images together, our method merely blends local regions to preserve global semantic information. Second, we adapt the logit loss from targeted attacks to non-targeted scenarios, mitigating the gradient vanishing problem of cross-entropy loss. Third, a perturbation smoothing loss is applied to suppress high-frequency noise and enhance transferability. Extensive experiments on FGSCR-42 and MTARSI datasets demonstrate superior performance over 12 state-of-the-art methods across 6 surrogate models. Notably, with ResNet as the surrogate on MTARSI, our method achieves a 17.28% average improvement in black-box attack success rate.
>
---
#### [new 068] Not All Splits Are Equal: Rethinking Attribute Generalization Across Unrelated Categories
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究属性预测任务在无关类别间的泛化能力。提出新评估方法，通过不同训练测试划分策略，测试模型对跨概念类别共享属性的识别能力，发现模型性能随类别相关性降低而下降，聚类方法效果最佳。**

- **链接: [http://arxiv.org/pdf/2509.06998v1](http://arxiv.org/pdf/2509.06998v1)**

> **作者:** Liviu Nicolae Fircă; Antonio Bărbălau; Dan Oneata; Elena Burceanu
>
> **摘要:** Can models generalize attribute knowledge across semantically and perceptually dissimilar categories? While prior work has addressed attribute prediction within narrow taxonomic or visually similar domains, it remains unclear whether current models can abstract attributes and apply them to conceptually distant categories. This work presents the first explicit evaluation for the robustness of the attribute prediction task under such conditions, testing whether models can correctly infer shared attributes between unrelated object types: e.g., identifying that the attribute "has four legs" is common to both "dogs" and "chairs". To enable this evaluation, we introduce train-test split strategies that progressively reduce correlation between training and test sets, based on: LLM-driven semantic grouping, embedding similarity thresholding, embedding-based clustering, and supercategory-based partitioning using ground-truth labels. Results show a sharp drop in performance as the correlation between training and test categories decreases, indicating strong sensitivity to split design. Among the evaluated methods, clustering yields the most effective trade-off, reducing hidden correlations while preserving learnability. These findings offer new insights into the limitations of current representations and inform future benchmark construction for attribute reasoning.
>
---
#### [new 069] PanoLAM: Large Avatar Model for Gaussian Full-Head Synthesis from One-shot Unposed Image
- **分类: cs.CV**

- **简介: 该论文提出PanoLAM框架，解决从单张未摆姿势图像快速生成高保真全头高斯模型的问题。通过合成数据训练，采用粗到细生成流程和双分支结构，实现高效重建与渲染。属于3D人脸重建任务。**

- **链接: [http://arxiv.org/pdf/2509.07552v1](http://arxiv.org/pdf/2509.07552v1)**

> **作者:** Peng Li; Yisheng He; Yingdong Hu; Yuan Dong; Weihao Yuan; Yuan Liu; Zilong Dong; Yike Guo
>
> **摘要:** We present a feed-forward framework for Gaussian full-head synthesis from a single unposed image. Unlike previous work that relies on time-consuming GAN inversion and test-time optimization, our framework can reconstruct the Gaussian full-head model given a single unposed image in a single forward pass. This enables fast reconstruction and rendering during inference. To mitigate the lack of large-scale 3D head assets, we propose a large-scale synthetic dataset from trained 3D GANs and train our framework using only synthetic data. For efficient high-fidelity generation, we introduce a coarse-to-fine Gaussian head generation pipeline, where sparse points from the FLAME model interact with the image features by transformer blocks for feature extraction and coarse shape reconstruction, which are then densified for high-fidelity reconstruction. To fully leverage the prior knowledge residing in pretrained 3D GANs for effective reconstruction, we propose a dual-branch framework that effectively aggregates the structured spherical triplane feature and unstructured point-based features for more effective Gaussian head reconstruction. Experimental results show the effectiveness of our framework towards existing work.
>
---
#### [new 070] MEGS$^{2}$: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MEGS²，解决3D高斯溅射在边缘设备上的内存瓶颈问题。通过使用轻量球面高斯和统一剪枝框架，减少内存占用，实现更高效的渲染，保持质量。属于三维重建与渲染优化任务。**

- **链接: [http://arxiv.org/pdf/2509.07021v1](http://arxiv.org/pdf/2509.07021v1)**

> **作者:** Jiarui Chen; Yikeng Chen; Yingshuang Zou; Ye Huang; Peng Wang; Yuan Liu; Yujing Sun; Wenping Wang
>
> **备注:** 14 pages, 4 figures
>
> **摘要:** 3D Gaussian Splatting (3DGS) has emerged as a dominant novel-view synthesis technique, but its high memory consumption severely limits its applicability on edge devices. A growing number of 3DGS compression methods have been proposed to make 3DGS more efficient, yet most only focus on storage compression and fail to address the critical bottleneck of rendering memory. To address this problem, we introduce MEGS$^{2}$, a novel memory-efficient framework that tackles this challenge by jointly optimizing two key factors: the total primitive number and the parameters per primitive, achieving unprecedented memory compression. Specifically, we replace the memory-intensive spherical harmonics with lightweight arbitrarily-oriented spherical Gaussian lobes as our color representations. More importantly, we propose a unified soft pruning framework that models primitive-number and lobe-number pruning as a single constrained optimization problem. Experiments show that MEGS$^{2}$ achieves a 50% static VRAM reduction and a 40% rendering VRAM reduction compared to existing methods, while maintaining comparable rendering quality.
>
---
#### [new 071] TextlessRAG: End-to-End Visual Document RAG by Speech Without Text
- **分类: cs.CV**

- **简介: 该论文提出TextlessRAG，解决语音直接查询视觉文档知识的问题。无需ASR、OCR等模块，实现端到端语音问答，并发布首个双语语音-文档数据集。属于视觉文档检索与问答任务。**

- **链接: [http://arxiv.org/pdf/2509.07538v1](http://arxiv.org/pdf/2509.07538v1)**

> **作者:** Peijin Xie; Shun Qian; Bingquan Liu; Dexin Wang; Lin Sun; Xiangzheng Zhang
>
> **备注:** 5 pages, 4 figures,
>
> **摘要:** Document images encapsulate a wealth of knowledge, while the portability of spoken queries enables broader and flexible application scenarios. Yet, no prior work has explored knowledge base question answering over visual document images with queries provided directly in speech. We propose TextlessRAG, the first end-to-end framework for speech-based question answering over large-scale document images. Unlike prior methods, TextlessRAG eliminates ASR, TTS and OCR, directly interpreting speech, retrieving relevant visual knowledge, and generating answers in a fully textless pipeline. To further boost performance, we integrate a layout-aware reranking mechanism to refine retrieval. Experiments demonstrate substantial improvements in both efficiency and accuracy. To advance research in this direction, we also release the first bilingual speech--document RAG dataset, featuring Chinese and English voice queries paired with multimodal document content. Both the dataset and our pipeline will be made available at repository:https://github.com/xiepeijinhit-hue/textlessrag
>
---
#### [new 072] In the Eye of MLLM: Benchmarking Egocentric Video Intent Understanding with Gaze-Guided Prompting
- **分类: cs.CV**

- **简介: 该论文提出EgoGazeVQA基准，用于评估MLLM在第一视角视频中结合注视信息理解用户意图的能力。任务是解决现有模型忽视注视线索的问题，通过生成并优化基于注视的问答对，提升模型对日常视频中用户意图的理解效果。**

- **链接: [http://arxiv.org/pdf/2509.07447v1](http://arxiv.org/pdf/2509.07447v1)**

> **作者:** Taiying Peng; Jiacheng Hua; Miao Liu; Feng Lu
>
> **摘要:** The emergence of advanced multimodal large language models (MLLMs) has significantly enhanced AI assistants' ability to process complex information across modalities. Recently, egocentric videos, by directly capturing user focus, actions, and context in an unified coordinate, offer an exciting opportunity to enable proactive and personalized AI user experiences with MLLMs. However, existing benchmarks overlook the crucial role of gaze as an indicator of user intent. To address this gap, we introduce EgoGazeVQA, an egocentric gaze-guided video question answering benchmark that leverages gaze information to improve the understanding of longer daily-life videos. EgoGazeVQA consists of gaze-based QA pairs generated by MLLMs and refined by human annotators. Our experiments reveal that existing MLLMs struggle to accurately interpret user intentions. In contrast, our gaze-guided intent prompting methods significantly enhance performance by integrating spatial, temporal, and intent-related cues. We further conduct experiments on gaze-related fine-tuning and analyze how gaze estimation accuracy impacts prompting effectiveness. These results underscore the value of gaze for more personalized and effective AI assistants in egocentric settings.
>
---
#### [new 073] Temporal Image Forensics: A Review and Critical Evaluation
- **分类: cs.CV**

- **简介: 该论文综述了基于时间痕迹的图像年龄估计技术，探讨传感器缺陷等特征，并提出新设置、验证缺陷特性、揭示内容偏差问题，评估神经网络学习能力，以提升时间图像取证的可靠性。属于图像取证任务，解决图像年龄估计与方法可靠性问题。**

- **链接: [http://arxiv.org/pdf/2509.07591v1](http://arxiv.org/pdf/2509.07591v1)**

> **作者:** Robert Jöchl; Andreas Uhl
>
> **摘要:** Temporal image forensics is the science of estimating the age of a digital image. Usually, time-dependent traces (age traces) introduced by the image acquisition pipeline are exploited for this purpose. In this review, a comprehensive overview of the field of temporal image forensics based on time-dependent traces from the image acquisition pipeline is given. This includes a detailed insight into the properties of known age traces (i.e., in-field sensor defects and sensor dust) and temporal image forensics techniques. Another key aspect of this work is to highlight the problem of content bias and to illustrate how important eXplainable Artificial Intelligence methods are to verify the reliability of temporal image forensics techniques. Apart from reviewing material presented in previous works, in this review: (i) a new (probably more realistic) forensic setting is proposed; (ii) the main properties (growth rate and spatial distribution) of in-field sensor defects are verified; (iii) it is shown that a method proposed to utilize in-field sensor defects for image age approximation actually exploits other traces (most likely content bias); (iv) the features learned by a neural network dating palmprint images are further investigated; (v) it is shown how easily a neural network can be distracted from learning age traces. For this purpose, previous work is analyzed, re-implemented if required and experiments are conducted.
>
---
#### [new 074] LINR Bridge: Vector Graphic Animation via Neural Implicits and Video Diffusion Priors
- **分类: cs.CV**

- **简介: 该论文提出LINR Bridge方法，通过神经隐式表示与视频扩散模型生成矢量图形动画，解决传统矢量动画制作耗时且缺乏灵活性的问题，实现高质量、自动化的矢量动画生成。**

- **链接: [http://arxiv.org/pdf/2509.07484v1](http://arxiv.org/pdf/2509.07484v1)**

> **作者:** Wenshuo Gao; Xicheng Lan; Luyao Zhang; Shuai Yang
>
> **备注:** 5 pages, ICIPW 2025, Website: https://gaowenshuo.github.io/LINR-bridge/
>
> **摘要:** Vector graphics, known for their scalability and user-friendliness, provide a unique approach to visual content compared to traditional pixel-based images. Animation of these graphics, driven by the motion of their elements, offers enhanced comprehensibility and controllability but often requires substantial manual effort. To automate this process, we propose a novel method that integrates implicit neural representations with text-to-video diffusion models for vector graphic animation. Our approach employs layered implicit neural representations to reconstruct vector graphics, preserving their inherent properties such as infinite resolution and precise color and shape constraints, which effectively bridges the large domain gap between vector graphics and diffusion models. The neural representations are then optimized using video score distillation sampling, which leverages motion priors from pretrained text-to-video diffusion models. Finally, the vector graphics are warped to match the representations resulting in smooth animation. Experimental results validate the effectiveness of our method in generating vivid and natural vector graphic animations, demonstrating significant improvement over existing techniques that suffer from limitations in flexibility and animation quality.
>
---
#### [new 075] Enhanced SegNet with Integrated Grad-CAM for Interpretable Retinal Layer Segmentation in OCT Images
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出改进的SegNet框架，用于可解释的OCT视网膜层分割。解决传统模型缺乏可解释性问题，通过结构优化、混合损失函数和Grad-CAM可视化提升性能与透明度，实现高精度且临床可信的自动分割。**

- **链接: [http://arxiv.org/pdf/2509.07795v1](http://arxiv.org/pdf/2509.07795v1)**

> **作者:** S M Asiful Islam Saky; Ugyen Tshering
>
> **摘要:** Optical Coherence Tomography (OCT) is essential for diagnosing conditions such as glaucoma, diabetic retinopathy, and age-related macular degeneration. Accurate retinal layer segmentation enables quantitative biomarkers critical for clinical decision-making, but manual segmentation is time-consuming and variable, while conventional deep learning models often lack interpretability. This work proposes an improved SegNet-based deep learning framework for automated and interpretable retinal layer segmentation. Architectural innovations, including modified pooling strategies, enhance feature extraction from noisy OCT images, while a hybrid loss function combining categorical cross-entropy and Dice loss improves performance for thin and imbalanced retinal layers. Gradient-weighted Class Activation Mapping (Grad-CAM) is integrated to provide visual explanations, allowing clinical validation of model decisions. Trained and validated on the Duke OCT dataset, the framework achieved 95.77% validation accuracy, a Dice coefficient of 0.9446, and a Jaccard Index (IoU) of 0.8951. Class-wise results confirmed robust performance across most layers, with challenges remaining for thinner boundaries. Grad-CAM visualizations highlighted anatomically relevant regions, aligning segmentation with clinical biomarkers and improving transparency. By combining architectural improvements, a customized hybrid loss, and explainable AI, this study delivers a high-performing SegNet-based framework that bridges the gap between accuracy and interpretability. The approach offers strong potential for standardizing OCT analysis, enhancing diagnostic efficiency, and fostering clinical trust in AI-driven ophthalmic tools.
>
---
#### [new 076] Evaluation of Machine Learning Reconstruction Techniques for Accelerated Brain MRI Scans
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文评估深度学习算法DeepFoqus-Accelerate在加速脑部MRI扫描中的诊断质量。研究使用公共和临床数据，对比AI重建与标准方法，结果显示AI在缩短扫描时间75%的同时保持高质量图像，支持高效诊疗流程。**

- **链接: [http://arxiv.org/pdf/2509.07193v1](http://arxiv.org/pdf/2509.07193v1)**

> **作者:** Jonathan I. Mandel; Shivaprakash Hiremath; Hedyeh Keshtgar; Timothy Scholl; Sadegh Raeisi
>
> **备注:** This work has been submitted to Radiology: Artificial Intelligence for possible publication
>
> **摘要:** This retrospective-prospective study evaluated whether a deep learning-based MRI reconstruction algorithm can preserve diagnostic quality in brain MRI scans accelerated up to fourfold, using both public and prospective clinical data. The study included 18 healthy volunteers (scans acquired at 3T, January 2024-March 2025), as well as selected fastMRI public datasets with diverse pathologies. Phase-encoding-undersampled 2D/3D T1, T2, and FLAIR sequences were reconstructed with DeepFoqus-Accelerate and compared with standard-of-care (SOC). Three board-certified neuroradiologists and two MRI technologists independently reviewed 36 paired SOC/AI reconstructions from both datasets using a 5-point Likert scale, while quantitative similarity was assessed for 408 scans and 1224 datasets using Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR), and Haar wavelet-based Perceptual Similarity Index (HaarPSI). No AI-reconstructed scan scored below 3 (minimally acceptable), and 95% scored $\geq 4$. Mean SSIM was 0.95 $\pm$ 0.03 (90% cases >0.90), PSNR >41.0 dB, and HaarPSI >0.94. Inter-rater agreement was slight to moderate. Rare artifacts did not affect diagnostic interpretation. These findings demonstrate that DeepFoqus-Accelerate enables robust fourfold brain MRI acceleration with 75% reduced scan time, while preserving diagnostic image quality and supporting improved workflow efficiency.
>
---
#### [new 077] Spectral and Rhythm Feature Performance Evaluation for Category and Class Level Audio Classification with Deep Convolutional Neural Networks
- **分类: cs.SD; cs.AI; cs.CV; cs.LG; eess.AS**

- **简介: 论文研究使用深度卷积神经网络对音频进行分类任务，比较不同频谱和节奏特征在环境声音数据集上的表现，发现梅尔频谱图和MFCC效果最佳。**

- **链接: [http://arxiv.org/pdf/2509.07756v1](http://arxiv.org/pdf/2509.07756v1)**

> **作者:** Friedrich Wolf-Monheim
>
> **摘要:** Next to decision tree and k-nearest neighbours algorithms deep convolutional neural networks (CNNs) are widely used to classify audio data in many domains like music, speech or environmental sounds. To train a specific CNN various spectral and rhythm features like mel-scaled spectrograms, mel-frequency cepstral coefficients (MFCC), cyclic tempograms, short-time Fourier transform (STFT) chromagrams, constant-Q transform (CQT) chromagrams and chroma energy normalized statistics (CENS) chromagrams can be used as digital image input data for the neural network. The performance of these spectral and rhythm features for audio category level as well as audio class level classification is investigated in detail with a deep CNN and the ESC-50 dataset with 2,000 labeled environmental audio recordings using an end-to-end deep learning pipeline. The evaluated metrics accuracy, precision, recall and F1 score for multiclass classification clearly show that the mel-scaled spectrograms and the mel-frequency cepstral coefficients (MFCC) perform significantly better then the other spectral and rhythm features investigated in this research for audio classification tasks using deep CNNs.
>
---
#### [new 078] Benchmarking Vision Transformers and CNNs for Thermal Photovoltaic Fault Detection with Explainable AI Validation
- **分类: cs.LG; cs.CV**

- **简介: 论文比较CNN和ViT在热光伏故障检测中的性能，利用XRAI验证模型与热物理原理的一致性。任务是提升AI在能源监控中的可解释性，解决模型决策缺乏物理依据的问题，实现高准确率且符合物理规律的故障检测。**

- **链接: [http://arxiv.org/pdf/2509.07039v1](http://arxiv.org/pdf/2509.07039v1)**

> **作者:** Serra Aksoy
>
> **备注:** 28 Pages, 4 Figures
>
> **摘要:** Artificial intelligence deployment for automated photovoltaic (PV) monitoring faces interpretability barriers that limit adoption in energy infrastructure applications. While deep learning achieves high accuracy in thermal fault detection, validation that model decisions align with thermal physics principles remains lacking, creating deployment hesitancy where understanding model reasoning is critical. This study provides a systematic comparison of convolutional neural networks (ResNet-18, EfficientNet-B0) and vision transformers (ViT-Tiny, Swin-Tiny) for thermal PV fault detection, using XRAI saliency analysis to assess alignment with thermal physics principles. This represents the first systematic comparison of CNNs and vision transformers for thermal PV fault detection with physics-validated interpretability. Evaluation on 20,000 infrared images spanning normal operation and 11 fault categories shows that Swin Transformer achieves the highest performance (94% binary accuracy; 73% multiclass accuracy) compared to CNN approaches. XRAI analysis reveals that models learn physically meaningful features, such as localized hotspots for cell defects, linear thermal paths for diode failures, and thermal boundaries for vegetation shading, consistent with expected thermal signatures. However, performance varies significantly across fault types: electrical faults achieve strong detection (F1-scores >0.90) while environmental factors like soiling remain challenging (F1-scores 0.20-0.33), indicating limitations imposed by thermal imaging resolution. The thermal physics-guided interpretability approach provides methodology for validating AI decision-making in energy monitoring applications, addressing deployment barriers in renewable energy infrastructure.
>
---
#### [new 079] A smart fridge with AI-enabled food computing
- **分类: eess.SY; cs.CV; cs.SE; cs.SY; C.3; J.7**

- **简介: 论文提出一种基于AI的智能冰箱系统，通过ESP32-CAM实现食品监测与管理。解决多层物品遮挡导致的检测不准问题，采用改进的焦距损失和温度缩放提升模型可靠性，优化家庭食品管理与减少浪费。属于目标检测与物联网应用任务。**

- **链接: [http://arxiv.org/pdf/2509.07400v1](http://arxiv.org/pdf/2509.07400v1)**

> **作者:** Khue Nong Thuc; Khoa Tran Nguyen Anh; Tai Nguyen Huy; Du Nguyen Hao Hong; Khanh Dinh Ba
>
> **摘要:** The Internet of Things (IoT) plays a crucial role in enabling seamless connectivity and intelligent home automation, particularly in food management. By integrating IoT with computer vision, the smart fridge employs an ESP32-CAM to establish a monitoring subsystem that enhances food management efficiency through real-time food detection, inventory tracking, and temperature monitoring. This benefits waste reduction, grocery planning improvement, and household consumption optimization. In high-density inventory conditions, capturing partial or layered images complicates object detection, as overlapping items and occluded views hinder accurate identification and counting. Besides, varied angles and obscured details in multi-layered setups reduce algorithm reliability, often resulting in miscounts or misclassifications. Our proposed system is structured into three core modules: data pre-processing, object detection and management, and a web-based visualization. To address the challenge of poor model calibration caused by overconfident predictions, we implement a variant of focal loss that mitigates over-confidence and under-confidence in multi-category classification. This approach incorporates adaptive, class-wise error calibration via temperature scaling and evaluates the distribution of predicted probabilities across methods. Our results demonstrate that robust functional calibration significantly improves detection reliability under varying lighting conditions and scalability challenges. Further analysis demonstrates a practical, user-focused approach to modern food management, advancing sustainable living goals through reduced waste and more informed consumption.
>
---
#### [new 080] EfficientNet in Digital Twin-based Cardiac Arrest Prediction and Analysis
- **分类: cs.LG; cs.CV**

- **简介: 论文提出结合EfficientNet与数字孪生技术，用于提高心源性猝死的早期预测与分析。任务是提升心脏骤停的识别与管理，通过深度学习提取心血管图像特征，并构建个性化数字孪生模型辅助评估与治疗方案分析。**

- **链接: [http://arxiv.org/pdf/2509.07388v1](http://arxiv.org/pdf/2509.07388v1)**

> **作者:** Qasim Zia; Avais Jan; Zafar Iqbal; Muhammad Mumtaz Ali; Mukarram Ali; Murray Patterson
>
> **摘要:** Cardiac arrest is one of the biggest global health problems, and early identification and management are key to enhancing the patient's prognosis. In this paper, we propose a novel framework that combines an EfficientNet-based deep learning model with a digital twin system to improve the early detection and analysis of cardiac arrest. We use compound scaling and EfficientNet to learn the features of cardiovascular images. In parallel, the digital twin creates a realistic and individualized cardiovascular system model of the patient based on data received from the Internet of Things (IoT) devices attached to the patient, which can help in the constant assessment of the patient and the impact of possible treatment plans. As shown by our experiments, the proposed system is highly accurate in its prediction abilities and, at the same time, efficient. Combining highly advanced techniques such as deep learning and digital twin (DT) technology presents the possibility of using an active and individual approach to predicting cardiac disease.
>
---
#### [new 081] Adversarial Attacks on Audio Deepfake Detection: A Benchmark and Comparative Study
- **分类: cs.SD; cs.AI; cs.CV; cs.LG**

- **简介: 该论文研究音频深度伪造检测方法在对抗攻击下的性能，分析其优缺点，评估五种数据集上的表现，旨在提升检测器的鲁棒性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.07132v1](http://arxiv.org/pdf/2509.07132v1)**

> **作者:** Kutub Uddin; Muhammad Umar Farooq; Awais Khan; Khalid Mahmood Malik
>
> **摘要:** The widespread use of generative AI has shown remarkable success in producing highly realistic deepfakes, posing a serious threat to various voice biometric applications, including speaker verification, voice biometrics, audio conferencing, and criminal investigations. To counteract this, several state-of-the-art (SoTA) audio deepfake detection (ADD) methods have been proposed to identify generative AI signatures to distinguish between real and deepfake audio. However, the effectiveness of these methods is severely undermined by anti-forensic (AF) attacks that conceal generative signatures. These AF attacks span a wide range of techniques, including statistical modifications (e.g., pitch shifting, filtering, noise addition, and quantization) and optimization-based attacks (e.g., FGSM, PGD, C \& W, and DeepFool). In this paper, we investigate the SoTA ADD methods and provide a comparative analysis to highlight their effectiveness in exposing deepfake signatures, as well as their vulnerabilities under adversarial conditions. We conducted an extensive evaluation of ADD methods on five deepfake benchmark datasets using two categories: raw and spectrogram-based approaches. This comparative analysis enables a deeper understanding of the strengths and limitations of SoTA ADD methods against diverse AF attacks. It does not only highlight vulnerabilities of ADD methods, but also informs the design of more robust and generalized detectors for real-world voice biometrics. It will further guide future research in developing adaptive defense strategies that can effectively counter evolving AF techniques.
>
---
#### [new 082] Enhancing Online Learning by Integrating Biosensors and Multimodal Learning Analytics for Detecting and Predicting Student Behavior: A Review
- **分类: cs.HC; cs.AI; cs.CV**

- **简介: 该论文综述了融合生物传感器与多模态学习分析在在线教育中检测与预测学生行为的应用。旨在解决学生行为理解与个性化学习优化问题，总结54项研究，分析方法与趋势，推动自适应学习系统发展。**

- **链接: [http://arxiv.org/pdf/2509.07742v1](http://arxiv.org/pdf/2509.07742v1)**

> **作者:** Alvaro Becerra; Ruth Cobos; Charles Lang
>
> **备注:** Accepted for publication in Behaviour & Information Technology (Taylor & Francis). Final published version will be available soon at https://www.tandfonline.com/journals/tbit20
>
> **摘要:** In modern online learning, understanding and predicting student behavior is crucial for enhancing engagement and optimizing educational outcomes. This systematic review explores the integration of biosensors and Multimodal Learning Analytics (MmLA) to analyze and predict student behavior during computer-based learning sessions. We examine key challenges, including emotion and attention detection, behavioral analysis, experimental design, and demographic considerations in data collection. Our study highlights the growing role of physiological signals, such as heart rate, brain activity, and eye-tracking, combined with traditional interaction data and self-reports to gain deeper insights into cognitive states and engagement levels. We synthesize findings from 54 key studies, analyzing commonly used methodologies such as advanced machine learning algorithms and multimodal data pre-processing techniques. The review identifies current research trends, limitations, and emerging directions in the field, emphasizing the transformative potential of biosensor-driven adaptive learning systems. Our findings suggest that integrating multimodal data can facilitate personalized learning experiences, real-time feedback, and intelligent educational interventions, ultimately advancing toward a more customized and adaptive online learning experience.
>
---
#### [new 083] DepthVision: Robust Vision-Language Understanding through GAN-Based LiDAR-to-RGB Synthesis
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出DepthVision框架，通过GAN生成RGB图像以增强视觉-语言理解，解决机器人在低光或视觉退化环境下的可靠操作问题。利用LiDAR点云合成RGB图像，并结合LAMA模块适应光照条件，提升模型在低光环境下的性能，无需微调下游模型。**

- **链接: [http://arxiv.org/pdf/2509.07463v1](http://arxiv.org/pdf/2509.07463v1)**

> **作者:** Sven Kirchner; Nils Purschke; Ross Greer; Alois C. Knoll
>
> **摘要:** Ensuring reliable robot operation when visual input is degraded or insufficient remains a central challenge in robotics. This letter introduces DepthVision, a framework for multimodal scene understanding designed to address this problem. Unlike existing Vision-Language Models (VLMs), which use only camera-based visual input alongside language, DepthVision synthesizes RGB images from sparse LiDAR point clouds using a conditional generative adversarial network (GAN) with an integrated refiner network. These synthetic views are then combined with real RGB data using a Luminance-Aware Modality Adaptation (LAMA), which blends the two types of data dynamically based on ambient lighting conditions. This approach compensates for sensor degradation, such as darkness or motion blur, without requiring any fine-tuning of downstream vision-language models. We evaluate DepthVision on real and simulated datasets across various models and tasks, with particular attention to safety-critical tasks. The results demonstrate that our approach improves performance in low-light conditions, achieving substantial gains over RGB-only baselines while preserving compatibility with frozen VLMs. This work highlights the potential of LiDAR-guided RGB synthesis for achieving robust robot operation in real-world environments.
>
---
#### [new 084] SVGauge: Towards Human-Aligned Evaluation for SVG Generation
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 该论文提出SVGauge，用于解决SVG生成的评估问题。现有指标无法满足SVG的符号与矢量特性，SVGauge结合视觉保真度与语义一致性，实现更贴合人类判断的评估，提升文本到SVG生成模型的基准测试效果。**

- **链接: [http://arxiv.org/pdf/2509.07127v1](http://arxiv.org/pdf/2509.07127v1)**

> **作者:** Leonardo Zini; Elia Frigieri; Sebastiano Aloscari; Marcello Generali; Lorenzo Dodi; Robert Dosen; Lorenzo Baraldi
>
> **备注:** Accepted at 23rd edition of International Conference on Image Analysis and Processing 2025
>
> **摘要:** Generated Scalable Vector Graphics (SVG) images demand evaluation criteria tuned to their symbolic and vectorial nature: criteria that existing metrics such as FID, LPIPS, or CLIPScore fail to satisfy. In this paper, we introduce SVGauge, the first human-aligned, reference based metric for text-to-SVG generation. SVGauge jointly measures (i) visual fidelity, obtained by extracting SigLIP image embeddings and refining them with PCA and whitening for domain alignment, and (ii) semantic consistency, captured by comparing BLIP-2-generated captions of the SVGs against the original prompts in the combined space of SBERT and TF-IDF. Evaluation on the proposed SHE benchmark shows that SVGauge attains the highest correlation with human judgments and reproduces system-level rankings of eight zero-shot LLM-based generators more faithfully than existing metrics. Our results highlight the necessity of vector-specific evaluation and provide a practical tool for benchmarking future text-to-SVG generation models.
>
---
#### [new 085] Kernel VICReg for Self-Supervised Learning in Reproducing Kernel Hilbert Space
- **分类: stat.ML; cs.CV; cs.LG**

- **简介: 该论文提出Kernel VICReg方法，将自监督学习目标提升到再生核希尔伯特空间，解决欧氏空间中非线性结构捕捉不足的问题。通过核化损失项，实现无需显式映射的非线性特征学习，提升复杂数据任务性能。**

- **链接: [http://arxiv.org/pdf/2509.07289v1](http://arxiv.org/pdf/2509.07289v1)**

> **作者:** M. Hadi Sepanj; Benyamin Ghojogh; Paul Fieguth
>
> **摘要:** Self-supervised learning (SSL) has emerged as a powerful paradigm for representation learning by optimizing geometric objectives--such as invariance to augmentations, variance preservation, and feature decorrelation--without requiring labels. However, most existing methods operate in Euclidean space, limiting their ability to capture nonlinear dependencies and geometric structures. In this work, we propose Kernel VICReg, a novel self-supervised learning framework that lifts the VICReg objective into a Reproducing Kernel Hilbert Space (RKHS). By kernelizing each term of the loss-variance, invariance, and covariance--we obtain a general formulation that operates on double-centered kernel matrices and Hilbert-Schmidt norms, enabling nonlinear feature learning without explicit mappings. We demonstrate that Kernel VICReg not only avoids representational collapse but also improves performance on tasks with complex or small-scale data. Empirical evaluations across MNIST, CIFAR-10, STL-10, TinyImageNet, and ImageNet100 show consistent gains over Euclidean VICReg, with particularly strong improvements on datasets where nonlinear structures are prominent. UMAP visualizations further confirm that kernel-based embeddings exhibit better isometry and class separation. Our results suggest that kernelizing SSL objectives is a promising direction for bridging classical kernel methods with modern representation learning.
>
---
#### [new 086] GCond: Gradient Conflict Resolution via Accumulation-based Stabilization for Large-Scale Multi-Task Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出GCond方法，解决多任务学习中的梯度冲突问题。通过结合梯度积累与自适应仲裁机制，提升计算效率与性能，在多个模型与数据集上验证其有效性与可扩展性。属于多任务学习优化任务。**

- **链接: [http://arxiv.org/pdf/2509.07252v1](http://arxiv.org/pdf/2509.07252v1)**

> **作者:** Evgeny Alves Limarenko; Anastasiia Alexandrovna Studenikina
>
> **备注:** Preprint. Submitted to PeerJ
>
> **摘要:** In multi-task learning (MTL), gradient conflict poses a significant challenge. Effective methods for addressing this problem, including PCGrad, CAGrad, and GradNorm, in their original implementations are computationally demanding, which significantly limits their application in modern large models and transformers. We propose Gradient Conductor (GCond), a method that builds upon PCGrad principles by combining them with gradient accumulation and an adaptive arbitration mechanism. We evaluated GCond on self-supervised learning tasks using MobileNetV3-Small and ConvNeXt architectures on the ImageNet 1K dataset and a combined head and neck CT scan dataset, comparing the proposed method against baseline linear combinations and state-of-the-art gradient conflict resolution methods. The stochastic mode of GCond achieved a two-fold computational speedup while maintaining optimization quality, and demonstrated superior performance across all evaluated metrics, achieving lower L1 and SSIM losses compared to other methods on both datasets. GCond exhibited high scalability, being successfully applied to both compact models (MobileNetV3-Small) and large architectures (ConvNeXt-tiny and ConvNeXt-Base). It also showed compatibility with modern optimizers such as AdamW and Lion/LARS. Therefore, GCond offers a scalable and efficient solution to the problem of gradient conflicts in multi-task learning.
>
---
#### [new 087] Neural Cone Radiosity for Interactive Global Illumination with Glossy Materials
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出神经锥辐射度方法，解决光泽材质全局光照渲染中高频率、视角依赖的辐射分布建模难题。通过反射感知的射线锥编码，提升网络对高频率反射分布的建模能力，实现高效实时高质量渲染。**

- **链接: [http://arxiv.org/pdf/2509.07522v1](http://arxiv.org/pdf/2509.07522v1)**

> **作者:** Jierui Ren; Haojie Jin; Bo Pang; Yisong Chen; Guoping Wang; Sheng Li
>
> **摘要:** Modeling of high-frequency outgoing radiance distributions has long been a key challenge in rendering, particularly for glossy material. Such distributions concentrate radiative energy within a narrow lobe and are highly sensitive to changes in view direction. However, existing neural radiosity methods, which primarily rely on positional feature encoding, exhibit notable limitations in capturing these high-frequency, strongly view-dependent radiance distributions. To address this, we propose a highly-efficient approach by reflectance-aware ray cone encoding based on the neural radiosity framework, named neural cone radiosity. The core idea is to employ a pre-filtered multi-resolution hash grid to accurately approximate the glossy BSDF lobe, embedding view-dependent reflectance characteristics directly into the encoding process through continuous spatial aggregation. Our design not only significantly improves the network's ability to model high-frequency reflection distributions but also effectively handles surfaces with a wide range of glossiness levels, from highly glossy to low-gloss finishes. Meanwhile, our method reduces the network's burden in fitting complex radiance distributions, allowing the overall architecture to remain compact and efficient. Comprehensive experimental results demonstrate that our method consistently produces high-quality, noise-free renderings in real time under various glossiness conditions, and delivers superior fidelity and realism compared to baseline approaches.
>
---
#### [new 088] Understanding Ice Crystal Habit Diversity with Self-Supervised Learning
- **分类: physics.ao-ph; cs.CV**

- **简介: 该论文利用自监督学习分析冰晶形状多样性，解决冰云气候建模难题。通过预训练视觉Transformer学习冰晶形态表征，并用于量化冰晶多样性，提升气候系统中冰晶作用的刻画精度。属于计算机视觉与气候科学交叉任务。**

- **链接: [http://arxiv.org/pdf/2509.07688v1](http://arxiv.org/pdf/2509.07688v1)**

> **作者:** Joseph Ko; Hariprasath Govindarajan; Fredrik Lindsten; Vanessa Przybylo; Kara Sulia; Marcus van Lier-Walqui; Kara Lamb
>
> **摘要:** Ice-containing clouds strongly impact climate, but they are hard to model due to ice crystal habit (i.e., shape) diversity. We use self-supervised learning (SSL) to learn latent representations of crystals from ice crystal imagery. By pre-training a vision transformer with many cloud particle images, we learn robust representations of crystal morphology, which can be used for various science-driven tasks. Our key contributions include (1) validating that our SSL approach can be used to learn meaningful representations, and (2) presenting a relevant application where we quantify ice crystal diversity with these latent representations. Our results demonstrate the power of SSL-driven representations to improve the characterization of ice crystals and subsequently constrain their role in Earth's climate system.
>
---
#### [new 089] Can SSD-Mamba2 Unlock Reinforcement Learning for End-to-End Motion Control?
- **分类: cs.RO; cs.AI; cs.CV; cs.SY; eess.IV; eess.SY**

- **简介: 论文提出基于SSD-Mamba2的跨模态强化学习框架，解决端到端运动控制中感知-动作策略的融合问题。通过高效状态空间模型实现低延迟、长依赖建模，提升控制性能与训练效率。**

- **链接: [http://arxiv.org/pdf/2509.07593v1](http://arxiv.org/pdf/2509.07593v1)**

> **作者:** Gavin Tao; Yinuo Wang; Jinzhao Zhou
>
> **备注:** 4 figures and 6 tables
>
> **摘要:** End-to-end reinforcement learning for motion control promises unified perception-action policies that scale across embodiments and tasks, yet most deployed controllers are either blind (proprioception-only) or rely on fusion backbones with unfavorable compute-memory trade-offs. Recurrent controllers struggle with long-horizon credit assignment, and Transformer-based fusion incurs quadratic cost in token length, limiting temporal and spatial context. We present a vision-driven cross-modal RL framework built on SSD-Mamba2, a selective state-space backbone that applies state-space duality (SSD) to enable both recurrent and convolutional scanning with hardware-aware streaming and near-linear scaling. Proprioceptive states and exteroceptive observations (e.g., depth tokens) are encoded into compact tokens and fused by stacked SSD-Mamba2 layers. The selective state-space updates retain long-range dependencies with markedly lower latency and memory use than quadratic self-attention, enabling longer look-ahead, higher token resolution, and stable training under limited compute. Policies are trained end-to-end under curricula that randomize terrain and appearance and progressively increase scene complexity. A compact, state-centric reward balances task progress, energy efficiency, and safety. Across diverse motion-control scenarios, our approach consistently surpasses strong state-of-the-art baselines in return, safety (collisions and falls), and sample efficiency, while converging faster at the same compute budget. These results suggest that SSD-Mamba2 provides a practical fusion backbone for scalable, foresightful, and efficient end-to-end motion control.
>
---
## 更新

#### [replaced 001] P3-SAM: Native 3D Part Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.06784v2](http://arxiv.org/pdf/2509.06784v2)**

> **作者:** Changfeng Ma; Yang Li; Xinhao Yan; Jiachen Xu; Yunhan Yang; Chunshi Wang; Zibo Zhao; Yanwen Guo; Zhuo Chen; Chunchao Guo
>
> **备注:** Tech Report
>
> **摘要:** Segmenting 3D assets into their constituent parts is crucial for enhancing 3D understanding, facilitating model reuse, and supporting various applications such as part generation. However, current methods face limitations such as poor robustness when dealing with complex objects and cannot fully automate the process. In this paper, we propose a native 3D point-promptable part segmentation model termed P3-SAM, designed to fully automate the segmentation of any 3D objects into components. Inspired by SAM, P3-SAM consists of a feature extractor, multiple segmentation heads, and an IoU predictor, enabling interactive segmentation for users. We also propose an algorithm to automatically select and merge masks predicted by our model for part instance segmentation. Our model is trained on a newly built dataset containing nearly 3.7 million models with reasonable segmentation labels. Comparisons show that our method achieves precise segmentation results and strong robustness on any complex objects, attaining state-of-the-art performance. Our code will be released soon.
>
---
#### [replaced 002] C-DiffDet+: Fusing Global Scene Context with Generative Denoising for High-Fidelity Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.00578v3](http://arxiv.org/pdf/2509.00578v3)**

> **作者:** Abdellah Zakaria Sellam; Ilyes Benaissa; Salah Eddine Bekhouche; Abdenour Hadid; Vito Renó; Cosimo Distante
>
> **摘要:** Fine-grained object detection in challenging visual domains, such as vehicle damage assessment, presents a formidable challenge even for human experts to resolve reliably. While DiffusionDet has advanced the state-of-the-art through conditional denoising diffusion, its performance remains limited by local feature conditioning in context-dependent scenarios. We address this fundamental limitation by introducing Context-Aware Fusion (CAF), which leverages cross-attention mechanisms to integrate global scene context with local proposal features directly. The global context is generated using a separate dedicated encoder that captures comprehensive environmental information, enabling each object proposal to attend to scene-level understanding. Our framework significantly enhances the generative detection paradigm by enabling each object proposal to attend to comprehensive environmental information. Experimental results demonstrate an improvement over state-of-the-art models on the CarDD benchmark, establishing new performance benchmarks for context-aware object detection in fine-grained domains
>
---
#### [replaced 003] IntuiTF: MLLM-Guided Transfer Function Optimization for Direct Volume Rendering
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18407v2](http://arxiv.org/pdf/2506.18407v2)**

> **作者:** Yiyao Wang; Bo Pan; Ke Wang; Han Liu; Jinyuan Mao; Yuxin Liu; Minfeng Zhu; Xiuqi Huang; Weifeng Chen; Bo Zhang; Wei Chen
>
> **摘要:** Direct volume rendering (DVR) is a fundamental technique for visualizing volumetric data, where transfer functions (TFs) play a crucial role in extracting meaningful structures. However, designing effective TFs remains unintuitive due to the semantic gap between user intent and TF parameter space. Although numerous TF optimization methods have been proposed to mitigate this issue, existing approaches still face two major challenges: the vast exploration space and limited generalizability. To address these issues, we propose IntuiTF, a novel framework that leverages Multimodal Large Language Models (MLLMs) to guide TF optimization in alignment with user intent. Specifically, our method consists of two key components: (1) an evolution-driven explorer for effective exploration of the TF space, and (2) an MLLM-guided human-aligned evaluator that provides generalizable visual feedback on rendering quality. The explorer and the evaluator together establish an efficient Trial-Insight-Replanning paradigm for TF space exploration. We further extend our framework with an interactive TF design system. We demonstrate the broad applicability of our framework through three case studies and validate the effectiveness of each component through extensive experiments. We strongly recommend readers check our cases, demo video, and source code at: https://github.com/wyysteelhead/IntuiTF
>
---
#### [replaced 004] Efficient Deep Learning-based Forward Solvers for Brain Tumor Growth Models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.08226v2](http://arxiv.org/pdf/2501.08226v2)**

> **作者:** Zeineb Haouari; Jonas Weidner; Yeray Martin-Ruisanchez; Ivan Ezhov; Aswathi Varma; Daniel Rueckert; Bjoern Menze; Benedikt Wiestler
>
> **摘要:** Glioblastoma, a highly aggressive brain tumor, poses major challenges due to its poor prognosis and high morbidity rates. Partial differential equation-based models offer promising potential to enhance therapeutic outcomes by simulating patient-specific tumor behavior for improved radiotherapy planning. However, model calibration remains a bottleneck due to the high computational demands of optimization methods like Monte Carlo sampling and evolutionary algorithms. To address this, we recently introduced an approach leveraging a neural forward solver with gradient-based optimization to significantly reduce calibration time. This approach requires a highly accurate and fully differentiable forward model. We investigate multiple architectures, including (i) an enhanced TumorSurrogate, (ii) a modified nnU-Net, and (iii) a 3D Vision Transformer (ViT). The nnU-Net achieved the best overall results, excelling in both tumor outline matching and voxel-level prediction of tumor cell concentration. It yielded the lowest MSE in tumor cell concentration compared to ground truth numerical simulation and the highest Dice score across all tumor cell concentration thresholds. Our study demonstrates significant enhancement in forward solver performance and outlines important future research directions.
>
---
#### [replaced 005] Coefficients-Preserving Sampling for Reinforcement Learning with Flow Matching
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.05952v2](http://arxiv.org/pdf/2509.05952v2)**

> **作者:** Feng Wang; Zihao Yu
>
> **备注:** work in progress
>
> **摘要:** Reinforcement Learning (RL) has recently emerged as a powerful technique for improving image and video generation in Diffusion and Flow Matching models, specifically for enhancing output quality and alignment with prompts. A critical step for applying online RL methods on Flow Matching is the introduction of stochasticity into the deterministic framework, commonly realized by Stochastic Differential Equation (SDE). Our investigation reveals a significant drawback to this approach: SDE-based sampling introduces pronounced noise artifacts in the generated images, which we found to be detrimental to the reward learning process. A rigorous theoretical analysis traces the origin of this noise to an excess of stochasticity injected during inference. To address this, we draw inspiration from Denoising Diffusion Implicit Models (DDIM) to reformulate the sampling process. Our proposed method, Coefficients-Preserving Sampling (CPS), eliminates these noise artifacts. This leads to more accurate reward modeling, ultimately enabling faster and more stable convergence for reinforcement learning-based optimizers like Flow-GRPO and Dance-GRPO. Code will be released at https://github.com/IamCreateAI/FlowCPS
>
---
#### [replaced 006] Signal-Based Malware Classification Using 1D CNNs
- **分类: cs.CR; cs.AI; cs.CV; cs.LG; I.2.6; K.6.5**

- **链接: [http://arxiv.org/pdf/2509.06548v2](http://arxiv.org/pdf/2509.06548v2)**

> **作者:** Jack Wilkie; Hanan Hindy; Ivan Andonovic; Christos Tachtatzis; Robert Atkinson
>
> **备注:** Accepted for publication in Springer Cybersecurity (2025)
>
> **摘要:** Malware classification is a contemporary and ongoing challenge in cyber-security: modern obfuscation techniques are able to evade traditional static analysis, while dynamic analysis is too resource intensive to be deployed at a large scale. One prominent line of research addresses these limitations by converting malware binaries into 2D images by heuristically reshaping them into a 2D grid before resizing using Lanczos resampling. These images can then be classified based on their textural information using computer vision approaches. While this approach can detect obfuscated malware more effectively than static analysis, the process of converting files into 2D images results in significant information loss due to both quantisation noise, caused by rounding to integer pixel values, and the introduction of 2D dependencies which do not exist in the original data. This loss of signal limits the classification performance of the downstream model. This work addresses these weaknesses by instead resizing the files into 1D signals which avoids the need for heuristic reshaping, and additionally these signals do not suffer from quantisation noise due to being stored in a floating-point format. It is shown that existing 2D CNN architectures can be readily adapted to classify these 1D signals for improved performance. Furthermore, a bespoke 1D convolutional neural network, based on the ResNet architecture and squeeze-and-excitation layers, was developed to classify these signals and evaluated on the MalNet dataset. It was found to achieve state-of-the-art performance on binary, type, and family level classification with F1 scores of 0.874, 0.503, and 0.507, respectively, paving the way for future models to operate on the proposed signal modality.
>
---
#### [replaced 007] Understanding Museum Exhibits using Vision-Language Reasoning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.01370v2](http://arxiv.org/pdf/2412.01370v2)**

> **作者:** Ada-Astrid Balauca; Sanjana Garai; Stefan Balauca; Rasesh Udayakumar Shetty; Naitik Agrawal; Dhwanil Subhashbhai Shah; Yuqian Fu; Xi Wang; Kristina Toutanova; Danda Pani Paudel; Luc Van Gool
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Museums serve as repositories of cultural heritage and historical artifacts from diverse epochs, civilizations, and regions, preserving well-documented collections that encapsulate vast knowledge, which, when systematically structured into large-scale datasets, can train specialized models. Visitors engage with exhibits through curiosity and questions, making expert domain-specific models essential for interactive query resolution and gaining historical insights. Understanding exhibits from images requires analyzing visual features and linking them to historical knowledge to derive meaningful correlations. We facilitate such reasoning by (a) collecting and curating a large-scale dataset of 65M images and 200M question-answer pairs for exhibits from all around the world; (b) training large vision-language models (VLMs) on the collected dataset; (c) benchmarking their ability on five visual question answering tasks, specifically designed to reflect real-world inquiries and challenges observed in museum settings. The complete dataset is labeled by museum experts, ensuring the quality and the practical significance of the labels. We train two VLMs from different categories: BLIP with vision-language aligned embeddings, but lacking the expressive power of large language models, and the LLaVA model, a powerful instruction-tuned LLM enriched with vision-language reasoning capabilities. Through extensive experiments, we find that while both model types effectively answer visually grounded questions, large vision-language models excel in queries requiring deeper historical context and reasoning. We further demonstrate the necessity of fine-tuning models on large-scale domain-specific datasets by showing that our fine-tuned models significantly outperform current SOTA VLMs in answering questions related to specific attributes, highlighting their limitations in handling complex, nuanced queries.
>
---
#### [replaced 008] Large-scale Pre-training for Grounded Video Caption Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10781v3](http://arxiv.org/pdf/2503.10781v3)**

> **作者:** Evangelos Kazakos; Cordelia Schmid; Josef Sivic
>
> **备注:** Accepted at ICCV 2025. Erratum: An earlier version reported ablations (Table 6 & Fig. 6) with pre-training on a 50k subset of HowToGround1M + fine-tuning on iGround. In the ICCV camera-ready, Table 6 already used the full dataset, but Fig. 6 and a sentence in the text were mistakenly left on 50k. All now use the full HowToGround1M
>
> **摘要:** We propose a novel approach for captioning and object grounding in video, where the objects in the caption are grounded in the video via temporally dense bounding boxes. We introduce the following contributions. First, we present a large-scale automatic annotation method that aggregates frame-level captions grounded with bounding boxes into temporally dense and consistent annotations. We apply this approach on the HowTo100M dataset to construct a large-scale pre-training dataset, named HowToGround1M. We also introduce a Grounded Video Caption Generation model, dubbed GROVE, and pre-train the model on HowToGround1M. Second, we introduce iGround--a dataset of 3513 videos with manually annotated captions and dense spatio-temporally grounded bounding boxes. This allows us to measure progress on this challenging problem, as well as to fine-tune our model on this small-scale but high-quality data. Third, we demonstrate that our approach achieves state-of-the-art results on the proposed iGround dataset, as well as on the VidSTG, ActivityNet-Entities, GroundingYouTube, and YouCook-Interactions datasets. Our ablations demonstrate the importance of pre-training on our automatically annotated HowToGround1M dataset followed by fine-tuning on the manually annotated iGround dataset and validate the key technical contributions of our model. The dataset and code are available at https://ekazakos.github.io/grounded_video_caption_generation/.
>
---
#### [replaced 009] Prompt the Unseen: Evaluating Visual-Language Alignment Beyond Supervision
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.00700v2](http://arxiv.org/pdf/2509.00700v2)**

> **作者:** Raehyuk Jung; Seungjun Yu; Hyunjung Shim
>
> **备注:** Link to publicly available codes is added
>
> **摘要:** Vision-Language Models (VLMs) combine a vision encoder and a large language model (LLM) through alignment training, showing strong performance on multimodal tasks. A central component in this architecture is the projection layer, which maps visual features into the LLM's embedding space. Despite its importance, its ability to generalize to unseen visual concepts has not been systematically evaluated. To address this, we propose a benchmark for evaluating projection-layer generalization. We adapt object detection datasets (rich in fine-grained annotations) into a prompting format and design train/test splits with disjoint label sets, enabling precise control over seen and unseen concept separation. Experimental results show that the projection layer retains about 79 to 88 percent of the performance on unseen classes compared to seen ones across various settings, suggesting a non-trivial level of generalization even without explicit alignment supervision on those concepts. We further analyze this behavior through a mechanistic interpretability lens. Our findings indicate that the feed-forward network in the projection layer functions like a key-value memory, processing seen and unseen tokens in similar ways. This study introduces a new evaluation framework for alignment generalization and highlights the potential for efficient VLM training with limited aligned data.
>
---
#### [replaced 010] PATS: Proficiency-Aware Temporal Sampling for Multi-View Sports Skill Assessment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.04996v3](http://arxiv.org/pdf/2506.04996v3)**

> **作者:** Edoardo Bianchi; Antonio Liotta
>
> **备注:** Accepted at the 2025 4th IEEE International Workshop on Sport Technology and Research
>
> **摘要:** Automated sports skill assessment requires capturing fundamental movement patterns that distinguish expert from novice performance, yet current video sampling methods disrupt the temporal continuity essential for proficiency evaluation. To this end, we introduce Proficiency-Aware Temporal Sampling (PATS), a novel sampling strategy that preserves complete fundamental movements within continuous temporal segments for multi-view skill assessment. PATS adaptively segments videos to ensure each analyzed portion contains full execution of critical performance components, repeating this process across multiple segments to maximize information coverage while maintaining temporal coherence. Evaluated on the EgoExo4D benchmark with SkillFormer, PATS surpasses the state-of-the-art accuracy across all viewing configurations (+0.65% to +3.05%) and delivers substantial gains in challenging domains (+26.22% bouldering, +2.39% music, +1.13% basketball). Systematic analysis reveals that PATS successfully adapts to diverse activity characteristics-from high-frequency sampling for dynamic sports to fine-grained segmentation for sequential skills-demonstrating its effectiveness as an adaptive approach to temporal sampling that advances automated skill assessment for real-world applications.
>
---
#### [replaced 011] "Humor, Art, or Misinformation?": A Multimodal Dataset for Intent-Aware Synthetic Image Detection
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2508.20670v2](http://arxiv.org/pdf/2508.20670v2)**

> **作者:** Anastasios Skoularikis; Stefanos-Iordanis Papadopoulos; Symeon Papadopoulos; Panagiotis C. Petrantonakis
>
> **摘要:** Recent advances in multimodal AI have enabled progress in detecting synthetic and out-of-context content. However, existing efforts largely overlook the intent behind AI-generated images. To fill this gap, we introduce S-HArM, a multimodal dataset for intent-aware classification, comprising 9,576 "in the wild" image-text pairs from Twitter/X and Reddit, labeled as Humor/Satire, Art, or Misinformation. Additionally, we explore three prompting strategies (image-guided, description-guided, and multimodally-guided) to construct a large-scale synthetic training dataset with Stable Diffusion. We conduct an extensive comparative study including modality fusion, contrastive learning, reconstruction networks, attention mechanisms, and large vision-language models. Our results show that models trained on image- and multimodally-guided data generalize better to "in the wild" content, due to preserved visual context. However, overall performance remains limited, highlighting the complexity of inferring intent and the need for specialized architectures.
>
---
#### [replaced 012] BEAM: Bridging Physically-based Rendering and Gaussian Modeling for Relightable Volumetric Video
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.08297v2](http://arxiv.org/pdf/2502.08297v2)**

> **作者:** Yu Hong; Yize Wu; Zhehao Shen; Chengcheng Guo; Yuheng Jiang; Yingliang Zhang; Jingyi Yu; Lan Xu
>
> **摘要:** Volumetric video enables immersive experiences by capturing dynamic 3D scenes, enabling diverse applications for virtual reality, education, and telepresence. However, traditional methods struggle with fixed lighting conditions, while neural approaches face trade-offs in efficiency, quality, or adaptability for relightable scenarios. To address these limitations, we present BEAM, a novel pipeline that bridges 4D Gaussian representations with physically-based rendering (PBR) to produce high-quality, relightable volumetric videos from multi-view RGB footage. BEAM recovers detailed geometry and PBR properties via a series of available Gaussian-based techniques. It first combines Gaussian-based human performance tracking with geometry-aware rasterization in a coarse-to-fine optimization framework to recover spatially and temporally consistent geometries. We further enhance Gaussian attributes by incorporating PBR properties step by step. We generate roughness via a multi-view-conditioned diffusion model, and then derive AO and base color using a 2D-to-3D strategy, incorporating a tailored Gaussian-based ray tracer for efficient visibility computation. Once recovered, these dynamic, relightable assets integrate seamlessly into traditional CG pipelines, supporting real-time rendering with deferred shading and offline rendering with ray tracing. By offering realistic, lifelike visualizations under diverse lighting conditions, BEAM opens new possibilities for interactive entertainment, storytelling, and creative visualization.
>
---
#### [replaced 013] Ultra-Low-Latency Spiking Neural Networks with Temporal-Dependent Integrate-and-Fire Neuron Model for Objects Detection
- **分类: cs.CV; cs.AI; I.4.0; I.2.6**

- **链接: [http://arxiv.org/pdf/2508.20392v2](http://arxiv.org/pdf/2508.20392v2)**

> **作者:** Chengjun Zhang; Yuhao Zhang; Jie Yang; Mohamad Sawan
>
> **备注:** 12 pages, 8 figures
>
> **摘要:** Spiking Neural Networks (SNNs), inspired by the brain, are characterized by minimal power consumption and swift inference capabilities on neuromorphic hardware, and have been widely applied to various visual perception tasks. Current ANN-SNN conversion methods have achieved excellent results in classification tasks with ultra-low time-steps, but their performance in visual detection tasks remains suboptimal. In this paper, we propose a delay-spike approach to mitigate the issue of residual membrane potential caused by heterogeneous spiking patterns. Furthermore, we propose a novel temporal-dependent Integrate-and-Fire (tdIF) neuron architecture for SNNs. This enables Integrate-and-fire (IF) neurons to dynamically adjust their accumulation and firing behaviors based on the temporal order of time-steps. Our method enables spikes to exhibit distinct temporal properties, rather than relying solely on frequency-based representations. Moreover, the tdIF neuron maintains energy consumption on par with traditional IF neuron. We demonstrate that our method achieves more precise feature representation with lower time-steps, enabling high performance and ultra-low latency in visual detection tasks. In this study, we conduct extensive evaluation of the tdIF method across two critical vision tasks: object detection and lane line detection. The results demonstrate that the proposed method surpasses current ANN-SNN conversion approaches, achieving state-of-the-art performance with ultra-low latency (within 5 time-steps).
>
---
#### [replaced 014] Aesthetic Image Captioning with Saliency Enhanced MLLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.04378v3](http://arxiv.org/pdf/2509.04378v3)**

> **作者:** Yilin Tao; Jiashui Huang; Huaze Xu; Ling Shao
>
> **摘要:** Aesthetic Image Captioning (AIC) aims to generate textual descriptions of image aesthetics, becoming a key research direction in the field of computational aesthetics. In recent years, pretrained Multimodal Large Language Models (MLLMs) have advanced rapidly, leading to a significant increase in image aesthetics research that integrates both visual and textual modalities. However, most existing studies on image aesthetics primarily focus on predicting aesthetic ratings and have shown limited application in AIC. Existing AIC works leveraging MLLMs predominantly rely on fine-tuning methods without specifically adapting MLLMs to focus on target aesthetic content. To address this limitation, we propose the Aesthetic Saliency Enhanced Multimodal Large Language Model (ASE-MLLM), an end-to-end framework that explicitly incorporates aesthetic saliency into MLLMs. Within this framework, we introduce the Image Aesthetic Saliency Module (IASM), which efficiently and effectively extracts aesthetic saliency features from images. Additionally, we design IAS-ViT as the image encoder for MLLMs, this module fuses aesthetic saliency features with original image features via a cross-attention mechanism. To the best of our knowledge, ASE-MLLM is the first framework to integrate image aesthetic saliency into MLLMs specifically for AIC tasks. Extensive experiments demonstrated that our approach significantly outperformed traditional methods and generic MLLMs on current mainstream AIC benchmarks, achieving state-of-the-art (SOTA) performance.
>
---
#### [replaced 015] HieraRS: A Hierarchical Segmentation Paradigm for Remote Sensing Enabling Multi-Granularity Interpretation and Cross-Domain Transfer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.08741v2](http://arxiv.org/pdf/2507.08741v2)**

> **作者:** Tianlong Ai; Tianzhu Liu; Haochen Jiang; Yanfeng Gu
>
> **备注:** 17 pages, 11 figures
>
> **摘要:** Hierarchical land cover and land use (LCLU) classification aims to assign pixel-wise labels with multiple levels of semantic granularity to remote sensing (RS) imagery. However, existing deep learning-based methods face two major challenges: 1) They predominantly adopt a flat classification paradigm, which limits their ability to generate end-to-end multi-granularity hierarchical predictions aligned with tree-structured hierarchies used in practice. 2) Most cross-domain studies focus on performance degradation caused by sensor or scene variations, with limited attention to transferring LCLU models to cross-domain tasks with heterogeneous hierarchies (e.g., LCLU to crop classification). These limitations hinder the flexibility and generalization of LCLU models in practical applications. To address these challenges, we propose HieraRS, a novel hierarchical interpretation paradigm that enables multi-granularity predictions and supports the efficient transfer of LCLU models to cross-domain tasks with heterogeneous tree-structured hierarchies. We introduce the Bidirectional Hierarchical Consistency Constraint Mechanism (BHCCM), which can be seamlessly integrated into mainstream flat classification models to generate hierarchical predictions, while improving both semantic consistency and classification accuracy. Furthermore, we present TransLU, a dual-branch cross-domain transfer framework comprising two key components: Cross-Domain Knowledge Sharing (CDKS) and Cross-Domain Semantic Alignment (CDSA). TransLU supports dynamic category expansion and facilitates the effective adaptation of LCLU models to heterogeneous hierarchies. In addition, we construct MM-5B, a large-scale multi-modal hierarchical land use dataset featuring pixel-wise annotations. The code and MM-5B dataset will be released at: https://github.com/AI-Tianlong/HieraRS.
>
---
#### [replaced 016] PINGS: Gaussian Splatting Meets Distance Fields within a Point-Based Implicit Neural Map
- **分类: cs.RO; cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2502.05752v2](http://arxiv.org/pdf/2502.05752v2)**

> **作者:** Yue Pan; Xingguang Zhong; Liren Jin; Louis Wiesmann; Marija Popović; Jens Behley; Cyrill Stachniss
>
> **备注:** 15 pages, 8 figures, presented at RSS 2025
>
> **摘要:** Robots benefit from high-fidelity reconstructions of their environment, which should be geometrically accurate and photorealistic to support downstream tasks. While this can be achieved by building distance fields from range sensors and radiance fields from cameras, realising scalable incremental mapping of both fields consistently and at the same time with high quality is challenging. In this paper, we propose a novel map representation that unifies a continuous signed distance field and a Gaussian splatting radiance field within an elastic and compact point-based implicit neural map. By enforcing geometric consistency between these fields, we achieve mutual improvements by exploiting both modalities. We present a novel LiDAR-visual SLAM system called PINGS using the proposed map representation and evaluate it on several challenging large-scale datasets. Experimental results demonstrate that PINGS can incrementally build globally consistent distance and radiance fields encoded with a compact set of neural points. Compared to state-of-the-art methods, PINGS achieves superior photometric and geometric rendering at novel views by constraining the radiance field with the distance field. Furthermore, by utilizing dense photometric cues and multi-view consistency from the radiance field, PINGS produces more accurate distance fields, leading to improved odometry estimation and mesh reconstruction. We also provide an open-source implementation of PING at: https://github.com/PRBonn/PINGS.
>
---
#### [replaced 017] RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.01907v2](http://arxiv.org/pdf/2509.01907v2)**

> **作者:** Zhenyuan Chen; Chenxi Wang; Ningyu Zhang; Feng Zhang
>
> **备注:** under review
>
> **摘要:** Remote sensing is critical for disaster monitoring, yet existing datasets lack temporal image pairs and detailed textual annotations. While single-snapshot imagery dominates current resources, it fails to capture dynamic disaster impacts over time. To address this gap, we introduce the Remote Sensing Change Caption (RSCC) dataset, a large-scale benchmark comprising 62,315 pre-/post-disaster image pairs (spanning earthquakes, floods, wildfires, and more) paired with rich, human-like change captions. By bridging the temporal and semantic divide in remote sensing data, RSCC enables robust training and evaluation of vision-language models for disaster-aware bi-temporal understanding. Our results highlight RSCC's ability to facilitate detailed disaster-related analysis, paving the way for more accurate, interpretable, and scalable vision-language applications in remote sensing. Code and dataset are available at https://github.com/Bili-Sakura/RSCC.
>
---
#### [replaced 018] Visuospatial Cognitive Assistant
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.12312v4](http://arxiv.org/pdf/2505.12312v4)**

> **作者:** Qi Feng
>
> **备注:** 31 pages, 10 figures, 6 tables
>
> **摘要:** Video-based spatial cognition is vital for robotics and embodied AI but challenges current Vision-Language Models (VLMs). This paper makes two key contributions. First, we introduce ViCA (Visuospatial Cognitive Assistant)-322K, a diverse dataset of 322,003 QA pairs from real-world indoor videos (ARKitScenes, ScanNet, ScanNet++), offering supervision for 3D metadata-grounded queries and video-based complex reasoning. Second, we develop ViCA-7B, fine-tuned on ViCA-322K, which achieves new state-of-the-art on all eight VSI-Bench tasks, outperforming existing models, including larger ones (e.g., +26.1 on Absolute Distance). For interpretability, we present ViCA-Thinking-2.68K, a dataset with explicit reasoning chains, and fine-tune ViCA-7B to create ViCA-7B-Thinking, a model that articulates its spatial reasoning. Our work highlights the importance of targeted data and suggests paths for improved temporal-spatial modeling. We release all resources to foster research in robust visuospatial intelligence.
>
---
#### [replaced 019] Closed-Loop Unsupervised Representation Disentanglement with $β$-VAE Distillation and Diffusion Probabilistic Feedback
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.02346v2](http://arxiv.org/pdf/2402.02346v2)**

> **作者:** Xin Jin; Bohan Li; BAAO Xie; Wenyao Zhang; Jinming Liu; Ziqiang Li; Tao Yang; Wenjun Zeng
>
> **备注:** ECCV 2024
>
> **摘要:** Representation disentanglement may help AI fundamentally understand the real world and thus benefit both discrimination and generation tasks. It currently has at least three unresolved core issues: (i) heavy reliance on label annotation and synthetic data -- causing poor generalization on natural scenarios; (ii) heuristic/hand-craft disentangling constraints make it hard to adaptively achieve an optimal training trade-off; (iii) lacking reasonable evaluation metric, especially for the real label-free data. To address these challenges, we propose a \textbf{C}losed-\textbf{L}oop unsupervised representation \textbf{Dis}entanglement approach dubbed \textbf{CL-Dis}. Specifically, we use diffusion-based autoencoder (Diff-AE) as a backbone while resorting to $\beta$-VAE as a co-pilot to extract semantically disentangled representations. The strong generation ability of diffusion model and the good disentanglement ability of VAE model are complementary. To strengthen disentangling, VAE-latent distillation and diffusion-wise feedback are interconnected in a closed-loop system for a further mutual promotion. Then, a self-supervised \textbf{Navigation} strategy is introduced to identify interpretable semantic directions in the disentangled latent space. Finally, a new metric based on content tracking is designed to evaluate the disentanglement effect. Experiments demonstrate the superiority of CL-Dis on applications like real image manipulation and visual analysis.
>
---
#### [replaced 020] BuzzSet v1.0: A Dataset for Pollinator Detection in Field Conditions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.19762v4](http://arxiv.org/pdf/2508.19762v4)**

> **作者:** Ahmed Emam; Mohamed Elbassiouny; Julius Miller; Patrick Donworth; Sabine Seidel; Ribana Roscher
>
> **备注:** We need to make major revisions to the manuscript, which will take longer than we expected
>
> **摘要:** Pollinator insects such as honeybees and bumblebees are vital to global food production and ecosystem stability, yet their populations are declining due to anthropogenic and environmental stressors. Scalable, automated monitoring in agricultural environments remains an open challenge due to the difficulty of detecting small, fast-moving, and often camouflaged insects. To address this, we present BuzzSet v1.0, a large-scale dataset of high-resolution pollinator images collected under real field conditions. BuzzSet contains 7,856 manually verified images with more than 8,000 annotated instances across three classes: honeybees, bumblebees, and unidentified insects. Initial annotations were produced using a YOLOv12 model trained on external data and refined through human verification with open-source tools. All images were preprocessed into 256 x 256 tiles to improve the detection of small insects. We provide baselines using the RF-DETR transformer-based object detector. The model achieves strong classification accuracy with F1 scores of 0.94 and 0.92 for honeybees and bumblebees, with minimal confusion between these categories. The unidentified class remains more difficult due to label ambiguity and fewer samples, yet still contributes insights for robustness evaluation. Overall detection performance (mAP at 0.50 of 0.559) illustrates the challenging nature of the dataset and its potential to drive advances in small object detection under realistic ecological conditions. Future work focuses on expanding the dataset to version 2.0 with additional annotations and evaluating further detection strategies. BuzzSet establishes a benchmark for ecological computer vision, with the primary challenge being reliable detection of insects frequently camouflaged within natural vegetation, highlighting an open problem for future research.
>
---
#### [replaced 021] VIM-GS: Visual-Inertial Monocular Gaussian Splatting via Object-level Guidance in Large Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.06685v2](http://arxiv.org/pdf/2509.06685v2)**

> **作者:** Shengkai Zhang; Yuhe Liu; Guanjun Wu; Jianhua He; Xinggang Wang; Mozi Chen; Kezhong Liu
>
> **备注:** Withdrawn due to an error in the author list & incomplete experimental results
>
> **摘要:** VIM-GS is a Gaussian Splatting (GS) framework using monocular images for novel-view synthesis (NVS) in large scenes. GS typically requires accurate depth to initiate Gaussian ellipsoids using RGB-D/stereo cameras. Their limited depth sensing range makes it difficult for GS to work in large scenes. Monocular images, however, lack depth to guide the learning and lead to inferior NVS results. Although large foundation models (LFMs) for monocular depth estimation are available, they suffer from cross-frame inconsistency, inaccuracy for distant scenes, and ambiguity in deceptive texture cues. This paper aims to generate dense, accurate depth images from monocular RGB inputs for high-definite GS rendering. The key idea is to leverage the accurate but sparse depth from visual-inertial Structure-from-Motion (SfM) to refine the dense but coarse depth from LFMs. To bridge the sparse input and dense output, we propose an object-segmented depth propagation algorithm that renders the depth of pixels of structured objects. Then we develop a dynamic depth refinement module to handle the crippled SfM depth of dynamic objects and refine the coarse LFM depth. Experiments using public and customized datasets demonstrate the superior rendering quality of VIM-GS in large scenes.
>
---
#### [replaced 022] Conditional Video Generation for High-Efficiency Video Compression
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.15269v3](http://arxiv.org/pdf/2507.15269v3)**

> **作者:** Fangqiu Yi; Jingyu Xu; Jiawei Shao; Chi Zhang; Xuelong Li
>
> **备注:** Critical methodology flaws invalidate key results
>
> **摘要:** Perceptual studies demonstrate that conditional diffusion models excel at reconstructing video content aligned with human visual perception. Building on this insight, we propose a video compression framework that leverages conditional diffusion models for perceptually optimized reconstruction. Specifically, we reframe video compression as a conditional generation task, where a generative model synthesizes video from sparse, yet informative signals. Our approach introduces three key modules: (1) Multi-granular conditioning that captures both static scene structure and dynamic spatio-temporal cues; (2) Compact representations designed for efficient transmission without sacrificing semantic richness; (3) Multi-condition training with modality dropout and role-aware embeddings, which prevent over-reliance on any single modality and enhance robustness. Extensive experiments show that our method significantly outperforms both traditional and neural codecs on perceptual quality metrics such as Fr\'echet Video Distance (FVD) and LPIPS, especially under high compression ratios.
>
---
#### [replaced 023] MIRROR: Multi-Modal Pathological Self-Supervised Representation Learning via Modality Alignment and Retention
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.00374v3](http://arxiv.org/pdf/2503.00374v3)**

> **作者:** Tianyi Wang; Jianan Fan; Dingxin Zhang; Dongnan Liu; Yong Xia; Heng Huang; Weidong Cai
>
> **备注:** 18 pages, 7 figures, 10 tables. Code available at https://github.com/TianyiFranklinWang/MIRROR. Project page: https://tianyifranklinwang.github.io/MIRROR
>
> **摘要:** Histopathology and transcriptomics are fundamental modalities in oncology, encapsulating the morphological and molecular aspects of the disease. Multi-modal self-supervised learning has demonstrated remarkable potential in learning pathological representations by integrating diverse data sources. Conventional multi-modal integration methods primarily emphasize modality alignment, while paying insufficient attention to retaining the modality-specific structures. However, unlike conventional scenarios where multi-modal inputs share highly overlapping features, histopathology and transcriptomics exhibit pronounced heterogeneity, offering orthogonal yet complementary insights. Histopathology provides morphological and spatial context, elucidating tissue architecture and cellular topology, whereas transcriptomics delineates molecular signatures through gene expression patterns. This inherent disparity introduces a major challenge in aligning them while maintaining modality-specific fidelity. To address these challenges, we present MIRROR, a novel multi-modal representation learning method designed to foster both modality alignment and retention. MIRROR employs dedicated encoders to extract comprehensive features for each modality, which is further complemented by a modality alignment module to achieve seamless integration between phenotype patterns and molecular profiles. Furthermore, a modality retention module safeguards unique attributes from each modality, while a style clustering module mitigates redundancy and enhances disease-relevant information by modeling and aligning consistent pathological signatures within a clustering space. Extensive evaluations on TCGA cohorts for cancer subtyping and survival analysis highlight MIRROR's superior performance, demonstrating its effectiveness in constructing comprehensive oncological feature representations and benefiting the cancer diagnosis.
>
---
#### [replaced 024] A Novel Image Similarity Metric for Scene Composition Structure
- **分类: cs.CV; cs.IT; math.IT**

- **链接: [http://arxiv.org/pdf/2508.05037v4](http://arxiv.org/pdf/2508.05037v4)**

> **作者:** Md Redwanul Haque; Manzur Murshed; Manoranjan Paul; Tsz-Kwan Lee
>
> **备注:** 2025 IEEE ICIP (Workshop: Generative AI for World Simulations and Communications). Code at https://github.com/RedwanPlague/scssim
>
> **摘要:** The rapid advancement of generative AI models necessitates novel methods for evaluating image quality that extend beyond human perception. A critical concern for these models is the preservation of an image's underlying Scene Composition Structure (SCS), which defines the geometric relationships among objects and the background, their relative positions, sizes, orientations, etc. Maintaining SCS integrity is paramount for ensuring faithful and structurally accurate GenAI outputs. Traditional image similarity metrics often fall short in assessing SCS. Pixel-level approaches are overly sensitive to minor visual noise, while perception-based metrics prioritize human aesthetic appeal, neither adequately capturing structural fidelity. Furthermore, recent neural-network-based metrics introduce training overheads and potential generalization issues. We introduce the SCS Similarity Index Measure (SCSSIM), a novel, analytical, and training-free metric that quantifies SCS preservation by exploiting statistical measures derived from the Cuboidal hierarchical partitioning of images, robustly capturing non-object-based structural relationships. Our experiments demonstrate SCSSIM's high invariance to non-compositional distortions, accurately reflecting unchanged SCS. Conversely, it shows a strong monotonic decrease for compositional distortions, precisely indicating when SCS has been altered. Compared to existing metrics, SCSSIM exhibits superior properties for structural evaluation, making it an invaluable tool for developing and evaluating generative models, ensuring the integrity of scene composition.
>
---
#### [replaced 025] Frequency Domain Enhanced U-Net for Low-Frequency Information-Rich Image Segmentation in Surgical and Deep-Sea Exploration Robots
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.03829v3](http://arxiv.org/pdf/2502.03829v3)**

> **作者:** Guohao Huo; Ruiting Dai; Jinliang Liu; Ling Shao; Hao Tang
>
> **摘要:** In deep-sea exploration and surgical robotics scenarios, environmental lighting and device resolution limitations often cause high-frequency feature attenuation. Addressing the differences in frequency band sensitivity between CNNs and the human visual system (mid-frequency sensitivity with low-frequency sensitivity surpassing high-frequency), we experimentally quantified the CNN contrast sensitivity function and proposed a wavelet adaptive spectrum fusion (WASF) method inspired by biological vision mechanisms to balance cross-frequency image features. Furthermore, we designed a perception frequency block (PFB) that integrates WASF to enhance frequency-domain feature extraction. Based on this, we developed the FE-UNet model, which employs a SAM2 backbone network and incorporates fine-tuned Hiera-Large modules to ensure segmentation accuracy while improving generalization capability. Experiments demonstrate that FE-UNet achieves state-of-the-art performance in cross-domain tasks such as marine organism segmentation and polyp segmentation, showcasing robust adaptability and significant application potential. The code will be released soon.
>
---
#### [replaced 026] Generalizable Humanoid Manipulation with 3D Diffusion Policies
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.10803v3](http://arxiv.org/pdf/2410.10803v3)**

> **作者:** Yanjie Ze; Zixuan Chen; Wenhao Wang; Tianyi Chen; Xialin He; Ying Yuan; Xue Bin Peng; Jiajun Wu
>
> **备注:** IROS 2025. Project website: https://humanoid-manipulation.github.io
>
> **摘要:** Humanoid robots capable of autonomous operation in diverse environments have long been a goal for roboticists. However, autonomous manipulation by humanoid robots has largely been restricted to one specific scene, primarily due to the difficulty of acquiring generalizable skills and the expensiveness of in-the-wild humanoid robot data. In this work, we build a real-world robotic system to address this challenging problem. Our system is mainly an integration of 1) a whole-upper-body robotic teleoperation system to acquire human-like robot data, 2) a 25-DoF humanoid robot platform with a height-adjustable cart and a 3D LiDAR sensor, and 3) an improved 3D Diffusion Policy learning algorithm for humanoid robots to learn from noisy human data. We run more than 2000 episodes of policy rollouts on the real robot for rigorous policy evaluation. Empowered by this system, we show that using only data collected in one single scene and with only onboard computing, a full-sized humanoid robot can autonomously perform skills in diverse real-world scenarios. Videos are available at https://humanoid-manipulation.github.io .
>
---
#### [replaced 027] SGCNeRF: Few-Shot Neural Rendering via Sparse Geometric Consistency Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2404.00992v3](http://arxiv.org/pdf/2404.00992v3)**

> **作者:** Yuru Xiao; Xianming Liu; Deming Zhai; Kui Jiang; Junjun Jiang; Xiangyang Ji
>
> **备注:** Accepted by IEEE Transactions on Circuits and Systems for Video Technology
>
> **摘要:** Neural Radiance Field (NeRF) technology has made significant strides in creating novel viewpoints. However, its effectiveness is hampered when working with sparsely available views, often leading to performance dips due to overfitting. FreeNeRF attempts to overcome this limitation by integrating implicit geometry regularization, which incrementally improves both geometry and textures. Nonetheless, an initial low positional encoding bandwidth results in the exclusion of high-frequency elements. The quest for a holistic approach that simultaneously addresses overfitting and the preservation of high-frequency details remains ongoing. This study presents a novel feature-matching-based sparse geometry regularization module, enhanced by a spatially consistent geometry filtering mechanism and a frequency-guided geometric regularization strategy. This module excels at accurately identifying high-frequency keypoints, effectively preserving fine structural details. Through progressive refinement of geometry and textures across NeRF iterations, we unveil an effective few-shot neural rendering architecture, designated as SGCNeRF, for enhanced novel view synthesis. Our experiments demonstrate that SGCNeRF not only achieves superior geometry-consistent outcomes but also surpasses FreeNeRF, with improvements of 0.7 dB in PSNR on LLFF and DTU.
>
---
#### [replaced 028] GCRPNet: Graph-Enhanced Contextual and Regional Perception Network for Salient Object Detection in Optical Remote Sensing Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.10542v3](http://arxiv.org/pdf/2508.10542v3)**

> **作者:** Mengyu Ren; Yutong Li; Hua Li; Runmin Cong; Sam Kwong
>
> **摘要:** Salient object detection (SOD) in optical remote sensing images (ORSIs) faces numerous challenges, including significant variations in target scales and low contrast between targets and the background. Existing methods based on vision transformers (ViTs) and convolutional neural networks (CNNs) architectures aim to leverage both global and local features, but the difficulty in effectively integrating these heterogeneous features limits their overall performance. To overcome these limitations, we propose a graph-enhanced contextual and regional perception network (GCRPNet), which builds upon the Mamba architecture to simultaneously capture long-range dependencies and enhance regional feature representation. Specifically, we employ the visual state space (VSS) encoder to extract multi-scale features. To further achieve deep guidance and enhancement of these features, we first design a difference-similarity guided hierarchical graph attention module (DS-HGAM). This module strengthens cross-layer interaction capabilities between features of different scales while enhancing the model's structural perception,allowing it to distinguish between foreground and background more effectively. Then, we design the LEVSS block as the decoder of GCRPNet. This module integrates our proposed adaptive scanning strategy and multi-granularity collaborative attention enhancement module (MCAEM). It performs adaptive patch scanning on feature maps processed via multi-scale convolutions, thereby capturing rich local region information and enhancing Mamba's local modeling capability. Extensive experimental results demonstrate that the proposed model achieves state-of-the-art performance, validating its effectiveness and superiority.
>
---
#### [replaced 029] Comparative Analysis of Lightweight Deep Learning Models for Memory-Constrained Devices
- **分类: cs.CV; cs.AI; 68-XX (Primary) 68Txx, 68T07 (Secondary)**

- **链接: [http://arxiv.org/pdf/2505.03303v2](http://arxiv.org/pdf/2505.03303v2)**

> **作者:** Tasnim Shahriar
>
> **备注:** 22 pages, 10 figures, 4 tables, submitted to Springer - Pattern Recognition and Image Analysis
>
> **摘要:** This paper presents a comprehensive evaluation of lightweight deep learning models for image classification, emphasizing their suitability for deployment in resource-constrained environments such as low-memory devices. Five state-of-the-art architectures - MobileNetV3 Small, ResNet18, SqueezeNet, EfficientNetV2-S, and ShuffleNetV2 - are benchmarked across three diverse datasets: CIFAR-10, CIFAR-100, and Tiny ImageNet. The models are assessed using four key performance metrics: classification accuracy, inference time, floating-point operations (FLOPs), and model size. Additionally, we investigate the impact of hyperparameter tuning, data augmentation, and training paradigms by comparing pretrained models with scratch-trained counterparts, focusing on MobileNetV3 Small. Our findings reveal that transfer learning significantly enhances model accuracy and computational efficiency, particularly for complex datasets like Tiny ImageNet. EfficientNetV2 consistently achieves the highest accuracy, while MobileNetV3 offers the best balance between accuracy and efficiency, and SqueezeNet excels in inference speed and compactness. This study highlights critical trade-offs between accuracy and efficiency, offering actionable insights for deploying lightweight models in real-world applications where computational resources are limited. By addressing these challenges, this research contributes to optimizing deep learning systems for edge computing and mobile platforms.
>
---
#### [replaced 030] RealRep: Generalized SDR-to-HDR Conversion via Attribute-Disentangled Representation Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.07322v2](http://arxiv.org/pdf/2505.07322v2)**

> **作者:** Gang He; Siqi Wang; Kepeng Xu; Lin Zhang; Li Xu; Weiran Wang; Yu-Wing Tai
>
> **摘要:** High-Dynamic-Range Wide-Color-Gamut (HDR-WCG) technology is becoming increasingly widespread, driving a growing need for converting Standard Dynamic Range (SDR) content to HDR. Existing methods primarily rely on fixed tone mapping operators, which struggle to handle the diverse appearances and degradations commonly present in real-world SDR content. To address this limitation, we propose a generalized SDR-to-HDR framework that enhances robustness by learning attribute-disentangled representations. Central to our approach is Realistic Attribute-Disentangled Representation Learning (RealRep), which explicitly disentangles luminance and chrominance components to capture intrinsic content variations across different SDR distributions. Furthermore, we design a Luma-/Chroma-aware negative exemplar generation strategy that constructs degradation-sensitive contrastive pairs, effectively modeling tone discrepancies across SDR styles. Building on these attribute-level priors, we introduce the Degradation-Domain Aware Controlled Mapping Network (DDACMNet), a lightweight, two-stage framework that performs adaptive hierarchical mapping guided by a control-aware normalization mechanism. DDACMNet dynamically modulates the mapping process via degradation-conditioned features, enabling robust adaptation across diverse degradation domains. Extensive experiments demonstrate that RealRep consistently outperforms state-of-the-art methods in both generalization and perceptually faithful HDR color gamut reconstruction.
>
---
#### [replaced 031] Texture- and Shape-based Adversarial Attacks for Overhead Image Vehicle Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.16358v2](http://arxiv.org/pdf/2412.16358v2)**

> **作者:** Mikael Yeghiazaryan; Sai Abhishek Siddhartha Namburu; Emily Kim; Stanislav Panev; Celso de Melo; Fernando De la Torre; Jessica K. Hodgins
>
> **备注:** This version corresponds to the paper accepted for presentation at ICIP 2025
>
> **摘要:** Detecting vehicles in aerial images is difficult due to complex backgrounds, small object sizes, shadows, and occlusions. Although recent deep learning advancements have improved object detection, these models remain susceptible to adversarial attacks (AAs), challenging their reliability. Traditional AA strategies often ignore practical implementation constraints. Our work proposes realistic and practical constraints on texture (lowering resolution, limiting modified areas, and color ranges) and analyzes the impact of shape modifications on attack performance. We conducted extensive experiments with three object detector architectures, demonstrating the performance-practicality trade-off: more practical modifications tend to be less effective, and vice versa. We release both code and data to support reproducibility at https://github.com/humansensinglab/texture-shape-adversarial-attacks.
>
---
#### [replaced 032] F1: A Vision-Language-Action Model Bridging Understanding and Generation to Actions
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.06951v2](http://arxiv.org/pdf/2509.06951v2)**

> **作者:** Qi Lv; Weijie Kong; Hao Li; Jia Zeng; Zherui Qiu; Delin Qu; Haoming Song; Qizhi Chen; Xiang Deng; Jiangmiao Pang
>
> **备注:** Homepage: https://aopolin-lv.github.io/F1-VLA/
>
> **摘要:** Executing language-conditioned tasks in dynamic visual environments remains a central challenge in embodied AI. Existing Vision-Language-Action (VLA) models predominantly adopt reactive state-to-action mappings, often leading to short-sighted behaviors and poor robustness in dynamic scenes. In this paper, we introduce F1, a pretrained VLA framework which integrates the visual foresight generation into decision-making pipeline. F1 adopts a Mixture-of-Transformer architecture with dedicated modules for perception, foresight generation, and control, thereby bridging understanding, generation, and actions. At its core, F1 employs a next-scale prediction mechanism to synthesize goal-conditioned visual foresight as explicit planning targets. By forecasting plausible future visual states, F1 reformulates action generation as a foresight-guided inverse dynamics problem, enabling actions that implicitly achieve visual goals. To endow F1 with robust and generalizable capabilities, we propose a three-stage training recipe on an extensive dataset comprising over 330k trajectories across 136 diverse tasks. This training scheme enhances modular reasoning and equips the model with transferable visual foresight, which is critical for complex and dynamic environments. Extensive evaluations on real-world tasks and simulation benchmarks demonstrate F1 consistently outperforms existing approaches, achieving substantial gains in both task success rate and generalization ability.
>
---
#### [replaced 033] PromptEnhancer: A Simple Approach to Enhance Text-to-Image Models via Chain-of-Thought Prompt Rewriting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.04545v2](http://arxiv.org/pdf/2509.04545v2)**

> **作者:** Linqing Wang; Ximing Xing; Yiji Cheng; Zhiyuan Zhao; Jiale Tao; Qixun Wang; Ruihuang Li; Xin Li; Mingrui Wu; Xinchi Deng; Chunyu Wang; Qinglin Lu
>
> **备注:** technical report
>
> **摘要:** Recent advancements in text-to-image (T2I) diffusion models have demonstrated remarkable capabilities in generating high-fidelity images. However, these models often struggle to faithfully render complex user prompts, particularly in aspects like attribute binding, negation, and compositional relationships. This leads to a significant mismatch between user intent and the generated output. To address this challenge, we introduce PromptEnhancer, a novel and universal prompt rewriting framework that enhances any pretrained T2I model without requiring modifications to its weights. Unlike prior methods that rely on model-specific fine-tuning or implicit reward signals like image-reward scores, our framework decouples the rewriter from the generator. We achieve this by training a Chain-of-Thought (CoT) rewriter through reinforcement learning, guided by a dedicated reward model we term the AlignEvaluator. The AlignEvaluator is trained to provide explicit and fine-grained feedback based on a systematic taxonomy of 24 key points, which are derived from a comprehensive analysis of common T2I failure modes. By optimizing the CoT rewriter to maximize the reward from our AlignEvaluator, our framework learns to generate prompts that are more precisely interpreted by T2I models. Extensive experiments on the HunyuanImage 2.1 model demonstrate that PromptEnhancer significantly improves image-text alignment across a wide range of semantic and compositional challenges. Furthermore, we introduce a new, high-quality human preference benchmark to facilitate future research in this direction.
>
---
#### [replaced 034] BranchGRPO: Stable and Efficient GRPO with Structured Branching in Diffusion Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.06040v2](http://arxiv.org/pdf/2509.06040v2)**

> **作者:** Yuming Li; Yikai Wang; Yuying Zhu; Zhongyu Zhao; Ming Lu; Qi She; Shanghang Zhang
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Recent advancements in aligning image and video generative models via GRPO have achieved remarkable gains in enhancing human preference alignment. However, these methods still face high computational costs from on-policy rollouts and excessive SDE sampling steps, as well as training instability due to sparse rewards. In this paper, we propose BranchGRPO, a novel method that introduces a branch sampling policy updating the SDE sampling process. By sharing computation across common prefixes and pruning low-reward paths and redundant depths, BranchGRPO substantially lowers the per-update compute cost while maintaining or improving exploration diversity. This work makes three main contributions: (1) a branch sampling scheme that reduces rollout and training cost; (2) a tree-based advantage estimator incorporating dense process-level rewards; and (3) pruning strategies exploiting path and depth redundancy to accelerate convergence and boost performance. Experiments on image and video preference alignment show that BranchGRPO improves alignment scores by 16% over strong baselines, while cutting training time by 50%.
>
---
#### [replaced 035] Large Language Models for Crash Detection in Video: A Survey of Methods, Datasets, and Challenges
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.02074v2](http://arxiv.org/pdf/2507.02074v2)**

> **作者:** Sanjeda Akter; Ibne Farabi Shihab; Anuj Sharma
>
> **摘要:** Crash detection from video feeds is a critical problem in intelligent transportation systems. Recent developments in large language models (LLMs) and vision-language models (VLMs) have transformed how we process, reason about, and summarize multimodal information. This paper surveys recent methods leveraging LLMs for crash detection from video data. We present a structured taxonomy of fusion strategies, summarize key datasets, analyze model architectures, compare performance benchmarks, and discuss ongoing challenges and opportunities. Our review provides a foundation for future research in this fast-growing intersection of video understanding and foundation models.
>
---
#### [replaced 036] A multi-task neural network for atypical mitosis recognition under domain shift
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.21035v3](http://arxiv.org/pdf/2508.21035v3)**

> **作者:** Gennaro Percannella; Mattia Sarno; Francesco Tortorella; Mario Vento
>
> **备注:** Approach for MIDOG25 track 2
>
> **摘要:** Recognizing atypical mitotic figures in histopathology images allows physicians to correctly assess tumor aggressiveness. Although machine learning models could be exploited for automatically performing such a task, under domain shift these models suffer from significative performance drops. In this work, an approach based on multi-task learning is proposed for addressing this problem. By exploiting auxiliary tasks, correlated to the main classification task, the proposed approach, submitted to the track 2 of the MItosis DOmain Generalization (MIDOG) challenge, aims to aid the model to focus only on the object to classify, ignoring the domain varying background of the image. The proposed approach shows promising performance in a preliminary evaluation conducted on three distinct datasets, i.e., the MIDOG 2025 Atypical Training Set, the Ami-Br dataset, as well as the preliminary test set of the MIDOG25 challenge.
>
---
#### [replaced 037] Interleaving Reasoning for Better Text-to-Image Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.06945v2](http://arxiv.org/pdf/2509.06945v2)**

> **作者:** Wenxuan Huang; Shuang Chen; Zheyong Xie; Shaosheng Cao; Shixiang Tang; Yufan Shen; Qingyu Yin; Wenbo Hu; Xiaoman Wang; Yuntian Tang; Junbo Qiao; Yue Guo; Yao Hu; Zhenfei Yin; Philip Torr; Yu Cheng; Wanli Ouyang; Shaohui Lin
>
> **摘要:** Unified multimodal understanding and generation models recently have achieve significant improvement in image generation capability, yet a large gap remains in instruction following and detail preservation compared to systems that tightly couple comprehension with generation such as GPT-4o. Motivated by recent advances in interleaving reasoning, we explore whether such reasoning can further improve Text-to-Image (T2I) generation. We introduce Interleaving Reasoning Generation (IRG), a framework that alternates between text-based thinking and image synthesis: the model first produces a text-based thinking to guide an initial image, then reflects on the result to refine fine-grained details, visual quality, and aesthetics while preserving semantics. To train IRG effectively, we propose Interleaving Reasoning Generation Learning (IRGL), which targets two sub-goals: (1) strengthening the initial think-and-generate stage to establish core content and base quality, and (2) enabling high-quality textual reflection and faithful implementation of those refinements in a subsequent image. We curate IRGL-300K, a dataset organized into six decomposed learning modes that jointly cover learning text-based thinking, and full thinking-image trajectories. Starting from a unified foundation model that natively emits interleaved text-image outputs, our two-stage training first builds robust thinking and reflection, then efficiently tunes the IRG pipeline in the full thinking-image trajectory data. Extensive experiments show SoTA performance, yielding absolute gains of 5-10 points on GenEval, WISE, TIIF, GenAI-Bench, and OneIG-EN, alongside substantial improvements in visual quality and fine-grained fidelity. The code, model weights and datasets will be released in: https://github.com/Osilly/Interleaving-Reasoning-Generation .
>
---
#### [replaced 038] $π^3$: Permutation-Equivariant Visual Geometry Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.13347v2](http://arxiv.org/pdf/2507.13347v2)**

> **作者:** Yifan Wang; Jianjun Zhou; Haoyi Zhu; Wenzheng Chang; Yang Zhou; Zizun Li; Junyi Chen; Jiangmiao Pang; Chunhua Shen; Tong He
>
> **备注:** Project page: https://yyfz.github.io/pi3/
>
> **摘要:** We introduce $\pi^3$, a feed-forward neural network that offers a novel approach to visual geometry reconstruction, breaking the reliance on a conventional fixed reference view. Previous methods often anchor their reconstructions to a designated viewpoint, an inductive bias that can lead to instability and failures if the reference is suboptimal. In contrast, $\pi^3$ employs a fully permutation-equivariant architecture to predict affine-invariant camera poses and scale-invariant local point maps without any reference frames. This design not only makes our model inherently robust to input ordering, but also leads to higher accuracy and performance. These advantages enable our simple and bias-free approach to achieve state-of-the-art performance on a wide range of tasks, including camera pose estimation, monocular/video depth estimation, and dense point map reconstruction. Code and models are publicly available.
>
---
#### [replaced 039] A Decade of Wheat Mapping for Lebanon
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.11366v3](http://arxiv.org/pdf/2504.11366v3)**

> **作者:** Hasan Wehbi; Hasan Nasrallah; Mohamad Hasan Zahweh; Zeinab Takach; Veera Ganesh Yalla; Ali J. Ghandour
>
> **摘要:** Wheat accounts for approximately 20% of the world's caloric intake, making it a vital component of global food security. Given this importance, mapping wheat fields plays a crucial role in enabling various stakeholders, including policy makers, researchers, and agricultural organizations, to make informed decisions regarding food security, supply chain management, and resource allocation. In this paper, we tackle the problem of accurately mapping wheat fields out of satellite images by introducing an improved pipeline for winter wheat segmentation, as well as presenting a case study on a decade-long analysis of wheat mapping in Lebanon. We integrate a Temporal Spatial Vision Transformer (TSViT) with Parameter-Efficient Fine Tuning (PEFT) and a novel post-processing pipeline based on the Fields of The World (FTW) framework. Our proposed pipeline addresses key challenges encountered in existing approaches, such as the clustering of small agricultural parcels in a single large field. By merging wheat segmentation with precise field boundary extraction, our method produces geometrically coherent and semantically rich maps that enable us to perform in-depth analysis such as tracking crop rotation pattern over years. Extensive evaluations demonstrate improved boundary delineation and field-level precision, establishing the potential of the proposed framework in operational agricultural monitoring and historical trend analysis. By allowing for accurate mapping of wheat fields, this work lays the foundation for a range of critical studies and future advances, including crop monitoring and yield estimation.
>
---
#### [replaced 040] SkillFormer: Unified Multi-View Video Understanding for Proficiency Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08665v4](http://arxiv.org/pdf/2505.08665v4)**

> **作者:** Edoardo Bianchi; Antonio Liotta
>
> **备注:** Accepted at the 2025 18th International Conference on Machine Vision
>
> **摘要:** Assessing human skill levels in complex activities is a challenging problem with applications in sports, rehabilitation, and training. In this work, we present SkillFormer, a parameter-efficient architecture for unified multi-view proficiency estimation from egocentric and exocentric videos. Building on the TimeSformer backbone, SkillFormer introduces a CrossViewFusion module that fuses view-specific features using multi-head cross-attention, learnable gating, and adaptive self-calibration. We leverage Low-Rank Adaptation to fine-tune only a small subset of parameters, significantly reducing training costs. In fact, when evaluated on the EgoExo4D dataset, SkillFormer achieves state-of-the-art accuracy in multi-view settings while demonstrating remarkable computational efficiency, using 4.5x fewer parameters and requiring 3.75x fewer training epochs than prior baselines. It excels in multiple structured tasks, confirming the value of multi-view integration for fine-grained skill assessment.
>
---
#### [replaced 041] GeoChain: Multimodal Chain-of-Thought for Geographic Reasoning
- **分类: cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.00785v3](http://arxiv.org/pdf/2506.00785v3)**

> **作者:** Sahiti Yerramilli; Nilay Pande; Rynaa Grover; Jayant Sravan Tamarapalli
>
> **摘要:** This paper introduces GeoChain, a large-scale benchmark for evaluating step-by-step geographic reasoning in multimodal large language models (MLLMs). Leveraging 1.46 million Mapillary street-level images, GeoChain pairs each image with a 21-step chain-of-thought (CoT) question sequence (over 30 million Q&A pairs). These sequences guide models from coarse attributes to fine-grained localization across four reasoning categories - visual, spatial, cultural, and precise geolocation - annotated by difficulty. Images are also enriched with semantic segmentation (150 classes) and a visual locatability score. Our benchmarking of contemporary MLLMs (GPT-4.1 variants, Claude 3.7, Gemini 2.5 variants) on a diverse 2,088-image subset reveals consistent challenges: models frequently exhibit weaknesses in visual grounding, display erratic reasoning, and struggle to achieve accurate localization, especially as the reasoning complexity escalates. GeoChain offers a robust diagnostic methodology, critical for fostering significant advancements in complex geographic reasoning within MLLMs.
>
---
#### [replaced 042] InteractPro: A Unified Framework for Motion-Aware Image Composition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.10090v3](http://arxiv.org/pdf/2409.10090v3)**

> **作者:** Weijing Tao; Xiaofeng Yang; Miaomiao Cui; Guosheng Lin
>
> **摘要:** We introduce InteractPro, a comprehensive framework for dynamic motion-aware image composition. At its core is InteractPlan, an intelligent planner that leverages a Large Vision Language Model (LVLM) for scenario analysis and object placement, determining the optimal composition strategy to achieve realistic motion effects. Based on each scenario, InteractPlan selects between our two specialized modules: InteractPhys and InteractMotion. InteractPhys employs an enhanced Material Point Method (MPM)-based simulation to produce physically faithful and controllable object-scene interactions, capturing diverse and abstract events that require true physical modeling. InteractMotion, in contrast, is a training-free method based on pretrained video diffusion. Traditional composition approaches suffer from two major limitations: requiring manual planning for object placement and generating static, motionless outputs. By unifying simulation-based and diffusion-based methods under planner guidance, InteractPro overcomes these challenges, ensuring richly motion-aware compositions. Extensive quantitative and qualitative evaluations demonstrate InteractPro's effectiveness in producing controllable, and coherent compositions across varied scenarios.
>
---
#### [replaced 043] Towards Visuospatial Cognition via Hierarchical Fusion of Visual Experts
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.12363v4](http://arxiv.org/pdf/2505.12363v4)**

> **作者:** Qi Feng
>
> **备注:** 26 pages, 19 figures, 4 tables
>
> **摘要:** While Multimodal Large Language Models (MLLMs) excel at general vision-language tasks, visuospatial cognition - reasoning about spatial layouts, relations, and dynamics - remains a significant challenge. Existing models often lack the necessary architectural components and specialized training data for fine-grained spatial understanding. We introduce ViCA2 (Visuospatial Cognitive Assistant 2), a novel MLLM designed to enhance spatial reasoning. ViCA2 features a dual vision encoder architecture integrating SigLIP for semantics and Hiera for spatial structure, coupled with a token ratio control mechanism for efficiency. We also developed ViCA-322K, a new large-scale dataset with over 322,000 spatially grounded question-answer pairs for targeted instruction tuning. On the challenging VSI-Bench benchmark, our ViCA2-7B model achieves a state-of-the-art average score of 56.8, significantly surpassing larger open-source models (e.g., LLaVA-NeXT-Video-72B, 40.9) and leading proprietary models (Gemini-1.5 Pro, 45.4). This demonstrates the effectiveness of our approach in achieving strong visuospatial intelligence with a compact model. We release ViCA2, its codebase, and the ViCA-322K dataset to facilitate further research.
>
---
#### [replaced 044] GraspCoT: Integrating Physical Property Reasoning for 6-DoF Grasping under Flexible Language Instructions
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.16013v2](http://arxiv.org/pdf/2503.16013v2)**

> **作者:** Xiaomeng Chu; Jiajun Deng; Guoliang You; Wei Liu; Xingchen Li; Jianmin Ji; Yanyong Zhang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Flexible instruction-guided 6-DoF grasping is a significant yet challenging task for real-world robotic systems. Existing methods utilize the contextual understanding capabilities of the large language models (LLMs) to establish mappings between expressions and targets, allowing robots to comprehend users' intentions in the instructions. However, the LLM's knowledge about objects' physical properties remains underexplored despite its tight relevance to grasping. In this work, we propose GraspCoT, a 6-DoF grasp detection framework that integrates a Chain-of-Thought (CoT) reasoning mechanism oriented to physical properties, guided by auxiliary question-answering (QA) tasks. Particularly, we design a set of QA templates to enable hierarchical reasoning that includes three stages: target parsing, physical property analysis, and grasp action selection. Moreover, GraspCoT presents a unified multimodal LLM architecture, which encodes multi-view observations of 3D scenes into 3D-aware visual tokens, and then jointly embeds these visual tokens with CoT-derived textual tokens within LLMs to generate grasp pose predictions. Furthermore, we present IntentGrasp, a large-scale benchmark that fills the gap in public datasets for multi-object grasp detection under diverse and indirect verbal commands. Extensive experiments on IntentGrasp demonstrate the superiority of our method, with additional validation in real-world robotic applications confirming its practicality. The code is available at https://github.com/cxmomo/GraspCoT.
>
---
#### [replaced 045] FilterRAG: Zero-Shot Informed Retrieval-Augmented Generation to Mitigate Hallucinations in VQA
- **分类: cs.CV; cs.CL; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.18536v2](http://arxiv.org/pdf/2502.18536v2)**

> **作者:** Nobin Sarwar
>
> **备注:** 12 pages, 6 figures and 2 tables; Accepted at ICCV 2025 Workshop on Building Foundation Models You Can Trust (T2FM)
>
> **摘要:** Visual Question Answering requires models to generate accurate answers by integrating visual and textual understanding. However, VQA models still struggle with hallucinations, producing convincing but incorrect answers, particularly in knowledge-driven and Out-of-Distribution scenarios. We introduce FilterRAG, a retrieval-augmented framework that combines BLIP-VQA with Retrieval-Augmented Generation to ground answers in external knowledge sources like Wikipedia and DBpedia. FilterRAG achieves 36.5% accuracy on the OK-VQA dataset, demonstrating its effectiveness in reducing hallucinations and improving robustness in both in-domain and Out-of-Distribution settings. These findings highlight the potential of FilterRAG to improve Visual Question Answering systems for real-world deployment.
>
---
#### [replaced 046] Light-Weight Cross-Modal Enhancement Method with Benchmark Construction for UAV-based Open-Vocabulary Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.06011v2](http://arxiv.org/pdf/2509.06011v2)**

> **作者:** Zhenhai Weng; Xinjie Li; Can Wu; Weijie He; Jianfeng Lv; Dong Zhou; Zhongliang Yu
>
> **摘要:** Open-Vocabulary Object Detection (OVD) faces severe performance degradation when applied to UAV imagery due to the domain gap from ground-level datasets. To address this challenge, we propose a complete UAV-oriented solution that combines both dataset construction and model innovation. First, we design a refined UAV-Label Engine, which efficiently resolves annotation redundancy, inconsistency, and ambiguity, enabling the generation of largescale UAV datasets. Based on this engine, we construct two new benchmarks: UAVDE-2M, with over 2.4M instances across 1,800+ categories, and UAVCAP-15K, providing rich image-text pairs for vision-language pretraining. Second, we introduce the Cross-Attention Gated Enhancement (CAGE) module, a lightweight dual-path fusion design that integrates cross-attention, adaptive gating, and global FiLM modulation for robust textvision alignment. By embedding CAGE into the YOLO-World-v2 framework, our method achieves significant gains in both accuracy and efficiency, notably improving zero-shot detection on VisDrone by +5.3 mAP while reducing parameters and GFLOPs, and demonstrating strong cross-domain generalization on SIMD. Extensive experiments and real-world UAV deployment confirm the effectiveness and practicality of our proposed solution for UAV-based OVD
>
---
#### [replaced 047] DIP: Unsupervised Dense In-Context Post-training of Visual Representations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18463v3](http://arxiv.org/pdf/2506.18463v3)**

> **作者:** Sophia Sirko-Galouchenko; Spyros Gidaris; Antonin Vobecky; Andrei Bursuc; Nicolas Thome
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** We introduce DIP, a novel unsupervised post-training method designed to enhance dense image representations in large-scale pretrained vision encoders for in-context scene understanding. Unlike prior approaches that rely on complex self-distillation architectures, our method trains the vision encoder using pseudo-tasks that explicitly simulate downstream in-context scenarios, inspired by meta-learning principles. To enable post-training on unlabeled data, we propose an automatic mechanism for generating in-context tasks that combines a pretrained diffusion model and the vision encoder itself. DIP is simple, unsupervised, and computationally efficient, requiring less than 9 hours on a single A100 GPU. By learning dense representations through pseudo in-context tasks, it achieves strong performance across a wide variety of downstream real-world in-context scene understanding tasks. It outperforms both the initial vision encoder and prior methods, offering a practical and effective solution for improving dense representations. Code available here: https://github.com/sirkosophia/DIP
>
---
#### [replaced 048] Don't Splat your Gaussians: Volumetric Ray-Traced Primitives for Modeling and Rendering Scattering and Emissive Media
- **分类: cs.GR; cs.CV; I.3.2; I.3.3; I.3.6; I.3.5; I.3.7**

- **链接: [http://arxiv.org/pdf/2405.15425v3](http://arxiv.org/pdf/2405.15425v3)**

> **作者:** Jorge Condor; Sebastien Speierer; Lukas Bode; Aljaz Bozic; Simon Green; Piotr Didyk; Adrian Jarabo
>
> **备注:** 17 pages, 17 figures
>
> **摘要:** Efficient scene representations are essential for many computer graphics applications. A general unified representation that can handle both surfaces and volumes simultaneously, remains a research challenge. Inspired by recent methods for scene reconstruction that leverage mixtures of 3D Gaussians to model radiance fields, we formalize and generalize the modeling of scattering and emissive media using mixtures of simple kernel-based volumetric primitives. We introduce closed-form solutions for transmittance and free-flight distance sampling for different kernels, and propose several optimizations to use our method efficiently within any off-the-shelf volumetric path tracer. We demonstrate our method as a compact and efficient alternative to other forms of volume modeling for forward and inverse rendering of scattering media. Furthermore, we adapt and showcase our method in radiance field optimization and rendering, providing additional flexibility compared to current state of the art given its ray-tracing formulation. We also introduce the Epanechnikov kernel and demonstrate its potential as an efficient alternative to the traditionally-used Gaussian kernel in scene reconstruction tasks. The versatility and physically-based nature of our approach allows us to go beyond radiance fields and bring to kernel-based modeling and rendering any path-tracing enabled functionality such as scattering, relighting and complex camera models.
>
---
#### [replaced 049] HueManity: Probing Fine-Grained Visual Perception in MLLMs
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.03194v3](http://arxiv.org/pdf/2506.03194v3)**

> **作者:** Rynaa Grover; Jayant Sravan Tamarapalli; Sahiti Yerramilli; Nilay Pande
>
> **摘要:** Multimodal Large Language Models (MLLMs) excel at high-level visual reasoning, but their performance on nuanced perceptual tasks remains surprisingly limited. We present HueManity, a benchmark designed to assess visual perception in MLLMs. The dataset comprises 83,850 images featuring two-character alphanumeric strings embedded in Ishihara test style dot patterns, challenging models on precise pattern recognition. Our evaluation of nine state-of-the-art MLLMs on HueManity demonstrates a significant performance deficit compared to human and traditional computer vision baselines. The best-performing MLLM achieved a 33.6% accuracy on the numeric `easy' task and a striking 3% on the alphanumeric `hard' task. In contrast, human participants achieved near-perfect scores (100% and 95.6%), and a fine-tuned ResNet50 model reached accuracies of 96.5% and 94.5%. These results highlight a critical gap in the visual capabilities of current MLLMs. Our analysis further explores potential architectural and training-paradigm factors contributing to this perceptual gap in MLLMs. We open-source HueManity dataset and code to foster further research in improving perceptual robustness of MLLMs.
>
---
#### [replaced 050] VMGNet: A Low Computational Complexity Robotic Grasping Network Based on VMamba with Multi-Scale Feature Fusion
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.12520v2](http://arxiv.org/pdf/2411.12520v2)**

> **作者:** Yuhao Jin; Qizhong Gao; Xiaohui Zhu; Yong Yue; Eng Gee Lim; Yuqing Chen; Prudence Wong; Yijie Chu
>
> **备注:** This work is part of ongoing research, and we are further developing new techniques based on these results. To avoid premature disclosure of incomplete content, we request withdrawal of the current version and will resubmit once the study is more complete
>
> **摘要:** While deep learning-based robotic grasping technology has demonstrated strong adaptability, its computational complexity has also significantly increased, making it unsuitable for scenarios with high real-time requirements. Therefore, we propose a low computational complexity and high accuracy model named VMGNet for robotic grasping. For the first time, we introduce the Visual State Space into the robotic grasping field to achieve linear computational complexity, thereby greatly reducing the model's computational cost. Meanwhile, to improve the accuracy of the model, we propose an efficient and lightweight multi-scale feature fusion module, named Fusion Bridge Module, to extract and fuse information at different scales. We also present a new loss function calculation method to enhance the importance differences between subtasks, improving the model's fitting ability. Experiments show that VMGNet has only 8.7G Floating Point Operations and an inference time of 8.1 ms on our devices. VMGNet also achieved state-of-the-art performance on the Cornell and Jacquard public datasets. To validate VMGNet's effectiveness in practical applications, we conducted real grasping experiments in multi-object scenarios, and VMGNet achieved an excellent performance with a 94.4% success rate in real-world grasping tasks. The video for the real-world robotic grasping experiments is available at https://youtu.be/S-QHBtbmLc4.
>
---
#### [replaced 051] Atomizer: Generalizing to new modalities by breaking satellite images down to a set of scalars
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.13542v2](http://arxiv.org/pdf/2506.13542v2)**

> **作者:** Hugo Riffaud de Turckheim; Sylvain Lobry; Roberto Interdonato; Diego Marcos
>
> **摘要:** The growing number of Earth observation satellites has led to increasingly diverse remote sensing data, with varying spatial, spectral, and temporal configurations. Most existing models rely on fixed input formats and modality-specific encoders, which require retraining when new configurations are introduced, limiting their ability to generalize across modalities. We introduce Atomizer, a flexible architecture that represents remote sensing images as sets of scalars, each corresponding to a spectral band value of a pixel. Each scalar is enriched with contextual metadata (acquisition time, spatial resolution, wavelength, and bandwidth), producing an atomic representation that allows a single encoder to process arbitrary modalities without interpolation or resampling. Atomizer uses structured tokenization with Fourier features and non-uniform radial basis functions to encode content and context, and maps tokens into a latent space via cross-attention. Under modality-disjoint evaluations, Atomizer outperforms standard models and demonstrates robust performance across varying resolutions and spatial sizes.
>
---
#### [replaced 052] MaRVL-QA: A Benchmark for Mathematical Reasoning over Visual Landscapes
- **分类: cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.17180v2](http://arxiv.org/pdf/2508.17180v2)**

> **作者:** Nilay Pande; Sahiti Yerramilli; Jayant Sravan Tamarapalli; Rynaa Grover
>
> **摘要:** A key frontier for Multimodal Large Language Models (MLLMs) is the ability to perform deep mathematical and spatial reasoning directly from images, moving beyond their established success in semantic description. Mathematical surface plots provide a rigorous testbed for this capability, as they isolate the task of reasoning from the semantic noise common in natural images. To measure progress on this frontier, we introduce MaRVL-QA (Mathematical Reasoning over Visual Landscapes), a new benchmark designed to quantitatively evaluate these core reasoning skills. The benchmark comprises two novel tasks: Topological Counting, identifying and enumerating features like local maxima; and Transformation Recognition, recognizing applied geometric transformations. Generated from a curated library of functions with rigorous ambiguity filtering, our evaluation on MaRVL-QA reveals that even state-of-the-art MLLMs struggle significantly, often resorting to superficial heuristics instead of robust spatial reasoning. MaRVL-QA provides a challenging new tool for the research community to measure progress, expose model limitations, and guide the development of MLLMs with more profound reasoning abilities.
>
---
#### [replaced 053] LiDARCrafter: Dynamic 4D World Modeling from LiDAR Sequences
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.03692v2](http://arxiv.org/pdf/2508.03692v2)**

> **作者:** Ao Liang; Youquan Liu; Yu Yang; Dongyue Lu; Linfeng Li; Lingdong Kong; Huaici Zhao; Wei Tsang Ooi
>
> **备注:** Preprint; 28 pages, 18 figures, 12 tables; Project Page at https://lidarcrafter.github.io
>
> **摘要:** Generative world models have become essential data engines for autonomous driving, yet most existing efforts focus on videos or occupancy grids, overlooking the unique LiDAR properties. Extending LiDAR generation to dynamic 4D world modeling presents challenges in controllability, temporal coherence, and evaluation standardization. To this end, we present LiDARCrafter, a unified framework for 4D LiDAR generation and editing. Given free-form natural language inputs, we parse instructions into ego-centric scene graphs, which condition a tri-branch diffusion network to generate object structures, motion trajectories, and geometry. These structured conditions enable diverse and fine-grained scene editing. Additionally, an autoregressive module generates temporally coherent 4D LiDAR sequences with smooth transitions. To support standardized evaluation, we establish a comprehensive benchmark with diverse metrics spanning scene-, object-, and sequence-level aspects. Experiments on the nuScenes dataset using this benchmark demonstrate that LiDARCrafter achieves state-of-the-art performance in fidelity, controllability, and temporal consistency across all levels, paving the way for data augmentation and simulation. The code and benchmark are released to the community.
>
---
#### [replaced 054] Self Supervised Networks for Learning Latent Space Representations of Human Body Scans and Motions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.03475v2](http://arxiv.org/pdf/2411.03475v2)**

> **作者:** Emmanuel Hartman; Nicolas Charon; Martin Bauer
>
> **备注:** 15y pages, 11 figures, 4 tables
>
> **摘要:** This paper introduces self-supervised neural network models to tackle several fundamental problems in the field of 3D human body analysis and processing. First, we propose VariShaPE (Varifold Shape Parameter Estimator), a novel architecture for the retrieval of latent space representations of body shapes and poses. This network offers a fast and robust method to estimate the embedding of arbitrary unregistered meshes into the latent space. Second, we complement the estimation of latent codes with MoGeN (Motion Geometry Network) a framework that learns the geometry on the latent space itself. This is achieved by lifting the body pose parameter space into a higher dimensional Euclidean space in which body motion mini-sequences from a training set of 4D data can be approximated by simple linear interpolation. Using the SMPL latent space representation we illustrate how the combination of these network models, once trained, can be used to perform a variety of tasks with very limited computational cost. This includes operations such as motion interpolation, extrapolation and transfer as well as random shape and pose generation.
>
---
#### [replaced 055] CountQA: How Well Do MLLMs Count in the Wild?
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.06585v2](http://arxiv.org/pdf/2508.06585v2)**

> **作者:** Jayant Sravan Tamarapalli; Rynaa Grover; Nilay Pande; Sahiti Yerramilli
>
> **摘要:** Multimodal Large Language Models (MLLMs) demonstrate remarkable fluency in understanding visual scenes, yet they exhibit a critical lack in a fundamental cognitive skill: object counting. This blind spot severely limits their reliability in real-world applications. To date, this capability has been largely unevaluated in complex scenarios, as existing benchmarks either feature sparse object densities or are confined to specific visual domains, failing to test models under realistic conditions. Addressing this gap, we introduce CountQA, a challenging new benchmark designed to probe this deficiency. Comprising over 1,500 question-answer pairs, CountQA features real-world images with high object density, clutter, and occlusion. We investigate this weakness by evaluating 15 prominent MLLMs on the CountQA benchmark and reveal that the top-performing model achieves a mere 42.9% accuracy, with performance declining as object counts rise. By providing a dedicated benchmark to diagnose and rectify this core weakness, CountQA paves the way for a new generation of MLLMs that are not only descriptively fluent but also numerically grounded and spatially aware. We will open-source the dataset and code upon paper acceptance to foster further research.
>
---
#### [replaced 056] OOD-SEG: Exploiting out-of-distribution detection techniques for learning image segmentation from sparse multi-class positive-only annotations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.09553v3](http://arxiv.org/pdf/2411.09553v3)**

> **作者:** Junwen Wang; Zhonghao Wang; Oscar MacCormac; Jonathan Shapey; Tom Vercauteren
>
> **摘要:** Despite significant advancements, segmentation based on deep neural networks in medical and surgical imaging faces several challenges, two of which we aim to address in this work. First, acquiring complete pixel-level segmentation labels for medical images is time-consuming and requires domain expertise. Second, typical segmentation pipelines cannot detect out-of-distribution (OOD) pixels, leaving them prone to spurious outputs during deployment. In this work, we propose a novel segmentation approach which broadly falls within the positive-unlabelled (PU) learning paradigm and exploits tools from OOD detection techniques. Our framework learns only from sparsely annotated pixels from multiple positive-only classes and does not use any annotation for the background class. These multi-class positive annotations naturally fall within the in-distribution (ID) set. Unlabelled pixels may contain positive classes but also negative ones, including what is typically referred to as \emph{background} in standard segmentation formulations. Here, we forgo the need for background annotation and consider these together with any other unseen classes as part of the OOD set. Our framework can integrate, at a pixel-level, any OOD detection approaches designed for classification tasks. To address the lack of existing OOD datasets and established evaluation metric for medical image segmentation, we propose a cross-validation strategy that treats held-out labelled classes as OOD. Extensive experiments on both multi-class hyperspectral and RGB surgical imaging datasets demonstrate the robustness and generalisation capability of our proposed framework.
>
---
#### [replaced 057] Evolving from Unknown to Known: Retentive Angular Representation Learning for Incremental Open Set Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.06570v2](http://arxiv.org/pdf/2509.06570v2)**

> **作者:** Runqing Yang; Yimin Fu; Changyuan Wu; Zhunga Liu
>
> **备注:** 10 pages, 6 figures, 2025 IEEE/CVF International Conference on Computer Vision Workshops
>
> **摘要:** Existing open set recognition (OSR) methods are typically designed for static scenarios, where models aim to classify known classes and identify unknown ones within fixed scopes. This deviates from the expectation that the model should incrementally identify newly emerging unknown classes from continuous data streams and acquire corresponding knowledge. In such evolving scenarios, the discriminability of OSR decision boundaries is hard to maintain due to restricted access to former training data, causing severe inter-class confusion. To solve this problem, we propose retentive angular representation learning (RARL) for incremental open set recognition (IOSR). In RARL, unknown representations are encouraged to align around inactive prototypes within an angular space constructed under the equiangular tight frame, thereby mitigating excessive representation drift during knowledge updates. Specifically, we adopt a virtual-intrinsic interactive (VII) training strategy, which compacts known representations by enforcing clear inter-class margins through boundary-proximal virtual classes. Furthermore, a stratified rectification strategy is designed to refine decision boundaries, mitigating representation bias and feature space distortion caused by imbalances between old/new and positive/negative class samples. We conduct thorough evaluations on CIFAR100 and TinyImageNet datasets and establish a new benchmark for IOSR. Experimental results across various task setups demonstrate that the proposed method achieves state-of-the-art performance.
>
---
#### [replaced 058] Decoupled Sparse Priors Guided Diffusion Compression Model for Point Clouds
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2411.13860v2](http://arxiv.org/pdf/2411.13860v2)**

> **作者:** Xiaoge Zhang; Zijie Wu; Mehwish Nasim; Mingtao Feng; Ajmal Mian
>
> **摘要:** Lossy compression methods rely on an autoencoder to transform a point cloud into latent points for storage, leaving the inherent redundancy of latent representations unexplored. To reduce redundancy in latent points, we propose a sparse priors guided method that achieves high reconstruction quality, especially at high compression ratios. This is accomplished by a dual-density scheme separately processing the latent points (intended for reconstruction) and the decoupled sparse priors (intended for storage). Our approach features an efficient dual-density data flow that relaxes size constraints on latent points, and hybridizes a progressive conditional diffusion model to encapsulate essential details for reconstruction within the conditions, which are decoupled hierarchically to intra-point and inter-point priors. Specifically, our method encodes the original point cloud into latent points and decoupled sparse priors through separate encoders. Latent points serve as intermediates, while sparse priors act as adaptive conditions. We then employ a progressive attention-based conditional denoiser to generate latent points conditioned on the decoupled priors, allowing the denoiser to dynamically attend to geometric and semantic cues from the priors at each encoding and decoding layer. Additionally, we integrate the local distribution into the arithmetic encoder and decoder to enhance local context modeling of the sparse points. The original point cloud is reconstructed through a point decoder. Compared to state-of-the-art, our method obtains superior rate-distortion trade-off, evidenced by extensive evaluations on the ShapeNet dataset and standard test datasets from MPEG group including 8iVFB, and Owlii.
>
---
#### [replaced 059] Audio-centric Video Understanding Benchmark without Text Shortcut
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.19951v2](http://arxiv.org/pdf/2503.19951v2)**

> **作者:** Yudong Yang; Jimin Zhuang; Guangzhi Sun; Changli Tang; Yixuan Li; Peihan Li; Yifan Jiang; Wei Li; Zejun Ma; Chao Zhang
>
> **备注:** Accepted for publication in the Proceedings of EMNLP 2025 (Main Conference)
>
> **摘要:** Audio often serves as an auxiliary modality in video understanding tasks of audio-visual large language models (LLMs), merely assisting in the comprehension of visual information. However, a thorough understanding of videos significantly depends on auditory information, as audio offers critical context, emotional cues, and semantic meaning that visual data alone often lacks. This paper proposes an audio-centric video understanding benchmark (AVUT) to evaluate the video comprehension capabilities of multimodal LLMs with a particular focus on auditory information. AVUT introduces a suite of carefully designed audio-centric tasks, holistically testing the understanding of both audio content and audio-visual interactions in videos. Moreover, this work points out the text shortcut problem that largely exists in other benchmarks where the correct answer can be found from question text alone without needing videos. AVUT addresses this problem by proposing a answer permutation-based filtering mechanism. A thorough evaluation across a diverse range of open-source and proprietary multimodal LLMs is performed, followed by the analyses of deficiencies in audio-visual LLMs. Demos and data are available at https://github.com/lark-png/AVUT.
>
---
#### [replaced 060] SAMba-UNet: SAM2-Mamba UNet for Cardiac MRI in Medical Robotic Perception
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16304v2](http://arxiv.org/pdf/2505.16304v2)**

> **作者:** Guohao Huo; Ruiting Dai; Ling Shao; Hao Tang
>
> **摘要:** To address complex pathological feature extraction in automated cardiac MRI segmentation, we propose SAMba-UNet, a novel dual-encoder architecture that synergistically combines the vision foundation model SAM2, the linear-complexity state-space model Mamba, and the classical UNet to achieve cross-modal collaborative feature learning; to overcome domain shifts between natural images and medical scans, we introduce a Dynamic Feature Fusion Refiner that employs multi-scale pooling and channel-spatial dual-path calibration to strengthen small-lesion and fine-structure representation, and we design a Heterogeneous Omni-Attention Convergence Module (HOACM) that fuses SAM2's local positional semantics with Mamba's long-range dependency modeling via global contextual attention and branch-selective emphasis, yielding substantial gains in both global consistency and boundary precision-on the ACDC cardiac MRI benchmark, SAMba-UNet attains a Dice of 0.9103 and HD95 of 1.0859 mm, notably improving boundary localization for challenging structures like the right ventricle, and its robust, high-fidelity segmentation maps are directly applicable as a perception module within intelligent medical and surgical robotic systems to support preoperative planning, intraoperative navigation, and postoperative complication screening; the code will be open-sourced to facilitate clinical translation and further validation.
>
---
#### [replaced 061] MSCPT: Few-shot Whole Slide Image Classification with Multi-scale and Context-focused Prompt Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.11505v3](http://arxiv.org/pdf/2408.11505v3)**

> **作者:** Minghao Han; Linhao Qu; Dingkang Yang; Xukun Zhang; Xiaoying Wang; Lihua Zhang
>
> **备注:** This work has been submitted to the IEEE TMI for possible publication
>
> **摘要:** Multiple instance learning (MIL) has become a standard paradigm for the weakly supervised classification of whole slide images (WSIs). However, this paradigm relies on using a large number of labeled WSIs for training. The lack of training data and the presence of rare diseases pose significant challenges for these methods. Prompt tuning combined with pre-trained Vision-Language models (VLMs) is an effective solution to the Few-shot Weakly Supervised WSI Classification (FSWC) task. Nevertheless, applying prompt tuning methods designed for natural images to WSIs presents three significant challenges: 1) These methods fail to fully leverage the prior knowledge from the VLM's text modality; 2) They overlook the essential multi-scale and contextual information in WSIs, leading to suboptimal results; and 3) They lack exploration of instance aggregation methods. To address these problems, we propose a Multi-Scale and Context-focused Prompt Tuning (MSCPT) method for FSWC task. Specifically, MSCPT employs the frozen large language model to generate pathological visual language prior knowledge at multiple scales, guiding hierarchical prompt tuning. Additionally, we design a graph prompt tuning module to learn essential contextual information within WSI, and finally, a non-parametric cross-guided instance aggregation module has been introduced to derive the WSI-level features. Extensive experiments, visualizations, and interpretability analyses were conducted on five datasets and three downstream tasks using three VLMs, demonstrating the strong performance of our MSCPT. All codes have been made publicly accessible at https://github.com/Hanminghao/MSCPT.
>
---
#### [replaced 062] POEv2: a flexible and robust framework for generic line segment detection and wireframe line segment detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.19742v2](http://arxiv.org/pdf/2508.19742v2)**

> **作者:** Chenguang Liu; Chisheng Wang; Yuhua Cai; Chuanhua Zhu; Qingquan Li
>
> **摘要:** Line segment detection in images has been studied for several decades. Existing line segment detectors can be roughly divided into two categories: generic line segment detectors and wireframe line segment detectors. Generic line segment detectors aim to detect all meaningful line segments in images and traditional approaches usually fall into this category. Recent deep learning based approaches are mostly wireframe line segment detectors. They detect only line segments that are geometrically meaningful and have large spatial support. Due to the difference in the aim of design, the performance of generic line segment detectors for the task of wireframe line segment detection won't be satisfactory, and vice versa. In this work, we propose a robust framework that can be used for both generic line segment detection and wireframe line segment detection. The proposed method is an improved version of the Pixel Orientation Estimation (POE) method. It is thus named as POEv2. POEv2 detects line segments from edge strength maps, and can be combined with any edge detector. We show in our experiments that by combining the proposed POEv2 with an efficient edge detector, it achieves state-of-the-art performance on three publicly available datasets.
>
---
#### [replaced 063] PIN: A Knowledge-Intensive Dataset for Paired and Interleaved Multimodal Documents
- **分类: cs.AI; cs.CL; cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2406.13923v3](http://arxiv.org/pdf/2406.13923v3)**

> **作者:** Junjie Wang; Yuxiang Zhang; Minghao Liu; Yin Zhang; Yatai Ji; Weihao Xuan; Nie Lin; Kang Zhu; Zhiqiang Lin; Yiming Ren; Chunyang Jiang; Yiyao Yu; Zekun Wang; Tiezhen Wang; Wenhao Huang; Jie Fu; Qunshu Lin; Yujiu Yang; Ge Zhang; Ruibin Yuan; Bei Chen; Wenhu Chen
>
> **备注:** Technical report v1.0
>
> **摘要:** Recent advancements in large multimodal models (LMMs) have leveraged extensive multimodal datasets to enhance capabilities in complex knowledge-driven tasks. However, persistent challenges in perceptual and reasoning errors limit their efficacy, particularly in interpreting intricate visual data and deducing multimodal relationships. To address these issues, we introduce PIN (Paired and INterleaved multimodal documents), a novel data format designed to foster a deeper integration of visual and textual knowledge. The PIN format uniquely combines semantically rich Markdown files, which preserve fine-grained textual structures, with holistic overall images that capture the complete document layout. Following this format, we construct and release two large-scale, open-source datasets: PIN-200M (~200 million documents) and PIN-14M (~14 million), compiled from diverse web and scientific sources in both English and Chinese. To maximize usability, we provide detailed statistical analyses and equip the datasets with quality signals, enabling researchers to easily filter and select data for specific tasks. Our work provides the community with a versatile data format and substantial resources, offering a foundation for new research in pre-training strategies and the development of more powerful knowledge-intensive LMMs.
>
---
#### [replaced 064] DMS-Net:Dual-Modal Multi-Scale Siamese Network for Binocular Fundus Image Classification
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.18046v3](http://arxiv.org/pdf/2504.18046v3)**

> **作者:** Guohao Huo; Zibo Lin; Zitong Wang; Ruiting Dai; Hao Tang
>
> **摘要:** Ophthalmic diseases pose a significant global health burden. However, traditional diagnostic methods and existing monocular image-based deep learning approaches often overlook the pathological correlations between the two eyes. In practical medical robotic diagnostic scenarios, paired retinal images (binocular fundus images) are frequently required as diagnostic evidence. To address this, we propose DMS-Net-a dual-modal multi-scale siamese network for binocular retinal image classification. The framework employs a weight-sharing siamese ResNet-152 architecture to concurrently extract deep semantic features from bilateral fundus images. To tackle challenges like indistinct lesion boundaries and diffuse pathological distributions, we introduce the OmniPool Spatial Integrator Module (OSIM), which achieves multi-resolution feature aggregation through multi-scale adaptive pooling and spatial attention mechanisms. Furthermore, the Calibrated Analogous Semantic Fusion Module (CASFM) leverages spatial-semantic recalibration and bidirectional attention mechanisms to enhance cross-modal interaction, aggregating modality-agnostic representations of fundus structures. To fully exploit the differential semantic information of lesions present in bilateral fundus features, we introduce the Cross-Modal Contrastive Alignment Module (CCAM). Additionally, to enhance the aggregation of lesion-correlated semantic information, we introduce the Cross-Modal Integrative Alignment Module (CIAM). Evaluation on the ODIR-5K dataset demonstrates that DMS-Net achieves state-of-the-art performance with an accuracy of 82.9%, recall of 84.5%, and a Cohen's kappa coefficient of 83.2%, showcasing robust capacity in detecting symmetrical pathologies and improving clinical decision-making for ocular diseases. Code and the processed dataset will be released subsequently.
>
---
#### [replaced 065] Grounding DINO-US-SAM: Text-Prompted Multi-Organ Segmentation in Ultrasound with LoRA-Tuned Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.23903v3](http://arxiv.org/pdf/2506.23903v3)**

> **作者:** Hamza Rasaee; Taha Koleilat; Hassan Rivaz
>
> **备注:** 11 pages, 3 figures, 7 tables
>
> **摘要:** Accurate and generalizable object segmentation in ultrasound imaging remains a significant challenge due to anatomical variability, diverse imaging protocols, and limited annotated data. In this study, we propose a prompt-driven vision-language model (VLM) that integrates Grounding DINO with SAM2 (Segment Anything Model2) to enable object segmentation across multiple ultrasound organs. A total of 18 public ultrasound datasets, encompassing the breast, thyroid, liver, prostate, kidney, and paraspinal muscle, were utilized. These datasets were divided into 15 for fine-tuning and validation of Grounding DINO using Low Rank Adaptation (LoRA) to the ultrasound domain, and 3 were held out entirely for testing to evaluate performance in unseen distributions. Comprehensive experiments demonstrate that our approach outperforms state-of-the-art segmentation methods, including UniverSeg, MedSAM, MedCLIP-SAM, BiomedParse, and SAMUS on most seen datasets while maintaining strong performance on unseen datasets without additional fine-tuning. These results underscore the promise of VLMs in scalable and robust ultrasound image analysis, reducing dependence on large, organ-specific annotated datasets. We will publish our code on code.sonography.ai after acceptance.
>
---
#### [replaced 066] Missing Fine Details in Images: Last Seen in High Frequencies
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.05441v2](http://arxiv.org/pdf/2509.05441v2)**

> **作者:** Tejaswini Medi; Hsien-Yi Wang; Arianna Rampini; Margret Keuper
>
> **摘要:** Latent generative models have shown remarkable progress in high-fidelity image synthesis, typically using a two-stage training process that involves compressing images into latent embeddings via learned tokenizers in the first stage. The quality of generation strongly depends on how expressive and well-optimized these latent embeddings are. While various methods have been proposed to learn effective latent representations, generated images often lack realism, particularly in textured regions with sharp transitions, due to loss of fine details governed by high frequencies. We conduct a detailed frequency decomposition of existing state-of-the-art (SOTA) latent tokenizers and show that conventional objectives inherently prioritize low-frequency reconstruction, often at the expense of high-frequency fidelity. Our analysis reveals these latent tokenizers exhibit a bias toward low-frequency information during optimization, leading to over-smoothed outputs and visual artifacts that diminish perceptual quality. To address this, we propose a wavelet-based, frequency-aware variational autoencoder (FA-VAE) framework that explicitly decouples the optimization of low- and high-frequency components. This decoupling enables improved reconstruction of fine textures while preserving global structure. Moreover, we integrate our frequency-preserving latent embeddings into a SOTA latent diffusion model, resulting in sharper and more realistic image generation. Our approach bridges the fidelity gap in current latent tokenizers and emphasizes the importance of frequency-aware optimization for realistic image synthesis, with broader implications for applications in content creation, neural rendering, and medical imaging.
>
---
#### [replaced 067] A Challenging Benchmark of Anime Style Recognition
- **分类: cs.CV; I.2.10**

- **链接: [http://arxiv.org/pdf/2204.14034v2](http://arxiv.org/pdf/2204.14034v2)**

> **作者:** Haotang Li; Shengtao Guo; Kailin Lyu; Xiao Yang; Tianchen Chen; Jianqing Zhu; Huanqiang Zeng
>
> **备注:** accepted by CVPRW 2022
>
> **摘要:** Given two images of different anime roles, anime style recognition (ASR) aims to learn abstract painting style to determine whether the two images are from the same work, which is an interesting but challenging problem. Unlike biometric recognition, such as face recognition, iris recognition, and person re-identification, ASR suffers from a much larger semantic gap but receives less attention. In this paper, we propose a challenging ASR benchmark. Firstly, we collect a large-scale ASR dataset (LSASRD), which contains 20,937 images of 190 anime works and each work at least has ten different roles. In addition to the large-scale, LSASRD contains a list of challenging factors, such as complex illuminations, various poses, theatrical colors and exaggerated compositions. Secondly, we design a cross-role protocol to evaluate ASR performance, in which query and gallery images must come from different roles to validate an ASR model is to learn abstract painting style rather than learn discriminative features of roles. Finally, we apply two powerful person re-identification methods, namely, AGW and TransReID, to construct the baseline performance on LSASRD. Surprisingly, the recent transformer model (i.e., TransReID) only acquires a 42.24% mAP on LSASRD. Therefore, we believe that the ASR task of a huge semantic gap deserves deep and long-term research. We will open our dataset and code at https://github.com/nkjcqvcpi/ASR.
>
---
#### [replaced 068] From Images to Insights: Explainable Biodiversity Monitoring with Plain Language Habitat Explanations
- **分类: cs.CV; cs.AI; cs.ET**

- **链接: [http://arxiv.org/pdf/2506.10559v2](http://arxiv.org/pdf/2506.10559v2)**

> **作者:** Yutong Zhou; Masahiro Ryo
>
> **备注:** AISE workshop camera-ready version @ ECAI 2025
>
> **摘要:** Explaining why the species lives at a particular location is important for understanding ecological systems and conserving biodiversity. However, existing ecological workflows are fragmented and often inaccessible to non-specialists. We propose an end-to-end visual-to-causal framework that transforms a species image into interpretable causal insights about its habitat preference. The system integrates species recognition, global occurrence retrieval, pseudo-absence sampling, and climate data extraction. We then discover causal structures among environmental features and estimate their influence on species occurrence using modern causal inference methods. Finally, we generate statistically grounded, human-readable causal explanations from structured templates and large language models. We demonstrate the framework on a bee and a flower species and report early results as part of an ongoing project, showing the potential of the multimodal AI assistant backed up by a recommended ecological modeling practice for describing species habitat in human-understandable language. Our code is available at: https://github.com/Yutong-Zhou-cv/BioX.
>
---
#### [replaced 069] Interpretable Text-Guided Image Clustering via Iterative Search
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.12514v2](http://arxiv.org/pdf/2506.12514v2)**

> **作者:** Bingchen Zhao; Oisin Mac Aodha
>
> **摘要:** Traditional clustering methods aim to group unlabeled data points based on their similarity to each other. However, clustering, in the absence of additional information, is an ill-posed problem as there may be many different, yet equally valid, ways to partition a dataset. Distinct users may want to use different criteria to form clusters in the same data, e.g. shape v.s. color. Recently introduced text-guided image clustering methods aim to address this ambiguity by allowing users to specify the criteria of interest using natural language instructions. This instruction provides the necessary context and control needed to obtain clusters that are more aligned with the users' intent. We propose a new text-guided clustering approach named ITGC that uses an iterative discovery process, guided by an unsupervised clustering objective, to generate interpretable visual concepts that better capture the criteria expressed in a user's instructions. We report superior performance compared to existing methods across a wide variety of image clustering and fine-grained classification benchmarks.
>
---
#### [replaced 070] Evaluation of Alignment-Regularity Characteristics in Deformable Image Registration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07185v2](http://arxiv.org/pdf/2503.07185v2)**

> **作者:** Vasiliki Sideri-Lampretsa; Daniel Rueckert; Huaqi Qiu
>
> **摘要:** Evaluating deformable image registration (DIR) is challenging due to the inherent trade-off between achieving high alignment accuracy and maintaining deformation regularity. In this work, we introduce a novel evaluation scheme based on the alignment-regularity characteristic (ARC) to systematically capture and analyze this trade-off. We first introduce the ARC curves, which describe the performance of a given registration algorithm as a spectrum measured by alignment and regularity metrics. We further adopt a HyperNetwork-based approach that learns to continuously interpolate across the full regularization range, accelerating the construction and improving the sample density of ARC curves. We empirically demonstrate our evaluation scheme using representative learning-based deformable image registration methods with various network architectures and transformation models on two public datasets. We present a range of findings not evident from existing evaluation practices and provide general recommendations for model evaluation and selection using our evaluation scheme. All code relevant is made publicly available.
>
---
#### [replaced 071] Seeing More, Saying More: Lightweight Language Experts are Dynamic Video Token Compressors
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.00969v2](http://arxiv.org/pdf/2509.00969v2)**

> **作者:** Xiangchen Wang; Jinrui Zhang; Teng Wang; Haigang Zhang; Feng Zheng
>
> **备注:** 17 pages, 8 figures, EMNLP2025
>
> **摘要:** Recent advancements in large video-language models have revolutionized video understanding tasks. However, their efficiency is significantly constrained by processing high volumes of visual tokens. Existing token compression strategies apply a fixed compression ratio, ignoring the variability in semantic density among different video clips. Consequently, this lead to inadequate representation of information-rich clips due to insufficient tokens and unnecessary computation on static or content-poor ones. To address this, we propose LangDC, a Language-aware Dynamic Token Compressor. LangDC leverages a lightweight language model to describe video clips, converting them into soft caption tokens as visual representations. Trained with our proposed semantic density-aware supervision, LangDC aims to 1) cover key visual cues necessary for downstream task reasoning and 2) dynamically adjust compression ratios based on scene richness, reflected by descriptions length. Our design mimics how humans dynamically express what they see: complex scenes (seeing more) elicit more detailed language to convey nuances (saying more), whereas simpler scenes are described with fewer words. Experimental results show that our method reduces FLOPs by 49% compared to VideoGPT+ while maintaining competitive performance. Furthermore, qualitative results demonstrate our approach adaptively adjusts the token compression ratio based on video segment richness.
>
---
#### [replaced 072] HodgeFormer: Transformers for Learnable Operators on Triangular Meshes through Data-Driven Hodge Matrices
- **分类: cs.GR; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.01839v3](http://arxiv.org/pdf/2509.01839v3)**

> **作者:** Akis Nousias; Stavros Nousias
>
> **备注:** 13 pages, 11 figures, 9 tables
>
> **摘要:** Currently, prominent Transformer architectures applied on graphs and meshes for shape analysis tasks employ traditional attention layers that heavily utilize spectral features requiring costly eigenvalue decomposition-based methods. To encode the mesh structure, these methods derive positional embeddings, that heavily rely on eigenvalue decomposition based operations, e.g. on the Laplacian matrix, or on heat-kernel signatures, which are then concatenated to the input features. This paper proposes a novel approach inspired by the explicit construction of the Hodge Laplacian operator in Discrete Exterior Calculus as a product of discrete Hodge operators and exterior derivatives, i.e. $(L := \star_0^{-1} d_0^T \star_1 d_0)$. We adjust the Transformer architecture in a novel deep learning layer that utilizes the multi-head attention mechanism to approximate Hodge matrices $\star_0$, $\star_1$ and $\star_2$ and learn families of discrete operators $L$ that act on mesh vertices, edges and faces. Our approach results in a computationally-efficient architecture that achieves comparable performance in mesh segmentation and classification tasks, through a direct learning framework, while eliminating the need for costly eigenvalue decomposition operations or complex preprocessing operations.
>
---
#### [replaced 073] Enhancing Traffic Incident Response through Sub-Second Temporal Localization with HybridMamba
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.03235v2](http://arxiv.org/pdf/2504.03235v2)**

> **作者:** Ibne Farabi Shihab; Sanjeda Akter; Anuj Sharma
>
> **摘要:** Traffic crash detection in long-form surveillance videos is essential for improving emergency response and infrastructure planning, yet remains difficult due to the brief and infrequent nature of crash events. We present \textbf{HybridMamba}, a novel architecture integrating visual transformers with state-space temporal modeling to achieve high-precision crash time localization. Our approach introduces multi-level token compression and hierarchical temporal processing to maintain computational efficiency without sacrificing temporal resolution. Evaluated on a large-scale dataset from the Iowa Department of Transportation, HybridMamba achieves a mean absolute error of \textbf{1.50 seconds} for 2-minute videos ($p<0.01$ compared to baselines), with \textbf{65.2%} of predictions falling within one second of the ground truth. It outperforms recent video-language models (e.g., TimeChat, VideoLLaMA-2) by up to 3.95 seconds while using significantly fewer parameters (3B vs. 13--72B). Our results demonstrate effective temporal localization across various video durations (2--40 minutes) and diverse environmental conditions, highlighting HybridMamba's potential for fine-grained temporal localization in traffic surveillance while identifying challenges that remain for extended deployment.
>
---
#### [replaced 074] Semi-SMD: Semi-Supervised Metric Depth Estimation via Surrounding Cameras for Autonomous Driving
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.19713v3](http://arxiv.org/pdf/2503.19713v3)**

> **作者:** Yusen Xie; Zhengmin Huang; Shaojie Shen; Jun Ma
>
> **摘要:** In this paper, we introduce Semi-SMD, a novel metric depth estimation framework tailored for surrounding cameras equipment in autonomous driving. In this work, the input data consists of adjacent surrounding frames and camera parameters. We propose a unified spatial-temporal-semantic fusion module to construct the visual fused features. Cross-attention components for surrounding cameras and adjacent frames are utilized to focus on metric scale information refinement and temporal feature matching. Building on this, we propose a pose estimation framework using surrounding cameras, their corresponding estimated depths, and extrinsic parameters, which effectively address the scale ambiguity in multi-camera setups. Moreover, semantic world model and monocular depth estimation world model are integrated to supervised the depth estimation, which improve the quality of depth estimation. We evaluate our algorithm on DDAD and nuScenes datasets, and the results demonstrate that our method achieves state-of-the-art performance in terms of surrounding camera based depth estimation quality. The source code will be available on https://github.com/xieyuser/Semi-SMD.
>
---
#### [replaced 075] TractGraphFormer: Anatomically Informed Hybrid Graph CNN-Transformer Network for Interpretable Sex and Age Prediction from Diffusion MRI Tractography
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.08883v2](http://arxiv.org/pdf/2407.08883v2)**

> **作者:** Yuqian Chen; Fan Zhang; Meng Wang; Leo R. Zekelman; Suheyla Cetin-Karayumak; Tengfei Xue; Chaoyi Zhang; Yang Song; Jarrett Rushmore; Nikos Makris; Yogesh Rathi; Weidong Cai; Lauren J. O'Donnell
>
> **备注:** 27 pages, 5 figures
>
> **摘要:** The relationship between brain connections and non-imaging phenotypes is increasingly studied using deep neural networks. However, the local and global properties of brain white matter networks are often overlooked in convolutional network design. We introduce TractGraphFormer, a hybrid Graph CNN-Transformer deep learning framework tailored for diffusion MRI tractography. This model leverages local anatomical characteristics and global feature dependencies of white matter structures. The Graph CNN module captures white matter geometry and grey matter connectivity to aggregate local features from anatomically similar white matter connections, while the Transformer module uses self-attention to enhance global information learning. Additionally, TractGraphFormer includes an attention module for interpreting predictive white matter connections. We apply TractGraphFormer to tasks of sex and age prediction. TractGraphFormer shows strong performance in large datasets of children (n=9345) and young adults (n=1065). Overall, our approach suggests that widespread connections in the WM are predictive of the sex and age of an individual. For each prediction task, consistent predictive anatomical tracts are identified across the two datasets. The proposed approach highlights the potential of integrating local anatomical information and global feature dependencies to improve prediction performance in machine learning with diffusion MRI tractography.
>
---
#### [replaced 076] AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.22291v2](http://arxiv.org/pdf/2507.22291v2)**

> **作者:** Christopher F. Brown; Michal R. Kazmierski; Valerie J. Pasquarella; William J. Rucklidge; Masha Samsikova; Chenhui Zhang; Evan Shelhamer; Estefania Lahera; Olivia Wiles; Simon Ilyushchenko; Noel Gorelick; Lihui Lydia Zhang; Sophia Alj; Emily Schechter; Sean Askay; Oliver Guinan; Rebecca Moore; Alexis Boukouvalas; Pushmeet Kohli
>
> **摘要:** Unprecedented volumes of Earth observation data are continually collected around the world, but high-quality labels remain scarce given the effort required to make physical measurements and observations. This has led to considerable investment in bespoke modeling efforts translating sparse labels into maps. Here we introduce AlphaEarth Foundations, an embedding field model yielding a highly general, geospatial representation that assimilates spatial, temporal, and measurement contexts across multiple sources, enabling accurate and efficient production of maps and monitoring systems from local to global scales. The embeddings generated by AlphaEarth Foundations are the only to consistently outperform a suite of other well-known/widely accepted featurization approaches tested on a diverse set of mapping evaluations without re-training. We have released a dataset of global, annual, analysis-ready embedding field layers from 2017 through 2024.
>
---
#### [replaced 077] PnP-Flow: Plug-and-Play Image Restoration with Flow Matching
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.02423v3](http://arxiv.org/pdf/2410.02423v3)**

> **作者:** Ségolène Martin; Anne Gagneux; Paul Hagemann; Gabriele Steidl
>
> **摘要:** In this paper, we introduce Plug-and-Play (PnP) Flow Matching, an algorithm for solving imaging inverse problems. PnP methods leverage the strength of pre-trained denoisers, often deep neural networks, by integrating them in optimization schemes. While they achieve state-of-the-art performance on various inverse problems in imaging, PnP approaches face inherent limitations on more generative tasks like inpainting. On the other hand, generative models such as Flow Matching pushed the boundary in image sampling yet lack a clear method for efficient use in image restoration. We propose to combine the PnP framework with Flow Matching (FM) by defining a time-dependent denoiser using a pre-trained FM model. Our algorithm alternates between gradient descent steps on the data-fidelity term, reprojections onto the learned FM path, and denoising. Notably, our method is computationally efficient and memory-friendly, as it avoids backpropagation through ODEs and trace computations. We evaluate its performance on denoising, super-resolution, deblurring, and inpainting tasks, demonstrating superior results compared to existing PnP algorithms and Flow Matching based state-of-the-art methods.
>
---
#### [replaced 078] Detect Changes like Humans: Incorporating Semantic Priors for Improved Change Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.16918v2](http://arxiv.org/pdf/2412.16918v2)**

> **作者:** Yuhang Gan; Wenjie Xuan; Zhiming Luo; Lei Fang; Zengmao Wang; Juhua Liu; Bo Du
>
> **备注:** 25-09 accepted by IEEE TGRS
>
> **摘要:** When given two similar images, humans identify their differences by comparing the appearance (e.g., color, texture) with the help of semantics (e.g., objects, relations). However, mainstream binary change detection models adopt a supervised training paradigm, where the annotated binary change map is the main constraint. Thus, such methods primarily emphasize difference-aware features between bi-temporal images, and the semantic understanding of changed landscapes is undermined, resulting in limited accuracy in the face of noise and illumination variations. To this end, this paper explores incorporating semantic priors from visual foundation models to improve the ability to detect changes. Firstly, we propose a Semantic-Aware Change Detection network (SA-CDNet), which transfers the knowledge of visual foundation models (i.e., FastSAM) to change detection. Inspired by the human visual paradigm, a novel dual-stream feature decoder is derived to distinguish changes by combining semantic-aware features and difference-aware features. Secondly, we explore a single-temporal pre-training strategy for better adaptation of visual foundation models. With pseudo-change data constructed from single-temporal segmentation datasets, we employ an extra branch of proxy semantic segmentation task for pre-training. We explore various settings like dataset combinations and landscape types, thus providing valuable insights. Experimental results on five challenging benchmarks demonstrate the superiority of our method over the existing state-of-the-art methods. The code is available at $\href{https://github.com/DREAMXFAR/SA-CDNet}{github}$.
>
---
#### [replaced 079] Hybrid-Regularized Magnitude Pruning for Robust Federated Learning under Covariate Shift
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.15010v2](http://arxiv.org/pdf/2412.15010v2)**

> **作者:** Ozgu Goksu; Nicolas Pugeault
>
> **摘要:** Federated Learning offers a solution for decentralised model training, addressing the difficulties associated with distributed data and privacy in machine learning. However, the fact of data heterogeneity in federated learning frequently hinders the global model's generalisation, leading to low performance and adaptability to unseen data. This problem is particularly critical for specialised applications such as medical imaging, where both the data and the number of clients are limited. In this paper, we empirically demonstrate that inconsistencies in client-side training distributions substantially degrade the performance of federated learning models across multiple benchmark datasets. We propose a novel FL framework using a combination of pruning and regularisation of clients' training to improve the sparsity, redundancy, and robustness of neural connections, and thereby the resilience to model aggregation. To address a relatively unexplored dimension of data heterogeneity, we further introduce a novel benchmark dataset, CelebA-Gender, specifically designed to control for within-class distributional shifts across clients based on attribute variations, thereby complementing the predominant focus on inter-class imbalance in prior federated learning research. Comprehensive experiments on many datasets like CIFAR-10, MNIST, and the newly introduced CelebA-Gender dataset demonstrate that our method consistently outperforms standard FL baselines, yielding more robust and generalizable models in heterogeneous settings.
>
---
#### [replaced 080] A Data-Free Analytical Quantization Scheme for Deep Learning Models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.07391v3](http://arxiv.org/pdf/2412.07391v3)**

> **作者:** Ahmed Luqman; Khuzemah Qazi; Murray Patterson; Malik Jahan Khan; Imdadullah Khan
>
> **备注:** Accepted for publication in IEEE International Conference on Data Mining (ICDM 2025)
>
> **摘要:** Despite the success of CNN models on a variety of Image classification and segmentation tasks, their extensive computational and storage demands pose considerable challenges for real-world deployment on resource-constrained devices. Quantization is one technique that aims to alleviate these large storage requirements and speed up the inference process by reducing the precision of model parameters to lower-bit representations. In this paper, we introduce a novel post-training quantization method for model weights. Our method finds optimal clipping thresholds and scaling factors along with mathematical guarantees that our method minimizes quantization noise. Empirical results on real-world datasets demonstrate that our quantization scheme significantly reduces model size and computational requirements while preserving model accuracy.
>
---
#### [replaced 081] Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13061v3](http://arxiv.org/pdf/2502.13061v3)**

> **作者:** Jingbiao Mei; Jinghong Chen; Guangyu Yang; Weizhe Lin; Bill Byrne
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While Large Multimodal Models (LMMs) have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both supervised fine-tuning (SFT) and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Analysis reveals that our approach achieves improved robustness under adversarial attacks compared to SFT models. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems. Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability. Code available at https://github.com/JingbiaoMei/RGCL
>
---
#### [replaced 082] One Flight Over the Gap: A Survey from Perspective to Panoramic Vision
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.04444v2](http://arxiv.org/pdf/2509.04444v2)**

> **作者:** Xin Lin; Xian Ge; Dizhe Zhang; Zhaoliang Wan; Xianshun Wang; Xiangtai Li; Wenjie Jiang; Bo Du; Dacheng Tao; Ming-Hsuan Yang; Lu Qi
>
> **备注:** Project Page: https://insta360-research-team.github.io/Survey-of-Panorama/
>
> **摘要:** Driven by the demand for spatial intelligence and holistic scene perception, omnidirectional images (ODIs), which provide a complete 360\textdegree{} field of view, are receiving growing attention across diverse applications such as virtual reality, autonomous driving, and embodied robotics. Despite their unique characteristics, ODIs exhibit remarkable differences from perspective images in geometric projection, spatial distribution, and boundary continuity, making it challenging for direct domain adaption from perspective methods. This survey reviews recent panoramic vision techniques with a particular emphasis on the perspective-to-panorama adaptation. We first revisit the panoramic imaging pipeline and projection methods to build the prior knowledge required for analyzing the structural disparities. Then, we summarize three challenges of domain adaptation: severe geometric distortions near the poles, non-uniform sampling in Equirectangular Projection (ERP), and periodic boundary continuity. Building on this, we cover 20+ representative tasks drawn from more than 300 research papers in two dimensions. On one hand, we present a cross-method analysis of representative strategies for addressing panoramic specific challenges across different tasks. On the other hand, we conduct a cross-task comparison and classify panoramic vision into four major categories: visual quality enhancement and assessment, visual understanding, multimodal understanding, and visual generation. In addition, we discuss open challenges and future directions in data, models, and applications that will drive the advancement of panoramic vision research. We hope that our work can provide new insight and forward looking perspectives to advance the development of panoramic vision technologies. Our project page is https://insta360-research-team.github.io/Survey-of-Panorama
>
---
#### [replaced 083] Involution and BSConv Multi-Depth Distillation Network for Lightweight Image Super-Resolution
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.14779v2](http://arxiv.org/pdf/2503.14779v2)**

> **作者:** Akram Khatami-Rizi; Ahmad Mahmoudi-Aznaveh
>
> **摘要:** Single-image super-resolution (SISR) is a fundamental problem in computer vision that aims to reconstruct high-resolution (HR) images from low-resolution (LR) inputs. Although convolutional neural networks (CNNs) have achieved substantial advancements, deeper architectures often introduce excessive parameters, higher memory usage, and computational cost, limiting their applicability on resource-constrained devices. Recent research has thus focused on lightweight architectures that preserve accuracy while reducing complexity. This paper presents the Involution and BSConv Multi-Depth Distillation Network (IBMDN), a lightweight and effective architecture for SISR. The proposed IBMDN comprises Involution and BSConv Multi-Depth Distillation Blocks (IBMDB) and a Contrast and High-Frequency Attention Block (CHFAB). IBMDB employs varying combinations of Involution and BSConv at multiple depths to perform efficient feature extraction while minimizing computational complexity. CHFAB, a lightweight self-attention mechanism, focuses on extracting high-frequency and contrast information to enhance perceptual quality in the reconstructed images. The flexible design of IBMDB enables it to be seamlessly integrated into diverse SISR frameworks, including information distillation, transformer-based, and GAN-based models. Extensive experiments demonstrate that incorporating IBMDB significantly reduces memory usage, parameters, and floating-point operations (FLOPs), while achieving improvements in both pixel-wise accuracy and visual quality. The source code is available at: https://github.com/akramkhatami/IBMDN.
>
---
#### [replaced 084] Hybrid Swin Attention Networks for Simultaneously Low-Dose PET and CT Denoising
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.06591v2](http://arxiv.org/pdf/2509.06591v2)**

> **作者:** Yichao Liu; Hengzhi Xue; YueYang Teng
>
> **摘要:** Low-dose computed tomography (LDCT) and positron emission tomography (PET) have emerged as safer alternatives to conventional imaging modalities by significantly reducing radiation exposure. However, this reduction often results in increased noise and artifacts, which can compromise diagnostic accuracy. Consequently, denoising for LDCT/PET has become a vital area of research aimed at enhancing image quality while maintaining radiation safety. In this study, we introduce a novel Hybrid Swin Attention Network (HSANet), which incorporates Efficient Global Attention (EGA) modules and a hybrid upsampling module. The EGA modules enhance both spatial and channel-wise interaction, improving the network's capacity to capture relevant features, while the hybrid upsampling module mitigates the risk of overfitting to noise. We validate the proposed approach using a publicly available LDCT/PET dataset. Experimental results demonstrate that HSANet achieves superior denoising performance compared to existing methods, while maintaining a lightweight model size suitable for deployment on GPUs with standard memory configurations. This makes our approach highly practical for real-world clinical applications.
>
---
#### [replaced 085] SPACE-iT: Spatial-Aware Curriculum Exploration and Feedback-Driven Adaptive Augmentation for Vision Transformer Distillation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.10582v2](http://arxiv.org/pdf/2506.10582v2)**

> **作者:** Jihyeon Seong; Hyunkyung Han
>
> **备注:** 7 pages
>
> **摘要:** Knowledge distillation (KD) has proven to be a powerful technique for improving the performance of Vision Transformers (ViTs). However, traditional KD methods often treat all image patches uniformly, overlooking spatial variations in learning difficulty. To address this limitation, we propose SPACE-iT, a novel framework for Spatial-Aware Curriculum Exploration via Feedback-Driven Adaptive Augmentation. At its core, SPACE-iT computes spatial confidence scores at the attention, patch, and logit levels. This confidence map supports a two-fold strategy: (1) dynamically modulating the distillation loss, and (2) guiding an adaptive augmentation module that intensifies reverse curriculum learning. By establishing a feedback-driven reverse curriculum that initially exposes students to challenging regions-progressing from hard to easy-SPACE-iT enables more effective learning of complex spatial patterns and achieves superior performance over vanilla distillation, without introducing additional memory overhead.
>
---
