# 计算机视觉 cs.CV

- **最新发布 96 篇**

- **更新 69 篇**

## 最新发布

#### [new 001] Hyperspectral Smoke Segmentation via Mixture of Prototypes
- **分类: cs.CV**

- **简介: 该论文属于烟雾分割任务，解决传统方法在光谱信息不足和复杂环境下的局限性。提出混合原型网络，结合高光谱与多光谱数据，提升分割效果。**

- **链接: [https://arxiv.org/pdf/2602.10858v1](https://arxiv.org/pdf/2602.10858v1)**

> **作者:** Lujian Yao; Haitao Zhao; Xianghai Kong; Yuhan Xu
>
> **备注:** 35 pages, 14 figures
>
> **摘要:** Smoke segmentation is critical for wildfire management and industrial safety applications. Traditional visible-light-based methods face limitations due to insufficient spectral information, particularly struggling with cloud interference and semi-transparent smoke regions. To address these challenges, we introduce hyperspectral imaging for smoke segmentation and present the first hyperspectral smoke segmentation dataset (HSSDataset) with carefully annotated samples collected from over 18,000 frames across 20 real-world scenarios using a Many-to-One annotations protocol. However, different spectral bands exhibit varying discriminative capabilities across spatial regions, necessitating adaptive band weighting strategies. We decompose this into three technical challenges: spectral interaction contamination, limited spectral pattern modeling, and complex weighting router problems. We propose a mixture of prototypes (MoP) network with: (1) Band split for spectral isolation, (2) Prototype-based spectral representation for diverse patterns, and (3) Dual-level router for adaptive spatial-aware band weighting. We further construct a multispectral dataset (MSSDataset) with RGB-infrared images. Extensive experiments validate superior performance across both hyperspectral and multispectral modalities, establishing a new paradigm for spectral-based smoke segmentation.
>
---
#### [new 002] Monte Carlo Maximum Likelihood Reconstruction for Digital Holography with Speckle
- **分类: cs.CV**

- **简介: 该论文属于数字全息图像重建任务，解决相干成像中散斑噪声问题。通过提出PGD-MC方法，实现高效且准确的MLE重构，提升重建质量和计算效率。**

- **链接: [https://arxiv.org/pdf/2602.10344v1](https://arxiv.org/pdf/2602.10344v1)**

> **作者:** Xi Chen; Arian Maleki; Shirin Jalali
>
> **摘要:** In coherent imaging, speckle is statistically modeled as multiplicative noise, posing a fundamental challenge for image reconstruction. While maximum likelihood estimation (MLE) provides a principled framework for speckle mitigation, its application to coherent imaging system such as digital holography with finite apertures is hindered by the prohibitive cost of high-dimensional matrix inversion, especially at high resolutions. This computational burden has prevented the use of MLE-based reconstruction with physically accurate aperture modeling. In this work, we propose a randomized linear algebra approach that enables scalable MLE optimization without explicit matrix inversions in gradient computation. By exploiting the structural properties of sensing matrix and using conjugate gradient for likelihood gradient evaluation, the proposed algorithm supports accurate aperture modeling without the simplifying assumptions commonly imposed for tractability. We term the resulting method projected gradient descent with Monte Carlo estimation (PGD-MC). The proposed PGD-MC framework (i) demonstrates robustness to diverse and physically accurate aperture models, (ii) achieves substantial improvements in reconstruction quality and computational efficiency, and (iii) scales effectively to high-resolution digital holography. Extensive experiments incorporating three representative denoisers as regularization show that PGD-MC provides a flexible and effective MLE-based reconstruction framework for digital holography with finite apertures, consistently outperforming prior Plug-and-Play model-based iterative reconstruction methods in both accuracy and speed. Our code is available at: https://github.com/Computational-Imaging-RU/MC_Maximum_Likelihood_Digital_Holography_Speckle.
>
---
#### [new 003] Improving Medical Visual Reinforcement Fine-Tuning via Perception and Reasoning Augmentation
- **分类: cs.CV**

- **简介: 该论文属于医疗图像领域的强化学习任务，旨在解决视觉感知与推理不足的问题。通过引入VRFT-Aug框架，增强模型的视觉和推理能力，提升医疗应用效果。**

- **链接: [https://arxiv.org/pdf/2602.10619v1](https://arxiv.org/pdf/2602.10619v1)**

> **作者:** Guangjing Yang; ZhangYuan Yu; Ziyuan Qin; Xinyuan Song; Huahui Yi; Qingbo Kang; Jun Gao; Yiyue Li; Chenlin Du; Qicheng Lao
>
> **备注:** CPAL 2026
>
> **摘要:** While recent advances in Reinforcement Fine-Tuning (RFT) have shown that rule-based reward schemes can enable effective post-training for large language models, their extension to cross-modal, vision-centric domains remains largely underexplored. This limitation is especially pronounced in the medical imaging domain, where effective performance requires both robust visual perception and structured reasoning. In this work, we address this gap by proposing VRFT-Aug, a visual reinforcement fine-tuning framework tailored for the medical domain. VRFT-Aug introduces a series of training strategies designed to augment both perception and reasoning, including prior knowledge injection, perception-driven policy refinement, medically informed reward shaping, and behavioral imitation. Together, these methods aim to stabilize and improve the RFT process. Through extensive experiments across multiple medical datasets, we show that our approaches consistently outperform both standard supervised fine-tuning and RFT baselines. Moreover, we provide empirically grounded insights and practical training heuristics that can be generalized to other medical image tasks. We hope this work contributes actionable guidance and fresh inspiration for the ongoing effort to develop reliable, reasoning-capable models for high-stakes medical applications.
>
---
#### [new 004] Conditional Uncertainty-Aware Political Deepfake Detection with Stochastic Convolutional Neural Networks
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于政治深度伪造检测任务，旨在解决现有系统无法有效评估预测不确定性的问题。通过引入随机卷积神经网络，提升检测的可靠性与风险感知能力。**

- **链接: [https://arxiv.org/pdf/2602.10343v1](https://arxiv.org/pdf/2602.10343v1)**

> **作者:** Rafael-Petruţ Gardoş
>
> **备注:** 21 pages, 12 figures, 18 tables
>
> **摘要:** Recent advances in generative image models have enabled the creation of highly realistic political deepfakes, posing risks to information integrity, public trust, and democratic processes. While automated deepfake detectors are increasingly deployed in moderation and investigative pipelines, most existing systems provide only point predictions and fail to indicate when outputs are unreliable, being an operationally critical limitation in high-stakes political contexts. This work investigates conditional, uncertainty-aware political deepfake detection using stochastic convolutional neural networks within an empirical, decision-oriented reliability framework. Rather than treating uncertainty as a purely Bayesian construct, it is evaluated through observable criteria, including calibration quality, proper scoring rules, and its alignment with prediction errors under both global and confidence-conditioned analyses. A politically focused binary image dataset is constructed via deterministic metadata filtering from a large public real-synthetic corpus. Two pretrained CNN backbones (ResNet-18 and EfficientNet-B4) are fully fine-tuned for classification. Deterministic inference is compared with single-pass stochastic prediction, Monte Carlo dropout with multiple forward passes, temperature scaling, and ensemble-based uncertainty surrogates. Evaluation reports ROC-AUC, thresholded confusion matrices, calibration metrics, and generator-disjoint out-of-distribution performance. Results demonstrate that calibrated probabilistic outputs and uncertainty estimates enable risk-aware moderation policies. A systematic confidence-band analysis further clarifies when uncertainty provides operational value beyond predicted confidence, delineating both the benefits and limitations of uncertainty-aware deepfake detection in political settings.
>
---
#### [new 005] (MGS)$^2$-Net: Unifying Micro-Geometric Scale and Macro-Geometric Structure for Cross-View Geo-Localization
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于跨视角地理定位任务，解决航拍图像与卫星图间几何错位问题。提出(MGS)$^2$框架，融合宏观结构与微观尺度，提升定位精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.10704v1](https://arxiv.org/pdf/2602.10704v1)**

> **作者:** Minglei Li; Mengfan He; Chao Chen; Ziyang Meng
>
> **摘要:** Cross-view geo-localization (CVGL) is pivotal for GNSS-denied UAV navigation but remains brittle under the drastic geometric misalignment between oblique aerial views and orthographic satellite references. Existing methods predominantly operate within a 2D manifold, neglecting the underlying 3D geometry where view-dependent vertical facades (macro-structure) and scale variations (micro-scale) severely corrupt feature alignment. To bridge this gap, we propose (MGS)$^2$, a geometry-grounded framework. The core of our innovation is the Macro-Geometric Structure Filtering (MGSF) module. Unlike pixel-wise matching sensitive to noise, MGSF leverages dilated geometric gradients to physically filter out high-frequency facade artifacts while enhancing the view-invariant horizontal plane, directly addressing the domain shift. To guarantee robust input for this structural filtering, we explicitly incorporate a Micro-Geometric Scale Adaptation (MGSA) module. MGSA utilizes depth priors to dynamically rectify scale discrepancies via multi-branch feature fusion. Furthermore, a Geometric-Appearance Contrastive Distillation (GACD) loss is designed to strictly discriminate against oblique occlusions. Extensive experiments demonstrate that (MGS)$^2$ achieves state-of-the-art performance, recording a Recall@1 of 97.5\% on University-1652 and 97.02\% on SUES-200. Furthermore, the framework exhibits superior cross-dataset generalization against geometric ambiguity. The code is available at: \href{https://github.com/GabrielLi1473/MGS-Net}{https://github.com/GabrielLi1473/MGS-Net}.
>
---
#### [new 006] A Vision-Language Foundation Model for Zero-shot Clinical Collaboration and Automated Concept Discovery in Dermatology
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出DermFM-Zero，一种用于皮肤科的视觉-语言基础模型，解决临床协作与概念发现问题，通过多模态训练实现零样本诊断和检索。**

- **链接: [https://arxiv.org/pdf/2602.10624v1](https://arxiv.org/pdf/2602.10624v1)**

> **作者:** Siyuan Yan; Xieji Li; Dan Mo; Philipp Tschandl; Yiwen Jiang; Zhonghua Wang; Ming Hu; Lie Ju; Cristina Vico-Alonso; Yizhen Zheng; Jiahe Liu; Juexiao Zhou; Camilla Chello; Jen G. Cheung; Julien Anriot; Luc Thomas; Clare Primiero; Gin Tan; Aik Beng Ng; Simon See; Xiaoying Tang; Albert Ip; Xiaoyang Liao; Adrian Bowling; Martin Haskett; Shuang Zhao; Monika Janda; H. Peter Soyer; Victoria Mar; Harald Kittler; Zongyuan Ge
>
> **备注:** reports
>
> **摘要:** Medical foundation models have shown promise in controlled benchmarks, yet widespread deployment remains hindered by reliance on task-specific fine-tuning. Here, we introduce DermFM-Zero, a dermatology vision-language foundation model trained via masked latent modelling and contrastive learning on over 4 million multimodal data points. We evaluated DermFM-Zero across 20 benchmarks spanning zero-shot diagnosis and multimodal retrieval, achieving state-of-the-art performance without task-specific adaptation. We further evaluated its zero-shot capabilities in three multinational reader studies involving over 1,100 clinicians. In primary care settings, AI assistance enabled general practitioners to nearly double their differential diagnostic accuracy across 98 skin conditions. In specialist settings, the model significantly outperformed board-certified dermatologists in multimodal skin cancer assessment. In collaborative workflows, AI assistance enabled non-experts to surpass unassisted experts while improving management appropriateness. Finally, we show that DermFM-Zero's latent representations are interpretable: sparse autoencoders unsupervisedly disentangle clinically meaningful concepts that outperform predefined-vocabulary approaches and enable targeted suppression of artifact-induced biases, enhancing robustness without retraining. These findings demonstrate that a foundation model can provide effective, safe, and transparent zero-shot clinical decision support.
>
---
#### [new 007] Enhancing Underwater Images via Adaptive Semantic-aware Codebook Learning
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于 underwater image enhancement 任务，解决因场景组件退化不一致导致的颜色失真和细节丢失问题。提出 SUCode 方法，通过语义感知的代码本进行自适应增强。**

- **链接: [https://arxiv.org/pdf/2602.10586v1](https://arxiv.org/pdf/2602.10586v1)**

> **作者:** Bosen Lin; Feng Gao; Yanwei Yu; Junyu Dong; Qian Du
>
> **备注:** Accepted for publication in IEEE TGRS 2026
>
> **摘要:** Underwater Image Enhancement (UIE) is an ill-posed problem where natural clean references are not available, and the degradation levels vary significantly across semantic regions. Existing UIE methods treat images with a single global model and ignore the inconsistent degradation of different scene components. This oversight leads to significant color distortions and loss of fine details in heterogeneous underwater scenes, especially where degradation varies significantly across different image regions. Therefore, we propose SUCode (Semantic-aware Underwater Codebook Network), which achieves adaptive UIE from semantic-aware discrete codebook representation. Compared with one-shot codebook-based methods, SUCode exploits semantic-aware, pixel-level codebook representation tailored to heterogeneous underwater degradation. A three-stage training paradigm is employed to represent raw underwater image features to avoid pseudo ground-truth contamination. Gated Channel Attention Module (GCAM) and Frequency-Aware Feature Fusion (FAFF) jointly integrate channel and frequency cues for faithful color restoration and texture recovery. Extensive experiments on multiple benchmarks demonstrate that SUCode achieves state-of-the-art performance, outperforming recent UIE methods on both reference and no-reference metrics. The code will be made public available at https://github.com/oucailab/SUCode.
>
---
#### [new 008] Towards Remote Sensing Change Detection with Neural Memory
- **分类: cs.CV**

- **简介: 该论文属于遥感变化检测任务，旨在解决长距离依赖捕捉与计算效率的矛盾。提出ChangeTitans框架，结合神经记忆和分段局部注意力，提升检测精度与效率。**

- **链接: [https://arxiv.org/pdf/2602.10491v1](https://arxiv.org/pdf/2602.10491v1)**

> **作者:** Zhenyu Yang; Gensheng Pei; Yazhou Yao; Tianfei Zhou; Lizhong Ding; Fumin Shen
>
> **备注:** accepted by IEEE Transactions on Geoscience & Remote Sensing
>
> **摘要:** Remote sensing change detection is essential for environmental monitoring, urban planning, and related applications. However, current methods often struggle to capture long-range dependencies while maintaining computational efficiency. Although Transformers can effectively model global context, their quadratic complexity poses scalability challenges, and existing linear attention approaches frequently fail to capture intricate spatiotemporal relationships. Drawing inspiration from the recent success of Titans in language tasks, we present ChangeTitans, the Titans-based framework for remote sensing change detection. Specifically, we propose VTitans, the first Titans-based vision backbone that integrates neural memory with segmented local attention, thereby capturing long-range dependencies while mitigating computational overhead. Next, we present a hierarchical VTitans-Adapter to refine multi-scale features across different network layers. Finally, we introduce TS-CBAM, a two-stream fusion module leveraging cross-temporal attention to suppress pseudo-changes and enhance detection accuracy. Experimental evaluations on four benchmark datasets (LEVIR-CD, WHU-CD, LEVIR-CD+, and SYSU-CD) demonstrate that ChangeTitans achieves state-of-the-art results, attaining \textbf{84.36\%} IoU and \textbf{91.52\%} F1-score on LEVIR-CD, while remaining computationally competitive.
>
---
#### [new 009] Towards Learning a Generalizable 3D Scene Representation from 2D Observations
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D场景重建任务，旨在从2D观测中学习可泛化的3D占用表示。解决传统方法在全局坐标系下泛化能力不足的问题，通过整合多视角信息实现无需微调的场景预测。**

- **链接: [https://arxiv.org/pdf/2602.10943v1](https://arxiv.org/pdf/2602.10943v1)**

> **作者:** Martin Gromniak; Jan-Gerrit Habekost; Sebastian Kamp; Sven Magg; Stefan Wermter
>
> **备注:** Paper accepted at ESANN 2026
>
> **摘要:** We introduce a Generalizable Neural Radiance Field approach for predicting 3D workspace occupancy from egocentric robot observations. Unlike prior methods operating in camera-centric coordinates, our model constructs occupancy representations in a global workspace frame, making it directly applicable to robotic manipulation. The model integrates flexible source views and generalizes to unseen object arrangements without scene-specific finetuning. We demonstrate the approach on a humanoid robot and evaluate predicted geometry against 3D sensor ground truth. Trained on 40 real scenes, our model achieves 26mm reconstruction error, including occluded regions, validating its ability to infer complete 3D occupancy beyond traditional stereo vision methods.
>
---
#### [new 010] Resource-Efficient RGB-Only Action Recognition for Edge Deployment
- **分类: cs.CV; cs.PF**

- **简介: 该论文属于动作识别任务，旨在解决边缘设备上资源受限的问题。提出一种轻量级RGB-only网络，提升效率与实用性。**

- **链接: [https://arxiv.org/pdf/2602.10818v1](https://arxiv.org/pdf/2602.10818v1)**

> **作者:** Dongsik Yoon; Jongeun Kim; Dayeon Lee
>
> **备注:** Under review
>
> **摘要:** Action recognition on edge devices poses stringent constraints on latency, memory, storage, and power consumption. While auxiliary modalities such as skeleton and depth information can enhance recognition performance, they often require additional sensors or computationally expensive pose-estimation pipelines, limiting practicality for edge use. In this work, we propose a compact RGB-only network tailored for efficient on-device inference. Our approach builds upon an X3D-style backbone augmented with Temporal Shift, and further introduces selective temporal adaptation and parameter-free attention. Extensive experiments on the NTU RGB+D 60 and 120 benchmarks demonstrate a strong accuracy-efficiency balance. Moreover, deployment-level profiling on the Jetson Orin Nano verifies a smaller on-device footprint and practical resource utilization compared to existing RGB-based action recognition techniques.
>
---
#### [new 011] Ecological mapping with geospatial foundation models
- **分类: cs.CV**

- **简介: 该论文探讨GFMs在生态制图中的应用，解决其有效性、挑战与机遇问题。通过微调模型进行土地利用分类、森林功能特征和泥炭地检测，验证GFMs优于传统模型。**

- **链接: [https://arxiv.org/pdf/2602.10720v1](https://arxiv.org/pdf/2602.10720v1)**

> **作者:** Craig Mahlasi; Gciniwe S. Baloyi; Zaheed Gaffoor; Levente Klein; Anne Jones; Etienne Vos; Michal Muszynski; Geoffrey Dawson; Campbell Watson
>
> **摘要:** Geospatial foundation models (GFMs) are a fast-emerging paradigm for various geospatial tasks, such as ecological mapping. However, the utility of GFMs has not been fully explored for high-value use cases. This study aims to explore the utility, challenges and opportunities associated with the application of GFMs for ecological uses. In this regard, we fine-tune several pretrained AI models, namely, Prithvi-E0-2.0 and TerraMind, across three use cases, and compare this with a baseline ResNet-101 model. Firstly, we demonstrate TerraMind's LULC generation capabilities. Lastly, we explore the utility of the GFMs in forest functional trait mapping and peatlands detection. In all experiments, the GFMs outperform the baseline ResNet models. In general TerraMind marginally outperforms Prithvi. However, with additional modalities TerraMind significantly outperforms the baseline ResNet and Prithvi models. Nonetheless, consideration should be given to the divergence of input data from pretrained modalities. We note that these models would benefit from higher resolution and more accurate labels, especially for use cases where pixel-level dynamics need to be mapped.
>
---
#### [new 012] Spectral-Spatial Contrastive Learning Framework for Regression on Hyperspectral Data
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于回归任务，针对高光谱数据的回归问题，提出一种光谱-空间对比学习框架，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.10745v1](https://arxiv.org/pdf/2602.10745v1)**

> **作者:** Mohamad Dhaini; Paul Honeine; Maxime Berar; Antonin Van Exem
>
> **摘要:** Contrastive learning has demonstrated great success in representation learning, especially for image classification tasks. However, there is still a shortage in studies targeting regression tasks, and more specifically applications on hyperspectral data. In this paper, we propose a spectral-spatial contrastive learning framework for regression tasks for hyperspectral data, in a model-agnostic design allowing to enhance backbones such as 3D convolutional and transformer-based networks. Moreover, we provide a collection of transformations relevant for augmenting hyperspectral data. Experiments on synthetic and real datasets show that the proposed framework and transformations significantly improve the performance of all studied backbone models.
>
---
#### [new 013] MapVerse: A Benchmark for Geospatial Question Answering on Diverse Real-World Maps
- **分类: cs.CV**

- **简介: 该论文提出MapVerse，一个用于评估地理空间问答的基准数据集，解决现有数据集范围窄、依赖人工生成内容的问题。任务属于多模态地理信息理解，旨在提升模型在真实地图上的推理能力。**

- **链接: [https://arxiv.org/pdf/2602.10518v1](https://arxiv.org/pdf/2602.10518v1)**

> **作者:** Sharat Bhat; Harshita Khandelwal; Tushar Kataria; Vivek Gupta
>
> **摘要:** Maps are powerful carriers of structured and contextual knowledge, encompassing geography, demographics, infrastructure, and environmental patterns. Reasoning over such knowledge requires models to integrate spatial relationships, visual cues, real-world context, and domain-specific expertise-capabilities that current large language models (LLMs) and vision-language models (VLMs) still struggle to exhibit consistently. Yet, datasets used to benchmark VLMs on map-based reasoning remain narrow in scope, restricted to specific domains, and heavily reliant on artificially generated content (outputs from LLMs or pipeline-based methods), offering limited depth for evaluating genuine geospatial reasoning. To address this gap, we present MapVerse, a large-scale benchmark built on real-world maps. It comprises 11,837 human-authored question-answer pairs across 1,025 maps, spanning ten diverse map categories and multiple question categories for each. The dataset provides a rich setting for evaluating map reading, interpretation, and multimodal reasoning. We evaluate ten state-of-the-art models against our benchmark to establish baselines and quantify reasoning gaps. Beyond overall performance, we conduct fine-grained categorical analyses to assess model inference across multiple dimensions and investigate the visual factors shaping reasoning outcomes. Our findings reveal that while current VLMs perform competitively on classification-style tasks, both open- and closed-source models fall short on advanced tasks requiring complex spatial reasoning.
>
---
#### [new 014] Colorimeter-Supervised Skin Tone Estimation from Dermatoscopic Images for Fairness Auditing
- **分类: cs.CV**

- **简介: 该论文属于皮肤色调估计任务，解决皮肤病图像模型公平性审计中的标注缺失问题，通过神经网络预测皮肤类型和颜色参数，提升模型偏差评估能力。**

- **链接: [https://arxiv.org/pdf/2602.10265v1](https://arxiv.org/pdf/2602.10265v1)**

> **作者:** Marin Benčević; Krešimir Romić; Ivana Hartmann Tolić; Irena Galić
>
> **备注:** Preprint submitted to Computer Methods and Programs in Biomedicine
>
> **摘要:** Neural-network-based diagnosis from dermatoscopic images is increasingly used for clinical decision support, yet studies report performance disparities across skin tones. Fairness auditing of these models is limited by the lack of reliable skin-tone annotations in public dermatoscopy datasets. We address this gap with neural networks that predict Fitzpatrick skin type via ordinal regression and the Individual Typology Angle (ITA) via color regression, using in-person Fitzpatrick labels and colorimeter measurements as targets. We further leverage extensive pretraining on synthetic and real dermatoscopic and clinical images. The Fitzpatrick model achieves agreement comparable to human crowdsourced annotations, and ITA predictions show high concordance with colorimeter-derived ITA, substantially outperforming pixel-averaging approaches. Applying these estimators to ISIC 2020 and MILK10k, we find that fewer than 1% of subjects belong to Fitzpatrick types V and VI. We release code and pretrained models as an open-source tool for rapid skin-tone annotation and bias auditing. This is, to our knowledge, the first dermatoscopic skin-tone estimation neural network validated against colorimeter measurements, and it supports growing evidence of clinically relevant performance gaps across skin-tone groups.
>
---
#### [new 015] Interpretable Vision Transformers in Image Classification via SVDA
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决Vision Transformers注意力机制不透明的问题。通过引入SVDA机制，提升注意力的可解释性与结构化。**

- **链接: [https://arxiv.org/pdf/2602.10994v1](https://arxiv.org/pdf/2602.10994v1)**

> **作者:** Vasileios Arampatzakis; George Pavlidis; Nikolaos Mitianoudis; Nikos Papamarkos
>
> **备注:** 10 pages, 4 figures, submitted to IEEE Access
>
> **摘要:** Vision Transformers (ViTs) have achieved state-of-the-art performance in image classification, yet their attention mechanisms often remain opaque and exhibit dense, non-structured behaviors. In this work, we adapt our previously proposed SVD-Inspired Attention (SVDA) mechanism to the ViT architecture, introducing a geometrically grounded formulation that enhances interpretability, sparsity, and spectral structure. We apply the use of interpretability indicators -- originally proposed with SVDA -- to monitor attention dynamics during training and assess structural properties of the learned representations. Experimental evaluations on four widely used benchmarks -- CIFAR-10, FashionMNIST, CIFAR-100, and ImageNet-100 -- demonstrate that SVDA consistently yields more interpretable attention patterns without sacrificing classification accuracy. While the current framework offers descriptive insights rather than prescriptive guidance, our results establish SVDA as a comprehensive and informative tool for analyzing and developing structured attention models in computer vision. This work lays the foundation for future advances in explainable AI, spectral diagnostics, and attention-based model compression.
>
---
#### [new 016] When the Prompt Becomes Visual: Vision-Centric Jailbreak Attacks for Large Image Editing Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视觉驱动的图像编辑模型安全问题，提出视觉中心的越狱攻击方法VJA，旨在揭示模型在视觉输入下的安全风险，并提供防御方案。**

- **链接: [https://arxiv.org/pdf/2602.10179v1](https://arxiv.org/pdf/2602.10179v1)**

> **作者:** Jiacheng Hou; Yining Sun; Ruochong Jin; Haochen Han; Fangming Liu; Wai Kin Victor Chan; Alex Jinpeng Wang
>
> **备注:** Project homepage: https://csu-jpg.github.io/vja.github.io/
>
> **摘要:** Recent advances in large image editing models have shifted the paradigm from text-driven instructions to vision-prompt editing, where user intent is inferred directly from visual inputs such as marks, arrows, and visual-text prompts. While this paradigm greatly expands usability, it also introduces a critical and underexplored safety risk: the attack surface itself becomes visual. In this work, we propose Vision-Centric Jailbreak Attack (VJA), the first visual-to-visual jailbreak attack that conveys malicious instructions purely through visual inputs. To systematically study this emerging threat, we introduce IESBench, a safety-oriented benchmark for image editing models. Extensive experiments on IESBench demonstrate that VJA effectively compromises state-of-the-art commercial models, achieving attack success rates of up to 80.9% on Nano Banana Pro and 70.1% on GPT-Image-1.5. To mitigate this vulnerability, we propose a training-free defense based on introspective multimodal reasoning, which substantially improves the safety of poorly aligned models to a level comparable with commercial systems, without auxiliary guard models and with negligible computational overhead. Our findings expose new vulnerabilities, provide both a benchmark and practical defense to advance safe and trustworthy modern image editing systems. Warning: This paper contains offensive images created by large image editing models.
>
---
#### [new 017] DMP-3DAD: Cross-Category 3D Anomaly Detection via Realistic Depth Map Projection with Few Normal Samples
- **分类: cs.CV**

- **简介: 该论文属于跨类别3D异常检测任务，解决少量正常样本下的异常识别问题。提出DMP-3DAD框架，通过深度图投影和特征相似性实现无需训练的检测。**

- **链接: [https://arxiv.org/pdf/2602.10806v1](https://arxiv.org/pdf/2602.10806v1)**

> **作者:** Zi Wang; Katsuya Hotta; Koichiro Kamide; Yawen Zou; Jianjian Qin; Chao Zhang; Jun Yu
>
> **摘要:** Cross-category anomaly detection for 3D point clouds aims to determine whether an unseen object belongs to a target category using only a few normal examples. Most existing methods rely on category-specific training, which limits their flexibility in few-shot scenarios. In this paper, we propose DMP-3DAD, a training-free framework for cross-category 3D anomaly detection based on multi-view realistic depth map projection. Specifically, by converting point clouds into a fixed set of realistic depth images, our method leverages a frozen CLIP visual encoder to extract multi-view representations and performs anomaly detection via weighted feature similarity, which does not require any fine-tuning or category-dependent adaptation. Extensive experiments on the ShapeNetPart dataset demonstrate that DMP-3DAD achieves state-of-the-art performance under few-shot setting. The results show that the proposed approach provides a simple yet effective solution for practical cross-category 3D anomaly detection.
>
---
#### [new 018] ERGO: Excess-Risk-Guided Optimization for High-Fidelity Monocular 3D Gaussian Splatting
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出ERGO框架，解决单目3D高保真重建问题，通过分解优化损失提升对噪声监督信号的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.10278v1](https://arxiv.org/pdf/2602.10278v1)**

> **作者:** Zehua Ma; Hanhui Li; Zhenyu Xie; Xiaonan Luo; Michael Kampffmeyer; Feng Gao; Xiaodan Liang
>
> **摘要:** Generating 3D content from a single image remains a fundamentally challenging and ill-posed problem due to the inherent absence of geometric and textural information in occluded regions. While state-of-the-art generative models can synthesize auxiliary views to provide additional supervision, these views inevitably contain geometric inconsistencies and textural misalignments that propagate and amplify artifacts during 3D reconstruction. To effectively harness these imperfect supervisory signals, we propose an adaptive optimization framework guided by excess risk decomposition, termed ERGO. Specifically, ERGO decomposes the optimization losses in 3D Gaussian splatting into two components, i.e., excess risk that quantifies the suboptimality gap between current and optimal parameters, and Bayes error that models the irreducible noise inherent in synthesized views. This decomposition enables ERGO to dynamically estimate the view-specific excess risk and adaptively adjust loss weights during optimization. Furthermore, we introduce geometry-aware and texture-aware objectives that complement the excess-risk-derived weighting mechanism, establishing a synergistic global-local optimization paradigm. Consequently, ERGO demonstrates robustness against supervision noise while consistently enhancing both geometric fidelity and textural quality of the reconstructed 3D content. Extensive experiments on the Google Scanned Objects dataset and the OmniObject3D dataset demonstrate the superiority of ERGO over existing state-of-the-art methods.
>
---
#### [new 019] Healthy Harvests: A Comparative Look at Guava Disease Classification Using InceptionV3
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像分类任务，旨在解决番石榴疾病识别问题。通过使用InceptionV3和ResNet50模型进行分类，并采用数据增强和SHAP分析提升模型性能与可解释性。**

- **链接: [https://arxiv.org/pdf/2602.10967v1](https://arxiv.org/pdf/2602.10967v1)**

> **作者:** Samanta Ghosh; Shaila Afroz Anika; Umma Habiba Ahmed; B. M. Shahria Alam; Mohammad Tahmid Noor; Nishat Tasnim Niloy
>
> **备注:** 6 pages, 13 figures, his is the author's accepted manuscript of a paper accepted for publication in the Proceedings of the 16th International IEEE Conference on Computing, Communication and Networking Technologies (ICCCNT 2025). The final published version will be available via IEEE Xplore
>
> **摘要:** Guava fruits often suffer from many diseases. This can harm fruit quality and fruit crop yield. Early identification is important for minimizing damage and ensuring fruit health. This study focuses on 3 different categories for classifying diseases. These are Anthracnose, Fruit flies, and Healthy fruit. The data set used in this study is collected from Mendeley Data. This dataset contains 473 original images of Guava. These images vary in size and format. The original dataset was resized to 256x256 pixels with RGB color mode for better consistency. After this, the Data augmentation process is applied to improve the dataset by generating variations of the original images. The augmented dataset consists of 3784 images using advanced preprocessing techniques. Two deep learning models were implemented to classify the images. The InceptionV3 model is well known for its advanced framework. These apply multiple convolutional filters for obtaining different features effectively. On the other hand, the ResNet50 model helps to train deeper networks by using residual learning. The InceptionV3 model achieved the impressive accuracy of 98.15%, and ResNet50got 94.46% accuracy. Data mixing methods such as CutMix and MixUp were applied to enhance the model's robustness. The confusion matrix was used to evaluate the overall model performance of both InceptionV3 and Resnet50. Additionally, SHAP analysis is used to improve interpretability, which helps to find the significant parts of the image for the model prediction. This study purposes to highlight how advanced models enhan
>
---
#### [new 020] Text-to-Vector Conversion for Residential Plan Design
- **分类: cs.CV**

- **简介: 该论文属于文本到矢量的转换任务，旨在解决从文本生成高质量矢量住宅图纸的问题。通过改进算法提升视觉质量和结构化程度。**

- **链接: [https://arxiv.org/pdf/2602.10757v1](https://arxiv.org/pdf/2602.10757v1)**

> **作者:** Egor Bazhenov; Stepan Kasai; Viacheslav Shalamov; Valeria Efimova
>
> **备注:** 4 pages, 1 figure
>
> **摘要:** Computer graphics, comprising both raster and vector components, is a fundamental part of modern science, industry, and digital communication. While raster graphics offer ease of use, its pixel-based structure limits scalability. Vector graphics, defined by mathematical primitives, provides scalability without quality loss, however, it is more complex to produce. For design and architecture, the versatility of vector graphics is paramount, despite its computational demands. This paper introduces a novel method for generating vector residential plans from textual descriptions. Our approach surpasses existing solutions by approximately 5% in CLIPScore-based visual quality, benefiting from its inherent handling of right angles and flexible settings. Additionally, we present a new algorithm for vectorizing raster plans into structured vector images. Such images have a better CLIPscore compared to others by about 4%.
>
---
#### [new 021] FastUSP: A Multi-Level Collaborative Acceleration Framework for Distributed Diffusion Model Inference
- **分类: cs.CV**

- **简介: 该论文属于分布式扩散模型推理任务，旨在解决USP框架中的效率问题。提出FastUSP框架，通过多级优化提升推理速度。**

- **链接: [https://arxiv.org/pdf/2602.10940v1](https://arxiv.org/pdf/2602.10940v1)**

> **作者:** Guandong Li
>
> **摘要:** Large-scale diffusion models such as FLUX (12B parameters) and Stable Diffusion 3 (8B parameters) require multi-GPU parallelism for efficient inference. Unified Sequence Parallelism (USP), which combines Ulysses and Ring attention mechanisms, has emerged as the state-of-the-art approach for distributed attention computation. However, existing USP implementations suffer from significant inefficiencies including excessive kernel launch overhead and suboptimal computation-communication scheduling. In this paper, we propose \textbf{FastUSP}, a multi-level optimization framework that integrates compile-level optimization (graph compilation with CUDA Graphs and computation-communication reordering), communication-level optimization (FP8 quantized collective communication), and operator-level optimization (pipelined Ring attention with double buffering). We evaluate FastUSP on FLUX (12B) and Qwen-Image models across 2, 4, and 8 NVIDIA RTX 5090 GPUs. On FLUX, FastUSP achieves consistent \textbf{1.12$\times$--1.16$\times$} end-to-end speedup over baseline USP, with compile-level optimization contributing the dominant improvement. On Qwen-Image, FastUSP achieves \textbf{1.09$\times$} speedup on 2 GPUs; on 4--8 GPUs, we identify a PyTorch Inductor compatibility limitation with Ring attention that prevents compile optimization, while baseline USP scales to 1.30$\times$--1.46$\times$ of 2-GPU performance. We further provide a detailed analysis of the performance characteristics of distributed diffusion inference, revealing that kernel launch overhead -- rather than communication latency -- is the primary bottleneck on modern high-bandwidth GPU interconnects.
>
---
#### [new 022] HII-DPO: Eliminate Hallucination via Accurate Hallucination-Inducing Counterfactual Images
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型领域，旨在解决模型 hallucination 问题。通过生成诱导幻觉的图像，揭示场景依赖的幻觉模式，并构建数据集提升模型对齐效果。**

- **链接: [https://arxiv.org/pdf/2602.10425v1](https://arxiv.org/pdf/2602.10425v1)**

> **作者:** Yilin Yang; Zhenghui Guo; Yuke Wang; Omprakash Gnawali; Sheng Di; Chengming Zhang
>
> **摘要:** Large Vision-Language Models (VLMs) have achieved remarkable success across diverse multimodal tasks but remain vulnerable to hallucinations rooted in inherent language bias. Despite recent progress, existing hallucination mitigation methods often overlook the underlying hallucination patterns driven by language bias. In this work, we design a novel pipeline to accurately synthesize Hallucination-Inducing Images (HIIs). Using synthesized HIIs, we reveal a consistent scene-conditioned hallucination pattern: models tend to mention objects that are highly typical of the scene even when visual evidence is removed. To quantify the susceptibility of VLMs to this hallucination pattern, we establish the Masked-Object-Hallucination (MOH) benchmark to rigorously evaluate existing state-of-the-art alignment frameworks. Finally, we leverage HIIs to construct high-quality preference datasets for fine-grained alignment. Experimental results demonstrate that our approach effectively mitigates hallucinations while preserving general model capabilities. Specifically, our method achieves up to a 38% improvement over the current state-of-the-art on standard hallucination benchmarks.
>
---
#### [new 023] MetaphorStar: Image Metaphor Understanding and Reasoning with End-to-End Visual Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.CY**

- **简介: 该论文属于图像隐喻理解任务，旨在解决AI系统在文化、情感和上下文理解上的不足。提出MetaphorStar框架，通过视觉强化学习提升图像隐喻推理能力。**

- **链接: [https://arxiv.org/pdf/2602.10575v1](https://arxiv.org/pdf/2602.10575v1)**

> **作者:** Chenhao Zhang; Yazhe Niu; Hongsheng Li
>
> **备注:** 14 pages, 4 figures, 11 tables; Code: https://github.com/MING-ZCH/MetaphorStar, Model & Dataset: https://huggingface.co/collections/MING-ZCH/metaphorstar
>
> **摘要:** Metaphorical comprehension in images remains a critical challenge for Nowadays AI systems. While Multimodal Large Language Models (MLLMs) excel at basic Visual Question Answering (VQA), they consistently struggle to grasp the nuanced cultural, emotional, and contextual implications embedded in visual content. This difficulty stems from the task's demand for sophisticated multi-hop reasoning, cultural context, and Theory of Mind (ToM) capabilities, which current models lack. To fill this gap, we propose MetaphorStar, the first end-to-end visual reinforcement learning (RL) framework for image implication tasks. Our framework includes three core components: the fine-grained dataset TFQ-Data, the visual RL method TFQ-GRPO, and the well-structured benchmark TFQ-Bench. Our fully open-source MetaphorStar family, trained using TFQ-GRPO on TFQ-Data, significantly improves performance by an average of 82.6% on the image implication benchmarks. Compared with 20+ mainstream MLLMs, MetaphorStar-32B achieves state-of-the-art (SOTA) on Multiple-Choice Question and Open-Style Question, significantly outperforms the top closed-source model Gemini-3.0-pro on True-False Question. Crucially, our experiments reveal that learning image implication tasks improves the general understanding ability, especially the complex visual reasoning ability. We further provide a systematic analysis of model parameter scaling, training data scaling, and the impact of different model architectures and training strategies, demonstrating the broad applicability of our method. We open-sourced all model weights, datasets, and method code at https://metaphorstar.github.io.
>
---
#### [new 024] PuriLight: A Lightweight Shuffle and Purification Framework for Monocular Depth Estimation
- **分类: cs.CV**

- **简介: 该论文属于单目深度估计任务，旨在解决计算效率与细节保留的矛盾。提出PuriLight框架，通过三个模块实现轻量且精确的特征提取与处理。**

- **链接: [https://arxiv.org/pdf/2602.11066v1](https://arxiv.org/pdf/2602.11066v1)**

> **作者:** Yujie Chen; Li Zhang; Xiaomeng Chu; Tian Zhang
>
> **备注:** 8 pages, 6figures, accepted by European Conference on Artificial Intelligence (ECAI2025)
>
> **摘要:** We propose PuriLight, a lightweight and efficient framework for self-supervised monocular depth estimation, to address the dual challenges of computational efficiency and detail preservation. While recent advances in self-supervised depth estimation have reduced reliance on ground truth supervision, existing approaches remain constrained by either bulky architectures compromising practicality or lightweight models sacrificing structural precision. These dual limitations underscore the critical need to develop lightweight yet structurally precise architectures. Our framework addresses these limitations through a three-stage architecture incorporating three novel modules: the Shuffle-Dilation Convolution (SDC) module for local feature extraction, the Rotation-Adaptive Kernel Attention (RAKA) module for hierarchical feature enhancement, and the Deep Frequency Signal Purification (DFSP) module for global feature purification. Through effective collaboration, these modules enable PuriLight to achieve both lightweight and accurate feature extraction and processing. Extensive experiments demonstrate that PuriLight achieves state-of-the-art performance with minimal training parameters while maintaining exceptional computational efficiency. Codes will be available at https://github.com/ishrouder/PuriLight.
>
---
#### [new 025] Characterizing and Optimizing the Spatial Kernel of Multi Resolution Hash Encodings
- **分类: cs.CV**

- **简介: 该论文研究多分辨率哈希编码的物理特性，解决其空间行为理解不足的问题。通过分析点扩散函数，提出R-MHE优化架构，提升性能并减少各向异性。**

- **链接: [https://arxiv.org/pdf/2602.10495v1](https://arxiv.org/pdf/2602.10495v1)**

> **作者:** Tianxiang Dai; Jonathan Fan
>
> **备注:** ICLR 2026 (Poster); LaTeX source; 11 figures; 7 tables
>
> **摘要:** Multi-Resolution Hash Encoding (MHE), the foundational technique behind Instant Neural Graphics Primitives, provides a powerful parameterization for neural fields. However, its spatial behavior lacks rigorous understanding from a physical systems perspective, leading to reliance on heuristics for hyperparameter selection. This work introduces a novel analytical approach that characterizes MHE by examining its Point Spread Function (PSF), which is analogous to the Green's function of the system. This methodology enables a quantification of the encoding's spatial resolution and fidelity. We derive a closed-form approximation for the collision-free PSF, uncovering inherent grid-induced anisotropy and a logarithmic spatial profile. We establish that the idealized spatial bandwidth, specifically the Full Width at Half Maximum (FWHM), is determined by the average resolution, $N_{\text{avg}}$. This leads to a counterintuitive finding: the effective resolution of the model is governed by the broadened empirical FWHM (and therefore $N_{\text{avg}}$), rather than the finest resolution $N_{\max}$, a broadening effect we demonstrate arises from optimization dynamics. Furthermore, we analyze the impact of finite hash capacity, demonstrating how collisions introduce speckle noise and degrade the Signal-to-Noise Ratio (SNR). Leveraging these theoretical insights, we propose Rotated MHE (R-MHE), an architecture that applies distinct rotations to the input coordinates at each resolution level. R-MHE mitigates anisotropy while maintaining the efficiency and parameter count of the original MHE. This study establishes a methodology based on physical principles that moves beyond heuristics to characterize and optimize MHE.
>
---
#### [new 026] 1%>100%: High-Efficiency Visual Adapter with Complex Linear Projection Optimization
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉模型适应任务，解决高效微调问题。提出CoLin适配器，仅用1%参数实现高效训练，优于全微调和经典delta-tuning。**

- **链接: [https://arxiv.org/pdf/2602.10513v1](https://arxiv.org/pdf/2602.10513v1)**

> **作者:** Dongshuo Yin; Xue Yang; Deng-Ping Fan; Shi-Min Hu
>
> **摘要:** Deploying vision foundation models typically relies on efficient adaptation strategies, whereas conventional full fine-tuning suffers from prohibitive costs and low efficiency. While delta-tuning has proven effective in boosting the performance and efficiency of LLMs during adaptation, its advantages cannot be directly transferred to the fine-tuning pipeline of vision foundation models. To push the boundaries of adaptation efficiency for vision tasks, we propose an adapter with Complex Linear Projection Optimization (CoLin). For architecture, we design a novel low-rank complex adapter that introduces only about 1% parameters to the backbone. For efficiency, we theoretically prove that low-rank composite matrices suffer from severe convergence issues during training, and address this challenge with a tailored loss. Extensive experiments on object detection, segmentation, image classification, and rotated object detection (remote sensing scenario) demonstrate that CoLin outperforms both full fine-tuning and classical delta-tuning approaches with merely 1% parameters for the first time, providing a novel and efficient solution for deployment of vision foundation models. We release the code on https://github.com/DongshuoYin/CoLin.
>
---
#### [new 027] Beyond VLM-Based Rewards: Diffusion-Native Latent Reward Modeling
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于扩散模型的偏好优化任务，旨在解决VLM奖励计算成本高和域不匹配问题。提出DiNa-LRM，在扩散状态中直接进行偏好学习，提升效率与效果。**

- **链接: [https://arxiv.org/pdf/2602.11146v1](https://arxiv.org/pdf/2602.11146v1)**

> **作者:** Gongye Liu; Bo Yang; Yida Zhi; Zhizhou Zhong; Lei Ke; Didan Deng; Han Gao; Yongxiang Huang; Kaihao Zhang; Hongbo Fu; Wenhan Luo
>
> **备注:** Code: https://github.com/HKUST-C4G/diffusion-rm
>
> **摘要:** Preference optimization for diffusion and flow-matching models relies on reward functions that are both discriminatively robust and computationally efficient. Vision-Language Models (VLMs) have emerged as the primary reward provider, leveraging their rich multimodal priors to guide alignment. However, their computation and memory cost can be substantial, and optimizing a latent diffusion generator through a pixel-space reward introduces a domain mismatch that complicates alignment. In this paper, we propose DiNa-LRM, a diffusion-native latent reward model that formulates preference learning directly on noisy diffusion states. Our method introduces a noise-calibrated Thurstone likelihood with diffusion-noise-dependent uncertainty. DiNa-LRM leverages a pretrained latent diffusion backbone with a timestep-conditioned reward head, and supports inference-time noise ensembling, providing a diffusion-native mechanism for test-time scaling and robust rewarding. Across image alignment benchmarks, DiNa-LRM substantially outperforms existing diffusion-based reward baselines and achieves performance competitive with state-of-the-art VLMs at a fraction of the computational cost. In preference optimization, we demonstrate that DiNa-LRM improves preference optimization dynamics, enabling faster and more resource-efficient model alignment.
>
---
#### [new 028] AMAP-APP: Efficient Segmentation and Morphometry Quantification of Fluorescent Microscopy Images of Podocytes
- **分类: cs.CV**

- **简介: 该论文属于图像分割与形态量化任务，旨在解决传统方法计算量大、无界面及依赖Linux的问题。通过优化算法提升效率并提高精度，实现跨平台应用。**

- **链接: [https://arxiv.org/pdf/2602.10663v1](https://arxiv.org/pdf/2602.10663v1)**

> **作者:** Arash Fatehi; David Unnersjö-Jess; Linus Butt; Noémie Moreau; Thomas Benzing; Katarzyna Bozek
>
> **摘要:** Background: Automated podocyte foot process quantification is vital for kidney research, but the established "Automatic Morphological Analysis of Podocytes" (AMAP) method is hindered by high computational demands, a lack of a user interface, and Linux dependency. We developed AMAP-APP, a cross-platform desktop application designed to overcome these barriers. Methods: AMAP-APP optimizes efficiency by replacing intensive instance segmentation with classic image processing while retaining the original semantic segmentation model. It introduces a refined Region of Interest (ROI) algorithm to improve precision. Validation involved 365 mouse and human images (STED and confocal), benchmarking performance against the original AMAP via Pearson correlation and Two One-Sided T-tests (TOST). Results: AMAP-APP achieved a 147-fold increase in processing speed on consumer hardware. Morphometric outputs (area, perimeter, circularity, and slit diaphragm density) showed high correlation (r>0.90) and statistical equivalence (TOST P<0.05) to the original method. Additionally, the new ROI algorithm demonstrated superior accuracy compared to the original, showing reduced deviation from manual delineations. Conclusion: AMAP-APP democratizes deep learning-based podocyte morphometry. By eliminating the need for high-performance computing clusters and providing a user-friendly interface for Windows, macOS, and Linux, it enables widespread adoption in nephrology research and potential clinical diagnostics.
>
---
#### [new 029] AugVLA-3D: Depth-Driven Feature Augmentation for Vision-Language-Action Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在解决2D观测与3D决策之间的差距。通过引入深度估计和动作先验，增强3D特征表示，提升模型的感知与控制能力。**

- **链接: [https://arxiv.org/pdf/2602.10698v1](https://arxiv.org/pdf/2602.10698v1)**

> **作者:** Zhifeng Rao; Wenlong Chen; Lei Xie; Xia Hua; Dongfu Yin; Zhen Tian; F. Richard Yu
>
> **摘要:** Vision-Language-Action (VLA) models have recently achieved remarkable progress in robotic perception and control, yet most existing approaches primarily rely on VLM trained using 2D images, which limits their spatial understanding and action grounding in complex 3D environments. To address this limitation, we propose a novel framework that integrates depth estimation into VLA models to enrich 3D feature representations. Specifically, we employ a depth estimation baseline called VGGT to extract geometry-aware 3D cues from standard RGB inputs, enabling efficient utilization of existing large-scale 2D datasets while implicitly recovering 3D structural information. To further enhance the reliability of these depth-derived features, we introduce a new module called action assistant, which constrains the learned 3D representations with action priors and ensures their consistency with downstream control tasks. By fusing the enhanced 3D features with conventional 2D visual tokens, our approach significantly improves the generalization ability and robustness of VLA models. Experimental results demonstrate that the proposed method not only strengthens perception in geometrically ambiguous scenarios but also leads to superior action prediction accuracy. This work highlights the potential of depth-driven data augmentation and auxiliary expert supervision for bridging the gap between 2D observations and 3D-aware decision-making in robotic systems.
>
---
#### [new 030] Flow caching for autoregressive video generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成任务，解决 autoregressive 模型生成视频速度慢的问题。提出 FlowCache 框架，通过分块缓存策略提升生成效率。**

- **链接: [https://arxiv.org/pdf/2602.10825v1](https://arxiv.org/pdf/2602.10825v1)**

> **作者:** Yuexiao Ma; Xuzhe Zheng; Jing Xu; Xiwei Xu; Feng Ling; Xiawu Zheng; Huafeng Kuang; Huixia Li; Xing Wang; Xuefeng Xiao; Fei Chao; Rongrong Ji
>
> **摘要:** Autoregressive models, often built on Transformer architectures, represent a powerful paradigm for generating ultra-long videos by synthesizing content in sequential chunks. However, this sequential generation process is notoriously slow. While caching strategies have proven effective for accelerating traditional video diffusion models, existing methods assume uniform denoising across all frames-an assumption that breaks down in autoregressive models where different video chunks exhibit varying similarity patterns at identical timesteps. In this paper, we present FlowCache, the first caching framework specifically designed for autoregressive video generation. Our key insight is that each video chunk should maintain independent caching policies, allowing fine-grained control over which chunks require recomputation at each timestep. We introduce a chunkwise caching strategy that dynamically adapts to the unique denoising characteristics of each chunk, complemented by a joint importance-redundancy optimized KV cache compression mechanism that maintains fixed memory bounds while preserving generation quality. Our method achieves remarkable speedups of 2.38 times on MAGI-1 and 6.7 times on SkyReels-V2, with negligible quality degradation (VBench: 0.87 increase and 0.79 decrease respectively). These results demonstrate that FlowCache successfully unlocks the potential of autoregressive models for real-time, ultra-long video generation-establishing a new benchmark for efficient video synthesis at scale. The code is available at https://github.com/mikeallen39/FlowCache.
>
---
#### [new 031] Comp2Comp: Open-Source Software with FDA-Cleared Artificial Intelligence Algorithms for Computed Tomography Image Analysis
- **分类: cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决开源算法缺乏验证、商业方案不透明的问题。作者开发并验证了两个FDA认证的深度学习模型，用于CT图像的腹部主动脉和骨密度分析。**

- **链接: [https://arxiv.org/pdf/2602.10364v1](https://arxiv.org/pdf/2602.10364v1)**

> **作者:** Adrit Rao; Malte Jensen; Andrea T. Fisher; Louis Blankemeier; Pauline Berens; Arash Fereydooni; Seth Lirette; Eren Alkan; Felipe C. Kitamura; Juan M. Zambrano Chaves; Eduardo Reis; Arjun Desai; Marc H. Willis; Jason Hom; Andrew Johnston; Leon Lenchik; Robert D. Boutin; Eduardo M. J. M. Farina; Augusto S. Serpa; Marcelo S. Takahashi; Jordan Perchik; Steven A. Rothenberg; Jamie L. Schroeder; Ross Filice; Leonardo K. Bittencourt; Hari Trivedi; Marly van Assen; John Mongan; Kimberly Kallianos; Oliver Aalami; Akshay S. Chaudhari
>
> **备注:** Adrit Rao, Malte Jensen, Andrea T. Fisher, Louis Blankemeier: Co-first authors. Oliver Aalami, Akshay S. Chaudhari: Co-senior authors
>
> **摘要:** Artificial intelligence allows automatic extraction of imaging biomarkers from already-acquired radiologic images. This paradigm of opportunistic imaging adds value to medical imaging without additional imaging costs or patient radiation exposure. However, many open-source image analysis solutions lack rigorous validation while commercial solutions lack transparency, leading to unexpected failures when deployed. Here, we report development and validation for two of the first fully open-sourced, FDA-510(k)-cleared deep learning pipelines to mitigate both challenges: Abdominal Aortic Quantification (AAQ) and Bone Mineral Density (BMD) estimation are both offered within the Comp2Comp package for opportunistic analysis of computed tomography scans. AAQ segments the abdominal aorta to assess aneurysm size; BMD segments vertebral bodies to estimate trabecular bone density and osteoporosis risk. AAQ-derived maximal aortic diameters were compared against radiologist ground-truth measurements on 258 patient scans enriched for abdominal aortic aneurysms from four external institutions. BMD binary classifications (low vs. normal bone density) were compared against concurrent DXA scan ground truths obtained on 371 patient scans from four external institutions. AAQ had an overall mean absolute error of 1.57 mm (95% CI 1.38-1.80 mm). BMD had a sensitivity of 81.0% (95% CI 74.0-86.8%) and specificity of 78.4% (95% CI 72.3-83.7%). Comp2Comp AAQ and BMD demonstrated sufficient accuracy for clinical use. Open-sourcing these algorithms improves transparency of typically opaque FDA clearance processes, allows hospitals to test the algorithms before cumbersome clinical pilots, and provides researchers with best-in-class methods.
>
---
#### [new 032] Why Does RL Generalize Better Than SFT? A Data-Centric Perspective on VLM Post-Training
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究视觉语言模型后训练中的泛化问题，旨在解释为何强化学习比监督微调泛化更好。通过分析数据难度，提出DC-SFT方法提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.10815v1](https://arxiv.org/pdf/2602.10815v1)**

> **作者:** Aojun Lu; Tao Feng; Hangjie Yuan; Wei Li; Yanan Sun
>
> **摘要:** The adaptation of large-scale Vision-Language Models (VLMs) through post-training reveals a pronounced generalization gap: models fine-tuned with Reinforcement Learning (RL) consistently achieve superior out-of-distribution (OOD) performance compared to those trained with Supervised Fine-Tuning (SFT). This paper posits a data-centric explanation for this phenomenon, contending that RL's generalization advantage arises from an implicit data filtering mechanism that inherently prioritizes medium-difficulty training samples. To test this hypothesis, we systematically evaluate the OOD generalization of SFT models across training datasets of varying difficulty levels. Our results confirm that data difficulty is a critical factor, revealing that training on hard samples significantly degrades OOD performance. Motivated by this finding, we introduce Difficulty-Curated SFT (DC-SFT), a straightforward method that explicitly filters the training set based on sample difficulty. Experiments show that DC-SFT not only substantially enhances OOD generalization over standard SFT, but also surpasses the performance of RL-based training, all while providing greater stability and computational efficiency. This work offers a data-centric account of the OOD generalization gap in VLMs and establishes a more efficient pathway to achieving robust generalization. Code is available at https://github.com/byyx666/DC-SFT.
>
---
#### [new 033] Beyond Closed-Pool Video Retrieval: A Benchmark and Agent Framework for Real-World Video Search and Moment Localization
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出RVMS-Bench基准和RACLO框架，解决真实场景下基于模糊记忆的视频检索与时刻定位问题。**

- **链接: [https://arxiv.org/pdf/2602.10159v1](https://arxiv.org/pdf/2602.10159v1)**

> **作者:** Tao Yu; Yujia Yang; Haopeng Jin; Junhao Gong; Xinlong Chen; Yuxuan Zhou; Shanbin Zhang; Jiabing Yang; Xinming Wang; Hongzhu Yi; Ping Nie; Kai Zou; Zhang Zhang; Yan Huang; Liang Wang; Yeshani; Ruiwen Tao; Jin Ma; Haijin Liang; Jinwen Luo
>
> **备注:** 49 pages, 9 figures
>
> **摘要:** Traditional video retrieval benchmarks focus on matching precise descriptions to closed video pools, failing to reflect real-world searches characterized by fuzzy, multi-dimensional memories on the open web. We present \textbf{RVMS-Bench}, a comprehensive system for evaluating real-world video memory search. It consists of \textbf{1,440 samples} spanning \textbf{20 diverse categories} and \textbf{four duration groups}, sourced from \textbf{real-world open-web videos}. RVMS-Bench utilizes a hierarchical description framework encompassing \textbf{Global Impression, Key Moment, Temporal Context, and Auditory Memory} to mimic realistic multi-dimensional search cues, with all samples strictly verified via a human-in-the-loop protocol. We further propose \textbf{RACLO}, an agentic framework that employs abductive reasoning to simulate the human ``Recall-Search-Verify'' cognitive process, effectively addressing the challenge of searching for videos via fuzzy memories in the real world. Experiments reveal that existing MLLMs still demonstrate insufficient capabilities in real-world Video Retrieval and Moment Localization based on fuzzy memories. We believe this work will facilitate the advancement of video retrieval robustness in real-world unstructured scenarios.
>
---
#### [new 034] Flow Matching with Uncertainty Quantification and Guidance
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于生成模型任务，旨在解决样本质量不一致问题。提出UA-Flow方法，同时预测速度场和不确定性，提升生成质量与可靠性。**

- **链接: [https://arxiv.org/pdf/2602.10326v1](https://arxiv.org/pdf/2602.10326v1)**

> **作者:** Juyeop Han; Lukas Lao Beyer; Sertac Karaman
>
> **摘要:** Despite the remarkable success of sampling-based generative models such as flow matching, they can still produce samples of inconsistent or degraded quality. To assess sample reliability and generate higher-quality outputs, we propose uncertainty-aware flow matching (UA-Flow), a lightweight extension of flow matching that predicts the velocity field together with heteroscedastic uncertainty. UA-Flow estimates per-sample uncertainty by propagating velocity uncertainty through the flow dynamics. These uncertainty estimates act as a reliability signal for individual samples, and we further use them to steer generation via uncertainty-aware classifier guidance and classifier-free guidance. Experiments on image generation show that UA-Flow produces uncertainty signals more highly correlated with sample fidelity than baseline methods, and that uncertainty-guided sampling further improves generation quality.
>
---
#### [new 035] FGAA-FPN: Foreground-Guided Angle-Aware Feature Pyramid Network for Oriented Object Detection
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决遥感图像中方向物体检测的挑战，通过引入前景引导和角度感知模块提升检测性能。**

- **链接: [https://arxiv.org/pdf/2602.10710v1](https://arxiv.org/pdf/2602.10710v1)**

> **作者:** Jialin Ma
>
> **备注:** Submitted to The Visual Computer
>
> **摘要:** With the increasing availability of high-resolution remote sensing and aerial imagery, oriented object detection has become a key capability for geographic information updating, maritime surveillance, and disaster response. However, it remains challenging due to cluttered backgrounds, severe scale variation, and large orientation changes. Existing approaches largely improve performance through multi-scale feature fusion with feature pyramid networks or contextual modeling with attention, but they often lack explicit foreground modeling and do not leverage geometric orientation priors, which limits feature discriminability. To overcome these limitations, we propose FGAA-FPN, a Foreground-Guided Angle-Aware Feature Pyramid Network for oriented object detection. FGAA-FPN is built on a hierarchical functional decomposition that accounts for the distinct spatial resolution and semantic abstraction across pyramid levels, thereby strengthening multi-scale representations. Concretely, a Foreground-Guided Feature Modulation module learns foreground saliency under weak supervision to enhance object regions and suppress background interference in low-level features. In parallel, an Angle-Aware Multi-Head Attention module encodes relative orientation relationships to guide global interactions among high-level semantic features. Extensive experiments on DOTA v1.0 and DOTA v1.5 demonstrate that FGAA-FPN achieves state-of-the-art results, reaching 75.5% and 68.3% mAP, respectively.
>
---
#### [new 036] Chain-of-Look Spatial Reasoning for Dense Surgical Instrument Counting
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于密集手术器械计数任务，旨在解决复杂场景下器械密集导致的计数困难问题。提出Chain-of-Look框架，通过结构化视觉链提升计数准确性。**

- **链接: [https://arxiv.org/pdf/2602.11024v1](https://arxiv.org/pdf/2602.11024v1)**

> **作者:** Rishikesh Bhyri; Brian R Quaranto; Philip J Seger; Kaity Tung; Brendan Fox; Gene Yang; Steven D. Schwaitzberg; Junsong Yuan; Nan Xi; Peter C W Kim
>
> **备注:** Accepted to WACV 2026. This version includes additional authors who contributed during the rebuttal phase
>
> **摘要:** Accurate counting of surgical instruments in Operating Rooms (OR) is a critical prerequisite for ensuring patient safety during surgery. Despite recent progress of large visual-language models and agentic AI, accurately counting such instruments remains highly challenging, particularly in dense scenarios where instruments are tightly clustered. To address this problem, we introduce Chain-of-Look, a novel visual reasoning framework that mimics the sequential human counting process by enforcing a structured visual chain, rather than relying on classic object detection which is unordered. This visual chain guides the model to count along a coherent spatial trajectory, improving accuracy in complex scenes. To further enforce the physical plausibility of the visual chain, we introduce the neighboring loss function, which explicitly models the spatial constraints inherent to densely packed surgical instruments. We also present SurgCount-HD, a new dataset comprising 1,464 high-density surgical instrument images. Extensive experiments demonstrate that our method outperforms state-of-the-art approaches for counting (e.g., CountGD, REC) as well as Multimodality Large Language Models (e.g., Qwen, ChatGPT) in the challenging task of dense surgical instrument counting.
>
---
#### [new 037] Multimodal Information Fusion for Chart Understanding: A Survey of MLLMs -- Evolution, Limitations, and Cognitive Enhancement
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于图表理解任务，旨在解决多模态信息融合问题。通过分析挑战、分类任务与数据集，梳理方法演进，提出未来改进方向。**

- **链接: [https://arxiv.org/pdf/2602.10138v1](https://arxiv.org/pdf/2602.10138v1)**

> **作者:** Zhihang Yi; Jian Zhao; Jiancheng Lv; Tao Wang
>
> **摘要:** Chart understanding is a quintessential information fusion task, requiring the seamless integration of graphical and textual data to extract meaning. The advent of Multimodal Large Language Models (MLLMs) has revolutionized this domain, yet the landscape of MLLM-based chart analysis remains fragmented and lacks systematic organization. This survey provides a comprehensive roadmap of this nascent frontier by structuring the domain's core components. We begin by analyzing the fundamental challenges of fusing visual and linguistic information in charts. We then categorize downstream tasks and datasets, introducing a novel taxonomy of canonical and non-canonical benchmarks to highlight the field's expanding scope. Subsequently, we present a comprehensive evolution of methodologies, tracing the progression from classic deep learning techniques to state-of-the-art MLLM paradigms that leverage sophisticated fusion strategies. By critically examining the limitations of current models, particularly their perceptual and reasoning deficits, we identify promising future directions, including advanced alignment techniques and reinforcement learning for cognitive enhancement. This survey aims to equip researchers and practitioners with a structured understanding of how MLLMs are transforming chart information fusion and to catalyze progress toward more robust and reliable systems.
>
---
#### [new 038] OmniVL-Guard: Towards Unified Vision-Language Forgery Detection and Grounding via Balanced RL
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态伪造检测任务，旨在解决真实场景中文本、图像和视频交织的伪造内容检测与定位问题。提出OmniVL-Guard框架，通过平衡强化学习实现统一检测与定位。**

- **链接: [https://arxiv.org/pdf/2602.10687v1](https://arxiv.org/pdf/2602.10687v1)**

> **作者:** Jinjie Shen; Jing Wu; Yaxiong Wang; Lechao Cheng; Shengeng Tang; Tianrui Hui; Nan Pu; Zhun Zhong
>
> **备注:** 38 pages, DeepFake Detection
>
> **摘要:** Existing forgery detection methods are often limited to uni-modal or bi-modal settings, failing to handle the interleaved text, images, and videos prevalent in real-world misinformation. To bridge this gap, this paper targets to develop a unified framework for omnibus vision-language forgery detection and grounding. In this unified setting, the {interplay} between diverse modalities and the dual requirements of simultaneous detection and localization pose a critical ``difficulty bias`` problem: the simpler veracity classification task tends to dominate the gradients, leading to suboptimal performance in fine-grained grounding during multi-task optimization. To address this challenge, we propose \textbf{OmniVL-Guard}, a balanced reinforcement learning framework for omnibus vision-language forgery detection and grounding. Particularly, OmniVL-Guard comprises two core designs: Self-Evolving CoT Generatio and Adaptive Reward Scaling Policy Optimization (ARSPO). {Self-Evolving CoT Generation} synthesizes high-quality reasoning paths, effectively overcoming the cold-start challenge. Building upon this, {Adaptive Reward Scaling Policy Optimization (ARSPO)} dynamically modulates reward scales and task weights, ensuring a balanced joint optimization. Extensive experiments demonstrate that OmniVL-Guard significantly outperforms state-of-the-art methods and exhibits zero-shot robust generalization across out-of-domain scenarios.
>
---
#### [new 039] Dual-End Consistency Model
- **分类: cs.CV**

- **简介: 该论文属于生成模型任务，旨在解决一致性模型训练不稳定和采样不灵活的问题。提出DE-CM模型，通过选择关键子轨迹提升训练稳定性与生成效果。**

- **链接: [https://arxiv.org/pdf/2602.10764v1](https://arxiv.org/pdf/2602.10764v1)**

> **作者:** Linwei Dong; Ruoyu Guo; Ge Bai; Zehuan Yuan; Yawei Luo; Changqing Zou
>
> **摘要:** The slow iterative sampling nature remains a major bottleneck for the practical deployment of diffusion and flow-based generative models. While consistency models (CMs) represent a state-of-the-art distillation-based approach for efficient generation, their large-scale application is still limited by two key issues: training instability and inflexible sampling. Existing methods seek to mitigate these problems through architectural adjustments or regularized objectives, yet overlook the critical reliance on trajectory selection. In this work, we first conduct an analysis on these two limitations: training instability originates from loss divergence induced by unstable self-supervised term, whereas sampling inflexibility arises from error accumulation. Based on these insights and analysis, we propose the Dual-End Consistency Model (DE-CM) that selects vital sub-trajectory clusters to achieve stable and effective training. DE-CM decomposes the PF-ODE trajectory and selects three critical sub-trajectories as optimization targets. Specifically, our approach leverages continuous-time CMs objectives to achieve few-step distillation and utilizes flow matching as a boundary regularizer to stabilize the training process. Furthermore, we propose a novel noise-to-noisy (N2N) mapping that can map noise to any point, thereby alleviating the error accumulation in the first step. Extensive experimental results show the effectiveness of our method: it achieves a state-of-the-art FID score of 1.70 in one-step generation on the ImageNet 256x256 dataset, outperforming existing CM-based one-step approaches.
>
---
#### [new 040] Chatting with Images for Introspective Visual Thinking
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言理解任务，旨在解决多图像推理中视觉信息与语言语义对齐不足的问题。提出“聊天式图像交互”框架，通过语言引导的特征调制实现更精准的跨模态对齐。**

- **链接: [https://arxiv.org/pdf/2602.11073v1](https://arxiv.org/pdf/2602.11073v1)**

> **作者:** Junfei Wu; Jian Guan; Qiang Liu; Shu Wu; Liang Wang; Wei Wu; Tienie Tan
>
> **摘要:** Current large vision-language models (LVLMs) typically rely on text-only reasoning based on a single-pass visual encoding, which often leads to loss of fine-grained visual information. Recently the proposal of ''thinking with images'' attempts to alleviate this limitation by manipulating images via external tools or code; however, the resulting visual states are often insufficiently grounded in linguistic semantics, impairing effective cross-modal alignment - particularly when visual semantics or geometric relationships must be reasoned over across distant regions or multiple images. To address these challenges, we propose ''chatting with images'', a new framework that reframes visual manipulation as language-guided feature modulation. Under the guidance of expressive language prompts, the model dynamically performs joint re-encoding over multiple image regions, enabling tighter coupling between linguistic reasoning and visual state updates. We instantiate this paradigm in ViLaVT, a novel LVLM equipped with a dynamic vision encoder explicitly designed for such interactive visual reasoning, and trained it with a two-stage curriculum combining supervised fine-tuning and reinforcement learning to promote effective reasoning behaviors. Extensive experiments across eight benchmarks demonstrate that ViLaVT achieves strong and consistent improvements, with particularly pronounced gains on complex multi-image and video-based spatial reasoning tasks.
>
---
#### [new 041] VFGS-Net: Frequency-Guided State-Space Learning for Topology-Preserving Retinal Vessel Segmentation
- **分类: cs.CV**

- **简介: 该论文属于视网膜血管分割任务，旨在解决细小血管保留与全局连通性维持难题。提出VFGS-Net，融合频域感知与空间建模，提升分割精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.10978v1](https://arxiv.org/pdf/2602.10978v1)**

> **作者:** Ruiqi Song; Lei Liu; Ya-Nan Zhang; Chao Wang; Xiaoning Li; Nan Mu
>
> **摘要:** Accurate retinal vessel segmentation is a critical prerequisite for quantitative analysis of retinal images and computer-aided diagnosis of vascular diseases such as diabetic retinopathy. However, the elongated morphology, wide scale variation, and low contrast of retinal vessels pose significant challenges for existing methods, making it difficult to simultaneously preserve fine capillaries and maintain global topological continuity. To address these challenges, we propose the Vessel-aware Frequency-domain and Global Spatial modeling Network (VFGS-Net), an end-to-end segmentation framework that seamlessly integrates frequency-aware feature enhancement, dual-path convolutional representation learning, and bidirectional asymmetric spatial state-space modeling within a unified architecture. Specifically, VFGS-Net employs a dual-path feature convolution module to jointly capture fine-grained local textures and multi-scale contextual semantics. A novel vessel-aware frequency-domain channel attention mechanism is introduced to adaptively reweight spectral components, thereby enhancing vessel-relevant responses in high-level features. Furthermore, at the network bottleneck, we propose a bidirectional asymmetric Mamba2-based spatial modeling block to efficiently capture long-range spatial dependencies and strengthen the global continuity of vascular structures. Extensive experiments on four publicly available retinal vessel datasets demonstrate that VFGS-Net achieves competitive or superior performance compared to state-of-the-art methods. Notably, our model consistently improves segmentation accuracy for fine vessels, complex branching patterns, and low-contrast regions, highlighting its robustness and clinical potential.
>
---
#### [new 042] AurigaNet: A Real-Time Multi-Task Network for Enhanced Urban Driving Perception
- **分类: cs.CV**

- **简介: 论文提出AurigaNet，属于自动驾驶感知任务，解决多任务协同问题。整合目标检测、车道线检测和可行驶区域分割，提升准确性和实时性。**

- **链接: [https://arxiv.org/pdf/2602.10660v1](https://arxiv.org/pdf/2602.10660v1)**

> **作者:** Kiarash Ghasemzadeh; Sedigheh Dehghani
>
> **摘要:** Self-driving cars hold significant potential to reduce traffic accidents, alleviate congestion, and enhance urban mobility. However, developing reliable AI systems for autonomous vehicles remains a substantial challenge. Over the past decade, multi-task learning has emerged as a powerful approach to address complex problems in driving perception. Multi-task networks offer several advantages, including increased computational efficiency, real-time processing capabilities, optimized resource utilization, and improved generalization. In this study, we present AurigaNet, an advanced multi-task network architecture designed to push the boundaries of autonomous driving perception. AurigaNet integrates three critical tasks: object detection, lane detection, and drivable area instance segmentation. The system is trained and evaluated using the BDD100K dataset, renowned for its diversity in driving conditions. Key innovations of AurigaNet include its end-to-end instance segmentation capability, which significantly enhances both accuracy and efficiency in path estimation for autonomous vehicles. Experimental results demonstrate that AurigaNet achieves an 85.2% IoU in drivable area segmentation, outperforming its closest competitor by 0.7%. In lane detection, AurigaNet achieves a remarkable 60.8% IoU, surpassing other models by more than 30%. Furthermore, the network achieves an mAP@0.5:0.95 of 47.6% in traffic object detection, exceeding the next leading model by 2.9%. Additionally, we validate the practical feasibility of AurigaNet by deploying it on embedded devices such as the Jetson Orin NX, where it demonstrates competitive real-time performance. These results underscore AurigaNet's potential as a robust and efficient solution for autonomous driving perception systems. The code can be found here https://github.com/KiaRational/AurigaNet.
>
---
#### [new 043] XSPLAIN: XAI-enabling Splat-based Prototype Learning for Attribute-aware INterpretability
- **分类: cs.CV**

- **简介: 该论文提出XSPLAIN，解决3DGS分类的可解释性问题。通过原型学习和特征解耦，提升模型透明度，增强用户信任。**

- **链接: [https://arxiv.org/pdf/2602.10239v1](https://arxiv.org/pdf/2602.10239v1)**

> **作者:** Dominik Galus; Julia Farganus; Tymoteusz Zapala; Mikołaj Czachorowski; Piotr Borycki; Przemysław Spurek; Piotr Syga
>
> **摘要:** 3D Gaussian Splatting (3DGS) has rapidly become a standard for high-fidelity 3D reconstruction, yet its adoption in multiple critical domains is hindered by the lack of interpretability of the generation models as well as classification of the Splats. While explainability methods exist for other 3D representations, like point clouds, they typically rely on ambiguous saliency maps that fail to capture the volumetric coherence of Gaussian primitives. We introduce XSPLAIN, the first ante-hoc, prototype-based interpretability framework designed specifically for 3DGS classification. Our approach leverages a voxel-aggregated PointNet backbone and a novel, invertible orthogonal transformation that disentangles feature channels for interpretability while strictly preserving the original decision boundaries. Explanations are grounded in representative training examples, enabling intuitive ``this looks like that'' reasoning without any degradation in classification performance. A rigorous user study (N=51) demonstrates a decisive preference for our approach: participants selected XSPLAIN explanations 48.4\% of the time as the best, significantly outperforming baselines $(p<0.001)$, showing that XSPLAIN provides transparency and user trust. The source code for this work is available at: https://github.com/Solvro/ml-splat-xai
>
---
#### [new 044] A Diffusion-Based Generative Prior Approach to Sparse-view Computed Tomography
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于CT图像重建任务，旨在解决稀疏视角下图像质量差的问题。通过结合扩散生成模型与优化算法，提升重建效果。**

- **链接: [https://arxiv.org/pdf/2602.10722v1](https://arxiv.org/pdf/2602.10722v1)**

> **作者:** Davide Evangelista; Pasquale Cascarano; Elena Loli Piccolomini
>
> **备注:** 13 pages, 5 figures, 1 table
>
> **摘要:** The reconstruction of X-rays CT images from sparse or limited-angle geometries is a highly challenging task. The lack of data typically results in artifacts in the reconstructed image and may even lead to object distortions. For this reason, the use of deep generative models in this context has great interest and potential success. In the Deep Generative Prior (DGP) framework, the use of diffusion-based generative models is combined with an iterative optimization algorithm for the reconstruction of CT images from sinograms acquired under sparse geometries, to maintain the explainability of a model-based approach while introducing the generative power of a neural network. There are therefore several aspects that can be further investigated within these frameworks to improve reconstruction quality, such as image generation, the model, and the iterative algorithm used to solve the minimization problem, for which we propose modifications with respect to existing approaches. The results obtained even under highly sparse geometries are very promising, although further research is clearly needed in this direction.
>
---
#### [new 045] Eliminating VAE for Fast and High-Resolution Generative Detail Restoration
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率任务，旨在解决扩散模型推理慢、内存占用高的问题。通过移除VAE并引入多阶段对抗蒸馏等方法，提升速度与内存效率。**

- **链接: [https://arxiv.org/pdf/2602.10630v1](https://arxiv.org/pdf/2602.10630v1)**

> **作者:** Yan Wang; Shijie Zhao; Junlin Li; Li Zhang
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Diffusion models have attained remarkable breakthroughs in the real-world super-resolution (SR) task, albeit at slow inference and high demand on devices. To accelerate inference, recent works like GenDR adopt step distillation to minimize the step number to one. However, the memory boundary still restricts the maximum processing size, necessitating tile-by-tile restoration of high-resolution images. Through profiling the pipeline, we pinpoint that the variational auto-encoder (VAE) is the bottleneck of latency and memory. To completely solve the problem, we leverage pixel-(un)shuffle operations to eliminate the VAE, reversing the latent-based GenDR to pixel-space GenDR-Pix. However, upscale with x8 pixelshuffle may induce artifacts of repeated patterns. To alleviate the distortion, we propose a multi-stage adversarial distillation to progressively remove the encoder and decoder. Specifically, we utilize generative features from the previous stage models to guide adversarial discrimination. Moreover, we propose random padding to augment generative features and avoid discriminator collapse. We also introduce a masked Fourier space loss to penalize the outliers of amplitude. To improve inference performance, we empirically integrate a padding-based self-ensemble with classifier-free guidance to improve inference scaling. Experimental results show that GenDR-Pix performs 2.8x acceleration and 60% memory-saving compared to GenDR with negligible visual degradation, surpassing other one-step diffusion SR. Against all odds, GenDR-Pix can restore 4K image in only 1 second and 6GB.
>
---
#### [new 046] End-to-End LiDAR optimization for 3D point cloud registration
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D点云配准任务，解决LiDAR设计与下游任务脱节的问题。通过动态调整传感器参数，联合优化采集与配准过程，提升精度与效率。**

- **链接: [https://arxiv.org/pdf/2602.10492v1](https://arxiv.org/pdf/2602.10492v1)**

> **作者:** Siddhant Katyan; Marc-André Gardner; Jean-François Lalonde
>
> **备注:** 36th British Machine Vision Conference 2025, {BMVC} 2025, Sheffield, UK, November 24-27, 2025. Project page: https://lvsn.github.io/e2e-lidar-registration/
>
> **摘要:** LiDAR sensors are a key modality for 3D perception, yet they are typically designed independently of downstream tasks such as point cloud registration. Conventional registration operates on pre-acquired datasets with fixed LiDAR configurations, leading to suboptimal data collection and significant computational overhead for sampling, noise filtering, and parameter tuning. In this work, we propose an adaptive LiDAR sensing framework that dynamically adjusts sensor parameters, jointly optimizing LiDAR acquisition and registration hyperparameters. By integrating registration feedback into the sensing loop, our approach optimally balances point density, noise, and sparsity, improving registration accuracy and efficiency. Evaluations in the CARLA simulation demonstrate that our method outperforms fixed-parameter baselines while retaining generalization abilities, highlighting the potential of adaptive LiDAR for autonomous perception and robotic applications.
>
---
#### [new 047] Chart Specification: Structural Representations for Incentivizing VLM Reasoning in Chart-to-Code Generation
- **分类: cs.CV**

- **简介: 该论文属于图表到代码生成任务，旨在解决VLM生成代码时结构不准确的问题。通过引入结构化中间表示和奖励机制，提升生成代码的语义一致性与结构正确性。**

- **链接: [https://arxiv.org/pdf/2602.10880v1](https://arxiv.org/pdf/2602.10880v1)**

> **作者:** Minggui He; Mingchen Dai; Jian Zhang; Yilun Liu; Shimin Tao; Pufan Zeng; Osamu Yoshie; Yuya Ieiri
>
> **备注:** under review
>
> **摘要:** Vision-Language Models (VLMs) have shown promise in generating plotting code from chart images, yet achieving structural fidelity remains challenging. Existing approaches largely rely on supervised fine-tuning, encouraging surface-level token imitation rather than faithful modeling of underlying chart structure, which often leads to hallucinated or semantically inconsistent outputs. We propose Chart Specification, a structured intermediate representation that shifts training from text imitation to semantically grounded supervision. Chart Specification filters syntactic noise to construct a structurally balanced training set and supports a Spec-Align Reward that provides fine-grained, verifiable feedback on structural correctness, enabling reinforcement learning to enforce consistent plotting logic. Experiments on three public benchmarks show that our method consistently outperforms prior approaches. With only 3K training samples, we achieve strong data efficiency, surpassing leading baselines by up to 61.7% on complex benchmarks, and scaling to 4K samples establishes new state-of-the-art results across all evaluated metrics. Overall, our results demonstrate that precise structural supervision offers an efficient pathway to high-fidelity chart-to-code generation. Code and dataset are available at: https://github.com/Mighten/chart-specification-paper
>
---
#### [new 048] MPA: Multimodal Prototype Augmentation for Few-Shot Learning
- **分类: cs.CV**

- **简介: 该论文属于少样本学习任务，旨在解决传统方法仅依赖视觉模态、缺乏多模态信息的问题。提出MPA框架，融合语义增强、多视角增强和不确定性吸收，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.10143v1](https://arxiv.org/pdf/2602.10143v1)**

> **作者:** Liwen Wu; Wei Wang; Lei Zhao; Zhan Gao; Qika Lin; Shaowen Yao; Zuozhu Liu; Bin Pu
>
> **备注:** This paper has been accepted by AAAI 2026
>
> **摘要:** Recently, few-shot learning (FSL) has become a popular task that aims to recognize new classes from only a few labeled examples and has been widely applied in fields such as natural science, remote sensing, and medical images. However, most existing methods focus only on the visual modality and compute prototypes directly from raw support images, which lack comprehensive and rich multimodal information. To address these limitations, we propose a novel Multimodal Prototype Augmentation FSL framework called MPA, including LLM-based Multi-Variant Semantic Enhancement (LMSE), Hierarchical Multi-View Augmentation (HMA), and an Adaptive Uncertain Class Absorber (AUCA). LMSE leverages large language models to generate diverse paraphrased category descriptions, enriching the support set with additional semantic cues. HMA exploits both natural and multi-view augmentations to enhance feature diversity (e.g., changes in viewing distance, camera angles, and lighting conditions). AUCA models uncertainty by introducing uncertain classes via interpolation and Gaussian sampling, effectively absorbing uncertain samples. Extensive experiments on four single-domain and six cross-domain FSL benchmarks demonstrate that MPA achieves superior performance compared to existing state-of-the-art methods across most settings. Notably, MPA surpasses the second-best method by 12.29% and 24.56% in the single-domain and cross-domain setting, respectively, in the 5-way 1-shot setting.
>
---
#### [new 049] Dynamic Frequency Modulation for Controllable Text-driven Image Generation
- **分类: cs.CV**

- **简介: 该论文属于文本驱动图像生成任务，旨在解决修改文本提示时导致结构变化的问题。通过分析频率成分影响，提出一种无需训练的动态频率调制方法，有效平衡结构保持与语义更新。**

- **链接: [https://arxiv.org/pdf/2602.10662v1](https://arxiv.org/pdf/2602.10662v1)**

> **作者:** Tiandong Shi; Ling Zhao; Ji Qi; Jiayi Ma; Chengli Peng
>
> **摘要:** The success of text-guided diffusion models has established a new image generation paradigm driven by the iterative refinement of text prompts. However, modifying the original text prompt to achieve the expected semantic adjustments often results in unintended global structure changes that disrupt user intent. Existing methods rely on empirical feature map selection for intervention, whose performance heavily depends on appropriate selection, leading to suboptimal stability. This paper tries to solve the aforementioned problem from a frequency perspective and analyzes the impact of the frequency spectrum of noisy latent variables on the hierarchical emergence of the structure framework and fine-grained textures during the generation process. We find that lower-frequency components are primarily responsible for establishing the structure framework in the early generation stage. Their influence diminishes over time, giving way to higher-frequency components that synthesize fine-grained textures. In light of this, we propose a training-free frequency modulation method utilizing a frequency-dependent weighting function with dynamic decay. This method maintains the structure framework consistency while permitting targeted semantic modifications. By directly manipulating the noisy latent variable, the proposed method avoids the empirical selection of internal feature maps. Extensive experiments demonstrate that the proposed method significantly outperforms current state-of-the-art methods, achieving an effective balance between preserving structure and enabling semantic updates.
>
---
#### [new 050] FastFlow: Accelerating The Generative Flow Matching Models with Bandit Inference
- **分类: cs.CV**

- **简介: 该论文提出FastFlow，用于加速流匹配模型的生成过程。针对传统方法计算慢、需重训练的问题，FastFlow通过近似部分步骤和多臂老虎机策略，在不损失质量的前提下提升速度。属于图像与视频生成任务。**

- **链接: [https://arxiv.org/pdf/2602.11105v1](https://arxiv.org/pdf/2602.11105v1)**

> **作者:** Divya Jyoti Bajpai; Dhruv Bhardwaj; Soumya Roy; Tejas Duseja; Harsh Agarwal; Aashay Sandansing; Manjesh Kumar Hanawal
>
> **备注:** Accepted at International Conference on Learning Representations (ICLR) 2026
>
> **摘要:** Flow-matching models deliver state-of-the-art fidelity in image and video generation, but the inherent sequential denoising process renders them slower. Existing acceleration methods like distillation, trajectory truncation, and consistency approaches are static, require retraining, and often fail to generalize across tasks. We propose FastFlow, a plug-and-play adaptive inference framework that accelerates generation in flow matching models. FastFlow identifies denoising steps that produce only minor adjustments to the denoising path and approximates them without using the full neural network models used for velocity predictions. The approximation utilizes finite-difference velocity estimates from prior predictions to efficiently extrapolate future states, enabling faster advancements along the denoising path at zero compute cost. This enables skipping computation at intermediary steps. We model the decision of how many steps to safely skip before requiring a full model computation as a multi-armed bandit problem. The bandit learns the optimal skips to balance speed with performance. FastFlow integrates seamlessly with existing pipelines and generalizes across image generation, video generation, and editing tasks. Experiments demonstrate a speedup of over 2.6x while maintaining high-quality outputs. The source code for this work can be found at https://github.com/Div290/FastFlow.
>
---
#### [new 051] RSHallu: Dual-Mode Hallucination Evaluation for Remote-Sensing Multimodal Large Language Models with Domain-Tailored Mitigation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对遥感多模态大模型中的幻觉问题，提出RSHallu系统，包含幻觉分类、基准测试和缓解策略，提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2602.10799v1](https://arxiv.org/pdf/2602.10799v1)**

> **作者:** Zihui Zhou; Yong Feng; Yanying Chen; Guofan Duan; Zhenxi Song; Mingliang Zhou; Weijia Jia
>
> **摘要:** Multimodal large language models (MLLMs) are increasingly adopted in remote sensing (RS) and have shown strong performance on tasks such as RS visual grounding (RSVG), RS visual question answering (RSVQA), and multimodal dialogue. However, hallucinations, which are responses inconsistent with the input RS images, severely hinder their deployment in high-stakes scenarios (e.g., emergency management and agricultural monitoring) and remain under-explored in RS. In this work, we present RSHallu, a systematic study with three deliverables: (1) we formalize RS hallucinations with an RS-oriented taxonomy and introduce image-level hallucination to capture RS-specific inconsistencies beyond object-centric errors (e.g., modality, resolution, and scene-level semantics); (2) we build a hallucination benchmark RSHalluEval (2,023 QA pairs) and enable dual-mode checking, supporting high-precision cloud auditing and low-cost reproducible local checking via a compact checker fine-tuned on RSHalluCheck dataset (15,396 QA pairs); and (3) we introduce a domain-tailored dataset RSHalluShield (30k QA pairs) for training-friendly mitigation and further propose training-free plug-and-play strategies, including decoding-time logit correction and RS-aware prompting. Across representative RS-MLLMs, our mitigation improves the hallucination-free rate by up to 21.63 percentage points under a unified protocol, while maintaining competitive performance on downstream RS tasks (RSVQA/RSVG). Code and datasets will be released.
>
---
#### [new 052] Med-SegLens: Latent-Level Model Diffing for Interpretable Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决模型不透明和数据集偏移问题。通过潜层模型差异分析，识别关键特征并提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.10508v1](https://arxiv.org/pdf/2602.10508v1)**

> **作者:** Salma J. Ahmed; Emad A. Mohammed; Azam Asilian Bidgoli
>
> **摘要:** Modern segmentation models achieve strong predictive performance but remain largely opaque, limiting our ability to diagnose failures, understand dataset shift, or intervene in a principled manner. We introduce Med-SegLens, a model-diffing framework that decomposes segmentation model activations into interpretable latent features using sparse autoencoders trained on SegFormer and U-Net. Through cross-architecture and cross-dataset latent alignment across healthy, adult, pediatric, and sub-Saharan African glioma cohorts, we identify a stable backbone of shared representations, while dataset shift is driven by differential reliance on population-specific latents. We show that these latents act as causal bottlenecks for segmentation failures, and that targeted latent-level interventions can correct errors and improve cross-dataset adaption without retraining, recovering performance in 70% of failure cases and improving Dice score from 39.4% to 74.2%. Our results demonstrate that latent-level model diffing provides a practical and mechanistic tool for diagnosing failures and mitigating dataset shift in segmentation models.
>
---
#### [new 053] DEGMC: Denoising Diffusion Models Based on Riemannian Equivariant Group Morphological Convolutions
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决DDPM模型中几何特征提取和网络等变性问题。通过引入黎曼流形上的群形态卷积，提升模型对几何结构的捕捉能力。**

- **链接: [https://arxiv.org/pdf/2602.10221v1](https://arxiv.org/pdf/2602.10221v1)**

> **作者:** El Hadji S. Diop; Thierno Fall; Mohamed Daoudi
>
> **摘要:** In this work, we address two major issues in recent Denoising Diffusion Probabilistic Models (DDPM): {\bf 1)} geometric key feature extraction and {\bf 2)} network equivariance. Since the DDPM prediction network relies on the U-net architecture, which is theoretically only translation equivariant, we introduce a geometric approach combined with an equivariance property of the more general Euclidean group, which includes rotations, reflections, and permutations. We introduce the notion of group morphological convolutions in Riemannian manifolds, which are derived from the viscosity solutions of first-order Hamilton-Jacobi-type partial differential equations (PDEs) that act as morphological multiscale dilations and erosions. We add a convection term to the model and solve it using the method of characteristics. This helps us better capture nonlinearities, represent thin geometric structures, and incorporate symmetries into the learning process. Experimental results on the MNIST, RotoMNIST, and CIFAR-10 datasets show noticeable improvements compared to the baseline DDPM model.
>
---
#### [new 054] LaSSM: Efficient Semantic-Spatial Query Decoding via Local Aggregation and State Space Models for 3D Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文提出LaSSM，解决3D点云实例分割中的查询初始化和计算效率问题，通过语义-空间初始化和状态空间模型提升性能与效率。**

- **链接: [https://arxiv.org/pdf/2602.11007v1](https://arxiv.org/pdf/2602.11007v1)**

> **作者:** Lei Yao; Yi Wang; Yawen Cui; Moyun Liu; Lap-Pui Chau
>
> **备注:** Accepted at IEEE-TCSVT
>
> **摘要:** Query-based 3D scene instance segmentation from point clouds has attained notable performance. However, existing methods suffer from the query initialization dilemma due to the sparse nature of point clouds and rely on computationally intensive attention mechanisms in query decoders. We accordingly introduce LaSSM, prioritizing simplicity and efficiency while maintaining competitive performance. Specifically, we propose a hierarchical semantic-spatial query initializer to derive the query set from superpoints by considering both semantic cues and spatial distribution, achieving comprehensive scene coverage and accelerated convergence. We further present a coordinate-guided state space model (SSM) decoder that progressively refines queries. The novel decoder features a local aggregation scheme that restricts the model to focus on geometrically coherent regions and a spatial dual-path SSM block to capture underlying dependencies within the query set by integrating associated coordinates information. Our design enables efficient instance prediction, avoiding the incorporation of noisy information and reducing redundant computation. LaSSM ranks first place on the latest ScanNet++ V2 leaderboard, outperforming the previous best method by 2.5% mAP with only 1/3 FLOPs, demonstrating its superiority in challenging large-scale scene instance segmentation. LaSSM also achieves competitive performance on ScanNet, ScanNet200, S3DIS and ScanNet++ V1 benchmarks with less computational cost. Extensive ablation studies and qualitative results validate the effectiveness of our design. The code and weights are available at https://github.com/RayYoh/LaSSM.
>
---
#### [new 055] First International StepUP Competition for Biometric Footstep Recognition: Methods, Results and Remaining Challenges
- **分类: cs.CV; cs.LG**

- **简介: 该论文介绍了一项国际竞赛，旨在解决生物步态识别中的泛化与鲁棒性问题。使用StepUP-P150数据集，参赛者开发识别模型，但面对新鞋类仍存在挑战。**

- **链接: [https://arxiv.org/pdf/2602.11086v1](https://arxiv.org/pdf/2602.11086v1)**

> **作者:** Robyn Larracy; Eve MacDonald; Angkoon Phinyomark; Saeid Rezaei; Mahdi Laghaei; Ali Hajighasem; Aaron Tabor; Erik Scheme
>
> **备注:** to be published in 2025 IEEE International Joint Conference on Biometrics (IJCB)
>
> **摘要:** Biometric footstep recognition, based on a person's unique pressure patterns under their feet during walking, is an emerging field with growing applications in security and safety. However, progress in this area has been limited by the lack of large, diverse datasets necessary to address critical challenges such as generalization to new users and robustness to shifts in factors like footwear or walking speed. The recent release of the UNB StepUP-P150 dataset, the largest and most comprehensive collection of high-resolution footstep pressure recordings to date, opens new opportunities for addressing these challenges through deep learning. To mark this milestone, the First International StepUP Competition for Biometric Footstep Recognition was launched. Competitors were tasked with developing robust recognition models using the StepUP-P150 dataset that were then evaluated on a separate, dedicated test set designed to assess verification performance under challenging variations, given limited and relatively homogeneous reference data. The competition attracted global participation, with 23 registered teams from academia and industry. The top-performing team, Saeid_UCC, achieved the best equal error rate (EER) of 10.77% using a generative reward machine (GRM) optimization strategy. Overall, the competition showcased strong solutions, but persistent challenges in generalizing to unfamiliar footwear highlight a critical area for future work.
>
---
#### [new 056] 3DXTalker: Unifying Identity, Lip Sync, Emotion, and Spatial Dynamics in Expressive 3D Talking Avatars
- **分类: cs.CV**

- **简介: 该论文属于3D Talking Avatar生成任务，旨在解决身份保持、唇形同步、情感表达和空间动态问题。通过数据优化和模型改进，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2602.10516v1](https://arxiv.org/pdf/2602.10516v1)**

> **作者:** Zhongju Wang; Zhenhong Sun; Beier Wang; Yifu Wang; Daoyi Dong; Huadong Mo; Hongdong Li
>
> **摘要:** Audio-driven 3D talking avatar generation is increasingly important in virtual communication, digital humans, and interactive media, where avatars must preserve identity, synchronize lip motion with speech, express emotion, and exhibit lifelike spatial dynamics, collectively defining a broader objective of expressivity. However, achieving this remains challenging due to insufficient training data with limited subject identities, narrow audio representations, and restricted explicit controllability. In this paper, we propose 3DXTalker, an expressive 3D talking avatar through data-curated identity modeling, audio-rich representations, and spatial dynamics controllability. 3DXTalker enables scalable identity modeling via 2D-to-3D data curation pipeline and disentangled representations, alleviating data scarcity and improving identity generalization. Then, we introduce frame-wise amplitude and emotional cues beyond standard speech embeddings, ensuring superior lip synchronization and nuanced expression modulation. These cues are unified by a flow-matching-based transformer for coherent facial dynamics. Moreover, 3DXTalker also enables natural head-pose motion generation while supporting stylized control via prompt-based conditioning. Extensive experiments show that 3DXTalker integrates lip synchronization, emotional expression, and head-pose dynamics within a unified framework, achieves superior performance in 3D talking avatar generation.
>
---
#### [new 057] OccFace: Unified Occlusion-Aware Facial Landmark Detection with Per-Point Visibility
- **分类: cs.CV**

- **简介: 该论文属于人脸关键点检测任务，解决遮挡下检测不准确的问题。提出OccFace框架，统一检测100个关键点并预测可见性，提升遮挡场景下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.10728v1](https://arxiv.org/pdf/2602.10728v1)**

> **作者:** Xinhao Xiang; Zhengxin Li; Saurav Dhakad; Theo Bancroft; Jiawei Zhang; Weiyang Li
>
> **摘要:** Accurate facial landmark detection under occlusion remains challenging, especially for human-like faces with large appearance variation and rotation-driven self-occlusion. Existing detectors typically localize landmarks while handling occlusion implicitly, without predicting per-point visibility that downstream applications can benefits. We present OccFace, an occlusion-aware framework for universal human-like faces, including humans, stylized characters, and other non-human designs. OccFace adopts a unified dense 100-point layout and a heatmap-based backbone, and adds an occlusion module that jointly predicts landmark coordinates and per-point visibility by combining local evidence with cross-landmark context. Visibility supervision mixes manual labels with landmark-aware masking that derives pseudo visibility from mask-heatmap overlap. We also create an occlusion-aware evaluation suite reporting NME on visible vs. occluded landmarks and benchmarking visibility with Occ AP, F1@0.5, and ROC-AUC, together with a dataset annotated with 100-point landmarks and per-point visibility. Experiments show improved robustness under external occlusion and large head rotations, especially on occluded regions, while preserving accuracy on visible landmarks.
>
---
#### [new 058] Fast Person Detection Using YOLOX With AI Accelerator For Train Station Safety
- **分类: cs.CV**

- **简介: 论文研究在火车站应用YOLOX与AI加速器进行快速人员检测，解决安全问题。对比了Hailo-8与Jetson Orin Nano的性能，结果显示Hailo-8准确率更高且延迟更低。**

- **链接: [https://arxiv.org/pdf/2602.10593v1](https://arxiv.org/pdf/2602.10593v1)**

> **作者:** Mas Nurul Achmadiah; Novendra Setyawan; Achmad Arif Bryantono; Chi-Chia Sun; Wen-Kai Kuo
>
> **备注:** 6 pages, 8 figures, 2 tables. Presented at 2024 International Electronics Symposium (IES). IEEE DOI: 10.1109/IES63037.2024.10665874
>
> **摘要:** Recently, Image processing has advanced Faster and applied in many fields, including health, industry, and transportation. In the transportation sector, object detection is widely used to improve security, for example, in traffic security and passenger crossings at train stations. Some accidents occur in the train crossing area at the station, like passengers uncarefully when passing through the yellow line. So further security needs to be developed. Additional technology is required to reduce the number of accidents. This paper focuses on passenger detection applications at train stations using YOLOX and Edge AI Accelerator hardware. the performance of the AI accelerator will be compared with Jetson Orin Nano. The experimental results show that the Hailo-8 AI hardware accelerator has higher accuracy than Jetson Orin Nano (improvement of over 12%) and has lower latency than Jetson Orin Nano (reduced 20 ms).
>
---
#### [new 059] Stride-Net: Fairness-Aware Disentangled Representation Learning for Chest X-Ray Diagnosis
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决深度学习模型在胸片诊断中对特定群体不公平的问题。提出Stride-Net框架，通过解耦表示学习提升公平性与准确性。**

- **链接: [https://arxiv.org/pdf/2602.10875v1](https://arxiv.org/pdf/2602.10875v1)**

> **作者:** Darakshan Rashid; Raza Imam; Dwarikanath Mahapatra; Brejesh Lall
>
> **备注:** 6 pages, 2 Tables, 3 Figures. Our code is available https://github.com/Daraksh/Fairness_StrideNet
>
> **摘要:** Deep neural networks for chest X-ray classification achieve strong average performance, yet often underperform for specific demographic subgroups, raising critical concerns about clinical safety and equity. Existing debiasing methods frequently yield inconsistent improvements across datasets or attain fairness by degrading overall diagnostic utility, treating fairness as a post hoc constraint rather than a property of the learned representation. In this work, we propose Stride-Net (Sensitive Attribute Resilient Learning via Disentanglement and Learnable Masking with Embedding Alignment), a fairness-aware framework that learns disease-discriminative yet demographically invariant representations for chest X-ray analysis. Stride-Net operates at the patch level, using a learnable stride-based mask to select label-aligned image regions while suppressing sensitive attribute information through adversarial confusion loss. To anchor representations in clinical semantics and discourage shortcut learning, we further enforce semantic alignment between image features and BioBERT-based disease label embeddings via Group Optimal Transport. We evaluate Stride-Net on the MIMIC-CXR and CheXpert benchmarks across race and intersectional race-gender subgroups. Across architectures including ResNet and Vision Transformers, Stride-Net consistently improves fairness metrics while matching or exceeding baseline accuracy, achieving a more favorable accuracy-fairness trade-off than prior debiasing approaches. Our code is available at https://github.com/Daraksh/Fairness_StrideNet.
>
---
#### [new 060] ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决世界模型在动态物体建模上的不足。提出TR-World模型，通过时间残差提取动态信息，提升未来场景预测精度。**

- **链接: [https://arxiv.org/pdf/2602.10884v1](https://arxiv.org/pdf/2602.10884v1)**

> **作者:** Jinqing Zhang; Zehua Fu; Zelin Xu; Wenying Dai; Qingjie Liu; Yunhong Wang
>
> **备注:** ICLR 2026
>
> **摘要:** The comprehensive understanding capabilities of world models for driving scenarios have significantly improved the planning accuracy of end-to-end autonomous driving frameworks. However, the redundant modeling of static regions and the lack of deep interaction with trajectories hinder world models from exerting their full effectiveness. In this paper, we propose Temporal Residual World Model (TR-World), which focuses on dynamic object modeling. By calculating the temporal residuals of scene representations, the information of dynamic objects can be extracted without relying on detection and tracking. TR-World takes only temporal residuals as input, thus predicting the future spatial distribution of dynamic objects more precisely. By combining the prediction with the static object information contained in the current BEV features, accurate future BEV features can be obtained. Furthermore, we propose Future-Guided Trajectory Refinement (FGTR) module, which conducts interaction between prior trajectories (predicted from the current scene representation) and the future BEV features. This module can not only utilize future road conditions to refine trajectories, but also provides sparse spatial-temporal supervision on future BEV features to prevent world model collapse. Comprehensive experiments conducted on the nuScenes and NAVSIM datasets demonstrate that our method, namely ResWorld, achieves state-of-the-art planning performance. The code is available at https://github.com/mengtan00/ResWorld.git.
>
---
#### [new 061] ArtisanGS: Interactive Tools for Gaussian Splat Selection with AI and Human in the Loop
- **分类: cs.CV**

- **简介: 该论文属于3D场景编辑任务，解决3DGS对象提取与可控编辑问题。提出交互式工具，结合AI与用户引导，实现高效分割与局部编辑。**

- **链接: [https://arxiv.org/pdf/2602.10173v1](https://arxiv.org/pdf/2602.10173v1)**

> **作者:** Clement Fuji Tsang; Anita Hu; Or Perel; Carsten Kolve; Maria Shugrina
>
> **备注:** 12 pages, includes supplementary material
>
> **摘要:** Representation in the family of 3D Gaussian Splats (3DGS) are growing into a viable alternative to traditional graphics for an expanding number of application, including recent techniques that facilitate physics simulation and animation. However, extracting usable objects from in-the-wild captures remains challenging and controllable editing techniques for this representation are limited. Unlike the bulk of emerging techniques, focused on automatic solutions or high-level editing, we introduce an interactive suite of tools centered around versatile Gaussian Splat selection and segmentation. We propose a fast AI-driven method to propagate user-guided 2D selection masks to 3DGS selections. This technique allows for user intervention in the case of errors and is further coupled with flexible manual selection and segmentation tools. These allow a user to achieve virtually any binary segmentation of an unstructured 3DGS scene. We evaluate our toolset against the state-of-the-art for Gaussian Splat selection and demonstrate their utility for downstream applications by developing a user-guided local editing approach, leveraging a custom Video Diffusion Model. With flexible selection tools, users have direct control over the areas that the AI can modify. Our selection and editing tools can be used for any in-the-wild capture without additional optimization.
>
---
#### [new 062] Interpretable Vision Transformers in Monocular Depth Estimation via SVDA
- **分类: cs.CV**

- **简介: 该论文属于单目深度估计任务，旨在解决Transformer模型中注意力机制不透明的问题。通过引入SVDA，提升模型的可解释性，同时保持预测精度。**

- **链接: [https://arxiv.org/pdf/2602.11005v1](https://arxiv.org/pdf/2602.11005v1)**

> **作者:** Vasileios Arampatzakis; George Pavlidis; Nikolaos Mitianoudis; Nikos Papamarkos
>
> **备注:** 8 pages, 2 figures, submitted to CVPR Conference 2026
>
> **摘要:** Monocular depth estimation is a central problem in computer vision with applications in robotics, AR, and autonomous driving, yet the self-attention mechanisms that drive modern Transformer architectures remain opaque. We introduce SVD-Inspired Attention (SVDA) into the Dense Prediction Transformer (DPT), providing the first spectrally structured formulation of attention for dense prediction tasks. SVDA decouples directional alignment from spectral modulation by embedding a learnable diagonal matrix into normalized query-key interactions, enabling attention maps that are intrinsically interpretable rather than post-hoc approximations. Experiments on KITTI and NYU-v2 show that SVDA preserves or slightly improves predictive accuracy while adding only minor computational overhead. More importantly, SVDA unlocks six spectral indicators that quantify entropy, rank, sparsity, alignment, selectivity, and robustness. These reveal consistent cross-dataset and depth-wise patterns in how attention organizes during training, insights that remain inaccessible in standard Transformers. By shifting the role of attention from opaque mechanism to quantifiable descriptor, SVDA redefines interpretability in monocular depth estimation and opens a principled avenue toward transparent dense prediction models.
>
---
#### [new 063] PhyCritic: Multimodal Critic Models for Physical AI
- **分类: cs.CV**

- **简介: 该论文提出PhyCritic，解决物理AI任务中评估与对齐问题，通过两阶段强化学习优化多模态批评模型，提升物理感知与推理能力。**

- **链接: [https://arxiv.org/pdf/2602.11124v1](https://arxiv.org/pdf/2602.11124v1)**

> **作者:** Tianyi Xiong; Shihao Wang; Guilin Liu; Yi Dong; Ming Li; Heng Huang; Jan Kautz; Zhiding Yu
>
> **摘要:** With the rapid development of large multimodal models, reliable judge and critic models have become essential for open-ended evaluation and preference alignment, providing pairwise preferences, numerical scores, and explanatory justifications for assessing model-generated responses. However, existing critics are primarily trained in general visual domains such as captioning or image question answering, leaving physical AI tasks involving perception, causal reasoning, and planning largely underexplored. We introduce PhyCritic, a multimodal critic model optimized for physical AI through a two-stage RLVR pipeline: a physical skill warmup stage that enhances physically oriented perception and reasoning, followed by self-referential critic finetuning, where the critic generates its own prediction as an internal reference before judging candidate responses, improving judgment stability and physical correctness. Across both physical and general-purpose multimodal judge benchmarks, PhyCritic achieves strong performance gains over open-source baselines and, when applied as a policy model, further improves perception and reasoning in physically grounded tasks.
>
---
#### [new 064] A Low-Rank Defense Method for Adversarial Attack on Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于防御任务，旨在对抗扩散模型的对抗攻击。提出LoRD方法，通过低秩适应检测并防御对抗样本，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.10319v1](https://arxiv.org/pdf/2602.10319v1)**

> **作者:** Jiaxuan Zhu; Siyu Huang
>
> **备注:** Accepted by ICME2025
>
> **摘要:** Recently, adversarial attacks for diffusion models as well as their fine-tuning process have been developed rapidly. To prevent the abuse of these attack algorithms from affecting the practical application of diffusion models, it is critical to develop corresponding defensive strategies. In this work, we propose an efficient defensive strategy, named Low-Rank Defense (LoRD), to defend the adversarial attack on Latent Diffusion Models (LDMs). LoRD introduces the merging idea and a balance parameter, combined with the low-rank adaptation (LoRA) modules, to detect and defend the adversarial samples. Based on LoRD, we build up a defense pipeline that applies the learned LoRD modules to help diffusion models defend against attack algorithms. Our method ensures that the LDM fine-tuned on both adversarial and clean samples can still generate high-quality images. To demonstrate the effectiveness of our approach, we conduct extensive experiments on facial and landscape images, and our method shows significantly better defense performance compared to the baseline methods.
>
---
#### [new 065] Enhancing Weakly Supervised Multimodal Video Anomaly Detection through Text Guidance
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于弱监督多模态视频异常检测任务，旨在提升文本模态的利用效果，解决特征提取困难和多模态融合不平衡问题，提出文本引导框架与多尺度融合模块。**

- **链接: [https://arxiv.org/pdf/2602.10549v1](https://arxiv.org/pdf/2602.10549v1)**

> **作者:** Shengyang Sun; Jiashen Hua; Junyi Feng; Xiaojin Gong
>
> **备注:** Accepted by IEEE Transactions on Multimedia
>
> **摘要:** Weakly supervised multimodal video anomaly detection has gained significant attention, yet the potential of the text modality remains under-explored. Text provides explicit semantic information that can enhance anomaly characterization and reduce false alarms. However, extracting effective text features is challenging due to the inability of general-purpose language models to capture anomaly-specific nuances and the scarcity of relevant descriptions. Furthermore, multimodal fusion often suffers from redundancy and imbalance. To address these issues, we propose a novel text-guided framework. First, we introduce an in-context learning-based multi-stage text augmentation mechanism to generate high-quality anomaly text samples for fine-tuning the text feature extractor. Second, we design a multi-scale bottleneck Transformer fusion module that uses compressed bottleneck tokens to progressively integrate information across modalities, mitigating redundancy and imbalance. Experiments on UCF-Crime and XD-Violence demonstrate state-of-the-art performance.
>
---
#### [new 066] SurfPhase: 3D Interfacial Dynamics in Two-Phase Flows from Sparse Videos
- **分类: cs.CV**

- **简介: 该论文提出SurfPhase，用于从稀疏视频中重建两相流的三维界面动态。解决的是多相流界面测量难题，通过结合高斯表面和扩散模型实现高质量重建与视图合成。**

- **链接: [https://arxiv.org/pdf/2602.11154v1](https://arxiv.org/pdf/2602.11154v1)**

> **作者:** Yue Gao; Hong-Xing Yu; Sanghyeon Chang; Qianxi Fu; Bo Zhu; Yoonjin Won; Juan Carlos Niebles; Jiajun Wu
>
> **备注:** The first two authors contributed equally. Project website: https://yuegao.me/SurfPhase
>
> **摘要:** Interfacial dynamics in two-phase flows govern momentum, heat, and mass transfer, yet remain difficult to measure experimentally. Classical techniques face intrinsic limitations near moving interfaces, while existing neural rendering methods target single-phase flows with diffuse boundaries and cannot handle sharp, deformable liquid-vapor interfaces. We propose SurfPhase, a novel model for reconstructing 3D interfacial dynamics from sparse camera views. Our approach integrates dynamic Gaussian surfels with a signed distance function formulation for geometric consistency, and leverages a video diffusion model to synthesize novel-view videos to refine reconstruction from sparse observations. We evaluate on a new dataset of high-speed pool boiling videos, demonstrating high-quality view synthesis and velocity estimation from only two camera views. Project website: https://yuegao.me/SurfPhase.
>
---
#### [new 067] Enhancing Predictability of Multi-Tenant DNN Inference for Autonomous Vehicles' Perception
- **分类: cs.CV; cs.AI; cs.RO; eess.SY**

- **简介: 该论文属于自动驾驶感知任务，旨在提升多租户DNN推理的可预测性。通过动态选择关键帧和感兴趣区域，减少计算量并保持精度，解决资源受限下的实时性问题。**

- **链接: [https://arxiv.org/pdf/2602.11004v1](https://arxiv.org/pdf/2602.11004v1)**

> **作者:** Liangkai Liu; Kang G. Shin; Jinkyu Lee; Chengmo Yang; Weisong Shi
>
> **备注:** 13 pages, 12 figures
>
> **摘要:** Autonomous vehicles (AVs) rely on sensors and deep neural networks (DNNs) to perceive their surrounding environment and make maneuver decisions in real time. However, achieving real-time DNN inference in the AV's perception pipeline is challenging due to the large gap between the computation requirement and the AV's limited resources. Most, if not all, of existing studies focus on optimizing the DNN inference time to achieve faster perception by compressing the DNN model with pruning and quantization. In contrast, we present a Predictable Perception system with DNNs (PP-DNN) that reduce the amount of image data to be processed while maintaining the same level of accuracy for multi-tenant DNNs by dynamically selecting critical frames and regions of interest (ROIs). PP-DNN is based on our key insight that critical frames and ROIs for AVs vary with the AV's surrounding environment. However, it is challenging to identify and use critical frames and ROIs in multi-tenant DNNs for predictable inference. Given image-frame streams, PP-DNN leverages an ROI generator to identify critical frames and ROIs based on the similarities of consecutive frames and traffic scenarios. PP-DNN then leverages a FLOPs predictor to predict multiply-accumulate operations (MACs) from the dynamic critical frames and ROIs. The ROI scheduler coordinates the processing of critical frames and ROIs with multiple DNN models. Finally, we design a detection predictor for the perception of non-critical frames. We have implemented PP-DNN in an ROS-based AV pipeline and evaluated it with the BDD100K and the nuScenes dataset. PP-DNN is observed to significantly enhance perception predictability, increasing the number of fusion frames by up to 7.3x, reducing the fusion delay by >2.6x and fusion-delay variations by >2.3x, improving detection completeness by 75.4% and the cost-effectiveness by up to 98% over the baseline.
>
---
#### [new 068] Self-Supervised Image Super-Resolution Quality Assessment based on Content-Free Multi-Model Oriented Representation Learning
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像超分辨率质量评估任务，解决真实场景下SR图像质量评估难题。提出S3 RIQA方法，通过自监督学习和多模型表示学习提升评估性能。**

- **链接: [https://arxiv.org/pdf/2602.10744v1](https://arxiv.org/pdf/2602.10744v1)**

> **作者:** Kian Majlessi; Amir Masoud Soltani; Mohammad Ebrahim Mahdavi; Aurelien Gourrier; Peyman Adibi
>
> **摘要:** Super-resolution (SR) applied to real-world low-resolution (LR) images often results in complex, irregular degradations that stem from the inherent complexity of natural scene acquisition. In contrast to SR artifacts arising from synthetic LR images created under well-defined scenarios, those distortions are highly unpredictable and vary significantly across different real-life contexts. Consequently, assessing the quality of SR images (SR-IQA) obtained from realistic LR, remains a challenging and underexplored problem. In this work, we introduce a no-reference SR-IQA approach tailored for such highly ill-posed realistic settings. The proposed method enables domain-adaptive IQA for real-world SR applications, particularly in data-scarce domains. We hypothesize that degradations in super-resolved images are strongly dependent on the underlying SR algorithms, rather than being solely determined by image content. To this end, we introduce a self-supervised learning (SSL) strategy that first pretrains multiple SR model oriented representations in a pretext stage. Our contrastive learning framework forms positive pairs from images produced by the same SR model and negative pairs from those generated by different methods, independent of image content. The proposed approach S3 RIQA, further incorporates targeted preprocessing to extract complementary quality information and an auxiliary task to better handle the various degradation profiles associated with different SR scaling factors. To this end, we constructed a new dataset, SRMORSS, to support unsupervised pretext training; it includes a wide range of SR algorithms applied to numerous real LR images, which addresses a gap in existing datasets. Experiments on real SR-IQA benchmarks demonstrate that S3 RIQA consistently outperforms most state-of-the-art relevant metrics.
>
---
#### [new 069] The Garbage Dataset (GD): A Multi-Class Image Benchmark for Automated Waste Segregation
- **分类: cs.CV**

- **简介: 该论文提出垃圾数据集（GD），用于自动垃圾分类任务，解决分类准确性和环境影响问题，通过构建多类别数据集并评估深度学习模型性能。**

- **链接: [https://arxiv.org/pdf/2602.10500v1](https://arxiv.org/pdf/2602.10500v1)**

> **作者:** Suman Kunwar
>
> **备注:** 11 pages 10 figures and 1 table
>
> **摘要:** This study introduces the Garbage Dataset (GD), a publicly available image dataset designed to advance automated waste segregation through machine learning and computer vision. It's a diverse dataset covering 10 common household waste categories: metal, glass, biological, paper, battery, trash, cardboard, shoes, clothes, and plastic. The dataset comprises 13,348 labeled images collected through multiple methods, including DWaste mobile app and curated web sources. Methods included rigorous validation through checksums and outlier detection, analysis of class imbalance and visual separability via PCA/t-SNE, and assessment of background complexity using entropy and saliency measures. The dataset was benchmarked using state-of-the-art deep learning models (EfficientNetV2M, EfficientNetV2S, MobileNet, ResNet50, ResNet101) evaluated on performance metrics and operational carbon emissions. Experiment results indicate EfficientNetV2S achieved the highest performance with 96.19% accuracy and a 0.96 F1-score, though with a moderate carbon cost. Analysis revealed inherent dataset characteristics including class imbalance, a skew toward high-outlier classes (plastic, cardboard, paper), and brightness variations that require consideration. The main conclusion is that GD provides a valuable, real-world benchmark for waste classification research while highlighting important challenges such as class imbalance, background complexity, and environmental trade-offs in model selection that must be addressed for practical deployment. The dataset is publicly released to support further research in environmental sustainability applications.
>
---
#### [new 070] C^2ROPE: Causal Continuous Rotary Positional Encoding for 3D Large Multimodal-Models Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D多模态模型任务，解决RoPE在视觉处理中的空间连续性和因果关系建模不足问题，提出C^2RoPE改进位置编码机制。**

- **链接: [https://arxiv.org/pdf/2602.10551v1](https://arxiv.org/pdf/2602.10551v1)**

> **作者:** Guanting Ye; Qiyan Zhao; Wenhao Yu; Xiaofeng Zhang; Jianmin Ji; Yanyong Zhang; Ka-Veng Yuen
>
> **备注:** Accepted in ICRA 2026
>
> **摘要:** Recent advances in 3D Large Multimodal Models (LMMs) built on Large Language Models (LLMs) have established the alignment of 3D visual features with LLM representations as the dominant paradigm. However, the inherited Rotary Position Embedding (RoPE) introduces limitations for multimodal processing. Specifically, applying 1D temporal positional indices disrupts the continuity of visual features along the column dimension, resulting in spatial locality loss. Moreover, RoPE follows the prior that temporally closer image tokens are more causally related, leading to long-term decay in attention allocation and causing the model to progressively neglect earlier visual tokens as the sequence length increases. To address these issues, we propose C^2RoPE, an improved RoPE that explicitly models local spatial Continuity and spatial Causal relationships for visual processing. C^2RoPE introduces a spatio-temporal continuous positional embedding mechanism for visual tokens. It first integrates 1D temporal positions with Cartesian-based spatial coordinates to construct a triplet hybrid positional index, and then employs a frequency allocation strategy to encode spatio-temporal positional information across the three index components. Additionally, we introduce Chebyshev Causal Masking, which determines causal dependencies by computing the Chebyshev distance of image tokens in 2D space. Evaluation results across various benchmarks, including 3D scene reasoning and 3D visual question answering, demonstrate C^2RoPE's effectiveness. The code is be available at https://github.com/ErikZ719/C2RoPE.
>
---
#### [new 071] RealHD: A High-Quality Dataset for Robust Detection of State-of-the-Art AI-Generated Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于AI生成图像检测任务，旨在解决现有数据集质量低、多样性不足的问题。作者构建了高质量的RealHD数据集，并提出一种基于噪声熵的轻量检测方法。**

- **链接: [https://arxiv.org/pdf/2602.10546v1](https://arxiv.org/pdf/2602.10546v1)**

> **作者:** Hanzhe Yu; Yun Ye; Jintao Rong; Qi Xuan; Chen Ma
>
> **备注:** Published in the Proceedings of the 33rd ACM International Conference on Multimedia (ACM MM 2025)
>
> **摘要:** The rapid advancement of generative AI has raised concerns about the authenticity of digital images, as highly realistic fake images can now be generated at low cost, potentially increasing societal risks. In response, several datasets have been established to train detection models aimed at distinguishing AI-generated images from real ones. However, existing datasets suffer from limited generalization, low image quality, overly simple prompts, and insufficient image diversity. To address these limitations, we propose a high-quality, large-scale dataset comprising over 730,000 images across multiple categories, including both real and AI-generated images. The generated images are synthesized via state-of-the-art methods, including text-to-image generation (guided by over 10,000 carefully designed prompts), image inpainting, image refinement, and face swapping. Each generated image is annotated with its generation method and category. Inpainting images further include binary masks to indicate inpainted regions, providing rich metadata for analysis. Compared to existing datasets, detection models trained on our dataset demonstrate superior generalization capabilities. Our dataset not only serves as a strong benchmark for evaluating detection methods but also contributes to advancing the robustness of AI-generated image detection techniques. Building upon this, we propose a lightweight detection method based on image noise entropy, which transforms the original image into an entropy tensor of Non-Local Means (NLM) noise before classification. Extensive experiments demonstrate that models trained on our dataset achieve strong generalization, and our method delivers competitive performance, establishing a solid baseline for future research. The dataset and source code are publicly available at https://real-hd.github.io.
>
---
#### [new 072] DeepImageSearch: Benchmarking Multimodal Agents for Context-Aware Image Retrieval in Visual Histories
- **分类: cs.CV; cs.IR**

- **简介: 该论文属于图像检索任务，旨在解决传统系统忽略视觉流中上下文依赖的问题。提出DeepImageSearch框架，通过多步骤推理实现基于上下文的图像检索。**

- **链接: [https://arxiv.org/pdf/2602.10809v1](https://arxiv.org/pdf/2602.10809v1)**

> **作者:** Chenlong Deng; Mengjie Deng; Junjie Wu; Dun Zeng; Teng Wang; Qingsong Xie; Jiadeng Huang; Shengjie Ma; Changwang Zhang; Zhaoxiang Wang; Jun Wang; Yutao Zhu; Zhicheng Dou
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** Existing multimodal retrieval systems excel at semantic matching but implicitly assume that query-image relevance can be measured in isolation. This paradigm overlooks the rich dependencies inherent in realistic visual streams, where information is distributed across temporal sequences rather than confined to single snapshots. To bridge this gap, we introduce DeepImageSearch, a novel agentic paradigm that reformulates image retrieval as an autonomous exploration task. Models must plan and perform multi-step reasoning over raw visual histories to locate targets based on implicit contextual cues. We construct DISBench, a challenging benchmark built on interconnected visual data. To address the scalability challenge of creating context-dependent queries, we propose a human-model collaborative pipeline that employs vision-language models to mine latent spatiotemporal associations, effectively offloading intensive context discovery before human verification. Furthermore, we build a robust baseline using a modular agent framework equipped with fine-grained tools and a dual-memory system for long-horizon navigation. Extensive experiments demonstrate that DISBench poses significant challenges to state-of-the-art models, highlighting the necessity of incorporating agentic reasoning into next-generation retrieval systems.
>
---
#### [new 073] VERA: Identifying and Leveraging Visual Evidence Retrieval Heads in Long-Context Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型任务，旨在解决长文本理解问题。通过分析注意力机制，发现并利用视觉证据检索头提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.10146v1](https://arxiv.org/pdf/2602.10146v1)**

> **作者:** Rongcan Pei; Huan Li; Fang Guo; Qi Zhu
>
> **备注:** 12 pages, 12 figures
>
> **摘要:** While Vision-Language Models (VLMs) have shown promise in textual understanding, they face significant challenges when handling long context and complex reasoning tasks. In this paper, we dissect the internal mechanisms governing long-context processing in VLMs to understand their performance bottlenecks. Through the lens of attention analysis, we identify specific Visual Evidence Retrieval (VER) Heads - a sparse, dynamic set of attention heads critical for locating visual cues during reasoning, distinct from static OCR heads. We demonstrate that these heads are causal to model performance; masking them leads to significant degradation. Leveraging this discovery, we propose VERA (Visual Evidence Retrieval Augmentation), a training-free framework that detects model uncertainty (i.e., entropy) to trigger the explicit verbalization of visual evidence attended by VER heads. Comprehensive experiments demonstrate that VERA significantly improves long-context understanding of open-source VLMs: it yields an average relative improvement of 21.3% on Qwen3-VL-8B-Instruct and 20.1% on GLM-4.1V-Thinking across five benchmarks.
>
---
#### [new 074] TwiFF (Think With Future Frames): A Large-Scale Dataset for Dynamic Visual Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出TwiFF数据集和模型，解决动态视觉推理任务中的时间动态捕捉问题，通过视频生成与图像理解结合，提升动态场景下的视觉问答性能。**

- **链接: [https://arxiv.org/pdf/2602.10675v1](https://arxiv.org/pdf/2602.10675v1)**

> **作者:** Junhua Liu; Zhangcheng Wang; Zhike Han; Ningli Wang; Guotao Liang; Kun Kuang
>
> **备注:** preprint
>
> **摘要:** Visual Chain-of-Thought (VCoT) has emerged as a promising paradigm for enhancing multimodal reasoning by integrating visual perception into intermediate reasoning steps. However, existing VCoT approaches are largely confined to static scenarios and struggle to capture the temporal dynamics essential for tasks such as instruction, prediction, and camera motion. To bridge this gap, we propose TwiFF-2.7M, the first large-scale, temporally grounded VCoT dataset derived from $2.7$ million video clips, explicitly designed for dynamic visual question and answer. Accompanying this, we introduce TwiFF-Bench, a high-quality evaluation benchmark of $1,078$ samples that assesses both the plausibility of reasoning trajectories and the correctness of final answers in open-ended dynamic settings. Building on these foundations, we propose the TwiFF model, a unified modal that synergistically leverages pre-trained video generation and image comprehension capabilities to produce temporally coherent visual reasoning cues-iteratively generating future action frames and textual reasoning. Extensive experiments demonstrate that TwiFF significantly outperforms existing VCoT methods and Textual Chain-of-Thought baselines on dynamic reasoning tasks, which fully validates the effectiveness for visual question answering in dynamic scenarios. Our code and data is available at https://github.com/LiuJunhua02/TwiFF.
>
---
#### [new 075] From Steering to Pedalling: Do Autonomous Driving VLMs Generalize to Cyclist-Assistive Spatial Perception and Planning?
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于交通感知与决策任务，旨在解决自动驾驶模型在骑行者辅助场景下的泛化能力问题。通过构建CyclingVQA基准，评估模型在骑行者视角下的空间理解和交通规则推理能力。**

- **链接: [https://arxiv.org/pdf/2602.10771v1](https://arxiv.org/pdf/2602.10771v1)**

> **作者:** Krishna Kanth Nakka; Vedasri Nakka
>
> **备注:** Preprint
>
> **摘要:** Cyclists often encounter safety-critical situations in urban traffic, highlighting the need for assistive systems that support safe and informed decision-making. Recently, vision-language models (VLMs) have demonstrated strong performance on autonomous driving benchmarks, suggesting their potential for general traffic understanding and navigation-related reasoning. However, existing evaluations are predominantly vehicle-centric and fail to assess perception and reasoning from a cyclist-centric viewpoint. To address this gap, we introduce CyclingVQA, a diagnostic benchmark designed to probe perception, spatio-temporal understanding, and traffic-rule-to-lane reasoning from a cyclist's perspective. Evaluating 31+ recent VLMs spanning general-purpose, spatially enhanced, and autonomous-driving-specialized models, we find that current models demonstrate encouraging capabilities, while also revealing clear areas for improvement in cyclist-centric perception and reasoning, particularly in interpreting cyclist-specific traffic cues and associating signs with the correct navigational lanes. Notably, several driving-specialized models underperform strong generalist VLMs, indicating limited transfer from vehicle-centric training to cyclist-assistive scenarios. Finally, through systematic error analysis, we identify recurring failure modes to guide the development of more effective cyclist-assistive intelligent systems.
>
---
#### [new 076] Enhancing YOLOv11n for Reliable Child Detection in Noisy Surveillance Footage
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在提升低质量监控视频中儿童的检测效果。通过改进YOLOv11n，引入增强策略和SAHI技术，提高小目标和遮挡情况下的检测精度。**

- **链接: [https://arxiv.org/pdf/2602.10592v1](https://arxiv.org/pdf/2602.10592v1)**

> **作者:** Khanh Linh Tran; Minh Nguyen Dang; Thien Nguyen Trong; Hung Nguyen Quoc; Linh Nguyen Kieu
>
> **摘要:** This paper presents a practical and lightweight solution for enhancing child detection in low-quality surveillance footage, a critical component in real-world missing child alert and daycare monitoring systems. Building upon the efficient YOLOv11n architecture, we propose a deployment-ready pipeline that improves detection under challenging conditions including occlusion, small object size, low resolution, motion blur, and poor lighting commonly found in existing CCTV infrastructures. Our approach introduces a domain-specific augmentation strategy that synthesizes realistic child placements using spatial perturbations such as partial visibility, truncation, and overlaps, combined with photometric degradations including lighting variation and noise. To improve recall of small and partially occluded instances, we integrate Slicing Aided Hyper Inference (SAHI) at inference time. All components are trained and evaluated on a filtered, child-only subset of the Roboflow Daycare dataset. Compared to the baseline YOLOv11n, our enhanced system achieves a mean Average Precision at 0.5 IoU (mAP@0.5) of 0.967 and a mean Average Precision averaged over IoU thresholds from 0.5 to 0.95 (mAP@0.5:0.95) of 0.783, yielding absolute improvements of 0.7 percent and 2.3 percent, respectively, without architectural changes. Importantly, the entire pipeline maintains compatibility with low-power edge devices and supports real-time performance, making it particularly well suited for low-cost or resource-constrained industrial surveillance deployments. The example augmented dataset and the source code used to generate it are available at: https://github.com/html-ptit/Data-Augmentation-YOLOv11n-child-detection
>
---
#### [new 077] Multi-encoder ConvNeXt Network with Smooth Attentional Feature Fusion for Multispectral Semantic Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MeCSAFNet，用于多光谱图像的语义分割任务，通过多编码器和注意力融合提升分割精度与效率。**

- **链接: [https://arxiv.org/pdf/2602.10137v1](https://arxiv.org/pdf/2602.10137v1)**

> **作者:** Leo Thomas Ramos; Angel D. Sappa
>
> **备注:** This is an extended version of the study presented at IEEE SoutheastCon2025. It presents substantial new content and original contributions beyond the previous version, including an expanded and enhanced background, new architectural refinements, additional experiments conducted on a broader range of datasets and experimental scenarios, and a more comprehensive analysis of results
>
> **摘要:** This work proposes MeCSAFNet, a multi-branch encoder-decoder architecture for land cover segmentation in multispectral imagery. The model separately processes visible and non-visible channels through dual ConvNeXt encoders, followed by individual decoders that reconstruct spatial information. A dedicated fusion decoder integrates intermediate features at multiple scales, combining fine spatial cues with high-level spectral representations. The feature fusion is further enhanced with CBAM attention, and the ASAU activation function contributes to stable and efficient optimization. The model is designed to process different spectral configurations, including a 4-channel (4c) input combining RGB and NIR bands, as well as a 6-channel (6c) input incorporating NDVI and NDWI indices. Experiments on the Five-Billion-Pixels (FBP) and Potsdam datasets demonstrate significant performance gains. On FBP, MeCSAFNet-base (6c) surpasses U-Net (4c) by +19.21%, U-Net (6c) by +14.72%, SegFormer (4c) by +19.62%, and SegFormer (6c) by +14.74% in mIoU. On Potsdam, MeCSAFNet-large (4c) improves over DeepLabV3+ (4c) by +6.48%, DeepLabV3+ (6c) by +5.85%, SegFormer (4c) by +9.11%, and SegFormer (6c) by +4.80% in mIoU. The model also achieves consistent gains over several recent state-of-the-art approaches. Moreover, compact variants of MeCSAFNet deliver notable performance with lower training time and reduced inference cost, supporting their deployment in resource-constrained environments.
>
---
#### [new 078] PMMA: The Polytechnique Montreal Mobility Aids Dataset
- **分类: cs.CV**

- **简介: 该论文提出PMMA数据集，用于行人检测任务，特别是使用辅助设备的行人。旨在提升对特殊群体的识别能力，包含多种行人类型，并测试了多个检测模型与跟踪算法。**

- **链接: [https://arxiv.org/pdf/2602.10259v1](https://arxiv.org/pdf/2602.10259v1)**

> **作者:** Qingwu Liu; Nicolas Saunier; Guillaume-Alexandre Bilodeau
>
> **备注:** Submitted to the journal IEEE Transactions on Intelligent Transportation Systems, under review
>
> **摘要:** This study introduces a new object detection dataset of pedestrians using mobility aids, named PMMA. The dataset was collected in an outdoor environment, where volunteers used wheelchairs, canes, and walkers, resulting in nine categories of pedestrians: pedestrians, cane users, two types of walker users, whether walking or resting, five types of wheelchair users, including wheelchair users, people pushing empty wheelchairs, and three types of users pushing occupied wheelchairs, including the entire pushing group, the pusher and the person seated on the wheelchair. To establish a benchmark, seven object detection models (Faster R-CNN, CenterNet, YOLOX, DETR, Deformable DETR, DINO, and RT-DETR) and three tracking algorithms (ByteTrack, BOT-SORT, and OC-SORT) were implemented under the MMDetection framework. Experimental results show that YOLOX, Deformable DETR, and Faster R-CNN achieve the best detection performance, while the differences among the three trackers are relatively small. The PMMA dataset is publicly available at https://doi.org/10.5683/SP3/XJPQUG, and the video processing and model training code is available at https://github.com/DatasetPMMA/PMMA.
>
---
#### [new 079] AD$^2$: Analysis and Detection of Adversarial Threats in Visual Perception for End-to-End Autonomous Driving Systems
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自动驾驶安全任务，旨在解决视觉感知中的对抗威胁问题。通过分析三种攻击向量，提出AD²检测模型以提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.10160v1](https://arxiv.org/pdf/2602.10160v1)**

> **作者:** Ishan Sahu; Somnath Hazra; Somak Aditya; Soumyajit Dey
>
> **备注:** Accepted to WACV 2026
>
> **摘要:** End-to-end autonomous driving systems have achieved significant progress, yet their adversarial robustness remains largely underexplored. In this work, we conduct a closed-loop evaluation of state-of-the-art autonomous driving agents under black-box adversarial threat models in CARLA. Specifically, we consider three representative attack vectors on the visual perception pipeline: (i) a physics-based blur attack induced by acoustic waves, (ii) an electromagnetic interference attack that distorts captured images, and (iii) a digital attack that adds ghost objects as carefully crafted bounded perturbations on images. Our experiments on two advanced agents, Transfuser and Interfuser, reveal severe vulnerabilities to such attacks, with driving scores dropping by up to 99% in the worst case, raising valid safety concerns. To help mitigate such threats, we further propose a lightweight Attack Detection model for Autonomous Driving systems (AD$^2$) based on attention mechanisms that capture spatial-temporal consistency. Comprehensive experiments across multi-camera inputs on CARLA show that our detector achieves superior detection capability and computational efficiency compared to existing approaches.
>
---
#### [new 080] HairWeaver: Few-Shot Photorealistic Hair Motion Synthesis with Sim-to-Real Guided Video Diffusion
- **分类: cs.CV**

- **简介: 论文提出HairWeaver，解决人体头发动态生成问题。通过扩散模型和两个模块，实现逼真、自然的头发动画，提升动画真实感。**

- **链接: [https://arxiv.org/pdf/2602.11117v1](https://arxiv.org/pdf/2602.11117v1)**

> **作者:** Di Chang; Ji Hou; Aljaz Bozic; Assaf Neuberger; Felix Juefei-Xu; Olivier Maury; Gene Wei-Chin Lin; Tuur Stuyck; Doug Roble; Mohammad Soleymani; Stephane Grabli
>
> **备注:** Website: https://boese0601.github.io/hairweaver/
>
> **摘要:** We present HairWeaver, a diffusion-based pipeline that animates a single human image with realistic and expressive hair dynamics. While existing methods successfully control body pose, they lack specific control over hair, and as a result, fail to capture the intricate hair motions, resulting in stiff and unrealistic animations. HairWeaver overcomes this limitation using two specialized modules: a Motion-Context-LoRA to integrate motion conditions and a Sim2Real-Domain-LoRA to preserve the subject's photoreal appearance across different data domains. These lightweight components are designed to guide a video diffusion backbone while maintaining its core generative capabilities. By training on a specialized dataset of dynamic human motion generated from a CG simulator, HairWeaver affords fine control over hair motion and ultimately learns to produce highly realistic hair that responds naturally to movement. Comprehensive evaluations demonstrate that our approach sets a new state of the art, producing lifelike human hair animations with dynamic details.
>
---
#### [new 081] DFIC: Towards a balanced facial image dataset for automatic ICAO compliance verification
- **分类: cs.CV**

- **简介: 该论文提出DFIC数据集，用于自动验证ICAO合规性，解决手动检查效率低的问题，通过大量标注数据提升识别准确性。**

- **链接: [https://arxiv.org/pdf/2602.10985v1](https://arxiv.org/pdf/2602.10985v1)**

> **作者:** Nuno Gonçalves; Diogo Nunes; Carla Guerra; João Marcos
>
> **摘要:** Ensuring compliance with ISO/IEC and ICAO standards for facial images in machine-readable travel documents (MRTDs) is essential for reliable identity verification, but current manual inspection methods are inefficient in high-demand environments. This paper introduces the DFIC dataset, a novel comprehensive facial image dataset comprising around 58,000 annotated images and 2706 videos of more than 1000 subjects, that cover a broad range of non-compliant conditions, in addition to compliant portraits. Our dataset provides a more balanced demographic distribution than the existing public datasets, with one partition that is nearly uniformly distributed, facilitating the development of automated ICAO compliance verification methods. Using DFIC, we fine-tuned a novel method that heavily relies on spatial attention mechanisms for the automatic validation of ICAO compliance requirements, and we have compared it with the state-of-the-art aimed at ICAO compliance verification, demonstrating improved results. DFIC dataset is now made public (https://github.com/visteam-isr-uc/DFIC) for the training and validation of new models, offering an unprecedented diversity of faces, that will improve both robustness and adaptability to the intrinsically diverse combinations of faces and props that can be presented to the validation system. These results emphasize the potential of DFIC to enhance automated ICAO compliance methods but it can also be used in many other applications that aim to improve the security, privacy, and fairness of facial recognition systems.
>
---
#### [new 082] Multimodal Priors-Augmented Text-Driven 3D Human-Object Interaction Generation
- **分类: cs.CV**

- **简介: 该论文属于文本驱动的3D人-物交互生成任务，旨在解决人类动作不优、物体动作不自然及交互弱的问题。通过引入多模态先验、改进物体表示、设计混合专家模型和级联扩散框架来提升生成效果。**

- **链接: [https://arxiv.org/pdf/2602.10659v1](https://arxiv.org/pdf/2602.10659v1)**

> **作者:** Yin Wang; Ziyao Zhang; Zhiying Leng; Haitian Liu; Frederick W. B. Li; Mu Li; Xiaohui Liang
>
> **摘要:** We address the challenging task of text-driven 3D human-object interaction (HOI) motion generation. Existing methods primarily rely on a direct text-to-HOI mapping, which suffers from three key limitations due to the significant cross-modality gap: (Q1) sub-optimal human motion, (Q2) unnatural object motion, and (Q3) weak interaction between humans and objects. To address these challenges, we propose MP-HOI, a novel framework grounded in four core insights: (1) Multimodal Data Priors: We leverage multimodal data (text, image, pose/object) from large multimodal models as priors to guide HOI generation, which tackles Q1 and Q2 in data modeling. (2) Enhanced Object Representation: We improve existing object representations by incorporating geometric keypoints, contact features, and dynamic properties, enabling expressive object representations, which tackles Q2 in data representation. (3) Multimodal-Aware Mixture-of-Experts (MoE) Model: We propose a modality-aware MoE model for effective multimodal feature fusion paradigm, which tackles Q1 and Q2 in feature fusion. (4) Cascaded Diffusion with Interaction Supervision: We design a cascaded diffusion framework that progressively refines human-object interaction features under dedicated supervision, which tackles Q3 in interaction refinement. Comprehensive experiments demonstrate that MP-HOI outperforms existing approaches in generating high-fidelity and fine-grained HOI motions.
>
---
#### [new 083] VideoSTF: Stress-Testing Output Repetition in Video Large Language Models
- **分类: cs.CV; cs.CR; cs.MM**

- **简介: 该论文属于视频语言模型研究，针对视频大模型中的输出重复问题进行测试与分析，提出VideoSTF框架评估模型稳定性。**

- **链接: [https://arxiv.org/pdf/2602.10639v1](https://arxiv.org/pdf/2602.10639v1)**

> **作者:** Yuxin Cao; Wei Song; Shangzhi Xu; Jingling Xue; Jin Song Dong
>
> **摘要:** Video Large Language Models (VideoLLMs) have recently achieved strong performance in video understanding tasks. However, we identify a previously underexplored generation failure: severe output repetition, where models degenerate into self-reinforcing loops of repeated phrases or sentences. This failure mode is not captured by existing VideoLLM benchmarks, which focus primarily on task accuracy and factual correctness. We introduce VideoSTF, the first framework for systematically measuring and stress-testing output repetition in VideoLLMs. VideoSTF formalizes repetition using three complementary n-gram-based metrics and provides a standardized testbed of 10,000 diverse videos together with a library of controlled temporal transformations. Using VideoSTF, we conduct pervasive testing, temporal stress testing, and adversarial exploitation across 10 advanced VideoLLMs. We find that output repetition is widespread and, critically, highly sensitive to temporal perturbations of video inputs. Moreover, we show that simple temporal transformations can efficiently induce repetitive degeneration in a black-box setting, exposing output repetition as an exploitable security vulnerability. Our results reveal output repetition as a fundamental stability issue in modern VideoLLMs and motivate stability-aware evaluation for video-language systems. Our evaluation code and scripts are available at: https://github.com/yuxincao22/VideoSTF_benchmark.
>
---
#### [new 084] A Systematic Review on Data-Driven Brain Deformation Modeling for Image-Guided Neurosurgery
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学影像任务，旨在解决神经外科中脑组织变形的补偿问题。通过系统综述，分析了2020至2025年间AI驱动的脑变形建模方法及其性能评估。**

- **链接: [https://arxiv.org/pdf/2602.10155v1](https://arxiv.org/pdf/2602.10155v1)**

> **作者:** Tiago Assis; Colin P. Galvin; Joshua P. Castillo; Nazim Haouchine; Marta Kersten-Oertel; Zeyu Gao; Mireia Crispin-Ortuzar; Stephen J. Price; Thomas Santarius; Yangming Ou; Sarah Frisken; Nuno C. Garcia; Alexandra J. Golby; Reuben Dorent; Ines P. Machado
>
> **备注:** 31 pages, 7 figures, 3 tables. Submitted to Medical Image Analysis
>
> **摘要:** Accurate compensation of brain deformation is a critical challenge for reliable image-guided neurosurgery, as surgical manipulation and tumor resection induce tissue motion that misaligns preoperative planning images with intraoperative anatomy and longitudinal studies. In this systematic review, we synthesize recent AI-driven approaches developed between January 2020 and April 2025 for modeling and correcting brain deformation. A comprehensive literature search was conducted in PubMed, IEEE Xplore, Scopus, and Web of Science, with predefined inclusion and exclusion criteria focused on computational methods applied to brain deformation compensation for neurosurgical imaging, resulting in 41 studies meeting these criteria. We provide a unified analysis of methodological strategies, including deep learning-based image registration, direct deformation field regression, synthesis-driven multimodal alignment, resection-aware architectures addressing missing correspondences, and hybrid models that integrate biomechanical priors. We also examine dataset utilization, reported evaluation metrics, validation protocols, and how uncertainty and generalization have been assessed across studies. While AI-based deformation models demonstrate promising performance and computational efficiency, current approaches exhibit limitations in out-of-distribution robustness, standardized benchmarking, interpretability, and readiness for clinical deployment. Our review highlights these gaps and outlines opportunities for future research aimed at achieving more robust, generalizable, and clinically translatable deformation compensation solutions for neurosurgical guidance. By organizing recent advances and critically evaluating evaluation practices, this work provides a comprehensive foundation for researchers and clinicians engaged in developing and applying AI-based brain deformation methods.
>
---
#### [new 085] From Circuits to Dynamics: Understanding and Stabilizing Failure in 3D Diffusion Transformers
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究3D点云补全任务，针对扩散Transformer模型在输入微小扰动下出现输出碎片化的故障（Meltdown），通过分析注意力机制和扩散动力学提出稳定方法PowerRemap。**

- **链接: [https://arxiv.org/pdf/2602.11130v1](https://arxiv.org/pdf/2602.11130v1)**

> **作者:** Maximilian Plattner; Fabian Paischer; Johannes Brandstetter; Arturs Berzins
>
> **摘要:** Reliable surface completion from sparse point clouds underpins many applications spanning content creation and robotics. While 3D diffusion transformers attain state-of-the-art results on this task, we uncover that they exhibit a catastrophic mode of failure: arbitrarily small on-surface perturbations to the input point cloud can fracture the output into multiple disconnected pieces -- a phenomenon we call Meltdown. Using activation-patching from mechanistic interpretability, we localize Meltdown to a single early denoising cross-attention activation. We find that the singular-value spectrum of this activation provides a scalar proxy: its spectral entropy rises when fragmentation occurs and returns to baseline when patched. Interpreted through diffusion dynamics, we show that this proxy tracks a symmetry-breaking bifurcation of the reverse process. Guided by this insight, we introduce PowerRemap, a test-time control that stabilizes sparse point-cloud conditioning. We demonstrate that Meltdown persists across state-of-the-art architectures (WaLa, Make-a-Shape), datasets (GSO, SimJEB) and denoising strategies (DDPM, DDIM), and that PowerRemap effectively counters this failure with stabilization rates of up to 98.3%. Overall, this work is a case study on how diffusion model behavior can be understood and guided based on mechanistic analysis, linking a circuit-level cross-attention mechanism to diffusion-dynamics accounts of trajectory bifurcations.
>
---
#### [new 086] SecureScan: An AI-Driven Multi-Layer Framework for Malware and Phishing Detection Using Logistic Regression and Threat Intelligence Integration
- **分类: cs.CR; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于恶意软件和网络钓鱼检测任务，旨在解决传统系统效果下降的问题。提出SecureScan框架，结合逻辑回归、启发式分析和威胁情报，提升检测准确性和稳定性。**

- **链接: [https://arxiv.org/pdf/2602.10750v1](https://arxiv.org/pdf/2602.10750v1)**

> **作者:** Rumman Firdos; Aman Dangi
>
> **摘要:** The growing sophistication of modern malware and phishing campaigns has diminished the effectiveness of traditional signature-based intrusion detection systems. This work presents SecureScan, an AI-driven, triple-layer detection framework that integrates logistic regression-based classification, heuristic analysis, and external threat intelligence via the VirusTotal API for comprehensive triage of URLs, file hashes, and binaries. The proposed architecture prioritizes efficiency by filtering known threats through heuristics, classifying uncertain samples using machine learning, and validating borderline cases with third-party intelligence. On benchmark datasets, SecureScan achieves 93.1 percent accuracy with balanced precision (0.87) and recall (0.92), demonstrating strong generalization and reduced overfitting through threshold-based decision calibration. A calibrated threshold and gray-zone logic (0.45-0.55) were introduced to minimize false positives and enhance real-world stability. Experimental results indicate that a lightweight statistical model, when augmented with calibrated verification and external intelligence, can achieve reliability and performance comparable to more complex deep learning systems.
>
---
#### [new 087] Kill it with FIRE: On Leveraging Latent Space Directions for Runtime Backdoor Mitigation in Deep Neural Networks
- **分类: cs.LG; cs.AI; cs.CR; cs.CV**

- **简介: 该论文属于模型安全任务，解决后门攻击问题。提出FIRE方法，在推理阶段通过调整隐空间方向消除触发器，实现高效防御。**

- **链接: [https://arxiv.org/pdf/2602.10780v1](https://arxiv.org/pdf/2602.10780v1)**

> **作者:** Enrico Ahlers; Daniel Passon; Yannic Noller; Lars Grunske
>
> **摘要:** Machine learning models are increasingly present in our everyday lives; as a result, they become targets of adversarial attackers seeking to manipulate the systems we interact with. A well-known vulnerability is a backdoor introduced into a neural network by poisoned training data or a malicious training process. Backdoors can be used to induce unwanted behavior by including a certain trigger in the input. Existing mitigations filter training data, modify the model, or perform expensive input modifications on samples. If a vulnerable model has already been deployed, however, those strategies are either ineffective or inefficient. To address this gap, we propose our inference-time backdoor mitigation approach called FIRE (Feature-space Inference-time REpair). We hypothesize that a trigger induces structured and repeatable changes in the model's internal representation. We view the trigger as directions in the latent spaces between layers that can be applied in reverse to correct the inference mechanism. Therefore, we turn the backdoored model against itself by manipulating its latent representations and moving a poisoned sample's features along the backdoor directions to neutralize the trigger. Our evaluation shows that FIRE has low computational overhead and outperforms current runtime mitigations on image benchmarks across various attacks, datasets, and network architectures.
>
---
#### [new 088] Viewpoint Recommendation for Point Cloud Labeling through Interaction Cost Modeling
- **分类: cs.HC; cs.CV**

- **简介: 该论文属于点云标注任务，旨在降低标注时间成本。通过建模lasso选择时间，推荐最优视角以提高标注效率。**

- **链接: [https://arxiv.org/pdf/2602.10871v1](https://arxiv.org/pdf/2602.10871v1)**

> **作者:** Yu Zhang; Xinyi Zhao; Chongke Bi; Siming Chen
>
> **备注:** Accepted to IEEE TVCG
>
> **摘要:** Semantic segmentation of 3D point clouds is important for many applications, such as autonomous driving. To train semantic segmentation models, labeled point cloud segmentation datasets are essential. Meanwhile, point cloud labeling is time-consuming for annotators, which typically involves tuning the camera viewpoint and selecting points by lasso. To reduce the time cost of point cloud labeling, we propose a viewpoint recommendation approach to reduce annotators' labeling time costs. We adapt Fitts' law to model the time cost of lasso selection in point clouds. Using the modeled time cost, the viewpoint that minimizes the lasso selection time cost is recommended to the annotator. We build a data labeling system for semantic segmentation of 3D point clouds that integrates our viewpoint recommendation approach. The system enables users to navigate to recommended viewpoints for efficient annotation. Through an ablation study, we observed that our approach effectively reduced the data labeling time cost. We also qualitatively compare our approach with previous viewpoint selection approaches on different datasets.
>
---
#### [new 089] SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes
- **分类: cs.RO; cs.AI; cs.CV; cs.GR**

- **简介: 该论文提出SceneSmith，用于生成逼真室内场景的模拟环境，解决现有场景缺乏多样性和物理复杂性的问题。通过自然语言生成高质量仿真场景，提升机器人训练效果。**

- **链接: [https://arxiv.org/pdf/2602.09153v1](https://arxiv.org/pdf/2602.09153v1)**

> **作者:** Nicholas Pfaff; Thomas Cohn; Sergey Zakharov; Rick Cory; Russ Tedrake
>
> **备注:** Project page: https://scenesmith.github.io/
>
> **摘要:** Simulation has become a key tool for training and evaluating home robots at scale, yet existing environments fail to capture the diversity and physical complexity of real indoor spaces. Current scene synthesis methods produce sparsely furnished rooms that lack the dense clutter, articulated furniture, and physical properties essential for robotic manipulation. We introduce SceneSmith, a hierarchical agentic framework that generates simulation-ready indoor environments from natural language prompts. SceneSmith constructs scenes through successive stages$\unicode{x2013}$from architectural layout to furniture placement to small object population$\unicode{x2013}$each implemented as an interaction among VLM agents: designer, critic, and orchestrator. The framework tightly integrates asset generation through text-to-3D synthesis for static objects, dataset retrieval for articulated objects, and physical property estimation. SceneSmith generates 3-6x more objects than prior methods, with <2% inter-object collisions and 96% of objects remaining stable under physics simulation. In a user study with 205 participants, it achieves 92% average realism and 91% average prompt faithfulness win rates against baselines. We further demonstrate that these environments can be used in an end-to-end pipeline for automatic robot policy evaluation.
>
---
#### [new 090] GENIUS: Generative Fluid Intelligence Evaluation Suite
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出GENIUS基准，用于评估生成模型的动态推理能力。针对现有基准仅测试记忆知识的不足，GENIUS聚焦于生成流体智能，涵盖模式归纳、约束执行和情境适应三项任务，旨在推动模型向灵活推理发展。**

- **链接: [https://arxiv.org/pdf/2602.11144v1](https://arxiv.org/pdf/2602.11144v1)**

> **作者:** Ruichuan An; Sihan Yang; Ziyu Guo; Wei Dai; Zijun Shen; Haodong Li; Renrui Zhang; Xinyu Wei; Guopeng Li; Wenshan Wu; Wentao Zhang
>
> **摘要:** Unified Multimodal Models (UMMs) have shown remarkable progress in visual generation. Yet, existing benchmarks predominantly assess $\textit{Crystallized Intelligence}$, which relies on recalling accumulated knowledge and learned schemas. This focus overlooks $\textit{Generative Fluid Intelligence (GFI)}$: the capacity to induce patterns, reason through constraints, and adapt to novel scenarios on the fly. To rigorously assess this capability, we introduce $\textbf{GENIUS}$ ($\textbf{GEN}$ Fluid $\textbf{I}$ntelligence Eval$\textbf{U}$ation $\textbf{S}$uite). We formalize $\textit{GFI}$ as a synthesis of three primitives. These include $\textit{Inducing Implicit Patterns}$ (e.g., inferring personalized visual preferences), $\textit{Executing Ad-hoc Constraints}$ (e.g., visualizing abstract metaphors), and $\textit{Adapting to Contextual Knowledge}$ (e.g., simulating counter-intuitive physics). Collectively, these primitives challenge models to solve problems grounded entirely in the immediate context. Our systematic evaluation of 12 representative models reveals significant performance deficits in these tasks. Crucially, our diagnostic analysis disentangles these failure modes. It demonstrates that deficits stem from limited context comprehension rather than insufficient intrinsic generative capability. To bridge this gap, we propose a training-free attention intervention strategy. Ultimately, $\textbf{GENIUS}$ establishes a rigorous standard for $\textit{GFI}$, guiding the field beyond knowledge utilization toward dynamic, general-purpose reasoning. Our dataset and code will be released at: $\href{https://github.com/arctanxarc/GENIUS}{https://github.com/arctanxarc/GENIUS}$.
>
---
#### [new 091] ENIGMA: EEG-to-Image in 15 Minutes Using Less Than 1% of the Parameters
- **分类: q-bio.NC; cs.AI; cs.CV; cs.HC**

- **简介: 该论文提出ENIGMA模型，解决EEG到图像重建任务，旨在提升模型部署效率和泛化能力，通过简化架构和减少参数实现快速适应新受试者。**

- **链接: [https://arxiv.org/pdf/2602.10361v1](https://arxiv.org/pdf/2602.10361v1)**

> **作者:** Reese Kneeland; Wangshu Jiang; Ugo Bruzadin Nunes; Paul Steven Scotti; Arnaud Delorme; Jonathan Xu
>
> **摘要:** To be practical for real-life applications, models for brain-computer interfaces must be easily and quickly deployable on new subjects, effective on affordable scanning hardware, and small enough to run locally on accessible computing resources. To directly address these current limitations, we introduce ENIGMA, a multi-subject electroencephalography (EEG)-to-Image decoding model that reconstructs seen images from EEG recordings and achieves state-of-the-art (SOTA) performance on the research-grade THINGS-EEG2 and consumer-grade AllJoined-1.6M benchmarks, while fine-tuning effectively on new subjects with as little as 15 minutes of data. ENIGMA boasts a simpler architecture and requires less than 1% of the trainable parameters necessary for previous approaches. Our approach integrates a subject-unified spatio-temporal backbone along with a set of multi-subject latent alignment layers and an MLP projector to map raw EEG signals to a rich visual latent space. We evaluate our approach using a broad suite of image reconstruction metrics that have been standardized in the adjacent field of fMRI-to-Image research, and we describe the first EEG-to-Image study to conduct extensive behavioral evaluations of our reconstructions using human raters. Our simple and robust architecture provides a significant performance boost across both research-grade and consumer-grade EEG hardware, and a substantial improvement in fine-tuning efficiency and inference cost. Finally, we provide extensive ablations to determine the architectural choices most responsible for our performance gains in both single and multi-subject cases across multiple benchmark datasets. Collectively, our work provides a substantial step towards the development of practical brain-computer interface applications.
>
---
#### [new 092] ContactGaussian-WM: Learning Physics-Grounded World Model from Videos
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人感知与建模任务，旨在解决数据稀缺下复杂物理交互建模问题。提出ContactGaussian-WM框架，通过视觉与物理联合学习，提升环境建模精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.11021v1](https://arxiv.org/pdf/2602.11021v1)**

> **作者:** Meizhong Wang; Wanxin Jin; Kun Cao; Lihua Xie; Yiguang Hong
>
> **摘要:** Developing world models that understand complex physical interactions is essential for advancing robotic planning and simulation.However, existing methods often struggle to accurately model the environment under conditions of data scarcity and complex contact-rich dynamic motion.To address these challenges, we propose ContactGaussian-WM, a differentiable physics-grounded rigid-body world model capable of learning intricate physical laws directly from sparse and contact-rich video sequences.Our framework consists of two core components: (1) a unified Gaussian representation for both visual appearance and collision geometry, and (2) an end-to-end differentiable learning framework that differentiates through a closed-form physics engine to infer physical properties from sparse visual observations.Extensive simulations and real-world evaluations demonstrate that ContactGaussian-WM outperforms state-of-the-art methods in learning complex scenarios, exhibiting robust generalization capabilities.Furthermore, we showcase the practical utility of our framework in downstream applications, including data synthesis and real-time MPC.
>
---
#### [new 093] Uncertainty-Aware Ordinal Deep Learning for cross-Dataset Diabetic Retinopathy Grading
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于糖尿病视网膜病变分级任务，旨在解决跨数据集的准确且可靠诊断问题。提出一种考虑不确定性的序数深度学习方法，提升模型泛化能力和可信度。**

- **链接: [https://arxiv.org/pdf/2602.10315v1](https://arxiv.org/pdf/2602.10315v1)**

> **作者:** Ali El Bellaj; Aya Benradi; Salman El Youssoufi; Taha El Marzouki; Mohammed-Amine Cheddadi
>
> **摘要:** Diabetes mellitus is a chronic metabolic disorder characterized by persistent hyperglycemia due to insufficient insulin production or impaired insulin utilization. One of its most severe complications is diabetic retinopathy (DR), a progressive retinal disease caused by microvascular damage, leading to hemorrhages, exudates, and potential vision loss. Early and reliable detection of DR is therefore critical for preventing irreversible blindness. In this work, we propose an uncertainty-aware deep learning framework for automated DR severity grading that explicitly models the ordinal nature of disease progression. Our approach combines a convolutional backbone with lesion-query attention pooling and an evidential Dirichlet-based ordinal regression head, enabling both accurate severity prediction and principled estimation of predictive uncertainty. The model is trained using an ordinal evidential loss with annealed regularization to encourage calibrated confidence under domain shift. We evaluate the proposed method on a multi-domain training setup combining APTOS, Messidor-2, and a subset of EyePACS fundus datasets. Experimental results demonstrate strong cross-dataset generalization, achieving competitive classification accuracy and high quadratic weighted kappa on held-out test sets, while providing meaningful uncertainty estimates for low-confidence cases. These results suggest that ordinal evidential learning is a promising direction for robust and clinically reliable diabetic retinopathy grading.
>
---
#### [new 094] From Representational Complementarity to Dual Systems: Synergizing VLM and Vision-Only Backbones for End-to-End Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在提升端到端驾驶规划性能。通过结合视觉语言模型与纯视觉模型，探索两者互补性，提出混合与双策略系统以提高表现。**

- **链接: [https://arxiv.org/pdf/2602.10719v1](https://arxiv.org/pdf/2602.10719v1)**

> **作者:** Sining Ang; Yuguang Yang; Chenxu Dang; Canyu Chen; Cheng Chi; Haiyan Liu; Xuanyao Mao; Jason Bao; Xuliang; Bingchuan Sun; Yan Wang
>
> **备注:** 22 pages (10 pages main text + 12 pages appendix), 18 figures
>
> **摘要:** Vision-Language-Action (VLA) driving augments end-to-end (E2E) planning with language-enabled backbones, yet it remains unclear what changes beyond the usual accuracy--cost trade-off. We revisit this question with 3--RQ analysis in RecogDrive by instantiating the system with a full VLM and vision-only backbones, all under an identical diffusion Transformer planner. RQ1: At the backbone level, the VLM can introduce additional subspaces upon the vision-only backbones. RQ2: This unique subspace leads to a different behavioral in some long-tail scenario: the VLM tends to be more aggressive whereas ViT is more conservative, and each decisively wins on about 2--3% of test scenarios; With an oracle that selects, per scenario, the better trajectory between the VLM and ViT branches, we obtain an upper bound of 93.58 PDMS. RQ3: To fully harness this observation, we propose HybridDriveVLA, which runs both ViT and VLM branches and selects between their endpoint trajectories using a learned scorer, improving PDMS to 92.10. Finally, DualDriveVLA implements a practical fast--slow policy: it runs ViT by default and invokes the VLM only when the scorer's confidence falls below a threshold; calling the VLM on 15% of scenarios achieves 91.00 PDMS while improving throughput by 3.2x. Code will be released.
>
---
#### [new 095] Beyond Calibration: Confounding Pathology Limits Foundation Model Specificity in Abdominal Trauma CT
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决基础模型在腹部创伤CT中特异性不足的问题。通过对比不同模型在不同病理情况下的表现，发现特异性下降主要由负类异质性引起。**

- **链接: [https://arxiv.org/pdf/2602.10359v1](https://arxiv.org/pdf/2602.10359v1)**

> **作者:** Jineel H Raythatha; Shuchang Ye; Jeremy Hsu; Jinman Kim
>
> **备注:** 26 pages, 4 figures, 4 tables
>
> **摘要:** Purpose: Translating foundation models into clinical practice requires evaluating their performance under compound distribution shift, where severe class imbalance coexists with heterogeneous imaging appearances. This challenge is relevant for traumatic bowel injury, a rare but high-mortality diagnosis. We investigated whether specificity deficits in foundation models are associated with heterogeneity in the negative class. Methods: This retrospective study used the multi-institutional, RSNA Abdominal Traumatic Injury CT dataset (2019-2023), comprising scans from 23 centres. Two foundation models (MedCLIP, zero-shot; RadDINO, linear probe) were compared against three task-specific approaches (CNN, Transformer, Ensemble). Models were trained on 3,147 patients (2.3% bowel injury prevalence) and evaluated on an enriched 100-patient test set. To isolate negative-class effects, specificity was assessed in patients without bowel injury who had concurrent solid organ injury (n=58) versus no abdominal pathology (n=50). Results: Foundation models achieved equivalent discrimination to task-specific models (AUC, 0.64-0.68 versus 0.58-0.64) with higher sensitivity (79-91% vs 41-74%) but lower specificity (33-50% vs 50-88%). All models demonstrated high specificity in patients without abdominal pathology (84-100%). When solid organ injuries were present, specificity declined substantially for foundation models (50-51 percentage points) compared with smaller reductions of 12-41 percentage points for task-specific models. Conclusion: Foundation models matched task-specific discrimination without task-specific training, but their specificity deficits were driven primarily by confounding negative-class heterogeneity rather than prevalence alone. Susceptibility to negative-class heterogeneity decreased progressively with labelled training, suggesting adaptation is required before clinical implementation.
>
---
#### [new 096] URBAN-SPIN: A street-level bikeability index to inform design implementations in historical city centres
- **分类: physics.soc-ph; cs.CV; cs.CY**

- **简介: 该论文属于城市规划与交通研究任务，旨在解决历史城区骑行体验评估问题。通过构建街景感知框架，整合数据与主观评价，提出可迁移的自行车友好度指数。**

- **链接: [https://arxiv.org/pdf/2602.10124v1](https://arxiv.org/pdf/2602.10124v1)**

> **作者:** Haining Ding; Chenxi Wang; Michal Gath-Morad
>
> **备注:** 32 pages, 10 figures
>
> **摘要:** Cycling is reported by an average of 35\% of adults at least once per week across 28 countries, and as vulnerable road users directly exposed to their surroundings, cyclists experience the street at an intensity unmatched by other modes. Yet the street-level features that shape this experience remain under-analysed, particularly in historical urban contexts where spatial constraints rule out large-scale infrastructural change and where typological context is often overlooked. This study develops a perception-led, typology-based, and data-integrated framework that explicitly models street typologies and their sub-classifications to evaluate how visual and spatial configurations shape cycling experience. Drawing on the Cambridge Cycling Experience Video Dataset (CCEVD), a first-person and handlebar-mounted corpus developed in this study, we extract fine-grained streetscape indicators with computer vision and pair them with built-environment variables and subjective ratings from a Balanced Incomplete Block Design (BIBD) survey, thereby constructing a typology-sensitive Bikeability Index that integrates subjective and perceived dimensions with physical metrics for segment-level comparison. Statistical analysis shows that perceived bikeability arises from cumulative, context-specific interactions among features. While greenness and openness consistently enhance comfort and pleasure, enclosure, imageability, and building continuity display threshold or divergent effects contingent on street type and subtype. AI-assisted visual redesigns further demonstrate that subtle, targeted changes can yield meaningful perceptual gains without large-scale structural interventions. The framework offers a transferable model for evaluating and improving cycling conditions in heritage cities through perceptually attuned, typology-aware design strategies.
>
---
## 更新

#### [replaced 001] Deformation-Recovery Diffusion Model (DRDM): Instance Deformation for Image Manipulation and Synthesis
- **分类: eess.IV; cs.CE; cs.CV**

- **链接: [https://arxiv.org/pdf/2407.07295v3](https://arxiv.org/pdf/2407.07295v3)**

> **作者:** Jian-Qing Zheng; Yuanhan Mo; Yang Sun; Jiahua Li; Fuping Wu; Ziyang Wang; Tonia Vincent; Bartłomiej W. Papież
>
> **备注:** accepted by Medical Image Analysis
>
> **摘要:** In medical imaging, the diffusion models have shown great potential for synthetic image generation tasks. However, these approaches often lack the interpretable connections between the generated and real images and can create anatomically implausible structures or illusions. To address these limitations, we propose the Deformation-Recovery Diffusion Model (DRDM), a novel diffusion-based generative model that emphasises morphological transformation through deformation fields rather than direct image synthesis. DRDM introduces a topology-preserving deformation field generation strategy, which randomly samples and integrates multi-scale Deformation Velocity Fields (DVFs). DRDM is trained to learn to recover unrealistic deformation components, thus restoring randomly deformed images to a realistic distribution. This formulation enables the generation of diverse yet anatomically plausible deformations that preserve structural integrity, thereby improving data augmentation and synthesis for downstream tasks such as few-shot learning and image registration. Experiments on cardiac Magnetic Resonance Imaging and pulmonary Computed Tomography show that DRDM is capable of creating diverse, large-scale deformations, while maintaining anatomical plausibility of deformation fields. Additional evaluations on 2D image segmentation and 3D image registration tasks indicate notable performance gains, underscoring DRDM's potential to enhance both image manipulation and generative modelling in medical imaging applications. Project page: https://jianqingzheng.github.io/def_diff_rec/
>
---
#### [replaced 002] Robust Vision Systems for Connected and Autonomous Vehicles: Security Challenges and Attack Vectors
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.09740v2](https://arxiv.org/pdf/2602.09740v2)**

> **作者:** Sandeep Gupta; Roberto Passerone
>
> **备注:** Submitted to IEEE Transactions on Intelligent Vehicles
>
> **摘要:** This article investigates the robustness of vision systems in Connected and Autonomous Vehicles (CAVs), which is critical for developing Level-5 autonomous driving capabilities. Safe and reliable CAV navigation undeniably depends on robust vision systems that enable accurate detection of objects, lane markings, and traffic signage. We analyze the key sensors and vision components essential for CAV navigation to derive a reference architecture for CAV vision system (CAVVS). This reference architecture provides a basis for identifying potential attack surfaces of CAVVS. Subsequently, we elaborate on identified attack vectors targeting each attack surface, rigorously evaluating their implications for confidentiality, integrity, and availability (CIA). Our study provides a comprehensive understanding of attack vector dynamics in vision systems, which is crucial for formulating robust security measures that can uphold the principles of the CIA triad.
>
---
#### [replaced 003] ChartMuseum: Testing Visual Reasoning Capabilities of Large Vision-Language Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉推理任务，旨在评估大视觉-语言模型的图表理解能力。研究发现现有模型在视觉推理上表现不足，提出ChartMuseum基准以更准确地衡量模型与人类的表现差异。**

- **链接: [https://arxiv.org/pdf/2505.13444v3](https://arxiv.org/pdf/2505.13444v3)**

> **作者:** Liyan Tang; Grace Kim; Xinyu Zhao; Thom Lake; Wenxuan Ding; Fangcong Yin; Prasann Singhal; Manya Wadhwa; Zeyu Leo Liu; Zayne Sprague; Ramya Namuduri; Bodun Hu; Juan Diego Rodriguez; Puyuan Peng; Greg Durrett
>
> **备注:** NeurIPS 2025 Datasets & Benchmarks
>
> **摘要:** Chart understanding presents a unique challenge for large vision-language models (LVLMs), as it requires the integration of sophisticated textual and visual reasoning capabilities. However, current LVLMs exhibit a notable imbalance between these skills, falling short on visual reasoning that is difficult to perform in text. We conduct a case study using a synthetic dataset solvable only through visual reasoning and show that model performance degrades significantly with increasing visual complexity, while human performance remains robust. We then introduce ChartMuseum, a new Chart Question Answering (QA) benchmark containing 1,162 expert-annotated questions spanning multiple reasoning types, curated from real-world charts across 184 sources, specifically built to evaluate complex visual and textual reasoning. Unlike prior chart understanding benchmarks -- where frontier models perform similarly and near saturation -- our benchmark exposes a substantial gap between model and human performance, while effectively differentiating model capabilities: although humans achieve 93% accuracy, the best-performing model Gemini-2.5-Pro attains only 63.0%, and the leading open-source LVLM Qwen2.5-VL-72B-Instruct achieves only 38.5%. Moreover, on questions requiring primarily visual reasoning, all models experience a 35%-55% performance drop from text-reasoning-heavy question performance. Lastly, our qualitative error analysis reveals specific categories of visual reasoning that are challenging for current LVLMs.
>
---
#### [replaced 004] Equivariant symmetry-aware head pose estimation for fetal MRI
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.04890v5](https://arxiv.org/pdf/2512.04890v5)**

> **作者:** Ramya Muthukrishnan; Borjan Gagoski; Aryn Lee; P. Ellen Grant; Elfar Adalsteinsson; Polina Golland; Benjamin Billot
>
> **摘要:** We present E(3)-Pose, a novel fast pose estimation method that jointly and explicitly models rotation equivariance and object symmetry. Our work is motivated by the challenging problem of accounting for fetal head motion during a diagnostic MRI scan. We aim to enable automatic adaptive prescription of 2D diagnostic MRI slices with 6-DoF head pose estimation, supported by 3D MRI volumes rapidly acquired before each 2D slice. Existing methods struggle to generalize to clinical volumes, due to pose ambiguities induced by inherent anatomical symmetries, as well as low resolution, noise, and artifacts. In contrast, E(3)-Pose captures anatomical symmetries and rigid pose equivariance by construction, and yields robust estimates of the fetal head pose. Our experiments on publicly available and representative clinical fetal MRI datasets demonstrate the superior robustness and generalization of our method across domains. Crucially, E(3)-Pose achieves state-of-the-art accuracy on clinical MRI volumes, supporting future clinical translation. Our implementation is publicly available at github.com/MedicalVisionGroup/E3-Pose.
>
---
#### [replaced 005] Unveiling the "Fairness Seesaw": Discovering and Mitigating Gender and Race Bias in Vision-Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于自然语言处理任务，旨在解决视觉-语言模型中的性别和种族偏见问题。通过分析模型内部机制，提出改进方法以提升公平性。**

- **链接: [https://arxiv.org/pdf/2505.23798v2](https://arxiv.org/pdf/2505.23798v2)**

> **作者:** Jian Lan; Udo Schlegel; Tanveer Hannan; Gengyuan Zhang; Haokun Chen; Thomas Seidl
>
> **摘要:** Although Vision-Language Models (VLMs) have achieved remarkable success, the knowledge mechanisms underlying their social biases remain a black box, where fairness- and ethics-related problems harm certain groups of people in society. It is unknown to what extent VLMs yield gender and race bias in generative responses. In this paper, we conduct a systematic discovery of gender and race bias in state-of-the-art VLMs, focusing not only on surface-level responses but also on the internal probability distributions and hidden state dynamics. Our empirical analysis reveals three critical findings: 1) The Fairness Paradox: Models often generate fair text labels while maintaining highly skewed confidence scores (mis-calibration) toward specific social groups. 2) Layer-wise Fluctuation: Fairness knowledge is not uniformly distributed; it peaks in intermediate layers and undergoes substantial knowledge erosion in the final layers. 3) Residual Discrepancy: Within a single hidden layer, different residual streams carry conflicting social knowledge - some reinforcing fairness while others amplifying bias. Leveraging these insights, we propose RES-FAIR (RESidual Flow Adjustment for Inference Recalibration), a post-hoc framework that mitigates bias by localizing and projecting hidden states away from biased residual directions while amplifying fair components. Evaluations on PAIRS and SocialCounterfactuals datasets demonstrate that our discovery-based approach significantly improves response fairness and confidence calibration without compromising general reasoning abilities. Our work provides a new lens for understanding how multi-modal models store and process sensitive social information.
>
---
#### [replaced 006] Generalization of Diffusion Models Arises with a Balanced Representation Space
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.20963v2](https://arxiv.org/pdf/2512.20963v2)**

> **作者:** Zekai Zhang; Xiao Li; Xiang Li; Lianghe Shi; Meng Wu; Molei Tao; Qing Qu
>
> **备注:** Accepted at ICLR 2026. 40 pages, 19 figures. The first two authors contributed equally
>
> **摘要:** Diffusion models excel at generating high-quality, diverse samples, yet they risk memorizing training data when overfit to the training objective. We analyze the distinctions between memorization and generalization in diffusion models through the lens of representation learning. By investigating a two-layer ReLU denoising autoencoder (DAE), we prove that (i) memorization corresponds to the model storing raw training samples in the learned weights for encoding and decoding, yielding localized spiky representations, whereas (ii) generalization arises when the model captures local data statistics, producing balanced representations. Furthermore, we validate these theoretical findings on real-world unconditional and text-to-image diffusion models, demonstrating that the same representation structures emerge in deep generative models with significant practical implications. Building on these insights, we propose a representation-based method for detecting memorization and a training-free editing technique that allows precise control via representation steering. Together, our results highlight that learning good representations is central to novel and meaningful generative modeling.
>
---
#### [replaced 007] Sim2real Image Translation Enables Viewpoint-Robust Policies from Fixed-Camera Datasets
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉机器人控制任务，解决相机视角变化导致的策略脆弱问题。通过提出MANGO方法，实现模拟到现实的图像翻译，提升策略的视角鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.09605v2](https://arxiv.org/pdf/2601.09605v2)**

> **作者:** Jeremiah Coholich; Justin Wit; Robert Azarcon; Zsolt Kira
>
> **摘要:** Vision-based policies for robot manipulation have achieved significant recent success, but are still brittle to distribution shifts such as camera viewpoint variations. Robot demonstration data is scarce and often lacks appropriate variation in camera viewpoints. Simulation offers a way to collect robot demonstrations at scale with comprehensive coverage of different viewpoints, but presents a visual sim2real challenge. To bridge this gap, we propose MANGO -- an unpaired image translation method with a novel segmentation-conditioned InfoNCE loss, a highly-regularized discriminator design, and a modified PatchNCE loss. We find that these elements are crucial for maintaining viewpoint consistency during sim2real translation. When training MANGO, we only require a small amount of fixed-camera data from the real world, but show that our method can generate diverse unseen viewpoints by translating simulated observations. In this domain, MANGO outperforms all other image translation methods we tested. Imitation-learning policies trained on data augmented by MANGO are able to achieve success rates as high as 60% on views that the non-augmented policy fails completely on.
>
---
#### [replaced 008] VeriSciQA: An Auto-Verified Dataset for Scientific Visual Question Answering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19899v3](https://arxiv.org/pdf/2511.19899v3)**

> **作者:** Yuyi Li; Daoyuan Chen; Zhen Wang; Yutong Lu; Yaliang Li
>
> **摘要:** Large Vision-Language Models (LVLMs) show promise for scientific applications, yet open-source models still struggle with Scientific Visual Question Answering (SVQA), namely answering questions about figures from scientific papers. A key bottleneck is the lack of public, large-scale, high-quality SVQA datasets. Although recent work uses LVLMs to synthesize data at scale, we identify systematic errors in their resulting QA pairs, stemming from LVLMs' inherent limitations and information asymmetry between figures and text. To address these challenges, we propose a Cross-Modal verification framework that generates questions and answers purely from figure-citing paragraphs, then verifies them against the figures themselves, leveraging the inherent text-figure alignment in scientific papers to filter out erroneous QA pairs. We instantiate this framework to curate VeriSciQA, a dataset of 20,272 QA pairs spanning 20 scientific domains and 12 figure types. Difficulty assessment reveals a notable accuracy gap between the best open-source model (65%) and the best proprietary model (80.5%), demonstrating room for improvement. Moreover, models fine-tuned on VeriSciQA achieve consistent improvements on SVQA benchmarks, with performance gains that scale with data size, surpassing models trained on existing datasets. Human evaluation further validates the improved quality of VeriSciQA. These results demonstrate that continued data expansion via our scalable framework can further advance SVQA capability in the open-source community. Our dataset is publicly available at https://huggingface.co/datasets/datajuicer/VeriSciQA.
>
---
#### [replaced 009] ADGaussian: Generalizable Gaussian Splatting for Autonomous Driving via Multi-modal Joint Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.00437v2](https://arxiv.org/pdf/2504.00437v2)**

> **作者:** Qi Song; Chenghong Li; Haotong Lin; Sida Peng; Rui Huang
>
> **备注:** The paper is accepted by ICRA 2026 and the project page can be found at https://maggiesong7.github.io/research/ADGaussian/
>
> **摘要:** We present a novel approach, termed ADGaussian, for generalizable street scene reconstruction. The proposed method enables high-quality rendering from merely single-view input. Unlike prior Gaussian Splatting methods that primarily focus on geometry refinement, we emphasize the importance of joint optimization of image and depth features for accurate Gaussian prediction. To this end, we first incorporate sparse LiDAR depth as an additional input modality, formulating the Gaussian prediction process as a joint learning framework of visual information and geometric clue. Furthermore, we propose a Multi-modal Feature Matching strategy coupled with a Multi-scale Gaussian Decoding model to enhance the joint refinement of multi-modal features, thereby enabling efficient multi-modal Gaussian learning. Extensive experiments on Waymo and KITTI demonstrate that our ADGaussian achieves state-of-the-art performance and exhibits superior zero-shot generalization capabilities in novel-view shifting.
>
---
#### [replaced 010] Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于强化学习任务，旨在解决DPO方法中数据利用效率低的问题。通过引入动态权重机制，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2506.10054v3](https://arxiv.org/pdf/2506.10054v3)**

> **作者:** Shangpin Peng; Weinong Wang; Zhuotao Tian; Senqiao Yang; Xing Wu; Haotian Xu; Chengquan Zhang; Takashi Isobe; Baotian Hu; Min Zhang
>
> **备注:** Accepted by ICLR 2026. Code & models: https://github.com/pspdada/Uni-DPO
>
> **摘要:** Direct Preference Optimization (DPO) has emerged as a cornerstone of reinforcement learning from human feedback (RLHF) due to its simplicity and efficiency. However, existing DPO-based methods typically treat all preference pairs equally, overlooking substantial variations in data quality and learning difficulty, which leads to inefficient data utilization and suboptimal performance. To address this limitation, we propose Uni-DPO, a unified dynamic preference optimization framework that jointly considers (a) the inherent quality of preference pairs and (b) the model's evolving performance during training. By adaptively reweighting samples based on both factors, Uni-DPO enables more effective use of preference data and achieves superior performance. Extensive experiments across models and benchmarks demonstrate the effectiveness and generalization of Uni-DPO. On textual tasks, Gemma-2-9B-IT fine-tuned with Uni-DPO surpasses the leading LLM, Claude 3 Opus, by 6.7 points on Arena-Hard. On mathematical and multimodal tasks, Uni-DPO consistently outperforms baseline methods across all benchmarks, providing strong empirical evidence of its effectiveness and robustness.
>
---
#### [replaced 011] Kelix Technique Report
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.09843v2](https://arxiv.org/pdf/2602.09843v2)**

> **作者:** Boyang Ding; Chenglong Chu; Dunju Zang; Han Li; Jiangxia Cao; Kun Gai; Muhao Wei; Ruiming Tang; Shiyao Wang; Siyang Mao; Xinchen Luo; Yahui Liu; Zhixin Ling; Zhuoran Yang; Ziming Li; Chengru Song; Guorui Zhou; Guowang Zhang; Hao Peng; Hao Wang; Jiaxin Deng; Jin Ouyang; Jinghao Zhang; Lejian Ren; Qianqian Wang; Qigen Hu; Tao Wang; Xingmei Wang; Yiping Yang; Zixing Zhang; Ziqi Wang
>
> **备注:** Work in progress
>
> **摘要:** Autoregressive large language models (LLMs) scale well by expressing diverse tasks as sequences of discrete natural-language tokens and training with next-token prediction, which unifies comprehension and generation under self-supervision. Extending this paradigm to multimodal data requires a shared, discrete representation across modalities. However, most vision-language models (VLMs) still rely on a hybrid interface: discrete text tokens paired with continuous Vision Transformer (ViT) features. Because supervision is largely text-driven, these models are often biased toward understanding and cannot fully leverage large-scale self-supervised learning on non-text data. Recent work has explored discrete visual tokenization to enable fully autoregressive multimodal modeling, showing promising progress toward unified understanding and generation. Yet existing discrete vision tokens frequently lose information due to limited code capacity, resulting in noticeably weaker understanding than continuous-feature VLMs. We present Kelix, a fully discrete autoregressive unified model that closes the understanding gap between discrete and continuous visual representations.
>
---
#### [replaced 012] Accelerating Streaming Video Large Language Models via Hierarchical Token Compression
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.00891v2](https://arxiv.org/pdf/2512.00891v2)**

> **作者:** Yiyu Wang; Xuyang Liu; Xiyan Gui; Xinying Lin; Boxue Yang; Chenfei Liao; Tailai Chen; Linfeng Zhang
>
> **备注:** Code is avaliable at \url{https://github.com/lern-to-write/STC}
>
> **摘要:** Streaming Video Large Language Models (VideoLLMs) have demonstrated impressive performance across various video understanding tasks, but they face significant challenges in real-time deployment due to the high computational cost of processing dense visual tokens from continuous video streams. In streaming video scenarios, the primary bottleneck lies in the Vision Transformer (ViT) encoding stage, where redundant processing of temporally similar frames leads to inefficiency. Additionally, inflated token sequences during LLM pre-filling further exacerbate latency and memory overhead. To address these challenges, we propose \textbf{S}treaming \textbf{T}oken \textbf{C}ompression (\textbf{STC}), a plug-and-play hierarchical framework that seamlessly integrates into existing streaming VideoLLMs, optimizing both ViT encoding and LLM pre-filling stages to accelerate processing. STC introduces two token-level accelerators: \textbf{STC-Cacher}, which reduces ViT encoding overhead by caching and reusing features from temporally similar frames, and \textbf{STC-Pruner}, which compresses the visual token sequence before it enters the LLM, preserving only the most salient tokens based on both spatial and temporal relevance. Extensive experiments on four baseline streaming VideoLLMs across five benchmarks demonstrate that STC outperforms other compression methods. Notably, STC retains up to \textbf{99\%} of accuracy on the ReKV framework while reducing ViT encoding latency and LLM pre-filling latency by \textbf{24.5\%} and \textbf{45.3\%}.
>
---
#### [replaced 013] Localized Control in Diffusion Models via Latent Vector Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.01991v2](https://arxiv.org/pdf/2602.01991v2)**

> **作者:** Pablo Domingo-Gregorio; Javier Ruiz-Hidalgo
>
> **摘要:** Diffusion models emerged as a leading approach in text-to-image generation, producing high-quality images from textual descriptions. However, attempting to achieve detailed control to get a desired image solely through text remains a laborious trial-and-error endeavor. Recent methods have introduced image-level controls alongside with text prompts, using prior images to extract conditional information such as edges, segmentation and depth maps. While effective, these methods apply conditions uniformly across the entire image, limiting localized control. In this paper, we propose a novel methodology to enable precise local control over user-defined regions of an image, while leaving to the diffusion model the task of autonomously generating the remaining areas according to the original prompt. Our approach introduces a new training framework that incorporates masking features and an additional loss term, which leverages the prediction of the initial latent vector at any diffusion step to enhance the correspondence between the current step and the final sample in the latent space. Extensive experiments demonstrate that our method effectively synthesizes high-quality images with controlled local conditions.
>
---
#### [replaced 014] CamReasoner: Reinforcing Camera Movement Understanding via Structured Spatial Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.00181v2](https://arxiv.org/pdf/2602.00181v2)**

> **作者:** Hang Wu; Yujun Cai; Zehao Li; Haonan Ge; Bowen Sun; Junsong Yuan; Yiwei Wang
>
> **摘要:** Understanding camera dynamics is a fundamental pillar of video spatial intelligence. However, existing multimodal models predominantly treat this task as a black-box classification, often confusing physically distinct motions by relying on superficial visual patterns rather than geometric cues. We present CamReasoner, a framework that reformulates camera movement understanding as a structured inference process to bridge the gap between perception and cinematic logic. Our approach centers on the Observation-Thinking-Answer (O-T-A) paradigm, which compels the model to decode spatio-temporal cues such as trajectories and view frustums within an explicit reasoning block. To instill this capability, we construct a Large-scale Inference Trajectory Suite comprising 18k SFT reasoning chains and 38k RL feedback samples. Notably, we are the first to employ RL for logical alignment in this domain, ensuring motion inferences are grounded in physical geometry rather than contextual guesswork. By applying Reinforcement Learning to the Observation-Think-Answer (O-T-A) reasoning paradigm, CamReasoner effectively suppresses hallucinations and achieves state-of-the-art performance across multiple benchmarks.
>
---
#### [replaced 015] Are Dense Labels Always Necessary for 3D Object Detection from Point Cloud?
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2403.02818v2](https://arxiv.org/pdf/2403.02818v2)**

> **作者:** Chenqiang Gao; Chuandong Liu; Jun Shu; Fangcen Liu; Jiang Liu; Luyu Yang; Xinbo Gao; Deyu Meng
>
> **备注:** update
>
> **摘要:** Current state-of-the-art (SOTA) 3D object detection methods often require a large amount of 3D bounding box annotations for training. However, collecting such large-scale densely-supervised datasets is notoriously costly. To reduce the cumbersome data annotation process, we propose a novel sparsely-annotated framework, in which we just annotate one 3D object per scene. Such a sparse annotation strategy could significantly reduce the heavy annotation burden, while inexact and incomplete sparse supervision may severely deteriorate the detection performance. To address this issue, we develop the SS3D++ method that alternatively improves 3D detector training and confident fully-annotated scene generation in a unified learning scheme. Using sparse annotations as seeds, we progressively generate confident fully-annotated scenes based on designing a missing-annotated instance mining module and reliable background mining module. Our proposed method produces competitive results when compared with SOTA weakly-supervised methods using the same or even more annotation costs. Besides, compared with SOTA fully-supervised methods, we achieve on-par or even better performance on the KITTI dataset with about 5x less annotation cost, and 90% of their performance on the Waymo dataset with about 15x less annotation cost. The additional unlabeled training scenes could further boost the performance.
>
---
#### [replaced 016] CostNav: A Navigation Benchmark for Real-World Economic-Cost Evaluation of Physical AI Agents
- **分类: cs.AI; cs.CE; cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出CostNav，一个用于评估物理AI代理经济成本的导航基准，解决传统导航任务忽略实际经济约束的问题。通过真实数据与仿真结合，揭示现有方法在商业可行性上的不足。**

- **链接: [https://arxiv.org/pdf/2511.20216v4](https://arxiv.org/pdf/2511.20216v4)**

> **作者:** Haebin Seong; Sungmin Kim; Yongjun Cho; Myunchul Joe; Geunwoo Kim; Yubeen Park; Sunhoo Kim; Yoonshik Kim; Suhwan Choi; Jaeyoon Jung; Jiyong Youn; Jinmyung Kwak; Sunghee Ahn; Jaemin Lee; Younggil Do; Seungyeop Yi; Woojin Cheong; Minhyeok Oh; Minchan Kim; Seongjae Kang; Samwoo Seong; Youngjae Yu; Yunsung Lee
>
> **摘要:** While current navigation benchmarks prioritize task success in simplified settings, they neglect the multidimensional economic constraints essential for the real-world commercialization of autonomous delivery systems. We introduce CostNav, an Economic Navigation Benchmark that evaluates physical AI agents through comprehensive economic cost-revenue analysis aligned with real-world business operations. By integrating industry-standard data - such as SEC filings and AIS injury reports - with Isaac Sim's detailed collision and cargo dynamics, CostNav transcends simple task completion to accurately evaluate business value in complex, real-world scenarios. To our knowledge, CostNav is the first work to quantitatively expose the gap between navigation research metrics and commercial viability, revealing that optimizing for task success on a simplified task fundamentally differs from optimizing for real-world economic deployment. Our evaluation of rule-based Nav2 navigation shows that current approaches are not economically viable: the contribution margin is -22.81/run (AMCL) and -12.87/run (GPS), resulting in no break-even point. We challenge the community to develop navigation policies that achieve economic viability on CostNav. We remain method-agnostic, evaluating success solely on the metric of cost rather than the underlying architecture. All resources are available at https://github.com/worv-ai/CostNav.
>
---
#### [replaced 017] Non-Contrastive Vision-Language Learning with Predictive Embedding Alignment
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.00653v2](https://arxiv.org/pdf/2602.00653v2)**

> **作者:** Lukas Kuhn; Giuseppe Serra; Florian Buettner
>
> **摘要:** Vision-language models have transformed multimodal representation learning, yet dominant contrastive approaches like CLIP require large batch sizes, careful negative sampling, and extensive hyperparameter tuning. We introduce NOVA, a NOn-contrastive Vision-language Alignment framework based on joint embedding prediction with distributional regularization. NOVA aligns visual representations to a frozen, domain-specific text encoder by predicting text embeddings from augmented image views, while enforcing an isotropic Gaussian structure via Sketched Isotropic Gaussian Regularization (SIGReg). This eliminates the need for negative sampling, momentum encoders, or stop-gradients, reducing the training objective to a single hyperparameter. We evaluate NOVA on zeroshot chest X-ray classification using ClinicalBERT as the text encoder and Vision Transformers trained from scratch on MIMIC-CXR. On zero-shot classification across three benchmark datasets, NOVA outperforms multiple standard baselines while exhibiting substantially more consistent training runs. Our results demonstrate that non-contrastive vision-language pretraining offers a simpler, more stable, and more effective alternative to contrastive methods.
>
---
#### [replaced 018] MITI: SLAM Benchmark for Laparoscopic Surgery
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2202.11496v2](https://arxiv.org/pdf/2202.11496v2)**

> **作者:** Regine Hartwig; Daniel Ostler; Jean-Claude Rosenthal; Hubertus Feußner; Dirk Wilhelm; Dirk Wollherr
>
> **备注:** This submission is withdrawn because it is a duplicate of "Constrained Visual-Inertial Localization With Application And Benchmark in Laparoscopic Surgery" (arXiv:2202.11075). The withdrawn version contains less complete information. Readers are directed to the full version
>
> **摘要:** We propose a new benchmark for evaluating stereoscopic visual-inertial computer vision algorithms (SLAM/ SfM/ 3D Reconstruction/ Visual-Inertial Odometry) for minimally invasive surgical (MIS) interventions in the abdomen. Our MITI Dataset available at [https://mediatum.ub.tum.de/1621941] provides all the necessary data by a complete recording of a handheld surgical intervention at Research Hospital Rechts der Isar of TUM. It contains multimodal sensor information from IMU, stereoscopic video, and infrared (IR) tracking as ground truth for evaluation. Furthermore, calibration for the stereoscope, accelerometer, magnetometer, the rigid transformations in the sensor setup, and time-offsets are available. We wisely chose a suitable intervention that contains very few cutting and tissue deformation and shows a full scan of the abdomen with a handheld camera such that it is ideal for testing SLAM algorithms. Intending to promote the progress of visual-inertial algorithms designed for MIS application, we hope that our clinical training dataset helps and enables researchers to enhance algorithms.
>
---
#### [replaced 019] WaymoQA: A Multi-View Visual Question Answering Dataset for Safety-Critical Reasoning in Autonomous Driving
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.20022v2](https://arxiv.org/pdf/2511.20022v2)**

> **作者:** Seungjun Yu; Seonho Lee; Namho Kim; Jaeyo Shin; Junsung Park; Wonjeong Ryu; Raehyuk Jung; Hyunjung Shim
>
> **摘要:** Recent advancements in multimodal large language models (MLLMs) have shown strong understanding of driving scenes, drawing interest in their application to autonomous driving. However, high-level reasoning in safety-critical scenarios, where avoiding one traffic risk can create another, remains a major challenge. Such reasoning is often infeasible with only a single front view and requires a comprehensive view of the environment, which we achieve through multi-view inputs. We define Safety-Critical Reasoning as a new task that leverages multi-view inputs to address this challenge. Then, we distill Safety-Critical Reasoning into two stages: first resolve the immediate risk, then mitigate the decision-induced downstream risks. To support this, we introduce WaymoQA, a dataset of 35,000 human-annotated question-answer pairs covering complex, high-risk driving scenarios. The dataset includes multiple-choice and open-ended formats across both image and video modalities. Experiments reveal that existing MLLMs underperform in safety-critical scenarios compared to normal scenes, but fine-tuning with WaymoQA significantly improves their reasoning ability, highlighting the effectiveness of our dataset in developing safer and more reasoning-capable driving agents. Our code and data are provided in https://github.com/sjyu001/WaymoQA
>
---
#### [replaced 020] MIND: Benchmarking Memory Consistency and Action Control in World Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.08025v2](https://arxiv.org/pdf/2602.08025v2)**

> **作者:** Yixuan Ye; Xuanyu Lu; Yuxin Jiang; Yuchao Gu; Rui Zhao; Qiwei Liang; Jiachun Pan; Fengda Zhang; Weijia Wu; Alex Jinpeng Wang
>
> **摘要:** World models aim to understand, remember, and predict dynamic visual environments, yet a unified benchmark for evaluating their fundamental abilities remains lacking. To address this gap, we introduce MIND, the first open-domain closed-loop revisited benchmark for evaluating Memory consIstency and action coNtrol in worlD models. MIND contains 250 high-quality videos at 1080p and 24 FPS, including 100 (first-person) + 100 (third-person) video clips under a shared action space and 25 + 25 clips across varied action spaces covering eight diverse scenes. We design an efficient evaluation framework to measure two core abilities: memory consistency and action control, capturing temporal stability and contextual coherence across viewpoints. Furthermore, we design various action spaces, including different character movement speeds and camera rotation angles, to evaluate the action generalization capability across different action spaces under shared scenes. To facilitate future performance benchmarking on MIND, we introduce MIND-World, a novel interactive Video-to-World baseline. Extensive experiments demonstrate the completeness of MIND and reveal key challenges in current world models, including the difficulty of maintaining long-term memory consistency and generalizing across action spaces. Code: https://github.com/CSU-JPG/MIND.
>
---
#### [replaced 021] Visibility-Aware Language Aggregation for Open-Vocabulary Segmentation in 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.05515v2](https://arxiv.org/pdf/2509.05515v2)**

> **作者:** Sen Wang; Kunyi Li; Siyun Liang; Elena Alegret; Jing Ma; Nassir Navab; Stefano Gasperini
>
> **备注:** Project page: https://vala3d.github.io
>
> **摘要:** Recently, distilling open-vocabulary language features from 2D images into 3D Gaussians has attracted significant attention. Although existing methods achieve impressive language-based interactions of 3D scenes, we observe two fundamental issues: background Gaussians contributing negligibly to a rendered pixel get the same feature as the dominant foreground ones, and multi-view inconsistencies due to view-specific noise in language embeddings. We introduce Visibility-Aware Language Aggregation (VALA), a lightweight yet effective method that computes marginal contributions for each ray and applies a visibility-aware gate to retain only visible Gaussians. Moreover, we propose a streaming weighted geometric median in cosine space to merge noisy multi-view features. Our method yields a robust, view-consistent language feature embedding in a fast and memory-efficient manner. VALA improves open-vocabulary localization and segmentation across reference datasets, consistently surpassing existing works. More results are available at https://vala3d.github.io
>
---
#### [replaced 022] Out of the box age estimation through facial imagery: A Comprehensive Benchmark of Vision-Language Models vs. out-of-the-box Traditional Architectures
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.07815v2](https://arxiv.org/pdf/2602.07815v2)**

> **作者:** Simiao Ren; Xingyu Shen; Ankit Raj; Albert Dai; Caroline; Zhang; Yuan Xu; Zexi Chen; Siqi Wu; Chen Gong; Yuxin Zhang
>
> **摘要:** Facial age estimation plays a critical role in content moderation, age verification, and deepfake detection. However, no prior benchmark has systematically compared modern vision-language models (VLMs) with specialized age estimation architectures. We present the first large-scale cross-paradigm benchmark, evaluating 34 models - 22 specialized architectures with publicly available pretrained weights and 12 general-purpose VLMs - across eight standard datasets (UTKFace, IMDB-WIKI, MORPH, AFAD, CACD, FG-NET, APPA-REAL, and AgeDB), totaling 1,100 test images per model. Our key finding is striking: zero-shot VLMs significantly outperform most specialized models, achieving an average mean absolute error (MAE) of 5.65 years compared to 9.88 years for non-LLM models. The best-performing VLM (Gemini 3 Flash Preview, MAE 4.32) surpasses the strongest non-LLM model (MiVOLO, MAE 5.10) by 15%. MiVOLO - unique in combining face and body features using Vision Transformers - is the only specialized model that remains competitive with VLMs. We further analyze age verification at the 18-year threshold and find that most non-LLM models exhibit false adult rates between 39% and 100% for minors, whereas VLMs reduce this to 16%-29%. Additionally, coarse age binning (8-9 classes) consistently increases MAE beyond 13 years. Stratified analysis across 14 age groups reveals that all models struggle most at extreme ages (under 5 and over 65). Overall, these findings challenge the assumption that task-specific architectures are necessary for high-performance age estimation and suggest that future work should focus on distilling VLM capabilities into efficient specialized models.
>
---
#### [replaced 023] Thermal odometry and dense mapping using learned odometry and Gaussian splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.07493v2](https://arxiv.org/pdf/2602.07493v2)**

> **作者:** Tianhao Zhou; Yujia Chen; Zhihao Zhan; Yuhang Ming; Jianzhu Huai
>
> **备注:** 11 pages, 2 figures, 5 tables
>
> **摘要:** Thermal infrared sensors, with wavelengths longer than smoke particles, can capture imagery independent of darkness, dust, and smoke. This robustness has made them increasingly valuable for motion estimation and environmental perception in robotics, particularly in adverse conditions. Existing thermal odometry and mapping approaches, however, are predominantly geometric and often fail across diverse datasets while lacking the ability to produce dense maps. Motivated by the efficiency and high-quality reconstruction ability of recent Gaussian Splatting (GS) techniques, we propose TOM-GS, a thermal odometry and mapping method that integrates learning-based odometry with GS-based dense mapping. TOM-GS is among the first GS-based SLAM systems tailored for thermal cameras, featuring dedicated thermal image enhancement and monocular depth integration. Extensive experiments on motion estimation and novel-view rendering demonstrate that TOM-GS outperforms existing learning-based methods, confirming the benefits of learning-based pipelines for robust thermal odometry and dense reconstruction.
>
---
#### [replaced 024] Towards Privacy-Guaranteed Label Unlearning in Vertical Federated Learning: Few-Shot Forgetting without Disclosure
- **分类: cs.LG; cs.CR; cs.CV**

- **链接: [https://arxiv.org/pdf/2410.10922v2](https://arxiv.org/pdf/2410.10922v2)**

> **作者:** Hanlin Gu; Hong Xi Tae; Chee Seng Chan; Lixin Fan
>
> **备注:** We introduce the first method for label unlearning in vertical federated learning (VFL), focused on preventing label leakage by the active party
>
> **摘要:** This paper addresses the critical challenge of unlearning in Vertical Federated Learning (VFL), a setting that has received far less attention than its horizontal counterpart. Specifically, we propose the first method tailored to \textit{label unlearning} in VFL, where labels play a dual role as both essential inputs and sensitive information. To this end, we employ a representation-level manifold mixup mechanism to generate synthetic embeddings for both unlearned and retained samples. This is to provide richer signals for the subsequent gradient-based label forgetting and recovery steps. These augmented embeddings are then subjected to gradient-based label forgetting, effectively removing the associated label information from the model. To recover performance on the retained data, we introduce a recovery-phase optimization step that refines the remaining embeddings. This design achieves effective label unlearning while maintaining computational efficiency. We validate our method through extensive experiments on diverse datasets, including MNIST, CIFAR-10, CIFAR-100, ModelNet, Brain Tumor MRI, COVID-19 Radiography, and Yahoo Answers demonstrate strong efficacy and scalability. Overall, this work establishes a new direction for unlearning in VFL, showing that re-imagining mixup as an efficient mechanism can unlock practical and utility-preserving unlearning. The code is publicly available at \href{https://github.com/bryanhx/Towards-Privacy-Guaranteed-Label-Unlearning-in-Vertical-Federated-Learning}{https://github.com/bryanhx/Towards-Privacy-Guaranteed-Label-Unlearning-in-Vertical-Federated-Learning}
>
---
#### [replaced 025] ZebraPose: Zebra Detection and Pose Estimation using only Synthetic Data
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于动物检测与姿态估计任务，解决真实数据收集困难及合成数据泛化性差的问题。通过生成高质量合成数据，实现无需真实数据的准确检测与姿态估计。**

- **链接: [https://arxiv.org/pdf/2408.10831v2](https://arxiv.org/pdf/2408.10831v2)**

> **作者:** Elia Bonetto; Aamir Ahmad
>
> **备注:** 17 pages, 5 tables, 13 figures. Published in WACV 2026
>
> **摘要:** Collecting and labeling large real-world wild animal datasets is impractical, costly, error-prone, and labor-intensive. For animal monitoring tasks, as detection, tracking, and pose estimation, out-of-distribution viewpoints (e.g. aerial) are also typically needed but rarely found in publicly available datasets. To solve this, existing approaches synthesize data with simplistic techniques that then necessitate strategies to bridge the synthetic-to-real gap. Therefore, real images, style constraints, complex animal models, or pre-trained networks are often leveraged. In contrast, we generate a fully synthetic dataset using a 3D photorealistic simulator and demonstrate that it can eliminate such needs for detecting and estimating 2D poses of wild zebras. Moreover, existing top-down 2D pose estimation approaches using synthetic data assume reliable detection models. However, these often fail in out-of-distribution scenarios, e.g. those that include wildlife or aerial imagery. Our method overcomes this by enabling the training of both tasks using the same synthetic dataset. Through extensive benchmarks, we show that models trained from scratch exclusively on our synthetic data generalize well to real images. We perform these using multiple real-world and synthetic datasets, pre-trained and randomly initialized backbones, and different image resolutions. Code, results, models, and data can be found athttps://zebrapose.is.tue.mpg.de/.
>
---
#### [replaced 026] SKEL-CF: Coarse-to-Fine Biomechanical Skeleton and Surface Mesh Recovery
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20157v4](https://arxiv.org/pdf/2511.20157v4)**

> **作者:** Da Li; Jiping Jin; Xuanlong Yu; Wei Liu; Xiaodong Cun; Kai Chen; Rui Fan; Jiangang Kong; Xi Shen
>
> **备注:** Project page: https://pokerman8.github.io/SKEL-CF/
>
> **摘要:** Parametric 3D human models such as SMPL have driven significant advances in human pose and shape estimation, yet their simplified kinematics limit biomechanical realism. The recently proposed SKEL model addresses this limitation by re-rigging SMPL with an anatomically accurate skeleton. However, estimating SKEL parameters directly remains challenging due to limited training data, perspective ambiguities, and the inherent complexity of human articulation. We introduce SKEL-CF, a coarse-to-fine framework for SKEL parameter estimation. SKEL-CF employs a transformer-based encoder-decoder architecture, where the encoder predicts coarse camera and SKEL parameters, and the decoder progressively refines them in successive layers. To ensure anatomically consistent supervision, we convert the existing SMPL-based dataset 4DHuman into a SKEL-aligned version, 4DHuman-SKEL, providing high-quality training data for SKEL estimation. In addition, to mitigate depth and scale ambiguities, we explicitly incorporate camera modeling into the SKEL-CF pipeline and demonstrate its importance across diverse viewpoints. Extensive experiments validate the effectiveness of the proposed design. On the challenging MOYO dataset, SKEL-CF achieves 85.0 MPJPE / 51.4 PA-MPJPE, significantly outperforming the previous SKEL-based state-of-the-art HSMR (104.5 / 79.6). These results establish SKEL-CF as a scalable and anatomically faithful framework for human motion analysis, facilitating the use of computer vision techniques in biomechanics-related analysis. Our implementation is available on the project page: https://pokerman8.github.io/SKEL-CF/.
>
---
#### [replaced 027] A UAV-Based VNIR Hyperspectral Benchmark Dataset for Landmine and UXO Detection
- **分类: eess.IV; cs.CV; eess.SP**

- **链接: [https://arxiv.org/pdf/2510.02700v2](https://arxiv.org/pdf/2510.02700v2)**

> **作者:** Sagar Lekhak; Emmett J. Ientilucci; Jasper Baur; Susmita Ghosh
>
> **备注:** This work was accepted and presented as an oral paper at the Indian Geoscience and Remote Sensing Symposium (InGARSS) 2025 and appears in the IEEE InGARSS 2025 Proceedings
>
> **摘要:** This paper introduces a novel benchmark dataset of Visible and Near-Infrared (VNIR) hyperspectral imagery acquired via an unmanned aerial vehicle (UAV) platform for landmine and unexploded ordnance (UXO) detection research. The dataset was collected over a controlled test field seeded with 143 realistic surrogate landmine and UXO targets, including surface, partially buried, and fully buried configurations. Data acquisition was performed using a Headwall Nano-Hyperspec sensor mounted on a multi-sensor drone platform, flown at an altitude of approximately 20.6 m, capturing 270 contiguous spectral bands spanning 398-1002 nm. Radiometric calibration, orthorectification, and mosaicking were performed followed by reflectance retrieval using a two-point Empirical Line Method (ELM), with reference spectra acquired using an SVC spectroradiometer. Cross-validation against six reference objects yielded RMSE values below 1.0 and SAM values between 1 and 6 degrees in the 400-900 nm range, demonstrating high spectral fidelity. The dataset is released alongside raw radiance cubes, GCP/AeroPoint data, and reference spectra to support reproducible research. This contribution fills a critical gap in open-access UAV-based hyperspectral data for landmine detection and offers a multi-sensor benchmark when combined with previously published drone-based electromagnetic induction (EMI) data from the same test field.
>
---
#### [replaced 028] GenDR: Lighten Generative Detail Restoration
- **分类: cs.CV; cs.AI; eess.IV**

- **链接: [https://arxiv.org/pdf/2503.06790v3](https://arxiv.org/pdf/2503.06790v3)**

> **作者:** Yan Wang; Shijie Zhao; Kexin Zhang; Junlin Li; Li Zhang
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Although recent research applying text-to-image (T2I) diffusion models to real-world super-resolution (SR) has achieved remarkable progress, the misalignment of their targets leads to a suboptimal trade-off between inference speed and detail fidelity. Specifically, the T2I task requires multiple inference steps to synthesize images matching to prompts and reduces the latent dimension to lower generating difficulty. Contrariwise, SR can restore high-frequency details in fewer inference steps, but it necessitates a more reliable variational auto-encoder (VAE) to preserve input information. However, most diffusion-based SRs are multistep and use 4-channel VAEs, while existing models with 16-channel VAEs are overqualified diffusion transformers, e.g., FLUX (12B). To align the target, we present a one-step diffusion model for generative detail restoration, GenDR, distilled from a tailored diffusion model with a larger latent space. In detail, we train a new SD2.1-VAE16 (0.9B) via representation alignment to expand the latent space without increasing the model size. Regarding step distillation, we propose consistent score identity distillation (CiD) that incorporates SR task-specific loss into score distillation to leverage more SR priors and align the training target. Furthermore, we extend CiD with adversarial learning and representation alignment (CiDA) to enhance perceptual quality and accelerate training. We also polish the pipeline to achieve a more efficient inference. Experimental results demonstrate that GenDR achieves state-of-the-art performance in both quantitative metrics and visual fidelity.
>
---
#### [replaced 029] Shortest-Path Flow Matching with Mixture-Conditioned Bases for OOD Generalization to Unseen Conditions
- **分类: cs.LG; cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2601.11827v2](https://arxiv.org/pdf/2601.11827v2)**

> **作者:** Andrea Rubbi; Amir Akbarnejad; Mohammad Vali Sanian; Aryan Yazdan Parast; Hesam Asadollahzadeh; Arian Amani; Naveed Akhtar; Sarah Cooper; Andrew Bassett; Pietro Liò; Lassi Paavolainen; Sattar Vakili; Mo Lotfollahi
>
> **摘要:** Robust generalization under distribution shift remains a key challenge for conditional generative modeling: conditional flow-based methods often fit the training conditions well but fail to extrapolate to unseen ones. We introduce SP-FM, a shortest-path flow-matching framework that improves out-of-distribution (OOD) generalization by conditioning both the base distribution and the flow field on the condition. Specifically, SP-FM learns a condition-dependent base distribution parameterized as a flexible, learnable mixture, together with a condition-dependent vector field trained via shortest-path flow matching. Conditioning the base allows the model to adapt its starting distribution across conditions, enabling smooth interpolation and more reliable extrapolation beyond the observed training range. We provide theoretical insights into the resulting conditional transport and show how mixture-conditioned bases enhance robustness under shift. Empirically, SP-FM is effective across heterogeneous domains, including predicting responses to unseen perturbations in single-cell transcriptomics and modeling treatment effects in high-content microscopy--based drug screening. Overall, SP-FM provides a simple yet effective plug-in strategy for improving conditional generative modeling and OOD generalization across diverse domains.
>
---
#### [replaced 030] Order from Chaos: Physical World Understanding from Glitchy Gameplay Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.16471v2](https://arxiv.org/pdf/2601.16471v2)**

> **作者:** Meng Cao; Haoran Tang; Haoze Zhao; Mingfei Han; Ruyang Liu; Qiang Sun; Xiaojun Chang; Ian Reid; Xiaodan Liang
>
> **备注:** Accepted by TMLR
>
> **摘要:** Understanding the physical world, including object dynamics, material properties, and causal interactions, remains a core challenge in artificial intelligence. Although recent multi-modal large language models (MLLMs) have demonstrated impressive general reasoning capabilities, they still fall short of achieving human-level understanding of physical principles. Existing datasets for physical reasoning either rely on real-world videos, which incur high annotation costs, or on synthetic simulations, which suffer from limited realism and diversity. In this paper, we propose a novel paradigm that leverages glitches in gameplay videos, referring to visual anomalies that violate predefined physical laws, as a rich and scalable supervision source for physical world understanding. We introduce PhysGame, an meta information guided instruction-tuning dataset containing 140,057 glitch-centric question-answer pairs across five physical domains and sixteen fine-grained categories. To ensure data accuracy, we design a prompting strategy that utilizes gameplay metadata such as titles and descriptions to guide high-quality QA generation. Complementing PhysGame, we construct GameBench, an expert-annotated benchmark with 880 glitch-identified gameplay videos designed to evaluate physical reasoning capabilities. Extensive experiments show that PhysGame significantly enhances both Game2Real transferability, improving the real world physical reasoning performance of Qwen2.5VL by 2.5% on PhysBench, and Game2General transferability, yielding a 1.9% gain on the MVBench benchmark. Moreover, PhysGame-tuned models achieve a 3.7% absolute improvement on GameBench, demonstrating enhanced robustness in detecting physical implausibilities. These results indicate that learning from gameplay anomalies offers a scalable and effective pathway toward advancing physical world understanding in multimodal intelligence.
>
---
#### [replaced 031] Corruption-Aware Training of Latent Video Diffusion Models for Robust Text-to-Video Generation
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2505.21545v2](https://arxiv.org/pdf/2505.21545v2)**

> **作者:** Chika Maduabuchi; Hao Chen; Yujin Han; Jindong Wang
>
> **备注:** Code: https://github.com/chikap421/catlvdm
>
> **摘要:** Latent Video Diffusion Models (LVDMs) have achieved state-of-the-art generative quality for image and video generation; however, they remain brittle under noisy conditioning, where small perturbations in text or multimodal embeddings can cascade over timesteps and cause semantic drift. Existing corruption strategies from image diffusion (e.g., Gaussian, Uniform) fail in video settings because static noise disrupts temporal fidelity. In this paper, we propose CAT-LVDM, a corruption-aware training framework with structured, data-aligned noise injection tailored for video diffusion. Our two operators, Batch-Centered Noise Injection (BCNI) and Spectrum-Aware Contextual Noise (SACN), align perturbations with batch semantics or spectral dynamics to preserve coherence. CAT-LVDM yields substantial gains: BCNI reduces FVD by 31.9 percent on WebVid-2M, MSR-VTT, and MSVD, while SACN improves UCF-101 by 12.3 percent, outperforming Gaussian, Uniform, and large diffusion baselines such as DEMO (2.3B) and LaVie (3B) despite training on 5x less data. Ablations confirm the unique value of low-rank, data-aligned noise, and theoretical analysis establishes why these operators tighten robustness and generalization bounds. CAT-LVDM thus introduces a principled framework for robust video diffusion and further demonstrates transferability to autoregressive generation and multimodal video understanding models.
>
---
#### [replaced 032] Geospatial Representation Learning: A Survey from Deep Learning to The LLM Era
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2505.09651v2](https://arxiv.org/pdf/2505.09651v2)**

> **作者:** Xixuan Hao; Yutian Jiang; Xingchen Zou; Jiabo Liu; Yifang Yin; Song Gao; Flora Salim; Tianrui Li; Yuxuan Liang
>
> **摘要:** The ability to transform location-centric geospatial data into meaningful computational representations has become fundamental to modern spatial analysis and decision-making. Geospatial Representation Learning (GRL), the process of automatically extracting latent structures and semantic patterns from geographic data, is undergoing a profound transformation through two successive technological revolutions: the deep learning breakthrough and the emerging large language model (LLM) paradigm. While deep neural networks (DNNs) have demonstrated remarkable success in automated feature extraction from structured and semi-structured geospatial data (e.g., satellite imagery, GPS trajectories), the recent integration of LLMs introduces transformative capabilities for cross-modal geospatial reasoning and unstructured geo-textual data processing. This survey presents a comprehensive review of geospatial representation learning across both technological eras, organizing them into a structured taxonomy based on the complete pipeline comprising: (1) data perspective, (2) methodological perspective, and (3) application perspective. We also highlight current advancements, discuss existing limitations, and propose potential future research directions in the LLM and foundation model era. This work offers a thorough exploration of the field and provides a roadmap for further innovation in GRL. The summary of the up-to-date paper list can be found in https://github.com/CityMind-Lab/Awesome-Geospatial-Representation-Learning and will undergo continuous updates.
>
---
#### [replaced 033] FD-DB: Frequency-Decoupled Dual-Branch Network for Unpaired Synthetic-to-Real Domain Translation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.09476v2](https://arxiv.org/pdf/2602.09476v2)**

> **作者:** Chuanhai Zang; Jiabao Hu; XW Song
>
> **备注:** 26 pages, 13 figures, 2 tables. Code available at https://github.com/tryzang/FD-DB
>
> **摘要:** Synthetic data provide low-cost, accurately annotated samples for geometry-sensitive vision tasks, but appearance and imaging differences between synthetic and real domains cause severe domain shift and degrade downstream performance. Unpaired synthetic-to-real translation can reduce this gap without paired supervision, yet existing methods often face a trade-off between photorealism and structural stability: unconstrained generation may introduce deformation or spurious textures, while overly rigid constraints limit adaptation to real-domain statistics. We propose FD-DB, a frequency-decoupled dual-branch model that separates appearance transfer into low-frequency interpretable editing and high-frequency residual compensation. The interpretable branch predicts physically meaningful editing parameters (white balance, exposure, contrast, saturation, blur, and grain) to build a stable low-frequency appearance base with strong content preservation. The free branch complements fine details through residual generation, and a gated fusion mechanism combines the two branches under explicit frequency constraints to limit low-frequency drift. We further adopt a two-stage training schedule that first stabilizes the editing branch and then releases the residual branch to improve optimization stability. Experiments on the YCB-V dataset show that FD-DB improves real-domain appearance consistency and significantly boosts downstream semantic segmentation performance while preserving geometric and semantic structures.
>
---
#### [replaced 034] TwistNet-2D: Learning Second-Order Channel Interactions via Spiral Twisting for Texture Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.07262v2](https://arxiv.org/pdf/2602.07262v2)**

> **作者:** Junbo Jacob Lian; Feng Xiong; Yujun Sun; Kaichen Ouyang; Zong Ke; Mingyang Yu; Shengwei Fu; Zhong Rui; Zhang Yujun; Huiling Chen
>
> **备注:** Code is available at https://github.com/junbolian/TwistNet-2D
>
> **摘要:** Second-order feature statistics are central to texture recognition, yet current methods face a fundamental tension: bilinear pooling and Gram matrices capture global channel correlations but collapse spatial structure, while self-attention models spatial context through weighted aggregation rather than explicit pairwise feature interactions. We introduce TwistNet-2D, a lightweight module that computes \emph{local} pairwise channel products under directional spatial displacement, jointly encoding where features co-occur and how they interact. The core component, Spiral-Twisted Channel Interaction (STCI), shifts one feature map along a prescribed direction before element-wise channel multiplication, thereby capturing the cross-position co-occurrence patterns characteristic of structured and periodic textures. Aggregating four directional heads with learned channel reweighting and injecting the result through a sigmoid-gated residual path, \TwistNet incurs only 3.5% additional parameters and 2% additional FLOPs over ResNet-18, yet consistently surpasses both parameter-matched and substantially larger baselines -- including ConvNeXt, Swin Transformer, and hybrid CNN--Transformer architectures -- across four texture and fine-grained recognition benchmarks.
>
---
#### [replaced 035] Catching the Details: Self-Distilled RoI Predictors for Fine-Grained MLLM Perception
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.16944v3](https://arxiv.org/pdf/2509.16944v3)**

> **作者:** Yuheng Shi; Xiaohuan Pei; Minjing Dong; Chang Xu
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Multimodal Large Language Models (MLLMs) require high-resolution visual information to perform fine-grained perception, yet processing entire high-resolution images is computationally prohibitive. While recent methods leverage a Region-of-Interest (RoI) mechanism to focus on salient areas, they typically present a difficult trade-off: training-based approaches depend on large-scale annotated datasets, while training-free methods that utilize the model's internal attention are computationally inefficient and less accurate, requiring either multi-pass prefill stages or reliance on the slow auto-regressive decoding process. In this paper, we propose an efficient, annotation-free Self-Distilled Region Proposal Network (SD-RPN) that resolves this trade-off. The SD-RPN is built around a pipeline that transforms the noisy attention maps from the MLLM's middle layers into high-quality pseudo-RoI labels by explicitly denoising the signal and resolving ambiguity. We use these labels to train a lightweight Region Proposal Network (RPN) that learns a more precise localization. This RPN is also highly efficient, predicting the RoI in a single forward pass using features from the MLLM's middle layers, decoupling RoI identification from the auto-regressive generation and avoiding costly multi-pass operations. To validate our approach, we integrate the framework into multiple MLLM families. Despite being trained on only a few (e.g. 10K) question-answer pairs, our method demonstrates exceptional data efficiency and generalization, achieving over a 10% absolute accuracy improvement on unseen benchmarks, including TextVQA, DocVQA, and V-Star. Our work presents a practical and scalable solution for enhancing the fine-grained perception of MLLMs without requiring costly supervision or full model fine-tuning. Code is available at https://github.com/YuHengsss/SD-RPN.
>
---
#### [replaced 036] LighthouseGS: Indoor Structure-aware 3D Gaussian Splatting for Panorama-Style Mobile Captures
- **分类: cs.GR; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.06109v2](https://arxiv.org/pdf/2507.06109v2)**

> **作者:** Seungoh Han; Jaehoon Jang; Hyunsu Kim; Jaeheung Surh; Junhyung Kwak; Hyowon Ha; Kyungdon Joo
>
> **备注:** WACV 2026
>
> **摘要:** We introduce LighthouseGS, a practical novel view synthesis framework based on 3D Gaussian Splatting that utilizes simple panorama-style captures from a single mobile device. While convenient, this rotation-dominant motion and narrow baseline make accurate camera pose and 3D point estimation challenging, especially in textureless indoor scenes. To address these challenges, LighthouseGS leverages rough geometric priors, such as mobile device camera poses and monocular depth estimation, and utilizes indoor planar structures. Specifically, we propose a new initialization method called plane scaffold assembly to generate consistent 3D points on these structures, followed by a stable pruning strategy to enhance geometry and optimization stability. Additionally, we present geometric and photometric corrections to resolve inconsistencies from motion drift and auto-exposure in mobile devices. Tested on real and synthetic indoor scenes, LighthouseGS delivers photorealistic rendering, outperforming state-of-the-art methods and enabling applications like panoramic view synthesis and object placement. Project page: https://vision3d-lab.github.io/lighthousegs/
>
---
#### [replaced 037] H2OFlow: Grounding Human-Object Affordances with 3D Generative Models and Dense Diffused Flows
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.21769v2](https://arxiv.org/pdf/2510.21769v2)**

> **作者:** Harry Zhang; Luca Carlone
>
> **摘要:** Understanding how humans interact with the surrounding environment, and specifically reasoning about object interactions and affordances, is a critical challenge in computer vision, robotics, and AI. Current approaches often depend on labor-intensive, hand-labeled datasets capturing real-world or simulated human-object interaction (HOI) tasks, which are costly and time-consuming to produce. Furthermore, most existing methods for 3D affordance understanding are limited to contact-based analysis, neglecting other essential aspects of human-object interactions, such as orientation (\eg, humans might have a preferential orientation with respect certain objects, such as a TV) and spatial occupancy (\eg, humans are more likely to occupy certain regions around an object, like the front of a microwave rather than its back). To address these limitations, we introduce \emph{H2OFlow}, a novel framework that comprehensively learns 3D HOI affordances -- encompassing contact, orientation, and spatial occupancy -- using only synthetic data generated from 3D generative models. H2OFlow employs a dense 3D-flow-based representation, learned through a dense diffusion process operating on point clouds. This learned flow enables the discovery of rich 3D affordances without the need for human annotations. Through extensive quantitative and qualitative evaluations, we demonstrate that H2OFlow generalizes effectively to real-world objects and surpasses prior methods that rely on manual annotations or mesh-based representations in modeling 3D affordance.
>
---
#### [replaced 038] RepAir: A Framework for Airway Segmentation and Discontinuity Correction in CT
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14649v2](https://arxiv.org/pdf/2511.14649v2)**

> **作者:** John M. Oyer; Ali Namvar; Benjamin A. Hoff; Wassim W. Labaki; Ella A. Kazerooni; Charles R. Hatt; Fernando J. Martinez; MeiLan K. Han; Craig J. Galbán; Sundaresh Ram
>
> **备注:** 4 pages, 3 figures, 1 table. Oral presentation accepted to SSIAI 2026 Conference on Jan 20, 2026
>
> **摘要:** Accurate airway segmentation from chest computed tomography (CT) scans is essential for quantitative lung analysis, yet manual annotation is impractical and many automated U-Net-based methods yield disconnected components that hinder reliable biomarker extraction. We present RepAir, a three-stage framework for robust 3D airway segmentation that combines an nnU-Net-based network with anatomically informed topology correction. The segmentation network produces an initial airway mask, after which a skeleton-based algorithm identifies potential discontinuities and proposes reconnections. A 1D convolutional classifier then determines which candidate links correspond to true anatomical branches versus false or obstructed paths. We evaluate RepAir on two distinct datasets: ATM'22, comprising annotated CT scans from predominantly healthy subjects and AeroPath, encompassing annotated scans with severe airway pathology. Across both datasets, RepAir outperforms existing 3D U-Net-based approaches such as Bronchinet and NaviAirway on both voxel-level and topological metrics, and produces more complete and anatomically consistent airway trees while maintaining high segmentation accuracy.
>
---
#### [replaced 039] City Navigation in the Wild: Exploring Emergent Navigation from Web-Scale Knowledge in MLLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.15933v3](https://arxiv.org/pdf/2512.15933v3)**

> **作者:** Dwip Dalal; Utkarsh Mishra; Narendra Ahuja; Nebojsa Jojic
>
> **备注:** Accepted at EACL 2026 (ORAL)
>
> **摘要:** Leveraging multimodal large language models (MLLMs) to develop embodied agents offers significant promise for addressing complex real-world tasks. However, current evaluation benchmarks remain predominantly language-centric or heavily reliant on simulated environments, rarely probing the nuanced, knowledge-intensive reasoning essential for practical, real-world scenarios. To bridge this critical gap, we introduce the task of Sparsely Grounded Visual Navigation, explicitly designed to evaluate the sequential decision-making abilities of MLLMs in challenging, knowledge-intensive real-world environment. We operationalize this task with CityNav, a comprehensive benchmark encompassing four diverse global cities, specifically constructed to assess raw MLLM-driven agents in city navigation. Agents are required to rely solely on visual inputs and internal multimodal reasoning to sequentially navigate 50+ decision points without additional environmental annotations or specialized architectural modifications. Crucially, agents must autonomously achieve localization through interpreting city-specific cues and recognizing landmarks, perform spatial reasoning, and strategically plan and execute routes to their destinations. Through extensive evaluations, we demonstrate that current state-of-the-art MLLMs, reasoning techniques (e.g., GEPA, chain-of-thought, reflection) and competitive baseline PReP significantly underperform in this challenging setting. To address this, we propose Verbalization of Path(VoP), which explicitly grounds the agent's internal reasoning by probing city-scale cognitive maps (key landmarks and directions toward the destination) from the MLLM, substantially enhancing navigation success. Project Webpage: https://dwipddalal.github.io/AgentNav/
>
---
#### [replaced 040] GeoPurify: A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.02186v2](https://arxiv.org/pdf/2510.02186v2)**

> **作者:** Weijia Dou; Xu Zhang; Yi Bin; Jian Liu; Bo Peng; Guoqing Wang; Yang Yang; Heng Tao Shen
>
> **备注:** Accepted at ICLR 2026. Code available at: https://github.com/tj12323/GeoPurify
>
> **摘要:** Recent attempts to transfer features from 2D Vision-Language Models (VLMs) to 3D semantic segmentation expose a persistent trade-off. Directly projecting 2D features into 3D yields noisy and fragmented predictions, whereas enforcing geometric coherence necessitates costly training pipelines and large-scale annotated 3D data. We argue that this limitation stems from the dominant segmentation-and-matching paradigm, which fails to reconcile 2D semantics with 3D geometric structure. The geometric cues are not eliminated during the 2D-to-3D transfer but remain latent within the noisy and view-aggregated features. To exploit this property, we propose GeoPurify that applies a small Student Affinity Network to purify 2D VLM-generated 3D point features using geometric priors distilled from a 3D self-supervised teacher model. During inference, we devise a Geometry-Guided Pooling module to further denoise the point cloud and ensure the semantic and structural consistency. Benefiting from latent geometric information and the learned affinity network, GeoPurify effectively mitigates the trade-off and achieves superior data efficiency. Extensive experiments on major 3D benchmarks demonstrate that GeoPurify achieves or surpasses state-of-the-art performance while utilizing only about 1.5% of the training data.
>
---
#### [replaced 041] Defect-aware Hybrid Prompt Optimization via Progressive Tuning for Zero-Shot Multi-type Anomaly Detection and Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.09446v2](https://arxiv.org/pdf/2512.09446v2)**

> **作者:** Nadeem Nazer; Hongkuan Zhou; Lavdim Halilaj; Ylli Sadikaj; Steffen Staab
>
> **摘要:** Recent vision language models (VLMs) like CLIP have demonstrated impressive anomaly detection performance under significant distribution shift by utilizing high-level semantic information through text prompts. However, these models often neglect fine-grained details, such as which kind of anomalies, like "hole", "cut", "scratch" that could provide more specific insight into the nature of anomalies. We argue that recognizing fine-grained anomaly types 1) enriches the representation of "abnormal" with structured semantics, narrowing the gap between coarse anomaly signals and fine-grained defect categories; 2) enables manufacturers to understand the root causes of the anomaly and implement more targeted and appropriate corrective measures quickly. While incorporating such detailed semantic information is crucial, designing handcrafted prompts for each defect type is both time-consuming and susceptible to human bias. For this reason, we introduce DAPO, a novel approach for Defect-aware Prompt Optimization based on progressive tuning for the zero-shot multi-type and binary anomaly detection and segmentation under distribution shifts. Our approach aligns anomaly-relevant image features with their corresponding text semantics by learning hybrid defect-aware prompts with both fixed textual anchors and learnable token embeddings. We conducted experiments on public benchmarks (MPDD, VisA, MVTec-AD, MAD, and Real-IAD) and an internal dataset. The results suggest that compared to the baseline models, DAPO achieves a 3.7% average improvement in AUROC and average precision metrics at the image level under distribution shift, and a 6.5% average improvement in localizing novel anomaly types under zero-shot settings.
>
---
#### [replaced 042] Enhancing Vehicle Detection under Adverse Weather Conditions with Contrastive Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.21916v2](https://arxiv.org/pdf/2509.21916v2)**

> **作者:** Boying Li; Chang Liu; Petter Kyösti; Mattias Öhman; Devashish Singha Roy; Sofia Plazzi; Hamam Mokayed; Olle Hagner
>
> **摘要:** Aside from common challenges in remote sensing like small, sparse targets and computation cost limitations, detecting vehicles from UAV images in the Nordic regions faces strong visibility challenges and domain shifts caused by diverse levels of snow coverage. Although annotated data are expensive, unannotated data is cheaper to obtain by simply flying the drones. In this work, we proposed a sideload-CL-adaptation framework that enables the use of unannotated data to improve vehicle detection using lightweight models. Specifically, we propose to train a CNN-based representation extractor through contrastive learning on the unannotated data in the pretraining stage, and then sideload it to a frozen YOLO11n backbone in the fine-tuning stage. To find a robust sideload-CL-adaptation, we conducted extensive experiments to compare various fusion methods and granularity. Our proposed sideload-CL-adaptation model improves the detection performance by 3.8% to 9.5% in terms of mAP50 on the NVD dataset.
>
---
#### [replaced 043] FANVID: A Benchmark for Face and License Plate Recognition in Low-Resolution Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.07304v2](https://arxiv.org/pdf/2506.07304v2)**

> **作者:** Kavitha Viswanathan; Vrinda Goel; Shlesh Gholap; Devayan Ghosh; Madhav Gupta; Dhruvi Ganatra; Sanket Potdar; Amit Sethi
>
> **摘要:** Real-world surveillance often renders faces and license plates unrecognizable in individual low-resolution (LR) frames, hindering reliable identification. To advance temporal recognition models, we present FANVID, a novel video-based benchmark comprising nearly 1,463 LR clips (180 x 320, 20--60 FPS) featuring 63 identities and 49 license plates from three English-speaking countries. Each video includes distractor faces and plates, increasing task difficulty and realism. The dataset contains 31,096 manually verified bounding boxes and labels. FANVID defines two tasks: (1) face matching -- detecting LR faces and matching them to high-resolution mugshots, and (2) license plate recognition -- extracting text from LR plates without a predefined database. Videos are downsampled from high-resolution sources to ensure that faces and text are indecipherable in single frames, requiring models to exploit temporal information. We introduce evaluation metrics adapted from mean Average Precision at IoU > 0.5, prioritizing identity correctness for faces and character-level accuracy for text. A baseline method with pre-trained video super-resolution, detection, and recognition achieved performance scores of 0.58 (face matching) and 0.42 (plate recognition), highlighting both the feasibility and challenge of the tasks. FANVID's selection of faces and plates balances diversity with recognition challenge. We release the software for data access, evaluation, baseline, and annotation to support reproducibility and extension. FANVID aims to catalyze innovation in temporal modeling for LR recognition, with applications in surveillance, forensics, and autonomous vehicles.
>
---
#### [replaced 044] SoulX-FlashHead: Oracle-guided Generation of Infinite Real-time Streaming Talking Heads
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.07449v3](https://arxiv.org/pdf/2602.07449v3)**

> **作者:** Tan Yu; Qian Qiao; Le Shen; Ke Zhou; Jincheng Hu; Dian Sheng; Bo Hu; Haoming Qin; Jun Gao; Changhai Zhou; Shunshun Yin; Siyuan Liu
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Achieving a balance between high-fidelity visual quality and low-latency streaming remains a formidable challenge in audio-driven portrait generation. Existing large-scale models often suffer from prohibitive computational costs, while lightweight alternatives typically compromise on holistic facial representations and temporal stability. In this paper, we propose SoulX-FlashHead, a unified 1.3B-parameter framework designed for real-time, infinite-length, and high-fidelity streaming video generation. To address the instability of audio features in streaming scenarios, we introduce Streaming-Aware Spatiotemporal Pre-training equipped with a Temporal Audio Context Cache mechanism, which ensures robust feature extraction from short audio fragments. Furthermore, to mitigate the error accumulation and identity drift inherent in long-sequence autoregressive generation, we propose Oracle-Guided Bidirectional Distillation, leveraging ground-truth motion priors to provide precise physical guidance. We also present VividHead, a large-scale, high-quality dataset containing 782 hours of strictly aligned footage to support robust training. Extensive experiments demonstrate that SoulX-FlashHead achieves state-of-the-art performance on HDTF and VFHQ benchmarks. Notably, our Lite variant achieves an inference speed of 96 FPS on a single NVIDIA RTX 4090, facilitating ultra-fast interaction without sacrificing visual coherence.
>
---
#### [replaced 045] GMG: A Video Prediction Method Based on Global Focus and Motion Guided
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.11297v3](https://arxiv.org/pdf/2503.11297v3)**

> **作者:** Yuhao Du; Hui Liu; Haoxiang Peng; Xinyuan Cheng; Chengrong Wu; Jiankai Zhang
>
> **摘要:** Recent years, weather forecasting has gained significant attention. However, accurately predicting weather remains a challenge due to the rapid variability of meteorological data and potential teleconnections. Current spatiotemporal forecasting models primarily rely on convolution operations or sliding windows for feature extraction. These methods are limited by the size of the convolutional kernel or sliding window, making it difficult to capture and identify potential teleconnection features in meteorological data. Additionally, weather data often involve non-rigid bodies, whose motion processes are accompanied by unpredictable deformations, further complicating the forecasting task. In this paper, we propose the GMG model to address these two core challenges. The Global Focus Module, a key component of our model, enhances the global receptive field, while the Motion Guided Module adapts to the growth or dissipation processes of non-rigid bodies. Through extensive evaluations, our method demonstrates competitive performance across various complex tasks, providing a novel approach to improving the predictive accuracy of complex spatiotemporal data.
>
---
#### [replaced 046] Neural-Augmented Kelvinlet for Real-Time Soft Tissue Deformation Modeling
- **分类: cs.GR; cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于手术仿真任务，旨在解决软组织实时变形建模问题。通过结合物理先验与神经网络，提升预测精度与物理合理性。**

- **链接: [https://arxiv.org/pdf/2506.08043v4](https://arxiv.org/pdf/2506.08043v4)**

> **作者:** Ashkan Shahbazi; Kyvia Pereira; Jon S. Heiselman; Elaheh Akbari; Annie C. Benson; Sepehr Seifi; Xinyuan Liu; Garrison L. Johnston; Jie Ying Wu; Nabil Simaan; Michael I. Miga; Soheil Kolouri
>
> **摘要:** Accurate and efficient modeling of soft-tissue interactions is fundamental for advancing surgical simulation, surgical robotics, and model-based surgical automation. To achieve real-time latency, classical Finite Element Method (FEM) solvers are often replaced with neural approximations; however, naively training such models in a fully data-driven manner without incorporating physical priors frequently leads to poor generalization and physically implausible predictions. We present a novel physics-informed neural simulation framework that enables real-time prediction of soft-tissue deformations under complex single- and multi-grasper interactions. Our approach integrates Kelvinlet-based analytical priors with large-scale FEM data, capturing both linear and nonlinear tissue responses. This hybrid design improves predictive accuracy and physical plausibility across diverse neural architectures while maintaining the low-latency performance required for interactive applications. We validate our method on challenging surgical manipulation tasks involving standard laparoscopic grasping tools, demonstrating substantial improvements in deformation fidelity and temporal stability over existing baselines. These results establish Kelvinlet-augmented learning as a principled and computationally efficient paradigm for real-time, physics-aware soft-tissue simulation in surgical AI.
>
---
#### [replaced 047] Spectrum from Defocus: Fast Spectral Imaging with Chromatic Focal Stack
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2503.20184v2](https://arxiv.org/pdf/2503.20184v2)**

> **作者:** M. Kerem Aydin; Yi-Chun Hung; Jaclyn Pytlarz; Qi Guo; Emma Alexander
>
> **摘要:** Hyperspectral cameras face harsh trade-offs between spatial, spectral, and temporal resolution in inherently low-photon conditions. Computational imaging systems break through these trade-offs with compressive sensing, but have required complex optics and/or extensive compute. We present Spectrum from Defocus (SfD), a chromatic focal sweep method that achieves state-of-the-art hyperspectral imaging with only two off-the-shelf lenses, a grayscale sensor, and less than one second of reconstruction time. By capturing a chromatically-aberrated focal stack that preserves nearly all incident light, and reconstructing it with a fast physics-based iterative algorithm, SfD delivers sharp, accurate hyperspectral images. The combination of photon efficiency, optical simplicity, and physical interpretability makes SfD a promising solution for fast, compact, interpretable hyperspectral imaging.
>
---
#### [replaced 048] Monocular Normal Estimation via Shading Sequence Estimation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.09929v2](https://arxiv.org/pdf/2602.09929v2)**

> **作者:** Zongrui Li; Xinhua Ma; Minghui Hu; Yunqing Zhao; Yingchen Yu; Qian Zheng; Chang Liu; Xudong Jiang; Song Bai
>
> **备注:** Accepted by ICLR 2026 (Oral Presentation)
>
> **摘要:** Monocular normal estimation aims to estimate the normal map from a single RGB image of an object under arbitrary lights. Existing methods rely on deep models to directly predict normal maps. However, they often suffer from 3D misalignment: while the estimated normal maps may appear to have a correct appearance, the reconstructed surfaces often fail to align with the geometric details. We argue that this misalignment stems from the current paradigm: the model struggles to distinguish and reconstruct varying geometry represented in normal maps, as the differences in underlying geometry are reflected only through relatively subtle color variations. To address this issue, we propose a new paradigm that reformulates normal estimation as shading sequence estimation, where shading sequences are more sensitive to various geometric information. Building on this paradigm, we present RoSE, a method that leverages image-to-video generative models to predict shading sequences. The predicted shading sequences are then converted into normal maps by solving a simple ordinary least-squares problem. To enhance robustness and better handle complex objects, RoSE is trained on a synthetic dataset, MultiShade, with diverse shapes, materials, and light conditions. Experiments demonstrate that RoSE achieves state-of-the-art performance on real-world benchmark datasets for object-based monocular normal estimation.
>
---
#### [replaced 049] Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.09741v2](https://arxiv.org/pdf/2510.09741v2)**

> **作者:** Dwip Dalal; Gautam Vashishtha; Utkarsh Mishra; Jeonghwan Kim; Madhav Kanda; Hyeonjeong Ha; Svetlana Lazebnik; Heng Ji; Unnat Jain
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Multimodal large language models (MLLMs) often miss small details and spatial relations in cluttered scenes, leading to errors in fine-grained perceptual grounding. We introduce AttWarp, a lightweight method that allocates more resolution to query-relevant content while compressing less informative areas, all while preserving global context. At test time, the approach uses an MLLM's cross-modal attention to perform rectilinear warping of the input image, reallocating spatial resolution toward regions the model deems important, without changing model weights or architecture. This attention-guided warping preserves all original image information but redistributes it non-uniformly, so small objects and subtle relationships become easier for the same model to read while the global layout remains intact. Across five benchmarks (TextVQA, GQA, DocVQA, POPE, MMMU) and four MLLMs (LLaVA, Qwen-VL, InternVL, and InstructBLIP), AttWarp consistently improves accuracy, strengthens compositional reasoning, and reduces hallucinations, outperforming four competitive baselines that manipulate raw images at test time. Together, these results show that attention-guided warping prioritizes information relevant to the query while preserving context, and that the same MLLMs perform better when given such warped inputs.
>
---
#### [replaced 050] SnapGen++: Unleashing Diffusion Transformers for Efficient High-Fidelity Image Generation on Edge Devices
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.08303v2](https://arxiv.org/pdf/2601.08303v2)**

> **作者:** Dongting Hu; Aarush Gupta; Magzhan Gabidolla; Arpit Sahni; Huseyin Coskun; Yanyu Li; Yerlan Idelbayev; Ahsan Mahmood; Aleksei Lebedev; Dishani Lahiri; Anujraaj Goyal; Ju Hu; Mingming Gong; Sergey Tulyakov; Anil Kag
>
> **备注:** Project page: https://snap-research.github.io/snapgenplusplus/
>
> **摘要:** Recent advances in diffusion transformers (DiTs) have set new standards in image generation, yet remain impractical for on-device deployment due to their high computational and memory costs. In this work, we present an efficient DiT framework tailored for mobile and edge devices that achieves transformer-level generation quality under strict resource constraints. Our design combines three key components. First, we propose a compact DiT architecture with an adaptive global-local sparse attention mechanism that balances global context modeling and local detail preservation. Second, we propose an elastic training framework that jointly optimizes sub-DiTs of varying capacities within a unified supernetwork, allowing a single model to dynamically adjust for efficient inference across different hardware. Finally, we develop Knowledge-Guided Distribution Matching Distillation, a step-distillation pipeline that integrates the DMD objective with knowledge transfer from few-step teacher models, producing high-fidelity and low-latency generation (e.g., 4-step) suitable for real-time on-device use. Together, these contributions enable scalable, efficient, and high-quality diffusion models for deployment on diverse hardware.
>
---
#### [replaced 051] SAIL: Self-Amplified Iterative Learning for Diffusion Model Alignment with Minimal Human Feedback
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.05380v2](https://arxiv.org/pdf/2602.05380v2)**

> **作者:** Xiaoxuan He; Siming Fu; Wanli Li; Zhiyuan Li; Dacheng Yin; Kang Rong; Fengyun Rao; Bo Zhang
>
> **摘要:** Aligning diffusion models with human preferences remains challenging, particularly when reward models are unavailable or impractical to obtain, and collecting large-scale preference datasets is prohibitively expensive. \textit{This raises a fundamental question: can we achieve effective alignment using only minimal human feedback, without auxiliary reward models, by unlocking the latent capabilities within diffusion models themselves?} In this paper, we propose \textbf{SAIL} (\textbf{S}elf-\textbf{A}mplified \textbf{I}terative \textbf{L}earning), a novel framework that enables diffusion models to act as their own teachers through iterative self-improvement. Starting from a minimal seed set of human-annotated preference pairs, SAIL operates in a closed-loop manner where the model progressively generates diverse samples, self-annotates preferences based on its evolving understanding, and refines itself using this self-augmented dataset. To ensure robust learning and prevent catastrophic forgetting, we introduce a ranked preference mixup strategy that carefully balances exploration with adherence to initial human priors. Extensive experiments demonstrate that SAIL consistently outperforms state-of-the-art methods across multiple benchmarks while using merely 6\% of the preference data required by existing approaches, revealing that diffusion models possess remarkable self-improvement capabilities that, when properly harnessed, can effectively replace both large-scale human annotation and external reward models.
>
---
#### [replaced 052] ProAPO: Progressively Automatic Prompt Optimization for Visual Classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.19844v4](https://arxiv.org/pdf/2502.19844v4)**

> **作者:** Xiangyan Qu; Gaopeng Gou; Jiamin Zhuang; Jing Yu; Kun Song; Qihao Wang; Yili Li; Gang Xiong
>
> **备注:** Accepted to the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2025
>
> **摘要:** Vision-language models (VLMs) have made significant progress in image classification by training with large-scale paired image-text data. Their performances largely depend on the prompt quality. While recent methods show that visual descriptions generated by large language models (LLMs) enhance the generalization of VLMs, class-specific prompts may be inaccurate or lack discrimination due to the hallucination in LLMs. In this paper, we aim to find visually discriminative prompts for fine-grained categories with minimal supervision and no human-in-the-loop. An evolution-based algorithm is proposed to progressively optimize language prompts from task-specific templates to class-specific descriptions. Unlike optimizing templates, the search space shows an explosion in class-specific candidate prompts. This increases prompt generation costs, iterative times, and the overfitting problem. To this end, we first introduce several simple yet effective edit-based and evolution-based operations to generate diverse candidate prompts by one-time query of LLMs. Then, two sampling strategies are proposed to find a better initial search point and reduce traversed categories, saving iteration costs. Moreover, we apply a novel fitness score with entropy constraints to mitigate overfitting. In a challenging one-shot image classification setting, our method outperforms existing textual prompt-based methods and improves LLM-generated description methods across 13 datasets. Meanwhile, we demonstrate that our optimal prompts improve adapter-based methods and transfer effectively across different backbones.
>
---
#### [replaced 053] OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.11093v2](https://arxiv.org/pdf/2503.11093v2)**

> **作者:** Yuan Liu; Saihui Hou; Saijie Hou; Jiabao Du; Shibei Meng; Yongzhen Huang
>
> **摘要:** Image Difference Captioning (IDC) aims to generate natural language descriptions of subtle differences between image pairs, requiring both precise visual change localization and coherent semantic expression. Despite recent advancements, existing datasets often lack breadth and depth, limiting their applicability in complex and dynamic environments: (1) from a breadth perspective, current datasets are constrained to limited variations of objects in specific scenes, and (2) from a depth perspective, prior benchmarks often provide overly simplistic descriptions. To address these challenges, we introduce OmniDiff, a comprehensive dataset comprising 324 diverse scenarios-spanning real-world complex environments and 3D synthetic settings-with fine-grained human annotations averaging 60 words in length and covering 12 distinct change types. Building on this foundation, we propose M$^3$Diff, a MultiModal large language model enhanced by a plug-and-play Multi-scale Differential Perception (MDP) module. This module improves the model's ability to accurately identify and describe inter-image differences while maintaining the foundational model's generalization capabilities. With the addition of the OmniDiff dataset, M$^3$Diff achieves state-of-the-art performance across multiple benchmarks, including Spot-the-Diff, IEdit, CLEVR-Change, CLEVR-DC, and OmniDiff, demonstrating significant improvements in cross-scenario difference recognition accuracy compared to existing methods. The dataset, code, and models will be made publicly available to support further research.
>
---
#### [replaced 054] WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出WorldArena，一个统一基准，用于评估具身世界模型的感知质量和功能实用性，解决当前评价碎片化问题。**

- **链接: [https://arxiv.org/pdf/2602.08971v2](https://arxiv.org/pdf/2602.08971v2)**

> **作者:** Yu Shang; Zhuohang Li; Yiding Ma; Weikang Su; Xin Jin; Ziyou Wang; Lei Jin; Xin Zhang; Yinzhou Tang; Haisheng Su; Chen Gao; Wei Wu; Xihui Liu; Dhruv Shah; Zhaoxiang Zhang; Zhibo Chen; Jun Zhu; Yonghong Tian; Tat-Seng Chua; Wenwu Zhu; Yong Li
>
> **摘要:** While world models have emerged as a cornerstone of embodied intelligence by enabling agents to reason about environmental dynamics through action-conditioned prediction, their evaluation remains fragmented. Current evaluation of embodied world models has largely focused on perceptual fidelity (e.g., video generation quality), overlooking the functional utility of these models in downstream decision-making tasks. In this work, we introduce WorldArena, a unified benchmark designed to systematically evaluate embodied world models across both perceptual and functional dimensions. WorldArena assesses models through three dimensions: video perception quality, measured with 16 metrics across six sub-dimensions; embodied task functionality, which evaluates world models as data engines, policy evaluators, and action planners integrating with subjective human evaluation. Furthermore, we propose EWMScore, a holistic metric integrating multi-dimensional performance into a single interpretable index. Through extensive experiments on 14 representative models, we reveal a significant perception-functionality gap, showing that high visual quality does not necessarily translate into strong embodied task capability. WorldArena benchmark with the public leaderboard is released at https://world-arena.ai, providing a framework for tracking progress toward truly functional world models in embodied AI.
>
---
#### [replaced 055] A New Dataset and Performance Benchmark for Real-time Spacecraft Segmentation in Onboard Computers
- **分类: cs.CV; cs.AI; eess.IV**

- **链接: [https://arxiv.org/pdf/2507.10775v2](https://arxiv.org/pdf/2507.10775v2)**

> **作者:** Jeffrey Joan Sam; Janhavi Sathe; Nikhil Chigali; Naman Gupta; Radhey Ruparel; Yicheng Jiang; Janmajay Singh; James W. Berck; Arko Barman
>
> **摘要:** Spacecraft deployed in outer space are routinely subjected to various forms of damage due to exposure to hazardous environments. In addition, there are significant risks to the subsequent process of in-space repairs through human extravehicular activity or robotic manipulation, incurring substantial operational costs. Recent developments in image segmentation could enable the development of reliable and cost-effective autonomous inspection systems. While these models often require large amounts of training data to achieve satisfactory results, publicly available annotated spacecraft segmentation data are very scarce. Here, we present a new dataset of nearly 64k annotated spacecraft images that was created using real spacecraft models, superimposed on a mixture of real and synthetic backgrounds generated using NASA's TTALOS pipeline. To mimic camera distortions and noise in real-world image acquisition, we also added different types of noise and distortion to the images. Our dataset includes images with several real-world challenges, including noise, camera distortions, glare, varying lighting conditions, varying field of view, partial spacecraft visibility, brightly-lit city backgrounds, densely patterned and confounding backgrounds, aurora borealis, and a wide variety of spacecraft geometries. Finally, we finetuned YOLOv8 and YOLOv11 models for spacecraft segmentation to generate performance benchmarks for the dataset under well-defined hardware and inference time constraints to mimic real-world image segmentation challenges for real-time onboard applications in space on NASA's inspector spacecraft. The resulting models, when tested under these constraints, achieved a Dice score of 0.92, Hausdorff distance of 0.69, and an inference time of about 0.5 second. The dataset and models for performance benchmark are available at https://github.com/RiceD2KLab/SWiM.
>
---
#### [replaced 056] CoRe3D: Collaborative Reasoning as a Foundation for 3D Intelligence
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.12768v2](https://arxiv.org/pdf/2512.12768v2)**

> **作者:** Tianjiao Yu; Xinzhuo Li; Yifan Shen; Yuanzhe Liu; Ismini Lourentzou
>
> **摘要:** Recent advances in large multimodal models suggest that explicit reasoning mechanisms play a critical role in improving model reliability, interpretability, and cross-modal alignment. While such reasoning-centric approaches have been proven effective in language and vision tasks, their extension to 3D remains underdeveloped. CoRe3D introduces a unified 3D understanding and generation reasoning framework that jointly operates over semantic and spatial abstractions, enabling high-level intent inferred from language to directly guide low-level 3D content formation. Central to this design is a spatially grounded reasoning representation that decomposes 3D latent space into localized regions, allowing the model to reason over geometry in a compositional and procedural manner. By tightly coupling semantic chain-of-thought inference with structured spatial reasoning, CoRe3D produces 3D outputs that exhibit strong local consistency and faithful alignment with linguistic descriptions.
>
---
#### [replaced 057] Multi-Level Feature Fusion for Continual Learning in Visual Quality Inspection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.00725v2](https://arxiv.org/pdf/2601.00725v2)**

> **作者:** Johannes C. Bauer; Paul Geng; Stephan Trattnig; Petr Dokládal; Rüdiger Daub
>
> **备注:** Accepted at the 2025 IEEE 13th International Conference on Control, Mechatronics and Automation (ICCMA)
>
> **摘要:** Deep neural networks show great potential for automating various visual quality inspection tasks in manufacturing. However, their applicability is limited in more volatile scenarios, such as remanufacturing, where the inspected products and defect patterns often change. In such settings, deployed models require frequent adaptation to novel conditions, effectively posing a continual learning problem. To enable quick adaptation, the necessary training processes must be computationally efficient while still avoiding effects like catastrophic forgetting. This work presents a multi-level feature fusion (MLFF) approach that aims to improve both aspects simultaneously by utilizing representations from different depths of a pretrained network. We show that our approach is able to match the performance of end-to-end training for different quality inspection problems while using significantly less trainable parameters. Furthermore, it reduces catastrophic forgetting and improves generalization robustness to new product types or defects.
>
---
#### [replaced 058] Adapt before Continual Learning
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.03956v4](https://arxiv.org/pdf/2506.03956v4)**

> **作者:** Aojun Lu; Tao Feng; Hangjie Yuan; Chunhui Ding; Yanan Sun
>
> **备注:** Accepted to AAAI2026
>
> **摘要:** Continual Learning (CL) seeks to enable neural networks to incrementally acquire new knowledge (plasticity) while retaining existing knowledge (stability). Although pre-trained models (PTMs) have provided a strong foundation for CL, existing approaches face a fundamental challenge in balancing these two competing objectives. Current methods typically address stability by freezing the PTM backbone, which severely limits the model's plasticity, particularly when incoming data distribution diverges largely from the pre-training data. Alternatively, sequentially fine-tuning the entire PTM can adapt to new knowledge but often leads to catastrophic forgetting, highlighting the critical stability-plasticity trade-off in PTM-based CL. To address this limitation, we propose Adapting PTMs before the core CL} process (ACL), a novel framework that introduces a plug-and-play adaptation phase prior to learning each new task. During this phase, ACL refines the PTM backbone by aligning embeddings with their original class prototypes while distancing them from irrelevant classes. This mechanism theoretically and empirically demonstrates desirable balance between stability and plasticity, significantly improving CL performance across benchmarks and integrated methods. Code is available at https://github.com/byyx666/ACL_code.
>
---
#### [replaced 059] MME-Emotion: A Holistic Evaluation Benchmark for Emotional Intelligence in Multimodal Large Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.09210v2](https://arxiv.org/pdf/2508.09210v2)**

> **作者:** Fan Zhang; Zebang Cheng; Chong Deng; Haoxuan Li; Zheng Lian; Qian Chen; Huadai Liu; Wen Wang; Yi-Fan Zhang; Renrui Zhang; Ziyu Guo; Zhihong Zhu; Hao Wu; Haixin Wang; Yefeng Zheng; Xiaojiang Peng; Xian Wu; Kun Wang; Xiangang Li; Jieping Ye; Pheng-Ann Heng
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have catalyzed transformative progress in affective computing, enabling models to exhibit emergent emotional intelligence. Despite substantial methodological progress, current emotional benchmarks remain limited, as it is still unknown: (a) the generalization abilities of MLLMs across distinct scenarios, and (b) their reasoning capabilities to identify the triggering factors behind emotional states. To bridge these gaps, we present \textbf{MME-Emotion}, a systematic benchmark that assesses both emotional understanding and reasoning capabilities of MLLMs, enjoying \textit{scalable capacity}, \textit{diverse settings}, and \textit{unified protocols}. As the largest emotional intelligence benchmark for MLLMs, MME-Emotion contains over 6,000 curated video clips with task-specific questioning-answering (QA) pairs, spanning broad scenarios to formulate eight emotional tasks. It further incorporates a holistic evaluation suite with hybrid metrics for emotion recognition and reasoning, analyzed through a multi-agent system framework. Through a rigorous evaluation of 20 advanced MLLMs, we uncover both their strengths and limitations, yielding several key insights: \ding{182} Current MLLMs exhibit unsatisfactory emotional intelligence, with the best-performing model achieving only $39.3\%$ recognition score and $56.0\%$ Chain-of-Thought (CoT) score on our benchmark. \ding{183} Generalist models (\emph{e.g.}, Gemini-2.5-Pro) derive emotional intelligence from generalized multimodal understanding capabilities, while specialist models (\emph{e.g.}, R1-Omni) can achieve comparable performance through domain-specific post-training adaptation. By introducing MME-Emotion, we hope that it can serve as a foundation for advancing MLLMs' emotional intelligence in the future.
>
---
#### [replaced 060] DiCo: Disentangled Concept Representation for Text-to-image Person Re-identification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.10053v2](https://arxiv.org/pdf/2601.10053v2)**

> **作者:** Giyeol Kim; Chanho Eom
>
> **摘要:** Text-to-image person re-identification (TIReID) aims to retrieve person images from a large gallery given free-form textual descriptions. TIReID is challenging due to the substantial modality gap between visual appearances and textual expressions, as well as the need to model fine-grained correspondences that distinguish individuals with similar attributes such as clothing color, texture, or outfit style. To address these issues, we propose DiCo (Disentangled Concept Representation), a novel framework that achieves hierarchical and disentangled cross-modal alignment. DiCo introduces a shared slot-based representation, where each slot acts as a part-level anchor across modalities and is further decomposed into multiple concept blocks. This design enables the disentanglement of complementary attributes (\textit{e.g.}, color, texture, shape) while maintaining consistent part-level correspondence between image and text. Extensive experiments on CUHK-PEDES, ICFG-PEDES, and RSTPReid demonstrate that our framework achieves competitive performance with state-of-the-art methods, while also enhancing interpretability through explicit slot- and block-level representations for more fine-grained retrieval results.
>
---
#### [replaced 061] GeoZero: Incentivizing Reasoning from Scratch on Geospatial Scenes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22645v2](https://arxiv.org/pdf/2511.22645v2)**

> **作者:** Di Wang; Shunyu Liu; Wentao Jiang; Fengxiang Wang; Yi Liu; Xiaolei Qin; Zhiming Luo; Chaoyang Zhou; Haonan Guo; Jing Zhang; Bo Du; Dacheng Tao; Liangpei Zhang
>
> **备注:** Code, data, and models will be publicly available at https://github.com/MiliLab/GeoZero
>
> **摘要:** Multimodal large language models (MLLMs) have undergone rapid development in advancing geospatial scene understanding. Recent studies have sought to enhance the reasoning capabilities of remote sensing MLLMs, typically through cold-start training with elaborately curated chain-of-thought (CoT) data. However, this approach not only incurs substantial annotation costs but also introduces human biases that may limit the diversity of model reasoning. To address these challenges, we propose GeoZero, a framework that enables MLLMs to perform geospatial reasoning without any predefined CoT supervision. Specifically, we construct two datasets, GeoZero-Instruct and GeoZero-Hard. GeoZero-Instruct allows the model to acquire preliminary geospatial knowledge through supervised fine-tuning, while GeoZero-Hard stimulates deep reasoning during the subsequent reinforcement learning stage. Furthermore, we introduce Answer-Anchored Group Relative Policy Optimization (A$^2$GRPO), where the reasoning process is regularized by the model's own answers, encouraging diverse yet accurate thinking. Extensive experiments on multiple remote sensing vision-language benchmarks demonstrate that GeoZero not only surpasses existing state-of-the-art methods but also fosters universal emergent reasoning capabilities across diverse geospatial tasks. Code, data, and models will be publicly available at https://github.com/MiliLab/GeoZero.
>
---
#### [replaced 062] From Preferences to Prejudice: The Role of Alignment Tuning in Shaping Social Bias in Video Diffusion Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决视频扩散模型中的社会偏见问题。通过引入VideoBiasEval框架，分析偏见在对齐过程中的演变与强化。**

- **链接: [https://arxiv.org/pdf/2510.17247v2](https://arxiv.org/pdf/2510.17247v2)**

> **作者:** Zefan Cai; Haoyi Qiu; Haozhe Zhao; Ke Wan; Jiachen Li; Jiuxiang Gu; Wen Xiao; Nanyun Peng; Junjie Hu
>
> **备注:** TMLR
>
> **摘要:** Recent advances in video diffusion models have significantly enhanced text-to-video generation, particularly through alignment tuning using reward models trained on human preferences. While these methods improve visual quality, they can unintentionally encode and amplify social biases. To systematically trace how such biases evolve throughout the alignment pipeline, we introduce VideoBiasEval, a comprehensive diagnostic framework for evaluating social representation in video generation. Grounded in established social bias taxonomies, VideoBiasEval employs an event-based prompting strategy to disentangle semantic content (actions and contexts) from actor attributes (gender and ethnicity). It further introduces multi-granular metrics to evaluate (1) overall ethnicity bias, (2) gender bias conditioned on ethnicity, (3) distributional shifts in social attributes across model variants, and (4) the temporal persistence of bias within videos. Using this framework, we conduct the first end-to-end analysis connecting biases in human preference datasets, their amplification in reward models, and their propagation through alignment-tuned video diffusion models. Our results reveal that alignment tuning not only strengthens representational biases but also makes them temporally stable, producing smoother yet more stereotyped portrayals. These findings highlight the need for bias-aware evaluation and mitigation throughout the alignment process to ensure fair and socially responsible video generation.
>
---
#### [replaced 063] Splat and Distill: Augmenting Teachers with Feed-Forward 3D Reconstruction For 3D-Aware Distillation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.06032v2](https://arxiv.org/pdf/2602.06032v2)**

> **作者:** David Shavin; Sagie Benaim
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Vision Foundation Models (VFMs) have achieved remarkable success when applied to various downstream 2D tasks. Despite their effectiveness, they often exhibit a critical lack of 3D awareness. To this end, we introduce Splat and Distill, a framework that instills robust 3D awareness into 2D VFMs by augmenting the teacher model with a fast, feed-forward 3D reconstruction pipeline. Given 2D features produced by a teacher model, our method first lifts these features into an explicit 3D Gaussian representation, in a feedforward manner. These 3D features are then ``splatted" onto novel viewpoints, producing a set of novel 2D feature maps used to supervise the student model, ``distilling" geometrically grounded knowledge. By replacing slow per-scene optimization of prior work with our feed-forward lifting approach, our framework avoids feature-averaging artifacts, creating a dynamic learning process where the teacher's consistency improves alongside that of the student. We conduct a comprehensive evaluation on a suite of downstream tasks, including monocular depth estimation, surface normal estimation, multi-view correspondence, and semantic segmentation. Our method significantly outperforms prior works, not only achieving substantial gains in 3D awareness but also enhancing the underlying semantic richness of 2D features. Project page is available at https://davidshavin4.github.io/Splat-and-Distill/
>
---
#### [replaced 064] CER-HV: A CER-Based Human-in-the-Loop Framework for Cleaning Datasets Applied to Arabic-Script HTR
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.16713v2](https://arxiv.org/pdf/2601.16713v2)**

> **作者:** Sana Al-azzawi; Elisa Barney; Marcus Liwicki
>
> **摘要:** Handwritten text recognition (HTR) for Arabic-script languages still lags behind Latin-script HTR, despite recent advances in model architectures, datasets, and benchmarks. We show that data quality is a significant limiting factor in many published datasets and propose CER-HV (CER-based Ranking with Human Verification) as a framework to detect and clean label errors. CER-HV combines a CER-based noise detector, built on a carefully configured Convolutional Recurrent Neural Network (CRNN) with early stopping to avoid overfitting noisy samples, and a human-in-the-loop (HITL) step that verifies high-ranking samples. The framework reveals that several existing datasets contain previously underreported problems, including transcription, segmentation, orientation, and non-text content errors. These have been identified with up to 90 percent precision in the Muharaf and 80-86 percent in the PHTI datasets. We also show that our CRNN achieves state-of-the-art performance across five of the six evaluated datasets, reaching 8.45 percent Character Error Rate (CER) on KHATT (Arabic), 8.26 percent on PHTI (Pashto), 10.66 percent on Ajami, and 10.11 percent on Muharaf (Arabic), all without any data cleaning. We establish a new baseline of 11.3 percent CER on the PHTD (Persian) dataset. Applying CER-HV improves the evaluation CER by 0.3-0.6 percent on the cleaner datasets and 1.0-1.8 percent on the noisier ones. Although our experiments focus on documents written in an Arabic-script language, including Arabic, Persian, Urdu, Ajami, and Pashto, the framework is general and can be applied to other text recognition datasets.
>
---
#### [replaced 065] SpatialReward: Bridging the Perception Gap in Online RL for Image Editing via Explicit Spatial Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.07458v2](https://arxiv.org/pdf/2602.07458v2)**

> **作者:** Yancheng Long; Yankai Yang; Hongyang Wei; Wei Chen; Tianke Zhang; Haonan fan; Changyi Liu; Kaiyu Jiang; Jiankang Chen; Kaiyu Tang; Bin Wen; Fan Yang; Tingting Gao; Han Li; Shuo Yang
>
> **摘要:** Online Reinforcement Learning (RL) offers a promising avenue for complex image editing but is currently constrained by the scarcity of reliable and fine-grained reward signals. Existing evaluators frequently struggle with a critical perception gap we term "Attention Collapse," where models neglect cross-image comparisons and fail to capture fine-grained details, resulting in inaccurate perception and miscalibrated scores. To address these limitations, we propose SpatialReward, a reward model that enforces precise verification via explicit spatial reasoning. By anchoring reasoning to predicted edit regions, SpatialReward grounds semantic judgments in pixel-level evidence, significantly enhancing evaluative accuracy. Trained on a curated 260k spatial-aware dataset, our model achieves state-of-the-art performance on MMRB2 and EditReward-Bench, and outperforms proprietary evaluators on our proposed MultiEditReward-Bench. Furthermore, SpatialReward serves as a robust signal in online RL, boosting OmniGen2 by +0.90 on GEdit-Bench--surpassing the leading discriminative model and doubling the gain of GPT-4.1 (+0.45). These results demonstrate that spatial reasoning is essential for unlocking effective alignment in image editing.
>
---
#### [replaced 066] From Pixels to Images: A Structural Survey of Deep Learning Paradigms in Remote Sensing Image Semantic Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.15147v2](https://arxiv.org/pdf/2505.15147v2)**

> **作者:** Quanwei Liu; Tao Huang; Jiaqi Yang; Wei Xiang
>
> **备注:** 34 pages, 9 figures, 5 tables
>
> **摘要:** Semantic segmentation (SS) of RSIs enables the fine-grained interpretation of surface features, making it a critical task in RS analysis. With the increasing diversity and volume of RSIs collected by sensors on various platforms, traditional processing methods struggle to maintain efficiency and accuracy. In response, deep learning (DL) has emerged as a transformative approach, enabling substantial advances in remote sensing image semantic segmentation (RSISS) by automating hierarchical feature extraction and improving segmentation performance across diverse modalities. As data scale and model capacity have increased, DL-based RSISS has undergone a structural evolution from pixel-level and patch-based classification to tile-level, end-to-end segmentation, and, more recently, to image-level modelling with vision foundation models. However, existing reviews often focus on individual components, such as supervision strategies or fusion stages, and lack a unified operational perspective aligned with segmentation granularity and the training/inference pipeline. This paper provides a comprehensive review by organizing DL-based RSISS into a pixel-patch-tile-image hierarchy, covering early pixel-based methods, prevailing patch-based and tile-based techniques, and emerging image-based approaches. This review offers a holistic and structured understanding of DL-based RSISS, highlighting representative datasets, comparative insights, and open challenges related to data scale, model efficiency, domain robustness, and multimodal integration. Furthermore, to facilitate reproducible research, curated code collections are provided at: https://github.com/quanweiliu/PatchwiseClsFra and https://github.com/quanweiliu/TilewiseSegFra.
>
---
#### [replaced 067] Contextual Range-View Projection for 3D LiDAR Point Clouds
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.18301v2](https://arxiv.org/pdf/2601.18301v2)**

> **作者:** Seyedali Mousavi; Seyedhamidreza Mousavi; Masoud Daneshtalab
>
> **摘要:** Range-view projection provides an efficient method for transforming 3D LiDAR point clouds into 2D range image representations, enabling effective processing with 2D deep learning models. However, a major challenge in this projection is the many-to-one conflict, where multiple 3D points are mapped onto the same pixel in the range image, requiring a selection strategy. Existing approaches typically retain the point with the smallest depth (closest to the LiDAR), disregarding semantic relevance and object structure, which leads to the loss of important contextual information. In this paper, we extend the depth-based selection rule by incorporating contextual information from both instance centers and class labels, introducing two mechanisms: \textit{Centerness-Aware Projection (CAP)} and \textit{Class-Weighted-Aware Projection (CWAP)}. In CAP, point depths are adjusted according to their distance from the instance center, thereby prioritizing central instance points over noisy boundary and background points. In CWAP, object classes are prioritized through user-defined weights, offering flexibility in the projection strategy. Our evaluations on the SemanticKITTI dataset show that CAP preserves more instance points during projection, achieving up to a 3.1\% mIoU improvement compared to the baseline. Furthermore, CWAP enhances the performance of targeted classes while having a negligible impact on the performance of other classes
>
---
#### [replaced 068] Symmetrization Weighted Binary Cross-Entropy: Modeling Perceptual Asymmetry for Human-Consistent Neural Edge Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2501.13365v4](https://arxiv.org/pdf/2501.13365v4)**

> **作者:** Hao Shu
>
> **备注:** 39 pages
>
> **摘要:** Edge detection (ED) is a fundamental perceptual process in computer vision, forming the structural basis for high-level reasoning tasks such as segmentation, recognition, and scene understanding. Despite substantial progress achieved by deep neural networks, most ED models attain high numerical accuracy but fail to produce visually sharp and perceptually consistent edges, thereby limiting their reliability in intelligent vision systems. To address this issue, this study introduces the Symmetrization Weighted Binary Cross-Entropy (SWBCE) loss, a perception-inspired formulation that extends the conventional WBCE by incorporating prediction-guided symmetry. SWBCE explicitly models the perceptual asymmetry in human edge recognition, wherein edge decisions require stronger evidence than non-edge ones, aligning the optimization process with human perceptual discrimination. The resulting symmetric learning mechanism jointly enhances edge recall and suppresses false positives, achieving a superior balance between quantitative accuracy and perceptual fidelity. Extensive experiments across multiple benchmark datasets and representative ED architectures demonstrate that SWBCE can outperform existing loss functions in both numerical evaluation and visual quality. Particularly with the HED-EES model, the SSIM can be improved by about 15% on BRIND, and in all experiments, training by SWBCE consistently obtains the best perceptual results. Beyond edge detection, the proposed perceptual loss offers a generalizable optimization principle for soft computing and neural learning systems, particularly in scenarios where asymmetric perceptual reasoning plays a critical role.
>
---
#### [replaced 069] Fake-HR1: Rethinking Reasoning of Vision Language Model for Synthetic Image Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.10042v2](https://arxiv.org/pdf/2602.10042v2)**

> **作者:** Changjiang Jiang; Xinkuan Sha; Fengchang Yu; Jingjing Liu; Jian Liu; Mingqi Fang; Chenfeng Zhang; Wei Lu
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Recent studies have demonstrated that incorporating Chain-of-Thought (CoT) reasoning into the detection process can enhance a model's ability to detect synthetic images. However, excessively lengthy reasoning incurs substantial resource overhead, including token consumption and latency, which is particularly redundant when handling obviously generated forgeries. To address this issue, we propose Fake-HR1, a large-scale hybrid-reasoning model that, to the best of our knowledge, is the first to adaptively determine whether reasoning is necessary based on the characteristics of the generative detection task. To achieve this, we design a two-stage training framework: we first perform Hybrid Fine-Tuning (HFT) for cold-start initialization, followed by online reinforcement learning with Hybrid-Reasoning Grouped Policy Optimization (HGRPO) to implicitly learn when to select an appropriate reasoning mode. Experimental results show that Fake-HR1 adaptively performs reasoning across different types of queries, surpassing existing LLMs in both reasoning ability and generative detection performance, while significantly improving response efficiency.
>
---
