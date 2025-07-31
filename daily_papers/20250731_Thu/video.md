# 计算机视觉 cs.CV

- **最新发布 84 篇**

- **更新 82 篇**

## 最新发布

#### [new 001] Bridging the Gap in Missing Modalities: Leveraging Knowledge Distillation and Style Matching for Brain Tumor Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决脑肿瘤分割中缺失模态带来的挑战。作者提出了MST-KDNet模型，结合多尺度知识蒸馏与风格匹配，提升分割精度与鲁棒性，尤其在模态缺失情况下表现优越。**

- **链接: [http://arxiv.org/pdf/2507.22626v1](http://arxiv.org/pdf/2507.22626v1)**

> **作者:** Shenghao Zhu; Yifei Chen; Weihong Chen; Yuanhan Wang; Chang Liu; Shuo Jiang; Feiwei Qin; Changmiao Wang
>
> **备注:** 11 pages, 2 figures
>
> **摘要:** Accurate and reliable brain tumor segmentation, particularly when dealing with missing modalities, remains a critical challenge in medical image analysis. Previous studies have not fully resolved the challenges of tumor boundary segmentation insensitivity and feature transfer in the absence of key imaging modalities. In this study, we introduce MST-KDNet, aimed at addressing these critical issues. Our model features Multi-Scale Transformer Knowledge Distillation to effectively capture attention weights at various resolutions, Dual-Mode Logit Distillation to improve the transfer of knowledge, and a Global Style Matching Module that integrates feature matching with adversarial learning. Comprehensive experiments conducted on the BraTS and FeTS 2024 datasets demonstrate that MST-KDNet surpasses current leading methods in both Dice and HD95 scores, particularly in conditions with substantial modality loss. Our approach shows exceptional robustness and generalization potential, making it a promising candidate for real-world clinical applications. Our source code is available at https://github.com/Quanato607/MST-KDNet.
>
---
#### [new 002] Learning from Heterogeneous Structural MRI via Collaborative Domain Adaptation for Late-Life Depression Assessment
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学影像分析任务，旨在解决小样本与多中心数据异质性对晚年抑郁症诊断的影响。作者提出了一种基于视觉变换器与卷积网络的协作域适应框架，通过自监督适应与协同训练提升跨域泛化能力，实现了更准确的抑郁症检测。**

- **链接: [http://arxiv.org/pdf/2507.22321v1](http://arxiv.org/pdf/2507.22321v1)**

> **作者:** Yuzhen Gao; Qianqian Wang; Yongheng Sun; Cui Wang; Yongquan Liang; Mingxia Liu
>
> **摘要:** Accurate identification of late-life depression (LLD) using structural brain MRI is essential for monitoring disease progression and facilitating timely intervention. However, existing learning-based approaches for LLD detection are often constrained by limited sample sizes (e.g., tens), which poses significant challenges for reliable model training and generalization. Although incorporating auxiliary datasets can expand the training set, substantial domain heterogeneity, such as differences in imaging protocols, scanner hardware, and population demographics, often undermines cross-domain transferability. To address this issue, we propose a Collaborative Domain Adaptation (CDA) framework for LLD detection using T1-weighted MRIs. The CDA leverages a Vision Transformer (ViT) to capture global anatomical context and a Convolutional Neural Network (CNN) to extract local structural features, with each branch comprising an encoder and a classifier. The CDA framework consists of three stages: (a) supervised training on labeled source data, (b) self-supervised target feature adaptation and (c) collaborative training on unlabeled target data. We first train ViT and CNN on source data, followed by self-supervised target feature adaptation by minimizing the discrepancy between classifier outputs from two branches to make the categorical boundary clearer. The collaborative training stage employs pseudo-labeled and augmented target-domain MRIs, enforcing prediction consistency under strong and weak augmentation to enhance domain robustness and generalization. Extensive experiments conducted on multi-site T1-weighted MRI data demonstrate that the CDA consistently outperforms state-of-the-art unsupervised domain adaptation methods.
>
---
#### [new 003] SpectraSentinel: LightWeight Dual-Stream Real-Time Drone Detection, Tracking and Payload Identification
- **分类: cs.CV**

- **简介: 该论文属于无人机监控任务，旨在解决实时检测、跟踪及识别无人机载荷的问题。作者提出了SpectraSentinel框架，采用双流YOLOv11n模型分别处理红外和可见光数据，避免早期融合，优化各自模态的预处理和训练策略，实现高效准确的无人机监控。**

- **链接: [http://arxiv.org/pdf/2507.22650v1](http://arxiv.org/pdf/2507.22650v1)**

> **作者:** Shahriar Kabir; Istiak Ahmmed Rifti; H. M. Shadman Tabib; Mushfiqur Rahman; Sadatul Islam Sadi; Hasnaen Adil; Ahmed Mahir Sultan Rumi; Ch Md Rakin Haider
>
> **摘要:** The proliferation of drones in civilian airspace has raised urgent security concerns, necessitating robust real-time surveillance systems. In response to the 2025 VIP Cup challenge tasks - drone detection, tracking, and payload identification - we propose a dual-stream drone monitoring framework. Our approach deploys independent You Only Look Once v11-nano (YOLOv11n) object detectors on parallel infrared (thermal) and visible (RGB) data streams, deliberately avoiding early fusion. This separation allows each model to be specifically optimized for the distinct characteristics of its input modality, addressing the unique challenges posed by small aerial objects in diverse environmental conditions. We customize data preprocessing and augmentation strategies per domain - such as limiting color jitter for IR imagery - and fine-tune training hyperparameters to enhance detection performance under conditions of heavy noise, low light, and motion blur. The resulting lightweight YOLOv11n models demonstrate high accuracy in distinguishing drones from birds and in classifying payload types, all while maintaining real-time performance. This report details the rationale for a dual-modality design, the specialized training pipelines, and the architectural optimizations that collectively enable efficient and accurate drone surveillance across RGB and IR channels.
>
---
#### [new 004] Social-Pose: Enhancing Trajectory Prediction with Human Body Pose
- **分类: cs.CV**

- **简介: 该论文属于轨迹预测任务，旨在解决自动驾驶中人类轨迹预测不够准确的问题。通过引入人体姿态信息，提出“Social-Pose”模型，利用注意力机制捕捉姿态与社交关系，提升了LSTM、GAN等多种模型在多个数据集上的预测性能。**

- **链接: [http://arxiv.org/pdf/2507.22742v1](http://arxiv.org/pdf/2507.22742v1)**

> **作者:** Yang Gao; Saeed Saadatnejad; Alexandre Alahi
>
> **备注:** Accepted to IEEE Transactions on Intelligent Transportation Systems (T-ITS)
>
> **摘要:** Accurate human trajectory prediction is one of the most crucial tasks for autonomous driving, ensuring its safety. Yet, existing models often fail to fully leverage the visual cues that humans subconsciously communicate when navigating the space. In this work, we study the benefits of predicting human trajectories using human body poses instead of solely their Cartesian space locations in time. We propose `Social-pose', an attention-based pose encoder that effectively captures the poses of all humans in a scene and their social relations. Our method can be integrated into various trajectory prediction architectures. We have conducted extensive experiments on state-of-the-art models (based on LSTM, GAN, MLP, and Transformer), and showed improvements over all of them on synthetic (Joint Track Auto) and real (Human3.6M, Pedestrians and Cyclists in Road Traffic, and JRDB) datasets. We also explored the advantages of using 2D versus 3D poses, as well as the effect of noisy poses and the application of our pose-based predictor in robot navigation scenarios.
>
---
#### [new 005] LIDAR: Lightweight Adaptive Cue-Aware Fusion Vision Mamba for Multimodal Segmentation of Structural Cracks
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态裂缝分割任务，旨在解决现有方法在自适应感知和跨模态特征融合上的不足。作者提出了一种轻量级网络LIDAR，包含LacaVSS和LD3CF模块，结合动态扫描策略与频域感知，实现高效的像素级裂缝分割。**

- **链接: [http://arxiv.org/pdf/2507.22477v1](http://arxiv.org/pdf/2507.22477v1)**

> **作者:** Hui Liu; Chen Jia; Fan Shi; Xu Cheng; Mengfei Shi; Xia Xie; Shengyong Chen
>
> **摘要:** Achieving pixel-level segmentation with low computational cost using multimodal data remains a key challenge in crack segmentation tasks. Existing methods lack the capability for adaptive perception and efficient interactive fusion of cross-modal features. To address these challenges, we propose a Lightweight Adaptive Cue-Aware Vision Mamba network (LIDAR), which efficiently perceives and integrates morphological and textural cues from different modalities under multimodal crack scenarios, generating clear pixel-level crack segmentation maps. Specifically, LIDAR is composed of a Lightweight Adaptive Cue-Aware Visual State Space module (LacaVSS) and a Lightweight Dual Domain Dynamic Collaborative Fusion module (LD3CF). LacaVSS adaptively models crack cues through the proposed mask-guided Efficient Dynamic Guided Scanning Strategy (EDG-SS), while LD3CF leverages an Adaptive Frequency Domain Perceptron (AFDP) and a dual-pooling fusion strategy to effectively capture spatial and frequency-domain cues across modalities. Moreover, we design a Lightweight Dynamically Modulated Multi-Kernel convolution (LDMK) to perceive complex morphological structures with minimal computational overhead, replacing most convolutional operations in LIDAR. Experiments on three datasets demonstrate that our method outperforms other state-of-the-art (SOTA) methods. On the light-field depth dataset, our method achieves 0.8204 in F1 and 0.8465 in mIoU with only 5.35M parameters. Code and datasets are available at https://github.com/Karl1109/LIDAR-Mamba.
>
---
#### [new 006] FaceGCD: Generalized Face Discovery via Dynamic Prefix Generation
- **分类: cs.CV**

- **简介: 论文提出“广义人脸发现”（GFD）任务，结合人脸识别与广义类别发现，旨在同时识别已知人脸并发现新身份。针对现有方法在人脸高细粒度与高基数上的不足，作者设计了FaceGCD，通过动态生成轻量级前缀，提升开放世界下的人脸识别性能，实现SOTA效果。**

- **链接: [http://arxiv.org/pdf/2507.22353v1](http://arxiv.org/pdf/2507.22353v1)**

> **作者:** Yunseok Oh; Dong-Wan Choi
>
> **备注:** BMVC 2025 Accepted
>
> **摘要:** Recognizing and differentiating among both familiar and unfamiliar faces is a critical capability for face recognition systems and a key step toward artificial general intelligence (AGI). Motivated by this ability, this paper introduces generalized face discovery (GFD), a novel open-world face recognition task that unifies traditional face identification with generalized category discovery (GCD). GFD requires recognizing both labeled and unlabeled known identities (IDs) while simultaneously discovering new, previously unseen IDs. Unlike typical GCD settings, GFD poses unique challenges due to the high cardinality and fine-grained nature of face IDs, rendering existing GCD approaches ineffective. To tackle this problem, we propose FaceGCD, a method that dynamically constructs instance-specific feature extractors using lightweight, layer-wise prefixes. These prefixes are generated on the fly by a HyperNetwork, which adaptively outputs a set of prefix generators conditioned on each input image. This dynamic design enables FaceGCD to capture subtle identity-specific cues without relying on high-capacity static models. Extensive experiments demonstrate that FaceGCD significantly outperforms existing GCD methods and a strong face recognition baseline, ArcFace, achieving state-of-the-art results on the GFD task and advancing toward open-world face recognition.
>
---
#### [new 007] On the Reliability of Vision-Language Models Under Adversarial Frequency-Domain Perturbations
- **分类: cs.CV**

- **简介: 该论文研究视觉-语言模型（VLMs）在对抗性频率域扰动下的可靠性，属于多模态感知任务。论文揭示了VLM在图像真实性检测和图像描述生成任务中对频率域扰动的脆弱性，通过设计频率域图像变换，验证其对多个先进模型的普遍影响，强调构建更鲁棒多模态系统的重要性。**

- **链接: [http://arxiv.org/pdf/2507.22398v1](http://arxiv.org/pdf/2507.22398v1)**

> **作者:** Jordan Vice; Naveed Akhtar; Yansong Gao; Richard Hartley; Ajmal Mian
>
> **备注:** Keywords: Vision-Language Models, Frequency-Domain Perturbations, Adversarial Robustness, Image Authenticity, Reliability
>
> **摘要:** Vision-Language Models (VLMs) are increasingly used as perceptual modules for visual content reasoning, including through captioning and DeepFake detection. In this work, we expose a critical vulnerability of VLMs when exposed to subtle, structured perturbations in the frequency domain. Specifically, we highlight how these feature transformations undermine authenticity/DeepFake detection and automated image captioning tasks. We design targeted image transformations, operating in the frequency domain to systematically adjust VLM outputs when exposed to frequency-perturbed real and synthetic images. We demonstrate that the perturbation injection method generalizes across five state-of-the-art VLMs which includes different-parameter Qwen2/2.5 and BLIP models. Experimenting across ten real and generated image datasets reveals that VLM judgments are sensitive to frequency-based cues and may not wholly align with semantic content. Crucially, we show that visually-imperceptible spatial frequency transformations expose the fragility of VLMs deployed for automated image captioning and authenticity detection tasks. Our findings under realistic, black-box constraints challenge the reliability of VLMs, underscoring the need for robust multimodal perception systems.
>
---
#### [new 008] DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion
- **分类: cs.CV**

- **简介: 该论文属于深度学习安全任务，旨在解决神经网络模型中的后门攻击检测问题。作者提出了一种无需训练数据的触发器逆向方法DISTIL，通过扩散模型结合目标分类器生成潜在触发模式，有效区分干净与中毒模型，提升了后门检测的准确性。**

- **链接: [http://arxiv.org/pdf/2507.22813v1](http://arxiv.org/pdf/2507.22813v1)**

> **作者:** Hossein Mirzaei; Zeinab Taghavi; Sepehr Rezaee; Masoud Hadi; Moein Madadi; Mackenzie W. Mathis
>
> **备注:** ICCV 2025
>
> **摘要:** Deep neural networks have demonstrated remarkable success across numerous tasks, yet they remain vulnerable to Trojan (backdoor) attacks, raising serious concerns about their safety in real-world mission-critical applications. A common countermeasure is trigger inversion -- reconstructing malicious "shortcut" patterns (triggers) inserted by an adversary during training. Current trigger-inversion methods typically search the full pixel space under specific assumptions but offer no assurances that the estimated trigger is more than an adversarial perturbation that flips the model output. Here, we propose a data-free, zero-shot trigger-inversion strategy that restricts the search space while avoiding strong assumptions on trigger appearance. Specifically, we incorporate a diffusion-based generator guided by the target classifier; through iterative generation, we produce candidate triggers that align with the internal representations the model relies on for malicious behavior. Empirical evaluations, both quantitative and qualitative, show that our approach reconstructs triggers that effectively distinguish clean versus Trojaned models. DISTIL surpasses alternative methods by high margins, achieving up to 7.1% higher accuracy on the BackdoorBench dataset and a 9.4% improvement on trojaned object detection model scanning, offering a promising new direction for reliable backdoor defense without reliance on extensive data or strong prior assumptions about triggers. The code is available at https://github.com/AdaptiveMotorControlLab/DISTIL.
>
---
#### [new 009] MINR: Implicit Neural Representations with Masked Image Modelling
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于自监督学习任务，旨在解决现有方法如MAE在不同掩码策略下表现不稳定、对分布外数据泛化能力差的问题。作者提出MINR框架，结合隐式神经表示与掩码图像建模，实现更鲁棒、更通用的图像重建。实验表明MINR在多个场景下优于MAE，同时降低模型复杂度。**

- **链接: [http://arxiv.org/pdf/2507.22404v1](http://arxiv.org/pdf/2507.22404v1)**

> **作者:** Sua Lee; Joonhun Lee; Myungjoo Kang
>
> **备注:** Accepted to the ICCV 2023 workshop on Out-of-Distribution Generalization in Computer Vision
>
> **摘要:** Self-supervised learning methods like masked autoencoders (MAE) have shown significant promise in learning robust feature representations, particularly in image reconstruction-based pretraining task. However, their performance is often strongly dependent on the masking strategies used during training and can degrade when applied to out-of-distribution data. To address these limitations, we introduce the masked implicit neural representations (MINR) framework that synergizes implicit neural representations with masked image modeling. MINR learns a continuous function to represent images, enabling more robust and generalizable reconstructions irrespective of masking strategies. Our experiments demonstrate that MINR not only outperforms MAE in in-domain scenarios but also in out-of-distribution settings, while reducing model complexity. The versatility of MINR extends to various self-supervised learning applications, confirming its utility as a robust and efficient alternative to existing frameworks.
>
---
#### [new 010] Aleatoric Uncertainty Medical Image Segmentation Estimation via Flow Matching
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，旨在量化分割中的偶然不确定性。现有方法在建模分割分布时表达能力受限，无法准确捕捉不确定性。作者提出基于条件流匹配的方法，通过学习精确密度并生成多个分割样本，以像素级方差反映数据分布，有效捕捉模糊区域的不确定性，提供可靠不确定性图。**

- **链接: [http://arxiv.org/pdf/2507.22418v1](http://arxiv.org/pdf/2507.22418v1)**

> **作者:** Phi Van Nguyen; Ngoc Huynh Trinh; Duy Minh Lam Nguyen; Phu Loc Nguyen; Quoc Long Tran
>
> **摘要:** Quantifying aleatoric uncertainty in medical image segmentation is critical since it is a reflection of the natural variability observed among expert annotators. A conventional approach is to model the segmentation distribution using the generative model, but current methods limit the expression ability of generative models. While current diffusion-based approaches have demonstrated impressive performance in approximating the data distribution, their inherent stochastic sampling process and inability to model exact densities limit their effectiveness in accurately capturing uncertainty. In contrast, our proposed method leverages conditional flow matching, a simulation-free flow-based generative model that learns an exact density, to produce highly accurate segmentation results. By guiding the flow model on the input image and sampling multiple data points, our approach synthesizes segmentation samples whose pixel-wise variance reliably reflects the underlying data distribution. This sampling strategy captures uncertainties in regions with ambiguous boundaries, offering robust quantification that mirrors inter-annotator differences. Experimental results demonstrate that our method not only achieves competitive segmentation accuracy but also generates uncertainty maps that provide deeper insights into the reliability of the segmentation outcomes. The code for this paper is freely available at https://github.com/huynhspm/Data-Uncertainty
>
---
#### [new 011] MoCHA: Advanced Vision-Language Reasoning with MoE Connector and Hierarchical Group Attention
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言推理任务，旨在解决现有视觉大模型训练和推理成本高、视觉细节提取与跨模态融合效果不佳的问题。论文提出MoCHA框架，融合多种视觉编码器、设计稀疏专家连接模块和分层组注意力机制，提升模型性能与效率。**

- **链接: [http://arxiv.org/pdf/2507.22805v1](http://arxiv.org/pdf/2507.22805v1)**

> **作者:** Yuqi Pang; Bowen Yang; Yun Cao; Fan Rong; Xiaoyu Li; Chen He
>
> **摘要:** Vision large language models (VLLMs) are focusing primarily on handling complex and fine-grained visual information by incorporating advanced vision encoders and scaling up visual models. However, these approaches face high training and inference costs, as well as challenges in extracting visual details, effectively bridging across modalities. In this work, we propose a novel visual framework, MoCHA, to address these issues. Our framework integrates four vision backbones (i.e., CLIP, SigLIP, DINOv2 and ConvNeXt) to extract complementary visual features and is equipped with a sparse Mixture of Experts Connectors (MoECs) module to dynamically select experts tailored to different visual dimensions. To mitigate redundant or insufficient use of the visual information encoded by the MoECs module, we further design a Hierarchical Group Attention (HGA) with intra- and inter-group operations and an adaptive gating strategy for encoded visual features. We train MoCHA on two mainstream LLMs (e.g., Phi2-2.7B and Vicuna-7B) and evaluate their performance across various benchmarks. Notably, MoCHA outperforms state-of-the-art open-weight models on various tasks. For example, compared to CuMo (Mistral-7B), our MoCHA (Phi2-2.7B) presents outstanding abilities to mitigate hallucination by showing improvements of 3.25% in POPE and to follow visual instructions by raising 153 points on MME. Finally, ablation studies further confirm the effectiveness and robustness of the proposed MoECs and HGA in improving the overall performance of MoCHA.
>
---
#### [new 012] ShortFT: Diffusion Model Alignment via Shortcut-based Fine-Tuning
- **分类: cs.CV**

- **简介: 该论文属于扩散模型对齐任务，旨在解决传统方法因长去噪链导致的计算成本高和梯度爆炸问题。作者提出ShortFT，通过使用短去噪链进行微调，提升扩散模型与奖励函数的对齐效果，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.22604v1](http://arxiv.org/pdf/2507.22604v1)**

> **作者:** Xiefan Guo; Miaomiao Cui; Liefeng Bo; Di Huang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Backpropagation-based approaches aim to align diffusion models with reward functions through end-to-end backpropagation of the reward gradient within the denoising chain, offering a promising perspective. However, due to the computational costs and the risk of gradient explosion associated with the lengthy denoising chain, existing approaches struggle to achieve complete gradient backpropagation, leading to suboptimal results. In this paper, we introduce Shortcut-based Fine-Tuning (ShortFT), an efficient fine-tuning strategy that utilizes the shorter denoising chain. More specifically, we employ the recently researched trajectory-preserving few-step diffusion model, which enables a shortcut over the original denoising chain, and construct a shortcut-based denoising chain of shorter length. The optimization on this chain notably enhances the efficiency and effectiveness of fine-tuning the foundational model. Our method has been rigorously tested and can be effectively applied to various reward functions, significantly improving alignment performance and surpassing state-of-the-art alternatives.
>
---
#### [new 013] Object Recognition Datasets and Challenges: A Review
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉任务中的目标识别研究。旨在分析常用数据集的特点，解决数据集选择与评估问题。作者统计并描述了160多个数据集，综述了主流基准和评测指标，为研究者提供参考。**

- **链接: [http://arxiv.org/pdf/2507.22361v1](http://arxiv.org/pdf/2507.22361v1)**

> **作者:** Aria Salari; Abtin Djavadifar; Xiangrui Liu; Homayoun Najjaran
>
> **摘要:** Object recognition is among the fundamental tasks in the computer vision applications, paving the path for all other image understanding operations. In every stage of progress in object recognition research, efforts have been made to collect and annotate new datasets to match the capacity of the state-of-the-art algorithms. In recent years, the importance of the size and quality of datasets has been intensified as the utility of the emerging deep network techniques heavily relies on training data. Furthermore, datasets lay a fair benchmarking means for competitions and have proved instrumental to the advancements of object recognition research by providing quantifiable benchmarks for the developed models. Taking a closer look at the characteristics of commonly-used public datasets seems to be an important first step for data-driven and machine learning researchers. In this survey, we provide a detailed analysis of datasets in the highly investigated object recognition areas. More than 160 datasets have been scrutinized through statistics and descriptions. Additionally, we present an overview of the prominent object recognition benchmarks and competitions, along with a description of the metrics widely adopted for evaluation purposes in the computer vision community. All introduced datasets and challenges can be found online at github.com/AbtinDjavadifar/ORDC.
>
---
#### [new 014] UFV-Splatter: Pose-Free Feed-Forward 3D Gaussian Splatting Adapted to Unfavorable Views
- **分类: cs.CV**

- **简介: 该论文属于3D重建与渲染任务，旨在解决现有前馈3D高斯泼溅模型在不利视角输入下效果差的问题。作者提出UFV-Splatter，通过引入低秩适应层、高斯适配模块和对齐方法，使模型能处理未知姿态的输入，仅需有利视角图像训练，提升了实际场景适用性。**

- **链接: [http://arxiv.org/pdf/2507.22342v1](http://arxiv.org/pdf/2507.22342v1)**

> **作者:** Yuki Fujimura; Takahiro Kushida; Kazuya Kitano; Takuya Funatomi; Yasuhiro Mukaigawa
>
> **备注:** Project page: https://yfujimura.github.io/UFV-Splatter_page/
>
> **摘要:** This paper presents a pose-free, feed-forward 3D Gaussian Splatting (3DGS) framework designed to handle unfavorable input views. A common rendering setup for training feed-forward approaches places a 3D object at the world origin and renders it from cameras pointed toward the origin -- i.e., from favorable views, limiting the applicability of these models to real-world scenarios involving varying and unknown camera poses. To overcome this limitation, we introduce a novel adaptation framework that enables pretrained pose-free feed-forward 3DGS models to handle unfavorable views. We leverage priors learned from favorable images by feeding recentered images into a pretrained model augmented with low-rank adaptation (LoRA) layers. We further propose a Gaussian adapter module to enhance the geometric consistency of the Gaussians derived from the recentered inputs, along with a Gaussian alignment method to render accurate target views for training. Additionally, we introduce a new training strategy that utilizes an off-the-shelf dataset composed solely of favorable images. Experimental results on both synthetic images from the Google Scanned Objects dataset and real images from the OmniObject3D dataset validate the effectiveness of our method in handling unfavorable input views.
>
---
#### [new 015] MergeSAM: Unsupervised change detection of remote sensing images based on the Segment Anything Model
- **分类: cs.CV**

- **简介: 该论文属于遥感图像处理任务，旨在解决高分辨率遥感图像的无监督变化检测问题。针对对象分裂、合并等复杂变化，提出MergeSAM方法，结合Segment Anything Model（SAM）的分割能力，设计MaskMatching与MaskSplitting策略，提升变化检测效果。**

- **链接: [http://arxiv.org/pdf/2507.22675v1](http://arxiv.org/pdf/2507.22675v1)**

> **作者:** Meiqi Hu; Lingzhi Lu; Chengxi Han; Xiaoping Liu
>
> **备注:** 4 pages
>
> **摘要:** Recently, large foundation models trained on vast datasets have demonstrated exceptional capabilities in feature extraction and general feature representation. The ongoing advancements in deep learning-driven large models have shown great promise in accelerating unsupervised change detection methods, thereby enhancing the practical applicability of change detection technologies. Building on this progress, this paper introduces MergeSAM, an innovative unsupervised change detection method for high-resolution remote sensing imagery, based on the Segment Anything Model (SAM). Two novel strategies, MaskMatching and MaskSplitting, are designed to address real-world complexities such as object splitting, merging, and other intricate changes. The proposed method fully leverages SAM's object segmentation capabilities to construct multitemporal masks that capture complex changes, embedding the spatial structure of land cover into the change detection process.
>
---
#### [new 016] Segment Anything for Video: A Comprehensive Review of Video Object Segmentation and Tracking from Past to Future
- **分类: cs.CV**

- **简介: 该论文属于视频目标分割与跟踪（VOST）任务，旨在解决传统方法在领域泛化、时间一致性和计算效率上的不足。论文系统回顾了基于SAM和SAM2模型的方法，从历史信息保留、当前帧特征优化到未来运动预测三个方面展开，梳理了从早期记忆架构到流式记忆与实时分割的发展，同时探讨了当前进展与未来研究方向。**

- **链接: [http://arxiv.org/pdf/2507.22792v1](http://arxiv.org/pdf/2507.22792v1)**

> **作者:** Guoping Xu; Jayaram K. Udupa; Yajun Yu; Hua-Chieh Shao; Songlin Zhao; Wei Liu; You Zhang
>
> **备注:** 45 pages, 21 figures
>
> **摘要:** Video Object Segmentation and Tracking (VOST) presents a complex yet critical challenge in computer vision, requiring robust integration of segmentation and tracking across temporally dynamic frames. Traditional methods have struggled with domain generalization, temporal consistency, and computational efficiency. The emergence of foundation models like the Segment Anything Model (SAM) and its successor, SAM2, has introduced a paradigm shift, enabling prompt-driven segmentation with strong generalization capabilities. Building upon these advances, this survey provides a comprehensive review of SAM/SAM2-based methods for VOST, structured along three temporal dimensions: past, present, and future. We examine strategies for retaining and updating historical information (past), approaches for extracting and optimizing discriminative features from the current frame (present), and motion prediction and trajectory estimation mechanisms for anticipating object dynamics in subsequent frames (future). In doing so, we highlight the evolution from early memory-based architectures to the streaming memory and real-time segmentation capabilities of SAM2. We also discuss recent innovations such as motion-aware memory selection and trajectory-guided prompting, which aim to enhance both accuracy and efficiency. Finally, we identify remaining challenges including memory redundancy, error accumulation, and prompt inefficiency, and suggest promising directions for future research. This survey offers a timely and structured overview of the field, aiming to guide researchers and practitioners in advancing the state of VOST through the lens of foundation models.
>
---
#### [new 017] DACA-Net: A Degradation-Aware Conditional Diffusion Network for Underwater Image Enhancement
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于水下图像增强任务，旨在解决水下图像因光学效应导致的颜色失真、低可视性等问题。作者提出DACA-Net，通过预测图像退化程度并结合扩散模型与物理先验，实现自适应图像恢复，提升了色彩保真度与结构细节。**

- **链接: [http://arxiv.org/pdf/2507.22501v1](http://arxiv.org/pdf/2507.22501v1)**

> **作者:** Chang Huang; Jiahang Cao; Jun Ma; Kieren Yu; Cong Li; Huayong Yang; Kaishun Wu
>
> **备注:** accepted by ACM MM 2025
>
> **摘要:** Underwater images typically suffer from severe colour distortions, low visibility, and reduced structural clarity due to complex optical effects such as scattering and absorption, which greatly degrade their visual quality and limit the performance of downstream visual perception tasks. Existing enhancement methods often struggle to adaptively handle diverse degradation conditions and fail to leverage underwater-specific physical priors effectively. In this paper, we propose a degradation-aware conditional diffusion model to enhance underwater images adaptively and robustly. Given a degraded underwater image as input, we first predict its degradation level using a lightweight dual-stream convolutional network, generating a continuous degradation score as semantic guidance. Based on this score, we introduce a novel conditional diffusion-based restoration network with a Swin UNet backbone, enabling adaptive noise scheduling and hierarchical feature refinement. To incorporate underwater-specific physical priors, we further propose a degradation-guided adaptive feature fusion module and a hybrid loss function that combines perceptual consistency, histogram matching, and feature-level contrast. Comprehensive experiments on benchmark datasets demonstrate that our method effectively restores underwater images with superior colour fidelity, perceptual quality, and structural details. Compared with SOTA approaches, our framework achieves significant improvements in both quantitative metrics and qualitative visual assessments.
>
---
#### [new 018] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation
- **分类: cs.CV**

- **简介: 该论文属于多模态任务，旨在解决指向性音视频分割（RAVS）中多模态信息融合与深层理解的问题。作者提出了新数据集OmniAVS，包含2,098个视频和59,458个多模态表达，并设计了模型OISA，利用MLLM进行推理与细粒度理解，提升了分割效果。**

- **链接: [http://arxiv.org/pdf/2507.22886v1](http://arxiv.org/pdf/2507.22886v1)**

> **作者:** Kaining Ying; Henghui Ding; Guanquan Jie; Yu-Gang Jiang
>
> **备注:** ICCV 2025, Project Page: https://henghuiding.com/OmniAVS/
>
> **摘要:** Referring audio-visual segmentation (RAVS) has recently seen significant advancements, yet challenges remain in integrating multimodal information and deeply understanding and reasoning about audiovisual content. To extend the boundaries of RAVS and facilitate future research in this field, we propose Omnimodal Referring Audio-Visual Segmentation (OmniAVS), a new dataset containing 2,098 videos and 59,458 multimodal referring expressions. OmniAVS stands out with three key innovations: (1) 8 types of multimodal expressions that flexibly combine text, speech, sound, and visual cues; (2) an emphasis on understanding audio content beyond just detecting their presence; and (3) the inclusion of complex reasoning and world knowledge in expressions. Furthermore, we introduce Omnimodal Instructed Segmentation Assistant (OISA), to address the challenges of multimodal reasoning and fine-grained understanding of audiovisual content in OmniAVS. OISA uses MLLM to comprehend complex cues and perform reasoning-based segmentation. Extensive experiments show that OISA outperforms existing methods on OmniAVS and achieves competitive results on other related tasks.
>
---
#### [new 019] ScreenCoder: Advancing Visual-to-Code Generation for Front-End Automation via Modular Multimodal Agents
- **分类: cs.CV**

- **简介: 该论文属于前端自动化任务，旨在解决从UI设计到前端代码的自动转换问题。现有方法依赖文本提示，难以捕捉视觉意图。论文提出ScreenCoder框架，分三阶段（定位、规划、生成）实现视觉到代码的生成，并构建大规模数据引擎提升效果，显著优化布局、结构与代码质量。**

- **链接: [http://arxiv.org/pdf/2507.22827v1](http://arxiv.org/pdf/2507.22827v1)**

> **作者:** Yilei Jiang; Yaozhi Zheng; Yuxuan Wan; Jiaming Han; Qunzhong Wang; Michael R. Lyu; Xiangyu Yue
>
> **摘要:** Automating the transformation of user interface (UI) designs into front-end code holds significant promise for accelerating software development and democratizing design workflows. While recent large language models (LLMs) have demonstrated progress in text-to-code generation, many existing approaches rely solely on natural language prompts, limiting their effectiveness in capturing spatial layout and visual design intent. In contrast, UI development in practice is inherently multimodal, often starting from visual sketches or mockups. To address this gap, we introduce a modular multi-agent framework that performs UI-to-code generation in three interpretable stages: grounding, planning, and generation. The grounding agent uses a vision-language model to detect and label UI components, the planning agent constructs a hierarchical layout using front-end engineering priors, and the generation agent produces HTML/CSS code via adaptive prompt-based synthesis. This design improves robustness, interpretability, and fidelity over end-to-end black-box methods. Furthermore, we extend the framework into a scalable data engine that automatically produces large-scale image-code pairs. Using these synthetic examples, we fine-tune and reinforce an open-source VLM, yielding notable gains in UI understanding and code quality. Extensive experiments demonstrate that our approach achieves state-of-the-art performance in layout accuracy, structural coherence, and code correctness. Our code is made publicly available at https://github.com/leigest519/ScreenCoder.
>
---
#### [new 020] HOLA: Enhancing Audio-visual Deepfake Detection via Hierarchical Contextual Aggregations and Efficient Pre-training
- **分类: cs.CV**

- **简介: 该论文属于视频级多模态深度伪造检测任务，旨在解决生成AI带来的深伪视频检测难题。论文提出了HOLA框架，通过分层上下文聚合、高效预训练和跨模态学习，提升检测性能。实验表明其方法在挑战赛中排名第一，AUC超越第二名0.0476。**

- **链接: [http://arxiv.org/pdf/2507.22781v1](http://arxiv.org/pdf/2507.22781v1)**

> **作者:** Xuecheng Wu; Danlei Huang; Heli Sun; Xinyi Yin; Yifan Wang; Hao Wang; Jia Zhang; Fei Wang; Peihao Guo; Suyu Xing; Junxiao Xue; Liang He
>
> **摘要:** Advances in Generative AI have made video-level deepfake detection increasingly challenging, exposing the limitations of current detection techniques. In this paper, we present HOLA, our solution to the Video-Level Deepfake Detection track of 2025 1M-Deepfakes Detection Challenge. Inspired by the success of large-scale pre-training in the general domain, we first scale audio-visual self-supervised pre-training in the multimodal video-level deepfake detection, which leverages our self-built dataset of 1.81M samples, thereby leading to a unified two-stage framework. To be specific, HOLA features an iterative-aware cross-modal learning module for selective audio-visual interactions, hierarchical contextual modeling with gated aggregations under the local-global perspective, and a pyramid-like refiner for scale-aware cross-grained semantic enhancements. Moreover, we propose the pseudo supervised singal injection strategy to further boost model performance. Extensive experiments across expert models and MLLMs impressivly demonstrate the effectiveness of our proposed HOLA. We also conduct a series of ablation studies to explore the crucial design factors of our introduced components. Remarkably, our HOLA ranks 1st, outperforming the second by 0.0476 AUC on the TestA set.
>
---
#### [new 021] VL-Cogito: Progressive Curriculum Reinforcement Learning for Advanced Multimodal Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态推理任务，旨在解决现有模型在不同领域和难度任务中表现不稳定的问题。作者提出了VL-Cogito模型，基于多阶段渐进课程强化学习（PCuRL）框架，引入难度软加权和动态长度奖励机制，提升模型在数学、科学、逻辑等多模态任务中的推理能力与效率。**

- **链接: [http://arxiv.org/pdf/2507.22607v1](http://arxiv.org/pdf/2507.22607v1)**

> **作者:** Ruifeng Yuan; Chenghao Xiao; Sicong Leng; Jianyu Wang; Long Li; Weiwen Xu; Hou Pong Chan; Deli Zhao; Tingyang Xu; Zhongyu Wei; Hao Zhang; Yu Rong
>
> **备注:** 21 pages, 5 figures, 6 tables. Work in progress
>
> **摘要:** Reinforcement learning has proven its effectiveness in enhancing the reasoning capabilities of large language models. Recent research efforts have progressively extended this paradigm to multimodal reasoning tasks. Due to the inherent complexity and diversity of multimodal tasks, especially in semantic content and problem formulations, existing models often exhibit unstable performance across various domains and difficulty levels. To address these limitations, we propose VL-Cogito, an advanced multimodal reasoning model trained via a novel multi-stage Progressive Curriculum Reinforcement Learning (PCuRL) framework. PCuRL systematically guides the model through tasks of gradually increasing difficulty, substantially improving its reasoning abilities across diverse multimodal contexts. The framework introduces two key innovations: (1) an online difficulty soft weighting mechanism, dynamically adjusting training difficulty across successive RL training stages; and (2) a dynamic length reward mechanism, which encourages the model to adaptively regulate its reasoning path length according to task complexity, thus balancing reasoning efficiency with correctness. Experimental evaluations demonstrate that VL-Cogito consistently matches or surpasses existing reasoning-oriented models across mainstream multimodal benchmarks spanning mathematics, science, logic, and general understanding, validating the effectiveness of our approach.
>
---
#### [new 022] Subtyping Breast Lesions via Generative Augmentation based Long-tailed Recognition in Ultrasound
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决乳腺病变亚型分类中数据分布长尾、样本不均衡的问题。作者提出了一种基于生成增强的双阶段框架，结合强化学习自适应采样和可控生成网络，提升分类性能并保持类特征。**

- **链接: [http://arxiv.org/pdf/2507.22568v1](http://arxiv.org/pdf/2507.22568v1)**

> **作者:** Shijing Chen; Xinrui Zhou; Yuhao Wang; Yuhao Huang; Ao Chang; Dong Ni; Ruobing Huang
>
> **备注:** MICCAI2025 Early Accept. 11 pages, 3 figures, 2 tables
>
> **摘要:** Accurate identification of breast lesion subtypes can facilitate personalized treatment and interventions. Ultrasound (US), as a safe and accessible imaging modality, is extensively employed in breast abnormality screening and diagnosis. However, the incidence of different subtypes exhibits a skewed long-tailed distribution, posing significant challenges for automated recognition. Generative augmentation provides a promising solution to rectify data distribution. Inspired by this, we propose a dual-phase framework for long-tailed classification that mitigates distributional bias through high-fidelity data synthesis while avoiding overuse that corrupts holistic performance. The framework incorporates a reinforcement learning-driven adaptive sampler, dynamically calibrating synthetic-real data ratios by training a strategic multi-agent to compensate for scarcities of real data while ensuring stable discriminative capability. Furthermore, our class-controllable synthetic network integrates a sketch-grounded perception branch that harnesses anatomical priors to maintain distinctive class features while enabling annotation-free inference. Extensive experiments on an in-house long-tailed and a public imbalanced breast US datasets demonstrate that our method achieves promising performance compared to state-of-the-art approaches. More synthetic images can be found at https://github.com/Stinalalala/Breast-LT-GenAug.
>
---
#### [new 023] TopoLiDM: Topology-Aware LiDAR Diffusion Models for Interpretable and Realistic LiDAR Point Cloud Generation
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于自动驾驶中的LiDAR点云生成任务，旨在解决现有方法在几何真实性和全局拓扑一致性上的不足。作者提出TopoLiDM，结合图神经网络与扩散模型，并引入拓扑正则化，以提升生成质量与可解释性。**

- **链接: [http://arxiv.org/pdf/2507.22454v1](http://arxiv.org/pdf/2507.22454v1)**

> **作者:** Jiuming Liu; Zheng Huang; Mengmeng Liu; Tianchen Deng; Francesco Nex; Hao Cheng; Hesheng Wang
>
> **备注:** Accepted by IROS 2025. Code:https://github.com/IRMVLab/TopoLiDM
>
> **摘要:** LiDAR scene generation is critical for mitigating real-world LiDAR data collection costs and enhancing the robustness of downstream perception tasks in autonomous driving. However, existing methods commonly struggle to capture geometric realism and global topological consistency. Recent LiDAR Diffusion Models (LiDMs) predominantly embed LiDAR points into the latent space for improved generation efficiency, which limits their interpretable ability to model detailed geometric structures and preserve global topological consistency. To address these challenges, we propose TopoLiDM, a novel framework that integrates graph neural networks (GNNs) with diffusion models under topological regularization for high-fidelity LiDAR generation. Our approach first trains a topological-preserving VAE to extract latent graph representations by graph construction and multiple graph convolutional layers. Then we freeze the VAE and generate novel latent topological graphs through the latent diffusion models. We also introduce 0-dimensional persistent homology (PH) constraints, ensuring the generated LiDAR scenes adhere to real-world global topological structures. Extensive experiments on the KITTI-360 dataset demonstrate TopoLiDM's superiority over state-of-the-art methods, achieving improvements of 22.6% lower Frechet Range Image Distance (FRID) and 9.2% lower Minimum Matching Distance (MMD). Notably, our model also enables fast generation speed with an average inference time of 1.68 samples/s, showcasing its scalability for real-world applications. We will release the related codes at https://github.com/IRMVLab/TopoLiDM.
>
---
#### [new 024] Bi-Level Optimization for Self-Supervised AI-Generated Face Detection
- **分类: cs.CV**

- **简介: 该论文属于AI生成人脸检测任务，旨在解决现有方法依赖特定生成器数据、泛化能力差的问题。作者提出一种基于双层优化的自监督学习方法，通过预训练视觉编码器并优化预任务权重，提升对AI生成人脸的检测性能，实验表明其方法在多种设置下优于现有技术。**

- **链接: [http://arxiv.org/pdf/2507.22824v1](http://arxiv.org/pdf/2507.22824v1)**

> **作者:** Mian Zou; Nan Zhong; Baosheng Yu; Yibing Zhan; Kede Ma
>
> **摘要:** AI-generated face detectors trained via supervised learning typically rely on synthesized images from specific generators, limiting their generalization to emerging generative techniques. To overcome this limitation, we introduce a self-supervised method based on bi-level optimization. In the inner loop, we pretrain a vision encoder only on photographic face images using a set of linearly weighted pretext tasks: classification of categorical exchangeable image file format (EXIF) tags, ranking of ordinal EXIF tags, and detection of artificial face manipulations. The outer loop then optimizes the relative weights of these pretext tasks to enhance the coarse-grained detection of manipulated faces, serving as a proxy task for identifying AI-generated faces. In doing so, it aligns self-supervised learning more closely with the ultimate goal of AI-generated face detection. Once pretrained, the encoder remains fixed, and AI-generated faces are detected either as anomalies under a Gaussian mixture model fitted to photographic face features or by a lightweight two-layer perceptron serving as a binary classifier. Extensive experiments demonstrate that our detectors significantly outperform existing approaches in both one-class and binary classification settings, exhibiting strong generalization to unseen generators.
>
---
#### [new 025] A Linear N-Point Solver for Structure and Motion from Asynchronous Tracks
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉中的结构与运动估计任务，旨在解决异步点对应关系下的线性结构与运动恢复问题。针对滚动快门、事件相机等导致的非同步数据，提出了一种统一的线性N点求解方法，能有效估计线速度和3D点，并在多种传感器数据上验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2507.22733v1](http://arxiv.org/pdf/2507.22733v1)**

> **作者:** Hang Su; Yunlong Feng; Daniel Gehrig; Panfeng Jiang; Ling Gao; Xavier Lagorce; Laurent Kneip
>
> **摘要:** Structure and continuous motion estimation from point correspondences is a fundamental problem in computer vision that has been powered by well-known algorithms such as the familiar 5-point or 8-point algorithm. However, despite their acclaim, these algorithms are limited to processing point correspondences originating from a pair of views each one representing an instantaneous capture of the scene. Yet, in the case of rolling shutter cameras, or more recently, event cameras, this synchronization breaks down. In this work, we present a unified approach for structure and linear motion estimation from 2D point correspondences with arbitrary timestamps, from an arbitrary set of views. By formulating the problem in terms of first-order dynamics and leveraging a constant velocity motion model, we derive a novel, linear point incidence relation allowing for the efficient recovery of both linear velocity and 3D points with predictable degeneracies and solution multiplicities. Owing to its general formulation, it can handle correspondences from a wide range of sensing modalities such as global shutter, rolling shutter, and event cameras, and can even combine correspondences from different collocated sensors. We validate the effectiveness of our solver on both simulated and real-world data, where we show consistent improvement across all modalities when compared to recent approaches. We believe our work opens the door to efficient structure and motion estimation from asynchronous data. Code can be found at https://github.com/suhang99/AsyncTrack-Motion-Solver.
>
---
#### [new 026] Shallow Features Matter: Hierarchical Memory with Heterogeneous Interaction for Unsupervised Video Object Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于无监督视频目标分割（UVOS）任务，旨在解决缺乏像素级先验知识导致分割精度不足的问题。论文提出了一种分层记忆架构与异构交互机制，融合浅层像素信息与高层语义特征，实现了更精确的视频目标分割，达到当前最优性能。**

- **链接: [http://arxiv.org/pdf/2507.22465v1](http://arxiv.org/pdf/2507.22465v1)**

> **作者:** Zheng Xiangyu; He Songcheng; Li Wanyun; Li Xiaoqiang; Zhang Wei
>
> **备注:** Accepted to ACM MM'25: The 33rd ACM International Conference on Multimedia Proceedings
>
> **摘要:** Unsupervised Video Object Segmentation (UVOS) aims to predict pixel-level masks for the most salient objects in videos without any prior annotations. While memory mechanisms have been proven critical in various video segmentation paradigms, their application in UVOS yield only marginal performance gains despite sophisticated design. Our analysis reveals a simple but fundamental flaw in existing methods: over-reliance on memorizing high-level semantic features. UVOS inherently suffers from the deficiency of lacking fine-grained information due to the absence of pixel-level prior knowledge. Consequently, memory design relying solely on high-level features, which predominantly capture abstract semantic cues, is insufficient to generate precise predictions. To resolve this fundamental issue, we propose a novel hierarchical memory architecture to incorporate both shallow- and high-level features for memory, which leverages the complementary benefits of pixel and semantic information. Furthermore, to balance the simultaneous utilization of the pixel and semantic memory features, we propose a heterogeneous interaction mechanism to perform pixel-semantic mutual interactions, which explicitly considers their inherent feature discrepancies. Through the design of Pixel-guided Local Alignment Module (PLAM) and Semantic-guided Global Integration Module (SGIM), we achieve delicate integration of the fine-grained details in shallow-level memory and the semantic representations in high-level memory. Our Hierarchical Memory with Heterogeneous Interaction Network (HMHI-Net) consistently achieves state-of-the-art performance across all UVOS and video saliency detection benchmarks. Moreover, HMHI-Net consistently exhibits high performance across different backbones, further demonstrating its superiority and robustness. Project page: https://github.com/ZhengxyFlow/HMHI-Net .
>
---
#### [new 027] LAMA-Net: A Convergent Network Architecture for Dual-Domain Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于图像重建任务，旨在解决稀疏视角CT成像中的重建问题。作者提出了LAMA-Net，基于可学习的交替最小化算法，并提供其收敛性证明。进一步设计了iLAMA-Net以提升性能。方法结合图像和测量域信息，实现稳定、鲁棒的重建，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.22316v1](http://arxiv.org/pdf/2507.22316v1)**

> **作者:** Chi Ding; Qingchao Zhang; Ge Wang; Xiaojing Ye; Yunmei Chen
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2410.21111
>
> **摘要:** We propose a learnable variational model that learns the features and leverages complementary information from both image and measurement domains for image reconstruction. In particular, we introduce a learned alternating minimization algorithm (LAMA) from our prior work, which tackles two-block nonconvex and nonsmooth optimization problems by incorporating a residual learning architecture in a proximal alternating framework. In this work, our goal is to provide a complete and rigorous convergence proof of LAMA and show that all accumulation points of a specified subsequence of LAMA must be Clarke stationary points of the problem. LAMA directly yields a highly interpretable neural network architecture called LAMA-Net. Notably, in addition to the results shown in our prior work, we demonstrate that the convergence property of LAMA yields outstanding stability and robustness of LAMA-Net in this work. We also show that the performance of LAMA-Net can be further improved by integrating a properly designed network that generates suitable initials, which we call iLAMA-Net. To evaluate LAMA-Net/iLAMA-Net, we conduct several experiments and compare them with several state-of-the-art methods on popular benchmark datasets for Sparse-View Computed Tomography.
>
---
#### [new 028] Efficient Spatial-Temporal Modeling for Real-Time Video Analysis: A Unified Framework for Action Recognition and Object Tracking
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉任务，旨在解决实时视频分析中时空信息处理效率与准确率不平衡的问题。作者提出了一种统一框架，结合并行序列建模与分层注意力机制，实现动作识别与目标跟踪的联合建模。实验表明其方法在多个数据集上性能优越，推理速度更快。**

- **链接: [http://arxiv.org/pdf/2507.22421v1](http://arxiv.org/pdf/2507.22421v1)**

> **作者:** Shahla John
>
> **摘要:** Real-time video analysis remains a challenging problem in computer vision, requiring efficient processing of both spatial and temporal information while maintaining computational efficiency. Existing approaches often struggle to balance accuracy and speed, particularly in resource-constrained environments. In this work, we present a unified framework that leverages advanced spatial-temporal modeling techniques for simultaneous action recognition and object tracking. Our approach builds upon recent advances in parallel sequence modeling and introduces a novel hierarchical attention mechanism that adaptively focuses on relevant spatial regions across temporal sequences. We demonstrate that our method achieves state-of-the-art performance on standard benchmarks while maintaining real-time inference speeds. Extensive experiments on UCF-101, HMDB-51, and MOT17 datasets show improvements of 3.2% in action recognition accuracy and 2.8% in tracking precision compared to existing methods, with 40% faster inference time.
>
---
#### [new 029] DeltaVLM: Interactive Remote Sensing Image Change Analysis via Instruction-guided Difference Perception
- **分类: cs.CV; I.2.10; I.4.8; I.5.4**

- **简介: 该论文属于遥感图像变化分析任务，旨在解决现有方法无法支持交互式查询的问题。作者提出了DeltaVLM模型与ChangeChat-105k数据集，实现指令引导的多轮变化感知与分析，提升交互性与准确性。**

- **链接: [http://arxiv.org/pdf/2507.22346v1](http://arxiv.org/pdf/2507.22346v1)**

> **作者:** Pei Deng; Wenqian Zhou; Hanlin Wu
>
> **备注:** 12 pages, 5 figures. Submitted to IEEE Transactions on Geoscience and Remote Sensing (TGRS). Code and dataset are available at https://github.com/hanlinwu/DeltaVLM
>
> **摘要:** Accurate interpretation of land-cover changes in multi-temporal satellite imagery is critical for real-world scenarios. However, existing methods typically provide only one-shot change masks or static captions, limiting their ability to support interactive, query-driven analysis. In this work, we introduce remote sensing image change analysis (RSICA) as a new paradigm that combines the strengths of change detection and visual question answering to enable multi-turn, instruction-guided exploration of changes in bi-temporal remote sensing images. To support this task, we construct ChangeChat-105k, a large-scale instruction-following dataset, generated through a hybrid rule-based and GPT-assisted process, covering six interaction types: change captioning, classification, quantification, localization, open-ended question answering, and multi-turn dialogues. Building on this dataset, we propose DeltaVLM, an end-to-end architecture tailored for interactive RSICA. DeltaVLM features three innovations: (1) a fine-tuned bi-temporal vision encoder to capture temporal differences; (2) a visual difference perception module with a cross-semantic relation measuring (CSRM) mechanism to interpret changes; and (3) an instruction-guided Q-former to effectively extract query-relevant difference information from visual changes, aligning them with textual instructions. We train DeltaVLM on ChangeChat-105k using a frozen large language model, adapting only the vision and alignment modules to optimize efficiency. Extensive experiments and ablation studies demonstrate that DeltaVLM achieves state-of-the-art performance on both single-turn captioning and multi-turn interactive change analysis, outperforming existing multimodal large language models and remote sensing vision-language models. Code, dataset and pre-trained weights are available at https://github.com/hanlinwu/DeltaVLM.
>
---
#### [new 030] Gems: Group Emotion Profiling Through Multimodal Situational Understanding
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多模态情感分析任务，旨在通过情境理解进行群体情绪建模。它提出GEMS框架，结合Swin-Transformer与S3Attention，预测个体、群体及事件层面的情绪。论文扩展了VGAF数据集，提供更细致的群体情感分析，并验证了模型在新基准上的有效性。**

- **链接: [http://arxiv.org/pdf/2507.22393v1](http://arxiv.org/pdf/2507.22393v1)**

> **作者:** Anubhav Kataria; Surbhi Madan; Shreya Ghosh; Tom Gedeon; Abhinav Dhall
>
> **摘要:** Understanding individual, group and event level emotions along with contextual information is crucial for analyzing a multi-person social situation. To achieve this, we frame emotion comprehension as the task of predicting fine-grained individual emotion to coarse grained group and event level emotion. We introduce GEMS that leverages a multimodal swin-transformer and S3Attention based architecture, which processes an input scene, group members, and context information to generate joint predictions. Existing multi-person emotion related benchmarks mainly focus on atomic interactions primarily based on emotion perception over time and group level. To this end, we extend and propose VGAF-GEMS to provide more fine grained and holistic analysis on top of existing group level annotation of VGAF dataset. GEMS aims to predict basic discrete and continuous emotions (including valence and arousal) as well as individual, group and event level perceived emotions. Our benchmarking effort links individual, group and situational emotional responses holistically. The quantitative and qualitative comparisons with adapted state-of-the-art models demonstrate the effectiveness of GEMS framework on VGAF-GEMS benchmarking. We believe that it will pave the way of further research. The code and data is available at: https://github.com/katariaak579/GEMS
>
---
#### [new 031] Robust Adverse Weather Removal via Spectral-based Spatial Grouping
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像恢复任务，旨在解决多种恶劣天气导致的复杂图像退化问题。现有方法难以应对多样化的局部退化，为此作者提出了SSGformer模型。该模型通过频谱分解和分组注意力机制，结合空间和通道注意力，有效提升了多天气图像复原的性能。**

- **链接: [http://arxiv.org/pdf/2507.22498v1](http://arxiv.org/pdf/2507.22498v1)**

> **作者:** Yuhwan Jeong; Yunseo Yang; Youngjo Yoon; Kuk-Jin Yoon
>
> **备注:** accepted by ICCV25
>
> **摘要:** Adverse weather conditions cause diverse and complex degradation patterns, driving the development of All-in-One (AiO) models. However, recent AiO solutions still struggle to capture diverse degradations, since global filtering methods like direct operations on the frequency domain fail to handle highly variable and localized distortions. To address these issue, we propose Spectral-based Spatial Grouping Transformer (SSGformer), a novel approach that leverages spectral decomposition and group-wise attention for multi-weather image restoration. SSGformer decomposes images into high-frequency edge features using conventional edge detection and low-frequency information via Singular Value Decomposition. We utilize multi-head linear attention to effectively model the relationship between these features. The fused features are integrated with the input to generate a grouping-mask that clusters regions based on the spatial similarity and image texture. To fully leverage this mask, we introduce a group-wise attention mechanism, enabling robust adverse weather removal and ensuring consistent performance across diverse weather conditions. We also propose a Spatial Grouping Transformer Block that uses both channel attention and spatial attention, effectively balancing feature-wise relationships and spatial dependencies. Extensive experiments show the superiority of our approach, validating its effectiveness in handling the varied and intricate adverse weather degradations.
>
---
#### [new 032] COOkeD: Ensemble-based OOD detection in the era of zero-shot CLIP
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像识别中的分布外（OOD）检测任务，旨在解决测试时未知类别检测问题。现有方法受限于分类器性能且分裂为监督与零样本设置。作者提出COOkeD，通过集成闭集分类器、CLIP零样本分类器及线性探针分类器，实现更优OOD检测。方法模块化、开销小，并在多种基准和挑战场景下表现优异。**

- **链接: [http://arxiv.org/pdf/2507.22576v1](http://arxiv.org/pdf/2507.22576v1)**

> **作者:** Galadrielle Humblot-Renaux; Gianni Franchi; Sergio Escalera; Thomas B. Moeslund
>
> **备注:** accepted at ICCVW'25 - Systematic Trust in AI Models: Ensuring Fairness, Reliability, Explainability, and Accountability in Machine Learning Frameworks
>
> **摘要:** Out-of-distribution (OOD) detection is an important building block in trustworthy image recognition systems as unknown classes may arise at test-time. OOD detection methods typically revolve around a single classifier, leading to a split in the research field between the classical supervised setting (e.g. ResNet18 classifier trained on CIFAR100) vs. the zero-shot setting (class names fed as prompts to CLIP). In both cases, an overarching challenge is that the OOD detection performance is implicitly constrained by the classifier's capabilities on in-distribution (ID) data. In this work, we show that given a little open-mindedness from both ends, remarkable OOD detection can be achieved by instead creating a heterogeneous ensemble - COOkeD combines the predictions of a closed-world classifier trained end-to-end on a specific dataset, a zero-shot CLIP classifier, and a linear probe classifier trained on CLIP image features. While bulky at first sight, this approach is modular, post-hoc and leverages the availability of pre-trained VLMs, thus introduces little overhead compared to training a single standard classifier. We evaluate COOkeD on popular CIFAR100 and ImageNet benchmarks, but also consider more challenging, realistic settings ranging from training-time label noise, to test-time covariate shift, to zero-shot shift which has been previously overlooked. Despite its simplicity, COOkeD achieves state-of-the-art performance and greater robustness compared to both classical and CLIP-based OOD detection methods. Code is available at https://github.com/glhr/COOkeD
>
---
#### [new 033] Viser: Imperative, Web-based 3D Visualization in Python
- **分类: cs.CV; cs.RO**

- **简介: 论文提出Viser，一个面向Python的3D可视化库，用于计算机视觉与机器人任务。要解决的问题是现有工具复杂、难扩展。Viser提供易用、可组合的3D与2D界面元素，支持快速搭建可视化应用。核心工作包括设计命令式API与基于网页的查看器，提升兼容性与交互体验。**

- **链接: [http://arxiv.org/pdf/2507.22885v1](http://arxiv.org/pdf/2507.22885v1)**

> **作者:** Brent Yi; Chung Min Kim; Justin Kerr; Gina Wu; Rebecca Feng; Anthony Zhang; Jonas Kulhanek; Hongsuk Choi; Yi Ma; Matthew Tancik; Angjoo Kanazawa
>
> **备注:** Code and docs: https://viser.studio
>
> **摘要:** We present Viser, a 3D visualization library for computer vision and robotics. Viser aims to bring easy and extensible 3D visualization to Python: we provide a comprehensive set of 3D scene and 2D GUI primitives, which can be used independently with minimal setup or composed to build specialized interfaces. This technical report describes Viser's features, interface, and implementation. Key design choices include an imperative-style API and a web-based viewer, which improve compatibility with modern programming patterns and workflows.
>
---
#### [new 034] AlphaDent: A dataset for automated tooth pathology detection
- **分类: cs.CV; cs.LG**

- **简介: 论文提出AlphaDent数据集，用于牙齿病理自动检测。任务为实例分割，旨在解决牙齿病变识别问题。工作包括构建含1200余张图像的数据集，标注9类病变，并训练神经网络验证分割效果，结果表现优异。数据集与代码开源。**

- **链接: [http://arxiv.org/pdf/2507.22512v1](http://arxiv.org/pdf/2507.22512v1)**

> **作者:** Evgeniy I. Sosnin; Yuriy L. Vasilev; Roman A. Solovyev; Aleksandr L. Stempkovskiy; Dmitry V. Telpukhov; Artem A. Vasilev; Aleksandr A. Amerikanov; Aleksandr Y. Romanov
>
> **摘要:** In this article, we present a new unique dataset for dental research - AlphaDent. This dataset is based on the DSLR camera photographs of the teeth of 295 patients and contains over 1200 images. The dataset is labeled for solving the instance segmentation problem and is divided into 9 classes. The article provides a detailed description of the dataset and the labeling format. The article also provides the details of the experiment on neural network training for the Instance Segmentation problem using this dataset. The results obtained show high quality of predictions. The dataset is published under an open license; and the training/inference code and model weights are also available under open licenses.
>
---
#### [new 035] From Sharp to Blur: Unsupervised Domain Adaptation for 2D Human Pose Estimation Under Extreme Motion Blur Using Event Cameras
- **分类: cs.CV**

- **简介: 该论文属于2D人体姿态估计任务，旨在解决极端运动模糊下模型性能下降的问题。利用事件相机高时间分辨率和抗模糊特性，提出无监督域适应方法，通过事件增强生成模糊图像，并设计学生-教师框架优化伪标签，实现无需目标域标注的鲁棒姿态估计。**

- **链接: [http://arxiv.org/pdf/2507.22438v1](http://arxiv.org/pdf/2507.22438v1)**

> **作者:** Youngho Kim; Hoonhee Cho; Kuk-Jin Yoon
>
> **摘要:** Human pose estimation is critical for applications such as rehabilitation, sports analytics, and AR/VR systems. However, rapid motion and low-light conditions often introduce motion blur, significantly degrading pose estimation due to the domain gap between sharp and blurred images. Most datasets assume stable conditions, making models trained on sharp images struggle in blurred environments. To address this, we introduce a novel domain adaptation approach that leverages event cameras, which capture high temporal resolution motion data and are inherently robust to motion blur. Using event-based augmentation, we generate motion-aware blurred images, effectively bridging the domain gap between sharp and blurred domains without requiring paired annotations. Additionally, we develop a student-teacher framework that iteratively refines pseudo-labels, leveraging mutual uncertainty masking to eliminate incorrect labels and enable more effective learning. Experimental results demonstrate that our approach outperforms conventional domain-adaptive human pose estimation methods, achieving robust pose estimation under motion blur without requiring annotations in the target domain. Our findings highlight the potential of event cameras as a scalable and effective solution for domain adaptation in real-world motion blur environments. Our project codes are available at https://github.com/kmax2001/EvSharp2Blur.
>
---
#### [new 036] Zero-Shot Image Anomaly Detection Using Generative Foundation Models
- **分类: cs.CV**

- **简介: 该论文属于图像异常检测任务，旨在解决开放世界环境下识别分布外（OOD）输入的问题。论文提出利用去噪扩散模型（DDMs）的去噪轨迹和Stein得分误差，结合SSIM指标，实现无需针对每个目标数据集重新训练的零样本异常检测方法。**

- **链接: [http://arxiv.org/pdf/2507.22692v1](http://arxiv.org/pdf/2507.22692v1)**

> **作者:** Lemar Abdi; Amaan Valiuddin; Francisco Caetano; Christiaan Viviers; Fons van der Sommen
>
> **备注:** Accepted at the workshop of Anomaly Detection with Foundation Models, ICCV 2025
>
> **摘要:** Detecting out-of-distribution (OOD) inputs is pivotal for deploying safe vision systems in open-world environments. We revisit diffusion models, not as generators, but as universal perceptual templates for OOD detection. This research explores the use of score-based generative models as foundational tools for semantic anomaly detection across unseen datasets. Specifically, we leverage the denoising trajectories of Denoising Diffusion Models (DDMs) as a rich source of texture and semantic information. By analyzing Stein score errors, amplified through the Structural Similarity Index Metric (SSIM), we introduce a novel method for identifying anomalous samples without requiring re-training on each target dataset. Our approach improves over state-of-the-art and relies on training a single model on one dataset -- CelebA -- which we find to be an effective base distribution, even outperforming more commonly used datasets like ImageNet in several settings. Experimental results show near-perfect performance on some benchmarks, with notable headroom on others, highlighting both the strength and future potential of generative foundation models in anomaly detection.
>
---
#### [new 037] LCS: An AI-based Low-Complexity Scaler for Power-Efficient Super-Resolution of Game Content
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像超分辨率任务，旨在通过轻量AI模型减轻GPU渲染负载。论文提出LCS模型，利用对抗训练和模型压缩技术，在低功耗设备上实现高质量游戏图像放大，优于现有硬件方案。**

- **链接: [http://arxiv.org/pdf/2507.22873v1](http://arxiv.org/pdf/2507.22873v1)**

> **作者:** Simon Pochinda; Momen K. Tageldeen; Mark Thompson; Tony Rinaldi; Troy Giorshev; Keith Lee; Jie Zhou; Frederick Walls
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** The increasing complexity of content rendering in modern games has led to a problematic growth in the workload of the GPU. In this paper, we propose an AI-based low-complexity scaler (LCS) inspired by state-of-the-art efficient super-resolution (ESR) models which could offload the workload on the GPU to a low-power device such as a neural processing unit (NPU). The LCS is trained on GameIR image pairs natively rendered at low and high resolution. We utilize adversarial training to encourage reconstruction of perceptually important details, and apply reparameterization and quantization techniques to reduce model complexity and size. In our comparative analysis we evaluate the LCS alongside the publicly available AMD hardware-based Edge Adaptive Scaling Function (EASF) and AMD FidelityFX Super Resolution 1 (FSR1) on five different metrics, and find that the LCS achieves better perceptual quality, demonstrating the potential of ESR models for upscaling on resource-constrained devices.
>
---
#### [new 038] RainbowPrompt: Diversity-Enhanced Prompt-Evolving for Continual Learning
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于持续学习任务，旨在解决提示学习中任务知识融合不足、多样性受限的问题。作者提出RainbowPrompt方法，通过动态聚合多样化的任务专属提示，并引入可学习门控机制优化提示演化过程，从而提升新任务学习效果。实验验证了其在图像分类和视频动作识别中的有效性。**

- **链接: [http://arxiv.org/pdf/2507.22553v1](http://arxiv.org/pdf/2507.22553v1)**

> **作者:** Kiseong Hong; Gyeong-hyeon Kim; Eunwoo Kim
>
> **备注:** Accepted by the 2025 IEEE/CVF International Conference on Computer Vision (ICCV 2025)
>
> **摘要:** Prompt-based continual learning provides a rehearsal-free solution by tuning small sets of parameters while keeping pre-trained models frozen. To meet the complex demands of sequential tasks, it is crucial to integrate task-specific knowledge within prompts effectively. However, existing works rely on either fixed learned prompts (i.e., prompts whose representations remain unchanged during new task learning) or on prompts generated from an entangled task-shared space, limiting the representational diversity of the integrated prompt. To address this issue, we propose a novel prompt-evolving mechanism to adaptively aggregate base prompts (i.e., task-specific prompts) into a unified prompt while ensuring diversity. By transforming and aligning base prompts, both previously learned and newly introduced, our approach continuously evolves accumulated knowledge to facilitate learning new tasks. We further introduce a learnable probabilistic gate that adaptively determines which layers to activate during the evolution process. We validate our method on image classification and video action recognition tasks in class-incremental learning, achieving average gains of 9.07% and 7.40% over existing methods across all scenarios.
>
---
#### [new 039] Modality-Aware Feature Matching: A Comprehensive Review of Single- and Cross-Modality Techniques
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉中的特征匹配任务，旨在解决不同模态数据间的匹配问题。论文综述了传统方法与深度学习方法在RGB图像、深度图像、3D点云等多种模态上的应用，强调了模态感知技术的进展，提升了跨模态匹配的鲁棒性与适应性。**

- **链接: [http://arxiv.org/pdf/2507.22791v1](http://arxiv.org/pdf/2507.22791v1)**

> **作者:** Weide Liu; Wei Zhou; Jun Liu; Ping Hu; Jun Cheng; Jungong Han; Weisi Lin
>
> **摘要:** Feature matching is a cornerstone task in computer vision, essential for applications such as image retrieval, stereo matching, 3D reconstruction, and SLAM. This survey comprehensively reviews modality-based feature matching, exploring traditional handcrafted methods and emphasizing contemporary deep learning approaches across various modalities, including RGB images, depth images, 3D point clouds, LiDAR scans, medical images, and vision-language interactions. Traditional methods, leveraging detectors like Harris corners and descriptors such as SIFT and ORB, demonstrate robustness under moderate intra-modality variations but struggle with significant modality gaps. Contemporary deep learning-based methods, exemplified by detector-free strategies like CNN-based SuperPoint and transformer-based LoFTR, substantially improve robustness and adaptability across modalities. We highlight modality-aware advancements, such as geometric and depth-specific descriptors for depth images, sparse and dense learning methods for 3D point clouds, attention-enhanced neural networks for LiDAR scans, and specialized solutions like the MIND descriptor for complex medical image matching. Cross-modal applications, particularly in medical image registration and vision-language tasks, underscore the evolution of feature matching to handle increasingly diverse data interactions.
>
---
#### [new 040] DepR: Depth Guided Single-view Scene Reconstruction with Instance-level Diffusion
- **分类: cs.CV**

- **简介: 该论文属于单视角场景重建任务，旨在解决现有方法未能充分利用深度信息的问题。论文提出DepR方法，通过在训练和推理中引入深度引导的扩散模型，结合实例级生成与布局优化，提升了重建效果与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.22825v1](http://arxiv.org/pdf/2507.22825v1)**

> **作者:** Qingcheng Zhao; Xiang Zhang; Haiyang Xu; Zeyuan Chen; Jianwen Xie; Yuan Gao; Zhuowen Tu
>
> **备注:** ICCV 2025
>
> **摘要:** We propose DepR, a depth-guided single-view scene reconstruction framework that integrates instance-level diffusion within a compositional paradigm. Instead of reconstructing the entire scene holistically, DepR generates individual objects and subsequently composes them into a coherent 3D layout. Unlike previous methods that use depth solely for object layout estimation during inference and therefore fail to fully exploit its rich geometric information, DepR leverages depth throughout both training and inference. Specifically, we introduce depth-guided conditioning to effectively encode shape priors into diffusion models. During inference, depth further guides DDIM sampling and layout optimization, enhancing alignment between the reconstruction and the input image. Despite being trained on limited synthetic data, DepR achieves state-of-the-art performance and demonstrates strong generalization in single-view scene reconstruction, as shown through evaluations on both synthetic and real-world datasets.
>
---
#### [new 041] Enhancing efficiency in paediatric brain tumour segmentation using a pathologically diverse single-center clinical dataset
- **分类: cs.CV; physics.med-ph**

- **简介: 该论文属于医学图像分割任务，旨在提升儿童脑肿瘤MRI图像的自动分割效率与准确性。研究使用深度学习模型nnU-Net，在多种脑肿瘤类型和MRI序列上进行训练与验证。结果表明模型在整体肿瘤和T2高信号区域表现良好，接近人工标注水平，但在增强肿瘤和囊性成分分割上仍存在挑战，同时探索了简化MRI协议的可行性。**

- **链接: [http://arxiv.org/pdf/2507.22152v1](http://arxiv.org/pdf/2507.22152v1)**

> **作者:** A. Piffer; J. A. Buchner; A. G. Gennari; P. Grehten; S. Sirin; E. Ross; I. Ezhov; M. Rosier; J. C. Peeken; M. Piraud; B. Menze; A. Guerreiro Stücklin; A. Jakab; F. Kofler
>
> **备注:** A. Jakab and F. Kofler have shared last authorship
>
> **摘要:** Background Brain tumours are the most common solid malignancies in children, encompassing diverse histological, molecular subtypes and imaging features and outcomes. Paediatric brain tumours (PBTs), including high- and low-grade gliomas (HGG, LGG), medulloblastomas (MB), ependymomas, and rarer forms, pose diagnostic and therapeutic challenges. Deep learning (DL)-based segmentation offers promising tools for tumour delineation, yet its performance across heterogeneous PBT subtypes and MRI protocols remains uncertain. Methods A retrospective single-centre cohort of 174 paediatric patients with HGG, LGG, medulloblastomas (MB), ependymomas, and other rarer subtypes was used. MRI sequences included T1, T1 post-contrast (T1-C), T2, and FLAIR. Manual annotations were provided for four tumour subregions: whole tumour (WT), T2-hyperintensity (T2H), enhancing tumour (ET), and cystic component (CC). A 3D nnU-Net model was trained and tested (121/53 split), with segmentation performance assessed using the Dice similarity coefficient (DSC) and compared against intra- and inter-rater variability. Results The model achieved robust performance for WT and T2H (mean DSC: 0.85), comparable to human annotator variability (mean DSC: 0.86). ET segmentation was moderately accurate (mean DSC: 0.75), while CC performance was poor. Segmentation accuracy varied by tumour type, MRI sequence combination, and location. Notably, T1, T1-C, and T2 alone produced results nearly equivalent to the full protocol. Conclusions DL is feasible for PBTs, particularly for T2H and WT. Challenges remain for ET and CC segmentation, highlighting the need for further refinement. These findings support the potential for protocol simplification and automation to enhance volumetric assessment and streamline paediatric neuro-oncology workflows.
>
---
#### [new 042] UAVScenes: A Multi-Modal Dataset for UAVs
- **分类: cs.CV**

- **简介: 该论文属于多模态无人机感知任务，旨在解决现有数据集在高阶场景理解任务上的局限性。作者基于MARS-LVIG数据集，新增了逐帧语义标注和6自由度姿态信息，支持语义分割、深度估计、定位等任务。**

- **链接: [http://arxiv.org/pdf/2507.22412v1](http://arxiv.org/pdf/2507.22412v1)**

> **作者:** Sijie Wang; Siqi Li; Yawei Zhang; Shangshu Yu; Shenghai Yuan; Rui She; Quanjiang Guo; JinXuan Zheng; Ong Kang Howe; Leonrich Chandra; Shrivarshann Srijeyan; Aditya Sivadas; Toshan Aggarwal; Heyuan Liu; Hongming Zhang; Chujie Chen; Junyu Jiang; Lihua Xie; Wee Peng Tay
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Multi-modal perception is essential for unmanned aerial vehicle (UAV) operations, as it enables a comprehensive understanding of the UAVs' surrounding environment. However, most existing multi-modal UAV datasets are primarily biased toward localization and 3D reconstruction tasks, or only support map-level semantic segmentation due to the lack of frame-wise annotations for both camera images and LiDAR point clouds. This limitation prevents them from being used for high-level scene understanding tasks. To address this gap and advance multi-modal UAV perception, we introduce UAVScenes, a large-scale dataset designed to benchmark various tasks across both 2D and 3D modalities. Our benchmark dataset is built upon the well-calibrated multi-modal UAV dataset MARS-LVIG, originally developed only for simultaneous localization and mapping (SLAM). We enhance this dataset by providing manually labeled semantic annotations for both frame-wise images and LiDAR point clouds, along with accurate 6-degree-of-freedom (6-DoF) poses. These additions enable a wide range of UAV perception tasks, including segmentation, depth estimation, 6-DoF localization, place recognition, and novel view synthesis (NVS). Our dataset is available at https://github.com/sijieaaa/UAVScenes
>
---
#### [new 043] Graph-Guided Dual-Level Augmentation for 3D Scene Segmentation
- **分类: cs.CV**

- **简介: 该论文属于3D点云语义分割任务，旨在提升数据增强效果。现有方法忽略场景全局结构依赖，导致增强场景不够真实。作者提出双层级图引导增强框架，结合局部几何与语义约束及全局拓扑约束，生成高质量3D场景，提升分割性能。**

- **链接: [http://arxiv.org/pdf/2507.22668v1](http://arxiv.org/pdf/2507.22668v1)**

> **作者:** Hongbin Lin; Yifan Jiang; Juangui Xu; Jesse Jiaxi Xu; Yi Lu; Zhengyu Hu; Ying-Cong Chen; Hao Wang
>
> **备注:** 15 pages, 11 figures, to be published in ACMMM 2025 Conference
>
> **摘要:** 3D point cloud segmentation aims to assign semantic labels to individual points in a scene for fine-grained spatial understanding. Existing methods typically adopt data augmentation to alleviate the burden of large-scale annotation. However, most augmentation strategies only focus on local transformations or semantic recomposition, lacking the consideration of global structural dependencies within scenes. To address this limitation, we propose a graph-guided data augmentation framework with dual-level constraints for realistic 3D scene synthesis. Our method learns object relationship statistics from real-world data to construct guiding graphs for scene generation. Local-level constraints enforce geometric plausibility and semantic consistency between objects, while global-level constraints maintain the topological structure of the scene by aligning the generated layout with the guiding graph. Extensive experiments on indoor and outdoor datasets demonstrate that our framework generates diverse and high-quality augmented scenes, leading to consistent improvements in point cloud segmentation performance across various models.
>
---
#### [new 044] Image-Guided Shape-from-Template Using Mesh Inextensibility Constraints
- **分类: cs.CV**

- **简介: 该论文属于形状重建任务，旨在解决基于图像的模板形变物体三维重建问题。现有方法在遮挡严重时性能下降，或依赖大量数据和监督。本文提出一种无监督方法，结合图像特征与网格不可伸展约束，实现更快速、准确的重建，尤其在细节和遮挡场景下表现突出。**

- **链接: [http://arxiv.org/pdf/2507.22699v1](http://arxiv.org/pdf/2507.22699v1)**

> **作者:** Thuy Tran; Ruochen Chen; Shaifali Parashar
>
> **备注:** Accepted to ICCV 2025. Total 13 pages, 9 figures, 9 tables
>
> **摘要:** Shape-from-Template (SfT) refers to the class of methods that reconstruct the 3D shape of a deforming object from images/videos using a 3D template. Traditional SfT methods require point correspondences between images and the texture of the 3D template in order to reconstruct 3D shapes from images/videos in real time. Their performance severely degrades when encountered with severe occlusions in the images because of the unavailability of correspondences. In contrast, modern SfT methods use a correspondence-free approach by incorporating deep neural networks to reconstruct 3D objects, thus requiring huge amounts of data for supervision. Recent advances use a fully unsupervised or self-supervised approach by combining differentiable physics and graphics to deform 3D template to match input images. In this paper, we propose an unsupervised SfT which uses only image observations: color features, gradients and silhouettes along with a mesh inextensibility constraint to reconstruct at a $400\times$ faster pace than (best-performing) unsupervised SfT. Moreover, when it comes to generating finer details and severe occlusions, our method outperforms the existing methodologies by a large margin. Code is available at https://github.com/dvttran/nsft.
>
---
#### [new 045] Runtime Failure Hunting for Physics Engine Based Software Systems: How Far Can We Go?
- **分类: cs.CV; cs.AI; cs.MM; cs.SE**

- **简介: 该论文属于软件工程任务，旨在解决物理引擎软件中的物理失效问题。通过大规模实证研究，提出物理失效分类法，评估检测方法有效性，并总结开发者改进建议，以提升物理引擎软件的可靠性。**

- **链接: [http://arxiv.org/pdf/2507.22099v1](http://arxiv.org/pdf/2507.22099v1)**

> **作者:** Shuqing Li; Qiang Chen; Xiaoxue Ren; Michael R. Lyu
>
> **摘要:** Physics Engines (PEs) are fundamental software frameworks that simulate physical interactions in applications ranging from entertainment to safety-critical systems. Despite their importance, PEs suffer from physics failures, deviations from expected physical behaviors that can compromise software reliability, degrade user experience, and potentially cause critical failures in autonomous vehicles or medical robotics. Current testing approaches for PE-based software are inadequate, typically requiring white-box access and focusing on crash detection rather than semantically complex physics failures. This paper presents the first large-scale empirical study characterizing physics failures in PE-based software. We investigate three research questions addressing the manifestations of physics failures, the effectiveness of detection techniques, and developer perceptions of current detection practices. Our contributions include: (1) a taxonomy of physics failure manifestations; (2) a comprehensive evaluation of detection methods including deep learning, prompt-based techniques, and large multimodal models; and (3) actionable insights from developer experiences for improving detection approaches. To support future research, we release PhysiXFails, code, and other materials at https://sites.google.com/view/physics-failure-detection.
>
---
#### [new 046] TR-PTS: Task-Relevant Parameter and Token Selection for Efficient Tuning
- **分类: cs.CV**

- **简介: 该论文属于视觉任务中的参数高效微调研究。针对大模型微调成本高的问题，提出TR-PTS方法，通过任务相关参数与令牌选择，结合FIM与动态令牌合并，提升效率与性能，优于全微调。**

- **链接: [http://arxiv.org/pdf/2507.22872v1](http://arxiv.org/pdf/2507.22872v1)**

> **作者:** Siqi Luo; Haoran Yang; Yi Xin; Mingyang Yi; Guangyang Wu; Guangtao Zhai; Xiaohong Liu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Large pre-trained models achieve remarkable performance in vision tasks but are impractical for fine-tuning due to high computational and storage costs. Parameter-Efficient Fine-Tuning (PEFT) methods mitigate this issue by updating only a subset of parameters; however, most existing approaches are task-agnostic, failing to fully exploit task-specific adaptations, which leads to suboptimal efficiency and performance. To address this limitation, we propose Task-Relevant Parameter and Token Selection (TR-PTS), a task-driven framework that enhances both computational efficiency and accuracy. Specifically, we introduce Task-Relevant Parameter Selection, which utilizes the Fisher Information Matrix (FIM) to identify and fine-tune only the most informative parameters in a layer-wise manner, while keeping the remaining parameters frozen. Simultaneously, Task-Relevant Token Selection dynamically preserves the most informative tokens and merges redundant ones, reducing computational overhead. By jointly optimizing parameters and tokens, TR-PTS enables the model to concentrate on task-discriminative information. We evaluate TR-PTS on benchmark, including FGVC and VTAB-1k, where it achieves state-of-the-art performance, surpassing full fine-tuning by 3.40% and 10.35%, respectively. The code are available at https://github.com/synbol/TR-PTS.
>
---
#### [new 047] SmartCLIP: Modular Vision-language Alignment with Identification Guarantees
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言预训练任务，旨在解决CLIP模型在图像-文本对齐中的信息错位和表征纠缠问题。作者提出SmartCLIP框架，实现跨粒度语义对齐，并保证语义信息保留与解耦，提升模型在下游任务中的泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.22264v1](http://arxiv.org/pdf/2507.22264v1)**

> **作者:** Shaoan Xie; Lingjing Kong; Yujia Zheng; Yu Yao; Zeyu Tang; Eric P. Xing; Guangyi Chen; Kun Zhang
>
> **备注:** CVPR2025
>
> **摘要:** Contrastive Language-Image Pre-training (CLIP)~\citep{radford2021learning} has emerged as a pivotal model in computer vision and multimodal learning, achieving state-of-the-art performance at aligning visual and textual representations through contrastive learning. However, CLIP struggles with potential information misalignment in many image-text datasets and suffers from entangled representation. On the one hand, short captions for a single image in datasets like MSCOCO may describe disjoint regions in the image, leaving the model uncertain about which visual features to retain or disregard. On the other hand, directly aligning long captions with images can lead to the retention of entangled details, preventing the model from learning disentangled, atomic concepts -- ultimately limiting its generalization on certain downstream tasks involving short prompts. In this paper, we establish theoretical conditions that enable flexible alignment between textual and visual representations across varying levels of granularity. Specifically, our framework ensures that a model can not only \emph{preserve} cross-modal semantic information in its entirety but also \emph{disentangle} visual representations to capture fine-grained textual concepts. Building on this foundation, we introduce \ours, a novel approach that identifies and aligns the most relevant visual and textual representations in a modular manner. Superior performance across various tasks demonstrates its capability to handle information misalignment and supports our identification theory. The code is available at https://github.com/Mid-Push/SmartCLIP.
>
---
#### [new 048] Color as the Impetus: Transforming Few-Shot Learner
- **分类: cs.CV**

- **简介: 该论文属于元学习与少样本分类任务，旨在解决少样本学习中忽略颜色信息的问题。作者提出了ColorSense Learner框架，模拟人类颜色感知机制，通过通道间特征提取和交互学习，强化颜色信息的利用。同时引入ColorSense Distiller进行知识蒸馏，提升模型泛化与迁移能力。实验验证了方法在多个少样本基准上的有效性。**

- **链接: [http://arxiv.org/pdf/2507.22136v1](http://arxiv.org/pdf/2507.22136v1)**

> **作者:** Chaofei Qi; Zhitai Liu; Jianbin Qiu
>
> **摘要:** Humans possess innate meta-learning capabilities, partly attributable to their exceptional color perception. In this paper, we pioneer an innovative viewpoint on few-shot learning by simulating human color perception mechanisms. We propose the ColorSense Learner, a bio-inspired meta-learning framework that capitalizes on inter-channel feature extraction and interactive learning. By strategically emphasizing distinct color information across different channels, our approach effectively filters irrelevant features while capturing discriminative characteristics. Color information represents the most intuitive visual feature, yet conventional meta-learning methods have predominantly neglected this aspect, focusing instead on abstract feature differentiation across categories. Our framework bridges the gap via synergistic color-channel interactions, enabling better intra-class commonality extraction and larger inter-class differences. Furthermore, we introduce a meta-distiller based on knowledge distillation, ColorSense Distiller, which incorporates prior teacher knowledge to augment the student network's meta-learning capacity. We've conducted comprehensive coarse/fine-grained and cross-domain experiments on eleven few-shot benchmarks for validation. Numerous experiments reveal that our methods have extremely strong generalization ability, robustness, and transferability, and effortless handle few-shot classification from the perspective of color perception.
>
---
#### [new 049] LOTS of Fashion! Multi-Conditioning for Image Generation via Sketch-Text Pairing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像生成任务，旨在解决基于草图与文本的时尚设计图像生成问题。作者提出了LOTS方法，通过结合全局描述与局部草图-文本对信息，并引入扩散模型中的分步合并策略，实现更精准的时尚图像生成。**

- **链接: [http://arxiv.org/pdf/2507.22627v1](http://arxiv.org/pdf/2507.22627v1)**

> **作者:** Federico Girella; Davide Talon; Ziyue Liu; Zanxi Ruan; Yiming Wang; Marco Cristani
>
> **备注:** Accepted at ICCV25 (Oral). Project page: https://intelligolabs.github.io/lots/
>
> **摘要:** Fashion design is a complex creative process that blends visual and textual expressions. Designers convey ideas through sketches, which define spatial structure and design elements, and textual descriptions, capturing material, texture, and stylistic details. In this paper, we present LOcalized Text and Sketch for fashion image generation (LOTS), an approach for compositional sketch-text based generation of complete fashion outlooks. LOTS leverages a global description with paired localized sketch + text information for conditioning and introduces a novel step-based merging strategy for diffusion adaptation. First, a Modularized Pair-Centric representation encodes sketches and text into a shared latent space while preserving independent localized features; then, a Diffusion Pair Guidance phase integrates both local and global conditioning via attention-based guidance within the diffusion model's multi-step denoising process. To validate our method, we build on Fashionpedia to release Sketchy, the first fashion dataset where multiple text-sketch pairs are provided per image. Quantitative results show LOTS achieves state-of-the-art image generation performance on both global and localized metrics, while qualitative examples and a human evaluation study highlight its unprecedented level of design customization.
>
---
#### [new 050] Robust Deepfake Detection for Electronic Know Your Customer Systems Using Registered Images
- **分类: cs.CV**

- **简介: 该论文属于深度伪造检测任务，旨在解决电子身份验证系统中深伪攻击的检测问题。论文提出一种结合视频时序分析与注册图像比对的检测算法，能全面识别换脸与表情重演攻击，并提升对图像退化的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.22601v1](http://arxiv.org/pdf/2507.22601v1)**

> **作者:** Takuma Amada; Kazuya Kakizaki; Taiki Miyagawa; Akinori F. Ebihara; Kaede Shiohara; Toshihiko Yamasaki
>
> **备注:** Accepted to 19th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2025)
>
> **摘要:** In this paper, we present a deepfake detection algorithm specifically designed for electronic Know Your Customer (eKYC) systems. To ensure the reliability of eKYC systems against deepfake attacks, it is essential to develop a robust deepfake detector capable of identifying both face swapping and face reenactment, while also being robust to image degradation. We address these challenges through three key contributions: (1)~Our approach evaluates the video's authenticity by detecting temporal inconsistencies in identity vectors extracted by face recognition models, leading to comprehensive detection of both face swapping and face reenactment. (2)~In addition to processing video input, the algorithm utilizes a registered image (assumed to be genuine) to calculate identity discrepancies between the input video and the registered image, significantly improving detection accuracy. (3)~We find that employing a face feature extractor trained on a larger dataset enhances both detection performance and robustness against image degradation. Our experimental results show that our proposed method accurately detects both face swapping and face reenactment comprehensively and is robust against various forms of unseen image degradation. Our source code is publicly available https://github.com/TaikiMiyagawa/DeepfakeDetection4eKYC.
>
---
#### [new 051] Estimating 2D Camera Motion with Hybrid Motion Basis
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉中的2D相机运动估计任务，旨在解决现有方法在复杂非线性变换场景中表现不足的问题。论文提出了CamFlow框架，结合物理和随机运动基底建模相机运动，并设计了基于拉普拉斯分布的混合概率损失函数。通过新构建的基准测试验证，CamFlow在多种场景中表现出更优的鲁棒性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.22480v1](http://arxiv.org/pdf/2507.22480v1)**

> **作者:** Haipeng Li; Tianhao Zhou; Zhanglei Yang; Yi Wu; Yan Chen; Zijing Mao; Shen Cheng; Bing Zeng; Shuaicheng Liu
>
> **备注:** ICCV 2025
>
> **摘要:** Estimating 2D camera motion is a fundamental computer vision task that models the projection of 3D camera movements onto the 2D image plane. Current methods rely on either homography-based approaches, limited to planar scenes, or meshflow techniques that use grid-based local homographies but struggle with complex non-linear transformations. A key insight of our work is that combining flow fields from different homographies creates motion patterns that cannot be represented by any single homography. We introduce CamFlow, a novel framework that represents camera motion using hybrid motion bases: physical bases derived from camera geometry and stochastic bases for complex scenarios. Our approach includes a hybrid probabilistic loss function based on the Laplace distribution that enhances training robustness. For evaluation, we create a new benchmark by masking dynamic objects in existing optical flow datasets to isolate pure camera motion. Experiments show CamFlow outperforms state-of-the-art methods across diverse scenarios, demonstrating superior robustness and generalization in zero-shot settings. Code and datasets are available at our project page: https://lhaippp.github.io/CamFlow/.
>
---
#### [new 052] Trade-offs in Image Generation: How Do Different Dimensions Interact?
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决模型在多个生成维度（如质量、多样性等）间的权衡问题。作者构建了TRIG-Bench数据集和TRIGScore评估指标，提出了关系识别系统与维度权衡图（DTM），用于分析和优化生成模型在不同维度间的平衡。**

- **链接: [http://arxiv.org/pdf/2507.22100v1](http://arxiv.org/pdf/2507.22100v1)**

> **作者:** Sicheng Zhang; Binzhu Xie; Zhonghao Yan; Yuli Zhang; Donghao Zhou; Xiaofei Chen; Shi Qiu; Jiaqi Liu; Guoyang Xie; Zhichao Lu
>
> **备注:** Accepted in ICCV 2025, Codebase: https://github.com/fesvhtr/TRIG
>
> **摘要:** Model performance in text-to-image (T2I) and image-to-image (I2I) generation often depends on multiple aspects, including quality, alignment, diversity, and robustness. However, models' complex trade-offs among these dimensions have rarely been explored due to (1) the lack of datasets that allow fine-grained quantification of these trade-offs, and (2) the use of a single metric for multiple dimensions. To bridge this gap, we introduce TRIG-Bench (Trade-offs in Image Generation), which spans 10 dimensions (Realism, Originality, Aesthetics, Content, Relation, Style, Knowledge, Ambiguity, Toxicity, and Bias), contains 40,200 samples, and covers 132 pairwise dimensional subsets. Furthermore, we develop TRIGScore, a VLM-as-judge metric that automatically adapts to various dimensions. Based on TRIG-Bench and TRIGScore, we evaluate 14 models across T2I and I2I tasks. In addition, we propose the Relation Recognition System to generate the Dimension Trade-off Map (DTM) that visualizes the trade-offs among model-specific capabilities. Our experiments demonstrate that DTM consistently provides a comprehensive understanding of the trade-offs between dimensions for each type of generative model. Notably, we show that the model's dimension-specific weaknesses can be mitigated through fine-tuning on DTM to enhance overall performance. Code is available at: https://github.com/fesvhtr/TRIG
>
---
#### [new 053] Wall Shear Stress Estimation in Abdominal Aortic Aneurysms: Towards Generalisable Neural Surrogate Models
- **分类: cs.CV**

- **简介: 该论文旨在通过几何深度学习方法，快速估算腹主动脉瘤中的血流动力学参数（如壁面剪切应力），以替代传统计算流体力学模拟。论文提出了一种具有几何鲁棒性的神经网络模型，并验证其在不同解剖结构、边界条件和网格分辨率下的泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.22817v1](http://arxiv.org/pdf/2507.22817v1)**

> **作者:** Patryk Rygiel; Julian Suk; Christoph Brune; Kak Khee Yeung; Jelmer M. Wolterink
>
> **摘要:** Abdominal aortic aneurysms (AAAs) are pathologic dilatations of the abdominal aorta posing a high fatality risk upon rupture. Studying AAA progression and rupture risk often involves in-silico blood flow modelling with computational fluid dynamics (CFD) and extraction of hemodynamic factors like time-averaged wall shear stress (TAWSS) or oscillatory shear index (OSI). However, CFD simulations are known to be computationally demanding. Hence, in recent years, geometric deep learning methods, operating directly on 3D shapes, have been proposed as compelling surrogates, estimating hemodynamic parameters in just a few seconds. In this work, we propose a geometric deep learning approach to estimating hemodynamics in AAA patients, and study its generalisability to common factors of real-world variation. We propose an E(3)-equivariant deep learning model utilising novel robust geometrical descriptors and projective geometric algebra. Our model is trained to estimate transient WSS using a dataset of CT scans of 100 AAA patients, from which lumen geometries are extracted and reference CFD simulations with varying boundary conditions are obtained. Results show that the model generalizes well within the distribution, as well as to the external test set. Moreover, the model can accurately estimate hemodynamics across geometry remodelling and changes in boundary conditions. Furthermore, we find that a trained model can be applied to different artery tree topologies, where new and unseen branches are added during inference. Finally, we find that the model is to a large extent agnostic to mesh resolution. These results show the accuracy and generalisation of the proposed model, and highlight its potential to contribute to hemodynamic parameter estimation in clinical practice.
>
---
#### [new 054] Recognizing Actions from Robotic View for Natural Human-Robot Interaction
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自然人机交互中的动作识别任务，旨在解决机器人在复杂场景中远距离识别人类动作的问题。论文构建了大规模数据集ACTIVE，并提出ACTIVE-PC方法，有效提升了移动机器人视角下远距离动作识别的准确性。**

- **链接: [http://arxiv.org/pdf/2507.22522v1](http://arxiv.org/pdf/2507.22522v1)**

> **作者:** Ziyi Wang; Peiming Li; Hong Liu; Zhichao Deng; Can Wang; Jun Liu; Junsong Yuan; Mengyuan Liu
>
> **备注:** 8 pages, 4 figures, Accepted to ICCV2025
>
> **摘要:** Natural Human-Robot Interaction (N-HRI) requires robots to recognize human actions at varying distances and states, regardless of whether the robot itself is in motion or stationary. This setup is more flexible and practical than conventional human action recognition tasks. However, existing benchmarks designed for traditional action recognition fail to address the unique complexities in N-HRI due to limited data, modalities, task categories, and diversity of subjects and environments. To address these challenges, we introduce ACTIVE (Action from Robotic View), a large-scale dataset tailored specifically for perception-centric robotic views prevalent in mobile service robots. ACTIVE comprises 30 composite action categories, 80 participants, and 46,868 annotated video instances, covering both RGB and point cloud modalities. Participants performed various human actions in diverse environments at distances ranging from 3m to 50m, while the camera platform was also mobile, simulating real-world scenarios of robot perception with varying camera heights due to uneven ground. This comprehensive and challenging benchmark aims to advance action and attribute recognition research in N-HRI. Furthermore, we propose ACTIVE-PC, a method that accurately perceives human actions at long distances using Multilevel Neighborhood Sampling, Layered Recognizers, Elastic Ellipse Query, and precise decoupling of kinematic interference from human actions. Experimental results demonstrate the effectiveness of ACTIVE-PC. Our code is available at: https://github.com/wangzy01/ACTIVE-Action-from-Robotic-View.
>
---
#### [new 055] Temporally Consistent Unsupervised Segmentation for Mobile Robot Perception
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于移动机器人感知任务，旨在解决无监督分割在非结构化环境中缺乏时间一致性的问题。作者提出了Frontier-Seg方法，利用DINOv2特征进行超像素聚类，并在视频帧间保持时间一致性，实现无需人工标注的地形分割。**

- **链接: [http://arxiv.org/pdf/2507.22194v1](http://arxiv.org/pdf/2507.22194v1)**

> **作者:** Christian Ellis; Maggie Wigness; Craig Lennon; Lance Fiondella
>
> **摘要:** Rapid progress in terrain-aware autonomous ground navigation has been driven by advances in supervised semantic segmentation. However, these methods rely on costly data collection and labor-intensive ground truth labeling to train deep models. Furthermore, autonomous systems are increasingly deployed in unrehearsed, unstructured environments where no labeled data exists and semantic categories may be ambiguous or domain-specific. Recent zero-shot approaches to unsupervised segmentation have shown promise in such settings but typically operate on individual frames, lacking temporal consistency-a critical property for robust perception in unstructured environments. To address this gap we introduce Frontier-Seg, a method for temporally consistent unsupervised segmentation of terrain from mobile robot video streams. Frontier-Seg clusters superpixel-level features extracted from foundation model backbones-specifically DINOv2-and enforces temporal consistency across frames to identify persistent terrain boundaries or frontiers without human supervision. We evaluate Frontier-Seg on a diverse set of benchmark datasets-including RUGD and RELLIS-3D-demonstrating its ability to perform unsupervised segmentation across unstructured off-road environments.
>
---
#### [new 056] Hydra-Bench: A Benchmark for Multi-Modal Leaf Wetness Sensing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于农业监测中的叶面湿润检测任务，旨在解决现有传感器在自然条件下检测精度和鲁棒性不足的问题。作者构建了一个多模态数据集（Hydra-Bench），包含毫米波雷达、SAR图像和RGB图像，并基于Hydra模型进行了多模态融合实验，评估了不同方法在不同距离下的性能，为未来算法优化提供了基准。**

- **链接: [http://arxiv.org/pdf/2507.22685v1](http://arxiv.org/pdf/2507.22685v1)**

> **作者:** Yimeng Liu; Maolin Gan; Yidong Ren; Gen Li; Jingkai Lin; Younsuk Dong; Zhichao Cao
>
> **摘要:** Leaf wetness detection is a crucial task in agricultural monitoring, as it directly impacts the prediction and protection of plant diseases. However, existing sensing systems suffer from limitations in robustness, accuracy, and environmental resilience when applied to natural leaves under dynamic real-world conditions. To address these challenges, we introduce a new multi-modal dataset specifically designed for evaluating and advancing machine learning algorithms in leaf wetness detection. Our dataset comprises synchronized mmWave raw data, Synthetic Aperture Radar (SAR) images, and RGB images collected over six months from five diverse plant species in both controlled and outdoor field environments. We provide detailed benchmarks using the Hydra model, including comparisons against single modality baselines and multiple fusion strategies, as well as performance under varying scan distances. Additionally, our dataset can serve as a benchmark for future SAR imaging algorithm optimization, enabling a systematic evaluation of detection accuracy under diverse conditions.
>
---
#### [new 057] HOG-CNN: Integrating Histogram of Oriented Gradients with Convolutional Neural Networks for Retinal Image Classification
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在解决视网膜疾病（如糖尿病视网膜病变、青光眼、黄斑变性）的自动诊断问题。论文提出HOG-CNN模型，融合手工特征（HOG）与深度学习（CNN）特征，提升分类性能。实验表明其在多个数据集上表现优异，适用于资源有限的临床环境。**

- **链接: [http://arxiv.org/pdf/2507.22274v1](http://arxiv.org/pdf/2507.22274v1)**

> **作者:** Faisal Ahmed
>
> **备注:** 13 pages; 5 figures
>
> **摘要:** The analysis of fundus images is critical for the early detection and diagnosis of retinal diseases such as Diabetic Retinopathy (DR), Glaucoma, and Age-related Macular Degeneration (AMD). Traditional diagnostic workflows, however, often depend on manual interpretation and are both time- and resource-intensive. To address these limitations, we propose an automated and interpretable clinical decision support framework based on a hybrid feature extraction model called HOG-CNN. Our key contribution lies in the integration of handcrafted Histogram of Oriented Gradients (HOG) features with deep convolutional neural network (CNN) representations. This fusion enables our model to capture both local texture patterns and high-level semantic features from retinal fundus images. We evaluated our model on three public benchmark datasets: APTOS 2019 (for binary and multiclass DR classification), ORIGA (for Glaucoma detection), and IC-AMD (for AMD diagnosis); HOG-CNN demonstrates consistently high performance. It achieves 98.5\% accuracy and 99.2 AUC for binary DR classification, and 94.2 AUC for five-class DR classification. On the IC-AMD dataset, it attains 92.8\% accuracy, 94.8\% precision, and 94.5 AUC, outperforming several state-of-the-art models. For Glaucoma detection on ORIGA, our model achieves 83.9\% accuracy and 87.2 AUC, showing competitive performance despite dataset limitations. We show, through comprehensive appendix studies, the complementary strength of combining HOG and CNN features. The model's lightweight and interpretable design makes it particularly suitable for deployment in resource-constrained clinical environments. These results position HOG-CNN as a robust and scalable tool for automated retinal disease screening.
>
---
#### [new 058] AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data
- **分类: cs.CV; cs.LG**

- **简介: 论文提出AlphaEarth Foundations，一种基于嵌入场模型的地理空间表示方法，旨在利用稀疏标签数据实现全球范围的精准、高效制图。该工作解决了地球观测数据丰富但高质量标签稀缺的问题，通过融合多源时空信息，生成可直接用于多种制图任务的嵌入层，无需重新训练即可超越现有特征提取方法。**

- **链接: [http://arxiv.org/pdf/2507.22291v1](http://arxiv.org/pdf/2507.22291v1)**

> **作者:** Christopher F. Brown; Michal R. Kazmierski; Valerie J. Pasquarella; William J. Rucklidge; Masha Samsikova; Chenhui Zhang; Evan Shelhamer; Estefania Lahera; Olivia Wiles; Simon Ilyushchenko; Noel Gorelick; Lihui Lydia Zhang; Sophia Alj; Emily Schechter; Sean Askay; Oliver Guinan; Rebecca Moore; Alexis Boukouvalas; Pushmeet Kohli
>
> **摘要:** Unprecedented volumes of Earth observation data are continually collected around the world, but high-quality labels remain scarce given the effort required to make physical measurements and observations. This has led to considerable investment in bespoke modeling efforts translating sparse labels into maps. Here we introduce AlphaEarth Foundations, an embedding field model yielding a highly general, geospatial representation that assimilates spatial, temporal, and measurement contexts across multiple sources, enabling accurate and efficient production of maps and monitoring systems from local to global scales. The embeddings generated by AlphaEarth Foundations are the only to consistently outperform all previous featurization approaches tested on a diverse set of mapping evaluations without re-training. We will release a dataset of global, annual, analysis-ready embedding field layers from 2017 through 2024.
>
---
#### [new 059] HRVVS: A High-resolution Video Vasculature Segmentation Network via Hierarchical Autoregressive Residual Priors
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像处理任务，旨在解决肝切除手术视频中血管分割不准确的问题。作者构建了一个高质量的标注数据集，并提出了一种高分辨率视频血管分割网络HRVVS，通过引入预训练的视觉自回归模型和动态记忆解码器，提升了分割精度与细节保留。**

- **链接: [http://arxiv.org/pdf/2507.22530v1](http://arxiv.org/pdf/2507.22530v1)**

> **作者:** Xincheng Yao; Yijun Yang; Kangwei Guo; Ruiqiang Xiao; Haipeng Zhou; Haisu Tao; Jian Yang; Lei Zhu
>
> **摘要:** The segmentation of the hepatic vasculature in surgical videos holds substantial clinical significance in the context of hepatectomy procedures. However, owing to the dearth of an appropriate dataset and the inherently complex task characteristics, few researches have been reported in this domain. To address this issue, we first introduce a high quality frame-by-frame annotated hepatic vasculature dataset containing 35 long hepatectomy videos and 11442 high-resolution frames. On this basis, we propose a novel high-resolution video vasculature segmentation network, dubbed as HRVVS. We innovatively embed a pretrained visual autoregressive modeling (VAR) model into different layers of the hierarchical encoder as prior information to reduce the information degradation generated during the downsampling process. In addition, we designed a dynamic memory decoder on a multi-view segmentation network to minimize the transmission of redundant information while preserving more details between frames. Extensive experiments on surgical video datasets demonstrate that our proposed HRVVS significantly outperforms the state-of-the-art methods. The source code and dataset will be publicly available at \href{https://github.com/scott-yjyang/xx}{https://github.com/scott-yjyang/HRVVS}.
>
---
#### [new 060] Advancing Fetal Ultrasound Image Quality Assessment in Low-Resource Settings
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像质量评估任务，旨在解决低资源地区因缺乏专业人员导致的胎儿超声图像质量评估难题。作者基于预训练模型FetalCLIP，通过参数高效微调方法LoRA构建了FetalCLIP$_{CLS}$模型，并结合分割模型提升分类性能，取得了较高F1分数，推动了资源有限地区产前护理的发展。**

- **链接: [http://arxiv.org/pdf/2507.22802v1](http://arxiv.org/pdf/2507.22802v1)**

> **作者:** Dongli He; Hu Wang; Mohammad Yaqub
>
> **备注:** Accepted to the MICCAI 2025 MIRASOL Workshop
>
> **摘要:** Accurate fetal biometric measurements, such as abdominal circumference, play a vital role in prenatal care. However, obtaining high-quality ultrasound images for these measurements heavily depends on the expertise of sonographers, posing a significant challenge in low-income countries due to the scarcity of trained personnel. To address this issue, we leverage FetalCLIP, a vision-language model pretrained on a curated dataset of over 210,000 fetal ultrasound image-caption pairs, to perform automated fetal ultrasound image quality assessment (IQA) on blind-sweep ultrasound data. We introduce FetalCLIP$_{CLS}$, an IQA model adapted from FetalCLIP using Low-Rank Adaptation (LoRA), and evaluate it on the ACOUSLIC-AI dataset against six CNN and Transformer baselines. FetalCLIP$_{CLS}$ achieves the highest F1 score of 0.757. Moreover, we show that an adapted segmentation model, when repurposed for classification, further improves performance, achieving an F1 score of 0.771. Our work demonstrates how parameter-efficient fine-tuning of fetal ultrasound foundation models can enable task-specific adaptations, advancing prenatal care in resource-limited settings. The experimental code is available at: https://github.com/donglihe-hub/FetalCLIP-IQA.
>
---
#### [new 061] CapRecover: A Cross-Modality Feature Inversion Attack Framework on Vision Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于隐私安全任务，旨在解决视觉语言模型中因中间特征泄露导致的语义信息泄露问题。作者提出了CapRecover框架，通过跨模态特征反转技术，直接从中间特征恢复语义内容，如标签或描述。同时提出添加随机噪声的防护方法，有效防止泄露且无需额外训练成本。**

- **链接: [http://arxiv.org/pdf/2507.22828v1](http://arxiv.org/pdf/2507.22828v1)**

> **作者:** Kedong Xiu; Saiqian Zhang
>
> **备注:** 9 pages, accepted by the 2025 ACM Multimedia Conference
>
> **摘要:** As Vision-Language Models (VLMs) are increasingly deployed in split-DNN configurations--with visual encoders (e.g., ResNet, ViT) operating on user devices and sending intermediate features to the cloud--there is a growing privacy risk from semantic information leakage. Existing approaches to reconstructing images from these intermediate features often result in blurry, semantically ambiguous images. To directly address semantic leakage, we propose CapRecover, a cross-modality inversion framework that recovers high-level semantic content, such as labels or captions, directly from intermediate features without image reconstruction. We evaluate CapRecover on multiple datasets and victim models, demonstrating strong performance in semantic recovery. Specifically, CapRecover achieves up to 92.71% Top-1 label accuracy on CIFAR-10 and generates fluent captions from ResNet50 features on COCO2017 with ROUGE-L scores up to 0.52. Our analysis further reveals that deeper convolutional layers encode significantly more semantic information compared to shallow layers. To mitigate semantic leakage, we introduce a simple yet effective protection method: adding random noise to intermediate features at each layer and removing the noise in the next layer. Experimental results show that this approach prevents semantic leakage without additional training costs.
>
---
#### [new 062] Exploiting Diffusion Prior for Task-driven Image Restoration
- **分类: cs.CV**

- **简介: 本文属于任务驱动图像恢复（TDIR）任务，旨在解决低质量图像输入导致高层视觉任务性能下降的问题。作者提出EDTR方法，利用扩散先验恢复任务相关细节，通过在扩散过程中引入像素误差和少量去噪步骤，有效提升多任务性能与视觉质量。**

- **链接: [http://arxiv.org/pdf/2507.22459v1](http://arxiv.org/pdf/2507.22459v1)**

> **作者:** Jaeha Kim; Junghun Oh; Kyoung Mu Lee
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Task-driven image restoration (TDIR) has recently emerged to address performance drops in high-level vision tasks caused by low-quality (LQ) inputs. Previous TDIR methods struggle to handle practical scenarios in which images are degraded by multiple complex factors, leaving minimal clues for restoration. This motivates us to leverage the diffusion prior, one of the most powerful natural image priors. However, while the diffusion prior can help generate visually plausible results, using it to restore task-relevant details remains challenging, even when combined with recent TDIR methods. To address this, we propose EDTR, which effectively harnesses the power of diffusion prior to restore task-relevant details. Specifically, we propose directly leveraging useful clues from LQ images in the diffusion process by generating from pixel-error-based pre-restored LQ images with mild noise added. Moreover, we employ a small number of denoising steps to prevent the generation of redundant details that dilute crucial task-related information. We demonstrate that our method effectively utilizes diffusion prior for TDIR, significantly enhancing task performance and visual quality across diverse tasks with multiple complex degradations.
>
---
#### [new 063] Exploring the Application of Visual Question Answering (VQA) for Classroom Activity Monitoring
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉问答（VQA）任务，旨在解决课堂行为监测问题。作者评估了LLaMA2、LLaMA3、QWEN3和NVILA等模型在课堂视频分析中的适用性，并提出了BAV-Classroom-VQA数据集，展示了这些模型在行为相关视觉问答中的潜力。**

- **链接: [http://arxiv.org/pdf/2507.22369v1](http://arxiv.org/pdf/2507.22369v1)**

> **作者:** Sinh Trong Vu; Hieu Trung Pham; Dung Manh Nguyen; Hieu Minh Hoang; Nhu Hoang Le; Thu Ha Pham; Tai Tan Mai
>
> **摘要:** Classroom behavior monitoring is a critical aspect of educational research, with significant implications for student engagement and learning outcomes. Recent advancements in Visual Question Answering (VQA) models offer promising tools for automatically analyzing complex classroom interactions from video recordings. In this paper, we investigate the applicability of several state-of-the-art open-source VQA models, including LLaMA2, LLaMA3, QWEN3, and NVILA, in the context of classroom behavior analysis. To facilitate rigorous evaluation, we introduce our BAV-Classroom-VQA dataset derived from real-world classroom video recordings at the Banking Academy of Vietnam. We present the methodology for data collection, annotation, and benchmark the performance of the selected VQA models on this dataset. Our initial experimental results demonstrate that all four models achieve promising performance levels in answering behavior-related visual questions, showcasing their potential in future classroom analytics and intervention systems.
>
---
#### [new 064] HQ-CLIP: Leveraging Large Vision-Language Models to Create High-Quality Image-Text Datasets and CLIP Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型任务，旨在解决图像-文本对数据质量提升问题。通过构建LVLM驱动的数据优化流程，生成多粒度标注数据VLM-150M，并提出结合负描述与短标签的训练范式，训练出性能优越的HQ-CLIP模型。**

- **链接: [http://arxiv.org/pdf/2507.22431v1](http://arxiv.org/pdf/2507.22431v1)**

> **作者:** Zhixiang Wei; Guangting Wang; Xiaoxiao Ma; Ke Mei; Huaian Chen; Yi Jin; Fengyun Rao
>
> **摘要:** Large-scale but noisy image-text pair data have paved the way for the success of Contrastive Language-Image Pretraining (CLIP). As the foundation vision encoder, CLIP in turn serves as the cornerstone for most large vision-language models (LVLMs). This interdependence naturally raises an interesting question: Can we reciprocally leverage LVLMs to enhance the quality of image-text pair data, thereby opening the possibility of a self-reinforcing cycle for continuous improvement? In this work, we take a significant step toward this vision by introducing an LVLM-driven data refinement pipeline. Our framework leverages LVLMs to process images and their raw alt-text, generating four complementary textual formulas: long positive descriptions, long negative descriptions, short positive tags, and short negative tags. Applying this pipeline to the curated DFN-Large dataset yields VLM-150M, a refined dataset enriched with multi-grained annotations. Based on this dataset, we further propose a training paradigm that extends conventional contrastive learning by incorporating negative descriptions and short tags as additional supervised signals. The resulting model, namely HQ-CLIP, demonstrates remarkable improvements across diverse benchmarks. Within a comparable training data scale, our approach achieves state-of-the-art performance in zero-shot classification, cross-modal retrieval, and fine-grained visual understanding tasks. In retrieval benchmarks, HQ-CLIP even surpasses standard CLIP models trained on the DFN-2B dataset, which contains 10$\times$ more training data than ours. All code, data, and models are available at https://zxwei.site/hqclip.
>
---
#### [new 065] Moiré Zero: An Efficient and High-Performance Neural Architecture for Moiré Removal
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于图像处理任务，旨在解决由频率混叠引起的摩尔纹消除问题。作者提出了一种高效神经网络MZNet，包含多尺度双注意力模块、多形状大卷积核模块及特征融合跳跃连接，有效去除摩尔纹并保持低计算成本，适用于实际应用。**

- **链接: [http://arxiv.org/pdf/2507.22407v1](http://arxiv.org/pdf/2507.22407v1)**

> **作者:** Seungryong Lee; Woojeong Baek; Younghyun Kim; Eunwoo Kim; Haru Moon; Donggon Yoo; Eunbyung Park
>
> **备注:** Project page: https://sngryonglee.github.io/MoireZero
>
> **摘要:** Moir\'e patterns, caused by frequency aliasing between fine repetitive structures and a camera sensor's sampling process, have been a significant obstacle in various real-world applications, such as consumer photography and industrial defect inspection. With the advancements in deep learning algorithms, numerous studies-predominantly based on convolutional neural networks-have suggested various solutions to address this issue. Despite these efforts, existing approaches still struggle to effectively eliminate artifacts due to the diverse scales, orientations, and color shifts of moir\'e patterns, primarily because the constrained receptive field of CNN-based architectures limits their ability to capture the complex characteristics of moir\'e patterns. In this paper, we propose MZNet, a U-shaped network designed to bring images closer to a 'Moire-Zero' state by effectively removing moir\'e patterns. It integrates three specialized components: Multi-Scale Dual Attention Block (MSDAB) for extracting and refining multi-scale features, Multi-Shape Large Kernel Convolution Block (MSLKB) for capturing diverse moir\'e structures, and Feature Fusion-Based Skip Connection for enhancing information flow. Together, these components enhance local texture restoration and large-scale artifact suppression. Experiments on benchmark datasets demonstrate that MZNet achieves state-of-the-art performance on high-resolution datasets and delivers competitive results on lower-resolution dataset, while maintaining a low computational cost, suggesting that it is an efficient and practical solution for real-world applications. Project page: https://sngryonglee.github.io/MoireZero
>
---
#### [new 066] AI in Agriculture: A Survey of Deep Learning Techniques for Crops, Fisheries and Livestock
- **分类: cs.CV**

- **简介: 该论文属于人工智能在农业领域的应用任务，旨在解决农作物、渔业和畜牧业中的技术挑战。论文系统综述了200多篇研究，涵盖传统机器学习、深度学习及视觉-语言模型在农业中的应用，分析了数据多样性、评估指标和地域分布等实施难题，并提出了多模态数据融合、边缘设备部署和可适配模型等未来研究方向。**

- **链接: [http://arxiv.org/pdf/2507.22101v1](http://arxiv.org/pdf/2507.22101v1)**

> **作者:** Umair Nawaz; Muhammad Zaigham Zaheer; Fahad Shahbaz Khan; Hisham Cholakkal; Salman Khan; Rao Muhammad Anwer
>
> **摘要:** Crops, fisheries and livestock form the backbone of global food production, essential to feed the ever-growing global population. However, these sectors face considerable challenges, including climate variability, resource limitations, and the need for sustainable management. Addressing these issues requires efficient, accurate, and scalable technological solutions, highlighting the importance of artificial intelligence (AI). This survey presents a systematic and thorough review of more than 200 research works covering conventional machine learning approaches, advanced deep learning techniques (e.g., vision transformers), and recent vision-language foundation models (e.g., CLIP) in the agriculture domain, focusing on diverse tasks such as crop disease detection, livestock health management, and aquatic species monitoring. We further cover major implementation challenges such as data variability and experimental aspects: datasets, performance evaluation metrics, and geographical focus. We finish the survey by discussing potential open research directions emphasizing the need for multimodal data integration, efficient edge-device deployment, and domain-adaptable AI models for diverse farming environments. Rapid growth of evolving developments in this field can be actively tracked on our project page: https://github.com/umair1221/AI-in-Agriculture
>
---
#### [new 067] Generative Active Learning for Long-tail Trajectory Prediction via Controllable Diffusion Model
- **分类: cs.CV**

- **简介: 该论文属于轨迹预测任务，旨在解决自动驾驶中罕见长尾场景预测效果差的问题。作者提出GALTraj，通过生成式主动学习，在不修改模型结构的前提下，利用可控扩散模型增强训练中的尾部样本，提升模型在长尾和头部样本上的预测性能。**

- **链接: [http://arxiv.org/pdf/2507.22615v1](http://arxiv.org/pdf/2507.22615v1)**

> **作者:** Daehee Park; Monu Surana; Pranav Desai; Ashish Mehta; Reuben MV John; Kuk-Jin Yoon
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** While data-driven trajectory prediction has enhanced the reliability of autonomous driving systems, it still struggles with rarely observed long-tail scenarios. Prior works addressed this by modifying model architectures, such as using hypernetworks. In contrast, we propose refining the training process to unlock each model's potential without altering its structure. We introduce Generative Active Learning for Trajectory prediction (GALTraj), the first method to successfully deploy generative active learning into trajectory prediction. It actively identifies rare tail samples where the model fails and augments these samples with a controllable diffusion model during training. In our framework, generating scenarios that are diverse, realistic, and preserve tail-case characteristics is paramount. Accordingly, we design a tail-aware generation method that applies tailored diffusion guidance to generate trajectories that both capture rare behaviors and respect traffic rules. Unlike prior simulation methods focused solely on scenario diversity, GALTraj is the first to show how simulator-driven augmentation benefits long-tail learning in trajectory prediction. Experiments on multiple trajectory datasets (WOMD, Argoverse2) with popular backbones (QCNet, MTR) confirm that our method significantly boosts performance on tail samples and also enhances accuracy on head samples.
>
---
#### [new 068] GVD: Guiding Video Diffusion Model for Scalable Video Distillation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频数据蒸馏任务，旨在解决大规模视频数据计算和存储开销大的问题。作者提出GVD方法，利用扩散模型进行视频蒸馏，联合提取时空特征，在MiniUCF和HMDB51数据集上显著优于现有方法，能在低帧数下保持高性能，同时支持高分辨率和高实例蒸馏。**

- **链接: [http://arxiv.org/pdf/2507.22360v1](http://arxiv.org/pdf/2507.22360v1)**

> **作者:** Kunyang Li; Jeffrey A Chan Santiago; Sarinda Dhanesh Samarasinghe; Gaowen Liu; Mubarak Shah
>
> **摘要:** To address the larger computation and storage requirements associated with large video datasets, video dataset distillation aims to capture spatial and temporal information in a significantly smaller dataset, such that training on the distilled data has comparable performance to training on all of the data. We propose GVD: Guiding Video Diffusion, the first diffusion-based video distillation method. GVD jointly distills spatial and temporal features, ensuring high-fidelity video generation across diverse actions while capturing essential motion information. Our method's diverse yet representative distillations significantly outperform previous state-of-the-art approaches on the MiniUCF and HMDB51 datasets across 5, 10, and 20 Instances Per Class (IPC). Specifically, our method achieves 78.29 percent of the original dataset's performance using only 1.98 percent of the total number of frames in MiniUCF. Additionally, it reaches 73.83 percent of the performance with just 3.30 percent of the frames in HMDB51. Experimental results across benchmark video datasets demonstrate that GVD not only achieves state-of-the-art performance but can also generate higher resolution videos and higher IPC without significantly increasing computational cost.
>
---
#### [new 069] Visual Language Models as Zero-Shot Deepfake Detectors
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像分类任务，旨在解决深度伪造（deepfake）图像检测问题。现有方法依赖专门训练的分类器，仅关注图像域。论文提出基于视觉语言模型（VLM）的零样本检测方法，在新数据集和DFDC-P数据集上验证了其优于传统方法的检测性能。**

- **链接: [http://arxiv.org/pdf/2507.22469v1](http://arxiv.org/pdf/2507.22469v1)**

> **作者:** Viacheslav Pirogov
>
> **备注:** Accepted to the ICML 2025 Workshop on Reliable and Responsible Foundation Models
>
> **摘要:** The contemporary phenomenon of deepfakes, utilizing GAN or diffusion models for face swapping, presents a substantial and evolving threat in digital media, identity verification, and a multitude of other systems. The majority of existing methods for detecting deepfakes rely on training specialized classifiers to distinguish between genuine and manipulated images, focusing only on the image domain without incorporating any auxiliary tasks that could enhance robustness. In this paper, inspired by the zero-shot capabilities of Vision Language Models, we propose a novel VLM-based approach to image classification and then evaluate it for deepfake detection. Specifically, we utilize a new high-quality deepfake dataset comprising 60,000 images, on which our zero-shot models demonstrate superior performance to almost all existing methods. Subsequently, we compare the performance of the best-performing architecture, InstructBLIP, on the popular deepfake dataset DFDC-P against traditional methods in two scenarios: zero-shot and in-domain fine-tuning. Our results demonstrate the superiority of VLMs over traditional classifiers.
>
---
#### [new 070] A Dual-Feature Extractor Framework for Accurate Back Depth and Spine Morphology Estimation from Monocular RGB Images
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决脊柱侧弯评估中X射线的辐射和可及性问题。作者提出一种双特征提取框架，从单目RGB图像中准确估计背部深度和脊柱形态。通过设计GAMA-Net网络，结合Patch-Based Hybrid Attention模块和Adaptive Multiscale Feature Fusion模块，实现高精度深度估计和脊柱曲线生成，验证了深度信息对脊柱形态估计的有效性。**

- **链接: [http://arxiv.org/pdf/2507.22691v1](http://arxiv.org/pdf/2507.22691v1)**

> **作者:** Yuxin Wei; Yue Zhang; Moxin Zhao; Chang Shi; Jason P. Y. Cheung; Teng Zhang; Nan Meng
>
> **摘要:** Scoliosis is a prevalent condition that impacts both physical health and appearance, with adolescent idiopathic scoliosis (AIS) being the most common form. Currently, the main AIS assessment tool, X-rays, poses significant limitations, including radiation exposure and limited accessibility in poor and remote areas. To address this problem, the current solutions are using RGB images to analyze spine morphology. However, RGB images are highly susceptible to environmental factors, such as lighting conditions, compromising model stability and generalizability. Therefore, in this study, we propose a novel pipeline to accurately estimate the depth information of the unclothed back, compensating for the limitations of 2D information, and then estimate spine morphology by integrating both depth and surface information. To capture the subtle depth variations of the back surface with precision, we design an adaptive multiscale feature learning network named Grid-Aware Multiscale Adaptive Network (GAMA-Net). This model uses dual encoders to extract both patch-level and global features, which are then interacted by the Patch-Based Hybrid Attention (PBHA) module. The Adaptive Multiscale Feature Fusion (AMFF) module is used to dynamically fuse information in the decoder. As a result, our depth estimation model achieves remarkable accuracy across three different evaluation metrics, with scores of nearly 78.2%, 93.6%, and 97.5%, respectively. To further validate the effectiveness of the predicted depth, we integrate both surface and depth information for spine morphology estimation. This integrated approach enhances the accuracy of spine curve generation, achieving an impressive performance of up to 97%.
>
---
#### [new 071] Pathology Foundation Models are Scanner Sensitive: Benchmark and Mitigation with Contrastive ScanGen Loss
- **分类: q-bio.QM; cs.AI; cs.CV; eess.IV; q-bio.TO; I.2; I.2.6; I.4; I.4.7; I.5; J.3; J.6**

- **简介: 该论文属于计算病理学任务，旨在解决深度学习模型在不同扫描仪间的表现差异问题。作者提出ScanGen对比损失函数，缓解扫描仪偏差，提升模型在跨扫描仪数据上的泛化能力，实验验证了其在EGFR突变预测中的有效性。**

- **链接: [http://arxiv.org/pdf/2507.22092v1](http://arxiv.org/pdf/2507.22092v1)**

> **作者:** Gianluca Carloni; Biagio Brattoli; Seongho Keum; Jongchan Park; Taebum Lee; Chang Ho Ahn; Sergio Pereira
>
> **备注:** Accepted (Oral) in MedAGI 2025 International Workshop at MICCAI Conference
>
> **摘要:** Computational pathology (CPath) has shown great potential in mining actionable insights from Whole Slide Images (WSIs). Deep Learning (DL) has been at the center of modern CPath, and while it delivers unprecedented performance, it is also known that DL may be affected by irrelevant details, such as those introduced during scanning by different commercially available scanners. This may lead to scanner bias, where the model outputs for the same tissue acquired by different scanners may vary. In turn, it hinders the trust of clinicians in CPath-based tools and their deployment in real-world clinical practices. Recent pathology Foundation Models (FMs) promise to provide better domain generalization capabilities. In this paper, we benchmark FMs using a multi-scanner dataset and show that FMs still suffer from scanner bias. Following this observation, we propose ScanGen, a contrastive loss function applied during task-specific fine-tuning that mitigates scanner bias, thereby enhancing the models' robustness to scanner variations. Our approach is applied to the Multiple Instance Learning task of Epidermal Growth Factor Receptor (EGFR) mutation prediction from H\&E-stained WSIs in lung cancer. We observe that ScanGen notably enhances the ability to generalize across scanners, while retaining or improving the performance of EGFR mutation prediction.
>
---
#### [new 072] Towards Blind Bitstream-corrupted Video Recovery via a Visual Foundation Model-driven Framework
- **分类: eess.IV; cs.AI; cs.CV; cs.MM**

- **简介: 该论文属于视频恢复任务，旨在解决比特流受损视频的盲恢复问题。现有方法依赖人工标注受损区域，效率低且效果受限。论文提出一种基于视觉基础模型的框架，包含DAC模型与CFC模块，实现自动定位损坏区域并优化特征补全，提升恢复质量，无需人工标注。**

- **链接: [http://arxiv.org/pdf/2507.22481v1](http://arxiv.org/pdf/2507.22481v1)**

> **作者:** Tianyi Liu; Kejun Wu; Chen Cai; Yi Wang; Kim-Hui Yap; Lap-Pui Chau
>
> **备注:** 10 pages, 5 figures, accepted by ACMMM 2025
>
> **摘要:** Video signals are vulnerable in multimedia communication and storage systems, as even slight bitstream-domain corruption can lead to significant pixel-domain degradation. To recover faithful spatio-temporal content from corrupted inputs, bitstream-corrupted video recovery has recently emerged as a challenging and understudied task. However, existing methods require time-consuming and labor-intensive annotation of corrupted regions for each corrupted video frame, resulting in a large workload in practice. In addition, high-quality recovery remains difficult as part of the local residual information in corrupted frames may mislead feature completion and successive content recovery. In this paper, we propose the first blind bitstream-corrupted video recovery framework that integrates visual foundation models with a recovery model, which is adapted to different types of corruption and bitstream-level prompts. Within the framework, the proposed Detect Any Corruption (DAC) model leverages the rich priors of the visual foundation model while incorporating bitstream and corruption knowledge to enhance corruption localization and blind recovery. Additionally, we introduce a novel Corruption-aware Feature Completion (CFC) module, which adaptively processes residual contributions based on high-level corruption understanding. With VFM-guided hierarchical feature augmentation and high-level coordination in a mixture-of-residual-experts (MoRE) structure, our method suppresses artifacts and enhances informative residuals. Comprehensive evaluations show that the proposed method achieves outstanding performance in bitstream-corrupted video recovery without requiring a manually labeled mask sequence. The demonstrated effectiveness will help to realize improved user experience, wider application scenarios, and more reliable multimedia communication and storage systems.
>
---
#### [new 073] RCR-AF: Enhancing Model Generalization via Rademacher Complexity Reduction Activation Function
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于深度学习模型鲁棒性提升任务，旨在解决模型易受对抗攻击的问题。论文提出了一种新的激活函数RCR-AF，结合了GELU和ReLU的优点，并通过控制模型复杂度增强鲁棒性。理论分析和实验验证表明，RCR-AF在标准训练和对抗训练下均表现出更优的准确性和抗攻击能力。**

- **链接: [http://arxiv.org/pdf/2507.22446v1](http://arxiv.org/pdf/2507.22446v1)**

> **作者:** Yunrui Yu; Kafeng Wang; Hang Su; Jun Zhu
>
> **摘要:** Despite their widespread success, deep neural networks remain critically vulnerable to adversarial attacks, posing significant risks in safety-sensitive applications. This paper investigates activation functions as a crucial yet underexplored component for enhancing model robustness. We propose a Rademacher Complexity Reduction Activation Function (RCR-AF), a novel activation function designed to improve both generalization and adversarial resilience. RCR-AF uniquely combines the advantages of GELU (including smoothness, gradient stability, and negative information retention) with ReLU's desirable monotonicity, while simultaneously controlling both model sparsity and capacity through built-in clipping mechanisms governed by two hyperparameters, $\alpha$ and $\gamma$. Our theoretical analysis, grounded in Rademacher complexity, demonstrates that these parameters directly modulate the model's Rademacher complexity, offering a principled approach to enhance robustness. Comprehensive empirical evaluations show that RCR-AF consistently outperforms widely-used alternatives (ReLU, GELU, and Swish) in both clean accuracy under standard training and in adversarial robustness within adversarial training paradigms.
>
---
#### [new 074] trAIce3D: A Prompt-Driven Transformer Based U-Net for Semantic Segmentation of Microglial Cells from Large-Scale 3D Microscopy Images
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分割任务，旨在解决大规模3D显微图像中小胶质细胞（包括细胞体和分支）精确分割的难题。现有方法存在对重叠结构和噪声敏感、依赖手动调参或半自动操作等问题。论文提出了trAIce3D，一种基于Transformer的3D U-Net架构，采用两阶段方法：第一阶段自监督分割细胞体，第二阶段以细胞体坐标为提示，精细化分割分支结构。该方法提升了分割精度和泛化能力，适用于复杂细胞形态的规模化分析，并可扩展至其他神经细胞类型。**

- **链接: [http://arxiv.org/pdf/2507.22635v1](http://arxiv.org/pdf/2507.22635v1)**

> **作者:** MohammadAmin Alamalhoda; Arsalan Firoozi; Alessandro Venturino; Sandra Siegert
>
> **备注:** 10 pages, 2 figures
>
> **摘要:** The shape of a cell contains essential information about its function within the biological system. Segmenting these structures from large-scale 3D microscopy images is challenging, limiting clinical insights especially for microglia, immune-associated cells involved in neurodegenerative diseases. Existing segmentation methods mainly focus on cell bodies, struggle with overlapping structures, perform poorly on noisy images, require hyperparameter tuning for each new dataset, or rely on tedious semi-automated approaches. We introduce trAIce3D, a deep-learning architecture designed for precise microglia segmentation, capturing both somas and branches. It employs a two-stage approach: first, a 3D U-Net with vision transformers in the encoder detects somas using a sliding-window technique to cover the entire image. Then, the same architecture, enhanced with cross-attention blocks in skip connections, refines each soma and its branches by using soma coordinates as a prompt and a 3D window around the target cell as input. Training occurs in two phases: self-supervised Soma Segmentation, followed by prompt-based Branch Segmentation, leveraging pre-trained weights from the first phase. Trained and evaluated on a dataset of 41,230 microglial cells, trAIce3D significantly improves segmentation accuracy and generalization, enabling scalable analysis of complex cellular morphologies. While optimized for microglia, its architecture can extend to other intricate cell types, such as neurons and astrocytes, broadening its impact on neurobiological research.
>
---
#### [new 075] Exploration of Low-Cost but Accurate Radar-Based Human Motion Direction Determination
- **分类: eess.SP; cs.CV; 68T45; I.5.4**

- **简介: 该论文属于雷达人体运动方向识别任务，旨在解决低成本高精度运动方向判定问题。通过生成雷达多普勒时域图，结合特征增强模型，并采用轻量混合神经网络结构进行方向识别，验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2507.22567v1](http://arxiv.org/pdf/2507.22567v1)**

> **作者:** Weicheng Gao
>
> **备注:** 5 pages, 5 figures, 2 tables
>
> **摘要:** This work is completed on a whim after discussions with my junior colleague. The motion direction angle affects the micro-Doppler spectrum width, thus determining the human motion direction can provide important prior information for downstream tasks such as gait recognition. However, Doppler-Time map (DTM)-based methods still have room for improvement in achieving feature augmentation and motion determination simultaneously. In response, a low-cost but accurate radar-based human motion direction determination (HMDD) method is explored in this paper. In detail, the radar-based human gait DTMs are first generated, and then the feature augmentation is achieved using feature linking model. Subsequently, the HMDD is implemented through a lightweight and fast Vision Transformer-Convolutional Neural Network hybrid model structure. The effectiveness of the proposed method is verified through open-source dataset. The open-source code of this work is released at: https://github.com/JoeyBGOfficial/Low-Cost-Accurate-Radar-Based-Human-Motion-Direction-Determination.
>
---
#### [new 076] FGFP: A Fractional Gaussian Filter and Pruning for Deep Neural Networks Compression
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于深度神经网络压缩任务，旨在解决模型在边缘设备部署时计算和存储负担过重的问题。论文提出了一种名为FGFP的框架，结合分数阶高斯滤波器和自适应非结构化剪枝，以实现高效压缩。**

- **链接: [http://arxiv.org/pdf/2507.22527v1](http://arxiv.org/pdf/2507.22527v1)**

> **作者:** Kuan-Ting Tu; Po-Hsien Yu; Yu-Syuan Tseng; Shao-Yi Chien
>
> **备注:** 8 pages, 2 figures, 4 tables, Accepted by ICML 2025 Workshop (TTODLer-FM)
>
> **摘要:** Network compression techniques have become increasingly important in recent years because the loads of Deep Neural Networks (DNNs) are heavy for edge devices in real-world applications. While many methods compress neural network parameters, deploying these models on edge devices remains challenging. To address this, we propose the fractional Gaussian filter and pruning (FGFP) framework, which integrates fractional-order differential calculus and Gaussian function to construct fractional Gaussian filters (FGFs). To reduce the computational complexity of fractional-order differential operations, we introduce Gr\"unwald-Letnikov fractional derivatives to approximate the fractional-order differential equation. The number of parameters for each kernel in FGF is minimized to only seven. Beyond the architecture of Fractional Gaussian Filters, our FGFP framework also incorporates Adaptive Unstructured Pruning (AUP) to achieve higher compression ratios. Experiments on various architectures and benchmarks show that our FGFP framework outperforms recent methods in accuracy and compression. On CIFAR-10, ResNet-20 achieves only a 1.52% drop in accuracy while reducing the model size by 85.2%. On ImageNet2012, ResNet-50 achieves only a 1.63% drop in accuracy while reducing the model size by 69.1%.
>
---
#### [new 077] Eyepiece-free pupil-optimized holographic near-eye displays
- **分类: physics.optics; cs.CV**

- **简介: 该论文属于显示技术任务，旨在解决全息近眼显示中因瞳孔限制导致的图像质量下降问题。作者提出了一种无需目镜的优化方法，通过球面相位调制生成多视角，并联合优化幅度与相位分布，有效缓解图像退化，提升虚拟现实/增强现实显示效果。**

- **链接: [http://arxiv.org/pdf/2507.22420v1](http://arxiv.org/pdf/2507.22420v1)**

> **作者:** Jie Zhou; Shuyang Xie; Yang Wu; Lei Jiang; Yimou Luo; Jun Wang
>
> **摘要:** Computer-generated holography (CGH) represents a transformative visualization approach for next-generation immersive virtual and augmented reality (VR/AR) displays, enabling precise wavefront modulation and naturally providing comprehensive physiological depth cues without the need for bulky optical assemblies. Despite significant advancements in computational algorithms enhancing image quality and achieving real-time generation, practical implementations of holographic near-eye displays (NEDs) continue to face substantial challenges arising from finite and dynamically varying pupil apertures, which degrade image quality and compromise user experience. In this study, we introduce an eyepiece-free pupil-optimized holographic NED. Our proposed method employs a customized spherical phase modulation strategy to generate multiple viewpoints within the pupil, entirely eliminating the dependence on conventional optical eyepieces. Through the joint optimization of amplitude and phase distributions across these viewpoints, the method markedly mitigates image degradation due to finite pupil sampling and resolves inapparent depth cues induced by the spherical phase. The demonstrated method signifies a substantial advancement toward the realization of compact, lightweight, and flexible holographic NED systems, fulfilling stringent requirements for future VR/AR display technologies.
>
---
#### [new 078] Hate in Plain Sight: On the Risks of Moderating AI-Generated Hateful Illusions
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于图像内容安全任务，旨在解决AI生成的“仇恨幻象”绕过内容审核的问题。作者构建了包含1571张仇恨幻象的Hateful Illusion数据集，评估现有审核模型效果差，并提出改进方法。**

- **链接: [http://arxiv.org/pdf/2507.22617v1](http://arxiv.org/pdf/2507.22617v1)**

> **作者:** Yiting Qu; Ziqing Yang; Yihan Ma; Michael Backes; Savvas Zannettou; Yang Zhang
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Recent advances in text-to-image diffusion models have enabled the creation of a new form of digital art: optical illusions--visual tricks that create different perceptions of reality. However, adversaries may misuse such techniques to generate hateful illusions, which embed specific hate messages into harmless scenes and disseminate them across web communities. In this work, we take the first step toward investigating the risks of scalable hateful illusion generation and the potential for bypassing current content moderation models. Specifically, we generate 1,860 optical illusions using Stable Diffusion and ControlNet, conditioned on 62 hate messages. Of these, 1,571 are hateful illusions that successfully embed hate messages, either overtly or subtly, forming the Hateful Illusion dataset. Using this dataset, we evaluate the performance of six moderation classifiers and nine vision language models (VLMs) in identifying hateful illusions. Experimental results reveal significant vulnerabilities in existing moderation models: the detection accuracy falls below 0.245 for moderation classifiers and below 0.102 for VLMs. We further identify a critical limitation in their vision encoders, which mainly focus on surface-level image details while overlooking the secondary layer of information, i.e., hidden messages. To address this risk, we explore preliminary mitigation measures and identify the most effective approaches from the perspectives of image transformations and training-level strategies.
>
---
#### [new 079] Theoretical Analysis of Relative Errors in Gradient Computations for Adversarial Attacks with CE Loss
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于对抗攻击任务，旨在解决梯度计算中的浮点误差导致攻击效果下降的问题。论文分析了交叉熵损失在不同攻击场景下的误差特性，提出T-MIFPE损失函数以减小浮点误差影响，提升了攻击效果与鲁棒性评估准确性。**

- **链接: [http://arxiv.org/pdf/2507.22428v1](http://arxiv.org/pdf/2507.22428v1)**

> **作者:** Yunrui Yu; Hang Su; Cheng-zhong Xu; Zhizhong Su; Jun Zhu
>
> **摘要:** Gradient-based adversarial attacks using the Cross-Entropy (CE) loss often suffer from overestimation due to relative errors in gradient computation induced by floating-point arithmetic. This paper provides a rigorous theoretical analysis of these errors, conducting the first comprehensive study of floating-point computation errors in gradient-based attacks across four distinct scenarios: (i) unsuccessful untargeted attacks, (ii) successful untargeted attacks, (iii) unsuccessful targeted attacks, and (iv) successful targeted attacks. We establish theoretical foundations characterizing the behavior of relative numerical errors under different attack conditions, revealing previously unknown patterns in gradient computation instability, and identify floating-point underflow and rounding as key contributors. Building on this insight, we propose the Theoretical MIFPE (T-MIFPE) loss function, which incorporates an optimal scaling factor $T = t^*$ to minimize the impact of floating-point errors, thereby enhancing the accuracy of gradient computation in adversarial attacks. Extensive experiments on the MNIST, CIFAR-10, and CIFAR-100 datasets demonstrate that T-MIFPE outperforms existing loss functions, including CE, C\&W, DLR, and MIFPE, in terms of attack potency and robustness evaluation accuracy.
>
---
#### [new 080] Learned Off-aperture Encoding for Wide Field-of-view RGBD Imaging
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于光学成像任务，旨在解决宽视场成像中图像质量下降的问题。通过将衍射光学元件（DOE）置于非孔径位置，实现对波前的局部控制，结合折射-衍射混合光学系统，优化深度和彩色成像质量。实验验证了该方法在宽视场下成像性能的提升。**

- **链接: [http://arxiv.org/pdf/2507.22523v1](http://arxiv.org/pdf/2507.22523v1)**

> **作者:** Haoyu Wei; Xin Liu; Yuhui Liu; Qiang Fu; Wolfgang Heidrich; Edmund Y. Lam; Yifan Peng
>
> **备注:** To be published in IEEE Transactions on Pattern Analysis and Machine Intelligence
>
> **摘要:** End-to-end (E2E) designed imaging systems integrate coded optical designs with decoding algorithms to enhance imaging fidelity for diverse visual tasks. However, existing E2E designs encounter significant challenges in maintaining high image fidelity at wide fields of view, due to high computational complexity, as well as difficulties in modeling off-axis wave propagation while accounting for off-axis aberrations. In particular, the common approach of placing the encoding element into the aperture or pupil plane results in only a global control of the wavefront. To overcome these limitations, this work explores an additional design choice by positioning a DOE off-aperture, enabling a spatial unmixing of the degrees of freedom and providing local control over the wavefront over the image plane. Our approach further leverages hybrid refractive-diffractive optical systems by linking differentiable ray and wave optics modeling, thereby optimizing depth imaging quality and demonstrating system versatility. Experimental results reveal that the off-aperture DOE enhances the imaging quality by over 5 dB in PSNR at a FoV of approximately $45^\circ$ when paired with a simple thin lens, outperforming traditional on-aperture systems. Furthermore, we successfully recover color and depth information at nearly $28^\circ$ FoV using off-aperture DOE configurations with compound optics. Physical prototypes for both applications validate the effectiveness and versatility of the proposed method.
>
---
#### [new 081] Whole-brain Transferable Representations from Large-Scale fMRI Data Improve Task-Evoked Brain Activity Decoding
- **分类: eess.IV; cs.CV**

- **简介: 该论文旨在通过迁移学习提升功能性磁共振成像（fMRI）中任务诱发脑活动的解码能力。为解决fMRI数据高维、低信噪比和数据量少的问题，作者提出了STDA-SwiFT模型，结合时空注意力机制和自监督对比学习，从大规模fMRI数据中学习可迁移的表征，并显著提升了多种感知和认知任务的解码性能。**

- **链接: [http://arxiv.org/pdf/2507.22378v1](http://arxiv.org/pdf/2507.22378v1)**

> **作者:** Yueh-Po Peng; Vincent K. M. Cheung; Li Su
>
> **摘要:** A fundamental challenge in neuroscience is to decode mental states from brain activity. While functional magnetic resonance imaging (fMRI) offers a non-invasive approach to capture brain-wide neural dynamics with high spatial precision, decoding from fMRI data -- particularly from task-evoked activity -- remains challenging due to its high dimensionality, low signal-to-noise ratio, and limited within-subject data. Here, we leverage recent advances in computer vision and propose STDA-SwiFT, a transformer-based model that learns transferable representations from large-scale fMRI datasets via spatial-temporal divided attention and self-supervised contrastive learning. Using pretrained voxel-wise representations from 995 subjects in the Human Connectome Project (HCP), we show that our model substantially improves downstream decoding performance of task-evoked activity across multiple sensory and cognitive domains, even with minimal data preprocessing. We demonstrate performance gains from larger receptor fields afforded by our memory-efficient attention mechanism, as well as the impact of functional relevance in pretraining data when fine-tuning on small samples. Our work showcases transfer learning as a viable approach to harness large-scale datasets to overcome challenges in decoding brain activity from fMRI data.
>
---
#### [new 082] A Segmentation Framework for Accurate Diagnosis of Amyloid Positivity without Structural Images
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决仅使用PET图像准确分割脑区并判断淀粉样蛋白阳性的问题。作者采用3D U-Net模型，基于200例PET数据训练，实现了高精度脑区分割与淀粉样蛋白阳性分类，准确率达98%，AUC为0.99，展示了无需结构影像的诊断潜力。**

- **链接: [http://arxiv.org/pdf/2507.22336v1](http://arxiv.org/pdf/2507.22336v1)**

> **作者:** Penghan Zhu; Shurui Mei; Shushan Chen; Xiaobo Chu; Shanbo He; Ziyi Liu
>
> **摘要:** This study proposes a deep learning-based framework for automated segmentation of brain regions and classification of amyloid positivity using positron emission tomography (PET) images alone, without the need for structural MRI or CT. A 3D U-Net architecture with four layers of depth was trained and validated on a dataset of 200 F18-florbetapir amyloid-PET scans, with an 130/20/50 train/validation/test split. Segmentation performance was evaluated using Dice similarity coefficients across 30 brain regions, with scores ranging from 0.45 to 0.88, demonstrating high anatomical accuracy, particularly in subcortical structures. Quantitative fidelity of PET uptake within clinically relevant regions. Precuneus, prefrontal cortex, gyrus rectus, and lateral temporal cortex was assessed using normalized root mean square error, achieving values as low as 0.0011. Furthermore, the model achieved a classification accuracy of 0.98 for amyloid positivity based on regional uptake quantification, with an area under the ROC curve (AUC) of 0.99. These results highlight the model's potential for integration into PET only diagnostic pipelines, particularly in settings where structural imaging is not available. This approach reduces dependence on coregistration and manual delineation, enabling scalable, reliable, and reproducible analysis in clinical and research applications. Future work will focus on clinical validation and extension to diverse PET tracers including C11 PiB and other F18 labeled compounds.
>
---
#### [new 083] Mesh based segmentation for automated margin line generation on incisors receiving crown treatment
- **分类: cs.CE; cs.CV; cs.LG**

- **简介: 该论文属于医学图像处理任务，旨在解决牙冠修复中手动标注牙体边缘线不一致的问题。作者提出了一种基于深度学习的自动边缘线生成方法，使用网格神经网络进行分割，结合交叉验证、投票分类、图割优化等技术，最终通过样条拟合预测边缘线，提高了牙冠设计的自动化与一致性。**

- **链接: [http://arxiv.org/pdf/2507.22859v1](http://arxiv.org/pdf/2507.22859v1)**

> **作者:** Ammar Alsheghri; Ying Zhang; Farnoosh Ghadiri; Julia Keren; Farida Cheriet; Francois Guibault
>
> **摘要:** Dental crowns are essential dental treatments for restoring damaged or missing teeth of patients. Recent design approaches of dental crowns are carried out using commercial dental design software. Once a scan of a preparation is uploaded to the software, a dental technician needs to manually define a precise margin line on the preparation surface, which constitutes a non-repeatable and inconsistent procedure. This work proposes a new framework to determine margin lines automatically and accurately using deep learning. A dataset of incisor teeth was provided by a collaborating dental laboratory to train a deep learning segmentation model. A mesh-based neural network was modified by changing its input channels and used to segment the prepared tooth into two regions such that the margin line is contained within the boundary faces separating the two regions. Next, k-fold cross-validation was used to train 5 models, and a voting classifier technique was used to combine their results to enhance the segmentation. After that, boundary smoothing and optimization using the graph cut method were applied to refine the segmentation results. Then, boundary faces separating the two regions were selected to represent the margin line faces. A spline was approximated to best fit the centers of the boundary faces to predict the margin line. Our results show that an ensemble model combined with maximum probability predicted the highest number of successful test cases (7 out of 13) based on a maximum distance threshold of 200 m (representing human error) between the predicted and ground truth point clouds. It was also demonstrated that the better the quality of the preparation, the smaller the divergence between the predicted and ground truth margin lines (Spearman's rank correlation coefficient of -0.683). We provide the train and test datasets for the community.
>
---
#### [new 084] Tapping into the Black Box: Uncovering Aligned Representations in Pretrained Neural Networks
- **分类: cs.LG; cs.CV; cs.NE; I.2.6; I.4.10**

- **简介: 该论文属于计算机视觉与深度学习可解释性任务，旨在解决神经网络“黑箱”问题。作者提出通过修改反向传播过程，提取ReLU网络中的隐式线性模型决策边界，生成高分辨率、与感知对齐的特征图。该方法揭示了网络依赖的可解释模式，有助于知识发现与构建可信AI系统。**

- **链接: [http://arxiv.org/pdf/2507.22832v1](http://arxiv.org/pdf/2507.22832v1)**

> **作者:** Maciej Satkiewicz
>
> **备注:** 15 pages, 4 figures, preprint
>
> **摘要:** In this paper we argue that ReLU networks learn an implicit linear model we can actually tap into. We describe that alleged model formally and show that we can approximately pull its decision boundary back to the input space with certain simple modification to the backward pass. The resulting gradients (called excitation pullbacks) reveal high-resolution input- and target-specific features of remarkable perceptual alignment on a number of popular ImageNet-pretrained deep architectures. This strongly suggests that neural networks do, in fact, rely on learned interpretable patterns that can be recovered after training. Thus, our findings may have profound implications for knowledge discovery and the development of dependable artificial systems.
>
---
## 更新

#### [replaced 001] Counting Stacked Objects
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.19149v4](http://arxiv.org/pdf/2411.19149v4)**

> **作者:** Corentin Dumery; Noa Etté; Aoxiang Fan; Ren Li; Jingyi Xu; Hieu Le; Pascal Fua
>
> **备注:** ICCV25 Oral. Datasets and code can be found at https://corentindumery.github.io/projects/stacks.html
>
> **摘要:** Visual object counting is a fundamental computer vision task underpinning numerous real-world applications, from cell counting in biomedicine to traffic and wildlife monitoring. However, existing methods struggle to handle the challenge of stacked 3D objects in which most objects are hidden by those above them. To address this important yet underexplored problem, we propose a novel 3D counting approach that decomposes the task into two complementary subproblems - estimating the 3D geometry of the object stack and the occupancy ratio from multi-view images. By combining geometric reconstruction and deep learning-based depth analysis, our method can accurately count identical objects within containers, even when they are irregularly stacked. We validate our 3D Counting pipeline on diverse real-world and large-scale synthetic datasets, which we will release publicly to facilitate further research.
>
---
#### [replaced 002] FloPE: Flower Pose Estimation for Precision Pollination
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11692v2](http://arxiv.org/pdf/2503.11692v2)**

> **作者:** Rashik Shrestha; Madhav Rijal; Trevor Smith; Yu Gu
>
> **备注:** Accepted to IROS 2025. Project page: https://wvu-irl.github.io/flope-irl/
>
> **摘要:** This study presents Flower Pose Estimation (FloPE), a real-time flower pose estimation framework for computationally constrained robotic pollination systems. Robotic pollination has been proposed to supplement natural pollination to ensure global food security due to the decreased population of natural pollinators. However, flower pose estimation for pollination is challenging due to natural variability, flower clusters, and high accuracy demands due to the flowers' fragility when pollinating. This method leverages 3D Gaussian Splatting to generate photorealistic synthetic datasets with precise pose annotations, enabling effective knowledge distillation from a high-capacity teacher model to a lightweight student model for efficient inference. The approach was evaluated on both single and multi-arm robotic platforms, achieving a mean pose estimation error of 0.6 cm and 19.14 degrees within a low computational cost. Our experiments validate the effectiveness of FloPE, achieving up to 78.75% pollination success rate and outperforming prior robotic pollination techniques.
>
---
#### [replaced 003] PARTE: Part-Guided Texturing for 3D Human Reconstruction from a Single Image
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.17332v4](http://arxiv.org/pdf/2507.17332v4)**

> **作者:** Hyeongjin Nam; Donghwan Kim; Gyeongsik Moon; Kyoung Mu Lee
>
> **备注:** Published at ICCV 2025, 22 pages including the supplementary material
>
> **摘要:** The misaligned human texture across different human parts is one of the main limitations of existing 3D human reconstruction methods. Each human part, such as a jacket or pants, should maintain a distinct texture without blending into others. The structural coherence of human parts serves as a crucial cue to infer human textures in the invisible regions of a single image. However, most existing 3D human reconstruction methods do not explicitly exploit such part segmentation priors, leading to misaligned textures in their reconstructions. In this regard, we present PARTE, which utilizes 3D human part information as a key guide to reconstruct 3D human textures. Our framework comprises two core components. First, to infer 3D human part information from a single image, we propose a 3D part segmentation module (PartSegmenter) that initially reconstructs a textureless human surface and predicts human part labels based on the textureless surface. Second, to incorporate part information into texture reconstruction, we introduce a part-guided texturing module (PartTexturer), which acquires prior knowledge from a pre-trained image generation network on texture alignment of human parts. Extensive experiments demonstrate that our framework achieves state-of-the-art quality in 3D human reconstruction. The project page is available at https://hygenie1228.github.io/PARTE/.
>
---
#### [replaced 004] Equivariant Flow Matching for Point Cloud Assembly
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.21539v2](http://arxiv.org/pdf/2505.21539v2)**

> **作者:** Ziming Wang; Nan Xue; Rebecka Jörnsten
>
> **摘要:** The goal of point cloud assembly is to reconstruct a complete 3D shape by aligning multiple point cloud pieces. This work presents a novel equivariant solver for assembly tasks based on flow matching models. We first theoretically show that the key to learning equivariant distributions via flow matching is to learn related vector fields. Based on this result, we propose an assembly model, called equivariant diffusion assembly (Eda), which learns related vector fields conditioned on the input pieces. We further construct an equivariant path for Eda, which guarantees high data efficiency of the training process. Our numerical results show that Eda is highly competitive on practical datasets, and it can even handle the challenging situation where the input pieces are non-overlapped.
>
---
#### [replaced 005] Metric Convolutions: A Unifying Theory to Adaptive Image Convolutions
- **分类: cs.CV; math.DG**

- **链接: [http://arxiv.org/pdf/2406.05400v2](http://arxiv.org/pdf/2406.05400v2)**

> **作者:** Thomas Dagès; Michael Lindenbaum; Alfred M. Bruckstein
>
> **备注:** Updated version, Accepted for publication at the IEEE/CVF International Conference on Computer Vision (ICCV) 2025
>
> **摘要:** Standard convolutions are prevalent in image processing and deep learning, but their fixed kernels limits adaptability. Several deformation strategies of the reference kernel grid have been proposed. Yet, they lack a unified theoretical framework. By returning to a metric perspective for images, now seen as two-dimensional manifolds equipped with notions of local and geodesic distances, either symmetric (Riemannian) or not (Finsler), we provide a unifying principle: the kernel positions are samples of unit balls of implicit metrics. With this new perspective, we also propose metric convolutions, a novel approach that samples unit balls from explicit signal-dependent metrics, providing interpretable operators with geometric regularisation. This framework, compatible with gradient-based optimisation, can directly replace existing convolutions applied to either input images or deep features of neural networks. Metric convolutions typically require fewer parameters and provide better generalisation. Our approach shows competitive performance in standard denoising and classification tasks.
>
---
#### [replaced 006] Beyond Image Prior: Embedding Noise Prior into Conditional Denoising Transformer
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2407.09094v2](http://arxiv.org/pdf/2407.09094v2)**

> **作者:** Yuanfei Huang; Hua Huang
>
> **备注:** Accepted by International Journal of Computer Vision (IJCV)
>
> **摘要:** Existing learning-based denoising methods typically train models to generalize the image prior from large-scale datasets, suffering from the variability in noise distributions encountered in real-world scenarios. In this work, we propose a new perspective on the denoising challenge by highlighting the distinct separation between noise and image priors. This insight forms the basis for our development of conditional optimization framework, designed to overcome the constraints of traditional denoising framework. To this end, we introduce a Locally Noise Prior Estimation (LoNPE) algorithm, which accurately estimates the noise prior directly from a single raw noisy image. This estimation acts as an explicit prior representation of the camera sensor's imaging environment, distinct from the image prior of scenes. Additionally, we design an auxiliary learnable LoNPE network tailored for practical application to sRGB noisy images. Leveraging the estimated noise prior, we present a novel Conditional Denoising Transformer (Condformer), by incorporating the noise prior into a conditional self-attention mechanism. This integration allows the Condformer to segment the optimization process into multiple explicit subspaces, significantly enhancing the model's generalization and flexibility. Extensive experimental evaluations on both synthetic and real-world datasets, demonstrate that the proposed method achieves superior performance over current state-of-the-art methods. The source code is available at https://github.com/YuanfeiHuang/Condformer.
>
---
#### [replaced 007] Diffusion-based Adversarial Identity Manipulation for Facial Privacy Protection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.21646v3](http://arxiv.org/pdf/2504.21646v3)**

> **作者:** Liqin Wang; Qianyue Hu; Wei Lu; Xiangyang Luo
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** The success of face recognition (FR) systems has led to serious privacy concerns due to potential unauthorized surveillance and user tracking on social networks. Existing methods for enhancing privacy fail to generate natural face images that can protect facial privacy. In this paper, we propose diffusion-based adversarial identity manipulation (DiffAIM) to generate natural and highly transferable adversarial faces against malicious FR systems. To be specific, we manipulate facial identity within the low-dimensional latent space of a diffusion model. This involves iteratively injecting gradient-based adversarial identity guidance during the reverse diffusion process, progressively steering the generation toward the desired adversarial faces. The guidance is optimized for identity convergence towards a target while promoting semantic divergence from the source, facilitating effective impersonation while maintaining visual naturalness. We further incorporate structure-preserving regularization to preserve facial structure consistency during manipulation. Extensive experiments on both face verification and identification tasks demonstrate that compared with the state-of-the-art, DiffAIM achieves stronger black-box attack transferability while maintaining superior visual quality. We also demonstrate the effectiveness of the proposed approach for commercial FR APIs, including Face++ and Aliyun.
>
---
#### [replaced 008] Learning Only with Images: Visual Reinforcement Learning with Reasoning, Rendering, and Visual Feedback
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.20766v2](http://arxiv.org/pdf/2507.20766v2)**

> **作者:** Yang Chen; Yufan Shen; Wenxuan Huang; Sheng Zhou; Qunshu Lin; Xinyu Cai; Zhi Yu; Jiajun Bu; Botian Shi; Yu Qiao
>
> **摘要:** Multimodal Large Language Models (MLLMs) exhibit impressive performance across various visual tasks. Subsequent investigations into enhancing their visual reasoning abilities have significantly expanded their performance envelope. However, a critical bottleneck in the advancement of MLLMs toward deep visual reasoning is their heavy reliance on curated image-text supervision. To solve this problem, we introduce a novel framework termed ``Reasoning-Rendering-Visual-Feedback'' (RRVF), which enables MLLMs to learn complex visual reasoning from only raw images. This framework builds on the ``Asymmetry of Verification'' principle to train MLLMs, i.e., verifying the rendered output against a source image is easier than generating it. We demonstrate that this relative ease provides an ideal reward signal for optimization via Reinforcement Learning (RL) training, reducing reliance on the image-text supervision. Guided by the above principle, RRVF implements a closed-loop iterative process encompassing reasoning, rendering, and visual feedback components, enabling the model to perform self-correction through multi-turn interactions, while this pipeline can be optimized end-to-end by the GRPO algorithm. Extensive evaluations are conducted on image-to-code generation across two diverse domains: data charts and web interfaces. The RRVF-trained model not only outperforms existing open-source MLLMs and supervised fine-tuning baselines but also exhibits superior generalization to unseen datasets. Critically, the model's performance surpasses that of the more advanced MLLM used to provide the feedback signal during training. This work establishes a self-improvement paradigm that offers a viable path to robust, generalizable models without reliance on explicit supervision. Code will be available at https://github.com/L-O-I/RRVF.
>
---
#### [replaced 009] AstroLoc: Robust Space to Ground Image Localizer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.07003v2](http://arxiv.org/pdf/2502.07003v2)**

> **作者:** Gabriele Berton; Alex Stoken; Carlo Masone
>
> **备注:** https://astro-loc.github.io/
>
> **摘要:** Astronauts take thousands of photos of Earth per day from the International Space Station, which, once localized on Earth's surface, are used for a multitude of tasks, ranging from climate change research to disaster management. The localization process, which has been performed manually for decades, has recently been approached through image retrieval solutions: given an astronaut photo, find its most similar match among a large database of geo-tagged satellite images, in a task called Astronaut Photography Localization (APL). Yet, existing APL approaches are trained only using satellite images, without taking advantage of the millions open-source astronaut photos. In this work we present the first APL pipeline capable of leveraging astronaut photos for training. We first produce full localization information for 300,000 manually weakly labeled astronaut photos through an automated pipeline, and then use these images to train a model, called AstroLoc. AstroLoc learns a robust representation of Earth's surface features through two losses: astronaut photos paired with their matching satellite counterparts in a pairwise loss, and a second loss on clusters of satellite imagery weighted by their relevance to astronaut photography via unsupervised mining. We find that AstroLoc achieves a staggering 35% average improvement in recall@1 over previous SOTA, pushing the limits of existing datasets with a recall@100 consistently over 99%. Finally, we note that AstroLoc, without any fine-tuning, provides excellent results for related tasks like the lost-in-space satellite problem and historical space imagery localization.
>
---
#### [replaced 010] TartanGround: A Large-Scale Dataset for Ground Robot Perception and Navigation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10696v2](http://arxiv.org/pdf/2505.10696v2)**

> **作者:** Manthan Patel; Fan Yang; Yuheng Qiu; Cesar Cadena; Sebastian Scherer; Marco Hutter; Wenshan Wang
>
> **备注:** Accepted for publication to IEEE/RSJ IROS 2025
>
> **摘要:** We present TartanGround, a large-scale, multi-modal dataset to advance the perception and autonomy of ground robots operating in diverse environments. This dataset, collected in various photorealistic simulation environments includes multiple RGB stereo cameras for 360-degree coverage, along with depth, optical flow, stereo disparity, LiDAR point clouds, ground truth poses, semantic segmented images, and occupancy maps with semantic labels. Data is collected using an integrated automatic pipeline, which generates trajectories mimicking the motion patterns of various ground robot platforms, including wheeled and legged robots. We collect 910 trajectories across 70 environments, resulting in 1.5 million samples. Evaluations on occupancy prediction and SLAM tasks reveal that state-of-the-art methods trained on existing datasets struggle to generalize across diverse scenes. TartanGround can serve as a testbed for training and evaluation of a broad range of learning-based tasks, including occupancy prediction, SLAM, neural scene representation, perception-based navigation, and more, enabling advancements in robotic perception and autonomy towards achieving robust models generalizable to more diverse scenarios. The dataset and codebase are available on the webpage: https://tartanair.org/tartanground
>
---
#### [replaced 011] When Tokens Talk Too Much: A Survey of Multimodal Long-Context Token Compression across Images, Videos, and Audios
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.20198v3](http://arxiv.org/pdf/2507.20198v3)**

> **作者:** Kele Shao; Keda Tao; Kejia Zhang; Sicheng Feng; Mu Cai; Yuzhang Shang; Haoxuan You; Can Qin; Yang Sui; Huan Wang
>
> **备注:** For ongoing updates and to track the latest advances in this promising area, we maintain a public repository: https://github.com/cokeshao/Awesome-Multimodal-Token-Compression
>
> **摘要:** Multimodal large language models (MLLMs) have made remarkable strides, largely driven by their ability to process increasingly long and complex contexts, such as high-resolution images, extended video sequences, and lengthy audio input. While this ability significantly enhances MLLM capabilities, it introduces substantial computational challenges, primarily due to the quadratic complexity of self-attention mechanisms with numerous input tokens. To mitigate these bottlenecks, token compression has emerged as an auspicious and critical approach, efficiently reducing the number of tokens during both training and inference. In this paper, we present the first systematic survey and synthesis of the burgeoning field of multimodal long context token compression. Recognizing that effective compression strategies are deeply tied to the unique characteristics and redundancies of each modality, we categorize existing approaches by their primary data focus, enabling researchers to quickly access and learn methods tailored to their specific area of interest: (1) image-centric compression, which addresses spatial redundancy in visual data; (2) video-centric compression, which tackles spatio-temporal redundancy in dynamic sequences; and (3) audio-centric compression, which handles temporal and spectral redundancy in acoustic signals. Beyond this modality-driven categorization, we further dissect methods based on their underlying mechanisms, including transformation-based, similarity-based, attention-based, and query-based approaches. By providing a comprehensive and structured overview, this survey aims to consolidate current progress, identify key challenges, and inspire future research directions in this rapidly evolving domain. We also maintain a public repository to continuously track and update the latest advances in this promising area.
>
---
#### [replaced 012] Enhancing Multimodal In-Context Learning for Image Classification through Coreset Optimization
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.14200v2](http://arxiv.org/pdf/2504.14200v2)**

> **作者:** Huiyi Chen; Jiawei Peng; Kaihua Tang; Xin Geng; Xu Yang
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** In-context learning (ICL) enables Large Vision-Language Models (LVLMs) to adapt to new tasks without parameter updates, using a few demonstrations from a large support set. However, selecting informative demonstrations leads to high computational and memory costs. While some methods explore selecting a small and representative coreset in the text classification, evaluating all support set samples remains costly, and discarded samples lead to unnecessary information loss. These methods may also be less effective for image classification due to differences in feature spaces. Given these limitations, we propose Key-based Coreset Optimization (KeCO), a novel framework that leverages untapped data to construct a compact and informative coreset. We introduce visual features as keys within the coreset, which serve as the anchor for identifying samples to be updated through different selection strategies. By leveraging untapped samples from the support set, we update the keys of selected coreset samples, enabling the randomly initialized coreset to evolve into a more informative coreset under low computational cost. Through extensive experiments on coarse-grained and fine-grained image classification benchmarks, we demonstrate that KeCO effectively enhances ICL performance for image classification task, achieving an average improvement of more than 20\%. Notably, we evaluate KeCO under a simulated online scenario, and the strong performance in this scenario highlights the practical value of our framework for resource-constrained real-world scenarios.
>
---
#### [replaced 013] Exploring the Frontier of Vision-Language Models: A Survey of Current Methodologies and Future Directions
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2404.07214v3](http://arxiv.org/pdf/2404.07214v3)**

> **作者:** Akash Ghosh; Arkadeep Acharya; Sriparna Saha; Vinija Jain; Aman Chadha
>
> **备注:** One of the first survey on Visual Language Models
>
> **摘要:** The advent of Large Language Models (LLMs) has significantly reshaped the trajectory of the AI revolution. Nevertheless, these LLMs exhibit a notable limitation, as they are primarily adept at processing textual information. To address this constraint, researchers have endeavored to integrate visual capabilities with LLMs, resulting in the emergence of Vision-Language Models (VLMs). These advanced models are instrumental in tackling more intricate tasks such as image captioning and visual question answering. In our comprehensive survey paper, we delve into the key advancements within the realm of VLMs. Our classification organizes VLMs into three distinct categories: models dedicated to vision-language understanding, models that process multimodal inputs to generate unimodal (textual) outputs and models that both accept and produce multimodal inputs and outputs.This classification is based on their respective capabilities and functionalities in processing and generating various modalities of data.We meticulously dissect each model, offering an extensive analysis of its foundational architecture, training data sources, as well as its strengths and limitations wherever possible, providing readers with a comprehensive understanding of its essential components. We also analyzed the performance of VLMs in various benchmark datasets. By doing so, we aim to offer a nuanced understanding of the diverse landscape of VLMs. Additionally, we underscore potential avenues for future research in this dynamic domain, anticipating further breakthroughs and advancements.
>
---
#### [replaced 014] CLIP-IT: CLIP-based Pairing for Histology Images Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.16181v3](http://arxiv.org/pdf/2504.16181v3)**

> **作者:** Banafsheh Karimian; Giulia Avanzato; Soufian Belharbi; Alexis Guichemerre; Luke McCaffrey; Mohammadhadi Shateri; Eric Granger
>
> **摘要:** Multimodal learning has shown promise in medical image analysis, combining complementary modalities like histology images and text. Vision-language models (VLMs) capture rich diagnostic cues but often require large paired datasets and prompt- or text-based inference, limiting their practicality due to annotation cost, privacy, and compute demands. Crucially, available free unpaired external text, like pathology reports, can still provide complementary diagnostic cues if semantically relevant content is retrievable per image. To address this, we introduce CLIP-IT, a novel framework that relies on rich unpaired text reports, eliminating paired data requirement. Specifically, CLIP-IT uses a CLIP model pre-trained on histology image-text pairs from a separate dataset to retrieve the most relevant unpaired textual report for each image in the target unimodal dataset. These reports, sourced from the same disease domain and tissue type, form pseudo-pairs that reflect shared clinical semantics rather than exact alignment. Knowledge from these texts is distilled into the vision model during training, while LoRA-based adaptation mitigates the semantic gap between unaligned modalities. At inference time, only the improved vision model is used, with minimal computational overhead, enabling efficient pairing-free multimodal deployment. Experiments on histology image datasets confirm that CLIP-IT consistently improves classification accuracy over both unimodal and multimodal CLIP-based baselines in most cases, without the burden of paired data training or inference-time complexity.
>
---
#### [replaced 015] SpatialViz-Bench: Automatically Generated Spatial Visualization Reasoning Tasks for MLLMs
- **分类: cs.CV; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2507.07610v3](http://arxiv.org/pdf/2507.07610v3)**

> **作者:** Siting Wang; Luoyang Sun; Cheng Deng; Kun Shao; Minnan Pei; Zheng Tian; Haifeng Zhang; Jun Wang
>
> **摘要:** Humans can directly imagine and manipulate visual images in their minds, a capability known as spatial visualization. While multi-modal Large Language Models (MLLMs) support imagination-based reasoning, spatial visualization remains insufficiently evaluated, typically embedded within broader mathematical and logical assessments. Existing evaluations often rely on IQ tests or math competitions that may overlap with training data, compromising assessment reliability. To this end, we introduce SpatialViz-Bench, a comprehensive multi-modal benchmark for spatial visualization with 12 tasks across 4 sub-abilities, comprising 1,180 automatically generated problems. Our evaluation of 33 state-of-the-art MLLMs not only reveals wide performance variations and demonstrates the benchmark's strong discriminative power, but also uncovers counter-intuitive findings: models show difficulty perception misaligned with human intuition, exhibit dramatic 2Dto-3D performance cliffs, default to formulaic derivation over visualization, and paradoxically suffer performance degradation from Chain-of-Thought prompting in open-source models. Through statistical and qualitative analysis of error types, SpatialViz-Bench demonstrates that state-of-the-art MLLMs continue to exhibit deficiencies in spatial visualization tasks, thereby addressing a significant lacuna in the field. The benchmark data and evaluation code are publicly available.
>
---
#### [replaced 016] StoryTeller: Improving Long Video Description through Global Audio-Visual Character Identification
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.07076v3](http://arxiv.org/pdf/2411.07076v3)**

> **作者:** Yichen He; Yuan Lin; Jianchao Wu; Hanchong Zhang; Yuchen Zhang; Ruicheng Le
>
> **摘要:** Existing large vision-language models (LVLMs) are largely limited to processing short, seconds-long videos and struggle with generating coherent descriptions for extended video spanning minutes or more. Long video description introduces new challenges, such as consistent character identification and plot-level descriptions incorporating both visual and audio information. To address these, we figure out audio-visual character identification, matching character names to each dialogue, as a key factor. We propose StoryTeller, a system for generating dense descriptions of long videos, incorporating both low-level visual concepts and high-level plot information. StoryTeller uses a multimodal large language model that integrates visual, audio, and text modalities to perform audio-visual character identification on minute-long video clips. The results are then fed into a LVLM to enhance consistency of video description. We validate our approach on movie description tasks and introduce MovieStory101, a dataset with dense descriptions for three-minute movie clips. To evaluate long video descriptions, we create StoryQA, a large set of multiple-choice questions for MovieStory101 test set. We assess descriptions by inputting them into GPT-4 to answer these questions, using accuracy as an automatic evaluation metric. Experiments show that StoryTeller outperforms all open and closed-source baselines on StoryQA, achieving 9.5% higher accuracy than the strongest baseline, Gemini-1.5-pro, and demonstrating a +15.56% advantage in human side-by-side evaluations. Additionally, incorporating audio-visual character identification from StoryTeller improves the performance of all video description models, with Gemini-1.5-pro and GPT-4o showing relative improvement of 5.5% and 13.0%, respectively, in accuracy on StoryQA.
>
---
#### [replaced 017] Clinical Utility of Foundation Segmentation Models in Musculoskeletal MRI: Biomarker Fidelity and Predictive Outcomes
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.13376v2](http://arxiv.org/pdf/2501.13376v2)**

> **作者:** Gabrielle Hoyer; Michelle W Tong; Rupsa Bhattacharjee; Valentina Pedoia; Sharmila Majumdar
>
> **备注:** Code repository: https://github.com/gabbieHoyer/AutoMedLabel; Supplementary data and tables: https://doi.org/10.6084/m9.figshare.29633207. This submission replaces an earlier draft titled "Scalable Evaluation Framework for Foundation Models in MSK MRI."
>
> **摘要:** Effective segmentation is fundamental for quantitative medical imaging; however, foundation segmentation models remain insufficiently evaluated for accuracy and biomarker fidelity across the diverse anatomical contexts and imaging protocols encountered in musculoskeletal (MSK) MRI. We evaluate three widely used segmentation models (SAM, SAM2, MedSAM) across eleven MSK MRI datasets spanning the knee, hip, spine, shoulder, and thigh. Our framework assesses both zero-shot and finetuned performance, with attention to segmentation accuracy, generalizability across imaging protocols, and reliability of derived quantitative biomarkers. Finetuned models showed consistent agreement with expert measurements for biomarkers including cartilage thickness, disc height, muscle volume, and compositional T1rho/T2 values. Automated prompting through the AutoLabel system enabled scalable segmentation, with moderate trade-offs in accuracy. As proof of concept, we applied the validated system to (i) a three-stage knee MRI triage cascade and (ii) a longitudinal landmark model that predicts total knee replacement and incident osteoarthritis. The framework offers a transparent method for benchmarking segmentation tools and connecting model performance to clinical imaging priorities.
>
---
#### [replaced 018] Collaborative Perceiver: Elevating Vision-based 3D Object Detection via Local Density-Aware Spatial Occupancy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.21358v2](http://arxiv.org/pdf/2507.21358v2)**

> **作者:** Jicheng Yuan; Manh Nguyen Duc; Qian Liu; Manfred Hauswirth; Danh Le Phuoc
>
> **备注:** The manuscript has been accepted by ICONIP2025
>
> **摘要:** Vision-based bird's-eye-view (BEV) 3D object detection has advanced significantly in autonomous driving by offering cost-effectiveness and rich contextual information. However, existing methods often construct BEV representations by collapsing extracted object features, neglecting intrinsic environmental contexts, such as roads and pavements. This hinders detectors from comprehensively perceiving the characteristics of the physical world. To alleviate this, we introduce a multi-task learning framework, Collaborative Perceiver (CoP), that leverages spatial occupancy as auxiliary information to mine consistent structural and conceptual similarities shared between 3D object detection and occupancy prediction tasks, bridging gaps in spatial representations and feature refinement. To this end, we first propose a pipeline to generate dense occupancy ground truths incorporating local density information (LDO) for reconstructing detailed environmental information. Next, we employ a voxel-height-guided sampling (VHS) strategy to distill fine-grained local features according to distinct object properties. Furthermore, we develop a global-local collaborative feature fusion (CFF) module that seamlessly integrates complementary knowledge between both tasks, thus composing more robust BEV representations. Extensive experiments on the nuScenes benchmark demonstrate that CoP outperforms existing vision-based frameworks, achieving 49.5\% mAP and 59.2\% NDS on the test set. Code and supplementary materials are available at this link https://github.com/jichengyuan/Collaborative-Perceiver.
>
---
#### [replaced 019] DMCIE: Diffusion Model with Concatenation of Inputs and Errors to Improve the Accuracy of the Segmentation of Brain Tumors in MRI Images
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.00983v2](http://arxiv.org/pdf/2507.00983v2)**

> **作者:** Sara Yavari; Rahul Nitin Pandya; Jacob Furst
>
> **摘要:** Accurate segmentation of brain tumors in MRI scans is essential for reliable clinical diagnosis and effective treatment planning. Recently, diffusion models have demonstrated remarkable effectiveness in image generation and segmentation tasks. This paper introduces a novel approach to corrective segmentation based on diffusion models. We propose DMCIE (Diffusion Model with Concatenation of Inputs and Errors), a novel framework for accurate brain tumor segmentation in multi-modal MRI scans. We employ a 3D U-Net to generate an initial segmentation mask, from which an error map is generated by identifying the differences between the prediction and the ground truth. The error map, concatenated with the original MRI images, are used to guide a diffusion model. Using multimodal MRI inputs (T1, T1ce, T2, FLAIR), DMCIE effectively enhances segmentation accuracy by focusing on misclassified regions, guided by the original inputs. Evaluated on the BraTS2020 dataset, DMCIE outperforms several state-of-the-art diffusion-based segmentation methods, achieving a Dice Score of 93.46 and an HD95 of 5.94 mm. These results highlight the effectiveness of error-guided diffusion in producing precise and reliable brain tumor segmentations.
>
---
#### [replaced 020] SteerX: Creating Any Camera-Free 3D and 4D Scenes with Geometric Steering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.12024v2](http://arxiv.org/pdf/2503.12024v2)**

> **作者:** Byeongjun Park; Hyojun Go; Hyelin Nam; Byung-Hoon Kim; Hyungjin Chung; Changick Kim
>
> **备注:** Project page: https://byeongjun-park.github.io/SteerX/
>
> **摘要:** Recent progress in 3D/4D scene generation emphasizes the importance of physical alignment throughout video generation and scene reconstruction. However, existing methods improve the alignment separately at each stage, making it difficult to manage subtle misalignments arising from another stage. Here, we present SteerX, a zero-shot inference-time steering method that unifies scene reconstruction into the generation process, tilting data distributions toward better geometric alignment. To this end, we introduce two geometric reward functions for 3D/4D scene generation by using pose-free feed-forward scene reconstruction models. Through extensive experiments, we demonstrate the effectiveness of SteerX in improving 3D/4D scene generation.
>
---
#### [replaced 021] UI-E2I-Synth: Advancing GUI Grounding with Large-Scale Instruction Synthesis
- **分类: cs.HC; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.11257v4](http://arxiv.org/pdf/2504.11257v4)**

> **作者:** Xinyi Liu; Xiaoyi Zhang; Ziyun Zhang; Yan Lu
>
> **摘要:** Recent advancements in Large Vision-Language Models are accelerating the development of Graphical User Interface (GUI) agents that utilize human-like vision perception capabilities to enhance productivity on digital devices. Compared to approaches predicated on GUI metadata, which are platform-dependent and vulnerable to implementation variations, vision-based approaches offer broader applicability. In this vision-based paradigm, the GUI instruction grounding, which maps user instruction to the location of corresponding element on the given screenshot, remains a critical challenge, particularly due to limited public training dataset and resource-intensive manual instruction data annotation. In this paper, we delve into unexplored challenges in this task including element-to-screen ratio, unbalanced element type, and implicit instruction. To address these challenges, we introduce a large-scale data synthesis pipeline UI-E2I-Synth for generating varying complex instruction datasets using GPT-4o instead of human annotators. Furthermore, we propose a new GUI instruction grounding benchmark UI-I2E-Bench, which is designed to address the limitations of existing benchmarks by incorporating diverse annotation aspects. Our model, trained on the synthesized data, achieves superior performance in GUI instruction grounding, demonstrating the advancements of proposed data synthesis pipeline. The proposed benchmark, accompanied by extensive analyses, provides practical insights for future research in GUI grounding. We will release corresponding artifacts at https://microsoft.github.io/FIVE-UI-Evol/ .
>
---
#### [replaced 022] Move to Understand a 3D Scene: Bridging Visual Grounding and Exploration for Efficient and Versatile Embodied Navigation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.04047v2](http://arxiv.org/pdf/2507.04047v2)**

> **作者:** Ziyu Zhu; Xilin Wang; Yixuan Li; Zhuofan Zhang; Xiaojian Ma; Yixin Chen; Baoxiong Jia; Wei Liang; Qian Yu; Zhidong Deng; Siyuan Huang; Qing Li
>
> **备注:** Embodied AI; 3D Vision Language Understanding; ICCV 2025 Highlight; https://mtu3d.github.io; Spatial intelligence
>
> **摘要:** Embodied scene understanding requires not only comprehending visual-spatial information that has been observed but also determining where to explore next in the 3D physical world. Existing 3D Vision-Language (3D-VL) models primarily focus on grounding objects in static observations from 3D reconstruction, such as meshes and point clouds, but lack the ability to actively perceive and explore their environment. To address this limitation, we introduce \underline{\textbf{M}}ove \underline{\textbf{t}}o \underline{\textbf{U}}nderstand (\textbf{\model}), a unified framework that integrates active perception with \underline{\textbf{3D}} vision-language learning, enabling embodied agents to effectively explore and understand their environment. This is achieved by three key innovations: 1) Online query-based representation learning, enabling direct spatial memory construction from RGB-D frames, eliminating the need for explicit 3D reconstruction. 2) A unified objective for grounding and exploring, which represents unexplored locations as frontier queries and jointly optimizes object grounding and frontier selection. 3) End-to-end trajectory learning that combines \textbf{V}ision-\textbf{L}anguage-\textbf{E}xploration pre-training over a million diverse trajectories collected from both simulated and real-world RGB-D sequences. Extensive evaluations across various embodied navigation and question-answering benchmarks show that MTU3D outperforms state-of-the-art reinforcement learning and modular navigation approaches by 14\%, 23\%, 9\%, and 2\% in success rate on HM3D-OVON, GOAT-Bench, SG3D, and A-EQA, respectively. \model's versatility enables navigation using diverse input modalities, including categories, language descriptions, and reference images. These findings highlight the importance of bridging visual grounding and exploration for embodied intelligence.
>
---
#### [replaced 023] Clustering via Self-Supervised Diffusion
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.04283v2](http://arxiv.org/pdf/2507.04283v2)**

> **作者:** Roy Uziel; Irit Chelly; Oren Freifeld; Ari Pakman
>
> **摘要:** Diffusion models, widely recognized for their success in generative tasks, have not yet been applied to clustering. We introduce Clustering via Diffusion (CLUDI), a self-supervised framework that combines the generative power of diffusion models with pre-trained Vision Transformer features to achieve robust and accurate clustering. CLUDI is trained via a teacher-student paradigm: the teacher uses stochastic diffusion-based sampling to produce diverse cluster assignments, which the student refines into stable predictions. This stochasticity acts as a novel data augmentation strategy, enabling CLUDI to uncover intricate structures in high-dimensional data. Extensive evaluations on challenging datasets demonstrate that CLUDI achieves state-of-the-art performance in unsupervised classification, setting new benchmarks in clustering robustness and adaptability to complex data distributions. Our code is available at https://github.com/BGU-CS-VIL/CLUDI.
>
---
#### [replaced 024] Multimodal LLMs as Customized Reward Models for Text-to-Image Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.21391v2](http://arxiv.org/pdf/2507.21391v2)**

> **作者:** Shijie Zhou; Ruiyi Zhang; Huaisheng Zhu; Branislav Kveton; Yufan Zhou; Jiuxiang Gu; Jian Chen; Changyou Chen
>
> **备注:** Accepted at ICCV 2025. Code available at https://github.com/sjz5202/LLaVA-Reward
>
> **摘要:** We introduce LLaVA-Reward, an efficient reward model designed to automatically evaluate text-to-image (T2I) generations across multiple perspectives, leveraging pretrained multimodal large language models (MLLMs). Existing MLLM-based approaches require instruction-following data for supervised fine-tuning and evaluate generation quality on analyzing text response, which is time-consuming and difficult to train. To address this problem, we propose LLaVA-Reward, which directly utilizes the hidden states of MLLMs given text-image pairs. To enhance the bidirectional interaction between visual and textual representations in decoder-only MLLMs, we further propose adding a Skip-connection Cross Attention (SkipCA) module. This design enhances text-image correlation reasoning by connecting early-layer visual features with later-layer hidden representations. In addition, LLaVA-Reward supports different types of preference data for efficient fine-tuning, including paired preference data and unpaired data. We train LLaVA-Reward on four evaluation perspectives: text-image alignment, fidelity/artifact, safety, and overall ranking. Empirical results demonstrate that LLaVA-Reward outperforms conventional and MLLM-based methods in generating human-aligned scores for automatic evaluations and inference-time scaling in text-to-image generations.
>
---
#### [replaced 025] Language Driven Occupancy Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.16072v3](http://arxiv.org/pdf/2411.16072v3)**

> **作者:** Zhu Yu; Bowen Pang; Lizhe Liu; Runmin Zhang; Qiang Li; Si-Yuan Cao; Maochun Luo; Mingxia Chen; Sheng Yang; Hui-Liang Shen
>
> **备注:** ICCV 2025; Project Page: https://github.com/pkqbajng/LOcc
>
> **摘要:** We introduce LOcc, an effective and generalizable framework for open-vocabulary occupancy (OVO) prediction. Previous approaches typically supervise the networks through coarse voxel-to-text correspondences via image features as intermediates or noisy and sparse correspondences from voxel-based model-view projections. To alleviate the inaccurate supervision, we propose a semantic transitive labeling pipeline to generate dense and fine-grained 3D language occupancy ground truth. Our pipeline presents a feasible way to dig into the valuable semantic information of images, transferring text labels from images to LiDAR point clouds and ultimately to voxels, to establish precise voxel-to-text correspondences. By replacing the original prediction head of supervised occupancy models with a geometry head for binary occupancy states and a language head for language features, LOcc effectively uses the generated language ground truth to guide the learning of 3D language volume. Through extensive experiments, we demonstrate that our transitive semantic labeling pipeline can produce more accurate pseudo-labeled ground truth, diminishing labor-intensive human annotations. Additionally, we validate LOcc across various architectures, where all models consistently outperform state-of-the-art zero-shot occupancy prediction approaches on the Occ3D-nuScenes dataset.
>
---
#### [replaced 026] UI-AGILE: Advancing GUI Agents with Effective Reinforcement Learning and Precise Inference-Time Grounding
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.22025v2](http://arxiv.org/pdf/2507.22025v2)**

> **作者:** Shuquan Lian; Yuhang Wu; Jia Ma; Zihan Song; Bingqi Chen; Xiawu Zheng; Hui Li
>
> **摘要:** The emergence of Multimodal Large Language Models (MLLMs) has driven significant advances in Graphical User Interface (GUI) agent capabilities. Nevertheless, existing GUI agent training and inference techniques still suffer from a dilemma for reasoning designs, ineffective reward, and visual noise. To address these issues, we introduce UI-AGILE, a comprehensive framework enhancing GUI agents at both the training and inference stages. For training, we propose a suite of improvements to the Supervised Fine-Tuning (SFT) process: 1) a Continuous Reward function to incentivize high-precision grounding; 2) a "Simple Thinking" reward to balance planning with speed and grounding accuracy; and 3) a Cropping-based Resampling strategy to mitigate the sparse reward problem and improve learning on complex tasks. For inference, we present Decomposed Grounding with Selection, a novel method that dramatically improves grounding accuracy on high-resolution displays by breaking the image into smaller, manageable parts. Experiments show that UI-AGILE achieves the state-of-the-art performance on two benchmarks ScreenSpot-Pro and ScreenSpot-v2. For instance, using both our proposed training and inference enhancement methods brings 23% grounding accuracy improvement over the best baseline on ScreenSpot-Pro.
>
---
#### [replaced 027] MAVFlow: Preserving Paralinguistic Elements with Conditional Flow Matching for Zero-Shot AV2AV Multilingual Translation
- **分类: eess.AS; cs.CV; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.11026v2](http://arxiv.org/pdf/2503.11026v2)**

> **作者:** Sungwoo Cho; Jeongsoo Choi; Sungnyun Kim; Se-Young Yun
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Despite recent advances in text-to-speech (TTS) models, audio-visual-to-audio-visual (AV2AV) translation still faces a critical challenge: maintaining speaker consistency between the original and translated vocal and facial features. To address this issue, we propose a conditional flow matching (CFM) zero-shot audio-visual renderer that utilizes strong dual guidance from both audio and visual modalities. By leveraging multimodal guidance with CFM, our model robustly preserves speaker-specific characteristics and enhances zero-shot AV2AV translation abilities. For the audio modality, we enhance the CFM process by integrating robust speaker embeddings with x-vectors, which serve to bolster speaker consistency. Additionally, we convey emotional nuances to the face rendering module. The guidance provided by both audio and visual cues remains independent of semantic or linguistic content, allowing our renderer to effectively handle zero-shot translation tasks for monolingual speakers in different languages. We empirically demonstrate that the inclusion of high-quality mel-spectrograms conditioned on facial information not only enhances the quality of the synthesized speech but also positively influences facial generation, leading to overall performance improvements in LSE and FID score. Our code is available at https://github.com/Peter-SungwooCho/MAVFlow.
>
---
#### [replaced 028] FOCoOp: Enhancing Out-of-Distribution Robustness in Federated Prompt Learning for Vision-Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.16218v3](http://arxiv.org/pdf/2506.16218v3)**

> **作者:** Xinting Liao; Weiming Liu; Jiaming Qian; Pengyang Zhou; Jiahe Xu; Wenjie Wang; Chaochao Chen; Xiaolin Zheng; Tat-Seng Chua
>
> **备注:** Accepted by ICML25
>
> **摘要:** Federated prompt learning (FPL) for vision-language models is a powerful approach to collaboratively adapt models across distributed clients while preserving data privacy. However, existing FPL approaches suffer from a trade-off between performance and robustness, particularly in out-of-distribution (OOD) shifts, limiting their reliability in real-world scenarios. The inherent in-distribution (ID) data heterogeneity among different clients makes it more challenging to maintain this trade-off. To fill this gap, we introduce a Federated OOD-aware Context Optimization (FOCoOp) framework, which captures diverse distributions among clients using ID global prompts, local prompts, and OOD prompts. Specifically, FOCoOp leverages three sets of prompts to create both class-level and distribution-level separations, which adapt to OOD shifts through bi-level distributionally robust optimization. Additionally, FOCoOp improves the discrimination consistency among clients, i.e., calibrating global prompts, seemingly OOD prompts, and OOD prompts by semi-unbalanced optimal transport. The extensive experiments on real-world datasets demonstrate that FOCoOp effectively captures decentralized heterogeneous distributions and enhances robustness of different OOD shifts. The project is available at GitHub.
>
---
#### [replaced 029] MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10289v2](http://arxiv.org/pdf/2503.10289v2)**

> **作者:** Zebin He; Mingxin Yang; Shuhui Yang; Yixuan Tang; Tao Wang; Kaihao Zhang; Guanying Chen; Yuhong Liu; Jie Jiang; Chunchao Guo; Wenhan Luo
>
> **摘要:** Physically-based rendering (PBR) has become a cornerstone in modern computer graphics, enabling realistic material representation and lighting interactions in 3D scenes. In this paper, we present MaterialMVP, a novel end-to-end model for generating PBR textures from 3D meshes and image prompts, addressing key challenges in multi-view material synthesis. Our approach leverages Reference Attention to extract and encode informative latent from the input reference images, enabling intuitive and controllable texture generation. We also introduce a Consistency-Regularized Training strategy to enforce stability across varying viewpoints and illumination conditions, ensuring illumination-invariant and geometrically consistent results. Additionally, we propose Dual-Channel Material Generation, which separately optimizes albedo and metallic-roughness (MR) textures while maintaining precise spatial alignment with the input images through Multi-Channel Aligned Attention. Learnable material embeddings are further integrated to capture the distinct properties of albedo and MR. Experimental results demonstrate that our model generates PBR textures with realistic behavior across diverse lighting scenarios, outperforming existing methods in both consistency and quality for scalable 3D asset creation.
>
---
#### [replaced 030] Learning to See in the Extremely Dark
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21132v2](http://arxiv.org/pdf/2506.21132v2)**

> **作者:** Hai Jiang; Binhao Guan; Zhen Liu; Xiaohong Liu; Jian Yu; Zheng Liu; Songchen Han; Shuaicheng Liu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Learning-based methods have made promising advances in low-light RAW image enhancement, while their capability to extremely dark scenes where the environmental illuminance drops as low as 0.0001 lux remains to be explored due to the lack of corresponding datasets. To this end, we propose a paired-to-paired data synthesis pipeline capable of generating well-calibrated extremely low-light RAW images at three precise illuminance ranges of 0.01-0.1 lux, 0.001-0.01 lux, and 0.0001-0.001 lux, together with high-quality sRGB references to comprise a large-scale paired dataset named See-in-the-Extremely-Dark (SIED) to benchmark low-light RAW image enhancement approaches. Furthermore, we propose a diffusion-based framework that leverages the generative ability and intrinsic denoising property of diffusion models to restore visually pleasing results from extremely low-SNR RAW inputs, in which an Adaptive Illumination Correction Module (AICM) and a color consistency loss are introduced to ensure accurate exposure correction and color restoration. Extensive experiments on the proposed SIED and publicly available benchmarks demonstrate the effectiveness of our method. The code and dataset are available at https://github.com/JianghaiSCU/SIED.
>
---
#### [replaced 031] Contrastive Test-Time Composition of Multiple LoRA Models for Image Generation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.19776v2](http://arxiv.org/pdf/2403.19776v2)**

> **作者:** Tuna Han Salih Meral; Enis Simsar; Federico Tombari; Pinar Yanardag
>
> **摘要:** Low-Rank Adaptation (LoRA) has emerged as a powerful and popular technique for personalization, enabling efficient adaptation of pre-trained image generation models for specific tasks without comprehensive retraining. While employing individual pre-trained LoRA models excels at representing single concepts, such as those representing a specific dog or a cat, utilizing multiple LoRA models to capture a variety of concepts in a single image still poses a significant challenge. Existing methods often fall short, primarily because the attention mechanisms within different LoRA models overlap, leading to scenarios where one concept may be completely ignored (e.g., omitting the dog) or where concepts are incorrectly combined (e.g., producing an image of two cats instead of one cat and one dog). We introduce CLoRA, a training-free approach that addresses these limitations by updating the attention maps of multiple LoRA models at test-time, and leveraging the attention maps to create semantic masks for fusing latent representations. This enables the generation of composite images that accurately reflect the characteristics of each LoRA. Our comprehensive qualitative and quantitative evaluations demonstrate that CLoRA significantly outperforms existing methods in multi-concept image generation using LoRAs.
>
---
#### [replaced 032] GS-Occ3D: Scaling Vision-only Occupancy Reconstruction for Autonomous Driving with Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.19451v2](http://arxiv.org/pdf/2507.19451v2)**

> **作者:** Baijun Ye; Minghui Qin; Saining Zhang; Moonjun Gong; Shaoting Zhu; Zebang Shen; Luan Zhang; Lu Zhang; Hao Zhao; Hang Zhao
>
> **备注:** ICCV 2025. Project Page: https://gs-occ3d.github.io/
>
> **摘要:** Occupancy is crucial for autonomous driving, providing essential geometric priors for perception and planning. However, existing methods predominantly rely on LiDAR-based occupancy annotations, which limits scalability and prevents leveraging vast amounts of potential crowdsourced data for auto-labeling. To address this, we propose GS-Occ3D, a scalable vision-only framework that directly reconstructs occupancy. Vision-only occupancy reconstruction poses significant challenges due to sparse viewpoints, dynamic scene elements, severe occlusions, and long-horizon motion. Existing vision-based methods primarily rely on mesh representation, which suffer from incomplete geometry and additional post-processing, limiting scalability. To overcome these issues, GS-Occ3D optimizes an explicit occupancy representation using an Octree-based Gaussian Surfel formulation, ensuring efficiency and scalability. Additionally, we decompose scenes into static background, ground, and dynamic objects, enabling tailored modeling strategies: (1) Ground is explicitly reconstructed as a dominant structural element, significantly improving large-area consistency; (2) Dynamic vehicles are separately modeled to better capture motion-related occupancy patterns. Extensive experiments on the Waymo dataset demonstrate that GS-Occ3D achieves state-of-the-art geometry reconstruction results. By curating vision-only binary occupancy labels from diverse urban scenes, we show their effectiveness for downstream occupancy models on Occ3D-Waymo and superior zero-shot generalization on Occ3D-nuScenes. It highlights the potential of large-scale vision-based occupancy reconstruction as a new paradigm for scalable auto-labeling. Project Page: https://gs-occ3d.github.io/
>
---
#### [replaced 033] Fine-Tuning Visual Autoregressive Models for Subject-Driven Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.02612v2](http://arxiv.org/pdf/2504.02612v2)**

> **作者:** Jiwoo Chung; Sangeek Hyun; Hyunjun Kim; Eunseo Koh; MinKyu Lee; Jae-Pil Heo
>
> **备注:** Accepted to ICCV 2025. Project page: https://jiwoogit.github.io/ARBooth/
>
> **摘要:** Recent advances in text-to-image generative models have enabled numerous practical applications, including subject-driven generation, which fine-tunes pretrained models to capture subject semantics from only a few examples. While diffusion-based models produce high-quality images, their extensive denoising steps result in significant computational overhead, limiting real-world applicability. Visual autoregressive (VAR) models, which predict next-scale tokens rather than spatially adjacent ones, offer significantly faster inference suitable for practical deployment. In this paper, we propose the first VAR-based approach for subject-driven generation. However, naive fine-tuning VAR leads to computational overhead, language drift, and reduced diversity. To address these challenges, we introduce selective layer tuning to reduce complexity and prior distillation to mitigate language drift. Additionally, we found that the early stages have a greater influence on the generation of subject than the latter stages, which merely synthesize minor details. Based on this finding, we propose scale-wise weighted tuning, which prioritizes coarser resolutions for promoting the model to focus on the subject-relevant information instead of local details. Extensive experiments validate that our method significantly outperforms diffusion-based baselines across various metrics and demonstrates its practical usage.
>
---
#### [replaced 034] Interpretable Open-Vocabulary Referring Object Detection with Reverse Contrast Attention
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.19891v2](http://arxiv.org/pdf/2507.19891v2)**

> **作者:** Drandreb Earl O. Juanico; Rowel O. Atienza; Jeffrey Kenneth Go
>
> **备注:** To be published in the ICCVW 2025 Proceedings
>
> **摘要:** We propose Reverse Contrast Attention (RCA), a plug-in method that enhances object localization in vision-language transformers without retraining. RCA reweights final-layer attention by suppressing extremes and amplifying mid-level activations to let semantically relevant but subdued tokens guide predictions. We evaluate it on Open Vocabulary Referring Object Detection (OV-RefOD), introducing FitAP, a confidence-free average precision metric based on IoU and box area. RCA improves FitAP in 11 out of 15 open-source VLMs, with gains up to $+26.6\%$. Effectiveness aligns with attention sharpness and fusion timing; while late-fusion models benefit consistently, models like $\texttt{DeepSeek-VL2}$ also improve, pointing to capacity and disentanglement as key factors. RCA offers both interpretability and performance gains for multimodal transformers. Codes and dataset are available from https://github.com/earl-juanico/rca
>
---
#### [replaced 035] Differential Contrastive Training for Gaze Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.20128v3](http://arxiv.org/pdf/2502.20128v3)**

> **作者:** Lin Zhang; Yi Tian; XiYun Wang; Wanru Xu; Yi Jin; Yaping Huang
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** The complex application scenarios have raised critical requirements for precise and generalizable gaze estimation methods. Recently, the pre-trained CLIP has achieved remarkable performance on various vision tasks, but its potentials have not been fully exploited in gaze estimation. In this paper, we propose a novel Differential Contrastive Training strategy, which boosts gaze estimation performance with the help of the CLIP. Accordingly, a Differential Contrastive Gaze Estimation network (DCGaze) composed of a Visual Appearance-aware branch and a Semantic Differential-aware branch is introduced. The Visual Appearance-aware branch is essentially a primary gaze estimation network and it incorporates an Adaptive Feature-refinement Unit (AFU) and a Double-head Gaze Regressor (DGR), which both help the primary network to extract informative and gaze-related appearance features. Moreover, the Semantic Difference-aware branch is designed on the basis of the CLIP's text encoder to reveal the semantic difference of gazes. This branch could further empower the Visual Appearance-aware branch with the capability of characterizing the gaze-related semantic information. Extensive experimental results on four challenging datasets over within and cross-domain tasks demonstrate the effectiveness of our DCGaze.The code is available at https://github.com/LinZhang-bjtu/DCGaze.
>
---
#### [replaced 036] Distance and Collision Probability Estimation from Gaussian Surface Models
- **分类: cs.RO; cs.CG; cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2402.00186v3](http://arxiv.org/pdf/2402.00186v3)**

> **作者:** Kshitij Goel; Wennie Tabib
>
> **备注:** Accepted at IROS 2025
>
> **摘要:** This paper describes continuous-space methodologies to estimate the collision probability, Euclidean distance and gradient between an ellipsoidal robot model and an environment surface modeled as a set of Gaussian distributions. Continuous-space collision probability estimation is critical for uncertainty-aware motion planning. Most collision detection and avoidance approaches assume the robot is modeled as a sphere, but ellipsoidal representations provide tighter approximations and enable navigation in cluttered and narrow spaces. State-of-the-art methods derive the Euclidean distance and gradient by processing raw point clouds, which is computationally expensive for large workspaces. Recent advances in Gaussian surface modeling (e.g. mixture models, splatting) enable compressed and high-fidelity surface representations. Few methods exist to estimate continuous-space occupancy from such models. They require Gaussians to model free space and are unable to estimate the collision probability, Euclidean distance and gradient for an ellipsoidal robot. The proposed methods bridge this gap by extending prior work in ellipsoid-to-ellipsoid Euclidean distance and collision probability estimation to Gaussian surface models. A geometric blending approach is also proposed to improve collision probability estimation. The approaches are evaluated with numerical 2D and 3D experiments using real-world point cloud data. Methods for efficient calculation of these quantities are demonstrated to execute within a few microseconds per ellipsoid pair using a single-thread on low-power CPUs of modern embedded computers
>
---
#### [replaced 037] StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21541v3](http://arxiv.org/pdf/2506.21541v3)**

> **作者:** Chuxin Wang; Yixin Zha; Wenfei Yang; Tianzhu Zhang
>
> **备注:** Accepted by ICCV 2025, website: https://chuxwa.github.io/project_StruMamba3D/
>
> **摘要:** Recently, Mamba-based methods have demonstrated impressive performance in point cloud representation learning by leveraging State Space Model (SSM) with the efficient context modeling ability and linear complexity. However, these methods still face two key issues that limit the potential of SSM: Destroying the adjacency of 3D points during SSM processing and failing to retain long-sequence memory as the input length increases in downstream tasks. To address these issues, we propose StruMamba3D, a novel paradigm for self-supervised point cloud representation learning. It enjoys several merits. First, we design spatial states and use them as proxies to preserve spatial dependencies among points. Second, we enhance the SSM with a state-wise update strategy and incorporate a lightweight convolution to facilitate interactions between spatial states for efficient structure modeling. Third, our method reduces the sensitivity of pre-trained Mamba-based models to varying input lengths by introducing a sequence length-adaptive strategy. Experimental results across four downstream tasks showcase the superior performance of our method. In addition, our method attains the SOTA 95.1% accuracy on ModelNet40 and 92.75% accuracy on the most challenging split of ScanObjectNN without voting strategy.
>
---
#### [replaced 038] Seed Selection for Human-Oriented Image Reconstruction via Guided Diffusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.05363v3](http://arxiv.org/pdf/2506.05363v3)**

> **作者:** Yui Tatsumi; Ziyue Zeng; Hiroshi Watanabe
>
> **备注:** Accepted by 2025 IEEE 14th Global Conference on Consumer Electronics (GCCE 2025)
>
> **摘要:** Conventional methods for scalable image coding for humans and machines require the transmission of additional information to achieve scalability. A recent diffusion-based approach avoids this by generating human-oriented images from machine-oriented images without extra bitrate. However, it utilizes a single random seed, which may lead to suboptimal image quality. In this paper, we propose a seed selection method that identifies the optimal seed from multiple candidates to improve image quality without increasing the bitrate. To reduce the computational cost, selection is performed based on intermediate outputs obtained from early steps of the reverse diffusion process. Experimental results demonstrate that our proposed method outperforms the baseline, which uses a single random seed without selection, across multiple evaluation metrics.
>
---
#### [replaced 039] Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.06838v5](http://arxiv.org/pdf/2501.06838v5)**

> **作者:** Du Chen; Liyi Chen; Zhengqiang Zhang; Lei Zhang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Implicit Neural Representations (INR) have been successfully employed for Arbitrary-scale Super-Resolution (ASR). However, INR-based models need to query the multi-layer perceptron module numerous times and render a pixel in each query, resulting in insufficient representation capability and low computational efficiency. Recently, Gaussian Splatting (GS) has shown its advantages over INR in both visual quality and rendering speed in 3D tasks, which motivates us to explore whether GS can be employed for the ASR task. However, directly applying GS to ASR is exceptionally challenging because the original GS is an optimization-based method through overfitting each single scene, while in ASR we aim to learn a single model that can generalize to different images and scaling factors. We overcome these challenges by developing two novel techniques. Firstly, to generalize GS for ASR, we elaborately design an architecture to predict the corresponding image-conditioned Gaussians of the input low-resolution image in a feed-forward manner. Each Gaussian can fit the shape and direction of an area of complex textures, showing powerful representation capability. Secondly, we implement an efficient differentiable 2D GPU/CUDA-based scale-aware rasterization to render super-resolved images by sampling discrete RGB values from the predicted continuous Gaussians. Via end-to-end training, our optimized network, namely GSASR, can perform ASR for any image and unseen scaling factors. Extensive experiments validate the effectiveness of our proposed method. The code and models are available at https://github.com/ChrisDud0257/GSASR.
>
---
#### [replaced 040] Skull-stripping induces shortcut learning in MRI-based Alzheimer's disease classification
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.15831v3](http://arxiv.org/pdf/2501.15831v3)**

> **作者:** Christian Tinauer; Maximilian Sackl; Rudolf Stollberger; Reinhold Schmidt; Stefan Ropele; Christian Langkammer
>
> **摘要:** Objectives: High classification accuracy of Alzheimer's disease (AD) from structural MRI has been achieved using deep neural networks, yet the specific image features contributing to these decisions remain unclear. In this study, the contributions of T1-weighted (T1w) gray-white matter texture, volumetric information, and preprocessing -- particularly skull-stripping -- were systematically assessed. Methods: A dataset of 990 matched T1w MRIs from AD patients and cognitively normal controls from the ADNI database were used. Preprocessing was varied through skull-stripping and intensity binarization to isolate texture and shape contributions. A 3D convolutional neural network was trained on each configuration, and classification performance was compared using exact McNemar tests with discrete Bonferroni-Holm correction. Feature relevance was analyzed using Layer-wise Relevance Propagation, image similarity metrics, and spectral clustering of relevance maps. Results: Despite substantial differences in image content, classification accuracy, sensitivity, and specificity remained stable across preprocessing conditions. Models trained on binarized images preserved performance, indicating minimal reliance on gray-white matter texture. Instead, volumetric features -- particularly brain contours introduced through skull-stripping -- were consistently used by the models. Conclusions: This behavior reflects a shortcut learning phenomenon, where preprocessing artifacts act as potentially unintended cues. The resulting Clever Hans effect emphasizes the critical importance of interpretability tools to reveal hidden biases and to ensure robust and trustworthy deep learning in medical imaging.
>
---
#### [replaced 041] SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2412.20104v5](http://arxiv.org/pdf/2412.20104v5)**

> **作者:** Wenkun He; Yun Liu; Ruitao Liu; Li Yi
>
> **备注:** 27 pages, 10 figures, 20 tables. Accepted by ICCV 2025
>
> **摘要:** Synthesizing realistic human-object interaction motions is a critical problem in VR/AR and human animation. Unlike the commonly studied scenarios involving a single human or hand interacting with one object, we address a more generic multi-body setting with arbitrary numbers of humans, hands, and objects. This complexity introduces significant challenges in synchronizing motions due to the high correlations and mutual influences among bodies. To address these challenges, we introduce SyncDiff, a novel method for multi-body interaction synthesis using a synchronized motion diffusion strategy. SyncDiff employs a single diffusion model to capture the joint distribution of multi-body motions. To enhance motion fidelity, we propose a frequency-domain motion decomposition scheme. Additionally, we introduce a new set of alignment scores to emphasize the synchronization of different body motions. SyncDiff jointly optimizes both data sample likelihood and alignment likelihood through an explicit synchronization strategy. Extensive experiments across four datasets with various multi-body configurations demonstrate the superiority of SyncDiff over existing state-of-the-art motion synthesis methods.
>
---
#### [replaced 042] Ultra3D: Efficient and High-Fidelity 3D Generation with Part Attention
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.17745v2](http://arxiv.org/pdf/2507.17745v2)**

> **作者:** Yiwen Chen; Zhihao Li; Yikai Wang; Hu Zhang; Qin Li; Chi Zhang; Guosheng Lin
>
> **备注:** Project Page: https://buaacyw.github.io/ultra3d/
>
> **摘要:** Recent advances in sparse voxel representations have significantly improved the quality of 3D content generation, enabling high-resolution modeling with fine-grained geometry. However, existing frameworks suffer from severe computational inefficiencies due to the quadratic complexity of attention mechanisms in their two-stage diffusion pipelines. In this work, we propose Ultra3D, an efficient 3D generation framework that significantly accelerates sparse voxel modeling without compromising quality. Our method leverages the compact VecSet representation to efficiently generate a coarse object layout in the first stage, reducing token count and accelerating voxel coordinate prediction. To refine per-voxel latent features in the second stage, we introduce Part Attention, a geometry-aware localized attention mechanism that restricts attention computation within semantically consistent part regions. This design preserves structural continuity while avoiding unnecessary global attention, achieving up to 6.7x speed-up in latent generation. To support this mechanism, we construct a scalable part annotation pipeline that converts raw meshes into part-labeled sparse voxels. Extensive experiments demonstrate that Ultra3D supports high-resolution 3D generation at 1024 resolution and achieves state-of-the-art performance in both visual fidelity and user preference.
>
---
#### [replaced 043] RaGS: Unleashing 3D Gaussian Splatting from 4D Radar and Monocular Cues for 3D Object Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.19856v2](http://arxiv.org/pdf/2507.19856v2)**

> **作者:** Xiaokai Bai; Chenxu Zhou; Lianqing Zheng; Si-Yuan Cao; Jianan Liu; Xiaohan Zhang; Zhengzhuang Zhang; Hui-liang Shen
>
> **备注:** 9 pages, 6 figures, conference
>
> **摘要:** 4D millimeter-wave radar has emerged as a promising sensor for autonomous driving, but effective 3D object detection from both 4D radar and monocular images remains a challenge. Existing fusion approaches typically rely on either instance-based proposals or dense BEV grids, which either lack holistic scene understanding or are limited by rigid grid structures. To address these, we propose RaGS, the first framework to leverage 3D Gaussian Splatting (GS) as representation for fusing 4D radar and monocular cues in 3D object detection. 3D GS naturally suits 3D object detection by modeling the scene as a field of Gaussians, dynamically allocating resources on foreground objects and providing a flexible, resource-efficient solution. RaGS uses a cascaded pipeline to construct and refine the Gaussian field. It starts with the Frustum-based Localization Initiation (FLI), which unprojects foreground pixels to initialize coarse 3D Gaussians positions. Then, the Iterative Multimodal Aggregation (IMA) fuses semantics and geometry, refining the limited Gaussians to the regions of interest. Finally, the Multi-level Gaussian Fusion (MGF) renders the Gaussians into multi-level BEV features for 3D object detection. By dynamically focusing on sparse objects within scenes, RaGS enable object concentrating while offering comprehensive scene perception. Extensive experiments on View-of-Delft, TJ4DRadSet, and OmniHD-Scenes benchmarks demonstrate its state-of-the-art performance. Code will be released.
>
---
#### [replaced 044] Scaling RL to Long Videos
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.07966v3](http://arxiv.org/pdf/2507.07966v3)**

> **作者:** Yukang Chen; Wei Huang; Baifeng Shi; Qinghao Hu; Hanrong Ye; Ligeng Zhu; Zhijian Liu; Pavlo Molchanov; Jan Kautz; Xiaojuan Qi; Sifei Liu; Hongxu Yin; Yao Lu; Song Han
>
> **备注:** Code at https://github.com/NVlabs/Long-RL and model at https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B
>
> **摘要:** We introduce a full-stack framework that scales up reasoning in vision-language models (VLMs) to long videos, leveraging reinforcement learning. We address the unique challenges of long video reasoning by integrating three critical components: (1) a large-scale dataset, LongVideo-Reason, comprising 104K long video QA pairs with high-quality reasoning annotations across diverse domains such as sports, games, and vlogs; (2) a two-stage training pipeline that extends VLMs with chain-of-thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL); and (3) a training infrastructure for long video RL, named Multi-modal Reinforcement Sequence Parallelism (MR-SP), which incorporates sequence parallelism and a vLLM-based engine tailored for long video, using cached video embeddings for efficient rollout and prefilling. In our experiments, LongVILA-R1-7B achieves strong performance on video benchmarks, reaching 65.1% and 71.1% accuracy on VideoMME without and with subtitles, respectively, and consistently outperforming LongVILA-7B across multiple benchmarks. Moreover, LongVILA-R1-7B supports processing up to 8,192 video frames per video, and configurable FPS settings. Notably, our MR-SP system achieves up to 2.1x speedup on long video RL training. In addition, we release our training system for public availability that supports RL training on various modalities (video, text, and audio), various models (VILA and Qwen series), and even image and video generation models. On a single A100 node (8 GPUs), it supports RL training on hour-long videos (e.g., 3,600 frames).
>
---
#### [replaced 045] ViM-VQ: Efficient Post-Training Vector Quantization for Visual Mamba
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.09509v2](http://arxiv.org/pdf/2503.09509v2)**

> **作者:** Juncan Deng; Shuaiting Li; Zeyu Wang; Kedong Xu; Hong Gu; Kejie Huang
>
> **摘要:** Visual Mamba networks (ViMs) extend the selective state space model (Mamba) to various vision tasks and demonstrate significant potential. As a promising compression technique, vector quantization (VQ) decomposes network weights into codebooks and assignments, significantly reducing memory usage and computational latency, thereby enabling the deployment of ViMs on edge devices. Although existing VQ methods have achieved extremely low-bit quantization (e.g., 3-bit, 2-bit, and 1-bit) in convolutional neural networks and Transformer-based networks, directly applying these methods to ViMs results in unsatisfactory accuracy. We identify several key challenges: 1) The weights of Mamba-based blocks in ViMs contain numerous outliers, significantly amplifying quantization errors. 2) When applied to ViMs, the latest VQ methods suffer from excessive memory consumption, lengthy calibration procedures, and suboptimal performance in the search for optimal codewords. In this paper, we propose ViM-VQ, an efficient post-training vector quantization method tailored for ViMs. ViM-VQ consists of two innovative components: 1) a fast convex combination optimization algorithm that efficiently updates both the convex combinations and the convex hulls to search for optimal codewords, and 2) an incremental vector quantization strategy that incrementally confirms optimal codewords to mitigate truncation errors. Experimental results demonstrate that ViM-VQ achieves state-of-the-art performance in low-bit quantization across various visual tasks.
>
---
#### [replaced 046] FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.10487v2](http://arxiv.org/pdf/2504.10487v2)**

> **作者:** Yasser Benigmim; Mohammad Fahes; Tuan-Hung Vu; Andrei Bursuc; Raoul de Charette
>
> **备注:** ICCV 2025; Project Page: https://yasserben.github.io/FLOSS/
>
> **摘要:** In this paper, we challenge the conventional practice in Open-Vocabulary Semantic Segmentation (OVSS) of using averaged class-wise text embeddings, which are typically obtained by encoding each class name with multiple templates (e.g., a photo of <class>, a sketch of a <class>). We investigate the impact of templates for OVSS, and find that for each class, there exist single-template classifiers--which we refer to as class-experts--that significantly outperform the conventional averaged classifier. First, to identify these class-experts, we introduce a novel approach that estimates them without any labeled data or training. By leveraging the class-wise prediction entropy of single-template classifiers, we select those yielding the lowest entropy as the most reliable class-experts. Second, we combine the outputs of class-experts in a new fusion process. Our plug-and-play method, coined FLOSS, is orthogonal and complementary to existing OVSS methods, offering an improvement without the need for additional labels or training. Extensive experiments show that FLOSS consistently enhances state-of-the-art OVSS models, generalizes well across datasets with different distribution shifts, and delivers substantial improvements in low-data scenarios where only a few unlabeled images are available. Our code is available at https://github.com/yasserben/FLOSS .
>
---
#### [replaced 047] Calibrated Multi-Preference Optimization for Aligning Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.02588v2](http://arxiv.org/pdf/2502.02588v2)**

> **作者:** Kyungmin Lee; Xiaohang Li; Qifei Wang; Junfeng He; Junjie Ke; Ming-Hsuan Yang; Irfan Essa; Jinwoo Shin; Feng Yang; Yinxiao Li
>
> **备注:** CVPR 2025, Project page: https://kyungmnlee.github.io/capo.github.io/
>
> **摘要:** Aligning text-to-image (T2I) diffusion models with preference optimization is valuable for human-annotated datasets, but the heavy cost of manual data collection limits scalability. Using reward models offers an alternative, however, current preference optimization methods fall short in exploiting the rich information, as they only consider pairwise preference distribution. Furthermore, they lack generalization to multi-preference scenarios and struggle to handle inconsistencies between rewards. To address this, we present Calibrated Preference Optimization (CaPO), a novel method to align T2I diffusion models by incorporating the general preference from multiple reward models without human annotated data. The core of our approach involves a reward calibration method to approximate the general preference by computing the expected win-rate against the samples generated by the pretrained models. Additionally, we propose a frontier-based pair selection method that effectively manages the multi-preference distribution by selecting pairs from Pareto frontiers. Finally, we use regression loss to fine-tune diffusion models to match the difference between calibrated rewards of a selected pair. Experimental results show that CaPO consistently outperforms prior methods, such as Direct Preference Optimization (DPO), in both single and multi-reward settings validated by evaluation on T2I benchmarks, including GenEval and T2I-Compbench.
>
---
#### [replaced 048] SMAFormer: Synergistic Multi-Attention Transformer for Medical Image Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.00346v4](http://arxiv.org/pdf/2409.00346v4)**

> **作者:** Fuchen Zheng; Xuhang Chen; Weihuang Liu; Haolun Li; Yingtie Lei; Jiahui He; Chi-Man Pun; Shounjun Zhou
>
> **备注:** Accepted by IEEE BIBM 2024
>
> **摘要:** In medical image segmentation, specialized computer vision techniques, notably transformers grounded in attention mechanisms and residual networks employing skip connections, have been instrumental in advancing performance. Nonetheless, previous models often falter when segmenting small, irregularly shaped tumors. To this end, we introduce SMAFormer, an efficient, Transformer-based architecture that fuses multiple attention mechanisms for enhanced segmentation of small tumors and organs. SMAFormer can capture both local and global features for medical image segmentation. The architecture comprises two pivotal components. First, a Synergistic Multi-Attention (SMA) Transformer block is proposed, which has the benefits of Pixel Attention, Channel Attention, and Spatial Attention for feature enrichment. Second, addressing the challenge of information loss incurred during attention mechanism transitions and feature fusion, we design a Feature Fusion Modulator. This module bolsters the integration between the channel and spatial attention by mitigating reshaping-induced information attrition. To evaluate our method, we conduct extensive experiments on various medical image segmentation tasks, including multi-organ, liver tumor, and bladder tumor segmentation, achieving state-of-the-art results. Code and models are available at: https://github.com/CXH-Research/SMAFormer.
>
---
#### [replaced 049] CLIP-HandID: Vision-Language Model for Hand-Based Person Identification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.12447v3](http://arxiv.org/pdf/2506.12447v3)**

> **作者:** Nathanael L. Baisa; Babu Pallam; Amudhavel Jayavel
>
> **摘要:** This paper introduces a novel approach to person identification using hand images, designed specifically for criminal investigations. The method is particularly valuable in serious crimes such as sexual abuse, where hand images are often the only identifiable evidence available. Our proposed method, CLIP-HandID, leverages a pre-trained foundational vision-language model - CLIP - to efficiently learn discriminative deep feature representations from hand images (input to CLIP's image encoder) using textual prompts as semantic guidance. Since hand images are labeled with indexes rather than text descriptions, we employ a textual inversion network to learn pseudo-tokens that encode specific visual contexts or appearance attributes. These learned pseudo-tokens are then incorporated into textual prompts, which are fed into CLIP's text encoder to leverage its multi-modal reasoning and enhance generalization for identification. Through extensive evaluations on two large, publicly available hand datasets with multi-ethnic representation, we demonstrate that our method significantly outperforms existing approaches.
>
---
#### [replaced 050] ComicsPAP: understanding comic strips by picking the correct panel
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.08561v3](http://arxiv.org/pdf/2503.08561v3)**

> **作者:** Emanuele Vivoli; Artemis Llabrés; Mohamed Ali Souibgui; Marco Bertini; Ernest Valveny Llobet; Dimosthenis Karatzas
>
> **摘要:** Large multimodal models (LMMs) have made impressive strides in image captioning, VQA, and video comprehension, yet they still struggle with the intricate temporal and spatial cues found in comics. To address this gap, we introduce ComicsPAP, a large-scale benchmark designed for comic strip understanding. Comprising over 100k samples and organized into 5 subtasks under a Pick-a-Panel framework, ComicsPAP demands models to identify the missing panel in a sequence. Our evaluations, conducted under both multi-image and single-image protocols, reveal that current state-of-the-art LMMs perform near chance on these tasks, underscoring significant limitations in capturing sequential and contextual dependencies. To close the gap, we adapted LMMs for comic strip understanding, obtaining better results on ComicsPAP than 10x bigger models, demonstrating that ComicsPAP offers a robust resource to drive future research in multimodal comic comprehension.
>
---
#### [replaced 051] STaR: Seamless Spatial-Temporal Aware Motion Retargeting with Penetration and Consistency Constraints
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.06504v2](http://arxiv.org/pdf/2504.06504v2)**

> **作者:** Xiaohang Yang; Qing Wang; Jiahao Yang; Gregory Slabaugh; Shanxin Yuan
>
> **备注:** Accepted by ICCV 2025, 13 pages, 9 figures; Code page: https://github.com/XiaohangYang829/STaR
>
> **摘要:** Motion retargeting seeks to faithfully replicate the spatio-temporal motion characteristics of a source character onto a target character with a different body shape. Apart from motion semantics preservation, ensuring geometric plausibility and maintaining temporal consistency are also crucial for effective motion retargeting. However, many existing methods prioritize either geometric plausibility or temporal consistency. Neglecting geometric plausibility results in interpenetration while neglecting temporal consistency leads to motion jitter. In this paper, we propose a novel sequence-to-sequence model for seamless Spatial-Temporal aware motion Retargeting (STaR), with penetration and consistency constraints. STaR consists of two modules: (1) a spatial module that incorporates dense shape representation and a novel limb penetration constraint to ensure geometric plausibility while preserving motion semantics, and (2) a temporal module that utilizes a temporal transformer and a novel temporal consistency constraint to predict the entire motion sequence at once while enforcing multi-level trajectory smoothness. The seamless combination of the two modules helps us achieve a good balance between the semantic, geometric, and temporal targets. Extensive experiments on the Mixamo and ScanRet datasets demonstrate that our method produces plausible and coherent motions while significantly reducing interpenetration rates compared with other approaches. Code page: https://github.com/XiaohangYang829/STaR.
>
---
#### [replaced 052] Automated MRI Tumor Segmentation using hybrid U-Net with Transformer and Efficient Attention
- **分类: eess.IV; cs.CV; I.4.6; I.2.6; I.4.9**

- **链接: [http://arxiv.org/pdf/2506.15562v2](http://arxiv.org/pdf/2506.15562v2)**

> **作者:** Syed Haider Ali; Asrar Ahmad; Muhammad Ali; Asifullah Khan; Nadeem Shaukat
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** Cancer is an abnormal growth with potential to invade locally and metastasize to distant organs. Accurate auto-segmentation of the tumor and surrounding normal tissues is required for radiotherapy treatment plan optimization. Recent AI-based segmentation models are generally trained on large public datasets, which lack the heterogeneity of local patient populations. While these studies advance AI-based medical image segmentation, research on local datasets is necessary to develop and integrate AI tumor segmentation models directly into hospital software for efficient and accurate oncology treatment planning and execution. This study enhances tumor segmentation using computationally efficient hybrid UNet-Transformer models on magnetic resonance imaging (MRI) datasets acquired from a local hospital under strict privacy protection. We developed a robust data pipeline for seamless DICOM extraction and preprocessing, followed by extensive image augmentation to ensure model generalization across diverse clinical settings, resulting in a total dataset of 6080 images for training. Our novel architecture integrates UNet-based convolutional neural networks with a transformer bottleneck and complementary attention modules, including efficient attention, Squeeze-and-Excitation (SE) blocks, Convolutional Block Attention Module (CBAM), and ResNeXt blocks. To accelerate convergence and reduce computational demands, we used a maximum batch size of 8 and initialized the encoder with pretrained ImageNet weights, training the model on dual NVIDIA T4 GPUs via checkpointing to overcome Kaggle's runtime limits. Quantitative evaluation on the local MRI dataset yielded a Dice similarity coefficient of 0.764 and an Intersection over Union (IoU) of 0.736, demonstrating competitive performance despite limited data and underscoring the importance of site-specific model development for clinical deployment.
>
---
#### [replaced 053] FOF-X: Towards Real-time Detailed Human Reconstruction from a Single Image
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.05961v2](http://arxiv.org/pdf/2412.05961v2)**

> **作者:** Qiao Feng; Yuanwang Yang; Yebin Liu; Yu-Kun Lai; Jingyu Yang; Kun Li
>
> **备注:** Extended journal version of our previous conference paper: FOF: Learning Fourier Occupancy Field for Monocular Real-time Human Reconstruction (arXiv:2206.02194)
>
> **摘要:** We introduce FOF-X for real-time reconstruction of detailed human geometry from a single image. Balancing real-time speed against high-quality results is a persistent challenge, mainly due to the high computational demands of existing 3D representations. To address this, we propose Fourier Occupancy Field (FOF), an efficient 3D representation by learning the Fourier series. The core of FOF is to factorize a 3D occupancy field into a 2D vector field, retaining topology and spatial relationships within the 3D domain while facilitating compatibility with 2D convolutional neural networks. Such a representation bridges the gap between 3D and 2D domains, enabling the integration of human parametric models as priors and enhancing the reconstruction robustness. Based on FOF, we design a new reconstruction framework, FOF-X, to avoid the performance degradation caused by texture and lighting. This enables our real-time reconstruction system to better handle the domain gap between training images and real images. Additionally, in FOF-X, we enhance the inter-conversion algorithms between FOF and mesh representations with a Laplacian constraint and an automaton-based discontinuity matcher, improving both quality and robustness. We validate the strengths of our approach on different datasets and real-captured data, where FOF-X achieves new state-of-the-art results. The code has already been released for research purposes at https://cic.tju.edu.cn/faculty/likun/projects/FOFX/index.html.
>
---
#### [replaced 054] VistaDepth: Frequency Modulation with Bias Reweighting for Enhanced Far-range Depth Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.15095v4](http://arxiv.org/pdf/2504.15095v4)**

> **作者:** Mingxia Zhan; Li Zhang; Xiaomeng Chu; Beibei Wang; Yanyong Zhang
>
> **摘要:** Monocular depth estimation predicts per-pixel depth from a single RGB image. While recent methods have shown promise by leveraging diffusion models, they often struggle to accurately reconstruct far-range regions. This difficulty stems from two compounding factors. First, the standard spatially uniform diffusion objective fails to adapt to the varying frequency content across a depth map. Second, the long-tail depth distribution heavily biases models toward near-range regions. To address these limitations, we introduce VistaDepth, a novel framework named for its ability to accurately reconstruct far-range vistas, which integrates adaptive frequency-domain feature processing with an adaptive loss-balancing mechanism into the diffusion pipeline. Central to our approach is the Latent Frequency Modulation module, which dynamically refines spectral responses in the latent feature space, effectively preserving structural detail. Additionally, we introduce BiasMap, a mechanism that applies adaptive weights directly to the diffusion loss in the latent space, focusing supervision on under-represented far-range regions. These innovations collectively achieve superior depth perception performance across near- and far-range depths while preserving fine detail. Experiments show that VistaDepth achieves state-of-the-art performance for diffusion-based MDE, particularly excelling in reconstructing detailed and accurate depth in far-range regions.
>
---
#### [replaced 055] R-LiViT: A LiDAR-Visual-Thermal Dataset Enabling Vulnerable Road User Focused Roadside Perception
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.17122v3](http://arxiv.org/pdf/2503.17122v3)**

> **作者:** Jonas Mirlach; Lei Wan; Andreas Wiedholz; Hannan Ejaz Keen; Andreas Eich
>
> **备注:** 11 pages, 8 figures, accepted at ICCV2025
>
> **摘要:** In autonomous driving, the integration of roadside perception systems is essential for overcoming occlusion challenges and enhancing the safety of Vulnerable Road Users(VRUs). While LiDAR and visual (RGB) sensors are commonly used, thermal imaging remains underrepresented in datasets, despite its acknowledged advantages for VRU detection in extreme lighting conditions. In this paper, we present R-LiViT, the first dataset to combine LiDAR, RGB, and thermal imaging from a roadside perspective, with a strong focus on VRUs. R-LiViT captures three intersections during both day and night, ensuring a diverse dataset. It includes 10,000 LiDAR frames and 2,400 temporally and spatially aligned RGB and thermal images across 150 traffic scenarios, with 7 and 8 annotated classes respectively, providing a comprehensive resource for tasks such as object detection and tracking. The dataset and the code for reproducing our evaluation results are made publicly available.
>
---
#### [replaced 056] $S^2M^2$: Scalable Stereo Matching Model for Reliable Depth Estimation
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.13229v3](http://arxiv.org/pdf/2507.13229v3)**

> **作者:** Junhong Min; Youngpil Jeon; Jimin Kim; Minyong Choi
>
> **备注:** 8 pages, 5 figures, ICCV accepted paper
>
> **摘要:** The pursuit of a generalizable stereo matching model, capable of performing well across varying resolutions and disparity ranges without dataset-specific fine-tuning, has revealed a fundamental trade-off. Iterative local search methods achieve high scores on constrained benchmarks, but their core mechanism inherently limits the global consistency required for true generalization. However, global matching architectures, while theoretically more robust, have historically been rendered infeasible by prohibitive computational and memory costs. We resolve this dilemma with $S^2M^2$: a global matching architecture that achieves state-of-the-art accuracy and high efficiency without relying on cost volume filtering or deep refinement stacks. Our design integrates a multi-resolution transformer for robust long-range correspondence, trained with a novel loss function that concentrates probability on feasible matches. This approach enables a more robust joint estimation of disparity, occlusion, and confidence. $S^2M^2$ establishes a new state of the art on Middlebury v3 and ETH3D benchmarks, significantly outperforming prior methods in most metrics while reconstructing high-quality details with competitive efficiency.
>
---
#### [replaced 057] Can GPT-4o mini and Gemini 2.0 Flash Predict Fine-Grained Fashion Product Attributes? A Zero-Shot Analysis
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.09950v2](http://arxiv.org/pdf/2507.09950v2)**

> **作者:** Shubham Shukla; Kunal Sonalkar
>
> **备注:** Version 2: Added a missing citation
>
> **摘要:** The fashion retail business is centered around the capacity to comprehend products. Product attribution helps in comprehending products depending on the business process. Quality attribution improves the customer experience as they navigate through millions of products offered by a retail website. It leads to well-organized product catalogs. In the end, product attribution directly impacts the 'discovery experience' of the customer. Although large language models (LLMs) have shown remarkable capabilities in understanding multimodal data, their performance on fine-grained fashion attribute recognition remains under-explored. This paper presents a zero-shot evaluation of state-of-the-art LLMs that balance performance with speed and cost efficiency, mainly GPT-4o-mini and Gemini 2.0 Flash. We have used the dataset DeepFashion-MultiModal (https://github.com/yumingj/DeepFashion-MultiModal) to evaluate these models in the attribution tasks of fashion products. Our study evaluates these models across 18 categories of fashion attributes, offering insight into where these models excel. We only use images as the sole input for product information to create a constrained environment. Our analysis shows that Gemini 2.0 Flash demonstrates the strongest overall performance with a macro F1 score of 56.79% across all attributes, while GPT-4o-mini scored a macro F1 score of 43.28%. Through detailed error analysis, our findings provide practical insights for deploying these LLMs in production e-commerce product attribution-related tasks and highlight the need for domain-specific fine-tuning approaches. This work also lays the groundwork for future research in fashion AI and multimodal attribute extraction.
>
---
#### [replaced 058] The Cooperative Network Architecture: Learning Structured Networks as Representation of Sensory Patterns
- **分类: cs.CV; cs.AI; cs.LG; cs.NE**

- **链接: [http://arxiv.org/pdf/2407.05650v4](http://arxiv.org/pdf/2407.05650v4)**

> **作者:** Pascal J. Sager; Jan M. Deriu; Benjamin F. Grewe; Thilo Stadelmann; Christoph von der Malsburg
>
> **摘要:** We introduce the Cooperative Network Architecture (CNA), a model that represents sensory signals using structured, recurrently connected networks of neurons, termed "nets." Nets are dynamically assembled from overlapping net fragments, which are learned based on statistical regularities in sensory input. This architecture offers robustness to noise, deformation, and out-of-distribution data, addressing challenges in current vision systems from a novel perspective. We demonstrate that net fragments can be learned without supervision and flexibly recombined to encode novel patterns, enabling figure completion and resilience to noise. Our findings establish CNA as a promising paradigm for developing neural representations that integrate local feature processing with global structure formation, providing a foundation for future research on invariant object recognition.
>
---
#### [replaced 059] Co-AttenDWG: Co-Attentive Dimension-Wise Gating and Expert Fusion for Multi-Modal Offensive Content Detection
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19010v2](http://arxiv.org/pdf/2505.19010v2)**

> **作者:** Md. Mithun Hossain; Md. Shakil Hossain; Sudipto Chaki; M. F. Mridha
>
> **摘要:** Multi-modal learning has emerged as a crucial research direction, as integrating textual and visual information can substantially enhance performance in tasks such as classification, retrieval, and scene understanding. Despite advances with large pre-trained models, existing approaches often suffer from insufficient cross-modal interactions and rigid fusion strategies, failing to fully harness the complementary strengths of different modalities. To address these limitations, we propose Co-AttenDWG, co-attention with dimension-wise gating, and expert fusion. Our approach first projects textual and visual features into a shared embedding space, where a dedicated co-attention mechanism enables simultaneous, fine-grained interactions between modalities. This is further strengthened by a dimension-wise gating network, which adaptively modulates feature contributions at the channel level to emphasize salient information. In parallel, dual-path encoders independently refine modality-specific representations, while an additional cross-attention layer aligns the modalities further. The resulting features are aggregated via an expert fusion module that integrates learned gating and self-attention, yielding a robust unified representation. Experimental results on the MIMIC and SemEval Memotion 1.0 datasets show that Co-AttenDWG achieves state-of-the-art performance and superior cross-modal alignment, highlighting its effectiveness for diverse multi-modal applications.
>
---
#### [replaced 060] TurboReg: TurboClique for Robust and Efficient Point Cloud Registration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.01439v3](http://arxiv.org/pdf/2507.01439v3)**

> **作者:** Shaocheng Yan; Pengcheng Shi; Zhenjun Zhao; Kaixin Wang; Kuang Cao; Ji Wu; Jiayuan Li
>
> **备注:** ICCV-2025 Accepted Paper
>
> **摘要:** Robust estimation is essential in correspondence-based Point Cloud Registration (PCR). Existing methods using maximal clique search in compatibility graphs achieve high recall but suffer from exponential time complexity, limiting their use in time-sensitive applications. To address this challenge, we propose a fast and robust estimator, TurboReg, built upon a novel lightweight clique, TurboClique, and a highly parallelizable Pivot-Guided Search (PGS) algorithm. First, we define the TurboClique as a 3-clique within a highly-constrained compatibility graph. The lightweight nature of the 3-clique allows for efficient parallel searching, and the highly-constrained compatibility graph ensures robust spatial consistency for stable transformation estimation. Next, PGS selects matching pairs with high SC$^2$ scores as pivots, effectively guiding the search toward TurboCliques with higher inlier ratios. Moreover, the PGS algorithm has linear time complexity and is significantly more efficient than the maximal clique search with exponential time complexity. Extensive experiments show that TurboReg achieves state-of-the-art performance across multiple real-world datasets, with substantial speed improvements. For example, on the 3DMatch+FCGF dataset, TurboReg (1K) operates $208.22\times$ faster than 3DMAC while also achieving higher recall. Our code is accessible at \href{https://github.com/Laka-3DV/TurboReg}{\texttt{TurboReg}}.
>
---
#### [replaced 061] I2VControl: Disentangled and Unified Video Motion Synthesis Control
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.17765v3](http://arxiv.org/pdf/2411.17765v3)**

> **作者:** Wanquan Feng; Tianhao Qi; Jiawei Liu; Mingzhen Sun; Pengqi Tu; Tianxiang Ma; Fei Dai; Songtao Zhao; Siyu Zhou; Qian He
>
> **备注:** Accepted to ICCV 2025. Project page: https://wanquanf.github.io/I2VControl
>
> **摘要:** Motion controllability is crucial in video synthesis. However, most previous methods are limited to single control types, and combining them often results in logical conflicts. In this paper, we propose a disentangled and unified framework, namely I2VControl, to overcome the logical conflicts. We rethink camera control, object dragging, and motion brush, reformulating all tasks into a consistent representation based on point trajectories, each managed by a dedicated formulation. Accordingly, we propose a spatial partitioning strategy, where each unit is assigned to a concomitant control category, enabling diverse control types to be dynamically orchestrated within a single synthesis pipeline without conflicts. Furthermore, we design an adapter structure that functions as a plug-in for pre-trained models and is agnostic to specific model architectures. We conduct extensive experiments, achieving excellent performance on various control tasks, and our method further facilitates user-driven creative combinations, enhancing innovation and creativity. Project page: https://wanquanf.github.io/I2VControl .
>
---
#### [replaced 062] Unsupervised Multi-Parameter Inverse Solving for Reducing Ring Artifacts in 3D X-Ray CBCT
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.05853v3](http://arxiv.org/pdf/2412.05853v3)**

> **作者:** Qing Wu; Hongjiang Wei; Jingyi Yu; Yuyao Zhang
>
> **摘要:** Ring artifacts are prevalent in 3D cone-beam computed tomography (CBCT) due to non-ideal responses of X-ray detectors, substantially affecting image quality and diagnostic reliability. Existing state-of-the-art (SOTA) ring artifact reduction (RAR) methods rely on supervised learning with large-scale paired CT datasets. While effective in-domain, supervised methods tend to struggle to fully capture the physical characteristics of ring artifacts, leading to pronounced performance drops in complex real-world acquisitions. Moreover, their scalability to 3D CBCT is limited by high memory demands. In this work, we propose Riner, a new unsupervised RAR method. Based on a theoretical analysis of ring artifact formation, we reformulate RAR as a multi-parameter inverse problem, where the non-ideal responses of X-ray detectors are parameterized as solvable physical variables. Using a new differentiable forward model, Riner can jointly learn the implicit neural representation of artifact-free images and estimate the physical parameters directly from CT measurements, without external training data. Additionally, Riner is memory-friendly due to its ray-based optimization, enhancing its usability in large-scale 3D CBCT. Experiments on both simulated and real-world datasets show Riner outperforms existing SOTA supervised methods.
>
---
#### [replaced 063] Exploring Textual Semantics Diversity for Image Transmission in Semantic Communication Systems using Visual Language Model
- **分类: cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2503.19386v2](http://arxiv.org/pdf/2503.19386v2)**

> **作者:** Peishan Huang; Dong Li
>
> **摘要:** In recent years, the rapid development of machine learning has brought reforms and challenges to traditional communication systems. Semantic communication has appeared as an effective strategy to effectively extract relevant semantic signals semantic segmentation labels and image features for image transmission. However, the insufficient number of extracted semantic features of images will potentially result in a low reconstruction accuracy, which hinders the practical applications and still remains challenging for solving. In order to fill this gap, this letter proposes a multi-text transmission semantic communication (Multi-SC) system, which uses the visual language model (VLM) to assist in the transmission of image semantic signals. Unlike previous image transmission semantic communication systems, the proposed system divides the image into multiple blocks and extracts multiple text information from the image using a modified large language and visual assistant (LLaVA), and combines semantic segmentation tags with semantic text for image recovery. Simulation results show that the proposed text semantics diversity scheme can significantly improve the reconstruction accuracy compared with related works.
>
---
#### [replaced 064] Application of Vision-Language Model to Pedestrians Behavior and Scene Understanding in Autonomous Driving
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2501.06680v2](http://arxiv.org/pdf/2501.06680v2)**

> **作者:** Haoxiang Gao; Li Zhang; Yu Zhao; Zhou Yang; Jinghan Cao
>
> **摘要:** Vision-language models (VLMs) have become a promising approach to enhancing perception and decision-making in autonomous driving. The gap remains in applying VLMs to understand complex scenarios interacting with pedestrians and efficient vehicle deployment. In this paper, we propose a knowledge distillation method that transfers knowledge from large-scale vision-language foundation models to efficient vision networks, and we apply it to pedestrian behavior prediction and scene understanding tasks, achieving promising results in generating more diverse and comprehensive semantic attributes. We also utilize multiple pre-trained models and ensemble techniques to boost the model's performance. We further examined the effectiveness of the model after knowledge distillation; the results show significant metric improvements in open-vocabulary perception and trajectory prediction tasks, which can potentially enhance the end-to-end performance of autonomous driving.
>
---
#### [replaced 065] Addressing Representation Collapse in Vector Quantized Models with One Linear Layer
- **分类: cs.LG; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.02038v2](http://arxiv.org/pdf/2411.02038v2)**

> **作者:** Yongxin Zhu; Bocheng Li; Yifei Xin; Zhihua Xia; Linli Xu
>
> **备注:** Accepted at ICCV2025
>
> **摘要:** Vector Quantization (VQ) is essential for discretizing continuous representations in unsupervised learning but suffers from representation collapse, causing low codebook utilization and limiting scalability. Existing solutions often rely on complex optimizations or reduce latent dimensionality, which compromises model capacity and fails to fully solve the problem. We identify the root cause as disjoint codebook optimization, where only a few code vectors are updated via gradient descent. To fix this, we propose \textbf{Sim}ple\textbf{VQ}, which reparameterizes code vectors through a learnable linear transformation layer over a latent basis, optimizing the \textit{entire linear space} rather than nearest \textit{individual code vectors}. Although the multiplication of two linear matrices is equivalent to applying a single linear layer, this simple approach effectively prevents collapse. Extensive experiments on image and audio tasks demonstrate that SimVQ improves codebook usage, is easy to implement, and generalizes well across modalities and architectures.
>
---
#### [replaced 066] ChartM$^3$: Benchmarking Chart Editing with Multimodal Instructions
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.21167v2](http://arxiv.org/pdf/2507.21167v2)**

> **作者:** Donglu Yang; Liang Zhang; Zihao Yue; Liangyu Chen; Yichen Xu; Wenxuan Wang; Qin Jin
>
> **摘要:** Charts are a fundamental visualization format widely used in data analysis across research and industry. While enabling users to edit charts based on high-level intentions is of great practical value, existing methods primarily rely on natural language instructions, which are often too ambiguous to support fine-grained editing. In this work, we introduce a novel paradigm for multimodal chart editing, where user intent is expressed through a combination of natural language and visual indicators that explicitly highlight the elements to be modified. To support this paradigm, we present Chart$\text{M}^3$, a new benchmark for Multimodal chart editing with Multi-level complexity and Multi-perspective evaluation. Chart$\text{M}^3$ contains 1,000 samples spanning four levels of editing difficulty. Each sample includes triplets in the form of (chart, code, multimodal instructions). To comprehensively evaluate chart editing models, Chart$\text{M}^3$ provides metrics that assess both visual appearance and code correctness. Our benchmark reveals significant limitations in current multimodal large language models (MLLMs), including GPT-4o, particularly in their ability to interpret and act on visual indicators. To address this, we construct Chart$\text{M}^3$-Train, a large-scale training set with 24,000 multimodal chart editing samples. Fine-tuning MLLMs on this dataset leads to substantial improvements, demonstrating the importance of multimodal supervision in building practical chart editing systems. Our datasets, codes, and evaluation tools are available at https://github.com/MLrollIT/ChartM3. %https://github.com/MLrollIT/ChartM3Our datasets, codes, and evaluation tools are available at https://github.com/yaolinli/VCE.
>
---
#### [replaced 067] Advances in Feed-Forward 3D Reconstruction and View Synthesis: A Survey
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.14501v2](http://arxiv.org/pdf/2507.14501v2)**

> **作者:** Jiahui Zhang; Yuelei Li; Anpei Chen; Muyu Xu; Kunhao Liu; Jianyuan Wang; Xiao-Xiao Long; Hanxue Liang; Zexiang Xu; Hao Su; Christian Theobalt; Christian Rupprecht; Andrea Vedaldi; Hanspeter Pfister; Shijian Lu; Fangneng Zhan
>
> **备注:** A project page associated with this survey is available at https://fnzhan.com/projects/Feed-Forward-3D
>
> **摘要:** 3D reconstruction and view synthesis are foundational problems in computer vision, graphics, and immersive technologies such as augmented reality (AR), virtual reality (VR), and digital twins. Traditional methods rely on computationally intensive iterative optimization in a complex chain, limiting their applicability in real-world scenarios. Recent advances in feed-forward approaches, driven by deep learning, have revolutionized this field by enabling fast and generalizable 3D reconstruction and view synthesis. This survey offers a comprehensive review of feed-forward techniques for 3D reconstruction and view synthesis, with a taxonomy according to the underlying representation architectures including point cloud, 3D Gaussian Splatting (3DGS), Neural Radiance Fields (NeRF), etc. We examine key tasks such as pose-free reconstruction, dynamic 3D reconstruction, and 3D-aware image and video synthesis, highlighting their applications in digital humans, SLAM, robotics, and beyond. In addition, we review commonly used datasets with detailed statistics, along with evaluation protocols for various downstream tasks. We conclude by discussing open research challenges and promising directions for future work, emphasizing the potential of feed-forward approaches to advance the state of the art in 3D vision.
>
---
#### [replaced 068] DriveIndia: An Object Detection Dataset for Diverse Indian Traffic Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.19912v2](http://arxiv.org/pdf/2507.19912v2)**

> **作者:** Rishav Kumar; D. Santhosh Reddy; P. Rajalakshmi
>
> **备注:** Accepted at ITSC 2025 Conference
>
> **摘要:** We introduce DriveIndia, a large-scale object detection dataset purpose-built to capture the complexity and unpredictability of Indian traffic environments. The dataset contains 66,986 high-resolution images annotated in YOLO format across 24 traffic-relevant object categories, encompassing diverse conditions such as varied weather (fog, rain), illumination changes, heterogeneous road infrastructure, and dense, mixed traffic patterns and collected over 120+ hours and covering 3,400+ kilometers across urban, rural, and highway routes. DriveIndia offers a comprehensive benchmark for real-world autonomous driving challenges. We provide baseline results using state-of-the-art YOLO family models, with the top-performing variant achieving a mAP50 of 78.7\%. Designed to support research in robust, generalizable object detection under uncertain road conditions, DriveIndia will be publicly available via the TiHAN-IIT Hyderabad dataset repository (https://tihan.iith.ac.in/tiand-datasets/).
>
---
#### [replaced 069] RecConv: Efficient Recursive Convolutions for Multi-Frequency Representations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.19628v3](http://arxiv.org/pdf/2412.19628v3)**

> **作者:** Mingshu Zhao; Yi Luo; Yong Ouyang
>
> **备注:** Tech report; Added supplementary material; Added more experiments;
>
> **摘要:** Recent advances in vision transformers (ViTs) have demonstrated the advantage of global modeling capabilities, prompting widespread integration of large-kernel convolutions for enlarging the effective receptive field (ERF). However, the quadratic scaling of parameter count and computational complexity (FLOPs) with respect to kernel size poses significant efficiency and optimization challenges. This paper introduces RecConv, a recursive decomposition strategy that efficiently constructs multi-frequency representations using small-kernel convolutions. RecConv establishes a linear relationship between parameter growth and decomposing levels which determines the effective receptive field $k\times 2^\ell$ for a base kernel $k$ and $\ell$ levels of decomposition, while maintaining constant FLOPs regardless of the ERF expansion. Specifically, RecConv achieves a parameter expansion of only $\ell+2$ times and a maximum FLOPs increase of $5/3$ times, compared to the exponential growth ($4^\ell$) of standard and depthwise convolutions. RecNeXt-M3 outperforms RepViT-M1.1 by 1.9 $AP^{box}$ on COCO with similar FLOPs. This innovation provides a promising avenue towards designing efficient and compact networks across various modalities. Codes and models can be found at https://github.com/suous/RecNeXt.
>
---
#### [replaced 070] Predict Patient Self-reported Race from Skin Histological Images
- **分类: cs.CV; cs.CE**

- **链接: [http://arxiv.org/pdf/2507.21912v2](http://arxiv.org/pdf/2507.21912v2)**

> **作者:** Shengjia Chen; Ruchika Verma; Kevin Clare; Jannes Jegminat; Eugenia Alleva; Kuan-lin Huang; Brandon Veremis; Thomas Fuchs; Gabriele Campanella
>
> **备注:** Accepted to the MICCAI Workshop on Fairness of AI in Medical Imaging (FAIMI), 2025
>
> **摘要:** Artificial Intelligence (AI) has demonstrated success in computational pathology (CPath) for disease detection, biomarker classification, and prognosis prediction. However, its potential to learn unintended demographic biases, particularly those related to social determinants of health, remains understudied. This study investigates whether deep learning models can predict self-reported race from digitized dermatopathology slides and identifies potential morphological shortcuts. Using a multisite dataset with a racially diverse population, we apply an attention-based mechanism to uncover race-associated morphological features. After evaluating three dataset curation strategies to control for confounding factors, the final experiment showed that White and Black demographic groups retained high prediction performance (AUC: 0.799, 0.762), while overall performance dropped to 0.663. Attention analysis revealed the epidermis as a key predictive feature, with significant performance declines when these regions were removed. These findings highlight the need for careful data curation and bias mitigation to ensure equitable AI deployment in pathology. Code available at: https://github.com/sinai-computational-pathology/CPath_SAIF.
>
---
#### [replaced 071] RTMap: Real-Time Recursive Mapping with Change Detection and Localization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.00980v2](http://arxiv.org/pdf/2507.00980v2)**

> **作者:** Yuheng Du; Sheng Yang; Lingxuan Wang; Zhenghua Hou; Chengying Cai; Zhitao Tan; Mingxia Chen; Shi-Sheng Huang; Qiang Li
>
> **摘要:** While recent online HD mapping methods relieve burdened offline pipelines and solve map freshness, they remain limited by perceptual inaccuracies, occlusion in dense traffic, and an inability to fuse multi-agent observations. We propose RTMap to enhance these single-traversal methods by persistently crowdsourcing a multi-traversal HD map as a self-evolutional memory. On onboard agents, RTMap simultaneously addresses three core challenges in an end-to-end fashion: (1) Uncertainty-aware positional modeling for HD map elements, (2) probabilistic-aware localization w.r.t. the crowdsourced prior-map, and (3) real-time detection for possible road structural changes. Experiments on several public autonomous driving datasets demonstrate our solid performance on both the prior-aided map quality and the localization accuracy, demonstrating our effectiveness of robustly serving downstream prediction and planning modules while gradually improving the accuracy and freshness of the crowdsourced prior-map asynchronously. Our source-code will be made publicly available at https://github.com/CN-ADLab/RTMap.
>
---
#### [replaced 072] FastTrackTr:Towards Fast Multi-Object Tracking with Transformers
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.15811v4](http://arxiv.org/pdf/2411.15811v4)**

> **作者:** Pan Liao; Feng Yang; Di Wu; Jinwen Yu; Wenhui Zhao; Dingwen Zhang
>
> **摘要:** Transformer-based multi-object tracking (MOT) methods have captured the attention of many researchers in recent years. However, these models often suffer from slow inference speeds due to their structure or other issues. To address this problem, we revisited the Joint Detection and Tracking (JDT) method by looking back at past approaches. By integrating the original JDT approach with some advanced theories, this paper employs an efficient method of information transfer between frames on the DETR, constructing a fast and novel JDT-type MOT framework: FastTrackTr. Thanks to the superiority of this information transfer method, our approach not only reduces the number of queries required during tracking but also avoids the excessive introduction of network structures, ensuring model simplicity. Experimental results indicate that our method has the potential to achieve real-time tracking and exhibits competitive tracking accuracy across multiple datasets.
>
---
#### [replaced 073] PolyPose: Localizing Deformable Anatomy in 3D from Sparse 2D X-ray Images using Polyrigid Transforms
- **分类: cs.CV; physics.med-ph**

- **链接: [http://arxiv.org/pdf/2505.19256v3](http://arxiv.org/pdf/2505.19256v3)**

> **作者:** Vivek Gopalakrishnan; Neel Dey; Polina Golland
>
> **备注:** Code available at https://github.com/eigenvivek/polypose
>
> **摘要:** Determining the 3D pose of a patient from a limited set of 2D X-ray images is a critical task in interventional settings. While preoperative volumetric imaging (e.g., CT and MRI) provides precise 3D localization and visualization of anatomical targets, these modalities cannot be acquired during procedures, where fast 2D imaging (X-ray) is used instead. To integrate volumetric guidance into intraoperative procedures, we present PolyPose, a simple and robust method for deformable 2D/3D registration. PolyPose parameterizes complex 3D deformation fields as a composition of rigid transforms, leveraging the biological constraint that individual bones do not bend in typical motion. Unlike existing methods that either assume no inter-joint movement or fail outright in this under-determined setting, our polyrigid formulation enforces anatomically plausible priors that respect the piecewise rigid nature of human movement. This approach eliminates the need for expensive deformation regularizers that require patient- and procedure-specific hyperparameter optimization. Across extensive experiments on diverse datasets from orthopedic surgery and radiotherapy, we show that this strong inductive bias enables PolyPose to successfully align the patient's preoperative volume to as few as two X-ray images, thereby providing crucial 3D guidance in challenging sparse-view and limited-angle settings where current registration methods fail.
>
---
#### [replaced 074] Gaussian On-the-Fly Splatting: A Progressive Framework for Robust Near Real-Time 3DGS Optimization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.13086v2](http://arxiv.org/pdf/2503.13086v2)**

> **作者:** Yiwei Xu; Yifei Yu; Wentian Gan; Tengfei Wang; Zongqian Zhan; Hao Cheng; Xin Wang
>
> **摘要:** 3D Gaussian Splatting (3DGS) achieves high-fidelity rendering with fast real-time performance, but existing methods rely on offline training after full Structure-from-Motion (SfM) processing. In contrast, this work introduces Gaussian on-the-fly Splatting (abbreviated as On-the-Fly GS), a progressive framework enabling near real-time 3DGS optimization during image capture. As each image arrives, its pose and sparse points are updated via On-the-Fly SfM, and newly optimized Gaussians are immediately integrated into the 3DGS field. To achieve this, we propose a progressive Local & Semi-Global optimization to prioritize the new image and its neighbors by their corresponding overlapping relationship, allowing the new image and its overlapping images to get more training. To further stabilize training across previous and new images, an adaptive learning rate schedule balances the iterations and the learning rate. Extensive experiments on multiple benchmarks show that our On-the-Fly GS reduces training time significantly, optimizing each new image in seconds with minimal rendering loss, offering one of the first practical steps toward rapid, progressive 3DGS reconstruction.
>
---
#### [replaced 075] The Importance of Facial Features in Vision-based Sign Language Recognition: Eyes, Mouth or Full Face?
- **分类: cs.CV; cs.CL; eess.IV**

- **链接: [http://arxiv.org/pdf/2507.20884v2](http://arxiv.org/pdf/2507.20884v2)**

> **作者:** Dinh Nam Pham; Eleftherios Avramidis
>
> **备注:** Accepted at 9th International Workshop on Sign Language Translation and Avatar Technologies @ ACM IVA'25
>
> **摘要:** Non-manual facial features play a crucial role in sign language communication, yet their importance in automatic sign language recognition (ASLR) remains underexplored. While prior studies have shown that incorporating facial features can improve recognition, related work often relies on hand-crafted feature extraction and fails to go beyond the comparison of manual features versus the combination of manual and facial features. In this work, we systematically investigate the contribution of distinct facial regionseyes, mouth, and full faceusing two different deep learning models (a CNN-based model and a transformer-based model) trained on an SLR dataset of isolated signs with randomly selected classes. Through quantitative performance and qualitative saliency map evaluation, we reveal that the mouth is the most important non-manual facial feature, significantly improving accuracy. Our findings highlight the necessity of incorporating facial features in ASLR.
>
---
#### [replaced 076] TextSAM-EUS: Text Prompt Learning for SAM to Accurately Segment Pancreatic Tumor in Endoscopic Ultrasound
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.18082v3](http://arxiv.org/pdf/2507.18082v3)**

> **作者:** Pascal Spiegler; Taha Koleilat; Arash Harirpoush; Corey S. Miller; Hassan Rivaz; Marta Kersten-Oertel; Yiming Xiao
>
> **备注:** Accepted to ICCV 2025 Workshop CVAMD
>
> **摘要:** Pancreatic cancer carries a poor prognosis and relies on endoscopic ultrasound (EUS) for targeted biopsy and radiotherapy. However, the speckle noise, low contrast, and unintuitive appearance of EUS make segmentation of pancreatic tumors with fully supervised deep learning (DL) models both error-prone and dependent on large, expert-curated annotation datasets. To address these challenges, we present TextSAM-EUS, a novel, lightweight, text-driven adaptation of the Segment Anything Model (SAM) that requires no manual geometric prompts at inference. Our approach leverages text prompt learning (context optimization) through the BiomedCLIP text encoder in conjunction with a LoRA-based adaptation of SAM's architecture to enable automatic pancreatic tumor segmentation in EUS, tuning only 0.86% of the total parameters. On the public Endoscopic Ultrasound Database of the Pancreas, TextSAM-EUS with automatic prompts attains 82.69% Dice and 85.28% normalized surface distance (NSD), and with manual geometric prompts reaches 83.10% Dice and 85.70% NSD, outperforming both existing state-of-the-art (SOTA) supervised DL models and foundation models (e.g., SAM and its variants). As the first attempt to incorporate prompt learning in SAM-based medical image segmentation, TextSAM-EUS offers a practical option for efficient and robust automatic EUS segmentation. Code is available at https://github.com/HealthX-Lab/TextSAM-EUS .
>
---
#### [replaced 077] OpenEarthSensing: Large-Scale Fine-Grained Benchmark for Open-World Remote Sensing
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2502.20668v2](http://arxiv.org/pdf/2502.20668v2)**

> **作者:** Xiang Xiang; Zhuo Xu; Yao Deng; Qinhao Zhou; Yifan Liang; Ke Chen; Qingfang Zheng; Yaowei Wang; Xilin Chen; Wen Gao
>
> **备注:** Full version with dataset details in Appendix
>
> **摘要:** The advancement of remote sensing, including satellite systems, facilitates the continuous acquisition of remote sensing imagery globally, introducing novel challenges for achieving open-world tasks. Deployed models need to continuously adjust to a constant influx of new data, which frequently exhibits diverse shifts from the data encountered during the training phase. To effectively handle the new data, models are required to detect semantic shifts, adapt to covariate shifts, and continuously update their parameters without forgetting learned knowledge, as has been considered in works on a variety of open-world tasks. However, existing studies are typically conducted within a single dataset to simulate realistic conditions, with a lack of large-scale benchmarks capable of evaluating multiple open-world tasks. In this paper, we introduce \textbf{OpenEarthSensing (OES)}, a large-scale fine-grained benchmark for open-world remote sensing. OES includes 189 scene and object categories, covering the vast majority of potential semantic shifts that may occur in the real world. Additionally, to provide a more comprehensive testbed for evaluating the generalization performance, OES encompasses five data domains with significant covariate shifts, including two RGB satellite domains, one RGB aerial domain, one multispectral RGB domain, and one infrared domain. We evaluate the baselines and existing methods for diverse tasks on OES, demonstrating that it serves as a meaningful and challenging benchmark for open-world remote sensing. The proposed dataset OES is available at https://haiv-lab.github.io/OES.
>
---
#### [replaced 078] See Different, Think Better: Visual Variations Mitigating Hallucinations in LVLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.22003v2](http://arxiv.org/pdf/2507.22003v2)**

> **作者:** Ziyun Dai; Xiaoqiang Li; Shaohua Zhang; Yuanchen Wu; Jide Li
>
> **备注:** Accepted by ACM MM25
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities in visual understanding and multimodal reasoning. However, LVLMs frequently exhibit hallucination phenomena, manifesting as the generated textual responses that demonstrate inconsistencies with the provided visual content. Existing hallucination mitigation methods are predominantly text-centric, the challenges of visual-semantic alignment significantly limit their effectiveness, especially when confronted with fine-grained visual understanding scenarios. To this end, this paper presents ViHallu, a Vision-Centric Hallucination mitigation framework that enhances visual-semantic alignment through Visual Variation Image Generation and Visual Instruction Construction. ViHallu introduces visual variation images with controllable visual alterations while maintaining the overall image structure. These images, combined with carefully constructed visual instructions, enable LVLMs to better understand fine-grained visual content through fine-tuning, allowing models to more precisely capture the correspondence between visual content and text, thereby enhancing visual-semantic alignment. Extensive experiments on multiple benchmarks show that ViHallu effectively enhances models' fine-grained visual understanding while significantly reducing hallucination tendencies. Furthermore, we release ViHallu-Instruction, a visual instruction dataset specifically designed for hallucination mitigation and visual-semantic alignment. Code is available at https://github.com/oliviadzy/ViHallu.
>
---
#### [replaced 079] Positive-Augmented Contrastive Learning for Vision-and-Language Evaluation and Training
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2410.07336v2](http://arxiv.org/pdf/2410.07336v2)**

> **作者:** Sara Sarto; Nicholas Moratelli; Marcella Cornia; Lorenzo Baraldi; Rita Cucchiara
>
> **备注:** International Journal of Computer Vision (2025)
>
> **摘要:** Despite significant advancements in caption generation, existing evaluation metrics often fail to capture the full quality or fine-grained details of captions. This is mainly due to their reliance on non-specific human-written references or noisy pre-training data. Still, finding an effective metric is crucial not only for captions evaluation but also for the generation phase. Metrics can indeed play a key role in the fine-tuning stage of captioning models, ultimately enhancing the quality of the generated captions. In this paper, we propose PAC-S++, a learnable metric that leverages the CLIP model, pre-trained on both web-collected and cleaned data and regularized through additional pairs of generated visual and textual positive samples. Exploiting this stronger and curated pre-training, we also apply PAC-S++ as a reward in the Self-Critical Sequence Training (SCST) stage typically employed to fine-tune captioning models. Extensive experiments on different image and video datasets highlight the effectiveness of PAC-S++ compared to popular metrics for the task, including its sensitivity to object hallucinations. Furthermore, we show that integrating PAC-S++ into the fine-tuning stage of a captioning model results in semantically richer captions with fewer repetitions and grammatical errors. Evaluations on out-of-domain benchmarks further demonstrate the efficacy of our fine-tuning approach in enhancing model capabilities. Source code and trained models are publicly available at: https://github.com/aimagelab/pacscore.
>
---
#### [replaced 080] Embodied Web Agents: Bridging Physical-Digital Realms for Integrated Agent Intelligence
- **分类: cs.AI; cs.CL; cs.CV; cs.MM; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.15677v3](http://arxiv.org/pdf/2506.15677v3)**

> **作者:** Yining Hong; Rui Sun; Bingxuan Li; Xingcheng Yao; Maxine Wu; Alexander Chien; Da Yin; Ying Nian Wu; Zhecan James Wang; Kai-Wei Chang
>
> **摘要:** AI agents today are mostly siloed - they either retrieve and reason over vast amount of digital information and knowledge obtained online; or interact with the physical world through embodied perception, planning and action - but rarely both. This separation limits their ability to solve tasks that require integrated physical and digital intelligence, such as cooking from online recipes, navigating with dynamic map data, or interpreting real-world landmarks using web knowledge. We introduce Embodied Web Agents, a novel paradigm for AI agents that fluidly bridge embodiment and web-scale reasoning. To operationalize this concept, we first develop the Embodied Web Agents task environments, a unified simulation platform that tightly integrates realistic 3D indoor and outdoor environments with functional web interfaces. Building upon this platform, we construct and release the Embodied Web Agents Benchmark, which encompasses a diverse suite of tasks including cooking, navigation, shopping, tourism, and geolocation - all requiring coordinated reasoning across physical and digital realms for systematic assessment of cross-domain intelligence. Experimental results reveal significant performance gaps between state-of-the-art AI systems and human capabilities, establishing both challenges and opportunities at the intersection of embodied cognition and web-scale knowledge access. All datasets, codes and websites are publicly available at our project page https://embodied-web-agent.github.io/.
>
---
#### [replaced 081] Anti-Inpainting: A Proactive Defense Approach against Malicious Diffusion-based Inpainters under Unknown Conditions
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.13023v2](http://arxiv.org/pdf/2505.13023v2)**

> **作者:** Yimao Guo; Zuomin Qu; Wei Lu; Xiangyang Luo
>
> **摘要:** With the increasing prevalence of diffusion-based malicious image manipulation, existing proactive defense methods struggle to safeguard images against tampering under unknown conditions. To address this, we propose Anti-Inpainting, a proactive defense approach that achieves protection comprising three novel modules. First, we introduce a multi-level deep feature extractor to obtain intricate features from the diffusion denoising process, enhancing protective effectiveness. Second, we design a multi-scale, semantic-preserving data augmentation technique to enhance the transferability of adversarial perturbations across unknown conditions. Finally, we propose a selection-based distribution deviation optimization strategy to bolster protection against manipulations guided by diverse random seeds. Extensive experiments on InpaintGuardBench and CelebA-HQ demonstrate that Anti-Inpainting effectively defends against diffusion-based inpainters under unknown conditions. Additionally, our approach demonstrates robustness against various image purification methods and transferability across different diffusion model versions.
>
---
#### [replaced 082] Harnessing Diffusion-Yielded Score Priors for Image Restoration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.20590v2](http://arxiv.org/pdf/2507.20590v2)**

> **作者:** Xinqi Lin; Fanghua Yu; Jinfan Hu; Zhiyuan You; Wu Shi; Jimmy S. Ren; Jinjin Gu; Chao Dong
>
> **摘要:** Deep image restoration models aim to learn a mapping from degraded image space to natural image space. However, they face several critical challenges: removing degradation, generating realistic details, and ensuring pixel-level consistency. Over time, three major classes of methods have emerged, including MSE-based, GAN-based, and diffusion-based methods. However, they fail to achieve a good balance between restoration quality, fidelity, and speed. We propose a novel method, HYPIR, to address these challenges. Our solution pipeline is straightforward: it involves initializing the image restoration model with a pre-trained diffusion model and then fine-tuning it with adversarial training. This approach does not rely on diffusion loss, iterative sampling, or additional adapters. We theoretically demonstrate that initializing adversarial training from a pre-trained diffusion model positions the initial restoration model very close to the natural image distribution. Consequently, this initialization improves numerical stability, avoids mode collapse, and substantially accelerates the convergence of adversarial training. Moreover, HYPIR inherits the capabilities of diffusion models with rich user control, enabling text-guided restoration and adjustable texture richness. Requiring only a single forward pass, it achieves faster convergence and inference speed than diffusion-based methods. Extensive experiments show that HYPIR outperforms previous state-of-the-art methods, achieving efficient and high-quality image restoration.
>
---
