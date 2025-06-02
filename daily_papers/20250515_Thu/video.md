# 计算机视觉 cs.CV

- **最新发布 93 篇**

- **更新 41 篇**

## 最新发布

#### [new 001] WSCIF: A Weakly-Supervised Color Intelligence Framework for Tactical Anomaly Detection in Surveillance Keyframes
- **分类: cs.CV; cs.AI; es: 68T10, 68T05, 62H35, 68U10; I.4.9; I.5.1; I.2.10**

- **简介: 该论文属于监控视频异常检测任务，旨在解决无标签、数据敏感环境下传统深度学习模型难以部署的问题。提出基于颜色特征的轻量框架WSCIF，融合KMeans聚类与RGB直方图建模，实现关键帧结构异常和颜色突变的复合检测，验证了战术预警能力并突出色彩特征的低语义战场价值。**

- **链接: [http://arxiv.org/pdf/2505.09129v1](http://arxiv.org/pdf/2505.09129v1)**

> **作者:** Wei Meng
>
> **备注:** 17 pages, 3 figures, 3 tables. The paper proposes a lightweight weakly-supervised color intelligence model for tactical video anomaly detection, tested on anonymized African surveillance data
>
> **摘要:** The deployment of traditional deep learning models in high-risk security tasks in an unlabeled, data-non-exploitable video intelligence environment faces significant challenges. In this paper, we propose a lightweight anomaly detection framework based on color features for surveillance video clips in a high sensitivity tactical mission, aiming to quickly identify and interpret potential threat events under resource-constrained and data-sensitive conditions. The method fuses unsupervised KMeans clustering with RGB channel histogram modeling to achieve composite detection of structural anomalies and color mutation signals in key frames. The experiment takes an operation surveillance video occurring in an African country as a research sample, and successfully identifies multiple highly anomalous frames related to high-energy light sources, target presence, and reflective interference under the condition of no access to the original data. The results show that this method can be effectively used for tactical assassination warning, suspicious object screening and environmental drastic change monitoring with strong deployability and tactical interpretation value. The study emphasizes the importance of color features as low semantic battlefield signal carriers, and its battlefield intelligent perception capability will be further extended by combining graph neural networks and temporal modeling in the future.
>
---
#### [new 002] Crowd Scene Analysis using Deep Learning Techniques
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究拥挤场景分析，解决人群计数和异常检测任务。针对数据标注成本高及场景复杂性问题，提出自监督多列CNN模型；针对异常检测的环境干扰问题，设计基于VGG19和LSTM的时空模型，通过残差块提升性能，在多个数据集上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.08834v1](http://arxiv.org/pdf/2505.08834v1)**

> **作者:** Muhammad Junaid Asif
>
> **备注:** MS Graduate Research Thesis
>
> **摘要:** Our research is focused on two main applications of crowd scene analysis crowd counting and anomaly detection In recent years a large number of researches have been presented in the domain of crowd counting We addressed two main challenges in this domain 1 Deep learning models are datahungry paradigms and always need a large amount of annotated data for the training of algorithm It is timeconsuming and costly task to annotate such large amount of data Selfsupervised training is proposed to deal with this challenge 2 MCNN consists of multicolumns of CNN with different sizes of filters by presenting a novel approach based on a combination of selfsupervised training and MultiColumn CNN This enables the model to learn features at different levels and makes it effective in dealing with challenges of occluded scenes nonuniform density complex backgrounds and scale invariation The proposed model was evaluated on publicly available data sets such as ShanghaiTech and UCFQNRF by means of MAE and MSE A spatiotemporal model based on VGG19 is proposed for crowd anomaly detection addressing challenges like lighting environmental conditions unexpected objects and scalability The model extracts spatial and temporal features allowing it to be generalized to realworld scenes Spatial features are learned using CNN while temporal features are learned using LSTM blocks The model works on binary classification and can detect normal or abnormal behavior The models performance is improved by replacing fully connected layers with dense residual blocks Experiments on the Hockey Fight dataset and SCVD dataset show our models outperform other stateoftheart approaches
>
---
#### [new 003] Contactless Cardiac Pulse Monitoring Using Event Cameras
- **分类: cs.CV; cs.ET; cs.LG; eess.IV**

- **简介: 该论文研究基于事件相机的无接触心率监测任务，解决传统相机动态范围与时延限制问题。通过CNN模型从面部事件流中提取脉搏信号，实验显示事件相机（60/120 FPS）的RMSE（2.13-2.54 bpm）优于30 FPS传统相机（3.32 bpm），验证了该技术在远程医疗的潜力。**

- **链接: [http://arxiv.org/pdf/2505.09529v1](http://arxiv.org/pdf/2505.09529v1)**

> **作者:** Mohamed Moustafa; Joseph Lemley; Peter Corcoran
>
> **备注:** This paper is a preprint of a paper submitted to IEEE Access and is currently under review
>
> **摘要:** Time event cameras are a novel technology for recording scene information at extremely low latency and with low power consumption. Event cameras output a stream of events that encapsulate pixel-level light intensity changes within the scene, capturing information with a higher dynamic range and temporal resolution than traditional cameras. This study investigates the contact-free reconstruction of an individual's cardiac pulse signal from time event recording of their face using a supervised convolutional neural network (CNN) model. An end-to-end model is trained to extract the cardiac signal from a two-dimensional representation of the event stream, with model performance evaluated based on the accuracy of the calculated heart rate. The experimental results confirm that physiological cardiac information in the facial region is effectively preserved within the event stream, showcasing the potential of this novel sensor for remote heart rate monitoring. The model trained on event frames achieves a root mean square error (RMSE) of 3.32 beats per minute (bpm) compared to the RMSE of 2.92 bpm achieved by the baseline model trained on standard camera frames. Furthermore, models trained on event frames generated at 60 and 120 FPS outperformed the 30 FPS standard camera results, achieving an RMSE of 2.54 and 2.13 bpm, respectively.
>
---
#### [new 004] Promoting SAM for Camouflaged Object Detection via Selective Key Point-based Guidance
- **分类: cs.CV**

- **简介: 该论文研究伪装目标检测（COD）任务，旨在解决Segment Anything Model（SAM）在COD中效果不佳的问题。通过设计PPT-net预测目标存在概率，并结合关键点选择算法（KPS）对比引导SAM分割，实现了无需专业模型即可高效完成COD，在三个数据集上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.09123v1](http://arxiv.org/pdf/2505.09123v1)**

> **作者:** Guoying Liang; Su Yang
>
> **摘要:** Big model has emerged as a new research paradigm that can be applied to various down-stream tasks with only minor effort for domain adaption. Correspondingly, this study tackles Camouflaged Object Detection (COD) leveraging the Segment Anything Model (SAM). The previous studies declared that SAM is not workable for COD but this study reveals that SAM works if promoted properly, for which we devise a new framework to render point promotions: First, we develop the Promotion Point Targeting Network (PPT-net) to leverage multi-scale features in predicting the probabilities of camouflaged objects' presences at given candidate points over the image. Then, we develop a key point selection (KPS) algorithm to deploy both positive and negative point promotions contrastively to SAM to guide the segmentation. It is the first work to facilitate big model for COD and achieves plausible results experimentally over the existing methods on 3 data sets under 6 metrics. This study demonstrates an off-the-shelf methodology for COD by leveraging SAM, which gains advantage over designing professional models from scratch, not only in performance, but also in turning the problem to a less challenging task, that is, seeking informative but not exactly precise promotions.
>
---
#### [new 005] TUGS: Physics-based Compact Representation of Underwater Scenes by Tensorized Gaussian
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究水下3D场景重建任务，旨在解决现有方法无法准确建模物体与水体交互、计算成本高的问题。提出TUGS方法，通过张量化高阶高斯建模和物理驱动介质估计模块，降低参数量的同时精确模拟水下光衰减与散射效应，在真实数据中实现高效高质量渲染，适用于水下无人机。**

- **链接: [http://arxiv.org/pdf/2505.08811v1](http://arxiv.org/pdf/2505.08811v1)**

> **作者:** Shijie Lian; Ziyi Zhang; Laurence Tianruo Yang and; Mengyu Ren; Debin Liu; Hua Li
>
> **摘要:** Underwater 3D scene reconstruction is crucial for undewater robotic perception and navigation. However, the task is significantly challenged by the complex interplay between light propagation, water medium, and object surfaces, with existing methods unable to model their interactions accurately. Additionally, expensive training and rendering costs limit their practical application in underwater robotic systems. Therefore, we propose Tensorized Underwater Gaussian Splatting (TUGS), which can effectively solve the modeling challenges of the complex interactions between object geometries and water media while achieving significant parameter reduction. TUGS employs lightweight tensorized higher-order Gaussians with a physics-based underwater Adaptive Medium Estimation (AME) module, enabling accurate simulation of both light attenuation and backscatter effects in underwater environments. Compared to other NeRF-based and GS-based methods designed for underwater, TUGS is able to render high-quality underwater images with faster rendering speeds and less memory usage. Extensive experiments on real-world underwater datasets have demonstrated that TUGS can efficiently achieve superior reconstruction quality using a limited number of parameters, making it particularly suitable for memory-constrained underwater UAV applications
>
---
#### [new 006] MetaUAS: Universal Anomaly Segmentation with One-Prompt Meta-Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MetaUAS，解决通用视觉异常分割任务，旨在摆脱传统对视觉-语言模型和专用数据集的依赖。通过将异常检测转化为变化分割，利用合成图像对训练纯视觉模型，结合软特征对齐模块处理几何差异，实现仅需单张正常图像提示即可检测任意异常，无需语言指导且训练效率高。**

- **链接: [http://arxiv.org/pdf/2505.09265v1](http://arxiv.org/pdf/2505.09265v1)**

> **作者:** Bin-Bin Gao
>
> **备注:** Accepted by NeurIPS 2024
>
> **摘要:** Zero- and few-shot visual anomaly segmentation relies on powerful vision-language models that detect unseen anomalies using manually designed textual prompts. However, visual representations are inherently independent of language. In this paper, we explore the potential of a pure visual foundation model as an alternative to widely used vision-language models for universal visual anomaly segmentation. We present a novel paradigm that unifies anomaly segmentation into change segmentation. This paradigm enables us to leverage large-scale synthetic image pairs, featuring object-level and local region changes, derived from existing image datasets, which are independent of target anomaly datasets. We propose a one-prompt Meta-learning framework for Universal Anomaly Segmentation (MetaUAS) that is trained on this synthetic dataset and then generalizes well to segment any novel or unseen visual anomalies in the real world. To handle geometrical variations between prompt and query images, we propose a soft feature alignment module that bridges paired-image change perception and single-image semantic segmentation. This is the first work to achieve universal anomaly segmentation using a pure vision model without relying on special anomaly detection datasets and pre-trained visual-language models. Our method effectively and efficiently segments any anomalies with only one normal image prompt and enjoys training-free without guidance from language. Our MetaUAS significantly outperforms previous zero-shot, few-shot, and even full-shot anomaly segmentation methods. The code and pre-trained models are available at https://github.com/gaobb/MetaUAS.
>
---
#### [new 007] Towards SFW sampling for diffusion models via external conditioning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于生成模型安全控制任务，旨在解决扩散模型生成不适宜内容（NSFW）的问题。提出一种无需微调的安全采样器，通过外部多模态模型（如CLIP）引导生成轨迹偏离敏感区域，支持用户自定义过滤类别。实验表明该方法在抑制不良内容的同时保持图像质量，验证了外部条件干预的有效性。**

- **链接: [http://arxiv.org/pdf/2505.08817v1](http://arxiv.org/pdf/2505.08817v1)**

> **作者:** Camilo Carvajal Reyes; Joaquín Fontbona; Felipe Tobar
>
> **备注:** Accepcted at IJCNN 2025
>
> **摘要:** Score-based generative models (SBM), also known as diffusion models, are the de facto state of the art for image synthesis. Despite their unparalleled performance, SBMs have recently been in the spotlight for being tricked into creating not-safe-for-work (NSFW) content, such as violent images and non-consensual nudity. Current approaches that prevent unsafe generation are based on the models' own knowledge, and the majority of them require fine-tuning. This article explores the use of external sources for ensuring safe outputs in SBMs. Our safe-for-work (SFW) sampler implements a Conditional Trajectory Correction step that guides the samples away from undesired regions in the ambient space using multimodal models as the source of conditioning. Furthermore, using Contrastive Language Image Pre-training (CLIP), our method admits user-defined NSFW classes, which can vary in different settings. Our experiments on the text-to-image SBM Stable Diffusion validate that the proposed SFW sampler effectively reduces the generation of explicit content while being competitive with other fine-tuning-based approaches, as assessed via independent NSFW detectors. Moreover, we evaluate the impact of the SFW sampler on image quality and show that the proposed correction scheme comes at a minor cost with negligible effect on samples not needing correction. Our study confirms the suitability of the SFW sampler towards aligned SBM models and the potential of using model-agnostic conditioning for the prevention of unwanted images.
>
---
#### [new 008] Beyond General Prompts: Automated Prompt Refinement using Contrastive Class Alignment Scores for Disambiguating Objects in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型（VLM）的提示优化任务，解决人工设计提示导致的检测性能波动问题。提出基于对比类对齐评分（CCAS）的自动提示优化方法：用大语言模型生成候选提示，通过CCAS筛选语义对齐目标类且区分混淆类的优质提示，无需额外训练或标注数据即可提升检测精度，形成可扩展的模型无关解决方案。**

- **链接: [http://arxiv.org/pdf/2505.09139v1](http://arxiv.org/pdf/2505.09139v1)**

> **作者:** Lucas Choi; Ross Greer
>
> **摘要:** Vision-language models (VLMs) offer flexible object detection through natural language prompts but suffer from performance variability depending on prompt phrasing. In this paper, we introduce a method for automated prompt refinement using a novel metric called the Contrastive Class Alignment Score (CCAS), which ranks prompts based on their semantic alignment with a target object class while penalizing similarity to confounding classes. Our method generates diverse prompt candidates via a large language model and filters them through CCAS, computed using prompt embeddings from a sentence transformer. We evaluate our approach on challenging object categories, demonstrating that our automatic selection of high-precision prompts improves object detection accuracy without the need for additional model training or labeled data. This scalable and model-agnostic pipeline offers a principled alternative to manual prompt engineering for VLM-based detection systems.
>
---
#### [new 009] MrTrack: Register Mamba for Needle Tracking with Rapid Reciprocating Motion during Ultrasound-Guided Aspiration Biopsy
- **分类: cs.CV**

- **简介: 该论文研究超声引导穿刺活检中快速往复运动下的针头追踪任务，解决现有跟踪器无法应对运动模糊与成像退化的问题。提出MrTrack框架，利用Mamba结构建立注册机制，通过历史特征提取全局上下文并存储时序线索，结合自监督多样化损失防止特征坍缩。实验证明其精度、鲁棒性和推理效率优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.09450v1](http://arxiv.org/pdf/2505.09450v1)**

> **作者:** Yuelin Zhang; Qingpeng Ding; Long Lei; Yongxuan Feng; Raymond Shing-Yan Tang; Shing Shin Cheng
>
> **备注:** Early Accepted by MICCAI 2025
>
> **摘要:** Ultrasound-guided fine needle aspiration (FNA) biopsy is a common minimally invasive diagnostic procedure. However, an aspiration needle tracker addressing rapid reciprocating motion is still missing. MrTrack, an aspiration needle tracker with a mamba-based register mechanism, is proposed. MrTrack leverages a Mamba-based register extractor to sequentially distill global context from each historical search map, storing these temporal cues in a register bank. The Mamba-based register retriever then retrieves temporal prompts from the register bank to provide external cues when current vision features are temporarily unusable due to rapid reciprocating motion and imaging degradation. A self-supervised register diversify loss is proposed to encourage feature diversity and dimension independence within the learned register, mitigating feature collapse. Comprehensive experiments conducted on both motorized and manual aspiration datasets demonstrate that MrTrack not only outperforms state-of-the-art trackers in accuracy and robustness but also achieves superior inference efficiency.
>
---
#### [new 010] UniCAD: Efficient and Extendable Architecture for Multi-Task Computer-Aided Diagnosis System
- **分类: cs.CV**

- **简介: 论文提出UniCAD，面向多任务计算机辅助诊断（CAD），解决现有系统开发效率低、资源消耗大及缺乏开源平台的问题。通过预训练视觉模型适配医学图像，结合低秩调参（仅0.17%参数）和模块化即插即用架构，实现跨2D/3D任务的高效扩展，并建立开源平台共享轻量模型，实验证明其精度与效率优于传统方法。**

- **链接: [http://arxiv.org/pdf/2505.09178v1](http://arxiv.org/pdf/2505.09178v1)**

> **作者:** Yitao Zhu; Yuan Yin; Zhenrong Shen; Zihao Zhao; Haiyu Song; Sheng Wang; Dinggang Shen; Qian Wang
>
> **备注:** 14 pages
>
> **摘要:** The growing complexity and scale of visual model pre-training have made developing and deploying multi-task computer-aided diagnosis (CAD) systems increasingly challenging and resource-intensive. Furthermore, the medical imaging community lacks an open-source CAD platform to enable the rapid creation of efficient and extendable diagnostic models. To address these issues, we propose UniCAD, a unified architecture that leverages the robust capabilities of pre-trained vision foundation models to seamlessly handle both 2D and 3D medical images while requiring only minimal task-specific parameters. UniCAD introduces two key innovations: (1) Efficiency: A low-rank adaptation strategy is employed to adapt a pre-trained visual model to the medical image domain, achieving performance on par with fully fine-tuned counterparts while introducing only 0.17% trainable parameters. (2) Plug-and-Play: A modular architecture that combines a frozen foundation model with multiple plug-and-play experts, enabling diverse tasks and seamless functionality expansion. Building on this unified CAD architecture, we establish an open-source platform where researchers can share and access lightweight CAD experts, fostering a more equitable and efficient research ecosystem. Comprehensive experiments across 12 diverse medical datasets demonstrate that UniCAD consistently outperforms existing methods in both accuracy and deployment efficiency. The source code and project page are available at https://mii-laboratory.github.io/UniCAD/.
>
---
#### [new 011] BLIP3-o: A Family of Fully Open Unified Multimodal Models-Architecture, Training and Dataset
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态统一任务，旨在同时优化图像理解与生成。针对现有统一框架中生成能力不足的问题，提出用扩散Transformer生成CLIP图像特征（替代传统VAE），设计分阶段预训练策略（先理解后生成），并构建高质量指令数据集BLIP3o-60k，最终开发出开源模型BLIP3-o，在主流基准测试中实现双任务最优性能。**

- **链接: [http://arxiv.org/pdf/2505.09568v1](http://arxiv.org/pdf/2505.09568v1)**

> **作者:** Jiuhai Chen; Zhiyang Xu; Xichen Pan; Yushi Hu; Can Qin; Tom Goldstein; Lifu Huang; Tianyi Zhou; Saining Xie; Silvio Savarese; Le Xue; Caiming Xiong; Ran Xu
>
> **摘要:** Unifying image understanding and generation has gained growing attention in recent research on multimodal models. Although design choices for image understanding have been extensively studied, the optimal model architecture and training recipe for a unified framework with image generation remain underexplored. Motivated by the strong potential of autoregressive and diffusion models for high-quality generation and scalability, we conduct a comprehensive study of their use in unified multimodal settings, with emphasis on image representations, modeling objectives, and training strategies. Grounded in these investigations, we introduce a novel approach that employs a diffusion transformer to generate semantically rich CLIP image features, in contrast to conventional VAE-based representations. This design yields both higher training efficiency and improved generative quality. Furthermore, we demonstrate that a sequential pretraining strategy for unified models-first training on image understanding and subsequently on image generation-offers practical advantages by preserving image understanding capability while developing strong image generation ability. Finally, we carefully curate a high-quality instruction-tuning dataset BLIP3o-60k for image generation by prompting GPT-4o with a diverse set of captions covering various scenes, objects, human gestures, and more. Building on our innovative model design, training recipe, and datasets, we develop BLIP3-o, a suite of state-of-the-art unified multimodal models. BLIP3-o achieves superior performance across most of the popular benchmarks spanning both image understanding and generation tasks. To facilitate future research, we fully open-source our models, including code, model weights, training scripts, and pretraining and instruction tuning datasets.
>
---
#### [new 012] TopoDiT-3D: Topology-Aware Diffusion Transformer with Bottleneck Structure for 3D Point Cloud Generation
- **分类: cs.CV**

- **简介: 该论文研究3D点云生成任务，解决现有扩散Transformer方法忽视全局拓扑信息（如孔洞）导致生成质量受限的问题。提出TopoDiT-3D模型，设计基于感知重采样的瓶颈结构，融合持久同调提取的拓扑特征，并过滤冗余局部特征，提升形状一致性和训练效率。实验验证其在生成质量、多样性和效率上的优势。**

- **链接: [http://arxiv.org/pdf/2505.09140v1](http://arxiv.org/pdf/2505.09140v1)**

> **作者:** Zechao Guan; Feng Yan; Shuai Du; Lin Ma; Qingshan Liu
>
> **摘要:** Recent advancements in Diffusion Transformer (DiT) models have significantly improved 3D point cloud generation. However, existing methods primarily focus on local feature extraction while overlooking global topological information, such as voids, which are crucial for maintaining shape consistency and capturing complex geometries. To address this limitation, we propose TopoDiT-3D, a Topology-Aware Diffusion Transformer with a bottleneck structure for 3D point cloud generation. Specifically, we design the bottleneck structure utilizing Perceiver Resampler, which not only offers a mode to integrate topological information extracted through persistent homology into feature learning, but also adaptively filters out redundant local features to improve training efficiency. Experimental results demonstrate that TopoDiT-3D outperforms state-of-the-art models in visual quality, diversity, and training efficiency. Furthermore, TopoDiT-3D demonstrates the importance of rich topological information for 3D point cloud generation and its synergy with conventional local feature learning. Videos and code are available at https://github.com/Zechao-Guan/TopoDiT-3D.
>
---
#### [new 013] OpenLKA: An Open Dataset of Lane Keeping Assist from Recent Car Models under Real-world Driving Conditions
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出首个开放车道保持辅助（LKA）评估数据集OpenLKA，解决现有系统因数据封闭导致真实性能研究不足的问题。通过整合50+车型的400小时多模态数据（CAN总线、视频、算法输出及场景标注），支持LKA系统性能评测、安全隐患分析及道路设施评估。**

- **链接: [http://arxiv.org/pdf/2505.09092v1](http://arxiv.org/pdf/2505.09092v1)**

> **作者:** Yuhang Wang; Abdulaziz Alhuraish; Shengming Yuan; Hao Zhou
>
> **摘要:** Lane Keeping Assist (LKA) is widely adopted in modern vehicles, yet its real-world performance remains underexplored due to proprietary systems and limited data access. This paper presents OpenLKA, the first open, large-scale dataset for LKA evaluation and improvement. It includes 400 hours of driving data from 50+ production vehicle models, collected through extensive road testing in Tampa, Florida and global contributions from the Comma.ai driving community. The dataset spans a wide range of challenging scenarios, including complex road geometries, degraded lane markings, adverse weather, lighting conditions and surrounding traffic. The dataset is multimodal, comprising: i) full CAN bus streams, decoded using custom reverse-engineered DBC files to extract key LKA events (e.g., system disengagements, lane detection failures); ii) synchronized high-resolution dash-cam video; iii) real-time outputs from Openpilot, providing accurate estimates of road curvature and lane positioning; iv) enhanced scene annotations generated by Vision Language Models, describing lane visibility, pavement quality, weather, lighting, and traffic conditions. By integrating vehicle-internal signals with high-fidelity perception and rich semantic context, OpenLKA provides a comprehensive platform for benchmarking the real-world performance of production LKA systems, identifying safety-critical operational scenarios, and assessing the readiness of current road infrastructure for autonomous driving. The dataset is publicly available at: https://github.com/OpenLKA/OpenLKA.
>
---
#### [new 014] FaceShield: Explainable Face Anti-Spoofing with Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于人脸反欺诈任务，解决现有方法缺乏可解释性及专用多模态模型/数据集的问题。提出FaceShield多模态大模型，结合原始图像与先验知识，采用辅助视觉感知和掩码策略提升泛化能力，构建预训练与微调数据集，实现在分类、推理、定位等任务上超越传统模型的效果。**

- **链接: [http://arxiv.org/pdf/2505.09415v1](http://arxiv.org/pdf/2505.09415v1)**

> **作者:** Hongyang Wang; Yichen Shi; Zhuofu Tao; Yuhao Gao; Liepiao Zhang; Xun Lin; Jun Feng; Xiaochen Yuan; Zitong Yu; Xiaochun Cao
>
> **摘要:** Face anti-spoofing (FAS) is crucial for protecting facial recognition systems from presentation attacks. Previous methods approached this task as a classification problem, lacking interpretability and reasoning behind the predicted results. Recently, multimodal large language models (MLLMs) have shown strong capabilities in perception, reasoning, and decision-making in visual tasks. However, there is currently no universal and comprehensive MLLM and dataset specifically designed for FAS task. To address this gap, we propose FaceShield, a MLLM for FAS, along with the corresponding pre-training and supervised fine-tuning (SFT) datasets, FaceShield-pre10K and FaceShield-sft45K. FaceShield is capable of determining the authenticity of faces, identifying types of spoofing attacks, providing reasoning for its judgments, and detecting attack areas. Specifically, we employ spoof-aware vision perception (SAVP) that incorporates both the original image and auxiliary information based on prior knowledge. We then use an prompt-guided vision token masking (PVTM) strategy to random mask vision tokens, thereby improving the model's generalization ability. We conducted extensive experiments on three benchmark datasets, demonstrating that FaceShield significantly outperforms previous deep learning models and general MLLMs on four FAS tasks, i.e., coarse-grained classification, fine-grained classification, reasoning, and attack localization. Our instruction datasets, protocols, and codes will be released soon.
>
---
#### [new 015] Intelligent Road Anomaly Detection with Real-time Notification System for Enhanced Road Safety
- **分类: cs.CV; cs.SY; eess.IV; eess.SY**

- **简介: 该论文属于智能交通系统的目标检测与分类任务，旨在解决道路坑洞、裂缝等异常引发交通事故的问题。研究开发了基于树莓派、摄像头和深度学习模型的实时检测系统，可识别多种裂缝类型、评估严重程度并统计数量，同步向云端上报数据并预警附近车辆，提升道路安全。**

- **链接: [http://arxiv.org/pdf/2505.08882v1](http://arxiv.org/pdf/2505.08882v1)**

> **作者:** Ali Almakhluk; Uthman Baroudi; Yasser El-Alfy
>
> **摘要:** This study aims to improve transportation safety, especially traffic safety. Road damage anomalies such as potholes and cracks have emerged as a significant and recurring cause for accidents. To tackle this problem and improve road safety, a comprehensive system has been developed to detect potholes, cracks (e.g. alligator, transverse, longitudinal), classify their sizes, and transmit this data to the cloud for appropriate action by authorities. The system also broadcasts warning signals to nearby vehicles warning them if a severe anomaly is detected on the road. Moreover, the system can count road anomalies in real-time. It is emulated through the utilization of Raspberry Pi, a camera module, deep learning model, laptop, and cloud service. Deploying this innovative solution aims to proactively enhance road safety by notifying relevant authorities and drivers about the presence of potholes and cracks to take actions, thereby mitigating potential accidents arising from this prevalent road hazard leading to safer road conditions for the whole community.
>
---
#### [new 016] Don't Forget your Inverse DDIM for Image Editing
- **分类: cs.CV; I.2.10; I.5.0**

- **简介: 该论文属于图像编辑任务，旨在解决现有扩散模型方法效率低、重建质量差的问题。提出SAGE方法，基于DDIM框架引入自注意力引导机制，通过逆向过程生成的注意力图优化未编辑区域重建，减少全局精确重构需求。实验验证其性能优于现有方法，用户偏好和定量评估均领先。**

- **链接: [http://arxiv.org/pdf/2505.09571v1](http://arxiv.org/pdf/2505.09571v1)**

> **作者:** Guillermo Gomez-Trenado; Pablo Mesejo; Oscar Cordón; Stéphane Lathuilière
>
> **备注:** 12 pages, 12 figures, code available at https://guillermogotre.github.io/sage/
>
> **摘要:** The field of text-to-image generation has undergone significant advancements with the introduction of diffusion models. Nevertheless, the challenge of editing real images persists, as most methods are either computationally intensive or produce poor reconstructions. This paper introduces SAGE (Self-Attention Guidance for image Editing) - a novel technique leveraging pre-trained diffusion models for image editing. SAGE builds upon the DDIM algorithm and incorporates a novel guidance mechanism utilizing the self-attention layers of the diffusion U-Net. This mechanism computes a reconstruction objective based on attention maps generated during the inverse DDIM process, enabling efficient reconstruction of unedited regions without the need to precisely reconstruct the entire input image. Thus, SAGE directly addresses the key challenges in image editing. The superiority of SAGE over other methods is demonstrated through quantitative and qualitative evaluations and confirmed by a statistically validated comprehensive user study, in which all 47 surveyed users preferred SAGE over competing methods. Additionally, SAGE ranks as the top-performing method in seven out of 10 quantitative analyses and secures second and third places in the remaining three.
>
---
#### [new 017] Denoising and Alignment: Rethinking Domain Generalization for Multimodal Face Anti-Spoofing
- **分类: cs.CV**

- **简介: 该论文针对多模态人脸反欺诈任务中因模态偏差和域偏移导致的泛化问题，提出MMDA框架。通过CLIP驱动的去噪对齐机制（MD2A模块抑制噪声，RS2策略对齐多域数据），结合U-DSA模块增强表征适应性，提升跨域检测效果，实验验证其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.09484v1](http://arxiv.org/pdf/2505.09484v1)**

> **作者:** Yingjie Ma; Xun Lin; Zitong Yu; Xin Liu; Xiaochen Yuan; Weicheng Xie; Linlin Shen
>
> **摘要:** Face Anti-Spoofing (FAS) is essential for the security of facial recognition systems in diverse scenarios such as payment processing and surveillance. Current multimodal FAS methods often struggle with effective generalization, mainly due to modality-specific biases and domain shifts. To address these challenges, we introduce the \textbf{M}ulti\textbf{m}odal \textbf{D}enoising and \textbf{A}lignment (\textbf{MMDA}) framework. By leveraging the zero-shot generalization capability of CLIP, the MMDA framework effectively suppresses noise in multimodal data through denoising and alignment mechanisms, thereby significantly enhancing the generalization performance of cross-modal alignment. The \textbf{M}odality-\textbf{D}omain Joint \textbf{D}ifferential \textbf{A}ttention (\textbf{MD2A}) module in MMDA concurrently mitigates the impacts of domain and modality noise by refining the attention mechanism based on extracted common noise features. Furthermore, the \textbf{R}epresentation \textbf{S}pace \textbf{S}oft (\textbf{RS2}) Alignment strategy utilizes the pre-trained CLIP model to align multi-domain multimodal data into a generalized representation space in a flexible manner, preserving intricate representations and enhancing the model's adaptability to various unseen conditions. We also design a \textbf{U}-shaped \textbf{D}ual \textbf{S}pace \textbf{A}daptation (\textbf{U-DSA}) module to enhance the adaptability of representations while maintaining generalization performance. These improvements not only enhance the framework's generalization capabilities but also boost its ability to represent complex representations. Our experimental results on four benchmark datasets under different evaluation protocols demonstrate that the MMDA framework outperforms existing state-of-the-art methods in terms of cross-domain generalization and multimodal detection accuracy. The code will be released soon.
>
---
#### [new 018] Towards Adaptive Meta-Gradient Adversarial Examples for Visual Tracking
- **分类: cs.CV**

- **简介: 该论文研究视觉跟踪模型的对抗攻击问题，提出自适应元梯度攻击方法（AMGA），通过集成多模型、元学习及动量机制增强对抗样本的迁移性和攻击性，缩小白盒与黑盒攻击差距，提升跟踪器安全性测试效果。**

- **链接: [http://arxiv.org/pdf/2505.08999v1](http://arxiv.org/pdf/2505.08999v1)**

> **作者:** Wei-Long Tian; Peng Gao; Xiao Liu; Long Xu; Hamido Fujita; Hanan Aljuai; Mao-Li Wang
>
> **摘要:** In recent years, visual tracking methods based on convolutional neural networks and Transformers have achieved remarkable performance and have been successfully applied in fields such as autonomous driving. However, the numerous security issues exposed by deep learning models have gradually affected the reliable application of visual tracking methods in real-world scenarios. Therefore, how to reveal the security vulnerabilities of existing visual trackers through effective adversarial attacks has become a critical problem that needs to be addressed. To this end, we propose an adaptive meta-gradient adversarial attack (AMGA) method for visual tracking. This method integrates multi-model ensembles and meta-learning strategies, combining momentum mechanisms and Gaussian smoothing, which can significantly enhance the transferability and attack effectiveness of adversarial examples. AMGA randomly selects models from a large model repository, constructs diverse tracking scenarios, and iteratively performs both white- and black-box adversarial attacks in each scenario, optimizing the gradient directions of each model. This paradigm minimizes the gap between white- and black-box adversarial attacks, thus achieving excellent attack performance in black-box scenarios. Extensive experimental results on large-scale datasets such as OTB2015, LaSOT, and GOT-10k demonstrate that AMGA significantly improves the attack performance, transferability, and deception of adversarial examples. Codes and data are available at https://github.com/pgao-lab/AMGA.
>
---
#### [new 019] MAKE: Multi-Aspect Knowledge-Enhanced Vision-Language Pretraining for Zero-shot Dermatological Assessment
- **分类: cs.CV**

- **简介: 该论文提出MAKE框架，用于零样本皮肤病学评估任务。针对现有视觉语言预训练（VLP）在皮肤科因文本限制和缺乏结构化知识导致效果受限的问题，设计了多角度知识增强策略：分解临床文本为知识子项、细粒度对齐图像特征、基于临床重要性的权重分配。通过40万图像-文本对预训练，在疾病分类等任务中超越现有模型。**

- **链接: [http://arxiv.org/pdf/2505.09372v1](http://arxiv.org/pdf/2505.09372v1)**

> **作者:** Siyuan Yan; Xieji Li; Ming Hu; Yiwen Jiang; Zhen Yu; Zongyuan Ge
>
> **备注:** MICCAI2025 early acceptance; First two authors contribute equally
>
> **摘要:** Dermatological diagnosis represents a complex multimodal challenge that requires integrating visual features with specialized clinical knowledge. While vision-language pretraining (VLP) has advanced medical AI, its effectiveness in dermatology is limited by text length constraints and the lack of structured texts. In this paper, we introduce MAKE, a Multi-Aspect Knowledge-Enhanced vision-language pretraining framework for zero-shot dermatological tasks. Recognizing that comprehensive dermatological descriptions require multiple knowledge aspects that exceed standard text constraints, our framework introduces: (1) a multi-aspect contrastive learning strategy that decomposes clinical narratives into knowledge-enhanced sub-texts through large language models, (2) a fine-grained alignment mechanism that connects subcaptions with diagnostically relevant image features, and (3) a diagnosis-guided weighting scheme that adaptively prioritizes different sub-captions based on clinical significance prior. Through pretraining on 403,563 dermatological image-text pairs collected from education resources, MAKE significantly outperforms state-of-the-art VLP models on eight datasets across zero-shot skin disease classification, concept annotation, and cross-modal retrieval tasks. Our code will be made publicly available at https: //github.com/SiyuanYan1/MAKE.
>
---
#### [new 020] BioVFM-21M: Benchmarking and Scaling Self-Supervised Vision Foundation Models for Biomedical Image Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究生物医学图像自监督视觉基础模型的规模化训练，解决医学领域缺乏扩展规律指导的问题。通过构建BioVFM-21M大规模数据集（2100万图像），分析模型规模、训练方法等因素的影响，提出BioVFM模型，在12个医学基准中超越现有方法，证明规模化需结合任务特性与数据多样性。**

- **链接: [http://arxiv.org/pdf/2505.09329v1](http://arxiv.org/pdf/2505.09329v1)**

> **作者:** Jiarun Liu; Hong-Yu Zhou; Weijian Huang; Hao Yang; Dongning Song; Tao Tan; Yong Liang; Shanshan Wang
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Scaling up model and data size have demonstrated impressive performance improvement over a wide range of tasks. Despite extensive studies on scaling behaviors for general-purpose tasks, medical images exhibit substantial differences from natural data. It remains unclear the key factors in developing medical vision foundation models at scale due to the absence of an extensive understanding of scaling behavior in the medical domain. In this paper, we explored the scaling behavior across model sizes, training algorithms, data sizes, and imaging modalities in developing scalable medical vision foundation models by self-supervised learning. To support scalable pretraining, we introduce BioVFM-21M, a large-scale biomedical image dataset encompassing a wide range of biomedical image modalities and anatomies. We observed that scaling up does provide benefits but varies across tasks. Additional analysis reveals several factors correlated with scaling benefits. Finally, we propose BioVFM, a large-scale medical vision foundation model pretrained on 21 million biomedical images, which outperforms the previous state-of-the-art foundation models across 12 medical benchmarks. Our results highlight that while scaling up is beneficial for pursuing better performance, task characteristics, data diversity, pretraining methods, and computational efficiency remain critical considerations for developing scalable medical foundation models.
>
---
#### [new 021] Generative AI for Autonomous Driving: Frontiers and Opportunities
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文为综述研究，探讨生成式AI在自动驾驶中的应用。聚焦实现Level 5全自动驾驶，梳理了VAE、扩散模型、LLM等技术在数据生成、决策规划等场景的进展，分析合成数据、数字孪生等应用方向，并指出泛化性、安全性等挑战及未来研究路径。**

- **链接: [http://arxiv.org/pdf/2505.08854v1](http://arxiv.org/pdf/2505.08854v1)**

> **作者:** Yuping Wang; Shuo Xing; Cui Can; Renjie Li; Hongyuan Hua; Kexin Tian; Zhaobin Mo; Xiangbo Gao; Keshu Wu; Sulong Zhou; Hengxu You; Juntong Peng; Junge Zhang; Zehao Wang; Rui Song; Mingxuan Yan; Walter Zimmer; Xingcheng Zhou; Peiran Li; Zhaohan Lu; Chia-Ju Chen; Yue Huang; Ryan A. Rossi; Lichao Sun; Hongkai Yu; Zhiwen Fan; Frank Hao Yang; Yuhao Kang; Ross Greer; Chenxi Liu; Eun Hak Lee; Xuan Di; Xinyue Ye; Liu Ren; Alois Knoll; Xiaopeng Li; Shuiwang Ji; Masayoshi Tomizuka; Marco Pavone; Tianbao Yang; Jing Du; Ming-Hsuan Yang; Hua Wei; Ziran Wang; Yang Zhou; Jiachen Li; Zhengzhong Tu
>
> **摘要:** Generative Artificial Intelligence (GenAI) constitutes a transformative technological wave that reconfigures industries through its unparalleled capabilities for content creation, reasoning, planning, and multimodal understanding. This revolutionary force offers the most promising path yet toward solving one of engineering's grandest challenges: achieving reliable, fully autonomous driving, particularly the pursuit of Level 5 autonomy. This survey delivers a comprehensive and critical synthesis of the emerging role of GenAI across the autonomous driving stack. We begin by distilling the principles and trade-offs of modern generative modeling, encompassing VAEs, GANs, Diffusion Models, and Large Language Models (LLMs). We then map their frontier applications in image, LiDAR, trajectory, occupancy, video generation as well as LLM-guided reasoning and decision making. We categorize practical applications, such as synthetic data workflows, end-to-end driving strategies, high-fidelity digital twin systems, smart transportation networks, and cross-domain transfer to embodied AI. We identify key obstacles and possibilities such as comprehensive generalization across rare cases, evaluation and safety checks, budget-limited implementation, regulatory compliance, ethical concerns, and environmental effects, while proposing research plans across theoretical assurances, trust metrics, transport integration, and socio-technical influence. By unifying these threads, the survey provides a forward-looking reference for researchers, engineers, and policymakers navigating the convergence of generative AI and advanced autonomous mobility. An actively maintained repository of cited works is available at https://github.com/taco-group/GenAI4AD.
>
---
#### [new 022] PDE: Gene Effect Inspired Parameter Dynamic Evolution for Low-light Image Enhancement
- **分类: cs.CV**

- **简介: 该论文针对低光图像增强任务，提出参数动态演化(PDE)方法以解决现有模型因"基因效应"导致的性能受限问题。通过模拟生物进化中的基因突变和重组机制，动态调整网络参数，克服静态参数对图像适应性不足的缺陷，提升模型能力利用率。**

- **链接: [http://arxiv.org/pdf/2505.09196v1](http://arxiv.org/pdf/2505.09196v1)**

> **作者:** Tong Li; Lizhi Wang; Hansen Feng; Lin Zhu; Hua Huang
>
> **备注:** 11 pages, 9 tables, 9 figures
>
> **摘要:** Low-light image enhancement (LLIE) is a fundamental task in computational photography, aiming to improve illumination, reduce noise, and enhance image quality. While recent advancements focus on designing increasingly complex neural network models, we observe a peculiar phenomenon: resetting certain parameters to random values unexpectedly improves enhancement performance for some images. Drawing inspiration from biological genes, we term this phenomenon the gene effect. The gene effect limits enhancement performance, as even random parameters can sometimes outperform learned ones, preventing models from fully utilizing their capacity. In this paper, we investigate the reason and propose a solution. Based on our observations, we attribute the gene effect to static parameters, analogous to how fixed genetic configurations become maladaptive when environments change. Inspired by biological evolution, where adaptation to new environments relies on gene mutation and recombination, we propose parameter dynamic evolution (PDE) to adapt to different images and mitigate the gene effect. PDE employs a parameter orthogonal generation technique and the corresponding generated parameters to simulate gene recombination and gene mutation, separately. Experiments validate the effectiveness of our techniques. The code will be released to the public.
>
---
#### [new 023] Efficient LiDAR Reflectance Compression via Scanning Serialization
- **分类: cs.CV; eess.IV**

- **简介: 该论文研究LiDAR点云反射率的高效压缩任务，解决现有方法未充分利用反射率特性的问题。提出SerLiC框架：通过扫描顺序序列化将3D点云转为1D序列，结合传感器索引、距离和反射率构建上下文表征，并采用改进的Mamba模型进行序列建模。方法在压缩率（提升22%）和效率（>10fps）上优于现有技术。**

- **链接: [http://arxiv.org/pdf/2505.09433v1](http://arxiv.org/pdf/2505.09433v1)**

> **作者:** Jiahao Zhu; Kang You; Dandan Ding; Zhan Ma
>
> **摘要:** Reflectance attributes in LiDAR point clouds provide essential information for downstream tasks but remain underexplored in neural compression methods. To address this, we introduce SerLiC, a serialization-based neural compression framework to fully exploit the intrinsic characteristics of LiDAR reflectance. SerLiC first transforms 3D LiDAR point clouds into 1D sequences via scan-order serialization, offering a device-centric perspective for reflectance analysis. Each point is then tokenized into a contextual representation comprising its sensor scanning index, radial distance, and prior reflectance, for effective dependencies exploration. For efficient sequential modeling, Mamba is incorporated with a dual parallelization scheme, enabling simultaneous autoregressive dependency capture and fast processing. Extensive experiments demonstrate that SerLiC attains over 2x volume reduction against the original reflectance data, outperforming the state-of-the-art method by up to 22% reduction of compressed bits while using only 2% of its parameters. Moreover, a lightweight version of SerLiC achieves > 10 fps (frames per second) with just 111K parameters, which is attractive for real-world applications.
>
---
#### [new 024] Graph-based Online Monitoring of Train Driver States via Facial and Skeletal Features
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于行为监测任务，旨在解决火车司机疲劳引发的安全隐患。研究者提出基于有向图神经网络的在线监测系统，通过融合面部与骨骼特征实现三态（警觉/非警觉/病理）分类，构建包含病理状态的新数据集。实验表明多特征融合模型准确率达80.88%，显著优于单模态方法。**

- **链接: [http://arxiv.org/pdf/2505.08800v1](http://arxiv.org/pdf/2505.08800v1)**

> **作者:** Olivia Nocentini; Marta Lagomarsino; Gokhan Solak; Younggeol Cho; Qiyi Tong; Marta Lorenzini; Arash Ajoudani
>
> **摘要:** Driver fatigue poses a significant challenge to railway safety, with traditional systems like the dead-man switch offering limited and basic alertness checks. This study presents an online behavior-based monitoring system utilizing a customised Directed-Graph Neural Network (DGNN) to classify train driver's states into three categories: alert, not alert, and pathological. To optimize input representations for the model, an ablation study was performed, comparing three feature configurations: skeletal-only, facial-only, and a combination of both. Experimental results show that combining facial and skeletal features yields the highest accuracy (80.88%) in the three-class model, outperforming models using only facial or skeletal features. Furthermore, this combination achieves over 99% accuracy in the binary alertness classification. Additionally, we introduced a novel dataset that, for the first time, incorporates simulated pathological conditions into train driver monitoring, broadening the scope for assessing risks related to fatigue and health. This work represents a step forward in enhancing railway safety through advanced online monitoring using vision-based technologies.
>
---
#### [new 025] Sparse Point Cloud Patches Rendering via Splitting 2D Gaussians
- **分类: cs.CV**

- **简介: 该论文属于点云渲染任务，旨在解决现有方法依赖类别先验、密集点云或后处理的问题。提出基于2D高斯预测的渲染方法，通过双模块架构初始化并利用点云法线、颜色等信息归一化高斯，结合分割解码器复制细化高斯分布，适配稀疏点云。该方法无需后处理即可跨类别泛化，在多个数据集上实现SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.09413v1](http://arxiv.org/pdf/2505.09413v1)**

> **作者:** Ma Changfeng; Bi Ran; Guo Jie; Wang Chongjun; Guo Yanwen
>
> **备注:** CVPR 2025 Accepted
>
> **摘要:** Current learning-based methods predict NeRF or 3D Gaussians from point clouds to achieve photo-realistic rendering but still depend on categorical priors, dense point clouds, or additional refinements. Hence, we introduce a novel point cloud rendering method by predicting 2D Gaussians from point clouds. Our method incorporates two identical modules with an entire-patch architecture enabling the network to be generalized to multiple datasets. The module normalizes and initializes the Gaussians utilizing the point cloud information including normals, colors and distances. Then, splitting decoders are employed to refine the initial Gaussians by duplicating them and predicting more accurate results, making our methodology effectively accommodate sparse point clouds as well. Once trained, our approach exhibits direct generalization to point clouds across different categories. The predicted Gaussians are employed directly for rendering without additional refinement on the rendered images, retaining the benefits of 2D Gaussians. We conduct extensive experiments on various datasets, and the results demonstrate the superiority and generalization of our method, which achieves SOTA performance. The code is available at https://github.com/murcherful/GauPCRender}{https://github.com/murcherful/GauPCRender.
>
---
#### [new 026] FedSaaS: Class-Consistency Federated Semantic Segmentation via Global Prototype Supervision and Local Adversarial Harmonization
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究联邦语义分割，解决数据异构导致的类别表示模糊问题。提出FedSaaS框架，通过全局原型监督统一客户端类别特征，利用本地对抗协调融合全局与局部分支，并结合对比损失增强语义一致性，显著提升分割精度及类别一致性。**

- **链接: [http://arxiv.org/pdf/2505.09385v1](http://arxiv.org/pdf/2505.09385v1)**

> **作者:** Xiaoyang Yu; Xiaoming Wu; Xin Wang; Dongrun Li; Ming Yang; Peng Cheng
>
> **摘要:** Federated semantic segmentation enables pixel-level classification in images through collaborative learning while maintaining data privacy. However, existing research commonly overlooks the fine-grained class relationships within the semantic space when addressing heterogeneous problems, particularly domain shift. This oversight results in ambiguities between class representation. To overcome this challenge, we propose a novel federated segmentation framework that strikes class consistency, termed FedSaaS. Specifically, we introduce class exemplars as a criterion for both local- and global-level class representations. On the server side, the uploaded class exemplars are leveraged to model class prototypes, which supervise global branch of clients, ensuring alignment with global-level representation. On the client side, we incorporate an adversarial mechanism to harmonize contributions of global and local branches, leading to consistent output. Moreover, multilevel contrastive losses are employed on both sides to enforce consistency between two-level representations in the same semantic space. Extensive experiments on several driving scene segmentation datasets demonstrate that our framework outperforms state-of-the-art methods, significantly improving average segmentation accuracy and effectively addressing the class-consistency representation problem.
>
---
#### [new 027] AMSnet 2.0: A Large AMS Database with AI Segmentation for Net Detection
- **分类: cs.CV**

- **简介: 该论文属于电路网表检测任务，旨在解决多模态大模型因缺乏高质量数据难以识别电路原理图的问题。提出基于AI分割的鲁棒检测方法，恢复元件位置并实现数字重建。扩展原AMSnet数据集至AMSnet 2.0，新增2,686个电路，包含多格式网表、数字原理图及位置信息，提升训练数据质量。**

- **链接: [http://arxiv.org/pdf/2505.09155v1](http://arxiv.org/pdf/2505.09155v1)**

> **作者:** Yichen Shi; Zhuofu Tao; Yuhao Gao; Li Huang; Hongyang Wang; Zhiping Yu; Ting-Jung Lin; Lei He
>
> **备注:** accepted by LAD25
>
> **摘要:** Current multimodal large language models (MLLMs) struggle to understand circuit schematics due to their limited recognition capabilities. This could be attributed to the lack of high-quality schematic-netlist training data. Existing work such as AMSnet applies schematic parsing to generate netlists. However, these methods rely on hard-coded heuristics and are difficult to apply to complex or noisy schematics in this paper. We therefore propose a novel net detection mechanism based on segmentation with high robustness. The proposed method also recovers positional information, allowing digital reconstruction of schematics. We then expand AMSnet dataset with schematic images from various sources and create AMSnet 2.0. AMSnet 2.0 contains 2,686 circuits with schematic images, Spectre-formatted netlists, OpenAccess digital schematics, and positional information for circuit components and nets, whereas AMSnet only includes 792 circuits with SPICE netlists but no digital schematics.
>
---
#### [new 028] A Surrogate Model for the Forward Design of Multi-layered Metasurface-based Radar Absorbing Structures
- **分类: cs.CV**

- **简介: 该论文属于电磁优化设计任务，旨在解决传统全波仿真设计多层超表面雷达吸波结构时计算量大、耗时的问题。提出基于卷积神经网络（CNN）的替代模型，通过Huber损失函数预测结构反射特性，在保证精度的同时大幅缩短计算时间，实验验证了模型高效性。**

- **链接: [http://arxiv.org/pdf/2505.09251v1](http://arxiv.org/pdf/2505.09251v1)**

> **作者:** Vineetha Joy; Aditya Anand; Nidhi; Anshuman Kumar; Amit Sethi; Hema Singh
>
> **摘要:** Metasurface-based radar absorbing structures (RAS) are highly preferred for applications like stealth technology, electromagnetic (EM) shielding, etc. due to their capability to achieve frequency selective absorption characteristics with minimal thickness and reduced weight penalty. However, the conventional approach for the EM design and optimization of these structures relies on forward simulations, using full wave simulation tools, to predict the electromagnetic (EM) response of candidate meta atoms. This process is computationally intensive, extremely time consuming and requires exploration of large design spaces. To overcome this challenge, we propose a surrogate model that significantly accelerates the prediction of EM responses of multi-layered metasurface-based RAS. A convolutional neural network (CNN) based architecture with Huber loss function has been employed to estimate the reflection characteristics of the RAS model. The proposed model achieved a cosine similarity of 99.9% and a mean square error of 0.001 within 1000 epochs of training. The efficiency of the model has been established via full wave simulations as well as experiment where it demonstrated significant reduction in computational time while maintaining high predictive accuracy.
>
---
#### [new 029] SparseMeXT Unlocking the Potential of Sparse Representations for HD Map Construction
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对在线高精地图构建任务，解决稀疏表示方法性能不足的问题。提出专用网络架构、稀疏-密集分割辅助任务和物理先验去噪模块，在nuScenes数据集实现68.9% mAP，超越密集方法并兼顾效率，重塑领域性能-效率权衡基准。**

- **链接: [http://arxiv.org/pdf/2505.08808v1](http://arxiv.org/pdf/2505.08808v1)**

> **作者:** Anqing Jiang; Jinhao Chai; Yu Gao; Yiru Wang; Yuwen Heng; Zhigang Sun; Hao Sun; Zezhong Zhao; Li Sun; Jian Zhou; Lijuan Zhu; Shugong Xu; Hao Zhao
>
> **摘要:** Recent advancements in high-definition \emph{HD} map construction have demonstrated the effectiveness of dense representations, which heavily rely on computationally intensive bird's-eye view \emph{BEV} features. While sparse representations offer a more efficient alternative by avoiding dense BEV processing, existing methods often lag behind due to the lack of tailored designs. These limitations have hindered the competitiveness of sparse representations in online HD map construction. In this work, we systematically revisit and enhance sparse representation techniques, identifying key architectural and algorithmic improvements that bridge the gap with--and ultimately surpass--dense approaches. We introduce a dedicated network architecture optimized for sparse map feature extraction, a sparse-dense segmentation auxiliary task to better leverage geometric and semantic cues, and a denoising module guided by physical priors to refine predictions. Through these enhancements, our method achieves state-of-the-art performance on the nuScenes dataset, significantly advancing HD map construction and centerline detection. Specifically, SparseMeXt-Tiny reaches a mean average precision \emph{mAP} of 55.5% at 32 frames per second \emph{fps}, while SparseMeXt-Base attains 65.2% mAP. Scaling the backbone and decoder further, SparseMeXt-Large achieves an mAP of 68.9% at over 20 fps, establishing a new benchmark for sparse representations in HD map construction. These results underscore the untapped potential of sparse methods, challenging the conventional reliance on dense representations and redefining efficiency-performance trade-offs in the field.
>
---
#### [new 030] Beyond Pixels: Leveraging the Language of Soccer to Improve Spatio-Temporal Action Detection in Broadcast Videos
- **分类: cs.CV**

- **简介: 该论文针对足球视频时空动作检测（STAD）任务中高召回率场景下因缺乏上下文理解导致的低精度问题，提出结合足球战术逻辑的改进方法。通过Transformer编码器-解码器模型，将原始球员动作序列与比赛状态信息融合，利用团队动态和长期时序建模实现动作序列去噪，提升低置信度场景下的检测可靠性。**

- **链接: [http://arxiv.org/pdf/2505.09455v1](http://arxiv.org/pdf/2505.09455v1)**

> **作者:** Jeremie Ochin; Raphael Chekroun; Bogdan Stanciulescu; Sotiris Manitsaris
>
> **备注:** 12 pages, submitted to Advanced Concepts for Intelligent Vision Systems 2025
>
> **摘要:** State-of-the-art spatio-temporal action detection (STAD) methods show promising results for extracting soccer events from broadcast videos. However, when operated in the high-recall, low-precision regime required for exhaustive event coverage in soccer analytics, their lack of contextual understanding becomes apparent: many false positives could be resolved by considering a broader sequence of actions and game-state information. In this work, we address this limitation by reasoning at the game level and improving STAD through the addition of a denoising sequence transduction task. Sequences of noisy, context-free player-centric predictions are processed alongside clean game state information using a Transformer-based encoder-decoder model. By modeling extended temporal context and reasoning jointly over team-level dynamics, our method leverages the "language of soccer" - its tactical regularities and inter-player dependencies - to generate "denoised" sequences of actions. This approach improves both precision and recall in low-confidence regimes, enabling more reliable event extraction from broadcast video and complementing existing pixel-based methods.
>
---
#### [new 031] DRRNet: Macro-Micro Feature Fusion and Dual Reverse Refinement for Camouflaged Object Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于伪装目标检测任务，旨在解决因目标与背景高度相似导致的边缘丢失和背景干扰问题。提出DRRNet四阶段模型，通过全局-局部特征融合模块整合多尺度语义信息，并设计逆向细化模块结合空间边缘和频域降噪进行双重优化，提升边界连续性和抗干扰能力，实验性能超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.09168v1](http://arxiv.org/pdf/2505.09168v1)**

> **作者:** Jianlin Sun; Xiaolin Fang; Juwei Guan; Dongdong Gui; Teqi Wang; Tongxin Zhu
>
> **摘要:** The core challenge in Camouflage Object Detection (COD) lies in the indistinguishable similarity between targets and backgrounds in terms of color, texture, and shape. This causes existing methods to either lose edge details (such as hair-like fine structures) due to over-reliance on global semantic information or be disturbed by similar backgrounds (such as vegetation patterns) when relying solely on local features. We propose DRRNet, a four-stage architecture characterized by a "context-detail-fusion-refinement" pipeline to address these issues. Specifically, we introduce an Omni-Context Feature Extraction Module to capture global camouflage patterns and a Local Detail Extraction Module to supplement microstructural information for the full-scene context module. We then design a module for forming dual representations of scene understanding and structural awareness, which fuses panoramic features and local features across various scales. In the decoder, we also introduce a reverse refinement module that leverages spatial edge priors and frequency-domain noise suppression to perform a two-stage inverse refinement of the output. By applying two successive rounds of inverse refinement, the model effectively suppresses background interference and enhances the continuity of object boundaries. Experimental results demonstrate that DRRNet significantly outperforms state-of-the-art methods on benchmark datasets. Our code is available at https://github.com/jerrySunning/DRRNet.
>
---
#### [new 032] Differentiable Channel Selection in Self-Attention For Person Re-Identification
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对行人重识别任务，提出可微分通道选择注意力模块（DCS-Attention），解决传统自注意力机制中信息冗余问题。通过信息瓶颈原理设计变分优化目标，动态筛选关键特征通道，兼容固定/可搜索主干网络，提升特征判别力，在多个基准测试中达到SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.08961v1](http://arxiv.org/pdf/2505.08961v1)**

> **作者:** Yancheng Wang; Nebojsa Jojic; Yingzhen Yang
>
> **摘要:** In this paper, we propose a novel attention module termed the Differentiable Channel Selection Attention module, or the DCS-Attention module. In contrast with conventional self-attention, the DCS-Attention module features selection of informative channels in the computation of the attention weights. The selection of the feature channels is performed in a differentiable manner, enabling seamless integration with DNN training. Our DCS-Attention is compatible with either fixed neural network backbones or learnable backbones with Differentiable Neural Architecture Search (DNAS), leading to DCS with Fixed Backbone (DCS-FB) and DCS-DNAS, respectively. Importantly, our DCS-Attention is motivated by the principle of Information Bottleneck (IB), and a novel variational upper bound for the IB loss, which can be optimized by SGD, is derived and incorporated into the training loss of the networks with the DCS-Attention modules. In this manner, a neural network with DCS-Attention modules is capable of selecting the most informative channels for feature extraction so that it enjoys state-of-the-art performance for the Re-ID task. Extensive experiments on multiple person Re-ID benchmarks using both DCS-FB and DCS-DNAS show that DCS-Attention significantly enhances the prediction accuracy of DNNs for person Re-ID, which demonstrates the effectiveness of DCS-Attention in learning discriminative features critical to identifying person identities. The code of our work is available at https://github.com/Statistical-Deep-Learning/DCS-Attention.
>
---
#### [new 033] Recent Advances in Medical Imaging Segmentation: A Survey
- **分类: cs.CV**

- **简介: 该论文是医学影像分割领域的综述，属于调研任务。针对数据获取难、标注复杂、模型泛化及跨域适应性差等问题，梳理了生成式AI、小样本学习等前沿方法，总结了理论框架、技术进展与应用，并探讨了模型实用化瓶颈及未来研究方向。**

- **链接: [http://arxiv.org/pdf/2505.09274v1](http://arxiv.org/pdf/2505.09274v1)**

> **作者:** Fares Bougourzi; Abdenour Hadid
>
> **摘要:** Medical imaging is a cornerstone of modern healthcare, driving advancements in diagnosis, treatment planning, and patient care. Among its various tasks, segmentation remains one of the most challenging problem due to factors such as data accessibility, annotation complexity, structural variability, variation in medical imaging modalities, and privacy constraints. Despite recent progress, achieving robust generalization and domain adaptation remains a significant hurdle, particularly given the resource-intensive nature of some proposed models and their reliance on domain expertise. This survey explores cutting-edge advancements in medical image segmentation, focusing on methodologies such as Generative AI, Few-Shot Learning, Foundation Models, and Universal Models. These approaches offer promising solutions to longstanding challenges. We provide a comprehensive overview of the theoretical foundations, state-of-the-art techniques, and recent applications of these methods. Finally, we discuss inherent limitations, unresolved issues, and future research directions aimed at enhancing the practicality and accessibility of segmentation models in medical imaging. We are maintaining a \href{https://github.com/faresbougourzi/Awesome-DL-for-Medical-Imaging-Segmentation}{GitHub Repository} to continue tracking and updating innovations in this field.
>
---
#### [new 034] Zero-shot Quantization: A Comprehensive Survey
- **分类: cs.CV**

- **简介: 该论文为综述，聚焦零样本量化（ZSQ）任务，解决传统量化依赖训练数据的局限性。通过定义问题、分类数据生成策略方法（分析原理与挑战），总结进展并展望未来方向，推动无数据场景下的模型轻量化研究。**

- **链接: [http://arxiv.org/pdf/2505.09188v1](http://arxiv.org/pdf/2505.09188v1)**

> **作者:** Minjun Kim; Jaehyeon Choi; Jongkeun Lee; Wonjin Cho; U Kang
>
> **备注:** IJCAI 2025 Survey Track
>
> **摘要:** Network quantization has proven to be a powerful approach to reduce the memory and computational demands of deep learning models for deployment on resource-constrained devices. However, traditional quantization methods often rely on access to training data, which is impractical in many real-world scenarios due to privacy, security, or regulatory constraints. Zero-shot Quantization (ZSQ) emerges as a promising solution, achieving quantization without requiring any real data. In this paper, we provide a comprehensive overview of ZSQ methods and their recent advancements. First, we provide a formal definition of the ZSQ problem and highlight the key challenges. Then, we categorize the existing ZSQ methods into classes based on data generation strategies, and analyze their motivations, core ideas, and key takeaways. Lastly, we suggest future research directions to address the remaining limitations and advance the field of ZSQ. To the best of our knowledge, this paper is the first in-depth survey on ZSQ.
>
---
#### [new 035] Neural Video Compression using 2D Gaussian Splatting
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **简介: 该论文属于神经视频压缩任务，旨在解决传统深度学习方法计算量大、难以实时应用的问题。通过引入2D高斯泼溅技术，结合内容感知初始化和帧间冗余消除机制，将编码速度提升88%，首次实现基于高斯泼溅的实时神经视频编解码方案。**

- **链接: [http://arxiv.org/pdf/2505.09324v1](http://arxiv.org/pdf/2505.09324v1)**

> **作者:** Lakshya Gupta; Imran N. Junejo
>
> **备注:** 9 pages, 8 figures
>
> **摘要:** The computer vision and image processing research community has been involved in standardizing video data communications for the past many decades, leading to standards such as AVC, HEVC, VVC, AV1, AV2, etc. However, recent groundbreaking works have focused on employing deep learning-based techniques to replace the traditional video codec pipeline to a greater affect. Neural video codecs (NVC) create an end-to-end ML-based solution that does not rely on any handcrafted features (motion or edge-based) and have the ability to learn content-aware compression strategies, offering better adaptability and higher compression efficiency than traditional methods. This holds a great potential not only for hardware design, but also for various video streaming platforms and applications, especially video conferencing applications such as MS-Teams or Zoom that have found extensive usage in classrooms and workplaces. However, their high computational demands currently limit their use in real-time applications like video conferencing. To address this, we propose a region-of-interest (ROI) based neural video compression model that leverages 2D Gaussian Splatting. Unlike traditional codecs, 2D Gaussian Splatting is capable of real-time decoding and can be optimized using fewer data points, requiring only thousands of Gaussians for decent quality outputs as opposed to millions in 3D scenes. In this work, we designed a video pipeline that speeds up the encoding time of the previous Gaussian splatting-based image codec by 88% by using a content-aware initialization strategy paired with a novel Gaussian inter-frame redundancy-reduction mechanism, enabling Gaussian splatting to be used for a video-codec solution, the first of its kind solution in this neural video codec space.
>
---
#### [new 036] Text-driven Motion Generation: Overview, Challenges and Directions
- **分类: cs.CV**

- **简介: 该论文综述文本驱动动作生成任务，旨在通过自然语言直接创建人体动作，解决传统方法依赖预定义动作输入的问题。工作包括回顾传统动作合成模型，分类现代方法（VAE、扩散、混合架构及动作表示策略），分析数据集与评估指标，并指出领域挑战与未来研究方向。**

- **链接: [http://arxiv.org/pdf/2505.09379v1](http://arxiv.org/pdf/2505.09379v1)**

> **作者:** Ali Rida Sahili; Najett Neji; Hedi Tabia
>
> **备注:** 17 pages, 5 tables
>
> **摘要:** Text-driven motion generation offers a powerful and intuitive way to create human movements directly from natural language. By removing the need for predefined motion inputs, it provides a flexible and accessible approach to controlling animated characters. This makes it especially useful in areas like virtual reality, gaming, human-computer interaction, and robotics. In this review, we first revisit the traditional perspective on motion synthesis, where models focused on predicting future poses from observed initial sequences, often conditioned on action labels. We then provide a comprehensive and structured survey of modern text-to-motion generation approaches, categorizing them from two complementary perspectives: (i) architectural, dividing methods into VAE-based, diffusion-based, and hybrid models; and (ii) motion representation, distinguishing between discrete and continuous motion generation strategies. In addition, we explore the most widely used datasets, evaluation methods, and recent benchmarks that have shaped progress in this area. With this survey, we aim to capture where the field currently stands, bring attention to its key challenges and limitations, and highlight promising directions for future exploration. We hope this work offers a valuable starting point for researchers and practitioners working to push the boundaries of language-driven human motion synthesis.
>
---
#### [new 037] Towards Understanding Deep Learning Model in Image Recognition via Coverage Test
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于深度学习模型安全测试任务，旨在分析不同覆盖指标与模型深度、结构的关联。通过实验对比LeNet、VGG、ResNet等不同架构和深度的模型，研究四种覆盖指标（功能/边界/层次/结构）的规律，并探索覆盖度与数据集大小的关系，为DNN安全测试提供实证依据。**

- **链接: [http://arxiv.org/pdf/2505.08814v1](http://arxiv.org/pdf/2505.08814v1)**

> **作者:** Wenkai Li; Xiaoqi Li; Yingjie Mao; Yishun Wang
>
> **摘要:** Deep neural networks (DNNs) play a crucial role in the field of artificial intelligence, and their security-related testing has been a prominent research focus. By inputting test cases, the behavior of models is examined for anomalies, and coverage metrics are utilized to determine the extent of neurons covered by these test cases. With the widespread application and advancement of DNNs, different types of neural behaviors have garnered attention, leading to the emergence of various coverage metrics for neural networks. However, there is currently a lack of empirical research on these coverage metrics, specifically in analyzing the relationships and patterns between model depth, configuration information, and neural network coverage. This paper aims to investigate the relationships and patterns of four coverage metrics: primary functionality, boundary, hierarchy, and structural coverage. A series of empirical experiments were conducted, selecting LeNet, VGG, and ResNet as different DNN architectures, along with 10 models of varying depths ranging from 5 to 54 layers, to compare and study the relationships between different depths, configuration information, and various neural network coverage metrics. Additionally, an investigation was carried out on the relationships between modified decision/condition coverage and dataset size. Finally, three potential future directions are proposed to further contribute to the security testing of DNN Models.
>
---
#### [new 038] Using Foundation Models as Pseudo-Label Generators for Pre-Clinical 4D Cardiac CT Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决人类与猪心脏CT数据间的领域差异问题。提出利用基础模型生成伪标签，通过无监督自训练方法迭代优化猪心脏4D CT分割，无需人工标注，提升分割精度并改善时序一致性。**

- **链接: [http://arxiv.org/pdf/2505.09564v1](http://arxiv.org/pdf/2505.09564v1)**

> **作者:** Anne-Marie Rickmann; Stephanie L. Thorn; Shawn S. Ahn; Supum Lee; Selen Uman; Taras Lysyy; Rachel Burns; Nicole Guerrera; Francis G. Spinale; Jason A. Burdick; Albert J. Sinusas; James S. Duncan
>
> **备注:** accepted at FIMH 2025
>
> **摘要:** Cardiac image segmentation is an important step in many cardiac image analysis and modeling tasks such as motion tracking or simulations of cardiac mechanics. While deep learning has greatly advanced segmentation in clinical settings, there is limited work on pre-clinical imaging, notably in porcine models, which are often used due to their anatomical and physiological similarity to humans. However, differences between species create a domain shift that complicates direct model transfer from human to pig data. Recently, foundation models trained on large human datasets have shown promise for robust medical image segmentation; yet their applicability to porcine data remains largely unexplored. In this work, we investigate whether foundation models can generate sufficiently accurate pseudo-labels for pig cardiac CT and propose a simple self-training approach to iteratively refine these labels. Our method requires no manually annotated pig data, relying instead on iterative updates to improve segmentation quality. We demonstrate that this self-training process not only enhances segmentation accuracy but also smooths out temporal inconsistencies across consecutive frames. Although our results are encouraging, there remains room for improvement, for example by incorporating more sophisticated self-training strategies and by exploring additional foundation models and other cardiac imaging technologies.
>
---
#### [new 039] RobustSpring: Benchmarking Robustness to Image Corruptions for Optical Flow, Scene Flow and Stereo
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于计算机视觉模型鲁棒性评估任务，旨在解决现有光流/场景流/立体算法基准忽视现实图像干扰（如噪声/雨）的问题。研究者提出了RobustSpring数据集，通过对Spring数据集施加20类时间-深度一致的图像退化，构建含2万张损坏图像的基准，并设计新指标量化模型鲁棒性，发现高精度模型未必具备抗干扰能力。**

- **链接: [http://arxiv.org/pdf/2505.09368v1](http://arxiv.org/pdf/2505.09368v1)**

> **作者:** Jenny Schmalfuss; Victor Oei; Lukas Mehl; Madlen Bartsch; Shashank Agnihotri; Margret Keuper; Andrés Bruhn
>
> **摘要:** Standard benchmarks for optical flow, scene flow, and stereo vision algorithms generally focus on model accuracy rather than robustness to image corruptions like noise or rain. Hence, the resilience of models to such real-world perturbations is largely unquantified. To address this, we present RobustSpring, a comprehensive dataset and benchmark for evaluating robustness to image corruptions for optical flow, scene flow, and stereo models. RobustSpring applies 20 different image corruptions, including noise, blur, color changes, quality degradations, and weather distortions, in a time-, stereo-, and depth-consistent manner to the high-resolution Spring dataset, creating a suite of 20,000 corrupted images that reflect challenging conditions. RobustSpring enables comparisons of model robustness via a new corruption robustness metric. Integration with the Spring benchmark enables public two-axis evaluations of both accuracy and robustness. We benchmark a curated selection of initial models, observing that accurate models are not necessarily robust and that robustness varies widely by corruption type. RobustSpring is a new computer vision benchmark that treats robustness as a first-class citizen to foster models that combine accuracy with resilience. It will be available at https://spring-benchmark.org.
>
---
#### [new 040] Learning Cocoercive Conservative Denoisers via Helmholtz Decomposition for Poisson Inverse Problems
- **分类: cs.CV; cs.LG; math.FA; math.OC; 94A08, 47H10, 47J26, 46N10, 47N10**

- **简介: 该论文针对泊松逆问题中的图像复原任务，解决传统即插即用方法因非扩张去噪器限制性能的问题。提出基于Helmholtz分解的协同保守去噪器(CoCo)，通过哈密顿正则化和谱正则化训练，构建隐式弱凸先验模型，在理论收敛保障下实现更优去噪效果。**

- **链接: [http://arxiv.org/pdf/2505.08909v1](http://arxiv.org/pdf/2505.08909v1)**

> **作者:** Deliang Wei; Peng Chen; Haobo Xu; Jiale Yao; Fang Li; Tieyong Zeng
>
> **备注:** 31 pages
>
> **摘要:** Plug-and-play (PnP) methods with deep denoisers have shown impressive results in imaging problems. They typically require strong convexity or smoothness of the fidelity term and a (residual) non-expansive denoiser for convergence. These assumptions, however, are violated in Poisson inverse problems, and non-expansiveness can hinder denoising performance. To address these challenges, we propose a cocoercive conservative (CoCo) denoiser, which may be (residual) expansive, leading to improved denoising. By leveraging the generalized Helmholtz decomposition, we introduce a novel training strategy that combines Hamiltonian regularization to promote conservativeness and spectral regularization to ensure cocoerciveness. We prove that CoCo denoiser is a proximal operator of a weakly convex function, enabling a restoration model with an implicit weakly convex prior. The global convergence of PnP methods to a stationary point of this restoration model is established. Extensive experimental results demonstrate that our approach outperforms closely related methods in both visual quality and quantitative metrics.
>
---
#### [new 041] UWAV: Uncertainty-weighted Weakly-supervised Audio-Visual Video Parsing
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 该论文研究弱监督音频-视觉视频解析任务，解决现有方法生成伪标签时忽略片段关联及预测偏差的问题。提出UWAV模型，通过不确定性加权评估伪标签可靠性，并引入特征混合正则化优化训练。实验表明其在多数据集上超越现有方法，验证了有效性。**

- **链接: [http://arxiv.org/pdf/2505.09615v1](http://arxiv.org/pdf/2505.09615v1)**

> **作者:** Yung-Hsuan Lai; Janek Ebbers; Yu-Chiang Frank Wang; François Germain; Michael Jeffrey Jones; Moitreya Chatterjee
>
> **备注:** CVPR 2025
>
> **摘要:** Audio-Visual Video Parsing (AVVP) entails the challenging task of localizing both uni-modal events (i.e., those occurring exclusively in either the visual or acoustic modality of a video) and multi-modal events (i.e., those occurring in both modalities concurrently). Moreover, the prohibitive cost of annotating training data with the class labels of all these events, along with their start and end times, imposes constraints on the scalability of AVVP techniques unless they can be trained in a weakly-supervised setting, where only modality-agnostic, video-level labels are available in the training data. To this end, recently proposed approaches seek to generate segment-level pseudo-labels to better guide model training. However, the absence of inter-segment dependencies when generating these pseudo-labels and the general bias towards predicting labels that are absent in a segment limit their performance. This work proposes a novel approach towards overcoming these weaknesses called Uncertainty-weighted Weakly-supervised Audio-visual Video Parsing (UWAV). Additionally, our innovative approach factors in the uncertainty associated with these estimated pseudo-labels and incorporates a feature mixup based training regularization for improved training. Empirical results show that UWAV outperforms state-of-the-art methods for the AVVP task on multiple metrics, across two different datasets, attesting to its effectiveness and generalizability.
>
---
#### [new 042] A 2D Semantic-Aware Position Encoding for Vision Transformers
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对计算机视觉中ViT的位置编码问题，提出了一种2D语义感知位置编码方法（SaPE²）。传统方法依赖一维线性位置关系，忽视图像块间的语义关联，导致模型泛化与平移等变性不足。SaPE²通过动态适应局部内容生成位置表征，提升跨分辨率适应能力，强化语义相似区域的关联，改进视觉任务性能。**

- **链接: [http://arxiv.org/pdf/2505.09466v1](http://arxiv.org/pdf/2505.09466v1)**

> **作者:** Xi Chen; Shiyang Zhou; Muqi Huang; Jiaxu Feng; Yun Xiong; Kun Zhou; Biao Yang; Yuhui Zhang; Huishuai Bao; Sijia Peng; Chuan Li; Feng Shi
>
> **备注:** 14 pages, 4 figures, 3 tables
>
> **摘要:** Vision transformers have demonstrated significant advantages in computer vision tasks due to their ability to capture long-range dependencies and contextual relationships through self-attention. However, existing position encoding techniques, which are largely borrowed from natural language processing, fail to effectively capture semantic-aware positional relationships between image patches. Traditional approaches like absolute position encoding and relative position encoding primarily focus on 1D linear position relationship, often neglecting the semantic similarity between distant yet contextually related patches. These limitations hinder model generalization, translation equivariance, and the ability to effectively handle repetitive or structured patterns in images. In this paper, we propose 2-Dimensional Semantic-Aware Position Encoding ($\text{SaPE}^2$), a novel position encoding method with semantic awareness that dynamically adapts position representations by leveraging local content instead of fixed linear position relationship or spatial coordinates. Our method enhances the model's ability to generalize across varying image resolutions and scales, improves translation equivariance, and better aggregates features for visually similar but spatially distant patches. By integrating $\text{SaPE}^2$ into vision transformers, we bridge the gap between position encoding and perceptual similarity, thereby improving performance on computer vision tasks.
>
---
#### [new 043] LightLab: Controlling Light Sources in Images with Diffusion Models
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于图像光照编辑任务，旨在解决现有方法依赖多视图输入或缺乏显式光源控制的问题。作者提出基于扩散模型的方法，利用少量真实图像对和大规模合成数据微调模型，通过光的线性特性合成可控光照变化的图像，实现了对光源强度与颜色的精准编辑，效果优于现有技术。**

- **链接: [http://arxiv.org/pdf/2505.09608v1](http://arxiv.org/pdf/2505.09608v1)**

> **作者:** Nadav Magar; Amir Hertz; Eric Tabellion; Yael Pritch; Alex Rav-Acha; Ariel Shamir; Yedid Hoshen
>
> **备注:** Project Page: https://nadmag.github.io/LightLab/
>
> **摘要:** We present a simple, yet effective diffusion-based method for fine-grained, parametric control over light sources in an image. Existing relighting methods either rely on multiple input views to perform inverse rendering at inference time, or fail to provide explicit control over light changes. Our method fine-tunes a diffusion model on a small set of real raw photograph pairs, supplemented by synthetically rendered images at scale, to elicit its photorealistic prior for relighting. We leverage the linearity of light to synthesize image pairs depicting controlled light changes of either a target light source or ambient illumination. Using this data and an appropriate fine-tuning scheme, we train a model for precise illumination changes with explicit control over light intensity and color. Lastly, we show how our method can achieve compelling light editing results, and outperforms existing methods based on user preference.
>
---
#### [new 044] Unsupervised Multiview Contrastive Language-Image Joint Learning with Pseudo-Labeled Prompts Via Vision-Language Model for 3D/4D Facial Expression Recognition
- **分类: cs.CV**

- **简介: 该论文针对3D/4D面部表情识别任务，提出无监督多视角视觉-语言联合学习框架MultiviewVLM。通过生成文本伪标签实现情感语义对齐，构建多视角联合嵌入空间，结合对比学习和稳定正负样本采样提升特征判别力，并设计梯度友好损失函数优化训练过程。**

- **链接: [http://arxiv.org/pdf/2505.09336v1](http://arxiv.org/pdf/2505.09336v1)**

> **作者:** Muzammil Behzad
>
> **摘要:** In this paper, we introduce MultiviewVLM, a vision-language model designed for unsupervised contrastive multiview representation learning of facial emotions from 3D/4D data. Our architecture integrates pseudo-labels derived from generated textual prompts to guide implicit alignment of emotional semantics. To capture shared information across multi-views, we propose a joint embedding space that aligns multiview representations without requiring explicit supervision. We further enhance the discriminability of our model through a novel multiview contrastive learning strategy that leverages stable positive-negative pair sampling. A gradient-friendly loss function is introduced to promote smoother and more stable convergence, and the model is optimized for distributed training to ensure scalability. Extensive experiments demonstrate that MultiviewVLM outperforms existing state-of-the-art methods and can be easily adapted to various real-world applications with minimal modifications.
>
---
#### [new 045] Zero-Shot Multi-modal Large Language Model v.s. Supervised Deep Learning: A Comparative Study on CT-Based Intracranial Hemorrhage Subtyping
- **分类: cs.CV**

- **简介: 该论文属于医学影像分类任务，对比零样本多模态大语言模型(MLLMs)与传统深度学习在CT颅内出血亚型分型的性能。研究使用192例NCCT数据，测试GPT-4o等MLLM与ResNet等模型，发现传统方法在出血分类和亚型识别中准确率更高，但MLLMs通过语言交互增强了结果可解释性，展现了医疗图像分析的潜力。**

- **链接: [http://arxiv.org/pdf/2505.09252v1](http://arxiv.org/pdf/2505.09252v1)**

> **作者:** Yinuo Wang; Yue Zeng; Kai Chen; Cai Meng; Chao Pan; Zhouping Tang
>
> **摘要:** Introduction: Timely identification of intracranial hemorrhage (ICH) subtypes on non-contrast computed tomography is critical for prognosis prediction and therapeutic decision-making, yet remains challenging due to low contrast and blurring boundaries. This study evaluates the performance of zero-shot multi-modal large language models (MLLMs) compared to traditional deep learning methods in ICH binary classification and subtyping. Methods: We utilized a dataset provided by RSNA, comprising 192 NCCT volumes. The study compares various MLLMs, including GPT-4o, Gemini 2.0 Flash, and Claude 3.5 Sonnet V2, with conventional deep learning models, including ResNet50 and Vision Transformer. Carefully crafted prompts were used to guide MLLMs in tasks such as ICH presence, subtype classification, localization, and volume estimation. Results: The results indicate that in the ICH binary classification task, traditional deep learning models outperform MLLMs comprehensively. For subtype classification, MLLMs also exhibit inferior performance compared to traditional deep learning models, with Gemini 2.0 Flash achieving an macro-averaged precision of 0.41 and a macro-averaged F1 score of 0.31. Conclusion: While MLLMs excel in interactive capabilities, their overall accuracy in ICH subtyping is inferior to deep networks. However, MLLMs enhance interpretability through language interactions, indicating potential in medical imaging analysis. Future efforts will focus on model refinement and developing more precise MLLMs to improve performance in three-dimensional medical image processing.
>
---
#### [new 046] Prioritizing Image-Related Tokens Enhances Vision-Language Pre-Training
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文针对视觉语言预训练任务，解决传统方法因均匀处理所有文本标记导致的噪声拟合和幻觉问题。提出PRIOR方法，通过纯文本LLM生成重要性权重，在损失函数中优先学习图像相关标记，在两种模型架构中分别实现19%和8%性能提升，并展现出更强扩展性。**

- **链接: [http://arxiv.org/pdf/2505.08971v1](http://arxiv.org/pdf/2505.08971v1)**

> **作者:** Yangyi Chen; Hao Peng; Tong Zhang; Heng Ji
>
> **备注:** The code will be available at https://github.com/Yangyi-Chen/PRIOR
>
> **摘要:** In standard large vision-language models (LVLMs) pre-training, the model typically maximizes the joint probability of the caption conditioned on the image via next-token prediction (NTP); however, since only a small subset of caption tokens directly relates to the visual content, this naive NTP unintentionally fits the model to noise and increases the risk of hallucination. We present PRIOR, a simple vision-language pre-training approach that addresses this issue by prioritizing image-related tokens through differential weighting in the NTP loss, drawing from the importance sampling framework. PRIOR introduces a reference model-a text-only large language model (LLM) trained on the captions without image inputs, to weight each token based on its probability for LVLMs training. Intuitively, tokens that are directly related to the visual inputs are harder to predict without the image and thus receive lower probabilities from the text-only reference LLM. During training, we implement a token-specific re-weighting term based on the importance scores to adjust each token's loss. We implement PRIOR in two distinct settings: LVLMs with visual encoders and LVLMs without visual encoders. We observe 19% and 8% average relative improvement, respectively, on several vision-language benchmarks compared to NTP. In addition, PRIOR exhibits superior scaling properties, as demonstrated by significantly higher scaling coefficients, indicating greater potential for performance gains compared to NTP given increasing compute and data.
>
---
#### [new 047] Optimizing Neuro-Fuzzy and Colonial Competition Algorithms for Skin Cancer Diagnosis in Dermatoscopic Images
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在提升皮肤癌诊断准确率。针对临床资源不足和AI应用不成熟的问题，研究融合神经模糊和殖民竞争算法处理皮肤镜图像，在ISIC数据库的560张图像中实现94%的准确率，辅助早期黑色素瘤检测。**

- **链接: [http://arxiv.org/pdf/2505.08886v1](http://arxiv.org/pdf/2505.08886v1)**

> **作者:** Hamideh Khaleghpour; Brett McKinney
>
> **备注:** 7 pages, 10 figures. Accepted at the 2nd Asia Pacific Computer Systems Conference (APCS 2024), March 15-17, 2024
>
> **摘要:** The rising incidence of skin cancer, coupled with limited public awareness and a shortfall in clinical expertise, underscores an urgent need for advanced diagnostic aids. Artificial Intelligence (AI) has emerged as a promising tool in this domain, particularly for distinguishing malignant from benign skin lesions. Leveraging publicly available datasets of skin lesions, researchers have been developing AI-based diagnostic solutions. However, the integration of such computer systems in clinical settings is still nascent. This study aims to bridge this gap by employing a fusion of image processing techniques and machine learning algorithms, specifically neuro-fuzzy and colonial competition approaches. Applied to dermoscopic images from the ISIC database, our method achieved a notable accuracy of 94% on a dataset of 560 images. These results underscore the potential of our approach in aiding clinicians in the early detection of melanoma, thereby contributing significantly to skin cancer diagnostics.
>
---
#### [new 048] Endo-CLIP: Progressive Self-Supervised Pre-training on Raw Colonoscopy Records
- **分类: cs.CV; cs.AI**

- **简介: Endo-CLIP提出一种自监督预训练框架，针对结肠镜图像-文本数据中的背景干扰、复杂术语及多病灶歧义问题，通过三阶段（清洗、细粒度对比学习、患者级跨注意力）优化CLIP模型，提升息肉检测与分类的零/少样本性能，推动精准内镜分析。**

- **链接: [http://arxiv.org/pdf/2505.09435v1](http://arxiv.org/pdf/2505.09435v1)**

> **作者:** Yili He; Yan Zhu; Peiyao Fu; Ruijie Yang; Tianyi Chen; Zhihua Wang; Quanlin Li; Pinghong Zhou; Xian Yang; Shuo Wang
>
> **备注:** Early accepted to MICCAI 2025
>
> **摘要:** Pre-training on image-text colonoscopy records offers substantial potential for improving endoscopic image analysis, but faces challenges including non-informative background images, complex medical terminology, and ambiguous multi-lesion descriptions. We introduce Endo-CLIP, a novel self-supervised framework that enhances Contrastive Language-Image Pre-training (CLIP) for this domain. Endo-CLIP's three-stage framework--cleansing, attunement, and unification--addresses these challenges by (1) removing background frames, (2) leveraging large language models to extract clinical attributes for fine-grained contrastive learning, and (3) employing patient-level cross-attention to resolve multi-polyp ambiguities. Extensive experiments demonstrate that Endo-CLIP significantly outperforms state-of-the-art pre-training methods in zero-shot and few-shot polyp detection and classification, paving the way for more accurate and clinically relevant endoscopic analysis.
>
---
#### [new 049] MoRAL: Motion-aware Multi-Frame 4D Radar and LiDAR Fusion for Robust 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶3D物体检测任务，旨在解决多模态融合中雷达点云跨帧错位及动态信息利用不足的问题。提出MoRAL框架，通过运动补偿雷达编码器修正点云对齐，并设计注意力门控模块融合雷达运动特征增强动态目标检测，在VoD数据集上取得最优性能。**

- **链接: [http://arxiv.org/pdf/2505.09422v1](http://arxiv.org/pdf/2505.09422v1)**

> **作者:** Xiangyuan Peng; Yu Wang; Miao Tang; Bierzynski Kay; Lorenzo Servadei; Robert Wille
>
> **摘要:** Reliable autonomous driving systems require accurate detection of traffic participants. To this end, multi-modal fusion has emerged as an effective strategy. In particular, 4D radar and LiDAR fusion methods based on multi-frame radar point clouds have demonstrated the effectiveness in bridging the point density gap. However, they often neglect radar point clouds' inter-frame misalignment caused by object movement during accumulation and do not fully exploit the object dynamic information from 4D radar. In this paper, we propose MoRAL, a motion-aware multi-frame 4D radar and LiDAR fusion framework for robust 3D object detection. First, a Motion-aware Radar Encoder (MRE) is designed to compensate for inter-frame radar misalignment from moving objects. Later, a Motion Attention Gated Fusion (MAGF) module integrate radar motion features to guide LiDAR features to focus on dynamic foreground objects. Extensive evaluations on the View-of-Delft (VoD) dataset demonstrate that MoRAL outperforms existing methods, achieving the highest mAP of 73.30% in the entire area and 88.68% in the driving corridor. Notably, our method also achieves the best AP of 69.67% for pedestrians in the entire area and 96.25% for cyclists in the driving corridor.
>
---
#### [new 050] Test-Time Augmentation for Pose-invariant Face Recognition
- **分类: cs.CV**

- **简介: 该论文属于姿态不变人脸识别任务，旨在解决现有方法需反复训练及正面化导致失真的问题。提出Pose-TTA方法，在测试时通过生成多姿态匹配图像进行对比，避免身份信息损失，并设计加权特征聚合策略消除合成数据偏差。该方法无需模型重训练，可无缝集成现有系统。**

- **链接: [http://arxiv.org/pdf/2505.09256v1](http://arxiv.org/pdf/2505.09256v1)**

> **作者:** Jaemin Jung; Youngjoon Jang; Joon Son Chung
>
> **摘要:** The goal of this paper is to enhance face recognition performance by augmenting head poses during the testing phase. Existing methods often rely on training on frontalised images or learning pose-invariant representations, yet both approaches typically require re-training and testing for each dataset, involving a substantial amount of effort. In contrast, this study proposes Pose-TTA, a novel approach that aligns faces at inference time without additional training. To achieve this, we employ a portrait animator that transfers the source image identity into the pose of a driving image. Instead of frontalising a side-profile face -- which can introduce distortion -- Pose-TTA generates matching side-profile images for comparison, thereby reducing identity information loss. Furthermore, we propose a weighted feature aggregation strategy to address any distortions or biases arising from the synthetic data, thus enhancing the reliability of the augmented images. Extensive experiments on diverse datasets and with various pre-trained face recognition models demonstrate that Pose-TTA consistently improves inference performance. Moreover, our method is straightforward to integrate into existing face recognition pipelines, as it requires no retraining or fine-tuning of the underlying recognition models.
>
---
#### [new 051] Variational Visual Question Answering
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多模态视觉问答（VQA）模型的可靠性问题（如校准不足和分布外泛化差），提出变分学习方法IVON替代传统AdamW优化，通过参数后验分布提升模型校准度与弃权能力，在保持精度的同时显著降低校准误差（超50%）并提升OOD场景下覆盖率（8%）。属于多模态模型可靠性优化任务。**

- **链接: [http://arxiv.org/pdf/2505.09591v1](http://arxiv.org/pdf/2505.09591v1)**

> **作者:** Tobias Jan Wieczorek; Nathalie Daun; Mohammad Emtiyaz Khan; Marcus Rohrbach
>
> **备注:** 19 pages, 16 figures, under review at ICCV 2025
>
> **摘要:** Despite remarkable progress in multimodal models for Visual Question Answering (VQA), there remain major reliability concerns because the models can often be overconfident and miscalibrated, especially in out-of-distribution (OOD) settings. Plenty has been done to address such issues for unimodal models, but little work exists for multimodal cases. Here, we address unreliability in multimodal models by proposing a Variational VQA approach. Specifically, instead of fine-tuning vision-language models by using AdamW, we employ a recently proposed variational algorithm called IVON, which yields a posterior distribution over model parameters. Through extensive experiments, we show that our approach improves calibration and abstentions without sacrificing the accuracy of AdamW. For instance, compared to AdamW fine-tuning, we reduce Expected Calibration Error by more than 50% compared to the AdamW baseline and raise Coverage by 4% vs. SOTA (for a fixed risk of 1%). In the presence of distribution shifts, the performance gain is even higher, achieving 8% Coverage (@ 1% risk) improvement vs. SOTA when 50% of test cases are OOD. Overall, we present variational learning as a viable option to enhance the reliability of multimodal models.
>
---
#### [new 052] Learning to Detect Multi-class Anomalies with Just One Normal Image Prompt
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究多类异常检测任务，解决现有自注意力重建模型因特征一致性导致漏检及低分辨率分割不准的问题。提出OneNIP方法，仅用一张正常图像提示重建正常特征并恢复异常特征，结合监督细化器优化像素级分割，在多个工业数据集上性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.09264v1](http://arxiv.org/pdf/2505.09264v1)**

> **作者:** Bin-Bin Gao
>
> **备注:** Accepted by ECCV 2024
>
> **摘要:** Unsupervised reconstruction networks using self-attention transformers have achieved state-of-the-art performance for multi-class (unified) anomaly detection with a single model. However, these self-attention reconstruction models primarily operate on target features, which may result in perfect reconstruction for both normal and anomaly features due to high consistency with context, leading to failure in detecting anomalies. Additionally, these models often produce inaccurate anomaly segmentation due to performing reconstruction in a low spatial resolution latent space. To enable reconstruction models enjoying high efficiency while enhancing their generalization for unified anomaly detection, we propose a simple yet effective method that reconstructs normal features and restores anomaly features with just One Normal Image Prompt (OneNIP). In contrast to previous work, OneNIP allows for the first time to reconstruct or restore anomalies with just one normal image prompt, effectively boosting unified anomaly detection performance. Furthermore, we propose a supervised refiner that regresses reconstruction errors by using both real normal and synthesized anomalous images, which significantly improves pixel-level anomaly segmentation. OneNIP outperforms previous methods on three industry anomaly detection benchmarks: MVTec, BTAD, and VisA. The code and pre-trained models are available at https://github.com/gaobb/OneNIP.
>
---
#### [new 053] Flash-VL 2B: Optimizing Vision-Language Model Performance for Ultra-Low Latency and High Throughput
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型优化任务，旨在解决实时应用中模型延迟高、吞吐量低的问题。通过架构优化、令牌压缩、数据训练策略及隐式语义拼接技术，在保证精度的同时提升处理速度。经11项基准测试验证，实现了高效能实时部署。**

- **链接: [http://arxiv.org/pdf/2505.09498v1](http://arxiv.org/pdf/2505.09498v1)**

> **作者:** Bo Zhang; Shuo Li; Runhe Tian; Yang Yang; Jixin Tang; Jinhao Zhou; Lin Ma
>
> **备注:** 18 pages, 7 figures
>
> **摘要:** In this paper, we introduce Flash-VL 2B, a novel approach to optimizing Vision-Language Models (VLMs) for real-time applications, targeting ultra-low latency and high throughput without sacrificing accuracy. Leveraging advanced architectural enhancements and efficient computational strategies, Flash-VL 2B is designed to maximize throughput by reducing processing time while maintaining competitive performance across multiple vision-language benchmarks. Our approach includes tailored architectural choices, token compression mechanisms, data curation, training schemes, and a novel image processing technique called implicit semantic stitching that effectively balances computational load and model performance. Through extensive evaluations on 11 standard VLM benchmarks, we demonstrate that Flash-VL 2B achieves state-of-the-art results in both speed and accuracy, making it a promising solution for deployment in resource-constrained environments and large-scale real-time applications.
>
---
#### [new 054] Seeing Beyond the Scene: Enhancing Vision-Language Models with Interactional Reasoning
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型场景理解任务，旨在解决传统场景图方法在交互推理和记忆泛化上的不足。提出ISGR框架，通过双流图构建、交互查询激活知识及强化记忆策略，增强模型对物体协作的主动推理能力，提升复杂场景理解效果。**

- **链接: [http://arxiv.org/pdf/2505.09118v1](http://arxiv.org/pdf/2505.09118v1)**

> **作者:** Dayong Liang; Changmeng Zheng; Zhiyuan Wen; Yi Cai; Xiao-Yong Wei; Qing Li
>
> **摘要:** Traditional scene graphs primarily focus on spatial relationships, limiting vision-language models' (VLMs) ability to reason about complex interactions in visual scenes. This paper addresses two key challenges: (1) conventional detection-to-construction methods produce unfocused, contextually irrelevant relationship sets, and (2) existing approaches fail to form persistent memories for generalizing interaction reasoning to new scenes. We propose Interaction-augmented Scene Graph Reasoning (ISGR), a framework that enhances VLMs' interactional reasoning through three complementary components. First, our dual-stream graph constructor combines SAM-powered spatial relation extraction with interaction-aware captioning to generate functionally salient scene graphs with spatial grounding. Second, we employ targeted interaction queries to activate VLMs' latent knowledge of object functionalities, converting passive recognition into active reasoning about how objects work together. Finally, we introduce a lone-term memory reinforcement learning strategy with a specialized interaction-focused reward function that transforms transient patterns into long-term reasoning heuristics. Extensive experiments demonstrate that our approach significantly outperforms baseline methods on interaction-heavy reasoning benchmarks, with particularly strong improvements on complex scene understanding tasks. The source code can be accessed at https://github.com/open_upon_acceptance.
>
---
#### [new 055] Predicting butterfly species presence from satellite imagery using soft contrastive regularisation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多物种存在预测任务，旨在利用卫星影像提升生物多样性监测效率。研究针对传统方法依赖栖息地映射的局限，构建了英国蝴蝶物种数据集，优化ResNet模型预测多物种分布，并提出软监督对比正则化损失函数，有效处理概率标签，提升了高生物多样性区域的预测精度。**

- **链接: [http://arxiv.org/pdf/2505.09306v1](http://arxiv.org/pdf/2505.09306v1)**

> **作者:** Thijs L van der Plas; Stephen Law; Michael JO Pocock
>
> **备注:** To be published in the 2025 CVPR FGVC12 workshop
>
> **摘要:** The growing demand for scalable biodiversity monitoring methods has fuelled interest in remote sensing data, due to its widespread availability and extensive coverage. Traditionally, the application of remote sensing to biodiversity research has focused on mapping and monitoring habitats, but with increasing availability of large-scale citizen-science wildlife observation data, recent methods have started to explore predicting multi-species presence directly from satellite images. This paper presents a new data set for predicting butterfly species presence from satellite data in the United Kingdom. We experimentally optimise a Resnet-based model to predict multi-species presence from 4-band satellite images, and find that this model especially outperforms the mean rate baseline for locations with high species biodiversity. To improve performance, we develop a soft, supervised contrastive regularisation loss that is tailored to probabilistic labels (such as species-presence data), and demonstrate that this improves prediction accuracy. In summary, our new data set and contrastive regularisation method contribute to the open challenge of accurately predicting species biodiversity from remote sensing data, which is key for efficient biodiversity monitoring.
>
---
#### [new 056] Behind Maya: Building a Multilingual Vision Language Model
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态视觉语言任务，旨在解决现有模型在低资源语言和文化多样性场景下的性能缺陷。研究者提出了开源模型Maya，通过构建基于LLaVA的八语种图文预训练数据集，并开发支持多语言的视觉语言模型，提升跨文化场景下的图文理解能力。**

- **链接: [http://arxiv.org/pdf/2505.08910v1](http://arxiv.org/pdf/2505.08910v1)**

> **作者:** Nahid Alam; Karthik Reddy Kanjula; Surya Guthikonda; Timothy Chung; Bala Krishna S Vegesna; Abhipsha Das; Anthony Susevski; Ryan Sze-Yin Chan; S M Iftekhar Uddin; Shayekh Bin Islam; Roshan Santhosh; Snegha A; Drishti Sharma; Chen Liu; Isha Chaturvedi; Genta Indra Winata; Ashvanth. S; Snehanshu Mukherjee; Alham Fikri Aji
>
> **备注:** Accepted at VLM4ALL CVPR 2025 Workshop
>
> **摘要:** In recent times, we have seen a rapid development of large Vision-Language Models (VLMs). They have shown impressive results on academic benchmarks, primarily in widely spoken languages but lack performance on low-resource languages and varied cultural contexts. To address these limitations, we introduce Maya, an open-source Multilingual VLM. Our contributions are: 1) a multilingual image-text pretraining dataset in eight languages, based on the LLaVA pretraining dataset; and 2) a multilingual image-text model supporting these languages, enhancing cultural and linguistic comprehension in vision-language tasks. Code available at https://github.com/nahidalam/maya.
>
---
#### [new 057] Few-Shot Anomaly-Driven Generation for Anomaly Classification and Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对工业检测中异常样本稀缺问题，提出Few-Shot异常生成方法AnoGen，属于异常检测任务。通过三阶段工作：学习异常分布、扩散模型生成逼真异常数据、弱监督训练模型，有效提升异常分类与分割性能，在MVTec数据集上验证了方法有效性（如DRAEM分割AU-PR提升5.8%），解决现有方法合成异常与真实数据语义差异大的瓶颈。**

- **链接: [http://arxiv.org/pdf/2505.09263v1](http://arxiv.org/pdf/2505.09263v1)**

> **作者:** Guan Gui; Bin-Bin Gao; Jun Liu; Chengjie Wang; Yunsheng Wu
>
> **备注:** Accepted by ECCV 2024
>
> **摘要:** Anomaly detection is a practical and challenging task due to the scarcity of anomaly samples in industrial inspection. Some existing anomaly detection methods address this issue by synthesizing anomalies with noise or external data. However, there is always a large semantic gap between synthetic and real-world anomalies, resulting in weak performance in anomaly detection. To solve the problem, we propose a few-shot Anomaly-driven Generation (AnoGen) method, which guides the diffusion model to generate realistic and diverse anomalies with only a few real anomalies, thereby benefiting training anomaly detection models. Specifically, our work is divided into three stages. In the first stage, we learn the anomaly distribution based on a few given real anomalies and inject the learned knowledge into an embedding. In the second stage, we use the embedding and given bounding boxes to guide the diffusion model to generate realistic and diverse anomalies on specific objects (or textures). In the final stage, we propose a weakly-supervised anomaly detection method to train a more powerful model with generated anomalies. Our method builds upon DRAEM and DesTSeg as the foundation model and conducts experiments on the commonly used industrial anomaly detection dataset, MVTec. The experiments demonstrate that our generated anomalies effectively improve the model performance of both anomaly classification and segmentation tasks simultaneously, \eg, DRAEM and DseTSeg achieved a 5.8\% and 1.5\% improvement in AU-PR metric on segmentation task, respectively. The code and generated anomalous data are available at https://github.com/gaobb/AnoGen.
>
---
#### [new 058] FreeDriveRF: Monocular RGB Dynamic NeRF without Poses for Autonomous Driving via Point-Level Dynamic-Static Decoupling
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶动态场景重建任务，旨在解决现有动态NeRF依赖多传感器和精确姿态的问题。提出FreeDriveRF方法，仅用单目RGB图像，通过点级动态-静态解耦减少伪影，结合光流约束动态建模与动态流优化姿态，提升无界场景重建效果。**

- **链接: [http://arxiv.org/pdf/2505.09406v1](http://arxiv.org/pdf/2505.09406v1)**

> **作者:** Yue Wen; Liang Song; Yijia Liu; Siting Zhu; Yanzi Miao; Lijun Han; Hesheng Wang
>
> **备注:** 7 pages, 9 figures, accepted by ICRA2025
>
> **摘要:** Dynamic scene reconstruction for autonomous driving enables vehicles to perceive and interpret complex scene changes more precisely. Dynamic Neural Radiance Fields (NeRFs) have recently shown promising capability in scene modeling. However, many existing methods rely heavily on accurate poses inputs and multi-sensor data, leading to increased system complexity. To address this, we propose FreeDriveRF, which reconstructs dynamic driving scenes using only sequential RGB images without requiring poses inputs. We innovatively decouple dynamic and static parts at the early sampling level using semantic supervision, mitigating image blurring and artifacts. To overcome the challenges posed by object motion and occlusion in monocular camera, we introduce a warped ray-guided dynamic object rendering consistency loss, utilizing optical flow to better constrain the dynamic modeling process. Additionally, we incorporate estimated dynamic flow to constrain the pose optimization process, improving the stability and accuracy of unbounded scene reconstruction. Extensive experiments conducted on the KITTI and Waymo datasets demonstrate the superior performance of our method in dynamic scene modeling for autonomous driving.
>
---
#### [new 059] Conformal Bounds on Full-Reference Image Quality for Imaging Inverse Problems
- **分类: cs.CV**

- **简介: 该论文针对成像逆问题（如去噪、加速MRI）中因真实图像未知而无法直接计算全参考图像质量（FRIQ）指标的问题，提出结合保形预测和近似后验采样的方法，构建具有统计保证的FRIQ上下界，确保在用户指定误差概率内有效，以辅助医疗成像等安全关键场景的质量评估。**

- **链接: [http://arxiv.org/pdf/2505.09528v1](http://arxiv.org/pdf/2505.09528v1)**

> **作者:** Jeffrey Wen; Rizwan Ahmad; Philip Schniter
>
> **摘要:** In imaging inverse problems, we would like to know how close the recovered image is to the true image in terms of full-reference image quality (FRIQ) metrics like PSNR, SSIM, LPIPS, etc. This is especially important in safety-critical applications like medical imaging, where knowing that, say, the SSIM was poor could potentially avoid a costly misdiagnosis. But since we don't know the true image, computing FRIQ is non-trivial. In this work, we combine conformal prediction with approximate posterior sampling to construct bounds on FRIQ that are guaranteed to hold up to a user-specified error probability. We demonstrate our approach on image denoising and accelerated magnetic resonance imaging (MRI) problems. Code is available at https://github.com/jwen307/quality_uq.
>
---
#### [new 060] Examining Deployment and Refinement of the VIOLA-AI Intracranial Hemorrhage Model Using an Interactive NeoMedSys Platform
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医疗AI模型临床部署与优化任务，旨在解决AI工具在放射科实际应用中的效率及性能提升问题。研究开发了NeoMedSys平台，集成模型部署、测试及优化功能，并在挪威急诊科实测三个月，通过实时反馈和迭代训练，显著提升了颅内出血检测模型（VIOLA-AI）的灵敏度、特异性及AUC值。**

- **链接: [http://arxiv.org/pdf/2505.09380v1](http://arxiv.org/pdf/2505.09380v1)**

> **作者:** Qinghui Liu; Jon Nesvold; Hanna Raaum; Elakkyen Murugesu; Martin Røvang; Bradley J Maclntosh; Atle Bjørnerud; Karoline Skogen
>
> **备注:** 19 pages, 11 figures, on submission to BMC Methods
>
> **摘要:** Background: There are many challenges and opportunities in the clinical deployment of AI tools in radiology. The current study describes a radiology software platform called NeoMedSys that can enable efficient deployment and refinements of AI models. We evaluated the feasibility and effectiveness of running NeoMedSys for three months in real-world clinical settings and focused on improvement performance of an in-house developed AI model (VIOLA-AI) designed for intracranial hemorrhage (ICH) detection. Methods: NeoMedSys integrates tools for deploying, testing, and optimizing AI models with a web-based medical image viewer, annotation system, and hospital-wide radiology information systems. A pragmatic investigation was deployed using clinical cases of patients presenting to the largest Emergency Department in Norway (site-1) with suspected traumatic brain injury (TBI) or patients with suspected stroke (site-2). We assessed ICH classification performance as VIOLA-AI encountered new data and underwent pre-planned model retraining. Performance metrics included sensitivity, specificity, accuracy, and the area under the receiver operating characteristic curve (AUC). Results: NeoMedSys facilitated iterative improvements in the AI model, significantly enhancing its diagnostic accuracy. Automated bleed detection and segmentation were reviewed in near real-time to facilitate re-training VIOLA-AI. The iterative refinement process yielded a marked improvement in classification sensitivity, rising to 90.3% (from 79.2%), and specificity that reached 89.3% (from 80.7%). The bleed detection ROC analysis for the entire sample demonstrated a high area-under-the-curve (AUC) of 0.949 (from 0.873). Model refinement stages were associated with notable gains, highlighting the value of real-time radiologist feedback.
>
---
#### [new 061] Multimodal Fusion of Glucose Monitoring and Food Imagery for Caloric Content Prediction
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多模态健康监测任务，旨在解决传统血糖监测无法精准预测饮食热量的问题。研究提出融合血糖时序数据、微生物组/人口统计信息和餐前图像，采用注意力机制与卷积网络提取特征并进行后期融合。模型在40人数据集上验证，误差较基线降低50%，提升了糖尿病管理的饮食评估准确性。**

- **链接: [http://arxiv.org/pdf/2505.09018v1](http://arxiv.org/pdf/2505.09018v1)**

> **作者:** Adarsh Kumar
>
> **摘要:** Effective dietary monitoring is critical for managing Type 2 diabetes, yet accurately estimating caloric intake remains a major challenge. While continuous glucose monitors (CGMs) offer valuable physiological data, they often fall short in capturing the full nutritional profile of meals due to inter-individual and meal-specific variability. In this work, we introduce a multimodal deep learning framework that jointly leverages CGM time-series data, Demographic/Microbiome, and pre-meal food images to enhance caloric estimation. Our model utilizes attention based encoding and a convolutional feature extraction for meal imagery, multi-layer perceptrons for CGM and Microbiome data followed by a late fusion strategy for joint reasoning. We evaluate our approach on a curated dataset of over 40 participants, incorporating synchronized CGM, Demographic and Microbiome data and meal photographs with standardized caloric labels. Our model achieves a Root Mean Squared Relative Error (RMSRE) of 0.2544, outperforming the baselines models by over 50%. These findings demonstrate the potential of multimodal sensing to improve automated dietary assessment tools for chronic disease management.
>
---
#### [new 062] Camera-Only 3D Panoptic Scene Completion for Autonomous Driving through Differentiable Object Shapes
- **分类: cs.CV**

- **简介: 该论文研究自动驾驶中的3D全景场景补全任务，解决现有方法在遮挡区域预测与同类物体实例区分不足的问题。通过提出可集成到现有模型的对象模块和全景模块，利用体素网格标注数据实现可微分物体形状学习，扩展了3D语义场景补全框架，增强环境感知能力。**

- **链接: [http://arxiv.org/pdf/2505.09562v1](http://arxiv.org/pdf/2505.09562v1)**

> **作者:** Nicola Marinello; Simen Cassiman; Jonas Heylen; Marc Proesmans; Luc Van Gool
>
> **备注:** Accepted to CVPR 2025 Workshop on Autonomous Driving
>
> **摘要:** Autonomous vehicles need a complete map of their surroundings to plan and act. This has sparked research into the tasks of 3D occupancy prediction, 3D scene completion, and 3D panoptic scene completion, which predict a dense map of the ego vehicle's surroundings as a voxel grid. Scene completion extends occupancy prediction by predicting occluded regions of the voxel grid, and panoptic scene completion further extends this task by also distinguishing object instances within the same class; both aspects are crucial for path planning and decision-making. However, 3D panoptic scene completion is currently underexplored. This work introduces a novel framework for 3D panoptic scene completion that extends existing 3D semantic scene completion models. We propose an Object Module and Panoptic Module that can easily be integrated with 3D occupancy and scene completion methods presented in the literature. Our approach leverages the available annotations in occupancy benchmarks, allowing individual object shapes to be learned as a differentiable problem. The code is available at https://github.com/nicolamarinello/OffsetOcc .
>
---
#### [new 063] Marigold: Affordable Adaptation of Diffusion-Based Image Generators for Image Analysis
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于计算机视觉领域，研究如何利用预训练扩散模型（如Stable Diffusion）进行密集图像分析任务（深度估计等）。针对数据稀缺场景下传统预训练方法受限的问题，提出Marigold框架：通过微调协议提取生成模型的视觉知识，仅需小规模合成数据与单GPU训练，即可实现零样本泛化，保持原模型架构的前提下完成图像解析任务。**

- **链接: [http://arxiv.org/pdf/2505.09358v1](http://arxiv.org/pdf/2505.09358v1)**

> **作者:** Bingxin Ke; Kevin Qu; Tianfu Wang; Nando Metzger; Shengyu Huang; Bo Li; Anton Obukhov; Konrad Schindler
>
> **备注:** Journal extension of our CVPR 2024 paper, featuring new tasks, improved efficiency, high-resolution capabilities, and enhanced accessibility
>
> **摘要:** The success of deep learning in computer vision over the past decade has hinged on large labeled datasets and strong pretrained models. In data-scarce settings, the quality of these pretrained models becomes crucial for effective transfer learning. Image classification and self-supervised learning have traditionally been the primary methods for pretraining CNNs and transformer-based architectures. Recently, the rise of text-to-image generative models, particularly those using denoising diffusion in a latent space, has introduced a new class of foundational models trained on massive, captioned image datasets. These models' ability to generate realistic images of unseen content suggests they possess a deep understanding of the visual world. In this work, we present Marigold, a family of conditional generative models and a fine-tuning protocol that extracts the knowledge from pretrained latent diffusion models like Stable Diffusion and adapts them for dense image analysis tasks, including monocular depth estimation, surface normals prediction, and intrinsic decomposition. Marigold requires minimal modification of the pre-trained latent diffusion model's architecture, trains with small synthetic datasets on a single GPU over a few days, and demonstrates state-of-the-art zero-shot generalization. Project page: https://marigoldcomputervision.github.io
>
---
#### [new 064] Generative AI for Urban Planning: Synthesizing Satellite Imagery via Diffusion Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于生成式AI任务，利用扩散模型生成城市规划卫星图像。针对现有方法生成不现实、难扩展的问题，结合ControlNet改进Stable Diffusion，整合OpenStreetMap土地数据和约束条件，生成高保真、多样化的城市景观。通过多城市数据验证模型性能，定量评估显示高FID/KID分数，定性评估表明生成图像优于真实图像，为规划流程和公众参与提供新工具。**

- **链接: [http://arxiv.org/pdf/2505.08833v1](http://arxiv.org/pdf/2505.08833v1)**

> **作者:** Qingyi Wang; Yuebing Liang; Yunhan Zheng; Kaiyuan Xu; Jinhua Zhao; Shenhao Wang
>
> **摘要:** Generative AI offers new opportunities for automating urban planning by creating site-specific urban layouts and enabling flexible design exploration. However, existing approaches often struggle to produce realistic and practical designs at scale. Therefore, we adapt a state-of-the-art Stable Diffusion model, extended with ControlNet, to generate high-fidelity satellite imagery conditioned on land use descriptions, infrastructure, and natural environments. To overcome data availability limitations, we spatially link satellite imagery with structured land use and constraint information from OpenStreetMap. Using data from three major U.S. cities, we demonstrate that the proposed diffusion model generates realistic and diverse urban landscapes by varying land-use configurations, road networks, and water bodies, facilitating cross-city learning and design diversity. We also systematically evaluate the impacts of varying language prompts and control imagery on the quality of satellite imagery generation. Our model achieves high FID and KID scores and demonstrates robustness across diverse urban contexts. Qualitative assessments from urban planners and the general public show that generated images align closely with design descriptions and constraints, and are often preferred over real images. This work establishes a benchmark for controlled urban imagery generation and highlights the potential of generative AI as a tool for enhancing planning workflows and public engagement.
>
---
#### [new 065] 2D-3D Attention and Entropy for Pose Robust 2D Facial Recognition
- **分类: cs.CV**

- **简介: 该论文属于人脸识别任务，旨在解决因姿态差异导致的识别性能下降问题。通过提出一种域自适应框架，结合共享注意力机制（关联2D图像与3D点云特征）和联合熵正则化损失（增强跨模态一致性），提升大姿态差异下的识别鲁棒性。实验在FaceScape和ARL-VTF数据集上验证有效性，性能显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.09073v1](http://arxiv.org/pdf/2505.09073v1)**

> **作者:** J. Brennan Peace; Shuowen Hu; Benjamin S. Riggan
>
> **备注:** To appear at the IEEE International Conference on Automatic Face and Gesture 2025 (FG2025)
>
> **摘要:** Despite recent advances in facial recognition, there remains a fundamental issue concerning degradations in performance due to substantial perspective (pose) differences between enrollment and query (probe) imagery. Therefore, we propose a novel domain adaptive framework to facilitate improved performances across large discrepancies in pose by enabling image-based (2D) representations to infer properties of inherently pose invariant point cloud (3D) representations. Specifically, our proposed framework achieves better pose invariance by using (1) a shared (joint) attention mapping to emphasize common patterns that are most correlated between 2D facial images and 3D facial data and (2) a joint entropy regularizing loss to promote better consistency$\unicode{x2014}$enhancing correlations among the intersecting 2D and 3D representations$\unicode{x2014}$by leveraging both attention maps. This framework is evaluated on FaceScape and ARL-VTF datasets, where it outperforms competitive methods by achieving profile (90$\unicode{x00b0}$$\unicode{x002b}$) TAR @ 1$\unicode{x0025}$ FAR improvements of at least 7.1$\unicode{x0025}$ and 1.57$\unicode{x0025}$, respectively.
>
---
#### [new 066] OptiGait-LGBM: An Efficient Approach of Gait-based Person Re-identification in Non-Overlapping Regions
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 该论文属于步态识别任务，解决非重叠视角、复杂环境下行人重识别准确率低及计算效率问题。提出OptiGait-LGBM模型，利用骨骼关节点构建数据集RUET-GAIT，降低内存消耗，并通过轻量级分类方法提升性能，在精度、内存和训练速度上优于主流模型。**

- **链接: [http://arxiv.org/pdf/2505.08801v1](http://arxiv.org/pdf/2505.08801v1)**

> **作者:** Md. Sakib Hassan Chowdhury; Md. Hafiz Ahamed; Bishowjit Paul; Sarafat Hussain Abhi; Abu Bakar Siddique; Md. Robius Sany
>
> **备注:** 12 pages, 17 figures
>
> **摘要:** Gait recognition, known for its ability to identify individuals from a distance, has gained significant attention in recent times due to its non-intrusive verification. While video-based gait identification systems perform well on large public datasets, their performance drops when applied to real-world, unconstrained gait data due to various factors. Among these, uncontrolled outdoor environments, non-overlapping camera views, varying illumination, and computational efficiency are core challenges in gait-based authentication. Currently, no dataset addresses all these challenges simultaneously. In this paper, we propose an OptiGait-LGBM model capable of recognizing person re-identification under these constraints using a skeletal model approach, which helps mitigate inconsistencies in a person's appearance. The model constructs a dataset from landmark positions, minimizing memory usage by using non-sequential data. A benchmark dataset, RUET-GAIT, is introduced to represent uncontrolled gait sequences in complex outdoor environments. The process involves extracting skeletal joint landmarks, generating numerical datasets, and developing an OptiGait-LGBM gait classification model. Our aim is to address the aforementioned challenges with minimal computational cost compared to existing methods. A comparative analysis with ensemble techniques such as Random Forest and CatBoost demonstrates that the proposed approach outperforms them in terms of accuracy, memory usage, and training time. This method provides a novel, low-cost, and memory-efficient video-based gait recognition solution for real-world scenarios.
>
---
#### [new 067] Optimizing Urban Critical Green Space Development Using Machine Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于城市绿地优化任务，旨在解决德黑兰绿地规划效率问题。研究整合多源数据（气象、环境、社会），利用随机森林模型（准确率94%）识别植被缺失区域，结合夜间地表温度、人口敏感度等关键指标生成优先开发地图，并通过微气候模拟验证绿色屋顶技术可降温0.67°C，为城市规划提供决策工具。**

- **链接: [http://arxiv.org/pdf/2505.09175v1](http://arxiv.org/pdf/2505.09175v1)**

> **作者:** Mohammad Ganjirad; Mahmoud Reza Delavar; Hossein Bagheri; Mohammad Mehdi Azizi
>
> **摘要:** This paper presents a novel framework for prioritizing urban green space development in Tehran using diverse socio-economic, environmental, and sensitivity indices. The indices were derived from various sources including Google Earth Engine, air pollution measurements, municipal reports and the Weather Research & Forecasting (WRF) model. The WRF model was used to estimate the air temperature at a 1 km resolution due to insufficient meteorological stations, yielding RMSE and MAE values of 0.96{\deg}C and 0.92{\deg}C, respectively. After data preparation, several machine learning models were used for binary vegetation cover classification including XGBoost, LightGBM, Random Forest (RF) and Extra Trees. RF achieved the highest performance, exceeding 94% in Overall Accuracy, Recall, and F1-score. Then, the probability of areas lacking vegetation cover was assessed using socio-economic, environmental and sensitivity indices. This resulted in the RF generating an urban green space development prioritization map. Feature Importance Analysis revealed that the most significant indices were nightly land surface temperature (LST) and sensitive population. Finally, the framework performance was validated through microclimate simulation to assess the critical areas after and before the green space development by green roofs. The simulation demonstrated reducing air temperature by up to 0.67{\deg}C after utilizing the green roof technology in critical areas. As a result, this framework provides a valuable tool for urban planners to develop green spaces.
>
---
#### [new 068] BiECVC: Gated Diversification of Bidirectional Contexts for Learned Video Compression
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于学习型双向视频压缩任务，解决现有方法因上下文提取单一且无法动态抑制无效信息导致性能不足的问题。提出BiECVC框架，通过多样化局部/非局部建模、线性注意力及动态门控机制增强上下文利用，实验表明其码率较VTM降低13%-15%，成为首个全面超越VTM的学习型编解码器。**

- **链接: [http://arxiv.org/pdf/2505.09193v1](http://arxiv.org/pdf/2505.09193v1)**

> **作者:** Wei Jiang; Junru Li; Kai Zhang; Li Zhang
>
> **备注:** The first learned video codec that surpasses VTM 13.2 RA across all standard test datasets. Code will be available at https://github.com/JiangWeibeta/ECVC
>
> **摘要:** Recent forward prediction-based learned video compression (LVC) methods have achieved impressive results, even surpassing VVC reference software VTM under the Low Delay B (LDB) configuration. In contrast, learned bidirectional video compression (BVC) remains underexplored and still lags behind its forward-only counterparts. This performance gap is mainly due to the limited ability to extract diverse and accurate contexts: most existing BVCs primarily exploit temporal motion while neglecting non-local correlations across frames. Moreover, they lack the adaptability to dynamically suppress harmful contexts arising from fast motion or occlusion. To tackle these challenges, we propose BiECVC, a BVC framework that incorporates diversified local and non-local context modeling along with adaptive context gating. For local context enhancement, BiECVC reuses high-quality features from lower layers and aligns them using decoded motion vectors without introducing extra motion overhead.To model non-local dependencies efficiently, we adopt a linear attention mechanism that balances performance and complexity. To further mitigate the impact of inaccurate context prediction, we introduce Bidirectional Context Gating, inspired by data-dependent decay in recent autoregressive language models, to dynamically filter contextual information based on conditional coding results. Extensive experiments demonstrate that BiECVC achieves state-of-the-art performance, reducing the bit-rate by 13.4% and 15.7% compared to VTM 13.2 under the Random Access (RA) configuration with intra periods of 32 and 64, respectively. To our knowledge, BiECVC is the first learned video codec to surpass VTM 13.2 RA across all standard test datasets. Code will be available at https://github.com/JiangWeibeta/ECVC.
>
---
#### [new 069] Q-space Guided Collaborative Attention Translation Network for Flexible Diffusion-Weighted Images Synthesis
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出Q-CATN网络，解决灵活q空间采样下多壳高分辨率扩散加权成像(DWI)的合成问题。通过协作注意力机制融合多模态MRI数据，动态适配不同采样方案，结合解剖约束保持成像精度，在HCP数据集上超越现有方法，为临床提供灵活高效的合成工具。**

- **链接: [http://arxiv.org/pdf/2505.09323v1](http://arxiv.org/pdf/2505.09323v1)**

> **作者:** Pengli Zhu; Yingji Fu; Nanguang Chen; Anqi Qiu
>
> **备注:** MICCAI 2025
>
> **摘要:** This study, we propose a novel Q-space Guided Collaborative Attention Translation Networks (Q-CATN) for multi-shell, high-angular resolution DWI (MS-HARDI) synthesis from flexible q-space sampling, leveraging the commonly acquired structural MRI data. Q-CATN employs a collaborative attention mechanism to effectively extract complementary information from multiple modalities and dynamically adjust its internal representations based on flexible q-space information, eliminating the need for fixed sampling schemes. Additionally, we introduce a range of task-specific constraints to preserve anatomical fidelity in DWI, enabling Q-CATN to accurately learn the intrinsic relationships between directional DWI signal distributions and q-space. Extensive experiments on the Human Connectome Project (HCP) dataset demonstrate that Q-CATN outperforms existing methods, including 1D-qDL, 2D-qDL, MESC-SD, and QGAN, in estimating parameter maps and fiber tracts both quantitatively and qualitatively, while preserving fine-grained details. Notably, its ability to accommodate flexible q-space sampling highlights its potential as a promising toolkit for clinical and research applications. Our code is available at https://github.com/Idea89560041/Q-CATN.
>
---
#### [new 070] Total Variation-Based Image Decomposition and Denoising for Microscopy Images
- **分类: eess.IV; cond-mat.mtrl-sci; cs.CV**

- **简介: 该论文研究显微镜图像去噪与分解任务，解决噪声及干扰信号导致图像质量下降的问题。基于总变差方法，评估TV-L1、Huber-ROF和TGV-L1模型性能，发现Huber-ROF灵活性最佳，TGV-L1去噪效果最优。开发了开源工具AiSurf支持多种显微镜图像处理。**

- **链接: [http://arxiv.org/pdf/2505.08843v1](http://arxiv.org/pdf/2505.08843v1)**

> **作者:** Marco Corrias; Giada Franceschi; Michele Riva; Alberto Tampieri; Karin Föttinger; Ulrike Diebold; Thomas Pock; Cesare Franchini
>
> **摘要:** Experimentally acquired microscopy images are unavoidably affected by the presence of noise and other unwanted signals, which degrade their quality and might hide relevant features. With the recent increase in image acquisition rate, modern denoising and restoration solutions become necessary. This study focuses on image decomposition and denoising of microscopy images through a workflow based on total variation (TV), addressing images obtained from various microscopy techniques, including atomic force microscopy (AFM), scanning tunneling microscopy (STM), and scanning electron microscopy (SEM). Our approach consists in restoring an image by extracting its unwanted signal components and subtracting them from the raw one, or by denoising it. We evaluate the performance of TV-$L^1$, Huber-ROF, and TGV-$L^1$ in achieving this goal in distinct study cases. Huber-ROF proved to be the most flexible one, while TGV-$L^1$ is the most suitable for denoising. Our results suggest a wider applicability of this method in microscopy, restricted not only to STM, AFM, and SEM images. The Python code used for this study is publicly available as part of AiSurf. It is designed to be integrated into experimental workflows for image acquisition or can be used to denoise previously acquired images.
>
---
#### [new 071] Robustness Analysis against Adversarial Patch Attacks in Fully Unmanned Stores
- **分类: cs.CR; cs.AI; cs.CV**

- **简介: 该论文属于人工智能安全任务，研究无人商店AI结账系统对抗补丁攻击的鲁棒性。针对物体检测模型易受物理对抗补丁（隐藏、生成、篡改）攻击导致安全漏洞的问题，提出颜色直方图相似性损失函数和边界框评估指标，通过数字模拟和实体环境测试攻击有效性，并探讨黑盒攻击防御策略。**

- **链接: [http://arxiv.org/pdf/2505.08835v1](http://arxiv.org/pdf/2505.08835v1)**

> **作者:** Hyunsik Na; Wonho Lee; Seungdeok Roh; Sohee Park; Daeseon Choi
>
> **摘要:** The advent of convenient and efficient fully unmanned stores equipped with artificial intelligence-based automated checkout systems marks a new era in retail. However, these systems have inherent artificial intelligence security vulnerabilities, which are exploited via adversarial patch attacks, particularly in physical environments. This study demonstrated that adversarial patches can severely disrupt object detection models used in unmanned stores, leading to issues such as theft, inventory discrepancies, and interference. We investigated three types of adversarial patch attacks -- Hiding, Creating, and Altering attacks -- and highlighted their effectiveness. We also introduce the novel color histogram similarity loss function by leveraging attacker knowledge of the color information of a target class object. Besides the traditional confusion-matrix-based attack success rate, we introduce a new bounding-boxes-based metric to analyze the practical impact of these attacks. Starting with attacks on object detection models trained on snack and fruit datasets in a digital environment, we evaluated the effectiveness of adversarial patches in a physical testbed that mimicked a real unmanned store with RGB cameras and realistic conditions. Furthermore, we assessed the robustness of these attacks in black-box scenarios, demonstrating that shadow attacks can enhance success rates of attacks even without direct access to model parameters. Our study underscores the necessity for robust defense strategies to protect unmanned stores from adversarial threats. Highlighting the limitations of the current defense mechanisms in real-time detection systems and discussing various proactive measures, we provide insights into improving the robustness of object detection models and fortifying unmanned retail environments against these attacks.
>
---
#### [new 072] FoldNet: Learning Generalizable Closed-Loop Policy for Garment Folding via Keypoint-Driven Asset and Demonstration Synthesis
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究机器人衣物折叠任务，解决数据不足及策略泛化问题。提出基于关键点的合成数据集构建方法，结合生成纹理和闭环模仿学习训练策略，设计KG-DAgger增强鲁棒性，实验显示真实成功率提升25%至75%。**

- **链接: [http://arxiv.org/pdf/2505.09109v1](http://arxiv.org/pdf/2505.09109v1)**

> **作者:** Yuxing Chen; Bowen Xiao; He Wang
>
> **摘要:** Due to the deformability of garments, generating a large amount of high-quality data for robotic garment manipulation tasks is highly challenging. In this paper, we present a synthetic garment dataset that can be used for robotic garment folding. We begin by constructing geometric garment templates based on keypoints and applying generative models to generate realistic texture patterns. Leveraging these keypoint annotations, we generate folding demonstrations in simulation and train folding policies via closed-loop imitation learning. To improve robustness, we propose KG-DAgger, which uses a keypoint-based strategy to generate demonstration data for recovering from failures. KG-DAgger significantly improves the model performance, boosting the real-world success rate by 25\%. After training with 15K trajectories (about 2M image-action pairs), the model achieves a 75\% success rate in the real world. Experiments in both simulation and real-world settings validate the effectiveness of our proposed framework.
>
---
#### [new 073] APR-Transformer: Initial Pose Estimation for Localization in Complex Environments through Absolute Pose Regression
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出APR-Transformer模型，解决GNSS缺失环境下机器人/自动驾驶定位算法因初始位姿不准导致的精度问题。通过图像或LiDAR数据实现绝对位姿回归（3D位置与方向），在标准数据集和自建复杂场景数据集中达到最优性能，并通过实车部署验证了可靠性。**

- **链接: [http://arxiv.org/pdf/2505.09356v1](http://arxiv.org/pdf/2505.09356v1)**

> **作者:** Srinivas Ravuri; Yuan Xu; Martin Ludwig Zehetner; Ketan Motlag; Sahin Albayrak
>
> **备注:** 8 pages with 6 figures
>
> **摘要:** Precise initialization plays a critical role in the performance of localization algorithms, especially in the context of robotics, autonomous driving, and computer vision. Poor localization accuracy is often a consequence of inaccurate initial poses, particularly noticeable in GNSS-denied environments where GPS signals are primarily relied upon for initialization. Recent advances in leveraging deep neural networks for pose regression have led to significant improvements in both accuracy and robustness, especially in estimating complex spatial relationships and orientations. In this paper, we introduce APR-Transformer, a model architecture inspired by state-of-the-art methods, which predicts absolute pose (3D position and 3D orientation) using either image or LiDAR data. We demonstrate that our proposed method achieves state-of-the-art performance on established benchmark datasets such as the Radar Oxford Robot-Car and DeepLoc datasets. Furthermore, we extend our experiments to include our custom complex APR-BeIntelli dataset. Additionally, we validate the reliability of our approach in GNSS-denied environments by deploying the model in real-time on an autonomous test vehicle. This showcases the practical feasibility and effectiveness of our approach. The source code is available at:https://github.com/GT-ARC/APR-Transformer.
>
---
#### [new 074] Meta-learning Slice-to-Volume Reconstruction in Fetal Brain MRI using Implicit Neural Representations
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文针对胎儿脑MRI中运动伪影导致的低分辨率切片重建问题，提出基于隐式神经表示的元学习方法，实现无需预对齐的运动校正、异常处理与超分辨率三维重建。通过自监督元学习注入先验知识，在严重运动干扰下提升重建质量50%并缩短50%耗时，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.09565v1](http://arxiv.org/pdf/2505.09565v1)**

> **作者:** Maik Dannecker; Thomas Sanchez; Meritxell Bach Cuadra; Özgün Turgut; Anthony N. Price; Lucilio Cordero-Grande; Vanessa Kyriakopoulou; Joseph V. Hajnal; Daniel Rueckert
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** High-resolution slice-to-volume reconstruction (SVR) from multiple motion-corrupted low-resolution 2D slices constitutes a critical step in image-based diagnostics of moving subjects, such as fetal brain Magnetic Resonance Imaging (MRI). Existing solutions struggle with image artifacts and severe subject motion or require slice pre-alignment to achieve satisfying reconstruction performance. We propose a novel SVR method to enable fast and accurate MRI reconstruction even in cases of severe image and motion corruption. Our approach performs motion correction, outlier handling, and super-resolution reconstruction with all operations being entirely based on implicit neural representations. The model can be initialized with task-specific priors through fully self-supervised meta-learning on either simulated or real-world data. In extensive experiments including over 480 reconstructions of simulated and clinical MRI brain data from different centers, we prove the utility of our method in cases of severe subject motion and image artifacts. Our results demonstrate improvements in reconstruction quality, especially in the presence of severe motion, compared to state-of-the-art methods, and up to 50% reduction in reconstruction time.
>
---
#### [new 075] GreenFactory: Ensembling Zero-Cost Proxies to Estimate Performance of Neural Networks
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于神经架构搜索（NAS）任务，旨在解决传统性能评估耗时及现有零成本代理泛化性差、无法预测准确率的问题。提出GreenFactory方法，通过随机森林回归器集成多个零成本代理，直接预测模型测试精度。实验在NATS-Bench数据集上验证了其高相关性结果，证明了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2505.09344v1](http://arxiv.org/pdf/2505.09344v1)**

> **作者:** Gabriel Cortês; Nuno Lourenço; Paolo Romano; Penousal Machado
>
> **摘要:** Determining the performance of a Deep Neural Network during Neural Architecture Search processes is essential for identifying optimal architectures and hyperparameters. Traditionally, this process requires training and evaluation of each network, which is time-consuming and resource-intensive. Zero-cost proxies estimate performance without training, serving as an alternative to traditional training. However, recent proxies often lack generalization across diverse scenarios and provide only relative rankings rather than predicted accuracies. To address these limitations, we propose GreenFactory, an ensemble of zero-cost proxies that leverages a random forest regressor to combine multiple predictors' strengths and directly predict model test accuracy. We evaluate GreenFactory on NATS-Bench, achieving robust results across multiple datasets. Specifically, GreenFactory achieves high Kendall correlations on NATS-Bench-SSS, indicating substantial agreement between its predicted scores and actual performance: 0.907 for CIFAR-10, 0.945 for CIFAR-100, and 0.920 for ImageNet-16-120. Similarly, on NATS-Bench-TSS, we achieve correlations of 0.921 for CIFAR-10, 0.929 for CIFAR-100, and 0.908 for ImageNet-16-120, showcasing its reliability in both search spaces.
>
---
#### [new 076] Toward Accessible and Safe Live Streaming Using Distributed Content Filtering with MoQ
- **分类: cs.MM; cs.CV; cs.DC; cs.NI**

- **简介: 该论文研究实时流媒体内容审核，旨在解决直播中有害内容过滤的延迟问题。通过扩展Media Over QUIC协议，提出分布式过滤方案，允许客户端设备参与分析并仅删除违规视频片段，实现低延迟（增加1个GOP时长）的安全直播，应用于光敏人群的频闪消除场景。**

- **链接: [http://arxiv.org/pdf/2505.08990v1](http://arxiv.org/pdf/2505.08990v1)**

> **作者:** Andrew C. Freeman
>
> **备注:** Accepted to the ICME 2025 LIVES workshop
>
> **摘要:** Live video streaming is increasingly popular on social media platforms. With the growth of live streaming comes an increased need for robust content moderation to remove dangerous, illegal, or otherwise objectionable content. Whereas video on demand distribution enables offline content analysis, live streaming imposes restrictions on latency for both analysis and distribution. In this paper, we present extensions to the in-progress Media Over QUIC Transport protocol that enable real-time content moderation in one-to-many video live streams. Importantly, our solution removes only the video segments that contain objectionable content, allowing playback resumption as soon as the stream conforms to content policies again. Content analysis tasks may be transparently distributed to arbitrary client devices. We implement and evaluate our system in the context of light strobe removal for photosensitive viewers, finding that streaming clients experience an increased latency of only one group-of-pictures duration.
>
---
#### [new 077] Validation of Conformal Prediction in Cervical Atypia Classification
- **分类: eess.IV; cs.AI; cs.CV; cs.LG; q-bio.QM**

- **简介: 该论文属于医疗图像分类任务，针对深度学习模型在宫颈非典型增生分类中过度自信、无法可靠反映诊断不确定性的问题，验证了三种符合预测方法生成预测集的效果。通过专家标注分析，发现传统评估高估性能，现有方法产生的预测集与人类标注一致性不足，并探索了模型处理模糊及分布外数据的能力。**

- **链接: [http://arxiv.org/pdf/2505.08845v1](http://arxiv.org/pdf/2505.08845v1)**

> **作者:** Misgina Tsighe Hagos; Antti Suutala; Dmitrii Bychkov; Hakan Kücükel; Joar von Bahr; Milda Poceviciute; Johan Lundin; Nina Linder; Claes Lundström
>
> **摘要:** Deep learning based cervical cancer classification can potentially increase access to screening in low-resource regions. However, deep learning models are often overconfident and do not reliably reflect diagnostic uncertainty. Moreover, they are typically optimized to generate maximum-likelihood predictions, which fail to convey uncertainty or ambiguity in their results. Such challenges can be addressed using conformal prediction, a model-agnostic framework for generating prediction sets that contain likely classes for trained deep-learning models. The size of these prediction sets indicates model uncertainty, contracting as model confidence increases. However, existing conformal prediction evaluation primarily focuses on whether the prediction set includes or covers the true class, often overlooking the presence of extraneous classes. We argue that prediction sets should be truthful and valuable to end users, ensuring that the listed likely classes align with human expectations rather than being overly relaxed and including false positives or unlikely classes. In this study, we comprehensively validate conformal prediction sets using expert annotation sets collected from multiple annotators. We evaluate three conformal prediction approaches applied to three deep-learning models trained for cervical atypia classification. Our expert annotation-based analysis reveals that conventional coverage-based evaluations overestimate performance and that current conformal prediction methods often produce prediction sets that are not well aligned with human labels. Additionally, we explore the capabilities of the conformal prediction methods in identifying ambiguous and out-of-distribution data.
>
---
#### [new 078] Template-Guided Reconstruction of Pulmonary Segments with Neural Implicit Functions
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 该论文属于医学图像3D重建任务，旨在解决传统深度学习方法在肺段重建中计算效率低、精度不足的问题。提出基于神经隐函数的方法，通过变形可学习模板实现高精度肺段建模，引入临床评估指标，并构建Lung3D数据集（含800标注样本）。方法优于现有技术，代码数据开源。**

- **链接: [http://arxiv.org/pdf/2505.08919v1](http://arxiv.org/pdf/2505.08919v1)**

> **作者:** Kangxian Xie; Yufei Zhu; Kaiming Kuang; Li Zhang; Hongwei Bran Li; Mingchen Gao; Jiancheng Yang
>
> **备注:** In revision process
>
> **摘要:** High-quality 3D reconstruction of pulmonary segments plays a crucial role in segmentectomy and surgical treatment planning for lung cancer. Due to the resolution requirement of the target reconstruction, conventional deep learning-based methods often suffer from computational resource constraints or limited granularity. Conversely, implicit modeling is favored due to its computational efficiency and continuous representation at any resolution. We propose a neural implicit function-based method to learn a 3D surface to achieve anatomy-aware, precise pulmonary segment reconstruction, represented as a shape by deforming a learnable template. Additionally, we introduce two clinically relevant evaluation metrics to assess the reconstruction comprehensively. Further, due to the absence of publicly available shape datasets to benchmark reconstruction algorithms, we developed a shape dataset named Lung3D, including the 3D models of 800 labeled pulmonary segments and the corresponding airways, arteries, veins, and intersegmental veins. We demonstrate that the proposed approach outperforms existing methods, providing a new perspective for pulmonary segment reconstruction. Code and data will be available at https://github.com/M3DV/ImPulSe.
>
---
#### [new 079] In-Context Learning for Label-Efficient Cancer Image Classification in Oncology
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文研究肿瘤图像分类任务，旨在解决AI依赖大量标注数据及需重复训练的局限性。通过上下文学习，利用少量样本使视觉语言模型（如GPT-4o）在推理时适应新任务，无需重新训练。实验表明，模型在少样本场景下性能显著提升，验证了该方法在资源受限医疗场景的可行性。**

- **链接: [http://arxiv.org/pdf/2505.08798v1](http://arxiv.org/pdf/2505.08798v1)**

> **作者:** Mobina Shrestha; Bishwas Mandal; Vishal Mandal; Asis Shrestha
>
> **摘要:** The application of AI in oncology has been limited by its reliance on large, annotated datasets and the need for retraining models for domain-specific diagnostic tasks. Taking heed of these limitations, we investigated in-context learning as a pragmatic alternative to model retraining by allowing models to adapt to new diagnostic tasks using only a few labeled examples at inference, without the need for retraining. Using four vision-language models (VLMs)-Paligemma, CLIP, ALIGN and GPT-4o, we evaluated the performance across three oncology datasets: MHIST, PatchCamelyon and HAM10000. To the best of our knowledge, this is the first study to compare the performance of multiple VLMs on different oncology classification tasks. Without any parameter updates, all models showed significant gains with few-shot prompting, with GPT-4o reaching an F1 score of 0.81 in binary classification and 0.60 in multi-class classification settings. While these results remain below the ceiling of fully fine-tuned systems, they highlight the potential of ICL to approximate task-specific behavior using only a handful of examples, reflecting how clinicians often reason from prior cases. Notably, open-source models like Paligemma and CLIP demonstrated competitive gains despite their smaller size, suggesting feasibility for deployment in computing constrained clinical environments. Overall, these findings highlight the potential of ICL as a practical solution in oncology, particularly for rare cancers and resource-limited contexts where fine-tuning is infeasible and annotated data is difficult to obtain.
>
---
#### [new 080] DCSNet: A Lightweight Knowledge Distillation-Based Model with Explainable AI for Lung Cancer Diagnosis from Histopathological Images
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分析任务，针对肺癌诊断中深度学习模型计算量大、缺乏透明度的问题，提出基于知识蒸馏的轻量级模型DCSNet。通过将ResNet50等复杂教师模型知识迁移到学生模型，并融合可解释AI技术，实现在资源受限环境下的高效透明诊断。**

- **链接: [http://arxiv.org/pdf/2505.09334v1](http://arxiv.org/pdf/2505.09334v1)**

> **作者:** Sadman Sakib Alif; Nasim Anzum Promise; Fiaz Al Abid; Aniqua Nusrat Zereen
>
> **摘要:** Lung cancer is a leading cause of cancer-related deaths globally, where early detection and accurate diagnosis are critical for improving survival rates. While deep learning, particularly convolutional neural networks (CNNs), has revolutionized medical image analysis by detecting subtle patterns indicative of early-stage lung cancer, its adoption faces challenges. These models are often computationally expensive and require significant resources, making them unsuitable for resource constrained environments. Additionally, their lack of transparency hinders trust and broader adoption in sensitive fields like healthcare. Knowledge distillation addresses these challenges by transferring knowledge from large, complex models (teachers) to smaller, lightweight models (students). We propose a knowledge distillation-based approach for lung cancer detection, incorporating explainable AI (XAI) techniques to enhance model transparency. Eight CNNs, including ResNet50, EfficientNetB0, EfficientNetB3, and VGG16, are evaluated as teacher models. We developed and trained a lightweight student model, Distilled Custom Student Network (DCSNet) using ResNet50 as the teacher. This approach not only ensures high diagnostic performance in resource-constrained settings but also addresses transparency concerns, facilitating the adoption of AI-driven diagnostic tools in healthcare.
>
---
#### [new 081] RT-cache: Efficient Robot Trajectory Retrieval System
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出RT-cache，属于机器人轨迹优化任务，旨在解决视觉-语言-动作模型推理延迟高的问题。通过构建大规模轨迹记忆库并检索复用多步运动片段，降低计算成本。结合记忆构建与检索模块，实现高效数据匹配，提升实时操控效率，实验验证其在真实场景中加速任务执行的效果。**

- **链接: [http://arxiv.org/pdf/2505.09040v1](http://arxiv.org/pdf/2505.09040v1)**

> **作者:** Owen Kwon; Abraham George; Alison Bartsch; Amir Barati Farimani
>
> **备注:** 9 pages, 5 figures. Submitted to an IEEE robotics conference
>
> **摘要:** This paper introduces RT-cache, a novel trajectorymemory pipeline that accelerates real-world robot inference by leveraging big-data retrieval and learning from experience. While modern Vision-Language-Action (VLA) models can handle diverse robotic tasks, they often incur high per-step inference costs, resulting in significant latency, sometimes minutes per task. In contrast, RT-cache stores a large-scale Memory of previously successful robot trajectories and retrieves relevant multistep motion snippets, drastically reducing inference overhead. By integrating a Memory Builder with a Trajectory Retrieval, we develop an efficient retrieval process that remains tractable even for extremely large datasets. RT-cache flexibly accumulates real-world experiences and replays them whenever the current scene matches past states, adapting quickly to new or unseen environments with only a few additional samples. Experiments on the Open-X Embodiment Dataset and other real-world data demonstrate that RT-cache completes tasks both faster and more successfully than a baseline lacking retrieval, suggesting a practical, data-driven solution for real-time manipulation.
>
---
#### [new 082] Spec2VolCAMU-Net: A Spectrogram-to-Volume Model for EEG-to-fMRI Reconstruction based on Multi-directional Time-Frequency Convolutional Attention Encoder and Vision-Mamba U-Net
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于EEG到fMRI的跨模态重建任务，旨在解决现有方法忽略跨通道时频特征、计算效率低的问题。通过多向时频卷积注意力编码器提取EEG频谱特征，结合Vision-Mamba U-Net解码器进行高效空间建模，以轻量结构实现高精度三维fMRI重建，在多个数据集上刷新了SSIM与PSNR指标，适用于实时场景。**

- **链接: [http://arxiv.org/pdf/2505.09521v1](http://arxiv.org/pdf/2505.09521v1)**

> **作者:** Dongyi He; Shiyang Li; Bin Jiang; He Yan
>
> **摘要:** High-resolution functional magnetic resonance imaging (fMRI) is essential for mapping human brain activity; however, it remains costly and logistically challenging. If comparable volumes could be generated directly from widely available scalp electroencephalography (EEG), advanced neuroimaging would become significantly more accessible. Existing EEG-to-fMRI generators rely on plain CNNs that fail to capture cross-channel time-frequency cues or on heavy transformer/GAN decoders that strain memory and stability. We propose Spec2VolCAMU-Net, a lightweight spectrogram-to-volume generator that confronts these issues via a Multi-directional Time-Frequency Convolutional Attention Encoder, stacking temporal, spectral and joint convolutions with self-attention, and a Vision-Mamba U-Net decoder whose linear-time state-space blocks enable efficient long-range spatial modelling. Trained end-to-end with a hybrid SSI-MSE loss, Spec2VolCAMU-Net achieves state-of-the-art fidelity on three public benchmarks, recording SSIMs of 0.693 on NODDI, 0.725 on Oddball and 0.788 on CN-EPFL, representing improvements of 14.5%, 14.9%, and 16.9% respectively over previous best SSIM scores. Furthermore, it achieves competitive PSNR scores, particularly excelling on the CN-EPFL dataset with a 4.6% improvement over the previous best PSNR, thus striking a better balance in reconstruction quality. The proposed model is lightweight and efficient, making it suitable for real-time applications in clinical and research settings. The code is available at https://github.com/hdy6438/Spec2VolCAMU-Net.
>
---
#### [new 083] IntrinsicEdit: Precise generative image manipulation in intrinsic space
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于生成式图像编辑领域，旨在解决现有扩散模型控制精度不足、任务单一的问题。通过构建基于RGB-X扩散框架的固有图像潜在空间，提出解耦通道操作与精准扩散反转方法，在保持物体身份的同时，实现像素级精确的多任务图像编辑（调色、换物、全局光照等），无需额外数据或模型微调。**

- **链接: [http://arxiv.org/pdf/2505.08889v1](http://arxiv.org/pdf/2505.08889v1)**

> **作者:** Linjie Lyu; Valentin Deschaintre; Yannick Hold-Geoffroy; Miloš Hašan; Jae Shin Yoon; Thomas Leimkühler; Christian Theobalt; Iliyan Georgiev
>
> **备注:** SIGGRAPH 2025 Journal track
>
> **摘要:** Generative diffusion models have advanced image editing with high-quality results and intuitive interfaces such as prompts and semantic drawing. However, these interfaces lack precise control, and the associated methods typically specialize on a single editing task. We introduce a versatile, generative workflow that operates in an intrinsic-image latent space, enabling semantic, local manipulation with pixel precision for a range of editing operations. Building atop the RGB-X diffusion framework, we address key challenges of identity preservation and intrinsic-channel entanglement. By incorporating exact diffusion inversion and disentangled channel manipulation, we enable precise, efficient editing with automatic resolution of global illumination effects -- all without additional data collection or model fine-tuning. We demonstrate state-of-the-art performance across a variety of tasks on complex images, including color and texture adjustments, object insertion and removal, global relighting, and their combinations.
>
---
#### [new 084] Parameter-Efficient Fine-Tuning of Vision Foundation Model for Forest Floor Segmentation from UAV Imagery
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于无人机图像分割任务，旨在解决森林地面复杂场景下物体分割困难的问题。通过参数高效微调（PEFT）方法改进Segment Anything Model（SAM），仅调整少量参数使其适应树桩、植被等目标分割。采用适配器结构达到最高精度，同时提出低秩适配（LoRA）作为轻量化替代方案。**

- **链接: [http://arxiv.org/pdf/2505.08932v1](http://arxiv.org/pdf/2505.08932v1)**

> **作者:** Mohammad Wasil; Ahmad Drak; Brennan Penfold; Ludovico Scarton; Maximilian Johenneken; Alexander Asteroth; Sebastian Houben
>
> **备注:** Accepted to the Novel Approaches for Precision Agriculture and Forestry with Autonomous Robots IEEE ICRA Workshop - 2025
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are increasingly used for reforestation and forest monitoring, including seed dispersal in hard-to-reach terrains. However, a detailed understanding of the forest floor remains a challenge due to high natural variability, quickly changing environmental parameters, and ambiguous annotations due to unclear definitions. To address this issue, we adapt the Segment Anything Model (SAM), a vision foundation model with strong generalization capabilities, to segment forest floor objects such as tree stumps, vegetation, and woody debris. To this end, we employ parameter-efficient fine-tuning (PEFT) to fine-tune a small subset of additional model parameters while keeping the original weights fixed. We adjust SAM's mask decoder to generate masks corresponding to our dataset categories, allowing for automatic segmentation without manual prompting. Our results show that the adapter-based PEFT method achieves the highest mean intersection over union (mIoU), while Low-rank Adaptation (LoRA), with fewer parameters, offers a lightweight alternative for resource-constrained UAV platforms.
>
---
#### [new 085] TransDiffuser: End-to-end Trajectory Generation with Decorrelated Multi-modal Representation for Autonomous Driving
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文研究自动驾驶轨迹规划任务，解决生成多样化高质量轨迹时的模式崩溃问题。提出TransDiffuser模型，通过编码器-解码器架构结合扩散模型，利用场景信息作为多模态条件输入，并设计表征解耦优化机制提升生成效果。在NAVSIM基准上取得94.85 PDMS，超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.09315v1](http://arxiv.org/pdf/2505.09315v1)**

> **作者:** Xuefeng Jiang; Yuan Ma; Pengxiang Li; Leimeng Xu; Xin Wen; Kun Zhan; Zhongpu Xia; Peng Jia; XianPeng Lang; Sheng Sun
>
> **备注:** Under review
>
> **摘要:** In recent years, diffusion model has shown its potential across diverse domains from vision generation to language modeling. Transferring its capabilities to modern autonomous driving systems has also emerged as a promising direction.In this work, we propose TransDiffuser, an encoder-decoder based generative trajectory planning model for end-to-end autonomous driving. The encoded scene information serves as the multi-modal conditional input of the denoising decoder. To tackle the mode collapse dilemma in generating high-quality diverse trajectories, we introduce a simple yet effective multi-modal representation decorrelation optimization mechanism during the training process.TransDiffuser achieves PDMS of 94.85 on the NAVSIM benchmark, surpassing previous state-of-the-art methods without any anchor-based prior trajectories.
>
---
#### [new 086] Neural BRDF Importance Sampling by Reparameterization
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于计算机图形学中的渲染优化任务，旨在解决神经双向反射分布函数（BRDF）重要性采样效率低、灵活性差的问题。提出基于参数重构的采样方法，替代传统可逆网络和多步推理，将分布学习转化为积分替换问题，实现了低方差、高速度的BRDF样本生成。**

- **链接: [http://arxiv.org/pdf/2505.08998v1](http://arxiv.org/pdf/2505.08998v1)**

> **作者:** Liwen Wu; Sai Bi; Zexiang Xu; Hao Tan; Kai Zhang; Fujun Luan; Haolin Lu; Ravi Ramamoorthi
>
> **摘要:** Neural bidirectional reflectance distribution functions (BRDFs) have emerged as popular material representations for enhancing realism in physically-based rendering. Yet their importance sampling remains a significant challenge. In this paper, we introduce a reparameterization-based formulation of neural BRDF importance sampling that seamlessly integrates into the standard rendering pipeline with precise generation of BRDF samples. The reparameterization-based formulation transfers the distribution learning task to a problem of identifying BRDF integral substitutions. In contrast to previous methods that rely on invertible networks and multi-step inference to reconstruct BRDF distributions, our model removes these constraints, which offers greater flexibility and efficiency. Our variance and performance analysis demonstrates that our reparameterization method achieves the best variance reduction in neural BRDF renderings while maintaining high inference speeds compared to existing baselines.
>
---
#### [new 087] Ultrasound Report Generation with Multimodal Large Language Models for Standardized Texts
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学超声报告自动生成任务，旨在解决超声图像多变、依赖操作者及标准化文本生成难题。提出多器官、多语言统一框架，通过碎片化双语训练和视觉模型微调对齐图文，提升生成准确性和跨器官/语言一致性，指标优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.08838v1](http://arxiv.org/pdf/2505.08838v1)**

> **作者:** Peixuan Ge; Tongkun Su; Faqin Lv; Baoliang Zhao; Peng Zhang; Chi Hong Wong; Liang Yao; Yu Sun; Zenan Wang; Pak Kin Wong; Ying Hu
>
> **摘要:** Ultrasound (US) report generation is a challenging task due to the variability of US images, operator dependence, and the need for standardized text. Unlike X-ray and CT, US imaging lacks consistent datasets, making automation difficult. In this study, we propose a unified framework for multi-organ and multilingual US report generation, integrating fragment-based multilingual training and leveraging the standardized nature of US reports. By aligning modular text fragments with diverse imaging data and curating a bilingual English-Chinese dataset, the method achieves consistent and clinically accurate text generation across organ sites and languages. Fine-tuning with selective unfreezing of the vision transformer (ViT) further improves text-image alignment. Compared to the previous state-of-the-art KMVE method, our approach achieves relative gains of about 2\% in BLEU scores, approximately 3\% in ROUGE-L, and about 15\% in CIDEr, while significantly reducing errors such as missing or incorrect content. By unifying multi-organ and multi-language report generation into a single, scalable framework, this work demonstrates strong potential for real-world clinical workflows.
>
---
#### [new 088] Thoughts on Objectives of Sparse and Hierarchical Masked Image Model
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于自监督学习任务，研究图像预训练中掩码模式对性能的影响。针对SparK模型，提出新型Mesh Mask模式，分析其提升效果，探索稀疏分层掩码设计对模型优化的作用。**

- **链接: [http://arxiv.org/pdf/2505.08819v1](http://arxiv.org/pdf/2505.08819v1)**

> **作者:** Asahi Miyazaki; Tsuyoshi Okita
>
> **备注:** 9 pages, 11 figures
>
> **摘要:** Masked image modeling is one of the most poplular objectives of training. Recently, the SparK model has been proposed with superior performance among self-supervised learning models. This paper proposes a new mask pattern for this SparK model, proposing it as the Mesh Mask-ed SparK model. We report the effect of the mask pattern used for image masking in pre-training on performance.
>
---
#### [new 089] DPN-GAN: Inducing Periodic Activations in Generative Adversarial Networks for High-Fidelity Audio Synthesis
- **分类: cs.SD; cs.AI; cs.CV; cs.LG; eess.AS**

- **简介: 该论文提出DPN-GAN，用于高保真音频合成任务，解决传统GAN依赖梅尔谱导致分辨率低、模式崩溃的问题。通过周期性ReLU激活函数引入音频周期特性，设计可变形卷积模块实现多分辨率生成，并改进判别器结构。实验证明其音频质量优于现有模型，且具备鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.09091v1](http://arxiv.org/pdf/2505.09091v1)**

> **作者:** Zeeshan Ahmad; Shudi Bao; Meng Chen
>
> **摘要:** In recent years, generative adversarial networks (GANs) have made significant progress in generating audio sequences. However, these models typically rely on bandwidth-limited mel-spectrograms, which constrain the resolution of generated audio sequences, and lead to mode collapse during conditional generation. To address this issue, we propose Deformable Periodic Network based GAN (DPN-GAN), a novel GAN architecture that incorporates a kernel-based periodic ReLU activation function to induce periodic bias in audio generation. This innovative approach enhances the model's ability to capture and reproduce intricate audio patterns. In particular, our proposed model features a DPN module for multi-resolution generation utilizing deformable convolution operations, allowing for adaptive receptive fields that improve the quality and fidelity of the synthetic audio. Additionally, we enhance the discriminator network using deformable convolution to better distinguish between real and generated samples, further refining the audio quality. We trained two versions of the model: DPN-GAN small (38.67M parameters) and DPN-GAN large (124M parameters). For evaluation, we use five different datasets, covering both speech synthesis and music generation tasks, to demonstrate the efficiency of the DPN-GAN. The experimental results demonstrate that DPN-GAN delivers superior performance on both out-of-distribution and noisy data, showcasing its robustness and adaptability. Trained across various datasets, DPN-GAN outperforms state-of-the-art GAN architectures on standard evaluation metrics, and exhibits increased robustness in synthesized audio.
>
---
#### [new 090] Adaptive Security Policy Management in Cloud Environments Using Reinforcement Learning
- **分类: cs.CR; cs.CV; cs.DC; cs.LG; cs.NI**

- **简介: 该论文属于云环境动态安全策略管理任务，旨在解决静态策略难以应对威胁演变和资源弹性的问题。提出基于强化学习的框架，采用深度Q网络和近端策略优化算法，通过云遥测数据动态调整防火墙、IAM等策略。实验显示其检测率提升至92%（静态82%），响应时间减少58%，兼顾合规与资源效率。**

- **链接: [http://arxiv.org/pdf/2505.08837v1](http://arxiv.org/pdf/2505.08837v1)**

> **作者:** Muhammad Saqib; Dipkumar Mehta; Fnu Yashu; Shubham Malhotra
>
> **备注:** 10 pages, 6 figures, 1 table
>
> **摘要:** The security of cloud environments, such as Amazon Web Services (AWS), is complex and dynamic. Static security policies have become inadequate as threats evolve and cloud resources exhibit elasticity [1]. This paper addresses the limitations of static policies by proposing a security policy management framework that uses reinforcement learning (RL) to adapt dynamically. Specifically, we employ deep reinforcement learning algorithms, including deep Q Networks and proximal policy optimization, enabling the learning and continuous adjustment of controls such as firewall rules and Identity and Access Management (IAM) policies. The proposed RL based solution leverages cloud telemetry data (AWS Cloud Trail logs, network traffic data, threat intelligence feeds) to continuously refine security policies, maximizing threat mitigation, and compliance while minimizing resource impact. Experimental results demonstrate that our adaptive RL based framework significantly outperforms static policies, achieving higher intrusion detection rates (92% compared to 82% for static policies) and substantially reducing incident detection and response times by 58%. In addition, it maintains high conformity with security requirements and efficient resource usage. These findings validate the effectiveness of adaptive reinforcement learning approaches in improving cloud security policy management.
>
---
#### [new 091] UMotion: Uncertainty-driven Human Motion Estimation from Inertial and Ultra-wideband Units
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 该论文属于3D人体运动估计任务，旨在解决惯性传感器(IMU)的漂移、姿态模糊及超宽带(UWB)遮挡问题。研究者提出UMotion框架，通过紧耦合无迹卡尔曼滤波(UKF)实时融合IMU与UWB数据，结合人体测量学约束，优化人体形态与姿态估计，提升运动追踪精度。**

- **链接: [http://arxiv.org/pdf/2505.09393v1](http://arxiv.org/pdf/2505.09393v1)**

> **作者:** Huakun Liu; Hiroki Ota; Xin Wei; Yutaro Hirao; Monica Perusquia-Hernandez; Hideaki Uchiyama; Kiyoshi Kiyokawa
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** Sparse wearable inertial measurement units (IMUs) have gained popularity for estimating 3D human motion. However, challenges such as pose ambiguity, data drift, and limited adaptability to diverse bodies persist. To address these issues, we propose UMotion, an uncertainty-driven, online fusing-all state estimation framework for 3D human shape and pose estimation, supported by six integrated, body-worn ultra-wideband (UWB) distance sensors with IMUs. UWB sensors measure inter-node distances to infer spatial relationships, aiding in resolving pose ambiguities and body shape variations when combined with anthropometric data. Unfortunately, IMUs are prone to drift, and UWB sensors are affected by body occlusions. Consequently, we develop a tightly coupled Unscented Kalman Filter (UKF) framework that fuses uncertainties from sensor data and estimated human motion based on individual body shape. The UKF iteratively refines IMU and UWB measurements by aligning them with uncertain human motion constraints in real-time, producing optimal estimates for each. Experiments on both synthetic and real-world datasets demonstrate the effectiveness of UMotion in stabilizing sensor data and the improvement over state of the art in pose accuracy.
>
---
#### [new 092] EDBench: Large-Scale Electron Density Data for Molecular Modeling
- **分类: physics.chem-ph; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于分子机器学习力场的数据集构建任务，旨在解决电子密度（ED）数据因计算成本高而稀缺的问题。作者构建了EDBench——包含330万分子的ED数据集，并设计了预测、检索、生成等基准任务，验证了基于学习的ED计算方法在保持精度的同时显著降低计算成本，为电子尺度的分子建模提供基础。**

- **链接: [http://arxiv.org/pdf/2505.09262v1](http://arxiv.org/pdf/2505.09262v1)**

> **作者:** Hongxin Xiang; Ke Li; Mingquan Liu; Zhixiang Cheng; Bin Yao; Wenjie Du; Jun Xia; Li Zeng; Xin Jin; Xiangxiang Zeng
>
> **摘要:** Existing molecular machine learning force fields (MLFFs) generally focus on the learning of atoms, molecules, and simple quantum chemical properties (such as energy and force), but ignore the importance of electron density (ED) $\rho(r)$ in accurately understanding molecular force fields (MFFs). ED describes the probability of finding electrons at specific locations around atoms or molecules, which uniquely determines all ground state properties (such as energy, molecular structure, etc.) of interactive multi-particle systems according to the Hohenberg-Kohn theorem. However, the calculation of ED relies on the time-consuming first-principles density functional theory (DFT) which leads to the lack of large-scale ED data and limits its application in MLFFs. In this paper, we introduce EDBench, a large-scale, high-quality dataset of ED designed to advance learning-based research at the electronic scale. Built upon the PCQM4Mv2, EDBench provides accurate ED data, covering 3.3 million molecules. To comprehensively evaluate the ability of models to understand and utilize electronic information, we design a suite of ED-centric benchmark tasks spanning prediction, retrieval, and generation. Our evaluation on several state-of-the-art methods demonstrates that learning from EDBench is not only feasible but also achieves high accuracy. Moreover, we show that learning-based method can efficiently calculate ED with comparable precision while significantly reducing the computational cost relative to traditional DFT calculations. All data and benchmarks from EDBench will be freely available, laying a robust foundation for ED-driven drug discovery and materials science.
>
---
#### [new 093] Multi-step manipulation task and motion planning guided by video demonstration
- **分类: cs.RO; cs.CV; cs.SY; eess.SY**

- **简介: 该论文属于机器人任务与运动规划领域，旨在解决复杂多步骤操作中顺序依赖及视频指导泛化问题。通过扩展RRT算法，结合视频提取的物体位姿与接触状态，设计支持时序约束的规划框架，并开发轨迹优化方法实现真实机器人部署。实验基于新构建的跨平台多任务基准验证有效性。**

- **链接: [http://arxiv.org/pdf/2505.08949v1](http://arxiv.org/pdf/2505.08949v1)**

> **作者:** Kateryna Zorina; David Kovar; Mederic Fourmy; Florent Lamiraux; Nicolas Mansard; Justin Carpentier; Josef Sivic; Vladimir Petrik
>
> **摘要:** This work aims to leverage instructional video to solve complex multi-step task-and-motion planning tasks in robotics. Towards this goal, we propose an extension of the well-established Rapidly-Exploring Random Tree (RRT) planner, which simultaneously grows multiple trees around grasp and release states extracted from the guiding video. Our key novelty lies in combining contact states and 3D object poses extracted from the guiding video with a traditional planning algorithm that allows us to solve tasks with sequential dependencies, for example, if an object needs to be placed at a specific location to be grasped later. We also investigate the generalization capabilities of our approach to go beyond the scene depicted in the instructional video. To demonstrate the benefits of the proposed video-guided planning approach, we design a new benchmark with three challenging tasks: (I) 3D re-arrangement of multiple objects between a table and a shelf, (ii) multi-step transfer of an object through a tunnel, and (iii) transferring objects using a tray similar to a waiter transfers dishes. We demonstrate the effectiveness of our planning algorithm on several robots, including the Franka Emika Panda and the KUKA KMR iiwa. For a seamless transfer of the obtained plans to the real robot, we develop a trajectory refinement approach formulated as an optimal control problem (OCP).
>
---
## 更新

#### [replaced 001] Comparing Quantum Annealing and Spiking Neuromorphic Computing for Sampling Binary Sparse Coding QUBO Problems
- **分类: cs.ET; cs.CV; cs.DM; cs.NE; quant-ph**

- **链接: [http://arxiv.org/pdf/2405.20525v2](http://arxiv.org/pdf/2405.20525v2)**

> **作者:** Kyle Henke; Elijah Pelofske; Garrett Kenyon; Georg Hahn
>
> **摘要:** We consider the problem of computing a sparse binary representation of an image. To be precise, given an image and an overcomplete, non-orthonormal basis, we aim to find a sparse binary vector indicating the minimal set of basis vectors that when added together best reconstruct the given input. We formulate this problem with an $L_2$ loss on the reconstruction error, and an $L_0$ (or, equivalently, an $L_1$) loss on the binary vector enforcing sparsity. This yields a quadratic unconstrained binary optimization problem (QUBO), whose optimal solution(s) in general is NP-hard to find. The contribution of this work is twofold. First, we solve the sparse representation QUBOs by solving them both on a D-Wave quantum annealer with Pegasus chip connectivity via minor embedding, as well as on the Intel Loihi 2 spiking neuromorphic processor using a stochastic Non-equilibrium Boltzmann Machine (NEBM). Second, we deploy Quantum Evolution Monte Carlo with Reverse Annealing and iterated warm starting on Loihi 2 to evolve the solution quality from the respective machines. The solutions are benchmarked against simulated annealing, a classical heuristic, and the optimal solutions are computed using CPLEX. Iterated reverse quantum annealing performs similarly to simulated annealing, although simulated annealing is always able to sample the optimal solution whereas quantum annealing was not always able to. The Loihi 2 solutions that are sampled are on average more sparse than the solutions from any of the other methods. We demonstrate that both quantum annealing and neuromorphic computing are suitable for binary sparse coding QUBOs, and that Loihi 2 outperforms a D-Wave quantum annealer standard linear-schedule anneal, while iterated reverse quantum annealing performs much better than both unmodified linear-schedule quantum annealing and iterated warm starting on Loihi 2.
>
---
#### [replaced 002] 3D Cartoon Face Generation with Controllable Expressions from a Single GAN Image
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2207.14425v2](http://arxiv.org/pdf/2207.14425v2)**

> **作者:** Hao Wang; Wenhao Shen; Guosheng Lin; Steven C. H. Hoi; Chunyan Miao
>
> **备注:** IJCNN 2025. Code: https://github.com/hwang1996/3D-Cartoon-Face-Generation
>
> **摘要:** In this paper, we investigate an open research task of generating 3D cartoon face shapes from single 2D GAN generated human faces and without 3D supervision, where we can also manipulate the facial expressions of the 3D shapes. To this end, we discover the semantic meanings of StyleGAN latent space, such that we are able to produce face images of various expressions, poses, and lighting conditions by controlling the latent codes. Specifically, we first finetune the pretrained StyleGAN face model on the cartoon datasets. By feeding the same latent codes to face and cartoon generation models, we aim to realize the translation from 2D human face images to cartoon styled avatars. We then discover semantic directions of the GAN latent space, in an attempt to change the facial expressions while preserving the original identity. As we do not have any 3D annotations for cartoon faces, we manipulate the latent codes to generate images with different poses and lighting conditions, such that we can reconstruct the 3D cartoon face shapes. We validate the efficacy of our method on three cartoon datasets qualitatively and quantitatively.
>
---
#### [replaced 003] Optimal-state Dynamics Estimation for Physics-based Human Motion Capture from Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.07795v4](http://arxiv.org/pdf/2410.07795v4)**

> **作者:** Cuong Le; Viktor Johansson; Manon Kok; Bastian Wandt
>
> **备注:** 17 pages, 7 figure, NeurIPS 2024
>
> **摘要:** Human motion capture from monocular videos has made significant progress in recent years. However, modern approaches often produce temporal artifacts, e.g. in form of jittery motion and struggle to achieve smooth and physically plausible motions. Explicitly integrating physics, in form of internal forces and exterior torques, helps alleviating these artifacts. Current state-of-the-art approaches make use of an automatic PD controller to predict torques and reaction forces in order to re-simulate the input kinematics, i.e. the joint angles of a predefined skeleton. However, due to imperfect physical models, these methods often require simplifying assumptions and extensive preprocessing of the input kinematics to achieve good performance. To this end, we propose a novel method to selectively incorporate the physics models with the kinematics observations in an online setting, inspired by a neural Kalman-filtering approach. We develop a control loop as a meta-PD controller to predict internal joint torques and external reaction forces, followed by a physics-based motion simulation. A recurrent neural network is introduced to realize a Kalman filter that attentively balances the kinematics input and simulated motion, resulting in an optimal-state dynamics prediction. We show that this filtering step is crucial to provide an online supervision that helps balancing the shortcoming of the respective input motions, thus being important for not only capturing accurate global motion trajectories but also producing physically plausible human poses. The proposed approach excels in the physics-based human pose estimation task and demonstrates the physical plausibility of the predictive dynamics, compared to state of the art. The code is available on https://github.com/cuongle1206/OSDCap
>
---
#### [replaced 004] Monocular Online Reconstruction with Enhanced Detail Preservation
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.07887v2](http://arxiv.org/pdf/2505.07887v2)**

> **作者:** Songyin Wu; Zhaoyang Lv; Yufeng Zhu; Duncan Frost; Zhengqin Li; Ling-Qi Yan; Carl Ren; Richard Newcombe; Zhao Dong
>
> **备注:** Accepted to SIGGRAPH 2025 (Conference Track). Project page: https://poiw.github.io/MODP
>
> **摘要:** We propose an online 3D Gaussian-based dense mapping framework for photorealistic details reconstruction from a monocular image stream. Our approach addresses two key challenges in monocular online reconstruction: distributing Gaussians without relying on depth maps and ensuring both local and global consistency in the reconstructed maps. To achieve this, we introduce two key modules: the Hierarchical Gaussian Management Module for effective Gaussian distribution and the Global Consistency Optimization Module for maintaining alignment and coherence at all scales. In addition, we present the Multi-level Occupancy Hash Voxels (MOHV), a structure that regularizes Gaussians for capturing details across multiple levels of granularity. MOHV ensures accurate reconstruction of both fine and coarse geometries and textures, preserving intricate details while maintaining overall structural integrity. Compared to state-of-the-art RGB-only and even RGB-D methods, our framework achieves superior reconstruction quality with high computational efficiency. Moreover, it integrates seamlessly with various tracking systems, ensuring generality and scalability.
>
---
#### [replaced 005] Bayesian computation with generative diffusion models by Multilevel Monte Carlo
- **分类: stat.CO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.15511v4](http://arxiv.org/pdf/2409.15511v4)**

> **作者:** Abdul-Lateef Haji-Ali; Marcelo Pereyra; Luke Shaw; Konstantinos Zygalakis
>
> **备注:** 13 images
>
> **摘要:** Generative diffusion models have recently emerged as a powerful strategy to perform stochastic sampling in Bayesian inverse problems, delivering remarkably accurate solutions for a wide range of challenging applications. However, diffusion models often require a large number of neural function evaluations per sample in order to deliver accurate posterior samples. As a result, using diffusion models as stochastic samplers for Monte Carlo integration in Bayesian computation can be highly computationally expensive, particularly in applications that require a substantial number of Monte Carlo samples for conducting uncertainty quantification analyses. This cost is especially high in large-scale inverse problems such as computational imaging, which rely on large neural networks that are expensive to evaluate. With quantitative imaging applications in mind, this paper presents a Multilevel Monte Carlo strategy that significantly reduces the cost of Bayesian computation with diffusion models. This is achieved by exploiting cost-accuracy trade-offs inherent to diffusion models to carefully couple models of different levels of accuracy in a manner that significantly reduces the overall cost of the calculation, without reducing the final accuracy. The proposed approach achieves a $4\times$-to-$8\times$ reduction in computational cost w.r.t. standard techniques across three benchmark imaging problems.
>
---
#### [replaced 006] GarmentGS: Point-Cloud Guided Gaussian Splatting for High-Fidelity Non-Watertight 3D Garment Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.02126v2](http://arxiv.org/pdf/2505.02126v2)**

> **作者:** Zhihao Tang; Shenghao Yang; Hongtao Zhang; Mingbo Zhao
>
> **备注:** Accepted by ICMR 2025
>
> **摘要:** Traditional 3D garment creation requires extensive manual operations, resulting in time and labor costs. Recently, 3D Gaussian Splatting has achieved breakthrough progress in 3D scene reconstruction and rendering, attracting widespread attention and opening new pathways for 3D garment reconstruction. However, due to the unstructured and irregular nature of Gaussian primitives, it is difficult to reconstruct high-fidelity, non-watertight 3D garments. In this paper, we present GarmentGS, a dense point cloud-guided method that can reconstruct high-fidelity garment surfaces with high geometric accuracy and generate non-watertight, single-layer meshes. Our method introduces a fast dense point cloud reconstruction module that can complete garment point cloud reconstruction in 10 minutes, compared to traditional methods that require several hours. Furthermore, we use dense point clouds to guide the movement, flattening, and rotation of Gaussian primitives, enabling better distribution on the garment surface to achieve superior rendering effects and geometric accuracy. Through numerical and visual comparisons, our method achieves fast training and real-time rendering while maintaining competitive quality.
>
---
#### [replaced 007] Leveraging Segment Anything Model for Source-Free Domain Adaptation via Dual Feature Guided Auto-Prompting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08527v2](http://arxiv.org/pdf/2505.08527v2)**

> **作者:** Zheang Huai; Hui Tang; Yi Li; Zhuangzhuang Chen; Xiaomeng Li
>
> **摘要:** Source-free domain adaptation (SFDA) for segmentation aims at adapting a model trained in the source domain to perform well in the target domain with only the source model and unlabeled target data.Inspired by the recent success of Segment Anything Model (SAM) which exhibits the generality of segmenting images of various modalities and in different domains given human-annotated prompts like bounding boxes or points, we for the first time explore the potentials of Segment Anything Model for SFDA via automatedly finding an accurate bounding box prompt. We find that the bounding boxes directly generated with existing SFDA approaches are defective due to the domain gap.To tackle this issue, we propose a novel Dual Feature Guided (DFG) auto-prompting approach to search for the box prompt. Specifically, the source model is first trained in a feature aggregation phase, which not only preliminarily adapts the source model to the target domain but also builds a feature distribution well-prepared for box prompt search. In the second phase, based on two feature distribution observations, we gradually expand the box prompt with the guidance of the target model feature and the SAM feature to handle the class-wise clustered target features and the class-wise dispersed target features, respectively. To remove the potentially enlarged false positive regions caused by the over-confident prediction of the target model, the refined pseudo-labels produced by SAM are further postprocessed based on connectivity analysis. Experiments on 3D and 2D datasets indicate that our approach yields superior performance compared to conventional methods. Code is available at https://github.com/xmed-lab/DFG.
>
---
#### [replaced 008] Neural Brain: A Neuroscience-inspired Framework for Embodied Agents
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.07634v2](http://arxiv.org/pdf/2505.07634v2)**

> **作者:** Jian Liu; Xiongtao Shi; Thai Duy Nguyen; Haitian Zhang; Tianxiang Zhang; Wei Sun; Yanjie Li; Athanasios V. Vasilakos; Giovanni Iacca; Arshad Ali Khan; Arvind Kumar; Jae Won Cho; Ajmal Mian; Lihua Xie; Erik Cambria; Lin Wang
>
> **备注:** 51 pages, 17 figures, 9 tables
>
> **摘要:** The rapid evolution of artificial intelligence (AI) has shifted from static, data-driven models to dynamic systems capable of perceiving and interacting with real-world environments. Despite advancements in pattern recognition and symbolic reasoning, current AI systems, such as large language models, remain disembodied, unable to physically engage with the world. This limitation has driven the rise of embodied AI, where autonomous agents, such as humanoid robots, must navigate and manipulate unstructured environments with human-like adaptability. At the core of this challenge lies the concept of Neural Brain, a central intelligence system designed to drive embodied agents with human-like adaptability. A Neural Brain must seamlessly integrate multimodal sensing and perception with cognitive capabilities. Achieving this also requires an adaptive memory system and energy-efficient hardware-software co-design, enabling real-time action in dynamic environments. This paper introduces a unified framework for the Neural Brain of embodied agents, addressing two fundamental challenges: (1) defining the core components of Neural Brain and (2) bridging the gap between static AI models and the dynamic adaptability required for real-world deployment. To this end, we propose a biologically inspired architecture that integrates multimodal active sensing, perception-cognition-action function, neuroplasticity-based memory storage and updating, and neuromorphic hardware/software optimization. Furthermore, we also review the latest research on embodied agents across these four aspects and analyze the gap between current AI systems and human intelligence. By synthesizing insights from neuroscience, we outline a roadmap towards the development of generalizable, autonomous agents capable of human-level intelligence in real-world scenarios.
>
---
#### [replaced 009] Gradient Attention Map Based Verification of Deep Convolutional Neural Networks with Application to X-ray Image Datasets
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.21227v2](http://arxiv.org/pdf/2504.21227v2)**

> **作者:** Omid Halimi Milani; Amanda Nikho; Lauren Mills; Marouane Tliba; Ahmet Enis Cetin; Mohammed H. Elnagar
>
> **备注:** 13 pages, 7 figures, accepted at IEEE VLSI Test Symposium (VTS) 2025
>
> **摘要:** Deep learning models have great potential in medical imaging, including orthodontics and skeletal maturity assessment. However, applying a model to data different from its training set can lead to unreliable predictions that may impact patient care. To address this, we propose a comprehensive verification framework that evaluates model suitability through multiple complementary strategies. First, we introduce a Gradient Attention Map (GAM)-based approach that analyzes attention patterns using Grad-CAM and compares them via similarity metrics such as IoU, Dice Similarity, SSIM, Cosine Similarity, Pearson Correlation, KL Divergence, and Wasserstein Distance. Second, we extend verification to early convolutional feature maps, capturing structural mis-alignments missed by attention alone. Finally, we incorporate an additional garbage class into the classification model to explicitly reject out-of-distribution inputs. Experimental results demonstrate that these combined methods effectively identify unsuitable models and inputs, promoting safer and more reliable deployment of deep learning in medical imaging.
>
---
#### [replaced 010] BOP-Distrib: Revisiting 6D Pose Estimation Benchmarks for Better Evaluation under Visual Ambiguities
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.17297v3](http://arxiv.org/pdf/2408.17297v3)**

> **作者:** Boris Meden; Asma Brazi; Fabrice Mayran de Chamisso; Steve Bourgeois; Vincent Lepetit
>
> **摘要:** 6D pose estimation aims at determining the object pose that best explains the camera observation. The unique solution for non-ambiguous objects can turn into a multi-modal pose distribution for symmetrical objects or when occlusions of symmetry-breaking elements happen, depending on the viewpoint. Currently, 6D pose estimation methods are benchmarked on datasets that consider, for their ground truth annotations, visual ambiguities as only related to global object symmetries, whereas they should be defined per-image to account for the camera viewpoint. We thus first propose an automatic method to re-annotate those datasets with a 6D pose distribution specific to each image, taking into account the object surface visibility in the image to correctly determine the visual ambiguities. Second, given this improved ground truth, we re-evaluate the state-of-the-art single pose methods and show that this greatly modifies the ranking of these methods. Third, as some recent works focus on estimating the complete set of solutions, we derive a precision/recall formulation to evaluate them against our image-wise distribution ground truth, making it the first benchmark for pose distribution methods on real images.
>
---
#### [replaced 011] Learning Traffic Anomalies from Generative Models on Real-Time Observations
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.01391v2](http://arxiv.org/pdf/2502.01391v2)**

> **作者:** Fotis I. Giasemis; Alexandros Sopasakis
>
> **摘要:** Accurate detection of traffic anomalies is crucial for effective urban traffic management and congestion mitigation. We use the Spatiotemporal Generative Adversarial Network (STGAN) framework combining Graph Neural Networks and Long Short-Term Memory networks to capture complex spatial and temporal dependencies in traffic data. We apply STGAN to real-time, minute-by-minute observations from 42 traffic cameras across Gothenburg, Sweden, collected over several months in 2020. The images are processed to compute a flow metric representing vehicle density, which serves as input for the model. Training is conducted on data from April to November 2020, and validation is performed on a separate dataset from November 14 to 23, 2020. Our results demonstrate that the model effectively detects traffic anomalies with high precision and low false positive rates. The detected anomalies include camera signal interruptions, visual artifacts, and extreme weather conditions affecting traffic flow.
>
---
#### [replaced 012] Embodied-Reasoner: Synergizing Visual Search, Reasoning, and Action for Embodied Interactive Tasks
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.21696v2](http://arxiv.org/pdf/2503.21696v2)**

> **作者:** Wenqi Zhang; Mengna Wang; Gangao Liu; Xu Huixin; Yiwei Jiang; Yongliang Shen; Guiyang Hou; Zhe Zheng; Hang Zhang; Xin Li; Weiming Lu; Peng Li; Yueting Zhuang
>
> **备注:** Code: https://github.com/zwq2018/embodied_reasoner Dataset: https://huggingface.co/datasets/zwq2018/embodied_reasoner
>
> **摘要:** Recent advances in deep thinking models have demonstrated remarkable reasoning capabilities on mathematical and coding tasks. However, their effectiveness in embodied domains which require continuous interaction with environments through image action interleaved trajectories remains largely -unexplored. We present Embodied Reasoner, a model that extends o1 style reasoning to interactive embodied search tasks. Unlike mathematical reasoning that relies primarily on logical deduction, embodied scenarios demand spatial understanding, temporal reasoning, and ongoing self-reflection based on interaction history. To address these challenges, we synthesize 9.3k coherent Observation-Thought-Action trajectories containing 64k interactive images and 90k diverse thinking processes (analysis, spatial reasoning, reflection, planning, and verification). We develop a three-stage training pipeline that progressively enhances the model's capabilities through imitation learning, self-exploration via rejection sampling, and self-correction through reflection tuning. The evaluation shows that our model significantly outperforms those advanced visual reasoning models, e.g., it exceeds OpenAI o1, o3-mini, and Claude-3.7 by +9\%, 24\%, and +13\%. Analysis reveals our model exhibits fewer repeated searches and logical inconsistencies, with particular advantages in complex long-horizon tasks. Real-world environments also show our superiority while exhibiting fewer repeated searches and logical inconsistency cases.
>
---
#### [replaced 013] MGPATH: Vision-Language Model with Multi-Granular Prompt Learning for Few-Shot WSI Classification
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.07409v3](http://arxiv.org/pdf/2502.07409v3)**

> **作者:** Anh-Tien Nguyen; Duy Minh Ho Nguyen; Nghiem Tuong Diep; Trung Quoc Nguyen; Nhat Ho; Jacqueline Michelle Metsch; Miriam Cindy Maurer; Daniel Sonntag; Hanibal Bohnenberger; Anne-Christin Hauschild
>
> **摘要:** Whole slide pathology image classification presents challenges due to gigapixel image sizes and limited annotation labels, hindering model generalization. This paper introduces a prompt learning method to adapt large vision-language models for few-shot pathology classification. We first extend the Prov-GigaPath vision foundation model, pre-trained on 1.3 billion pathology image tiles, into a vision-language model by adding adaptors and aligning it with medical text encoders via contrastive learning on 923K image-text pairs. The model is then used to extract visual features and text embeddings from few-shot annotations and fine-tunes with learnable prompt embeddings. Unlike prior methods that combine prompts with frozen features using prefix embeddings or self-attention, we propose multi-granular attention that compares interactions between learnable prompts with individual image patches and groups of them. This approach improves the model's ability to capture both fine-grained details and broader context, enhancing its recognition of complex patterns across sub-regions. To further improve accuracy, we leverage (unbalanced) optimal transport-based visual-text distance to secure model robustness by mitigating perturbations that might occur during the data augmentation process. Empirical experiments on lung, kidney, and breast pathology modalities validate the effectiveness of our approach; thereby, we surpass several of the latest competitors and consistently improve performance across diverse architectures, including CLIP, PLIP, and Prov-GigaPath integrated PLIP. We release our implementations and pre-trained models at this MGPATH.
>
---
#### [replaced 014] EiHi Net: Out-of-Distribution Generalization Paradigm
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2209.14946v3](http://arxiv.org/pdf/2209.14946v3)**

> **作者:** Qinglai Wei; Beiming Yuan; Diancheng Chen
>
> **摘要:** This paper develops a new EiHi net to solve the out-of-distribution (OoD) generalization problem in deep learning. EiHi net is a model learning paradigm that can be blessed on any visual backbone. This paradigm can change the previous learning method of the deep model, namely find out correlations between inductive sample features and corresponding categories, which suffers from pseudo correlations between indecisive features and labels. We fuse SimCLR and VIC-Reg via explicitly and dynamically establishing the original - positive - negative sample pair as a minimal learning element, the deep model iteratively establishes a relationship close to the causal one between features and labels, while suppressing pseudo correlations. To further validate the proposed model, and strengthen the established causal relationships, we develop a human-in-the-loop strategy, with few guidance samples, to prune the representation space directly. Finally, it is shown that the developed EiHi net makes significant improvements in the most difficult and typical OoD dataset Nico, compared with the current SOTA results, without any domain ($e.g.$ background, irrelevant features) information.
>
---
#### [replaced 015] DiffCloud: Real-to-Sim from Point Clouds with Differentiable Simulation and Rendering of Deformable Objects
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2204.03139v2](http://arxiv.org/pdf/2204.03139v2)**

> **作者:** Priya Sundaresan; Rika Antonova; Jeannette Bohg
>
> **摘要:** Research in manipulation of deformable objects is typically conducted on a limited range of scenarios, because handling each scenario on hardware takes significant effort. Realistic simulators with support for various types of deformations and interactions have the potential to speed up experimentation with novel tasks and algorithms. However, for highly deformable objects it is challenging to align the output of a simulator with the behavior of real objects. Manual tuning is not intuitive, hence automated methods are needed. We view this alignment problem as a joint perception-inference challenge and demonstrate how to use recent neural network architectures to successfully perform simulation parameter inference from real point clouds. We analyze the performance of various architectures, comparing their data and training requirements. Furthermore, we propose to leverage differentiable point cloud sampling and differentiable simulation to significantly reduce the time to achieve the alignment. We employ an efficient way to propagate gradients from point clouds to simulated meshes and further through to the physical simulation parameters, such as mass and stiffness. Experiments with highly deformable objects show that our method can achieve comparable or better alignment with real object behavior, while reducing the time needed to achieve this by more than an order of magnitude. Videos and supplementary material are available at https://diffcloud.github.io.
>
---
#### [replaced 016] Simulating Dynamic Tumor Contrast Enhancement in Breast MRI using Conditional Generative Adversarial Networks
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.18872v2](http://arxiv.org/pdf/2409.18872v2)**

> **作者:** Richard Osuala; Smriti Joshi; Apostolia Tsirikoglou; Lidia Garrucho; Walter H. L. Pinaya; Daniel M. Lang; Julia A. Schnabel; Oliver Diaz; Karim Lekadir
>
> **摘要:** This paper presents a method for virtual contrast enhancement in breast MRI, offering a promising non-invasive alternative to traditional contrast agent-based DCE-MRI acquisition. Using a conditional generative adversarial network, we predict DCE-MRI images, including jointly-generated sequences of multiple corresponding DCE-MRI timepoints, from non-contrast-enhanced MRIs, enabling tumor localization and characterization without the associated health risks. Furthermore, we qualitatively and quantitatively evaluate the synthetic DCE-MRI images, proposing a multi-metric Scaled Aggregate Measure (SAMe), assessing their utility in a tumor segmentation downstream task, and conclude with an analysis of the temporal patterns in multi-sequence DCE-MRI generation. Our approach demonstrates promising results in generating realistic and useful DCE-MRI sequences, highlighting the potential of virtual contrast enhancement for improving breast cancer diagnosis and treatment, particularly for patients where contrast agent administration is contraindicated.
>
---
#### [replaced 017] AdaWorld: Learning Adaptable World Models with Latent Actions
- **分类: cs.AI; cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.18938v3](http://arxiv.org/pdf/2503.18938v3)**

> **作者:** Shenyuan Gao; Siyuan Zhou; Yilun Du; Jun Zhang; Chuang Gan
>
> **备注:** ICML 2025. Project page: https://adaptable-world-model.github.io/, code: https://github.com/Little-Podi/AdaWorld, model: https://huggingface.co/Little-Podi/AdaWorld
>
> **摘要:** World models aim to learn action-controlled future prediction and have proven essential for the development of intelligent agents. However, most existing world models rely heavily on substantial action-labeled data and costly training, making it challenging to adapt to novel environments with heterogeneous actions through limited interactions. This limitation can hinder their applicability across broader domains. To overcome this limitation, we propose AdaWorld, an innovative world model learning approach that enables efficient adaptation. The key idea is to incorporate action information during the pretraining of world models. This is achieved by extracting latent actions from videos in a self-supervised manner, capturing the most critical transitions between frames. We then develop an autoregressive world model that conditions on these latent actions. This learning paradigm enables highly adaptable world models, facilitating efficient transfer and learning of new actions even with limited interactions and finetuning. Our comprehensive experiments across multiple environments demonstrate that AdaWorld achieves superior performance in both simulation quality and visual planning.
>
---
#### [replaced 018] METDrive: Multi-modal End-to-end Autonomous Driving with Temporal Guidance
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.12667v3](http://arxiv.org/pdf/2409.12667v3)**

> **作者:** Ziang Guo; Xinhao Lin; Zakhar Yagudin; Artem Lykov; Yong Wang; Yanqiang Li; Dzmitry Tsetserukou
>
> **备注:** Accepted by ICRA
>
> **摘要:** Multi-modal end-to-end autonomous driving has shown promising advancements in recent work. By embedding more modalities into end-to-end networks, the system's understanding of both static and dynamic aspects of the driving environment is enhanced, thereby improving the safety of autonomous driving. In this paper, we introduce METDrive, an end-to-end system that leverages temporal guidance from the embedded time series features of ego states, including rotation angles, steering, throttle signals, and waypoint vectors. The geometric features derived from perception sensor data and the time series features of ego state data jointly guide the waypoint prediction with the proposed temporal guidance loss function. We evaluated METDrive on the CARLA leaderboard benchmarks, achieving a driving score of 70%, a route completion score of 94%, and an infraction score of 0.78.
>
---
#### [replaced 019] A Deep Learning Approach for Pixel-level Material Classification via Hyperspectral Imaging
- **分类: eess.IV; cs.AI; cs.CV; I.5; I.2.10**

- **链接: [http://arxiv.org/pdf/2409.13498v2](http://arxiv.org/pdf/2409.13498v2)**

> **作者:** Savvas Sifnaios; George Arvanitakis; Fotios K. Konstantinidis; Georgios Tsimiklis; Angelos Amditis; Panayiotis Frangos
>
> **备注:** 13 pages, 15 figures, 6 equations
>
> **摘要:** Recent advancements in computer vision, particularly in detection, segmentation, and classification, have significantly impacted various domains. However, these advancements are tied to RGB-based systems, which are insufficient for applications in industries like waste sorting, pharmaceuticals, and defense, where advanced object characterization beyond shape or color is necessary. Hyperspectral (HS) imaging, capturing both spectral and spatial information, addresses these limitations and offers advantages over conventional technologies such as X-ray fluorescence and Raman spectroscopy, particularly in terms of speed, cost, and safety. This study evaluates the potential of combining HS imaging with deep learning for material characterization. The research involves: i) designing an experimental setup with HS camera, conveyor, and controlled lighting; ii) generating a multi-object dataset of various plastics (HDPE, PET, PP, PS) with semi-automated mask generation and Raman spectroscopy-based labeling; and iii) developing a deep learning model trained on HS images for pixel-level material classification. The model achieved 99.94\% classification accuracy, demonstrating robustness in color, size, and shape invariance, and effectively handling material overlap. Limitations, such as challenges with black objects, are also discussed. Extending computer vision beyond RGB to HS imaging proves feasible, overcoming major limitations of traditional methods and showing strong potential for future applications.
>
---
#### [replaced 020] Error correcting 2D-3D cascaded network for myocardial infarct scar segmentation on late gadolinium enhancement cardiac magnetic resonance images
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2306.14725v2](http://arxiv.org/pdf/2306.14725v2)**

> **作者:** Matthias Schwab; Mathias Pamminger; Christian Kremser; Daniel Obmann; Markus Haltmeier; Agnes Mayr
>
> **摘要:** Late gadolinium enhancement (LGE) cardiac magnetic resonance (CMR) imaging is considered the in vivo reference standard for assessing infarct size (IS) and microvascular obstruction (MVO) in ST-elevation myocardial infarction (STEMI) patients. However, the exact quantification of those markers of myocardial infarct severity remains challenging and very time-consuming. As LGE distribution patterns can be quite complex and hard to delineate from the blood pool or epicardial fat, automatic segmentation of LGE CMR images is challenging. In this work, we propose a cascaded framework of two-dimensional and three-dimensional convolutional neural networks (CNNs) which enables to calculate the extent of myocardial infarction in a fully automated way. By artificially generating segmentation errors which are characteristic for 2D CNNs during training of the cascaded framework we are enforcing the detection and correction of 2D segmentation errors and hence improve the segmentation accuracy of the entire method. The proposed method was trained and evaluated on two publicly available datasets. We perform comparative experiments where we show that our framework outperforms state-of-the-art reference methods in segmentation of myocardial infarction. Furthermore, in extensive ablation studies we show the advantages that come with the proposed error correcting cascaded method. The code of this project is publicly available at https://github.com/matthi99/EcorC.git
>
---
#### [replaced 021] G-MSGINet: A Grouped Multi-Scale Graph-Involution Network for Contactless Fingerprint Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08233v2](http://arxiv.org/pdf/2505.08233v2)**

> **作者:** Santhoshkumar Peddi; Soham Bandyopadhyay; Debasis Samanta
>
> **摘要:** This paper presents G-MSGINet, a unified and efficient framework for robust contactless fingerprint recognition that jointly performs minutiae localization and identity embedding directly from raw input images. Existing approaches rely on multi-branch architectures, orientation labels, or complex preprocessing steps, which limit scalability and generalization across real-world acquisition scenarios. In contrast, the proposed architecture introduces the GMSGI layer, a novel computational module that integrates grouped pixel-level involution, dynamic multi-scale kernel generation, and graph-based relational modelling into a single processing unit. Stacked GMSGI layers progressively refine both local minutiae-sensitive features and global topological representations through end-to-end optimization. The architecture eliminates explicit orientation supervision and adapts graph connectivity directly from learned kernel descriptors, thereby capturing meaningful structural relationships among fingerprint regions without fixed heuristics. Extensive experiments on three benchmark datasets, namely PolyU, CFPose, and Benchmark 2D/3D, demonstrate that G-MSGINet consistently achieves minutiae F1-scores in the range of $0.83\pm0.02$ and Rank-1 identification accuracies between 97.0% and 99.1%, while maintaining an Equal Error Rate (EER) as low as 0.5%. These results correspond to improvements of up to 4.8% in F1-score and 1.4% in Rank-1 accuracy when compared to prior methods, using only 0.38 million parameters and 6.63 giga floating-point operations, which represents up to ten times fewer parameters than competitive baselines. This highlights the scalability and effectiveness of G-MSGINet in real-world contactless biometric recognition scenarios.
>
---
#### [replaced 022] Iterative Occlusion-Aware Light Field Depth Estimation using 4D Geometrical Cues
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2403.02043v2](http://arxiv.org/pdf/2403.02043v2)**

> **作者:** Rui Lourenço; Lucas Thomaz; Eduardo A. B. Silva; Sergio M. M. Faria
>
> **摘要:** Light field cameras and multi-camera arrays have emerged as promising solutions for accurately estimating depth by passively capturing light information. This is possible because the 3D information of a scene is embedded in the 4D light field geometry. Commonly, depth estimation methods extract this information relying on gradient information, heuristic-based optimisation models, or learning-based approaches. This paper focuses mainly on explicitly understanding and exploiting 4D geometrical cues for light field depth estimation. Thus, a novel method is proposed, based on a non-learning-based optimisation approach for depth estimation that explicitly considers surface normal accuracy and occlusion regions by utilising a fully explainable 4D geometric model of the light field. The 4D model performs depth/disparity estimation by determining the orientations and analysing the intersections of key 2D planes in 4D space, which are the images of 3D-space points in the 4D light field. Experimental results show that the proposed method outperforms both learning-based and non-learning-based state-of-the-art methods in terms of surface normal angle accuracy, achieving a Median Angle Error on planar surfaces, on average, 26.3$\%$ lower than the state-of-the-art, and still being competitive with state-of-the-art methods in terms of MSE ${\times}$ 100 and Badpix 0.07.
>
---
#### [replaced 023] A Call to Arms: AI Should be Critical for Social Media Analysis of Conflict Zones
- **分类: cs.CY; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2311.00810v3](http://arxiv.org/pdf/2311.00810v3)**

> **作者:** Afia Abedin; Abdul Bais; Cody Buntain; Laura Courchesne; Brian McQuinn; Matthew E. Taylor; Muhib Ullah
>
> **摘要:** The massive proliferation of social media data represents a transformative opportunity for conflict studies and for tracking the proliferation and use of weaponry, as conflicts are increasingly documented in these online spaces. At the same time, the scale and types of data available are problematic for traditional open-source intelligence. This paper focuses on identifying specific weapon systems and the insignias of the armed groups using them as documented in the Ukraine war, as these tasks are critical to operational intelligence and tracking weapon proliferation, especially given the scale of international military aid given to Ukraine. The large scale of social media makes manual assessment difficult, however, so this paper presents early work that uses computer vision models to support this task. We demonstrate that these models can both identify weapons embedded in images shared in social media and how the resulting collection of military-relevant images and their post times interact with the offline, real-world conflict. Not only can we then track changes in the prevalence of images of tanks, land mines, military trucks, etc., we find correlations among time series data associated with these images and the daily fatalities in this conflict. This work shows substantial opportunity for examining similar online documentation of conflict contexts, and we also point to future avenues where computer vision can be further improved for these open-source intelligence tasks.
>
---
#### [replaced 024] State-of-the-Art Periorbital Distance Prediction and Disease Classification Using Periorbital Features
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.18769v5](http://arxiv.org/pdf/2409.18769v5)**

> **作者:** George R. Nahass; Sasha Hubschman; Jeffrey C. Peterson; Ghasem Yazdanpanah; Nicholas Tomaras; Madison Cheung; Alex Palacios; Kevin Heinze; Chad A. Purnell; Pete Setabutr; Ann Q. Tran; Darvin Yi
>
> **备注:** 25 pages, 12 figures, 16 tables
>
> **摘要:** Periorbital distances are critical markers for diagnosing and monitoring a range of oculoplastic and craniofacial conditions. Manual measurement, however, is subjective and prone to intergrader variability. Automated methods have been developed but remain limited by standardized imaging requirements, small datasets, and a narrow focus on individual measurements. We developed a segmentation pipeline trained on a domain-specific dataset of healthy eyes and compared its performance against the Segment Anything Model (SAM) and the prior benchmark, PeriorbitAI. Segmentation accuracy was evaluated across multiple disease classes and imaging conditions. We further investigated the use of predicted periorbital distances as features for disease classification under in-distribution (ID) and out-of-distribution (OOD) settings, comparing shallow classifiers, CNNs, and fusion models. Our segmentation model achieved state-of-the-art accuracy across all datasets, with error rates within intergrader variability and superior performance relative to SAM and PeriorbitAI. In classification tasks, models trained on periorbital distances matched CNN performance on ID data (77--78\% accuracy) and substantially outperformed CNNs under OOD conditions (63--68\% accuracy vs. 14\%). Fusion models achieved the highest ID accuracy (80\%) but were sensitive to degraded CNN features under OOD shifts. Segmentation-derived periorbital distances provide robust, explainable features for disease classification and generalize better under domain shift than CNN image classifiers. These results establish a new benchmark for periorbital distance prediction and highlight the potential of anatomy-based AI pipelines for real-world deployment in oculoplastic and craniofacial care.
>
---
#### [replaced 025] F$^3$Loc: Fusion and Filtering for Floorplan Localization
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2403.03370v2](http://arxiv.org/pdf/2403.03370v2)**

> **作者:** Changan Chen; Rui Wang; Christoph Vogel; Marc Pollefeys
>
> **备注:** 10 pages, 11 figure, accepted to CVPR 2024 (fixed typo eq.8: s_x,s_y, s_phi -> x, y, phi)
>
> **摘要:** In this paper we propose an efficient data-driven solution to self-localization within a floorplan. Floorplan data is readily available, long-term persistent and inherently robust to changes in the visual appearance. Our method does not require retraining per map and location or demand a large database of images of the area of interest. We propose a novel probabilistic model consisting of an observation and a novel temporal filtering module. Operating internally with an efficient ray-based representation, the observation module consists of a single and a multiview module to predict horizontal depth from images and fuses their results to benefit from advantages offered by either methodology. Our method operates on conventional consumer hardware and overcomes a common limitation of competing methods that often demand upright images. Our full system meets real-time requirements, while outperforming the state-of-the-art by a significant margin.
>
---
#### [replaced 026] PRISM: A Unified Framework for Photorealistic Reconstruction and Intrinsic Scene Modeling
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.14219v2](http://arxiv.org/pdf/2504.14219v2)**

> **作者:** Alara Dirik; Tuanfeng Wang; Duygu Ceylan; Stefanos Zafeiriou; Anna Frühstück
>
> **摘要:** We present PRISM, a unified framework that enables multiple image generation and editing tasks in a single foundational model. Starting from a pre-trained text-to-image diffusion model, PRISM proposes an effective fine-tuning strategy to produce RGB images along with intrinsic maps (referred to as X layers) simultaneously. Unlike previous approaches, which infer intrinsic properties individually or require separate models for decomposition and conditional generation, PRISM maintains consistency across modalities by generating all intrinsic layers jointly. It supports diverse tasks, including text-to-RGBX generation, RGB-to-X decomposition, and X-to-RGBX conditional generation. Additionally, PRISM enables both global and local image editing through conditioning on selected intrinsic layers and text prompts. Extensive experiments demonstrate the competitive performance of PRISM both for intrinsic image decomposition and conditional image generation while preserving the base model's text-to-image generation capability.
>
---
#### [replaced 027] HCMA: Hierarchical Cross-model Alignment for Grounded Text-to-Image Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.06512v2](http://arxiv.org/pdf/2505.06512v2)**

> **作者:** Hang Wang; Zhi-Qi Cheng; Chenhao Lin; Chao Shen; Lei Zhang
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Text-to-image synthesis has progressed to the point where models can generate visually compelling images from natural language prompts. Yet, existing methods often fail to reconcile high-level semantic fidelity with explicit spatial control, particularly in scenes involving multiple objects, nuanced relations, or complex layouts. To bridge this gap, we propose a Hierarchical Cross-Modal Alignment (HCMA) framework for grounded text-to-image generation. HCMA integrates two alignment modules into each diffusion sampling step: a global module that continuously aligns latent representations with textual descriptions to ensure scene-level coherence, and a local module that employs bounding-box layouts to anchor objects at specified locations, enabling fine-grained spatial control. Extensive experiments on the MS-COCO 2014 validation set show that HCMA surpasses state-of-the-art baselines, achieving a 0.69 improvement in Frechet Inception Distance (FID) and a 0.0295 gain in CLIP Score. These results demonstrate HCMA's effectiveness in faithfully capturing intricate textual semantics while adhering to user-defined spatial constraints, offering a robust solution for semantically grounded image generation.Our code is available at https://github.com/hwang-cs-ime/HCMA
>
---
#### [replaced 028] Towards Autonomous UAV Visual Object Search in City Space: Benchmark and Agentic Methodology
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.08765v2](http://arxiv.org/pdf/2505.08765v2)**

> **作者:** Yatai Ji; Zhengqiu Zhu; Yong Zhao; Beidan Liu; Chen Gao; Yihao Zhao; Sihang Qiu; Yue Hu; Quanjun Yin; Yong Li
>
> **摘要:** Aerial Visual Object Search (AVOS) tasks in urban environments require Unmanned Aerial Vehicles (UAVs) to autonomously search for and identify target objects using visual and textual cues without external guidance. Existing approaches struggle in complex urban environments due to redundant semantic processing, similar object distinction, and the exploration-exploitation dilemma. To bridge this gap and support the AVOS task, we introduce CityAVOS, the first benchmark dataset for autonomous search of common urban objects. This dataset comprises 2,420 tasks across six object categories with varying difficulty levels, enabling comprehensive evaluation of UAV agents' search capabilities. To solve the AVOS tasks, we also propose PRPSearcher (Perception-Reasoning-Planning Searcher), a novel agentic method powered by multi-modal large language models (MLLMs) that mimics human three-tier cognition. Specifically, PRPSearcher constructs three specialized maps: an object-centric dynamic semantic map enhancing spatial perception, a 3D cognitive map based on semantic attraction values for target reasoning, and a 3D uncertainty map for balanced exploration-exploitation search. Also, our approach incorporates a denoising mechanism to mitigate interference from similar objects and utilizes an Inspiration Promote Thought (IPT) prompting mechanism for adaptive action planning. Experimental results on CityAVOS demonstrate that PRPSearcher surpasses existing baselines in both success rate and search efficiency (on average: +37.69% SR, +28.96% SPL, -30.69% MSS, and -46.40% NE). While promising, the performance gap compared to humans highlights the need for better semantic reasoning and spatial exploration capabilities in AVOS tasks. This work establishes a foundation for future advances in embodied target search. Dataset and source code are available at https://anonymous.4open.science/r/CityAVOS-3DF8.
>
---
#### [replaced 029] WaveGuard: Robust Deepfake Detection and Source Tracing via Dual-Tree Complex Wavelet and Graph Neural Networks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08614v2](http://arxiv.org/pdf/2505.08614v2)**

> **作者:** Ziyuan He; Zhiqing Guo; Liejun Wang; Gaobo Yang; Yunfeng Diao; Dan Ma
>
> **备注:** 11 pages, 5 figures, 4 tables
>
> **摘要:** Deepfake technology poses increasing risks such as privacy invasion and identity theft. To address these threats, we propose WaveGuard, a proactive watermarking framework that enhances robustness and imperceptibility via frequency-domain embedding and graph-based structural consistency. Specifically, we embed watermarks into high-frequency sub-bands using Dual-Tree Complex Wavelet Transform (DT-CWT) and employ a Structural Consistency Graph Neural Network (SC-GNN) to preserve visual quality. We also design an attention module to refine embedding precision. Experimental results on face swap and reenactment tasks demonstrate that WaveGuard outperforms state-of-the-art methods in both robustness and visual quality. Code is available at https://github.com/vpsg-research/WaveGuard.
>
---
#### [replaced 030] Using Few-Shot Learning to Classify Primary Lung Cancer and Other Malignancy with Lung Metastasis in Cytological Imaging via Endobronchial Ultrasound Procedures
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2404.06080v4](http://arxiv.org/pdf/2404.06080v4)**

> **作者:** Ching-Kai Lin; Di-Chun Wei; Yun-Chien Cheng
>
> **摘要:** This study presents a computer-aided diagnosis (CAD) system to assist early detection of lung metastases during endobronchial ultrasound (EBUS) procedures, significantly reducing follow-up time and enabling timely treatment. Due to limited cytology images and morphological similarities among cells, classifying lung metastases is challenging, and existing research rarely targets this issue directly.To overcome data scarcity and improve classification, the authors propose a few-shot learning model using a hybrid pretrained backbone with fine-grained classification and contrastive learning. Parameter-efficient fine-tuning on augmented support sets enhances generalization and transferability. The model achieved 49.59% accuracy, outperforming existing methods. With 20 image samples, accuracy improved to 55.48%, showing strong potential for identifying rare or novel cancer types in low-data clinical environments.
>
---
#### [replaced 031] An ocean front detection and tracking algorithm
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.15250v4](http://arxiv.org/pdf/2502.15250v4)**

> **作者:** Yishuo Wang; Feng Zhou; Qicheng Meng; Muping Zhou; Zhijun Hu; Chengqing Zhang; Tianhao Zhao
>
> **摘要:** Existing ocean front detection methods--including histogram-based variance analysis, Lyapunov exponent, gradient thresholding, and machine learning--suffer from critical limitations: discontinuous outputs, over-detection, reliance on single-threshold decisions, and lack of open-source implementations. To address these challenges, this paper proposes the Bayesian Front Detection and Tracking framework with Metric Space Analysis (BFDT-MSA). The framework introduces three innovations: (1) a Bayesian decision mechanism that integrates gradient priors and field operators to eliminate manual threshold sensitivity; (2) morphological refinement algorithms for merging fragmented fronts, deleting spurious rings, and thinning frontal zones to pixel-level accuracy; and (3) a novel metric space definition for temporal front tracking, enabling systematic analysis of front evolution. Validated on global SST data (2022--2024), BFDT-MSA reduces over-detection by $73\%$ compared to histogram-based methods while achieving superior intensity ($0.16^\circ$C/km), continuity, and spatiotemporal coherence. The open-source release bridges a critical gap in reproducible oceanographic research.
>
---
#### [replaced 032] MCP-MedSAM: A Powerful Lightweight Medical Segment Anything Model Trained with a Single GPU in Just One Day
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.05888v2](http://arxiv.org/pdf/2412.05888v2)**

> **作者:** Donghang Lyu; Ruochen Gao; Marius Staring
>
> **备注:** Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA)
>
> **摘要:** Medical image segmentation involves partitioning medical images into meaningful regions, with a focus on identifying anatomical structures and lesions. It has broad applications in healthcare, and deep learning methods have enabled significant advancements in automating this process. Recently, the introduction of the Segmentation Anything Model (SAM), the first foundation model for segmentation task, has prompted researchers to adapt it for the medical domain to improve performance across various tasks. However, SAM's large model size and high GPU requirements hinder its scalability and development in the medical domain. In this work, we propose MCP-MedSAM, a powerful and lightweight medical SAM model designed to be trainable on a single A100 GPU with 40GB of memory within one day while delivering superior segmentation performance. Recognizing the significant internal differences between modalities and the need for direct segmentation target information within bounding boxes, we introduce two kinds of prompts: the modality prompt and the content prompt. After passing through the prompt encoder, their embedding representations can further improve the segmentation performance by incorporating more relevant information without adding significant training overhead. Additionally, we adopt an effective modality-based data sampling strategy to address data imbalance between modalities, ensuring more balanced performance across all modalities. Our method was trained and evaluated using a large-scale challenge dataset, compared to top-ranking methods on the challenge leaderboard, MCP-MedSAM achieved superior performance while requiring only one day of training on a single GPU. The code is publicly available at \textcolor{blue}{https://github.com/dong845/MCP-MedSAM}.}
>
---
#### [replaced 033] HybridMQA: Exploring Geometry-Texture Interactions for Colored Mesh Quality Assessment
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2412.01986v2](http://arxiv.org/pdf/2412.01986v2)**

> **作者:** Armin Shafiee Sarvestani; Sheyang Tang; Zhou Wang
>
> **摘要:** Mesh quality assessment (MQA) models play a critical role in the design, optimization, and evaluation of mesh operation systems in a wide variety of applications. Current MQA models, whether model-based methods using topology-aware features or projection-based approaches working on rendered 2D projections, often fail to capture the intricate interactions between texture and 3D geometry. We introduce HybridMQA, a first-of-its-kind hybrid full-reference colored MQA framework that integrates model-based and projection-based approaches, capturing complex interactions between textural information and 3D structures for enriched quality representations. Our method employs graph learning to extract detailed 3D representations, which are then projected to 2D using a novel feature rendering process that precisely aligns them with colored projections. This enables the exploration of geometry-texture interactions via cross-attention, producing comprehensive mesh quality representations. Extensive experiments demonstrate HybridMQA's superior performance across diverse datasets, highlighting its ability to effectively leverage geometry-texture interactions for a thorough understanding of mesh quality. Our implementation will be made publicly available.
>
---
#### [replaced 034] ALEN: A Dual-Approach for Uniform and Non-Uniform Low-Light Image Enhancement
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.19708v5](http://arxiv.org/pdf/2407.19708v5)**

> **作者:** Ezequiel Perez-Zarate; Oscar Ramos-Soto; Chunxiao Liu; Diego Oliva; Marco Perez-Cisneros
>
> **备注:** Minor updates and corrections
>
> **摘要:** Low-light image enhancement is an important task in computer vision, essential for improving the visibility and quality of images captured in non-optimal lighting conditions. Inadequate illumination can lead to significant information loss and poor image quality, impacting various applications such as surveillance. photography, or even autonomous driving. In this regard, automated methods have been developed to automatically adjust illumination in the image for a better visual perception. Current enhancement techniques often use specific datasets to enhance low-light images, but still present challenges when adapting to diverse real-world conditions, where illumination degradation may be localized to specific regions. To address this challenge, the Adaptive Light Enhancement Network (ALEN) is introduced, whose main approach is the use of a classification mechanism to determine whether local or global illumination enhancement is required. Subsequently, estimator networks adjust illumination based on this classification and simultaneously enhance color fidelity. ALEN integrates the Light Classification Network (LCNet) for illuminance categorization, complemented by the Single-Channel Network (SCNet), and Multi-Channel Network (MCNet) for precise estimation of illumination and color, respectively. Extensive experiments on publicly available datasets for low-light conditions were carried out to underscore ALEN's robust generalization capabilities, demonstrating superior performance in both quantitative metrics and qualitative assessments when compared to recent state-of-the-art methods. The ALEN not only enhances image quality in terms of visual perception but also represents an advancement in high-level vision tasks, such as semantic segmentation, as presented in this work. The code of this method is available at https://github.com/xingyumex/ALEN
>
---
#### [replaced 035] Thermal Detection of People with Mobility Restrictions for Barrier Reduction at Traffic Lights Controlled Intersections
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08568v2](http://arxiv.org/pdf/2505.08568v2)**

> **作者:** Xiao Ni; Carsten Kuehnel; Xiaoyi Jiang
>
> **摘要:** Rapid advances in deep learning for computer vision have driven the adoption of RGB camera-based adaptive traffic light systems to improve traffic safety and pedestrian comfort. However, these systems often overlook the needs of people with mobility restrictions. Moreover, the use of RGB cameras presents significant challenges, including limited detection performance under adverse weather or low-visibility conditions, as well as heightened privacy concerns. To address these issues, we propose a fully automated, thermal detector-based traffic light system that dynamically adjusts signal durations for individuals with walking impairments or mobility burden and triggers the auditory signal for visually impaired individuals, thereby advancing towards barrier-free intersection for all users. To this end, we build the thermal dataset for people with mobility restrictions (TD4PWMR), designed to capture diverse pedestrian scenarios, particularly focusing on individuals with mobility aids or mobility burden under varying environmental conditions, such as different lighting, weather, and crowded urban settings. While thermal imaging offers advantages in terms of privacy and robustness to adverse conditions, it also introduces inherent hurdles for object detection due to its lack of color and fine texture details and generally lower resolution of thermal images. To overcome these limitations, we develop YOLO-Thermal, a novel variant of the YOLO architecture that integrates advanced feature extraction and attention mechanisms for enhanced detection accuracy and robustness in thermal imaging. Experiments demonstrate that the proposed thermal detector outperforms existing detectors, while the proposed traffic light system effectively enhances barrier-free intersection. The source codes and dataset are available at https://github.com/leon2014dresden/YOLO-THERMAL.
>
---
#### [replaced 036] Benchmarking Large Vision-Language Models on Fine-Grained Image Tasks: A Comprehensive Evaluation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.14988v2](http://arxiv.org/pdf/2504.14988v2)**

> **作者:** Hong-Tao Yu; Xiu-Shen Wei; Yuxin Peng; Serge Belongie
>
> **摘要:** Recent advancements in Large Vision-Language Models (LVLMs) have demonstrated remarkable multimodal perception capabilities, garnering significant attention. While numerous evaluation studies have emerged, assessing LVLMs both holistically and on specialized tasks, fine-grained image tasks-fundamental to computer vision-remain largely unexplored. To fill this gap, we introduce a comprehensive fine-grained evaluation benchmark, i.e., FG-BMK, comprising 1.01 million questions and 0.33 million images. Our evaluation systematically examines LVLMs from both human-oriented and machine-oriented perspectives, focusing on their semantic recognition and fine-grained feature representation capabilities. Through extensive experiments on twelve representative LVLMs/VLMs, we uncover key findings regarding the influence of training paradigms, modality alignment, perturbation susceptibility, and fine-grained category reasoning on task performance. This work provides critical insights into the limitations of current LVLMs and offers guidance for future data construction and model design in the development of more advanced LVLMs. Our code is open-source and available at https://github.com/SEU-VIPGroup/FG-BMK.
>
---
#### [replaced 037] One Homography is All You Need: IMM-based Joint Homography and Multiple Object State Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.02562v3](http://arxiv.org/pdf/2409.02562v3)**

> **作者:** Paul Johannes Claasen; Johan Pieter de Villiers
>
> **备注:** Preprint submitted to Expert Systems with Applications
>
> **摘要:** A novel online MOT algorithm, IMM Joint Homography State Estimation (IMM-JHSE), is proposed. IMM-JHSE uses an initial homography estimate as the only additional 3D information, whereas other 3D MOT methods use regular 3D measurements. By jointly modelling the homography matrix and its dynamics as part of track state vectors, IMM-JHSE removes the explicit influence of camera motion compensation techniques on predicted track position states, which was prevalent in previous approaches. Expanding upon this, static and dynamic camera motion models are combined using an IMM filter. A simple bounding box motion model is used to predict bounding box positions to incorporate image plane information. In addition to applying an IMM to camera motion, a non-standard IMM approach is applied where bounding-box-based BIoU scores are mixed with ground-plane-based Mahalanobis distances in an IMM-like fashion to perform association only, making IMM-JHSE robust to motion away from the ground plane. Finally, IMM-JHSE makes use of dynamic process and measurement noise estimation techniques. IMM-JHSE improves upon related techniques, including UCMCTrack, OC-SORT, C-BIoU and ByteTrack on the DanceTrack and KITTI-car datasets, increasing HOTA by 2.64 and 2.11, respectively, while offering competitive performance on the MOT17, MOT20 and KITTI-pedestrian datasets. Using publicly available detections, IMM-JHSE outperforms almost all other 2D MOT methods and is outperformed only by 3D MOT methods -- some of which are offline -- on the KITTI-car dataset. Compared to tracking-by-attention methods, IMM-JHSE shows remarkably similar performance on the DanceTrack dataset and outperforms them on the MOT17 dataset. The code is publicly available: https://github.com/Paulkie99/imm-jhse.
>
---
#### [replaced 038] Efficient approximation of Earth Mover's Distance Based on Nearest Neighbor Search
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2401.07378v3](http://arxiv.org/pdf/2401.07378v3)**

> **作者:** Guangyu Meng; Ruyu Zhou; Liu Liu; Peixian Liang; Fang Liu; Danny Chen; Michael Niemier; X. Sharon Hu
>
> **摘要:** Earth Mover's Distance (EMD) is an important similarity measure between two distributions, used in computer vision and many other application domains. However, its exact calculation is computationally and memory intensive, which hinders its scalability and applicability for large-scale problems. Various approximate EMD algorithms have been proposed to reduce computational costs, but they suffer lower accuracy and may require additional memory usage or manual parameter tuning. In this paper, we present a novel approach, NNS-EMD, to approximate EMD using Nearest Neighbor Search (NNS), in order to achieve high accuracy, low time complexity, and high memory efficiency. The NNS operation reduces the number of data points compared in each NNS iteration and offers opportunities for parallel processing. We further accelerate NNS-EMD via vectorization on GPU, which is especially beneficial for large datasets. We compare NNS-EMD with both the exact EMD and state-of-the-art approximate EMD algorithms on image classification and retrieval tasks. We also apply NNS-EMD to calculate transport mapping and realize color transfer between images. NNS-EMD can be 44x to 135x faster than the exact EMD implementation, and achieves superior accuracy, speedup, and memory efficiency over existing approximate EMD methods.
>
---
#### [replaced 039] Reflecting Topology Consistency and Abnormality via Learnable Attentions for Airway Labeling
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.23854v2](http://arxiv.org/pdf/2410.23854v2)**

> **作者:** Chenyu Li; Minghui Zhang; Chuyan Zhang; Yun Gu
>
> **摘要:** Accurate airway anatomical labeling is crucial for clinicians to identify and navigate complex bronchial structures during bronchoscopy. Automatic airway anatomical labeling is challenging due to significant individual variability and anatomical variations. Previous methods are prone to generate inconsistent predictions, which is harmful for preoperative planning and intraoperative navigation. This paper aims to address these challenges by proposing a novel method that enhances topological consistency and improves the detection of abnormal airway branches. We propose a novel approach incorporating two modules: the Soft Subtree Consistency (SSC) and the Abnormal Branch Saliency (ABS). The SSC module constructs a soft subtree to capture clinically relevant topological relationships, allowing for flexible feature aggregation within and across subtrees. The ABS module facilitates the interaction between node features and prototypes to distinguish abnormal branches, preventing the erroneous aggregation of features between normal and abnormal nodes. Evaluated on a challenging dataset characterized by severe airway distortion and atrophy, our method achieves superior performance compared to state-of-the-art approaches. Specifically, it attains a 91.4% accuracy at the segmental level and an 83.7% accuracy at the subsegmental level, representing a 1.4% increase in subsegmental accuracy and a 3.1% increase in topological consistency. Notably, the method demonstrates reliable performance in cases with disease-induced airway deformities, ensuring consistent and accurate labeling.
>
---
#### [replaced 040] Video-R1: Reinforcing Video Reasoning in MLLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.21776v2](http://arxiv.org/pdf/2503.21776v2)**

> **作者:** Kaituo Feng; Kaixiong Gong; Bohao Li; Zonghao Guo; Yibing Wang; Tianshuo Peng; Junfei Wu; Xiaoying Zhang; Benyou Wang; Xiangyu Yue
>
> **备注:** Project page: https://github.com/tulerfeng/Video-R1
>
> **摘要:** Inspired by DeepSeek-R1's success in eliciting reasoning abilities through rule-based reinforcement learning (RL), we introduce Video-R1 as the first attempt to systematically explore the R1 paradigm for incentivizing video reasoning within multimodal large language models (MLLMs). However, directly applying RL training with the GRPO algorithm to video reasoning presents two primary challenges: (i) a lack of temporal modeling for video reasoning, and (ii) the scarcity of high-quality video-reasoning data. To address these issues, we first propose the T-GRPO algorithm, which encourages models to utilize temporal information in videos for reasoning. Additionally, instead of relying solely on video data, we incorporate high-quality image-reasoning data into the training process. We have constructed two datasets: Video-R1-CoT-165k for SFT cold start and Video-R1-260k for RL training, both comprising image and video data. Experimental results demonstrate that Video-R1 achieves significant improvements on video reasoning benchmarks such as VideoMMMU and VSI-Bench, as well as on general video benchmarks including MVBench and TempCompass, etc. Notably, Video-R1-7B attains a 37.1% accuracy on video spatial reasoning benchmark VSI-bench, surpassing the commercial proprietary model GPT-4o. All code, models, and data are released in: https://github.com/tulerfeng/Video-R1.
>
---
#### [replaced 041] The RaspGrade Dataset: Towards Automatic Raspberry Ripeness Grading with Deep Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08537v2](http://arxiv.org/pdf/2505.08537v2)**

> **作者:** Mohamed Lamine Mekhalfi; Paul Chippendale; Fabio Poiesi; Samuele Bonecher; Gilberto Osler; Nicola Zancanella
>
> **摘要:** This research investigates the application of computer vision for rapid, accurate, and non-invasive food quality assessment, focusing on the novel challenge of real-time raspberry grading into five distinct classes within an industrial environment as the fruits move along a conveyor belt. To address this, a dedicated dataset of raspberries, namely RaspGrade, was acquired and meticulously annotated. Instance segmentation experiments revealed that accurate fruit-level masks can be obtained; however, the classification of certain raspberry grades presents challenges due to color similarities and occlusion, while others are more readily distinguishable based on color. The acquired and annotated RaspGrade dataset is accessible on Hugging Face at: https://huggingface.co/datasets/FBK-TeV/RaspGrade.
>
---
