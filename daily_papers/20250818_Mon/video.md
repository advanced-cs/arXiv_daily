# 计算机视觉 cs.CV

- **最新发布 118 篇**

- **更新 64 篇**

## 最新发布

#### [new 001] Personalized Face Super-Resolution with Identity Decoupling and Fitting
- **分类: cs.CV**

- **简介: 该论文属于人脸超分辨率任务，解决极端退化下身份信息丢失问题。通过身份解耦与适配方法提升重建质量与身份一致性。**

- **链接: [http://arxiv.org/pdf/2508.10937v1](http://arxiv.org/pdf/2508.10937v1)**

> **作者:** Jiarui Yang; Hang Guo; Wen Huang; Tao Dai; Shutao Xia
>
> **摘要:** In recent years, face super-resolution (FSR) methods have achieved remarkable progress, generally maintaining high image fidelity and identity (ID) consistency under standard settings. However, in extreme degradation scenarios (e.g., scale $> 8\times$), critical attributes and ID information are often severely lost in the input image, making it difficult for conventional models to reconstruct realistic and ID-consistent faces. Existing methods tend to generate hallucinated faces under such conditions, producing restored images lacking authentic ID constraints. To address this challenge, we propose a novel FSR method with Identity Decoupling and Fitting (IDFSR), designed to enhance ID restoration under large scaling factors while mitigating hallucination effects. Our approach involves three key designs: 1) \textbf{Masking} the facial region in the low-resolution (LR) image to eliminate unreliable ID cues; 2) \textbf{Warping} a reference image to align with the LR input, providing style guidance; 3) Leveraging \textbf{ID embeddings} extracted from ground truth (GT) images for fine-grained ID modeling and personalized adaptation. We first pretrain a diffusion-based model to explicitly decouple style and ID by forcing it to reconstruct masked LR face regions using both style and identity embeddings. Subsequently, we freeze most network parameters and perform lightweight fine-tuning of the ID embedding using a small set of target ID images. This embedding encodes fine-grained facial attributes and precise ID information, significantly improving both ID consistency and perceptual quality. Extensive quantitative evaluations and visual comparisons demonstrate that the proposed IDFSR substantially outperforms existing approaches under extreme degradation, particularly achieving superior performance on ID consistency.
>
---
#### [new 002] A Survey on Video Temporal Grounding with Multimodal Large Language Model
- **分类: cs.CV**

- **简介: 该论文属于视频时间定位任务，旨在解决如何利用多模态大语言模型提升视频细粒度理解。工作包括系统梳理现有方法、分析其功能角色、训练策略及视频特征处理技术。**

- **链接: [http://arxiv.org/pdf/2508.10922v1](http://arxiv.org/pdf/2508.10922v1)**

> **作者:** Jianlong Wu; Wei Liu; Ye Liu; Meng Liu; Liqiang Nie; Zhouchen Lin; Chang Wen Chen
>
> **备注:** 20 pages,6 figures,survey
>
> **摘要:** The recent advancement in video temporal grounding (VTG) has significantly enhanced fine-grained video understanding, primarily driven by multimodal large language models (MLLMs). With superior multimodal comprehension and reasoning abilities, VTG approaches based on MLLMs (VTG-MLLMs) are gradually surpassing traditional fine-tuned methods. They not only achieve competitive performance but also excel in generalization across zero-shot, multi-task, and multi-domain settings. Despite extensive surveys on general video-language understanding, comprehensive reviews specifically addressing VTG-MLLMs remain scarce. To fill this gap, this survey systematically examines current research on VTG-MLLMs through a three-dimensional taxonomy: 1) the functional roles of MLLMs, highlighting their architectural significance; 2) training paradigms, analyzing strategies for temporal reasoning and task adaptation; and 3) video feature processing techniques, which determine spatiotemporal representation effectiveness. We further discuss benchmark datasets, evaluation protocols, and summarize empirical findings. Finally, we identify existing limitations and propose promising research directions. For additional resources and details, readers are encouraged to visit our repository at https://github.com/ki-lw/Awesome-MLLMs-for-Video-Temporal-Grounding.
>
---
#### [new 003] A Coarse-to-Fine Human Pose Estimation Method based on Two-stage Distillation and Progressive Graph Neural Network
- **分类: cs.CV**

- **简介: 该论文属于人体姿态估计任务，旨在解决现有方法计算资源消耗大、泛化能力不足的问题。通过两阶段知识蒸馏和渐进式图神经网络提升模型精度与轻量化水平。**

- **链接: [http://arxiv.org/pdf/2508.11212v1](http://arxiv.org/pdf/2508.11212v1)**

> **作者:** Zhangjian Ji; Wenjin Zhang; Shaotong Qiao; Kai Feng; Yuhua Qian
>
> **摘要:** Human pose estimation has been widely applied in the human-centric understanding and generation, but most existing state-of-the-art human pose estimation methods require heavy computational resources for accurate predictions. In order to obtain an accurate, robust yet lightweight human pose estimator, one feasible way is to transfer pose knowledge from a powerful teacher model to a less-parameterized student model by knowledge distillation. However, the traditional knowledge distillation framework does not fully explore the contextual information among human joints. Thus, in this paper, we propose a novel coarse-to-fine two-stage knowledge distillation framework for human pose estimation. In the first-stage distillation, we introduce the human joints structure loss to mine the structural information among human joints so as to transfer high-level semantic knowledge from the teacher model to the student model. In the second-stage distillation, we utilize an Image-Guided Progressive Graph Convolutional Network (IGP-GCN) to refine the initial human pose obtained from the first-stage distillation and supervise the training of the IGP-GCN in the progressive way by the final output pose of teacher model. The extensive experiments on the benchmark dataset: COCO keypoint and CrowdPose datasets, show that our proposed method performs favorably against lots of the existing state-of-the-art human pose estimation methods, especially for the more complex CrowdPose dataset, the performance improvement of our model is more significant.
>
---
#### [new 004] HOID-R1: Reinforcement Learning for Open-World Human-Object Interaction Detection Reasoning with Multimodal Large Language Model
- **分类: cs.CV**

- **简介: 该论文属于人-物交互检测任务，解决开放世界下模型依赖文本忽略空间信息的问题。通过强化学习融合思维链和多模态优化，提升检测精度与泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.11350v1](http://arxiv.org/pdf/2508.11350v1)**

> **作者:** Zhenhao Zhang; Hanqing Wang; Xiangyu Zeng; Ziyu Cheng; Jiaxin Liu; Haoyu Yan; Zhirui Liu; Kaiyang Ji; Tianxiang Gui; Ke Hu; Kangyi Chen; Yahao Fan; Mokai Pan
>
> **摘要:** Understanding and recognizing human-object interaction (HOI) is a pivotal application in AR/VR and robotics. Recent open-vocabulary HOI detection approaches depend exclusively on large language models for richer textual prompts, neglecting their inherent 3D spatial understanding capabilities. To address this shortcoming, we introduce HOID-R1, the first HOI detection framework that integrates chain-of-thought (CoT) guided supervised fine-tuning (SFT) with group relative policy optimization (GRPO) within a reinforcement learning (RL) paradigm. Specifically, we initially apply SFT to imbue the model with essential reasoning capabilities, forcing the model to articulate its thought process in the output. Subsequently, we integrate GRPO to leverage multi-reward signals for policy optimization, thereby enhancing alignment across diverse modalities. To mitigate hallucinations in the CoT reasoning, we introduce an "MLLM-as-a-judge" mechanism that supervises the CoT outputs, further improving generalization. Extensive experiments show that HOID-R1 achieves state-of-the-art performance on HOI detection benchmarks and outperforms existing methods in open-world generalization to novel scenarios.
>
---
#### [new 005] Leveraging the RETFound foundation model for optic disc segmentation in retinal images
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文将RETFound模型应用于视盘分割任务，解决 retinal 图像分析中的基础问题。通过微调获得优异性能，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.11354v1](http://arxiv.org/pdf/2508.11354v1)**

> **作者:** Zhenyi Zhao; Muthu Rama Krishnan Mookiah; Emanuele Trucco
>
> **摘要:** RETFound is a well-known foundation model (FM) developed for fundus camera and optical coherence tomography images. It has shown promising performance across multiple datasets in diagnosing diseases, both eye-specific and systemic, from retinal images. However, to our best knowledge, it has not been used for other tasks. We present the first adaptation of RETFound for optic disc segmentation, a ubiquitous and foundational task in retinal image analysis. The resulting segmentation system outperforms state-of-the-art, segmentation-specific baseline networks after training a head with only a very modest number of task-specific examples. We report and discuss results with four public datasets, IDRID, Drishti-GS, RIM-ONE-r3, and REFUGE, and a private dataset, GoDARTS, achieving about 96% Dice consistently across all datasets. Overall, our method obtains excellent performance in internal verification, domain generalization and domain adaptation, and exceeds most of the state-of-the-art baseline results. We discuss the results in the framework of the debate about FMs as alternatives to task-specific architectures. The code is available at: [link to be added after the paper is accepted]
>
---
#### [new 006] Not There Yet: Evaluating Vision Language Models in Simulating the Visual Perception of People with Low Vision
- **分类: cs.CV; cs.AI; cs.HC**

- **简介: 该论文属于视觉-语言模型评估任务，旨在解决模拟低视力人群视觉感知的问题。通过构建数据集并测试模型生成响应的准确性，发现结合多种信息可提升模拟效果。**

- **链接: [http://arxiv.org/pdf/2508.10972v1](http://arxiv.org/pdf/2508.10972v1)**

> **作者:** Rosiana Natalie; Wenqian Xu; Ruei-Che Chang; Rada Mihalcea; Anhong Guo
>
> **摘要:** Advances in vision language models (VLMs) have enabled the simulation of general human behavior through their reasoning and problem solving capabilities. However, prior research has not investigated such simulation capabilities in the accessibility domain. In this paper, we evaluate the extent to which VLMs can simulate the vision perception of low vision individuals when interpreting images. We first compile a benchmark dataset through a survey study with 40 low vision participants, collecting their brief and detailed vision information and both open-ended and multiple-choice image perception and recognition responses to up to 25 images. Using these responses, we construct prompts for VLMs (GPT-4o) to create simulated agents of each participant, varying the included information on vision information and example image responses. We evaluate the agreement between VLM-generated responses and participants' original answers. Our results indicate that VLMs tend to infer beyond the specified vision ability when given minimal prompts, resulting in low agreement (0.59). The agreement between the agent' and participants' responses remains low when only either the vision information (0.59) or example image responses (0.59) are provided, whereas a combination of both significantly increase the agreement (0.70, p < 0.0001). Notably, a single example combining both open-ended and multiple-choice responses, offers significant performance improvements over either alone (p < 0.0001), while additional examples provided minimal benefits (p > 0.05).
>
---
#### [new 007] Domain-aware Category-level Geometry Learning Segmentation for 3D Point Clouds
- **分类: cs.CV**

- **简介: 该论文属于3D点云语义分割任务，解决域泛化问题。通过引入类别级几何嵌入和几何一致性学习，提升模型在未见环境中的泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.11265v1](http://arxiv.org/pdf/2508.11265v1)**

> **作者:** Pei He; Lingling Li; Licheng Jiao; Ronghua Shang; Fang Liu; Shuang Wang; Xu Liu; Wenping Ma
>
> **备注:** to be published in International Conference on Computer Vision, ICCV 2025
>
> **摘要:** Domain generalization in 3D segmentation is a critical challenge in deploying models to unseen environments. Current methods mitigate the domain shift by augmenting the data distribution of point clouds. However, the model learns global geometric patterns in point clouds while ignoring the category-level distribution and alignment. In this paper, a category-level geometry learning framework is proposed to explore the domain-invariant geometric features for domain generalized 3D semantic segmentation. Specifically, Category-level Geometry Embedding (CGE) is proposed to perceive the fine-grained geometric properties of point cloud features, which constructs the geometric properties of each class and couples geometric embedding to semantic learning. Secondly, Geometric Consistent Learning (GCL) is proposed to simulate the latent 3D distribution and align the category-level geometric embeddings, allowing the model to focus on the geometric invariant information to improve generalization. Experimental results verify the effectiveness of the proposed method, which has very competitive segmentation accuracy compared with the state-of-the-art domain generalized point cloud methods.
>
---
#### [new 008] RMFAT: Recurrent Multi-scale Feature Atmospheric Turbulence Mitigator
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11409v1](http://arxiv.org/pdf/2508.11409v1)**

> **作者:** Zhiming Liu; Nantheera Anantrasirichai
>
> **摘要:** Atmospheric turbulence severely degrades video quality by introducing distortions such as geometric warping, blur, and temporal flickering, posing significant challenges to both visual clarity and temporal consistency. Current state-of-the-art methods are based on transformer and 3D architectures and require multi-frame input, but their large computational cost and memory usage limit real-time deployment, especially in resource-constrained scenarios. In this work, we propose RMFAT: Recurrent Multi-scale Feature Atmospheric Turbulence Mitigator, designed for efficient and temporally consistent video restoration under AT conditions. RMFAT adopts a lightweight recurrent framework that restores each frame using only two inputs at a time, significantly reducing temporal window size and computational burden. It further integrates multi-scale feature encoding and decoding with temporal warping modules at both encoder and decoder stages to enhance spatial detail and temporal coherence. Extensive experiments on synthetic and real-world atmospheric turbulence datasets demonstrate that RMFAT not only outperforms existing methods in terms of clarity restoration (with nearly a 9\% improvement in SSIM) but also achieves significantly improved inference speed (more than a fourfold reduction in runtime), making it particularly suitable for real-time atmospheric turbulence suppression tasks.
>
---
#### [new 009] Unified Knowledge Distillation Framework: Fine-Grained Alignment and Geometric Relationship Preservation for Deep Face Recognition
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于人脸识别任务，解决知识蒸馏中细节与关系信息丢失的问题，提出统一框架融合两种新损失函数，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2508.11376v1](http://arxiv.org/pdf/2508.11376v1)**

> **作者:** Durgesh Mishra; Rishabh Uikey
>
> **备注:** The paper spans a total of 14 pages, 10 pages for the main content (including references) and 4 pages for the appendix. The main paper contains 3 figures and 1 table, while the appendix includes 1 pseudo-code algorithm and 4 tables. The work was recently accepted for publication at IJCB 2025
>
> **摘要:** Knowledge Distillation is crucial for optimizing face recognition models for deployment in computationally limited settings, such as edge devices. Traditional KD methods, such as Raw L2 Feature Distillation or Feature Consistency loss, often fail to capture both fine-grained instance-level details and complex relational structures, leading to suboptimal performance. We propose a unified approach that integrates two novel loss functions, Instance-Level Embedding Distillation and Relation-Based Pairwise Similarity Distillation. Instance-Level Embedding Distillation focuses on aligning individual feature embeddings by leveraging a dynamic hard mining strategy, thereby enhancing learning from challenging examples. Relation-Based Pairwise Similarity Distillation captures relational information through pairwise similarity relationships, employing a memory bank mechanism and a sample mining strategy. This unified framework ensures both effective instance-level alignment and preservation of geometric relationships between samples, leading to a more comprehensive distillation process. Our unified framework outperforms state-of-the-art distillation methods across multiple benchmark face recognition datasets, as demonstrated by extensive experimental evaluations. Interestingly, when using strong teacher networks compared to the student, our unified KD enables the student to even surpass the teacher's accuracy.
>
---
#### [new 010] Is ChatGPT-5 Ready for Mammogram VQA?
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学影像分析任务，旨在评估GPT-5在乳腺X光片VQA中的表现，解决其在临床应用中的准确性不足问题。**

- **链接: [http://arxiv.org/pdf/2508.11628v1](http://arxiv.org/pdf/2508.11628v1)**

> **作者:** Qiang Li; Shansong Wang; Mingzhe Hu; Mojtaba Safari; Zachary Eidex; Xiaofeng Yang
>
> **摘要:** Mammogram visual question answering (VQA) integrates image interpretation with clinical reasoning and has potential to support breast cancer screening. We systematically evaluated the GPT-5 family and GPT-4o model on four public mammography datasets (EMBED, InBreast, CMMD, CBIS-DDSM) for BI-RADS assessment, abnormality detection, and malignancy classification tasks. GPT-5 consistently was the best performing model but lagged behind both human experts and domain-specific fine-tuned models. On EMBED, GPT-5 achieved the highest scores among GPT variants in density (56.8%), distortion (52.5%), mass (64.5%), calcification (63.5%), and malignancy (52.8%) classification. On InBreast, it attained 36.9% BI-RADS accuracy, 45.9% abnormality detection, and 35.0% malignancy classification. On CMMD, GPT-5 reached 32.3% abnormality detection and 55.0% malignancy accuracy. On CBIS-DDSM, it achieved 69.3% BI-RADS accuracy, 66.0% abnormality detection, and 58.2% malignancy accuracy. Compared with human expert estimations, GPT-5 exhibited lower sensitivity (63.5%) and specificity (52.3%). While GPT-5 exhibits promising capabilities for screening tasks, its performance remains insufficient for high-stakes clinical imaging applications without targeted domain adaptation and optimization. However, the tremendous improvements in performance from GPT-4o to GPT-5 show a promising trend in the potential for general large language models (LLMs) to assist with mammography VQA tasks.
>
---
#### [new 011] LoRAtorio: An intrinsic approach to LoRA Skill Composition
- **分类: cs.CV**

- **简介: 该论文属于文本生成任务，解决多LoRA适配器组合问题。提出LoRAtorio框架，通过潜在空间分析和权重聚合提升组合效果。**

- **链接: [http://arxiv.org/pdf/2508.11624v1](http://arxiv.org/pdf/2508.11624v1)**

> **作者:** Niki Foteinopoulou; Ignas Budvytis; Stephan Liwicki
>
> **备注:** 32 pages, 17 figures
>
> **摘要:** Low-Rank Adaptation (LoRA) has become a widely adopted technique in text-to-image diffusion models, enabling the personalisation of visual concepts such as characters, styles, and objects. However, existing approaches struggle to effectively compose multiple LoRA adapters, particularly in open-ended settings where the number and nature of required skills are not known in advance. In this work, we present LoRAtorio, a novel train-free framework for multi-LoRA composition that leverages intrinsic model behaviour. Our method is motivated by two key observations: (1) LoRA adapters trained on narrow domains produce denoised outputs that diverge from the base model, and (2) when operating out-of-distribution, LoRA outputs show behaviour closer to the base model than when conditioned in distribution. The balance between these two observations allows for exceptional performance in the single LoRA scenario, which nevertheless deteriorates when multiple LoRAs are loaded. Our method operates in the latent space by dividing it into spatial patches and computing cosine similarity between each patch's predicted noise and that of the base model. These similarities are used to construct a spatially-aware weight matrix, which guides a weighted aggregation of LoRA outputs. To address domain drift, we further propose a modification to classifier-free guidance that incorporates the base model's unconditional score into the composition. We extend this formulation to a dynamic module selection setting, enabling inference-time selection of relevant LoRA adapters from a large pool. LoRAtorio achieves state-of-the-art performance, showing up to a 1.3% improvement in ClipScore and a 72.43% win rate in GPT-4V pairwise evaluations, and generalises effectively to multiple latent diffusion models.
>
---
#### [new 012] Probing the Representational Power of Sparse Autoencoders in Vision Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究稀疏自编码器在视觉模型中的表示能力，解决其在视觉领域应用不足的问题，通过实验验证其在语义理解、泛化和可控生成方面的有效性。**

- **链接: [http://arxiv.org/pdf/2508.11277v1](http://arxiv.org/pdf/2508.11277v1)**

> **作者:** Matthew Lyle Olson; Musashi Hinck; Neale Ratzlaff; Changbai Li; Phillip Howard; Vasudev Lal; Shao-Yen Tseng
>
> **备注:** ICCV 2025 Findings
>
> **摘要:** Sparse Autoencoders (SAEs) have emerged as a popular tool for interpreting the hidden states of large language models (LLMs). By learning to reconstruct activations from a sparse bottleneck layer, SAEs discover interpretable features from the high-dimensional internal representations of LLMs. Despite their popularity with language models, SAEs remain understudied in the visual domain. In this work, we provide an extensive evaluation the representational power of SAEs for vision models using a broad range of image-based tasks. Our experimental results demonstrate that SAE features are semantically meaningful, improve out-of-distribution generalization, and enable controllable generation across three vision model architectures: vision embedding models, multi-modal LMMs and diffusion models. In vision embedding models, we find that learned SAE features can be used for OOD detection and provide evidence that they recover the ontological structure of the underlying model. For diffusion models, we demonstrate that SAEs enable semantic steering through text encoder manipulation and develop an automated pipeline for discovering human-interpretable attributes. Finally, we conduct exploratory experiments on multi-modal LLMs, finding evidence that SAE features reveal shared representations across vision and language modalities. Our study provides a foundation for SAE evaluation in vision models, highlighting their strong potential improving interpretability, generalization, and steerability in the visual domain.
>
---
#### [new 013] Controlling Multimodal LLMs via Reward-guided Decoding
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态大模型控制任务，旨在提升模型的视觉定位能力。通过奖励引导解码方法，实现对模型输出精度和召回率的动态控制。**

- **链接: [http://arxiv.org/pdf/2508.11616v1](http://arxiv.org/pdf/2508.11616v1)**

> **作者:** Oscar Mañas; Pierluca D'Oro; Koustuv Sinha; Adriana Romero-Soriano; Michal Drozdzal; Aishwarya Agrawal
>
> **备注:** Published at ICCV 2025
>
> **摘要:** As Multimodal Large Language Models (MLLMs) gain widespread applicability, it is becoming increasingly desirable to adapt them for diverse user needs. In this paper, we study the adaptation of MLLMs through controlled decoding. To achieve this, we introduce the first method for reward-guided decoding of MLLMs and demonstrate its application in improving their visual grounding. Our method involves building reward models for visual grounding and using them to guide the MLLM's decoding process. Concretely, we build two separate reward models to independently control the degree of object precision and recall in the model's output. Our approach enables on-the-fly controllability of an MLLM's inference process in two ways: first, by giving control over the relative importance of each reward function during decoding, allowing a user to dynamically trade off object precision for recall in image captioning tasks; second, by giving control over the breadth of the search during decoding, allowing the user to control the trade-off between the amount of test-time compute and the degree of visual grounding. We evaluate our method on standard object hallucination benchmarks, showing that it provides significant controllability over MLLM inference, while consistently outperforming existing hallucination mitigation methods.
>
---
#### [new 014] UWB-PostureGuard: A Privacy-Preserving RF Sensing System for Continuous Ergonomic Sitting Posture Monitoring
- **分类: cs.CV; cs.HC; eess.SP**

- **简介: 该论文属于姿态监测任务，解决传统方法隐私和舒适性问题。通过UWB技术实现无接触、持续的坐姿监控，提升健康管理水平。**

- **链接: [http://arxiv.org/pdf/2508.11115v1](http://arxiv.org/pdf/2508.11115v1)**

> **作者:** Haotang Li; Zhenyu Qi; Sen He; Kebin Peng; Sheng Tan; Yili Ren; Tomas Cerny; Jiyue Zhao; Zi Wang
>
> **摘要:** Improper sitting posture during prolonged computer use has become a significant public health concern. Traditional posture monitoring solutions face substantial barriers, including privacy concerns with camera-based systems and user discomfort with wearable sensors. This paper presents UWB-PostureGuard, a privacy-preserving ultra-wideband (UWB) sensing system that advances mobile technologies for preventive health management through continuous, contactless monitoring of ergonomic sitting posture. Our system leverages commercial UWB devices, utilizing comprehensive feature engineering to extract multiple ergonomic sitting posture features. We develop PoseGBDT to effectively capture temporal dependencies in posture patterns, addressing limitations of traditional frame-wise classification approaches. Extensive real-world evaluation across 10 participants and 19 distinct postures demonstrates exceptional performance, achieving 99.11% accuracy while maintaining robustness against environmental variables such as clothing thickness, additional devices, and furniture configurations. Our system provides a scalable, privacy-preserving mobile health solution on existing platforms for proactive ergonomic management, improving quality of life at low costs.
>
---
#### [new 015] VFM-Guided Semi-Supervised Detection Transformer for Source-Free Object Detection in Remote Sensing Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11167v1](http://arxiv.org/pdf/2508.11167v1)**

> **作者:** Jianhong Han; Yupei Wang; Liang Chen
>
> **备注:** Manuscript submitted to IEEE TGRS
>
> **摘要:** Unsupervised domain adaptation methods have been widely explored to bridge domain gaps. However, in real-world remote-sensing scenarios, privacy and transmission constraints often preclude access to source domain data, which limits their practical applicability. Recently, Source-Free Object Detection (SFOD) has emerged as a promising alternative, aiming at cross-domain adaptation without relying on source data, primarily through a self-training paradigm. Despite its potential, SFOD frequently suffers from training collapse caused by noisy pseudo-labels, especially in remote sensing imagery with dense objects and complex backgrounds. Considering that limited target domain annotations are often feasible in practice, we propose a Vision foundation-Guided DEtection TRansformer (VG-DETR), built upon a semi-supervised framework for SFOD in remote sensing images. VG-DETR integrates a Vision Foundation Model (VFM) into the training pipeline in a "free lunch" manner, leveraging a small amount of labeled target data to mitigate pseudo-label noise while improving the detector's feature-extraction capability. Specifically, we introduce a VFM-guided pseudo-label mining strategy that leverages the VFM's semantic priors to further assess the reliability of the generated pseudo-labels. By recovering potentially correct predictions from low-confidence outputs, our strategy improves pseudo-label quality and quantity. In addition, a dual-level VFM-guided alignment method is proposed, which aligns detector features with VFM embeddings at both the instance and image levels. Through contrastive learning among fine-grained prototypes and similarity matching between feature maps, this dual-level alignment further enhances the robustness of feature representations against domain gaps. Extensive experiments demonstrate that VG-DETR achieves superior performance in source-free remote sensing detection tasks.
>
---
#### [new 016] Thyme: Think Beyond Images
- **分类: cs.CV**

- **简介: 该论文提出Thyme，解决多模态大模型在图像处理与逻辑推理结合不足的问题，通过代码生成实现图像操作和数学计算，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2508.11630v1](http://arxiv.org/pdf/2508.11630v1)**

> **作者:** Yi-Fan Zhang; Xingyu Lu; Shukang Yin; Chaoyou Fu; Wei Chen; Xiao Hu; Bin Wen; Kaiyu Jiang; Changyi Liu; Tianke Zhang; Haonan Fan; Kaibing Chen; Jiankang Chen; Haojie Ding; Kaiyu Tang; Zhang Zhang; Liang Wang; Fan Yang; Tingting Gao; Guorui Zhou
>
> **备注:** Project page: https://thyme-vl.github.io/
>
> **摘要:** Following OpenAI's introduction of the ``thinking with images'' concept, recent efforts have explored stimulating the use of visual information in the reasoning process to enhance model performance in perception and reasoning tasks. However, to the best of our knowledge, no open-source work currently offers a feature set as rich as proprietary models (O3), which can perform diverse image manipulations and simultaneously enhance logical reasoning capabilities through code. In this paper, we make a preliminary attempt in this direction by introducing Thyme (Think Beyond Images), a novel paradigm for enabling MLLMs to transcend existing ``think with images'' approaches by autonomously generating and executing diverse image processing and computational operations via executable code. This approach not only facilitates a rich, on-the-fly set of image manipulations (e.g., cropping, rotation, contrast enhancement) but also allows for mathematical computations, all while maintaining high autonomy in deciding when and how to apply these operations. We activate this capability through a two-stage training strategy: an initial SFT on a curated dataset of 500K samples to teach code generation, followed by a RL phase to refine decision-making. For the RL stage, we manually collect and design high-resolution question-answer pairs to increase the learning difficulty, and we propose GRPO-ATS (Group Relative Policy Optimization with Adaptive Temperature Sampling), an algorithm that applies distinct temperatures to text and code generation to balance reasoning exploration with code execution precision. We conduct extensive experimental analysis and ablation studies. Comprehensive evaluations on nearly 20 benchmarks show that Thyme yields significant and consistent performance gains, particularly in challenging high-resolution perception and complex reasoning tasks.
>
---
#### [new 017] Handwritten Text Recognition of Historical Manuscripts Using Transformer-Based Models
- **分类: cs.CV; cs.AI; cs.DL; cs.LG**

- **简介: 该论文属于历史手写文本识别任务，旨在解决稀有转录、语言变化和多样手写风格的问题。通过改进预处理和数据增强技术，提升了模型性能。**

- **链接: [http://arxiv.org/pdf/2508.11499v1](http://arxiv.org/pdf/2508.11499v1)**

> **作者:** Erez Meoded
>
> **摘要:** Historical handwritten text recognition (HTR) is essential for unlocking the cultural and scholarly value of archival documents, yet digitization is often hindered by scarce transcriptions, linguistic variation, and highly diverse handwriting styles. In this study, we apply TrOCR, a state-of-the-art transformer-based HTR model, to 16th-century Latin manuscripts authored by Rudolf Gwalther. We investigate targeted image preprocessing and a broad suite of data augmentation techniques, introducing four novel augmentation methods designed specifically for historical handwriting characteristics. We also evaluate ensemble learning approaches to leverage the complementary strengths of augmentation-trained models. On the Gwalther dataset, our best single-model augmentation (Elastic) achieves a Character Error Rate (CER) of 1.86, while a top-5 voting ensemble achieves a CER of 1.60 - representing a 50% relative improvement over the best reported TrOCR_BASE result and a 42% improvement over the previous state of the art. These results highlight the impact of domain-specific augmentations and ensemble strategies in advancing HTR performance for historical manuscripts.
>
---
#### [new 018] Exploring the Tradeoff Between Diversity and Discrimination for Continuous Category Discovery
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11173v1](http://arxiv.org/pdf/2508.11173v1)**

> **作者:** Ruobing Jiang; Yang Liu; Haobing Liu; Yanwei Yu; Chunyang Wang
>
> **备注:** Accepted by CIKM 2025. 10 pages, 5 figures,
>
> **摘要:** Continuous category discovery (CCD) aims to automatically discover novel categories in continuously arriving unlabeled data. This is a challenging problem considering that there is no number of categories and labels in the newly arrived data, while also needing to mitigate catastrophic forgetting. Most CCD methods cannot handle the contradiction between novel class discovery and classification well. They are also prone to accumulate errors in the process of gradually discovering novel classes. Moreover, most of them use knowledge distillation and data replay to prevent forgetting, occupying more storage space. To address these limitations, we propose Independence-based Diversity and Orthogonality-based Discrimination (IDOD). IDOD mainly includes independent enrichment of diversity module, joint discovery of novelty module, and continuous increment by orthogonality module. In independent enrichment, the backbone is trained separately using contrastive loss to avoid it focusing only on features for classification. Joint discovery transforms multi-stage novel class discovery into single-stage, reducing error accumulation impact. Continuous increment by orthogonality module generates mutually orthogonal prototypes for classification and prevents forgetting with lower space overhead via representative representation replay. Experimental results show that on challenging fine-grained datasets, our method outperforms the state-of-the-art methods.
>
---
#### [new 019] GANDiff FR: Hybrid GAN Diffusion Synthesis for Causal Bias Attribution in Face Recognition
- **分类: cs.CV**

- **简介: 该论文属于人脸识别中的公平性审计任务，旨在解决模型偏见问题。通过混合GAN与扩散模型生成可控人脸数据，分析并减少识别偏差。**

- **链接: [http://arxiv.org/pdf/2508.11334v1](http://arxiv.org/pdf/2508.11334v1)**

> **作者:** Md Asgor Hossain Reaj; Rajan Das Gupta; Md Yeasin Rahat; Nafiz Fahad; Md Jawadul Hasan; Tze Hui Liew
>
> **备注:** Accepted in ICCVDM '25
>
> **摘要:** We introduce GANDiff FR, the first synthetic framework that precisely controls demographic and environmental factors to measure, explain, and reduce bias with reproducible rigor. GANDiff FR unifies StyleGAN3-based identity-preserving generation with diffusion-based attribute control, enabling fine-grained manipulation of pose around 30 degrees, illumination (four directions), and expression (five levels) under ceteris paribus conditions. We synthesize 10,000 demographically balanced faces across five cohorts validated for realism via automated detection (98.2%) and human review (89%) to isolate and quantify bias drivers. Benchmarking ArcFace, CosFace, and AdaFace under matched operating points shows AdaFace reduces inter-group TPR disparity by 60% (2.5% vs. 6.3%), with illumination accounting for 42% of residual bias. Cross-dataset evaluation on RFW, BUPT, and CASIA WebFace confirms strong synthetic-to-real transfer (r 0.85). Despite around 20% computational overhead relative to pure GANs, GANDiff FR yields three times more attribute-conditioned variants, establishing a reproducible, regulation-aligned (EU AI Act) standard for fairness auditing. Code and data are released to support transparent, scalable bias evaluation.
>
---
#### [new 020] Generalized Decoupled Learning for Enhancing Open-Vocabulary Dense Perception
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于开放词汇密集感知任务，旨在解决视觉模型在未定义类别上的性能不足问题。通过分离内容与上下文特征，提升局部区分性和空间一致性。**

- **链接: [http://arxiv.org/pdf/2508.11256v1](http://arxiv.org/pdf/2508.11256v1)**

> **作者:** Junjie Wang; Keyu Chen; Yulin Li; Bin Chen; Hengshuang Zhao; Xiaojuan Qi; Zhuotao Tian
>
> **备注:** arXiv admin note: text overlap with arXiv:2505.04410
>
> **摘要:** Dense visual perception tasks have been constrained by their reliance on predefined categories, limiting their applicability in real-world scenarios where visual concepts are unbounded. While Vision-Language Models (VLMs) like CLIP have shown promise in open-vocabulary tasks, their direct application to dense perception often leads to suboptimal performance due to limitations in local feature representation. In this work, we present our observation that CLIP's image tokens struggle to effectively aggregate information from spatially or semantically related regions, resulting in features that lack local discriminability and spatial consistency. To address this issue, we propose DeCLIP, a novel framework that enhances CLIP by decoupling the self-attention module to obtain ``content'' and ``context'' features respectively. \revise{The context features are enhanced by jointly distilling semantic correlations from Vision Foundation Models (VFMs) and object integrity cues from diffusion models, thereby enhancing spatial consistency. In parallel, the content features are aligned with image crop representations and constrained by region correlations from VFMs to improve local discriminability. Extensive experiments demonstrate that DeCLIP establishes a solid foundation for open-vocabulary dense perception, consistently achieving state-of-the-art performance across a broad spectrum of tasks, including 2D detection and segmentation, 3D instance segmentation, video instance segmentation, and 6D object pose estimation.} Code is available at https://github.com/xiaomoguhz/DeCLIP
>
---
#### [new 021] CoreEditor: Consistent 3D Editing via Correspondence-constrained Diffusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11603v1](http://arxiv.org/pdf/2508.11603v1)**

> **作者:** Zhe Zhu; Honghua Chen; Peng Li; Mingqiang Wei
>
> **摘要:** Text-driven 3D editing seeks to modify 3D scenes according to textual descriptions, and most existing approaches tackle this by adapting pre-trained 2D image editors to multi-view inputs. However, without explicit control over multi-view information exchange, they often fail to maintain cross-view consistency, leading to insufficient edits and blurry details. We introduce CoreEditor, a novel framework for consistent text-to-3D editing. The key innovation is a correspondence-constrained attention mechanism that enforces precise interactions between pixels expected to remain consistent throughout the diffusion denoising process. Beyond relying solely on geometric alignment, we further incorporate semantic similarity estimated during denoising, enabling more reliable correspondence modeling and robust multi-view editing. In addition, we design a selective editing pipeline that allows users to choose preferred results from multiple candidates, offering greater flexibility and user control. Extensive experiments show that CoreEditor produces high-quality, 3D-consistent edits with sharper details, significantly outperforming prior methods.
>
---
#### [new 022] Training-Free Anomaly Generation via Dual-Attention Enhancement in Diffusion Model
- **分类: cs.CV**

- **简介: 该论文属于工业异常检测任务，解决数据稀缺问题。提出AAG框架，无需训练即可生成高质量异常图像，提升下游检测性能。**

- **链接: [http://arxiv.org/pdf/2508.11550v1](http://arxiv.org/pdf/2508.11550v1)**

> **作者:** Zuo Zuo; Jiahao Dong; Yanyun Qu; Zongze Wu
>
> **摘要:** Industrial anomaly detection (AD) plays a significant role in manufacturing where a long-standing challenge is data scarcity. A growing body of works have emerged to address insufficient anomaly data via anomaly generation. However, these anomaly generation methods suffer from lack of fidelity or need to be trained with extra data. To this end, we propose a training-free anomaly generation framework dubbed AAG, which is based on Stable Diffusion (SD)'s strong generation ability for effective anomaly image generation. Given a normal image, mask and a simple text prompt, AAG can generate realistic and natural anomalies in the specific regions and simultaneously keep contents in other regions unchanged. In particular, we propose Cross-Attention Enhancement (CAE) to re-engineer the cross-attention mechanism within Stable Diffusion based on the given mask. CAE increases the similarity between visual tokens in specific regions and text embeddings, which guides these generated visual tokens in accordance with the text description. Besides, generated anomalies need to be more natural and plausible with object in given image. We propose Self-Attention Enhancement (SAE) which improves similarity between each normal visual token and anomaly visual tokens. SAE ensures that generated anomalies are coherent with original pattern. Extensive experiments on MVTec AD and VisA datasets demonstrate effectiveness of AAG in anomaly generation and its utility. Furthermore, anomaly images generated by AAG can bolster performance of various downstream anomaly inspection tasks.
>
---
#### [new 023] Data-Driven Deepfake Image Detection Method -- The 2024 Global Deepfake Image Detection Challenge
- **分类: cs.CV**

- **简介: 该论文属于深度伪造图像检测任务，旨在识别人脸图像是否为深度伪造，并输出概率。采用Swin Transformer V2-B网络结合数据增强方法提升模型性能。**

- **链接: [http://arxiv.org/pdf/2508.11464v1](http://arxiv.org/pdf/2508.11464v1)**

> **作者:** Xiaoya Zhu; Yibing Nan; Shiguo Lian
>
> **摘要:** With the rapid development of technology in the field of AI, deepfake technology has emerged as a double-edged sword. It has not only created a large amount of AI-generated content but also posed unprecedented challenges to digital security. The task of the competition is to determine whether a face image is a Deepfake image and output its probability score of being a Deepfake image. In the image track competition, our approach is based on the Swin Transformer V2-B classification network. And online data augmentation and offline sample generation methods are employed to enrich the diversity of training samples and increase the generalization ability of the model. Finally, we got the award of excellence in Deepfake image detection.
>
---
#### [new 024] Generating Dialogues from Egocentric Instructional Videos for Task Assistance: Dataset, Method and Benchmark
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11192v1](http://arxiv.org/pdf/2508.11192v1)**

> **作者:** Lavisha Aggarwal; Vikas Bahirwani; Lin Li; Andrea Colaco
>
> **摘要:** Many everyday tasks ranging from fixing appliances, cooking recipes to car maintenance require expert knowledge, especially when tasks are complex and multi-step. Despite growing interest in AI agents, there is a scarcity of dialogue-video datasets grounded for real world task assistance. In this paper, we propose a simple yet effective approach that transforms single-person instructional videos into task-guidance two-person dialogues, aligned with fine grained steps and video-clips. Our fully automatic approach, powered by large language models, offers an efficient alternative to the substantial cost and effort required for human-assisted data collection. Using this technique, we build HowToDIV, a large-scale dataset containing 507 conversations, 6636 question-answer pairs and 24 hours of videoclips across diverse tasks in cooking, mechanics, and planting. Each session includes multi-turn conversation where an expert teaches a novice user how to perform a task step by step, while observing user's surrounding through a camera and microphone equipped wearable device. We establish the baseline benchmark performance on HowToDIV dataset through Gemma-3 model for future research on this new task of dialogues for procedural-task assistance.
>
---
#### [new 025] Noise Matters: Optimizing Matching Noise for Diffusion Classifiers
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，解决扩散分类器的噪声不稳定问题。通过优化噪声提升分类稳定性与速度。**

- **链接: [http://arxiv.org/pdf/2508.11330v1](http://arxiv.org/pdf/2508.11330v1)**

> **作者:** Yanghao Wang; Long Chen
>
> **摘要:** Although today's pretrained discriminative vision-language models (e.g., CLIP) have demonstrated strong perception abilities, such as zero-shot image classification, they also suffer from the bag-of-words problem and spurious bias. To mitigate these problems, some pioneering studies leverage powerful generative models (e.g., pretrained diffusion models) to realize generalizable image classification, dubbed Diffusion Classifier (DC). Specifically, by randomly sampling a Gaussian noise, DC utilizes the differences of denoising effects with different category conditions to classify categories. Unfortunately, an inherent and notorious weakness of existing DCs is noise instability: different random sampled noises lead to significant performance changes. To achieve stable classification performance, existing DCs always ensemble the results of hundreds of sampled noises, which significantly reduces the classification speed. To this end, we firstly explore the role of noise in DC, and conclude that: there are some ``good noises'' that can relieve the instability. Meanwhile, we argue that these good noises should meet two principles: Frequency Matching and Spatial Matching. Regarding both principles, we propose a novel Noise Optimization method to learn matching (i.e., good) noise for DCs: NoOp. For frequency matching, NoOp first optimizes a dataset-specific noise: Given a dataset and a timestep t, optimize one randomly initialized parameterized noise. For Spatial Matching, NoOp trains a Meta-Network that adopts an image as input and outputs image-specific noise offset. The sum of optimized noise and noise offset will be used in DC to replace random noise. Extensive ablations on various datasets demonstrated the effectiveness of NoOp.
>
---
#### [new 026] Advancing 3D Scene Understanding with MV-ScanQA Multi-View Reasoning Evaluation and TripAlign Pre-training Dataset
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于3D视觉语言任务，解决现有数据集缺乏多视角推理和上下文对齐的问题。提出MV-ScanQA和TripAlign数据集，并开发LEGO模型以提升多视角场景理解。**

- **链接: [http://arxiv.org/pdf/2508.11058v1](http://arxiv.org/pdf/2508.11058v1)**

> **作者:** Wentao Mo; Qingchao Chen; Yuxin Peng; Siyuan Huang; Yang Liu
>
> **备注:** Accepeted to ACM MM 25
>
> **摘要:** The advancement of 3D vision-language (3D VL) learning is hindered by several limitations in existing 3D VL datasets: they rarely necessitate reasoning beyond a close range of objects in single viewpoint, and annotations often link instructions to single objects, missing richer contextual alignments between multiple objects. This significantly curtails the development of models capable of deep, multi-view 3D scene understanding over distant objects. To address these challenges, we introduce MV-ScanQA, a novel 3D question answering dataset where 68% of questions explicitly require integrating information from multiple views (compared to less than 7% in existing datasets), thereby rigorously testing multi-view compositional reasoning. To facilitate the training of models for such demanding scenarios, we present TripAlign dataset, a large-scale and low-cost 2D-3D-language pre-training corpus containing 1M <2D view, set of 3D objects, text> triplets that explicitly aligns groups of contextually related objects with text, providing richer, view-grounded multi-object multimodal alignment signals than previous single-object annotations. We further develop LEGO, a baseline method for the multi-view reasoning challenge in MV-ScanQA, transferring knowledge from pre-trained 2D LVLMs to 3D domain with TripAlign. Empirically, LEGO pre-trained on TripAlign achieves state-of-the-art performance not only on the proposed MV-ScanQA, but also on existing benchmarks for 3D dense captioning and question answering. Datasets and code are available at https://matthewdm0816.github.io/tripalign-mvscanqa.
>
---
#### [new 027] iWatchRoad: Scalable Detection and Geospatial Visualization of Potholes for Smart Cities
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于智能交通任务，解决道路坑洼检测与可视化问题。通过YOLO模型和GPS技术实现自动检测、定位并实时映射坑洼，提升道路维护效率。**

- **链接: [http://arxiv.org/pdf/2508.10945v1](http://arxiv.org/pdf/2508.10945v1)**

> **作者:** Rishi Raj Sahoo; Surbhi Saswati Mohanty; Subhankar Mishra
>
> **备注:** Under review
>
> **摘要:** Potholes on the roads are a serious hazard and maintenance burden. This poses a significant threat to road safety and vehicle longevity, especially on the diverse and under-maintained roads of India. In this paper, we present a complete end-to-end system called iWatchRoad for automated pothole detection, Global Positioning System (GPS) tagging, and real time mapping using OpenStreetMap (OSM). We curated a large, self-annotated dataset of over 7,000 frames captured across various road types, lighting conditions, and weather scenarios unique to Indian environments, leveraging dashcam footage. This dataset is used to fine-tune, Ultralytics You Only Look Once (YOLO) model to perform real time pothole detection, while a custom Optical Character Recognition (OCR) module was employed to extract timestamps directly from video frames. The timestamps are synchronized with GPS logs to geotag each detected potholes accurately. The processed data includes the potholes' details and frames as metadata is stored in a database and visualized via a user friendly web interface using OSM. iWatchRoad not only improves detection accuracy under challenging conditions but also provides government compatible outputs for road assessment and maintenance planning through the metadata visible on the website. Our solution is cost effective, hardware efficient, and scalable, offering a practical tool for urban and rural road management in developing regions, making the system automated. iWatchRoad is available at https://smlab.niser.ac.in/project/iwatchroad
>
---
#### [new 028] Residual-based Efficient Bidirectional Diffusion Model for Image Dehazing and Haze Generation
- **分类: cs.CV**

- **简介: 该论文属于图像去雾任务，解决现有方法仅能去雾而无法双向转换的问题。提出RBDM模型，通过残差和扩散机制实现高效双向转换。**

- **链接: [http://arxiv.org/pdf/2508.11134v1](http://arxiv.org/pdf/2508.11134v1)**

> **作者:** Bing Liu; Le Wang; Hao Liu; Mingming Liu
>
> **备注:** 7 pages, 5 figures, 2025 ICME Accepted
>
> **摘要:** Current deep dehazing methods only focus on removing haze from hazy images, lacking the capability to translate between hazy and haze-free images. To address this issue, we propose a residual-based efficient bidirectional diffusion model (RBDM) that can model the conditional distributions for both dehazing and haze generation. Firstly, we devise dual Markov chains that can effectively shift the residuals and facilitate bidirectional smooth transitions between them. Secondly, the RBDM perturbs the hazy and haze-free images at individual timesteps and predicts the noise in the perturbed data to simultaneously learn the conditional distributions. Finally, to enhance performance on relatively small datasets and reduce computational costs, our method introduces a unified score function learned on image patches instead of entire images. Our RBDM successfully implements size-agnostic bidirectional transitions between haze-free and hazy images with only 15 sampling steps. Extensive experiments demonstrate that the proposed method achieves superior or at least comparable performance to state-of-the-art methods on both synthetic and real-world datasets.
>
---
#### [new 029] LEARN: A Story-Driven Layout-to-Image Generation Framework for STEM Instruction
- **分类: cs.CV**

- **简介: 该论文属于STEM教育中的图像生成任务，旨在解决抽象科学概念可视化问题。通过布局引导和语义对齐生成连贯视觉序列，支持高层次思维训练。**

- **链接: [http://arxiv.org/pdf/2508.11153v1](http://arxiv.org/pdf/2508.11153v1)**

> **作者:** Maoquan Zhang; Bisser Raytchev; Xiujuan Sun
>
> **备注:** The International Conference on Neural Information Processing (ICONIP) 2025
>
> **摘要:** LEARN is a layout-aware diffusion framework designed to generate pedagogically aligned illustrations for STEM education. It leverages a curated BookCover dataset that provides narrative layouts and structured visual cues, enabling the model to depict abstract and sequential scientific concepts with strong semantic alignment. Through layout-conditioned generation, contrastive visual-semantic training, and prompt modulation, LEARN produces coherent visual sequences that support mid-to-high-level reasoning in line with Bloom's taxonomy while reducing extraneous cognitive load as emphasized by Cognitive Load Theory. By fostering spatially organized and story-driven narratives, the framework counters fragmented attention often induced by short-form media and promotes sustained conceptual focus. Beyond static diagrams, LEARN demonstrates potential for integration with multimodal systems and curriculum-linked knowledge graphs to create adaptive, exploratory educational content. As the first generative approach to unify layout-based storytelling, semantic structure learning, and cognitive scaffolding, LEARN represents a novel direction for generative AI in education. The code and dataset will be released to facilitate future research and practical deployment.
>
---
#### [new 030] Delving into Dynamic Scene Cue-Consistency for Robust 3D Multi-Object Tracking
- **分类: cs.CV**

- **简介: 该论文属于3D多目标跟踪任务，旨在解决复杂场景下跟踪不稳定的问题。通过引入动态场景线索一致性机制，提升跟踪鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.11323v1](http://arxiv.org/pdf/2508.11323v1)**

> **作者:** Haonan Zhang; Xinyao Wang; Boxi Wu; Tu Zheng; Wang Yunhua; Zheng Yang
>
> **摘要:** 3D multi-object tracking is a critical and challenging task in the field of autonomous driving. A common paradigm relies on modeling individual object motion, e.g., Kalman filters, to predict trajectories. While effective in simple scenarios, this approach often struggles in crowded environments or with inaccurate detections, as it overlooks the rich geometric relationships between objects. This highlights the need to leverage spatial cues. However, existing geometry-aware methods can be susceptible to interference from irrelevant objects, leading to ambiguous features and incorrect associations. To address this, we propose focusing on cue-consistency: identifying and matching stable spatial patterns over time. We introduce the Dynamic Scene Cue-Consistency Tracker (DSC-Track) to implement this principle. Firstly, we design a unified spatiotemporal encoder using Point Pair Features (PPF) to learn discriminative trajectory embeddings while suppressing interference. Secondly, our cue-consistency transformer module explicitly aligns consistent feature representations between historical tracks and current detections. Finally, a dynamic update mechanism preserves salient spatiotemporal information for stable online tracking. Extensive experiments on the nuScenes and Waymo Open Datasets validate the effectiveness and robustness of our approach. On the nuScenes benchmark, for instance, our method achieves state-of-the-art performance, reaching 73.2% and 70.3% AMOTA on the validation and test sets, respectively.
>
---
#### [new 031] Perception in Plan: Coupled Perception and Planning for End-to-End Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶任务，解决感知与规划分离导致的效率问题，提出VeteranAD框架，将感知融入规划过程，提升驾驶性能。**

- **链接: [http://arxiv.org/pdf/2508.11488v1](http://arxiv.org/pdf/2508.11488v1)**

> **作者:** Bozhou Zhang; Jingyu Li; Nan Song; Li Zhang
>
> **摘要:** End-to-end autonomous driving has achieved remarkable advancements in recent years. Existing methods primarily follow a perception-planning paradigm, where perception and planning are executed sequentially within a fully differentiable framework for planning-oriented optimization. We further advance this paradigm through a perception-in-plan framework design, which integrates perception into the planning process. This design facilitates targeted perception guided by evolving planning objectives over time, ultimately enhancing planning performance. Building on this insight, we introduce VeteranAD, a coupled perception and planning framework for end-to-end autonomous driving. By incorporating multi-mode anchored trajectories as planning priors, the perception module is specifically designed to gather traffic elements along these trajectories, enabling comprehensive and targeted perception. Planning trajectories are then generated based on both the perception results and the planning priors. To make perception fully serve planning, we adopt an autoregressive strategy that progressively predicts future trajectories while focusing on relevant regions for targeted perception at each step. With this simple yet effective design, VeteranAD fully unleashes the potential of planning-oriented end-to-end methods, leading to more accurate and reliable driving behavior. Extensive experiments on the NAVSIM and Bench2Drive datasets demonstrate that our VeteranAD achieves state-of-the-art performance.
>
---
#### [new 032] Cost-Effective Active Labeling for Data-Efficient Cervical Cell Classification
- **分类: cs.CV; q-bio.TO**

- **简介: 该论文属于宫颈细胞分类任务，旨在降低标注成本。通过主动标注方法，高效选择最具信息量的样本进行标注，提升数据效率。**

- **链接: [http://arxiv.org/pdf/2508.11340v1](http://arxiv.org/pdf/2508.11340v1)**

> **作者:** Yuanlin Liu; Zhihan Zhou; Mingqiang Wei; Youyi Song
>
> **备注:** accepted by CW2025
>
> **摘要:** Information on the number and category of cervical cells is crucial for the diagnosis of cervical cancer. However, existing classification methods capable of automatically measuring this information require the training dataset to be representative, which consumes an expensive or even unaffordable human cost. We herein propose active labeling that enables us to construct a representative training dataset using a much smaller human cost for data-efficient cervical cell classification. This cost-effective method efficiently leverages the classifier's uncertainty on the unlabeled cervical cell images to accurately select images that are most beneficial to label. With a fast estimation of the uncertainty, this new algorithm exhibits its validity and effectiveness in enhancing the representative ability of the constructed training dataset. The extensive empirical results confirm its efficacy again in navigating the usage of human cost, opening the avenue for data-efficient cervical cell classification.
>
---
#### [new 033] Privacy Enhancement for Gaze Data Using a Noise-Infused Autoencoder
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于隐私保护任务，旨在防止眼动数据被用于身份识别，同时保持数据可用性。通过噪声注入自编码器实现隐私与效用的平衡。**

- **链接: [http://arxiv.org/pdf/2508.10918v1](http://arxiv.org/pdf/2508.10918v1)**

> **作者:** Samantha Aziz; Oleg Komogortsev
>
> **备注:** IJCB 2025; 11 pages, 7 figures
>
> **摘要:** We present a privacy-enhancing mechanism for gaze signals using a latent-noise autoencoder that prevents users from being re-identified across play sessions without their consent, while retaining the usability of the data for benign tasks. We evaluate privacy-utility trade-offs across biometric identification and gaze prediction tasks, showing that our approach significantly reduces biometric identifiability with minimal utility degradation. Unlike prior methods in this direction, our framework retains physiologically plausible gaze patterns suitable for downstream use, which produces favorable privacy-utility trade-off. This work advances privacy in gaze-based systems by providing a usable and effective mechanism for protecting sensitive gaze data.
>
---
#### [new 034] MedSAMix: A Training-Free Model Merging Approach for Medical Image Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11032v1](http://arxiv.org/pdf/2508.11032v1)**

> **作者:** Yanwu Yang; Guinan Su; Jiesi Hu; Francesco Sammarco; Jonas Geiping; Thomas Wolfers
>
> **摘要:** Universal medical image segmentation models have emerged as a promising paradigm due to their strong generalizability across diverse tasks, showing great potential for a wide range of clinical applications. This potential has been partly driven by the success of general-purpose vision models such as the Segment Anything Model (SAM), which has inspired the development of various fine-tuned variants for medical segmentation tasks. However, fine-tuned variants like MedSAM are trained on comparatively limited medical imaging data that often suffers from heterogeneity, scarce annotations, and distributional shifts. These challenges limit their ability to generalize across a wide range of medical segmentation tasks. In this regard, we propose MedSAMix, a training-free model merging method that integrates the strengths of both generalist models (e.g., SAM) and specialist models (e.g., MedSAM) for medical image segmentation. In contrast to traditional model merging approaches that rely on manual configuration and often result in suboptimal outcomes, we propose a zero-order optimization method to automatically discover optimal layer-wise merging solutions. Furthermore, for clinical applications, we develop two regimes to meet the demand of domain-specificity and generalizability in different scenarios by single-task optimization and multi-objective optimization respectively. Extensive evaluations on 25 medical segmentation tasks demonstrate that MedSAMix effectively mitigates model bias and consistently improves performance in both domain-specific accuracy and generalization, achieving improvements of 6.67% on specialized tasks and 4.37% on multi-task evaluations.
>
---
#### [new 035] VSF: Simple, Efficient, and Effective Negative Guidance in Few-Step Image Generation Models By \underline{V}alue \underline{S}ign \underline{F}lip
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，解决负提示引导问题。提出VSF方法，通过翻转注意力值符号高效抑制不良内容，提升生成质量与负提示遵循度。**

- **链接: [http://arxiv.org/pdf/2508.10931v1](http://arxiv.org/pdf/2508.10931v1)**

> **作者:** Wenqi Guo; Shan Du
>
> **摘要:** We introduce Value Sign Flip (VSF), a simple and efficient method for incorporating negative prompt guidance in few-step diffusion and flow-matching image generation models. Unlike existing approaches such as classifier-free guidance (CFG), NASA, and NAG, VSF dynamically suppresses undesired content by flipping the sign of attention values from negative prompts. Our method requires only small computational overhead and integrates effectively with MMDiT-style architectures such as Stable Diffusion 3.5 Turbo, as well as cross-attention-based models like Wan. We validate VSF on challenging datasets with complex prompt pairs and demonstrate superior performance in both static image and video generation tasks. Experimental results show that VSF significantly improves negative prompt adherence compared to prior methods in few-step models, and even CFG in non-few-step models, while maintaining competitive image quality. Code and ComfyUI node are available in https://github.com/weathon/VSF/tree/main.
>
---
#### [new 036] Semantically Guided Adversarial Testing of Vision Models Using Language Models
- **分类: cs.CV; cs.CR; cs.LG; 68T45, 68T01, 68T07, 68T10, 68M25; I.2.10; I.5.4; I.2.6; I.2.7; K.6.5**

- **简介: 该论文属于视觉模型对抗测试任务，解决目标标签选择问题。通过语义引导框架，利用语言模型提升对抗攻击效果与可解释性。**

- **链接: [http://arxiv.org/pdf/2508.11341v1](http://arxiv.org/pdf/2508.11341v1)**

> **作者:** Katarzyna Filus; Jorge M. Cruz-Duarte
>
> **备注:** 12 pages, 4 figures, 3 tables. Submitted for peer review
>
> **摘要:** In targeted adversarial attacks on vision models, the selection of the target label is a critical yet often overlooked determinant of attack success. This target label corresponds to the class that the attacker aims to force the model to predict. Now, existing strategies typically rely on randomness, model predictions, or static semantic resources, limiting interpretability, reproducibility, or flexibility. This paper then proposes a semantics-guided framework for adversarial target selection using the cross-modal knowledge transfer from pretrained language and vision-language models. We evaluate several state-of-the-art models (BERT, TinyLLAMA, and CLIP) as similarity sources to select the most and least semantically related labels with respect to the ground truth, forming best- and worst-case adversarial scenarios. Our experiments on three vision models and five attack methods reveal that these models consistently render practical adversarial targets and surpass static lexical databases, such as WordNet, particularly for distant class relationships. We also observe that static testing of target labels offers a preliminary assessment of the effectiveness of similarity sources, \textit{a priori} testing. Our results corroborate the suitability of pretrained models for constructing interpretable, standardized, and scalable adversarial benchmarks across architectures and datasets.
>
---
#### [new 037] Are Large Pre-trained Vision Language Models Effective Construction Safety Inspectors?
- **分类: cs.CV**

- **简介: 该论文属于建筑安全检测任务，旨在解决VLMs在该领域应用数据不足的问题。提出ConstructionSite 10k数据集，用于评估和训练VLMs。**

- **链接: [http://arxiv.org/pdf/2508.11011v1](http://arxiv.org/pdf/2508.11011v1)**

> **作者:** Xuezheng Chen; Zhengbo Zou
>
> **摘要:** Construction safety inspections typically involve a human inspector identifying safety concerns on-site. With the rise of powerful Vision Language Models (VLMs), researchers are exploring their use for tasks such as detecting safety rule violations from on-site images. However, there is a lack of open datasets to comprehensively evaluate and further fine-tune VLMs in construction safety inspection. Current applications of VLMs use small, supervised datasets, limiting their applicability in tasks they are not directly trained for. In this paper, we propose the ConstructionSite 10k, featuring 10,000 construction site images with annotations for three inter-connected tasks, including image captioning, safety rule violation visual question answering (VQA), and construction element visual grounding. Our subsequent evaluation of current state-of-the-art large pre-trained VLMs shows notable generalization abilities in zero-shot and few-shot settings, while additional training is needed to make them applicable to actual construction sites. This dataset allows researchers to train and evaluate their own VLMs with new architectures and techniques, providing a valuable benchmark for construction safety inspection.
>
---
#### [new 038] Relative Pose Regression with Pose Auto-Encoders: Enhancing Accuracy and Data Efficiency for Retail Applications
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于相对位姿回归任务，旨在提升零售场景中的相机定位精度与数据效率。通过引入PAE增强APR，实现无需额外数据的精修策略。**

- **链接: [http://arxiv.org/pdf/2508.10933v1](http://arxiv.org/pdf/2508.10933v1)**

> **作者:** Yoli Shavit; Yosi Keller
>
> **备注:** Accepted to ICCVW 2025
>
> **摘要:** Accurate camera localization is crucial for modern retail environments, enabling enhanced customer experiences, streamlined inventory management, and autonomous operations. While Absolute Pose Regression (APR) from a single image offers a promising solution, approaches that incorporate visual and spatial scene priors tend to achieve higher accuracy. Camera Pose Auto-Encoders (PAEs) have recently been introduced to embed such priors into APR. In this work, we extend PAEs to the task of Relative Pose Regression (RPR) and propose a novel re-localization scheme that refines APR predictions using PAE-based RPR, without requiring additional storage of images or pose data. We first introduce PAE-based RPR and establish its effectiveness by comparing it with image-based RPR models of equivalent architectures. We then demonstrate that our refinement strategy, driven by a PAE-based RPR, enhances APR localization accuracy on indoor benchmarks. Notably, our method is shown to achieve competitive performance even when trained with only 30% of the data, substantially reducing the data collection burden for retail deployment. Our code and pre-trained models are available at: https://github.com/yolish/camera-pose-auto-encoders
>
---
#### [new 039] CSNR and JMIM Based Spectral Band Selection for Reducing Metamerism in Urban Driving
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，解决metamerism导致的VRU识别困难问题，通过HSI band选择提升VRU与背景的区分度。**

- **链接: [http://arxiv.org/pdf/2508.10962v1](http://arxiv.org/pdf/2508.10962v1)**

> **作者:** Jiarong Li; Imad Ali Shah; Diarmaid Geever; Fiachra Collins; Enda Ward; Martin Glavin; Edward Jones; Brian Deegan
>
> **备注:** Under Review at IEEE OJITS, July, 2025
>
> **摘要:** Protecting Vulnerable Road Users (VRU) is a critical safety challenge for automotive perception systems, particularly under visual ambiguity caused by metamerism, a phenomenon where distinct materials appear similar in RGB imagery. This work investigates hyperspectral imaging (HSI) to overcome this limitation by capturing unique material signatures beyond the visible spectrum, especially in the Near-Infrared (NIR). To manage the inherent high-dimensionality of HSI data, we propose a band selection strategy that integrates information theory techniques (joint mutual information maximization, correlation analysis) with a novel application of an image quality metric (contrast signal-to-noise ratio) to identify the most spectrally informative bands. Using the Hyperspectral City V2 (H-City) dataset, we identify three informative bands (497 nm, 607 nm, and 895 nm, $\pm$27 nm) and reconstruct pseudo-color images for comparison with co-registered RGB. Quantitative results demonstrate increased dissimilarity and perceptual separability of VRU from the background. The selected HSI bands yield improvements of 70.24%, 528.46%, 1206.83%, and 246.62% for dissimilarity (Euclidean, SAM, $T^2$) and perception (CIE $\Delta E$) metrics, consistently outperforming RGB and confirming a marked reduction in metameric confusion. By providing a spectrally optimized input, our method enhances VRU separability, establishing a robust foundation for downstream perception tasks in Advanced Driver Assistance Systems (ADAS) and Autonomous Driving (AD), ultimately contributing to improved road safety.
>
---
#### [new 040] Multi-State Tracker: Enhancing Efficient Object Tracking via Multi-State Specialization and Interaction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11531v1](http://arxiv.org/pdf/2508.11531v1)**

> **作者:** Shilei Wang; Gong Cheng; Pujian Lai; Dong Gao; Junwei Han
>
> **摘要:** Efficient trackers achieve faster runtime by reducing computational complexity and model parameters. However, this efficiency often compromises the expense of weakened feature representation capacity, thus limiting their ability to accurately capture target states using single-layer features. To overcome this limitation, we propose Multi-State Tracker (MST), which utilizes highly lightweight state-specific enhancement (SSE) to perform specialized enhancement on multi-state features produced by multi-state generation (MSG) and aggregates them in an interactive and adaptive manner using cross-state interaction (CSI). This design greatly enhances feature representation while incurring minimal computational overhead, leading to improved tracking robustness in complex environments. Specifically, the MSG generates multiple state representations at multiple stages during feature extraction, while SSE refines them to highlight target-specific features. The CSI module facilitates information exchange between these states and ensures the integration of complementary features. Notably, the introduced SSE and CSI modules adopt a highly lightweight hidden state adaptation-based state space duality (HSA-SSD) design, incurring only 0.1 GFLOPs in computation and 0.66 M in parameters. Experimental results demonstrate that MST outperforms all previous efficient trackers across multiple datasets, significantly improving tracking accuracy and robustness. In particular, it shows excellent runtime performance, with an AO score improvement of 4.5\% over the previous SOTA efficient tracker HCAT on the GOT-10K dataset. The code is available at https://github.com/wsumel/MST.
>
---
#### [new 041] Deep Learning for Automated Identification of Vietnamese Timber Species: A Tool for Ecological Monitoring and Conservation
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决越南木材物种自动识别问题。通过构建数据集并应用深度学习模型，实现高效准确的物种识别。**

- **链接: [http://arxiv.org/pdf/2508.10938v1](http://arxiv.org/pdf/2508.10938v1)**

> **作者:** Tianyu Song; Van-Doan Duong; Thi-Phuong Le; Ton Viet Ta
>
> **摘要:** Accurate identification of wood species plays a critical role in ecological monitoring, biodiversity conservation, and sustainable forest management. Traditional classification approaches relying on macroscopic and microscopic inspection are labor-intensive and require expert knowledge. In this study, we explore the application of deep learning to automate the classification of ten wood species commonly found in Vietnam. A custom image dataset was constructed from field-collected wood samples, and five state-of-the-art convolutional neural network architectures--ResNet50, EfficientNet, MobileViT, MobileNetV3, and ShuffleNetV2--were evaluated. Among these, ShuffleNetV2 achieved the best balance between classification performance and computational efficiency, with an average accuracy of 99.29\% and F1-score of 99.35\% over 20 independent runs. These results demonstrate the potential of lightweight deep learning models for real-time, high-accuracy species identification in resource-constrained environments. Our work contributes to the growing field of ecological informatics by providing scalable, image-based solutions for automated wood classification and forest biodiversity assessment.
>
---
#### [new 042] Automated Building Heritage Assessment Using Street-Level Imagery
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11486v1](http://arxiv.org/pdf/2508.11486v1)**

> **作者:** Kristina Dabrock; Tim Johansson; Anna Donarelli; Mikael Mangold; Noah Pflugradt; Jann Michael Weinand; Jochen Linßen
>
> **摘要:** Detailed data is required to quantify energy conservation measures in buildings, such as envelop retrofits, without compromising cultural heritage. Novel artificial intelligence tools may improve efficiency in identifying heritage values in buildings compared to costly and time-consuming traditional inventories. In this study, the large language model GPT was used to detect various aspects of cultural heritage value in fa\c{c}ade images. Using this data and building register data as features, machine learning models were trained to classify multi-family and non-residential buildings in Stockholm, Sweden. Validation against an expert-created inventory shows a macro F1-score of 0.71 using a combination of register data and features retrieved from GPT, and a score of 0.60 using only GPT-derived data. The presented methodology can contribute to a higher-quality database and thus support careful energy efficiency measures and integrated consideration of heritage value in large-scale energetic refurbishment scenarios.
>
---
#### [new 043] ImagiDrive: A Unified Imagination-and-Planning Framework for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决环境理解与路径规划的整合问题。提出ImagiDrive框架，结合视觉语言模型与驾驶世界模型，提升预测与决策性能。**

- **链接: [http://arxiv.org/pdf/2508.11428v1](http://arxiv.org/pdf/2508.11428v1)**

> **作者:** Jingyu Li; Bozhou Zhang; Xin Jin; Jiankang Deng; Xiatian Zhu; Li Zhang
>
> **摘要:** Autonomous driving requires rich contextual comprehension and precise predictive reasoning to navigate dynamic and complex environments safely. Vision-Language Models (VLMs) and Driving World Models (DWMs) have independently emerged as powerful recipes addressing different aspects of this challenge. VLMs provide interpretability and robust action prediction through their ability to understand multi-modal context, while DWMs excel in generating detailed and plausible future driving scenarios essential for proactive planning. Integrating VLMs with DWMs is an intuitive, promising, yet understudied strategy to exploit the complementary strengths of accurate behavioral prediction and realistic scene generation. Nevertheless, this integration presents notable challenges, particularly in effectively connecting action-level decisions with high-fidelity pixel-level predictions and maintaining computational efficiency. In this paper, we propose ImagiDrive, a novel end-to-end autonomous driving framework that integrates a VLM-based driving agent with a DWM-based scene imaginer to form a unified imagination-and-planning loop. The driving agent predicts initial driving trajectories based on multi-modal inputs, guiding the scene imaginer to generate corresponding future scenarios. These imagined scenarios are subsequently utilized to iteratively refine the driving agent's planning decisions. To address efficiency and predictive accuracy challenges inherent in this integration, we introduce an early stopping mechanism and a trajectory selection strategy. Extensive experimental validation on the nuScenes and NAVSIM datasets demonstrates the robustness and superiority of ImagiDrive over previous alternatives under both open-loop and closed-loop conditions.
>
---
#### [new 044] Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D语义占用预测任务，解决协作场景下通信成本高和依赖深度监督的问题。提出基于稀疏3D语义高斯点云的协作方法，提升性能并减少通信量。**

- **链接: [http://arxiv.org/pdf/2508.10936v1](http://arxiv.org/pdf/2508.10936v1)**

> **作者:** Cheng Chen; Hao Huang; Saurabh Bagchi
>
> **摘要:** Collaborative perception enables connected vehicles to share information, overcoming occlusions and extending the limited sensing range inherent in single-agent (non-collaborative) systems. Existing vision-only methods for 3D semantic occupancy prediction commonly rely on dense 3D voxels, which incur high communication costs, or 2D planar features, which require accurate depth estimation or additional supervision, limiting their applicability to collaborative scenarios. To address these challenges, we propose the first approach leveraging sparse 3D semantic Gaussian splatting for collaborative 3D semantic occupancy prediction. By sharing and fusing intermediate Gaussian primitives, our method provides three benefits: a neighborhood-based cross-agent fusion that removes duplicates and suppresses noisy or inconsistent Gaussians; a joint encoding of geometry and semantics in each primitive, which reduces reliance on depth supervision and allows simple rigid alignment; and sparse, object-centric messages that preserve structural information while reducing communication volume. Extensive experiments demonstrate that our approach outperforms single-agent perception and baseline collaborative methods by +8.42 and +3.28 points in mIoU, and +5.11 and +22.41 points in IoU, respectively. When further reducing the number of transmitted Gaussians, our method still achieves a +1.9 improvement in mIoU, using only 34.6% communication volume, highlighting robust performance under limited communication budgets.
>
---
#### [new 045] Remove360: Benchmarking Residuals After Object Removal in 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11431v1](http://arxiv.org/pdf/2508.11431v1)**

> **作者:** Simona Kocour; Assia Benbihi; Torsten Sattler
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2503.17574
>
> **摘要:** Understanding what semantic information persists after object removal is critical for privacy-preserving 3D reconstruction and editable scene representations. In this work, we introduce a novel benchmark and evaluation framework to measure semantic residuals, the unintended semantic traces left behind, after object removal in 3D Gaussian Splatting. We conduct experiments across a diverse set of indoor and outdoor scenes, showing that current methods can preserve semantic information despite the absence of visual geometry. We also release Remove360, a dataset of pre/post-removal RGB images and object-level masks captured in real-world environments. While prior datasets have focused on isolated object instances, Remove360 covers a broader and more complex range of indoor and outdoor scenes, enabling evaluation of object removal in the context of full-scene representations. Given ground truth images of a scene before and after object removal, we assess whether we can truly eliminate semantic presence, and if downstream models can still infer what was removed. Our findings reveal critical limitations in current 3D object removal techniques and underscore the need for more robust solutions capable of handling real-world complexity. The evaluation framework is available at github.com/spatial-intelligence-ai/Remove360.git. Data are available at huggingface.co/datasets/simkoc/Remove360.
>
---
#### [new 046] Data-Driven Abdominal Phenotypes of Type 2 Diabetes in Lean, Overweight, and Obese Cohorts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11063v1](http://arxiv.org/pdf/2508.11063v1)**

> **作者:** Lucas W. Remedios; Chloe Choe; Trent M. Schwartz; Dingjie Su; Gaurav Rudravaram; Chenyu Gao; Aravind R. Krishnan; Adam M. Saunders; Michael E. Kim; Shunxing Bao; Alvin C. Powers; Bennett A. Landman; John Virostko
>
> **摘要:** Purpose: Although elevated BMI is a well-known risk factor for type 2 diabetes, the disease's presence in some lean adults and absence in others with obesity suggests that detailed body composition may uncover abdominal phenotypes of type 2 diabetes. With AI, we can now extract detailed measurements of size, shape, and fat content from abdominal structures in 3D clinical imaging at scale. This creates an opportunity to empirically define body composition signatures linked to type 2 diabetes risk and protection using large-scale clinical data. Approach: To uncover BMI-specific diabetic abdominal patterns from clinical CT, we applied our design four times: once on the full cohort (n = 1,728) and once on lean (n = 497), overweight (n = 611), and obese (n = 620) subgroups separately. Briefly, our experimental design transforms abdominal scans into collections of explainable measurements through segmentation, classifies type 2 diabetes through a cross-validated random forest, measures how features contribute to model-estimated risk or protection through SHAP analysis, groups scans by shared model decision patterns (clustering from SHAP) and links back to anatomical differences (classification). Results: The random-forests achieved mean AUCs of 0.72-0.74. There were shared type 2 diabetes signatures in each group; fatty skeletal muscle, older age, greater visceral and subcutaneous fat, and a smaller or fat-laden pancreas. Univariate logistic regression confirmed the direction of 14-18 of the top 20 predictors within each subgroup (p < 0.05). Conclusions: Our findings suggest that abdominal drivers of type 2 diabetes may be consistent across weight classes.
>
---
#### [new 047] AIM: Amending Inherent Interpretability via Self-Supervised Masking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11502v1](http://arxiv.org/pdf/2508.11502v1)**

> **作者:** Eyad Alshami; Shashank Agnihotri; Bernt Schiele; Margret Keuper
>
> **备注:** Accepted at International Conference on Computer Vision (ICCV) 2025
>
> **摘要:** It has been observed that deep neural networks (DNNs) often use both genuine as well as spurious features. In this work, we propose "Amending Inherent Interpretability via Self-Supervised Masking" (AIM), a simple yet interestingly effective method that promotes the network's utilization of genuine features over spurious alternatives without requiring additional annotations. In particular, AIM uses features at multiple encoding stages to guide a self-supervised, sample-specific feature-masking process. As a result, AIM enables the training of well-performing and inherently interpretable models that faithfully summarize the decision process. We validate AIM across a diverse range of challenging datasets that test both out-of-distribution generalization and fine-grained visual understanding. These include general-purpose classification benchmarks such as ImageNet100, HardImageNet, and ImageWoof, as well as fine-grained classification datasets such as Waterbirds, TravelingBirds, and CUB-200. AIM demonstrates significant dual benefits: interpretability improvements, as measured by the Energy Pointing Game (EPG) score, and accuracy gains over strong baselines. These consistent gains across domains and architectures provide compelling evidence that AIM promotes the use of genuine and meaningful features that directly contribute to improved generalization and human-aligned interpretability.
>
---
#### [new 048] ViPE: Video Pose Engine for 3D Geometric Perception
- **分类: cs.CV; cs.GR; cs.RO; eess.IV**

- **简介: 该论文属于3D几何感知任务，旨在解决从视频中准确估计相机参数和深度的问题。提出ViPE工具，高效处理多种视频场景并生成精确标注数据。**

- **链接: [http://arxiv.org/pdf/2508.10934v1](http://arxiv.org/pdf/2508.10934v1)**

> **作者:** Jiahui Huang; Qunjie Zhou; Hesam Rabeti; Aleksandr Korovko; Huan Ling; Xuanchi Ren; Tianchang Shen; Jun Gao; Dmitry Slepichev; Chen-Hsuan Lin; Jiawei Ren; Kevin Xie; Joydeep Biswas; Laura Leal-Taixe; Sanja Fidler
>
> **备注:** Paper website: https://research.nvidia.com/labs/toronto-ai/vipe/
>
> **摘要:** Accurate 3D geometric perception is an important prerequisite for a wide range of spatial AI systems. While state-of-the-art methods depend on large-scale training data, acquiring consistent and precise 3D annotations from in-the-wild videos remains a key challenge. In this work, we introduce ViPE, a handy and versatile video processing engine designed to bridge this gap. ViPE efficiently estimates camera intrinsics, camera motion, and dense, near-metric depth maps from unconstrained raw videos. It is robust to diverse scenarios, including dynamic selfie videos, cinematic shots, or dashcams, and supports various camera models such as pinhole, wide-angle, and 360{\deg} panoramas. We have benchmarked ViPE on multiple benchmarks. Notably, it outperforms existing uncalibrated pose estimation baselines by 18%/50% on TUM/KITTI sequences, and runs at 3-5FPS on a single GPU for standard input resolutions. We use ViPE to annotate a large-scale collection of videos. This collection includes around 100K real-world internet videos, 1M high-quality AI-generated videos, and 2K panoramic videos, totaling approximately 96M frames -- all annotated with accurate camera poses and dense depth maps. We open-source ViPE and the annotated dataset with the hope of accelerating the development of spatial AI systems.
>
---
#### [new 049] Logic Unseen: Revealing the Logical Blindspots of Vision-Language Models
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于视觉-语言模型逻辑理解任务，旨在解决VLMs在逻辑推理上的不足。通过构建LogicBench基准和提出LogicCLIP框架，提升模型的逻辑敏感性。**

- **链接: [http://arxiv.org/pdf/2508.11317v1](http://arxiv.org/pdf/2508.11317v1)**

> **作者:** Yuchen Zhou; Jiayu Tang; Shuo Yang; Xiaoyan Xiao; Yuqin Dai; Wenhao Yang; Chao Gou; Xiaobo Xia; Tat-Seng Chua
>
> **摘要:** Vision-Language Models (VLMs), exemplified by CLIP, have emerged as foundational for multimodal intelligence. However, their capacity for logical understanding remains significantly underexplored, resulting in critical ''logical blindspots'' that limit their reliability in practical applications. To systematically diagnose this, we introduce LogicBench, a comprehensive benchmark with over 50,000 vision-language pairs across 9 logical categories and 4 diverse scenarios: images, videos, anomaly detection, and medical diagnostics. Our evaluation reveals that existing VLMs, even the state-of-the-art ones, fall at over 40 accuracy points below human performance, particularly in challenging tasks like Causality and Conditionality, highlighting their reliance on surface semantics over critical logical structures. To bridge this gap, we propose LogicCLIP, a novel training framework designed to boost VLMs' logical sensitivity through advancements in both data generation and optimization objectives. LogicCLIP utilizes logic-aware data generation and a contrastive learning strategy that combines coarse-grained alignment, a fine-grained multiple-choice objective, and a novel logical structure-aware objective. Extensive experiments demonstrate LogicCLIP's substantial improvements in logical comprehension across all LogicBench domains, significantly outperforming baselines. Moreover, LogicCLIP retains, and often surpasses, competitive performance on general vision-language benchmarks, demonstrating that the enhanced logical understanding does not come at the expense of general alignment. We believe that LogicBench and LogicCLIP will be important resources for advancing VLM logical capabilities.
>
---
#### [new 050] Vision-Language Models display a strong gender bias
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型研究，旨在检测模型中的性别偏见。通过分析图像与文本嵌入的相似性，评估模型对不同性别关联的倾向，揭示潜在的社会刻板印象。**

- **链接: [http://arxiv.org/pdf/2508.11262v1](http://arxiv.org/pdf/2508.11262v1)**

> **作者:** Aiswarya Konavoor; Raj Abhijit Dandekar; Rajat Dandekar; Sreedath Panat
>
> **摘要:** Vision-language models (VLM) align images and text in a shared representation space that is useful for retrieval and zero-shot transfer. Yet, this alignment can encode and amplify social stereotypes in subtle ways that are not obvious from standard accuracy metrics. In this study, we test whether the contrastive vision-language encoder exhibits gender-linked associations when it places embeddings of face images near embeddings of short phrases that describe occupations and activities. We assemble a dataset of 220 face photographs split by perceived binary gender and a set of 150 unique statements distributed across six categories covering emotional labor, cognitive labor, domestic labor, technical labor, professional roles, and physical labor. We compute unit-norm image embeddings for every face and unit-norm text embeddings for every statement, then define a statement-level association score as the difference between the mean cosine similarity to the male set and the mean cosine similarity to the female set, where positive values indicate stronger association with the male set and negative values indicate stronger association with the female set. We attach bootstrap confidence intervals by resampling images within each gender group, aggregate by category with a separate bootstrap over statements, and run a label-swap null model that estimates the level of mean absolute association we would expect if no gender structure were present. The outcome is a statement-wise and category-wise map of gender associations in a contrastive vision-language space, accompanied by uncertainty, simple sanity checks, and a robust gender bias evaluation framework.
>
---
#### [new 051] Training-free Dimensionality Reduction via Feature Truncation: Enhancing Efficiency in Privacy-preserving Multi-Biometric Systems
- **分类: cs.CV**

- **简介: 该论文属于隐私保护的多模态生物识别任务，旨在解决加密下模板效率问题。通过特征截断实现无训练降维，提升加密处理效率并保持识别性能。**

- **链接: [http://arxiv.org/pdf/2508.11419v1](http://arxiv.org/pdf/2508.11419v1)**

> **作者:** Florian Bayer; Maximilian Russo; Christian Rathgeb
>
> **摘要:** Biometric recognition is widely used, making the privacy and security of extracted templates a critical concern. Biometric Template Protection schemes, especially those utilizing Homomorphic Encryption, introduce significant computational challenges due to increased workload. Recent advances in deep neural networks have enabled state-of-the-art feature extraction for face, fingerprint, and iris modalities. The ubiquity and affordability of biometric sensors further facilitate multi-modal fusion, which can enhance security by combining features from different modalities. This work investigates the biometric performance of reduced multi-biometric template sizes. Experiments are conducted on an in-house virtual multi-biometric database, derived from DNN-extracted features for face, fingerprint, and iris, using the FRGC, MCYT, and CASIA databases. The evaluated approaches are (i) explainable and straightforward to implement under encryption, (ii) training-free, and (iii) capable of generalization. Dimensionality reduction of feature vectors leads to fewer operations in the Homomorphic Encryption (HE) domain, enabling more efficient encrypted processing while maintaining biometric accuracy and security at a level equivalent to or exceeding single-biometric recognition. Our results demonstrate that, by fusing feature vectors from multiple modalities, template size can be reduced by 67 % with no loss in Equal Error Rate (EER) compared to the best-performing single modality.
>
---
#### [new 052] Hyperspectral vs. RGB for Pedestrian Segmentation in Urban Driving Scenes: A Comparative Study
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11301v1](http://arxiv.org/pdf/2508.11301v1)**

> **作者:** Jiarong Li; Imad Ali Shah; Enda Ward; Martin Glavin; Edward Jones; Brian Deegan
>
> **备注:** Submitted to IEEE ICVES, July, 2025
>
> **摘要:** Pedestrian segmentation in automotive perception systems faces critical safety challenges due to metamerism in RGB imaging, where pedestrians and backgrounds appear visually indistinguishable.. This study investigates the potential of hyperspectral imaging (HSI) for enhanced pedestrian segmentation in urban driving scenarios using the Hyperspectral City v2 (H-City) dataset. We compared standard RGB against two dimensionality-reduction approaches by converting 128-channel HSI data into three-channel representations: Principal Component Analysis (PCA) and optimal band selection using Contrast Signal-to-Noise Ratio with Joint Mutual Information Maximization (CSNR-JMIM). Three semantic segmentation models were evaluated: U-Net, DeepLabV3+, and SegFormer. CSNR-JMIM consistently outperformed RGB with an average improvements of 1.44% in Intersection over Union (IoU) and 2.18% in F1-score for pedestrian segmentation. Rider segmentation showed similar gains with 1.43% IoU and 2.25% F1-score improvements. These improved performance results from enhanced spectral discrimination of optimally selected HSI bands effectively reducing false positives. This study demonstrates robust pedestrian segmentation through optimal HSI band selection, showing significant potential for safety-critical automotive applications.
>
---
#### [new 053] TACR-YOLO: A Real-time Detection Framework for Abnormal Human Behaviors Enhanced with Coordinate and Task-Aware Representations
- **分类: cs.CV**

- **简介: 该论文属于异常人体行为检测任务，解决小目标、任务冲突和多尺度融合问题。提出TACR-YOLO框架，引入坐标注意力、任务感知模块和优化网络，提升检测精度与速度。**

- **链接: [http://arxiv.org/pdf/2508.11478v1](http://arxiv.org/pdf/2508.11478v1)**

> **作者:** Xinyi Yin; Wenbo Yuan; Xuecheng Wu; Liangyu Fu; Danlei Huang
>
> **备注:** 8 pages, 4 figures, accepted by IJCNN 2025
>
> **摘要:** Abnormal Human Behavior Detection (AHBD) under special scenarios is becoming increasingly crucial. While YOLO-based detection methods excel in real-time tasks, they remain hindered by challenges including small objects, task conflicts, and multi-scale fusion in AHBD. To tackle them, we propose TACR-YOLO, a new real-time framework for AHBD. We introduce a Coordinate Attention Module to enhance small object detection, a Task-Aware Attention Module to deal with classification-regression conflicts, and a Strengthen Neck Network for refined multi-scale fusion, respectively. In addition, we optimize Anchor Box sizes using K-means clustering and deploy DIoU-Loss to improve bounding box regression. The Personnel Anomalous Behavior Detection (PABD) dataset, which includes 8,529 samples across four behavior categories, is also presented. Extensive experimental results indicate that TACR-YOLO achieves 91.92% mAP on PABD, with competitive speed and robustness. Ablation studies highlight the contribution of each improvement. This work provides new insights for abnormal behavior detection under special scenarios, advancing its progress.
>
---
#### [new 054] FantasyTalking2: Timestep-Layer Adaptive Preference Optimization for Audio-Driven Portrait Animation
- **分类: cs.CV**

- **简介: 该论文属于音频驱动的人像动画任务，解决多维偏好对齐问题。提出Talking-Critic和TLPO框架，提升唇形同步、动作自然度和视觉质量。**

- **链接: [http://arxiv.org/pdf/2508.11255v1](http://arxiv.org/pdf/2508.11255v1)**

> **作者:** MengChao Wang; Qiang Wang; Fan Jiang; Mu Xu
>
> **备注:** https://fantasy-amap.github.io/fantasy-talking2/
>
> **摘要:** Recent advances in audio-driven portrait animation have demonstrated impressive capabilities. However, existing methods struggle to align with fine-grained human preferences across multiple dimensions, such as motion naturalness, lip-sync accuracy, and visual quality. This is due to the difficulty of optimizing among competing preference objectives, which often conflict with one another, and the scarcity of large-scale, high-quality datasets with multidimensional preference annotations. To address these, we first introduce Talking-Critic, a multimodal reward model that learns human-aligned reward functions to quantify how well generated videos satisfy multidimensional expectations. Leveraging this model, we curate Talking-NSQ, a large-scale multidimensional human preference dataset containing 410K preference pairs. Finally, we propose Timestep-Layer adaptive multi-expert Preference Optimization (TLPO), a novel framework for aligning diffusion-based portrait animation models with fine-grained, multidimensional preferences. TLPO decouples preferences into specialized expert modules, which are then fused across timesteps and network layers, enabling comprehensive, fine-grained enhancement across all dimensions without mutual interference. Experiments demonstrate that Talking-Critic significantly outperforms existing methods in aligning with human preference ratings. Meanwhile, TLPO achieves substantial improvements over baseline models in lip-sync accuracy, motion naturalness, and visual quality, exhibiting superior performance in both qualitative and quantitative evaluations. Ours project page: https://fantasy-amap.github.io/fantasy-talking2/
>
---
#### [new 055] CHARM3R: Towards Unseen Camera Height Robust Monocular 3D Detector
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于单目3D目标检测任务，解决相机高度变化导致的性能下降问题。通过分析深度估计影响并提出CHAMR3R模型，提升对未见高度的泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.11185v1](http://arxiv.org/pdf/2508.11185v1)**

> **作者:** Abhinav Kumar; Yuliang Guo; Zhihao Zhang; Xinyu Huang; Liu Ren; Xiaoming Liu
>
> **备注:** ICCV 2025
>
> **摘要:** Monocular 3D object detectors, while effective on data from one ego camera height, struggle with unseen or out-of-distribution camera heights. Existing methods often rely on Plucker embeddings, image transformations or data augmentation. This paper takes a step towards this understudied problem by first investigating the impact of camera height variations on state-of-the-art (SoTA) Mono3D models. With a systematic analysis on the extended CARLA dataset with multiple camera heights, we observe that depth estimation is a primary factor influencing performance under height variations. We mathematically prove and also empirically observe consistent negative and positive trends in mean depth error of regressed and ground-based depth models, respectively, under camera height changes. To mitigate this, we propose Camera Height Robust Monocular 3D Detector (CHARM3R), which averages both depth estimates within the model. CHARM3R improves generalization to unseen camera heights by more than $45\%$, achieving SoTA performance on the CARLA dataset. Codes and Models at https://github.com/abhi1kumar/CHARM3R
>
---
#### [new 056] Denoise-then-Retrieve: Text-Conditioned Video Denoising for Video Moment Retrieval
- **分类: cs.CV**

- **简介: 该论文属于视频瞬间检索任务，解决视频中无关片段干扰问题。提出DRNet模型，通过去噪净化视频表示，提升检索精度。**

- **链接: [http://arxiv.org/pdf/2508.11313v1](http://arxiv.org/pdf/2508.11313v1)**

> **作者:** Weijia Liu; Jiuxin Cao; Bo Miao; Zhiheng Fu; Xuelin Zhu; Jiawei Ge; Bo Liu; Mehwish Nasim; Ajmal Mian
>
> **备注:** Accepted by IJCAI 2025
>
> **摘要:** Current text-driven Video Moment Retrieval (VMR) methods encode all video clips, including irrelevant ones, disrupting multimodal alignment and hindering optimization. To this end, we propose a denoise-then-retrieve paradigm that explicitly filters text-irrelevant clips from videos and then retrieves the target moment using purified multimodal representations. Following this paradigm, we introduce the Denoise-then-Retrieve Network (DRNet), comprising Text-Conditioned Denoising (TCD) and Text-Reconstruction Feedback (TRF) modules. TCD integrates cross-attention and structured state space blocks to dynamically identify noisy clips and produce a noise mask to purify multimodal video representations. TRF further distills a single query embedding from purified video representations and aligns it with the text embedding, serving as auxiliary supervision for denoising during training. Finally, we perform conditional retrieval using text embeddings on purified video representations for accurate VMR. Experiments on Charades-STA and QVHighlights demonstrate that our approach surpasses state-of-the-art methods on all metrics. Furthermore, our denoise-then-retrieve paradigm is adaptable and can be seamlessly integrated into advanced VMR models to boost performance.
>
---
#### [new 057] A CLIP-based Uncertainty Modal Modeling (UMM) Framework for Pedestrian Re-Identification in Autonomous Driving
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于行人重识别任务，解决多模态数据缺失问题。提出UMM框架，融合多模态信息，提升模型鲁棒性与效率。**

- **链接: [http://arxiv.org/pdf/2508.11218v1](http://arxiv.org/pdf/2508.11218v1)**

> **作者:** Jialin Li; Shuqi Wu; Ning Wang
>
> **摘要:** Re-Identification (ReID) is a critical technology in intelligent perception systems, especially within autonomous driving, where onboard cameras must identify pedestrians across views and time in real-time to support safe navigation and trajectory prediction. However, the presence of uncertain or missing input modalities--such as RGB, infrared, sketches, or textual descriptions--poses significant challenges to conventional ReID approaches. While large-scale pre-trained models offer strong multimodal semantic modeling capabilities, their computational overhead limits practical deployment in resource-constrained environments. To address these challenges, we propose a lightweight Uncertainty Modal Modeling (UMM) framework, which integrates a multimodal token mapper, synthetic modality augmentation strategy, and cross-modal cue interactive learner. Together, these components enable unified feature representation, mitigate the impact of missing modalities, and extract complementary information across different data types. Additionally, UMM leverages CLIP's vision-language alignment ability to fuse multimodal inputs efficiently without extensive finetuning. Experimental results demonstrate that UMM achieves strong robustness, generalization, and computational efficiency under uncertain modality conditions, offering a scalable and practical solution for pedestrian re-identification in autonomous driving scenarios.
>
---
#### [new 058] IPG: Incremental Patch Generation for Generalized Adversarial Patch Training
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2508.10946v1](http://arxiv.org/pdf/2508.10946v1)**

> **作者:** Wonho Lee; Hyunsik Na; Jisu Lee; Daeseon Choi
>
> **摘要:** The advent of adversarial patches poses a significant challenge to the robustness of AI models, particularly in the domain of computer vision tasks such as object detection. In contradistinction to traditional adversarial examples, these patches target specific regions of an image, resulting in the malfunction of AI models. This paper proposes Incremental Patch Generation (IPG), a method that generates adversarial patches up to 11.1 times more efficiently than existing approaches while maintaining comparable attack performance. The efficacy of IPG is demonstrated by experiments and ablation studies including YOLO's feature distribution visualization and adversarial training results, which show that it produces well-generalized patches that effectively cover a broader range of model vulnerabilities. Furthermore, IPG-generated datasets can serve as a robust knowledge foundation for constructing a robust model, enabling structured representation, advanced reasoning, and proactive defenses in AI security ecosystems. The findings of this study suggest that IPG has considerable potential for future utilization not only in adversarial patch defense but also in real-world applications such as autonomous vehicles, security systems, and medical imaging, where AI models must remain resilient to adversarial attacks in dynamic and high-stakes environments.
>
---
#### [new 059] From Promise to Practical Reality: Transforming Diffusion MRI Analysis with Fast Deep Learning Enhancement
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.10950v1](http://arxiv.org/pdf/2508.10950v1)**

> **作者:** Xinyi Wang; Michael Barnett; Frederique Boonstra; Yael Barnett; Mariano Cabezas; Arkiev D'Souza; Matthew C. Kiernan; Kain Kyle; Meng Law; Lynette Masters; Zihao Tang; Stephen Tisch; Sicong Tu; Anneke Van Der Walt; Dongang Wang; Fernando Calamante; Weidong Cai; Chenyu Wang
>
> **备注:** 24 pages, 5 figures
>
> **摘要:** Fiber orientation distribution (FOD) is an advanced diffusion MRI modeling technique that represents complex white matter fiber configurations, and a key step for subsequent brain tractography and connectome analysis. Its reliability and accuracy, however, heavily rely on the quality of the MRI acquisition and the subsequent estimation of the FODs at each voxel. Generating reliable FODs from widely available clinical protocols with single-shell and low-angular-resolution acquisitions remains challenging but could potentially be addressed with recent advances in deep learning-based enhancement techniques. Despite advancements, existing methods have predominantly been assessed on healthy subjects, which have proved to be a major hurdle for their clinical adoption. In this work, we validate a newly optimized enhancement framework, FastFOD-Net, across healthy controls and six neurological disorders. This accelerated end-to-end deep learning framework enhancing FODs with superior performance and delivering training/inference efficiency for clinical use ($60\times$ faster comparing to its predecessor). With the most comprehensive clinical evaluation to date, our work demonstrates the potential of FastFOD-Net in accelerating clinical neuroscience research, empowering diffusion MRI analysis for disease differentiation, improving interpretability in connectome applications, and reducing measurement errors to lower sample size requirements. Critically, this work will facilitate the more widespread adoption of, and build clinical trust in, deep learning based methods for diffusion MRI enhancement. Specifically, FastFOD-Net enables robust analysis of real-world, clinical diffusion MRI data, comparable to that achievable with high-quality research acquisitions.
>
---
#### [new 060] Fine-Grained VLM Fine-tuning via Latent Hierarchical Adapter Learning
- **分类: cs.CV**

- **简介: 该论文属于少样本分类任务，旨在解决VLMs在对齐视觉与文本表示时的不足。通过引入潜在语义层次的适配器，增强模型对已知和未知类别的适应能力。**

- **链接: [http://arxiv.org/pdf/2508.11176v1](http://arxiv.org/pdf/2508.11176v1)**

> **作者:** Yumiao Zhao; Bo Jiang; Yuhe Ding; Xiao Wang; Jin Tang; Bin Luo
>
> **摘要:** Adapter-based approaches have garnered attention for fine-tuning pre-trained Vision-Language Models (VLMs) on few-shot classification tasks. These methods strive to develop a lightweight module that better aligns visual and (category) textual representations, thereby enhancing performance on downstream few-shot learning tasks. However, existing adapters generally learn/align (category) textual-visual modalities via explicit spatial proximity in the underlying embedding space, which i) fails to capture the inherent one-to-many associations between categories and image samples and ii) struggles to establish accurate associations between the unknown categories and images. To address these issues, inspired by recent works on hyperbolic learning, we develop a novel Latent Hierarchical Adapter (LatHAdapter) for fine-tuning VLMs on downstream few-shot classification tasks. The core of LatHAdapter is to exploit the latent semantic hierarchy of downstream training data and employ it to provide richer, fine-grained guidance for the adapter learning process. Specifically, LatHAdapter first introduces some learnable `attribute' prompts as the bridge to align categories and images. Then, it projects the categories, attribute prompts, and images within each batch in a hyperbolic space, and employs hierarchical regularization to learn the latent semantic hierarchy of them, thereby fully modeling the inherent one-to-many associations among categories, learnable attributes, and image samples. Extensive experiments on four challenging few-shot tasks show that the proposed LatHAdapter consistently outperforms many other fine-tuning approaches, particularly in adapting known classes and generalizing to unknown classes.
>
---
#### [new 061] Can Multi-modal (reasoning) LLMs detect document manipulation?
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.11021v1](http://arxiv.org/pdf/2508.11021v1)**

> **作者:** Zisheng Liang; Kidus Zewde; Rudra Pratap Singh; Disha Patil; Zexi Chen; Jiayu Xue; Yao Yao; Yifei Chen; Qinzhe Liu; Simiao Ren
>
> **备注:** arXiv admin note: text overlap with arXiv:2503.20084
>
> **摘要:** Document fraud poses a significant threat to industries reliant on secure and verifiable documentation, necessitating robust detection mechanisms. This study investigates the efficacy of state-of-the-art multi-modal large language models (LLMs)-including OpenAI O1, OpenAI 4o, Gemini Flash (thinking), Deepseek Janus, Grok, Llama 3.2 and 4, Qwen 2 and 2.5 VL, Mistral Pixtral, and Claude 3.5 and 3.7 Sonnet-in detecting fraudulent documents. We benchmark these models against each other and prior work on document fraud detection techniques using a standard dataset with real transactional documents. Through prompt optimization and detailed analysis of the models' reasoning processes, we evaluate their ability to identify subtle indicators of fraud, such as tampered text, misaligned formatting, and inconsistent transactional sums. Our results reveal that top-performing multi-modal LLMs demonstrate superior zero-shot generalization, outperforming conventional methods on out-of-distribution datasets, while several vision LLMs exhibit inconsistent or subpar performance. Notably, model size and advanced reasoning capabilities show limited correlation with detection accuracy, suggesting task-specific fine-tuning is critical. This study underscores the potential of multi-modal LLMs in enhancing document fraud detection systems and provides a foundation for future research into interpretable and scalable fraud mitigation strategies.
>
---
#### [new 062] Causality Matters: How Temporal Information Emerges in Video Language Models
- **分类: cs.CV**

- **简介: 该论文属于视频语言模型任务，研究如何通过因果注意力机制实现时间理解。工作包括分析时间信息的整合路径，并提出两种优化策略。**

- **链接: [http://arxiv.org/pdf/2508.11576v1](http://arxiv.org/pdf/2508.11576v1)**

> **作者:** Yumeng Shi; Quanyu Long; Yin Wu; Wenya Wang
>
> **摘要:** Video language models (VideoLMs) have made significant progress in multimodal understanding. However, temporal understanding, which involves identifying event order, duration, and relationships across time, still remains a core challenge. Prior works emphasize positional encodings (PEs) as a key mechanism for encoding temporal structure. Surprisingly, we find that removing or modifying PEs in video inputs yields minimal degradation in the performance of temporal understanding. In contrast, reversing the frame sequence while preserving the original PEs causes a substantial drop. To explain this behavior, we conduct substantial analysis experiments to trace how temporal information is integrated within the model. We uncover a causal information pathway: temporal cues are progressively synthesized through inter-frame attention, aggregated in the final frame, and subsequently integrated into the query tokens. This emergent mechanism shows that temporal reasoning emerges from inter-visual token interactions under the constraints of causal attention, which implicitly encodes temporal structure. Based on these insights, we propose two efficiency-oriented strategies: staged cross-modal attention and a temporal exit mechanism for early token truncation. Experiments on two benchmarks validate the effectiveness of both approaches. To the best of our knowledge, this is the first work to systematically investigate video temporal understanding in VideoLMs, offering insights for future model improvement.
>
---
#### [new 063] TimeMachine: Fine-Grained Facial Age Editing with Identity Preservation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11284v1](http://arxiv.org/pdf/2508.11284v1)**

> **作者:** Yilin Mi; Qixin Yan; Zheng-Peng Duan; Chunle Guo; Hubery Yin; Hao Liu; Chen Li; Chongyi Li
>
> **摘要:** With the advancement of generative models, facial image editing has made significant progress. However, achieving fine-grained age editing while preserving personal identity remains a challenging task.In this paper, we propose TimeMachine, a novel diffusion-based framework that achieves accurate age editing while keeping identity features unchanged. To enable fine-grained age editing, we inject high-precision age information into the multi-cross attention module, which explicitly separates age-related and identity-related features. This design facilitates more accurate disentanglement of age attributes, thereby allowing precise and controllable manipulation of facial aging.Furthermore, we propose an Age Classifier Guidance (ACG) module that predicts age directly in the latent space, instead of performing denoising image reconstruction during training. By employing a lightweight module to incorporate age constraints, this design enhances age editing accuracy by modest increasing training cost. Additionally, to address the lack of large-scale, high-quality facial age datasets, we construct a HFFA dataset (High-quality Fine-grained Facial-Age dataset) which contains one million high-resolution images labeled with identity and facial attributes. Experimental results demonstrate that TimeMachine achieves state-of-the-art performance in fine-grained age editing while preserving identity consistency.
>
---
#### [new 064] Does the Skeleton-Recall Loss Really Work?
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.11374v1](http://arxiv.org/pdf/2508.11374v1)**

> **作者:** Devansh Arora; Nitin Kumar; Sukrit Gupta
>
> **摘要:** Image segmentation is an important and widely performed task in computer vision. Accomplishing effective image segmentation in diverse settings often requires custom model architectures and loss functions. A set of models that specialize in segmenting thin tubular structures are topology preservation-based loss functions. These models often utilize a pixel skeletonization process claimed to generate more precise segmentation masks of thin tubes and better capture the structures that other models often miss. One such model, Skeleton Recall Loss (SRL) proposed by Kirchhoff et al.~\cite {kirchhoff2024srl}, was stated to produce state-of-the-art results on benchmark tubular datasets. In this work, we performed a theoretical analysis of the gradients for the SRL loss. Upon comparing the performance of the proposed method on some of the tubular datasets (used in the original work, along with some additional datasets), we found that the performance of SRL-based segmentation models did not exceed traditional baseline models. By providing both a theoretical explanation and empirical evidence, this work critically evaluates the limitations of topology-based loss functions, offering valuable insights for researchers aiming to develop more effective segmentation models for complex tubular structures.
>
---
#### [new 065] A Cross-Modal Rumor Detection Scheme via Contrastive Learning by Exploring Text and Image internal Correlations
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于谣言检测任务，解决图像与文本信息关联不足的问题。通过对比学习和多尺度对齐，提升跨模态特征融合效果，增强谣言识别能力。**

- **链接: [http://arxiv.org/pdf/2508.11141v1](http://arxiv.org/pdf/2508.11141v1)**

> **作者:** Bin Ma; Yifei Zhang; Yongjin Xian; Qi Li; Linna Zhou; Gongxun Miao
>
> **摘要:** Existing rumor detection methods often neglect the content within images as well as the inherent relationships between contexts and images across different visual scales, thereby resulting in the loss of critical information pertinent to rumor identification. To address these issues, this paper presents a novel cross-modal rumor detection scheme based on contrastive learning, namely the Multi-scale Image and Context Correlation exploration algorithm (MICC). Specifically, we design an SCLIP encoder to generate unified semantic embeddings for text and multi-scale image patches through contrastive pretraining, enabling their relevance to be measured via dot-product similarity. Building upon this, a Cross-Modal Multi-Scale Alignment module is introduced to identify image regions most relevant to the textual semantics, guided by mutual information maximization and the information bottleneck principle, through a Top-K selection strategy based on a cross-modal relevance matrix constructed between the text and multi-scale image patches. Moreover, a scale-aware fusion network is designed to integrate the highly correlated multi-scale image features with global text features by assigning adaptive weights to image regions based on their semantic importance and cross-modal relevance. The proposed methodology has been extensively evaluated on two real-world datasets. The experimental results demonstrate that it achieves a substantial performance improvement over existing state-of-the-art approaches in rumor detection, highlighting its effectiveness and potential for practical applications.
>
---
#### [new 066] CineTrans: Learning to Generate Videos with Cinematic Transitions via Masked Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决多镜头视频过渡不稳定的问题。通过构建数据集和设计掩码机制，提出CineTrans框架，实现电影风格的连贯多镜头视频生成。**

- **链接: [http://arxiv.org/pdf/2508.11484v1](http://arxiv.org/pdf/2508.11484v1)**

> **作者:** Xiaoxue Wu; Bingjie Gao; Yu Qiao; Yaohui Wang; Xinyuan Chen
>
> **备注:** 27 pages, 20 figures
>
> **摘要:** Despite significant advances in video synthesis, research into multi-shot video generation remains in its infancy. Even with scaled-up models and massive datasets, the shot transition capabilities remain rudimentary and unstable, largely confining generated videos to single-shot sequences. In this work, we introduce CineTrans, a novel framework for generating coherent multi-shot videos with cinematic, film-style transitions. To facilitate insights into the film editing style, we construct a multi-shot video-text dataset Cine250K with detailed shot annotations. Furthermore, our analysis of existing video diffusion models uncovers a correspondence between attention maps in the diffusion model and shot boundaries, which we leverage to design a mask-based control mechanism that enables transitions at arbitrary positions and transfers effectively in a training-free setting. After fine-tuning on our dataset with the mask mechanism, CineTrans produces cinematic multi-shot sequences while adhering to the film editing style, avoiding unstable transitions or naive concatenations. Finally, we propose specialized evaluation metrics for transition control, temporal consistency and overall quality, and demonstrate through extensive experiments that CineTrans significantly outperforms existing baselines across all criteria.
>
---
#### [new 067] HierOctFusion: Multi-scale Octree-based 3D Shape Generation via Part-Whole-Hierarchy Message Passing
- **分类: cs.CV**

- **简介: 该论文属于3D形状生成任务，旨在解决现有方法忽略语义层次结构和计算效率低的问题。提出HierOctFusion模型，通过多尺度八叉树结构和层次消息传递提升生成效果。**

- **链接: [http://arxiv.org/pdf/2508.11106v1](http://arxiv.org/pdf/2508.11106v1)**

> **作者:** Xinjie Gao; Bi'an Du; Wei Hu
>
> **摘要:** 3D content generation remains a fundamental yet challenging task due to the inherent structural complexity of 3D data. While recent octree-based diffusion models offer a promising balance between efficiency and quality through hierarchical generation, they often overlook two key insights: 1) existing methods typically model 3D objects as holistic entities, ignoring their semantic part hierarchies and limiting generalization; and 2) holistic high-resolution modeling is computationally expensive, whereas real-world objects are inherently sparse and hierarchical, making them well-suited for layered generation. Motivated by these observations, we propose HierOctFusion, a part-aware multi-scale octree diffusion model that enhances hierarchical feature interaction for generating fine-grained and sparse object structures. Furthermore, we introduce a cross-attention conditioning mechanism that injects part-level information into the generation process, enabling semantic features to propagate effectively across hierarchical levels from parts to the whole. Additionally, we construct a 3D dataset with part category annotations using a pre-trained segmentation model to facilitate training and evaluation. Experiments demonstrate that HierOctFusion achieves superior shape quality and efficiency compared to prior methods.
>
---
#### [new 068] Unifying Scale-Aware Depth Prediction and Perceptual Priors for Monocular Endoscope Pose Estimation and Tissue Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11282v1](http://arxiv.org/pdf/2508.11282v1)**

> **作者:** Muzammil Khan; Enzo Kerkhof; Matteo Fusaglia; Koert Kuhlmann; Theo Ruers; Françoise J. Siepel
>
> **备注:** 18 pages, 8 figures, 3 Tables, submitted to IEEE Access for review
>
> **摘要:** Accurate endoscope pose estimation and 3D tissue surface reconstruction significantly enhances monocular minimally invasive surgical procedures by enabling accurate navigation and improved spatial awareness. However, monocular endoscope pose estimation and tissue reconstruction face persistent challenges, including depth ambiguity, physiological tissue deformation, inconsistent endoscope motion, limited texture fidelity, and a restricted field of view. To overcome these limitations, a unified framework for monocular endoscopic tissue reconstruction that integrates scale-aware depth prediction with temporally-constrained perceptual refinement is presented. This framework incorporates a novel MAPIS-Depth module, which leverages Depth Pro for robust initialisation and Depth Anything for efficient per-frame depth prediction, in conjunction with L-BFGS-B optimisation, to generate pseudo-metric depth estimates. These estimates are temporally refined by computing pixel correspondences using RAFT and adaptively blending flow-warped frames based on LPIPS perceptual similarity, thereby reducing artefacts arising from physiological tissue deformation and motion. To ensure accurate registration of the synthesised pseudo-RGBD frames from MAPIS-Depth, a novel WEMA-RTDL module is integrated, optimising both rotation and translation. Finally, truncated signed distance function-based volumetric fusion and marching cubes are applied to extract a comprehensive 3D surface mesh. Evaluations on HEVD and SCARED, with ablation and comparative analyses, demonstrate the framework's robustness and superiority over state-of-the-art methods.
>
---
#### [new 069] MedAtlas: Evaluating LLMs for Multi-Round, Multi-Task Medical Reasoning Across Diverse Imaging Modalities and Clinical Text
- **分类: cs.CV**

- **简介: 该论文属于医疗AI任务，旨在解决多轮、多模态医学推理问题。提出MedAtlas基准框架，支持多任务、多图像交互和高临床真实性推理。**

- **链接: [http://arxiv.org/pdf/2508.10947v1](http://arxiv.org/pdf/2508.10947v1)**

> **作者:** Ronghao Xu; Zhen Huang; Yangbo Wei; Xiaoqian Zhou; Zikang Xu; Ting Liu; Zihang Jiang; S. Kevin Zhou
>
> **摘要:** Artificial intelligence has demonstrated significant potential in clinical decision-making; however, developing models capable of adapting to diverse real-world scenarios and performing complex diagnostic reasoning remains a major challenge. Existing medical multi-modal benchmarks are typically limited to single-image, single-turn tasks, lacking multi-modal medical image integration and failing to capture the longitudinal and multi-modal interactive nature inherent to clinical practice. To address this gap, we introduce MedAtlas, a novel benchmark framework designed to evaluate large language models on realistic medical reasoning tasks. MedAtlas is characterized by four key features: multi-turn dialogue, multi-modal medical image interaction, multi-task integration, and high clinical fidelity. It supports four core tasks: open-ended multi-turn question answering, closed-ended multi-turn question answering, multi-image joint reasoning, and comprehensive disease diagnosis. Each case is derived from real diagnostic workflows and incorporates temporal interactions between textual medical histories and multiple imaging modalities, including CT, MRI, PET, ultrasound, and X-ray, requiring models to perform deep integrative reasoning across images and clinical texts. MedAtlas provides expert-annotated gold standards for all tasks. Furthermore, we propose two novel evaluation metrics: Round Chain Accuracy and Error Propagation Resistance. Benchmark results with existing multi-modal models reveal substantial performance gaps in multi-stage clinical reasoning. MedAtlas establishes a challenging evaluation platform to advance the development of robust and trustworthy medical AI.
>
---
#### [new 070] An Efficient Medical Image Classification Method Based on a Lightweight Improved ConvNeXt-Tiny Architecture
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在提升资源受限环境下的分类性能。通过优化ConvNeXt-Tiny结构和设计新损失函数，提高准确率并降低计算复杂度。**

- **链接: [http://arxiv.org/pdf/2508.11532v1](http://arxiv.org/pdf/2508.11532v1)**

> **作者:** Jingsong Xia; Yue Yin; Xiuhan Li
>
> **摘要:** Intelligent analysis of medical imaging plays a crucial role in assisting clinical diagnosis. However, achieving efficient and high-accuracy image classification in resource-constrained computational environments remains challenging. This study proposes a medical image classification method based on an improved ConvNeXt-Tiny architecture. Through structural optimization and loss function design, the proposed method enhances feature extraction capability and classification performance while reducing computational complexity. Specifically, the method introduces a dual global pooling (Global Average Pooling and Global Max Pooling) feature fusion strategy into the ConvNeXt-Tiny backbone to simultaneously preserve global statistical features and salient response information. A lightweight channel attention module, termed Squeeze-and-Excitation Vector (SEVector), is designed to improve the adaptive allocation of channel weights while minimizing parameter overhead. Additionally, a Feature Smoothing Loss is incorporated into the loss function to enhance intra-class feature consistency and suppress intra-class variance. Under CPU-only conditions (8 threads), the method achieves a maximum classification accuracy of 89.10% on the test set within 10 training epochs, exhibiting a stable convergence trend in loss values. Experimental results demonstrate that the proposed method effectively improves medical image classification performance in resource-limited settings, providing a feasible and efficient solution for the deployment and promotion of medical imaging analysis models.
>
---
#### [new 071] HQ-OV3D: A High Box Quality Open-World 3D Detection Framework based on Diffision Model
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于3D目标检测任务，解决开放世界场景下伪标签质量低的问题。提出HQ-OV3D框架，通过生成和优化高质量伪标签提升检测性能。**

- **链接: [http://arxiv.org/pdf/2508.10935v1](http://arxiv.org/pdf/2508.10935v1)**

> **作者:** Qi Liu; Yabei Li; Hongsong Wang; Lei He
>
> **摘要:** Traditional closed-set 3D detection frameworks fail to meet the demands of open-world applications like autonomous driving. Existing open-vocabulary 3D detection methods typically adopt a two-stage pipeline consisting of pseudo-label generation followed by semantic alignment. While vision-language models (VLMs) recently have dramatically improved the semantic accuracy of pseudo-labels, their geometric quality, particularly bounding box precision, remains commonly neglected.To address this issue, we propose a High Box Quality Open-Vocabulary 3D Detection (HQ-OV3D) framework, dedicated to generate and refine high-quality pseudo-labels for open-vocabulary classes. The framework comprises two key components: an Intra-Modality Cross-Validated (IMCV) Proposal Generator that utilizes cross-modality geometric consistency to generate high-quality initial 3D proposals, and an Annotated-Class Assisted (ACA) Denoiser that progressively refines 3D proposals by leveraging geometric priors from annotated categories through a DDIM-based denoising mechanism.Compared to the state-of-the-art method, training with pseudo-labels generated by our approach achieves a 7.37% improvement in mAP on novel classes, demonstrating the superior quality of the pseudo-labels produced by our framework. HQ-OV3D can serve not only as a strong standalone open-vocabulary 3D detector but also as a plug-in high-quality pseudo-label generator for existing open-vocabulary detection or annotation pipelines.
>
---
#### [new 072] EVCtrl: Efficient Control Adapter for Visual Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.10963v1](http://arxiv.org/pdf/2508.10963v1)**

> **作者:** Zixiang Yang; Yue Ma; Yinhan Zhang; Shanhui Mo; Dongrui Liu; Linfeng Zhang
>
> **摘要:** Visual generation includes both image and video generation, training probabilistic models to create coherent, diverse, and semantically faithful content from scratch. While early research focused on unconditional sampling, practitioners now demand controllable generation that allows precise specification of layout, pose, motion, or style. While ControlNet grants precise spatial-temporal control, its auxiliary branch markedly increases latency and introduces redundant computation in both uncontrolled regions and denoising steps, especially for video. To address this problem, we introduce EVCtrl, a lightweight, plug-and-play control adapter that slashes overhead without retraining the model. Specifically, we propose a spatio-temporal dual caching strategy for sparse control information. For spatial redundancy, we first profile how each layer of DiT-ControlNet responds to fine-grained control, then partition the network into global and local functional zones. A locality-aware cache focuses computation on the local zones that truly need the control signal, skipping the bulk of redundant computation in global regions. For temporal redundancy, we selectively omit unnecessary denoising steps to improve efficiency. Extensive experiments on CogVideo-Controlnet, Wan2.1-Controlnet, and Flux demonstrate that our method is effective in image and video control generation without the need for training. For example, it achieves 2.16 and 2.05 times speedups on CogVideo-Controlnet and Wan2.1-Controlnet, respectively, with almost no degradation in generation quality.Codes are available in the supplementary materials.
>
---
#### [new 073] OpenConstruction: A Systematic Synthesis of Open Visual Datasets for Data-Centric Artificial Intelligence in Construction Monitoring
- **分类: cs.CV**

- **简介: 该论文属于建筑监测领域，旨在解决现有视觉数据集质量参差不齐的问题，通过系统整理51个公开数据集，构建OpenConstruction平台，推动更可靠的人工智能应用。**

- **链接: [http://arxiv.org/pdf/2508.11482v1](http://arxiv.org/pdf/2508.11482v1)**

> **作者:** Ruoxin Xiong; Yanyu Wang; Jiannan Cai; Kaijian Liu; Yuansheng Zhu; Pingbo Tang; Nora El-Gohary
>
> **摘要:** The construction industry increasingly relies on visual data to support Artificial Intelligence (AI) and Machine Learning (ML) applications for site monitoring. High-quality, domain-specific datasets, comprising images, videos, and point clouds, capture site geometry and spatiotemporal dynamics, including the location and interaction of objects, workers, and materials. However, despite growing interest in leveraging visual datasets, existing resources vary widely in sizes, data modalities, annotation quality, and representativeness of real-world construction conditions. A systematic review to categorize their data characteristics and application contexts is still lacking, limiting the community's ability to fully understand the dataset landscape, identify critical gaps, and guide future directions toward more effective, reliable, and scalable AI applications in construction. To address this gap, this study conducts an extensive search of academic databases and open-data platforms, yielding 51 publicly available visual datasets that span the 2005-2024 period. These datasets are categorized using a structured data schema covering (i) data fundamentals (e.g., size and license), (ii) data modalities (e.g., RGB and point cloud), (iii) annotation frameworks (e.g., bounding boxes), and (iv) downstream application domains (e.g., progress tracking). This study synthesizes these findings into an open-source catalog, OpenConstruction, supporting data-driven method development. Furthermore, the study discusses several critical limitations in the existing construction dataset landscape and presents a roadmap for future data infrastructure anchored in the Findability, Accessibility, Interoperability, and Reusability (FAIR) principles. By reviewing the current landscape and outlining strategic priorities, this study supports the advancement of data-centric solutions in the construction sector.
>
---
#### [new 074] NIRMAL Pooling: An Adaptive Max Pooling Approach with Non-linear Activation for Enhanced Image Classification
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在提升CNN性能。提出NIRMAL Pooling，结合自适应最大池化与非线性激活，增强特征表达与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.10940v1](http://arxiv.org/pdf/2508.10940v1)**

> **作者:** Nirmal Gaud; Krishna Kumar Jha; Jhimli Adhikari; Adhini Nasarin P S; Joydeep Das; Samarth S Deshpande; Nitasha Barara; Vaduguru Venkata Ramya; Santu Saha; Mehmet Tarik Baran; Sarangi Venkateshwarlu; Anusha M D; Surej Mouli; Preeti Katiyar; Vipin Kumar Chaudhary
>
> **备注:** 6 pages, 2 figures
>
> **摘要:** This paper presents NIRMAL Pooling, a novel pooling layer for Convolutional Neural Networks (CNNs) that integrates adaptive max pooling with non-linear activation function for image classification tasks. The acronym NIRMAL stands for Non-linear Activation, Intermediate Aggregation, Reduction, Maximum, Adaptive, and Localized. By dynamically adjusting pooling parameters based on desired output dimensions and applying a Rectified Linear Unit (ReLU) activation post-pooling, NIRMAL Pooling improves robustness and feature expressiveness. We evaluated its performance against standard Max Pooling on three benchmark datasets: MNIST Digits, MNIST Fashion, and CIFAR-10. NIRMAL Pooling achieves test accuracies of 99.25% (vs. 99.12% for Max Pooling) on MNIST Digits, 91.59% (vs. 91.44%) on MNIST Fashion, and 70.49% (vs. 68.87%) on CIFAR-10, demonstrating consistent improvements, particularly on complex datasets. This work highlights the potential of NIRMAL Pooling to enhance CNN performance in diverse image recognition tasks, offering a flexible and reliable alternative to traditional pooling methods.
>
---
#### [new 075] ORBIT: An Object Property Reasoning Benchmark for Visual Inference Tasks
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.10956v1](http://arxiv.org/pdf/2508.10956v1)**

> **作者:** Abhishek Kolari; Mohammadhossein Khojasteh; Yifan Jiang; Floris den Hengst; Filip Ilievski
>
> **摘要:** While vision-language models (VLMs) have made remarkable progress on many popular visual question answering (VQA) benchmarks, it remains unclear whether they abstract and reason over depicted objects. Inspired by human object categorisation, object property reasoning involves identifying and recognising low-level details and higher-level abstractions. While current VQA benchmarks consider a limited set of object property attributes like size, they typically blend perception and reasoning, and lack representativeness in terms of reasoning and image categories. To this end, we introduce a systematic evaluation framework with images of three representative types, three reasoning levels of increasing complexity, and four object property dimensions driven by prior work on commonsense reasoning. We develop a procedure to instantiate this benchmark into ORBIT, a multi-level reasoning VQA benchmark for object properties comprising 360 images paired with a total of 1,080 count-based questions. Experiments with 12 state-of-the-art VLMs in zero-shot settings reveal significant limitations compared to humans, with the best-performing model only reaching 40\% accuracy. VLMs struggle particularly with realistic (photographic) images, counterfactual reasoning about physical and functional properties, and higher counts. ORBIT points to the need to develop methods for scalable benchmarking, generalize annotation guidelines, and explore additional reasoning VLMs. We make the ORBIT benchmark and the experimental code available to support such endeavors.
>
---
#### [new 076] Better Supervised Fine-tuning for VQA: Integer-Only Loss
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频质量评估任务，旨在解决现有方法精度低和损失计算效率差的问题。通过构建整数标签并设计目标掩码策略，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2508.11170v1](http://arxiv.org/pdf/2508.11170v1)**

> **作者:** Baihong Qian; Haotian Fan; Wenjie Liao; Yunqiu Wang; Tao Li; Junhui Cui
>
> **摘要:** With the rapid advancement of vision language models(VLM), their ability to assess visual content based on specific criteria and dimensions has become increasingly critical for applications such as video-theme consistency assessment and visual quality scoring. However, existing methods often suffer from imprecise results and inefficient loss calculation, which limit the focus of the model on key evaluation indicators. To address this, we propose IOVQA(Integer-only VQA), a novel fine-tuning approach tailored for VLMs to enhance their performance in video quality assessment tasks. The key innovation of IOVQA lies in its label construction and its targeted loss calculation mechanism. Specifically, during dataset curation, we constrain the model's output to integers within the range of [10,50], ensuring numerical stability, and convert decimal Overall_MOS to integer before using them as labels. We also introduce a target-mask strategy: when computing the loss, only the first two-digit-integer of the label is unmasked, forcing the model to learn the critical components of the numerical evaluation. After fine-tuning the Qwen2.5-VL model using the constructed dataset, experimental results demonstrate that the proposed method significantly improves the model's accuracy and consistency in the VQA task, ranking 3rd in VQualA 2025 GenAI-Bench AIGC Video Quality Assessment Challenge -- Track I. Our work highlights the effectiveness of merely leaving integer labels during fine-tuning, providing an effective idea for optimizing VLMs in quantitative evaluation scenarios.
>
---
#### [new 077] CoFi: A Fast Coarse-to-Fine Few-Shot Pipeline for Glomerular Basement Membrane Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11469v1](http://arxiv.org/pdf/2508.11469v1)**

> **作者:** Hongjin Fang; Daniel Reisenbüchler; Kenji Ikemura; Mert R. Sabuncu; Yihe Yang; Ruining Deng
>
> **摘要:** Accurate segmentation of the glomerular basement membrane (GBM) in electron microscopy (EM) images is fundamental for quantifying membrane thickness and supporting the diagnosis of various kidney diseases. While supervised deep learning approaches achieve high segmentation accuracy, their reliance on extensive pixel-level annotation renders them impractical for clinical workflows. Few-shot learning can reduce this annotation burden but often struggles to capture the fine structural details necessary for GBM analysis. In this study, we introduce CoFi, a fast and efficient coarse-to-fine few-shot segmentation pipeline designed for GBM delineation in EM images. CoFi first trains a lightweight neural network using only three annotated images to produce an initial coarse segmentation mask. This mask is then automatically processed to generate high-quality point prompts with morphology-aware pruning, which are subsequently used to guide SAM in refining the segmentation. The proposed method achieved exceptional GBM segmentation performance, with a Dice coefficient of 74.54% and an inference speed of 1.9 FPS. We demonstrate that CoFi not only alleviates the annotation and computational burdens associated with conventional methods, but also achieves accurate and reliable segmentation results. The pipeline's speed and annotation efficiency make it well-suited for research and hold strong potential for clinical applications in renal pathology. The pipeline is publicly available at: https://github.com/ddrrnn123/CoFi.
>
---
#### [new 078] MM-R1: Unleashing the Power of Unified Multimodal Large Language Models for Personalized Image Generation
- **分类: cs.CV**

- **简介: 该论文属于多模态任务，解决统一MLLM在个性化图像生成中的难题。通过X-CoT和GRPO方法，实现零样本高保真图像生成。**

- **链接: [http://arxiv.org/pdf/2508.11433v1](http://arxiv.org/pdf/2508.11433v1)**

> **作者:** Qian Liang; Yujia Wu; Kuncheng Li; Jiwei Wei; Shiyuan He; Jinyu Guo; Ning Xie
>
> **摘要:** Multimodal Large Language Models (MLLMs) with unified architectures excel across a wide range of vision-language tasks, yet aligning them with personalized image generation remains a significant challenge. Existing methods for MLLMs are frequently subject-specific, demanding a data-intensive fine-tuning process for every new subject, which limits their scalability. In this paper, we introduce MM-R1, a framework that integrates a cross-modal Chain-of-Thought (X-CoT) reasoning strategy to unlock the inherent potential of unified MLLMs for personalized image generation. Specifically, we structure personalization as an integrated visual reasoning and generation process: (1) grounding subject concepts by interpreting and understanding user-provided images and contextual cues, and (2) generating personalized images conditioned on both the extracted subject representations and user prompts. To further enhance the reasoning capability, we adopt Grouped Reward Proximal Policy Optimization (GRPO) to explicitly align the generation. Experiments demonstrate that MM-R1 unleashes the personalization capability of unified MLLMs to generate images with high subject fidelity and strong text alignment in a zero-shot manner.
>
---
#### [new 079] UAV-VL-R1: Generalizing Vision-Language Models via Supervised Fine-Tuning and Multi-Stage GRPO for UAV Visual Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11196v1](http://arxiv.org/pdf/2508.11196v1)**

> **作者:** Jiajin Guan; Haibo Mei; Bonan Zhang; Dan Liu; Yuanshuang Fu; Yue Zhang
>
> **摘要:** Recent advances in vision-language models (VLMs) have demonstrated strong generalization in natural image tasks. However, their performance often degrades on unmanned aerial vehicle (UAV)-based aerial imagery, which features high resolution, complex spatial semantics, and strict real-time constraints. These challenges limit the applicability of general-purpose VLMs to structured aerial reasoning tasks. To address these challenges, we propose UAV-VL-R1, a lightweight VLM explicitly designed for aerial visual reasoning. It is trained using a hybrid method that combines supervised fine-tuning (SFT) and multi-stage reinforcement learning (RL). We leverage the group relative policy optimization (GRPO) algorithm to promote structured and interpretable reasoning through rule-guided rewards and intra-group policy alignment. To support model training and evaluation, we introduce a high-resolution visual question answering dataset named HRVQA-VL, which consists of 50,019 annotated samples covering eight UAV-relevant reasoning tasks, including object counting, transportation recognition, and spatial scene inference. Experimental results show that UAV-VL-R1 achieves a 48.17% higher zero-shot accuracy than the Qwen2-VL-2B-Instruct baseline and even outperforms its 72B-scale variant, which is 36x larger, on multiple tasks. Ablation studies reveal that while SFT improves semantic alignment, it may reduce reasoning diversity in mathematical tasks. GRPO-based RL compensates for this limitation by enhancing logical flexibility and the robustness of inference. Additionally, UAV-VL-R1 requires only 3.9GB of memory under FP16 inference and can be quantized to 2.5GB with INT8, supporting real-time deployment on resource-constrained UAV platforms.
>
---
#### [new 080] SelfAdapt: Unsupervised Domain Adaptation of Cell Segmentation Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于细胞分割任务，解决模型在不同数据域上性能下降的问题。通过无监督域适应方法SelfAdapt提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.11411v1](http://arxiv.org/pdf/2508.11411v1)**

> **作者:** Fabian H. Reith; Jannik Franzen; Dinesh R. Palli; J. Lorenz Rumberger; Dagmar Kainmueller
>
> **备注:** 8 pages, 3 figures. To appear in the proceedings of the BioImage Computing (BIC) Workshop @ ICCVW 2025. This is the accepted author manuscript (camera-ready version)
>
> **摘要:** Deep neural networks have become the go-to method for biomedical instance segmentation. Generalist models like Cellpose demonstrate state-of-the-art performance across diverse cellular data, though their effectiveness often degrades on domains that differ from their training data. While supervised fine-tuning can address this limitation, it requires annotated data that may not be readily available. We propose SelfAdapt, a method that enables the adaptation of pre-trained cell segmentation models without the need for labels. Our approach builds upon student-teacher augmentation consistency training, introducing L2-SP regularization and label-free stopping criteria. We evaluate our method on the LiveCell and TissueNet datasets, demonstrating relative improvements in AP0.5 of up to 29.64% over baseline Cellpose. Additionally, we show that our unsupervised adaptation can further improve models that were previously fine-tuned with supervision. We release SelfAdapt as an easy-to-use extension of the Cellpose framework. The code for our method is publicly available at https: //github.com/Kainmueller-Lab/self_adapt.
>
---
#### [new 081] Index-Aligned Query Distillation for Transformer-based Incremental Object Detection
- **分类: cs.CV**

- **简介: 该论文属于增量目标检测任务，解决模型在学习新类别时遗忘旧类别知识的问题。提出IAQD方法，通过索引对齐的查询蒸馏保留旧知识，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2508.11339v1](http://arxiv.org/pdf/2508.11339v1)**

> **作者:** Mingxiao Ma; Shunyao Zhu; Guoliang Kang
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Incremental object detection (IOD) aims to continuously expand the capability of a model to detect novel categories while preserving its performance on previously learned ones. When adopting a transformer-based detection model to perform IOD, catastrophic knowledge forgetting may inevitably occur, meaning the detection performance on previously learned categories may severely degenerate. Previous typical methods mainly rely on knowledge distillation (KD) to mitigate the catastrophic knowledge forgetting of transformer-based detection models. Specifically, they utilize Hungarian Matching to build a correspondence between the queries of the last-phase and current-phase detection models and align the classifier and regressor outputs between matched queries to avoid knowledge forgetting. However, we observe that in IOD task, Hungarian Matching is not a good choice. With Hungarian Matching, the query of the current-phase model may match different queries of the last-phase model at different iterations during KD. As a result, the knowledge encoded in each query may be reshaped towards new categories, leading to the forgetting of previously encoded knowledge of old categories. Based on our observations, we propose a new distillation approach named Index-Aligned Query Distillation (IAQD) for transformer-based IOD. Beyond using Hungarian Matching, IAQD establishes a correspondence between queries of the previous and current phase models that have the same index. Moreover, we perform index-aligned distillation only on partial queries which are critical for the detection of previous categories. In this way, IAQD largely preserves the previous semantic and spatial encoding capabilities without interfering with the learning of new categories. Extensive experiments on representative benchmarks demonstrate that IAQD effectively mitigates knowledge forgetting, achieving new state-of-the-art performance.
>
---
#### [new 082] Topological Structure Description for Artcode Detection Using the Shape of Orientation Histogram
- **分类: cs.CV; cs.HC; cs.MM; I.4.10; I.5.4**

- **简介: 该论文属于Artcode检测任务，旨在识别具有拓扑结构的装饰标记。通过提出形状方向直方图特征描述符，有效表征Artcode的拓扑结构并实现检测。**

- **链接: [http://arxiv.org/pdf/2508.10942v1](http://arxiv.org/pdf/2508.10942v1)**

> **作者:** Liming Xu; Dave Towey; Andrew P. French; Steve Benford
>
> **备注:** This work is an extension of an ACM MM'17 workshop paper (Xu et al, 2017), which was completed in late 2017 and early 2018 during the first author's doctoral studies at the University of Nottingham. This paper includes 42 pages, 25 figures, 7 tables, and 13,536 words
>
> **摘要:** The increasing ubiquity of smartphones and resurgence of VR/AR techniques, it is expected that our everyday environment may soon be decorating with objects connecting with virtual elements. Alerting to the presence of these objects is therefore the first step for motivating follow-up further inspection and triggering digital material attached to the objects. This work studies a special kind of these objects -- Artcodes -- a human-meaningful and machine-readable decorative markers that camouflage themselves with freeform appearance by encoding information into their topology. We formulate this problem of recongising the presence of Artcodes as Artcode proposal detection, a distinct computer vision task that classifies topologically similar but geometrically and semantically different objects as a same class. To deal with this problem, we propose a new feature descriptor, called the shape of orientation histogram, to describe the generic topological structure of an Artcode. We collect datasets and conduct comprehensive experiments to evaluate the performance of the Artcode detection proposer built upon this new feature vector. Our experimental results show the feasibility of the proposed feature vector for representing topological structures and the effectiveness of the system for detecting Artcode proposals. Although this work is an initial attempt to develop a feature-based system for detecting topological objects like Artcodes, it would open up new interaction opportunities and spark potential applications of topological object detection.
>
---
#### [new 083] DashCam Video: A complementary low-cost data stream for on-demand forest-infrastructure system monitoring
- **分类: cs.CV; cs.ET**

- **简介: 该论文属于城市基础设施监测任务，旨在利用车载摄像头数据实现低成本、实时的物体定位与结构评估，解决传统方法成本高、效率低的问题。**

- **链接: [http://arxiv.org/pdf/2508.11591v1](http://arxiv.org/pdf/2508.11591v1)**

> **作者:** Durga Joshi; Chandi Witharana; Robert Fahey; Thomas Worthley; Zhe Zhu; Diego Cerrai
>
> **备注:** 35 Pages, 15 figures
>
> **摘要:** Our study introduces a novel, low-cost, and reproducible framework for real-time, object-level structural assessment and geolocation of roadside vegetation and infrastructure with commonly available but underutilized dashboard camera (dashcam) video data. We developed an end-to-end pipeline that combines monocular depth estimation, depth error correction, and geometric triangulation to generate accurate spatial and structural data from street-level video streams from vehicle-mounted dashcams. Depth maps were first estimated using a state-of-the-art monocular depth model, then refined via a gradient-boosted regression framework to correct underestimations, particularly for distant objects. The depth correction model achieved strong predictive performance (R2 = 0.92, MAE = 0.31 on transformed scale), significantly reducing bias beyond 15 m. Further, object locations were estimated using GPS-based triangulation, while object heights were calculated using pin hole camera geometry. Our method was evaluated under varying conditions of camera placement and vehicle speed. Low-speed vehicle with inside camera gave the highest accuracy, with mean geolocation error of 2.83 m, and mean absolute error (MAE) in height estimation of 2.09 m for trees and 0.88 m for poles. To the best of our knowledge, it is the first framework to combine monocular depth modeling, triangulated GPS-based geolocation, and real-time structural assessment for urban vegetation and infrastructure using consumer-grade video data. Our approach complements conventional RS methods, such as LiDAR and image by offering a fast, real-time, and cost-effective solution for object-level monitoring of vegetation risks and infrastructure exposure, making it especially valuable for utility companies, and urban planners aiming for scalable and frequent assessments in dynamic urban environments.
>
---
#### [new 084] TrajSV: A Trajectory-based Model for Sports Video Representations and Applications
- **分类: cs.CV; cs.IR**

- **简介: 该论文属于体育视频分析任务，解决数据不足和缺乏有效轨迹框架的问题。提出TrajSV模型，通过轨迹信息提升视频表示，应用于视频检索、动作定位和视频生成。**

- **链接: [http://arxiv.org/pdf/2508.11569v1](http://arxiv.org/pdf/2508.11569v1)**

> **作者:** Zheng Wang; Shihao Xu; Wei Shi
>
> **备注:** This paper has been accepted by TCSVT
>
> **摘要:** Sports analytics has received significant attention from both academia and industry in recent years. Despite the growing interest and efforts in this field, several issues remain unresolved, including (1) data unavailability, (2) lack of an effective trajectory-based framework, and (3) requirement for sufficient supervision labels. In this paper, we present TrajSV, a trajectory-based framework that addresses various issues in existing studies. TrajSV comprises three components: data preprocessing, Clip Representation Network (CRNet), and Video Representation Network (VRNet). The data preprocessing module extracts player and ball trajectories from sports broadcast videos. CRNet utilizes a trajectory-enhanced Transformer module to learn clip representations based on these trajectories. Additionally, VRNet learns video representations by aggregating clip representations and visual features with an encoder-decoder architecture. Finally, a triple contrastive loss is introduced to optimize both video and clip representations in an unsupervised manner. The experiments are conducted on three broadcast video datasets to verify the effectiveness of TrajSV for three types of sports (i.e., soccer, basketball, and volleyball) with three downstream applications (i.e., sports video retrieval, action spotting, and video captioning). The results demonstrate that TrajSV achieves state-of-the-art performance in sports video retrieval, showcasing a nearly 70% improvement. It outperforms baselines in action spotting, achieving state-of-the-art results in 9 out of 17 action categories, and demonstrates a nearly 20% improvement in video captioning. Additionally, we introduce a deployed system along with the three applications based on TrajSV.
>
---
#### [new 085] Semi-supervised Image Dehazing via Expectation-Maximization and Bidirectional Brownian Bridge Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于图像去雾任务，解决真实场景下缺乏配对数据的问题。通过半监督学习和扩散模型，提升去雾效果。**

- **链接: [http://arxiv.org/pdf/2508.11165v1](http://arxiv.org/pdf/2508.11165v1)**

> **作者:** Bing Liu; Le Wang; Mingming Liu; Hao Liu; Rui Yao; Yong Zhou; Peng Liu; Tongqiang Xia
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Existing dehazing methods deal with real-world haze images with difficulty, especially scenes with thick haze. One of the main reasons is the lack of real-world paired data and robust priors. To avoid the costly collection of paired hazy and clear images, we propose an efficient semi-supervised image dehazing method via Expectation-Maximization and Bidirectional Brownian Bridge Diffusion Models (EM-B3DM) with a two-stage learning scheme. In the first stage, we employ the EM algorithm to decouple the joint distribution of paired hazy and clear images into two conditional distributions, which are then modeled using a unified Brownian Bridge diffusion model to directly capture the structural and content-related correlations between hazy and clear images. In the second stage, we leverage the pre-trained model and large-scale unpaired hazy and clear images to further improve the performance of image dehazing. Additionally, we introduce a detail-enhanced Residual Difference Convolution block (RDC) to capture gradient-level information, significantly enhancing the model's representation capability. Extensive experiments demonstrate that our EM-B3DM achieves superior or at least comparable performance to state-of-the-art methods on both synthetic and real-world datasets.
>
---
#### [new 086] Analysis of the Compaction Behavior of Textile Reinforcements in Low-Resolution In-Situ CT Scans via Machine-Learning and Descriptor-Based Methods
- **分类: cs.CV; cond-mat.mtrl-sci; physics.app-ph**

- **链接: [http://arxiv.org/pdf/2508.10943v1](http://arxiv.org/pdf/2508.10943v1)**

> **作者:** Christian Düreth; Jan Condé-Wolter; Marek Danczak; Karsten Tittmann; Jörn Jaschinski; Andreas Hornig; Maik Gude
>
> **备注:** submitted to Elsevier Composite Part C: Open Access (JCOMC-D-25-00212), 16 pages, 8 Figures, and 3 Tables
>
> **摘要:** A detailed understanding of material structure across multiple scales is essential for predictive modeling of textile-reinforced composites. Nesting -- characterized by the interlocking of adjacent fabric layers through local interpenetration and misalignment of yarns -- plays a critical role in defining mechanical properties such as stiffness, permeability, and damage tolerance. This study presents a framework to quantify nesting behavior in dry textile reinforcements under compaction using low-resolution computed tomography (CT). In-situ compaction experiments were conducted on various stacking configurations, with CT scans acquired at 20.22 $\mu$m per voxel resolution. A tailored 3D{-}UNet enabled semantic segmentation of matrix, weft, and fill phases across compaction stages corresponding to fiber volume contents of 50--60 %. The model achieved a minimum mean Intersection-over-Union of 0.822 and an $F1$ score of 0.902. Spatial structure was subsequently analyzed using the two-point correlation function $S_2$, allowing for probabilistic extraction of average layer thickness and nesting degree. The results show strong agreement with micrograph-based validation. This methodology provides a robust approach for extracting key geometrical features from industrially relevant CT data and establishes a foundation for reverse modeling and descriptor-based structural analysis of composite preforms.
>
---
#### [new 087] Versatile Video Tokenization with Generative 2D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于视频处理任务，解决传统视频令牌化方法灵活性不足的问题。提出GVT模型，利用生成式2D高斯点云实现更高效的时空令牌表示。**

- **链接: [http://arxiv.org/pdf/2508.11183v1](http://arxiv.org/pdf/2508.11183v1)**

> **作者:** Zhenghao Chen; Zicong Chen; Lei Liu; Yiming Wu; Dong Xu
>
> **摘要:** Video tokenization procedure is critical for a wide range of video processing tasks. Most existing approaches directly transform video into fixed-grid and patch-wise tokens, which exhibit limited versatility. Spatially, uniformly allocating a fixed number of tokens often leads to over-encoding in low-information regions. Temporally, reducing redundancy remains challenging without explicitly distinguishing between static and dynamic content. In this work, we propose the Gaussian Video Transformer (GVT), a versatile video tokenizer built upon a generative 2D Gaussian Splatting (2DGS) strategy. We first extract latent rigid features from a video clip and represent them with a set of 2D Gaussians generated by our proposed Spatio-Temporal Gaussian Embedding (STGE) mechanism in a feed-forward manner. Such generative 2D Gaussians not only enhance spatial adaptability by assigning higher (resp., lower) rendering weights to regions with higher (resp., lower) information content during rasterization, but also improve generalization by avoiding per-video optimization.To enhance the temporal versatility, we introduce a Gaussian Set Partitioning (GSP) strategy that separates the 2D Gaussians into static and dynamic sets, which explicitly model static content shared across different time-steps and dynamic content specific to each time-step, enabling a compact representation.We primarily evaluate GVT on the video reconstruction, while also assessing its performance on action recognition and compression using the UCF101, Kinetics, and DAVIS datasets. Extensive experiments demonstrate that GVT achieves a state-of-the-art video reconstruction quality, outperforms the baseline MAGVIT-v2 in action recognition, and delivers comparable compression performance.
>
---
#### [new 088] Empowering Multimodal LLMs with External Tools: A Comprehensive Survey
- **分类: cs.CV; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2508.10955v1](http://arxiv.org/pdf/2508.10955v1)**

> **作者:** Wenbin An; Jiahao Nie; Yaqiang Wu; Feng Tian; Shijian Lu; Qinghua Zheng
>
> **备注:** 21 pages, 361 references
>
> **摘要:** By integrating the perception capabilities of multimodal encoders with the generative power of Large Language Models (LLMs), Multimodal Large Language Models (MLLMs), exemplified by GPT-4V, have achieved great success in various multimodal tasks, pointing toward a promising pathway to artificial general intelligence. Despite this progress, the limited quality of multimodal data, poor performance on many complex downstream tasks, and inadequate evaluation protocols continue to hinder the reliability and broader applicability of MLLMs across diverse domains. Inspired by the human ability to leverage external tools for enhanced reasoning and problem-solving, augmenting MLLMs with external tools (e.g., APIs, expert models, and knowledge bases) offers a promising strategy to overcome these challenges. In this paper, we present a comprehensive survey on leveraging external tools to enhance MLLM performance. Our discussion is structured along four key dimensions about external tools: (1) how they can facilitate the acquisition and annotation of high-quality multimodal data; (2) how they can assist in improving MLLM performance on challenging downstream tasks; (3) how they enable comprehensive and accurate evaluation of MLLMs; (4) the current limitations and future directions of tool-augmented MLLMs. Through this survey, we aim to underscore the transformative potential of external tools in advancing MLLM capabilities, offering a forward-looking perspective on their development and applications. The project page of this paper is publicly available athttps://github.com/Lackel/Awesome-Tools-for-MLLMs.
>
---
#### [new 089] Reinforcing Video Reasoning Segmentation to Think Before It Segments
- **分类: cs.CV**

- **简介: 该论文属于视频语义分割任务，解决传统方法在时空推理和可解释性上的不足。提出Veason-R1模型，通过强化学习和思维链训练提升分割性能与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.11538v1](http://arxiv.org/pdf/2508.11538v1)**

> **作者:** Sitong Gong; Lu Zhang; Yunzhi Zhuge; Xu Jia; Pingping Zhang; Huchuan Lu
>
> **备注:** 12 pages
>
> **摘要:** Video reasoning segmentation (VRS) endeavors to delineate referred objects in videos guided by implicit instructions that encapsulate human intent and temporal logic. Previous approaches leverage large vision language models (LVLMs) to encode object semantics into <SEG> tokens for mask prediction. However, this paradigm suffers from limited interpretability during inference and suboptimal performance due to inadequate spatiotemporal reasoning. Drawing inspiration from seminal breakthroughs in reinforcement learning, we introduce Veason-R1, a specialized LVLM for VRS that emphasizes structured reasoning in segmentation. Veason-R1 is trained through Group Relative Policy Optimization (GRPO) augmented with Chain-of-Thought (CoT) initialization. To begin with, we curate high-quality CoT training data to instill structured reasoning trajectories, bridging video-level semantics and frame-level spatial grounding, yielding the supervised fine-tuned model Veason-SFT. Subsequently, GRPO fine-tuning encourages efficient exploration of the reasoning space by optimizing reasoning chains. To this end, we incorporate a holistic reward mechanism that synergistically enhances spatial alignment and temporal consistency, bolstering keyframe localization and fine-grained grounding. Comprehensive empirical evaluations demonstrate that Veason-R1 achieves state-of-the-art performance on multiple benchmarks, surpassing prior art by significant margins (e.g., +1.3 J &F in ReVOS and +10.0 J &F in ReasonVOS), while exhibiting robustness to hallucinations (+8.8 R). Our code and model weights will be available at Veason-R1.
>
---
#### [new 090] Inside Knowledge: Graph-based Path Generation with Explainable Data Augmentation and Curriculum Learning for Visual Indoor Navigation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉室内导航任务，解决无GPS环境下的路径预测问题。提出基于图的路径生成方法，结合数据增强和课程学习，实现仅依赖视觉的实时导航。**

- **链接: [http://arxiv.org/pdf/2508.11446v1](http://arxiv.org/pdf/2508.11446v1)**

> **作者:** Daniel Airinei; Elena Burceanu; Marius Leordeanu
>
> **备注:** Accepted at the International Conference on Computer Vision Workshops 2025
>
> **摘要:** Indoor navigation is a difficult task, as it generally comes with poor GPS access, forcing solutions to rely on other sources of information. While significant progress continues to be made in this area, deployment to production applications is still lacking, given the complexity and additional requirements of current solutions. Here, we introduce an efficient, real-time and easily deployable deep learning approach, based on visual input only, that can predict the direction towards a target from images captured by a mobile device. Our technical approach, based on a novel graph-based path generation method, combined with explainable data augmentation and curriculum learning, includes contributions that make the process of data collection, annotation and training, as automatic as possible, efficient and robust. On the practical side, we introduce a novel largescale dataset, with video footage inside a relatively large shopping mall, in which each frame is annotated with the correct next direction towards different specific target destinations. Different from current methods, ours relies solely on vision, avoiding the need of special sensors, additional markers placed along the path, knowledge of the scene map or internet access. We also created an easy to use application for Android, which we plan to make publicly available. We make all our data and code available along with visual demos on our project site
>
---
#### [new 091] A Real-time Concrete Crack Detection and Segmentation Model Based on YOLOv11
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11517v1](http://arxiv.org/pdf/2508.11517v1)**

> **作者:** Shaoze Huang; Qi Liu; Chao Chen; Yuhang Chen
>
> **摘要:** Accelerated aging of transportation infrastructure in the rapidly developing Yangtze River Delta region necessitates efficient concrete crack detection, as crack deterioration critically compromises structural integrity and regional economic growth. To overcome the limitations of inefficient manual inspection and the suboptimal performance of existing deep learning models, particularly for small-target crack detection within complex backgrounds, this paper proposes YOLOv11-KW-TA-FP, a multi-task concrete crack detection and segmentation model based on the YOLOv11n architecture. The proposed model integrates a three-stage optimization framework: (1) Embedding dynamic KernelWarehouse convolution (KWConv) within the backbone network to enhance feature representation through a dynamic kernel sharing mechanism; (2) Incorporating a triple attention mechanism (TA) into the feature pyramid to strengthen channel-spatial interaction modeling; and (3) Designing an FP-IoU loss function to facilitate adaptive bounding box regression penalization. Experimental validation demonstrates that the enhanced model achieves significant performance improvements over the baseline, attaining 91.3% precision, 76.6% recall, and 86.4% mAP@50. Ablation studies confirm the synergistic efficacy of the proposed modules. Furthermore, robustness tests indicate stable performance under conditions of data scarcity and noise interference. This research delivers an efficient computer vision solution for automated infrastructure inspection, exhibiting substantial practical engineering value.
>
---
#### [new 092] G-CUT3R: Guided 3D Reconstruction with Camera and Depth Prior Integration
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D场景重建任务，旨在利用深度、相机校准等先验信息提升重建效果。通过改进CUT3R模型，融合多模态特征，提高重建精度与灵活性。**

- **链接: [http://arxiv.org/pdf/2508.11379v1](http://arxiv.org/pdf/2508.11379v1)**

> **作者:** Ramil Khafizov; Artem Komarichev; Ruslan Rakhimov; Peter Wonka; Evgeny Burnaev
>
> **摘要:** We introduce G-CUT3R, a novel feed-forward approach for guided 3D scene reconstruction that enhances the CUT3R model by integrating prior information. Unlike existing feed-forward methods that rely solely on input images, our method leverages auxiliary data, such as depth, camera calibrations, or camera positions, commonly available in real-world scenarios. We propose a lightweight modification to CUT3R, incorporating a dedicated encoder for each modality to extract features, which are fused with RGB image tokens via zero convolution. This flexible design enables seamless integration of any combination of prior information during inference. Evaluated across multiple benchmarks, including 3D reconstruction and other multi-view tasks, our approach demonstrates significant performance improvements, showing its ability to effectively utilize available priors while maintaining compatibility with varying input modalities.
>
---
#### [new 093] Hierarchical Graph Feature Enhancement with Adaptive Frequency Modulation for Visual Recognition
- **分类: cs.CV**

- **简介: 该论文属于视觉识别任务，旨在解决CNN在建模复杂拓扑关系和非局部语义上的不足。提出HGFE框架，结合图结构与自适应频率调制，提升特征表示与结构感知能力。**

- **链接: [http://arxiv.org/pdf/2508.11497v1](http://arxiv.org/pdf/2508.11497v1)**

> **作者:** Feiyue Zhao; Zhichao Zhang
>
> **摘要:** Convolutional neural networks (CNNs) have demonstrated strong performance in visual recognition tasks, but their inherent reliance on regular grid structures limits their capacity to model complex topological relationships and non-local semantics within images. To address this limita tion, we propose the hierarchical graph feature enhancement (HGFE), a novel framework that integrates graph-based rea soning into CNNs to enhance both structural awareness and feature representation. HGFE builds two complementary levels of graph structures: intra-window graph convolution to cap ture local spatial dependencies and inter-window supernode interactions to model global semantic relationships. Moreover, we introduce an adaptive frequency modulation module that dynamically balances low-frequency and high-frequency signal propagation, preserving critical edge and texture information while mitigating over-smoothing. The proposed HGFE module is lightweight, end-to-end trainable, and can be seamlessly integrated into standard CNN backbone networks. Extensive experiments on CIFAR-100 (classification), PASCAL VOC, and VisDrone (detection), as well as CrackSeg and CarParts (segmentation), validated the effectiveness of the HGFE in improving structural representation and enhancing overall recognition performance.
>
---
#### [new 094] Enhancing Supervised Composed Image Retrieval via Reasoning-Augmented Representation Engineering
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像检索任务，解决监督式组合图像检索问题。通过引入PMTFR框架和金字塔分块模块，提升视觉信息理解与检索效果。**

- **链接: [http://arxiv.org/pdf/2508.11272v1](http://arxiv.org/pdf/2508.11272v1)**

> **作者:** Jun Li; Kai Li; Shaoguo Liu; Tingting Gao
>
> **摘要:** Composed Image Retrieval (CIR) presents a significant challenge as it requires jointly understanding a reference image and a modified textual instruction to find relevant target images. Some existing methods attempt to use a two-stage approach to further refine retrieval results. However, this often requires additional training of a ranking model. Despite the success of Chain-of-Thought (CoT) techniques in reducing training costs for language models, their application in CIR tasks remains limited -- compressing visual information into text or relying on elaborate prompt designs. Besides, existing works only utilize it for zero-shot CIR, as it is challenging to achieve satisfactory results in supervised CIR with a well-trained model. In this work, we proposed a framework that includes the Pyramid Matching Model with Training-Free Refinement (PMTFR) to address these challenges. Through a simple but effective module called Pyramid Patcher, we enhanced the Pyramid Matching Model's understanding of visual information at different granularities. Inspired by representation engineering, we extracted representations from COT data and injected them into the LVLMs. This approach allowed us to obtain refined retrieval scores in the Training-Free Refinement paradigm without relying on explicit textual reasoning, further enhancing performance. Extensive experiments on CIR benchmarks demonstrate that PMTFR surpasses state-of-the-art methods in supervised CIR tasks. The code will be made public.
>
---
#### [new 095] HistoViT: Vision Transformer for Accurate and Scalable Histopathological Cancer Diagnosis
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于癌症分类任务，旨在解决病理图像中癌症准确诊断的问题。提出基于ViT的框架，提升分类准确率与可扩展性。**

- **链接: [http://arxiv.org/pdf/2508.11181v1](http://arxiv.org/pdf/2508.11181v1)**

> **作者:** Faisal Ahmed
>
> **备注:** 13 pages, 3 Figures
>
> **摘要:** Accurate and scalable cancer diagnosis remains a critical challenge in modern pathology, particularly for malignancies such as breast, prostate, bone, and cervical, which exhibit complex histological variability. In this study, we propose a transformer-based deep learning framework for multi-class tumor classification in histopathological images. Leveraging a fine-tuned Vision Transformer (ViT) architecture, our method addresses key limitations of conventional convolutional neural networks, offering improved performance, reduced preprocessing requirements, and enhanced scalability across tissue types. To adapt the model for histopathological cancer images, we implement a streamlined preprocessing pipeline that converts tiled whole-slide images into PyTorch tensors and standardizes them through data normalization. This ensures compatibility with the ViT architecture and enhances both convergence stability and overall classification performance. We evaluate our model on four benchmark datasets: ICIAR2018 (breast), SICAPv2 (prostate), UT-Osteosarcoma (bone), and SipakMed (cervical) dataset -- demonstrating consistent outperformance over existing deep learning methods. Our approach achieves classification accuracies of 99.32%, 96.92%, 95.28%, and 96.94% for breast, prostate, bone, and cervical cancers respectively, with area under the ROC curve (AUC) scores exceeding 99% across all datasets. These results confirm the robustness, generalizability, and clinical potential of transformer-based architectures in digital pathology. Our work represents a significant advancement toward reliable, automated, and interpretable cancer diagnosis systems that can alleviate diagnostic burdens and improve healthcare outcomes.
>
---
#### [new 096] The Role of Radiographic Knee Alignment in Knee Replacement Outcomes and Opportunities for Artificial Intelligence-Driven Assessment
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.10941v1](http://arxiv.org/pdf/2508.10941v1)**

> **作者:** Zhisen Hu; David S. Johnson; Aleksei Tiulpin; Timothy F. Cootes; Claudia Lindner
>
> **摘要:** Prevalent knee osteoarthritis (OA) imposes substantial burden on health systems with no cure available. Its ultimate treatment is total knee replacement (TKR). Complications from surgery and recovery are difficult to predict in advance, and numerous factors may affect them. Radiographic knee alignment is one of the key factors that impacts TKR outcomes, affecting outcomes such as postoperative pain or function. Recently, artificial intelligence (AI) has been introduced to the automatic analysis of knee radiographs, for example, to automate knee alignment measurements. Existing review articles tend to focus on knee OA diagnosis and segmentation of bones or cartilages in MRI rather than exploring knee alignment biomarkers for TKR outcomes and their assessment. In this review, we first examine the current scoring protocols for evaluating TKR outcomes and potential knee alignment biomarkers associated with these outcomes. We then discuss existing AI-based approaches for generating knee alignment biomarkers from knee radiographs, and explore future directions for knee alignment assessment and TKR outcome prediction.
>
---
#### [new 097] Allen: Rethinking MAS Design through Step-Level Policy Autonomy
- **分类: cs.MA; cs.CV**

- **简介: 该论文属于多智能体系统设计任务，旨在提升策略自主性并平衡协作效率与控制。通过重新定义执行单元和四层状态架构实现目标。**

- **链接: [http://arxiv.org/pdf/2508.11294v1](http://arxiv.org/pdf/2508.11294v1)**

> **作者:** Qiangong Zhou; Zhiting Wang; Mingyou Yao; Zongyang Liu
>
> **摘要:** We introduce a new Multi-Agent System (MAS) - Allen, designed to address two core challenges in current MAS design: (1) improve system's policy autonomy, empowering agents to dynamically adapt their behavioral strategies, and (2) achieving the trade-off between collaborative efficiency, task supervision, and human oversight in complex network topologies. Our core insight is to redefine the basic execution unit in the MAS, allowing agents to autonomously form different patterns by combining these units. We have constructed a four-tier state architecture (Task, Stage, Agent, Step) to constrain system behavior from both task-oriented and execution-oriented perspectives. This achieves a unification of topological optimization and controllable progress. Allen grants unprecedented Policy Autonomy, while making a trade-off for the controllability of the collaborative structure. The project code has been open source at: https://github.com/motern88/Allen
>
---
#### [new 098] Match & Choose: Model Selection Framework for Fine-tuning Text-to-Image Diffusion Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于文本生成图像任务，解决预训练模型选择问题。提出M&C框架，通过匹配图高效预测最佳微调模型。**

- **链接: [http://arxiv.org/pdf/2508.10993v1](http://arxiv.org/pdf/2508.10993v1)**

> **作者:** Basile Lewandowski; Robert Birke; Lydia Y. Chen
>
> **摘要:** Text-to-image (T2I) models based on diffusion and transformer architectures advance rapidly. They are often pretrained on large corpora, and openly shared on a model platform, such as HuggingFace. Users can then build up AI applications, e.g., generating media contents, by adopting pretrained T2I models and fine-tuning them on the target dataset. While public pretrained T2I models facilitate the democratization of the models, users face a new challenge: which model can be best fine-tuned based on the target data domain? Model selection is well addressed in classification tasks, but little is known in (pretrained) T2I models and their performance indication on the target domain. In this paper, we propose the first model selection framework, M&C, which enables users to efficiently choose a pretrained T2I model from a model platform without exhaustively fine-tuning them all on the target dataset. The core of M&C is a matching graph, which consists of: (i) nodes of available models and profiled datasets, and (ii) edges of model-data and data-data pairs capturing the fine-tuning performance and data similarity, respectively. We then build a model that, based on the inputs of model/data feature, and, critically, the graph embedding feature, extracted from the matching graph, predicts the model achieving the best quality after fine-tuning for the target domain. We evaluate M&C on choosing across ten T2I models for 32 datasets against three baselines. Our results show that M&C successfully predicts the best model for fine-tuning in 61.3% of the cases and a closely performing model for the rest.
>
---
#### [new 099] Fluid Dynamics and Domain Reconstruction from Noisy Flow Images Using Physics-Informed Neural Networks and Quasi-Conformal Mapping
- **分类: math.NA; cs.CV; cs.NA**

- **简介: 该论文属于流体动力学图像重建任务，解决噪声流场图像的去噪与几何重构问题。通过物理信息神经网络和拟共形映射，联合优化速度场与流体区域。**

- **链接: [http://arxiv.org/pdf/2508.11216v1](http://arxiv.org/pdf/2508.11216v1)**

> **作者:** Han Zhang; Xue-Cheng Tai; Jean-Michel Morel; Raymond H. Chan
>
> **摘要:** Blood flow imaging provides important information for hemodynamic behavior within the vascular system and plays an essential role in medical diagnosis and treatment planning. However, obtaining high-quality flow images remains a significant challenge. In this work, we address the problem of denoising flow images that may suffer from artifacts due to short acquisition times or device-induced errors. We formulate this task as an optimization problem, where the objective is to minimize the discrepancy between the modeled velocity field, constrained to satisfy the Navier-Stokes equations, and the observed noisy velocity data. To solve this problem, we decompose it into two subproblems: a fluid subproblem and a geometry subproblem. The fluid subproblem leverages a Physics-Informed Neural Network to reconstruct the velocity field from noisy observations, assuming a fixed domain. The geometry subproblem aims to infer the underlying flow region by optimizing a quasi-conformal mapping that deforms a reference domain. These two subproblems are solved in an alternating Gauss-Seidel fashion, iteratively refining both the velocity field and the domain. Upon convergence, the framework yields a high-quality reconstruction of the flow image. We validate the proposed method through experiments on synthetic flow data in a converging channel geometry under varying levels of Gaussian noise, and on real-like flow data in an aortic geometry with signal-dependent noise. The results demonstrate the effectiveness and robustness of the approach. Additionally, ablation studies are conducted to assess the influence of key hyperparameters.
>
---
#### [new 100] Subcortical Masks Generation in CT Images via Ensemble-Based Cross-Domain Label Transfer
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决CT图像缺乏标注数据的问题。通过融合MRI模型生成高质量CT分割标签，构建首个公开的CT亚皮层分割数据集。**

- **链接: [http://arxiv.org/pdf/2508.11450v1](http://arxiv.org/pdf/2508.11450v1)**

> **作者:** Augustine X. W. Lee; Pak-Hei Yeung; Jagath C. Rajapakse
>
> **摘要:** Subcortical segmentation in neuroimages plays an important role in understanding brain anatomy and facilitating computer-aided diagnosis of traumatic brain injuries and neurodegenerative disorders. However, training accurate automatic models requires large amounts of labelled data. Despite the availability of publicly available subcortical segmentation datasets for Magnetic Resonance Imaging (MRI), a significant gap exists for Computed Tomography (CT). This paper proposes an automatic ensemble framework to generate high-quality subcortical segmentation labels for CT scans by leveraging existing MRI-based models. We introduce a robust ensembling pipeline to integrate them and apply it to unannotated paired MRI-CT data, resulting in a comprehensive CT subcortical segmentation dataset. Extensive experiments on multiple public datasets demonstrate the superior performance of our proposed framework. Furthermore, using our generated CT dataset, we train segmentation models that achieve improved performance on related segmentation tasks. To facilitate future research, we make our source code, generated dataset, and trained models publicly available at https://github.com/SCSE-Biomedical-Computing-Group/CT-Subcortical-Segmentation, marking the first open-source release for CT subcortical segmentation to the best of our knowledge.
>
---
#### [new 101] Visual Perception Engine: Fast and Flexible Multi-Head Inference for Robotic Vision Tasks
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文针对机器人视觉任务中的计算冗余和内存占用问题，提出VPEngine框架，通过共享基础模型和并行任务头实现高效多任务处理。**

- **链接: [http://arxiv.org/pdf/2508.11584v1](http://arxiv.org/pdf/2508.11584v1)**

> **作者:** Jakub Łucki; Jonathan Becktor; Georgios Georgakis; Robert Royce; Shehryar Khattak
>
> **备注:** 6 pages, 6 figures, 2 tables
>
> **摘要:** Deploying multiple machine learning models on resource-constrained robotic platforms for different perception tasks often results in redundant computations, large memory footprints, and complex integration challenges. In response, this work presents Visual Perception Engine (VPEngine), a modular framework designed to enable efficient GPU usage for visual multitasking while maintaining extensibility and developer accessibility. Our framework architecture leverages a shared foundation model backbone that extracts image representations, which are efficiently shared, without any unnecessary GPU-CPU memory transfers, across multiple specialized task-specific model heads running in parallel. This design eliminates the computational redundancy inherent in feature extraction component when deploying traditional sequential models while enabling dynamic task prioritization based on application demands. We demonstrate our framework's capabilities through an example implementation using DINOv2 as the foundation model with multiple task (depth, object detection and semantic segmentation) heads, achieving up to 3x speedup compared to sequential execution. Building on CUDA Multi-Process Service (MPS), VPEngine offers efficient GPU utilization and maintains a constant memory footprint while allowing per-task inference frequencies to be adjusted dynamically during runtime. The framework is written in Python and is open source with ROS2 C++ (Humble) bindings for ease of use by the robotics community across diverse robotic platforms. Our example implementation demonstrates end-to-end real-time performance at $\geq$50 Hz on NVIDIA Jetson Orin AGX for TensorRT optimized models.
>
---
#### [new 102] Model Interpretability and Rationale Extraction by Input Mask Optimization
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于模型可解释性任务，旨在生成神经网络预测的可解释理由。通过输入掩码优化，提取简洁有效的解释，无需训练专用模型。**

- **链接: [http://arxiv.org/pdf/2508.11388v1](http://arxiv.org/pdf/2508.11388v1)**

> **作者:** Marc Brinner; Sina Zarriess
>
> **摘要:** Concurrent to the rapid progress in the development of neural-network based models in areas like natural language processing and computer vision, the need for creating explanations for the predictions of these black-box models has risen steadily. We propose a new method to generate extractive explanations for predictions made by neural networks, that is based on masking parts of the input which the model does not consider to be indicative of the respective class. The masking is done using gradient-based optimization combined with a new regularization scheme that enforces sufficiency, comprehensiveness and compactness of the generated explanation, three properties that are known to be desirable from the related field of rationale extraction in natural language processing. In this way, we bridge the gap between model interpretability and rationale extraction, thereby proving that the latter of which can be performed without training a specialized model, only on the basis of a trained classifier. We further apply the same method to image inputs and obtain high quality explanations for image classifications, which indicates that the conditions proposed for rationale extraction in natural language processing are more broadly applicable to different input types.
>
---
#### [new 103] Failures to Surface Harmful Contents in Video Large Language Models
- **分类: cs.MM; cs.CV**

- **简介: 该论文属于视频大语言模型安全任务，旨在解决模型遗漏有害内容的问题。研究揭示了模型设计缺陷，并提出攻击方法验证问题严重性。**

- **链接: [http://arxiv.org/pdf/2508.10974v1](http://arxiv.org/pdf/2508.10974v1)**

> **作者:** Yuxin Cao; Wei Song; Derui Wang; Jingling Xue; Jin Song Dong
>
> **备注:** 11 pages, 8 figures
>
> **摘要:** Video Large Language Models (VideoLLMs) are increasingly deployed on numerous critical applications, where users rely on auto-generated summaries while casually skimming the video stream. We show that this interaction hides a critical safety gap: if harmful content is embedded in a video, either as full-frame inserts or as small corner patches, state-of-the-art VideoLLMs rarely mention the harmful content in the output, despite its clear visibility to human viewers. A root-cause analysis reveals three compounding design flaws: (1) insufficient temporal coverage resulting from the sparse, uniformly spaced frame sampling used by most leading VideoLLMs, (2) spatial information loss introduced by aggressive token downsampling within sampled frames, and (3) encoder-decoder disconnection, whereby visual cues are only weakly utilized during text generation. Leveraging these insights, we craft three zero-query black-box attacks, aligning with these flaws in the processing pipeline. Our large-scale evaluation across five leading VideoLLMs shows that the harmfulness omission rate exceeds 90% in most cases. Even when harmful content is clearly present in all frames, these models consistently fail to identify it. These results underscore a fundamental vulnerability in current VideoLLMs' designs and highlight the urgent need for sampling strategies, token compression, and decoding mechanisms that guarantee semantic coverage rather than speed alone.
>
---
#### [new 104] Boosting the Robustness-Accuracy Trade-off of SNNs by Robust Temporal Self-Ensemble
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11279v1](http://arxiv.org/pdf/2508.11279v1)**

> **作者:** Jihang Wang; Dongcheng Zhao; Ruolin Chen; Qian Zhang; Yi Zeng
>
> **摘要:** Spiking Neural Networks (SNNs) offer a promising direction for energy-efficient and brain-inspired computing, yet their vulnerability to adversarial perturbations remains poorly understood. In this work, we revisit the adversarial robustness of SNNs through the lens of temporal ensembling, treating the network as a collection of evolving sub-networks across discrete timesteps. This formulation uncovers two critical but underexplored challenges-the fragility of individual temporal sub-networks and the tendency for adversarial vulnerabilities to transfer across time. To overcome these limitations, we propose Robust Temporal self-Ensemble (RTE), a training framework that improves the robustness of each sub-network while reducing the temporal transferability of adversarial perturbations. RTE integrates both objectives into a unified loss and employs a stochastic sampling strategy for efficient optimization. Extensive experiments across multiple benchmarks demonstrate that RTE consistently outperforms existing training methods in robust-accuracy trade-off. Additional analyses reveal that RTE reshapes the internal robustness landscape of SNNs, leading to more resilient and temporally diversified decision boundaries. Our study highlights the importance of temporal structure in adversarial learning and offers a principled foundation for building robust spiking models.
>
---
#### [new 105] Scene Graph-Guided Proactive Replanning for Failure-Resilient Embodied Agent
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人自主导航任务，解决环境变化导致的执行失败问题。通过场景图对比和轻量推理模块实现主动重规划，提升任务成功率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.11286v1](http://arxiv.org/pdf/2508.11286v1)**

> **作者:** Che Rin Yu; Daewon Chae; Dabin Seo; Sangwon Lee; Hyeongwoo Im; Jinkyu Kim
>
> **摘要:** When humans perform everyday tasks, we naturally adjust our actions based on the current state of the environment. For instance, if we intend to put something into a drawer but notice it is closed, we open it first. However, many autonomous robots lack this adaptive awareness. They often follow pre-planned actions that may overlook subtle yet critical changes in the scene, which can result in actions being executed under outdated assumptions and eventual failure. While replanning is critical for robust autonomy, most existing methods respond only after failures occur, when recovery may be inefficient or infeasible. While proactive replanning holds promise for preventing failures in advance, current solutions often rely on manually designed rules and extensive supervision. In this work, we present a proactive replanning framework that detects and corrects failures at subtask boundaries by comparing scene graphs constructed from current RGB-D observations against reference graphs extracted from successful demonstrations. When the current scene fails to align with reference trajectories, a lightweight reasoning module is activated to diagnose the mismatch and adjust the plan. Experiments in the AI2-THOR simulator demonstrate that our approach detects semantic and spatial mismatches before execution failures occur, significantly improving task success and robustness.
>
---
#### [new 106] StyleMM: Stylized 3D Morphable Face Model via Text-Driven Aligned Image Translation
- **分类: cs.GR; cs.AI; cs.CV; cs.MM; 51-04; I.3.8; I.4.9**

- **简介: 该论文属于3D人脸风格化任务，解决如何根据文本描述生成风格化3DMM的问题。通过文本引导的图像翻译和模型微调实现风格化人脸生成。**

- **链接: [http://arxiv.org/pdf/2508.11203v1](http://arxiv.org/pdf/2508.11203v1)**

> **作者:** Seungmi Lee; Kwan Yun; Junyong Noh
>
> **备注:** Pacific graphics 2025, CGF, 15 pages
>
> **摘要:** We introduce StyleMM, a novel framework that can construct a stylized 3D Morphable Model (3DMM) based on user-defined text descriptions specifying a target style. Building upon a pre-trained mesh deformation network and a texture generator for original 3DMM-based realistic human faces, our approach fine-tunes these models using stylized facial images generated via text-guided image-to-image (i2i) translation with a diffusion model, which serve as stylization targets for the rendered mesh. To prevent undesired changes in identity, facial alignment, or expressions during i2i translation, we introduce a stylization method that explicitly preserves the facial attributes of the source image. By maintaining these critical attributes during image stylization, the proposed approach ensures consistent 3D style transfer across the 3DMM parameter space through image-based training. Once trained, StyleMM enables feed-forward generation of stylized face meshes with explicit control over shape, expression, and texture parameters, producing meshes with consistent vertex connectivity and animatability. Quantitative and qualitative evaluations demonstrate that our approach outperforms state-of-the-art methods in terms of identity-level facial diversity and stylization capability. The code and videos are available at [kwanyun.github.io/stylemm_page](kwanyun.github.io/stylemm_page).
>
---
#### [new 107] Relative Position Matters: Trajectory Prediction and Planning with Polar Representation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11492v1](http://arxiv.org/pdf/2508.11492v1)**

> **作者:** Bozhou Zhang; Nan Song; Bingzhao Gao; Li Zhang
>
> **摘要:** Trajectory prediction and planning in autonomous driving are highly challenging due to the complexity of predicting surrounding agents' movements and planning the ego agent's actions in dynamic environments. Existing methods encode map and agent positions and decode future trajectories in Cartesian coordinates. However, modeling the relationships between the ego vehicle and surrounding traffic elements in Cartesian space can be suboptimal, as it does not naturally capture the varying influence of different elements based on their relative distances and directions. To address this limitation, we adopt the Polar coordinate system, where positions are represented by radius and angle. This representation provides a more intuitive and effective way to model spatial changes and relative relationships, especially in terms of distance and directional influence. Based on this insight, we propose Polaris, a novel method that operates entirely in Polar coordinates, distinguishing itself from conventional Cartesian-based approaches. By leveraging the Polar representation, this method explicitly models distance and direction variations and captures relative relationships through dedicated encoding and refinement modules, enabling more structured and spatially aware trajectory prediction and planning. Extensive experiments on the challenging prediction (Argoverse 2) and planning benchmarks (nuPlan) demonstrate that Polaris achieves state-of-the-art performance.
>
---
#### [new 108] Guiding WaveMamba with Frequency Maps for Image Debanding
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像去带任务，解决低比特率压缩导致的带状伪影问题。通过Wavelet状态空间模型和频率掩码图实现有效去带并保留细节。**

- **链接: [http://arxiv.org/pdf/2508.11331v1](http://arxiv.org/pdf/2508.11331v1)**

> **作者:** Xinyi Wang; Smaranda Tasmoc; Nantheera Anantrasirichai; Angeliki Katsenou
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Compression at low bitrates in modern codecs often introduces banding artifacts, especially in smooth regions such as skies. These artifacts degrade visual quality and are common in user-generated content due to repeated transcoding. We propose a banding restoration method that employs the Wavelet State Space Model and a frequency masking map to preserve high-frequency details. Furthermore, we provide a benchmark of open-source banding restoration methods and evaluate their performance on two public banding image datasets. Experimentation on the available datasets suggests that the proposed post-processing approach effectively suppresses banding compared to the state-of-the-art method (a DBI value of 0.082 on BAND-2k) while preserving image textures. Visual inspections of the results confirm this. Code and supplementary material are available at: https://github.com/xinyiW915/Debanding-PCS2025.
>
---
#### [new 109] Robust Convolution Neural ODEs via Contractivity-promoting regularization
- **分类: cs.LG; cs.CV; cs.SY; eess.SY**

- **简介: 该论文属于图像分类任务，旨在提升Convolutional NODEs的鲁棒性。通过引入收缩理论和正则化方法，增强模型对噪声和对抗攻击的抵抗能力。**

- **链接: [http://arxiv.org/pdf/2508.11432v1](http://arxiv.org/pdf/2508.11432v1)**

> **作者:** Muhammad Zakwan; Liang Xu; Giancarlo Ferrari-Trecate
>
> **备注:** Accepted in IEEE CDC2025, Rio de Janeiro, Brazil
>
> **摘要:** Neural networks can be fragile to input noise and adversarial attacks. In this work, we consider Convolutional Neural Ordinary Differential Equations (NODEs), a family of continuous-depth neural networks represented by dynamical systems, and propose to use contraction theory to improve their robustness. For a contractive dynamical system two trajectories starting from different initial conditions converge to each other exponentially fast. Contractive Convolutional NODEs can enjoy increased robustness as slight perturbations of the features do not cause a significant change in the output. Contractivity can be induced during training by using a regularization term involving the Jacobian of the system dynamics. To reduce the computational burden, we show that it can also be promoted using carefully selected weight regularization terms for a class of NODEs with slope-restricted activation functions. The performance of the proposed regularizers is illustrated through benchmark image classification tasks on MNIST and FashionMNIST datasets, where images are corrupted by different kinds of noise and attacks.
>
---
#### [new 110] SPG: Style-Prompting Guidance for Style-Specific Content Creation
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于文本到图像生成任务，解决视觉风格控制难题。提出SPG方法，通过风格噪声引导扩散过程，实现风格一致的图像生成。**

- **链接: [http://arxiv.org/pdf/2508.11476v1](http://arxiv.org/pdf/2508.11476v1)**

> **作者:** Qian Liang; Zichong Chen; Yang Zhou; Hui Huang
>
> **备注:** Accepted to the Journal track of Pacific Graphics 2025
>
> **摘要:** Although recent text-to-image (T2I) diffusion models excel at aligning generated images with textual prompts, controlling the visual style of the output remains a challenging task. In this work, we propose Style-Prompting Guidance (SPG), a novel sampling strategy for style-specific image generation. SPG constructs a style noise vector and leverages its directional deviation from unconditional noise to guide the diffusion process toward the target style distribution. By integrating SPG with Classifier-Free Guidance (CFG), our method achieves both semantic fidelity and style consistency. SPG is simple, robust, and compatible with controllable frameworks like ControlNet and IPAdapter, making it practical and widely applicable. Extensive experiments demonstrate the effectiveness and generality of our approach compared to state-of-the-art methods. Code is available at https://github.com/Rumbling281441/SPG.
>
---
#### [new 111] Temporally-Similar Structure-Aware Spatiotemporal Fusion of Satellite Images
- **分类: eess.SP; cs.CV**

- **简介: 该论文属于卫星图像时空融合任务，旨在解决噪声环境下结构信息丢失问题。通过引入TGTV和TGEC机制，提升融合效果与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.11259v1](http://arxiv.org/pdf/2508.11259v1)**

> **作者:** Ryosuke Isono; Shunsuke Ono
>
> **备注:** Submitted to IEEE Transactions on Geoscience and Remote Sensing. arXiv admin note: text overlap with arXiv:2308.00500
>
> **摘要:** This paper proposes a novel spatiotemporal (ST) fusion framework for satellite images, named Temporally-Similar Structure-Aware ST fusion (TSSTF). ST fusion is a promising approach to address the trade-off between the spatial and temporal resolution of satellite images. In real-world scenarios, observed satellite images are severely degraded by noise due to measurement equipment and environmental conditions. Consequently, some recent studies have focused on enhancing the robustness of ST fusion methods against noise. However, existing noise-robust ST fusion approaches often fail to capture fine spatial structure, leading to oversmoothing and artifacts. To address this issue, TSSTF introduces two key mechanisms: Temporally-Guided Total Variation (TGTV) and Temporally-Guided Edge Constraint (TGEC). TGTV is a novel regularization function that promotes spatial piecewise smoothness while preserving structural details, guided by a reference high spatial resolution image acquired on a nearby date. TGEC enforces consistency in edge locations between two temporally adjacent images, while allowing for spectral variations. We formulate the ST fusion task as a constrained optimization problem incorporating TGTV and TGEC, and develop an efficient algorithm based on a preconditioned primal-dual splitting method. Experimental results demonstrate that TSSTF performs comparably to state-of-the-art methods under noise-free conditions and outperforms them under noisy conditions. Additionally, we provide a comprehensive set of recommended parameter values that consistently yield high performance across diverse target regions and noise conditions, aiming to enhance reproducibility and practical utility.
>
---
#### [new 112] LKFMixer: Exploring Large Kernel Feature For Efficient Image Super-Resolution
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像超分辨率任务，旨在解决传统方法计算量大、难以轻量化的问题。提出LKFMixer模型，通过大卷积核和结构优化实现高效高质的图像超分辨率。**

- **链接: [http://arxiv.org/pdf/2508.11391v1](http://arxiv.org/pdf/2508.11391v1)**

> **作者:** Yinggan Tang; Quanwei Hu
>
> **摘要:** The success of self-attention (SA) in Transformer demonstrates the importance of non-local information to image super-resolution (SR), but the huge computing power required makes it difficult to implement lightweight models. To solve this problem, we propose a pure convolutional neural network (CNN) model, LKFMixer, which utilizes large convolutional kernel to simulate the ability of self-attention to capture non-local features. Specifically, we increase the kernel size to 31 to obtain the larger receptive field as possible, and reduce the parameters and computations by coordinate decomposition. Meanwhile, a spatial feature modulation block (SFMB) is designed to enhance the focus of feature information on both spatial and channel dimension. In addition, by introducing feature selection block (FSB), the model can adaptively adjust the weights between local features and non-local features. Extensive experiments show that the proposed LKFMixer family outperform other state-of-the-art (SOTA) methods in terms of SR performance and reconstruction quality. In particular, compared with SwinIR-light on Manga109 dataset, LKFMixer-L achieves 0.6dB PSNR improvement at $\times$4 scale, while the inference speed is $\times$5 times faster. The code is available at https://github.com/Supereeeee/LKFMixer.
>
---
#### [new 113] Semi-Supervised Learning with Online Knowledge Distillation for Skin Lesion Classification
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于皮肤病变分类任务，解决标注数据不足的问题。通过集成学习与在线知识蒸馏，提升模型性能并减少对大量标注数据的依赖。**

- **链接: [http://arxiv.org/pdf/2508.11511v1](http://arxiv.org/pdf/2508.11511v1)**

> **作者:** Siyamalan Manivannan
>
> **摘要:** Deep Learning has emerged as a promising approach for skin lesion analysis. However, existing methods mostly rely on fully supervised learning, requiring extensive labeled data, which is challenging and costly to obtain. To alleviate this annotation burden, this study introduces a novel semi-supervised deep learning approach that integrates ensemble learning with online knowledge distillation for enhanced skin lesion classification. Our methodology involves training an ensemble of convolutional neural network models, using online knowledge distillation to transfer insights from the ensemble to its members. This process aims to enhance the performance of each model within the ensemble, thereby elevating the overall performance of the ensemble itself. Post-training, any individual model within the ensemble can be deployed at test time, as each member is trained to deliver comparable performance to the ensemble. This is particularly beneficial in resource-constrained environments. Experimental results demonstrate that the knowledge-distilled individual model performs better than independently trained models. Our approach demonstrates superior performance on both the \emph{International Skin Imaging Collaboration} 2018 and 2019 public benchmark datasets, surpassing current state-of-the-art results. By leveraging ensemble learning and online knowledge distillation, our method reduces the need for extensive labeled data while providing a more resource-efficient solution for skin lesion classification in real-world scenarios.
>
---
#### [new 114] LD-LAudio-V1: Video-to-Long-Form-Audio Generation Extension with Dual Lightweight Adapters
- **分类: cs.SD; cs.AI; cs.CV; eess.AS**

- **简介: 该论文属于视频到长音频生成任务，解决长时音频同步与质量问题，提出LD-LAudio-V1模型，采用双轻量适配器提升生成效果。**

- **链接: [http://arxiv.org/pdf/2508.11074v1](http://arxiv.org/pdf/2508.11074v1)**

> **作者:** Haomin Zhang; Kristin Qi; Shuxin Yang; Zihao Chen; Chaofan Ding; Xinhan Di
>
> **备注:** Gen4AVC@ICCV: 1st Workshop on Generative AI for Audio-Visual Content Creation
>
> **摘要:** Generating high-quality and temporally synchronized audio from video content is essential for video editing and post-production tasks, enabling the creation of semantically aligned audio for silent videos. However, most existing approaches focus on short-form audio generation for video segments under 10 seconds or rely on noisy datasets for long-form video-to-audio zsynthesis. To address these limitations, we introduce LD-LAudio-V1, an extension of state-of-the-art video-to-audio models and it incorporates dual lightweight adapters to enable long-form audio generation. In addition, we release a clean and human-annotated video-to-audio dataset that contains pure sound effects without noise or artifacts. Our method significantly reduces splicing artifacts and temporal inconsistencies while maintaining computational efficiency. Compared to direct fine-tuning with short training videos, LD-LAudio-V1 achieves significant improvements across multiple metrics: $FD_{\text{passt}}$ 450.00 $\rightarrow$ 327.29 (+27.27%), $FD_{\text{panns}}$ 34.88 $\rightarrow$ 22.68 (+34.98%), $FD_{\text{vgg}}$ 3.75 $\rightarrow$ 1.28 (+65.87%), $KL_{\text{panns}}$ 2.49 $\rightarrow$ 2.07 (+16.87%), $KL_{\text{passt}}$ 1.78 $\rightarrow$ 1.53 (+14.04%), $IS_{\text{panns}}$ 4.17 $\rightarrow$ 4.30 (+3.12%), $IB_{\text{score}}$ 0.25 $\rightarrow$ 0.28 (+12.00%), $Energy\Delta10\text{ms}$ 0.3013 $\rightarrow$ 0.1349 (+55.23%), $Energy\Delta10\text{ms(vs.GT)}$ 0.0531 $\rightarrow$ 0.0288 (+45.76%), and $Sem.\,Rel.$ 2.73 $\rightarrow$ 3.28 (+20.15%). Our dataset aims to facilitate further research in long-form video-to-audio generation and is available at https://github.com/deepreasonings/long-form-video2audio.
>
---
#### [new 115] AnatoMaskGAN: GNN-Driven Slice Feature Fusion and Noise Augmentation for Medical Semantic Image Synthesis
- **分类: eess.IV; cs.CV; I.4.9**

- **简介: 该论文属于医学图像合成任务，解决GAN生成图像缺乏空间一致性和结构多样性问题。提出AnatoMaskGAN，融合切片特征、引入三维噪声和优化纹理，提升重建精度和感知质量。**

- **链接: [http://arxiv.org/pdf/2508.11375v1](http://arxiv.org/pdf/2508.11375v1)**

> **作者:** Zonglin Wu; Yule Xue; Qianxiang Hu; Yaoyao Feng; Yuqi Ma; Shanxiong Chen
>
> **备注:** 8 pages
>
> **摘要:** Medical semantic-mask synthesis boosts data augmentation and analysis, yet most GAN-based approaches still produce one-to-one images and lack spatial consistency in complex scans. To address this, we propose AnatoMaskGAN, a novel synthesis framework that embeds slice-related spatial features to precisely aggregate inter-slice contextual dependencies, introduces diverse image-augmentation strategies, and optimizes deep feature learning to improve performance on complex medical images. Specifically, we design a GNN-based strongly correlated slice-feature fusion module to model spatial relationships between slices and integrate contextual information from neighboring slices, thereby capturing anatomical details more comprehensively; we introduce a three-dimensional spatial noise-injection strategy that weights and fuses spatial features with noise to enhance modeling of structural diversity; and we incorporate a grayscale-texture classifier to optimize grayscale distribution and texture representation during generation. Extensive experiments on the public L2R-OASIS and L2R-Abdomen CT datasets show that AnatoMaskGAN raises PSNR on L2R-OASIS to 26.50 dB (0.43 dB higher than the current state of the art) and achieves an SSIM of 0.8602 on L2R-Abdomen CT--a 0.48 percentage-point gain over the best model, demonstrating its superiority in reconstruction accuracy and perceptual quality. Ablation studies that successively remove the slice-feature fusion module, spatial 3D noise-injection strategy, and grayscale-texture classifier reveal that each component contributes significantly to PSNR, SSIM, and LPIPS, further confirming the independent value of each core design in enhancing reconstruction accuracy and perceptual quality.
>
---
#### [new 116] GenFlowRL: Shaping Rewards with Generative Object-Centric Flow in Visual Reinforcement Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉强化学习任务，旨在解决生成数据质量依赖和精细操作难题。通过引入生成式对象中心流，提取低维特征以学习泛化性强的策略。**

- **链接: [http://arxiv.org/pdf/2508.11049v1](http://arxiv.org/pdf/2508.11049v1)**

> **作者:** Kelin Yu; Sheng Zhang; Harshit Soora; Furong Huang; Heng Huang; Pratap Tokekar; Ruohan Gao
>
> **备注:** Published at ICCV 2025
>
> **摘要:** Recent advances have shown that video generation models can enhance robot learning by deriving effective robot actions through inverse dynamics. However, these methods heavily depend on the quality of generated data and struggle with fine-grained manipulation due to the lack of environment feedback. While video-based reinforcement learning improves policy robustness, it remains constrained by the uncertainty of video generation and the challenges of collecting large-scale robot datasets for training diffusion models. To address these limitations, we propose GenFlowRL, which derives shaped rewards from generated flow trained from diverse cross-embodiment datasets. This enables learning generalizable and robust policies from diverse demonstrations using low-dimensional, object-centric features. Experiments on 10 manipulation tasks, both in simulation and real-world cross-embodiment evaluations, demonstrate that GenFlowRL effectively leverages manipulation features extracted from generated object-centric flow, consistently achieving superior performance across diverse and challenging scenarios. Our Project Page: https://colinyu1.github.io/genflowrl
>
---
#### [new 117] Efficient Image-to-Image Schrödinger Bridge for CT Field of View Extension
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于CT图像重建任务，旨在解决FOV扩展问题。通过提出I$^2$SB模型，实现快速准确的图像扩展，提升临床应用可行性。**

- **链接: [http://arxiv.org/pdf/2508.11211v1](http://arxiv.org/pdf/2508.11211v1)**

> **作者:** Zhenhao Li; Long Yang; Xiaojie Yin; Haijun Yu; Jiazhou Wang; Hongbin Han; Weigang Hu; Yixing Huang
>
> **备注:** 10 pages
>
> **摘要:** Computed tomography (CT) is a cornerstone imaging modality for non-invasive, high-resolution visualization of internal anatomical structures. However, when the scanned object exceeds the scanner's field of view (FOV), projection data are truncated, resulting in incomplete reconstructions and pronounced artifacts near FOV boundaries. Conventional reconstruction algorithms struggle to recover accurate anatomy from such data, limiting clinical reliability. Deep learning approaches have been explored for FOV extension, with diffusion generative models representing the latest advances in image synthesis. Yet, conventional diffusion models are computationally demanding and slow at inference due to their iterative sampling process. To address these limitations, we propose an efficient CT FOV extension framework based on the image-to-image Schr\"odinger Bridge (I$^2$SB) diffusion model. Unlike traditional diffusion models that synthesize images from pure Gaussian noise, I$^2$SB learns a direct stochastic mapping between paired limited-FOV and extended-FOV images. This direct correspondence yields a more interpretable and traceable generative process, enhancing anatomical consistency and structural fidelity in reconstructions. I$^2$SB achieves superior quantitative performance, with root-mean-square error (RMSE) values of 49.8\,HU on simulated noisy data and 152.0HU on real data, outperforming state-of-the-art diffusion models such as conditional denoising diffusion probabilistic models (cDDPM) and patch-based diffusion methods. Moreover, its one-step inference enables reconstruction in just 0.19s per 2D slice, representing over a 700-fold speedup compared to cDDPM (135s) and surpassing diffusionGAN (0.58s), the second fastest. This combination of accuracy and efficiency makes I$^2$SB highly suitable for real-time or clinical deployment.
>
---
#### [new 118] Deep Learning-Based Automated Segmentation of Uterine Myomas
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决子宫肌瘤自动分割难题。通过使用公开数据集，建立分割基准以提高准确性与可比性。**

- **链接: [http://arxiv.org/pdf/2508.11010v1](http://arxiv.org/pdf/2508.11010v1)**

> **作者:** Tausifa Jan Saleem; Mohammad Yaqub
>
> **摘要:** Uterine fibroids (myomas) are the most common benign tumors of the female reproductive system, particularly among women of childbearing age. With a prevalence exceeding 70%, they pose a significant burden on female reproductive health. Clinical symptoms such as abnormal uterine bleeding, infertility, pelvic pain, and pressure-related discomfort play a crucial role in guiding treatment decisions, which are largely influenced by the size, number, and anatomical location of the fibroids. Magnetic Resonance Imaging (MRI) is a non-invasive and highly accurate imaging modality commonly used by clinicians for the diagnosis of uterine fibroids. Segmenting uterine fibroids requires a precise assessment of both the uterus and fibroids on MRI scans, including measurements of volume, shape, and spatial location. However, this process is labor intensive and time consuming and subjected to variability due to intra- and inter-expert differences at both pre- and post-treatment stages. As a result, there is a critical need for an accurate and automated segmentation method for uterine fibroids. In recent years, deep learning algorithms have shown re-markable improvements in medical image segmentation, outperforming traditional methods. These approaches offer the potential for fully automated segmentation. Several studies have explored the use of deep learning models to achieve automated segmentation of uterine fibroids. However, most of the previous work has been conducted using private datasets, which poses challenges for validation and comparison between studies. In this study, we leverage the publicly available Uterine Myoma MRI Dataset (UMD) to establish a baseline for automated segmentation of uterine fibroids, enabling standardized evaluation and facilitating future research in this domain.
>
---
## 更新

#### [replaced 001] MCA-Bench: A Multimodal Benchmark for Evaluating CAPTCHA Robustness Against VLM-based Attacks
- **分类: cs.CV; I.4.9**

- **链接: [http://arxiv.org/pdf/2506.05982v4](http://arxiv.org/pdf/2506.05982v4)**

> **作者:** Zonglin Wu; Yule Xue; Yaoyao Feng; Xiaolong Wang; Yiren Song
>
> **备注:** we update the paper, add more experiments, and update the teammates
>
> **摘要:** As automated attack techniques rapidly advance, CAPTCHAs remain a critical defense mechanism against malicious bots. However, existing CAPTCHA schemes encompass a diverse range of modalities -- from static distorted text and obfuscated images to interactive clicks, sliding puzzles, and logic-based questions -- yet the community still lacks a unified, large-scale, multimodal benchmark to rigorously evaluate their security robustness. To address this gap, we introduce MCA-Bench, a comprehensive and reproducible benchmarking suite that integrates heterogeneous CAPTCHA types into a single evaluation protocol. Leveraging a shared vision-language model backbone, we fine-tune specialized cracking agents for each CAPTCHA category, enabling consistent, cross-modal assessments. Extensive experiments reveal that MCA-Bench effectively maps the vulnerability spectrum of modern CAPTCHA designs under varied attack settings, and crucially offers the first quantitative analysis of how challenge complexity, interaction depth, and model solvability interrelate. Based on these findings, we propose three actionable design principles and identify key open challenges, laying the groundwork for systematic CAPTCHA hardening, fair benchmarking, and broader community collaboration. Datasets and code are available online.
>
---
#### [replaced 002] Med3DVLM: An Efficient Vision-Language Model for 3D Medical Image Analysis
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2503.20047v2](http://arxiv.org/pdf/2503.20047v2)**

> **作者:** Yu Xin; Gorkem Can Ates; Kuang Gong; Wei Shao
>
> **摘要:** Vision-language models (VLMs) have shown promise in 2D medical image analysis, but extending them to 3D remains challenging due to the high computational demands of volumetric data and the difficulty of aligning 3D spatial features with clinical text. We present Med3DVLM, a 3D VLM designed to address these challenges through three key innovations: (1) DCFormer, an efficient encoder that uses decomposed 3D convolutions to capture fine-grained spatial features at scale; (2) SigLIP, a contrastive learning strategy with pairwise sigmoid loss that improves image-text alignment without relying on large negative batches; and (3) a dual-stream MLP-Mixer projector that fuses low- and high-level image features with text embeddings for richer multi-modal representations. We evaluate our model on the M3D dataset, which includes radiology reports and VQA data for 120,084 3D medical images. Results show that Med3DVLM achieves superior performance across multiple benchmarks. For image-text retrieval, it reaches 61.00% R@1 on 2,000 samples, significantly outperforming the current state-of-the-art M3D model (19.10%). For report generation, it achieves a METEOR score of 36.42% (vs. 14.38%). In open-ended visual question answering (VQA), it scores 36.76% METEOR (vs. 33.58%), and in closed-ended VQA, it achieves 79.95% accuracy (vs. 75.78%). These results highlight Med3DVLM's ability to bridge the gap between 3D imaging and language, enabling scalable, multi-task reasoning across clinical applications. Our code is publicly available at https://github.com/mirthAI/Med3DVLM.
>
---
#### [replaced 003] Image-to-Text for Medical Reports Using Adaptive Co-Attention and Triple-LSTM Module
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18297v3](http://arxiv.org/pdf/2503.18297v3)**

> **作者:** Yishen Liu; Shengda Luo; Zishao Zhong; Hudan Pan
>
> **摘要:** Medical report generation requires specialized expertise that general large models often fail to accurately capture. Moreover, the inherent repetition and similarity in medical data make it difficult for models to extract meaningful features, resulting in a tendency to overfit. So in this paper, we propose a multimodal model, Co-Attention Triple-LSTM Network (CA-TriNet), a deep learning model that combines transformer architectures with a Multi-LSTM network. Its Co-Attention module synergistically links a vision transformer with a text transformer to better differentiate medical images with similarities, augmented by an adaptive weight operator to catch and amplify image labels with minor similarities. Furthermore, its Triple-LSTM module refines generated sentences using targeted image objects. Extensive evaluations over three public datasets have demonstrated that CA-TriNet outperforms state-of-the-art models in terms of comprehensive ability, even pre-trained large language models on some metrics.
>
---
#### [replaced 004] Casual3DHDR: High Dynamic Range 3D Gaussian Splatting from Casually Captured Videos
- **分类: cs.CV; cs.GR; cs.MM**

- **链接: [http://arxiv.org/pdf/2504.17728v2](http://arxiv.org/pdf/2504.17728v2)**

> **作者:** Shucheng Gong; Lingzhe Zhao; Wenpu Li; Hong Xie; Yin Zhang; Shiyu Zhao; Peidong Liu
>
> **备注:** Published in ACM Multimedia 2025. Project page: https://lingzhezhao.github.io/CasualHDRSplat/ (Previously titled "CasualHDRSplat: Robust High Dynamic Range 3D Gaussian Splatting from Casually Captured Videos")
>
> **摘要:** Photo-realistic novel view synthesis from multi-view images, such as neural radiance field (NeRF) and 3D Gaussian Splatting (3DGS), has gained significant attention for its superior performance. However, most existing methods rely on low dynamic range (LDR) images, limiting their ability to capture detailed scenes in high-contrast environments. While some prior works address high dynamic range (HDR) scene reconstruction, they typically require multi-view sharp images with varying exposure times captured at fixed camera positions, which is time-consuming and impractical. To make data acquisition more flexible, we propose \textbf{Casual3DHDR}, a robust one-stage method that reconstructs 3D HDR scenes from casually-captured auto-exposure (AE) videos, even under severe motion blur and unknown, varying exposure times. Our approach integrates a continuous camera trajectory into a unified physical imaging model, jointly optimizing exposure times, camera trajectory, and the camera response function (CRF). Extensive experiments on synthetic and real-world datasets demonstrate that \textbf{Casual3DHDR} outperforms existing methods in robustness and rendering quality. Our source code and dataset will be available at https://lingzhezhao.github.io/CasualHDRSplat/
>
---
#### [replaced 005] PhysLab: A Benchmark Dataset for Multi-Granularity Visual Parsing of Physics Experiments
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.06631v2](http://arxiv.org/pdf/2506.06631v2)**

> **作者:** Minghao Zou; Qingtian Zeng; Yongping Miao; Shangkun Liu; Zilong Wang; Hantao Liu; Wei Zhou
>
> **摘要:** Visual parsing of images and videos is critical for a wide range of real-world applications. However, progress in this field is constrained by limitations of existing datasets: (1) insufficient annotation granularity, which impedes fine-grained scene understanding and high-level reasoning; (2) limited coverage of domains, particularly a lack of datasets tailored for educational scenarios; and (3) lack of explicit procedural guidance, with minimal logical rules and insufficient representation of structured task process. To address these gaps, we introduce PhysLab, the first video dataset that captures students conducting complex physics experiments. The dataset includes four representative experiments that feature diverse scientific instruments and rich human-object interaction (HOI) patterns. PhysLab comprises 620 long-form videos and provides multilevel annotations that support a variety of vision tasks, including action recognition, object detection, HOI analysis, etc. We establish strong baselines and perform extensive evaluations to highlight key challenges in the parsing of procedural educational videos. We expect PhysLab to serve as a valuable resource for advancing fine-grained visual parsing, facilitating intelligent classroom systems, and fostering closer integration between computer vision and educational technologies. The dataset and the evaluation toolkit are publicly available at https://github.com/ZMH-SDUST/PhysLab.
>
---
#### [replaced 006] Part Segmentation of Human Meshes via Multi-View Human Parsing
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2507.18655v3](http://arxiv.org/pdf/2507.18655v3)**

> **作者:** James Dickens; Kamyar Hamad
>
> **摘要:** Recent advances in point cloud deep learning have led to models that achieve high per-part labeling accuracy on large-scale point clouds, using only the raw geometry of unordered point sets. In parallel, the field of human parsing focuses on predicting body part and clothing/accessory labels from images. This work aims to bridge these two domains by enabling per-vertex semantic segmentation of large-scale human meshes. To achieve this, a pseudo-ground truth labeling pipeline is developed for the Thuman2.1 dataset: meshes are first aligned to a canonical pose, segmented from multiple viewpoints, and the resulting point-level labels are then backprojected onto the original mesh to produce per-point pseudo ground truth annotations. Subsequently, a novel, memory-efficient sampling strategy is introduced, a windowed iterative farthest point sampling (FPS) with space-filling curve-based serialization to effectively downsample the point clouds. This is followed by a purely geometric segmentation using PointTransformer, enabling semantic parsing of human meshes without relying on texture information. Experimental results confirm the effectiveness and accuracy of the proposed approach. Project code and pre-processed data is available at https://github.com/JamesMcCullochDickens/Human3DParsing/tree/master.
>
---
#### [replaced 007] A Closer Look at Multimodal Representation Collapse
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.22483v2](http://arxiv.org/pdf/2505.22483v2)**

> **作者:** Abhra Chaudhuri; Anjan Dutta; Tu Bui; Serban Georgescu
>
> **备注:** International Conference on Machine Learning (ICML) 2025 (Spotlight)
>
> **摘要:** We aim to develop a fundamental understanding of modality collapse, a recently observed empirical phenomenon wherein models trained for multimodal fusion tend to rely only on a subset of the modalities, ignoring the rest. We show that modality collapse happens when noisy features from one modality are entangled, via a shared set of neurons in the fusion head, with predictive features from another, effectively masking out positive contributions from the predictive features of the former modality and leading to its collapse. We further prove that cross-modal knowledge distillation implicitly disentangles such representations by freeing up rank bottlenecks in the student encoder, denoising the fusion-head outputs without negatively impacting the predictive features from either modality. Based on the above findings, we propose an algorithm that prevents modality collapse through explicit basis reallocation, with applications in dealing with missing modalities. Extensive experiments on multiple multimodal benchmarks validate our theoretical claims. Project page: https://abhrac.github.io/mmcollapse/.
>
---
#### [replaced 008] SVG-Head: Hybrid Surface-Volumetric Gaussians for High-Fidelity Head Reconstruction and Real-Time Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.09597v2](http://arxiv.org/pdf/2508.09597v2)**

> **作者:** Heyi Sun; Cong Wang; Tian-Xing Xu; Jingwei Huang; Di Kang; Chunchao Guo; Song-Hai Zhang
>
> **备注:** Accepted by ICCV 2025. Project page: https://heyy-sun.github.io/SVG-Head/
>
> **摘要:** Creating high-fidelity and editable head avatars is a pivotal challenge in computer vision and graphics, boosting many AR/VR applications. While recent advancements have achieved photorealistic renderings and plausible animation, head editing, especially real-time appearance editing, remains challenging due to the implicit representation and entangled modeling of the geometry and global appearance. To address this, we propose Surface-Volumetric Gaussian Head Avatar (SVG-Head), a novel hybrid representation that explicitly models the geometry with 3D Gaussians bound on a FLAME mesh and leverages disentangled texture images to capture the global appearance. Technically, it contains two types of Gaussians, in which surface Gaussians explicitly model the appearance of head avatars using learnable texture images, facilitating real-time texture editing, while volumetric Gaussians enhance the reconstruction quality of non-Lambertian regions (e.g., lips and hair). To model the correspondence between 3D world and texture space, we provide a mesh-aware Gaussian UV mapping method, which leverages UV coordinates given by the FLAME mesh to obtain sharp texture images and real-time rendering speed. A hierarchical optimization strategy is further designed to pursue the optimal performance in both reconstruction quality and editing flexibility. Experiments on the NeRSemble dataset show that SVG-Head not only generates high-fidelity rendering results, but also is the first method to obtain explicit texture images for Gaussian head avatars and support real-time appearance editing.
>
---
#### [replaced 009] DSConv: Dynamic Splitting Convolution for Pansharpening
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.06147v2](http://arxiv.org/pdf/2508.06147v2)**

> **作者:** Xuanyu Liu; Bonan An
>
> **备注:** The content of the paper is not yet fully developed, and the proposed approach requires further optimization. Additionally, the experimental results are incomplete and need to be supplemented. Therefore, I request the withdrawal of this submission for further revision and improvements
>
> **摘要:** Aiming to obtain a high-resolution image, pansharpening involves the fusion of a multi-spectral image (MS) and a panchromatic image (PAN), the low-level vision task remaining significant and challenging in contemporary research. Most existing approaches rely predominantly on standard convolutions, few making the effort to adaptive convolutions, which are effective owing to the inter-pixel correlations of remote sensing images. In this paper, we propose a novel strategy for dynamically splitting convolution kernels in conjunction with attention, selecting positions of interest, and splitting the original convolution kernel into multiple smaller kernels, named DSConv. The proposed DSConv more effectively extracts features of different positions within the receptive field, enhancing the network's generalization, optimization, and feature representation capabilities. Furthermore, we innovate and enrich concepts of dynamic splitting convolution and provide a novel network architecture for pansharpening capable of achieving the tasks more efficiently, building upon this methodology. Adequate fair experiments illustrate the effectiveness and the state-of-the-art performance attained by DSConv.Comprehensive and rigorous discussions proved the superiority and optimal usage conditions of DSConv.
>
---
#### [replaced 010] Compositional Zero-shot Learning via Progressive Language-based Observations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2311.14749v2](http://arxiv.org/pdf/2311.14749v2)**

> **作者:** Lin Li; Guikun Chen; Zhen Wang; Jun Xiao; Long Chen
>
> **摘要:** Compositional zero-shot learning aims to recognize unseen state-object compositions by leveraging known primitives (state and object) during training. However, effectively modeling interactions between primitives and generalizing knowledge to novel compositions remains a perennial challenge. There are two key factors: object-conditioned and state-conditioned variance, i.e., the appearance of states (or objects) can vary significantly when combined with different objects (or states). For instance, the state "old" can signify a vintage design for a "car" or an advanced age for a "cat". In this paper, we argue that these variances can be mitigated by predicting composition categories based on pre-observed primitive. To this end, we propose Progressive Language-based Observations (PLO), which can dynamically determine a better observation order of primitives. These observations comprise a series of concepts or languages that allow the model to understand image content in a step-by-step manner. Specifically, PLO adopts pre-trained vision-language models (VLMs) to empower the model with observation capabilities. We further devise two variants: 1) PLO-VLM: a two-step method, where a pre-observing classifier dynamically determines the observation order of two primitives. 2) PLO-LLM: a multi-step scheme, which utilizes large language models (LLMs) to craft composition-specific prompts for step-by-step observing. Extensive ablations on three challenging datasets demonstrate the superiority of PLO compared with state-of-the-art methods, affirming its abilities in compositional recognition.
>
---
#### [replaced 011] Automatic brain tumor segmentation in 2D intra-operative ultrasound images using magnetic resonance imaging tumor annotations
- **分类: eess.IV; cs.CV; cs.LG; I.4.6; J.3**

- **链接: [http://arxiv.org/pdf/2411.14017v3](http://arxiv.org/pdf/2411.14017v3)**

> **作者:** Mathilde Faanes; Ragnhild Holden Helland; Ole Solheim; Sébastien Muller; Ingerid Reinertsen
>
> **备注:** 14 pages, 5 figures
>
> **摘要:** Automatic segmentation of brain tumors in intra-operative ultrasound (iUS) images could facilitate localization of tumor tissue during resection surgery. The lack of large annotated datasets limits the current models performances. In this paper, we investigated the use of tumor annotations in magnetic resonance imaging (MRI) scans, which are more accessible than annotations in iUS images, for training of deep learning models for iUS brain tumor segmentation. We used 180 annotated MRI scans with corresponding unannotated iUS images, and 29 annotated iUS images. Image registration was performed to transfer the MRI annotations to the corresponding iUS images before training the nnU-Net model with different configurations of the data and label origins. The results showed no significant difference in Dice score for a model trained with only MRI annotated tumors compared to models trained with only iUS annotations and both, and to expert annotations, indicating that MRI tumor annotations can be used as a substitute for iUS tumor annotations to train a deep learning model for automatic brain tumor segmentation in iUS images. The best model obtained an average Dice score of $0.62\pm0.31$, compared to $0.67\pm0.25$ for an expert neurosurgeon, where the performance on larger tumors were similar, but lower for the models on smaller tumors. In addition, the results showed that removing smaller tumors from the training sets improved the results. The main models are available here: https://github.com/mathildefaanes/us_brain_tumor_segmentation/tree/main
>
---
#### [replaced 012] Scanpath Prediction in Panoramic Videos via Expected Code Length Minimization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2305.02536v3](http://arxiv.org/pdf/2305.02536v3)**

> **作者:** Mu Li; Kanglong Fan; Kede Ma
>
> **摘要:** Predicting human scanpaths when exploring panoramic videos is a challenging task due to the spherical geometry and the multimodality of the input, and the inherent uncertainty and diversity of the output. Most previous methods fail to give a complete treatment of these characteristics, and thus are prone to errors. In this paper, we present a simple new criterion for scanpath prediction based on principles from lossy data compression. This criterion suggests minimizing the expected code length of quantized scanpaths in a training set, which corresponds to fitting a discrete conditional probability model via maximum likelihood. Specifically, the probability model is conditioned on two modalities: a viewport sequence as the deformation-reduced visual input and a set of relative historical scanpaths projected onto respective viewports as the aligned path input. The probability model is parameterized by a product of discretized Gaussian mixture models to capture the uncertainty and the diversity of scanpaths from different users. Most importantly, the training of the probability model does not rely on the specification of "ground-truth" scanpaths for imitation learning. We also introduce a proportional-integral-derivative (PID) controller-based sampler to generate realistic human-like scanpaths from the learned probability model. Experimental results demonstrate that our method consistently produces better quantitative scanpath results in terms of prediction accuracy (by comparing to the assumed "ground-truths") and perceptual realism (through machine discrimination) over a wide range of prediction horizons. We additionally verify the perceptual realism improvement via a formal psychophysical experiment and the generalization improvement on several unseen panoramic video datasets.
>
---
#### [replaced 013] IMU: Influence-guided Machine Unlearning
- **分类: cs.LG; cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.01620v2](http://arxiv.org/pdf/2508.01620v2)**

> **作者:** Xindi Fan; Jing Wu; Mingyi Zhou; Pengwei Liang; Dinh Phung
>
> **摘要:** Recent studies have shown that deep learning models are vulnerable to attacks and tend to memorize training data points, raising significant concerns about privacy leakage. This motivates the development of machine unlearning (MU), i.e., a paradigm that enables models to selectively forget specific data points upon request. However, most existing MU algorithms require partial or full fine-tuning on the retain set. This necessitates continued access to the original training data, which is often impractical due to privacy concerns and storage constraints. A few retain-data-free MU methods have been proposed, but some rely on access to auxiliary data and precomputed statistics of the retain set, while others scale poorly when forgetting larger portions of data. In this paper, we propose Influence-guided Machine Unlearning (IMU), a simple yet effective method that conducts MU using only the forget set. Specifically, IMU employs gradient ascent and innovatively introduces dynamic allocation of unlearning intensities across different data points based on their influences. This adaptive strategy significantly enhances unlearning effectiveness while maintaining model utility. Results across vision and language tasks demonstrate that IMU consistently outperforms existing retain-data-free MU methods.
>
---
#### [replaced 014] Learning Camera-Agnostic White-Balance Preferences
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.01342v2](http://arxiv.org/pdf/2507.01342v2)**

> **作者:** Luxi Zhao; Mahmoud Afifi; Michael S. Brown
>
> **摘要:** The image signal processor (ISP) pipeline in modern cameras consists of several modules that transform raw sensor data into visually pleasing images in a display color space. Among these, the auto white balance (AWB) module is essential for compensating for scene illumination. However, commercial AWB systems often strive to compute aesthetic white-balance preferences rather than accurate neutral color correction. While learning-based methods have improved AWB accuracy, they typically struggle to generalize across different camera sensors -- an issue for smartphones with multiple cameras. Recent work has explored cross-camera AWB, but most methods remain focused on achieving neutral white balance. In contrast, this paper is the first to address aesthetic consistency by learning a post-illuminant-estimation mapping that transforms neutral illuminant corrections into aesthetically preferred corrections in a camera-agnostic space. Once trained, our mapping can be applied after any neutral AWB module to enable consistent and stylized color rendering across unseen cameras. Our proposed model is lightweight -- containing only $\sim$500 parameters -- and runs in just 0.024 milliseconds on a typical flagship mobile CPU. Evaluated on a dataset of 771 smartphone images from three different cameras, our method achieves state-of-the-art performance while remaining fully compatible with existing cross-camera AWB techniques, introducing minimal computational and memory overhead.
>
---
#### [replaced 015] HepatoGEN: Generating Hepatobiliary Phase MRI with Perceptual and Adversarial Models
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.18405v2](http://arxiv.org/pdf/2504.18405v2)**

> **作者:** Jens Hooge; Gerard Sanroma-Guell; Faidra Stavropoulou; Alexander Ullmann; Gesine Knobloch; Mark Klemens; Carola Schmidt; Sabine Weckbach; Andreas Bolz
>
> **备注:** Author disagreement
>
> **摘要:** Dynamic contrast-enhanced magnetic resonance imaging (DCE-MRI) plays a crucial role in the detection and characterization of focal liver lesions, with the hepatobiliary phase (HBP) providing essential diagnostic information. However, acquiring HBP images requires prolonged scan times, which may compromise patient comfort and scanner throughput. In this study, we propose a deep learning based approach for synthesizing HBP images from earlier contrast phases (precontrast and transitional) and compare three generative models: a perceptual U-Net, a perceptual GAN (pGAN), and a denoising diffusion probabilistic model (DDPM). We curated a multi-site DCE-MRI dataset from diverse clinical settings and introduced a contrast evolution score (CES) to assess training data quality, enhancing model performance. Quantitative evaluation using pixel-wise and perceptual metrics, combined with qualitative assessment through blinded radiologist reviews, showed that pGAN achieved the best quantitative performance but introduced heterogeneous contrast in out-of-distribution cases. In contrast, the U-Net produced consistent liver enhancement with fewer artifacts, while DDPM underperformed due to limited preservation of fine structural details. These findings demonstrate the feasibility of synthetic HBP image generation as a means to reduce scan time without compromising diagnostic utility, highlighting the clinical potential of deep learning for dynamic contrast enhancement in liver MRI. A project demo is available at: https://jhooge.github.io/hepatogen
>
---
#### [replaced 016] MUNBa: Machine Unlearning via Nash Bargaining
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.15537v3](http://arxiv.org/pdf/2411.15537v3)**

> **作者:** Jing Wu; Mehrtash Harandi
>
> **摘要:** Machine Unlearning (MU) aims to selectively erase harmful behaviors from models while retaining the overall utility of the model. As a multi-task learning problem, MU involves balancing objectives related to forgetting specific concepts/data and preserving general performance. A naive integration of these forgetting and preserving objectives can lead to gradient conflicts and dominance, impeding MU algorithms from reaching optimal solutions. To address the gradient conflict and dominance issue, we reformulate MU as a two-player cooperative game, where the two players, namely, the forgetting player and the preservation player, contribute via their gradient proposals to maximize their overall gain and balance their contributions. To this end, inspired by the Nash bargaining theory, we derive a closed-form solution to guide the model toward the Pareto stationary point. Our formulation of MU guarantees an equilibrium solution, where any deviation from the final state would lead to a reduction in the overall objectives for both players, ensuring optimality in each objective. We evaluate our algorithm's effectiveness on a diverse set of tasks across image classification and image generation. Extensive experiments with ResNet, vision-language model CLIP, and text-to-image diffusion models demonstrate that our method outperforms state-of-the-art MU algorithms, achieving a better trade-off between forgetting and preserving. Our results also highlight improvements in forgetting precision, preservation of generalization, and robustness against adversarial attacks.
>
---
#### [replaced 017] FairT2I: Mitigating Social Bias in Text-to-Image Generation via Large Language Model-Assisted Detection and Attribute Rebalancing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.03826v2](http://arxiv.org/pdf/2502.03826v2)**

> **作者:** Jinya Sakurai; Issei Sato
>
> **摘要:** The proliferation of Text-to-Image (T2I) models has revolutionized content creation, providing powerful tools for diverse applications ranging from artistic expression to educational material development and marketing. Despite these technological advancements, significant ethical concerns arise from these models' reliance on large-scale datasets that often contain inherent societal biases. These biases are further amplified when AI-generated content is included in training data, potentially reinforcing and perpetuating stereotypes in the generated outputs. In this paper, we introduce FairT2I, a novel framework that harnesses large language models to detect and mitigate social biases in T2I generation. Our framework comprises two key components: (1) an LLM-based bias detection module that identifies potential social biases in generated images based on text prompts, and (2) an attribute rebalancing module that fine-tunes sensitive attributes within the T2I model to mitigate identified biases. Our extensive experiments across various T2I models and datasets show that FairT2I can significantly reduce bias while maintaining high-quality image generation. We conducted both qualitative user studies and quantitative non-parametric analyses in the generated image feature space, building upon the occupational dataset introduced in the Stable Bias study. Our results show that FairT2I successfully mitigates social biases and enhances the diversity of sensitive attributes in generated images. We further demonstrate, using the P2 dataset, that our framework can detect subtle biases that are challenging for human observers to perceive, extending beyond occupation-related prompts. On the basis of these findings, we introduce a new benchmark dataset for evaluating bias in T2I models.
>
---
#### [replaced 018] GBR: Generative Bundle Refinement for High-fidelity Gaussian Splatting with Enhanced Mesh Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.05908v2](http://arxiv.org/pdf/2412.05908v2)**

> **作者:** Jianing Zhang; Yuchao Zheng; Ziwei Li; Qionghai Dai; Xiaoyun Yuan
>
> **摘要:** Gaussian splatting has gained attention for its efficient representation and rendering of 3D scenes using continuous Gaussian primitives. However, it struggles with sparse-view inputs due to limited geometric and photometric information, causing ambiguities in depth, shape, and texture. we propose GBR: Generative Bundle Refinement, a method for high-fidelity Gaussian splatting and meshing using only 4-6 input views. GBR integrates a neural bundle adjustment module to enhance geometry accuracy and a generative depth refinement module to improve geometry fidelity. More specifically, the neural bundle adjustment module integrates a foundation network to produce initial 3D point maps and point matches from unposed images, followed by bundle adjustment optimization to improve multiview consistency and point cloud accuracy. The generative depth refinement module employs a diffusion-based strategy to enhance geometric details and fidelity while preserving the scale. Finally, for Gaussian splatting optimization, we propose a multimodal loss function incorporating depth and normal consistency, geometric regularization, and pseudo-view supervision, providing robust guidance under sparse-view conditions. Experiments on widely used datasets show that GBR significantly outperforms existing methods under sparse-view inputs. Additionally, GBR demonstrates the ability to reconstruct and render large-scale real-world scenes, such as the Pavilion of Prince Teng and the Great Wall, with remarkable details using only 6 views.
>
---
#### [replaced 019] Towards Physically Realizable Adversarial Attacks in Embodied Vision Navigation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2409.10071v5](http://arxiv.org/pdf/2409.10071v5)**

> **作者:** Meng Chen; Jiawei Tu; Chao Qi; Yonghao Dang; Feng Zhou; Wei Wei; Jianqin Yin
>
> **备注:** 7 pages, 7 figures, Accept by IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** The significant advancements in embodied vision navigation have raised concerns about its susceptibility to adversarial attacks exploiting deep neural networks. Investigating the adversarial robustness of embodied vision navigation is crucial, especially given the threat of 3D physical attacks that could pose risks to human safety. However, existing attack methods for embodied vision navigation often lack physical feasibility due to challenges in transferring digital perturbations into the physical world. Moreover, current physical attacks for object detection struggle to achieve both multi-view effectiveness and visual naturalness in navigation scenarios. To address this, we propose a practical attack method for embodied navigation by attaching adversarial patches to objects, where both opacity and textures are learnable. Specifically, to ensure effectiveness across varying viewpoints, we employ a multi-view optimization strategy based on object-aware sampling, which optimizes the patch's texture based on feedback from the vision-based perception model used in navigation. To make the patch inconspicuous to human observers, we introduce a two-stage opacity optimization mechanism, in which opacity is fine-tuned after texture optimization. Experimental results demonstrate that our adversarial patches decrease the navigation success rate by an average of 22.39%, outperforming previous methods in practicality, effectiveness, and naturalness. Code is available at: https://github.com/chen37058/Physical-Attacks-in-Embodied-Nav
>
---
#### [replaced 020] Towards Generalizable Forgery Detection and Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.21210v2](http://arxiv.org/pdf/2503.21210v2)**

> **作者:** Yueying Gao; Dongliang Chang; Bingyao Yu; Haotian Qin; Muxi Diao; Lei Chen; Kongming Liang; Zhanyu Ma
>
> **摘要:** Accurate and interpretable detection of AI-generated images is essential for mitigating risks associated with AI misuse. However, the substantial domain gap among generative models makes it challenging to develop a generalizable forgery detection model. Moreover, since every pixel in an AI-generated image is synthesized, traditional saliency-based forgery explanation methods are not well suited for this task. To address these challenges, we formulate detection and explanation as a unified Forgery Detection and Reasoning task (FDR-Task), leveraging Multi-Modal Large Language Models (MLLMs) to provide accurate detection through reliable reasoning over forgery attributes. To facilitate this task, we introduce the Multi-Modal Forgery Reasoning dataset (MMFR-Dataset), a large-scale dataset containing 120K images across 10 generative models, with 378K reasoning annotations on forgery attributes, enabling comprehensive evaluation of the FDR-Task. Furthermore, we propose FakeReasoning, a forgery detection and reasoning framework with three key components: 1) a dual-branch visual encoder that integrates CLIP and DINO to capture both high-level semantics and low-level artifacts; 2) a Forgery-Aware Feature Fusion Module that leverages DINO's attention maps and cross-attention mechanisms to guide MLLMs toward forgery-related clues; 3) a Classification Probability Mapper that couples language modeling and forgery detection, enhancing overall performance. Experiments across multiple generative models demonstrate that FakeReasoning not only achieves robust generalization but also outperforms state-of-the-art methods on both detection and reasoning tasks.
>
---
#### [replaced 021] Reverse Convolution and Its Applications to Image Restoration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.09824v2](http://arxiv.org/pdf/2508.09824v2)**

> **作者:** Xuhong Huang; Shiqi Liu; Kai Zhang; Ying Tai; Jian Yang; Hui Zeng; Lei Zhang
>
> **备注:** ICCV 2025; https://github.com/cszn/ConverseNet
>
> **摘要:** Convolution and transposed convolution are fundamental operators widely used in neural networks. However, transposed convolution (a.k.a. deconvolution) does not serve as a true inverse of convolution due to inherent differences in their mathematical formulations. To date, no reverse convolution operator has been established as a standard component in neural architectures. In this paper, we propose a novel depthwise reverse convolution operator as an initial attempt to effectively reverse depthwise convolution by formulating and solving a regularized least-squares optimization problem. We thoroughly investigate its kernel initialization, padding strategies, and other critical aspects to ensure its effective implementation. Building upon this operator, we further construct a reverse convolution block by combining it with layer normalization, 1$\times$1 convolution, and GELU activation, forming a Transformer-like structure. The proposed operator and block can directly replace conventional convolution and transposed convolution layers in existing architectures, leading to the development of ConverseNet. Corresponding to typical image restoration models such as DnCNN, SRResNet and USRNet, we train three variants of ConverseNet for Gaussian denoising, super-resolution and deblurring, respectively. Extensive experiments demonstrate the effectiveness of the proposed reverse convolution operator as a basic building module. We hope this work could pave the way for developing new operators in deep model design and applications.
>
---
#### [replaced 022] PRS-Med: Position Reasoning Segmentation with Vision-Language Model in Medical Imaging
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11872v3](http://arxiv.org/pdf/2505.11872v3)**

> **作者:** Quoc-Huy Trinh; Minh-Van Nguyen; Jung Zeng; Ulas Bagci; Debesh Jha
>
> **摘要:** Recent advancements in prompt-based medical image segmentation have enabled clinicians to identify tumors using simple input like bounding boxes or text prompts. However, existing methods face challenges when doctors need to interact through natural language or when position reasoning is required - understanding spatial relationships between anatomical structures and pathologies. We present PRS-Med, a framework that integrates vision-language models with segmentation capabilities to generate both accurate segmentation masks and corresponding spatial reasoning outputs. Additionally, we introduce the MMRS dataset (Multimodal Medical in Positional Reasoning Segmentation), which provides diverse, spatially-grounded question-answer pairs to address the lack of position reasoning data in medical imaging. PRS-Med demonstrates superior performance across six imaging modalities (CT, MRI, X-ray, ultrasound, endoscopy, RGB), significantly outperforming state-of-the-art methods in both segmentation accuracy and position reasoning. Our approach enables intuitive doctor-system interaction through natural language, facilitating more efficient diagnoses. Our dataset pipeline, model, and codebase will be released to foster further research in spatially-aware multimodal reasoning for medical applications.
>
---
#### [replaced 023] JMA: a General Algorithm to Craft Nearly Optimal Targeted Adversarial Example
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2401.01199v2](http://arxiv.org/pdf/2401.01199v2)**

> **作者:** Benedetta Tondi; Wei Guo; Niccolò Pancino; Mauro Barni
>
> **摘要:** Most of the approaches proposed so far to craft targeted adversarial examples against Deep Learning classifiers are highly suboptimal and typically rely on increasing the likelihood of the target class, thus implicitly focusing on one-hot encoding settings. In this paper, a more general, theoretically sound, targeted attack is proposed, which resorts to the minimization of a Jacobian-induced Mahalanobis distance term, taking into account the effort (in the input space) required to move the latent space representation of the input sample in a given direction. The minimization is solved by exploiting the Wolfe duality theorem, reducing the problem to the solution of a Non-Negative Least Square (NNLS) problem. The proposed algorithm (referred to as JMA) provides an optimal solution to a linearised version of the adversarial example problem originally introduced by Szegedy et al. The results of the experiments confirm the generality of the proposed attack which is proven to be effective under a wide variety of output encoding schemes. Noticeably, JMA is also effective in a multi-label classification scenario, being capable to induce a targeted modification of up to half the labels in complex multi-label classification scenarios, a capability that is out of reach of all the attacks proposed so far. As a further advantage, JMA requires very few iterations, thus resulting more efficient than existing methods.
>
---
#### [replaced 024] Refine-IQA: Multi-Stage Reinforcement Finetuning for Perceptual Image Quality Assessment
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.03763v2](http://arxiv.org/pdf/2508.03763v2)**

> **作者:** Ziheng Jia; Jiaying Qian; Zicheng Zhang; Zijian Chen; Xiongkuo Min
>
> **摘要:** Reinforcement fine-tuning (RFT) is a proliferating paradigm for LMM training. Analogous to high-level reasoning tasks, RFT is similarly applicable to low-level vision domains, including image quality assessment (IQA). Existing RFT-based IQA methods typically use rule-based output rewards to verify the model's rollouts but provide no reward supervision for the "think" process, leaving its correctness and efficacy uncontrolled. Furthermore, these methods typically fine-tune directly on downstream IQA tasks without explicitly enhancing the model's native low-level visual quality perception, which may constrain its performance upper bound. In response to these gaps, we propose the multi-stage RFT IQA framework (Refine-IQA). In Stage-1, we build the Refine-Perception-20K dataset (with 12 main distortions, 20,907 locally-distorted images, and over 55K RFT samples) and design multi-task reward functions to strengthen the model's visual quality perception. In Stage-2, targeting the quality scoring task, we introduce a probability difference reward involved strategy for "think" process supervision. The resulting Refine-IQA Series Models achieve outstanding performance on both perception and scoring tasks-and, notably, our paradigm activates a robust "think" (quality interpreting) capability that also attains exceptional results on the corresponding quality interpreting benchmark.
>
---
#### [replaced 025] RL-MoE: An Image-Based Privacy Preserving Approach In Intelligent Transportation System
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.09186v2](http://arxiv.org/pdf/2508.09186v2)**

> **作者:** Abdolazim Rezaei; Mehdi Sookhak; Mahboobeh Haghparast
>
> **摘要:** The proliferation of AI-powered cameras in Intelligent Transportation Systems (ITS) creates a severe conflict between the need for rich visual data and the right to privacy. Existing privacy-preserving methods, such as blurring or encryption, are often insufficient due to creating an undesirable trade-off where either privacy is compromised against advanced reconstruction attacks or data utility is critically degraded. To resolve this challenge, we propose RL-MoE, a novel framework that transforms sensitive visual data into privacy-preserving textual descriptions, eliminating the need for direct image transmission. RL-MoE uniquely combines a Mixture-of-Experts (MoE) architecture for nuanced, multi-aspect scene decomposition with a Reinforcement Learning (RL) agent that optimizes the generated text for a dual objective of semantic accuracy and privacy preservation. Extensive experiments demonstrate that RL-MoE provides superior privacy protection, reducing the success rate of replay attacks to just 9.4\% on the CFP-FP dataset, while simultaneously generating richer textual content than baseline methods. Our work provides a practical and scalable solution for building trustworthy AI systems in privacy-sensitive domains, paving the way for more secure smart city and autonomous vehicle networks.
>
---
#### [replaced 026] Tapping into the Black Box: Uncovering Aligned Representations in Pretrained Neural Networks
- **分类: cs.LG; cs.CV; cs.NE; I.2.6; I.4.10**

- **链接: [http://arxiv.org/pdf/2507.22832v2](http://arxiv.org/pdf/2507.22832v2)**

> **作者:** Maciej Satkiewicz
>
> **备注:** 11 pages, 3-page appendix, 4 figures, preprint; v2 changes: redacted abstract, slight reformulation of Hypothesis 1, extended motivation, unified notation, minor wording improvements
>
> **摘要:** In ReLU networks, gradients of output units can be seen as their input-level representations, as they correspond to the units' pullbacks through the active subnetwork. However, gradients of deeper models are notoriously misaligned, significantly contributing to their black-box nature. We claim that this is because active subnetworks are inherently noisy due to the ReLU hard-gating. To tackle that noise, we propose soft-gating in the backward pass only. The resulting input-level vector field (called ''excitation pullback'') exhibits remarkable perceptual alignment, revealing high-resolution input- and target-specific features that ''just make sense'', therefore establishing a compelling novel explanation method. Furthermore, we speculate that excitation pullbacks approximate (directionally) the gradients of a simpler model, linear in the network's path space, learned implicitly during optimization and largely determining the network's decision; thus arguing for the faithfulness of the produced explanations and their overall significance.
>
---
#### [replaced 027] An Analytical Theory of Spectral Bias in the Learning Dynamics of Diffusion Models
- **分类: cs.LG; cs.CV; math.ST; stat.ML; stat.TH; 68T07, 60G15; F.2.2; G.1.2; G.3; I.2.6**

- **链接: [http://arxiv.org/pdf/2503.03206v2](http://arxiv.org/pdf/2503.03206v2)**

> **作者:** Binxu Wang; Cengiz Pehlevan
>
> **备注:** 91 pages, 23 figures. Preprint
>
> **摘要:** We develop an analytical framework for understanding how the generated distribution evolves during diffusion model training. Leveraging a Gaussian-equivalence principle, we solve the full-batch gradient-flow dynamics of linear and convolutional denoisers and integrate the resulting probability-flow ODE, yielding analytic expressions for the generated distribution. The theory exposes a universal inverse-variance spectral law: the time for an eigen- or Fourier mode to match its target variance scales as $\tau\propto\lambda^{-1}$, so high-variance (coarse) structure is mastered orders of magnitude sooner than low-variance (fine) detail. Extending the analysis to deep linear networks and circulant full-width convolutions shows that weight sharing merely multiplies learning rates accelerating but not eliminating the bias whereas local convolution introduces a qualitatively different bias. Experiments on Gaussian and natural-image datasets confirm the spectral law persists in deep MLP-based UNet. Convolutional U-Nets, however, display rapid near-simultaneous emergence of many modes, implicating local convolution in reshaping learning dynamics. These results underscore how data covariance governs the order and speed with which diffusion models learn, and they call for deeper investigation of the unique inductive biases introduced by local convolution.
>
---
#### [replaced 028] HateClipSeg: A Segment-Level Annotated Dataset for Fine-Grained Hate Video Detection
- **分类: cs.CV; cs.AI; cs.CV, cs.MM; I.2.10**

- **链接: [http://arxiv.org/pdf/2508.01712v2](http://arxiv.org/pdf/2508.01712v2)**

> **作者:** Han Wang; Zhuoran Wang; Roy Ka-Wei Lee
>
> **备注:** 6 pages, 3 figures
>
> **摘要:** Detecting hate speech in videos remains challenging due to the complexity of multimodal content and the lack of fine-grained annotations in existing datasets. We present HateClipSeg, a large-scale multimodal dataset with both video-level and segment-level annotations, comprising over 11,714 segments labeled as Normal or across five Offensive categories: Hateful, Insulting, Sexual, Violence, Self-Harm, along with explicit target victim labels. Our three-stage annotation process yields high inter-annotator agreement (Krippendorff's alpha = 0.817). We propose three tasks to benchmark performance: (1) Trimmed Hateful Video Classification, (2) Temporal Hateful Video Localization, and (3) Online Hateful Video Classification. Results highlight substantial gaps in current models, emphasizing the need for more sophisticated multimodal and temporally aware approaches. The HateClipSeg dataset are publicly available at https://github.com/Social-AI-Studio/HateClipSeg.git.
>
---
#### [replaced 029] LSVG: Language-Guided Scene Graphs with 2D-Assisted Multi-Modal Encoding for 3D Visual Grounding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.04058v3](http://arxiv.org/pdf/2505.04058v3)**

> **作者:** Feng Xiao; Hongbin Xu; Guocan Zhao; Wenxiong Kang
>
> **摘要:** 3D visual grounding aims to localize the unique target described by natural languages in 3D scenes. The significant gap between 3D and language modalities makes it a notable challenge to distinguish multiple similar objects through the described spatial relationships. Current methods attempt to achieve cross-modal understanding in complex scenes via a target-centered learning mechanism, ignoring the modeling of referred objects. We propose a novel 3D visual grounding framework that constructs language-guided scene graphs with referred object discrimination to improve relational perception. The framework incorporates a dual-branch visual encoder that leverages pre-trained 2D semantics to enhance and supervise the multi-modal 3D encoding. Furthermore, we employ graph attention to promote relationship-oriented information fusion in cross-modal interaction. The learned object representations and scene graph structure enable effective alignment between 3D visual content and textual descriptions. Experimental results on popular benchmarks demonstrate our superior performance compared to state-of-the-art methods, especially in handling the challenges of multiple similar distractors.
>
---
#### [replaced 030] FancyVideo: Towards Dynamic and Consistent Video Generation via Cross-frame Textual Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.08189v4](http://arxiv.org/pdf/2408.08189v4)**

> **作者:** Jiasong Feng; Ao Ma; Jing Wang; Ke Cao; Zhanjie Zhang
>
> **备注:** Accepted by IJCAI 2025
>
> **摘要:** Synthesizing motion-rich and temporally consistent videos remains a challenge in artificial intelligence, especially when dealing with extended durations. Existing text-to-video (T2V) models commonly employ spatial cross-attention for text control, equivalently guiding different frame generations without frame-specific textual guidance. Thus, the model's capacity to comprehend the temporal logic conveyed in prompts and generate videos with coherent motion is restricted. To tackle this limitation, we introduce FancyVideo, an innovative video generator that improves the existing text-control mechanism with the well-designed Cross-frame Textual Guidance Module (CTGM). Specifically, CTGM incorporates the Temporal Information Injector (TII) and Temporal Affinity Refiner (TAR) at the beginning and end of cross-attention, respectively, to achieve frame-specific textual guidance. Firstly, TII injects frame-specific information from latent features into text conditions, thereby obtaining cross-frame textual conditions. Then, TAR refines the correlation matrix between cross-frame textual conditions and latent features along the time dimension. Extensive experiments comprising both quantitative and qualitative evaluations demonstrate the effectiveness of FancyVideo. Our approach achieves state-of-the-art T2V generation results on the EvalCrafter benchmark and facilitates the synthesis of dynamic and consistent videos. Note that the T2V process of FancyVideo essentially involves a text-to-image step followed by T+I2V. This means it also supports the generation of videos from user images, i.e., the image-to-video (I2V) task. A significant number of experiments have shown that its performance is also outstanding.
>
---
#### [replaced 031] Reconstructing Satellites in 3D from Amateur Telescope Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2404.18394v5](http://arxiv.org/pdf/2404.18394v5)**

> **作者:** Zhiming Chang; Boyang Liu; Yifei Xia; Youming Guo; Boxin Shi; He Sun
>
> **摘要:** Monitoring space objects is crucial for space situational awareness, yet reconstructing 3D satellite models from ground-based telescope images is challenging due to atmospheric turbulence, long observation distances, limited viewpoints, and low signal-to-noise ratios. In this paper, we propose a novel computational imaging framework that overcomes these obstacles by integrating a hybrid image pre-processing pipeline with a joint pose estimation and 3D reconstruction module based on controlled Gaussian Splatting (GS) and Branch-and-Bound (BnB) search. We validate our approach on both synthetic satellite datasets and on-sky observations of China's Tiangong Space Station and the International Space Station, achieving robust 3D reconstructions of low-Earth orbit satellites from ground-based data. Quantitative evaluations using SSIM, PSNR, LPIPS, and Chamfer Distance demonstrate that our method outperforms state-of-the-art NeRF-based approaches, and ablation studies confirm the critical role of each component. Our framework enables high-fidelity 3D satellite monitoring from Earth, offering a cost-effective alternative for space situational awareness. Project page: https://ai4scientificimaging.org/ReconstructingSatellites
>
---
#### [replaced 032] From Explainable to Explained AI: Ideas for Falsifying and Quantifying Explanations
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.09205v2](http://arxiv.org/pdf/2508.09205v2)**

> **作者:** Yoni Schirris; Eric Marcus; Jonas Teuwen; Hugo Horlings; Efstratios Gavves
>
> **备注:** 10 pages, 2 figures, 2 tables, submitted at MICCAI IMIMIC workshop
>
> **摘要:** Explaining deep learning models is essential for clinical integration of medical image analysis systems. A good explanation highlights if a model depends on spurious features that undermines generalization and harms a subset of patients or, conversely, may present novel biological insights. Although techniques like GradCAM can identify influential features, they are measurement tools that do not themselves form an explanation. We propose a human-machine-VLM interaction system tailored to explaining classifiers in computational pathology, including multi-instance learning for whole-slide images. Our proof of concept comprises (1) an AI-integrated slide viewer to run sliding-window experiments to test claims of an explanation, and (2) quantification of an explanation's predictiveness using general-purpose vision-language models. The results demonstrate that this allows us to qualitatively test claims of explanations and can quantifiably distinguish competing explanations. This offers a practical path from explainable AI to explained AI in digital pathology and beyond. Code and prompts are available at https://github.com/nki-ai/x2x.
>
---
#### [replaced 033] Zero-Shot Anomaly Detection with Dual-Branch Prompt Selection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.00777v2](http://arxiv.org/pdf/2508.00777v2)**

> **作者:** Zihan Wang; Samira Ebrahimi Kahou; Narges Armanfard
>
> **备注:** Accepted at BMVC 2025
>
> **摘要:** Zero-shot anomaly detection (ZSAD) enables identifying and localizing defects in unseen categories by relying solely on generalizable features rather than requiring any labeled examples of anomalies. However, existing ZSAD methods, whether using fixed or learned prompts, struggle under domain shifts because their training data are derived from limited training domains and fail to generalize to new distributions. In this paper, we introduce PILOT, a framework designed to overcome these challenges through two key innovations: (1) a novel dual-branch prompt learning mechanism that dynamically integrates a pool of learnable prompts with structured semantic attributes, enabling the model to adaptively weight the most relevant anomaly cues for each input image; and (2) a label-free test-time adaptation strategy that updates the learnable prompt parameters using high-confidence pseudo-labels from unlabeled test data. Extensive experiments on 13 industrial and medical benchmarks demonstrate that PILOT achieves state-of-the-art performance in both anomaly detection and localization under domain shift.
>
---
#### [replaced 034] PTQAT: A Hybrid Parameter-Efficient Quantization Algorithm for 3D Perception Tasks
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.10557v2](http://arxiv.org/pdf/2508.10557v2)**

> **作者:** Xinhao Wang; Zhiwei Lin; Zhongyu Xia; Yongtao Wang
>
> **备注:** 8 pages, Accepted by ICCVW 2025
>
> **摘要:** Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) represent two mainstream model quantization approaches. However, PTQ often leads to unacceptable performance degradation in quantized models, while QAT imposes substantial GPU memory requirements and extended training time due to weight fine-tuning. In this paper, we propose PTQAT, a novel general hybrid quantization algorithm for the efficient deployment of 3D perception networks. To address the speed accuracy trade-off between PTQ and QAT, our method selects critical layers for QAT fine-tuning and performs PTQ on the remaining layers. Contrary to intuition, fine-tuning the layers with smaller output discrepancies before and after quantization, rather than those with larger discrepancies, actually leads to greater improvements in the model's quantization accuracy. This means we better compensate for quantization errors during their propagation, rather than addressing them at the point where they occur. The proposed PTQAT achieves similar performance to QAT with more efficiency by freezing nearly 50% of quantifiable layers. Additionally, PTQAT is a universal quantization method that supports various quantization bit widths (4 bits) as well as different model architectures, including CNNs and Transformers. The experimental results on nuScenes across diverse 3D perception tasks, including object detection, semantic segmentation, and occupancy prediction, show that our method consistently outperforms QAT-only baselines. Notably, it achieves 0.2%-0.9% NDS and 0.3%-1.0% mAP gains in object detection, 0.3%-2.0% mIoU gains in semantic segmentation and occupancy prediction while fine-tuning fewer weights.
>
---
#### [replaced 035] SynBrain: Enhancing Visual-to-fMRI Synthesis via Probabilistic Representation Learning
- **分类: cs.LG; cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2508.10298v2](http://arxiv.org/pdf/2508.10298v2)**

> **作者:** Weijian Mai; Jiamin Wu; Yu Zhu; Zhouheng Yao; Dongzhan Zhou; Andrew F. Luo; Qihao Zheng; Wanli Ouyang; Chunfeng Song
>
> **摘要:** Deciphering how visual stimuli are transformed into cortical responses is a fundamental challenge in computational neuroscience. This visual-to-neural mapping is inherently a one-to-many relationship, as identical visual inputs reliably evoke variable hemodynamic responses across trials, contexts, and subjects. However, existing deterministic methods struggle to simultaneously model this biological variability while capturing the underlying functional consistency that encodes stimulus information. To address these limitations, we propose SynBrain, a generative framework that simulates the transformation from visual semantics to neural responses in a probabilistic and biologically interpretable manner. SynBrain introduces two key components: (i) BrainVAE models neural representations as continuous probability distributions via probabilistic learning while maintaining functional consistency through visual semantic constraints; (ii) A Semantic-to-Neural Mapper acts as a semantic transmission pathway, projecting visual semantics into the neural response manifold to facilitate high-fidelity fMRI synthesis. Experimental results demonstrate that SynBrain surpasses state-of-the-art methods in subject-specific visual-to-fMRI encoding performance. Furthermore, SynBrain adapts efficiently to new subjects with few-shot data and synthesizes high-quality fMRI signals that are effective in improving data-limited fMRI-to-image decoding performance. Beyond that, SynBrain reveals functional consistency across trials and subjects, with synthesized signals capturing interpretable patterns shaped by biological neural variability. The code will be made publicly available.
>
---
#### [replaced 036] ShoulderShot: Generating Over-the-Shoulder Dialogue Videos
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.07597v2](http://arxiv.org/pdf/2508.07597v2)**

> **作者:** Yuang Zhang; Junqi Cheng; Haoyu Zhao; Jiaxi Gu; Fangyuan Zou; Zenghui Lu; Peng Shu
>
> **摘要:** Over-the-shoulder dialogue videos are essential in films, short dramas, and advertisements, providing visual variety and enhancing viewers' emotional connection. Despite their importance, such dialogue scenes remain largely underexplored in video generation research. The main challenges include maintaining character consistency across different shots, creating a sense of spatial continuity, and generating long, multi-turn dialogues within limited computational budgets. Here, we present ShoulderShot, a framework that combines dual-shot generation with looping video, enabling extended dialogues while preserving character consistency. Our results demonstrate capabilities that surpass existing methods in terms of shot-reverse-shot layout, spatial continuity, and flexibility in dialogue length, thereby opening up new possibilities for practical dialogue video generation. Videos and comparisons are available at https://shouldershot.github.io.
>
---
#### [replaced 037] ViFusionTST: Deep Fusion of Time-Series Image Representations from Load Signals for Early Bed-Exit Prediction
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.22498v2](http://arxiv.org/pdf/2506.22498v2)**

> **作者:** Hao Liu; Yu Hu; Rakiba Rayhana; Ling Bai; Zheng Liu
>
> **摘要:** Bed-related falls remain a major source of injury in hospitals and long-term care facilities, yet many commercial alarms trigger only after a patient has already left the bed. We show that early bed-exit intent can be predicted using only one low-cost load cell mounted under a bed leg. The resulting load signals are first converted into a compact set of complementary images: an RGB line plot that preserves raw waveforms and three texture maps-recurrence plot, Markov transition field, and Gramian angular field-that expose higher-order dynamics. We introduce ViFusionTST, a dual-stream Swin Transformer that processes the line plot and texture maps in parallel and fuses them through cross-attention to learn data-driven modality weights. To provide a realistic benchmark, we collected six months of continuous data from 95 beds in a long-term-care facility. On this real-world dataset ViFusionTST reaches an accuracy of 0.885 and an F1 score of 0.794, surpassing recent 1D and 2D time-series baselines across F1, recall, accuracy, and AUPRC. The results demonstrate that image-based fusion of load-sensor signals for time series classification is a practical and effective solution for real-time, privacy-preserving fall prevention.
>
---
#### [replaced 038] LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.13420v3](http://arxiv.org/pdf/2501.13420v3)**

> **作者:** Jinghan You; Shanglin Li; Yuanrui Sun; Jiangchuan Wei; Mingyu Guo; Chao Feng; Jiao Ran
>
> **备注:** Accepted at ICCV25 as highlight paper, code released at https://github.com/bytedance/LVFace
>
> **摘要:** Vision Transformers (ViTs) have revolutionized large-scale visual modeling, yet remain underexplored in face recognition (FR) where CNNs still dominate. We identify a critical bottleneck: CNN-inspired training paradigms fail to unlock ViT's potential, leading to suboptimal performance and convergence instability.To address this challenge, we propose LVFace, a ViT-based FR model that integrates Progressive Cluster Optimization (PCO) to achieve superior results. Specifically, PCO sequentially applies negative class sub-sampling (NCS) for robust and fast feature alignment from random initialization, feature expectation penalties for centroid stabilization, performing cluster boundary refinement through full-batch training without NCS constraints. LVFace establishes a new state-of-the-art face recognition baseline, surpassing leading approaches such as UniFace and TopoFR across multiple benchmarks. Extensive experiments demonstrate that LVFace delivers consistent performance gains, while exhibiting scalability to large-scale datasets and compatibility with mainstream VLMs and LLMs. Notably, LVFace secured 1st place in the ICCV 2021 Masked Face Recognition (MFR)-Ongoing Challenge (March 2025), proving its efficacy in real-world scenarios. Project is available at https://github.com/bytedance/LVFace.
>
---
#### [replaced 039] AFR-CLIP: Enhancing Zero-Shot Industrial Anomaly Detection with Stateless-to-Stateful Anomaly Feature Rectification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.12910v2](http://arxiv.org/pdf/2503.12910v2)**

> **作者:** Jingyi Yuan; Chenqiang Gao; Pengyu Jie; Xuan Xia; Shangri Huang; Wanquan Liu
>
> **摘要:** Recently, zero-shot anomaly detection (ZSAD) has emerged as a pivotal paradigm for industrial inspection and medical diagnostics, detecting defects in novel objects without requiring any target-dataset samples during training. Existing CLIP-based ZSAD methods generate anomaly maps by measuring the cosine similarity between visual and textual features. However, CLIP's alignment with object categories instead of their anomalous states limits its effectiveness for anomaly detection. To address this limitation, we propose AFR-CLIP, a CLIP-based anomaly feature rectification framework. AFR-CLIP first performs image-guided textual rectification, embedding the implicit defect information from the image into a stateless prompt that describes the object category without indicating any anomalous state. The enriched textual embeddings are then compared with two pre-defined stateful (normal or abnormal) embeddings, and their text-on-text similarity yields the anomaly map that highlights defective regions. To further enhance perception to multi-scale features and complex anomalies, we introduce self prompting (SP) and multi-patch feature aggregation (MPFA) modules. Extensive experiments are conducted on eleven anomaly detection benchmarks across industrial and medical domains, demonstrating AFR-CLIP's superiority in ZSAD.
>
---
#### [replaced 040] Introducing Unbiased Depth into 2D Gaussian Splatting for High-accuracy Surface Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06587v3](http://arxiv.org/pdf/2503.06587v3)**

> **作者:** Yixin Yang; Yang Zhou; Hui Huang
>
> **备注:** Accepted to the Journal track of Pacific Graphics 2025
>
> **摘要:** Recently, 2D Gaussian Splatting (2DGS) has demonstrated superior geometry reconstruction quality than the popular 3DGS by using 2D surfels to approximate thin surfaces. However, it falls short when dealing with glossy surfaces, resulting in visible holes in these areas. We find that the reflection discontinuity causes the issue. To fit the jump from diffuse to specular reflection at different viewing angles, depth bias is introduced in the optimized Gaussian primitives. To address that, we first replace the depth distortion loss in 2DGS with a novel depth convergence loss, which imposes a strong constraint on depth continuity. Then, we rectify the depth criterion in determining the actual surface, which fully accounts for all the intersecting Gaussians along the ray. Qualitative and quantitative evaluations across various datasets reveal that our method significantly improves reconstruction quality, with more complete and accurate surfaces than 2DGS. Code is available at https://github.com/XiaoXinyyx/Unbiased_Surfel.
>
---
#### [replaced 041] GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.01006v5](http://arxiv.org/pdf/2507.01006v5)**

> **作者:** GLM-V Team; :; Wenyi Hong; Wenmeng Yu; Xiaotao Gu; Guo Wang; Guobing Gan; Haomiao Tang; Jiale Cheng; Ji Qi; Junhui Ji; Lihang Pan; Shuaiqi Duan; Weihan Wang; Yan Wang; Yean Cheng; Zehai He; Zhe Su; Zhen Yang; Ziyang Pan; Aohan Zeng; Baoxu Wang; Bin Chen; Boyan Shi; Changyu Pang; Chenhui Zhang; Da Yin; Fan Yang; Guoqing Chen; Jiazheng Xu; Jiale Zhu; Jiali Chen; Jing Chen; Jinhao Chen; Jinghao Lin; Jinjiang Wang; Junjie Chen; Leqi Lei; Letian Gong; Leyi Pan; Mingdao Liu; Mingde Xu; Mingzhi Zhang; Qinkai Zheng; Sheng Yang; Shi Zhong; Shiyu Huang; Shuyuan Zhao; Siyan Xue; Shangqin Tu; Shengbiao Meng; Tianshu Zhang; Tianwei Luo; Tianxiang Hao; Tianyu Tong; Wenkai Li; Wei Jia; Xiao Liu; Xiaohan Zhang; Xin Lyu; Xinyue Fan; Xuancheng Huang; Yanling Wang; Yadong Xue; Yanfeng Wang; Yanzi Wang; Yifan An; Yifan Du; Yiming Shi; Yiheng Huang; Yilin Niu; Yuan Wang; Yuanchang Yue; Yuchen Li; Yutao Zhang; Yuting Wang; Yu Wang; Yuxuan Zhang; Zhao Xue; Zhenyu Hou; Zhengxiao Du; Zihan Wang; Peng Zhang; Debing Liu; Bin Xu; Juanzi Li; Minlie Huang; Yuxiao Dong; Jie Tang
>
> **摘要:** We present GLM-4.1V-Thinking and GLM-4.5V, a family of vision-language models (VLMs) designed to advance general-purpose multimodal understanding and reasoning. In this report, we share our key findings in the development of the reasoning-centric training framework. We first develop a capable vision foundation model with significant potential through large-scale pre-training, which arguably sets the upper bound for the final performance. We then propose Reinforcement Learning with Curriculum Sampling (RLCS) to unlock the full potential of the model, leading to comprehensive capability enhancement across a diverse range of tasks, including STEM problem solving, video understanding, content recognition, coding, grounding, GUI-based agents, and long document interpretation. In a comprehensive evaluation across 42 public benchmarks, GLM-4.5V achieves state-of-the-art performance on nearly all tasks among open-source models of similar size, and demonstrates competitive or even superior results compared to closed-source models such as Gemini-2.5-Flash on challenging tasks including Coding and GUI Agents. Meanwhile, the smaller GLM-4.1V-9B-Thinking remains highly competitive-achieving superior results to the much larger Qwen2.5-VL-72B on 29 benchmarks. We open-source both GLM-4.1V-9B-Thinking and GLM-4.5V. Code, models and more information are released at https://github.com/zai-org/GLM-V.
>
---
#### [replaced 042] Omni-DPO: A Dual-Perspective Paradigm for Dynamic Preference Learning of LLMs
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.10054v2](http://arxiv.org/pdf/2506.10054v2)**

> **作者:** Shangpin Peng; Weinong Wang; Zhuotao Tian; Senqiao Yang; Xing Wu; Haotian Xu; Chengquan Zhang; Takashi Isobe; Baotian Hu; Min Zhang
>
> **摘要:** Direct Preference Optimization (DPO) has become a cornerstone of reinforcement learning from human feedback (RLHF) due to its simplicity and efficiency. However, existing DPO-based approaches typically treat all preference pairs uniformly, ignoring critical variations in their inherent quality and learning utility, leading to suboptimal data utilization and performance. To address this challenge, we propose Omni-DPO, a dual-perspective optimization framework that jointly accounts for (1) the inherent quality of each preference pair and (2) the model's evolving performance on those pairs. By adaptively weighting samples according to both data quality and the model's learning dynamics during training, Omni-DPO enables more effective training data utilization and achieves better performance. Experimental results on various models and benchmarks demonstrate the superiority and generalization capabilities of Omni-DPO. On textual understanding tasks, Gemma-2-9b-it finetuned with Omni-DPO beats the leading LLM, Claude 3 Opus, by a significant margin of 6.7 points on the Arena-Hard benchmark. On mathematical reasoning tasks, Omni-DPO consistently outperforms the baseline methods across all benchmarks, providing strong empirical evidence for the effectiveness and robustness of our approach. Code and models will be available at https://github.com/pspdada/Omni-DPO.
>
---
#### [replaced 043] Synthetic Data for Robust Stroke Segmentation
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2404.01946v3](http://arxiv.org/pdf/2404.01946v3)**

> **作者:** Liam Chalcroft; Ioannis Pappas; Cathy J. Price; John Ashburner
>
> **备注:** Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) https://melba-journal.org/2025:014
>
> **摘要:** Current deep learning-based approaches to lesion segmentation in neuroimaging often depend on high-resolution images and extensive annotated data, limiting clinical applicability. This paper introduces a novel synthetic data framework tailored for stroke lesion segmentation, expanding the SynthSeg methodology to incorporate lesion-specific augmentations that simulate diverse pathological features. Using a modified nnUNet architecture, our approach trains models with label maps from healthy and stroke datasets, facilitating segmentation across both normal and pathological tissue without reliance on specific sequence-based training. Evaluation across in-domain and out-of-domain (OOD) datasets reveals that our method matches state-of-the-art performance within the training domain and significantly outperforms existing methods on OOD data. By minimizing dependence on large annotated datasets and allowing for cross-sequence applicability, our framework holds potential to improve clinical neuroimaging workflows, particularly in stroke pathology. PyTorch training code and weights are publicly available at https://github.com/liamchalcroft/SynthStroke, along with an SPM toolbox featuring a plug-and-play model at https://github.com/liamchalcroft/SynthStrokeSPM.
>
---
#### [replaced 044] TokLIP: Marry Visual Tokens to CLIP for Multimodal Comprehension and Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.05422v2](http://arxiv.org/pdf/2505.05422v2)**

> **作者:** Haokun Lin; Teng Wang; Yixiao Ge; Yuying Ge; Zhichao Lu; Ying Wei; Qingfu Zhang; Zhenan Sun; Ying Shan
>
> **备注:** Technical Report
>
> **摘要:** Pioneering token-based works such as Chameleon and Emu3 have established a foundation for multimodal unification but face challenges of high training computational overhead and limited comprehension performance due to a lack of high-level semantics. In this paper, we introduce TokLIP, a visual tokenizer that enhances comprehension by semanticizing vector-quantized (VQ) tokens and incorporating CLIP-level semantics while enabling end-to-end multimodal autoregressive training with standard VQ tokens. TokLIP integrates a low-level discrete VQ tokenizer with a ViT-based token encoder to capture high-level continuous semantics. Unlike previous approaches (e.g., VILA-U) that discretize high-level features, TokLIP disentangles training objectives for comprehension and generation, allowing the direct application of advanced VQ tokenizers without the need for tailored quantization operations. Our empirical results demonstrate that TokLIP achieves exceptional data efficiency, empowering visual tokens with high-level semantic understanding while enhancing low-level generative capacity, making it well-suited for autoregressive Transformers in both comprehension and generation tasks. The code and models are available at https://github.com/TencentARC/TokLIP.
>
---
#### [replaced 045] GDSR: Global-Detail Integration through Dual-Branch Network with Wavelet Losses for Remote Sensing Image Super-Resolution
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.01460v4](http://arxiv.org/pdf/2501.01460v4)**

> **作者:** Qiwei Zhu; Kai Li; Guojing Zhang; Xiaoying Wang; Jianqiang Huang; Xilai Li
>
> **备注:** GDSR: Global-Detail Integration through Dual-Branch Network with Wavelet Losses for Remote Sensing Image Super-Resolution
>
> **摘要:** In recent years, deep neural networks, including Convolutional Neural Networks, Transformers, and State Space Models, have achieved significant progress in Remote Sensing Image (RSI) Super-Resolution (SR). However, existing SR methods typically overlook the complementary relationship between global and local dependencies. These methods either focus on capturing local information or prioritize global information, which results in models that are unable to effectively capture both global and local features simultaneously. Moreover, their computational cost becomes prohibitive when applied to large-scale RSIs. To address these challenges, we introduce the novel application of Receptance Weighted Key Value (RWKV) to RSI-SR, which captures long-range dependencies with linear complexity. To simultaneously model global and local features, we propose the Global-Detail dual-branch structure, GDSR, which performs SR by paralleling RWKV and convolutional operations to handle large-scale RSIs. Furthermore, we introduce the Global-Detail Reconstruction Module (GDRM) as an intermediary between the two branches to bridge their complementary roles. In addition, we propose the Dual-Group Multi-Scale Wavelet Loss, a wavelet-domain constraint mechanism via dual-group subband strategy and cross-resolution frequency alignment for enhanced reconstruction fidelity in RSI-SR. Extensive experiments under two degradation methods on several benchmarks, including AID, UCMerced, and RSSRD-QH, demonstrate that GSDR outperforms the state-of-the-art Transformer-based method HAT by an average of 0.09 dB in PSNR, while using only 63% of its parameters and 51% of its FLOPs, achieving an inference speed 3.2 times faster.
>
---
#### [replaced 046] Learning an Adaptive and View-Invariant Vision Transformer for Real-Time UAV Tracking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.20002v3](http://arxiv.org/pdf/2412.20002v3)**

> **作者:** You Wu; Yongxin Li; Mengyuan Liu; Xucheng Wang; Xiangyang Yang; Hengzhou Ye; Dan Zeng; Qijun Zhao; Shuiwang Li
>
> **摘要:** Transformer-based models have improved visual tracking, but most still cannot run in real time on resource-limited devices, especially for unmanned aerial vehicle (UAV) tracking. To achieve a better balance between performance and efficiency, we propose AVTrack, an adaptive computation tracking framework that adaptively activates transformer blocks through an Activation Module (AM), which dynamically optimizes the ViT architecture by selectively engaging relevant components. To address extreme viewpoint variations, we propose to learn view-invariant representations via mutual information (MI) maximization. In addition, we propose AVTrack-MD, an enhanced tracker incorporating a novel MI maximization-based multi-teacher knowledge distillation framework. Leveraging multiple off-the-shelf AVTrack models as teachers, we maximize the MI between their aggregated softened features and the corresponding softened feature of the student model, improving the generalization and performance of the student, especially under noisy conditions. Extensive experiments show that AVTrack-MD achieves performance comparable to AVTrack's performance while reducing model complexity and boosting average tracking speed by over 17\%. Codes is available at: https://github.com/wuyou3474/AVTrack.
>
---
#### [replaced 047] Effective Message Hiding with Order-Preserving Mechanisms
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2402.19160v5](http://arxiv.org/pdf/2402.19160v5)**

> **作者:** Gao Yu; Qiu Xuchong; Ye Zihan
>
> **备注:** BMVC 2024
>
> **摘要:** Message hiding, a technique that conceals secret message bits within a cover image, aims to achieve an optimal balance among message capacity, recovery accuracy, and imperceptibility. While convolutional neural networks have notably improved message capacity and imperceptibility, achieving high recovery accuracy remains challenging. This challenge arises because convolutional operations struggle to preserve the sequential order of message bits and effectively address the discrepancy between these two modalities. To address this, we propose StegaFormer, an innovative MLP-based framework designed to preserve bit order and enable global fusion between modalities. Specifically, StegaFormer incorporates three crucial components: Order-Preserving Message Encoder (OPME), Decoder (OPMD) and Global Message-Image Fusion (GMIF). OPME and OPMD aim to preserve the order of message bits by segmenting the entire sequence into equal-length segments and incorporating sequential information during encoding and decoding. Meanwhile, GMIF employs a cross-modality fusion mechanism to effectively fuse the features from the two uncorrelated modalities. Experimental results on the COCO and DIV2K datasets demonstrate that StegaFormer surpasses existing state-of-the-art methods in terms of recovery accuracy, message capacity, and imperceptibility. We will make our code publicly available.
>
---
#### [replaced 048] UI-Venus Technical Report: Building High-performance UI Agents with RFT
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.10833v2](http://arxiv.org/pdf/2508.10833v2)**

> **作者:** Zhangxuan Gu; Zhengwen Zeng; Zhenyu Xu; Xingran Zhou; Shuheng Shen; Yunfei Liu; Beitong Zhou; Changhua Meng; Tianyu Xia; Weizhi Chen; Yue Wen; Jingya Dou; Fei Tang; Jinzhen Lin; Yulin Liu; Zhenlin Guo; Yichen Gong; Heng Jia; Changlong Gao; Yuan Guo; Yong Deng; Zhenyu Guo; Liang Chen; Weiqiang Wang
>
> **摘要:** We present UI-Venus, a native UI agent that takes only screenshots as input based on a multimodal large language model. UI-Venus achieves SOTA performance on both UI grounding and navigation tasks using only several hundred thousand high-quality training samples through reinforcement finetune (RFT) based on Qwen2.5-VL. Specifically, the 7B and 72B variants of UI-Venus obtain 94.1% / 50.8% and 95.3% / 61.9% on the standard grounding benchmarks, i.e., Screenspot-V2 / Pro, surpassing the previous SOTA baselines including open-source GTA1 and closed-source UI-TARS-1.5. To show UI-Venus's summary and planing ability, we also evaluate it on the AndroidWorld, an online UI navigation arena, on which our 7B and 72B variants achieve 49.1% and 65.9% success rate, also beating existing models. To achieve this, we introduce carefully designed reward functions for both UI grounding and navigation tasks and corresponding efficient data cleaning strategies. To further boost navigation performance, we propose Self-Evolving Trajectory History Alignment & Sparse Action Enhancement that refine historical reasoning traces and balances the distribution of sparse but critical actions, leading to more coherent planning and better generalization in complex UI tasks. Our contributions include the publish of SOTA open-source UI agents, comprehensive data cleaning protocols and a novel self-evolving framework for improving navigation performance, which encourage further research and development in the community. Code is available at https://github.com/inclusionAI/UI-Venus.
>
---
#### [replaced 049] Efficient High-Resolution Visual Representation Learning with State Space Model for Human Pose Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.03174v2](http://arxiv.org/pdf/2410.03174v2)**

> **作者:** Hao Zhang; Yongqiang Ma; Wenqi Shao; Ping Luo; Nanning Zheng; Kaipeng Zhang
>
> **摘要:** Capturing long-range dependencies while preserving high-resolution visual representations is crucial for dense prediction tasks such as human pose estimation. Vision Transformers (ViTs) have advanced global modeling through self-attention but suffer from quadratic computational complexity with respect to token count, limiting their efficiency and scalability to high-resolution inputs, especially on mobile and resource-constrained devices. State Space Models (SSMs), exemplified by Mamba, offer an efficient alternative by combining global receptive fields with linear computational complexity, enabling scalable and resource-friendly sequence modeling. However, when applied to dense prediction tasks, existing visual SSMs face key limitations: weak spatial inductive bias, long-range forgetting from hidden state decay, and low-resolution outputs that hinder fine-grained localization. To address these issues, we propose the Dynamic Visual State Space (DVSS) block, which augments visual state space models with multi-scale convolutional operations to enhance local spatial representations and strengthen spatial inductive biases. Through architectural exploration and theoretical analysis, we incorporate deformable operation into the DVSS block, identifying it as an efficient and effective mechanism to enhance semantic aggregation and mitigate long-range forgetting via input-dependent, adaptive spatial sampling. We embed DVSS into a multi-branch high-resolution architecture to build HRVMamba, a novel model for efficient high-resolution representation learning. Extensive experiments on human pose estimation, image classification, and semantic segmentation show that HRVMamba performs competitively against leading CNN-, ViT-, and SSM-based baselines. Code is available at https://github.com/zhanghao5201/PoseVMamba.
>
---
#### [replaced 050] Seeing and Seeing Through the Glass: Real and Synthetic Data for Multi-Layer Depth Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11633v2](http://arxiv.org/pdf/2503.11633v2)**

> **作者:** Hongyu Wen; Yiming Zuo; Venkat Subramanian; Patrick Chen; Jia Deng
>
> **摘要:** Transparent objects are common in daily life, and understanding their multi-layer depth information -- perceiving both the transparent surface and the objects behind it -- is crucial for real-world applications that interact with transparent materials. In this paper, we introduce LayeredDepth, the first dataset with multi-layer depth annotations, including a real-world benchmark and a synthetic data generator, to support the task of multi-layer depth estimation. Our real-world benchmark consists of 1,500 images from diverse scenes, and evaluating state-of-the-art depth estimation methods on it reveals that they struggle with transparent objects. The synthetic data generator is fully procedural and capable of providing training data for this task with an unlimited variety of objects and scene compositions. Using this generator, we create a synthetic dataset with 15,300 images. Baseline models training solely on this synthetic dataset produce good cross-domain multi-layer depth estimation. Fine-tuning state-of-the-art single-layer depth models on it substantially improves their performance on transparent objects, with quadruplet accuracy on our benchmark increased from 55.14% to 75.20%. All images and validation annotations are available under CC0 at https://layereddepth.cs.princeton.edu.
>
---
#### [replaced 051] Preacher: Paper-to-Video Agentic System
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.09632v3](http://arxiv.org/pdf/2508.09632v3)**

> **作者:** Jingwei Liu; Ling Yang; Hao Luo; Fan Wang; Hongyan Li; Mengdi Wang
>
> **备注:** Code not ready
>
> **摘要:** The paper-to-video task converts a research paper into a structured video abstract, distilling key concepts, methods, and conclusions into an accessible, well-organized format. While state-of-the-art video generation models demonstrate potential, they are constrained by limited context windows, rigid video duration constraints, limited stylistic diversity, and an inability to represent domain-specific knowledge. To address these limitations, we introduce Preacher, the first paper-to-video agentic system. Preacher employs a topdown approach to decompose, summarize, and reformulate the paper, followed by bottom-up video generation, synthesizing diverse video segments into a coherent abstract. To align cross-modal representations, we define key scenes and introduce a Progressive Chain of Thought (P-CoT) for granular, iterative planning. Preacher successfully generates high-quality video abstracts across five research fields, demonstrating expertise beyond current video generation models. Code will be released at: https://github.com/GenVerse/Paper2Video
>
---
#### [replaced 052] Pathology-Guided AI System for Accurate Segmentation and Diagnosis of Cervical Spondylosis
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06114v2](http://arxiv.org/pdf/2503.06114v2)**

> **作者:** Qi Zhang; Xiuyuan Chen; Ziyi He; Lianming Wu; Kun Wang; Jianqi Sun; Hongxing Shen
>
> **摘要:** Cervical spondylosis, a complex and prevalent condition, demands precise and efficient diagnostic techniques for accurate assessment. While MRI offers detailed visualization of cervical spine anatomy, manual interpretation remains labor-intensive and prone to error. To address this, we developed an innovative AI-assisted Expert-based Diagnosis System that automates both segmentation and diagnosis of cervical spondylosis using MRI. Leveraging multi-center datasets of cervical MRI images from patients with cervical spondylosis, our system features a pathology-guided segmentation model capable of accurately segmenting key cervical anatomical structures. The segmentation is followed by an expert-based diagnostic framework that automates the calculation of critical clinical indicators. Our segmentation model achieved an impressive average Dice coefficient exceeding 0.90 across four cervical spinal anatomies and demonstrated enhanced accuracy in herniation areas. Diagnostic evaluation further showcased the system's precision, with the lowest mean average errors (MAE) for the C2-C7 Cobb angle and the Maximum Spinal Cord Compression (MSCC) coefficient. In addition, our method delivered high accuracy, precision, recall, and F1 scores in herniation localization, K-line status assessment, T2 hyperintensity detection, and Kang grading. Comparative analysis and external validation demonstrate that our system outperforms existing methods, establishing a new benchmark for segmentation and diagnostic tasks for cervical spondylosis.
>
---
#### [replaced 053] Marmot: Object-Level Self-Correction via Multi-Agent Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.20054v3](http://arxiv.org/pdf/2504.20054v3)**

> **作者:** Jiayang Sun; Hongbo Wang; Jie Cao; Huaibo Huang; Ran He
>
> **摘要:** While diffusion models excel at generating high-quality images, they often struggle with accurate counting, attributes, and spatial relationships in complex multi-object scenes. One potential solution involves employing Multimodal Large Language Model (MLLM) as an AI agent to construct a self-correction framework. However, these approaches heavily rely on the capabilities of the MLLMs used, often fail to account for all objects within the image, and suffer from cumulative distortions during multi-round editing processes. To address these challenges, we propose Marmot, a novel and generalizable framework that leverages Multi-Agent Reasoning for Multi-Object Self-Correcting to enhance image-text alignment. First, we employ a large language model as an Object-Aware Agent to perform object-level divide-and-conquer, automatically decomposing self-correction tasks into object-centric subtasks based on image descriptions. For each subtask, we construct an Object Correction System featuring a decision-execution-verification mechanism that operates exclusively on a single object's segmentation mask or the bounding boxes of object pairs, effectively mitigating inter-object interference and enhancing editing reliability. To efficiently integrate correction results from subtasks while avoiding cumulative distortions from multi-stage editing, we propose a Pixel-Domain Stitching Smoother, which employs mask-guided two-stage latent space optimization. This innovation enables parallel processing of subtasks, significantly improving runtime efficiency while preventing distortion accumulation. Extensive experiments demonstrate that Marmot significantly improves accuracy in object counting, attribute assignment, and spatial relationships for image generation tasks.
>
---
#### [replaced 054] IRL-VLA: Training an Vision-Language-Action Policy via Reward World Model
- **分类: cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.06571v3](http://arxiv.org/pdf/2508.06571v3)**

> **作者:** Anqing Jiang; Yu Gao; Yiru Wang; Zhigang Sun; Shuo Wang; Yuwen Heng; Hao Sun; Shichen Tang; Lijuan Zhu; Jinhao Chai; Jijun Wang; Zichong Gu; Hao Jiang; Li Sun
>
> **备注:** 9 pagres, 2 figures
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated potential in autonomous driving. However, two critical challenges hinder their development: (1) Existing VLA architectures are typically based on imitation learning in open-loop setup which tends to capture the recorded behaviors in the dataset, leading to suboptimal and constrained performance, (2) Close-loop training relies heavily on high-fidelity sensor simulation, where domain gaps and computational inefficiencies pose significant barriers. In this paper, we introduce IRL-VLA, a novel close-loop Reinforcement Learning via \textbf{I}nverse \textbf{R}einforcement \textbf{L}earning reward world model with a self-built VLA approach. Our framework proceeds in a three-stage paradigm: In the first stage, we propose a VLA architecture and pretrain the VLA policy via imitation learning. In the second stage, we construct a lightweight reward world model via inverse reinforcement learning to enable efficient close-loop reward computation. To further enhance planning performance, finally, we design specialized reward world model guidence reinforcement learning via PPO(Proximal Policy Optimization) to effectively balance the safety incidents, comfortable driving, and traffic efficiency. Our approach achieves state-of-the-art performance in NAVSIM v2 end-to-end driving benchmark, 1st runner up in CVPR2025 Autonomous Grand Challenge. We hope that our framework will accelerate VLA research in close-loop autonomous driving.
>
---
#### [replaced 055] SORT3D: Spatial Object-centric Reasoning Toolbox for Zero-Shot 3D Grounding Using Large Language Models
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.18684v2](http://arxiv.org/pdf/2504.18684v2)**

> **作者:** Nader Zantout; Haochen Zhang; Pujith Kachana; Jinkai Qiu; Guofei Chen; Ji Zhang; Wenshan Wang
>
> **备注:** 8 pages, 6 figures, published in IROS 2025
>
> **摘要:** Interpreting object-referential language and grounding objects in 3D with spatial relations and attributes is essential for robots operating alongside humans. However, this task is often challenging due to the diversity of scenes, large number of fine-grained objects, and complex free-form nature of language references. Furthermore, in the 3D domain, obtaining large amounts of natural language training data is difficult. Thus, it is important for methods to learn from little data and zero-shot generalize to new environments. To address these challenges, we propose SORT3D, an approach that utilizes rich object attributes from 2D data and merges a heuristics-based spatial reasoning toolbox with the ability of large language models (LLMs) to perform sequential reasoning. Importantly, our method does not require text-to-3D data for training and can be applied zero-shot to unseen environments. We show that SORT3D achieves state-of-the-art zero-shot performance on complex view-dependent grounding tasks on two benchmarks. We also implement the pipeline to run real-time on two autonomous vehicles and demonstrate that our approach can be used for object-goal navigation on previously unseen real-world environments. All source code for the system pipeline is publicly released at https://github.com/nzantout/SORT3D.
>
---
#### [replaced 056] Towards Consumer-Grade Cybersickness Prediction: Multi-Model Alignment for Real-Time Vision-Only Inference
- **分类: cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2501.01212v2](http://arxiv.org/pdf/2501.01212v2)**

> **作者:** Yitong Zhu; Zhuowen Liang; Yiming Wu; Tangyao Li; Yuyang Wang
>
> **摘要:** Cybersickness remains a major obstacle to the widespread adoption of immersive virtual reality (VR), particularly in consumer-grade environments. While prior methods rely on invasive signals such as electroencephalography (EEG) for high predictive accuracy, these approaches require specialized hardware and are impractical for real-world applications. In this work, we propose a scalable, deployable framework for personalized cybersickness prediction leveraging only non-invasive signals readily available from commercial VR headsets, including head motion, eye tracking, and physiological responses. Our model employs a modality-specific graph neural network enhanced with a Difference Attention Module to extract temporal-spatial embeddings capturing dynamic changes across modalities. A cross-modal alignment module jointly trains the video encoder to learn personalized traits by aligning video features with sensor-derived representations. Consequently, the model accurately predicts individual cybersickness using only video input during inference. Experimental results show our model achieves 88.4\% accuracy, closely matching EEG-based approaches (89.16\%), while reducing deployment complexity. With an average inference latency of 90ms, our framework supports real-time applications, ideal for integration into consumer-grade VR platforms without compromising personalization or performance. The code will be relesed at https://github.com/U235-Aurora/PTGNN.
>
---
#### [replaced 057] Physics-Guided Image Dehazing Diffusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.21385v2](http://arxiv.org/pdf/2504.21385v2)**

> **作者:** Shijun Zhou; Baojie Fan; Jiandong Tian
>
> **摘要:** Due to the domain gap between real-world and synthetic hazy images, current data-driven dehazing algorithms trained on synthetic datasets perform well on synthetic data but struggle to generalize to real-world scenarios. To address this challenge, we propose \textbf{I}mage \textbf{D}ehazing \textbf{D}iffusion \textbf{M}odels (IDDM), a novel diffusion process that incorporates the atmospheric scattering model into noise diffusion. IDDM aims to use the gradual haze formation process to help the denoising Unet robustly learn the distribution of clear images from the conditional input hazy images. We design a specialized training strategy centered around IDDM. Diffusion models are leveraged to bridge the domain gap from synthetic to real-world, while the atmospheric scattering model provides physical guidance for haze formation. During the forward process, IDDM simultaneously introduces haze and noise into clear images, and then robustly separates them during the sampling process. By training with physics-guided information, IDDM shows the ability of domain generalization, and effectively restores the real-world hazy images despite being trained on synthetic datasets. Extensive experiments demonstrate the effectiveness of our method through both quantitative and qualitative comparisons with state-of-the-art approaches.
>
---
#### [replaced 058] Wild2Avatar: Rendering Humans Behind Occlusions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2401.00431v2](http://arxiv.org/pdf/2401.00431v2)**

> **作者:** Tiange Xiang; Adam Sun; Scott Delp; Kazuki Kozuka; Li Fei-Fei; Ehsan Adeli
>
> **备注:** IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). Webpage: https://cs.stanford.edu/~xtiange/projects/wild2avatar/
>
> **摘要:** Rendering the visual appearance of moving humans from occluded monocular videos is a challenging task. Most existing research renders 3D humans under ideal conditions, requiring a clear and unobstructed scene. Those methods cannot be used to render humans in real-world scenes where obstacles may block the camera's view and lead to partial occlusions. In this work, we present Wild2Avatar, a neural rendering approach catered for occluded in-the-wild monocular videos. We propose occlusion-aware scene parameterization for decoupling the scene into three parts - occlusion, human, and background. Additionally, extensive objective functions are designed to help enforce the decoupling of the human from both the occlusion and the background and to ensure the completeness of the human model. We verify the effectiveness of our approach with experiments on in-the-wild videos.
>
---
#### [replaced 059] Lightweight Attribute Localizing Models for Pedestrian Attribute Recognition
- **分类: cs.CV; cs.NA; math.NA**

- **链接: [http://arxiv.org/pdf/2306.09822v2](http://arxiv.org/pdf/2306.09822v2)**

> **作者:** Ashish Jha; Dimitrii Ermilov; Konstantin Sobolev; Anh Huy Phan; Salman Ahmadi-Asl; Naveed Ahmed; Imran Junejo; Zaher AL Aghbari; Thar Baker; Ahmed Mohamed Khedr; Andrzej Cichocki
>
> **摘要:** Pedestrian Attribute Recognition (PAR) focuses on identifying various attributes in pedestrian images, with key applications in person retrieval, suspect re-identification, and soft biometrics. However, Deep Neural Networks (DNNs) for PAR often suffer from over-parameterization and high computational complexity, making them unsuitable for resource-constrained devices. Traditional tensor-based compression methods typically factorize layers without adequately preserving the gradient direction during compression, leading to inefficient compression and a significant accuracy loss. In this work, we propose a novel approach for determining the optimal ranks of low-rank layers, ensuring that the gradient direction of the compressed model closely aligns with that of the original model. This means that the compressed model effectively preserves the update direction of the full model, enabling more efficient compression for PAR tasks. The proposed procedure optimizes the compression ranks for each layer within the ALM model, followed by compression using CPD-EPC or truncated SVD. This results in a reduction in model complexity while maintaining high performance.
>
---
#### [replaced 060] HealthiVert-GAN: A Novel Framework of Pseudo-Healthy Vertebral Image Synthesis for Interpretable Compression Fracture Grading
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.05990v2](http://arxiv.org/pdf/2503.05990v2)**

> **作者:** Qi Zhang; Cheng Chuang; Shunan Zhang; Ziqi Zhao; Kun Wang; Jun Xu; Jianqi Sun
>
> **摘要:** Osteoporotic vertebral compression fractures (OVCFs) are prevalent in the elderly population, typically assessed on computed tomography (CT) scans by evaluating vertebral height loss. This assessment helps determine the fracture's impact on spinal stability and the need for surgical intervention. However, the absence of pre-fracture CT scans and standardized vertebral references leads to measurement errors and inter-observer variability, while irregular compression patterns further challenge the precise grading of fracture severity. While deep learning methods have shown promise in aiding OVCFs screening, they often lack interpretability and sufficient sensitivity, limiting their clinical applicability. To address these challenges, we introduce a novel vertebra synthesis-height loss quantification-OVCFs grading framework. Our proposed model, HealthiVert-GAN, utilizes a coarse-to-fine synthesis network designed to generate pseudo-healthy vertebral images that simulate the pre-fracture state of fractured vertebrae. This model integrates three auxiliary modules that leverage the morphology and height information of adjacent healthy vertebrae to ensure anatomical consistency. Additionally, we introduce the Relative Height Loss of Vertebrae (RHLV) as a quantification metric, which divides each vertebra into three sections to measure height loss between pre-fracture and post-fracture states, followed by fracture severity classification using a Support Vector Machine (SVM). Our approach achieves state-of-the-art classification performance on both the Verse2019 dataset and in-house dataset, and it provides cross-sectional distribution maps of vertebral height loss. This practical tool enhances diagnostic accuracy in clinical settings and assisting in surgical decision-making.
>
---
#### [replaced 061] Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.09736v2](http://arxiv.org/pdf/2508.09736v2)**

> **作者:** Lin Long; Yichen He; Wentao Ye; Yiyuan Pan; Yuan Lin; Hang Li; Junbo Zhao; Wei Li
>
> **摘要:** We introduce M3-Agent, a novel multimodal agent framework equipped with long-term memory. Like humans, M3-Agent can process real-time visual and auditory inputs to build and update its long-term memory. Beyond episodic memory, it also develops semantic memory, enabling it to accumulate world knowledge over time. Its memory is organized in an entity-centric, multimodal format, allowing deeper and more consistent understanding of the environment. Given an instruction, M3-Agent autonomously performs multi-turn, iterative reasoning and retrieves relevant information from memory to accomplish the task. To evaluate memory effectiveness and memory-based reasoning in multimodal agents, we develop M3-Bench, a new long-video question answering benchmark. M3-Bench comprises 100 newly recorded real-world videos captured from a robot's perspective (M3-Bench-robot) and 920 web-sourced videos across diverse scenarios (M3-Bench-web). We annotate question-answer pairs designed to test key capabilities essential for agent applications, such as human understanding, general knowledge extraction, and cross-modal reasoning. Experimental results show that M3-Agent, trained via reinforcement learning, outperforms the strongest baseline, a prompting agent using Gemini-1.5-pro and GPT-4o, achieving 6.7%, 7.7%, and 5.3% higher accuracy on M3-Bench-robot, M3-Bench-web and VideoMME-long, respectively. Our work advances the multimodal agents toward more human-like long-term memory and provides insights into their practical design. Model, code and data are available at https://github.com/bytedance-seed/m3-agent
>
---
#### [replaced 062] ImpliHateVid: A Benchmark Dataset and Two-stage Contrastive Learning Framework for Implicit Hate Speech Detection in Videos
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.06570v2](http://arxiv.org/pdf/2508.06570v2)**

> **作者:** Mohammad Zia Ur Rehman; Anukriti Bhatnagar; Omkar Kabde; Shubhi Bansal; Nagendra Kumar
>
> **备注:** Published in ACL 2025
>
> **摘要:** The existing research has primarily focused on text and image-based hate speech detection, video-based approaches remain underexplored. In this work, we introduce a novel dataset, ImpliHateVid, specifically curated for implicit hate speech detection in videos. ImpliHateVid consists of 2,009 videos comprising 509 implicit hate videos, 500 explicit hate videos, and 1,000 non-hate videos, making it one of the first large-scale video datasets dedicated to implicit hate detection. We also propose a novel two-stage contrastive learning framework for hate speech detection in videos. In the first stage, we train modality-specific encoders for audio, text, and image using contrastive loss by concatenating features from the three encoders. In the second stage, we train cross-encoders using contrastive learning to refine multimodal representations. Additionally, we incorporate sentiment, emotion, and caption-based features to enhance implicit hate detection. We evaluate our method on two datasets, ImpliHateVid for implicit hate speech detection and another dataset for general hate speech detection in videos, HateMM dataset, demonstrating the effectiveness of the proposed multimodal contrastive learning for hateful content detection in videos and the significance of our dataset.
>
---
#### [replaced 063] Visual-RAG: Benchmarking Text-to-Image Retrieval Augmented Generation for Visual Knowledge Intensive Queries
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.16636v2](http://arxiv.org/pdf/2502.16636v2)**

> **作者:** Yin Wu; Quanyu Long; Jing Li; Jianfei Yu; Wenya Wang
>
> **备注:** 21 pages, 6 figures, 17 tables
>
> **摘要:** Retrieval-augmented generation (RAG) is a paradigm that augments large language models (LLMs) with external knowledge to tackle knowledge-intensive question answering. While several benchmarks evaluate Multimodal LLMs (MLLMs) under Multimodal RAG settings, they predominantly retrieve from textual corpora and do not explicitly assess how models exploit visual evidence during generation. Consequently, there still lacks benchmark that isolates and measures the contribution of retrieved images in RAG. We introduce Visual-RAG, a question-answering benchmark that targets visually grounded, knowledge-intensive questions. Unlike prior work, Visual-RAG requires text-to-image retrieval and the integration of retrieved clue images to extract visual evidence for answer generation. With Visual-RAG, we evaluate 5 open-source and 3 proprietary MLLMs, showcasing that images provide strong evidence in augmented generation. However, even state-of-the-art models struggle to efficiently extract and utilize visual knowledge. Our results highlight the need for improved visual retrieval, grounding, and attribution in multimodal RAG systems.
>
---
#### [replaced 064] Blending 3D Geometry and Machine Learning for Multi-View Stereopsis
- **分类: cs.CV; cs.AI; cs.CG; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.03470v2](http://arxiv.org/pdf/2505.03470v2)**

> **作者:** Vibhas Vats; Md. Alimoor Reza; David Crandall; Soon-heung Jung
>
> **备注:** A pre-print -- accepted at Neurocomputing. arXiv admin note: substantial text overlap with arXiv:2310.19583
>
> **摘要:** Traditional multi-view stereo (MVS) methods primarily depend on photometric and geometric consistency constraints. In contrast, modern learning-based algorithms often rely on the plane sweep algorithm to infer 3D geometry, applying explicit geometric consistency (GC) checks only as a post-processing step, with no impact on the learning process itself. In this work, we introduce GC MVSNet plus plus, a novel approach that actively enforces geometric consistency of reference view depth maps across multiple source views (multi view) and at various scales (multi scale) during the learning phase (see Fig. 1). This integrated GC check significantly accelerates the learning process by directly penalizing geometrically inconsistent pixels, effectively halving the number of training iterations compared to other MVS methods. Furthermore, we introduce a densely connected cost regularization network with two distinct block designs simple and feature dense optimized to harness dense feature connections for enhanced regularization. Extensive experiments demonstrate that our approach achieves a new state of the art on the DTU and BlendedMVS datasets and secures second place on the Tanks and Temples benchmark. To our knowledge, GC MVSNet plus plus is the first method to enforce multi-view, multi-scale supervised geometric consistency during learning. Our code is available.
>
---
