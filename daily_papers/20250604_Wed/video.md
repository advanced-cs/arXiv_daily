# 计算机视觉 cs.CV

- **最新发布 177 篇**

- **更新 107 篇**

## 最新发布

#### [new 001] SViMo: Synchronized Diffusion for Video and Motion Generation in Hand-object Interaction Scenarios
- **分类: cs.CV**

- **简介: 该论文属于手-物交互生成任务，旨在解决现有方法依赖预定义模型和运动数据、泛化能力差的问题。作者提出SViMo框架，通过同步扩散过程联合生成视频与运动，引入三模态自适应调制和3D全注意力机制，并设计视觉感知的交互扩散模型，实现无需姿态引导的高保真、物理合理的HOI序列生成。**

- **链接: [http://arxiv.org/pdf/2506.02444v1](http://arxiv.org/pdf/2506.02444v1)**

> **作者:** Lingwei Dang; Ruizhi Shao; Hongwen Zhang; Wei Min; Yebin Liu; Qingyao Wu
>
> **摘要:** Hand-Object Interaction (HOI) generation has significant application potential. However, current 3D HOI motion generation approaches heavily rely on predefined 3D object models and lab-captured motion data, limiting generalization capabilities. Meanwhile, HOI video generation methods prioritize pixel-level visual fidelity, often sacrificing physical plausibility. Recognizing that visual appearance and motion patterns share fundamental physical laws in the real world, we propose a novel framework that combines visual priors and dynamic constraints within a synchronized diffusion process to generate the HOI video and motion simultaneously. To integrate the heterogeneous semantics, appearance, and motion features, our method implements tri-modal adaptive modulation for feature aligning, coupled with 3D full-attention for modeling inter- and intra-modal dependencies. Furthermore, we introduce a vision-aware 3D interaction diffusion model that generates explicit 3D interaction sequences directly from the synchronized diffusion outputs, then feeds them back to establish a closed-loop feedback cycle. This architecture eliminates dependencies on predefined object models or explicit pose guidance while significantly enhancing video-motion consistency. Experimental results demonstrate our method's superiority over state-of-the-art approaches in generating high-fidelity, dynamically plausible HOI sequences, with notable generalization capabilities in unseen real-world scenarios. Project page at \href{https://github.com/Droliven}{https://github.com/Droliven}.
>
---
#### [new 002] VLCD: Vision-Language Contrastive Distillation for Accurate and Efficient Automatic Placenta Analysis
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于医学图像分析任务，旨在解决现有自动胎盘病理检测方法计算量大、部署受限的问题。作者提出了VLCD方法，包含文本锚定的知识蒸馏策略和无监督预训练策略，以提升模型效率与准确性，尤其适用于资源有限环境。**

- **链接: [http://arxiv.org/pdf/2506.02229v1](http://arxiv.org/pdf/2506.02229v1)**

> **作者:** Manas Mehta; Yimu Pan; Kelly Gallagher; Alison D. Gernand; Jeffery A. Goldstein; Delia Mwinyelle; Leena Mithal; James Z. Wang
>
> **备注:** Proceedings of the 9th International Workshop on Health Intelligence, in conjunction with the Annual AAAI Conference on Artificial Intelligence, Philadelphia, Pennsylvania, March 2025
>
> **摘要:** Pathological examination of the placenta is an effective method for detecting and mitigating health risks associated with childbirth. Recent advancements in AI have enabled the use of photographs of the placenta and pathology reports for detecting and classifying signs of childbirth-related pathologies. However, existing automated methods are computationally extensive, which limits their deployability. We propose two modifications to vision-language contrastive learning (VLC) frameworks to enhance their accuracy and efficiency: (1) text-anchored vision-language contrastive knowledge distillation (VLCD)-a new knowledge distillation strategy for medical VLC pretraining, and (2) unsupervised predistillation using a large natural images dataset for improved initialization. Our approach distills efficient neural networks that match or surpass the teacher model in performance while achieving model compression and acceleration. Our results showcase the value of unsupervised predistillation in improving the performance and robustness of our approach, specifically for lower-quality images. VLCD serves as an effective way to improve the efficiency and deployability of medical VLC approaches, making AI-based healthcare solutions more accessible, especially in resource-constrained environments.
>
---
#### [new 003] Towards Geometry Problem Solving in the Large Model Era: A Survey
- **分类: cs.CV; math.GT**

- **简介: 该论文属于人工智能中的几何问题求解任务，旨在解决自动化几何推理的挑战。论文系统综述了大模型时代下几何问题求解的发展，聚焦于基准构建、图文解析与推理范式三方面，提出了统一分析框架，并指出了未来研究方向，如自动基准生成与可解释性神经符号集成。**

- **链接: [http://arxiv.org/pdf/2506.02690v1](http://arxiv.org/pdf/2506.02690v1)**

> **作者:** Yurui Zhao; Xiang Wang; Jiahong Liu; Irwin King; Zhitao Huang
>
> **备注:** 8pages, 4 figures, conference submission
>
> **摘要:** Geometry problem solving (GPS) represents a critical frontier in artificial intelligence, with profound applications in education, computer-aided design, and computational graphics. Despite its significance, automating GPS remains challenging due to the dual demands of spatial understanding and rigorous logical reasoning. Recent advances in large models have enabled notable breakthroughs, particularly for SAT-level problems, yet the field remains fragmented across methodologies, benchmarks, and evaluation frameworks. This survey systematically synthesizes GPS advancements through three core dimensions: (1) benchmark construction, (2) textual and diagrammatic parsing, and (3) reasoning paradigms. We further propose a unified analytical paradigm, assess current limitations, and identify emerging opportunities to guide future research toward human-level geometric reasoning, including automated benchmark generation and interpretable neuro-symbolic integration.
>
---
#### [new 004] RoadFormer : Local-Global Feature Fusion for Road Surface Classification in Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于道路表面分类（RSC）任务，旨在解决自动驾驶中细粒度路面类型识别问题。现有方法忽略相似路面纹理的区分，作者提出RoadFormer模型，融合局部与全局特征，并引入前景-背景模块提升复杂路面分类能力，取得显著效果。**

- **链接: [http://arxiv.org/pdf/2506.02358v1](http://arxiv.org/pdf/2506.02358v1)**

> **作者:** Tianze Wang; Zhang Zhang; Chao Sun
>
> **摘要:** The classification of the type of road surface (RSC) aims to utilize pavement features to identify the roughness, wet and dry conditions, and material information of the road surface. Due to its ability to effectively enhance road safety and traffic management, it has received widespread attention in recent years. In autonomous driving, accurate RSC allows vehicles to better understand the road environment, adjust driving strategies, and ensure a safer and more efficient driving experience. For a long time, vision-based RSC has been favored. However, existing visual classification methods have overlooked the exploration of fine-grained classification of pavement types (such as similar pavement textures). In this work, we propose a pure vision-based fine-grained RSC method for autonomous driving scenarios, which fuses local and global feature information through the stacking of convolutional and transformer modules. We further explore the stacking strategies of local and global feature extraction modules to find the optimal feature extraction strategy. In addition, since fine-grained tasks also face the challenge of relatively large intra-class differences and relatively small inter-class differences, we propose a Foreground-Background Module (FBM) that effectively extracts fine-grained context features of the pavement, enhancing the classification ability for complex pavements. Experiments conducted on a large-scale pavement dataset containing one million samples and a simplified dataset reorganized from this dataset achieved Top-1 classification accuracies of 92.52% and 96.50%, respectively, improving by 5.69% to 12.84% compared to SOTA methods. These results demonstrate that RoadFormer outperforms existing methods in RSC tasks, providing significant progress in improving the reliability of pavement perception in autonomous driving systems.
>
---
#### [new 005] RelationAdapter: Learning and Transferring Visual Relation with Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，旨在解决现有方法在非刚性变换和编辑意图迁移上的局限性。受大语言模型启发，作者提出RelationAdapter模块，结合扩散变压器模型，通过源-目标图像对提取并传递内容感知的编辑意图，并构建了Relation252K数据集进行评估，显著提升了生成质量和编辑性能。**

- **链接: [http://arxiv.org/pdf/2506.02528v1](http://arxiv.org/pdf/2506.02528v1)**

> **作者:** Yan Gong; Yiren Song; Yicheng Li; Chenglin Li; Yin Zhang
>
> **摘要:** Inspired by the in-context learning mechanism of large language models (LLMs), a new paradigm of generalizable visual prompt-based image editing is emerging. Existing single-reference methods typically focus on style or appearance adjustments and struggle with non-rigid transformations. To address these limitations, we propose leveraging source-target image pairs to extract and transfer content-aware editing intent to novel query images. To this end, we introduce RelationAdapter, a lightweight module that enables Diffusion Transformer (DiT) based models to effectively capture and apply visual transformations from minimal examples. We also introduce Relation252K, a comprehensive dataset comprising 218 diverse editing tasks, to evaluate model generalization and adaptability in visual prompt-driven scenarios. Experiments on Relation252K show that RelationAdapter significantly improves the model's ability to understand and transfer editing intent, leading to notable gains in generation quality and overall editing performance.
>
---
#### [new 006] Towards Explicit Geometry-Reflectance Collaboration for Generalized LiDAR Segmentation in Adverse Weather
- **分类: cs.CV**

- **简介: 该论文属于激光雷达语义分割任务，旨在解决恶劣天气下分割精度下降的问题。通过提出几何-反射协作框架，分离处理几何与反射特征，并采用多级协作模块抑制干扰信息，提升了模型鲁棒性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.02396v1](http://arxiv.org/pdf/2506.02396v1)**

> **作者:** Longyu Yang; Ping Hu; Shangbo Yuan; Lu Zhang; Jun Liu; Hengtao Shen; Xiaofeng Zhu
>
> **摘要:** Existing LiDAR semantic segmentation models often suffer from decreased accuracy when exposed to adverse weather conditions. Recent methods addressing this issue focus on enhancing training data through weather simulation or universal augmentation techniques. However, few works have studied the negative impacts caused by the heterogeneous domain shifts in the geometric structure and reflectance intensity of point clouds. In this paper, we delve into this challenge and address it with a novel Geometry-Reflectance Collaboration (GRC) framework that explicitly separates feature extraction for geometry and reflectance. Specifically, GRC employs a dual-branch architecture designed to independently process geometric and reflectance features initially, thereby capitalizing on their distinct characteristic. Then, GRC adopts a robust multi-level feature collaboration module to suppress redundant and unreliable information from both branches. Consequently, without complex simulation or augmentation, our method effectively extracts intrinsic information about the scene while suppressing interference, thus achieving better robustness and generalization in adverse weather conditions. We demonstrate the effectiveness of GRC through comprehensive experiments on challenging benchmarks, showing that our method outperforms previous approaches and establishes new state-of-the-art results.
>
---
#### [new 007] Smartflow: Enabling Scalable Spatiotemporal Geospatial Research
- **分类: cs.CV; cs.AI**

- **简介: 论文介绍了Smartflow框架，属于地理空间数据处理与分析任务，旨在解决大规模时空数据管理与模型开发难题。工作包括构建基于Kubernetes的云平台，整合STAC标准数据、数据立方体处理、模型实验管理和可扩展计算，支持大范围地理区域和图像档案的分析，并展示了用于监测重型建设的神经网络应用。**

- **链接: [http://arxiv.org/pdf/2506.03022v1](http://arxiv.org/pdf/2506.03022v1)**

> **作者:** David McVicar; Brian Avant; Adrian Gould; Diego Torrejon; Charles Della Porta; Ryan Mukherjee
>
> **摘要:** BlackSky introduces Smartflow, a cloud-based framework enabling scalable spatiotemporal geospatial research built on open-source tools and technologies. Using STAC-compliant catalogs as a common input, heterogeneous geospatial data can be processed into standardized datacubes for analysis and model training. Model experimentation is managed using a combination of tools, including ClearML, Tensorboard, and Apache Superset. Underpinning Smartflow is Kubernetes, which orchestrates the provisioning and execution of workflows to support both horizontal and vertical scalability. This combination of features makes Smartflow well-suited for geospatial model development and analysis over large geographic areas, time scales, and expansive image archives. We also present a novel neural architecture, built using Smartflow, to monitor large geographic areas for heavy construction. Qualitative results based on data from the IARPA Space-based Machine Automated Recognition Technique (SMART) program are presented that show the model is capable of detecting heavy construction throughout all major phases of development.
>
---
#### [new 008] Are classical deep neural networks weakly adversarially robust?
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像识别任务，旨在解决深度神经网络对抗攻击鲁棒性弱的问题。作者利用DNN各层特征路径与类别中心路径的相关性，提出一种无需复杂防御策略的对抗样本检测方法，实验表明其在保持较高干净准确率的同时具备一定对抗鲁棒性，挑战了传统认知。**

- **链接: [http://arxiv.org/pdf/2506.02016v1](http://arxiv.org/pdf/2506.02016v1)**

> **作者:** Nuolin Sun; Linyuan Wang; Dongyang Li; Bin Yan; Lei Li
>
> **摘要:** Adversarial attacks have received increasing attention and it has been widely recognized that classical DNNs have weak adversarial robustness. The most commonly used adversarial defense method, adversarial training, improves the adversarial accuracy of DNNs by generating adversarial examples and retraining the model. However, adversarial training requires a significant computational overhead. In this paper, inspired by existing studies focusing on the clustering properties of DNN output features at each layer and the Progressive Feedforward Collapse phenomenon, we propose a method for adversarial example detection and image recognition that uses layer-wise features to construct feature paths and computes the correlation between the examples feature paths and the class-centered feature paths. Experimental results show that the recognition method achieves 82.77% clean accuracy and 44.17% adversarial accuracy on the ResNet-20 with PFC. Compared to the adversarial training method with 77.64% clean accuracy and 52.94% adversarial accuracy, our method exhibits a trade-off without relying on computationally expensive defense strategies. Furthermore, on the standard ResNet-18, our method maintains this advantage with respective metrics of 80.01% and 46.1%. This result reveals inherent adversarial robustness in DNNs, challenging the conventional understanding of the weak adversarial robustness in DNNs.
>
---
#### [new 009] Learning Pyramid-structured Long-range Dependencies for 3D Human Pose Estimation
- **分类: cs.CV**

- **简介: 该论文属于3D人体姿态估计任务，旨在解决现有方法在建模长程依赖时引入噪声和模型复杂的问题。论文提出了一种金字塔图注意力模块（PGA）和金字塔图变换网络（PGFormer），通过跨尺度学习人体关节间的长程依赖关系，有效降低了误差并减小了模型规模。**

- **链接: [http://arxiv.org/pdf/2506.02853v1](http://arxiv.org/pdf/2506.02853v1)**

> **作者:** Mingjie Wei; Xuemei Xie; Yutong Zhong; Guangming Shi
>
> **备注:** Accepted by IEEE Transactions on Multimedia (TMM)
>
> **摘要:** Action coordination in human structure is indispensable for the spatial constraints of 2D joints to recover 3D pose. Usually, action coordination is represented as a long-range dependence among body parts. However, there are two main challenges in modeling long-range dependencies. First, joints should not only be constrained by other individual joints but also be modulated by the body parts. Second, existing methods make networks deeper to learn dependencies between non-linked parts. They introduce uncorrelated noise and increase the model size. In this paper, we utilize a pyramid structure to better learn potential long-range dependencies. It can capture the correlation across joints and groups, which complements the context of the human sub-structure. In an effective cross-scale way, it captures the pyramid-structured long-range dependence. Specifically, we propose a novel Pyramid Graph Attention (PGA) module to capture long-range cross-scale dependencies. It concatenates information from various scales into a compact sequence, and then computes the correlation between scales in parallel. Combining PGA with graph convolution modules, we develop a Pyramid Graph Transformer (PGFormer) for 3D human pose estimation, which is a lightweight multi-scale transformer architecture. It encapsulates human sub-structures into self-attention by pooling. Extensive experiments show that our approach achieves lower error and smaller model size than state-of-the-art methods on Human3.6M and MPI-INF-3DHP datasets. The code is available at https://github.com/MingjieWe/PGFormer.
>
---
#### [new 010] The Devil is in the Darkness: Diffusion-Based Nighttime Dehazing Anchored in Brightness Perception
- **分类: cs.CV**

- **简介: 该论文属于图像去雾任务，旨在解决夜间图像去雾并转换为白天亮度的问题。现有方法在数据合成和亮度映射上存在不足，本文提出了DiffND框架，包含亮度一致的数据合成方法和结合扩散模型与亮度感知的去雾模型，实现了更真实的夜间去雾效果。**

- **链接: [http://arxiv.org/pdf/2506.02395v1](http://arxiv.org/pdf/2506.02395v1)**

> **作者:** Xiaofeng Cong; Yu-Xin Zhang; Haoran Wei; Yeying Jin; Junming Hou; Jie Gui; Jing Zhang; Dacheng Tao
>
> **摘要:** While nighttime image dehazing has been extensively studied, converting nighttime hazy images to daytime-equivalent brightness remains largely unaddressed. Existing methods face two critical limitations: (1) datasets overlook the brightness relationship between day and night, resulting in the brightness mapping being inconsistent with the real world during image synthesis; and (2) models do not explicitly incorporate daytime brightness knowledge, limiting their ability to reconstruct realistic lighting. To address these challenges, we introduce the Diffusion-Based Nighttime Dehazing (DiffND) framework, which excels in both data synthesis and lighting reconstruction. Our approach starts with a data synthesis pipeline that simulates severe distortions while enforcing brightness consistency between synthetic and real-world scenes, providing a strong foundation for learning night-to-day brightness mapping. Next, we propose a restoration model that integrates a pre-trained diffusion model guided by a brightness perception network. This design harnesses the diffusion model's generative ability while adapting it to nighttime dehazing through brightness-aware optimization. Experiments validate our dataset's utility and the model's superior performance in joint haze removal and brightness mapping.
>
---
#### [new 011] Object-centric Self-improving Preference Optimization for Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决多模态大语言模型在细粒度视觉理解上的不足。通过提出对象中心的自优化偏好框架OSPO，利用模型自身推理能力构建高质量对比数据，提升生成效果。**

- **链接: [http://arxiv.org/pdf/2506.02015v1](http://arxiv.org/pdf/2506.02015v1)**

> **作者:** Yoonjin Oh; Yongjin Kim; Hyomin Kim; Donghwan Chi; Sungwoong Kim
>
> **摘要:** Recent advancements in Multimodal Large Language Models (MLLMs) have significantly improved both image understanding and generation capabilities. Despite these improvements, MLLMs still struggle with fine-grained visual comprehension, particularly in text-to-image generation tasks. While preference optimization methods have been explored to address these limitations in image understanding tasks, their application to image generation remains largely underexplored. To address this gap, we propose an Object-centric Self-improving Preference Optimization (OSPO) framework designed for text-to-image generation by MLLMs. OSPO leverages the intrinsic reasoning abilities of MLLMs without requiring any external datasets or models. OSPO emphasizes the importance of high-quality preference pair data, which is critical for effective preference optimization. To achieve this, it introduces a self-improving mechanism that autonomously constructs object-level contrastive preference pairs through object-centric prompt perturbation, densification and VQA scoring. This process eliminates ambiguous or disproportionate variations commonly found in naively generated preference pairs, thereby enhancing the effectiveness of preference optimization. We validate OSPO on three representative compositional text-to-image benchmarks, demonstrating substantial performance gains over baseline models.
>
---
#### [new 012] Dynamic-Aware Video Distillation: Optimizing Temporal Resolution Based on Video Semantics
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频数据集蒸馏任务，旨在解决视频数据冗余问题。现有方法假设视频时间冗余一致，效果受限。本文提出DAViD，采用强化学习根据视频语义动态调整时间分辨率，提升蒸馏效果，是首个结合语义与自适应时间分辨率的视频蒸馏方法。**

- **链接: [http://arxiv.org/pdf/2506.02021v1](http://arxiv.org/pdf/2506.02021v1)**

> **作者:** Yinjie Zhao; Heng Zhao; Bihan Wen; Yew-Soon Ong; Joey Tianyi Zhou
>
> **摘要:** With the rapid development of vision tasks and the scaling on datasets and models, redundancy reduction in vision datasets has become a key area of research. To address this issue, dataset distillation (DD) has emerged as a promising approach to generating highly compact synthetic datasets with significantly less redundancy while preserving essential information. However, while DD has been extensively studied for image datasets, DD on video datasets remains underexplored. Video datasets present unique challenges due to the presence of temporal information and varying levels of redundancy across different classes. Existing DD approaches assume a uniform level of temporal redundancy across all different video semantics, which limits their effectiveness on video datasets. In this work, we propose Dynamic-Aware Video Distillation (DAViD), a Reinforcement Learning (RL) approach to predict the optimal Temporal Resolution of the synthetic videos. A teacher-in-the-loop reward function is proposed to update the RL agent policy. To the best of our knowledge, this is the first study to introduce adaptive temporal resolution based on video semantics in video dataset distillation. Our approach significantly outperforms existing DD methods, demonstrating substantial improvements in performance. This work paves the way for future research on more efficient and semantic-adaptive video dataset distillation research.
>
---
#### [new 013] Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉-语言对齐任务，旨在解决无需人工标注的图文匹配问题。作者提出利用循环一致性作为监督信号，通过文本生成图像再回译的方式构建偏好数据集，并训练奖励模型提升图文对齐效果。方法在多项任务中表现优异，具有良好的扩展性。**

- **链接: [http://arxiv.org/pdf/2506.02095v1](http://arxiv.org/pdf/2506.02095v1)**

> **作者:** Hyojin Bahng; Caroline Chan; Fredo Durand; Phillip Isola
>
> **摘要:** Learning alignment between language and vision is a fundamental challenge, especially as multimodal data becomes increasingly detailed and complex. Existing methods often rely on collecting human or AI preferences, which can be costly and time-intensive. We propose an alternative approach that leverages cycle consistency as a supervisory signal. Given an image and generated text, we map the text back to image space using a text-to-image model and compute the similarity between the original image and its reconstruction. Analogously, for text-to-image generation, we measure the textual similarity between an input caption and its reconstruction through the cycle. We use the cycle consistency score to rank candidates and construct a preference dataset of 866K comparison pairs. The reward model trained on our dataset outperforms state-of-the-art alignment metrics on detailed captioning, with superior inference-time scalability when used as a verifier for Best-of-N sampling. Furthermore, performing DPO and Diffusion DPO using our dataset enhances performance across a wide range of vision-language tasks and text-to-image generation. Our dataset, model, and code are at https://cyclereward.github.io
>
---
#### [new 014] LEG-SLAM: Real-Time Language-Enhanced Gaussian Splatting for SLAM
- **分类: cs.CV**

- **简介: 论文提出LEG-SLAM，属于SLAM任务，旨在实现实时语义3D高斯泼溅建图与定位。解决现有方法难以融合语义信息且实时性差的问题，结合高斯泼溅、视觉语言特征提取和在线密集SLAM，实现高质量、语义丰富的实时场景重建。**

- **链接: [http://arxiv.org/pdf/2506.03073v1](http://arxiv.org/pdf/2506.03073v1)**

> **作者:** Roman Titkov; Egor Zubkov; Dmitry Yudin; Jaafar Mahmoud; Malik Mohrat; Gennady Sidorov
>
> **摘要:** Modern Gaussian Splatting methods have proven highly effective for real-time photorealistic rendering of 3D scenes. However, integrating semantic information into this representation remains a significant challenge, especially in maintaining real-time performance for SLAM (Simultaneous Localization and Mapping) applications. In this work, we introduce LEG-SLAM -- a novel approach that fuses an optimized Gaussian Splatting implementation with visual-language feature extraction using DINOv2 followed by a learnable feature compressor based on Principal Component Analysis, while enabling an online dense SLAM. Our method simultaneously generates high-quality photorealistic images and semantically labeled scene maps, achieving real-time scene reconstruction with more than 10 fps on the Replica dataset and 18 fps on ScanNet. Experimental results show that our approach significantly outperforms state-of-the-art methods in reconstruction speed while achieving competitive rendering quality. The proposed system eliminates the need for prior data preparation such as camera's ego motion or pre-computed static semantic maps. With its potential applications in autonomous robotics, augmented reality, and other interactive domains, LEG-SLAM represents a significant step forward in real-time semantic 3D Gaussian-based SLAM. Project page: https://titrom025.github.io/LEG-SLAM/
>
---
#### [new 015] UniWorld: High-Resolution Semantic Encoders for Unified Visual Understanding and Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉-语言统一建模任务，旨在解决现有模型在图像感知与编辑能力上的不足。受GPT-4o-Image启发，作者提出UniWorld框架，基于语义编码器实现图像理解、生成与编辑。仅用1%的数据量便超越BAGEL，在多任务上表现优异，并开源全部资源。**

- **链接: [http://arxiv.org/pdf/2506.03147v1](http://arxiv.org/pdf/2506.03147v1)**

> **作者:** Bin Lin; Zongjian Li; Xinhua Cheng; Yuwei Niu; Yang Ye; Xianyi He; Shenghai Yuan; Wangbo Yu; Shaodong Wang; Yunyang Ge; Yatian Pang; Li Yuan
>
> **摘要:** Although existing unified models deliver strong performance on vision-language understanding and text-to-image generation, their models are limited in exploring image perception and manipulation tasks, which are urgently desired by users for wide applications. Recently, OpenAI released their powerful GPT-4o-Image model for comprehensive image perception and manipulation, achieving expressive capability and attracting community interests. By observing the performance of GPT-4o-Image in our carefully constructed experiments, we infer that GPT-4o-Image leverages features extracted by semantic encoders instead of VAE, while VAEs are considered essential components in many image manipulation models. Motivated by such inspiring observations, we present a unified generative framework named UniWorld based on semantic features provided by powerful visual-language models and contrastive semantic encoders. As a result, we build a strong unified model using only 1% amount of BAGEL's data, which consistently outperforms BAGEL on image editing benchmarks. UniWorld also maintains competitive image understanding and generation capabilities, achieving strong performance across multiple image perception tasks. We fully open-source our models, including model weights, training and evaluation scripts, and datasets.
>
---
#### [new 016] DyTact: Capturing Dynamic Contacts in Hand-Object Manipulation
- **分类: cs.CV**

- **简介: 该论文属于手-物交互建模任务，旨在解决动态接触捕捉中因遮挡、复杂表面细节和现有技术限制导致的精度不足问题。作者提出了DyTact方法，通过结合2D高斯surfels与MANO网格，利用动态建模、优化加速及自适应采样策略，实现了高精度、高效的动态接触估计与新视角合成。**

- **链接: [http://arxiv.org/pdf/2506.03103v1](http://arxiv.org/pdf/2506.03103v1)**

> **作者:** Xiaoyan Cong; Angela Xing; Chandradeep Pokhariya; Rao Fu; Srinath Sridhar
>
> **摘要:** Reconstructing dynamic hand-object contacts is essential for realistic manipulation in AI character animation, XR, and robotics, yet it remains challenging due to heavy occlusions, complex surface details, and limitations in existing capture techniques. In this paper, we introduce DyTact, a markerless capture method for accurately capturing dynamic contact in hand-object manipulations in a non-intrusive manner. Our approach leverages a dynamic, articulated representation based on 2D Gaussian surfels to model complex manipulations. By binding these surfels to MANO meshes, DyTact harnesses the inductive bias of template models to stabilize and accelerate optimization. A refinement module addresses time-dependent high-frequency deformations, while a contact-guided adaptive sampling strategy selectively increases surfel density in contact regions to handle heavy occlusion. Extensive experiments demonstrate that DyTact not only achieves state-of-the-art dynamic contact estimation accuracy but also significantly improves novel view synthesis quality, all while operating with fast optimization and efficient memory usage. Project Page: https://oliver-cong02.github.io/DyTact.github.io/ .
>
---
#### [new 017] Research on Driving Scenario Technology Based on Multimodal Large Lauguage Model Optimization
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自动驾驶场景理解任务，旨在解决复杂驾驶环境下多模态模型的应用难题。论文提出了一种综合优化方法，涵盖动态提示调整、数据集构建、模型训练与部署，提升模型对关键驾驶任务的准确性和资源利用效率。**

- **链接: [http://arxiv.org/pdf/2506.02014v1](http://arxiv.org/pdf/2506.02014v1)**

> **作者:** Wang Mengjie; Zhu Huiping; Li Jian; Shi Wenxiu; Zhang Song
>
> **摘要:** With the advancement of autonomous and assisted driving technologies, higher demands are placed on the ability to understand complex driving scenarios. Multimodal general large models have emerged as a solution for this challenge. However, applying these models in vertical domains involves difficulties such as data collection, model training, and deployment optimization. This paper proposes a comprehensive method for optimizing multimodal models in driving scenarios, including cone detection, traffic light recognition, speed limit recommendation, and intersection alerts. The method covers key aspects such as dynamic prompt optimization, dataset construction, model training, and deployment. Specifically, the dynamic prompt optimization adjusts the prompts based on the input image content to focus on objects affecting the ego vehicle, enhancing the model's task-specific focus and judgment capabilities. The dataset is constructed by combining real and synthetic data to create a high-quality and diverse multimodal training dataset, improving the model's generalization in complex driving environments. In model training, advanced techniques like knowledge distillation, dynamic fine-tuning, and quantization are integrated to reduce storage and computational costs while boosting performance. Experimental results show that this systematic optimization method not only significantly improves the model's accuracy in key tasks but also achieves efficient resource utilization, providing strong support for the practical application of driving scenario perception technologies.
>
---
#### [new 018] Improve Multi-Modal Embedding Learning via Explicit Hard Negative Gradient Amplifying
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多模态表示学习任务，旨在提升多模态大语言模型中的对比学习效果。通过分析info-NCE损失的梯度，提出显式放大难负样本梯度的方法，增强模型学习判别性嵌入的能力。基于LLaVA-OneVision-7B架构训练的模型在MMEB基准上达到SOTA，并在自研MLLM QQMM上取得榜首成绩。**

- **链接: [http://arxiv.org/pdf/2506.02020v1](http://arxiv.org/pdf/2506.02020v1)**

> **作者:** Youze Xue; Dian Li; Gang Liu
>
> **摘要:** With the rapid advancement of multi-modal large language models (MLLMs) in recent years, the foundational Contrastive Language-Image Pretraining (CLIP) framework has been successfully extended to MLLMs, enabling more powerful and universal multi-modal embeddings for a wide range of retrieval tasks. Despite these developments, the core contrastive learning paradigm remains largely unchanged from CLIP-style models to MLLMs. Within this framework, the effective mining of hard negative samples continues to be a critical factor for enhancing performance. Prior works have introduced both offline and online strategies for hard negative mining to improve the efficiency of contrastive learning. While these approaches have led to improved multi-modal embeddings, the specific contribution of each hard negative sample to the learning process has not been thoroughly investigated. In this work, we conduct a detailed analysis of the gradients of the info-NCE loss with respect to the query, positive, and negative samples, elucidating the role of hard negatives in updating model parameters. Building upon this analysis, we propose to explicitly amplify the gradients associated with hard negative samples, thereby encouraging the model to learn more discriminative embeddings. Our multi-modal embedding model, trained with the proposed Explicit Gradient Amplifier and based on the LLaVA-OneVision-7B architecture, achieves state-of-the-art performance on the MMEB benchmark compared to previous methods utilizing the same MLLM backbone. Furthermore, when integrated with our self-developed MLLM, QQMM, our approach attains the top rank on the MMEB leaderboard. Code and models are released on https://github.com/QQ-MM/QQMM-embed.
>
---
#### [new 019] Implicit Deformable Medical Image Registration with Learnable Kernels
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像配准任务，旨在解决现有AI方法在形变估计中不可靠的问题。作者提出一种基于可学习核函数的隐式形变配准框架，通过稀疏关键点重建密集位移场，并采用由粗到细的层次结构提升精度。实验表明该方法在零样本注册任务中表现优异，兼具准确性与临床实用性。**

- **链接: [http://arxiv.org/pdf/2506.02150v1](http://arxiv.org/pdf/2506.02150v1)**

> **作者:** Stefano Fogarollo; Gregor Laimer; Reto Bale; Matthias Harders
>
> **备注:** MICCAI 2025 Provisional Accept
>
> **摘要:** Deformable medical image registration is an essential task in computer-assisted interventions. This problem is particularly relevant to oncological treatments, where precise image alignment is necessary for tracking tumor growth, assessing treatment response, and ensuring accurate delivery of therapies. Recent AI methods can outperform traditional techniques in accuracy and speed, yet they often produce unreliable deformations that limit their clinical adoption. In this work, we address this challenge and introduce a novel implicit registration framework that can predict accurate and reliable deformations. Our insight is to reformulate image registration as a signal reconstruction problem: we learn a kernel function that can recover the dense displacement field from sparse keypoint correspondences. We integrate our method in a novel hierarchical architecture, and estimate the displacement field in a coarse-to-fine manner. Our formulation also allows for efficient refinement at test time, permitting clinicians to easily adjust registrations when needed. We validate our method on challenging intra-patient thoracic and abdominal zero-shot registration tasks, using public and internal datasets from the local University Hospital. Our method not only shows competitive accuracy to state-of-the-art approaches, but also bridges the generalization gap between implicit and explicit registration techniques. In particular, our method generates deformations that better preserve anatomical relationships and matches the performance of specialized commercial systems, underscoring its potential for clinical adoption.
>
---
#### [new 020] Efficient Test-time Adaptive Object Detection via Sensitivity-Guided Pruning
- **分类: cs.CV**

- **简介: 该论文属于持续测试时自适应目标检测（CTTA-OD）任务，旨在解决在连续域偏移下模型适应性与计算效率难以兼顾的问题。作者提出基于敏感性引导剪枝的方法，通过量化特征通道对域差异的敏感度，抑制或剪枝敏感通道，并引入随机通道重激活机制以恢复潜在有用特征，从而在提升适应性能的同时降低计算开销。**

- **链接: [http://arxiv.org/pdf/2506.02462v1](http://arxiv.org/pdf/2506.02462v1)**

> **作者:** Kunyu Wang; Xueyang Fu; Xin Lu; Chengjie Ge; Chengzhi Cao; Wei Zhai; Zheng-Jun Zha
>
> **备注:** Accepted as CVPR 2025 oral paper
>
> **摘要:** Continual test-time adaptive object detection (CTTA-OD) aims to online adapt a source pre-trained detector to ever-changing environments during inference under continuous domain shifts. Most existing CTTA-OD methods prioritize effectiveness while overlooking computational efficiency, which is crucial for resource-constrained scenarios. In this paper, we propose an efficient CTTA-OD method via pruning. Our motivation stems from the observation that not all learned source features are beneficial; certain domain-sensitive feature channels can adversely affect target domain performance. Inspired by this, we introduce a sensitivity-guided channel pruning strategy that quantifies each channel based on its sensitivity to domain discrepancies at both image and instance levels. We apply weighted sparsity regularization to selectively suppress and prune these sensitive channels, focusing adaptation efforts on invariant ones. Additionally, we introduce a stochastic channel reactivation mechanism to restore pruned channels, enabling recovery of potentially useful features and mitigating the risks of early pruning. Extensive experiments on three benchmarks show that our method achieves superior adaptation performance while reducing computational overhead by 12% in FLOPs compared to the recent SOTA method.
>
---
#### [new 021] Explicitly Modeling Subcortical Vision with a Neuro-Inspired Front-End Improves CNN Robustness
- **分类: cs.CV; q-bio.NC**

- **简介: 该论文属于计算机视觉任务，旨在提升卷积神经网络（CNN）在面对视觉扰动和域外图像时的鲁棒性。受生物视觉启发，作者提出了Early Vision Networks（EVNets），结合模拟初级视觉皮层（V1）的VOneBlock和新设计的SubcorticalBlock，以更好地模仿生物视觉处理机制。实验表明，EVNets不仅提升了模型对V1响应的对齐度和形状偏好，还在多种鲁棒性评估中超越了基础CNN架构，并与数据增强方法结合后进一步提升了性能。**

- **链接: [http://arxiv.org/pdf/2506.03089v1](http://arxiv.org/pdf/2506.03089v1)**

> **作者:** Lucas Piper; Arlindo L. Oliveira; Tiago Marques
>
> **摘要:** Convolutional neural networks (CNNs) trained on object recognition achieve high task performance but continue to exhibit vulnerability under a range of visual perturbations and out-of-domain images, when compared with biological vision. Prior work has demonstrated that coupling a standard CNN with a front-end block (VOneBlock) that mimics the primate primary visual cortex (V1) can improve overall model robustness. Expanding on this, we introduce Early Vision Networks (EVNets), a new class of hybrid CNNs that combine the VOneBlock with a novel SubcorticalBlock, whose architecture draws from computational models in neuroscience and is parameterized to maximize alignment with subcortical responses reported across multiple experimental studies. Without being optimized to do so, the assembly of the SubcorticalBlock with the VOneBlock improved V1 alignment across most standard V1 benchmarks, and better modeled extra-classical receptive field phenomena. In addition, EVNets exhibit stronger emergent shape bias and overperform the base CNN architecture by 8.5% on an aggregate benchmark of robustness evaluations, including adversarial perturbations, common corruptions, and domain shifts. Finally, we show that EVNets can be further improved when paired with a state-of-the-art data augmentation technique, surpassing the performance of the isolated data augmentation approach by 7.3% on our robustness benchmark. This result reveals complementary benefits between changes in architecture to better mimic biology and training-based machine learning approaches.
>
---
#### [new 022] GaRA-SAM: Robustifying Segment Anything Model with Gated-Rank Adaptation
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务，旨在提升Segment Anything Model（SAM）在输入退化情况下的鲁棒性。作者提出Gated-Rank Adaptation（GaRA），通过在SAM中引入轻量级适配器，动态调整权重矩阵的秩，实现细粒度、输入感知的增强，从而提高模型在复杂场景下的性能。**

- **链接: [http://arxiv.org/pdf/2506.02882v1](http://arxiv.org/pdf/2506.02882v1)**

> **作者:** Sohyun Lee; Yeho Kwon; Lukas Hoyer; Suha Kwak
>
> **摘要:** Improving robustness of the Segment Anything Model (SAM) to input degradations is critical for its deployment in high-stakes applications such as autonomous driving and robotics. Our approach to this challenge prioritizes three key aspects: first, parameter efficiency to maintain the inherent generalization capability of SAM; second, fine-grained and input-aware robustification to precisely address the input corruption; and third, adherence to standard training protocols for ease of training. To this end, we propose gated-rank adaptation (GaRA). GaRA introduces lightweight adapters into intermediate layers of the frozen SAM, where each adapter dynamically adjusts the effective rank of its weight matrix based on the input by selectively activating (rank-1) components of the matrix using a learned gating module. This adjustment enables fine-grained and input-aware robustification without compromising the generalization capability of SAM. Our model, GaRA-SAM, significantly outperforms prior work on all robust segmentation benchmarks. In particular, it surpasses the previous best IoU score by up to 21.3\%p on ACDC, a challenging real corrupted image dataset.
>
---
#### [new 023] Context as Memory: Scene-Consistent Interactive Long Video Generation with Memory Retrieval
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决长视频生成中的场景一致性问题。现有方法难以有效利用历史上下文作为记忆。论文提出“Context-as-Memory”方法，将历史帧直接作为内存存储并输入模型，结合视野重叠检索关键帧，减少计算开销。实验表明其在长视频生成中具有更强的记忆能力和泛化性。**

- **链接: [http://arxiv.org/pdf/2506.03141v1](http://arxiv.org/pdf/2506.03141v1)**

> **作者:** Jiwen Yu; Jianhong Bai; Yiran Qin; Quande Liu; Xintao Wang; Pengfei Wan; Di Zhang; Xihui Liu
>
> **摘要:** Recent advances in interactive video generation have shown promising results, yet existing approaches struggle with scene-consistent memory capabilities in long video generation due to limited use of historical context. In this work, we propose Context-as-Memory, which utilizes historical context as memory for video generation. It includes two simple yet effective designs: (1) storing context in frame format without additional post-processing; (2) conditioning by concatenating context and frames to be predicted along the frame dimension at the input, requiring no external control modules. Furthermore, considering the enormous computational overhead of incorporating all historical context, we propose the Memory Retrieval module to select truly relevant context frames by determining FOV (Field of View) overlap between camera poses, which significantly reduces the number of candidate frames without substantial information loss. Experiments demonstrate that Context-as-Memory achieves superior memory capabilities in interactive long video generation compared to SOTAs, even generalizing effectively to open-domain scenarios not seen during training. The link of our project page is https://context-as-memory.github.io/.
>
---
#### [new 024] METok: Multi-Stage Event-based Token Compression for Efficient Long Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决长视频处理中计算冗余和效率低的问题。作者提出METok框架，通过三阶段基于事件的视觉标记压缩方法，在不损失准确率的前提下显著提升推理效率并降低内存消耗。**

- **链接: [http://arxiv.org/pdf/2506.02850v1](http://arxiv.org/pdf/2506.02850v1)**

> **作者:** Mengyue Wang; Shuo Chen; Kristian Kersting; Volker Tresp; Yunpu Ma
>
> **备注:** 14 pages, 10 figures
>
> **摘要:** Recent advances in Video Large Language Models (VLLMs) have significantly enhanced their ability to understand video content. Nonetheless, processing long videos remains challenging due to high computational demands and the redundancy present in the visual data. In this work, we propose METok, a training-free, Multi-stage Event-based Token compression framework designed to accelerate VLLMs' inference while preserving accuracy. METok progressively eliminates redundant visual tokens across three critical stages: (1) event-aware compression during vision encoding, (2) hierarchical token pruning in the prefilling stage based on semantic alignment and event importance, and (3) a decoding-stage KV Cache optimization that further reduces memory consumption. Our experiments on diverse video benchmarks demonstrate that METok achieves an optimal trade-off between efficiency and accuracy by dynamically selecting informative visual tokens. For instance, equipping LongVA-7B with METok realizes an 80.6% FLOPs reduction and 93.5% KV Cache memory savings, all while maintaining comparable or even superior accuracy.
>
---
#### [new 025] Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成模型对齐任务，旨在解决人类偏好多样性和现有方法优化过度、目标错位问题。论文提出SmPO-Diffusion方法，通过平滑偏好分布和反转技术模拟轨迹偏好，提升DPO目标并实现更精准对齐。**

- **链接: [http://arxiv.org/pdf/2506.02698v1](http://arxiv.org/pdf/2506.02698v1)**

> **作者:** Yunhong Lu; Qichao Wang; Hengyuan Cao; Xiaoyin Xu; Min Zhang
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Direct Preference Optimization (DPO) aligns text-to-image (T2I) generation models with human preferences using pairwise preference data. Although substantial resources are expended in collecting and labeling datasets, a critical aspect is often neglected: \textit{preferences vary across individuals and should be represented with more granularity.} To address this, we propose SmPO-Diffusion, a novel method for modeling preference distributions to improve the DPO objective, along with a numerical upper bound estimation for the diffusion optimization objective. First, we introduce a smoothed preference distribution to replace the original binary distribution. We employ a reward model to simulate human preferences and apply preference likelihood averaging to improve the DPO loss, such that the loss function approaches zero when preferences are similar. Furthermore, we utilize an inversion technique to simulate the trajectory preference distribution of the diffusion model, enabling more accurate alignment with the optimization objective. Our approach effectively mitigates issues of excessive optimization and objective misalignment present in existing methods through straightforward modifications. Our SmPO-Diffusion achieves state-of-the-art performance in preference evaluation, outperforming baselines across metrics with lower training costs. The project page is https://jaydenlyh.github.io/SmPO-project-page/.
>
---
#### [new 026] Sparse-vDiT: Unleashing the Power of Sparse Attention to Accelerate Video Diffusion Transformers
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视频生成任务，旨在解决视频扩散Transformer（vDiT）中注意力机制计算复杂度高、推理延迟大的问题。通过分析注意力图的稀疏模式，提出Sparse-vDiT框架，利用稀疏计算优化推理效率，在多个模型上实现显著加速，同时保持生成视频的高质量。**

- **链接: [http://arxiv.org/pdf/2506.03065v1](http://arxiv.org/pdf/2506.03065v1)**

> **作者:** Pengtao Chen; Xianfang Zeng; Maosen Zhao; Peng Ye; Mingzhu Shen; Wei Cheng; Gang Yu; Tao Chen
>
> **摘要:** While Diffusion Transformers (DiTs) have achieved breakthroughs in video generation, this long sequence generation task remains constrained by the quadratic complexity of attention mechanisms, resulting in significant inference latency. Through detailed analysis of attention maps in Video Diffusion Transformer (vDiT), we identify three recurring sparsity patterns: diagonal, multi-diagonal, and vertical-stripe structures. And even 3-6\% attention heads can be skipped. Crucially, these patterns exhibit strong layer-depth and head-position correlations but show limited dependence on the input content. Leveraging these findings, we propose Sparse-vDiT, a sparsity acceleration framework for vDiT comprising: 1) Pattern-optimized sparse kernels that replace dense attention with computationally efficient implementations for each identified sparsity pattern. 2) An offline sparse diffusion search algorithm that selects the optimal sparse computation strategy per layer and head via hardware-aware cost modeling. After determining the optimal configuration, we fuse heads within the same layer that share the same attention strategy, enhancing inference efficiency. Integrated into state-of-the-art vDiT models (CogVideoX1.5, HunyuanVideo, and Wan2.1), Sparse-vDiT achieves 2.09$\times$, 2.38$\times$, and 1.67$\times$ theoretical FLOP reduction, and actual inference speedups of 1.76$\times$, 1.85$\times$, and 1.58$\times$, respectively, while maintaining high visual fidelity, with PSNR values reaching 24.13, 27.09, and 22.59. Our work demonstrates that latent structural sparsity in vDiTs can be systematically exploited for long video synthesis.
>
---
#### [new 027] MVTD: A Benchmark Dataset for Maritime Visual Object Tracking
- **分类: cs.CV**

- **简介: 该论文属于视觉目标跟踪任务，旨在解决通用算法在复杂海事环境中性能下降的问题。作者构建了专门的海事视觉跟踪数据集MVTD，包含182个视频序列和四类目标，并验证了现有算法在此领域的性能退化及通过微调带来的提升效果。**

- **链接: [http://arxiv.org/pdf/2506.02866v1](http://arxiv.org/pdf/2506.02866v1)**

> **作者:** Ahsan Baidar Bakht; Muhayy Ud Din; Sajid Javed; Irfan Hussain
>
> **备注:** Submited to Nature Scientific Data
>
> **摘要:** Visual Object Tracking (VOT) is a fundamental task with widespread applications in autonomous navigation, surveillance, and maritime robotics. Despite significant advances in generic object tracking, maritime environments continue to present unique challenges, including specular water reflections, low-contrast targets, dynamically changing backgrounds, and frequent occlusions. These complexities significantly degrade the performance of state-of-the-art tracking algorithms, highlighting the need for domain-specific datasets. To address this gap, we introduce the Maritime Visual Tracking Dataset (MVTD), a comprehensive and publicly available benchmark specifically designed for maritime VOT. MVTD comprises 182 high-resolution video sequences, totaling approximately 150,000 frames, and includes four representative object classes: boat, ship, sailboat, and unmanned surface vehicle (USV). The dataset captures a diverse range of operational conditions and maritime scenarios, reflecting the real-world complexities of maritime environments. We evaluated 14 recent SOTA tracking algorithms on the MVTD benchmark and observed substantial performance degradation compared to their performance on general-purpose datasets. However, when fine-tuned on MVTD, these models demonstrate significant performance gains, underscoring the effectiveness of domain adaptation and the importance of transfer learning in specialized tracking contexts. The MVTD dataset fills a critical gap in the visual tracking community by providing a realistic and challenging benchmark for maritime scenarios. Dataset and Source Code can be accessed here "https://github.com/AhsanBaidar/MVTD".
>
---
#### [new 028] Application of convolutional neural networks in image super-resolution
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于图像超分辨率任务，旨在解决不同卷积神经网络方法在该领域的差异与联系缺乏系统总结的问题。论文分析了基于CNN的多种插值方法和模块，并通过实验比较其性能，最后指出了潜在研究方向与不足。**

- **链接: [http://arxiv.org/pdf/2506.02604v1](http://arxiv.org/pdf/2506.02604v1)**

> **作者:** Tian Chunwei; Song Mingjian; Zuo Wangmeng; Du Bo; Zhang Yanning; Zhang Shichao
>
> **备注:** It has been accepted by CAAI transactions on intelligent systems, in Chinese language
>
> **摘要:** Due to strong learning abilities of convolutional neural networks (CNNs), they have become mainstream methods for image super-resolution. However, there are big differences of different deep learning methods with different types. There is little literature to summarize relations and differences of different methods in image super-resolution. Thus, summarizing these literatures are important, according to loading capacity and execution speed of devices. This paper first introduces principles of CNNs in image super-resolution, then introduces CNNs based bicubic interpolation, nearest neighbor interpolation, bilinear interpolation, transposed convolution, sub-pixel layer, meta up-sampling for image super-resolution to analyze differences and relations of different CNNs based interpolations and modules, and compare performance of these methods by experiments. Finally, this paper gives potential research points and drawbacks and summarizes the whole paper, which can facilitate developments of CNNs in image super-resolution.
>
---
#### [new 029] ControlMambaIR: Conditional Controls with State-Space Model for Image Restoration
- **分类: cs.CV**

- **简介: 该论文属于图像恢复任务，旨在解决去雨、去模糊和去噪中的感知质量问题。作者提出ControlMambaIR方法，结合Mamba架构与扩散模型，实现精细化的条件控制。实验表明其在多种数据集上优于现有方法，尤其在感知指标LPIPS和FID上表现突出。**

- **链接: [http://arxiv.org/pdf/2506.02633v1](http://arxiv.org/pdf/2506.02633v1)**

> **作者:** Cheng Yang; Lijing Liang; Zhixun Su
>
> **摘要:** This paper proposes ControlMambaIR, a novel image restoration method designed to address perceptual challenges in image deraining, deblurring, and denoising tasks. By integrating the Mamba network architecture with the diffusion model, the condition network achieves refined conditional control, thereby enhancing the control and optimization of the image generation process. To evaluate the robustness and generalization capability of our method across various image degradation conditions, extensive experiments were conducted on several benchmark datasets, including Rain100H, Rain100L, GoPro, and SSID. The results demonstrate that our proposed approach consistently surpasses existing methods in perceptual quality metrics, such as LPIPS and FID, while maintaining comparable performance in image distortion metrics, including PSNR and SSIM, highlighting its effectiveness and adaptability. Notably, ablation experiments reveal that directly noise prediction in the diffusion process achieves better performance, effectively balancing noise suppression and detail preservation. Furthermore, the findings indicate that the Mamba architecture is particularly well-suited as a conditional control network for diffusion models, outperforming both CNN- and Attention-based approaches in this context. Overall, these results highlight the flexibility and effectiveness of ControlMambaIR in addressing a range of image restoration perceptual challenges.
>
---
#### [new 030] EDITOR: Effective and Interpretable Prompt Inversion for Text-to-Image Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于文本到图像扩散模型的提示逆任务，旨在解决生成图像的文本提示还原问题。现有方法在语义连贯性和效率上存在不足。作者提出EDITOR方法，结合图像描述模型初始化嵌入、潜在空间反演优化及嵌入转文本技术，提升了图像相似性、文本对齐和可解释性，并展示了其在多种生成任务中的应用效果。**

- **链接: [http://arxiv.org/pdf/2506.03067v1](http://arxiv.org/pdf/2506.03067v1)**

> **作者:** Mingzhe Li; Gehao Zhang; Zhenting Wang; Shiqing Ma; Siqi Pan; Richard Cartwright; Juan Zhai
>
> **摘要:** Text-to-image generation models~(e.g., Stable Diffusion) have achieved significant advancements, enabling the creation of high-quality and realistic images based on textual descriptions. Prompt inversion, the task of identifying the textual prompt used to generate a specific artifact, holds significant potential for applications including data attribution, model provenance, and watermarking validation. Recent studies introduced a delayed projection scheme to optimize for prompts representative of the vocabulary space, though challenges in semantic fluency and efficiency remain. Advanced image captioning models or visual large language models can generate highly interpretable prompts, but they often lack in image similarity. In this paper, we propose a prompt inversion technique called \sys for text-to-image diffusion models, which includes initializing embeddings using a pre-trained image captioning model, refining them through reverse-engineering in the latent space, and converting them to texts using an embedding-to-text model. Our experiments on the widely-used datasets, such as MS COCO, LAION, and Flickr, show that our method outperforms existing methods in terms of image similarity, textual alignment, prompt interpretability and generalizability. We further illustrate the application of our generated prompts in tasks such as cross-concept image synthesis, concept manipulation, evolutionary multi-concept generation and unsupervised segmentation.
>
---
#### [new 031] FORLA:Federated Object-centric Representation Learning with Slot Attention
- **分类: cs.CV; cs.LG**

- **简介: 论文提出FORLA，一种联邦物体中心表征学习框架，旨在从异构无标签数据集中学习高效视觉表征。核心问题是跨域无监督特征对齐与适应。方法包括共享特征适配器和槽注意力模块，通过学生-教师架构优化，实现对象级表征的跨客户端对齐。实验表明其在多数据集上优于集中式基线，在物体发现任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2506.02964v1](http://arxiv.org/pdf/2506.02964v1)**

> **作者:** Guiqiu Liao; Matjaz Jogan; Eric Eaton; Daniel A. Hashimoto
>
> **备注:** 24 pages, 6 figures
>
> **摘要:** Learning efficient visual representations across heterogeneous unlabeled datasets remains a central challenge in federated learning. Effective federated representations require features that are jointly informative across clients while disentangling domain-specific factors without supervision. We introduce FORLA, a novel framework for federated object-centric representation learning and feature adaptation across clients using unsupervised slot attention. At the core of our method is a shared feature adapter, trained collaboratively across clients to adapt features from foundation models, and a shared slot attention module that learns to reconstruct the adapted features. To optimize this adapter, we design a two-branch student-teacher architecture. In each client, a student decoder learns to reconstruct full features from foundation models, while a teacher decoder reconstructs their adapted, low-dimensional counterpart. The shared slot attention module bridges cross-domain learning by aligning object-level representations across clients. Experiments in multiple real-world datasets show that our framework not only outperforms centralized baselines on object discovery but also learns a compact, universal representation that generalizes well across domains. This work highlights federated slot attention as an effective tool for scalable, unsupervised visual representation learning from cross-domain data with distributed concepts.
>
---
#### [new 032] PBR-SR: Mesh PBR Texture Super Resolution from 2D Image Priors
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率任务，旨在解决低分辨率PBR纹理的高质量超分辨问题。作者提出PBR-SR方法，利用预训练图像先验，通过可微渲染与多视角一致性约束，迭代优化并生成高分辨率PBR纹理，无需额外训练，提升了纹理质量和渲染效果。**

- **链接: [http://arxiv.org/pdf/2506.02846v1](http://arxiv.org/pdf/2506.02846v1)**

> **作者:** Yujin Chen; Yinyu Nie; Benjamin Ummenhofer; Reiner Birkl; Michael Paulitsch; Matthias Nießner
>
> **备注:** Project page: https://terencecyj.github.io/projects/PBR-SR/, Video: https://youtu.be/eaM5S3Mt1RM
>
> **摘要:** We present PBR-SR, a novel method for physically based rendering (PBR) texture super resolution (SR). It outputs high-resolution, high-quality PBR textures from low-resolution (LR) PBR input in a zero-shot manner. PBR-SR leverages an off-the-shelf super-resolution model trained on natural images, and iteratively minimizes the deviations between super-resolution priors and differentiable renderings. These enhancements are then back-projected into the PBR map space in a differentiable manner to produce refined, high-resolution textures. To mitigate view inconsistencies and lighting sensitivity, which is common in view-based super-resolution, our method applies 2D prior constraints across multi-view renderings, iteratively refining the shared, upscaled textures. In parallel, we incorporate identity constraints directly in the PBR texture domain to ensure the upscaled textures remain faithful to the LR input. PBR-SR operates without any additional training or data requirements, relying entirely on pretrained image priors. We demonstrate that our approach produces high-fidelity PBR textures for both artist-designed and AI-generated meshes, outperforming both direct SR models application and prior texture optimization methods. Our results show high-quality outputs in both PBR and rendering evaluations, supporting advanced applications such as relighting.
>
---
#### [new 033] ORV: 4D Occupancy-centric Robot Video Generation
- **分类: cs.CV**

- **简介: 论文提出ORV框架，用于生成机器人视频，属于机器人仿真与视频生成任务。解决现有方法控制精度低、泛化差的问题。工作是利用4D语义占据序列作为细粒度表示，实现高质量、多视角的机器人操作视频生成。**

- **链接: [http://arxiv.org/pdf/2506.03079v1](http://arxiv.org/pdf/2506.03079v1)**

> **作者:** Xiuyu Yang; Bohan Li; Shaocong Xu; Nan Wang; Chongjie Ye; Zhaoxi Chen; Minghan Qin; Yikang Ding; Xin Jin; Hang Zhao; Hao Zhao
>
> **备注:** Project page: https://orangesodahub.github.io/ORV/ ; Code: https://github.com/OrangeSodahub/ORV
>
> **摘要:** Acquiring real-world robotic simulation data through teleoperation is notoriously time-consuming and labor-intensive. Recently, action-driven generative models have gained widespread adoption in robot learning and simulation, as they eliminate safety concerns and reduce maintenance efforts. However, the action sequences used in these methods often result in limited control precision and poor generalization due to their globally coarse alignment. To address these limitations, we propose ORV, an Occupancy-centric Robot Video generation framework, which utilizes 4D semantic occupancy sequences as a fine-grained representation to provide more accurate semantic and geometric guidance for video generation. By leveraging occupancy-based representations, ORV enables seamless translation of simulation data into photorealistic robot videos, while ensuring high temporal consistency and precise controllability. Furthermore, our framework supports the simultaneous generation of multi-view videos of robot gripping operations - an important capability for downstream robotic learning tasks. Extensive experimental results demonstrate that ORV consistently outperforms existing baseline methods across various datasets and sub-tasks. Demo, Code and Model: https://orangesodahub.github.io/ORV
>
---
#### [new 034] Dense Match Summarization for Faster Two-view Estimation
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉中的姿态估计任务，旨在解决基于密集匹配的两视角相对位姿估计计算效率低的问题。作者提出了一种高效的匹配汇总方法，在保持精度的同时显著提升了计算速度。**

- **链接: [http://arxiv.org/pdf/2506.02893v1](http://arxiv.org/pdf/2506.02893v1)**

> **作者:** Jonathan Astermark; Anders Heyden; Viktor Larsson
>
> **备注:** Accepted to Computer Vision and Pattern Recognition (CVPR) 2025
>
> **摘要:** In this paper, we speed up robust two-view relative pose from dense correspondences. Previous work has shown that dense matchers can significantly improve both accuracy and robustness in the resulting pose. However, the large number of matches comes with a significantly increased runtime during robust estimation in RANSAC. To avoid this, we propose an efficient match summarization scheme which provides comparable accuracy to using the full set of dense matches, while having 10-100x faster runtime. We validate our approach on standard benchmark datasets together with multiple state-of-the-art dense matchers.
>
---
#### [new 035] Quantifying task-relevant representational similarity using decision variable correlation
- **分类: cs.CV; cs.LG; q-bio.NC; q-bio.QM**

- **简介: 该论文属于神经科学与人工智能交叉任务，旨在解决模型与大脑在任务相关表征上的相似性问题。作者提出决策变量相关（DVC）方法，量化分类任务中个体样本的解码决策相关性，以评估深度模型与猴子V4/IT脑区在决策策略上的相似性。研究发现模型间相似性高，但模型与猴子相似性低，且随ImageNet性能提升而下降，表明二者在任务相关表征上存在根本差异。**

- **链接: [http://arxiv.org/pdf/2506.02164v1](http://arxiv.org/pdf/2506.02164v1)**

> **作者:** Yu; Qian; Wilson S. Geisler; Xue-Xin Wei
>
> **摘要:** Previous studies have compared the brain and deep neural networks trained on image classification. Intriguingly, while some suggest that their representations are highly similar, others argued the opposite. Here, we propose a new approach to characterize the similarity of the decision strategies of two observers (models or brains) using decision variable correlation (DVC). DVC quantifies the correlation between decoded decisions on individual samples in a classification task and thus can capture task-relevant information rather than general representational alignment. We evaluate this method using monkey V4/IT recordings and models trained on image classification tasks. We find that model--model similarity is comparable to monkey--monkey similarity, whereas model--monkey similarity is consistently lower and, surprisingly, decreases with increasing ImageNet-1k performance. While adversarial training enhances robustness, it does not improve model--monkey similarity in task-relevant dimensions; however, it markedly increases model--model similarity. Similarly, pre-training on larger datasets does not improve model--monkey similarity. These results suggest a fundamental divergence between the task-relevant representations in monkey V4/IT and those learned by models trained on image classification tasks.
>
---
#### [new 036] LumosFlow: Motion-Guided Long Video Generation
- **分类: cs.CV**

- **简介: 该论文属于长视频生成任务，旨在解决生成内容重复、过渡不自然的问题。作者提出LumosFlow框架，通过大运动文本到视频扩散模型生成关键帧，并利用光流扩散模型与控制网络合成中间帧，实现15倍插值，提升视频的连贯性与质量。**

- **链接: [http://arxiv.org/pdf/2506.02497v1](http://arxiv.org/pdf/2506.02497v1)**

> **作者:** Jiahao Chen; Hangjie Yuan; Yichen Qian; Jingyun Liang; Jiazheng Xing; Pengwei Liu; Weihua Chen; Fan Wang; Bing Su
>
> **摘要:** Long video generation has gained increasing attention due to its widespread applications in fields such as entertainment and simulation. Despite advances, synthesizing temporally coherent and visually compelling long sequences remains a formidable challenge. Conventional approaches often synthesize long videos by sequentially generating and concatenating short clips, or generating key frames and then interpolate the intermediate frames in a hierarchical manner. However, both of them still remain significant challenges, leading to issues such as temporal repetition or unnatural transitions. In this paper, we revisit the hierarchical long video generation pipeline and introduce LumosFlow, a framework introduce motion guidance explicitly. Specifically, we first employ the Large Motion Text-to-Video Diffusion Model (LMTV-DM) to generate key frames with larger motion intervals, thereby ensuring content diversity in the generated long videos. Given the complexity of interpolating contextual transitions between key frames, we further decompose the intermediate frame interpolation into motion generation and post-hoc refinement. For each pair of key frames, the Latent Optical Flow Diffusion Model (LOF-DM) synthesizes complex and large-motion optical flows, while MotionControlNet subsequently refines the warped results to enhance quality and guide intermediate frame generation. Compared with traditional video frame interpolation, we achieve 15x interpolation, ensuring reasonable and continuous motion between adjacent frames. Experiments show that our method can generate long videos with consistent motion and appearance. Code and models will be made publicly available upon acceptance. Our project page: https://jiahaochen1.github.io/LumosFlow/
>
---
#### [new 037] Enhancing Abnormality Identification: Robust Out-of-Distribution Strategies for Deepfake Detection
- **分类: cs.CV**

- **简介: 该论文属于深度伪造检测任务，旨在解决在开放场景中识别未见过的生成模型所产生数据的问题。作者提出了两种新的分布外检测方法：一种基于图像重建，另一种结合注意力机制。实验表明，这些方法在深度伪造检测中表现优异，具有较强的泛化能力与应用潜力。**

- **链接: [http://arxiv.org/pdf/2506.02857v1](http://arxiv.org/pdf/2506.02857v1)**

> **作者:** Luca Maiano; Fabrizio Casadei; Irene Amerini
>
> **摘要:** Detecting deepfakes has become a critical challenge in Computer Vision and Artificial Intelligence. Despite significant progress in detection techniques, generalizing them to open-set scenarios continues to be a persistent difficulty. Neural networks are often trained on the closed-world assumption, but with new generative models constantly evolving, it is inevitable to encounter data generated by models that are not part of the training distribution. To address these challenges, in this paper, we propose two novel Out-Of-Distribution (OOD) detection approaches. The first approach is trained to reconstruct the input image, while the second incorporates an attention mechanism for detecting OODs. Our experiments validate the effectiveness of the proposed approaches compared to existing state-of-the-art techniques. Our method achieves promising results in deepfake detection and ranks among the top-performing configurations on the benchmark, demonstrating their potential for robust, adaptable solutions in dynamic, real-world applications.
>
---
#### [new 038] QARI-OCR: High-Fidelity Arabic Text Recognition through Multimodal Large Language Model Adaptation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于光学字符识别（OCR）任务，旨在解决阿拉伯文因连写、变体和音符标记导致的识别难题。作者基于Qwen2-VL-2B-Instruct模型开发了Qari-OCR，通过合成数据迭代微调，实现高精度阿拉伯文识别，并发布多个版本提升识别效果与适用性。**

- **链接: [http://arxiv.org/pdf/2506.02295v1](http://arxiv.org/pdf/2506.02295v1)**

> **作者:** Ahmed Wasfy; Omer Nacar; Abdelakreem Elkhateb; Mahmoud Reda; Omar Elshehy; Adel Ammar; Wadii Boulila
>
> **摘要:** The inherent complexities of Arabic script; its cursive nature, diacritical marks (tashkeel), and varied typography, pose persistent challenges for Optical Character Recognition (OCR). We present Qari-OCR, a series of vision-language models derived from Qwen2-VL-2B-Instruct, progressively optimized for Arabic through iterative fine-tuning on specialized synthetic datasets. Our leading model, QARI v0.2, establishes a new open-source state-of-the-art with a Word Error Rate (WER) of 0.160, Character Error Rate (CER) of 0.061, and BLEU score of 0.737 on diacritically-rich texts. Qari-OCR demonstrates superior handling of tashkeel, diverse fonts, and document layouts, alongside impressive performance on low-resolution images. Further explorations (QARI v0.3) showcase strong potential for structural document understanding and handwritten text. This work delivers a marked improvement in Arabic OCR accuracy and efficiency, with all models and datasets released to foster further research.
>
---
#### [new 039] Controllable Human-centric Keyframe Interpolation with Generative Prior
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决复杂人体动作关键帧间插值问题。现有方法缺乏3D几何引导，难以生成合理中间帧且控制性差。作者提出PoseFuse3D-KI，结合3D人体姿态与2D编码，提升插值质量，并构建CHKI-Video数据集评估性能。**

- **链接: [http://arxiv.org/pdf/2506.03119v1](http://arxiv.org/pdf/2506.03119v1)**

> **作者:** Zujin Guo; Size Wu; Zhongang Cai; Wei Li; Chen Change Loy
>
> **备注:** Project Page: https://gseancdat.github.io/projects/PoseFuse3D_KI
>
> **摘要:** Existing interpolation methods use pre-trained video diffusion priors to generate intermediate frames between sparsely sampled keyframes. In the absence of 3D geometric guidance, these methods struggle to produce plausible results for complex, articulated human motions and offer limited control over the synthesized dynamics. In this paper, we introduce PoseFuse3D Keyframe Interpolator (PoseFuse3D-KI), a novel framework that integrates 3D human guidance signals into the diffusion process for Controllable Human-centric Keyframe Interpolation (CHKI). To provide rich spatial and structural cues for interpolation, our PoseFuse3D, a 3D-informed control model, features a novel SMPL-X encoder that transforms 3D geometry and shape into the 2D latent conditioning space, alongside a fusion network that integrates these 3D cues with 2D pose embeddings. For evaluation, we build CHKI-Video, a new dataset annotated with both 2D poses and 3D SMPL-X parameters. We show that PoseFuse3D-KI consistently outperforms state-of-the-art baselines on CHKI-Video, achieving a 9% improvement in PSNR and a 38% reduction in LPIPS. Comprehensive ablations demonstrate that our PoseFuse3D model improves interpolation fidelity.
>
---
#### [new 040] Go Beyond Earth: Understanding Human Actions and Scenes in Microgravity Environments
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决微重力环境下人类动作和场景理解的缺失问题。作者构建了首个微重力基准数据集MicroG-4M，包含近5,000个视频片段及多种标注，支持动作识别、视频描述生成和视觉问答三项任务，推动空间应用中鲁棒性视频理解的发展。**

- **链接: [http://arxiv.org/pdf/2506.02845v1](http://arxiv.org/pdf/2506.02845v1)**

> **作者:** Di Wen; Lei Qi; Kunyu Peng; Kailun Yang; Fei Teng; Ao Luo; Jia Fu; Yufan Chen; Ruiping Liu; Yitian Shi; M. Saquib Sarfraz; Rainer Stiefelhagen
>
> **备注:** 15 pages, 3 figures, submitted to NeurIPS 2025
>
> **摘要:** Despite substantial progress in video understanding, most existing datasets are limited to Earth's gravitational conditions. However, microgravity alters human motion, interactions, and visual semantics, revealing a critical gap for real-world vision systems. This presents a challenge for domain-robust video understanding in safety-critical space applications. To address this, we introduce MicroG-4M, the first benchmark for spatio-temporal and semantic understanding of human activities in microgravity. Constructed from real-world space missions and cinematic simulations, the dataset includes 4,759 clips covering 50 actions, 1,238 context-rich captions, and over 7,000 question-answer pairs on astronaut activities and scene understanding. MicroG-4M supports three core tasks: fine-grained multi-label action recognition, temporal video captioning, and visual question answering, enabling a comprehensive evaluation of both spatial localization and semantic reasoning in microgravity contexts. We establish baselines using state-of-the-art models. All data, annotations, and code are available at https://github.com/LEI-QI-233/HAR-in-Space.
>
---
#### [new 041] Fairness through Feedback: Addressing Algorithmic Misgendering in Automatic Gender Recognition
- **分类: cs.CV**

- **简介: 这篇论文属于机器学习公平性任务，旨在解决自动性别识别（AGR）系统对非二元性别群体的误判问题。作者提出应区分生理性别、社会性别与性别表达，并主张通过反馈机制让用户纠正系统输出，以提升公平性。**

- **链接: [http://arxiv.org/pdf/2506.02017v1](http://arxiv.org/pdf/2506.02017v1)**

> **作者:** Camilla Quaresmini; Giacomo Zanotti
>
> **摘要:** Automatic Gender Recognition (AGR) systems are an increasingly widespread application in the Machine Learning (ML) landscape. While these systems are typically understood as detecting gender, they often classify datapoints based on observable features correlated at best with either male or female sex. In addition to questionable binary assumptions, from an epistemological point of view, this is problematic for two reasons. First, there exists a gap between the categories the system is meant to predict (woman versus man) and those onto which their output reasonably maps (female versus male). What is more, gender cannot be inferred on the basis of such observable features. This makes AGR tools often unreliable, especially in the case of non-binary and gender non-conforming people. We suggest a theoretical and practical rethinking of AGR systems. To begin, distinctions are made between sex, gender, and gender expression. Then, we build upon the observation that, unlike algorithmic misgendering, human-human misgendering is open to the possibility of re-evaluation and correction. We suggest that analogous dynamics should be recreated in AGR, giving users the possibility to correct the system's output. While implementing such a feedback mechanism could be regarded as diminishing the system's autonomy, it represents a way to significantly increase fairness levels in AGR. This is consistent with the conceptual change of paradigm that we advocate for AGR systems, which should be understood as tools respecting individuals' rights and capabilities of self-expression and determination.
>
---
#### [new 042] Approximate Borderline Sampling using Granular-Ball for Classification Tasks
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于分类任务，旨在解决数据采样中的边界模糊和噪声问题。通过提出基于粒球的限制扩散生成方法（RD-GBG）与近似边界采样方法（GBABS），实现更精确的边界表示与噪声处理，提升分类性能。**

- **链接: [http://arxiv.org/pdf/2506.02366v1](http://arxiv.org/pdf/2506.02366v1)**

> **作者:** Qin Xie; Qinghua Zhang; Shuyin Xia
>
> **摘要:** Data sampling enhances classifier efficiency and robustness through data compression and quality improvement. Recently, the sampling method based on granular-ball (GB) has shown promising performance in generality and noisy classification tasks. However, some limitations remain, including the absence of borderline sampling strategies and issues with class boundary blurring or shrinking due to overlap between GBs. In this paper, an approximate borderline sampling method using GBs is proposed for classification tasks. First, a restricted diffusion-based GB generation (RD-GBG) method is proposed, which prevents GB overlaps by constrained expansion, preserving precise geometric representation of GBs via redefined ones. Second, based on the concept of heterogeneous nearest neighbor, a GB-based approximate borderline sampling (GBABS) method is proposed, which is the first general sampling method capable of both borderline sampling and improving the quality of class noise datasets. Additionally, since RD-GBG incorporates noise detection and GBABS focuses on borderline samples, GBABS performs outstandingly on class noise datasets without the need for an optimal purity threshold. Experimental results demonstrate that the proposed methods outperform the GB-based sampling method and several representative sampling methods. Our source code is publicly available at https://github.com/CherylTse/GBABS.
>
---
#### [new 043] Solving Inverse Problems with FLAIR
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于图像逆问题任务，旨在利用流式生成模型作为先验信息解决图像重建问题。主要解决现有方法在恢复罕见模式、数据一致性及正则化调节方面的不足。提出了FLAIR框架，结合变分目标、轨迹调整和时变校准技术，提升了重建质量和样本多样性。**

- **链接: [http://arxiv.org/pdf/2506.02680v1](http://arxiv.org/pdf/2506.02680v1)**

> **作者:** Julius Erbach; Dominik Narnhofer; Andreas Dombos; Bernt Schiele; Jan Eric Lenssen; Konrad Schindler
>
> **摘要:** Flow-based latent generative models such as Stable Diffusion 3 are able to generate images with remarkable quality, even enabling photorealistic text-to-image generation. Their impressive performance suggests that these models should also constitute powerful priors for inverse imaging problems, but that approach has not yet led to comparable fidelity. There are several key obstacles: (i) the encoding into a lower-dimensional latent space makes the underlying (forward) mapping non-linear; (ii) the data likelihood term is usually intractable; and (iii) learned generative models struggle to recover rare, atypical data modes during inference. We present FLAIR, a novel training free variational framework that leverages flow-based generative models as a prior for inverse problems. To that end, we introduce a variational objective for flow matching that is agnostic to the type of degradation, and combine it with deterministic trajectory adjustments to recover atypical modes. To enforce exact consistency with the observed data, we decouple the optimization of the data fidelity and regularization terms. Moreover, we introduce a time-dependent calibration scheme in which the strength of the regularization is modulated according to off-line accuracy estimates. Results on standard imaging benchmarks demonstrate that FLAIR consistently outperforms existing diffusion- and flow-based methods in terms of reconstruction quality and sample diversity.
>
---
#### [new 044] Entity Image and Mixed-Modal Image Retrieval Datasets
- **分类: cs.CV; cs.IR**

- **简介: 该论文属于多模态图像检索任务，旨在解决现有数据集缺乏对图文混合模态检索的深入跨模态理解评估的问题。论文构建了两个新数据集：实体图像数据集（EI）和混合模态图像检索数据集（MMIR），并设计了单实体与多实体查询类型，以评估模型在结合视觉与文本信息下的检索能力。**

- **链接: [http://arxiv.org/pdf/2506.02291v1](http://arxiv.org/pdf/2506.02291v1)**

> **作者:** Cristian-Ioan Blaga; Paul Suganthan; Sahil Dua; Krishna Srinivasan; Enrique Alfonseca; Peter Dornbach; Tom Duerig; Imed Zitouni; Zhe Dong
>
> **摘要:** Despite advances in multimodal learning, challenging benchmarks for mixed-modal image retrieval that combines visual and textual information are lacking. This paper introduces a novel benchmark to rigorously evaluate image retrieval that demands deep cross-modal contextual understanding. We present two new datasets: the Entity Image Dataset (EI), providing canonical images for Wikipedia entities, and the Mixed-Modal Image Retrieval Dataset (MMIR), derived from the WIT dataset. The MMIR benchmark features two challenging query types requiring models to ground textual descriptions in the context of provided visual entities: single entity-image queries (one entity image with descriptive text) and multi-entity-image queries (multiple entity images with relational text). We empirically validate the benchmark's utility as both a training corpus and an evaluation set for mixed-modal retrieval. The quality of both datasets is further affirmed through crowd-sourced human annotations. The datasets are accessible through the GitHub page: https://github.com/google-research-datasets/wit-retrieval.
>
---
#### [new 045] Do You See Me : A Multidimensional Benchmark for Evaluating Visual Perception in Multimodal LLMs
- **分类: cs.CV**

- **简介: 该论文属于视觉感知评估任务，旨在解决多模态大语言模型（MLLM）在视觉理解上的瓶颈问题。作者构建了一个包含1,758张图像和2,612个问题的基准测试“Do You See Me”，涵盖7个子任务，评估MLLM的视觉感知能力，发现当前模型表现远低于人类，尤其在复杂任务中差距更大，并分析了失败原因。**

- **链接: [http://arxiv.org/pdf/2506.02022v1](http://arxiv.org/pdf/2506.02022v1)**

> **作者:** Aditya Kanade; Tanuja Ganu
>
> **摘要:** Multimodal Large Language Models (MLLMs) show reasoning promise, yet their visual perception is a critical bottleneck. Strikingly, MLLMs can produce correct answers even while misinterpreting crucial visual elements, masking these underlying failures. Our preliminary study on a joint perception-reasoning dataset revealed that for one leading MLLM, 29% of its correct answers to reasoning questions still exhibited visual perception errors. To systematically address this, we introduce "Do You See Me", a scalable benchmark with 1,758 images and 2,612 questions. It spans seven human-psychology inspired subtasks in 2D and 3D, featuring controllable complexity to rigorously evaluate MLLM visual skills. Our findings on 3 leading closed-source and 5 major open-source models reveal a stark deficit: humans achieve 96.49% accuracy, while top MLLMs average below 50%. This performance gap widens rapidly with increased task complexity (e.g., from 12% to 45% in the visual form constancy subtask). Further analysis into the root causes suggests that failures stem from challenges like misallocated visual attention and the instability of internal representations for fine-grained details, especially at or below encoder patch resolution. This underscores an urgent need for MLLMs with truly robust visual perception. The benchmark dataset, source code and evaluation scripts are available at https://github.com/microsoft/Do-You-See-Me.
>
---
#### [new 046] DCI: Dual-Conditional Inversion for Boosting Diffusion-Based Image Editing
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，旨在解决扩散模型中重构精度与编辑灵活性的权衡问题。作者提出Dual-Conditional Inversion（DCI），通过联合条件优化，同时最小化潜在噪声差距和重构误差，提升了图像重构质量和编辑准确性。**

- **链接: [http://arxiv.org/pdf/2506.02560v1](http://arxiv.org/pdf/2506.02560v1)**

> **作者:** Zixiang Li; Haoyu Wang; Wei Wang; Chuangchuang Tan; Yunchao Wei; Yao Zhao
>
> **摘要:** Diffusion models have achieved remarkable success in image generation and editing tasks. Inversion within these models aims to recover the latent noise representation for a real or generated image, enabling reconstruction, editing, and other downstream tasks. However, to date, most inversion approaches suffer from an intrinsic trade-off between reconstruction accuracy and editing flexibility. This limitation arises from the difficulty of maintaining both semantic alignment and structural consistency during the inversion process. In this work, we introduce Dual-Conditional Inversion (DCI), a novel framework that jointly conditions on the source prompt and reference image to guide the inversion process. Specifically, DCI formulates the inversion process as a dual-condition fixed-point optimization problem, minimizing both the latent noise gap and the reconstruction error under the joint guidance. This design anchors the inversion trajectory in both semantic and visual space, leading to more accurate and editable latent representations. Our novel setup brings new understanding to the inversion process. Extensive experiments demonstrate that DCI achieves state-of-the-art performance across multiple editing tasks, significantly improving both reconstruction quality and editing precision. Furthermore, we also demonstrate that our method achieves strong results in reconstruction tasks, implying a degree of robustness and generalizability approaching the ultimate goal of the inversion process.
>
---
#### [new 047] InterRVOS: Interaction-aware Referring Video Object Segmentation
- **分类: cs.CV**

- **简介: 该论文属于视频目标分割任务，旨在解决通过自然语言表达定位视频中交互对象的问题。作者提出了新任务InterRVOS，强调分割交互中的“动作发起者”和“目标”对象，并构建了大规模数据集InterRVOS-8K及基准模型ReVIOSa，提升了对复杂交互关系的建模能力。**

- **链接: [http://arxiv.org/pdf/2506.02356v1](http://arxiv.org/pdf/2506.02356v1)**

> **作者:** Woojeong Jin; Seongchan Kim; Seungryong Kim
>
> **摘要:** Referring video object segmentation aims to segment the object in a video corresponding to a given natural language expression. While prior works have explored various referring scenarios, including motion-centric or multi-instance expressions, most approaches still focus on localizing a single target object in isolation. However, in comprehensive video understanding, an object's role is often defined by its interactions with other entities, which are largely overlooked in existing datasets and models. In this work, we introduce Interaction-aware referring video object sgementation (InterRVOS), a new task that requires segmenting both actor and target entities involved in an interaction. Each interactoin is described through a pair of complementary expressions from different semantic perspectives, enabling fine-grained modeling of inter-object relationships. To tackle this task, we propose InterRVOS-8K, the large-scale and automatically constructed dataset containing diverse interaction-aware expressions with corresponding masks, including challenging cases such as motion-only multi-instance expressions. We also present a baseline architecture, ReVIOSa, designed to handle actor-target segmentation from a single expression, achieving strong performance in both standard and interaction-focused settings. Furthermore, we introduce an actor-target-aware evalaution setting that enables a more targeted assessment of interaction understanding. Experimental results demonstrate that our approach outperforms prior methods in modeling complex object interactions for referring video object segmentation task, establishing a strong foundation for future research in interaction-centric video understanding. Our project page is available at \href{https://cvlab-kaist.github.io/InterRVOS}{https://cvlab-kaist.github.io/InterRVOS}.
>
---
#### [new 048] VidEvent: A Large Dataset for Understanding Dynamic Evolution of Events in Videos
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频事件理解任务，旨在解决视频中复杂事件结构建模与预测的问题。作者构建了大规模数据集VidEvent，包含23,000多个标注事件，并提供基线模型用于事件脚本提取与预测，推动相关算法研究。**

- **链接: [http://arxiv.org/pdf/2506.02448v1](http://arxiv.org/pdf/2506.02448v1)**

> **作者:** Baoyu Liang; Qile Su; Shoutai Zhu; Yuchen Liang; Chao Tong
>
> **摘要:** Despite the significant impact of visual events on human cognition, understanding events in videos remains a challenging task for AI due to their complex structures, semantic hierarchies, and dynamic evolution. To address this, we propose the task of video event understanding that extracts event scripts and makes predictions with these scripts from videos. To support this task, we introduce VidEvent, a large-scale dataset containing over 23,000 well-labeled events, featuring detailed event structures, broad hierarchies, and logical relations extracted from movie recap videos. The dataset was created through a meticulous annotation process, ensuring high-quality and reliable event data. We also provide comprehensive baseline models offering detailed descriptions of their architecture and performance metrics. These models serve as benchmarks for future research, facilitating comparisons and improvements. Our analysis of VidEvent and the baseline models highlights the dataset's potential to advance video event understanding and encourages the exploration of innovative algorithms and models. The dataset and related resources are publicly available at www.videvent.top.
>
---
#### [new 049] MIND: Material Interface Generation from UDFs for Non-Manifold Surface Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D形状重建任务，旨在解决从无符号距离场（UDF）中提取非流形网格的难题。现有方法依赖局部符号距离场（SDF），易引入拓扑错误且无法处理非流形结构。论文提出MIND方法，通过构建多标签全局场来区分不同区域，结合UDF生成材料界面，实现非流形网格提取，显著优于现有技术。**

- **链接: [http://arxiv.org/pdf/2506.02938v1](http://arxiv.org/pdf/2506.02938v1)**

> **作者:** Xuhui Chen; Fei Hou; Wencheng Wang; Hong Qin; Ying He
>
> **摘要:** Unsigned distance fields (UDFs) are widely used in 3D deep learning due to their ability to represent shapes with arbitrary topology. While prior work has largely focused on learning UDFs from point clouds or multi-view images, extracting meshes from UDFs remains challenging, as the learned fields rarely attain exact zero distances. A common workaround is to reconstruct signed distance fields (SDFs) locally from UDFs to enable surface extraction via Marching Cubes. However, this often introduces topological artifacts such as holes or spurious components. Moreover, local SDFs are inherently incapable of representing non-manifold geometry, leading to complete failure in such cases. To address this gap, we propose MIND (Material Interface from Non-manifold Distance fields), a novel algorithm for generating material interfaces directly from UDFs, enabling non-manifold mesh extraction from a global perspective. The core of our method lies in deriving a meaningful spatial partitioning from the UDF, where the target surface emerges as the interface between distinct regions. We begin by computing a two-signed local field to distinguish the two sides of manifold patches, and then extend this to a multi-labeled global field capable of separating all sides of a non-manifold structure. By combining this multi-labeled field with the input UDF, we construct material interfaces that support non-manifold mesh extraction via a multi-labeled Marching Cubes algorithm. Extensive experiments on UDFs generated from diverse data sources, including point cloud reconstruction, multi-view reconstruction, and medial axis transforms, demonstrate that our approach robustly handles complex non-manifold surfaces and significantly outperforms existing methods.
>
---
#### [new 050] Co-Evidential Fusion with Information Volume for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决半监督学习中多源不确定性难以有效利用的问题。作者提出了基于广义证据深度学习的协同证据融合策略和信息量度量方法，优化模型对标注与未标注数据的学习，提升分割性能。**

- **链接: [http://arxiv.org/pdf/2506.02492v1](http://arxiv.org/pdf/2506.02492v1)**

> **作者:** Yuanpeng He; Lijian Li; Tianxiang Zhan; Chi-Man Pun; Wenpin Jiao; Zhi Jin
>
> **摘要:** Although existing semi-supervised image segmentation methods have achieved good performance, they cannot effectively utilize multiple sources of voxel-level uncertainty for targeted learning. Therefore, we propose two main improvements. First, we introduce a novel pignistic co-evidential fusion strategy using generalized evidential deep learning, extended by traditional D-S evidence theory, to obtain a more precise uncertainty measure for each voxel in medical samples. This assists the model in learning mixed labeled information and establishing semantic associations between labeled and unlabeled data. Second, we introduce the concept of information volume of mass function (IVUM) to evaluate the constructed evidence, implementing two evidential learning schemes. One optimizes evidential deep learning by combining the information volume of the mass function with original uncertainty measures. The other integrates the learning pattern based on the co-evidential fusion strategy, using IVUM to design a new optimization objective. Experiments on four datasets demonstrate the competitive performance of our method.
>
---
#### [new 051] Flexiffusion: Training-Free Segment-Wise Neural Architecture Search for Efficient Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于神经架构搜索（NAS）任务，旨在解决扩散模型（DMs）推理速度慢、计算成本高的问题。作者提出Flexiffusion，一种无需训练的段落式NAS框架，通过优化生成流程中的计算步骤组合，并引入轻量级评估指标rFID，实现高效模型加速，在多个数据集上取得显著加速效果且质量损失小。**

- **链接: [http://arxiv.org/pdf/2506.02488v1](http://arxiv.org/pdf/2506.02488v1)**

> **作者:** Hongtao Huang; Xiaojun Chang; Lina Yao
>
> **摘要:** Diffusion models (DMs) are powerful generative models capable of producing high-fidelity images but are constrained by high computational costs due to iterative multi-step inference. While Neural Architecture Search (NAS) can optimize DMs, existing methods are hindered by retraining requirements, exponential search complexity from step-wise optimization, and slow evaluation relying on massive image generation. To address these challenges, we propose Flexiffusion, a training-free NAS framework that jointly optimizes generation schedules and model architectures without modifying pre-trained parameters. Our key insight is to decompose the generation process into flexible segments of equal length, where each segment dynamically combines three step types: full (complete computation), partial (cache-reused computation), and null (skipped computation). This segment-wise search space reduces the candidate pool exponentially compared to step-wise NAS while preserving architectural diversity. Further, we introduce relative FID (rFID), a lightweight evaluation metric for NAS that measures divergence from a teacher model's outputs instead of ground truth, slashing evaluation time by over $90\%$. In practice, Flexiffusion achieves at least $2\times$ acceleration across LDMs, Stable Diffusion, and DDPMs on ImageNet and MS-COCO, with FID degradation under $5\%$, outperforming prior NAS and caching methods. Notably, it attains $5.1\times$ speedup on Stable Diffusion with near-identical CLIP scores. Our work pioneers a resource-efficient paradigm for searching high-speed DMs without sacrificing quality.
>
---
#### [new 052] Deep Learning for Retinal Degeneration Assessment: A Comprehensive Analysis of the MARIO AMD Progression Challenge
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在利用深度学习评估视网膜变性疾病（AMD）进展。论文围绕MARIO挑战展开，主要解决基于OCT图像的AMD活动变化检测与未来演变预测问题。工作包括构建多模态数据集、设计两项挑战任务，并评估各团队算法性能，最终确立AI在AMD监测中的基准表现。**

- **链接: [http://arxiv.org/pdf/2506.02976v1](http://arxiv.org/pdf/2506.02976v1)**

> **作者:** Rachid Zeghlache; Ikram Brahim; Pierre-Henri Conze; Mathieu Lamard; Mohammed El Amine Lazouni; Zineb Aziza Elaouaber; Leila Ryma Lazouni; Christopher Nielsen; Ahmad O. Ahsan; Matthias Wilms; Nils D. Forkert; Lovre Antonio Budimir; Ivana Matovinović; Donik Vršnak; Sven Lončarić; Philippe Zhang; Weili Jiang; Yihao Li; Yiding Hao; Markus Frohmann; Patrick Binder; Marcel Huber; Taha Emre; Teresa Finisterra Araújo; Marzieh Oghbaie; Hrvoje Bogunović; Amerens A. Bekkers; Nina M. van Liebergen; Hugo J. Kuijf; Abdul Qayyum; Moona Mazher; Steven A. Niederer; Alberto J. Beltrán-Carrero; Juan J. Gómez-Valverde; Javier Torresano-Rodríquez; Álvaro Caballero-Sastre; María J. Ledesma Carbayo; Yosuke Yamagishi; Yi Ding; Robin Peretzke; Alexandra Ertl; Maximilian Fischer; Jessica Kächele; Sofiane Zehar; Karim Boukli Hacene; Thomas Monfort; Béatrice Cochener; Mostafa El Habib Daho; Anas-Alexis Benyoussef; Gwenolé Quellec
>
> **备注:** MARIO-MICCAI-CHALLENGE 2024
>
> **摘要:** The MARIO challenge, held at MICCAI 2024, focused on advancing the automated detection and monitoring of age-related macular degeneration (AMD) through the analysis of optical coherence tomography (OCT) images. Designed to evaluate algorithmic performance in detecting neovascular activity changes within AMD, the challenge incorporated unique multi-modal datasets. The primary dataset, sourced from Brest, France, was used by participating teams to train and test their models. The final ranking was determined based on performance on this dataset. An auxiliary dataset from Algeria was used post-challenge to evaluate population and device shifts from submitted solutions. Two tasks were involved in the MARIO challenge. The first one was the classification of evolution between two consecutive 2D OCT B-scans. The second one was the prediction of future AMD evolution over three months for patients undergoing anti-vascular endothelial growth factor (VEGF) therapy. Thirty-five teams participated, with the top 12 finalists presenting their methods. This paper outlines the challenge's structure, tasks, data characteristics, and winning methodologies, setting a benchmark for AMD monitoring using OCT, infrared imaging, and clinical data (such as the number of visits, age, gender, etc.). The results of this challenge indicate that artificial intelligence (AI) performs as well as a physician in measuring AMD progression (Task 1) but is not yet able of predicting future evolution (Task 2).
>
---
#### [new 053] SurgVLM: A Large Vision-Language Model and Systematic Evaluation Benchmark for Surgical Intelligence
- **分类: cs.CV; 68T45; I.2.10**

- **简介: 该论文属于医疗视觉-语言模型任务，旨在解决手术智能中缺乏高质量多模态数据与专用模型的问题。作者构建了大规模手术数据库SurgVLM-DB，并基于此提出SurgVLM模型及评估基准SurgVLM-Bench，用于多手术任务分析。**

- **链接: [http://arxiv.org/pdf/2506.02555v1](http://arxiv.org/pdf/2506.02555v1)**

> **作者:** Zhitao Zeng; Zhu Zhuo; Xiaojun Jia; Erli Zhang; Junde Wu; Jiaan Zhang; Yuxuan Wang; Chang Han Low; Jian Jiang; Zilong Zheng; Xiaochun Cao; Yutong Ban; Qi Dou; Yang Liu; Yueming Jin
>
> **备注:** 29 pages, 5 figures
>
> **摘要:** Foundation models have achieved transformative success across biomedical domains by enabling holistic understanding of multimodal data. However, their application in surgery remains underexplored. Surgical intelligence presents unique challenges - requiring surgical visual perception, temporal analysis, and reasoning. Existing general-purpose vision-language models fail to address these needs due to insufficient domain-specific supervision and the lack of a large-scale high-quality surgical database. To bridge this gap, we propose SurgVLM, one of the first large vision-language foundation models for surgical intelligence, where this single universal model can tackle versatile surgical tasks. To enable this, we construct a large-scale multimodal surgical database, SurgVLM-DB, comprising over 1.81 million frames with 7.79 million conversations, spanning more than 16 surgical types and 18 anatomical structures. We unify and reorganize 23 public datasets across 10 surgical tasks, followed by standardizing labels and doing hierarchical vision-language alignment to facilitate comprehensive coverage of gradually finer-grained surgical tasks, from visual perception, temporal analysis, to high-level reasoning. Building upon this comprehensive dataset, we propose SurgVLM, which is built upon Qwen2.5-VL, and undergoes instruction tuning to 10+ surgical tasks. We further construct a surgical multimodal benchmark, SurgVLM-Bench, for method evaluation. SurgVLM-Bench consists of 6 popular and widely-used datasets in surgical domain, covering several crucial downstream tasks. Based on SurgVLM-Bench, we evaluate the performance of our SurgVLM (3 SurgVLM variants: SurgVLM-7B, SurgVLM-32B, and SurgVLM-72B), and conduct comprehensive comparisons with 14 mainstream commercial VLMs (e.g., GPT-4o, Gemini 2.0 Flash, Qwen2.5-Max).
>
---
#### [new 054] AnimeShooter: A Multi-Shot Animation Dataset for Reference-Guided Video Generation
- **分类: cs.CV**

- **简介: 该论文属于参考引导的多镜头动画生成任务，旨在解决现有数据集中缺乏角色参考与叙事连贯性的问题。作者构建了AnimeShooter数据集，包含多层次标注和同步音频，并提出AnimeShooterGen模型，结合多模态大语言模型与扩散模型，实现视觉一致且符合参考的动画视频生成。**

- **链接: [http://arxiv.org/pdf/2506.03126v1](http://arxiv.org/pdf/2506.03126v1)**

> **作者:** Lu Qiu; Yizhuo Li; Yuying Ge; Yixiao Ge; Ying Shan; Xihui Liu
>
> **备注:** Project released at: https://qiulu66.github.io/animeshooter/
>
> **摘要:** Recent advances in AI-generated content (AIGC) have significantly accelerated animation production. To produce engaging animations, it is essential to generate coherent multi-shot video clips with narrative scripts and character references. However, existing public datasets primarily focus on real-world scenarios with global descriptions, and lack reference images for consistent character guidance. To bridge this gap, we present AnimeShooter, a reference-guided multi-shot animation dataset. AnimeShooter features comprehensive hierarchical annotations and strong visual consistency across shots through an automated pipeline. Story-level annotations provide an overview of the narrative, including the storyline, key scenes, and main character profiles with reference images, while shot-level annotations decompose the story into consecutive shots, each annotated with scene, characters, and both narrative and descriptive visual captions. Additionally, a dedicated subset, AnimeShooter-audio, offers synchronized audio tracks for each shot, along with audio descriptions and sound sources. To demonstrate the effectiveness of AnimeShooter and establish a baseline for the reference-guided multi-shot video generation task, we introduce AnimeShooterGen, which leverages Multimodal Large Language Models (MLLMs) and video diffusion models. The reference image and previously generated shots are first processed by MLLM to produce representations aware of both reference and context, which are then used as the condition for the diffusion model to decode the subsequent shot. Experimental results show that the model trained on AnimeShooter achieves superior cross-shot visual consistency and adherence to reference visual guidance, which highlight the value of our dataset for coherent animated video generation.
>
---
#### [new 055] Empowering Functional Neuroimaging: A Pre-trained Generative Framework for Unified Representation of Neural Signals
- **分类: cs.CV**

- **简介: 该论文属于神经影像与脑机接口任务，旨在解决多模态功能神经影像获取成本高、可行性受限及数据代表性不足导致的模型不公平问题。作者提出一种基于生成式人工智能的统一表征框架，可生成缺失模态和少数群体的神经影像数据，降低采集成本并提升模型公平性。**

- **链接: [http://arxiv.org/pdf/2506.02433v1](http://arxiv.org/pdf/2506.02433v1)**

> **作者:** Weiheng Yao; Xuhang Chen; Shuqiang Wang
>
> **摘要:** Multimodal functional neuroimaging enables systematic analysis of brain mechanisms and provides discriminative representations for brain-computer interface (BCI) decoding. However, its acquisition is constrained by high costs and feasibility limitations. Moreover, underrepresentation of specific groups undermines fairness of BCI decoding model. To address these challenges, we propose a unified representation framework for multimodal functional neuroimaging via generative artificial intelligence (AI). By mapping multimodal functional neuroimaging into a unified representation space, the proposed framework is capable of generating data for acquisition-constrained modalities and underrepresented groups. Experiments show that the framework can generate data consistent with real brain activity patterns, provide insights into brain mechanisms, and improve performance on downstream tasks. More importantly, it can enhance model fairness by augmenting data for underrepresented groups. Overall, the framework offers a new paradigm for decreasing the cost of acquiring multimodal functional neuroimages and enhancing the fairness of BCI decoding models.
>
---
#### [new 056] ANT: Adaptive Neural Temporal-Aware Text-to-Motion Model
- **分类: cs.CV**

- **简介: 该论文属于文本到动作生成任务，旨在解决扩散模型在生成过程中语义条件静态导致的时序需求不匹配问题。作者提出了ANT框架，包含三个模块：语义自适应、动态无分类器引导调度和时序语义重加权，提升了生成效果与语义对齐能力。**

- **链接: [http://arxiv.org/pdf/2506.02452v1](http://arxiv.org/pdf/2506.02452v1)**

> **作者:** Wenshuo Chen; Kuimou Yu; Haozhe Jia; Kaishen Yuan; Bowen Tian; Songning Lai; Hongru Xiao; Erhang Zhang; Lei Wang; Yutao Yue
>
> **摘要:** While diffusion models advance text-to-motion generation, their static semantic conditioning ignores temporal-frequency demands: early denoising requires structural semantics for motion foundations while later stages need localized details for text alignment. This mismatch mirrors biological morphogenesis where developmental phases demand distinct genetic programs. Inspired by epigenetic regulation governing morphological specialization, we propose **(ANT)**, an **A**daptive **N**eural **T**emporal-Aware architecture. ANT orchestrates semantic granularity through: **(i) Semantic Temporally Adaptive (STA) Module:** Automatically partitions denoising into low-frequency structural planning and high-frequency refinement via spectral analysis. **(ii) Dynamic Classifier-Free Guidance scheduling (DCFG):** Adaptively adjusts conditional to unconditional ratio enhancing efficiency while maintaining fidelity. **(iii) Temporal-semantic reweighting:** Quantitatively aligns text influence with phase requirements. Extensive experiments show that ANT can be applied to various baselines, significantly improving model performance, and achieving state-of-the-art semantic alignment on StableMoFusion.
>
---
#### [new 057] High Performance Space Debris Tracking in Complex Skylight Backgrounds with a Large-Scale Dataset
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于空间碎片跟踪任务，旨在解决复杂天空背景下实时准确追踪空间碎片的问题。作者提出了一种基于深度学习的SDT-Net模型，并构建了大规模数据集SDTD用于训练与评估，最终在真实数据上验证了模型的有效性与迁移能力。**

- **链接: [http://arxiv.org/pdf/2506.02614v1](http://arxiv.org/pdf/2506.02614v1)**

> **作者:** Guohang Zhuang; Weixi Song; Jinyang Huang; Chenwei Yang; Yan Lu
>
> **摘要:** With the rapid development of space exploration, space debris has attracted more attention due to its potential extreme threat, leading to the need for real-time and accurate debris tracking. However, existing methods are mainly based on traditional signal processing, which cannot effectively process the complex background and dense space debris. In this paper, we propose a deep learning-based Space Debris Tracking Network~(SDT-Net) to achieve highly accurate debris tracking. SDT-Net effectively represents the feature of debris, enhancing the efficiency and stability of end-to-end model learning. To train and evaluate this model effectively, we also produce a large-scale dataset Space Debris Tracking Dataset (SDTD) by a novel observation-based data simulation scheme. SDTD contains 18,040 video sequences with a total of 62,562 frames and covers 250,000 synthetic space debris. Extensive experiments validate the effectiveness of our model and the challenging of our dataset. Furthermore, we test our model on real data from the Antarctic Station, achieving a MOTA score of 70.6%, which demonstrates its strong transferability to real-world scenarios. Our dataset and code will be released soon.
>
---
#### [new 058] Motion aware video generative model
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成任务，旨在解决生成视频中运动不真实的问题。通过分析物理运动在频域中的特征，提出物理运动损失函数和频域增强模块，提升视频的物理合理性和视觉质量，适用于多种视频扩散模型。**

- **链接: [http://arxiv.org/pdf/2506.02244v1](http://arxiv.org/pdf/2506.02244v1)**

> **作者:** Bowen Xue; Giuseppe Claudio Guarnera; Shuang Zhao; Zahra Montazeri
>
> **摘要:** Recent advances in diffusion-based video generation have yielded unprecedented quality in visual content and semantic coherence. However, current approaches predominantly rely on statistical learning from vast datasets without explicitly modeling the underlying physics of motion, resulting in subtle yet perceptible non-physical artifacts that diminish the realism of generated videos. This paper introduces a physics-informed frequency domain approach to enhance the physical plausibility of generated videos. We first conduct a systematic analysis of the frequency-domain characteristics of diverse physical motions (translation, rotation, scaling), revealing that each motion type exhibits distinctive and identifiable spectral signatures. Building on this theoretical foundation, we propose two complementary components: (1) a physical motion loss function that quantifies and optimizes the conformity of generated videos to ideal frequency-domain motion patterns, and (2) a frequency domain enhancement module that progressively learns to adjust video features to conform to physical motion constraints while preserving original network functionality through a zero-initialization strategy. Experiments across multiple video diffusion architectures demonstrate that our approach significantly enhances motion quality and physical plausibility without compromising visual quality or semantic alignment. Our frequency-domain physical motion framework generalizes effectively across different video generation architectures, offering a principled approach to incorporating physical constraints into deep learning-based video synthesis pipelines. This work seeks to establish connections between data-driven models and physics-based motion models.
>
---
#### [new 059] FreeScene: Mixed Graph Diffusion for 3D Scene Synthesis from Free Prompts
- **分类: cs.CV**

- **简介: 该论文属于3D室内场景生成任务，旨在解决现有方法在控制性与易用性间的权衡问题。通过FreeScene框架，结合文本/图像输入与图结构控制，提出MG-DiT模型实现高质量、多任务场景生成，提升可控性与生成效果。**

- **链接: [http://arxiv.org/pdf/2506.02781v1](http://arxiv.org/pdf/2506.02781v1)**

> **作者:** Tongyuan Bai; Wangyuanfan Bai; Dong Chen; Tieru Wu; Manyi Li; Rui Ma
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** Controllability plays a crucial role in the practical applications of 3D indoor scene synthesis. Existing works either allow rough language-based control, that is convenient but lacks fine-grained scene customization, or employ graph based control, which offers better controllability but demands considerable knowledge for the cumbersome graph design process. To address these challenges, we present FreeScene, a user-friendly framework that enables both convenient and effective control for indoor scene synthesis.Specifically, FreeScene supports free-form user inputs including text description and/or reference images, allowing users to express versatile design intentions. The user inputs are adequately analyzed and integrated into a graph representation by a VLM-based Graph Designer. We then propose MG-DiT, a Mixed Graph Diffusion Transformer, which performs graph-aware denoising to enhance scene generation. Our MG-DiT not only excels at preserving graph structure but also offers broad applicability to various tasks, including, but not limited to, text-to-scene, graph-to-scene, and rearrangement, all within a single model. Extensive experiments demonstrate that FreeScene provides an efficient and user-friendly solution that unifies text-based and graph based scene synthesis, outperforming state-of-the-art methods in terms of both generation quality and controllability in a range of applications.
>
---
#### [new 060] Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment
- **分类: cs.CV; cs.LG**

- **简介: 论文提出Diff2Flow框架，旨在将预训练扩散模型的知识高效迁移至流匹配（FM）模型。该工作属于生成模型领域，解决FM模型微调计算成本高、扩散模型与FM范式差异大导致的知识迁移困难问题。通过时间步重标度、插值对齐和速度场转换，实现高效FM微调。实验表明其在参数效率和多任务性能上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.02221v1](http://arxiv.org/pdf/2506.02221v1)**

> **作者:** Johannes Schusterbauer; Ming Gui; Frank Fundel; Björn Ommer
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** Diffusion models have revolutionized generative tasks through high-fidelity outputs, yet flow matching (FM) offers faster inference and empirical performance gains. However, current foundation FM models are computationally prohibitive for finetuning, while diffusion models like Stable Diffusion benefit from efficient architectures and ecosystem support. This work addresses the critical challenge of efficiently transferring knowledge from pre-trained diffusion models to flow matching. We propose Diff2Flow, a novel framework that systematically bridges diffusion and FM paradigms by rescaling timesteps, aligning interpolants, and deriving FM-compatible velocity fields from diffusion predictions. This alignment enables direct and efficient FM finetuning of diffusion priors with no extra computation overhead. Our experiments demonstrate that Diff2Flow outperforms na\"ive FM and diffusion finetuning particularly under parameter-efficient constraints, while achieving superior or competitive performance across diverse downstream tasks compared to state-of-the-art methods. We will release our code at https://github.com/CompVis/diff2flow.
>
---
#### [new 061] Generative Perception of Shape and Material from Differential Motion
- **分类: cs.CV**

- **简介: 该论文属于视觉感知任务，旨在解决单视角下物体形状与材质的模糊性问题。作者提出了一种基于微分运动的生成感知模型，通过短视频生成形状与材质图，利用连续运动信息提升视觉推理能力，取得了多模态预测和高质量估计的效果。**

- **链接: [http://arxiv.org/pdf/2506.02473v1](http://arxiv.org/pdf/2506.02473v1)**

> **作者:** Xinran Nicole Han; Ko Nishino; Todd Zickler
>
> **摘要:** Perceiving the shape and material of an object from a single image is inherently ambiguous, especially when lighting is unknown and unconstrained. Despite this, humans can often disentangle shape and material, and when they are uncertain, they often move their head slightly or rotate the object to help resolve the ambiguities. Inspired by this behavior, we introduce a novel conditional denoising-diffusion model that generates samples of shape-and-material maps from a short video of an object undergoing differential motions. Our parameter-efficient architecture allows training directly in pixel-space, and it generates many disentangled attributes of an object simultaneously. Trained on a modest number of synthetic object-motion videos with supervision on shape and material, the model exhibits compelling emergent behavior: For static observations, it produces diverse, multimodal predictions of plausible shape-and-material maps that capture the inherent ambiguities; and when objects move, the distributions quickly converge to more accurate explanations. The model also produces high-quality shape-and-material estimates for less ambiguous, real-world objects. By moving beyond single-view to continuous motion observations, our work suggests a generative perception approach for improving visual reasoning in physically-embodied systems.
>
---
#### [new 062] Small Aid, Big Leap: Efficient Test-Time Adaptation for Vision-Language Models with AdaptNet
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的测试时适应任务，旨在解决现有方法计算成本高、扩展性差的问题。作者提出SAIL框架，通过轻量级的AdaptNet实现高效模型适应，采用基于置信度的插值权重和梯度感知重置策略，减少计算开销并缓解灾难性遗忘，提升模型在实际部署中的效果与可扩展性。**

- **链接: [http://arxiv.org/pdf/2506.02671v1](http://arxiv.org/pdf/2506.02671v1)**

> **作者:** Xiao Chen; Jiazhen Huang; Qinting Jiang; Fanding Huang; Xianghua Fu; Jingyan Jiang; Zhi Wang
>
> **摘要:** Test-time adaptation (TTA) has emerged as a critical technique for enhancing the generalization capability of vision-language models (VLMs) during inference. However, existing approaches often incur substantial computational costs and exhibit poor scalability, primarily due to sample-wise adaptation granularity and reliance on costly auxiliary designs such as data augmentation. To address these limitations, we introduce SAIL (Small Aid, Big Leap), a novel adapter-based TTA framework that leverages a lightweight, learnable AdaptNet to enable efficient and scalable model adaptation. As SAIL's core, a frozen pre-trained VLM collaborates with AdaptNet through a confidence-based interpolation weight, generating robust predictions during inference. These predictions serve as self-supervised targets to align AdaptNet's outputs through efficient batch-wise processing, dramatically reducing computational costs without modifying the VLM or requiring memory caches. To mitigate catastrophic forgetting during continual adaptation, we propose a gradient-aware reset strategy driven by a gradient drift indicator (GDI), which dynamically detects domain transitions and strategically resets AdaptNet for stable adaptation. Extensive experiments across diverse benchmarks on two scenarios demonstrate that SAIL achieves state-of-the-art performance while maintaining low computational costs. These results highlight SAIL's effectiveness, efficiency and scalability for real-world deployment. The code will be released upon acceptance.
>
---
#### [new 063] PAID: Pairwise Angular-Invariant Decomposition for Continual Test-Time Adaptation
- **分类: cs.CV**

- **简介: 该论文属于持续测试时适应（CTTA）任务，旨在解决模型在推理过程中适应变化环境的问题。现有方法忽视了预训练权重中蕴含的领域不变先验信息。作者分析发现权重间的成对角度结构具有跨领域稳定性，据此提出PAID方法，在适应过程中保持这一结构，仅更新权重大小和方向的正交矩阵，从而提升模型适应能力。**

- **链接: [http://arxiv.org/pdf/2506.02453v1](http://arxiv.org/pdf/2506.02453v1)**

> **作者:** Kunyu Wang; Xueyang Fu; Yunfei Bao; Chengjie Ge; Chengzhi Cao; Wei Zhai; Zheng-Jun Zha
>
> **摘要:** Continual Test-Time Adaptation (CTTA) aims to online adapt a pre-trained model to changing environments during inference. Most existing methods focus on exploiting target data, while overlooking another crucial source of information, the pre-trained weights, which encode underutilized domain-invariant priors. This paper takes the geometric attributes of pre-trained weights as a starting point, systematically analyzing three key components: magnitude, absolute angle, and pairwise angular structure. We find that the pairwise angular structure remains stable across diverse corrupted domains and encodes domain-invariant semantic information, suggesting it should be preserved during adaptation. Based on this insight, we propose PAID (Pairwise Angular-Invariant Decomposition), a prior-driven CTTA method that decomposes weight into magnitude and direction, and introduces a learnable orthogonal matrix via Householder reflections to globally rotate direction while preserving the pairwise angular structure. During adaptation, only the magnitudes and the orthogonal matrices are updated. PAID achieves consistent improvements over recent SOTA methods on four widely used CTTA benchmarks, demonstrating that preserving pairwise angular structure offers a simple yet effective principle for CTTA.
>
---
#### [new 064] EgoVLM: Policy Optimization for Egocentric Video Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型任务，旨在解决第一视角视频理解中的时空推理问题。作者提出了EgoVLM模型，通过无监督强化学习方法GRPO进行优化，并引入关键帧奖励机制。模型在EgoSchema等基准上显著优于通用模型，增强了可解释性，适用于具身智能应用。**

- **链接: [http://arxiv.org/pdf/2506.03097v1](http://arxiv.org/pdf/2506.03097v1)**

> **作者:** Ashwin Vinod; Shrey Pandit; Aditya Vavre; Linshen Liu
>
> **备注:** Our Code can be found at https://github.com/adityavavre/VidEgoVLM
>
> **摘要:** Emerging embodied AI applications, such as wearable cameras and autonomous agents, have underscored the need for robust reasoning from first person video streams. We introduce EgoVLM, a vision-language model specifically designed to integrate visual comprehension and spatial-temporal reasoning within egocentric video contexts. EgoVLM is fine-tuned via Group Relative Policy Optimization (GRPO), a reinforcement learning method adapted to align model outputs with human-like reasoning steps. Following DeepSeek R1-Zero's approach, we directly tune using RL without any supervised fine-tuning phase on chain-of-thought (CoT) data. We evaluate EgoVLM on egocentric video question answering benchmarks and show that domain-specific training substantially improves performance over general-purpose VLMs. Our EgoVLM-3B, trained exclusively on non-CoT egocentric data, outperforms the base Qwen2.5-VL 3B and 7B models by 14.33 and 13.87 accuracy points on the EgoSchema benchmark, respectively. By explicitly generating reasoning traces, EgoVLM enhances interpretability, making it well-suited for downstream applications. Furthermore, we introduce a novel keyframe-based reward that incorporates salient frame selection to guide reinforcement learning optimization. This reward formulation opens a promising avenue for future exploration in temporally grounded egocentric reasoning.
>
---
#### [new 065] OASIS: Online Sample Selection for Continual Visual Instruction Tuning
- **分类: cs.CV**

- **简介: 该论文属于持续视觉指令微调（CVIT）任务，旨在解决在线流数据中训练延迟和分布变化导致的样本选择低效问题。作者提出了OASIS方法，通过动态调整每批样本数量并减少冗余，实现高效训练。实验表明其仅用25%数据即可达到全量训练效果，且性能优于现有技术。**

- **链接: [http://arxiv.org/pdf/2506.02011v1](http://arxiv.org/pdf/2506.02011v1)**

> **作者:** Minjae Lee; Minhyuk Seo; Tingyu Qu; Tinne Tuytelaars; Jonghyun Choi
>
> **摘要:** In continual visual instruction tuning (CVIT) scenarios, where multi-modal data continuously arrive in an online streaming manner, training delays from large-scale data significantly hinder real-time adaptation. While existing data selection strategies reduce training overheads, they rely on pre-trained reference models, which are impractical in CVIT setups due to unknown future data. Recent reference model-free online sample selection methods address this issue but typically select a fixed number of samples per batch (e.g., top-k), causing them to suffer from distribution shifts where informativeness varies across batches. To address these limitations, we propose OASIS, an adaptive online sample selection approach for CVIT that: (1) dynamically adjusts selected samples per batch based on relative inter-batch informativeness, and (2) minimizes redundancy of selected samples through iterative selection score updates. Empirical results across various MLLMs, such as LLaVA-1.5 and Qwen-VL-2.5, show that OASIS achieves comparable performance to full-data training using only 25% of the data and outperforms the state-of-the-art.
>
---
#### [new 066] RRCANet: Recurrent Reusable-Convolution Attention Network for Infrared Small Target Detection
- **分类: cs.CV**

- **简介: 该论文属于红外小目标检测任务，旨在解决小目标因尺寸小、亮度低等特点导致的检测难题。作者提出RRCA-Net网络，通过循环可重用卷积块和双交互注意力模块，提升特征提取与上下文信息融合能力，并设计了DpT-k损失函数以优化训练过程，在保持参数量少的同时取得良好性能。**

- **链接: [http://arxiv.org/pdf/2506.02393v1](http://arxiv.org/pdf/2506.02393v1)**

> **作者:** Yongxian Liu; Boyang Li; Ting Liu; Zaiping Lin; Wei An
>
> **摘要:** Infrared small target detection is a challenging task due to its unique characteristics (e.g., small, dim, shapeless and changeable). Recently published CNN-based methods have achieved promising performance with heavy feature extraction and fusion modules. To achieve efficient and effective detection, we propose a recurrent reusable-convolution attention network (RRCA-Net) for infrared small target detection. Specifically, RRCA-Net incorporates reusable-convolution block (RuCB) in a recurrent manner without introducing extra parameters. With the help of the repetitive iteration in RuCB, the high-level information of small targets in the deep layers can be well maintained and further refined. Then, a dual interactive attention aggregation module (DIAAM) is proposed to promote the mutual enhancement and fusion of refined information. In this way, RRCA-Net can both achieve high-level feature refinement and enhance the correlation of contextual information between adjacent layers. Moreover, to achieve steady convergence, we design a target characteristic inspired loss function (DpT-k loss) by integrating physical and mathematical constraints. Experimental results on three benchmark datasets (e.g. NUAA-SIRST, IRSTD-1k, DenseSIRST) demonstrate that our RRCA-Net can achieve comparable performance to the state-of-the-art methods while maintaining a small number of parameters, and act as a plug and play module to introduce consistent performance improvement for several popular IRSTD methods. Our code will be available at https://github.com/yongxianLiu/ soon.
>
---
#### [new 067] Enhancing Monocular Height Estimation via Weak Supervision from Imperfect Labels
- **分类: cs.CV**

- **简介: 该论文属于单目高度估计任务，旨在解决因缺乏高质量标签导致模型泛化能力差的问题。作者提出了一种基于弱监督的集成学习方法，利用不完整、不精确的标签训练网络，并设计了适配的损失函数与架构。实验表明，在DFC23和GBH数据集上性能提升明显，具备更好的跨域适应性。**

- **链接: [http://arxiv.org/pdf/2506.02534v1](http://arxiv.org/pdf/2506.02534v1)**

> **作者:** Sining Chen; Yilei Shi; Xiao Xiang Zhu
>
> **摘要:** Monocular height estimation is considered the most efficient and cost-effective means of 3D perception in remote sensing, and it has attracted much attention since the emergence of deep learning. While training neural networks requires a large amount of data, data with perfect labels are scarce and only available within developed regions. The trained models therefore lack generalizability, which limits the potential for large-scale application of existing methods. We tackle this problem for the first time, by introducing data with imperfect labels into training pixel-wise height estimation networks, including labels that are incomplete, inexact, and inaccurate compared to high-quality labels. We propose an ensemble-based pipeline compatible with any monocular height estimation network. Taking the challenges of noisy labels, domain shift, and long-tailed distribution of height values into consideration, we carefully design the architecture and loss functions to leverage the information concealed in imperfect labels using weak supervision through balanced soft losses and ordinal constraints. We conduct extensive experiments on two datasets with different resolutions, DFC23 (0.5 to 1 m) and GBH (3 m). The results indicate that the proposed pipeline outperforms baselines by achieving more balanced performance across various domains, leading to improvements of average root mean square errors up to 22.94 %, and 18.62 % on DFC23 and GBH, respectively. The efficacy of each design component is validated through ablation studies. Code is available at https://github.com/zhu-xlab/weakim2h.
>
---
#### [new 068] Large-scale Self-supervised Video Foundation Model for Intelligent Surgery
- **分类: cs.CV**

- **简介: 该论文属于手术视频理解任务，旨在解决现有方法缺乏对动态手术场景的时空建模问题。作者构建了大规模手术视频数据集，并提出SurgVISTA框架，通过时空联合建模和知识蒸馏提升细粒度特征学习，显著优化了手术智能系统的性能。**

- **链接: [http://arxiv.org/pdf/2506.02692v1](http://arxiv.org/pdf/2506.02692v1)**

> **作者:** Shu Yang; Fengtao Zhou; Leon Mayer; Fuxiang Huang; Yiliang Chen; Yihui Wang; Sunan He; Yuxiang Nie; Xi Wang; Ömer Sümer; Yueming Jin; Huihui Sun; Shuchang Xu; Alex Qinyang Liu; Zheng Li; Jing Qin; Jeremy YuenChun Teoh; Lena Maier-Hein; Hao Chen
>
> **摘要:** Computer-Assisted Intervention (CAI) has the potential to revolutionize modern surgery, with surgical scene understanding serving as a critical component in supporting decision-making, improving procedural efficacy, and ensuring intraoperative safety. While existing AI-driven approaches alleviate annotation burdens via self-supervised spatial representation learning, their lack of explicit temporal modeling during pre-training fundamentally restricts the capture of dynamic surgical contexts, resulting in incomplete spatiotemporal understanding. In this work, we introduce the first video-level surgical pre-training framework that enables joint spatiotemporal representation learning from large-scale surgical video data. To achieve this, we constructed a large-scale surgical video dataset comprising 3,650 videos and approximately 3.55 million frames, spanning more than 20 surgical procedures and over 10 anatomical structures. Building upon this dataset, we propose SurgVISTA (Surgical Video-level Spatial-Temporal Architecture), a reconstruction-based pre-training method that captures intricate spatial structures and temporal dynamics through joint spatiotemporal modeling. Additionally, SurgVISTA incorporates image-level knowledge distillation guided by a surgery-specific expert to enhance the learning of fine-grained anatomical and semantic features. To validate its effectiveness, we established a comprehensive benchmark comprising 13 video-level datasets spanning six surgical procedures across four tasks. Extensive experiments demonstrate that SurgVISTA consistently outperforms both natural- and surgical-domain pre-trained models, demonstrating strong potential to advance intelligent surgical systems in clinically meaningful scenarios.
>
---
#### [new 069] CamCloneMaster: Enabling Reference-based Camera Control for Video Generation
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决复杂摄像机运动控制不便的问题。作者提出了CamCloneMaster框架，通过参考视频实现无需参数设定的摄像机控制，并构建了大规模数据集进行训练与评估，提升了控制灵活性和视觉效果。**

- **链接: [http://arxiv.org/pdf/2506.03140v1](http://arxiv.org/pdf/2506.03140v1)**

> **作者:** Yawen Luo; Jianhong Bai; Xiaoyu Shi; Menghan Xia; Xintao Wang; Pengfei Wan; Di Zhang; Kun Gai; Tianfan Xue
>
> **备注:** Project Page: https://camclonemaster.github.io/
>
> **摘要:** Camera control is crucial for generating expressive and cinematic videos. Existing methods rely on explicit sequences of camera parameters as control conditions, which can be cumbersome for users to construct, particularly for intricate camera movements. To provide a more intuitive camera control method, we propose CamCloneMaster, a framework that enables users to replicate camera movements from reference videos without requiring camera parameters or test-time fine-tuning. CamCloneMaster seamlessly supports reference-based camera control for both Image-to-Video and Video-to-Video tasks within a unified framework. Furthermore, we present the Camera Clone Dataset, a large-scale synthetic dataset designed for camera clone learning, encompassing diverse scenes, subjects, and camera movements. Extensive experiments and user studies demonstrate that CamCloneMaster outperforms existing methods in terms of both camera controllability and visual quality.
>
---
#### [new 070] Video-Level Language-Driven Video-Based Visible-Infrared Person Re-Identification
- **分类: cs.CV**

- **简介: 该论文属于视频跨模态行人重识别任务，旨在解决可见光与红外视频序列间的模态差异问题。作者提出VLD框架，通过语言驱动的方式生成模ality共享的文本提示，并结合时空信息建模，提升了跨模态匹配性能，达到当前最优效果。**

- **链接: [http://arxiv.org/pdf/2506.02439v1](http://arxiv.org/pdf/2506.02439v1)**

> **作者:** Shuang Li; Jiaxu Leng; Changjiang Kuang; Mingpi Tan; Xinbo Gao
>
> **备注:** Accepted by IEEE TIFS
>
> **摘要:** Video-based Visible-Infrared Person Re-Identification (VVI-ReID) aims to match pedestrian sequences across modalities by extracting modality-invariant sequence-level features. As a high-level semantic representation, language provides a consistent description of pedestrian characteristics in both infrared and visible modalities. Leveraging the Contrastive Language-Image Pre-training (CLIP) model to generate video-level language prompts and guide the learning of modality-invariant sequence-level features is theoretically feasible. However, the challenge of generating and utilizing modality-shared video-level language prompts to address modality gaps remains a critical problem. To address this problem, we propose a simple yet powerful framework, video-level language-driven VVI-ReID (VLD), which consists of two core modules: invariant-modality language prompting (IMLP) and spatial-temporal prompting (STP). IMLP employs a joint fine-tuning strategy for the visual encoder and the prompt learner to effectively generate modality-shared text prompts and align them with visual features from different modalities in CLIP's multimodal space, thereby mitigating modality differences. Additionally, STP models spatiotemporal information through two submodules, the spatial-temporal hub (STH) and spatial-temporal aggregation (STA), which further enhance IMLP by incorporating spatiotemporal information into text prompts. The STH aggregates and diffuses spatiotemporal information into the [CLS] token of each frame across the vision transformer (ViT) layers, whereas STA introduces dedicated identity-level loss and specialized multihead attention to ensure that the STH focuses on identity-relevant spatiotemporal feature aggregation. The VLD framework achieves state-of-the-art results on two VVI-ReID benchmarks. The code will be released at https://github.com/Visuang/VLD.
>
---
#### [new 071] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型的空间推理任务，旨在解决当前模型在复杂空间理解上的不足。作者构建了一个全面的基准OmniSpatial，包含四大类和50个子类，涵盖动态推理、复杂空间逻辑等，并收集了1.5K问题对模型进行评估，揭示了现有模型的局限性，并提出了未来研究方向。**

- **链接: [http://arxiv.org/pdf/2506.03135v1](http://arxiv.org/pdf/2506.03135v1)**

> **作者:** Mengdi Jia; Zekun Qi; Shaochen Zhang; Wenyao Zhang; Xinqiang Yu; Jiawei He; He Wang; Li Yi
>
> **备注:** Project Page: https://qizekun.github.io/omnispatial/
>
> **摘要:** Spatial reasoning is a key aspect of cognitive psychology and remains a major bottleneck for current vision-language models (VLMs). While extensive research has aimed to evaluate or improve VLMs' understanding of basic spatial relations, such as distinguishing left from right, near from far, and object counting, these tasks represent only the most fundamental level of spatial reasoning. In this work, we introduce OmniSpatial, a comprehensive and challenging benchmark for spatial reasoning, grounded in cognitive psychology. OmniSpatial covers four major categories: dynamic reasoning, complex spatial logic, spatial interaction, and perspective-taking, with 50 fine-grained subcategories. Through Internet data crawling and careful manual annotation, we construct over 1.5K question-answer pairs. Extensive experiments show that both open- and closed-source VLMs, as well as existing reasoning and spatial understanding models, exhibit significant limitations in comprehensive spatial understanding. We further analyze failure cases and propose potential directions for future research.
>
---
#### [new 072] Medical World Model: Generative Simulation of Tumor Evolution for Treatment Planning
- **分类: cs.CV**

- **简介: 该论文提出“医学世界模型”（MeWM），旨在通过生成模拟肿瘤演变过程辅助临床决策。任务是医学中的治疗规划，解决如何有效预测疾病动态并优化个体化治疗方案的问题。工作包括构建视觉语言模型作为策略模型、肿瘤生成模型作为动态模型，并引入逆动力学模型评估治疗效果。结果显示其在模拟肿瘤变化和优化治疗协议方面优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.02327v1](http://arxiv.org/pdf/2506.02327v1)**

> **作者:** Yijun Yang; Zhao-Yang Wang; Qiuping Liu; Shuwen Sun; Kang Wang; Rama Chellappa; Zongwei Zhou; Alan Yuille; Lei Zhu; Yu-Dong Zhang; Jieneng Chen
>
> **摘要:** Providing effective treatment and making informed clinical decisions are essential goals of modern medicine and clinical care. We are interested in simulating disease dynamics for clinical decision-making, leveraging recent advances in large generative models. To this end, we introduce the Medical World Model (MeWM), the first world model in medicine that visually predicts future disease states based on clinical decisions. MeWM comprises (i) vision-language models to serve as policy models, and (ii) tumor generative models as dynamics models. The policy model generates action plans, such as clinical treatments, while the dynamics model simulates tumor progression or regression under given treatment conditions. Building on this, we propose the inverse dynamics model that applies survival analysis to the simulated post-treatment tumor, enabling the evaluation of treatment efficacy and the selection of the optimal clinical action plan. As a result, the proposed MeWM simulates disease dynamics by synthesizing post-treatment tumors, with state-of-the-art specificity in Turing tests evaluated by radiologists. Simultaneously, its inverse dynamics model outperforms medical-specialized GPTs in optimizing individualized treatment protocols across all metrics. Notably, MeWM improves clinical decision-making for interventional physicians, boosting F1-score in selecting the optimal TACE protocol by 13%, paving the way for future integration of medical world models as the second readers.
>
---
#### [new 073] Guiding Registration with Emergent Similarity from Pre-Trained Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于医学图像配准任务，旨在解决传统相似性损失在复杂场景下（如解剖结构缺失）导致配准不准的问题。作者利用预训练扩散模型提取语义特征，作为相似性度量来引导可变形配准网络，提升了多模态2D和单模态3D医学图像的配准效果。**

- **链接: [http://arxiv.org/pdf/2506.02419v1](http://arxiv.org/pdf/2506.02419v1)**

> **作者:** Nurislam Tursynbek; Hastings Greer; Basar Demir; Marc Niethammer
>
> **备注:** MICCAI 2025
>
> **摘要:** Diffusion models, while trained for image generation, have emerged as powerful foundational feature extractors for downstream tasks. We find that off-the-shelf diffusion models, trained exclusively to generate natural RGB images, can identify semantically meaningful correspondences in medical images. Building on this observation, we propose to leverage diffusion model features as a similarity measure to guide deformable image registration networks. We show that common intensity-based similarity losses often fail in challenging scenarios, such as when certain anatomies are visible in one image but absent in another, leading to anatomically inaccurate alignments. In contrast, our method identifies true semantic correspondences, aligning meaningful structures while disregarding those not present across images. We demonstrate superior performance of our approach on two tasks: multimodal 2D registration (DXA to X-Ray) and monomodal 3D registration (brain-extracted to non-brain-extracted MRI). Code: https://github.com/uncbiag/dgir
>
---
#### [new 074] Zero-Shot Tree Detection and Segmentation from Aerial Forest Imagery
- **分类: cs.CV**

- **简介: 该论文属于遥感图像中的树木检测与分割任务。旨在解决传统方法依赖大量标注数据的问题。工作内容是探索使用零样本方式的先进分割模型SAM2，进行单木分割和检测，并结合现有检测模型提升效果。**

- **链接: [http://arxiv.org/pdf/2506.03114v1](http://arxiv.org/pdf/2506.03114v1)**

> **作者:** Michelle Chen; David Russell; Amritha Pallavoor; Derek Young; Jane Wu
>
> **备注:** Code: https://github.com/open-forest-observatory/tree-detection-framework
>
> **摘要:** Large-scale delineation of individual trees from remote sensing imagery is crucial to the advancement of ecological research, particularly as climate change and other environmental factors rapidly transform forest landscapes across the world. Current RGB tree segmentation methods rely on training specialized machine learning models with labeled tree datasets. While these learning-based approaches can outperform manual data collection when accurate, the existing models still depend on training data that's hard to scale. In this paper, we investigate the efficacy of using a state-of-the-art image segmentation model, Segment Anything Model 2 (SAM2), in a zero-shot manner for individual tree detection and segmentation. We evaluate a pretrained SAM2 model on two tasks in this domain: (1) zero-shot segmentation and (2) zero-shot transfer by using predictions from an existing tree detection model as prompts. Our results suggest that SAM2 not only has impressive generalization capabilities, but also can form a natural synergy with specialized methods trained on in-domain labeled data. We find that applying large pretrained models to problems in remote sensing is a promising avenue for future progress. We make our code available at: https://github.com/open-forest-observatory/tree-detection-framework.
>
---
#### [new 075] LayoutRAG: Retrieval-Augmented Model for Content-agnostic Conditional Layout Generation
- **分类: cs.CV**

- **简介: 该论文属于可控布局生成任务，旨在根据给定条件生成合理的视觉布局。它通过检索匹配的布局模板并结合参考引导生成，解决了现有模型难以有效利用未显式提供的布局信息的问题。论文提出了一种基于检索和条件调制注意力的新方法，在生成过程中更有效地融合参考模板知识。**

- **链接: [http://arxiv.org/pdf/2506.02697v1](http://arxiv.org/pdf/2506.02697v1)**

> **作者:** Yuxuan Wu; Le Wang; Sanping Zhou; Mengnan Liu; Gang Hua; Haoxiang Li
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Controllable layout generation aims to create plausible visual arrangements of element bounding boxes within a graphic design according to certain optional constraints, such as the type or position of a specific component. While recent diffusion or flow-matching models have achieved considerable advances in multifarious conditional generation tasks, there remains considerable room for generating optimal arrangements under given conditions. In this work, we propose to carry out layout generation through retrieving by conditions and reference-guided generation. Specifically, we retrieve appropriate layout templates according to given conditions as references. The references are then utilized to guide the denoising or flow-based transport process. By retrieving layouts compatible with the given conditions, we can uncover the potential information not explicitly provided in the given condition. Such an approach offers more effective guidance to the model during the generation process, in contrast to previous models that feed the condition to the model and let the model infer the unprovided layout attributes directly. Meanwhile, we design a condition-modulated attention that selectively absorbs retrieval knowledge, adapting to the difference between retrieved templates and given conditions. Extensive experiment results show that our method successfully produces high-quality layouts that meet the given conditions and outperforms existing state-of-the-art models. Code will be released upon acceptance.
>
---
#### [new 076] ByteMorph: Benchmarking Instruction-Guided Image Editing with Non-Rigid Motions
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，旨在解决基于指令的非刚性运动图像编辑问题。现有方法多关注静态场景或刚性变换，难以处理复杂动态编辑。论文提出ByteMorph框架，包含大规模数据集ByteMorph-6M和基线模型ByteMorpher，支持多样化的非刚性运动编辑。**

- **链接: [http://arxiv.org/pdf/2506.03107v1](http://arxiv.org/pdf/2506.03107v1)**

> **作者:** Di Chang; Mingdeng Cao; Yichun Shi; Bo Liu; Shengqu Cai; Shijie Zhou; Weilin Huang; Gordon Wetzstein; Mohammad Soleymani; Peng Wang
>
> **备注:** Website: https://boese0601.github.io/bytemorph Dataset: https://huggingface.co/datasets/ByteDance-Seed/BM-6M Benchmark: https://huggingface.co/datasets/ByteDance-Seed/BM-Bench Code: https://github.com/ByteDance-Seed/BM-code Demo: https://huggingface.co/spaces/Boese0601/ByteMorph-Demo
>
> **摘要:** Editing images with instructions to reflect non-rigid motions, camera viewpoint shifts, object deformations, human articulations, and complex interactions, poses a challenging yet underexplored problem in computer vision. Existing approaches and datasets predominantly focus on static scenes or rigid transformations, limiting their capacity to handle expressive edits involving dynamic motion. To address this gap, we introduce ByteMorph, a comprehensive framework for instruction-based image editing with an emphasis on non-rigid motions. ByteMorph comprises a large-scale dataset, ByteMorph-6M, and a strong baseline model built upon the Diffusion Transformer (DiT), named ByteMorpher. ByteMorph-6M includes over 6 million high-resolution image editing pairs for training, along with a carefully curated evaluation benchmark ByteMorph-Bench. Both capture a wide variety of non-rigid motion types across diverse environments, human figures, and object categories. The dataset is constructed using motion-guided data generation, layered compositing techniques, and automated captioning to ensure diversity, realism, and semantic coherence. We further conduct a comprehensive evaluation of recent instruction-based image editing methods from both academic and commercial domains.
>
---
#### [new 077] Auto-Labeling Data for Object Detection
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务中的目标检测领域，旨在解决传统标注成本高昂的问题。论文提出了一种无需真实标签的训练方法，利用已有的视觉-语言模型生成伪标签，用于训练轻量级检测模型。实验表明，该方法在保持性能的同时显著降低了标注时间和成本。**

- **链接: [http://arxiv.org/pdf/2506.02359v1](http://arxiv.org/pdf/2506.02359v1)**

> **作者:** Brent A. Griffin; Manushree Gangwar; Jacob Sela; Jason J. Corso
>
> **摘要:** Great labels make great models. However, traditional labeling approaches for tasks like object detection have substantial costs at scale. Furthermore, alternatives to fully-supervised object detection either lose functionality or require larger models with prohibitive computational costs for inference at scale. To that end, this paper addresses the problem of training standard object detection models without any ground truth labels. Instead, we configure previously-trained vision-language foundation models to generate application-specific pseudo "ground truth" labels. These auto-generated labels directly integrate with existing model training frameworks, and we subsequently train lightweight detection models that are computationally efficient. In this way, we avoid the costs of traditional labeling, leverage the knowledge of vision-language models, and keep the efficiency of lightweight models for practical application. We perform exhaustive experiments across multiple labeling configurations, downstream inference models, and datasets to establish best practices and set an extensive auto-labeling benchmark. From our results, we find that our approach is a viable alternative to standard labeling in that it maintains competitive performance on multiple datasets and substantially reduces labeling time and costs.
>
---
#### [new 078] HaploOmni: Unified Single Transformer for Multimodal Video Understanding and Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态视频理解和生成任务，旨在解决统一模型架构下的跨模态兼容性问题。作者提出HaploOmni，通过多模态预训练策略、特征预缩放和AdaLN技术，实现高效的统一单模型处理。**

- **链接: [http://arxiv.org/pdf/2506.02975v1](http://arxiv.org/pdf/2506.02975v1)**

> **作者:** Yicheng Xiao; Lin Song; Rui Yang; Cheng Cheng; Zunnan Xu; Zhaoyang Zhang; Yixiao Ge; Xiu Li; Ying Shan
>
> **摘要:** With the advancement of language models, unified multimodal understanding and generation have made significant strides, with model architectures evolving from separated components to unified single-model frameworks. This paper explores an efficient training paradigm to build a single transformer for unified multimodal understanding and generation. Specifically, we propose a multimodal warmup strategy utilizing prior knowledge to extend capabilities. To address cross-modal compatibility challenges, we introduce feature pre-scaling and multimodal AdaLN techniques. Integrating the proposed technologies, we present the HaploOmni, a new single multimodal transformer. With limited training costs, HaploOmni achieves competitive performance across multiple image and video understanding and generation benchmarks over advanced unified models. All codes will be made public at https://github.com/Tencent/HaploVLM.
>
---
#### [new 079] ViTNF: Leveraging Neural Fields to Boost Vision Transformers in Generalized Category Discovery
- **分类: cs.CV; 68T07; I.5.1**

- **简介: 该论文属于开放世界识别中的广义类别发现（GCD）任务，旨在利用已知类别数据识别未知类别样本。现有方法训练成本高且特征提取能力未被充分利用。论文提出ViTNF架构，用神经场分类器替代传统MLP头，通过静态神经场函数构建高效少样本分类器，并优化训练流程，显著降低训练难度和样本需求，提升了识别新类别的性能。实验表明其在多个数据集上表现优异，准确率大幅提升。**

- **链接: [http://arxiv.org/pdf/2506.02367v1](http://arxiv.org/pdf/2506.02367v1)**

> **作者:** Jiayi Su; Dequan Jin
>
> **备注:** 22 pages, 3 figures
>
> **摘要:** Generalized category discovery (GCD) is a highly popular task in open-world recognition, aiming to identify unknown class samples using known class data. By leveraging pre-training, meta-training, and fine-tuning, ViT achieves excellent few-shot learning capabilities. Its MLP head is a feedforward network, trained synchronously with the entire network in the same process, increasing the training cost and difficulty without fully leveraging the power of the feature extractor. This paper proposes a new architecture by replacing the MLP head with a neural field-based one. We first present a new static neural field function to describe the activity distribution of the neural field and then use two static neural field functions to build an efficient few-shot classifier. This neural field-based (NF) classifier consists of two coupled static neural fields. It stores the feature information of support samples by its elementary field, the known categories by its high-level field, and the category information of support samples by its cross-field connections. We replace the MLP head with the proposed NF classifier, resulting in a novel architecture ViTNF, and simplify the three-stage training mode by pre-training the feature extractor on source tasks and training the NF classifier with support samples in meta-testing separately, significantly reducing ViT's demand for training samples and the difficulty of model training. To enhance the model's capability in identifying new categories, we provide an effective algorithm to determine the lateral interaction scale of the elementary field. Experimental results demonstrate that our model surpasses existing state-of-the-art methods on CIFAR-100, ImageNet-100, CUB-200, and Standard Cars, achieving dramatic accuracy improvements of 19\% and 16\% in new and all classes, respectively, indicating a notable advantage in GCD.
>
---
#### [new 080] ToothForge: Automatic Dental Shape Generation using Synchronized Spectral Embeddings
- **分类: cs.CV**

- **简介: 该论文提出ToothForge，用于自动生成3D牙齿形状，解决牙科数据稀疏问题。通过谱域建模和同步频域嵌入，实现高分辨率牙齿网格快速生成，并克服分解不稳定性和固定连接限制。方法适用于牙科及其他医学形状分析场景。**

- **链接: [http://arxiv.org/pdf/2506.02702v1](http://arxiv.org/pdf/2506.02702v1)**

> **作者:** Tibor Kubík; François Guibault; Michal Španěl; Hervé Lombaert
>
> **备注:** Information Processing in Medical Imaging (IPMI2025)
>
> **摘要:** We introduce ToothForge, a spectral approach for automatically generating novel 3D teeth, effectively addressing the sparsity of dental shape datasets. By operating in the spectral domain, our method enables compact machine learning modeling, allowing the generation of high-resolution tooth meshes in milliseconds. However, generating shape spectra comes with the instability of the decomposed harmonics. To address this, we propose modeling the latent manifold on synchronized frequential embeddings. Spectra of all data samples are aligned to a common basis prior to the training procedure, effectively eliminating biases introduced by the decomposition instability. Furthermore, synchronized modeling removes the limiting factor imposed by previous methods, which require all shapes to share a common fixed connectivity. Using a private dataset of real dental crowns, we observe a greater reconstruction quality of the synthetized shapes, exceeding those of models trained on unaligned embeddings. We also explore additional applications of spectral analysis in digital dentistry, such as shape compression and interpolation. ToothForge facilitates a range of approaches at the intersection of spectral analysis and machine learning, with fewer restrictions on mesh structure. This makes it applicable for shape analysis not only in dentistry, but also in broader medical applications, where guaranteeing consistent connectivity across shapes from various clinics is unrealistic. The code is available at https://github.com/tiborkubik/toothForge.
>
---
#### [new 081] CNVSRC 2024: The Second Chinese Continuous Visual Speech Recognition Challenge
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于中文连续视觉语音识别任务，旨在解决大词汇量下的视觉语音识别问题。论文通过改进基线系统和引入新数据集提升识别效果，推动相关技术发展。**

- **链接: [http://arxiv.org/pdf/2506.02010v1](http://arxiv.org/pdf/2506.02010v1)**

> **作者:** Zehua Liu; Xiaolou Li; Chen Chen; Lantian Li; Dong Wang
>
> **备注:** to be published in INTERSPEECH 2025
>
> **摘要:** This paper presents the second Chinese Continuous Visual Speech Recognition Challenge (CNVSRC 2024), which builds on CNVSRC 2023 to advance research in Chinese Large Vocabulary Continuous Visual Speech Recognition (LVC-VSR). The challenge evaluates two test scenarios: reading in recording studios and Internet speech. CNVSRC 2024 uses the same datasets as its predecessor CNVSRC 2023, which involves CN-CVS for training and CNVSRC-Single/Multi for development and evaluation. However, CNVSRC 2024 introduced two key improvements: (1) a stronger baseline system, and (2) an additional dataset, CN-CVS2-P1, for open tracks to improve data volume and diversity. The new challenge has demonstrated several important innovations in data preprocessing, feature extraction, model design, and training strategies, further pushing the state-of-the-art in Chinese LVC-VSR. More details and resources are available at the official website.
>
---
#### [new 082] OpenFace 3.0: A Lightweight Multitask System for Comprehensive Facial Behavior Analysis
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉与多模态交互任务，旨在解决面部行为自动分析问题。论文提出了OpenFace 3.0，一个轻量级、多功能的开源工具包，支持面部关键点检测、动作单元识别、眼动估计和情绪识别，提升了性能、速度和内存效率，适用于多样化场景并支持社区贡献。**

- **链接: [http://arxiv.org/pdf/2506.02891v1](http://arxiv.org/pdf/2506.02891v1)**

> **作者:** Jiewen Hu; Leena Mathur; Paul Pu Liang; Louis-Philippe Morency
>
> **备注:** IEEE FG 2025, \c{opyright} 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work
>
> **摘要:** In recent years, there has been increasing interest in automatic facial behavior analysis systems from computing communities such as vision, multimodal interaction, robotics, and affective computing. Building upon the widespread utility of prior open-source facial analysis systems, we introduce OpenFace 3.0, an open-source toolkit capable of facial landmark detection, facial action unit detection, eye-gaze estimation, and facial emotion recognition. OpenFace 3.0 contributes a lightweight unified model for facial analysis, trained with a multi-task architecture across diverse populations, head poses, lighting conditions, video resolutions, and facial analysis tasks. By leveraging the benefits of parameter sharing through a unified model and training paradigm, OpenFace 3.0 exhibits improvements in prediction performance, inference speed, and memory efficiency over similar toolkits and rivals state-of-the-art models. OpenFace 3.0 can be installed and run with a single line of code and operate in real-time without specialized hardware. OpenFace 3.0 code for training models and running the system is freely available for research purposes and supports contributions from the community.
>
---
#### [new 083] Towards In-the-wild 3D Plane Reconstruction from a Single Image
- **分类: cs.CV**

- **简介: 该论文属于3D计算机视觉任务，旨在解决单张图像的零样本3D平面重建问题。作者提出了ZeroPlane框架，通过多域数据训练和新模块设计，实现跨场景的平面检测与重建，显著提升重建精度和泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.02493v1](http://arxiv.org/pdf/2506.02493v1)**

> **作者:** Jiachen Liu; Rui Yu; Sili Chen; Sharon X. Huang; Hengkai Guo
>
> **备注:** CVPR 2025 Highlighted Paper
>
> **摘要:** 3D plane reconstruction from a single image is a crucial yet challenging topic in 3D computer vision. Previous state-of-the-art (SOTA) methods have focused on training their system on a single dataset from either indoor or outdoor domain, limiting their generalizability across diverse testing data. In this work, we introduce a novel framework dubbed ZeroPlane, a Transformer-based model targeting zero-shot 3D plane detection and reconstruction from a single image, over diverse domains and environments. To enable data-driven models across multiple domains, we have curated a large-scale planar benchmark, comprising over 14 datasets and 560,000 high-resolution, dense planar annotations for diverse indoor and outdoor scenes. To address the challenge of achieving desirable planar geometry on multi-dataset training, we propose to disentangle the representation of plane normal and offset, and employ an exemplar-guided, classification-then-regression paradigm to learn plane and offset respectively. Additionally, we employ advanced backbones as image encoder, and present an effective pixel-geometry-enhanced plane embedding module to further facilitate planar reconstruction. Extensive experiments across multiple zero-shot evaluation datasets have demonstrated that our approach significantly outperforms previous methods on both reconstruction accuracy and generalizability, especially over in-the-wild data. Our code and data are available at: https://github.com/jcliu0428/ZeroPlane.
>
---
#### [new 084] Generalized Category Discovery via Reciprocal Learning and Class-Wise Distribution Regularization
- **分类: cs.CV**

- **简介: 该论文属于通用类别发现任务，旨在利用有标签数据的知识识别未标签样本中的新类别。现有方法因自监督不可靠导致分类性能下降。作者提出RLCD方法，结合互惠学习框架和类分布正则化，提升模型对新类的识别能力，同时保持对已知类的判别力，有效解决了学习偏差问题。**

- **链接: [http://arxiv.org/pdf/2506.02334v1](http://arxiv.org/pdf/2506.02334v1)**

> **作者:** Duo Liu; Zhiquan Tan; Linglan Zhao; Zhongqiang Zhang; Xiangzhong Fang; Weiran Huang
>
> **备注:** ICML2025 Poster
>
> **摘要:** Generalized Category Discovery (GCD) aims to identify unlabeled samples by leveraging the base knowledge from labeled ones, where the unlabeled set consists of both base and novel classes. Since clustering methods are time-consuming at inference, parametric-based approaches have become more popular. However, recent parametric-based methods suffer from inferior base discrimination due to unreliable self-supervision. To address this issue, we propose a Reciprocal Learning Framework (RLF) that introduces an auxiliary branch devoted to base classification. During training, the main branch filters the pseudo-base samples to the auxiliary branch. In response, the auxiliary branch provides more reliable soft labels for the main branch, leading to a virtuous cycle. Furthermore, we introduce Class-wise Distribution Regularization (CDR) to mitigate the learning bias towards base classes. CDR essentially increases the prediction confidence of the unlabeled data and boosts the novel class performance. Combined with both components, our proposed method, RLCD, achieves superior performance in all classes with negligible extra computation. Comprehensive experiments across seven GCD datasets validate its superiority. Our codes are available at https://github.com/APORduo/RLCD.
>
---
#### [new 085] IllumiCraft: Unified Geometry and Illumination Diffusion for Controllable Video Generation
- **分类: cs.CV; cs.AI; cs.LG; cs.MM**

- **简介: 该论文属于视频生成任务，旨在解决现有扩散模型在控制场景光照和视觉外观时缺乏几何信息整合的问题。作者提出了IllumiCraft框架，结合HDR视频图、重照明帧和3D点轨迹，实现高质量、可控的视频生成，提升时间一致性和真实感。**

- **链接: [http://arxiv.org/pdf/2506.03150v1](http://arxiv.org/pdf/2506.03150v1)**

> **作者:** Yuanze Lin; Yi-Wen Chen; Yi-Hsuan Tsai; Ronald Clark; Ming-Hsuan Yang
>
> **备注:** Tech Report
>
> **摘要:** Although diffusion-based models can generate high-quality and high-resolution video sequences from textual or image inputs, they lack explicit integration of geometric cues when controlling scene lighting and visual appearance across frames. To address this limitation, we propose IllumiCraft, an end-to-end diffusion framework accepting three complementary inputs: (1) high-dynamic-range (HDR) video maps for detailed lighting control; (2) synthetically relit frames with randomized illumination changes (optionally paired with a static background reference image) to provide appearance cues; and (3) 3D point tracks that capture precise 3D geometry information. By integrating the lighting, appearance, and geometry cues within a unified diffusion architecture, IllumiCraft generates temporally coherent videos aligned with user-defined prompts. It supports background-conditioned and text-conditioned video relighting and provides better fidelity than existing controllable video generation methods. Project Page: https://yuanze-lin.me/IllumiCraft_page
>
---
#### [new 086] NTIRE 2025 XGC Quality Assessment Challenge: Methods and Results
- **分类: cs.CV**

- **简介: 该论文属于图像处理与视频质量评估任务，旨在解决用户生成视频、AI生成视频和2D/3D talking head的质量评估问题。论文组织了三组挑战赛，分别使用FineVD-GC、Q-Eval-Video和THQA-NTIRE数据集，吸引了多个团队参与并提交优于基线的方法，推动相关领域发展。**

- **链接: [http://arxiv.org/pdf/2506.02875v1](http://arxiv.org/pdf/2506.02875v1)**

> **作者:** Xiaohong Liu; Xiongkuo Min; Qiang Hu; Xiaoyun Zhang; Jie Guo; Guangtao Zhai; Shushi Wang; Yingjie Zhou; Lu Liu; Jingxin Li; Liu Yang; Farong Wen; Li Xu; Yanwei Jiang; Xilei Zhu; Chunyi Li; Zicheng Zhang; Huiyu Duan; Xiele Wu; Yixuan Gao; Yuqin Cao; Jun Jia; Wei Sun; Jiezhang Cao; Radu Timofte; Baojun Li; Jiamian Huang; Dan Luo; Tao Liu; Weixia Zhang; Bingkun Zheng; Junlin Chen; Ruikai Zhou; Meiya Chen; Yu Wang; Hao Jiang; Xiantao Li; Yuxiang Jiang; Jun Tang; Yimeng Zhao; Bo Hu; Zelu Qi; Chaoyang Zhang; Fei Zhao; Ping Shi; Lingzhi Fu; Heng Cong; Shuai He; Rongyu Zhang; Jiarong He; Zongyao Hu; Wei Luo; Zihao Yu; Fengbin Guan; Yiting Lu; Xin Li; Zhibo Chen; Mengjing Su; Yi Wang; Tuo Chen; Chunxiao Li; Shuaiyu Zhao; Jiaxin Wen; Chuyi Lin; Sitong Liu; Ningxin Chu; Jing Wan; Yu Zhou; Baoying Chen; Jishen Zeng; Jiarui Liu; Xianjin Liu; Xin Chen; Lanzhi Zhou; Hangyu Li; You Han; Bibo Xiang; Zhenjie Liu; Jianzhang Lu; Jialin Gui; Renjie Lu; Shangfei Wang; Donghao Zhou; Jingyu Lin; Quanjian Song; Jiancheng Huang; Yufeng Yang; Changwei Wang; Shupeng Zhong; Yang Yang; Lihuo He; Jia Liu; Yuting Xing; Tida Fang; Yuchun Jin
>
> **备注:** NTIRE 2025 XGC Quality Assessment Challenge Report. arXiv admin note: text overlap with arXiv:2404.16687
>
> **摘要:** This paper reports on the NTIRE 2025 XGC Quality Assessment Challenge, which will be held in conjunction with the New Trends in Image Restoration and Enhancement Workshop (NTIRE) at CVPR 2025. This challenge is to address a major challenge in the field of video and talking head processing. The challenge is divided into three tracks, including user generated video, AI generated video and talking head. The user-generated video track uses the FineVD-GC, which contains 6,284 user generated videos. The user-generated video track has a total of 125 registered participants. A total of 242 submissions are received in the development phase, and 136 submissions are received in the test phase. Finally, 5 participating teams submitted their models and fact sheets. The AI generated video track uses the Q-Eval-Video, which contains 34,029 AI-Generated Videos (AIGVs) generated by 11 popular Text-to-Video (T2V) models. A total of 133 participants have registered in this track. A total of 396 submissions are received in the development phase, and 226 submissions are received in the test phase. Finally, 6 participating teams submitted their models and fact sheets. The talking head track uses the THQA-NTIRE, which contains 12,247 2D and 3D talking heads. A total of 89 participants have registered in this track. A total of 225 submissions are received in the development phase, and 118 submissions are received in the test phase. Finally, 8 participating teams submitted their models and fact sheets. Each participating team in every track has proposed a method that outperforms the baseline, which has contributed to the development of fields in three tracks.
>
---
#### [new 087] Automated Measurement of Optic Nerve Sheath Diameter Using Ocular Ultrasound Video
- **分类: cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决手动测量视神经鞘直径（ONSD）依赖操作者经验的问题。作者提出一种结合KCF跟踪、SLIC分割和GMM-KL方法的自动化框架，实现从眼部超声视频中自动选取最佳帧并精确测量ONSD，具有较高临床应用潜力。**

- **链接: [http://arxiv.org/pdf/2506.02789v1](http://arxiv.org/pdf/2506.02789v1)**

> **作者:** Renxing Li; Weiyi Tang; Peiqi Li; Qiming Huang; Jiayuan She; Shengkai Li; Haoran Xu; Yeyun Wan; Jing Liu; Hailong Fu; Xiang Li; Jiangang Chen
>
> **备注:** 17 pages, 9 figures
>
> **摘要:** Objective. Elevated intracranial pressure (ICP) is recognized as a biomarker of secondary brain injury, with a significant linear correlation observed between optic nerve sheath diameter (ONSD) and ICP. Frequent monitoring of ONSD could effectively support dynamic evaluation of ICP. However, ONSD measurement is heavily reliant on the operator's experience and skill, particularly in manually selecting the optimal frame from ultrasound sequences and measuring ONSD. Approach. This paper presents a novel method to automatically identify the optimal frame from video sequences for ONSD measurement by employing the Kernel Correlation Filter (KCF) tracking algorithm and Simple Linear Iterative Clustering (SLIC) segmentation algorithm. The optic nerve sheath is mapped and measured using a Gaussian Mixture Model (GMM) combined with a KL-divergence-based method. Results. When compared with the average measurements of two expert clinicians, the proposed method achieved a mean error, mean squared deviation, and intraclass correlation coefficient (ICC) of 0.04, 0.054, and 0.782, respectively. Significance. The findings suggest that this method provides highly accurate automated ONSD measurements, showing potential for clinical application.
>
---
#### [new 088] Astrophotography turbulence mitigation via generative models
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于图像恢复任务，旨在解决地面望远镜因大气湍流导致的天文图像质量下降问题。作者提出了AstroDiff方法，利用扩散模型的生成先验和恢复能力来减轻湍流影响，实验表明其在严重湍流条件下优于现有学习方法，提升了图像感知质量和结构保真度。**

- **链接: [http://arxiv.org/pdf/2506.02981v1](http://arxiv.org/pdf/2506.02981v1)**

> **作者:** Joonyeoup Kim; Yu Yuan; Xingguang Zhang; Xijun Wang; Stanley Chan
>
> **摘要:** Photography is the cornerstone of modern astronomical and space research. However, most astronomical images captured by ground-based telescopes suffer from atmospheric turbulence, resulting in degraded imaging quality. While multi-frame strategies like lucky imaging can mitigate some effects, they involve intensive data acquisition and complex manual processing. In this paper, we propose AstroDiff, a generative restoration method that leverages both the high-quality generative priors and restoration capabilities of diffusion models to mitigate atmospheric turbulence. Extensive experiments demonstrate that AstroDiff outperforms existing state-of-the-art learning-based methods in astronomical image turbulence mitigation, providing higher perceptual quality and better structural fidelity under severe turbulence conditions. Our code and additional results are available at https://web-six-kappa-66.vercel.app/
>
---
#### [new 089] FaceSleuth: Learning-Driven Single-Orientation Attention Verifies Vertical Dominance in Micro-Expression Recognition
- **分类: cs.CV**

- **简介: 该论文属于微表情识别任务，旨在解决如何有效捕捉面部细微垂直运动并抑制身份特征干扰的问题。论文提出FaceSleuth模型，包含增强垂直运动的CVA模块、定位信号的Focalizer和引导学习的AU嵌入，并设计可学习方向的SOA模块验证垂直注意力最优性，实验证明其性能领先。**

- **链接: [http://arxiv.org/pdf/2506.02695v1](http://arxiv.org/pdf/2506.02695v1)**

> **作者:** Linquan Wu; Tianxiang Jiang; Wenhao Duan; Yini Fang; Jacky Keung
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** Micro-expression recognition (MER) demands models that can amplify millisecond-level, low-amplitude facial motions while suppressing identity-specific appearance. We introduce FaceSleuth, a dual-stream architecture that (1) enhances motion along the empirically dominant vertical axix through a Continuously Vertical Attention (CVA) block, (2) localises the resulting signals with a Facial Position Focalizer built on hierarchical cross-window attention, and (3) steers feature learning toward physiologically meaningful regions via lightweight Action-Unit embeddings. To examine whether the hand-chosen vertical axis is indeed optimal, we further propose a Single-Orientation Attention (SOA) module that learns its own pooling direction end-to-end. SOA is differentiable, adds only 0.16 % parameters, and collapses to CVA when the learned angle converges to {\Pi}/2. In practice, SOA reliably drifts to 88{\deg}, confirming the effectiveness of the vertical prior while delivering consistent gains. On three standard MER benchmarks, FaceSleuth with CVA already surpasses previous state-of-the-art methods; plugging in SOA lifts accuracy and F1 score performance to 95.1 % / 0.918 on CASME II, 87.1 % / 0.840 on SAMM, and 92.9 % / 0.917 on MMEW without sacrificing model compactness. These results establish a new state of the art and, for the first time, provide empirical evidence that the vertical attention bias is the most discriminative orientation for MER.
>
---
#### [new 090] Fire360: A Benchmark for Robust Perception and Episodic Memory in Degraded 360-Degree Firefighting Videos
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出了Fire360，一个用于评估在恶劣环境下感知与推理能力的基准数据集，主要针对消防场景。其任务包括视觉问答、动作描述、目标定位、安全推理和变形目标检索（TOR）。论文旨在提升AI在烟雾、低光等复杂条件下的鲁棒感知与记忆能力。**

- **链接: [http://arxiv.org/pdf/2506.02167v1](http://arxiv.org/pdf/2506.02167v1)**

> **作者:** Aditi Tiwari; Farzaneh Masoud; Dac Trong Nguyen; Jill Kraft; Heng Ji; Klara Nahrstedt
>
> **备注:** 20 pages, 9 figures, 6 tables
>
> **摘要:** Modern AI systems struggle most in environments where reliability is critical - scenes with smoke, poor visibility, and structural deformation. Each year, tens of thousands of firefighters are injured on duty, often due to breakdowns in situational perception. We introduce Fire360, a benchmark for evaluating perception and reasoning in safety-critical firefighting scenarios. The dataset includes 228 360-degree videos from professional training sessions under diverse conditions (e.g., low light, thermal distortion), annotated with action segments, object locations, and degradation metadata. Fire360 supports five tasks: Visual Question Answering, Temporal Action Captioning, Object Localization, Safety-Critical Reasoning, and Transformed Object Retrieval (TOR). TOR tests whether models can match pristine exemplars to fire-damaged counterparts in unpaired scenes, evaluating transformation-invariant recognition. While human experts achieve 83.5% on TOR, models like GPT-4o lag significantly, exposing failures in reasoning under degradation. By releasing Fire360 and its evaluation suite, we aim to advance models that not only see, but also remember, reason, and act under uncertainty. The dataset is available at: https://uofi.box.com/v/fire360dataset.
>
---
#### [new 091] Synthetic Iris Image Databases and Identity Leakage: Risks and Mitigation Strategies
- **分类: cs.CV**

- **简介: 该论文属于生物特征数据安全任务，旨在解决合成虹膜图像数据库中的身份泄露风险问题。论文综述了多种虹膜图像生成方法，并探讨了其潜在的隐私风险及缓解策略。**

- **链接: [http://arxiv.org/pdf/2506.02626v1](http://arxiv.org/pdf/2506.02626v1)**

> **作者:** Ada Sawilska; Mateusz Trokielewicz
>
> **摘要:** This paper presents a comprehensive overview of iris image synthesis methods, which can alleviate the issues associated with gathering large, diverse datasets of biometric data from living individuals, which are considered pivotal for biometric methods development. These methods for synthesizing iris data range from traditional, hand crafted image processing-based techniques, through various iterations of GAN-based image generators, variational autoencoders (VAEs), as well as diffusion models. The potential and fidelity in iris image generation of each method is discussed and examples of inferred predictions are provided. Furthermore, the risks of individual biometric features leakage from the training sets are considered, together with possible strategies for preventing them, which have to be implemented should these generative methods be considered a valid replacement of real-world biometric datasets.
>
---
#### [new 092] Towards Auto-Annotation from Annotation Guidelines: A Benchmark through 3D LiDAR Detection
- **分类: cs.CV**

- **简介: 论文提出AnnoGuide基准，旨在通过标注指南实现自动数据标注，解决3D LiDAR检测中依赖人工标注的问题。利用nuScenes数据集和多模态方法，结合图像检测模型与LiDAR点云投影聚类，构建3D立方体标注。虽提升mAP至21.9，但仍需发展LiDAR基础模型。**

- **链接: [http://arxiv.org/pdf/2506.02914v1](http://arxiv.org/pdf/2506.02914v1)**

> **作者:** Yechi Ma; Wei Hua; Shu Kong
>
> **摘要:** A crucial yet under-appreciated prerequisite in machine learning solutions for real-applications is data annotation: human annotators are hired to manually label data according to detailed, expert-crafted guidelines. This is often a laborious, tedious, and costly process. To study methods for facilitating data annotation, we introduce a new benchmark AnnoGuide: Auto-Annotation from Annotation Guidelines. It aims to evaluate automated methods for data annotation directly from expert-defined annotation guidelines, eliminating the need for manual labeling. As a case study, we repurpose the well-established nuScenes dataset, commonly used in autonomous driving research, which provides comprehensive annotation guidelines for labeling LiDAR point clouds with 3D cuboids across 18 object classes. These guidelines include a few visual examples and textual descriptions, but no labeled 3D cuboids in LiDAR data, making this a novel task of multi-modal few-shot 3D detection without 3D annotations. The advances of powerful foundation models (FMs) make AnnoGuide especially timely, as FMs offer promising tools to tackle its challenges. We employ a conceptually straightforward pipeline that (1) utilizes open-source FMs for object detection and segmentation in RGB images, (2) projects 2D detections into 3D using known camera poses, and (3) clusters LiDAR points within the frustum of each 2D detection to generate a 3D cuboid. Starting with a non-learned solution that leverages off-the-shelf FMs, we progressively refine key components and achieve significant performance improvements, boosting 3D detection mAP from 12.1 to 21.9! Nevertheless, our results highlight that AnnoGuide remains an open and challenging problem, underscoring the urgent need for developing LiDAR-based FMs. We release our code and models at GitHub: https://annoguide.github.io/annoguide3Dbenchmark
>
---
#### [new 093] FuseLIP: Multimodal Embeddings via Early Fusion of Discrete Tokens
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多模态嵌入任务，旨在解决现有方法无法原生处理多模态输入的问题。作者提出FuseLIP，通过早期融合文本和图像token，使用单一Transformer模型进行多模态表示学习，提升了多模态任务性能。**

- **链接: [http://arxiv.org/pdf/2506.03096v1](http://arxiv.org/pdf/2506.03096v1)**

> **作者:** Christian Schlarmann; Francesco Croce; Nicolas Flammarion; Matthias Hein
>
> **备注:** Code and models available at https://github.com/chs20/fuselip
>
> **摘要:** Contrastive language-image pre-training aligns the features of text-image pairs in a common latent space via distinct encoders for each modality. While this approach achieves impressive performance in several zero-shot tasks, it cannot natively handle multimodal inputs, i.e., encoding image and text into a single feature vector. As a remedy, it is common practice to use additional modules to merge the features extracted by the unimodal encoders. In this work, we present FuseLIP, an alternative architecture for multimodal embedding. Leveraging recent progress in discrete image tokenizers, we propose to use a single transformer model which operates on an extended vocabulary of text and image tokens. This early fusion approach allows the different modalities to interact at each depth of encoding and obtain richer representations compared to common late fusion. We collect new datasets for multimodal pre-training and evaluation, designing challenging tasks for multimodal encoder models. We show that FuseLIP outperforms other approaches in multimodal embedding tasks such as VQA and text-guided image transformation retrieval, while being comparable to baselines on unimodal tasks.
>
---
#### [new 094] MERIT: Multilingual Semantic Retrieval with Interleaved Multi-Condition Query
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文属于多语言语义检索任务，旨在解决现有数据集和模型在处理多条件、多语言及多图像检索时的局限性。作者构建了首个支持多条件查询的多语言语义检索数据集MERIT，并提出Coral框架，结合嵌入重建与对比学习，显著提升检索性能。**

- **链接: [http://arxiv.org/pdf/2506.03144v1](http://arxiv.org/pdf/2506.03144v1)**

> **作者:** Wei Chow; Yuan Gao; Linfeng Li; Xian Wang; Qi Xu; Hang Song; Lingdong Kong; Ran Zhou; Yi Zeng; Yidong Cai; Botian Jiang; Shilin Xu; Jiajun Zhang; Minghui Qiu; Xiangtai Li; Tianshu Yang; Siliang Tang; Juncheng Li
>
> **备注:** Preprint; Project Page, Code, and Dataset at: https://merit-2025.github.io/
>
> **摘要:** Semantic retrieval is crucial for modern applications yet remains underexplored in current research. Existing datasets are limited to single languages, single images, or singular retrieval conditions, often failing to fully exploit the expressive capacity of visual information as evidenced by maintained performance when images are replaced with captions. However, practical retrieval scenarios frequently involve interleaved multi-condition queries with multiple images. Hence, this paper introduces MERIT, the first multilingual dataset for interleaved multi-condition semantic retrieval, comprising 320,000 queries with 135,000 products in 5 languages, covering 7 distinct product categories. Extensive experiments on MERIT identify existing models's limitation: focusing solely on global semantic information while neglecting specific conditional elements in queries. Consequently, we propose Coral, a novel fine-tuning framework that adapts pre-trained MLLMs by integrating embedding reconstruction to preserve fine-grained conditional elements and contrastive learning to extract comprehensive global semantics. Experiments demonstrate that Coral achieves a 45.9% performance improvement over conventional approaches on MERIT, with strong generalization capabilities validated across 8 established retrieval benchmarks. Collectively, our contributions - a novel dataset, identification of critical limitations in existing approaches, and an innovative fine-tuning framework - establish a foundation for future research in interleaved multi-condition semantic retrieval.
>
---
#### [new 095] GeneA-SLAM2: Dynamic SLAM with AutoEncoder-Preprocessed Genetic Keypoints Resampling and Depth Variance-Guided Dynamic Region Removal
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于SLAM（同时定位与地图构建）任务，旨在解决动态环境中动态物体干扰定位的问题。通过引入深度方差约束和自编码器预处理，提出GeneA-SLAM2系统，实现更精准的动态区域剔除与关键点均匀采样，提升动态场景下的定位精度。**

- **链接: [http://arxiv.org/pdf/2506.02736v1](http://arxiv.org/pdf/2506.02736v1)**

> **作者:** Shufan Qing; Anzhen Li; Qiandi Wang; Yuefeng Niu; Mingchen Feng; Guoliang Hu; Jinqiao Wu; Fengtao Nan; Yingchun Fan
>
> **摘要:** Existing semantic SLAM in dynamic environments mainly identify dynamic regions through object detection or semantic segmentation methods. However, in certain highly dynamic scenarios, the detection boxes or segmentation masks cannot fully cover dynamic regions. Therefore, this paper proposes a robust and efficient GeneA-SLAM2 system that leverages depth variance constraints to handle dynamic scenes. Our method extracts dynamic pixels via depth variance and creates precise depth masks to guide the removal of dynamic objects. Simultaneously, an autoencoder is used to reconstruct keypoints, improving the genetic resampling keypoint algorithm to obtain more uniformly distributed keypoints and enhance the accuracy of pose estimation. Our system was evaluated on multiple highly dynamic sequences. The results demonstrate that GeneA-SLAM2 maintains high accuracy in dynamic scenes compared to current methods. Code is available at: https://github.com/qingshufan/GeneA-SLAM2.
>
---
#### [new 096] Random Registers for Cross-Domain Few-Shot Learning
- **分类: cs.CV**

- **简介: 该论文属于跨域少样本学习任务，旨在解决源域与目标域数据分布差异大时模型泛化能力不足的问题。作者发现随机初始化的寄存器（random registers）比可学习提示（prompt tuning）更有利于迁移性能，并提出一种增强注意力扰动的方法，提升模型在目标域的表现。**

- **链接: [http://arxiv.org/pdf/2506.02843v1](http://arxiv.org/pdf/2506.02843v1)**

> **作者:** Shuai Yi; Yixiong Zou; Yuhua Li; Ruixuan Li
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Cross-domain few-shot learning (CDFSL) aims to transfer knowledge from a data-sufficient source domain to data-scarce target domains. Although Vision Transformer (ViT) has shown superior capability in many vision tasks, its transferability against huge domain gaps in CDFSL is still under-explored. In this paper, we find an intriguing phenomenon: during the source-domain training, prompt tuning, as a common way to train ViT, could be harmful for the generalization of ViT in target domains, but setting them to random noises (i.e., random registers) could consistently improve target-domain performance. We then delve into this phenomenon for an interpretation. We find that learnable prompts capture domain information during the training on the source dataset, which views irrelevant visual patterns as vital cues for recognition. This can be viewed as a kind of overfitting and increases the sharpness of the loss landscapes. In contrast, random registers are essentially a novel way of perturbing attention for the sharpness-aware minimization, which helps the model find a flattened minimum in loss landscapes, increasing the transferability. Based on this phenomenon and interpretation, we further propose a simple but effective approach for CDFSL to enhance the perturbation on attention maps by adding random registers on the semantic regions of image tokens, improving the effectiveness and efficiency of random registers. Extensive experiments on four benchmarks validate our rationale and state-of-the-art performance. Codes and models are available at https://github.com/shuaiyi308/REAP.
>
---
#### [new 097] TIIF-Bench: How Does Your T2I Model Follow Your Instructions?
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决现有模型评估基准在提示多样性、复杂度及评价指标上的不足。作者构建了TIIF-Bench，包含5000个多维提示和100个高质量设计提示，分三类难度评估模型对文本指令的理解与执行能力，并提出新评价框架以揭示当前模型的局限性。**

- **链接: [http://arxiv.org/pdf/2506.02161v1](http://arxiv.org/pdf/2506.02161v1)**

> **作者:** Xinyu Wei; Jinrui Zhang; Zeqing Wang; Hongyang Wei; Zhen Guo; Lei Zhang
>
> **备注:** 23 pages, 12 figures, 11 tables
>
> **摘要:** The rapid advancements of Text-to-Image (T2I) models have ushered in a new phase of AI-generated content, marked by their growing ability to interpret and follow user instructions. However, existing T2I model evaluation benchmarks fall short in limited prompt diversity and complexity, as well as coarse evaluation metrics, making it difficult to evaluate the fine-grained alignment performance between textual instructions and generated images. In this paper, we present TIIF-Bench (Text-to-Image Instruction Following Benchmark), aiming to systematically assess T2I models' ability in interpreting and following intricate textual instructions. TIIF-Bench comprises a set of 5000 prompts organized along multiple dimensions, which are categorized into three levels of difficulties and complexities. To rigorously evaluate model robustness to varying prompt lengths, we provide a short and a long version for each prompt with identical core semantics. Two critical attributes, i.e., text rendering and style control, are introduced to evaluate the precision of text synthesis and the aesthetic coherence of T2I models. In addition, we collect 100 high-quality designer level prompts that encompass various scenarios to comprehensively assess model performance. Leveraging the world knowledge encoded in large vision language models, we propose a novel computable framework to discern subtle variations in T2I model outputs. Through meticulous benchmarking of mainstream T2I models on TIIF-Bench, we analyze the pros and cons of current T2I models and reveal the limitations of current T2I benchmarks. Project Page: https://a113n-w3i.github.io/TIIF_Bench/.
>
---
#### [new 098] Native-Resolution Image Synthesis
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像生成任务，旨在解决传统方法在固定分辨率和方形图像上的局限性。作者提出了一种原生分辨率建模方法，通过设计Native-resolution diffusion Transformer（NiT），实现对任意分辨率和宽高比的图像生成。实验表明，NiT在标准数据集上达到SOTA，并展现出强大的零样本泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.03131v1](http://arxiv.org/pdf/2506.03131v1)**

> **作者:** Zidong Wang; Lei Bai; Xiangyu Yue; Wanli Ouyang; Yiyuan Zhang
>
> **备注:** Project Page: https://wzdthu.github.io/NiT/
>
> **摘要:** We introduce native-resolution image synthesis, a novel generative modeling paradigm that enables the synthesis of images at arbitrary resolutions and aspect ratios. This approach overcomes the limitations of conventional fixed-resolution, square-image methods by natively handling variable-length visual tokens, a core challenge for traditional techniques. To this end, we introduce the Native-resolution diffusion Transformer (NiT), an architecture designed to explicitly model varying resolutions and aspect ratios within its denoising process. Free from the constraints of fixed formats, NiT learns intrinsic visual distributions from images spanning a broad range of resolutions and aspect ratios. Notably, a single NiT model simultaneously achieves the state-of-the-art performance on both ImageNet-256x256 and 512x512 benchmarks. Surprisingly, akin to the robust zero-shot capabilities seen in advanced large language models, NiT, trained solely on ImageNet, demonstrates excellent zero-shot generalization performance. It successfully generates high-fidelity images at previously unseen high resolutions (e.g., 1536 x 1536) and diverse aspect ratios (e.g., 16:9, 3:1, 4:3), as shown in Figure 1. These findings indicate the significant potential of native-resolution modeling as a bridge between visual generative modeling and advanced LLM methodologies.
>
---
#### [new 099] MemoryOut: Learning Principal Features via Multimodal Sparse Filtering Network for Semi-supervised Video Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于视频异常检测任务，旨在解决现有方法在区分正常与异常模式及捕捉高层语义上的不足。工作包括：提出稀疏特征过滤模块抑制过度泛化，设计专家混合架构提升特征多样性，并融合视觉-语言模型增强语义建模能力。**

- **链接: [http://arxiv.org/pdf/2506.02535v1](http://arxiv.org/pdf/2506.02535v1)**

> **作者:** Juntong Li; Lingwei Dang; Yukun Su; Yun Hao; Qingxin Xiao; Yongwei Nie; Qingyao Wu
>
> **摘要:** Video Anomaly Detection (VAD) methods based on reconstruction or prediction face two critical challenges: (1) strong generalization capability often results in accurate reconstruction or prediction of abnormal events, making it difficult to distinguish normal from abnormal patterns; (2) reliance only on low-level appearance and motion cues limits their ability to identify high-level semantic in abnormal events from complex scenes. To address these limitations, we propose a novel VAD framework with two key innovations. First, to suppress excessive generalization, we introduce the Sparse Feature Filtering Module (SFFM) that employs bottleneck filters to dynamically and adaptively remove abnormal information from features. Unlike traditional memory modules, it does not need to memorize the normal prototypes across the training dataset. Further, we design the Mixture of Experts (MoE) architecture for SFFM. Each expert is responsible for extracting specialized principal features during running time, and different experts are selectively activated to ensure the diversity of the learned principal features. Second, to overcome the neglect of semantics in existing methods, we integrate a Vision-Language Model (VLM) to generate textual descriptions for video clips, enabling comprehensive joint modeling of semantic, appearance, and motion cues. Additionally, we enforce modality consistency through semantic similarity constraints and motion frame-difference contrastive loss. Extensive experiments on multiple public datasets validate the effectiveness of our multimodal joint modeling framework and sparse feature filtering paradigm. Project page at https://qzfm.github.io/sfn_vad_project_page/.
>
---
#### [new 100] One-Step Diffusion-based Real-World Image Super-Resolution with Visual Perception Distillation
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率任务，旨在解决扩散模型在单步推理中语义对齐不足、感知质量低的问题。作者提出VPD-SR框架，包含语义监督和高频感知损失，并结合对抗训练，实现高效且高质量的图像重建。**

- **链接: [http://arxiv.org/pdf/2506.02605v1](http://arxiv.org/pdf/2506.02605v1)**

> **作者:** Xue Wu; Jingwei Xin; Zhijun Tu; Jie Hu; Jie Li; Nannan Wang; Xinbo Gao
>
> **摘要:** Diffusion-based models have been widely used in various visual generation tasks, showing promising results in image super-resolution (SR), while typically being limited by dozens or even hundreds of sampling steps. Although existing methods aim to accelerate the inference speed of multi-step diffusion-based SR methods through knowledge distillation, their generated images exhibit insufficient semantic alignment with real images, resulting in suboptimal perceptual quality reconstruction, specifically reflected in the CLIPIQA score. These methods still have many challenges in perceptual quality and semantic fidelity. Based on the challenges, we propose VPD-SR, a novel visual perception diffusion distillation framework specifically designed for SR, aiming to construct an effective and efficient one-step SR model. Specifically, VPD-SR consists of two components: Explicit Semantic-aware Supervision (ESS) and High-Frequency Perception (HFP) loss. Firstly, the ESS leverages the powerful visual perceptual understanding capabilities of the CLIP model to extract explicit semantic supervision, thereby enhancing semantic consistency. Then, Considering that high-frequency information contributes to the visual perception quality of images, in addition to the vanilla distillation loss, the HFP loss guides the student model to restore the missing high-frequency details in degraded images that are critical for enhancing perceptual quality. Lastly, we expand VPD-SR in adversarial training manner to further enhance the authenticity of the generated content. Extensive experiments conducted on synthetic and real-world datasets demonstrate that the proposed VPD-SR achieves superior performance compared to both previous state-of-the-art methods and the teacher model with just one-step sampling.
>
---
#### [new 101] A Dynamic Transformer Network for Vehicle Detection
- **分类: cs.CV**

- **简介: 该论文属于车辆检测任务，旨在提升复杂交通场景下的检测性能。针对现有方法在不同光照和遮挡条件下表现受限的问题，作者提出了动态Transformer网络（DTNet），引入动态卷积、混合注意力机制和位置相关卷积，以增强模型的适应性与检测精度。**

- **链接: [http://arxiv.org/pdf/2506.02765v1](http://arxiv.org/pdf/2506.02765v1)**

> **作者:** Chunwei Tian; Kai Liu; Bob Zhang; Zhixiang Huang; Chia-Wen Lin; David Zhang
>
> **备注:** 8 pages, 5 figures. This paper has been accepted for publication in IEEE Transactions on Consumer Electronics
>
> **摘要:** Stable consumer electronic systems can assist traffic better. Good traffic consumer electronic systems require collaborative work between traffic algorithms and hardware. However, performance of popular traffic algorithms containing vehicle detection methods based on deep networks via learning data relation rather than learning differences in different lighting and occlusions is limited. In this paper, we present a dynamic Transformer network for vehicle detection (DTNet). DTNet utilizes a dynamic convolution to guide a deep network to dynamically generate weights to enhance adaptability of an obtained detector. Taking into relations of different information account, a mixed attention mechanism based channel attention and Transformer is exploited to strengthen relations of channels and pixels to extract more salient information for vehicle detection. To overcome the drawback of difference in an image account, a translation-variant convolution relies on spatial location information to refine obtained structural information for vehicle detection. Experimental results illustrate that our DTNet is competitive for vehicle detection. Code of the proposed DTNet can be obtained at https://github.com/hellloxiaotian/DTNet.
>
---
#### [new 102] Hierarchical Question-Answering for Driving Scene Understanding Using Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 论文提出一种基于视觉-语言模型的分层问答方法，用于自动驾驶场景理解。任务是实现高效、细致的驾驶场景解析。通过构建问题树，结合模型推理与模板生成，平衡效率与细节，显著降低推理时间，并在自定义数据集上验证了有效性。**

- **链接: [http://arxiv.org/pdf/2506.02615v1](http://arxiv.org/pdf/2506.02615v1)**

> **作者:** Safaa Abdullahi Moallim Mohamud; Minjin Baek; Dong Seog Han
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** In this paper, we present a hierarchical question-answering (QA) approach for scene understanding in autonomous vehicles, balancing cost-efficiency with detailed visual interpretation. The method fine-tunes a compact vision-language model (VLM) on a custom dataset specific to the geographical area in which the vehicle operates to capture key driving-related visual elements. At the inference stage, the hierarchical QA strategy decomposes the scene understanding task into high-level and detailed sub-questions. Instead of generating lengthy descriptions, the VLM navigates a structured question tree, where answering high-level questions (e.g., "Is it possible for the ego vehicle to turn left at the intersection?") triggers more detailed sub-questions (e.g., "Is there a vehicle approaching the intersection from the opposite direction?"). To optimize inference time, questions are dynamically skipped based on previous answers, minimizing computational overhead. The extracted answers are then synthesized using handcrafted templates to ensure coherent, contextually accurate scene descriptions. We evaluate the proposed approach on the custom dataset using GPT reference-free scoring, demonstrating its competitiveness with state-of-the-art methods like GPT-4o in capturing key scene details while achieving significantly lower inference time. Moreover, qualitative results from real-time deployment highlight the proposed approach's capacity to capture key driving elements with minimal latency.
>
---
#### [new 103] Kernel-based Unsupervised Embedding Alignment for Enhanced Visual Representation in Vision-language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型任务，旨在解决CLIP模型在细粒度感知上的不足。通过提出一种基于核的无监督嵌入对齐方法，将CLIP与DINOv2的视觉表征进行对齐，在不破坏与文本编码器兼容性的前提下，提升了零样本识别、细粒度推理和定位能力，进而优化下游多模态模型表现。**

- **链接: [http://arxiv.org/pdf/2506.02557v1](http://arxiv.org/pdf/2506.02557v1)**

> **作者:** Shizhan Gong; Yankai Jiang; Qi Dou; Farzan Farnia
>
> **备注:** ICML 2025
>
> **摘要:** Vision-language models, such as CLIP, have achieved significant success in aligning visual and textual representations, becoming essential components of many multi-modal large language models (MLLMs) like LLaVA and OpenFlamingo. However, numerous studies have identified CLIP's limited fine-grained perception as a critical drawback, leading to substantial failures in downstream MLLMs. In contrast, vision-centric foundation models like DINOv2 demonstrate remarkable capabilities in capturing fine details from images. In this work, we propose a novel kernel-based method to align CLIP's visual representation with that of DINOv2, ensuring that the resulting embeddings maintain compatibility with text embeddings while enhancing perceptual capabilities. Our alignment objective is designed for efficient stochastic optimization. Following this image-only alignment fine-tuning, the visual encoder retains compatibility with the frozen text encoder and exhibits significant improvements in zero-shot object recognition, fine-grained spatial reasoning, and localization. By integrating the aligned visual encoder, downstream MLLMs also demonstrate enhanced performance.
>
---
#### [new 104] Probabilistic Online Event Downsampling
- **分类: cs.CV; cs.ET**

- **简介: 该论文属于事件相机数据处理任务，旨在解决高带宽和计算需求问题。作者提出POLED框架，通过概率密度函数在线评估事件重要性，实现自适应降采样，并引入零样本降采样概念，保持降采样后数据在多种任务上的可用性，如分类、检测等，提升性能与适用性。**

- **链接: [http://arxiv.org/pdf/2506.02547v1](http://arxiv.org/pdf/2506.02547v1)**

> **作者:** Andreu Girbau-Xalabarder; Jun Nagata; Shinichi Sumiyoshi
>
> **备注:** Accepted at CVPR 2025 Event-Vision workshop
>
> **摘要:** Event cameras capture scene changes asynchronously on a per-pixel basis, enabling extremely high temporal resolution. However, this advantage comes at the cost of high bandwidth, memory, and computational demands. To address this, prior work has explored event downsampling, but most approaches rely on fixed heuristics or threshold-based strategies, limiting their adaptability. Instead, we propose a probabilistic framework, POLED, that models event importance through an event-importance probability density function (ePDF), which can be arbitrarily defined and adapted to different applications. Our approach operates in a purely online setting, estimating event importance on-the-fly from raw event streams, enabling scene-specific adaptation. Additionally, we introduce zero-shot event downsampling, where downsampled events must remain usable for models trained on the original event stream, without task-specific adaptation. We design a contour-preserving ePDF that prioritizes structurally important events and evaluate our method across four datasets and tasks--object classification, image interpolation, surface normal estimation, and object detection--demonstrating that intelligent sampling is crucial for maintaining performance under event-budget constraints.
>
---
#### [new 105] Iterative Self-Improvement of Vision Language Models for Image Scoring and Self-Explanation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于图像评分与解释任务，旨在提升视觉语言模型在评分的同时生成可信的自然语言解释。通过自训练和偏好优化，结合两个数据集迭代训练，改善评分准确性和解释一致性。**

- **链接: [http://arxiv.org/pdf/2506.02708v1](http://arxiv.org/pdf/2506.02708v1)**

> **作者:** Naoto Tanji; Toshihiko Yamasaki
>
> **备注:** Accepted to ICIP2025
>
> **摘要:** Image scoring is a crucial task in numerous real-world applications. To trust a model's judgment, understanding its rationale is essential. This paper proposes a novel training method for Vision Language Models (VLMs) to generate not only image scores but also corresponding justifications in natural language. Leveraging only an image scoring dataset and an instruction-tuned VLM, our method enables self-training, utilizing the VLM's generated text without relying on external data or models. In addition, we introduce a simple method for creating a dataset designed to improve alignment between predicted scores and their textual justifications. By iteratively training the model with Direct Preference Optimization on two distinct datasets and merging them, we can improve both scoring accuracy and the coherence of generated explanations.
>
---
#### [new 106] Rig3R: Rig-Aware Conditioning for Learned 3D Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D重建与姿态估计任务，旨在解决多相机 rig 场景中结构与姿态联合估计的问题。作者提出了 Rig3R 模型，通过引入 rig 感知的条件机制，在有或无 rig 元数据的情况下均能实现高效 3D 重建和 rig 结构推断，提升了性能并简化流程。**

- **链接: [http://arxiv.org/pdf/2506.02265v1](http://arxiv.org/pdf/2506.02265v1)**

> **作者:** Samuel Li; Pujith Kachana; Prajwal Chidananda; Saurabh Nair; Yasutaka Furukawa; Matthew Brown
>
> **摘要:** Estimating agent pose and 3D scene structure from multi-camera rigs is a central task in embodied AI applications such as autonomous driving. Recent learned approaches such as DUSt3R have shown impressive performance in multiview settings. However, these models treat images as unstructured collections, limiting effectiveness in scenarios where frames are captured from synchronized rigs with known or inferable structure. To this end, we introduce Rig3R, a generalization of prior multiview reconstruction models that incorporates rig structure when available, and learns to infer it when not. Rig3R conditions on optional rig metadata including camera ID, time, and rig poses to develop a rig-aware latent space that remains robust to missing information. It jointly predicts pointmaps and two types of raymaps: a pose raymap relative to a global frame, and a rig raymap relative to a rig-centric frame consistent across time. Rig raymaps allow the model to infer rig structure directly from input images when metadata is missing. Rig3R achieves state-of-the-art performance in 3D reconstruction, camera pose estimation, and rig discovery, outperforming both traditional and learned methods by 17-45% mAA across diverse real-world rig datasets, all in a single forward pass without post-processing or iterative refinement.
>
---
#### [new 107] InterMamba: Efficient Human-Human Interaction Generation with Adaptive Spatio-Temporal Mamba
- **分类: cs.CV**

- **简介: 该论文属于人机交互生成任务，旨在高效生成人类互动动作。现有方法基于Transformer，效率低。作者提出InterMamba，采用Mamba框架，引入自适应时空模块，提升长序列建模效率。实验表明，其性能优越，参数更少、速度更快。**

- **链接: [http://arxiv.org/pdf/2506.03084v1](http://arxiv.org/pdf/2506.03084v1)**

> **作者:** Zizhao Wu; Yingying Sun; Yiming Chen; Xiaoling Gu; Ruyu Liu; Jiazhou Chen
>
> **摘要:** Human-human interaction generation has garnered significant attention in motion synthesis due to its vital role in understanding humans as social beings. However, existing methods typically rely on transformer-based architectures, which often face challenges related to scalability and efficiency. To address these issues, we propose a novel, efficient human-human interaction generation method based on the Mamba framework, designed to meet the demands of effectively capturing long-sequence dependencies while providing real-time feedback. Specifically, we introduce an adaptive spatio-temporal Mamba framework that utilizes two parallel SSM branches with an adaptive mechanism to integrate the spatial and temporal features of motion sequences. To further enhance the model's ability to capture dependencies within individual motion sequences and the interactions between different individual sequences, we develop two key modules: the self-adaptive spatio-temporal Mamba module and the cross-adaptive spatio-temporal Mamba module, enabling efficient feature learning. Extensive experiments demonstrate that our method achieves state-of-the-art results on two interaction datasets with remarkable quality and efficiency. Compared to the baseline method InterGen, our approach not only improves accuracy but also requires a minimal parameter size of just 66M ,only 36% of InterGen's, while achieving an average inference speed of 0.57 seconds, which is 46% of InterGen's execution time.
>
---
#### [new 108] Targeted Forgetting of Image Subgroups in CLIP Models
- **分类: cs.CV**

- **简介: 该论文属于模型遗忘任务，旨在解决CLIP模型中细粒度知识遗忘的问题。现有方法依赖预训练数据或仅支持粗粒度遗忘，而本文提出一种三阶段方法，在不访问预训练数据的情况下，选择性遗忘特定图像子组，同时保持模型整体性能和零样本能力。**

- **链接: [http://arxiv.org/pdf/2506.03117v1](http://arxiv.org/pdf/2506.03117v1)**

> **作者:** Zeliang Zhang; Gaowen Liu; Charles Fleming; Ramana Rao Kompella; Chenliang Xu
>
> **备注:** 12 Figures,5 Pages. The project page is \url{https://zhangaipi.github.io/forget_clip/}
>
> **摘要:** Foundation models (FMs) such as CLIP have demonstrated impressive zero-shot performance across various tasks by leveraging large-scale, unsupervised pre-training. However, they often inherit harmful or unwanted knowledge from noisy internet-sourced datasets, compromising their reliability in real-world applications. Existing model unlearning methods either rely on access to pre-trained datasets or focus on coarse-grained unlearning (e.g., entire classes), leaving a critical gap for fine-grained unlearning. In this paper, we address the challenging scenario of selectively forgetting specific portions of knowledge within a class, without access to pre-trained data, while preserving the model's overall performance. We propose a novel three-stage approach that progressively unlearns targeted knowledge while mitigating over-forgetting. It consists of (1) a forgetting stage to fine-tune the CLIP on samples to be forgotten, (2) a reminding stage to restore performance on retained samples, and (3) a restoring stage to recover zero-shot capabilities using model souping. Additionally, we introduce knowledge distillation to handle the distribution disparity between forgetting, retaining samples, and unseen pre-trained data. Extensive experiments on CIFAR-10, ImageNet-1K, and style datasets demonstrate that our approach effectively unlearns specific subgroups while maintaining strong zero-shot performance on semantically similar subgroups and other categories, significantly outperforming baseline unlearning methods, which lose effectiveness under the CLIP unlearning setting.
>
---
#### [new 109] Revisiting Continuity of Image Tokens for Cross-domain Few-shot Learning
- **分类: cs.CV**

- **简介: 该论文属于跨域少样本学习任务，旨在解决视觉Transformer在跨域且数据稀缺情况下性能下降的问题。通过分析图像块连续性对模型泛化的影响，发现破坏连续性可减少对大模式的依赖，提升小模式迁移效果。据此提出新方法，在目标域中更有效地缩小领域差距，取得优于现有方法的表现。**

- **链接: [http://arxiv.org/pdf/2506.03110v1](http://arxiv.org/pdf/2506.03110v1)**

> **作者:** Shuai Yi; Yixiong Zou; Yuhua Li; Ruixuan Li
>
> **备注:** Accepted by ICML 2025(spotlight)
>
> **摘要:** Vision Transformer (ViT) has achieved remarkable success due to its large-scale pretraining on general domains, but it still faces challenges when applying it to downstream distant domains that have only scarce training data, which gives rise to the Cross-Domain Few-Shot Learning (CDFSL) task. Inspired by Self-Attention's insensitivity to token orders, we find an interesting phenomenon neglected in current works: disrupting the continuity of image tokens (i.e., making pixels not smoothly transited across patches) in ViT leads to a noticeable performance decline in the general (source) domain but only a marginal decrease in downstream target domains. This questions the role of image tokens' continuity in ViT's generalization under large domain gaps. In this paper, we delve into this phenomenon for an interpretation. We find continuity aids ViT in learning larger spatial patterns, which are harder to transfer than smaller ones, enlarging domain distances. Meanwhile, it implies that only smaller patterns within each patch could be transferred under extreme domain gaps. Based on this interpretation, we further propose a simple yet effective method for CDFSL that better disrupts the continuity of image tokens, encouraging the model to rely less on large patterns and more on smaller ones. Extensive experiments show the effectiveness of our method in reducing domain gaps and outperforming state-of-the-art works. Codes and models are available at https://github.com/shuaiyi308/ReCIT.
>
---
#### [new 110] FlySearch: Exploring how vision-language models explore
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于视觉-语言模型（VLM）任务，旨在解决复杂场景中目标搜索与导航问题。作者提出FlySearch环境，评估VLM在不同难度任务中的表现，发现其在主动探索任务中效果不佳，主要问题包括视觉幻觉、上下文误解和任务规划失败，并尝试通过微调改进。**

- **链接: [http://arxiv.org/pdf/2506.02896v1](http://arxiv.org/pdf/2506.02896v1)**

> **作者:** Adam Pardyl; Dominik Matuszek; Mateusz Przebieracz; Marek Cygan; Bartosz Zieliński; Maciej Wołczyk
>
> **摘要:** The real world is messy and unstructured. Uncovering critical information often requires active, goal-driven exploration. It remains to be seen whether Vision-Language Models (VLMs), which recently emerged as a popular zero-shot tool in many difficult tasks, can operate effectively in such conditions. In this paper, we answer this question by introducing FlySearch, a 3D, outdoor, photorealistic environment for searching and navigating to objects in complex scenes. We define three sets of scenarios with varying difficulty and observe that state-of-the-art VLMs cannot reliably solve even the simplest exploration tasks, with the gap to human performance increasing as the tasks get harder. We identify a set of central causes, ranging from vision hallucination, through context misunderstanding, to task planning failures, and we show that some of them can be addressed by finetuning. We publicly release the benchmark, scenarios, and the underlying codebase.
>
---
#### [new 111] Unified Attention Modeling for Efficient Free-Viewing and Visual Search via Shared Representations
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算视觉注意力建模任务，旨在探索自由观看与任务驱动视觉搜索之间是否存在共享表征。作者基于HAT模型设计统一架构，验证其可迁移性。结果表明，自由观看训练的模型能高效迁移到视觉搜索任务，性能下降仅3.86%，同时降低92.29%计算量与31.23%参数量。**

- **链接: [http://arxiv.org/pdf/2506.02764v1](http://arxiv.org/pdf/2506.02764v1)**

> **作者:** Fatma Youssef Mohammed; Kostas Alexis
>
> **备注:** Accepted to the 2025 IEEE International Conference on Development and Learning (ICDL)
>
> **摘要:** Computational human attention modeling in free-viewing and task-specific settings is often studied separately, with limited exploration of whether a common representation exists between them. This work investigates this question and proposes a neural network architecture that builds upon the Human Attention transformer (HAT) to test the hypothesis. Our results demonstrate that free-viewing and visual search can efficiently share a common representation, allowing a model trained in free-viewing attention to transfer its knowledge to task-driven visual search with a performance drop of only 3.86% in the predicted fixation scanpaths, measured by the semantic sequence score (SemSS) metric which reflects the similarity between predicted and human scanpaths. This transfer reduces computational costs by 92.29% in terms of GFLOPs and 31.23% in terms of trainable parameters.
>
---
#### [new 112] ReSpace: Text-Driven 3D Scene Synthesis and Editing with Preference Alignment
- **分类: cs.CV; I.2.10; I.2.7**

- **简介: 该论文属于3D室内场景生成与编辑任务，旨在解决现有方法在语义表达、布局复杂度和编辑灵活性上的不足。作者提出了ReSpace框架，利用自回归语言模型实现文本驱动的场景生成与编辑，并引入结构化表示和双阶段训练策略，优化物体添加与整体布局。**

- **链接: [http://arxiv.org/pdf/2506.02459v1](http://arxiv.org/pdf/2506.02459v1)**

> **作者:** Martin JJ. Bucher; Iro Armeni
>
> **备注:** 20 pages, 17 figures (incl. appendix)
>
> **摘要:** Scene synthesis and editing has emerged as a promising direction in computer graphics. Current trained approaches for 3D indoor scenes either oversimplify object semantics through one-hot class encodings (e.g., 'chair' or 'table'), require masked diffusion for editing, ignore room boundaries, or rely on floor plan renderings that fail to capture complex layouts. In contrast, LLM-based methods enable richer semantics via natural language (e.g., 'modern studio with light wood furniture') but do not support editing, remain limited to rectangular layouts or rely on weak spatial reasoning from implicit world models. We introduce ReSpace, a generative framework for text-driven 3D indoor scene synthesis and editing using autoregressive language models. Our approach features a compact structured scene representation with explicit room boundaries that frames scene editing as a next-token prediction task. We leverage a dual-stage training approach combining supervised fine-tuning and preference alignment, enabling a specially trained language model for object addition that accounts for user instructions, spatial geometry, object semantics, and scene-level composition. For scene editing, we employ a zero-shot LLM to handle object removal and prompts for addition. We further introduce a novel voxelization-based evaluation that captures fine-grained geometry beyond 3D bounding boxes. Experimental results surpass state-of-the-art on object addition while maintaining competitive results on full scene synthesis.
>
---
#### [new 113] VTGaussian-SLAM: RGBD SLAM for Large Scale Scenes with Splatting View-Tied 3D Gaussians
- **分类: cs.CV**

- **简介: 该论文属于RGBD SLAM任务，旨在解决现有基于3D高斯表示的方法在大规模场景中因GPU内存限制导致的扩展性差问题。作者提出了一种新的3D表示方法——视图绑定的3D高斯（view-tied 3D Gaussians），通过将高斯分布与深度像素绑定，简化了其位置、旋转和方差的学习过程，减少了存储需求，从而在有限内存下支持更多高斯以提升局部细节表达。此外，他们设计了新的跟踪与建图策略，避免全程优化所有高斯参数，提高了渲染质量和定位精度，实现了更优的大规模场景重建效果。**

- **链接: [http://arxiv.org/pdf/2506.02741v1](http://arxiv.org/pdf/2506.02741v1)**

> **作者:** Pengchong Hu; Zhizhong Han
>
> **备注:** ICML 2025
>
> **摘要:** Jointly estimating camera poses and mapping scenes from RGBD images is a fundamental task in simultaneous localization and mapping (SLAM). State-of-the-art methods employ 3D Gaussians to represent a scene, and render these Gaussians through splatting for higher efficiency and better rendering. However, these methods cannot scale up to extremely large scenes, due to the inefficient tracking and mapping strategies that need to optimize all 3D Gaussians in the limited GPU memories throughout the training to maintain the geometry and color consistency to previous RGBD observations. To resolve this issue, we propose novel tracking and mapping strategies to work with a novel 3D representation, dubbed view-tied 3D Gaussians, for RGBD SLAM systems. View-tied 3D Gaussians is a kind of simplified Gaussians, which is tied to depth pixels, without needing to learn locations, rotations, and multi-dimensional variances. Tying Gaussians to views not only significantly saves storage but also allows us to employ many more Gaussians to represent local details in the limited GPU memory. Moreover, our strategies remove the need of maintaining all Gaussians learnable throughout the training, while improving rendering quality, and tracking accuracy. We justify the effectiveness of these designs, and report better performance over the latest methods on the widely used benchmarks in terms of rendering and tracking accuracy and scalability. Please see our project page for code and videos at https://machineperceptionlab.github.io/VTGaussian-SLAM-Project .
>
---
#### [new 114] Self-Disentanglement and Re-Composition for Cross-Domain Few-Shot Segmentation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文属于跨域少样本分割任务，旨在解决知识迁移中的特征纠缠问题。现有方法因平等对待所有特征比较而影响性能。作者通过分析ViT结构发现该问题，并提出加权比较策略以解耦特征并重组，提升模型泛化与微调效果。实验表明新方法在1-shot和5-shot设置下均优于当前最优方法。**

- **链接: [http://arxiv.org/pdf/2506.02677v1](http://arxiv.org/pdf/2506.02677v1)**

> **作者:** Jintao Tong; Yixiong Zou; Guangyao Chen; Yuhua Li; Ruixuan Li
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Cross-Domain Few-Shot Segmentation (CD-FSS) aims to transfer knowledge from a source-domain dataset to unseen target-domain datasets with limited annotations. Current methods typically compare the distance between training and testing samples for mask prediction. However, we find an entanglement problem exists in this widely adopted method, which tends to bind sourcedomain patterns together and make each of them hard to transfer. In this paper, we aim to address this problem for the CD-FSS task. We first find a natural decomposition of the ViT structure, based on which we delve into the entanglement problem for an interpretation. We find the decomposed ViT components are crossly compared between images in distance calculation, where the rational comparisons are entangled with those meaningless ones by their equal importance, leading to the entanglement problem. Based on this interpretation, we further propose to address the entanglement problem by learning to weigh for all comparisons of ViT components, which learn disentangled features and re-compose them for the CD-FSS task, benefiting both the generalization and finetuning. Experiments show that our model outperforms the state-of-the-art CD-FSS method by 1.92% and 1.88% in average accuracy under 1-shot and 5-shot settings, respectively.
>
---
#### [new 115] Hierarchical Self-Prompting SAM: A Prompt-Free Medical Image Segmentation Framework
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决SAM模型在无手动提示下的分割难题。作者提出HSP-SAM框架，通过引入抽象提示学习实现自提示机制，摆脱了对位置提示的依赖，提升了模型在多种医学图像上的分割性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.02854v1](http://arxiv.org/pdf/2506.02854v1)**

> **作者:** Mengmeng Zhang; Xingyuan Dai; Yicheng Sun; Jing Wang; Yueyang Yao; Xiaoyan Gong; Fuze Cong; Feiyue Wang; Yisheng Lv
>
> **摘要:** Although the Segment Anything Model (SAM) is highly effective in natural image segmentation, it requires dependencies on prompts, which limits its applicability to medical imaging where manual prompts are often unavailable. Existing efforts to fine-tune SAM for medical segmentation typically struggle to remove this dependency. We propose Hierarchical Self-Prompting SAM (HSP-SAM), a novel self-prompting framework that enables SAM to achieve strong performance in prompt-free medical image segmentation. Unlike previous self-prompting methods that remain limited to positional prompts similar to vanilla SAM, we are the first to introduce learning abstract prompts during the self-prompting process. This simple and intuitive self-prompting framework achieves superior performance on classic segmentation tasks such as polyp and skin lesion segmentation, while maintaining robustness across diverse medical imaging modalities. Furthermore, it exhibits strong generalization to unseen datasets, achieving improvements of up to 14.04% over previous state-of-the-art methods on some challenging benchmarks. These results suggest that abstract prompts encapsulate richer and higher-dimensional semantic information compared to positional prompts, thereby enhancing the model's robustness and generalization performance. All models and codes will be released upon acceptance.
>
---
#### [new 116] SAB3R: Semantic-Augmented Backbone in 3D Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出新任务“Map and Locate”，统一开放词汇分割与3D重建目标，旨在从无姿态视频生成点云并根据语言查询分割物体。为解决此任务，作者基于MASt3R模型，引入SAB3R方法，通过轻量级蒸馏策略融合2D视觉骨干（如CLIP、DINOv2）的语义特征，在单次前向传播中实现高效语义分割与3D重建。**

- **链接: [http://arxiv.org/pdf/2506.02112v1](http://arxiv.org/pdf/2506.02112v1)**

> **作者:** Xuweiyi Chen; Tian Xia; Sihan Xu; Jianing Yang; Joyce Chai; Zezhou Cheng
>
> **备注:** Project page: https://uva-computer-vision-lab.github.io/sab3r/
>
> **摘要:** We introduce a new task, Map and Locate, which unifies the traditionally distinct objectives of open-vocabulary segmentation - detecting and segmenting object instances based on natural language queries - and 3D reconstruction, the process of estimating a scene's 3D structure from visual inputs. Specifically, Map and Locate involves generating a point cloud from an unposed video and segmenting object instances based on open-vocabulary queries. This task serves as a critical step toward real-world embodied AI applications and introduces a practical task that bridges reconstruction, recognition and reorganization. To tackle this task, we introduce a simple yet effective baseline, which we denote as SAB3R. Our approach builds upon MASt3R, a recent breakthrough in 3D computer vision, and incorporates a lightweight distillation strategy. This method transfers dense, per-pixel semantic features from 2D vision backbones (eg, CLIP and DINOv2) to enhance MASt3R's capabilities. Without introducing any auxiliary frozen networks, our model generates per-pixel semantic features and constructs cohesive point maps in a single forward pass. Compared to separately deploying MASt3R and CLIP, our unified model, SAB3R, achieves superior performance on the Map and Locate benchmark. Furthermore, we evaluate SAB3R on both 2D semantic segmentation and 3D tasks to comprehensively validate its effectiveness.
>
---
#### [new 117] RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS
- **分类: cs.CV**

- **简介: 该论文属于3D高斯泼溅（3DGS）任务，旨在解决动态物体干扰导致的渲染伪影问题。作者提出RobustSplat方法，通过延迟高斯增长策略和级联掩码引导机制，有效区分静态结构与动态干扰，提升重建鲁棒性与渲染质量。**

- **链接: [http://arxiv.org/pdf/2506.02751v1](http://arxiv.org/pdf/2506.02751v1)**

> **作者:** Chuanyu Fu; Yuqi Zhang; Kunbin Yao; Guanying Chen; Yuan Xiong; Chuan Huang; Shuguang Cui; Xiaochun Cao
>
> **备注:** Project page: https://fcyycf.github.io/RobustSplat/
>
> **摘要:** 3D Gaussian Splatting (3DGS) has gained significant attention for its real-time, photo-realistic rendering in novel-view synthesis and 3D modeling. However, existing methods struggle with accurately modeling scenes affected by transient objects, leading to artifacts in the rendered images. We identify that the Gaussian densification process, while enhancing scene detail capture, unintentionally contributes to these artifacts by growing additional Gaussians that model transient disturbances. To address this, we propose RobustSplat, a robust solution based on two critical designs. First, we introduce a delayed Gaussian growth strategy that prioritizes optimizing static scene structure before allowing Gaussian splitting/cloning, mitigating overfitting to transient objects in early optimization. Second, we design a scale-cascaded mask bootstrapping approach that first leverages lower-resolution feature similarity supervision for reliable initial transient mask estimation, taking advantage of its stronger semantic consistency and robustness to noise, and then progresses to high-resolution supervision to achieve more precise mask prediction. Extensive experiments on multiple challenging datasets show that our method outperforms existing methods, clearly demonstrating the robustness and effectiveness of our method. Our project page is https://fcyycf.github.io/RobustSplat/.
>
---
#### [new 118] Self-Supervised Spatial Correspondence Across Modalities
- **分类: cs.CV**

- **简介: 该论文属于跨模态图像匹配任务，旨在解决不同视觉模态（如RGB、深度、热成像）间的像素级空间对应问题。作者提出一种自监督方法，扩展对比随机游走框架，学习循环一致的特征表示，无需标注数据或对齐图像。**

- **链接: [http://arxiv.org/pdf/2506.03148v1](http://arxiv.org/pdf/2506.03148v1)**

> **作者:** Ayush Shrivastava; Andrew Owens
>
> **备注:** CVPR 2025. Project link: https://www.ayshrv.com/cmrw . Code: https://github.com/ayshrv/cmrw
>
> **摘要:** We present a method for finding cross-modal space-time correspondences. Given two images from different visual modalities, such as an RGB image and a depth map, our model identifies which pairs of pixels correspond to the same physical points in the scene. To solve this problem, we extend the contrastive random walk framework to simultaneously learn cycle-consistent feature representations for both cross-modal and intra-modal matching. The resulting model is simple and has no explicit photo-consistency assumptions. It can be trained entirely using unlabeled data, without the need for any spatially aligned multimodal image pairs. We evaluate our method on both geometric and semantic correspondence tasks. For geometric matching, we consider challenging tasks such as RGB-to-depth and RGB-to-thermal matching (and vice versa); for semantic matching, we evaluate on photo-sketch and cross-style image alignment. Our method achieves strong performance across all benchmarks.
>
---
#### [new 119] LinkTo-Anime: A 2D Animation Optical Flow Dataset from 3D Model Rendering
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉任务，旨在解决现有光流数据集不适用于二维动画角色运动的问题。论文构建了首个针对cel动画角色运动的高质量光流数据集LinkTo-Anime，包含丰富标注，并提供了多种光流估计方法的综合基准。**

- **链接: [http://arxiv.org/pdf/2506.02733v1](http://arxiv.org/pdf/2506.02733v1)**

> **作者:** Xiaoyi Feng; Kaifeng Zou; Caichun Cen; Tao Huang; Hui Guo; Zizhou Huang; Yingli Zhao; Mingqing Zhang; Diwei Wang; Yuntao Zou; Dagang Li
>
> **摘要:** Existing optical flow datasets focus primarily on real-world simulation or synthetic human motion, but few are tailored to Celluloid(cel) anime character motion: a domain with unique visual and motion characteristics. To bridge this gap and facilitate research in optical flow estimation and downstream tasks such as anime video generation and line drawing colorization, we introduce LinkTo-Anime, the first high-quality dataset specifically designed for cel anime character motion generated with 3D model rendering. LinkTo-Anime provides rich annotations including forward and backward optical flow, occlusion masks, and Mixamo Skeleton. The dataset comprises 395 video sequences, totally 24,230 training frames, 720 validation frames, and 4,320 test frames. Furthermore, a comprehensive benchmark is constructed with various optical flow estimation methods to analyze the shortcomings and limitations across multiple datasets.
>
---
#### [new 120] VisuRiddles: Fine-grained Perception is a Primary Bottleneck for Multimodal Large Language Models in Abstract Visual Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态大语言模型的抽象视觉推理任务，旨在解决其在感知抽象图形上的瓶颈问题。作者提出了VisuRiddles基准和Perceptual Riddle Synthesizer（PRS）框架，通过合成具有细粒度感知描述的谜题，提升模型的抽象视觉理解和推理能力。**

- **链接: [http://arxiv.org/pdf/2506.02537v1](http://arxiv.org/pdf/2506.02537v1)**

> **作者:** Hao Yan; Handong Zheng; Hao Wang; Liang Yin; Xingchen Liu; Zhenbiao Cao; Xinxing Su; Zihao Chen; Jihao Wu; Minghui Liao; Chao Weng; Wei Chen; Yuliang Liu; Xiang Bai
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** Recent strides in multimodal large language models (MLLMs) have significantly advanced their performance in many reasoning tasks. However, Abstract Visual Reasoning (AVR) remains a critical challenge, primarily due to limitations in perceiving abstract graphics. To tackle this issue, we investigate the bottlenecks in current MLLMs and synthesize training data to improve their abstract visual perception. First, we propose VisuRiddles, a benchmark for AVR, featuring tasks meticulously constructed to assess models' reasoning capacities across five core dimensions and two high-level reasoning categories. Second, we introduce the Perceptual Riddle Synthesizer (PRS), an automated framework for generating riddles with fine-grained perceptual descriptions. PRS not only generates valuable training data for abstract graphics but also provides fine-grained perceptual description, crucially allowing for supervision over intermediate reasoning stages and thereby improving both training efficacy and model interpretability. Our extensive experimental results on VisuRiddles empirically validate that fine-grained visual perception is the principal bottleneck and our synthesis framework markedly enhances the performance of contemporary MLLMs on these challenging tasks. Our code and dataset will be released at https://github.com/yh-hust/VisuRiddles
>
---
#### [new 121] DFBench: Benchmarking Deepfake Image Detection Capability of Large Multimodal Models
- **分类: cs.CV**

- **简介: 该论文属于图像真实性验证任务，旨在解决当前深度伪造检测方法无法应对多样化AI生成内容的问题。作者构建了大规模基准DFBench，包含54万张真实与AI生成图像，并提出MoA-DF方法，结合多个大模型提升检测性能。**

- **链接: [http://arxiv.org/pdf/2506.03007v1](http://arxiv.org/pdf/2506.03007v1)**

> **作者:** Jiarui Wang; Huiyu Duan; Juntong Wang; Ziheng Jia; Woo Yi Yang; Xiaorong Zhu; Yu Zhao; Jiaying Qian; Yuke Xing; Guangtao Zhai; Xiongkuo Min
>
> **摘要:** With the rapid advancement of generative models, the realism of AI-generated images has significantly improved, posing critical challenges for verifying digital content authenticity. Current deepfake detection methods often depend on datasets with limited generation models and content diversity that fail to keep pace with the evolving complexity and increasing realism of the AI-generated content. Large multimodal models (LMMs), widely adopted in various vision tasks, have demonstrated strong zero-shot capabilities, yet their potential in deepfake detection remains largely unexplored. To bridge this gap, we present \textbf{DFBench}, a large-scale DeepFake Benchmark featuring (i) broad diversity, including 540,000 images across real, AI-edited, and AI-generated content, (ii) latest model, the fake images are generated by 12 state-of-the-art generation models, and (iii) bidirectional benchmarking and evaluating for both the detection accuracy of deepfake detectors and the evasion capability of generative models. Based on DFBench, we propose \textbf{MoA-DF}, Mixture of Agents for DeepFake detection, leveraging a combined probability strategy from multiple LMMs. MoA-DF achieves state-of-the-art performance, further proving the effectiveness of leveraging LMMs for deepfake detection. Database and codes are publicly available at https://github.com/IntMeGroup/DFBench.
>
---
#### [new 122] BEVCALIB: LiDAR-Camera Calibration via Geometry-Guided Bird's-Eye View Representations
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶中的多模态感知融合任务，旨在解决激光雷达（LiDAR）与相机之间的标定问题。现有方法依赖大量控制环境数据且无法适应运动中变化。论文提出BEVCALIB模型，首次利用鸟瞰图（BEV）特征从原始数据进行标定，并引入特征选择机制提升效率。在多个数据集上表现优越，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.02587v1](http://arxiv.org/pdf/2506.02587v1)**

> **作者:** Weiduo Yuan; Jerry Li; Justin Yue; Divyank Shah; Konstantinos Karydis; Hang Qiu
>
> **摘要:** Accurate LiDAR-camera calibration is fundamental to fusing multi-modal perception in autonomous driving and robotic systems. Traditional calibration methods require extensive data collection in controlled environments and cannot compensate for the transformation changes during the vehicle/robot movement. In this paper, we propose the first model that uses bird's-eye view (BEV) features to perform LiDAR camera calibration from raw data, termed BEVCALIB. To achieve this, we extract camera BEV features and LiDAR BEV features separately and fuse them into a shared BEV feature space. To fully utilize the geometric information from the BEV feature, we introduce a novel feature selector to filter the most important features in the transformation decoder, which reduces memory consumption and enables efficient training. Extensive evaluations on KITTI, NuScenes, and our own dataset demonstrate that BEVCALIB establishes a new state of the art. Under various noise conditions, BEVCALIB outperforms the best baseline in the literature by an average of (47.08%, 82.32%) on KITTI dataset, and (78.17%, 68.29%) on NuScenes dataset, in terms of (translation, rotation), respectively. In the open-source domain, it improves the best reproducible baseline by one order of magnitude. Our code and demo results are available at https://cisl.ucr.edu/BEVCalib.
>
---
#### [new 123] PAIR-Net: Enhancing Egocentric Speaker Detection via Pretrained Audio-Visual Fusion and Alignment Loss
- **分类: cs.CV**

- **简介: 该论文属于主动说话人检测任务，旨在解决第一视角视频中因视角不稳定、运动模糊等问题导致的检测困难。论文提出PAIR-Net模型，融合预训练音频编码器与视觉主干网络，并引入对齐损失优化多模态融合，在Ego4D数据集上取得最优性能。**

- **链接: [http://arxiv.org/pdf/2506.02247v1](http://arxiv.org/pdf/2506.02247v1)**

> **作者:** Yu Wang; Juhyung Ha; David J. Crandall
>
> **备注:** 4 pages, 1 figure, and 1 table
>
> **摘要:** Active speaker detection (ASD) in egocentric videos presents unique challenges due to unstable viewpoints, motion blur, and off-screen speech sources - conditions under which traditional visual-centric methods degrade significantly. We introduce PAIR-Net (Pretrained Audio-Visual Integration with Regularization Network), an effective model that integrates a partially frozen Whisper audio encoder with a fine-tuned AV-HuBERT visual backbone to robustly fuse cross-modal cues. To counteract modality imbalance, we introduce an inter-modal alignment loss that synchronizes audio and visual representations, enabling more consistent convergence across modalities. Without relying on multi-speaker context or ideal frontal views, PAIR-Net achieves state-of-the-art performance on the Ego4D ASD benchmark with 76.6% mAP, surpassing LoCoNet and STHG by 8.2% and 12.9% mAP, respectively. Our results highlight the value of pretrained audio priors and alignment-based fusion for robust ASD under real-world egocentric conditions.
>
---
#### [new 124] SG2VID: Scene Graphs Enable Fine-Grained Control for Video Synthesis
- **分类: cs.CV**

- **简介: 该论文属于视频合成任务，旨在解决手术模拟中缺乏真实感和可控性的问题。作者提出SG2VID模型，利用场景图实现细粒度控制，生成高质量手术视频，并验证了其在数据增强和阶段检测中的应用效果。**

- **链接: [http://arxiv.org/pdf/2506.03082v1](http://arxiv.org/pdf/2506.03082v1)**

> **作者:** Ssharvien Kumar Sivakumar; Yannik Frisch; Ghazal Ghazaei; Anirban Mukhopadhyay
>
> **摘要:** Surgical simulation plays a pivotal role in training novice surgeons, accelerating their learning curve and reducing intra-operative errors. However, conventional simulation tools fall short in providing the necessary photorealism and the variability of human anatomy. In response, current methods are shifting towards generative model-based simulators. Yet, these approaches primarily focus on using increasingly complex conditioning for precise synthesis while neglecting the fine-grained human control aspect. To address this gap, we introduce SG2VID, the first diffusion-based video model that leverages Scene Graphs for both precise video synthesis and fine-grained human control. We demonstrate SG2VID's capabilities across three public datasets featuring cataract and cholecystectomy surgery. While SG2VID outperforms previous methods both qualitatively and quantitatively, it also enables precise synthesis, providing accurate control over tool and anatomy's size and movement, entrance of new tools, as well as the overall scene layout. We qualitatively motivate how SG2VID can be used for generative augmentation and present an experiment demonstrating its ability to improve a downstream phase detection task when the training set is extended with our synthetic videos. Finally, to showcase SG2VID's ability to retain human control, we interact with the Scene Graphs to generate new video samples depicting major yet rare intra-operative irregularities.
>
---
#### [new 125] Open-PMC-18M: A High-Fidelity Large Scale Medical Dataset for Multimodal Representation Learning
- **分类: cs.CV**

- **简介: 该论文属于医学图像与文本表示学习任务，旨在解决生物医学文献中复合图像的子图提取难题。作者提出了一种基于Transformer的可扩展子图提取方法，在大规模合成数据上训练并实现先进性能。据此构建了Open-PMC-18M数据集，包含1800万对高质量医学图像和文本对，提升了跨模态模型在检索、零样本分类等任务上的表现。**

- **链接: [http://arxiv.org/pdf/2506.02738v1](http://arxiv.org/pdf/2506.02738v1)**

> **作者:** Negin Baghbanzadeh; Sajad Ashkezari; Elham Dolatabadi; Arash Afkanpour
>
> **备注:** 15 pages
>
> **摘要:** Compound figures, which are multi-panel composites containing diverse subfigures, are ubiquitous in biomedical literature, yet large-scale subfigure extraction remains largely unaddressed. Prior work on subfigure extraction has been limited in both dataset size and generalizability, leaving a critical open question: How does high-fidelity image-text alignment via large-scale subfigure extraction impact representation learning in vision-language models? We address this gap by introducing a scalable subfigure extraction pipeline based on transformer-based object detection, trained on a synthetic corpus of 500,000 compound figures, and achieving state-of-the-art performance on both ImageCLEF 2016 and synthetic benchmarks. Using this pipeline, we release OPEN-PMC-18M, a large-scale high quality biomedical vision-language dataset comprising 18 million clinically relevant subfigure-caption pairs spanning radiology, microscopy, and visible light photography. We train and evaluate vision-language models on our curated datasets and show improved performance across retrieval, zero-shot classification, and robustness benchmarks, outperforming existing baselines. We release our dataset, models, and code to support reproducible benchmarks and further study into biomedical vision-language modeling and representation learning.
>
---
#### [new 126] Leveraging Large Language Models in Visual Speech Recognition: Model Scaling, Context-Aware Decoding, and Iterative Polishing
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于视觉语音识别（VSR）任务，旨在通过利用大语言模型（LLMs）提升VSR性能。论文研究了LLM规模对识别效果的影响，并提出上下文感知解码和迭代优化方法，以提高识别准确率。**

- **链接: [http://arxiv.org/pdf/2506.02012v1](http://arxiv.org/pdf/2506.02012v1)**

> **作者:** Zehua Liu; Xiaolou Li; Li Guo; Lantian Li; Dong Wang
>
> **摘要:** Visual Speech Recognition (VSR) transcribes speech by analyzing lip movements. Recently, Large Language Models (LLMs) have been integrated into VSR systems, leading to notable performance improvements. However, the potential of LLMs has not been extensively studied, and how to effectively utilize LLMs in VSR tasks remains unexplored. This paper systematically explores how to better leverage LLMs for VSR tasks and provides three key contributions: (1) Scaling Test: We study how the LLM size affects VSR performance, confirming a scaling law in the VSR task. (2) Context-Aware Decoding: We add contextual text to guide the LLM decoding, improving recognition accuracy. (3) Iterative Polishing: We propose iteratively refining LLM outputs, progressively reducing recognition errors. Extensive experiments demonstrate that by these designs, the great potential of LLMs can be largely harnessed, leading to significant VSR performance improvement.
>
---
#### [new 127] DCM: Dual-Expert Consistency Model for Efficient and High-Quality Video Generation
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决扩散模型计算开销大、一致性模型在视频生成中效果下降的问题。作者提出DCM方法，通过双专家模型分工学习语义和细节，并引入新损失函数提升生成质量与效率。**

- **链接: [http://arxiv.org/pdf/2506.03123v1](http://arxiv.org/pdf/2506.03123v1)**

> **作者:** Zhengyao Lv; Chenyang Si; Tianlin Pan; Zhaoxi Chen; Kwan-Yee K. Wong; Yu Qiao; Ziwei Liu
>
> **摘要:** Diffusion Models have achieved remarkable results in video synthesis but require iterative denoising steps, leading to substantial computational overhead. Consistency Models have made significant progress in accelerating diffusion models. However, directly applying them to video diffusion models often results in severe degradation of temporal consistency and appearance details. In this paper, by analyzing the training dynamics of Consistency Models, we identify a key conflicting learning dynamics during the distillation process: there is a significant discrepancy in the optimization gradients and loss contributions across different timesteps. This discrepancy prevents the distilled student model from achieving an optimal state, leading to compromised temporal consistency and degraded appearance details. To address this issue, we propose a parameter-efficient \textbf{Dual-Expert Consistency Model~(DCM)}, where a semantic expert focuses on learning semantic layout and motion, while a detail expert specializes in fine detail refinement. Furthermore, we introduce Temporal Coherence Loss to improve motion consistency for the semantic expert and apply GAN and Feature Matching Loss to enhance the synthesis quality of the detail expert.Our approach achieves state-of-the-art visual quality with significantly reduced sampling steps, demonstrating the effectiveness of expert specialization in video diffusion model distillation. Our code and models are available at \href{https://github.com/Vchitect/DCM}{https://github.com/Vchitect/DCM}.
>
---
#### [new 128] Revisiting End-to-End Learning with Slide-level Supervision in Computational Pathology
- **分类: cs.CV**

- **简介: 该论文属于计算病理学任务，旨在解决全切片图像分析中端到端学习性能不足的问题。通过提出新的多实例学习方法ABMILX，结合全局相关注意力优化与多头机制，并采用高效的采样策略，使端到端模型在多个基准上超越传统两阶段方法，同时保持计算效率。**

- **链接: [http://arxiv.org/pdf/2506.02408v1](http://arxiv.org/pdf/2506.02408v1)**

> **作者:** Wenhao Tang; Rong Qin; Heng Fang; Fengtao Zhou; Hao Chen; Xiang Li; Ming-Ming Cheng
>
> **摘要:** Pre-trained encoders for offline feature extraction followed by multiple instance learning (MIL) aggregators have become the dominant paradigm in computational pathology (CPath), benefiting cancer diagnosis and prognosis. However, performance limitations arise from the absence of encoder fine-tuning for downstream tasks and disjoint optimization with MIL. While slide-level supervised end-to-end (E2E) learning is an intuitive solution to this issue, it faces challenges such as high computational demands and suboptimal results. These limitations motivate us to revisit E2E learning. We argue that prior work neglects inherent E2E optimization challenges, leading to performance disparities compared to traditional two-stage methods. In this paper, we pioneer the elucidation of optimization challenge caused by sparse-attention MIL and propose a novel MIL called ABMILX. It mitigates this problem through global correlation-based attention refinement and multi-head mechanisms. With the efficient multi-scale random patch sampling strategy, an E2E trained ResNet with ABMILX surpasses SOTA foundation models under the two-stage paradigm across multiple challenging benchmarks, while remaining computationally efficient (<10 RTX3090 hours). We show the potential of E2E learning in CPath and calls for greater research focus in this area. The code is https://github.com/DearCaat/E2E-WSI-ABMILX.
>
---
#### [new 129] Improving Knowledge Distillation Under Unknown Covariate Shift Through Confidence-Guided Data Augmentation
- **分类: cs.CV**

- **简介: 该论文属于知识蒸馏任务，旨在解决协变量偏移下的泛化问题。作者提出了一种基于扩散模型的置信度引导数据增强方法，通过最大化师生模型分歧生成困难样本，提升学生模型对未知伪特征的鲁棒性，从而在多个数据集上取得了更好的性能表现。**

- **链接: [http://arxiv.org/pdf/2506.02294v1](http://arxiv.org/pdf/2506.02294v1)**

> **作者:** Niclas Popp; Kevin Alexander Laube; Matthias Hein; Lukas Schott
>
> **摘要:** Large foundation models trained on extensive datasets demonstrate strong zero-shot capabilities in various domains. To replicate their success when data and model size are constrained, knowledge distillation has become an established tool for transferring knowledge from foundation models to small student networks. However, the effectiveness of distillation is critically limited by the available training data. This work addresses the common practical issue of covariate shift in knowledge distillation, where spurious features appear during training but not at test time. We ask the question: when these spurious features are unknown, yet a robust teacher is available, is it possible for a student to also become robust to them? We address this problem by introducing a novel diffusion-based data augmentation strategy that generates images by maximizing the disagreement between the teacher and the student, effectively creating challenging samples that the student struggles with. Experiments demonstrate that our approach significantly improves worst group and mean group accuracy on CelebA and SpuCo Birds as well as the spurious mAUC on spurious ImageNet under covariate shift, outperforming state-of-the-art diffusion-based data augmentation baselines
>
---
#### [new 130] Towards Better De-raining Generalization via Rainy Characteristics Memorization and Replay
- **分类: cs.CV**

- **简介: 该论文属于图像去雨任务，旨在提升模型在不同真实雨天场景中的泛化能力。现有方法因训练数据有限，难以应对多变的雨况。论文提出一种渐进式学习框架，结合GAN生成新数据、知识蒸馏与回放机制，模拟人脑记忆与学习过程，使网络持续积累多数据集知识，增强对未见雨景的去雨效果。**

- **链接: [http://arxiv.org/pdf/2506.02477v1](http://arxiv.org/pdf/2506.02477v1)**

> **作者:** Kunyu Wang; Xueyang Fu; Chengzhi Cao; Chengjie Ge; Wei Zhai; Zheng-Jun Zha
>
> **摘要:** Current image de-raining methods primarily learn from a limited dataset, leading to inadequate performance in varied real-world rainy conditions. To tackle this, we introduce a new framework that enables networks to progressively expand their de-raining knowledge base by tapping into a growing pool of datasets, significantly boosting their adaptability. Drawing inspiration from the human brain's ability to continuously absorb and generalize from ongoing experiences, our approach borrow the mechanism of the complementary learning system. Specifically, we first deploy Generative Adversarial Networks (GANs) to capture and retain the unique features of new data, mirroring the hippocampus's role in learning and memory. Then, the de-raining network is trained with both existing and GAN-synthesized data, mimicking the process of hippocampal replay and interleaved learning. Furthermore, we employ knowledge distillation with the replayed data to replicate the synergy between the neocortex's activity patterns triggered by hippocampal replays and the pre-existing neocortical knowledge. This comprehensive framework empowers the de-raining network to amass knowledge from various datasets, continually enhancing its performance on previously unseen rainy scenes. Our testing on three benchmark de-raining networks confirms the framework's effectiveness. It not only facilitates continuous knowledge accumulation across six datasets but also surpasses state-of-the-art methods in generalizing to new real-world scenarios.
>
---
#### [new 131] Pan-Arctic Permafrost Landform and Human-built Infrastructure Feature Detection with Vision Transformers and Location Embeddings
- **分类: cs.CV; I.4.6; I.5.4; I.5.2; I.2.10**

- **简介: 该论文属于遥感图像处理任务，旨在解决北极地区冻土地貌和人工设施检测问题。利用视觉Transformer（ViTs）并融合位置嵌入，提升模型在多样光谱条件下的泛化能力。实验表明，结合位置信息的ViT在多个检测任务中优于传统CNN方法，特别是在冻融滑塌检测中F1分数显著提高。**

- **链接: [http://arxiv.org/pdf/2506.02868v1](http://arxiv.org/pdf/2506.02868v1)**

> **作者:** Amal S. Perera; David Fernandez; Chandi Witharana; Elias Manos; Michael Pimenta; Anna K. Liljedahl; Ingmar Nitze; Yili Yang; Todd Nicholson; Chia-Yu Hsu; Wenwen Li; Guido Grosse
>
> **备注:** 20 pages, 2 column IEEE format, 13 Figures
>
> **摘要:** Accurate mapping of permafrost landforms, thaw disturbances, and human-built infrastructure at pan-Arctic scale using sub-meter satellite imagery is increasingly critical. Handling petabyte-scale image data requires high-performance computing and robust feature detection models. While convolutional neural network (CNN)-based deep learning approaches are widely used for remote sensing (RS),similar to the success in transformer based large language models, Vision Transformers (ViTs) offer advantages in capturing long-range dependencies and global context via attention mechanisms. ViTs support pretraining via self-supervised learning-addressing the common limitation of labeled data in Arctic feature detection and outperform CNNs on benchmark datasets. Arctic also poses challenges for model generalization, especially when features with the same semantic class exhibit diverse spectral characteristics. To address these issues for Arctic feature detection, we integrate geospatial location embeddings into ViTs to improve adaptation across regions. This work investigates: (1) the suitability of pre-trained ViTs as feature extractors for high-resolution Arctic remote sensing tasks, and (2) the benefit of combining image and location embeddings. Using previously published datasets for Arctic feature detection, we evaluate our models on three tasks-detecting ice-wedge polygons (IWP), retrogressive thaw slumps (RTS), and human-built infrastructure. We empirically explore multiple configurations to fuse image embeddings and location embeddings. Results show that ViTs with location embeddings outperform prior CNN-based models on two of the three tasks including F1 score increase from 0.84 to 0.92 for RTS detection, demonstrating the potential of transformer-based models with spatial awareness for Arctic RS applications.
>
---
#### [new 132] Multi-level and Multi-modal Action Anticipation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于动作预判任务，旨在从部分观察视频中预测未来动作。它试图解决传统方法忽略多模态信息及层级语义建模的问题。论文引入了结合视觉与文本线索的多模态方法，并提出细粒度标签生成和时间一致性损失函数，以提升预测准确性。实验表明其方法在多个数据集上达到最优性能。**

- **链接: [http://arxiv.org/pdf/2506.02382v1](http://arxiv.org/pdf/2506.02382v1)**

> **作者:** Seulgi Kim; Ghazal Kaviani; Mohit Prabhushankar; Ghassan AlRegib
>
> **备注:** Accepted in 2025 IEEE International Conference on Image Processing (ICIP)
>
> **摘要:** Action anticipation, the task of predicting future actions from partially observed videos, is crucial for advancing intelligent systems. Unlike action recognition, which operates on fully observed videos, action anticipation must handle incomplete information. Hence, it requires temporal reasoning, and inherent uncertainty handling. While recent advances have been made, traditional methods often focus solely on visual modalities, neglecting the potential of integrating multiple sources of information. Drawing inspiration from human behavior, we introduce \textit{Multi-level and Multi-modal Action Anticipation (m\&m-Ant)}, a novel multi-modal action anticipation approach that combines both visual and textual cues, while explicitly modeling hierarchical semantic information for more accurate predictions. To address the challenge of inaccurate coarse action labels, we propose a fine-grained label generator paired with a specialized temporal consistency loss function to optimize performance. Extensive experiments on widely used datasets, including Breakfast, 50 Salads, and DARai, demonstrate the effectiveness of our approach, achieving state-of-the-art results with an average anticipation accuracy improvement of 3.08\% over existing methods. This work underscores the potential of multi-modal and hierarchical modeling in advancing action anticipation and establishes a new benchmark for future research in the field. Our code is available at: https://github.com/olivesgatech/mM-ant.
>
---
#### [new 133] SAMJ: Fast Image Annotation on ImageJ/Fiji via Segment Anything Model
- **分类: cs.CV**

- **简介: 该论文属于图像处理任务，旨在解决生物医学图像标注效率低的问题。作者开发了SAMJ插件，集成Segment Anything Model至ImageJ/Fiji平台，实现快速、交互式的图像标注，提升大规模科学图像数据的标注效率与易用性。**

- **链接: [http://arxiv.org/pdf/2506.02783v1](http://arxiv.org/pdf/2506.02783v1)**

> **作者:** Carlos Garcia-Lopez-de-Haro; Caterina Fuster-Barcelo; Curtis T. Rueden; Jonathan Heras; Vladimir Ulman; Daniel Franco-Barranco; Adrian Ines; Kevin W. Eliceiri; Jean-Christophe Olivo-Marin; Jean-Yves Tinevez; Daniel Sage; Arrate Munoz-Barrutia
>
> **摘要:** Mask annotation remains a significant bottleneck in AI-driven biomedical image analysis due to its labor-intensive nature. To address this challenge, we introduce SAMJ, a user-friendly ImageJ/Fiji plugin leveraging the Segment Anything Model (SAM). SAMJ enables seamless, interactive annotations with one-click installation on standard computers. Designed for real-time object delineation in large scientific images, SAMJ is an easy-to-use solution that simplifies and accelerates the creation of labeled image datasets.
>
---
#### [new 134] A TRPCA-Inspired Deep Unfolding Network for Hyperspectral Image Denoising via Thresholded t-SVD and Top-K Sparse Transformer
- **分类: cs.CV**

- **简介: 该论文属于图像处理任务，旨在解决高光谱图像受复杂混合噪声影响的问题。作者提出了一种基于张量鲁棒主成分分析（TRPCA）的深度展开网络（DU-TRPCA），结合低秩和稀疏模块，通过阈值张量奇异值分解和Top-K稀疏Transformer实现高效去噪，提升了去噪效果与模型可解释性。**

- **链接: [http://arxiv.org/pdf/2506.02364v1](http://arxiv.org/pdf/2506.02364v1)**

> **作者:** Liang Li; Jianli Zhao; Sheng Fang; Siyu Chen; Hui Sun
>
> **备注:** 11 pages,6 figures
>
> **摘要:** Hyperspectral images (HSIs) are often degraded by complex mixed noise during acquisition and transmission, making effective denoising essential for subsequent analysis. Recent hybrid approaches that bridge model-driven and data-driven paradigms have shown great promise. However, most of these approaches lack effective alternation between different priors or modules, resulting in loosely coupled regularization and insufficient exploitation of their complementary strengths. Inspired by tensor robust principal component analysis (TRPCA), we propose a novel deep unfolding network (DU-TRPCA) that enforces stage-wise alternation between two tightly integrated modules: low-rank and sparse. The low-rank module employs thresholded tensor singular value decomposition (t-SVD), providing a widely adopted convex surrogate for tensor low-rankness and has been demonstrated to effectively capture the global spatial-spectral structure of HSIs. The Top-K sparse transformer module adaptively imposes sparse constraints, directly matching the sparse regularization in TRPCA and enabling effective removal of localized outliers and complex noise. This tightly coupled architecture preserves the stage-wise alternation between low-rank approximation and sparse refinement inherent in TRPCA, while enhancing representational capacity through attention mechanisms. Extensive experiments on synthetic and real-world HSIs demonstrate that DU-TRPCA surpasses state-of-the-art methods under severe mixed noise, while offering interpretability benefits and stable denoising dynamics inspired by iterative optimization. Code is available at https://github.com/liangli97/TRPCA-Deep-Unfolding-HSI-Denoising.
>
---
#### [new 135] Contrast & Compress: Learning Lightweight Embeddings for Short Trajectories
- **分类: cs.CV**

- **简介: 该论文属于轨迹检索任务，旨在解决短距离轨迹的语义与方向相似性高效检索问题。作者提出一种基于Transformer编码器和对比学习的框架，通过Cosine与FFT相似度指标分析，学习低维、可解释的轨迹嵌入表示，提升检索性能并兼顾计算效率。**

- **链接: [http://arxiv.org/pdf/2506.02571v1](http://arxiv.org/pdf/2506.02571v1)**

> **作者:** Abhishek Vivekanandan; Christian Hubschneider; J. Marius Zöllner
>
> **备注:** Submitted for peer review
>
> **摘要:** The ability to retrieve semantically and directionally similar short-range trajectories with both accuracy and efficiency is foundational for downstream applications such as motion forecasting and autonomous navigation. However, prevailing approaches often depend on computationally intensive heuristics or latent anchor representations that lack interpretability and controllability. In this work, we propose a novel framework for learning fixed-dimensional embeddings for short trajectories by leveraging a Transformer encoder trained with a contrastive triplet loss that emphasize the importance of discriminative feature spaces for trajectory data. We analyze the influence of Cosine and FFT-based similarity metrics within the contrastive learning paradigm, with a focus on capturing the nuanced directional intent that characterizes short-term maneuvers. Our empirical evaluation on the Argoverse 2 dataset demonstrates that embeddings shaped by Cosine similarity objectives yield superior clustering of trajectories by both semantic and directional attributes, outperforming FFT-based baselines in retrieval tasks. Notably, we show that compact Transformer architectures, even with low-dimensional embeddings (e.g., 16 dimensions, but qualitatively down to 4), achieve a compelling balance between retrieval performance (minADE, minFDE) and computational overhead, aligning with the growing demand for scalable and interpretable motion priors in real-time systems. The resulting embeddings provide a compact, semantically meaningful, and efficient representation of trajectory data, offering a robust alternative to heuristic similarity measures and paving the way for more transparent and controllable motion forecasting pipelines.
>
---
#### [new 136] Modelship Attribution: Tracing Multi-Stage Manipulations Across Generative Models
- **分类: cs.CV**

- **简介: 该论文属于图像生成与安全领域，旨在解决多阶段图像篡改的溯源问题。现有方法难以应对复杂编辑场景，作者提出“Modelship Attribution”任务，追踪图像修改过程中使用的生成模型及顺序。他们构建了首个相关数据集，并设计了专门的Transformer框架（MAT）进行模型溯源，提升了多阶段篡改的识别效果。**

- **链接: [http://arxiv.org/pdf/2506.02405v1](http://arxiv.org/pdf/2506.02405v1)**

> **作者:** Zhiya Tan; Xin Zhang; Joey Tianyi Zhou
>
> **摘要:** As generative techniques become increasingly accessible, authentic visuals are frequently subjected to iterative alterations by various individuals employing a variety of tools. Currently, to avoid misinformation and ensure accountability, a lot of research on detection and attribution is emerging. Although these methods demonstrate promise in single-stage manipulation scenarios, they fall short when addressing complex real-world iterative manipulation. In this paper, we are the first, to the best of our knowledge, to systematically model this real-world challenge and introduce a novel method to solve it. We define a task called "Modelship Attribution", which aims to trace the evolution of manipulated images by identifying the generative models involved and reconstructing the sequence of edits they performed. To realistically simulate this scenario, we utilize three generative models, StyleMapGAN, DiffSwap, and FacePartsSwap, that sequentially modify distinct regions of the same image. This process leads to the creation of the first modelship dataset, comprising 83,700 images (16,740 images*5). Given that later edits often overwrite the fingerprints of earlier models, the focus shifts from extracting blended fingerprints to characterizing each model's distinctive editing patterns. To tackle this challenge, we introduce the modelship attribution transformer (MAT), a purpose-built framework designed to effectively recognize and attribute the contributions of various models within complex, multi-stage manipulation workflows. Through extensive experiments and comparative analysis with other related methods, our results, including comprehensive ablation studies, demonstrate that the proposed approach is a highly effective solution for modelship attribution.
>
---
#### [new 137] SVGenius: Benchmarking LLMs in SVG Understanding, Editing and Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉与图形处理任务，旨在解决现有SVG处理模型评估不足的问题。作者构建了SVGenus基准测试，包含2,377个查询，覆盖理解、编辑和生成三个维度，系统评估22种主流模型的性能，揭示其在复杂性增加时的表现限制，并强调推理增强训练的有效性。**

- **链接: [http://arxiv.org/pdf/2506.03139v1](http://arxiv.org/pdf/2506.03139v1)**

> **作者:** Siqi Chen; Xinyu Dong; Haolei Xu; Xingyu Wu; Fei Tang; Hang Zhang; Yuchen Yan; Linjuan Wu; Wenqi Zhang; Guiyang Hou; Yongliang Shen; Weiming Lu; Yueting Zhuang
>
> **备注:** 19 pages,4 figures, Project page: https://zju-real.github.io/SVGenius, Code: https://github.com/ZJU-REAL/SVGenius-Bench
>
> **摘要:** Large Language Models (LLMs) and Multimodal LLMs have shown promising capabilities for SVG processing, yet existing benchmarks suffer from limited real-world coverage, lack of complexity stratification, and fragmented evaluation paradigms. We introduce SVGenius, a comprehensive benchmark comprising 2,377 queries across three progressive dimensions: understanding, editing, and generation. Built on real-world data from 24 application domains with systematic complexity stratification, SVGenius evaluates models through 8 task categories and 18 metrics. We assess 22 mainstream models spanning different scales, architectures, training paradigms, and accessibility levels. Our analysis reveals that while proprietary models significantly outperform open-source counterparts, all models exhibit systematic performance degradation with increasing complexity, indicating fundamental limitations in current approaches; however, reasoning-enhanced training proves more effective than pure scaling for overcoming these limitations, though style transfer remains the most challenging capability across all model types. SVGenius establishes the first systematic evaluation framework for SVG processing, providing crucial insights for developing more capable vector graphics models and advancing automated graphic design applications. Appendix and supplementary materials (including all data and code) are available at https://zju-real.github.io/SVGenius.
>
---
#### [new 138] RATE-Nav: Region-Aware Termination Enhancement for Zero-shot Object Navigation with Vision-Language Models
- **分类: cs.CV**

- **简介: 论文属于具身人工智能中的目标导航任务，旨在解决零样本条件下冗余探索与探索失败问题。提出RATE-Nav方法，通过区域感知的终止机制提升导航效率，结合几何预测与视觉语言模型实现有效探索终止。**

- **链接: [http://arxiv.org/pdf/2506.02354v1](http://arxiv.org/pdf/2506.02354v1)**

> **作者:** Junjie Li; Nan Zhang; Xiaoyang Qu; Kai Lu; Guokuan Li; Jiguang Wan; Jianzong Wang
>
> **备注:** Accepted by the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)
>
> **摘要:** Object Navigation (ObjectNav) is a fundamental task in embodied artificial intelligence. Although significant progress has been made in semantic map construction and target direction prediction in current research, redundant exploration and exploration failures remain inevitable. A critical but underexplored direction is the timely termination of exploration to overcome these challenges. We observe a diminishing marginal effect between exploration steps and exploration rates and analyze the cost-benefit relationship of exploration. Inspired by this, we propose RATE-Nav, a Region-Aware Termination-Enhanced method. It includes a geometric predictive region segmentation algorithm and region-Based exploration estimation algorithm for exploration rate calculation. By leveraging the visual question answering capabilities of visual language models (VLMs) and exploration rates enables efficient termination.RATE-Nav achieves a success rate of 67.8% and an SPL of 31.3% on the HM3D dataset. And on the more challenging MP3D dataset, RATE-Nav shows approximately 10% improvement over previous zero-shot methods.
>
---
#### [new 139] HRTR: A Single-stage Transformer for Fine-grained Sub-second Action Segmentation in Stroke Rehabilitation
- **分类: cs.CV**

- **简介: 该论文属于动作分割任务，旨在解决中风康复中精细、亚秒级动作检测难题。作者提出HRTR模型，一种单阶段高分辨率时间Transformer，无需多阶段方法或后处理，即可实现精准的动作分类与时间定位。**

- **链接: [http://arxiv.org/pdf/2506.02472v1](http://arxiv.org/pdf/2506.02472v1)**

> **作者:** Halil Ismail Helvaci; Justin Philip Huber; Jihye Bae; Sen-ching Samson Cheung
>
> **摘要:** Stroke rehabilitation often demands precise tracking of patient movements to monitor progress, with complexities of rehabilitation exercises presenting two critical challenges: fine-grained and sub-second (under one-second) action detection. In this work, we propose the High Resolution Temporal Transformer (HRTR), to time-localize and classify high-resolution (fine-grained), sub-second actions in a single-stage transformer, eliminating the need for multi-stage methods and post-processing. Without any refinements, HRTR outperforms state-of-the-art systems on both stroke related and general datasets, achieving Edit Score (ES) of 70.1 on StrokeRehab Video, 69.4 on StrokeRehab IMU, and 88.4 on 50Salads.
>
---
#### [new 140] Hyperspectral Image Generation with Unmixing Guided Diffusion Model
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于遥感图像生成任务，旨在解决现有模型依赖条件生成、多样性不足的问题。作者提出了一种基于解混引导的扩散模型，包含解混自编码器和丰度扩散模块，降低计算复杂度并保证物理一致性。此外，还引入了两个新评价指标，提升了高光谱图像生成质量与多样性。**

- **链接: [http://arxiv.org/pdf/2506.02601v1](http://arxiv.org/pdf/2506.02601v1)**

> **作者:** Shiyu Shen; Bin Pan; Ziye Zhang; Zhenwei Shi
>
> **摘要:** Recently, hyperspectral image generation has received increasing attention, but existing generative models rely on conditional generation schemes, which limits the diversity of generated images. Diffusion models are popular for their ability to generate high-quality samples, but adapting these models from RGB to hyperspectral data presents the challenge of high dimensionality and physical constraints. To address these challenges, we propose a novel diffusion model guided by hyperspectral unmixing. Our model comprises two key modules: an unmixing autoencoder module and an abundance diffusion module. The unmixing autoencoder module leverages unmixing guidance to shift the generative task from the image space to the low-dimensional abundance space, significantly reducing computational complexity while preserving high fidelity. The abundance diffusion module generates samples that satisfy the constraints of non-negativity and unity, ensuring the physical consistency of the reconstructed HSIs. Additionally, we introduce two evaluation metrics tailored to hyperspectral data. Empirical results, evaluated using both traditional metrics and our proposed metrics, indicate that our model is capable of generating high-quality and diverse hyperspectral images, offering an advancement in hyperspectral data generation.
>
---
#### [new 141] Technical Report for Ego4D Long-Term Action Anticipation Challenge 2025
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于Ego4D长期动作预测任务，旨在解决第一视角视频中未来动作的准确预测问题。作者提出一个三阶段框架：特征提取、动作识别和长期动作预测。使用高性能视觉编码器和Transformer进行动作识别，并结合语言模型预测未来动作序列，最终取得CVPR 2025挑战赛第一名。**

- **链接: [http://arxiv.org/pdf/2506.02550v1](http://arxiv.org/pdf/2506.02550v1)**

> **作者:** Qiaohui Chu; Haoyu Zhang; Yisen Feng; Meng Liu; Weili Guan; Yaowei Wang; Liqiang Nie
>
> **备注:** The champion solution for the Ego4D Long-Term Action Anticipation Challenge at the CVPR EgoVis Workshop 2025
>
> **摘要:** In this report, we present a novel three-stage framework developed for the Ego4D Long-Term Action Anticipation (LTA) task. Inspired by recent advances in foundation models, our method consists of three stages: feature extraction, action recognition, and long-term action anticipation. First, visual features are extracted using a high-performance visual encoder. The features are then fed into a Transformer to predict verbs and nouns, with a verb-noun co-occurrence matrix incorporated to enhance recognition accuracy. Finally, the predicted verb-noun pairs are formatted as textual prompts and input into a fine-tuned large language model (LLM) to anticipate future action sequences. Our framework achieves first place in this challenge at CVPR 2025, establishing a new state-of-the-art in long-term action prediction. Our code will be released at https://github.com/CorrineQiu/Ego4D-LTA-Challenge-2025.
>
---
#### [new 142] Johnny: Structuring Representation Space to Enhance Machine Abstract Reasoning Ability
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于机器抽象推理任务，旨在解决传统模型在Raven渐进矩阵（RPM）任务中过度依赖选项池配置的问题。作者提出了Johnny架构和Spin-Transformer网络，通过构建表征空间和优化注意力机制，提升AI的抽象推理能力，并验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2506.01970v1](http://arxiv.org/pdf/2506.01970v1)**

> **作者:** Ruizhuo Song; Beiming Yuan
>
> **备注:** 15 pages, 15 figures, 5 tables
>
> **摘要:** This paper thoroughly investigates the challenges of enhancing AI's abstract reasoning capabilities, with a particular focus on Raven's Progressive Matrices (RPM) tasks involving complex human-like concepts. Firstly, it dissects the empirical reality that traditional end-to-end RPM-solving models heavily rely on option pool configurations, highlighting that this dependency constrains the model's reasoning capabilities. To address this limitation, the paper proposes the Johnny architecture - a novel representation space-based framework for RPM-solving. Through the synergistic operation of its Representation Extraction Module and Reasoning Module, Johnny significantly enhances reasoning performance by supplementing primitive negative option configurations with a learned representation space. Furthermore, to strengthen the model's capacity for capturing positional relationships among local features, the paper introduces the Spin-Transformer network architecture, accompanied by a lightweight Straw Spin-Transformer variant that reduces computational overhead through parameter sharing and attention mechanism optimization. Experimental evaluations demonstrate that both Johnny and Spin-Transformer achieve superior performance on RPM tasks, offering innovative methodologies for advancing AI's abstract reasoning capabilities.
>
---
#### [new 143] VolTex: Food Volume Estimation using Text-Guided Segmentation and Neural Surface Reconstruction
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于食品体积估计任务，旨在解决现有方法在食物对象选择上的不足。通过文本引导的分割与神经表面重建技术，VolTex实现了对特定食品对象的精确选择和高保真3D重建，从而提高了食品体积估计的准确性。**

- **链接: [http://arxiv.org/pdf/2506.02895v1](http://arxiv.org/pdf/2506.02895v1)**

> **作者:** Ahmad AlMughrabi; Umair Haroon; Ricardo Marques; Petia Radeva
>
> **摘要:** Accurate food volume estimation is crucial for dietary monitoring, medical nutrition management, and food intake analysis. Existing 3D Food Volume estimation methods accurately compute the food volume but lack for food portions selection. We present VolTex, a framework that improves \change{the food object selection} in food volume estimation. Allowing users to specify a target food item via text input to be segmented, our method enables the precise selection of specific food objects in real-world scenes. The segmented object is then reconstructed using the Neural Surface Reconstruction method to generate high-fidelity 3D meshes for volume computation. Extensive evaluations on the MetaFood3D dataset demonstrate the effectiveness of our approach in isolating and reconstructing food items for accurate volume estimation. The source code is accessible at https://github.com/GCVCG/VolTex.
>
---
#### [new 144] MotionRAG-Diff: A Retrieval-Augmented Diffusion Framework for Long-Term Music-to-Dance Generation
- **分类: cs.SD; cs.CV; cs.GR; eess.AS**

- **简介: 该论文属于音乐驱动舞蹈生成任务，旨在解决长期舞蹈序列生成中连贯性、同步性和创造性不足的问题。作者提出MotionRAG-Diff框架，结合检索增强生成与扩散模型，实现高质量、与音乐同步的舞蹈生成。**

- **链接: [http://arxiv.org/pdf/2506.02661v1](http://arxiv.org/pdf/2506.02661v1)**

> **作者:** Mingyang Huang; Peng Zhang; Bang Zhang
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Generating long-term, coherent, and realistic music-conditioned dance sequences remains a challenging task in human motion synthesis. Existing approaches exhibit critical limitations: motion graph methods rely on fixed template libraries, restricting creative generation; diffusion models, while capable of producing novel motions, often lack temporal coherence and musical alignment. To address these challenges, we propose $\textbf{MotionRAG-Diff}$, a hybrid framework that integrates Retrieval-Augmented Generation (RAG) with diffusion-based refinement to enable high-quality, musically coherent dance generation for arbitrary long-term music inputs. Our method introduces three core innovations: (1) A cross-modal contrastive learning architecture that aligns heterogeneous music and dance representations in a shared latent space, establishing unsupervised semantic correspondence without paired data; (2) An optimized motion graph system for efficient retrieval and seamless concatenation of motion segments, ensuring realism and temporal coherence across long sequences; (3) A multi-condition diffusion model that jointly conditions on raw music signals and contrastive features to enhance motion quality and global synchronization. Extensive experiments demonstrate that MotionRAG-Diff achieves state-of-the-art performance in motion quality, diversity, and music-motion synchronization accuracy. This work establishes a new paradigm for music-driven dance generation by synergizing retrieval-based template fidelity with diffusion-based creative enhancement.
>
---
#### [new 145] Grasp2Grasp: Vision-Based Dexterous Grasp Translation via Schrödinger Bridges
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人抓取迁移任务，旨在解决不同结构机械手间的抓取意图传递问题。通过引入Schrödinger Bridge方法，结合视觉观测与物理感知代价函数，实现无需配对数据或仿真的跨手型抓取生成，提升了抓取稳定性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.02489v1](http://arxiv.org/pdf/2506.02489v1)**

> **作者:** Tao Zhong; Jonah Buchanan; Christine Allen-Blanchette
>
> **备注:** 19 pages, 4 figures
>
> **摘要:** We propose a new approach to vision-based dexterous grasp translation, which aims to transfer grasp intent across robotic hands with differing morphologies. Given a visual observation of a source hand grasping an object, our goal is to synthesize a functionally equivalent grasp for a target hand without requiring paired demonstrations or hand-specific simulations. We frame this problem as a stochastic transport between grasp distributions using the Schr\"odinger Bridge formalism. Our method learns to map between source and target latent grasp spaces via score and flow matching, conditioned on visual observations. To guide this translation, we introduce physics-informed cost functions that encode alignment in base pose, contact maps, wrench space, and manipulability. Experiments across diverse hand-object pairs demonstrate our approach generates stable, physically grounded grasps with strong generalization. This work enables semantic grasp transfer for heterogeneous manipulators and bridges vision-based grasping with probabilistic generative modeling.
>
---
#### [new 146] PhysGaia: A Physics-Aware Dataset of Multi-Body Interactions for Dynamic Novel View Synthesis
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 该论文属于动态新视角合成任务，旨在解决现有数据集缺乏物理感知的问题。论文构建了PhysGaia数据集，包含多物体复杂互动及多种物理材质，严格遵循物理规律，并提供粒子轨迹与物理参数用于评估。论文还提供了DyNVS模型的集成方案，推动物理感知动态场景建模研究。**

- **链接: [http://arxiv.org/pdf/2506.02794v1](http://arxiv.org/pdf/2506.02794v1)**

> **作者:** Mijeong Kim; Gunhee Kim; Jungyoon Choi; Wonjae Roh; Bohyung Han
>
> **备注:** Project page: http://cvlab.snu.ac.kr/research/PhysGaia, Data: https://huggingface.co/datasets/mijeongkim/PhysGaia/tree/main
>
> **摘要:** We introduce PhysGaia, a novel physics-aware dataset specifically designed for Dynamic Novel View Synthesis (DyNVS), encompassing both structured objects and unstructured physical phenomena. Unlike existing datasets that primarily focus on photorealistic reconstruction, PhysGaia is created to actively support physics-aware dynamic scene modeling. Our dataset provides complex dynamic scenarios with rich interactions among multiple objects, where they realistically collide with each other and exchange forces. Furthermore, it contains a diverse range of physical materials, such as liquid, gas, viscoelastic substance, and textile, which moves beyond the rigid bodies prevalent in existing datasets. All scenes in PhysGaia are faithfully generated to strictly adhere to physical laws, leveraging carefully selected material-specific physics solvers. To enable quantitative evaluation of physical modeling, our dataset provides essential ground-truth information, including 3D particle trajectories and physics parameters, e.g., viscosity. To facilitate research adoption, we also provide essential integration pipelines for using state-of-the-art DyNVS models with our dataset and report their results. By addressing the critical lack of datasets for physics-aware modeling, PhysGaia will significantly advance research in dynamic view synthesis, physics-based scene understanding, and deep learning models integrated with physical simulation -- ultimately enabling more faithful reconstruction and interpretation of complex dynamic scenes. Our datasets and codes are available in the project website, http://cvlab.snu.ac.kr/research/PhysGaia.
>
---
#### [new 147] EyeNavGS: A 6-DoF Navigation Dataset and Record-n-Replay Software for Real-World 3DGS Scenes in VR
- **分类: cs.MM; cs.CV; cs.GR; cs.HC**

- **简介: 该论文属于虚拟现实与3D场景重建任务，旨在解决缺乏真实用户导航数据的问题。作者构建了EyeNavGS数据集，包含46名参与者在12个真实场景中的6自由度导航轨迹及眼动数据，并开发了配套软件工具，助力视点预测、自适应渲染等研究。**

- **链接: [http://arxiv.org/pdf/2506.02380v1](http://arxiv.org/pdf/2506.02380v1)**

> **作者:** Zihao Ding; Cheng-Tse Lee; Mufeng Zhu; Tao Guan; Yuan-Chun Sun; Cheng-Hsin Hsu; Yao Liu
>
> **摘要:** 3D Gaussian Splatting (3DGS) is an emerging media representation that reconstructs real-world 3D scenes in high fidelity, enabling 6-degrees-of-freedom (6-DoF) navigation in virtual reality (VR). However, developing and evaluating 3DGS-enabled applications and optimizing their rendering performance, require realistic user navigation data. Such data is currently unavailable for photorealistic 3DGS reconstructions of real-world scenes. This paper introduces EyeNavGS (EyeNavGS), the first publicly available 6-DoF navigation dataset featuring traces from 46 participants exploring twelve diverse, real-world 3DGS scenes. The dataset was collected at two sites, using the Meta Quest Pro headsets, recording the head pose and eye gaze data for each rendered frame during free world standing 6-DoF navigation. For each of the twelve scenes, we performed careful scene initialization to correct for scene tilt and scale, ensuring a perceptually-comfortable VR experience. We also release our open-source SIBR viewer software fork with record-and-replay functionalities and a suite of utility tools for data processing, conversion, and visualization. The EyeNavGS dataset and its accompanying software tools provide valuable resources for advancing research in 6-DoF viewport prediction, adaptive streaming, 3D saliency, and foveated rendering for 3DGS scenes. The EyeNavGS dataset is available at: https://symmru.github.io/EyeNavGS/.
>
---
#### [new 148] Memorization to Generalization: Emergence of Diffusion Models from Associative Memory
- **分类: cs.LG; cond-mat.dis-nn; cs.CV; q-bio.NC; stat.ML**

- **简介: 该论文研究扩散模型在生成建模中的记忆与泛化能力，从联想记忆系统角度分析其训练和生成阶段。任务是揭示扩散模型中虚假状态的存在及其在数据量变化下的演化现象。工作包括理论预测、实验验证虚假吸引子状态，并解释其在小样本和大样本数据下的不同表现。**

- **链接: [http://arxiv.org/pdf/2505.21777v1](http://arxiv.org/pdf/2505.21777v1)**

> **作者:** Bao Pham; Gabriel Raya; Matteo Negri; Mohammed J. Zaki; Luca Ambrogioni; Dmitry Krotov
>
> **摘要:** Hopfield networks are associative memory (AM) systems, designed for storing and retrieving patterns as local minima of an energy landscape. In the classical Hopfield model, an interesting phenomenon occurs when the amount of training data reaches its critical memory load $- spurious\,\,states$, or unintended stable points, emerge at the end of the retrieval dynamics, leading to incorrect recall. In this work, we examine diffusion models, commonly used in generative modeling, from the perspective of AMs. The training phase of diffusion model is conceptualized as memory encoding (training data is stored in the memory). The generation phase is viewed as an attempt of memory retrieval. In the small data regime the diffusion model exhibits a strong memorization phase, where the network creates distinct basins of attraction around each sample in the training set, akin to the Hopfield model below the critical memory load. In the large data regime, a different phase appears where an increase in the size of the training set fosters the creation of new attractor states that correspond to manifolds of the generated samples. Spurious states appear at the boundary of this transition and correspond to emergent attractor states, which are absent in the training set, but, at the same time, have distinct basins of attraction around them. Our findings provide: a novel perspective on the memorization-generalization phenomenon in diffusion models via the lens of AMs, theoretical prediction of existence of spurious states, empirical validation of this prediction in commonly-used diffusion models.
>
---
#### [new 149] Are Pixel-Wise Metrics Reliable for Sparse-View Computed Tomography Reconstruction?
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像重建任务，旨在解决稀疏视角CT重建中传统像素度量无法准确评估解剖结构完整性的问题。论文提出了一种新的解剖感知评价指标和CARE框架，通过训练引入结构惩罚以提升重建的结构性，有效改善了器官、肠道和血管等结构的完整性。**

- **链接: [http://arxiv.org/pdf/2506.02093v1](http://arxiv.org/pdf/2506.02093v1)**

> **作者:** Tianyu Lin; Xinran Li; Chuntung Zhuang; Qi Chen; Yuanhao Cai; Kai Ding; Alan L. Yuille; Zongwei Zhou
>
> **摘要:** Widely adopted evaluation metrics for sparse-view CT reconstruction--such as Structural Similarity Index Measure and Peak Signal-to-Noise Ratio--prioritize pixel-wise fidelity but often fail to capture the completeness of critical anatomical structures, particularly small or thin regions that are easily missed. To address this limitation, we propose a suite of novel anatomy-aware evaluation metrics designed to assess structural completeness across anatomical structures, including large organs, small organs, intestines, and vessels. Building on these metrics, we introduce CARE, a Completeness-Aware Reconstruction Enhancement framework that incorporates structural penalties during training to encourage anatomical preservation of significant structures. CARE is model-agnostic and can be seamlessly integrated into analytical, implicit, and generative methods. When applied to these methods, CARE substantially improves structural completeness in CT reconstructions, achieving up to +32% improvement for large organs, +22% for small organs, +40% for intestines, and +36% for vessels.
>
---
#### [new 150] DIAMOND: An LLM-Driven Agent for Context-Aware Baseball Highlight Summarization
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于体育视频摘要任务，旨在解决传统方法在捕捉比赛亮点时缺乏战略深度和叙事连贯性问题。论文提出DIAMOND框架，结合统计分析与大语言模型的自然语言理解，提升棒球比赛关键时刻的识别效果。实验表明其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.02351v1](http://arxiv.org/pdf/2506.02351v1)**

> **作者:** Jeonghun Kang; Soonmok Kwon; Joonseok Lee; Byung-Hak Kim
>
> **备注:** To appear in the First REALM (Research on Agent Language Models) workshop at ACL 2025
>
> **摘要:** Traditional approaches -- such as Win Probability Added (WPA)-based ranking or computer vision-driven event detection -- can identify scoring plays but often miss strategic depth, momentum shifts, and storyline progression. Manual curation remains the gold standard but is resource-intensive and not scalable. We introduce DIAMOND, an LLM-driven agent for context-aware baseball highlight summarization that integrates structured sports analytics with natural language reasoning. DIAMOND leverages sabermetric features -- Win Expectancy, WPA, and Leverage Index -- to quantify play importance, while an LLM module enhances selection based on contextual narrative value. This hybrid approach ensures both quantitative rigor and qualitative richness, surpassing the limitations of purely statistical or vision-based systems. Evaluated on five diverse Korean Baseball Organization League games, DIAMOND improves F1-score from 42.9% (WPA-only) to 84.8%, outperforming both commercial and statistical baselines. Though limited in scale, our results highlight the potential of modular, interpretable agent-based frameworks for event-level summarization in sports and beyond.
>
---
#### [new 151] GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 论文提出GUI-Actor，旨在解决基于视觉语言模型（VLM）的GUI代理中的视觉定位问题。该方法通过引入注意力机制和一个用于评估动作区域的验证器，实现无需坐标的视觉定位，提升模型在不同屏幕分辨率和布局上的泛化能力，并在多个基准测试中表现优异。**

- **链接: [http://arxiv.org/pdf/2506.03143v1](http://arxiv.org/pdf/2506.03143v1)**

> **作者:** Qianhui Wu; Kanzhi Cheng; Rui Yang; Chaoyun Zhang; Jianwei Yang; Huiqiang Jiang; Jian Mu; Baolin Peng; Bo Qiao; Reuben Tan; Si Qin; Lars Liden; Qingwei Lin; Huan Zhang; Tong Zhang; Jianbing Zhang; Dongmei Zhang; Jianfeng Gao
>
> **摘要:** One of the principal challenges in building VLM-powered GUI agents is visual grounding, i.e., localizing the appropriate screen region for action execution based on both the visual content and the textual plans. Most existing work formulates this as a text-based coordinate generation task. However, these approaches suffer from several limitations: weak spatial-semantic alignment, inability to handle ambiguous supervision targets, and a mismatch between the dense nature of screen coordinates and the coarse, patch-level granularity of visual features extracted by models like Vision Transformers. In this paper, we propose GUI-Actor, a VLM-based method for coordinate-free GUI grounding. At its core, GUI-Actor introduces an attention-based action head that learns to align a dedicated <ACTOR> token with all relevant visual patch tokens, enabling the model to propose one or more action regions in a single forward pass. In line with this, we further design a grounding verifier to evaluate and select the most plausible action region from the candidates proposed for action execution. Extensive experiments show that GUI-Actor outperforms prior state-of-the-art methods on multiple GUI action grounding benchmarks, with improved generalization to unseen screen resolutions and layouts. Notably, GUI-Actor-7B even surpasses UI-TARS-72B (38.1) on ScreenSpot-Pro, achieving scores of 40.7 with Qwen2-VL and 44.6 with Qwen2.5-VL as backbones. Furthermore, by incorporating the verifier, we find that fine-tuning only the newly introduced action head (~100M parameters for 7B model) while keeping the VLM backbone frozen is sufficient to achieve performance comparable to previous state-of-the-art models, highlighting that GUI-Actor can endow the underlying VLM with effective grounding capabilities without compromising its general-purpose strengths.
>
---
#### [new 152] NTIRE 2025 Challenge on RAW Image Restoration and Super-Resolution
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像处理任务，旨在解决RAW图像的去模糊、去噪及超分辨率重建问题。主办方组织了NTIRE 2025挑战赛，吸引230人参与，最终45组提交方案。论文总结了当前RAW图像修复与超分领域的最新方法与成果。**

- **链接: [http://arxiv.org/pdf/2506.02197v1](http://arxiv.org/pdf/2506.02197v1)**

> **作者:** Marcos V. Conde; Radu Timofte; Zihao Lu; Xiangyu Kongand Xiaoxia Xingand Fan Wangand Suejin Hanand MinKyu Parkand Tianyu Zhangand Xin Luoand Yeda Chenand Dong Liuand Li Pangand Yuhang Yangand Hongzhong Wangand Xiangyong Caoand Ruixuan Jiangand Senyan Xuand Siyuan Jiangand Xueyang Fuand Zheng-Jun Zhaand Tianyu Haoand Yuhong Heand Ruoqi Liand Yueqi Yangand Xiang Yuand Guanlan Hongand Minmin Yiand Yuanjia Chenand Liwen Zhangand Zijie Jinand Cheng Liand Lian Liuand Wei Songand Heng Sunand Yubo Wangand Jinghua Wangand Jiajie Luand Watchara Ruangsangand
>
> **备注:** CVPR 2025 - New Trends in Image Restoration and Enhancement (NTIRE)
>
> **摘要:** This paper reviews the NTIRE 2025 RAW Image Restoration and Super-Resolution Challenge, highlighting the proposed solutions and results. New methods for RAW Restoration and Super-Resolution could be essential in modern Image Signal Processing (ISP) pipelines, however, this problem is not as explored as in the RGB domain. The goal of this challenge is two fold, (i) restore RAW images with blur and noise degradations, (ii) upscale RAW Bayer images by 2x, considering unknown noise and blur. In the challenge, a total of 230 participants registered, and 45 submitted results during thee challenge period. This report presents the current state-of-the-art in RAW Restoration.
>
---
#### [new 153] Robust Federated Learning against Noisy Clients via Masked Optimization
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **简介: 该论文属于联邦学习任务，旨在解决客户端数据标注噪声影响模型性能的问题。作者提出了两阶段优化框架MaskedOptim，第一阶段检测高噪声客户端，第二阶段通过端到端标签修正机制提升数据质量，并采用几何中位数聚合策略增强模型鲁棒性。实验表明该方法在多种噪声场景下均表现优异。**

- **链接: [http://arxiv.org/pdf/2506.02079v1](http://arxiv.org/pdf/2506.02079v1)**

> **作者:** Xuefeng Jiang; Tian Wen; Zhiqin Yang; Lvhua Wu; Yufeng Chen; Sheng Sun; Yuwei Wang; Min Liu
>
> **备注:** Under review
>
> **摘要:** In recent years, federated learning (FL) has made significant advance in privacy-sensitive applications. However, it can be hard to ensure that FL participants provide well-annotated data for training. The corresponding annotations from different clients often contain complex label noise at varying levels. This label noise issue has a substantial impact on the performance of the trained models, and clients with greater noise levels can be largely attributed for this degradation. To this end, it is necessary to develop an effective optimization strategy to alleviate the adverse effects of these noisy clients.In this study, we present a two-stage optimization framework, MaskedOptim, to address this intricate label noise problem. The first stage is designed to facilitate the detection of noisy clients with higher label noise rates. The second stage focuses on rectifying the labels of the noisy clients' data through an end-to-end label correction mechanism, aiming to mitigate the negative impacts caused by misinformation within datasets. This is achieved by learning the potential ground-truth labels of the noisy clients' datasets via backpropagation. To further enhance the training robustness, we apply the geometric median based model aggregation instead of the commonly-used vanilla averaged model aggregation. We implement sixteen related methods and conduct evaluations on three image datasets and one text dataset with diverse label noise patterns for a comprehensive comparison. Extensive experimental results indicate that our proposed framework shows its robustness in different scenarios. Additionally, our label correction framework effectively enhances the data quality of the detected noisy clients' local datasets. % Our codes will be open-sourced to facilitate related research communities. Our codes are available via https://github.com/Sprinter1999/MaskedOptim .
>
---
#### [new 154] HiLO: High-Level Object Fusion for Autonomous Driving using Transformers
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于自动驾驶中的多传感器融合任务，旨在提升环境感知的准确性。针对现有方法计算复杂或性能不足的问题，作者提出HiLO，结合改进卡尔曼滤波与Transformer进行高层对象融合，在真实数据集上验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2506.02554v1](http://arxiv.org/pdf/2506.02554v1)**

> **作者:** Timo Osterburg; Franz Albers; Christopher Diehl; Rajesh Pushparaj; Torsten Bertram
>
> **备注:** 6 pages, accepted at IEEE Intelligent Vehicles Symposium (IV) 2025
>
> **摘要:** The fusion of sensor data is essential for a robust perception of the environment in autonomous driving. Learning-based fusion approaches mainly use feature-level fusion to achieve high performance, but their complexity and hardware requirements limit their applicability in near-production vehicles. High-level fusion methods offer robustness with lower computational requirements. Traditional methods, such as the Kalman filter, dominate this area. This paper modifies the Adapted Kalman Filter (AKF) and proposes a novel transformer-based high-level object fusion method called HiLO. Experimental results demonstrate improvements of $25.9$ percentage points in $\textrm{F}_1$ score and $6.1$ percentage points in mean IoU. Evaluation on a new large-scale real-world dataset demonstrates the effectiveness of the proposed approaches. Their generalizability is further validated by cross-domain evaluation between urban and highway scenarios. Code, data, and models are available at https://github.com/rst-tu-dortmund/HiLO .
>
---
#### [new 155] Rodrigues Network for Learning Robot Actions
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人动作学习任务，旨在解决现有神经网络缺乏对关节运动结构的归纳偏置问题。作者提出了基于罗德里格斯公式的可学习神经模块（Neural Rodrigues Operator）和专用网络RodriNet，增强了动作预测与模仿学习效果，应用于机械臂控制和手部重建，验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2506.02618v1](http://arxiv.org/pdf/2506.02618v1)**

> **作者:** Jialiang Zhang; Haoran Geng; Yang You; Congyue Deng; Pieter Abbeel; Jitendra Malik; Leonidas Guibas
>
> **摘要:** Understanding and predicting articulated actions is important in robot learning. However, common architectures such as MLPs and Transformers lack inductive biases that reflect the underlying kinematic structure of articulated systems. To this end, we propose the Neural Rodrigues Operator, a learnable generalization of the classical forward kinematics operation, designed to inject kinematics-aware inductive bias into neural computation. Building on this operator, we design the Rodrigues Network (RodriNet), a novel neural architecture specialized for processing actions. We evaluate the expressivity of our network on two synthetic tasks on kinematic and motion prediction, showing significant improvements compared to standard backbones. We further demonstrate its effectiveness in two realistic applications: (i) imitation learning on robotic benchmarks with the Diffusion Policy, and (ii) single-image 3D hand reconstruction. Our results suggest that integrating structured kinematic priors into the network architecture improves action learning in various domains.
>
---
#### [new 156] DPO Learning with LLMs-Judge Signal for Computer Use Agents
- **分类: cs.AI; cs.CV**

- **简介: 论文研究计算机使用代理（CUA）任务，旨在解决现有系统依赖云端推理带来的隐私和扩展性问题。工作提出一种可在本地运行的轻量视觉-语言模型，并引入LLM-as-Judge框架自动生成高质量强化学习数据。实验表明该方法在OS-World基准上优于基线模型，推动私密、高效且通用的GUI代理发展。**

- **链接: [http://arxiv.org/pdf/2506.03095v1](http://arxiv.org/pdf/2506.03095v1)**

> **作者:** Man Luo; David Cobbley; Xin Su; Shachar Rosenman; Vasudev Lal; Shao-Yen Tseng; Phillip Howard
>
> **摘要:** Computer use agents (CUA) are systems that automatically interact with graphical user interfaces (GUIs) to complete tasks. CUA have made significant progress with the advent of large vision-language models (VLMs). However, these agents typically rely on cloud-based inference with substantial compute demands, raising critical privacy and scalability concerns, especially when operating on personal devices. In this work, we take a step toward privacy-preserving and resource-efficient agents by developing a lightweight vision-language model that runs entirely on local machines. To train this compact agent, we introduce an LLM-as-Judge framework that automatically evaluates and filters synthetic interaction trajectories, producing high-quality data for reinforcement learning without human annotation. Experiments on the OS-World benchmark demonstrate that our fine-tuned local model outperforms existing baselines, highlighting a promising path toward private, efficient, and generalizable GUI agents.
>
---
#### [new 157] HIEGNet: A Heterogenous Graph Neural Network Including the Immune Environment in Glomeruli Classification
- **分类: cs.LG; cs.AI; cs.CV; q-bio.QM**

- **简介: 该论文属于医学图像分类任务，旨在解决肾小球健康状态分类问题。通过构建包含肾小球及其周围免疫细胞的异质图，并提出HIEGNet模型，有效整合免疫环境信息，提升了分类性能与患者间泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.02542v1](http://arxiv.org/pdf/2506.02542v1)**

> **作者:** Niklas Kormann; Masoud Ramuz; Zeeshan Nisar; Nadine S. Schaadt; Hendrik Annuth; Benjamin Doerr; Friedrich Feuerhake; Thomas Lampert; Johannes F. Lutzeyer
>
> **备注:** Accepted for poster presentation at MIDL 2025
>
> **摘要:** Graph Neural Networks (GNNs) have recently been found to excel in histopathology. However, an important histopathological task, where GNNs have not been extensively explored, is the classification of glomeruli health as an important indicator in nephropathology. This task presents unique difficulties, particularly for the graph construction, i.e., the identification of nodes, edges, and informative features. In this work, we propose a pipeline composed of different traditional and machine learning-based computer vision techniques to identify nodes, edges, and their corresponding features to form a heterogeneous graph. We then proceed to propose a novel heterogeneous GNN architecture for glomeruli classification, called HIEGNet, that integrates both glomeruli and their surrounding immune cells. Hence, HIEGNet is able to consider the immune environment of each glomerulus in its classification. Our HIEGNet was trained and tested on a dataset of Whole Slide Images from kidney transplant patients. Experimental results demonstrate that HIEGNet outperforms several baseline models and generalises best between patients among all baseline models. Our implementation is publicly available at https://github.com/nklsKrmnn/HIEGNet.git.
>
---
#### [new 158] Interaction Field Matching: Overcoming Limitations of Electrostatic Models
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于数据生成与传输任务，旨在解决电场匹配模型（EFM）在建模复杂电场时的局限性。作者提出交互场匹配（IFM），扩展了EFM的适用范围，并设计了一种受夸克相互作用启发的交互场模型，有效解决了电场建模问题，验证于多个数据和图像传输实验。**

- **链接: [http://arxiv.org/pdf/2506.02950v1](http://arxiv.org/pdf/2506.02950v1)**

> **作者:** Stepan I. Manukhov; Alexander Kolesov; Vladimir V. Palyulin; Alexander Korotin
>
> **摘要:** Electrostatic field matching (EFM) has recently appeared as a novel physics-inspired paradigm for data generation and transfer using the idea of an electric capacitor. However, it requires modeling electrostatic fields using neural networks, which is non-trivial because of the necessity to take into account the complex field outside the capacitor plates. In this paper, we propose Interaction Field Matching (IFM), a generalization of EFM which allows using general interaction fields beyond the electrostatic one. Furthermore, inspired by strong interactions between quarks and antiquarks in physics, we design a particular interaction field realization which solves the problems which arise when modeling electrostatic fields in EFM. We show the performance on a series of toy and image data transfer problems.
>
---
#### [new 159] A Tree-guided CNN for image super-resolution
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像超分辨率任务，旨在提升图像恢复质量。现有方法难以有效利用网络结构中的关键层信息。作者提出TSRNet，采用树形架构增强关键节点作用，并引入余弦变换获取跨域信息，结合Adan优化器提升训练效果，实验证明其在高质量图像恢复上具有优势。**

- **链接: [http://arxiv.org/pdf/2506.02585v1](http://arxiv.org/pdf/2506.02585v1)**

> **作者:** Chunwei Tian; Mingjian Song; Xiaopeng Fan; Xiangtao Zheng; Bob Zhang; David Zhang
>
> **备注:** This paper has been accepted for publication in IEEE Transactions on Consumer Electronics. 10 pages, 6 figures. Its code can be obtained at https://github.com/hellloxiaotian/TSRNet
>
> **摘要:** Deep convolutional neural networks can extract more accurate structural information via deep architectures to obtain good performance in image super-resolution. However, it is not easy to find effect of important layers in a single network architecture to decrease performance of super-resolution. In this paper, we design a tree-guided CNN for image super-resolution (TSRNet). It uses a tree architecture to guide a deep network to enhance effect of key nodes to amplify the relation of hierarchical information for improving the ability of recovering images. To prevent insufficiency of the obtained structural information, cosine transform techniques in the TSRNet are used to extract cross-domain information to improve the performance of image super-resolution. Adaptive Nesterov momentum optimizer (Adan) is applied to optimize parameters to boost effectiveness of training a super-resolution model. Extended experiments can verify superiority of the proposed TSRNet for restoring high-quality images. Its code can be obtained at https://github.com/hellloxiaotian/TSRNet.
>
---
#### [new 160] Dynamic mapping from static labels: remote sensing dynamic sample generation with temporal-spectral embedding
- **分类: eess.IV; cs.CV; cs.MM**

- **简介: 论文属于遥感地理映射任务，旨在解决静态样本快速过时的问题。工作提出TasGen框架，通过时间-光谱嵌入自动生动生成动态样本，模拟地表变化，减少人工标注需求。**

- **链接: [http://arxiv.org/pdf/2506.02574v1](http://arxiv.org/pdf/2506.02574v1)**

> **作者:** Shuai Yuan; Shuang Chen; Tianwu Lin; Jie Wang; Peng Gong
>
> **摘要:** Accurate remote sensing geographic mapping depends heavily on representative and timely sample data. However, rapid changes in land surface dynamics necessitate frequent updates, quickly rendering previously collected samples obsolete and imposing significant labor demands for continuous manual updates. In this study, we aim to address this problem by dynamic sample generation using existing single-date static labeled samples. We introduce TasGen, a two-stage automated framework to automatically generate dynamic samples, designed to simultaneously model spectral and temporal dependencies in time-series remote sensing imagery via temporal-spectral embedding, capturing land surface changes without additional manual annotations.
>
---
#### [new 161] Multi-modal brain MRI synthesis based on SwinUNETR
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像合成任务，旨在解决临床中脑部MRI多模态成像中部分模态缺失的问题。作者应用SwinUNETR网络，结合Transformer与CNN优势，实现对缺失模态的高质量合成，提升生成图像的细节与诊断价值。**

- **链接: [http://arxiv.org/pdf/2506.02467v1](http://arxiv.org/pdf/2506.02467v1)**

> **作者:** Haowen Pang; Weiyan Guo; Chuyang Ye
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Multi-modal brain magnetic resonance imaging (MRI) plays a crucial role in clinical diagnostics by providing complementary information across different imaging modalities. However, a common challenge in clinical practice is missing MRI modalities. In this paper, we apply SwinUNETR to the synthesize of missing modalities in brain MRI. SwinUNETR is a novel neural network architecture designed for medical image analysis, integrating the strengths of Swin Transformer and convolutional neural networks (CNNs). The Swin Transformer, a variant of the Vision Transformer (ViT), incorporates hierarchical feature extraction and window-based self-attention mechanisms, enabling it to capture both local and global contextual information effectively. By combining the Swin Transformer with CNNs, SwinUNETR merges global context awareness with detailed spatial resolution. This hybrid approach addresses the challenges posed by the varying modality characteristics and complex brain structures, facilitating the generation of accurate and realistic synthetic images. We evaluate the performance of SwinUNETR on brain MRI datasets and demonstrate its superior capability in generating clinically valuable images. Our results show significant improvements in image quality, anatomical consistency, and diagnostic value.
>
---
#### [new 162] Is PMBOK Guide the Right Fit for AI? Re-evaluating Project Management in the Face of Artificial Intelligence Projects
- **分类: cs.SE; cs.CV; D.2.9; I.4**

- **简介: 该论文任务是评估PMBOK指南在人工智能项目中的适用性。它要解决的问题是传统项目管理框架在AI项目中存在局限，如数据管理、迭代开发和伦理问题不足。论文分析了这些差距，并提出了整合数据生命周期、采用敏捷方法及嵌入伦理考量的改进方案。**

- **链接: [http://arxiv.org/pdf/2506.02214v1](http://arxiv.org/pdf/2506.02214v1)**

> **作者:** Alexey Burdakov; Max Jaihyun Ahn
>
> **备注:** 9 pages, 1 figure
>
> **摘要:** This paper critically evaluates the applicability of the Project Management Body of Knowledge (PMBOK) Guide framework to Artificial Intelligence (AI) software projects, highlighting key limitations and proposing tailored adaptations. Unlike traditional projects, AI initiatives rely heavily on complex data, iterative experimentation, and specialized expertise while navigating significant ethical considerations. Our analysis identifies gaps in the PMBOK Guide, including its limited focus on data management, insufficient support for iterative development, and lack of guidance on ethical and multidisciplinary challenges. To address these deficiencies, we recommend integrating data lifecycle management, adopting iterative and AI project management frameworks, and embedding ethical considerations within project planning and execution. Additionally, we explore alternative approaches that better align with AI's dynamic and exploratory nature. We aim to enhance project management practices for AI software projects by bridging these gaps.
>
---
#### [new 163] HumanRAM: Feed-forward Human Reconstruction and Animation Model using Transformers
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于3D人体重建与动画任务，旨在解决传统方法依赖复杂捕捉和耗时优化的问题。论文提出了HumanRAM，将人体重建与动画统一于Transformer框架中，通过显式姿态条件实现高质量、可控的单目或稀疏图像输入重建与动画生成。**

- **链接: [http://arxiv.org/pdf/2506.03118v1](http://arxiv.org/pdf/2506.03118v1)**

> **作者:** Zhiyuan Yu; Zhe Li; Hujun Bao; Can Yang; Xiaowei Zhou
>
> **备注:** Accepted by SIGGRAPH 2025 (Conference Track). Project page: https://zju3dv.github.io/humanram/
>
> **摘要:** 3D human reconstruction and animation are long-standing topics in computer graphics and vision. However, existing methods typically rely on sophisticated dense-view capture and/or time-consuming per-subject optimization procedures. To address these limitations, we propose HumanRAM, a novel feed-forward approach for generalizable human reconstruction and animation from monocular or sparse human images. Our approach integrates human reconstruction and animation into a unified framework by introducing explicit pose conditions, parameterized by a shared SMPL-X neural texture, into transformer-based large reconstruction models (LRM). Given monocular or sparse input images with associated camera parameters and SMPL-X poses, our model employs scalable transformers and a DPT-based decoder to synthesize realistic human renderings under novel viewpoints and novel poses. By leveraging the explicit pose conditions, our model simultaneously enables high-quality human reconstruction and high-fidelity pose-controlled animation. Experiments show that HumanRAM significantly surpasses previous methods in terms of reconstruction accuracy, animation fidelity, and generalization performance on real-world datasets. Video results are available at https://zju3dv.github.io/humanram/.
>
---
#### [new 164] PartComposer: Learning and Composing Part-Level Concepts from Single-Image Examples
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决从单张图片学习并组合局部概念的问题。现有方法需大量数据或难以捕捉细粒度概念。作者提出PartComposer，通过动态数据合成和最大化去噪隐变量与结构化概念码的互信息，实现细粒度概念解耦与可控组合。**

- **链接: [http://arxiv.org/pdf/2506.03004v1](http://arxiv.org/pdf/2506.03004v1)**

> **作者:** Junyu Liu; R. Kenny Jones; Daniel Ritchie
>
> **摘要:** We present PartComposer: a framework for part-level concept learning from single-image examples that enables text-to-image diffusion models to compose novel objects from meaningful components. Existing methods either struggle with effectively learning fine-grained concepts or require a large dataset as input. We propose a dynamic data synthesis pipeline generating diverse part compositions to address one-shot data scarcity. Most importantly, we propose to maximize the mutual information between denoised latents and structured concept codes via a concept predictor, enabling direct regulation on concept disentanglement and re-composition supervision. Our method achieves strong disentanglement and controllable composition, outperforming subject and part-level baselines when mixing concepts from the same, or different, object categories.
>
---
#### [new 165] SiamNAS: Siamese Surrogate Model for Dominance Relation Prediction in Multi-objective Neural Architecture Search
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 论文提出SiamNAS，通过Siamese网络构建代理模型预测架构间的支配关系，解决多目标神经架构搜索中计算代价高的问题。该方法在NAS-Bench-201上快速找到Pareto最优解，显著降低计算成本，属于多目标优化任务。**

- **链接: [http://arxiv.org/pdf/2506.02623v1](http://arxiv.org/pdf/2506.02623v1)**

> **作者:** Yuyang Zhou; Ferrante Neri; Yew-Soon Ong; Ruibin Bai
>
> **备注:** Genetic and Evolutionary Computation Conference (GECCO' 25)
>
> **摘要:** Modern neural architecture search (NAS) is inherently multi-objective, balancing trade-offs such as accuracy, parameter count, and computational cost. This complexity makes NAS computationally expensive and nearly impossible to solve without efficient approximations. To address this, we propose a novel surrogate modelling approach that leverages an ensemble of Siamese network blocks to predict dominance relationships between candidate architectures. Lightweight and easy to train, the surrogate achieves 92% accuracy and replaces the crowding distance calculation in the survivor selection strategy with a heuristic rule based on model size. Integrated into a framework termed SiamNAS, this design eliminates costly evaluations during the search process. Experiments on NAS-Bench-201 demonstrate the framework's ability to identify Pareto-optimal solutions with significantly reduced computational costs. The proposed SiamNAS identified a final non-dominated set containing the best architecture in NAS-Bench-201 for CIFAR-10 and the second-best for ImageNet, in terms of test error rate, within 0.01 GPU days. This proof-of-concept study highlights the potential of the proposed Siamese network surrogate model to generalise to multi-tasking optimisation, enabling simultaneous optimisation across tasks. Additionally, it offers opportunities to extend the approach for generating Sets of Pareto Sets (SOS), providing diverse Pareto-optimal solutions for heterogeneous task settings.
>
---
#### [new 166] Dual encoding feature filtering generalized attention UNET for retinal vessel segmentation
- **分类: eess.IV; cs.CV; I.4; I.5**

- **简介: 该论文属于医学图像分割任务，旨在解决视网膜血管分割中的训练数据不足、分布不均和特征提取不充分等问题。作者提出了DEFFA-UNet模型，引入双编码器结构、特征过滤融合模块及注意力引导的跳跃连接，提升分割精度与模型泛化能力，并通过创新的数据增强方法增强鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.02312v1](http://arxiv.org/pdf/2506.02312v1)**

> **作者:** Md Tauhidul Islam; Wu Da-Wen; Tang Qing-Qing; Zhao Kai-Yang; Yin Teng; Li Yan-Fei; Shang Wen-Yi; Liu Jing-Yu; Zhang Hai-Xian
>
> **摘要:** Retinal blood vessel segmentation is crucial for diagnosing ocular and cardiovascular diseases. Although the introduction of U-Net in 2015 by Olaf Ronneberger significantly advanced this field, yet issues like limited training data, imbalance data distribution, and inadequate feature extraction persist, hindering both the segmentation performance and optimal model generalization. Addressing these critical issues, the DEFFA-Unet is proposed featuring an additional encoder to process domain-invariant pre-processed inputs, thereby improving both richer feature encoding and enhanced model generalization. A feature filtering fusion module is developed to ensure the precise feature filtering and robust hybrid feature fusion. In response to the task-specific need for higher precision where false positives are very costly, traditional skip connections are replaced with the attention-guided feature reconstructing fusion module. Additionally, innovative data augmentation and balancing methods are proposed to counter data scarcity and distribution imbalance, further boosting the robustness and generalization of the model. With a comprehensive suite of evaluation metrics, extensive validations on four benchmark datasets (DRIVE, CHASEDB1, STARE, and HRF) and an SLO dataset (IOSTAR), demonstrate the proposed method's superiority over both baseline and state-of-the-art models. Particularly the proposed method significantly outperforms the compared methods in cross-validation model generalization.
>
---
#### [new 167] SynthRL: Scaling Visual Reasoning with Verifiable Data Synthesis
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于视觉推理任务，旨在解决现有RL训练数据不足和难度不够问题。作者提出了SynthRL方法，通过自动合成可验证的高质量训练数据，提升模型在复杂视觉数学推理任务上的表现。实验表明，该方法显著提高了模型在多个跨领域基准上的性能。**

- **链接: [http://arxiv.org/pdf/2506.02096v1](http://arxiv.org/pdf/2506.02096v1)**

> **作者:** Zijian Wu; Jinjie Ni; Xiangyan Liu; Zichen Liu; Hang Yan; Michael Qizhe Shieh
>
> **摘要:** Vision-language models (VLMs) trained via reinforcement learning with verifiable reward (RLVR) have shown notable progress in scaling test-time compute effectively. In this work, we investigate how synthesized RL data can further improve RLVR. To this end, we propose \textbf{SynthRL}-a scalable and guaranteed pipeline for automatic data scaling in reasoning-oriented RL training. SynthRL comprises three key stages: (1) selecting seed questions with appropriate distribution, (2) augmenting them into more challenging variants while preserving the original answers, and (3) a guaranteed verification stage that ensures near-perfect correctness and difficulty enhancement. Our empirical experiments demonstrate SynthRL's scalability and effectiveness. When applied to the MMK12 dataset, SynthRL synthesizes over 3.3K additional verifiable, challenging questions from approximately 8K seed samples. Models trained with our synthesized data achieve consistent gains across five out-of-domain visual math reasoning benchmarks, with a significant improvement over baseline models trained on seed data alone. Notably, detailed analysis reveals that the gains are more pronounced on the most challenging evaluation samples, highlighting SynthRL's effectiveness in eliciting deeper and more complex reasoning patterns.
>
---
#### [new 168] Alzheimers Disease Classification in Functional MRI With 4D Joint Temporal-Spatial Kernels in Novel 4D CNN Model
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决阿尔茨海默病的早期诊断问题。作者提出了一种新的4D卷积神经网络模型，使用4D时空核提取功能磁共振成像（fMRI）中的时空特征，相比传统3D模型提升了分类性能，有助于更早检测疾病并改进干预手段。**

- **链接: [http://arxiv.org/pdf/2506.02060v1](http://arxiv.org/pdf/2506.02060v1)**

> **作者:** Javier Salazar Cavazos; Scott Peltier
>
> **备注:** Published in International Society for Magnetic Resonance in Medicine (ISMRM) 2025 under submission number 3398
>
> **摘要:** Previous works in the literature apply 3D spatial-only models on 4D functional MRI data leading to possible sub-par feature extraction to be used for downstream tasks like classification. In this work, we aim to develop a novel 4D convolution network to extract 4D joint temporal-spatial kernels that not only learn spatial information but in addition also capture temporal dynamics. Experimental results show promising performance in capturing spatial-temporal data in functional MRI compared to 3D models. The 4D CNN model improves Alzheimers disease diagnosis for rs-fMRI data, enabling earlier detection and better interventions. Future research could explore task-based fMRI applications and regression tasks, enhancing understanding of cognitive performance and disease progression.
>
---
#### [new 169] Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态生成评估任务，旨在解决现有评估模型忽视文本到图像生成任务及缺乏大规模人类评估数据的问题。作者构建了包含双向生成任务数据的Minos-Corpus，提出数据选择与平衡、Mix-SFT训练方法，并应用DPO优化训练，最终开发出性能领先的多模态评估模型Minos。**

- **链接: [http://arxiv.org/pdf/2506.02494v1](http://arxiv.org/pdf/2506.02494v1)**

> **作者:** Junzhe Zhang; Huixuan Zhang; Xinyu Hu; Li Lin; Mingqi Gao; Shi Qiu; Xiaojun Wan
>
> **摘要:** Evaluation is important for multimodal generation tasks. With the rapid progress of MLLMs, there is growing interest in applying MLLMs to build general evaluation systems. However, existing work overlooks two aspects: (1) the development of evaluation capabilities for text-to-image (T2I) generation task, and (2) the incorporation of large-scale human evaluation data. In this paper, we introduce Minos-Corpus, a large-scale multimodal evaluation dataset that combines evaluation data from both human and GPT. The corpus contains evaluation data across both image-to-text(I2T) and T2I generation tasks. Based on this corpus, we propose Data Selection and Balance, Mix-SFT training methods, and apply DPO to develop Minos, a multimodal evaluation model built upon a 7B backbone. Minos achieves state-of-the-art (SoTA) performance among all open-source evaluation models of similar scale on the average of evaluation performance on all tasks, and outperforms all open-source and closed-source models on evaluation of T2I generation task. Extensive experiments demonstrate the importance of leveraging high-quality human evaluation data and jointly training on evaluation data from both I2T and T2I generation tasks.
>
---
#### [new 170] EWGN: Elastic Weight Generation and Context Switching in Deep Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于人工智能领域中的持续学习任务，旨在解决神经网络在切换任务时出现的灾难性遗忘问题。论文提出了一种名为Elastic Weight Generative Networks（EWGN）的新架构，通过一个辅助网络动态生成主网络的权重，实现任务间的上下文切换，从而减少任务间的干扰，提升模型对先前任务知识的保持能力。实验基于MNIST和Fashion-MNIST数据集，评估了不同网络结构与学习算法在持续学习场景下的表现。**

- **链接: [http://arxiv.org/pdf/2506.02065v1](http://arxiv.org/pdf/2506.02065v1)**

> **作者:** Shriraj P. Sawant; Krishna P. Miyapuram
>
> **摘要:** The ability to learn and retain a wide variety of tasks is a hallmark of human intelligence that has inspired research in artificial general intelligence. Continual learning approaches provide a significant step towards achieving this goal. It has been known that task variability and context switching are challenging for learning in neural networks. Catastrophic forgetting refers to the poor performance on retention of a previously learned task when a new task is being learned. Switching between different task contexts can be a useful approach to mitigate the same by preventing the interference between the varying task weights of the network. This paper introduces Elastic Weight Generative Networks (EWGN) as an idea for context switching between two different tasks. The proposed EWGN architecture uses an additional network that generates the weights of the primary network dynamically while consolidating the weights learned. The weight generation is input-dependent and thus enables context switching. Using standard computer vision datasets, namely MNIST and fashion-MNIST, we analyse the retention of previously learned task representations in Fully Connected Networks, Convolutional Neural Networks, and EWGN architectures with Stochastic Gradient Descent and Elastic Weight Consolidation learning algorithms. Understanding dynamic weight generation and context-switching ability can be useful in enabling continual learning for improved performance.
>
---
#### [new 171] SemVink: Advancing VLMs' Semantic Understanding of Optical Illusions via Visual Global Thinking
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉-语言模型（VLM）任务，旨在解决VLM在识别光学错觉和隐藏内容中的语义理解缺陷。作者构建了包含112张图像的HC-Bench基准测试，提出SemVink方法，通过缩小图像分辨率提升VLM准确率至99%以上，揭示其对低级视觉操作的缺失问题，并倡导多尺度处理模型的发展。**

- **链接: [http://arxiv.org/pdf/2506.02803v1](http://arxiv.org/pdf/2506.02803v1)**

> **作者:** Sifan Li; Yujun Cai; Yiwei Wang
>
> **摘要:** Vision-language models (VLMs) excel in semantic tasks but falter at a core human capability: detecting hidden content in optical illusions or AI-generated images through perceptual adjustments like zooming. We introduce HC-Bench, a benchmark of 112 images with hidden text, objects, and illusions, revealing that leading VLMs achieve near-zero accuracy (0-5.36%)-even with explicit prompting. Humans resolve such ambiguities instinctively, yet VLMs fail due to an overreliance on high-level semantics. Strikingly, we propose SemVink (Semantic Visual Thinking) by simply scaling images to low resolutions (32-128 pixels), which unlocks >99% accuracy by eliminating redundant visual noise. This exposes a critical architectural flaw: VLMs prioritize abstract reasoning over low-level visual operations crucial for real-world robustness. Our work urges a shift toward hybrid models integrating multi-scale processing, bridging the gap between computational vision and human cognition for applications in medical imaging, security, and beyond.
>
---
#### [new 172] Rethinking Machine Unlearning in Image Generation Models
- **分类: cs.AI; cs.CL; cs.CR; cs.CV**

- **简介: 该论文属于图像生成模型中的机器遗忘任务，旨在解决数据隐私与内容安全问题。现有方法存在任务不明确、评估框架和指标不可靠等挑战。作者提出了CatIGMU任务分类框架、EvalIGMU评估框架及DataIGMU高质量数据集，以提升算法设计与评估的可靠性。**

- **链接: [http://arxiv.org/pdf/2506.02761v1](http://arxiv.org/pdf/2506.02761v1)**

> **作者:** Renyang Liu; Wenjie Feng; Tianwei Zhang; Wei Zhou; Xueqi Cheng; See-Kiong Ng
>
> **备注:** Accepted by ACM CCS 2025
>
> **摘要:** With the surge and widespread application of image generation models, data privacy and content safety have become major concerns and attracted great attention from users, service providers, and policymakers. Machine unlearning (MU) is recognized as a cost-effective and promising means to address these challenges. Despite some advancements, image generation model unlearning (IGMU) still faces remarkable gaps in practice, e.g., unclear task discrimination and unlearning guidelines, lack of an effective evaluation framework, and unreliable evaluation metrics. These can hinder the understanding of unlearning mechanisms and the design of practical unlearning algorithms. We perform exhaustive assessments over existing state-of-the-art unlearning algorithms and evaluation standards, and discover several critical flaws and challenges in IGMU tasks. Driven by these limitations, we make several core contributions, to facilitate the comprehensive understanding, standardized categorization, and reliable evaluation of IGMU. Specifically, (1) We design CatIGMU, a novel hierarchical task categorization framework. It provides detailed implementation guidance for IGMU, assisting in the design of unlearning algorithms and the construction of testbeds. (2) We introduce EvalIGMU, a comprehensive evaluation framework. It includes reliable quantitative metrics across five critical aspects. (3) We construct DataIGM, a high-quality unlearning dataset, which can be used for extensive evaluations of IGMU, training content detectors for judgment, and benchmarking the state-of-the-art unlearning algorithms. With EvalIGMU and DataIGM, we discover that most existing IGMU algorithms cannot handle the unlearning well across different evaluation dimensions, especially for preservation and robustness. Code and models are available at https://github.com/ryliu68/IGMU.
>
---
#### [new 173] Surgical Foundation Model Leveraging Compression and Entropy Maximization for Image-Guided Surgical Assistance
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决微创手术视频理解中缺乏标注数据的问题。作者提出了一种新的自监督学习框架Compress-to-Explore（C2E），通过压缩和最大化熵来学习紧凑且信息丰富的表示，从而提升多种手术相关机器学习任务的性能。**

- **链接: [http://arxiv.org/pdf/2506.01980v1](http://arxiv.org/pdf/2506.01980v1)**

> **作者:** Lianhao Yin; Ozanan Meireles; Guy Rosman; Daniela Rus
>
> **摘要:** Real-time video understanding is critical to guide procedures in minimally invasive surgery (MIS). However, supervised learning approaches require large, annotated datasets that are scarce due to annotation efforts that are prohibitive, e.g., in medical fields. Although self-supervision methods can address such limitations, current self-supervised methods often fail to capture structural and physical information in a form that generalizes across tasks. We propose Compress-to-Explore (C2E), a novel self-supervised framework that leverages Kolmogorov complexity to learn compact, informative representations from surgical videos. C2E uses entropy-maximizing decoders to compress images while preserving clinically relevant details, improving encoder performance without labeled data. Trained on large-scale unlabeled surgical datasets, C2E demonstrates strong generalization across a variety of surgical ML tasks, such as workflow classification, tool-tissue interaction classification, segmentation, and diagnosis tasks, providing improved performance as a surgical visual foundation model. As we further show in the paper, the model's internal compact representation better disentangles features from different structural parts of images. The resulting performance improvements highlight the yet untapped potential of self-supervised learning to enhance surgical AI and improve outcomes in MIS.
>
---
#### [new 174] Rethinking Post-Unlearning Behavior of Large Vision-Language Models
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于机器遗忘任务，旨在解决大型视觉语言模型（LVLMs）在隐私数据遗忘后出现的不良响应问题。现有方法导致响应退化、幻觉或过度拒绝，影响生成质量。论文提出PUBG方法，引导模型生成既保护隐私又具信息量和视觉依据的响应，有效缓解遗忘副作用。**

- **链接: [http://arxiv.org/pdf/2506.02541v1](http://arxiv.org/pdf/2506.02541v1)**

> **作者:** Minsung Kim; Nakyeong Yang; Kyomin Jung
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Machine unlearning is used to mitigate the privacy risks of Large Vision-Language Models (LVLMs) arising from training on large-scale web data. However, existing unlearning methods often fail to carefully select substitute outputs for forget targets, resulting in Unlearning Aftermaths-undesirable behaviors such as degenerate, hallucinated, or excessively refused responses. We highlight that, especially for generative LVLMs, it is crucial to consider the quality and informativeness of post-unlearning responses rather than relying solely on naive suppression. To address this, we introduce a new unlearning task for LVLMs that requires models to provide privacy-preserving yet informative and visually grounded responses. We also propose PUBG, a novel unlearning method that explicitly guides post-unlearning behavior toward a desirable output distribution. Experiments show that, while existing methods suffer from Unlearning Aftermaths despite successfully preventing privacy violations, PUBG effectively mitigates these issues, generating visually grounded and informative responses without privacy leakage for forgotten targets.
>
---
#### [new 175] Simulate Any Radar: Attribute-Controllable Radar Simulation via Waveform Parameter Embedding
- **分类: eess.SP; cs.CV**

- **简介: 该论文属于雷达数据模拟任务，旨在解决现有雷达仿真方法依赖硬件参数、效率低及可控性差的问题。作者提出SA-Radar，通过波形参数嵌入实现属性可控的雷达张量生成，并构建ICFAR-Net网络与混合数据集，提升模拟数据的真实性与应用效果。**

- **链接: [http://arxiv.org/pdf/2506.03134v1](http://arxiv.org/pdf/2506.03134v1)**

> **作者:** Weiqing Xiao; Hao Huang; Chonghao Zhong; Yujie Lin; Nan Wang; Xiaoxue Chen; Zhaoxi Chen; Saining Zhang; Shuocheng Yang; Pierre Merriaux; Lei Lei; Hao Zhao
>
> **备注:** Code: https://github.com/zhuxing0/SA-Radar Project page: https://zhuxing0.github.io/projects/SA-Radar
>
> **摘要:** We present SA-Radar (Simulate Any Radar), a radar simulation approach that enables controllable and efficient generation of radar cubes conditioned on customizable radar attributes. Unlike prior generative or physics-based simulators, SA-Radar integrates both paradigms through a waveform-parameterized attribute embedding. We design ICFAR-Net, a 3D U-Net conditioned on radar attributes encoded via waveform parameters, which captures signal variations induced by different radar configurations. This formulation bypasses the need for detailed radar hardware specifications and allows efficient simulation of range-azimuth-Doppler (RAD) tensors across diverse sensor settings. We further construct a mixed real-simulated dataset with attribute annotations to robustly train the network. Extensive evaluations on multiple downstream tasks-including 2D/3D object detection and radar semantic segmentation-demonstrate that SA-Radar's simulated data is both realistic and effective, consistently improving model performance when used standalone or in combination with real data. Our framework also supports simulation in novel sensor viewpoints and edited scenes, showcasing its potential as a general-purpose radar data engine for autonomous driving applications. Code and additional materials are available at https://zhuxing0.github.io/projects/SA-Radar.
>
---
#### [new 176] Unrolling Nonconvex Graph Total Variation for Image Denoising
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像去噪任务，旨在解决传统方法在非凸正则化优化时易陷入局部最优的问题。论文提出了一种基于图的非凸全变分（NC-GTV）方法，结合ℓ2保真项构建整体凸目标函数，并设计线性时间ADMM算法进行优化，进一步将其展开为轻量级神经网络用于参数学习，实现了更优的去噪效果。**

- **链接: [http://arxiv.org/pdf/2506.02381v1](http://arxiv.org/pdf/2506.02381v1)**

> **作者:** Songlin Wei; Gene Cheung; Fei Chen; Ivan Selesnick
>
> **摘要:** Conventional model-based image denoising optimizations employ convex regularization terms, such as total variation (TV) that convexifies the $\ell_0$-norm to promote sparse signal representation. Instead, we propose a new non-convex total variation term in a graph setting (NC-GTV), such that when combined with an $\ell_2$-norm fidelity term for denoising, leads to a convex objective with no extraneous local minima. We define NC-GTV using a new graph variant of the Huber function, interpretable as a Moreau envelope. The crux is the selection of a parameter $a$ characterizing the graph Huber function that ensures overall objective convexity; we efficiently compute $a$ via an adaptation of Gershgorin Circle Theorem (GCT). To minimize the convex objective, we design a linear-time algorithm based on Alternating Direction Method of Multipliers (ADMM) and unroll it into a lightweight feed-forward network for data-driven parameter learning. Experiments show that our method outperforms unrolled GTV and other representative image denoising schemes, while employing far fewer network parameters.
>
---
#### [new 177] FlexPainter: Flexible and Multi-View Consistent Texture Generation
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于3D纹理生成任务，旨在解决生成高质量、多视角一致且可控的纹理图问题。提出了FlexPainter方法，通过构建共享条件嵌入空间实现多模态控制，利用图像扩散先验生成多视角图像，并引入视图同步与自适应加权模块提升局部一致性，最终结合纹理补全和增强模型生成高分辨率纹理。**

- **链接: [http://arxiv.org/pdf/2506.02620v1](http://arxiv.org/pdf/2506.02620v1)**

> **作者:** Dongyu Yan; Leyi Wu; Jiantao Lin; Luozhou Wang; Tianshuo Xu; Zhifei Chen; Zhen Yang; Lie Xu; Shunsi Zhang; Yingcong Chen
>
> **备注:** 11 pages, 10 figures in main paper, 10 pages, 12 figures in supplementary
>
> **摘要:** Texture map production is an important part of 3D modeling and determines the rendering quality. Recently, diffusion-based methods have opened a new way for texture generation. However, restricted control flexibility and limited prompt modalities may prevent creators from producing desired results. Furthermore, inconsistencies between generated multi-view images often lead to poor texture generation quality. To address these issues, we introduce \textbf{FlexPainter}, a novel texture generation pipeline that enables flexible multi-modal conditional guidance and achieves highly consistent texture generation. A shared conditional embedding space is constructed to perform flexible aggregation between different input modalities. Utilizing such embedding space, we present an image-based CFG method to decompose structural and style information, achieving reference image-based stylization. Leveraging the 3D knowledge within the image diffusion prior, we first generate multi-view images simultaneously using a grid representation to enhance global understanding. Meanwhile, we propose a view synchronization and adaptive weighting module during diffusion sampling to further ensure local consistency. Finally, a 3D-aware texture completion model combined with a texture enhancement model is used to generate seamless, high-resolution texture maps. Comprehensive experiments demonstrate that our framework significantly outperforms state-of-the-art methods in both flexibility and generation quality.
>
---
## 更新

#### [replaced 001] Uneven Event Modeling for Partially Relevant Video Retrieval
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.00891v2](http://arxiv.org/pdf/2506.00891v2)**

> **作者:** Sa Zhu; Huashan Chen; Wanqian Zhang; Jinchao Zhang; Zexian Yang; Xiaoshuai Hao; Bo Li
>
> **备注:** Accepted by ICME 2025
>
> **摘要:** Given a text query, partially relevant video retrieval (PRVR) aims to retrieve untrimmed videos containing relevant moments, wherein event modeling is crucial for partitioning the video into smaller temporal events that partially correspond to the text. Previous methods typically segment videos into a fixed number of equal-length clips, resulting in ambiguous event boundaries. Additionally, they rely on mean pooling to compute event representations, inevitably introducing undesired misalignment. To address these, we propose an Uneven Event Modeling (UEM) framework for PRVR. We first introduce the Progressive-Grouped Video Segmentation (PGVS) module, to iteratively formulate events in light of both temporal dependencies and semantic similarity between consecutive frames, enabling clear event boundaries. Furthermore, we also propose the Context-Aware Event Refinement (CAER) module to refine the event representation conditioned the text's cross-attention. This enables event representations to focus on the most relevant frames for a given text, facilitating more precise text-video alignment. Extensive experiments demonstrate that our method achieves state-of-the-art performance on two PRVR benchmarks. Code is available at https://github.com/Sasa77777779/UEM.git.
>
---
#### [replaced 002] DC-ControlNet: Decoupling Inter- and Intra-Element Conditions in Image Generation with Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.14779v2](http://arxiv.org/pdf/2502.14779v2)**

> **作者:** Hongji Yang; Wencheng Han; Yucheng Zhou; Jianbing Shen
>
> **摘要:** In this paper, we introduce DC (Decouple)-ControlNet, a highly flexible and precisely controllable framework for multi-condition image generation. The core idea behind DC-ControlNet is to decouple control conditions, transforming global control into a hierarchical system that integrates distinct elements, contents, and layouts. This enables users to mix these individual conditions with greater flexibility, leading to more efficient and accurate image generation control. Previous ControlNet-based models rely solely on global conditions, which affect the entire image and lack the ability of element- or region-specific control. This limitation reduces flexibility and can cause condition misunderstandings in multi-conditional image generation. To address these challenges, we propose both intra-element and Inter-element Controllers in DC-ControlNet. The Intra-Element Controller handles different types of control signals within individual elements, accurately describing the content and layout characteristics of the object. For interactions between elements, we introduce the Inter-Element Controller, which accurately handles multi-element interactions and occlusion based on user-defined relationships. Extensive evaluations show that DC-ControlNet significantly outperforms existing ControlNet models and Layout-to-Image generative models in terms of control flexibility and precision in multi-condition control. Our project website is available at: https://um-lab.github.io/DC-ControlNet/
>
---
#### [replaced 003] OpenS2V-Nexus: A Detailed Benchmark and Million-Scale Dataset for Subject-to-Video Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20292v4](http://arxiv.org/pdf/2505.20292v4)**

> **作者:** Shenghai Yuan; Xianyi He; Yufan Deng; Yang Ye; Jinfa Huang; Bin Lin; Jiebo Luo; Li Yuan
>
> **备注:** Code and Dataset: https://github.com/PKU-YuanGroup/OpenS2V-Nexus
>
> **摘要:** Subject-to-Video (S2V) generation aims to create videos that faithfully incorporate reference content, providing enhanced flexibility in the production of videos. To establish the infrastructure for S2V generation, we propose OpenS2V-Nexus, consisting of (i) OpenS2V-Eval, a fine-grained benchmark, and (ii) OpenS2V-5M, a million-scale dataset. In contrast to existing S2V benchmarks inherited from VBench that focus on global and coarse-grained assessment of generated videos, OpenS2V-Eval focuses on the model's ability to generate subject-consistent videos with natural subject appearance and identity fidelity. For these purposes, OpenS2V-Eval introduces 180 prompts from seven major categories of S2V, which incorporate both real and synthetic test data. Furthermore, to accurately align human preferences with S2V benchmarks, we propose three automatic metrics, NexusScore, NaturalScore and GmeScore, to separately quantify subject consistency, naturalness, and text relevance in generated videos. Building on this, we conduct a comprehensive evaluation of 18 representative S2V models, highlighting their strengths and weaknesses across different content. Moreover, we create the first open-source large-scale S2V generation dataset OpenS2V-5M, which consists of five million high-quality 720P subject-text-video triples. Specifically, we ensure subject-information diversity in our dataset by (1) segmenting subjects and building pairing information via cross-video associations and (2) prompting GPT-Image-1 on raw frames to synthesize multi-view representations. Through OpenS2V-Nexus, we deliver a robust infrastructure to accelerate future S2V generation research.
>
---
#### [replaced 004] Likelihood-Scheduled Score-Based Generative Modeling for Fully 3D PET Image Reconstruction
- **分类: physics.med-ph; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.04339v2](http://arxiv.org/pdf/2412.04339v2)**

> **作者:** George Webber; Yuya Mizuno; Oliver D. Howes; Alexander Hammers; Andrew P. King; Andrew J. Reader
>
> **备注:** 12 pages, 14 figures. Author's accepted manuscript, IEEE Transactions on Medical Imaging
>
> **摘要:** Medical image reconstruction with pre-trained score-based generative models (SGMs) has advantages over other existing state-of-the-art deep-learned reconstruction methods, including improved resilience to different scanner setups and advanced image distribution modeling. SGM-based reconstruction has recently been applied to simulated positron emission tomography (PET) datasets, showing improved contrast recovery for out-of-distribution lesions relative to the state-of-the-art. However, existing methods for SGM-based reconstruction from PET data suffer from slow reconstruction, burdensome hyperparameter tuning and slice inconsistency effects (in 3D). In this work, we propose a practical methodology for fully 3D reconstruction that accelerates reconstruction and reduces the number of critical hyperparameters by matching the likelihood of an SGM's reverse diffusion process to a current iterate of the maximum-likelihood expectation maximization algorithm. Using the example of low-count reconstruction from simulated [$^{18}$F]DPA-714 datasets, we show our methodology can match or improve on the NRMSE and SSIM of existing state-of-the-art SGM-based PET reconstruction while reducing reconstruction time and the need for hyperparameter tuning. We evaluate our methodology against state-of-the-art supervised and conventional reconstruction algorithms. Finally, we demonstrate a first-ever implementation of SGM-based reconstruction for real 3D PET data, specifically [$^{18}$F]DPA-714 data, where we integrate perpendicular pre-trained SGMs to eliminate slice inconsistency issues.
>
---
#### [replaced 005] Ola: Pushing the Frontiers of Omni-Modal Language Model
- **分类: cs.CV; cs.CL; cs.MM; cs.SD; eess.AS; eess.IV**

- **链接: [http://arxiv.org/pdf/2502.04328v3](http://arxiv.org/pdf/2502.04328v3)**

> **作者:** Zuyan Liu; Yuhao Dong; Jiahui Wang; Ziwei Liu; Winston Hu; Jiwen Lu; Yongming Rao
>
> **摘要:** Recent advances in large language models, particularly following GPT-4o, have sparked increasing interest in developing omni-modal models capable of understanding more modalities. While some open-source alternatives have emerged, there is still a notable lag behind specialized single-modality models in performance. In this paper, we present Ola, an Omni-modal Language model that achieves competitive performance across image, video, and audio understanding compared to specialized counterparts, pushing the frontiers of the omni-modal language model to a large extent. We conduct a comprehensive exploration of architectural design, data curation, and training strategies essential for building a robust omni-modal model. Ola incorporates advanced visual understanding and audio recognition capabilities through several critical and effective improvements over mainstream baselines. Moreover, we rethink inter-modal relationships during omni-modal training, emphasizing cross-modal alignment with video as a central bridge, and propose a progressive training pipeline that begins with the most distinct modalities and gradually moves towards closer modality alignment. Extensive experiments demonstrate that Ola surpasses existing open omni-modal LLMs across all modalities while achieving highly competitive performance compared to state-of-the-art specialized models of similar sizes. We aim to make Ola a fully open omni-modal understanding solution to advance future research in this emerging field. Model weights, code, and data are open-sourced at https://github.com/Ola-Omni/Ola.
>
---
#### [replaced 006] Mitigating Visual Forgetting via Take-along Visual Conditioning for Multi-modal Long CoT Reasoning
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.13360v2](http://arxiv.org/pdf/2503.13360v2)**

> **作者:** Hai-Long Sun; Zhun Sun; Houwen Peng; Han-Jia Ye
>
> **备注:** Accepted to ACL 2025. The project page is available at https://sun-hailong.github.io/projects/TVC
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have demonstrated enhanced reasoning capabilities, evolving from Chain-of-Thought (CoT) prompting to advanced, product-oriented solutions like OpenAI o1. During our re-implementation of this model, we noticed that in multimodal tasks requiring visual input (e.g., geometry problems), Multimodal LLMs (MLLMs) struggle to maintain focus on the visual information, in other words, MLLMs suffer from a gradual decline in attention to visual information as reasoning progresses, causing text-over-relied outputs. To investigate this, we ablate image inputs during long-chain reasoning. Concretely, we truncate the reasoning process midway, then re-complete the reasoning process with the input image removed. We observe only a ~2% accuracy drop on MathVista's test-hard subset, revealing the model's textual outputs dominate the following reasoning process. Motivated by this, we propose Take-along Visual Conditioning (TVC), a strategy that shifts image input to critical reasoning stages and compresses redundant visual tokens via dynamic pruning. This methodology helps the model retain attention to the visual components throughout the reasoning. Our approach achieves state-of-the-art performance on average across five mathematical reasoning benchmarks (+3.4 points vs previous sota), demonstrating the effectiveness of TVC in enhancing multimodal reasoning systems.
>
---
#### [replaced 007] DeepSPV: A Deep Learning Pipeline for 3D Spleen Volume Estimation from 2D Ultrasound Images
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.11190v2](http://arxiv.org/pdf/2411.11190v2)**

> **作者:** Zhen Yuan; David Stojanovski; Lei Li; Alberto Gomez; Haran Jogeesvaran; Esther Puyol-Antón; Baba Inusa; Andrew P. King
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2308.08038
>
> **摘要:** Splenomegaly, the enlargement of the spleen, is an important clinical indicator for various associated medical conditions, such as sickle cell disease (SCD). Spleen length measured from 2D ultrasound is the most widely used metric for characterising spleen size. However, it is still considered a surrogate measure, and spleen volume remains the gold standard for assessing spleen size. Accurate spleen volume measurement typically requires 3D imaging modalities, such as computed tomography or magnetic resonance imaging, but these are not widely available, especially in the Global South which has a high prevalence of SCD. In this work, we introduce a deep learning pipeline, DeepSPV, for precise spleen volume estimation from single or dual 2D ultrasound images. The pipeline involves a segmentation network and a variational autoencoder for learning low-dimensional representations from the estimated segmentations. We investigate three approaches for spleen volume estimation and our best model achieves 86.62%/92.5% mean relative volume accuracy (MRVA) under single-view/dual-view settings, surpassing the performance of human experts. In addition, the pipeline can provide confidence intervals for the volume estimates as well as offering benefits in terms of interpretability, which further support clinicians in decision-making when identifying splenomegaly. We evaluate the full pipeline using a highly realistic synthetic dataset generated by a diffusion model, achieving an overall MRVA of 83.0% from a single 2D ultrasound image. Our proposed DeepSPV is the first work to use deep learning to estimate 3D spleen volume from 2D ultrasound images and can be seamlessly integrated into the current clinical workflow for spleen assessment.
>
---
#### [replaced 008] VectorPainter: Advanced Stylized Vector Graphics Synthesis Using Stroke-Style Priors
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.02962v4](http://arxiv.org/pdf/2405.02962v4)**

> **作者:** Juncheng Hu; Ximing Xing; Jing Zhang; Qian Yu
>
> **备注:** Accepted by 2025 IEEE International Conference on Multimedia and Expo (ICME). IEEE, 2025. Project page: https://hjc-owo.github.io/VectorPainterProject/
>
> **摘要:** We introduce VectorPainter, a novel framework designed for reference-guided text-to-vector-graphics synthesis. Based on our observation that the style of strokes can be an important aspect to distinguish different artists, our method reforms the task into synthesize a desired vector graphics by rearranging stylized strokes, which are vectorized from the reference images. Specifically, our method first converts the pixels of the reference image into a series of vector strokes, and then generates a vector graphic based on the input text description by optimizing the positions and colors of these vector strokes. To precisely capture the style of the reference image in the vectorized strokes, we propose an innovative vectorization method that employs an imitation learning strategy. To preserve the style of the strokes throughout the generation process, we introduce a style-preserving loss function. Extensive experiments have been conducted to demonstrate the superiority of our approach over existing works in stylized vector graphics synthesis, as well as the effectiveness of the various components of our method.
>
---
#### [replaced 009] OmniAudio: Generating Spatial Audio from 360-Degree Video
- **分类: eess.AS; cs.CV; cs.SD**

- **链接: [http://arxiv.org/pdf/2504.14906v3](http://arxiv.org/pdf/2504.14906v3)**

> **作者:** Huadai Liu; Tianyi Luo; Kaicheng Luo; Qikai Jiang; Peiwen Sun; Jialei Wang; Rongjie Huang; Qian Chen; Wen Wang; Xiangtai Li; Shiliang Zhang; Zhijie Yan; Zhou Zhao; Wei Xue
>
> **备注:** ICML 2025
>
> **摘要:** Traditional video-to-audio generation techniques primarily focus on perspective video and non-spatial audio, often missing the spatial cues necessary for accurately representing sound sources in 3D environments. To address this limitation, we introduce a novel task, 360V2SA, to generate spatial audio from 360-degree videos, specifically producing First-order Ambisonics (FOA) audio - a standard format for representing 3D spatial audio that captures sound directionality and enables realistic 3D audio reproduction. We first create Sphere360, a novel dataset tailored for this task that is curated from real-world data. We also design an efficient semi-automated pipeline for collecting and cleaning paired video-audio data. To generate spatial audio from 360-degree video, we propose a novel framework OmniAudio, which leverages self-supervised pre-training using both spatial audio data (in FOA format) and large-scale non-spatial data. Furthermore, OmniAudio features a dual-branch framework that utilizes both panoramic and perspective video inputs to capture comprehensive local and global information from 360-degree videos. Experimental results demonstrate that OmniAudio achieves state-of-the-art performance across both objective and subjective metrics on Sphere360. Code and datasets are available at https://github.com/liuhuadai/OmniAudio. The project website is available at https://OmniAudio-360V2SA.github.io.
>
---
#### [replaced 010] FIHA: Autonomous Hallucination Evaluation in Vision-Language Models with Davidson Scene Graphs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.13612v2](http://arxiv.org/pdf/2409.13612v2)**

> **作者:** Bowen Yan; Zhengsong Zhang; Liqiang Jing; Eftekhar Hossain; Xinya Du
>
> **备注:** Accepted by Findings of ACL 2025
>
> **摘要:** The rapid development of Large Vision-Language Models (LVLMs) often comes with widespread hallucination issues, making cost-effective and comprehensive assessments increasingly vital. Current approaches mainly rely on costly annotations and are not comprehensive -- in terms of evaluating all aspects such as relations, attributes, and dependencies between aspects. Therefore, we introduce the FIHA (autonomous Fine-graIned Hallucination evAluation evaluation in LVLMs), which could access hallucination LVLMs in the LLM-free and annotation-free way and model the dependency between different types of hallucinations. FIHA can generate Q&A pairs on any image dataset at minimal cost, enabling hallucination assessment from both image and caption. Based on this approach, we introduce a benchmark called FIHA-v1, which consists of diverse questions on various images from MSCOCO and Foggy. Furthermore, we use the Davidson Scene Graph (DSG) to organize the structure among Q&A pairs, in which we can increase the reliability of the evaluation. We evaluate representative models using FIHA-v1, highlighting their limitations and challenges. We released our code and data.
>
---
#### [replaced 011] LeYOLO, New Embedded Architecture for Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.14239v2](http://arxiv.org/pdf/2406.14239v2)**

> **作者:** Lilian Hollard; Lucas Mohimont; Nathalie Gaveau; Luiz Angelo Steffenel
>
> **备注:** https://crv.pubpub.org/pub/sae4lpdf
>
> **摘要:** Efficient computation in deep neural networks is crucial for real-time object detection. However, recent advancements primarily result from improved high-performing hardware rather than improving parameters and FLOP efficiency. This is especially evident in the latest YOLO architectures, where speed is prioritized over lightweight design. As a result, object detection models optimized for low-resource environments like microcontrollers have received less attention. For devices with limited computing power, existing solutions primarily rely on SSDLite or combinations of low-parameter classifiers, creating a noticeable gap between YOLO-like architectures and truly efficient lightweight detectors. This raises a key question: Can a model optimized for parameter and FLOP efficiency achieve accuracy levels comparable to mainstream YOLO models? To address this, we introduce two key contributions to object detection models using MSCOCO as a base validation set. First, we propose LeNeck, a general-purpose detection framework that maintains inference speed comparable to SSDLite while significantly improving accuracy and reducing parameter count. Second, we present LeYOLO, an efficient object detection model designed to enhance computational efficiency in YOLO-based architectures. LeYOLO effectively bridges the gap between SSDLite-based detectors and YOLO models, offering high accuracy in a model as compact as MobileNets. Both contributions are particularly well-suited for mobile, embedded, and ultra-low-power devices, including microcontrollers, where computational efficiency is critical.
>
---
#### [replaced 012] MMLA: Multi-Environment, Multi-Species, Low-Altitude Drone Dataset
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.07744v2](http://arxiv.org/pdf/2504.07744v2)**

> **作者:** Jenna Kline; Samuel Stevens; Guy Maalouf; Camille Rondeau Saint-Jean; Dat Nguyen Ngoc; Majid Mirmehdi; David Guerin; Tilo Burghardt; Elzbieta Pastucha; Blair Costelloe; Matthew Watson; Thomas Richardson; Ulrik Pagh Schultz Lundquist
>
> **备注:** Accepted at CVPR Workshop, CV4Animals 2025
>
> **摘要:** Real-time wildlife detection in drone imagery supports critical ecological and conservation monitoring. However, standard detection models like YOLO often fail to generalize across locations and struggle with rare species, limiting their use in automated drone deployments. We present MMLA, a novel multi-environment, multi-species, low-altitude drone dataset collected across three sites (Ol Pejeta Conservancy and Mpala Research Centre in Kenya, and The Wilds in Ohio), featuring six species (zebras, giraffes, onagers, and African wild dogs). The dataset contains 811K annotations from 37 high-resolution videos. Baseline YOLO models show performance disparities across locations while fine-tuning YOLOv11m on MMLA improves mAP50 to 82%, a 52-point gain over baseline. Our results underscore the need for diverse training data to enable robust animal detection in autonomous drone systems.
>
---
#### [replaced 013] Learning on Model Weights using Tree Experts
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.13569v3](http://arxiv.org/pdf/2410.13569v3)**

> **作者:** Eliahu Horwitz; Bar Cavia; Jonathan Kahana; Yedid Hoshen
>
> **备注:** CVPR 2025. Project page: https://horwitz.ai/probex/
>
> **摘要:** The number of publicly available models is rapidly increasing, yet most remain undocumented. Users looking for suitable models for their tasks must first determine what each model does. Training machine learning models to infer missing documentation directly from model weights is challenging, as these weights often contain significant variation unrelated to model functionality (denoted nuisance). Here, we identify a key property of real-world models: most public models belong to a small set of Model Trees, where all models within a tree are fine-tuned from a common ancestor (e.g., a foundation model). Importantly, we find that within each tree there is less nuisance variation between models. Concretely, while learning across Model Trees requires complex architectures, even a linear classifier trained on a single model layer often works within trees. While effective, these linear classifiers are computationally expensive, especially when dealing with larger models that have many parameters. To address this, we introduce Probing Experts (ProbeX), a theoretically motivated and lightweight method. Notably, ProbeX is the first probing method specifically designed to learn from the weights of a single hidden model layer. We demonstrate the effectiveness of ProbeX by predicting the categories in a model's training dataset based only on its weights. Excitingly, ProbeX can map the weights of Stable Diffusion into a weight-language embedding space, enabling model search via text, i.e., zero-shot model classification.
>
---
#### [replaced 014] SyncSDE: A Probabilistic Framework for Diffusion Synchronization
- **分类: cs.LG; cs.CV; cs.GR; stat.ML**

- **链接: [http://arxiv.org/pdf/2503.21555v2](http://arxiv.org/pdf/2503.21555v2)**

> **作者:** Hyunjun Lee; Hyunsoo Lee; Sookwan Han
>
> **备注:** Accepted to CVPR2025. Project Page: https://hjl1013.github.io/SyncSDE/
>
> **摘要:** There have been many attempts to leverage multiple diffusion models for collaborative generation, extending beyond the original domain. A prominent approach involves synchronizing multiple diffusion trajectories by mixing the estimated scores to artificially correlate the generation processes. However, existing methods rely on naive heuristics, such as averaging, without considering task specificity. These approaches do not clarify why such methods work and often produce suboptimal results when a heuristic suitable for one task is blindly applied to others. In this paper, we present a probabilistic framework for analyzing why diffusion synchronization works and reveal where heuristics should be focused; modeling correlations between multiple trajectories and adapting them to each specific task. We further identify optimal correlation models per task, achieving better results than previous approaches that apply a single heuristic across all tasks without justification.
>
---
#### [replaced 015] GETReason: Enhancing Image Context Extraction through Hierarchical Multi-Agent Reasoning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21863v3](http://arxiv.org/pdf/2505.21863v3)**

> **作者:** Shikhhar Siingh; Abhinav Rawat; Chitta Baral; Vivek Gupta
>
> **摘要:** Publicly significant images from events hold valuable contextual information, crucial for journalism and education. However, existing methods often struggle to extract this relevance accurately. To address this, we introduce GETReason (Geospatial Event Temporal Reasoning), a framework that moves beyond surface-level image descriptions to infer deeper contextual meaning. We propose that extracting global event, temporal, and geospatial information enhances understanding of an image's significance. Additionally, we introduce GREAT (Geospatial Reasoning and Event Accuracy with Temporal Alignment), a new metric for evaluating reasoning-based image understanding. Our layered multi-agent approach, assessed using a reasoning-weighted metric, demonstrates that meaningful insights can be inferred, effectively linking images to their broader event context.
>
---
#### [replaced 016] NeurIPS 2023 Competition: Privacy Preserving Federated Learning Document VQA
- **分类: cs.LG; cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.03730v2](http://arxiv.org/pdf/2411.03730v2)**

> **作者:** Marlon Tobaben; Mohamed Ali Souibgui; Rubèn Tito; Khanh Nguyen; Raouf Kerkouche; Kangsoo Jung; Joonas Jälkö; Lei Kang; Andrey Barsky; Vincent Poulain d'Andecy; Aurélie Joseph; Aashiq Muhamed; Kevin Kuo; Virginia Smith; Yusuke Yamasaki; Takumi Fukami; Kenta Niwa; Iifan Tyou; Hiro Ishii; Rio Yokota; Ragul N; Rintu Kutum; Josep Llados; Ernest Valveny; Antti Honkela; Mario Fritz; Dimosthenis Karatzas
>
> **备注:** 33 pages, 7 figures; published in TMLR 06/2025 https://openreview.net/forum?id=3HKNwejEEq
>
> **摘要:** The Privacy Preserving Federated Learning Document VQA (PFL-DocVQA) competition challenged the community to develop provably private and communication-efficient solutions in a federated setting for a real-life use case: invoice processing. The competition introduced a dataset of real invoice documents, along with associated questions and answers requiring information extraction and reasoning over the document images. Thereby, it brings together researchers and expertise from the document analysis, privacy, and federated learning communities. Participants fine-tuned a pre-trained, state-of-the-art Document Visual Question Answering model provided by the organizers for this new domain, mimicking a typical federated invoice processing setup. The base model is a multi-modal generative language model, and sensitive information could be exposed through either the visual or textual input modality. Participants proposed elegant solutions to reduce communication costs while maintaining a minimum utility threshold in track 1 and to protect all information from each document provider using differential privacy in track 2. The competition served as a new testbed for developing and testing private federated learning methods, simultaneously raising awareness about privacy within the document image analysis and recognition community. Ultimately, the competition analysis provides best practices and recommendations for successfully running privacy-focused federated learning challenges in the future.
>
---
#### [replaced 017] Concept Corrector: Erase concepts on the fly for text-to-image diffusion models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.16368v3](http://arxiv.org/pdf/2502.16368v3)**

> **作者:** Zheling Meng; Bo Peng; Xiaochuan Jin; Yueming Lyu; Wei Wang; Jing Dong; Tieniu Tan
>
> **摘要:** Text-to-image diffusion models have demonstrated the underlying risk of generating various unwanted content, such as sexual elements. To address this issue, the task of concept erasure has been introduced, aiming to erase any undesired concepts that the models can generate. Previous methods, whether training-based or training-free, have primarily focused on the input side, i.e., texts. However, they often suffer from incomplete erasure due to limitations in the generalization from limited prompts to diverse image content. In this paper, motivated by the notion that concept erasure on the output side, i.e., generated images, may be more direct and effective, we propose Concept Corrector. It checks target concepts based on visual features provided by final generated images predicted at certain time steps. Further, it incorporates Concept Removal Attention to erase generated concept features. It overcomes the limitations of existing methods, which are either unable to remove the concept features that have been generated in images or rely on the assumption that the related concept words are contained in input prompts. In the whole pipeline, our method changes no model parameters and only requires a given target concept as well as the corresponding replacement content, which is easy to implement. To the best of our knowledge, this is the first erasure method based on intermediate-generated images, achieving the ability to erase concepts on the fly. The experiments on various concepts demonstrate its impressive erasure performance.
>
---
#### [replaced 018] Unsupervised Time-Series Signal Analysis with Autoencoders and Vision Transformers: A Review of Architectures and Applications
- **分类: cs.LG; cs.AI; cs.CV; eess.SP; I.5.4; I.2.6; H.2.8**

- **链接: [http://arxiv.org/pdf/2504.16972v2](http://arxiv.org/pdf/2504.16972v2)**

> **作者:** Hossein Ahmadi; Sajjad Emdadi Mahdimahalleh; Arman Farahat; Banafsheh Saffari
>
> **摘要:** The rapid growth of unlabeled time-series data in domains such as wireless communications, radar, biomedical engineering, and the Internet of Things (IoT) has driven advancements in unsupervised learning. This review synthesizes recent progress in applying autoencoders and vision transformers for unsupervised signal analysis, focusing on their architectures, applications, and emerging trends. We explore how these models enable feature extraction, anomaly detection, and classification across diverse signal types, including electrocardiograms, radar waveforms, and IoT sensor data. The review highlights the strengths of hybrid architectures and self-supervised learning, while identifying challenges in interpretability, scalability, and domain generalization. By bridging methodological innovations and practical applications, this work offers a roadmap for developing robust, adaptive models for signal intelligence.
>
---
#### [replaced 019] MoBluRF: Motion Deblurring Neural Radiance Fields for Blurry Monocular Video
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2312.13528v3](http://arxiv.org/pdf/2312.13528v3)**

> **作者:** Minh-Quan Viet Bui; Jongmin Park; Jihyong Oh; Munchurl Kim
>
> **备注:** Accepted to IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2025. The first two authors contributed equally to this work (equal contribution). The last two authors are co-corresponding authors. Please visit our project page at https://kaist-viclab.github.io/moblurf-site/
>
> **摘要:** Neural Radiance Fields (NeRF), initially developed for static scenes, have inspired many video novel view synthesis techniques. However, the challenge for video view synthesis arises from motion blur, a consequence of object or camera movements during exposure, which hinders the precise synthesis of sharp spatio-temporal views. In response, we propose a novel motion deblurring NeRF framework for blurry monocular video, called MoBluRF, consisting of a Base Ray Initialization (BRI) stage and a Motion Decomposition-based Deblurring (MDD) stage. In the BRI stage, we coarsely reconstruct dynamic 3D scenes and jointly initialize the base rays which are further used to predict latent sharp rays, using the inaccurate camera pose information from the given blurry frames. In the MDD stage, we introduce a novel Incremental Latent Sharp-rays Prediction (ILSP) approach for the blurry monocular video frames by decomposing the latent sharp rays into global camera motion and local object motion components. We further propose two loss functions for effective geometry regularization and decomposition of static and dynamic scene components without any mask supervision. Experiments show that MoBluRF outperforms qualitatively and quantitatively the recent state-of-the-art methods with large margins.
>
---
#### [replaced 020] Constant Rate Scheduling: Constant-Rate Distributional Change for Efficient Training and Sampling in Diffusion Models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.12188v3](http://arxiv.org/pdf/2411.12188v3)**

> **作者:** Shuntaro Okada; Kenji Doi; Ryota Yoshihashi; Hirokatsu Kataoka; Tomohiro Tanaka
>
> **备注:** 44 pages, 20 figures, 25 tables
>
> **摘要:** We propose a general approach to optimize noise schedules for training and sampling in diffusion models. Our approach optimizes the noise schedules to ensure a constant rate of change in the probability distribution of diffused data throughout the diffusion process. Any distance metric for measuring the probability-distributional change is applicable to our approach, and we introduce three distance metrics. We evaluated the effectiveness of our approach on unconditional and class-conditional image-generation tasks using the LSUN (Horse, Bedroom, Church), ImageNet, FFHQ, and CIFAR10 datasets. Through extensive experiments, we confirmed that our approach broadly improves the performance of pixel-space and latent-space diffusion models regardless of the dataset, sampler, and number of function evaluations ranging from 5 to 250. Notably, by using our approach for optimizing both training and sampling schedules, we achieved a state-of-the-art FID score of 2.03 without sacrificing mode coverage on LSUN Horse 256 $\times$ 256.
>
---
#### [replaced 021] A Novel Benchmark for Few-Shot Semantic Segmentation in the Era of Foundation Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2401.11311v3](http://arxiv.org/pdf/2401.11311v3)**

> **作者:** Reda Bensaid; Vincent Gripon; François Leduc-Primeau; Lukas Mauch; Ghouthi Boukli Hacene; Fabien Cardinaux
>
> **摘要:** Few-shot semantic segmentation (FSS) is a crucial challenge in computer vision, driving extensive research into a diverse range of methods, from advanced meta-learning techniques to simple transfer learning baselines. With the emergence of vision foundation models (VFM) serving as generalist feature extractors, we seek to explore the adaptation of these models for FSS. While current FSS benchmarks focus on adapting pre-trained models to new tasks with few images, they emphasize in-domain generalization, making them less suitable for VFM trained on large-scale web datasets. To address this, we propose a novel realistic benchmark with a simple and straightforward adaptation process tailored for this task. Using this benchmark, we conduct a comprehensive comparative analysis of prominent VFM and semantic segmentation models. To evaluate their effectiveness, we leverage various adaption methods, ranging from linear probing to parameter efficient fine-tuning (PEFT) and full fine-tuning. Our findings show that models designed for segmentation can be outperformed by self-supervised (SSL) models. On the other hand, while PEFT methods yields competitive performance, they provide little discrepancy in the obtained results compared to other methods, highlighting the critical role of the feature extractor in determining results. To our knowledge, this is the first study on the adaptation of VFM for FSS.
>
---
#### [replaced 022] T-TAME: Trainable Attention Mechanism for Explaining Convolutional Networks and Vision Transformers
- **分类: cs.CV; cs.AI; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2403.04523v2](http://arxiv.org/pdf/2403.04523v2)**

> **作者:** Mariano V. Ntrougkas; Nikolaos Gkalelis; Vasileios Mezaris
>
> **备注:** Accepted
>
> **摘要:** The development and adoption of Vision Transformers and other deep-learning architectures for image classification tasks has been rapid. However, the "black box" nature of neural networks is a barrier to adoption in applications where explainability is essential. While some techniques for generating explanations have been proposed, primarily for Convolutional Neural Networks, adapting such techniques to the new paradigm of Vision Transformers is non-trivial. This paper presents T-TAME, Transformer-compatible Trainable Attention Mechanism for Explanations, a general methodology for explaining deep neural networks used in image classification tasks. The proposed architecture and training technique can be easily applied to any convolutional or Vision Transformer-like neural network, using a streamlined training approach. After training, explanation maps can be computed in a single forward pass; these explanation maps are comparable to or outperform the outputs of computationally expensive perturbation-based explainability techniques, achieving SOTA performance. We apply T-TAME to three popular deep learning classifier architectures, VGG-16, ResNet-50, and ViT-B-16, trained on the ImageNet dataset, and we demonstrate improvements over existing state-of-the-art explainability methods. A detailed analysis of the results and an ablation study provide insights into how the T-TAME design choices affect the quality of the generated explanation maps.
>
---
#### [replaced 023] OmniTalker: One-shot Real-time Text-Driven Talking Audio-Video Generation With Multimodal Style Mimicking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.02433v2](http://arxiv.org/pdf/2504.02433v2)**

> **作者:** Zhongjian Wang; Peng Zhang; Jinwei Qi; Guangyuan Wang; Chaonan Ji; Sheng Xu; Bang Zhang; Liefeng Bo
>
> **备注:** Project Page https://humanaigc.github.io/omnitalker
>
> **摘要:** Although significant progress has been made in audio-driven talking head generation, text-driven methods remain underexplored. In this work, we present OmniTalker, a unified framework that jointly generates synchronized talking audio-video content from input text while emulating the speaking and facial movement styles of the target identity, including speech characteristics, head motion, and facial dynamics. Our framework adopts a dual-branch diffusion transformer (DiT) architecture, with one branch dedicated to audio generation and the other to video synthesis. At the shallow layers, cross-modal fusion modules are introduced to integrate information between the two modalities. In deeper layers, each modality is processed independently, with the generated audio decoded by a vocoder and the video rendered using a GAN-based high-quality visual renderer. Leveraging the in-context learning capability of DiT through a masked-infilling strategy, our model can simultaneously capture both audio and visual styles without requiring explicit style extraction modules. Thanks to the efficiency of the DiT backbone and the optimized visual renderer, OmniTalker achieves real-time inference at 25 FPS. To the best of our knowledge, OmniTalker is the first one-shot framework capable of jointly modeling speech and facial styles in real time. Extensive experiments demonstrate its superiority over existing methods in terms of generation quality, particularly in preserving style consistency and ensuring precise audio-video synchronization, all while maintaining efficient inference.
>
---
#### [replaced 024] MTevent: A Multi-Task Event Camera Dataset for 6D Pose Estimation and Moving Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11282v2](http://arxiv.org/pdf/2505.11282v2)**

> **作者:** Shrutarv Awasthi; Anas Gouda; Sven Franke; Jérôme Rutinowski; Frank Hoffmann; Moritz Roidl
>
> **备注:** Accepted at 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW); Fifth International Workshop on Event-Based Vision
>
> **摘要:** Mobile robots are reaching unprecedented speeds, with platforms like Unitree B2, and Fraunhofer O3dyn achieving maximum speeds between 5 and 10 m/s. However, effectively utilizing such speeds remains a challenge due to the limitations of RGB cameras, which suffer from motion blur and fail to provide real-time responsiveness. Event cameras, with their asynchronous operation, and low-latency sensing, offer a promising alternative for high-speed robotic perception. In this work, we introduce MTevent, a dataset designed for 6D pose estimation and moving object detection in highly dynamic environments with large detection distances. Our setup consists of a stereo-event camera and an RGB camera, capturing 75 scenes, each on average 16 seconds, and featuring 16 unique objects under challenging conditions such as extreme viewing angles, varying lighting, and occlusions. MTevent is the first dataset to combine high-speed motion, long-range perception, and real-world object interactions, making it a valuable resource for advancing event-based vision in robotics. To establish a baseline, we evaluate the task of 6D pose estimation using NVIDIA's FoundationPose on RGB images, achieving an Average Recall of 0.22 with ground-truth masks, highlighting the limitations of RGB-based approaches in such dynamic settings. With MTevent, we provide a novel resource to improve perception models and foster further research in high-speed robotic vision. The dataset is available for download https://huggingface.co/datasets/anas-gouda/MTevent
>
---
#### [replaced 025] Hierarchical Relational Learning for Few-Shot Knowledge Graph Completion
- **分类: cs.LG; cs.CV; I.2**

- **链接: [http://arxiv.org/pdf/2209.01205v4](http://arxiv.org/pdf/2209.01205v4)**

> **作者:** Han Wu; Jie Yin; Bala Rajaratnam; Jianyuan Guo
>
> **备注:** Published at ICLR 2023
>
> **摘要:** Knowledge graphs (KGs) are powerful in terms of their inference abilities, but are also notorious for their incompleteness and long-tail distribution of relations. To address these challenges and expand the coverage of KGs, few-shot KG completion aims to make predictions for triplets involving novel relations when only a few training triplets are provided as reference. Previous methods have focused on designing local neighbor aggregators to learn entity-level information and/or imposing a potentially invalid sequential dependency assumption at the triplet level to learn meta relation information. However, pairwise triplet-level interactions and context-level relational information have been largely overlooked for learning meta representations of few-shot relations. In this paper, we propose a hierarchical relational learning method (HiRe) for few-shot KG completion. By jointly capturing three levels of relational information (entity-level, triplet-level and context-level), HiRe can effectively learn and refine meta representations of few-shot relations, and thus generalize well to new unseen relations. Extensive experiments on benchmark datasets validate the superiority of HiRe over state-of-the-art methods. The code can be found in https://github.com/alexhw15/HiRe.git.
>
---
#### [replaced 026] No Training, No Problem: Rethinking Classifier-Free Guidance for Diffusion Models
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2407.02687v2](http://arxiv.org/pdf/2407.02687v2)**

> **作者:** Seyedmorteza Sadat; Manuel Kansy; Otmar Hilliges; Romann M. Weber
>
> **备注:** Published as a conference paper at ICLR 2025
>
> **摘要:** Classifier-free guidance (CFG) has become the standard method for enhancing the quality of conditional diffusion models. However, employing CFG requires either training an unconditional model alongside the main diffusion model or modifying the training procedure by periodically inserting a null condition. There is also no clear extension of CFG to unconditional models. In this paper, we revisit the core principles of CFG and introduce a new method, independent condition guidance (ICG), which provides the benefits of CFG without the need for any special training procedures. Our approach streamlines the training process of conditional diffusion models and can also be applied during inference on any pre-trained conditional model. Additionally, by leveraging the time-step information encoded in all diffusion networks, we propose an extension of CFG, called time-step guidance (TSG), which can be applied to any diffusion model, including unconditional ones. Our guidance techniques are easy to implement and have the same sampling cost as CFG. Through extensive experiments, we demonstrate that ICG matches the performance of standard CFG across various conditional diffusion models. Moreover, we show that TSG improves generation quality in a manner similar to CFG, without relying on any conditional information.
>
---
#### [replaced 027] Open-world Machine Learning: A Systematic Review and Future Directions
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2403.01759v3](http://arxiv.org/pdf/2403.01759v3)**

> **作者:** Fei Zhu; Shijie Ma; Zhen Cheng; Xu-Yao Zhang; Zhaoxiang Zhang; Dacheng Tao; Cheng-Lin Liu
>
> **摘要:** Machine learning has achieved remarkable success in many applications. However, existing studies are largely based on the closed-world assumption, which assumes that the environment is stationary, and the model is fixed once deployed. In many real-world applications, this fundamental and rather naive assumption may not hold because an open environment is complex, dynamic, and full of unknowns. In such cases, rejecting unknowns, discovering novelties, and then continually learning them, could enable models to be safe and evolve continually as biological systems do. This article presents a holistic view of open-world machine learning by investigating unknown rejection, novelty discovery, and continual learning in a unified paradigm. The challenges, principles, and limitations of current methodologies are discussed in detail. Furthermore, widely used benchmarks, metrics, and performances are summarized. Finally, we discuss several potential directions for further progress in the field. By providing a comprehensive introduction to the emerging open-world machine learning paradigm, this article aims to help researchers build more powerful AI systems in their respective fields, and to promote the development of artificial general intelligence.
>
---
#### [replaced 028] Robust 6DoF Pose Estimation Against Depth Noise and a Comprehensive Evaluation on a Mobile Dataset
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2309.13570v5](http://arxiv.org/pdf/2309.13570v5)**

> **作者:** Zixun Huang; Keling Yao; Seth Z. Zhao; Chuanyu Pan; Allen Y. Yang
>
> **摘要:** Robust 6DoF pose estimation with mobile devices is the foundation for applications in robotics, augmented reality, and digital twin localization. In this paper, we extensively investigate the robustness of existing RGBD-based 6DoF pose estimation methods against varying levels of depth sensor noise. We highlight that existing 6DoF pose estimation methods suffer significant performance discrepancies due to depth measurement inaccuracies. In response to the robustness issue, we present a simple and effective transformer-based 6DoF pose estimation approach called DTTDNet, featuring a novel geometric feature filtering module and a Chamfer distance loss for training. Moreover, we advance the field of robust 6DoF pose estimation and introduce a new dataset -- Digital Twin Tracking Dataset Mobile (DTTD-Mobile), tailored for digital twin object tracking with noisy depth data from the mobile RGBD sensor suite of the Apple iPhone 14 Pro. Extensive experiments demonstrate that DTTDNet significantly outperforms state-of-the-art methods at least 4.32, up to 60.74 points in ADD metrics on the DTTD-Mobile. More importantly, our approach exhibits superior robustness to varying levels of measurement noise, setting a new benchmark for robustness to measurement noise. The project page is publicly available at https://openark-berkeley.github.io/DTTDNet/.
>
---
#### [replaced 029] Spatio-Temporal Fuzzy-oriented Multi-Modal Meta-Learning for Fine-grained Emotion Recognition
- **分类: cs.CV; cs.LG; cs.NE**

- **链接: [http://arxiv.org/pdf/2412.13541v4](http://arxiv.org/pdf/2412.13541v4)**

> **作者:** Jingyao Wang; Wenwen Qiang; Changwen Zheng; Fuchun Sun
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Fine-grained emotion recognition (FER) plays a vital role in various fields, such as disease diagnosis, personalized recommendations, and multimedia mining. However, existing FER methods face three key challenges in real-world applications: (i) they rely on large amounts of continuously annotated data to ensure accuracy since emotions are complex and ambiguous in reality, which is costly and time-consuming; (ii) they cannot capture the temporal heterogeneity caused by changing emotion patterns, because they usually assume that the temporal correlation within sampling periods is the same; (iii) they do not consider the spatial heterogeneity of different FER scenarios, that is, the distribution of emotion information in different data may have bias or interference. To address these challenges, we propose a Spatio-Temporal Fuzzy-oriented Multi-modal Meta-learning framework (ST-F2M). Specifically, ST-F2M first divides the multi-modal videos into multiple views, and each view corresponds to one modality of one emotion. Multiple randomly selected views for the same emotion form a meta-training task. Next, ST-F2M uses an integrated module with spatial and temporal convolutions to encode the data of each task, reflecting the spatial and temporal heterogeneity. Then it adds fuzzy semantic information to each task based on generalized fuzzy rules, which helps handle the complexity and ambiguity of emotions. Finally, ST-F2M learns emotion-related general meta-knowledge through meta-recurrent neural networks to achieve fast and robust fine-grained emotion recognition. Extensive experiments show that ST-F2M outperforms various state-of-the-art methods in terms of accuracy and model efficiency. In addition, we construct ablation studies and further analysis to explore why ST-F2M performs well.
>
---
#### [replaced 030] X-Driver: Explainable Autonomous Driving with Vision-Language Models
- **分类: cs.RO; cs.CL; cs.CV; cs.ET**

- **链接: [http://arxiv.org/pdf/2505.05098v2](http://arxiv.org/pdf/2505.05098v2)**

> **作者:** Wei Liu; Jiyuan Zhang; Binxiong Zheng; Yufeng Hu; Yingzhan Lin; Zengfeng Zeng
>
> **摘要:** End-to-end autonomous driving has advanced significantly, offering benefits such as system simplicity and stronger driving performance in both open-loop and closed-loop settings than conventional pipelines. However, existing frameworks still suffer from low success rates in closed-loop evaluations, highlighting their limitations in real-world deployment. In this paper, we introduce X-Driver, a unified multi-modal large language models(MLLMs) framework designed for closed-loop autonomous driving, leveraging Chain-of-Thought(CoT) and autoregressive modeling to enhance perception and decision-making. We validate X-Driver across multiple autonomous driving tasks using public benchmarks in CARLA simulation environment, including Bench2Drive[6]. Our experimental results demonstrate superior closed-loop performance, surpassing the current state-of-the-art(SOTA) while improving the interpretability of driving decisions. These findings underscore the importance of structured reasoning in end-to-end driving and establish X-Driver as a strong baseline for future research in closed-loop autonomous driving.
>
---
#### [replaced 031] Reinforcing Video Reasoning with Focused Thinking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.24718v2](http://arxiv.org/pdf/2505.24718v2)**

> **作者:** Jisheng Dang; Jingze Wu; Teng Wang; Xuanhui Lin; Nannan Zhu; Hongbo Chen; Wei-Shi Zheng; Meng Wang; Tat-Seng Chua
>
> **摘要:** Recent advancements in reinforcement learning, particularly through Group Relative Policy Optimization (GRPO), have significantly improved multimodal large language models for complex reasoning tasks. However, two critical limitations persist: 1) they often produce unfocused, verbose reasoning chains that obscure salient spatiotemporal cues and 2) binary rewarding fails to account for partially correct answers, resulting in high reward variance and inefficient learning. In this paper, we propose TW-GRPO, a novel framework that enhances visual reasoning with focused thinking and dense reward granularity. Specifically, we employs a token weighting mechanism that prioritizes tokens with high informational density (estimated by intra-group variance), suppressing redundant tokens like generic reasoning prefixes. Furthermore, we reformulate RL training by shifting from single-choice to multi-choice QA tasks, where soft rewards enable finer-grained gradient estimation by distinguishing partial correctness. Additionally, we propose question-answer inversion, a data augmentation strategy to generate diverse multi-choice samples from existing benchmarks. Experiments demonstrate state-of-the-art performance on several video reasoning and general understanding benchmarks. Notably, TW-GRPO achieves 50.4\% accuracy on CLEVRER (18.8\% improvement over Video-R1) and 65.8\% on MMVU. Our codes are available at \href{https://github.com/longmalongma/TW-GRPO}.
>
---
#### [replaced 032] Low-Resolution Self-Attention for Semantic Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2310.05026v3](http://arxiv.org/pdf/2310.05026v3)**

> **作者:** Yu-Huan Wu; Shi-Chen Zhang; Yun Liu; Le Zhang; Xin Zhan; Daquan Zhou; Jiashi Feng; Ming-Ming Cheng; Liangli Zhen
>
> **备注:** Accepted by IEEE TPAMI; 14 pages, 6 figures, 14 tables
>
> **摘要:** Semantic segmentation tasks naturally require high-resolution information for pixel-wise segmentation and global context information for class prediction. While existing vision transformers demonstrate promising performance, they often utilize high-resolution context modeling, resulting in a computational bottleneck. In this work, we challenge conventional wisdom and introduce the Low-Resolution Self-Attention (LRSA) mechanism to capture global context at a significantly reduced computational cost, i.e., FLOPs. Our approach involves computing self-attention in a fixed low-resolution space regardless of the input image's resolution, with additional 3x3 depth-wise convolutions to capture fine details in the high-resolution space. We demonstrate the effectiveness of our LRSA approach by building the LRFormer, a vision transformer with an encoder-decoder structure. Extensive experiments on the ADE20K, COCO-Stuff, and Cityscapes datasets demonstrate that LRFormer outperforms state-of-the-art models. Code is available at https://github.com/yuhuan-wu/LRFormer.
>
---
#### [replaced 033] Can Large Language Models Challenge CNNs in Medical Image Analysis?
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.23503v2](http://arxiv.org/pdf/2505.23503v2)**

> **作者:** Shibbir Ahmed; Shahnewaz Karim Sakib; Anindya Bijoy Das
>
> **摘要:** This study presents a multimodal AI framework designed for precisely classifying medical diagnostic images. Utilizing publicly available datasets, the proposed system compares the strengths of convolutional neural networks (CNNs) and different large language models (LLMs). This in-depth comparative analysis highlights key differences in diagnostic performance, execution efficiency, and environmental impacts. Model evaluation was based on accuracy, F1-score, average execution time, average energy consumption, and estimated $CO_2$ emission. The findings indicate that although CNN-based models can outperform various multimodal techniques that incorporate both images and contextual information, applying additional filtering on top of LLMs can lead to substantial performance gains. These findings highlight the transformative potential of multimodal AI systems to enhance the reliability, efficiency, and scalability of medical diagnostics in clinical settings.
>
---
#### [replaced 034] HunyuanVideo-Avatar: High-Fidelity Audio-Driven Human Animation for Multiple Characters
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20156v2](http://arxiv.org/pdf/2505.20156v2)**

> **作者:** Yi Chen; Sen Liang; Zixiang Zhou; Ziyao Huang; Yifeng Ma; Junshu Tang; Qin Lin; Yuan Zhou; Qinglin Lu
>
> **摘要:** Recent years have witnessed significant progress in audio-driven human animation. However, critical challenges remain in (i) generating highly dynamic videos while preserving character consistency, (ii) achieving precise emotion alignment between characters and audio, and (iii) enabling multi-character audio-driven animation. To address these challenges, we propose HunyuanVideo-Avatar, a multimodal diffusion transformer (MM-DiT)-based model capable of simultaneously generating dynamic, emotion-controllable, and multi-character dialogue videos. Concretely, HunyuanVideo-Avatar introduces three key innovations: (i) A character image injection module is designed to replace the conventional addition-based character conditioning scheme, eliminating the inherent condition mismatch between training and inference. This ensures the dynamic motion and strong character consistency; (ii) An Audio Emotion Module (AEM) is introduced to extract and transfer the emotional cues from an emotion reference image to the target generated video, enabling fine-grained and accurate emotion style control; (iii) A Face-Aware Audio Adapter (FAA) is proposed to isolate the audio-driven character with latent-level face mask, enabling independent audio injection via cross-attention for multi-character scenarios. These innovations empower HunyuanVideo-Avatar to surpass state-of-the-art methods on benchmark datasets and a newly proposed wild dataset, generating realistic avatars in dynamic, immersive scenarios.
>
---
#### [replaced 035] ViFOR: A Fourier-Enhanced Vision Transformer for Multi-Image Super-Resolution in Earth System
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.12427v3](http://arxiv.org/pdf/2502.12427v3)**

> **作者:** Ehsan Zeraatkar; Salah A Faroughi; Jelena Tešić
>
> **摘要:** Super-resolution (SR) is crucial for enhancing the spatial resolution of Earth System Model (ESM) data, thereby enabling more precise analysis of environmental processes. This paper introduces ViFOR, a novel SR algorithm integrating Vision Transformers (ViTs) with Fourier-based Implicit Neural Representation Networks (INRs). ViFOR effectively captures global context and high-frequency details essential for accurate SR reconstruction by embedding Fourier-based activation functions within the transformer architecture. Extensive experiments demonstrate that ViFOR consistently outperforms state-of-the-art methods, including ViT, SIREN, and SRGANs, in terms of Peak Signal-to-Noise Ratio (PSNR) and Mean Squared Error (MSE) for both global and local imagery. ViFOR achieves PSNR improvements of up to 4.18 dB, 1.56 dB, and 1.73 dB over ViT on full-image Source Temperature, Shortwave, and Longwave Flux datasets. These results highlight ViFOR's effectiveness and potential for advancing high-resolution climate data analysis.
>
---
#### [replaced 036] Visual-TCAV: Concept-based Attribution and Saliency Maps for Post-hoc Explainability in Image Classification
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.05698v2](http://arxiv.org/pdf/2411.05698v2)**

> **作者:** Antonio De Santis; Riccardo Campi; Matteo Bianchi; Marco Brambilla
>
> **备注:** Preprint currently under review
>
> **摘要:** Convolutional Neural Networks (CNNs) have seen significant performance improvements in recent years. However, due to their size and complexity, they function as black-boxes, leading to transparency concerns. State-of-the-art saliency methods generate local explanations that highlight the area in the input image where a class is identified but cannot explain how a concept of interest contributes to the prediction, which is essential for bias mitigation. On the other hand, concept-based methods, such as TCAV (Testing with Concept Activation Vectors), provide insights into how sensitive is the network to a concept, but cannot compute its attribution in a specific prediction nor show its location within the input image. This paper introduces a novel post-hoc explainability framework, Visual-TCAV, which aims to bridge the gap between these methods by providing both local and global explanations for CNN-based image classification. Visual-TCAV uses Concept Activation Vectors (CAVs) to generate saliency maps that show where concepts are recognized by the network. Moreover, it can estimate the attribution of these concepts to the output of any class using a generalization of Integrated Gradients. This framework is evaluated on popular CNN architectures, with its validity further confirmed via experiments where ground truth for explanations is known, and a comparison with TCAV. Our code is available at https://github.com/DataSciencePolimi/Visual-TCAV.
>
---
#### [replaced 037] A Comparative Study of Scanpath Models in Graph-Based Visualization
- **分类: cs.HC; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.24160v3](http://arxiv.org/pdf/2503.24160v3)**

> **作者:** Angela Lopez-Cardona; Parvin Emami; Sebastian Idesis; Saravanakumar Duraisamy; Luis A. Leiva; Ioannis Arapakis
>
> **摘要:** Information Visualization (InfoVis) systems utilize visual representations to enhance data interpretation. Understanding how visual attention is allocated is essential for optimizing interface design. However, collecting Eye-tracking (ET) data presents challenges related to cost, privacy, and scalability. Computational models provide alternatives for predicting gaze patterns, thereby advancing InfoVis research. In our study, we conducted an ET experiment with 40 participants who analyzed graphs while responding to questions of varying complexity within the context of digital forensics. We compared human scanpaths with synthetic ones generated by models such as DeepGaze, UMSS, and Gazeformer. Our research evaluates the accuracy of these models and examines how question complexity and number of nodes influence performance. This work contributes to the development of predictive modeling in visual analytics, offering insights that can enhance the design and effectiveness of InfoVis systems.
>
---
#### [replaced 038] Efficient RAW Image Deblurring with Adaptive Frequency Modulation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.24407v2](http://arxiv.org/pdf/2505.24407v2)**

> **作者:** Wenlong Jiao; Binglong Li; Wei Shang; Ping Wang; Dongwei Ren
>
> **备注:** The code will be available at https://github.com/WenlongJiao/FrENet
>
> **摘要:** Image deblurring plays a crucial role in enhancing visual clarity across various applications. Although most deep learning approaches primarily focus on sRGB images, which inherently lose critical information during the image signal processing pipeline, RAW images, being unprocessed and linear, possess superior restoration potential but remain underexplored. Deblurring RAW images presents unique challenges, particularly in handling frequency-dependent blur while maintaining computational efficiency. To address these issues, we propose Frequency Enhanced Network (FrENet), a framework specifically designed for RAW-to-RAW deblurring that operates directly in the frequency domain. We introduce a novel Adaptive Frequency Positional Modulation module, which dynamically adjusts frequency components according to their spectral positions, thereby enabling precise control over the deblurring process. Additionally, frequency domain skip connections are adopted to further preserve high-frequency details. Experimental results demonstrate that FrENet surpasses state-of-the-art deblurring methods in RAW image deblurring, achieving significantly better restoration quality while maintaining high efficiency in terms of reduced MACs. Furthermore, FrENet's adaptability enables it to be extended to sRGB images, where it delivers comparable or superior performance compared to methods specifically designed for sRGB data. The code will be available at https://github.com/WenlongJiao/FrENet .
>
---
#### [replaced 039] ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents
- **分类: cs.CV; cs.AI; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2502.18017v2](http://arxiv.org/pdf/2502.18017v2)**

> **作者:** Qiuchen Wang; Ruixue Ding; Zehui Chen; Weiqi Wu; Shihang Wang; Pengjun Xie; Feng Zhao
>
> **摘要:** Understanding information from visually rich documents remains a significant challenge for traditional Retrieval-Augmented Generation (RAG) methods. Existing benchmarks predominantly focus on image-based question answering (QA), overlooking the fundamental challenges of efficient retrieval, comprehension, and reasoning within dense visual documents. To bridge this gap, we introduce ViDoSeek, a novel dataset designed to evaluate RAG performance on visually rich documents requiring complex reasoning. Based on it, we identify key limitations in current RAG approaches: (i) purely visual retrieval methods struggle to effectively integrate both textual and visual features, and (ii) previous approaches often allocate insufficient reasoning tokens, limiting their effectiveness. To address these challenges, we propose ViDoRAG, a novel multi-agent RAG framework tailored for complex reasoning across visual documents. ViDoRAG employs a Gaussian Mixture Model (GMM)-based hybrid strategy to effectively handle multi-modal retrieval. To further elicit the model's reasoning capabilities, we introduce an iterative agent workflow incorporating exploration, summarization, and reflection, providing a framework for investigating test-time scaling in RAG domains. Extensive experiments on ViDoSeek validate the effectiveness and generalization of our approach. Notably, ViDoRAG outperforms existing methods by over 10% on the competitive ViDoSeek benchmark. The code is available at https://github.com/Alibaba-NLP/ViDoRAG.
>
---
#### [replaced 040] T-FAKE: Synthesizing Thermal Images for Facial Landmarking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.15127v3](http://arxiv.org/pdf/2408.15127v3)**

> **作者:** Philipp Flotho; Moritz Piening; Anna Kukleva; Gabriele Steidl
>
> **备注:** 22 pages, 12 figures, Philipp Flotho and Moritz Piening share equal contribution
>
> **摘要:** Facial analysis is a key component in a wide range of applications such as healthcare, autonomous driving, and entertainment. Despite the availability of various facial RGB datasets, the thermal modality, which plays a crucial role in life sciences, medicine, and biometrics, has been largely overlooked. To address this gap, we introduce the T-FAKE dataset, a new large-scale synthetic thermal dataset with sparse and dense landmarks. To facilitate the creation of the dataset, we propose a novel RGB2Thermal loss function, which enables the domain-adaptive transfer of RGB faces to thermal style. By utilizing the Wasserstein distance between thermal and RGB patches and the statistical analysis of clinical temperature distributions on faces, we ensure that the generated thermal images closely resemble real samples. Using RGB2Thermal style transfer based on our RGB2Thermal loss function, we create the large-scale synthetic thermal T-FAKE dataset with landmark and segmentation annotations. Leveraging our novel T-FAKE dataset, probabilistic landmark prediction, and label adaptation networks, we demonstrate significant improvements in landmark detection methods on thermal images across different landmark conventions. Our models show excellent performance with both sparse 70-point landmarks and dense 478-point landmark annotations. Moreover, our RGB2Thermal loss leads to notable results in terms of perceptual evaluation and temperature prediction.
>
---
#### [replaced 041] Rethinking Diffusion Posterior Sampling: From Conditional Score Estimator to Maximizing a Posterior
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.18913v2](http://arxiv.org/pdf/2501.18913v2)**

> **作者:** Tongda Xu; Xiyan Cai; Xinjie Zhang; Xingtong Ge; Dailan He; Ming Sun; Jingjing Liu; Ya-Qin Zhang; Jian Li; Yan Wang
>
> **备注:** ICLR 2025
>
> **摘要:** Recent advancements in diffusion models have been leveraged to address inverse problems without additional training, and Diffusion Posterior Sampling (DPS) (Chung et al., 2022a) is among the most popular approaches. Previous analyses suggest that DPS accomplishes posterior sampling by approximating the conditional score. While in this paper, we demonstrate that the conditional score approximation employed by DPS is not as effective as previously assumed, but rather aligns more closely with the principle of maximizing a posterior (MAP). This assertion is substantiated through an examination of DPS on 512x512 ImageNet images, revealing that: 1) DPS's conditional score estimation significantly diverges from the score of a well-trained conditional diffusion model and is even inferior to the unconditional score; 2) The mean of DPS's conditional score estimation deviates significantly from zero, rendering it an invalid score estimation; 3) DPS generates high-quality samples with significantly lower diversity. In light of the above findings, we posit that DPS more closely resembles MAP than a conditional score estimator, and accordingly propose the following enhancements to DPS: 1) we explicitly maximize the posterior through multi-step gradient ascent and projection; 2) we utilize a light-weighted conditional score estimator trained with only 100 images and 8 GPU hours. Extensive experimental results indicate that these proposed improvements significantly enhance DPS's performance. The source code for these improvements is provided in https://github.com/tongdaxu/Rethinking-Diffusion-Posterior-Sampling-From-Conditional-Score-Estimator-to-Maximizing-a-Posterior.
>
---
#### [replaced 042] Diving into Self-Evolving Training for Multimodal Reasoning
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.17451v2](http://arxiv.org/pdf/2412.17451v2)**

> **作者:** Wei Liu; Junlong Li; Xiwen Zhang; Fan Zhou; Yu Cheng; Junxian He
>
> **备注:** ICML 2025, Project Page: https://mstar-lmm.github.io
>
> **摘要:** Self-evolving trainin--where models iteratively learn from their own outputs--has emerged as a key approach for complex reasoning tasks, addressing the scarcity of high-quality chain-of-thought data. However, its effectiveness in multimodal reasoning, a domain more intricate than text-only reasoning, remains underexplored, and the understanding of critical factors in this training paradigm remains limited. Furthermore, a central challenge for this training method is performance saturation, which impedes further improvements and scalability. Inspired by reinforcement learning (RL), in this paper, we reframe self-evolving training for multimodal reasoning through the lens of RL, identifying three pivotal factors: Training Method, Reward Model, and Prompt Variation. Through systematic analysis, we establish relatively optimal design principles that significantly enhance multimodal reasoning capabilities. Moreover, delving deeper into training dynamics, we uncover the roots of saturation and propose a new automatic balancing mechanism to mitigate this limitation. Building on these insights, we propose M-STAR (Multimodal Self-evolving Training for Reasoning), a framework that achieves consistent performance gains across models of varying sizes and diverse benchmarks. All resources are made publicly available at https://mstar-lmm.github.io.
>
---
#### [replaced 043] 3D Trajectory Reconstruction of Moving Points Based on Asynchronous Cameras
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.00541v2](http://arxiv.org/pdf/2506.00541v2)**

> **作者:** Huayu Huang; Banglei Guan; Yang Shang; Qifeng Yu
>
> **备注:** This paper has been accepted by Acta Mechanica Sinica
>
> **摘要:** Photomechanics is a crucial branch of solid mechanics. The localization of point targets constitutes a fundamental problem in optical experimental mechanics, with extensive applications in various missions of UAVs. Localizing moving targets is crucial for analyzing their motion characteristics and dynamic properties. Reconstructing the trajectories of points from asynchronous cameras is a significant challenge. It encompasses two coupled sub-problems: trajectory reconstruction and camera synchronization. Present methods typically address only one of these sub-problems individually. This paper proposes a 3D trajectory reconstruction method for point targets based on asynchronous cameras, simultaneously solving both sub-problems. Firstly, we extend the trajectory intersection method to asynchronous cameras to resolve the limitation of traditional triangulation that requires camera synchronization. Secondly, we develop models for camera temporal information and target motion, based on imaging mechanisms and target dynamics characteristics. The parameters are optimized simultaneously to achieve trajectory reconstruction without accurate time parameters. Thirdly, we optimize the camera rotations alongside the camera time information and target motion parameters, using tighter and more continuous constraints on moving points. The reconstruction accuracy is significantly improved, especially when the camera rotations are inaccurate. Finally, the simulated and real-world experimental results demonstrate the feasibility and accuracy of the proposed method. The real-world results indicate that the proposed algorithm achieved a localization error of 112.95 m at an observation range of 15 ~ 20 km.
>
---
#### [replaced 044] VR-Robo: A Real-to-Sim-to-Real Framework for Visual Robot Navigation and Locomotion
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.01536v3](http://arxiv.org/pdf/2502.01536v3)**

> **作者:** Shaoting Zhu; Linzhan Mou; Derun Li; Baijun Ye; Runhan Huang; Hang Zhao
>
> **备注:** Project Page: https://vr-robo.github.io/
>
> **摘要:** Recent success in legged robot locomotion is attributed to the integration of reinforcement learning and physical simulators. However, these policies often encounter challenges when deployed in real-world environments due to sim-to-real gaps, as simulators typically fail to replicate visual realism and complex real-world geometry. Moreover, the lack of realistic visual rendering limits the ability of these policies to support high-level tasks requiring RGB-based perception like ego-centric navigation. This paper presents a Real-to-Sim-to-Real framework that generates photorealistic and physically interactive "digital twin" simulation environments for visual navigation and locomotion learning. Our approach leverages 3D Gaussian Splatting (3DGS) based scene reconstruction from multi-view images and integrates these environments into simulations that support ego-centric visual perception and mesh-based physical interactions. To demonstrate its effectiveness, we train a reinforcement learning policy within the simulator to perform a visual goal-tracking task. Extensive experiments show that our framework achieves RGB-only sim-to-real policy transfer. Additionally, our framework facilitates the rapid adaptation of robot policies with effective exploration capability in complex new environments, highlighting its potential for applications in households and factories.
>
---
#### [replaced 045] CMRINet: Joint Groupwise Registration and Segmentation for Cardiac Function Quantification from Cine-MRI
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.16452v2](http://arxiv.org/pdf/2505.16452v2)**

> **作者:** Mohamed S. Elmahdy; Marius Staring; Patrick J. H. de Koning; Samer Alabed; Mahan Salehi; Faisal Alandejani; Michael Sharkey; Ziad Aldabbagh; Andrew J. Swift; Rob J. van der Geest
>
> **备注:** 15 pages, 7 figures, 1 appendix
>
> **摘要:** Accurate and efficient quantification of cardiac function is essential for the estimation of prognosis of cardiovascular diseases (CVDs). One of the most commonly used metrics for evaluating cardiac pumping performance is left ventricular ejection fraction (LVEF). However, LVEF can be affected by factors such as inter-observer variability and varying pre-load and after-load conditions, which can reduce its reproducibility. Additionally, cardiac dysfunction may not always manifest as alterations in LVEF, such as in heart failure and cardiotoxicity diseases. An alternative measure that can provide a relatively load-independent quantitative assessment of myocardial contractility is myocardial strain and strain rate. By using LVEF in combination with myocardial strain, it is possible to obtain a thorough description of cardiac function. Automated estimation of LVEF and other volumetric measures from cine-MRI sequences can be achieved through segmentation models, while strain calculation requires the estimation of tissue displacement between sequential frames, which can be accomplished using registration models. These tasks are often performed separately, potentially limiting the assessment of cardiac function. To address this issue, in this study we propose an end-to-end deep learning (DL) model that jointly estimates groupwise (GW) registration and segmentation for cardiac cine-MRI images. The proposed anatomically-guided Deep GW network was trained and validated on a large dataset of 4-chamber view cine-MRI image series of 374 subjects. A quantitative comparison with conventional GW registration using elastix and two DL-based methods showed that the proposed model improved performance and substantially reduced computation time.
>
---
#### [replaced 046] Enhancing Target-unspecific Tasks through a Features Matrix
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.03414v5](http://arxiv.org/pdf/2505.03414v5)**

> **作者:** Fangming Cui; Yonggang Zhang; Xuan Wang; Xinmei Tian; Jun Yu
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Recent developments in prompt learning of large Vision-Language Models (VLMs) have significantly improved performance in target-specific tasks. However, these prompting methods often struggle to tackle the target-unspecific or generalizable tasks effectively. It may be attributed to the fact that overfitting training causes the model to forget its general knowledge. The general knowledge has a strong promotion on target-unspecific tasks. To alleviate this issue, we propose a novel Features Matrix (FM) approach designed to enhance these models on target-unspecific tasks. Our method extracts and leverages general knowledge, shaping a Features Matrix (FM). Specifically, the FM captures the semantics of diverse inputs from a deep and fine perspective, preserving essential general knowledge, which mitigates the risk of overfitting. Representative evaluations demonstrate that: 1) the FM is compatible with existing frameworks as a generic and flexible module, and 2) the FM significantly showcases its effectiveness in enhancing target-unspecific tasks (base-to-novel generalization, domain generalization, and cross-dataset generalization), achieving state-of-the-art performance.
>
---
#### [replaced 047] Effective Dual-Region Augmentation for Reduced Reliance on Large Amounts of Labeled Data
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.13077v2](http://arxiv.org/pdf/2504.13077v2)**

> **作者:** Prasanna Reddy Pulakurthi; Majid Rabbani; Celso M. de Melo; Sohail A. Dianat; Raghuveer M. Rao
>
> **备注:** 9 pages, 2 figures, 4 tables, Accepted to SPIE DSC 2025 Conference: Synthetic Data for Artificial Intelligence and Machine Learning: Tools, Techniques, and Applications III
>
> **摘要:** This paper introduces a novel dual-region augmentation approach designed to reduce reliance on large-scale labeled datasets while improving model robustness and adaptability across diverse computer vision tasks, including source-free domain adaptation (SFDA) and person re-identification (ReID). Our method performs targeted data transformations by applying random noise perturbations to foreground objects and spatially shuffling background patches. This effectively increases the diversity of the training data, improving model robustness and generalization. Evaluations on the PACS dataset for SFDA demonstrate that our augmentation strategy consistently outperforms existing methods, achieving significant accuracy improvements in both single-target and multi-target adaptation settings. By augmenting training data through structured transformations, our method enables model generalization across domains, providing a scalable solution for reducing reliance on manually annotated datasets. Furthermore, experiments on Market-1501 and DukeMTMC-reID datasets validate the effectiveness of our approach for person ReID, surpassing traditional augmentation techniques. The code is available at https://github.com/PrasannaPulakurthi/Foreground-Background-Augmentation
>
---
#### [replaced 048] InfoChartQA: A Benchmark for Multimodal Question Answering on Infographic Charts
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.19028v3](http://arxiv.org/pdf/2505.19028v3)**

> **作者:** Minzhi Lin; Tianchi Xie; Mengchen Liu; Yilin Ye; Changjian Chen; Shixia Liu
>
> **摘要:** Understanding infographic charts with design-driven visual elements (e.g., pictograms, icons) requires both visual recognition and reasoning, posing challenges for multimodal large language models (MLLMs). However, existing visual-question answering benchmarks fall short in evaluating these capabilities of MLLMs due to the lack of paired plain charts and visual-element-based questions. To bridge this gap, we introduce InfoChartQA, a benchmark for evaluating MLLMs on infographic chart understanding. It includes 5,642 pairs of infographic and plain charts, each sharing the same underlying data but differing in visual presentations. We further design visual-element-based questions to capture their unique visual designs and communicative intent. Evaluation of 20 MLLMs reveals a substantial performance decline on infographic charts, particularly for visual-element-based questions related to metaphors. The paired infographic and plain charts enable fine-grained error analysis and ablation studies, which highlight new opportunities for advancing MLLMs in infographic chart understanding. We release InfoChartQA at https://github.com/CoolDawnAnt/InfoChartQA.
>
---
#### [replaced 049] Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.02416v2](http://arxiv.org/pdf/2410.02416v2)**

> **作者:** Seyedmorteza Sadat; Otmar Hilliges; Romann M. Weber
>
> **备注:** Published as a conference paper at ICLR 2025
>
> **摘要:** Classifier-free guidance (CFG) is crucial for improving both generation quality and alignment between the input condition and final output in diffusion models. While a high guidance scale is generally required to enhance these aspects, it also causes oversaturation and unrealistic artifacts. In this paper, we revisit the CFG update rule and introduce modifications to address this issue. We first decompose the update term in CFG into parallel and orthogonal components with respect to the conditional model prediction and observe that the parallel component primarily causes oversaturation, while the orthogonal component enhances image quality. Accordingly, we propose down-weighting the parallel component to achieve high-quality generations without oversaturation. Additionally, we draw a connection between CFG and gradient ascent and introduce a new rescaling and momentum method for the CFG update rule based on this insight. Our approach, termed adaptive projected guidance (APG), retains the quality-boosting advantages of CFG while enabling the use of higher guidance scales without oversaturation. APG is easy to implement and introduces practically no additional computational overhead to the sampling process. Through extensive experiments, we demonstrate that APG is compatible with various conditional diffusion models and samplers, leading to improved FID, recall, and saturation scores while maintaining precision comparable to CFG, making our method a superior plug-and-play alternative to standard classifier-free guidance.
>
---
#### [replaced 050] A minimalistic representation model for head direction system
- **分类: q-bio.NC; cs.AI; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2411.10596v2](http://arxiv.org/pdf/2411.10596v2)**

> **作者:** Minglu Zhao; Dehong Xu; Deqian Kong; Wen-Hao Zhang; Ying Nian Wu
>
> **备注:** Proceedings of the Annual Meeting of the Cognitive Science Society (CogSci 2025)
>
> **摘要:** We present a minimalistic representation model for the head direction (HD) system, aiming to learn a high-dimensional representation of head direction that captures essential properties of HD cells. Our model is a representation of rotation group $U(1)$, and we study both the fully connected version and convolutional version. We demonstrate the emergence of Gaussian-like tuning profiles and a 2D circle geometry in both versions of the model. We also demonstrate that the learned model is capable of accurate path integration.
>
---
#### [replaced 051] The Male CEO and the Female Assistant: Evaluation and Mitigation of Gender Biases in Text-To-Image Generation of Dual Subjects
- **分类: cs.CV; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2402.11089v4](http://arxiv.org/pdf/2402.11089v4)**

> **作者:** Yixin Wan; Kai-Wei Chang
>
> **摘要:** Recent large-scale T2I models like DALLE-3 have made progress in reducing gender stereotypes when generating single-person images. However, significant biases remain when generating images with more than one person. To systematically evaluate this, we propose the Paired Stereotype Test (PST) framework, which queries T2I models to depict two individuals assigned with male-stereotyped and female-stereotyped social identities, respectively (e.g. "a CEO" and "an Assistant"). This contrastive setting often triggers T2I models to generate gender-stereotyped images. Using PST, we evaluate two aspects of gender biases -- the well-known bias in gendered occupation and a novel aspect: bias in organizational power. Experiments show that over 74\% images generated by DALLE-3 display gender-occupational biases. Additionally, compared to single-person settings, DALLE-3 is more likely to perpetuate male-associated stereotypes under PST. We further propose FairCritic, a novel and interpretable framework that leverages an LLM-based critic model to i) detect bias in generated images, and ii) adaptively provide feedback to T2I models for improving fairness. FairCritic achieves near-perfect fairness on PST, overcoming the limitations of previous prompt-based intervention approaches.
>
---
#### [replaced 052] Evaluating and Advancing Multimodal Large Language Models in Perception Ability Lens
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.14725v2](http://arxiv.org/pdf/2411.14725v2)**

> **作者:** Feng Chen; Chenhui Gou; Jing Liu; Yang Yang; Zhaoyang Li; Jiyuan Zhang; Zhenbang Sun; Bohan Zhuang; Qi Wu
>
> **备注:** Code repository: https://github.com/Chenfeng1271/AbilityLens/tree/main
>
> **摘要:** As multimodal large language models (MLLMs) advance rapidly, rigorous evaluation has become essential, providing further guidance for their development. In this work, we focus on a unified and robust evaluation of \textbf{vision perception} abilities, the foundational skill of MLLMs. We find that existing perception benchmarks, each focusing on different question types, domains, and evaluation metrics, introduce significant evaluation variance, complicating comprehensive assessments of perception abilities when relying on any single benchmark. To address this, we introduce \textbf{AbilityLens}, a unified benchmark designed to evaluate MLLMs in six key perception abilities (ranging from counting, OCR, to understanding structural data), focusing on both accuracy and stability, with each ability encompassing diverse types of questions, domains, and metrics. With the assistance of AbilityLens, we: (1) identify the strengths and weaknesses of current main-stream MLLMs, highlighting stability patterns and revealing a notable performance gap between state-of-the-art open-source and closed-source models; (2) uncover interesting ability conflict and early convergence phenomena during MLLM training; (3) reveal the primary reason of ability conflict is data mixing ratio and LLM model size; and (4) discuss the effectiveness of some straightforward strategies \eg, fine-tuning and model merging, to solve the ability conflict. The benchmark and online leaderboard is released in https://github.com/Chenfeng1271/AbilityLens.
>
---
#### [replaced 053] GASP: Gaussian Splatting for Physic-Based Simulations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.05819v2](http://arxiv.org/pdf/2409.05819v2)**

> **作者:** Piotr Borycki; Weronika Smolak; Joanna Waczyńska; Marcin Mazur; Sławomir Tadeja; Przemysław Spurek
>
> **摘要:** Physics simulation is paramount for modeling and utilization of 3D scenes in various real-world applications. However, its integration with state-of-the-art 3D scene rendering techniques such as Gaussian Splatting (GS) remains challenging. Existing models use additional meshing mechanisms, including triangle or tetrahedron meshing, marching cubes, or cage meshes. As an alternative, we can modify the physics grounded Newtonian dynamics to align with 3D Gaussian components. Current models take the first-order approximation of a deformation map, which locally approximates the dynamics by linear transformations. In contrast, our Gaussian Splatting for Physics-Based Simulations (GASP) model uses such a map (without any modifications) and flat Gaussian distributions, which are parameterized by three points (mesh faces). Subsequently, each 3D point (mesh face node) is treated as a discrete entity within a 3D space. Consequently, the problem of modeling Gaussian components is reduced to working with 3D points. Additionally, the information on mesh faces can be used to incorporate further properties into the physics model, facilitating the use of triangles. Resulting solution can be integrated into any physics engine that can be treated as a black box. As demonstrated in our studies, the proposed model exhibits superior performance on a diverse range of benchmark datasets designed for 3D object rendering.
>
---
#### [replaced 054] Mobile-Agent-V: A Video-Guided Approach for Effortless and Efficient Operational Knowledge Injection in Mobile Automation
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.17110v3](http://arxiv.org/pdf/2502.17110v3)**

> **作者:** Junyang Wang; Haiyang Xu; Xi Zhang; Ming Yan; Ji Zhang; Fei Huang; Jitao Sang
>
> **备注:** 17 pages, 7 figures, 9 tables
>
> **摘要:** The exponential rise in mobile device usage necessitates streamlined automation for effective task management, yet many AI frameworks fall short due to inadequate operational expertise. While manually written knowledge can bridge this gap, it is often burdensome and inefficient. We introduce Mobile-Agent-V, an innovative framework that utilizes video as a guiding tool to effortlessly and efficiently inject operational knowledge into mobile automation processes. By deriving knowledge directly from video content, Mobile-Agent-V eliminates manual intervention, significantly reducing the effort and time required for knowledge acquisition. To rigorously evaluate this approach, we propose Mobile-Knowledge, a benchmark tailored to assess the impact of external knowledge on mobile agent performance. Our experimental findings demonstrate that Mobile-Agent-V enhances performance by 36% compared to existing methods, underscoring its effortless and efficient advantages in mobile automation.
>
---
#### [replaced 055] Self-supervised Learning of Event-guided Video Frame Interpolation for Rolling Shutter Frames
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2306.15507v2](http://arxiv.org/pdf/2306.15507v2)**

> **作者:** Yunfan Lu; Guoqiang Liang; Yiran Shen; Lin Wang
>
> **备注:** An earlier version of this paper (ID: 1845) was submitted to ICCV 2023 in March 2023. The work has been substantially revised and accepted by IEEE Transactions on Visualization and Computer Graphics (TVCG)
>
> **摘要:** Most consumer cameras use rolling shutter (RS) exposure, which often leads to distortions such as skew and jelly effects. These videos are further limited by bandwidth and frame rate constraints. In this paper, we explore the potential of event cameras, which offer high temporal resolution. We propose a framework to recover global shutter (GS) high-frame-rate videos without RS distortion by combining an RS camera and an event camera. Due to the lack of real-world datasets, our framework adopts a self-supervised strategy based on a displacement field, a dense 3D spatiotemporal representation of pixel motion during exposure. This enables mutual reconstruction between RS and GS frames and facilitates slow-motion recovery. We combine RS frames with the displacement field to generate GS frames, and integrate inverse mapping and RS frame warping for self-supervision. Experiments on four datasets show that our method removes distortion, reduces bandwidth usage by 94 percent, and achieves 16 ms per frame at 32x interpolation.
>
---
#### [replaced 056] PixelCAM: Pixel Class Activation Mapping for Histology Image Classification and ROI Localization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.24135v2](http://arxiv.org/pdf/2503.24135v2)**

> **作者:** Alexis Guichemerre; Soufiane Belharbi; Mohammadhadi Shateri; Luke McCaffrey; Eric Granger
>
> **备注:** 43 pages, 24 figures, Medical Imaging with Deep Learning (MIDL 2025)
>
> **摘要:** Weakly supervised object localization (WSOL) methods allow training models to classify images and localize ROIs. WSOL only requires low-cost image-class annotations yet provides a visually interpretable classifier. Standard WSOL methods rely on class activation mapping (CAM) methods to produce spatial localization maps according to a single- or two-step strategy. While both strategies have made significant progress, they still face several limitations with histology images. Single-step methods can easily result in under- or over-activation due to the limited visual ROI saliency in histology images and scarce localization cues. They also face the well-known issue of asynchronous convergence between classification and localization tasks. The two-step approach is sub-optimal because it is constrained to a frozen classifier, limiting the capacity for localization. Moreover, these methods also struggle when applied to out-of-distribution (OOD) datasets. In this paper, a multi-task approach for WSOL is introduced for simultaneous training of both tasks to address the asynchronous convergence problem. In particular, localization is performed in the pixel-feature space of an image encoder that is shared with classification. This allows learning discriminant features and accurate delineation of foreground/background regions to support ROI localization and image classification. We propose PixelCAM, a cost-effective foreground/background pixel-wise classifier in the pixel-feature space that allows for spatial object localization. Using partial-cross entropy, PixelCAM is trained using pixel pseudo-labels collected from a pretrained WSOL model. Both image and pixel-wise classifiers are trained simultaneously using standard gradient descent. In addition, our pixel classifier can easily be integrated into CNN- and transformer-based architectures without any modifications.
>
---
#### [replaced 057] Scalable Multi-Robot Informative Path Planning for Target Mapping via Deep Reinforcement Learning
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.16967v3](http://arxiv.org/pdf/2409.16967v3)**

> **作者:** Apoorva Vashisth; Manav Kulshrestha; Damon Conover; Aniket Bera
>
> **摘要:** Autonomous robots are widely utilized for mapping and exploration tasks due to their cost-effectiveness. Multi-robot systems offer scalability and efficiency, especially in terms of the number of robots deployed in more complex environments. These tasks belong to the set of Multi-Robot Informative Path Planning (MRIPP) problems. In this paper, we propose a deep reinforcement learning approach for the MRIPP problem. We aim to maximize the number of discovered stationary targets in an unknown 3D environment while operating under resource constraints (such as path length). Here, each robot aims to maximize discovered targets, avoid unknown static obstacles, and prevent inter-robot collisions while operating under communication and resource constraints. We utilize the centralized training and decentralized execution paradigm to train a single policy neural network. A key aspect of our approach is our coordination graph that prioritizes visiting regions not yet explored by other robots. Our learned policy can be copied onto any number of robots for deployment in more complex environments not seen during training. Our approach outperforms state-of-the-art approaches by at least 26.2% in terms of the number of discovered targets while requiring a planning time of less than 2 sec per step. We present results for more complex environments with up to 64 robots and compare success rates against baseline planners. Our code and trained model are available at - https://github.com/AccGen99/marl_ipp
>
---
#### [replaced 058] SceneSplat: Gaussian Splatting-based Scene Understanding with Vision-Language Pretraining
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18052v2](http://arxiv.org/pdf/2503.18052v2)**

> **作者:** Yue Li; Qi Ma; Runyi Yang; Huapeng Li; Mengjiao Ma; Bin Ren; Nikola Popovic; Nicu Sebe; Ender Konukoglu; Theo Gevers; Luc Van Gool; Martin R. Oswald; Danda Pani Paudel
>
> **备注:** Our code, model, and dataset will be released at https://unique1i.github.io/SceneSplat_webpage/
>
> **摘要:** Recognizing arbitrary or previously unseen categories is essential for comprehensive real-world 3D scene understanding. Currently, all existing methods rely on 2D or textual modalities during training or together at inference. This highlights the clear absence of a model capable of processing 3D data alone for learning semantics end-to-end, along with the necessary data to train such a model. Meanwhile, 3D Gaussian Splatting (3DGS) has emerged as the de facto standard for 3D scene representation across various vision tasks. However, effectively integrating semantic reasoning into 3DGS in a generalizable manner remains an open challenge. To address these limitations, we introduce SceneSplat, to our knowledge the first large-scale 3D indoor scene understanding approach that operates natively on 3DGS. Furthermore, we propose a self-supervised learning scheme that unlocks rich 3D feature learning from unlabeled scenes. To power the proposed methods, we introduce SceneSplat-7K, the first large-scale 3DGS dataset for indoor scenes, comprising 7916 scenes derived from seven established datasets, such as ScanNet and Matterport3D. Generating SceneSplat-7K required computational resources equivalent to 150 GPU days on an L4 GPU, enabling standardized benchmarking for 3DGS-based reasoning for indoor scenes. Our exhaustive experiments on SceneSplat-7K demonstrate the significant benefit of the proposed method over the established baselines.
>
---
#### [replaced 059] SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.04417v4](http://arxiv.org/pdf/2410.04417v4)**

> **作者:** Yuan Zhang; Chun-Kai Fan; Junpeng Ma; Wenzhao Zheng; Tao Huang; Kuan Cheng; Denis Gudovskiy; Tomoyuki Okuno; Yohei Nakata; Kurt Keutzer; Shanghang Zhang
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** In vision-language models (VLMs), visual tokens usually bear a significant amount of computational overhead despite sparsity of information in them when compared to text tokens. To address this, most existing methods learn a network to prune redundant visual tokens using certain training data. Differently, we propose a text-guided training-free token optimization mechanism dubbed SparseVLM that eliminates the need of extra parameters or fine-tuning costs. Given that visual tokens complement text tokens in VLM's linguistic reasoning, we select relevant text tokens to rate the significance of visual tokens using self-attention matrices and, then, prune visual tokens using the proposed strategy to maximize sparsity while retaining information. In particular, we introduce a rank-based strategy to adaptively determine the sparsification ratio for each layer, alongside a token recycling method that compresses pruned tokens into more compact representations. Experimental results show that SparseVLM increases the efficiency of various VLMs in a number of image and video understanding tasks. For example, LLaVA when equipped with SparseVLM achieves 54% reduction in FLOPs, 37% decrease in CUDA latency while maintaining 97% of its original accuracy. Our code is available at https://github.com/Gumpest/SparseVLMs.
>
---
#### [replaced 060] LLM-Guided Taxonomy and Hierarchical Uncertainty for 3D Point Cloud Active Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18924v2](http://arxiv.org/pdf/2505.18924v2)**

> **作者:** Chenxi Li; Nuo Chen; Fengyun Tan; Yantong Chen; Bochun Yuan; Tianrui Li; Chongshou Li
>
> **摘要:** We present a novel active learning framework for 3D point cloud semantic segmentation that, for the first time, integrates large language models (LLMs) to construct hierarchical label structures and guide uncertainty-based sample selection. Unlike prior methods that treat labels as flat and independent, our approach leverages LLM prompting to automatically generate multi-level semantic taxonomies and introduces a recursive uncertainty projection mechanism that propagates uncertainty across hierarchy levels. This enables spatially diverse, label-aware point selection that respects the inherent semantic structure of 3D scenes. Experiments on S3DIS and ScanNet v2 show that our method achieves up to 4% mIoU improvement under extremely low annotation budgets (e.g., 0.02%), substantially outperforming existing baselines. Our results highlight the untapped potential of LLMs as knowledge priors in 3D vision and establish hierarchical uncertainty modeling as a powerful paradigm for efficient point cloud annotation.
>
---
#### [replaced 061] OralBBNet: Spatially Guided Dental Segmentation of Panoramic X-Rays with Bounding Box Priors
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.03747v2](http://arxiv.org/pdf/2406.03747v2)**

> **作者:** Devichand Budagam; Azamat Zhanatuly Imanbayev; Iskander Rafailovich Akhmetov; Aleksandr Sinitca; Sergey Antonov; Dmitrii Kaplun
>
> **备注:** Under Review, Biomedical Signal Processing Control
>
> **摘要:** Teeth segmentation and recognition play a vital role in a variety of dental applications and diagnostic procedures. The integration of deep learning models has facilitated the development of precise and automated segmentation methods. Although prior research has explored teeth segmentation, not many methods have successfully performed tooth segmentation and detection simultaneously. This study presents UFBA-425, a dental dataset derived from the UFBA-UESC dataset, featuring bounding box and polygon annotations for 425 panoramic dental X-rays. Additionally, this work introduces OralBBNet, an architecture featuring distinct segmentation and detection heads as U-Net and YOLOv8, respectively. OralBBNet is designed to improve the accuracy and robustness of tooth classification and segmentation on panoramic X-rays by leveraging the complementary strengths of U-Net and YOLOv8. Our approach achieved a 1-3% improvement in mean average precision (mAP) for teeth detection compared to existing techniques and a 15-20% improvement in the dice score for teeth segmentation over U-Net over various tooth categories and 2-4% improvement in the dice score when compared with other segmentation architectures. The results of this study establish a foundation for the wider implementation of object detection models in dental diagnostics.
>
---
#### [replaced 062] TestDG: Test-time Domain Generalization for Continual Test-time Adaptation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.04981v2](http://arxiv.org/pdf/2504.04981v2)**

> **作者:** Sohyun Lee; Nayeong Kim; Juwon Kang; Seong Joon Oh; Suha Kwak
>
> **摘要:** This paper studies continual test-time adaptation (CTTA), the task of adapting a model to constantly changing unseen domains in testing while preserving previously learned knowledge. Existing CTTA methods mostly focus on adaptation to the current test domain only, overlooking generalization to arbitrary test domains a model may face in the future. To tackle this limitation, we present a novel online test-time domain generalization framework for CTTA, dubbed TestDG. TestDG aims to learn features invariant to both current and previous test domains on the fly during testing, improving the potential for effective generalization to future domains. To this end, we propose a new model architecture and a test-time adaptation strategy dedicated to learning domain-invariant features, along with a new data structure and optimization algorithm for effectively managing information from previous test domains. TestDG achieved state of the art on four public CTTA benchmarks. Moreover, it showed superior generalization to unseen test domains.
>
---
#### [replaced 063] InfoSAM: Fine-Tuning the Segment Anything Model from An Information-Theoretic Perspective
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21920v2](http://arxiv.org/pdf/2505.21920v2)**

> **作者:** Yuanhong Zhang; Muyao Yuan; Weizhan Zhang; Tieliang Gong; Wen Wen; Jiangyong Ying; Weijie Shi
>
> **备注:** Accepted by ICML 2025 (spotlight)
>
> **摘要:** The Segment Anything Model (SAM), a vision foundation model, exhibits impressive zero-shot capabilities in general tasks but struggles in specialized domains. Parameter-efficient fine-tuning (PEFT) is a promising approach to unleash the potential of SAM in novel scenarios. However, existing PEFT methods for SAM neglect the domain-invariant relations encoded in the pre-trained model. To bridge this gap, we propose InfoSAM, an information-theoretic approach that enhances SAM fine-tuning by distilling and preserving its pre-trained segmentation knowledge. Specifically, we formulate the knowledge transfer process as two novel mutual information-based objectives: (i) to compress the domain-invariant relation extracted from pre-trained SAM, excluding pseudo-invariant information as possible, and (ii) to maximize mutual information between the relational knowledge learned by the teacher (pre-trained SAM) and the student (fine-tuned model). The proposed InfoSAM establishes a robust distillation framework for PEFT of SAM. Extensive experiments across diverse benchmarks validate InfoSAM's effectiveness in improving SAM family's performance on real-world tasks, demonstrating its adaptability and superiority in handling specialized scenarios.
>
---
#### [replaced 064] VRAG-RL: Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning with Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.22019v2](http://arxiv.org/pdf/2505.22019v2)**

> **作者:** Qiuchen Wang; Ruixue Ding; Yu Zeng; Zehui Chen; Lin Chen; Shihang Wang; Pengjun Xie; Fei Huang; Feng Zhao
>
> **摘要:** Effectively retrieving, reasoning and understanding visually rich information remains a challenge for RAG methods. Traditional text-based methods cannot handle visual-related information. On the other hand, current vision-based RAG approaches are often limited by fixed pipelines and frequently struggle to reason effectively due to the insufficient activation of the fundamental capabilities of models. As RL has been proven to be beneficial for model reasoning, we introduce VRAG-RL, a novel RL framework tailored for complex reasoning across visually rich information. With this framework, VLMs interact with search engines, autonomously sampling single-turn or multi-turn reasoning trajectories with the help of visual perception tokens and undergoing continual optimization based on these samples. Our approach highlights key limitations of RL in RAG domains: (i) Prior Multi-modal RAG approaches tend to merely incorporate images into the context, leading to insufficient reasoning token allocation and neglecting visual-specific perception; and (ii) When models interact with search engines, their queries often fail to retrieve relevant information due to the inability to articulate requirements, thereby leading to suboptimal performance. To address these challenges, we define an action space tailored for visually rich inputs, with actions including cropping and scaling, allowing the model to gather information from a coarse-to-fine perspective. Furthermore, to bridge the gap between users' original inquiries and the retriever, we employ a simple yet effective reward that integrates query rewriting and retrieval performance with a model-based reward. Our VRAG-RL optimizes VLMs for RAG tasks using specially designed RL strategies, aligning the model with real-world applications. The code is available at https://github.com/Alibaba-NLP/VRAG.
>
---
#### [replaced 065] Generative Emotion Cause Explanation in Multimodal Conversations
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.02430v2](http://arxiv.org/pdf/2411.02430v2)**

> **作者:** Lin Wang; Xiaocui Yang; Shi Feng; Daling Wang; Yifei Zhang; Zhitao Zhang
>
> **摘要:** Multimodal conversation, a crucial form of human communication, carries rich emotional content, making the exploration of the causes of emotions within it a research endeavor of significant importance. However, existing research on the causes of emotions typically employs an utterance selection method within a single textual modality to locate causal utterances. This approach remains limited to coarse-grained assessments, lacks nuanced explanations of emotional causation, and demonstrates inadequate capability in identifying multimodal emotional triggers. Therefore, we introduce a task-\textbf{Multimodal Emotion Cause Explanation in Conversation (MECEC)}. This task aims to generate a summary based on the multimodal context of conversations, clearly and intuitively describing the reasons that trigger a given emotion. To adapt to this task, we develop a new dataset (ECEM) based on the MELD dataset. ECEM combines video clips with detailed explanations of character emotions, helping to explore the causal factors behind emotional expression in multimodal conversations. A novel approach, FAME-Net, is further proposed, that harnesses the power of Large Language Models (LLMs) to analyze visual data and accurately interpret the emotions conveyed through facial expressions in videos. By exploiting the contagion effect of facial emotions, FAME-Net effectively captures the emotional causes of individuals engaged in conversations. Our experimental results on the newly constructed dataset show that FAME-Net outperforms several excellent baselines. Code and dataset are available at https://github.com/3222345200/FAME-Net.
>
---
#### [replaced 066] Towards Computation- and Communication-efficient Computational Pathology
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.02628v2](http://arxiv.org/pdf/2504.02628v2)**

> **作者:** Chu Han; Bingchao Zhao; Jiatai Lin; Shanshan Lyu; Longfei Wang; Tianpeng Deng; Cheng Lu; Changhong Liang; Hannah Y. Wen; Xiaojing Guo; Zhenwei Shi; Zaiyi Liu
>
> **摘要:** Despite the impressive performance across a wide range of applications, current computational pathology models face significant diagnostic efficiency challenges due to their reliance on high-magnification whole-slide image analysis. This limitation severely compromises their clinical utility, especially in time-sensitive diagnostic scenarios and situations requiring efficient data transfer. To address these issues, we present a novel computation- and communication-efficient framework called Magnification-Aligned Global-Local Transformer (MAG-GLTrans). Our approach significantly reduces computational time, file transfer requirements, and storage overhead by enabling effective analysis using low-magnification inputs rather than high-magnification ones. The key innovation lies in our proposed magnification alignment (MAG) mechanism, which employs self-supervised learning to bridge the information gap between low and high magnification levels by effectively aligning their feature representations. Through extensive evaluation across various fundamental CPath tasks, MAG-GLTrans demonstrates state-of-the-art classification performance while achieving remarkable efficiency gains: up to 10.7 times reduction in computational time and over 20 times reduction in file transfer and storage requirements. Furthermore, we highlight the versatility of our MAG framework through two significant extensions: (1) its applicability as a feature extractor to enhance the efficiency of any CPath architecture, and (2) its compatibility with existing foundation models and histopathology-specific encoders, enabling them to process low-magnification inputs with minimal information loss. These advancements position MAG-GLTrans as a particularly promising solution for time-sensitive applications, especially in the context of intraoperative frozen section diagnosis where both accuracy and efficiency are paramount.
>
---
#### [replaced 067] StimuVAR: Spatiotemporal Stimuli-aware Video Affective Reasoning with Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.00304v2](http://arxiv.org/pdf/2409.00304v2)**

> **作者:** Yuxiang Guo; Faizan Siddiqui; Yang Zhao; Rama Chellappa; Shao-Yuan Lo
>
> **备注:** Paper is accepted by IJCV
>
> **摘要:** Predicting and reasoning how a video would make a human feel is crucial for developing socially intelligent systems. Although Multimodal Large Language Models (MLLMs) have shown impressive video understanding capabilities, they tend to focus more on the semantic content of videos, often overlooking emotional stimuli. Hence, most existing MLLMs fall short in estimating viewers' emotional reactions and providing plausible explanations. To address this issue, we propose StimuVAR, a spatiotemporal Stimuli-aware framework for Video Affective Reasoning (VAR) with MLLMs. StimuVAR incorporates a two-level stimuli-aware mechanism: frame-level awareness and token-level awareness. Frame-level awareness involves sampling video frames with events that are most likely to evoke viewers' emotions. Token-level awareness performs tube selection in the token space to make the MLLM concentrate on emotion-triggered spatiotemporal regions. Furthermore, we create VAR instruction data to perform affective training, steering MLLMs' reasoning strengths towards emotional focus and thereby enhancing their affective reasoning ability. To thoroughly assess the effectiveness of VAR, we provide a comprehensive evaluation protocol with extensive metrics. StimuVAR is the first MLLM-based method for viewer-centered VAR. Experiments demonstrate its superiority in understanding viewers' emotional responses to videos and providing coherent and insightful explanations. Our code is available at https://github.com/EthanG97/StimuVAR
>
---
#### [replaced 068] MedVAE: Efficient Automated Interpretation of Medical Images with Large-Scale Generalizable Autoencoders
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.14753v2](http://arxiv.org/pdf/2502.14753v2)**

> **作者:** Maya Varma; Ashwin Kumar; Rogier van der Sluijs; Sophie Ostmeier; Louis Blankemeier; Pierre Chambon; Christian Bluethgen; Jip Prince; Curtis Langlotz; Akshay Chaudhari
>
> **备注:** MIDL 2025 (Oral)
>
> **摘要:** Medical images are acquired at high resolutions with large fields of view in order to capture fine-grained features necessary for clinical decision-making. Consequently, training deep learning models on medical images can incur large computational costs. In this work, we address the challenge of downsizing medical images in order to improve downstream computational efficiency while preserving clinically-relevant features. We introduce MedVAE, a family of six large-scale 2D and 3D autoencoders capable of encoding medical images as downsized latent representations and decoding latent representations back to high-resolution images. We train MedVAE autoencoders using a novel two-stage training approach with 1,052,730 medical images. Across diverse tasks obtained from 20 medical image datasets, we demonstrate that (1) utilizing MedVAE latent representations in place of high-resolution images when training downstream models can lead to efficiency benefits (up to 70x improvement in throughput) while simultaneously preserving clinically-relevant features and (2) MedVAE can decode latent representations back to high-resolution images with high fidelity. Our work demonstrates that large-scale, generalizable autoencoders can help address critical efficiency challenges in the medical domain. Our code is available at https://github.com/StanfordMIMI/MedVAE.
>
---
#### [replaced 069] MCU: An Evaluation Framework for Open-Ended Game Agents
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2310.08367v4](http://arxiv.org/pdf/2310.08367v4)**

> **作者:** Xinyue Zheng; Haowei Lin; Kaichen He; Zihao Wang; Zilong Zheng; Yitao Liang
>
> **摘要:** Developing AI agents capable of interacting with open-world environments to solve diverse tasks is a compelling challenge. However, evaluating such open-ended agents remains difficult, with current benchmarks facing scalability limitations. To address this, we introduce Minecraft Universe (MCU), a comprehensive evaluation framework set within the open-world video game Minecraft. MCU incorporates three key components: (1) an expanding collection of 3,452 composable atomic tasks that encompasses 11 major categories and 41 subcategories of challenges; (2) a task composition mechanism capable of generating infinite diverse tasks with varying difficulty; and (3) a general evaluation framework that achieves 91.5\% alignment with human ratings for open-ended task assessment. Empirical results reveal that even state-of-the-art foundation agents struggle with the increasing diversity and complexity of tasks. These findings highlight the necessity of MCU as a robust benchmark to drive progress in AI agent development within open-ended environments. Our evaluation code and scripts are available at https://github.com/CraftJarvis/MCU.
>
---
#### [replaced 070] Point Cloud Mixture-of-Domain-Experts Model for 3D Self-supervised Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.09886v3](http://arxiv.org/pdf/2410.09886v3)**

> **作者:** Yaohua Zha; Tao Dai; Hang Guo; Yanzi Wang; Bin Chen; Ke Chen; Shu-Tao Xia
>
> **备注:** Accepted to IJCAI 2025
>
> **摘要:** Point clouds, as a primary representation of 3D data, can be categorized into scene domain point clouds and object domain point clouds. Point cloud self-supervised learning (SSL) has become a mainstream paradigm for learning 3D representations. However, existing point cloud SSL primarily focuses on learning domain-specific 3D representations within a single domain, neglecting the complementary nature of cross-domain knowledge, which limits the learning of 3D representations. In this paper, we propose to learn a comprehensive Point cloud Mixture-of-Domain-Experts model (Point-MoDE) via a block-to-scene pre-training strategy. Specifically, we first propose a mixture-of-domain-expert model consisting of scene domain experts and multiple shared object domain experts. Furthermore, we propose a block-to-scene pretraining strategy, which leverages the features of point blocks in the object domain to regress their initial positions in the scene domain through object-level block mask reconstruction and scene-level block position regression. By integrating the complementary knowledge between object and scene, this strategy simultaneously facilitates the learning of both object-domain and scene-domain representations, leading to a more comprehensive 3D representation. Extensive experiments in downstream tasks demonstrate the superiority of our model.
>
---
#### [replaced 071] Beyond Face Swapping: A Diffusion-Based Digital Human Benchmark for Multimodal Deepfake Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.16512v4](http://arxiv.org/pdf/2505.16512v4)**

> **作者:** Jiaxin Liu; Jia Wang; Saihui Hou; Min Ren; Huijia Wu; Long Ma; Renwang Pei; Zhaofeng He
>
> **摘要:** In recent years, the explosive advancement of deepfake technology has posed a critical and escalating threat to public security: diffusion-based digital human generation. Unlike traditional face manipulation methods, such models can generate highly realistic videos with consistency via multimodal control signals. Their flexibility and covertness pose severe challenges to existing detection strategies. To bridge this gap, we introduce DigiFakeAV, the new large-scale multimodal digital human forgery dataset based on diffusion models. Leveraging five of the latest digital human generation methods and a voice cloning method, we systematically construct a dataset comprising 60,000 videos (8.4 million frames), covering multiple nationalities, skin tones, genders, and real-world scenarios, significantly enhancing data diversity and realism. User studies demonstrate that the misrecognition rate by participants for DigiFakeAV reaches as high as 68%. Moreover, the substantial performance degradation of existing detection models on our dataset further highlights its challenges. To address this problem, we propose DigiShield, an effective detection baseline based on spatiotemporal and cross-modal fusion. By jointly modeling the 3D spatiotemporal features of videos and the semantic-acoustic features of audio, DigiShield achieves state-of-the-art (SOTA) performance on the DigiFakeAV and shows strong generalization on other datasets.
>
---
#### [replaced 072] Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs
- **分类: cs.CL; cs.AI; cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2502.11184v2](http://arxiv.org/pdf/2502.11184v2)**

> **作者:** Wenxuan Wang; Xiaoyuan Liu; Kuiyi Gao; Jen-tse Huang; Youliang Yuan; Pinjia He; Shuai Wang; Zhaopeng Tu
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Multimodal Large Language Models (MLLMs) have expanded the capabilities of traditional language models by enabling interaction through both text and images. However, ensuring the safety of these models remains a significant challenge, particularly in accurately identifying whether multimodal content is safe or unsafe-a capability we term safety awareness. In this paper, we introduce MMSafeAware, the first comprehensive multimodal safety awareness benchmark designed to evaluate MLLMs across 29 safety scenarios with 1500 carefully curated image-prompt pairs. MMSafeAware includes both unsafe and over-safety subsets to assess models abilities to correctly identify unsafe content and avoid over-sensitivity that can hinder helpfulness. Evaluating nine widely used MLLMs using MMSafeAware reveals that current models are not sufficiently safe and often overly sensitive; for example, GPT-4V misclassifies 36.1% of unsafe inputs as safe and 59.9% of benign inputs as unsafe. We further explore three methods to improve safety awareness-prompting-based approaches, visual contrastive decoding, and vision-centric reasoning fine-tuning-but find that none achieve satisfactory performance. Our findings highlight the profound challenges in developing MLLMs with robust safety awareness, underscoring the need for further research in this area. All the code and data will be publicly available to facilitate future research.
>
---
#### [replaced 073] A Similarity Paradigm Through Textual Regularization Without Forgetting
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.14376v2](http://arxiv.org/pdf/2502.14376v2)**

> **作者:** Fangming Cui; Jan Fong; Rongfei Zeng; Xinmei Tian; Jun Yu
>
> **摘要:** Prompt learning has emerged as a promising method for adapting pre-trained visual-language models (VLMs) to a range of downstream tasks. While optimizing the context can be effective for improving performance on specific tasks, it can often lead to poor generalization performance on unseen classes or datasets sampled from different distributions. It may be attributed to the fact that textual prompts tend to overfit downstream data distributions, leading to the forgetting of generalized knowledge derived from hand-crafted prompts. In this paper, we propose a novel method called Similarity Paradigm with Textual Regularization (SPTR) for prompt learning without forgetting. SPTR is a two-pronged design based on hand-crafted prompts that is an inseparable framework. 1) To avoid forgetting general textual knowledge, we introduce the optimal transport as a textual regularization to finely ensure approximation with hand-crafted features and tuning textual features. 2) In order to continuously unleash the general ability of multiple hand-crafted prompts, we propose a similarity paradigm for natural alignment score and adversarial alignment score to improve model robustness for generalization. Both modules share a common objective in addressing generalization issues, aiming to maximize the generalization capability derived from multiple hand-crafted prompts. Four representative tasks (i.e., non-generalization few-shot learning, base-to-novel generalization, cross-dataset generalization, domain generalization) across 11 datasets demonstrate that SPTR outperforms existing prompt learning methods.
>
---
#### [replaced 074] DiffVLA: Vision-Language Guided Diffusion Planning for Autonomous Driving
- **分类: cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.19381v4](http://arxiv.org/pdf/2505.19381v4)**

> **作者:** Anqing Jiang; Yu Gao; Zhigang Sun; Yiru Wang; Jijun Wang; Jinghao Chai; Qian Cao; Yuweng Heng; Hao Jiang; Yunda Dong; Zongzheng Zhang; Xianda Guo; Hao Sun; Hao Zhao
>
> **备注:** 4pages
>
> **摘要:** Research interest in end-to-end autonomous driving has surged owing to its fully differentiable design integrating modular tasks, i.e. perception, prediction and planing, which enables optimization in pursuit of the ultimate goal. Despite the great potential of the end-to-end paradigm, existing methods suffer from several aspects including expensive BEV (bird's eye view) computation, action diversity, and sub-optimal decision in complex real-world scenarios. To address these challenges, we propose a novel hybrid sparse-dense diffusion policy, empowered by a Vision-Language Model (VLM), called Diff-VLA. We explore the sparse diffusion representation for efficient multi-modal driving behavior. Moreover, we rethink the effectiveness of VLM driving decision and improve the trajectory generation guidance through deep interaction across agent, map instances and VLM output. Our method shows superior performance in Autonomous Grand Challenge 2025 which contains challenging real and reactive synthetic scenarios. Our methods achieves 45.0 PDMS.
>
---
#### [replaced 075] Scene Structure Guidance Network: Unfolding Graph Partitioning into Pixel-Wise Feature Learning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2301.00555v2](http://arxiv.org/pdf/2301.00555v2)**

> **作者:** Jisu Shin; Seunghyun Shin; Hae-Gon Jeon
>
> **备注:** 35 pages, 14 figures, journal extension version of SSGNet (https://ojs.aaai.org/index.php/AAAI/article/view/25322)
>
> **摘要:** Understanding the informative structures of scenes is essential for low-level vision tasks. Unfortunately, it is difficult to obtain a concrete visual definition of the informative structures because influences of visual features are task-specific. In this paper, we propose a single general neural network architecture for extracting task-specific structure guidance for scenes. To do this, we first analyze traditional spectral clustering methods, which computes a set of eigenvectors to model a segmented graph forming small compact structures on image domains. We then unfold the traditional graph-partitioning problem into a learnable network, named \textit{Scene Structure Guidance Network (SSGNet)}, to represent the task-specific informative structures. The SSGNet yields a set of coefficients of eigenvectors that produces explicit feature representations of image structures. In addition, our SSGNet is light-weight ($\sim$ 56K parameters), and can be used as a plug-and-play module for off-the-shelf architectures. We optimize the SSGNet without any supervision by proposing two novel training losses that enforce task-specific scene structure generation during training. Our main contribution is to show that such a simple network can achieve state-of-the-art results for several low-level vision applications. We also demonstrate that our network generalizes well on unseen datasets, compared to existing methods which use structural embedding frameworks. We further propose a lighter version of SSGNet ($\sim$ 29K parameters) for depth computation, SSGNet-D, and successfully execute it on edge computing devices like Jetson AGX Orin, improving the performance of baseline network, even in the wild, with little computational delay.
>
---
#### [replaced 076] Datasheets Aren't Enough: DataRubrics for Automated Quality Metrics and Accountability
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.01789v2](http://arxiv.org/pdf/2506.01789v2)**

> **作者:** Genta Indra Winata; David Anugraha; Emmy Liu; Alham Fikri Aji; Shou-Yi Hung; Aditya Parashar; Patrick Amadeus Irawan; Ruochen Zhang; Zheng-Xin Yong; Jan Christian Blaise Cruz; Niklas Muennighoff; Seungone Kim; Hanyang Zhao; Sudipta Kar; Kezia Erina Suryoraharjo; M. Farid Adilazuarda; En-Shiun Annie Lee; Ayu Purwarianti; Derry Tanti Wijaya; Monojit Choudhury
>
> **备注:** Preprint
>
> **摘要:** High-quality datasets are fundamental to training and evaluating machine learning models, yet their creation-especially with accurate human annotations-remains a significant challenge. Many dataset paper submissions lack originality, diversity, or rigorous quality control, and these shortcomings are often overlooked during peer review. Submissions also frequently omit essential details about dataset construction and properties. While existing tools such as datasheets aim to promote transparency, they are largely descriptive and do not provide standardized, measurable methods for evaluating data quality. Similarly, metadata requirements at conferences promote accountability but are inconsistently enforced. To address these limitations, this position paper advocates for the integration of systematic, rubric-based evaluation metrics into the dataset review process-particularly as submission volumes continue to grow. We also explore scalable, cost-effective methods for synthetic data generation, including dedicated tools and LLM-as-a-judge approaches, to support more efficient evaluation. As a call to action, we introduce DataRubrics, a structured framework for assessing the quality of both human- and model-generated datasets. Leveraging recent advances in LLM-based evaluation, DataRubrics offers a reproducible, scalable, and actionable solution for dataset quality assessment, enabling both authors and reviewers to uphold higher standards in data-centric research. We also release code to support reproducibility of LLM-based evaluations at https://github.com/datarubrics/datarubrics.
>
---
#### [replaced 077] SASP: Strip-Aware Spatial Perception for Fine-Grained Bird Image Classification
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.24380v2](http://arxiv.org/pdf/2505.24380v2)**

> **作者:** Zheng Wang
>
> **摘要:** Fine-grained bird image classification (FBIC) is not only of great significance for ecological monitoring and species identification, but also holds broad research value in the fields of image recognition and fine-grained visual modeling. Compared with general image classification tasks, FBIC poses more formidable challenges: 1) the differences in species size and imaging distance result in the varying sizes of birds presented in the images; 2) complex natural habitats often introduce strong background interference; 3) and highly flexible poses such as flying, perching, or foraging result in substantial intra-class variability. These factors collectively make it difficult for traditional methods to stably extract discriminative features, thereby limiting the generalizability and interpretability of models in real-world applications. To address these challenges, this paper proposes a fine-grained bird classification framework based on strip-aware spatial perception, which aims to capture long-range spatial dependencies across entire rows or columns in bird images, thereby enhancing the model's robustness and interpretability. The proposed method incorporates two novel modules: extensional perception aggregator (EPA) and channel semantic weaving (CSW). Specifically, EPA integrates local texture details with global structural cues by aggregating information across horizontal and vertical spatial directions. CSW further refines the semantic representations by adaptively fusing long-range and short-range information along the channel dimension. Built upon a ResNet-50 backbone, the model enables jump-wise connection of extended structural features across the spatial domain. Experimental results on the CUB-200-2011 dataset demonstrate that our framework achieves significant performance improvements while maintaining architectural efficiency.
>
---
#### [replaced 078] PointCloud-Text Matching: Benchmark Datasets and a Baseline
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2403.19386v3](http://arxiv.org/pdf/2403.19386v3)**

> **作者:** Yanglin Feng; Yang Qin; Dezhong Peng; Hongyuan Zhu; Xi Peng; Peng Hu
>
> **备注:** The version submitted this time has been significantly revised and improved on the previous version
>
> **摘要:** In this paper, we present and study a new instance-level retrieval task: PointCloud-Text Matching (PTM), which aims to identify the exact cross-modal instance that matches a given point-cloud query or text query. PTM has potential applications in various scenarios, such as indoor/urban-canyon localization and scene retrieval. However, there is a lack of suitable and targeted datasets for PTM in practice. To address this issue, we present a new PTM benchmark dataset, namely SceneDepict-3D2T. We observe that the data poses significant challenges due to its inherent characteristics, such as the sparsity, noise, or disorder of point clouds and the ambiguity, vagueness, or incompleteness of texts, which render existing cross-modal matching methods ineffective for PTM. To overcome these challenges, we propose a PTM baseline, named Robust PointCloud-Text Matching method (RoMa). RoMa consists of two key modules: a Dual Attention Perception module (DAP) and a Robust Negative Contrastive Learning module (RNCL). Specifically, DAP leverages token-level and feature-level attention mechanisms to adaptively focus on useful local and global features, and aggregate them into common representations, thereby reducing the adverse impact of noise and ambiguity. To handle noisy correspondence, RNCL enhances robustness against mismatching by dividing negative pairs into clean and noisy subsets and assigning them forward and reverse optimization directions, respectively. We conduct extensive experiments on our benchmarks and demonstrate the superiority of our RoMa.
>
---
#### [replaced 079] P-TAME: Explain Any Image Classifier with Trained Perturbations
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.17813v2](http://arxiv.org/pdf/2501.17813v2)**

> **作者:** Mariano V. Ntrougkas; Vasileios Mezaris; Ioannis Patras
>
> **备注:** Published in IEEE Open Journal of Signal Processing (Volume 6)
>
> **摘要:** The adoption of Deep Neural Networks (DNNs) in critical fields where predictions need to be accompanied by justifications is hindered by their inherent black-box nature. In this paper, we introduce P-TAME (Perturbation-based Trainable Attention Mechanism for Explanations), a model-agnostic method for explaining DNN-based image classifiers. P-TAME employs an auxiliary image classifier to extract features from the input image, bypassing the need to tailor the explanation method to the internal architecture of the backbone classifier being explained. Unlike traditional perturbation-based methods, which have high computational requirements, P-TAME offers an efficient alternative by generating high-resolution explanations in a single forward pass during inference. We apply P-TAME to explain the decisions of VGG-16, ResNet-50, and ViT-B-16, three distinct and widely used image classifiers. Quantitative and qualitative results show that our method matches or outperforms previous explainability methods, including model-specific approaches. Code and trained models will be released upon acceptance.
>
---
#### [replaced 080] Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21179v3](http://arxiv.org/pdf/2505.21179v3)**

> **作者:** Dar-Yen Chen; Hmrishav Bandyopadhyay; Kai Zou; Yi-Zhe Song
>
> **摘要:** Negative guidance -- explicitly suppressing unwanted attributes -- remains a fundamental challenge in diffusion models, particularly in few-step sampling regimes. While Classifier-Free Guidance (CFG) works well in standard settings, it fails under aggressive sampling step compression due to divergent predictions between positive and negative branches. We present Normalized Attention Guidance (NAG), an efficient, training-free mechanism that applies extrapolation in attention space with L1-based normalization and refinement. NAG restores effective negative guidance where CFG collapses while maintaining fidelity. Unlike existing approaches, NAG generalizes across architectures (UNet, DiT), sampling regimes (few-step, multi-step), and modalities (image, video), functioning as a \textit{universal} plug-in with minimal computational overhead. Through extensive experimentation, we demonstrate consistent improvements in text alignment (CLIP Score), fidelity (FID, PFID), and human-perceived quality (ImageReward). Our ablation studies validate each design component, while user studies confirm significant preference for NAG-guided outputs. As a model-agnostic inference-time approach requiring no retraining, NAG provides effortless negative guidance for all modern diffusion frameworks -- pseudocode in the Appendix!
>
---
#### [replaced 081] On the Expressiveness of Visual Prompt Experts
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.18936v5](http://arxiv.org/pdf/2501.18936v5)**

> **作者:** Minh Le; Anh Nguyen; Huy Nguyen; Chau Nguyen; Anh Tran; Nhat Ho
>
> **备注:** 44 pages, 8 figures, 20 tables
>
> **摘要:** Visual Prompt Tuning (VPT) has proven effective for parameter-efficient adaptation of pre-trained vision models to downstream tasks by inserting task-specific learnable prompt tokens. Despite its empirical success, a comprehensive theoretical understanding of VPT remains an active area of research. Building on the recently established connection between Mixture of Experts (MoE) and prompt-based methods, wherein each attention head can be conceptualized as a composition of multiple MoE models, we reinterpret VPT as the introduction of new prompt experts into these MoE structures. We identify a key limitation in existing VPT frameworks: the restricted functional expressiveness of prompt experts, which remain static and thus limited in their adaptability. To address this, we propose Visual Adaptive Prompt Tuning (VAPT), a novel method that endows prompt experts with enhanced expressiveness while preserving parameter efficiency. Empirical evaluations on VTAB-1K and FGVC demonstrate that VAPT achieves substantial performance improvements, surpassing fully fine-tuned baselines by 7.34% and 1.04%, respectively. Moreover, VAPT consistently outperforms VPT while requiring fewer additional parameters. Furthermore, our theoretical analysis indicates that VAPT achieves optimal sample efficiency. Collectively, these results underscore the theoretical grounding and empirical advantages of our approach.
>
---
#### [replaced 082] PHISWID: Physics-Inspired Underwater Image Dataset Synthesized from RGB-D Images
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2404.03998v3](http://arxiv.org/pdf/2404.03998v3)**

> **作者:** Reina Kaneko; Takumi Ueda; Hiroshi Higashi; Yuichi Tanaka
>
> **摘要:** This paper introduces the physics-inspired synthesized underwater image dataset (PHISWID), a dataset tailored for enhancing underwater image processing through physics-inspired image synthesis. For underwater image enhancement, data-driven approaches (e.g., deep neural networks) typically demand extensive datasets, yet acquiring paired clean atmospheric images and degraded underwater images poses significant challenges. Existing datasets have limited contributions to image enhancement due to lack of physics models, publicity, and ground-truth atmospheric images. PHISWID addresses these issues by offering a set of paired atmospheric and underwater images. Specifically, underwater images are synthetically degraded by color degradation and marine snow artifacts from atmospheric RGB-D images. It is enabled based on a physics-based underwater image observation model. Our synthetic approach generates a large quantity of the pairs, enabling effective training of deep neural networks and objective image quality assessment. Through benchmark experiments with some datasets and image enhancement methods, we validate that our dataset can improve the image enhancement performance. Our dataset, which is publicly available, contributes to the development in underwater image processing.
>
---
#### [replaced 083] Learning from True-False Labels via Multi-modal Prompt Retrieving
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2405.15228v2](http://arxiv.org/pdf/2405.15228v2)**

> **作者:** Zhongnian Li; Jinghao Xu; Peng Ying; Meng Wei; Xinzheng Xu
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** Pre-trained Vision-Language Models (VLMs) exhibit strong zero-shot classification abilities, demonstrating great potential for generating weakly supervised labels. Unfortunately, existing weakly supervised learning methods are short of ability in generating accurate labels via VLMs. In this paper, we propose a novel weakly supervised labeling setting, namely True-False Labels (TFLs) which can achieve high accuracy when generated by VLMs. The TFL indicates whether an instance belongs to the label, which is randomly and uniformly sampled from the candidate label set. Specifically, we theoretically derive a risk-consistent estimator to explore and utilize the conditional probability distribution information of TFLs. Besides, we propose a convolutional-based Multi-modal Prompt Retrieving (MRP) method to bridge the gap between the knowledge of VLMs and target learning tasks. Experimental results demonstrate the effectiveness of the proposed TFL setting and MRP learning method. The code to reproduce the experiments is at https://github.com/Tranquilxu/TMP.
>
---
#### [replaced 084] S4-Driver: Scalable Self-Supervised Driving Multimodal Large Language Modelwith Spatio-Temporal Visual Representation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.24139v2](http://arxiv.org/pdf/2505.24139v2)**

> **作者:** Yichen Xie; Runsheng Xu; Tong He; Jyh-Jing Hwang; Katie Luo; Jingwei Ji; Hubert Lin; Letian Chen; Yiren Lu; Zhaoqi Leng; Dragomir Anguelov; Mingxing Tan
>
> **备注:** Accepted by CVPR2025; Project website: s4-driver.github.io
>
> **摘要:** The latest advancements in multi-modal large language models (MLLMs) have spurred a strong renewed interest in end-to-end motion planning approaches for autonomous driving. Many end-to-end approaches rely on human annotations to learn intermediate perception and prediction tasks, while purely self-supervised approaches--which directly learn from sensor inputs to generate planning trajectories without human annotations often underperform the state of the art. We observe a key gap in the input representation space: end-to-end approaches built on MLLMs are often pretrained with reasoning tasks in 2D image space rather than the native 3D space in which autonomous vehicles plan. To this end, we propose S4-Driver, a scalable self-supervised motion planning algorithm with spatio-temporal visual representation, based on the popular PaLI multimodal large language model. S4-Driver uses a novel sparse volume strategy to seamlessly transform the strong visual representation of MLLMs from perspective view to 3D space without the need to finetune the vision encoder. This representation aggregates multi-view and multi-frame visual inputs and enables better prediction of planning trajectories in 3D space. To validate our method, we run experiments on both nuScenes and Waymo Open Motion Dataset (with in-house camera data). Results show that S4-Driver performs favorably against existing supervised multi-task approaches while requiring no human annotations. It also demonstrates great scalability when pretrained on large volumes of unannotated driving logs.
>
---
#### [replaced 085] S3D: Sketch-Driven 3D Model Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.04185v2](http://arxiv.org/pdf/2505.04185v2)**

> **作者:** Hail Song; Wonsik Shin; Naeun Lee; Soomin Chung; Nojun Kwak; Woontack Woo
>
> **备注:** Accepted as a short paper to the GMCV Workshop at CVPR'25
>
> **摘要:** Generating high-quality 3D models from 2D sketches is a challenging task due to the inherent ambiguity and sparsity of sketch data. In this paper, we present S3D, a novel framework that converts simple hand-drawn sketches into detailed 3D models. Our method utilizes a U-Net-based encoder-decoder architecture to convert sketches into face segmentation masks, which are then used to generate a 3D representation that can be rendered from novel views. To ensure robust consistency between the sketch domain and the 3D output, we introduce a novel style-alignment loss that aligns the U-Net bottleneck features with the initial encoder outputs of the 3D generation module, significantly enhancing reconstruction fidelity. To further enhance the network's robustness, we apply augmentation techniques to the sketch dataset. This streamlined framework demonstrates the effectiveness of S3D in generating high-quality 3D models from sketch inputs. The source code for this project is publicly available at https://github.com/hailsong/S3D.
>
---
#### [replaced 086] CodeEnhance: A Codebook-Driven Approach for Low-Light Image Enhancement
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2404.05253v3](http://arxiv.org/pdf/2404.05253v3)**

> **作者:** Xu Wu; XianXu Hou; Zhihui Lai; Jie Zhou; Ya-nan Zhang; Witold Pedrycz; Linlin Shen
>
> **备注:** 10 pages, 13 figures
>
> **摘要:** Low-light image enhancement (LLIE) aims to improve low-illumination images. However, existing methods face two challenges: (1) uncertainty in restoration from diverse brightness degradations; (2) loss of texture and color information caused by noise suppression and light enhancement. In this paper, we propose a novel enhancement approach, CodeEnhance, by leveraging quantized priors and image refinement to address these challenges. In particular, we reframe LLIE as learning an image-to-code mapping from low-light images to discrete codebook, which has been learned from high-quality images. To enhance this process, a Semantic Embedding Module (SEM) is introduced to integrate semantic information with low-level features, and a Codebook Shift (CS) mechanism, designed to adapt the pre-learned codebook to better suit the distinct characteristics of our low-light dataset. Additionally, we present an Interactive Feature Transformation (IFT) module to refine texture and color information during image reconstruction, allowing for interactive enhancement based on user preferences. Extensive experiments on both real-world and synthetic benchmarks demonstrate that the incorporation of prior knowledge and controllable information transfer significantly enhances LLIE performance in terms of quality and fidelity. The proposed CodeEnhance exhibits superior robustness to various degradations, including uneven illumination, noise, and color distortion.
>
---
#### [replaced 087] Adversarial Robustness of AI-Generated Image Detectors in the Real World
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.01574v3](http://arxiv.org/pdf/2410.01574v3)**

> **作者:** Sina Mavali; Jonas Ricker; David Pape; Asja Fischer; Lea Schönherr
>
> **摘要:** The rapid advancement of Generative Artificial Intelligence (GenAI) capabilities is accompanied by a concerning rise in its misuse. In particular the generation of credible misinformation in the form of images poses a significant threat to the public trust in democratic processes. Consequently, there is an urgent need to develop tools to reliably distinguish between authentic and AI-generated content. The majority of detection methods are based on neural networks that are trained to recognize forensic artifacts. In this work, we demonstrate that current state-of-the-art classifiers are vulnerable to adversarial examples under real-world conditions. Through extensive experiments, comprising four detection methods and five attack algorithms, we show that an attacker can dramatically decrease classification performance, without internal knowledge of the detector's architecture. Notably, most attacks remain effective even when images are degraded during the upload to, e.g., social media platforms. In a case study, we demonstrate that these robustness challenges are also found in commercial tools by conducting black-box attacks on HIVE, a proprietary online GenAI media detector. In addition, we evaluate the robustness of using generated features of a robust pre-trained model and showed that this increases the robustness, while not reaching the performance on benign inputs. These results, along with the increasing potential of GenAI to erode public trust, underscore the need for more research and new perspectives on methods to prevent its misuse.
>
---
#### [replaced 088] Inclusion 2024 Global Multimedia Deepfake Detection Challenge: Towards Multi-dimensional Face Forgery Detection
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2412.20833v2](http://arxiv.org/pdf/2412.20833v2)**

> **作者:** Yi Zhang; Weize Gao; Changtao Miao; Man Luo; Jianshu Li; Wenzhong Deng; Zhe Li; Bingyu Hu; Weibin Yao; Yunfeng Diao; Wenbo Zhou; Tao Gong; Qi Chu
>
> **备注:** Inclusion 2024 Global Multimedia Deepfake Detection Competition Top Team Technical Report
>
> **摘要:** In this paper, we present the Global Multimedia Deepfake Detection held concurrently with the Inclusion 2024. Our Multimedia Deepfake Detection aims to detect automatic image and audio-video manipulations including but not limited to editing, synthesis, generation, Photoshop,etc. Our challenge has attracted 1500 teams from all over the world, with about 5000 valid result submission counts. We invite the top 20 teams to present their solutions to the challenge, from which the top 3 teams are awarded prizes in the grand finale. In this paper, we present the solutions from the top 3 teams of the two tracks, to boost the research work in the field of image and audio-video forgery detection. The methodologies developed through the challenge will contribute to the development of next-generation deepfake detection systems and we encourage participants to open source their methods.
>
---
#### [replaced 089] Foundation Models for Remote Sensing and Earth Observation: A Survey
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.16602v3](http://arxiv.org/pdf/2410.16602v3)**

> **作者:** Aoran Xiao; Weihao Xuan; Junjue Wang; Jiaxing Huang; Dacheng Tao; Shijian Lu; Naoto Yokoya
>
> **备注:** Accepted to IEEE Geoscience and Remote Sensing Magazine (GRSM)
>
> **摘要:** Remote Sensing (RS) is a crucial technology for observing, monitoring, and interpreting our planet, with broad applications across geoscience, economics, humanitarian fields, etc. While artificial intelligence (AI), particularly deep learning, has achieved significant advances in RS, unique challenges persist in developing more intelligent RS systems, including the complexity of Earth's environments, diverse sensor modalities, distinctive feature patterns, varying spatial and spectral resolutions, and temporal dynamics. Meanwhile, recent breakthroughs in large Foundation Models (FMs) have expanded AI's potential across many domains due to their exceptional generalizability and zero-shot transfer capabilities. However, their success has largely been confined to natural data like images and video, with degraded performance and even failures for RS data of various non-optical modalities. This has inspired growing interest in developing Remote Sensing Foundation Models (RSFMs) to address the complex demands of Earth Observation (EO) tasks, spanning the surface, atmosphere, and oceans. This survey systematically reviews the emerging field of RSFMs. It begins with an outline of their motivation and background, followed by an introduction of their foundational concepts. It then categorizes and reviews existing RSFM studies including their datasets and technical contributions across Visual Foundation Models (VFMs), Visual-Language Models (VLMs), Large Language Models (LLMs), and beyond. In addition, we benchmark these models against publicly available datasets, discuss existing challenges, and propose future research directions in this rapidly evolving field. A project associated with this survey has been built at https://github.com/xiaoaoran/awesome-RSFMs .
>
---
#### [replaced 090] SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.16765v2](http://arxiv.org/pdf/2411.16765v2)**

> **作者:** Shester Gueuwou; Xiaodan Du; Greg Shakhnarovich; Karen Livescu; Alexander H. Liu
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** Sign language processing has traditionally relied on task-specific models, limiting the potential for transfer learning across tasks. Pre-training methods for sign language have typically focused on either supervised pre-training, which cannot take advantage of unlabeled data, or context-independent (frame or video segment) representations, which ignore the effects of relationships across time in sign language. We introduce SHuBERT (Sign Hidden-Unit BERT), a self-supervised contextual representation model learned from approximately 1,000 hours of American Sign Language video. SHuBERT adapts masked token prediction objectives to multi-stream visual sign language input, learning to predict multiple targets corresponding to clustered hand, face, and body pose streams. SHuBERT achieves state-of-the-art performance across multiple tasks including sign language translation, isolated sign language recognition, and fingerspelling detection.
>
---
#### [replaced 091] Video Motion Graphs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.20218v2](http://arxiv.org/pdf/2503.20218v2)**

> **作者:** Haiyang Liu; Zhan Xu; Fa-Ting Hong; Hsin-Ping Huang; Yi Zhou; Yang Zhou
>
> **备注:** 14 pages,10 figures
>
> **摘要:** We present Video Motion Graphs, a system designed to generate realistic human motion videos. Using a reference video and conditional signals such as music or motion tags, the system synthesizes new videos by first retrieving video clips with gestures matching the conditions and then generating interpolation frames to seamlessly connect clip boundaries. The core of our approach is HMInterp, a robust Video Frame Interpolation (VFI) model that enables seamless interpolation of discontinuous frames, even for complex motion scenarios like dancing. HMInterp i) employs a dual-branch interpolation approach, combining a Motion Diffusion Model for human skeleton motion interpolation with a diffusion-based video frame interpolation model for final frame generation. ii) adopts condition progressive training to effectively leverage identity strong and weak conditions, such as images and pose. These designs ensure both high video texture quality and accurate motion trajectory. Results show that our Video Motion Graphs outperforms existing generative- and retrieval-based methods for multi-modal conditioned human motion video generation. Project page can be found at https://h-liu1997.github.io/Video-Motion-Graphs/
>
---
#### [replaced 092] Improving Heart Rejection Detection in XPCI Images Using Synthetic Data Augmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19746v2](http://arxiv.org/pdf/2505.19746v2)**

> **作者:** Jakov Samardžija; Donik Vršnak; Sven Lončarić
>
> **备注:** For the time being, the paper needs to be withdrawn so that a more extensive evaluation of the results can be conducted to validate the approach. Furthermore, additional authors will need to be added, which will be addressed if the study's results prove satisfactory
>
> **摘要:** Accurate identification of acute cellular rejection (ACR) in endomyocardial biopsies is essential for effective management of heart transplant patients. However, the rarity of high-grade rejection cases (3R) presents a significant challenge for training robust deep learning models. This work addresses the class imbalance problem by leveraging synthetic data generation using StyleGAN to augment the limited number of real 3R images. Prior to GAN training, histogram equalization was applied to standardize image appearance and improve the consistency of tissue representation. StyleGAN was trained on available 3R biopsy patches and subsequently used to generate 10,000 realistic synthetic images. These were combined with real 0R samples, that is samples without rejection, in various configurations to train ResNet-18 classifiers for binary rejection classification. Three classifier variants were evaluated: one trained on real 0R and synthetic 3R images, another using both synthetic and additional real samples, and a third trained solely on real data. All models were tested on an independent set of real biopsy images. Results demonstrate that synthetic data improves classification performance, particularly when used in combination with real samples. The highest-performing model, which used both real and synthetic images, achieved strong precision and recall for both classes. These findings underscore the value of hybrid training strategies and highlight the potential of GAN-based data augmentation in biomedical image analysis, especially in domains constrained by limited annotated datasets.
>
---
#### [replaced 093] SignMusketeers: An Efficient Multi-Stream Approach for Sign Language Translation at Scale
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.06907v2](http://arxiv.org/pdf/2406.06907v2)**

> **作者:** Shester Gueuwou; Xiaodan Du; Greg Shakhnarovich; Karen Livescu
>
> **备注:** Accepted to ACL (Findings) 2025
>
> **摘要:** A persistent challenge in sign language video processing, including the task of sign to written language translation, is how we learn representations of sign language in an effective and efficient way that preserves the important attributes of these languages, while remaining invariant to irrelevant visual differences. Informed by the nature and linguistics of signed languages, our proposed method focuses on just the most relevant parts in a signing video: the face, hands and body pose of the signer. However, instead of fully relying on pose estimation from off-the-shelf pose tracking models, which have inconsistent performance for hands and faces, we propose to learn a representation of the complex handshapes and facial expressions of sign languages in a self-supervised fashion. Our approach is based on learning from individual frames (rather than video sequences) and is therefore much more efficient than prior work on sign language pre-training. Compared to a recent model that established a new state of the art in sign language translation on the How2Sign dataset, our approach yields similar translation performance, using less than 3\% of the compute.
>
---
#### [replaced 094] Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms
- **分类: cs.CL; cs.AI; cs.CV; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.20322v2](http://arxiv.org/pdf/2505.20322v2)**

> **作者:** Mengru Wang; Ziwen Xu; Shengyu Mao; Shumin Deng; Zhaopeng Tu; Huajun Chen; Ningyu Zhang
>
> **备注:** ACL 2025
>
> **摘要:** Precise control over language model generation is vital for ensuring both safety and reliability. Although prompt engineering and steering are commonly used to intervene in model behaviors, the vast number of parameters in models often results in highly intertwined internal representations. This interdependency can limit control precision and sometimes lead to unintended side effects. Recent research has explored the use of sparse autoencoders (SAE) to disentangle knowledge in high-dimensional spaces for steering. However, these applications have been limited to toy tasks owing to the nontrivial issue of locating atomic knowledge components. In this paper, we propose Steering Target Atoms (STA), a novel method that isolates and manipulates disentangled knowledge components to enhance safety. Comprehensive experiments demonstrate the effectiveness of our approach. Further analysis reveals that steering exhibits superior robustness and flexibility, particularly in adversarial scenarios. We also apply the steering strategy to the large reasoning model, confirming its effectiveness in precise reasoning control.
>
---
#### [replaced 095] Shape and Texture Recognition in Large Vision-Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.23062v2](http://arxiv.org/pdf/2503.23062v2)**

> **作者:** Sagi Eppel; Mor Bismut; Alona Faktor-Strugatski
>
> **摘要:** Shapes and textures are the basic building blocks of visual perception. The ability to identify shapes regardless of orientation, texture, or context, and to recognize textures and materials independently of their associated objects, is essential for a general visual understanding of the world. This work introduces the Large Shape and Textures dataset (LAS&T), a giant collection of highly diverse shapes and textures, created by unsupervised extraction of patterns from natural images. This dataset is used to benchmark how effectively leading Large Vision-Language Models (LVLMs) understand shapes, textures, and materials in 2D and 3D scenes. For shape recognition, we test the models' ability to match images of identical shapes that differ in orientation, texture, color, or environment. Our results show that the shape recognition capabilities of the LVLMs remain significantly below human performance. LVLMs rely predominantly on high-level and semantic features and struggle with abstract shapes lacking clear class associations. For texture and material recognition, we evaluated the models' ability to identify images with identical textures and materials across different objects and environments. Interestingly, leading LVLMs approach human-level performance in recognizing materials in 3D scenes, yet substantially underperform humans when identifying simpler more abstract 2D textures. These results are consistent across a wide range of leading VLMs (GPT/Gemini/LLama/Qwen) and foundation vision models (DINO/CLIP), exposing major deficiencies in the ability of leading models to understand fundamental visual concepts. In contrast, simple nets trained directly for these tasks achieve high accuracy. The LAS&T dataset has been made available.
>
---
#### [replaced 096] Chain-of-Jailbreak Attack for Image Generation Models via Editing Step by Step
- **分类: cs.CL; cs.AI; cs.CR; cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2410.03869v2](http://arxiv.org/pdf/2410.03869v2)**

> **作者:** Wenxuan Wang; Kuiyi Gao; Youliang Yuan; Jen-tse Huang; Qiuzhi Liu; Shuai Wang; Wenxiang Jiao; Zhaopeng Tu
>
> **备注:** Accepted by ACL 2025 Findings
>
> **摘要:** Text-based image generation models, such as Stable Diffusion and DALL-E 3, hold significant potential in content creation and publishing workflows, making them the focus in recent years. Despite their remarkable capability to generate diverse and vivid images, considerable efforts are being made to prevent the generation of harmful content, such as abusive, violent, or pornographic material. To assess the safety of existing models, we introduce a novel jailbreaking method called Chain-of-Jailbreak (CoJ) attack, which compromises image generation models through a step-by-step editing process. Specifically, for malicious queries that cannot bypass the safeguards with a single prompt, we intentionally decompose the query into multiple sub-queries. The image generation models are then prompted to generate and iteratively edit images based on these sub-queries. To evaluate the effectiveness of our CoJ attack method, we constructed a comprehensive dataset, CoJ-Bench, encompassing nine safety scenarios, three types of editing operations, and three editing elements. Experiments on four widely-used image generation services provided by GPT-4V, GPT-4o, Gemini 1.5 and Gemini 1.5 Pro, demonstrate that our CoJ attack method can successfully bypass the safeguards of models for over 60% cases, which significantly outperforms other jailbreaking methods (i.e., 14%). Further, to enhance these models' safety against our CoJ attack method, we also propose an effective prompting-based method, Think Twice Prompting, that can successfully defend over 95% of CoJ attack. We release our dataset and code to facilitate the AI safety research.
>
---
#### [replaced 097] Dynamic-I2V: Exploring Image-to-Video Generation Models via Multimodal LLM
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19901v3](http://arxiv.org/pdf/2505.19901v3)**

> **作者:** Peng Liu; Xiaoming Ren; Fengkai Liu; Qingsong Xie; Quanlong Zheng; Yanhao Zhang; Haonan Lu; Yujiu Yang
>
> **摘要:** Recent advancements in image-to-video (I2V) generation have shown promising performance in conventional scenarios. However, these methods still encounter significant challenges when dealing with complex scenes that require a deep understanding of nuanced motion and intricate object-action relationships. To address these challenges, we present Dynamic-I2V, an innovative framework that integrates Multimodal Large Language Models (MLLMs) to jointly encode visual and textual conditions for a diffusion transformer (DiT) architecture. By leveraging the advanced multimodal understanding capabilities of MLLMs, our model significantly improves motion controllability and temporal coherence in synthesized videos. The inherent multimodality of Dynamic-I2V further enables flexible support for diverse conditional inputs, extending its applicability to various downstream generation tasks. Through systematic analysis, we identify a critical limitation in current I2V benchmarks: a significant bias towards favoring low-dynamic videos, stemming from an inadequate balance between motion complexity and visual quality metrics. To resolve this evaluation gap, we propose DIVE - a novel assessment benchmark specifically designed for comprehensive dynamic quality measurement in I2V generation. In conclusion, extensive quantitative and qualitative experiments confirm that Dynamic-I2V attains state-of-the-art performance in image-to-video generation, particularly revealing significant improvements of 42.5%, 7.9%, and 11.8% in dynamic range, controllability, and quality, respectively, as assessed by the DIVE metric in comparison to existing methods.
>
---
#### [replaced 098] MedEBench: Revisiting Text-instructed Image Editing on Medical Domain
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.01921v2](http://arxiv.org/pdf/2506.01921v2)**

> **作者:** Minghao Liu; Zhitao He; Zhiyuan Fan; Qingyun Wang; Yi R. Fung
>
> **摘要:** Text-guided image editing has seen rapid progress in natural image domains, but its adaptation to medical imaging remains limited and lacks standardized evaluation. Clinically, such editing holds promise for simulating surgical outcomes, creating personalized teaching materials, and enhancing patient communication. To bridge this gap, we introduce \textbf{MedEBench}, a comprehensive benchmark for evaluating text-guided medical image editing. It consists of 1,182 clinically sourced image-prompt triplets spanning 70 tasks across 13 anatomical regions. MedEBench offers three key contributions: (1) a clinically relevant evaluation framework covering Editing Accuracy, Contextual Preservation, and Visual Quality, supported by detailed descriptions of expected change and ROI (Region of Interest) masks; (2) a systematic comparison of seven state-of-the-art models, revealing common failure patterns; and (3) a failure analysis protocol based on attention grounding, using IoU between attention maps and ROIs to identify mislocalization. MedEBench provides a solid foundation for developing and evaluating reliable, clinically meaningful medical image editing systems.
>
---
#### [replaced 099] GvT: A Graph-based Vision Transformer with Talking-Heads Utilizing Sparsity, Trained from Scratch on Small Datasets
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2404.04924v2](http://arxiv.org/pdf/2404.04924v2)**

> **作者:** Dongjing Shan; guiqiang chen
>
> **备注:** The authors withdraw this article to revise and improve the paper through substantial adjustments and rewriting
>
> **摘要:** Vision Transformers (ViTs) have achieved impressive results in large-scale image classification. However, when training from scratch on small datasets, there is still a significant performance gap between ViTs and Convolutional Neural Networks (CNNs), which is attributed to the lack of inductive bias. To address this issue, we propose a Graph-based Vision Transformer (GvT) that utilizes graph convolutional projection and graph-pooling. In each block, queries and keys are calculated through graph convolutional projection based on the spatial adjacency matrix, while dot-product attention is used in another graph convolution to generate values. When using more attention heads, the queries and keys become lower-dimensional, making their dot product an uninformative matching function. To overcome this low-rank bottleneck in attention heads, we employ talking-heads technology based on bilinear pooled features and sparse selection of attention tensors. This allows interaction among filtered attention scores and enables each attention mechanism to depend on all queries and keys. Additionally, we apply graph-pooling between two intermediate blocks to reduce the number of tokens and aggregate semantic information more effectively. Our experimental results show that GvT produces comparable or superior outcomes to deep convolutional networks and surpasses vision transformers without pre-training on large datasets. The code for our proposed model is publicly available on the website.
>
---
#### [replaced 100] AvatarShield: Visual Reinforcement Learning for Human-Centric Video Forgery Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.15173v2](http://arxiv.org/pdf/2505.15173v2)**

> **作者:** Zhipei Xu; Xuanyu Zhang; Xing Zhou; Jian Zhang
>
> **摘要:** The rapid advancement of Artificial Intelligence Generated Content (AIGC) technologies, particularly in video generation, has led to unprecedented creative capabilities but also increased threats to information integrity, identity security, and public trust. Existing detection methods, while effective in general scenarios, lack robust solutions for human-centric videos, which pose greater risks due to their realism and potential for legal and ethical misuse. Moreover, current detection approaches often suffer from poor generalization, limited scalability, and reliance on labor-intensive supervised fine-tuning. To address these challenges, we propose AvatarShield, the first interpretable MLLM-based framework for detecting human-centric fake videos, enhanced via Group Relative Policy Optimization (GRPO). Through our carefully designed accuracy detection reward and temporal compensation reward, it effectively avoids the use of high-cost text annotation data, enabling precise temporal modeling and forgery detection. Meanwhile, we design a dual-encoder architecture, combining high-level semantic reasoning and low-level artifact amplification to guide MLLMs in effective forgery detection. We further collect FakeHumanVid, a large-scale human-centric video benchmark that includes synthesis methods guided by pose, audio, and text inputs, enabling rigorous evaluation of detection methods in real-world scenes. Extensive experiments show that AvatarShield significantly outperforms existing approaches in both in-domain and cross-domain detection, setting a new standard for human-centric video forensics.
>
---
#### [replaced 101] Shallow Diffuse: Robust and Invisible Watermarking through Low-Dimensional Subspaces in Diffusion Models
- **分类: cs.LG; cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.21088v2](http://arxiv.org/pdf/2410.21088v2)**

> **作者:** Wenda Li; Huijie Zhang; Qing Qu
>
> **摘要:** The widespread use of AI-generated content from diffusion models has raised significant concerns regarding misinformation and copyright infringement. Watermarking is a crucial technique for identifying these AI-generated images and preventing their misuse. In this paper, we introduce Shallow Diffuse, a new watermarking technique that embeds robust and invisible watermarks into diffusion model outputs. Unlike existing approaches that integrate watermarking throughout the entire diffusion sampling process, Shallow Diffuse decouples these steps by leveraging the presence of a low-dimensional subspace in the image generation process. This method ensures that a substantial portion of the watermark lies in the null space of this subspace, effectively separating it from the image generation process. Our theoretical and empirical analyses show that this decoupling strategy greatly enhances the consistency of data generation and the detectability of the watermark. Extensive experiments further validate that our Shallow Diffuse outperforms existing watermarking methods in terms of robustness and consistency. The codes will be released at https://github.com/liwd190019/Shallow-Diffuse.
>
---
#### [replaced 102] Latent Wavelet Diffusion: Enabling 4K Image Synthesis for Free
- **分类: cs.CV; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2506.00433v2](http://arxiv.org/pdf/2506.00433v2)**

> **作者:** Luigi Sigillo; Shengfeng He; Danilo Comminiello
>
> **摘要:** High-resolution image synthesis remains a core challenge in generative modeling, particularly in balancing computational efficiency with the preservation of fine-grained visual detail. We present Latent Wavelet Diffusion (LWD), a lightweight framework that enables any latent diffusion model to scale to ultra-high-resolution image generation (2K to 4K) for free. LWD introduces three key components: (1) a scale-consistent variational autoencoder objective that enhances the spectral fidelity of latent representations; (2) wavelet energy maps that identify and localize detail-rich spatial regions within the latent space; and (3) a time-dependent masking strategy that focuses denoising supervision on high-frequency components during training. LWD requires no architectural modifications and incurs no additional computational overhead. Despite its simplicity, it consistently improves perceptual quality and reduces FID in ultra-high-resolution image synthesis, outperforming strong baseline models. These results highlight the effectiveness of frequency-aware, signal-driven supervision as a principled and efficient approach for high-resolution generative modeling.
>
---
#### [replaced 103] Bayesian Prompt Flow Learning for Zero-Shot Anomaly Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10080v3](http://arxiv.org/pdf/2503.10080v3)**

> **作者:** Zhen Qu; Xian Tao; Xinyi Gong; Shichen Qu; Qiyu Chen; Zhengtao Zhang; Xingang Wang; Guiguang Ding
>
> **摘要:** Recently, vision-language models (e.g. CLIP) have demonstrated remarkable performance in zero-shot anomaly detection (ZSAD). By leveraging auxiliary data during training, these models can directly perform cross-category anomaly detection on target datasets, such as detecting defects on industrial product surfaces or identifying tumors in organ tissues. Existing approaches typically construct text prompts through either manual design or the optimization of learnable prompt vectors. However, these methods face several challenges: 1) handcrafted prompts require extensive expert knowledge and trial-and-error; 2) single-form learnable prompts struggle to capture complex anomaly semantics; and 3) an unconstrained prompt space limits generalization to unseen categories. To address these issues, we propose Bayesian Prompt Flow Learning (Bayes-PFL), which models the prompt space as a learnable probability distribution from a Bayesian perspective. Specifically, a prompt flow module is designed to learn both image-specific and image-agnostic distributions, which are jointly utilized to regularize the text prompt space and improve the model's generalization on unseen categories. These learned distributions are then sampled to generate diverse text prompts, effectively covering the prompt space. Additionally, a residual cross-model attention (RCA) module is introduced to better align dynamic text embeddings with fine-grained image features. Extensive experiments on 15 industrial and medical datasets demonstrate our method's superior performance. The code is available at https://github.com/xiaozhen228/Bayes-PFL.
>
---
#### [replaced 104] Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.19475v3](http://arxiv.org/pdf/2504.19475v3)**

> **作者:** Sonia Joseph; Praneet Suresh; Lorenz Hufe; Edward Stevinson; Robert Graham; Yash Vadi; Danilo Bzdok; Sebastian Lapuschkin; Lee Sharkey; Blake Aaron Richards
>
> **备注:** 4 pages, 3 figures, 9 tables. Oral and Tutorial at the CVPR Mechanistic Interpretability for Vision (MIV) Workshop
>
> **摘要:** Robust tooling and publicly available pre-trained models have helped drive recent advances in mechanistic interpretability for language models. However, similar progress in vision mechanistic interpretability has been hindered by the lack of accessible frameworks and pre-trained weights. We present Prisma (Access the codebase here: https://github.com/Prisma-Multimodal/ViT-Prisma), an open-source framework designed to accelerate vision mechanistic interpretability research, providing a unified toolkit for accessing 75+ vision and video transformers; support for sparse autoencoder (SAE), transcoder, and crosscoder training; a suite of 80+ pre-trained SAE weights; activation caching, circuit analysis tools, and visualization tools; and educational resources. Our analysis reveals surprising findings, including that effective vision SAEs can exhibit substantially lower sparsity patterns than language SAEs, and that in some instances, SAE reconstructions can decrease model loss. Prisma enables new research directions for understanding vision model internals while lowering barriers to entry in this emerging field.
>
---
#### [replaced 105] We Should Chart an Atlas of All the World's Models
- **分类: cs.LG; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10633v2](http://arxiv.org/pdf/2503.10633v2)**

> **作者:** Eliahu Horwitz; Nitzan Kurer; Jonathan Kahana; Liel Amar; Yedid Hoshen
>
> **备注:** Project page: https://horwitz.ai/model-atlas
>
> **摘要:** Public model repositories now contain millions of models, yet most models remain undocumented and effectively lost. In this position paper, we advocate for charting the world's model population in a unified structure we call the Model Atlas: a graph that captures models, their attributes, and the weight transformations that connect them. The Model Atlas enables applications in model forensics, meta-ML research, and model discovery, challenging tasks given today's unstructured model repositories. However, because most models lack documentation, large atlas regions remain uncharted. Addressing this gap motivates new machine learning methods that treat models themselves as data, inferring properties such as functionality, performance, and lineage directly from their weights. We argue that a scalable path forward is to bypass the unique parameter symmetries that plague model weights. Charting all the world's models will require a community effort, and we hope its broad utility will rally researchers toward this goal.
>
---
#### [replaced 106] Symmetrical Visual Contrastive Optimization: Aligning Vision-Language Models with Minimal Contrastive Images
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13928v2](http://arxiv.org/pdf/2502.13928v2)**

> **作者:** Shengguang Wu; Fan-Yun Sun; Kaiyue Wen; Nick Haber
>
> **备注:** Accepted to ACL 2025 Main. Project Website: https://s-vco.github.io/
>
> **摘要:** Recent studies have shown that Large Vision-Language Models (VLMs) tend to neglect image content and over-rely on language-model priors, resulting in errors in visually grounded tasks and hallucinations. We hypothesize that this issue arises because existing VLMs are not explicitly trained to generate texts that are accurately grounded in fine-grained image details. To enhance visual feedback during VLM training, we propose S-VCO (Symmetrical Visual Contrastive Optimization), a novel finetuning objective that steers the model toward capturing important visual details and aligning them with corresponding text tokens. To further facilitate this detailed alignment, we introduce MVC, a paired image-text dataset built by automatically filtering and augmenting visual counterfactual data to challenge the model with hard contrastive cases involving Minimal Visual Contrasts. Experiments show that our method consistently improves VLM performance across diverse benchmarks covering various abilities and domains, achieving up to a 22% reduction in hallucinations, and significant gains in vision-centric and general tasks. Notably, these improvements become increasingly pronounced in benchmarks with higher visual dependency. In short, S-VCO offers a significant enhancement of VLM's visually-dependent task performance while retaining or even improving the model's general abilities. We opensource our code at https://s-vco.github.io/
>
---
#### [replaced 107] VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.13106v4](http://arxiv.org/pdf/2501.13106v4)**

> **作者:** Boqiang Zhang; Kehan Li; Zesen Cheng; Zhiqiang Hu; Yuqian Yuan; Guanzheng Chen; Sicong Leng; Yuming Jiang; Hang Zhang; Xin Li; Peng Jin; Wenqi Zhang; Fan Wang; Lidong Bing; Deli Zhao
>
> **备注:** BZ, KL, ZC, ZH, YY, GC, SL, YJ, HZ, and XL contributed equally to this project. Code: https://github.com/DAMO-NLP-SG/VideoLLaMA3
>
> **摘要:** In this paper, we propose VideoLLaMA3, a more advanced multimodal foundation model for image and video understanding. The core design philosophy of VideoLLaMA3 is vision-centric. The meaning of "vision-centric" is two-fold: the vision-centric training paradigm and vision-centric framework design. The key insight of our vision-centric training paradigm is that high-quality image-text data is crucial for both image and video understanding. Instead of preparing massive video-text datasets, we focus on constructing large-scale and high-quality image-text datasets. VideoLLaMA3 has four training stages: 1) Vision Encoder Adaptation, which enables vision encoder to accept images of variable resolutions as input; 2) Vision-Language Alignment, which jointly tunes the vision encoder, projector, and LLM with large-scale image-text data covering multiple types (including scene images, documents, charts) as well as text-only data. 3) Multi-task Fine-tuning, which incorporates image-text SFT data for downstream tasks and video-text data to establish a foundation for video understanding. 4) Video-centric Fine-tuning, which further improves the model's capability in video understanding. As for the framework design, to better capture fine-grained details in images, the pretrained vision encoder is adapted to encode images of varying sizes into vision tokens with corresponding numbers, rather than a fixed number of tokens. For video inputs, we reduce the number of vision tokens according to their similarity so that the representation of videos will be more precise and compact. Benefit from vision-centric designs, VideoLLaMA3 achieves compelling performances in both image and video understanding benchmarks.
>
---
